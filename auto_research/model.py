"""可组合、可追溯的系统级研究改动。默认配置保持现有前向与 LQD 定义。"""
from types import SimpleNamespace
import torch
from torch import nn
import torch.nn.functional as F
from exp_8.models import (AProCoPE, Exp12AProCoPEWatchCrossAttentionTextCNNModel,
                         _safe_key_padding_mask, _safe_transformer_encoder_with_mask)
from model.common.pooling import masked_softmax


class ResearchPosition(AProCoPE):
    def _contextual_coordinates(self, features, mask, raw):
        cfg = self.research_config
        mode = cfg.get("descriptor", "original")
        if mode == "original":
            return super()._contextual_coordinates(features, mask, raw)
        u = self.transition_projector(features) * mask[..., None]
        if mode == "smooth3":
            numerator = F.avg_pool1d(u.transpose(1, 2), 3, 1, 1).transpose(1, 2)
            denominator = F.avg_pool1d(mask[:, None].float(), 3, 1, 1).transpose(1, 2)
            u = numerator / denominator.clamp_min(1e-6)
        pair = mask[:, 1:] & mask[:, :-1]
        delta = (u[:, 1:] - u[:, :-1]) * pair[..., None]
        back, ahead = torch.zeros_like(u), torch.zeros_like(u)
        back[:, 1:], ahead[:, :-1] = delta, delta
        if mode == "unit_direction":
            back = F.normalize(back, dim=-1)
            ahead = F.normalize(ahead, dim=-1)
        gap = torch.zeros_like(raw)
        gap[:, 1:] = (raw[:, 1:] - raw[:, :-1]).clamp_min(1e-6) * pair
        parts = [u, back, ahead, back.abs(), ahead.abs(), back * ahead]
        removals = {"current_gap": [1, 2, 3, 4, 5], "no_current": [0],
                    "no_interaction": [5], "magnitude_only": [1, 2, 5],
                    "direction_only": [3, 4, 5]}
        for j in removals.get(mode, []):
            parts[j] = torch.zeros_like(parts[j])
        descriptor = torch.cat(parts + [gap[..., None]], -1)
        eta = self.warp_alpha * torch.tanh(self.transition_mlp(descriptor).squeeze(-1))
        eta = eta * torch.cat([torch.zeros_like(mask[:, :1]), pair], 1)
        weight = gap * eta.float().exp().to(gap.dtype)
        last = raw.gather(1, (mask.sum(1)-1).clamp_min(0)[:, None])
        span = (last - raw[:, :1]).clamp_min(0)
        adjusted = weight * span / weight.sum(1, keepdim=True).clamp_min(1e-8)
        return (raw[:, :1] + adjusted.cumsum(1)) * mask, eta


class ResearchModel(Exp12AProCoPEWatchCrossAttentionTextCNNModel):
    def __init__(self, num_labels, config):
        super().__init__(backbone_name="convnext_tiny", pretrained=False, freeze_stages=1,
                         feature_dim=512, attn_dim=256, hidden_dim=1024, num_labels=num_labels,
                         dropout=.2, encoder_chunk_size=4096, num_heads=4, num_layers=2,
                         use_label_graph=True, label_graph_type="label_hypergraph",
                         label_hypergraph_edges=2, text_vocab_size=8192, text_embed_dim=128,
                         textcnn_kernel_sizes=(2, 3, 4), position_variant="apro_full",
                         apro_position_dim=64, apro_warp_alpha=1.5, apro_fourier_frequencies=8)
        self.instance_encoder.backbone = nn.Sequential(nn.Flatten(1), nn.Linear(768, 512),
                                                       nn.ReLU(inplace=True), nn.Dropout(.2))
        self.config = config
        self.epoch = 0
        self._safe_attention_mask_path = True
        # 所有变体先构建相同基础网络；非相关层具有相同初始权重。
        p = self.apro_positioner
        p.__class__ = ResearchPosition
        p.research_config = config
        p.warp_alpha = float(config.get("warp_alpha", 1.5))
        p.route = config.get("route", "both")
        if config.get("zero_warp_init"):
            nn.init.zeros_(p.transition_mlp[-1].weight)
            nn.init.zeros_(p.transition_mlp[-1].bias)
        if config.get("position") == "anchored":
            p.transition_mode = "none"
        elif config.get("position") in {"original_pe", "no_pe"}:
            self.position_variant = config["position"]
        if config.get("label_position"):
            self.label_position = nn.Parameter(torch.zeros(num_labels, 3))

    def _encode_position(self, features, mask, indices, counts):
        if self.position_variant != "apro_full":
            return super()._encode_position(features, mask, indices, counts)
        source = features.detach() if self.config.get("detach_position_input") else features
        absolute, bias, outputs = self.apro_positioner(source, mask, indices, counts)
        absolute_scale = float(self.config.get("absolute_scale", 1))
        relative_scale = float(self.config.get("relative_scale", 1))
        if self.config.get("position_warmup", 0):
            scale = min(1., (self.epoch + 1) / self.config["position_warmup"])
            absolute_scale *= scale
            relative_scale *= scale
        if absolute is not None:
            absolute = absolute * absolute_scale
            outputs["research_absolute"] = absolute
        if bias is not None:
            bias = bias * relative_scale
        early = self.config.get("placement", "all") not in {"late_absolute", "late_both"}
        positioned = features + absolute if absolute is not None and early else features
        return positioned * mask[..., None], bias, outputs

    def encode_long_mil(self, images, mask, instance_indices=None, original_image_counts=None):
        placement = self.config.get("placement", "all")
        if placement == "all" and not self.config.get("label_position") and not self.config.get("bypass_graph"):
            return super().encode_long_mil(images, mask, instance_indices, original_image_counts)
        h, outputs = self.encode_instances(images, mask)
        h, bias, pos = self._encode_position(h, mask, instance_indices, original_image_counts)
        for index, layer in enumerate(self.context_encoder.layers):
            last = index == len(self.context_encoder.layers) - 1
            if last and placement in {"late_absolute", "late_both"}:
                h = h + pos.get("research_absolute", 0)
            use_relative = not ((placement == "relative_first" and index > 0)
                                or (placement in {"relative_last", "late_both"} and not last))
            block = torch.zeros(len(h), 4, h.shape[1], h.shape[1], device=h.device, dtype=h.dtype)
            if bias is not None and use_relative:
                block = block + bias
            block = block.masked_fill(~mask[:, None, None, :], -1e4).reshape(-1, h.shape[1], h.shape[1])
            h = _safe_transformer_encoder_with_mask(SimpleNamespace(layers=[layer], norm=None), h, block)
        if self.context_encoder.norm is not None:
            h = self.context_encoder.norm(h)
        h = h * mask[..., None]
        if self.config.get("label_position"):
            c, r = pos["apro_context_coordinates"], pos["apro_raw_coordinates"]
            descriptor = torch.stack([c-.5, c-r, (c-.5).square()], -1)
            gated = self.mil_pool.attn.v(h) * self.mil_pool.attn.u(h)
            scores = self.mil_pool.attn.w(gated).transpose(1, 2)
            scores = scores + torch.einsum("btd,ld->blt", descriptor, self.label_position)
            attn = masked_softmax(scores, mask=mask[:, None], dim=-1)
            z = torch.einsum("blt,btd->bld", attn, h)
        else:
            z, attn = self.mil_pool(h, mask)
        z, graph = (z, {}) if self.config.get("bypass_graph") else self.refine_labels(z)
        outputs.update(pos)
        outputs.update(graph)
        return h, z, attn, outputs

    def _build_watch_cross_attention_outputs(self, *, images, label_embeds, attention, features,
                                           extra_outputs, labels, watch_token_ids,
                                           watch_token_mask, use_gate):
        z = label_embeds
        image_logits = self.classify(z)
        tokens, token_mask, _, active_bool = self.text_encoder(
            watch_token_ids, watch_token_mask, batch_size=len(images), device=images.device)
        if self.training and self.config.get("report_dropout", 0):
            active_bool = active_bool & (torch.rand(len(z), device=z.device) >= self.config["report_dropout"])
        safe, padding = _safe_key_padding_mask(tokens, token_mask)
        detach = self.config.get("fusion_detach_visual", False)
        query = z.detach() if detach or self.config.get("detach_query") else z
        retrieved, weights = self.text_cross_attn(query + self.label_query_bias, safe, safe,
                                                 key_padding_mask=padding, need_weights=True,
                                                 average_attn_weights=True)
        active = active_bool[:, None, None].to(retrieved.dtype)
        retrieved = retrieved * active
        visual = z.detach() if detach else z

        def fuse(text, visual_input=visual):
            gates = torch.sigmoid(self.text_gate(torch.cat([visual_input, text], -1))) * active
            return self.classify(visual_input + gates * text), gates

        logits, gates = fuse(retrieved)
        lqd = logits.sum() * 0
        mode = self.config.get("lqd_mode", "original")
        if self.training and labels is not None and mode != "off":
            target, known = labels
            terms = []
            for shift in range(1, self.num_labels):
                perm = (torch.arange(self.num_labels, device=z.device) + shift) % self.num_labels
                valid = known & known[:, perm] & (target > .5) & (target[:, perm] < .5) & active_bool[:, None]
                if valid.any():
                    swapped_visual = visual.detach() if mode == "detach_swapped_visual" else visual
                    swapped, _ = fuse(retrieved[:, perm], swapped_visual)
                    if mode == "ranking":
                        terms.append(F.softplus(float(self.config.get("lqd_margin", .2)) - logits[valid] + swapped[valid]))
                    else:
                        terms.append(.5 * (F.softplus(-logits[valid]) + F.softplus(swapped[valid])))
            if terms:
                lqd = torch.cat(terms).mean()
        extra_outputs.update(image_only_logits=image_logits, watch_cross_attention=weights,
                             watch_text_gate=gates.squeeze(-1), research_lqd=lqd,
                             research_visual=z, research_text=retrieved)
        return self.build_outputs(logits=logits, attention=attention, features=features, extra_outputs=extra_outputs)
