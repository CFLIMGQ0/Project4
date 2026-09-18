"""本次外部实验专用：任意标签数、部分标签监督；不改动既有论文实验。"""
from __future__ import annotations

import torch
import torch.nn.functional as F

from exp_8.models import Exp12AProCoPEWatchCrossAttentionTextCNNModel, _safe_key_padding_mask
from training.losses import AsymmetricLossMultiLabel


def masked_asl(logits, targets, known):
    # 直接选择已知元素调用原ASL，保证全已知时与论文实现完全一致。
    return AsymmetricLossMultiLabel()(logits[known], targets[known]) if known.any() else logits.sum() * 0


def masked_bce(logits, targets, known, pos_weight=None):
    values = F.binary_cross_entropy_with_logits(logits.float(), targets.float(),
                                                pos_weight=pos_weight, reduction="none")
    return (values * known).sum() / known.sum().clamp_min(1)


class PartialLabelAMEF(Exp12AProCoPEWatchCrossAttentionTextCNNModel):
    """保持融合结构；查询置换由三标签的两个循环位移推广为C-1个位移。"""

    def _build_watch_cross_attention_outputs(self, *, images, label_embeds, attention, features,
                                           extra_outputs, labels, watch_token_ids,
                                           watch_token_mask, use_gate):
        image_logits = self.classify(label_embeds)
        tokens, text_mask, _, active_bool = self.text_encoder(
            watch_token_ids, watch_token_mask, batch_size=images.shape[0], device=images.device)
        safe, padding = _safe_key_padding_mask(tokens, text_mask)
        retrieved, weights = self.text_cross_attn(label_embeds + self.label_query_bias, safe, safe,
                                                 key_padding_mask=padding, need_weights=True,
                                                 average_attn_weights=True)
        active = active_bool[:, None, None].to(retrieved.dtype)
        retrieved = retrieved * active

        def fuse(text):
            gate = torch.sigmoid(self.text_gate(torch.cat([label_embeds, text], -1))) if use_gate else torch.ones_like(text[..., :1])
            gate = gate * active
            return self.classify(label_embeds + gate * text), gate

        logits, gates = fuse(retrieved)
        auxiliary = {"image_aux": logits.sum() * 0, "label_query_consistency": logits.sum() * 0}
        if labels is not None:
            targets, known = labels
            auxiliary["image_aux"] = masked_bce(image_logits, targets, known)
            if self.training:
                terms = []
                for shift in range(1, self.num_labels):
                    permutation = (torch.arange(self.num_labels, device=images.device) + shift) % self.num_labels
                    valid = known & known[:, permutation] & (targets > .5) & (targets[:, permutation] < .5) & active_bool[:, None]
                    if valid.any():
                        swapped, _ = fuse(retrieved[:, permutation])
                        terms.append(.5 * (F.softplus(-logits[valid]) + F.softplus(swapped[valid])))
                if terms:
                    auxiliary["label_query_consistency"] = torch.cat(terms).mean()
        extra_outputs.update(image_only_logits=image_logits, watch_cross_attention=weights,
                             watch_text_gate=gates.squeeze(-1), aux_losses=auxiliary)
        return self.build_outputs(logits=logits, attention=attention, features=features, extra_outputs=extra_outputs)


def forward_image(model, key, images, image_mask, targets, known, training):
    # DTFD的伪包logits通过本次前向的只读钩子取出，不改动主模型结构。
    group_logits = []
    hook = None
    if key == "dtfd_mil" and training:
        hook = model.group_classifier.register_forward_hook(lambda _m, _i, out: group_logits.append(out))
    try:
        output = model(images, image_mask)
    finally:
        if hook:
            hook.remove()
    if not training:
        return output
    aux = output.setdefault("aux_losses", {})
    if key == "dtfd_mil":
        from sotas.task1.gastro_sota.common import split_group_ranges
        stacked = torch.stack(group_logits, 1)
        valid = torch.stack([image_mask[:, a:b].any(1) for a, b in
                             split_group_ranges(images.shape[1], model.num_groups)], 1)
        observed = valid[..., None] & known[:, None, :]
        # 保持原DTFD的按有效伪包求和标度，全已知时与原损失相同。
        aux["pseudo_bag"] = masked_bce(stacked, targets[:, None, :].expand_as(stacked), observed) * targets.shape[1]
    if key in {"clam_mb", "clam_sb"}:
        logits = model.instance_classifier(output["instance_features"])
        attn = output["attention"]
        terms = []
        for i in range(len(images)):
            valid_indices = torch.where(image_mask[i])[0]
            k = min(model.instance_topk, len(valid_indices))
            scores = attn[i, :, valid_indices]
            top = valid_indices[scores.topk(k, dim=-1).indices]
            bottom = valid_indices[(-scores).topk(k, dim=-1).indices]
            il = logits[i].T
            top_logits, bottom_logits = il.gather(1, top), il.gather(1, bottom)
            obs = known[i, :, None].expand_as(top_logits)
            top_targets = targets[i, :, None].expand_as(top_logits)
            terms.append((F.binary_cross_entropy_with_logits(top_logits.float(), top_targets.float(), reduction="none")
                          + F.binary_cross_entropy_with_logits(bottom_logits.float(), torch.zeros_like(bottom_logits).float(), reduction="none"))[obs])
        selected = torch.cat(terms)
        aux["instance_clustering"] = .5 * selected.mean() if selected.numel() else logits.sum() * 0
    return output


def forward_fusion(model, key, images, image_mask, token_ids, token_mask, positions, counts, targets, known, training):
    kwargs = dict(images=images, mask=image_mask, watch_token_ids=token_ids, watch_token_mask=token_mask)
    if key == "amef_multimodal":
        return model(**kwargs, instance_indices=positions, original_image_counts=counts,
                     labels=(targets, known) if training else None)
    output = model(**kwargs)
    if training and key == "task2_mmfnet_2024":
        output["aux_losses"]["image_branch"] = masked_bce(output["image_only_logits"], targets, known)
        output["aux_losses"]["text_branch"] = masked_bce(output["text_only_logits"], targets, known)
    return output
