from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from .demo_backbones import build_backbone
from .demo_losses import (
    consistency_loss,
    hard_negative_suppression,
    prototype_binary_contrastive_loss,
)


def _masked_max(x: torch.Tensor, mask: torch.Tensor, dim: int = -1) -> torch.Tensor:
    x = x.masked_fill(~mask, torch.finfo(x.dtype).min)
    return x.max(dim=dim).values


class DemoColoCountAwareDebiasMIL(nn.Module):
    """肠镜 advanced：双分支 + top-k candidate + prototype memory + 去偏约束。"""

    def __init__(
        self,
        backbone_name: str = "resnet50",
        pretrained: bool = True,
        freeze_stages: int = 0,
        feature_dim: int = 512,
        topk_lesion: int = 8,
        topk_context: int = 8,
        prototype_k: int = 8,
        binary_num_classes: int = 2,
        dropout: float = 0.2,
    ) -> None:
        super().__init__()
        self.topk_lesion = topk_lesion
        self.topk_context = topk_context
        self.binary_num_classes = binary_num_classes

        self.encoder, out_dim = build_backbone(
            backbone_name=backbone_name,
            pretrained=pretrained,
            out_dim=feature_dim,
            freeze_stages=freeze_stages,
            projector_dropout=dropout,
        )

        self.lesion_branch = nn.Sequential(
            nn.Linear(out_dim, feature_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(feature_dim, feature_dim),
            nn.GELU(),
        )
        self.context_branch = nn.Sequential(
            nn.Linear(out_dim, feature_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(feature_dim, feature_dim),
            nn.GELU(),
        )
        self.fusion_gate = nn.Sequential(nn.Linear(feature_dim * 2, feature_dim), nn.Sigmoid())

        self.instance_lesion_scorer = nn.Linear(feature_dim, 1)

        bag_dim = feature_dim * 2
        if binary_num_classes == 2:
            self.binary_head = nn.Linear(bag_dim, 1)
            self.lesion_head = nn.Linear(feature_dim, 1)
            self.context_head = nn.Linear(feature_dim, 1)
        else:
            self.binary_head = nn.Linear(bag_dim, binary_num_classes)
            self.lesion_head = nn.Linear(feature_dim, binary_num_classes)
            self.context_head = nn.Linear(feature_dim, binary_num_classes)

        # count head: single vs multi，仅对阳性 bag 有效
        self.count_head = nn.Linear(bag_dim, 2)

        self.normal_prototypes = nn.Parameter(torch.randn(prototype_k, feature_dim) * 0.02)
        self.polyp_prototypes = nn.Parameter(torch.randn(prototype_k, feature_dim) * 0.02)

        self.proto_scale = nn.Parameter(torch.tensor(1.0))
        self.proto_bias = nn.Parameter(torch.tensor(0.0))

    def _encode(self, images: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        b, n, c, h, w = images.shape
        x = images.reshape(b * n, c, h, w)
        base = self.encoder(x).reshape(b, n, -1)

        z_lesion = self.lesion_branch(base)
        z_context = self.context_branch(base)
        gate = self.fusion_gate(torch.cat([z_lesion, z_context], dim=-1))
        fused = gate * z_lesion + (1.0 - gate) * z_context
        return z_lesion, z_context, fused, base

    def _select_tokens(
        self,
        fused: torch.Tensor,
        lesion_logits: torch.Tensor,
        mask: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, list[list[int]], list[list[int]]]:
        b, _, d = fused.shape
        lesion_pools = []
        context_pools = []
        lesion_only = []
        context_only = []
        lesion_indices: list[list[int]] = []
        context_indices: list[list[int]] = []

        for i in range(b):
            valid_n = int(mask[i].sum().item())
            if valid_n <= 0:
                lesion_pools.append(torch.zeros(d, device=fused.device, dtype=fused.dtype))
                context_pools.append(torch.zeros(d, device=fused.device, dtype=fused.dtype))
                lesion_only.append(torch.zeros(d, device=fused.device, dtype=fused.dtype))
                context_only.append(torch.zeros(d, device=fused.device, dtype=fused.dtype))
                lesion_indices.append([])
                context_indices.append([])
                continue

            valid_logits = lesion_logits[i, :valid_n]
            k_l = min(self.topk_lesion, valid_n)
            k_c = min(self.topk_context, valid_n)

            top_idx = torch.topk(valid_logits, k=k_l, dim=0).indices
            low_idx = torch.topk(-valid_logits, k=k_c, dim=0).indices

            lesion_tok = fused[i, top_idx, :]
            context_tok = fused[i, low_idx, :]

            lesion_w = torch.softmax(valid_logits[top_idx], dim=0).unsqueeze(-1)
            lesion_pool = (lesion_tok * lesion_w).sum(dim=0)
            context_pool = context_tok.mean(dim=0)

            lesion_pools.append(lesion_pool)
            context_pools.append(context_pool)
            lesion_only.append(lesion_tok.mean(dim=0))
            context_only.append(context_pool)
            lesion_indices.append(top_idx.detach().cpu().tolist())
            context_indices.append(low_idx.detach().cpu().tolist())

        return (
            torch.stack(lesion_pools, dim=0),
            torch.stack(context_pools, dim=0),
            torch.stack(lesion_only, dim=0),
            torch.stack(context_only, dim=0),
            lesion_indices,
            context_indices,
        )

    def _prototype_similarity(self, fused: torch.Tensor, mask: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        tok = F.normalize(fused, dim=-1)
        normal_p = F.normalize(self.normal_prototypes, dim=-1)
        polyp_p = F.normalize(self.polyp_prototypes, dim=-1)

        sim_normal = torch.einsum("bnd,kd->bnk", tok, normal_p).max(dim=-1).values
        sim_polyp = torch.einsum("bnd,kd->bnk", tok, polyp_p).max(dim=-1).values

        normal_bag = _masked_max(sim_normal, mask)
        polyp_bag = _masked_max(sim_polyp, mask)
        return normal_bag, polyp_bag

    def forward(
        self,
        images: torch.Tensor,
        mask: torch.Tensor,
        labels: torch.Tensor | None = None,
        count_labels: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor | dict[str, torch.Tensor]]:
        z_lesion, z_context, fused, _ = self._encode(images)
        lesion_logits_inst = self.instance_lesion_scorer(z_lesion).squeeze(-1)
        lesion_logits_inst = lesion_logits_inst.masked_fill(~mask, torch.finfo(lesion_logits_inst.dtype).min)

        (
            lesion_pool,
            context_pool,
            lesion_only_pool,
            context_only_pool,
            lesion_idx,
            context_idx,
        ) = self._select_tokens(fused=fused, lesion_logits=lesion_logits_inst, mask=mask)

        bag_repr = torch.cat([lesion_pool, context_pool], dim=-1)
        binary_logits = self.binary_head(bag_repr)
        if self.binary_num_classes == 2:
            binary_logits = binary_logits.squeeze(-1)

        lesion_only_logits = self.lesion_head(lesion_only_pool)
        context_only_logits = self.context_head(context_only_pool)
        if self.binary_num_classes == 2:
            lesion_only_logits = lesion_only_logits.squeeze(-1)
            context_only_logits = context_only_logits.squeeze(-1)

        count_logits = self.count_head(bag_repr)

        normal_sim, polyp_sim = self._prototype_similarity(fused=fused, mask=mask)
        proto_delta = polyp_sim - normal_sim
        if self.binary_num_classes == 2:
            binary_logits = binary_logits + self.proto_scale * proto_delta + self.proto_bias

        max_l = max((len(x) for x in lesion_idx), default=0)
        max_c = max((len(x) for x in context_idx), default=0)
        lesion_idx_tensor = torch.full((len(lesion_idx), max_l), -1, dtype=torch.long, device=images.device)
        context_idx_tensor = torch.full((len(context_idx), max_c), -1, dtype=torch.long, device=images.device)
        for i, idx_list in enumerate(lesion_idx):
            if idx_list:
                lesion_idx_tensor[i, : len(idx_list)] = torch.tensor(idx_list, dtype=torch.long, device=images.device)
        for i, idx_list in enumerate(context_idx):
            if idx_list:
                context_idx_tensor[i, : len(idx_list)] = torch.tensor(idx_list, dtype=torch.long, device=images.device)

        aux_losses: dict[str, torch.Tensor] = {}
        if labels is not None and self.binary_num_classes == 2:
            labels_float = labels.float()
            aux_losses["proto"] = prototype_binary_contrastive_loss(normal_sim, polyp_sim, labels).reshape(1)
            aux_losses["hard_negative"] = hard_negative_suppression(
                instance_scores=torch.sigmoid(lesion_logits_inst.clamp(min=-20, max=20)),
                binary_labels=labels,
                mask=mask,
            ).reshape(1)
            aux_losses["consistency"] = consistency_loss(
                torch.sigmoid(lesion_only_logits),
                torch.sigmoid(context_only_logits),
            ).reshape(1)

            if count_labels is not None:
                valid_count = (labels_float > 0.5) & (count_labels >= 0)
                if valid_count.any():
                    aux_losses["count"] = F.cross_entropy(count_logits[valid_count], count_labels[valid_count]).reshape(1)
                else:
                    aux_losses["count"] = torch.zeros((1,), device=images.device)

        return {
            "logits": binary_logits,
            "count_logits": count_logits,
            "attention": torch.softmax(
                lesion_logits_inst.masked_fill(~mask, torch.finfo(lesion_logits_inst.dtype).min),
                dim=-1,
            ),
            "instance_scores": lesion_logits_inst,
            "prototype_similarity": torch.stack([normal_sim, polyp_sim], dim=-1),
            "lesion_candidate_indices": lesion_idx_tensor,
            "context_candidate_indices": context_idx_tensor,
            "aux_losses": aux_losses,
        }
