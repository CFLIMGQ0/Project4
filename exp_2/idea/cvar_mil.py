"""
CVaR MIL

问题动机：
- `reflux`、`ulcer`、`polyp` 这类标签往往只在极少数帧上出现
- 均值或普通注意力会把这些稀疏证据稀释掉

核心想法：
- 把每个标签的 bag 风险看成“上分位证据”的条件期望
- 只对高分实例分位区间做加权聚合，近似 CVaR 风险
- 再与标准注意力结果做可学习融合，避免完全丢掉全局上下文
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn

from exp_2.common import Exp2AttentionMILBase


class LabelwiseCVaRPool(nn.Module):
    def __init__(self, feature_dim: int, num_labels: int, tail_fraction: float = 0.25) -> None:
        super().__init__()
        self.num_labels = num_labels
        self.tail_fraction = float(tail_fraction)
        self.scorer = nn.Linear(feature_dim, num_labels)

    def forward(self, features: torch.Tensor, mask: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        scores = self.scorer(features).transpose(1, 2)
        batch_size, _, num_instances = scores.shape
        attention = torch.zeros_like(scores)
        pooled = torch.zeros(
            batch_size,
            self.num_labels,
            features.shape[-1],
            device=features.device,
            dtype=features.dtype,
        )
        mask_bool = mask.to(dtype=torch.bool)

        for batch_index in range(batch_size):
            valid_num = int(mask_bool[batch_index].sum().item())
            if valid_num <= 0:
                continue
            use_k = max(1, int(math.ceil(valid_num * self.tail_fraction)))
            current_scores = scores[batch_index, :, :valid_num]
            topk_values, topk_indices = torch.topk(current_scores, k=use_k, dim=-1)
            topk_weights = torch.softmax(topk_values, dim=-1).to(dtype=attention.dtype)
            attention[batch_index].scatter_(1, topk_indices, topk_weights)
            pooled[batch_index] = torch.einsum(
                "ln,nd->ld",
                attention[batch_index, :, :valid_num],
                features[batch_index, :valid_num],
            )
        return pooled, attention


class CVARMILModel(Exp2AttentionMILBase):
    def __init__(
        self,
        backbone_name: str = "resnet50",
        pretrained: bool = True,
        freeze_stages: int = 0,
        feature_dim: int = 512,
        attn_dim: int = 256,
        num_labels: int = 8,
        dropout: float = 0.2,
        tail_fraction: float = 0.25,
    ) -> None:
        super().__init__(
            backbone_name=backbone_name,
            pretrained=pretrained,
            freeze_stages=freeze_stages,
            feature_dim=feature_dim,
            attn_dim=attn_dim,
            num_labels=num_labels,
            dropout=dropout,
        )
        self.cvar_pool = LabelwiseCVaRPool(feature_dim=feature_dim, num_labels=num_labels, tail_fraction=tail_fraction)
        init_mix = torch.tensor([0.8, 0.8, 0.8, 0.45, 0.45, 0.35, 0.3, 0.3], dtype=torch.float32)
        if init_mix.numel() < num_labels:
            init_mix = torch.cat([init_mix, init_mix.new_full((num_labels - init_mix.numel(),), 0.35)], dim=0)
        self.mix_logits = nn.Parameter(torch.logit(init_mix[:num_labels].clamp(1e-4, 1.0 - 1e-4)))

    def forward(self, images: torch.Tensor, mask: torch.Tensor) -> dict[str, torch.Tensor]:
        features = self.encode_instances(images)
        base_bag, base_attention = self.mil_pool(features, mask)
        cvar_bag, cvar_attention = self.cvar_pool(features, mask)

        mix = torch.sigmoid(self.mix_logits).view(1, self.num_labels, 1)
        bag_embeds = mix * base_bag + (1.0 - mix) * cvar_bag
        logits = self.classify_embeddings(bag_embeds)
        attention = mix * base_attention + (1.0 - mix) * cvar_attention

        return {
            "logits": logits,
            "attention": attention,
            "instance_features": features,
            "base_attention": base_attention,
            "cvar_attention": cvar_attention,
        }


__all__ = ["CVARMILModel"]
