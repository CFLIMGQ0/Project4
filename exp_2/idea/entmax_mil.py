"""
Entmax/Sparsemax MIL

问题动机：
- `TASK2` 的弱标签通常只出现在少数实例里，softmax 注意力容易把概率摊薄
- 后期训练时，softmax 还会把很多“次优实例”一起拉高，带来假阳性

核心想法：
- 用 Tsallis 熵家族里的稀疏归一化替代 softmax
- 当前实现使用更稳定的 sparsemax 变体，让注意力天然稀疏
- 目标是让模型更像“只盯住少数真正可疑帧”
"""

from __future__ import annotations

import torch
import torch.nn as nn

from exp_2.common import Exp2AttentionMILBase


def sparsemax(logits: torch.Tensor, dim: int = -1) -> torch.Tensor:
    shifted = logits - logits.max(dim=dim, keepdim=True).values
    z_sorted, _ = torch.sort(shifted, dim=dim, descending=True)
    z_cumsum = z_sorted.cumsum(dim=dim) - 1.0
    k = torch.arange(1, z_sorted.size(dim) + 1, device=logits.device, dtype=logits.dtype)
    view_shape = [1] * logits.dim()
    view_shape[dim] = -1
    k = k.view(view_shape)
    support = z_sorted > (z_cumsum / k)
    k_z = support.sum(dim=dim, keepdim=True).clamp_min(1)
    tau = z_cumsum.gather(dim, k_z.long() - 1) / k_z.to(logits.dtype)
    return torch.clamp(shifted - tau, min=0.0)


def masked_sparsemax(logits: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    expanded_mask = mask.to(dtype=torch.bool)
    while expanded_mask.dim() < logits.dim():
        expanded_mask = expanded_mask.unsqueeze(1)
    masked_logits = logits.masked_fill(~expanded_mask, torch.finfo(logits.dtype).min)
    probs = sparsemax(masked_logits, dim=-1)
    probs = probs * expanded_mask.to(dtype=probs.dtype)
    denom = probs.sum(dim=-1, keepdim=True).clamp_min(1e-12)
    return probs / denom


class SparseMultiLabelAttentionMIL(nn.Module):
    def __init__(self, in_dim: int, attn_dim: int, num_labels: int, dropout: float = 0.1) -> None:
        super().__init__()
        self.v = nn.Sequential(nn.Linear(in_dim, attn_dim), nn.Tanh(), nn.Dropout(dropout))
        self.u = nn.Sequential(nn.Linear(in_dim, attn_dim), nn.Sigmoid(), nn.Dropout(dropout))
        self.w = nn.Linear(attn_dim, num_labels)

    def forward(self, x: torch.Tensor, mask: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        gate = self.v(x) * self.u(x)
        attn_logits = self.w(gate).transpose(1, 2)
        attention = masked_sparsemax(attn_logits, mask)
        bag_embed = torch.einsum("bln,bnd->bld", attention, x)
        return bag_embed, attention


class EntmaxMILModel(Exp2AttentionMILBase):
    def __init__(
        self,
        backbone_name: str = "resnet50",
        pretrained: bool = True,
        freeze_stages: int = 0,
        feature_dim: int = 512,
        attn_dim: int = 256,
        num_labels: int = 8,
        dropout: float = 0.2,
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
        self.mil_pool = SparseMultiLabelAttentionMIL(
            in_dim=feature_dim,
            attn_dim=attn_dim,
            num_labels=num_labels,
            dropout=dropout,
        )


__all__ = ["EntmaxMILModel"]
