from __future__ import annotations

import torch
import torch.nn as nn


def masked_softmax(logits: torch.Tensor, mask: torch.Tensor, dim: int = -1) -> torch.Tensor:
    """对变长序列执行 softmax，仅在有效实例上归一化。"""
    mask = mask.to(dtype=torch.bool)
    while mask.dim() < logits.dim():
        mask = mask.unsqueeze(1)
    masked_logits = logits.masked_fill(~mask, torch.finfo(logits.dtype).min)
    probs = torch.softmax(masked_logits, dim=dim)
    probs = probs * mask.to(probs.dtype)
    denom = probs.sum(dim=dim, keepdim=True).clamp_min(1e-12)
    return probs / denom


class GatedAttention(nn.Module):
    def __init__(self, in_dim: int, attn_dim: int, num_heads: int = 1, dropout: float = 0.1) -> None:
        super().__init__()
        self.v = nn.Sequential(nn.Linear(in_dim, attn_dim), nn.Tanh(), nn.Dropout(dropout))
        self.u = nn.Sequential(nn.Linear(in_dim, attn_dim), nn.Sigmoid(), nn.Dropout(dropout))
        self.w = nn.Linear(attn_dim, num_heads)

    def forward(self, x: torch.Tensor, mask: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        gate = self.v(x) * self.u(x)
        attn_logits = self.w(gate).transpose(1, 2)
        attn = masked_softmax(attn_logits, mask=mask.unsqueeze(1), dim=-1)
        bag_embed = torch.einsum("bhn,bnd->bhd", attn, x)
        return bag_embed, attn


class SingleAttentionMIL(nn.Module):
    def __init__(self, in_dim: int, attn_dim: int, dropout: float = 0.1) -> None:
        super().__init__()
        self.attn = GatedAttention(in_dim=in_dim, attn_dim=attn_dim, num_heads=1, dropout=dropout)

    def forward(self, x: torch.Tensor, mask: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        bag_embed, attn = self.attn(x, mask)
        return bag_embed[:, 0, :], attn[:, 0, :]


class MultiLabelAttentionMIL(nn.Module):
    def __init__(self, in_dim: int, attn_dim: int, num_labels: int, dropout: float = 0.1) -> None:
        super().__init__()
        self.attn = GatedAttention(in_dim=in_dim, attn_dim=attn_dim, num_heads=num_labels, dropout=dropout)

    def forward(self, x: torch.Tensor, mask: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        return self.attn(x, mask)
