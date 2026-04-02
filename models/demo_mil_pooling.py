from __future__ import annotations

import torch
import torch.nn as nn


def masked_softmax(logits: torch.Tensor, mask: torch.Tensor, dim: int = -1) -> torch.Tensor:
    """对变长序列进行 softmax，mask 为 False 的位置不参与归一化。"""
    mask = mask.to(dtype=torch.bool)
    while mask.dim() < logits.dim():
        mask = mask.unsqueeze(1)
    masked_logits = logits.masked_fill(~mask, torch.finfo(logits.dtype).min)
    probs = torch.softmax(masked_logits, dim=dim)
    probs = probs * mask.to(probs.dtype)
    denom = probs.sum(dim=dim, keepdim=True).clamp_min(1e-12)
    return probs / denom


class DemoGatedAttention(nn.Module):
    def __init__(self, in_dim: int, attn_dim: int, num_heads: int = 1, dropout: float = 0.1) -> None:
        super().__init__()
        self.v = nn.Sequential(nn.Linear(in_dim, attn_dim), nn.Tanh(), nn.Dropout(dropout))
        self.u = nn.Sequential(nn.Linear(in_dim, attn_dim), nn.Sigmoid(), nn.Dropout(dropout))
        self.w = nn.Linear(attn_dim, num_heads)
        self.num_heads = num_heads

    def forward(self, x: torch.Tensor, mask: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """
        x: [B, N, D]
        mask: [B, N]
        return:
          bag_embed: [B, H, D]
          attn: [B, H, N]
        """
        gate = self.v(x) * self.u(x)
        attn_logits = self.w(gate).transpose(1, 2)  # [B, H, N]
        attn = masked_softmax(attn_logits, mask=mask.unsqueeze(1), dim=-1)
        bag_embed = torch.einsum("bhn,bnd->bhd", attn, x)
        return bag_embed, attn


class DemoRelationEncoder(nn.Module):
    """支持多种关系建模模式的轻量实现。"""

    def __init__(
        self,
        d_model: int,
        nhead: int = 8,
        num_layers: int = 2,
        dropout: float = 0.1,
        relation_type: str = "transformer",
    ) -> None:
        super().__init__()
        self.relation_type = relation_type.lower()

        if self.relation_type in {"transformer", "nystrom", "set_transformer"}:
            layer = nn.TransformerEncoderLayer(
                d_model=d_model,
                nhead=nhead,
                dim_feedforward=d_model * 4,
                dropout=dropout,
                batch_first=True,
                activation="gelu",
                norm_first=True,
            )
            self.encoder = nn.TransformerEncoder(layer, num_layers=num_layers)
        elif self.relation_type in {"mamba", "state_space"}:
            # 用双向 GRU 近似状态空间序列建模能力，避免额外依赖。
            self.encoder = nn.GRU(
                input_size=d_model,
                hidden_size=d_model // 2,
                num_layers=max(1, num_layers),
                dropout=dropout if num_layers > 1 else 0.0,
                batch_first=True,
                bidirectional=True,
            )
        elif self.relation_type in {"none", "identity"}:
            self.encoder = nn.Identity()
        else:
            raise ValueError(f"不支持的 relation_type: {relation_type}")

        self.norm = nn.LayerNorm(d_model)

    def forward(self, x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        """
        x: [B, N, D], mask: [B, N]
        """
        if isinstance(self.encoder, nn.TransformerEncoder):
            out = self.encoder(x, src_key_padding_mask=~mask.bool())
        elif isinstance(self.encoder, nn.GRU):
            out, _ = self.encoder(x)
        else:
            out = self.encoder(x)

        out = self.norm(out + x)
        out = out * mask.unsqueeze(-1).to(out.dtype)
        return out


class DemoSingleAttentionMIL(nn.Module):
    def __init__(self, in_dim: int, attn_dim: int, dropout: float = 0.1) -> None:
        super().__init__()
        self.attn = DemoGatedAttention(in_dim=in_dim, attn_dim=attn_dim, num_heads=1, dropout=dropout)

    def forward(self, x: torch.Tensor, mask: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        bag_embed, attn = self.attn(x, mask)
        return bag_embed[:, 0, :], attn[:, 0, :]


class DemoMultiLabelAttentionMIL(nn.Module):
    def __init__(self, in_dim: int, attn_dim: int, num_labels: int, dropout: float = 0.1) -> None:
        super().__init__()
        self.attn = DemoGatedAttention(in_dim=in_dim, attn_dim=attn_dim, num_heads=num_labels, dropout=dropout)

    def forward(self, x: torch.Tensor, mask: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """
        return:
          bag_embed: [B, L, D]
          attn: [B, L, N]
        """
        return self.attn(x, mask)
