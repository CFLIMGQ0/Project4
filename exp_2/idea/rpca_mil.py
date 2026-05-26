"""
RPCA MIL

问题动机：
- 强标签更像“广泛背景模式”，弱标签更像“稀疏病灶残差”
- 共享表征里这两类模式混在一起时，尾标签容易被头标签带偏

核心想法：
- 用低秩路径建模“稳定公共模式”
- 用残差路径建模“局部异常模式”
- 再让每个标签自己学共享/残差的融合比例
"""

from __future__ import annotations

import torch
import torch.nn as nn

from exp_2.common import Exp2AttentionMILBase


class RPCAMILModel(Exp2AttentionMILBase):
    def __init__(
        self,
        backbone_name: str = "resnet50",
        pretrained: bool = True,
        freeze_stages: int = 0,
        feature_dim: int = 512,
        attn_dim: int = 256,
        num_labels: int = 8,
        dropout: float = 0.2,
        rank_dim: int = 64,
        sparsity_weight: float = 0.01,
        low_rank_weight: float = 0.001,
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
        self.sparsity_weight = float(sparsity_weight)
        self.low_rank_weight = float(low_rank_weight)
        self.low_rank_encoder = nn.Linear(feature_dim, rank_dim)
        self.low_rank_decoder = nn.Linear(rank_dim, feature_dim)
        self.shared_pool = self.mil_pool
        self.residual_pool = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Linear(feature_dim, attn_dim),
                    nn.GELU(),
                    nn.Linear(attn_dim, 1),
                )
                for _ in range(num_labels)
            ]
        )
        init_mix = torch.tensor([0.9, 0.9, 0.9, 0.4, 0.35, 0.3, 0.3, 0.25], dtype=torch.float32)
        if init_mix.numel() < num_labels:
            init_mix = torch.cat([init_mix, init_mix.new_full((num_labels - init_mix.numel(),), 0.3)], dim=0)
        self.mix_logits = nn.Parameter(torch.logit(init_mix[:num_labels].clamp(1e-4, 1.0 - 1e-4)))

    def forward(self, images: torch.Tensor, mask: torch.Tensor) -> dict[str, torch.Tensor]:
        features = self.encode_instances(images)
        latent = self.low_rank_encoder(features)
        shared_features = self.low_rank_decoder(latent)
        residual_features = features - shared_features

        shared_bag, shared_attention = self.shared_pool(shared_features, mask)

        residual_logits = []
        for scorer in self.residual_pool:
            residual_logits.append(scorer(residual_features).squeeze(-1))
        residual_attention_logits = torch.stack(residual_logits, dim=1)
        residual_attention_logits = residual_attention_logits.masked_fill(
            ~mask.unsqueeze(1).bool(),
            torch.finfo(residual_attention_logits.dtype).min,
        )
        residual_attention = torch.softmax(residual_attention_logits, dim=-1)
        residual_attention = residual_attention * mask.unsqueeze(1).to(dtype=residual_attention.dtype)
        residual_attention = residual_attention / residual_attention.sum(dim=-1, keepdim=True).clamp_min(1e-12)
        residual_bag = torch.einsum("bln,bnd->bld", residual_attention, residual_features)

        mix = torch.sigmoid(self.mix_logits).view(1, self.num_labels, 1)
        bag_embeds = mix * shared_bag + (1.0 - mix) * residual_bag
        logits = self.classify_embeddings(bag_embeds)

        aux_losses = {
            "rpca_sparse": residual_features.abs().mean() * self.sparsity_weight,
            "rpca_low_rank": latent.pow(2).mean() * self.low_rank_weight,
        }
        attention = mix * shared_attention + (1.0 - mix) * residual_attention

        return {
            "logits": logits,
            "attention": attention,
            "instance_features": features,
            "shared_attention": shared_attention,
            "residual_attention": residual_attention,
            "aux_losses": aux_losses,
        }


__all__ = ["RPCAMILModel"]
