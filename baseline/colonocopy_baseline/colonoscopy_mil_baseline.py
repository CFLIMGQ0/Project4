from __future__ import annotations

import torch
import torch.nn as nn

from model.common import SingleAttentionMIL, build_backbone


class ColonoscopyMILBaseline(nn.Module):
    """肠镜二分类 baseline。"""

    def __init__(
        self,
        backbone_name: str = "resnet50",
        pretrained: bool = True,
        freeze_stages: int = 0,
        feature_dim: int = 512,
        attn_dim: int = 256,
        dropout: float = 0.2,
    ) -> None:
        super().__init__()
        self.encoder, out_dim = build_backbone(
            backbone_name=backbone_name,
            pretrained=pretrained,
            out_dim=feature_dim,
            freeze_stages=freeze_stages,
            projector_dropout=dropout,
        )
        self.shared_proj = nn.Sequential(
            nn.Linear(out_dim, feature_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(feature_dim, feature_dim),
            nn.ReLU(inplace=True),
        )
        self.mil_pool = SingleAttentionMIL(in_dim=feature_dim, attn_dim=attn_dim, dropout=dropout)
        self.classifier = nn.Linear(feature_dim, 1)

    def encode_instances(self, images: torch.Tensor) -> torch.Tensor:
        batch_size, num_instances, channels, height, width = images.shape
        x = images.reshape(batch_size * num_instances, channels, height, width)
        features = self.encoder(x).reshape(batch_size, num_instances, -1)
        return self.shared_proj(features)

    def forward(self, images: torch.Tensor, mask: torch.Tensor) -> dict[str, torch.Tensor]:
        features = self.encode_instances(images)
        bag_embed, attention = self.mil_pool(features, mask)
        logits = self.classifier(bag_embed).squeeze(-1)
        return {
            "logits": logits,
            "attention": attention,
            "instance_features": features,
        }
