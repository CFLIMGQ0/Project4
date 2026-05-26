from __future__ import annotations

import torch
import torch.nn as nn

from .common import BaseGastroBaseline, masked_mean_pool, uniform_attention_from_mask


class GastroMeanPoolBaseline(BaseGastroBaseline):
    """均值池化胃镜 baseline。"""

    def __init__(
        self,
        backbone_name: str = "resnet50",
        pretrained: bool = True,
        freeze_stages: int = 0,
        feature_dim: int = 512,
        hidden_dim: int = 512,
        num_labels: int = 3,
        dropout: float = 0.2,
    ) -> None:
        super().__init__(
            backbone_name=backbone_name,
            pretrained=pretrained,
            freeze_stages=freeze_stages,
            feature_dim=feature_dim,
            num_labels=num_labels,
            dropout=dropout,
        )
        self.head = nn.Sequential(
            nn.Linear(feature_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_labels),
        )

    def forward(self, images: torch.Tensor, mask: torch.Tensor) -> dict[str, torch.Tensor]:
        features = self.encode_instances(images)
        bag_embed = masked_mean_pool(features, mask)
        return {
            "logits": self.head(bag_embed),
            "attention": uniform_attention_from_mask(mask, self.num_labels),
            "instance_features": features,
        }
