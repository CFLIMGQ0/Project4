from __future__ import annotations

import torch
import torch.nn as nn

from .demo_backbones import build_backbone
from .demo_mil_pooling import DemoSingleAttentionMIL


class DemoColoMILBaseline(nn.Module):
    """肠镜 baseline：单 attention MIL，支持二分类/三分类扩展。"""

    def __init__(
        self,
        backbone_name: str = "resnet50",
        pretrained: bool = True,
        freeze_stages: int = 0,
        feature_dim: int = 512,
        attn_dim: int = 256,
        num_classes: int = 2,
        dropout: float = 0.2,
    ) -> None:
        super().__init__()
        self.num_classes = num_classes

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

        self.mil_pool = DemoSingleAttentionMIL(in_dim=feature_dim, attn_dim=attn_dim, dropout=dropout)
        if num_classes == 2:
            self.classifier = nn.Linear(feature_dim, 1)
        else:
            self.classifier = nn.Linear(feature_dim, num_classes)

    def encode_instances(self, images: torch.Tensor) -> torch.Tensor:
        b, n, c, h, w = images.shape
        x = images.reshape(b * n, c, h, w)
        feat = self.encoder(x).reshape(b, n, -1)
        feat = self.shared_proj(feat)
        return feat

    def forward(self, images: torch.Tensor, mask: torch.Tensor) -> dict[str, torch.Tensor]:
        feats = self.encode_instances(images)
        bag_embed, attn = self.mil_pool(feats, mask)
        logits = self.classifier(bag_embed)
        if self.num_classes == 2:
            logits = logits.squeeze(-1)

        return {
            "logits": logits,
            "attention": attn,
            "instance_features": feats,
        }
