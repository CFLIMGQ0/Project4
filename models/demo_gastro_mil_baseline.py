from __future__ import annotations

import torch
import torch.nn as nn

from .demo_backbones import build_backbone
from .demo_mil_pooling import DemoMultiLabelAttentionMIL


class DemoGastroMILBaseline(nn.Module):
    """胃镜多标签 MIL baseline：共享实例编码 + 标签独立 attention head。"""

    def __init__(
        self,
        backbone_name: str = "resnet50",
        pretrained: bool = True,
        freeze_stages: int = 0,
        feature_dim: int = 512,
        attn_dim: int = 256,
        num_labels: int = 3,
        dropout: float = 0.2,
    ) -> None:
        super().__init__()
        self.num_labels = num_labels

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

        self.mil_pool = DemoMultiLabelAttentionMIL(
            in_dim=feature_dim,
            attn_dim=attn_dim,
            num_labels=num_labels,
            dropout=dropout,
        )
        self.classifiers = nn.ModuleList([nn.Linear(feature_dim, 1) for _ in range(num_labels)])

    def encode_instances(self, images: torch.Tensor) -> torch.Tensor:
        b, n, c, h, w = images.shape
        x = images.reshape(b * n, c, h, w)
        feat = self.encoder(x)
        feat = feat.reshape(b, n, -1)
        feat = self.shared_proj(feat)
        return feat

    def forward(self, images: torch.Tensor, mask: torch.Tensor) -> dict[str, torch.Tensor]:
        """
        images: [B, N, C, H, W]
        mask: [B, N]
        """
        feats = self.encode_instances(images)
        bag_embeds, attn = self.mil_pool(feats, mask)  # [B, L, D], [B, L, N]

        logits = []
        for i in range(self.num_labels):
            logits.append(self.classifiers[i](bag_embeds[:, i, :]).squeeze(-1))
        logits = torch.stack(logits, dim=1)  # [B, L]

        return {
            "logits": logits,
            "attention": attn,
            "instance_features": feats,
        }
