from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from model.common import build_backbone


class InstanceEncoder(nn.Module):
    """共享实例编码器。"""

    def __init__(
        self,
        backbone_name: str,
        pretrained: bool,
        freeze_stages: int,
        feature_dim: int,
        dropout: float,
    ) -> None:
        super().__init__()
        self.backbone, out_dim = build_backbone(
            backbone_name=backbone_name,
            pretrained=pretrained,
            out_dim=feature_dim,
            freeze_stages=freeze_stages,
            projector_dropout=dropout,
        )
        self.projector = nn.Sequential(
            nn.Linear(out_dim, feature_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(feature_dim, feature_dim),
            nn.GELU(),
        )

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        batch_size, num_instances, channels, height, width = images.shape
        x = images.reshape(batch_size * num_instances, channels, height, width)
        features = self.backbone(x).reshape(batch_size, num_instances, -1)
        return self.projector(features)


class LabelGraphReasoner(nn.Module):
    """基于可学习标签图对标签表征做一次关系传播。"""

    def __init__(self, num_labels: int, feature_dim: int, dropout: float) -> None:
        super().__init__()
        self.label_tokens = nn.Parameter(torch.randn(num_labels, feature_dim) * 0.02)
        self.refine = nn.Sequential(
            nn.Linear(feature_dim * 2, feature_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )

    def forward(self, bag_embeds: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        label_tokens = F.normalize(self.label_tokens, dim=-1)
        graph_logits = torch.matmul(label_tokens, label_tokens.transpose(0, 1)) / math.sqrt(label_tokens.shape[-1])
        label_graph = torch.softmax(graph_logits, dim=-1)

        propagated = torch.einsum("lk,bkd->bld", label_graph, bag_embeds)
        refined = bag_embeds + self.refine(torch.cat([bag_embeds, propagated], dim=-1))
        return refined, label_graph
