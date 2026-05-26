from __future__ import annotations

import math

import torch
import torch.nn as nn

from .common import GastroSOTAInstanceEncoder, LabelwiseClassifier


class GastroDSMILSOTA(nn.Module):
    """DSMIL 思路的关键实例关系聚合模型。"""

    def __init__(
        self,
        backbone_name: str = "convnext_tiny",
        pretrained: bool = True,
        freeze_stages: int = 1,
        feature_dim: int = 512,
        hidden_dim: int = 256,
        num_labels: int = 3,
        dropout: float = 0.2,
    ) -> None:
        super().__init__()
        self.num_labels = num_labels
        self.instance_encoder = GastroSOTAInstanceEncoder(
            backbone_name=backbone_name,
            pretrained=pretrained,
            freeze_stages=freeze_stages,
            feature_dim=feature_dim,
            dropout=dropout,
        )
        self.instance_classifier = nn.Linear(feature_dim, num_labels)
        self.query_proj = nn.Linear(feature_dim, feature_dim)
        self.key_proj = nn.Linear(feature_dim, feature_dim)
        self.value_proj = nn.Linear(feature_dim, feature_dim)
        self.bag_classifier = LabelwiseClassifier(feature_dim, hidden_dim, num_labels, dropout)

    def forward(self, images: torch.Tensor, mask: torch.Tensor) -> dict[str, torch.Tensor]:
        features = self.instance_encoder(images)
        instance_logits = self.instance_classifier(features)
        masked_instance_logits = instance_logits.masked_fill(
            ~mask.unsqueeze(-1).to(dtype=torch.bool),
            torch.finfo(instance_logits.dtype).min,
        )
        critical_indices = masked_instance_logits.transpose(1, 2).argmax(dim=-1)
        critical_features = torch.gather(
            features.unsqueeze(1).expand(-1, self.num_labels, -1, -1),
            2,
            critical_indices.unsqueeze(-1).unsqueeze(-1).expand(-1, -1, 1, features.size(-1)),
        ).squeeze(2)

        query = self.query_proj(critical_features)
        key = self.key_proj(features)
        value = self.value_proj(features)
        attention_logits = torch.einsum("bld,bnd->bln", query, key) / math.sqrt(features.size(-1))
        attention_logits = attention_logits.masked_fill(
            ~mask.unsqueeze(1).to(dtype=torch.bool),
            torch.finfo(attention_logits.dtype).min,
        )
        attention = torch.softmax(attention_logits, dim=-1)
        attention = attention * mask.unsqueeze(1).to(dtype=attention.dtype)
        attention = attention / attention.sum(dim=-1, keepdim=True).clamp_min(1e-12)

        bag_embeds = torch.einsum("bln,bnd->bld", attention, value)
        bag_logits = self.bag_classifier(bag_embeds)
        critical_logits = torch.gather(
            instance_logits.transpose(1, 2),
            2,
            critical_indices.unsqueeze(-1),
        ).squeeze(-1)

        return {
            "logits": 0.5 * (bag_logits + critical_logits),
            "attention": attention,
            "instance_features": features,
        }
