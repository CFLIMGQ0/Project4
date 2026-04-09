from __future__ import annotations

import torch
import torch.nn as nn

from model.common import MultiLabelAttentionMIL

from .modules import InstanceEncoder, LabelGraphReasoner


class GastroLabelGraphMIL(nn.Module):
    """胃镜标签图传播 MIL。"""

    def __init__(
        self,
        backbone_name: str = "convnext_tiny",
        pretrained: bool = True,
        freeze_stages: int = 0,
        feature_dim: int = 512,
        attn_dim: int = 256,
        num_labels: int = 3,
        dropout: float = 0.2,
    ) -> None:
        super().__init__()
        self.num_labels = num_labels
        self.instance_encoder = InstanceEncoder(
            backbone_name=backbone_name,
            pretrained=pretrained,
            freeze_stages=freeze_stages,
            feature_dim=feature_dim,
            dropout=dropout,
        )
        self.mil_pool = MultiLabelAttentionMIL(
            in_dim=feature_dim,
            attn_dim=attn_dim,
            num_labels=num_labels,
            dropout=dropout,
        )
        self.label_graph_reasoner = LabelGraphReasoner(
            num_labels=num_labels,
            feature_dim=feature_dim,
            dropout=dropout,
        )
        self.classifiers = nn.ModuleList([nn.Linear(feature_dim, 1) for _ in range(num_labels)])

    def encode_instances(self, images: torch.Tensor) -> torch.Tensor:
        return self.instance_encoder(images)

    def forward(self, images: torch.Tensor, mask: torch.Tensor) -> dict[str, torch.Tensor]:
        features = self.encode_instances(images)
        bag_embeds, attention = self.mil_pool(features, mask)
        refined_embeds, label_graph = self.label_graph_reasoner(bag_embeds)

        logits = []
        for label_index in range(self.num_labels):
            logits.append(self.classifiers[label_index](refined_embeds[:, label_index, :]).squeeze(-1))

        return {
            "logits": torch.stack(logits, dim=1),
            "attention": attention,
            "instance_features": features,
            "label_graph": label_graph,
        }
