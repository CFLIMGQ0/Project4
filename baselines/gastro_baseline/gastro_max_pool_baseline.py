from __future__ import annotations

import torch
import torch.nn as nn

from .common import BaseGastroBaseline


class GastroMaxPoolBaseline(BaseGastroBaseline):
    """按标签选取最高响应实例的胃镜 baseline。"""

    def __init__(
        self,
        backbone_name: str = "resnet50",
        pretrained: bool = True,
        freeze_stages: int = 0,
        feature_dim: int = 512,
        hidden_dim: int = 256,
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
        self.instance_scorers = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Linear(feature_dim, hidden_dim),
                    nn.ReLU(inplace=True),
                    nn.Dropout(dropout),
                    nn.Linear(hidden_dim, 1),
                )
                for _ in range(num_labels)
            ]
        )
        self.classifiers = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Linear(feature_dim, hidden_dim),
                    nn.ReLU(inplace=True),
                    nn.Dropout(dropout),
                    nn.Linear(hidden_dim, 1),
                )
                for _ in range(num_labels)
            ]
        )

    def forward(self, images: torch.Tensor, mask: torch.Tensor) -> dict[str, torch.Tensor]:
        features = self.encode_instances(images)
        valid_mask = mask.to(dtype=torch.bool)

        attention_list: list[torch.Tensor] = []
        logits_list: list[torch.Tensor] = []
        for label_index in range(self.num_labels):
            score_logits = self.instance_scorers[label_index](features).squeeze(-1)
            score_logits = score_logits.masked_fill(~valid_mask, torch.finfo(score_logits.dtype).min)
            max_indices = score_logits.argmax(dim=1, keepdim=True)
            gather_index = max_indices.unsqueeze(-1).expand(-1, 1, features.size(-1))
            bag_embed = features.gather(1, gather_index).squeeze(1)
            logits_list.append(self.classifiers[label_index](bag_embed).squeeze(-1))

            attention = torch.zeros_like(score_logits)
            attention.scatter_(1, max_indices, 1.0)
            attention = attention * valid_mask.to(dtype=attention.dtype)
            attention_list.append(attention)

        return {
            "logits": torch.stack(logits_list, dim=1),
            "attention": torch.stack(attention_list, dim=1),
            "instance_features": features,
        }
