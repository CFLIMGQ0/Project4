from __future__ import annotations

import torch
import torch.nn as nn

from .common import BaseGastroBaseline, masked_score_logits, normalize_topk_weights


class GastroTopKMILBaseline(BaseGastroBaseline):
    """基于标签相关 top-k 实例聚合的胃镜 baseline。"""

    def __init__(
        self,
        backbone_name: str = "resnet50",
        pretrained: bool = True,
        freeze_stages: int = 0,
        feature_dim: int = 512,
        hidden_dim: int = 256,
        num_labels: int = 3,
        dropout: float = 0.2,
        topk: int = 4,
    ) -> None:
        super().__init__(
            backbone_name=backbone_name,
            pretrained=pretrained,
            freeze_stages=freeze_stages,
            feature_dim=feature_dim,
            num_labels=num_labels,
            dropout=dropout,
        )
        self.topk = max(1, int(topk))
        self.instance_scorer = nn.Sequential(
            nn.Linear(feature_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_labels),
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
        score_logits = self.instance_scorer(features).transpose(1, 2)
        score_logits = masked_score_logits(score_logits, mask)

        topk = min(self.topk, features.size(1))
        topk_scores, topk_indices = torch.topk(score_logits, k=topk, dim=-1)
        expanded_mask = mask.unsqueeze(1).expand(-1, self.num_labels, -1)
        topk_valid_mask = torch.gather(expanded_mask, 2, topk_indices).to(dtype=torch.bool)
        topk_weights = normalize_topk_weights(topk_scores, topk_valid_mask)

        expanded_features = features.unsqueeze(1).expand(-1, self.num_labels, -1, -1)
        gathered_features = torch.gather(
            expanded_features,
            2,
            topk_indices.unsqueeze(-1).expand(-1, -1, -1, features.size(-1)),
        )
        bag_embeds = (gathered_features * topk_weights.unsqueeze(-1)).sum(dim=2)

        logits = []
        for label_index in range(self.num_labels):
            logits.append(self.classifiers[label_index](bag_embeds[:, label_index, :]).squeeze(-1))

        attention = topk_weights.new_zeros(features.size(0), self.num_labels, features.size(1))
        attention.scatter_(2, topk_indices, topk_weights.to(dtype=attention.dtype))
        attention = attention * expanded_mask.to(dtype=attention.dtype)

        return {
            "logits": torch.stack(logits, dim=1),
            "attention": attention,
            "instance_features": features,
        }
