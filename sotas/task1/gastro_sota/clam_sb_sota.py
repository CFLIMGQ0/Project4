from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from .common import GastroSOTAInstanceEncoder, attention_entropy, build_mlp


class GastroCLAMSBSOTA(nn.Module):
    """CLAM-SB 思路的单分支聚合模型。"""

    def __init__(
        self,
        backbone_name: str = "convnext_tiny",
        pretrained: bool = True,
        freeze_stages: int = 1,
        feature_dim: int = 512,
        hidden_dim: int = 256,
        num_labels: int = 3,
        dropout: float = 0.2,
        instance_topk: int = 4,
    ) -> None:
        super().__init__()
        self.num_labels = num_labels
        self.instance_topk = max(1, int(instance_topk))
        self.instance_encoder = GastroSOTAInstanceEncoder(
            backbone_name=backbone_name,
            pretrained=pretrained,
            freeze_stages=freeze_stages,
            feature_dim=feature_dim,
            dropout=dropout,
        )
        self.attention = build_mlp(feature_dim, hidden_dim, 1, dropout)
        self.bag_classifier = build_mlp(feature_dim, hidden_dim, num_labels, dropout)
        self.instance_classifier = nn.Linear(feature_dim, num_labels)

    def forward(
        self,
        images: torch.Tensor,
        mask: torch.Tensor,
        labels: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        features = self.instance_encoder(images)
        attention_logits = self.attention(features).squeeze(-1)
        attention_logits = attention_logits.masked_fill(~mask.to(dtype=torch.bool), torch.finfo(attention_logits.dtype).min)
        attention = torch.softmax(attention_logits, dim=-1)
        attention = attention * mask.to(dtype=attention.dtype)
        attention = attention / attention.sum(dim=-1, keepdim=True).clamp_min(1e-12)

        bag_embed = torch.einsum("bn,bnd->bd", attention, features)
        logits = self.bag_classifier(bag_embed)

        aux_losses: dict[str, torch.Tensor] = {
            "attention_entropy": attention_entropy(attention.unsqueeze(1), mask),
        }
        if labels is not None:
            instance_logits = self.instance_classifier(features)
            topk = min(self.instance_topk, features.size(1))
            top_indices = torch.topk(attention_logits, k=topk, dim=-1).indices
            bottom_indices = torch.topk(-attention_logits, k=topk, dim=-1).indices
            top_scores = torch.gather(instance_logits, 1, top_indices.unsqueeze(-1).expand(-1, -1, self.num_labels))
            bottom_scores = torch.gather(
                instance_logits,
                1,
                bottom_indices.unsqueeze(-1).expand(-1, -1, self.num_labels),
            )
            top_targets = labels.unsqueeze(1).expand(-1, topk, -1).to(dtype=top_scores.dtype)
            bottom_targets = torch.zeros_like(bottom_scores)
            aux_losses["instance_clustering"] = 0.5 * (
                F.binary_cross_entropy_with_logits(top_scores, top_targets)
                + F.binary_cross_entropy_with_logits(bottom_scores, bottom_targets)
            )

        return {
            "logits": logits,
            "attention": attention.unsqueeze(1).repeat(1, self.num_labels, 1),
            "instance_features": features,
            "aux_losses": aux_losses,
        }
