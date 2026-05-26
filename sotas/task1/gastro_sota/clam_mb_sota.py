from __future__ import annotations

import torch
import torch.nn as nn

from .common import (
    GastroSOTAInstanceEncoder,
    GatedLabelAttention,
    LabelwiseClassifier,
    attention_diversity,
    attention_entropy,
    build_instance_pseudo_loss,
)


class GastroCLAMMBSOTA(nn.Module):
    """CLAM-MB 思路的多分支多标签聚合模型。"""

    def __init__(
        self,
        backbone_name: str = "convnext_tiny",
        pretrained: bool = True,
        freeze_stages: int = 1,
        feature_dim: int = 512,
        attn_dim: int = 256,
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
        self.label_attention = GatedLabelAttention(
            in_dim=feature_dim,
            attn_dim=attn_dim,
            num_labels=num_labels,
            dropout=dropout,
        )
        self.bag_classifier = LabelwiseClassifier(feature_dim, hidden_dim, num_labels, dropout)
        self.instance_classifier = nn.Linear(feature_dim, num_labels)

    def forward(
        self,
        images: torch.Tensor,
        mask: torch.Tensor,
        labels: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        features = self.instance_encoder(images)
        bag_embeds, attention = self.label_attention(features, mask)
        logits = self.bag_classifier(bag_embeds)

        aux_losses: dict[str, torch.Tensor] = {
            "attention_entropy": attention_entropy(attention, mask),
            "attention_diversity": attention_diversity(attention),
        }
        if labels is not None:
            instance_logits = self.instance_classifier(features)
            aux_losses["instance_clustering"] = build_instance_pseudo_loss(
                instance_logits=instance_logits,
                attention=attention,
                labels=labels,
                mask=mask,
                topk=self.instance_topk,
            )

        return {
            "logits": logits,
            "attention": attention,
            "instance_features": features,
            "aux_losses": aux_losses,
        }
