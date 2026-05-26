from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from .common import (
    GastroSOTAInstanceEncoder,
    GatedLabelAttention,
    LabelwiseClassifier,
    masked_label_attention,
    split_group_ranges,
)


class GastroDTFDMILSOTA(nn.Module):
    """DTFD-MIL 思路的双层伪 bag 蒸馏模型。"""

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
        num_groups: int = 4,
    ) -> None:
        super().__init__()
        self.num_labels = num_labels
        self.num_groups = max(1, int(num_groups))
        self.instance_encoder = GastroSOTAInstanceEncoder(
            backbone_name=backbone_name,
            pretrained=pretrained,
            freeze_stages=freeze_stages,
            feature_dim=feature_dim,
            dropout=dropout,
        )
        self.full_scorer = nn.Sequential(
            nn.Linear(feature_dim, attn_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(attn_dim, num_labels),
        )
        self.group_attention = GatedLabelAttention(feature_dim, attn_dim, num_labels, dropout)
        self.group_classifier = LabelwiseClassifier(feature_dim, hidden_dim, num_labels, dropout)
        self.final_classifier = LabelwiseClassifier(feature_dim, hidden_dim, num_labels, dropout)

    def forward(
        self,
        images: torch.Tensor,
        mask: torch.Tensor,
        labels: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        features = self.instance_encoder(images)
        full_logits = self.full_scorer(features).transpose(1, 2)
        attention = masked_label_attention(full_logits, mask)

        group_ranges = split_group_ranges(features.size(1), self.num_groups)
        group_embeds = []
        group_logits = []
        group_valid = []
        for start, end in group_ranges:
            group_features = features[:, start:end, :]
            group_mask = mask[:, start:end]
            embeds, _ = self.group_attention(group_features, group_mask)
            logits = self.group_classifier(embeds)
            group_embeds.append(embeds)
            group_logits.append(logits)
            group_valid.append(group_mask.any(dim=1))

        stacked_embeds = torch.stack(group_embeds, dim=1)
        stacked_logits = torch.stack(group_logits, dim=1)
        group_valid_mask = torch.stack(group_valid, dim=1)

        group_scores = torch.sigmoid(stacked_logits).transpose(1, 2)
        group_scores = group_scores.masked_fill(
            ~group_valid_mask.unsqueeze(1).to(dtype=torch.bool),
            torch.finfo(group_scores.dtype).min,
        )
        group_weights = torch.softmax(group_scores, dim=-1)
        group_weights = group_weights * group_valid_mask.unsqueeze(1).to(dtype=group_weights.dtype)
        group_weights = group_weights / group_weights.sum(dim=-1, keepdim=True).clamp_min(1e-12)

        final_embeds = torch.einsum("blg,bgld->bld", group_weights, stacked_embeds)
        logits = self.final_classifier(final_embeds)

        aux_losses: dict[str, torch.Tensor] = {}
        if labels is not None:
            repeated_targets = labels.unsqueeze(1).expand(-1, stacked_logits.size(1), -1).to(dtype=stacked_logits.dtype)
            valid_groups = group_valid_mask.unsqueeze(-1).to(dtype=stacked_logits.dtype)
            pseudo_loss = F.binary_cross_entropy_with_logits(
                stacked_logits,
                repeated_targets,
                reduction="none",
            )
            aux_losses["pseudo_bag"] = (pseudo_loss * valid_groups).sum() / valid_groups.sum().clamp_min(1.0)

        return {
            "logits": logits,
            "attention": attention,
            "instance_features": features,
            "aux_losses": aux_losses,
        }
