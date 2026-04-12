from __future__ import annotations

import torch
import torch.nn as nn

from model.common import build_backbone, masked_softmax


def build_shared_projection(feature_dim: int, dropout: float) -> nn.Sequential:
    return nn.Sequential(
        nn.Linear(feature_dim, feature_dim),
        nn.ReLU(inplace=True),
        nn.Dropout(dropout),
        nn.Linear(feature_dim, feature_dim),
        nn.ReLU(inplace=True),
    )


def masked_mean_pool(features: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    mask_float = mask.to(dtype=features.dtype).unsqueeze(-1)
    denom = mask_float.sum(dim=1).clamp_min(1.0)
    return (features * mask_float).sum(dim=1) / denom


def uniform_attention_from_mask(mask: torch.Tensor, num_labels: int) -> torch.Tensor:
    attention = mask.to(dtype=torch.float32).unsqueeze(1).repeat(1, num_labels, 1)
    denom = attention.sum(dim=-1, keepdim=True).clamp_min(1e-12)
    return attention / denom


def masked_score_logits(score_logits: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    return score_logits.masked_fill(~mask.unsqueeze(1).to(dtype=torch.bool), torch.finfo(score_logits.dtype).min)


def normalize_topk_weights(score_logits: torch.Tensor, valid_mask: torch.Tensor) -> torch.Tensor:
    masked_logits = score_logits.masked_fill(~valid_mask, torch.finfo(score_logits.dtype).min)
    weights = torch.softmax(masked_logits, dim=-1)
    weights = weights * valid_mask.to(dtype=weights.dtype)
    denom = weights.sum(dim=-1, keepdim=True).clamp_min(1e-12)
    return weights / denom


class BaseGastroBaseline(nn.Module):
    def __init__(
        self,
        backbone_name: str = "resnet50",
        pretrained: bool = True,
        freeze_stages: int = 0,
        feature_dim: int = 512,
        num_labels: int = 3,
        dropout: float = 0.2,
    ) -> None:
        super().__init__()
        self.num_labels = num_labels
        self.feature_dim = feature_dim
        self.encoder, _ = build_backbone(
            backbone_name=backbone_name,
            pretrained=pretrained,
            out_dim=feature_dim,
            freeze_stages=freeze_stages,
            projector_dropout=dropout,
        )
        self.shared_proj = build_shared_projection(feature_dim=feature_dim, dropout=dropout)

    def encode_instances(self, images: torch.Tensor) -> torch.Tensor:
        batch_size, num_instances, channels, height, width = images.shape
        x = images.reshape(batch_size * num_instances, channels, height, width)
        features = self.encoder(x).reshape(batch_size, num_instances, -1)
        return self.shared_proj(features)

    def attention_from_scores(self, score_logits: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        return masked_softmax(score_logits, mask=mask.unsqueeze(1), dim=-1)
