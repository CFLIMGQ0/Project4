from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from model.common import build_backbone, masked_softmax


def build_mlp(
    in_dim: int,
    hidden_dim: int,
    out_dim: int,
    dropout: float,
    *,
    activate_last: bool = False,
) -> nn.Sequential:
    layers: list[nn.Module] = [
        nn.Linear(in_dim, hidden_dim),
        nn.GELU(),
        nn.Dropout(dropout),
        nn.Linear(hidden_dim, out_dim),
    ]
    if activate_last:
        layers.extend([nn.GELU(), nn.Dropout(dropout)])
    return nn.Sequential(*layers)


def masked_mean_pool(features: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    mask_float = mask.to(dtype=features.dtype).unsqueeze(-1)
    denom = mask_float.sum(dim=1).clamp_min(1.0)
    return (features * mask_float).sum(dim=1) / denom


def gather_label_topk(
    tensor: torch.Tensor,
    indices: torch.Tensor,
) -> torch.Tensor:
    return torch.gather(
        tensor,
        2,
        indices.unsqueeze(-1).expand(-1, -1, -1, tensor.size(-1)),
    )


def attention_entropy(attention: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    masked_attention = attention * mask.unsqueeze(1).to(dtype=attention.dtype)
    entropy = -(masked_attention.clamp_min(1e-8) * masked_attention.clamp_min(1e-8).log()).sum(dim=-1)
    return entropy.mean()


def attention_diversity(attention: torch.Tensor) -> torch.Tensor:
    if attention.size(1) <= 1:
        return attention.new_zeros(())
    normalized = F.normalize(attention, dim=-1)
    similarity = torch.matmul(normalized, normalized.transpose(1, 2))
    label_count = attention.size(1)
    eye = torch.eye(label_count, device=attention.device, dtype=attention.dtype).unsqueeze(0)
    penalty = (similarity * (1.0 - eye)).sum(dim=(1, 2)) / max(1, label_count * (label_count - 1))
    return penalty.mean()


def masked_label_attention(score_logits: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    return masked_softmax(score_logits, mask=mask.unsqueeze(1), dim=-1)


def ensure_num_heads(feature_dim: int, requested_heads: int) -> int:
    heads = max(1, min(int(requested_heads), int(feature_dim)))
    while feature_dim % heads != 0 and heads > 1:
        heads -= 1
    return heads


class GastroSOTAInstanceEncoder(nn.Module):
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


class GatedLabelAttention(nn.Module):
    def __init__(self, in_dim: int, attn_dim: int, num_labels: int, dropout: float) -> None:
        super().__init__()
        self.v = nn.Sequential(nn.Linear(in_dim, attn_dim), nn.Tanh(), nn.Dropout(dropout))
        self.u = nn.Sequential(nn.Linear(in_dim, attn_dim), nn.Sigmoid(), nn.Dropout(dropout))
        self.w = nn.Linear(attn_dim, num_labels)

    def forward(self, x: torch.Tensor, mask: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        logits = self.w(self.v(x) * self.u(x)).transpose(1, 2)
        attention = masked_label_attention(logits, mask)
        embeds = torch.einsum("bln,bnd->bld", attention, x)
        return embeds, attention


class LabelwiseClassifier(nn.Module):
    def __init__(self, feature_dim: int, hidden_dim: int, num_labels: int, dropout: float) -> None:
        super().__init__()
        self.classifiers = nn.ModuleList(
            [build_mlp(feature_dim, hidden_dim, 1, dropout) for _ in range(num_labels)]
        )

    def forward(self, embeds: torch.Tensor) -> torch.Tensor:
        logits = []
        for label_index, classifier in enumerate(self.classifiers):
            logits.append(classifier(embeds[:, label_index, :]).squeeze(-1))
        return torch.stack(logits, dim=1)


def build_instance_pseudo_loss(
    instance_logits: torch.Tensor,
    attention: torch.Tensor,
    labels: torch.Tensor,
    mask: torch.Tensor,
    topk: int,
) -> torch.Tensor:
    valid_mask = mask.to(dtype=torch.bool).unsqueeze(1)
    masked_attention = attention.masked_fill(~valid_mask, torch.finfo(attention.dtype).min)
    k = min(max(1, int(topk)), attention.size(-1))
    topk_indices = torch.topk(masked_attention, k=k, dim=-1).indices
    bottomk_indices = torch.topk(-masked_attention, k=k, dim=-1).indices

    expanded_logits = instance_logits.transpose(1, 2).unsqueeze(-1)
    expanded_logits = expanded_logits.squeeze(-1)
    topk_scores = torch.gather(expanded_logits, 2, topk_indices)
    bottomk_scores = torch.gather(expanded_logits, 2, bottomk_indices)

    topk_targets = labels.unsqueeze(-1).expand(-1, -1, k).to(dtype=topk_scores.dtype)
    bottomk_targets = torch.zeros_like(bottomk_scores)

    loss_top = F.binary_cross_entropy_with_logits(topk_scores, topk_targets)
    loss_bottom = F.binary_cross_entropy_with_logits(bottomk_scores, bottomk_targets)
    return 0.5 * (loss_top + loss_bottom)


def split_group_ranges(num_instances: int, num_groups: int) -> list[tuple[int, int]]:
    group_count = max(1, min(int(num_groups), int(num_instances)))
    chunk_size = math.ceil(num_instances / group_count)
    ranges: list[tuple[int, int]] = []
    start = 0
    while start < num_instances:
        end = min(num_instances, start + chunk_size)
        ranges.append((start, end))
        start = end
    return ranges
