from __future__ import annotations

import math
from typing import Any

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from model.common import MultiLabelAttentionMIL, build_backbone


DEFAULT_CLASS_PRIOR = (0.15, 0.08, 0.75, 0.12, 0.10, 0.06, 0.04, 0.03)
DEFAULT_HEAD_LABEL_INDICES = (0, 1, 2)


def resolve_class_prior(num_labels: int, class_prior: list[float] | tuple[float, ...] | None = None) -> list[float]:
    base = list(class_prior) if class_prior is not None else list(DEFAULT_CLASS_PRIOR)
    if not base:
        base = [0.1] * num_labels
    if len(base) < num_labels:
        base.extend([base[-1]] * (num_labels - len(base)))
    return [float(item) for item in base[:num_labels]]


def resolve_head_label_indices(
    num_labels: int,
    head_label_indices: list[int] | tuple[int, ...] | None = None,
) -> tuple[int, ...]:
    indices = DEFAULT_HEAD_LABEL_INDICES if head_label_indices is None else tuple(int(item) for item in head_label_indices)
    valid = sorted({index for index in indices if 0 <= index < num_labels})
    return tuple(valid)


def build_tail_mask(num_labels: int, head_label_indices: list[int] | tuple[int, ...] | None = None) -> torch.Tensor:
    mask = torch.ones(num_labels, dtype=torch.bool)
    for index in resolve_head_label_indices(num_labels, head_label_indices):
        mask[index] = False
    return mask


def fbeta_score_from_counts(tp: int, fp: int, fn: int, beta: float = 1.0) -> float:
    beta_sq = float(beta) ** 2
    denom = (1.0 + beta_sq) * tp + beta_sq * fn + fp
    if denom <= 0:
        return 0.0
    return float((1.0 + beta_sq) * tp / denom)


def search_best_thresholds(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    *,
    beta_by_label: list[float] | tuple[float, ...] | None = None,
    min_threshold: float = 0.05,
    max_threshold: float = 0.95,
    num_steps: int = 19,
) -> np.ndarray:
    num_labels = int(y_prob.shape[1])
    beta_list = list(beta_by_label) if beta_by_label is not None else [1.0] * num_labels
    if len(beta_list) < num_labels:
        beta_list.extend([beta_list[-1]] * (num_labels - len(beta_list)))

    search_grid = np.linspace(min_threshold, max_threshold, num=num_steps, dtype=np.float32)
    thresholds: list[float] = []

    for label_index in range(num_labels):
        label_true = y_true[:, label_index].astype(np.int64)
        label_prob = y_prob[:, label_index]
        beta = float(beta_list[label_index])
        best_threshold = 0.5
        best_score = -1.0

        for threshold in search_grid:
            label_pred = (label_prob >= float(threshold)).astype(np.int64)
            tp = int(((label_pred == 1) & (label_true == 1)).sum())
            fp = int(((label_pred == 1) & (label_true == 0)).sum())
            fn = int(((label_pred == 0) & (label_true == 1)).sum())
            score = fbeta_score_from_counts(tp, fp, fn, beta=beta)
            if score > best_score:
                best_score = score
                best_threshold = float(threshold)

        thresholds.append(best_threshold)

    return np.asarray(thresholds, dtype=np.float32)


class PerLabelThresholdModule(nn.Module):
    def __init__(self, num_labels: int, init_threshold: float = 0.5) -> None:
        super().__init__()
        clipped = min(max(float(init_threshold), 1e-4), 1.0 - 1e-4)
        init_logit = math.log(clipped / (1.0 - clipped))
        self.threshold_logits = nn.Parameter(torch.full((num_labels,), init_logit))

    @property
    def thresholds(self) -> torch.Tensor:
        return torch.sigmoid(self.threshold_logits)

    def set_thresholds(self, thresholds: list[float] | np.ndarray | torch.Tensor) -> None:
        if torch.is_tensor(thresholds):
            values = thresholds.detach().float().cpu().numpy()
        else:
            values = np.asarray(thresholds, dtype=np.float32)
        values = np.clip(values.reshape(-1), 1e-4, 1.0 - 1e-4)
        with torch.no_grad():
            tensor = torch.tensor(values, dtype=torch.float32, device=self.threshold_logits.device)
            self.threshold_logits.copy_(torch.log(tensor / (1.0 - tensor)))


def masked_topk_softmax_pool(
    features: torch.Tensor,
    scores: torch.Tensor,
    mask: torch.Tensor,
    k: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    batch_size, num_labels, num_instances = scores.shape
    device = features.device
    attn = torch.zeros_like(scores)
    pooled = torch.zeros(batch_size, num_labels, features.shape[-1], device=device, dtype=features.dtype)
    mask_bool = mask.to(dtype=torch.bool)

    for batch_index in range(batch_size):
        valid_num = int(mask_bool[batch_index].sum().item())
        if valid_num <= 0:
            continue
        use_k = min(max(1, k), valid_num)
        current_scores = scores[batch_index, :, :valid_num]
        topk_values, topk_indices = torch.topk(current_scores, k=use_k, dim=-1)
        topk_weights = torch.softmax(topk_values, dim=-1).to(dtype=attn.dtype)
        attn[batch_index].scatter_(1, topk_indices, topk_weights)
        pooled[batch_index] = torch.einsum("ln,nd->ld", attn[batch_index, :, :valid_num], features[batch_index, :valid_num])

    return pooled, attn


class Exp2AttentionMILBase(nn.Module):
    def __init__(
        self,
        *,
        backbone_name: str = "resnet50",
        pretrained: bool = True,
        freeze_stages: int = 0,
        feature_dim: int = 512,
        attn_dim: int = 256,
        num_labels: int = 8,
        dropout: float = 0.2,
    ) -> None:
        super().__init__()
        self.num_labels = int(num_labels)
        self.feature_dim = int(feature_dim)
        self.encoder, out_dim = build_backbone(
            backbone_name=backbone_name,
            pretrained=pretrained,
            out_dim=feature_dim,
            freeze_stages=freeze_stages,
            projector_dropout=dropout,
        )
        self.shared_proj = nn.Sequential(
            nn.Linear(out_dim, feature_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(feature_dim, feature_dim),
            nn.ReLU(inplace=True),
        )
        self.mil_pool = MultiLabelAttentionMIL(
            in_dim=feature_dim,
            attn_dim=attn_dim,
            num_labels=num_labels,
            dropout=dropout,
        )
        self.classifiers = nn.ModuleList([nn.Linear(feature_dim, 1) for _ in range(num_labels)])

    def encode_instances(self, images: torch.Tensor) -> torch.Tensor:
        batch_size, num_instances, channels, height, width = images.shape
        flattened = images.reshape(batch_size * num_instances, channels, height, width)
        features = self.encoder(flattened).reshape(batch_size, num_instances, -1)
        return self.shared_proj(features)

    def pool_instances(self, features: torch.Tensor, mask: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        return self.mil_pool(features, mask)

    def transform_embeddings(
        self,
        bag_embeds: torch.Tensor,
        attention: torch.Tensor,
        features: torch.Tensor,
        mask: torch.Tensor,
        **kwargs: Any,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        del attention, features, mask, kwargs
        return bag_embeds, {}

    def classify_embeddings(self, bag_embeds: torch.Tensor) -> torch.Tensor:
        logits = [
            self.classifiers[label_index](bag_embeds[:, label_index, :]).squeeze(-1)
            for label_index in range(self.num_labels)
        ]
        return torch.stack(logits, dim=1)

    def forward(self, images: torch.Tensor, mask: torch.Tensor, **kwargs: Any) -> dict[str, torch.Tensor]:
        features = self.encode_instances(images)
        bag_embeds, attention = self.pool_instances(features, mask)
        bag_embeds, extra_outputs = self.transform_embeddings(
            bag_embeds=bag_embeds,
            attention=attention,
            features=features,
            mask=mask,
            **kwargs,
        )
        outputs = {
            "logits": self.classify_embeddings(bag_embeds),
            "attention": attention,
            "instance_features": features,
        }
        outputs.update(extra_outputs)
        return outputs


__all__ = [
    "DEFAULT_CLASS_PRIOR",
    "DEFAULT_HEAD_LABEL_INDICES",
    "PerLabelThresholdModule",
    "Exp2AttentionMILBase",
    "build_tail_mask",
    "fbeta_score_from_counts",
    "masked_topk_softmax_pool",
    "resolve_class_prior",
    "resolve_head_label_indices",
    "search_best_thresholds",
]
