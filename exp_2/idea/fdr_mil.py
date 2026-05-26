"""
FDR-Regularized MIL

问题动机：
- `exp_1` 里最明显的问题不是“完全看不见弱标签”，而是弱标签假阳性太多
- 特别是 `active`、`atrophy`、`reflux`，经常出现 recall 很高、precision 很低

核心想法：
- 训练时直接惩罚尾标签的期望假发现率（FDR）
- 推理时不再统一用 0.5，而是对尾标签用更偏 precision 的阈值搜索
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn.functional as F

from exp_2.common import (
    Exp2AttentionMILBase,
    PerLabelThresholdModule,
    build_tail_mask,
    resolve_head_label_indices,
    search_best_thresholds,
)


class FDRMILModel(Exp2AttentionMILBase):
    def __init__(
        self,
        backbone_name: str = "resnet50",
        pretrained: bool = True,
        freeze_stages: int = 0,
        feature_dim: int = 512,
        attn_dim: int = 256,
        num_labels: int = 8,
        dropout: float = 0.2,
        fdr_target: float = 0.35,
        fdr_weight: float = 0.2,
        tail_beta: float = 0.75,
        head_label_indices: list[int] | tuple[int, ...] | None = None,
    ) -> None:
        super().__init__(
            backbone_name=backbone_name,
            pretrained=pretrained,
            freeze_stages=freeze_stages,
            feature_dim=feature_dim,
            attn_dim=attn_dim,
            num_labels=num_labels,
            dropout=dropout,
        )
        self.fdr_target = float(fdr_target)
        self.fdr_weight = float(fdr_weight)
        self.tail_beta = float(tail_beta)
        self.head_label_indices = resolve_head_label_indices(num_labels, head_label_indices)
        self.register_buffer("tail_mask", build_tail_mask(num_labels, self.head_label_indices), persistent=False)
        self.threshold = PerLabelThresholdModule(num_labels=num_labels)

    def get_label_thresholds(self) -> torch.Tensor:
        return self.threshold.thresholds.detach()

    def update_label_thresholds_from_validation(self, y_true: np.ndarray, y_prob: np.ndarray) -> np.ndarray:
        beta_by_label = [1.0] * self.num_labels
        for label_index in range(self.num_labels):
            if bool(self.tail_mask[label_index].item()):
                beta_by_label[label_index] = self.tail_beta
        thresholds = search_best_thresholds(y_true, y_prob, beta_by_label=beta_by_label)
        self.threshold.set_thresholds(thresholds)
        return thresholds

    def forward(self, images: torch.Tensor, mask: torch.Tensor) -> dict[str, torch.Tensor]:
        outputs = super().forward(images, mask)
        outputs["label_thresholds"] = self.threshold.thresholds
        return outputs

    def compute_loss(
        self,
        outputs: dict[str, torch.Tensor],
        labels: torch.Tensor,
        criterion,
        current_epoch: float = 0.0,
        train_mode: bool = False,
    ) -> torch.Tensor:
        del current_epoch, train_mode
        logits = outputs["logits"]
        base_loss = criterion(logits, labels)
        if isinstance(base_loss, torch.Tensor) and base_loss.ndim > 0:
            base_loss = base_loss.mean()

        probs = torch.sigmoid(logits)
        expected_fp = (probs * (1.0 - labels)).sum(dim=0)
        expected_pred = probs.sum(dim=0).clamp_min(1e-6)
        per_label_fdr = expected_fp / expected_pred
        tail_fdr = per_label_fdr[self.tail_mask]
        fdr_penalty = torch.relu(tail_fdr - self.fdr_target).pow(2).mean() if tail_fdr.numel() > 0 else logits.new_zeros(())

        if self.head_label_indices:
            head_index = torch.tensor(self.head_label_indices, device=logits.device)
            head_anchor = F.binary_cross_entropy_with_logits(logits[:, head_index], labels[:, head_index])
        else:
            head_anchor = logits.new_zeros(())

        return base_loss + self.fdr_weight * fdr_penalty + 0.15 * head_anchor


__all__ = ["FDRMILModel"]
