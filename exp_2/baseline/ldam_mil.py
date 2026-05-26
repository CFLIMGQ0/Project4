"""
LDAM MIL

参考来源：
- Cao et al., NeurIPS 2019
- Learning Imbalanced Datasets with Label-Distribution-Aware Margin Loss

项目改写说明：
- 原论文针对单标签 margin，这里把 margin 改为逐标签的正类 margin
- 同时保留 DRW（deferred re-weighting）思想，避免一开始就把尾标签权重拉得过猛
"""

from __future__ import annotations

import torch
import torch.nn.functional as F

from exp_2.common import Exp2AttentionMILBase, resolve_class_prior


class LDAMMILModel(Exp2AttentionMILBase):
    def __init__(
        self,
        backbone_name: str = "resnet50",
        pretrained: bool = True,
        freeze_stages: int = 0,
        feature_dim: int = 512,
        attn_dim: int = 256,
        num_labels: int = 8,
        dropout: float = 0.2,
        class_prior: list[float] | tuple[float, ...] | None = None,
        max_margin: float = 0.5,
        drw_start_epoch: int = 10,
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
        prior = torch.tensor(resolve_class_prior(num_labels, class_prior), dtype=torch.float32)
        raw_margin = 1.0 / prior.clamp_min(1e-6).pow(0.25)
        margin = raw_margin / raw_margin.max() * float(max_margin)
        drw_weight = (1.0 / prior.clamp_min(1e-6).sqrt()).clamp(max=12.0)
        drw_weight = drw_weight / drw_weight.mean().clamp_min(1e-6)
        self.register_buffer("margin", margin)
        self.register_buffer("drw_weight", drw_weight)
        self.drw_start_epoch = int(drw_start_epoch)

    def compute_loss(
        self,
        outputs: dict[str, torch.Tensor],
        labels: torch.Tensor,
        criterion,
        current_epoch: float = 0.0,
        train_mode: bool = False,
    ) -> torch.Tensor:
        del criterion, train_mode
        logits = outputs["logits"]
        adjusted_logits = logits - labels * self.margin.unsqueeze(0)
        bce = F.binary_cross_entropy_with_logits(adjusted_logits, labels, reduction="none")

        if float(current_epoch) >= float(self.drw_start_epoch):
            weights = labels * self.drw_weight.unsqueeze(0) + (1.0 - labels)
            bce = bce * weights

        return bce.mean()


__all__ = ["LDAMMILModel"]
