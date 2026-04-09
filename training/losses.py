from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class FocalBCEWithLogitsLoss(nn.Module):
    def __init__(self, alpha: float = 0.25, gamma: float = 2.0, reduction: str = "mean") -> None:
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        bce = F.binary_cross_entropy_with_logits(logits, targets, reduction="none")
        prob = torch.sigmoid(logits)
        p_t = prob * targets + (1.0 - prob) * (1.0 - targets)
        alpha_factor = self.alpha * targets + (1.0 - self.alpha) * (1.0 - targets)
        modulating_factor = (1.0 - p_t).pow(self.gamma)
        loss = alpha_factor * modulating_factor * bce
        if self.reduction == "sum":
            return loss.sum()
        if self.reduction == "none":
            return loss
        return loss.mean()


class AsymmetricLossMultiLabel(nn.Module):
    """适用于多标签不平衡场景。"""

    def __init__(
        self,
        gamma_pos: float = 1.0,
        gamma_neg: float = 4.0,
        clip: float = 0.05,
        eps: float = 1e-8,
    ) -> None:
        super().__init__()
        self.gamma_pos = gamma_pos
        self.gamma_neg = gamma_neg
        self.clip = clip
        self.eps = eps

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        xs_pos = torch.sigmoid(logits)
        xs_neg = 1.0 - xs_pos

        if self.clip > 0:
            xs_neg = (xs_neg + self.clip).clamp(max=1.0)

        loss_pos = targets * torch.log(xs_pos.clamp(min=self.eps))
        loss_neg = (1.0 - targets) * torch.log(xs_neg.clamp(min=self.eps))
        loss = loss_pos + loss_neg

        pt = xs_pos * targets + xs_neg * (1.0 - targets)
        gamma = self.gamma_pos * targets + self.gamma_neg * (1.0 - targets)
        focal_weight = (1.0 - pt).pow(gamma)
        return (-loss * focal_weight).mean()


def build_multilabel_criterion(loss_name: str, pos_weight: torch.Tensor | None = None) -> nn.Module:
    name = loss_name.lower()
    if name in {"bce", "bcewithlogits", "weighted_bce"}:
        return nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    if name in {"focal", "focal_bce"}:
        return FocalBCEWithLogitsLoss()
    if name in {"asl", "asymmetric", "asymmetric_loss"}:
        return AsymmetricLossMultiLabel()
    raise ValueError(f"未知多标签损失: {loss_name}")


def build_binary_criterion(loss_name: str, pos_weight: torch.Tensor | None = None) -> nn.Module:
    return build_multilabel_criterion(loss_name=loss_name, pos_weight=pos_weight)
