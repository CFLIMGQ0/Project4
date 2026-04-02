from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class DemoFocalBCEWithLogitsLoss(nn.Module):
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
        if self.reduction == "mean":
            return loss.mean()
        if self.reduction == "sum":
            return loss.sum()
        return loss


class DemoAsymmetricLossMultiLabel(nn.Module):
    """Asymmetric Loss for multi-label classification."""

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

        loss = -loss * focal_weight
        return loss.mean()


def build_multilabel_criterion(
    loss_name: str,
    pos_weight: torch.Tensor | None = None,
    focal_alpha: float = 0.25,
    focal_gamma: float = 2.0,
) -> nn.Module:
    name = loss_name.lower()
    if name in {"bce", "bcewithlogits", "weighted_bce"}:
        return nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    if name in {"focal", "focal_bce"}:
        return DemoFocalBCEWithLogitsLoss(alpha=focal_alpha, gamma=focal_gamma)
    if name in {"asl", "asymmetric", "asymmetric_loss"}:
        return DemoAsymmetricLossMultiLabel()
    raise ValueError(f"未知多标签损失: {loss_name}")


def build_binary_or_multiclass_criterion(
    num_classes: int,
    loss_name: str,
    pos_weight: torch.Tensor | None = None,
    focal_alpha: float = 0.25,
    focal_gamma: float = 2.0,
) -> nn.Module:
    if num_classes == 2:
        return build_multilabel_criterion(
            loss_name=loss_name,
            pos_weight=pos_weight,
            focal_alpha=focal_alpha,
            focal_gamma=focal_gamma,
        )

    # 多分类场景默认 CrossEntropy，保留扩展空间。
    return nn.CrossEntropyLoss()


def expert_balance_loss(expert_weights: torch.Tensor) -> torch.Tensor:
    """鼓励专家使用分布更均衡，缓解 routing collapse。"""
    # expert_weights: [B, E]
    mean_usage = expert_weights.mean(dim=0)
    mean_usage = mean_usage / mean_usage.sum().clamp_min(1e-12)
    uniform = torch.full_like(mean_usage, 1.0 / mean_usage.numel())
    loss = F.kl_div(mean_usage.log(), uniform, reduction="batchmean")
    return loss


def consistency_loss(prob_a: torch.Tensor, prob_b: torch.Tensor) -> torch.Tensor:
    return F.mse_loss(prob_a, prob_b)


def prototype_pull_push_loss(
    prototype_scores: torch.Tensor,
    labels: torch.Tensor,
    neg_margin: float = 0.2,
) -> torch.Tensor:
    """
    prototype_scores: [B, L], 值域通常在 [-1, 1]
    labels: [B, L]
    """
    pos_term = (1.0 - prototype_scores) * labels
    neg_term = F.relu(prototype_scores - neg_margin) * (1.0 - labels)
    return (pos_term + neg_term).mean()


def prototype_binary_contrastive_loss(
    normal_sim: torch.Tensor,
    polyp_sim: torch.Tensor,
    binary_labels: torch.Tensor,
    margin: float = 0.2,
) -> torch.Tensor:
    """
    normal_sim/polyp_sim: [B]
    binary_labels: [B] in {0,1}
    """
    y = binary_labels.float()
    pos_loss = F.relu(margin - (polyp_sim - normal_sim)) * y
    neg_loss = F.relu(margin - (normal_sim - polyp_sim)) * (1.0 - y)
    return (pos_loss + neg_loss).mean()


def hard_negative_suppression(
    instance_scores: torch.Tensor,
    binary_labels: torch.Tensor,
    mask: torch.Tensor,
    margin: float = 0.35,
) -> torch.Tensor:
    """
    仅对正常样本（label=0）抑制高响应实例。
    instance_scores: [B, N] (sigmoid score)
    """
    neg_mask = (binary_labels == 0).float().unsqueeze(-1)
    penalty = F.relu(instance_scores - margin) * mask.float() * neg_mask
    denom = (mask.float() * neg_mask).sum().clamp_min(1.0)
    return penalty.sum() / denom
