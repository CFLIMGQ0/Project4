"""
RAL MIL

参考来源：
- Park et al., ICCVW 2023
- Robust Asymmetric Loss for Multi-Label Long-Tailed Learning

项目改写说明：
- 用非对称多标签损失作为主项
- 加入 polynomial 修正项与 hill-style 分离正则，降低超参敏感性
"""

from __future__ import annotations

import torch
import torch.nn.functional as F

from exp_2.common import Exp2AttentionMILBase, resolve_class_prior


class RALMILModel(Exp2AttentionMILBase):
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
        loss_gamma_pos: float = 1.0,
        loss_gamma_neg: float = 4.0,
        loss_clip: float = 0.05,
        poly_epsilon: float = 1.0,
        hill_weight: float = 0.05,
        hill_margin: float = 0.2,
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
        pos_weight = (1.0 / prior.clamp_min(1e-6).sqrt()).clamp(max=8.0)
        pos_weight = pos_weight / pos_weight.mean().clamp_min(1e-6)
        self.register_buffer("pos_weight", pos_weight)
        self.loss_gamma_pos = float(loss_gamma_pos)
        self.loss_gamma_neg = float(loss_gamma_neg)
        self.loss_clip = float(loss_clip)
        self.poly_epsilon = float(poly_epsilon)
        self.hill_weight = float(hill_weight)
        self.hill_margin = float(hill_margin)

    def _hill_regularizer(self, logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        pos_mask = labels > 0.5
        neg_mask = labels <= 0.5
        pos_count = pos_mask.sum(dim=0)
        neg_count = neg_mask.sum(dim=0)
        valid = (pos_count > 0) & (neg_count > 0)
        if not valid.any():
            return logits.new_zeros(())

        pos_mean = (logits * pos_mask).sum(dim=0) / pos_count.clamp_min(1.0)
        neg_mean = (logits * neg_mask).sum(dim=0) / neg_count.clamp_min(1.0)
        margin_gap = pos_mean[valid] - neg_mean[valid]
        return torch.relu(self.hill_margin - margin_gap).mean()

    def compute_loss(
        self,
        outputs: dict[str, torch.Tensor],
        labels: torch.Tensor,
        criterion,
        current_epoch: float = 0.0,
        train_mode: bool = False,
    ) -> torch.Tensor:
        del criterion, current_epoch, train_mode
        logits = outputs["logits"]
        probs = torch.sigmoid(logits)
        neg_probs = 1.0 - probs
        if self.loss_clip > 0:
            neg_probs = (neg_probs + self.loss_clip).clamp(max=1.0)

        pos_term = labels * torch.log(probs.clamp(min=1e-6))
        neg_term = (1.0 - labels) * torch.log(neg_probs.clamp(min=1e-6))
        pt = probs * labels + (1.0 - probs) * (1.0 - labels)
        gamma = self.loss_gamma_pos * labels + self.loss_gamma_neg * (1.0 - labels)
        focal = (1.0 - pt).pow(gamma)
        poly = self.poly_epsilon * (1.0 - pt)
        weights = labels * self.pos_weight.unsqueeze(0) + (1.0 - labels)

        base_loss = -weights * (pos_term + neg_term) * focal + weights * poly
        hill_loss = self._hill_regularizer(logits, labels)
        return base_loss.mean() + self.hill_weight * hill_loss


__all__ = ["RALMILModel"]
