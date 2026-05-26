"""
CTURS MIL

参考来源：
- Guo & Wang, CVPR 2021
- Long-Tailed Multi-Label Visual Recognition by Collaborative Training on Uniform and Re-Balanced Samplings

项目改写说明：
- 原论文强调“均匀采样分支 + 重平衡采样分支 + 跨分支一致性”
- 当前项目训练管线不额外切换 dataloader，因此这里采用“共享特征 + 双头双损失 + 一致性”的轻量改写
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from exp_2.common import Exp2AttentionMILBase, resolve_class_prior


class CTURSMILModel(Exp2AttentionMILBase):
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
        logit_adj_tau: float = 1.0,
        consistency_weight: float = 0.1,
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
        self.register_buffer("class_prior", prior)
        self.register_buffer("rebalance_weight", (1.0 / prior.clamp_min(1e-4)).clamp(max=12.0))
        self.logit_adj_tau = float(logit_adj_tau)
        self.consistency_weight = float(consistency_weight)
        self.uniform_classifiers = self.classifiers
        self.rebalanced_classifiers = nn.ModuleList([nn.Linear(feature_dim, 1) for _ in range(num_labels)])

    def forward(self, images: torch.Tensor, mask: torch.Tensor) -> dict[str, torch.Tensor]:
        features = self.encode_instances(images)
        bag_embeds, attention = self.mil_pool(features, mask)

        uniform_logits = torch.stack(
            [head(bag_embeds[:, label_index, :]).squeeze(-1) for label_index, head in enumerate(self.uniform_classifiers)],
            dim=1,
        )
        rebalanced_raw = torch.stack(
            [head(bag_embeds[:, label_index, :]).squeeze(-1) for label_index, head in enumerate(self.rebalanced_classifiers)],
            dim=1,
        )
        compensation = -self.logit_adj_tau * torch.log(self.class_prior.clamp_min(1e-6)).unsqueeze(0)
        rebalanced_logits = rebalanced_raw + compensation
        final_logits = 0.5 * (uniform_logits + rebalanced_logits)

        return {
            "logits": final_logits,
            "uniform_logits": uniform_logits,
            "rebalanced_logits": rebalanced_logits,
            "attention": attention,
            "instance_features": features,
        }

    def compute_loss(
        self,
        outputs: dict[str, torch.Tensor],
        labels: torch.Tensor,
        criterion,
        current_epoch: float = 0.0,
        train_mode: bool = False,
    ) -> torch.Tensor:
        del criterion, current_epoch, train_mode
        uniform_loss = F.binary_cross_entropy_with_logits(outputs["uniform_logits"], labels)
        rebalance_bce = F.binary_cross_entropy_with_logits(outputs["rebalanced_logits"], labels, reduction="none")
        rebalance_weights = labels * self.rebalance_weight.unsqueeze(0) + (1.0 - labels)
        rebalance_loss = (rebalance_bce * rebalance_weights).mean()
        consistency_loss = F.mse_loss(
            torch.sigmoid(outputs["uniform_logits"]),
            torch.sigmoid(outputs["rebalanced_logits"]),
        )
        return uniform_loss + rebalance_loss + self.consistency_weight * consistency_loss


__all__ = ["CTURSMILModel"]
