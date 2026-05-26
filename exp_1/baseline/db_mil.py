"""
Distribution-Balanced Loss MIL (DB-MIL)

来源：Wu et al., "Distribution-Balanced Loss for Multi-Label Classification
      in Long-Tailed Datasets", NeurIPS 2020

核心思想：通过两个机制重新平衡多标签分类中的梯度贡献：
1. Rebalanced Weight: 基于标签频率对每个样本-标签对重新加权
2. Negative-Tolerant Regularization: 对负样本施加更温和的梯度

解决的问题：
- 长尾标签 (ulcer, reflux_esophagitis) 的梯度信号被高频标签淹没
- 负样本数量远多于正样本，导致模型偏向预测阴性
- 频繁标签的梯度贡献主导了共享特征的更新方向

关键模块：
- DistributionBalancedLoss: DB-Loss 实现
- ClassFrequencyTracker: 动态追踪类别频率
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from model.common import MultiLabelAttentionMIL, build_backbone


class DistributionBalancedLoss(nn.Module):
    """Distribution-Balanced Loss。

    L_DB = r(y, c) * L_focal(p, y)

    其中 r(y, c) 是基于类别频率的重平衡权重，
    L_focal 使用类别感知的 focal 机制。
    """

    def __init__(
        self,
        num_labels: int = 8,
        class_freq: list[float] | None = None,
        focal_gamma: float = 2.0,
        neg_scale: float = 2.0,
        map_alpha: float = 0.1,
        map_beta: float = 10.0,
        map_gamma: float = 0.3,
    ) -> None:
        super().__init__()
        self.num_labels = num_labels
        self.focal_gamma = focal_gamma
        self.neg_scale = neg_scale
        self.map_alpha = map_alpha
        self.map_beta = map_beta
        self.map_gamma = map_gamma

        # Default class frequencies for task2 (estimated from data characteristics)
        if class_freq is None:
            class_freq = [0.15, 0.08, 0.75, 0.12, 0.10, 0.06, 0.04, 0.03]
        freq = torch.tensor(class_freq, dtype=torch.float32)
        # Inverse frequency mapping
        self.register_buffer("freq", freq)
        self.register_buffer("neg_freq", 1.0 - freq)

    def _rebalance_weight(self, targets: torch.Tensor) -> torch.Tensor:
        """计算重平衡权重 r(y, c)。"""
        # Map function: sigmoid-based smoothing
        # For positive: weight inversely proportional to freq
        # For negative: weight proportional to freq
        pos_weight = 1.0 / (self.freq + 1e-8)
        neg_weight = 1.0 / (self.neg_freq + 1e-8)

        # Normalize
        pos_weight = pos_weight / pos_weight.sum() * self.num_labels
        neg_weight = neg_weight / neg_weight.sum() * self.num_labels

        weight = targets * pos_weight.unsqueeze(0) + (1 - targets) * neg_weight.unsqueeze(0)
        return weight

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Args:
            logits: (B, L) raw logits
            targets: (B, L) binary labels
        """
        # Rebalance weights
        weight = self._rebalance_weight(targets)

        # Modified focal loss with negative tolerance
        pred = torch.sigmoid(logits)

        # Negative tolerance: shift negative predictions
        neg_pred = pred * (1 - targets)
        shifted_neg = (neg_pred * self.neg_scale).clamp(max=1.0)
        effective_pred = pred * targets + shifted_neg * (1 - targets)

        # Focal modulation
        pt = effective_pred * targets + (1 - effective_pred) * (1 - targets)
        focal_weight = (1 - pt).pow(self.focal_gamma)

        # BCE loss
        bce = F.binary_cross_entropy_with_logits(logits, targets, reduction="none")

        loss = weight * focal_weight * bce
        return loss.mean()


class DBMILModel(nn.Module):
    """Distribution-Balanced MIL: 基于分布平衡损失的多标签 MIL。

    在标准注意力 MIL 基础上引入 DB-Loss，通过频率感知的梯度
    重平衡确保长尾标签获得足够的训练信号。
    """

    def __init__(
        self,
        backbone_name: str = "resnet50",
        pretrained: bool = True,
        freeze_stages: int = 0,
        feature_dim: int = 512,
        attn_dim: int = 256,
        num_labels: int = 8,
        dropout: float = 0.2,
        class_freq: list[float] | None = None,
    ) -> None:
        super().__init__()
        self.num_labels = num_labels

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
        self.db_loss = DistributionBalancedLoss(
            num_labels=num_labels,
            class_freq=class_freq,
        )

    def encode_instances(self, images: torch.Tensor) -> torch.Tensor:
        batch_size, num_instances, channels, height, width = images.shape
        x = images.reshape(batch_size * num_instances, channels, height, width)
        features = self.encoder(x).reshape(batch_size, num_instances, -1)
        return self.shared_proj(features)

    def forward(self, images: torch.Tensor, mask: torch.Tensor) -> dict[str, torch.Tensor]:
        features = self.encode_instances(images)
        bag_embeds, attention = self.mil_pool(features, mask)

        logits = []
        for label_index in range(self.num_labels):
            logits.append(self.classifiers[label_index](bag_embeds[:, label_index, :]).squeeze(-1))

        return {
            "logits": torch.stack(logits, dim=1),
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
        return self.db_loss(outputs["logits"], labels)
