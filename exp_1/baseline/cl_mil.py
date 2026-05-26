"""
Curriculum Learning MIL (CL-MIL)

来源：
- Bengio et al., "Curriculum Learning", ICML 2009
- Kumar et al., "Self-Paced Learning with Diversity", NeurIPS 2010
- 长尾分类中的课程学习变体

核心思想：采用标签难度感知的课程学习策略：
1. 训练早期主要学习"容易"的标签（强标签）
2. 随着训练进行，逐步增加"困难"标签（弱标签）的权重
3. 使用自适应节奏控制器根据标签级别的学习进度调整课程

解决的问题：
- 弱标签在训练初期被强标签的梯度信号干扰
- 模型来不及学会强标签的稳定特征就被弱标签噪声干扰
- 训练后期强标签过拟合而弱标签欠拟合

关键模块：
- CurriculumScheduler: 标签难度感知的课程调度器
- SelfPacedWeighter: 基于当前损失的自适应样本加权
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from model.common import MultiLabelAttentionMIL, build_backbone


class CurriculumScheduler(nn.Module):
    """标签难度感知的课程调度器。

    维护每个标签的"困难度"分数，根据训练进度控制每个标签
    的损失权重。使用 sigmoid 调度：

    w_l(t) = sigma((t - t_start_l) * speed_l)

    其中 t_start_l 是标签 l 开始全力训练的时间点，
    speed_l 控制过渡的平滑程度。
    """

    def __init__(
        self,
        num_labels: int = 8,
        label_difficulty: list[float] | None = None,
        total_epochs: int = 30,
        warmup_fraction: float = 0.2,
    ) -> None:
        super().__init__()
        self.num_labels = num_labels
        self.total_epochs = total_epochs

        if label_difficulty is None:
            # Estimated difficulty: 0=easy, 1=hardest
            # [smt, neoplasm, gastritis, active, atrophy, polyp, reflux, ulcer]
            label_difficulty = [0.1, 0.15, 0.05, 0.6, 0.4, 0.65, 0.8, 0.9]

        difficulty = torch.tensor(label_difficulty, dtype=torch.float32)
        # Start epoch: harder labels start later
        start_epoch = difficulty * total_epochs * (1 - warmup_fraction)
        self.register_buffer("start_epoch", start_epoch)
        self.register_buffer("difficulty", difficulty)

        # Speed of curriculum transition
        self.speed = nn.Parameter(torch.full((num_labels,), 5.0))

    def forward(self, current_epoch: float) -> torch.Tensor:
        """返回每个标签的当前课程权重。"""
        t = torch.tensor(current_epoch, device=self.start_epoch.device)
        # Sigmoid schedule
        weights = torch.sigmoid(self.speed * (t - self.start_epoch))
        # Ensure minimum weight (never completely zero)
        weights = weights.clamp(min=0.1)
        return weights


class SelfPacedWeighter(nn.Module):
    """基于当前损失的自适应样本加权。

    对于每个样本-标签对，根据其当前损失值决定权重：
    - 低损失（容易）：高权重（早期学习）
    - 高损失（困难）：权重随训练进度增加

    v(l, lambda) = max(0, 1 - l / lambda)

    lambda 从小到大递增（先学易后学难）。
    """

    def __init__(self, initial_lambda: float = 0.5, lambda_growth: float = 0.05) -> None:
        super().__init__()
        self.lambda_val = initial_lambda
        self.lambda_growth = lambda_growth

    def update_lambda(self) -> None:
        self.lambda_val += self.lambda_growth

    def lambda_at(self, current_epoch: float) -> float:
        return self.lambda_val + max(0.0, float(current_epoch)) * self.lambda_growth

    def forward(self, per_sample_loss: torch.Tensor, current_epoch: float | None = None) -> torch.Tensor:
        """
        Args:
            per_sample_loss: (B, L) per-sample per-label loss
        Returns:
            weights: (B, L) self-paced weights
        """
        lambda_val = self.lambda_val if current_epoch is None else self.lambda_at(current_epoch)
        weights = (1.0 - per_sample_loss / max(lambda_val, 1e-6)).clamp(min=0.0)
        return weights


class CLMILModel(nn.Module):
    """Curriculum Learning MIL: 基于课程学习的多标签 MIL。

    结合标签难度课程调度和自适应样本加权，让模型先稳固学习
    强标签，再逐步攻克弱标签。
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
        label_difficulty: list[float] | None = None,
        total_epochs: int = 30,
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
        self.curriculum = CurriculumScheduler(
            num_labels=num_labels,
            label_difficulty=label_difficulty,
            total_epochs=total_epochs,
        )
        self.self_paced = SelfPacedWeighter()

    def encode_instances(self, images: torch.Tensor) -> torch.Tensor:
        batch_size, num_instances, channels, height, width = images.shape
        x = images.reshape(batch_size * num_instances, channels, height, width)
        features = self.encoder(x).reshape(batch_size, num_instances, -1)
        return self.shared_proj(features)

    def forward(
        self,
        images: torch.Tensor,
        mask: torch.Tensor,
        current_epoch: float = 0.0,
    ) -> dict[str, torch.Tensor]:
        features = self.encode_instances(images)
        bag_embeds, attention = self.mil_pool(features, mask)

        logits = []
        for label_index in range(self.num_labels):
            logits.append(self.classifiers[label_index](bag_embeds[:, label_index, :]).squeeze(-1))

        logits_tensor = torch.stack(logits, dim=1)

        result = {
            "logits": logits_tensor,
            "attention": attention,
            "instance_features": features,
        }

        if self.training:
            curriculum_weights = self.curriculum(current_epoch)
            result["curriculum_weights"] = curriculum_weights

        return result

    def compute_loss(
        self,
        outputs: dict[str, torch.Tensor],
        labels: torch.Tensor,
        criterion,
        current_epoch: float = 0.0,
        train_mode: bool = False,
    ) -> torch.Tensor:
        del criterion
        per_label_loss = F.binary_cross_entropy_with_logits(outputs["logits"], labels, reduction="none")
        curriculum_weights = self.curriculum(current_epoch).unsqueeze(0)
        self_paced_weights = self.self_paced(per_label_loss.detach(), current_epoch=current_epoch)
        weights = (curriculum_weights * self_paced_weights).clamp(min=0.05, max=5.0)
        weights = weights / weights.mean().clamp_min(1e-6)
        if not train_mode:
            weights = curriculum_weights / curriculum_weights.mean().clamp_min(1e-6)
        return (weights * per_label_loss).mean()
