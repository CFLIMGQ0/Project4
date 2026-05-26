"""
Cost-Sensitive Meta-Learning MIL (CSML-MIL)

来源：Shu et al., "Meta-Weight-Net: Learning an Explicit Mapping
      For Sample Weighting", NeurIPS 2019

核心思想：使用一个轻量的元网络 (meta-net) 学习每个样本-标签对的
         最优训练权重。元网络以当前损失为输入，输出样本权重，
         通过在平衡验证集上优化元目标来训练。

解决的问题：
- 手动设计的 class weight 或 focal loss 参数需要大量调优
- 不同训练阶段同一标签的最优权重不同
- 需要自动适应不同样本和标签的训练需求

关键模块：
- MetaWeightNet: 损失到权重的映射网络
- LabelConditionedWeighter: 标签感知的样本权重生成器
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from model.common import MultiLabelAttentionMIL, build_backbone


class MetaWeightNet(nn.Module):
    """元权重网络：将样本-标签的损失值映射为训练权重。

    网络结构：loss_value -> MLP -> weight
    标签条件化：每个标签有独立的权重映射路径。

    元学习目标：在平衡验证集上最小化损失。
    """

    def __init__(self, num_labels: int = 8, hidden_dim: int = 64) -> None:
        super().__init__()
        self.num_labels = num_labels
        # Shared feature extractor
        self.shared = nn.Sequential(
            nn.Linear(1, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(inplace=True),
        )
        # Per-label weight heads
        self.label_heads = nn.ModuleList([
            nn.Sequential(
                nn.Linear(hidden_dim, 1),
                nn.Sigmoid(),  # Weight in (0, 1)
            ) for _ in range(num_labels)
        ])

    def forward(self, per_label_loss: torch.Tensor) -> torch.Tensor:
        """
        Args:
            per_label_loss: (B, L) per-sample per-label loss values
        Returns:
            weights: (B, L) learned sample weights
        """
        batch_size = per_label_loss.shape[0]
        all_weights = []

        for l in range(self.num_labels):
            loss_input = per_label_loss[:, l:l+1]  # (B, 1)
            hidden = self.shared(loss_input)  # (B, hidden_dim)
            weight = self.label_heads[l](hidden).squeeze(-1)  # (B,)
            all_weights.append(weight)

        return torch.stack(all_weights, dim=1)  # (B, L)


class LabelConditionedWeighter(nn.Module):
    """标签条件化的样本权重生成器。

    综合考虑：
    1. 样本-标签的当前损失大小
    2. 标签的全局难度（动态追踪）
    3. 标签的类别频率先验
    """

    def __init__(self, num_labels: int = 8, feature_dim: int = 512, hidden_dim: int = 64) -> None:
        super().__init__()
        self.num_labels = num_labels
        # Context-aware weighter: uses both loss value and bag embedding
        self.context_net = nn.Sequential(
            nn.Linear(feature_dim + 1, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, 1),
            nn.Sigmoid(),
        )
        # Running average of per-label loss (for difficulty tracking)
        self.register_buffer("running_loss", torch.zeros(num_labels))
        self.register_buffer("running_count", torch.zeros(num_labels))

    def update_running_stats(self, per_label_loss: torch.Tensor) -> None:
        with torch.no_grad():
            self.running_loss += per_label_loss.sum(dim=0)
            self.running_count += per_label_loss.shape[0]

    def get_difficulty_weights(self) -> torch.Tensor:
        avg_loss = self.running_loss / self.running_count.clamp_min(1)
        # Higher loss = harder = higher weight
        weights = avg_loss / avg_loss.sum().clamp_min(1e-8) * self.num_labels
        return weights.clamp(min=0.5, max=3.0)

    def forward(
        self,
        bag_embeds: torch.Tensor,
        per_label_loss: torch.Tensor,
    ) -> torch.Tensor:
        """
        Args:
            bag_embeds: (B, L, D)
            per_label_loss: (B, L)
        Returns:
            weights: (B, L)
        """
        all_weights = []
        for l in range(self.num_labels):
            ctx_input = torch.cat([
                bag_embeds[:, l, :],
                per_label_loss[:, l:l+1],
            ], dim=-1)
            w = self.context_net(ctx_input).squeeze(-1)
            all_weights.append(w)

        base_weights = torch.stack(all_weights, dim=1)

        # Scale by difficulty
        if self.training:
            self.update_running_stats(per_label_loss.detach())
        diff_weights = self.get_difficulty_weights()

        return base_weights * diff_weights.unsqueeze(0)


class CSMLMILModel(nn.Module):
    """Cost-Sensitive Meta-Learning MIL: 基于元学习的代价敏感 MIL。

    使用元权重网络自动学习每个样本-标签对的最优训练权重，
    无需手动设计 class weight 或调节 focal loss 参数。
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
        self.meta_weighter = MetaWeightNet(num_labels=num_labels)
        self.label_weighter = LabelConditionedWeighter(
            num_labels=num_labels,
            feature_dim=feature_dim,
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

        logits_tensor = torch.stack(logits, dim=1)

        result = {
            "logits": logits_tensor,
            "attention": attention,
            "instance_features": features,
            "bag_embeds": bag_embeds,
        }

        return result

    def compute_loss(
        self,
        outputs: dict[str, torch.Tensor],
        labels: torch.Tensor,
        criterion,
        current_epoch: float = 0.0,
        train_mode: bool = False,
    ) -> torch.Tensor:
        del criterion, current_epoch
        per_label_loss = F.binary_cross_entropy_with_logits(outputs["logits"], labels, reduction="none")
        if not train_mode:
            return per_label_loss.mean()

        meta_weights = self.meta_weighter(per_label_loss.detach())
        label_weights = self.label_weighter(outputs["bag_embeds"].detach(), per_label_loss.detach())
        weights = (meta_weights * label_weights).clamp(min=0.05, max=5.0)
        weights = weights / weights.mean().clamp_min(1e-6)
        return (weights * per_label_loss).mean()
