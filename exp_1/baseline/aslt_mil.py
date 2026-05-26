"""
Asymmetric Loss with Label-wise Thresholding MIL (ASLT-MIL)

来源：
- Ridnik et al., "Asymmetric Loss For Multi-Label Classification", ICCV 2021
- 阈值移动法来自长尾多标签分类文献

核心思想：在标准 ASL 的基础上：
1. 每个标签使用独立的自适应阈值（而非统一的 0.5）
2. 阈值通过验证集上的 F1 最优化自动搜索
3. 训练时引入 logit adjustment 补偿类别先验

解决的问题：
- 统一阈值 0.5 对长尾标签不公平（正样本少，模型输出概率偏低）
- ASL 虽缓解不平衡，但阈值固定仍限制弱标签 recall
- 弱标签需要更低的阈值才能获得合理的 recall

关键模块：
- LogitAdjustment: 基于类别先验的 logit 偏移
- PerLabelThreshold: 可学习或搜索的每标签阈值
"""

from __future__ import annotations

import math
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F

from model.common import MultiLabelAttentionMIL, build_backbone


class LogitAdjustmentLayer(nn.Module):
    """Logit Adjustment: 基于类别先验频率偏移 logits。

    logit_adjusted = logit + tau * log(pi_y)

    其中 pi_y 是标签 y 的正样本先验概率。
    这使得稀有标签的 logit 被"提升"，频繁标签的 logit 被"抑制"。
    """

    def __init__(self, num_labels: int, class_prior: list[float] | None = None, tau: float = 1.0) -> None:
        super().__init__()
        if class_prior is None:
            class_prior = [0.15, 0.08, 0.75, 0.12, 0.10, 0.06, 0.04, 0.03]
        prior = torch.tensor(class_prior, dtype=torch.float32)
        log_prior = torch.log(prior.clamp(min=1e-8))
        self.register_buffer("log_prior", log_prior)
        self.tau = nn.Parameter(torch.tensor(tau))

    def forward(self, logits: torch.Tensor) -> torch.Tensor:
        return logits + self.tau * self.log_prior.unsqueeze(0)


class PerLabelThresholdModule(nn.Module):
    """可学习的每标签阈值模块。

    每个标签有独立的阈值参数，用 sigmoid 约束在 (0, 1) 范围内。
    训练时可以端到端优化，或在验证集上搜索后固定。
    """

    def __init__(self, num_labels: int, init_threshold: float = 0.5) -> None:
        super().__init__()
        # Initialize in logit space
        init_logit = math.log(init_threshold / (1 - init_threshold))
        self.threshold_logits = nn.Parameter(torch.full((num_labels,), init_logit))

    @property
    def thresholds(self) -> torch.Tensor:
        return torch.sigmoid(self.threshold_logits)

    def forward(self, probs: torch.Tensor) -> torch.Tensor:
        """返回阈值化后的预测。"""
        return (probs > self.thresholds.unsqueeze(0)).float()

    def set_thresholds(self, thresholds: list[float]) -> None:
        """手动设置阈值（例如从验证集搜索得到的最优值）。"""
        with torch.no_grad():
            t = torch.tensor(thresholds, dtype=torch.float32).clamp(min=1e-4, max=1.0 - 1e-4)
            self.threshold_logits.copy_(torch.log(t / (1 - t)))


class ASLTMILModel(nn.Module):
    """Asymmetric Loss + Label-wise Threshold MIL。

    在 ASL 基础上加入 logit adjustment 和每标签自适应阈值，
    专门针对长尾多标签场景优化。
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
        class_prior: list[float] | None = None,
        logit_adj_tau: float = 1.0,
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
        self.logit_adj = LogitAdjustmentLayer(
            num_labels=num_labels,
            class_prior=class_prior,
            tau=logit_adj_tau,
        )
        self.threshold = PerLabelThresholdModule(num_labels=num_labels)

    def encode_instances(self, images: torch.Tensor) -> torch.Tensor:
        batch_size, num_instances, channels, height, width = images.shape
        x = images.reshape(batch_size * num_instances, channels, height, width)
        features = self.encoder(x).reshape(batch_size, num_instances, -1)
        return self.shared_proj(features)

    def forward(self, images: torch.Tensor, mask: torch.Tensor) -> dict[str, torch.Tensor]:
        features = self.encode_instances(images)
        bag_embeds, attention = self.mil_pool(features, mask)

        raw_logits = []
        for label_index in range(self.num_labels):
            raw_logits.append(self.classifiers[label_index](bag_embeds[:, label_index, :]).squeeze(-1))
        raw_logits_tensor = torch.stack(raw_logits, dim=1)

        # Apply logit adjustment
        adjusted_logits = self.logit_adj(raw_logits_tensor)

        return {
            "logits": adjusted_logits,
            "raw_logits": raw_logits_tensor,
            "attention": attention,
            "instance_features": features,
            "label_thresholds": self.threshold.thresholds,
        }

    def get_label_thresholds(self) -> torch.Tensor:
        return self.threshold.thresholds.detach()

    def update_label_thresholds_from_validation(self, y_true: np.ndarray, y_prob: np.ndarray) -> np.ndarray:
        thresholds: list[float] = []
        search_grid = np.linspace(0.05, 0.95, num=19, dtype=np.float32)

        for label_index in range(self.num_labels):
            best_threshold = 0.5
            best_f1 = -1.0
            label_true = y_true[:, label_index].astype(np.int64)
            label_prob = y_prob[:, label_index]

            for threshold in search_grid:
                label_pred = (label_prob >= float(threshold)).astype(np.int64)
                tp = int(((label_pred == 1) & (label_true == 1)).sum())
                fp = int(((label_pred == 1) & (label_true == 0)).sum())
                fn = int(((label_pred == 0) & (label_true == 1)).sum())
                denom = 2 * tp + fp + fn
                f1 = 0.0 if denom <= 0 else (2.0 * tp) / float(denom)
                if f1 > best_f1:
                    best_f1 = f1
                    best_threshold = float(threshold)

            thresholds.append(best_threshold)

        self.threshold.set_thresholds(thresholds)
        return np.asarray(thresholds, dtype=np.float32)
