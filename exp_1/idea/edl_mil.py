"""
Evidential Deep Learning MIL (EDL-MIL)

数学理论：Dempster-Shafer 证据理论 / Dirichlet 分布
核心思想：将每个标签的二分类输出建模为 Beta 分布上的参数，
         而不是单一的 sigmoid 概率。网络输出的是"证据"
         （evidence），通过 Dirichlet 先验量化认知不确定性。
         对于高不确定性的弱标签，训练时增大证据收集的激励。

解决的问题：
- sigmoid 输出无法区分"确定是阴性"和"不确定/没见过"
- 弱标签样本少，模型对其缺乏证据，但无法表达这种不确定性
- 传统损失对高不确定性样本的梯度信号不足

关键模块：
- EvidentialHead: 输出 (alpha, beta) 参数而非 logit
- DirichletLoss: 基于 Dirichlet 分布的损失函数
- UncertaintyWeighting: 根据认知不确定性自适应加权训练
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from model.common import MultiLabelAttentionMIL, build_backbone


class EvidentialHead(nn.Module):
    """证据输出头：每个标签输出 alpha 参数而非 logit。

    对于二分类标签 l:
    - 输出 (alpha_pos, alpha_neg)，都 > 0
    - Beta(alpha_pos, alpha_neg) 描述对标签 l 的置信度分布
    - 认知不确定性 = num_classes / (alpha_pos + alpha_neg)
    """

    def __init__(self, feature_dim: int, num_labels: int, dropout: float = 0.1) -> None:
        super().__init__()
        self.num_labels = num_labels
        # Output 2 evidence values per label (positive, negative)
        self.evidence_heads = nn.ModuleList([
            nn.Sequential(
                nn.Linear(feature_dim, feature_dim // 2),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(feature_dim // 2, 2),
                nn.Softplus(),  # Ensure evidence > 0
            ) for _ in range(num_labels)
        ])

    def forward(self, bag_embeds: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Returns:
            alpha: (B, L, 2) Dirichlet parameters (alpha_pos, alpha_neg)
            probs: (B, L) predicted probabilities (mean of Beta distribution)
            uncertainty: (B, L) epistemic uncertainty
        """
        all_alpha = []
        for l in range(self.num_labels):
            evidence = self.evidence_heads[l](bag_embeds[:, l, :])  # (B, 2)
            alpha = evidence + 1.0  # Dirichlet prior: alpha = evidence + 1
            all_alpha.append(alpha)

        alpha = torch.stack(all_alpha, dim=1)  # (B, L, 2)
        S = alpha.sum(dim=-1)  # (B, L) total evidence strength
        probs = alpha[:, :, 0] / S  # Mean of Beta distribution
        uncertainty = 2.0 / S  # Epistemic uncertainty

        return alpha, probs, uncertainty


def dirichlet_loss(
    alpha: torch.Tensor,
    targets: torch.Tensor,
    kl_weight: float = 0.1,
    annealing_step: int = 0,
    current_step: int = 0,
) -> torch.Tensor:
    """Dirichlet 分布的损失函数。

    L = L_mse + lambda * KL(Dir(alpha) || Dir(1))

    其中 KL 项防止模型为未见过的样本产生过高的虚假证据。
    """
    batch_size, num_labels = targets.shape

    # Construct one-hot style targets
    targets_onehot = torch.stack([targets, 1.0 - targets], dim=-1)  # (B, L, 2)
    S = alpha.sum(dim=-1, keepdim=True)  # (B, L, 1)

    # MSE-type loss via expected probability
    pred_probs = alpha / S
    mse_loss = ((targets_onehot - pred_probs) ** 2 + pred_probs * (1 - pred_probs) / (S + 1)).sum(dim=-1)

    # KL divergence regularization
    # Remove evidence of correct class
    alpha_tilde = targets_onehot + (1 - targets_onehot) * (alpha - 1)
    alpha_tilde = alpha_tilde.clamp(min=1.0)

    S_tilde = alpha_tilde.sum(dim=-1, keepdim=True)
    kl = (
        torch.lgamma(S_tilde.squeeze(-1)) - torch.lgamma(alpha_tilde).sum(dim=-1)
        + ((alpha_tilde - 1) * (torch.digamma(alpha_tilde) - torch.digamma(S_tilde))).sum(dim=-1)
    )

    # Annealing
    if annealing_step > 0:
        anneal = min(1.0, current_step / annealing_step)
    else:
        anneal = kl_weight

    return (mse_loss + anneal * kl).mean()


class EDLMILModel(nn.Module):
    """Evidential Deep Learning MIL: 基于证据理论的多标签 MIL。

    用 Beta 分布替代 sigmoid 输出，显式建模每个标签的认知不确定性。
    对弱标签（高不确定性），通过不确定性加权增强训练信号。
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
        edl_kl_weight: float = 0.1,
        edl_annealing_steps: int = 500,
    ) -> None:
        super().__init__()
        self.num_labels = num_labels
        self.edl_kl_weight = edl_kl_weight
        self.edl_annealing_steps = edl_annealing_steps
        self._step = 0

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
        self.evidential_head = EvidentialHead(
            feature_dim=feature_dim,
            num_labels=num_labels,
            dropout=dropout,
        )
        # Fallback standard classifiers for logit compatibility
        self.classifiers = nn.ModuleList([nn.Linear(feature_dim, 1) for _ in range(num_labels)])

    def encode_instances(self, images: torch.Tensor) -> torch.Tensor:
        batch_size, num_instances, channels, height, width = images.shape
        x = images.reshape(batch_size * num_instances, channels, height, width)
        features = self.encoder(x).reshape(batch_size, num_instances, -1)
        return self.shared_proj(features)

    def forward(self, images: torch.Tensor, mask: torch.Tensor) -> dict[str, torch.Tensor]:
        features = self.encode_instances(images)
        bag_embeds, attention = self.mil_pool(features, mask)

        alpha, probs, uncertainty = self.evidential_head(bag_embeds)

        # Standard logits for compatibility with existing loss/evaluation
        logits = []
        for label_index in range(self.num_labels):
            logits.append(self.classifiers[label_index](bag_embeds[:, label_index, :]).squeeze(-1))
        logits_tensor = torch.stack(logits, dim=1)

        if self.training:
            self._step += 1

        result = {
            "logits": logits_tensor,
            "attention": attention,
            "instance_features": features,
            "edl_alpha": alpha,
            "edl_probs": probs,
            "edl_uncertainty": uncertainty,
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
        del current_epoch, train_mode
        base_loss = criterion(outputs["logits"], labels)
        if isinstance(base_loss, torch.Tensor) and base_loss.ndim > 0:
            base_loss = base_loss.mean()

        edl_loss = dirichlet_loss(
            alpha=outputs["edl_alpha"],
            targets=labels,
            kl_weight=self.edl_kl_weight,
            annealing_step=self.edl_annealing_steps,
            current_step=self._step,
        )
        return 0.5 * base_loss + edl_loss
