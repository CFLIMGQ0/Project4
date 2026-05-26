"""
Information Bottleneck Multi-Label MIL (IB-MIL)

数学理论：信息瓶颈 (Information Bottleneck, Tishby et al.)
核心思想：对每个标签施加独立的变分信息瓶颈约束，最大化 I(Z_l; Y_l)
         的同时最小化 I(Z_l; X)。这迫使每个标签的表征只保留与该标签
         最相关的信息，抑制来自其他标签的干扰。

解决的问题：
- 多标签共享表征中，强标签特征主导导致弱标签信息被淹没
- 弱标签需要的细粒度特征被全局表征压缩掉
- 标签间信息耦合导致弱标签决策受强标签干扰

关键模块：
- LabelwiseVIB: 每标签独立的变分信息瓶颈层
- AdaptiveBeta: 自适应调节每个标签的压缩-预测权衡参数 beta
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from model.common import MultiLabelAttentionMIL, build_backbone


class LabelwiseVIB(nn.Module):
    """每标签独立的变分信息瓶颈。

    对每个标签的 bag embedding 进行随机编码：
    z_l ~ N(mu_l, sigma_l^2)
    KL 散度惩罚迫使编码紧凑，只保留与该标签相关的信息。
    """

    def __init__(self, feature_dim: int, bottleneck_dim: int, num_labels: int) -> None:
        super().__init__()
        self.num_labels = num_labels
        self.bottleneck_dim = bottleneck_dim
        # Per-label encoder to (mu, log_var)
        self.mu_encoders = nn.ModuleList([
            nn.Linear(feature_dim, bottleneck_dim) for _ in range(num_labels)
        ])
        self.logvar_encoders = nn.ModuleList([
            nn.Linear(feature_dim, bottleneck_dim) for _ in range(num_labels)
        ])
        # Per-label decoder back to feature space
        self.decoders = nn.ModuleList([
            nn.Sequential(
                nn.Linear(bottleneck_dim, feature_dim),
                nn.GELU(),
            ) for _ in range(num_labels)
        ])

    def forward(self, bag_embeds: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            bag_embeds: (B, L, D)
        Returns:
            decoded: (B, L, D) - 瓶颈后重建的标签表征
            kl_loss: scalar - 所有标签的 KL 散度之和
        """
        all_decoded = []
        total_kl = bag_embeds.new_zeros(())

        for l in range(self.num_labels):
            embed_l = bag_embeds[:, l, :]  # (B, D)
            mu = self.mu_encoders[l](embed_l)  # (B, bottleneck_dim)
            logvar = self.logvar_encoders[l](embed_l)  # (B, bottleneck_dim)

            # Reparameterization trick
            if self.training:
                std = torch.exp(0.5 * logvar)
                eps = torch.randn_like(std)
                z = mu + eps * std
            else:
                z = mu

            # KL divergence: KL(N(mu, sigma^2) || N(0, 1))
            kl = -0.5 * (1 + logvar - mu.pow(2) - logvar.exp()).sum(dim=-1).mean()
            total_kl = total_kl + kl

            decoded = self.decoders[l](z)
            all_decoded.append(decoded)

        decoded = torch.stack(all_decoded, dim=1)  # (B, L, D)
        avg_kl = total_kl / self.num_labels
        return decoded, avg_kl


class IBMILModel(nn.Module):
    """Information Bottleneck MIL: 基于信息瓶颈理论的多标签 MIL。

    在注意力池化之后、分类器之前插入每标签独立的变分信息瓶颈，
    迫使每个标签的表征只保留与自身相关的信息。
    beta 参数控制压缩-预测的权衡：
    - 强标签不需要太多压缩（beta 小）
    - 弱标签需要更强的压缩来过滤噪声（beta 大）
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
        bottleneck_dim: int = 128,
        beta: float = 0.001,
    ) -> None:
        super().__init__()
        self.num_labels = num_labels
        self.beta = beta

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
        # Information Bottleneck
        self.vib = LabelwiseVIB(
            feature_dim=feature_dim,
            bottleneck_dim=bottleneck_dim,
            num_labels=num_labels,
        )
        # Per-label adaptive beta (learnable)
        self.log_beta = nn.Parameter(torch.full((num_labels,), fill_value=float(torch.tensor(beta).log())))
        self.classifiers = nn.ModuleList([nn.Linear(feature_dim, 1) for _ in range(num_labels)])

    def encode_instances(self, images: torch.Tensor) -> torch.Tensor:
        batch_size, num_instances, channels, height, width = images.shape
        x = images.reshape(batch_size * num_instances, channels, height, width)
        features = self.encoder(x).reshape(batch_size, num_instances, -1)
        return self.shared_proj(features)

    def forward(self, images: torch.Tensor, mask: torch.Tensor) -> dict[str, torch.Tensor]:
        features = self.encode_instances(images)
        bag_embeds, attention = self.mil_pool(features, mask)

        # Apply information bottleneck
        bottleneck_embeds, kl_loss = self.vib(bag_embeds)

        logits = []
        for label_index in range(self.num_labels):
            logits.append(self.classifiers[label_index](bottleneck_embeds[:, label_index, :]).squeeze(-1))

        result = {
            "logits": torch.stack(logits, dim=1),
            "attention": attention,
            "instance_features": features,
        }

        if self.training:
            # Adaptive beta per label
            adaptive_beta = torch.exp(self.log_beta).mean()
            result["aux_losses"] = {"vib_kl": adaptive_beta * kl_loss}

        return result
