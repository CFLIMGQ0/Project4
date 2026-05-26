"""
Optimal Transport Label Alignment MIL (OT-MIL)

数学理论：最优传输 (Optimal Transport / Wasserstein Distance)
核心思想：用 Sinkhorn 算法求解实例到标签原型之间的最优传输计划，
         替代传统的注意力池化。最优传输计划天然保证每个标签都能
         分配到足够的实例，防止强标签垄断所有实例的注意力。

解决的问题：
- 长尾标签因注意力竞争不足而被强标签压制
- 传统 softmax attention 的赢者通吃问题
- 弱标签 (ulcer, reflux_esophagitis) 无法获得有效实例关注

关键模块：
- SinkhornOT: 可微分的 Sinkhorn 最优传输求解器
- LabelPrototypes: 可学习的标签原型嵌入
- OTPooling: 基于传输计划的加权池化
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from model.common import build_backbone, masked_softmax


class SinkhornOT(nn.Module):
    """可微分 Sinkhorn 最优传输求解器。

    给定 cost matrix C (B, L, N)，求解传输计划 T
    使得 sum(T * C) 最小，同时满足行列边际约束。
    使用熵正则化使问题可微分。
    """

    def __init__(self, num_iters: int = 10, epsilon: float = 0.1, label_mass_min: float = 0.05) -> None:
        super().__init__()
        self.num_iters = num_iters
        self.epsilon = epsilon
        self.label_mass_min = label_mass_min

    def forward(
        self,
        cost: torch.Tensor,
        mask: torch.Tensor,
        label_mass: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """
        Args:
            cost: (B, L, N) cost matrix (label-to-instance distance)
            mask: (B, N) instance validity mask
            label_mass: (B, L) desired mass per label, defaults to uniform
        Returns:
            transport_plan: (B, L, N) optimal transport plan
        """
        batch_size, num_labels, num_instances = cost.shape

        # Kernel matrix from cost
        K = torch.exp(-cost / self.epsilon)
        # Mask invalid instances
        mask_expanded = mask.unsqueeze(1).to(dtype=K.dtype)
        K = K * mask_expanded

        # Marginals
        if label_mass is None:
            # Each label gets equal mass, with minimum floor for rare labels
            mu = torch.ones(batch_size, num_labels, device=cost.device, dtype=cost.dtype) / num_labels
        else:
            mu = label_mass
            mu = mu.clamp(min=self.label_mass_min)
            mu = mu / mu.sum(dim=-1, keepdim=True)

        # Instance marginal: uniform over valid instances
        valid_count = mask.sum(dim=-1, keepdim=True).clamp_min(1.0)
        nu = mask.to(dtype=cost.dtype) / valid_count

        # Sinkhorn iterations
        u = torch.ones(batch_size, num_labels, 1, device=cost.device, dtype=cost.dtype)
        v = torch.ones(batch_size, 1, num_instances, device=cost.device, dtype=cost.dtype)

        for _ in range(self.num_iters):
            # Row normalization
            Kv = (K * v).sum(dim=-1, keepdim=True).clamp_min(1e-10)
            u = mu.unsqueeze(-1) / Kv
            # Column normalization
            Ku = (K * u).sum(dim=1, keepdim=True).clamp_min(1e-10)
            v = nu.unsqueeze(1) / Ku

        transport_plan = u * K * v
        # Re-mask
        transport_plan = transport_plan * mask_expanded
        # Normalize rows to sum to 1 for use as attention weights
        row_sum = transport_plan.sum(dim=-1, keepdim=True).clamp_min(1e-10)
        transport_plan = transport_plan / row_sum

        return transport_plan


class OTAttentionMIL(nn.Module):
    """基于最优传输的多标签注意力池化。"""

    def __init__(
        self,
        in_dim: int,
        attn_dim: int,
        num_labels: int,
        num_sinkhorn_iters: int = 10,
        ot_epsilon: float = 0.1,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.num_labels = num_labels
        # Label prototypes
        self.label_protos = nn.Parameter(torch.randn(num_labels, in_dim) * 0.02)
        # Instance projector for cost computation
        self.inst_proj = nn.Sequential(
            nn.Linear(in_dim, attn_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(attn_dim, in_dim),
        )
        self.sinkhorn = SinkhornOT(
            num_iters=num_sinkhorn_iters,
            epsilon=ot_epsilon,
        )
        # Learnable label mass (for imbalanced labels)
        self.label_mass_logits = nn.Parameter(torch.zeros(num_labels))

    def forward(
        self, features: torch.Tensor, mask: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        batch_size = features.shape[0]

        # Project instances
        inst_embed = self.inst_proj(features)
        inst_embed_norm = F.normalize(inst_embed, dim=-1)
        proto_norm = F.normalize(self.label_protos, dim=-1)

        # Cost matrix: negative cosine similarity (lower = more similar)
        # (B, N, D) x (L, D)^T -> (B, N, L) -> (B, L, N)
        similarity = torch.einsum("bnd,ld->bnl", inst_embed_norm, proto_norm)
        cost = 1.0 - similarity  # (B, N, L)
        cost = cost.permute(0, 2, 1)  # (B, L, N)

        # Label mass from learnable logits
        label_mass = torch.softmax(self.label_mass_logits, dim=0)
        label_mass = label_mass.unsqueeze(0).expand(batch_size, -1)

        # Solve OT
        transport_plan = self.sinkhorn(cost, mask, label_mass)

        # Pool using transport plan as attention
        bag_embeds = torch.einsum("bln,bnd->bld", transport_plan, features)

        return bag_embeds, transport_plan


class OTMILModel(nn.Module):
    """Optimal Transport MIL: 基于最优传输理论的多标签 MIL。

    用 Sinkhorn OT 替代传统注意力池化，确保每个标签都能获得
    足够的实例分配，天然解决长尾标签被压制的问题。
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
        num_sinkhorn_iters: int = 10,
        ot_epsilon: float = 0.1,
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
        self.ot_pool = OTAttentionMIL(
            in_dim=feature_dim,
            attn_dim=attn_dim,
            num_labels=num_labels,
            num_sinkhorn_iters=num_sinkhorn_iters,
            ot_epsilon=ot_epsilon,
            dropout=dropout,
        )
        self.classifiers = nn.ModuleList([nn.Linear(feature_dim, 1) for _ in range(num_labels)])

    def encode_instances(self, images: torch.Tensor) -> torch.Tensor:
        batch_size, num_instances, channels, height, width = images.shape
        x = images.reshape(batch_size * num_instances, channels, height, width)
        features = self.encoder(x).reshape(batch_size, num_instances, -1)
        return self.shared_proj(features)

    def forward(self, images: torch.Tensor, mask: torch.Tensor) -> dict[str, torch.Tensor]:
        features = self.encode_instances(images)
        bag_embeds, attention = self.ot_pool(features, mask)

        logits = []
        for label_index in range(self.num_labels):
            logits.append(self.classifiers[label_index](bag_embeds[:, label_index, :]).squeeze(-1))

        return {
            "logits": torch.stack(logits, dim=1),
            "attention": attention,
            "instance_features": features,
        }
