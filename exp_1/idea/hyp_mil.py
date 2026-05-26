"""
Hyperbolic Label Embedding MIL (Hyp-MIL)

数学理论：双曲几何 / Poincare 球模型
核心思想：将标签表征嵌入到双曲空间（Poincare 球）中进行图传播。
         双曲空间天然适合表示层次结构，因为其指数增长的体积可以
         高效编码树状层级关系。这使得子标签（如 gastritis_active）
         能够从父标签（gastritis）继承表征信息。

解决的问题：
- 标签间的层次关系（gastritis -> gastritis_active, gastritis -> atrophy）
  在欧氏空间中难以有效建模
- 稀有子标签无法从常见父标签借力
- 标签图推理在欧氏空间中对层次深度不敏感

关键模块：
- PoincareBall: Poincare 球模型的基本运算（Mobius 加法、指数映射等）
- HyperbolicLabelGraph: 在双曲空间中进行标签图传播
- HyperbolicClassifier: 基于双曲距离的分类器
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from model.common import MultiLabelAttentionMIL, build_backbone


# ---- Poincare Ball Operations ----

def project_to_ball(x: torch.Tensor, c: float = 1.0, eps: float = 1e-5) -> torch.Tensor:
    """将点投影回 Poincare 球内部。"""
    max_norm = (1.0 - eps) / (c ** 0.5)
    norm = x.norm(dim=-1, keepdim=True).clamp_min(1e-10)
    cond = norm > max_norm
    projected = x / norm * max_norm
    return torch.where(cond, projected, x)


def mobius_add(x: torch.Tensor, y: torch.Tensor, c: float = 1.0) -> torch.Tensor:
    """Mobius 加法: x oplus y in Poincare ball."""
    x_sq = (x * x).sum(dim=-1, keepdim=True)
    y_sq = (y * y).sum(dim=-1, keepdim=True)
    xy = (x * y).sum(dim=-1, keepdim=True)

    num = (1 + 2 * c * xy + c * y_sq) * x + (1 - c * x_sq) * y
    denom = 1 + 2 * c * xy + c * c * x_sq * y_sq
    return project_to_ball(num / denom.clamp_min(1e-10), c)


def expmap0(v: torch.Tensor, c: float = 1.0) -> torch.Tensor:
    """从原点的指数映射：将切空间向量映射到 Poincare 球。"""
    v_norm = v.norm(dim=-1, keepdim=True).clamp_min(1e-10)
    sqrt_c = c ** 0.5
    return project_to_ball(
        torch.tanh(sqrt_c * v_norm) * v / (sqrt_c * v_norm), c
    )


def logmap0(y: torch.Tensor, c: float = 1.0) -> torch.Tensor:
    """到原点的对数映射：将 Poincare 球上的点映射回切空间。"""
    y_norm = y.norm(dim=-1, keepdim=True).clamp_min(1e-10)
    sqrt_c = c ** 0.5
    return torch.atanh(sqrt_c * y_norm).clamp(max=5.0) * y / (sqrt_c * y_norm)


def hyperbolic_distance(x: torch.Tensor, y: torch.Tensor, c: float = 1.0) -> torch.Tensor:
    """Poincare 球上两点之间的测地距离。"""
    diff = mobius_add(-x, y, c)
    diff_norm = diff.norm(dim=-1).clamp_min(1e-10)
    sqrt_c = c ** 0.5
    return (2.0 / sqrt_c) * torch.atanh(sqrt_c * diff_norm).clamp(max=10.0)


class HyperbolicLabelGraph(nn.Module):
    """在双曲空间中进行标签图传播。

    1. 将标签嵌入映射到 Poincare 球
    2. 用双曲距离计算标签间亲和度
    3. 在双曲空间中做加权聚合（通过切空间近似）
    """

    def __init__(
        self,
        num_labels: int,
        feature_dim: int,
        curvature: float = 1.0,
        dropout: float = 0.2,
    ) -> None:
        super().__init__()
        self.num_labels = num_labels
        self.feature_dim = feature_dim
        self.c = curvature

        # Learnable label tokens in tangent space (will be mapped to Poincare ball)
        self.label_tokens = nn.Parameter(torch.randn(num_labels, feature_dim) * 0.02)
        # Euclidean-to-hyperbolic projector
        self.to_hyp = nn.Linear(feature_dim, feature_dim)
        # Hyperbolic-to-Euclidean decoder
        self.from_hyp = nn.Linear(feature_dim, feature_dim)
        # Refinement
        self.refine = nn.Sequential(
            nn.Linear(feature_dim * 2, feature_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )

    def forward(self, bag_embeds: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            bag_embeds: (B, L, D) label-specific bag embeddings in Euclidean space
        Returns:
            refined: (B, L, D) graph-propagated embeddings
            affinity: (L, L) label affinity matrix
        """
        batch_size = bag_embeds.shape[0]

        # Map label tokens to Poincare ball
        hyp_tokens = expmap0(self.label_tokens, self.c)  # (L, D)

        # Compute label affinity via hyperbolic distance
        # Lower distance = higher affinity
        dist_matrix = torch.zeros(self.num_labels, self.num_labels, device=bag_embeds.device)
        for i in range(self.num_labels):
            for j in range(self.num_labels):
                dist_matrix[i, j] = hyperbolic_distance(
                    hyp_tokens[i].unsqueeze(0),
                    hyp_tokens[j].unsqueeze(0),
                    self.c,
                )
        affinity = torch.softmax(-dist_matrix / math.sqrt(self.feature_dim), dim=-1)

        # Map bag embeddings to hyperbolic space
        bag_tangent = self.to_hyp(bag_embeds)  # (B, L, D)
        bag_hyp = expmap0(bag_tangent, self.c)  # (B, L, D)

        # Propagate in tangent space (first-order approximation)
        bag_tangent_for_prop = logmap0(bag_hyp, self.c)  # (B, L, D)
        propagated_tangent = torch.einsum("lk,bkd->bld", affinity, bag_tangent_for_prop)
        propagated_hyp = expmap0(propagated_tangent, self.c)

        # Map back to Euclidean
        propagated_eucl = self.from_hyp(logmap0(propagated_hyp, self.c))

        # Residual refinement
        refined = bag_embeds + self.refine(torch.cat([bag_embeds, propagated_eucl], dim=-1))
        return refined, affinity


class HypMILModel(nn.Module):
    """Hyperbolic MIL: 基于双曲几何的多标签 MIL。

    在标签图推理阶段使用 Poincare 球模型，让标签间的层次关系
    （如 gastritis -> gastritis_active）得到几何上的天然建模。
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
        curvature: float = 1.0,
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
        self.hyp_graph = HyperbolicLabelGraph(
            num_labels=num_labels,
            feature_dim=feature_dim,
            curvature=curvature,
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
        bag_embeds, attention = self.mil_pool(features, mask)

        refined_embeds, label_graph = self.hyp_graph(bag_embeds)

        logits = []
        for label_index in range(self.num_labels):
            logits.append(self.classifiers[label_index](refined_embeds[:, label_index, :]).squeeze(-1))

        return {
            "logits": torch.stack(logits, dim=1),
            "attention": attention,
            "instance_features": features,
            "label_graph": label_graph,
        }
