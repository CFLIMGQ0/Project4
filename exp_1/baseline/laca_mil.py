"""
Label-Aware Cross-Attention MIL (LACA-MIL)

来源：受 BLIP-2 Q-Former (Li et al., 2023) 启发，适配 MIL 分类

核心思想：引入可学习的标签查询 (label queries)，通过交叉注意力
         机制让每个标签查询选择性地关注最相关的实例特征。
         与简单的注意力池化不同，这里使用完整的 Transformer
         交叉注意力，让标签查询之间也能互相感知。

解决的问题：
- 传统注意力池化中标签间共享权重，弱标签被强标签主导
- 标签查询之间无交互，不能利用标签间的互补信息
- 弱标签缺乏独立的特征提取路径

关键模块：
- LabelQueryCrossAttention: 标签查询对实例特征的交叉注意力
- LabelQuerySelfAttention: 标签查询间的自注意力（捕获标签关系）
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from model.common import build_backbone, masked_softmax


class LabelQueryAttentionBlock(nn.Module):
    """标签查询注意力块：self-attention + cross-attention。

    类似 Q-Former 的结构：
    1. Label queries 之间做 self-attention（标签关系建模）
    2. Label queries 对 instance features 做 cross-attention（特征选择）
    """

    def __init__(
        self,
        feature_dim: int,
        num_heads: int = 4,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.self_attn = nn.MultiheadAttention(
            embed_dim=feature_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True,
        )
        self.cross_attn = nn.MultiheadAttention(
            embed_dim=feature_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True,
        )
        self.norm1 = nn.LayerNorm(feature_dim)
        self.norm2 = nn.LayerNorm(feature_dim)
        self.norm3 = nn.LayerNorm(feature_dim)
        self.ffn = nn.Sequential(
            nn.Linear(feature_dim, feature_dim * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(feature_dim * 2, feature_dim),
            nn.Dropout(dropout),
        )

    def forward(
        self,
        label_queries: torch.Tensor,
        instance_features: torch.Tensor,
        instance_mask: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            label_queries: (B, L, D)
            instance_features: (B, N, D)
            instance_mask: (B, N)
        Returns:
            updated_queries: (B, L, D)
            cross_attention_weights: (B, L, N)
        """
        # Self-attention among label queries
        residual = label_queries
        q = self.norm1(label_queries)
        q, _ = self.self_attn(q, q, q)
        label_queries = residual + q

        # Cross-attention: labels attend to instances
        residual = label_queries
        q = self.norm2(label_queries)
        key_padding_mask = ~instance_mask.to(dtype=torch.bool)
        q, cross_attn_weights = self.cross_attn(
            q, instance_features, instance_features,
            key_padding_mask=key_padding_mask,
        )
        label_queries = residual + q

        # FFN
        residual = label_queries
        label_queries = residual + self.ffn(self.norm3(label_queries))

        return label_queries, cross_attn_weights


class LACAMILModel(nn.Module):
    """Label-Aware Cross-Attention MIL: 基于标签查询交叉注意力的 MIL。

    使用可学习的标签查询通过 Transformer 式的交叉注意力选择性
    聚合实例特征。标签间的自注意力允许弱标签从强标签借取信息。
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
        num_query_layers: int = 2,
        num_heads: int = 4,
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

        # Learnable label query embeddings
        self.label_queries = nn.Parameter(torch.randn(num_labels, feature_dim) * 0.02)

        # Instance context encoder (optional transformer)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=feature_dim,
            nhead=num_heads,
            dim_feedforward=feature_dim * 2,
            dropout=dropout,
            batch_first=True,
            norm_first=True,
        )
        self.instance_encoder = nn.TransformerEncoder(encoder_layer, num_layers=1)

        # Label query attention layers
        self.query_layers = nn.ModuleList([
            LabelQueryAttentionBlock(
                feature_dim=feature_dim,
                num_heads=num_heads,
                dropout=dropout,
            ) for _ in range(num_query_layers)
        ])

        self.classifiers = nn.ModuleList([nn.Linear(feature_dim, 1) for _ in range(num_labels)])

    def encode_instances(self, images: torch.Tensor) -> torch.Tensor:
        batch_size, num_instances, channels, height, width = images.shape
        x = images.reshape(batch_size * num_instances, channels, height, width)
        features = self.encoder(x).reshape(batch_size, num_instances, -1)
        return self.shared_proj(features)

    def forward(self, images: torch.Tensor, mask: torch.Tensor) -> dict[str, torch.Tensor]:
        features = self.encode_instances(images)

        # Instance context modeling
        contextual_features = self.instance_encoder(
            features,
            src_key_padding_mask=~mask.to(dtype=torch.bool),
        )

        # Initialize label queries per batch
        batch_size = features.shape[0]
        queries = self.label_queries.unsqueeze(0).expand(batch_size, -1, -1)

        # Apply cross-attention layers
        all_attn = None
        for layer in self.query_layers:
            queries, attn_weights = layer(queries, contextual_features, mask)
            all_attn = attn_weights

        # Classify
        logits = []
        for label_index in range(self.num_labels):
            logits.append(self.classifiers[label_index](queries[:, label_index, :]).squeeze(-1))

        # Construct attention-like output (B, L, N) for compatibility
        if all_attn is None:
            attention = torch.zeros(batch_size, self.num_labels, features.shape[1], device=features.device)
        else:
            attention = all_attn

        return {
            "logits": torch.stack(logits, dim=1),
            "attention": attention,
            "instance_features": features,
        }
