from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from model.common import build_backbone


class InstanceEncoder(nn.Module):
    """共享实例编码器。"""

    def __init__(
        self,
        backbone_name: str,
        pretrained: bool,
        freeze_stages: int,
        feature_dim: int,
        dropout: float,
    ) -> None:
        super().__init__()
        self.backbone, out_dim = build_backbone(
            backbone_name=backbone_name,
            pretrained=pretrained,
            out_dim=feature_dim,
            freeze_stages=freeze_stages,
            projector_dropout=dropout,
        )
        self.projector = nn.Sequential(
            nn.Linear(out_dim, feature_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(feature_dim, feature_dim),
            nn.GELU(),
        )

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        batch_size, num_instances, channels, height, width = images.shape
        x = images.reshape(batch_size * num_instances, channels, height, width)
        features = self.backbone(x).reshape(batch_size, num_instances, -1)
        return self.projector(features)


class LabelGraphReasoner(nn.Module):
    """基于可学习标签图对标签表征做一次关系传播。"""

    def __init__(self, num_labels: int, feature_dim: int, dropout: float) -> None:
        super().__init__()
        self.label_tokens = nn.Parameter(torch.randn(num_labels, feature_dim) * 0.02)
        self.refine = nn.Sequential(
            nn.Linear(feature_dim * 2, feature_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )

    def forward(self, bag_embeds: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        label_tokens = F.normalize(self.label_tokens, dim=-1)
        graph_logits = torch.matmul(label_tokens, label_tokens.transpose(0, 1)) / math.sqrt(label_tokens.shape[-1])
        label_graph = torch.softmax(graph_logits, dim=-1)

        propagated = torch.einsum("lk,bkd->bld", label_graph, bag_embeds)
        refined = bag_embeds + self.refine(torch.cat([bag_embeds, propagated], dim=-1))
        return refined, label_graph


class LabelSelfAttentionReasoner(nn.Module):
    """用通用标签自注意力替代可学习标签图传播。"""

    def __init__(self, num_labels: int, feature_dim: int, dropout: float) -> None:
        super().__init__()
        self.num_labels = int(num_labels)
        self.norm = nn.LayerNorm(feature_dim)
        self.self_attention = nn.MultiheadAttention(
            embed_dim=feature_dim,
            num_heads=1,
            dropout=dropout,
            batch_first=True,
        )
        self.refine = nn.Sequential(
            nn.Linear(feature_dim * 2, feature_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )

    def forward(self, bag_embeds: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        normalized = self.norm(bag_embeds)
        attended, attention = self.self_attention(
            normalized,
            normalized,
            normalized,
            need_weights=True,
            average_attn_weights=True,
        )
        refined = bag_embeds + self.refine(torch.cat([bag_embeds, attended], dim=-1))
        label_graph = attention.mean(dim=0) if attention.dim() == 3 else attention
        return refined, label_graph


class StaticLabelGCNReasoner(nn.Module):
    """基于训练集标签共现先验的静态 GCN 式标签传播。"""

    def __init__(
        self,
        num_labels: int,
        feature_dim: int,
        dropout: float,
        adjacency: list[list[float]] | torch.Tensor | None = None,
    ) -> None:
        super().__init__()
        self.num_labels = int(num_labels)
        if adjacency is None:
            graph = torch.eye(self.num_labels, dtype=torch.float32)
        else:
            graph = torch.as_tensor(adjacency, dtype=torch.float32)
            if graph.shape != (self.num_labels, self.num_labels):
                raise ValueError("label_graph_prior 的形状必须是 num_labels x num_labels")
            graph = graph.clamp_min(0.0)
            graph = graph + torch.eye(self.num_labels, dtype=torch.float32)
        graph = graph / graph.sum(dim=-1, keepdim=True).clamp_min(1e-12)
        self.register_buffer("label_graph", graph)
        self.message = nn.Sequential(
            nn.Linear(feature_dim, feature_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        self.refine = nn.Sequential(
            nn.Linear(feature_dim * 2, feature_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )

    def forward(self, bag_embeds: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        graph = self.label_graph.to(device=bag_embeds.device, dtype=bag_embeds.dtype)
        messages = self.message(bag_embeds)
        propagated = torch.einsum("lk,bkd->bld", graph, messages)
        refined = bag_embeds + self.refine(torch.cat([bag_embeds, propagated], dim=-1))
        return refined, graph


class DynamicLabelGATReasoner(nn.Module):
    """按当前样本动态估计标签关系的 GAT 式标签传播。"""

    def __init__(self, num_labels: int, feature_dim: int, dropout: float) -> None:
        super().__init__()
        self.num_labels = int(num_labels)
        self.query = nn.Linear(feature_dim, feature_dim)
        self.key = nn.Linear(feature_dim, feature_dim)
        self.value = nn.Linear(feature_dim, feature_dim)
        self.dropout = nn.Dropout(dropout)
        self.refine = nn.Sequential(
            nn.Linear(feature_dim * 2, feature_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )

    def forward(self, bag_embeds: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        query = self.query(bag_embeds)
        key = self.key(bag_embeds)
        value = self.value(bag_embeds)
        graph_logits = torch.matmul(query, key.transpose(1, 2)) / math.sqrt(query.shape[-1])
        attention = torch.softmax(graph_logits, dim=-1)
        propagated = torch.matmul(self.dropout(attention), value)
        refined = bag_embeds + self.refine(torch.cat([bag_embeds, propagated], dim=-1))
        label_graph = attention.mean(dim=0)
        return refined, label_graph


class LabelTransformerReasoner(nn.Module):
    """用 Transformer encoder 建模标签间上下文关系。"""

    def __init__(self, num_labels: int, feature_dim: int, dropout: float, num_heads: int = 4) -> None:
        super().__init__()
        self.num_labels = int(num_labels)
        head_count = int(num_heads) if feature_dim % int(num_heads) == 0 else 1
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=feature_dim,
            nhead=head_count,
            dim_feedforward=feature_dim * 2,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=1)
        self.refine = nn.Sequential(
            nn.Linear(feature_dim * 2, feature_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )

    def forward(self, bag_embeds: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        contextual = self.encoder(bag_embeds)
        refined = bag_embeds + self.refine(torch.cat([bag_embeds, contextual], dim=-1))
        normalized = F.normalize(contextual, dim=-1)
        label_graph = torch.softmax(torch.matmul(normalized, normalized.transpose(1, 2)), dim=-1).mean(dim=0)
        return refined, label_graph


class LowRankLabelGraphReasoner(nn.Module):
    """用低秩可学习邻接矩阵替代完整标签图。"""

    def __init__(self, num_labels: int, feature_dim: int, dropout: float, rank: int = 2) -> None:
        super().__init__()
        self.num_labels = int(num_labels)
        self.rank = max(1, int(rank))
        self.source_factors = nn.Parameter(torch.randn(self.num_labels, self.rank) * 0.02)
        self.target_factors = nn.Parameter(torch.randn(self.num_labels, self.rank) * 0.02)
        self.message = nn.Sequential(
            nn.Linear(feature_dim, feature_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        self.refine = nn.Sequential(
            nn.Linear(feature_dim * 2, feature_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )

    def forward(self, bag_embeds: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        graph_logits = torch.matmul(self.source_factors, self.target_factors.transpose(0, 1)) / math.sqrt(self.rank)
        label_graph = torch.softmax(graph_logits, dim=-1)
        messages = self.message(bag_embeds)
        propagated = torch.einsum("lk,bkd->bld", label_graph, messages)
        refined = bag_embeds + self.refine(torch.cat([bag_embeds, propagated], dim=-1))
        return refined, label_graph


class CosineLabelGraphReasoner(nn.Module):
    """按标签表征余弦相似度动态构图。"""

    def __init__(self, num_labels: int, feature_dim: int, dropout: float) -> None:
        super().__init__()
        self.num_labels = int(num_labels)
        self.logit_scale = nn.Parameter(torch.tensor(1.0))
        self.value = nn.Sequential(
            nn.Linear(feature_dim, feature_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        self.refine = nn.Sequential(
            nn.Linear(feature_dim * 2, feature_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )

    def forward(self, bag_embeds: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        normalized = F.normalize(bag_embeds, dim=-1)
        graph_logits = torch.matmul(normalized, normalized.transpose(1, 2)) * self.logit_scale.exp().clamp(max=20.0)
        attention = torch.softmax(graph_logits, dim=-1)
        propagated = torch.matmul(attention, self.value(bag_embeds))
        refined = bag_embeds + self.refine(torch.cat([bag_embeds, propagated], dim=-1))
        label_graph = attention.mean(dim=0)
        return refined, label_graph


class LabelMLPMixerReasoner(nn.Module):
    """用 MLP-Mixer 风格的 token mixing 做标签关系建模。"""

    def __init__(self, num_labels: int, feature_dim: int, dropout: float) -> None:
        super().__init__()
        self.num_labels = int(num_labels)
        hidden_labels = max(self.num_labels * 2, 4)
        self.label_norm = nn.LayerNorm(feature_dim)
        self.feature_norm = nn.LayerNorm(feature_dim)
        self.token_fc1 = nn.Linear(self.num_labels, hidden_labels)
        self.token_fc2 = nn.Linear(hidden_labels, self.num_labels)
        self.feature_mixer = nn.Sequential(
            nn.Linear(feature_dim, feature_dim * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(feature_dim * 2, feature_dim),
        )
        self.dropout = nn.Dropout(dropout)
        self.refine = nn.Sequential(
            nn.Linear(feature_dim * 2, feature_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )

    def forward(self, bag_embeds: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        normalized = self.label_norm(bag_embeds)
        token_hidden = F.gelu(self.token_fc1(normalized.transpose(1, 2)))
        token_update = self.token_fc2(self.dropout(token_hidden)).transpose(1, 2)
        mixed = bag_embeds + token_update
        mixed = mixed + self.feature_mixer(self.feature_norm(mixed))
        refined = bag_embeds + self.refine(torch.cat([bag_embeds, mixed], dim=-1))

        graph_logits = torch.matmul(self.token_fc2.weight, self.token_fc1.weight)
        label_graph = torch.softmax(graph_logits, dim=-1)
        return refined, label_graph


class LabelHypergraphReasoner(nn.Module):
    """用可学习超边聚合多标签高阶关系。"""

    def __init__(self, num_labels: int, feature_dim: int, dropout: float, num_hyperedges: int = 2) -> None:
        super().__init__()
        self.num_labels = int(num_labels)
        self.num_hyperedges = max(1, int(num_hyperedges))
        self.incidence_logits = nn.Parameter(torch.randn(self.num_labels, self.num_hyperedges) * 0.02)
        self.message = nn.Sequential(
            nn.Linear(feature_dim, feature_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        self.refine = nn.Sequential(
            nn.Linear(feature_dim * 2, feature_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )

    def forward(self, bag_embeds: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        node_to_edge = torch.softmax(self.incidence_logits, dim=0)
        edge_to_node = torch.softmax(self.incidence_logits, dim=1)
        messages = self.message(bag_embeds)
        edge_embeds = torch.einsum("le,bld->bed", node_to_edge, messages)
        propagated = torch.einsum("le,bed->bld", edge_to_node, edge_embeds)
        refined = bag_embeds + self.refine(torch.cat([bag_embeds, propagated], dim=-1))

        label_graph = torch.matmul(edge_to_node, node_to_edge.transpose(0, 1))
        label_graph = label_graph / label_graph.sum(dim=-1, keepdim=True).clamp_min(1e-12)
        return refined, label_graph
