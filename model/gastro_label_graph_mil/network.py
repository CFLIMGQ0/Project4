from __future__ import annotations

import torch
import torch.nn as nn

from model.common import MultiLabelAttentionMIL, SingleAttentionMIL

from .modules import (
    CosineLabelGraphReasoner,
    DynamicLabelGATReasoner,
    InstanceEncoder,
    LabelHypergraphReasoner,
    LabelGraphReasoner,
    LabelMLPMixerReasoner,
    LabelSelfAttentionReasoner,
    LabelTransformerReasoner,
    LowRankLabelGraphReasoner,
    StaticLabelGCNReasoner,
)


class GastroLabelGraphMIL(nn.Module):
    """胃镜标签图传播 MIL。"""

    def __init__(
        self,
        backbone_name: str = "convnext_tiny",
        pretrained: bool = True,
        freeze_stages: int = 0,
        feature_dim: int = 512,
        attn_dim: int = 256,
        num_labels: int = 3,
        dropout: float = 0.2,
        use_label_graph: bool = True,
        label_graph_type: str = "learnable",
        label_graph_prior: list[list[float]] | None = None,
        label_graph_rank: int = 2,
        label_graph_heads: int = 4,
        label_hypergraph_edges: int = 2,
        use_label_wise_attention: bool = True,
        attention_type: str = "label_specific",
        pooling_type: str = "label_attention",
    ) -> None:
        super().__init__()
        self.num_labels = num_labels
        self.use_label_graph = bool(use_label_graph)
        self.label_graph_type = str(label_graph_type).strip().lower() or "learnable"
        self.use_label_wise_attention = bool(use_label_wise_attention)
        self.attention_type = str(attention_type).strip().lower() or "label_specific"
        self.pooling_type = str(pooling_type).strip().lower() or "label_attention"

        if self.label_graph_type not in {
            "learnable",
            "self_attention",
            "static_gcn",
            "dynamic_gat",
            "label_transformer",
            "low_rank_graph",
            "cosine_graph",
            "label_mlp_mixer",
            "label_hypergraph",
        }:
            raise ValueError(f"不支持的 label_graph_type: {label_graph_type}")
        if self.pooling_type not in {"label_attention", "shared_attention", "mean"}:
            raise ValueError(f"不支持的 pooling_type: {pooling_type}")
        if self.attention_type not in {"label_specific", "shared", "none"}:
            raise ValueError(f"不支持的 attention_type: {attention_type}")
        if self.pooling_type == "label_attention" and self.attention_type != "label_specific":
            raise ValueError("pooling_type=label_attention 时 attention_type 必须为 label_specific")
        if self.pooling_type == "shared_attention" and self.attention_type != "shared":
            raise ValueError("pooling_type=shared_attention 时 attention_type 必须为 shared")
        if self.pooling_type == "mean" and self.attention_type != "none":
            raise ValueError("pooling_type=mean 时 attention_type 必须为 none")

        self.instance_encoder = InstanceEncoder(
            backbone_name=backbone_name,
            pretrained=pretrained,
            freeze_stages=freeze_stages,
            feature_dim=feature_dim,
            dropout=dropout,
        )
        if self.pooling_type == "label_attention":
            self.mil_pool = MultiLabelAttentionMIL(
                in_dim=feature_dim,
                attn_dim=attn_dim,
                num_labels=num_labels,
                dropout=dropout,
            )
        elif self.pooling_type == "shared_attention":
            self.mil_pool = SingleAttentionMIL(
                in_dim=feature_dim,
                attn_dim=attn_dim,
                dropout=dropout,
            )
        else:
            self.mil_pool = None

        self.label_graph_reasoner = None
        if self.use_label_graph and self.label_graph_type == "learnable":
            self.label_graph_reasoner = LabelGraphReasoner(
                num_labels=num_labels,
                feature_dim=feature_dim,
                dropout=dropout,
            )
        elif self.use_label_graph and self.label_graph_type == "self_attention":
            self.label_graph_reasoner = LabelSelfAttentionReasoner(
                num_labels=num_labels,
                feature_dim=feature_dim,
                dropout=dropout,
            )
        elif self.use_label_graph and self.label_graph_type == "static_gcn":
            self.label_graph_reasoner = StaticLabelGCNReasoner(
                num_labels=num_labels,
                feature_dim=feature_dim,
                dropout=dropout,
                adjacency=label_graph_prior,
            )
        elif self.use_label_graph and self.label_graph_type == "dynamic_gat":
            self.label_graph_reasoner = DynamicLabelGATReasoner(
                num_labels=num_labels,
                feature_dim=feature_dim,
                dropout=dropout,
            )
        elif self.use_label_graph and self.label_graph_type == "label_transformer":
            self.label_graph_reasoner = LabelTransformerReasoner(
                num_labels=num_labels,
                feature_dim=feature_dim,
                dropout=dropout,
                num_heads=label_graph_heads,
            )
        elif self.use_label_graph and self.label_graph_type == "low_rank_graph":
            self.label_graph_reasoner = LowRankLabelGraphReasoner(
                num_labels=num_labels,
                feature_dim=feature_dim,
                dropout=dropout,
                rank=label_graph_rank,
            )
        elif self.use_label_graph and self.label_graph_type == "cosine_graph":
            self.label_graph_reasoner = CosineLabelGraphReasoner(
                num_labels=num_labels,
                feature_dim=feature_dim,
                dropout=dropout,
            )
        elif self.use_label_graph and self.label_graph_type == "label_mlp_mixer":
            self.label_graph_reasoner = LabelMLPMixerReasoner(
                num_labels=num_labels,
                feature_dim=feature_dim,
                dropout=dropout,
            )
        elif self.use_label_graph and self.label_graph_type == "label_hypergraph":
            self.label_graph_reasoner = LabelHypergraphReasoner(
                num_labels=num_labels,
                feature_dim=feature_dim,
                dropout=dropout,
                num_hyperedges=label_hypergraph_edges,
            )
        self.classifiers = nn.ModuleList([nn.Linear(feature_dim, 1) for _ in range(num_labels)])

    def encode_instances(self, images: torch.Tensor) -> torch.Tensor:
        return self.instance_encoder(images)

    def _mean_pool(self, features: torch.Tensor, mask: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        weights = mask.to(dtype=features.dtype)
        denom = weights.sum(dim=1, keepdim=True).clamp_min(1.0)
        bag_embed = torch.sum(features * weights.unsqueeze(-1), dim=1) / denom
        label_embeds = bag_embed.unsqueeze(1).expand(-1, self.num_labels, -1)
        attention = (weights / denom).unsqueeze(1).expand(-1, self.num_labels, -1)
        return label_embeds, attention

    def _pool_instances(self, features: torch.Tensor, mask: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        if self.pooling_type == "mean":
            return self._mean_pool(features, mask)

        if self.mil_pool is None:
            raise RuntimeError("mil_pool 未初始化")

        bag_embeds, attention = self.mil_pool(features, mask)
        if self.pooling_type == "shared_attention":
            bag_embeds = bag_embeds.unsqueeze(1).expand(-1, self.num_labels, -1)
            attention = attention.unsqueeze(1).expand(-1, self.num_labels, -1)
        return bag_embeds, attention

    def forward(self, images: torch.Tensor, mask: torch.Tensor) -> dict[str, torch.Tensor]:
        features = self.encode_instances(images)
        bag_embeds, attention = self._pool_instances(features, mask)
        if self.label_graph_reasoner is not None:
            refined_embeds, label_graph = self.label_graph_reasoner(bag_embeds)
        else:
            refined_embeds = bag_embeds
            label_graph = torch.eye(self.num_labels, device=features.device, dtype=features.dtype)

        logits = []
        for label_index in range(self.num_labels):
            logits.append(self.classifiers[label_index](refined_embeds[:, label_index, :]).squeeze(-1))

        return {
            "logits": torch.stack(logits, dim=1),
            "attention": attention,
            "instance_features": features,
            "label_graph": label_graph,
        }
