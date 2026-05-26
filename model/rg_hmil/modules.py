from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from model.common import GatedAttention, MultiLabelAttentionMIL, masked_softmax
from model.gastro_label_graph_mil.modules import LabelGraphReasoner


class InstanceRelevancePredictor(nn.Module):
    def __init__(
        self,
        feature_dim: int = 512,
        hidden_dim: int = 256,
        num_regions: int = 6,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.shared = nn.Sequential(
            nn.Linear(feature_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
        )
        self.region_head = nn.Linear(hidden_dim, num_regions)
        self.relevance_head = nn.Linear(hidden_dim, 1)

    def forward(self, instance_features: torch.Tensor, mask: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        hidden = self.shared(instance_features)
        region_logits = self.region_head(hidden)
        relevance_logits = self.relevance_head(hidden)
        valid_mask = mask.unsqueeze(-1).to(dtype=torch.bool)
        region_logits = region_logits.masked_fill(~valid_mask, 0.0)
        relevance_logits = relevance_logits.masked_fill(~valid_mask, 0.0)
        return region_logits, relevance_logits


class RelevanceAwareMultiLabelAttention(nn.Module):
    def __init__(self, in_dim: int, attn_dim: int, num_labels: int, dropout: float = 0.1) -> None:
        super().__init__()
        self.attn = GatedAttention(in_dim=in_dim, attn_dim=attn_dim, num_heads=num_labels, dropout=dropout)
        self.alpha = nn.Parameter(torch.zeros(num_labels))

    def forward(
        self,
        x: torch.Tensor,
        mask: torch.Tensor,
        relevance_logits: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        gate = self.attn.v(x) * self.attn.u(x)
        attn_logits = self.attn.w(gate).transpose(1, 2)

        if relevance_logits is not None:
            relevance = torch.sigmoid(relevance_logits.squeeze(-1))
            relevance_bias = torch.log(relevance.clamp_min(1e-6))
            attn_logits = attn_logits + self.alpha.view(1, -1, 1) * relevance_bias.unsqueeze(1)

        attn_weights = masked_softmax(attn_logits, mask.unsqueeze(1), dim=-1)
        bag_embed = torch.einsum("bln,bnd->bld", attn_weights, x)
        return bag_embed, attn_weights


class AnatomicalInstanceGrouper(nn.Module):
    def __init__(
        self,
        in_dim: int = 512,
        attn_dim: int = 256,
        num_labels: int = 8,
        num_regions: int = 6,
        condition_dim: int = 128,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.num_regions = num_regions
        self.region_attn = MultiLabelAttentionMIL(in_dim=in_dim, attn_dim=attn_dim, num_labels=num_labels, dropout=dropout)
        self.region_embeddings = nn.Embedding(num_regions, in_dim)
        self.condition_mlp = nn.Sequential(
            nn.Linear(num_regions * in_dim + num_regions, condition_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(condition_dim, condition_dim),
        )
        self.region_weight = nn.Linear(in_dim, 1)

    def forward(
        self,
        instance_features: torch.Tensor,
        mask: torch.Tensor,
        region_logits: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        batch_size, _, feature_dim = instance_features.shape
        region_probs = F.softmax(region_logits, dim=-1)
        region_probs = region_probs * mask.unsqueeze(-1).to(dtype=region_probs.dtype)

        region_dist = region_probs.sum(dim=1)
        region_dist = region_dist / region_dist.sum(dim=-1, keepdim=True).clamp_min(1e-8)

        region_embeds: list[torch.Tensor] = []
        region_mean_feats: list[torch.Tensor] = []
        region_strengths: list[torch.Tensor] = []

        for region_index in range(self.num_regions):
            region_mask_soft = region_probs[:, :, region_index]
            region_strength = region_mask_soft.sum(dim=1)
            hard_mask = region_mask_soft > 0.1
            fallback_mask = ~hard_mask.any(dim=1, keepdim=True)
            hard_mask = torch.where(fallback_mask, mask, hard_mask)

            region_token = self.region_embeddings.weight[region_index].view(1, 1, feature_dim)
            region_features = instance_features + region_token
            region_embed, _ = self.region_attn(region_features, hard_mask)
            region_embeds.append(region_embed)

            weighted_feat = torch.einsum("bn,bnd->bd", region_mask_soft, instance_features)
            weighted_feat = weighted_feat / region_strength.unsqueeze(-1).clamp_min(1e-8)
            region_mean_feats.append(weighted_feat)
            region_strengths.append(region_strength)

        region_embeds_tensor = torch.stack(region_embeds, dim=1)
        region_mean_feats_tensor = torch.stack(region_mean_feats, dim=1)
        region_strength_tensor = torch.stack(region_strengths, dim=1)

        condition_input = torch.cat(
            [region_mean_feats_tensor.reshape(batch_size, -1), region_dist],
            dim=-1,
        )
        condition_vector = self.condition_mlp(condition_input)

        region_weight_logits = self.region_weight(region_mean_feats_tensor).squeeze(-1)
        valid_region_mask = region_strength_tensor > 1e-6
        region_weight_logits = region_weight_logits.masked_fill(
            ~valid_region_mask,
            torch.finfo(region_weight_logits.dtype).min,
        )
        region_weights = torch.softmax(region_weight_logits, dim=-1)
        region_weights = region_weights * valid_region_mask.to(dtype=region_weights.dtype)
        region_weights = region_weights / region_weights.sum(dim=-1, keepdim=True).clamp_min(1e-8)
        regional_embed = torch.einsum("br,brld->bld", region_weights, region_embeds_tensor)
        return regional_embed, condition_vector


class ConditionalHierarchicalLabelGraphReasoner(nn.Module):
    def __init__(
        self,
        num_labels: int = 8,
        feature_dim: int = 512,
        condition_dim: int = 128,
        dropout: float = 0.2,
    ) -> None:
        super().__init__()
        self.num_labels = num_labels
        self.feature_dim = feature_dim
        self.label_tokens = nn.Parameter(torch.randn(num_labels, feature_dim) * 0.02)
        self.refine = nn.Sequential(
            nn.Linear(feature_dim * 2, feature_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        self.graph_modulator = nn.Sequential(
            nn.Linear(condition_dim, num_labels * num_labels),
            nn.Sigmoid(),
        )
        self.register_buffer("hierarchy_parent_map", self._build_parent_map(num_labels))

    @staticmethod
    def _build_parent_map(num_labels: int) -> torch.Tensor:
        parent_map = torch.full((num_labels,), -1, dtype=torch.long)
        if num_labels >= 5:
            parent_map[3] = 2
            parent_map[4] = 2
        return parent_map

    def forward(
        self,
        bag_embeds: torch.Tensor,
        condition_vector: torch.Tensor | None = None,
        all_logits: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        batch_size = bag_embeds.shape[0]
        norm_tokens = F.normalize(self.label_tokens, dim=-1)
        graph_logits = torch.matmul(norm_tokens, norm_tokens.transpose(0, 1)) / math.sqrt(float(self.feature_dim))
        static_graph = torch.softmax(graph_logits, dim=-1)

        if condition_vector is not None:
            modulation = self.graph_modulator(condition_vector).view(batch_size, self.num_labels, self.num_labels)
            dynamic_graph = static_graph.unsqueeze(0) * modulation
        else:
            dynamic_graph = static_graph.unsqueeze(0).expand(batch_size, -1, -1)

        if all_logits is not None:
            gate_columns: list[torch.Tensor] = []
            for child_index in range(self.num_labels):
                parent_index = int(self.hierarchy_parent_map[child_index].item())
                if parent_index < 0:
                    gate_columns.append(
                        torch.ones(
                            batch_size,
                            1,
                            device=all_logits.device,
                            dtype=all_logits.dtype,
                        )
                    )
                else:
                    gate_columns.append(torch.sigmoid(all_logits[:, parent_index]).unsqueeze(-1))
            label_gate = torch.cat(gate_columns, dim=1)
            hierarchy_gate = label_gate.unsqueeze(2) * label_gate.unsqueeze(1)
            dynamic_graph = dynamic_graph * hierarchy_gate

        dynamic_graph = dynamic_graph / dynamic_graph.sum(dim=-1, keepdim=True).clamp_min(1e-8)
        propagated = torch.bmm(dynamic_graph, bag_embeds)
        refined = bag_embeds + self.refine(torch.cat([bag_embeds, propagated], dim=-1))
        return refined, dynamic_graph


__all__ = [
    "AnatomicalInstanceGrouper",
    "ConditionalHierarchicalLabelGraphReasoner",
    "InstanceRelevancePredictor",
    "LabelGraphReasoner",
    "RelevanceAwareMultiLabelAttention",
]
