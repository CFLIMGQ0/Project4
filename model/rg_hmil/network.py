from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from model.common import MultiLabelAttentionMIL
from model.gastro_label_graph_mil.modules import InstanceEncoder, LabelGraphReasoner

from .modules import (
    AnatomicalInstanceGrouper,
    ConditionalHierarchicalLabelGraphReasoner,
    InstanceRelevancePredictor,
    RelevanceAwareMultiLabelAttention,
)


class RGHMIL(nn.Module):
    """报告引导的层次化多标签 MIL。"""

    def __init__(
        self,
        backbone_name: str = "convnext_tiny",
        pretrained: bool = True,
        freeze_stages: int = 0,
        feature_dim: int = 512,
        attn_dim: int = 256,
        num_labels: int = 8,
        num_regions: int = 6,
        condition_dim: int = 128,
        dropout: float = 0.2,
        use_text_guidance: bool = True,
        use_region_grouping: bool = True,
        use_conditional_graph: bool = True,
        use_hierarchy: bool = True,
    ) -> None:
        super().__init__()
        self.num_labels = num_labels
        self.use_text_guidance = use_text_guidance
        self.use_region_grouping = use_region_grouping
        self.use_conditional_graph = use_conditional_graph
        self.use_hierarchy = use_hierarchy

        self.instance_encoder = InstanceEncoder(
            backbone_name=backbone_name,
            pretrained=pretrained,
            freeze_stages=freeze_stages,
            feature_dim=feature_dim,
            dropout=dropout,
        )
        self.relevance_predictor = InstanceRelevancePredictor(
            feature_dim=feature_dim,
            hidden_dim=256,
            num_regions=num_regions,
            dropout=0.1,
        )
        if use_text_guidance:
            self.global_attn = RelevanceAwareMultiLabelAttention(
                in_dim=feature_dim,
                attn_dim=attn_dim,
                num_labels=num_labels,
                dropout=0.1,
            )
        else:
            self.global_attn = MultiLabelAttentionMIL(
                in_dim=feature_dim,
                attn_dim=attn_dim,
                num_labels=num_labels,
                dropout=0.1,
            )

        if use_region_grouping:
            self.region_aggregator = AnatomicalInstanceGrouper(
                in_dim=feature_dim,
                attn_dim=attn_dim,
                num_labels=num_labels,
                num_regions=num_regions,
                condition_dim=condition_dim,
                dropout=0.1,
            )
        else:
            self.region_aggregator = None

        if use_conditional_graph:
            self.label_graph_reasoner = ConditionalHierarchicalLabelGraphReasoner(
                num_labels=num_labels,
                feature_dim=feature_dim,
                condition_dim=condition_dim,
                dropout=dropout,
            )
        else:
            self.label_graph_reasoner = LabelGraphReasoner(
                num_labels=num_labels,
                feature_dim=feature_dim,
                dropout=dropout,
            )

        self.classifiers = nn.ModuleList([nn.Linear(feature_dim, 1) for _ in range(num_labels)])

    def encode_instances(self, images: torch.Tensor) -> torch.Tensor:
        return self.instance_encoder(images)

    def _classify_label_embeds(self, embeds: torch.Tensor) -> torch.Tensor:
        logits = []
        for label_index, classifier in enumerate(self.classifiers):
            logits.append(classifier(embeds[:, label_index, :]).squeeze(-1))
        return torch.stack(logits, dim=1)

    def forward(
        self,
        images: torch.Tensor,
        mask: torch.Tensor,
        pseudo_region_labels: torch.Tensor | None = None,
        pseudo_relevance: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        features = self.encode_instances(images)
        region_logits, relevance_logits = self.relevance_predictor(features, mask)

        if self.use_text_guidance:
            global_embed, global_attn = self.global_attn(features, mask, relevance_logits)
        else:
            global_embed, global_attn = self.global_attn(features, mask)

        condition_vector = None
        if self.region_aggregator is not None:
            regional_embed, condition_vector = self.region_aggregator(features, mask, region_logits)
            fused_embed = global_embed + regional_embed
        else:
            fused_embed = global_embed

        pre_logits = self._classify_label_embeds(fused_embed)
        if isinstance(self.label_graph_reasoner, ConditionalHierarchicalLabelGraphReasoner):
            refined_embed, dynamic_graph = self.label_graph_reasoner(
                fused_embed,
                condition_vector=condition_vector,
                all_logits=pre_logits if self.use_hierarchy else None,
            )
        else:
            refined_embed, dynamic_graph = self.label_graph_reasoner(fused_embed)

        logits = self._classify_label_embeds(refined_embed)

        aux_losses: dict[str, torch.Tensor] = {}
        if self.training and pseudo_region_labels is not None:
            aux_losses["region_cls"] = F.cross_entropy(
                region_logits.reshape(-1, region_logits.shape[-1]),
                pseudo_region_labels.reshape(-1),
                ignore_index=-100,
            )
        if self.training and pseudo_relevance is not None:
            valid = pseudo_relevance >= 0
            if valid.any():
                aux_losses["relevance"] = F.binary_cross_entropy_with_logits(
                    relevance_logits.squeeze(-1)[valid],
                    pseudo_relevance[valid],
                )

        return {
            "logits": logits,
            "attention": global_attn,
            "instance_features": features,
            "label_graph": dynamic_graph,
            "region_logits": region_logits,
            "relevance_logits": relevance_logits,
            "aux_losses": aux_losses,
        }


__all__ = ["RGHMIL"]
