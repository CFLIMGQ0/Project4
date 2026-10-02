"""ACPE 与下游标签证据组织的轻量研究变体。"""

from __future__ import annotations

import torch
import torch.nn as nn

from exp_8.models import Exp12AProCoPEWatchCrossAttentionTextCNNModel, _safe_transformer_encoder_with_mask
from model.common.pooling import masked_softmax


class LabelPositionInteractionMixin:
    """将同一 ACPE 坐标以一种可控方式接入标签级视觉表征。"""

    def __init__(self, *, interaction_variant: str = "centroid", **kwargs):
        super().__init__(**kwargs)
        if interaction_variant not in {"centroid", "attention_bias"}:
            raise ValueError(f"未知标签位置交互变体：{interaction_variant}")
        self.interaction_variant = interaction_variant
        if interaction_variant == "centroid":
            self.position_summary = nn.Linear(1, self.feature_dim, bias=False)
            nn.init.zeros_(self.position_summary.weight)
        else:
            self.label_position_weights = nn.Parameter(torch.zeros(self.num_labels))

    def encode_long_mil(
        self,
        images: torch.Tensor,
        mask: torch.Tensor,
        instance_indices: torch.Tensor | None = None,
        original_image_counts: torch.Tensor | None = None,
    ):
        features, extra_outputs = self.encode_instances(images, mask)
        features, attention_bias, position_outputs = self._encode_position(
            features, mask, instance_indices, original_image_counts
        )
        if attention_bias is None:
            context_features = self.context_encoder(features, src_key_padding_mask=~mask)
        else:
            batch_size, num_heads, num_instances, _ = attention_bias.shape
            key_invalid = (~mask)[:, None, None, :]
            attention_bias = attention_bias.masked_fill(key_invalid, -1e4)
            encoder_mask = attention_bias.reshape(batch_size * num_heads, num_instances, num_instances)
            if getattr(self, "_safe_attention_mask_path", False):
                context_features = _safe_transformer_encoder_with_mask(
                    self.context_encoder, features, encoder_mask
                )
            else:
                context_features = self.context_encoder(features, mask=encoder_mask)
        context_features = context_features * mask.unsqueeze(-1).to(dtype=context_features.dtype)
        coordinate = position_outputs["apro_context_coordinates"]

        if self.interaction_variant == "centroid":
            bag_embeds, attention = self.mil_pool(context_features, mask)
            centroid = torch.einsum("blt,bt->bl", attention, coordinate).unsqueeze(-1)
            bag_embeds = bag_embeds + self.position_summary(centroid)
            position_outputs["label_position_centroid"] = centroid.squeeze(-1)
        else:
            gated = self.mil_pool.attn.v(context_features) * self.mil_pool.attn.u(context_features)
            score = self.mil_pool.attn.w(gated).transpose(1, 2)
            score = score + self.label_position_weights[None, :, None] * (coordinate[:, None, :] - 0.5)
            attention = masked_softmax(score, mask=mask.unsqueeze(1), dim=-1)
            bag_embeds = torch.einsum("blt,btd->bld", attention, context_features)

        label_embeds, graph_outputs = self.refine_labels(bag_embeds)
        extra_outputs.update(position_outputs)
        extra_outputs.update(graph_outputs)
        return context_features, label_embeds, attention, extra_outputs


class LabelPositionInteractionModel(
    LabelPositionInteractionMixin, Exp12AProCoPEWatchCrossAttentionTextCNNModel
):
    """完整监督数据集的交互变体。"""
