from __future__ import annotations

import math

import torch
import torch.nn as nn

from .common import GastroSOTAInstanceEncoder, LabelwiseClassifier, ensure_num_heads


class GastroTransMILSOTA(nn.Module):
    """TransMIL 思路的标签 token 相关建模模型。"""

    def __init__(
        self,
        backbone_name: str = "convnext_tiny",
        pretrained: bool = True,
        freeze_stages: int = 1,
        feature_dim: int = 512,
        hidden_dim: int = 256,
        num_labels: int = 3,
        dropout: float = 0.2,
        num_heads: int = 8,
        num_layers: int = 2,
    ) -> None:
        super().__init__()
        self.num_labels = num_labels
        self.instance_encoder = GastroSOTAInstanceEncoder(
            backbone_name=backbone_name,
            pretrained=pretrained,
            freeze_stages=freeze_stages,
            feature_dim=feature_dim,
            dropout=dropout,
        )
        effective_heads = ensure_num_heads(feature_dim, num_heads)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=feature_dim,
            nhead=effective_heads,
            dim_feedforward=feature_dim * 2,
            dropout=dropout,
            batch_first=True,
            norm_first=True,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=max(1, int(num_layers)))
        self.label_tokens = nn.Parameter(torch.randn(num_labels, feature_dim) * 0.02)
        self.bag_classifier = LabelwiseClassifier(feature_dim, hidden_dim, num_labels, dropout)

    def forward(self, images: torch.Tensor, mask: torch.Tensor) -> dict[str, torch.Tensor]:
        features = self.instance_encoder(images)
        label_tokens = self.label_tokens.unsqueeze(0).expand(features.size(0), -1, -1)
        sequence = torch.cat([label_tokens, features], dim=1)
        padding_mask = torch.cat(
            [
                torch.zeros(features.size(0), self.num_labels, device=mask.device, dtype=torch.bool),
                ~mask.to(dtype=torch.bool),
            ],
            dim=1,
        )
        encoded = self.transformer(sequence, src_key_padding_mask=padding_mask)
        label_states = encoded[:, : self.num_labels, :]
        instance_states = encoded[:, self.num_labels :, :]

        attention_logits = torch.einsum("bld,bnd->bln", label_states, instance_states) / math.sqrt(features.size(-1))
        attention_logits = attention_logits.masked_fill(
            ~mask.unsqueeze(1).to(dtype=torch.bool),
            torch.finfo(attention_logits.dtype).min,
        )
        attention = torch.softmax(attention_logits, dim=-1)
        attention = attention * mask.unsqueeze(1).to(dtype=attention.dtype)
        attention = attention / attention.sum(dim=-1, keepdim=True).clamp_min(1e-12)

        refined_embeds = label_states + torch.einsum("bln,bnd->bld", attention, instance_states)
        logits = self.bag_classifier(refined_embeds)
        return {
            "logits": logits,
            "attention": attention,
            "instance_features": instance_states,
        }
