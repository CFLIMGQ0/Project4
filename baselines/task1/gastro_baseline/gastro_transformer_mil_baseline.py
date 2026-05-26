from __future__ import annotations

import torch
import torch.nn as nn

from model.common import MultiLabelAttentionMIL

from .common import BaseGastroBaseline


def _resolve_num_heads(feature_dim: int, requested_heads: int) -> int:
    effective_heads = max(1, min(int(requested_heads), int(feature_dim)))
    while feature_dim % effective_heads != 0 and effective_heads > 1:
        effective_heads -= 1
    return effective_heads


class GastroTransformerMILBaseline(BaseGastroBaseline):
    """先做实例上下文建模，再做标签注意力聚合的胃镜 baseline。"""

    def __init__(
        self,
        backbone_name: str = "resnet50",
        pretrained: bool = True,
        freeze_stages: int = 0,
        feature_dim: int = 512,
        attn_dim: int = 256,
        num_labels: int = 3,
        dropout: float = 0.2,
        num_heads: int = 8,
        num_layers: int = 2,
    ) -> None:
        super().__init__(
            backbone_name=backbone_name,
            pretrained=pretrained,
            freeze_stages=freeze_stages,
            feature_dim=feature_dim,
            num_labels=num_labels,
            dropout=dropout,
        )
        effective_heads = _resolve_num_heads(feature_dim, num_heads)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=feature_dim,
            nhead=effective_heads,
            dim_feedforward=feature_dim * 2,
            dropout=dropout,
            batch_first=True,
            norm_first=True,
        )
        self.context_encoder = nn.TransformerEncoder(encoder_layer, num_layers=max(1, int(num_layers)))
        self.mil_pool = MultiLabelAttentionMIL(
            in_dim=feature_dim,
            attn_dim=attn_dim,
            num_labels=num_labels,
            dropout=dropout,
        )
        self.classifiers = nn.ModuleList([nn.Linear(feature_dim, 1) for _ in range(num_labels)])

    def forward(self, images: torch.Tensor, mask: torch.Tensor) -> dict[str, torch.Tensor]:
        features = self.encode_instances(images)
        contextual_features = self.context_encoder(
            features,
            src_key_padding_mask=~mask.to(dtype=torch.bool),
        )
        bag_embeds, attention = self.mil_pool(contextual_features, mask)

        logits = []
        for label_index in range(self.num_labels):
            logits.append(self.classifiers[label_index](bag_embeds[:, label_index, :]).squeeze(-1))

        return {
            "logits": torch.stack(logits, dim=1),
            "attention": attention,
            "instance_features": contextual_features,
        }
