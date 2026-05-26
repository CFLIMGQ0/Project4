from __future__ import annotations

from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F

from exp_4.models import Exp4BaseModel, TimePositionEncoding
from model.common import MultiLabelAttentionMIL


def _classify_with(classifiers: nn.ModuleList, label_embeds: torch.Tensor) -> torch.Tensor:
    logits = [
        classifiers[label_index](label_embeds[:, label_index, :]).squeeze(-1)
        for label_index in range(len(classifiers))
    ]
    return torch.stack(logits, dim=1)


def _attention_entropy_loss(attention: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    masked_attention = attention * mask.unsqueeze(1).to(dtype=attention.dtype)
    entropy = -(masked_attention.clamp_min(1e-8) * masked_attention.clamp_min(1e-8).log()).sum(dim=-1)
    return -entropy.mean()


class Exp6DualStreamLongMILModel(Exp4BaseModel):
    """原图-ROI 双路 Long-MIL。

    模型共享实例编码器和 long-context encoder，然后分别用全图 mask 与 ROI mask 聚合证据。
    ROI 分支通过按标签 gate 作为局部补充，不再替代全图上下文。
    """

    def __init__(
        self,
        *,
        num_layers: int = 2,
        num_heads: int = 4,
        hidden_dim: int = 1024,
        roi_gate_init: float = -1.0,
        use_type_embedding: bool = True,
        **kwargs: Any,
    ) -> None:
        kwargs["use_label_graph"] = bool(kwargs.get("use_label_graph", True))
        super().__init__(**kwargs)
        self.use_type_embedding = bool(use_type_embedding)
        self.position_encoding = TimePositionEncoding(self.feature_dim)
        self.type_embedding = nn.Embedding(2, self.feature_dim) if self.use_type_embedding else None
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=self.feature_dim,
            nhead=max(1, int(num_heads)),
            dim_feedforward=int(hidden_dim),
            dropout=float(kwargs.get("dropout", 0.2)),
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.context_encoder = nn.TransformerEncoder(encoder_layer, num_layers=max(1, int(num_layers)))
        self.roi_pool = MultiLabelAttentionMIL(
            in_dim=self.feature_dim,
            attn_dim=int(kwargs.get("attn_dim", 256)),
            num_labels=self.num_labels,
            dropout=float(kwargs.get("dropout", 0.2)),
        )
        self.roi_classifiers = nn.ModuleList([nn.Linear(self.feature_dim, 1) for _ in range(self.num_labels)])
        self.roi_gate = nn.Parameter(torch.full((self.num_labels,), float(roi_gate_init)))

    def forward(
        self,
        images: torch.Tensor,
        mask: torch.Tensor,
        instance_types: torch.Tensor | None = None,
        **kwargs: Any,
    ) -> dict[str, torch.Tensor]:
        del kwargs
        if instance_types is None:
            instance_types = torch.zeros_like(mask, dtype=torch.long)
        else:
            instance_types = instance_types.to(device=mask.device, dtype=torch.long).clamp(0, 1)

        full_mask = mask & (instance_types == 0)
        roi_mask = mask & (instance_types == 1)

        features, extra_outputs = self.encode_instances(images, mask)
        if self.type_embedding is not None:
            type_features = self.type_embedding(instance_types)
            features = features + type_features * mask.unsqueeze(-1).to(dtype=features.dtype)

        features = self.position_encoding(features, mask)
        context_features = self.context_encoder(features, src_key_padding_mask=~mask)
        context_features = context_features * mask.unsqueeze(-1).to(dtype=context_features.dtype)

        global_bag_embeds, global_attention = self.mil_pool(context_features, full_mask)
        global_label_embeds, graph_outputs = self.refine_labels(global_bag_embeds)
        global_logits = self.classify(global_label_embeds)

        roi_bag_embeds, roi_attention = self.roi_pool(context_features, roi_mask)
        roi_label_embeds, _ = self.refine_labels(roi_bag_embeds)
        roi_logits = _classify_with(self.roi_classifiers, roi_label_embeds)

        has_roi = roi_mask.any(dim=1).to(dtype=global_logits.dtype).unsqueeze(-1)
        gate = torch.sigmoid(self.roi_gate).view(1, self.num_labels)
        logits = global_logits + has_roi * gate * roi_logits

        attention_gate = gate.unsqueeze(-1)
        attention = global_attention + has_roi.unsqueeze(-1) * attention_gate * roi_attention
        attention = attention * mask.unsqueeze(1).to(dtype=attention.dtype)
        attention = attention / attention.sum(dim=-1, keepdim=True).clamp_min(1e-12)

        aux_losses = {
            "view_consistency": F.mse_loss(torch.sigmoid(logits), torch.sigmoid(global_logits)),
            "attention_entropy": _attention_entropy_loss(attention, mask),
        }

        extra_outputs.update(graph_outputs)
        extra_outputs.update(
            {
                "global_logits": global_logits,
                "roi_logits": roi_logits,
                "roi_gate": torch.sigmoid(self.roi_gate),
                "roi_attention": roi_attention,
                "global_attention": global_attention,
                "aux_losses": aux_losses,
            }
        )
        return self.build_outputs(
            logits=logits,
            attention=attention,
            features=context_features,
            extra_outputs=extra_outputs,
        )


EXP6_CLASS_REGISTRY = {
    "exp6_dual_stream_long_mil": Exp6DualStreamLongMILModel,
}
EXP6_MODEL_NAMES = tuple(EXP6_CLASS_REGISTRY.keys())


def build_exp6_model(model_name: str, **kwargs: Any) -> nn.Module:
    base_keys = {
        "backbone_name",
        "pretrained",
        "freeze_stages",
        "feature_dim",
        "attn_dim",
        "num_labels",
        "dropout",
        "use_label_graph",
        "encoder_chunk_size",
        "use_quality_gate",
        "num_layers",
        "num_heads",
        "hidden_dim",
        "roi_gate_init",
        "use_type_embedding",
    }
    model_kwargs = {key: value for key, value in kwargs.items() if key in base_keys}
    model_cls = EXP6_CLASS_REGISTRY.get(model_name)
    if model_cls is None:
        raise ValueError(f"未知 exp_6 模型名: {model_name}")
    return model_cls(**model_kwargs)
