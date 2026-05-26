from __future__ import annotations

from typing import Any

import torch
import torch.nn as nn

from exp_4.models import Exp4BaseModel, TimePositionEncoding
from model.gastro_label_graph_mil.modules import LabelHypergraphReasoner


STRUCTURED_FIELD_NAMES = ("reportTitle", "age", "sex", "hp", "operationValue")
STRUCTURED_CATEGORICAL_FIELDS = ("reportTitle", "sex", "hp", "operationValue")
STRUCTURED_NUMERIC_FIELDS = ("age",)
STRUCTURED_FIELD_TO_INDEX = {name: index for index, name in enumerate(STRUCTURED_FIELD_NAMES)}
STRUCTURED_CATEGORICAL_TO_INDEX = {name: index for index, name in enumerate(STRUCTURED_CATEGORICAL_FIELDS)}
STRUCTURED_NUMERIC_TO_INDEX = {name: index for index, name in enumerate(STRUCTURED_NUMERIC_FIELDS)}


def _normalize_structured_fields(raw_fields: Any) -> tuple[str, ...]:
    if raw_fields is None:
        return ()
    if isinstance(raw_fields, str):
        fields = [item.strip() for item in raw_fields.split(",") if item.strip()]
    else:
        fields = [str(item).strip() for item in raw_fields if str(item).strip()]
    unknown = [field for field in fields if field not in STRUCTURED_FIELD_TO_INDEX]
    if unknown:
        raise ValueError(f"未知结构化字段: {unknown}")
    return tuple(dict.fromkeys(fields))


class StructuredFieldEncoder(nn.Module):
    def __init__(
        self,
        *,
        category_sizes: dict[str, int],
        selected_fields: tuple[str, ...],
        field_embed_dim: int,
        output_dim: int,
        dropout: float,
        structured_dropout: float,
        modality_dropout: float,
    ) -> None:
        super().__init__()
        self.selected_fields = tuple(selected_fields)
        self.structured_dropout = float(structured_dropout)
        self.modality_dropout = float(modality_dropout)

        self.category_embeddings = nn.ModuleDict()
        for field_name in STRUCTURED_CATEGORICAL_FIELDS:
            vocab_size = max(2, int(category_sizes.get(field_name, 2)))
            self.category_embeddings[field_name] = nn.Embedding(vocab_size, field_embed_dim)

        self.numeric_projections = nn.ModuleDict(
            {
                "age": nn.Sequential(
                    nn.Linear(1, field_embed_dim),
                    nn.GELU(),
                    nn.LayerNorm(field_embed_dim),
                )
            }
        )
        self.fuse = nn.Sequential(
            nn.LayerNorm(field_embed_dim),
            nn.Linear(field_embed_dim, output_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(output_dim, output_dim),
        )

        selected_mask = torch.zeros(len(STRUCTURED_FIELD_NAMES), dtype=torch.float32)
        for field_name in self.selected_fields:
            selected_mask[STRUCTURED_FIELD_TO_INDEX[field_name]] = 1.0
        self.register_buffer("selected_field_mask", selected_mask, persistent=False)

    @property
    def has_selected_fields(self) -> bool:
        return bool(self.selected_fields)

    def _field_weight(
        self,
        structured_mask: torch.Tensor,
        field_name: str,
    ) -> torch.Tensor:
        field_index = STRUCTURED_FIELD_TO_INDEX[field_name]
        weight = structured_mask[:, field_index].to(dtype=torch.float32)
        weight = weight * self.selected_field_mask[field_index].to(dtype=weight.dtype)
        if self.training and self.structured_dropout > 0.0:
            keep = torch.rand_like(weight) >= self.structured_dropout
            weight = weight * keep.to(dtype=weight.dtype)
        return weight

    def forward(
        self,
        structured_categorical: torch.Tensor | None,
        structured_numeric: torch.Tensor | None,
        structured_mask: torch.Tensor | None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if structured_mask is None:
            if structured_categorical is not None:
                batch_size = structured_categorical.shape[0]
                device = structured_categorical.device
            elif structured_numeric is not None:
                batch_size = structured_numeric.shape[0]
                device = structured_numeric.device
            else:
                batch_size = 1
                device = self.selected_field_mask.device
            output_dim = self.fuse[-1].out_features
            return (
                torch.zeros(batch_size, output_dim, device=device),
                torch.zeros(batch_size, device=device),
            )

        batch_size = structured_mask.shape[0]
        device = structured_mask.device
        field_embeddings: list[torch.Tensor] = []
        field_weights: list[torch.Tensor] = []

        if structured_categorical is not None:
            for field_name in STRUCTURED_CATEGORICAL_FIELDS:
                cat_index = STRUCTURED_CATEGORICAL_TO_INDEX[field_name]
                values = structured_categorical[:, cat_index].clamp_min(0)
                vocab_size = self.category_embeddings[field_name].num_embeddings
                values = values.clamp_max(vocab_size - 1)
                weight = self._field_weight(structured_mask, field_name)
                field_embeddings.append(self.category_embeddings[field_name](values) * weight.unsqueeze(-1))
                field_weights.append(weight)

        if structured_numeric is not None:
            age_index = STRUCTURED_NUMERIC_TO_INDEX["age"]
            age_values = structured_numeric[:, age_index : age_index + 1].to(dtype=torch.float32)
            weight = self._field_weight(structured_mask, "age")
            field_embeddings.append(self.numeric_projections["age"](age_values) * weight.unsqueeze(-1))
            field_weights.append(weight)

        if not field_embeddings:
            output_dim = self.fuse[-1].out_features
            return (
                torch.zeros(batch_size, output_dim, device=device),
                torch.zeros(batch_size, device=device),
            )

        stacked_embeddings = torch.stack(field_embeddings, dim=1)
        stacked_weights = torch.stack(field_weights, dim=1)
        if self.training and self.modality_dropout > 0.0:
            keep_modality = (torch.rand(batch_size, device=device) >= self.modality_dropout).to(dtype=stacked_weights.dtype)
            stacked_weights = stacked_weights * keep_modality.unsqueeze(-1)
            stacked_embeddings = stacked_embeddings * keep_modality.view(batch_size, 1, 1)

        denom = stacked_weights.sum(dim=1, keepdim=True).clamp_min(1.0)
        pooled = stacked_embeddings.sum(dim=1) / denom
        active_mask = (stacked_weights.sum(dim=1) > 0).to(dtype=pooled.dtype)
        structured_embed = self.fuse(pooled) * active_mask.unsqueeze(-1)
        return structured_embed, active_mask


class Exp8StructuredLateGateMILModel(Exp4BaseModel):
    """Long-MIL 图像分支 + 结构化字段 label-wise gated late fusion。"""

    def __init__(
        self,
        *,
        num_layers: int = 2,
        num_heads: int = 4,
        hidden_dim: int = 1024,
        structured_fields: list[str] | tuple[str, ...] | str | None = None,
        structured_category_sizes: dict[str, int] | None = None,
        structured_field_embed_dim: int = 64,
        structured_dropout: float = 0.2,
        modality_dropout: float = 0.15,
        label_graph_type: str = "label_hypergraph",
        label_hypergraph_edges: int = 2,
        **kwargs: Any,
    ) -> None:
        kwargs["use_label_graph"] = bool(kwargs.get("use_label_graph", True))
        super().__init__(**kwargs)
        self.label_graph_type = str(label_graph_type).strip().lower() or "label_hypergraph"
        if self.label_graph_type not in {"learnable", "label_hypergraph"}:
            raise ValueError(f"exp8_structured_late_gate_mil 不支持的 label_graph_type: {label_graph_type}")
        if self.use_label_graph and self.label_graph_type == "label_hypergraph":
            self.label_graph_reasoner = LabelHypergraphReasoner(
                num_labels=self.num_labels,
                feature_dim=self.feature_dim,
                dropout=float(kwargs.get("dropout", 0.2)),
                num_hyperedges=int(label_hypergraph_edges),
            )
        self.selected_structured_fields = _normalize_structured_fields(structured_fields)
        self.position_encoding = TimePositionEncoding(self.feature_dim)
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
        self.structured_encoder = StructuredFieldEncoder(
            category_sizes=structured_category_sizes or {},
            selected_fields=self.selected_structured_fields,
            field_embed_dim=int(structured_field_embed_dim),
            output_dim=self.feature_dim,
            dropout=float(kwargs.get("dropout", 0.2)),
            structured_dropout=float(structured_dropout),
            modality_dropout=float(modality_dropout),
        )
        self.structured_label_proj = nn.Linear(self.feature_dim, self.num_labels * self.feature_dim)
        self.fusion_gate = nn.Sequential(
            nn.LayerNorm(self.feature_dim * 2),
            nn.Linear(self.feature_dim * 2, self.feature_dim),
            nn.GELU(),
            nn.Dropout(float(kwargs.get("dropout", 0.2))),
            nn.Linear(self.feature_dim, 1),
        )

    def forward(
        self,
        images: torch.Tensor,
        mask: torch.Tensor,
        structured_categorical: torch.Tensor | None = None,
        structured_numeric: torch.Tensor | None = None,
        structured_mask: torch.Tensor | None = None,
        **kwargs: Any,
    ) -> dict[str, torch.Tensor]:
        del kwargs
        features, extra_outputs = self.encode_instances(images, mask)
        features = self.position_encoding(features, mask)
        context_features = self.context_encoder(features, src_key_padding_mask=~mask)
        context_features = context_features * mask.unsqueeze(-1).to(dtype=context_features.dtype)
        bag_embeds, attention = self.mil_pool(context_features, mask)
        label_embeds, graph_outputs = self.refine_labels(bag_embeds)
        extra_outputs.update(graph_outputs)

        if not self.structured_encoder.has_selected_fields:
            logits = self.classify(label_embeds)
            return self.build_outputs(
                logits=logits,
                attention=attention,
                features=context_features,
                extra_outputs=extra_outputs,
            )

        batch_size = images.shape[0]
        if structured_categorical is None:
            structured_categorical = torch.zeros(
                batch_size,
                len(STRUCTURED_CATEGORICAL_FIELDS),
                dtype=torch.long,
                device=images.device,
            )
        if structured_numeric is None:
            structured_numeric = torch.zeros(
                batch_size,
                len(STRUCTURED_NUMERIC_FIELDS),
                dtype=context_features.dtype,
                device=images.device,
            )
        if structured_mask is None:
            structured_mask = torch.zeros(
                batch_size,
                len(STRUCTURED_FIELD_NAMES),
                dtype=context_features.dtype,
                device=images.device,
            )

        structured_embed, structured_active = self.structured_encoder(
            structured_categorical=structured_categorical,
            structured_numeric=structured_numeric,
            structured_mask=structured_mask,
        )
        structured_label_embeds = self.structured_label_proj(structured_embed)
        structured_label_embeds = structured_label_embeds.view(
            structured_embed.shape[0],
            self.num_labels,
            self.feature_dim,
        )
        gate_inputs = torch.cat([label_embeds, structured_label_embeds], dim=-1)
        gates = torch.sigmoid(self.fusion_gate(gate_inputs)).squeeze(-1)
        gates = gates * structured_active.unsqueeze(-1).to(dtype=gates.dtype)
        fused_label_embeds = label_embeds + gates.unsqueeze(-1) * structured_label_embeds
        logits = self.classify(fused_label_embeds)

        extra_outputs.update(
            {
                "structured_gates": gates,
                "structured_active": structured_active,
                "aux_losses": {
                    "structured_gate_l1": gates.mean(),
                },
            }
        )
        return self.build_outputs(
            logits=logits,
            attention=attention,
            features=context_features,
            extra_outputs=extra_outputs,
        )


EXP8_CLASS_REGISTRY = {
    "exp8_structured_late_gate_mil": Exp8StructuredLateGateMILModel,
}
EXP8_MODEL_NAMES = tuple(EXP8_CLASS_REGISTRY.keys())


def build_exp8_model(model_name: str, **kwargs: Any) -> nn.Module:
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
        "structured_fields",
        "structured_category_sizes",
        "structured_field_embed_dim",
        "structured_dropout",
        "modality_dropout",
        "label_graph_type",
        "label_hypergraph_edges",
    }
    model_kwargs = {key: value for key, value in kwargs.items() if key in base_keys}
    model_cls = EXP8_CLASS_REGISTRY.get(model_name)
    if model_cls is None:
        raise ValueError(f"未知 exp_8 模型名: {model_name}")
    return model_cls(**model_kwargs)
