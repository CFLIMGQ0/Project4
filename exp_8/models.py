from __future__ import annotations

import hashlib
import math
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F

from exp_4.models import Exp4BaseModel, TimePositionEncoding
from model.gastro_label_graph_mil.modules import LabelHypergraphReasoner


STRUCTURED_FIELD_NAMES = ("reportTitle", "age", "sex", "hp", "operationValue")
STRUCTURED_CATEGORICAL_FIELDS = ("reportTitle", "sex", "hp", "operationValue")
STRUCTURED_NUMERIC_FIELDS = ("age",)
STRUCTURED_FIELD_TO_INDEX = {name: index for index, name in enumerate(STRUCTURED_FIELD_NAMES)}
STRUCTURED_CATEGORICAL_TO_INDEX = {name: index for index, name in enumerate(STRUCTURED_CATEGORICAL_FIELDS)}
STRUCTURED_NUMERIC_TO_INDEX = {name: index for index, name in enumerate(STRUCTURED_NUMERIC_FIELDS)}

LABEL_PROTOTYPE_TERMS = (
    ("食管黏膜下隆起", "食管黏膜下肿物", "食管SMT", "食管隆起性病变"),
    ("食管黏膜病变", "食管肿物", "食管占位", "食管新生物", "食管早癌可疑"),
    ("慢性胃炎", "活动性胃炎", "萎缩性胃炎", "糜烂性胃炎", "胆汁反流性胃炎"),
)


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


def _stable_unit_vector(text: str, dim: int) -> torch.Tensor:
    digest = hashlib.sha1(text.encode("utf-8")).digest()
    seed = int.from_bytes(digest[:8], byteorder="little", signed=False) % (2**31 - 1)
    generator = torch.Generator()
    generator.manual_seed(seed)
    vector = torch.randn(int(dim), generator=generator)
    return F.normalize(vector, dim=0)


def _masked_mean(values: torch.Tensor, mask: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    weights = mask.to(dtype=values.dtype)
    denom = weights.sum(dim=1, keepdim=True).clamp_min(1.0)
    pooled = (values * weights.unsqueeze(-1)).sum(dim=1) / denom
    active = weights.sum(dim=1) > 0
    return pooled, active


def _safe_key_padding_mask(tokens: torch.Tensor, token_mask: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    safe_mask = token_mask.bool().clone()
    empty_rows = ~safe_mask.any(dim=1)
    if empty_rows.any():
        safe_mask[empty_rows, 0] = True
        tokens = tokens.clone()
        tokens[empty_rows, 0] = 0.0
    return tokens, ~safe_mask


def _masked_instance_softmax(scores: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    masked_scores = scores.masked_fill(~mask, -1e4)
    attention = torch.softmax(masked_scores, dim=-1)
    attention = attention * mask.to(dtype=attention.dtype)
    return attention / attention.sum(dim=-1, keepdim=True).clamp_min(1e-12)


def _optional_bce(logits: torch.Tensor, labels: torch.Tensor | None) -> torch.Tensor:
    if labels is None:
        return torch.zeros((), device=logits.device, dtype=logits.dtype)
    return F.binary_cross_entropy_with_logits(logits, labels.to(dtype=logits.dtype))


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


class HashedTextEncoder(nn.Module):
    """轻量文本 encoder：使用数据侧哈希 token，不依赖外部 BERT 权重。"""

    def __init__(
        self,
        *,
        vocab_size: int,
        token_embed_dim: int,
        output_dim: int,
        dropout: float,
    ) -> None:
        super().__init__()
        self.vocab_size = max(128, int(vocab_size))
        self.token_embedding = nn.Embedding(self.vocab_size, int(token_embed_dim), padding_idx=0)
        self.token_proj = nn.Sequential(
            nn.LayerNorm(int(token_embed_dim)),
            nn.Linear(int(token_embed_dim), output_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(output_dim, output_dim),
        )
        self.pooled_proj = nn.Sequential(
            nn.LayerNorm(output_dim),
            nn.Linear(output_dim, output_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(output_dim, output_dim),
        )

    def forward(
        self,
        token_ids: torch.Tensor | None,
        token_mask: torch.Tensor | None,
        *,
        batch_size: int,
        device: torch.device,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        if token_ids is None:
            token_ids = torch.zeros(batch_size, 1, dtype=torch.long, device=device)
        else:
            token_ids = token_ids.to(device=device, dtype=torch.long).clamp_min(0).clamp_max(self.vocab_size - 1)

        if token_mask is None:
            token_mask = token_ids > 0
        else:
            token_mask = token_mask.to(device=device).bool() & (token_ids > 0)

        token_embeddings = self.token_embedding(token_ids)
        token_features = self.token_proj(token_embeddings) * token_mask.unsqueeze(-1).to(dtype=token_embeddings.dtype)
        pooled, active = _masked_mean(token_features, token_mask)
        pooled = self.pooled_proj(pooled) * active.unsqueeze(-1).to(dtype=pooled.dtype)
        return token_features, token_mask, pooled, active


class TextCNNSequenceEncoder(nn.Module):
    """面向交叉注意力的多尺度 TextCNN 编码器。

    与纯文本对照实验中的 TextCNN 使用相同的嵌入维度、卷积核和通道规则，
    但同时保留逐位置的局部卷积特征，供标签查询交叉注意力使用。
    """

    def __init__(
        self,
        *,
        vocab_size: int,
        token_embed_dim: int,
        output_dim: int,
        dropout: float,
        kernel_sizes: tuple[int, ...] = (2, 3, 4),
    ) -> None:
        super().__init__()
        normalized_kernel_sizes = tuple(int(value) for value in kernel_sizes)
        if not normalized_kernel_sizes or any(value <= 0 for value in normalized_kernel_sizes):
            raise ValueError("TextCNN kernel_sizes 必须为非空正整数序列")
        self.vocab_size = max(128, int(vocab_size))
        self.kernel_sizes = normalized_kernel_sizes
        channels = max(64, int(output_dim) // len(normalized_kernel_sizes))
        self.token_embedding = nn.Embedding(self.vocab_size, int(token_embed_dim), padding_idx=0)
        self.convolutions = nn.ModuleList(
            nn.Conv1d(int(token_embed_dim), channels, kernel_size=kernel_size)
            for kernel_size in normalized_kernel_sizes
        )
        self.token_projection = nn.Sequential(
            nn.LayerNorm(channels * len(normalized_kernel_sizes)),
            nn.Linear(channels * len(normalized_kernel_sizes), int(output_dim)),
            nn.GELU(),
            nn.Dropout(dropout),
        )

    def forward(
        self,
        token_ids: torch.Tensor | None,
        token_mask: torch.Tensor | None,
        *,
        batch_size: int,
        device: torch.device,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        if token_ids is None:
            token_ids = torch.zeros(batch_size, 1, dtype=torch.long, device=device)
        else:
            token_ids = token_ids.to(device=device, dtype=torch.long).clamp_min(0).clamp_max(self.vocab_size - 1)

        if token_mask is None:
            token_mask = token_ids > 0
        else:
            token_mask = token_mask.to(device=device).bool() & (token_ids > 0)

        embedded = self.token_embedding(token_ids).transpose(1, 2)
        local_features: list[torch.Tensor] = []
        for convolution, kernel_size in zip(self.convolutions, self.kernel_sizes):
            left_padding = (kernel_size - 1) // 2
            right_padding = kernel_size - 1 - left_padding
            features = torch.relu(convolution(F.pad(embedded, (left_padding, right_padding))))
            local_features.append(features.transpose(1, 2))

        token_features = self.token_projection(torch.cat(local_features, dim=-1))
        token_features = token_features * token_mask.unsqueeze(-1).to(dtype=token_features.dtype)
        active = token_mask.any(dim=1)
        masked_features = token_features.masked_fill(~token_mask.unsqueeze(-1), -1e4)
        pooled = masked_features.amax(dim=1)
        pooled = torch.where(active.unsqueeze(-1), pooled, torch.zeros_like(pooled))
        return token_features, token_mask, pooled, active


class Exp8LongMILBase(Exp4BaseModel):
    """exp_8 共用的 64 图 Long-MIL 图像底座。"""

    def __init__(
        self,
        *,
        num_layers: int = 2,
        num_heads: int = 4,
        hidden_dim: int = 1024,
        label_graph_type: str = "label_hypergraph",
        label_hypergraph_edges: int = 2,
        **kwargs: Any,
    ) -> None:
        kwargs["use_label_graph"] = bool(kwargs.get("use_label_graph", True))
        for unused_key in (
            "structured_fields",
            "structured_category_sizes",
            "structured_field_embed_dim",
            "structured_dropout",
            "modality_dropout",
            "prototype_dropout",
            "prototype_mix",
            "graph_prior_mix",
            "text_vocab_size",
            "text_embed_dim",
            "contrast_temperature",
            "use_context_encoder",
            "watch_fusion_mode",
            "use_text_gate",
        ):
            kwargs.pop(unused_key, None)
        super().__init__(**kwargs)
        self.label_graph_type = str(label_graph_type).strip().lower() or "label_hypergraph"
        if self.label_graph_type not in {"learnable", "label_hypergraph"}:
            raise ValueError(f"exp_8 不支持的 label_graph_type: {label_graph_type}")
        if self.use_label_graph and self.label_graph_type == "label_hypergraph":
            self.label_graph_reasoner = LabelHypergraphReasoner(
                num_labels=self.num_labels,
                feature_dim=self.feature_dim,
                dropout=float(kwargs.get("dropout", 0.2)),
                num_hyperedges=int(label_hypergraph_edges),
            )
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

    def encode_long_mil(
        self,
        images: torch.Tensor,
        mask: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, dict[str, torch.Tensor]]:
        features, extra_outputs = self.encode_instances(images, mask)
        features = self.position_encoding(features, mask)
        context_features = self.context_encoder(features, src_key_padding_mask=~mask)
        context_features = context_features * mask.unsqueeze(-1).to(dtype=context_features.dtype)
        bag_embeds, attention = self.mil_pool(context_features, mask)
        label_embeds, graph_outputs = self.refine_labels(bag_embeds)
        extra_outputs.update(graph_outputs)
        return context_features, label_embeds, attention, extra_outputs

    def image_only_outputs(
        self,
        images: torch.Tensor,
        mask: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        context_features, label_embeds, attention, extra_outputs = self.encode_long_mil(images, mask)
        logits = self.classify(label_embeds)
        return self.build_outputs(
            logits=logits,
            attention=attention,
            features=context_features,
            extra_outputs=extra_outputs,
        )


class Exp8StructuredLateGateMILModel(Exp8LongMILBase):
    """Long-MIL 图像分支 + 结构化字段 label-wise gated late fusion。"""

    def __init__(
        self,
        *,
        structured_fields: list[str] | tuple[str, ...] | str | None = None,
        structured_category_sizes: dict[str, int] | None = None,
        structured_field_embed_dim: int = 64,
        structured_dropout: float = 0.2,
        modality_dropout: float = 0.15,
        **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)
        self.selected_structured_fields = _normalize_structured_fields(structured_fields)
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
        context_features, label_embeds, attention, extra_outputs = self.encode_long_mil(images, mask)

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


class Exp8StructuredLateGateLongMILModel(Exp8StructuredLateGateMILModel):
    """`exp8_mm_struct_late_gate` 的正式注册名。"""


class Exp8LabelProtoGraphLongMILModel(Exp8LongMILBase):
    """固定标签文本原型 + label graph 语义约束。"""

    def __init__(
        self,
        *,
        prototype_dropout: float = 0.1,
        prototype_mix: float = 0.35,
        graph_prior_mix: float = 0.3,
        **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)
        max_terms = max(len(items) for items in LABEL_PROTOTYPE_TERMS)
        proto_tokens = torch.zeros(self.num_labels, max_terms, self.feature_dim)
        proto_mask = torch.zeros(self.num_labels, max_terms, dtype=torch.bool)
        for label_index in range(self.num_labels):
            terms = LABEL_PROTOTYPE_TERMS[label_index] if label_index < len(LABEL_PROTOTYPE_TERMS) else (f"label_{label_index}",)
            for term_index, term in enumerate(terms):
                proto_tokens[label_index, term_index] = _stable_unit_vector(term, self.feature_dim)
                proto_mask[label_index, term_index] = True
        self.register_buffer("prototype_tokens", proto_tokens)
        self.register_buffer("prototype_mask", proto_mask)
        self.prototype_attention = nn.Sequential(
            nn.LayerNorm(self.feature_dim),
            nn.Linear(self.feature_dim, 1),
        )
        self.prototype_projector = nn.Sequential(
            nn.LayerNorm(self.feature_dim),
            nn.Linear(self.feature_dim, self.feature_dim),
            nn.GELU(),
            nn.Dropout(float(prototype_dropout)),
            nn.Linear(self.feature_dim, self.feature_dim),
        )
        self.proto_gate = nn.Sequential(
            nn.LayerNorm(self.feature_dim * 2),
            nn.Linear(self.feature_dim * 2, self.feature_dim),
            nn.GELU(),
            nn.Linear(self.feature_dim, 1),
        )
        self.graph_prior_refine = nn.Sequential(
            nn.LayerNorm(self.feature_dim * 2),
            nn.Linear(self.feature_dim * 2, self.feature_dim),
            nn.GELU(),
            nn.Dropout(float(kwargs.get("dropout", 0.2))),
        )
        self.prototype_mix = float(prototype_mix)
        self.graph_prior_mix = float(graph_prior_mix)

    def _pooled_prototypes(self) -> torch.Tensor:
        scores = self.prototype_attention(self.prototype_tokens).squeeze(-1)
        scores = scores.masked_fill(~self.prototype_mask, -1e4)
        weights = torch.softmax(scores, dim=-1)
        pooled = torch.einsum("lk,lkd->ld", weights, self.prototype_tokens)
        return self.prototype_projector(pooled)

    def forward(
        self,
        images: torch.Tensor,
        mask: torch.Tensor,
        labels: torch.Tensor | None = None,
        **kwargs: Any,
    ) -> dict[str, torch.Tensor]:
        del kwargs
        context_features, label_embeds, attention, extra_outputs = self.encode_long_mil(images, mask)
        prototypes = self._pooled_prototypes().to(dtype=label_embeds.dtype, device=label_embeds.device)
        proto_norm = F.normalize(prototypes, dim=-1)
        graph_prior_logits = torch.matmul(proto_norm, proto_norm.transpose(0, 1))
        graph_prior = torch.softmax(graph_prior_logits, dim=-1)
        prior_message = torch.einsum("lk,bkd->bld", graph_prior, label_embeds)
        label_embeds = label_embeds + self.graph_prior_mix * self.graph_prior_refine(
            torch.cat([label_embeds, prior_message], dim=-1)
        )

        proto_expanded = prototypes.unsqueeze(0).expand(label_embeds.shape[0], -1, -1)
        gate = torch.sigmoid(self.proto_gate(torch.cat([label_embeds, proto_expanded], dim=-1)))
        fused_label_embeds = label_embeds + self.prototype_mix * gate * proto_expanded
        logits = self.classify(fused_label_embeds)

        cosine_loss = 1.0 - F.cosine_similarity(label_embeds, proto_expanded, dim=-1)
        if labels is not None:
            label_weights = labels.to(dtype=cosine_loss.dtype)
            proto_align = (cosine_loss * label_weights).sum() / label_weights.sum().clamp_min(1.0)
        else:
            proto_align = cosine_loss.mean()
        graph_loss = torch.zeros((), device=logits.device, dtype=logits.dtype)
        if "label_graph" in extra_outputs and torch.is_tensor(extra_outputs["label_graph"]):
            graph_loss = F.mse_loss(extra_outputs["label_graph"].to(dtype=graph_prior.dtype), graph_prior)

        extra_outputs.update(
            {
                "label_prototypes": prototypes,
                "prototype_gate": gate.squeeze(-1),
                "prototype_graph_prior": graph_prior,
                "aux_losses": {
                    "proto_align": proto_align,
                    "graph_prior": graph_loss,
                },
            }
        )
        return self.build_outputs(
            logits=logits,
            attention=attention,
            features=context_features,
            extra_outputs=extra_outputs,
        )


class Exp8TextContrastDistillLongMILModel(Exp8LongMILBase):
    """训练期图文对齐蒸馏；推理 logits 保持 image-only。"""

    def __init__(
        self,
        *,
        text_vocab_size: int = 8192,
        text_embed_dim: int = 128,
        contrast_temperature: float = 0.07,
        **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)
        dropout = float(kwargs.get("dropout", 0.2))
        self.text_encoder = HashedTextEncoder(
            vocab_size=int(text_vocab_size),
            token_embed_dim=int(text_embed_dim),
            output_dim=self.feature_dim,
            dropout=dropout,
        )
        self.image_projector = nn.Sequential(
            nn.LayerNorm(self.feature_dim),
            nn.Linear(self.feature_dim, self.feature_dim),
            nn.GELU(),
            nn.Linear(self.feature_dim, self.feature_dim),
        )
        self.text_projector = nn.Sequential(
            nn.LayerNorm(self.feature_dim),
            nn.Linear(self.feature_dim, self.feature_dim),
            nn.GELU(),
            nn.Linear(self.feature_dim, self.feature_dim),
        )
        self.contrast_temperature = max(1e-3, float(contrast_temperature))

    def forward(
        self,
        images: torch.Tensor,
        mask: torch.Tensor,
        text_token_ids: torch.Tensor | None = None,
        text_token_mask: torch.Tensor | None = None,
        **kwargs: Any,
    ) -> dict[str, torch.Tensor]:
        del kwargs
        context_features, label_embeds, attention, extra_outputs = self.encode_long_mil(images, mask)
        logits = self.classify(label_embeds)

        aux_losses: dict[str, torch.Tensor] = {}
        if self.training:
            _, _, text_embed, text_active = self.text_encoder(
                text_token_ids,
                text_token_mask,
                batch_size=images.shape[0],
                device=images.device,
            )
            image_embed = label_embeds.mean(dim=1)
            image_proj = F.normalize(self.image_projector(image_embed), dim=-1)
            text_proj = F.normalize(self.text_projector(text_embed), dim=-1)
            if text_active.any():
                active = text_active.to(device=images.device)
                aux_losses["text_align"] = F.mse_loss(image_proj[active], text_proj[active])
                if int(active.sum().item()) > 1:
                    active_image = image_proj[active]
                    active_text = text_proj[active]
                    logits_itc = torch.matmul(active_image, active_text.transpose(0, 1)) / self.contrast_temperature
                    targets = torch.arange(logits_itc.shape[0], device=logits_itc.device)
                    aux_losses["text_itc"] = 0.5 * (
                        F.cross_entropy(logits_itc, targets)
                        + F.cross_entropy(logits_itc.transpose(0, 1), targets)
                    )
                else:
                    aux_losses["text_itc"] = torch.zeros((), device=images.device, dtype=logits.dtype)
            else:
                aux_losses["text_align"] = torch.zeros((), device=images.device, dtype=logits.dtype)
                aux_losses["text_itc"] = torch.zeros((), device=images.device, dtype=logits.dtype)

        if aux_losses:
            extra_outputs["aux_losses"] = aux_losses
        return self.build_outputs(
            logits=logits,
            attention=attention,
            features=context_features,
            extra_outputs=extra_outputs,
        )


class Exp8WatchCrossAttentionLongMILModel(Exp8LongMILBase):
    """检查所见 watch 文本与图像 label token cross-attention 融合。"""

    def __init__(
        self,
        *,
        text_vocab_size: int = 8192,
        text_embed_dim: int = 128,
        num_heads: int = 4,
        **kwargs: Any,
    ) -> None:
        super().__init__(num_heads=num_heads, **kwargs)
        dropout = float(kwargs.get("dropout", 0.2))
        self.text_encoder = HashedTextEncoder(
            vocab_size=int(text_vocab_size),
            token_embed_dim=int(text_embed_dim),
            output_dim=self.feature_dim,
            dropout=dropout,
        )
        head_count = int(num_heads) if self.feature_dim % int(num_heads) == 0 else 1
        self.text_cross_attn = nn.MultiheadAttention(
            embed_dim=self.feature_dim,
            num_heads=max(1, head_count),
            dropout=dropout,
            batch_first=True,
        )
        self.text_gate = nn.Sequential(
            nn.LayerNorm(self.feature_dim * 2),
            nn.Linear(self.feature_dim * 2, self.feature_dim),
            nn.GELU(),
            nn.Linear(self.feature_dim, 1),
        )

    def _build_watch_cross_attention_outputs(
        self,
        *,
        images: torch.Tensor,
        label_embeds: torch.Tensor,
        attention: torch.Tensor,
        features: torch.Tensor,
        extra_outputs: dict[str, torch.Tensor],
        labels: torch.Tensor | None,
        watch_token_ids: torch.Tensor | None,
        watch_token_mask: torch.Tensor | None,
        use_gate: bool,
    ) -> dict[str, torch.Tensor]:
        image_only_logits = self.classify(label_embeds)
        text_tokens, text_mask, _, text_active = self.text_encoder(
            watch_token_ids,
            watch_token_mask,
            batch_size=images.shape[0],
            device=images.device,
        )
        safe_tokens, key_padding_mask = _safe_key_padding_mask(text_tokens, text_mask)
        text_label_embeds, cross_attention = self.text_cross_attn(
            label_embeds,
            safe_tokens,
            safe_tokens,
            key_padding_mask=key_padding_mask,
            need_weights=True,
            average_attn_weights=True,
        )
        active = text_active.view(-1, 1, 1).to(dtype=text_label_embeds.dtype)
        text_label_embeds = text_label_embeds * active
        if use_gate:
            gates = torch.sigmoid(self.text_gate(torch.cat([label_embeds, text_label_embeds], dim=-1)))
            gates = gates * active
        else:
            gates = torch.ones(
                label_embeds.shape[0],
                label_embeds.shape[1],
                1,
                device=label_embeds.device,
                dtype=label_embeds.dtype,
            ) * active
        fused_label_embeds = label_embeds + gates * text_label_embeds
        logits = self.classify(fused_label_embeds)

        extra_outputs.update(
            {
                "image_only_logits": image_only_logits,
                "watch_cross_attention": cross_attention,
                "watch_text_gate": gates.squeeze(-1),
                "aux_losses": {
                    "image_aux": _optional_bce(image_only_logits, labels),
                },
            }
        )
        return self.build_outputs(
            logits=logits,
            attention=attention,
            features=features,
            extra_outputs=extra_outputs,
        )

    def forward(
        self,
        images: torch.Tensor,
        mask: torch.Tensor,
        labels: torch.Tensor | None = None,
        watch_token_ids: torch.Tensor | None = None,
        watch_token_mask: torch.Tensor | None = None,
        **kwargs: Any,
    ) -> dict[str, torch.Tensor]:
        del kwargs
        context_features, label_embeds, attention, extra_outputs = self.encode_long_mil(images, mask)
        return self._build_watch_cross_attention_outputs(
            images=images,
            label_embeds=label_embeds,
            attention=attention,
            features=context_features,
            extra_outputs=extra_outputs,
            labels=labels,
            watch_token_ids=watch_token_ids,
            watch_token_mask=watch_token_mask,
            use_gate=True,
        )


class Exp8WatchCrossAttentionTextCNNLongMILModel(Exp8WatchCrossAttentionLongMILModel):
    """TASK3 主模型：仅将原 watch 文本编码器替换为 TextCNN。"""

    def __init__(
        self,
        *,
        text_vocab_size: int = 8192,
        text_embed_dim: int = 128,
        textcnn_kernel_sizes: tuple[int, ...] | list[int] = (2, 3, 4),
        **kwargs: Any,
    ) -> None:
        super().__init__(
            text_vocab_size=text_vocab_size,
            text_embed_dim=text_embed_dim,
            **kwargs,
        )
        dropout = float(kwargs.get("dropout", 0.2))
        self.text_encoder = TextCNNSequenceEncoder(
            vocab_size=int(text_vocab_size),
            token_embed_dim=int(text_embed_dim),
            output_dim=self.feature_dim,
            dropout=dropout,
            kernel_sizes=tuple(int(value) for value in textcnn_kernel_sizes),
        )


class Exp9WatchNoTextLongMILModel(Exp8LongMILBase):
    """exp_9 消融：保留 exp8 watch 图像底座，但不输入 watch 文本。"""

    def forward(
        self,
        images: torch.Tensor,
        mask: torch.Tensor,
        **kwargs: Any,
    ) -> dict[str, torch.Tensor]:
        del kwargs
        return self.image_only_outputs(images, mask)


class Exp9WatchCrossAttentionNoContextModel(Exp8WatchCrossAttentionLongMILModel):
    """exp_9 消融：去掉位置编码与 Transformer context encoder。"""

    def encode_long_mil(
        self,
        images: torch.Tensor,
        mask: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, dict[str, torch.Tensor]]:
        features, extra_outputs = self.encode_instances(images, mask)
        bag_embeds, attention = self.mil_pool(features, mask)
        label_embeds, graph_outputs = self.refine_labels(bag_embeds)
        extra_outputs.update(graph_outputs)
        return features, label_embeds, attention, extra_outputs


class Exp9WatchPooledLateFusionLongMILModel(Exp8LongMILBase):
    """exp_9 消融：去掉 cross-attention，改用 watch pooled embedding late fusion。"""

    def __init__(
        self,
        *,
        text_vocab_size: int = 8192,
        text_embed_dim: int = 128,
        **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)
        dropout = float(kwargs.get("dropout", 0.2))
        self.text_encoder = HashedTextEncoder(
            vocab_size=int(text_vocab_size),
            token_embed_dim=int(text_embed_dim),
            output_dim=self.feature_dim,
            dropout=dropout,
        )
        self.text_gate = nn.Sequential(
            nn.LayerNorm(self.feature_dim * 2),
            nn.Linear(self.feature_dim * 2, self.feature_dim),
            nn.GELU(),
            nn.Linear(self.feature_dim, 1),
        )

    def forward(
        self,
        images: torch.Tensor,
        mask: torch.Tensor,
        labels: torch.Tensor | None = None,
        watch_token_ids: torch.Tensor | None = None,
        watch_token_mask: torch.Tensor | None = None,
        **kwargs: Any,
    ) -> dict[str, torch.Tensor]:
        del kwargs
        context_features, label_embeds, attention, extra_outputs = self.encode_long_mil(images, mask)
        image_only_logits = self.classify(label_embeds)
        _, _, text_embed, text_active = self.text_encoder(
            watch_token_ids,
            watch_token_mask,
            batch_size=images.shape[0],
            device=images.device,
        )
        text_label_embeds = text_embed.unsqueeze(1).expand(-1, self.num_labels, -1)
        active = text_active.view(-1, 1, 1).to(dtype=text_label_embeds.dtype)
        text_label_embeds = text_label_embeds * active
        gates = torch.sigmoid(self.text_gate(torch.cat([label_embeds, text_label_embeds], dim=-1)))
        gates = gates * active
        fused_label_embeds = label_embeds + gates * text_label_embeds
        logits = self.classify(fused_label_embeds)

        extra_outputs.update(
            {
                "image_only_logits": image_only_logits,
                "watch_text_gate": gates.squeeze(-1),
                "aux_losses": {
                    "image_aux": _optional_bce(image_only_logits, labels),
                },
            }
        )
        return self.build_outputs(
            logits=logits,
            attention=attention,
            features=context_features,
            extra_outputs=extra_outputs,
        )


class Exp9WatchCrossAttentionNoGateLongMILModel(Exp8WatchCrossAttentionLongMILModel):
    """exp_9 消融：保留 watch cross-attention，但去掉 gate。"""

    def forward(
        self,
        images: torch.Tensor,
        mask: torch.Tensor,
        labels: torch.Tensor | None = None,
        watch_token_ids: torch.Tensor | None = None,
        watch_token_mask: torch.Tensor | None = None,
        **kwargs: Any,
    ) -> dict[str, torch.Tensor]:
        del kwargs
        context_features, label_embeds, attention, extra_outputs = self.encode_long_mil(images, mask)
        return self._build_watch_cross_attention_outputs(
            images=images,
            label_embeds=label_embeds,
            attention=attention,
            features=context_features,
            extra_outputs=extra_outputs,
            labels=labels,
            watch_token_ids=watch_token_ids,
            watch_token_mask=watch_token_mask,
            use_gate=False,
        )


class Exp11ModuleAblationModel(Exp8LongMILBase):
    """四模块全因子消融的参数化模型。

    M1 控制位置编码与 Transformer context，M2 由 ``label_graph_type``
    控制普通标签图或标签超图，M3 控制 cross-attention，M4 控制门控。
    当只启用 M4 时，使用池化后的 watch 文本作为门控融合输入。
    """

    def __init__(
        self,
        *,
        use_context_encoder: bool = False,
        watch_fusion_mode: str = "none",
        use_text_gate: bool = False,
        text_vocab_size: int = 8192,
        text_embed_dim: int = 128,
        num_heads: int = 4,
        **kwargs: Any,
    ) -> None:
        super().__init__(num_heads=num_heads, **kwargs)
        self.use_context_encoder = bool(use_context_encoder)
        self.watch_fusion_mode = str(watch_fusion_mode).strip().lower() or "none"
        if self.watch_fusion_mode not in {"none", "cross_attention", "pooled"}:
            raise ValueError(f"exp11 不支持的 watch_fusion_mode: {watch_fusion_mode}")
        self.use_text_gate = bool(use_text_gate)
        if self.watch_fusion_mode == "none" and self.use_text_gate:
            raise ValueError("exp11 启用 M4 时必须提供 pooled 或 cross_attention 文本融合输入")

        dropout = float(kwargs.get("dropout", 0.2))
        self.text_encoder = None
        self.text_cross_attn = None
        self.text_gate = None
        if self.watch_fusion_mode != "none":
            self.text_encoder = HashedTextEncoder(
                vocab_size=int(text_vocab_size),
                token_embed_dim=int(text_embed_dim),
                output_dim=self.feature_dim,
                dropout=dropout,
            )
        if self.watch_fusion_mode == "cross_attention":
            head_count = int(num_heads) if self.feature_dim % int(num_heads) == 0 else 1
            self.text_cross_attn = nn.MultiheadAttention(
                embed_dim=self.feature_dim,
                num_heads=max(1, head_count),
                dropout=dropout,
                batch_first=True,
            )
        if self.use_text_gate:
            self.text_gate = nn.Sequential(
                nn.LayerNorm(self.feature_dim * 2),
                nn.Linear(self.feature_dim * 2, self.feature_dim),
                nn.GELU(),
                nn.Linear(self.feature_dim, 1),
            )

    def encode_long_mil(
        self,
        images: torch.Tensor,
        mask: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, dict[str, torch.Tensor]]:
        if self.use_context_encoder:
            return super().encode_long_mil(images, mask)
        features, extra_outputs = self.encode_instances(images, mask)
        bag_embeds, attention = self.mil_pool(features, mask)
        label_embeds, graph_outputs = self.refine_labels(bag_embeds)
        extra_outputs.update(graph_outputs)
        return features, label_embeds, attention, extra_outputs

    def forward(
        self,
        images: torch.Tensor,
        mask: torch.Tensor,
        labels: torch.Tensor | None = None,
        watch_token_ids: torch.Tensor | None = None,
        watch_token_mask: torch.Tensor | None = None,
        **kwargs: Any,
    ) -> dict[str, torch.Tensor]:
        del kwargs
        features, label_embeds, attention, extra_outputs = self.encode_long_mil(images, mask)
        if self.watch_fusion_mode == "none":
            logits = self.classify(label_embeds)
            return self.build_outputs(
                logits=logits,
                attention=attention,
                features=features,
                extra_outputs=extra_outputs,
            )

        if self.text_encoder is None:
            raise RuntimeError("exp11 文本融合已启用，但 text_encoder 未初始化")
        image_only_logits = self.classify(label_embeds)
        text_tokens, text_mask, text_embed, text_active = self.text_encoder(
            watch_token_ids,
            watch_token_mask,
            batch_size=images.shape[0],
            device=images.device,
        )
        if self.watch_fusion_mode == "cross_attention":
            if self.text_cross_attn is None:
                raise RuntimeError("exp11 cross-attention 未初始化")
            safe_tokens, key_padding_mask = _safe_key_padding_mask(text_tokens, text_mask)
            text_label_embeds, text_attention = self.text_cross_attn(
                label_embeds,
                safe_tokens,
                safe_tokens,
                key_padding_mask=key_padding_mask,
                need_weights=True,
                average_attn_weights=True,
            )
            extra_outputs["watch_cross_attention"] = text_attention
        else:
            text_label_embeds = text_embed.unsqueeze(1).expand(-1, self.num_labels, -1)

        active = text_active.view(-1, 1, 1).to(dtype=text_label_embeds.dtype)
        text_label_embeds = text_label_embeds * active
        if self.text_gate is not None:
            gates = torch.sigmoid(self.text_gate(torch.cat([label_embeds, text_label_embeds], dim=-1)))
            gates = gates * active
        else:
            gates = torch.ones_like(text_label_embeds[..., :1]) * active
        fused_label_embeds = label_embeds + gates * text_label_embeds
        logits = self.classify(fused_label_embeds)
        extra_outputs.update(
            {
                "image_only_logits": image_only_logits,
                "watch_text_gate": gates.squeeze(-1),
                "aux_losses": {"image_aux": _optional_bce(image_only_logits, labels)},
            }
        )
        return self.build_outputs(
            logits=logits,
            attention=attention,
            features=features,
            extra_outputs=extra_outputs,
        )


class Exp8TextGuidedTop64AlignMILModel(Exp8LongMILBase):
    """文本引导的 64 图实例重加权 + 图文对齐 MIL 第一版。"""

    def __init__(
        self,
        *,
        text_vocab_size: int = 8192,
        text_embed_dim: int = 128,
        num_heads: int = 4,
        **kwargs: Any,
    ) -> None:
        super().__init__(num_heads=num_heads, **kwargs)
        dropout = float(kwargs.get("dropout", 0.2))
        self.text_encoder = HashedTextEncoder(
            vocab_size=int(text_vocab_size),
            token_embed_dim=int(text_embed_dim),
            output_dim=self.feature_dim,
            dropout=dropout,
        )
        self.text_query = nn.Sequential(
            nn.LayerNorm(self.feature_dim),
            nn.Linear(self.feature_dim, self.feature_dim),
            nn.GELU(),
            nn.Linear(self.feature_dim, self.feature_dim),
        )
        self.guided_label_proj = nn.Linear(self.feature_dim, self.num_labels * self.feature_dim)
        head_count = int(num_heads) if self.feature_dim % int(num_heads) == 0 else 1
        self.text_cross_attn = nn.MultiheadAttention(
            embed_dim=self.feature_dim,
            num_heads=max(1, head_count),
            dropout=dropout,
            batch_first=True,
        )
        self.guided_gate = nn.Sequential(
            nn.LayerNorm(self.feature_dim * 2),
            nn.Linear(self.feature_dim * 2, self.feature_dim),
            nn.GELU(),
            nn.Linear(self.feature_dim, 1),
        )

    def forward(
        self,
        images: torch.Tensor,
        mask: torch.Tensor,
        labels: torch.Tensor | None = None,
        guided_text_token_ids: torch.Tensor | None = None,
        guided_text_token_mask: torch.Tensor | None = None,
        **kwargs: Any,
    ) -> dict[str, torch.Tensor]:
        del kwargs
        context_features, label_embeds, attention, extra_outputs = self.encode_long_mil(images, mask)
        image_only_logits = self.classify(label_embeds)
        text_tokens, text_mask, text_embed, text_active = self.text_encoder(
            guided_text_token_ids,
            guided_text_token_mask,
            batch_size=images.shape[0],
            device=images.device,
        )
        text_query = self.text_query(text_embed)
        guided_scores = torch.einsum("bnd,bd->bn", context_features, text_query) / math.sqrt(float(self.feature_dim))
        guided_attention = _masked_instance_softmax(guided_scores, mask)
        guided_bag = torch.einsum("bn,bnd->bd", guided_attention, context_features)
        guided_label_embeds = self.guided_label_proj(guided_bag).view(
            images.shape[0],
            self.num_labels,
            self.feature_dim,
        )
        guided_label_embeds = guided_label_embeds * text_active.view(-1, 1, 1).to(dtype=guided_label_embeds.dtype)

        safe_tokens, key_padding_mask = _safe_key_padding_mask(text_tokens, text_mask)
        text_label_embeds, text_attention = self.text_cross_attn(
            label_embeds,
            safe_tokens,
            safe_tokens,
            key_padding_mask=key_padding_mask,
            need_weights=True,
            average_attn_weights=True,
        )
        text_label_embeds = text_label_embeds * text_active.view(-1, 1, 1).to(dtype=text_label_embeds.dtype)
        guided_context = guided_label_embeds + text_label_embeds
        gates = torch.sigmoid(self.guided_gate(torch.cat([label_embeds, guided_context], dim=-1)))
        gates = gates * text_active.view(-1, 1, 1).to(dtype=gates.dtype)
        fused_label_embeds = label_embeds + gates * guided_context
        logits = self.classify(fused_label_embeds)

        active = text_active.to(device=images.device)
        if active.any():
            text_align = 1.0 - F.cosine_similarity(guided_bag[active], text_embed[active], dim=-1).mean()
        else:
            text_align = torch.zeros((), device=images.device, dtype=logits.dtype)
        entropy = -(guided_attention * guided_attention.clamp_min(1e-12).log()).sum(dim=-1).mean()
        fused_prob = torch.sigmoid(logits)
        image_prob = torch.sigmoid(image_only_logits).detach()
        consistency = F.mse_loss(fused_prob, image_prob)

        extra_outputs.update(
            {
                "image_only_logits": image_only_logits,
                "text_guided_attention": guided_attention,
                "text_token_attention": text_attention,
                "text_guided_gate": gates.squeeze(-1),
                "aux_losses": {
                    "image_aux": _optional_bce(image_only_logits, labels),
                    "text_align": text_align,
                    "attention_sparse": entropy,
                    "consistency": consistency,
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
    "exp8_mm_struct_late_gate": Exp8StructuredLateGateLongMILModel,
    "exp8_mm_label_proto_graph": Exp8LabelProtoGraphLongMILModel,
    "exp8_mm_text_contrast_distill": Exp8TextContrastDistillLongMILModel,
    "exp8_mm_watch_cross_attn": Exp8WatchCrossAttentionLongMILModel,
    "exp8_mm_watch_cross_attn_textcnn": Exp8WatchCrossAttentionTextCNNLongMILModel,
    "exp8_mm_text_guided_top64_align": Exp8TextGuidedTop64AlignMILModel,
    "exp9_watch_no_text": Exp9WatchNoTextLongMILModel,
    "exp9_watch_no_context": Exp9WatchCrossAttentionNoContextModel,
    "exp9_watch_no_cross_attn_pool_fusion": Exp9WatchPooledLateFusionLongMILModel,
    "exp9_watch_cross_attn_no_gate": Exp9WatchCrossAttentionNoGateLongMILModel,
    "exp11_module_ablation": Exp11ModuleAblationModel,
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
        "prototype_dropout",
        "prototype_mix",
        "graph_prior_mix",
        "text_vocab_size",
        "text_embed_dim",
        "contrast_temperature",
        "use_context_encoder",
        "watch_fusion_mode",
        "use_text_gate",
        "textcnn_kernel_sizes",
    }
    model_kwargs = {key: value for key, value in kwargs.items() if key in base_keys}
    model_cls = EXP8_CLASS_REGISTRY.get(model_name)
    if model_cls is None:
        raise ValueError(f"未知 exp_8 模型名: {model_name}")
    return model_cls(**model_kwargs)
