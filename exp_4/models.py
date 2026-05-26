from __future__ import annotations

import math
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F

from model.common import MultiLabelAttentionMIL, masked_softmax
from model.gastro_label_graph_mil.modules import InstanceEncoder, LabelGraphReasoner


def _chunked_encode(instance_encoder: InstanceEncoder, images: torch.Tensor, chunk_size: int) -> torch.Tensor:
    batch_size, num_instances, channels, height, width = images.shape
    flattened = images.reshape(batch_size * num_instances, channels, height, width)
    chunk_size = max(1, int(chunk_size))

    if flattened.shape[0] <= chunk_size:
        features = instance_encoder.backbone(flattened)
        features = instance_encoder.projector(features)
        return features.reshape(batch_size, num_instances, -1)

    encoded_chunks = []
    for chunk in flattened.split(chunk_size, dim=0):
        chunk_features = instance_encoder.backbone(chunk)
        encoded_chunks.append(instance_encoder.projector(chunk_features))
    features = torch.cat(encoded_chunks, dim=0)
    return features.reshape(batch_size, num_instances, -1)


def _build_view_mask(mask: torch.Tensor, keep_ratio: float, view_index: int, training: bool) -> torch.Tensor:
    keep_ratio = min(1.0, max(0.05, float(keep_ratio)))
    if keep_ratio >= 1.0:
        return mask

    valid_counts = mask.sum(dim=1)
    view_mask = torch.zeros_like(mask)
    for batch_index in range(mask.shape[0]):
        valid_num = int(valid_counts[batch_index].item())
        if valid_num <= 0:
            continue
        keep_num = max(1, int(math.ceil(valid_num * keep_ratio)))
        if training:
            scores = torch.rand(valid_num, device=mask.device)
            selected = torch.topk(scores, k=keep_num, dim=0).indices
        else:
            start = (view_index * keep_num) % valid_num
            selected = (torch.arange(keep_num, device=mask.device) + start) % valid_num
        view_mask[batch_index, selected] = True
    return view_mask & mask


class Exp4BaseModel(nn.Module):
    def __init__(
        self,
        *,
        backbone_name: str = "convnext_tiny",
        pretrained: bool = True,
        freeze_stages: int = 1,
        feature_dim: int = 512,
        attn_dim: int = 256,
        num_labels: int = 3,
        dropout: float = 0.2,
        use_label_graph: bool = False,
        encoder_chunk_size: int = 16,
        use_quality_gate: bool = False,
    ) -> None:
        super().__init__()
        self.num_labels = int(num_labels)
        self.feature_dim = int(feature_dim)
        self.encoder_chunk_size = int(encoder_chunk_size)
        self.use_label_graph = bool(use_label_graph)
        self.use_quality_gate = bool(use_quality_gate)
        self.instance_encoder = InstanceEncoder(
            backbone_name=backbone_name,
            pretrained=pretrained,
            freeze_stages=freeze_stages,
            feature_dim=feature_dim,
            dropout=dropout,
        )
        self.mil_pool = MultiLabelAttentionMIL(
            in_dim=feature_dim,
            attn_dim=attn_dim,
            num_labels=num_labels,
            dropout=dropout,
        )
        self.label_graph_reasoner = (
            LabelGraphReasoner(num_labels=num_labels, feature_dim=feature_dim, dropout=dropout)
            if self.use_label_graph
            else None
        )
        self.quality_gate = (
            nn.Sequential(
                nn.LayerNorm(feature_dim),
                nn.Linear(feature_dim, max(16, feature_dim // 4)),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(max(16, feature_dim // 4), 1),
            )
            if self.use_quality_gate
            else None
        )
        self.classifiers = nn.ModuleList([nn.Linear(feature_dim, 1) for _ in range(num_labels)])

    def encode_instances(self, images: torch.Tensor, mask: torch.Tensor) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        features = _chunked_encode(self.instance_encoder, images, self.encoder_chunk_size)
        features = features * mask.to(dtype=features.dtype).unsqueeze(-1)
        extra: dict[str, torch.Tensor] = {}
        if self.quality_gate is not None:
            quality_scores = torch.sigmoid(self.quality_gate(features)).squeeze(-1)
            quality_scores = quality_scores * mask.to(dtype=quality_scores.dtype)
            features = features * quality_scores.unsqueeze(-1)
            extra["quality_scores"] = quality_scores
        return features, extra

    def refine_labels(self, bag_embeds: torch.Tensor) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        if self.label_graph_reasoner is None:
            return bag_embeds, {}
        refined_embeds, label_graph = self.label_graph_reasoner(bag_embeds)
        return refined_embeds, {"label_graph": label_graph}

    def classify(self, label_embeds: torch.Tensor) -> torch.Tensor:
        logits = [
            self.classifiers[label_index](label_embeds[:, label_index, :]).squeeze(-1)
            for label_index in range(self.num_labels)
        ]
        return torch.stack(logits, dim=1)

    def build_outputs(
        self,
        *,
        logits: torch.Tensor,
        attention: torch.Tensor,
        features: torch.Tensor,
        extra_outputs: dict[str, torch.Tensor] | None = None,
    ) -> dict[str, torch.Tensor]:
        outputs = {
            "logits": logits,
            "attention": attention,
            "instance_features": features,
        }
        if extra_outputs:
            outputs.update(extra_outputs)
        return outputs


class FullFeatureMILModel(Exp4BaseModel):
    def forward(self, images: torch.Tensor, mask: torch.Tensor, **kwargs: Any) -> dict[str, torch.Tensor]:
        del kwargs
        features, extra_outputs = self.encode_instances(images, mask)
        bag_embeds, attention = self.mil_pool(features, mask)
        label_embeds, graph_outputs = self.refine_labels(bag_embeds)
        logits = self.classify(label_embeds)
        extra_outputs.update(graph_outputs)
        return self.build_outputs(
            logits=logits,
            attention=attention,
            features=features,
            extra_outputs=extra_outputs,
        )


class HierarchicalLabelAwarePool(nn.Module):
    def __init__(
        self,
        *,
        feature_dim: int,
        attn_dim: int,
        num_labels: int,
        chunk_size: int,
        dropout: float,
    ) -> None:
        super().__init__()
        self.chunk_size = max(1, int(chunk_size))
        self.num_labels = int(num_labels)
        self.subbag_pool = MultiLabelAttentionMIL(
            in_dim=feature_dim,
            attn_dim=attn_dim,
            num_labels=num_labels,
            dropout=dropout,
        )
        self.patient_queries = nn.Parameter(torch.randn(num_labels, feature_dim) * 0.02)

    def _pad(self, features: torch.Tensor, mask: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, int]:
        num_instances = features.shape[1]
        pad_num = (self.chunk_size - num_instances % self.chunk_size) % self.chunk_size
        if pad_num <= 0:
            return features, mask, num_instances
        feature_pad = torch.zeros(
            features.shape[0],
            pad_num,
            features.shape[-1],
            dtype=features.dtype,
            device=features.device,
        )
        mask_pad = torch.zeros(features.shape[0], pad_num, dtype=mask.dtype, device=mask.device)
        return torch.cat([features, feature_pad], dim=1), torch.cat([mask, mask_pad], dim=1), num_instances

    def forward(self, features: torch.Tensor, mask: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        padded_features, padded_mask, original_instances = self._pad(features, mask)
        batch_size, padded_instances, feature_dim = padded_features.shape
        num_subbags = padded_instances // self.chunk_size

        subbag_features = padded_features.reshape(batch_size, num_subbags, self.chunk_size, feature_dim)
        subbag_mask = padded_mask.reshape(batch_size, num_subbags, self.chunk_size)
        flat_features = subbag_features.reshape(batch_size * num_subbags, self.chunk_size, feature_dim)
        flat_mask = subbag_mask.reshape(batch_size * num_subbags, self.chunk_size)

        subbag_embeds, subbag_instance_attention = self.subbag_pool(flat_features, flat_mask)
        subbag_embeds = subbag_embeds.reshape(batch_size, num_subbags, self.num_labels, feature_dim)
        subbag_instance_attention = subbag_instance_attention.reshape(
            batch_size,
            num_subbags,
            self.num_labels,
            self.chunk_size,
        )

        subbag_valid = subbag_mask.any(dim=-1)
        patient_scores = torch.einsum("bsld,ld->bsl", subbag_embeds, self.patient_queries)
        patient_scores = patient_scores / math.sqrt(float(feature_dim))
        patient_attention = masked_softmax(patient_scores.transpose(1, 2), subbag_valid.unsqueeze(1), dim=-1)
        bag_embeds = torch.einsum("bls,bsld->bld", patient_attention, subbag_embeds)

        combined_attention = (
            patient_attention.transpose(1, 2).unsqueeze(-1) * subbag_instance_attention
        )
        combined_attention = combined_attention.permute(0, 2, 1, 3).reshape(
            batch_size,
            self.num_labels,
            padded_instances,
        )
        combined_attention = combined_attention[:, :, :original_instances]
        combined_attention = combined_attention * mask.unsqueeze(1).to(dtype=combined_attention.dtype)
        denom = combined_attention.sum(dim=-1, keepdim=True).clamp_min(1e-12)
        combined_attention = combined_attention / denom
        return bag_embeds, combined_attention, patient_attention


class HierarchicalFullMILModel(Exp4BaseModel):
    def __init__(
        self,
        *,
        subbag_size: int = 8,
        **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)
        self.hier_pool = HierarchicalLabelAwarePool(
            feature_dim=self.feature_dim,
            attn_dim=int(kwargs.get("attn_dim", 256)),
            num_labels=self.num_labels,
            chunk_size=subbag_size,
            dropout=float(kwargs.get("dropout", 0.2)),
        )

    def forward(self, images: torch.Tensor, mask: torch.Tensor, **kwargs: Any) -> dict[str, torch.Tensor]:
        del kwargs
        features, extra_outputs = self.encode_instances(images, mask)
        bag_embeds, attention, subbag_attention = self.hier_pool(features, mask)
        label_embeds, graph_outputs = self.refine_labels(bag_embeds)
        logits = self.classify(label_embeds)
        extra_outputs.update(graph_outputs)
        extra_outputs["subbag_attention"] = subbag_attention
        return self.build_outputs(
            logits=logits,
            attention=attention,
            features=features,
            extra_outputs=extra_outputs,
        )


class MultiSampleLabelGraphMILModel(Exp4BaseModel):
    def __init__(
        self,
        *,
        num_views: int = 4,
        view_keep_ratio: float = 0.75,
        **kwargs: Any,
    ) -> None:
        kwargs["use_label_graph"] = True
        super().__init__(**kwargs)
        self.num_views = max(1, int(num_views))
        self.view_keep_ratio = float(view_keep_ratio)

    def forward(self, images: torch.Tensor, mask: torch.Tensor, **kwargs: Any) -> dict[str, torch.Tensor]:
        del kwargs
        features, extra_outputs = self.encode_instances(images, mask)
        logits_list = []
        attention_list = []
        last_graph_outputs: dict[str, torch.Tensor] = {}
        for view_index in range(self.num_views):
            view_mask = _build_view_mask(mask, self.view_keep_ratio, view_index, self.training)
            bag_embeds, attention = self.mil_pool(features, view_mask)
            label_embeds, graph_outputs = self.refine_labels(bag_embeds)
            logits_list.append(self.classify(label_embeds))
            attention_list.append(attention)
            last_graph_outputs = graph_outputs

        logits = torch.stack(logits_list, dim=0).mean(dim=0)
        attention = torch.stack(attention_list, dim=0).mean(dim=0)
        attention = attention * mask.unsqueeze(1).to(dtype=attention.dtype)
        attention = attention / attention.sum(dim=-1, keepdim=True).clamp_min(1e-12)
        extra_outputs.update(last_graph_outputs)
        return self.build_outputs(
            logits=logits,
            attention=attention,
            features=features,
            extra_outputs=extra_outputs,
        )


class TimePositionEncoding(nn.Module):
    def __init__(self, feature_dim: int) -> None:
        super().__init__()
        self.position_proj = nn.Sequential(
            nn.Linear(1, feature_dim),
            nn.GELU(),
            nn.Linear(feature_dim, feature_dim),
        )

    def forward(self, features: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        batch_size, num_instances, _ = features.shape
        if num_instances <= 1:
            positions = torch.zeros(batch_size, num_instances, 1, device=features.device, dtype=features.dtype)
        else:
            positions = torch.linspace(
                0.0,
                1.0,
                steps=num_instances,
                device=features.device,
                dtype=features.dtype,
            ).view(1, num_instances, 1)
            positions = positions.expand(batch_size, num_instances, 1)
        return (features + self.position_proj(positions)) * mask.unsqueeze(-1).to(dtype=features.dtype)


class LongMILModel(Exp4BaseModel):
    def __init__(
        self,
        *,
        num_layers: int = 2,
        num_heads: int = 4,
        hidden_dim: int = 1024,
        **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)
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

    def forward(self, images: torch.Tensor, mask: torch.Tensor, **kwargs: Any) -> dict[str, torch.Tensor]:
        del kwargs
        features, extra_outputs = self.encode_instances(images, mask)
        features = self.position_encoding(features, mask)
        context_features = self.context_encoder(features, src_key_padding_mask=~mask)
        context_features = context_features * mask.unsqueeze(-1).to(dtype=context_features.dtype)
        bag_embeds, attention = self.mil_pool(context_features, mask)
        label_embeds, graph_outputs = self.refine_labels(bag_embeds)
        logits = self.classify(label_embeds)
        extra_outputs.update(graph_outputs)
        return self.build_outputs(
            logits=logits,
            attention=attention,
            features=context_features,
            extra_outputs=extra_outputs,
        )


class MambaLikeMixerBlock(nn.Module):
    def __init__(
        self,
        *,
        feature_dim: int,
        expand: int = 2,
        kernel_size: int = 5,
        dropout: float = 0.2,
    ) -> None:
        super().__init__()
        hidden_dim = feature_dim * max(1, int(expand))
        kernel_size = max(3, int(kernel_size))
        if kernel_size % 2 == 0:
            kernel_size += 1
        self.norm = nn.LayerNorm(feature_dim)
        self.in_proj = nn.Linear(feature_dim, hidden_dim * 2)
        self.depthwise_conv = nn.Conv1d(
            hidden_dim,
            hidden_dim,
            kernel_size=kernel_size,
            padding=kernel_size // 2,
            groups=hidden_dim,
        )
        self.out_proj = nn.Linear(hidden_dim, feature_dim)
        self.dropout = nn.Dropout(dropout)
        self.ffn_norm = nn.LayerNorm(feature_dim)
        self.ffn = nn.Sequential(
            nn.Linear(feature_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, feature_dim),
        )

    def forward(self, features: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        mask_float = mask.unsqueeze(-1).to(dtype=features.dtype)
        hidden, gate = self.in_proj(self.norm(features)).chunk(2, dim=-1)
        hidden = hidden * mask_float
        mixed = self.depthwise_conv(hidden.transpose(1, 2)).transpose(1, 2)
        mixed = mixed[:, : features.shape[1], :]
        mixed = F.silu(mixed) * torch.sigmoid(gate)
        features = features + self.dropout(self.out_proj(mixed))
        features = features + self.dropout(self.ffn(self.ffn_norm(features)))
        return features * mask_float


class MambaMILModel(Exp4BaseModel):
    def __init__(
        self,
        *,
        num_layers: int = 2,
        mamba_expand: int = 2,
        mamba_kernel_size: int = 5,
        **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)
        self.position_encoding = TimePositionEncoding(self.feature_dim)
        self.mixer_blocks = nn.ModuleList(
            [
                MambaLikeMixerBlock(
                    feature_dim=self.feature_dim,
                    expand=mamba_expand,
                    kernel_size=mamba_kernel_size,
                    dropout=float(kwargs.get("dropout", 0.2)),
                )
                for _ in range(max(1, int(num_layers)))
            ]
        )

    def forward(self, images: torch.Tensor, mask: torch.Tensor, **kwargs: Any) -> dict[str, torch.Tensor]:
        del kwargs
        features, extra_outputs = self.encode_instances(images, mask)
        features = self.position_encoding(features, mask)
        for block in self.mixer_blocks:
            features = block(features, mask)
        bag_embeds, attention = self.mil_pool(features, mask)
        label_embeds, graph_outputs = self.refine_labels(bag_embeds)
        logits = self.classify(label_embeds)
        extra_outputs.update(graph_outputs)
        return self.build_outputs(
            logits=logits,
            attention=attention,
            features=features,
            extra_outputs=extra_outputs,
        )


EXP4_CLASS_REGISTRY = {
    "multi_sample_lg_mil": MultiSampleLabelGraphMILModel,
    "full_feature_mil": FullFeatureMILModel,
    "hier_full_mil": HierarchicalFullMILModel,
    "hier_full_lg_mil": HierarchicalFullMILModel,
    "long_mil": LongMILModel,
    "mamba_mil": MambaMILModel,
}
EXP4_MODEL_NAMES = tuple(EXP4_CLASS_REGISTRY.keys())


def build_exp4_model(model_name: str, **kwargs: Any) -> nn.Module:
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
    }
    model_kwargs = {key: value for key, value in kwargs.items() if key in base_keys}
    if model_name == "hier_full_lg_mil":
        model_kwargs["use_label_graph"] = True
        model_kwargs.setdefault("use_quality_gate", True)
        model_kwargs["subbag_size"] = kwargs.get("subbag_size", 8)
    elif model_name == "hier_full_mil":
        model_kwargs["use_label_graph"] = False
        model_kwargs["subbag_size"] = kwargs.get("subbag_size", 8)
    elif model_name == "full_feature_mil":
        model_kwargs["use_label_graph"] = False
    elif model_name == "multi_sample_lg_mil":
        model_kwargs["num_views"] = kwargs.get("num_views", 4)
        model_kwargs["view_keep_ratio"] = kwargs.get("view_keep_ratio", 0.75)
    elif model_name in {"long_mil", "mamba_mil"}:
        model_kwargs.setdefault("use_label_graph", True)
        model_kwargs["num_layers"] = kwargs.get("num_layers", 2)
        if model_name == "long_mil":
            model_kwargs["num_heads"] = kwargs.get("num_heads", 4)
            model_kwargs["hidden_dim"] = kwargs.get("hidden_dim", 1024)
        else:
            model_kwargs["mamba_expand"] = kwargs.get("mamba_expand", 2)
            model_kwargs["mamba_kernel_size"] = kwargs.get("mamba_kernel_size", 5)

    model_cls = EXP4_CLASS_REGISTRY.get(model_name)
    if model_cls is None:
        raise ValueError(f"未知 exp_4 模型名: {model_name}")
    return model_cls(**model_kwargs)
