from __future__ import annotations

from typing import Any

import torch
import torch.nn as nn

from .multimodal_sotas import Task2MultimodalSOTABase, _masked_mean, _safe_key_padding_mask


class CaMCheXTaskAdapted(Task2MultimodalSOTABase):
    """将 CaMCheX 的多视图、文本上下文和 ML-Decoder 适配到有序三维检查。"""

    def __init__(self, *, hidden_dim: int = 1024, num_layers: int = 2, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        from third_party.camchex_reference.ml_decoder import MLDecoder

        self.segment_embedding = nn.Parameter(torch.randn(2, self.feature_dim) * 0.02)
        layer = nn.TransformerEncoderLayer(
            d_model=self.feature_dim,
            nhead=self.num_heads,
            dim_feedforward=int(hidden_dim),
            dropout=self.dropout_rate,
            activation="gelu",
            batch_first=True,
        )
        self.transformer_encoder = nn.TransformerEncoder(layer, num_layers=max(1, int(num_layers)))
        self.decoder = MLDecoder(
            num_classes=self.num_labels,
            initial_num_features=self.feature_dim,
            decoder_embedding=self.feature_dim,
        )

    def forward(
        self,
        images: torch.Tensor,
        mask: torch.Tensor,
        watch_token_ids: torch.Tensor | None = None,
        watch_token_mask: torch.Tensor | None = None,
        **kwargs: Any,
    ) -> dict[str, Any]:
        del kwargs
        image_tokens, _, attention, text_tokens, text_mask, _, _ = self.encode_modalities(
            images, mask, watch_token_ids, watch_token_mask
        )
        safe_text, safe_text_mask, _ = _safe_key_padding_mask(text_tokens, text_mask)
        image_tokens = image_tokens + self.segment_embedding[0].view(1, 1, -1)
        safe_text = safe_text + self.segment_embedding[1].view(1, 1, -1)
        tokens = torch.cat([image_tokens, safe_text], dim=1)
        valid = torch.cat([mask, safe_text_mask], dim=1)
        encoded = self.transformer_encoder(tokens, src_key_padding_mask=~valid)
        return self.build_outputs(
            logits=self.decoder(encoded, mask=~valid),
            attention=attention,
            image_tokens=image_tokens,
        )


class Med3DVLMTaskAdapted(Task2MultimodalSOTABase):
    """使用 Med3DVLM 官方 low/high hybrid MLP projector 的任务适配版本。"""

    def __init__(self, *, hidden_dim: int = 1024, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        from third_party.med3dvlm_reference.mlp import LowHighHybridMLP

        self.visual_projector = LowHighHybridMLP(
            low_input_size=self.feature_dim,
            high_input_size=self.feature_dim,
            output_size=self.feature_dim,
            mlp_depth=2,
        )
        self.text_cross_attention = nn.MultiheadAttention(
            self.feature_dim,
            self.num_heads,
            dropout=self.dropout_rate,
            batch_first=True,
        )
        self.classifier = nn.Sequential(
            nn.LayerNorm(self.feature_dim),
            nn.Linear(self.feature_dim, int(hidden_dim)),
            nn.GELU(),
            nn.Dropout(self.dropout_rate),
            nn.Linear(int(hidden_dim), self.num_labels),
        )

    def forward(
        self,
        images: torch.Tensor,
        mask: torch.Tensor,
        watch_token_ids: torch.Tensor | None = None,
        watch_token_mask: torch.Tensor | None = None,
        **kwargs: Any,
    ) -> dict[str, Any]:
        del kwargs
        image_tokens, _, attention, text_tokens, text_mask, _, _ = self.encode_modalities(
            images, mask, watch_token_ids, watch_token_mask
        )
        image_mean = _masked_mean(image_tokens, mask)
        masked = image_tokens.masked_fill(~mask.unsqueeze(-1), float("-inf"))
        image_max = masked.amax(dim=1)
        image_max = torch.where(torch.isfinite(image_max), image_max, torch.zeros_like(image_max))
        visual_tokens = self.visual_projector(
            (image_mean.unsqueeze(1), image_max.unsqueeze(1))
        )
        safe_text, _, text_padding = _safe_key_padding_mask(text_tokens, text_mask)
        retrieved, _ = self.text_cross_attention(
            visual_tokens,
            safe_text,
            safe_text,
            key_padding_mask=text_padding,
            need_weights=False,
        )
        fused = (visual_tokens + retrieved).mean(dim=1)
        return self.build_outputs(
            logits=self.classifier(fused),
            attention=attention,
            image_tokens=image_tokens,
        )


class M3FMTaskAdapted(Task2MultimodalSOTABase):
    """使用 M3FM 任务提示查询和多尺度图文 token 的任务适配版本。"""

    def __init__(self, *, hidden_dim: int = 1024, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self.task_prompt = nn.Parameter(torch.randn(1, 1, self.feature_dim) * 0.02)
        self.task_attention = nn.MultiheadAttention(
            self.feature_dim,
            self.num_heads,
            dropout=self.dropout_rate,
            batch_first=True,
        )
        self.task_norm = nn.LayerNorm(self.feature_dim)
        self.task_mlp = nn.Sequential(
            nn.Linear(self.feature_dim, int(hidden_dim)),
            nn.GELU(),
            nn.Dropout(self.dropout_rate),
            nn.Linear(int(hidden_dim), self.feature_dim),
        )
        self.classifier = nn.Sequential(
            nn.LayerNorm(self.feature_dim),
            nn.Linear(self.feature_dim, int(hidden_dim)),
            nn.GELU(),
            nn.Dropout(self.dropout_rate),
            nn.Linear(int(hidden_dim), self.num_labels),
        )

    def forward(
        self,
        images: torch.Tensor,
        mask: torch.Tensor,
        watch_token_ids: torch.Tensor | None = None,
        watch_token_mask: torch.Tensor | None = None,
        **kwargs: Any,
    ) -> dict[str, Any]:
        del kwargs
        image_tokens, _, attention, text_tokens, text_mask, _, _ = self.encode_modalities(
            images, mask, watch_token_ids, watch_token_mask
        )
        midpoint = max(1, image_tokens.shape[1] // 2)
        first_mask = mask.clone()
        first_mask[:, midpoint:] = False
        second_mask = mask.clone()
        second_mask[:, :midpoint] = False
        visual_scales = torch.stack(
            [
                _masked_mean(image_tokens, mask),
                _masked_mean(image_tokens, first_mask),
                _masked_mean(image_tokens, second_mask),
            ],
            dim=1,
        )
        safe_text, safe_text_mask, text_padding = _safe_key_padding_mask(text_tokens, text_mask)
        context = torch.cat([visual_scales, safe_text], dim=1)
        context_mask = torch.cat(
            [torch.ones(images.shape[0], 3, dtype=torch.bool, device=images.device), safe_text_mask], dim=1
        )
        query = self.task_prompt.expand(images.shape[0], -1, -1)
        task_output, _ = self.task_attention(
            query,
            context,
            context,
            key_padding_mask=~context_mask,
            need_weights=False,
        )
        task_output = self.task_norm(query + task_output)
        task_output = task_output + self.task_mlp(task_output)
        return self.build_outputs(
            logits=self.classifier(task_output[:, 0]),
            attention=attention,
            image_tokens=image_tokens,
        )


TASK_ADAPTED_VLM_REGISTRY = {
    "task2_camchex_adapted": CaMCheXTaskAdapted,
    "task2_med3dvlm_adapted": Med3DVLMTaskAdapted,
    "task2_m3fm_adapted": M3FMTaskAdapted,
}


__all__ = [
    "CaMCheXTaskAdapted",
    "Med3DVLMTaskAdapted",
    "M3FMTaskAdapted",
    "TASK_ADAPTED_VLM_REGISTRY",
]
