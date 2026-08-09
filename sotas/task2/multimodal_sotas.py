from __future__ import annotations

import math
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F

from exp_4.models import Exp4BaseModel
from exp_8.models import TextCNNSequenceEncoder


def _masked_mean(values: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    weights = mask.to(dtype=values.dtype).unsqueeze(-1)
    return (values * weights).sum(dim=1) / weights.sum(dim=1).clamp_min(1.0)


def _safe_key_padding_mask(
    values: torch.Tensor,
    mask: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """避免 MultiheadAttention 遇到整行 padding 后产生 NaN。"""

    safe_values = values
    safe_mask = mask.bool().clone()
    empty = ~safe_mask.any(dim=1)
    if empty.any():
        safe_values = values.clone()
        safe_values[empty, 0] = 0.0
        safe_mask[empty, 0] = True
    return safe_values, safe_mask, ~safe_mask


def _symmetric_kl(first: torch.Tensor, second: torch.Tensor, temperature: float = 1.0) -> torch.Tensor:
    temperature = max(float(temperature), 1e-6)
    first_log = F.log_softmax(first / temperature, dim=-1)
    second_log = F.log_softmax(second / temperature, dim=-1)
    first_prob = first_log.exp()
    second_prob = second_log.exp()
    return 0.5 * (
        F.kl_div(first_log, second_prob, reduction="batchmean")
        + F.kl_div(second_log, first_prob, reduction="batchmean")
    ) * (temperature**2)


def _mean_pool_pairs(values: torch.Tensor, mask: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """构造相邻 token 的第二尺度表示。"""

    if values.shape[1] <= 1:
        return values, mask
    if values.shape[1] % 2:
        values = F.pad(values, (0, 0, 0, 1))
        mask = F.pad(mask, (0, 1), value=False)
    batch_size, token_count, feature_dim = values.shape
    paired_values = values.reshape(batch_size, token_count // 2, 2, feature_dim)
    paired_mask = mask.reshape(batch_size, token_count // 2, 2)
    weights = paired_mask.to(dtype=values.dtype).unsqueeze(-1)
    pooled = (paired_values * weights).sum(dim=2) / weights.sum(dim=2).clamp_min(1.0)
    return pooled, paired_mask.any(dim=2)


class Task2MultimodalSOTABase(Exp4BaseModel):
    """统一的数据接口与编码底座，不引入本文的超图推理模块。"""

    def __init__(
        self,
        *,
        text_vocab_size: int = 8192,
        text_embed_dim: int = 128,
        textcnn_kernel_sizes: tuple[int, ...] | list[int] = (2, 3, 4),
        num_heads: int = 4,
        **kwargs: Any,
    ) -> None:
        for unused_key in (
            "num_layers",
            "correlation_threshold",
            "alignment_temperature",
            "contrast_temperature",
            "contrast_queue_size",
            "window_size",
        ):
            kwargs.pop(unused_key, None)
        kwargs["use_label_graph"] = False
        super().__init__(**kwargs)
        self.num_heads = self._valid_head_count(self.feature_dim, num_heads)
        self.dropout_rate = float(kwargs.get("dropout", 0.2))
        self.text_encoder = TextCNNSequenceEncoder(
            vocab_size=int(text_vocab_size),
            token_embed_dim=int(text_embed_dim),
            output_dim=self.feature_dim,
            dropout=self.dropout_rate,
            kernel_sizes=tuple(int(value) for value in textcnn_kernel_sizes),
        )

    @staticmethod
    def _valid_head_count(feature_dim: int, requested: int) -> int:
        heads = max(1, min(int(requested), int(feature_dim)))
        while feature_dim % heads != 0 and heads > 1:
            heads -= 1
        return heads

    def encode_modalities(
        self,
        images: torch.Tensor,
        mask: torch.Tensor,
        watch_token_ids: torch.Tensor | None,
        watch_token_mask: torch.Tensor | None,
    ) -> tuple[
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
    ]:
        image_tokens, _ = self.encode_instances(images, mask)
        image_label_embeds, attention = self.mil_pool(image_tokens, mask)
        text_tokens, text_mask, text_pooled, text_active = self.text_encoder(
            watch_token_ids,
            watch_token_mask,
            batch_size=images.shape[0],
            device=images.device,
        )
        return (
            image_tokens,
            image_label_embeds,
            attention,
            text_tokens,
            text_mask,
            text_pooled,
            text_active,
        )

    @staticmethod
    def build_outputs(
        *,
        logits: torch.Tensor,
        attention: torch.Tensor,
        image_tokens: torch.Tensor,
        extra: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        outputs: dict[str, Any] = {
            "logits": logits,
            "attention": attention,
            "instance_features": image_tokens,
        }
        if extra:
            outputs.update(extra)
        return outputs


class HasanImageTextFusion2024(Task2MultimodalSOTABase):
    """Hasan et al. (2024)：视觉/文本特征拼接后使用轻量 CNN 分类器。"""

    def __init__(self, *, hidden_dim: int = 1024, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self.modality_conv = nn.Sequential(
            nn.Conv1d(self.feature_dim, self.feature_dim, kernel_size=2),
            nn.GELU(),
            nn.Dropout(self.dropout_rate),
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
        image_tokens, image_labels, attention, _, _, text_pooled, _ = self.encode_modalities(
            images, mask, watch_token_ids, watch_token_mask
        )
        image_pooled = image_labels.mean(dim=1)
        modalities = torch.stack([image_pooled, text_pooled], dim=1).transpose(1, 2)
        fused = self.modality_conv(modalities).squeeze(-1)
        return self.build_outputs(
            logits=self.classifier(fused),
            attention=attention,
            image_tokens=image_tokens,
        )


class SAIF2025(Task2MultimodalSOTABase):
    """SAIF (2025)：自适应相关性掩码、KL 对齐和无参数伪注意力融合。"""

    def __init__(
        self,
        *,
        hidden_dim: int = 1024,
        correlation_threshold: float = 0.5,
        alignment_temperature: float = 1.0,
        **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)
        self.correlation_threshold = float(correlation_threshold)
        self.alignment_temperature = float(alignment_temperature)
        self.image_to_text = nn.Linear(self.feature_dim, self.feature_dim)
        self.classifier = nn.Sequential(
            nn.LayerNorm(self.feature_dim * 2),
            nn.Linear(self.feature_dim * 2, int(hidden_dim)),
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
        image_tokens, image_labels, attention, _, _, text_pooled, text_active = self.encode_modalities(
            images, mask, watch_token_ids, watch_token_mask
        )
        image_pooled = image_labels.mean(dim=1)
        mapped_image = self.image_to_text(image_pooled)
        element_correlation = torch.sign(mapped_image) * torch.sign(text_pooled)
        soft_mask = torch.sigmoid(12.0 * (element_correlation - self.correlation_threshold))
        if not self.training:
            soft_mask = (soft_mask >= 0.5).to(dtype=soft_mask.dtype)
        active = text_active.to(dtype=image_pooled.dtype).unsqueeze(-1)
        correlated_image = mapped_image * soft_mask * active
        correlated_text = text_pooled * soft_mask * active

        scores = torch.einsum("bd,bnd->bn", correlated_text, image_tokens)
        scores = scores / math.sqrt(float(self.feature_dim))
        scores = scores.masked_fill(~mask, -1e4)
        pseudo_attention = torch.softmax(scores, dim=-1) * mask.to(dtype=scores.dtype)
        pseudo_attention = pseudo_attention / pseudo_attention.sum(dim=-1, keepdim=True).clamp_min(1e-12)
        attended_image = torch.einsum("bn,bnd->bd", pseudo_attention, image_tokens)
        fused = torch.cat([attended_image, image_pooled], dim=-1)
        alignment = _symmetric_kl(
            correlated_image,
            correlated_text,
            temperature=self.alignment_temperature,
        )
        return self.build_outputs(
            logits=self.classifier(fused),
            attention=attention,
            image_tokens=image_tokens,
            extra={
                "saif_correlation_mask": soft_mask,
                "saif_pseudo_attention": pseudo_attention,
                "aux_losses": {"alignment": alignment},
            },
        )


class CrossReplaceBlock(nn.Module):
    """MMTF 的低注意力 token 跨模态子空间交换。"""

    def __init__(self, feature_dim: int, num_heads: int, hidden_dim: int, dropout: float) -> None:
        super().__init__()
        self.self_attention = nn.MultiheadAttention(
            feature_dim,
            num_heads,
            dropout=dropout,
            batch_first=True,
        )
        self.norm1 = nn.LayerNorm(feature_dim)
        self.norm2 = nn.LayerNorm(feature_dim)
        self.feed_forward = nn.Sequential(
            nn.Linear(feature_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, feature_dim),
            nn.Dropout(dropout),
        )

    def _encode(
        self,
        values: torch.Tensor,
        mask: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        safe_values, safe_mask, padding_mask = _safe_key_padding_mask(values, mask)
        attended, weights = self.self_attention(
            safe_values,
            safe_values,
            safe_values,
            key_padding_mask=padding_mask,
            need_weights=True,
            average_attn_weights=True,
        )
        encoded = self.norm1(safe_values + attended)
        encoded = self.norm2(encoded + self.feed_forward(encoded))
        encoded = encoded * mask.unsqueeze(-1).to(dtype=encoded.dtype)
        return encoded, weights, safe_mask

    def forward(
        self,
        image_tokens: torch.Tensor,
        image_mask: torch.Tensor,
        text_tokens: torch.Tensor,
        text_mask: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        image_encoded, image_weights, _ = self._encode(image_tokens, image_mask)
        text_encoded, text_weights, _ = self._encode(text_tokens, text_mask)
        image_importance = image_weights.mean(dim=1)
        text_importance = text_weights.mean(dim=1)
        image_threshold = image_importance.masked_fill(~image_mask, float("nan")).nanmedian(dim=1).values
        text_threshold = text_importance.masked_fill(~text_mask, float("nan")).nanmedian(dim=1).values
        image_low = (image_importance < image_threshold.unsqueeze(-1)) & image_mask
        text_low = (text_importance < text_threshold.unsqueeze(-1)) & text_mask
        image_mean = _masked_mean(image_encoded, image_mask)
        text_mean = _masked_mean(text_encoded, text_mask)
        image_encoded = torch.where(image_low.unsqueeze(-1), text_mean.unsqueeze(1), image_encoded)
        text_encoded = torch.where(text_low.unsqueeze(-1), image_mean.unsqueeze(1), text_encoded)
        return image_encoded, text_encoded


class MMTF2025(Task2MultimodalSOTABase):
    """MMTF (2025)：多尺度 token 与跨模态低注意力子空间交换。"""

    def __init__(
        self,
        *,
        hidden_dim: int = 1024,
        num_layers: int = 2,
        **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)
        self.image_cls = nn.Parameter(torch.randn(1, 1, self.feature_dim) * 0.02)
        self.text_cls = nn.Parameter(torch.randn(1, 1, self.feature_dim) * 0.02)
        self.layers = nn.ModuleList(
            [
                CrossReplaceBlock(
                    self.feature_dim,
                    self.num_heads,
                    int(hidden_dim),
                    self.dropout_rate,
                )
                for _ in range(max(1, int(num_layers)))
            ]
        )
        self.fusion = nn.Sequential(
            nn.LayerNorm(self.feature_dim * 2),
            nn.Linear(self.feature_dim * 2, self.feature_dim),
            nn.GELU(),
            nn.Dropout(self.dropout_rate),
        )
        self.classifier = nn.Linear(self.feature_dim, self.num_labels)

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
        image_coarse, image_coarse_mask = _mean_pool_pairs(image_tokens, mask)
        text_coarse, text_coarse_mask = _mean_pool_pairs(text_tokens, text_mask)
        image_tokens = torch.cat([image_tokens, image_coarse], dim=1)
        image_mask = torch.cat([mask, image_coarse_mask], dim=1)
        text_tokens = torch.cat([text_tokens, text_coarse], dim=1)
        text_mask = torch.cat([text_mask, text_coarse_mask], dim=1)

        batch_size = images.shape[0]
        image_tokens = torch.cat([self.image_cls.expand(batch_size, -1, -1), image_tokens], dim=1)
        text_tokens = torch.cat([self.text_cls.expand(batch_size, -1, -1), text_tokens], dim=1)
        image_mask = F.pad(image_mask, (1, 0), value=True)
        text_mask = F.pad(text_mask, (1, 0), value=True)
        for layer in self.layers:
            image_tokens, text_tokens = layer(image_tokens, image_mask, text_tokens, text_mask)

        image_cls = image_tokens[:, 0]
        text_cls = text_tokens[:, 0]
        fused = self.fusion(torch.cat([image_cls, text_cls], dim=-1))
        alignment = 1.0 - F.cosine_similarity(image_cls, text_cls, dim=-1).mean()
        return self.build_outputs(
            logits=self.classifier(fused),
            attention=attention,
            image_tokens=image_tokens[:, 1 : 1 + mask.shape[1]],
            extra={"aux_losses": {"alignment": alignment}},
        )


class RadFuse2025(Task2MultimodalSOTABase):
    """RadFuse (2025)：模态内自注意力、双向交叉注意力与 InfoNCE。"""

    def __init__(
        self,
        *,
        hidden_dim: int = 1024,
        num_layers: int = 2,
        contrast_temperature: float = 0.07,
        contrast_queue_size: int = 256,
        **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)
        image_layer = nn.TransformerEncoderLayer(
            self.feature_dim,
            self.num_heads,
            int(hidden_dim),
            self.dropout_rate,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        text_layer = nn.TransformerEncoderLayer(
            self.feature_dim,
            self.num_heads,
            int(hidden_dim),
            self.dropout_rate,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.image_encoder = nn.TransformerEncoder(image_layer, num_layers=max(1, int(num_layers)))
        self.report_encoder = nn.TransformerEncoder(text_layer, num_layers=max(1, int(num_layers)))
        self.image_to_text = nn.MultiheadAttention(
            self.feature_dim, self.num_heads, dropout=self.dropout_rate, batch_first=True
        )
        self.text_to_image = nn.MultiheadAttention(
            self.feature_dim, self.num_heads, dropout=self.dropout_rate, batch_first=True
        )
        self.fusion = nn.Sequential(
            nn.LayerNorm(self.feature_dim * 4),
            nn.Linear(self.feature_dim * 4, int(hidden_dim)),
            nn.GELU(),
            nn.Dropout(self.dropout_rate),
            nn.Linear(int(hidden_dim), self.num_labels),
        )
        self.contrast_temperature = float(contrast_temperature)
        self.contrast_queue_size = max(1, int(contrast_queue_size))
        self.register_buffer(
            "image_queue",
            torch.zeros(self.contrast_queue_size, self.feature_dim),
        )
        self.register_buffer(
            "text_queue",
            torch.zeros(self.contrast_queue_size, self.feature_dim),
        )
        self.register_buffer("queue_pointer", torch.zeros((), dtype=torch.long))
        self.register_buffer("queue_filled", torch.zeros((), dtype=torch.long))

    def _queued_infonce(
        self,
        image_embed: torch.Tensor,
        text_embed: torch.Tensor,
    ) -> torch.Tensor:
        """用跨batch队列为batch=1提供负样本，保持RadFuse对比目标有效。"""

        image_embed = F.normalize(image_embed, dim=-1)
        text_embed = F.normalize(text_embed, dim=-1)
        filled = int(self.queue_filled.item())
        if filled > 0:
            positive = (image_embed * text_embed).sum(dim=-1, keepdim=True)
            text_negatives = self.text_queue[:filled].detach().clone()
            image_negatives = self.image_queue[:filled].detach().clone()
            image_logits = torch.cat(
                [positive, image_embed @ text_negatives.transpose(0, 1)],
                dim=1,
            )
            text_logits = torch.cat(
                [positive, text_embed @ image_negatives.transpose(0, 1)],
                dim=1,
            )
            image_logits = image_logits / max(self.contrast_temperature, 1e-6)
            text_logits = text_logits / max(self.contrast_temperature, 1e-6)
            targets = torch.zeros(image_embed.shape[0], dtype=torch.long, device=image_embed.device)
            loss = 0.5 * (
                F.cross_entropy(image_logits, targets)
                + F.cross_entropy(text_logits, targets)
            )
        else:
            loss = image_embed.new_zeros(())

        if self.training:
            with torch.no_grad():
                for image_item, text_item in zip(image_embed.detach(), text_embed.detach()):
                    pointer = int(self.queue_pointer.item())
                    self.image_queue[pointer].copy_(image_item)
                    self.text_queue[pointer].copy_(text_item)
                    self.queue_pointer.fill_((pointer + 1) % self.contrast_queue_size)
                    self.queue_filled.fill_(min(filled + 1, self.contrast_queue_size))
                    filled = int(self.queue_filled.item())
        return loss

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
        safe_text, safe_text_mask, text_padding = _safe_key_padding_mask(text_tokens, text_mask)
        image_encoded = self.image_encoder(image_tokens, src_key_padding_mask=~mask)
        text_encoded = self.report_encoder(safe_text, src_key_padding_mask=text_padding)
        image_encoded = image_encoded * mask.unsqueeze(-1).to(dtype=image_encoded.dtype)
        text_encoded = text_encoded * text_mask.unsqueeze(-1).to(dtype=text_encoded.dtype)

        image_cross, image_to_text_weights = self.image_to_text(
            image_encoded,
            safe_text,
            safe_text,
            key_padding_mask=text_padding,
            need_weights=True,
        )
        text_cross, text_to_image_weights = self.text_to_image(
            safe_text,
            image_encoded,
            image_encoded,
            key_padding_mask=~mask,
            need_weights=True,
        )
        image_pool = _masked_mean(image_encoded, mask)
        text_pool = _masked_mean(text_encoded, safe_text_mask)
        image_cross_pool = _masked_mean(image_cross, mask)
        text_cross_pool = _masked_mean(text_cross, safe_text_mask)
        fused = torch.cat([image_pool, text_pool, image_cross_pool, text_cross_pool], dim=-1)
        contrastive = self._queued_infonce(image_pool, text_pool)
        return self.build_outputs(
            logits=self.fusion(fused),
            attention=attention,
            image_tokens=image_encoded,
            extra={
                "image_to_text_attention": image_to_text_weights,
                "text_to_image_attention": text_to_image_weights,
                "aux_losses": {"contrastive": contrastive},
            },
        )


class MMFNet2024(Task2MultimodalSOTABase):
    """MMFNet (2024)：移位窗口视觉注意力与异构双向交叉注意力。"""

    def __init__(
        self,
        *,
        hidden_dim: int = 1024,
        window_size: int = 8,
        **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)
        self.window_size = max(2, int(window_size))
        self.window_attention = nn.MultiheadAttention(
            self.feature_dim, self.num_heads, dropout=self.dropout_rate, batch_first=True
        )
        self.image_to_text = nn.MultiheadAttention(
            self.feature_dim, self.num_heads, dropout=self.dropout_rate, batch_first=True
        )
        self.text_to_image = nn.MultiheadAttention(
            self.feature_dim, self.num_heads, dropout=self.dropout_rate, batch_first=True
        )
        self.image_norm = nn.LayerNorm(self.feature_dim)
        self.text_norm = nn.LayerNorm(self.feature_dim)
        self.image_classifier = nn.Linear(self.feature_dim, self.num_labels)
        self.text_classifier = nn.Linear(self.feature_dim, self.num_labels)
        self.fusion = nn.Sequential(
            nn.LayerNorm(self.feature_dim * 4),
            nn.Linear(self.feature_dim * 4, int(hidden_dim)),
            nn.GELU(),
            nn.Dropout(self.dropout_rate),
            nn.Linear(int(hidden_dim), self.num_labels),
        )

    def _shifted_window_encode(
        self,
        values: torch.Tensor,
        mask: torch.Tensor,
        shift: int,
    ) -> torch.Tensor:
        if shift:
            values = torch.roll(values, shifts=-shift, dims=1)
            mask = torch.roll(mask, shifts=-shift, dims=1)
        encoded_chunks: list[torch.Tensor] = []
        for start in range(0, values.shape[1], self.window_size):
            chunk = values[:, start : start + self.window_size]
            chunk_mask = mask[:, start : start + self.window_size]
            safe_chunk, _, padding = _safe_key_padding_mask(chunk, chunk_mask)
            attended, _ = self.window_attention(
                safe_chunk,
                safe_chunk,
                safe_chunk,
                key_padding_mask=padding,
                need_weights=False,
            )
            encoded_chunks.append((safe_chunk + attended) * chunk_mask.unsqueeze(-1).to(dtype=chunk.dtype))
        encoded = torch.cat(encoded_chunks, dim=1)
        if shift:
            encoded = torch.roll(encoded, shifts=shift, dims=1)
        return encoded

    def forward(
        self,
        images: torch.Tensor,
        mask: torch.Tensor,
        labels: torch.Tensor | None = None,
        watch_token_ids: torch.Tensor | None = None,
        watch_token_mask: torch.Tensor | None = None,
        **kwargs: Any,
    ) -> dict[str, Any]:
        del kwargs
        image_tokens, _, attention, text_tokens, text_mask, _, _ = self.encode_modalities(
            images, mask, watch_token_ids, watch_token_mask
        )
        image_tokens = self._shifted_window_encode(image_tokens, mask, shift=0)
        image_tokens = self._shifted_window_encode(
            image_tokens,
            mask,
            shift=self.window_size // 2,
        )
        safe_text, safe_text_mask, text_padding = _safe_key_padding_mask(text_tokens, text_mask)
        image_cross, image_to_text_weights = self.image_to_text(
            image_tokens,
            safe_text,
            safe_text,
            key_padding_mask=text_padding,
            need_weights=True,
        )
        text_cross, text_to_image_weights = self.text_to_image(
            safe_text,
            image_tokens,
            image_tokens,
            key_padding_mask=~mask,
            need_weights=True,
        )
        image_pool = self.image_norm(_masked_mean(image_tokens, mask))
        text_pool = self.text_norm(_masked_mean(safe_text, safe_text_mask))
        image_cross_pool = _masked_mean(image_cross, mask)
        text_cross_pool = _masked_mean(text_cross, safe_text_mask)
        logits = self.fusion(
            torch.cat([image_pool, text_pool, image_cross_pool, text_cross_pool], dim=-1)
        )
        image_logits = self.image_classifier(image_pool)
        text_logits = self.text_classifier(text_pool)
        if labels is None:
            image_aux = logits.new_zeros(())
            text_aux = logits.new_zeros(())
        else:
            targets = labels.to(dtype=logits.dtype)
            image_aux = F.binary_cross_entropy_with_logits(image_logits, targets)
            text_aux = F.binary_cross_entropy_with_logits(text_logits, targets)
        return self.build_outputs(
            logits=logits,
            attention=attention,
            image_tokens=image_tokens,
            extra={
                "image_to_text_attention": image_to_text_weights,
                "text_to_image_attention": text_to_image_weights,
                "image_only_logits": image_logits,
                "text_only_logits": text_logits,
                "aux_losses": {
                    "image_branch": image_aux,
                    "text_branch": text_aux,
                },
            },
        )


TASK2_MULTIMODAL_SOTA_CLASS_REGISTRY = {
    "task2_hasan_itf_2024": HasanImageTextFusion2024,
    "task2_mmfnet_2024": MMFNet2024,
    "task2_saif_2025": SAIF2025,
    "task2_mmtf_2025": MMTF2025,
    "task2_radfuse_2025": RadFuse2025,
}
TASK2_MULTIMODAL_SOTA_MODEL_NAMES = tuple(TASK2_MULTIMODAL_SOTA_CLASS_REGISTRY)


def build_task2_multimodal_sota(model_name: str, **kwargs: Any) -> nn.Module:
    model_cls = TASK2_MULTIMODAL_SOTA_CLASS_REGISTRY.get(model_name)
    if model_cls is None:
        raise ValueError(f"未知 TASK2 图文 SOTA 模型名: {model_name}")
    return model_cls(**kwargs)


__all__ = [
    "HasanImageTextFusion2024",
    "MMFNet2024",
    "SAIF2025",
    "MMTF2025",
    "RadFuse2025",
    "TASK2_MULTIMODAL_SOTA_CLASS_REGISTRY",
    "TASK2_MULTIMODAL_SOTA_MODEL_NAMES",
    "build_task2_multimodal_sota",
]
