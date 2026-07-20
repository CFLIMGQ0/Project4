from __future__ import annotations

from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F


def masked_mean(values: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    weights = mask.unsqueeze(-1).to(dtype=values.dtype)
    return (values * weights).sum(dim=1) / weights.sum(dim=1).clamp_min(1.0)


class HashedMeanTextEncoder(nn.Module):
    """论文 HashTextEncoder 风格：哈希 ID、token MLP、均值池化和 pooled MLP。"""

    def __init__(self, vocab_size: int, embed_dim: int, output_dim: int, dropout: float) -> None:
        super().__init__()
        self.output_dim = output_dim
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.token_projection = nn.Sequential(
            nn.LayerNorm(embed_dim),
            nn.Linear(embed_dim, output_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(output_dim, output_dim),
        )
        self.pooled_projection = nn.Sequential(
            nn.LayerNorm(output_dim),
            nn.Linear(output_dim, output_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(output_dim, output_dim),
        )

    def forward(self, token_ids: torch.Tensor, token_mask: torch.Tensor) -> torch.Tensor:
        token_features = self.token_projection(self.embedding(token_ids))
        return self.pooled_projection(masked_mean(token_features, token_mask))


class VocabAttentionTextEncoder(nn.Module):
    """训练集词表 embedding + 可学习 token attention pooling。"""

    def __init__(self, vocab_size: int, embed_dim: int, output_dim: int, dropout: float) -> None:
        super().__init__()
        self.output_dim = output_dim
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.token_projection = nn.Sequential(
            nn.LayerNorm(embed_dim),
            nn.Linear(embed_dim, output_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        self.attention = nn.Sequential(
            nn.Linear(output_dim, max(32, output_dim // 2)),
            nn.Tanh(),
            nn.Linear(max(32, output_dim // 2), 1),
        )

    def forward(self, token_ids: torch.Tensor, token_mask: torch.Tensor) -> torch.Tensor:
        features = self.token_projection(self.embedding(token_ids))
        scores = self.attention(features).squeeze(-1).masked_fill(~token_mask, -1e4)
        weights = torch.softmax(scores, dim=-1) * token_mask.to(dtype=features.dtype)
        weights = weights / weights.sum(dim=-1, keepdim=True).clamp_min(1e-12)
        return torch.einsum("bt,btd->bd", weights, features)


class TextCNNEncoder(nn.Module):
    """训练集词表 embedding + 多尺度一维卷积。"""

    def __init__(
        self,
        vocab_size: int,
        embed_dim: int,
        output_dim: int,
        dropout: float,
        kernel_sizes: tuple[int, ...],
    ) -> None:
        super().__init__()
        self.output_dim = output_dim
        channels = max(64, output_dim // len(kernel_sizes))
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.convolutions = nn.ModuleList(
            nn.Conv1d(embed_dim, channels, kernel_size=kernel_size) for kernel_size in kernel_sizes
        )
        self.projection = nn.Sequential(
            nn.LayerNorm(channels * len(kernel_sizes)),
            nn.Linear(channels * len(kernel_sizes), output_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )

    def forward(self, token_ids: torch.Tensor, token_mask: torch.Tensor) -> torch.Tensor:
        values = self.embedding(token_ids).transpose(1, 2)
        pooled: list[torch.Tensor] = []
        float_mask = token_mask.to(dtype=values.dtype).unsqueeze(1)
        for convolution in self.convolutions:
            features = torch.relu(convolution(values))
            kernel_size = int(convolution.kernel_size[0])
            window_counts = F.conv1d(
                float_mask,
                torch.ones(1, 1, kernel_size, device=values.device, dtype=values.dtype),
            )
            valid_windows = window_counts >= float(kernel_size)
            masked_features = features.masked_fill(~valid_windows, -1e4)
            pooled_features = masked_features.amax(dim=-1)
            has_valid_window = valid_windows.any(dim=-1)
            pooled.append(torch.where(has_valid_window, pooled_features, torch.zeros_like(pooled_features)))
        return self.projection(torch.cat(pooled, dim=-1))


class BiGRUTextEncoder(nn.Module):
    """训练集词表 embedding + 双向 GRU。"""

    def __init__(
        self,
        vocab_size: int,
        embed_dim: int,
        output_dim: int,
        dropout: float,
        rnn_hidden_dim: int,
    ) -> None:
        super().__init__()
        self.output_dim = output_dim
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.gru = nn.GRU(embed_dim, rnn_hidden_dim, batch_first=True, bidirectional=True)
        self.projection = nn.Sequential(
            nn.LayerNorm(rnn_hidden_dim * 2),
            nn.Linear(rnn_hidden_dim * 2, output_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )

    def forward(self, token_ids: torch.Tensor, token_mask: torch.Tensor) -> torch.Tensor:
        sequence, _ = self.gru(self.embedding(token_ids))
        return self.projection(masked_mean(sequence, token_mask))


class TransformerTextEncoder(nn.Module):
    """训练集词表 embedding + 位置编码 + Transformer encoder。"""

    def __init__(
        self,
        vocab_size: int,
        output_dim: int,
        dropout: float,
        max_length: int,
        num_layers: int,
        num_heads: int,
    ) -> None:
        super().__init__()
        self.output_dim = output_dim
        self.embedding = nn.Embedding(vocab_size, output_dim, padding_idx=0)
        self.position_embedding = nn.Parameter(torch.zeros(1, max_length, output_dim))
        layer = nn.TransformerEncoderLayer(
            d_model=output_dim,
            nhead=num_heads,
            dim_feedforward=output_dim * 4,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(layer, num_layers=num_layers)
        self.output_norm = nn.LayerNorm(output_dim)

    def forward(self, token_ids: torch.Tensor, token_mask: torch.Tensor) -> torch.Tensor:
        features = self.embedding(token_ids) + self.position_embedding[:, : token_ids.shape[1]]
        features = self.encoder(features, src_key_padding_mask=~token_mask)
        return self.output_norm(masked_mean(features, token_mask))


class TextMLPClassifier(nn.Module):
    """所有 exp10 编码器共用的三标签 MLP 分类头。"""

    def __init__(
        self,
        encoder: nn.Module,
        encoder_dim: int,
        mlp_hidden_dim: int,
        num_labels: int,
        dropout: float,
    ) -> None:
        super().__init__()
        self.encoder = encoder
        self.classifier = nn.Sequential(
            nn.LayerNorm(encoder_dim),
            nn.Linear(encoder_dim, mlp_hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(mlp_hidden_dim, num_labels),
        )

    def forward(self, token_ids: torch.Tensor, token_mask: torch.Tensor) -> torch.Tensor:
        return self.classifier(self.encoder(token_ids, token_mask))


def build_text_classifier(
    model_name: str,
    *,
    vocabulary_size: int,
    hash_vocab_size: int,
    num_labels: int,
    max_length: int,
    model_config: dict[str, Any],
) -> TextMLPClassifier:
    embed_dim = int(model_config["token_embed_dim"])
    encoder_dim = int(model_config["encoder_dim"])
    dropout = float(model_config["dropout"])
    common_vocab_size = hash_vocab_size if model_name == "hashed_mean_encoder" else vocabulary_size

    if model_name == "hashed_mean_encoder":
        encoder = HashedMeanTextEncoder(common_vocab_size, embed_dim, encoder_dim, dropout)
    elif model_name == "vocab_attention_encoder":
        encoder = VocabAttentionTextEncoder(common_vocab_size, embed_dim, encoder_dim, dropout)
    elif model_name == "textcnn_encoder":
        encoder = TextCNNEncoder(
            common_vocab_size,
            embed_dim,
            encoder_dim,
            dropout,
            tuple(int(value) for value in model_config["textcnn_kernel_sizes"]),
        )
    elif model_name == "bigru_encoder":
        encoder = BiGRUTextEncoder(
            common_vocab_size,
            embed_dim,
            encoder_dim,
            dropout,
            int(model_config["rnn_hidden_dim"]),
        )
    elif model_name == "transformer_encoder":
        encoder = TransformerTextEncoder(
            common_vocab_size,
            encoder_dim,
            dropout,
            max_length,
            int(model_config["transformer_layers"]),
            int(model_config["transformer_heads"]),
        )
    else:
        raise ValueError(f"未知 exp10 文本编码器: {model_name}")

    return TextMLPClassifier(
        encoder=encoder,
        encoder_dim=encoder.output_dim,
        mlp_hidden_dim=int(model_config["mlp_hidden_dim"]),
        num_labels=num_labels,
        dropout=dropout,
    )


__all__ = ["TextMLPClassifier", "build_text_classifier"]
