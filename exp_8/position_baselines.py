"""原论文位置机制的CT切片注意力适配；不提供伪造的标量恢复坐标。"""
from __future__ import annotations

import math

import torch
from torch import nn
from torch.nn import functional as functional


class ComRoPEAP(nn.Module):
    def __init__(self, heads: int, head_dim: int, block_size: int = 2):
        super().__init__()
        if head_dim % block_size:
            raise ValueError("ComRoPE头维度必须整除旋转块大小")
        self.block_size = block_size
        self.freqs = nn.Parameter(torch.randn(heads, head_dim // block_size, block_size, block_size))

    def rotations(self, positions):
        generator = self.freqs - self.freqs.transpose(-1, -2)
        return torch.matrix_exp(positions[:, None, :, None, None, None] * generator[None, :, None])

    def forward(self, query, key, positions):
        rotation = self.rotations(positions)
        def apply(value):
            blocks = value.reshape(*value.shape[:-1], -1, self.block_size, 1)
            return (rotation @ blocks).reshape_as(value)
        return apply(query), apply(key)


class VideoRoPE(nn.Module):
    def __init__(self, head_dim: int, theta: float = 1000000.0, temporal_spacing: float = 2.0):
        super().__init__()
        if head_dim != 128:
            raise ValueError("本次复现固定使用官方16/24/24划分，对应128维注意力头")
        self.temporal_spacing = temporal_spacing
        self.register_buffer("inverse_frequency", theta ** (-torch.arange(0, head_dim, 2).float() / head_dim))
        axes = [1 if index % 2 == 0 else 2 for index in range(48)] + [0] * 16
        self.register_buffer("frequency_axes", torch.tensor(axes, dtype=torch.long))

    def angles(self, coordinates):
        selected = coordinates[:, self.frequency_axes].permute(0, 2, 1)
        phase = selected * self.inverse_frequency
        return torch.cat((phase, phase), dim=-1)[:, None]

    def forward(self, query, key, coordinates):
        angles = self.angles(coordinates)
        def apply(value):
            first, second = value.chunk(2, dim=-1)
            rotated = torch.cat((-second, first), dim=-1)
            return value * angles.cos() + rotated * angles.sin()
        return apply(query), apply(key)


class DAPEV2Kerple(nn.Module):
    def __init__(self, heads: int, width: int = 32, kernel_size: int = 3):
        super().__init__()
        self.bias_p = nn.Parameter(torch.rand(heads, 1, 1) * 2)
        self.bias_a = nn.Parameter(torch.rand(heads, 1, 1))
        self.mlp2 = nn.Sequential(
            nn.Conv2d(heads * 2, width, (1, kernel_size), padding=(0, kernel_size // 2)),
            nn.LeakyReLU(),
            nn.Conv2d(width, heads, (1, kernel_size), padding=(0, kernel_size // 2)),
        )

    def forward(self, scores, valid, causal=False):
        slots = torch.arange(scores.shape[-1], device=scores.device, dtype=scores.dtype)
        difference = slots[:, None] - slots[None, :]
        difference = difference.tril() if causal else difference.abs()
        with torch.no_grad():
            self.bias_p.clamp_(min=0.01)
            self.bias_a.clamp_(min=0.01)
        bias = -self.bias_p * torch.log1p(self.bias_a * difference)
        features = torch.cat((scores, bias[None].expand(scores.shape[0], -1, -1, -1)), dim=1)
        pair_valid = valid[:, None, :, None] & valid[:, None, None, :]
        features = features.masked_fill(~pair_valid, 0)
        if causal:
            features = features.tril()
        hidden = self.mlp2[1](self.mlp2[0](features))
        hidden = hidden.masked_fill(~pair_valid, 0)
        return scores + bias + self.mlp2[2](hidden)


def path_causal_scores(query, key, direction, beta):
    length = query.shape[-2]
    weighted = direction * beta[..., None]
    identity = torch.eye(length, device=query.device, dtype=query.dtype)
    triangular = (weighted @ direction.transpose(-1, -2)).tril(-1)
    projected_keys = (weighted @ key.transpose(-1, -2)).tril(-1)
    transformed = torch.linalg.solve_triangular(identity + triangular, projected_keys,
                                                upper=False, unitriangular=True)
    return ((query @ key.transpose(-1, -2)).tril()
            - (query @ direction.transpose(-1, -2)).tril() @ transformed)


class PaTH(nn.Module):
    def __init__(self, feature_dim: int, heads: int):
        super().__init__()
        self.heads = heads
        self.w_proj = nn.Sequential(nn.Linear(feature_dim, 32, bias=False),
                                    nn.Linear(32, feature_dim, bias=False))
        self.w_conv = nn.Conv1d(feature_dim, feature_dim, 3, groups=feature_dim, bias=False)
        self.beta_proj = nn.Linear(feature_dim, heads)

    def transition(self, features, valid):
        projected = self.w_proj(features) * valid[..., None]
        convolved = self.w_conv(functional.pad(projected.transpose(1, 2), (2, 0)))
        direction = functional.silu(convolved).transpose(1, 2)
        direction = direction.reshape(*features.shape[:2], self.heads, -1).transpose(1, 2)
        direction = functional.normalize(direction, dim=-1) * valid[:, None, :, None]
        beta = 2 * self.beta_proj(features).sigmoid().transpose(1, 2)
        return direction, beta * valid[:, None]

    def forward(self, query, key, features, valid, causal=False):
        direction, beta = self.transition(features, valid)
        lower = path_causal_scores(query, key, direction, beta)
        if causal:
            return lower
        reverse_direction, reverse_beta = self.transition(features.flip(1), valid.flip(1))
        upper = path_causal_scores(query.flip(-2), key.flip(-2), reverse_direction, reverse_beta).flip((-2, -1))
        return lower.tril() + upper.triu(1)


class PositionSelfAttention(nn.Module):
    def __init__(self, original: nn.MultiheadAttention, variant: str):
        super().__init__()
        self.embed_dim, self.num_heads = original.embed_dim, original.num_heads
        self.head_dim = self.embed_dim // self.num_heads
        self.in_proj_weight, self.in_proj_bias = original.in_proj_weight, original.in_proj_bias
        self.out_proj = original.out_proj
        self.dropout = original.dropout
        self.batch_first, self._qkv_same_embed_dim = True, True
        self.variant = variant
        builders = {
            "comrope_ap": lambda: ComRoPEAP(self.num_heads, self.head_dim),
            "videorope": lambda: VideoRoPE(self.head_dim),
            "path": lambda: PaTH(self.embed_dim, self.num_heads),
            "dape_v2_kerple": lambda: DAPEV2Kerple(self.num_heads),
            "identity": nn.Identity,
        }
        self.position_module = builders[variant]()

    def merge_masks(self, attn_mask, key_padding_mask, query):
        mask_type = None
        merged_mask = None
        if key_padding_mask is not None:
            mask_type = 1
            merged_mask = key_padding_mask
        if attn_mask is not None:
            batch_size, sequence_length, _ = query.shape
            mask_type = 2
            if attn_mask.dim() == 3:
                expanded = attn_mask.view(batch_size, -1, sequence_length, sequence_length)
            else:
                expanded = attn_mask.view(1, 1, sequence_length, sequence_length).expand(
                    batch_size, self.num_heads, -1, -1
                )
            merged_mask = expanded
            if key_padding_mask is not None:
                padding = key_padding_mask.view(batch_size, 1, 1, sequence_length).expand(
                    -1, self.num_heads, -1, -1
                )
                merged_mask = expanded + padding
        return merged_mask, mask_type

    def forward(self, query, key, value, key_padding_mask=None, need_weights=True,
                attn_mask=None, average_attn_weights=True, is_causal=False):
        if query is not key or query is not value:
            raise ValueError("本适配仅用于切片自注意力，不替换跨模态注意力")
        if attn_mask is not None:
            raise ValueError("替换ACPE后不应再传入其相对注意力偏置")
        valid = torch.ones(query.shape[:2], dtype=torch.bool, device=query.device)
        if key_padding_mask is not None:
            invalid = key_padding_mask if key_padding_mask.dtype == torch.bool else key_padding_mask < 0
            valid = ~invalid
        if not valid.any(dim=-1).all():
            raise ValueError("不允许全空切片输入")
        features = query * valid[..., None]
        projected = functional.linear(features, self.in_proj_weight, self.in_proj_bias)
        queries, keys, values = [part.reshape(*part.shape[:2], self.num_heads, self.head_dim).transpose(1, 2)
                                 for part in projected.chunk(3, dim=-1)]
        if self.variant == "comrope_ap":
            slots = (valid.cumsum(-1) - 1).clamp_min(0).to(query.dtype)
            positions = slots / (valid.sum(-1, keepdim=True) - 1).clamp_min(1)
            queries, keys = self.position_module(queries, keys, positions)
        elif self.variant == "videorope":
            slots = (valid.cumsum(-1) - 1).clamp_min(0).to(query.dtype)
            coordinates = (slots * self.position_module.temporal_spacing)[:, None].expand(-1, 3, -1)
            queries, keys = self.position_module(queries, keys, coordinates)
        if self.variant == "path":
            scores = self.position_module(queries, keys, features, valid, causal=is_causal) / math.sqrt(self.head_dim)
        else:
            scores = (queries @ keys.transpose(-1, -2)) / math.sqrt(self.head_dim)
        if self.variant == "dape_v2_kerple":
            scores = self.position_module(scores, valid, causal=is_causal)
        scores = scores.masked_fill(~valid[:, None, None, :], float("-inf"))
        if is_causal:
            invalid = torch.ones(scores.shape[-2:], dtype=torch.bool, device=scores.device).triu(1)
            scores = scores.masked_fill(invalid, float("-inf"))
        probabilities = scores.softmax(-1)
        probabilities = functional.dropout(probabilities, self.dropout, self.training)
        context = (probabilities @ values).transpose(1, 2).reshape_as(query)
        output = self.out_proj(context) * valid[..., None]
        weights = probabilities.mean(1) if average_attn_weights else probabilities
        return output, weights if need_weights else None


def replace_slice_attention(model, variant):
    if model.position_variant != "no_pe" or model.apro_positioner is not None:
        raise ValueError("必须先禁用ACPE，防止叠加两种位置模块")
    for layer in model.context_encoder.layers:
        layer.self_attn = PositionSelfAttention(layer.self_attn, variant)
    model.position_baseline = variant
    return model
