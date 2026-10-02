"""按检查样本调节 ACPE 坐标调整强度的受控研究变体。"""

from __future__ import annotations

import math

import torch
import torch.nn as nn

from exp_8.models import Exp12AProCoPEWatchCrossAttentionTextCNNModel


class ConfidenceGatePositioner(nn.Module):
    """保留原 ACPE 的全部权重与位置路径，仅加入一个样本级可信度门。"""

    def __init__(self, base: nn.Module, initial_logit: float = 4.0):
        super().__init__()
        self.base = base
        self.gate_weight = nn.Parameter(torch.zeros(base.feature_dim))
        self.gate_bias = nn.Parameter(torch.tensor(float(initial_logit)))

    def forward(self, features, mask, instance_indices, original_image_counts):
        base = self.base
        raw = base.acquisition_coordinates(
            mask, instance_indices, original_image_counts, dtype=features.dtype
        )
        contextual, eta = base._contextual_coordinates(features, mask, raw)
        valid = mask.unsqueeze(-1).to(features.dtype)
        pooled = (features * valid).sum(dim=1) / valid.sum(dim=1).clamp_min(1.0)
        gate = torch.sigmoid(
            (pooled * self.gate_weight).sum(dim=-1) / math.sqrt(base.feature_dim) + self.gate_bias
        )
        contextual = raw + gate.unsqueeze(-1) * (contextual - raw)
        contextual = contextual * mask.to(contextual.dtype)

        absolute = None
        if base.route in {"absolute", "both"}:
            absolute_input = torch.cat(
                [base.fourier(raw), base.fourier(contextual), base.fourier(contextual - raw)], dim=-1
            )
            absolute = base.absolute_projector(absolute_input)
            absolute = absolute * valid

        attention_bias = None
        if base.route in {"relative", "both"}:
            delta_raw = raw.unsqueeze(2) - raw.unsqueeze(1)
            delta_context = contextual.unsqueeze(2) - contextual.unsqueeze(1)
            relative_input = torch.cat(
                [
                    base.fourier(delta_raw),
                    base.fourier(delta_context),
                    base.fourier(delta_context - delta_raw),
                ],
                dim=-1,
            )
            attention_bias = 2.0 * torch.tanh(base.relative_bias(relative_input))
            attention_bias = attention_bias.permute(0, 3, 1, 2).contiguous()
            valid_pairs = mask[:, None, :, None] & mask[:, None, None, :]
            attention_bias = attention_bias * valid_pairs.to(attention_bias.dtype)

        diagnostics = {
            "apro_raw_coordinates": raw,
            "apro_context_coordinates": contextual,
            "apro_transition_eta": eta,
            "acpe_confidence_gate": gate,
        }
        return absolute, attention_bias, diagnostics


class ConfidenceGateMixin:
    """可与完整标签监督和 AMOS 部分标签监督模型组合。"""

    def __init__(self, *, confidence_initial_logit: float = 4.0, **kwargs):
        super().__init__(**kwargs)
        if self.apro_positioner is None:
            raise ValueError("可信度门仅适用于完整 ACPE 模型")
        self.apro_positioner = ConfidenceGatePositioner(
            self.apro_positioner, initial_logit=confidence_initial_logit
        )


class ConfidenceGateModel(ConfidenceGateMixin, Exp12AProCoPEWatchCrossAttentionTextCNNModel):
    """完整标签监督数据集的样本级坐标门。"""
