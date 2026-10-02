"""在保持采集结构约束的前提下学习样本级 ACPE 形变尺度。"""

from __future__ import annotations

import math

import torch
import torch.nn as nn

from exp_8.models import Exp12AProCoPEWatchCrossAttentionTextCNNModel


class EtaScalePositioner(nn.Module):
    """用视觉全局状态和采集间隔不规则度调节原 ACPE 的形变。"""

    def __init__(self, base: nn.Module):
        super().__init__()
        self.base = base
        self.feature_weight = nn.Parameter(torch.zeros(base.feature_dim))
        self.gap_weight = nn.Parameter(torch.zeros(()))
        self.bias = nn.Parameter(torch.zeros(()))

    def forward(self, features, mask, instance_indices, original_image_counts):
        base = self.base
        raw = base.acquisition_coordinates(
            mask, instance_indices, original_image_counts, dtype=features.dtype
        )
        valid = mask.unsqueeze(-1).to(features.dtype)
        pooled = (features * valid).sum(dim=1) / valid.sum(dim=1).clamp_min(1.0)
        gap = (raw[:, 1:] - raw[:, :-1]).clamp_min(0)
        valid_gap = mask[:, 1:] & mask[:, :-1]
        gap = gap * valid_gap.to(gap.dtype)
        mean_gap = gap.sum(dim=1) / valid_gap.sum(dim=1).clamp_min(1)
        maximum_gap = gap.amax(dim=1) if gap.shape[1] else torch.zeros_like(mean_gap)
        irregularity = maximum_gap / mean_gap.clamp_min(1e-6)
        score = (
            (pooled * self.feature_weight).sum(dim=-1) / math.sqrt(base.feature_dim)
            + self.gap_weight * torch.log1p(irregularity)
            + self.bias
        )
        scale = 1.0 + torch.tanh(score)
        original_alpha = base.warp_alpha
        # 原模块只在当前前向中读取 warp_alpha；调用后立即恢复常量。
        base.warp_alpha = original_alpha * scale.unsqueeze(-1)
        try:
            absolute, attention_bias, diagnostics = base(
                features, mask, instance_indices, original_image_counts
            )
        finally:
            base.warp_alpha = original_alpha
        diagnostics["acpe_eta_scale"] = scale
        diagnostics["acpe_gap_irregularity"] = irregularity
        return absolute, attention_bias, diagnostics


class EtaScaleMixin:
    """可与完整标签监督和 AMOS 部分标签监督模型组合。"""

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        if self.apro_positioner is None:
            raise ValueError("样本级形变尺度仅适用于完整 ACPE")
        self.apro_positioner = EtaScalePositioner(self.apro_positioner)


class EtaScaleModel(EtaScaleMixin, Exp12AProCoPEWatchCrossAttentionTextCNNModel):
    """完整标签监督数据集的样本级形变尺度。"""
