"""只改变位置来源或变形的独立对照，不改动历史模型实现。"""
from __future__ import annotations

import torch

from exp_8.models import AProCoPE


class AcquisitionIndexOnly(AProCoPE):
    """保留两条位置注入路径，严格设置 c=r、eta=0。"""

    def _contextual_coordinates(self, features, mask, raw_coordinates):
        # 消耗与完整 ACPE 相同的 dropout 随机数，便于同种子配对训练。
        # 此结果不进入预测，因此变形网络不会获得梯度。
        with torch.no_grad():
            super()._contextual_coordinates(features, mask, raw_coordinates)
        return raw_coordinates, torch.zeros_like(raw_coordinates)


def install_control(model, variant):
    """所有条件先构建同一个完整模型，确保共有参数的初始化完全一致。"""
    assert model.position_variant == "apro_full"
    positioner = model.apro_positioner
    assert isinstance(positioner, AProCoPE)
    assert tuple(positioner.transition_groups) == (1, 2, 3, 4, 5, 6)
    assert positioner.route == "both" and positioner.transition_include_gap
    if variant == "no_pe":
        model.position_variant = "no_pe"
    elif variant == "uniform_pe":
        # 与历史 Original PE 相同：对当前输入槽位进行标量 MLP 编码。
        model.position_variant = "original_pe"
    elif variant == "acquisition_index_pe":
        positioner.__class__ = AcquisitionIndexOnly
        for module in (positioner.transition_projector, positioner.transition_mlp):
            module.requires_grad_(False)
    elif variant != "acpe":
        raise ValueError(variant)
    model.index_control_variant = variant
    return model
