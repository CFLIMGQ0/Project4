#!/usr/bin/env python3
"""在三个公开数据集上测试 ACPE 的连续删片训练增强。"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys

import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "src/scripts"))

import run_acpe_system_interaction as base
from amos_mm_model_adapters import PartialLabelAMEF
from exp_8.acpe_block_augmentation import BlockAugmentationMixin, BlockAugmentationModel
from run_physionet_ct_ich_table2_baselines import CachedFeatureBackbone

VARIANT = "block_aug"
ORIGINAL_PROTOCOL = base.protocol


class PartialLabelBlockAugmentation(BlockAugmentationMixin, PartialLabelAMEF):
    """AMOS-MM 部分标签监督对应的连续删片候选。"""


def block_parameters(_variant: str) -> dict:
    import yaml
    from scripts.task3_apro_cope_ablation_scheduler import base_model_params

    params = base_model_params("apro_full")
    main = yaml.safe_load((ROOT / "src/configs/task3/t3_main_model.yaml").read_text())
    params["label_query_consistency_weight"] = float(
        main["model"]["params"]["label_query_consistency_weight"]
    )
    params["image_aux_weight"] = 0.0
    params["block_aug_weight"] = 0.5
    return params


def block_build(dataset: str, _variant: str, count: int):
    params = dict(block_parameters(VARIANT))
    auxiliary = {key.removesuffix("_weight"): params.pop(key)
                 for key in list(params) if key.endswith("_weight")}
    klass = PartialLabelBlockAugmentation if dataset == "amos_mm" else BlockAugmentationModel
    model = klass(**params, pretrained=False, num_labels=count)
    model.instance_encoder.backbone = CachedFeatureBackbone(
        input_dim=768, output_dim=params["feature_dim"], dropout=params["dropout"]
    )
    return model, auxiliary


def block_protocol(dataset: str, variant: str) -> dict:
    value = ORIGINAL_PROTOCOL(dataset, variant)
    for relative in ("src/exp_8/acpe_block_augmentation.py", "src/scripts/run_acpe_block_augmentation.py"):
        value["sources"][relative] = hashlib.sha256((ROOT / relative).read_bytes()).hexdigest()
    value["purpose"] = (
        "训练期额外构造单个连续缺失图像块，保留原始采集索引；"
        "只对删片视图的图像标签 logits 加 0.5 倍监督，推理结构和原 ACPE 不变"
    )
    value["block_deletion_fraction"] = 0.5
    value["block_aug_weight"] = 0.5
    value.pop("protocol_sha256")
    value["protocol_sha256"] = hashlib.sha256(
        json.dumps(value, sort_keys=True, ensure_ascii=False).encode()
    ).hexdigest()
    return value


def install() -> None:
    base.parameters = block_parameters
    base.build_model = block_build
    base.protocol = block_protocol


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", choices=base.DATASETS, required=True)
    parser.add_argument("--fold", type=int, choices=range(1, 6), default=1)
    parser.add_argument("--initialize-only", action="store_true")
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()
    install()
    digest = base.initialize(args.dataset, VARIANT)
    if args.initialize_only:
        print(digest)
        return
    if args.smoke:
        base.smoke(args.dataset, VARIANT)
        return
    torch.set_num_threads(4)
    torch.backends.mha.set_fastpath_enabled(False)
    {"ct_rate": base.run_ct, "amos_mm": base.run_amos, "mr_rate_1k": base.run_mr}[
        args.dataset
    ](VARIANT, args.fold, digest)


if __name__ == "__main__":
    main()
