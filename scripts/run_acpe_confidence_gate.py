#!/usr/bin/env python3
"""在三个公开数据集上验证样本级 ACPE 坐标可信度门。"""

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
from exp_8.acpe_confidence_gate import ConfidenceGateMixin, ConfidenceGateModel
from run_physionet_ct_ich_table2_baselines import CachedFeatureBackbone

VARIANT = "confidence_gate"
ORIGINAL_PROTOCOL = base.protocol


class PartialLabelConfidenceGate(ConfidenceGateMixin, PartialLabelAMEF):
    """AMOS-MM 部分标签监督对应的可信度门变体。"""


def confidence_parameters(_variant: str) -> dict:
    params = base_model_parameters()
    return params


def base_model_parameters() -> dict:
    import yaml
    from scripts.task3_apro_cope_ablation_scheduler import base_model_params

    params = base_model_params("apro_full")
    main = yaml.safe_load((ROOT / "src/configs/task3/t3_main_model.yaml").read_text())
    params["label_query_consistency_weight"] = float(
        main["model"]["params"]["label_query_consistency_weight"]
    )
    params["image_aux_weight"] = 0.0
    return params


def confidence_build(dataset: str, _variant: str, count: int):
    params = dict(base_model_parameters())
    if _variant == "confidence_half":
        params["confidence_initial_logit"] = 0.0
    auxiliary = {key.removesuffix("_weight"): params.pop(key)
                 for key in list(params) if key.endswith("_weight")}
    klass = PartialLabelConfidenceGate if dataset == "amos_mm" else ConfidenceGateModel
    model = klass(**params, pretrained=False, num_labels=count)
    model.instance_encoder.backbone = CachedFeatureBackbone(
        input_dim=768, output_dim=params["feature_dim"], dropout=params["dropout"]
    )
    return model, auxiliary


def confidence_protocol(dataset: str, variant: str) -> dict:
    value = ORIGINAL_PROTOCOL(dataset, variant)
    for relative in ("src/exp_8/acpe_confidence_gate.py", "src/scripts/run_acpe_confidence_gate.py"):
        value["sources"][relative] = hashlib.sha256((ROOT / relative).read_bytes()).hexdigest()
    value["purpose"] = (
        "仅对 ACPE 上下文调整量施加由检查视觉特征决定的样本级可信度门；"
        "保留采集锚点、端点、单调性、绝对和相对两条位置路径"
    )
    value["initial_gate"] = float(torch.sigmoid(torch.tensor(0.0 if variant == "confidence_half" else 4.0)))
    value.pop("protocol_sha256")
    value["protocol_sha256"] = hashlib.sha256(
        json.dumps(value, sort_keys=True, ensure_ascii=False).encode()
    ).hexdigest()
    return value


def install() -> None:
    base.parameters = confidence_parameters
    base.build_model = confidence_build
    base.protocol = confidence_protocol


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", choices=base.DATASETS, required=True)
    parser.add_argument("--variant", choices=("confidence_gate", "confidence_half"), default="confidence_gate")
    parser.add_argument("--fold", type=int, choices=range(1, 6), default=1)
    parser.add_argument("--initialize-only", action="store_true")
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()
    install()
    variant = args.variant
    digest = base.initialize(args.dataset, variant)
    if args.initialize_only:
        print(digest)
        return
    if args.smoke:
        base.smoke(args.dataset, variant)
        return
    torch.set_num_threads(4)
    torch.backends.mha.set_fastpath_enabled(False)
    {"ct_rate": base.run_ct, "amos_mm": base.run_amos, "mr_rate_1k": base.run_mr}[
        args.dataset
    ](variant, args.fold, digest)


if __name__ == "__main__":
    main()
