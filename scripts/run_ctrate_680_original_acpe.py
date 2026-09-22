#!/usr/bin/env python3
"""CT-RATE原始ACPE五折重跑入口，协议路径可跨主机复现。"""
from __future__ import annotations

import argparse
import fcntl
import hashlib
import json
from pathlib import Path
import sys

import yaml

import run_ctrate_680_apro_transition_dim85 as base


ROOT = base.ROOT
SCRIPTS = base.SCRIPTS
MODEL_KEY = "amef_multimodal"
POSITION_VARIANT = "apro_full"
MODELS = {MODEL_KEY: "AMEF-MIL (原始ACPE)"}
LABELS = base.LABELS
INPUT = base.INPUT
SETTINGS = base.SETTINGS
ORIGINAL_PROTOCOL = base.ORIGINAL_PROTOCOL


def parameters():
    from scripts.task3_apro_cope_ablation_scheduler import base_model_params

    main = yaml.safe_load((ROOT / "src/configs/task3/t3_main_model.yaml").read_text())
    model = base_model_params(POSITION_VARIANT)
    model["label_query_consistency_weight"] = main["model"]["params"]["label_query_consistency_weight"]
    model["image_aux_weight"] = 0.0
    return {MODEL_KEY: model}


def protocol():
    value = ORIGINAL_PROTOCOL()
    root_prefix = str(ROOT.resolve()) + "/"

    def stable_paths(item):
        if isinstance(item, dict):
            return {
                (key.removeprefix(root_prefix) if isinstance(key, str) and key.startswith(root_prefix) else key):
                stable_paths(child)
                for key, child in item.items()
            }
        if isinstance(item, list):
            return [stable_paths(child) for child in item]
        if isinstance(item, str) and item.startswith(root_prefix):
            return item.removeprefix(root_prefix)
        return item

    value = stable_paths(value)
    value["models"] = MODELS
    value["model_parameters"] = parameters()
    value["position_ablation"] = (
        "position_variant=apro_full; original ACPE: position_dim=64, "
        "two-layer Fourier projections, transition_dim defaults to position_dim"
    )
    value["runner"] = "src/scripts/run_ctrate_680_original_acpe.py"
    value["source_sha256"][value["runner"]] = hashlib.sha256(
        Path(__file__).read_bytes()
    ).hexdigest()
    value.pop("protocol_sha256", None)
    value["protocol_sha256"] = hashlib.sha256(
        json.dumps(value, sort_keys=True, ensure_ascii=False).encode()
    ).hexdigest()
    return value


def configure():
    base.core.POSITION_VARIANT = POSITION_VARIANT
    base.core.INPUT = INPUT
    base.core.LABELS = LABELS
    base.core.MODELS = MODELS
    base.core.SETTINGS = SETTINGS
    base.core.parameters = parameters
    base.core.protocol = protocol
    base.core.build_model = base.build_model
    base.core.configure()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path,
                        default=ROOT / "outputs/ct_rate_680/amef_original_acpe_rerun_fivefold_v2")
    parser.add_argument("--devices", type=int, nargs="+", default=[0, 1, 2, 3])
    parser.add_argument("--models", nargs="+", choices=[MODEL_KEY], default=None)
    parser.add_argument("--worker-index", type=int)
    parser.add_argument("--audit-only", action="store_true")
    parser.add_argument("--aggregate", action="store_true")
    args = parser.parse_args()
    configure()
    if args.worker_index is not None:
        base.core.core.worker(args)
        return
    args.output_dir.mkdir(parents=True, exist_ok=True)
    with (args.output_dir / "run.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        current = protocol()
        saved = args.output_dir / "protocol.json"
        if saved.exists() and json.loads(saved.read_text())["protocol_sha256"] != current["protocol_sha256"]:
            raise ValueError("原始ACPE协议或源码改变，禁止混用已有结果")
        base.core.core.save_json(saved, current)
        if args.audit_only:
            print(json.dumps({"protocol_sha256": current["protocol_sha256"], "fold_jobs": 5,
                              "position_variant": POSITION_VARIANT, "apro_position_dim": 64,
                              "apro_transition_dim": 64}, ensure_ascii=False))
            return
        if args.aggregate:
            print(f"已汇总{base.core.core.aggregate(args, current['protocol_sha256'])}/5个折次。", flush=True)
            return
        raise RuntimeError("原始ACPE动态多卡实验请使用run_ctrate_original_acpe_pool.py")


if __name__ == "__main__":
    main()
