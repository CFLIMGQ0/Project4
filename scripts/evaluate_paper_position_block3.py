#!/usr/bin/env python3
"""沿用既有完整序列 Block3 掩码，对缺失的位置方法 checkpoint 补做推理。"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "src/scripts"))

import evaluate_public_block3 as block3
import evaluate_amos_mrrate_deletion_classification as base
from exp_8.position_baselines import replace_slice_attention
from run_physionet_ct_ich_table2_baselines import CachedFeatureBackbone

VARIANTS = ("comrope_ap", "videorope", "path", "dape_v2_kerple")
OUTPUT = ROOT / "outputs/paper_results/table2_block3"


def checkpoint_root(dataset: str, variant: str) -> Path:
    if dataset == "amos_mm":
        return ROOT / "outputs/amos_mm/position_replacements_7_labels" / variant
    return ROOT / "outputs/mr_rate_1k/position_replacements" / variant


def load_position_model(dataset: str, variant: str, fold: int, device: torch.device,
                        checkpoint_root: Path | None = None):
    if variant not in VARIANTS:
        return ORIGINAL_LOAD(dataset, variant, fold, device, checkpoint_root)
    root = checkpoint_root or globals()["checkpoint_root"](dataset, variant)
    folder = root / ("7_labels" if dataset == "amos_mm" else "") / f"fold_{fold}" / "amef_multimodal"
    config = json.loads((folder / "config.json").read_text())
    checkpoint_path = folder / "best_model.pt"
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    params = dict(config.get("model", config.get("parameters", {})))
    assert params.pop("ct_attention_variant", None) == variant
    for key in list(params):
        if key.endswith("_weight"):
            params.pop(key)
    from exp_8.models import Exp12AProCoPEWatchCrossAttentionTextCNNModel
    model = Exp12AProCoPEWatchCrossAttentionTextCNNModel(
        **params, pretrained=False, num_labels=7 if dataset == "amos_mm" else 4)
    model.instance_encoder.backbone = CachedFeatureBackbone(768, params["feature_dim"], params["dropout"])
    replace_slice_attention(model, variant)
    model.load_state_dict(checkpoint["state_dict"], strict=True)
    model.to(device).eval()
    thresholds = checkpoint.get("thresholds")
    if isinstance(thresholds, dict):
        thresholds = list(thresholds.values())
    return model, np.asarray(thresholds, dtype=np.float32), str(checkpoint_path.relative_to(ROOT)), config


ORIGINAL_LOAD = base.load_model


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", choices=("amos_mm", "mr_rate"), required=True)
    parser.add_argument("--variant", choices=VARIANTS, required=True)
    parser.add_argument("--fold", type=int, choices=range(1, 6), required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--batch-size", type=int, default=16)
    args = parser.parse_args()
    output = OUTPUT / args.dataset / args.variant / f"fold_{args.fold}.json"
    if output.exists():
        payload = json.loads(output.read_text())
        if len(payload.get("metrics", [])) == 4 and payload.get("deletion_mode") == "block3":
            return
        raise RuntimeError(f"现有结果不完整，禁止覆盖：{output}")
    base.load_model = load_position_model
    base.VARIANTS[args.variant] = args.variant
    block3.run_classification(argparse.Namespace(
        dataset=args.dataset, variant=args.variant, fold=args.fold,
        checkpoint_root=checkpoint_root(args.dataset, args.variant),
        output=output, device=args.device, batch_size=args.batch_size, limit_cases=None))


if __name__ == "__main__":
    main()
