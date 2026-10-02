#!/usr/bin/env python3
"""CT-RATE 既有六种位置模型的统一完整序列 Block3 删片推理。"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys

import numpy as np
import torch
from sklearn.metrics import f1_score

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "src/scripts"))

from evaluate_figure1_grid import batch_for, fold_cases, save_json, tokenize
from exp_8.models import Exp12AProCoPEWatchCrossAttentionTextCNNModel
from exp_8.position_baselines import replace_slice_attention
from paper_block_deletion import block_sampling
from run_physionet_ct_ich_table2_baselines import CachedFeatureBackbone

VARIANTS = ("original_pe", "comrope_ap", "videorope", "path", "dape_v2_kerple", "acpe")
RATIOS = (0, 25, 50, 75)
OUT = ROOT / "outputs/paper_results/table2_ct_block3"


def load_model(variant: str, fold: int, device: torch.device):
    root = ROOT / "outputs/ct_rate_680/position_baselines" / ("apro_full" if variant == "acpe" else variant)
    folder = root / f"fold_{fold}" / "amef_multimodal"
    config = json.loads((folder / "config.json").read_text())
    checkpoint_path = folder / "best_model.pt"
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    params = dict(config["model"])
    assert params.pop("ct_attention_variant") == ("apro_full" if variant == "acpe" else variant)
    for key in list(params):
        if key.endswith("_weight"):
            params.pop(key)
    model = Exp12AProCoPEWatchCrossAttentionTextCNNModel(**params, pretrained=False, num_labels=3)
    model.instance_encoder.backbone = CachedFeatureBackbone(768, params["feature_dim"], params["dropout"])
    if variant not in ("acpe", "original_pe"):
        replace_slice_attention(model, variant)
    model.load_state_dict(checkpoint["state_dict"], strict=True)
    model.to(device).eval()
    thresholds = checkpoint["thresholds"]
    if isinstance(thresholds, dict):
        thresholds = list(thresholds.values())
    return model, np.asarray(thresholds, dtype=np.float32),


def load_records(cases: list[int]) -> dict[int, dict]:
    records = {}
    for case in cases:
        path = ROOT / "outputs/paper_results/figure1_features/ct_rate" / f"{case:04d}.npz"
        with np.load(path, allow_pickle=False) as cache:
            indices = cache["source_indices"].astype(np.int64)
            features = cache["features"].astype(np.float32)
            total = int(cache["original_count"])
        extra = ROOT / "outputs/paper_results/table2_ct_extra_features" / f"{case:04d}.npz"
        if extra.exists():
            with np.load(extra, allow_pickle=False) as cache:
                indices = np.concatenate((indices, cache["source_indices"].astype(np.int64)))
                features = np.concatenate((features, cache["features"].astype(np.float32)))
            order = np.argsort(indices)
            indices, features = indices[order], features[order]
        if not np.array_equal(indices, np.unique(indices)):
            raise ValueError(f"重复源切片：{case}")
        records[case] = {"indices": indices, "features": features, "source_count": total}
    return records


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fold", type=int, choices=range(1,6), required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--output-root", type=Path, default=OUT)
    args = parser.parse_args()
    torch.set_num_threads(2)
    torch.backends.mha.set_fastpath_enabled(False)
    device = torch.device(args.device)
    rows = json.loads((ROOT / "outputs/ct_rate_680/experiment/samples.json").read_text())
    labels = np.asarray([row["labels"] for row in rows], dtype=np.int64)
    test, validation = fold_cases("ct_rate", args.fold)
    original_dir = ROOT / "outputs/ct_rate_680/experiment/features"
    eligible = []
    for case in test:
        with np.load(original_dir / f"{case:04d}.npz", allow_pickle=False) as cache:
            if int(cache["original_count"]) >= 256:
                eligible.append(case)
    if not eligible:
        raise RuntimeError(f"fold{args.fold} 无75%后仍可采样64层的病例")
    records = load_records(eligible)
    token_ids, token_mask = tokenize("ct_rate", rows, validation)
    output = args.output_root / f"fold_{args.fold}.json"
    if output.exists():
        payload = json.loads(output.read_text())
        if payload.get("eligible_indices") != eligible or payload.get("seed") != 42:
            raise RuntimeError(f"已有结果协议不符：{output}")
        if all(len(payload.get("results", {}).get(variant, {})) == 4 for variant in VARIANTS):
            return
        results, sources = payload["results"], payload["checkpoint_sources"]
    else:
        results, sources = {}, {}
    for variant in VARIANTS:
        if len(results.get(variant, {})) == 4:
            continue
        model, thresholds = load_model(variant, args.fold, device)
        folder = ROOT / "outputs/ct_rate_680/position_baselines" / ("apro_full" if variant == "acpe" else variant)
        checkpoint = folder / f"fold_{args.fold}" / "amef_multimodal/best_model.pt"
        sources[variant] = {"path": str(checkpoint.relative_to(ROOT)),
                            "sha256": hashlib.sha256(checkpoint.read_bytes()).hexdigest(),
                            "thresholds": thresholds.tolist()}
        results[variant] = {}
        for ratio in RATIOS:
            selected = {case: np.asarray(block_sampling(records[case]["source_count"], ratio,
                                  1 if ratio == 0 else 3, case)["selected_raw_indices"], dtype=np.int64)
                        for case in eligible}
            probabilities = []
            with torch.inference_mode():
                for start in range(0, len(eligible), args.batch_size):
                    cases = eligible[start:start+args.batch_size]
                    images, mask, positions, counts = batch_for(cases, selected, records, device)
                    output_values = model(images=images, mask=mask,
                        watch_token_ids=token_ids[cases].to(device),
                        watch_token_mask=token_mask[cases].to(device),
                        instance_indices=positions, original_image_counts=counts)
                    logits = output_values["logits"]
                    if not torch.isfinite(logits).all():
                        raise FloatingPointError(f"非有限分类输出：{variant}/fold{args.fold}/{ratio}")
                    probabilities.append(logits.sigmoid().float().cpu().numpy())
            prob = np.concatenate(probabilities)
            score = float(f1_score(labels[eligible], prob >= thresholds[None], average="macro", zero_division=0))
            results[variant][str(ratio)] = {"macro_f1": score, "num_cases": len(eligible)}
            print(f"CT-RATE fold{args.fold} {variant} {ratio}%: N={len(eligible)} F1={score:.4f}", flush=True)
        del model
        torch.cuda.empty_cache()
        save_json(output, {"dataset": "ct_rate", "fold": args.fold, "test_indices": test,
                           "eligible_indices": eligible, "excluded_indices": sorted(set(test)-set(eligible)),
                           "seed": 42, "B": 3, "checkpoint_sources": sources,
                           "protocol": "完整原序列固定B3删片后均匀采样64张；75%后不足64者在四比例共同排除；所有模型共用原始掩码，隐藏原始索引。",
                           "results": results})


if __name__ == "__main__":
    main()
