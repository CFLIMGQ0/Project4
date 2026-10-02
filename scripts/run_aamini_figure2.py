#!/usr/bin/env python3
"""Evaluate the Figure-2 structured-missingness grid on AA-Mini.

AA-Mini's frozen input cache contains at most 64 uniformly sampled slices per
examination.  This runner applies the existing 65-condition deletion protocol
to that cached input sequence and retains the original acquisition indices for
the coordinate metric.  It does not silently claim access to uncached slices.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import statistics
import sys
import time
from pathlib import Path

import numpy as np
import torch
from tqdm import tqdm

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src/scripts"))
sys.path.insert(0, str(ROOT / "src"))
from paper_block_deletion import RATIOS, BLOCKS, block_sampling

OUT = ROOT / "outputs/aamini_figure2_20261002"
AA_FEATURES = ROOT / "outputs/abdomenatlas3_mini/experiment/features"
AA_INPUT = ROOT / "outputs/abdomenatlas3_mini/experiment_anonymous_findings"
AA_FULL = ROOT / "outputs/abdomenatlas3_mini/table1_fivefold_original_uit_anonymous_findings"
AA_ORIGINAL = ROOT / "outputs/abdomenatlas3_mini/table2_8_fivefold_anonymous_findings/position_original"
GRID = [(r, b) for r in RATIOS for b in ((1,) if r == 0 else BLOCKS)]


def read(path: Path):
    return json.loads(path.read_text())


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def build_model(kind: str, fold: int, device: torch.device):
    """Construct the exact model family used by the saved AA-Mini checkpoint."""
    if kind == "acpe":
        import run_abdomenatlas3_table1 as runner

        params = runner.parameters()["amef_multimodal"]
        checkpoint = AA_FULL / f"fold_{fold}" / "amef_multimodal" / "best_model.pt"
    else:
        import run_abdomenatlas3_analysis as runner

        params = runner.params_for("position_original")
        checkpoint = AA_ORIGINAL / f"fold_{fold}" / "amef_multimodal" / "best_model.pt"
    # Build directly instead of using the mutable adapter globals shared by
    # the table runner and the ablation runner.
    from exp_8.models import Exp12AProCoPEWatchCrossAttentionTextCNNModel
    from run_physionet_ct_ich_table2_baselines import CachedFeatureBackbone

    values = dict(params)
    for name in list(values):
        if name.endswith("_weight"):
            values.pop(name)
    model = Exp12AProCoPEWatchCrossAttentionTextCNNModel(
        **values, pretrained=False, num_labels=3
    )
    model.instance_encoder.backbone = CachedFeatureBackbone(
        768, values["feature_dim"], values["dropout"]
    )
    saved = torch.load(checkpoint, map_location="cpu", weights_only=False)
    model.load_state_dict(saved["state_dict"], strict=True)
    model.to(device).eval()
    return model, np.asarray(saved["thresholds"], dtype=np.float32), checkpoint


def load_fold(fold: int):
    rows = read(AA_INPUT / "samples.json")
    folds = read(AA_INPUT / "patient_folds.json")["folds"]
    cases = list(map(int, folds[fold - 1]))
    records = {}
    excluded = []
    for case in tqdm(cases, desc=f"AA-Mini 折{fold} 核验删除网格", mininterval=5):
        with np.load(AA_FEATURES / f"{case:04d}.npz", allow_pickle=False) as z:
            features = z["features"].astype(np.float32)
            positions = z["slice_indices"].astype(np.int64)
            original_count = int(z["original_count"])
        assert features.shape == (len(positions), 768)
        assert np.isfinite(features).all() and np.all(np.diff(positions) > 0)
        try:
            # Draw the deletion mask in the original acquisition coordinate
            # system, then retain cached uniform-64 features outside it.
            selections = {}
            for ratio, blocks in GRID:
                deletion = block_sampling(original_count, ratio, blocks, case)
                deleted = np.asarray(deletion["deleted_raw_indices"], dtype=np.int64)
                selected = positions[~np.isin(positions, deleted)]
                if len(selected) < 3:
                    raise ValueError(
                        f"缓存序列删除后不足3张：N={original_count}, ratio={ratio}, B={blocks}"
                    )
                selections[f"{ratio:02d}_B{blocks}"] = {
                    "selected_raw_indices": selected.tolist()
                }
        except ValueError as error:
            excluded.append({"case_index": case, "reason": str(error)})
            continue
        records[case] = {
            "features": features,
            "positions": positions,
            "original_count": original_count,
            "selections": selections,
            "labels": np.asarray(rows[case]["labels"], dtype=np.float32),
        }
    return rows, records, excluded


def encode_text(rows, cases):
    from training.data import _encode_text_fields

    values = [
        _encode_text_fields({"watch": rows[case]["findings_masked"]}, ("watch",),
                            max_length=512, vocab_size=8192)
        for case in cases
    ]
    return torch.stack([value[0] for value in values]), torch.stack([value[1] for value in values])


def batch_for(records, cases, key):
    selected = [records[case]["selections"][key]["selected_raw_indices"] for case in cases]
    n = max(len(value) for value in selected)
    images = torch.zeros(len(cases), n, 768, 1, 1, dtype=torch.float32)
    mask = torch.zeros(len(cases), n, dtype=torch.bool)
    positions = torch.full((len(cases), n), -1, dtype=torch.long)
    counts = torch.zeros(len(cases), dtype=torch.long)
    for index, case in enumerate(cases):
        record = records[case]
        raw_slots = np.asarray(selected[index], dtype=np.int64)
        slots = np.searchsorted(record["positions"], raw_slots)
        assert np.all(slots < len(record["positions"]))
        assert np.array_equal(record["positions"][slots], raw_slots)
        length = len(slots)
        images[index, :length, :, 0, 0] = torch.from_numpy(record["features"][slots])
        mask[index, :length] = True
        positions[index, :length] = torch.from_numpy(raw_slots)
        counts[index] = record["original_count"]
    return images, mask, positions, counts


def f1_score(y: np.ndarray, p: np.ndarray, thresholds: np.ndarray) -> float:
    pred = p >= thresholds.reshape(1, -1)
    truth = y > 0.5
    values = []
    for label in range(y.shape[1]):
        tp = np.sum(pred[:, label] & truth[:, label])
        fp = np.sum(pred[:, label] & ~truth[:, label])
        fn = np.sum(~pred[:, label] & truth[:, label])
        values.append(2 * tp / max(2 * tp + fp + fn, 1))
    return float(np.mean(values))


@torch.inference_mode()
def evaluate_fold(fold: int, device: torch.device, batch_size: int):
    rows, records, excluded = load_fold(fold)
    cases = sorted(records)
    token_ids, token_mask = encode_text(rows, cases)
    acpe, acpe_thresholds, acpe_ckpt = build_model("acpe", fold, device)
    original, original_thresholds, original_ckpt = build_model("original", fold, device)
    results = {}
    for ratio, blocks in tqdm(GRID, desc=f"AA-Mini 折{fold} 65条件评估", mininterval=5):
        key = f"{ratio:02d}_B{blocks}"
        probs = {"acpe": [], "original": []}
        labels = []
        acc = {"acpe": [], "original": []}
        for start in range(0, len(cases), batch_size):
            subset = cases[start:start + batch_size]
            images, mask, positions, counts = batch_for(records, subset, key)
            ids = torch.tensor([cases.index(case) for case in subset], dtype=torch.long)
            kwargs = {
                "images": images.to(device), "mask": mask.to(device),
                "watch_token_ids": token_ids[ids].to(device),
                "watch_token_mask": token_mask[ids].to(device),
                "instance_indices": positions.to(device),
                "original_image_counts": counts.to(device),
            }
            for name, model in (("acpe", acpe), ("original", original)):
                with torch.autocast(device_type=device.type, dtype=torch.float16):
                    output = model(**kwargs)
                logits = output["logits"].float()
                assert torch.isfinite(logits).all()
                probs[name].append(logits.sigmoid().cpu().numpy())
                if name == "acpe":
                    context = output["apro_context_coordinates"].float().cpu().numpy()
                    for j, case in enumerate(subset):
                        length = int(mask[j].sum())
                        truth = positions[j, :length].numpy().astype(np.float64) / max(1, int(counts[j]) - 1)
                        value = context[j, :length]
                        assert np.isfinite(value).all() and np.all(np.diff(value) >= -1e-6)
                        pred = (value - value[0]) / max(float(value[-1] - value[0]), 1e-12)
                        acc[name].append(float(np.mean(np.abs(pred[1:-1] - truth[1:-1]) <= .05)))
                else:
                    for j, case in enumerate(subset):
                        length = int(mask[j].sum())
                        truth = positions[j, :length].numpy().astype(np.float64) / max(1, int(counts[j]) - 1)
                        pred = np.linspace(0., 1., length)
                        acc[name].append(float(np.mean(np.abs(pred[1:-1] - truth[1:-1]) <= .05)))
            labels.append(np.stack([records[case]["labels"] for case in subset]))
        y = np.concatenate(labels)
        metrics = {}
        for name, threshold in (("acpe", acpe_thresholds), ("original", original_thresholds)):
            probability = np.concatenate(probs[name])
            metrics[name] = {"macro_f1": f1_score(y, probability, threshold),
                             "acc005": float(np.mean(acc[name]))}
        results[key] = {
            "ratio": ratio, "blocks": blocks, "num_cases": len(cases),
            **metrics,
            "delta_f1": metrics["acpe"]["macro_f1"] - metrics["original"]["macro_f1"],
            "delta_acc005": metrics["acpe"]["acc005"] - metrics["original"]["acc005"],
        }
    out = OUT / "folds" / f"fold_{fold}"
    out.mkdir(parents=True, exist_ok=True)
    (out / "grid.json").write_text(json.dumps({
        "dataset": "AA-Mini", "fold": fold, "evaluation_split": "test",
        "input_sequence": "cached_uniform64", "num_cases": len(cases),
        "case_indices": cases, "excluded": excluded,
        "acpe_checkpoint": str(acpe_ckpt.relative_to(ROOT)),
        "original_checkpoint": str(original_ckpt.relative_to(ROOT)),
        "results": results,
    }, ensure_ascii=False, indent=2) + "\n")
    del acpe, original
    torch.cuda.empty_cache() if device.type == "cuda" else None
    return results


def aggregate():
    objects = []
    for fold in range(1, 6):
        path = OUT / "folds" / f"fold_{fold}" / "grid.json"
        if path.exists():
            objects.append(read(path))
    rows = []
    for ratio, blocks in GRID:
        key = f"{ratio:02d}_B{blocks}"
        values = [obj["results"][key] for obj in objects if key in obj["results"]]
        if len(values) != len(objects) or not values:
            continue
        row = {"dataset": "AA-Mini", "ratio": ratio, "B": blocks,
               "cases": min(obj["num_cases"] for obj in objects),
               "folds": len(values)}
        for field in ("acpe", "original", "delta_f1", "delta_acc005"):
            if field in ("acpe", "original"):
                for metric in ("macro_f1", "acc005"):
                    numbers = [item[field][metric] for item in values]
                    row[f"{field}_{metric}_mean"] = statistics.mean(numbers)
                    row[f"{field}_{metric}_std"] = statistics.stdev(numbers) if len(numbers) > 1 else 0.0
            else:
                numbers = [item[field] for item in values]
                row[f"{field}_mean"] = statistics.mean(numbers)
                row[f"{field}_std"] = statistics.stdev(numbers) if len(numbers) > 1 else 0.0
        rows.append(row)
    OUT.mkdir(parents=True, exist_ok=True)
    with (OUT / "grid_mean_std.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]) if rows else ["dataset"])
        writer.writeheader(); writer.writerows(rows)
    (OUT / "summary.json").write_text(json.dumps({"completed_folds": len(objects), "conditions": len(rows)}, indent=2) + "\n")
    (OUT / "provenance.json").write_text(json.dumps({
        "dataset": "AA-Mini",
        "completed_folds": len(objects),
        "conditions_per_fold": 65,
        "models": "full ACPE versus position_original, both loaded from frozen five-fold checkpoints",
        "deletion_protocol": "block_sampling on the original acquisition index range, ratios 0--80% and B=1--8",
        "input_scope": "deletion masks are applied to the cached uniformly sampled 64-image sequence; retained samples keep original acquisition indices",
        "coordinate_metric": "normalized contextual/original acquisition coordinates, Acc@0.05",
        "caveat": "uncached original slices were not re-encoded on this host; this result is a cached-64 robustness evaluation",
    }, ensure_ascii=False, indent=2) + "\n")
    return rows


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--folds", nargs="+", type=int, choices=range(1, 6))
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--aggregate", action="store_true")
    args = parser.parse_args()
    if args.aggregate:
        aggregate(); return
    torch.set_num_threads(2)
    torch.backends.mha.set_fastpath_enabled(False)
    for fold in args.folds or [1, 2, 3, 4, 5]:
        evaluate_fold(fold, torch.device(args.device), args.batch_size)
    aggregate()


if __name__ == "__main__":
    main()
