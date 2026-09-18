#!/usr/bin/env python3
"""Run the Table 2 image-only MIL baselines and AMEF image branch on CQ500."""

from __future__ import annotations

import argparse
import csv
import json
import re
from pathlib import Path

import numpy as np

import run_physionet_ct_ich_table2_baselines as base


LABEL_COLUMNS = ("IPH", "MassEffect", "MidlineShift")
LABEL_NAMES = ("IPH", "Mass Effect", "Midline Shift")
base.LABEL_NAMES = LABEL_NAMES


def parse_args() -> argparse.Namespace:
    project_root = Path(__file__).resolve().parents[2]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--reads",
        type=Path,
        default=project_root / "datasets" / "cq500" / "raw" / "reads.csv",
    )
    parser.add_argument(
        "--feature-cache",
        type=Path,
        default=project_root / "outputs" / "cq500" / "convnext_tiny_scan_features.npz",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=project_root / "outputs" / "cq500" / "table2_image_baselines",
    )
    parser.add_argument("--models", nargs="+", default=list(base.MODEL_SPECS))
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--weight-decay", type=float, default=0.02)
    parser.add_argument("--warmup-ratio", type=float, default=0.2)
    parser.add_argument("--max-instances", type=int, default=28)
    parser.add_argument("--instance-dropout", type=float, default=0.25)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def majority_label(row: dict[str, str], label: str) -> int:
    return int(sum(int(row[f"R{reader}:{label}"]) for reader in (1, 2, 3)) >= 2)


def load_bags(
    reads_path: Path, feature_cache: Path
) -> tuple[np.ndarray, list[np.ndarray], np.ndarray]:
    cached = np.load(feature_cache)
    features = cached["features"].astype(np.float32)
    cached_ids = cached["patient_ids"].astype(np.int64)
    cached_orders = cached["slice_numbers"].astype(np.int64)
    labels: dict[int, np.ndarray] = {}
    with reads_path.open("r", encoding="utf-8-sig", newline="") as file:
        for row in csv.DictReader(file):
            scan_id = int(re.search(r"(\d+)$", row["name"]).group(1))
            labels[scan_id] = np.asarray(
                [majority_label(row, label) for label in LABEL_COLUMNS], dtype=np.int64
            )
    patient_ids = np.asarray(sorted(np.unique(cached_ids)), dtype=np.int64)
    missing = sorted(set(labels) - set(patient_ids.tolist()))
    if missing:
        raise RuntimeError(f"特征缓存缺少{len(missing)}例检查：{missing[:10]}")
    bags: list[np.ndarray] = []
    targets: list[np.ndarray] = []
    for patient_id in patient_ids:
        indices = np.flatnonzero(cached_ids == patient_id)
        indices = indices[np.argsort(cached_orders[indices])]
        bags.append(features[indices])
        targets.append(labels[int(patient_id)])
    return patient_ids, bags, np.stack(targets)


def main() -> None:
    args = parse_args()
    unknown = [model for model in args.models if model not in base.MODEL_SPECS]
    if unknown:
        raise ValueError(f"未知模型：{unknown}")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    base.seed_everything(args.seed)
    patient_ids, bags, targets = load_bags(args.reads, args.feature_cache)
    folds = base.multilabel_folds(targets, args.folds, args.seed)
    split_payload = {
        "patient_ids": patient_ids.tolist(),
        "labels": list(LABEL_NAMES),
        "positive_counts": targets.sum(axis=0).tolist(),
        "co_positive_count": int((targets.sum(axis=1) >= 2).sum()),
        "all_negative_count": int((targets.sum(axis=1) == 0).sum()),
        "folds": [patient_ids[indices].tolist() for indices in folds],
        "fold_positive_counts": [targets[indices].sum(axis=0).tolist() for indices in folds],
    }
    (args.output_dir / "patient_folds.json").write_text(
        json.dumps(split_payload, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    results = []
    for model_key in args.models:
        result = base.run_model(model_key, patient_ids, bags, targets, folds, args)
        result["protocol"].update(
            {
                "label_consensus": "positive when at least two of three radiologists agree",
                "slice_sampling": "one preferred axial series and up to 28 central 5-mm-equivalent slices",
                "ct_windows": "brain, bone, and subdural windows encoded as three channels",
            }
        )
        (args.output_dir / f"{model_key}.json").write_text(
            json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        results.append(result)
    base.write_summary(results, args.output_dir)


if __name__ == "__main__":
    main()
