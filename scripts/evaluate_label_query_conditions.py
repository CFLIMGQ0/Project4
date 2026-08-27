#!/usr/bin/env python3
"""Evaluate correct, cross-label, and pooled text conditions on one full test split."""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np
import torch
import yaml
from tqdm import tqdm

ROOT = Path(__file__).resolve().parents[1]
PROJECT_ROOT = ROOT.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.plot_wle_label_query_retrieval_swap import (
    LABEL_DISPLAY_NAMES,
    build_loader,
    counterfactual_probabilities,
    resolve_device,
)
from scripts.task3_tsne import build_model, load_masked_records, load_test_records


DATASETS = (
    "regular_white_light",
    "chromoscopic",
    "surgical",
    "ultrasound",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", choices=DATASETS, required=True)
    parser.add_argument("--fold", type=int, default=1)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=PROJECT_ROOT / "temp_img/label_query_conditions_full",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    np.random.seed(2026)
    torch.manual_seed(2026)
    os.environ.setdefault("PROJECT4_DISABLE_DISK_CACHE_WRITE", "1")
    cfg = yaml.safe_load(
        (ROOT / "configs/task3/t3_main_model.yaml").read_text(encoding="utf-8")
    )
    train_root = PROJECT_ROOT / "outputs/train_runs/task3/t3_main_model"
    fold_dir = train_root / args.dataset / f"fold_{args.fold}"
    records = load_masked_records(train_root / "records_cache.json")
    test_records = load_test_records(fold_dir / "split_manifest.csv", records)
    device = resolve_device(args.device)
    model = build_model(cfg, fold_dir / "checkpoints/best_macro_f1.ckpt", device)
    loader = build_loader(test_records, cfg, seed=2026, num_workers=args.num_workers)

    labels_list: list[np.ndarray] = []
    probabilities_list: list[np.ndarray] = []
    with torch.inference_mode():
        for batch in tqdm(loader, desc=f"Evaluate {args.dataset}", unit="exam"):
            labels, probabilities = counterfactual_probabilities(model, batch, device)
            labels_list.append(labels)
            probabilities_list.append(probabilities)
    labels = np.concatenate(labels_list, axis=0)
    probabilities = np.concatenate(probabilities_list, axis=0)

    positive_values: dict[str, np.ndarray] = {}
    per_label: dict[str, object] = {}
    condition_names = ("correct", "cross_label", "pooled")
    for label_index, label_name in enumerate(LABEL_DISPLAY_NAMES):
        positive = labels[:, label_index] > 0.5
        per_label[label_name] = {
            "positive_instances": int(positive.sum()),
            **{
                condition_names[condition_index]: float(
                    probabilities[positive, condition_index, label_index].mean()
                )
                for condition_index in range(3)
            },
        }
        for condition_index, condition_name in enumerate(condition_names):
            positive_values[f"label_{label_index}_{condition_name}"] = probabilities[
                positive, condition_index, label_index
            ]

    overall = {}
    for condition_name in condition_names:
        current = np.concatenate(
            [positive_values[f"label_{index}_{condition_name}"] for index in range(3)]
        )
        overall[condition_name] = float(current.mean())
    overall["positive_label_observations"] = int(
        sum(per_label[name]["positive_instances"] for name in LABEL_DISPLAY_NAMES)
    )
    payload = {
        "dataset": args.dataset,
        "fold": args.fold,
        "test_examinations": len(test_records),
        "overall": overall,
        "labels": per_label,
    }
    output_dir = args.output_dir.resolve() / args.dataset / f"fold_{args.fold}"
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "stats.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    np.savez_compressed(output_dir / "positive_conditions.npz", **positive_values)
    print(json.dumps(payload, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
