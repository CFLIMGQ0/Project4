#!/usr/bin/env python3
"""Compare label-wise Top-5 attention overlap on the same WLE held-out images."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import yaml


ROOT = Path(__file__).resolve().parents[1]
PROJECT_ROOT = ROOT.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.plot_apro_mechanism_preview import (
    build_cache_dataset,
    make_batch,
    move_batch,
    uniform_indices,
)
from scripts.plot_wle_mechanism_evidence import records_from_manifest
from sotas.task1.gastro_sota import build_gastro_sota


TABLE2_ROOT = PROJECT_ROOT / "outputs/train_runs/task3/t3_table2_5fold"
WLE_KEY = "regular_white_light"
OURS_SOURCE = PROJECT_ROOT / "temp_img/wle_label_attention_mechanism.json"
OUTPUT_STEM = PROJECT_ROOT / "temp_img/wle_attention_overlap_baselines"
MODEL_KEYS = ("clam_mb", "dsmil", "transmil", "dtfd_mil")
DISPLAY_NAMES = {
    "shared_attention": "Shared\nattention",
    "clam_mb": "CLAM-MB",
    "dsmil": "DSMIL",
    "transmil": "TransMIL",
    "dtfd_mil": "DTFD-MIL",
    "ours": "Ours",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--fold", type=int, default=1)
    parser.add_argument("--max-images", type=int, default=64)
    return parser.parse_args()


def configure_style() -> None:
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 10,
            "axes.titlesize": 12,
            "axes.labelsize": 10.5,
            "xtick.labelsize": 9.5,
            "ytick.labelsize": 9.5,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "savefig.facecolor": "white",
        }
    )


def load_baseline(model_key: str, fold: int, device: torch.device) -> torch.nn.Module:
    run_dir = TABLE2_ROOT / "image" / WLE_KEY / f"fold_{fold}" / model_key
    config = yaml.safe_load((run_dir / "config.yaml").read_text(encoding="utf-8"))
    model = build_gastro_sota(
        str(config["model_name"]),
        num_labels=3,
        pretrained=False,
        **dict(config["model_params"]),
    )
    checkpoint_path = run_dir / "checkpoints/best_macro_f1.ckpt"
    try:
        checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    except TypeError:
        checkpoint = torch.load(checkpoint_path, map_location="cpu")
    model.load_state_dict(checkpoint["model_state"], strict=True)
    return model.to(device).eval()


def pairwise_topk_jaccard(attention: torch.Tensor, top_k: int = 5) -> float:
    attention = attention[0].float().cpu()
    use_k = min(int(top_k), attention.shape[-1])
    top_sets = [set(torch.topk(row, k=use_k).indices.tolist()) for row in attention]
    values: list[float] = []
    for left in range(len(top_sets)):
        for right in range(left + 1, len(top_sets)):
            union = top_sets[left] | top_sets[right]
            values.append(len(top_sets[left] & top_sets[right]) / max(len(union), 1))
    return float(np.mean(values))


@torch.inference_mode()
def collect(args: argparse.Namespace) -> dict[str, Any]:
    fold = int(args.fold)
    manifest = TABLE2_ROOT / "data_splits" / WLE_KEY / f"fold_{fold}" / "split_manifest.csv"
    records = records_from_manifest(manifest)
    dataset = build_cache_dataset(records)
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    models = {key: load_baseline(key, fold, device) for key in MODEL_KEYS}
    values: dict[str, list[float]] = {key: [] for key in MODEL_KEYS}

    for record_index, record in enumerate(records, start=1):
        indices = uniform_indices(len(record["image_paths"]), int(args.max_images))
        batch, _ = make_batch(record, indices, dataset)
        batch_device = move_batch(batch, device)
        for key, model in models.items():
            with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=device.type == "cuda"):
                outputs = model(batch_device["images"], batch_device["mask"])
            values[key].append(pairwise_topk_jaccard(outputs["attention"], top_k=5))
        if record_index % 10 == 0 or record_index == len(records):
            print(f"WLE overlap audit: {record_index}/{len(records)}", flush=True)

    ours_payload = json.loads(OURS_SOURCE.read_text(encoding="utf-8"))["overlap"]
    if int(ours_payload["fold"]) != fold or int(ours_payload["num_held_out_examinations"]) != len(records):
        raise RuntimeError("Ours overlap source does not match the selected WLE fold")
    ours_values = list(ours_payload["multi_label_attention"]["exam_mean_jaccard"])
    all_values: dict[str, list[float]] = {
        "shared_attention": [1.0] * len(records),
        **values,
        "ours": ours_values,
    }
    summaries = {}
    for key, method_values in all_values.items():
        array = np.asarray(method_values, dtype=float)
        summaries[key] = {
            "mean": float(array.mean()),
            "median": float(np.median(array)),
            "ci95": float(1.96 * array.std(ddof=1) / np.sqrt(array.size)) if array.size > 1 else 0.0,
            "values": method_values,
        }
    return {
        "dataset": "WLE",
        "fold": fold,
        "num_held_out_examinations": len(records),
        "input_protocol": f"same {int(args.max_images)} uniformly sampled images per examination",
        "metric": "mean pairwise Jaccard overlap among three label-specific Top-5 image sets",
        "lower_means": "stronger separation of label-specific selected images",
        "methods": summaries,
    }


def plot(payload: dict[str, Any]) -> None:
    order = ("shared_attention", "clam_mb", "dsmil", "transmil", "dtfd_mil", "ours")
    colors = ("#9A9A9A", "#6F9FD8", "#6F9FD8", "#6F9FD8", "#6F9FD8", "#C23B3B")
    rng = np.random.default_rng(2026)
    figure, axis = plt.subplots(figsize=(9.4, 5.5))
    for position, (key, color) in enumerate(zip(order, colors)):
        values = np.asarray(payload["methods"][key]["values"], dtype=float)
        jitter = rng.uniform(-0.12, 0.12, size=values.size)
        axis.scatter(position + jitter, values, s=10, alpha=0.20, color=color, edgecolors="none")
        axis.boxplot(
            values,
            positions=[position],
            widths=0.48,
            patch_artist=True,
            showfliers=False,
            medianprops={"color": "black", "linewidth": 1.3},
            whiskerprops={"color": color},
            capprops={"color": color},
            boxprops={"facecolor": color, "edgecolor": color, "alpha": 0.42},
        )
        mean = float(values.mean())
        label_y = min(1.04, max(mean + 0.055, float(np.quantile(values, 0.75)) + 0.035))
        axis.text(
            position,
            label_y,
            f"{100 * mean:.1f}%",
            ha="center",
            va="bottom",
            color="#C23B3B" if key == "ours" else "#333333",
            fontweight="bold",
        )
    axis.set_xticks(range(len(order)), [DISPLAY_NAMES[key] for key in order])
    axis.set_ylim(-0.04, 1.10)
    axis.set_ylabel("Pairwise Top-5 Jaccard overlap")
    axis.set_title("WLE: label-specific evidence separation across MIL methods", fontweight="bold")
    axis.grid(axis="y", color="#E5E5E5", linewidth=0.7)
    axis.spines[["top", "right"]].set_visible(False)
    figure.text(
        0.5,
        0.015,
        f"Held-out fold {payload['fold']} examinations (n={payload['num_held_out_examinations']}); same 64 images per examination; lower is more label-specific",
        ha="center",
        color="#555555",
    )
    figure.tight_layout(rect=(0, 0.05, 1, 1))
    figure.savefig(OUTPUT_STEM.with_suffix(".png"), dpi=300, bbox_inches="tight")
    figure.savefig(OUTPUT_STEM.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(figure)
    OUTPUT_STEM.with_suffix(".json").write_text(json.dumps(payload, indent=2), encoding="utf-8")


def main() -> None:
    args = parse_args()
    configure_style()
    plot(collect(args))


if __name__ == "__main__":
    main()
