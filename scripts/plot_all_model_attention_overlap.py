#!/usr/bin/env python3
"""Plot label-wise Top-5 evidence overlap for every available image MIL model."""

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
from tqdm import tqdm


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
from scripts.plot_wle_attention_overlap_baselines import pairwise_topk_jaccard
from scripts.plot_wle_mechanism_evidence import records_from_manifest
from baselines.task1.gastro_baseline import GASTRO_BASELINE_CLASS_REGISTRY, build_gastro_baseline
from sotas.task1.gastro_sota import build_gastro_sota


TABLE2_ROOT = PROJECT_ROOT / "outputs/train_runs/task3/t3_table2_5fold"
POSITION_ROOT = PROJECT_ROOT / "outputs/train_runs/task3/t3_apro_cope_ablation/apro_full"
SOURCE_PATH = PROJECT_ROOT / "temp_img/attention_evidence_four_datasets.json"
OUTPUT_STEM = PROJECT_ROOT / "temp_img/attention_overlap_all_models_four_datasets"

DATASETS = (
    ("regular_white_light", "WLE"),
    ("chromoscopic", "Chromoscopic"),
    ("surgical", "Surgical"),
    ("ultrasound", "EUS"),
)

# These methods were already evaluated with the identical protocol in SOURCE_PATH.
REUSED_KEYS = ("clam_mb", "dsmil", "transmil", "dtfd_mil", "ours")
# Compute the remaining trained models so the figure covers the complete Table-2 image suite.
COMPUTED_KEYS = (
    "mean_pooling",
    "max_pooling",
    "attention_mil",
    "transformer_context_mil",
    "topk_mil",
    "clam_sb",
)
METHOD_ORDER = (
    "shared_attention",
    "mean_pooling",
    "max_pooling",
    "attention_mil",
    "transformer_context_mil",
    "topk_mil",
    "clam_sb",
    "clam_mb",
    "dsmil",
    "transmil",
    "dtfd_mil",
    "ours",
)
METHOD_NAMES = {
    "shared_attention": "Shared\ncontrol",
    "mean_pooling": "Mean\npooling",
    "max_pooling": "Max\npooling",
    "attention_mil": "Attention\nMIL",
    "transformer_context_mil": "Transformer-\ncontext MIL",
    "topk_mil": "Top-k\nMIL",
    "clam_sb": "CLAM-SB",
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
    parser.add_argument(
        "--dataset",
        choices=[key for key, _ in DATASETS],
        help="Only compute one dataset and save an intermediate shard.",
    )
    parser.add_argument(
        "--merge-only",
        action="store_true",
        help="Merge existing dataset shards and render the final figure.",
    )
    return parser.parse_args()


def configure_style() -> None:
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 9.0,
            "axes.titlesize": 11,
            "axes.labelsize": 9.5,
            "xtick.labelsize": 7.7,
            "ytick.labelsize": 8.5,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "savefig.facecolor": "white",
        }
    )


def load_baseline(
    model_key: str,
    dataset: str,
    fold: int,
    device: torch.device,
) -> torch.nn.Module:
    run_dir = TABLE2_ROOT / "image" / dataset / f"fold_{fold}" / model_key
    config = yaml.safe_load((run_dir / "config.yaml").read_text(encoding="utf-8"))
    model_name = str(config["model_name"])
    builder = build_gastro_baseline if model_name in GASTRO_BASELINE_CLASS_REGISTRY else build_gastro_sota
    model = builder(model_name, num_labels=3, pretrained=False, **dict(config["model_params"]))
    checkpoint_path = run_dir / "checkpoints/best_macro_f1.ckpt"
    try:
        checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    except TypeError:
        checkpoint = torch.load(checkpoint_path, map_location="cpu")
    model.load_state_dict(checkpoint["model_state"], strict=True)
    return model.to(device).eval()


def native_evidence_scores(
    model_key: str,
    model: torch.nn.Module,
    outputs: dict[str, torch.Tensor],
    mask: torch.Tensor,
) -> torch.Tensor:
    """Return the model's native per-label instance ranking scores.

    Top-k MIL and max pooling expose only the selected weights in ``attention``.
    Their pre-selection scoring heads are used here so Top-5 does not contain
    arbitrary zero-weight ties. Other models use their returned attention maps.
    """

    features = outputs["instance_features"]
    if model_key == "topk_mil":
        scores = model.instance_scorer(features).transpose(1, 2)
    elif model_key == "max_pooling":
        scores = torch.stack(
            [scorer(features).squeeze(-1) for scorer in model.instance_scorers],
            dim=1,
        )
    else:
        return outputs["attention"]
    return scores.masked_fill(
        ~mask.to(dtype=torch.bool).unsqueeze(1),
        torch.finfo(scores.dtype).min,
    )


def summary(values: list[float]) -> dict[str, Any]:
    array = np.asarray(values, dtype=float)
    return {
        "mean": float(array.mean()),
        "median": float(np.median(array)),
        "ci95": float(1.96 * array.std(ddof=1) / np.sqrt(array.size)) if array.size > 1 else 0.0,
        "values": values,
    }


@torch.inference_mode()
def collect_dataset(
    args: argparse.Namespace,
    dataset: str,
    display_name: str,
    source: dict[str, Any],
) -> dict[str, Any]:
    fold = int(args.fold)
    manifest = POSITION_ROOT / dataset / f"fold_{fold}" / "split_manifest.csv"
    records = records_from_manifest(manifest)
    source_dataset = source["datasets"][dataset]
    if len(records) != int(source_dataset["num_held_out_examinations"]):
        raise RuntimeError(f"{display_name} source record count does not match the current manifest")

    cache_dataset = build_cache_dataset(records)
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    models = {key: load_baseline(key, dataset, fold, device) for key in COMPUTED_KEYS}
    values: dict[str, list[float]] = {key: [] for key in COMPUTED_KEYS}

    progress = tqdm(records, desc=f"{display_name} all-model overlap", unit="exam")
    for record in progress:
        indices = uniform_indices(len(record["image_paths"]), int(args.max_images))
        batch, _ = make_batch(record, indices, cache_dataset)
        batch_device = move_batch(batch, device)
        images = batch_device["images"]
        mask = batch_device["mask"]
        for key, model in models.items():
            with torch.autocast(
                device_type=device.type,
                dtype=torch.float16,
                enabled=device.type == "cuda",
            ):
                outputs = model(images, mask)
                evidence = native_evidence_scores(key, model, outputs, mask)
            values[key].append(pairwise_topk_jaccard(evidence, top_k=5))

    methods: dict[str, Any] = {
        "shared_attention": summary([1.0] * len(records)),
        **{key: summary(values[key]) for key in COMPUTED_KEYS},
    }
    for key in REUSED_KEYS:
        methods[key] = source_dataset["overlap"]["methods"][key]

    return {
        "dataset": dataset,
        "display_name": display_name,
        "fold": fold,
        "num_held_out_examinations": len(records),
        "max_images": int(args.max_images),
        "overlap": {"top_k": 5, "methods": {key: methods[key] for key in METHOD_ORDER}},
    }


def shard_path(dataset: str) -> Path:
    return OUTPUT_STEM.with_name(f"{OUTPUT_STEM.name}_{dataset}.json")


def plot(payloads: dict[str, Any]) -> None:
    neutral = "#999999"
    baseline = "#6F9FD8"
    ours = "#C23B3B"
    colors = [neutral if key == "shared_attention" else ours if key == "ours" else baseline for key in METHOD_ORDER]
    figure, axes = plt.subplots(2, 2, figsize=(17.2, 9.4), sharey=True)
    rng = np.random.default_rng(2026)

    for axis, (dataset, display_name) in zip(axes.flat, DATASETS):
        payload = payloads[dataset]
        for position, (key, color) in enumerate(zip(METHOD_ORDER, colors)):
            values = np.asarray(payload["overlap"]["methods"][key]["values"], dtype=float)
            jitter = rng.uniform(-0.10, 0.10, size=values.size)
            axis.scatter(position + jitter, values, s=7, alpha=0.13, color=color, edgecolors="none")
            axis.boxplot(
                values,
                positions=[position],
                widths=0.45,
                patch_artist=True,
                showfliers=False,
                medianprops={"color": "black", "linewidth": 1.0},
                whiskerprops={"color": color},
                capprops={"color": color},
                boxprops={"facecolor": color, "edgecolor": color, "alpha": 0.40},
            )
            mean = float(values.mean())
            label_y = min(1.045, max(mean + 0.042, float(np.quantile(values, 0.75)) + 0.022))
            axis.text(
                position,
                label_y,
                f"{100 * mean:.1f}%",
                ha="center",
                va="bottom",
                fontsize=7.2,
                color=ours if key == "ours" else "#333333",
                fontweight="bold",
            )
        axis.set_xticks(range(len(METHOD_ORDER)), [METHOD_NAMES[key] for key in METHOD_ORDER])
        axis.set_ylim(-0.04, 1.10)
        axis.set_title(
            f"{display_name} (n={payload['num_held_out_examinations']})",
            loc="left",
            fontweight="bold",
        )
        axis.grid(axis="y", color="#E5E5E5", linewidth=0.7)
        axis.spines[["top", "right"]].set_visible(False)

    axes[0, 0].set_ylabel("Pairwise Top-5 evidence-set Jaccard overlap")
    axes[1, 0].set_ylabel("Pairwise Top-5 evidence-set Jaccard overlap")
    figure.suptitle(
        "A  Label-specific evidence separation across all image MIL models",
        fontweight="bold",
        fontsize=14,
    )
    figure.text(
        0.5,
        0.010,
        "Held-out fold 1; same 64 sampled images per examination; lower overlap is more label-specific",
        ha="center",
        color="#555555",
    )
    figure.tight_layout(rect=(0, 0.045, 1, 0.96), h_pad=2.0, w_pad=1.8)
    figure.savefig(OUTPUT_STEM.with_suffix(".png"), dpi=300, bbox_inches="tight")
    figure.savefig(OUTPUT_STEM.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(figure)


def merge_and_plot(args: argparse.Namespace) -> None:
    payloads = {
        dataset: json.loads(shard_path(dataset).read_text(encoding="utf-8"))
        for dataset, _ in DATASETS
    }
    output = {
        "protocol": {
            "fold": int(args.fold),
            "max_images": int(args.max_images),
            "metric": "mean pairwise Jaccard among three label-specific Top-5 evidence sets",
            "score_definition": (
                "native label attention; pre-selection label-instance scores for Top-k MIL and max pooling"
            ),
            "lower_means": "stronger separation of label-specific selected images",
        },
        "datasets": payloads,
    }
    OUTPUT_STEM.with_suffix(".json").write_text(json.dumps(output, indent=2), encoding="utf-8")
    plot(payloads)


def main() -> None:
    args = parse_args()
    configure_style()
    OUTPUT_STEM.parent.mkdir(parents=True, exist_ok=True)
    if args.merge_only:
        merge_and_plot(args)
        return

    source = json.loads(SOURCE_PATH.read_text(encoding="utf-8"))
    selected = DATASETS if args.dataset is None else tuple(item for item in DATASETS if item[0] == args.dataset)
    for dataset, display_name in selected:
        payload = collect_dataset(args, dataset, display_name, source)
        shard_path(dataset).write_text(json.dumps(payload, indent=2), encoding="utf-8")
    if args.dataset is None:
        merge_and_plot(args)


if __name__ == "__main__":
    main()
