#!/usr/bin/env python3
"""Generate four-dataset attention separation and deletion evidence figures."""

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
    load_model,
    make_batch,
    move_batch,
    uniform_indices,
)
from scripts.plot_wle_attention_overlap_baselines import pairwise_topk_jaccard
from scripts.plot_wle_mechanism_evidence import records_from_manifest
from sotas.task1.gastro_sota import build_gastro_sota


TABLE2_ROOT = PROJECT_ROOT / "outputs/train_runs/task3/t3_table2_5fold"
POSITION_ROOT = PROJECT_ROOT / "outputs/train_runs/task3/t3_apro_cope_ablation/apro_full"
OUTPUT_DIR = PROJECT_ROOT / "temp_img"
DATASETS = (
    ("regular_white_light", "WLE"),
    ("chromoscopic", "Chromoscopic"),
    ("surgical", "Surgical"),
    ("ultrasound", "EUS"),
)
BASELINE_KEYS = ("clam_mb", "dsmil", "transmil", "dtfd_mil")
METHOD_ORDER = ("shared_attention", *BASELINE_KEYS, "ours")
METHOD_NAMES = {
    "shared_attention": "Shared\nattention",
    "clam_mb": "CLAM-MB",
    "dsmil": "DSMIL",
    "transmil": "TransMIL",
    "dtfd_mil": "DTFD-MIL",
    "ours": "Ours",
}
WLE_OVERLAP_SOURCE = OUTPUT_DIR / "wle_attention_overlap_baselines.json"
WLE_DELETION_SOURCE = OUTPUT_DIR / "wle_attention_deletion_faithfulness.json"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", default="cuda:3")
    parser.add_argument("--fold", type=int, default=1)
    parser.add_argument("--max-images", type=int, default=64)
    parser.add_argument("--random-repeats", type=int, default=3)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--recompute-wle", action="store_true")
    return parser.parse_args()


def configure_style() -> None:
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 9.5,
            "axes.titlesize": 11,
            "axes.labelsize": 9.5,
            "xtick.labelsize": 8.5,
            "ytick.labelsize": 8.5,
            "legend.fontsize": 8.5,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "savefig.facecolor": "white",
        }
    )


def load_baseline(model_key: str, dataset: str, fold: int, device: torch.device) -> torch.nn.Module:
    run_dir = TABLE2_ROOT / "image" / dataset / f"fold_{fold}" / model_key
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


def mean_ci95(values: list[float] | np.ndarray) -> tuple[float, float]:
    array = np.asarray(values, dtype=float)
    mean = float(array.mean())
    ci = float(1.96 * array.std(ddof=1) / np.sqrt(array.size)) if array.size > 1 else 0.0
    return mean, ci


def confidence(probabilities: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    return torch.where(targets > 0.5, probabilities, 1.0 - probabilities).mean(dim=1)


def pooled_probabilities(
    model: torch.nn.Module,
    features: torch.Tensor,
    attention: torch.Tensor,
) -> torch.Tensor:
    bag_embeds = torch.einsum("bln,bnd->bld", attention, features)
    refined, _ = model.refine_labels(bag_embeds)
    return torch.sigmoid(model.classify(refined)).float()


def delete_attention(
    attention: torch.Tensor,
    fraction: float,
    mode: str,
    rng: np.random.Generator,
) -> torch.Tensor:
    if fraction <= 0:
        return attention
    result = attention.clone()
    num_instances = attention.shape[-1]
    delete_count = min(num_instances - 1, max(1, int(round(fraction * num_instances))))
    for label_index in range(attention.shape[1]):
        values = attention[0, label_index]
        if mode == "top":
            positions = torch.topk(values, k=delete_count, largest=True).indices
        elif mode == "bottom":
            positions = torch.topk(values, k=delete_count, largest=False).indices
        elif mode == "random":
            positions = torch.as_tensor(
                rng.choice(num_instances, size=delete_count, replace=False),
                device=values.device,
                dtype=torch.long,
            )
        else:
            raise ValueError(mode)
        result[0, label_index, positions] = 0
    return result / result.sum(dim=-1, keepdim=True).clamp_min(1e-12)


@torch.inference_mode()
def collect_dataset(args: argparse.Namespace, dataset: str, display_name: str) -> dict[str, Any]:
    fold = int(args.fold)
    manifest = POSITION_ROOT / dataset / f"fold_{fold}" / "split_manifest.csv"
    records = records_from_manifest(manifest)
    cache_dataset = build_cache_dataset(records)
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    baselines = {key: load_baseline(key, dataset, fold, device) for key in BASELINE_KEYS}
    ours_model = load_model("apro_full", dataset, fold, device)
    overlaps: dict[str, list[float]] = {key: [] for key in BASELINE_KEYS}
    overlaps["ours"] = []
    fractions = np.asarray([0.0, 0.10, 0.20, 0.30, 0.40], dtype=float)
    modes = ("top", "random", "bottom")
    drops = {mode: [[] for _ in fractions] for mode in modes}
    rng = np.random.default_rng(int(args.seed))

    for record_index, record in enumerate(records, start=1):
        indices = uniform_indices(len(record["image_paths"]), int(args.max_images))
        batch, _ = make_batch(record, indices, cache_dataset)
        batch_device = move_batch(batch, device)
        images = batch_device["images"]
        mask = batch_device["mask"]

        for key, model in baselines.items():
            with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=device.type == "cuda"):
                outputs = model(images, mask)
            overlaps[key].append(pairwise_topk_jaccard(outputs["attention"], top_k=5))

        with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=device.type == "cuda"):
            features, _, attention, _ = ours_model.encode_long_mil(
                images,
                mask,
                batch_device["instance_indices"],
                batch_device["original_image_counts"],
            )
            targets = batch_device["labels"].float()
            original_probabilities = pooled_probabilities(ours_model, features, attention)
            original_confidence = confidence(original_probabilities, targets)
        overlaps["ours"].append(pairwise_topk_jaccard(attention, top_k=5))

        for fraction_index, fraction in enumerate(fractions):
            for mode in modes:
                repeats = int(args.random_repeats) if mode == "random" and fraction > 0 else 1
                perturbed_confidences = []
                for _ in range(repeats):
                    perturbed_attention = delete_attention(attention, float(fraction), mode, rng)
                    with torch.autocast(
                        device_type=device.type,
                        dtype=torch.float16,
                        enabled=device.type == "cuda",
                    ):
                        probabilities = pooled_probabilities(ours_model, features, perturbed_attention)
                    perturbed_confidences.append(confidence(probabilities, targets))
                perturbed = torch.stack(perturbed_confidences).mean(dim=0)
                drops[mode][fraction_index].append(
                    float((original_confidence - perturbed).float().cpu().item())
                )

        if record_index % 10 == 0 or record_index == len(records):
            print(f"{display_name}: {record_index}/{len(records)}", flush=True)

    method_values = {"shared_attention": [1.0] * len(records), **overlaps}
    overlap_payload: dict[str, Any] = {"top_k": 5, "methods": {}}
    for key in METHOD_ORDER:
        values = method_values[key]
        mean, ci = mean_ci95(values)
        overlap_payload["methods"][key] = {
            "mean": mean,
            "median": float(np.median(values)),
            "ci95": ci,
            "values": values,
        }

    deletion_payload: dict[str, Any] = {
        "fractions": fractions.tolist(),
        "random_repeats": int(args.random_repeats),
        "curves": {},
    }
    for mode in modes:
        values = [np.asarray(group, dtype=float) for group in drops[mode]]
        means = [float(group.mean()) for group in values]
        cis = [
            float(1.96 * group.std(ddof=1) / np.sqrt(group.size)) if group.size > 1 else 0.0
            for group in values
        ]
        deletion_payload["curves"][mode] = {"mean": means, "ci95": cis}

    return {
        "dataset": dataset,
        "display_name": display_name,
        "fold": fold,
        "num_held_out_examinations": len(records),
        "max_images": int(args.max_images),
        "overlap": overlap_payload,
        "deletion": deletion_payload,
    }


def load_wle() -> dict[str, Any]:
    overlap = json.loads(WLE_OVERLAP_SOURCE.read_text(encoding="utf-8"))
    deletion = json.loads(WLE_DELETION_SOURCE.read_text(encoding="utf-8"))
    return {
        "dataset": "regular_white_light",
        "display_name": "WLE",
        "fold": int(overlap["fold"]),
        "num_held_out_examinations": int(overlap["num_held_out_examinations"]),
        "max_images": 64,
        "overlap": {"top_k": 5, "methods": overlap["methods"]},
        "deletion": {
            "fractions": deletion["fractions"],
            "random_repeats": deletion["random_repeats"],
            "curves": deletion["curves"],
        },
    }


def plot_overlap(all_payloads: dict[str, Any]) -> None:
    colors = ("#999999", "#6F9FD8", "#6F9FD8", "#6F9FD8", "#6F9FD8", "#C23B3B")
    figure, axes = plt.subplots(2, 2, figsize=(12.2, 9.0), sharey=True)
    rng = np.random.default_rng(2026)
    for axis, (dataset, display_name) in zip(axes.flat, DATASETS):
        payload = all_payloads[dataset]
        for position, (key, color) in enumerate(zip(METHOD_ORDER, colors)):
            values = np.asarray(payload["overlap"]["methods"][key]["values"], dtype=float)
            jitter = rng.uniform(-0.10, 0.10, size=values.size)
            axis.scatter(position + jitter, values, s=8, alpha=0.16, color=color, edgecolors="none")
            axis.boxplot(
                values,
                positions=[position],
                widths=0.45,
                patch_artist=True,
                showfliers=False,
                medianprops={"color": "black", "linewidth": 1.1},
                whiskerprops={"color": color},
                capprops={"color": color},
                boxprops={"facecolor": color, "edgecolor": color, "alpha": 0.40},
            )
            mean = float(values.mean())
            label_y = min(1.04, max(mean + 0.045, float(np.quantile(values, 0.75)) + 0.025))
            axis.text(
                position,
                label_y,
                f"{100 * mean:.1f}%",
                ha="center",
                va="bottom",
                fontsize=8,
                color="#C23B3B" if key == "ours" else "#333333",
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
    axes[0, 0].set_ylabel("Pairwise Top-5 Jaccard overlap")
    axes[1, 0].set_ylabel("Pairwise Top-5 Jaccard overlap")
    figure.suptitle(
        "A  Label-specific evidence separation across four gastroscopy datasets",
        fontweight="bold",
        fontsize=14,
    )
    figure.text(
        0.5,
        0.012,
        "Held-out fold 1; same 64 sampled images per examination; lower overlap is more label-specific",
        ha="center",
        color="#555555",
    )
    figure.tight_layout(rect=(0, 0.045, 1, 0.96), h_pad=2.0, w_pad=1.8)
    figure.savefig(OUTPUT_DIR / "attention_overlap_four_datasets.png", dpi=300, bbox_inches="tight")
    figure.savefig(OUTPUT_DIR / "attention_overlap_four_datasets.pdf", bbox_inches="tight")
    plt.close(figure)


def plot_deletion(all_payloads: dict[str, Any]) -> None:
    styles = {
        "top": ("High-attention images", "#C23B3B", "o"),
        "random": ("Random images", "#6F6F6F", "s"),
        "bottom": ("Low-attention images", "#2878B5", "^"),
    }
    figure, axes = plt.subplots(2, 2, figsize=(11.8, 8.8))
    for axis_index, (axis, (dataset, display_name)) in enumerate(zip(axes.flat, DATASETS)):
        payload = all_payloads[dataset]
        fractions = 100 * np.asarray(payload["deletion"]["fractions"], dtype=float)
        for key, (label, color, marker) in styles.items():
            means = 100 * np.asarray(payload["deletion"]["curves"][key]["mean"], dtype=float)
            cis = 100 * np.asarray(payload["deletion"]["curves"][key]["ci95"], dtype=float)
            axis.plot(
                fractions,
                means,
                color=color,
                marker=marker,
                linewidth=2.0,
                markersize=5.5,
                label=label,
            )
            axis.fill_between(fractions, means - cis, means + cis, color=color, alpha=0.13)
        axis.axhline(0, color="#333333", linewidth=0.9)
        axis.set_title(
            f"{display_name} (n={payload['num_held_out_examinations']})",
            loc="left",
            fontweight="bold",
        )
        axis.set_xlabel("Deleted images per label (%)")
        axis.set_ylabel("True-class confidence decrease (pp)")
        axis.grid(color="#E5E5E5", linewidth=0.7)
        axis.spines[["top", "right"]].set_visible(False)
        if axis_index == 0:
            axis.legend(frameon=False, loc="best")
    figure.suptitle(
        "B  Label-attention deletion faithfulness across four gastroscopy datasets",
        fontweight="bold",
        fontsize=14,
    )
    figure.text(
        0.5,
        0.012,
        "Held-out fold 1; mean and 95% CI; positive values indicate confidence loss after deletion",
        ha="center",
        color="#555555",
    )
    figure.tight_layout(rect=(0, 0.045, 1, 0.96), h_pad=2.0, w_pad=1.8)
    figure.savefig(OUTPUT_DIR / "attention_deletion_four_datasets.png", dpi=300, bbox_inches="tight")
    figure.savefig(OUTPUT_DIR / "attention_deletion_four_datasets.pdf", bbox_inches="tight")
    plt.close(figure)


def main() -> None:
    args = parse_args()
    configure_style()
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    all_payloads: dict[str, Any] = {}
    for dataset, display_name in DATASETS:
        if dataset == "regular_white_light" and not args.recompute_wle:
            all_payloads[dataset] = load_wle()
        else:
            all_payloads[dataset] = collect_dataset(args, dataset, display_name)
    output_payload = {
        "protocol": {
            "fold": int(args.fold),
            "max_images": int(args.max_images),
            "overlap_metric": "mean pairwise Jaccard among the three label-specific Top-5 image sets",
            "deletion_metric": "original true-class confidence minus confidence after attention-weight deletion",
            "random_repeats": int(args.random_repeats),
        },
        "datasets": all_payloads,
    }
    (OUTPUT_DIR / "attention_evidence_four_datasets.json").write_text(
        json.dumps(output_payload, indent=2), encoding="utf-8"
    )
    plot_overlap(all_payloads)
    plot_deletion(all_payloads)


if __name__ == "__main__":
    main()
