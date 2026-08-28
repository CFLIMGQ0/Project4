#!/usr/bin/env python3
"""Generate two mechanism-level WLE evidence figures without using Macro F1."""

from __future__ import annotations

import argparse
import copy
import csv
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

from exp_8 import build_exp8_model
from scripts.plot_apro_mechanism_preview import (
    build_cache_dataset,
    load_model,
    make_batch,
    move_batch,
    uniform_indices,
)
from scripts.task3_main_model_5fold import apply_watch_mask


LABELS = ("SMT", "EML", "Gastritis")
OUTPUT_DIR = PROJECT_ROOT / "temp_img"
RECORDS_CACHE = PROJECT_ROOT / "outputs/train_runs/task3/t3_main_model/records_cache.json"
POSITION_ROOT = PROJECT_ROOT / "outputs/train_runs/task3/t3_apro_cope_ablation/apro_full"
MODULE_ROOT = PROJECT_ROOT / "outputs/train_runs/task3/t3_table4_module_ablation"
DELETION_SOURCE = OUTPUT_DIR / "wle_fig2_attention_deletion.json"
ATTENTION_OUTPUT = OUTPUT_DIR / "wle_label_attention_mechanism.json"
WLE_KEY = "regular_white_light"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--attention-fold", type=int, default=1)
    parser.add_argument("--hypergraph-fold", type=int, default=2)
    parser.add_argument("--mode", choices=("attention", "hypergraph", "all"), default="all")
    return parser.parse_args()


def configure_style() -> None:
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 10,
            "axes.titlesize": 11.5,
            "axes.labelsize": 10.5,
            "xtick.labelsize": 9.5,
            "ytick.labelsize": 9.5,
            "legend.fontsize": 9,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "savefig.facecolor": "white",
        }
    )


def load_record_map() -> dict[str, dict[str, Any]]:
    payload = json.loads(RECORDS_CACHE.read_text(encoding="utf-8"))
    records = copy.deepcopy(payload["records"])
    apply_watch_mask(records, enabled=True)
    return {str(record["exam_dir"]): record for record in records}


def records_from_manifest(manifest: Path) -> list[dict[str, Any]]:
    record_map = load_record_map()
    selected: list[dict[str, Any]] = []
    with manifest.open("r", encoding="utf-8-sig", newline="") as file:
        for row in csv.DictReader(file):
            if str(row.get("split", "")).strip().lower() == "test":
                selected.append(record_map[str(row["exam_dir"])])
    return selected


def mean_ci95(values: list[float] | np.ndarray) -> tuple[float, float]:
    array = np.asarray(values, dtype=float)
    mean = float(array.mean())
    ci = float(1.96 * array.std(ddof=1) / np.sqrt(array.size)) if array.size > 1 else 0.0
    return mean, ci


def save_figure(figure: plt.Figure, stem: str) -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    figure.savefig(OUTPUT_DIR / f"{stem}.png", dpi=300, bbox_inches="tight")
    figure.savefig(OUTPUT_DIR / f"{stem}.pdf", bbox_inches="tight")
    plt.close(figure)


@torch.inference_mode()
def collect_attention_overlap(args: argparse.Namespace) -> dict[str, Any]:
    fold = int(args.attention_fold)
    manifest = POSITION_ROOT / WLE_KEY / f"fold_{fold}" / "split_manifest.csv"
    records = records_from_manifest(manifest)
    dataset = build_cache_dataset(records)
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    model = load_model("apro_full", WLE_KEY, fold, device)
    exam_overlaps: list[float] = []
    pair_overlaps: list[float] = []

    for record_index, record in enumerate(records, start=1):
        indices = uniform_indices(len(record["image_paths"]), 64)
        batch, _ = make_batch(record, indices, dataset)
        batch_device = move_batch(batch, device)
        with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=device.type == "cuda"):
            _, _, attention, _ = model.encode_long_mil(
                batch_device["images"],
                batch_device["mask"],
                batch_device["instance_indices"],
                batch_device["original_image_counts"],
            )
        attention = attention[0].float().cpu()
        top_k = min(5, attention.shape[-1])
        top_sets = [set(torch.topk(row, k=top_k).indices.tolist()) for row in attention]
        current: list[float] = []
        for left in range(len(top_sets)):
            for right in range(left + 1, len(top_sets)):
                union = top_sets[left] | top_sets[right]
                overlap = len(top_sets[left] & top_sets[right]) / max(len(union), 1)
                pair_overlaps.append(float(overlap))
                current.append(float(overlap))
        exam_overlaps.append(float(np.mean(current)))
        if record_index % 20 == 0 or record_index == len(records):
            print(f"attention overlap: {record_index}/{len(records)}", flush=True)

    mean, ci = mean_ci95(exam_overlaps)
    return {
        "dataset": "WLE",
        "fold": fold,
        "num_held_out_examinations": len(records),
        "top_k": 5,
        "shared_attention_overlap": 1.0,
        "multi_label_attention": {
            "exam_mean_jaccard": exam_overlaps,
            "pairwise_jaccard": pair_overlaps,
            "mean": mean,
            "ci95": ci,
        },
    }


def plot_attention(overlap: dict[str, Any], deletion: dict[str, Any]) -> None:
    figure, axes = plt.subplots(1, 2, figsize=(11.4, 4.8))
    axis = axes[0]
    rng = np.random.default_rng(2026)
    multi = np.asarray(overlap["multi_label_attention"]["exam_mean_jaccard"], dtype=float)
    shared = np.ones_like(multi)
    for index, (values, color) in enumerate(((shared, "#8C8C8C"), (multi, "#2878B5"))):
        jitter = rng.uniform(-0.10, 0.10, size=values.size)
        axis.scatter(index + jitter, values, s=11, alpha=0.28, color=color, edgecolors="none")
        box = axis.boxplot(
            values,
            positions=[index],
            widths=0.42,
            patch_artist=True,
            showfliers=False,
            medianprops={"color": "black", "linewidth": 1.4},
            whiskerprops={"color": color},
            capprops={"color": color},
            boxprops={"facecolor": color, "edgecolor": color, "alpha": 0.35},
        )
        del box
    axis.set_xticks([0, 1], ["Shared attention\n(baseline)", "Multi-label\nattention"])
    axis.set_ylim(-0.04, 1.08)
    axis.set_ylabel("Pairwise Top-5 Jaccard overlap")
    axis.set_title("A  Label-specific evidence separation", loc="left", fontweight="bold")
    axis.text(0, 1.025, "100%", ha="center", va="bottom", fontweight="bold", color="#555555")
    axis.text(
        1,
        min(1.02, float(multi.mean()) + 0.08),
        f"{100 * multi.mean():.1f}%",
        ha="center",
        va="bottom",
        fontweight="bold",
        color="#2878B5",
    )
    axis.grid(axis="y", color="#E5E5E5", linewidth=0.7)

    axis = axes[1]
    fractions = 100 * np.asarray(deletion["fractions"], dtype=float)
    styles = {
        "top": ("High-attention", "#C23B3B", "o"),
        "random": ("Random", "#777777", "s"),
        "bottom": ("Low-attention", "#2878B5", "^"),
    }
    for key, (label, color, marker) in styles.items():
        means = 100 * np.asarray(deletion["curves"][key]["mean"], dtype=float)
        cis = 100 * np.asarray(deletion["curves"][key]["ci95"], dtype=float)
        axis.plot(fractions, means, color=color, marker=marker, linewidth=2.0, label=label)
        axis.fill_between(fractions, means - cis, means + cis, color=color, alpha=0.13)
    axis.axhline(0, color="#333333", linewidth=0.9)
    axis.set_xlabel("Deleted images (%)")
    axis.set_ylabel("True-class confidence decrease (pp)")
    axis.set_title("B  Attention faithfulness by deletion", loc="left", fontweight="bold")
    axis.grid(color="#E5E5E5", linewidth=0.7)
    axis.legend(frameon=False, loc="upper left")
    for axis in axes:
        axis.spines[["top", "right"]].set_visible(False)
    figure.suptitle("WLE: multi-label attention selects distinct, decision-relevant images", fontweight="bold")
    figure.text(
        0.5,
        0.015,
        f"Held-out fold {overlap['fold']} examinations (n={overlap['num_held_out_examinations']}); deletion bands show 95% CI",
        ha="center",
        color="#555555",
    )
    figure.tight_layout(rect=(0, 0.05, 1, 0.95), w_pad=2.8)
    save_figure(figure, "wle_label_attention_mechanism")
    payload = {"overlap": overlap, "deletion": deletion}
    (OUTPUT_DIR / "wle_label_attention_mechanism.json").write_text(
        json.dumps(payload, indent=2), encoding="utf-8"
    )


def load_module_model(variant: str, fold: int, device: torch.device) -> torch.nn.Module:
    run_dir = MODULE_ROOT / variant / WLE_KEY / f"fold_{fold}"
    config = yaml.safe_load((run_dir / "config.yaml").read_text(encoding="utf-8"))
    params = dict(config["model_params"])
    for unused_key in ("image_aux_weight", "image_distill_weight", "image_distill_temperature"):
        params.pop(unused_key, None)
    model = build_exp8_model(
        model_name=str(config["model_name"]),
        num_labels=len(LABELS),
        pretrained=False,
        **params,
    )
    checkpoint_path = run_dir / "checkpoints/best_macro_f1.ckpt"
    try:
        checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    except TypeError:
        checkpoint = torch.load(checkpoint_path, map_location="cpu")
    missing, unexpected = model.load_state_dict(checkpoint["model_state"], strict=False)
    if missing or unexpected:
        raise RuntimeError(f"Checkpoint mismatch: missing={missing}, unexpected={unexpected}")
    return model.to(device).eval()


def summarize_copositive(rows: list[dict[str, float]]) -> dict[str, Any]:
    summary: dict[str, Any] = {"n": len(rows), "methods": {}}
    for method in ("none", "ordinary_graph", "label_hypergraph"):
        method_summary: dict[str, Any] = {}
        for metric in ("positive_sensitivity", "complete_set_recovery", "positive_confidence"):
            values = [row[f"{method}_{metric}"] for row in rows]
            mean, ci = mean_ci95(values)
            method_summary[metric] = {"mean": mean, "ci95": ci, "values": values}
        summary["methods"][method] = method_summary
    return summary


@torch.inference_mode()
def collect_hypergraph(args: argparse.Namespace) -> dict[str, Any]:
    fold = int(args.hypergraph_fold)
    ordinary_run = MODULE_ROOT / "modules_none" / WLE_KEY / f"fold_{fold}"
    hyper_run = MODULE_ROOT / "modules_2" / WLE_KEY / f"fold_{fold}"
    ordinary_manifest = ordinary_run / "split_manifest.csv"
    hyper_manifest = hyper_run / "split_manifest.csv"
    if ordinary_manifest.read_bytes() != hyper_manifest.read_bytes():
        raise RuntimeError("Ordinary-graph and hypergraph split manifests differ")
    records = records_from_manifest(ordinary_manifest)
    dataset = build_cache_dataset(records)
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    ordinary_model = load_module_model("modules_none", fold, device)
    hyper_model = load_module_model("modules_2", fold, device)
    rows: list[dict[str, float]] = []

    for record_index, record in enumerate(records, start=1):
        indices = uniform_indices(len(record["image_paths"]), 64)
        batch, _ = make_batch(record, indices, dataset)
        batch_device = move_batch(batch, device)
        targets = batch_device["labels"].float()
        if int(targets.sum().item()) < 2:
            continue
        with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=device.type == "cuda"):
            ordinary_features, _ = ordinary_model.encode_instances(
                batch_device["images"], batch_device["mask"]
            )
            ordinary_bags, _ = ordinary_model.mil_pool(ordinary_features, batch_device["mask"])
            none_probabilities = torch.sigmoid(ordinary_model.classify(ordinary_bags)).float()
            ordinary_refined, _ = ordinary_model.refine_labels(ordinary_bags)
            ordinary_probabilities = torch.sigmoid(ordinary_model.classify(ordinary_refined)).float()

            hyper_features, _ = hyper_model.encode_instances(batch_device["images"], batch_device["mask"])
            hyper_bags, _ = hyper_model.mil_pool(hyper_features, batch_device["mask"])
            hyper_refined, _ = hyper_model.refine_labels(hyper_bags)
            hyper_probabilities = torch.sigmoid(hyper_model.classify(hyper_refined)).float()

        row: dict[str, float] = {}
        positives = targets > 0.5
        for method, probabilities in (
            ("none", none_probabilities),
            ("ordinary_graph", ordinary_probabilities),
            ("label_hypergraph", hyper_probabilities),
        ):
            predictions = probabilities >= 0.5
            recovered = predictions[positives]
            row[f"{method}_positive_sensitivity"] = float(recovered.float().mean().cpu().item())
            row[f"{method}_complete_set_recovery"] = float(recovered.all().cpu().item())
            row[f"{method}_positive_confidence"] = float(probabilities[positives].mean().cpu().item())
        rows.append(row)
        if record_index % 20 == 0 or record_index == len(records):
            print(f"hypergraph evidence: {record_index}/{len(records)}", flush=True)

    return {
        "dataset": "WLE",
        "fold": fold,
        "population": "held-out examinations with at least two positive labels",
        "threshold": 0.5,
        "comparison": {
            "none": "inference-time bypass of the ordinary-graph checkpoint",
            "ordinary_graph": "trained learnable ordinary-graph checkpoint",
            "label_hypergraph": "trained label-hypergraph checkpoint",
        },
        "copositive": summarize_copositive(rows),
    }


def plot_hypergraph(payload: dict[str, Any]) -> None:
    methods = ("none", "ordinary_graph", "label_hypergraph")
    labels = ("No reasoning*", "Ordinary graph", "Label hypergraph")
    colors = ("#8C8C8C", "#E69F00", "#C23B3B")
    metrics = (
        ("positive_sensitivity", "Co-positive label sensitivity", "A  Recovery across positive labels"),
        ("complete_set_recovery", "Complete positive-set recovery", "B  Examination-level joint recovery"),
    )
    figure, axes = plt.subplots(1, 2, figsize=(11.2, 4.8), sharey=True)
    for axis, (metric, ylabel, title) in zip(axes, metrics):
        means = np.asarray(
            [payload["copositive"]["methods"][method][metric]["mean"] for method in methods]
        )
        cis = np.asarray(
            [payload["copositive"]["methods"][method][metric]["ci95"] for method in methods]
        )
        positions = np.arange(len(methods))
        axis.bar(positions, means, color=colors, width=0.62, alpha=0.92)
        axis.errorbar(positions, means, yerr=cis, fmt="none", ecolor="#333333", capsize=4, linewidth=1.2)
        for position, mean in zip(positions, means):
            axis.text(position, mean + 0.025, f"{100 * mean:.1f}%", ha="center", fontweight="bold")
        axis.set_xticks(positions, labels)
        axis.set_ylim(0, 1.08)
        axis.set_ylabel(ylabel)
        axis.set_title(title, loc="left", fontweight="bold")
        axis.grid(axis="y", color="#E5E5E5", linewidth=0.7)
        axis.spines[["top", "right"]].set_visible(False)
    figure.suptitle("WLE: hypergraph reasoning improves coexisting-label recovery", fontweight="bold")
    figure.text(
        0.5,
        0.018,
        f"Held-out fold {payload['fold']} multi-positive examinations (n={payload['copositive']['n']}); mean and 95% CI.  *Inference-time bypass.",
        ha="center",
        color="#555555",
    )
    figure.tight_layout(rect=(0, 0.06, 1, 0.95), w_pad=2.6)
    save_figure(figure, "wle_label_hypergraph_mechanism")
    (OUTPUT_DIR / "wle_label_hypergraph_mechanism.json").write_text(
        json.dumps(payload, indent=2), encoding="utf-8"
    )


def main() -> None:
    args = parse_args()
    configure_style()
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    if args.mode in {"attention", "all"}:
        if DELETION_SOURCE.exists():
            deletion = json.loads(DELETION_SOURCE.read_text(encoding="utf-8"))
        elif ATTENTION_OUTPUT.exists():
            deletion = json.loads(ATTENTION_OUTPUT.read_text(encoding="utf-8"))["deletion"]
        else:
            raise FileNotFoundError("Missing WLE attention-deletion statistics")
        plot_attention(collect_attention_overlap(args), deletion)
    if args.mode in {"hypergraph", "all"}:
        hypergraph = collect_hypergraph(args)
        metrics = hypergraph["copositive"]["methods"]
        for metric in ("positive_sensitivity", "complete_set_recovery"):
            values = {method: metrics[method][metric]["mean"] for method in metrics}
            print(metric, values, flush=True)
        plot_hypergraph(hypergraph)


if __name__ == "__main__":
    main()
