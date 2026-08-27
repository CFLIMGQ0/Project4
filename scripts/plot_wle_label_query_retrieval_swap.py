#!/usr/bin/env python3
"""Plot a sampled WLE counterfactual test of label-query text retrieval."""

from __future__ import annotations

import argparse
import json
import os
import random
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import yaml
from torch.utils.data import DataLoader
from tqdm import tqdm

ROOT = Path(__file__).resolve().parents[1]
PROJECT_ROOT = ROOT.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.task3_tsne import build_model, load_masked_records, load_test_records
from training.data import InstanceAwareBatchSampler, MILBagDataset, mil_collate_fn


LABEL_DISPLAY_NAMES = (
    "Esophageal SMT",
    "Esophageal mucosal lesion",
    "Gastritis",
)
CONDITION_NAMES = (
    "Correct label retrieval",
    "Cross-label replacement",
    "Pooled-text baseline",
)
CONDITION_COLORS = ("#C23B3B", "#4C78A8", "#999999")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fold", type=int, default=1)
    parser.add_argument("--sample-size", type=int, default=48)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument(
        "--output",
        type=Path,
        default=PROJECT_ROOT / "temp_img/wle_label_query_retrieval_swap_sample.png",
    )
    return parser.parse_args()


def resolve_device(raw: str) -> torch.device:
    if raw != "auto":
        return torch.device(raw)
    if not torch.cuda.is_available():
        return torch.device("cpu")
    free = [torch.cuda.mem_get_info(index)[0] for index in range(torch.cuda.device_count())]
    return torch.device(f"cuda:{int(np.argmax(free))}")


def build_loader(
    records: list[dict[str, object]],
    cfg: dict[str, object],
    *,
    seed: int,
    num_workers: int,
) -> DataLoader:
    run_cfg = cfg["training"]["run_overrides"]
    max_instances = int(run_cfg["eval_max_instances"])
    dataset = MILBagDataset(
        records=records,
        task_name="task2",
        max_instances=max_instances,
        min_instances=1,
        bag_sampling_strategy=str(run_cfg["eval_sampling_strategy"]),
        is_train=False,
        image_size=int(cfg["training"]["image_size"]),
        random_instance_dropout=0.0,
        image_cache_mode="disk",
        image_cache_dir=PROJECT_ROOT / "datasets/image_cache/shared",
        image_cache_manifest=PROJECT_ROOT / "datasets/image_cache/task3_cache_manifest.jsonl.gz",
        memory_cache_size=0,
        split_name="test",
    )
    sampler = InstanceAwareBatchSampler(
        records=records,
        max_instances_per_bag=max_instances,
        min_instances_per_bag=1,
        batch_size=1,
        max_instances_per_batch=int(run_cfg["eval_max_batch_instances"]),
        shuffle=False,
        seed=seed,
    )
    loader_kwargs: dict[str, object] = {
        "batch_sampler": sampler,
        "num_workers": max(0, num_workers),
        "pin_memory": torch.cuda.is_available(),
        "collate_fn": mil_collate_fn,
        "persistent_workers": False,
    }
    if num_workers > 0:
        loader_kwargs["prefetch_factor"] = 1
    return DataLoader(dataset, **loader_kwargs)


def safe_text_inputs(
    text_tokens: torch.Tensor,
    text_mask: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    safe_mask = text_mask.bool().clone()
    empty_rows = ~safe_mask.any(dim=1)
    if empty_rows.any():
        safe_mask[empty_rows, 0] = True
        text_tokens = text_tokens.clone()
        text_tokens[empty_rows, 0] = 0.0
    return text_tokens, ~safe_mask


def classify_with_text(
    model: torch.nn.Module,
    label_embeds: torch.Tensor,
    text_embeds: torch.Tensor,
    active: torch.Tensor,
) -> torch.Tensor:
    text_embeds = text_embeds * active
    gates = torch.sigmoid(model.text_gate(torch.cat([label_embeds, text_embeds], dim=-1)))
    gates = gates * active
    return torch.sigmoid(model.classify(label_embeds + gates * text_embeds))


def counterfactual_probabilities(
    model: torch.nn.Module,
    batch: dict[str, object],
    device: torch.device,
) -> tuple[np.ndarray, np.ndarray]:
    images = batch["images"].to(device, non_blocking=True)
    mask = batch["mask"].to(device, non_blocking=True)
    watch_ids = batch["watch_token_ids"].to(device, non_blocking=True)
    watch_mask = batch["watch_token_mask"].to(device, non_blocking=True)
    labels = batch["labels"].float().cpu().numpy()

    _, label_embeds, _, _ = model.encode_long_mil(images, mask)
    text_tokens, text_mask, text_pooled, text_active = model.text_encoder(
        watch_ids,
        watch_mask,
        batch_size=images.shape[0],
        device=device,
    )
    safe_tokens, key_padding_mask = safe_text_inputs(text_tokens, text_mask)
    text_queries = label_embeds + model.label_query_bias
    retrieved, _ = model.text_cross_attn(
        text_queries,
        safe_tokens,
        safe_tokens,
        key_padding_mask=key_padding_mask,
        need_weights=False,
    )
    active = text_active.view(-1, 1, 1).to(dtype=retrieved.dtype)
    correct = classify_with_text(model, label_embeds, retrieved, active)

    swapped_probabilities = []
    for permutation in ((1, 2, 0), (2, 0, 1)):
        swapped = retrieved[:, permutation, :]
        swapped_probabilities.append(classify_with_text(model, label_embeds, swapped, active))
    swapped = torch.stack(swapped_probabilities, dim=0).mean(dim=0)

    pooled = text_pooled.unsqueeze(1).expand(-1, label_embeds.shape[1], -1)
    pooled_probabilities = classify_with_text(model, label_embeds, pooled, active)
    stacked = torch.stack((correct, swapped, pooled_probabilities), dim=1)
    return labels, stacked.float().cpu().numpy()


def mean_ci95(values: np.ndarray) -> tuple[float, float]:
    mean = float(values.mean())
    if values.size < 2:
        return mean, 0.0
    return mean, float(1.96 * values.std(ddof=1) / np.sqrt(values.size))


def plot(values_by_label: list[list[np.ndarray]], output: Path) -> None:
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 9,
            "axes.labelsize": 10,
            "axes.titlesize": 11,
            "legend.fontsize": 8.5,
            "savefig.facecolor": "white",
        }
    )
    figure, axes = plt.subplots(1, 3, figsize=(11.8, 3.9))
    rng = np.random.default_rng(2026)
    for label_index, axis in enumerate(axes):
        all_values = np.concatenate(values_by_label[label_index])
        lower = max(0.0, float(all_values.min()) - 0.06)
        upper = min(1.0, float(all_values.max()) + 0.08)
        for condition_index, values in enumerate(values_by_label[label_index]):
            x_position = float(condition_index)
            jitter = rng.normal(0.0, 0.035, size=values.size)
            axis.scatter(
                np.full(values.size, x_position) + jitter,
                values,
                s=12,
                color=CONDITION_COLORS[condition_index],
                alpha=0.18,
                linewidths=0,
                zorder=2,
            )
            mean, ci = mean_ci95(values)
            axis.errorbar(
                x_position,
                mean,
                yerr=ci,
                fmt="o",
                markersize=7,
                color=CONDITION_COLORS[condition_index],
                markeredgecolor="white",
                markeredgewidth=0.8,
                capsize=4,
                linewidth=1.5,
                zorder=4,
                label=CONDITION_NAMES[condition_index],
            )
            axis.text(
                x_position,
                min(upper - 0.012, mean + ci + 0.018),
                f"{mean:.3f}",
                ha="center",
                va="bottom",
                fontsize=8,
                fontweight="bold",
                color="#222222",
            )
        axis.set_title(LABEL_DISPLAY_NAMES[label_index], loc="left", fontweight="bold")
        axis.set_xlim(-0.45, 2.45)
        axis.set_ylim(lower, upper)
        axis.set_xticks((0, 1, 2), ("Correct", "Cross-label", "Pooled"))
        axis.grid(axis="y", color="#E5E5E5", linewidth=0.7)
        axis.spines[["top", "right"]].set_visible(False)
        if label_index == 0:
            axis.set_ylabel("Positive-label confidence")

    handles, labels = axes[0].get_legend_handles_labels()
    figure.legend(
        handles,
        labels,
        loc="upper right",
        bbox_to_anchor=(0.99, 1.01),
        frameon=False,
        ncol=3,
        columnspacing=1.1,
        handletextpad=0.4,
    )
    figure.tight_layout(rect=(0, 0, 1, 0.90), w_pad=2.0)
    output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output, dpi=300, bbox_inches="tight")
    plt.close(figure)


def main() -> None:
    args = parse_args()
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    os.environ.setdefault("PROJECT4_DISABLE_DISK_CACHE_WRITE", "1")

    config_path = ROOT / "configs/task3/t3_main_model.yaml"
    cfg = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    train_dir = PROJECT_ROOT / "outputs/train_runs/task3/t3_main_model"
    record_map = load_masked_records(train_dir / "records_cache.json")
    fold_dir = train_dir / "regular_white_light" / f"fold_{args.fold}"
    test_records = load_test_records(fold_dir / "split_manifest.csv", record_map)
    rng = np.random.default_rng(args.seed)
    sample_count = min(int(args.sample_size), len(test_records))
    selected_indices = sorted(rng.choice(len(test_records), size=sample_count, replace=False).tolist())
    sampled_records = [test_records[index] for index in selected_indices]

    device = resolve_device(args.device)
    model = build_model(cfg, fold_dir / "checkpoints/best_macro_f1.ckpt", device)
    loader = build_loader(
        sampled_records,
        cfg,
        seed=args.seed,
        num_workers=args.num_workers,
    )

    all_labels = []
    all_probabilities = []
    with torch.inference_mode():
        for batch in tqdm(loader, desc="WLE label-query counterfactual", unit="exam"):
            labels, probabilities = counterfactual_probabilities(model, batch, device)
            all_labels.append(labels)
            all_probabilities.append(probabilities)
    labels = np.concatenate(all_labels, axis=0)
    probabilities = np.concatenate(all_probabilities, axis=0)

    values_by_label: list[list[np.ndarray]] = []
    summary: dict[str, object] = {
        "dataset": "WLE",
        "fold": int(args.fold),
        "sampled_examinations": int(sample_count),
        "seed": int(args.seed),
        "labels": {},
    }
    for label_index, label_name in enumerate(LABEL_DISPLAY_NAMES):
        positive = labels[:, label_index] > 0.5
        current = [probabilities[positive, condition_index, label_index] for condition_index in range(3)]
        values_by_label.append(current)
        summary["labels"][label_name] = {
            "positive_instances": int(positive.sum()),
            **{
                CONDITION_NAMES[index]: float(values.mean())
                for index, values in enumerate(current)
            },
        }

    plot(values_by_label, args.output.resolve())
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    print(args.output.resolve())


if __name__ == "__main__":
    main()
