#!/usr/bin/env python3
"""Compare attention-deletion faithfulness across WLE MIL models."""

from __future__ import annotations

import argparse
import json
import math
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

from baselines.task1.gastro_baseline import build_gastro_baseline
from scripts.plot_apro_mechanism_preview import (
    build_cache_dataset,
    load_model,
    make_batch,
    move_batch,
    uniform_indices,
)
from scripts.plot_wle_mechanism_evidence import records_from_manifest
from sotas.task1.gastro_sota import build_gastro_sota
from sotas.task1.gastro_sota.common import split_group_ranges


TABLE2_ROOT = PROJECT_ROOT / "outputs/train_runs/task3/t3_table2_5fold"
WLE_KEY = "regular_white_light"
OUTPUT_STEM = PROJECT_ROOT / "temp_img/wle_attention_faithfulness_baselines"
MODEL_KEYS = ("attention_mil", "clam_mb", "dsmil", "transmil", "dtfd_mil", "ours")
DISPLAY_NAMES = {
    "attention_mil": "Attention\nMIL",
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
    parser.add_argument("--random-repeats", type=int, default=3)
    parser.add_argument("--seed", type=int, default=2026)
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


def load_checkpoint(model: torch.nn.Module, checkpoint_path: Path, device: torch.device) -> torch.nn.Module:
    try:
        checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    except TypeError:
        checkpoint = torch.load(checkpoint_path, map_location="cpu")
    model.load_state_dict(checkpoint["model_state"], strict=True)
    return model.to(device).eval()


def load_table2_model(model_key: str, fold: int, device: torch.device) -> torch.nn.Module:
    run_dir = TABLE2_ROOT / "image" / WLE_KEY / f"fold_{fold}" / model_key
    config = yaml.safe_load((run_dir / "config.yaml").read_text(encoding="utf-8"))
    kwargs = {"num_labels": 3, "pretrained": False, **dict(config["model_params"])}
    if model_key == "attention_mil":
        model = build_gastro_baseline(str(config["model_name"]), **kwargs)
    else:
        model = build_gastro_sota(str(config["model_name"]), **kwargs)
    return load_checkpoint(model, run_dir / "checkpoints/best_macro_f1.ckpt", device)


def labelwise_linear(classifiers: torch.nn.ModuleList, embeds: torch.Tensor) -> torch.Tensor:
    return torch.stack(
        [classifier(embeds[:, index, :]).squeeze(-1) for index, classifier in enumerate(classifiers)],
        dim=1,
    )


def forward_from_features(
    key: str,
    model: torch.nn.Module,
    features: torch.Tensor,
    mask: torch.Tensor,
    instance_indices: torch.Tensor,
    original_image_counts: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    if key == "attention_mil":
        embeds, attention = model.mil_pool(features, mask)
        return labelwise_linear(model.classifiers, embeds), attention

    if key == "clam_mb":
        embeds, attention = model.label_attention(features, mask)
        return model.bag_classifier(embeds), attention

    if key == "dsmil":
        instance_logits = model.instance_classifier(features)
        masked_logits = instance_logits.masked_fill(
            ~mask.unsqueeze(-1).to(dtype=torch.bool), torch.finfo(instance_logits.dtype).min
        )
        critical_indices = masked_logits.transpose(1, 2).argmax(dim=-1)
        critical_features = torch.gather(
            features.unsqueeze(1).expand(-1, model.num_labels, -1, -1),
            2,
            critical_indices.unsqueeze(-1).unsqueeze(-1).expand(-1, -1, 1, features.size(-1)),
        ).squeeze(2)
        query = model.query_proj(critical_features)
        key_features = model.key_proj(features)
        value = model.value_proj(features)
        attention_logits = torch.einsum("bld,bnd->bln", query, key_features) / math.sqrt(features.size(-1))
        attention_logits = attention_logits.masked_fill(
            ~mask.unsqueeze(1).to(dtype=torch.bool), torch.finfo(attention_logits.dtype).min
        )
        attention = torch.softmax(attention_logits, dim=-1)
        attention = attention * mask.unsqueeze(1).to(dtype=attention.dtype)
        attention = attention / attention.sum(dim=-1, keepdim=True).clamp_min(1e-12)
        embeds = torch.einsum("bln,bnd->bld", attention, value)
        bag_logits = model.bag_classifier(embeds)
        critical_logits = torch.gather(
            instance_logits.transpose(1, 2), 2, critical_indices.unsqueeze(-1)
        ).squeeze(-1)
        return 0.5 * (bag_logits + critical_logits), attention

    if key == "transmil":
        label_tokens = model.label_tokens.unsqueeze(0).expand(features.size(0), -1, -1)
        sequence = torch.cat([label_tokens, features], dim=1)
        padding_mask = torch.cat(
            [
                torch.zeros(features.size(0), model.num_labels, device=mask.device, dtype=torch.bool),
                ~mask.to(dtype=torch.bool),
            ],
            dim=1,
        )
        encoded = model.transformer(sequence, src_key_padding_mask=padding_mask)
        label_states = encoded[:, : model.num_labels, :]
        instance_states = encoded[:, model.num_labels :, :]
        attention_logits = torch.einsum("bld,bnd->bln", label_states, instance_states) / math.sqrt(
            features.size(-1)
        )
        attention_logits = attention_logits.masked_fill(
            ~mask.unsqueeze(1).to(dtype=torch.bool), torch.finfo(attention_logits.dtype).min
        )
        attention = torch.softmax(attention_logits, dim=-1)
        attention = attention * mask.unsqueeze(1).to(dtype=attention.dtype)
        attention = attention / attention.sum(dim=-1, keepdim=True).clamp_min(1e-12)
        embeds = label_states + torch.einsum("bln,bnd->bld", attention, instance_states)
        return model.bag_classifier(embeds), attention

    if key == "dtfd_mil":
        full_logits = model.full_scorer(features).transpose(1, 2)
        full_logits = full_logits.masked_fill(
            ~mask.unsqueeze(1).to(dtype=torch.bool), torch.finfo(full_logits.dtype).min
        )
        attention = torch.softmax(full_logits, dim=-1)
        attention = attention * mask.unsqueeze(1).to(dtype=attention.dtype)
        attention = attention / attention.sum(dim=-1, keepdim=True).clamp_min(1e-12)
        group_embeds = []
        group_logits = []
        group_valid = []
        for start, end in split_group_ranges(features.size(1), model.num_groups):
            current_features = features[:, start:end, :]
            current_mask = mask[:, start:end]
            embeds, _ = model.group_attention(current_features, current_mask)
            group_embeds.append(embeds)
            group_logits.append(model.group_classifier(embeds))
            group_valid.append(current_mask.any(dim=1))
        stacked_embeds = torch.stack(group_embeds, dim=1)
        stacked_logits = torch.stack(group_logits, dim=1)
        valid_mask = torch.stack(group_valid, dim=1)
        group_scores = torch.sigmoid(stacked_logits).transpose(1, 2)
        group_scores = group_scores.masked_fill(
            ~valid_mask.unsqueeze(1), torch.finfo(group_scores.dtype).min
        )
        group_weights = torch.softmax(group_scores, dim=-1)
        group_weights = group_weights * valid_mask.unsqueeze(1).to(dtype=group_weights.dtype)
        group_weights = group_weights / group_weights.sum(dim=-1, keepdim=True).clamp_min(1e-12)
        embeds = torch.einsum("blg,bgld->bld", group_weights, stacked_embeds)
        return model.final_classifier(embeds), attention

    if key == "ours":
        positioned, attention_bias, _ = model._encode_position(
            features, mask, instance_indices, original_image_counts
        )
        if attention_bias is None:
            context = model.context_encoder(positioned, src_key_padding_mask=~mask)
        else:
            batch_size, num_heads, num_instances, _ = attention_bias.shape
            attention_bias = attention_bias.masked_fill((~mask)[:, None, None, :], -1e4)
            context = model.context_encoder(
                positioned,
                mask=attention_bias.reshape(batch_size * num_heads, num_instances, num_instances),
            )
        context = context * mask.unsqueeze(-1).to(dtype=context.dtype)
        embeds, attention = model.mil_pool(context, mask)
        refined, _ = model.refine_labels(embeds)
        return model.classify(refined), attention

    raise ValueError(key)


def encode_features(key: str, model: torch.nn.Module, images: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    if key == "attention_mil":
        return model.encode_instances(images)
    if key == "ours":
        features, _ = model.encode_instances(images, mask)
        return features
    return model.instance_encoder(images)


def deletion_masks(
    attention: torch.Tensor,
    mask: torch.Tensor,
    fraction: float,
    mode: str,
    repeats: int,
    rng: np.random.Generator,
) -> torch.Tensor:
    rows: list[torch.Tensor] = []
    valid_positions = torch.where(mask[0])[0]
    delete_count = min(
        valid_positions.numel() - 1,
        max(1, int(round(float(fraction) * valid_positions.numel()))),
    )
    for _ in range(repeats):
        for label_index in range(attention.shape[1]):
            row = mask[0].clone()
            if mode == "top":
                scores = attention[0, label_index, valid_positions]
                selected = valid_positions[torch.topk(scores, k=delete_count).indices]
            elif mode == "random":
                chosen = rng.choice(valid_positions.numel(), size=delete_count, replace=False)
                selected = valid_positions[torch.as_tensor(chosen, device=mask.device, dtype=torch.long)]
            else:
                raise ValueError(mode)
            row[selected] = False
            rows.append(row)
    return torch.stack(rows, dim=0)


def selected_label_confidence(probabilities: torch.Tensor, targets: torch.Tensor, repeats: int) -> float:
    label_count = targets.shape[1]
    probabilities = probabilities.reshape(repeats, label_count, label_count)
    indices = torch.arange(label_count, device=probabilities.device)
    selected = probabilities[:, indices, indices]
    target_values = targets[0].unsqueeze(0).expand(repeats, -1)
    confidence = torch.where(target_values > 0.5, selected, 1.0 - selected)
    return float(confidence.mean().float().cpu().item())


@torch.inference_mode()
def collect(args: argparse.Namespace) -> dict[str, Any]:
    fold = int(args.fold)
    manifest = TABLE2_ROOT / "data_splits" / WLE_KEY / f"fold_{fold}" / "split_manifest.csv"
    records = records_from_manifest(manifest)
    dataset = build_cache_dataset(records)
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    models = {key: load_table2_model(key, fold, device) for key in MODEL_KEYS if key != "ours"}
    models["ours"] = load_model("apro_full", WLE_KEY, fold, device)
    fractions = np.asarray([0.10, 0.20, 0.30, 0.40], dtype=float)
    rng = np.random.default_rng(int(args.seed))
    method_rows: dict[str, list[dict[str, Any]]] = {key: [] for key in MODEL_KEYS}

    for record_index, record in enumerate(records, start=1):
        indices = uniform_indices(len(record["image_paths"]), int(args.max_images))
        batch, _ = make_batch(record, indices, dataset)
        batch_device = move_batch(batch, device)
        images = batch_device["images"]
        base_mask = batch_device["mask"].to(dtype=torch.bool)
        targets = batch_device["labels"].float()
        base_indices = batch_device["instance_indices"]
        base_counts = batch_device["original_image_counts"]

        for key, model in models.items():
            with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=device.type == "cuda"):
                features = encode_features(key, model, images, base_mask)
                original_logits, original_attention = forward_from_features(
                    key, model, features, base_mask, base_indices, base_counts
                )
            original_probabilities = torch.sigmoid(original_logits).float()
            original_confidence = torch.where(
                targets > 0.5, original_probabilities, 1.0 - original_probabilities
            ).mean()
            top_drops: list[float] = []
            random_drops: list[float] = []

            for fraction in fractions:
                for mode, repeats, destination in (
                    ("top", 1, top_drops),
                    ("random", int(args.random_repeats), random_drops),
                ):
                    masks = deletion_masks(
                        original_attention,
                        base_mask,
                        float(fraction),
                        mode,
                        repeats,
                        rng,
                    )
                    batch_size = masks.shape[0]
                    repeated_features = features.expand(batch_size, -1, -1)
                    repeated_indices = base_indices.expand(batch_size, -1)
                    repeated_counts = base_counts.expand(batch_size)
                    with torch.autocast(
                        device_type=device.type,
                        dtype=torch.float16,
                        enabled=device.type == "cuda",
                    ):
                        logits, _ = forward_from_features(
                            key,
                            model,
                            repeated_features,
                            masks,
                            repeated_indices,
                            repeated_counts,
                        )
                    perturbed_confidence = selected_label_confidence(
                        torch.sigmoid(logits).float(), targets, repeats
                    )
                    destination.append(float(original_confidence.float().cpu().item()) - perturbed_confidence)

            differences = np.asarray(top_drops) - np.asarray(random_drops)
            method_rows[key].append(
                {
                    "top_drop": top_drops,
                    "random_drop": random_drops,
                    "faithfulness_score": float(differences.mean()),
                }
            )

        if record_index % 10 == 0 or record_index == len(records):
            print(f"WLE faithfulness audit: {record_index}/{len(records)}", flush=True)

    summaries: dict[str, Any] = {}
    for key, rows in method_rows.items():
        scores = np.asarray([row["faithfulness_score"] for row in rows], dtype=float)
        top_curves = np.asarray([row["top_drop"] for row in rows], dtype=float)
        random_curves = np.asarray([row["random_drop"] for row in rows], dtype=float)
        summaries[key] = {
            "mean": float(scores.mean()),
            "median": float(np.median(scores)),
            "ci95": float(1.96 * scores.std(ddof=1) / np.sqrt(scores.size)),
            "values": scores.tolist(),
            "top_curve_mean": top_curves.mean(axis=0).tolist(),
            "random_curve_mean": random_curves.mean(axis=0).tolist(),
        }
    return {
        "dataset": "WLE",
        "fold": fold,
        "num_held_out_examinations": len(records),
        "input_protocol": f"same {int(args.max_images)} uniformly sampled images per examination",
        "fractions": fractions.tolist(),
        "random_repeats": int(args.random_repeats),
        "metric": "mean across deletion fractions of (high-attention confidence drop - random confidence drop)",
        "intervention": "label-specific feature deletion followed by re-running each model's aggregation and prediction layers",
        "higher_means": "more faithful attention ranking",
        "methods": summaries,
    }


def plot(payload: dict[str, Any]) -> None:
    order = MODEL_KEYS
    colors = ("#9A9A9A", "#6F9FD8", "#6F9FD8", "#6F9FD8", "#6F9FD8", "#C23B3B")
    rng = np.random.default_rng(2026)
    figure, axis = plt.subplots(figsize=(9.4, 5.5))
    all_values = [100 * np.asarray(payload["methods"][key]["values"], dtype=float) for key in order]
    lower = min(float(values.min()) for values in all_values)
    upper = max(float(values.max()) for values in all_values)
    span = max(upper - lower, 0.01)
    for position, (key, color, values) in enumerate(zip(order, colors, all_values)):
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
        label_y = float(np.quantile(values, 0.75)) + 0.045 * span
        axis.text(
            position,
            label_y,
            f"{mean:+.3f} pp",
            ha="center",
            va="bottom",
            color="#C23B3B" if key == "ours" else "#333333",
            fontweight="bold",
        )
    padding = 0.10 * span
    axis.set_ylim(lower - padding, upper + 0.16 * span)
    axis.axhline(0, color="#333333", linewidth=0.9)
    axis.set_xticks(range(len(order)), [DISPLAY_NAMES[key] for key in order])
    axis.set_ylabel("Attention faithfulness score (pp)")
    axis.set_title("WLE: decision relevance of selected image evidence", fontweight="bold")
    axis.grid(axis="y", color="#E5E5E5", linewidth=0.7)
    axis.spines[["top", "right"]].set_visible(False)
    figure.text(
        0.5,
        0.015,
        "Mean AOPC(high-attention deletion − random deletion); higher is more faithful",
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
