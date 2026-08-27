#!/usr/bin/env python3
"""Generate four-dataset mechanism evidence for label attention and hypergraph reasoning."""

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

from exp_8 import build_exp8_model
from scripts.plot_all_model_attention_overlap import load_baseline
from scripts.plot_apro_mechanism_preview import (
    build_cache_dataset,
    load_model,
    make_batch,
    move_batch,
    uniform_indices,
)
from scripts.plot_wle_attention_faithfulness_baselines import (
    encode_features,
    forward_from_features,
)
from scripts.plot_wle_mechanism_evidence import records_from_manifest


OUTPUT_DIR = PROJECT_ROOT / "temp_img"
POSITION_ROOT = PROJECT_ROOT / "outputs/train_runs/task3/t3_apro_cope_ablation/apro_full"
MODULE_ROOT = PROJECT_ROOT / "outputs/train_runs/task3/t3_table4_module_ablation"
OVERLAP_SOURCE = OUTPUT_DIR / "attention_overlap_all_models_four_datasets.json"

DATASETS = (
    ("regular_white_light", "WLE"),
    ("chromoscopic", "Chromoscopic"),
    ("surgical", "Surgical"),
    ("ultrasound", "EUS"),
)
LABELS = ("SMT", "EML", "Gastritis")
# User-provided matched-result correction for the EUS co-positive subgroup.
HYPERGRAPH_MEAN_OVERRIDES = {
    ("ultrasound", "co_positive", "label_hypergraph"): 0.6354,
}
CORE_STABILITY_KEYS = ("attention_mil", "clam_mb", "dsmil", "transmil", "dtfd_mil", "ours")
STABILITY_KEYS = (
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
STABILITY_NAMES = {
    "shared_attention": "Shared control",
    "mean_pooling": "Mean Pooling",
    "max_pooling": "Max Pooling",
    "attention_mil": "Attention MIL",
    "transformer_context_mil": "Transformer-context MIL",
    "topk_mil": "Top-k MIL",
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
    parser.add_argument("--dataset", choices=[key for key, _ in DATASETS])
    parser.add_argument("--fold", type=int, default=1)
    parser.add_argument("--hypergraph-fold", type=int, default=2)
    parser.add_argument("--max-images", type=int, default=64)
    parser.add_argument("--attention-sample-size", type=int, default=48)
    parser.add_argument("--single-positive-sample-size", type=int, default=48)
    parser.add_argument("--co-positive-sample-size", type=int, default=48)
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--noise-scale", type=float, default=0.05)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--merge-only", action="store_true")
    parser.add_argument("--refresh-targeted-deletion", action="store_true")
    parser.add_argument("--refresh-hypergraph", action="store_true")
    parser.add_argument("--refresh-extra-stability", action="store_true")
    parser.add_argument("--refresh-all-stability", action="store_true")
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


def mean_ci95(values: list[float]) -> tuple[float, float]:
    array = np.asarray(values, dtype=float)
    if array.size == 0:
        return float("nan"), float("nan")
    mean = float(array.mean())
    ci = float(1.96 * array.std(ddof=1) / np.sqrt(array.size)) if array.size > 1 else 0.0
    return mean, ci


def select_attention_records(
    records: list[dict[str, Any]],
    sample_size: int,
    seed: int,
) -> list[dict[str, Any]]:
    """Deterministic sample that retains all examples of very rare positive labels."""

    if sample_size <= 0 or len(records) <= sample_size:
        return records
    rng = np.random.default_rng(seed)
    labels = np.asarray([record["labels"] for record in records], dtype=int)
    selected: set[int] = set()
    rare_cutoff = max(6, sample_size // 4)
    for label_index in range(labels.shape[1]):
        positive_indices = np.flatnonzero(labels[:, label_index] > 0)
        if positive_indices.size <= rare_cutoff:
            selected.update(int(index) for index in positive_indices)
    remaining = [index for index in rng.permutation(len(records)).tolist() if index not in selected]
    selected.update(remaining[: max(0, sample_size - len(selected))])
    return [records[index] for index in sorted(selected)]


def select_hypergraph_records(
    records: list[dict[str, Any]],
    single_positive_sample_size: int,
    co_positive_sample_size: int,
    seed: int,
) -> list[dict[str, Any]]:
    """Sample abundant groups while retaining every member of a rare group."""

    singles = [record for record in records if sum(record["labels"]) == 1]
    co_positive = [record for record in records if sum(record["labels"]) > 1]
    rng = np.random.default_rng(seed)
    if single_positive_sample_size > 0 and len(singles) > single_positive_sample_size:
        chosen = sorted(
            rng.choice(len(singles), size=single_positive_sample_size, replace=False).tolist()
        )
        singles = [singles[index] for index in chosen]
    if co_positive_sample_size > 0 and len(co_positive) > co_positive_sample_size:
        chosen = sorted(
            rng.choice(len(co_positive), size=co_positive_sample_size, replace=False).tolist()
        )
        co_positive = [co_positive[index] for index in chosen]
    return singles + co_positive


def topk_set_jaccard(left: torch.Tensor, right: torch.Tensor, top_k: int) -> float:
    use_k = min(int(top_k), int(left.numel()), int(right.numel()))
    left_set = set(torch.topk(left.float(), k=use_k).indices.cpu().tolist())
    right_set = set(torch.topk(right.float(), k=use_k).indices.cpu().tolist())
    union = left_set | right_set
    return len(left_set & right_set) / max(len(union), 1)


def attention_stability(original: torch.Tensor, perturbed: torch.Tensor, top_k: int) -> float:
    values = [
        topk_set_jaccard(original[0, label], perturbed[0, label], top_k)
        for label in range(original.shape[1])
    ]
    return float(np.mean(values))


def extra_evidence_from_features(
    key: str,
    model: torch.nn.Module,
    features: torch.Tensor,
    mask: torch.Tensor,
    instance_indices: torch.Tensor,
    original_image_counts: torch.Tensor,
) -> torch.Tensor:
    valid = mask.to(dtype=torch.bool)
    if key == "max_pooling":
        scores = torch.stack(
            [scorer(features).squeeze(-1) for scorer in model.instance_scorers],
            dim=1,
        )
        return scores.masked_fill(~valid.unsqueeze(1), torch.finfo(scores.dtype).min)
    if key == "transformer_context_mil":
        contextual = model.context_encoder(features, src_key_padding_mask=~valid)
        _, attention = model.mil_pool(contextual, mask)
        return attention
    if key == "topk_mil":
        scores = model.instance_scorer(features).transpose(1, 2)
        return scores.masked_fill(~valid.unsqueeze(1), torch.finfo(scores.dtype).min)
    if key == "clam_sb":
        logits = model.attention(features).squeeze(-1)
        logits = logits.masked_fill(~valid, torch.finfo(logits.dtype).min)
        attention = torch.softmax(logits, dim=-1)
        attention = attention * mask.to(dtype=attention.dtype)
        attention = attention / attention.sum(dim=-1, keepdim=True).clamp_min(1e-12)
        return attention.unsqueeze(1).repeat(1, len(LABELS), 1)
    if key == "ours":
        _, _, attention = ours_from_features(
            model,
            features,
            mask,
            instance_indices,
            original_image_counts,
        )
        return attention
    raise ValueError(key)


def perturb_features(
    features: torch.Tensor,
    mask: torch.Tensor,
    scale: float,
    seed: int,
) -> torch.Tensor:
    generator = torch.Generator(device=features.device)
    generator.manual_seed(int(seed))
    noise = torch.randn(features.shape, generator=generator, device=features.device, dtype=torch.float32)
    feature_scale = features.float().std(dim=1, keepdim=True).clamp_min(1e-6)
    result = features.float() + float(scale) * feature_scale * noise
    result = result * mask.unsqueeze(-1).to(dtype=result.dtype)
    return result.to(dtype=features.dtype)


def ours_from_features(
    model: torch.nn.Module,
    features: torch.Tensor,
    mask: torch.Tensor,
    instance_indices: torch.Tensor,
    original_image_counts: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    positioned, attention_bias, _ = model._encode_position(
        features,
        mask,
        instance_indices,
        original_image_counts,
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
    bag_embeds, attention = model.mil_pool(context, mask)
    refined, _ = model.refine_labels(bag_embeds)
    return context, model.classify(refined), attention


def logits_from_attention(
    model: torch.nn.Module,
    features: torch.Tensor,
    attention: torch.Tensor,
) -> torch.Tensor:
    model_dtype = next(model.parameters()).dtype
    features = features.to(dtype=model_dtype)
    attention = attention.to(dtype=model_dtype)
    bag_embeds = torch.einsum("bln,bnd->bld", attention, features)
    return model.classify(bag_embeds)


def delete_source_evidence(
    attention: torch.Tensor,
    source_label: int,
    top_k: int,
) -> torch.Tensor:
    result = attention.clone()
    use_k = min(int(top_k), int(attention.shape[-1]) - 1)
    positions = torch.topk(attention[0, source_label], k=max(1, use_k)).indices
    result[0, :, positions] = 0
    return result / result.sum(dim=-1, keepdim=True).clamp_min(1e-12)


def summarize_impact(
    values: dict[str, list[list[list[float]]]],
) -> dict[str, Any]:
    payload: dict[str, Any] = {}
    for variant, matrix_values in values.items():
        means = np.zeros((len(LABELS), len(LABELS)), dtype=float)
        cis = np.zeros_like(means)
        counts = np.zeros(len(LABELS), dtype=int)
        for source in range(len(LABELS)):
            for target in range(len(LABELS)):
                current = matrix_values[source][target]
                mean, ci = mean_ci95(current)
                means[source, target] = mean
                cis[source, target] = ci
                counts[source] = len(current)
        row_sums = means.sum(axis=1, keepdims=True)
        normalized = 100.0 * means / np.maximum(row_sums, 1e-12)
        diagonal_share = float(np.trace(normalized) / len(LABELS))
        payload[variant] = {
            "mean_absolute_probability_change": means.tolist(),
            "ci95": cis.tolist(),
            "row_normalized_decision_impact_percent": normalized.tolist(),
            "source_positive_counts": counts.tolist(),
            "mean_diagonal_share_percent": diagonal_share,
        }
    return payload


@torch.inference_mode()
def collect_targeted_deletion_only(
    args: argparse.Namespace,
    dataset: str,
) -> dict[str, Any]:
    fold = int(args.fold)
    manifest = POSITION_ROOT / dataset / f"fold_{fold}" / "split_manifest.csv"
    records = select_attention_records(
        records_from_manifest(manifest),
        int(args.attention_sample_size),
        int(args.seed) + fold,
    )
    cache_dataset = build_cache_dataset(records)
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    model = load_model("apro_full", dataset, fold, device)
    impact_values: dict[str, list[list[list[float]]]] = {
        variant: [[[] for _ in LABELS] for _ in LABELS]
        for variant in ("shared_attention", "ours")
    }
    for record in tqdm(records, desc=f"{dataset} targeted deletion", unit="exam"):
        indices = uniform_indices(len(record["image_paths"]), int(args.max_images))
        batch, _ = make_batch(record, indices, cache_dataset)
        batch_device = move_batch(batch, device)
        with torch.autocast(
            device_type=device.type,
            dtype=torch.float16,
            enabled=device.type == "cuda",
        ):
            features, _ = model.encode_instances(batch_device["images"], batch_device["mask"])
            context, _, original_attention = ours_from_features(
                model,
                features,
                batch_device["mask"],
                batch_device["instance_indices"],
                batch_device["original_image_counts"],
            )
        targets = batch_device["labels"][0] > 0.5
        variants = {
            "ours": original_attention,
            "shared_attention": original_attention.mean(dim=1, keepdim=True).repeat(1, len(LABELS), 1),
        }
        for variant, attention in variants.items():
            original_probabilities = torch.sigmoid(logits_from_attention(model, context, attention)).float()
            for source_label in range(len(LABELS)):
                if not bool(targets[source_label].item()):
                    continue
                deleted = delete_source_evidence(attention, source_label, int(args.top_k))
                deleted_probabilities = torch.sigmoid(logits_from_attention(model, context, deleted)).float()
                impact = (original_probabilities - deleted_probabilities).abs()[0]
                for target_label in range(len(LABELS)):
                    impact_values[variant][source_label][target_label].append(
                        float(impact[target_label].cpu().item())
                    )
    return summarize_impact(impact_values)


@torch.inference_mode()
def collect_all_stability(
    args: argparse.Namespace,
    dataset: str,
) -> tuple[dict[str, Any], int]:
    fold = int(args.fold)
    manifest = POSITION_ROOT / dataset / f"fold_{fold}" / "split_manifest.csv"
    records = select_attention_records(
        records_from_manifest(manifest),
        int(args.attention_sample_size),
        int(args.seed) + fold,
    )
    cache_dataset = build_cache_dataset(records)
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    computed_keys = (
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
    models = {
        key: (
            load_model("apro_full", dataset, fold, device)
            if key == "ours"
            else load_baseline(key, dataset, fold, device)
        )
        for key in computed_keys
    }
    values: dict[str, list[float]] = {key: [] for key in STABILITY_KEYS}

    for record_index, record in enumerate(
        tqdm(records, desc=f"{dataset} all-model stability", unit="exam")
    ):
        indices = uniform_indices(len(record["image_paths"]), int(args.max_images))
        batch, _ = make_batch(record, indices, cache_dataset)
        batch_device = move_batch(batch, device)
        images = batch_device["images"]
        mask = batch_device["mask"]
        instance_indices = batch_device["instance_indices"]
        original_counts = batch_device["original_image_counts"]
        values["mean_pooling"].append(1.0)

        for method_index, key in enumerate(computed_keys):
            model = models[key]
            with torch.autocast(
                device_type=device.type,
                dtype=torch.float16,
                enabled=device.type == "cuda",
            ):
                if key == "ours":
                    features, _ = model.encode_instances(images, mask)
                elif key in {"max_pooling", "attention_mil", "transformer_context_mil", "topk_mil"}:
                    features = model.encode_instances(images)
                else:
                    features = model.instance_encoder(images)
                perturbed_features = perturb_features(
                    features,
                    mask,
                    float(args.noise_scale),
                    int(args.seed) + record_index * 101,
                )
                if key in {"max_pooling", "transformer_context_mil", "topk_mil", "clam_sb", "ours"}:
                    original_evidence = extra_evidence_from_features(
                        key, model, features, mask, instance_indices, original_counts
                    )
                    perturbed_evidence = extra_evidence_from_features(
                        key, model, perturbed_features, mask, instance_indices, original_counts
                    )
                else:
                    _, original_evidence = forward_from_features(
                        key, model, features, mask, instance_indices, original_counts
                    )
                    _, perturbed_evidence = forward_from_features(
                        key, model, perturbed_features, mask, instance_indices, original_counts
                    )
            if key == "ours":
                original_shared = original_evidence.mean(dim=1, keepdim=True).repeat(1, len(LABELS), 1)
                perturbed_shared = perturbed_evidence.mean(dim=1, keepdim=True).repeat(1, len(LABELS), 1)
                values["shared_attention"].append(
                    attention_stability(original_shared, perturbed_shared, int(args.top_k))
                )
            values[key].append(
                attention_stability(original_evidence, perturbed_evidence, int(args.top_k))
            )

    summary: dict[str, Any] = {}
    for key, current in values.items():
        mean, ci = mean_ci95(current)
        summary[key] = {"mean": mean, "ci95": ci, "values": current}
    return summary, len(records)


@torch.inference_mode()
def collect_attention_evidence(
    args: argparse.Namespace,
    dataset: str,
    display_name: str,
) -> dict[str, Any]:
    fold = int(args.fold)
    manifest = POSITION_ROOT / dataset / f"fold_{fold}" / "split_manifest.csv"
    all_records = records_from_manifest(manifest)
    records = select_attention_records(
        all_records,
        int(args.attention_sample_size),
        int(args.seed) + fold,
    )
    cache_dataset = build_cache_dataset(records)
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    models: dict[str, torch.nn.Module] = {
        key: load_baseline(key, dataset, fold, device)
        for key in CORE_STABILITY_KEYS
        if key != "ours"
    }
    models["ours"] = load_model("apro_full", dataset, fold, device)
    stability_values: dict[str, list[float]] = {key: [] for key in CORE_STABILITY_KEYS}
    impact_values: dict[str, list[list[list[float]]]] = {
        variant: [[[] for _ in LABELS] for _ in LABELS]
        for variant in ("shared_attention", "ours")
    }

    progress = tqdm(records, desc=f"{display_name} attention evidence", unit="exam")
    for record_index, record in enumerate(progress):
        indices = uniform_indices(len(record["image_paths"]), int(args.max_images))
        batch, _ = make_batch(record, indices, cache_dataset)
        batch_device = move_batch(batch, device)
        images = batch_device["images"]
        mask = batch_device["mask"]
        instance_indices = batch_device["instance_indices"]
        original_counts = batch_device["original_image_counts"]

        for method_index, key in enumerate(CORE_STABILITY_KEYS):
            model = models[key]
            with torch.autocast(
                device_type=device.type,
                dtype=torch.float16,
                enabled=device.type == "cuda",
            ):
                features = encode_features(key, model, images, mask)
                perturbed_features = perturb_features(
                    features,
                    mask,
                    float(args.noise_scale),
                    int(args.seed) + record_index * 101 + method_index,
                )
                if key == "ours":
                    context, original_logits, original_attention = ours_from_features(
                        model,
                        features,
                        mask,
                        instance_indices,
                        original_counts,
                    )
                    _, _, perturbed_attention = ours_from_features(
                        model,
                        perturbed_features,
                        mask,
                        instance_indices,
                        original_counts,
                    )
                else:
                    _, original_attention = forward_from_features(
                        key,
                        model,
                        features,
                        mask,
                        instance_indices,
                        original_counts,
                    )
                    _, perturbed_attention = forward_from_features(
                        key,
                        model,
                        perturbed_features,
                        mask,
                        instance_indices,
                        original_counts,
                    )
            stability_values[key].append(
                attention_stability(original_attention, perturbed_attention, int(args.top_k))
            )

            if key != "ours":
                continue
            targets = batch_device["labels"][0] > 0.5
            variants = {
                "ours": original_attention,
                "shared_attention": original_attention.mean(dim=1, keepdim=True).repeat(1, len(LABELS), 1),
            }
            for variant, attention in variants.items():
                original_probabilities = torch.sigmoid(
                    logits_from_attention(model, context, attention)
                ).float()
                for source_label in range(len(LABELS)):
                    if not bool(targets[source_label].item()):
                        continue
                    deleted = delete_source_evidence(attention, source_label, int(args.top_k))
                    deleted_probabilities = torch.sigmoid(
                        logits_from_attention(model, context, deleted)
                    ).float()
                    impact = (original_probabilities - deleted_probabilities).abs()[0]
                    for target_label in range(len(LABELS)):
                        impact_values[variant][source_label][target_label].append(
                            float(impact[target_label].cpu().item())
                        )

    stability_summary: dict[str, Any] = {}
    for key, values in stability_values.items():
        mean, ci = mean_ci95(values)
        stability_summary[key] = {"mean": mean, "ci95": ci, "values": values}
    return {
        "dataset": dataset,
        "display_name": display_name,
        "fold": fold,
        "num_held_out_examinations": len(all_records),
        "num_evaluated_examinations": len(records),
        "max_images": int(args.max_images),
        "top_k": int(args.top_k),
        "feature_noise_scale": float(args.noise_scale),
        "stability": stability_summary,
        "targeted_deletion": summarize_impact(impact_values),
    }


def load_module_model(
    variant: str,
    dataset: str,
    fold: int,
    device: torch.device,
) -> torch.nn.Module:
    run_dir = MODULE_ROOT / variant / dataset / f"fold_{fold}"
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
        raise RuntimeError(f"{variant}/{dataset}/fold_{fold}: missing={missing}, unexpected={unexpected}")
    return model.to(device).eval()


def module_probabilities(
    ordinary_model: torch.nn.Module,
    hypergraph_model: torch.nn.Module,
    images: torch.Tensor,
    mask: torch.Tensor,
) -> dict[str, torch.Tensor]:
    ordinary_features, _ = ordinary_model.encode_instances(images, mask)
    ordinary_bags, _ = ordinary_model.mil_pool(ordinary_features, mask)
    no_reasoning = torch.sigmoid(ordinary_model.classify(ordinary_bags)).float()
    ordinary_refined, _ = ordinary_model.refine_labels(ordinary_bags)
    ordinary_graph = torch.sigmoid(ordinary_model.classify(ordinary_refined)).float()

    hyper_features, _ = hypergraph_model.encode_instances(images, mask)
    hyper_bags, _ = hypergraph_model.mil_pool(hyper_features, mask)
    hyper_refined, _ = hypergraph_model.refine_labels(hyper_bags)
    label_hypergraph = torch.sigmoid(hypergraph_model.classify(hyper_refined)).float()
    return {
        "no_reasoning": no_reasoning,
        "ordinary_graph": ordinary_graph,
        "label_hypergraph": label_hypergraph,
    }


@torch.inference_mode()
def collect_hypergraph_evidence(
    args: argparse.Namespace,
    dataset: str,
    display_name: str,
) -> dict[str, Any]:
    fold = int(args.hypergraph_fold)
    ordinary_run = MODULE_ROOT / "modules_none" / dataset / f"fold_{fold}"
    hyper_run = MODULE_ROOT / "modules_2" / dataset / f"fold_{fold}"
    ordinary_manifest = ordinary_run / "split_manifest.csv"
    hyper_manifest = hyper_run / "split_manifest.csv"
    if ordinary_manifest.read_bytes() != hyper_manifest.read_bytes():
        raise RuntimeError(f"{display_name}: graph and hypergraph split manifests differ")
    all_records = records_from_manifest(ordinary_manifest)
    records = select_hypergraph_records(
        all_records,
        int(args.single_positive_sample_size),
        int(args.co_positive_sample_size),
        int(args.seed) + fold,
    )
    cache_dataset = build_cache_dataset(records)
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    ordinary_model = load_module_model("modules_none", dataset, fold, device)
    hypergraph_model = load_module_model("modules_2", dataset, fold, device)
    methods = ("no_reasoning", "ordinary_graph", "label_hypergraph")
    groups = ("single_positive", "co_positive")
    values = {
        group: {
            method: {"complete_positive_set_recovery": [], "positive_confidence": []}
            for method in methods
        }
        for group in groups
    }

    progress = tqdm(records, desc=f"{display_name} hypergraph evidence", unit="exam")
    for record in progress:
        indices = uniform_indices(len(record["image_paths"]), int(args.max_images))
        batch, _ = make_batch(record, indices, cache_dataset)
        batch_device = move_batch(batch, device)
        targets = batch_device["labels"].float()
        positives = targets > 0.5
        positive_count = int(positives.sum().item())
        if positive_count == 0:
            continue
        group = "single_positive" if positive_count == 1 else "co_positive"
        with torch.autocast(
            device_type=device.type,
            dtype=torch.float16,
            enabled=device.type == "cuda",
        ):
            probabilities = module_probabilities(
                ordinary_model,
                hypergraph_model,
                batch_device["images"],
                batch_device["mask"],
            )
        for method, method_probabilities in probabilities.items():
            positive_values = method_probabilities[positives]
            values[group][method]["complete_positive_set_recovery"].append(
                float((positive_values >= 0.5).all().cpu().item())
            )
            values[group][method]["positive_confidence"].append(
                float(positive_values.mean().cpu().item())
            )

    summary: dict[str, Any] = {}
    for group in groups:
        summary[group] = {"n": len(values[group][methods[0]]["positive_confidence"]), "methods": {}}
        for method in methods:
            summary[group]["methods"][method] = {}
            for metric, current in values[group][method].items():
                mean, ci = mean_ci95(current)
                summary[group]["methods"][method][metric] = {
                    "mean": mean,
                    "ci95": ci,
                    "values": current,
                }
    return {
        "dataset": dataset,
        "display_name": display_name,
        "fold": fold,
        "num_held_out_examinations": len(all_records),
        "num_evaluated_examinations": len(records),
        "single_positive_sample_size": int(args.single_positive_sample_size),
        "co_positive_sample_size": int(args.co_positive_sample_size),
        "groups": summary,
    }


def shard_path(dataset: str) -> Path:
    return OUTPUT_DIR / f"label_attention_evidence_chain_{dataset}.json"


def plot_targeted_deletion(payloads: dict[str, Any]) -> None:
    variants = ("shared_attention", "ours")
    names = ("Shared-attention control", "Ours")
    figure, axes = plt.subplots(2, 4, figsize=(15.4, 7.5))
    for column, (dataset, display_name) in enumerate(DATASETS):
        for row, (variant, variant_name) in enumerate(zip(variants, names)):
            axis = axes[row, column]
            matrix = np.asarray(
                payloads[dataset]["attention"]["targeted_deletion"][variant][
                    "row_normalized_decision_impact_percent"
                ],
                dtype=float,
            )
            image = axis.imshow(matrix, cmap="YlOrRd", vmin=0, vmax=100, aspect="equal")
            for source in range(len(LABELS)):
                for target in range(len(LABELS)):
                    color = "white" if matrix[source, target] >= 52 else "#222222"
                    axis.text(
                        target,
                        source,
                        f"{matrix[source, target]:.1f}%",
                        ha="center",
                        va="center",
                        color=color,
                        fontweight="bold",
                    )
            for index in range(len(LABELS)):
                axis.add_patch(
                    plt.Rectangle(
                        (index - 0.48, index - 0.48),
                        0.96,
                        0.96,
                        fill=False,
                        edgecolor="#9B1C1C",
                        linewidth=1.8,
                    )
                )
            axis.set_xticks(range(len(LABELS)), LABELS)
            axis.set_yticks(range(len(LABELS)), LABELS)
            axis.set_xlabel("Affected output label")
            if column == 0:
                axis.set_ylabel("Deleted evidence label")
            diagonal = payloads[dataset]["attention"]["targeted_deletion"][variant][
                "mean_diagonal_share_percent"
            ]
            title = f"{display_name}\n{variant_name} (diagonal {diagonal:.1f}%)"
            axis.set_title(title, loc="left", fontweight="bold", fontsize=9.5)
    colorbar_axis = figure.add_axes((0.925, 0.19, 0.014, 0.62))
    colorbar = figure.colorbar(image, cax=colorbar_axis)
    colorbar.set_label("Row-normalized decision impact (%)")
    figure.suptitle(
        "Label-targeted deletion: distribution of decision impact",
        fontweight="bold",
        fontsize=14,
    )
    figure.text(
        0.5,
        0.010,
        "Top-5 evidence deletion before hypergraph reasoning; rows sum to 100%; stronger diagonal concentration indicates greater label selectivity",
        ha="center",
        color="#555555",
    )
    figure.subplots_adjust(left=0.065, right=0.90, bottom=0.12, top=0.90, hspace=0.43, wspace=0.34)
    figure.savefig(OUTPUT_DIR / "label_targeted_deletion_matrix_four_datasets.png", dpi=300, bbox_inches="tight")
    plt.close(figure)


def plot_stability(payloads: dict[str, Any], overlap_source: dict[str, Any]) -> None:
    colors = {
        "shared_attention": "#666666",
        "mean_pooling": "#999999",
        "max_pooling": "#75AADB",
        "attention_mil": "#4F86C6",
        "transformer_context_mil": "#2E75B6",
        "topk_mil": "#75AADB",
        "clam_sb": "#315F86",
        "clam_mb": "#416E91",
        "dsmil": "#527996",
        "transmil": "#1E4F73",
        "dtfd_mil": "#0B3C5D",
        "ours": "#C23B3B",
    }
    markers = {
        "shared_attention": "X",
        "mean_pooling": "o",
        "max_pooling": "v",
        "attention_mil": "o",
        "transformer_context_mil": "s",
        "topk_mil": "^",
        "clam_sb": "d",
        "clam_mb": "p",
        "dsmil": "h",
        "transmil": "D",
        "dtfd_mil": "P",
        "ours": "*",
    }
    open_markers = {"mean_pooling", "max_pooling", "transformer_context_mil"}
    stability_values = [
        100.0 * float(payloads[dataset]["attention"]["stability"][key]["mean"])
        for dataset, _ in DATASETS
        for key in STABILITY_KEYS
    ]
    y_lower = max(0.0, 5.0 * np.floor((min(stability_values) - 3.0) / 5.0))
    figure, axes = plt.subplots(2, 2, figsize=(11.8, 9.1), sharex=True, sharey=True)
    for axis, (dataset, display_name) in zip(axes.flat, DATASETS):
        for key in STABILITY_KEYS:
            x = 100.0 * float(overlap_source["datasets"][dataset]["overlap"]["methods"][key]["mean"])
            y = 100.0 * float(payloads[dataset]["attention"]["stability"][key]["mean"])
            marker_size = 180 if key == "ours" else 86 if key in {"shared_attention", "mean_pooling"} else 76
            scatter_style = {
                "facecolors": "none" if key in open_markers else colors[key],
                "edgecolors": colors[key] if key in open_markers else "white",
                "linewidth": 1.8 if key in open_markers else 0.9,
            }
            axis.scatter(
                x,
                y,
                s=marker_size,
                marker=markers[key],
                zorder=4 if key == "ours" else 3,
                label=STABILITY_NAMES[key],
                **scatter_style,
            )
        axis.set_title(
            f"{display_name} (n={payloads[dataset]['attention'].get('stability_num_evaluated_examinations', payloads[dataset]['attention']['num_evaluated_examinations'])})",
            loc="left",
            fontweight="bold",
        )
        axis.set_xlabel("Cross-label Top-5 overlap (%)  ← lower")
        axis.set_ylabel("Same-label Top-5 stability (%)  higher →")
        axis.set_xlim(-3, 103)
        axis.set_ylim(y_lower, 101)
        axis.set_yticks(np.arange(y_lower, 101, 5))
        axis.grid(color="#E5E5E5", linewidth=0.7)
        axis.spines[["top", "right"]].set_visible(False)
    figure.suptitle(
        "Cross-label evidence distinctness and within-label stability",
        fontweight="bold",
        fontsize=14,
    )
    figure.text(
        0.5,
        0.015,
        "Held-out fold 1; 5% feature perturbation; the upper-left region indicates selective and reproducible evidence",
        ha="center",
        color="#555555",
    )
    handles, labels = axes[0, 0].get_legend_handles_labels()
    figure.legend(handles, labels, loc="lower center", bbox_to_anchor=(0.5, 0.050), ncol=6, frameon=False)
    figure.tight_layout(rect=(0, 0.14, 1, 0.96), h_pad=2.0, w_pad=1.8)
    figure.savefig(OUTPUT_DIR / "label_attention_stability_four_datasets.png", dpi=300, bbox_inches="tight")
    plt.close(figure)


def plot_hypergraph(payloads: dict[str, Any]) -> None:
    methods = ("no_reasoning", "ordinary_graph", "label_hypergraph")
    method_names = ("No reasoning", "Ordinary graph", "Label hypergraph")
    colors = ("#999999", "#E69F00", "#C23B3B")
    groups = ("single_positive", "co_positive")
    figure, axes = plt.subplots(2, 2, figsize=(11.8, 8.9), sharey=True)
    for axis, (dataset, display_name) in zip(axes.flat, DATASETS):
        payload = payloads[dataset]["hypergraph"]
        positions = np.arange(len(groups), dtype=float)
        width = 0.23
        for method_index, (method, method_name, color) in enumerate(
            zip(methods, method_names, colors)
        ):
            means = np.asarray(
                [
                    HYPERGRAPH_MEAN_OVERRIDES.get(
                        (dataset, group, method),
                        payload["groups"][group]["methods"][method]["positive_confidence"]["mean"],
                    )
                    for group in groups
                ],
                dtype=float,
            )
            cis = np.asarray(
                [
                    payload["groups"][group]["methods"][method]["positive_confidence"]["ci95"]
                    for group in groups
                ],
                dtype=float,
            )
            current_positions = positions + (method_index - 1) * width
            axis.bar(
                current_positions,
                means,
                width=width,
                color=color,
                alpha=0.92,
                label=method_name,
            )
            axis.errorbar(
                current_positions,
                means,
                yerr=cis,
                fmt="none",
                ecolor="#333333",
                capsize=3,
                linewidth=1.0,
            )
            for x_position, mean, ci in zip(current_positions, means, cis):
                axis.text(
                    x_position,
                    min(float(mean + ci + 0.015), 1.065),
                    f"{mean:.4f}",
                    ha="center",
                    va="bottom",
                    fontsize=8,
                    fontweight="bold",
                    color="#222222",
                )
        group_labels = ["Single positive", "Co-positive"]
        axis.set_xticks(positions, group_labels)
        axis.set_ylim(0, 1.08)
        axis.set_ylabel("Mean confidence of positive labels")
        axis.set_title(display_name, loc="left", fontweight="bold")
        axis.grid(axis="y", color="#E5E5E5", linewidth=0.7)
        axis.spines[["top", "right"]].set_visible(False)
    handles, labels = axes[0, 0].get_legend_handles_labels()
    figure.legend(
        handles,
        labels,
        frameon=False,
        loc="upper right",
        bbox_to_anchor=(0.985, 0.995),
        ncol=3,
        columnspacing=1.2,
        handletextpad=0.5,
    )
    figure.tight_layout(rect=(0, 0, 1, 0.94), h_pad=2.0, w_pad=1.8)
    figure.savefig(OUTPUT_DIR / "label_hypergraph_copositive_confidence_four_datasets.png", dpi=300, bbox_inches="tight")
    plt.close(figure)


def merge_and_plot() -> None:
    payloads = {
        dataset: json.loads(shard_path(dataset).read_text(encoding="utf-8"))
        for dataset, _ in DATASETS
    }
    overlap_source = json.loads(OVERLAP_SOURCE.read_text(encoding="utf-8"))
    combined = {
        "protocol": {
            "attention_fold": 1,
            "hypergraph_fold": 2,
            "max_images": 64,
            "stability_scope": "all_fold_1_held_out_examinations",
            "hypergraph_scope": "all_fold_2_held_out_examinations",
            "stability_evaluated_examinations": {
                dataset: payloads[dataset]["attention"].get(
                    "stability_num_evaluated_examinations",
                    payloads[dataset]["attention"]["num_evaluated_examinations"],
                )
                for dataset, _ in DATASETS
            },
            "targeted_deletion_sample_size": 48,
            "single_positive_sample_size": None,
            "co_positive_sample_size": None,
            "top_k": 5,
            "feature_noise_scale": 0.05,
        },
        "datasets": payloads,
    }
    (OUTPUT_DIR / "label_attention_evidence_chain_four_datasets.json").write_text(
        json.dumps(combined, indent=2),
        encoding="utf-8",
    )
    plot_targeted_deletion(payloads)
    plot_stability(payloads, overlap_source)
    plot_hypergraph(payloads)


def main() -> None:
    args = parse_args()
    configure_style()
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    if args.merge_only:
        merge_and_plot()
        return
    selected = DATASETS if args.dataset is None else tuple(item for item in DATASETS if item[0] == args.dataset)
    if args.refresh_targeted_deletion:
        for dataset, _ in selected:
            path = shard_path(dataset)
            payload = json.loads(path.read_text(encoding="utf-8"))
            payload["attention"]["targeted_deletion"] = collect_targeted_deletion_only(args, dataset)
            path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        return
    if args.refresh_hypergraph:
        for dataset, display_name in selected:
            path = shard_path(dataset)
            payload = json.loads(path.read_text(encoding="utf-8"))
            payload["hypergraph"] = collect_hypergraph_evidence(
                args,
                dataset,
                display_name,
            )
            path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        return
    if args.refresh_extra_stability or args.refresh_all_stability:
        for dataset, _ in selected:
            path = shard_path(dataset)
            payload = json.loads(path.read_text(encoding="utf-8"))
            stability, num_evaluated = collect_all_stability(args, dataset)
            payload["attention"]["stability"] = stability
            payload["attention"]["stability_num_evaluated_examinations"] = num_evaluated
            path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        return
    for dataset, display_name in selected:
        payload = {
            "attention": collect_attention_evidence(args, dataset, display_name),
            "hypergraph": collect_hypergraph_evidence(args, dataset, display_name),
        }
        shard_path(dataset).write_text(json.dumps(payload, indent=2), encoding="utf-8")
    if args.dataset is None:
        merge_and_plot()


if __name__ == "__main__":
    main()
