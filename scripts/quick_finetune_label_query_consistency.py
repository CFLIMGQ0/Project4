#!/usr/bin/env python3
"""Fast cached-feature fine-tuning for label-query consistency on one dataset/fold."""

from __future__ import annotations

import argparse
import copy
import json
import os
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
import yaml
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parents[1]
PROJECT_ROOT = ROOT.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.plot_wle_label_query_retrieval_swap import (
    LABEL_DISPLAY_NAMES,
    classify_with_text,
    resolve_device,
)
from scripts.quick_finetune_wle_label_query_constraint import (
    cache_embeddings,
    evaluate,
    forward_cached,
    load_split_records,
    select_records,
)
from scripts.task3_tsne import build_model, load_masked_records
from training.losses import AsymmetricLossMultiLabel


DATASET_KEYS = (
    "regular_white_light",
    "chromoscopic",
    "surgical",
    "ultrasound",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", choices=DATASET_KEYS, required=True)
    parser.add_argument("--fold", type=int, default=1)
    parser.add_argument("--train-sample-size", type=int, default=0)
    parser.add_argument("--test-sample-size", type=int, default=0)
    parser.add_argument("--epochs", type=int, default=15)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--learning-rate", type=float, default=2e-4)
    parser.add_argument("--consistency-weight", type=float, default=0.50)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=PROJECT_ROOT / "outputs/quick_finetune/label_query_consistency",
    )
    return parser.parse_args()


def maybe_select(
    records: list[dict[str, object]], sample_size: int, seed: int
) -> list[dict[str, object]]:
    if int(sample_size) <= 0 or int(sample_size) >= len(records):
        return records
    return select_records(records, sample_size, seed)


def consistency_loss(
    model: torch.nn.Module,
    label_embeds: torch.Tensor,
    retrieved: torch.Tensor,
    active_values: torch.Tensor,
    labels: torch.Tensor,
    correct_logits: torch.Tensor,
) -> torch.Tensor:
    active = active_values.view(-1, 1, 1).to(dtype=retrieved.dtype)
    consistency_terms = []
    for permutation in ((1, 2, 0), (2, 0, 1)):
        replaced = retrieved[:, permutation, :]
        replaced_gates = torch.sigmoid(
            model.text_gate(torch.cat([label_embeds, replaced], dim=-1))
        ) * active
        replaced_logits = model.classify(
            label_embeds + replaced_gates * replaced
        )
        replaced_targets = labels[:, permutation]
        valid_pair = (
            (labels > 0.5)
            & (replaced_targets < 0.5)
            & active_values.view(-1, 1).bool()
        )
        if valid_pair.any():
            positive_term = F.softplus(-correct_logits[valid_pair])
            replacement_term = F.softplus(replaced_logits[valid_pair])
            consistency_terms.append(0.5 * (positive_term + replacement_term))
    if not consistency_terms:
        return torch.zeros((), device=labels.device)
    return torch.cat(consistency_terms).mean()


def aggregate_positive_conditions(
    evaluation: dict[str, object],
) -> dict[str, float | int]:
    values = evaluation["values_by_label"]
    result: dict[str, float | int] = {}
    names = ("correct", "cross_label", "pooled")
    for condition_index, name in enumerate(names):
        concatenated = np.concatenate(
            [values[label_index][condition_index] for label_index in range(len(values))]
        )
        result[name] = float(concatenated.mean())
        if condition_index == 0:
            result["positive_label_observations"] = int(concatenated.size)
    return result


def serializable_evaluation(evaluation: dict[str, object]) -> dict[str, object]:
    return {
        key: value
        for key, value in evaluation.items()
        if key != "values_by_label"
    }


def main() -> None:
    args = parse_args()
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    os.environ.setdefault("PROJECT4_DISABLE_DISK_CACHE_WRITE", "1")
    device = resolve_device(args.device)

    cfg = yaml.safe_load(
        (ROOT / "configs/task3/t3_main_model.yaml").read_text(encoding="utf-8")
    )
    train_root = PROJECT_ROOT / "outputs/train_runs/task3/t3_main_model"
    fold_dir = train_root / args.dataset / f"fold_{args.fold}"
    record_map = load_masked_records(train_root / "records_cache.json")
    manifest = fold_dir / "split_manifest.csv"
    train_records = maybe_select(
        load_split_records(manifest, record_map, "train"),
        args.train_sample_size,
        args.seed,
    )
    test_records = maybe_select(
        load_split_records(manifest, record_map, "test"),
        args.test_sample_size,
        args.seed,
    )

    model = build_model(cfg, fold_dir / "checkpoints/best_macro_f1.ckpt", device)
    from scripts.plot_wle_label_query_retrieval_swap import build_loader

    train_cache = cache_embeddings(
        model,
        build_loader(train_records, cfg, seed=args.seed, num_workers=args.num_workers),
        device,
        f"Cache {args.dataset} train",
    )
    test_cache = cache_embeddings(
        model,
        build_loader(test_records, cfg, seed=args.seed, num_workers=args.num_workers),
        device,
        f"Cache {args.dataset} test",
    )
    before = evaluate(model, test_cache, device, args.batch_size)

    for parameter in model.parameters():
        parameter.requires_grad_(False)
    trainable_modules = (model.text_cross_attn,)
    for module in trainable_modules:
        for parameter in module.parameters():
            parameter.requires_grad_(True)
    initial_states = {
        "text_cross_attn": copy.deepcopy(model.text_cross_attn.state_dict()),
        "text_gate": copy.deepcopy(model.text_gate.state_dict()),
        "classifiers": copy.deepcopy(model.classifiers.state_dict()),
    }
    trainable_parameters = [
        parameter
        for module in trainable_modules
        for parameter in module.parameters()
    ]
    model.label_query_bias.requires_grad_(True)
    trainable_parameters.append(model.label_query_bias)
    optimizer = torch.optim.AdamW(
        trainable_parameters,
        lr=float(args.learning_rate),
        weight_decay=1e-4,
    )
    criterion = AsymmetricLossMultiLabel()
    generator = torch.Generator().manual_seed(args.seed)
    train_loader = DataLoader(
        train_cache,
        batch_size=args.batch_size,
        shuffle=True,
        generator=generator,
    )

    history: list[dict[str, float | int]] = []
    for epoch in range(1, args.epochs + 1):
        model.text_cross_attn.train()
        model.text_gate.eval()
        model.classifiers.eval()
        totals = {"classification": 0.0, "consistency": 0.0, "batches": 0}
        for label_embeds, tokens, token_mask, _, active_values, labels in train_loader:
            label_embeds = label_embeds.to(device)
            tokens = tokens.to(device)
            token_mask = token_mask.to(device)
            active_values = active_values.to(device)
            labels = labels.to(device)
            logits, retrieved, _ = forward_cached(
                model, label_embeds, tokens, token_mask, active_values
            )
            classification = criterion(logits, labels)
            consistency = consistency_loss(
                model,
                label_embeds,
                retrieved,
                active_values,
                labels,
                logits,
            )
            loss = classification + float(args.consistency_weight) * consistency
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(trainable_parameters, max_norm=1.0)
            optimizer.step()
            totals["classification"] += float(classification.detach().item())
            totals["consistency"] += float(consistency.detach().item())
            totals["batches"] += 1
        row = {
            "epoch": epoch,
            "classification_loss": totals["classification"] / max(totals["batches"], 1),
            "consistency_loss": totals["consistency"] / max(totals["batches"], 1),
        }
        history.append(row)
        if epoch == 1 or epoch % 5 == 0 or epoch == args.epochs:
            print(json.dumps(row, ensure_ascii=False), flush=True)

    model.eval()
    after = evaluate(model, test_cache, device, args.batch_size)
    output_dir = args.output_dir.resolve() / args.dataset / f"fold_{args.fold}"
    output_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path = output_dir / "label_query_consistency.ckpt"
    torch.save(
        {
            "source_checkpoint": str(fold_dir / "checkpoints/best_macro_f1.ckpt"),
            "model_updates": {
                "label_query_bias": model.label_query_bias.detach().cpu(),
                "text_cross_attn": model.text_cross_attn.state_dict(),
                "text_gate": model.text_gate.state_dict(),
                "classifiers": model.classifiers.state_dict(),
            },
            "initial_states": initial_states,
            "settings": vars(args),
        },
        checkpoint_path,
    )
    stats = {
        "dataset": args.dataset,
        "fold": args.fold,
        "train_samples": len(train_records),
        "test_samples": len(test_records),
        "settings": {
            "epochs": args.epochs,
            "learning_rate": args.learning_rate,
            "consistency_weight": args.consistency_weight,
            "seed": args.seed,
        },
        "before_overall": aggregate_positive_conditions(before),
        "after_overall": aggregate_positive_conditions(after),
        "before": serializable_evaluation(before),
        "after": serializable_evaluation(after),
        "history": history,
    }
    (output_dir / "stats.json").write_text(
        json.dumps(stats, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    np.savez_compressed(
        output_dir / "positive_conditions.npz",
        **{
            f"label_{label_index}_{condition_name}": after["values_by_label"][label_index][condition_index]
            for label_index in range(len(LABEL_DISPLAY_NAMES))
            for condition_index, condition_name in enumerate(("correct", "cross_label", "pooled"))
        },
    )
    print(json.dumps(stats, ensure_ascii=False, indent=2), flush=True)
    print(output_dir, flush=True)


if __name__ == "__main__":
    main()
