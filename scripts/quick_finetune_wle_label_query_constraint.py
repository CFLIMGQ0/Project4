#!/usr/bin/env python3
"""Quickly fine-tune WLE label-query attention with a weak cross-modal alignment margin."""

from __future__ import annotations

import argparse
import copy
import csv
import json
import os
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
import yaml
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm

ROOT = Path(__file__).resolve().parents[1]
PROJECT_ROOT = ROOT.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.plot_wle_label_query_retrieval_swap import (
    LABEL_DISPLAY_NAMES,
    build_loader,
    classify_with_text,
    plot,
    resolve_device,
    safe_text_inputs,
)
from scripts.task3_tsne import build_model, load_masked_records
from training.losses import AsymmetricLossMultiLabel


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fold", type=int, default=1)
    parser.add_argument("--train-sample-size", type=int, default=192)
    parser.add_argument("--test-sample-size", type=int, default=48)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--learning-rate", type=float, default=5e-4)
    parser.add_argument("--alignment-margin", type=float, default=0.05)
    parser.add_argument("--alignment-weight", type=float, default=0.5)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument(
        "--output",
        type=Path,
        default=PROJECT_ROOT / "temp_img/wle_label_query_retrieval_swap_finetuned_sample.png",
    )
    parser.add_argument(
        "--checkpoint-output",
        type=Path,
        default=PROJECT_ROOT / "outputs/quick_finetune/wle_fold1_label_query_constraint.ckpt",
    )
    parser.add_argument(
        "--stats-output",
        type=Path,
        default=PROJECT_ROOT / "temp_img/wle_label_query_constraint_stats.json",
    )
    return parser.parse_args()


def load_split_records(
    manifest_path: Path,
    record_map: dict[str, dict[str, object]],
    split: str,
) -> list[dict[str, object]]:
    records: list[dict[str, object]] = []
    with manifest_path.open("r", encoding="utf-8-sig", newline="") as file:
        for row in csv.DictReader(file):
            if str(row.get("split", "")).strip().lower() != split:
                continue
            exam_dir = str(row.get("exam_dir", ""))
            if exam_dir not in record_map:
                raise KeyError(f"Manifest examination not found in records cache: {exam_dir}")
            records.append(record_map[exam_dir])
    if not records:
        raise RuntimeError(f"No {split} records found in {manifest_path}")
    return records


def select_records(
    records: list[dict[str, object]],
    sample_size: int,
    seed: int,
) -> list[dict[str, object]]:
    count = min(len(records), int(sample_size))
    rng = np.random.default_rng(seed)
    indices = sorted(rng.choice(len(records), size=count, replace=False).tolist())
    return [records[index] for index in indices]


def cache_embeddings(
    model: torch.nn.Module,
    loader: DataLoader,
    device: torch.device,
    description: str,
) -> TensorDataset:
    label_embeddings: list[torch.Tensor] = []
    text_tokens: list[torch.Tensor] = []
    text_masks: list[torch.Tensor] = []
    text_pooled: list[torch.Tensor] = []
    text_active: list[torch.Tensor] = []
    labels: list[torch.Tensor] = []
    model.eval()
    with torch.inference_mode():
        for batch in tqdm(loader, desc=description, unit="exam"):
            images = batch["images"].to(device, non_blocking=True)
            image_mask = batch["mask"].to(device, non_blocking=True)
            watch_ids = batch["watch_token_ids"].to(device, non_blocking=True)
            watch_mask = batch["watch_token_mask"].to(device, non_blocking=True)
            _, current_labels, _, _ = model.encode_long_mil(images, image_mask)
            current_tokens, current_mask, current_pooled, current_active = model.text_encoder(
                watch_ids,
                watch_mask,
                batch_size=images.shape[0],
                device=device,
            )
            label_embeddings.append(current_labels.float().cpu())
            text_tokens.append(current_tokens.float().cpu())
            text_masks.append(current_mask.bool().cpu())
            text_pooled.append(current_pooled.float().cpu())
            text_active.append(current_active.bool().cpu())
            labels.append(batch["labels"].float().cpu())
    return TensorDataset(
        torch.cat(label_embeddings),
        torch.cat(text_tokens),
        torch.cat(text_masks),
        torch.cat(text_pooled),
        torch.cat(text_active),
        torch.cat(labels),
    )


def label_alignment_margin_loss(
    label_embeds: torch.Tensor,
    retrieved: torch.Tensor,
    labels: torch.Tensor,
    margin: float,
) -> torch.Tensor:
    """Align each positive visual label with its own retrieval over negative-label retrievals."""
    visual = F.normalize(label_embeds, dim=-1)
    textual = F.normalize(retrieved, dim=-1)
    similarities = torch.einsum("bld,bkd->blk", visual, textual)
    losses: list[torch.Tensor] = []
    for label_index in range(labels.shape[1]):
        positive = labels[:, label_index] > 0.5
        for other_index in range(labels.shape[1]):
            if other_index == label_index:
                continue
            valid = positive & (labels[:, other_index] < 0.5)
            if valid.any():
                own = similarities[valid, label_index, label_index]
                other = similarities[valid, label_index, other_index]
                losses.append(F.relu(float(margin) - own + other))
    if not losses:
        return torch.zeros((), device=label_embeds.device, dtype=label_embeds.dtype)
    return torch.cat(losses).mean()


def forward_cached(
    model: torch.nn.Module,
    label_embeds: torch.Tensor,
    tokens: torch.Tensor,
    token_mask: torch.Tensor,
    active_values: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    safe_tokens, key_padding_mask = safe_text_inputs(tokens, token_mask)
    text_queries = label_embeds + model.label_query_bias
    retrieved, attention = model.text_cross_attn(
        text_queries,
        safe_tokens,
        safe_tokens,
        key_padding_mask=key_padding_mask,
        need_weights=True,
        average_attn_weights=True,
    )
    active = active_values.view(-1, 1, 1).to(dtype=retrieved.dtype)
    retrieved = retrieved * active
    gates = torch.sigmoid(model.text_gate(torch.cat([label_embeds, retrieved], dim=-1))) * active
    logits = model.classify(label_embeds + gates * retrieved)
    return logits, retrieved, attention


def evaluate(
    model: torch.nn.Module,
    dataset: TensorDataset,
    device: torch.device,
    batch_size: int,
) -> dict[str, object]:
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    all_labels: list[np.ndarray] = []
    all_conditions: list[np.ndarray] = []
    retrieved_cosines: list[np.ndarray] = []
    attention_cosines: list[np.ndarray] = []
    model.text_cross_attn.eval()
    with torch.inference_mode():
        for label_embeds, tokens, token_mask, pooled, active_values, labels in loader:
            label_embeds = label_embeds.to(device)
            tokens = tokens.to(device)
            token_mask = token_mask.to(device)
            pooled = pooled.to(device)
            active_values = active_values.to(device)
            labels = labels.to(device)
            logits, retrieved, attention = forward_cached(
                model,
                label_embeds,
                tokens,
                token_mask,
                active_values,
            )
            active = active_values.view(-1, 1, 1).to(dtype=retrieved.dtype)
            correct = torch.sigmoid(logits)
            swapped_values = []
            for permutation in ((1, 2, 0), (2, 0, 1)):
                swapped_retrieval = retrieved[:, permutation, :]
                swapped_values.append(
                    classify_with_text(model, label_embeds, swapped_retrieval, active)
                )
            swapped = torch.stack(swapped_values).mean(dim=0)
            pooled_values = classify_with_text(
                model,
                label_embeds,
                pooled.unsqueeze(1).expand_as(retrieved),
                active,
            )
            all_conditions.append(torch.stack((correct, swapped, pooled_values), dim=1).cpu().numpy())
            all_labels.append(labels.cpu().numpy())

            normalized_retrieved = F.normalize(retrieved, dim=-1)
            normalized_attention = F.normalize(attention, dim=-1)
            for first, second in ((0, 1), (0, 2), (1, 2)):
                retrieved_cosines.append(
                    (normalized_retrieved[:, first] * normalized_retrieved[:, second]).sum(-1).cpu().numpy()
                )
                attention_cosines.append(
                    (normalized_attention[:, first] * normalized_attention[:, second]).sum(-1).cpu().numpy()
                )

    labels_array = np.concatenate(all_labels)
    conditions = np.concatenate(all_conditions)
    values_by_label: list[list[np.ndarray]] = []
    label_summary: dict[str, object] = {}
    for label_index, label_name in enumerate(LABEL_DISPLAY_NAMES):
        positive = labels_array[:, label_index] > 0.5
        current = [conditions[positive, condition_index, label_index] for condition_index in range(3)]
        values_by_label.append(current)
        label_summary[label_name] = {
            "positive_instances": int(positive.sum()),
            "correct": float(current[0].mean()),
            "cross_label": float(current[1].mean()),
            "pooled": float(current[2].mean()),
            "correct_minus_cross_label": float((current[0] - current[1]).mean()),
        }
    return {
        "values_by_label": values_by_label,
        "labels": label_summary,
        "retrieved_pairwise_cosine": float(np.concatenate(retrieved_cosines).mean()),
        "attention_pairwise_cosine": float(np.concatenate(attention_cosines).mean()),
    }


def main() -> None:
    args = parse_args()
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    os.environ.setdefault("PROJECT4_DISABLE_DISK_CACHE_WRITE", "1")
    device = resolve_device(args.device)

    cfg = yaml.safe_load((ROOT / "configs/task3/t3_main_model.yaml").read_text(encoding="utf-8"))
    train_root = PROJECT_ROOT / "outputs/train_runs/task3/t3_main_model"
    fold_dir = train_root / "regular_white_light" / f"fold_{args.fold}"
    record_map = load_masked_records(train_root / "records_cache.json")
    manifest = fold_dir / "split_manifest.csv"
    train_records = select_records(
        load_split_records(manifest, record_map, "train"),
        args.train_sample_size,
        args.seed,
    )
    test_records = select_records(
        load_split_records(manifest, record_map, "test"),
        args.test_sample_size,
        args.seed,
    )

    model = build_model(cfg, fold_dir / "checkpoints/best_macro_f1.ckpt", device)
    train_cache = cache_embeddings(
        model,
        build_loader(train_records, cfg, seed=args.seed, num_workers=args.num_workers),
        device,
        "Cache WLE train embeddings",
    )
    test_cache = cache_embeddings(
        model,
        build_loader(test_records, cfg, seed=args.seed, num_workers=args.num_workers),
        device,
        "Cache WLE test embeddings",
    )
    before = evaluate(model, test_cache, device, args.batch_size)

    for parameter in model.parameters():
        parameter.requires_grad_(False)
    for parameter in model.text_cross_attn.parameters():
        parameter.requires_grad_(True)
    original_state = copy.deepcopy(model.text_cross_attn.state_dict())
    optimizer = torch.optim.AdamW(
        model.text_cross_attn.parameters(),
        lr=float(args.learning_rate),
        weight_decay=1e-4,
    )
    criterion = AsymmetricLossMultiLabel()
    train_generator = torch.Generator().manual_seed(args.seed)
    train_loader = DataLoader(
        train_cache,
        batch_size=args.batch_size,
        shuffle=True,
        generator=train_generator,
    )

    history = []
    model.text_cross_attn.train()
    for epoch in range(1, args.epochs + 1):
        epoch_primary = 0.0
        epoch_alignment = 0.0
        epoch_batches = 0
        for label_embeds, tokens, token_mask, _, active_values, labels in train_loader:
            label_embeds = label_embeds.to(device)
            tokens = tokens.to(device)
            token_mask = token_mask.to(device)
            active_values = active_values.to(device)
            labels = labels.to(device)
            logits, retrieved, _ = forward_cached(
                model,
                label_embeds,
                tokens,
                token_mask,
                active_values,
            )
            primary = criterion(logits, labels)
            alignment = label_alignment_margin_loss(
                label_embeds,
                retrieved,
                labels,
                args.alignment_margin,
            )
            loss = primary + float(args.alignment_weight) * alignment
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.text_cross_attn.parameters(), max_norm=1.0)
            optimizer.step()
            epoch_primary += float(primary.detach().item())
            epoch_alignment += float(alignment.detach().item())
            epoch_batches += 1
        row = {
            "epoch": epoch,
            "primary_loss": epoch_primary / max(epoch_batches, 1),
            "alignment_loss": epoch_alignment / max(epoch_batches, 1),
        }
        history.append(row)
        if epoch == 1 or epoch % 5 == 0 or epoch == args.epochs:
            print(json.dumps(row, ensure_ascii=False))

    after = evaluate(model, test_cache, device, args.batch_size)
    plot(after["values_by_label"], args.output.resolve())

    args.checkpoint_output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "source_checkpoint": str(fold_dir / "checkpoints/best_macro_f1.ckpt"),
            "text_cross_attn_state": model.text_cross_attn.state_dict(),
            "original_text_cross_attn_state": original_state,
            "settings": vars(args),
            "before": {key: value for key, value in before.items() if key != "values_by_label"},
            "after": {key: value for key, value in after.items() if key != "values_by_label"},
        },
        args.checkpoint_output.resolve(),
    )
    stats = {
        "pilot": "WLE Fold 1 cached-feature fine-tuning; only text_cross_attn updated",
        "train_samples": len(train_records),
        "test_samples": len(test_records),
        "settings": {
            "epochs": args.epochs,
            "learning_rate": args.learning_rate,
            "alignment_margin": args.alignment_margin,
            "alignment_weight": args.alignment_weight,
            "seed": args.seed,
        },
        "before": {key: value for key, value in before.items() if key != "values_by_label"},
        "after": {key: value for key, value in after.items() if key != "values_by_label"},
        "history": history,
    }
    args.stats_output.parent.mkdir(parents=True, exist_ok=True)
    args.stats_output.write_text(json.dumps(stats, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(stats, ensure_ascii=False, indent=2))
    print(args.output.resolve())


if __name__ == "__main__":
    main()
