#!/usr/bin/env python3
"""评估 AMOS-MM 与 MR-RATE 在固定删除比例下的分类性能。"""
from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import f1_score, roc_auc_score

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

SEED = 42
FRACTIONS = (0.0, 0.25, 0.5, 0.75)
DATASET_INFO = {
    "mr_rate": {
        "display": "MR-RATE-1K",
        "input": ROOT / "outputs/mr_rate_1k/experiment",
        "acpe_checkpoint": ROOT / "outputs/mr_rate_1k/all_models_fivefold",
        "original_checkpoint": ROOT / "outputs/mr_rate_1k/position_replacements/original_pe",
        "output": ROOT / "outputs/mr_rate_1k/deletion_classification_seed42",
    },
    "amos_mm": {
        "display": "AMOS-MM 七标签",
        "input": ROOT / "outputs/amos_mm/experiment",
        "acpe_checkpoint": ROOT / "outputs/amos_mm/all_models/7_labels",
        "original_checkpoint": ROOT / "outputs/amos_mm/position_replacements_7_labels/original_pe/7_labels",
        "output": ROOT / "outputs/amos_mm/deletion_classification_7_labels_seed42",
    },
}
VARIANTS = {"acpe": "ACPE/AMEF", "original_pe": "Original PE"}
VARIANTS.update({
    "apro_absolute_only": "AMEF APro absolute-only",
    "apro_relative_only": "AMEF APro relative-only",
})


def save_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")
    temporary.replace(path)


def load_inputs(dataset: str):
    info = DATASET_INFO[dataset]
    rows = json.loads((info["input"] / "samples.json").read_text(encoding="utf-8"))
    records = []
    for row in rows:
        path = info["input"] / "features" / f"{row['case_index']:04d}.npz"
        if not path.exists():
            records.append(None)
            continue
        with np.load(path, allow_pickle=False) as cache:
            features = cache["features"].astype(np.float32)
            positions = cache["slice_indices"].astype(np.int64)
            count = int(cache["original_count"])
            assert features.shape == (len(positions), 768)
            assert len(positions) <= 64 and len(positions) >= 2
            assert np.isfinite(features).all() and np.all(np.diff(positions) > 0)
            assert 0 <= positions[0] <= positions[-1] < count
            records.append((features, positions, count))
    return rows, records


def test_indices(dataset: str, fold: int) -> tuple[list[int], list[int]]:
    info = DATASET_INFO[dataset]
    if dataset == "mr_rate":
        folds = json.loads((info["input"] / "patient_folds.json").read_text(encoding="utf-8"))["folds"]
        test = [int(x) for x in folds[fold - 1]]
        validation = [int(x) for x in folds[fold % 5]]
    else:
        splits = json.loads((info["input"] / "splits.json").read_text(encoding="utf-8"))
        test = [int(x) for x in splits["test"]]
        validation = [int(x) for x in splits["validation_folds"][fold - 1]]
    return test, validation


def select_indices(length: int, fraction: float, case_index: int) -> tuple[np.ndarray, np.ndarray]:
    deleted_count = int(math.floor(length * fraction))
    if deleted_count > length - 2:
        raise ValueError(f"删除后必须保留首尾：length={length}, fraction={fraction}")
    if deleted_count == 0:
        deleted = np.empty(0, dtype=np.int64)
    else:
        random = np.random.default_rng(np.random.SeedSequence([SEED, case_index, int(round(fraction * 100))]))
        deleted = np.sort(random.choice(np.arange(1, length - 1), deleted_count, replace=False))
    keep_mask = np.ones(length, dtype=bool)
    keep_mask[deleted] = False
    kept = np.flatnonzero(keep_mask).astype(np.int64)
    assert len(kept) == length - deleted_count and kept[0] == 0 and kept[-1] == length - 1
    return kept, deleted.astype(np.int64)


def load_model(dataset: str, variant: str, fold: int, device: torch.device,
               checkpoint_root: Path | None = None):
    info = DATASET_INFO[dataset]
    if checkpoint_root is not None:
        folder = Path(checkpoint_root).resolve()
        if dataset == "amos_mm":
            folder = folder / "7_labels"
        folder = folder / f"fold_{fold}" / "amef_multimodal"
    elif dataset == "mr_rate":
        folder = info["acpe_checkpoint"] if variant == "acpe" else info["original_checkpoint"]
        folder = folder / f"fold_{fold}" / "amef_multimodal"
    else:
        folder = info["acpe_checkpoint"] if variant == "acpe" else info["original_checkpoint"]
        folder = folder / f"fold_{fold}" / "amef_multimodal"
    checkpoint_path = folder / "best_model.pt"
    config = json.loads((folder / "config.json").read_text(encoding="utf-8"))
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    parameters = dict(checkpoint.get("model_parameters", config.get("model", config.get("parameters", {}))))
    for key in list(parameters):
        if key.endswith("_weight"):
            parameters.pop(key)
    from exp_8.models import Exp12AProCoPEWatchCrossAttentionTextCNNModel
    from run_physionet_ct_ich_table2_baselines import CachedFeatureBackbone

    model = Exp12AProCoPEWatchCrossAttentionTextCNNModel(
        **parameters,
        pretrained=False,
        num_labels=4 if dataset == "mr_rate" else 7,
    )
    model.instance_encoder.backbone = CachedFeatureBackbone(
        input_dim=768,
        output_dim=parameters["feature_dim"],
        dropout=parameters["dropout"],
    )
    torch.backends.mha.set_fastpath_enabled(False)
    model.load_state_dict(checkpoint["state_dict"], strict=True)
    model.to(device).eval()
    threshold_values = checkpoint.get("thresholds")
    if isinstance(threshold_values, dict):
        threshold_values = list(threshold_values.values())
    thresholds = np.asarray(threshold_values, dtype=np.float32)
    return model, thresholds, str(checkpoint_path.relative_to(ROOT)), config


def encode_text(dataset: str, rows, validation: list[int]):
    if dataset == "mr_rate":
        import run_mrrate_1k_amef as mr_core
        mr_core.fusion_base.SETTINGS = mr_core.SETTINGS
        from run_cq500_table2_multimodal_baselines import encode_descriptions
        texts = {row["case_index"]: row["findings_masked"] for row in rows}
        return encode_descriptions(texts, np.arange(len(rows), dtype=np.int64))
    from run_amos_mm_all_models import tokens_for
    return tokens_for("amef_multimodal", rows, np.asarray(validation, dtype=np.int64), 7)[:2]


def make_batch(records, case_indices: list[int], selected: dict[int, np.ndarray], device: torch.device):
    max_length = max(len(selected[index]) for index in case_indices)
    features = torch.zeros(len(case_indices), max_length, 768, 1, 1, dtype=torch.float32)
    mask = torch.zeros(len(case_indices), max_length, dtype=torch.bool)
    positions = torch.full((len(case_indices), max_length), -1, dtype=torch.long)
    counts = torch.zeros(len(case_indices), dtype=torch.long)
    for batch_index, case_index in enumerate(case_indices):
        record = records[case_index]
        if record is None:
            raise ValueError(f"测试病例没有可用特征：{case_index}")
        bag, source_positions, source_count = record
        local = selected[case_index]
        size = len(local)
        features[batch_index, :size, :, 0, 0] = torch.from_numpy(bag[local])
        positions[batch_index, :size] = torch.arange(size, dtype=torch.long)
        mask[batch_index, :size] = True
        counts[batch_index] = size
    return features.to(device), mask.to(device), positions.to(device), counts.to(device)


def calculate_metrics(targets: np.ndarray, probabilities: np.ndarray, thresholds: np.ndarray) -> dict:
    predictions = probabilities >= thresholds[None, :]
    values = {
        "macro_f1": float(f1_score(targets, predictions, average="macro", zero_division=0)),
        "micro_f1": float(f1_score(targets, predictions, average="micro", zero_division=0)),
        "macro_f1_at_0.5": float(f1_score(targets, probabilities >= 0.5, average="macro", zero_division=0)),
        "exact_match": float((predictions == targets).all(axis=1).mean()),
    }
    aucs = []
    for label in range(targets.shape[1]):
        if len(np.unique(targets[:, label])) == 2:
            aucs.append(roc_auc_score(targets[:, label], probabilities[:, label]))
    values["macro_auroc"] = float(np.mean(aucs)) if aucs else float("nan")
    return values


def run(args: argparse.Namespace) -> None:
    torch.set_num_threads(2)
    torch.backends.mha.set_fastpath_enabled(False)
    device = torch.device(args.device)
    rows, records = load_inputs(args.dataset)
    test, validation = test_indices(args.dataset, args.fold)
    if args.limit_cases:
        test = test[:args.limit_cases]
    model, thresholds, checkpoint, config = load_model(
        args.dataset, args.variant, args.fold, device, args.checkpoint_root
    )
    text_ids, text_mask = encode_text(args.dataset, rows, validation)
    labels = np.asarray([row["labels"][: len(thresholds)] for row in rows], dtype=np.int64)
    metrics = []
    manifests = []
    with torch.inference_mode():
        for fraction in FRACTIONS:
            selected = {}
            manifest_rows = []
            for case_index in test:
                record = records[case_index]
                if record is None:
                    raise ValueError(f"测试病例没有特征：{case_index}")
                kept, deleted = select_indices(len(record[0]), fraction, case_index)
                selected[case_index] = kept
                manifest_rows.append({
                    "case_index": case_index,
                    "delete_fraction": fraction,
                    "source_length": len(record[0]),
                    "deleted_count": int(len(deleted)),
                    "deleted_cache_indices": deleted.tolist(),
                    "kept_cache_indices": kept.tolist(),
                    "source_indices": record[1][kept].tolist(),
                    "source_count": int(record[2]),
                })
            probabilities = []
            for start in range(0, len(test), args.batch_size):
                batch_cases = test[start : start + args.batch_size]
                images, mask, positions, counts = make_batch(records, batch_cases, selected, device)
                output = model(
                    images=images,
                    mask=mask,
                    watch_token_ids=text_ids[batch_cases].to(device),
                    watch_token_mask=text_mask[batch_cases].to(device),
                    instance_indices=positions,
                    original_image_counts=counts,
                )
                logits = output["logits"]
                if not torch.isfinite(logits).all():
                    raise FloatingPointError(f"非有限logits：{args.dataset} {args.variant} fold{args.fold}")
                probabilities.append(logits.sigmoid().float().cpu().numpy())
            probabilities = np.concatenate(probabilities, axis=0)
            metric = calculate_metrics(labels[test], probabilities, thresholds)
            metrics.append({"delete_fraction": fraction, "test_cases": len(test), **metric})
            manifests.extend(manifest_rows)
    output = Path(args.output)
    save_json(output, {
        "dataset": args.dataset,
        "dataset_display": DATASET_INFO[args.dataset]["display"],
        "variant": args.variant,
        "variant_display": VARIANTS[args.variant],
        "fold": args.fold,
        "seed": SEED,
        "delete_fractions": list(FRACTIONS),
        "protocol": {
            "input": "已有64张冻结视觉特征；按缓存序列删除，不补齐、不插值",
            "delete_count": "floor(缓存序列长度×删除比例)",
            "endpoints": "始终保留缓存序列首尾特征",
            "position_input": "只输入删除后重新编号的等距槽位0..K-1与K；原始索引和总数仅用于真值保存",
            "threshold": "沿用该折验证集冻结阈值，不用删除后的测试集调阈值",
        },
        "checkpoint": checkpoint,
        "checkpoint_config": config,
        "test_indices": test,
        "validation_indices": validation,
        "thresholds": thresholds.tolist(),
        "metrics": metrics,
        "manifest": manifests,
    })
    print(f"完成 {args.dataset} {args.variant} fold{args.fold}: {output}", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", choices=sorted(DATASET_INFO), required=True)
    parser.add_argument("--variant", choices=sorted(VARIANTS), required=True)
    parser.add_argument("--fold", type=int, choices=range(1, 6), required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--checkpoint-root", type=Path, default=None,
                        help="分支checkpoint根目录；MR下含fold_*，AMOS下含7_labels/fold_*")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--limit-cases", type=int, default=0)
    args = parser.parse_args()
    run(args)


if __name__ == "__main__":
    main()
