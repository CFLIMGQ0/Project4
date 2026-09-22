#!/usr/bin/env python3
"""AMOS-MM/MR-RATE-1K完整原序列三连续区段删除评估。"""
from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "src/scripts"))
import evaluate_amos_mrrate_deletion_classification as base  # noqa: E402
import evaluate_amos_mrrate_position_recovery as recovery  # noqa: E402

SEED = 42
FRACTIONS = (0.0, 0.25, 0.5, 0.75)
TARGET = 64
VARIANTS = ("acpe", "original_pe")


def save_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")
    temporary.replace(path)


def load_block3_inputs(dataset: str):
    info = base.DATASET_INFO[dataset]
    rows = json.loads((info["input"] / "samples.json").read_text(encoding="utf-8"))
    feature_dir = ROOT / "outputs" / ("amos_mm" if dataset == "amos_mm" else "mr_rate_1k") / "block3_full_features"
    records = []
    excluded = {}
    for row in rows:
        path = feature_dir / f"{row['case_index']:04d}.npz"
        excluded_path = feature_dir / f"{row['case_index']:04d}.excluded.json"
        if excluded_path.is_file():
            excluded[row["case_index"]] = json.loads(excluded_path.read_text(encoding="utf-8"))
            records.append(None)
            continue
        if not path.is_file():
            records.append(None)
            continue
        with np.load(path, allow_pickle=False) as cache:
            features = cache["features"].astype(np.float32)
            source_indices = cache["source_indices"].astype(np.int64)
            source_count = int(cache["original_count"])
            sampling = json.loads(str(cache["sampling_json"]))
            if features.shape != (len(source_indices), 768) or not np.isfinite(features).all():
                raise ValueError(f"block3特征异常：{path}")
            if not np.array_equal(source_indices, np.unique(source_indices)):
                raise ValueError(f"block3源索引乱序或重复：{path}")
            for fraction in FRACTIONS:
                item = sampling[str(fraction)]
                selected = np.asarray(item["selected_raw_indices"], dtype=np.int64)
                if len(selected) != TARGET or selected[0] != 0 or selected[-1] != source_count - 1:
                    raise ValueError(f"block3采样异常：{path} fraction={fraction}")
                if not np.isin(selected, source_indices).all():
                    raise ValueError(f"block3缓存缺少最终切片：{path} fraction={fraction}")
            records.append((features, source_indices, source_count, sampling))
    return rows, records, excluded


def eligible_test(dataset: str, fold: int, records):
    test, validation = base.test_indices(dataset, fold)
    eligible = [index for index in test if records[index] is not None]
    return eligible, validation, [index for index in test if records[index] is None]


def selected_local(record, fraction: float) -> np.ndarray:
    source_indices = record[1]
    raw = np.asarray(record[3][str(fraction)]["selected_raw_indices"], dtype=np.int64)
    local = np.searchsorted(source_indices, raw)
    if len(local) != TARGET or not np.array_equal(source_indices[local], raw):
        raise ValueError("block3最终原始索引未映射到特征缓存")
    return local.astype(np.int64)


def manifest(record, case_index: int, fraction: float, dataset: str, fold: int) -> dict:
    item = dict(record[3][str(fraction)])
    item.update({
        "dataset": dataset, "fold": fold, "case_index": case_index,
        "source_length": record[2], "source_count": record[2],
        "source_indices": item["selected_raw_indices"],
        "kept_raw_indices_after_delete": [i for i, flag in enumerate(item["deleted_mask"]) if not flag],
    })
    return item


def run_classification(args: argparse.Namespace) -> None:
    torch.set_num_threads(2)
    torch.backends.mha.set_fastpath_enabled(False)
    device = torch.device(args.device)
    rows, records, excluded = load_block3_inputs(args.dataset)
    test, validation, excluded_test = eligible_test(args.dataset, args.fold, records)
    if args.limit_cases:
        test = test[:args.limit_cases]
    if not test:
        raise RuntimeError(f"没有可评估病例：{args.dataset} fold{args.fold}")
    model, thresholds, checkpoint, config = base.load_model(args.dataset, args.variant, args.fold, device, args.checkpoint_root)
    text_ids, text_mask = base.encode_text(args.dataset, rows, validation)
    labels = np.asarray([row["labels"][:len(thresholds)] for row in rows], dtype=np.int64)
    model_records = {index: records[index][:3] for index in test}
    metrics, manifests = [], []
    with torch.inference_mode():
        for fraction in FRACTIONS:
            selected = {index: selected_local(records[index], fraction) for index in test}
            manifests.extend([manifest(records[index], index, fraction, args.dataset, args.fold) for index in test])
            probabilities = []
            for start in range(0, len(test), args.batch_size):
                batch_cases = test[start:start + args.batch_size]
                images, mask, positions, counts = base.make_batch(model_records, batch_cases, selected, device)
                output = model(
                    images=images, mask=mask,
                    watch_token_ids=text_ids[batch_cases].to(device),
                    watch_token_mask=text_mask[batch_cases].to(device),
                    instance_indices=positions, original_image_counts=counts,
                )
                logits = output["logits"]
                if not torch.isfinite(logits).all():
                    raise FloatingPointError(f"非有限logits: {args.dataset} {args.variant} fold{args.fold}")
                probabilities.append(logits.sigmoid().float().cpu().numpy())
            values = np.concatenate(probabilities, axis=0)
            metrics.append({"delete_fraction": fraction, "test_cases": len(test), **base.calculate_metrics(labels[test], values, thresholds)})
    output = Path(args.output)
    save_json(output, {
        "dataset": args.dataset, "dataset_display": base.DATASET_INFO[args.dataset]["display"],
        "variant": args.variant, "variant_display": base.VARIANTS[args.variant], "fold": args.fold,
        "seed": SEED, "deletion_mode": "block3", "target_instances": TARGET,
        "checkpoint": checkpoint, "checkpoint_config": config,
        "test_indices": test, "original_test_indices": base.test_indices(args.dataset, args.fold)[0],
        "excluded_test_indices": excluded_test, "excluded_details": {str(k): v for k, v in excluded.items() if k in excluded_test},
        "validation_indices": validation, "thresholds": thresholds.tolist(), "metrics": metrics,
        "manifest": manifests,
        "protocol": {
            "source": "完整原始有序序列；不复用预采样64缓存作为删除对象",
            "operation": "floor(N*fraction)后删除三个随机放置、互不相邻的连续区段，再按剩余序号均匀采样T=64",
            "seed": 42, "position_input": "删除后重新编号的0..63等距槽位；原始索引只保存用于评估",
            "same_cases": "75%后不足T的测试病例在四个比例统一排除",
        },
    })
    print(f"完成 block3 分类 {args.dataset} {args.variant} fold{args.fold}: {output}", flush=True)


def normalize_context(context: np.ndarray):
    if context.ndim != 1 or len(context) < 2 or not np.isfinite(context).all():
        return np.full(len(context), np.nan), "invalid_context", False
    monotonic = bool(np.all(np.diff(context) >= -1e-7))
    span = float(context[-1] - context[0])
    if not math.isfinite(span) or abs(span) <= 1e-12:
        return np.full(len(context), np.nan), "invalid_span", monotonic
    prediction = (context - context[0]) / span
    if not np.isfinite(prediction).all():
        return np.full(len(context), np.nan), "invalid_prediction", monotonic
    return prediction, "non_monotonic" if not monotonic else "ok", monotonic


def position_batch(records, case_indices, selected, device):
    images = torch.zeros(len(case_indices), TARGET, 768, 1, 1, dtype=torch.float32)
    mask = torch.ones(len(case_indices), TARGET, dtype=torch.bool)
    slots = torch.arange(TARGET, dtype=torch.long).view(1, -1).expand(len(case_indices), -1).clone()
    counts = torch.full((len(case_indices),), TARGET, dtype=torch.long)
    for batch_index, case_index in enumerate(case_indices):
        features = records[case_index][0]
        local = selected[case_index]
        images[batch_index, :, :, 0, 0] = torch.from_numpy(features[local])
    return images.to(device), mask.to(device), slots.to(device), counts.to(device)


def position_metrics(prediction: np.ndarray, truth: np.ndarray) -> dict[str, float]:
    if not np.isfinite(prediction).all():
        return {key: float("nan") for key in ("PRE", "GRE", "Acc@0.02", "Acc@0.05")}
    error = np.abs(prediction - truth)
    return {
        "PRE": float(error[1:-1].mean()),
        "GRE": float(0.5 * np.abs(np.diff(prediction) - np.diff(truth)).sum()),
        "Acc@0.02": float((error[1:-1] <= 0.02).mean()),
        "Acc@0.05": float((error[1:-1] <= 0.05).mean()),
    }


def run_recovery(args: argparse.Namespace) -> None:
    torch.set_num_threads(2)
    torch.backends.mha.set_fastpath_enabled(False)
    device = torch.device(args.device)
    rows, records, excluded = load_block3_inputs(args.dataset)
    test, _, excluded_test = eligible_test(args.dataset, args.fold, records)
    if args.limit_cases:
        test = test[:args.limit_cases]
    if not test:
        raise RuntimeError(f"没有可评估病例：{args.dataset} fold{args.fold}")
    model = checkpoint = config = None
    if args.variant == "acpe":
        model, _, checkpoint, config = base.load_model(args.dataset, args.variant, args.fold, device, args.checkpoint_root)
        if getattr(model, "apro_positioner", None) is None:
            raise ValueError("ACPE checkpoint没有APro-CoPE位置模块")
    case_metrics, slice_rows, manifests = [], [], []
    with torch.inference_mode():
        for fraction in FRACTIONS:
            selected = {index: selected_local(records[index], fraction) for index in test}
            predictions = {}
            statuses = {}
            if args.variant == "acpe":
                for start in range(0, len(test), args.batch_size):
                    batch_cases = test[start:start + args.batch_size]
                    images, mask, slots, counts = position_batch(records, batch_cases, selected, device)
                    instance_features, _ = model.encode_instances(images, mask)
                    _, _, diagnostics = model._encode_position(instance_features, mask, slots, counts)
                    context = diagnostics.get("apro_context_coordinates")
                    if context is None:
                        raise RuntimeError("ACPE没有返回上下文坐标")
                    for batch_index, case_index in enumerate(batch_cases):
                        raw = context[batch_index].float().cpu().numpy()
                        prediction, status, monotonic = normalize_context(raw)
                        predictions[case_index] = prediction
                        statuses[case_index] = (status, monotonic)
            for case_index in test:
                record = records[case_index]
                sampling = record[3][str(fraction)]
                raw_indices = np.asarray(sampling["selected_raw_indices"], dtype=np.int64)
                truth = raw_indices.astype(np.float64) / float(record[2] - 1)
                baseline = np.linspace(0.0, 1.0, TARGET, dtype=np.float64)
                if args.variant == "acpe":
                    prediction = predictions[case_index]
                    status, monotonic = statuses[case_index]
                else:
                    prediction, status, monotonic = baseline, "original_pe_uniform", True
                base_metrics = position_metrics(baseline, truth)
                model_values = position_metrics(prediction, truth)
                case_metrics.append({
                    "dataset": args.dataset, "fold": args.fold, "case_index": case_index,
                    "delete_fraction": fraction, "method": args.variant,
                    **{f"baseline_{key}": value for key, value in base_metrics.items()},
                    **{f"model_{key}": value for key, value in model_values.items()},
                    "model_status": status, "model_monotonic": monotonic,
                })
                item = manifest(record, case_index, fraction, args.dataset, args.fold)
                manifests.append(item)
                raw_context = prediction if args.variant == "acpe" else np.full(TARGET, np.nan)
                for position, raw_index in enumerate(raw_indices):
                    slice_rows.append({
                        "dataset": args.dataset, "fold": args.fold, "case_index": case_index,
                        "delete_fraction": fraction, "selected_position": position,
                        "raw_index": int(raw_index), "true_position": float(truth[position]),
                        "uniform_position": float(baseline[position]),
                        "acpe_position": float(prediction[position]) if args.variant == "acpe" else float("nan"),
                        "raw_context": float(raw_context[position]) if np.isfinite(raw_context[position]) else float("nan"),
                        "model_status": status,
                    })
    output = Path(args.output)
    save_json(output, {
        "dataset": args.dataset, "dataset_display": base.DATASET_INFO[args.dataset]["display"],
        "variant": args.variant, "fold": args.fold, "seed": SEED, "target_instances": TARGET,
        "checkpoint": checkpoint, "checkpoint_config": config,
        "test_cases": test, "original_test_cases": base.test_indices(args.dataset, args.fold)[0],
        "excluded_test_cases": excluded_test, "excluded_details": {str(k): v for k, v in excluded.items() if k in excluded_test},
        "protocol": {
            "source": "完整原始有序序列；不复用预采样64缓存作为删除对象",
            "operation": "先删除三个连续区段，再均匀采样T=64",
            "position_input": "只传入重新编号的等距槽位0..63与T=64",
            "truth": "最终原始全序列索引/(N-1)",
            "prediction": "ACPE上下文标量坐标按首尾归一化；不排序、不裁剪、不校准",
            "baseline": "同一最终切片上的等距位置",
        },
        "case_metrics": case_metrics, "slice_rows": slice_rows, "manifests": manifests,
    })
    print(f"完成 block3 恢复 {args.dataset} {args.variant} fold{args.fold}: {output}", flush=True)


def aggregate() -> None:
    output_root = ROOT / "outputs/public_block3"
    output_root.mkdir(parents=True, exist_ok=True)
    rows = []
    exclusion_rows = []
    for dataset, folder_name in (("amos_mm", "amos_mm"), ("mr_rate", "mr_rate_1k")):
        for kind in ("classification", "recovery"):
            for variant in VARIANTS:
                folder = ROOT / "outputs" / folder_name / ("deletion_classification_block3_seed42" if kind == "classification" else "position_recovery_block3_seed42") / variant / "folds"
                payloads = [json.loads((folder / f"fold_{fold}.json").read_text(encoding="utf-8")) for fold in range(1, 6)]
                if variant == "acpe" and kind == "classification":
                    for fold, payload in enumerate(payloads, 1):
                        exclusion_rows.append({
                            "dataset": dataset, "fold": fold,
                            "original_test_cases": len(payload.get("original_test_cases", payload.get("original_test_indices", []))),
                            "eligible_test_cases": len(payload.get("test_cases", payload.get("test_indices", []))),
                            "excluded_test_cases": len(payload.get("excluded_test_cases", payload.get("excluded_test_indices", []))),
                        })
                for fraction in FRACTIONS:
                    if kind == "classification":
                        values = [next(item for item in p["metrics"] if item["delete_fraction"] == fraction) for p in payloads]
                        row = {"dataset": dataset, "kind": kind, "variant": variant, "delete_percent": int(fraction * 100), "folds": 5,
                               "test_cases": [p["metrics"][0]["test_cases"] for p in payloads]}
                        for metric in ("macro_f1", "micro_f1", "macro_f1_at_0.5", "exact_match", "macro_auroc"):
                            numbers = np.asarray([float(v[metric]) for v in values], dtype=float)
                            row[f"{metric}_mean"] = float(np.nanmean(numbers)); row[f"{metric}_std"] = float(np.nanstd(numbers, ddof=1))
                        rows.append(row)
                    else:
                        values = []
                        for payload in payloads:
                            items = [item for item in payload["case_metrics"] if item["delete_fraction"] == fraction]
                            values.append({metric: float(np.nanmean([item[f"model_{metric}"] for item in items])) for metric in ("PRE", "GRE", "Acc@0.02", "Acc@0.05")})
                        row = {"dataset": dataset, "kind": kind, "variant": variant, "delete_percent": int(fraction * 100), "folds": 5}
                        for metric in ("PRE", "GRE", "Acc@0.02", "Acc@0.05"):
                            numbers = np.asarray([v[metric] for v in values], dtype=float)
                            row[f"{metric}_mean"] = float(np.nanmean(numbers)); row[f"{metric}_std"] = float(np.nanstd(numbers, ddof=1))
                        rows.append(row)
    with (output_root / "summary.csv").open("w", encoding="utf-8-sig", newline="") as stream:
        fieldnames = []
        for row in rows:
            for field in row:
                if field not in fieldnames:
                    fieldnames.append(field)
        writer = csv.DictWriter(stream, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader(); writer.writerows(rows)
    with (output_root / "exclusion_counts.csv").open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(exclusion_rows[0]))
        writer.writeheader(); writer.writerows(exclusion_rows)
    lines = ["# AMOS-MM / MR-RATE-1K block3 删除实验", "", "固定随机种子42；完整原序列先删除三个非相邻连续区段，再均匀采样T=64。标准差为五折之间的样本标准差；75%后不足T的病例四个比例统一排除。", "", "## 病例口径", "", "| 数据集 | 折次 | 完整测试病例 | 实际纳入 | 统一排除 |", "|---|---:|---:|---:|---:|"]
    for item in exclusion_rows:
        lines.append(f"| {'AMOS-MM 七标签' if item['dataset']=='amos_mm' else 'MR-RATE-1K'} | {item['fold']} | {item['original_test_cases']} | {item['eligible_test_cases']} | {item['excluded_test_cases']} |")
    lines.extend(["", "## 分类 Macro-F1", "", "| 数据集 | 方法 | 删除比例 | Macro-F1 | 固定0.5 F1 |", "|---|---|---:|---:|---:|"])
    for row in rows:
        if row["kind"] == "classification":
            lines.append(f"| {'AMOS-MM 七标签' if row['dataset']=='amos_mm' else 'MR-RATE-1K'} | {'ACPE' if row['variant']=='acpe' else 'Original PE'} | {row['delete_percent']}% | {row['macro_f1_mean']:.4f} ± {row['macro_f1_std']:.4f} | {row['macro_f1_at_0.5_mean']:.4f} ± {row['macro_f1_at_0.5_std']:.4f} |")
    lines.extend(["", "## 位置恢复", "", "| 数据集 | 方法 | 删除比例 | PRE | GRE | Acc@0.02 | Acc@0.05 |", "|---|---|---:|---:|---:|---:|---:|"])
    for row in rows:
        if row["kind"] == "recovery":
            lines.append(f"| {'AMOS-MM 七标签' if row['dataset']=='amos_mm' else 'MR-RATE-1K'} | {'ACPE' if row['variant']=='acpe' else 'Original PE'} | {row['delete_percent']}% | " + " | ".join(f"{row[f'{m}_mean']:.4f} ± {row[f'{m}_std']:.4f}" for m in ("PRE", "GRE", "Acc@0.02", "Acc@0.05")) + " |")
    (output_root / "results.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(output_root / "results.md")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--kind", choices=("classification", "recovery"), required=False)
    parser.add_argument("--dataset", choices=("amos_mm", "mr_rate"), required=False)
    parser.add_argument("--variant", choices=VARIANTS, required=False)
    parser.add_argument("--fold", type=int, choices=range(1, 6), required=False)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--checkpoint-root", type=Path)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--limit-cases", type=int, default=0)
    parser.add_argument("--aggregate", action="store_true")
    args = parser.parse_args()
    if args.aggregate:
        aggregate()
        return
    for name in ("kind", "dataset", "variant", "fold", "output"):
        if getattr(args, name) is None:
            parser.error(f"缺少 --{name}")
    if args.kind == "classification":
        run_classification(args)
    else:
        run_recovery(args)


if __name__ == "__main__":
    main()
