#!/usr/bin/env python3
"""评估 AMOS-MM/MR-RATE 固定随机删除下的 APro-CoPE 位置恢复。"""
from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "src/scripts"))
from evaluate_amos_mrrate_deletion_classification import (  # noqa: E402
    DATASET_INFO,
    FRACTIONS,
    SEED,
    load_inputs,
    load_model,
    select_indices,
    test_indices,
)


def normalize_context(context: np.ndarray) -> tuple[np.ndarray, str, bool]:
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


def position_metrics(prediction: np.ndarray, truth: np.ndarray) -> dict[str, float]:
    if not np.isfinite(prediction).all():
        return {"PRE": float("nan"), "GRE": float("nan"), "Acc@0.02": float("nan"), "Acc@0.05": float("nan")}
    error = np.abs(prediction - truth)
    return {
        "PRE": float(error[1:-1].mean()),
        "GRE": float(0.5 * np.abs(np.diff(prediction) - np.diff(truth)).sum()),
        "Acc@0.02": float((error[1:-1] <= 0.02).mean()),
        "Acc@0.05": float((error[1:-1] <= 0.05).mean()),
    }


def position_batch(records, case_indices, selected, device):
    maximum = max(len(selected[index]) for index in case_indices)
    images = torch.zeros(len(case_indices), maximum, 768, 1, 1, dtype=torch.float32)
    mask = torch.zeros(len(case_indices), maximum, dtype=torch.bool)
    slots = torch.full((len(case_indices), maximum), -1, dtype=torch.long)
    counts = torch.ones(len(case_indices), dtype=torch.long)
    for batch_index, case_index in enumerate(case_indices):
        record = records[case_index]
        if record is None:
            raise ValueError(f"缺少测试特征：{case_index}")
        features, _, _ = record
        local = selected[case_index]
        length = len(local)
        images[batch_index, :length, :, 0, 0] = torch.from_numpy(features[local])
        mask[batch_index, :length] = True
        slots[batch_index, :length] = torch.arange(length, dtype=torch.long)
        counts[batch_index] = length
    return images.to(device), mask.to(device), slots.to(device), counts.to(device)


def evaluate(args: argparse.Namespace) -> None:
    torch.set_num_threads(2)
    torch.backends.mha.set_fastpath_enabled(False)
    device = torch.device(args.device)
    rows, records = load_inputs(args.dataset)
    test, _ = test_indices(args.dataset, args.fold)
    if args.limit_cases:
        test = test[:args.limit_cases]
    model, _, checkpoint, config = load_model(
        args.dataset, args.variant, args.fold, device, args.checkpoint_root
    )
    if getattr(model, "position_variant", None) not in {
        "apro_full", "apro_absolute_only", "apro_relative_only"
    } or model.apro_positioner is None:
        raise ValueError("checkpoint不是可提取标量上下文坐标的APro-CoPE分支")
    slice_rows = []
    case_rows = []
    manifests = []
    with torch.inference_mode():
        for fraction in FRACTIONS:
            selected = {}
            truth_by_case = {}
            baseline_by_case = {}
            for case_index in test:
                record = records[case_index]
                if record is None:
                    raise ValueError(f"测试病例没有特征：{case_index}")
                features, source_positions, source_count = record
                kept, deleted = select_indices(len(features), fraction, case_index)
                selected[case_index] = kept
                truth = source_positions[kept].astype(np.float64) / float(source_count - 1)
                baseline = np.linspace(0.0, 1.0, len(kept), dtype=np.float64)
                if not np.isclose(truth[0], 0.0) or not np.isclose(truth[-1], 1.0):
                    raise ValueError(f"病例{case_index}未保留完整序列首尾")
                truth_by_case[case_index] = truth
                baseline_by_case[case_index] = baseline
                manifests.append({
                    "case_index": case_index,
                    "delete_fraction": fraction,
                    "seed": SEED,
                    "source_length": len(features),
                    "deleted_count": len(deleted),
                    "deleted_cache_indices": deleted.tolist(),
                    "kept_cache_indices": kept.tolist(),
                    "source_indices": source_positions[kept].tolist(),
                    "source_count": int(source_count),
                })
            model_predictions = {}
            model_statuses = {}
            for start in range(0, len(test), args.batch_size):
                batch_cases = test[start : start + args.batch_size]
                images, mask, slots, counts = position_batch(records, batch_cases, selected, device)
                instance_features, _ = model.encode_instances(images, mask)
                _, _, diagnostics = model._encode_position(instance_features, mask, slots, counts)
                context = diagnostics.get("apro_context_coordinates")
                if context is None:
                    raise RuntimeError("APro-CoPE没有返回 apro_context_coordinates")
                for batch_index, case_index in enumerate(batch_cases):
                    length = len(selected[case_index])
                    raw_context = context[batch_index, :length].float().cpu().numpy()
                    prediction, status, monotonic = normalize_context(raw_context)
                    model_predictions[case_index] = prediction
                    model_statuses[case_index] = (status, monotonic, raw_context)
            for case_index in test:
                truth = truth_by_case[case_index]
                baseline = baseline_by_case[case_index]
                prediction = model_predictions[case_index]
                status, monotonic, raw_context = model_statuses[case_index]
                baseline_metrics = position_metrics(baseline, truth)
                model_metrics = position_metrics(prediction, truth)
                case_rows.append({
                    "dataset": args.dataset,
                    "fold": args.fold,
                    "case_index": case_index,
                    "delete_fraction": fraction,
                    "model_status": status,
                    "model_monotonic": monotonic,
                    **{f"baseline_{key}": value for key, value in baseline_metrics.items()},
                    **{f"model_{key}": value for key, value in model_metrics.items()},
                })
                for selected_position, (source_index, true, uniform, pred) in enumerate(
                    zip(records[case_index][1][selected[case_index]], truth, baseline, prediction)
                ):
                    slice_rows.append({
                        "dataset": args.dataset,
                        "fold": args.fold,
                        "case_index": case_index,
                        "delete_fraction": fraction,
                        "selected_position": selected_position,
                        "source_index": int(source_index),
                        "true_position": float(true),
                        "uniform_position": float(uniform),
                        "acpe_position": float(pred),
                        "acpe_raw_context": float(raw_context[selected_position]),
                        "model_status": status,
                    })
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "dataset": args.dataset,
        "variant": args.variant,
        "dataset_display": DATASET_INFO[args.dataset]["display"],
        "fold": args.fold,
        "seed": SEED,
        "checkpoint": checkpoint,
        "checkpoint_config": config,
        "test_cases": test,
        "protocol": {
            "input": "已有冻结视觉特征序列；随机删除缓存序列内部特征，保留首尾，不补齐、不插值",
            "position_input": "模型只接收删除后重新编号的等距槽位0..K-1与K；原始索引和总数只用于真值评估",
            "truth": "source slice index / (source original count - 1); frozen feature cache没有额外物理坐标元数据",
            "prediction": "APro-CoPE apro_context_coordinates按首尾归一化；不排序、不裁剪、不校准",
            "baseline": "同一删除mask与同一切片的等距位置",
        },
        "manifests": manifests,
        "case_metrics": case_rows,
        "slice_rows": slice_rows,
    }
    output.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"完成 {args.dataset} fold{args.fold}: {output}", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", choices=sorted(DATASET_INFO), required=True)
    parser.add_argument("--fold", type=int, choices=range(1, 6), required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--variant", choices=("acpe", "apro_absolute_only", "apro_relative_only"),
                        default="acpe")
    parser.add_argument("--checkpoint-root", type=Path, default=None,
                        help="分支checkpoint根目录；MR下含fold_*，AMOS下含7_labels/fold_*")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--limit-cases", type=int, default=0)
    evaluate(parser.parse_args())


if __name__ == "__main__":
    main()
