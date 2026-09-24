#!/usr/bin/env python3
"""从已有 AMOS-MM/MR-RATE-1K 位置恢复结果补算 gap 指标。"""

from __future__ import annotations

import csv
import argparse
import json
import math
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np
from tqdm import tqdm

from evaluate_ctrate_amef_position_recovery import GAP_EPS, crossing_gap_values


ROOT = Path(__file__).resolve().parents[2]
DATASETS = {
    "amos_mm": (ROOT / "outputs/amos_mm/position_recovery_random_seed42", "AMOS-MM 七标签"),
    "mr_rate_1k": (ROOT / "outputs/mr_rate_1k/position_recovery_random_seed42", "MR-RATE-1K"),
}
FRACTIONS = (0.25, 0.5, 0.75)
METRICS = (
    "PRE", "GRE", "Acc@0.02", "Acc@0.05", "CrossingGapMAE", "SGE", "WGE",
    "SWCG", "SCR", "SCR_count",
)


def write_csv(path: Path, rows: list[dict[str, Any]], fields: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def finite_std(values: list[float]) -> float:
    return float(np.std(np.asarray(values, dtype=np.float64), ddof=1)) if len(values) > 1 else 0.0


def fmt(mean: float, std: float) -> str:
    return "NA" if not math.isfinite(mean) else f"{mean:.4f} ± {std:.4f}"


def evaluate_case(rows: list[dict[str, Any]], manifest: dict[str, Any]) -> dict[str, Any]:
    rows.sort(key=lambda item: int(item["selected_position"]))
    # source_index 用于真值；删除清单的索引空间是最终缓存的 0..63 槽位。
    # CrossingGapMAE 必须在后者上判断区间是否跨过删除槽位。
    selected = np.asarray(
        manifest.get("kept_cache_indices", manifest.get("selected_remaining_slots")),
        dtype=np.int64,
    )
    true = np.asarray([float(item["true_position"]) for item in rows], dtype=np.float64)
    baseline = np.asarray([float(item["uniform_position"]) for item in rows], dtype=np.float64)
    model = np.asarray([float(item["acpe_position"]) for item in rows], dtype=np.float64)
    deleted = np.asarray(
        manifest.get("deleted_cache_indices", manifest.get("deleted_raw_indices", [])),
        dtype=np.int64,
    )
    total = int(max(selected.max(), deleted.max() if len(deleted) else -1) + 1)
    deleted_mask = np.zeros(total, dtype=bool)
    if len(deleted):
        deleted_mask[deleted] = True
    segment_ids = np.full(total, -1, dtype=np.int64)
    values, _ = crossing_gap_values(selected, true, baseline, model, deleted_mask, segment_ids)
    return values


def aggregate_dataset(
    dataset: str,
    output: Path,
    display: str,
    protocol: str = "random",
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    case_rows: list[dict[str, Any]] = []
    interval_rows: list[dict[str, Any]] = []
    shard_dir = output / "shards" if protocol == "random" else output / "acpe" / "folds"
    for shard in tqdm(sorted(shard_dir.glob("fold_*.json")), desc=f"汇总{display} {protocol}位置间距", unit="fold"):
        payload = json.loads(shard.read_text(encoding="utf-8"))
        manifests = {
            (
                int(item["case_index"]),
                float(item.get("delete_fraction", item.get("requested_delete_fraction"))),
            ): item
            for item in payload["manifests"]
        }
        grouped: dict[tuple[int, float], list[dict[str, Any]]] = defaultdict(list)
        for item in payload["slice_rows"]:
            grouped[(int(item["case_index"]), float(item["delete_fraction"]))].append(item)
        metric_lookup = {
            (int(item["case_index"]), float(item["delete_fraction"])): item
            for item in payload["case_metrics"]
        }
        for key, rows in sorted(grouped.items()):
            robust = evaluate_case(rows, manifests[key])
            base = dict(metric_lookup[key])
            base.update({
                "baseline_CrossingGapMAE": robust["baseline_CrossingGapMAE"],
                "model_CrossingGapMAE": robust["model_CrossingGapMAE"],
                "baseline_SGE": robust["baseline_SGE"],
                "model_SGE": robust["model_SGE"],
                "baseline_WGE": robust["baseline_WGE"],
                "model_WGE": robust["model_WGE"],
                "baseline_SWCG": robust["baseline_SWCG"],
                "model_SWCG": robust["model_SWCG"],
                "baseline_SWCG_sanity_error": robust["baseline_SWCG_sanity_error"],
                "baseline_SCR": robust["baseline_SCR"],
                "model_SCR": robust["model_SCR"],
                "baseline_SCR_count": robust["baseline_SCR_count"],
                "model_SCR_count": robust["model_SCR_count"],
                "severe_interval_count": robust["severe_interval_count"],
            })
            for stratum in ("small", "medium", "large"):
                base[f"baseline_{stratum}_GapMAE"] = robust[f"baseline_{stratum}_GapMAE"]
                base[f"model_{stratum}_GapMAE"] = robust[f"model_{stratum}_GapMAE"]
                base[f"{stratum}_interval_count"] = robust[f"{stratum}_interval_count"]
            case_rows.append(base)
            for interval in _interval_rows(rows, manifests[key], robust):
                interval_rows.append(interval)

    summary_rows: list[dict[str, Any]] = []
    stratified_rows: list[dict[str, Any]] = []
    sge_rows: list[dict[str, Any]] = []
    for fraction in FRACTIONS:
        selected = [row for row in case_rows if float(row["delete_fraction"]) == fraction]
        for method, prefix in (("Original PE", "baseline"), ("ACPE", "model")):
            per_fold: list[dict[str, Any]] = []
            for fold in range(1, 6):
                fold_rows = [row for row in selected if int(row["fold"]) == fold]
                fold_row: dict[str, Any] = {"fold": fold, "cases": len(fold_rows)}
                for metric in METRICS:
                    values = [float(row[f"{prefix}_{metric}"]) for row in fold_rows]
                    values = [value for value in values if math.isfinite(value)]
                    fold_row[f"{metric}_mean"] = float(np.mean(values)) if values else float("nan")
                per_fold.append(fold_row)
            summary: dict[str, Any] = {
                "Dataset": display,
                "Method": method,
                "Missing ratio": f"{int(round(fraction * 100))}%",
                "folds": 5,
                "cases": sum(item["cases"] for item in per_fold),
            }
            for metric in METRICS:
                values = [item[f"{metric}_mean"] for item in per_fold if math.isfinite(item[f"{metric}_mean"])]
                summary[f"{metric}_mean"] = float(np.mean(values)) if values else float("nan")
                summary[f"{metric}_std"] = finite_std(values)
                summary[f"{metric}_n"] = len(values)
            summary["SGE valid cases"] = sum(
                math.isfinite(float(row[f"{prefix}_SGE"])) for row in selected
            )
            summary["Severe interval total"] = sum(
                int(row["severe_interval_count"])
                for row in selected
                if math.isfinite(float(row[f"{prefix}_SGE"]))
            )
            summary_rows.append(summary)
            stratified: dict[str, Any] = {
                "Dataset": display,
                "Method": method,
                "Missing ratio": f"{int(round(fraction * 100))}%",
            }
            for stratum in ("small", "medium", "large"):
                values = [
                    float(row[f"{prefix}_{stratum}_GapMAE"])
                    for row in selected
                    if math.isfinite(float(row[f"{prefix}_{stratum}_GapMAE"]))
                ]
                stratified[f"{stratum.title()} GapMAE"] = fmt(float(np.mean(values)), finite_std(values)) if values else "NA"
                stratified[f"{stratum.title()} N"] = sum(int(row[f"{stratum}_interval_count"]) for row in selected)
            stratified_rows.append(stratified)
            sge_rows.append({
                "Method": method,
                "Missing ratio": f"{int(round(fraction * 100))}%",
                "SGE mean": summary["SGE_mean"],
                "SGE std": summary["SGE_std"],
                "SGE valid cases": summary["SGE valid cases"],
                "Severe interval total": summary["Severe interval total"],
            })

    write_csv(output / "case_metrics_gap.csv", case_rows, list(case_rows[0]))
    write_csv(output / "interval_metrics_gap.csv", interval_rows, list(interval_rows[0]))
    write_csv(output / "gap_summary.csv", summary_rows, list(summary_rows[0]))
    write_csv(output / "sge_summary.csv", sge_rows, list(sge_rows[0]))
    stratified_fields = [
        "Method", "Missing ratio", "Small GapMAE", "Medium GapMAE", "Large GapMAE",
        "Small N", "Medium N", "Large N",
    ]
    write_csv(
        output / "mismatch_stratified.csv",
        [{field: row[field] for field in stratified_fields} for row in stratified_rows],
        stratified_fields,
    )
    return summary_rows, stratified_rows


def _interval_rows(rows: list[dict[str, Any]], manifest: dict[str, Any], robust: dict[str, Any]) -> list[dict[str, Any]]:
    sorted_rows = sorted(rows, key=lambda item: int(item["selected_position"]))
    source_selected = [int(row.get("source_index", row.get("raw_index"))) for row in sorted_rows]
    selected = [
        int(value)
        for value in manifest.get("kept_cache_indices", manifest.get("selected_remaining_slots"))
    ]
    true = np.asarray([float(row["true_position"]) for row in sorted_rows])
    baseline = np.asarray([float(row["uniform_position"]) for row in sorted_rows])
    model = np.asarray([float(row["acpe_position"]) for row in sorted_rows])
    true_gaps = np.diff(true)
    baseline_gaps = np.diff(baseline)
    model_gaps = np.diff(model)
    uniform_gap = 1.0 / float(len(true) - 1)
    mismatch = np.abs(true_gaps - uniform_gap) / (uniform_gap + GAP_EPS)
    weight = mismatch ** 2
    severe = mismatch >= 0.5
    base_error = np.abs(true_gaps - uniform_gap)
    model_gap_error = np.abs(model_gaps - true_gaps)
    improved = model_gap_error < base_error
    deleted = set(int(value) for value in manifest.get(
        "deleted_cache_indices", manifest.get("deleted_raw_indices", [])
    ))
    result = []
    for index in range(len(true_gaps)):
        result.append({
            "case_index": int(rows[0]["case_index"]),
            "fold": int(rows[0]["fold"]),
            "delete_fraction": float(rows[0]["delete_fraction"]),
            "interval_index": index,
            "left_raw_index": source_selected[index],
            "right_raw_index": source_selected[index + 1],
            "true_gap": float(true_gaps[index]),
            "uniform_gap": float(uniform_gap),
            "mismatch": float(mismatch[index]),
            "weight": float(weight[index]),
            "severe_interval": bool(severe[index]),
            "severe": bool(severe[index]),
            "gap_stratum": "small" if mismatch[index] < 0.5 else "medium" if mismatch[index] < 1.0 else "large",
            "baseline_gap": float(baseline_gaps[index]),
            "model_gap": float(model_gaps[index]),
            "baseline_gap_error": float(base_error[index]),
            "model_gap_error": float(abs(model_gaps[index] - true_gaps[index])),
            "acpe_gap": float(model_gaps[index]),
            "base_error": float(base_error[index]),
            "acpe_error": float(abs(model_gaps[index] - true_gaps[index])),
            "improved": bool(improved[index]),
            "crossing_gap": bool(any(value in deleted for value in range(selected[index] + 1, selected[index + 1]))),
        })
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--protocol", choices=("random", "block3"), default="random",
        help="使用全随机删除或连续三个大块删除结果",
    )
    args = parser.parse_args()
    if args.protocol == "random":
        dataset_outputs = DATASETS
        root_summary = ROOT / "outputs/public_position_recovery_gap_summary.csv"
        root_stratified = ROOT / "outputs/public_position_recovery_mismatch_stratified.csv"
    else:
        dataset_outputs = {
            "amos_mm": (ROOT / "outputs/amos_mm/position_recovery_block3_seed42", "AMOS-MM 七标签"),
            "mr_rate_1k": (ROOT / "outputs/mr_rate_1k/position_recovery_block3_seed42", "MR-RATE-1K"),
        }
        root_summary = ROOT / "outputs/public_position_recovery_block3_gap_summary.csv"
        root_stratified = ROOT / "outputs/public_position_recovery_block3_mismatch_stratified.csv"
    all_summary: list[dict[str, Any]] = []
    all_stratified: list[dict[str, Any]] = []
    for dataset, (output, display) in dataset_outputs.items():
        summary, stratified = aggregate_dataset(dataset, output, display, args.protocol)
        all_summary.extend(summary)
        all_stratified.extend(stratified)
        print(f"已汇总 {display}: {output / 'gap_summary.csv'}")
    write_csv(root_summary, all_summary, list(all_summary[0]))
    write_csv(root_stratified, all_stratified, list(all_stratified[0]))


if __name__ == "__main__":
    main()
