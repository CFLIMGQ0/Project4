#!/usr/bin/env python3
"""汇总 AMOS-MM 与 MR-RATE 删除比例分类评估结果。"""
from __future__ import annotations

import csv
import json
import math
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
DATASETS = {
    "mr_rate": (ROOT / "outputs/mr_rate_1k/deletion_classification_seed42", "MR-RATE-1K"),
    "amos_mm": (ROOT / "outputs/amos_mm/deletion_classification_7_labels_seed42", "AMOS-MM 七标签"),
}
VARIANTS = {"original_pe": "Original PE", "acpe": "ACPE/AMEF"}
FRACTIONS = (0.0, 0.25, 0.5, 0.75)


def fmt(value: float, deviation: float | None = None) -> str:
    if not math.isfinite(value):
        return "NA"
    return f"{value:.4f}" if deviation is None else f"{value:.4f} ± {deviation:.4f}"


def main() -> None:
    for dataset, (output, display) in DATASETS.items():
        rows = []
        fold_rows = []
        missing = []
        for variant, variant_display in VARIANTS.items():
            for fold in range(1, 6):
                path = output / "shards" / variant / f"fold_{fold}.json"
                if not path.exists():
                    missing.append(str(path))
                    continue
                shard = json.loads(path.read_text(encoding="utf-8"))
                for item in shard["metrics"]:
                    fold_row = {
                        "dataset": dataset,
                        "variant": variant,
                        "variant_display": variant_display,
                        "fold": fold,
                        **item,
                    }
                    fold_rows.append(fold_row)
        if missing:
            raise RuntimeError(f"{display} 尚缺少 {len(missing)} 个折次结果，例如：{missing[0]}")
        for fraction in FRACTIONS:
            for variant, variant_display in VARIANTS.items():
                selected = [
                    row for row in fold_rows
                    if row["variant"] == variant and row["delete_fraction"] == fraction
                ]
                if len(selected) != 5:
                    raise RuntimeError(f"{display} {variant_display} {fraction:.0%} 只有{len(selected)}折")
                summary = {
                    "dataset": dataset,
                    "dataset_display": display,
                    "delete_fraction": fraction,
                    "delete_percent": int(round(fraction * 100)),
                    "variant": variant,
                    "variant_display": variant_display,
                    "folds": len(selected),
                    "test_cases_per_fold": selected[0]["test_cases"],
                }
                for metric in ("macro_f1", "micro_f1", "macro_f1_at_0.5", "macro_auroc", "exact_match"):
                    values = np.asarray([row[metric] for row in selected], dtype=float)
                    summary[f"{metric}_mean"] = float(np.nanmean(values))
                    summary[f"{metric}_std"] = float(np.nanstd(values, ddof=1))
                rows.append(summary)
        rows.sort(key=lambda row: (row["delete_fraction"], row["variant"]))
        output.mkdir(parents=True, exist_ok=True)
        fields = list(rows[0])
        with (output / "summary.csv").open("w", encoding="utf-8-sig", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=fields)
            writer.writeheader()
            writer.writerows(rows)
        with (output / "fold_metrics.csv").open("w", encoding="utf-8-sig", newline="") as stream:
            fields = list(fold_rows[0])
            writer = csv.DictWriter(stream, fieldnames=fields)
            writer.writeheader()
            writer.writerows(fold_rows)
        lines = [
            f"# {display} 随机删除比例分类对比",
            "",
            "固定随机种子 `42`；五折测试；每例使用已有最多64张冻结特征，删除 `floor(T×比例)` 个内部特征并保留首尾。",
            "位置输入保持已有训练接口中的原始切片索引与原始总数；两种方法使用完全相同的删除 mask。",
            "表中为五折均值 ± 样本标准差；阈值沿用各折验证集冻结阈值，未使用删除后的测试集调参。",
            "",
            "| 删除比例 | 方法 | Macro-F1 | Micro-F1 | Macro-AUROC | Exact match | Macro-F1@0.5 |",
            "|---:|---|---:|---:|---:|---:|---:|",
        ]
        for row in rows:
            lines.append(
                f"| {row['delete_percent']}% | {row['variant_display']} | "
                f"{fmt(row['macro_f1_mean'], row['macro_f1_std'])} | "
                f"{fmt(row['micro_f1_mean'], row['micro_f1_std'])} | "
                f"{fmt(row['macro_auroc_mean'], row['macro_auroc_std'])} | "
                f"{fmt(row['exact_match_mean'], row['exact_match_std'])} | "
                f"{fmt(row['macro_f1_at_0.5_mean'], row['macro_f1_at_0.5_std'])} |"
            )
        (output / "results.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
        print(f"已汇总 {display}: {output / 'results.md'}")


if __name__ == "__main__":
    main()
