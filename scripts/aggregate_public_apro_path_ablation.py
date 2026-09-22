#!/usr/bin/env python3
"""汇总MR-RATE/AMOS-MM两条APro路径的五折F1、删除分类和位置恢复结果。"""
from __future__ import annotations

import csv
import json
import math
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
DATASETS = ("mr_rate", "amos_mm")
VARIANTS = ("apro_absolute_only", "apro_relative_only")
FRACTIONS = (0.0, 0.25, 0.5, 0.75)
METRICS = ("PRE", "GRE", "Acc@0.02", "Acc@0.05")


def dump_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        return
    with path.open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def fmt(value: float, std: float | None = None) -> str:
    if not math.isfinite(value):
        return "NA"
    return f"{value:.4f} ± {std:.4f}" if std is not None and math.isfinite(std) else f"{value:.4f}"


def output_root(dataset: str) -> Path:
    return ROOT / "outputs" / ("mr_rate_1k" if dataset == "mr_rate" else "amos_mm")


def training_f1(dataset: str, variant: str) -> tuple[float, float]:
    if dataset == "mr_rate":
        path = output_root(dataset) / "position_replacements" / variant / "summary.json"
        payload = json.loads(path.read_text())
        if isinstance(payload, list):
            payload = payload[0]
        return float(payload["macro_f1_mean"]), float(payload["macro_f1_std"])
    path = output_root(dataset) / "position_replacements_7_labels" / variant / "summary.json"
    payload = json.loads(path.read_text())
    return float(payload["macro_f1_mean"]), float(payload["macro_f1_std"])


def classification(dataset: str, variant: str) -> list[dict]:
    root = output_root(dataset) / "deletion_classification_path_ablation_seed42" / variant / "shards"
    payloads = [json.loads((root / f"fold_{fold}.json").read_text()) for fold in range(1, 6)]
    rows = []
    for fraction in FRACTIONS:
        selected = [next(item for item in payload["metrics"] if item["delete_fraction"] == fraction)
                    for payload in payloads]
        row = {"dataset": dataset, "variant": variant, "delete_percent": int(fraction * 100),
               "folds": 5, "test_cases_per_fold": int(selected[0]["test_cases"])}
        for key in ("macro_f1", "micro_f1", "macro_f1_at_0.5", "exact_match", "macro_auroc"):
            values = np.asarray([item[key] for item in selected], dtype=float)
            row[f"{key}_mean"] = float(values.mean())
            row[f"{key}_std"] = float(values.std(ddof=1))
        rows.append(row)
    dump_csv(root.parent / "summary.csv", rows)
    return rows


def recovery(dataset: str, variant: str) -> list[dict]:
    root = output_root(dataset) / "position_recovery_path_ablation_seed42" / variant / "shards"
    payloads = [json.loads((root / f"fold_{fold}.json").read_text()) for fold in range(1, 6)]
    rows = []
    for fraction in FRACTIONS:
        selected = [[item for item in payload["case_metrics"] if item["delete_fraction"] == fraction]
                    for payload in payloads]
        for method in ("baseline", "model"):
            row = {"dataset": dataset, "variant": variant,
                   "delete_percent": int(fraction * 100),
                   "method": "等距位置基线" if method == "baseline" else variant,
                   "folds": 5, "cases": int(sum(len(items) for items in selected))}
            for metric in METRICS:
                fold_values = np.asarray([
                    np.nanmean([item[f"{method}_{metric}"] for item in items]) for items in selected
                ], dtype=float)
                finite = fold_values[np.isfinite(fold_values)]
                row[f"{metric}_mean"] = float(finite.mean()) if len(finite) else float("nan")
                row[f"{metric}_std"] = float(finite.std(ddof=1)) if len(finite) > 1 else float("nan")
                row[f"{metric}_fold_n"] = int(len(finite))
            rows.append(row)
    dump_csv(root.parent / "summary.csv", rows)
    return rows


def main() -> None:
    all_f1, all_cls, all_rec = [], [], []
    for dataset in DATASETS:
        for variant in VARIANTS:
            mean, std = training_f1(dataset, variant)
            all_f1.append({"dataset": dataset, "variant": variant, "macro_f1_mean": mean, "macro_f1_std": std})
            all_cls.extend(classification(dataset, variant))
            all_rec.extend(recovery(dataset, variant))
    output = ROOT / "outputs/public_apro_path_ablation_results.md"
    lines = [
        "# MR-RATE-1K / AMOS-MM APro路径消融",
        "",
        "固定随机种子 `42`。MR-RATE 使用固定五折测试折；AMOS-MM 使用固定200例人工测试集上的五个开发折训练。",
        "删除评估使用同一测试病例、同一随机删除mask；模型只接收删除后重新编号的等距槽位，原始索引只用于真值。",
        "表中标准差为五个折次（AMOS为五次训练/测试重复）之间的样本标准差；位置恢复先在每折测试病例上等权，再跨折汇总。",
        "",
        "## 五折分类 Macro-F1",
        "",
        "| 数据集 | 分支 | Macro-F1 |",
        "|---|---|---:|",
    ]
    for row in all_f1:
        display = "MR-RATE-1K" if row["dataset"] == "mr_rate" else "AMOS-MM 七标签"
        name = "APro absolute-only" if row["variant"] == "apro_absolute_only" else "APro relative-only"
        lines.append(f"| {display} | {name} | {fmt(row['macro_f1_mean'], row['macro_f1_std'])} |")
    lines.extend(["", "## 删除比例分类 Macro-F1", "", "| 数据集 | 分支 | 删除比例 | Macro-F1 | 固定0.5 F1 |", "|---|---|---:|---:|---:|"])
    for row in all_cls:
        display = "MR-RATE-1K" if row["dataset"] == "mr_rate" else "AMOS-MM 七标签"
        name = "APro absolute-only" if row["variant"] == "apro_absolute_only" else "APro relative-only"
        lines.append(f"| {display} | {name} | {row['delete_percent']}% | {fmt(row['macro_f1_mean'], row['macro_f1_std'])} | {fmt(row['macro_f1_at_0.5_mean'], row['macro_f1_at_0.5_std'])} |")
    lines.extend(["", "## 随机删除位置恢复", "", "| 数据集 | 分支 | 删除比例 | 方法 | PRE | GRE | Acc@0.02 | Acc@0.05 |", "|---|---|---:|---|---:|---:|---:|---:|"])
    for row in all_rec:
        display = "MR-RATE-1K" if row["dataset"] == "mr_rate" else "AMOS-MM 七标签"
        values = " | ".join(fmt(row[f"{metric}_mean"], row[f"{metric}_std"]) for metric in METRICS)
        lines.append(f"| {display} | {row['variant']} | {row['delete_percent']}% | {row['method']} | {values} |")
    lines.extend(["", "逐折原始 JSON、分类汇总 CSV 和恢复汇总 CSV 均保存在对应数据集的 `deletion_classification_path_ablation_seed42/` 与 `position_recovery_path_ablation_seed42/` 目录。"])
    output.write_text("\n".join(lines) + "\n", encoding="utf-8")
    dump_csv(ROOT / "outputs/public_apro_path_ablation_f1.csv", all_f1)
    print(output)


if __name__ == "__main__":
    main()
