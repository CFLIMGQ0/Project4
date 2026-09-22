#!/usr/bin/env python3
"""汇总 AMOS-MM/MR-RATE 随机删除位置恢复结果。"""
from __future__ import annotations

import csv
import json
import math
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
DATASETS = {
    "mr_rate": (ROOT / "outputs/mr_rate_1k/position_recovery_random_seed42", "MR-RATE-1K"),
    "amos_mm": (ROOT / "outputs/amos_mm/position_recovery_random_seed42", "AMOS-MM 七标签"),
}
FRACTIONS = (0.0, 0.25, 0.5, 0.75)
METRICS = ("PRE", "GRE", "Acc@0.02", "Acc@0.05")


def fmt(mean, std):
    if not math.isfinite(mean):
        return "NA"
    return f"{mean:.4f} ± {std:.4f}" if math.isfinite(std) else f"{mean:.4f}"


def main() -> None:
    for dataset, (output, display) in DATASETS.items():
        all_cases = []
        fold_rows = []
        for fold in range(1, 6):
            path = output / "shards" / f"fold_{fold}.json"
            if not path.exists():
                raise RuntimeError(f"缺少结果：{path}")
            payload = json.loads(path.read_text(encoding="utf-8"))
            all_cases.extend(payload["case_metrics"])
        fold_metrics = []
        for fraction in FRACTIONS:
            selected = [row for row in all_cases if row["delete_fraction"] == fraction]
            for fold in range(1, 6):
                if not [row for row in selected if row["fold"] == fold]:
                    raise RuntimeError(f"{display} 缺少 {fraction:.0%} fold{fold} 结果")
            for method in ("baseline", "model"):
                per_fold = []
                for fold in range(1, 6):
                    fold_selected = [row for row in selected if row["fold"] == fold]
                    fold_row = {
                        "dataset": dataset,
                        "delete_fraction": fraction,
                        "delete_percent": int(round(fraction * 100)),
                        "method": "等距位置基线" if method == "baseline" else "ACPE",
                        "fold": fold,
                        "cases": len(fold_selected),
                    }
                    for metric in METRICS:
                        values = np.asarray([item[f"{method}_{metric}"] for item in fold_selected], dtype=float)
                        finite = values[np.isfinite(values)]
                        fold_row[f"{metric}_mean"] = float(np.mean(finite)) if len(finite) else float("nan")
                        fold_row[f"{metric}_n"] = int(len(finite))
                    per_fold.append(fold_row)
                    fold_metrics.append(fold_row)
                row = {
                    "dataset": dataset,
                    "delete_fraction": fraction,
                    "delete_percent": int(round(fraction * 100)),
                    "method": "等距位置基线" if method == "baseline" else "ACPE",
                    "folds": 5,
                    "cases": int(sum(item["cases"] for item in per_fold)),
                }
                for metric in METRICS:
                    values = np.asarray([item[f"{metric}_mean"] for item in per_fold], dtype=float)
                    finite = values[np.isfinite(values)]
                    row[f"{metric}_mean"] = float(np.mean(finite)) if len(finite) else float("nan")
                    row[f"{metric}_std"] = float(np.std(finite, ddof=1)) if len(finite) > 1 else float("nan")
                    row[f"{metric}_n"] = int(len(finite))
                fold_rows.append(row)
        with (output / "summary.csv").open("w", encoding="utf-8-sig", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(fold_rows[0]))
            writer.writeheader()
            writer.writerows(fold_rows)
        with (output / "fold_metrics.csv").open("w", encoding="utf-8-sig", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(fold_metrics[0]))
            writer.writeheader()
            writer.writerows(fold_metrics)
        case_fields = list(all_cases[0])
        with (output / "case_metrics.csv").open("w", encoding="utf-8-sig", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=case_fields)
            writer.writeheader()
            writer.writerows(all_cases)
        lines = [
            f"# {display} 随机删除位置恢复",
            "",
            "固定随机种子 `42`；五折 checkpoint；每例使用已有冻结特征，删除内部特征并保留首尾。",
            "模型输入只使用删除后重新编号的等距槽位；原始索引仅用于计算真值。表中为五折测试折均值 ± 五折间样本标准差。",
            "",
            "| 删除比例 | 方法 | PRE ↓ | GRE ↓ | Acc@0.02 ↑ | Acc@0.05 ↑ |",
            "|---:|---|---:|---:|---:|---:|",
        ]
        for row in fold_rows:
            lines.append(
                f"| {row['delete_percent']}% | {row['method']} | "
                + " | ".join(fmt(row[f"{metric}_mean"], row[f"{metric}_std"]) for metric in METRICS)
                + " |"
            )
        lines.extend([
            "",
            "ACPE 位置来自现有 APro-CoPE 的 `apro_context_coordinates`，按首尾上下文坐标归一化；未增加回归头、排序、裁剪或测试集校准。",
            "冻结特征缓存未携带额外物理坐标元数据，因此真值采用原始缓存切片索引除以完整序列长度减一。",
        ])
        (output / "results.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
        print(f"已汇总 {display}: {output / 'results.md'}")


if __name__ == "__main__":
    main()
