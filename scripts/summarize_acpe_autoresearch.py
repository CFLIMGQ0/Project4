#!/usr/bin/env python3
"""汇总 ACPE 探索实验；只供研究决策，不自动写入论文。"""

from __future__ import annotations

import json
from pathlib import Path
from statistics import mean, stdev

ROOT = Path(__file__).resolve().parents[2]
TRAINING = ROOT / "outputs/acpe_autoresearch/training"
DELETION = ROOT / "outputs/acpe_autoresearch/deletion"
BASE = {
    "ct_rate": ROOT / "outputs/ct_rate_680/all_models_fivefold",
    "amos_mm": ROOT / "outputs/amos_mm/all_models",
    "mr_rate_1k": ROOT / "outputs/mr_rate_1k/all_models_fivefold",
}
NAMES = {"ct_rate": "CT-RATE", "amos_mm": "AMOS-MM", "mr_rate_1k": "MR-RATE-1K"}
VARIANTS = ("centroid", "attention_bias", "confidence_gate", "eta_scale", "block_aug", "confidence_half")
CONDITIONS = ("00_B1", "50_B1", "80_B1", "80_B8")


def training_f1(dataset: str, variant: str, fold: int):
    root = BASE[dataset] if variant == "base" else TRAINING / variant / dataset
    folder = root / ("7_labels" if dataset == "amos_mm" else "") / f"fold_{fold}" / "amef_multimodal"
    path = folder / ("result.json" if dataset == "amos_mm" else "test_metrics.json")
    if not path.exists():
        return None
    return float(json.loads(path.read_text())["macro_f1"])


def deletion_delta(dataset: str, variant: str, protocol: str, fold: int, condition: str):
    path = DELETION / protocol / variant / dataset / f"fold_{fold}.json"
    if not path.exists():
        return None
    result = json.loads(path.read_text())["results"][condition]
    return float(result["delta_macro_f1"]), float(result["delta_acc005"])


def format_values(values: list[float]) -> str:
    if not values:
        return "—"
    spread = stdev(values) if len(values) > 1 else 0.0
    return f"{mean(values):+.4f} ± {spread:.4f} ({len(values)} 折)"


def main():
    lines = [
        "# ACPE 系统交互探索结果（研究筛选，非论文正式结果）", "",
        "所有比较按相同数据集、相同折次配对。模型选择与阈值来自验证集。"
        "下面的测试集结果已用于探索性筛选，不能再作为独立确认。", "",
        "## 常规测试集 Macro-F1", "",
        "| 数据集 | 候选 | 配对折数 | 原 ACPE | 候选 | 候选－原 ACPE |", "|---|---|---:|---:|---:|---:|",
    ]
    for dataset in BASE:
        for variant in VARIANTS:
            paired = [(base, candidate) for fold in range(1, 6)
                      if (base := training_f1(dataset, "base", fold)) is not None
                      and (candidate := training_f1(dataset, variant, fold)) is not None]
            if paired:
                lines.append(f"| {NAMES[dataset]} | {variant} | {len(paired)} | "
                             f"{mean(a for a, _ in paired):.4f} | {mean(b for _, b in paired):.4f} | "
                             f"{format_values([b-a for a, b in paired])} |")
    for protocol in ("reindexed", "original"):
        lines += ["", f"## 固定删片：{protocol}", "",
                  "| 数据集 | 候选 | 条件 | ΔMacro-F1 | ΔAcc@0.05 |", "|---|---|---|---:|---:|"]
        for dataset in BASE:
            for variant in VARIANTS:
                for condition in CONDITIONS:
                    paired = [x for fold in range(1, 6)
                              if (x := deletion_delta(dataset, variant, protocol, fold, condition)) is not None]
                    if paired:
                        lines.append(f"| {NAMES[dataset]} | {variant} | {condition} | "
                                     f"{format_values([x for x, _ in paired])} | "
                                     f"{format_values([y for _, y in paired])} |")
    lines += ["", "`reindexed` 表示删片后对保留图像重新编号；`original` 表示保留原始采集索引。"
              "两种协议回答不同问题，不合并计算。Acc@0.05 只衡量采集索引一致性，不表示解剖距离。", ""]
    target = ROOT / "outputs/acpe_autoresearch/round1_summary.md"
    target.write_text("\n".join(lines), encoding="utf-8")
    print(target)


if __name__ == "__main__":
    main()
