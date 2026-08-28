#!/usr/bin/env python3
"""将TASK3五个图文对比模型的最终五折结果写入table.md。"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SUMMARY = Path(
    "/home/Lim/Project4/outputs/train_runs/task3/"
    "t3_multimodal_sotas_5fold/t3_table2_summary.json"
)

DATASETS = {
    "regular_white_light": "常规白光胃镜",
    "chromoscopic": "染色胃镜",
    "surgical": "手术胃镜",
    "ultrasound": "超声胃镜",
}
MODELS = {
    "task2_hasan_itf_2024": "Image–Text Feature Fusion (2024)†",
    "task2_mmfnet_2024": "MMFNet (2024)†",
    "task2_saif_2025": "SAIF (2025)†",
    "task2_mmtf_2025": "MMTF (2025)†",
    "task2_radfuse_2025": "RadFuse (2025)†",
}
METRICS = (
    "macro_f1",
    "micro_f1",
    "macro_roc_auc",
    "macro_pr_auc",
    "subset_accuracy",
    "hamming_loss",
    "kappa",
)
INTRO = (
    "以下为截至2026年8月4日的最终结果。原有16个模型和新增5个图文多模态对比模型"
    "均在4类胃镜数据集上完成5折实验，共计21个模型、84个模型—数据集组合和420折结果。"
    "数值表示五折测试结果的“均值 ± 标准差”。ProMEF-MIL直接复用已完成的"
    "`t3_main_model`五折结果，本轮未重复训练主模型。"
)
FOOTNOTE = (
    "† 表示依据原论文核心融合机制在本项目统一图像bag、掩码watch、数据划分与训练设置下的"
    "任务适配复现，并非原作者在本数据集上的官方结果。"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--summary", type=Path, default=DEFAULT_SUMMARY)
    parser.add_argument("--table", type=Path, default=ROOT / "table.md")
    parser.add_argument("--check-only", action="store_true")
    return parser.parse_args()


def finite_number(value: Any, label: str) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{label}不是有效数值：{value!r}") from exc
    if not math.isfinite(number):
        raise ValueError(f"{label}不是有限数值：{number}")
    return number


def load_rows(summary_path: Path) -> dict[tuple[str, str], str]:
    payload = json.loads(summary_path.read_text(encoding="utf-8"))
    if not isinstance(payload, list):
        raise ValueError(f"汇总文件顶层应为列表：{summary_path}")

    indexed: dict[tuple[str, str], dict[str, Any]] = {}
    for row in payload:
        if not isinstance(row, dict):
            continue
        model = str(row.get("model", ""))
        dataset = str(row.get("dataset", ""))
        if model in MODELS and dataset in DATASETS:
            key = (model, dataset)
            if key in indexed:
                raise ValueError(f"汇总中存在重复项：{key}")
            indexed[key] = row

    expected = {(model, dataset) for model in MODELS for dataset in DATASETS}
    missing = sorted(expected - set(indexed))
    if missing:
        raise ValueError(f"汇总缺少{len(missing)}个模型—数据集组合：{missing}")

    formatted: dict[tuple[str, str], str] = {}
    for key in sorted(expected):
        row = indexed[key]
        completed = int(row.get("completed_folds", 0))
        if completed != 5:
            raise ValueError(f"{key}仅完成{completed}/5折，拒绝写入最终表格")
        metric_cells = []
        for metric in METRICS:
            mean = finite_number(row.get(f"{metric}_mean"), f"{key}.{metric}_mean")
            std = finite_number(row.get(f"{metric}_std"), f"{key}.{metric}_std")
            metric_cells.append(f"{mean:.4f} ± {std:.4f}")
        formatted[key] = (
            f"| {MODELS[key[0]]} | 图文 | 5/5 | "
            + " | ".join(metric_cells)
            + " |"
        )
    return formatted


def section_bounds(lines: list[str], heading: str, end_prefix: str) -> tuple[int, int]:
    try:
        start = lines.index(heading)
    except ValueError as exc:
        raise ValueError(f"table.md中未找到标题：{heading}") from exc
    end = next(
        (index for index in range(start + 1, len(lines)) if lines[index].startswith(end_prefix)),
        len(lines),
    )
    return start, end


def update_table(table_path: Path, rows: dict[tuple[str, str], str], check_only: bool) -> None:
    original = table_path.read_text(encoding="utf-8")
    lines = original.splitlines()
    task_start, task_end = section_bounds(
        lines,
        "## TASK3：表2模型在四类胃镜数据集上的五折结果",
        "## ",
    )

    content_indices = [
        index
        for index in range(task_start + 1, task_end)
        if lines[index].strip()
    ]
    if not content_indices:
        raise ValueError("TASK3章节为空")
    lines[content_indices[0]] = INTRO

    footnote_indices = [
        index
        for index in range(task_start + 1, task_end)
        if lines[index].startswith("† ")
    ]
    if len(footnote_indices) != 1:
        raise ValueError(f"TASK3章节应有且仅有一条†表注，实际为{len(footnote_indices)}条")
    lines[footnote_indices[0]] = FOOTNOTE

    for dataset, display_name in DATASETS.items():
        heading = f"### {display_name}"
        dataset_start, dataset_end = section_bounds(lines, heading, "### ")
        dataset_end = min(dataset_end, task_end)
        for model, model_display in MODELS.items():
            prefix = f"| {model_display} |"
            matches = [
                index
                for index in range(dataset_start + 1, dataset_end)
                if lines[index].startswith(prefix)
            ]
            if len(matches) != 1:
                raise ValueError(
                    f"{display_name}表中{model_display}应出现1次，实际出现{len(matches)}次"
                )
            lines[matches[0]] = rows[(model, dataset)]

    updated = "\n".join(lines) + ("\n" if original.endswith("\n") else "")
    if check_only:
        print("[TASK3-TABLE] 校验通过；--check-only未修改table.md")
        return
    table_path.write_text(updated, encoding="utf-8")
    print(f"[TASK3-TABLE] 已写入最终五折结果：{table_path}")


def main() -> None:
    args = parse_args()
    rows = load_rows(args.summary.expanduser().resolve())
    update_table(args.table.expanduser().resolve(), rows, args.check_only)


if __name__ == "__main__":
    main()
