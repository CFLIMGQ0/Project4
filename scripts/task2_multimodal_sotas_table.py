#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


MODEL_DISPLAY_NAMES = {
    "task2_radfuse_2025": "RadFuse (2025)†",
    "task2_saif_2025": "SAIF (2025)†",
    "task2_mmtf_2025": "MMTF (2026)†",
    "task2_mmfnet_2024": "MMFNet (2024)†",
}
METRIC_KEYS = (
    "macro_f1",
    "micro_f1",
    "macro_roc_auc",
    "macro_pr_auc",
    "subset_accuracy",
    "hamming_loss",
    "kappa",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="汇总表2图文SOTA实验并更新table.md")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("/home/Lim/Project4/outputs/train_runs/task2/table2_multimodal_sotas"),
    )
    parser.add_argument(
        "--table",
        type=Path,
        default=Path("/home/Lim/Project4/src/table.md"),
    )
    parser.add_argument("--update-table", action="store_true")
    return parser.parse_args()


def load_model_name(run_dir: Path) -> str:
    config_path = run_dir / "config.yaml"
    if not config_path.is_file():
        return ""
    for line in config_path.read_text(encoding="utf-8").splitlines():
        if line.startswith("model_name:"):
            return line.split(":", 1)[1].strip()
    return ""


def find_results(output_dir: Path) -> dict[str, dict[str, float]]:
    candidates: dict[str, list[tuple[float, Path]]] = {name: [] for name in MODEL_DISPLAY_NAMES}
    if not output_dir.is_dir():
        return {}
    for metrics_path in output_dir.glob("*/test_macro_f1/metrics.json"):
        model_name = load_model_name(metrics_path.parents[1])
        if model_name in candidates:
            candidates[model_name].append((metrics_path.stat().st_mtime, metrics_path))

    results: dict[str, dict[str, float]] = {}
    for model_name, paths in candidates.items():
        if not paths:
            continue
        metrics_path = max(paths, key=lambda item: item[0])[1]
        payload: dict[str, Any] = json.loads(metrics_path.read_text(encoding="utf-8"))
        raw_metrics = payload.get("metrics", {})
        if not all(key in raw_metrics for key in METRIC_KEYS):
            continue
        results[model_name] = {key: float(raw_metrics[key]) for key in METRIC_KEYS}
    return results


def render_rows(results: dict[str, dict[str, float]]) -> list[str]:
    rows: list[str] = []
    for model_name, display_name in MODEL_DISPLAY_NAMES.items():
        metrics = results.get(model_name)
        if metrics is None:
            continue
        values = " | ".join(f"{metrics[key]:.4f}" for key in METRIC_KEYS)
        rows.append(f"| {display_name} | ✓ | ✓ | {values} |")
    return rows


def update_table(table_path: Path, rows: list[str]) -> None:
    text = table_path.read_text(encoding="utf-8")
    lines = text.splitlines()
    display_names = set(MODEL_DISPLAY_NAMES.values())
    lines = [
        line
        for line in lines
        if not any(line.startswith(f"| {display_name} |") for display_name in display_names)
    ]
    insert_index = next(
        (index for index, line in enumerate(lines) if line.startswith("| **ProMEF-MIL")),
        None,
    )
    if insert_index is None:
        raise RuntimeError("table.md中未找到ProMEF-MIL表2行")
    note = (
        "† 表示依据原论文核心融合机制在本项目统一图像bag、掩码watch、数据划分与训练设置下的"
        "任务适配复现，并非原作者在本数据集上的官方结果。"
    )
    note_prefix = "† 表示依据原论文核心融合机制"
    lines = [line for line in lines if not line.startswith(note_prefix)]
    lines[insert_index:insert_index] = rows
    table_end = next(
        index
        for index in range(insert_index + len(rows), len(lines))
        if lines[index].startswith("## ")
    )
    lines.insert(table_end, note)
    table_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    args = parse_args()
    results = find_results(args.output_dir)
    rows = render_rows(results)
    print(f"已完成 {len(rows)}/{len(MODEL_DISPLAY_NAMES)} 个模型")
    for row in rows:
        print(row)
    if args.update_table:
        if len(rows) != len(MODEL_DISPLAY_NAMES):
            missing = [name for name in MODEL_DISPLAY_NAMES if name not in results]
            raise RuntimeError(f"结果未齐全，拒绝更新table.md；缺少：{missing}")
        update_table(args.table, rows)
        print(f"已更新：{args.table}")


if __name__ == "__main__":
    main()
