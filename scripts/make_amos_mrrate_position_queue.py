#!/usr/bin/env python3
"""生成 AMOS-MM 七标签与 MR-RATE 位置编码替换的六卡动态队列。"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
VARIANTS = ("original_pe", "comrope_ap", "videorope", "path", "dape_v2_kerple")


def artifacts(dataset: str, variant: str, fold: int) -> list[str]:
    if dataset == "amos_mm":
        base = f"outputs/amos_mm/position_replacements_7_labels/{variant}/7_labels/fold_{fold}/amef_multimodal"
        names = ("result.json", "best_model.pt", "test_predictions.npz", "history.json", "config.json")
    else:
        base = f"outputs/mr_rate_1k/position_replacements/{variant}/fold_{fold}/amef_multimodal"
        names = ("completed.json", "best_model.pt", "test_metrics.json", "test_predictions.csv", "validation_predictions.csv", "history.json", "config.json")
    return [f"{base}/{name}" for name in names]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=ROOT / "outputs/amos_mrrate_position_suite/queue.json")
    args = parser.parse_args()
    jobs = []
    for dataset in ("amos_mm", "mr_rate"):
        for variant in VARIANTS:
            for fold in range(1, 6):
                name = f"{dataset}_{variant}_fold{fold}"
                jobs.append({
                    "id": name,
                    "phase": "train",
                    "memory_mib": 3500,
                    "timeout_seconds": 7200,
                    "command": [
                        "src/scripts/run_amos_mrrate_position_replacements.py",
                        "--dataset", dataset,
                        "--variant", variant,
                        "--fold", str(fold),
                    ],
                    "artifacts": artifacts(dataset, variant, fold),
                })
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps({
        "name": "AMOS-MM-7label-and-MR-RATE-position-replacements",
        "datasets": ["AMOS-MM(7标签)", "MR-RATE(4标签)"],
        "variants": list(VARIANTS),
        "stage_inputs": ["outputs/amos_mm/experiment", "outputs/mr_rate_1k/experiment"],
        "jobs": jobs,
        "policy": "204四卡+202两卡；按实时显存动态分配；允许同一张卡并行多个任务",
    }, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"已生成 {len(jobs)} 个训练任务：{args.output}")


if __name__ == "__main__":
    main()
