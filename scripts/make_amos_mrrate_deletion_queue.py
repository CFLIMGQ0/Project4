#!/usr/bin/env python3
"""生成 AMOS-MM/MR-RATE 删除比例分类评估的六卡动态队列。"""
from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "outputs/amos_mrrate_deletion_classification_seed42"
SCRIPT = "src/scripts/evaluate_amos_mrrate_deletion_classification.py"


def main() -> None:
    jobs = []
    for dataset in ("mr_rate", "amos_mm"):
        if dataset == "mr_rate":
            output = "outputs/mr_rate_1k/deletion_classification_seed42"
            acpe = "outputs/mr_rate_1k/all_models_fivefold"
            original = "outputs/mr_rate_1k/position_replacements/original_pe"
        else:
            output = "outputs/amos_mm/deletion_classification_7_labels_seed42"
            acpe = "outputs/amos_mm/all_models/7_labels"
            original = "outputs/amos_mm/position_replacements_7_labels/original_pe/7_labels"
        for variant, checkpoint_root in (("acpe", acpe), ("original_pe", original)):
            for fold in range(1, 6):
                shard = f"{output}/shards/{variant}/fold_{fold}.json"
                jobs.append({
                    "id": f"v2_{dataset}_{variant}_fold{fold}",
                    "phase": "evaluate",
                    "memory_mib": 3500,
                    "timeout_seconds": 3600,
                    "command": [
                        SCRIPT, "--dataset", dataset, "--variant", variant,
                        "--fold", str(fold), "--output", shard,
                    ],
                    "artifacts": [shard],
                })
    stage_inputs = [
        "outputs/mr_rate_1k/experiment",
        "outputs/amos_mm/experiment",
    ]
    for root in (
        "outputs/mr_rate_1k/all_models_fivefold",
        "outputs/mr_rate_1k/position_replacements/original_pe",
        "outputs/amos_mm/all_models/7_labels",
        "outputs/amos_mm/position_replacements_7_labels/original_pe/7_labels",
    ):
        for fold in range(1, 6):
            stage_inputs.append(f"{root}/fold_{fold}/amef_multimodal")
    OUT.mkdir(parents=True, exist_ok=True)
    queue = {
        "version": 2,
        "description": "MR-RATE/AMOS-MM固定seed42删除0/25/50/75分类评估；204四卡+202两卡；允许同卡多任务",
        "stage_inputs": stage_inputs,
        "jobs": jobs,
    }
    (OUT / "queue.json").write_text(json.dumps(queue, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    (OUT / "protocol.md").write_text(
        "# 删除比例分类评估\n\n"
        "固定随机种子 `42`；使用已有五折 AMEF/ACPE 与 Original PE checkpoint；"
        "每例已有最多64张冻结特征，删除 `floor(T×比例)` 个内部特征并保留首尾；"
        "保持已有训练接口中的原始切片索引和原始总数；删除后不补齐、不插值。\n\n"
        "队列由 `src/scripts/ct_six_gpu_queue.py` 动态观察204四卡和202两卡，"
        "按显存可用量允许同一张卡并行多个评估任务。\n",
        encoding="utf-8",
    )
    print(f"已生成 {len(jobs)} 个评估任务：{OUT / 'queue.json'}")


if __name__ == "__main__":
    main()
