#!/usr/bin/env python3
"""生成 AMOS-MM/MR-RATE 随机删除位置恢复六卡队列。"""
from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "outputs/amos_mrrate_position_recovery_seed42"


def main() -> None:
    jobs = []
    for dataset in ("mr_rate", "amos_mm"):
        if dataset == "mr_rate":
            output = "outputs/mr_rate_1k/position_recovery_random_seed42"
            checkpoint_root = "outputs/mr_rate_1k/all_models_fivefold"
        else:
            output = "outputs/amos_mm/position_recovery_random_seed42"
            checkpoint_root = "outputs/amos_mm/all_models/7_labels"
        for fold in range(1, 6):
            shard = f"{output}/shards/fold_{fold}.json"
            jobs.append({
                "id": f"v1_{dataset}_position_recovery_fold{fold}",
                "phase": "position_recovery",
                "memory_mib": 3000,
                "timeout_seconds": 3600,
                "command": [
                    "src/scripts/evaluate_amos_mrrate_position_recovery.py",
                    "--dataset", dataset, "--fold", str(fold), "--output", shard,
                ],
                "artifacts": [shard],
            })
    stage_inputs = ["outputs/mr_rate_1k/experiment", "outputs/amos_mm/experiment"]
    for root in ("outputs/mr_rate_1k/all_models_fivefold", "outputs/amos_mm/all_models/7_labels"):
        for fold in range(1, 6):
            stage_inputs.append(f"{root}/fold_{fold}/amef_multimodal")
    OUT.mkdir(parents=True, exist_ok=True)
    queue = {
        "version": 1,
        "description": "MR-RATE/AMOS-MM随机删除位置恢复；seed42；204四卡+202两卡；同卡允许多任务",
        "stage_inputs": stage_inputs,
        "jobs": jobs,
    }
    (OUT / "queue.json").write_text(json.dumps(queue, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"已生成 {len(jobs)} 个位置恢复任务：{OUT / 'queue.json'}")


if __name__ == "__main__":
    main()
