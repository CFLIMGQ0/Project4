#!/usr/bin/env python3
"""生成CT-RATE四种位置替换模块的六GPU动态训练队列。"""
from __future__ import annotations

import json

from ct_six_gpu_queue import ROOT, save
from run_ctrate_position_replacements import CONDITIONS, VARIANTS


QUEUE = ROOT / "outputs/ct_rate_680/position_replacement_suite/queue.json"


def main():
    if QUEUE.exists():
        raise FileExistsError("已有位置替换队列，不覆盖；直接续跑调度器")
    audits = {}
    smokes = {}
    for condition, settings in CONDITIONS.items():
        audit_path = settings["output"] / "audit.json"
        smoke_path = settings["output"] / "smoke_audit.json"
        audits[condition] = json.loads(audit_path.read_text())
        smokes[condition] = json.loads(smoke_path.read_text())
        if set(audits[condition]) != set(VARIANTS) or set(smokes[condition]) != set(VARIANTS):
            raise ValueError(f"{condition}审计或冒烟结果不完整")
        if not all(item["finite_loss_and_gradients"] and item["apro_disabled"] for item in smokes[condition].values()):
            raise ValueError(f"{condition}冒烟检查未通过")

    jobs = []
    for condition, settings in CONDITIONS.items():
        output = str(settings["output"].relative_to(ROOT))
        trained = []
        for fold in range(1, 6):
            for variant in VARIANTS:
                name = f"{condition}_{variant}_fold{fold}"
                trained.append(name)
                destination = f"{output}/{variant}/fold_{fold}/amef_multimodal"
                jobs.append({
                    "id": name,
                    "phase": "training",
                    "memory_mib": 2500,
                    "command": [
                        "src/scripts/run_ctrate_position_replacements.py",
                        "--condition", condition,
                        "--variant", variant,
                        "--fold", str(fold),
                    ],
                    "sync_inputs": [f"{output}/{variant}/protocol.json"],
                    "artifacts": [f"{destination}/{filename}" for filename in (
                        "completed.json", "test_metrics.json", "test_predictions.csv",
                        "best_model.pt", "config.json", "split_ids.json",
                    )],
                    "sync_output_dirs": [destination],
                    "timeout_seconds": 7200,
                })
        jobs.append({
            "id": f"{condition}_aggregate",
            "phase": "aggregate",
            "local_only": True,
            "memory_mib": 500,
            "command": [
                "src/scripts/run_ctrate_position_replacements.py",
                "--condition", condition,
                "--aggregate",
            ],
            "depends_on": trained,
            "artifacts": [f"{output}/{filename}" for filename in (
                "comparison.csv", "comparison.json", "comparison.md",
            )],
        })

    stage_inputs = [
        "outputs/ct_rate_680/experiment",
        "outputs/ct_rate_680/missing25_raw_hidden_fivefold/data",
        "outputs/ct_rate_680/missing50_raw_hidden_fivefold/data",
        "outputs/ct_rate_680/missing75_raw_hidden_fivefold/data",
        "outputs/ct_rate_680/position_baselines/upstream_manifest.json",
    ]
    save(QUEUE, {
        "description": "标准无删除及25/50/75%固定缺失；四种位置替换；204+202六GPU按实时显存动态并发",
        "stage_inputs": stage_inputs,
        "jobs": jobs,
    })
    print(f"已生成{len(jobs)}个任务：80个训练折，4个本机汇总。")


if __name__ == "__main__":
    main()
