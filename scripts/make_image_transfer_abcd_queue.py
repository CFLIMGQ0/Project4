#!/usr/bin/env python3
"""生成纯图像知识迁移三数据集五折六GPU动态队列。"""
from __future__ import annotations

from ct_six_gpu_queue import ROOT, save


QUEUE = ROOT / "outputs/image_transfer_abcd_seed42_v2/queue.json"
DATASETS = ("ct_rate", "amos_mm", "mr_rate")
VARIANTS = ("A", "B", "C", "D")


def main() -> None:
    if QUEUE.exists():
        raise FileExistsError(f"已有队列，不覆盖：{QUEUE}")
    jobs = []
    trained = []
    for dataset in DATASETS:
        for variant in VARIANTS:
            for fold in range(1, 6):
                job_id = f"image_transfer_{dataset}_{variant}_fold{fold}"
                output = f"outputs/image_transfer_abcd_seed42_v2/{dataset}/{variant}/fold_{fold}"
                trained.append(job_id)
                jobs.append({
                    "id": job_id,
                    "phase": "training",
                    "memory_mib": 3000,
                    "command": [
                        "src/scripts/run_image_transfer_abcd.py",
                        "--dataset", dataset,
                        "--variant", variant,
                        "--fold", str(fold),
                    ],
                    "artifacts": [f"{output}/{name}" for name in (
                        "config.json", "history.json", "best_model.pt",
                        "validation_predictions.csv", "test_predictions.csv",
                        "test_metrics.json",
                    )],
                    "timeout_seconds": 21600,
                })
    jobs.append({
        "id": "image_transfer_aggregate",
        "phase": "aggregate",
        "local_only": True,
        "memory_mib": 500,
        "command": ["src/scripts/run_image_transfer_abcd.py", "--aggregate"],
        "depends_on": trained,
        "artifacts": [
            "outputs/image_transfer_abcd_seed42_v2/fold_results.csv",
            "outputs/image_transfer_abcd_seed42_v2/summary.csv",
            "outputs/image_transfer_abcd_seed42_v2/effects.csv",
            "outputs/image_transfer_abcd_seed42_v2/summary.md",
            "outputs/image_transfer_abcd_seed42_v2/aggregate_manifest.json",
        ],
    })
    save(QUEUE, {
        "description": "仅CT-RATE、AMOS-MM、MR-RATE-1K；A/B/C/D各五折；204四卡+202两卡按实时显存动态并发，同卡允许多个任务",
        "remote_root": "/home/Lim/image_transfer_abcd_seed42_v2",
        "stage_inputs": [
            "outputs/ct_rate_680/experiment",
            "outputs/amos_mm/experiment",
            "outputs/mr_rate_1k/experiment",
            "outputs/lccf_ablation/ct_rate/full",
            "outputs/lccf_ablation/amos_mm/full/protocol.json",
            "outputs/lccf_ablation/amos_mm/full/7_labels",
            "outputs/lccf_ablation/mr_rate/full",
        ],
        "jobs": jobs,
    })
    print(f"已生成{len(jobs)}个任务：60个训练折、1个本机汇总；范围仅为三个指定数据集。")


if __name__ == "__main__":
    main()
