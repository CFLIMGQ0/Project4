#!/usr/bin/env python3
"""生成CQ500与PhysioNet CT-ICH位置编码替换六GPU队列。"""
from __future__ import annotations

import json

from ct_six_gpu_queue import ROOT, save
from run_public_ct_position_replacements import OUTPUTS, VARIANTS


QUEUE = ROOT / "outputs/public_ct_position_suite/queue.json"


def main():
    if QUEUE.exists():
        raise FileExistsError("已有公开CT位置替换队列，不覆盖；直接续跑调度器")
    for dataset, output in OUTPUTS.items():
        audit = json.loads((output / "audit.json").read_text())
        smoke = json.loads((output / "smoke_audit.json").read_text())
        if set(audit) != set(VARIANTS) or set(smoke) != set(VARIANTS):
            raise ValueError(f"{dataset}审计或冒烟结果不完整")
        if not all(item["finite_loss_and_gradients"] for item in smoke.values()):
            raise ValueError(f"{dataset}冒烟检查未通过")

    jobs = []
    for dataset, output_path in OUTPUTS.items():
        output = str(output_path.relative_to(ROOT))
        trained = []
        for fold in range(1, 6):
            for variant in VARIANTS:
                name = f"{dataset}_{variant}_fold{fold}"
                trained.append(name)
                destination = f"{output}/{variant}/fold_{fold}/"
                model = "amef_multimodal" if dataset == "cq500" else "amef_image_branch"
                destination += model
                filenames = ["completed.json", "test_metrics.json", "test_predictions.csv"]
                if dataset == "cq500":
                    filenames += ["best_model.pt", "config.json", "split_ids.json"]
                jobs.append({
                    "id": name,
                    "phase": "training",
                    "memory_mib": 2500,
                    "command": [
                        "src/scripts/run_public_ct_position_replacements.py",
                        "--dataset", dataset,
                        "--variant", variant,
                        "--fold", str(fold),
                    ],
                    "sync_inputs": [f"{output}/{variant}/protocol.json"],
                    "artifacts": [f"{destination}/{filename}" for filename in filenames],
                    "sync_output_dirs": [destination],
                    "timeout_seconds": 7200,
                })
        jobs.append({
            "id": f"{dataset}_aggregate",
            "phase": "aggregate",
            "local_only": True,
            "memory_mib": 500,
            "command": [
                "src/scripts/run_public_ct_position_replacements.py",
                "--dataset", dataset,
                "--aggregate",
            ],
            "depends_on": trained,
            "artifacts": [f"{output}/{filename}" for filename in (
                "comparison.csv", "comparison.json", "comparison.md",
            )],
        })

    stage_inputs = [
        "datasets/cq500/raw/reads.csv",
        "outputs/cq500/convnext_tiny_scan_features.npz",
        "outputs/cq500/image_descriptions/uniform64/descriptions_draft.csv",
        "outputs/cq500/table2_image_baselines/patient_folds.json",
        "outputs/cq500/table2_multimodal_baselines_uniform64/protocol.json",
        "datasets/physionet_ct_ich/computed-tomography-images-for-intracranial-hemorrhage-detection-and-segmentation-1.0.0/hemorrhage_diagnosis.csv",
        "outputs/physionet_ct_ich/mean_pool_baseline/convnext_tiny_slice_features.npz",
        "outputs/physionet_ct_ich/table2_image_baselines/patient_folds.json",
        "outputs/physionet_ct_ich/table2_image_baselines/amef_image_branch.json",
    ]
    save(QUEUE, {
        "description": "CQ500多模态与PhysioNet CT-ICH纯图像位置编码替换五折；204+202六GPU动态并发",
        "stage_inputs": stage_inputs,
        "jobs": jobs,
    })
    print(f"已生成{len(jobs)}个任务：50个训练折，2个汇总。")


if __name__ == "__main__":
    main()
