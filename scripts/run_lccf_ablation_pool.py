#!/usr/bin/env python3
"""LCCF消融六卡动态池：204四卡、202两卡，允许每卡按剩余显存运行多个折次。"""
from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

import ct_six_gpu_queue as pool
import run_lccf_ablation as experiment

ROOT = experiment.ROOT
RUNNER = "src/scripts/run_lccf_ablation.py"
OUTPUT = experiment.OUTPUT
REMOTE_ROOT = "/home/Lim/lccf_ablation_20260920"
QUEUE = OUTPUT / "gpu_pool/queue.json"


def artifacts(dataset: str, variant: str, fold: int) -> list[str]:
    if dataset == "amos_mm":
        folder = Path(f"outputs/lccf_ablation/{dataset}/{variant}/7_labels/fold_{fold}/{experiment.MODEL_KEY}")
        return [str(folder / name) for name in ("config.json", "history.json", "best_model.pt", "test_predictions.npz", "result.json")]
    folder = Path(f"outputs/lccf_ablation/{dataset}/{variant}/fold_{fold}/{experiment.MODEL_KEY}")
    return [str(folder / name) for name in (
        "config.json", "history.json", "best_model.pt", "validation_predictions.csv",
        "test_predictions.csv", "test_metrics.json", "completed.json",
    )]


def build_queue(memory_mib: int) -> None:
    jobs = []
    for dataset in ("ct_rate", "amos_mm", "mr_rate"):
        for variant in experiment.VARIANTS:
            for fold in range(1, 6):
                jobs.append({
                    "id": f"{dataset}__{variant}__fold_{fold}",
                    "phase": "train",
                    "memory_mib": memory_mib,
                    "timeout_seconds": 14400,
                    "command": [RUNNER, "--datasets", dataset, "--variant", variant, "--fold", str(fold)],
                    "artifacts": artifacts(dataset, variant, fold),
                })
    pool.save(QUEUE, {
        "description": "CT-RATE、AMOS-MM七标签与MR-RATE-1K LCCF单因素消融；17种唯一配置×5折×3数据集=255折次",
        "multiple_tasks_per_gpu": True,
        "gpu_layout": {"204": [0, 1, 2, 3], "202": [0, 1]},
        "remote_root": REMOTE_ROOT,
        "stage_inputs": ["outputs/ct_rate_680/experiment", "outputs/amos_mm/experiment",
                         "outputs/mr_rate_1k/experiment"],
        "jobs": jobs,
    })


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prepare-only", action="store_true")
    parser.add_argument("--memory-mib", type=int, default=1800,
                        help="每个任务的动态显存预算；调度器会按实际空闲显存继续塞入任务")
    parser.add_argument("--poll-seconds", type=int, default=5)
    args = parser.parse_args()
    if args.memory_mib <= 0:
        parser.error("--memory-mib必须大于零")
    subprocess.run([sys.executable, str(ROOT / RUNNER), "--datasets", "ct_rate", "amos_mm", "mr_rate", "--audit"],
                   cwd=ROOT, check=True)
    build_queue(args.memory_mib)
    print(f"已创建170个折次任务：{QUEUE}", flush=True)
    if args.prepare_only:
        return
    pool.REMOTE_ROOT = REMOTE_ROOT
    sys.argv = [sys.argv[0], "--queue", str(QUEUE), "--poll-seconds", str(args.poll_seconds)]
    pool.main()
    subprocess.run([sys.executable, str(ROOT / RUNNER), "--datasets", "ct_rate", "amos_mm", "mr_rate", "--aggregate"],
                   cwd=ROOT, check=True)


if __name__ == "__main__":
    main()
