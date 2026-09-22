#!/usr/bin/env python3
"""在204四卡、202两卡上按剩余显存多任务并发运行ACPE六组输入消融。"""
from __future__ import annotations

import argparse
from pathlib import Path
import shlex
import subprocess
import sys

import ct_six_gpu_queue as pool
import run_ctrate_acpe_transition_groups as experiment

ROOT = experiment.ROOT
OUTPUT = experiment.OUTPUT
RUNNER = "src/scripts/run_ctrate_acpe_transition_groups.py"
REMOTE_ROOT = "/new_data/Lim/ct_rate_acpe_transition_groups_20260920"
DEFAULT_QUEUE = OUTPUT / "gpu_pool/queue.json"
ORIGINAL_STAGE = pool.stage


def build_queue(path, memory_mib):
    relative_output = OUTPUT.relative_to(ROOT)
    jobs = []
    for fold in range(1, 6):
        for variant in experiment.VARIANTS:
            folder = relative_output / variant / f"fold_{fold}" / experiment.MODEL_KEY
            jobs.append({
                "id": f"acpe_groups_{variant.replace('+', '_gap')}_fold{fold}",
                "phase": "train", "memory_mib": memory_mib, "timeout_seconds": 7200,
                "command": [RUNNER, "--variant", variant, "--fold", str(fold),
                            "--output-root", str(relative_output)],
                "artifacts": [str(folder / name) for name in (
                    "config.json", "split_ids.json", "history.json", "best_model.pt",
                    "validation_predictions.csv", "test_predictions.csv", "test_metrics.json", "completed.json",
                )],
            })
    pool.save(path, {
        "description": "CT-RATE六种ACPE transition输入组合，每种五折共30任务；原始ACPE其它结构不变",
        "multiple_tasks_per_gpu": True,
        "gpu_layout": {"204": [0, 1, 2, 3], "202": [0, 1]},
        "remote_root": REMOTE_ROOT,
        "stage_inputs": [
            "outputs/ct_rate_680/experiment",
            str(experiment.BASELINE.relative_to(ROOT) / "fold_1/amef_multimodal/config.json"),
            *[str(relative_output / variant / "protocol.json") for variant in experiment.VARIANTS],
        ],
        "jobs": jobs,
    })


def audited_stage(queue):
    ORIGINAL_STAGE(queue)
    command = [pool.PYTHON, "-u", RUNNER, "--audit-all", "--output-root", str(OUTPUT.relative_to(ROOT))]
    result = pool.remote("cd " + shlex.quote(REMOTE_ROOT) + " && " + shlex.join(command),
                         capture_output=True, timeout=180)
    pool.save(OUTPUT / "remote_preflight.json", {
        "passed": True, "remote_root": REMOTE_ROOT, "stdout": result.stdout,
        "stderr": result.stderr, "description": "202端在启动30个任务前核对六种协议与基线训练配置",
    })
    print("202端六种协议全部核对通过，开始多任务分配。", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prepare-only", action="store_true")
    parser.add_argument("--memory-mib", type=int, default=2000)
    args = parser.parse_args()
    if args.memory_mib <= 0:
        parser.error("--memory-mib必须大于零")
    subprocess.run([sys.executable, str(ROOT / RUNNER), "--audit-all"], cwd=ROOT, check=True)
    build_queue(DEFAULT_QUEUE, args.memory_mib)
    print(f"30个折次任务的队列已准备：{DEFAULT_QUEUE}", flush=True)
    if args.prepare_only:
        return
    pool.REMOTE_ROOT = REMOTE_ROOT
    pool.stage = audited_stage
    sys.argv = [sys.argv[0], "--queue", str(DEFAULT_QUEUE), "--poll-seconds", "5"]
    pool.main()
    subprocess.run([sys.executable, str(ROOT / RUNNER), "--aggregate-all"], cwd=ROOT, check=True)


if __name__ == "__main__":
    main()
