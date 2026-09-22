#!/usr/bin/env python3
"""AMOS-MM与MR-RATE-1K六种ACPE过渡输入组合的两主机动态GPU池。"""
from __future__ import annotations

import shlex
import subprocess
import sys
from pathlib import Path

import ct_six_gpu_queue as pool
import run_amos_mrrate_transition_groups as experiment

ROOT = experiment.ROOT
RUNNER = "src/scripts/run_amos_mrrate_transition_groups.py"
OUTPUT = ROOT / "outputs/public_transition_groups"
REMOTE_ROOT = "/new_data/Lim/amos_mrrate_transition_groups_20260920"
DEFAULT_QUEUE = OUTPUT / "gpu_pool/queue.json"


def relative_output(dataset: str) -> Path:
    return Path("outputs/amos_mm/transition_groups_7_labels" if dataset == "amos_mm"
                else "outputs/mr_rate_1k/transition_groups")


def artifact_names(dataset: str) -> tuple[str, ...]:
    if dataset == "amos_mm":
        return ("result.json", "best_model.pt", "test_predictions.npz", "history.json", "config.json")
    return ("completed.json", "best_model.pt", "test_metrics.json", "test_predictions.csv",
            "validation_predictions.csv", "history.json", "config.json")


def build_queue(path: Path, memory_mib: int) -> None:
    jobs = []
    for dataset in ("amos_mm", "mr_rate"):
        for variant in experiment.VARIANTS:
            for fold in range(1, 6):
                folder = relative_output(dataset) / variant
                folder /= (Path("7_labels") / f"fold_{fold}" / experiment.MODEL_KEY
                           if dataset == "amos_mm" else Path(f"fold_{fold}") / experiment.MODEL_KEY)
                jobs.append({
                    "id": f"{dataset}__transition_{variant.replace('+', '_gap')}__fold{fold}",
                    "phase": "train",
                    "memory_mib": memory_mib,
                    "timeout_seconds": 7200,
                    "command": [RUNNER, "--dataset", dataset, "--variant", variant, "--fold", str(fold)],
                    "artifacts": [str(folder / name) for name in artifact_names(dataset)],
                })
    protocol_files = []
    for dataset in ("amos_mm", "mr_rate"):
        for variant in experiment.VARIANTS:
            protocol_files.append(str(relative_output(dataset) / variant / "protocol.json"))
    pool.save(path, {
        "description": "AMOS-MM七标签与MR-RATE-1K四标签；六种ACPE transition输入组合；每种五折，共60任务",
        "multiple_tasks_per_gpu": True,
        "gpu_layout": {"204": [0, 1, 2, 3], "202": [0, 1]},
        "remote_root": REMOTE_ROOT,
        "stage_inputs": [
            "outputs/amos_mm/experiment",
            "outputs/mr_rate_1k/experiment",
            *protocol_files,
        ],
        "jobs": jobs,
    })


def audited_stage(queue: dict) -> None:
    pool.ORIGINAL_STAGE(queue) if hasattr(pool, "ORIGINAL_STAGE") else pool.stage(queue)
    command = [pool.PYTHON, "-u", RUNNER, "--audit-all"]
    result = pool.remote("cd " + shlex.quote(REMOTE_ROOT) + " && " + shlex.join(command),
                         capture_output=True, timeout=180)
    pool.save(OUTPUT / "remote_preflight.json", {
        "passed": True,
        "remote_root": REMOTE_ROOT,
        "stdout": result.stdout,
        "stderr": result.stderr,
        "description": "202端启动前核对12组数据集/配置协议",
    })
    print("202端 AMOS/MR 六种配置协议全部核对通过。", flush=True)


def main() -> None:
    parser = __import__("argparse").ArgumentParser(description=__doc__)
    parser.add_argument("--prepare-only", action="store_true")
    parser.add_argument("--memory-mib", type=int, default=2000)
    args = parser.parse_args()
    if args.memory_mib <= 0:
        parser.error("--memory-mib必须大于零")
    subprocess.run([sys.executable, str(ROOT / RUNNER), "--audit-all"], cwd=ROOT, check=True)
    OUTPUT.mkdir(parents=True, exist_ok=True)
    for dataset in ("amos_mm", "mr_rate"):
        for variant in experiment.VARIANTS:
            subprocess.run([
                sys.executable, str(ROOT / RUNNER), "--dataset", dataset,
                "--variant", variant, "--initialize",
            ], cwd=ROOT, check=True)
    build_queue(DEFAULT_QUEUE, args.memory_mib)
    print(f"60个折次任务队列已准备：{DEFAULT_QUEUE}", flush=True)
    if args.prepare_only:
        return
    pool.REMOTE_ROOT = REMOTE_ROOT
    pool.ORIGINAL_STAGE = pool.stage
    pool.stage = audited_stage
    sys.argv = [sys.argv[0], "--queue", str(DEFAULT_QUEUE), "--poll-seconds", "5"]
    pool.main()
    subprocess.run([sys.executable, str(ROOT / RUNNER), "--aggregate-all"], cwd=ROOT, check=True)


if __name__ == "__main__":
    main()
