#!/usr/bin/env python3
"""CT-RATE原始ACPE五折重跑的六卡动态多任务队列。"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import subprocess
import sys

import ct_six_gpu_queue as queue


ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "outputs/ct_rate_680/amef_original_acpe_rerun_fivefold_v2"
RUNNER = "src/scripts/run_ctrate_680_original_acpe.py"
REMOTE_ROOT = "/new_data/Lim/ct_rate_original_acpe_rerun_v2_20260920"
DEVICES = [0, 1, 2, 3, 4]
MODEL = "amef_multimodal"


def save_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


def build_queue(path: Path) -> None:
    jobs = []
    for fold in range(1, 6):
        prefix = f"{OUT.relative_to(ROOT)}/fold_{fold}/{MODEL}"
        jobs.append({
            "id": f"ct_original_acpe_fold_{fold}",
            "phase": "train",
            "memory_mib": 2500,
            "timeout_seconds": 86400,
            "command": [RUNNER, "--worker-index", str(fold - 1), "--models", MODEL,
                         "--devices", *map(str, DEVICES), "--output-dir", str(OUT.relative_to(ROOT))],
            "artifacts": [
                f"{prefix}/config.json",
                f"{prefix}/split_ids.json",
                f"{prefix}/history.json",
                f"{prefix}/validation_predictions.csv",
                f"{prefix}/test_predictions.csv",
                f"{prefix}/best_model.pt",
                f"{prefix}/test_metrics.json",
                f"{prefix}/completed.json",
            ],
        })
    save_json(path, {
        "description": "CT-RATE 680例原始ACPE：position_dim=64、双层Fourier投影、默认alpha=1.5；独立五折重跑",
        "multiple_tasks_per_gpu": True,
        "gpu_layout": {"204": [0, 1, 2, 3], "202": [0, 1]},
        "remote_root": REMOTE_ROOT,
        "stage_inputs": [
            "outputs/ct_rate_680/experiment",
            str(OUT.relative_to(ROOT) / "protocol.json"),
        ],
        "jobs": jobs,
    })


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--queue", type=Path,
        default=ROOT / "outputs/ct_rate_680/original_acpe_pool_v2/queue.json",
    )
    parser.add_argument("--prepare-only", action="store_true")
    args = parser.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    subprocess.run([
        sys.executable, str(ROOT / RUNNER), "--audit-only", "--models", MODEL,
        "--output-dir", str(OUT),
    ], cwd=ROOT, check=True)
    build_queue(args.queue)
    print(f"已生成动态队列：{args.queue}", flush=True)
    if args.prepare_only:
        return
    queue.REMOTE_ROOT = REMOTE_ROOT
    queue.DEFAULT_QUEUE = args.queue
    sys.argv = [sys.argv[0], "--queue", str(args.queue)]
    queue.main()
    subprocess.run([
        sys.executable, str(ROOT / RUNNER), "--aggregate", "--models", MODEL,
        "--output-dir", str(OUT),
    ], cwd=ROOT, check=True)


if __name__ == "__main__":
    main()
