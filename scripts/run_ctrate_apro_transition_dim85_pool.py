#!/usr/bin/env python3
"""CT-RATE ACPE transition_dim=85五折实验的六卡动态多任务队列。"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import subprocess
import sys

import ct_six_gpu_queue as queue


ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "outputs/ct_rate_680/amef_apro_transition_dim85_fivefold"
RUNNER = "src/scripts/run_ctrate_680_apro_transition_dim85.py"
REMOTE_ROOT = "/new_data/Lim/ct_rate_apro_transition_dim85_20260920"
DEVICES = [0, 1, 2, 3, 4]
FOLDS = range(1, 6)


def save_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


def build_queue(path: Path) -> None:
    jobs = []
    for fold in FOLDS:
        prefix = f"{OUT.relative_to(ROOT)}/fold_{fold}/amef_multimodal"
        jobs.append({
            "id": f"ct_apro_transition_dim85_fold_{fold}",
            "phase": "train",
            "memory_mib": 2500,
            "timeout_seconds": 86400,
            "command": [RUNNER, "--worker-index", str(fold - 1), "--devices",
                         *map(str, DEVICES), "--output-dir", str(OUT.relative_to(ROOT))],
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
        "description": "CT-RATE 680例：仅ACPE前后过渡分支512->85、511->85->1；其它位置维度和结构不变",
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
        default=ROOT / "outputs/ct_rate_680/apro_transition_dim85_pool/queue.json",
    )
    parser.add_argument("--prepare-only", action="store_true")
    args = parser.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    subprocess.run([
        sys.executable, str(ROOT / RUNNER), "--audit-only", "--output-dir", str(OUT)
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
        sys.executable, str(ROOT / RUNNER), "--aggregate", "--output-dir", str(OUT)
    ], cwd=ROOT, check=True)


if __name__ == "__main__":
    main()
