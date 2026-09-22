#!/usr/bin/env python3
"""在本机可用GPU上动态并发准备公开数据集block3特征。"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import time
from collections import deque
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
PREPARE = ROOT / "src/scripts/prepare_public_block3_features.py"


def free_memory() -> dict[int, int]:
    try:
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=index,memory.free", "--format=csv,noheader,nounits"],
            cwd=ROOT, check=True, capture_output=True, text=True,
        )
    except (FileNotFoundError, subprocess.CalledProcessError):
        return {}
    values = {}
    for line in result.stdout.splitlines():
        parts = [item.strip() for item in line.split(",")]
        if len(parts) == 2:
            values[int(parts[0])] = int(float(parts[1]))
    return values


def cases(dataset: str) -> list[int]:
    if dataset == "amos_mm":
        split = json.loads((ROOT / "outputs/amos_mm/experiment/splits.json").read_text())
        return [int(item) for item in split["test"]]
    folds = json.loads((ROOT / "outputs/mr_rate_1k/experiment/patient_folds.json").read_text())["folds"]
    return sorted({int(item) for fold in folds for item in fold})


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--datasets", nargs="+", default=["amos_mm", "mr_rate"], choices=("amos_mm", "mr_rate"))
    parser.add_argument("--gpus", nargs="+", type=int, default=[1, 2, 3])
    parser.add_argument("--memory-per-task-mib", type=int, default=1800)
    parser.add_argument("--poll-seconds", type=float, default=5.0)
    parser.add_argument("--retry", type=int, default=2)
    args = parser.parse_args()
    pending = deque((dataset, case_index, 0) for dataset in args.datasets for case_index in cases(dataset))
    logs = ROOT / "outputs/public_block3/gpu_pool/prepare_logs"
    logs.mkdir(parents=True, exist_ok=True)
    running: dict[subprocess.Popen, tuple[str, int, int, int, Path]] = {}
    completed = 0
    failed = []
    started = time.monotonic()
    while pending or running:
        memory = free_memory()
        for gpu in args.gpus:
            launched = 0
            running_on_gpu = sum(item[3] == gpu for item in running.values())
            slot_limit = memory.get(gpu, 0) // args.memory_per_task_mib
            while pending and running_on_gpu + launched < slot_limit:
                dataset, case_index, retry = pending.popleft()
                log = logs / f"{dataset}_{case_index:04d}_try{retry}.log"
                command = ["python", "-u", str(PREPARE), "--dataset", dataset,
                           "--case-index", str(case_index), "--device", "cuda:0"]
                stream = log.open("w", encoding="utf-8")
                environment = dict(os.environ)
                environment["CUDA_VISIBLE_DEVICES"] = str(gpu)
                process = subprocess.Popen(command, cwd=ROOT, env=environment, stdout=stream, stderr=subprocess.STDOUT)
                running[process] = (dataset, case_index, retry, gpu, log)
                launched += 1
        finished = []
        for process, item in list(running.items()):
            if process.poll() is None:
                continue
            finished.append(process)
            dataset, case_index, retry, gpu, log = item
            if process.returncode == 0:
                completed += 1
            elif retry < args.retry:
                pending.append((dataset, case_index, retry + 1))
            else:
                failed.append({"dataset": dataset, "case_index": case_index, "gpu": gpu,
                               "returncode": process.returncode, "log": str(log.relative_to(ROOT))})
            print(f"[{completed}/{completed + len(pending) + len(running) - len(finished)}] {dataset} case={case_index} "
                  f"rc={process.returncode} gpu={gpu} pending={len(pending)} running={len(running)-len(finished)}", flush=True)
        for process in finished:
            running.pop(process, None)
        if pending or running:
            time.sleep(args.poll_seconds)
    report = {
        "state": "complete" if not failed else "failed",
        "completed": completed, "failed": failed,
        "elapsed_seconds": time.monotonic() - started, "gpus": args.gpus,
        "memory_per_task_mib": args.memory_per_task_mib,
    }
    report_path = ROOT / "outputs/public_block3/gpu_pool/prepare_report.json"
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False), flush=True)
    if failed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
