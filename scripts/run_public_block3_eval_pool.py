#!/usr/bin/env python3
"""动态并发运行 AMOS-MM/MR-RATE-1K block3 分类与恢复五折任务。"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import time
from collections import deque
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
RUNNER = ROOT / "src/scripts/evaluate_public_block3.py"


def free_memory() -> dict[int, int]:
    result = subprocess.run(
        ["nvidia-smi", "--query-gpu=index,memory.free", "--format=csv,noheader,nounits"],
        cwd=ROOT, check=True, capture_output=True, text=True,
    )
    return {int(parts[0].strip()): int(float(parts[1].strip()))
            for line in result.stdout.splitlines()
            if len(parts := line.split(",")) == 2}


def jobs():
    for dataset in ("amos_mm", "mr_rate"):
        folder = "amos_mm" if dataset == "amos_mm" else "mr_rate_1k"
        for kind in ("classification", "recovery"):
            name = "deletion_classification_block3_seed42" if kind == "classification" else "position_recovery_block3_seed42"
            for variant in ("acpe", "original_pe"):
                for fold in range(1, 6):
                    output = ROOT / "outputs" / folder / name / variant / "folds" / f"fold_{fold}.json"
                    yield dataset, kind, variant, fold, output


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gpus", nargs="+", type=int, default=[0, 1, 2, 3])
    parser.add_argument("--memory-per-task-mib", type=int, default=2400)
    parser.add_argument("--poll-seconds", type=float, default=5.0)
    parser.add_argument("--retry", type=int, default=2)
    args = parser.parse_args()
    pending = deque()
    for dataset, kind, variant, fold, output in jobs():
        if output.is_file():
            continue
        pending.append((dataset, kind, variant, fold, output, 0))
    logs = ROOT / "outputs/public_block3/gpu_pool/eval_logs"
    logs.mkdir(parents=True, exist_ok=True)
    running = {}
    failed = []
    completed = 0
    started = time.monotonic()
    while pending or running:
        memory = free_memory()
        for gpu in args.gpus:
            launched = 0
            running_on_gpu = sum(item[4] == gpu for item in running.values())
            slot_limit = memory.get(gpu, 0) // args.memory_per_task_mib
            while pending and running_on_gpu + launched < slot_limit:
                dataset, kind, variant, fold, output, retry = pending.popleft()
                log = logs / f"{dataset}_{kind}_{variant}_fold{fold}_try{retry}.log"
                command = ["python", "-u", str(RUNNER), "--kind", kind, "--dataset", dataset,
                           "--variant", variant, "--fold", str(fold), "--output", str(output),
                           "--device", "cuda:0", "--batch-size", "16"]
                environment = dict(os.environ)
                environment["CUDA_VISIBLE_DEVICES"] = str(gpu)
                process = subprocess.Popen(command, cwd=ROOT, env=environment,
                                           stdout=log.open("w", encoding="utf-8"), stderr=subprocess.STDOUT)
                running[process] = (dataset, kind, variant, fold, retry, gpu, log, output)
                launched += 1
        finished = []
        for process, item in list(running.items()):
            if process.poll() is None:
                continue
            finished.append(process)
            dataset, kind, variant, fold, retry, gpu, log, output = item
            if process.returncode == 0:
                completed += 1
            elif retry < args.retry:
                pending.append((dataset, kind, variant, fold, output, retry + 1))
            else:
                failed.append({"dataset": dataset, "kind": kind, "variant": variant,
                               "fold": fold, "returncode": process.returncode, "log": str(log.relative_to(ROOT))})
            print(f"完成/失败 {completed}，待处理 {len(pending)}，运行中 {len(running)-len(finished)}：{item[:4]} rc={process.returncode}", flush=True)
        for process in finished:
            running.pop(process, None)
        if pending or running:
            time.sleep(args.poll_seconds)
    report = {"state": "complete" if not failed else "failed", "completed": completed,
              "failed": failed, "elapsed_seconds": time.monotonic() - started, "gpus": args.gpus}
    report_path = ROOT / "outputs/public_block3/gpu_pool/eval_report.json"
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    if not failed:
        subprocess.run(["python", "-u", str(RUNNER), "--aggregate"], cwd=ROOT, check=True)
    print(json.dumps(report, ensure_ascii=False), flush=True)
    if failed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
