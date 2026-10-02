#!/usr/bin/env python3
"""按显存空余并发执行缺失的 2026 适配基线折次。"""

from __future__ import annotations

import fcntl
import json
import os
from pathlib import Path
import subprocess
import sys
import time

from run_paper2026_baselines import OUT, ROOT, REGISTRY

SCRIPT = ROOT / "src/scripts/run_paper2026_baselines.py"
GPUS = (0, 2, 3)  # GPU1 已被其他进程占用；202 主机当前两卡均由 vLLM 占用。
RESERVE_MIB = 6000
MAX_PER_GPU = 3


def free_memory() -> dict[int, int]:
    process = subprocess.run(["nvidia-smi", "--query-gpu=index,memory.free",
                              "--format=csv,noheader,nounits"], check=True, capture_output=True, text=True)
    return {int(parts[0]): int(parts[1]) for line in process.stdout.splitlines()
            if (parts := line.split(","))}


def completed(job: tuple[str, str, int]) -> bool:
    dataset, method, fold = job
    folder = OUT / dataset
    if dataset == "amos_mm":
        folder /= "7_labels"
    folder = folder / f"fold_{fold}" / method
    marker = folder / "completed.json"
    if dataset == "amos_mm":
        return (folder / "result.json").exists() and (folder / "test_predictions.npz").exists()
    return marker.exists() and (folder / "test_metrics.json").exists() and (folder / "test_predictions.csv").exists()


def save_state(data: dict) -> None:
    output = OUT / "gpu_pool_state.json"
    temporary = output.with_suffix(".tmp")
    temporary.write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    temporary.replace(output)


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    with (OUT / "gpu_pool.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        jobs = [(dataset, method, fold) for dataset in ("ct_rate", "amos_mm", "mr_rate_1k")
                for method in REGISTRY for fold in range(1, 6)]
        # Merlin 报告掩码待审计，只运行不读取报告的 GCF-Net。
        jobs += [("merlin_1k", "gcf_net_2026", fold) for fold in range(1, 6)]
        pending = [job for job in jobs if not completed(job)]
        active: dict[subprocess.Popen, tuple[tuple[str, str, int], int, Path]] = {}
        failures: list[dict] = []
        started = time.time()
        while pending or active:
            for process, (job, gpu, path) in list(active.items()):
                if process.poll() is None:
                    continue
                active.pop(process)
                if process.returncode or not completed(job):
                    failures.append({"job": job, "gpu": gpu, "exit_code": process.returncode,
                                     "log": str(path.relative_to(ROOT))})
                print(f"完成 {job}, GPU{gpu}, rc={process.returncode}, remaining={len(pending)}", flush=True)
            budgets = free_memory()
            for gpu in GPUS:
                count = sum(assigned_gpu == gpu for _, assigned_gpu, _ in active.values())
                while pending and count < MAX_PER_GPU and budgets.get(gpu, 0) >= RESERVE_MIB:
                    job = pending.pop(0)
                    dataset, method, fold = job
                    log = OUT / "gpu_pool_logs" / f"{dataset}_{method}_fold{fold}.log"
                    log.parent.mkdir(parents=True, exist_ok=True)
                    env = os.environ.copy()
                    env.update(CUDA_VISIBLE_DEVICES=str(gpu), OMP_NUM_THREADS="2",
                               TOKENIZERS_PARALLELISM="false", PAPER_MEMORY_FRACTION="0.30")
                    with log.open("a", encoding="utf-8") as stream:
                        process = subprocess.Popen([sys.executable, "-u", str(SCRIPT), "--dataset", dataset,
                                                    "--method", method, "--fold", str(fold)],
                                                   cwd=ROOT, env=env, stdout=stream, stderr=subprocess.STDOUT)
                    active[process] = (job, gpu, log)
                    budgets[gpu] -= RESERVE_MIB
                    count += 1
                    print(f"启动 {job} → GPU{gpu}", flush=True)
            save_state({"status": "running", "completed": sum(completed(job) for job in jobs),
                        "total": len(jobs), "pending": len(pending),
                        "active": [{"job": job, "gpu": gpu, "pid": process.pid} for process, (job, gpu, _) in active.items()],
                        "failures": failures, "updated_unix": time.time()})
            if pending or active:
                time.sleep(10)
        done = sum(completed(job) for job in jobs)
        save_state({"status": "complete" if done == len(jobs) and not failures else "incomplete",
                    "completed": done, "total": len(jobs), "failures": failures,
                    "wall_seconds": time.time() - started})
        if failures:
            raise SystemExit(1)


if __name__ == "__main__":
    main()
