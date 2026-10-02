#!/usr/bin/env python3
"""AA-Mini 表2--8五折任务池；按实际空闲显存动态放置。"""
from __future__ import annotations

import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[2]
RUNNER = ROOT / "src/scripts/run_abdomenatlas3_analysis.py"
OUT = ROOT / "outputs/abdomenatlas3_mini/table2_8_fivefold_anonymous_findings"
STATE = OUT / "scheduler_state.json"
LOGS = OUT / "logs"
sys.path.insert(0, str(ROOT / "src/scripts"))
import run_abdomenatlas3_analysis as experiment


def marker(variant: str, fold: int) -> Path:
    return OUT / variant / f"fold_{fold}" / "amef_multimodal" / "completed.json"


def free_memory() -> dict[int, int]:
    result = subprocess.run(
        ["nvidia-smi", "--query-gpu=index,memory.free", "--format=csv,noheader,nounits"],
        capture_output=True, text=True, check=True,
    )
    return {int(a.strip()): int(b.strip()) for a, b in
            (line.split(",", 1) for line in result.stdout.splitlines() if line.strip())}


def existing_processes():
    """接管同一队列中已由人工启动、仍在运行的折次。"""
    found = {}
    result = subprocess.run(["pgrep", "-af", "run_abdomenatlas3_analysis.py"], capture_output=True, text=True)
    for line in result.stdout.splitlines():
        parts = line.split(None, 1)
        if len(parts) != 2 or str(os.getpid()) == parts[0]:
            continue
        cmd = parts[1]
        for variant in experiment.VARIANTS:
            token = f"--variant {variant}"
            if token in cmd and "--fold " in cmd:
                try:
                    fold = int(cmd.split("--fold ", 1)[1].split()[0])
                except ValueError:
                    continue
                pid = int(parts[0])
                try:
                    environ = Path(f"/proc/{pid}/environ").read_bytes().split(b"\x00")
                    visible = next(item.split(b"=", 1)[1] for item in environ if item.startswith(b"CUDA_VISIBLE_DEVICES="))
                    gpu = int(visible.decode())
                except (FileNotFoundError, StopIteration, ValueError):
                    gpu = None
                found[(variant, fold)] = (pid, gpu)
                break
    return found


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    LOGS.mkdir(parents=True, exist_ok=True)
    gpus = [int(x) for x in os.environ.get("AA_ANALYSIS_GPUS", "0,2,3").split(",") if x.strip()]
    tasks = [(variant, fold) for variant in sorted(experiment.VARIANTS) for fold in range(1, 6)]
    # A complete full model is already part of Table 1 and is deliberately not
    # retrained in this queue.  All 30*5 non-full cells remain independent.
    active = {}
    for task, (pid, gpu) in existing_processes().items():
        if task in tasks and not marker(*task).exists():
            active[task] = {"pid": pid, "gpu": gpu, "started": time.time(), "adopted": True}
    start = time.time()
    while True:
        # Remove exited children; a completed marker is the only success signal.
        for task, item in list(active.items()):
            pid = item["pid"]
            alive = Path(f"/proc/{pid}").exists()
            if not alive:
                active.pop(task)
        done = sum(marker(*task).exists() for task in tasks)
        free = free_memory()
        used_slots = {gpu: sum(item.get("gpu") == gpu for item in active.values()) for gpu in gpus}
        waiting = [task for task in tasks if not marker(*task).exists() and task not in active]
        for gpu in gpus:
            # Cached-feature training currently reserves roughly 1--2 GiB per
            # process on this host.  Keep a conservative 0.8-GiB headroom and
            # allow up to three independent folds when nvidia-smi confirms it.
            while waiting and used_slots[gpu] < 3 and free.get(gpu, 0) >= 800 + 1800 * (used_slots[gpu] + 1):
                task = waiting.pop(0)
                variant, fold = task
                log = LOGS / f"{variant}_fold_{fold}.log"
                env = os.environ.copy()
                env.update(CUDA_VISIBLE_DEVICES=str(gpu), OMP_NUM_THREADS="2",
                           MKL_NUM_THREADS="2", OPENBLAS_NUM_THREADS="2",
                           PYTHONUNBUFFERED="1", TOKENIZERS_PARALLELISM="false")
                handle = log.open("a", encoding="utf-8")
                proc = subprocess.Popen([sys.executable, "-u", str(RUNNER),
                                         "--variant", variant, "--fold", str(fold)],
                                        cwd=ROOT, env=env, stdout=handle,
                                        stderr=subprocess.STDOUT, start_new_session=True)
                handle.close()
                active[task] = {"pid": proc.pid, "gpu": gpu, "started": time.time()}
                used_slots[gpu] += 1
                free[gpu] -= 10500
        state = {"status": "complete" if done == len(tasks) else "running",
                 "gpus": gpus, "total_tasks": len(tasks), "completed_tasks": done,
                 "active": [{"variant": v, "fold": f, **item} for (v, f), item in active.items()],
                 "waiting_tasks": len(waiting), "started_unix": start, "updated_unix": time.time()}
        STATE.write_text(json.dumps(state, ensure_ascii=False, indent=2), encoding="utf-8")
        if done == len(tasks):
            return 0
        time.sleep(30)


if __name__ == "__main__":
    raise SystemExit(main())
