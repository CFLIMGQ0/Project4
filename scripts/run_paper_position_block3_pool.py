#!/usr/bin/env python3
"""并发补齐表2已训练位置模型的 Block3 固定删除推理。"""

from __future__ import annotations

import json
import os
from pathlib import Path
import subprocess
import sys
import time

from evaluate_paper_position_block3 import OUTPUT, ROOT, VARIANTS

GPUS = (0, 2, 3)
SCRIPT = ROOT / "src/scripts/evaluate_paper_position_block3.py"


def available():
    result = subprocess.run(["nvidia-smi", "--query-gpu=index,memory.free",
                             "--format=csv,noheader,nounits"], check=True, capture_output=True, text=True)
    return {int(parts[0]): int(parts[1]) for line in result.stdout.splitlines()
            if (parts := line.split(","))}


def complete(job):
    dataset, variant, fold = job
    path = OUTPUT / dataset / variant / f"fold_{fold}.json"
    if not path.exists():
        return False
    payload = json.loads(path.read_text())
    return payload.get("deletion_mode") == "block3" and len(payload.get("metrics", [])) == 4


def main():
    jobs = [(dataset, variant, fold) for dataset in ("amos_mm", "mr_rate")
            for variant in VARIANTS for fold in range(1, 6)]
    pending = [job for job in jobs if not complete(job)]
    active = {}
    failures = []
    started = time.time()
    while pending or active:
        for process, (job, gpu, log) in list(active.items()):
            if process.poll() is None:
                continue
            active.pop(process)
            if process.returncode or not complete(job):
                failures.append({"job": job, "gpu": gpu, "log": str(log.relative_to(ROOT))})
            print(f"完成 {job} GPU{gpu} rc={process.returncode}", flush=True)
        memory = available()
        for gpu in GPUS:
            count = sum(item[1] == gpu for item in active.values())
            while pending and count < 3 and memory.get(gpu, 0) >= 3000:
                job = pending.pop(0)
                dataset, variant, fold = job
                log = OUTPUT / "logs" / f"{dataset}_{variant}_fold{fold}.log"
                log.parent.mkdir(parents=True, exist_ok=True)
                env = os.environ.copy()
                env.update(CUDA_VISIBLE_DEVICES=str(gpu), OMP_NUM_THREADS="2")
                with log.open("a") as stream:
                    process = subprocess.Popen([sys.executable, "-u", str(SCRIPT), "--dataset", dataset,
                                                "--variant", variant, "--fold", str(fold)],
                                               cwd=ROOT, env=env, stdout=stream, stderr=subprocess.STDOUT)
                active[process] = (job, gpu, log)
                memory[gpu] -= 3000
                count += 1
                print(f"启动 {job} → GPU{gpu}", flush=True)
        OUTPUT.mkdir(parents=True, exist_ok=True)
        (OUTPUT / "pool_state.json").write_text(json.dumps({"completed": sum(complete(j) for j in jobs),
            "total": len(jobs), "active": len(active), "pending": len(pending), "failures": failures,
            "updated_unix": time.time()}, ensure_ascii=False, indent=2))
        if pending or active:
            time.sleep(5)
    print(f"表2 Block3 位置替换推理完成 {sum(complete(j) for j in jobs)}/{len(jobs)}，失败{len(failures)}，耗时{time.time()-started:.1f}s", flush=True)
    if failures:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
