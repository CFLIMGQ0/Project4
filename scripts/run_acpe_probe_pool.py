#!/usr/bin/env python3
"""在 204 主机的三张空闲显卡上并行完成 ACPE 反事实诊断。"""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor, as_completed
import os
from pathlib import Path
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[2]
SCRIPT = ROOT / "src/scripts/probe_acpe_counterfactual.py"
OUT = ROOT / "outputs/acpe_autoresearch/counterfactual"
LOGS = ROOT / "outputs/acpe_autoresearch/logs"
ASSIGNMENTS = {"ct_rate": 0, "amos_mm": 2, "mr_rate_1k": 3}


def run_one(dataset: str, fold: int, protocol: str) -> tuple[str, int]:
    destination = OUT / protocol / dataset / f"fold_{fold}.json"
    name = f"{dataset}_fold{fold}_{protocol}"
    if destination.exists():
        return name, 0
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = str(ASSIGNMENTS[dataset])
    env["OMP_NUM_THREADS"] = "2"
    LOGS.mkdir(parents=True, exist_ok=True)
    command = [
        sys.executable, str(SCRIPT), "--dataset", dataset,
        "--fold", str(fold), "--index-protocol", protocol,
        "--batch-size", "32", "--output-root", str(OUT),
    ]
    with (LOGS / f"{name}.log").open("w", encoding="utf-8") as stream:
        process = subprocess.run(command, cwd=ROOT, env=env, stdout=stream, stderr=subprocess.STDOUT)
    return name, process.returncode


def main() -> None:
    tasks = [(dataset, fold, protocol)
             for protocol in ("reindexed", "original")
             for dataset in ASSIGNMENTS
             for fold in range(1, 6)]
    started = time.monotonic()
    failures = []
    # 每张可用显卡同时运行五折；同一卡上的原始索引协议随后执行。
    with ThreadPoolExecutor(max_workers=15) as pool:
        pending = {pool.submit(run_one, *task): task for task in tasks}
        for future in as_completed(pending):
            name, code = future.result()
            print(f"{name}: {'完成' if code == 0 else f'失败({code})'}", flush=True)
            if code:
                failures.append(name)
    print(f"总用时 {time.monotonic() - started:.1f} 秒；失败 {len(failures)} 项", flush=True)
    if failures:
        print("失败任务：" + ", ".join(failures), flush=True)
        raise SystemExit(1)


if __name__ == "__main__":
    main()
