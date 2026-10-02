#!/usr/bin/env python3
"""并行评估 ACPE 研究候选在固定删片条件下的分类与位置指标。"""

from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import os
from pathlib import Path
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[2]
RUNNER = ROOT / "src/scripts/evaluate_acpe_candidate_deletion.py"
LOGS = ROOT / "outputs/acpe_autoresearch/logs"
GPUS = {"ct_rate": 0, "amos_mm": 2, "mr_rate_1k": 3}
VARIANTS = ("centroid", "attention_bias", "confidence_gate", "confidence_half", "eta_scale", "block_aug")


def execute(dataset: str, variant: str, fold: int, protocol: str):
    name = f"deletion_{dataset}_{variant}_fold{fold}_{protocol}"
    destination = ROOT / "outputs/acpe_autoresearch/deletion" / protocol / variant / dataset / f"fold_{fold}.json"
    if destination.exists():
        return name, 0
    env = os.environ.copy()
    env.update(CUDA_VISIBLE_DEVICES=str(GPUS[dataset]), OMP_NUM_THREADS="2")
    command = [sys.executable, "-u", str(RUNNER), "--dataset", dataset, "--variant", variant,
               "--fold", str(fold), "--index-protocol", protocol]
    LOGS.mkdir(parents=True, exist_ok=True)
    with (LOGS / f"{name}.log").open("w", encoding="utf-8") as stream:
        process = subprocess.run(command, cwd=ROOT, env=env, stdout=stream, stderr=subprocess.STDOUT)
    return name, process.returncode


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--datasets", default=",".join(GPUS))
    parser.add_argument("--variants", default="centroid,attention_bias")
    parser.add_argument("--protocols", default="reindexed")
    parser.add_argument("--folds", default="1,2")
    args = parser.parse_args()
    datasets = tuple(x.strip() for x in args.datasets.split(",") if x.strip())
    variants = tuple(x.strip() for x in args.variants.split(",") if x.strip())
    protocols = tuple(x.strip() for x in args.protocols.split(",") if x.strip())
    folds = tuple(int(x.strip()) for x in args.folds.split(",") if x.strip())
    if any(d not in GPUS for d in datasets) or any(v not in VARIANTS for v in variants) or any(p not in {"reindexed", "original"} for p in protocols) or any(f not in range(1, 6) for f in folds):
        parser.error("存在无效的候选、索引协议或折次")
    tasks = [(dataset, variant, fold, protocol) for dataset in datasets for variant in variants
             for fold in folds for protocol in protocols]
    started = time.monotonic()
    failures = []
    with ThreadPoolExecutor(max_workers=min(12, len(tasks))) as pool:
        pending = {pool.submit(execute, *task): task for task in tasks}
        for future in as_completed(pending):
            name, code = future.result()
            print(f"{name}: {'完成' if code == 0 else f'失败({code})'}", flush=True)
            if code:
                failures.append(name)
    print(f"{len(tasks)}个删片评价用时 {time.monotonic()-started:.1f} 秒；失败 {len(failures)} 项", flush=True)
    if failures:
        print("失败任务：" + ", ".join(failures), flush=True)
        raise SystemExit(1)


if __name__ == "__main__":
    main()
