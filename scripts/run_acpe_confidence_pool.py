#!/usr/bin/env python3
"""在 204 主机三张空闲显卡上并行完成可信度门前两折筛选。"""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor, as_completed
import argparse
import os
from pathlib import Path
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[2]
LOGS = ROOT / "outputs/acpe_autoresearch/logs"
GPUS = {"ct_rate": 0, "amos_mm": 2, "mr_rate_1k": 3}
VARIANT = "confidence_gate"


def execute(dataset: str, fold: int):
    name = f"train_{dataset}_{VARIANT}_fold{fold}"
    folder = ROOT / "outputs/acpe_autoresearch/training" / VARIANT / dataset
    marker = (folder / f"7_labels/fold_{fold}/amef_multimodal/result.json") if dataset == "amos_mm" else (folder / f"fold_{fold}/amef_multimodal/completed.json")
    if marker.exists():
        return name, 0
    env = os.environ.copy()
    env.update(CUDA_VISIBLE_DEVICES=str(GPUS[dataset]), OMP_NUM_THREADS="4", TOKENIZERS_PARALLELISM="false")
    bootstrap = "import statistics,runpy,sys;sys.argv=[sys.argv[1],*sys.argv[2:]];runpy.run_path(sys.argv[0],run_name='__main__')"
    runner_name = {
        "confidence_gate": "run_acpe_confidence_gate.py",
        "confidence_half": "run_acpe_confidence_gate.py",
        "eta_scale": "run_acpe_eta_scale.py",
        "block_aug": "run_acpe_block_augmentation.py",
    }[VARIANT]
    runner = ROOT / "src/scripts" / runner_name
    command = [sys.executable, "-u", "-c", bootstrap, str(runner), "--dataset", dataset, "--fold", str(fold)]
    if VARIANT == "confidence_half":
        command += ["--variant", VARIANT]
    LOGS.mkdir(parents=True, exist_ok=True)
    with (LOGS / f"{name}.log").open("a", encoding="utf-8") as stream:
        process = subprocess.run(command, cwd=ROOT, env=env, stdout=stream, stderr=subprocess.STDOUT)
    return name, process.returncode


def main() -> None:
    global VARIANT
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--variant", choices=("confidence_gate", "confidence_half", "eta_scale", "block_aug"), default="confidence_gate")
    parser.add_argument("--datasets", default=",".join(GPUS), help="逗号分隔的数据集键")
    parser.add_argument("--folds", default="1,2", help="逗号分隔的折次")
    args = parser.parse_args()
    VARIANT = args.variant
    datasets = tuple(item.strip() for item in args.datasets.split(",") if item.strip())
    folds = tuple(int(item.strip()) for item in args.folds.split(",") if item.strip())
    if any(dataset not in GPUS for dataset in datasets):
        parser.error("未知数据集键")
    if any(fold not in range(1, 6) for fold in folds):
        parser.error("折次必须为1至5")
    started = time.monotonic()
    tasks = [(dataset, fold) for dataset in datasets for fold in folds]
    failures = []
    with ThreadPoolExecutor(max_workers=len(tasks)) as pool:
        pending = {pool.submit(execute, *task): task for task in tasks}
        for future in as_completed(pending):
            name, code = future.result()
            print(f"{name}: {'完成' if code == 0 else f'失败({code})'}", flush=True)
            if code:
                failures.append(name)
    print(f"{VARIANT} {len(tasks)}个训练任务用时 {time.monotonic()-started:.1f} 秒；失败 {len(failures)} 项", flush=True)
    if failures:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
