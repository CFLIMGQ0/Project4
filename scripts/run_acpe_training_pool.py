#!/usr/bin/env python3
"""在 204 主机三张空闲显卡上并行筛选两种轻量 ACPE 交互。"""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor, as_completed
import argparse
import os
from pathlib import Path
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[2]
RUNNER = ROOT / "src/scripts/run_acpe_system_interaction.py"
LOG_ROOT = ROOT / "outputs/acpe_autoresearch/logs"
GPUS = {"ct_rate": 0, "amos_mm": 2, "mr_rate_1k": 3}
VARIANTS = ("centroid", "attention_bias")


def execute(dataset: str, variant: str, fold: int) -> tuple[str, int]:
    name = f"train_{dataset}_{variant}_fold{fold}"
    folder = ROOT / "outputs/acpe_autoresearch/training" / variant / dataset
    marker = (folder / f"7_labels/fold_{fold}/amef_multimodal/result.json") if dataset == "amos_mm" else (folder / f"fold_{fold}/amef_multimodal/completed.json")
    if marker.exists():
        return name, 0
    env = os.environ.copy()
    env.update(CUDA_VISIBLE_DEVICES=str(GPUS[dataset]), OMP_NUM_THREADS="4", TOKENIZERS_PARALLELISM="false")
    LOG_ROOT.mkdir(parents=True, exist_ok=True)
    # 先载入标准库 statistics，避免 src/statistics.py 遮蔽训练循环所需的 statistics.mean。
    bootstrap = "import statistics,runpy,sys;sys.argv=[sys.argv[1],*sys.argv[2:]];runpy.run_path(sys.argv[0],run_name='__main__')"
    command = [sys.executable, "-u", "-c", bootstrap, str(RUNNER), "--dataset", dataset, "--variant", variant, "--fold", str(fold)]
    with (LOG_ROOT / f"{name}.log").open("a", encoding="utf-8") as stream:
        process = subprocess.run(command, cwd=ROOT, env=env, stdout=stream, stderr=subprocess.STDOUT)
    return name, process.returncode


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--datasets", default=",".join(GPUS), help="逗号分隔的数据集键")
    parser.add_argument("--variants", default=",".join(VARIANTS), help="逗号分隔的候选键")
    parser.add_argument("--folds", default="1", help="逗号分隔的折次")
    args = parser.parse_args()
    datasets = tuple(item.strip() for item in args.datasets.split(",") if item.strip())
    if any(item not in GPUS for item in datasets):
        parser.error("未知数据集键")
    variants = tuple(item.strip() for item in args.variants.split(",") if item.strip())
    if any(item not in VARIANTS for item in variants):
        parser.error("未知候选键")
    folds = tuple(int(item.strip()) for item in args.folds.split(",") if item.strip())
    if any(fold not in range(1, 6) for fold in folds):
        parser.error("折次必须为1至5")
    started = time.monotonic()
    tasks = [(dataset, variant, fold) for dataset in datasets for variant in variants for fold in folds]
    failures = []
    with ThreadPoolExecutor(max_workers=len(tasks)) as pool:
        pending = {pool.submit(execute, *task): task for task in tasks}
        for future in as_completed(pending):
            name, code = future.result()
            print(f"{name}: {'完成' if code == 0 else f'失败({code})'}", flush=True)
            if code:
                failures.append(name)
    print(f"{len(tasks)}个首折候选总用时 {time.monotonic()-started:.1f} 秒；失败 {len(failures)} 项", flush=True)
    if failures:
        print("失败任务：" + ", ".join(failures), flush=True)
        raise SystemExit(1)


if __name__ == "__main__":
    main()
