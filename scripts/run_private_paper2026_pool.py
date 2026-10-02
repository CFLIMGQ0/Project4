#!/usr/bin/env python3
"""按本机可用 GPU 容量排队运行已完成缓存准备的私有 2026 基线折次。"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src/scripts"))
from prepare_private_paper_features import exam_rows, feature_path, DATASETS
from run_paper2026_baselines import DISPLAY

SCRIPT = ROOT / "src/scripts/run_private_paper2026_baselines.py"
OUT = ROOT / "outputs/paper_results/new_baselines"
GPUS = (0, 2, 3)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--datasets", required=True,
                        help="逗号分隔的已完成特征准备的四个数据集目录名")
    parser.add_argument("--max-per-gpu", type=int, default=2)
    parser.add_argument("--poll-seconds", type=int, default=10)
    args = parser.parse_args()
    datasets = [name.strip() for name in args.datasets.split(",") if name.strip()]
    if not datasets or any(name not in DATASETS for name in datasets):
        raise ValueError(f"无效数据集列表：{datasets}")
    for dataset in datasets:
        missing = [row["exam_dir"] for row in exam_rows(dataset)
                   if not feature_path(dataset, row["exam_dir"]).exists()]
        if missing:
            raise RuntimeError(f"{dataset} 缺少 {len(missing)} 个检查特征，不能启动训练")
    jobs = [(dataset, method, fold) for dataset in datasets
            for method in DISPLAY for fold in range(1, 6)
            if not (OUT / dataset / f"fold_{fold}" / method / "completed.json").exists()]
    log_root = OUT / "private_pool_logs"
    log_root.mkdir(parents=True, exist_ok=True)
    state_path = log_root / ("_".join(datasets) + "_state.json")
    pending = jobs[:]
    active: dict[int, tuple[subprocess.Popen, tuple[str, str, int], object]] = {}
    active_gpu: dict[int, int] = {}
    finished, failed = [], []
    while pending or active:
        for gpu in GPUS:
            # 活跃作业的 GPU 映射单独存储，避免外部显卡 1 的占用被调度器误用。
            count = sum(1 for pid in active if active_gpu[pid] == gpu)
            while count < args.max_per_gpu and pending:
                dataset, method, fold = pending.pop(0)
                logfile = log_root / f"{dataset}_{method}_fold{fold}.log"
                stream = logfile.open("w", encoding="utf-8")
                env = dict(os.environ, CUDA_VISIBLE_DEVICES=str(gpu), OMP_NUM_THREADS="2",
                           PAPER_MEMORY_FRACTION=".38")
                command = [sys.executable, "-u", str(SCRIPT), "--dataset", dataset,
                           "--method", method, "--fold", str(fold)]
                proc = subprocess.Popen(command, cwd=ROOT, env=env, stdout=stream,
                                        stderr=subprocess.STDOUT)
                active[proc.pid] = (proc, (dataset, method, fold), stream)
                active_gpu[proc.pid] = gpu
                count += 1
                print(f"GPU{gpu} 启动 {dataset}/{method}/fold{fold} pid={proc.pid}", flush=True)
        time.sleep(args.poll_seconds)
        for pid, (proc, job, stream) in list(active.items()):
            code = proc.poll()
            if code is None:
                continue
            stream.close()
            del active[pid]
            gpu = active_gpu.pop(pid)
            record = {"dataset": job[0], "method": job[1], "fold": job[2],
                      "gpu": gpu, "exit_code": code}
            (finished if code == 0 else failed).append(record)
            print(f"GPU{gpu} {'完成' if code == 0 else '失败'} {job}: exit={code}", flush=True)
        state_path.write_text(json.dumps({"pending": pending, "active": [
            {"pid": pid, "job": job, "gpu": active_gpu[pid]} for pid, (_, job, _) in active.items()],
            "finished": finished, "failed": failed}, ensure_ascii=False, indent=2)+"\n", encoding="utf-8")
    if failed:
        raise RuntimeError(f"{len(failed)} 个私有基线折次失败，日志在 {log_root}")
    print(f"已完成 {len(finished)} 个私有基线折次", flush=True)


if __name__ == "__main__":
    main()
