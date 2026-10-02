#!/usr/bin/env python3
"""本机 GPU 0/2/3 上并行准备四个私有子数据集的缓存图像特征。"""

from __future__ import annotations

import json
import os
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[2]
SCRIPT = ROOT / "src/scripts/prepare_private_paper_features.py"
DATASETS = ("regular_white_light", "chromoscopic", "surgical", "ultrasound")
GPUS = (0, 2, 3)
LOG = ROOT / "outputs/paper_results/private_feature_logs"
STATE = LOG / "launch_state.json"


def main() -> None:
    LOG.mkdir(parents=True, exist_ok=True)
    if STATE.exists():
        raise RuntimeError(f"启动记录已存在，先检查旧进程：{STATE}")
    children = []
    for dataset in DATASETS:
        for index, gpu in enumerate(GPUS):
            logfile = LOG / f"{dataset}_worker{index}.log"
            env = dict(os.environ, CUDA_VISIBLE_DEVICES=str(gpu), OMP_NUM_THREADS="2")
            command = [sys.executable, "-u", str(SCRIPT), "--dataset", dataset,
                       "--worker-index", str(index), "--workers", "3"]
            stream = logfile.open("w", encoding="utf-8")
            proc = subprocess.Popen(command, cwd=ROOT, env=env, stdout=stream,
                                    stderr=subprocess.STDOUT, start_new_session=True)
            stream.close()
            children.append({"dataset": dataset, "worker_index": index, "gpu": gpu,
                             "pid": proc.pid, "log": str(logfile.relative_to(ROOT))})
    STATE.write_text(json.dumps(children, ensure_ascii=False, indent=2)+"\n", encoding="utf-8")
    print(f"已启动 {len(children)} 个特征进程；单 GPU 4 个私有提取进程。", flush=True)


if __name__ == "__main__":
    main()
