#!/usr/bin/env python3
"""新版文本的完整五折调度：每卡两个任务，继续已有图像特征提取。"""
from pathlib import Path
import run_abdomenatlas3_anonymous_table1 as experiment
import run_abdomenatlas3_pool as pool

if __name__ == "__main__":
    experiment.prepare_inputs()
    pool.OUT = experiment.OUT
    pool.initialize = experiment.base.initialize
    pool.aggregate = experiment.base.aggregate
    pool.TRAIN_RUNNER = Path(experiment.__file__).resolve()
    raise SystemExit(pool.main())
