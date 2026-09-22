#!/usr/bin/env python3
"""查看实际可见的六张GPU、任务进度及已有分类结果，不以队列启动代替实验完成。"""
from __future__ import annotations

import json
import time

from ct_six_gpu_queue import ROOT, DEFAULT_QUEUE, inventory, save


def main():
    folder = DEFAULT_QUEUE.parent
    status = json.loads((folder / "status.json").read_text())
    print(f"队列状态：{status['state']}，完成{status['completed']}/{status['total']}个步骤；状态年龄{time.time()-status['updated']:.0f}秒")
    devices = inventory()
    for host, record in devices.items():
        if not record["visible"]:
            print(f"警告：{host}不可见：{record['error']}")
        for device in record["devices"]:
            active = [name for name, task in status["active"].items() if task["host"] == host and task["gpu"] == device["index"]]
            print(f"{host} GPU{device['index']}：可用{device['free_mib']}MiB，利用率{device['utilization']}%，我方活动任务{len(active)}，{device['uuid']}")
    snapshot = {"timestamp": time.time(), "queue": status, "inventory": devices, "experiments": {}}
    for relative, expected in (("missing50_raw_hidden_fivefold", 10), ("missing75_raw_hidden_fivefold", 10), ("position_baselines", 30)):
        experiment = ROOT / "outputs/ct_rate_680" / relative
        completed = len(list(experiment.glob("*/fold_*/*/completed.json")))
        print(f"{relative}：分类折完成{completed}/{expected}")
        record = {"completed_folds": completed, "expected_folds": expected}
        sampling = experiment / "data/sampling.json"
        if sampling.exists():
            manifest = json.loads(sampling.read_text())
            count = len(list((experiment / "data/features").glob("????.npz")))
            print(f"  特征回收{count}/{manifest['retained_cases']}例；排除{manifest['excluded_cases']}例")
            record.update(feature_cases=count, retained_cases=manifest["retained_cases"], excluded_cases=manifest["excluded_cases"])
        summary = experiment / "comparison.json"
        if summary.exists():
            records = json.loads(summary.read_text())
            for result in records:
                print(f"  {result['model']}：Macro-F1 {result['macro_f1_mean']:.4f} ± {result['macro_f1_std']:.4f}")
            record["results"] = records
        snapshot["experiments"][relative] = record
    if status["failed"]:
        print("失败任务：" + ", ".join(status["failed"]))
    save(folder / "latest_check.json", snapshot)


if __name__ == "__main__":
    main()
