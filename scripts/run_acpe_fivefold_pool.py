#!/usr/bin/env python3
"""以隔离导入和每卡单任务方式续跑已有 ACPE 五折协议，不改动冻结的训练代码。"""

import os
import sys

# 项目的 src/statistics.py 与标准库重名；先在隔离解释器中载入标准库。
if __name__ == "__main__" and not sys.flags.isolated:
    os.execv(sys.executable, [sys.executable, "-I", os.path.abspath(__file__), *sys.argv[1:]])

import argparse
from datetime import datetime
import fcntl
import json
from pathlib import Path
import runpy
import shutil
import signal
import statistics
import subprocess
import time
import traceback

assert callable(statistics.mean) and callable(statistics.stdev)
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
from acpe_fivefold.common import OUT, BASE, PYTHON, DATASETS, read, write, digest
from acpe_fivefold.controller import aggregate, extend, specs, free_memory
from acpe_fivefold.plan import MAX_CANDIDATES


def log(message):
    print(f"[{datetime.now():%Y-%m-%d %H:%M:%S}] {message}", flush=True)


def refresh(candidates, plan, gpus):
    summary = aggregate(candidates, plan)
    summary["execution_gpus"] = {"204": gpus}
    summary["statistics_module"] = statistics.__file__
    write(OUT / "summary.json", summary)
    result_path = OUT / "results.md"
    result_path.write_text(result_path.read_text().replace(
        "当前只用204主机GPU0。", f"当前只用204主机GPU{'、'.join(map(str, gpus))}，每卡一个本轮训练进程。"))
    return summary


def check_frozen_sources(plan):
    for name, expected in plan["baseline_source_hashes"].items():
        assert digest(BASE / name) == expected, name
    for name, expected in plan["source_hashes"].items():
        assert digest(ROOT / "src/acpe_fivefold" / name) == expected, name


def launch_job(spec, gpu):
    path = OUT / "jobs" / (spec["id"] + ".json")
    if path.exists():
        assert read(path) == spec
    else:
        write(path, spec)
    env = os.environ.copy()
    env.update(CUDA_VISIBLE_DEVICES=str(gpu), OMP_NUM_THREADS="2", MKL_NUM_THREADS="2",
               OPENBLAS_NUM_THREADS="2", PYTHONUNBUFFERED="1")
    with (OUT / "logs" / (spec["id"] + ".log")).open("a") as handle:
        process = subprocess.Popen(
            [PYTHON, "-I", str(Path(__file__).resolve()), "--train-job", str(path)],
            cwd=ROOT, env=env, stdout=handle, stderr=subprocess.STDOUT)
    log(f"启动 {spec['id']}，GPU{gpu}，PID={process.pid}")
    return {"id": spec["id"], "gpu": gpu, "process": process, "stop_requested": False}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--aggregate", action="store_true")
    parser.add_argument("--train-job", type=Path)
    args = parser.parse_args()
    if args.train_job:
        sys.argv = ["acpe_fivefold.train", "--job", str(args.train_job)]
        runpy.run_module("acpe_fivefold.train", run_name="__main__")
        return
    plan, candidates = read(OUT / "protocol.json"), read(OUT / "candidates.json")
    check_frozen_sources(plan)
    policy = read(OUT / "control.json")
    gpus = sorted(set(policy.get("allowed_gpus", {}).get("204", [])))
    if args.aggregate:
        summary = refresh(candidates, plan, gpus)
        print(json.dumps({"completed_folds": summary["completed_folds"],
                          "completed_candidates": summary["completed_candidates"],
                          "statistics_module": statistics.__file__}, ensure_ascii=False))
        return
    lock = (OUT / "controller.lock").open("a")
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    state = read(OUT / "state.json")
    active = {}
    stopping = False
    previous_aggregate = 0
    summary = None
    write(OUT / "controller_identity.json", {"pid": os.getpid(), "started": time.time(),
          "host": "204", "gpus": gpus, "launcher": str(Path(__file__).resolve()),
          "launcher_sha256": digest(__file__), "statistics_module": statistics.__file__})

    def request_stop(signum, _frame):
        nonlocal stopping
        stopping = True
        log(f"收到信号 {signum}，等待本轮任务保存断点后退出。")

    signal.signal(signal.SIGTERM, request_stop)
    signal.signal(signal.SIGINT, request_stop)
    log(f"恢复既有五折队列，允许 GPU={gpus}，标准库 statistics={statistics.__file__}")
    try:
        while True:
            policy = read(OUT / "control.json")
            gpus = sorted(set(policy.get("allowed_gpus", {}).get("204", [])))
            paused = policy.get("paused", False) or stopping
            changed = False
            for gpu, job in list(active.items()):
                process = job["process"]
                if process.poll() is not None:
                    code = process.returncode
                    if code != 0:
                        state["failures"][job["id"]] = {"returncode": code, "time": time.time()}
                    else:
                        state["failures"].pop(job["id"], None)
                    log(f"结束 {job['id']}，GPU{gpu}，exit={code}")
                    del active[gpu]
                    changed = True
                elif (paused or gpu not in gpus) and not job["stop_requested"]:
                    process.send_signal(signal.SIGTERM)
                    job["stop_requested"] = True
                    log(f"请求 {job['id']} 保存断点并释放 GPU{gpu}")
            if changed or time.time() - previous_aggregate > 30:
                summary = refresh(candidates, plan, gpus)
                extend(candidates, summary, state)
                previous_aggregate = time.time()
                log(f"进度 {summary['completed_folds']}/{15 * len(candidates)} 折任务，"
                    f"完整方案 {summary['completed_candidates']}/{len(candidates)}")
            state.update(updated=time.time(), paused=paused, allowed_gpus={"204": gpus})
            state["active_jobs"] = [{"id": job["id"], "gpu": gpu, "pid": job["process"].pid}
                                    for gpu, job in active.items()]
            state["active"] = state["active_jobs"][0] if state["active_jobs"] else None
            write(OUT / "state.json", state)
            if stopping and not active:
                break
            if not paused:
                assert all((OUT / "features" / ds / "ready.json").exists() for ds in DATASETS)
                memory = free_memory()
                if shutil.disk_usage(ROOT).free / 2**30 < 12:
                    log("项目盘可用空间不足 12 GiB，暂不启动新任务。")
                    time.sleep(5)
                    continue
                active_ids = {job["id"] for job in active.values()}
                pending = [spec for spec in specs(candidates, plan)
                           if not (OUT / "runs" / spec["id"] / "result.json").exists()
                           and spec["id"] not in active_ids
                           and state["attempts"].get(spec["id"], 0) < 3]
                depth_ids = {c["id"] for c in candidates if c["kind"] == "depth"}
                pending.sort(key=lambda spec: 0 if spec["candidate"] in depth_ids else 1)
                for gpu in gpus:
                    if gpu in active or memory.get(gpu, 0) < 1800 or not pending:
                        continue
                    spec = pending.pop(0)
                    active[gpu] = launch_job(spec, gpu)
                    state["attempts"][spec["id"]] = state["attempts"].get(spec["id"], 0) + 1
                    state["active_jobs"] = [{"id": job["id"], "gpu": card, "pid": job["process"].pid}
                                            for card, job in active.items()]
                    state["active"] = state["active_jobs"][0]
                    write(OUT / "state.json", state)
                if not pending and not active and summary["completed_folds"] == 15 * len(candidates):
                    write(OUT / "completion.json", {"time": time.time(), "summary": summary,
                          "test_evaluated": False, "candidate_limit": MAX_CANDIDATES})
                    log("当前所有候选均已完成。")
                    break
            time.sleep(5)
    except BaseException:
        # 调度异常时让自己的训练保存退出，避免留下无人管理的 CUDA 进程。
        for job in active.values():
            if job["process"].poll() is None:
                job["process"].send_signal(signal.SIGTERM)
        raise


if __name__ == "__main__":
    main()
