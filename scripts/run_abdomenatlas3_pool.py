#!/usr/bin/env python3
"""本机两任务/卡：图像准备与文本训练并行，缓存完整后自动执行全部图像和图文五折。"""
from __future__ import annotations

import argparse
import fcntl
import json
import os
from pathlib import Path
import subprocess
import sys
import time

from prepare_abdomenatlas3_table1 import ROOT, OUT as INPUT, save_json
from run_abdomenatlas3_table1 import OUT, MODELS, TEXT_MODELS, initialize, aggregate

TRAIN_RUNNER = ROOT / "src/scripts/run_abdomenatlas3_table1.py"
MAX_TASKS_PER_GPU = 3


def gpu_free():
    result = subprocess.run(["nvidia-smi", "--query-gpu=index,memory.free", "--format=csv,noheader,nounits"],
                            text=True, capture_output=True, check=True, timeout=15)
    return {int(line.split(",")[0]): int(line.split(",")[1]) for line in result.stdout.splitlines()}


class ExistingFeatureProcess:
    """接管本项目仍在运行的任务，重启调度器时保留训练进度。"""
    def __init__(self, pid, expected=b"prepare_abdomenatlas3_table1.py"):
        self.pid = pid
        self.expected = expected

    def poll(self):
        path = Path(f"/proc/{self.pid}/cmdline")
        try:
            command = path.read_bytes()
        except FileNotFoundError:
            return 0
        return None if self.expected in command else 0


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gpus", type=int, nargs="+", default=[2, 3])
    parser.add_argument("--extra-jobs", type=Path, help="并行执行的额外任务清单，同样计入每卡两个任务上限")
    args = parser.parse_args()
    assert len(args.gpus) == len(set(args.gpus)) and args.gpus
    OUT.mkdir(parents=True, exist_ok=True)
    with (OUT / "pool.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        protocol = initialize()
        expected = len(json.loads((INPUT / "samples.json").read_text()))
        logs = OUT / "logs"
        logs.mkdir(exist_ok=True)
        jobs = [{"kind": "features", "index": i, "key": f"features_{i}"} for i in range(2)]
        extras = json.loads(args.extra_jobs.read_text()) if args.extra_jobs else []
        assert all(job["kind"] == "external" for job in extras)
        jobs += extras
        # 优先出文本结果；完整图像缓存就绪后首先运行ALM-MIL。
        order = ["amef_multimodal", *TEXT_MODELS, *[k for k in MODELS if k not in TEXT_MODELS and k != "amef_multimodal"]]
        jobs += [{"kind": "train", "model": model, "fold": fold, "key": f"{model}_fold_{fold}"}
                 for fold in range(1, 6) for model in order]
        active, failed, attempts = [], {}, {}
        last_training_kind = None
        start = time.time()
        previous_path = OUT / "pool_state.json"
        previous = json.loads(previous_path.read_text()) if previous_path.exists() else {}

        def complete(job):
            if job["kind"] == "features":
                path = INPUT / f"feature_worker_{job['index']}.json"
                return path.exists() and json.loads(path.read_text()).get("status") == "complete"
            if job["kind"] == "external":
                path = Path(job["completion_path"])
                if not path.exists():
                    return False
                assert json.loads(path.read_text())["protocol_sha256"] == job["protocol_sha256"]
                return True
            path = OUT / f"fold_{job['fold']}" / job["model"] / "completed.json"
            if path.exists():
                assert json.loads(path.read_text())["protocol_sha256"] == protocol["protocol_sha256"]
                return True
            return False

        for job in jobs[:2]:
            path = INPUT / f"feature_worker_{job['index']}.json"
            if not path.exists() or complete(job):
                continue
            state = json.loads(path.read_text())
            pid = state.get("pid")
            if not pid or not Path(f"/proc/{pid}/cmdline").exists():
                continue
            command = Path(f"/proc/{pid}/cmdline").read_bytes()
            if b"prepare_abdomenatlas3_table1.py" not in command:
                continue
            environment = Path(f"/proc/{pid}/environ").read_bytes().split(b"\x00")
            gpu = int(next(v.split(b"=", 1)[1] for v in environment if v.startswith(b"CUDA_VISIBLE_DEVICES=")))
            assert gpu in args.gpus, "运行中的特征进程不在本次GPU范围内"
            old_log = ROOT / "outputs/abdomenatlas3_mini/table1_fivefold_original_uit/logs" / f"{job['key']}.log"
            active.append({"job": job, "gpu": gpu, "process": ExistingFeatureProcess(pid),
                           "handle": None, "log": old_log, "started_unix": start})
            attempts[job["key"]] = 1
            print(f"继续使用 GPU{gpu} pid={pid} {job['key']}", flush=True)

        # 新增实验时仅替换调度器，现有训练子进程按PID、参数及GPU继续接管。
        by_key = {job["key"]: job for job in jobs}
        for saved in previous.get("active", []):
            if saved["job"]["kind"] == "features":
                continue
            pid, gpu = saved["pid"], saved["gpu"]
            path = Path(f"/proc/{pid}/cmdline")
            if not path.exists():
                continue
            command = path.read_bytes()
            if not command:
                continue
            job = by_key.get(saved["job"]["key"])
            assert job is not None and job == saved["job"], "运行中任务与新队列定义不一致"
            command_token = (job["arguments"][0] if job["kind"] == "external" else str(TRAIN_RUNNER)).encode()
            assert command_token in command and gpu in args.gpus, "不接管来源不明的进程"
            if job["kind"] == "external":
                argv = command.split(b"\x00")
                assert all(str(arg).encode() in argv for arg in job["arguments"])
            env = Path(f"/proc/{pid}/environ").read_bytes().split(b"\x00")
            visible = next(v.split(b"=", 1)[1] for v in env if v.startswith(b"CUDA_VISIBLE_DEVICES="))
            assert int(visible) == gpu
            assert sum(r["gpu"] == gpu for r in active) < MAX_TASKS_PER_GPU
            active.append({"job": job, "gpu": gpu, "process": ExistingFeatureProcess(pid, command_token),
                           "handle": None, "log": Path(saved["log"]), "started_unix": saved["started_unix"]})
            attempts[job["key"]] = 1
            print(f"继续训练 GPU{gpu} pid={pid} {job['key']}", flush=True)

        while True:
            changed = False
            for running in list(active):
                code = running["process"].poll()
                if code is None:
                    continue
                if running["handle"] is not None:
                    running["handle"].close()
                active.remove(running)
                job = running["job"]
                if code or not complete(job):
                    if attempts[job["key"]] >= 2:
                        failed[job["key"]] = {"exit_code": code, "log": str(running["log"])}
                    print(f"任务退出：{job['key']} code={code}，详见{running['log']}", flush=True)
                else:
                    print(f"任务完成：{job['key']}", flush=True)
                    changed = True
            cache_ready = all(complete(jobs[i]) for i in range(2))
            # 外部补做校验后已完成的任务不再列作失败。
            for job in jobs:
                if job["key"] in failed and complete(job):
                    failed.pop(job["key"])
            if cache_ready:
                count = sum(1 for p in (INPUT / "features").glob("*.npz") if not p.name.endswith(".tmp.npz"))
                assert count == expected, ("特征缓存不完整", count, expected)
            if changed:
                aggregate()
            stopped = (OUT / "STOP_AFTER_CURRENT").exists()
            waiting = [j for j in jobs if not complete(j) and j["key"] not in failed
                       and j["key"] not in {r["job"]["key"] for r in active}]
            free = gpu_free()
            if not stopped:
                for gpu in args.gpus:
                    while sum(r["gpu"] == gpu for r in active) < MAX_TASKS_PER_GPU:
                        if free.get(gpu, 0) < 6500:
                            break
                        eligible = [j for j in waiting if j["kind"] in ("features", "external") or j.get("model") in TEXT_MODELS or cache_ready]
                        if not eligible:
                            break
                        # 特征提取分散在两张卡，给每张卡保留一个训练槽位。
                        feature = [j for j in eligible if j["kind"] == "features"]
                        if feature and any(r["gpu"] == gpu and r["job"]["kind"] == "features" for r in active):
                            eligible = [j for j in eligible if j["kind"] != "features"]
                            if not eligible:
                                break
                        job = eligible[0]
                        # 缓存齐全后交替分配私有补跑和新数据集，避免长队列阻塞后者。
                        if job["kind"] != "features":
                            preferred = "train" if last_training_kind == "external" else "external"
                            preferred_jobs = [item for item in eligible if item["kind"] == preferred]
                            if preferred_jobs:
                                job = preferred_jobs[0]
                            last_training_kind = job["kind"]
                        env = os.environ.copy()
                        env.update(CUDA_VISIBLE_DEVICES=str(gpu), OMP_NUM_THREADS="2", MKL_NUM_THREADS="2",
                                   OPENBLAS_NUM_THREADS="2", TOKENIZERS_PARALLELISM="false", PYTHONUNBUFFERED="1")
                        if job["kind"] == "features":
                            cmd = [sys.executable, "-u", str(ROOT / "src/scripts/prepare_abdomenatlas3_table1.py"),
                                   "--worker-index", str(job["index"]), "--workers", "2"]
                        elif job["kind"] == "external":
                            cmd = [sys.executable, "-u", *job["arguments"]]
                        else:
                            cmd = [sys.executable, "-u", str(TRAIN_RUNNER),
                                   "--model", job["model"], "--fold", str(job["fold"])]
                        logfile = logs / f"{job['key']}.log"
                        handle = logfile.open("a")
                        process = subprocess.Popen(cmd, cwd=ROOT, env=env, stdout=handle, stderr=subprocess.STDOUT)
                        active.append({"job": job, "gpu": gpu, "process": process, "handle": handle,
                                       "log": logfile, "started_unix": time.time()})
                        attempts[job["key"]] = attempts.get(job["key"], 0)+1
                        waiting.remove(job)
                        free[gpu] -= 6500
                        print(f"启动 GPU{gpu} pid={process.pid} {job['key']}", flush=True)
            completed = sum(complete(j) for j in jobs if j["kind"] == "train")
            external_completed = sum(complete(j) for j in extras)
            all_done = completed == 110 and external_completed == len(extras)
            count = sum(1 for p in (INPUT / "features").glob("*.npz") if not p.name.endswith(".tmp.npz"))
            state = {"status": "running", "pid": os.getpid(), "gpus": args.gpus,
                     "max_tasks_per_gpu": MAX_TASKS_PER_GPU,
                     "completed_fold_jobs": completed, "expected_fold_jobs": 110,
                     "completed_extra_jobs": external_completed, "expected_extra_jobs": len(extras),
                     "feature_cases": count, "expected_cases": expected, "cache_ready": cache_ready,
                     "active": [{"job": r["job"], "gpu": r["gpu"], "pid": r["process"].pid,
                                 "started_unix": r["started_unix"], "log": str(r["log"])} for r in active],
                     "failed": failed, "started_unix": start, "updated_unix": time.time()}
            can_wait = any(j["kind"] in ("features", "external") or j.get("model") in TEXT_MODELS or cache_ready for j in waiting)
            if not active and (stopped or not can_wait):
                state["status"] = "complete" if all_done else "paused" if stopped else "incomplete"
                save_json(OUT / "pool_state.json", state)
                aggregate()
                return 0 if all_done or stopped else 1
            if not active:
                state["status"] = "waiting_for_gpu_memory"
            save_json(OUT / "pool_state.json", state)
            time.sleep(15)


if __name__ == "__main__":
    raise SystemExit(main())
