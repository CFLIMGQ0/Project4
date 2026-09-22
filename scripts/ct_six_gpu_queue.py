#!/usr/bin/env python3
"""204四卡与202两卡的显存动态队列，允许同卡多任务并显式报告离线GPU。"""
from __future__ import annotations

import argparse
import concurrent.futures
import fcntl
import hashlib
import json
import os
from pathlib import Path
import shlex
import shutil
import subprocess
import sys
import time
import traceback

import run_ctrate_680_amef_no_pe_gpu_pool as network

ROOT = Path(__file__).resolve().parents[2]
PYTHON = "/home/Lim/conda/envs/myenv/bin/python"
REMOTE_ROOT = "/new_data/Lim/ct_position_suite_20260919"
DEFAULT_QUEUE = ROOT / "outputs/ct_rate_680/position_suite/queue.json"


def save(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + f".{os.getpid()}.tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n")
    temporary.replace(path)


def run(command, **kwargs):
    return subprocess.run(command, check=True, text=True, timeout=kwargs.pop("timeout", 900), **kwargs)


def remote(command, **kwargs):
    return run([*network.SSH, network.REMOTE_HOST, command], **kwargs)


def sync(source, destination):
    run(["rsync", "-a", "--partial", "--timeout=180", "-e", shlex.join(network.SSH),
         str(source), str(destination)], timeout=900)


def inventory():
    command = "nvidia-smi --query-gpu=index,uuid,name,memory.free,memory.total,utilization.gpu --format=csv,noheader,nounits"
    result = {}
    for host, expected in (("204", 4), ("202", 2)):
        try:
            output = (run(shlex.split(command), capture_output=True, timeout=20) if host == "204"
                      else remote(command, capture_output=True, timeout=25)).stdout
            devices = []
            for line in output.splitlines():
                index, uuid, name, free, total, utilization = [field.strip() for field in line.split(",")]
                devices.append(dict(index=int(index), uuid=uuid, name=name, free_mib=int(free),
                                    total_mib=int(total), utilization=int(utilization)))
            if {device["index"] for device in devices} != set(range(expected)):
                raise RuntimeError(f"应有{expected}张卡，实际索引{[item['index'] for item in devices]}")
            result[host] = {"visible": True, "devices": devices}
        except Exception as error:
            result[host] = {"visible": False, "devices": [], "error": str(error)}
            print(f"警告：{host}显卡无法完整访问：{error}", flush=True)
    return result


def digest(job):
    return hashlib.sha256(json.dumps(job, sort_keys=True).encode()).hexdigest()


def marker(folder, job):
    return folder / "done" / f"{job['id']}.json"


def completed(folder, job):
    path = marker(folder, job)
    if not path.exists():
        return False
    if json.loads(path.read_text())["job_sha256"] != digest(job):
        raise ValueError(f"任务定义变化，禁止混用完成标记：{job['id']}")
    return all((ROOT / name).is_file() for name in job["artifacts"])


def disk_bytes(host):
    if host == "204":
        return shutil.disk_usage(ROOT).free
    return int(remote(f"df -Pk {shlex.quote(REMOTE_ROOT)} | tail -1",
                      capture_output=True, timeout=20).stdout.split()[3]) * 1024


def execute(folder, job, host, gpu, attempt):
    log = folder / "logs" / f"{job['id']}_{host}_gpu{gpu}_attempt{attempt}.log"
    log.parent.mkdir(parents=True, exist_ok=True)
    temporary_raw = f"{REMOTE_ROOT}/datasets/queue_raw/{job['id']}"
    started = time.time()
    with log.open("a") as stream:
        try:
            command = [PYTHON, "-u", *job["command"]]
            if host == "202":
                for relative in job.get("sync_inputs", []):
                    source = ROOT / relative
                    target = f"{REMOTE_ROOT}/{relative}"
                    remote("mkdir -p " + shlex.quote(target if source.is_dir() else str(Path(target).parent)))
                    sync(str(source) + "/" if source.is_dir() else source,
                         f"{network.REMOTE_HOST}:{target}/" if source.is_dir() else f"{network.REMOTE_HOST}:{target}")
                raw_files = job.get("raw_files", [])
                if raw_files:
                    size = sum((ROOT / "datasets/ct_rate_680" / name).stat().st_size for name in raw_files)
                    if disk_bytes(host) < size + 8 * 1024**3:
                        raise RuntimeError("202暂存原CT所需空间不足，保留至少8GiB余量")
                    for name in raw_files:
                        target = f"{temporary_raw}/{name}"
                        remote("mkdir -p " + shlex.quote(str(Path(target).parent)))
                        sync(ROOT / "datasets/ct_rate_680" / name, f"{network.REMOTE_HOST}:{target}")
                    command += ["--raw-root", temporary_raw]
                command = [*network.SSH, network.REMOTE_HOST,
                           "cd " + shlex.quote(REMOTE_ROOT) + " && exec " + shlex.join([
                               "timeout", "--kill-after=30s", str(job.get("timeout_seconds", 7200)),
                               "env", f"CUDA_VISIBLE_DEVICES={gpu}", "OMP_NUM_THREADS=2",
                               "TOKENIZERS_PARALLELISM=false", *command])]
            environment = os.environ.copy()
            environment.update(CUDA_VISIBLE_DEVICES=str(gpu), OMP_NUM_THREADS="2", TOKENIZERS_PARALLELISM="false")
            stream.write(f"{time.strftime('%F %T')} {host} GPU{gpu}\n")
            stream.flush()
            process = subprocess.Popen(command, cwd=ROOT, env=environment, stdout=stream, stderr=subprocess.STDOUT)
            save(folder / "processes" / f"{job['id']}.json", {"pid": process.pid, "host": host, "gpu": gpu,
                                                           "started": started, "command": command})
            try:
                returncode = process.wait(timeout=job.get("timeout_seconds", 7200) + 60)
            except subprocess.TimeoutExpired:
                process.terminate()
                try:
                    process.wait(timeout=30)
                except subprocess.TimeoutExpired:
                    process.kill()
                    process.wait(timeout=30)
                raise
            if returncode != 0:
                raise RuntimeError(f"任务退出码{returncode}，详见{log}")
            if host == "202":
                for relative in job["artifacts"]:
                    destination = ROOT / relative
                    destination.parent.mkdir(parents=True, exist_ok=True)
                    sync(f"{network.REMOTE_HOST}:{REMOTE_ROOT}/{relative}", destination)
                for relative in job.get("sync_output_dirs", []):
                    destination = ROOT / relative
                    destination.mkdir(parents=True, exist_ok=True)
                    sync(f"{network.REMOTE_HOST}:{REMOTE_ROOT}/{relative}/", str(destination) + "/")
                if raw_files:
                    remote("rm -rf -- " + shlex.quote(temporary_raw))
            missing = [name for name in job["artifacts"] if not (ROOT / name).is_file()]
            if missing:
                raise RuntimeError(f"任务未生成全部结果：{missing}")
            value = {"job_sha256": digest(job), "host": host, "gpu": gpu, "attempt": attempt,
                     "started": started, "finished": time.time(), "wall_seconds": time.time() - started,
                     "log": str(log), "success": True}
            save(marker(folder, job), value)
            return value
        except Exception:
            detail = traceback.format_exc()
            stream.write(detail)
            if host == "202" and job.get("raw_files"):
                try:
                    remote("rm -rf -- " + shlex.quote(temporary_raw))
                except Exception as cleanup_error:
                    stream.write(f"\nraw临时目录清理失败：{cleanup_error}\n")
            return {"success": False, "error": detail, "host": host, "gpu": gpu, "attempt": attempt,
                    "log": str(log), "finished": time.time()}


def stage(queue):
    remote("mkdir -p " + shlex.quote(REMOTE_ROOT + "/src"))
    if disk_bytes("202") < 8 * 1024**3:
        raise RuntimeError("202磁盘不足8GiB，拒绝继续占用")
    run(["rsync", "-a", "--exclude=.git", "--exclude=__pycache__", "--timeout=180", "-e",
         shlex.join(network.SSH), str(ROOT / "src") + "/", f"{network.REMOTE_HOST}:{REMOTE_ROOT}/src/"])
    for relative in queue.get("stage_inputs", []):
        source = ROOT / relative
        target = f"{REMOTE_ROOT}/{relative}"
        remote("mkdir -p " + shlex.quote(target if source.is_dir() else str(Path(target).parent)))
        sync(str(source) + "/" if source.is_dir() else source,
             f"{network.REMOTE_HOST}:{target}/" if source.is_dir() else f"{network.REMOTE_HOST}:{target}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--queue", type=Path, default=DEFAULT_QUEUE)
    parser.add_argument("--inventory-only", action="store_true")
    parser.add_argument("--max-feature-workers", type=int, default=8)
    parser.add_argument("--poll-seconds", type=int, default=10)
    args = parser.parse_args()
    if args.inventory_only:
        print(json.dumps(inventory(), ensure_ascii=False, indent=2))
        return
    folder = args.queue.resolve().parent
    folder.mkdir(parents=True, exist_ok=True)
    with (folder / "queue.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        devices = inventory()
        if not all(value["visible"] for value in devices.values()):
            save(folder / "status.json", {"state": "gpu_unreachable", "inventory": devices})
            raise RuntimeError("启动前必须确认204四卡和202两卡，详见状态文件")
        queue = json.loads(args.queue.read_text())
        global REMOTE_ROOT
        REMOTE_ROOT = queue.get("remote_root", REMOTE_ROOT)
        stage(queue)
        active, history, attempts = {}, [], {}
        if (folder / "history.json").exists():
            history = json.loads((folder / "history.json").read_text())
        attempt_file = folder / "attempts.json"
        if attempt_file.exists():
            attempts = json.loads(attempt_file.read_text())
        started = time.time()
        with concurrent.futures.ThreadPoolExecutor(max_workers=32) as pool:
            while True:
                queue = json.loads(args.queue.read_text())
                jobs = queue["jobs"]
                names = {job["id"] for job in jobs}
                if len(names) != len(jobs) or any(not set(job.get("depends_on", [])).issubset(names) for job in jobs):
                    raise ValueError("任务ID重复或依赖不存在")
                done = {job["id"] for job in jobs if completed(folder, job)}
                for name, running in list(active.items()):
                    if running["future"].done():
                        result = running["future"].result()
                        history.append({"id": name, **result})
                        save(folder / "history.json", history)
                        del active[name]
                devices = inventory()
                candidates = {}
                for host, payload in devices.items():
                    for device in payload["devices"]:
                        key = (host, device["index"])
                        candidates[key] = device["free_mib"]
                for running in active.values():
                    key = running["host"], running["gpu"]
                    if key in candidates:
                        candidates[key] -= running["memory_mib"]
                failures = [job["id"] for job in jobs if attempts.get(job["id"], 0) >= 3
                            and job["id"] not in done and job["id"] not in active]
                for job in jobs:
                    name = job["id"]
                    if name in done or name in active or name in failures or not set(job.get("depends_on", [])).issubset(done):
                        continue
                    if job["phase"] == "features" and sum(item["phase"] == "features" for item in active.values()) >= args.max_feature_workers:
                        continue
                    required = job.get("memory_mib", 2500)
                    available = [(sum(item["host"] == host and item["gpu"] == gpu for item in active.values()),
                                  -free, host, gpu) for (host, gpu), free in candidates.items()
                                 if free >= required and (not job.get("local_only") or host == "204")]
                    if not available:
                        continue
                    _, _, host, gpu = min(available)
                    if disk_bytes(host) < 8 * 1024**3:
                        print(f"警告：{host}磁盘不足8GiB，暂停增加任务", flush=True)
                        candidates[host, gpu] = 0
                        continue
                    attempts[name] = attempts.get(name, 0) + 1
                    save(attempt_file, attempts)
                    active[name] = {"host": host, "gpu": gpu, "phase": job["phase"], "memory_mib": required,
                                    "started": time.time(), "future": pool.submit(execute, folder, job, host, gpu, attempts[name])}
                    candidates[host, gpu] -= required
                    print(f"启动{name}：{host} GPU{gpu}，第{attempts[name]}次", flush=True)
                waiting = names - done - set(active) - set(failures)
                state = "running" if active or waiting else "failed" if failures else "complete"
                save(folder / "status.json", {"state": state, "completed": len(done), "total": len(jobs),
                     "inventory": devices, "active": {name: {key: value for key, value in item.items() if key != "future"}
                                                         for name, item in active.items()},
                     "failed": failures, "waiting": sorted(waiting), "updated": time.time(),
                     "wall_seconds": time.time() - started, "multiple_tasks_per_gpu": True})
                print(f"队列：{len(done)}/{len(jobs)}完成，{len(active)}活动，{len(failures)}失败", flush=True)
                if not active and (not waiting or (failures and not any(
                        name not in done and name not in failures and set(job.get("depends_on", [])).issubset(done)
                        for job in jobs for name in [job["id"]]))):
                    if failures:
                        final_state = json.loads((folder / "status.json").read_text())
                        final_state.update(state="failed", blocked_by_failed_dependencies=sorted(waiting),
                                           updated=time.time())
                        save(folder / "status.json", final_state)
                        raise RuntimeError(f"任务三次失败：{failures}；依赖任务未伪装为完成")
                    break
                time.sleep(args.poll_seconds)


if __name__ == "__main__":
    main()
