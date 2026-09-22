#!/usr/bin/env python3
"""切片缺失实验：本机提取特征，本机和202主机按剩余显存调度训练。"""
from __future__ import annotations

import argparse
import fcntl
import json
import os
from pathlib import Path
import shlex
import shutil
import subprocess
import sys
import time
import traceback

import run_ctrate_slice_missing as experiment
import run_ctrate_680_amef_no_pe_gpu_pool as network
from prepare_ctrate_680_experiment import save_json

ROOT = experiment.ROOT
SCRIPT = ROOT / "src/scripts/run_ctrate_slice_missing.py"


def checked(command):
    subprocess.run(command, cwd=ROOT, check=True)


def disk_free(host, remote_root):
    if host == "local":
        return shutil.disk_usage(ROOT).free
    output = network.remote_script(f"df -Pk {shlex.quote(remote_root)} | tail -1",
                                   capture_output=True).stdout
    return int(output.split()[3]) * 1024


def sync_to_remote(args, remote_root, remote_out):
    network.remote_script(f"mkdir -p {shlex.quote(remote_root + '/src')} {shlex.quote(remote_out)}")
    if disk_free("remote", remote_root) < 3 * 1024**3:
        raise RuntimeError("202主机目标磁盘不足3GiB，暂停同步；不删除其他实验数据")
    subprocess.run(["rsync", "-a", "--exclude=.git", "--exclude=__pycache__", "-e", shlex.join(network.SSH),
                    str(ROOT / "src") + "/", f"{network.REMOTE_HOST}:{remote_root}/src/"], check=True)
    network.rsync(str(args.output_dir / "data") + "/", f"{network.REMOTE_HOST}:{remote_out}/data/")
    for variant in experiment.VARIANTS:
        network.remote_script(f"mkdir -p {shlex.quote(remote_out + '/' + variant)}")
        network.rsync(args.output_dir / variant / "protocol.json",
                      f"{network.REMOTE_HOST}:{remote_out}/{variant}/protocol.json")
    command = shlex.join(["/home/Lim/conda/envs/myenv/bin/python", str(Path(remote_root) / SCRIPT.relative_to(ROOT)),
                          "--output-dir", remote_out, "--audit-only"])
    network.remote_script(f"cd {shlex.quote(remote_root)} && {command}")


def completed(args, job):
    if job["stage"] == "features":
        return all((args.output_dir / "data/features" / f"{index:04d}.npz").is_file()
                   for index in range(job["worker"], job["cases"], args.feature_workers))
    folder = args.output_dir / job["variant"] / f"fold_{job['fold']}" / experiment.MODEL_KEY
    marker = folder / "completed.json"
    if not marker.exists():
        return False
    expected = json.loads((args.output_dir / job["variant"] / "protocol.json").read_text())["protocol_sha256"]
    payload = json.loads(marker.read_text())
    if payload["protocol_sha256"] != expected or payload["fold"] != job["fold"]:
        raise ValueError(f"完成标记与当前协议不一致：{marker}")
    return all((folder / name).is_file() for name in ("best_model.pt", "test_metrics.json", "test_predictions.csv"))


def launch(args, job, host, gpu, remote_root, remote_out, attempt):
    arguments = (["--feature-worker", str(job["worker"]), "--feature-workers", str(args.feature_workers)]
                 if job["stage"] == "features" else ["--variant", job["variant"], "--fold", str(job["fold"])])
    log = args.output_dir / "logs" / f"{job['name']}_{host}_gpu{gpu}_attempt{attempt}.log"
    log.parent.mkdir(parents=True, exist_ok=True)
    environment = os.environ.copy()
    environment.update(CUDA_VISIBLE_DEVICES=str(gpu), OMP_NUM_THREADS="2", TOKENIZERS_PARALLELISM="false")
    if host == "local":
        command = [sys.executable, "-u", str(SCRIPT), "--output-dir", str(args.output_dir), *arguments]
    else:
        command = [*network.SSH, network.REMOTE_HOST,
                   f"cd {shlex.quote(remote_root)} && exec " + shlex.join([
                       "env", f"CUDA_VISIBLE_DEVICES={gpu}", "OMP_NUM_THREADS=2", "TOKENIZERS_PARALLELISM=false",
                       "/home/Lim/conda/envs/myenv/bin/python", "-u", str(Path(remote_root) / SCRIPT.relative_to(ROOT)),
                       "--output-dir", remote_out, *arguments])]
    with log.open("a", encoding="utf-8") as stream:
        stream.write(f"\n{time.strftime('%F %T')} {job['name']} {host} GPU{gpu}\n")
        stream.flush()
        process = subprocess.Popen(command, cwd=ROOT, env=environment, stdout=stream, stderr=subprocess.STDOUT)
    print(f"启动{job['name']}：{host} GPU{gpu}，日志{log.name}", flush=True)
    return {"job": job, "process": process, "host": host, "gpu": gpu, "started": time.time(), "log": str(log)}


def schedule(args, jobs, remote_root, remote_out, use_remote):
    pending = [job for job in jobs if not completed(args, job)]
    active, attempts, history = {}, {}, []
    failures = []
    state_path = args.output_dir / "gpu_pool_status.json"
    started = time.time()
    stage = jobs[0]["stage"]
    while pending or active:
        for name, running in list(active.items()):
            process = running["process"]
            if process.poll() is None:
                continue
            job = running["job"]
            if running["host"] == "remote" and process.returncode == 0:
                relative = f"{job['variant']}/fold_{job['fold']}/{experiment.MODEL_KEY}"
                destination = args.output_dir / relative
                destination.mkdir(parents=True, exist_ok=True)
                network.rsync(f"{network.REMOTE_HOST}:{remote_out}/{relative}/", str(destination) + "/")
            success = process.returncode == 0 and completed(args, job)
            event = {key: value for key, value in running.items() if key != "process"}
            event.update(exit_code=process.returncode, success=success, finished=time.time())
            history.append(event)
            save_json(args.output_dir / f"{stage}_job_history.json", history)
            del active[name]
            if not success:
                if attempts[name] < 3:
                    pending.append(job)
                else:
                    failures.append(name)
        memories = {"local": network.local_free()}
        if use_remote:
            memories["remote"] = network.remote_free()
        for running in active.values():
            if time.time() - running["started"] < 30:
                memories[running["host"]][running["gpu"]] -= args.minimum_free_mib
        for job in list(pending):
            candidates = [(free, host, gpu) for host, devices in memories.items() for gpu, free in devices.items()
                          if free >= args.minimum_free_mib]
            if not candidates:
                break
            _, host, gpu = max(candidates)
            if disk_free(host, remote_root) < 3 * 1024**3:
                raise RuntimeError(f"{host}目标磁盘不足3GiB，停止增加任务；请查看已启动任务的日志")
            name = job["name"]
            attempts[name] = attempts.get(name, 0) + 1
            active[name] = launch(args, job, host, gpu, remote_root, remote_out, attempts[name])
            memories[host][gpu] -= args.minimum_free_mib
            pending.remove(job)
        count = sum(completed(args, job) for job in jobs)
        save_json(state_path, {"state": "running", "stage": stage, "completed_jobs": count, "total_jobs": len(jobs),
                  "pending": [job["name"] for job in pending], "failed": failures,
                  "active": {name: {key: value for key, value in running.items() if key != "process"}
                             | {"pid": running["process"].pid} for name, running in active.items()},
                  "available_memory_after_launch_reservations": memories,
                  "minimum_free_mib": args.minimum_free_mib, "updated_unix": time.time()})
        if stage == "features":
            features = len(list((args.output_dir / "data/features").glob("*.npz")))
            print(f"特征提取：{features}/{jobs[0]['cases']}例，活动任务{len(active)}", flush=True)
        else:
            print(f"五折训练：{count}/{len(jobs)}个任务，活动任务{len(active)}", flush=True)
        if time.time() - started > args.stage_timeout_seconds:
            raise TimeoutError(f"{stage}阶段超过时间限制；活动任务仍可由状态文件追踪")
        if pending or active:
            time.sleep(args.poll_seconds)
    if failures:
        raise RuntimeError(f"以下任务三次失败：{failures}，详见逐次日志")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=experiment.DEFAULT_OUTPUT)
    parser.add_argument("--delete-fraction", type=float, choices=(0.25, 0.5, 0.75), default=0.25)
    parser.add_argument("--sampling-seed", type=int, default=42)
    parser.add_argument("--feature-workers", type=int, default=4)
    parser.add_argument("--minimum-free-mib", type=int, default=3500)
    parser.add_argument("--poll-seconds", type=int, default=15)
    parser.add_argument("--stage-timeout-seconds", type=int, default=14400)
    args = parser.parse_args()
    args.output_dir = args.output_dir.resolve()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    relative = args.output_dir.relative_to(ROOT)
    remote_root = f"/home/Lim/ct_rate_{args.output_dir.name}_20260919"
    remote_out = str(Path(remote_root) / relative)
    with (args.output_dir / "pipeline.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        started = time.time()
        try:
            checked([sys.executable, str(SCRIPT), "--output-dir", str(args.output_dir), "--prepare-only",
                     "--delete-fraction", str(args.delete_fraction), "--sampling-seed", str(args.sampling_seed)])
            cases = len(json.loads((args.output_dir / "data/samples.json").read_text()))
            jobs = [{"stage": "features", "worker": worker, "cases": cases, "name": f"features_{worker}"}
                    for worker in range(args.feature_workers)]
            schedule(args, jobs, remote_root, remote_out, use_remote=False)
            checked([sys.executable, str(SCRIPT), "--output-dir", str(args.output_dir), "--audit-only"])
            free, gpu = max((free, gpu) for gpu, free in network.local_free().items())
            if free < args.minimum_free_mib:
                raise RuntimeError("特征完成后没有足够显存进行训练冒烟检查，可稍后重启流水线续跑")
            checked(["env", f"CUDA_VISIBLE_DEVICES={gpu}", sys.executable, str(SCRIPT),
                     "--output-dir", str(args.output_dir), "--smoke-test"])
            sync_to_remote(args, remote_root, remote_out)
            jobs = [{"stage": "training", "variant": variant, "fold": fold, "name": f"{variant}_fold{fold}"}
                    for fold in range(1, 6) for variant in experiment.VARIANTS]
            schedule(args, jobs, remote_root, remote_out, use_remote=True)
            checked([sys.executable, str(SCRIPT), "--output-dir", str(args.output_dir), "--aggregate"])
            save_json(args.output_dir / "gpu_pool_status.json", {"state": "complete", "completed_jobs": 10,
                      "total_jobs": 10, "cases": cases, "delete_fraction": args.delete_fraction,
                      "finished_unix": time.time(), "wall_seconds": time.time() - started})
        except Exception:
            state_file = args.output_dir / "gpu_pool_status.json"
            state = json.loads(state_file.read_text()) if state_file.exists() else {}
            state.update(state="failed", traceback=traceback.format_exc(), updated_unix=time.time())
            save_json(state_file, state)
            raise


if __name__ == "__main__":
    main()
