#!/usr/bin/env python3
"""Merlin-1K 六卡任务池：每张卡按显存保留量尽可能并行多个模型/折次。"""

from __future__ import annotations

import fcntl
import json
import os
from pathlib import Path
import shlex
import signal
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "src"
OUT = ROOT / "outputs/merlin_1k/table1_all_models_fivefold"
INPUT = ROOT / "outputs/merlin_1k/experiment"
SCRIPT = SRC / "scripts/run_merlin_1k_all_models.py"

REMOTE_HOST = "Lim@172.16.170.202"
REMOTE_ROOT = "/home/Lim/merlin_1k_pool_20260925"
REMOTE_OUT = REMOTE_ROOT + "/outputs/merlin_1k/table1_all_models_fivefold"
SSH = ["ssh", "-i", "/home/Lim/.ssh/id_ed25519_project4_pool",
       "-o", "IdentitiesOnly=yes", "-o", "BatchMode=yes",
       "-o", "ConnectTimeout=8", "-o", "ServerAliveInterval=15",
       "-o", "ServerAliveCountMax=3"]
LOCAL_GPUS = (0, 1, 2, 3)
REMOTE_GPUS = (0, 1)
DEVICES = (0, 1, 2, 3, 4)

# One process is capped at 40% of a 24-GiB card.  Reserving 9.5 GiB per
# active process allows two image jobs per card while avoiding unsafe thirds.
TASK_RESERVATION_MIB = 9500
MEMORY_FRACTION = "0.40"
POLL_SECONDS = 20
MAX_RESTARTS = 3


def save_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


def command(args, **kwargs):
    return subprocess.run(args, check=True, text=True, **kwargs)


def local_free():
    output = command(["nvidia-smi", "--query-gpu=index,memory.free", "--format=csv,noheader,nounits"],
                     capture_output=True).stdout
    return {int(row.split(",")[0]): int(row.split(",")[1]) for row in output.strip().splitlines()}


def remote_script(script, capture_output=False):
    return command([*SSH, REMOTE_HOST, script], capture_output=capture_output)


def remote_free():
    output = remote_script("nvidia-smi --query-gpu=index,memory.free --format=csv,noheader,nounits",
                          capture_output=True).stdout
    return {int(row.split(",")[0]): int(row.split(",")[1]) for row in output.strip().splitlines()}


def rsync(source, target):
    command(["rsync", "-a", "--partial", "-e", shlex.join(SSH), str(source), target])


def stage_remote():
    if len(list((INPUT / "features").glob("*.npz"))) != 1000:
        raise RuntimeError("Merlin-1K 特征未完成，禁止启动六卡训练")
    OUT.mkdir(parents=True, exist_ok=True)
    # Create the local protocol before copying it to 202.  Relative paths in
    # the runner protocol make its digest identical on both hosts.
    command([sys.executable, str(SCRIPT), "--audit-only", "--output-dir", str(OUT)], cwd=ROOT)
    remote_script("mkdir -p " + shlex.quote(REMOTE_ROOT + "/src") + " " +
                  shlex.quote(REMOTE_ROOT + "/outputs/merlin_1k/experiment") + " " +
                  shlex.quote(REMOTE_OUT))
    rsync(str(SRC) + "/", f"{REMOTE_HOST}:{REMOTE_ROOT}/src/")
    rsync(str(INPUT) + "/", f"{REMOTE_HOST}:{REMOTE_ROOT}/outputs/merlin_1k/experiment/")
    rsync(OUT / "protocol.json", f"{REMOTE_HOST}:{REMOTE_OUT}/protocol.json")
    remote_script("cd " + shlex.quote(REMOTE_ROOT) + " && /home/Lim/conda/envs/myenv/bin/python "
                  "src/scripts/run_merlin_1k_all_models.py --audit-only "
                  "--output-dir " + shlex.quote(REMOTE_OUT))
    save_json(OUT / "remote_stage.json", {"host": REMOTE_HOST, "remote_root": REMOTE_ROOT,
              "remote_output": REMOTE_OUT, "features": 1000, "staged_unix": time.time()})


def launch_local(gpu, model, fold, log_path):
    environment = os.environ.copy()
    environment.update({"CUDA_VISIBLE_DEVICES": str(gpu), "OMP_NUM_THREADS": "2",
                        "TOKENIZERS_PARALLELISM": "false", "MERLIN_MEMORY_FRACTION": MEMORY_FRACTION})
    log_path.parent.mkdir(parents=True, exist_ok=True)
    stream = log_path.open("a", encoding="utf-8")
    stream.write(f"\n[{time.strftime('%F %T')}] 本机GPU{gpu}启动{model}折{fold}\n")
    stream.flush()
    process = subprocess.Popen([sys.executable, "-u", str(SCRIPT), "--worker-index", str(fold - 1),
                                "--devices", *map(str, DEVICES), "--models", model,
                                "--output-dir", str(OUT)], cwd=ROOT, env=environment,
                               stdout=stream, stderr=subprocess.STDOUT)
    stream.close()
    return process


def launch_remote(gpu, model, fold, log_path):
    remote_command = (
        "cd {root} && exec env CUDA_VISIBLE_DEVICES={gpu} OMP_NUM_THREADS=2 "
        "TOKENIZERS_PARALLELISM=false MERLIN_MEMORY_FRACTION={fraction} "
        "/home/Lim/conda/envs/myenv/bin/python -u src/scripts/run_merlin_1k_all_models.py "
        "--worker-index {worker} --devices 0 1 2 3 4 --models {model} --output-dir {out}"
    ).format(root=shlex.quote(REMOTE_ROOT), gpu=gpu, fraction=MEMORY_FRACTION,
             worker=fold - 1, model=shlex.quote(model), out=shlex.quote(REMOTE_OUT))
    log_path.parent.mkdir(parents=True, exist_ok=True)
    stream = log_path.open("a", encoding="utf-8")
    stream.write(f"\n[{time.strftime('%F %T')}] 202主机GPU{gpu}启动{model}折{fold}\n")
    stream.flush()
    process = subprocess.Popen([*SSH, REMOTE_HOST, remote_command], stdout=stream, stderr=subprocess.STDOUT)
    stream.close()
    return process


def done(job):
    model, fold = job
    folder = OUT / f"fold_{fold}" / model
    marker = folder / "completed.json"
    if not marker.exists():
        return False
    digest = json.loads((OUT / "protocol.json").read_text())["protocol_sha256"]
    payload = json.loads(marker.read_text())
    if payload.get("protocol_sha256") != digest:
        raise ValueError(f"结果协议不匹配：{marker}")
    if not all((folder / name).exists() for name in ("test_metrics.json", "test_predictions.csv")):
        raise ValueError(f"完成标记对应的结果不完整：{folder}")
    return True


def remote_done(job):
    model, fold = job
    marker = f"{REMOTE_OUT}/fold_{fold}/{model}/completed.json"
    result = remote_script("test -f " + shlex.quote(marker) + " && echo 1 || echo 0", capture_output=True)
    return result.stdout.strip() == "1"


def sync_remote_job(job):
    model, fold = job
    relative = f"fold_{fold}/{model}"
    destination = OUT / relative
    destination.mkdir(parents=True, exist_ok=True)
    rsync(f"{REMOTE_HOST}:{REMOTE_OUT}/{relative}/", str(destination) + "/")


def main():
    import traceback
    import run_merlin_1k_all_models as runner

    OUT.mkdir(parents=True, exist_ok=True)
    with (OUT / "gpu_pool.lock").open("a") as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as error:
            raise SystemExit("已有 Merlin-1K GPU 池调度器") from error
        if shutil_disk_free(OUT) < 3 * 1024**3:
            raise RuntimeError("本机结果盘剩余不足3GiB，停止新增任务")
        remote_available = remote_script("df -Pk /home/Lim | tail -1", capture_output=True).stdout
        if int(remote_available.split()[3]) * 1024 < 3 * 1024**3:
            raise RuntimeError("202主机结果盘剩余不足3GiB，停止新增任务")
        stage_remote()
        stopping = False

        def stop(_signal, _frame):
            nonlocal stopping
            stopping = True

        signal.signal(signal.SIGINT, stop)
        signal.signal(signal.SIGTERM, stop)
        jobs = [(model, fold) for model in runner.MODELS for fold in range(1, 6)]
        pending = [job for job in jobs if not done(job) and not remote_done(job)]
        active, attempts, failed, history = {}, {}, {}, []
        session_tag = time.strftime("%Y%m%d_%H%M%S")
        started = time.time()
        try:
            while (pending or active) and not stopping:
                for name, running in list(active.items()):
                    process = running["process"]
                    if process.poll() is None:
                        continue
                    job = running["job"]
                    if running["host"] == "remote" and process.returncode == 0:
                        sync_remote_job(job)
                    success = process.returncode == 0 and done(job)
                    event = {key: value for key, value in running.items() if key != "process"}
                    event.update(exit_code=process.returncode, success=success, finished_unix=time.time())
                    history.append(event)
                    save_json(OUT / f"gpu_job_history_{session_tag}.json", history)
                    del active[name]
                    if not success:
                        if attempts[name] < MAX_RESTARTS:
                            pending.append(job)
                        else:
                            failed[name] = running["log"]

                memories = {"local": local_free(), "remote": remote_free()}
                budgets = {host: dict(values) for host, values in memories.items()}
                # Reserve memory for every active process, not just newly
                # launched ones.  This gives a stable maximum of two image
                # jobs per 24-GiB card and still fills idle cards promptly.
                for running in active.values():
                    budgets[running["host"]][running["gpu"]] -= TASK_RESERVATION_MIB
                for job in list(pending):
                    candidates = [(free, host, gpu) for host, values in budgets.items()
                                  for gpu, free in values.items() if free >= TASK_RESERVATION_MIB]
                    if not candidates:
                        break
                    _, host, gpu = max(candidates)
                    model, fold = job
                    name = f"{model}_fold_{fold}"
                    attempts[name] = attempts.get(name, 0) + 1
                    log_path = OUT / "gpu_pool_logs" / f"{session_tag}_{name}_{host}_gpu{gpu}_attempt{attempts[name]}.log"
                    process = (launch_local(gpu, model, fold, log_path) if host == "local"
                               else launch_remote(gpu, model, fold, log_path))
                    active[name] = {"job": job, "host": host, "gpu": gpu, "process": process,
                                    "pid": process.pid, "started_unix": time.time(), "log": str(log_path)}
                    pending.remove(job)
                    budgets[host][gpu] -= TASK_RESERVATION_MIB
                    print(f"分配{name}到{host} GPU{gpu}，尝试{attempts[name]}", flush=True)

                completed = sum(done(job) for job in jobs)
                save_json(OUT / "gpu_pool_status.json", {
                    "state": "running" if active else "waiting_for_memory", "completed": completed,
                    "total": len(jobs), "pending_jobs": len(pending),
                    "task_reservation_mib": TASK_RESERVATION_MIB, "memory_fraction": MEMORY_FRACTION,
                    "memory": memories, "failed_jobs": failed,
                    "active": {name: {key: value for key, value in running.items() if key != "process"}
                               for name, running in active.items()}, "updated_unix": time.time()})
                print(f"Merlin-1K已完成{completed}/{len(jobs)}；排队{len(pending)}；失败{len(failed)}", flush=True)
                if pending or active:
                    time.sleep(POLL_SECONDS)

            command([sys.executable, str(SCRIPT), "--aggregate", "--output-dir", str(OUT)], cwd=ROOT)
            completed = sum(done(job) for job in jobs)
            status = "complete" if completed == len(jobs) else "incomplete"
            save_json(OUT / "gpu_pool_status.json", {"state": status, "completed": completed,
                      "total": len(jobs), "failed_jobs": failed, "wall_seconds": time.time() - started,
                      "updated_unix": time.time()})
        except Exception:
            save_json(OUT / "gpu_pool_status.json", {"state": "failed", "traceback": traceback.format_exc(),
                      "active": {name: {key: value for key, value in running.items() if key != "process"}
                                 for name, running in active.items()}, "updated_unix": time.time()})
            raise
        finally:
            if stopping:
                for running in active.values():
                    if running["process"].poll() is None:
                        running["process"].terminate()


def shutil_disk_free(path):
    """Avoid importing shutil in workers until the scheduler starts."""
    return os.statvfs(path).f_bavail * os.statvfs(path).f_frsize


if __name__ == "__main__":
    main()
