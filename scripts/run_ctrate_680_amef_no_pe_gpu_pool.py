#!/usr/bin/env python3
"""CT-RATE AMEF-MIL位置变体五折实验的本地/202主机动态GPU池。"""
from __future__ import annotations

import fcntl
import json
import os
from pathlib import Path
import shlex
import subprocess
import sys
import time


ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "outputs" / "ct_rate_680" / "amef_no_pe_fivefold"
INPUT = ROOT / "outputs" / "ct_rate_680" / "experiment"
SRC = ROOT / "src"
SCRIPT = SRC / "scripts" / "run_ctrate_680_amef_no_pe.py"
MR_STATUS = ROOT / "outputs" / "mr_rate_1k" / "all_models_fivefold" / "gpu_pool_status.json"
REMOTE_HOST = "Lim@172.16.170.202"
POSITION_VARIANT = "no_pe"
REMOTE_ROOT = "/home/Lim/ct_rate_680_amef_no_pe_20260919"
REMOTE_OUT = REMOTE_ROOT + "/outputs/ct_rate_680/amef_no_pe_fivefold"
SSH = ["ssh", "-i", "/home/Lim/.ssh/id_ed25519_project4_pool", "-o", "IdentitiesOnly=yes",
       "-o", "BatchMode=yes", "-o", "ConnectTimeout=8", "-o", "ServerAliveInterval=15",
       "-o", "ServerAliveCountMax=3"]
MINIMUM_FREE_MIB = 7500
POLL_SECONDS = 20


def run(command, **kwargs):
    return subprocess.run(command, check=True, text=True, **kwargs)


def save_json(path: Path, value):
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


def local_free():
    output = run(["nvidia-smi", "--query-gpu=index,memory.free", "--format=csv,noheader,nounits"],
                 capture_output=True).stdout
    return {int(index): int(free) for index, free in (line.split(",", 1) for line in output.splitlines() if line.strip())}


def remote_script(script: str, capture_output=False):
    return run([*SSH, REMOTE_HOST, script], capture_output=capture_output)


def remote_free():
    output = remote_script("nvidia-smi --query-gpu=index,memory.free --format=csv,noheader,nounits",
                           capture_output=True).stdout
    return {int(index): int(free) for index, free in (line.split(",", 1) for line in output.splitlines() if line.strip())}


def rsync(source, target):
    run(["rsync", "-a", "--partial", "-e", shlex.join(SSH), str(source), target])


def stage_remote():
    if len(list((INPUT / "features").glob("*.npz"))) != 680:
        raise RuntimeError(f"CT-RATE特征未完成，禁止启动{POSITION_VARIANT}五折实验")
    remote_script("mkdir -p " + shlex.quote(REMOTE_ROOT + "/src") + " " +
                  shlex.quote(REMOTE_ROOT + "/outputs/ct_rate_680/experiment") + " " +
                  shlex.quote(REMOTE_OUT))
    rsync(str(SRC) + "/", f"{REMOTE_HOST}:{REMOTE_ROOT}/src/")
    rsync(str(INPUT) + "/", f"{REMOTE_HOST}:{REMOTE_ROOT}/outputs/ct_rate_680/experiment/")
    rsync(OUT / "protocol.json", f"{REMOTE_HOST}:{REMOTE_OUT}/protocol.json")


def launch_local(gpu: int, worker: int, devices: list[int], log_path: Path):
    environment = os.environ.copy()
    environment.update(CUDA_VISIBLE_DEVICES=str(gpu), OMP_NUM_THREADS="4",
                       TOKENIZERS_PARALLELISM="false")
    log_path.parent.mkdir(parents=True, exist_ok=True)
    stream = log_path.open("a", encoding="utf-8")
    stream.write(f"\n[{time.strftime('%F %T')}] launch local GPU{gpu} worker{worker}\n")
    stream.flush()
    process = subprocess.Popen(
        [sys.executable, "-u", str(SCRIPT), "--worker-index", str(worker), "--devices",
         *map(str, devices), "--output-dir", str(OUT), "--position-variant", POSITION_VARIANT],
        cwd=ROOT, env=environment, stdout=stream, stderr=subprocess.STDOUT,
    )
    stream.close()
    return process


def launch_remote(gpu: int, worker: int, devices: list[int], log_path: Path):
    command = ("cd {root} && exec env CUDA_VISIBLE_DEVICES={gpu} OMP_NUM_THREADS=4 "
               "TOKENIZERS_PARALLELISM=false /home/Lim/conda/envs/myenv/bin/python -u "
               "src/scripts/run_ctrate_680_amef_no_pe.py --worker-index {worker} "
               "--devices {devices} --output-dir {out} --position-variant {variant}").format(
                   root=shlex.quote(REMOTE_ROOT), gpu=gpu, worker=worker,
                   devices=" ".join(map(str, devices)), out=shlex.quote(REMOTE_OUT),
                   variant=shlex.quote(POSITION_VARIANT))
    log_path.parent.mkdir(parents=True, exist_ok=True)
    stream = log_path.open("a", encoding="utf-8")
    stream.write(f"\n[{time.strftime('%F %T')}] launch remote GPU{gpu} worker{worker}\n")
    stream.flush()
    process = subprocess.Popen([*SSH, REMOTE_HOST, command], stdout=stream, stderr=subprocess.STDOUT)
    stream.close()
    return process


def local_completed():
    return len(list(OUT.glob("fold_*/*/completed.json")))


def remote_completed():
    result = remote_script("find " + shlex.quote(REMOTE_OUT) + " -path '*/completed.json' -type f | wc -l",
                           capture_output=True)
    return int(result.stdout.strip())


def wait_for_mr():
    while True:
        state = "missing"
        if MR_STATUS.is_file():
            try:
                state = json.loads(MR_STATUS.read_text()).get("state", "unknown")
            except json.JSONDecodeError:
                state = "unreadable"
        if state != "running":
            print(f"MR-RATE状态为{state}，准备启动CT-RATE {POSITION_VARIANT}五折。", flush=True)
            return
        print(f"MR-RATE仍在运行，CT-RATE {POSITION_VARIANT}五折继续排队。", flush=True)
        time.sleep(60)


def main():
    global OUT, POSITION_VARIANT, REMOTE_ROOT, REMOTE_OUT
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--position-variant", choices=("no_pe", "original_pe"), default="no_pe")
    parser.add_argument("--output-dir", type=Path)
    args = parser.parse_args()
    POSITION_VARIANT = args.position_variant
    OUT = args.output_dir or ROOT / "outputs" / "ct_rate_680" / f"amef_{POSITION_VARIANT}_fivefold"
    REMOTE_ROOT = f"/home/Lim/ct_rate_680_amef_{POSITION_VARIANT}_20260919"
    REMOTE_OUT = REMOTE_ROOT + f"/outputs/ct_rate_680/amef_{POSITION_VARIANT}_fivefold"
    OUT.mkdir(parents=True, exist_ok=True)
    with (OUT / "gpu_pool.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        stage_remote()
        devices = list(range(5))
        pending = list(range(5))
        active = {}
        retries = {fold: 0 for fold in pending}
        logs = OUT / "gpu_pool_logs"
        started = time.time()
        while pending or active:
            local_memory = local_free()
            remote_memory = remote_free()
            for key, payload in list(active.items()):
                process, fold = payload
                if process.poll() is None:
                    continue
                del active[key]
                if process.returncode != 0:
                    retries[fold] += 1
                    if retries[fold] <= 3:
                        pending.append(fold)
                    else:
                        raise RuntimeError(
                            f"CT {POSITION_VARIANT}折{fold + 1}连续失败{retries[fold]}次"
                        )
            for fold in list(pending):
                occupied = {(key.split(":", 2)[0], int(key.split(":", 2)[1])) for key in active}
                candidates = [(free, "local", gpu) for gpu, free in local_memory.items()
                              if free >= MINIMUM_FREE_MIB and ("local", gpu) not in occupied]
                candidates += [(free, "remote", gpu) for gpu, free in remote_memory.items()
                               if free >= MINIMUM_FREE_MIB and ("remote", gpu) not in occupied]
                if not candidates:
                    break
                _, host, gpu = max(candidates)
                pending.remove(fold)
                key = f"{host}:{gpu}:fold{fold + 1}"
                log_path = logs / f"{host}_gpu_{gpu}_fold_{fold + 1}.log"
                process = (launch_local(gpu, fold, devices, log_path) if host == "local"
                           else launch_remote(gpu, fold, devices, log_path))
                active[key] = (process, fold)
                local_memory = local_free()
                remote_memory = remote_free()
            completed = local_completed() + remote_completed()
            save_json(OUT / "gpu_pool_status.json", {
                "state": "running", "pending_folds": [fold + 1 for fold in pending],
                "active": {key: {"fold": fold + 1, "pid": process.pid}
                           for key, (process, fold) in active.items()},
                "local_memory": local_memory, "remote_memory": remote_memory,
                "completed": completed, "total": 5, "started_unix": started,
            })
            print(f"CT-RATE {POSITION_VARIANT}完成：{completed}/5个折次；排队：{[fold + 1 for fold in pending]}", flush=True)
            if not pending and not active:
                break
            time.sleep(POLL_SECONDS)
        for process in active.values():
            if process[0].poll() is None:
                process[0].terminate()
        for process, _ in active.values():
            process.wait(timeout=30)
        rsync(f"{REMOTE_HOST}:{REMOTE_OUT}/", str(OUT) + "/")
        result = subprocess.run([sys.executable, str(SCRIPT), "--aggregate", "--output-dir", str(OUT),
                                 "--position-variant", POSITION_VARIANT],
                                cwd=ROOT)
        if result.returncode:
            raise SystemExit(result.returncode)
        completed = local_completed()
        state = "complete" if completed == 5 else "incomplete"
        save_json(OUT / "gpu_pool_status.json", {
            "state": state, "completed": completed, "total": 5,
            "finished_unix": time.time(), "wall_seconds": time.time() - started,
        })


if __name__ == "__main__":
    main()
