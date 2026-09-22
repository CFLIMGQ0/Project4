#!/usr/bin/env python3
"""MR-RATE-1K与AMOS-MM七标签两条APro路径的两主机动态五折训练池。"""
from __future__ import annotations

import json
import os
from pathlib import Path
import shlex
import signal
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[2]
RUNNER = ROOT / "src/scripts/run_amos_mrrate_position_replacements.py"
REMOTE_HOST = "Lim@172.16.170.202"
REMOTE_ROOT = "/new_data/amef_public_apro_paths_20260920"
SSH = ["ssh", "-i", "/home/Lim/.ssh/id_ed25519_project4_pool", "-o", "IdentitiesOnly=yes",
       "-o", "BatchMode=yes", "-o", "ConnectTimeout=8", "-o", "ServerAliveInterval=15",
       "-o", "ServerAliveCountMax=3"]
LOCAL_GPUS = (0, 1, 2, 3)
REMOTE_GPUS = (0, 1)
RESERVE_MIB = 2200
POLL_SECONDS = 15
MAX_RETRIES = 2
VARIANTS = ("apro_absolute_only", "apro_relative_only")
DATASETS = ("mr_rate", "amos_mm")


def local_output(dataset: str, variant: str) -> Path:
    if dataset == "mr_rate":
        return ROOT / "outputs/mr_rate_1k/position_replacements" / variant
    return ROOT / "outputs/amos_mm/position_replacements_7_labels" / variant


def remote_output(dataset: str, variant: str) -> str:
    suffix = "outputs/mr_rate_1k/position_replacements" if dataset == "mr_rate" else "outputs/amos_mm/position_replacements_7_labels"
    return f"{REMOTE_ROOT}/{suffix}/{variant}"


def save_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


def run(command: list[str], **kwargs):
    return subprocess.run(command, check=True, text=True, **kwargs)


def rsync(source: str | Path, target: str, *options: str) -> None:
    run(["rsync", "-a", "--partial", *options, "-e", shlex.join(SSH), str(source), target])


def remote(command: str, capture_output: bool = False):
    return run([*SSH, REMOTE_HOST, command], capture_output=capture_output)


def free_memory() -> dict[str, dict[int, int]]:
    local_text = run(["nvidia-smi", "--query-gpu=index,memory.free", "--format=csv,noheader,nounits"],
                     capture_output=True).stdout
    local = {int(row.split(",")[0]): int(row.split(",")[1]) for row in local_text.strip().splitlines()}
    remote_text = remote("nvidia-smi --query-gpu=index,memory.free --format=csv,noheader,nounits", True).stdout
    remote_values = {int(row.split(",")[0]): int(row.split(",")[1]) for row in remote_text.strip().splitlines()}
    return {"local": local, "remote": remote_values}


def stage_remote() -> None:
    remote("mkdir -p " + shlex.quote(REMOTE_ROOT + "/src") + " " +
           shlex.quote(REMOTE_ROOT + "/outputs/mr_rate_1k") + " " +
           shlex.quote(REMOTE_ROOT + "/outputs/amos_mm"))
    rsync(str(ROOT / "src") + "/", f"{REMOTE_HOST}:{REMOTE_ROOT}/src/")
    for dataset in DATASETS:
        source = ROOT / "outputs" / ("mr_rate_1k" if dataset == "mr_rate" else "amos_mm") / "experiment"
        target = f"{REMOTE_HOST}:{REMOTE_ROOT}/outputs/{'mr_rate_1k' if dataset == 'mr_rate' else 'amos_mm'}/"
        rsync(source, target)
    for dataset in DATASETS:
        for variant in VARIANTS:
            remote(f"cd {shlex.quote(REMOTE_ROOT)} && /home/Lim/conda/envs/myenv/bin/python -u "
                   f"src/scripts/run_amos_mrrate_position_replacements.py --dataset {dataset} "
                   f"--variant {variant} --initialize")


def initialize_local() -> None:
    for dataset in DATASETS:
        for variant in VARIANTS:
            output = local_output(dataset, variant)
            run([sys.executable, str(RUNNER), "--dataset", dataset, "--variant", variant, "--initialize"], cwd=ROOT)
            save_json(output / "pool_protocol.json", {
                "dataset": dataset, "variant": variant, "seed": 42,
                "remote_root": REMOTE_ROOT, "memory_reservation_mib": RESERVE_MIB,
                "note": "同一训练协议；204四卡与202两卡按实时空闲显存动态分配，允许同卡多任务。",
            })


def marker(dataset: str, variant: str, fold: int, root: Path | None = None) -> Path:
    base = local_output(dataset, variant) if root is None else root
    if dataset == "mr_rate":
        return base / f"fold_{fold}" / "amef_multimodal" / "completed.json"
    return base / "7_labels" / f"fold_{fold}" / "amef_multimodal" / "result.json"


def job_key(dataset: str, variant: str, fold: int) -> str:
    return f"{dataset}__{variant}__fold_{fold}"


def launch_local(dataset: str, variant: str, fold: int, gpu: int, log: Path):
    log.parent.mkdir(parents=True, exist_ok=True)
    environment = os.environ.copy()
    environment.update({"CUDA_VISIBLE_DEVICES": str(gpu), "OMP_NUM_THREADS": "2",
                        "OPENBLAS_NUM_THREADS": "2", "TOKENIZERS_PARALLELISM": "false"})
    stream = log.open("a", encoding="utf-8")
    stream.write(f"[{time.strftime('%F %T')}] local GPU{gpu} {dataset} {variant} fold{fold}\n")
    stream.flush()
    process = subprocess.Popen([sys.executable, "-u", str(RUNNER), "--dataset", dataset,
                                "--variant", variant, "--fold", str(fold)],
                               cwd=ROOT, env=environment, stdout=stream, stderr=subprocess.STDOUT)
    stream.close()
    return process


def launch_remote(dataset: str, variant: str, fold: int, gpu: int, log: Path):
    log.parent.mkdir(parents=True, exist_ok=True)
    command = (f"cd {shlex.quote(REMOTE_ROOT)} && exec env CUDA_VISIBLE_DEVICES={gpu} OMP_NUM_THREADS=2 "
               f"OPENBLAS_NUM_THREADS=2 TOKENIZERS_PARALLELISM=false "
               f"/home/Lim/conda/envs/myenv/bin/python -u src/scripts/run_amos_mrrate_position_replacements.py "
               f"--dataset {dataset} --variant {variant} --fold {fold}")
    stream = log.open("a", encoding="utf-8")
    stream.write(f"[{time.strftime('%F %T')}] 202 GPU{gpu} {dataset} {variant} fold{fold}\n")
    stream.flush()
    process = subprocess.Popen([*SSH, REMOTE_HOST, command], stdout=stream, stderr=subprocess.STDOUT)
    stream.close()
    return process


def sync_remote_job(dataset: str, variant: str, fold: int) -> None:
    source = remote_output(dataset, variant)
    destination = local_output(dataset, variant)
    relative = f"7_labels/fold_{fold}/amef_multimodal" if dataset == "amos_mm" else f"fold_{fold}/amef_multimodal"
    (destination / relative).mkdir(parents=True, exist_ok=True)
    rsync(f"{REMOTE_HOST}:{source}/{relative}/", str(destination / relative) + "/")


def aggregate_local() -> None:
    for dataset in DATASETS:
        for variant in VARIANTS:
            run([sys.executable, str(RUNNER), "--dataset", dataset, "--variant", variant, "--aggregate"], cwd=ROOT)


def main() -> None:
    initialize_local()
    stage_remote()
    jobs = [(dataset, variant, fold) for dataset in DATASETS for variant in VARIANTS for fold in range(1, 6)]
    pending = [job for job in jobs if not marker(*job).exists()]
    active: dict[str, dict] = {}
    attempts: dict[str, int] = {}
    failed: dict[str, str] = {}
    history: list[dict] = []
    log_root = ROOT / "outputs/public_apro_path_ablation_pool/logs"
    status_path = ROOT / "outputs/public_apro_path_ablation_pool/status.json"
    stopping = False

    def stop(_signal, _frame):
        nonlocal stopping
        stopping = True

    signal.signal(signal.SIGINT, stop)
    signal.signal(signal.SIGTERM, stop)
    session = time.strftime("%Y%m%d_%H%M%S")
    started = time.time()
    while (pending or active) and not stopping:
        memories = free_memory()
        for name, running in list(active.items()):
            process = running["process"]
            if process.poll() is None:
                continue
            dataset, variant, fold = running["job"]
            if running["host"] == "remote" and process.returncode == 0:
                try:
                    sync_remote_job(dataset, variant, fold)
                except Exception as error:
                    running["sync_error"] = repr(error)
            success = process.returncode == 0 and marker(dataset, variant, fold).exists()
            event = {key: value for key, value in running.items() if key != "process"}
            event.update(exit_code=process.returncode, success=success, finished_unix=time.time())
            history.append(event)
            save_json(ROOT / "outputs/public_apro_path_ablation_pool/history.json", history)
            del active[name]
            if not success:
                if attempts[name] < MAX_RETRIES:
                    pending.append((dataset, variant, fold))
                else:
                    failed[name] = str(running.get("log", ""))

        active_counts = {host: {gpu: 0 for gpu in values} for host, values in memories.items()}
        for running in active.values():
            active_counts[running["host"]][running["gpu"]] += 1
        for job in list(pending):
            candidates = []
            for host, values in memories.items():
                for gpu, free in values.items():
                    if free - RESERVE_MIB * active_counts[host][gpu] >= RESERVE_MIB:
                        candidates.append((active_counts[host][gpu], -free, host, gpu))
            if not candidates:
                break
            _, _, host, gpu = min(candidates)
            dataset, variant, fold = job
            name = job_key(*job)
            attempts[name] = attempts.get(name, 0) + 1
            log = log_root / f"{session}_{name}_{host}_gpu{gpu}_attempt{attempts[name]}.log"
            process = (launch_local(dataset, variant, fold, gpu, log) if host == "local"
                       else launch_remote(dataset, variant, fold, gpu, log))
            active[name] = {"job": job, "host": host, "gpu": gpu, "pid": process.pid,
                            "process": process, "started_unix": time.time(), "log": str(log),
                            "attempt": attempts[name]}
            active_counts[host][gpu] += 1
            pending.remove(job)
            print(f"分配 {name} -> {host} GPU{gpu}；第{attempts[name]}次；显存预留{RESERVE_MIB}MiB", flush=True)

        completed = sum(marker(*job).exists() for job in jobs)
        save_json(status_path, {"state": "running" if active else "waiting_for_memory",
                                "completed": completed, "total": len(jobs), "pending": len(pending),
                                "failed": failed, "memory_reservation_mib": RESERVE_MIB,
                                "memory": memories,
                                "active": {name: {key: value for key, value in item.items() if key != "process"}
                                           for name, item in active.items()},
                                "updated_unix": time.time()})
        print(f"公共数据集APro分支：完成{completed}/{len(jobs)}，排队{len(pending)}，失败{len(failed)}", flush=True)
        if pending or active:
            time.sleep(POLL_SECONDS)

    for running in active.values():
        if running["process"].poll() is None:
            running["process"].terminate()
    if not stopping:
        aggregate_local()
        state = "complete" if not pending and not active and not failed else "incomplete"
    else:
        state = "stopped"
    save_json(status_path, {"state": state, "completed": sum(marker(*job).exists() for job in jobs),
                            "total": len(jobs), "pending": len(pending), "failed": failed,
                            "wall_seconds": time.time() - started, "updated_unix": time.time()})
    if failed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
