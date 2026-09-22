#!/usr/bin/env python3
"""两主机六卡动态执行公共数据集APro分支的删除分类与位置恢复评估。"""
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
CLASSIFY = ROOT / "src/scripts/evaluate_amos_mrrate_deletion_classification.py"
RECOVERY = ROOT / "src/scripts/evaluate_amos_mrrate_position_recovery.py"
REMOTE_HOST = "Lim@172.16.170.202"
REMOTE_ROOT = "/new_data/amef_public_apro_paths_20260920"
SSH = ["ssh", "-i", "/home/Lim/.ssh/id_ed25519_project4_pool", "-o", "IdentitiesOnly=yes",
       "-o", "BatchMode=yes", "-o", "ConnectTimeout=8", "-o", "ServerAliveInterval=15",
       "-o", "ServerAliveCountMax=3"]
RESERVE_MIB = 1800
POLL_SECONDS = 10
LOCAL_GPUS = (0, 1, 2, 3)
REMOTE_GPUS = (0, 1)
VARIANTS = ("apro_absolute_only", "apro_relative_only")
DATASETS = ("mr_rate", "amos_mm")


def run(command: list[str], **kwargs):
    return subprocess.run(command, check=True, text=True, **kwargs)


def rsync(source: str | Path, target: str, *options: str) -> None:
    run(["rsync", "-a", "--partial", *options, "-e", shlex.join(SSH), str(source), target])


def remote(command: str, capture_output: bool = False):
    return run([*SSH, REMOTE_HOST, command], capture_output=capture_output)


def save_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + f".{os.getpid()}.tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


def output_path(dataset: str, kind: str, variant: str, fold: int) -> Path:
    root = ROOT / "outputs" / ("mr_rate_1k" if dataset == "mr_rate" else "amos_mm")
    name = "position_recovery_path_ablation_seed42" if kind == "recovery" else "deletion_classification_path_ablation_seed42"
    return root / name / variant / "shards" / f"fold_{fold}.json"


def remote_output_path(dataset: str, kind: str, variant: str, fold: int) -> str:
    return str(Path(REMOTE_ROOT) / output_path(dataset, kind, variant, fold).relative_to(ROOT))


def checkpoint_root(dataset: str, variant: str, remote_mode: bool) -> str:
    root = (ROOT / "outputs" / ("mr_rate_1k/position_replacements" if dataset == "mr_rate"
                                else "amos_mm/position_replacements_7_labels") / variant)
    return str(root if not remote_mode else Path(REMOTE_ROOT) / root.relative_to(ROOT))


def stage_remote() -> None:
    remote("mkdir -p " + shlex.quote(REMOTE_ROOT + "/src") + " " + shlex.quote(REMOTE_ROOT + "/outputs"))
    rsync(str(ROOT / "src") + "/", f"{REMOTE_HOST}:{REMOTE_ROOT}/src/")
    for dataset in DATASETS:
        dataset_root = ROOT / "outputs" / ("mr_rate_1k" if dataset == "mr_rate" else "amos_mm")
        rsync(str(dataset_root / "experiment") + "/",
              f"{REMOTE_HOST}:{REMOTE_ROOT}/outputs/{dataset_root.name}/experiment/")
        source = dataset_root / ("position_replacements" if dataset == "mr_rate" else "position_replacements_7_labels")
        target = f"{REMOTE_HOST}:{REMOTE_ROOT}/outputs/{dataset_root.name}/{source.name}/"
        for variant in VARIANTS:
            rsync(str(source / variant) + "/", target + variant + "/")


def launch(job: tuple[str, str, str, int], host: str, gpu: int, log: Path):
    dataset, kind, variant, fold = job
    output = output_path(dataset, kind, variant, fold)
    output.parent.mkdir(parents=True, exist_ok=True)
    log.parent.mkdir(parents=True, exist_ok=True)
    if host == "local":
        script = CLASSIFY if kind == "classification" else RECOVERY
        env = os.environ.copy()
        env.update({"CUDA_VISIBLE_DEVICES": str(gpu), "OMP_NUM_THREADS": "2", "TOKENIZERS_PARALLELISM": "false"})
        command = [sys.executable, "-u", str(script), "--dataset", dataset, "--variant", variant,
                   "--fold", str(fold), "--checkpoint-root", checkpoint_root(dataset, variant, False),
                   "--output", str(output), "--batch-size", "32"]
        stream = log.open("a", encoding="utf-8")
        stream.write(f"[{time.strftime('%F %T')}] local GPU{gpu} {job}\n")
        stream.flush()
        process = subprocess.Popen(command, cwd=ROOT, env=env, stdout=stream, stderr=subprocess.STDOUT)
    else:
        script = "src/scripts/evaluate_amos_mrrate_deletion_classification.py" if kind == "classification" else "src/scripts/evaluate_amos_mrrate_position_recovery.py"
        remote_output = remote_output_path(dataset, kind, variant, fold)
        command = (f"cd {shlex.quote(REMOTE_ROOT)} && exec env CUDA_VISIBLE_DEVICES={gpu} OMP_NUM_THREADS=2 "
                   f"TOKENIZERS_PARALLELISM=false /home/Lim/conda/envs/myenv/bin/python -u {script} "
                   f"--dataset {dataset} --variant {variant} --fold {fold} "
                   f"--checkpoint-root {shlex.quote(checkpoint_root(dataset, variant, True))} "
                   f"--output {shlex.quote(remote_output)} --batch-size 32")
        stream = log.open("a", encoding="utf-8")
        stream.write(f"[{time.strftime('%F %T')}] 202 GPU{gpu} {job}\n")
        stream.flush()
        process = subprocess.Popen([*SSH, REMOTE_HOST, command], stdout=stream, stderr=subprocess.STDOUT)
    stream.close()
    return process


def sync_result(job: tuple[str, str, str, int]) -> None:
    dataset, kind, variant, fold = job
    source = remote_output_path(dataset, kind, variant, fold)
    destination = output_path(dataset, kind, variant, fold)
    destination.parent.mkdir(parents=True, exist_ok=True)
    rsync(f"{REMOTE_HOST}:{source}", str(destination))


def main() -> None:
    stage_remote()
    jobs = [(dataset, kind, variant, fold)
            for dataset in DATASETS for kind in ("classification", "recovery")
            for variant in VARIANTS for fold in range(1, 6)]
    pending = [job for job in jobs if not output_path(*job).exists()]
    active: dict[str, dict] = {}
    attempts: dict[str, int] = {}
    failed: dict[str, str] = {}
    history = []
    log_root = ROOT / "outputs/public_apro_path_evaluation_pool/logs"
    status_path = ROOT / "outputs/public_apro_path_evaluation_pool/status.json"
    stopping = False

    def stop(_signal, _frame):
        nonlocal stopping
        stopping = True

    signal.signal(signal.SIGINT, stop)
    signal.signal(signal.SIGTERM, stop)
    session = time.strftime("%Y%m%d_%H%M%S")
    started = time.time()
    while (pending or active) and not stopping:
        local_text = run(["nvidia-smi", "--query-gpu=index,memory.free", "--format=csv,noheader,nounits"], capture_output=True).stdout
        local_mem = {int(row.split(",")[0]): int(row.split(",")[1]) for row in local_text.strip().splitlines()}
        remote_text = remote("nvidia-smi --query-gpu=index,memory.free --format=csv,noheader,nounits", True).stdout
        remote_mem = {int(row.split(",")[0]): int(row.split(",")[1]) for row in remote_text.strip().splitlines()}
        memories = {"local": local_mem, "remote": remote_mem}
        for name, item in list(active.items()):
            process = item["process"]
            if process.poll() is None:
                continue
            job = item["job"]
            if item["host"] == "remote" and process.returncode == 0:
                try:
                    sync_result(job)
                except Exception as error:
                    item["sync_error"] = repr(error)
            success = process.returncode == 0 and output_path(*job).exists()
            event = {key: value for key, value in item.items() if key != "process"}
            event.update(exit_code=process.returncode, success=success, finished_unix=time.time())
            history.append(event)
            save_json(ROOT / "outputs/public_apro_path_evaluation_pool/history.json", history)
            del active[name]
            if not success:
                if attempts[name] < 2:
                    pending.append(job)
                else:
                    failed[name] = item["log"]
        counts = {host: {gpu: 0 for gpu in values} for host, values in memories.items()}
        for item in active.values():
            counts[item["host"]][item["gpu"]] += 1
        for job in list(pending):
            candidates = [(counts[host][gpu], -free, host, gpu)
                          for host, values in memories.items() for gpu, free in values.items()
                          if free - RESERVE_MIB * counts[host][gpu] >= RESERVE_MIB]
            if not candidates:
                break
            _, _, host, gpu = min(candidates)
            name = "__".join(map(str, job))
            attempts[name] = attempts.get(name, 0) + 1
            log = log_root / f"{session}_{name}_{host}_gpu{gpu}_attempt{attempts[name]}.log"
            process = launch(job, host, gpu, log)
            active[name] = {"job": job, "host": host, "gpu": gpu, "pid": process.pid,
                            "process": process, "started_unix": time.time(), "log": str(log),
                            "attempt": attempts[name]}
            counts[host][gpu] += 1
            pending.remove(job)
            print(f"评估任务 {name} -> {host} GPU{gpu}", flush=True)
        completed = sum(output_path(*job).exists() for job in jobs)
        save_json(status_path, {"state": "running" if active else "waiting_for_memory", "completed": completed,
                                "total": len(jobs), "pending": len(pending), "failed": failed,
                                "memory_reservation_mib": RESERVE_MIB, "memory": memories,
                                "updated_unix": time.time()})
        print(f"公共数据集删除评估：完成{completed}/{len(jobs)}，排队{len(pending)}，失败{len(failed)}", flush=True)
        if pending or active:
            time.sleep(POLL_SECONDS)
    for item in active.values():
        if item["process"].poll() is None:
            item["process"].terminate()
    state = "complete" if not pending and not active and not failed else "stopped" if stopping else "incomplete"
    save_json(status_path, {"state": state, "completed": sum(output_path(*job).exists() for job in jobs),
                            "total": len(jobs), "pending": len(pending), "failed": failed,
                            "wall_seconds": time.time() - started, "updated_unix": time.time()})
    if failed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
