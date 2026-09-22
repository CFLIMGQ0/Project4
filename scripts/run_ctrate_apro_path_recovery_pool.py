#!/usr/bin/env python3
"""CT-RATE 两条APro-CoPE路径的seed42位置恢复分片动态任务池。"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import shlex
import subprocess
import sys
import time


ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "src"
EVALUATOR = SRC / "scripts" / "evaluate_ctrate_amef_position_recovery.py"
MERGER = SRC / "scripts" / "merge_ctrate_position_recovery_groups.py"
EXPERIMENT = ROOT / "outputs" / "ct_rate_680" / "experiment"
FULL_CACHE = ROOT / "outputs" / "ct_rate_680" / "position_recovery_apro_path_full136_seed42" / "raw_feature_cache"
FULL_SPLIT = ROOT / "outputs" / "ct_rate_680" / "amef_apro_absolute_only_fivefold" / "fold_4" / "amef_multimodal" / "split_ids.json"
REMOTE_HOST = "Lim@172.16.170.202"
REMOTE_ROOT = "/home/Lim/ct_rate_680_apro_path_recovery_full136_20260920"
SSH = [
    "ssh", "-i", "/home/Lim/.ssh/id_ed25519_project4_pool", "-o", "IdentitiesOnly=yes",
    "-o", "BatchMode=yes", "-o", "ConnectTimeout=8", "-o", "ServerAliveInterval=15",
    "-o", "ServerAliveCountMax=3",
]
VARIANTS = ("apro_absolute_only", "apro_relative_only")
SHARD_COUNT = 24
RESERVED_JOB_MIB = 2000
MIN_HEADROOM_MIB = 500
POLL_SECONDS = 10


def run(command: list[str], **kwargs):
    return subprocess.run(command, check=True, text=True, **kwargs)


def save_json(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


def local_free() -> dict[int, int]:
    output = run(
        ["nvidia-smi", "--query-gpu=index,memory.free", "--format=csv,noheader,nounits"],
        capture_output=True,
    ).stdout
    return {int(index.strip()): int(free.strip()) for index, free in (
        line.split(",", 1) for line in output.splitlines() if line.strip()
    )}


def remote_script(script: str, capture_output: bool = False):
    return run([*SSH, REMOTE_HOST, script], capture_output=capture_output)


def remote_free() -> dict[int, int]:
    output = remote_script(
        "nvidia-smi --query-gpu=index,memory.free --format=csv,noheader,nounits",
        capture_output=True,
    ).stdout
    return {int(index.strip()): int(free.strip()) for index, free in (
        line.split(",", 1) for line in output.splitlines() if line.strip()
    )}


def rsync(source: str | Path, target: str) -> None:
    run(["rsync", "-a", "--partial", "-e", shlex.join(SSH), str(source), target])


def local_shard_root(variant: str) -> Path:
    return ROOT / "outputs" / "ct_rate_680" / f"position_recovery_{variant}_full136_seed42_shards"


def local_final_root(variant: str) -> Path:
    return ROOT / "outputs" / "ct_rate_680" / f"position_recovery_{variant}_full136_seed42"


def remote_shard_root(variant: str) -> str:
    return f"{REMOTE_ROOT}/outputs/ct_rate_680/position_recovery_{variant}_full136_seed42_shards"


def remote_checkpoint_root(variant: str) -> str:
    return f"{REMOTE_ROOT}/outputs/ct_rate_680/amef_{variant}_fivefold"


def read_case_indices() -> list[int]:
    values = sorted(int(value) for value in json.loads(FULL_SPLIT.read_text())["test"])
    if len(values) != 136:
        raise ValueError(f"固定测试集不是136例：{len(values)}")
    return values


def prepare_queue(case_indices: list[int]) -> list[Path]:
    queue_root = ROOT / "outputs" / "ct_rate_680" / "apro_path_full136_recovery_seed42_queue"
    queue_root.mkdir(parents=True, exist_ok=True)
    groups = [case_indices[index::SHARD_COUNT] for index in range(SHARD_COUNT)]
    paths = []
    for index, group in enumerate(groups):
        path = queue_root / f"case_group_{index:02d}.json"
        save_json(path, {"case_indices": group, "seed": 42, "group": index})
        paths.append(path)
    return paths


def prepare_outputs() -> None:
    for variant in VARIANTS:
        root = local_shard_root(variant)
        root.mkdir(parents=True, exist_ok=True)
        for index in range(SHARD_COUNT):
            (root / f"shard_{index:02d}").mkdir(parents=True, exist_ok=True)


def stage_remote(case_files: list[Path]) -> None:
    remote_script(
        "mkdir -p " + shlex.quote(f"{REMOTE_ROOT}/src") + " "
        + shlex.quote(f"{REMOTE_ROOT}/outputs/ct_rate_680") + " "
        + shlex.quote(f"{REMOTE_ROOT}/queue") + " "
        + shlex.quote(f"{REMOTE_ROOT}/full_cache")
    )
    rsync(str(SRC) + "/", f"{REMOTE_HOST}:{REMOTE_ROOT}/src/")
    rsync(str(EXPERIMENT) + "/", f"{REMOTE_HOST}:{REMOTE_ROOT}/outputs/ct_rate_680/experiment/")
    rsync(str(FULL_CACHE) + "/", f"{REMOTE_HOST}:{REMOTE_ROOT}/full_cache/")
    for path in case_files:
        rsync(path, f"{REMOTE_HOST}:{REMOTE_ROOT}/queue/{path.name}")
    for variant in VARIANTS:
        remote_script("mkdir -p " + shlex.quote(remote_checkpoint_root(variant)))
        rsync(
            str(ROOT / "outputs" / "ct_rate_680" / f"amef_{variant}_fivefold") + "/",
            f"{REMOTE_HOST}:{remote_checkpoint_root(variant)}/",
        )


def local_command(variant: str, shard: int, case_file: Path) -> list[str]:
    return [
        sys.executable, "-u", str(EVALUATOR),
        "--checkpoint-root", str(ROOT / "outputs" / "ct_rate_680" / f"amef_{variant}_fivefold"),
        "--output-dir", str(local_shard_root(variant) / f"shard_{shard:02d}"),
        "--case-indices-file", str(case_file), "--nonzero-seeds", "42",
        "--raw-cache-dir", str(FULL_CACHE), "--allow-short-inputs",
        "--skip-eligibility-check", "--skip-plot", "--device", "cuda:0",
    ]


def remote_command(variant: str, shard: int, case_file: Path, gpu: int) -> str:
    remote_case = f"{REMOTE_ROOT}/queue/{case_file.name}"
    return (
        f"cd {shlex.quote(REMOTE_ROOT)} && exec env CUDA_VISIBLE_DEVICES={gpu} "
        "OMP_NUM_THREADS=2 TOKENIZERS_PARALLELISM=false "
        "/home/Lim/conda/envs/myenv/bin/python -u "
        "src/scripts/evaluate_ctrate_amef_position_recovery.py "
        f"--checkpoint-root {shlex.quote(remote_checkpoint_root(variant))} "
        f"--output-dir {shlex.quote(remote_shard_root(variant) + f'/shard_{shard:02d}')} "
        f"--case-indices-file {shlex.quote(remote_case)} --nonzero-seeds 42 "
        f"--raw-cache-dir {shlex.quote(REMOTE_ROOT + '/full_cache')} "
        "--allow-short-inputs --skip-eligibility-check --skip-plot --device cuda:0"
    )


def launch_local(variant: str, shard: int, case_file: Path, gpu: int, log_path: Path) -> subprocess.Popen:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    stream = log_path.open("a", encoding="utf-8")
    stream.write(f"\n[{time.strftime('%F %T')}] 启动本地GPU{gpu}：{variant} shard_{shard:02d}\n")
    stream.flush()
    environment = os.environ.copy()
    environment.update({"CUDA_VISIBLE_DEVICES": str(gpu), "OMP_NUM_THREADS": "2", "TOKENIZERS_PARALLELISM": "false"})
    process = subprocess.Popen(local_command(variant, shard, case_file), cwd=ROOT, env=environment,
                               stdout=stream, stderr=subprocess.STDOUT)
    stream.close()
    return process


def launch_remote(variant: str, shard: int, case_file: Path, gpu: int, log_path: Path) -> subprocess.Popen:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    stream = log_path.open("a", encoding="utf-8")
    stream.write(f"\n[{time.strftime('%F %T')}] 启动202主机GPU{gpu}：{variant} shard_{shard:02d}\n")
    stream.flush()
    process = subprocess.Popen([*SSH, REMOTE_HOST, remote_command(variant, shard, case_file, gpu)],
                               stdout=stream, stderr=subprocess.STDOUT)
    stream.close()
    return process


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--poll-seconds", type=int, default=POLL_SECONDS)
    parser.add_argument("--reserved-job-mib", type=int, default=RESERVED_JOB_MIB)
    parser.add_argument("--min-headroom-mib", type=int, default=MIN_HEADROOM_MIB)
    args = parser.parse_args()

    case_indices = read_case_indices()
    case_files = prepare_queue(case_indices)
    prepare_outputs()
    stage_remote(case_files)
    pending = [(variant, shard) for variant in VARIANTS for shard in range(SHARD_COUNT)]
    active: dict[str, dict] = {}
    retries = {job: 0 for job in pending}
    log_root = ROOT / "outputs" / "ct_rate_680" / "apro_path_recovery_gpu_logs"
    started = time.time()

    while pending or active:
        local_memory = local_free()
        remote_memory = remote_free()
        for key, payload in list(active.items()):
            process = payload["process"]
            if process.poll() is None:
                continue
            del active[key]
            job = (payload["variant"], payload["shard"])
            if process.returncode != 0:
                retries[job] += 1
                if retries[job] <= 2:
                    pending.append(job)
                    print(f"恢复分片失败，重新排队{job}：第{retries[job]}次", flush=True)
                else:
                    raise RuntimeError(f"恢复分片连续失败：{job}")

        reservations: dict[tuple[str, int], int] = {}
        for payload in active.values():
            key = (payload["host"], payload["gpu"])
            reservations[key] = reservations.get(key, 0) + 1
        candidates = []
        for host, memory in (("local", local_memory), ("remote", remote_memory)):
            for gpu, free in memory.items():
                used_slots = reservations.get((host, gpu), 0)
                while free >= (used_slots + 1) * args.reserved_job_mib + args.min_headroom_mib:
                    candidates.append((free - used_slots * args.reserved_job_mib, host, gpu))
                    used_slots += 1
        candidates.sort(reverse=True)
        while pending and candidates:
            _, host, gpu = candidates.pop(0)
            variant, shard = pending.pop(0)
            case_file = case_files[shard]
            log_path = log_root / host / f"gpu_{gpu}_{variant}_shard_{shard:02d}.log"
            process = (
                launch_local(variant, shard, case_file, gpu, log_path)
                if host == "local" else launch_remote(variant, shard, case_file, gpu, log_path)
            )
            key = f"{variant}:shard_{shard:02d}:{host}:gpu_{gpu}:{process.pid}"
            active[key] = {"variant": variant, "shard": shard, "host": host, "gpu": gpu,
                           "process": process, "pid": process.pid, "started_unix": time.time()}

        status = {
            "state": "running", "started_unix": started,
            "pending": [{"variant": variant, "shard": shard} for variant, shard in pending],
            "active": {key: {k: v for k, v in payload.items() if k != "process"}
                       for key, payload in active.items()},
            "local_memory_mib": local_memory, "remote_memory_mib": remote_memory,
            "reserved_job_mib": args.reserved_job_mib,
        }
        save_json(ROOT / "outputs" / "ct_rate_680" / "apro_path_recovery_gpu_pool_status.json", status)
        print(f"恢复任务活动{len(active)}个、排队{len(pending)}个；本地显存{local_memory}；202显存{remote_memory}", flush=True)
        if pending or active:
            time.sleep(args.poll_seconds)

    for variant in VARIANTS:
        remote_exists = subprocess.run(
            [*SSH, REMOTE_HOST, "test -d " + shlex.quote(remote_shard_root(variant))],
            check=False,
        ).returncode == 0
        if remote_exists:
            rsync(f"{REMOTE_HOST}:{remote_shard_root(variant)}/", str(local_shard_root(variant)) + "/")
        shard_paths = [local_shard_root(variant) / f"shard_{index:02d}" for index in range(SHARD_COUNT)]
        run([
            sys.executable, "-u", str(MERGER), "--shards", *map(str, shard_paths),
            "--output", str(local_final_root(variant)),
        ], cwd=ROOT)
    save_json(ROOT / "outputs" / "ct_rate_680" / "apro_path_recovery_gpu_pool_status.json", {
        "state": "complete", "finished_unix": time.time(), "wall_seconds": time.time() - started,
        "variants": VARIANTS, "cases": len(case_indices), "shards_per_variant": SHARD_COUNT,
    })


if __name__ == "__main__":
    main()
