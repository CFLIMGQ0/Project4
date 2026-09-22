#!/usr/bin/env python3
"""CT-RATE 两条 APro-CoPE 路径消融的本地/202主机动态任务池。"""

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
TRAINER = SRC / "scripts" / "run_ctrate_680_amef_no_pe.py"
EXPERIMENT = ROOT / "outputs" / "ct_rate_680" / "experiment"
REMOTE_HOST = "Lim@172.16.170.202"
REMOTE_ROOT = "/home/Lim/ct_rate_680_apro_path_ablation_20260920"
SSH = [
    "ssh", "-i", "/home/Lim/.ssh/id_ed25519_project4_pool", "-o", "IdentitiesOnly=yes",
    "-o", "BatchMode=yes", "-o", "ConnectTimeout=8", "-o", "ServerAliveInterval=15",
    "-o", "ServerAliveCountMax=3",
]
VARIANTS = ("apro_absolute_only", "apro_relative_only")
FOLDS = tuple(range(1, 6))
MINIMUM_FREE_MIB = 7500
POLL_SECONDS = 20


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
    return {
        int(index.strip()): int(free.strip())
        for index, free in (line.split(",", 1) for line in output.splitlines() if line.strip())
    }


def remote_script(script: str, capture_output: bool = False):
    return run([*SSH, REMOTE_HOST, script], capture_output=capture_output)


def remote_free() -> dict[int, int]:
    output = remote_script(
        "nvidia-smi --query-gpu=index,memory.free --format=csv,noheader,nounits",
        capture_output=True,
    ).stdout
    return {
        int(index.strip()): int(free.strip())
        for index, free in (line.split(",", 1) for line in output.splitlines() if line.strip())
    }


def rsync(source: str | Path, target: str) -> None:
    run(["rsync", "-a", "--partial", "-e", shlex.join(SSH), str(source), target])


def output_dir(variant: str) -> Path:
    return ROOT / "outputs" / "ct_rate_680" / f"amef_{variant}_fivefold"


def remote_output_dir(variant: str) -> str:
    return f"{REMOTE_ROOT}/outputs/ct_rate_680/amef_{variant}_fivefold"


def prepare_variant(variant: str) -> Path:
    destination = output_dir(variant)
    destination.mkdir(parents=True, exist_ok=True)
    run([
        sys.executable, "-u", str(TRAINER), "--audit-only",
        "--position-variant", variant, "--output-dir", str(destination),
    ], cwd=ROOT)
    return destination


def stage_remote(variants: list[str]) -> None:
    if len(list((EXPERIMENT / "features").glob("*.npz"))) != 680:
        raise RuntimeError("CT-RATE特征缓存不是680例，停止远端同步")
    remote_script(
        "mkdir -p " + shlex.quote(f"{REMOTE_ROOT}/src") + " "
        + shlex.quote(f"{REMOTE_ROOT}/outputs/ct_rate_680")
    )
    rsync(str(SRC) + "/", f"{REMOTE_HOST}:{REMOTE_ROOT}/src/")
    rsync(str(EXPERIMENT) + "/", f"{REMOTE_HOST}:{REMOTE_ROOT}/outputs/ct_rate_680/experiment/")
    for variant in variants:
        remote_dir = remote_output_dir(variant)
        remote_script("mkdir -p " + shlex.quote(remote_dir))
        rsync(output_dir(variant) / "protocol.json", f"{REMOTE_HOST}:{remote_dir}/protocol.json")


def completed_local(variant: str) -> int:
    return sum((output_dir(variant) / f"fold_{fold}" / "amef_multimodal" / "completed.json").is_file()
               for fold in FOLDS)


def completed_remote(variant: str) -> int:
    result = remote_script(
        "find " + shlex.quote(remote_output_dir(variant))
        + " -path '*/amef_multimodal/completed.json' -type f | wc -l",
        capture_output=True,
    )
    return int(result.stdout.strip())


def local_command(variant: str, fold: int) -> list[str]:
    return [
        sys.executable, "-u", str(TRAINER), "--worker-index", str(fold - 1),
        "--devices", "0", "1", "2", "3", "4", "--output-dir", str(output_dir(variant)),
        "--position-variant", variant,
    ]


def remote_command(variant: str, fold: int, gpu: int) -> str:
    return (
        f"cd {shlex.quote(REMOTE_ROOT)} && exec env CUDA_VISIBLE_DEVICES={gpu} "
        "OMP_NUM_THREADS=4 TOKENIZERS_PARALLELISM=false "
        "/home/Lim/conda/envs/myenv/bin/python -u "
        "src/scripts/run_ctrate_680_amef_no_pe.py "
        f"--worker-index {fold - 1} --devices 0 1 2 3 4 "
        f"--output-dir {shlex.quote(remote_output_dir(variant))} "
        f"--position-variant {variant}"
    )


def launch_local(variant: str, fold: int, gpu: int, log_path: Path) -> subprocess.Popen:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    stream = log_path.open("a", encoding="utf-8")
    stream.write(f"\n[{time.strftime('%F %T')}] 启动本地GPU{gpu}：{variant} fold_{fold}\n")
    stream.flush()
    environment = os.environ.copy()
    environment.update({
        "CUDA_VISIBLE_DEVICES": str(gpu),
        "OMP_NUM_THREADS": "4",
        "TOKENIZERS_PARALLELISM": "false",
    })
    process = subprocess.Popen(
        local_command(variant, fold), cwd=ROOT, env=environment,
        stdout=stream, stderr=subprocess.STDOUT,
    )
    stream.close()
    return process


def launch_remote(variant: str, fold: int, gpu: int, log_path: Path) -> subprocess.Popen:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    stream = log_path.open("a", encoding="utf-8")
    stream.write(f"\n[{time.strftime('%F %T')}] 启动202主机GPU{gpu}：{variant} fold_{fold}\n")
    stream.flush()
    process = subprocess.Popen(
        [*SSH, REMOTE_HOST, remote_command(variant, fold, gpu)],
        stdout=stream, stderr=subprocess.STDOUT,
    )
    stream.close()
    return process


def aggregate(variant: str) -> None:
    run([
        sys.executable, "-u", str(TRAINER), "--aggregate",
        "--position-variant", variant, "--output-dir", str(output_dir(variant)),
    ], cwd=ROOT)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--poll-seconds", type=int, default=POLL_SECONDS)
    parser.add_argument("--min-free-mib", type=int, default=MINIMUM_FREE_MIB)
    parser.add_argument("--variants", default=",".join(VARIANTS))
    args = parser.parse_args()
    variants = [item.strip() for item in args.variants.split(",") if item.strip()]
    unknown = sorted(set(variants) - set(VARIANTS))
    if unknown:
        raise ValueError(f"未知位置路径：{unknown}")
    for variant in variants:
        prepare_variant(variant)
    stage_remote(variants)

    pending = [(variant, fold) for variant in variants for fold in FOLDS
               if not (output_dir(variant) / f"fold_{fold}" / "amef_multimodal" / "completed.json").is_file()]
    active: dict[str, dict] = {}
    retries = {job: 0 for job in pending}
    log_root = ROOT / "outputs" / "ct_rate_680" / "apro_path_ablation_gpu_logs"
    started = time.time()
    while pending or active:
        local_memory = local_free()
        remote_memory = remote_free()
        for key, payload in list(active.items()):
            process = payload["process"]
            if process.poll() is None:
                continue
            del active[key]
            job = (payload["variant"], payload["fold"])
            if process.returncode != 0:
                retries[job] += 1
                if retries[job] <= 3:
                    pending.append(job)
                    print(f"任务失败，重新排队 {job}（第{retries[job]}次）", flush=True)
                else:
                    raise RuntimeError(f"任务连续失败：{job}")

        occupied = {(payload["host"], payload["gpu"]) for payload in active.values()}
        candidates = [
            (free, "local", gpu) for gpu, free in local_memory.items()
            if free >= args.min_free_mib and ("local", gpu) not in occupied
        ] + [
            (free, "remote", gpu) for gpu, free in remote_memory.items()
            if free >= args.min_free_mib and ("remote", gpu) not in occupied
        ]
        candidates.sort(reverse=True)
        while pending and candidates:
            _, host, gpu = candidates.pop(0)
            variant, fold = pending.pop(0)
            log_path = log_root / host / f"gpu_{gpu}_{variant}_fold_{fold}.log"
            process = (
                launch_local(variant, fold, gpu, log_path)
                if host == "local" else launch_remote(variant, fold, gpu, log_path)
            )
            key = f"{variant}:fold_{fold}:{host}:gpu_{gpu}"
            active[key] = {
                "variant": variant, "fold": fold, "host": host, "gpu": gpu,
                "process": process, "pid": process.pid, "started_unix": time.time(),
            }

        progress = {
            "state": "running", "started_unix": started,
            "pending": [{"variant": variant, "fold": fold} for variant, fold in pending],
            "active": {
                key: {k: value for k, value in payload.items() if k != "process"}
                for key, payload in active.items()
            },
            "local_memory_mib": local_memory, "remote_memory_mib": remote_memory,
            "completed": {
                variant: completed_local(variant) + completed_remote(variant)
                for variant in variants
            },
            "total": {variant: len(FOLDS) for variant in variants},
        }
        save_json(ROOT / "outputs" / "ct_rate_680" / "apro_path_ablation_gpu_pool_status.json", progress)
        print(
            "；".join(
                f"{variant}完成{completed_local(variant) + completed_remote(variant)}/5"
                for variant in variants
            ) + f"；活动{len(active)}；排队{len(pending)}",
            flush=True,
        )
        if pending or active:
            time.sleep(args.poll_seconds)

    for variant in variants:
        rsync(f"{REMOTE_HOST}:{remote_output_dir(variant)}/", str(output_dir(variant)) + "/")
        aggregate(variant)
    save_json(ROOT / "outputs" / "ct_rate_680" / "apro_path_ablation_gpu_pool_status.json", {
        "state": "complete", "variants": variants, "finished_unix": time.time(),
        "wall_seconds": time.time() - started,
        "completed": {variant: completed_local(variant) for variant in variants},
    })


if __name__ == "__main__":
    main()
