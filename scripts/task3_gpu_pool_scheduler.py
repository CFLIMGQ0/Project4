#!/usr/bin/env python3
"""在 xmlg204 统一调度本机与 xmlg202 的 TASK3 单卡训练任务。"""

from __future__ import annotations

import argparse
import json
import os
import shlex
import subprocess
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

import task3_apro_cope_ablation_scheduler as apro
import task3_remaining_experiments_scheduler as base


POOL_STATE_PATH = base.TASK_ROOT / "pool_scheduler_state.json"


@dataclass(frozen=True)
class PoolHost:
    name: str
    project_root: Path
    python_bin: Path
    gpu_ids: tuple[int, ...]
    ssh_target: str | None = None
    ssh_key: Path | None = None

    @property
    def is_local(self) -> bool:
        return self.ssh_target is None

    @property
    def src_root(self) -> Path:
        return self.project_root / "src"


@dataclass
class PoolActiveJob:
    job: base.Job
    host: PoolHost
    gpu: int
    process: subprocess.Popen[str]
    log_file: Any
    started_at: float


class AdoptedProcess:
    """重启调度器后监视仍在运行的本地训练或SSH控制进程。"""

    def __init__(self, pid: int) -> None:
        self.pid = pid

    def poll(self) -> int | None:
        try:
            stat = Path(f"/proc/{self.pid}/stat").read_text(encoding="utf-8")
        except (FileNotFoundError, PermissionError, ProcessLookupError):
            return 0
        fields = stat.split()
        if len(fields) >= 3 and fields[2] == "Z":
            return 0
        return None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--groups", default="apro,distill,table3,table4")
    parser.add_argument("--variants", default="")
    parser.add_argument("--local-gpus", default="0,1,2,3")
    parser.add_argument("--remote-gpus", default="0,1")
    parser.add_argument("--remote-name", default="xmlg202")
    parser.add_argument("--remote-target", default="Lim@172.16.170.202")
    parser.add_argument("--remote-project-root", default="/home/Lim/Project4")
    parser.add_argument("--remote-python", default="/home/Lim/conda/envs/myenv/bin/python")
    parser.add_argument(
        "--ssh-key",
        default="/home/Lim/.ssh/id_ed25519_project4_pool",
    )
    parser.add_argument("--max-per-gpu", type=int, default=2)
    parser.add_argument("--estimated-memory-mb", type=int, default=6500)
    parser.add_argument("--min-headroom-mb", type=int, default=1000)
    parser.add_argument("--poll-seconds", type=int, default=30)
    parser.add_argument("--cache-preflight-samples", type=int, default=128)
    parser.add_argument("--max-retries", type=int, default=3)
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def parse_gpu_ids(raw: str) -> tuple[int, ...]:
    return tuple(int(item.strip()) for item in raw.split(",") if item.strip())


def ssh_prefix(host: PoolHost) -> list[str]:
    if host.is_local or not host.ssh_target:
        raise ValueError(f"{host.name}不是SSH节点")
    command = ["ssh"]
    if host.ssh_key is not None:
        command.extend(["-i", str(host.ssh_key)])
    command.extend(
        [
            "-o",
            "BatchMode=yes",
            "-o",
            "ConnectTimeout=10",
            host.ssh_target,
        ]
    )
    return command


def rsync_shell(host: PoolHost) -> str:
    parts = ["ssh"]
    if host.ssh_key is not None:
        parts.extend(["-i", str(host.ssh_key)])
    parts.extend(["-o", "BatchMode=yes", "-o", "ConnectTimeout=10"])
    return shlex.join(parts)


def replace_project_root(value: Any, remote_root: Path) -> Any:
    if isinstance(value, dict):
        return {key: replace_project_root(item, remote_root) for key, item in value.items()}
    if isinstance(value, list):
        return [replace_project_root(item, remote_root) for item in value]
    if isinstance(value, str):
        return value.replace(str(base.PROJECT_ROOT), str(remote_root))
    return value


def variant_group(variant: Any) -> str:
    return getattr(variant, "group", "apro")


def variant_config_path(variant: Any) -> Path:
    configured = getattr(variant, "config_path", None)
    if configured is not None:
        return Path(configured)
    return base.CONFIG_ROOT / variant.group / f"{variant.key}.yaml"


def job_config_path(job: Any) -> Path:
    configured = getattr(job, "config_path", None)
    if configured is not None:
        return Path(configured)
    return variant_config_path(job.variant)


def prepare_remote_configs(variants: list[Any], host: PoolHost) -> dict[str, Path]:
    local_root = base.CONFIG_ROOT / f"remote_{host.name}"
    remote_root = host.project_root / "outputs/train_runs/task3/t3_remaining_experiments/generated_configs_pool"
    mapping: dict[str, Path] = {}
    for variant in variants:
        group = variant_group(variant)
        source = variant_config_path(variant)
        payload = yaml.safe_load(source.read_text(encoding="utf-8"))
        payload = replace_project_root(payload, host.project_root)
        destination = local_root / group / f"{variant.key}.yaml"
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_text(
            yaml.safe_dump(payload, allow_unicode=True, sort_keys=False),
            encoding="utf-8",
        )
        mapping[variant.key] = remote_root / group / f"{variant.key}.yaml"

    mkdir_command = f"mkdir -p {shlex.quote(str(remote_root))}"
    subprocess.run(ssh_prefix(host) + [mkdir_command], check=True, text=True)
    subprocess.run(
        [
            "rsync",
            "-a",
            "-e",
            rsync_shell(host),
            f"{local_root}/",
            f"{host.ssh_target}:{remote_root}/",
        ],
        check=True,
        text=True,
    )
    return mapping


def gpu_free_memory(host: PoolHost) -> dict[int, int]:
    query = "nvidia-smi --query-gpu=index,memory.free --format=csv,noheader,nounits"
    try:
        if host.is_local:
            result = subprocess.run(
                shlex.split(query), check=True, capture_output=True, text=True, timeout=15
            )
        else:
            result = subprocess.run(
                ssh_prefix(host) + [query], check=True, capture_output=True, text=True, timeout=20
            )
    except (OSError, subprocess.SubprocessError):
        return {}
    values: dict[int, int] = {}
    for line in result.stdout.splitlines():
        if not line.strip():
            continue
        index, free = [item.strip() for item in line.split(",", 1)]
        values[int(index)] = int(free)
    return values


def remote_cache_preflight(host: PoolHost, sample_count: int) -> dict[str, Any]:
    code = (
        "import json; "
        "from scripts.task3_remaining_experiments_scheduler import cache_preflight; "
        f"print(json.dumps(cache_preflight({int(sample_count)}), ensure_ascii=False))"
    )
    command = (
        f"cd {shlex.quote(str(host.src_root))} && "
        f"{shlex.quote(str(host.python_bin))} -c {shlex.quote(code)}"
    )
    try:
        result = subprocess.run(
            ssh_prefix(host) + [command],
            check=True,
            capture_output=True,
            text=True,
            timeout=180,
        )
        return json.loads(result.stdout.strip().splitlines()[-1])
    except (OSError, subprocess.SubprocessError, json.JSONDecodeError, IndexError) as exc:
        return {"ready": False, "reason": f"远程缓存预检失败：{exc}"}


def remote_path(local_path: Path, host: PoolHost) -> Path:
    relative = local_path.relative_to(base.PROJECT_ROOT)
    return host.project_root / relative


def launch_job(
    job: base.Job,
    host: PoolHost,
    gpu: int,
    remote_configs: dict[str, Path],
) -> PoolActiveJob:
    job.log_path.parent.mkdir(parents=True, exist_ok=True)
    log_file = job.log_path.open("a", encoding="utf-8")
    if host.is_local:
        command = [
            str(base.PYTHON_BIN),
            str(base.RUNNER),
            "--config",
            str(job_config_path(job)),
            "--datasets",
            job.dataset,
            "--folds",
            str(job.fold),
        ]
        env = dict(os.environ)
        env["CUDA_VISIBLE_DEVICES"] = str(gpu)
        env["PROJECT4_DISABLE_DISK_CACHE_WRITE"] = "1"
    else:
        runner = host.src_root / "scripts/task3_main_model_5fold.py"
        conda_lib = host.python_bin.parent.parent / "lib"
        remote_command = [
            str(host.python_bin),
            str(runner),
            "--config",
            str(remote_configs[job.variant.key]),
            "--datasets",
            job.dataset,
            "--folds",
            str(job.fold),
        ]
        shell_command = (
            f"cd {shlex.quote(str(host.src_root))} && "
            f"export CUDA_VISIBLE_DEVICES={int(gpu)} PROJECT4_DISABLE_DISK_CACHE_WRITE=1 "
            f"LD_LIBRARY_PATH={shlex.quote(str(conda_lib))}${{LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}}; "
            f"exec {shlex.join(remote_command)}"
        )
        command = ssh_prefix(host) + [shell_command]
        env = None

    log_file.write(
        f"\n[POOL] HOST={host.name} GPU={gpu} START={shlex.join(command)}\n"
    )
    log_file.flush()
    process = subprocess.Popen(
        command,
        cwd=base.SRC_ROOT,
        env=env,
        stdout=log_file,
        stderr=subprocess.STDOUT,
        text=True,
        start_new_session=True,
    )
    return PoolActiveJob(
        job=job,
        host=host,
        gpu=gpu,
        process=process,
        log_file=log_file,
        started_at=time.time(),
    )


def collect_remote_result(item: PoolActiveJob) -> bool:
    host = item.host
    remote_run = remote_path(item.job.run_dir, host)
    checkpoint_dir = remote_run / "checkpoints"
    cleanup_targets = [
        checkpoint_dir / "best_micro_f1.ckpt",
        checkpoint_dir / "best_val_loss.ckpt",
        checkpoint_dir / "last.ckpt",
    ]
    cleanup = "rm -f -- " + " ".join(shlex.quote(str(path)) for path in cleanup_targets)
    try:
        subprocess.run(ssh_prefix(host) + [cleanup], check=True, text=True, timeout=30)
        item.job.run_dir.mkdir(parents=True, exist_ok=True)
        subprocess.run(
            [
                "rsync",
                "-a",
                "-e",
                rsync_shell(host),
                f"{host.ssh_target}:{remote_run}/",
                f"{item.job.run_dir}/",
            ],
            check=True,
            text=True,
            timeout=1800,
        )
    except (OSError, subprocess.SubprocessError) as exc:
        item.log_file.write(f"[POOL] 远程结果回收失败：{exc}\n")
        item.log_file.flush()
        return False
    return base.is_complete(item.job)


def write_state(
    *,
    total: int,
    pending: list[base.Job],
    active: list[PoolActiveJob],
    completed: set[str],
    retries: dict[str, int],
    free_memory: dict[str, dict[int, int]] | None = None,
    blocker: dict[str, Any] | None = None,
) -> None:
    base.atomic_json(
        POOL_STATE_PATH,
        {
            "updated_at": time.strftime("%Y-%m-%d %H:%M:%S"),
            "mode": "ssh_gpu_pool",
            "total_jobs": total,
            "completed_jobs": len(completed),
            "active_jobs": [
                {
                    "job": item.job.key,
                    "host": item.host.name,
                    "gpu": item.gpu,
                    "controller_pid": item.process.pid,
                    "started_at": time.strftime(
                        "%Y-%m-%d %H:%M:%S", time.localtime(item.started_at)
                    ),
                }
                for item in active
            ],
            "pending_jobs": len(pending),
            "retry_counts": retries,
            "gpu_free_memory_mb": free_memory or {},
            "blocker": blocker,
        },
    )


def restore_active_jobs(
    jobs: list[Any], hosts: tuple[PoolHost, ...]
) -> list[PoolActiveJob]:
    """从最近状态接管任务，避免调度器重启时中断或重复训练。"""
    try:
        payload = json.loads(POOL_STATE_PATH.read_text(encoding="utf-8"))
    except (FileNotFoundError, json.JSONDecodeError, OSError):
        return []
    job_by_key = {job.key: job for job in jobs}
    host_by_name = {host.name: host for host in hosts}
    restored: list[PoolActiveJob] = []
    for record in payload.get("active_jobs", []):
        job = job_by_key.get(record.get("job"))
        host = host_by_name.get(record.get("host"))
        try:
            gpu = int(record["gpu"])
            pid = int(record["controller_pid"])
            started_at = time.mktime(
                time.strptime(record["started_at"], "%Y-%m-%d %H:%M:%S")
            )
        except (KeyError, TypeError, ValueError):
            continue
        if job is None or host is None or gpu not in host.gpu_ids:
            continue
        job.log_path.parent.mkdir(parents=True, exist_ok=True)
        log_file = job.log_path.open("a", encoding="utf-8")
        log_file.write(
            f"[POOL] ADOPTED HOST={host.name} GPU={gpu} CONTROLLER_PID={pid}\n"
        )
        log_file.flush()
        restored.append(
            PoolActiveJob(
                job=job,
                host=host,
                gpu=gpu,
                process=AdoptedProcess(pid),
                log_file=log_file,
                started_at=started_at,
            )
        )
    return restored


def build_selected_work(groups: set[str], selected: set[str]) -> tuple[list[Any], list[Any]]:
    variants: list[Any] = []
    jobs: list[Any] = []
    regular_groups = groups - {"apro"}
    if regular_groups:
        regular_variants = base.build_variants(regular_groups)
        variants.extend(regular_variants)
        jobs.extend(base.build_jobs(regular_variants))
    if "apro" in groups:
        apro_variants = apro.materialize_variants(None)
        variants.extend(apro_variants)
        jobs.extend(
            apro.Job(variant, dataset, fold)
            for variant in apro_variants
            for dataset in apro.DATASETS
            for fold in apro.FOLDS
        )
    if selected:
        known = {variant.key for variant in variants}
        unknown = selected - known
        if unknown:
            raise ValueError(f"不存在配置：{sorted(unknown)}")
        variants = [variant for variant in variants if variant.key in selected]
        jobs = [job for job in jobs if job.variant.key in selected]
    return variants, jobs


def current_phase_jobs(pending: list[Any], active: list[PoolActiveJob]) -> list[Any]:
    """优先完成全部APro-CoPE配置，再放行其他实验。"""
    apro_is_active = any(variant_group(item.job.variant) == "apro" for item in active)
    apro_is_pending = any(variant_group(job.variant) == "apro" for job in pending)
    if apro_is_active or apro_is_pending:
        return [job for job in pending if variant_group(job.variant) == "apro"]
    return pending


def summarize_variants(variants: list[Any]) -> None:
    for variant in variants:
        config_path = variant_config_path(variant)
        log_path = variant.output_dir / "summary.log"
        log_path.parent.mkdir(parents=True, exist_ok=True)
        with log_path.open("a", encoding="utf-8") as log_file:
            subprocess.run(
                [str(base.PYTHON_BIN), str(base.RUNNER), "--config", str(config_path), "--summarize-only"],
                cwd=base.SRC_ROOT,
                stdout=log_file,
                stderr=subprocess.STDOUT,
                text=True,
                check=False,
            )


def main() -> None:
    args = parse_args()
    if args.max_per_gpu < 1:
        raise ValueError("每张GPU的TASK3任务上限必须大于等于1")
    groups = {item.strip() for item in args.groups.split(",") if item.strip()}
    unknown_groups = groups - {"apro", "distill", "table3", "table4"}
    if unknown_groups:
        raise ValueError(f"未知实验组：{sorted(unknown_groups)}")

    local = PoolHost(
        name="xmlg204",
        project_root=base.PROJECT_ROOT,
        python_bin=base.PYTHON_BIN,
        gpu_ids=parse_gpu_ids(args.local_gpus),
    )
    remote = PoolHost(
        name=args.remote_name,
        project_root=Path(args.remote_project_root),
        python_bin=Path(args.remote_python),
        gpu_ids=parse_gpu_ids(args.remote_gpus),
        ssh_target=args.remote_target,
        ssh_key=Path(args.ssh_key),
    )
    hosts = (local, remote)

    selected = {item.strip() for item in args.variants.split(",") if item.strip()}
    variants, jobs = build_selected_work(groups, selected)
    remote_configs = prepare_remote_configs(variants, remote)
    completed = {job.key for job in jobs if base.is_complete(job)}
    active = restore_active_jobs(jobs, hosts)
    active_keys = {item.job.key for item in active}
    pending = [
        job for job in jobs if job.key not in completed and job.key not in active_keys
    ]
    total = len(jobs)
    print(
        f"[T3-POOL] 节点=2，GPU=6，配置={len(variants)}，总折数={total}，"
        f"已完成={len(completed)}，待运行={len(pending)}",
        flush=True,
    )
    if active:
        print(f"[T3-POOL] 已接管运行中任务={len(active)}", flush=True)
    if args.dry_run:
        write_state(
            total=total,
            pending=pending,
            active=[],
            completed=completed,
            retries={},
        )
        return

    local_preflight = base.cache_preflight(args.cache_preflight_samples)
    remote_preflight = remote_cache_preflight(remote, args.cache_preflight_samples)
    if not local_preflight.get("ready") or not remote_preflight.get("ready"):
        blocker = {"xmlg204": local_preflight, remote.name: remote_preflight}
        write_state(
            total=total,
            pending=pending,
            active=[],
            completed=completed,
            retries={},
            blocker=blocker,
        )
        raise RuntimeError(f"卡池缓存预检失败：{blocker}")
    print(f"[T3-POOL] 双节点缓存预检通过：204={local_preflight}，202={remote_preflight}", flush=True)

    retries: dict[str, int] = {}
    preferred_host: dict[str, str] = {}
    while pending or active:
        still_active: list[PoolActiveJob] = []
        for item in active:
            return_code = item.process.poll()
            if return_code is None:
                still_active.append(item)
                continue
            succeeded = False
            if return_code == 0:
                if item.host.is_local:
                    succeeded = base.is_complete(item.job)
                    if succeeded:
                        base.remove_redundant_checkpoints(item.job)
                else:
                    succeeded = collect_remote_result(item)
            item.log_file.write(f"[POOL] HOST={item.host.name} EXIT={return_code} COMPLETE={succeeded}\n")
            item.log_file.close()
            if succeeded:
                completed.add(item.job.key)
                preferred_host.pop(item.job.key, None)
            else:
                retries[item.job.key] = retries.get(item.job.key, 0) + 1
                if retries[item.job.key] <= args.max_retries:
                    pending.append(item.job)
                    preferred_host[item.job.key] = item.host.name
                else:
                    print(f"[T3-POOL] 达到重试上限：{item.job.key}", flush=True)
        active = still_active

        free_by_host = {host.name: gpu_free_memory(host) for host in hosts}
        counts = {
            (host.name, gpu): sum(
                item.host.name == host.name and item.gpu == gpu for item in active
            )
            for host in hosts
            for gpu in host.gpu_ids
        }
        reserved = {key: 0 for key in counts}
        while pending:
            eligible_pending = current_phase_jobs(pending, active)
            if not eligible_pending:
                break
            slots: list[tuple[PoolHost, int]] = []
            for host in hosts:
                for gpu in host.gpu_ids:
                    key = (host.name, gpu)
                    if (
                        counts[key] < args.max_per_gpu
                        and free_by_host[host.name].get(gpu, 0) - reserved[key]
                        >= args.estimated_memory_mb + args.min_headroom_mb
                    ):
                        slots.append((host, gpu))
            if not slots:
                break

            selected_index = -1
            allowed_slots: list[tuple[PoolHost, int]] = []
            for job in eligible_pending:
                index = pending.index(job)
                preference = preferred_host.get(job.key)
                current_slots = [slot for slot in slots if preference in (None, slot[0].name)]
                if current_slots:
                    selected_index = index
                    allowed_slots = current_slots
                    break
            if selected_index < 0:
                break
            job = pending.pop(selected_index)
            host, gpu = max(
                allowed_slots,
                key=lambda slot: free_by_host[slot[0].name].get(slot[1], 0)
                - reserved[(slot[0].name, slot[1])],
            )
            active.append(launch_job(job, host, gpu, remote_configs))
            key = (host.name, gpu)
            counts[key] += 1
            reserved[key] += args.estimated_memory_mb
            print(f"[T3-POOL] 已投放 {job.key} -> {host.name}/GPU{gpu}", flush=True)

        write_state(
            total=total,
            pending=pending,
            active=active,
            completed=completed,
            retries=retries,
            free_memory=free_by_host,
        )
        time.sleep(max(10, args.poll_seconds))

    summarize_variants(variants)
    write_state(
        total=total,
        pending=[],
        active=[],
        completed=completed,
        retries=retries,
    )
    print(f"[T3-POOL] 全部完成：{len(completed)}/{total}", flush=True)


if __name__ == "__main__":
    main()
