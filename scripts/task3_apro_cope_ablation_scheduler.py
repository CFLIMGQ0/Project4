#!/usr/bin/env python3
"""按“Full APro-CoPE优先、其余消融随后”的顺序调度四数据集五折实验。"""

from __future__ import annotations

import argparse
import copy
import json
import os
import subprocess
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml


SRC_ROOT = Path(__file__).resolve().parents[1]
PROJECT_ROOT = SRC_ROOT.parent
PYTHON_BIN = Path("/xmlg/Lim/conda/envs/myenv/bin/python")
RUNNER = SRC_ROOT / "scripts/task3_main_model_5fold.py"
BASE_CONFIG = SRC_ROOT / "configs/task3/t3_main_model.yaml"
TASK_ROOT = PROJECT_ROOT / "outputs/train_runs/task3/t3_apro_cope_ablation"
CONFIG_ROOT = TASK_ROOT / "generated_configs"
LOG_ROOT = TASK_ROOT / "logs"
STATE_PATH = TASK_ROOT / "scheduler_state.json"
RECORDS_CACHE = PROJECT_ROOT / "outputs/train_runs/task3/t3_main_model/records_cache.json"
IMAGE_CACHE_ROOT = PROJECT_ROOT / "datasets/image_cache"
IMAGE_CACHE_MANIFEST = IMAGE_CACHE_ROOT / "task3_cache_manifest.jsonl.gz"
DATASETS = ("regular_white_light", "chromoscopic", "surgical", "ultrasound")
FOLDS = (1, 2, 3, 4, 5)
COMPLETE_RESULT_DIRS = ("test_macro_f1", "test_micro_f1", "test_val_loss")


VARIANT_DEFINITIONS = (
    ("apro_full", "Full APro-CoPE", 0),
    ("no_pe", "No PE + Transformer", 1),
    ("original_pe", "Original PE", 1),
    ("standard_cope", "Bidirectional Standard CoPE", 1),
    ("raw_acquisition_pe", "Raw Acquisition PE", 1),
    ("apro_pairwise", "APro-CoPE Pairwise Transition", 1),
    ("apro_no_conservation", "APro-CoPE w/o Mass Conservation", 1),
    ("apro_absolute_only", "APro-CoPE Absolute Only", 1),
    ("apro_relative_only", "APro-CoPE Relative Only", 1),
)


@dataclass(frozen=True)
class Variant:
    key: str
    display_name: str
    phase: int
    output_dir: Path
    config_path: Path


@dataclass(frozen=True)
class Job:
    variant: Variant
    dataset: str
    fold: int

    @property
    def key(self) -> str:
        return f"{self.variant.key}__{self.dataset}__fold_{self.fold}"

    @property
    def run_dir(self) -> Path:
        return self.variant.output_dir / self.dataset / f"fold_{self.fold}"

    @property
    def log_path(self) -> Path:
        return LOG_ROOT / self.variant.key / f"{self.dataset}_fold_{self.fold}.log"


@dataclass
class ActiveJob:
    job: Job
    gpu: int
    process: subprocess.Popen[str]
    log_file: Any
    started_at: float


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gpus", default="0,1,2,3")
    parser.add_argument("--max-per-gpu", type=int, default=2)
    parser.add_argument("--estimated-memory-mb", type=int, default=6800)
    parser.add_argument("--min-headroom-mb", type=int, default=1000)
    parser.add_argument("--poll-seconds", type=int, default=30)
    parser.add_argument("--max-retries", type=int, default=3)
    parser.add_argument("--variants", default="", help="逗号分隔的配置键；默认全部")
    parser.add_argument("--prepare-only", action="store_true")
    return parser.parse_args()


def read_yaml(path: Path) -> dict[str, Any]:
    payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"配置文件顶层必须为字典：{path}")
    return payload


def atomic_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(f"{path.suffix}.{os.getpid()}.tmp")
    temporary.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    temporary.replace(path)


def base_model_params(position_variant: str) -> dict[str, Any]:
    return {
        "backbone_name": "convnext_tiny",
        "freeze_stages": 1,
        "feature_dim": 512,
        "attn_dim": 256,
        "hidden_dim": 1024,
        "dropout": 0.2,
        "encoder_chunk_size": 16,
        "num_heads": 4,
        "num_layers": 2,
        "use_label_graph": True,
        "label_graph_type": "label_hypergraph",
        "label_hypergraph_edges": 2,
        "text_vocab_size": 8192,
        "text_embed_dim": 128,
        "textcnn_kernel_sizes": [2, 3, 4],
        "image_aux_weight": 0.0,
        "position_variant": position_variant,
        "apro_position_dim": 64,
        "apro_warp_alpha": 1.5,
        "apro_fourier_frequencies": 8,
    }


def run_overrides() -> dict[str, Any]:
    return {
        "batch_size": 1,
        "eval_batch_size": 1,
        "train_max_instances": 64,
        "eval_max_instances": 64,
        "train_max_batch_instances": 128,
        "eval_max_batch_instances": 128,
        "pin_memory": True,
        "persistent_workers": False,
        "loader_prefetch_factor": 1,
        "image_cache_mode": "disk",
        "image_cache_scope": "shared",
        "image_cache_dir": str(IMAGE_CACHE_ROOT),
        "image_cache_manifest": str(IMAGE_CACHE_MANIFEST),
        "image_cache_warmup": False,
        "memory_cache_size": 0,
        "random_instance_dropout": 0.0,
        "optimizer_name": "adamw",
        "lr": 0.00002,
        "weight_decay": 0.02,
        "warmup_ratio": 0.2,
        "grad_accum_steps": 4,
        "amp": True,
        "topk_evidence": 5,
        "loss_name": "asymmetric",
        "monitor_metric": "val_loss",
        "monitor_mode": "min",
        "train_sampling_strategy": "uniform",
        "eval_sampling_strategy": "uniform",
        "modality_level": "report_assist",
        "modality_fields": "image+masked_watch",
        "inference_inputs": "image+masked_watch",
        "leakage_note": "TASK3 APro-CoPE位置编码消融；仅输入图像与类别名称掩码后的watch，不输入watchResult。",
    }


def materialize_variants(selected: set[str] | None) -> list[Variant]:
    base = read_yaml(BASE_CONFIG)
    variants: list[Variant] = []
    known = {key for key, _, _ in VARIANT_DEFINITIONS}
    if selected:
        unknown = selected - known
        if unknown:
            raise ValueError(f"未知位置编码配置：{sorted(unknown)}")
    for key, display_name, phase in VARIANT_DEFINITIONS:
        if selected and key not in selected:
            continue
        output_dir = TASK_ROOT / key
        config_path = CONFIG_ROOT / f"{key}.yaml"
        cfg = copy.deepcopy(base)
        cfg["experiment_name"] = f"t3_{key}"
        cfg["paths"] = {
            "source_csv": str(PROJECT_ROOT / "datasets/task_data/task2/gastro_multilabel_task_datalist.csv"),
            "dataset_root": str(PROJECT_ROOT / "datasets/main_data"),
            "records_cache": str(RECORDS_CACHE),
            "allow_migrated_records_cache": True,
            "output_dir": str(output_dir),
        }
        cfg["model"] = {
            "model_name": "exp12_apro_cope_watch_cross_attn_textcnn",
            "params": base_model_params(key),
        }
        cfg["training"]["num_workers"] = 2
        cfg["training"]["run_overrides"] = run_overrides()
        cfg["entry_metadata"] = {
            "experiment_group": "apro_cope_position_ablation",
            "position_variant": key,
            "display_name": display_name,
            "phase": "full_first" if phase == 0 else "ablation_after_full",
        }
        config_path.parent.mkdir(parents=True, exist_ok=True)
        config_path.write_text(yaml.safe_dump(cfg, allow_unicode=True, sort_keys=False), encoding="utf-8")
        variants.append(Variant(key, display_name, phase, output_dir, config_path))
    return variants


def is_complete(job: Job) -> bool:
    if not (job.run_dir / "test_result.csv").is_file():
        return False
    return all((job.run_dir / name / "metrics.json").is_file() for name in COMPLETE_RESULT_DIRS)


def gpu_free_memory() -> dict[int, int]:
    result = subprocess.run(
        ["nvidia-smi", "--query-gpu=index,memory.free", "--format=csv,noheader,nounits"],
        check=True,
        capture_output=True,
        text=True,
    )
    values: dict[int, int] = {}
    for line in result.stdout.splitlines():
        if line.strip():
            index, free = [item.strip() for item in line.split(",", 1)]
            values[int(index)] = int(free)
    return values


def launch_job(job: Job, gpu: int) -> ActiveJob:
    job.log_path.parent.mkdir(parents=True, exist_ok=True)
    log_file = job.log_path.open("a", encoding="utf-8")
    command = [
        str(PYTHON_BIN),
        str(RUNNER),
        "--config",
        str(job.variant.config_path),
        "--datasets",
        job.dataset,
        "--folds",
        str(job.fold),
    ]
    env = dict(os.environ)
    env["CUDA_VISIBLE_DEVICES"] = str(gpu)
    env["PROJECT4_DISABLE_DISK_CACHE_WRITE"] = "1"
    log_file.write(f"\n[SCHEDULER] GPU={gpu} START={' '.join(command)}\n")
    log_file.flush()
    process = subprocess.Popen(
        command,
        cwd=SRC_ROOT,
        env=env,
        stdout=log_file,
        stderr=subprocess.STDOUT,
        text=True,
        start_new_session=True,
    )
    return ActiveJob(job, gpu, process, log_file, time.time())


def remove_redundant_checkpoints(job: Job) -> None:
    checkpoint_dir = job.run_dir / "checkpoints"
    for name in ("best_micro_f1.ckpt", "best_val_loss.ckpt", "last.ckpt"):
        path = checkpoint_dir / name
        if path.is_file():
            path.unlink()


def write_state(
    *,
    phase: int,
    total: int,
    completed: set[str],
    pending: list[Job],
    active: list[ActiveJob],
    retries: dict[str, int],
) -> None:
    atomic_json(
        STATE_PATH,
        {
            "updated_at": time.strftime("%Y-%m-%d %H:%M:%S"),
            "phase": "full" if phase == 0 else "ablations",
            "total_jobs": total,
            "completed_jobs": len(completed),
            "pending_jobs": len(pending),
            "active_jobs": [
                {
                    "job": item.job.key,
                    "gpu": item.gpu,
                    "pid": item.process.pid,
                    "elapsed_seconds": int(time.time() - item.started_at),
                }
                for item in active
            ],
            "retry_counts": retries,
        },
    )


def summarize_variant(variant: Variant) -> None:
    log_path = LOG_ROOT / variant.key / "summary.log"
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("a", encoding="utf-8") as log_file:
        subprocess.run(
            [str(PYTHON_BIN), str(RUNNER), "--config", str(variant.config_path), "--summarize-only"],
            cwd=SRC_ROOT,
            stdout=log_file,
            stderr=subprocess.STDOUT,
            text=True,
            check=False,
        )


def run_phase(
    *,
    phase: int,
    variants: list[Variant],
    gpu_ids: list[int],
    args: argparse.Namespace,
) -> None:
    jobs = [Job(variant, dataset, fold) for variant in variants for dataset in DATASETS for fold in FOLDS]
    completed = {job.key for job in jobs if is_complete(job)}
    pending = [job for job in jobs if job.key not in completed]
    active: list[ActiveJob] = []
    retries: dict[str, int] = {}
    total = len(jobs)
    print(
        f"[APRO-SCHEDULER] phase={phase} 配置={len(variants)} 总折数={total} "
        f"已完成={len(completed)} 待运行={len(pending)}",
        flush=True,
    )
    while pending or active:
        still_active: list[ActiveJob] = []
        for item in active:
            return_code = item.process.poll()
            if return_code is None:
                still_active.append(item)
                continue
            item.log_file.write(f"[SCHEDULER] EXIT={return_code}\n")
            item.log_file.close()
            if return_code == 0 and is_complete(item.job):
                completed.add(item.job.key)
                remove_redundant_checkpoints(item.job)
            else:
                retries[item.job.key] = retries.get(item.job.key, 0) + 1
                if retries[item.job.key] <= args.max_retries:
                    pending.append(item.job)
                else:
                    print(f"[APRO-SCHEDULER] 达到重试上限：{item.job.key}", flush=True)
        active = still_active

        free_memory = gpu_free_memory()
        counts = {gpu: sum(item.gpu == gpu for item in active) for gpu in gpu_ids}
        reserved = {gpu: 0 for gpu in gpu_ids}
        while pending:
            candidates = [
                gpu
                for gpu in gpu_ids
                if counts[gpu] < args.max_per_gpu
                and free_memory.get(gpu, 0) - reserved[gpu]
                >= args.estimated_memory_mb + args.min_headroom_mb
            ]
            if not candidates:
                break
            gpu = max(candidates, key=lambda value: free_memory.get(value, 0) - reserved[value])
            job = pending.pop(0)
            active.append(launch_job(job, gpu))
            counts[gpu] += 1
            reserved[gpu] += args.estimated_memory_mb
            print(f"[APRO-SCHEDULER] {job.key} -> GPU {gpu}", flush=True)

        write_state(
            phase=phase,
            total=total,
            completed=completed,
            pending=pending,
            active=active,
            retries=retries,
        )
        time.sleep(max(10, int(args.poll_seconds)))

    if len(completed) != total:
        raise RuntimeError(f"phase={phase} 未全部完成：{len(completed)}/{total}，不会进入下一阶段")
    for variant in variants:
        summarize_variant(variant)


def main() -> None:
    args = parse_args()
    if args.max_per_gpu not in {1, 2}:
        raise ValueError("每张GPU仅允许配置1或2个并行任务")
    if not PYTHON_BIN.is_file():
        raise FileNotFoundError(f"训练解释器不存在：{PYTHON_BIN}")
    if not RECORDS_CACHE.is_file() or not IMAGE_CACHE_MANIFEST.is_file():
        raise FileNotFoundError("TASK3样本缓存或图像缓存索引缺失")
    gpu_ids = [int(item.strip()) for item in args.gpus.split(",") if item.strip()]
    selected = {item.strip() for item in args.variants.split(",") if item.strip()} or None
    variants = materialize_variants(selected)
    if args.prepare_only:
        print(f"[APRO-SCHEDULER] 已生成{len(variants)}个配置：{CONFIG_ROOT}")
        return
    for phase in (0, 1):
        phase_variants = [variant for variant in variants if variant.phase == phase]
        if phase_variants:
            run_phase(phase=phase, variants=phase_variants, gpu_ids=gpu_ids, args=args)
    print("[APRO-SCHEDULER] Full APro-CoPE及全部位置编码消融已完成", flush=True)


if __name__ == "__main__":
    main()
