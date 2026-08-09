#!/usr/bin/env python3
"""生成并动态调度 TASK3 蒸馏、表3和表4的四数据集五折实验。"""

from __future__ import annotations

import argparse
import copy
import gzip
import hashlib
import json
import os
import subprocess
import sys
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
TASK_ROOT = PROJECT_ROOT / "outputs/train_runs/task3/t3_remaining_experiments"
CONFIG_ROOT = TASK_ROOT / "generated_configs"
LOG_ROOT = TASK_ROOT / "logs"
STATE_PATH = TASK_ROOT / "scheduler_state.json"
RECORDS_CACHE = PROJECT_ROOT / "outputs/train_runs/task3/t3_main_model/records_cache.json"
IMAGE_CACHE_ROOT = PROJECT_ROOT / "datasets/image_cache"
IMAGE_CACHE_SHARED = IMAGE_CACHE_ROOT / "shared"
IMAGE_CACHE_MANIFEST = IMAGE_CACHE_ROOT / "task3_cache_manifest.jsonl.gz"
DATASETS = ("regular_white_light", "chromoscopic", "surgical", "ultrasound")
FOLDS = (1, 2, 3, 4, 5)
COMPLETE_RESULT_DIRS = ("test_macro_f1", "test_micro_f1", "test_val_loss")


@dataclass(frozen=True)
class Variant:
    key: str
    group: str
    output_dir: Path
    model_name: str
    model_params: dict[str, Any]
    run_overrides: dict[str, Any]
    entry_metadata: dict[str, Any]


@dataclass(frozen=True)
class Job:
    variant: Variant
    dataset: str
    fold: int
    config_path: Path

    @property
    def key(self) -> str:
        return f"{self.variant.key}__{self.dataset}__fold_{self.fold}"

    @property
    def run_dir(self) -> Path:
        return self.variant.output_dir / self.dataset / f"fold_{self.fold}"

    @property
    def log_path(self) -> Path:
        return LOG_ROOT / self.variant.group / self.variant.key / f"{self.dataset}_fold_{self.fold}.log"


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
    parser.add_argument("--groups", default="distill,table3,table4")
    parser.add_argument(
        "--variants",
        default="",
        help="逗号分隔的配置键；为空时运行所选实验组内的全部配置",
    )
    parser.add_argument("--max-per-gpu", type=int, default=2)
    parser.add_argument("--estimated-memory-mb", type=int, default=6500)
    parser.add_argument("--min-headroom-mb", type=int, default=1000)
    parser.add_argument("--poll-seconds", type=int, default=30)
    parser.add_argument("--cache-wait-seconds", type=int, default=60)
    parser.add_argument("--cache-preflight-samples", type=int, default=128)
    parser.add_argument("--max-retries", type=int, default=3)
    parser.add_argument("--prepare-configs-only", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def read_yaml(path: Path) -> dict[str, Any]:
    payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"配置文件顶层必须是字典：{path}")
    return payload


def atomic_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(f"{path.suffix}.{os.getpid()}.tmp")
    temporary.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    temporary.replace(path)


def base_model_params() -> dict[str, Any]:
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
        "image_distill_weight": 0.0,
        "image_distill_temperature": 2.0,
    }


def common_run_overrides() -> dict[str, Any]:
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
    }


def build_variants(groups: set[str]) -> list[Variant]:
    variants: list[Variant] = []
    base_params = base_model_params()
    base_run = common_run_overrides()

    if "distill" in groups:
        for key, distill_weight in (("image_distill", 1.0), ("image_aux_only", 0.0)):
            params = {**base_params, "image_aux_weight": 0.5, "image_distill_weight": distill_weight}
            run = {
                **base_run,
                "modality_level": "train_time_distill",
                "modality_fields": "image+masked_watch(train_only)",
                "inference_inputs": "image",
                "leakage_note": "训练使用掩码watch，验证和测试严格只输入图像。",
            }
            output_name = "t3_image_branch_distill" if distill_weight > 0 else "t3_image_branch_aux_only"
            variants.append(
                Variant(
                    key=key,
                    group="distill",
                    output_dir=PROJECT_ROOT / f"outputs/train_runs/task3/{output_name}",
                    model_name="exp8_mm_watch_cross_attn_textcnn_image_distill",
                    model_params=params,
                    run_overrides=run,
                    entry_metadata={
                        "experiment_group": "knowledge_distillation",
                        "distillation_enabled": distill_weight > 0,
                        "inference_branch": "image_student",
                        "teacher_branch": "multimodal_fusion",
                    },
                )
            )

    if "table3" in groups:
        for use_context in (True, False):
            for instances in (16, 32, 48, 64, 80, 96):
                if use_context and instances == 64:
                    # 完整64图主模型的20折已经完成。
                    continue
                context_key = "with_context" if use_context else "no_context"
                key = f"{context_key}_instances_{instances}"
                params = {
                    **base_params,
                    "use_context_encoder": use_context,
                    "watch_fusion_mode": "cross_attention",
                    "use_text_gate": True,
                }
                run = {
                    **base_run,
                    "train_max_instances": instances,
                    "eval_max_instances": instances,
                    "train_max_batch_instances": max(128, instances),
                    "eval_max_batch_instances": max(128, instances),
                    "modality_level": "report_assist",
                    "modality_fields": "image+masked_watch",
                    "inference_inputs": "image+masked_watch",
                    "leakage_note": "TASK3表3仅改变图像数量及M1开关；watch已遮蔽类别名称。",
                }
                variants.append(
                    Variant(
                        key=key,
                        group="table3",
                        output_dir=PROJECT_ROOT / f"outputs/train_runs/task3/t3_table3_position_context/{key}",
                        model_name="exp11_module_ablation",
                        model_params=params,
                        run_overrides=run,
                        entry_metadata={
                            "experiment_group": "table3_position_context",
                            "M1": use_context,
                            "M2": True,
                            "M3": True,
                            "M4": True,
                            "max_instances": instances,
                        },
                    )
                )

    if "table4" in groups:
        for mask in range(16):
            active = {module for module in range(1, 5) if mask & (1 << (module - 1))}
            # 1234复用主模型；234复用表3的no_context_instances_64。
            if active in ({1, 2, 3, 4}, {2, 3, 4}):
                continue
            combo = "none" if not active else "".join(str(value) for value in sorted(active))
            use_m3 = 3 in active
            use_m4 = 4 in active
            fusion_mode = "cross_attention" if use_m3 else ("pooled" if use_m4 else "none")
            uses_watch = fusion_mode != "none"
            params = {
                **base_params,
                "use_context_encoder": 1 in active,
                "label_graph_type": "label_hypergraph" if 2 in active else "learnable",
                "watch_fusion_mode": fusion_mode,
                "use_text_gate": use_m4,
            }
            run = {
                **base_run,
                "modality_level": "report_assist" if uses_watch else "strict_deploy",
                "modality_fields": "image+masked_watch" if uses_watch else "image",
                "inference_inputs": "image+masked_watch" if uses_watch else "image",
                "leakage_note": f"TASK3表4四模块全因子组合{combo}；所有含文本组合使用TextCNN和掩码watch。",
            }
            variants.append(
                Variant(
                    key=f"modules_{combo}",
                    group="table4",
                    output_dir=PROJECT_ROOT / f"outputs/train_runs/task3/t3_table4_module_ablation/modules_{combo}",
                    model_name="exp11_module_ablation",
                    model_params=params,
                    run_overrides=run,
                    entry_metadata={
                        "experiment_group": "table4_full_factorial",
                        **{f"M{module}": module in active for module in range(1, 5)},
                        "module_combination": combo,
                    },
                )
            )
    return variants


def materialize_config(variant: Variant) -> Path:
    cfg = copy.deepcopy(read_yaml(BASE_CONFIG))
    cfg["experiment_name"] = variant.key
    cfg["paths"] = {
        "source_csv": str(PROJECT_ROOT / "datasets/task_data/task2/gastro_multilabel_task_datalist.csv"),
        "dataset_root": str(PROJECT_ROOT / "datasets/main_data"),
        "records_cache": str(RECORDS_CACHE),
        "allow_migrated_records_cache": True,
        "output_dir": str(variant.output_dir),
    }
    cfg["model"] = {"model_name": variant.model_name, "params": variant.model_params}
    cfg["training"]["num_workers"] = 2
    cfg["training"]["run_overrides"] = variant.run_overrides
    cfg["entry_metadata"] = variant.entry_metadata
    config_path = CONFIG_ROOT / variant.group / f"{variant.key}.yaml"
    config_path.parent.mkdir(parents=True, exist_ok=True)
    config_path.write_text(yaml.safe_dump(cfg, allow_unicode=True, sort_keys=False), encoding="utf-8")
    return config_path


def build_jobs(variants: list[Variant]) -> list[Job]:
    jobs: list[Job] = []
    for variant in variants:
        config_path = materialize_config(variant)
        for dataset in DATASETS:
            for fold in FOLDS:
                jobs.append(Job(variant=variant, dataset=dataset, fold=fold, config_path=config_path))
    return jobs


def is_complete(job: Job) -> bool:
    if not (job.run_dir / "test_result.csv").is_file():
        return False
    return all((job.run_dir / name / "metrics.json").is_file() for name in COMPLETE_RESULT_DIRS)


def remove_redundant_checkpoints(job: Job) -> None:
    checkpoint_dir = job.run_dir / "checkpoints"
    for name in ("best_micro_f1.ckpt", "best_val_loss.ckpt", "last.ckpt"):
        path = checkpoint_dir / name
        if path.is_file():
            path.unlink()


def cache_path(image_path: str, cache_image_size: int = 336) -> Path:
    source = Path(image_path).expanduser()
    try:
        resolved = source.resolve(strict=True)
        stat_result = resolved.stat()
        signature = f"{resolved}|{stat_result.st_size}|{stat_result.st_mtime_ns}|{cache_image_size}"
    except FileNotFoundError:
        resolved = source.resolve()
        signature = f"{resolved}|{cache_image_size}"
    digest = hashlib.sha1(signature.encode("utf-8")).hexdigest()
    return IMAGE_CACHE_SHARED / digest[:2] / f"{digest}.npy"


def cache_preflight(sample_count: int) -> dict[str, Any]:
    if not RECORDS_CACHE.is_file():
        return {"ready": False, "reason": f"缺少样本缓存：{RECORDS_CACHE}"}
    payload = json.loads(RECORDS_CACHE.read_text(encoding="utf-8"))
    all_paths = [
        image_path
        for record in payload.get("records", [])
        for image_path in record.get("image_paths", [])
    ]
    if not all_paths:
        return {"ready": False, "reason": "迁移样本缓存中没有图像路径"}
    if not IMAGE_CACHE_MANIFEST.is_file():
        return {"ready": False, "reason": f"缺少迁移缓存索引：{IMAGE_CACHE_MANIFEST}"}
    manifest_mapping: dict[str, str] = {}
    with gzip.open(IMAGE_CACHE_MANIFEST, "rt", encoding="utf-8") as file:
        for line in file:
            if not line.strip():
                continue
            row = json.loads(line)
            manifest_mapping[str(row["image_path"])] = str(row["cache_relpath"])
    count = min(max(1, sample_count), len(all_paths))
    indexes = [round(index * (len(all_paths) - 1) / max(1, count - 1)) for index in range(count)]
    sampled = [all_paths[index] for index in indexes]
    hits = sum(
        bool(manifest_mapping.get(path))
        and (IMAGE_CACHE_SHARED / manifest_mapping[path]).is_file()
        for path in sampled
    )
    return {
        "ready": hits == len(sampled),
        "sampled": len(sampled),
        "hits": hits,
        "total_record_images": len(all_paths),
        "total_cache_files": sum(1 for _ in IMAGE_CACHE_SHARED.rglob("*.npy")),
        "manifest_entries": len(manifest_mapping),
        "reason": "" if hits == len(sampled) else "迁移缓存索引未覆盖抽查图像",
    }


def gpu_free_memory() -> dict[int, int]:
    command = [
        "nvidia-smi",
        "--query-gpu=index,memory.free",
        "--format=csv,noheader,nounits",
    ]
    result = subprocess.run(command, check=True, capture_output=True, text=True)
    values: dict[int, int] = {}
    for line in result.stdout.splitlines():
        if not line.strip():
            continue
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
        str(job.config_path),
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
    return ActiveJob(job=job, gpu=gpu, process=process, log_file=log_file, started_at=time.time())


def write_state(
    *,
    total: int,
    pending: list[Job],
    active: list[ActiveJob],
    completed: set[str],
    failures: dict[str, int],
    blocker: dict[str, Any] | None = None,
) -> None:
    atomic_json(
        STATE_PATH,
        {
            "updated_at": time.strftime("%Y-%m-%d %H:%M:%S"),
            "total_jobs": total,
            "completed_jobs": len(completed),
            "active_jobs": [
                {
                    "job": item.job.key,
                    "gpu": item.gpu,
                    "pid": item.process.pid,
                    "started_at": time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(item.started_at)),
                }
                for item in active
            ],
            "pending_jobs": len(pending),
            "retry_counts": failures,
            "blocker": blocker,
        },
    )


def summarize_variants(variants: list[Variant]) -> None:
    for variant in variants:
        config_path = CONFIG_ROOT / variant.group / f"{variant.key}.yaml"
        log_path = LOG_ROOT / variant.group / variant.key / "summary.log"
        log_path.parent.mkdir(parents=True, exist_ok=True)
        with log_path.open("a", encoding="utf-8") as log_file:
            subprocess.run(
                [str(PYTHON_BIN), str(RUNNER), "--config", str(config_path), "--summarize-only"],
                cwd=SRC_ROOT,
                stdout=log_file,
                stderr=subprocess.STDOUT,
                text=True,
                check=False,
            )


def main() -> None:
    args = parse_args()
    gpu_ids = [int(item.strip()) for item in args.gpus.split(",") if item.strip()]
    groups = {item.strip() for item in args.groups.split(",") if item.strip()}
    unknown_groups = groups - {"distill", "table3", "table4"}
    if unknown_groups:
        raise ValueError(f"未知实验组：{sorted(unknown_groups)}")
    if args.max_per_gpu < 1 or args.max_per_gpu > 2:
        raise ValueError("本调度器要求每张GPU最多1或2个TASK3任务")
    if not PYTHON_BIN.is_file():
        raise FileNotFoundError(f"未找到训练Python：{PYTHON_BIN}")

    variants = build_variants(groups)
    selected_variants = {item.strip() for item in args.variants.split(",") if item.strip()}
    if selected_variants:
        known_variants = {variant.key for variant in variants}
        unknown_variants = selected_variants - known_variants
        if unknown_variants:
            raise ValueError(f"所选实验组中不存在配置：{sorted(unknown_variants)}")
        variants = [variant for variant in variants if variant.key in selected_variants]
    jobs = build_jobs(variants)
    completed = {job.key for job in jobs if is_complete(job)}
    pending = [job for job in jobs if job.key not in completed]
    total = len(jobs)
    print(f"[T3-SCHEDULER] 配置数={len(variants)}，总折数={total}，已完成={len(completed)}，待运行={len(pending)}")
    if args.prepare_configs_only or args.dry_run:
        write_state(total=total, pending=pending, active=[], completed=completed, failures={})
        return

    active: list[ActiveJob] = []
    retries: dict[str, int] = {}
    cache_ready = False
    while pending or active:
        if not cache_ready:
            preflight = cache_preflight(args.cache_preflight_samples)
            if not preflight["ready"]:
                write_state(
                    total=total,
                    pending=pending,
                    active=active,
                    completed=completed,
                    failures=retries,
                    blocker=preflight,
                )
                print(f"[T3-SCHEDULER] 缓存预检未通过：{preflight}；{args.cache_wait_seconds}秒后重试", flush=True)
                time.sleep(max(10, args.cache_wait_seconds))
                continue
            cache_ready = True
            print(f"[T3-SCHEDULER] 缓存预检通过：{preflight}", flush=True)

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
                    print(f"[T3-SCHEDULER] 达到重试上限：{item.job.key}", flush=True)
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
            gpu = max(candidates, key=lambda item: free_memory.get(item, 0) - reserved[item])
            job = pending.pop(0)
            active.append(launch_job(job, gpu))
            counts[gpu] += 1
            reserved[gpu] += args.estimated_memory_mb
            print(f"[T3-SCHEDULER] 已投放 {job.key} -> GPU {gpu}", flush=True)

        write_state(
            total=total,
            pending=pending,
            active=active,
            completed=completed,
            failures=retries,
        )
        time.sleep(max(10, args.poll_seconds))

    summarize_variants(variants)
    write_state(total=total, pending=[], active=[], completed=completed, failures=retries)
    print(f"[T3-SCHEDULER] 全部完成：{len(completed)}/{total}", flush=True)


if __name__ == "__main__":
    main()
