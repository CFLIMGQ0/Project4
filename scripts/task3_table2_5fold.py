#!/usr/bin/env python3
"""在四类胃镜子数据集上运行论文表2全部模型的五折实验。"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import shutil
import sys
from pathlib import Path
from typing import Any

import numpy as np
import yaml


def _ensure_project_runtime_python() -> None:
    try:
        import torch  # noqa: F401
        return
    except ImportError:
        pass
    candidate = Path("/home/Lim/anaconda3/envs/myenv/bin/python")
    if candidate.is_file() and Path(sys.executable).resolve() != candidate.resolve():
        os.execv(str(candidate), [str(candidate), *sys.argv])


_ensure_project_runtime_python()

import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from exp_10.data import TextRecord, build_train_vocabulary
from exp_10.masking import contains_answer_term
from exp_10.train_text_classification import train_one_model
from scripts.task1_table_5fold import (
    build_fold_context,
    inherit_run_overrides,
    source_run_payload,
    summarize_values,
)
from scripts.task3_main_model_5fold import (
    LABEL_NAMES,
    apply_watch_mask,
    label_stats,
    prepare_dataset_folds,
    records_cache_meta,
    split_dataset_records,
    write_csv,
    write_json,
)
from train import (
    auto_series_resume_checkpoint,
    is_auto_series_run_complete,
    load_existing_auto_series_result,
    load_model_config,
    load_train_config,
    run_model_job,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        type=Path,
        default=ROOT / "configs/task3/t3_table2_5fold.yaml",
    )
    parser.add_argument("--modality", choices=("image", "text", "all"), default="all")
    parser.add_argument("--models", default="", help="逗号分隔的配置模型名；默认选择对应模态全部模型")
    parser.add_argument("--datasets", default="", help="逗号分隔的数据集键；默认四个数据集")
    parser.add_argument("--folds", default="", help="逗号分隔的折号；默认1至5折")
    parser.add_argument("--num-shards", type=int, default=1)
    parser.add_argument("--shard-index", type=int, default=0)
    parser.add_argument("--prepare-only", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--summarize-only", action="store_true")
    return parser.parse_args()


def read_yaml(path: Path) -> dict[str, Any]:
    payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"配置文件顶层必须是字典：{path}")
    return payload


def parse_selection(raw: str, allowed: list[str], value_name: str) -> list[str]:
    if not raw.strip():
        return list(allowed)
    selected = [item.strip() for item in raw.split(",") if item.strip()]
    unknown = sorted(set(selected) - set(allowed))
    if unknown:
        raise ValueError(f"未知{value_name}：{unknown}；允许值={allowed}")
    return selected


def validate_shard(num_shards: int, shard_index: int) -> None:
    if num_shards < 1:
        raise ValueError("num_shards 必须大于等于1")
    if shard_index < 0 or shard_index >= num_shards:
        raise ValueError("shard_index 必须满足 0 <= shard_index < num_shards")


def load_cached_records(cfg: dict[str, Any]) -> list[dict[str, Any]]:
    path_cfg = cfg["paths"]
    cache_path = Path(path_cfg["records_cache"]).expanduser().resolve()
    if not cache_path.is_file():
        raise FileNotFoundError(f"未找到既有样本缓存，按要求不自动重建：{cache_path}")
    payload = json.loads(cache_path.read_text(encoding="utf-8"))
    source_csv = Path(path_cfg["source_csv"]).expanduser().resolve()
    dataset_root = Path(path_cfg["dataset_root"]).expanduser().resolve()
    expected_meta = records_cache_meta(source_csv, dataset_root)
    if payload.get("meta") != expected_meta:
        raise RuntimeError(f"既有样本缓存元数据与当前数据不一致：{cache_path}")
    records = payload.get("records")
    if not isinstance(records, list) or not records:
        raise RuntimeError(f"既有样本缓存为空或格式错误：{cache_path}")
    print(f"[TASK3-T2] 复用样本缓存：{cache_path}，样本数={len(records)}")
    return records


def prepare_records_and_folds(
    cfg: dict[str, Any],
    dataset_names: list[str],
    output_dir: Path,
) -> dict[str, list[list[dict[str, Any]]]]:
    records = load_cached_records(cfg)
    mask_audit = apply_watch_mask(records, True)
    grouped, excluded_titles = split_dataset_records(records, cfg)
    split_root = output_dir / "data_splits"
    prepared: dict[str, list[list[dict[str, Any]]]] = {}
    for dataset_name in dataset_names:
        prepared[dataset_name] = prepare_dataset_folds(
            dataset_name=dataset_name,
            records=grouped[dataset_name],
            cfg=cfg,
            output_dir=split_root,
        )
        print(f"[TASK3-T2] {dataset_name}: {label_stats(grouped[dataset_name])}")
    write_json(
        output_dir / "data_audit.json",
        {
            "source_records": len(records),
            "watch_mask": mask_audit,
            "excluded_records": len(excluded_titles),
            "excluded_report_titles": sorted(set(excluded_titles)),
            "datasets": {name: label_stats(items) for name, items in grouped.items()},
        },
    )
    return prepared


def build_raw_split(
    folds: list[list[dict[str, Any]]],
    fold_index: int,
) -> dict[str, list[dict[str, Any]]]:
    test_index = fold_index - 1
    val_index = fold_index % len(folds)
    return {
        "train": [
            record
            for index, fold in enumerate(folds)
            if index not in {test_index, val_index}
            for record in fold
        ],
        "val": list(folds[val_index]),
        "test": list(folds[test_index]),
    }


def image_model_entries(cfg: dict[str, Any]) -> list[dict[str, Any]]:
    entries: list[dict[str, Any]] = []
    common_overrides = dict(cfg["image_training"].get("common_run_overrides", {}))
    for item in cfg["image_models"]:
        source_dir = Path(item["source_run_dir"]).expanduser().resolve()
        source = source_run_payload(source_dir)
        run_overrides = inherit_run_overrides(source)
        run_overrides.update(common_overrides)
        run_overrides.pop("seed", None)
        model_params = source.get("model_params")
        if not isinstance(model_params, dict):
            raise ValueError(f"{source_dir}/config.yaml 缺少 model_params")
        entries.append(
            {
                **item,
                "source_run_dir": str(source_dir),
                "base_model_name": str(source["model_name"]),
                "model_params": dict(model_params),
                "run_overrides": run_overrides,
            }
        )
    return entries


def selected_model_entries(
    args: argparse.Namespace,
    cfg: dict[str, Any],
    modality: str,
) -> list[dict[str, Any]]:
    entries = image_model_entries(cfg) if modality == "image" else [dict(item) for item in cfg["text_models"]]
    names = parse_selection(args.models, [str(item["name"]) for item in entries], f"{modality}模型")
    name_set = set(names)
    return [item for item in entries if str(item["name"]) in name_set]


def check_disk_space(output_dir: Path, min_free_gb: float) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    free_gb = shutil.disk_usage(output_dir).free / (1024**3)
    if free_gb < min_free_gb:
        raise RuntimeError(
            f"剩余磁盘空间仅 {free_gb:.1f} GB，低于安全阈值 {min_free_gb:.1f} GB，停止新折训练"
        )


def cleanup_image_checkpoints(run_dir: Path, cfg: dict[str, Any]) -> None:
    if not is_auto_series_run_complete(run_dir):
        return
    retention = cfg["image_training"].get("checkpoint_retention", {})
    removable = retention.get("remove_after_completed_test", [])
    for filename in removable:
        path = run_dir / "checkpoints" / str(filename)
        if path.is_file():
            path.unlink()
            print(f"[TASK3-T2] 已清理冗余checkpoint：{path}")


def run_image_jobs(
    *,
    args: argparse.Namespace,
    cfg: dict[str, Any],
    output_dir: Path,
    prepared_folds: dict[str, list[list[dict[str, Any]]]],
    dataset_names: list[str],
    fold_indices: list[int],
) -> None:
    if not torch.cuda.is_available():
        raise RuntimeError("图像模型训练无法识别GPU")
    active_gpu_count = torch.cuda.device_count()
    image_cfg = cfg["image_training"]
    if bool(image_cfg.get("use_multi_gpu", True)) and active_gpu_count < 2:
        raise RuntimeError("表2图像模型保持原全局batch size运行，至少需要2张可见GPU")

    torch.set_float32_matmul_precision("medium")
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.benchmark = True
    train_cfg = load_train_config(ROOT / "configs/task2/train.yaml")
    model_cfg = load_model_config(ROOT / "configs/task2/model.yaml")
    entries = selected_model_entries(args, cfg, "image")
    jobs = [
        (entry, dataset_name, fold_index)
        for entry in entries
        for dataset_name in dataset_names
        for fold_index in fold_indices
    ]
    jobs = [job for index, job in enumerate(jobs) if index % args.num_shards == args.shard_index]
    print(f"[TASK3-T2] 图像任务数={len(jobs)}，可见GPU={active_gpu_count}")
    if args.dry_run:
        for entry, dataset_name, fold_index in jobs:
            print(f"[DRY-RUN] image/{entry['name']}/{dataset_name}/fold_{fold_index}")
        return

    base_context = {
        "output_root": output_dir,
        "task_selection_dir": str(Path(cfg["paths"]["source_csv"]).expanduser().resolve().parent.parent),
    }
    for entry, dataset_name, fold_index in jobs:
        run_dir = output_dir / "image" / dataset_name / f"fold_{fold_index}" / str(entry["name"])
        if is_auto_series_run_complete(run_dir):
            cleanup_image_checkpoints(run_dir, cfg)
            print(f"[TASK3-T2] 跳过已完成：image/{entry['name']}/{dataset_name}/fold_{fold_index}")
            continue
        check_disk_space(output_dir, float(image_cfg.get("min_free_disk_gb", 40)))
        raw_split = build_raw_split(prepared_folds[dataset_name], fold_index)
        fold_seed = int(cfg["seed"]) + fold_index
        effective_train_cfg = {
            **train_cfg,
            "seed": fold_seed,
            "class_balance": dict(cfg["class_balance"]),
        }
        fold_context = build_fold_context(
            base_context=base_context,
            task_csv=Path(cfg["paths"]["source_csv"]).expanduser().resolve(),
            split_data=raw_split,
            train_cfg=effective_train_cfg,
            task_name="task2",
        )
        balance_report = fold_context["tasks"]["task2"].get("balance_report")
        if balance_report:
            write_json(run_dir / "class_balance_report.json", balance_report)
        print(
            f"[TASK3-T2] 开始：image/{entry['name']}/{dataset_name}/fold_{fold_index}，"
            f"GPU={active_gpu_count}张"
        )
        result = run_model_job(
            model_name=str(entry["base_model_name"]),
            run_dir=run_dir,
            train_cfg=effective_train_cfg,
            model_cfg=model_cfg,
            training_context=fold_context,
            seed=fold_seed,
            max_epochs=int(image_cfg["max_epochs"]),
            patience=int(image_cfg["patience"]),
            image_size=int(image_cfg["image_size"]),
            num_workers=int(image_cfg["num_workers"]),
            pretrained=bool(image_cfg["pretrained"]),
            use_multi_gpu=bool(image_cfg.get("use_multi_gpu", True)),
            active_gpu_count=active_gpu_count,
            run_test=bool(image_cfg["run_test"]),
            run_overrides=dict(entry["run_overrides"]),
            model_param_override=dict(entry["model_params"]),
            entry_metadata={
                "task3_table2": True,
                "model": entry["name"],
                "display_name": entry["display_name"],
                "modality": str(entry.get("modality", "image")),
                "dataset": dataset_name,
                "fold": fold_index,
                "source_run_dir": entry["source_run_dir"],
            },
            resume_path=auto_series_resume_checkpoint(run_dir),
        )
        metrics = result.get("test_results", {}).get(cfg["selection_alias"], {}).get("metrics", {})
        print(
            f"[TASK3-T2] 完成：image/{entry['name']}/{dataset_name}/fold_{fold_index}，"
            f"macro_f1={metrics.get('macro_f1')}"
        )
        cleanup_image_checkpoints(run_dir, cfg)


def as_text_record(record: dict[str, Any]) -> TextRecord:
    text_raw = record.get("text_raw", {})
    watch = str(record.get("watch", text_raw.get("watch", "") if isinstance(text_raw, dict) else ""))
    if contains_answer_term(watch):
        raise RuntimeError(f"遮蔽文本仍包含答案词：{record.get('exam_dir', '')}")
    return TextRecord(
        patient_id=str(record.get("patient_id", "")),
        exam_dir=str(record.get("exam_dir", "")),
        masked_text=watch,
        labels=np.asarray(record["labels"], dtype=np.int64),
        mask_hits=(),
    )


def run_text_jobs(
    *,
    args: argparse.Namespace,
    cfg: dict[str, Any],
    output_dir: Path,
    prepared_folds: dict[str, list[list[dict[str, Any]]]],
    dataset_names: list[str],
    fold_indices: list[int],
) -> None:
    entries = selected_model_entries(args, cfg, "text")
    jobs = [
        (entry, dataset_name, fold_index)
        for entry in entries
        for dataset_name in dataset_names
        for fold_index in fold_indices
    ]
    jobs = [job for index, job in enumerate(jobs) if index % args.num_shards == args.shard_index]
    print(
        f"[TASK3-T2] 文本任务数={len(jobs)}，"
        f"shard={args.shard_index + 1}/{args.num_shards}"
    )
    if args.dry_run:
        for entry, dataset_name, fold_index in jobs:
            print(f"[DRY-RUN] text/{entry['name']}/{dataset_name}/fold_{fold_index}")
        return

    text_config_path = Path(cfg["text_config"]).expanduser().resolve()
    base_text_cfg = read_yaml(text_config_path)
    train_cfg = load_train_config(ROOT / "configs/task2/train.yaml")
    base_context = {
        "output_root": output_dir,
        "task_selection_dir": str(Path(cfg["paths"]["source_csv"]).expanduser().resolve().parent.parent),
    }
    for entry, dataset_name, fold_index in jobs:
        fold_dir = output_dir / "text" / dataset_name / f"fold_{fold_index}"
        run_dir = fold_dir / str(entry["name"])
        metrics_path = run_dir / "test_metrics.json"
        if metrics_path.is_file():
            print(f"[TASK3-T2] 跳过已完成：text/{entry['name']}/{dataset_name}/fold_{fold_index}")
            continue
        raw_split = build_raw_split(prepared_folds[dataset_name], fold_index)
        fold_seed = int(cfg["seed"]) + fold_index
        effective_train_cfg = {
            **train_cfg,
            "seed": fold_seed,
            "class_balance": dict(cfg["class_balance"]),
        }
        fold_context = build_fold_context(
            base_context=base_context,
            task_csv=Path(cfg["paths"]["source_csv"]).expanduser().resolve(),
            split_data=raw_split,
            train_cfg=effective_train_cfg,
            task_name="task2",
        )
        balanced_split = fold_context["tasks"]["task2"]["split"]
        text_splits = {
            split: [as_text_record(record) for record in records]
            for split, records in balanced_split.items()
        }
        original_train_text = [as_text_record(record) for record in raw_split["train"]]
        data_cfg = base_text_cfg["data"]
        vocabulary = build_train_vocabulary(
            original_train_text,
            max_vocab_size=int(data_cfg["vocab_size"]),
            min_frequency=int(data_cfg["min_token_frequency"]),
        )
        text_cfg = {
            **base_text_cfg,
            "seed": fold_seed,
            "experiment_name": cfg["experiment_name"],
        }
        run_dir.mkdir(parents=True, exist_ok=True)
        write_json(run_dir / "config.json", text_cfg)
        write_json(run_dir / "vocabulary.json", vocabulary)
        balance_report = fold_context["tasks"]["task2"].get("balance_report")
        if balance_report:
            write_json(run_dir / "class_balance_report.json", balance_report)
        print(f"[TASK3-T2] 开始：text/{entry['name']}/{dataset_name}/fold_{fold_index}")
        metrics = train_one_model(
            str(entry["name"]),
            text_cfg,
            text_splits,
            vocabulary,
            fold_dir,
        )
        print(
            f"[TASK3-T2] 完成：text/{entry['name']}/{dataset_name}/fold_{fold_index}，"
            f"macro_f1={metrics.get('macro_f1')}"
        )


def image_fold_metrics(
    run_dir: Path,
    base_model_name: str,
    selection_alias: str,
) -> dict[str, Any] | None:
    if not is_auto_series_run_complete(run_dir):
        return None
    result = load_existing_auto_series_result(run_dir, base_model_name)
    payload = result.get("test_results", {}).get(selection_alias, {})
    metrics = payload.get("metrics", {}) if isinstance(payload, dict) else {}
    return {
        **metrics,
        "best_epoch": payload.get("best_epoch", ""),
        "checkpoint_path": payload.get("checkpoint_path", ""),
    }


def summarize_rows(values: list[dict[str, Any]], metrics: list[str]) -> dict[str, Any]:
    payload: dict[str, Any] = {"completed_folds": len(values), "metrics": {}}
    for metric_name in metrics:
        numeric = []
        for row in values:
            try:
                value = float(row.get(metric_name, float("nan")))
            except (TypeError, ValueError):
                value = float("nan")
            if not math.isnan(value):
                numeric.append(value)
        payload["metrics"][metric_name] = summarize_values(numeric)
    return payload


def summarize_all(cfg: dict[str, Any], output_dir: Path) -> None:
    metrics = list(cfg["metrics"])
    global_rows: list[dict[str, Any]] = []
    image_entries = image_model_entries(cfg)
    for entry in image_entries:
        for dataset_name in cfg["datasets"]:
            fold_rows: list[dict[str, Any]] = []
            for fold_index in range(1, int(cfg["folds"]) + 1):
                run_dir = output_dir / "image" / dataset_name / f"fold_{fold_index}" / str(entry["name"])
                result = image_fold_metrics(
                    run_dir,
                    str(entry["base_model_name"]),
                    str(cfg["selection_alias"]),
                )
                if result is not None:
                    fold_rows.append(
                        {
                            "model": entry["name"],
                            "display_name": entry["display_name"],
                            "modality": str(entry.get("modality", "image")),
                            "dataset": dataset_name,
                            "fold": fold_index,
                            **result,
                        }
                    )
            summary = {
                "model": entry["name"],
                "display_name": entry["display_name"],
                "modality": str(entry.get("modality", "image")),
                "dataset": dataset_name,
                **summarize_rows(fold_rows, metrics),
            }
            model_dir = output_dir / "summaries" / str(entry["name"]) / dataset_name
            write_csv(model_dir / "fold_results.csv", fold_rows)
            write_json(model_dir / "fivefold_summary.json", summary)
            global_rows.append(flatten_summary(summary, metrics))

    for entry in cfg["text_models"]:
        for dataset_name in cfg["datasets"]:
            fold_rows = []
            for fold_index in range(1, int(cfg["folds"]) + 1):
                path = (
                    output_dir
                    / "text"
                    / dataset_name
                    / f"fold_{fold_index}"
                    / str(entry["name"])
                    / "test_metrics.json"
                )
                if not path.is_file():
                    continue
                payload = json.loads(path.read_text(encoding="utf-8"))
                fold_rows.append(
                    {
                        "model": entry["name"],
                        "display_name": entry["display_name"],
                        "modality": "text",
                        "dataset": dataset_name,
                        "fold": fold_index,
                        **{metric: payload.get(metric, float("nan")) for metric in metrics},
                        "best_epoch": payload.get("best_epoch", ""),
                    }
                )
            summary = {
                "model": entry["name"],
                "display_name": entry["display_name"],
                "modality": "text",
                "dataset": dataset_name,
                **summarize_rows(fold_rows, metrics),
            }
            model_dir = output_dir / "summaries" / str(entry["name"]) / dataset_name
            write_csv(model_dir / "fold_results.csv", fold_rows)
            write_json(model_dir / "fivefold_summary.json", summary)
            global_rows.append(flatten_summary(summary, metrics))

    main_summary_path = (
        Path(cfg["paths"]["main_model_output_dir"]).expanduser().resolve()
        / "t3_main_model_summary.json"
    )
    if main_summary_path.is_file():
        for item in json.loads(main_summary_path.read_text(encoding="utf-8")):
            row = {
                "model": cfg["main_model"]["name"],
                "display_name": cfg["main_model"]["display_name"],
                "modality": cfg["main_model"]["modality"],
                "dataset": item["dataset"],
                "completed_folds": item.get("completed_folds", 0),
            }
            for metric in metrics:
                row[f"{metric}_mean"] = item.get(f"{metric}_mean", float("nan"))
                row[f"{metric}_std"] = item.get(f"{metric}_std", float("nan"))
            global_rows.append(row)
        write_json(
            output_dir / "main_model_reference.json",
            {
                "reused": True,
                "source": str(main_summary_path),
                "model": cfg["main_model"],
            },
        )

    write_csv(output_dir / "t3_table2_summary.csv", global_rows)
    write_json(output_dir / "t3_table2_summary.json", global_rows)
    completed = sum(int(row.get("completed_folds", 0)) for row in global_rows)
    expected = (
        len(cfg["image_models"]) + len(cfg["text_models"]) + 1
    ) * len(cfg["datasets"]) * int(cfg["folds"])
    write_json(
        output_dir / "progress.json",
        {
            "completed_folds": completed,
            "expected_folds_including_reused_main_model": expected,
            "completion_ratio": completed / expected if expected else 0.0,
        },
    )
    print(f"[TASK3-T2] 汇总完成：{completed}/{expected}折")


def flatten_summary(summary: dict[str, Any], metrics: list[str]) -> dict[str, Any]:
    row = {
        "model": summary["model"],
        "display_name": summary["display_name"],
        "modality": summary["modality"],
        "dataset": summary["dataset"],
        "completed_folds": summary["completed_folds"],
    }
    for metric in metrics:
        payload = summary.get("metrics", {}).get(metric, {})
        row[f"{metric}_mean"] = payload.get("mean", float("nan"))
        row[f"{metric}_std"] = payload.get("std", float("nan"))
    return row


def main() -> None:
    args = parse_args()
    validate_shard(args.num_shards, args.shard_index)
    cfg = read_yaml(args.config.expanduser().resolve())
    output_dir = Path(cfg["paths"]["output_dir"]).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    write_json(output_dir / "experiment_config.json", cfg)

    if args.summarize_only:
        summarize_all(cfg, output_dir)
        return

    dataset_names = parse_selection(args.datasets, list(cfg["datasets"]), "数据集")
    fold_indices = [
        int(value)
        for value in parse_selection(
            args.folds,
            [str(index) for index in range(1, int(cfg["folds"]) + 1)],
            "折号",
        )
    ]
    prepared_folds = prepare_records_and_folds(cfg, dataset_names, output_dir)
    if args.prepare_only:
        print(f"[TASK3-T2] 数据准备完成：{output_dir}")
        return

    if args.modality in {"text", "all"}:
        run_text_jobs(
            args=args,
            cfg=cfg,
            output_dir=output_dir,
            prepared_folds=prepared_folds,
            dataset_names=dataset_names,
            fold_indices=fold_indices,
        )
    if args.modality in {"image", "all"}:
        run_image_jobs(
            args=args,
            cfg=cfg,
            output_dir=output_dir,
            prepared_folds=prepared_folds,
            dataset_names=dataset_names,
            fold_indices=fold_indices,
        )
    if args.num_shards == 1:
        summarize_all(cfg, output_dir)


if __name__ == "__main__":
    main()
