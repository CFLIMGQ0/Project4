#!/usr/bin/env python3
"""在四类胃镜子数据集上运行 TASK3 主模型五折训练、验证和测试。"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import sys
from pathlib import Path
from typing import Any

import yaml


def _ensure_project_runtime_python() -> None:
    try:
        import torch  # noqa: F401
        return
    except ImportError:
        pass
    candidate = Path("/xmlg/Lim/conda/envs/myenv/bin/python")
    if candidate.is_file() and Path(sys.executable).resolve() != candidate.resolve():
        os.execv(str(candidate), [str(candidate), *sys.argv])


_ensure_project_runtime_python()

import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from exp_10.masking import mask_answer_terms
from scripts.task1_table_5fold import build_fold_context, make_stratified_folds, summarize_values
from tasks import get_task_spec
from train import (
    auto_series_resume_checkpoint,
    build_task_records,
    is_auto_series_run_complete,
    load_existing_auto_series_result,
    load_model_config,
    load_train_config,
    run_model_job,
    to_builtin_type,
)


LABEL_NAMES = (
    "label_esophageal_smt",
    "label_esophageal_mucosal_or_tumor",
    "label_gastritis",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=ROOT / "configs/task3/t3_main_model.yaml")
    parser.add_argument("--datasets", default="", help="逗号分隔的数据集键；默认全部")
    parser.add_argument("--folds", default="", help="逗号分隔的折号；默认1至5折")
    parser.add_argument("--prepare-only", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--summarize-only", action="store_true")
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--num-shards", type=int, default=1)
    parser.add_argument("--shard-index", type=int, default=0)
    return parser.parse_args()


def read_yaml(path: Path) -> dict[str, Any]:
    payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"配置文件顶层必须是字典：{path}")
    return payload


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    # 多个独立单卡进程会并行启动。临时文件名包含进程号，
    # 避免它们同时更新公共审计文件时互相覆盖或触发 FileNotFoundError。
    temporary = path.with_suffix(f"{path.suffix}.{os.getpid()}.tmp")
    temporary.write_text(
        json.dumps(to_builtin_type(payload), ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    temporary.replace(path)


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames: list[str] = []
    for row in rows:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(key)
    temporary = path.with_suffix(f"{path.suffix}.{os.getpid()}.tmp")
    with temporary.open("w", encoding="utf-8-sig", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    temporary.replace(path)


def records_cache_meta(source_csv: Path, dataset_root: Path) -> dict[str, Any]:
    source_stat = source_csv.stat()
    return {
        "source_csv": str(source_csv.resolve()),
        "source_csv_size": int(source_stat.st_size),
        "source_csv_mtime_ns": int(source_stat.st_mtime_ns),
        "dataset_root": str(dataset_root.resolve()),
        "task_name": "task2",
        "min_instances": 1,
    }


def load_or_build_records(cfg: dict[str, Any], output_dir: Path) -> list[dict[str, Any]]:
    path_cfg = cfg["paths"]
    source_csv = Path(path_cfg["source_csv"]).expanduser().resolve()
    dataset_root = Path(path_cfg["dataset_root"]).expanduser().resolve()
    cache_path = Path(path_cfg.get("records_cache", output_dir / "records_cache.json")).expanduser().resolve()
    expected_meta = records_cache_meta(source_csv, dataset_root)
    if cache_path.is_file():
        payload = json.loads(cache_path.read_text(encoding="utf-8"))
        if payload.get("meta") == expected_meta and isinstance(payload.get("records"), list):
            print(f"[TASK3] 读取样本缓存：{cache_path}，样本数={len(payload['records'])}")
            return payload["records"]
        if bool(path_cfg.get("allow_migrated_records_cache", False)) and isinstance(payload.get("records"), list):
            records = payload["records"]
            if not records:
                raise RuntimeError(f"迁移样本缓存为空：{cache_path}")
            print(
                f"[TASK3] 读取迁移样本缓存：{cache_path}，样本数={len(records)}；"
                "保留记录内原始图像标识以匹配离线缓存。"
            )
            return records

    print("[TASK3] 首次构建样本缓存，将扫描检查目录中的图片。")
    records = build_task_records(
        task_csv_path=source_csv,
        task_name="task2",
        min_instances=1,
        dataset_root=dataset_root,
    )
    write_json(cache_path, {"meta": expected_meta, "records": records, "num_records": len(records)})
    print(f"[TASK3] 样本缓存已写入：{cache_path}")
    return records


def apply_watch_mask(records: list[dict[str, Any]], enabled: bool) -> dict[str, int]:
    masked_records = 0
    masked_terms = 0
    empty_watch = 0
    for record in records:
        text_raw = dict(record.get("text_raw", {}))
        raw_watch = str(text_raw.get("watch", record.get("watch", "")))
        if not raw_watch.strip():
            empty_watch += 1
        if enabled:
            masked_watch, hits = mask_answer_terms(raw_watch)
            masked_records += int(bool(hits))
            masked_terms += len(hits)
        else:
            masked_watch = raw_watch
        text_raw["watch"] = masked_watch
        record["text_raw"] = text_raw
        record["watch"] = masked_watch
    return {
        "records": len(records),
        "records_with_masked_terms": masked_records,
        "masked_term_occurrences": masked_terms,
        "empty_watch_records": empty_watch,
    }


def split_dataset_records(records: list[dict[str, Any]], cfg: dict[str, Any]) -> tuple[dict[str, list[dict[str, Any]]], list[str]]:
    title_to_dataset: dict[str, str] = {}
    for dataset_name, dataset_cfg in cfg["datasets"].items():
        for title in dataset_cfg["report_titles"]:
            normalized = str(title).strip()
            if normalized in title_to_dataset:
                raise ValueError(f"报告标题被重复归类：{normalized}")
            title_to_dataset[normalized] = dataset_name

    grouped = {name: [] for name in cfg["datasets"]}
    excluded_titles: list[str] = []
    for record in records:
        title = str(record.get("report_title", "")).strip()
        dataset_name = title_to_dataset.get(title)
        if dataset_name is None:
            excluded_titles.append(title)
            continue
        grouped[dataset_name].append(record)
    return grouped, excluded_titles


def label_stats(records: list[dict[str, Any]]) -> dict[str, Any]:
    patients = {str(record.get("patient_id", "")) for record in records}
    result: dict[str, Any] = {
        "patients": len(patients),
        "exams": len(records),
        "images": sum(len(record.get("image_paths", [])) for record in records),
    }
    for index, label_name in enumerate(LABEL_NAMES):
        positive = sum(int(record["labels"][index]) for record in records)
        result[label_name] = {
            "positive": positive,
            "negative": len(records) - positive,
        }
    return result


def manifest_rows(records: list[dict[str, Any]], split_name: str, fold: int) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for record in records:
        rows.append(
            {
                "fold": fold,
                "split": split_name,
                "patient_id": record.get("patient_id", ""),
                "exam_dir": record.get("exam_dir", ""),
                "report_title": record.get("report_title", ""),
                "img_num": len(record.get("image_paths", [])),
                **{label: int(record["labels"][index]) for index, label in enumerate(LABEL_NAMES)},
            }
        )
    return rows


def prepare_dataset_folds(
    *,
    dataset_name: str,
    records: list[dict[str, Any]],
    cfg: dict[str, Any],
    output_dir: Path,
) -> list[list[dict[str, Any]]]:
    folds = make_stratified_folds(
        records,
        label_names=list(LABEL_NAMES),
        folds=int(cfg["folds"]),
        seed=int(cfg["seed"]),
        group_by_patient=True,
    )
    dataset_dir = output_dir / dataset_name
    fold_audits: list[dict[str, Any]] = []
    for fold_index in range(1, int(cfg["folds"]) + 1):
        test_index = fold_index - 1
        val_index = fold_index % int(cfg["folds"])
        split_data = {
            "train": [record for index, fold in enumerate(folds) if index not in {test_index, val_index} for record in fold],
            "val": list(folds[val_index]),
            "test": list(folds[test_index]),
        }
        patient_sets = {
            key: {str(record.get("patient_id", "")) for record in value}
            for key, value in split_data.items()
        }
        leakage = {
            "train_val": sorted(patient_sets["train"] & patient_sets["val"]),
            "train_test": sorted(patient_sets["train"] & patient_sets["test"]),
            "val_test": sorted(patient_sets["val"] & patient_sets["test"]),
        }
        if any(leakage.values()):
            raise RuntimeError(f"{dataset_name} fold_{fold_index} 检测到患者泄漏：{leakage}")
        audit = {
            "fold": fold_index,
            "train": label_stats(split_data["train"]),
            "val": label_stats(split_data["val"]),
            "test": label_stats(split_data["test"]),
            "patient_leakage": leakage,
        }
        fold_audits.append(audit)
        rows = [
            *manifest_rows(split_data["train"], "train", fold_index),
            *manifest_rows(split_data["val"], "val", fold_index),
            *manifest_rows(split_data["test"], "test", fold_index),
        ]
        write_csv(dataset_dir / f"fold_{fold_index}" / "split_manifest.csv", rows)
        write_json(dataset_dir / f"fold_{fold_index}" / "split_stats.json", audit)
    write_json(dataset_dir / "fold_audit.json", fold_audits)
    return folds


def extract_fold_result(result: dict[str, Any], dataset_name: str, fold_index: int, cfg: dict[str, Any]) -> dict[str, Any]:
    selection_alias = str(cfg.get("selection_alias", "best_macro_f1"))
    payload = result.get("test_results", {}).get(selection_alias, {})
    metrics = payload.get("metrics", {}) if isinstance(payload, dict) else {}
    row: dict[str, Any] = {
        "dataset": dataset_name,
        "fold": fold_index,
        "best_epoch": payload.get("best_epoch", ""),
        "checkpoint_path": payload.get("checkpoint_path", ""),
        "run_dir": result.get("train_dir", ""),
    }
    for metric_name in cfg["metrics"]:
        row[metric_name] = metrics.get(metric_name, float("nan"))
    for label_name in LABEL_NAMES:
        for metric_name in ("recall", "precision", "f1", "roc_auc", "pr_auc"):
            row[f"{metric_name}_{label_name}"] = metrics.get(f"{metric_name}_{label_name}", float("nan"))
    return row


def summarize_dataset(dataset_name: str, dataset_dir: Path, cfg: dict[str, Any]) -> None:
    rows: list[dict[str, Any]] = []
    model_name = str(cfg["model"]["model_name"])
    for fold_index in range(1, int(cfg["folds"]) + 1):
        run_dir = dataset_dir / f"fold_{fold_index}"
        if not is_auto_series_run_complete(run_dir):
            continue
        result = load_existing_auto_series_result(run_dir, model_name)
        rows.append(extract_fold_result(result, dataset_name, fold_index, cfg))
    write_csv(dataset_dir / "fold_results.csv", rows)
    summary: dict[str, Any] = {"dataset": dataset_name, "completed_folds": len(rows), "metrics": {}}
    for metric_name in cfg["metrics"]:
        values = [float(row[metric_name]) for row in rows if not math.isnan(float(row[metric_name]))]
        summary["metrics"][metric_name] = summarize_values(values)
    write_json(dataset_dir / "fivefold_summary.json", summary)


def summarize_all(output_dir: Path, cfg: dict[str, Any]) -> None:
    summaries: list[dict[str, Any]] = []
    for dataset_name in cfg["datasets"]:
        dataset_dir = output_dir / dataset_name
        summarize_dataset(dataset_name, dataset_dir, cfg)
        summary_path = dataset_dir / "fivefold_summary.json"
        if summary_path.is_file():
            payload = json.loads(summary_path.read_text(encoding="utf-8"))
            row: dict[str, Any] = {"dataset": dataset_name, "completed_folds": payload.get("completed_folds", 0)}
            for metric_name in cfg["metrics"]:
                metric = payload.get("metrics", {}).get(metric_name, {})
                row[f"{metric_name}_mean"] = metric.get("mean", float("nan"))
                row[f"{metric_name}_std"] = metric.get("std", float("nan"))
            summaries.append(row)
    write_csv(output_dir / "t3_main_model_summary.csv", summaries)
    write_json(output_dir / "t3_main_model_summary.json", summaries)


def parse_selection(raw: str, allowed: list[str], value_name: str) -> list[str]:
    if not raw.strip():
        return allowed
    selected = [item.strip() for item in raw.split(",") if item.strip()]
    unknown = [item for item in selected if item not in allowed]
    if unknown:
        raise ValueError(f"未知{value_name}：{unknown}；允许值={allowed}")
    return selected


def main() -> None:
    args = parse_args()
    cfg = read_yaml(args.config.expanduser().resolve())
    if args.num_shards < 1 or not 0 <= args.shard_index < args.num_shards:
        raise ValueError("分片参数必须满足num_shards>=1且0<=shard_index<num_shards")
    output_dir = Path(cfg["paths"]["output_dir"]).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    write_json(output_dir / "experiment_config.json", cfg)

    if args.summarize_only:
        summarize_all(output_dir, cfg)
        print(f"[TASK3] 汇总已更新：{output_dir}")
        return

    records = load_or_build_records(cfg, output_dir)
    mask_audit = apply_watch_mask(records, bool(cfg.get("text", {}).get("mask_answer_terms", True)))
    grouped, excluded_titles = split_dataset_records(records, cfg)
    dataset_names = parse_selection(args.datasets, list(cfg["datasets"]), "数据集")
    selected_folds = [int(value) for value in parse_selection(args.folds, [str(i) for i in range(1, int(cfg["folds"]) + 1)], "折号")]

    global_audit = {
        "source_records": len(records),
        "watch_mask": mask_audit,
        "excluded_records": len(excluded_titles),
        "excluded_report_titles": sorted(set(excluded_titles)),
        "datasets": {name: label_stats(group_records) for name, group_records in grouped.items()},
    }
    write_json(output_dir / "data_audit.json", global_audit)

    prepared_folds: dict[str, list[list[dict[str, Any]]]] = {}
    for dataset_name in dataset_names:
        prepared_folds[dataset_name] = prepare_dataset_folds(
            dataset_name=dataset_name,
            records=grouped[dataset_name],
            cfg=cfg,
            output_dir=output_dir,
        )
        print(f"[TASK3] {dataset_name}: {label_stats(grouped[dataset_name])}")

    if args.prepare_only or args.dry_run:
        print(f"[TASK3] 数据准备与五折审计完成：{output_dir}")
        return

    if not torch.cuda.is_available() or torch.cuda.device_count() < 1:
        raise RuntimeError("当前进程无法识别GPU，TASK3训练未启动")

    torch.set_float32_matmul_precision("medium")
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.benchmark = True
    train_cfg = load_train_config(ROOT / "configs/task2/train.yaml")
    model_cfg = load_model_config(ROOT / "configs/task2/model.yaml")
    training_cfg = cfg["training"]
    model_name = str(cfg["model"]["model_name"])
    base_context = {
        "output_root": output_dir,
        "task_selection_dir": str(Path(cfg["paths"]["source_csv"]).expanduser().resolve().parent.parent),
    }

    jobs = [
        (dataset_name, fold_index)
        for dataset_name in dataset_names
        for fold_index in selected_folds
    ]
    jobs = [
        job
        for index, job in enumerate(jobs)
        if index % args.num_shards == args.shard_index
    ]
    print(f"[TASK3] 当前分片训练任务数={len(jobs)}，shard={args.shard_index + 1}/{args.num_shards}")

    for dataset_name, fold_index in jobs:
        dataset_dir = output_dir / dataset_name
        folds = prepared_folds[dataset_name]
        run_dir = dataset_dir / f"fold_{fold_index}"
        if args.force and run_dir.exists():
            raise RuntimeError("为保护已有结果，TASK3不自动删除fold目录；请先人工确认后再处理")
        if is_auto_series_run_complete(run_dir):
            print(f"[TASK3] 跳过已完成：{dataset_name}/fold_{fold_index}")
            continue

        test_index = fold_index - 1
        val_index = fold_index % int(cfg["folds"])
        split_data = {
            "train": [
                record
                for index, fold in enumerate(folds)
                if index not in {test_index, val_index}
                for record in fold
            ],
            "val": list(folds[val_index]),
            "test": list(folds[test_index]),
        }
        fold_seed = int(cfg["seed"]) + fold_index
        effective_train_cfg = {
            **train_cfg,
            "seed": fold_seed,
            "class_balance": dict(cfg["class_balance"]),
        }
        fold_context = build_fold_context(
            base_context=base_context,
            task_csv=Path(cfg["paths"]["source_csv"]).expanduser().resolve(),
            split_data=split_data,
            train_cfg=effective_train_cfg,
            task_name="task2",
        )
        balance_report = fold_context["tasks"]["task2"].get("balance_report")
        if balance_report:
            write_json(run_dir / "class_balance_report.json", balance_report)
        resume_path = auto_series_resume_checkpoint(run_dir)
        print(
            f"[TASK3] 开始训练：{dataset_name}/fold_{fold_index}，"
            f"GPU={torch.cuda.get_device_name(0)}"
        )
        result = run_model_job(
            model_name=model_name,
            run_dir=run_dir,
            train_cfg=effective_train_cfg,
            model_cfg=model_cfg,
            training_context=fold_context,
            seed=fold_seed,
            max_epochs=int(training_cfg["max_epochs"]),
            patience=int(training_cfg["patience"]),
            image_size=int(training_cfg["image_size"]),
            num_workers=int(training_cfg["num_workers"]),
            pretrained=bool(training_cfg["pretrained"]),
            use_multi_gpu=False,
            active_gpu_count=1,
            run_test=bool(training_cfg["run_test"]),
            run_overrides=dict(training_cfg["run_overrides"]),
            model_param_override=dict(cfg["model"]["params"]),
            entry_metadata={
                "task3": True,
                "dataset": dataset_name,
                "fold": fold_index,
                "text_encoder": "TextCNN",
                **dict(cfg.get("entry_metadata", {})),
            },
            resume_path=resume_path,
        )
        row = extract_fold_result(result, dataset_name, fold_index, cfg)
        print(f"[TASK3] 完成：{dataset_name}/fold_{fold_index}，macro_f1={row.get('macro_f1')}")
        summarize_dataset(dataset_name, dataset_dir, cfg)

    if args.num_shards == 1:
        summarize_all(output_dir, cfg)
    print(f"[TASK3] 当前任务全部完成：{output_dir}")


if __name__ == "__main__":
    main()
