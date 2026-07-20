#!/usr/bin/env python3
"""按胃镜检查类型评估 TASK2 最终模型在固定测试集上的性能。"""

from __future__ import annotations

import argparse
import csv
import inspect
import json
import os
import sys
from collections import Counter
from pathlib import Path
from typing import Any

SRC_ROOT = Path(__file__).resolve().parents[1]
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

import numpy as np
import torch
import yaml
from torch.utils.data import DataLoader

from exp_8 import build_exp8_model
from tasks import get_task_spec
from training.data import MILBagDataset, build_task_records, mil_collate_fn, split_records
from training.metrics import compute_multilabel_metrics, to_builtin_type

try:
    from tqdm import tqdm
except ImportError:  # pragma: no cover
    tqdm = None


PROJECT_ROOT = Path("/home/Lim/Project4")
DEFAULT_RUN_DIR = (
    PROJECT_ROOT
    / "outputs/train_runs/task2/exp_9_ablation/train_017_exp9_watch_cross_attn_no_image_aux"
)
DEFAULT_DATA_CSV = PROJECT_ROOT / "datasets/task_data/task2/gastro_multilabel_task_datalist.csv"
DEFAULT_DATASET_ROOT = PROJECT_ROOT / "datasets/main_data"
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "outputs/test_scope_groups"

GROUP_TITLES = {
    "regular_white_light": {
        "无痛胃镜检查报告",
        "胃镜检查报告",
        "一诊疗无痛胃镜报告",
        "职工体检胃镜(无痛)报告",
        "急诊胃镜下取异物报告",
    },
    "chromoscopic": {
        "放大染色胃镜精查报告",
        "无痛胃镜(含色素内镜)报告",
        "国际部无痛胃镜检查（含色素内镜）报告",
        "国际部胃镜检查（含色素内镜）报告",
    },
    "surgical": {
        "胃镜手术(住院)报告",
        "胃镜下切除手术报告",
        "胃镜下其他手术报告",
        "急诊胃镜报告",
    },
    "ultrasound": {
        "超声胃镜检查报告",
        "无痛超声胃镜报告",
    },
}

GROUP_DISPLAY_NAMES = {
    "regular_white_light": "常规白光胃镜",
    "chromoscopic": "染色胃镜",
    "surgical": "手术胃镜",
    "ultrasound": "超声胃镜",
    "other_or_hybrid": "其他/混合类型",
    "all_test": "完整测试集",
}

SUMMARY_METRICS = (
    "macro_f1",
    "micro_f1",
    "macro_roc_auc",
    "macro_pr_auc",
    "subset_accuracy",
    "hamming_loss",
    "kappa",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, default=DEFAULT_RUN_DIR)
    parser.add_argument("--checkpoint", type=Path, default=None)
    parser.add_argument("--data-csv", type=Path, default=DEFAULT_DATA_CSV)
    parser.add_argument("--dataset-root", type=Path, default=DEFAULT_DATASET_ROOT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--device", default="auto", help="auto、cpu、cuda或cuda:0")
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--max-samples", type=int, default=0, help="仅用于冒烟测试；0表示完整测试集")
    return parser.parse_args()


def read_yaml(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as file:
        payload = yaml.safe_load(file) or {}
    if not isinstance(payload, dict):
        raise ValueError(f"YAML 顶层必须为字典：{path}")
    return payload


def resolve_device(raw: str) -> torch.device:
    if raw == "auto":
        return torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    device = torch.device(raw)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("指定了CUDA设备，但当前PyTorch无法识别GPU")
    return device


def classify_report_title(title: str) -> str:
    normalized = str(title).strip()
    matches = [group_name for group_name, titles in GROUP_TITLES.items() if normalized in titles]
    if len(matches) > 1:
        raise ValueError(f"reportTitle 被重复归类：{normalized} -> {matches}")
    return matches[0] if matches else "other_or_hybrid"


def load_checkpoint_state(path: Path, device: torch.device) -> dict[str, torch.Tensor]:
    try:
        payload = torch.load(path, map_location=device, weights_only=False)
    except TypeError:
        payload = torch.load(path, map_location=device)
    if not isinstance(payload, dict):
        raise ValueError(f"checkpoint格式错误：{path}")
    state = payload.get("model_state", payload)
    if not isinstance(state, dict):
        raise ValueError(f"checkpoint中没有model_state：{path}")
    cleaned: dict[str, torch.Tensor] = {}
    for key, value in state.items():
        clean_key = str(key)
        while clean_key.startswith("module."):
            clean_key = clean_key[len("module.") :]
        cleaned[clean_key] = value
    return cleaned


def build_model(config: dict[str, Any], checkpoint: Path, device: torch.device) -> torch.nn.Module:
    model_name = str(config.get("model_name", "exp8_mm_watch_cross_attn"))
    params = dict(config.get("model_params", {}) or {})
    params.update({"pretrained": False, "num_labels": 3})
    model = build_exp8_model(model_name=model_name, **params)
    state = load_checkpoint_state(checkpoint, device)
    model.load_state_dict(state, strict=True)
    model.to(device)
    model.eval()
    return model


def build_test_records(
    config: dict[str, Any],
    data_csv: Path,
    dataset_root: Path,
) -> list[dict[str, Any]]:
    seed = int(config.get("seed", 2026))
    records = build_task_records(
        task_csv_path=data_csv,
        task_name="task2",
        min_instances=1,
        dataset_root=dataset_root,
    )
    split = split_records(
        records,
        seed=seed,
        ratios=(0.6, 0.2, 0.2),
        group_by_patient=True,
    )
    test_records = split["test"]
    expected = int(dict(config.get("split_stats", {}) or {}).get("test", len(test_records)))
    if len(test_records) != expected:
        raise RuntimeError(f"重建测试集数量不一致：当前{len(test_records)}，训练记录{expected}")
    return test_records


def build_loader(
    records: list[dict[str, Any]],
    config: dict[str, Any],
    num_workers: int,
) -> DataLoader:
    run_cfg = dict(config.get("run", {}) or {})
    cache_dir = run_cfg.get("resolved_image_cache_dir") or run_cfg.get("image_cache_dir")
    legacy_dirs = run_cfg.get("resolved_legacy_image_cache_dirs", [])
    dataset = MILBagDataset(
        records=records,
        task_name="task2",
        max_instances=int(run_cfg.get("eval_max_instances", 64)),
        min_instances=1,
        bag_sampling_strategy=str(run_cfg.get("eval_sampling_strategy", "uniform")),
        is_train=False,
        image_size=int(config.get("image_size", 224)),
        random_instance_dropout=0.0,
        image_cache_mode=str(run_cfg.get("image_cache_mode", "disk")),
        image_cache_dir=cache_dir,
        legacy_image_cache_dirs=legacy_dirs,
        memory_cache_size=0,
        split_name="test",
    )
    kwargs: dict[str, Any] = {
        "dataset": dataset,
        "batch_size": 1,
        "shuffle": False,
        "num_workers": max(0, int(num_workers)),
        "pin_memory": torch.cuda.is_available(),
        "persistent_workers": False,
        "collate_fn": mil_collate_fn,
    }
    if int(num_workers) > 0:
        kwargs["prefetch_factor"] = 1
    return DataLoader(**kwargs)


def move_tensor_batch(batch: dict[str, Any], device: torch.device) -> dict[str, Any]:
    return {
        key: value.to(device, non_blocking=True) if torch.is_tensor(value) else value
        for key, value in batch.items()
    }


def forward_model(model: torch.nn.Module, batch: dict[str, Any]) -> torch.Tensor:
    parameters = inspect.signature(model.forward).parameters
    kwargs: dict[str, Any] = {"images": batch["images"], "mask": batch["mask"]}
    for key in (
        "labels",
        "watch_token_ids",
        "watch_token_mask",
        "text_token_ids",
        "text_token_mask",
        "guided_text_token_ids",
        "guided_text_token_mask",
        "structured_categorical",
        "structured_numeric",
        "structured_mask",
    ):
        if key in parameters and key in batch:
            kwargs[key] = batch[key]
    output = model(**kwargs)
    if torch.is_tensor(output):
        logits = output
    elif isinstance(output, dict) and torch.is_tensor(output.get("logits")):
        logits = output["logits"]
    else:
        raise TypeError(f"无法从模型输出中获得logits：{type(output)}")
    return torch.sigmoid(logits)


def finite_or_none(value: Any) -> float | None:
    number = float(value)
    return number if np.isfinite(number) else None


def summarize_metrics(metrics: dict[str, Any]) -> dict[str, float | None]:
    return {key: finite_or_none(metrics[key]) for key in SUMMARY_METRICS}


def main() -> None:
    args = parse_args()
    os.environ.setdefault("PROJECT4_DISABLE_DISK_CACHE_WRITE", "1")
    args.output_dir.mkdir(parents=True, exist_ok=True)

    config = read_yaml(args.run_dir / "config.yaml")
    checkpoint = args.checkpoint or args.run_dir / "checkpoints/best_macro_f1.ckpt"
    device = resolve_device(args.device)
    print(f"评估设备：{device}")
    if device.type == "cpu":
        print("警告：当前未检测到CUDA，将使用CPU完成推理。")

    test_records = build_test_records(config, args.data_csv, args.dataset_root)
    if args.max_samples > 0:
        test_records = test_records[: int(args.max_samples)]
    title_counts = Counter(str(record.get("report_title", "")) for record in test_records)
    group_counts = Counter(classify_report_title(record.get("report_title", "")) for record in test_records)
    print("测试集分组：", dict(group_counts))

    loader = build_loader(test_records, config, args.num_workers)
    model = build_model(config, checkpoint, device)
    label_names = list(get_task_spec("task2").label_names)

    rows: list[dict[str, Any]] = []
    true_parts: list[np.ndarray] = []
    prob_parts: list[np.ndarray] = []
    iterator = tqdm(loader, total=len(loader), desc="四类胃镜测试", unit="检查") if tqdm else loader
    with torch.inference_mode():
        for batch_cpu in iterator:
            batch = move_tensor_batch(batch_cpu, device)
            probabilities = forward_model(model, batch)
            y_true = batch["labels"].detach().cpu().numpy().astype(np.int64)
            y_prob = probabilities.detach().cpu().numpy().astype(np.float32)
            true_parts.append(y_true)
            prob_parts.append(y_prob)
            for index, exam_dir in enumerate(batch_cpu["exam_dirs"]):
                report_title = str(batch_cpu["report_titles"][index])
                row: dict[str, Any] = {
                    "exam_dir": str(exam_dir),
                    "report_title": report_title,
                    "scope_group": classify_report_title(report_title),
                    "scope_group_cn": GROUP_DISPLAY_NAMES[classify_report_title(report_title)],
                }
                for label_index, label_name in enumerate(label_names):
                    row[f"true_{label_name}"] = int(y_true[index, label_index])
                    row[f"prob_{label_name}"] = float(y_prob[index, label_index])
                    row[f"pred_{label_name}"] = int(y_prob[index, label_index] >= 0.5)
                rows.append(row)

    y_true_all = np.concatenate(true_parts, axis=0)
    y_prob_all = np.concatenate(prob_parts, axis=0)
    row_groups = np.asarray([row["scope_group"] for row in rows], dtype=str)

    group_payload: dict[str, Any] = {}
    groups_to_report = [*GROUP_TITLES.keys(), "other_or_hybrid", "all_test"]
    for group_name in groups_to_report:
        mask = np.ones(len(rows), dtype=bool) if group_name == "all_test" else row_groups == group_name
        if not mask.any():
            continue
        metrics = compute_multilabel_metrics(
            y_true=y_true_all[mask],
            y_prob=y_prob_all[mask],
            label_names=label_names,
            threshold=0.5,
        )
        group_payload[group_name] = {
            "display_name": GROUP_DISPLAY_NAMES[group_name],
            "num_exams": int(mask.sum()),
            "metrics": to_builtin_type(metrics),
            "summary": summarize_metrics(metrics),
        }

    prediction_path = args.output_dir / "scope_group_predictions.csv"
    with prediction_path.open("w", encoding="utf-8-sig", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    summary_path = args.output_dir / "scope_group_metrics.csv"
    with summary_path.open("w", encoding="utf-8-sig", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=["scope_group", "display_name", "num_exams", *SUMMARY_METRICS])
        writer.writeheader()
        for group_name in groups_to_report:
            payload = group_payload.get(group_name)
            if payload is None:
                continue
            writer.writerow(
                {
                    "scope_group": group_name,
                    "display_name": payload["display_name"],
                    "num_exams": payload["num_exams"],
                    **payload["summary"],
                }
            )

    audit = {
        "run_dir": str(args.run_dir),
        "checkpoint": str(checkpoint),
        "device": str(device),
        "num_test_exams": len(rows),
        "group_counts": dict(group_counts),
        "report_title_counts": dict(title_counts),
        "group_title_mapping": {key: sorted(value) for key, value in GROUP_TITLES.items()},
        "groups": group_payload,
    }
    (args.output_dir / "scope_group_metrics.json").write_text(
        json.dumps(to_builtin_type(audit), ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(f"预测明细：{prediction_path}")
    print(f"指标汇总：{summary_path}")
    for group_name in groups_to_report:
        payload = group_payload.get(group_name)
        if payload is not None:
            print(payload["display_name"], payload["num_exams"], payload["summary"])


if __name__ == "__main__":
    main()
