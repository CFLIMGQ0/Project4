#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import inspect
import json
import math
import re
import os
import sys
from pathlib import Path
from typing import Any


def _ensure_project_runtime_python() -> None:
    import importlib.util

    if importlib.util.find_spec("torch") is not None:
        return
    if os.environ.get("PROJECT4_RUNTIME_REEXEC") == "1":
        return

    current_python = Path(sys.executable).resolve()
    candidate_strings: list[str] = []

    override = os.environ.get("PROJECT4_TRAIN_PYTHON", "").strip()
    if override:
        candidate_strings.append(override)

    if current_python.parent.name == "bin":
        conda_root = current_python.parent.parent
        candidate_strings.append(str(conda_root / "envs" / "myenv" / "bin" / "python"))

    candidate_strings.append("/home/Lim/anaconda3/envs/myenv/bin/python")

    seen: set[str] = set()
    for candidate_str in candidate_strings:
        if not candidate_str or candidate_str in seen:
            continue
        seen.add(candidate_str)
        candidate = Path(candidate_str).expanduser()
        if not candidate.is_file():
            continue
        if candidate.resolve() == current_python:
            continue

        os.environ["PROJECT4_RUNTIME_REEXEC"] = "1"
        print(
            f"[task1_distinct_significance.py] 当前解释器 {current_python} 缺少 torch，自动切换到 {candidate}",
            file=sys.stderr,
        )
        os.execv(str(candidate), [str(candidate), *sys.argv])


_ensure_project_runtime_python()

import numpy as np
import torch
import yaml

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from training.metrics import compute_multilabel_metrics, to_builtin_type
from train import (
    build_label_cooccurrence_prior,
    build_loaders,
    build_model_bundle,
    load_path_config,
    load_train_config,
    prepare_training_context,
    resolve_default_config_path,
    resolve_run_cfg,
    seed_everything,
)

try:
    from tqdm import tqdm
except Exception:  # pragma: no cover
    tqdm = None


LABEL_NAMES = [
    "label_esophageal_smt",
    "label_esophageal_mucosal_or_tumor",
    "label_gastritis",
]
LABEL_DISPLAY_NAMES = {
    "label_esophageal_smt": "Esophageal submucosal tumor",
    "label_esophageal_mucosal_or_tumor": "Esophageal mucosal lesion",
    "label_gastritis": "Gastritis",
}
LOWER_BETTER_METRICS = {"hamming_loss"}
METRIC_DISPLAY_NAMES = {
    "macro_recall": "Macro Recall",
    "macro_precision": "Macro Precision",
    "macro_f1": "Macro F1",
    "micro_f1": "Micro F1",
    "macro_roc_auc": "Macro ROC-AUC",
    "macro_pr_auc": "Macro PR-AUC",
    "subset_accuracy": "Subset accuracy",
    "hamming_loss": "Hamming loss",
    "kappa": "Kappa",
    "recall": "Recall",
    "precision": "Precision",
    "f1": "F1",
    "roc_auc": "ROC-AUC",
    "pr_auc": "PR-AUC",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="生成 TASK1 表2/表3/表4 的 CI 与显著性实验结果")
    parser.add_argument(
        "--config",
        type=str,
        default=str(ROOT / "configs" / "task1" / "auto_distinct.yaml"),
        help="TASK1 distinct 显著性实验配置文件",
    )
    parser.add_argument("--bootstrap-samples", type=int, default=None, help="覆盖 bootstrap 重采样次数")
    parser.add_argument("--force", action="store_true", help="强制重新推理并覆盖预测缓存")
    parser.add_argument("--dry-run", action="store_true", help="只检查配置和模型路径，不执行推理或 bootstrap")
    parser.add_argument(
        "--tables",
        type=str,
        default="table2,table3,table4",
        help="指定要运行的表，使用逗号分隔，可选 table2/table3/table4",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="auto",
        help="推理设备，如 auto、cuda、cuda:0 或 cpu；配合 CUDA_VISIBLE_DEVICES 可绑定具体显卡",
    )
    parser.add_argument(
        "--eval-batch-size",
        type=int,
        default=0,
        help="覆盖统计推理阶段的 eval_batch_size；仅影响推理吞吐，不改变指标计算",
    )
    return parser.parse_args()


def read_yaml(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(f"未找到配置文件：{path}")
    payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"配置文件格式错误：{path}")
    return payload


def safe_name(value: str) -> str:
    value = re.sub(r"[^A-Za-z0-9._-]+", "_", value.strip())
    return value.strip("_") or "unnamed"


def parse_selected_tables(raw_value: str) -> tuple[str, ...]:
    allowed = {"table2", "table3", "table4"}
    selected = tuple(dict.fromkeys(item.strip().lower() for item in raw_value.split(",") if item.strip()))
    if not selected:
        raise ValueError("--tables 至少需要指定一个表，例如 table2")
    unknown = [item for item in selected if item not in allowed]
    if unknown:
        raise ValueError(f"--tables 存在未知表名：{unknown}，仅支持 table2/table3/table4")
    return selected


def resolve_device(raw_device: str) -> torch.device:
    value = str(raw_device or "auto").strip().lower()
    if value == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(value)


def format_float(value: Any, digits: int = 4) -> str:
    try:
        numeric = float(value)
    except Exception:
        return "nan"
    if math.isnan(numeric):
        return "nan"
    if abs(numeric) < 0.0005 and numeric != 0:
        return "<0.001"
    return f"{numeric:.{digits}f}"


def format_ci(value: float, ci_low: float, ci_high: float) -> str:
    return f"{format_float(value)} [{format_float(ci_low)}, {format_float(ci_high)}]"


def format_p_value(value: float) -> str:
    if math.isnan(value):
        return "NA"
    if value < 0.001:
        return "<0.001"
    return f"{value:.4f}"


def load_test_result_row(run_dir: Path, alias: str) -> dict[str, Any]:
    csv_path = run_dir / "test_result.csv"
    if not csv_path.is_file():
        csv_path = run_dir / "test_report.csv"
    if not csv_path.is_file():
        return {}
    with csv_path.open("r", encoding="utf-8-sig", newline="") as file:
        rows = list(csv.DictReader(file))
    if not rows:
        return {}
    for row in rows:
        if str(row.get("checkpoint_alias", "")).strip() == alias:
            return row
    return rows[0]


def resolve_checkpoint_path(run_dir: Path, alias: str) -> Path:
    row = load_test_result_row(run_dir, alias)
    raw_path = str(row.get("checkpoint_path", "")).strip()
    if raw_path:
        path = Path(raw_path).expanduser()
        if path.is_file():
            return path.resolve()
    fallback = run_dir / "checkpoints" / f"{alias}.ckpt"
    if fallback.is_file():
        return fallback.resolve()
    raise FileNotFoundError(f"{run_dir} 缺少 {alias} checkpoint")


def normalize_task_name(raw_task_name: Any) -> str:
    value = str(raw_task_name or "").strip()
    return "task1" if value in {"", "gastro_multilabel", "task1_gastro3"} else value


def load_run_config(run_dir: Path, base_train_cfg: dict[str, Any]) -> tuple[str, dict[str, Any], dict[str, Any], int, int]:
    config_path = run_dir / "config.yaml"
    if not config_path.is_file():
        raise FileNotFoundError(f"未找到模型配置：{config_path}")
    payload = read_yaml(config_path)
    model_name = str(payload.get("model_name", "")).strip()
    if not model_name:
        raise ValueError(f"{config_path} 缺少 model_name")

    run_cfg = dict(base_train_cfg["default_run"])
    saved_run_cfg = payload.get("run", {})
    if isinstance(saved_run_cfg, dict):
        run_cfg.update(saved_run_cfg)
    run_cfg["image_cache_warmup"] = False
    run_cfg.setdefault("eval_sampling_strategy", base_train_cfg.get("eval_sampling_strategy", "uniform"))
    run_cfg.setdefault("train_sampling_strategy", base_train_cfg.get("train_sampling_strategy", "random"))

    model_params = payload.get("model_params", {})
    if not isinstance(model_params, dict):
        model_params = {}
    seed = int(payload.get("seed", base_train_cfg.get("seed", 42)))
    image_size = int(payload.get("image_size", base_train_cfg.get("image_size", 224)))
    return model_name, run_cfg, dict(model_params), seed, image_size


def build_test_loader(
    *,
    split_data: dict[str, list[dict[str, Any]]],
    task_name: str,
    image_size: int,
    num_workers: int,
    run_cfg: dict[str, Any],
    seed: int,
    min_instances: int,
) -> Any:
    resolved_cache_dir = run_cfg.get("resolved_image_cache_dir") or run_cfg.get("image_cache_dir")
    legacy_cache_dirs = run_cfg.get("resolved_legacy_image_cache_dirs", [])
    if not isinstance(legacy_cache_dirs, list):
        legacy_cache_dirs = []
    _, _, test_loader = build_loaders(
        split_data=split_data,
        task_name=task_name,
        image_size=image_size,
        num_workers=num_workers,
        train_batch_size=int(run_cfg.get("batch_size", 12)),
        eval_batch_size=int(run_cfg.get("eval_batch_size", 12)),
        train_max_instances=int(run_cfg.get("train_max_instances", 16)),
        eval_max_instances=int(run_cfg.get("eval_max_instances", 16)),
        min_instances=min_instances,
        train_sampling=str(run_cfg.get("train_sampling_strategy", "random")),
        eval_sampling=str(run_cfg.get("eval_sampling_strategy", "uniform")),
        random_instance_dropout=float(run_cfg.get("random_instance_dropout", 0.0)),
        train_max_batch_instances=int(run_cfg.get("train_max_batch_instances", 192)),
        eval_max_batch_instances=int(run_cfg.get("eval_max_batch_instances", 192)),
        seed=seed,
        pin_memory=bool(run_cfg.get("pin_memory", True)),
        persistent_workers=bool(run_cfg.get("persistent_workers", True)),
        loader_prefetch_factor=int(run_cfg.get("loader_prefetch_factor", 2)),
        image_cache_mode=str(run_cfg.get("image_cache_mode", "disk")),
        image_cache_dir=resolved_cache_dir,
        image_cache_manifest=str(run_cfg.get("image_cache_manifest", "")).strip() or None,
        legacy_image_cache_dirs=legacy_cache_dirs,
        image_cache_warmup=False,
        memory_cache_size=int(run_cfg.get("memory_cache_size", 0)),
    )
    return test_loader


def load_model_for_inference(
    *,
    run_dir: Path,
    checkpoint_path: Path,
    model_name: str,
    run_cfg: dict[str, Any],
    model_params: dict[str, Any],
    train_cfg: dict[str, Any],
    training_context: dict[str, Any],
    device: torch.device,
) -> torch.nn.Module:
    task_payload = training_context["tasks"]["task1"]
    split_data = task_payload["split"]
    pos_weight = task_payload["pos_weight"]
    if (
        model_name == "gastro_label_graph_mil"
        and str(model_params.get("label_graph_type", "")).strip().lower() == "static_gcn"
        and "label_graph_prior" not in model_params
    ):
        model_params = dict(model_params)
        model_params["label_graph_prior"] = build_label_cooccurrence_prior(split_data.get("train", []), LABEL_NAMES)

    model_cfg = {"models": {model_name: model_params}}
    model, _, _, _, _ = build_model_bundle(
        model_name=model_name,
        task_name="task1",
        run_cfg=run_cfg,
        model_param_cfg=model_cfg["models"][model_name],
        pretrained=False,
        max_epochs=int(train_cfg.get("max_epochs", 30)),
        patience=int(train_cfg.get("patience", 30)),
        pos_weight=pos_weight,
        use_multi_gpu=False,
        run_test=False,
    )
    try:
        checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    except TypeError:
        checkpoint = torch.load(checkpoint_path, map_location=device)
    state = checkpoint.get("model_state", checkpoint)
    model.load_state_dict(state, strict=True)
    model.to(device)
    model.eval()
    return model


def move_batch_to_device(batch: dict[str, Any], device: torch.device) -> dict[str, Any]:
    moved: dict[str, Any] = {}
    for key, value in batch.items():
        moved[key] = value.to(device, non_blocking=True) if torch.is_tensor(value) else value
    return moved


def forward_model(model: torch.nn.Module, batch: dict[str, Any]) -> dict[str, Any]:
    params = inspect.signature(model.forward).parameters
    kwargs: dict[str, Any] = {
        "images": batch["images"],
        "mask": batch["mask"],
    }
    optional_keys = (
        "labels",
        "instance_types",
        "pseudo_region_labels",
        "pseudo_relevance",
        "structured_categorical",
        "structured_numeric",
        "structured_mask",
    )
    for key in optional_keys:
        if key in params and key in batch:
            kwargs[key] = batch[key]
    if "current_epoch" in params:
        kwargs["current_epoch"] = 0.0
    output = model(**kwargs)
    if isinstance(output, dict):
        return output
    if torch.is_tensor(output):
        return {"logits": output}
    raise TypeError(f"模型 forward 返回类型不支持：{type(output)}")


def extract_probabilities(output: dict[str, Any]) -> torch.Tensor:
    for key in ("probabilities", "probs", "edl_probs"):
        value = output.get(key)
        if torch.is_tensor(value):
            return value
    return torch.sigmoid(output["logits"])


def extract_threshold(output: dict[str, Any], model: torch.nn.Module, default_threshold: float) -> np.ndarray:
    value: Any = output.get("label_thresholds") if isinstance(output, dict) else None
    if value is None and hasattr(model, "get_label_thresholds"):
        getter = getattr(model, "get_label_thresholds")
        if callable(getter):
            value = getter()
    if value is None:
        return np.full((len(LABEL_NAMES),), float(default_threshold), dtype=np.float32)
    if torch.is_tensor(value):
        array = value.detach().cpu().numpy().astype(np.float32)
    else:
        array = np.asarray(value, dtype=np.float32)
    if array.ndim > 1:
        array = array.reshape(-1, array.shape[-1]).mean(axis=0)
    array = array.reshape(-1)
    if array.shape[0] != len(LABEL_NAMES):
        return np.full((len(LABEL_NAMES),), float(default_threshold), dtype=np.float32)
    return array


def collect_predictions(
    *,
    display_name: str,
    run_dir: Path,
    alias: str,
    train_cfg: dict[str, Any],
    training_context: dict[str, Any],
    output_dir: Path,
    default_threshold: float,
    force: bool,
    num_workers: int,
    device: torch.device,
    cache_namespace: str,
    eval_batch_size_override: int,
) -> dict[str, Any]:
    prediction_dir = output_dir / "predictions" / safe_name(cache_namespace)
    prediction_dir.mkdir(parents=True, exist_ok=True)
    cache_path = prediction_dir / f"{safe_name(display_name)}.npz"
    metadata_path = prediction_dir / f"{safe_name(display_name)}.json"
    if cache_path.is_file() and metadata_path.is_file() and not force:
        data = np.load(cache_path, allow_pickle=False)
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        return {
            "display_name": display_name,
            "run_dir": str(run_dir),
            "checkpoint_path": metadata.get("checkpoint_path", ""),
            "best_epoch": metadata.get("best_epoch", ""),
            "y_true": data["y_true"],
            "y_prob": data["y_prob"],
            "threshold": data["threshold"],
            "exam_ids": data["exam_ids"].astype(str).tolist(),
            "metadata": metadata,
        }

    model_name, run_cfg, model_params, seed, image_size = load_run_config(run_dir, train_cfg)
    if eval_batch_size_override > 0:
        run_cfg["eval_batch_size"] = int(eval_batch_size_override)
    checkpoint_path = resolve_checkpoint_path(run_dir, alias)
    row = load_test_result_row(run_dir, alias)
    best_epoch = row.get("best_epoch", "")
    split_data = training_context["tasks"]["task1"]["split"]
    test_loader = build_test_loader(
        split_data=split_data,
        task_name="task1",
        image_size=image_size,
        num_workers=num_workers,
        run_cfg=run_cfg,
        seed=seed,
        min_instances=int(train_cfg.get("min_instances", 1)),
    )
    model = load_model_for_inference(
        run_dir=run_dir,
        checkpoint_path=checkpoint_path,
        model_name=model_name,
        run_cfg=run_cfg,
        model_params=model_params,
        train_cfg=train_cfg,
        training_context=training_context,
        device=device,
    )

    y_true_parts: list[np.ndarray] = []
    y_prob_parts: list[np.ndarray] = []
    exam_ids: list[str] = []
    threshold = np.full((len(LABEL_NAMES),), float(default_threshold), dtype=np.float32)
    iterator = tqdm(test_loader, desc=f"推理 {display_name}", dynamic_ncols=True, leave=False) if tqdm else test_loader
    with torch.inference_mode():
        for batch_cpu in iterator:
            batch = move_batch_to_device(batch_cpu, device)
            output = forward_model(model, batch)
            probs = extract_probabilities(output)
            threshold = extract_threshold(output, model, default_threshold)
            y_prob_parts.append(probs.detach().cpu().numpy())
            y_true_parts.append(batch["labels"].detach().cpu().numpy())
            for item in batch_cpu.get("exam_dirs", []):
                exam_ids.append(str(item))
    y_true = np.concatenate(y_true_parts, axis=0).astype(np.int64)
    y_prob = np.concatenate(y_prob_parts, axis=0).astype(np.float32)
    np.savez_compressed(
        cache_path,
        y_true=y_true,
        y_prob=y_prob,
        threshold=threshold.astype(np.float32),
        exam_ids=np.asarray(exam_ids, dtype=str),
    )
    metadata = {
        "display_name": display_name,
        "run_dir": str(run_dir),
        "model_name": model_name,
        "checkpoint_path": str(checkpoint_path),
        "best_epoch": best_epoch,
        "num_samples": int(y_true.shape[0]),
        "num_labels": int(y_true.shape[1]),
        "threshold": threshold.tolist(),
    }
    metadata_path.write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")
    return {
        "display_name": display_name,
        "run_dir": str(run_dir),
        "checkpoint_path": str(checkpoint_path),
        "best_epoch": best_epoch,
        "y_true": y_true,
        "y_prob": y_prob,
        "threshold": threshold,
        "exam_ids": exam_ids,
        "metadata": metadata,
    }


def compute_metrics_from_arrays(y_true: np.ndarray, y_prob: np.ndarray, threshold: np.ndarray) -> dict[str, Any]:
    return compute_multilabel_metrics(
        y_true=y_true.astype(np.int64),
        y_prob=y_prob.astype(np.float64),
        label_names=LABEL_NAMES,
        threshold=threshold,
    )


def bootstrap_metric_values(
    *,
    y_true: np.ndarray,
    y_prob: np.ndarray,
    threshold: np.ndarray,
    metric_keys: list[str],
    bootstrap_indices: np.ndarray,
) -> dict[str, np.ndarray]:
    values = {key: np.full((bootstrap_indices.shape[0],), np.nan, dtype=np.float64) for key in metric_keys}
    for sample_index, indices in enumerate(bootstrap_indices):
        metrics = compute_metrics_from_arrays(y_true[indices], y_prob[indices], threshold)
        for key in metric_keys:
            values[key][sample_index] = float(metrics.get(key, np.nan))
    return values


def bootstrap_label_metric_values(
    *,
    y_true: np.ndarray,
    y_prob: np.ndarray,
    threshold: np.ndarray,
    metric_names: list[str],
    bootstrap_indices: np.ndarray,
) -> dict[str, dict[str, np.ndarray]]:
    values = {
        label: {metric: np.full((bootstrap_indices.shape[0],), np.nan, dtype=np.float64) for metric in metric_names}
        for label in LABEL_NAMES
    }
    for sample_index, indices in enumerate(bootstrap_indices):
        metrics = compute_metrics_from_arrays(y_true[indices], y_prob[indices], threshold)
        for label in LABEL_NAMES:
            for metric_name in metric_names:
                values[label][metric_name][sample_index] = float(metrics.get(f"{metric_name}_{label}", np.nan))
    return values


def percentile_ci(values: np.ndarray) -> tuple[float, float]:
    clean = values[np.isfinite(values)]
    if clean.size == 0:
        return float("nan"), float("nan")
    return float(np.percentile(clean, 2.5)), float(np.percentile(clean, 97.5))


def paired_bootstrap_p_value(ref_values: np.ndarray, cmp_values: np.ndarray, metric_key: str) -> float:
    diff = cmp_values - ref_values if metric_key in LOWER_BETTER_METRICS else ref_values - cmp_values
    diff = diff[np.isfinite(diff)]
    if diff.size == 0:
        return float("nan")
    p_lower = float(np.mean(diff <= 0.0))
    p_upper = float(np.mean(diff >= 0.0))
    return min(1.0, 2.0 * min(p_lower, p_upper))


def summarize_model_row(
    *,
    entry: dict[str, Any],
    predictions: dict[str, Any],
    metric_keys: list[str],
    bootstrap_values: dict[str, np.ndarray],
) -> dict[str, Any]:
    metrics = compute_metrics_from_arrays(predictions["y_true"], predictions["y_prob"], predictions["threshold"])
    row: dict[str, Any] = {
        "group": entry.get("group", entry.get("category", "")),
        "name": entry["display_name"],
        "run_dir": predictions["run_dir"],
        "checkpoint_path": predictions["checkpoint_path"],
        "best_epoch": predictions["best_epoch"],
    }
    for key in metric_keys:
        value = float(metrics.get(key, np.nan))
        ci_low, ci_high = percentile_ci(bootstrap_values[key])
        row[key] = value
        row[f"{key}_ci_low"] = ci_low
        row[f"{key}_ci_high"] = ci_high
        row[f"{key}_ci_text"] = format_ci(value, ci_low, ci_high)
    return row


def build_comparison_rows(
    *,
    entries: list[dict[str, Any]],
    predictions_by_name: dict[str, dict[str, Any]],
    metric_keys: list[str],
    bootstrap_indices: np.ndarray,
    reference_name: str,
) -> tuple[list[dict[str, Any]], dict[str, dict[str, np.ndarray]]]:
    bootstrap_by_name: dict[str, dict[str, np.ndarray]] = {}
    for entry in entries:
        name = entry["display_name"]
        pred = predictions_by_name[name]
        bootstrap_by_name[name] = bootstrap_metric_values(
            y_true=pred["y_true"],
            y_prob=pred["y_prob"],
            threshold=pred["threshold"],
            metric_keys=metric_keys,
            bootstrap_indices=bootstrap_indices,
        )

    reference_values = bootstrap_by_name[reference_name]
    rows: list[dict[str, Any]] = []
    for entry in entries:
        name = entry["display_name"]
        row = summarize_model_row(
            entry=entry,
            predictions=predictions_by_name[name],
            metric_keys=metric_keys,
            bootstrap_values=bootstrap_by_name[name],
        )
        if name == reference_name:
            row["mean_p_value"] = ""
            row["metric_p_values"] = {}
        else:
            p_values = {
                key: paired_bootstrap_p_value(reference_values[key], bootstrap_by_name[name][key], key)
                for key in metric_keys
            }
            finite_values = [value for value in p_values.values() if not math.isnan(value)]
            row["mean_p_value"] = float(np.mean(finite_values)) if finite_values else float("nan")
            row["metric_p_values"] = p_values
        rows.append(row)
    return rows, bootstrap_by_name


def build_label_rows(
    *,
    prediction: dict[str, Any],
    metric_names: list[str],
    bootstrap_indices: np.ndarray,
) -> tuple[list[dict[str, Any]], dict[str, dict[str, np.ndarray]]]:
    bootstrap_values = bootstrap_label_metric_values(
        y_true=prediction["y_true"],
        y_prob=prediction["y_prob"],
        threshold=prediction["threshold"],
        metric_names=metric_names,
        bootstrap_indices=bootstrap_indices,
    )
    metrics = compute_metrics_from_arrays(prediction["y_true"], prediction["y_prob"], prediction["threshold"])
    rows: list[dict[str, Any]] = []
    for label in LABEL_NAMES:
        row: dict[str, Any] = {"label": LABEL_DISPLAY_NAMES[label]}
        for metric_name in metric_names:
            key = f"{metric_name}_{label}"
            value = float(metrics.get(key, np.nan))
            ci_low, ci_high = percentile_ci(bootstrap_values[label][metric_name])
            row[metric_name] = value
            row[f"{metric_name}_ci_low"] = ci_low
            row[f"{metric_name}_ci_high"] = ci_high
            row[f"{metric_name}_ci_text"] = format_ci(value, ci_low, ci_high)
        rows.append(row)
    return rows, bootstrap_values


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    fieldnames: list[str] = []
    for row in rows:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(key)
    with path.open("w", encoding="utf-8-sig", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: to_builtin_type(row.get(key, "")) for key in fieldnames})


def markdown_table(headers: list[str], body: list[list[str]]) -> str:
    lines = ["| " + " | ".join(headers) + " |", "| " + " | ".join(["---"] * len(headers)) + " |"]
    for row in body:
        lines.append("| " + " | ".join(row) + " |")
    return "\n".join(lines) + "\n"


def latex_metric_header(metric_key: str) -> str:
    display = METRIC_DISPLAY_NAMES.get(metric_key, metric_key)
    parts = display.split(" ")
    if len(parts) > 1:
        return r"\shortstack{" + r"\\".join(parts) + "}"
    return display


def write_comparison_tables(prefix: str, rows: list[dict[str, Any]], metric_keys: list[str], output_dir: Path) -> None:
    csv_rows: list[dict[str, Any]] = []
    for row in rows:
        flat = {
            "group_or_category": row.get("group", ""),
            "name": row["name"],
        }
        for key in metric_keys:
            flat[key] = row.get(key)
            flat[f"{key}_95ci"] = row.get(f"{key}_ci_text")
            flat[f"{key}_ci_low"] = row.get(f"{key}_ci_low")
            flat[f"{key}_ci_high"] = row.get(f"{key}_ci_high")
            p_values = row.get("metric_p_values", {})
            if isinstance(p_values, dict) and key in p_values:
                flat[f"{key}_p_value"] = p_values[key]
        flat["mean_p_value"] = row.get("mean_p_value", "")
        flat["run_dir"] = row.get("run_dir", "")
        flat["checkpoint_path"] = row.get("checkpoint_path", "")
        flat["best_epoch"] = row.get("best_epoch", "")
        csv_rows.append(flat)
    write_csv(output_dir / f"{prefix}.csv", csv_rows)

    headers = ["Group/Category", "Model/Module"] + [METRIC_DISPLAY_NAMES.get(key, key) for key in metric_keys] + ["Mean p-value"]
    body = []
    for row in rows:
        body.append(
            [str(row.get("group", "")), str(row["name"])]
            + [str(row.get(f"{key}_ci_text", "NA")) for key in metric_keys]
            + [format_p_value(row["mean_p_value"]) if isinstance(row.get("mean_p_value"), float) else "-"]
        )
    (output_dir / f"{prefix}.md").write_text(markdown_table(headers, body), encoding="utf-8")

    column_spec = "ll" + "c" * len(metric_keys) + "c"
    latex_lines = [
        r"\begin{table*}[t]",
        r"\scriptsize\sf\centering",
        rf"\caption{{{prefix.replace('_', ' ').title()}. Values are reported as point estimate [95\% CI]. The final column reports the mean paired-bootstrap p-value across all listed metrics.}}",
        r"\resizebox{\textwidth}{!}{%",
        rf"\begin{{tabular}}{{{column_spec}}}",
        r"\toprule",
        "Group/Category & Model/Module & "
        + " & ".join(latex_metric_header(key) for key in metric_keys)
        + r" & \shortstack{Mean\\$p$-value} \\",
        r"\midrule",
    ]
    for row in rows:
        p_text = format_p_value(row["mean_p_value"]) if isinstance(row.get("mean_p_value"), float) else "-"
        values = [str(row.get("group", "")), str(row["name"])]
        values += [str(row.get(f"{key}_ci_text", "NA")) for key in metric_keys]
        values.append(p_text)
        latex_lines.append(" & ".join(values) + r" \\")
    latex_lines.extend([r"\bottomrule", r"\end{tabular}", "}", r"\end{table*}", ""])
    (output_dir / f"{prefix}.tex").write_text("\n".join(latex_lines), encoding="utf-8")


def write_label_tables(prefix: str, rows: list[dict[str, Any]], metric_names: list[str], output_dir: Path) -> None:
    csv_rows: list[dict[str, Any]] = []
    for row in rows:
        flat = {"label": row["label"]}
        for key in metric_names:
            flat[key] = row.get(key)
            flat[f"{key}_95ci"] = row.get(f"{key}_ci_text")
            flat[f"{key}_ci_low"] = row.get(f"{key}_ci_low")
            flat[f"{key}_ci_high"] = row.get(f"{key}_ci_high")
        csv_rows.append(flat)
    write_csv(output_dir / f"{prefix}.csv", csv_rows)

    headers = ["Label"] + [METRIC_DISPLAY_NAMES.get(key, key) for key in metric_names]
    body = [[str(row["label"])] + [str(row.get(f"{key}_ci_text", "NA")) for key in metric_names] for row in rows]
    (output_dir / f"{prefix}.md").write_text(markdown_table(headers, body), encoding="utf-8")

    column_spec = "l" + "c" * len(metric_names)
    latex_lines = [
        r"\begin{table*}[t]",
        r"\footnotesize\sf\centering",
        rf"\caption{{{prefix.replace('_', ' ').title()}. Values are reported as point estimate [95\% CI].}}",
        rf"\begin{{tabular}}{{{column_spec}}}",
        r"\toprule",
        "Label & " + " & ".join(latex_metric_header(key) for key in metric_names) + r" \\",
        r"\midrule",
    ]
    for row in rows:
        values = [str(row["label"])] + [str(row.get(f"{key}_ci_text", "NA")) for key in metric_names]
        latex_lines.append(" & ".join(values) + r" \\")
    latex_lines.extend([r"\bottomrule", r"\end{tabular}", r"\end{table*}", ""])
    (output_dir / f"{prefix}.tex").write_text("\n".join(latex_lines), encoding="utf-8")


def validate_entries(entries: list[dict[str, Any]], label: str) -> None:
    for entry in entries:
        run_dir = Path(str(entry.get("run_dir", ""))).expanduser()
        if not run_dir.is_dir():
            raise FileNotFoundError(f"{label} 中模型目录不存在：{run_dir}")
        if not (run_dir / "config.yaml").is_file():
            raise FileNotFoundError(f"{label} 中模型目录缺少 config.yaml：{run_dir}")
        resolve_checkpoint_path(run_dir, "best_macro_f1")


def main() -> None:
    args = parse_args()
    cfg_path = Path(args.config).expanduser().resolve()
    cfg = read_yaml(cfg_path)
    selected_tables = parse_selected_tables(args.tables)
    cache_namespace = "all" if selected_tables == ("table2", "table3", "table4") else "_".join(selected_tables)
    selection_alias = str(cfg.get("selection_alias", "best_macro_f1")).strip() or "best_macro_f1"
    bootstrap_samples = int(args.bootstrap_samples or cfg.get("bootstrap_samples", 2000))
    seed = int(cfg.get("seed", 20260526))
    default_threshold = float(cfg.get("threshold", 0.5))
    force = bool(args.force or cfg.get("force_recompute_predictions", False))
    device = resolve_device(args.device)

    path_cfg = load_path_config(Path(resolve_default_config_path("task1", "path.yaml")))
    train_cfg = load_train_config(Path(resolve_default_config_path("task1", "train.yaml")))
    train_cfg["task_name"] = "task1"
    output_root = Path(path_cfg["output_dir"]).resolve() / train_cfg["train_run_dir_name"] / "task1"
    output_dir = output_root / str(cfg.get("output_dir_name", "exp_task1_distinct")).strip()
    output_dir.mkdir(parents=True, exist_ok=True)

    table2_entries = list(cfg.get("table2", {}).get("models", [])) if "table2" in selected_tables else []
    table4_entries = list(cfg.get("table4", {}).get("modules", [])) if "table4" in selected_tables else []
    table3_model = dict(cfg.get("table3", {}).get("model", {})) if "table3" in selected_tables else {}
    if table2_entries:
        validate_entries(table2_entries, "table2")
    if table4_entries:
        validate_entries(table4_entries, "table4")
    if table3_model:
        validate_entries([table3_model], "table3")
    if args.dry_run:
        print("[TASK1 distinct] dry-run 通过，模型路径和 checkpoint 均可访问。")
        print(f"[TASK1 distinct] 输出目录将为：{output_dir}")
        print(f"[TASK1 distinct] 运行表格：{','.join(selected_tables)}")
        print(f"[TASK1 distinct] 推理设备：{device}")
        return

    seed_everything(seed)
    training_context = prepare_training_context(
        path_cfg=path_cfg,
        train_cfg=train_cfg,
        seed=int(train_cfg.get("seed", 42)),
        max_exams_per_task=int(train_cfg.get("max_exams_per_task", 0)),
        required_tasks={"task1"},
    )
    num_workers = min(4, int(train_cfg.get("num_workers", 4)))

    all_entries: list[dict[str, Any]] = []
    seen: set[str] = set()
    for entry in [*table2_entries, *table4_entries, *([table3_model] if table3_model else [])]:
        name = str(entry.get("display_name", "")).strip()
        run_dir = str(entry.get("run_dir", "")).strip()
        key = f"{name}::{run_dir}"
        if key in seen:
            continue
        seen.add(key)
        all_entries.append(entry)

    predictions_by_name: dict[str, dict[str, Any]] = {}
    iterator = tqdm(all_entries, desc="收集模型预测", dynamic_ncols=True) if tqdm else all_entries
    for entry in iterator:
        display_name = str(entry["display_name"])
        pred = collect_predictions(
            display_name=display_name,
            run_dir=Path(str(entry["run_dir"])).expanduser().resolve(),
            alias=selection_alias,
            train_cfg=train_cfg,
            training_context=training_context,
            output_dir=output_dir,
            default_threshold=default_threshold,
            force=force,
            num_workers=num_workers,
            device=device,
            cache_namespace=cache_namespace,
            eval_batch_size_override=max(0, int(args.eval_batch_size)),
        )
        predictions_by_name[display_name] = pred

    n = int(next(iter(predictions_by_name.values()))["y_true"].shape[0])
    rng = np.random.default_rng(seed)
    bootstrap_indices = rng.integers(0, n, size=(bootstrap_samples, n), endpoint=False)
    bootstrap_filename = "bootstrap_indices.npy" if cache_namespace == "all" else f"bootstrap_indices_{cache_namespace}.npy"
    np.save(output_dir / bootstrap_filename, bootstrap_indices)

    overall_metrics = list(cfg.get("metrics", {}).get("overall", []))
    label_metrics = list(cfg.get("metrics", {}).get("label_wise", []))

    summary_tables: dict[str, Any] = {}
    if "table2" in selected_tables:
        print("[TASK1 distinct] 计算 Table 2 CI 与平均 p 值...")
        table2_rows, _table2_bootstrap = build_comparison_rows(
            entries=table2_entries,
            predictions_by_name=predictions_by_name,
            metric_keys=overall_metrics,
            bootstrap_indices=bootstrap_indices,
            reference_name=str(cfg.get("table2", {}).get("reference", "Label graph MIL")),
        )
        write_comparison_tables("table2_ci_p", table2_rows, overall_metrics, output_dir)
        summary_tables["table2"] = {
            "reference": cfg.get("table2", {}).get("reference", ""),
            "rows": table2_rows,
        }

    if "table3" in selected_tables:
        print("[TASK1 distinct] 计算 Table 3 分标签 CI...")
        table3_prediction = predictions_by_name[str(table3_model["display_name"])]
        table3_rows, _table3_bootstrap = build_label_rows(
            prediction=table3_prediction,
            metric_names=label_metrics,
            bootstrap_indices=bootstrap_indices,
        )
        write_label_tables("table3_labelwise_ci", table3_rows, label_metrics, output_dir)
        summary_tables["table3"] = {
            "model": table3_model.get("display_name", ""),
            "rows": table3_rows,
        }

    if "table4" in selected_tables:
        print("[TASK1 distinct] 计算 Table 4 CI 与平均 p 值...")
        table4_rows, _table4_bootstrap = build_comparison_rows(
            entries=table4_entries,
            predictions_by_name=predictions_by_name,
            metric_keys=overall_metrics,
            bootstrap_indices=bootstrap_indices,
            reference_name=str(cfg.get("table4", {}).get("reference", "Full Label Graph")),
        )
        write_comparison_tables("table4_ci_p", table4_rows, overall_metrics, output_dir)
        summary_tables["table4"] = {
            "reference": cfg.get("table4", {}).get("reference", ""),
            "rows": table4_rows,
        }

    summary = {
        "generated_config": str(cfg_path),
        "output_dir": str(output_dir),
        "selected_tables": list(selected_tables),
        "cache_namespace": cache_namespace,
        "selection_alias": selection_alias,
        "bootstrap_samples": bootstrap_samples,
        "seed": seed,
        "device": str(device),
        "num_test_examinations": n,
        "tables": summary_tables,
    }
    summary_filename = "distinct_summary.json" if cache_namespace == "all" else f"distinct_summary_{cache_namespace}.json"
    (output_dir / summary_filename).write_text(
        json.dumps(to_builtin_type(summary), ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    model_path_rows = [
        {
            "display_name": name,
            "run_dir": pred["run_dir"],
            "checkpoint_path": pred["checkpoint_path"],
            "best_epoch": pred["best_epoch"],
        }
        for name, pred in predictions_by_name.items()
    ]
    model_paths_filename = "model_paths.csv" if cache_namespace == "all" else f"model_paths_{cache_namespace}.csv"
    write_csv(output_dir / model_paths_filename, model_path_rows)
    print(f"[TASK1 distinct] 完成，结果目录：{output_dir}")


if __name__ == "__main__":
    main()
