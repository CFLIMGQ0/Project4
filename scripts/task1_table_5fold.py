#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import itertools
import json
import math
import os
import random
import re
import shutil
import sys
from pathlib import Path
from typing import Any

import numpy as np
import yaml


def _ensure_project_runtime_python() -> None:
    import importlib.util

    if importlib.util.find_spec("torch") is not None:
        return
    if os.environ.get("PROJECT4_5FOLD_RUNTIME_REEXEC") == "1":
        return

    current_python = Path(sys.executable).resolve()
    candidates: list[str] = []
    override = os.environ.get("PROJECT4_TRAIN_PYTHON", "").strip()
    if override:
        candidates.append(override)
    if current_python.parent.name == "bin":
        conda_root = current_python.parent.parent
        candidates.append(str(conda_root / "envs" / "myenv" / "bin" / "python"))
    candidates.append("/home/Lim/anaconda3/envs/myenv/bin/python")

    seen: set[str] = set()
    for candidate_text in candidates:
        if not candidate_text or candidate_text in seen:
            continue
        seen.add(candidate_text)
        candidate = Path(candidate_text).expanduser()
        if not candidate.is_file() or candidate.resolve() == current_python:
            continue
        os.environ["PROJECT4_5FOLD_RUNTIME_REEXEC"] = "1"
        print(
            f"[task1_table_5fold.py] 当前解释器 {current_python} 缺少 torch，自动切换到 {candidate}",
            file=sys.stderr,
        )
        os.execv(str(candidate), [str(candidate), *sys.argv])


_ensure_project_runtime_python()

import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tasks import get_task_spec
from tasks.common import derive_patient_id_from_exam_dir
from train import (  # noqa: E402
    TRACKER_ALIAS_TO_META,
    auto_series_resume_checkpoint,
    build_multilabel_minority_balance,
    build_task_records,
    compute_multilabel_pos_weight,
    is_auto_series_run_complete,
    load_existing_auto_series_result,
    load_model_config,
    load_path_config,
    load_train_config,
    maybe_limit_records,
    resolve_default_config_path,
    run_model_job,
    seed_everything,
    to_builtin_type,
)

try:
    from tqdm import tqdm
except Exception:  # pragma: no cover
    tqdm = None


OVERALL_DISPLAY_NAMES = {
    "macro_recall": "Macro Recall",
    "macro_precision": "Macro Precision",
    "macro_f1": "Macro F1",
    "macro_roc_auc": "Macro ROC-AUC",
    "macro_pr_auc": "Macro PR-AUC",
    "subset_accuracy": "Subset accuracy",
    "hamming_loss": "Hamming loss",
    "kappa": "Kappa",
}
LABEL_METRIC_DISPLAY_NAMES = {
    "recall": "Recall",
    "precision": "Precision",
    "f1": "F1",
    "roc_auc": "ROC-AUC",
    "pr_auc": "PR-AUC",
}
LABEL_DISPLAY_NAMES = {
    "label_esophageal_smt": "Esophageal submucosal tumor",
    "label_esophageal_mucosal_or_tumor": "Esophageal mucosal lesion",
    "label_gastritis": "Gastritis",
}
RUN_KEYS_TO_INHERIT = {
    "batch_size",
    "eval_batch_size",
    "train_max_instances",
    "eval_max_instances",
    "train_max_batch_instances",
    "eval_max_batch_instances",
    "pin_memory",
    "persistent_workers",
    "loader_prefetch_factor",
    "image_cache_mode",
    "image_cache_scope",
    "image_cache_dir",
    "image_cache_warmup",
    "memory_cache_size",
    "random_instance_dropout",
    "optimizer_name",
    "lr",
    "weight_decay",
    "warmup_ratio",
    "grad_accum_steps",
    "amp",
    "topk_evidence",
    "loss_name",
    "monitor_metric",
    "monitor_mode",
    "train_sampling_strategy",
    "eval_sampling_strategy",
}
T_CRITICAL_95 = {
    1: float("nan"),
    2: 12.706,
    3: 4.303,
    4: 3.182,
    5: 2.776,
    6: 2.571,
    7: 2.447,
    8: 2.365,
    9: 2.306,
    10: 2.262,
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="运行 TASK1 表2/表3/表4 的 5-fold 训练与汇总")
    parser.add_argument("--config", type=str, default=str(ROOT / "configs" / "task1" / "auto_5fold.yaml"))
    parser.add_argument("--path-config", type=str, default="")
    parser.add_argument("--train-config", type=str, default="")
    parser.add_argument("--model-config", type=str, default="")
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--patience", type=int, default=None)
    parser.add_argument("--image-size", type=int, default=None)
    parser.add_argument("--num-workers", type=int, default=None)
    parser.add_argument("--max-exams-per-task", type=int, default=None)
    parser.add_argument("--folds", type=int, default=None)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--no-pretrained", action="store_true")
    parser.add_argument("--disable-multi-gpu", action="store_true")
    return parser.parse_args()


def read_yaml(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(f"未找到配置文件：{path}")
    payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"配置文件格式错误：{path}")
    return payload


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames: list[str] = []
    for row in rows:
        for key in row.keys():
            if key not in fieldnames:
                fieldnames.append(key)
    with path.open("w", encoding="utf-8-sig", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def markdown_table(headers: list[str], rows: list[list[str]]) -> str:
    lines = ["| " + " | ".join(headers) + " |", "| " + " | ".join(["---"] * len(headers)) + " |"]
    for row in rows:
        lines.append("| " + " | ".join(row) + " |")
    return "\n".join(lines) + "\n"


def safe_float(value: Any) -> float:
    try:
        return float(value)
    except Exception:
        return float("nan")


def format_float(value: Any, digits: int = 4) -> str:
    numeric = safe_float(value)
    if math.isnan(numeric):
        return "NA"
    return f"{numeric:.{digits}f}"


def format_p_value(value: Any) -> str:
    numeric = safe_float(value)
    if math.isnan(numeric):
        return "-"
    if numeric < 0.001:
        return "$<$0.001"
    return f"{numeric:.4f}"


def latex_header(text: str) -> str:
    if " " in text:
        return r"\shortstack{" + text.replace(" ", r"\\") + "}"
    return text


def latex_metric_header(metric_name: str, display_map: dict[str, str]) -> str:
    return latex_header(display_map.get(metric_name, metric_name))


def latex_ci_cell(mean: float, low: float, high: float) -> str:
    return rf"\shortstack{{{format_float(mean)}\\{{[}}{format_float(low)}, {format_float(high)}{{]}}}}"


def dir_slug(text: str) -> str:
    slug = re.sub(r"[^A-Za-z0-9]+", "_", str(text).strip().lower()).strip("_")
    return slug or "model"


def source_run_payload(source_run_dir: Path) -> dict[str, Any]:
    config_path = source_run_dir / "config.yaml"
    if not config_path.is_file():
        raise FileNotFoundError(f"源训练目录缺少 config.yaml：{source_run_dir}")
    payload = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"源训练配置格式错误：{config_path}")
    return payload


def inherit_run_overrides(source_payload: dict[str, Any]) -> dict[str, Any]:
    run_payload = source_payload.get("run", {})
    if not isinstance(run_payload, dict):
        return {}
    return {key: run_payload[key] for key in RUN_KEYS_TO_INHERIT if key in run_payload}


def normalize_entry(raw: dict[str, Any], *, inherit_run_from_source: bool) -> dict[str, Any]:
    if not isinstance(raw, dict):
        raise ValueError("5-fold entry 必须是字典")
    source_text = str(raw.get("source_run_dir", "")).strip()
    source_run_dir = Path(source_text).expanduser() if source_text else None
    source_payload: dict[str, Any] = {}
    if source_run_dir is not None:
        source_payload = source_run_payload(source_run_dir)

    name = str(raw.get("name", "")).strip() or dir_slug(raw.get("display_name", "model"))
    display_name = str(raw.get("display_name", name)).strip()
    model_name = str(raw.get("base_model_name", raw.get("model_name", source_payload.get("model_name", "")))).strip()
    if not model_name:
        raise ValueError(f"{display_name} 缺少 model_name/base_model_name")

    model_params = {}
    if isinstance(source_payload.get("model_params"), dict):
        model_params.update(source_payload["model_params"])
    if isinstance(raw.get("model_params"), dict):
        model_params.update(raw["model_params"])

    run_overrides = {}
    if inherit_run_from_source:
        run_overrides.update(inherit_run_overrides(source_payload))
    if isinstance(raw.get("run_overrides"), dict):
        run_overrides.update(raw["run_overrides"])
    run_overrides.pop("seed", None)

    shared_key = str(raw.get("shared_key", "")).strip() or name
    return {
        **raw,
        "name": name,
        "shared_key": shared_key,
        "display_name": display_name,
        "base_model_name": model_name,
        "model_params": model_params,
        "run_overrides": run_overrides,
        "source_run_dir": str(source_run_dir) if source_run_dir is not None else "",
    }


def load_entries(cfg: dict[str, Any]) -> tuple[list[dict[str, Any]], dict[str, list[dict[str, Any]]]]:
    inherit_run_from_source = bool(cfg.get("inherit_run_from_source", True))
    table2 = [normalize_entry(item, inherit_run_from_source=inherit_run_from_source) for item in cfg.get("table2", {}).get("models", [])]
    table4 = [normalize_entry(item, inherit_run_from_source=inherit_run_from_source) for item in cfg.get("table4", {}).get("modules", [])]
    table3_model = normalize_entry(cfg.get("table3", {}).get("model", {}), inherit_run_from_source=inherit_run_from_source)

    all_entries: list[dict[str, Any]] = []
    seen: set[str] = set()
    for entry in [*table2, *table4, table3_model]:
        key = str(entry["shared_key"])
        if key in seen:
            continue
        seen.add(key)
        all_entries.append(entry)
    return all_entries, {"table2": table2, "table4": table4, "table3": [table3_model]}


def resolve_task_csv(path_cfg: dict[str, str], train_cfg: dict[str, Any], task_name: str) -> Path:
    output_root = Path(path_cfg["output_dir"]).resolve()
    task_data_root = Path(path_cfg.get("dataset_base_root", output_root)).resolve()
    task_selection_dir = task_data_root / train_cfg["task_selection_dir_name"]
    spec = get_task_spec(task_name)
    candidates = [
        task_selection_dir / spec.data_subdir / spec.datalist_filename,
        task_selection_dir / "task1" / "datalist.csv",
        task_selection_dir / "task1_gastro3" / "datalist.csv",
    ]
    for candidate in candidates:
        if candidate.is_file():
            return candidate
    raise FileNotFoundError("未找到 TASK1 datalist，可尝试先运行 scripts/task1_build_datalist.py")


def label_values(record: dict[str, Any], label_names: list[str]) -> tuple[int, ...]:
    labels = record.get("labels")
    if isinstance(labels, (list, tuple)):
        return tuple(int(labels[index]) if index < len(labels) else 0 for index in range(len(label_names)))
    if isinstance(labels, dict):
        return tuple(int(labels.get(label_name, 0)) for label_name in label_names)
    return tuple(int(record.get(label_name, 0)) for label_name in label_names)


def make_stratified_folds(
    records: list[dict[str, Any]],
    *,
    label_names: list[str],
    folds: int,
    seed: int,
    group_by_patient: bool = False,
) -> list[list[dict[str, Any]]]:
    rng = random.Random(seed)
    if group_by_patient:
        grouped: dict[str, list[dict[str, Any]]] = {}
        for record in records:
            patient_id = str(record.get("patient_id", "")).strip() or derive_patient_id_from_exam_dir(str(record.get("exam_dir", "")))
            grouped.setdefault(patient_id, []).append(record)
        units = list(grouped.values())
        combo_to_units: dict[tuple[int, ...], list[list[dict[str, Any]]]] = {}
        for unit in units:
            combo = tuple(int(any(label_values(record, label_names)[idx] for record in unit)) for idx in range(len(label_names)))
            combo_to_units.setdefault(combo, []).append(unit)
        fold_units: list[list[list[dict[str, Any]]]] = [[] for _ in range(folds)]
        for combo_index, combo in enumerate(sorted(combo_to_units.keys())):
            items = combo_to_units[combo]
            rng.shuffle(items)
            for item_index, unit in enumerate(items):
                fold_units[(combo_index + item_index) % folds].append(unit)
        return [[record for unit in fold for record in unit] for fold in fold_units]

    combo_to_records: dict[tuple[int, ...], list[dict[str, Any]]] = {}
    for record in records:
        combo_to_records.setdefault(label_values(record, label_names), []).append(record)

    split_folds: list[list[dict[str, Any]]] = [[] for _ in range(folds)]
    for combo_index, combo in enumerate(sorted(combo_to_records.keys())):
        items = combo_to_records[combo]
        rng.shuffle(items)
        for item_index, record in enumerate(items):
            split_folds[(combo_index + item_index) % folds].append(record)
    for fold in split_folds:
        rng.shuffle(fold)
    return split_folds


def build_fold_context(
    *,
    base_context: dict[str, Any],
    task_csv: Path,
    split_data: dict[str, list[dict[str, Any]]],
    train_cfg: dict[str, Any],
    task_name: str,
) -> dict[str, Any]:
    task_spec = get_task_spec(task_name)
    structured_fit_records = list(split_data["train"])
    class_balance_cfg = dict(train_cfg.get("class_balance", {"enabled": False}))
    balance_report = None
    if task_spec.is_multilabel and bool(class_balance_cfg.get("enabled", False)):
        balanced_train_records, balance_report = build_multilabel_minority_balance(
            train_records=split_data["train"],
            label_names=task_spec.label_names,
            cfg=class_balance_cfg,
            seed=int(train_cfg.get("seed", 42)),
        )
        split_data = {
            "train": balanced_train_records,
            "val": split_data["val"],
            "test": split_data["test"],
        }

    pos_weight = compute_multilabel_pos_weight(split_data["train"]) if split_data["train"] else [1.0 for _ in task_spec.label_names]
    return {
        "output_root": base_context["output_root"],
        "task_selection_dir": base_context["task_selection_dir"],
        "tasks": {
            task_name: {
                "csv_path": str(task_csv),
                "split": split_data,
                "pos_weight": pos_weight,
                "balance_report": balance_report,
                "structured_metadata": None,
                "structured_report_enrich": None,
            }
        },
        "task_stats": {
            task_name: {
                "total_records": sum(len(split_data[key]) for key in ("train", "val", "test")),
                "train_original_size": len(structured_fit_records),
                "train_size": len(split_data["train"]),
                "val_size": len(split_data["val"]),
                "test_size": len(split_data["test"]),
                "class_balance_added_records": int(balance_report.get("added_records", 0)) if balance_report else 0,
            }
        },
    }


def extract_result_row(
    *,
    result: dict[str, Any],
    entry: dict[str, Any],
    fold_index: int,
    selection_alias: str,
    metrics: list[str],
) -> dict[str, Any]:
    payload = result.get("test_results", {}).get(selection_alias, {})
    metric_payload = payload.get("metrics", {}) if isinstance(payload, dict) else {}
    row = {
        "fold": fold_index,
        "shared_key": entry["shared_key"],
        "name": entry["name"],
        "display_name": entry["display_name"],
        "base_model_name": entry["base_model_name"],
        "run_dir": result.get("train_dir", ""),
        "checkpoint_path": payload.get("checkpoint_path", ""),
        "best_epoch": payload.get("best_epoch", ""),
    }
    for metric in metrics:
        row[metric] = safe_float(metric_payload.get(metric, float("nan")))
    for label_key in LABEL_DISPLAY_NAMES:
        for metric in LABEL_METRIC_DISPLAY_NAMES:
            row[f"{metric}_{label_key}"] = safe_float(metric_payload.get(f"{metric}_{label_key}", float("nan")))
    return row


def summarize_values(values: list[float]) -> dict[str, float]:
    clean = np.asarray([value for value in values if not math.isnan(value)], dtype=np.float64)
    if clean.size == 0:
        return {"n": 0, "mean": float("nan"), "std": float("nan"), "ci_low": float("nan"), "ci_high": float("nan")}
    mean = float(np.mean(clean))
    std = float(np.std(clean, ddof=1)) if clean.size > 1 else 0.0
    if clean.size > 1:
        critical = T_CRITICAL_95.get(int(clean.size), 1.96)
        half_width = float(critical * std / math.sqrt(clean.size))
    else:
        half_width = 0.0
    return {
        "n": int(clean.size),
        "mean": mean,
        "std": std,
        "ci_low": mean - half_width,
        "ci_high": mean + half_width,
    }


def paired_sign_flip_p(reference_values: list[float], other_values: list[float]) -> float:
    pairs = [
        (safe_float(ref), safe_float(other))
        for ref, other in zip(reference_values, other_values)
        if not math.isnan(safe_float(ref)) and not math.isnan(safe_float(other))
    ]
    if not pairs:
        return float("nan")
    diffs = np.asarray([ref - other for ref, other in pairs], dtype=np.float64)
    if np.allclose(diffs, 0.0):
        return 1.0
    observed = abs(float(np.mean(diffs)))
    count = 0
    total = 0
    for signs in itertools.product((-1.0, 1.0), repeat=len(diffs)):
        total += 1
        stat = abs(float(np.mean(diffs * np.asarray(signs, dtype=np.float64))))
        if stat >= observed - 1e-12:
            count += 1
    return float(count / total)


def summarize_comparison_table(
    *,
    entries: list[dict[str, Any]],
    rows_by_key: dict[str, list[dict[str, Any]]],
    metric_names: list[str],
    reference_name: str,
    group_field: str,
) -> list[dict[str, Any]]:
    reference_entry = next((entry for entry in entries if str(entry["display_name"]) == reference_name), entries[0])
    reference_rows = rows_by_key.get(reference_entry["shared_key"], [])
    summary_rows: list[dict[str, Any]] = []
    for entry in entries:
        model_rows = rows_by_key.get(entry["shared_key"], [])
        summary: dict[str, Any] = {
            group_field: entry.get(group_field, entry.get("group", entry.get("category", ""))),
            "name": entry["display_name"],
            "shared_key": entry["shared_key"],
            "n_folds": len(model_rows),
        }
        p_values: list[float] = []
        for metric in metric_names:
            values = [safe_float(row.get(metric)) for row in model_rows]
            stats = summarize_values(values)
            summary[f"{metric}_mean"] = stats["mean"]
            summary[f"{metric}_std"] = stats["std"]
            summary[f"{metric}_ci_low"] = stats["ci_low"]
            summary[f"{metric}_ci_high"] = stats["ci_high"]
            summary[f"{metric}_text"] = (
                f"{format_float(stats['mean'])} [{format_float(stats['ci_low'])}, {format_float(stats['ci_high'])}]"
            )
            if entry["shared_key"] == reference_entry["shared_key"]:
                p_value = 1.0
            else:
                ref_values = [safe_float(row.get(metric)) for row in reference_rows]
                p_value = paired_sign_flip_p(ref_values, values)
            summary[f"{metric}_p_value"] = p_value
            if not math.isnan(p_value):
                p_values.append(p_value)
        summary["mean_p_value"] = float(np.mean(p_values)) if p_values else float("nan")
        summary_rows.append(summary)
    return summary_rows


def summarize_label_table(
    *,
    model_rows: list[dict[str, Any]],
    metric_names: list[str],
) -> list[dict[str, Any]]:
    summary_rows: list[dict[str, Any]] = []
    for label_key, label_display in LABEL_DISPLAY_NAMES.items():
        row: dict[str, Any] = {"label": label_display, "label_key": label_key, "n_folds": len(model_rows)}
        for metric in metric_names:
            key = f"{metric}_{label_key}"
            values = [safe_float(item.get(key)) for item in model_rows]
            stats = summarize_values(values)
            row[f"{metric}_mean"] = stats["mean"]
            row[f"{metric}_std"] = stats["std"]
            row[f"{metric}_ci_low"] = stats["ci_low"]
            row[f"{metric}_ci_high"] = stats["ci_high"]
            row[f"{metric}_text"] = (
                f"{format_float(stats['mean'])} [{format_float(stats['ci_low'])}, {format_float(stats['ci_high'])}]"
            )
        summary_rows.append(row)
    return summary_rows


def write_comparison_outputs(
    *,
    prefix: str,
    rows: list[dict[str, Any]],
    output_dir: Path,
    metric_names: list[str],
    group_field: str,
) -> None:
    csv_rows: list[dict[str, Any]] = []
    for row in rows:
        flat = {
            group_field: row.get(group_field, ""),
            "name": row.get("name", ""),
            "shared_key": row.get("shared_key", ""),
            "n_folds": row.get("n_folds", 0),
            "mean_p_value": row.get("mean_p_value", ""),
        }
        for metric in metric_names:
            for suffix in ("mean", "std", "ci_low", "ci_high", "text", "p_value"):
                flat[f"{metric}_{suffix}"] = row.get(f"{metric}_{suffix}", "")
        csv_rows.append(flat)
    write_csv(output_dir / f"{prefix}.csv", csv_rows)

    headers = [group_field.title(), "Model/Module"] + [OVERALL_DISPLAY_NAMES.get(metric, metric) for metric in metric_names] + ["Mean p-value"]
    body: list[list[str]] = []
    for row in rows:
        body.append(
            [str(row.get(group_field, "")), str(row.get("name", ""))]
            + [str(row.get(f"{metric}_text", "NA")) for metric in metric_names]
            + [format_p_value(row.get("mean_p_value"))]
        )
    (output_dir / f"{prefix}.md").write_text(markdown_table(headers, body), encoding="utf-8")

    column_spec = "ll" + "c" * len(metric_names) + "c"
    latex_lines = [
        r"\begin{table*}[t]",
        r"\scriptsize\sf\centering",
        rf"\caption{{{prefix.replace('_', ' ').title()}. Values are five-fold mean [95\% CI]. The final column reports the mean paired sign-flip p-value across metrics.}}",
        r"\resizebox{\textwidth}{!}{%",
        rf"\begin{{tabular}}{{{column_spec}}}",
        r"\toprule",
        f"{group_field.title()} & Model/Module & "
        + " & ".join(latex_metric_header(metric, OVERALL_DISPLAY_NAMES) for metric in metric_names)
        + r" & \shortstack{Mean\\$p$-value} \\",
        r"\midrule",
    ]
    for row in rows:
        values = [str(row.get(group_field, "")), str(row.get("name", ""))]
        values += [
            latex_ci_cell(
                safe_float(row.get(f"{metric}_mean")),
                safe_float(row.get(f"{metric}_ci_low")),
                safe_float(row.get(f"{metric}_ci_high")),
            )
            for metric in metric_names
        ]
        values.append(format_p_value(row.get("mean_p_value")))
        latex_lines.append(" & ".join(values) + r" \\")
    latex_lines.extend([r"\bottomrule", r"\end{tabular}", "}", r"\end{table*}", ""])
    (output_dir / f"{prefix}.tex").write_text("\n".join(latex_lines), encoding="utf-8")


def write_label_outputs(
    *,
    prefix: str,
    rows: list[dict[str, Any]],
    output_dir: Path,
    metric_names: list[str],
) -> None:
    csv_rows: list[dict[str, Any]] = []
    for row in rows:
        flat = {"label": row["label"], "label_key": row["label_key"], "n_folds": row.get("n_folds", 0)}
        for metric in metric_names:
            for suffix in ("mean", "std", "ci_low", "ci_high", "text"):
                flat[f"{metric}_{suffix}"] = row.get(f"{metric}_{suffix}", "")
        csv_rows.append(flat)
    write_csv(output_dir / f"{prefix}.csv", csv_rows)

    headers = ["Label"] + [LABEL_METRIC_DISPLAY_NAMES.get(metric, metric) for metric in metric_names]
    body = [[str(row["label"])] + [str(row.get(f"{metric}_text", "NA")) for metric in metric_names] for row in rows]
    (output_dir / f"{prefix}.md").write_text(markdown_table(headers, body), encoding="utf-8")

    column_spec = "l" + "c" * len(metric_names)
    latex_lines = [
        r"\begin{table*}[t]",
        r"\footnotesize\sf\centering",
        rf"\caption{{{prefix.replace('_', ' ').title()}. Values are five-fold mean [95\% CI].}}",
        r"\resizebox{\textwidth}{!}{%",
        rf"\begin{{tabular}}{{{column_spec}}}",
        r"\toprule",
        "Label & " + " & ".join(latex_metric_header(metric, LABEL_METRIC_DISPLAY_NAMES) for metric in metric_names) + r" \\",
        r"\midrule",
    ]
    for row in rows:
        values = [str(row["label"])]
        values += [
            latex_ci_cell(
                safe_float(row.get(f"{metric}_mean")),
                safe_float(row.get(f"{metric}_ci_low")),
                safe_float(row.get(f"{metric}_ci_high")),
            )
            for metric in metric_names
        ]
        latex_lines.append(" & ".join(values) + r" \\")
    latex_lines.extend([r"\bottomrule", r"\end{tabular}", "}", r"\end{table*}", ""])
    (output_dir / f"{prefix}.tex").write_text("\n".join(latex_lines), encoding="utf-8")


def main() -> None:
    args = parse_args()
    cfg_path = Path(args.config).expanduser().resolve()
    cfg = read_yaml(cfg_path)

    path_cfg_path = Path(args.path_config).expanduser() if str(args.path_config).strip() else Path(resolve_default_config_path("task1", "path.yaml"))
    train_cfg_path = Path(args.train_config).expanduser() if str(args.train_config).strip() else Path(resolve_default_config_path("task1", "train.yaml"))
    model_cfg_path = Path(args.model_config).expanduser() if str(args.model_config).strip() else Path(resolve_default_config_path("task1", "model.yaml"))

    path_cfg = load_path_config(path_cfg_path)
    train_cfg = load_train_config(train_cfg_path)
    train_cfg["task_name"] = "task1"
    model_cfg = load_model_config(model_cfg_path)

    seed = int(args.seed if args.seed is not None else cfg.get("seed", train_cfg.get("seed", 42)))
    max_epochs = int(args.epochs if args.epochs is not None else train_cfg["max_epochs"])
    patience = int(args.patience if args.patience is not None else train_cfg["patience"])
    image_size = int(args.image_size if args.image_size is not None else train_cfg["image_size"])
    num_workers = int(args.num_workers if args.num_workers is not None else train_cfg["num_workers"])
    max_exams_per_task = int(args.max_exams_per_task if args.max_exams_per_task is not None else train_cfg.get("max_exams_per_task", 0))
    folds = int(args.folds if args.folds is not None else cfg.get("folds", 5))
    if folds < 3:
        raise ValueError("5-fold 配置中的 folds 至少应为 3")

    output_root = Path(path_cfg["output_dir"]).resolve() / train_cfg["train_run_dir_name"] / "task1"
    output_dir = output_root / str(cfg.get("output_dir_name", "exp_task1_5fold")).strip()
    output_dir.mkdir(parents=True, exist_ok=True)

    all_entries, table_entries = load_entries(cfg)
    overall_metrics = list(cfg.get("metrics", {}).get("overall", []))
    label_metrics = list(cfg.get("metrics", {}).get("label_wise", []))
    selection_alias = str(cfg.get("selection_alias", "best_macro_f1")).strip()
    if selection_alias not in TRACKER_ALIAS_TO_META:
        raise ValueError(f"selection_alias 仅支持：{list(TRACKER_ALIAS_TO_META.keys())}")

    task_spec = get_task_spec("task1")
    task_csv = resolve_task_csv(path_cfg, train_cfg, "task1")
    records = build_task_records(
        task_csv_path=task_csv,
        task_name="task1",
        min_instances=int(train_cfg["min_instances"]),
        dataset_root=path_cfg.get("dataset_root"),
    )
    records = maybe_limit_records(records, max_num=max_exams_per_task, seed=seed)
    split_folds = make_stratified_folds(
        records,
        label_names=list(task_spec.label_names),
        folds=folds,
        seed=seed,
        group_by_patient=bool(train_cfg.get("group_by_patient", False)),
    )

    os.environ.setdefault("CUDA_DEVICE_ORDER", "PCI_BUS_ID")
    if "CUDA_VISIBLE_DEVICES" not in os.environ:
        os.environ["CUDA_VISIBLE_DEVICES"] = ",".join(str(item) for item in train_cfg["gpu_ids"])
    torch.set_float32_matmul_precision("medium")
    cuda_available = torch.cuda.is_available()
    visible_gpu_count = torch.cuda.device_count() if cuda_available else 0
    use_multi_gpu = (not args.disable_multi_gpu) and visible_gpu_count > 1
    active_gpu_count = visible_gpu_count if use_multi_gpu else (1 if visible_gpu_count > 0 else 0)
    if cuda_available:
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.benchmark = True
    pretrained = not args.no_pretrained

    print(
        f"[TASK1 5-fold] folds={folds} records={len(records)} output_dir={output_dir} "
        f"gpu_count={visible_gpu_count} use_multi_gpu={use_multi_gpu}"
    )
    print(f"[TASK1 5-fold] 唯一训练配置数={len(all_entries)}，预计训练次数={len(all_entries) * folds}")
    if args.dry_run:
        for fold_index, fold_records in enumerate(split_folds, start=1):
            val_index = (fold_index % folds) + 1
            train_size = sum(len(split_folds[index]) for index in range(folds) if index not in {fold_index - 1, val_index - 1})
            print(f"fold_{fold_index}: train={train_size} val={len(split_folds[val_index - 1])} test={len(fold_records)}")
        for entry in all_entries:
            print(f"- {entry['shared_key']} | {entry['display_name']} | {entry['base_model_name']}")
        return

    base_context = {
        "output_root": Path(path_cfg["output_dir"]).resolve(),
        "task_selection_dir": str((Path(path_cfg.get("dataset_base_root", path_cfg["output_dir"])).resolve() / train_cfg["task_selection_dir_name"])),
    }

    fold_result_rows: list[dict[str, Any]] = []
    model_path_rows: list[dict[str, Any]] = []
    result_by_fold_key: dict[tuple[int, str], dict[str, Any]] = {}
    force_rerun = bool(args.force or cfg.get("force_rerun", False))
    skip_completed = bool(cfg.get("skip_completed", True))
    run_test = bool(cfg.get("run_test", True))

    for fold_index in range(1, folds + 1):
        test_fold_index = fold_index - 1
        val_fold_index = fold_index % folds
        train_records = [
            record
            for idx, fold_records in enumerate(split_folds)
            if idx not in {test_fold_index, val_fold_index}
            for record in fold_records
        ]
        split_data = {
            "train": train_records,
            "val": list(split_folds[val_fold_index]),
            "test": list(split_folds[test_fold_index]),
        }
        fold_context = build_fold_context(
            base_context=base_context,
            task_csv=task_csv,
            split_data=split_data,
            train_cfg={**train_cfg, "seed": seed + fold_index},
            task_name="task1",
        )
        fold_dir = output_dir / f"fold_{fold_index}"
        fold_dir.mkdir(parents=True, exist_ok=True)
        (fold_dir / "split_stats.json").write_text(
            json.dumps(
                {
                    "fold": fold_index,
                    "train": len(split_data["train"]),
                    "val": len(split_data["val"]),
                    "test": len(split_data["test"]),
                },
                ensure_ascii=False,
                indent=2,
            ),
            encoding="utf-8",
        )

        iterator = enumerate(all_entries, start=1)
        if tqdm is not None:
            iterator = tqdm(list(iterator), desc=f"fold_{fold_index} 训练", dynamic_ncols=True)
        for model_index, entry in iterator:
            run_dir = fold_dir / f"train_{model_index:03d}_{dir_slug(entry['shared_key'])}"
            if force_rerun and run_dir.exists():
                shutil.rmtree(run_dir)

            if skip_completed and is_auto_series_run_complete(run_dir):
                print(f"[TASK1 5-fold] 跳过已完成：fold={fold_index} model={entry['display_name']}")
                result = load_existing_auto_series_result(run_dir, str(entry["base_model_name"]))
            else:
                resume_path = auto_series_resume_checkpoint(run_dir)
                print(f"[TASK1 5-fold] 开始训练：fold={fold_index}/{folds} model={entry['display_name']}")
                result = run_model_job(
                    model_name=str(entry["base_model_name"]),
                    run_dir=run_dir,
                    train_cfg=train_cfg,
                    model_cfg=model_cfg,
                    training_context=fold_context,
                    seed=seed + fold_index,
                    max_epochs=max_epochs,
                    patience=patience,
                    image_size=image_size,
                    num_workers=num_workers,
                    pretrained=pretrained,
                    use_multi_gpu=use_multi_gpu,
                    active_gpu_count=active_gpu_count,
                    run_test=run_test,
                    run_overrides=dict(entry.get("run_overrides", {})),
                    model_param_override=dict(entry.get("model_params", {})),
                    entry_metadata={
                        "five_fold": True,
                        "fold": fold_index,
                        "display_name": entry["display_name"],
                        "shared_key": entry["shared_key"],
                    },
                    resume_path=resume_path,
                )

            result_by_fold_key[(fold_index, str(entry["shared_key"]))] = result
            row = extract_result_row(
                result=result,
                entry=entry,
                fold_index=fold_index,
                selection_alias=selection_alias,
                metrics=overall_metrics,
            )
            fold_result_rows.append(row)
            model_path_rows.append(
                {
                    "fold": fold_index,
                    "shared_key": entry["shared_key"],
                    "display_name": entry["display_name"],
                    "run_dir": row["run_dir"],
                    "checkpoint_path": row["checkpoint_path"],
                    "best_epoch": row["best_epoch"],
                }
            )
            write_csv(output_dir / "fold_results.csv", fold_result_rows)
            write_csv(output_dir / "model_paths.csv", model_path_rows)

    rows_by_key: dict[str, list[dict[str, Any]]] = {}
    for row in fold_result_rows:
        rows_by_key.setdefault(str(row["shared_key"]), []).append(row)

    table2_summary = summarize_comparison_table(
        entries=table_entries["table2"],
        rows_by_key=rows_by_key,
        metric_names=overall_metrics,
        reference_name=str(cfg.get("table2", {}).get("reference", "Label graph MIL")),
        group_field="group",
    )
    write_comparison_outputs(
        prefix="table2_5fold_summary",
        rows=table2_summary,
        output_dir=output_dir,
        metric_names=overall_metrics,
        group_field="group",
    )

    table3_entry = table_entries["table3"][0]
    table3_rows = summarize_label_table(
        model_rows=rows_by_key.get(str(table3_entry["shared_key"]), []),
        metric_names=label_metrics,
    )
    write_label_outputs(
        prefix="table3_5fold_labelwise_summary",
        rows=table3_rows,
        output_dir=output_dir,
        metric_names=label_metrics,
    )

    table4_summary = summarize_comparison_table(
        entries=table_entries["table4"],
        rows_by_key=rows_by_key,
        metric_names=overall_metrics,
        reference_name=str(cfg.get("table4", {}).get("reference", "Full Label Graph")),
        group_field="category",
    )
    write_comparison_outputs(
        prefix="table4_5fold_summary",
        rows=table4_summary,
        output_dir=output_dir,
        metric_names=overall_metrics,
        group_field="category",
    )

    summary = {
        "config_path": str(cfg_path),
        "output_dir": str(output_dir),
        "folds": folds,
        "seed": seed,
        "records": len(records),
        "unique_models": len(all_entries),
        "selection_alias": selection_alias,
        "metrics": {"overall": overall_metrics, "label_wise": label_metrics},
        "table2": table2_summary,
        "table3": table3_rows,
        "table4": table4_summary,
    }
    (output_dir / "fivefold_summary.json").write_text(
        json.dumps(to_builtin_type(summary), ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(f"[TASK1 5-fold] 完成，结果目录：{output_dir}")


if __name__ == "__main__":
    main()
