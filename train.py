#!/usr/bin/env python3
from __future__ import annotations

import argparse
import copy
import csv
import json
import math
import os
import random
import re
import shutil
import subprocess
import sys
from dataclasses import asdict
from datetime import datetime
from pathlib import Path
from typing import Any, Callable


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
            f"[train.py] 当前解释器 {current_python} 缺少 torch，自动切换到 {candidate}",
            file=sys.stderr,
        )
        os.execv(str(candidate), [str(candidate), *sys.argv])


_ensure_project_runtime_python()

from runtime_guard import supervise_train_invocation_if_needed

if __name__ == "__main__":
    supervise_train_invocation_if_needed()

import numpy as np
import torch
import yaml
from torch.utils.data import DataLoader

from ablation_exp import build_all_ablation_experiments, list_ablation_experiment_names
from baselines import (
    GASTRO_BASELINE_CLASS_REGISTRY,
    GASTRO_BASELINE_MODEL_NAMES,
    build_gastro_baseline,
)
from exp_1 import EXP1_CLASS_REGISTRY, EXP1_MODEL_NAMES, build_exp1_model
from exp_2 import EXP2_CLASS_REGISTRY, EXP2_MODEL_NAMES, build_exp2_model
from exp_4 import EXP4_CLASS_REGISTRY, EXP4_MODEL_NAMES, build_exp4_model
from exp_5 import prepare_exp5_roi_cache
from exp_6 import EXP6_CLASS_REGISTRY, EXP6_MODEL_NAMES, build_exp6_model
from exp_8 import EXP8_CLASS_REGISTRY, EXP8_MODEL_NAMES, build_exp8_model
from model import GastroLabelGraphMIL, RGHMIL
from sotas import (
    GASTRO_SOTA_CLASS_REGISTRY,
    GASTRO_SOTA_MODEL_NAMES,
    build_gastro_sota,
)
from tasks import DEFAULT_GASTRO_TASK_NAME, get_task_spec, list_task_specs
from training import (
    InstanceAwareBatchSampler,
    MILBagDataset,
    Trainer,
    TrainerConfig,
    build_task_records,
    enrich_records_with_report_fields,
    prepare_structured_features,
    mil_collate_fn,
    split_records,
    to_builtin_type,
)


sys.dont_write_bytecode = True
os.environ.setdefault("PYTHONDONTWRITEBYTECODE", "1")

MODEL_SEQUENCE = (
    "gastro_baseline",
    "gastro_label_graph_mil",
)
CUSTOM_GASTRO_MODEL_NAMES = (
    "rg_hmil",
    "gastro_rg_hmil",
)
SUPPORTED_MODEL_NAMES = tuple(dict.fromkeys(MODEL_SEQUENCE + CUSTOM_GASTRO_MODEL_NAMES + GASTRO_BASELINE_MODEL_NAMES))
SUPPORTED_MODEL_NAMES = tuple(dict.fromkeys(SUPPORTED_MODEL_NAMES + GASTRO_SOTA_MODEL_NAMES))
SUPPORTED_MODEL_NAMES = tuple(dict.fromkeys(SUPPORTED_MODEL_NAMES + EXP1_MODEL_NAMES))
SUPPORTED_MODEL_NAMES = tuple(dict.fromkeys(SUPPORTED_MODEL_NAMES + EXP2_MODEL_NAMES))
SUPPORTED_MODEL_NAMES = tuple(dict.fromkeys(SUPPORTED_MODEL_NAMES + EXP4_MODEL_NAMES))
SUPPORTED_MODEL_NAMES = tuple(dict.fromkeys(SUPPORTED_MODEL_NAMES + EXP6_MODEL_NAMES))
SUPPORTED_MODEL_NAMES = tuple(dict.fromkeys(SUPPORTED_MODEL_NAMES + EXP8_MODEL_NAMES))
GASTRO_MODEL_NAMES = tuple(
    dict.fromkeys(
        (
            "gastro_baseline",
            "gastro_label_graph_mil",
            *CUSTOM_GASTRO_MODEL_NAMES,
            *GASTRO_BASELINE_MODEL_NAMES,
            *GASTRO_SOTA_MODEL_NAMES,
            *EXP1_MODEL_NAMES,
            *EXP2_MODEL_NAMES,
            *EXP4_MODEL_NAMES,
            *EXP6_MODEL_NAMES,
            *EXP8_MODEL_NAMES,
        )
    )
)

TRACKER_ALIAS_TO_META = {
    "best_macro_f1": {"metric_name": "macro_f1", "mode": "max"},
    "best_micro_f1": {"metric_name": "micro_f1", "mode": "max"},
    "best_val_loss": {"metric_name": "val_loss", "mode": "min"},
}
SERIES_TRACKER_ALIASES = tuple(TRACKER_ALIAS_TO_META.keys())
IMAGE_CACHE_SCOPE_VALUES = {"task", "shared"}
SHARED_IMAGE_CACHE_DIR_NAME = "shared"
DEFAULT_CLI_TASK_NAME = "task2"

AUTO_BASELINE_ALLOWED_MODEL_NAMES = tuple(
    name
    for name in SUPPORTED_MODEL_NAMES
    if name in GASTRO_BASELINE_CLASS_REGISTRY
)
AUTO_SOTA_ALLOWED_MODEL_NAMES = tuple(
    name
    for name in SUPPORTED_MODEL_NAMES
    if name in GASTRO_SOTA_CLASS_REGISTRY
)
AUTO_ABLATION_ALLOWED_MODEL_NAMES = ("gastro_label_graph_mil",)
AUTO_EXP1_ALLOWED_MODEL_NAMES = tuple(EXP1_MODEL_NAMES)
AUTO_EXP2_ALLOWED_MODEL_NAMES = tuple(EXP2_MODEL_NAMES)
AUTO_EXP3_ALLOWED_MODEL_NAMES = tuple(EXP1_MODEL_NAMES)
AUTO_EXP4_ALLOWED_MODEL_NAMES = ("gastro_label_graph_mil", *EXP4_MODEL_NAMES)
AUTO_EXP5_ALLOWED_MODEL_NAMES = ("roi_long_mil",)
AUTO_EXP6_ALLOWED_MODEL_NAMES = (
    "exp6_long_mil_64_no_roi",
    "exp6_roi_mix_64_32",
    "exp6_roi_mix_64_64",
    "exp6_roi_context_128_16",
    "exp6_roi_context_128_32",
    "exp6_roi_context_128_64",
    "exp6_roi_dual_128_16",
    "exp6_roi_dual_128_32",
    "exp6_roi_dual_128_64",
    "exp6_roi_filter_96_32",
    "exp6_roi_filter_128_32",
    "exp6_roi_cons_128_32",
)
AUTO_EXP7_ALLOWED_MODEL_NAMES = (
    "exp6_long_mil_64_no_roi",
    "exp6_roi_mix_64_16",
    "exp6_roi_mix_128_16",
    "exp6_roi_context_64_16",
    "exp6_roi_context_64_32",
    "exp6_roi_context_64_64",
    "exp6_long_mil_128_no_roi",
)
AUTO_EXP8_ALLOWED_MODEL_NAMES = (
    "exp8_mm_struct_late_gate",
    "exp8_mm_label_proto_graph",
    "exp8_mm_text_contrast_distill",
    "exp8_mm_watch_cross_attn",
    "exp8_mm_text_guided_top64_align",
)
AUTO_EXP8_MM_ABLATION_ALLOWED_MODEL_NAMES = (
    "exp8_mm_ablation_image_baseline",
    "exp8_mm_ablation_age",
    "exp8_mm_ablation_age_sex",
    "exp8_mm_ablation_age_sex_hp",
    "exp8_mm_ablation_reportTitle",
    "exp8_mm_ablation_operationValue",
    "exp8_mm_ablation_title_operation",
    "exp8_mm_ablation_all_structured",
    "exp8_mm_ablation_all_without_title",
    "exp8_mm_ablation_all_without_operation",
    "exp8_mm_ablation_all_without_hp",
    "exp8_mm_ablation_all_without_age",
    "exp8_mm_ablation_all_shuffle_title_test",
    "exp8_mm_ablation_all_shuffle_operation_test",
    "exp8_mm_ablation_all_shuffle_title_operation_test",
    "exp8_mm_ablation_shuffle_title_train",
    "exp8_mm_ablation_shuffle_operation_train",
)
AUTO_EXP9_ABLATION_INSTANCE_VALUES = (16, 32, 48, 64, 80, 96)
AUTO_EXP9_ABLATION_ALLOWED_MODEL_NAMES = tuple(
    [f"exp9_watch_instances_{instances}" for instances in AUTO_EXP9_ABLATION_INSTANCE_VALUES]
    + [f"exp9_watch_no_context_instances_{instances}" for instances in AUTO_EXP9_ABLATION_INSTANCE_VALUES]
    + [
        "exp9_watch_no_text",
        "exp9_watch_label_graph",
        "exp9_watch_no_cross_attn_pool_fusion",
        "exp9_watch_cross_attn_no_gate",
        "exp9_watch_cross_attn_no_image_aux",
    ]
)
AUTO_EXP11_MODULE_ABLATION_COMBINATIONS = (
    "none",
    "1",
    "2",
    "3",
    "4",
    "13",
    "14",
    "23",
    "24",
    "34",
)
AUTO_EXP11_MODULE_ABLATION_ALLOWED_MODEL_NAMES = tuple(
    f"exp11_module_ablation_{combo}" for combo in AUTO_EXP11_MODULE_ABLATION_COMBINATIONS
)

try:
    from tqdm import tqdm
except Exception:  # pragma: no cover
    tqdm = None

try:
    import optuna
except Exception:  # pragma: no cover
    optuna = None


class AutoExploreTrialFailed(RuntimeError):
    """自动探索单次 trial 失败，但不终止整个搜索流程。"""


class AutoBaselineRunFailed(RuntimeError):
    """自动 baseline 的单个模型训练失败，但不终止整个流程。"""


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)


def safe_float(value: Any) -> float:
    try:
        return float(value)
    except Exception:
        return float("nan")


def build_session_dir_name(prefix: str, suffix: str | None = None) -> str:
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    if suffix:
        return f"{prefix}_{suffix}_{timestamp}"
    return f"{prefix}_{timestamp}"


def format_param_overrides(overrides: dict[str, Any]) -> str:
    if not overrides:
        return "无覆盖参数"
    parts = [f"{key}={value}" for key, value in sorted(overrides.items())]
    return ", ".join(parts)


def format_metric_text(value: Any, digits: int = 4) -> str:
    numeric = safe_float(value)
    if np.isnan(numeric):
        return "nan"
    return f"{numeric:.{digits}f}"


def resolve_train_task_name(cli_task_name: str | None, train_cfg: dict[str, Any] | None = None) -> str:
    if cli_task_name and str(cli_task_name).strip():
        task_name = str(cli_task_name).strip()
    elif train_cfg is not None:
        task_name = str(train_cfg.get("task_name", DEFAULT_GASTRO_TASK_NAME)).strip() or DEFAULT_GASTRO_TASK_NAME
    else:
        task_name = DEFAULT_GASTRO_TASK_NAME
    return get_task_spec(task_name).name


def resolve_model_task_meta(model_name: str, selected_task_name: str | None = None) -> dict[str, Any]:
    if model_name in GASTRO_MODEL_NAMES:
        task_name = selected_task_name or DEFAULT_GASTRO_TASK_NAME
        task_spec = get_task_spec(task_name)
        if task_spec.family != "gastro":
            raise ValueError(f"模型 {model_name} 仅支持胃镜任务，当前收到任务: {task_name}")
    else:
        raise ValueError(f"未知模型名: {model_name}")

    return {
        "task_name": task_spec.name,
        "task_type": task_spec.task_type,
        "task_dir_name": task_spec.task_dir_name,
        "run_prefix": task_spec.run_prefix,
        "label_names": list(task_spec.label_names),
        "class_names": list(task_spec.class_names),
        "num_labels": task_spec.num_labels,
        "display_name": task_spec.display_name,
    }


def resolve_required_tasks(model_names: list[str], selected_task_name: str | None = None) -> set[str]:
    return {resolve_model_task_meta(model_name, selected_task_name)["task_name"] for model_name in model_names}


def resolve_series_entry_model_name(entry: dict[str, Any]) -> str:
    return str(entry.get("base_model_name", entry.get("name", ""))).strip()


def normalize_auto_ablations_selection(raw_value: Any) -> tuple[str, ...]:
    if raw_value is None:
        return ()
    if isinstance(raw_value, bool):
        return ("all",) if raw_value else ()
    if isinstance(raw_value, str):
        stripped = raw_value.strip()
        if not stripped or stripped.lower() in {"0", "false", "none", "off", "disable", "disabled", "no"}:
            return ()
        return tuple(item.strip() for item in stripped.split(",") if item.strip())
    if isinstance(raw_value, (list, tuple)):
        normalized = [str(item).strip() for item in raw_value if str(item).strip()]
        return tuple(normalized)
    if isinstance(raw_value, dict):
        if not bool(raw_value.get("enabled", False)):
            return ()
        target = raw_value.get("target", raw_value.get("selection", "all"))
        return normalize_auto_ablations_selection(target)
    raise ValueError("auto_ablations 仅支持 false、true、字符串、列表或字典配置")


def format_task_display_name(task_name: str) -> str:
    return get_task_spec(task_name).display_name


def build_training_config_summary_lines(
    run_cfg: dict[str, Any],
    *,
    image_size: int,
    num_workers: int,
) -> list[str]:
    return [
        (
            f"batch_size={run_cfg.get('batch_size')} "
            f"eval_batch_size={run_cfg.get('eval_batch_size')} "
            f"grad_accum_steps={run_cfg.get('grad_accum_steps')} "
            f"amp={bool(run_cfg.get('amp', True))}"
        ),
        (
            f"lr={run_cfg.get('lr')} "
            f"weight_decay={run_cfg.get('weight_decay')} "
            f"warmup_ratio={run_cfg.get('warmup_ratio')} "
            f"optimizer={run_cfg.get('optimizer_name')}"
        ),
        (
            f"train_max_instances={run_cfg.get('train_max_instances')} "
            f"eval_max_instances={run_cfg.get('eval_max_instances')} "
            f"train_max_batch_instances={run_cfg.get('train_max_batch_instances')} "
            f"eval_max_batch_instances={run_cfg.get('eval_max_batch_instances')}"
        ),
        (
            f"random_instance_dropout={run_cfg.get('random_instance_dropout')} "
            f"loss_name={run_cfg.get('loss_name')} "
            f"monitor={run_cfg.get('monitor_metric')}/{run_cfg.get('monitor_mode')}"
        ),
        (
            f"num_workers={num_workers} "
            f"image_size={image_size} "
            f"cache_mode={run_cfg.get('image_cache_mode')} "
            f"cache_scope={run_cfg.get('image_cache_scope', 'task')} "
            f"cache_warmup={bool(run_cfg.get('image_cache_warmup', False))}"
        ),
    ]


def next_run_index(task_dir: Path, run_prefix: str) -> int:
    pattern = re.compile(rf"^{re.escape(run_prefix)}_(\d+)(?:_para_auto)?$")
    max_index = 0
    if task_dir.is_dir():
        for child in task_dir.iterdir():
            if not child.is_dir():
                continue
            match = pattern.match(child.name)
            if match:
                max_index = max(max_index, int(match.group(1)))
    return max_index + 1


def parse_task_run_index(run_dir: Path, run_prefix: str, *, is_auto_explore: bool) -> int | None:
    suffix = r"_para_auto" if is_auto_explore else ""
    pattern = re.compile(rf"^{re.escape(run_prefix)}_(\d+){suffix}$")
    match = pattern.match(run_dir.name)
    if not match:
        return None
    return int(match.group(1))


def is_training_run_complete(run_dir: Path) -> bool:
    if not (run_dir / "test_result.csv").is_file():
        return False
    return all(
        (run_dir / result_dir / "metrics.json").is_file()
        for result_dir in ("test_macro_f1", "test_micro_f1", "test_val_loss")
    )


def training_run_matches_model(run_dir: Path, model_name: str) -> bool:
    config_path = run_dir / "config.yaml"
    if not config_path.is_file():
        return True
    try:
        payload = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    except Exception:
        return False
    if not isinstance(payload, dict):
        return False
    saved_model_name = str(payload.get("model_name", "")).strip()
    return saved_model_name == model_name


def find_resumable_task_run_dir(
    task_dir: Path,
    run_prefix: str,
    model_name: str,
    *,
    is_auto_explore: bool,
) -> tuple[Path, int, str] | None:
    candidates: list[tuple[int, Path]] = []
    if task_dir.is_dir():
        for child in task_dir.iterdir():
            if not child.is_dir():
                continue
            run_index = parse_task_run_index(child, run_prefix, is_auto_explore=is_auto_explore)
            if run_index is not None:
                candidates.append((run_index, child))

    for run_index, run_dir in sorted(candidates, key=lambda item: item[0], reverse=True):
        resume_path = run_dir / "checkpoints" / "last.ckpt"
        if (
            resume_path.is_file()
            and not is_training_run_complete(run_dir)
            and training_run_matches_model(run_dir, model_name)
        ):
            return run_dir, run_index, str(resume_path)
    return None


def allocate_task_run_dir(
    output_root: Path,
    train_cfg: dict[str, Any],
    model_name: str,
    *,
    is_auto_explore: bool,
) -> tuple[Path, dict[str, Any]]:
    selected_task_name = resolve_train_task_name(None, train_cfg)
    task_meta = resolve_model_task_meta(model_name, selected_task_name)
    task_dir = output_root / train_cfg["train_run_dir_name"] / task_meta["task_dir_name"]
    experiment_dir_name = train_cfg.get("experiment_dir_name", "")
    if experiment_dir_name:
        task_dir = task_dir / experiment_dir_name
    task_dir.mkdir(parents=True, exist_ok=True)

    single_run_dir_name = str(train_cfg.get("single_run_dir_name", "")).strip()
    if single_run_dir_name and not is_auto_explore:
        run_dir = task_dir / single_run_dir_name
        resume_path = run_dir / "checkpoints" / "last.ckpt"
        if (
            resume_path.is_file()
            and not is_training_run_complete(run_dir)
            and training_run_matches_model(run_dir, model_name)
        ):
            return run_dir, {
                **task_meta,
                "task_dir": str(task_dir),
                "run_index": -1,
                "is_auto_explore": is_auto_explore,
                "resume_path": str(resume_path),
                "run_action": "resume",
            }
        run_dir.mkdir(parents=True, exist_ok=True)
        return run_dir, {
            **task_meta,
            "task_dir": str(task_dir),
            "run_index": -1,
            "is_auto_explore": is_auto_explore,
            "resume_path": None,
            "run_action": "fixed",
        }

    resumable = None
    if not is_auto_explore:
        resumable = find_resumable_task_run_dir(
            task_dir,
            task_meta["run_prefix"],
            model_name,
            is_auto_explore=is_auto_explore,
        )
    if resumable is not None:
        run_dir, run_index, resume_path = resumable
        return run_dir, {
            **task_meta,
            "task_dir": str(task_dir),
            "run_index": run_index,
            "is_auto_explore": is_auto_explore,
            "resume_path": resume_path,
            "run_action": "resume",
        }

    run_index = next_run_index(task_dir, task_meta["run_prefix"])
    suffix = "_para_auto" if is_auto_explore else ""
    run_dir = task_dir / f"{task_meta['run_prefix']}_{run_index}{suffix}"
    run_dir.mkdir(parents=True, exist_ok=True)

    return run_dir, {
        **task_meta,
        "task_dir": str(task_dir),
        "run_index": run_index,
        "is_auto_explore": is_auto_explore,
        "resume_path": None,
        "run_action": "new",
    }


def auto_train_dir_name(trial_index: int) -> str:
    return f"train_{trial_index:03d}"


def next_auto_train_index(session_dir: Path) -> int:
    pattern = re.compile(r"^train_(\d+)(?:_.+)?$")
    max_index = 0
    if session_dir.is_dir():
        for child in session_dir.iterdir():
            if not child.is_dir():
                continue
            match = pattern.match(child.name)
            if match:
                max_index = max(max_index, int(match.group(1)))
    return max_index + 1


def auto_baseline_run_dir_name(run_index: int, model_name: str) -> str:
    return f"train_{run_index:03d}_{model_name}"


def normalize_auto_explore_space(raw_space: Any, allowed_keys: set[str]) -> dict[str, dict[str, Any]]:
    if raw_space is None:
        return {}
    if not isinstance(raw_space, dict):
        raise ValueError("auto_explore.search_space 格式错误，必须是字典")

    normalized: dict[str, dict[str, Any]] = {}
    for raw_name, raw_spec in raw_space.items():
        name = str(raw_name).strip()
        if name not in allowed_keys:
            raise ValueError(f"auto_explore.search_space 存在未知参数：{name}")
        if not isinstance(raw_spec, dict):
            raise ValueError(f"auto_explore.search_space.{name} 配置格式错误")

        spec_type = str(raw_spec.get("type", "choice")).strip().lower()
        enabled = bool(raw_spec.get("enabled", True))
        normalized_spec: dict[str, Any] = {"enabled": enabled, "type": spec_type}

        if spec_type == "choice":
            values = raw_spec.get("values")
            if not isinstance(values, list) or not values:
                raise ValueError(f"auto_explore.search_space.{name}.values 必须是非空列表")
            normalized_spec["values"] = values
        elif spec_type == "int":
            min_value = int(raw_spec.get("min"))
            max_value = int(raw_spec.get("max"))
            step = int(raw_spec.get("step", 1))
            if step <= 0:
                raise ValueError(f"auto_explore.search_space.{name}.step 必须大于 0")
            if min_value > max_value:
                raise ValueError(f"auto_explore.search_space.{name}.min 不能大于 max")
            normalized_spec.update({"min": min_value, "max": max_value, "step": step})
        elif spec_type in {"float", "log_float"}:
            min_value = float(raw_spec.get("min"))
            max_value = float(raw_spec.get("max"))
            if min_value > max_value:
                raise ValueError(f"auto_explore.search_space.{name}.min 不能大于 max")
            if spec_type == "log_float" and (min_value <= 0.0 or max_value <= 0.0):
                raise ValueError(f"auto_explore.search_space.{name} 使用 log_float 时，min/max 必须大于 0")
            normalized_spec.update({"min": min_value, "max": max_value})
            if "round" in raw_spec:
                round_digits = int(raw_spec.get("round"))
                if round_digits < 0:
                    raise ValueError(f"auto_explore.search_space.{name}.round 不能小于 0")
                normalized_spec["round"] = round_digits
        else:
            raise ValueError(
                f"auto_explore.search_space.{name}.type 仅支持 choice、int、float、log_float"
            )

        normalized[name] = normalized_spec

    return normalized


def normalize_stability_filter(raw_filter: Any, prefix: str) -> dict[str, Any]:
    if raw_filter is None:
        raw_filter = {}
    if not isinstance(raw_filter, dict):
        raise ValueError(f"{prefix} 配置格式错误")

    normalized = {
        "enabled": bool(raw_filter.get("enabled", True)),
        "min_epochs_trained": int(raw_filter.get("min_epochs_trained", 8)),
        "max_final_gap": float(raw_filter.get("max_final_gap", 0.02)),
        "max_val_loss_rebound_ratio": float(raw_filter.get("max_val_loss_rebound_ratio", 0.15)),
    }
    if normalized["min_epochs_trained"] < 1:
        raise ValueError(f"{prefix}.min_epochs_trained 必须大于等于 1")
    if normalized["max_final_gap"] < 0:
        raise ValueError(f"{prefix}.max_final_gap 不能小于 0")
    if normalized["max_val_loss_rebound_ratio"] < 0:
        raise ValueError(f"{prefix}.max_val_loss_rebound_ratio 不能小于 0")
    return normalized


def normalize_auto_explore_objective(raw_objective: Any, prefix: str) -> dict[str, Any]:
    if raw_objective is None:
        raw_objective = {}
    if not isinstance(raw_objective, dict):
        raise ValueError(f"{prefix} 配置格式错误")

    objective_name = str(raw_objective.get("name", "stable_tail_val_loss")).strip().lower()
    if objective_name != "stable_tail_val_loss":
        raise ValueError(f"{prefix}.name 当前仅支持 stable_tail_val_loss")

    normalized = {
        "name": objective_name,
        "mode": "min",
        "tail_epochs": int(raw_objective.get("tail_epochs", 5)),
        "tail_std_weight": float(raw_objective.get("tail_std_weight", 0.5)),
        "gap_penalty_weight": float(raw_objective.get("gap_penalty_weight", 2.0)),
        "rebound_penalty_weight": float(raw_objective.get("rebound_penalty_weight", 1.0)),
        "unstable_penalty": float(raw_objective.get("unstable_penalty", 0.1)),
    }
    if normalized["tail_epochs"] < 1:
        raise ValueError(f"{prefix}.tail_epochs 必须大于等于 1")
    for key in (
        "tail_std_weight",
        "gap_penalty_weight",
        "rebound_penalty_weight",
        "unstable_penalty",
    ):
        if normalized[key] < 0:
            raise ValueError(f"{prefix}.{key} 不能小于 0")
    return normalized


def normalize_optuna_settings(raw_optuna: Any, prefix: str) -> dict[str, Any]:
    if raw_optuna is None:
        raw_optuna = {}
    if not isinstance(raw_optuna, dict):
        raise ValueError(f"{prefix} 配置格式错误")

    pruner_raw = raw_optuna.get("pruner", {})
    if pruner_raw is None:
        pruner_raw = {}
    if not isinstance(pruner_raw, dict):
        raise ValueError(f"{prefix}.pruner 配置格式错误")

    sampler_name = str(raw_optuna.get("sampler", "tpe")).strip().lower()
    if sampler_name != "tpe":
        raise ValueError(f"{prefix}.sampler 当前仅支持 tpe")

    pruner_name = str(pruner_raw.get("name", "median")).strip().lower()
    if pruner_name not in {"median", "none"}:
        raise ValueError(f"{prefix}.pruner.name 仅支持 median 或 none")

    normalized = {
        "sampler": sampler_name,
        "n_startup_trials": int(raw_optuna.get("n_startup_trials", 8)),
        "multivariate": bool(raw_optuna.get("multivariate", False)),
        "group": bool(raw_optuna.get("group", False)),
        "pruner": {
            "name": pruner_name,
            "n_startup_trials": int(pruner_raw.get("n_startup_trials", 8)),
            "n_warmup_steps": int(pruner_raw.get("n_warmup_steps", 8)),
            "interval_steps": int(pruner_raw.get("interval_steps", 1)),
        },
    }
    if normalized["n_startup_trials"] < 0:
        raise ValueError(f"{prefix}.n_startup_trials 不能小于 0")
    if normalized["group"] and not normalized["multivariate"]:
        raise ValueError(f"{prefix}.group=true 时，必须同时设置 multivariate=true")
    for key in ("n_startup_trials", "n_warmup_steps", "interval_steps"):
        if normalized["pruner"][key] < 0:
            raise ValueError(f"{prefix}.pruner.{key} 不能小于 0")
    return normalized


def sample_search_value(spec: dict[str, Any], rng: random.Random) -> Any:
    spec_type = str(spec["type"])
    if spec_type == "choice":
        return rng.choice(list(spec["values"]))
    if spec_type == "int":
        min_value = int(spec["min"])
        max_value = int(spec["max"])
        step = int(spec["step"])
        num_steps = ((max_value - min_value) // step) + 1
        return min_value + step * rng.randrange(num_steps)
    if spec_type == "float":
        value = rng.uniform(float(spec["min"]), float(spec["max"]))
    elif spec_type == "log_float":
        log_min = math.log(float(spec["min"]))
        log_max = math.log(float(spec["max"]))
        value = math.exp(rng.uniform(log_min, log_max))
    else:
        raise ValueError(f"未知自动探索采样类型：{spec_type}")

    if "round" in spec:
        value = round(value, int(spec["round"]))
    return value


def sample_auto_explore_overrides(auto_explore_cfg: dict[str, Any], rng: random.Random) -> dict[str, Any]:
    overrides: dict[str, Any] = {}
    for param_name, spec in auto_explore_cfg["search_space"].items():
        if not bool(spec.get("enabled", False)):
            continue
        overrides[param_name] = sample_search_value(spec, rng)
    return overrides


def suggest_search_value_with_optuna(
    trial: Any,
    *,
    param_name: str,
    spec: dict[str, Any],
) -> Any:
    spec_type = str(spec["type"])
    if spec_type == "choice":
        return trial.suggest_categorical(param_name, list(spec["values"]))
    if spec_type == "int":
        return trial.suggest_int(
            param_name,
            int(spec["min"]),
            int(spec["max"]),
            step=int(spec.get("step", 1)),
        )
    if spec_type in {"float", "log_float"}:
        value = trial.suggest_float(
            param_name,
            float(spec["min"]),
            float(spec["max"]),
            log=(spec_type == "log_float"),
        )
        if "round" in spec:
            value = round(float(value), int(spec["round"]))
        return value
    raise ValueError(f"未知自动探索采样类型：{spec_type}")


def suggest_auto_explore_overrides_with_optuna(
    trial: Any,
    auto_explore_cfg: dict[str, Any],
) -> dict[str, Any]:
    overrides: dict[str, Any] = {}
    for param_name, spec in auto_explore_cfg["search_space"].items():
        if not bool(spec.get("enabled", False)):
            continue
        overrides[param_name] = suggest_search_value_with_optuna(
            trial,
            param_name=param_name,
            spec=spec,
        )
    return overrides


def iter_trial_progress(total: int, desc: str):
    indices = range(1, total + 1)
    if tqdm is not None:
        return tqdm(indices, desc=desc, dynamic_ncols=True)
    return indices


def build_optuna_sampler(optuna_cfg: dict[str, Any], seed: int) -> Any:
    if optuna is None:
        raise ModuleNotFoundError("未安装 optuna，请先执行 `pip install optuna` 后再运行自动探索。")
    return optuna.samplers.TPESampler(
        seed=seed,
        n_startup_trials=int(optuna_cfg.get("n_startup_trials", 8)),
        multivariate=bool(optuna_cfg.get("multivariate", True)),
        group=bool(optuna_cfg.get("group", False)),
    )


def build_optuna_pruner(optuna_cfg: dict[str, Any]) -> Any:
    if optuna is None:
        raise ModuleNotFoundError("未安装 optuna，请先执行 `pip install optuna` 后再运行自动探索。")

    pruner_cfg = optuna_cfg.get("pruner", {})
    pruner_name = str(pruner_cfg.get("name", "median")).strip().lower()
    if pruner_name == "none":
        return optuna.pruners.NopPruner()
    return optuna.pruners.MedianPruner(
        n_startup_trials=int(pruner_cfg.get("n_startup_trials", 8)),
        n_warmup_steps=int(pruner_cfg.get("n_warmup_steps", 8)),
        interval_steps=max(1, int(pruner_cfg.get("interval_steps", 1))),
    )


def optuna_direction_from_mode(mode: str) -> str:
    normalized_mode = str(mode).strip().lower()
    if normalized_mode == "min":
        return "minimize"
    if normalized_mode == "max":
        return "maximize"
    raise ValueError(f"未知优化方向：{mode}")


def resolve_session_candidate(
    all_results: dict[str, Any],
    *,
    remark_metric_alias: str,
    result_source: str,
    fallback_metric_name: str,
) -> list[dict[str, Any]]:
    candidates: list[dict[str, Any]] = []

    for model_name, payload in all_results.get("models", {}).items():
        if not isinstance(payload, dict):
            continue

        if result_source == "test_results":
            result_group = payload.get("test_results", {})
            candidate = result_group.get(remark_metric_alias, {}) if isinstance(result_group, dict) else {}
            metric_name = fallback_metric_name
            score = candidate.get(metric_name, float("nan"))
            if metric_name not in candidate:
                metrics = candidate.get("metrics", {}) if isinstance(candidate, dict) else {}
                if isinstance(metrics, dict):
                    score = metrics.get(metric_name, float("nan"))
            candidate_payload = {
                "score": score,
                "score_source": result_source,
                "metric_name": metric_name,
                "best_epoch": candidate.get("best_epoch", -1) if isinstance(candidate, dict) else -1,
                "checkpoint_path": candidate.get("checkpoint_path", "") if isinstance(candidate, dict) else "",
                "macro_f1": candidate.get("metrics", {}).get("macro_f1", float("nan"))
                if isinstance(candidate, dict) and isinstance(candidate.get("metrics"), dict)
                else float("nan"),
                "micro_f1": candidate.get("metrics", {}).get("micro_f1", float("nan"))
                if isinstance(candidate, dict) and isinstance(candidate.get("metrics"), dict)
                else float("nan"),
                "test_loss": candidate.get("test_loss", float("nan")) if isinstance(candidate, dict) else float("nan"),
            }
        elif result_source == "best_checkpoints":
            result_group = payload.get("best_checkpoints", {})
            candidate = result_group.get(remark_metric_alias, {}) if isinstance(result_group, dict) else {}
            metric_name = str(candidate.get("metric_name", fallback_metric_name)) if isinstance(candidate, dict) else fallback_metric_name
            candidate_payload = {
                "score": candidate.get("best_value", float("nan")) if isinstance(candidate, dict) else float("nan"),
                "score_source": result_source,
                "metric_name": metric_name,
                "best_epoch": candidate.get("best_epoch", -1) if isinstance(candidate, dict) else -1,
                "checkpoint_path": candidate.get("checkpoint_path", "") if isinstance(candidate, dict) else "",
                "macro_f1": float("nan"),
                "micro_f1": float("nan"),
                "test_loss": float("nan"),
            }
        else:
            raise ValueError(f"未知结果来源：{result_source}")

        try:
            score_value = float(candidate_payload["score"])
        except Exception:
            score_value = float("nan")

        candidates.append(
            {
                "train_dir": payload.get("train_dir_name", ""),
                "train_dir_path": payload.get("train_dir", ""),
                "model_name": model_name,
                "selection_checkpoint": remark_metric_alias,
                "selection_metric": candidate_payload["metric_name"],
                "score": score_value,
                "score_source": candidate_payload["score_source"],
                "best_epoch": int(candidate_payload["best_epoch"]),
                "checkpoint_path": candidate_payload["checkpoint_path"],
                "macro_f1": candidate_payload["macro_f1"],
                "micro_f1": candidate_payload["micro_f1"],
                "test_loss": candidate_payload["test_loss"],
            }
        )

    return candidates


def select_best_candidate(
    candidates: list[dict[str, Any]],
    mode: str,
    score_key: str = "score",
) -> dict[str, Any] | None:
    valid_candidates = [item for item in candidates if not np.isnan(float(item[score_key]))]
    if not valid_candidates:
        return None
    if mode == "max":
        return max(valid_candidates, key=lambda item: float(item[score_key]))
    return min(valid_candidates, key=lambda item: float(item[score_key]))


def analyze_training_log(
    log_path: Path,
    stability_filter: dict[str, Any],
    objective_cfg: dict[str, Any] | None = None,
) -> dict[str, Any]:
    analysis = {
        "log_path": str(log_path),
        "epochs_trained": 0,
        "best_val_loss": float("nan"),
        "best_val_epoch": -1,
        "final_train_loss": float("nan"),
        "final_val_loss": float("nan"),
        "final_gap": float("nan"),
        "val_loss_rebound": float("nan"),
        "val_loss_rebound_ratio": float("nan"),
        "final_train_macro_f1": float("nan"),
        "final_val_macro_f1": float("nan"),
        "tail_epochs": 0,
        "tail_val_loss_mean": float("nan"),
        "tail_val_loss_std": float("nan"),
        "objective_score": float("nan"),
        "objective_name": str(objective_cfg.get("name", "")) if isinstance(objective_cfg, dict) else "",
        "objective_mode": str(objective_cfg.get("mode", "")) if isinstance(objective_cfg, dict) else "",
        "objective_gap_penalty": float("nan"),
        "objective_rebound_penalty": float("nan"),
        "objective_unstable_penalty": float("nan"),
        "stable_convergence": False,
    }
    if not log_path.is_file():
        return analysis

    split_rows: dict[str, list[dict[str, Any]]] = {"train": [], "val": []}
    with log_path.open("r", encoding="utf-8", newline="") as file:
        reader = csv.DictReader(file)
        for row in reader:
            split = str(row.get("split", "")).strip()
            if split not in split_rows:
                continue
            row_copy = dict(row)
            try:
                row_copy["_epoch"] = int(row_copy.get("epoch", 0))
            except Exception:
                row_copy["_epoch"] = 0
            split_rows[split].append(row_copy)

    for split in split_rows:
        split_rows[split].sort(key=lambda item: int(item["_epoch"]))

    train_rows = split_rows["train"]
    val_rows = split_rows["val"]
    if not train_rows or not val_rows:
        return analysis

    final_train = train_rows[-1]
    final_val = val_rows[-1]
    best_val = min(val_rows, key=lambda item: safe_float(item.get("loss", float("nan"))))

    best_val_loss = safe_float(best_val.get("loss", float("nan")))
    final_train_loss = safe_float(final_train.get("loss", float("nan")))
    final_val_loss = safe_float(final_val.get("loss", float("nan")))
    val_loss_rebound = final_val_loss - best_val_loss
    rebound_ratio = (
        val_loss_rebound / max(best_val_loss, 1e-8)
        if not np.isnan(best_val_loss) and not np.isnan(val_loss_rebound)
        else float("nan")
    )
    final_gap = final_val_loss - final_train_loss

    analysis.update(
        {
            "epochs_trained": int(final_val["_epoch"]),
            "best_val_loss": best_val_loss,
            "best_val_epoch": int(best_val["_epoch"]),
            "final_train_loss": final_train_loss,
            "final_val_loss": final_val_loss,
            "final_gap": final_gap,
            "val_loss_rebound": val_loss_rebound,
            "val_loss_rebound_ratio": rebound_ratio,
            "final_train_macro_f1": safe_float(final_train.get("macro_f1", float("nan"))),
            "final_val_macro_f1": safe_float(final_val.get("macro_f1", float("nan"))),
        }
    )

    tail_epochs = min(
        len(val_rows),
        max(1, int(objective_cfg.get("tail_epochs", 5))) if isinstance(objective_cfg, dict) else 5,
    )
    tail_val_losses = [
        safe_float(item.get("loss", float("nan")))
        for item in val_rows[-tail_epochs:]
    ]
    valid_tail_val_losses = [value for value in tail_val_losses if not np.isnan(value)]
    if valid_tail_val_losses:
        analysis["tail_epochs"] = len(valid_tail_val_losses)
        analysis["tail_val_loss_mean"] = float(np.mean(valid_tail_val_losses))
        analysis["tail_val_loss_std"] = float(np.std(valid_tail_val_losses))

    if bool(stability_filter.get("enabled", True)):
        min_epochs_trained = int(stability_filter.get("min_epochs_trained", 8))
        max_final_gap = float(stability_filter.get("max_final_gap", 0.02))
        max_rebound_ratio = float(stability_filter.get("max_val_loss_rebound_ratio", 0.15))
        stable_convergence = (
            analysis["epochs_trained"] >= min_epochs_trained
            and not np.isnan(final_gap)
            and final_gap <= max_final_gap
            and not np.isnan(rebound_ratio)
            and rebound_ratio <= max_rebound_ratio
        )
        analysis["stable_convergence"] = bool(stable_convergence)

    if isinstance(objective_cfg, dict):
        tail_mean = safe_float(analysis.get("tail_val_loss_mean", float("nan")))
        tail_std = safe_float(analysis.get("tail_val_loss_std", float("nan")))
        gap_threshold = float(stability_filter.get("max_final_gap", 0.02))
        rebound_threshold = float(stability_filter.get("max_val_loss_rebound_ratio", 0.15))
        gap_excess = max(0.0, final_gap - gap_threshold) if not np.isnan(final_gap) else float("inf")
        rebound_excess = (
            max(0.0, rebound_ratio - rebound_threshold)
            if not np.isnan(rebound_ratio)
            else float("inf")
        )
        unstable_penalty = (
            float(objective_cfg.get("unstable_penalty", 0.1))
            if not bool(analysis.get("stable_convergence", False))
            else 0.0
        )

        if np.isnan(tail_mean):
            objective_score = float("inf")
        else:
            objective_score = tail_mean
            objective_score += float(objective_cfg.get("tail_std_weight", 0.5)) * max(0.0, tail_std)
            objective_score += float(objective_cfg.get("gap_penalty_weight", 2.0)) * gap_excess
            objective_score += float(objective_cfg.get("rebound_penalty_weight", 1.0)) * rebound_excess
            objective_score += unstable_penalty

        analysis.update(
            {
                "objective_score": float(objective_score),
                "objective_name": str(objective_cfg.get("name", "")),
                "objective_mode": str(objective_cfg.get("mode", "min")),
                "objective_gap_penalty": float(gap_excess) if not np.isinf(gap_excess) else float("inf"),
                "objective_rebound_penalty": float(rebound_excess)
                if not np.isinf(rebound_excess)
                else float("inf"),
                "objective_unstable_penalty": float(unstable_penalty),
            }
        )

    return analysis


def summarize_model_evaluation(log_analysis: dict[str, Any], stability_filter: dict[str, Any]) -> str:
    if int(log_analysis.get("epochs_trained", 0)) <= 0:
        return "缺少训练日志，无法评价"
    if bool(log_analysis.get("stable_convergence", False)):
        return "稳定收敛候选"

    reasons: list[str] = []
    min_epochs_trained = int(stability_filter.get("min_epochs_trained", 8))
    max_final_gap = float(stability_filter.get("max_final_gap", 0.02))
    max_rebound_ratio = float(stability_filter.get("max_val_loss_rebound_ratio", 0.15))

    if int(log_analysis.get("epochs_trained", 0)) < min_epochs_trained:
        reasons.append(
            f"训练轮数不足（{int(log_analysis.get('epochs_trained', 0))} < {min_epochs_trained}）"
        )
    final_gap = safe_float(log_analysis.get("final_gap", float("nan")))
    if not np.isnan(final_gap) and final_gap > max_final_gap:
        reasons.append("train/val loss 差距偏大")
    rebound_ratio = safe_float(log_analysis.get("val_loss_rebound_ratio", float("nan")))
    if not np.isnan(rebound_ratio) and rebound_ratio > max_rebound_ratio:
        reasons.append("验证损失后期反弹")

    if reasons:
        return "；".join(reasons)
    return "结果可用，但稳定性一般"


def build_model_evaluations(
    all_results: dict[str, Any],
    *,
    remark_metric_alias: str,
    remark_metric_name: str,
    result_source: str,
    stability_filter: dict[str, Any],
) -> list[dict[str, Any]]:
    candidates = resolve_session_candidate(
        all_results,
        remark_metric_alias=remark_metric_alias,
        result_source=result_source,
        fallback_metric_name=remark_metric_name,
    )
    evaluations: list[dict[str, Any]] = []

    for candidate in candidates:
        model_name = str(candidate["model_name"])
        payload = all_results.get("models", {}).get(model_name, {})
        log_analysis = analyze_training_log(
            Path(str(candidate["train_dir_path"])) / "log.csv",
            stability_filter,
        )
        evaluation_item = dict(candidate)
        evaluation_item.update(
            {
                "primary_monitor_metric": payload.get("primary_monitor_metric", ""),
                "primary_monitor_mode": payload.get("primary_monitor_mode", ""),
            }
        )
        evaluation_item.update(log_analysis)
        evaluation_item["evaluation"] = summarize_model_evaluation(log_analysis, stability_filter)
        evaluations.append(evaluation_item)

    return evaluations


def _extract_trial_params(trial_record: dict[str, Any]) -> dict[str, Any]:
    return {
        key.replace("param_", "", 1): value
        for key, value in trial_record.items()
        if key.startswith("param_")
    }


def _rank_auto_explore_records(
    trial_records: list[dict[str, Any]],
    selection_mode: str,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    successful_records = [item for item in trial_records if item.get("status") == "success"]
    pruned_records = [item for item in trial_records if item.get("status") == "pruned"]
    failed_records = [item for item in trial_records if item.get("status") == "failed"]

    if selection_mode == "max":
        successful_records = sorted(successful_records, key=lambda item: float(item["objective_score"]), reverse=True)
    else:
        successful_records = sorted(successful_records, key=lambda item: float(item["objective_score"]))

    stable_records = [item for item in successful_records if bool(item.get("stable_convergence", False))]
    return successful_records, pruned_records, failed_records, stable_records


def _summarize_test_results_for_text(test_results: dict[str, Any]) -> str:
    if not isinstance(test_results, dict) or not test_results:
        return "未记录测试结果"

    parts: list[str] = []
    for alias in ("best_macro_f1", "best_micro_f1", "best_val_loss"):
        payload = test_results.get(alias, {})
        if not isinstance(payload, dict):
            continue
        metrics = payload.get("metrics", {}) if isinstance(payload.get("metrics"), dict) else {}
        parts.append(
            f"{alias}: "
            f"macro_f1={format_metric_text(metrics.get('macro_f1'))}, "
            f"micro_f1={format_metric_text(metrics.get('micro_f1'))}, "
            f"test_loss={format_metric_text(payload.get('test_loss'))}"
        )
    return "；".join(parts) if parts else "未记录测试结果"


def write_auto_explore_notes(
    session_dir: Path,
    trial_records: list[dict[str, Any]],
    auto_explore_cfg: dict[str, Any],
    run_context: dict[str, Any],
) -> None:
    notes_path = session_dir / "notes.json"
    successful_records, pruned_records, failed_records, stable_records = _rank_auto_explore_records(
        trial_records,
        auto_explore_cfg["objective"]["mode"],
    )
    ordered_records = successful_records + pruned_records + failed_records
    best_record = successful_records[0] if successful_records else None
    best_stable_record = stable_records[0] if stable_records else None
    recommended_record = best_stable_record or best_record

    notes_payload = {
        "run_dir": str(session_dir),
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "task_name": run_context.get("task_name", ""),
        "task_dir_name": run_context.get("task_dir_name", ""),
        "run_index": run_context.get("run_index", -1),
        "model_name": run_context.get("model_name", ""),
        "config_path": auto_explore_cfg.get("config_path", ""),
        "goal": auto_explore_cfg.get("goal", ""),
        "selection": {
            "checkpoint_alias": auto_explore_cfg["selection_alias"],
            "metric_name": auto_explore_cfg["selection_metric_name"],
            "mode": auto_explore_cfg["selection_mode"],
            "result_source": auto_explore_cfg["result_source"],
        },
        "objective": auto_explore_cfg["objective"],
        "counts": {
            "configured_trials": int(auto_explore_cfg["num_trials"]),
            "completed_trials": len(trial_records),
            "successful_trials": len(successful_records),
            "pruned_trials": len(pruned_records),
            "failed_trials": len(failed_records),
            "stable_trials": len(stable_records),
        },
        "enabled_search_space": {
            key: value
            for key, value in auto_explore_cfg.get("search_space", {}).items()
            if bool(value.get("enabled", False))
        },
        "stability_filter": auto_explore_cfg.get("stability_filter", {}),
        "best_trial": {
            **(best_record or {}),
            "params": _extract_trial_params(best_record or {}),
        },
        "best_stable_trial": {
            **(best_stable_record or {}),
            "params": _extract_trial_params(best_stable_record or {}),
        },
        "stable_trials": [
            {
                **item,
                "params": _extract_trial_params(item),
            }
            for item in stable_records
        ],
        "recommended_trial": {
            **(recommended_record or {}),
            "params": _extract_trial_params(recommended_record or {}),
            "source": "best_stable_trial" if best_stable_record else "best_trial",
        },
        "trial_records": [
            {
                **item,
                "params": _extract_trial_params(item),
            }
            for item in ordered_records
        ],
    }

    notes_path.write_text(
        json.dumps(to_builtin_type(notes_payload), ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def write_auto_explore_remark(
    session_dir: Path,
    trial_records: list[dict[str, Any]],
    auto_explore_cfg: dict[str, Any],
    run_context: dict[str, Any],
) -> None:
    successful_records, pruned_records, failed_records, stable_records = _rank_auto_explore_records(
        trial_records,
        auto_explore_cfg["objective"]["mode"],
    )
    best_trial = successful_records[0] if successful_records else None
    best_stable_trial = stable_records[0] if stable_records else None
    recommended_trial = best_stable_trial or best_trial

    lines = [
        "自动探索结果摘要",
        f"运行目录: {session_dir}",
        f"任务: {run_context.get('task_name', '')}",
        f"模型: {run_context.get('model_name', '')}",
        (
            f"试验排序目标: {auto_explore_cfg['objective']['name']} / "
            f"{auto_explore_cfg['objective']['mode']}"
        ),
        (
            f"选择依据: {auto_explore_cfg['selection_alias']} / "
            f"{auto_explore_cfg['selection_metric_name']} / "
            f"{auto_explore_cfg['selection_mode']}"
        ),
        (
            f"完成情况: 共 {int(auto_explore_cfg['num_trials'])} 次，"
            f"成功 {len(successful_records)} 次，剪枝 {len(pruned_records)} 次，失败 {len(failed_records)} 次，"
            f"稳定候选 {len(stable_records)} 个"
        ),
    ]
    if auto_explore_cfg.get("goal"):
        lines.append(f"目标: {auto_explore_cfg['goal']}")

    if recommended_trial:
        lines.append(f"推荐训练目录: {recommended_trial.get('train_dir', '')}")
        lines.append(
            "推荐理由: "
            f"{auto_explore_cfg['objective']['name']}={format_metric_text(recommended_trial.get('objective_score'))}，"
            f"{auto_explore_cfg['selection_metric_name']}={format_metric_text(recommended_trial.get('best_score'))}，"
            f"best_epoch={recommended_trial.get('best_epoch', -1)}，"
            f"评价={recommended_trial.get('evaluation', '未评价')}"
        )
        lines.append(
            "推荐目录测试结果: "
            f"{_summarize_test_results_for_text(recommended_trial.get('test_results', {}))}"
        )
        best_params_text = format_param_overrides(_extract_trial_params(recommended_trial))
        lines.append(f"推荐目录参数: {best_params_text}")

    if stable_records:
        lines.append(
            "稳定候选: "
            + "，".join(item.get("train_dir", "") for item in stable_records[: min(5, len(stable_records))])
        )

    if successful_records:
        lines.append("训练目录概览:")
        for item in successful_records:
            params_text = format_param_overrides(_extract_trial_params(item))
            lines.append(
                f"- {item.get('train_dir', '')} | "
                f"{auto_explore_cfg['objective']['name']}={format_metric_text(item.get('objective_score'))} | "
                f"{auto_explore_cfg['selection_metric_name']}={format_metric_text(item.get('best_score'))} | "
                f"best_epoch={item.get('best_epoch', -1)} | "
                f"{item.get('evaluation', '未评价')} | "
                f"参数: {params_text}"
            )

    if pruned_records:
        lines.append("已剪枝训练目录:")
        for item in pruned_records:
            lines.append(
                f"- {item.get('train_dir', '')} | 已训练 {item.get('epochs_trained', 0)} 个 epoch | 原因: {item.get('error_message', '被 Optuna 剪枝')}"
            )

    if failed_records:
        lines.append("失败训练目录:")
        for item in failed_records:
            lines.append(
                f"- {item.get('train_dir', '')} | 错误: {item.get('error_message', '未知错误')}"
            )

    (session_dir / "remark.txt").write_text("\n".join(lines).strip() + "\n", encoding="utf-8")


LEGACY_CONFIG_CANDIDATES = {
    "path.yaml": ("configs/base/path.yaml", "configs/path.yaml"),
    "train.yaml": ("configs/base/train.yaml", "configs/train.yaml"),
    "model.yaml": ("configs/model.yaml",),
    "auto_explore.yaml": ("configs/auto_explore.yaml",),
    "auto_baselines.yaml": ("configs/auto_baselines.yaml",),
    "auto_sotas.yaml": ("configs/auto_sotas.yaml",),
    "auto_ablations.yaml": ("configs/auto_ablations.yaml",),
    "auto_distinct.yaml": ("configs/auto_distinct.yaml",),
    "auto_5fold.yaml": ("configs/auto_5fold.yaml",),
}


def resolve_default_config_path(task_name: str, filename: str) -> str:
    spec = get_task_spec(task_name)
    candidates = (f"configs/{spec.name}/{filename}", *LEGACY_CONFIG_CANDIDATES.get(filename, ()))
    for candidate in candidates:
        if Path(candidate).is_file():
            return candidate
    return candidates[0]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="训练 TASK1/TASK2 模型")
    parser.add_argument("--config", type=str, default="", help="路径配置文件，默认读取 configs/<task>/path.yaml")
    parser.add_argument("--train-config", type=str, default="", help="训练配置文件，默认读取 configs/<task>/train.yaml")
    parser.add_argument("--model-config", type=str, default="", help="模型配置文件，默认读取 configs/<task>/model.yaml")
    parser.add_argument(
        "--auto-explore-config",
        type=str,
        default="",
        help="自动探索配置文件，默认读取 configs/<task>/auto_explore.yaml",
    )
    parser.add_argument(
        "--auto-baselines-config",
        type=str,
        default="",
        help="自动 baseline 配置文件，默认读取 configs/<task>/auto_baselines.yaml",
    )
    parser.add_argument(
        "--auto-sotas-config",
        type=str,
        default="",
        help="自动 SOTA 配置文件，默认读取 configs/<task>/auto_sotas.yaml",
    )
    parser.add_argument(
        "--auto-ablations-config",
        type=str,
        default="",
        help="自动消融实验配置文件，默认读取 configs/<task>/auto_ablations.yaml",
    )
    parser.add_argument(
        "--auto-distinct-config",
        type=str,
        default="",
        help="TASK1 表2/表3/表4 CI 与显著性实验配置文件，默认读取 configs/<task>/auto_distinct.yaml",
    )
    parser.add_argument(
        "--auto-5fold-config",
        type=str,
        default="",
        help="TASK1 表2/表3/表4 5-fold 实验配置文件，默认读取 configs/<task>/auto_5fold.yaml",
    )
    parser.add_argument("--models", type=str, default="", help="仅运行指定模型，使用逗号分隔")
    parser.add_argument("--task", type=str, default="", help="显式指定任务，如 task1 / task2")
    parser.add_argument("--seed", type=int, default=None, help="覆盖 train.yaml 中的 seed")
    parser.add_argument("--epochs", type=int, default=None, help="覆盖 train.yaml 中的 max_epochs")
    parser.add_argument("--patience", type=int, default=None, help="覆盖 train.yaml 中的 patience")
    parser.add_argument("--image-size", type=int, default=None, help="覆盖 train.yaml 中的 image_size")
    parser.add_argument("--num-workers", type=int, default=None, help="覆盖 train.yaml 中的 num_workers")
    parser.add_argument("--max-exams-per-task", type=int, default=None, help="覆盖 train.yaml 中的 max_exams_per_task")
    parser.add_argument("--no-pretrained", action="store_true", help="禁用 ImageNet 预训练")
    parser.add_argument("--disable-multi-gpu", action="store_true", help="禁用 DataParallel")
    return parser.parse_args()


def resolve_config_path(raw_path: Any, config_path: Path) -> str:
    path = Path(str(raw_path)).expanduser()
    if not path.is_absolute():
        path = (config_path.parent / path).resolve()
    else:
        path = path.resolve()
    return str(path)


def load_path_config(config_path: Path) -> dict[str, str]:
    if not config_path.is_file():
        raise FileNotFoundError(f"未找到配置文件: {config_path}")

    payload = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict) or "paths" not in payload:
        raise ValueError("路径配置文件缺少 paths 字段")

    paths = payload["paths"]
    if not isinstance(paths, dict):
        raise ValueError("paths 字段格式错误")

    required = ["output_dir"]
    for key in required:
        if key not in paths:
            raise ValueError(f"paths 缺少字段: {key}")

    resolved = {
        "output_dir": resolve_config_path(paths["output_dir"], config_path),
    }
    if "dataset_base_root" in paths and str(paths["dataset_base_root"]).strip():
        resolved["dataset_base_root"] = resolve_config_path(paths["dataset_base_root"], config_path)
    if "dataset_root" in paths and str(paths["dataset_root"]).strip():
        resolved["dataset_root"] = resolve_config_path(paths["dataset_root"], config_path)
    if "valid_dicts_report_csv" in paths and str(paths["valid_dicts_report_csv"]).strip():
        resolved["valid_dicts_report_csv"] = resolve_config_path(paths["valid_dicts_report_csv"], config_path)
    return resolved


def _load_ratio(payload: dict[str, Any], key: str) -> tuple[float, float, float]:
    raw = payload.get(key, [0.6, 0.2, 0.2])
    if not isinstance(raw, list) or len(raw) != 3:
        raise ValueError(f"{key} 必须是长度为 3 的列表")
    ratios = tuple(float(item) for item in raw)
    if abs(sum(ratios) - 1.0) > 1e-8:
        raise ValueError(f"{key} 的和必须为 1")
    return ratios


def _normalize_name_list(raw_value: Any, field_name: str) -> list[str]:
    if raw_value is None or raw_value is False:
        return []
    if isinstance(raw_value, str):
        return [item.strip() for item in raw_value.split(",") if item.strip()]
    if isinstance(raw_value, list):
        return [str(item).strip() for item in raw_value if str(item).strip()]
    raise ValueError(f"{field_name} 必须是字符串或列表")


def normalize_class_balance_config(raw_value: Any) -> dict[str, Any]:
    if raw_value is None or raw_value is False:
        return {"enabled": False}
    if raw_value is True:
        raw_value = {"enabled": True}
    if not isinstance(raw_value, dict):
        raise ValueError("class_balance 必须是布尔值或字典")

    mode = str(raw_value.get("mode", "multilabel_minority_oversample")).strip().lower()
    if mode not in {"multilabel_minority_oversample"}:
        raise ValueError("class_balance.mode 当前仅支持 multilabel_minority_oversample")

    target_strategy = str(raw_value.get("target_strategy", "per_label_majority")).strip().lower()
    if target_strategy not in {"per_label_majority"}:
        raise ValueError("class_balance.target_strategy 当前仅支持 per_label_majority")

    apply_to = str(raw_value.get("apply_to", "train_only")).strip().lower()
    if apply_to != "train_only":
        raise ValueError("class_balance.apply_to 当前仅支持 train_only")

    top_candidate_pool = int(raw_value.get("top_candidate_pool", 32))
    if top_candidate_pool <= 0:
        raise ValueError("class_balance.top_candidate_pool 必须大于 0")
    candidate_sample_size = int(raw_value.get("candidate_sample_size", 256))
    if candidate_sample_size <= 0:
        raise ValueError("class_balance.candidate_sample_size 必须大于 0")

    allow_overshoot_ratio = float(raw_value.get("allow_overshoot_ratio", 0.05))
    if allow_overshoot_ratio < 0:
        raise ValueError("class_balance.allow_overshoot_ratio 不能小于 0")

    max_repeat_per_bag = int(raw_value.get("max_repeat_per_bag", 30))
    max_added_records = int(raw_value.get("max_added_records", 0))

    return {
        "enabled": bool(raw_value.get("enabled", False)),
        "mode": mode,
        "target_strategy": target_strategy,
        "apply_to": apply_to,
        "max_repeat_per_bag": max_repeat_per_bag,
        "max_added_records": max_added_records,
        "allow_overshoot_ratio": allow_overshoot_ratio,
        "top_candidate_pool": top_candidate_pool,
        "candidate_sample_size": candidate_sample_size,
        "prefer_multi_tail_positive": bool(raw_value.get("prefer_multi_tail_positive", True)),
        "label_names": _normalize_name_list(raw_value.get("label_names", []), "class_balance.label_names"),
        "tail_labels": _normalize_name_list(raw_value.get("tail_labels", []), "class_balance.tail_labels"),
        "report_filename": str(raw_value.get("report_filename", "class_balance_report.json")).strip()
        or "class_balance_report.json",
    }


def load_train_config(config_path: Path) -> dict[str, Any]:
    if not config_path.is_file():
        raise FileNotFoundError(f"未找到训练配置文件: {config_path}")

    payload = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("训练配置文件格式错误")

    gpu_ids_raw = payload.get("gpu_ids", [0])
    if not isinstance(gpu_ids_raw, list) or not gpu_ids_raw:
        raise ValueError("gpu_ids 必须是非空列表")

    enabled_models_raw = payload.get("enabled_models", list(MODEL_SEQUENCE))
    if not isinstance(enabled_models_raw, list):
        raise ValueError("enabled_models 必须是列表")

    enabled_models = [str(item).strip() for item in enabled_models_raw if str(item).strip()]
    unknown_enabled = [name for name in enabled_models if name not in SUPPORTED_MODEL_NAMES]
    if unknown_enabled:
        raise ValueError(f"enabled_models 中存在未知模型名：{unknown_enabled}")

    optimizer_name = str(payload.get("optimizer_name", "adamw")).strip().lower()
    if optimizer_name not in {"adamw", "adam", "sgd"}:
        raise ValueError("optimizer_name 仅支持 adamw、adam、sgd")

    image_cache_mode = str(payload.get("image_cache_mode", "none")).strip().lower()
    if image_cache_mode not in {"none", "memory", "disk", "memory_and_disk"}:
        raise ValueError("image_cache_mode 仅支持 none、memory、disk、memory_and_disk")

    image_cache_scope = str(payload.get("image_cache_scope", "task")).strip().lower() or "task"
    if image_cache_scope not in IMAGE_CACHE_SCOPE_VALUES:
        raise ValueError("image_cache_scope 仅支持 task、shared")

    loader_prefetch_factor = int(payload.get("loader_prefetch_factor", 2))
    if loader_prefetch_factor <= 0:
        raise ValueError("loader_prefetch_factor 必须大于 0")

    default_run = {
        "batch_size": int(payload.get("batch_size", 3)),
        "eval_batch_size": int(payload.get("eval_batch_size", 3)),
        "train_max_instances": int(payload.get("train_max_instances", 32)),
        "eval_max_instances": int(payload.get("eval_max_instances", 32)),
        "train_max_batch_instances": int(payload.get("train_max_batch_instances", 96)),
        "eval_max_batch_instances": int(payload.get("eval_max_batch_instances", 96)),
        "pin_memory": bool(payload.get("pin_memory", True)),
        "persistent_workers": bool(payload.get("persistent_workers", True)),
        "loader_prefetch_factor": loader_prefetch_factor,
        "image_cache_mode": image_cache_mode,
        "image_cache_scope": image_cache_scope,
        "image_cache_dir": str(payload.get("image_cache_dir", ".")).strip(),
        "image_cache_warmup": bool(payload.get("image_cache_warmup", False)),
        "memory_cache_size": int(payload.get("memory_cache_size", 0)),
        "random_instance_dropout": float(payload.get("random_instance_dropout", 0.05)),
        "lr": float(payload.get("lr", 2e-4)),
        "optimizer_name": optimizer_name,
        "weight_decay": float(payload.get("weight_decay", 1e-4)),
        "warmup_ratio": float(payload.get("warmup_ratio", 0.1)),
        "grad_accum_steps": int(payload.get("grad_accum_steps", 2)),
        "amp": bool(payload.get("amp", True)),
        "topk_evidence": int(payload.get("topk_evidence", 5)),
        "loss_name": str(payload.get("loss_name", "asymmetric")),
        "monitor_metric": str(payload.get("monitor_metric", "")).strip(),
        "monitor_mode": str(payload.get("monitor_mode", "")).strip().lower(),
    }

    auto_explore_raw = payload.get("auto_explore", False)
    if isinstance(auto_explore_raw, dict):
        auto_explore_enabled = bool(auto_explore_raw.get("enabled", False))
    else:
        auto_explore_enabled = bool(auto_explore_raw)

    auto_baselines_raw = payload.get("auto_baselines", False)
    if isinstance(auto_baselines_raw, dict):
        auto_baselines_enabled = bool(auto_baselines_raw.get("enabled", False))
    else:
        auto_baselines_enabled = bool(auto_baselines_raw)

    auto_sotas_raw = payload.get("auto_sotas", False)
    if isinstance(auto_sotas_raw, dict):
        auto_sotas_enabled = bool(auto_sotas_raw.get("enabled", False))
    else:
        auto_sotas_enabled = bool(auto_sotas_raw)

    auto_ablations_selection = normalize_auto_ablations_selection(payload.get("auto_ablations", False))
    auto_distinct_raw = payload.get("auto_distinct", False)
    if isinstance(auto_distinct_raw, dict):
        auto_distinct_enabled = bool(auto_distinct_raw.get("enabled", False))
    else:
        auto_distinct_enabled = bool(auto_distinct_raw)
    auto_5fold_raw = payload.get("auto_5fold", False)
    if isinstance(auto_5fold_raw, dict):
        auto_5fold_enabled = bool(auto_5fold_raw.get("enabled", False))
    else:
        auto_5fold_enabled = bool(auto_5fold_raw)
    auto_exp1_raw = payload.get("auto_exp_1", False)
    if isinstance(auto_exp1_raw, dict):
        auto_exp1_enabled = bool(auto_exp1_raw.get("enabled", False))
    else:
        auto_exp1_enabled = bool(auto_exp1_raw)
    auto_exp2_raw = payload.get("auto_exp_2", False)
    if isinstance(auto_exp2_raw, dict):
        auto_exp2_enabled = bool(auto_exp2_raw.get("enabled", False))
    else:
        auto_exp2_enabled = bool(auto_exp2_raw)
    auto_exp3_raw = payload.get("auto_exp_3", False)
    if isinstance(auto_exp3_raw, dict):
        auto_exp3_enabled = bool(auto_exp3_raw.get("enabled", False))
    else:
        auto_exp3_enabled = bool(auto_exp3_raw)
    auto_exp4_raw = payload.get("auto_exp_4", False)
    if isinstance(auto_exp4_raw, dict):
        auto_exp4_enabled = bool(auto_exp4_raw.get("enabled", False))
    else:
        auto_exp4_enabled = bool(auto_exp4_raw)
    auto_exp5_raw = payload.get("auto_exp_5", False)
    if isinstance(auto_exp5_raw, dict):
        auto_exp5_enabled = bool(auto_exp5_raw.get("enabled", False))
    else:
        auto_exp5_enabled = bool(auto_exp5_raw)
    auto_exp6_raw = payload.get("auto_exp_6", False)
    if isinstance(auto_exp6_raw, dict):
        auto_exp6_enabled = bool(auto_exp6_raw.get("enabled", False))
    else:
        auto_exp6_enabled = bool(auto_exp6_raw)
    auto_exp7_raw = payload.get("auto_exp_7", False)
    if isinstance(auto_exp7_raw, dict):
        auto_exp7_enabled = bool(auto_exp7_raw.get("enabled", False))
    else:
        auto_exp7_enabled = bool(auto_exp7_raw)
    auto_exp8_raw = payload.get("auto_exp_8", False)
    if isinstance(auto_exp8_raw, dict):
        auto_exp8_enabled = bool(auto_exp8_raw.get("enabled", False))
    else:
        auto_exp8_enabled = bool(auto_exp8_raw)
    auto_exp8_mm_ablation_raw = payload.get("auto_exp_8_mm_ablation", False)
    if isinstance(auto_exp8_mm_ablation_raw, dict):
        auto_exp8_mm_ablation_enabled = bool(auto_exp8_mm_ablation_raw.get("enabled", False))
    else:
        auto_exp8_mm_ablation_enabled = bool(auto_exp8_mm_ablation_raw)
    auto_exp9_ablation_raw = payload.get("auto_exp_9_ablation", False)
    if isinstance(auto_exp9_ablation_raw, dict):
        auto_exp9_ablation_enabled = bool(auto_exp9_ablation_raw.get("enabled", False))
    else:
        auto_exp9_ablation_enabled = bool(auto_exp9_ablation_raw)
    auto_exp11_module_ablation_raw = payload.get("auto_exp_11_module_ablation", False)
    if isinstance(auto_exp11_module_ablation_raw, dict):
        auto_exp11_module_ablation_enabled = bool(auto_exp11_module_ablation_raw.get("enabled", False))
    else:
        auto_exp11_module_ablation_enabled = bool(auto_exp11_module_ablation_raw)
    auto_exp2_skip_models = _normalize_name_list(
        payload.get("auto_exp_2_skip_models", []),
        "auto_exp_2_skip_models",
    )
    unknown_auto_exp2_skip = [
        name for name in auto_exp2_skip_models if name not in AUTO_EXP2_ALLOWED_MODEL_NAMES
    ]
    if unknown_auto_exp2_skip:
        raise ValueError(f"auto_exp_2_skip_models 中存在未知 exp_2 模型名：{unknown_auto_exp2_skip}")
    task_name = str(payload.get("task_name", DEFAULT_GASTRO_TASK_NAME)).strip() or DEFAULT_GASTRO_TASK_NAME
    get_task_spec(task_name)

    return {
        "gpu_ids": [int(item) for item in gpu_ids_raw],
        "task_name": task_name,
        "num_workers": int(payload.get("num_workers", 6)),
        "seed": int(payload.get("seed", 42)),
        "image_size": int(payload.get("image_size", 224)),
        "max_epochs": int(payload.get("max_epochs", 30)),
        "patience": int(payload.get("patience", 30)),
        "max_exams_per_task": int(payload.get("max_exams_per_task", 0)),
        "min_instances": int(payload.get("min_instances", 1)),
        "group_by_patient": bool(payload.get("group_by_patient", False)),
        "split_ratio": _load_ratio(payload, "split_ratio"),
        "train_sampling_strategy": str(payload.get("train_sampling_strategy", "random")),
        "eval_sampling_strategy": str(payload.get("eval_sampling_strategy", "uniform")),
        "task_selection_dir_name": str(payload.get("task_selection_dir_name", "task_data")),
        "train_run_dir_name": str(payload.get("train_run_dir_name", "train_runs")),
        "experiment_dir_name": str(payload.get("experiment_dir_name", "")).strip(),
        "single_run_dir_name": str(payload.get("single_run_dir_name", "")).strip(),
        "run_dir_prefix": str(payload.get("run_dir_prefix", "run")),
        "structured_min_category_count": int(payload.get("structured_min_category_count", 20)),
        "remark_metric_alias": str(payload.get("remark_metric_alias", "best_macro_f1")),
        "remark_metric_name": str(payload.get("remark_metric_name", "macro_f1")),
        "enabled_models": enabled_models,
        "default_run": default_run,
        "class_balance": normalize_class_balance_config(payload.get("class_balance", False)),
        "auto_explore": auto_explore_enabled,
        "auto_baselines": auto_baselines_enabled,
        "auto_sotas": auto_sotas_enabled,
        "auto_ablations": auto_ablations_selection,
        "auto_distinct": auto_distinct_enabled,
        "auto_5fold": auto_5fold_enabled,
        "auto_exp_1": auto_exp1_enabled,
        "auto_exp_2": auto_exp2_enabled,
        "auto_exp_3": auto_exp3_enabled,
        "auto_exp_4": auto_exp4_enabled,
        "auto_exp_5": auto_exp5_enabled,
        "auto_exp_6": auto_exp6_enabled,
        "auto_exp_7": auto_exp7_enabled,
        "auto_exp_8": auto_exp8_enabled,
        "auto_exp_8_mm_ablation": auto_exp8_mm_ablation_enabled,
        "auto_exp_9_ablation": auto_exp9_ablation_enabled,
        "auto_exp_11_module_ablation": auto_exp11_module_ablation_enabled,
        "auto_exp_5_roi": payload.get("auto_exp_5_roi", {}) or {},
        "auto_exp_6_roi": payload.get("auto_exp_6_roi", payload.get("auto_exp_5_roi", {})) or {},
        "auto_exp_7_roi": payload.get("auto_exp_7_roi", payload.get("auto_exp_6_roi", payload.get("auto_exp_5_roi", {}))) or {},
        "auto_exp_2_skip_models": auto_exp2_skip_models,
    }


def load_auto_explore_config(config_path: Path, allowed_run_keys: set[str]) -> dict[str, Any]:
    if not config_path.is_file():
        raise FileNotFoundError(f"未找到自动探索配置文件: {config_path}")

    payload = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("自动探索配置文件格式错误")

    selection_alias = str(payload.get("selection_alias", "best_val_loss")).strip()
    if selection_alias not in TRACKER_ALIAS_TO_META:
        raise ValueError(
            f"auto_explore.selection_alias 仅支持：{list(TRACKER_ALIAS_TO_META.keys())}"
        )

    result_source = str(payload.get("result_source", "best_checkpoints")).strip()
    if result_source != "best_checkpoints":
        raise ValueError("自动探索阶段的 auto_explore.result_source 当前仅支持 best_checkpoints")

    search_method = str(payload.get("search_method", "optuna_tpe")).strip().lower()
    if search_method not in {"optuna_tpe", "random"}:
        raise ValueError("auto_explore.search_method 仅支持 optuna_tpe 或 random")

    remark_raw = payload.get("remark", {})
    if remark_raw is None:
        remark_raw = {}
    if not isinstance(remark_raw, dict):
        raise ValueError("auto_explore.remark 配置格式错误")

    auto_explore_cfg = {
        "config_path": str(config_path.resolve()),
        "goal": str(payload.get("goal", "")).strip(),
        "num_trials": int(payload.get("num_trials", 12)),
        "trial_max_epochs": int(payload.get("trial_max_epochs", 12)),
        "trial_patience": int(payload.get("trial_patience", 6)),
        "trial_run_test": bool(payload.get("trial_run_test", False)),
        "search_method": search_method,
        "selection_alias": selection_alias,
        "selection_metric_name": TRACKER_ALIAS_TO_META[selection_alias]["metric_name"],
        "selection_mode": TRACKER_ALIAS_TO_META[selection_alias]["mode"],
        "result_source": result_source,
        "search_space": normalize_auto_explore_space(
            payload.get("search_space", {}),
            allowed_keys=allowed_run_keys,
        ),
        "stability_filter": normalize_stability_filter(
            payload.get("stability_filter", {}),
            prefix="auto_explore.stability_filter",
        ),
        "objective": normalize_auto_explore_objective(
            payload.get("objective", {}),
            prefix="auto_explore.objective",
        ),
        "optuna": normalize_optuna_settings(
            payload.get("optuna", {}),
            prefix="auto_explore.optuna",
        ),
        "remark": {
            "focus": str(remark_raw.get("focus", "")).strip(),
            "include_model_evaluations": bool(remark_raw.get("include_model_evaluations", True)),
            "include_search_space": bool(remark_raw.get("include_search_space", True)),
            "include_stability_filter": bool(remark_raw.get("include_stability_filter", True)),
        },
    }
    if auto_explore_cfg["num_trials"] <= 0:
        raise ValueError("auto_explore.num_trials 必须大于 0")
    if auto_explore_cfg["trial_max_epochs"] < 0 or auto_explore_cfg["trial_patience"] < 0:
        raise ValueError("auto_explore.trial_max_epochs 和 auto_explore.trial_patience 不能小于 0")
    return auto_explore_cfg


def normalize_auto_model_entries(
    raw_models: Any,
    allowed_run_keys: set[str],
    *,
    allowed_model_names: tuple[str, ...],
    config_prefix: str,
    selected_task_name: str,
) -> list[dict[str, Any]]:
    if not isinstance(raw_models, list) or not raw_models:
        raise ValueError(f"{config_prefix}.models 必须是非空列表")

    normalized: list[dict[str, Any]] = []
    seen_names: set[str] = set()
    for index, item in enumerate(raw_models, start=1):
        if not isinstance(item, dict):
            raise ValueError(f"{config_prefix}.models[{index}] 配置格式错误")

        model_name = str(item.get("name", "")).strip()
        if not model_name:
            raise ValueError(f"{config_prefix}.models[{index}].name 不能为空")
        if model_name not in allowed_model_names:
            raise ValueError(f"{config_prefix}.models[{index}].name 不支持: {model_name}")
        if model_name in seen_names:
            raise ValueError(f"{config_prefix}.models 中存在重复模型: {model_name}")

        model_params = item.get("model_params", {}) or {}
        if not isinstance(model_params, dict):
            raise ValueError(f"{config_prefix}.models[{index}].model_params 必须是字典")

        run_overrides = item.get("run_overrides", {}) or {}
        if not isinstance(run_overrides, dict):
            raise ValueError(f"{config_prefix}.models[{index}].run_overrides 必须是字典")
        unknown_run_overrides = [key for key in run_overrides.keys() if key not in allowed_run_keys]
        if unknown_run_overrides:
            raise ValueError(
                f"{config_prefix}.models[{index}].run_overrides 存在未知参数: {unknown_run_overrides}"
            )

        entry_seed_raw = item.get("seed", None)
        entry_seed = None if entry_seed_raw in (None, "") else int(entry_seed_raw)
        metadata = item.get("metadata", {}) or {}
        if not isinstance(metadata, dict):
            raise ValueError(f"{config_prefix}.models[{index}].metadata 必须是字典")

        task_meta = resolve_model_task_meta(model_name, selected_task_name)
        normalized.append(
            {
                "name": model_name,
                "display_name": str(item.get("display_name", model_name)).strip() or model_name,
                "enabled": bool(item.get("enabled", True)),
                "seed": entry_seed,
                "metadata": metadata,
                "model_params": model_params,
                "run_overrides": run_overrides,
                "task_name": task_meta["task_name"],
                "task_dir_name": task_meta["task_dir_name"],
            }
        )
        seen_names.add(model_name)

    return normalized


def load_auto_model_series_config(
    config_path: Path,
    allowed_run_keys: set[str],
    *,
    config_prefix: str,
    allowed_model_names: tuple[str, ...],
    default_output_dir_name: str,
    selected_task_name: str,
) -> dict[str, Any]:
    if not config_path.is_file():
        raise FileNotFoundError(f"未找到配置文件: {config_path}")

    payload = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"{config_prefix} 配置文件格式错误")

    selection_alias = str(payload.get("selection_alias", "best_macro_f1")).strip()
    if selection_alias not in TRACKER_ALIAS_TO_META:
        raise ValueError(
            f"{config_prefix}.selection_alias 仅支持：{list(TRACKER_ALIAS_TO_META.keys())}"
        )

    result_source = str(payload.get("result_source", "test_results")).strip()
    if result_source not in {"test_results", "best_checkpoints"}:
        raise ValueError(f"{config_prefix}.result_source 仅支持 test_results 或 best_checkpoints")

    remark_raw = payload.get("remark", {})
    if remark_raw is None:
        remark_raw = {}
    if not isinstance(remark_raw, dict):
        raise ValueError(f"{config_prefix}.remark 配置格式错误")

    series_cfg = {
        "config_path": str(config_path.resolve()),
        "goal": str(payload.get("goal", "")).strip(),
        "output_dir_name": str(payload.get("output_dir_name", default_output_dir_name)).strip() or default_output_dir_name,
        "run_test": bool(payload.get("run_test", True)),
        "selection_alias": selection_alias,
        "selection_metric_name": TRACKER_ALIAS_TO_META[selection_alias]["metric_name"],
        "selection_mode": TRACKER_ALIAS_TO_META[selection_alias]["mode"],
        "result_source": result_source,
        "stability_filter": normalize_stability_filter(
            payload.get("stability_filter", {}),
            prefix=f"{config_prefix}.stability_filter",
        ),
        "remark": {
            "focus": str(remark_raw.get("focus", "")).strip(),
            "include_model_evaluations": bool(remark_raw.get("include_model_evaluations", True)),
            "include_stability_filter": bool(remark_raw.get("include_stability_filter", True)),
        },
        "models": normalize_auto_model_entries(
            payload.get("models", []),
            allowed_run_keys=allowed_run_keys,
            allowed_model_names=allowed_model_names,
            config_prefix=config_prefix,
            selected_task_name=selected_task_name,
        ),
    }
    if not series_cfg["run_test"]:
        raise ValueError(f"{config_prefix}.run_test 当前必须为 true，保证每个模型训练后立即测试")
    if series_cfg["selection_metric_name"] == "val_loss" and result_source == "test_results":
        raise ValueError(f"{config_prefix} 使用 test_results 排序时，selection_alias 不能为 best_val_loss")

    enabled_models = [item for item in series_cfg["models"] if item["enabled"]]
    if not enabled_models:
        raise ValueError(f"{config_prefix}.models 至少需要启用一个模型")

    task_names = {item["task_name"] for item in enabled_models}
    if len(task_names) != 1:
        raise ValueError(f"{config_prefix} 当前仅支持单任务批量运行，请将胃镜和肠镜分开配置")

    series_cfg["task_name"] = enabled_models[0]["task_name"]
    series_cfg["task_dir_name"] = enabled_models[0]["task_dir_name"]
    return series_cfg


def load_auto_baselines_config(
    config_path: Path,
    allowed_run_keys: set[str],
    *,
    selected_task_name: str,
) -> dict[str, Any]:
    return load_auto_model_series_config(
        config_path,
        allowed_run_keys,
        config_prefix="auto_baselines",
        allowed_model_names=AUTO_BASELINE_ALLOWED_MODEL_NAMES,
        default_output_dir_name="baselines",
        selected_task_name=selected_task_name,
    )


def load_auto_sotas_config(
    config_path: Path,
    allowed_run_keys: set[str],
    *,
    selected_task_name: str,
) -> dict[str, Any]:
    return load_auto_model_series_config(
        config_path,
        allowed_run_keys,
        config_prefix="auto_sotas",
        allowed_model_names=AUTO_SOTA_ALLOWED_MODEL_NAMES,
        default_output_dir_name="sotas",
        selected_task_name=selected_task_name,
    )


def normalize_task1_module_instance_search(raw_value: Any, selected_task_name: str) -> dict[str, Any]:
    if raw_value is None or raw_value is False:
        return {"enabled": False}
    if raw_value is True:
        raw_value = {}
    if not isinstance(raw_value, dict):
        raise ValueError("task1_module_instance_search 仅支持 false、true 或字典配置")

    enabled = bool(raw_value.get("enabled", True))
    if enabled and selected_task_name != "task1":
        raise ValueError("task1_module_instance_search 仅支持 TASK1")
    if not enabled:
        return {"enabled": False}

    initial_train_max_instances = int(raw_value.get("initial_train_max_instances", 16))
    if initial_train_max_instances <= 0:
        raise ValueError("task1_module_instance_search.initial_train_max_instances 必须大于 0")

    raw_values = raw_value.get("train_max_instances_values", [8, 12, 16, 20, 24])
    if not isinstance(raw_values, list) or not raw_values:
        raise ValueError("task1_module_instance_search.train_max_instances_values 必须是非空列表")
    train_max_instances_values = []
    for item in raw_values:
        value = int(item)
        if value <= 0:
            raise ValueError("task1_module_instance_search.train_max_instances_values 必须全部大于 0")
        if value not in train_max_instances_values:
            train_max_instances_values.append(value)
    if initial_train_max_instances not in train_max_instances_values:
        train_max_instances_values.append(initial_train_max_instances)
        train_max_instances_values = sorted(train_max_instances_values)

    return {
        "enabled": True,
        "output_dir_name": str(raw_value.get("output_dir_name", "exp_task1_auto_module_instance_search")).strip()
        or "exp_task1_auto_module_instance_search",
        "initial_train_max_instances": initial_train_max_instances,
        "train_max_instances_values": train_max_instances_values,
        "rerun_modules_if_best_instances_differs": bool(
            raw_value.get("rerun_modules_if_best_instances_differs", True)
        ),
        "run_final_model_suite": bool(raw_value.get("run_final_model_suite", True)),
        "final_suite_dir_name": str(raw_value.get("final_suite_dir_name", "final_models_best_params")).strip()
        or "final_models_best_params",
        "fixed_seed": bool(raw_value.get("fixed_seed", True)),
    }


def normalize_auto_ablation_entries(
    raw_models: Any,
    allowed_run_keys: set[str],
    *,
    config_prefix: str,
    selected_task_name: str,
) -> list[dict[str, Any]]:
    if not isinstance(raw_models, list) or not raw_models:
        raise ValueError(f"{config_prefix}.models 必须是非空列表")

    normalized: list[dict[str, Any]] = []
    seen_names: set[str] = set()
    for index, item in enumerate(raw_models, start=1):
        if not isinstance(item, dict):
            raise ValueError(f"{config_prefix}.models[{index}] 配置格式错误")

        entry_name = str(item.get("name", "")).strip()
        if not entry_name:
            raise ValueError(f"{config_prefix}.models[{index}].name 不能为空")
        if entry_name in seen_names:
            raise ValueError(f"{config_prefix}.models 中存在重复条目: {entry_name}")

        base_model_name = str(item.get("base_model_name", "")).strip()
        if not base_model_name:
            raise ValueError(f"{config_prefix}.models[{index}].base_model_name 不能为空")
        if base_model_name not in AUTO_ABLATION_ALLOWED_MODEL_NAMES:
            raise ValueError(
                f"{config_prefix}.models[{index}].base_model_name 不支持: {base_model_name}"
            )

        model_params = item.get("model_params", {}) or {}
        if not isinstance(model_params, dict):
            raise ValueError(f"{config_prefix}.models[{index}].model_params 必须是字典")

        run_overrides = item.get("run_overrides", {}) or {}
        if not isinstance(run_overrides, dict):
            raise ValueError(f"{config_prefix}.models[{index}].run_overrides 必须是字典")
        unknown_run_overrides = [key for key in run_overrides.keys() if key not in allowed_run_keys]
        if unknown_run_overrides:
            raise ValueError(
                f"{config_prefix}.models[{index}].run_overrides 存在未知参数: {unknown_run_overrides}"
            )

        entry_seed_raw = item.get("seed", None)
        entry_seed = None if entry_seed_raw in (None, "") else int(entry_seed_raw)
        metadata = item.get("metadata", {}) or {}
        if not isinstance(metadata, dict):
            raise ValueError(f"{config_prefix}.models[{index}].metadata 必须是字典")

        task_meta = resolve_model_task_meta(base_model_name, selected_task_name)
        normalized.append(
            {
                "name": entry_name,
                "base_model_name": base_model_name,
                "display_name": str(item.get("display_name", entry_name)).strip() or entry_name,
                "enabled": bool(item.get("enabled", True)),
                "seed": entry_seed,
                "metadata": metadata,
                "model_params": model_params,
                "run_overrides": run_overrides,
                "task_name": task_meta["task_name"],
                "task_dir_name": task_meta["task_dir_name"],
            }
        )
        seen_names.add(entry_name)

    return normalized


def load_auto_ablations_config(
    config_path: Path,
    allowed_run_keys: set[str],
    *,
    selected_task_name: str,
) -> dict[str, Any]:
    if not config_path.is_file():
        raise FileNotFoundError(f"未找到配置文件: {config_path}")

    payload = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("auto_ablations 配置文件格式错误")

    selection_alias = str(payload.get("selection_alias", "best_macro_f1")).strip()
    if selection_alias not in TRACKER_ALIAS_TO_META:
        raise ValueError(
            f"auto_ablations.selection_alias 仅支持：{list(TRACKER_ALIAS_TO_META.keys())}"
        )

    result_source = str(payload.get("result_source", "test_results")).strip()
    if result_source not in {"test_results", "best_checkpoints"}:
        raise ValueError("auto_ablations.result_source 仅支持 test_results 或 best_checkpoints")

    output_root_dir_name = str(payload.get("output_root_dir_name", "ablations")).strip() or "ablations"
    common_goal = str(payload.get("goal", "")).strip()
    task1_module_instance_search = normalize_task1_module_instance_search(
        payload.get("task1_module_instance_search", False),
        selected_task_name,
    )

    remark_raw = payload.get("remark", {})
    if remark_raw is None:
        remark_raw = {}
    if not isinstance(remark_raw, dict):
        raise ValueError("auto_ablations.remark 配置格式错误")

    registry = {item["name"]: item for item in build_all_ablation_experiments(selected_task_name)}
    raw_experiments = payload.get("experiments")
    if raw_experiments is None:
        raw_experiments = [{"name": name, "enabled": True} for name in list_ablation_experiment_names(selected_task_name)]
    if not isinstance(raw_experiments, list):
        raise ValueError("auto_ablations.experiments 必须是列表")
    if not raw_experiments:
        raise ValueError(f"{get_task_spec(selected_task_name).display_name} 当前没有可用的消融实验配置")

    normalized_experiments: list[dict[str, Any]] = []
    seen_experiment_names: set[str] = set()
    for index, item in enumerate(raw_experiments, start=1):
        if not isinstance(item, dict):
            raise ValueError(f"auto_ablations.experiments[{index}] 配置格式错误")

        experiment_name = str(item.get("name", "")).strip()
        if not experiment_name:
            raise ValueError(f"auto_ablations.experiments[{index}].name 不能为空")
        if experiment_name not in registry:
            raise ValueError(
                f"auto_ablations.experiments[{index}].name 不支持: {experiment_name}"
            )
        if experiment_name in seen_experiment_names:
            raise ValueError(f"auto_ablations.experiments 中存在重复实验: {experiment_name}")

        experiment_payload = registry[experiment_name]
        experiment_display_name = (
            str(item.get("display_name", experiment_payload.get("display_name", experiment_name))).strip()
            or experiment_name
        )
        experiment_goal = (
            str(item.get("goal", experiment_payload.get("goal", common_goal))).strip()
            or common_goal
        )
        experiment_output_dir_name = (
            str(item.get("output_dir_name", experiment_payload.get("output_dir_name", experiment_name))).strip()
            or experiment_name
        )
        models = normalize_auto_ablation_entries(
            experiment_payload.get("models", []),
            allowed_run_keys=allowed_run_keys,
            config_prefix=f"auto_ablations.experiments[{index}]",
            selected_task_name=selected_task_name,
        )
        enabled_models = [entry for entry in models if entry["enabled"]]
        if not enabled_models:
            raise ValueError(f"auto_ablations.experiments[{index}] 至少需要启用一个条目")

        task_names = {entry["task_name"] for entry in enabled_models}
        if len(task_names) != 1:
            raise ValueError(
                f"auto_ablations.experiments[{index}] 当前仅支持单任务批量运行"
            )

        focus_text = str(
            item.get(
                "focus",
                f"{str(remark_raw.get('focus', '')).strip()}（{experiment_display_name}）"
                if str(remark_raw.get("focus", "")).strip()
                else f"汇总 {experiment_display_name} 的测试结果与稳定性表现",
            )
        ).strip()

        normalized_experiments.append(
            {
                "name": experiment_name,
                "display_name": experiment_display_name,
                "enabled": bool(item.get("enabled", True)),
                "config_path": str(config_path.resolve()),
                "goal": experiment_goal,
                "output_dir_name": f"{output_root_dir_name}/{experiment_output_dir_name}",
                "run_test": bool(payload.get("run_test", True)),
                "selection_alias": selection_alias,
                "selection_metric_name": TRACKER_ALIAS_TO_META[selection_alias]["metric_name"],
                "selection_mode": TRACKER_ALIAS_TO_META[selection_alias]["mode"],
                "result_source": result_source,
                "stability_filter": normalize_stability_filter(
                    payload.get("stability_filter", {}),
                    prefix="auto_ablations.stability_filter",
                ),
                "remark": {
                    "focus": focus_text,
                    "include_model_evaluations": bool(remark_raw.get("include_model_evaluations", True)),
                    "include_stability_filter": bool(remark_raw.get("include_stability_filter", True)),
                },
                "models": models,
                "task_name": enabled_models[0]["task_name"],
                "task_dir_name": enabled_models[0]["task_dir_name"],
                "task1_module_instance_search": task1_module_instance_search,
            }
        )
        seen_experiment_names.add(experiment_name)

    enabled_experiments = [item for item in normalized_experiments if item["enabled"]]
    if not enabled_experiments:
        raise ValueError("auto_ablations.experiments 至少需要启用一个实验")

    for experiment in enabled_experiments:
        if not experiment["run_test"]:
            raise ValueError("auto_ablations.run_test 当前必须为 true，保证每个训练完成后立即测试")
        if experiment["selection_metric_name"] == "val_loss" and experiment["result_source"] == "test_results":
            raise ValueError("auto_ablations 使用 test_results 排序时，selection_alias 不能为 best_val_loss")

    return {
        "config_path": str(config_path.resolve()),
        "goal": common_goal,
        "output_root_dir_name": output_root_dir_name,
        "task1_module_instance_search": task1_module_instance_search,
        "experiments": normalized_experiments,
    }


def load_model_config(config_path: Path) -> dict[str, dict[str, Any]]:
    if not config_path.is_file():
        raise FileNotFoundError(f"未找到模型配置文件: {config_path}")

    payload = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    if payload is None:
        return {"models": {}}
    if not isinstance(payload, dict):
        raise ValueError("模型配置文件格式错误")

    models_payload = payload.get("models", {})
    if not isinstance(models_payload, dict):
        raise ValueError("模型配置文件缺少 models 分组或格式错误")

    normalized: dict[str, dict[str, Any]] = {}
    for model_name, cfg in models_payload.items():
        if not isinstance(cfg, dict):
            raise ValueError(f"models.{model_name} 配置格式错误")
        normalized[model_name] = cfg
    return {"models": normalized}


def maybe_limit_records(records: list[dict[str, Any]], max_num: int, seed: int) -> list[dict[str, Any]]:
    if max_num <= 0 or len(records) <= max_num:
        return records
    rng = np.random.default_rng(seed)
    indices = np.arange(len(records))
    rng.shuffle(indices)
    keep_indices = indices[:max_num]
    return [records[int(index)] for index in keep_indices]


def build_compatible_split(
    records: list[dict[str, Any]],
    seed: int,
    ratios: tuple[float, float, float],
    group_by_patient: bool = False,
) -> tuple[dict[str, list[dict[str, Any]]], str]:
    if not records:
        return {"train": [], "val": [], "test": []}, "empty"

    regular_split = split_records(records, seed=seed, ratios=ratios, group_by_patient=group_by_patient)
    if all(len(regular_split[key]) > 0 for key in ("train", "val", "test")):
        return regular_split, "ratio_split"

    rng = random.Random(seed)
    shuffled = list(records)
    rng.shuffle(shuffled)

    num_records = len(shuffled)
    if num_records >= 3:
        fallback_split = {
            "train": shuffled[:-2],
            "val": [shuffled[-2]],
            "test": [shuffled[-1]],
        }
    elif num_records == 2:
        fallback_split = {
            "train": [shuffled[0]],
            "val": [shuffled[1]],
            "test": [shuffled[0]],
        }
    else:
        fallback_split = {
            "train": [shuffled[0]],
            "val": [shuffled[0]],
            "test": [shuffled[0]],
        }

    return fallback_split, "small_sample_fallback"


def compute_multilabel_pos_weight(train_records: list[dict[str, Any]]) -> list[float]:
    labels = np.asarray([record["labels"] for record in train_records], dtype=np.float32)
    pos = labels.sum(axis=0)
    neg = labels.shape[0] - pos
    weights = (neg + 1.0) / (pos + 1.0)
    return weights.astype(np.float32).tolist()


def compute_binary_pos_weight(train_records: list[dict[str, Any]]) -> list[float]:
    labels = np.asarray([record["label"] for record in train_records], dtype=np.int64)
    pos = float((labels == 1).sum())
    neg = float((labels == 0).sum())
    return [float((neg + 1.0) / (pos + 1.0))]


def resolve_structured_report_csv_path(path_cfg: dict[str, str], task_name: str) -> Path | None:
    task_spec = get_task_spec(task_name)
    configured_path = str(path_cfg.get("valid_dicts_report_csv", "")).strip()
    if configured_path:
        return Path(configured_path).expanduser().resolve()
    if task_spec.default_report_csv and str(path_cfg.get("dataset_base_root", "")).strip():
        return (Path(path_cfg["dataset_base_root"]).expanduser().resolve() / task_spec.default_report_csv).resolve()
    return None


def _multilabel_value_counts(
    records: list[dict[str, Any]],
    label_names: tuple[str, ...] | list[str],
) -> list[dict[str, Any]]:
    if not records:
        return [
            {
                "label": label_name,
                "positive": 0,
                "negative": 0,
                "minority_value": None,
                "minority_count": 0,
                "majority_count": 0,
                "imbalance_gap": 0,
                "positive_rate": 0.0,
            }
            for label_name in label_names
        ]

    labels = np.asarray([record["labels"] for record in records], dtype=np.int64)
    total = int(labels.shape[0])
    summary: list[dict[str, Any]] = []
    for label_index, label_name in enumerate(label_names):
        positive = int(labels[:, label_index].sum())
        negative = int(total - positive)
        if positive < negative:
            minority_value: int | None = 1
            minority_count = positive
            majority_count = negative
        elif negative < positive:
            minority_value = 0
            minority_count = negative
            majority_count = positive
        else:
            minority_value = None
            minority_count = positive
            majority_count = negative
        summary.append(
            {
                "label": label_name,
                "positive": positive,
                "negative": negative,
                "minority_value": minority_value,
                "minority_count": minority_count,
                "majority_count": majority_count,
                "imbalance_gap": int(abs(positive - negative)),
                "positive_rate": float(positive / total) if total > 0 else 0.0,
            }
        )
    return summary


def _copy_balanced_record(record: dict[str, Any], source_index: int, duplicate_index: int) -> dict[str, Any]:
    copied = dict(record)
    if "image_paths" in copied:
        copied["image_paths"] = list(copied["image_paths"])
    if "labels" in copied:
        copied["labels"] = list(copied["labels"])
    if "pseudo_region_labels" in copied:
        copied["pseudo_region_labels"] = list(copied["pseudo_region_labels"])
    if "pseudo_relevance" in copied:
        copied["pseudo_relevance"] = list(copied["pseudo_relevance"])
    if "structured_raw" in copied:
        copied["structured_raw"] = dict(copied["structured_raw"])
    if "text_raw" in copied:
        copied["text_raw"] = dict(copied["text_raw"])
    if "structured_categorical" in copied:
        copied["structured_categorical"] = list(copied["structured_categorical"])
    if "structured_numeric" in copied:
        copied["structured_numeric"] = list(copied["structured_numeric"])
    if "structured_mask" in copied:
        copied["structured_mask"] = list(copied["structured_mask"])
    copied["_balance_source_index"] = int(source_index)
    copied["_balance_duplicate_index"] = int(duplicate_index)
    copied["_balance_virtual_duplicate"] = True
    return copied


def _resolve_balance_label_indices(
    *,
    configured_labels: list[str],
    all_label_names: tuple[str, ...],
    field_name: str,
) -> list[int]:
    if not configured_labels:
        return list(range(len(all_label_names)))
    label_to_index = {label_name: index for index, label_name in enumerate(all_label_names)}
    unknown_labels = [label_name for label_name in configured_labels if label_name not in label_to_index]
    if unknown_labels:
        raise ValueError(f"{field_name} 中存在未知标签：{unknown_labels}")
    return [label_to_index[label_name] for label_name in configured_labels]


def build_multilabel_minority_balance(
    *,
    train_records: list[dict[str, Any]],
    label_names: tuple[str, ...],
    cfg: dict[str, Any],
    seed: int,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    before_summary = _multilabel_value_counts(train_records, label_names)
    if not cfg.get("enabled", False) or not train_records or not label_names:
        return train_records, {
            "enabled": bool(cfg.get("enabled", False)),
            "applied": False,
            "reason": "disabled_or_empty",
            "before": before_summary,
            "after": before_summary,
        }

    balance_indices = _resolve_balance_label_indices(
        configured_labels=list(cfg.get("label_names", [])),
        all_label_names=label_names,
        field_name="class_balance.label_names",
    )
    tail_indices = set(
        _resolve_balance_label_indices(
            configured_labels=list(cfg.get("tail_labels", [])),
            all_label_names=label_names,
            field_name="class_balance.tail_labels",
        )
    )

    labels = np.asarray([record["labels"] for record in train_records], dtype=np.int64)
    if labels.ndim != 2 or labels.shape[1] != len(label_names):
        raise ValueError("训练记录 labels 维度与任务标签数不一致，无法执行 class_balance")

    rng = random.Random(seed)
    max_repeat_per_bag = int(cfg.get("max_repeat_per_bag", 30))
    repeat_cap_enabled = max_repeat_per_bag > 0
    max_added_records = int(cfg.get("max_added_records", 0))
    allow_overshoot_ratio = float(cfg.get("allow_overshoot_ratio", 0.05))
    top_candidate_pool = int(cfg.get("top_candidate_pool", 32))
    candidate_sample_size = int(cfg.get("candidate_sample_size", 256))
    prefer_multi_tail_positive = bool(cfg.get("prefer_multi_tail_positive", True))

    current_counts = np.zeros((len(label_names), 2), dtype=np.int64)
    current_counts[:, 1] = labels.sum(axis=0)
    current_counts[:, 0] = labels.shape[0] - current_counts[:, 1]

    original_counts = current_counts.copy()
    original_minority_values = np.full((len(label_names),), -1, dtype=np.int64)
    original_target_counts = np.zeros((len(label_names),), dtype=np.int64)
    for label_index in range(len(label_names)):
        negative = int(original_counts[label_index, 0])
        positive = int(original_counts[label_index, 1])
        if positive < negative:
            original_minority_values[label_index] = 1
            original_target_counts[label_index] = negative
        elif negative < positive:
            original_minority_values[label_index] = 0
            original_target_counts[label_index] = positive
        else:
            original_target_counts[label_index] = positive

    repeat_counts = np.ones((len(train_records),), dtype=np.int64)
    duplicate_records: list[dict[str, Any]] = []
    added_by_source = np.zeros((len(train_records),), dtype=np.int64)
    added_by_label = np.zeros((len(label_names),), dtype=np.int64)
    exhausted_labels: set[int] = set()

    def current_balance_state() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        deficits = np.zeros((len(label_names),), dtype=np.int64)
        fixed_minority_values = np.full((len(label_names),), -1, dtype=np.int64)
        majority_counts = np.zeros((len(label_names),), dtype=np.int64)
        for label_idx in balance_indices:
            minority_value = int(original_minority_values[label_idx])
            if minority_value < 0:
                continue
            target_count = int(original_target_counts[label_idx])
            current = int(current_counts[label_idx, minority_value])
            fixed_minority_values[label_idx] = minority_value
            majority_counts[label_idx] = target_count
            deficits[label_idx] = max(0, target_count - current)
        return deficits, fixed_minority_values, majority_counts

    def candidate_score(source_index: int) -> float:
        record_labels = labels[source_index]
        before_gap = np.abs(current_counts[:, 1] - current_counts[:, 0]).astype(np.float32)
        after_counts = current_counts.copy()
        for idx, value in enumerate(record_labels):
            after_counts[idx, int(value)] += 1
        after_gap = np.abs(after_counts[:, 1] - after_counts[:, 0]).astype(np.float32)
        gap_improvement = before_gap - after_gap

        weights = np.ones((len(label_names),), dtype=np.float32)
        deficits, dynamic_minority_values, majority_counts = current_balance_state()
        for idx in balance_indices:
            if deficits[idx] > 0:
                weights[idx] += min(3.0, float(deficits[idx]) / max(1.0, float(majority_counts[idx])))
        for idx in tail_indices:
            if int(dynamic_minority_values[idx]) == 1:
                weights[idx] += 1.5

        score = float((gap_improvement * weights).sum())
        if prefer_multi_tail_positive and tail_indices:
            score += 0.25 * sum(1 for idx in tail_indices if int(record_labels[idx]) == 1)
        score -= 0.05 * float(repeat_counts[source_index] - 1)
        score += rng.random() * 1e-4
        return score

    while True:
        deficits, dynamic_minority_values, majority_counts = current_balance_state()
        active_labels = [idx for idx in balance_indices if deficits[idx] > 0 and idx not in exhausted_labels]
        if not active_labels:
            break
        if max_added_records > 0 and len(duplicate_records) >= max_added_records:
            break

        selected_label = max(
            active_labels,
            key=lambda idx: float(deficits[idx]) / max(1.0, float(majority_counts[idx])),
        )
        selected_minority_value = int(dynamic_minority_values[selected_label])
        candidate_indices = np.where(labels[:, selected_label] == selected_minority_value)[0].tolist()
        if repeat_cap_enabled:
            candidate_indices = [
                idx for idx in candidate_indices if int(repeat_counts[idx]) < max_repeat_per_bag
            ]
        if not candidate_indices:
            exhausted_labels.add(selected_label)
            continue
        if len(candidate_indices) > candidate_sample_size:
            candidate_indices = rng.sample(candidate_indices, candidate_sample_size)

        scored_candidates = sorted(
            ((candidate_score(idx), idx) for idx in candidate_indices),
            key=lambda item: item[0],
            reverse=True,
        )
        top_candidates = scored_candidates[: min(top_candidate_pool, len(scored_candidates))]
        chosen_index = rng.choice(top_candidates)[1]

        repeat_counts[chosen_index] += 1
        added_by_source[chosen_index] += 1
        record_labels = labels[chosen_index]
        for label_index, value in enumerate(record_labels):
            current_counts[label_index, int(value)] += 1
        for label_index in balance_indices:
            if (
                deficits[label_index] > 0
                and int(dynamic_minority_values[label_index]) >= 0
                and int(record_labels[label_index]) == int(dynamic_minority_values[label_index])
            ):
                added_by_label[label_index] += 1
        duplicate_records.append(
            _copy_balanced_record(
                train_records[chosen_index],
                source_index=chosen_index,
                duplicate_index=int(added_by_source[chosen_index]),
            )
        )

    balanced_records = [*train_records, *duplicate_records]
    after_summary = _multilabel_value_counts(balanced_records, label_names)
    unresolved: list[dict[str, Any]] = []
    final_deficits, final_minority_values, final_target_counts = current_balance_state()
    for label_index in balance_indices:
        minority_value = int(final_minority_values[label_index])
        if minority_value < 0:
            continue
        current = int(current_counts[label_index, minority_value])
        target = int(final_target_counts[label_index])
        if int(final_deficits[label_index]) > 0:
            unresolved.append(
                {
                    "label": label_names[label_index],
                    "minority_value": minority_value,
                    "current_minority_count": current,
                    "target_count": target,
                    "remaining_deficit": int(final_deficits[label_index]),
                }
            )

    repeat_histogram_counter: dict[int, int] = {}
    for repeat_count in repeat_counts.tolist():
        repeat_histogram_counter[int(repeat_count)] = repeat_histogram_counter.get(int(repeat_count), 0) + 1

    top_repeated_indices = sorted(
        [idx for idx, added in enumerate(added_by_source.tolist()) if added > 0],
        key=lambda idx: int(added_by_source[idx]),
        reverse=True,
    )[:20]
    top_repeated_records = []
    for idx in top_repeated_indices:
        top_repeated_records.append(
            {
                "source_index": int(idx),
                "exam_dir": str(train_records[idx].get("exam_dir", "")),
                "patient_id": str(train_records[idx].get("patient_id", "")),
                "added_copies": int(added_by_source[idx]),
                "total_exposures": int(repeat_counts[idx]),
                "labels": {
                    label_names[label_index]: int(labels[idx, label_index])
                    for label_index in range(len(label_names))
                },
            }
        )

    report = {
        "enabled": True,
        "applied": True,
        "mode": str(cfg.get("mode", "multilabel_minority_oversample")),
        "target_strategy": str(cfg.get("target_strategy", "per_label_majority")),
        "apply_to": "train_only",
        "seed": int(seed),
        "original_train_size": len(train_records),
        "balanced_train_size": len(balanced_records),
        "added_records": len(duplicate_records),
        "max_repeat_per_bag": max_repeat_per_bag,
        "max_observed_repeat": int(repeat_counts.max()) if len(repeat_counts) else 0,
        "max_added_records": max_added_records,
        "allow_overshoot_ratio": allow_overshoot_ratio,
        "candidate_sample_size": candidate_sample_size,
        "balanced_label_names": [label_names[idx] for idx in balance_indices],
        "tail_labels": [label_names[idx] for idx in sorted(tail_indices)],
        "before": before_summary,
        "after": after_summary,
        "original_target_per_label": [
            {
                "label": label_names[idx],
                "original_minority_value": int(original_minority_values[idx]) if int(original_minority_values[idx]) >= 0 else None,
                "original_target_count": int(original_target_counts[idx]),
            }
            for idx in range(len(label_names))
        ],
        "added_minority_exposures": [
            {
                "label": label_names[idx],
                "added_exposures": int(added_by_label[idx]),
            }
            for idx in range(len(label_names))
        ],
        "unresolved_targets": unresolved,
        "repeat_histogram": {str(key): value for key, value in sorted(repeat_histogram_counter.items())},
        "top_repeated_records": top_repeated_records,
    }
    return balanced_records, report


def ceil_to_multiple(value: int, divisor: int) -> int:
    if divisor <= 0:
        return value
    return ((value + divisor - 1) // divisor) * divisor


def normalize_batch_size(value: int, active_gpu_count: int) -> int:
    batch_size = max(1, int(value))
    if active_gpu_count <= 1:
        return batch_size
    return ceil_to_multiple(max(batch_size, active_gpu_count), active_gpu_count)


def task_image_cache_dir_name(task_name: str) -> str:
    return get_task_spec(task_name).data_subdir


def resolve_image_cache_directories(
    *,
    task_name: str,
    cache_root_dir: Path,
    run_cfg: dict[str, Any],
) -> tuple[Path | None, Path | None, list[Path]]:
    raw_cache_dir = str(run_cfg.get("image_cache_dir", "")).strip()
    if not raw_cache_dir:
        return None, None, []

    candidate_cache_dir = Path(raw_cache_dir).expanduser()
    resolved_cache_root_dir = (
        candidate_cache_dir.resolve()
        if candidate_cache_dir.is_absolute()
        else (cache_root_dir / candidate_cache_dir).resolve()
    )
    cache_scope = str(run_cfg.get("image_cache_scope", "task")).strip().lower() or "task"

    if cache_scope == "shared":
        resolved_cache_dir = resolved_cache_root_dir / SHARED_IMAGE_CACHE_DIR_NAME
        legacy_dir_names = [task_image_cache_dir_name(task_spec.name) for task_spec in list_task_specs()]
    else:
        resolved_cache_dir = resolved_cache_root_dir / task_image_cache_dir_name(task_name)
        legacy_dir_names = [SHARED_IMAGE_CACHE_DIR_NAME]

    legacy_cache_dirs: list[Path] = []
    seen_dirs = {resolved_cache_dir}
    for dir_name in legacy_dir_names:
        candidate_dir = resolved_cache_root_dir / dir_name
        if candidate_dir in seen_dirs:
            continue
        seen_dirs.add(candidate_dir)
        legacy_cache_dirs.append(candidate_dir)

    return resolved_cache_root_dir, resolved_cache_dir, legacy_cache_dirs


def build_loaders(
    split_data: dict[str, list[dict[str, Any]]],
    task_name: str,
    image_size: int,
    num_workers: int,
    train_batch_size: int,
    eval_batch_size: int,
    train_max_instances: int,
    eval_max_instances: int,
    min_instances: int,
    train_sampling: str,
    eval_sampling: str,
    random_instance_dropout: float,
    train_max_batch_instances: int,
    eval_max_batch_instances: int,
    seed: int,
    pin_memory: bool,
    persistent_workers: bool,
    loader_prefetch_factor: int,
    image_cache_mode: str,
    image_cache_dir: str | Path | None,
    legacy_image_cache_dirs: list[str | Path] | None,
    image_cache_warmup: bool,
    memory_cache_size: int,
    roi_enabled: bool = False,
    roi_index_path: str | Path | None = None,
    roi_max_crops_per_bag: int = 0,
    roi_max_crops_per_source: int = 1,
    roi_min_score: float = 0.0,
    structured_shuffle_fields: list[str] | tuple[str, ...] | None = None,
    structured_shuffle_apply_to: str = "none",
    structured_shuffle_seed: int = 0,
) -> tuple[DataLoader, DataLoader, DataLoader]:
    train_dataset = MILBagDataset(
        records=split_data["train"],
        task_name=task_name,
        max_instances=train_max_instances,
        min_instances=min_instances,
        bag_sampling_strategy=train_sampling,
        is_train=True,
        image_size=image_size,
        random_instance_dropout=random_instance_dropout,
        image_cache_mode=image_cache_mode,
        image_cache_dir=image_cache_dir,
        legacy_image_cache_dirs=legacy_image_cache_dirs,
        memory_cache_size=memory_cache_size,
        roi_enabled=roi_enabled,
        roi_index_path=roi_index_path,
        roi_max_crops_per_bag=roi_max_crops_per_bag,
        roi_max_crops_per_source=roi_max_crops_per_source,
        roi_min_score=roi_min_score,
        split_name="train",
        structured_shuffle_fields=structured_shuffle_fields,
        structured_shuffle_apply_to=structured_shuffle_apply_to,
        structured_shuffle_seed=structured_shuffle_seed,
    )
    val_dataset = MILBagDataset(
        records=split_data["val"],
        task_name=task_name,
        max_instances=eval_max_instances,
        min_instances=min_instances,
        bag_sampling_strategy=eval_sampling,
        is_train=False,
        image_size=image_size,
        random_instance_dropout=0.0,
        image_cache_mode=image_cache_mode,
        image_cache_dir=image_cache_dir,
        legacy_image_cache_dirs=legacy_image_cache_dirs,
        memory_cache_size=memory_cache_size,
        roi_enabled=roi_enabled,
        roi_index_path=roi_index_path,
        roi_max_crops_per_bag=roi_max_crops_per_bag,
        roi_max_crops_per_source=roi_max_crops_per_source,
        roi_min_score=roi_min_score,
        split_name="val",
        structured_shuffle_fields=structured_shuffle_fields,
        structured_shuffle_apply_to=structured_shuffle_apply_to,
        structured_shuffle_seed=structured_shuffle_seed,
    )
    test_dataset = MILBagDataset(
        records=split_data["test"],
        task_name=task_name,
        max_instances=eval_max_instances,
        min_instances=min_instances,
        bag_sampling_strategy=eval_sampling,
        is_train=False,
        image_size=image_size,
        random_instance_dropout=0.0,
        image_cache_mode=image_cache_mode,
        image_cache_dir=image_cache_dir,
        legacy_image_cache_dirs=legacy_image_cache_dirs,
        memory_cache_size=memory_cache_size,
        roi_enabled=roi_enabled,
        roi_index_path=roi_index_path,
        roi_max_crops_per_bag=roi_max_crops_per_bag,
        roi_max_crops_per_source=roi_max_crops_per_source,
        roi_min_score=roi_min_score,
        split_name="test",
        structured_shuffle_fields=structured_shuffle_fields,
        structured_shuffle_apply_to=structured_shuffle_apply_to,
        structured_shuffle_seed=structured_shuffle_seed,
    )

    train_sampler = InstanceAwareBatchSampler(
        records=split_data["train"],
        max_instances_per_bag=train_max_instances,
        min_instances_per_bag=min_instances,
        batch_size=train_batch_size,
        max_instances_per_batch=train_max_batch_instances,
        shuffle=True,
        seed=seed,
    )
    val_sampler = InstanceAwareBatchSampler(
        records=split_data["val"],
        max_instances_per_bag=eval_max_instances,
        min_instances_per_bag=min_instances,
        batch_size=eval_batch_size,
        max_instances_per_batch=eval_max_batch_instances,
        shuffle=False,
        seed=seed + 1,
    )
    test_sampler = InstanceAwareBatchSampler(
        records=split_data["test"],
        max_instances_per_bag=eval_max_instances,
        min_instances_per_bag=min_instances,
        batch_size=eval_batch_size,
        max_instances_per_batch=eval_max_batch_instances,
        shuffle=False,
        seed=seed + 2,
    )

    active_pin_memory = bool(pin_memory) and torch.cuda.is_available()
    active_persistent_workers = bool(persistent_workers) and num_workers > 0
    active_prefetch_factor = max(1, int(loader_prefetch_factor))

    if image_cache_warmup and train_dataset.use_disk_cache:
        train_dataset.prepare_image_cache(desc=f"{task_name}-train缓存预构建")
        val_dataset.prepare_image_cache(desc=f"{task_name}-val缓存预构建")
        test_dataset.prepare_image_cache(desc=f"{task_name}-test缓存预构建")

    loader_kwargs: dict[str, Any] = {
        "num_workers": num_workers,
        "pin_memory": active_pin_memory,
        "collate_fn": mil_collate_fn,
        "persistent_workers": active_persistent_workers,
    }
    if num_workers > 0:
        loader_kwargs["prefetch_factor"] = active_prefetch_factor

    train_loader = DataLoader(train_dataset, batch_sampler=train_sampler, **loader_kwargs)
    val_loader = DataLoader(val_dataset, batch_sampler=val_sampler, **loader_kwargs)
    test_loader = DataLoader(test_dataset, batch_sampler=test_sampler, **loader_kwargs)
    print("=" * 72)
    return train_loader, val_loader, test_loader


def model_run_dir(session_dir: Path, model_index: int, model_name: str) -> Path:
    return session_dir / f"{model_index:02d}_{model_name}"


def write_training_config(
    *,
    run_dir: Path,
    model_name: str,
    task_name: str,
    trainer_cfg: TrainerConfig,
    run_cfg: dict[str, Any],
    model_param_cfg: dict[str, Any],
    split_data: dict[str, list[dict[str, Any]]],
    image_size: int,
    num_workers: int,
    seed: int,
) -> None:
    config_path = run_dir / "config.yaml"
    if trainer_cfg.resume_path and config_path.is_file():
        print(f"检测到断点续训，保留已有配置文件：{config_path}")
        return

    config_payload = {
        "model_name": model_name,
        "task_name": task_name,
        "seed": seed,
        "image_size": image_size,
        "num_workers": num_workers,
        "trainer": asdict(trainer_cfg),
        "run": run_cfg,
        "model_params": model_param_cfg,
        "split_stats": {key: len(value) for key, value in split_data.items()},
    }
    config_path.write_text(
        yaml.safe_dump(to_builtin_type(config_payload), allow_unicode=True, sort_keys=False),
        encoding="utf-8",
    )


def write_structured_audit_files(run_dir: Path, structured_metadata: dict[str, Any] | None) -> None:
    if not isinstance(structured_metadata, dict):
        return

    metadata_path = run_dir / "structured_metadata.json"
    metadata_path.write_text(
        json.dumps(to_builtin_type(structured_metadata), ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    audit_rows = structured_metadata.get("audit", [])
    if isinstance(audit_rows, list) and audit_rows:
        fieldnames: list[str] = []
        for row in audit_rows:
            if not isinstance(row, dict):
                continue
            for key in row.keys():
                if key not in fieldnames:
                    fieldnames.append(key)
        if fieldnames:
            with (run_dir / "field_audit.csv").open("w", encoding="utf-8-sig", newline="") as file:
                writer = csv.DictWriter(file, fieldnames=fieldnames)
                writer.writeheader()
                for row in audit_rows:
                    if isinstance(row, dict):
                        writer.writerow({field: row.get(field, "") for field in fieldnames})

        missing_rate = {
            str(row.get("field", "")): row.get("train_missing_rate", "")
            for row in audit_rows
            if isinstance(row, dict) and str(row.get("field", "")).strip()
        }
        if missing_rate:
            (run_dir / "missing_rate.json").write_text(
                json.dumps(to_builtin_type(missing_rate), ensure_ascii=False, indent=2),
                encoding="utf-8",
            )


def write_run_remark(
    session_dir: Path,
    all_results: dict[str, Any],
    remark_metric_alias: str,
    remark_metric_name: str,
    result_source: str = "test_results",
    stability_filter: dict[str, Any] | None = None,
    remark_context: dict[str, Any] | None = None,
) -> None:
    active_stability_filter = stability_filter or {}
    include_model_evaluations = True
    include_stability_filter = True
    if isinstance(remark_context, dict):
        include_model_evaluations = bool(remark_context.get("include_model_evaluations", True))
        include_stability_filter = bool(remark_context.get("include_stability_filter", True))

    candidates = resolve_session_candidate(
        all_results,
        remark_metric_alias=remark_metric_alias,
        result_source=result_source,
        fallback_metric_name=remark_metric_name,
    )
    evaluations = build_model_evaluations(
        all_results,
        remark_metric_alias=remark_metric_alias,
        remark_metric_name=remark_metric_name,
        result_source=result_source,
        stability_filter=active_stability_filter,
    ) if include_model_evaluations else []
    mode = TRACKER_ALIAS_TO_META.get(remark_metric_alias, {}).get("mode", "max")
    best_candidate = select_best_candidate(evaluations, mode=mode) if evaluations else select_best_candidate(candidates, mode=mode)

    best_model_payload = {
        "model_name": best_candidate["model_name"] if best_candidate else "",
        "train_dir": best_candidate["train_dir"] if best_candidate else "",
        "train_dir_path": best_candidate["train_dir_path"] if best_candidate else "",
        "score": best_candidate["score"] if best_candidate else float("nan"),
        "best_epoch": best_candidate["best_epoch"] if best_candidate else -1,
        "checkpoint_path": best_candidate["checkpoint_path"] if best_candidate else "",
        "evaluation": best_candidate.get("evaluation", "") if best_candidate else "",
    }

    remark_payload = {
        "run_dir": str(session_dir),
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "selection": {
            "checkpoint_alias": remark_metric_alias,
            "metric_name": remark_metric_name,
            "result_source": result_source,
            "mode": mode,
        },
        "best_model": best_model_payload,
        "best_train_dir": best_model_payload["train_dir"],
        "best_train_dir_path": best_model_payload["train_dir_path"],
        "best_model_name": best_model_payload["model_name"],
        "best_score": best_model_payload["score"],
        "model_evaluations": evaluations,
    }
    if remark_context:
        remark_payload["remark_context"] = remark_context
    if active_stability_filter and include_stability_filter:
        remark_payload["stability_filter"] = active_stability_filter

    (session_dir / "remark.txt").write_text(
        yaml.safe_dump(to_builtin_type(remark_payload), allow_unicode=True, sort_keys=False),
        encoding="utf-8",
    )


def selected_model_names(args_models: str, train_cfg: dict[str, Any]) -> list[str]:
    if args_models.strip():
        names = [item.strip() for item in args_models.split(",") if item.strip()]
    else:
        names = list(train_cfg["enabled_models"])

    unknown = [name for name in names if name not in SUPPORTED_MODEL_NAMES]
    if unknown:
        raise ValueError(f"存在未知模型名：{unknown}")
    if not names:
        raise ValueError("没有可运行的模型，请检查 --models 或 train.yaml 中的 enabled_models 配置")
    return names


def selected_auto_model_entries(
    args_models: str,
    series_cfg: dict[str, Any],
    *,
    config_prefix: str,
) -> list[dict[str, Any]]:
    enabled_entries = [item for item in series_cfg["models"] if item["enabled"]]
    if not args_models.strip():
        return enabled_entries

    requested_names = [item.strip() for item in args_models.split(",") if item.strip()]
    selected_entries = [item for item in enabled_entries if item["name"] in requested_names]
    missing = [name for name in requested_names if name not in {item["name"] for item in enabled_entries}]
    if missing:
        raise ValueError(f"{config_prefix} 配置中不存在这些已启用模型：{missing}")
    return selected_entries


def selected_auto_baseline_model_entries(
    args_models: str,
    auto_baselines_cfg: dict[str, Any],
) -> list[dict[str, Any]]:
    return selected_auto_model_entries(
        args_models,
        auto_baselines_cfg,
        config_prefix="auto_baselines",
    )


def selected_auto_sota_model_entries(
    args_models: str,
    auto_sotas_cfg: dict[str, Any],
) -> list[dict[str, Any]]:
    return selected_auto_model_entries(
        args_models,
        auto_sotas_cfg,
        config_prefix="auto_sotas",
    )


def selected_auto_ablation_experiments(
    selections: tuple[str, ...],
    auto_ablations_cfg: dict[str, Any],
) -> list[dict[str, Any]]:
    enabled_experiments = [item for item in auto_ablations_cfg["experiments"] if item["enabled"]]
    if not selections:
        return []

    requested = [item.strip() for item in selections if item.strip()]
    requested_lower = {item.lower() for item in requested}
    if "all" in requested_lower:
        return enabled_experiments

    selected = [item for item in enabled_experiments if item["name"] in requested_lower]
    missing = [name for name in requested if name.lower() not in {item["name"] for item in enabled_experiments}]
    if missing:
        raise ValueError(f"auto_ablations 配置中不存在这些已启用实验：{missing}")
    return selected


def build_auto_exp1_entries(
    *,
    selected_task_name: str,
    requested_names: list[str] | None = None,
) -> list[dict[str, Any]]:
    requested_name_set = set(requested_names or [])
    entries: list[dict[str, Any]] = []

    for model_name in AUTO_EXP1_ALLOWED_MODEL_NAMES:
        if requested_name_set and model_name not in requested_name_set:
            continue
        task_meta = resolve_model_task_meta(model_name, selected_task_name)
        entries.append(
            {
                "name": model_name,
                "display_name": model_name,
                "enabled": True,
                "task_name": task_meta["task_name"],
                "task_dir_name": task_meta["task_dir_name"],
                "run_prefix": task_meta["run_prefix"],
                "model_params": {},
                "run_overrides": {},
            }
        )

    if requested_names:
        missing = [name for name in requested_names if name not in AUTO_EXP1_ALLOWED_MODEL_NAMES]
        if missing:
            raise ValueError(f"auto_exp_1 模式下这些模型不属于 exp_1：{missing}")
        if not entries:
            raise ValueError("auto_exp_1 模式下没有可运行的 exp_1 模型")

    return entries


def build_auto_exp1_config(
    *,
    train_cfg: dict[str, Any],
    selected_task_name: str,
    model_entries: list[dict[str, Any]],
) -> dict[str, Any]:
    if not model_entries:
        raise ValueError("auto_exp_1 模式下没有可运行的 exp_1 模型")

    task_names = {entry["task_name"] for entry in model_entries}
    if len(task_names) != 1:
        raise ValueError("auto_exp_1 当前仅支持单任务批量运行")

    task_meta = resolve_model_task_meta(model_entries[0]["name"], selected_task_name)
    selection_alias = str(train_cfg.get("remark_metric_alias", "best_macro_f1")).strip() or "best_macro_f1"
    selection_meta = TRACKER_ALIAS_TO_META.get(selection_alias, TRACKER_ALIAS_TO_META["best_macro_f1"])

    return {
        "config_path": "inline:auto_exp_1",
        "goal": "顺序训练 exp_1 下的全部模型，并统一生成结果摘要。",
        "task_name": task_meta["task_name"],
        "task_dir_name": task_meta["task_dir_name"],
        "output_dir_name": "auto_exp_1",
        "selection_alias": selection_alias,
        "selection_metric_name": str(selection_meta["metric_name"]),
        "selection_mode": str(selection_meta["mode"]),
        "result_source": "test_results",
        "stability_filter": {
            "enabled": True,
            "min_epochs_trained": 10,
            "max_final_gap": 0.05,
            "max_val_loss_rebound_ratio": 0.25,
        },
        "remark": {
            "focus": "汇总 TASK2 的 exp_1 模型测试结果与稳定性表现",
            "include_model_evaluations": True,
            "include_stability_filter": True,
        },
        "models": model_entries,
        "run_test": True,
    }


def build_auto_exp2_entries(
    *,
    selected_task_name: str,
    requested_names: list[str] | None = None,
    skip_names: list[str] | None = None,
) -> list[dict[str, Any]]:
    requested_name_set = set(requested_names or [])
    skip_name_set = set(skip_names or [])
    entries: list[dict[str, Any]] = []

    for model_name in AUTO_EXP2_ALLOWED_MODEL_NAMES:
        if model_name in skip_name_set:
            continue
        if requested_name_set and model_name not in requested_name_set:
            continue
        task_meta = resolve_model_task_meta(model_name, selected_task_name)
        run_overrides: dict[str, Any] = {}
        if model_name == "entmax_mil":
            run_overrides = {
                "batch_size": 2,
                "eval_batch_size": 2,
                "train_max_instances": 8,
                "eval_max_instances": 8,
                "train_max_batch_instances": 16,
                "eval_max_batch_instances": 16,
                "num_workers": 0,
                "disable_multi_gpu": True,
                "pin_memory": False,
                "persistent_workers": False,
                "loader_prefetch_factor": 1,
                "image_cache_warmup": False,
                "amp": False,
            }
        entries.append(
            {
                "name": model_name,
                "display_name": model_name,
                "enabled": True,
                "task_name": task_meta["task_name"],
                "task_dir_name": task_meta["task_dir_name"],
                "run_prefix": task_meta["run_prefix"],
                "model_params": {},
                "run_overrides": run_overrides,
            }
        )

    if requested_names:
        missing = [name for name in requested_names if name not in AUTO_EXP2_ALLOWED_MODEL_NAMES]
        if missing:
            raise ValueError(f"auto_exp_2 模式下这些模型不属于 exp_2：{missing}")
        if not entries:
            raise ValueError("auto_exp_2 模式下没有可运行的 exp_2 模型")

    if skip_names:
        invalid_skip = [name for name in skip_names if name not in AUTO_EXP2_ALLOWED_MODEL_NAMES]
        if invalid_skip:
            raise ValueError(f"auto_exp_2_skip_models 中存在未知 exp_2 模型名：{invalid_skip}")

    return entries


def build_auto_exp2_config(
    *,
    train_cfg: dict[str, Any],
    selected_task_name: str,
    model_entries: list[dict[str, Any]],
) -> dict[str, Any]:
    if not model_entries:
        raise ValueError("auto_exp_2 模式下没有可运行的 exp_2 模型")

    task_names = {entry["task_name"] for entry in model_entries}
    if len(task_names) != 1:
        raise ValueError("auto_exp_2 当前仅支持单任务批量运行")

    task_meta = resolve_model_task_meta(model_entries[0]["name"], selected_task_name)
    selection_alias = str(train_cfg.get("remark_metric_alias", "best_macro_f1")).strip() or "best_macro_f1"
    selection_meta = TRACKER_ALIAS_TO_META.get(selection_alias, TRACKER_ALIAS_TO_META["best_macro_f1"])

    return {
        "config_path": "inline:auto_exp_2",
        "goal": "顺序训练 exp_2 下的全部模型，并统一生成结果摘要。",
        "task_name": task_meta["task_name"],
        "task_dir_name": task_meta["task_dir_name"],
        "output_dir_name": "auto_exp_2",
        "selection_alias": selection_alias,
        "selection_metric_name": str(selection_meta["metric_name"]),
        "selection_mode": str(selection_meta["mode"]),
        "result_source": "test_results",
        "stability_filter": {
            "enabled": True,
            "min_epochs_trained": 10,
            "max_final_gap": 0.05,
            "max_val_loss_rebound_ratio": 0.25,
        },
        "remark": {
            "focus": "汇总 TASK2 的 exp_2 模型测试结果与问题导向改进效果",
            "include_model_evaluations": True,
            "include_stability_filter": True,
        },
        "models": model_entries,
        "run_test": True,
    }


def build_auto_exp3_entries(
    *,
    selected_task_name: str,
    requested_names: list[str] | None = None,
) -> list[dict[str, Any]]:
    requested_name_set = set(requested_names or [])
    entries: list[dict[str, Any]] = []

    for model_name in AUTO_EXP3_ALLOWED_MODEL_NAMES:
        if requested_name_set and model_name not in requested_name_set:
            continue
        task_meta = resolve_model_task_meta(model_name, selected_task_name)
        entries.append(
            {
                "name": model_name,
                "display_name": model_name,
                "enabled": True,
                "task_name": task_meta["task_name"],
                "task_dir_name": task_meta["task_dir_name"],
                "run_prefix": task_meta["run_prefix"],
                "model_params": {},
                "run_overrides": {},
            }
        )

    if requested_names:
        missing = [name for name in requested_names if name not in AUTO_EXP3_ALLOWED_MODEL_NAMES]
        if missing:
            raise ValueError(f"auto_exp_3 模式下这些模型不属于 exp_3：{missing}")
        if not entries:
            raise ValueError("auto_exp_3 模式下没有可运行的 exp_3 模型")

    return entries


def build_auto_exp3_config(
    *,
    train_cfg: dict[str, Any],
    selected_task_name: str,
    model_entries: list[dict[str, Any]],
) -> dict[str, Any]:
    if not model_entries:
        raise ValueError("auto_exp_3 模式下没有可运行的 exp_3 模型")

    task_names = {entry["task_name"] for entry in model_entries}
    if len(task_names) != 1:
        raise ValueError("auto_exp_3 当前仅支持单任务批量运行")

    task_meta = resolve_model_task_meta(model_entries[0]["name"], selected_task_name)
    selection_alias = str(train_cfg.get("remark_metric_alias", "best_macro_f1")).strip() or "best_macro_f1"
    selection_meta = TRACKER_ALIAS_TO_META.get(selection_alias, TRACKER_ALIAS_TO_META["best_macro_f1"])

    return {
        "config_path": "inline:auto_exp_3",
        "goal": "顺序训练 exp_1 下的全部模型，并在训练集类别平衡条件下统一生成 exp_3 结果摘要。",
        "task_name": task_meta["task_name"],
        "task_dir_name": task_meta["task_dir_name"],
        "output_dir_name": "auto_exp_3",
        "selection_alias": selection_alias,
        "selection_metric_name": str(selection_meta["metric_name"]),
        "selection_mode": str(selection_meta["mode"]),
        "result_source": "test_results",
        "stability_filter": {
            "enabled": True,
            "min_epochs_trained": 10,
            "max_final_gap": 0.05,
            "max_val_loss_rebound_ratio": 0.25,
        },
        "remark": {
            "focus": "汇总 TASK2 的 exp_3 模型测试结果；exp_3 复用 exp_1 模型集合，并启用训练集类别平衡",
            "include_model_evaluations": True,
            "include_stability_filter": True,
        },
        "models": model_entries,
        "run_test": True,
    }


def _build_exp4_sampled_long_bag_run_overrides() -> dict[str, Any]:
    return {
        "batch_size": 1,
        "eval_batch_size": 1,
        "grad_accum_steps": 4,
        "train_max_instances": 128,
        "eval_max_instances": 128,
        "train_max_batch_instances": 256,
        "eval_max_batch_instances": 256,
        "random_instance_dropout": 0.0,
        "train_sampling_strategy": "uniform",
        "eval_sampling_strategy": "uniform",
        "persistent_workers": False,
        "loader_prefetch_factor": 1,
    }


def _build_exp4_long_run_overrides() -> dict[str, Any]:
    return _build_exp4_sampled_long_bag_run_overrides()


def build_auto_exp4_entries(
    *,
    selected_task_name: str,
    requested_names: list[str] | None = None,
) -> list[dict[str, Any]]:
    requested_name_set = set(requested_names or [])
    entries: list[dict[str, Any]] = []

    for model_name in AUTO_EXP4_ALLOWED_MODEL_NAMES:
        if requested_name_set and model_name not in requested_name_set:
            continue
        task_meta = resolve_model_task_meta(model_name, selected_task_name)
        run_overrides: dict[str, Any] = {}
        model_params: dict[str, Any] = {}
        if model_name in {"full_feature_mil", "hier_full_mil", "hier_full_lg_mil", "mamba_mil"}:
            run_overrides = _build_exp4_sampled_long_bag_run_overrides()
            model_params = {"encoder_chunk_size": 8}
        elif model_name == "long_mil":
            run_overrides = _build_exp4_long_run_overrides()

        entries.append(
            {
                "name": model_name,
                "display_name": model_name,
                "enabled": True,
                "task_name": task_meta["task_name"],
                "task_dir_name": task_meta["task_dir_name"],
                "run_prefix": task_meta["run_prefix"],
                "model_params": model_params,
                "run_overrides": run_overrides,
            }
        )

    if requested_names:
        missing = [name for name in requested_names if name not in AUTO_EXP4_ALLOWED_MODEL_NAMES]
        if missing:
            raise ValueError(f"auto_exp_4 模式下这些模型不属于 exp_4：{missing}")
        if not entries:
            raise ValueError("auto_exp_4 模式下没有可运行的 exp_4 模型")

    return entries


def build_auto_exp4_config(
    *,
    train_cfg: dict[str, Any],
    selected_task_name: str,
    model_entries: list[dict[str, Any]],
) -> dict[str, Any]:
    if not model_entries:
        raise ValueError("auto_exp_4 模式下没有可运行的 exp_4 模型")

    task_names = {entry["task_name"] for entry in model_entries}
    if len(task_names) != 1:
        raise ValueError("auto_exp_4 当前仅支持单任务批量运行")

    task_meta = resolve_model_task_meta(model_entries[0]["name"], selected_task_name)
    selection_alias = str(train_cfg.get("remark_metric_alias", "best_macro_f1")).strip() or "best_macro_f1"
    selection_meta = TRACKER_ALIAS_TO_META.get(selection_alias, TRACKER_ALIAS_TO_META["best_macro_f1"])

    return {
        "config_path": "inline:auto_exp_4",
        "goal": "围绕固定采样可能遗漏病灶证据的问题，比较固定采样、多次采样、全量图像、分层全量、Long-MIL 与 MambaMIL。",
        "task_name": task_meta["task_name"],
        "task_dir_name": task_meta["task_dir_name"],
        "output_dir_name": "auto_exp_4",
        "selection_alias": selection_alias,
        "selection_metric_name": str(selection_meta["metric_name"]),
        "selection_mode": str(selection_meta["mode"]),
        "result_source": "test_results",
        "stability_filter": {
            "enabled": True,
            "min_epochs_trained": 10,
            "max_final_gap": 0.05,
            "max_val_loss_rebound_ratio": 0.25,
        },
        "remark": {
            "focus": "汇总 TASK2 exp_4 结果，重点判断全量/长序列 MIL 是否缓解固定采样遗漏关键证据的问题",
            "include_model_evaluations": True,
            "include_stability_filter": True,
        },
        "models": model_entries,
        "run_test": True,
    }


def _build_exp5_roi_long_run_overrides() -> dict[str, Any]:
    run_overrides = _build_exp4_long_run_overrides()
    run_overrides.update(
        {
            "roi_enabled": True,
            "roi_max_crops_per_bag": 64,
            "roi_max_crops_per_source": 1,
            "roi_min_score": 0.0,
        }
    )
    return run_overrides


def build_auto_exp5_entries(
    *,
    selected_task_name: str,
    requested_names: list[str] | None = None,
) -> list[dict[str, Any]]:
    requested_name_set = set(requested_names or [])
    entries: list[dict[str, Any]] = []

    for model_name in AUTO_EXP5_ALLOWED_MODEL_NAMES:
        if requested_name_set and model_name not in requested_name_set:
            continue
        base_model_name = "long_mil"
        task_meta = resolve_model_task_meta(base_model_name, selected_task_name)
        entries.append(
            {
                "name": model_name,
                "display_name": model_name,
                "enabled": True,
                "base_model_name": base_model_name,
                "task_name": task_meta["task_name"],
                "task_dir_name": task_meta["task_dir_name"],
                "run_prefix": task_meta["run_prefix"],
                "model_params": {},
                "run_overrides": _build_exp5_roi_long_run_overrides(),
            }
        )

    if requested_names:
        missing = [name for name in requested_names if name not in AUTO_EXP5_ALLOWED_MODEL_NAMES]
        if missing:
            raise ValueError(f"auto_exp_5 模式下这些模型不属于 exp_5：{missing}")
        if not entries:
            raise ValueError("auto_exp_5 模式下没有可运行的 exp_5 模型")

    return entries


def build_auto_exp5_config(
    *,
    train_cfg: dict[str, Any],
    selected_task_name: str,
    model_entries: list[dict[str, Any]],
) -> dict[str, Any]:
    if not model_entries:
        raise ValueError("auto_exp_5 模式下没有可运行的 exp_5 模型")

    task_names = {entry["task_name"] for entry in model_entries}
    if len(task_names) != 1:
        raise ValueError("auto_exp_5 当前仅支持单任务批量运行")

    task_meta = resolve_model_task_meta(resolve_series_entry_model_name(model_entries[0]), selected_task_name)
    selection_alias = str(train_cfg.get("remark_metric_alias", "best_macro_f1")).strip() or "best_macro_f1"
    selection_meta = TRACKER_ALIAS_TO_META.get(selection_alias, TRACKER_ALIAS_TO_META["best_macro_f1"])
    raw_roi_cfg = train_cfg.get("auto_exp_5_roi", {}) or {}
    if not isinstance(raw_roi_cfg, dict):
        raise ValueError("auto_exp_5_roi 必须是字典")

    return {
        "config_path": "inline:auto_exp_5",
        "goal": "在 Long-MIL 长序列建模基础上，自动用 SAM2.1 生成 ROI mask/crop，并把 ROI crop 作为额外实例加入 bag。",
        "task_name": task_meta["task_name"],
        "task_dir_name": task_meta["task_dir_name"],
        "output_dir_name": "auto_exp_5",
        "selection_alias": selection_alias,
        "selection_metric_name": str(selection_meta["metric_name"]),
        "selection_mode": str(selection_meta["mode"]),
        "result_source": "test_results",
        "stability_filter": {
            "enabled": True,
            "min_epochs_trained": 10,
            "max_final_gap": 0.05,
            "max_val_loss_rebound_ratio": 0.25,
        },
        "remark": {
            "focus": "汇总 TASK2 exp_5 ROI-Long-MIL 结果，重点判断 SAM2 ROI crop 是否提升局部病灶证据利用。",
            "include_model_evaluations": True,
            "include_stability_filter": True,
        },
        "roi": raw_roi_cfg,
        "models": model_entries,
        "run_test": True,
    }


def _build_exp6_no_roi_run_overrides(*, original_instances: int) -> dict[str, Any]:
    original_instances = max(1, int(original_instances))
    max_batch_instances = 128 if original_instances <= 64 else 256
    run_overrides = _build_exp4_long_run_overrides()
    run_overrides.update(
        {
            "train_max_instances": original_instances,
            "eval_max_instances": original_instances,
            "train_max_batch_instances": max_batch_instances,
            "eval_max_batch_instances": max_batch_instances,
            "roi_enabled": False,
            "roi_max_crops_per_bag": 0,
        }
    )
    return run_overrides


def _build_exp6_long64_run_overrides() -> dict[str, Any]:
    return _build_exp6_no_roi_run_overrides(original_instances=64)


def _build_exp6_long128_run_overrides() -> dict[str, Any]:
    return _build_exp6_no_roi_run_overrides(original_instances=128)


def _build_exp6_roi_run_overrides(
    *,
    original_instances: int,
    roi_crops: int,
    roi_min_score: float = 0.0,
) -> dict[str, Any]:
    original_instances = max(1, int(original_instances))
    roi_crops = max(0, int(roi_crops))
    total_instances = original_instances + roi_crops
    run_overrides = _build_exp4_long_run_overrides()
    run_overrides.update(
        {
            "train_max_instances": total_instances,
            "eval_max_instances": total_instances,
            "train_max_batch_instances": 256,
            "eval_max_batch_instances": 256,
            "roi_enabled": roi_crops > 0,
            "roi_max_crops_per_bag": roi_crops,
            "roi_max_crops_per_source": 1,
            "roi_min_score": float(roi_min_score),
        }
    )
    return run_overrides


def _build_exp6_entry(
    *,
    name: str,
    display_name: str,
    base_model_name: str,
    selected_task_name: str,
    run_overrides: dict[str, Any],
    model_params: dict[str, Any] | None = None,
) -> dict[str, Any]:
    task_meta = resolve_model_task_meta(base_model_name, selected_task_name)
    return {
        "name": name,
        "display_name": display_name,
        "enabled": True,
        "base_model_name": base_model_name,
        "task_name": task_meta["task_name"],
        "task_dir_name": task_meta["task_dir_name"],
        "run_prefix": task_meta["run_prefix"],
        "model_params": dict(model_params or {}),
        "run_overrides": run_overrides,
    }


def build_auto_exp6_entries(
    *,
    selected_task_name: str,
    requested_names: list[str] | None = None,
) -> list[dict[str, Any]]:
    requested_name_set = set(requested_names or [])
    dual_model_params = {
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
        "roi_gate_init": -1.0,
        "use_type_embedding": True,
    }
    specs = [
        {
            "name": "exp6_long_mil_64_no_roi",
            "display_name": "Long-MIL 64 原图无 ROI 公平对照",
            "base_model_name": "long_mil",
            "run_overrides": _build_exp6_long64_run_overrides(),
            "model_params": {},
        },
        {
            "name": "exp6_roi_mix_64_32",
            "display_name": "Long-MIL 64 原图 + 32 ROI 混合输入",
            "base_model_name": "long_mil",
            "run_overrides": _build_exp6_roi_run_overrides(original_instances=64, roi_crops=32),
            "model_params": {},
        },
        {
            "name": "exp6_roi_mix_64_64",
            "display_name": "Long-MIL 64 原图 + 64 ROI 混合输入",
            "base_model_name": "long_mil",
            "run_overrides": _build_exp6_roi_run_overrides(original_instances=64, roi_crops=64),
            "model_params": {},
        },
        {
            "name": "exp6_roi_context_128_16",
            "display_name": "Long-MIL 保留 128 原图 + 16 ROI",
            "base_model_name": "long_mil",
            "run_overrides": _build_exp6_roi_run_overrides(original_instances=128, roi_crops=16),
            "model_params": {},
        },
        {
            "name": "exp6_roi_context_128_32",
            "display_name": "Long-MIL 保留 128 原图 + 32 ROI",
            "base_model_name": "long_mil",
            "run_overrides": _build_exp6_roi_run_overrides(original_instances=128, roi_crops=32),
            "model_params": {},
        },
        {
            "name": "exp6_roi_context_128_64",
            "display_name": "Long-MIL 保留 128 原图 + 64 ROI",
            "base_model_name": "long_mil",
            "run_overrides": _build_exp6_roi_run_overrides(original_instances=128, roi_crops=64),
            "model_params": {},
        },
        {
            "name": "exp6_roi_dual_128_16",
            "display_name": "原图-ROI 双路 Long-MIL 128+16",
            "base_model_name": "exp6_dual_stream_long_mil",
            "run_overrides": _build_exp6_roi_run_overrides(original_instances=128, roi_crops=16),
            "model_params": dual_model_params,
        },
        {
            "name": "exp6_roi_dual_128_32",
            "display_name": "原图-ROI 双路 Long-MIL 128+32",
            "base_model_name": "exp6_dual_stream_long_mil",
            "run_overrides": _build_exp6_roi_run_overrides(original_instances=128, roi_crops=32),
            "model_params": dual_model_params,
        },
        {
            "name": "exp6_roi_dual_128_64",
            "display_name": "原图-ROI 双路 Long-MIL 128+64",
            "base_model_name": "exp6_dual_stream_long_mil",
            "run_overrides": _build_exp6_roi_run_overrides(original_instances=128, roi_crops=64),
            "model_params": dual_model_params,
        },
        {
            "name": "exp6_roi_filter_96_32",
            "display_name": "ROI 质量阈值过滤 96 原图 + 32 ROI",
            "base_model_name": "exp6_dual_stream_long_mil",
            "run_overrides": _build_exp6_roi_run_overrides(original_instances=96, roi_crops=32, roi_min_score=0.88),
            "model_params": dual_model_params,
        },
        {
            "name": "exp6_roi_filter_128_32",
            "display_name": "ROI 质量阈值过滤 128 原图 + 32 ROI",
            "base_model_name": "exp6_dual_stream_long_mil",
            "run_overrides": _build_exp6_roi_run_overrides(original_instances=128, roi_crops=32, roi_min_score=0.88),
            "model_params": dual_model_params,
        },
        {
            "name": "exp6_roi_cons_128_32",
            "display_name": "双路 ROI + 全局/融合预测一致性正则",
            "base_model_name": "exp6_dual_stream_long_mil",
            "run_overrides": _build_exp6_roi_run_overrides(original_instances=128, roi_crops=32),
            "model_params": {
                **dual_model_params,
                "view_consistency_weight": 0.1,
                "attention_entropy_weight": 0.001,
            },
        },
    ]

    entries = [
        _build_exp6_entry(
            name=str(spec["name"]),
            display_name=str(spec["display_name"]),
            base_model_name=str(spec["base_model_name"]),
            selected_task_name=selected_task_name,
            run_overrides=dict(spec["run_overrides"]),
            model_params=dict(spec.get("model_params", {})),
        )
        for spec in specs
        if not requested_name_set or str(spec["name"]) in requested_name_set
    ]

    if requested_names:
        missing = [name for name in requested_names if name not in AUTO_EXP6_ALLOWED_MODEL_NAMES]
        if missing:
            raise ValueError(f"auto_exp_6 模式下这些实验名不属于 exp_6：{missing}")
        if not entries:
            raise ValueError("auto_exp_6 模式下没有可运行的 exp_6 实验")

    return entries


def build_auto_exp6_config(
    *,
    train_cfg: dict[str, Any],
    selected_task_name: str,
    model_entries: list[dict[str, Any]],
) -> dict[str, Any]:
    if not model_entries:
        raise ValueError("auto_exp_6 模式下没有可运行的 exp_6 实验")

    task_names = {entry["task_name"] for entry in model_entries}
    if len(task_names) != 1:
        raise ValueError("auto_exp_6 当前仅支持单任务批量运行")

    task_meta = resolve_model_task_meta(resolve_series_entry_model_name(model_entries[0]), selected_task_name)
    selection_alias = str(train_cfg.get("remark_metric_alias", "best_macro_f1")).strip() or "best_macro_f1"
    selection_meta = TRACKER_ALIAS_TO_META.get(selection_alias, TRACKER_ALIAS_TO_META["best_macro_f1"])
    raw_roi_cfg = train_cfg.get("auto_exp_6_roi", train_cfg.get("auto_exp_5_roi", {})) or {}
    if not isinstance(raw_roi_cfg, dict):
        raise ValueError("auto_exp_6_roi 必须是字典")

    return {
        "config_path": "inline:auto_exp_6",
        "goal": "围绕 exp_6 计划执行 Long-MIL 64 公平对照、保留上下文的 ROI 追加实验、双路 ROI 融合和一致性正则实验。",
        "task_name": task_meta["task_name"],
        "task_dir_name": task_meta["task_dir_name"],
        "output_dir_name": "auto_exp_6",
        "selection_alias": selection_alias,
        "selection_metric_name": str(selection_meta["metric_name"]),
        "selection_mode": str(selection_meta["mode"]),
        "result_source": "test_results",
        "stability_filter": {
            "enabled": True,
            "min_epochs_trained": 10,
            "max_final_gap": 0.05,
            "max_val_loss_rebound_ratio": 0.25,
        },
        "remark": {
            "focus": "汇总 TASK2 exp_6 一键实验结果，重点比较 64 原图公平对照、ROI 追加数量、双路 ROI 融合和一致性正则。",
            "include_model_evaluations": True,
            "include_stability_filter": True,
        },
        "roi": raw_roi_cfg,
        "models": model_entries,
        "run_test": True,
    }


def build_auto_exp7_entries(
    *,
    selected_task_name: str,
    requested_names: list[str] | None = None,
) -> list[dict[str, Any]]:
    requested_name_set = set(requested_names or [])
    specs = [
        {
            "name": "exp6_long_mil_64_no_roi",
            "display_name": "Long-MIL 64 原图无 ROI 对照",
            "base_model_name": "long_mil",
            "run_overrides": _build_exp6_long64_run_overrides(),
            "model_params": {},
        },
        {
            "name": "exp6_roi_mix_64_16",
            "display_name": "ROI mix：64 原图 + 16 ROI",
            "base_model_name": "long_mil",
            "run_overrides": _build_exp6_roi_run_overrides(original_instances=64, roi_crops=16),
            "model_params": {},
        },
        {
            "name": "exp6_roi_mix_128_16",
            "display_name": "ROI mix：128 原图 + 16 ROI",
            "base_model_name": "long_mil",
            "run_overrides": _build_exp6_roi_run_overrides(original_instances=128, roi_crops=16),
            "model_params": {},
        },
        {
            "name": "exp6_roi_context_64_16",
            "display_name": "ROI context：64 原图 + 16 ROI",
            "base_model_name": "long_mil",
            "run_overrides": _build_exp6_roi_run_overrides(original_instances=64, roi_crops=16),
            "model_params": {},
        },
        {
            "name": "exp6_roi_context_64_32",
            "display_name": "ROI context：64 原图 + 32 ROI",
            "base_model_name": "long_mil",
            "run_overrides": _build_exp6_roi_run_overrides(original_instances=64, roi_crops=32),
            "model_params": {},
        },
        {
            "name": "exp6_roi_context_64_64",
            "display_name": "ROI context：64 原图 + 64 ROI",
            "base_model_name": "long_mil",
            "run_overrides": _build_exp6_roi_run_overrides(original_instances=64, roi_crops=64),
            "model_params": {},
        },
        {
            "name": "exp6_long_mil_128_no_roi",
            "display_name": "Long-MIL 128 原图无 ROI 对照",
            "base_model_name": "long_mil",
            "run_overrides": _build_exp6_long128_run_overrides(),
            "model_params": {},
        },
    ]

    entries = [
        _build_exp6_entry(
            name=str(spec["name"]),
            display_name=str(spec["display_name"]),
            base_model_name=str(spec["base_model_name"]),
            selected_task_name=selected_task_name,
            run_overrides=dict(spec["run_overrides"]),
            model_params=dict(spec.get("model_params", {})),
        )
        for spec in specs
        if not requested_name_set or str(spec["name"]) in requested_name_set
    ]

    if requested_names:
        missing = [name for name in requested_names if name not in AUTO_EXP7_ALLOWED_MODEL_NAMES]
        if missing:
            raise ValueError(f"auto_exp_7 模式下这些实验名不属于 exp_7：{missing}")
        if not entries:
            raise ValueError("auto_exp_7 模式下没有可运行的 exp_7 实验")

    return entries


def build_auto_exp7_config(
    *,
    train_cfg: dict[str, Any],
    selected_task_name: str,
    model_entries: list[dict[str, Any]],
) -> dict[str, Any]:
    if not model_entries:
        raise ValueError("auto_exp_7 模式下没有可运行的 exp_7 实验")

    task_names = {entry["task_name"] for entry in model_entries}
    if len(task_names) != 1:
        raise ValueError("auto_exp_7 当前仅支持单任务批量运行")

    task_meta = resolve_model_task_meta(resolve_series_entry_model_name(model_entries[0]), selected_task_name)
    selection_alias = str(train_cfg.get("remark_metric_alias", "best_macro_f1")).strip() or "best_macro_f1"
    selection_meta = TRACKER_ALIAS_TO_META.get(selection_alias, TRACKER_ALIAS_TO_META["best_macro_f1"])
    raw_roi_cfg = train_cfg.get("auto_exp_7_roi", train_cfg.get("auto_exp_6_roi", train_cfg.get("auto_exp_5_roi", {}))) or {}
    if not isinstance(raw_roi_cfg, dict):
        raise ValueError("auto_exp_7_roi 必须是字典")

    return {
        "config_path": "inline:auto_exp_7",
        "goal": "围绕 exp_7 计划比较 64/128 no-ROI 对照、ROI mix 输入和 ROI context 输入，确定加入 ROI 弱监督时更稳的 Long-MIL 输入组织方式。",
        "task_name": task_meta["task_name"],
        "task_dir_name": task_meta["task_dir_name"],
        "output_dir_name": "auto_exp_7",
        "selection_alias": selection_alias,
        "selection_metric_name": str(selection_meta["metric_name"]),
        "selection_mode": str(selection_meta["mode"]),
        "result_source": "test_results",
        "stability_filter": {
            "enabled": True,
            "min_epochs_trained": 10,
            "max_final_gap": 0.05,
            "max_val_loss_rebound_ratio": 0.25,
        },
        "remark": {
            "focus": "汇总 TASK2 exp_7 一键实验结果，重点比较 64/128 no-ROI 对照、ROI mix 与 ROI context 在不同原图/ROI 数量下的测试表现。",
            "include_model_evaluations": True,
            "include_stability_filter": True,
        },
        "roi": raw_roi_cfg,
        "models": model_entries,
        "run_test": True,
    }


def _format_exp8_modality_fields(fields: list[str] | tuple[str, ...]) -> str:
    return "image" if not fields else "image+" + "+".join(fields)


def _build_exp8_run_overrides(
    *,
    fields: list[str] | tuple[str, ...],
    leakage_note: str,
    shuffle_fields: list[str] | tuple[str, ...] | None = None,
    shuffle_apply_to: str = "none",
) -> dict[str, Any]:
    run_overrides = _build_exp6_long64_run_overrides()
    modality_fields = _format_exp8_modality_fields(fields)
    run_overrides.update(
        {
            "structured_fields": list(fields),
            "structured_shuffle_fields": list(shuffle_fields or []),
            "structured_shuffle_apply_to": str(shuffle_apply_to),
            "structured_shuffle_seed": 20260518,
            "modality_level": "strict_deploy",
            "modality_fields": modality_fields,
            "inference_inputs": "image" if not fields else "image+structured",
            "leakage_note": leakage_note,
        }
    )
    return run_overrides


def _build_exp8_model_params(fields: list[str] | tuple[str, ...]) -> dict[str, Any]:
    has_structured = bool(fields)
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
        "structured_fields": list(fields),
        "structured_field_embed_dim": 64,
        "structured_dropout": 0.2 if has_structured else 0.0,
        "modality_dropout": 0.15 if has_structured else 0.0,
        "structured_gate_l1_weight": 0.001 if has_structured else 0.0,
    }


def _build_exp8_entry(
    *,
    name: str,
    display_name: str,
    fields: list[str] | tuple[str, ...],
    selected_task_name: str,
    leakage_note: str,
    seed: int | None = None,
    shuffle_fields: list[str] | tuple[str, ...] | None = None,
    shuffle_apply_to: str = "none",
) -> dict[str, Any]:
    task_meta = resolve_model_task_meta("exp8_structured_late_gate_mil", selected_task_name)
    entry = {
        "name": name,
        "display_name": display_name,
        "enabled": True,
        "base_model_name": "exp8_structured_late_gate_mil",
        "task_name": task_meta["task_name"],
        "task_dir_name": task_meta["task_dir_name"],
        "run_prefix": task_meta["run_prefix"],
        "model_params": _build_exp8_model_params(fields),
        "run_overrides": _build_exp8_run_overrides(
            fields=fields,
            leakage_note=leakage_note,
            shuffle_fields=shuffle_fields,
            shuffle_apply_to=shuffle_apply_to,
        ),
    }
    if seed is not None:
        entry["seed"] = int(seed)
    return entry


def build_auto_exp8_mm_ablation_entries(
    *,
    selected_task_name: str,
    requested_names: list[str] | None = None,
    base_seed: int | None = None,
) -> list[dict[str, Any]]:
    requested_name_set = set(requested_names or [])
    all_fields = ["reportTitle", "age", "sex", "hp", "operationValue"]
    strict_note = "结构化短字段消融；不使用 watchResult，避免最终诊断文本泄漏。"
    specs = [
        ("exp8_mm_ablation_image_baseline", "图像 Long-MIL 64 原图基线", []),
        ("exp8_mm_ablation_age", "图像 + age", ["age"]),
        ("exp8_mm_ablation_age_sex", "图像 + age + sex", ["age", "sex"]),
        ("exp8_mm_ablation_age_sex_hp", "图像 + age + sex + hp", ["age", "sex", "hp"]),
        ("exp8_mm_ablation_reportTitle", "图像 + reportTitle", ["reportTitle"]),
        ("exp8_mm_ablation_operationValue", "图像 + operationValue", ["operationValue"]),
        ("exp8_mm_ablation_title_operation", "图像 + reportTitle + operationValue", ["reportTitle", "operationValue"]),
        ("exp8_mm_ablation_all_structured", "图像 + 全部结构化字段", all_fields),
        (
            "exp8_mm_ablation_all_without_title",
            "全字段去掉 reportTitle",
            ["age", "sex", "hp", "operationValue"],
        ),
        (
            "exp8_mm_ablation_all_without_operation",
            "全字段去掉 operationValue",
            ["reportTitle", "age", "sex", "hp"],
        ),
        (
            "exp8_mm_ablation_all_without_hp",
            "全字段去掉 hp",
            ["reportTitle", "age", "sex", "operationValue"],
        ),
        (
            "exp8_mm_ablation_all_without_age",
            "全字段去掉 age",
            ["reportTitle", "sex", "hp", "operationValue"],
        ),
    ]
    shuffle_specs = [
        (
            "exp8_mm_ablation_all_shuffle_title_test",
            "全字段训练，测试集置乱 reportTitle",
            all_fields,
            ["reportTitle"],
            "test",
        ),
        (
            "exp8_mm_ablation_all_shuffle_operation_test",
            "全字段训练，测试集置乱 operationValue",
            all_fields,
            ["operationValue"],
            "test",
        ),
        (
            "exp8_mm_ablation_all_shuffle_title_operation_test",
            "全字段训练，测试集同时置乱 reportTitle 与 operationValue",
            all_fields,
            ["reportTitle", "operationValue"],
            "test",
        ),
        (
            "exp8_mm_ablation_shuffle_title_train",
            "全字段训练/验证/测试均置乱 reportTitle",
            all_fields,
            ["reportTitle"],
            "all",
        ),
        (
            "exp8_mm_ablation_shuffle_operation_train",
            "全字段训练/验证/测试均置乱 operationValue",
            all_fields,
            ["operationValue"],
            "all",
        ),
    ]

    entries = [
        _build_exp8_entry(
            name=name,
            display_name=display_name,
            fields=fields,
            selected_task_name=selected_task_name,
            leakage_note=strict_note,
            seed=base_seed,
        )
        for name, display_name, fields in specs
        if not requested_name_set or name in requested_name_set
    ]
    entries.extend(
        _build_exp8_entry(
            name=name,
            display_name=display_name,
            fields=fields,
            selected_task_name=selected_task_name,
            leakage_note=(
                f"{strict_note} 本实验在 {shuffle_apply_to} split 内置乱 "
                f"{','.join(shuffle_fields)}，用于审计字段依赖。"
            ),
            shuffle_fields=shuffle_fields,
            shuffle_apply_to=shuffle_apply_to,
            seed=base_seed,
        )
        for name, display_name, fields, shuffle_fields, shuffle_apply_to in shuffle_specs
        if not requested_name_set or name in requested_name_set
    )

    if requested_names:
        missing = [
            name
            for name in requested_names
            if name not in AUTO_EXP8_MM_ABLATION_ALLOWED_MODEL_NAMES
        ]
        if missing:
            raise ValueError(f"auto_exp_8_mm_ablation 模式下这些实验名不属于 exp_8_mm_ablation：{missing}")
        if not entries:
            raise ValueError("auto_exp_8_mm_ablation 模式下没有可运行的实验")

    return entries


def build_auto_exp8_mm_ablation_config(
    *,
    train_cfg: dict[str, Any],
    selected_task_name: str,
    model_entries: list[dict[str, Any]],
) -> dict[str, Any]:
    if not model_entries:
        raise ValueError("auto_exp_8_mm_ablation 模式下没有可运行的实验")

    task_names = {entry["task_name"] for entry in model_entries}
    if len(task_names) != 1:
        raise ValueError("auto_exp_8_mm_ablation 当前仅支持单任务批量运行")

    task_meta = resolve_model_task_meta(resolve_series_entry_model_name(model_entries[0]), selected_task_name)
    selection_alias = str(train_cfg.get("remark_metric_alias", "best_macro_f1")).strip() or "best_macro_f1"
    selection_meta = TRACKER_ALIAS_TO_META.get(selection_alias, TRACKER_ALIAS_TO_META["best_macro_f1"])

    return {
        "config_path": "inline:auto_exp_8_mm_ablation",
        "goal": "围绕 exp_8_mm_ablation 执行 TASK2 图像 + 结构化短字段多模态消融；标签关系模块使用 TASK1 同款 label_hypergraph，并审计 reportTitle、operationValue 等流程相关字段依赖。",
        "task_name": task_meta["task_name"],
        "task_dir_name": task_meta["task_dir_name"],
        "experiment_dir_name": "",
        "output_dir_name": "exp_mm_ablation_hypergraph",
        "selection_alias": selection_alias,
        "selection_metric_name": str(selection_meta["metric_name"]),
        "selection_mode": str(selection_meta["mode"]),
        "result_source": "test_results",
        "stability_filter": {
            "enabled": True,
            "min_epochs_trained": 10,
            "max_final_gap": 0.05,
            "max_val_loss_rebound_ratio": 0.25,
        },
        "remark": {
            "focus": "汇总 exp_8_mm_ablation hyperGraph 结构化字段消融结果，重点比较 image-only、低风险字段、全字段、去字段与置乱审计。",
            "include_model_evaluations": True,
            "include_stability_filter": True,
        },
        "models": model_entries,
        "run_test": True,
    }


def _build_exp8_main_entry(
    *,
    name: str,
    display_name: str,
    selected_task_name: str,
    run_overrides: dict[str, Any],
    metadata: dict[str, Any],
    seed: int | None = None,
) -> dict[str, Any]:
    task_meta = resolve_model_task_meta(name, selected_task_name)
    entry = {
        "name": name,
        "display_name": display_name,
        "enabled": True,
        "base_model_name": name,
        "task_name": task_meta["task_name"],
        "task_dir_name": task_meta["task_dir_name"],
        "run_prefix": task_meta["run_prefix"],
        "model_params": {},
        "run_overrides": run_overrides,
        "metadata": metadata,
    }
    if seed is not None:
        entry["seed"] = int(seed)
    return entry


def _build_exp8_non_structured_run_overrides(
    *,
    modality_level: str,
    modality_fields: str,
    inference_inputs: str,
    leakage_note: str,
) -> dict[str, Any]:
    run_overrides = _build_exp6_long64_run_overrides()
    run_overrides.update(
        {
            "structured_fields": [],
            "modality_level": modality_level,
            "modality_fields": modality_fields,
            "inference_inputs": inference_inputs,
            "leakage_note": leakage_note,
        }
    )
    return run_overrides


def build_auto_exp8_entries(
    *,
    selected_task_name: str,
    requested_names: list[str] | None = None,
    base_seed: int | None = None,
) -> list[dict[str, Any]]:
    requested_name_set = set(requested_names or [])
    specs = [
        (
            "exp8_mm_struct_late_gate",
            "结构化字段 label-wise gated late fusion",
            _build_exp8_run_overrides(
                fields=["reportTitle", "age", "sex", "operationValue"],
                leakage_note=(
                    "exp_8 实验1：基于 exp_8_mm_ablation 非置乱最佳正式候选；"
                    "使用 reportTitle、age、sex、operationValue，不使用 watchResult。"
                    "reportTitle/operationValue 属于流程相关字段，需在论文中说明检查路径代理风险。"
                ),
            ),
            {
                "modality_level": "strict_deploy",
                "inference_inputs": "image+structured",
                "summary_name": "exp8_mm_struct_late_gate",
            },
        ),
        (
            "exp8_mm_label_proto_graph",
            "固定标签文本原型 + label graph",
            _build_exp8_non_structured_run_overrides(
                modality_level="fixed_proto",
                modality_fields="image+fixed_label_proto",
                inference_inputs="image+fixed_proto",
                leakage_note="exp_8 实验2：只使用固定标签文本原型，不输入个体报告文本或 watchResult。",
            ),
            {
                "modality_level": "fixed_proto",
                "inference_inputs": "image+fixed_proto",
                "summary_name": "exp8_mm_label_proto_graph",
            },
        ),
        (
            "exp8_mm_text_contrast_distill",
            "训练期图文对比蒸馏 image-only student",
            _build_exp8_non_structured_run_overrides(
                modality_level="train_time_distill",
                modality_fields="image+safe_text(train_only)",
                inference_inputs="image",
                leakage_note="exp_8 实验3：训练期使用安全报告字段做图文对齐，测试 logits 保持 image-only；不使用 watchResult。",
            ),
            {
                "modality_level": "train_time_distill",
                "inference_inputs": "image",
                "summary_name": "exp8_mm_text_contrast_distill",
            },
        ),
        (
            "exp8_mm_watch_cross_attn",
            "watch 文本与图像 token cross-attention",
            _build_exp8_non_structured_run_overrides(
                modality_level="report_assist",
                modality_fields="image+watch",
                inference_inputs="image+watch",
                leakage_note="exp_8 实验4：测试阶段输入 watch，属于报告辅助任务；不输入 watchResult。",
            ),
            {
                "modality_level": "report_assist",
                "inference_inputs": "image+watch",
                "summary_name": "exp8_mm_watch_cross_attn",
            },
        ),
        (
            "exp8_mm_text_guided_top64_align",
            "文本引导 64 图实例重加权 + 图文对齐",
            _build_exp8_non_structured_run_overrides(
                modality_level="report_assist",
                modality_fields="image+text_guided_top64",
                inference_inputs="image+text_guided_fields",
                leakage_note="exp_8 实验5：使用低到中风险文本字段进行实例重加权与对齐；当前实现为训练内 64 图重加权，不使用 watchResult。",
            ),
            {
                "modality_level": "report_assist",
                "inference_inputs": "image+text_guided_fields",
                "summary_name": "exp8_mm_text_guided_top64_align",
            },
        ),
    ]
    entries = [
        _build_exp8_main_entry(
            name=name,
            display_name=display_name,
            selected_task_name=selected_task_name,
            run_overrides=run_overrides,
            metadata=metadata,
            seed=(base_seed + index if base_seed is not None else None),
        )
        for index, (name, display_name, run_overrides, metadata) in enumerate(specs)
        if not requested_name_set or name in requested_name_set
    ]
    if requested_names:
        missing = [name for name in requested_names if name not in AUTO_EXP8_ALLOWED_MODEL_NAMES]
        if missing:
            raise ValueError(f"auto_exp_8 模式下这些实验名不属于 exp_8：{missing}")
        if not entries:
            raise ValueError("auto_exp_8 模式下没有可运行的实验")
    return entries


def build_auto_exp8_config(
    *,
    train_cfg: dict[str, Any],
    selected_task_name: str,
    model_entries: list[dict[str, Any]],
) -> dict[str, Any]:
    if not model_entries:
        raise ValueError("auto_exp_8 模式下没有可运行的实验")
    task_names = {entry["task_name"] for entry in model_entries}
    if len(task_names) != 1:
        raise ValueError("auto_exp_8 当前仅支持单任务批量运行")

    task_meta = resolve_model_task_meta(resolve_series_entry_model_name(model_entries[0]), selected_task_name)
    selection_alias = str(train_cfg.get("remark_metric_alias", "best_macro_f1")).strip() or "best_macro_f1"
    selection_meta = TRACKER_ALIAS_TO_META.get(selection_alias, TRACKER_ALIAS_TO_META["best_macro_f1"])
    return {
        "config_path": "inline:auto_exp_8",
        "goal": "执行 exp_8.md 中优先级前 5 个 TASK2 多模态融合实验。",
        "task_name": task_meta["task_name"],
        "task_dir_name": task_meta["task_dir_name"],
        "experiment_dir_name": "exp_8",
        "output_dir_name": "",
        "selection_alias": selection_alias,
        "selection_metric_name": str(selection_meta["metric_name"]),
        "selection_mode": str(selection_meta["mode"]),
        "result_source": "test_results",
        "stability_filter": {
            "enabled": True,
            "min_epochs_trained": 10,
            "max_final_gap": 0.05,
            "max_val_loss_rebound_ratio": 0.25,
        },
        "remark": {
            "focus": "汇总 exp_8 五个多模态主实验：结构化 late gate、固定标签原型、训练期图文蒸馏、watch cross-attention、文本引导 64 图对齐。",
            "include_model_evaluations": True,
            "include_stability_filter": True,
        },
        "models": model_entries,
        "run_test": True,
    }


def _build_exp9_watch_model_params(**overrides: Any) -> dict[str, Any]:
    params = {
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
        "image_aux_weight": 0.5,
    }
    params.update(overrides)
    return params


def _build_exp9_watch_run_overrides(
    *,
    instances: int = 64,
    modality_fields: str = "image+watch",
    inference_inputs: str = "image+watch",
    leakage_note: str,
    ablation_group: str,
) -> dict[str, Any]:
    run_overrides = _build_exp6_no_roi_run_overrides(original_instances=int(instances))
    run_overrides.update(
        {
            "structured_fields": [],
            "modality_level": "report_assist_ablation",
            "modality_fields": modality_fields,
            "inference_inputs": inference_inputs,
            "leakage_note": leakage_note,
            "ablation_group": ablation_group,
        }
    )
    return run_overrides


def _build_exp9_watch_entry(
    *,
    name: str,
    display_name: str,
    base_model_name: str,
    selected_task_name: str,
    model_params: dict[str, Any],
    run_overrides: dict[str, Any],
    metadata: dict[str, Any],
    seed: int | None = None,
) -> dict[str, Any]:
    task_meta = resolve_model_task_meta(base_model_name, selected_task_name)
    entry = {
        "name": name,
        "display_name": display_name,
        "enabled": True,
        "base_model_name": base_model_name,
        "task_name": task_meta["task_name"],
        "task_dir_name": task_meta["task_dir_name"],
        "run_prefix": task_meta["run_prefix"],
        "model_params": model_params,
        "run_overrides": run_overrides,
        "metadata": metadata,
    }
    if seed is not None:
        entry["seed"] = int(seed)
    return entry


def build_auto_exp9_ablation_entries(
    *,
    selected_task_name: str,
    requested_names: list[str] | None = None,
    base_seed: int | None = None,
) -> list[dict[str, Any]]:
    requested_name_set = set(requested_names or [])
    entries: list[dict[str, Any]] = []
    base_note = "exp_9_ablation：基于 exp8_mm_watch_cross_attn 的 watch 报告辅助模型消融；不使用 watchResult。"

    def add_entry(
        *,
        name: str,
        display_name: str,
        base_model_name: str,
        instances: int,
        ablation_group: str,
        modality_fields: str = "image+watch",
        inference_inputs: str = "image+watch",
        model_overrides: dict[str, Any] | None = None,
        leakage_note: str | None = None,
    ) -> None:
        if requested_name_set and name not in requested_name_set:
            return
        entries.append(
            _build_exp9_watch_entry(
                name=name,
                display_name=display_name,
                base_model_name=base_model_name,
                selected_task_name=selected_task_name,
                model_params=_build_exp9_watch_model_params(**(model_overrides or {})),
                run_overrides=_build_exp9_watch_run_overrides(
                    instances=instances,
                    modality_fields=modality_fields,
                    inference_inputs=inference_inputs,
                    leakage_note=leakage_note or base_note,
                    ablation_group=ablation_group,
                ),
                metadata={
                    "summary_name": name,
                    "ablation_group": ablation_group,
                    "instances": int(instances),
                    "inference_inputs": inference_inputs,
                },
                seed=base_seed,
            )
        )

    for instances in AUTO_EXP9_ABLATION_INSTANCE_VALUES:
        add_entry(
            name=f"exp9_watch_instances_{instances}",
            display_name=f"watch cross-attn 完整模型 | {instances} 图",
            base_model_name="exp8_mm_watch_cross_attn",
            instances=instances,
            ablation_group="instance_count",
            leakage_note=f"{base_note} 图像数量消融：最多输入 {instances} 张原图。",
        )

    for instances in AUTO_EXP9_ABLATION_INSTANCE_VALUES:
        add_entry(
            name=f"exp9_watch_no_context_instances_{instances}",
            display_name=f"去掉位置编码和 Transformer context | {instances} 图",
            base_model_name="exp9_watch_no_context",
            instances=instances,
            ablation_group="no_position_transformer_context",
            leakage_note=f"{base_note} 去掉位置编码和 Transformer context encoder，最多输入 {instances} 张原图。",
        )

    add_entry(
        name="exp9_watch_no_text",
        display_name="去掉 watch 文本，仅保留图像分支",
        base_model_name="exp9_watch_no_text",
        instances=64,
        ablation_group="watch_text",
        modality_fields="image",
        inference_inputs="image",
        leakage_note=f"{base_note} 去掉 watch 文本输入，用于估计 watch 文本整体贡献。",
    )
    add_entry(
        name="exp9_watch_label_graph",
        display_name="label_graph 替代 label_hypergraph",
        base_model_name="exp8_mm_watch_cross_attn",
        instances=64,
        ablation_group="label_relation",
        model_overrides={"label_graph_type": "learnable"},
        leakage_note=f"{base_note} 将 label_hypergraph 换回普通 learnable label_graph。",
    )
    add_entry(
        name="exp9_watch_no_cross_attn_pool_fusion",
        display_name="去掉 cross-attention，改用 watch pooled late fusion",
        base_model_name="exp9_watch_no_cross_attn_pool_fusion",
        instances=64,
        ablation_group="text_fusion",
        leakage_note=f"{base_note} 去掉 label-wise cross-attention，改用 watch pooled embedding late fusion。",
    )
    add_entry(
        name="exp9_watch_cross_attn_no_gate",
        display_name="保留 cross-attention，去掉 gate",
        base_model_name="exp9_watch_cross_attn_no_gate",
        instances=64,
        ablation_group="text_gate",
        leakage_note=f"{base_note} 保留 watch cross-attention，但去掉 gate，直接注入文本表征。",
    )
    add_entry(
        name="exp9_watch_cross_attn_no_image_aux",
        display_name="去掉 image_aux 辅助损失",
        base_model_name="exp8_mm_watch_cross_attn",
        instances=64,
        ablation_group="aux_loss",
        model_overrides={"image_aux_weight": 0.0},
        leakage_note=f"{base_note} 将 image_aux_weight 置为 0，检验图像分支辅助约束的作用。",
    )

    if requested_names:
        missing = [name for name in requested_names if name not in AUTO_EXP9_ABLATION_ALLOWED_MODEL_NAMES]
        if missing:
            raise ValueError(f"auto_exp_9_ablation 模式下这些实验名不属于 exp_9_ablation：{missing}")
        if not entries:
            raise ValueError("auto_exp_9_ablation 模式下没有可运行的实验")

    return entries


def build_auto_exp9_ablation_config(
    *,
    train_cfg: dict[str, Any],
    selected_task_name: str,
    model_entries: list[dict[str, Any]],
) -> dict[str, Any]:
    if not model_entries:
        raise ValueError("auto_exp_9_ablation 模式下没有可运行的实验")
    task_names = {entry["task_name"] for entry in model_entries}
    if len(task_names) != 1:
        raise ValueError("auto_exp_9_ablation 当前仅支持单任务批量运行")

    task_meta = resolve_model_task_meta(resolve_series_entry_model_name(model_entries[0]), selected_task_name)
    selection_alias = str(train_cfg.get("remark_metric_alias", "best_macro_f1")).strip() or "best_macro_f1"
    selection_meta = TRACKER_ALIAS_TO_META.get(selection_alias, TRACKER_ALIAS_TO_META["best_macro_f1"])
    return {
        "config_path": "inline:auto_exp_9_ablation",
        "goal": "围绕 exp8_mm_watch_cross_attn 执行 TASK2 exp_9_ablation，消融图像数量、位置编码/Transformer context、watch 文本、标签关系、文本融合、gate 与 image_aux。",
        "task_name": task_meta["task_name"],
        "task_dir_name": task_meta["task_dir_name"],
        "experiment_dir_name": "exp_9_ablation",
        "output_dir_name": "",
        "selection_alias": selection_alias,
        "selection_metric_name": str(selection_meta["metric_name"]),
        "selection_mode": str(selection_meta["mode"]),
        "result_source": "test_results",
        "stability_filter": {
            "enabled": True,
            "min_epochs_trained": 10,
            "max_final_gap": 0.05,
            "max_val_loss_rebound_ratio": 0.25,
        },
        "remark": {
            "focus": "汇总 exp_9_ablation：基于 exp8_mm_watch_cross_attn 的 17 组消融，重点比较图像数量、序列上下文、watch 文本、标签关系、文本融合方式、gate 和 image_aux。",
            "include_model_evaluations": True,
            "include_stability_filter": True,
        },
        "models": model_entries,
        "run_test": True,
    }


def build_auto_exp11_module_ablation_entries(
    *,
    selected_task_name: str,
    requested_names: list[str] | None = None,
    base_seed: int | None = None,
) -> list[dict[str, Any]]:
    requested_name_set = set(requested_names or [])
    entries: list[dict[str, Any]] = []
    module_names = {
        "1": "位置编码+上下文编码器",
        "2": "超图学习",
        "3": "交叉注意力",
        "4": "门控机制",
    }

    for combo in AUTO_EXP11_MODULE_ABLATION_COMBINATIONS:
        name = f"exp11_module_ablation_{combo}"
        if requested_name_set and name not in requested_name_set:
            continue
        active_modules = set() if combo == "none" else set(combo)
        use_m1 = "1" in active_modules
        use_m2 = "2" in active_modules
        use_m3 = "3" in active_modules
        use_m4 = "4" in active_modules
        fusion_mode = "cross_attention" if use_m3 else ("pooled" if use_m4 else "none")
        uses_watch = fusion_mode != "none"
        data_parallel_device_ids = [0, 1, 2]
        display_modules = "Baseline" if combo == "none" else " + ".join(
            f"M{module_id}" for module_id in combo
        )
        active_description = "、".join(module_names[module_id] for module_id in combo) if combo != "none" else "无新增模块"
        model_params = _build_exp9_watch_model_params(
            use_context_encoder=use_m1,
            label_graph_type="label_hypergraph" if use_m2 else "learnable",
            watch_fusion_mode=fusion_mode,
            use_text_gate=use_m4,
            image_aux_weight=0.5,
        )
        run_overrides = _build_exp9_watch_run_overrides(
            instances=64,
            modality_fields="image+watch" if uses_watch else "image",
            inference_inputs="image+watch" if uses_watch else "image",
            leakage_note=(
                f"exp11_module_ablation：四模块全因子消融缺失组合 {combo}；"
                f"启用模块为{active_description}；固定64张原图，image_aux_weight=0.5，与 exp_9 对齐。"
            ),
            ablation_group="module_combination",
        )
        run_overrides["data_parallel_device_ids"] = data_parallel_device_ids
        run_overrides["num_workers"] = 6
        run_overrides["loader_prefetch_factor"] = 2
        run_overrides["persistent_workers"] = True
        task_meta = resolve_model_task_meta("exp11_module_ablation", selected_task_name)
        entry = {
            "name": name,
            "display_name": f"{display_modules} | 组合={combo}",
            "enabled": True,
            "base_model_name": "exp11_module_ablation",
            "task_name": task_meta["task_name"],
            "task_dir_name": task_meta["task_dir_name"],
            "run_prefix": task_meta["run_prefix"],
            "model_params": model_params,
            "run_overrides": run_overrides,
            "metadata": {
                "summary_name": name,
                "ablation_group": "module_combination",
                "module_combo": combo,
                "use_m1_context": use_m1,
                "use_m2_hypergraph": use_m2,
                "use_m3_cross_attention": use_m3,
                "use_m4_gate": use_m4,
                "instances": 64,
                "gpu_count": len(data_parallel_device_ids),
                "data_parallel_device_ids": data_parallel_device_ids,
                "inference_inputs": "image+watch" if uses_watch else "image",
            },
        }
        if base_seed is not None:
            entry["seed"] = int(base_seed)
        entries.append(entry)

    if requested_names:
        missing = [
            name for name in requested_names
            if name not in AUTO_EXP11_MODULE_ABLATION_ALLOWED_MODEL_NAMES
        ]
        if missing:
            raise ValueError(f"auto_exp_11_module_ablation 模式下存在未知实验名：{missing}")
        if not entries:
            raise ValueError("auto_exp_11_module_ablation 模式下没有可运行的实验")
    return entries


def build_auto_exp11_module_ablation_config(
    *,
    train_cfg: dict[str, Any],
    selected_task_name: str,
    model_entries: list[dict[str, Any]],
) -> dict[str, Any]:
    if not model_entries:
        raise ValueError("auto_exp_11_module_ablation 模式下没有可运行的实验")
    task_names = {entry["task_name"] for entry in model_entries}
    if len(task_names) != 1:
        raise ValueError("auto_exp_11_module_ablation 当前仅支持单任务批量运行")

    task_meta = resolve_model_task_meta(resolve_series_entry_model_name(model_entries[0]), selected_task_name)
    selection_alias = str(train_cfg.get("remark_metric_alias", "best_macro_f1")).strip() or "best_macro_f1"
    selection_meta = TRACKER_ALIAS_TO_META.get(selection_alias, TRACKER_ALIAS_TO_META["best_macro_f1"])
    return {
        "config_path": "inline:auto_exp_11_module_ablation",
        "goal": "完成四模块全因子消融中缺失的10个组合实验。",
        "task_name": task_meta["task_name"],
        "task_dir_name": task_meta["task_dir_name"],
        "experiment_dir_name": "exp11_module_ablation",
        "output_dir_name": "",
        "selection_alias": selection_alias,
        "selection_metric_name": str(selection_meta["metric_name"]),
        "selection_mode": str(selection_meta["mode"]),
        "result_source": "test_results",
        "stability_filter": {
            "enabled": True,
            "min_epochs_trained": 10,
            "max_final_gap": 0.05,
            "max_val_loss_rebound_ratio": 0.25,
        },
        "remark": {
            "focus": "汇总 exp11_module_ablation 的10个缺失组合，比较 M1、M2、M3、M4 的主效应与交互作用。",
            "include_model_evaluations": True,
            "include_stability_filter": True,
        },
        "models": model_entries,
        "run_test": True,
    }


def resolve_run_cfg(train_cfg: dict[str, Any], model_name: str) -> dict[str, Any]:
    run_cfg = dict(train_cfg["default_run"])
    if model_name in {"full_feature_mil", "hier_full_mil", "hier_full_lg_mil", "mamba_mil"}:
        run_cfg.update(_build_exp4_sampled_long_bag_run_overrides())
    elif model_name == "long_mil":
        run_cfg.update(_build_exp4_long_run_overrides())
    elif model_name == "exp8_structured_late_gate_mil":
        run_cfg.update(
            _build_exp8_run_overrides(
                fields=["reportTitle", "age", "sex", "operationValue"],
                leakage_note=(
                    "exp_8 默认结构化 late-gate：复用 exp_8_mm_ablation 中非置乱正式候选最优的 "
                    "all_without_hp 字段组合；不使用 watchResult。"
                    "reportTitle/operationValue 属于流程相关字段，需标注检查路径代理风险。"
                ),
            )
        )
        run_cfg["seed"] = int(train_cfg.get("seed", 2026))
    elif model_name == "exp8_mm_struct_late_gate":
        run_cfg.update(
            _build_exp8_run_overrides(
                fields=["reportTitle", "age", "sex", "operationValue"],
                leakage_note=(
                    "exp_8 实验1：基于 exp_8_mm_ablation 非置乱最佳正式候选；"
                    "使用 reportTitle、age、sex、operationValue，不使用 watchResult。"
                    "reportTitle/operationValue 属于流程相关字段，需在论文中说明检查路径代理风险。"
                ),
            )
        )
    elif model_name == "exp8_mm_label_proto_graph":
        run_cfg.update(_build_exp6_long64_run_overrides())
        run_cfg.update(
            {
                "structured_fields": [],
                "modality_level": "fixed_proto",
                "modality_fields": "image+fixed_label_proto",
                "inference_inputs": "image+fixed_proto",
                "leakage_note": "exp_8 实验2：只使用固定标签文本原型，不输入个体报告文本或 watchResult。",
            }
        )
    elif model_name == "exp8_mm_text_contrast_distill":
        run_cfg.update(_build_exp6_long64_run_overrides())
        run_cfg.update(
            {
                "structured_fields": [],
                "modality_level": "train_time_distill",
                "modality_fields": "image+safe_text(train_only)",
                "inference_inputs": "image",
                "leakage_note": "exp_8 实验3：训练期使用安全报告字段做图文对齐，测试 logits 保持 image-only；不使用 watchResult。",
            }
        )
    elif model_name in {"exp8_mm_watch_cross_attn", "exp8_mm_watch_cross_attn_textcnn"}:
        run_cfg.update(_build_exp6_long64_run_overrides())
        run_cfg.update(
            {
                "structured_fields": [],
                "modality_level": "report_assist",
                "modality_fields": "image+watch",
                "inference_inputs": "image+watch",
                "leakage_note": (
                    "测试阶段输入类别名称掩码后的 watch，属于报告辅助任务；不输入 watchResult。"
                    if model_name == "exp8_mm_watch_cross_attn_textcnn"
                    else "exp_8 实验4：测试阶段输入 watch，属于报告辅助任务；不输入 watchResult。"
                ),
            }
        )
    elif model_name == "exp8_mm_text_guided_top64_align":
        run_cfg.update(_build_exp6_long64_run_overrides())
        run_cfg.update(
            {
                "structured_fields": [],
                "modality_level": "report_assist",
                "modality_fields": "image+text_guided_top64",
                "inference_inputs": "image+text_guided_fields",
                "leakage_note": "exp_8 实验5：使用低到中风险文本字段进行实例重加权与对齐；当前实现为训练内 64 图重加权，不使用 watchResult。",
            }
        )
    return run_cfg


def resolve_monitor_settings(
    run_cfg: dict[str, Any],
    *,
    default_metric: str,
    default_mode: str,
) -> tuple[str, str]:
    metric = str(run_cfg.get("monitor_metric", "")).strip() or default_metric
    mode = str(run_cfg.get("monitor_mode", "")).strip().lower() or default_mode
    if mode not in {"max", "min"}:
        raise ValueError("monitor_mode 仅支持 max 或 min")
    return metric, mode


def resolve_task_training_payload(
    training_context: dict[str, Any],
    task_name: str,
) -> tuple[dict[str, list[dict[str, Any]]], list[float]]:
    tasks_context = training_context.get("tasks", {})
    if task_name not in tasks_context:
        raise ValueError(f"训练上下文中不存在任务: {task_name}")
    payload = tasks_context[task_name]
    return payload["split"], payload["pos_weight"]


def build_test_result_metadata(run_cfg: dict[str, Any]) -> dict[str, Any] | None:
    metadata_keys = ("modality_level", "modality_fields", "inference_inputs", "leakage_note")
    metadata = {
        key: run_cfg.get(key, "")
        for key in metadata_keys
        if str(run_cfg.get(key, "")).strip()
    }
    return metadata or None


def _metadata_bool(value: Any, default: bool) -> bool:
    if isinstance(value, bool):
        return value
    if value is None:
        return default
    text = str(value).strip().lower()
    if text in {"1", "true", "yes", "y", "on"}:
        return True
    if text in {"0", "false", "no", "n", "off"}:
        return False
    return default


def infer_module_metadata(model_name: str, model_param_cfg: dict[str, Any]) -> dict[str, Any]:
    backbone = str(model_param_cfg.get("backbone_name", "convnext_tiny")).strip() or "convnext_tiny"

    if model_name == "gastro_label_graph_mil":
        pooling_type = str(model_param_cfg.get("pooling_type", "label_attention")).strip() or "label_attention"
        attention_type = str(model_param_cfg.get("attention_type", "label_specific")).strip() or "label_specific"
        use_label_graph = _metadata_bool(model_param_cfg.get("use_label_graph"), True)
        return {
            "backbone": backbone,
            "use_label_graph": use_label_graph,
            "label_graph_type": (
                str(model_param_cfg.get("label_graph_type", "learnable")).strip() or "learnable"
                if use_label_graph
                else "none"
            ),
            "use_label_wise_attention": _metadata_bool(
                model_param_cfg.get("use_label_wise_attention"),
                attention_type == "label_specific",
            ),
            "attention_type": attention_type,
            "pooling_type": pooling_type,
        }

    if model_name == "gastro_attention_mil_baseline":
        return {
            "backbone": backbone,
            "use_label_graph": False,
            "use_label_wise_attention": True,
            "attention_type": "label_specific",
            "pooling_type": "label_attention",
        }

    if model_name == "gastro_mean_pool_baseline":
        return {
            "backbone": backbone,
            "use_label_graph": False,
            "use_label_wise_attention": False,
            "attention_type": "none",
            "pooling_type": "mean",
        }

    return {
        "backbone": backbone,
        "use_label_graph": _metadata_bool(model_param_cfg.get("use_label_graph"), False),
        "label_graph_type": (
            str(model_param_cfg.get("label_graph_type", "")).strip()
            if _metadata_bool(model_param_cfg.get("use_label_graph"), False)
            else "none"
        ),
        "use_label_wise_attention": _metadata_bool(model_param_cfg.get("use_label_wise_attention"), False),
        "attention_type": str(model_param_cfg.get("attention_type", "")).strip(),
        "pooling_type": str(model_param_cfg.get("pooling_type", "")).strip(),
    }


def build_run_test_result_metadata(
    *,
    model_name: str,
    model_param_cfg: dict[str, Any],
    run_cfg: dict[str, Any],
    seed: int,
    entry_metadata: dict[str, Any] | None = None,
) -> dict[str, Any]:
    metadata = infer_module_metadata(model_name, model_param_cfg)
    metadata["seed"] = int(seed)
    metadata.update(build_test_result_metadata(run_cfg) or {})
    if isinstance(entry_metadata, dict):
        metadata.update(entry_metadata)
    return metadata


def attach_test_result_metadata(trainer_cfg: TrainerConfig, metadata: dict[str, Any]) -> None:
    existing = trainer_cfg.test_result_metadata if isinstance(trainer_cfg.test_result_metadata, dict) else {}
    merged = dict(metadata)
    merged.update(existing)
    trainer_cfg.test_result_metadata = merged or None


def _record_label_values(record: dict[str, Any], label_names: list[str]) -> list[int]:
    labels = record.get("labels")
    if isinstance(labels, dict):
        return [int(labels.get(label_name, 0)) for label_name in label_names]
    if isinstance(labels, (list, tuple)):
        return [int(labels[index]) if index < len(labels) else 0 for index, _ in enumerate(label_names)]
    return [int(record.get(label_name, 0)) for label_name in label_names]


def build_label_cooccurrence_prior(train_records: list[dict[str, Any]], label_names: list[str]) -> list[list[float]]:
    label_num = len(label_names)
    counts = np.zeros((label_num,), dtype=np.float64)
    cooccurrence = np.zeros((label_num, label_num), dtype=np.float64)
    for record in train_records:
        values = np.asarray(_record_label_values(record, label_names), dtype=np.float64)
        values = (values > 0).astype(np.float64)
        counts += values
        cooccurrence += np.outer(values, values)

    prior = np.eye(label_num, dtype=np.float64)
    for row_index in range(label_num):
        if counts[row_index] > 0:
            prior[row_index] = cooccurrence[row_index] / counts[row_index]
    return prior.tolist()


def build_model_bundle(
    model_name: str,
    task_name: str,
    run_cfg: dict[str, Any],
    model_param_cfg: dict[str, Any],
    pretrained: bool,
    max_epochs: int,
    patience: int,
    pos_weight: list[float],
    use_multi_gpu: bool,
    run_test: bool,
    resume_path: str | None = None,
) -> tuple[Any, TrainerConfig, str, list[str], list[str]]:
    task_meta = resolve_model_task_meta(model_name, task_name)
    resolved_task_name = str(task_meta["task_name"])
    task_type = str(task_meta["task_type"])
    label_names = list(task_meta["label_names"])
    class_names = list(task_meta["class_names"])
    num_labels = int(task_meta["num_labels"])

    if model_name in GASTRO_BASELINE_CLASS_REGISTRY:
        monitor_metric, monitor_mode = resolve_monitor_settings(
            run_cfg,
            default_metric="macro_auc",
            default_mode="max",
        )
        model = build_gastro_baseline(
            model_name=model_name,
            backbone_name=str(model_param_cfg.get("backbone_name", "resnet50")),
            pretrained=pretrained,
            freeze_stages=int(model_param_cfg.get("freeze_stages", 1)),
            feature_dim=int(model_param_cfg.get("feature_dim", 512)),
            attn_dim=int(model_param_cfg.get("attn_dim", 256)),
            hidden_dim=int(model_param_cfg.get("hidden_dim", 256)),
            num_labels=num_labels,
            dropout=float(model_param_cfg.get("dropout", 0.2)),
            topk=int(model_param_cfg.get("topk", 4)),
            num_heads=int(model_param_cfg.get("num_heads", 8)),
            num_layers=int(model_param_cfg.get("num_layers", 2)),
        )
        trainer_cfg = TrainerConfig(
            task_type=task_type,
            max_epochs=max_epochs,
            patience=patience,
            lr=float(run_cfg.get("lr", 2e-4)),
            optimizer_name=str(run_cfg.get("optimizer_name", "adamw")),
            weight_decay=float(run_cfg.get("weight_decay", 1e-4)),
            warmup_ratio=float(run_cfg.get("warmup_ratio", 0.1)),
            grad_accum_steps=int(run_cfg.get("grad_accum_steps", 2)),
            amp=bool(run_cfg.get("amp", True)),
            monitor_metric=monitor_metric,
            monitor_mode=monitor_mode,
            topk_evidence=int(run_cfg.get("topk_evidence", 5)),
            loss_name=str(run_cfg.get("loss_name", "asymmetric")),
            pos_weight=pos_weight,
            aux_loss_weights={},
            use_multi_gpu=use_multi_gpu,
            resume_path=resume_path,
            run_test=run_test,
        )
        return model, trainer_cfg, resolved_task_name, label_names, class_names

    if model_name in GASTRO_SOTA_CLASS_REGISTRY:
        monitor_metric, monitor_mode = resolve_monitor_settings(
            run_cfg,
            default_metric="macro_auc",
            default_mode="max",
        )
        model = build_gastro_sota(
            model_name=model_name,
            backbone_name=str(model_param_cfg.get("backbone_name", "convnext_tiny")),
            pretrained=pretrained,
            freeze_stages=int(model_param_cfg.get("freeze_stages", 1)),
            feature_dim=int(model_param_cfg.get("feature_dim", 512)),
            attn_dim=int(model_param_cfg.get("attn_dim", 256)),
            hidden_dim=int(model_param_cfg.get("hidden_dim", 256)),
            num_labels=num_labels,
            dropout=float(model_param_cfg.get("dropout", 0.2)),
            num_heads=int(model_param_cfg.get("num_heads", 8)),
            num_layers=int(model_param_cfg.get("num_layers", 2)),
            instance_topk=int(model_param_cfg.get("instance_topk", 4)),
            num_groups=int(model_param_cfg.get("num_groups", 4)),
        )
        aux_loss_weights = {
            "attention_entropy": float(model_param_cfg.get("attention_entropy_weight", 0.05)),
            "attention_diversity": float(model_param_cfg.get("attention_diversity_weight", 0.05)),
            "instance_clustering": float(model_param_cfg.get("instance_clustering_weight", 0.2)),
            "pseudo_bag": float(model_param_cfg.get("pseudo_bag_weight", 0.2)),
        }
        trainer_cfg = TrainerConfig(
            task_type=task_type,
            max_epochs=max_epochs,
            patience=patience,
            lr=float(run_cfg.get("lr", 2e-4)),
            optimizer_name=str(run_cfg.get("optimizer_name", "adamw")),
            weight_decay=float(run_cfg.get("weight_decay", 1e-4)),
            warmup_ratio=float(run_cfg.get("warmup_ratio", 0.1)),
            grad_accum_steps=int(run_cfg.get("grad_accum_steps", 2)),
            amp=bool(run_cfg.get("amp", True)),
            monitor_metric=monitor_metric,
            monitor_mode=monitor_mode,
            topk_evidence=int(run_cfg.get("topk_evidence", 5)),
            loss_name=str(run_cfg.get("loss_name", "asymmetric")),
            pos_weight=pos_weight,
            aux_loss_weights=aux_loss_weights,
            use_multi_gpu=use_multi_gpu,
            resume_path=resume_path,
            run_test=run_test,
        )
        return model, trainer_cfg, resolved_task_name, label_names, class_names

    if model_name in EXP1_CLASS_REGISTRY:
        monitor_metric, monitor_mode = resolve_monitor_settings(
            run_cfg,
            default_metric="macro_auc",
            default_mode="max",
        )
        model = build_exp1_model(
            model_name=model_name,
            backbone_name=str(model_param_cfg.get("backbone_name", "resnet50")),
            pretrained=pretrained,
            freeze_stages=int(model_param_cfg.get("freeze_stages", 1)),
            feature_dim=int(model_param_cfg.get("feature_dim", 512)),
            attn_dim=int(model_param_cfg.get("attn_dim", 256)),
            hidden_dim=int(model_param_cfg.get("hidden_dim", 256)),
            num_labels=num_labels,
            dropout=float(model_param_cfg.get("dropout", 0.2)),
            topk=int(model_param_cfg.get("topk", 4)),
            num_heads=int(model_param_cfg.get("num_heads", 8)),
            num_layers=int(model_param_cfg.get("num_layers", 2)),
            instance_topk=int(model_param_cfg.get("instance_topk", 4)),
            num_groups=int(model_param_cfg.get("num_groups", 4)),
            num_query_layers=int(model_param_cfg.get("num_query_layers", 2)),
            num_sinkhorn_iters=int(model_param_cfg.get("num_sinkhorn_iters", 10)),
            ot_epsilon=float(model_param_cfg.get("ot_epsilon", 0.1)),
            bottleneck_dim=int(model_param_cfg.get("bottleneck_dim", 128)),
            beta=float(model_param_cfg.get("beta", 0.001)),
            curvature=float(model_param_cfg.get("curvature", 1.0)),
            ortho_weight=float(model_param_cfg.get("ortho_weight", 0.01)),
            edl_kl_weight=float(model_param_cfg.get("edl_kl_weight", 0.1)),
            edl_annealing_steps=int(model_param_cfg.get("edl_annealing_steps", 500)),
            class_freq=model_param_cfg.get("class_freq"),
            class_prior=model_param_cfg.get("class_prior"),
            logit_adj_tau=float(model_param_cfg.get("logit_adj_tau", 1.0)),
            label_difficulty=model_param_cfg.get("label_difficulty"),
            total_epochs=int(model_param_cfg.get("total_epochs", max_epochs)),
        )
        trainer_cfg = TrainerConfig(
            task_type=task_type,
            max_epochs=max_epochs,
            patience=patience,
            lr=float(run_cfg.get("lr", 2e-4)),
            optimizer_name=str(run_cfg.get("optimizer_name", "adamw")),
            weight_decay=float(run_cfg.get("weight_decay", 1e-4)),
            warmup_ratio=float(run_cfg.get("warmup_ratio", 0.1)),
            grad_accum_steps=int(run_cfg.get("grad_accum_steps", 2)),
            amp=bool(run_cfg.get("amp", True)),
            monitor_metric=monitor_metric,
            monitor_mode=monitor_mode,
            topk_evidence=int(run_cfg.get("topk_evidence", 5)),
            loss_name=str(run_cfg.get("loss_name", "asymmetric")),
            pos_weight=pos_weight,
            aux_loss_weights={},
            use_multi_gpu=use_multi_gpu,
            resume_path=resume_path,
            run_test=run_test,
        )
        return model, trainer_cfg, resolved_task_name, label_names, class_names

    if model_name in EXP2_CLASS_REGISTRY:
        monitor_metric, monitor_mode = resolve_monitor_settings(
            run_cfg,
            default_metric="macro_auc",
            default_mode="max",
        )
        model = build_exp2_model(
            model_name=model_name,
            backbone_name=str(model_param_cfg.get("backbone_name", "resnet50")),
            pretrained=pretrained,
            freeze_stages=int(model_param_cfg.get("freeze_stages", 1)),
            feature_dim=int(model_param_cfg.get("feature_dim", 512)),
            attn_dim=int(model_param_cfg.get("attn_dim", 256)),
            hidden_dim=int(model_param_cfg.get("hidden_dim", 256)),
            num_labels=num_labels,
            dropout=float(model_param_cfg.get("dropout", 0.2)),
            topk=int(model_param_cfg.get("topk", 4)),
            num_heads=int(model_param_cfg.get("num_heads", 8)),
            num_layers=int(model_param_cfg.get("num_layers", 2)),
            instance_topk=int(model_param_cfg.get("instance_topk", 4)),
            num_groups=int(model_param_cfg.get("num_groups", 4)),
            num_query_layers=int(model_param_cfg.get("num_query_layers", 2)),
            class_freq=model_param_cfg.get("class_freq"),
            class_prior=model_param_cfg.get("class_prior"),
            logit_adj_tau=float(model_param_cfg.get("logit_adj_tau", 1.0)),
            consistency_weight=float(model_param_cfg.get("consistency_weight", 0.1)),
            eql_gamma=float(model_param_cfg.get("eql_gamma", 0.7)),
            eql_momentum=float(model_param_cfg.get("eql_momentum", 0.9)),
            max_margin=float(model_param_cfg.get("max_margin", 0.5)),
            drw_start_epoch=int(model_param_cfg.get("drw_start_epoch", 10)),
            norm_temperature=float(model_param_cfg.get("norm_temperature", 1.5)),
            loss_gamma_pos=float(model_param_cfg.get("loss_gamma_pos", 1.0)),
            loss_gamma_neg=float(model_param_cfg.get("loss_gamma_neg", 4.0)),
            loss_clip=float(model_param_cfg.get("loss_clip", 0.05)),
            poly_epsilon=float(model_param_cfg.get("poly_epsilon", 1.0)),
            hill_weight=float(model_param_cfg.get("hill_weight", 0.05)),
            hill_margin=float(model_param_cfg.get("hill_margin", 0.2)),
            tail_fraction=float(model_param_cfg.get("tail_fraction", 0.25)),
            rank_dim=int(model_param_cfg.get("rank_dim", 64)),
            sparsity_weight=float(model_param_cfg.get("sparsity_weight", 0.01)),
            low_rank_weight=float(model_param_cfg.get("low_rank_weight", 0.001)),
            fdr_target=float(model_param_cfg.get("fdr_target", 0.35)),
            fdr_weight=float(model_param_cfg.get("fdr_weight", 0.2)),
            tail_beta=float(model_param_cfg.get("tail_beta", 0.75)),
            head_label_indices=model_param_cfg.get("head_label_indices"),
            tail_topk=int(model_param_cfg.get("tail_topk", 4)),
            num_regions=int(model_param_cfg.get("num_regions", 6)),
            region_aux_weight=float(model_param_cfg.get("region_aux_weight", 0.2)),
            relevance_aux_weight=float(model_param_cfg.get("relevance_aux_weight", 0.1)),
            head_anchor_weight=float(model_param_cfg.get("head_anchor_weight", 0.25)),
            tail_branch_weight=float(model_param_cfg.get("tail_branch_weight", 0.15)),
        )
        trainer_cfg = TrainerConfig(
            task_type=task_type,
            max_epochs=max_epochs,
            patience=patience,
            lr=float(run_cfg.get("lr", 2e-4)),
            optimizer_name=str(run_cfg.get("optimizer_name", "adamw")),
            weight_decay=float(run_cfg.get("weight_decay", 1e-4)),
            warmup_ratio=float(run_cfg.get("warmup_ratio", 0.1)),
            grad_accum_steps=int(run_cfg.get("grad_accum_steps", 2)),
            amp=bool(run_cfg.get("amp", True)),
            monitor_metric=monitor_metric,
            monitor_mode=monitor_mode,
            topk_evidence=int(run_cfg.get("topk_evidence", 5)),
            loss_name=str(run_cfg.get("loss_name", "asymmetric")),
            pos_weight=pos_weight,
            aux_loss_weights={},
            use_multi_gpu=use_multi_gpu,
            resume_path=resume_path,
            run_test=run_test,
        )
        return model, trainer_cfg, resolved_task_name, label_names, class_names

    if model_name in EXP4_CLASS_REGISTRY:
        monitor_metric, monitor_mode = resolve_monitor_settings(
            run_cfg,
            default_metric="macro_auc",
            default_mode="max",
        )
        model = build_exp4_model(
            model_name=model_name,
            backbone_name=str(model_param_cfg.get("backbone_name", "convnext_tiny")),
            pretrained=pretrained,
            freeze_stages=int(model_param_cfg.get("freeze_stages", 1)),
            feature_dim=int(model_param_cfg.get("feature_dim", 512)),
            attn_dim=int(model_param_cfg.get("attn_dim", 256)),
            hidden_dim=int(model_param_cfg.get("hidden_dim", 1024)),
            num_labels=num_labels,
            dropout=float(model_param_cfg.get("dropout", 0.2)),
            num_heads=int(model_param_cfg.get("num_heads", 4)),
            num_layers=int(model_param_cfg.get("num_layers", 2)),
            subbag_size=int(model_param_cfg.get("subbag_size", 8)),
            encoder_chunk_size=int(model_param_cfg.get("encoder_chunk_size", 16)),
            use_label_graph=bool(model_param_cfg.get("use_label_graph", False)),
            use_quality_gate=bool(model_param_cfg.get("use_quality_gate", False)),
            num_views=int(model_param_cfg.get("num_views", 4)),
            view_keep_ratio=float(model_param_cfg.get("view_keep_ratio", 0.75)),
            mamba_expand=int(model_param_cfg.get("mamba_expand", 2)),
            mamba_kernel_size=int(model_param_cfg.get("mamba_kernel_size", 5)),
        )
        trainer_cfg = TrainerConfig(
            task_type=task_type,
            max_epochs=max_epochs,
            patience=patience,
            lr=float(run_cfg.get("lr", 2e-4)),
            optimizer_name=str(run_cfg.get("optimizer_name", "adamw")),
            weight_decay=float(run_cfg.get("weight_decay", 1e-4)),
            warmup_ratio=float(run_cfg.get("warmup_ratio", 0.1)),
            grad_accum_steps=int(run_cfg.get("grad_accum_steps", 2)),
            amp=bool(run_cfg.get("amp", True)),
            monitor_metric=monitor_metric,
            monitor_mode=monitor_mode,
            topk_evidence=int(run_cfg.get("topk_evidence", 5)),
            loss_name=str(run_cfg.get("loss_name", "asymmetric")),
            pos_weight=pos_weight,
            aux_loss_weights={},
            use_multi_gpu=use_multi_gpu,
            resume_path=resume_path,
            run_test=run_test,
        )
        return model, trainer_cfg, resolved_task_name, label_names, class_names

    if model_name in EXP6_CLASS_REGISTRY:
        monitor_metric, monitor_mode = resolve_monitor_settings(
            run_cfg,
            default_metric="macro_auc",
            default_mode="max",
        )
        model = build_exp6_model(
            model_name=model_name,
            backbone_name=str(model_param_cfg.get("backbone_name", "convnext_tiny")),
            pretrained=pretrained,
            freeze_stages=int(model_param_cfg.get("freeze_stages", 1)),
            feature_dim=int(model_param_cfg.get("feature_dim", 512)),
            attn_dim=int(model_param_cfg.get("attn_dim", 256)),
            hidden_dim=int(model_param_cfg.get("hidden_dim", 1024)),
            num_labels=num_labels,
            dropout=float(model_param_cfg.get("dropout", 0.2)),
            num_heads=int(model_param_cfg.get("num_heads", 4)),
            num_layers=int(model_param_cfg.get("num_layers", 2)),
            encoder_chunk_size=int(model_param_cfg.get("encoder_chunk_size", 16)),
            use_label_graph=bool(model_param_cfg.get("use_label_graph", True)),
            use_quality_gate=bool(model_param_cfg.get("use_quality_gate", False)),
            roi_gate_init=float(model_param_cfg.get("roi_gate_init", -1.0)),
            use_type_embedding=bool(model_param_cfg.get("use_type_embedding", True)),
        )
        aux_loss_weights = {
            "view_consistency": float(model_param_cfg.get("view_consistency_weight", 0.0)),
            "attention_entropy": float(model_param_cfg.get("attention_entropy_weight", 0.0)),
        }
        trainer_cfg = TrainerConfig(
            task_type=task_type,
            max_epochs=max_epochs,
            patience=patience,
            lr=float(run_cfg.get("lr", 2e-4)),
            optimizer_name=str(run_cfg.get("optimizer_name", "adamw")),
            weight_decay=float(run_cfg.get("weight_decay", 1e-4)),
            warmup_ratio=float(run_cfg.get("warmup_ratio", 0.1)),
            grad_accum_steps=int(run_cfg.get("grad_accum_steps", 2)),
            amp=bool(run_cfg.get("amp", True)),
            monitor_metric=monitor_metric,
            monitor_mode=monitor_mode,
            topk_evidence=int(run_cfg.get("topk_evidence", 5)),
            loss_name=str(run_cfg.get("loss_name", "asymmetric")),
            pos_weight=pos_weight,
            aux_loss_weights=aux_loss_weights,
            use_multi_gpu=use_multi_gpu,
            resume_path=resume_path,
            run_test=run_test,
        )
        return model, trainer_cfg, resolved_task_name, label_names, class_names

    if model_name in EXP8_CLASS_REGISTRY:
        monitor_metric, monitor_mode = resolve_monitor_settings(
            run_cfg,
            default_metric="macro_auc",
            default_mode="max",
        )
        model = build_exp8_model(
            model_name=model_name,
            backbone_name=str(model_param_cfg.get("backbone_name", "convnext_tiny")),
            pretrained=pretrained,
            freeze_stages=int(model_param_cfg.get("freeze_stages", 1)),
            feature_dim=int(model_param_cfg.get("feature_dim", 512)),
            attn_dim=int(model_param_cfg.get("attn_dim", 256)),
            hidden_dim=int(model_param_cfg.get("hidden_dim", 1024)),
            num_labels=num_labels,
            dropout=float(model_param_cfg.get("dropout", 0.2)),
            num_heads=int(model_param_cfg.get("num_heads", 4)),
            num_layers=int(model_param_cfg.get("num_layers", 2)),
            encoder_chunk_size=int(model_param_cfg.get("encoder_chunk_size", 16)),
            use_label_graph=bool(model_param_cfg.get("use_label_graph", True)),
            label_graph_type=str(model_param_cfg.get("label_graph_type", "label_hypergraph")),
            label_hypergraph_edges=int(model_param_cfg.get("label_hypergraph_edges", 2)),
            use_quality_gate=bool(model_param_cfg.get("use_quality_gate", False)),
            structured_fields=model_param_cfg.get("structured_fields", run_cfg.get("structured_fields", [])),
            structured_category_sizes=model_param_cfg.get("structured_category_sizes", {}),
            structured_field_embed_dim=int(model_param_cfg.get("structured_field_embed_dim", 64)),
            structured_dropout=float(model_param_cfg.get("structured_dropout", 0.2)),
            modality_dropout=float(model_param_cfg.get("modality_dropout", 0.15)),
            prototype_dropout=float(model_param_cfg.get("prototype_dropout", 0.1)),
            prototype_mix=float(model_param_cfg.get("prototype_mix", 0.35)),
            graph_prior_mix=float(model_param_cfg.get("graph_prior_mix", 0.3)),
            text_vocab_size=int(model_param_cfg.get("text_vocab_size", 8192)),
            text_embed_dim=int(model_param_cfg.get("text_embed_dim", 128)),
            contrast_temperature=float(model_param_cfg.get("contrast_temperature", 0.07)),
            use_context_encoder=bool(model_param_cfg.get("use_context_encoder", True)),
            watch_fusion_mode=str(model_param_cfg.get("watch_fusion_mode", "none")),
            use_text_gate=bool(model_param_cfg.get("use_text_gate", False)),
            textcnn_kernel_sizes=tuple(model_param_cfg.get("textcnn_kernel_sizes", (2, 3, 4))),
        )
        aux_loss_weights = {
            "structured_gate_l1": float(model_param_cfg.get("structured_gate_l1_weight", 0.0)),
            "proto_align": float(model_param_cfg.get("proto_align_weight", 0.0)),
            "graph_prior": float(model_param_cfg.get("graph_prior_weight", 0.0)),
            "text_align": float(model_param_cfg.get("text_align_weight", 0.0)),
            "text_itc": float(model_param_cfg.get("text_itc_weight", 0.0)),
            "image_aux": float(model_param_cfg.get("image_aux_weight", 0.0)),
            "attention_sparse": float(model_param_cfg.get("attention_sparse_weight", 0.0)),
            "consistency": float(model_param_cfg.get("consistency_weight", 0.0)),
        }
        trainer_cfg = TrainerConfig(
            task_type=task_type,
            max_epochs=max_epochs,
            patience=patience,
            lr=float(run_cfg.get("lr", 2e-4)),
            optimizer_name=str(run_cfg.get("optimizer_name", "adamw")),
            weight_decay=float(run_cfg.get("weight_decay", 1e-4)),
            warmup_ratio=float(run_cfg.get("warmup_ratio", 0.1)),
            grad_accum_steps=int(run_cfg.get("grad_accum_steps", 2)),
            amp=bool(run_cfg.get("amp", True)),
            monitor_metric=monitor_metric,
            monitor_mode=monitor_mode,
            topk_evidence=int(run_cfg.get("topk_evidence", 5)),
            loss_name=str(run_cfg.get("loss_name", "asymmetric")),
            pos_weight=pos_weight,
            aux_loss_weights=aux_loss_weights,
            use_multi_gpu=use_multi_gpu,
            resume_path=resume_path,
            run_test=run_test,
            test_result_metadata=build_test_result_metadata(run_cfg),
            data_parallel_device_ids=(
                [int(device_id) for device_id in run_cfg.get("data_parallel_device_ids", [])]
                or None
            ),
        )
        return model, trainer_cfg, resolved_task_name, label_names, class_names

    if model_name == "gastro_label_graph_mil":
        monitor_metric, monitor_mode = resolve_monitor_settings(
            run_cfg,
            default_metric="macro_auc",
            default_mode="max",
        )
        model = GastroLabelGraphMIL(
            backbone_name=str(model_param_cfg.get("backbone_name", "convnext_tiny")),
            pretrained=pretrained,
            freeze_stages=int(model_param_cfg.get("freeze_stages", 1)),
            feature_dim=int(model_param_cfg.get("feature_dim", 512)),
            attn_dim=int(model_param_cfg.get("attn_dim", 256)),
            num_labels=num_labels,
            dropout=float(model_param_cfg.get("dropout", 0.2)),
            use_label_graph=_metadata_bool(model_param_cfg.get("use_label_graph"), True),
            label_graph_type=str(model_param_cfg.get("label_graph_type", "learnable")),
            label_graph_prior=model_param_cfg.get("label_graph_prior"),
            label_graph_rank=int(model_param_cfg.get("label_graph_rank", 2)),
            label_graph_heads=int(model_param_cfg.get("label_graph_heads", 4)),
            label_hypergraph_edges=int(model_param_cfg.get("label_hypergraph_edges", 2)),
            use_label_wise_attention=_metadata_bool(model_param_cfg.get("use_label_wise_attention"), True),
            attention_type=str(model_param_cfg.get("attention_type", "label_specific")),
            pooling_type=str(model_param_cfg.get("pooling_type", "label_attention")),
        )
        trainer_cfg = TrainerConfig(
            task_type=task_type,
            max_epochs=max_epochs,
            patience=patience,
            lr=float(run_cfg.get("lr", 2e-4)),
            optimizer_name=str(run_cfg.get("optimizer_name", "adamw")),
            weight_decay=float(run_cfg.get("weight_decay", 1e-4)),
            warmup_ratio=float(run_cfg.get("warmup_ratio", 0.1)),
            grad_accum_steps=int(run_cfg.get("grad_accum_steps", 2)),
            amp=bool(run_cfg.get("amp", True)),
            monitor_metric=monitor_metric,
            monitor_mode=monitor_mode,
            topk_evidence=int(run_cfg.get("topk_evidence", 5)),
            loss_name=str(run_cfg.get("loss_name", "asymmetric")),
            pos_weight=pos_weight,
            aux_loss_weights={},
            use_multi_gpu=use_multi_gpu,
            resume_path=resume_path,
            run_test=run_test,
        )
        return model, trainer_cfg, resolved_task_name, label_names, class_names

    if model_name in {"rg_hmil", "gastro_rg_hmil"}:
        monitor_metric, monitor_mode = resolve_monitor_settings(
            run_cfg,
            default_metric="macro_auc",
            default_mode="max",
        )
        aux_loss_weights = {
            "region_cls": float(model_param_cfg.get("region_cls_weight", 0.5)),
            "relevance": float(model_param_cfg.get("relevance_weight", 0.5)),
        }
        model = RGHMIL(
            backbone_name=str(model_param_cfg.get("backbone_name", "convnext_tiny")),
            pretrained=pretrained,
            freeze_stages=int(model_param_cfg.get("freeze_stages", 1)),
            feature_dim=int(model_param_cfg.get("feature_dim", 512)),
            attn_dim=int(model_param_cfg.get("attn_dim", 256)),
            num_labels=num_labels,
            num_regions=int(model_param_cfg.get("num_regions", 6)),
            condition_dim=int(model_param_cfg.get("condition_dim", 128)),
            dropout=float(model_param_cfg.get("dropout", 0.2)),
            use_text_guidance=bool(model_param_cfg.get("use_text_guidance", True)),
            use_region_grouping=bool(model_param_cfg.get("use_region_grouping", True)),
            use_conditional_graph=bool(model_param_cfg.get("use_conditional_graph", True)),
            use_hierarchy=bool(model_param_cfg.get("use_hierarchy", True)),
        )
        trainer_cfg = TrainerConfig(
            task_type=task_type,
            max_epochs=max_epochs,
            patience=patience,
            lr=float(run_cfg.get("lr", 2e-4)),
            optimizer_name=str(run_cfg.get("optimizer_name", "adamw")),
            weight_decay=float(run_cfg.get("weight_decay", 1e-4)),
            warmup_ratio=float(run_cfg.get("warmup_ratio", 0.1)),
            grad_accum_steps=int(run_cfg.get("grad_accum_steps", 2)),
            amp=bool(run_cfg.get("amp", True)),
            monitor_metric=monitor_metric,
            monitor_mode=monitor_mode,
            topk_evidence=int(run_cfg.get("topk_evidence", 5)),
            loss_name=str(run_cfg.get("loss_name", "asymmetric")),
            pos_weight=pos_weight,
            aux_loss_weights=aux_loss_weights,
            use_multi_gpu=use_multi_gpu,
            resume_path=resume_path,
            run_test=run_test,
        )
        return model, trainer_cfg, resolved_task_name, label_names, class_names

    raise ValueError(f"未知模型名: {model_name}")


def run_single_model(
    *,
    model_name: str,
    model,
    trainer_cfg: TrainerConfig,
    split_data: dict[str, list[dict[str, Any]]],
    task_name: str,
    image_size: int,
    num_workers: int,
    run_dir: Path,
    seed: int,
    run_cfg: dict[str, Any],
    model_param_cfg: dict[str, Any],
    min_instances: int,
    train_sampling: str,
    eval_sampling: str,
    active_gpu_count: int,
    label_names: list[str],
    class_names: list[str],
    cache_root_dir: Path,
    on_validation_epoch_end: Callable[[int, float, dict[str, Any]], None] | None = None,
    structured_metadata: dict[str, Any] | None = None,
) -> dict[str, Any]:
    train_batch_size = normalize_batch_size(int(run_cfg.get("batch_size", 3)), active_gpu_count)
    eval_batch_size = normalize_batch_size(int(run_cfg.get("eval_batch_size", 3)), active_gpu_count)
    resolved_cache_root_dir, resolved_cache_dir, legacy_cache_dirs = resolve_image_cache_directories(
        task_name=task_name,
        cache_root_dir=cache_root_dir,
        run_cfg=run_cfg,
    )

    effective_run_cfg = dict(run_cfg)
    if resolved_cache_root_dir is not None:
        effective_run_cfg["resolved_image_cache_root_dir"] = str(resolved_cache_root_dir)
    if resolved_cache_dir is not None:
        effective_run_cfg["resolved_image_cache_dir"] = str(resolved_cache_dir)
        effective_run_cfg["image_cache_scope"] = str(run_cfg.get("image_cache_scope", "task")).strip().lower() or "task"
    if legacy_cache_dirs:
        effective_run_cfg["resolved_legacy_image_cache_dirs"] = [str(path) for path in legacy_cache_dirs]

    train_loader, val_loader, test_loader = build_loaders(
        split_data=split_data,
        task_name=task_name,
        image_size=image_size,
        num_workers=num_workers,
        train_batch_size=train_batch_size,
        eval_batch_size=eval_batch_size,
        train_max_instances=int(run_cfg.get("train_max_instances", 32)),
        eval_max_instances=int(run_cfg.get("eval_max_instances", 32)),
        min_instances=min_instances,
        train_sampling=train_sampling,
        eval_sampling=eval_sampling,
        random_instance_dropout=float(run_cfg.get("random_instance_dropout", 0.0)),
        train_max_batch_instances=int(run_cfg.get("train_max_batch_instances", 96)),
        eval_max_batch_instances=int(run_cfg.get("eval_max_batch_instances", 96)),
        seed=seed,
        pin_memory=bool(run_cfg.get("pin_memory", True)),
        persistent_workers=bool(run_cfg.get("persistent_workers", True)),
        loader_prefetch_factor=int(run_cfg.get("loader_prefetch_factor", 2)),
        image_cache_mode=str(run_cfg.get("image_cache_mode", "none")),
        image_cache_dir=resolved_cache_dir,
        legacy_image_cache_dirs=legacy_cache_dirs,
        image_cache_warmup=bool(run_cfg.get("image_cache_warmup", False)),
        memory_cache_size=int(run_cfg.get("memory_cache_size", 0)),
        roi_enabled=bool(run_cfg.get("roi_enabled", False)),
        roi_index_path=str(run_cfg.get("roi_index_path", "")).strip() or None,
        roi_max_crops_per_bag=int(run_cfg.get("roi_max_crops_per_bag", 0)),
        roi_max_crops_per_source=int(run_cfg.get("roi_max_crops_per_source", 1)),
        roi_min_score=float(run_cfg.get("roi_min_score", 0.0)),
        structured_shuffle_fields=run_cfg.get("structured_shuffle_fields", []),
        structured_shuffle_apply_to=str(run_cfg.get("structured_shuffle_apply_to", "none")),
        structured_shuffle_seed=int(run_cfg.get("structured_shuffle_seed", 0)),
    )

    run_dir.mkdir(parents=True, exist_ok=True)
    write_training_config(
        run_dir=run_dir,
        model_name=model_name,
        task_name=task_name,
        trainer_cfg=trainer_cfg,
        run_cfg=effective_run_cfg,
        model_param_cfg=model_param_cfg,
        split_data=split_data,
        image_size=image_size,
        num_workers=num_workers,
        seed=seed,
    )
    write_structured_audit_files(run_dir, structured_metadata)

    trainer = Trainer(
        model=model,
        cfg=trainer_cfg,
        run_dir=run_dir,
        train_loader=train_loader,
        val_loader=val_loader,
        test_loader=test_loader,
        label_names=label_names,
        class_names=class_names,
        seed=seed,
        on_validation_epoch_end=on_validation_epoch_end,
    )
    result = trainer.fit()
    result["model_name"] = model_name
    result["train_dir"] = str(run_dir)
    result["train_dir_name"] = run_dir.name
    return result


def run_model_job(
    *,
    model_name: str,
    run_dir: Path,
    train_cfg: dict[str, Any],
    model_cfg: dict[str, dict[str, Any]],
    training_context: dict[str, Any],
    seed: int,
    max_epochs: int,
    patience: int,
    image_size: int,
    num_workers: int,
    pretrained: bool,
    use_multi_gpu: bool,
    active_gpu_count: int,
    run_test: bool,
    run_overrides: dict[str, Any] | None = None,
    model_param_override: dict[str, Any] | None = None,
    entry_metadata: dict[str, Any] | None = None,
    resume_path: str | None = None,
    on_validation_epoch_end: Callable[[int, float, dict[str, Any]], None] | None = None,
) -> dict[str, Any]:
    run_cfg = resolve_run_cfg(train_cfg, model_name)
    if run_overrides:
        run_cfg.update(run_overrides)
    effective_seed = int(run_cfg.get("seed", seed))
    run_cfg["seed"] = effective_seed
    model_param_cfg = dict(model_cfg["models"].get(model_name, {}))
    if model_param_override:
        model_param_cfg.update(model_param_override)

    selected_task_name = resolve_train_task_name(None, train_cfg)
    task_meta = resolve_model_task_meta(model_name, selected_task_name)
    split_data, pos_weight = resolve_task_training_payload(training_context, task_meta["task_name"])
    if (
        model_name == "gastro_label_graph_mil"
        and str(model_param_cfg.get("label_graph_type", "")).strip().lower() == "static_gcn"
        and "label_graph_prior" not in model_param_cfg
    ):
        model_param_cfg["label_graph_prior"] = build_label_cooccurrence_prior(
            split_data.get("train", []),
            list(task_meta["label_names"]),
        )
    task_payload = training_context.get("tasks", {}).get(task_meta["task_name"], {})
    if task_payload.get("balance_report"):
        run_cfg["class_balance_report"] = task_payload["balance_report"]
    structured_metadata = task_payload.get("structured_metadata")
    if model_name in EXP8_CLASS_REGISTRY and isinstance(structured_metadata, dict):
        model_param_cfg["structured_category_sizes"] = dict(structured_metadata.get("category_sizes", {}))

    seed_everything(effective_seed)
    model, trainer_cfg, task_name, label_names, class_names = build_model_bundle(
        model_name=model_name,
        task_name=task_meta["task_name"],
        run_cfg=run_cfg,
        model_param_cfg=model_param_cfg,
        pretrained=pretrained,
        max_epochs=max_epochs,
        patience=patience,
        pos_weight=pos_weight,
        use_multi_gpu=use_multi_gpu,
        run_test=run_test,
        resume_path=resume_path,
    )
    attach_test_result_metadata(
        trainer_cfg,
        build_run_test_result_metadata(
            model_name=model_name,
            model_param_cfg=model_param_cfg,
            run_cfg=run_cfg,
            seed=effective_seed,
            entry_metadata=entry_metadata,
        ),
    )

    return run_single_model(
        model_name=model_name,
        model=model,
        trainer_cfg=trainer_cfg,
        split_data=split_data,
        task_name=task_name,
        image_size=image_size,
        num_workers=num_workers,
        run_dir=run_dir,
        seed=effective_seed,
        run_cfg=run_cfg,
        model_param_cfg=model_param_cfg,
        min_instances=train_cfg["min_instances"],
        train_sampling=str(run_cfg.get("train_sampling_strategy", train_cfg["train_sampling_strategy"])),
        eval_sampling=str(run_cfg.get("eval_sampling_strategy", train_cfg["eval_sampling_strategy"])),
        active_gpu_count=active_gpu_count,
        label_names=label_names,
        class_names=class_names,
        cache_root_dir=Path(training_context["task_selection_dir"]).resolve(),
        on_validation_epoch_end=on_validation_epoch_end,
        structured_metadata=structured_metadata if isinstance(structured_metadata, dict) else None,
    )


def prepare_training_context(
    *,
    path_cfg: dict[str, str],
    train_cfg: dict[str, Any],
    seed: int,
    max_exams_per_task: int,
    required_tasks: set[str],
) -> dict[str, Any]:
    output_root = Path(path_cfg["output_dir"]).resolve()
    task_data_root = Path(path_cfg.get("dataset_base_root", output_root)).resolve()
    task_selection_dir = task_data_root / train_cfg["task_selection_dir_name"]

    dataset_root = path_cfg.get("dataset_root")
    context = {
        "output_root": output_root,
        "task_selection_dir": str(task_selection_dir),
        "tasks": {},
        "task_stats": {},
    }

    legacy_candidates = {
        "task1": [
            task_selection_dir / "task1" / "datalist.csv",
            task_selection_dir / "task1_gastro3" / "datalist.csv",
        ],
        "task2": [
            task_selection_dir / "task2" / "datalist.csv",
            task_selection_dir / "task2_gastro3" / "datalist.csv",
        ],
    }

    for index, task_name in enumerate(sorted(required_tasks)):
        task_spec = get_task_spec(task_name)
        task_csv = task_selection_dir / task_spec.data_subdir / task_spec.datalist_filename
        if not task_csv.is_file():
            for legacy_csv in legacy_candidates.get(task_name, []):
                if legacy_csv.is_file():
                    task_csv = legacy_csv
                    break
            if not task_csv.is_file():
                build_script = f"python scripts/{task_name}_build_datalist.py"
                raise FileNotFoundError(
                    f"未找到任务筛选 CSV，请先运行 `{build_script}` 生成："
                    f"\n- {task_csv}"
                )

        records = build_task_records(
            task_csv_path=task_csv,
            task_name=task_name,
            min_instances=train_cfg["min_instances"],
            dataset_root=dataset_root,
        )
        report_enrich_report = None
        if task_name == "task2":
            report_enrich_report = enrich_records_with_report_fields(
                records,
                resolve_structured_report_csv_path(path_cfg, task_name),
            )
        records = maybe_limit_records(records, max_num=max_exams_per_task, seed=seed + index)
        split_data, _ = build_compatible_split(
            records,
            seed=seed + index,
            ratios=train_cfg["split_ratio"],
            group_by_patient=bool(train_cfg.get("group_by_patient", False)) and task_name == "task2",
        )
        structured_fit_records = list(split_data["train"])
        balance_report = None
        class_balance_cfg = dict(train_cfg.get("class_balance", {"enabled": False}))
        if task_spec.is_multilabel and bool(class_balance_cfg.get("enabled", False)):
            balanced_train_records, balance_report = build_multilabel_minority_balance(
                train_records=split_data["train"],
                label_names=task_spec.label_names,
                cfg=class_balance_cfg,
                seed=seed + index,
            )
            split_data = {
                "train": balanced_train_records,
                "val": split_data["val"],
                "test": split_data["test"],
            }
            report_filename = str(class_balance_cfg.get("report_filename", "class_balance_report.json")).strip()
            if report_filename:
                report_path = task_selection_dir / task_spec.data_subdir / report_filename
                balance_report["report_path"] = str(report_path)
                if os.environ.get("PROJECT4_SUPPRESS_CLASS_BALANCE_REPORT") != "1":
                    report_path.parent.mkdir(parents=True, exist_ok=True)
                    report_path.write_text(
                        json.dumps(to_builtin_type(balance_report), ensure_ascii=False, indent=2) + "\n",
                        encoding="utf-8",
                    )
        structured_metadata = None
        if task_name == "task2":
            structured_metadata = prepare_structured_features(
                split_data,
                fit_records=structured_fit_records,
                min_category_count=int(train_cfg.get("structured_min_category_count", 20)),
            )
        if task_spec.is_multilabel:
            pos_weight = compute_multilabel_pos_weight(split_data["train"]) if split_data["train"] else [1.0 for _ in task_spec.label_names]
        else:
            pos_weight = compute_binary_pos_weight(split_data["train"]) if split_data["train"] else [1.0]

        context["tasks"][task_name] = {
            "csv_path": str(task_csv),
            "split": split_data,
            "pos_weight": pos_weight,
            "balance_report": balance_report,
            "structured_metadata": structured_metadata,
            "structured_report_enrich": report_enrich_report,
        }
        context["task_stats"][task_name] = {
            "total_records": len(records),
            "train_original_size": int(balance_report.get("original_train_size", len(split_data["train"]))) if balance_report else len(split_data["train"]),
            "train_size": len(split_data["train"]),
            "val_size": len(split_data["val"]),
            "test_size": len(split_data["test"]),
            "class_balance_added_records": int(balance_report.get("added_records", 0)) if balance_report else 0,
        }

    return context


def run_training_session(
    *,
    session_dir: Path,
    names_to_run: list[str],
    train_cfg: dict[str, Any],
    model_cfg: dict[str, dict[str, Any]],
    training_context: dict[str, Any],
    seed: int,
    max_epochs: int,
    patience: int,
    image_size: int,
    num_workers: int,
    pretrained: bool,
    use_multi_gpu: bool,
    active_gpu_count: int,
    run_test: bool,
    run_overrides: dict[str, Any] | None = None,
    remark_metric_alias: str | None = None,
    remark_metric_name: str | None = None,
    remark_result_source: str = "test_results",
    remark_stability_filter: dict[str, Any] | None = None,
    remark_context: dict[str, Any] | None = None,
    session_extra_settings: dict[str, Any] | None = None,
) -> dict[str, Any]:
    all_results: dict[str, Any] = {
        "session_dir": str(session_dir),
        "models": {},
        "settings": {
            "seed": seed,
            "image_size": image_size,
            "max_epochs": max_epochs,
            "patience": patience,
            "num_workers": num_workers,
            "use_multi_gpu": use_multi_gpu,
            "pretrained": pretrained,
            "run_test": run_test,
            "run_overrides": run_overrides or {},
        },
    }
    if session_extra_settings:
        all_results["settings"].update(session_extra_settings)

    if remark_metric_alias and remark_metric_name:
        write_run_remark(
            session_dir,
            all_results,
            remark_metric_alias=remark_metric_alias,
            remark_metric_name=remark_metric_name,
            result_source=remark_result_source,
            stability_filter=remark_stability_filter,
            remark_context=remark_context,
        )

    selected_task_name = resolve_train_task_name(None, train_cfg)
    for model_index, model_name in enumerate(names_to_run, start=1):
        print(f"\n[{model_index}/{len(names_to_run)}] 开始训练 {model_name}")
        run_cfg = resolve_run_cfg(train_cfg, model_name)
        if run_overrides:
            run_cfg.update(run_overrides)
        model_param_cfg = dict(model_cfg["models"].get(model_name, {}))

        task_meta = resolve_model_task_meta(model_name, selected_task_name)
        split_data, pos_weight = resolve_task_training_payload(training_context, task_meta["task_name"])
        if (
            model_name == "gastro_label_graph_mil"
            and str(model_param_cfg.get("label_graph_type", "")).strip().lower() == "static_gcn"
            and "label_graph_prior" not in model_param_cfg
        ):
            model_param_cfg["label_graph_prior"] = build_label_cooccurrence_prior(
                split_data.get("train", []),
                list(task_meta["label_names"]),
            )
        task_payload = training_context.get("tasks", {}).get(task_meta["task_name"], {})
        if task_payload.get("balance_report"):
            run_cfg["class_balance_report"] = task_payload["balance_report"]
        structured_metadata = task_payload.get("structured_metadata")
        if model_name in EXP8_CLASS_REGISTRY and isinstance(structured_metadata, dict):
            model_param_cfg["structured_category_sizes"] = dict(structured_metadata.get("category_sizes", {}))

        model_seed = seed + model_index
        run_cfg["seed"] = int(model_seed)
        seed_everything(model_seed)
        model, trainer_cfg, task_name, label_names, class_names = build_model_bundle(
            model_name=model_name,
            task_name=task_meta["task_name"],
            run_cfg=run_cfg,
            model_param_cfg=model_param_cfg,
            pretrained=pretrained,
            max_epochs=max_epochs,
            patience=patience,
            pos_weight=pos_weight,
            use_multi_gpu=use_multi_gpu,
            run_test=run_test,
        )
        attach_test_result_metadata(
            trainer_cfg,
            build_run_test_result_metadata(
                model_name=model_name,
                model_param_cfg=model_param_cfg,
                run_cfg=run_cfg,
                seed=model_seed,
            ),
        )

        run_dir = model_run_dir(session_dir, model_index, model_name)
        result = run_single_model(
            model_name=model_name,
            model=model,
            trainer_cfg=trainer_cfg,
            split_data=split_data,
            task_name=task_name,
            image_size=image_size,
            num_workers=num_workers,
            run_dir=run_dir,
            seed=model_seed,
            run_cfg=run_cfg,
            model_param_cfg=model_param_cfg,
            min_instances=train_cfg["min_instances"],
            train_sampling=str(run_cfg.get("train_sampling_strategy", train_cfg["train_sampling_strategy"])),
            eval_sampling=str(run_cfg.get("eval_sampling_strategy", train_cfg["eval_sampling_strategy"])),
            active_gpu_count=active_gpu_count,
            label_names=label_names,
            class_names=class_names,
            cache_root_dir=Path(training_context["task_selection_dir"]).resolve(),
            structured_metadata=structured_metadata if isinstance(structured_metadata, dict) else None,
        )

        all_results["models"][model_name] = result
        if remark_metric_alias and remark_metric_name:
            write_run_remark(
                session_dir,
                all_results,
                remark_metric_alias=remark_metric_alias,
                remark_metric_name=remark_metric_name,
                result_source=remark_result_source,
                stability_filter=remark_stability_filter,
                remark_context=remark_context,
            )

    return all_results


def allocate_auto_series_dir(
    output_root: Path,
    train_cfg: dict[str, Any],
    auto_series_cfg: dict[str, Any],
) -> Path:
    task_dir = output_root / train_cfg["train_run_dir_name"] / auto_series_cfg["task_dir_name"]
    if "experiment_dir_name" in auto_series_cfg:
        experiment_dir_name = str(auto_series_cfg.get("experiment_dir_name", "")).strip()
    else:
        experiment_dir_name = str(train_cfg.get("experiment_dir_name", "")).strip()
    if experiment_dir_name:
        task_dir = task_dir / experiment_dir_name
    task_dir.mkdir(parents=True, exist_ok=True)
    output_dir_name = str(auto_series_cfg.get("output_dir_name", "")).strip()
    session_dir = task_dir / output_dir_name if output_dir_name else task_dir
    session_dir.mkdir(parents=True, exist_ok=True)
    return session_dir


def auto_series_result_dir_name(alias: str) -> str:
    return alias.replace("best_", "test_", 1)


def build_auto_series_record(
    entry: dict[str, Any],
    run_dir: Path,
) -> dict[str, Any]:
    return {
        "model_name": entry["name"],
        "base_model_name": resolve_series_entry_model_name(entry),
        "display_name": entry["display_name"],
        "status": "success",
        "train_dir": run_dir.name,
        "train_dir_path": str(run_dir),
        "selection_alias": "",
        "selection_metric": "",
        "selection_mode": "",
        "score": float("nan"),
        "best_epoch": -1,
        "checkpoint_path": "",
        "test_results": {},
        "seed": entry.get("seed", ""),
        "metadata": dict(entry.get("metadata", {})),
        "model_params": dict(entry.get("model_params", {})),
        "run_overrides": dict(entry.get("run_overrides", {})),
        "run_action": "",
        "error_message": "",
    }


def enrich_auto_series_record(
    record: dict[str, Any],
    *,
    result: dict[str, Any],
    auto_series_cfg: dict[str, Any],
) -> None:
    all_results = {"models": {record["model_name"]: result}}
    candidates = resolve_session_candidate(
        all_results,
        remark_metric_alias=auto_series_cfg["selection_alias"],
        result_source=auto_series_cfg["result_source"],
        fallback_metric_name=auto_series_cfg["selection_metric_name"],
    )
    candidate = select_best_candidate(candidates, mode=auto_series_cfg["selection_mode"])
    if candidate is None:
        raise RuntimeError(f"{record['model_name']} 未产出可比较结果")

    log_analysis = analyze_training_log(
        Path(record["train_dir_path"]) / "log.csv",
        auto_series_cfg["stability_filter"],
    )
    record.update(log_analysis)
    record["selection_alias"] = auto_series_cfg["selection_alias"]
    record["selection_metric"] = auto_series_cfg["selection_metric_name"]
    record["selection_mode"] = auto_series_cfg["selection_mode"]
    record["score"] = candidate["score"]
    record["best_epoch"] = candidate["best_epoch"]
    record["checkpoint_path"] = candidate["checkpoint_path"]
    record["test_results"] = result.get("test_results", {})
    record["evaluation"] = summarize_model_evaluation(
        log_analysis,
        auto_series_cfg["stability_filter"],
    )


def write_auto_series_notes(
    session_dir: Path,
    model_records: list[dict[str, Any]],
    all_results: dict[str, Any],
    auto_series_cfg: dict[str, Any],
) -> None:
    evaluations = build_model_evaluations(
        all_results,
        remark_metric_alias=auto_series_cfg["selection_alias"],
        remark_metric_name=auto_series_cfg["selection_metric_name"],
        result_source=auto_series_cfg["result_source"],
        stability_filter=auto_series_cfg["stability_filter"],
    )
    evaluation_map = {item["model_name"]: item for item in evaluations}
    best_candidate = select_best_candidate(
        evaluations,
        mode=auto_series_cfg["selection_mode"],
    ) if evaluations else None

    notes_payload = {
        "run_dir": str(session_dir),
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "config_path": auto_series_cfg["config_path"],
        "goal": auto_series_cfg.get("goal", ""),
        "task_name": auto_series_cfg["task_name"],
        "task_dir_name": auto_series_cfg["task_dir_name"],
        "output_dir_name": auto_series_cfg["output_dir_name"],
        "selection": {
            "checkpoint_alias": auto_series_cfg["selection_alias"],
            "metric_name": auto_series_cfg["selection_metric_name"],
            "mode": auto_series_cfg["selection_mode"],
            "result_source": auto_series_cfg["result_source"],
        },
        "counts": {
            "configured_models": len(auto_series_cfg["models"]),
            "enabled_models": len([item for item in auto_series_cfg["models"] if item["enabled"]]),
            "completed_models": len(model_records),
            "successful_models": len([item for item in model_records if item.get("status") == "success"]),
            "failed_models": len([item for item in model_records if item.get("status") == "failed"]),
            "interrupted_models": len([item for item in model_records if item.get("status") == "interrupted"]),
        },
        "best_model": best_candidate or {},
        "model_records": [
            {
                **item,
                "evaluation_detail": evaluation_map.get(item["model_name"], {}),
            }
            for item in model_records
        ],
    }
    if "roi_summary" in auto_series_cfg:
        notes_payload["roi_summary"] = auto_series_cfg["roi_summary"]
    (session_dir / "notes.json").write_text(
        json.dumps(to_builtin_type(notes_payload), ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def normalize_auto_series_run_dirs(
    session_dir: Path,
    model_entries: list[dict[str, Any]],
) -> None:
    for run_index, entry in enumerate(model_entries, start=1):
        expected_dir = session_dir / auto_baseline_run_dir_name(run_index, entry["name"])
        if expected_dir.exists():
            continue

        candidates = sorted(
            [
                child
                for child in session_dir.iterdir()
                if child.is_dir() and child.name.endswith(f"_{entry['name']}")
            ],
            key=lambda item: item.name,
        )
        if len(candidates) == 1:
            candidates[0].rename(expected_dir)


def is_auto_series_run_complete(run_dir: Path) -> bool:
    if not (run_dir / "test_result.csv").is_file():
        return False
    return all(
        (run_dir / auto_series_result_dir_name(alias) / "metrics.json").is_file()
        for alias in SERIES_TRACKER_ALIASES
    )


def auto_series_resume_checkpoint(run_dir: Path) -> str | None:
    last_path = run_dir / "checkpoints" / "last.ckpt"
    if last_path.is_file():
        return str(last_path)
    return None


def reset_auto_series_run_dir(run_dir: Path) -> None:
    if not run_dir.exists():
        return
    for child in run_dir.iterdir():
        if child.is_dir():
            shutil.rmtree(child)
        else:
            child.unlink()


def load_existing_auto_series_result(run_dir: Path, model_name: str) -> dict[str, Any]:
    config_payload: dict[str, Any] = {}
    config_path = run_dir / "config.yaml"
    if config_path.is_file():
        loaded = yaml.safe_load(config_path.read_text(encoding="utf-8"))
        if isinstance(loaded, dict):
            config_payload = loaded

    test_results: dict[str, Any] = {}
    best_checkpoints: dict[str, Any] = {}
    for alias in SERIES_TRACKER_ALIASES:
        metrics_path = run_dir / auto_series_result_dir_name(alias) / "metrics.json"
        if not metrics_path.is_file():
            continue
        payload = json.loads(metrics_path.read_text(encoding="utf-8"))
        if not isinstance(payload, dict):
            continue
        payload = dict(payload)
        payload["checkpoint_path"] = str((run_dir / "checkpoints" / f"{alias}.ckpt").resolve())
        payload["result_dir"] = str((run_dir / auto_series_result_dir_name(alias)).resolve())
        test_results[alias] = payload
        best_checkpoints[alias] = {
            "metric_name": str(payload.get("selection_metric", TRACKER_ALIAS_TO_META[alias]["metric_name"])),
            "mode": TRACKER_ALIAS_TO_META[alias]["mode"],
            "best_value": safe_float(payload.get("selection_value", float("nan"))),
            "best_epoch": int(payload.get("best_epoch", -1)),
            "checkpoint_path": str(payload.get("checkpoint_path", "")),
            "artifact_dir": str((run_dir / auto_series_result_dir_name(alias)).resolve()),
        }

    if not test_results:
        raise FileNotFoundError(f"{run_dir} 缺少已完成测试结果")

    trainer_cfg = config_payload.get("trainer", {}) if isinstance(config_payload.get("trainer"), dict) else {}
    return {
        "primary_monitor_metric": str(trainer_cfg.get("monitor_metric", "")),
        "primary_monitor_mode": str(trainer_cfg.get("monitor_mode", "")),
        "primary_best_epoch": -1,
        "primary_best_value": float("nan"),
        "best_checkpoints": best_checkpoints,
        "test_results": test_results,
        "model_name": model_name,
        "train_dir": str(run_dir),
        "train_dir_name": run_dir.name,
    }


def run_auto_model_series(
    *,
    series_label: str,
    progress_desc: str,
    train_cfg: dict[str, Any],
    auto_series_cfg: dict[str, Any],
    model_cfg: dict[str, dict[str, Any]],
    training_context: dict[str, Any],
    model_entries: list[dict[str, Any]],
    seed: int,
    max_epochs: int,
    patience: int,
    image_size: int,
    num_workers: int,
    pretrained: bool,
    use_multi_gpu: bool,
    active_gpu_count: int,
) -> dict[str, Any]:
    if not model_entries:
        raise ValueError(f"没有可运行的 {series_label} 模型")

    session_dir = allocate_auto_series_dir(training_context["output_root"], train_cfg, auto_series_cfg)
    normalize_auto_series_run_dirs(session_dir, model_entries)
    all_results: dict[str, Any] = {
        "session_dir": str(session_dir),
        "models": {},
        "settings": {
            "seed": seed,
            "max_epochs": max_epochs,
            "patience": patience,
            "image_size": image_size,
            "num_workers": num_workers,
            "run_test": bool(auto_series_cfg["run_test"]),
            "output_dir_name": auto_series_cfg["output_dir_name"],
        },
    }
    model_records: list[dict[str, Any]] = []

    print(f"\n[自动 {series_label}] 已开启。")
    print(f"[自动 {series_label}] 任务: {auto_series_cfg['task_name']}")
    print(f"[自动 {series_label}] 输出目录: {session_dir}")
    print(
        f"[自动 {series_label}] 选择指标: "
        f"{auto_series_cfg['selection_alias']} / {auto_series_cfg['selection_metric_name']} "
        f"({auto_series_cfg['selection_mode']})"
    )
    if auto_series_cfg.get("goal"):
        print(f"[自动 {series_label}] 目标: {auto_series_cfg['goal']}")
    print(f"[自动 {series_label}] 模型数量: {len(model_entries)}")

    def persist_state() -> None:
        write_auto_series_notes(session_dir, model_records, all_results, auto_series_cfg)
        write_run_remark(
            session_dir,
            all_results,
            remark_metric_alias=auto_series_cfg["selection_alias"],
            remark_metric_name=auto_series_cfg["selection_metric_name"],
            result_source=auto_series_cfg["result_source"],
            stability_filter=auto_series_cfg["stability_filter"],
            remark_context=auto_series_cfg["remark"],
        )
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    iterator = (
        tqdm(model_entries, desc=progress_desc, dynamic_ncols=True)
        if tqdm is not None
        else model_entries
    )
    for run_index, entry in enumerate(iterator, start=1):
        run_dir = session_dir / auto_baseline_run_dir_name(run_index, entry["name"])
        run_dir.mkdir(parents=True, exist_ok=True)
        base_model_name = resolve_series_entry_model_name(entry)
        entry_seed = int(entry["seed"]) if entry.get("seed") not in (None, "") else seed + run_index
        entry_run_overrides = dict(entry.get("run_overrides", {}))
        entry_num_workers = int(entry_run_overrides.get("num_workers", num_workers))
        if entry_num_workers < 0:
            entry_num_workers = num_workers
        entry_use_multi_gpu = use_multi_gpu
        if "disable_multi_gpu" in entry_run_overrides and bool(entry_run_overrides.get("disable_multi_gpu")):
            entry_use_multi_gpu = False
        elif "use_multi_gpu" in entry_run_overrides:
            entry_use_multi_gpu = bool(entry_run_overrides.get("use_multi_gpu"))
        entry_device_ids = [
            int(device_id)
            for device_id in entry_run_overrides.get("data_parallel_device_ids", [])
        ]
        if entry_use_multi_gpu and entry_device_ids:
            entry_active_gpu_count = len(entry_device_ids)
        else:
            entry_active_gpu_count = active_gpu_count if entry_use_multi_gpu else (1 if torch.cuda.is_available() else 0)

        print(
            f"\n[自动 {series_label}] "
            f"{run_index}/{len(model_entries)} | model={entry['name']} | train_dir={run_dir.name}"
        )
        if base_model_name != entry["name"]:
            print(f"[自动 {series_label}] base_model: {base_model_name}")
        if entry["model_params"]:
            print(f"[自动 {series_label}] model_params: {format_param_overrides(entry['model_params'])}")
        if entry["run_overrides"]:
            print(f"[自动 {series_label}] run_overrides: {format_param_overrides(entry['run_overrides'])}")
        print(f"[自动 {series_label}] seed: {entry_seed}")
        if (
            entry_num_workers != num_workers
            or entry_use_multi_gpu != use_multi_gpu
            or bool(entry_device_ids)
        ):
            print(
                f"[自动 {series_label}] 资源覆盖: "
                f"num_workers={entry_num_workers}, use_multi_gpu={entry_use_multi_gpu}, "
                f"device_ids={entry_device_ids or 'all'}, active_gpu_count={entry_active_gpu_count}"
            )

        record = build_auto_series_record(entry, run_dir)
        record["seed"] = entry_seed
        try:
            if is_auto_series_run_complete(run_dir):
                record["run_action"] = "skip_completed"
                print(f"[自动 {series_label}] 已存在完整测试结果，跳过训练。")
                result = load_existing_auto_series_result(run_dir, entry["name"])
            else:
                resume_path = auto_series_resume_checkpoint(run_dir)
                if resume_path is not None:
                    record["run_action"] = "resume"
                    print(f"[自动 {series_label}] 检测到未完成训练，使用 last.ckpt 继续训练。")
                else:
                    if any(run_dir.iterdir()):
                        print(f"[自动 {series_label}] 检测到不完整残留且无断点，将清理后重跑。")
                        reset_auto_series_run_dir(run_dir)
                        run_dir.mkdir(parents=True, exist_ok=True)
                    record["run_action"] = "restart"
                    resume_path = None

                result = run_model_job(
                    model_name=base_model_name,
                    run_dir=run_dir,
                    train_cfg=train_cfg,
                    model_cfg=model_cfg,
                    training_context=training_context,
                    seed=entry_seed,
                    max_epochs=max_epochs,
                    patience=patience,
                    image_size=image_size,
                    num_workers=entry_num_workers,
                    pretrained=pretrained,
                    use_multi_gpu=entry_use_multi_gpu,
                    active_gpu_count=entry_active_gpu_count,
                    run_test=bool(auto_series_cfg["run_test"]),
                    run_overrides=entry_run_overrides,
                    model_param_override=entry["model_params"],
                    entry_metadata=entry.get("metadata", {}),
                    resume_path=resume_path,
                )
            all_results["models"][entry["name"]] = result
            enrich_auto_series_record(
                record,
                result=result,
                auto_series_cfg=auto_series_cfg,
            )
        except KeyboardInterrupt as exc:
            record["status"] = "interrupted"
            record["error_message"] = str(exc) or "用户中断"
            log_analysis = analyze_training_log(
                run_dir / "log.csv",
                auto_series_cfg["stability_filter"],
            )
            record.update(log_analysis)
            record["evaluation"] = summarize_model_evaluation(
                log_analysis,
                auto_series_cfg["stability_filter"],
            )
            print(f"[自动 {series_label}] {entry['name']} 被中断，可下次继续。")
            raise
        except Exception as exc:
            record["status"] = "failed"
            record["error_message"] = str(exc)
            log_analysis = analyze_training_log(
                run_dir / "log.csv",
                auto_series_cfg["stability_filter"],
            )
            record.update(log_analysis)
            record["evaluation"] = summarize_model_evaluation(
                log_analysis,
                auto_series_cfg["stability_filter"],
            )
            print(f"[自动 {series_label}] {entry['name']} 失败：{exc}")
        finally:
            model_records.append(record)
            persist_state()
            if hasattr(iterator, "set_postfix"):
                best_evaluations = build_model_evaluations(
                    all_results,
                    remark_metric_alias=auto_series_cfg["selection_alias"],
                    remark_metric_name=auto_series_cfg["selection_metric_name"],
                    result_source=auto_series_cfg["result_source"],
                    stability_filter=auto_series_cfg["stability_filter"],
                )
                current_best = select_best_candidate(
                    best_evaluations,
                    mode=auto_series_cfg["selection_mode"],
                ) if best_evaluations else None
                if current_best is not None:
                    iterator.set_postfix(best=f"{float(current_best['score']):.4f}")

    print(f"\n自动 {series_label} 完成。")
    print(f"自动 {series_label} 备注：{session_dir / 'remark.txt'}")
    print(f"自动 {series_label} 摘要：{session_dir / 'notes.json'}")
    return {
        "session_dir": session_dir,
        "all_results": all_results,
        "model_records": model_records,
        "auto_series_cfg": auto_series_cfg,
    }


def run_auto_baselines(
    *,
    train_cfg: dict[str, Any],
    auto_baselines_cfg: dict[str, Any],
    model_cfg: dict[str, dict[str, Any]],
    training_context: dict[str, Any],
    model_entries: list[dict[str, Any]],
    seed: int,
    max_epochs: int,
    patience: int,
    image_size: int,
    num_workers: int,
    pretrained: bool,
    use_multi_gpu: bool,
    active_gpu_count: int,
) -> None:
    run_auto_model_series(
        series_label="Baselines",
        progress_desc="auto-baselines",
        train_cfg=train_cfg,
        auto_series_cfg=auto_baselines_cfg,
        model_cfg=model_cfg,
        training_context=training_context,
        model_entries=model_entries,
        seed=seed,
        max_epochs=max_epochs,
        patience=patience,
        image_size=image_size,
        num_workers=num_workers,
        pretrained=pretrained,
        use_multi_gpu=use_multi_gpu,
        active_gpu_count=active_gpu_count,
    )


def run_auto_sotas(
    *,
    train_cfg: dict[str, Any],
    auto_sotas_cfg: dict[str, Any],
    model_cfg: dict[str, dict[str, Any]],
    training_context: dict[str, Any],
    model_entries: list[dict[str, Any]],
    seed: int,
    max_epochs: int,
    patience: int,
    image_size: int,
    num_workers: int,
    pretrained: bool,
    use_multi_gpu: bool,
    active_gpu_count: int,
) -> None:
    run_auto_model_series(
        series_label="SOTAs",
        progress_desc="auto-sotas",
        train_cfg=train_cfg,
        auto_series_cfg=auto_sotas_cfg,
        model_cfg=model_cfg,
        training_context=training_context,
        model_entries=model_entries,
        seed=seed,
        max_epochs=max_epochs,
        patience=patience,
        image_size=image_size,
        num_workers=num_workers,
        pretrained=pretrained,
        use_multi_gpu=use_multi_gpu,
        active_gpu_count=active_gpu_count,
    )


def task1_search_root_output_name(experiment_cfg: dict[str, Any], search_cfg: dict[str, Any]) -> str:
    output_root_dir_name = str(experiment_cfg["output_dir_name"]).split("/", 1)[0]
    search_dir_name = str(search_cfg.get("output_dir_name", "exp_task1_auto_module_instance_search")).strip()
    return f"{output_root_dir_name}/{search_dir_name}"


def task1_search_root_dir(
    *,
    output_root: Path,
    train_cfg: dict[str, Any],
    experiment_cfg: dict[str, Any],
    search_cfg: dict[str, Any],
) -> Path:
    task_dir = output_root / train_cfg["train_run_dir_name"] / experiment_cfg["task_dir_name"]
    experiment_dir_name = str(train_cfg.get("experiment_dir_name", "")).strip()
    if experiment_dir_name:
        task_dir = task_dir / experiment_dir_name
    return task_dir / task1_search_root_output_name(experiment_cfg, search_cfg)


def train_max_batch_instances_for_search(train_cfg: dict[str, Any], train_max_instances: int) -> int:
    default_run = train_cfg["default_run"]
    default_instances = max(1, int(default_run.get("train_max_instances", 16)))
    default_batch_instances = max(1, int(default_run.get("train_max_batch_instances", default_instances)))
    multiplier = max(1, round(default_batch_instances / default_instances))
    return int(train_max_instances) * multiplier


def clone_task1_search_entry(
    entry: dict[str, Any],
    *,
    name: str | None,
    display_name: str | None,
    search_stage: str,
    train_max_instances: int,
    train_cfg: dict[str, Any],
    seed: int | None,
    summary_order: int,
) -> dict[str, Any]:
    cloned = copy.deepcopy(entry)
    if name:
        cloned["name"] = name
    if display_name:
        cloned["display_name"] = display_name
    if seed is not None:
        cloned["seed"] = int(seed)

    train_max_batch_instances = train_max_batch_instances_for_search(train_cfg, train_max_instances)
    run_overrides = dict(cloned.get("run_overrides", {}))
    run_overrides.update(
        {
            "train_max_instances": int(train_max_instances),
            "train_max_batch_instances": int(train_max_batch_instances),
        }
    )
    cloned["run_overrides"] = run_overrides

    metadata = dict(cloned.get("metadata", {}))
    original_experiment_name = str(metadata.get("experiment_name", entry.get("name", ""))).strip()
    original_summary_name = str(metadata.get("summary_name", entry.get("display_name", entry.get("name", "")))).strip()
    metadata.update(
        {
            "experiment_name": cloned["name"],
            "summary_name": cloned["display_name"],
            "seed_group_name": cloned["name"],
            "summary_order": int(summary_order),
            "search_stage": search_stage,
            "original_experiment_name": original_experiment_name,
            "original_summary_name": original_summary_name,
            "train_max_instances": int(train_max_instances),
            "train_max_batch_instances": int(train_max_batch_instances),
        }
    )
    cloned["metadata"] = metadata
    return cloned


def build_task1_search_stage_cfg(
    experiment_cfg: dict[str, Any],
    search_cfg: dict[str, Any],
    *,
    stage_dir_name: str,
    display_name: str,
    goal: str,
    models: list[dict[str, Any]],
) -> dict[str, Any]:
    stage_cfg = copy.deepcopy(experiment_cfg)
    stage_cfg.update(
        {
            "name": f"{experiment_cfg['name']}_{stage_dir_name}",
            "display_name": display_name,
            "goal": goal,
            "output_dir_name": f"{task1_search_root_output_name(experiment_cfg, search_cfg)}/{stage_dir_name}",
            "models": models,
            "remark": {
                **dict(experiment_cfg.get("remark", {})),
                "focus": goal,
            },
        }
    )
    return stage_cfg


def task1_search_select_best(
    stage_result: dict[str, Any],
    stage_cfg: dict[str, Any],
) -> dict[str, Any]:
    evaluations = build_model_evaluations(
        stage_result["all_results"],
        remark_metric_alias=stage_cfg["selection_alias"],
        remark_metric_name=stage_cfg["selection_metric_name"],
        result_source=stage_cfg["result_source"],
        stability_filter=stage_cfg["stability_filter"],
    )
    best_candidate = select_best_candidate(evaluations, mode=stage_cfg["selection_mode"])
    if best_candidate is None:
        raise RuntimeError(f"{stage_cfg['display_name']} 未产出可用的最优结果")
    return best_candidate


def task1_search_stage_rows(
    stage_name: str,
    stage_result: dict[str, Any],
    stage_cfg: dict[str, Any],
    entry_map: dict[str, dict[str, Any]],
) -> list[dict[str, Any]]:
    evaluations = build_model_evaluations(
        stage_result["all_results"],
        remark_metric_alias=stage_cfg["selection_alias"],
        remark_metric_name=stage_cfg["selection_metric_name"],
        result_source=stage_cfg["result_source"],
        stability_filter=stage_cfg["stability_filter"],
    )
    rows: list[dict[str, Any]] = []
    for item in evaluations:
        entry = entry_map.get(str(item.get("model_name", "")), {})
        metadata = entry.get("metadata", {}) if isinstance(entry.get("metadata"), dict) else {}
        run_overrides = entry.get("run_overrides", {}) if isinstance(entry.get("run_overrides"), dict) else {}
        model_params = entry.get("model_params", {}) if isinstance(entry.get("model_params"), dict) else {}
        label_graph_type = (
            str(model_params.get("label_graph_type", "")).strip()
            if bool(model_params.get("use_label_graph", False))
            else "none"
        )
        rows.append(
            {
                "stage": stage_name,
                "model_name": item.get("model_name", ""),
                "display_name": entry.get("display_name", item.get("model_name", "")),
                "original_experiment_name": metadata.get("original_experiment_name", item.get("model_name", "")),
                "label_graph_type": label_graph_type,
                "use_label_graph": model_params.get("use_label_graph", ""),
                "train_max_instances": run_overrides.get("train_max_instances", metadata.get("train_max_instances", "")),
                "train_max_batch_instances": run_overrides.get(
                    "train_max_batch_instances",
                    metadata.get("train_max_batch_instances", ""),
                ),
                "score": item.get("score", float("nan")),
                "macro_f1": item.get("macro_f1", float("nan")),
                "micro_f1": item.get("micro_f1", float("nan")),
                "best_epoch": item.get("best_epoch", -1),
                "train_dir": item.get("train_dir", ""),
                "train_dir_path": item.get("train_dir_path", ""),
                "checkpoint_path": item.get("checkpoint_path", ""),
                "evaluation": item.get("evaluation", ""),
            }
        )
    return rows


def best_task1_search_entry_from_final_best(
    final_best: dict[str, Any],
    module_entries: list[dict[str, Any]],
) -> dict[str, Any]:
    model_name = str(final_best.get("model_name", "")).strip()
    matched = next((item for item in module_entries if item["name"] == model_name), None)
    if matched is not None:
        return matched
    original_name = str(final_best.get("original_experiment_name", "")).strip()
    matched = next((item for item in module_entries if item["name"] == original_name), None)
    if matched is not None:
        return matched
    raise RuntimeError(f"无法从最终最优结果定位模块配置: {model_name}")


def task1_final_suite_model_params(model_name: str) -> dict[str, Any]:
    common = {
        "backbone_name": "convnext_tiny",
        "freeze_stages": 1,
        "feature_dim": 512,
        "dropout": 0.2,
    }
    if model_name == "gastro_attention_mil_baseline":
        return {**common, "attn_dim": 256}
    if model_name == "gastro_mean_pool_baseline":
        return {**common, "hidden_dim": 512}
    if model_name == "gastro_max_pool_baseline":
        return {**common, "hidden_dim": 256}
    if model_name == "gastro_transformer_mil_baseline":
        return {**common, "attn_dim": 256, "num_heads": 8, "num_layers": 2}
    if model_name == "gastro_topk_mil_baseline":
        return {**common, "hidden_dim": 256, "topk": 4}
    if model_name == "gastro_transmil_sota":
        return {**common, "hidden_dim": 256, "num_heads": 8, "num_layers": 2}
    if model_name == "gastro_dsmil_sota":
        return {**common, "hidden_dim": 256}
    if model_name == "gastro_dtfd_mil_sota":
        return {**common, "attn_dim": 256, "hidden_dim": 256, "num_groups": 4, "pseudo_bag_weight": 0.2}
    if model_name == "gastro_clam_mb_sota":
        return {
            **common,
            "attn_dim": 256,
            "hidden_dim": 256,
            "instance_topk": 4,
            "attention_entropy_weight": 0.05,
            "attention_diversity_weight": 0.05,
            "instance_clustering_weight": 0.2,
        }
    if model_name == "gastro_clam_sb_sota":
        return {
            **common,
            "hidden_dim": 256,
            "instance_topk": 4,
            "attention_entropy_weight": 0.05,
            "instance_clustering_weight": 0.2,
        }
    raise ValueError(f"未知最终模型套件模型名: {model_name}")


def make_task1_final_suite_entry(
    *,
    name: str,
    base_model_name: str,
    display_name: str,
    model_params: dict[str, Any],
    train_cfg: dict[str, Any],
    train_max_instances: int,
    seed: int | None,
    summary_order: int,
) -> dict[str, Any]:
    train_max_batch_instances = train_max_batch_instances_for_search(train_cfg, train_max_instances)
    task_meta = resolve_model_task_meta(base_model_name, "task1")
    return {
        "name": name,
        "base_model_name": base_model_name,
        "display_name": display_name,
        "enabled": True,
        "seed": int(seed) if seed is not None else None,
        "metadata": {
            "experiment_name": name,
            "summary_name": display_name,
            "seed_group_name": name,
            "summary_order": int(summary_order),
            "search_stage": "final_models_best_params",
            "train_max_instances": int(train_max_instances),
            "train_max_batch_instances": int(train_max_batch_instances),
            "backbone": "convnext_tiny",
        },
        "model_params": model_params,
        "run_overrides": {
            "train_max_instances": int(train_max_instances),
            "train_max_batch_instances": int(train_max_batch_instances),
        },
        "task_name": task_meta["task_name"],
        "task_dir_name": task_meta["task_dir_name"],
    }


def build_task1_final_model_suite_entries(
    *,
    final_best: dict[str, Any],
    module_entries: list[dict[str, Any]],
    train_cfg: dict[str, Any],
    train_max_instances: int,
    seed: int | None,
) -> list[dict[str, Any]]:
    best_label_graph_entry = copy.deepcopy(best_task1_search_entry_from_final_best(final_best, module_entries))
    best_label_graph_params = dict(best_label_graph_entry.get("model_params", {}))
    best_label_graph_params["backbone_name"] = "convnext_tiny"
    suite_specs = [
        ("final_label_graph_mil", "gastro_label_graph_mil", "Label graph MIL", best_label_graph_params),
        ("final_attention_mil", "gastro_attention_mil_baseline", "Attention MIL", task1_final_suite_model_params("gastro_attention_mil_baseline")),
        ("final_mean_pooling", "gastro_mean_pool_baseline", "Mean pooling", task1_final_suite_model_params("gastro_mean_pool_baseline")),
        ("final_transformer_context_mil", "gastro_transformer_mil_baseline", "Transformer-context MIL", task1_final_suite_model_params("gastro_transformer_mil_baseline")),
        ("final_topk_mil", "gastro_topk_mil_baseline", "Top-k MIL", task1_final_suite_model_params("gastro_topk_mil_baseline")),
        ("final_max_pooling", "gastro_max_pool_baseline", "Max pooling", task1_final_suite_model_params("gastro_max_pool_baseline")),
        ("final_transmil", "gastro_transmil_sota", "TransMIL", task1_final_suite_model_params("gastro_transmil_sota")),
        ("final_dsmil", "gastro_dsmil_sota", "DSMIL", task1_final_suite_model_params("gastro_dsmil_sota")),
        ("final_dtfd_mil", "gastro_dtfd_mil_sota", "DTFD-MIL", task1_final_suite_model_params("gastro_dtfd_mil_sota")),
        ("final_clam_mb", "gastro_clam_mb_sota", "CLAM-MB", task1_final_suite_model_params("gastro_clam_mb_sota")),
        ("final_clam_sb", "gastro_clam_sb_sota", "CLAM-SB", task1_final_suite_model_params("gastro_clam_sb_sota")),
    ]
    return [
        make_task1_final_suite_entry(
            name=name,
            base_model_name=base_model_name,
            display_name=display_name,
            model_params=model_params,
            train_cfg=train_cfg,
            train_max_instances=train_max_instances,
            seed=seed,
            summary_order=index,
        )
        for index, (name, base_model_name, display_name, model_params) in enumerate(suite_specs, start=1)
    ]


def write_task1_module_instance_search_summary(
    search_root: Path,
    *,
    rows: list[dict[str, Any]],
    stage1_best: dict[str, Any],
    stage2_best: dict[str, Any],
    final_best: dict[str, Any],
    final_suite_best: dict[str, Any] | None = None,
    initial_train_max_instances: int,
    best_train_max_instances: int,
    reran_stage3: bool,
    ran_final_suite: bool = False,
) -> None:
    search_root.mkdir(parents=True, exist_ok=True)
    csv_fields = [
        "stage",
        "model_name",
        "display_name",
        "original_experiment_name",
        "label_graph_type",
        "use_label_graph",
        "train_max_instances",
        "train_max_batch_instances",
        "score",
        "macro_f1",
        "micro_f1",
        "best_epoch",
        "train_dir",
        "train_dir_path",
        "checkpoint_path",
        "evaluation",
    ]
    with (search_root / "module_instance_search_summary.csv").open("w", encoding="utf-8-sig", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=csv_fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in csv_fields})

    summary_payload = {
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "initial_train_max_instances": int(initial_train_max_instances),
        "best_train_max_instances": int(best_train_max_instances),
        "reran_stage3": bool(reran_stage3),
        "stage1_best_module_at_initial_instances": stage1_best,
        "stage2_best_train_max_instances": stage2_best,
        "final_best_module": final_best,
        "final_suite_best_model": final_suite_best or {},
        "ran_final_model_suite": bool(ran_final_suite),
        "rows": rows,
    }
    (search_root / "module_instance_search_summary.json").write_text(
        json.dumps(to_builtin_type(summary_payload), ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    lines = [
        "# TASK1 模块与 train_max_instances 自动探索",
        "",
        f"- 初始模块消融 train_max_instances: `{initial_train_max_instances}`",
        f"- 最优 train_max_instances: `{best_train_max_instances}`",
        f"- 是否回跑最佳实例数下的模块消融: `{'yes' if reran_stage3 else 'no'}`",
        f"- 是否运行最终模型套件: `{'yes' if ran_final_suite else 'no'}`",
        f"- 阶段 1 最优模块: `{stage1_best.get('model_name', '')}`，macro_f1={format_metric_text(stage1_best.get('score'))}",
        f"- 最终最优模块: `{final_best.get('model_name', '')}`，macro_f1={format_metric_text(final_best.get('score'))}",
    ]
    if final_suite_best is not None:
        lines.append(
            f"- 最终模型套件最优模型: `{final_suite_best.get('model_name', '')}`，"
            f"macro_f1={format_metric_text(final_suite_best.get('score'))}"
        )
    lines.extend(
        [
            "",
            "| 阶段 | 模型/配置 | graph type | train_max_instances | macro F1 | micro F1 | best epoch | 训练目录 |",
            "|---|---|---|---:|---:|---:|---:|---|",
        ]
    )
    for row in rows:
        lines.append(
            "| "
            f"{row.get('stage', '')} | "
            f"{row.get('display_name', row.get('model_name', ''))} | "
            f"{row.get('label_graph_type', '')} | "
            f"{row.get('train_max_instances', '')} | "
            f"{format_metric_text(row.get('macro_f1'))} | "
            f"{format_metric_text(row.get('micro_f1'))} | "
            f"{row.get('best_epoch', '')} | "
            f"{row.get('train_dir', '')} |"
        )
    (search_root / "module_instance_search_summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def run_task1_module_instance_search(
    *,
    experiment_cfg: dict[str, Any],
    train_cfg: dict[str, Any],
    model_cfg: dict[str, dict[str, Any]],
    training_context: dict[str, Any],
    seed: int,
    max_epochs: int,
    patience: int,
    image_size: int,
    num_workers: int,
    pretrained: bool,
    use_multi_gpu: bool,
    active_gpu_count: int,
) -> None:
    search_cfg = experiment_cfg.get("task1_module_instance_search", {})
    if not bool(search_cfg.get("enabled", False)):
        raise ValueError("task1_module_instance_search 未启用")

    module_entries = [item for item in experiment_cfg["models"] if item["enabled"]]
    if not module_entries:
        raise ValueError("TASK1 模块与实例数自动探索缺少模块配置")

    fixed_seed = seed if bool(search_cfg.get("fixed_seed", True)) else None
    initial_instances = int(search_cfg["initial_train_max_instances"])
    instance_values = [int(item) for item in search_cfg["train_max_instances_values"]]

    print("\n[TASK1 自动探索] 模块与 train_max_instances 联合探索已开启。")
    print(f"[TASK1 自动探索] 阶段 1: train_max_instances={initial_instances}，模块数量={len(module_entries)}")

    stage1_entries = [
        clone_task1_search_entry(
            entry,
            name=entry["name"],
            display_name=entry["display_name"],
            search_stage="stage1_module_at_initial_instances",
            train_max_instances=initial_instances,
            train_cfg=train_cfg,
            seed=fixed_seed,
            summary_order=index,
        )
        for index, entry in enumerate(module_entries, start=1)
    ]
    stage1_cfg = build_task1_search_stage_cfg(
        experiment_cfg,
        search_cfg,
        stage_dir_name=f"stage1_modules_train_max_instances{initial_instances}",
        display_name=f"Stage1 Modules @ train_max_instances={initial_instances}",
        goal=f"固定 train_max_instances={initial_instances}，比较 10 个 label graph reasoner 模块。",
        models=stage1_entries,
    )
    stage1_result = run_auto_model_series(
        series_label="TASK1Search/Stage1Modules",
        progress_desc="task1-search-stage1-modules",
        train_cfg=train_cfg,
        auto_series_cfg=stage1_cfg,
        model_cfg=model_cfg,
        training_context=training_context,
        model_entries=stage1_entries,
        seed=seed + 11000,
        max_epochs=max_epochs,
        patience=patience,
        image_size=image_size,
        num_workers=num_workers,
        pretrained=pretrained,
        use_multi_gpu=use_multi_gpu,
        active_gpu_count=active_gpu_count,
    )
    stage1_best = task1_search_select_best(stage1_result, stage1_cfg)
    best_module_name = str(stage1_best["model_name"])
    best_module_entry = next((item for item in module_entries if item["name"] == best_module_name), None)
    if best_module_entry is None:
        raise RuntimeError(f"阶段 1 最优模块 {best_module_name} 不在模块配置中")

    print(
        f"[TASK1 自动探索] 阶段 1 最优模块: {best_module_name} "
        f"macro_f1={format_metric_text(stage1_best.get('score'))}"
    )
    print(f"[TASK1 自动探索] 阶段 2: 固定 {best_module_name}，比较 train_max_instances={instance_values}")

    stage2_entries: list[dict[str, Any]] = []
    for index, instances in enumerate(instance_values, start=1):
        name = f"train_max_instances_{instances:02d}_{best_module_name}"
        display_name = f"{best_module_entry['display_name']} | train_max_instances={instances}"
        stage2_entries.append(
            clone_task1_search_entry(
                best_module_entry,
                name=name,
                display_name=display_name,
                search_stage="stage2_train_max_instances",
                train_max_instances=instances,
                train_cfg=train_cfg,
                seed=fixed_seed,
                summary_order=index,
            )
        )
    stage2_cfg = build_task1_search_stage_cfg(
        experiment_cfg,
        search_cfg,
        stage_dir_name=f"stage2_train_max_instances_for_{best_module_name}",
        display_name=f"Stage2 train_max_instances for {best_module_name}",
        goal=f"固定阶段 1 最优模块 {best_module_name}，比较 train_max_instances={instance_values}。",
        models=stage2_entries,
    )
    stage2_result = run_auto_model_series(
        series_label="TASK1Search/Stage2Instances",
        progress_desc="task1-search-stage2-instances",
        train_cfg=train_cfg,
        auto_series_cfg=stage2_cfg,
        model_cfg=model_cfg,
        training_context=training_context,
        model_entries=stage2_entries,
        seed=seed + 12000,
        max_epochs=max_epochs,
        patience=patience,
        image_size=image_size,
        num_workers=num_workers,
        pretrained=pretrained,
        use_multi_gpu=use_multi_gpu,
        active_gpu_count=active_gpu_count,
    )
    stage2_best = task1_search_select_best(stage2_result, stage2_cfg)
    stage2_entry_map = {item["name"]: item for item in stage2_entries}
    best_stage2_entry = stage2_entry_map[str(stage2_best["model_name"])]
    best_train_max_instances = int(best_stage2_entry["run_overrides"]["train_max_instances"])
    print(
        f"[TASK1 自动探索] 阶段 2 最优 train_max_instances={best_train_max_instances} "
        f"macro_f1={format_metric_text(stage2_best.get('score'))}"
    )

    rows: list[dict[str, Any]] = []
    rows.extend(task1_search_stage_rows("stage1_modules", stage1_result, stage1_cfg, {item["name"]: item for item in stage1_entries}))
    rows.extend(task1_search_stage_rows("stage2_train_max_instances", stage2_result, stage2_cfg, stage2_entry_map))

    reran_stage3 = False
    final_best = stage1_best
    if (
        bool(search_cfg.get("rerun_modules_if_best_instances_differs", True))
        and best_train_max_instances != initial_instances
    ):
        reran_stage3 = True
        print(
            f"[TASK1 自动探索] 阶段 3: 最优 train_max_instances={best_train_max_instances}，"
            "重新比较 10 个模块。"
        )
        stage3_entries = [
            clone_task1_search_entry(
                entry,
                name=entry["name"],
                display_name=entry["display_name"],
                search_stage="stage3_module_at_best_instances",
                train_max_instances=best_train_max_instances,
                train_cfg=train_cfg,
                seed=fixed_seed,
                summary_order=index,
            )
            for index, entry in enumerate(module_entries, start=1)
        ]
        stage3_cfg = build_task1_search_stage_cfg(
            experiment_cfg,
            search_cfg,
            stage_dir_name=f"stage3_modules_train_max_instances{best_train_max_instances}",
            display_name=f"Stage3 Modules @ train_max_instances={best_train_max_instances}",
            goal=f"固定最佳 train_max_instances={best_train_max_instances}，重新比较 10 个 label graph reasoner 模块。",
            models=stage3_entries,
        )
        stage3_result = run_auto_model_series(
            series_label="TASK1Search/Stage3Modules",
            progress_desc="task1-search-stage3-modules",
            train_cfg=train_cfg,
            auto_series_cfg=stage3_cfg,
            model_cfg=model_cfg,
            training_context=training_context,
            model_entries=stage3_entries,
            seed=seed + 13000,
            max_epochs=max_epochs,
            patience=patience,
            image_size=image_size,
            num_workers=num_workers,
            pretrained=pretrained,
            use_multi_gpu=use_multi_gpu,
            active_gpu_count=active_gpu_count,
        )
        final_best = task1_search_select_best(stage3_result, stage3_cfg)
        rows.extend(task1_search_stage_rows("stage3_modules", stage3_result, stage3_cfg, {item["name"]: item for item in stage3_entries}))

    final_suite_best: dict[str, Any] | None = None
    ran_final_suite = False
    if bool(search_cfg.get("run_final_model_suite", True)):
        ran_final_suite = True
        print(
            "[TASK1 自动探索] 最终模型套件: 使用最佳 train_max_instances "
            f"{best_train_max_instances} 运行 11 个固定模型，backbone=convnext_tiny。"
        )
        final_suite_entries = build_task1_final_model_suite_entries(
            final_best=final_best,
            module_entries=module_entries,
            train_cfg=train_cfg,
            train_max_instances=best_train_max_instances,
            seed=fixed_seed,
        )
        final_suite_dir_name = str(search_cfg.get("final_suite_dir_name", "final_models_best_params")).strip()
        final_suite_cfg = build_task1_search_stage_cfg(
            experiment_cfg,
            search_cfg,
            stage_dir_name=final_suite_dir_name,
            display_name="Final models @ best params",
            goal=(
                "固定自动探索得到的最佳 train_max_instances 和 ConvNeXt-Tiny backbone，"
                "运行最终 11 个模型对比。"
            ),
            models=final_suite_entries,
        )
        final_suite_result = run_auto_model_series(
            series_label="TASK1Search/FinalModels",
            progress_desc="task1-search-final-models",
            train_cfg=train_cfg,
            auto_series_cfg=final_suite_cfg,
            model_cfg=model_cfg,
            training_context=training_context,
            model_entries=final_suite_entries,
            seed=seed + 14000,
            max_epochs=max_epochs,
            patience=patience,
            image_size=image_size,
            num_workers=num_workers,
            pretrained=pretrained,
            use_multi_gpu=use_multi_gpu,
            active_gpu_count=active_gpu_count,
        )
        final_suite_best = task1_search_select_best(final_suite_result, final_suite_cfg)
        rows.extend(
            task1_search_stage_rows(
                "final_models_best_params",
                final_suite_result,
                final_suite_cfg,
                {item["name"]: item for item in final_suite_entries},
            )
        )

    search_root = task1_search_root_dir(
        output_root=Path(training_context["output_root"]),
        train_cfg=train_cfg,
        experiment_cfg=experiment_cfg,
        search_cfg=search_cfg,
    )
    write_task1_module_instance_search_summary(
        search_root,
        rows=rows,
        stage1_best=stage1_best,
        stage2_best=stage2_best,
        final_best=final_best,
        final_suite_best=final_suite_best,
        initial_train_max_instances=initial_instances,
        best_train_max_instances=best_train_max_instances,
        reran_stage3=reran_stage3,
        ran_final_suite=ran_final_suite,
    )
    print(f"[TASK1 自动探索] 汇总目录: {search_root}")
    print(f"[TASK1 自动探索] 最终模块: {final_best.get('model_name', '')}")
    print(f"[TASK1 自动探索] 最优 train_max_instances: {best_train_max_instances}")
    if final_suite_best is not None:
        print(f"[TASK1 自动探索] 最终模型套件最优模型: {final_suite_best.get('model_name', '')}")


def run_auto_ablations(
    *,
    train_cfg: dict[str, Any],
    auto_ablations_experiments: list[dict[str, Any]],
    model_cfg: dict[str, dict[str, Any]],
    training_context: dict[str, Any],
    seed: int,
    max_epochs: int,
    patience: int,
    image_size: int,
    num_workers: int,
    pretrained: bool,
    use_multi_gpu: bool,
    active_gpu_count: int,
) -> None:
    if not auto_ablations_experiments:
        raise ValueError("没有可运行的消融实验")

    print("\n[自动 Ablations] 已开启。")
    print(f"[自动 Ablations] 实验数量: {len(auto_ablations_experiments)}")

    ran_task1_module_instance_search = False
    for experiment_index, experiment_cfg in enumerate(auto_ablations_experiments, start=1):
        enabled_entries = [item for item in experiment_cfg["models"] if item["enabled"]]
        print(
            f"\n[自动 Ablations] {experiment_index}/{len(auto_ablations_experiments)} "
            f"| experiment={experiment_cfg['name']} | {experiment_cfg['display_name']}"
        )
        if bool(experiment_cfg.get("task1_module_instance_search", {}).get("enabled", False)):
            run_task1_module_instance_search(
                experiment_cfg=experiment_cfg,
                train_cfg=train_cfg,
                model_cfg=model_cfg,
                training_context=training_context,
                seed=seed,
                max_epochs=max_epochs,
                patience=patience,
                image_size=image_size,
                num_workers=num_workers,
                pretrained=pretrained,
                use_multi_gpu=use_multi_gpu,
                active_gpu_count=active_gpu_count,
            )
            ran_task1_module_instance_search = True
            continue
        run_auto_model_series(
            series_label=f"Ablations/{experiment_cfg['display_name']}",
            progress_desc=f"auto-ablation-{experiment_cfg['name']}",
            train_cfg=train_cfg,
            auto_series_cfg=experiment_cfg,
            model_cfg=model_cfg,
            training_context=training_context,
            model_entries=enabled_entries,
            seed=seed + experiment_index * 10000,
            max_epochs=max_epochs,
            patience=patience,
            image_size=image_size,
            num_workers=num_workers,
            pretrained=pretrained,
            use_multi_gpu=use_multi_gpu,
            active_gpu_count=active_gpu_count,
        )

    if ran_task1_module_instance_search:
        return

    first_experiment = auto_ablations_experiments[0]
    if first_experiment.get("task_name") == "task1":
        try:
            from scripts.task1_ablation_summary import write_task1_ablation_summaries

            output_root_dir_name = str(first_experiment["output_dir_name"]).split("/", 1)[0]
            task_dir = Path(training_context["output_root"]) / train_cfg["train_run_dir_name"] / first_experiment["task_dir_name"]
            experiment_dir_name = str(train_cfg.get("experiment_dir_name", "")).strip()
            if experiment_dir_name:
                task_dir = task_dir / experiment_dir_name
            write_task1_ablation_summaries(ablation_root=task_dir / output_root_dir_name)
        except Exception as exc:
            print(f"[自动 Ablations] TASK1 消融汇总生成失败：{exc}")


def run_auto_exp1(
    *,
    train_cfg: dict[str, Any],
    auto_exp1_cfg: dict[str, Any],
    model_cfg: dict[str, dict[str, Any]],
    training_context: dict[str, Any],
    model_entries: list[dict[str, Any]],
    seed: int,
    max_epochs: int,
    patience: int,
    image_size: int,
    num_workers: int,
    pretrained: bool,
    use_multi_gpu: bool,
    active_gpu_count: int,
) -> None:
    run_auto_model_series(
        series_label="EXP1",
        progress_desc="auto-exp1",
        train_cfg=train_cfg,
        auto_series_cfg=auto_exp1_cfg,
        model_cfg=model_cfg,
        training_context=training_context,
        model_entries=model_entries,
        seed=seed,
        max_epochs=max_epochs,
        patience=patience,
        image_size=image_size,
        num_workers=num_workers,
        pretrained=pretrained,
        use_multi_gpu=use_multi_gpu,
        active_gpu_count=active_gpu_count,
    )


def run_auto_exp2(
    *,
    train_cfg: dict[str, Any],
    auto_exp2_cfg: dict[str, Any],
    model_cfg: dict[str, dict[str, Any]],
    training_context: dict[str, Any],
    model_entries: list[dict[str, Any]],
    seed: int,
    max_epochs: int,
    patience: int,
    image_size: int,
    num_workers: int,
    pretrained: bool,
    use_multi_gpu: bool,
    active_gpu_count: int,
) -> None:
    run_auto_model_series(
        series_label="EXP2",
        progress_desc="auto-exp2",
        train_cfg=train_cfg,
        auto_series_cfg=auto_exp2_cfg,
        model_cfg=model_cfg,
        training_context=training_context,
        model_entries=model_entries,
        seed=seed,
        max_epochs=max_epochs,
        patience=patience,
        image_size=image_size,
        num_workers=num_workers,
        pretrained=pretrained,
        use_multi_gpu=use_multi_gpu,
        active_gpu_count=active_gpu_count,
    )


def run_auto_exp3(
    *,
    train_cfg: dict[str, Any],
    auto_exp3_cfg: dict[str, Any],
    model_cfg: dict[str, dict[str, Any]],
    training_context: dict[str, Any],
    model_entries: list[dict[str, Any]],
    seed: int,
    max_epochs: int,
    patience: int,
    image_size: int,
    num_workers: int,
    pretrained: bool,
    use_multi_gpu: bool,
    active_gpu_count: int,
) -> None:
    run_auto_model_series(
        series_label="EXP3",
        progress_desc="auto-exp3",
        train_cfg=train_cfg,
        auto_series_cfg=auto_exp3_cfg,
        model_cfg=model_cfg,
        training_context=training_context,
        model_entries=model_entries,
        seed=seed,
        max_epochs=max_epochs,
        patience=patience,
        image_size=image_size,
        num_workers=num_workers,
        pretrained=pretrained,
        use_multi_gpu=use_multi_gpu,
        active_gpu_count=active_gpu_count,
    )


def run_auto_exp4(
    *,
    train_cfg: dict[str, Any],
    auto_exp4_cfg: dict[str, Any],
    model_cfg: dict[str, dict[str, Any]],
    training_context: dict[str, Any],
    model_entries: list[dict[str, Any]],
    seed: int,
    max_epochs: int,
    patience: int,
    image_size: int,
    num_workers: int,
    pretrained: bool,
    use_multi_gpu: bool,
    active_gpu_count: int,
) -> None:
    run_auto_model_series(
        series_label="EXP4",
        progress_desc="auto-exp4",
        train_cfg=train_cfg,
        auto_series_cfg=auto_exp4_cfg,
        model_cfg=model_cfg,
        training_context=training_context,
        model_entries=model_entries,
        seed=seed,
        max_epochs=max_epochs,
        patience=patience,
        image_size=image_size,
        num_workers=num_workers,
        pretrained=pretrained,
        use_multi_gpu=use_multi_gpu,
        active_gpu_count=active_gpu_count,
    )


def run_auto_exp5(
    *,
    train_cfg: dict[str, Any],
    auto_exp5_cfg: dict[str, Any],
    model_cfg: dict[str, dict[str, Any]],
    training_context: dict[str, Any],
    model_entries: list[dict[str, Any]],
    seed: int,
    max_epochs: int,
    patience: int,
    image_size: int,
    num_workers: int,
    pretrained: bool,
    use_multi_gpu: bool,
    active_gpu_count: int,
) -> None:
    roi_summary = prepare_exp5_roi_cache(
        training_context=training_context,
        task_name=str(auto_exp5_cfg["task_name"]),
        roi_cfg=auto_exp5_cfg.get("roi", {}),
    )
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    if not roi_summary.get("enabled", False):
        raise ValueError("auto_exp_5 需要启用 ROI 预处理")

    for entry in model_entries:
        run_overrides = dict(entry.get("run_overrides", {}))
        run_overrides.update(
            {
                "roi_enabled": True,
                "roi_index_path": str(roi_summary["roi_index_path"]),
                "roi_root_dir": str(roi_summary["roi_root_dir"]),
                "roi_max_crops_per_bag": int(run_overrides.get("roi_max_crops_per_bag", 64)),
                "roi_max_crops_per_source": int(run_overrides.get("roi_max_crops_per_source", 1)),
                "roi_min_score": float(run_overrides.get("roi_min_score", 0.0)),
            }
        )
        entry["run_overrides"] = run_overrides

    auto_exp5_cfg = dict(auto_exp5_cfg)
    auto_exp5_cfg["roi_summary"] = roi_summary
    run_auto_model_series(
        series_label="EXP5",
        progress_desc="auto-exp5",
        train_cfg=train_cfg,
        auto_series_cfg=auto_exp5_cfg,
        model_cfg=model_cfg,
        training_context=training_context,
        model_entries=model_entries,
        seed=seed,
        max_epochs=max_epochs,
        patience=patience,
        image_size=image_size,
        num_workers=num_workers,
        pretrained=pretrained,
        use_multi_gpu=use_multi_gpu,
        active_gpu_count=active_gpu_count,
    )


def run_auto_exp6(
    *,
    train_cfg: dict[str, Any],
    auto_exp6_cfg: dict[str, Any],
    model_cfg: dict[str, dict[str, Any]],
    training_context: dict[str, Any],
    model_entries: list[dict[str, Any]],
    seed: int,
    max_epochs: int,
    patience: int,
    image_size: int,
    num_workers: int,
    pretrained: bool,
    use_multi_gpu: bool,
    active_gpu_count: int,
) -> None:
    needs_roi = any(bool(entry.get("run_overrides", {}).get("roi_enabled", False)) for entry in model_entries)
    roi_summary: dict[str, Any] = {"enabled": False, "roi_index_path": "", "roi_root_dir": ""}
    if needs_roi:
        roi_summary = prepare_exp5_roi_cache(
            training_context=training_context,
            task_name=str(auto_exp6_cfg["task_name"]),
            roi_cfg=auto_exp6_cfg.get("roi", {}),
        )
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        if not roi_summary.get("enabled", False):
            raise ValueError("auto_exp_6 中包含 ROI 实验，需要启用 ROI 预处理")

    for entry in model_entries:
        run_overrides = dict(entry.get("run_overrides", {}))
        if bool(run_overrides.get("roi_enabled", False)):
            run_overrides.update(
                {
                    "roi_enabled": True,
                    "roi_index_path": str(roi_summary["roi_index_path"]),
                    "roi_root_dir": str(roi_summary["roi_root_dir"]),
                    "roi_max_crops_per_bag": int(run_overrides.get("roi_max_crops_per_bag", 32)),
                    "roi_max_crops_per_source": int(run_overrides.get("roi_max_crops_per_source", 1)),
                    "roi_min_score": float(run_overrides.get("roi_min_score", 0.0)),
                }
            )
        entry["run_overrides"] = run_overrides

    auto_exp6_cfg = dict(auto_exp6_cfg)
    auto_exp6_cfg["roi_summary"] = roi_summary
    run_auto_model_series(
        series_label="EXP6",
        progress_desc="auto-exp6",
        train_cfg=train_cfg,
        auto_series_cfg=auto_exp6_cfg,
        model_cfg=model_cfg,
        training_context=training_context,
        model_entries=model_entries,
        seed=seed,
        max_epochs=max_epochs,
        patience=patience,
        image_size=image_size,
        num_workers=num_workers,
        pretrained=pretrained,
        use_multi_gpu=use_multi_gpu,
        active_gpu_count=active_gpu_count,
    )


def run_auto_exp7(
    *,
    train_cfg: dict[str, Any],
    auto_exp7_cfg: dict[str, Any],
    model_cfg: dict[str, dict[str, Any]],
    training_context: dict[str, Any],
    model_entries: list[dict[str, Any]],
    seed: int,
    max_epochs: int,
    patience: int,
    image_size: int,
    num_workers: int,
    pretrained: bool,
    use_multi_gpu: bool,
    active_gpu_count: int,
) -> None:
    needs_roi = any(bool(entry.get("run_overrides", {}).get("roi_enabled", False)) for entry in model_entries)
    roi_summary: dict[str, Any] = {"enabled": False, "roi_index_path": "", "roi_root_dir": ""}
    if needs_roi:
        roi_summary = prepare_exp5_roi_cache(
            training_context=training_context,
            task_name=str(auto_exp7_cfg["task_name"]),
            roi_cfg=auto_exp7_cfg.get("roi", {}),
        )
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        if not roi_summary.get("enabled", False):
            raise ValueError("auto_exp_7 中包含 ROI 实验，需要启用 ROI 预处理")

    for entry in model_entries:
        run_overrides = dict(entry.get("run_overrides", {}))
        if bool(run_overrides.get("roi_enabled", False)):
            run_overrides.update(
                {
                    "roi_enabled": True,
                    "roi_index_path": str(roi_summary["roi_index_path"]),
                    "roi_root_dir": str(roi_summary["roi_root_dir"]),
                    "roi_max_crops_per_bag": int(run_overrides.get("roi_max_crops_per_bag", 32)),
                    "roi_max_crops_per_source": int(run_overrides.get("roi_max_crops_per_source", 1)),
                    "roi_min_score": float(run_overrides.get("roi_min_score", 0.0)),
                }
            )
        entry["run_overrides"] = run_overrides

    auto_exp7_cfg = dict(auto_exp7_cfg)
    auto_exp7_cfg["roi_summary"] = roi_summary
    run_auto_model_series(
        series_label="EXP7",
        progress_desc="auto-exp7",
        train_cfg=train_cfg,
        auto_series_cfg=auto_exp7_cfg,
        model_cfg=model_cfg,
        training_context=training_context,
        model_entries=model_entries,
        seed=seed,
        max_epochs=max_epochs,
        patience=patience,
        image_size=image_size,
        num_workers=num_workers,
        pretrained=pretrained,
        use_multi_gpu=use_multi_gpu,
        active_gpu_count=active_gpu_count,
    )


def run_auto_exp8(
    *,
    train_cfg: dict[str, Any],
    auto_exp8_cfg: dict[str, Any],
    model_cfg: dict[str, dict[str, Any]],
    training_context: dict[str, Any],
    model_entries: list[dict[str, Any]],
    seed: int,
    max_epochs: int,
    patience: int,
    image_size: int,
    num_workers: int,
    pretrained: bool,
    use_multi_gpu: bool,
    active_gpu_count: int,
) -> None:
    run_auto_model_series(
        series_label="EXP8",
        progress_desc="auto-exp8",
        train_cfg=train_cfg,
        auto_series_cfg=auto_exp8_cfg,
        model_cfg=model_cfg,
        training_context=training_context,
        model_entries=model_entries,
        seed=seed,
        max_epochs=max_epochs,
        patience=patience,
        image_size=image_size,
        num_workers=num_workers,
        pretrained=pretrained,
        use_multi_gpu=use_multi_gpu,
        active_gpu_count=active_gpu_count,
    )


def run_auto_exp8_mm_ablation(
    *,
    train_cfg: dict[str, Any],
    auto_exp8_mm_ablation_cfg: dict[str, Any],
    model_cfg: dict[str, dict[str, Any]],
    training_context: dict[str, Any],
    model_entries: list[dict[str, Any]],
    seed: int,
    max_epochs: int,
    patience: int,
    image_size: int,
    num_workers: int,
    pretrained: bool,
    use_multi_gpu: bool,
    active_gpu_count: int,
) -> None:
    run_auto_model_series(
        series_label="EXP8-MM-Ablation",
        progress_desc="auto-exp8-mm-ablation",
        train_cfg=train_cfg,
        auto_series_cfg=auto_exp8_mm_ablation_cfg,
        model_cfg=model_cfg,
        training_context=training_context,
        model_entries=model_entries,
        seed=seed,
        max_epochs=max_epochs,
        patience=patience,
        image_size=image_size,
        num_workers=num_workers,
        pretrained=pretrained,
        use_multi_gpu=use_multi_gpu,
        active_gpu_count=active_gpu_count,
    )


def run_auto_exp9_ablation(
    *,
    train_cfg: dict[str, Any],
    auto_exp9_ablation_cfg: dict[str, Any],
    model_cfg: dict[str, dict[str, Any]],
    training_context: dict[str, Any],
    model_entries: list[dict[str, Any]],
    seed: int,
    max_epochs: int,
    patience: int,
    image_size: int,
    num_workers: int,
    pretrained: bool,
    use_multi_gpu: bool,
    active_gpu_count: int,
) -> None:
    run_auto_model_series(
        series_label="EXP9-Ablation",
        progress_desc="auto-exp9-ablation",
        train_cfg=train_cfg,
        auto_series_cfg=auto_exp9_ablation_cfg,
        model_cfg=model_cfg,
        training_context=training_context,
        model_entries=model_entries,
        seed=seed,
        max_epochs=max_epochs,
        patience=patience,
        image_size=image_size,
        num_workers=num_workers,
        pretrained=pretrained,
        use_multi_gpu=use_multi_gpu,
        active_gpu_count=active_gpu_count,
    )


def run_auto_exp11_module_ablation(
    *,
    train_cfg: dict[str, Any],
    auto_exp11_module_ablation_cfg: dict[str, Any],
    model_cfg: dict[str, dict[str, Any]],
    training_context: dict[str, Any],
    model_entries: list[dict[str, Any]],
    seed: int,
    max_epochs: int,
    patience: int,
    image_size: int,
    num_workers: int,
    pretrained: bool,
    use_multi_gpu: bool,
    active_gpu_count: int,
) -> None:
    run_auto_model_series(
        series_label="EXP11-Module-Ablation",
        progress_desc="auto-exp11-module-ablation",
        train_cfg=train_cfg,
        auto_series_cfg=auto_exp11_module_ablation_cfg,
        model_cfg=model_cfg,
        training_context=training_context,
        model_entries=model_entries,
        seed=seed,
        max_epochs=max_epochs,
        patience=patience,
        image_size=image_size,
        num_workers=num_workers,
        pretrained=pretrained,
        use_multi_gpu=use_multi_gpu,
        active_gpu_count=active_gpu_count,
    )


def build_auto_explore_trial_record(
    *,
    trial_index: int,
    model_name: str,
    train_dir: Path,
    auto_explore_cfg: dict[str, Any],
    run_overrides: dict[str, Any],
) -> dict[str, Any]:
    trial_record: dict[str, Any] = {
        "trial_index": trial_index,
        "status": "success",
        "model_name": model_name,
        "train_dir": train_dir.name,
        "train_dir_path": str(train_dir),
        "search_method": auto_explore_cfg.get("search_method", ""),
        "selection_alias": auto_explore_cfg["selection_alias"],
        "selection_metric": auto_explore_cfg["selection_metric_name"],
        "selection_mode": auto_explore_cfg["selection_mode"],
        "objective_name": auto_explore_cfg["objective"]["name"],
        "objective_mode": auto_explore_cfg["objective"]["mode"],
        "best_epoch": -1,
        "best_score": float("nan"),
        "objective_score": float("nan"),
        "checkpoint_path": "",
        "epochs_trained": 0,
        "best_val_loss": float("nan"),
        "best_val_epoch": -1,
        "final_train_loss": float("nan"),
        "final_val_loss": float("nan"),
        "final_gap": float("nan"),
        "val_loss_rebound": float("nan"),
        "val_loss_rebound_ratio": float("nan"),
        "tail_epochs": 0,
        "tail_val_loss_mean": float("nan"),
        "tail_val_loss_std": float("nan"),
        "stable_convergence": False,
        "evaluation": "",
        "test_results": {},
        "error_message": "",
    }
    for key, value in sorted(run_overrides.items()):
        trial_record[f"param_{key}"] = value
    return trial_record


def enrich_auto_explore_trial_record(
    trial_record: dict[str, Any],
    *,
    trial_dir: Path,
    auto_explore_cfg: dict[str, Any],
) -> None:
    log_analysis = analyze_training_log(
        trial_dir / "log.csv",
        auto_explore_cfg["stability_filter"],
        objective_cfg=auto_explore_cfg["objective"],
    )
    trial_record.update(log_analysis)
    trial_record["evaluation"] = summarize_model_evaluation(
        log_analysis,
        auto_explore_cfg["stability_filter"],
    )


def build_optuna_validation_callback(trial: Any) -> Callable[[int, float, dict[str, Any]], None]:
    if optuna is None:
        raise ModuleNotFoundError("未安装 optuna，请先执行 `pip install optuna` 后再运行自动探索。")

    def _callback(epoch: int, val_loss: float, val_metrics: dict[str, Any]) -> None:
        del val_metrics
        report_value = safe_float(val_loss)
        if np.isnan(report_value):
            report_value = float("inf")
        trial.report(report_value, step=int(epoch))
        if trial.should_prune():
            raise optuna.TrialPruned(f"epoch={epoch}, val_loss={report_value:.6f}")

    return _callback


def run_auto_explore(
    *,
    train_cfg: dict[str, Any],
    auto_explore_cfg: dict[str, Any],
    model_cfg: dict[str, dict[str, Any]],
    training_context: dict[str, Any],
    model_name: str,
    seed: int,
    base_max_epochs: int,
    base_patience: int,
    image_size: int,
    num_workers: int,
    pretrained: bool,
    use_multi_gpu: bool,
    active_gpu_count: int,
) -> None:
    active_search_params = [
        name for name, spec in auto_explore_cfg["search_space"].items() if bool(spec.get("enabled", False))
    ]
    if not active_search_params:
        raise ValueError("auto_explore=true 时，至少需要开启一个 search_space 参数")

    output_root = training_context["output_root"]
    session_dir, run_meta = allocate_task_run_dir(
        output_root,
        train_cfg,
        model_name,
        is_auto_explore=True,
    )
    run_context = {
        **run_meta,
        "model_name": model_name,
    }

    trial_max_epochs = int(auto_explore_cfg["trial_max_epochs"]) if int(auto_explore_cfg["trial_max_epochs"]) > 0 else base_max_epochs
    raw_trial_patience = int(auto_explore_cfg["trial_patience"]) if int(auto_explore_cfg["trial_patience"]) > 0 else base_patience
    trial_patience = min(raw_trial_patience, trial_max_epochs)

    print("\n[自动探索] 已开启。")
    print(f"[自动探索] 任务: {run_meta['task_name']} | 模型: {model_name}")
    print(f"[自动探索] 搜索方式: {auto_explore_cfg['search_method']}")
    print(f"[自动探索] 试验次数: {auto_explore_cfg['num_trials']}")
    print(
        "[自动探索] 选择指标: "
        f"{auto_explore_cfg['selection_alias']} / {auto_explore_cfg['selection_metric_name']} "
        f"({auto_explore_cfg['selection_mode']})"
    )
    print(
        "[自动探索] 排序目标: "
        f"{auto_explore_cfg['objective']['name']} ({auto_explore_cfg['objective']['mode']})"
    )
    if auto_explore_cfg.get("goal"):
        print(f"[自动探索] 目标: {auto_explore_cfg['goal']}")
    print(f"[自动探索] 搜索参数: {', '.join(active_search_params)}")
    print(f"[自动探索] 每个 trial 使用 max_epochs={trial_max_epochs}, patience={trial_patience}")
    print(f"[自动探索] trial_run_test={bool(auto_explore_cfg.get('trial_run_test', False))}")
    print(f"[自动探索] 输出目录: {session_dir}")

    trial_records: list[dict[str, Any]] = []
    def persist_trial_state() -> None:
        write_auto_explore_notes(session_dir, trial_records, auto_explore_cfg, run_context)
        write_auto_explore_remark(session_dir, trial_records, auto_explore_cfg, run_context)
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    def finalize_successful_trial(
        *,
        trial_record: dict[str, Any],
        trial_dir: Path,
        result: dict[str, Any],
    ) -> None:
        all_results = {"models": {model_name: result}}
        candidates = resolve_session_candidate(
            all_results,
            remark_metric_alias=auto_explore_cfg["selection_alias"],
            result_source=auto_explore_cfg["result_source"],
            fallback_metric_name=auto_explore_cfg["selection_metric_name"],
        )
        best_candidate = select_best_candidate(candidates, mode=auto_explore_cfg["selection_mode"])
        if best_candidate is None:
            raise RuntimeError("当前 trial 未产出可比较的验证集结果")

        trial_record["best_epoch"] = best_candidate["best_epoch"]
        trial_record["best_score"] = best_candidate["score"]
        trial_record["checkpoint_path"] = best_candidate["checkpoint_path"]
        trial_record["test_results"] = result.get("test_results", {})
        enrich_auto_explore_trial_record(
            trial_record,
            trial_dir=trial_dir,
            auto_explore_cfg=auto_explore_cfg,
        )
        objective_score = safe_float(trial_record.get("objective_score", float("nan")))
        if not np.isfinite(objective_score):
            raise RuntimeError("当前 trial 未产出可比较的稳定收敛目标分数")

        print(
            "[自动探索] 当前 trial 最优: "
            f"train_dir={trial_dir.name}, "
            f"objective={format_metric_text(trial_record.get('objective_score'), digits=6)}, "
            f"score={format_metric_text(best_candidate['score'], digits=6)}, "
            f"epoch={best_candidate['best_epoch']}, "
            f"stable={trial_record['stable_convergence']}"
        )

    if auto_explore_cfg["search_method"] == "random":
        rng = random.Random(seed)
        progress = iter_trial_progress(int(auto_explore_cfg["num_trials"]), desc=f"{run_meta['run_prefix']}-auto")

        for trial_index in progress:
            run_overrides = sample_auto_explore_overrides(auto_explore_cfg, rng)
            trial_dir = session_dir / auto_train_dir_name(trial_index)
            trial_dir.mkdir(parents=True, exist_ok=True)

            print(
                "\n[自动探索] "
                f"Trial {trial_index:03d}/{int(auto_explore_cfg['num_trials']):03d} "
                f"| {format_param_overrides(run_overrides)}"
            )

            trial_record = build_auto_explore_trial_record(
                trial_index=trial_index,
                model_name=model_name,
                train_dir=trial_dir,
                auto_explore_cfg=auto_explore_cfg,
                run_overrides=run_overrides,
            )

            try:
                result = run_model_job(
                    model_name=model_name,
                    run_dir=trial_dir,
                    train_cfg=train_cfg,
                    model_cfg=model_cfg,
                    training_context=training_context,
                    seed=seed + trial_index,
                    max_epochs=trial_max_epochs,
                    patience=trial_patience,
                    image_size=image_size,
                    num_workers=num_workers,
                    pretrained=pretrained,
                    use_multi_gpu=use_multi_gpu,
                    active_gpu_count=active_gpu_count,
                    run_test=bool(auto_explore_cfg.get("trial_run_test", False)),
                    run_overrides=run_overrides,
                )
                finalize_successful_trial(
                    trial_record=trial_record,
                    trial_dir=trial_dir,
                    result=result,
                )
            except Exception as exc:
                trial_record["status"] = "failed"
                trial_record["error_message"] = str(exc)
                enrich_auto_explore_trial_record(
                    trial_record,
                    trial_dir=trial_dir,
                    auto_explore_cfg=auto_explore_cfg,
                )
                print(f"[自动探索] Trial {trial_index:03d} 失败：{exc}")
            finally:
                trial_records.append(trial_record)
                persist_trial_state()

                successful_records = [item for item in trial_records if item.get("status") == "success"]
                if successful_records and hasattr(progress, "set_postfix"):
                    current_best = select_best_candidate(
                        successful_records,
                        mode=auto_explore_cfg["objective"]["mode"],
                        score_key="objective_score",
                    )
                    if current_best is not None:
                        progress.set_postfix(best=f"{float(current_best['objective_score']):.4f}")
    else:
        if optuna is None:
            raise ModuleNotFoundError("未安装 optuna，请先执行 `pip install optuna` 后再运行自动探索。")

        progress_bar = (
            tqdm(total=int(auto_explore_cfg["num_trials"]), desc=f"{run_meta['run_prefix']}-optuna", dynamic_ncols=True)
            if tqdm is not None
            else None
        )

        def objective(trial: Any) -> float:
            trial_index = int(trial.number) + 1
            run_overrides = suggest_auto_explore_overrides_with_optuna(trial, auto_explore_cfg)
            trial_dir = session_dir / auto_train_dir_name(trial_index)
            trial_dir.mkdir(parents=True, exist_ok=True)

            print(
                "\n[自动探索] "
                f"Trial {trial_index:03d}/{int(auto_explore_cfg['num_trials']):03d} "
                f"| {format_param_overrides(run_overrides)}"
            )

            trial_record = build_auto_explore_trial_record(
                trial_index=trial_index,
                model_name=model_name,
                train_dir=trial_dir,
                auto_explore_cfg=auto_explore_cfg,
                run_overrides=run_overrides,
            )

            try:
                result = run_model_job(
                    model_name=model_name,
                    run_dir=trial_dir,
                    train_cfg=train_cfg,
                    model_cfg=model_cfg,
                    training_context=training_context,
                    seed=seed + trial_index,
                    max_epochs=trial_max_epochs,
                    patience=trial_patience,
                    image_size=image_size,
                    num_workers=num_workers,
                    pretrained=pretrained,
                    use_multi_gpu=use_multi_gpu,
                    active_gpu_count=active_gpu_count,
                    run_test=bool(auto_explore_cfg.get("trial_run_test", False)),
                    run_overrides=run_overrides,
                    on_validation_epoch_end=build_optuna_validation_callback(trial),
                )
                finalize_successful_trial(
                    trial_record=trial_record,
                    trial_dir=trial_dir,
                    result=result,
                )
                return float(trial_record["objective_score"])
            except optuna.TrialPruned as exc:
                trial_record["status"] = "pruned"
                trial_record["error_message"] = str(exc)
                enrich_auto_explore_trial_record(
                    trial_record,
                    trial_dir=trial_dir,
                    auto_explore_cfg=auto_explore_cfg,
                )
                print(f"[自动探索] Trial {trial_index:03d} 被剪枝：{exc}")
                raise
            except Exception as exc:
                trial_record["status"] = "failed"
                trial_record["error_message"] = str(exc)
                enrich_auto_explore_trial_record(
                    trial_record,
                    trial_dir=trial_dir,
                    auto_explore_cfg=auto_explore_cfg,
                )
                print(f"[自动探索] Trial {trial_index:03d} 失败：{exc}")
                raise AutoExploreTrialFailed(str(exc))
            finally:
                trial_records.append(trial_record)
                persist_trial_state()

        def on_trial_complete(study: Any, frozen_trial: Any) -> None:
            del frozen_trial
            if progress_bar is None:
                return
            progress_bar.update(1)
            try:
                progress_bar.set_postfix(best=f"{float(study.best_value):.4f}")
            except Exception:
                return

        study = optuna.create_study(
            direction=optuna_direction_from_mode(auto_explore_cfg["objective"]["mode"]),
            sampler=build_optuna_sampler(auto_explore_cfg["optuna"], seed=seed),
            pruner=build_optuna_pruner(auto_explore_cfg["optuna"]),
        )
        try:
            study.optimize(
                objective,
                n_trials=int(auto_explore_cfg["num_trials"]),
                callbacks=[on_trial_complete],
                catch=(AutoExploreTrialFailed,),
                show_progress_bar=False,
            )
        finally:
            if progress_bar is not None:
                progress_bar.close()

    print("\n自动探索完成。")
    print(f"自动探索备注：{session_dir / 'remark.txt'}")
    print(f"自动探索摘要：{session_dir / 'notes.json'}")


def main() -> None:
    args = parse_args()

    preliminary_task_name = resolve_train_task_name(
        args.task if str(args.task).strip() else DEFAULT_CLI_TASK_NAME,
        None,
    )

    path_config_path = (
        Path(args.config)
        if str(args.config).strip()
        else Path(resolve_default_config_path(preliminary_task_name, "path.yaml"))
    )
    train_config_path = (
        Path(args.train_config)
        if str(args.train_config).strip()
        else Path(resolve_default_config_path(preliminary_task_name, "train.yaml"))
    )

    path_cfg = load_path_config(path_config_path)
    train_cfg = load_train_config(train_config_path)
    selected_task_name = resolve_train_task_name(args.task if str(args.task).strip() else None, train_cfg)
    train_cfg["task_name"] = selected_task_name

    if not str(args.config).strip() and selected_task_name != preliminary_task_name:
        path_config_path = Path(resolve_default_config_path(selected_task_name, "path.yaml"))
        path_cfg = load_path_config(path_config_path)

    model_config_path = (
        Path(args.model_config)
        if str(args.model_config).strip()
        else Path(resolve_default_config_path(selected_task_name, "model.yaml"))
    )
    model_cfg = load_model_config(model_config_path)
    allowed_run_keys = set(train_cfg["default_run"].keys()) - {"monitor_metric", "monitor_mode"}

    if bool(train_cfg["auto_explore"]) and (
        bool(train_cfg["auto_baselines"])
        or bool(train_cfg["auto_sotas"])
        or bool(train_cfg["auto_ablations"])
        or bool(train_cfg["auto_distinct"])
        or bool(train_cfg["auto_5fold"])
        or bool(train_cfg["auto_exp_1"])
        or bool(train_cfg["auto_exp_2"])
        or bool(train_cfg["auto_exp_3"])
        or bool(train_cfg["auto_exp_4"])
        or bool(train_cfg["auto_exp_5"])
        or bool(train_cfg["auto_exp_6"])
        or bool(train_cfg["auto_exp_7"])
        or bool(train_cfg["auto_exp_8"])
        or bool(train_cfg["auto_exp_8_mm_ablation"])
        or bool(train_cfg["auto_exp_9_ablation"])
        or bool(train_cfg["auto_exp_11_module_ablation"])
    ):
        raise ValueError("auto_explore 与其他自动批量实验模式不能同时开启，请二选一")

    auto_mode_count = int(bool(train_cfg["auto_baselines"]))
    auto_mode_count += int(bool(train_cfg["auto_sotas"]))
    auto_mode_count += int(bool(train_cfg["auto_exp_1"]))
    auto_mode_count += int(bool(train_cfg["auto_exp_2"]))
    auto_mode_count += int(bool(train_cfg["auto_exp_3"]))
    auto_mode_count += int(bool(train_cfg["auto_exp_4"]))
    auto_mode_count += int(bool(train_cfg["auto_exp_5"]))
    auto_mode_count += int(bool(train_cfg["auto_exp_6"]))
    auto_mode_count += int(bool(train_cfg["auto_exp_7"]))
    auto_mode_count += int(bool(train_cfg["auto_exp_8"]))
    auto_mode_count += int(bool(train_cfg["auto_exp_8_mm_ablation"]))
    auto_mode_count += int(bool(train_cfg["auto_exp_9_ablation"]))
    auto_mode_count += int(bool(train_cfg["auto_exp_11_module_ablation"]))
    auto_mode_count += int(bool(train_cfg["auto_ablations"]))
    auto_mode_count += int(bool(train_cfg["auto_distinct"]))
    auto_mode_count += int(bool(train_cfg["auto_5fold"]))
    if auto_mode_count > 1:
        raise ValueError("自动批量实验模式不能同时开启多个，请只保留一个自动模式")

    seed = args.seed if args.seed is not None else train_cfg["seed"]
    max_epochs = args.epochs if args.epochs is not None else train_cfg["max_epochs"]
    patience = args.patience if args.patience is not None else train_cfg["patience"]
    image_size = args.image_size if args.image_size is not None else train_cfg["image_size"]
    num_workers = args.num_workers if args.num_workers is not None else train_cfg["num_workers"]
    max_exams_per_task = (
        args.max_exams_per_task if args.max_exams_per_task is not None else train_cfg["max_exams_per_task"]
    )

    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = ",".join(str(item) for item in train_cfg["gpu_ids"])

    torch.set_float32_matmul_precision("medium")
    requested_gpu_count = len(train_cfg["gpu_ids"])
    cuda_available = torch.cuda.is_available()
    visible_gpu_count = torch.cuda.device_count() if cuda_available else 0
    if requested_gpu_count > 0 and not cuda_available:
        raise RuntimeError(
            "训练配置请求使用 GPU，但当前 PyTorch 无法初始化 CUDA。"
            "请检查 torch/CUDA/NumPy/驱动版本；不要让训练静默退回 CPU。"
        )
    if requested_gpu_count > 0 and visible_gpu_count < requested_gpu_count:
        raise RuntimeError(
            f"训练配置请求 {requested_gpu_count} 张 GPU: {train_cfg['gpu_ids']}，"
            f"但 PyTorch 当前仅可见 {visible_gpu_count} 张。"
        )

    if cuda_available:
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.benchmark = True

    use_multi_gpu = (not args.disable_multi_gpu) and visible_gpu_count > 1
    active_gpu_count = visible_gpu_count if use_multi_gpu else (1 if visible_gpu_count > 0 else 0)
    print(
        "CUDA状态 "
        f"available={cuda_available} visible_gpu_count={visible_gpu_count} "
        f"gpu_ids={train_cfg['gpu_ids']} use_multi_gpu={use_multi_gpu} "
        f"active_gpu_count={active_gpu_count}"
    )
    if cuda_available:
        for gpu_index in range(visible_gpu_count):
            print(f"GPU {gpu_index}: {torch.cuda.get_device_name(gpu_index)}")
    pretrained = not args.no_pretrained

    if bool(train_cfg["auto_distinct"]):
        if selected_task_name != "task1":
            raise ValueError("auto_distinct 当前仅用于 TASK1 的表2/表3/表4 CI 与显著性实验")
        auto_distinct_config_path = (
            Path(args.auto_distinct_config)
            if str(args.auto_distinct_config).strip()
            else Path(resolve_default_config_path(selected_task_name, "auto_distinct.yaml"))
        )
        distinct_script_path = Path(__file__).resolve().parent / "scripts" / "task1_distinct_significance.py"
        print("[TASK1 distinct] 启动表2/表3/表4 CI 与显著性实验")
        print(f"[TASK1 distinct] 配置文件: {auto_distinct_config_path}")
        subprocess.run(
            [
                sys.executable,
                str(distinct_script_path),
                "--config",
                str(auto_distinct_config_path),
            ],
            check=True,
        )
        return

    if bool(train_cfg["auto_5fold"]):
        if selected_task_name != "task1":
            raise ValueError("auto_5fold 当前仅用于 TASK1 的表2/表3/表4 5-fold 实验")
        auto_5fold_config_path = (
            Path(args.auto_5fold_config)
            if str(args.auto_5fold_config).strip()
            else Path(resolve_default_config_path(selected_task_name, "auto_5fold.yaml"))
        )
        fold_script_path = Path(__file__).resolve().parent / "scripts" / "task1_table_5fold.py"
        command = [
            sys.executable,
            str(fold_script_path),
            "--config",
            str(auto_5fold_config_path),
            "--path-config",
            str(path_config_path),
            "--train-config",
            str(train_config_path),
            "--model-config",
            str(model_config_path),
        ]
        if args.seed is not None:
            command.extend(["--seed", str(args.seed)])
        if args.epochs is not None:
            command.extend(["--epochs", str(args.epochs)])
        if args.patience is not None:
            command.extend(["--patience", str(args.patience)])
        if args.image_size is not None:
            command.extend(["--image-size", str(args.image_size)])
        if args.num_workers is not None:
            command.extend(["--num-workers", str(args.num_workers)])
        if args.max_exams_per_task is not None:
            command.extend(["--max-exams-per-task", str(args.max_exams_per_task)])
        if args.no_pretrained:
            command.append("--no-pretrained")
        if args.disable_multi_gpu:
            command.append("--disable-multi-gpu")
        print("[TASK1 5-fold] 启动表2/表3/表4 5-fold 实验")
        print(f"[TASK1 5-fold] 配置文件: {auto_5fold_config_path}")
        subprocess.run(command, check=True)
        return

    auto_baselines_cfg: dict[str, Any] | None = None
    auto_baseline_entries: list[dict[str, Any]] = []
    auto_sotas_cfg: dict[str, Any] | None = None
    auto_sota_entries: list[dict[str, Any]] = []
    auto_ablations_cfg: dict[str, Any] | None = None
    auto_ablation_experiments: list[dict[str, Any]] = []
    auto_exp1_cfg: dict[str, Any] | None = None
    auto_exp1_entries: list[dict[str, Any]] = []
    auto_exp2_cfg: dict[str, Any] | None = None
    auto_exp2_entries: list[dict[str, Any]] = []
    auto_exp3_cfg: dict[str, Any] | None = None
    auto_exp3_entries: list[dict[str, Any]] = []
    auto_exp4_cfg: dict[str, Any] | None = None
    auto_exp4_entries: list[dict[str, Any]] = []
    auto_exp5_cfg: dict[str, Any] | None = None
    auto_exp5_entries: list[dict[str, Any]] = []
    auto_exp6_cfg: dict[str, Any] | None = None
    auto_exp6_entries: list[dict[str, Any]] = []
    auto_exp7_cfg: dict[str, Any] | None = None
    auto_exp7_entries: list[dict[str, Any]] = []
    auto_exp8_cfg: dict[str, Any] | None = None
    auto_exp8_entries: list[dict[str, Any]] = []
    auto_exp8_mm_ablation_cfg: dict[str, Any] | None = None
    auto_exp8_mm_ablation_entries: list[dict[str, Any]] = []
    auto_exp9_ablation_cfg: dict[str, Any] | None = None
    auto_exp9_ablation_entries: list[dict[str, Any]] = []
    auto_exp11_module_ablation_cfg: dict[str, Any] | None = None
    auto_exp11_module_ablation_entries: list[dict[str, Any]] = []
    requested_names = [item.strip() for item in args.models.split(",") if item.strip()]

    if bool(train_cfg["auto_baselines"]):
        auto_baselines_cfg = load_auto_baselines_config(
            Path(args.auto_baselines_config)
            if str(args.auto_baselines_config).strip()
            else Path(resolve_default_config_path(selected_task_name, "auto_baselines.yaml")),
            allowed_run_keys=allowed_run_keys,
            selected_task_name=selected_task_name,
        )
        baseline_requested = [
            name for name in requested_names
            if name in {item["name"] for item in auto_baselines_cfg["models"]}
        ]
        auto_baseline_entries = selected_auto_baseline_model_entries(
            ",".join(baseline_requested),
            auto_baselines_cfg,
        ) if baseline_requested or not requested_names else []

    if bool(train_cfg["auto_sotas"]):
        auto_sotas_cfg = load_auto_sotas_config(
            Path(args.auto_sotas_config)
            if str(args.auto_sotas_config).strip()
            else Path(resolve_default_config_path(selected_task_name, "auto_sotas.yaml")),
            allowed_run_keys=allowed_run_keys,
            selected_task_name=selected_task_name,
        )
        sota_requested = [
            name for name in requested_names
            if name in {item["name"] for item in auto_sotas_cfg["models"]}
        ]
        auto_sota_entries = selected_auto_sota_model_entries(
            ",".join(sota_requested),
            auto_sotas_cfg,
        ) if sota_requested or not requested_names else []

    if bool(train_cfg["auto_ablations"]):
        auto_ablations_cfg = load_auto_ablations_config(
            Path(args.auto_ablations_config)
            if str(args.auto_ablations_config).strip()
            else Path(resolve_default_config_path(selected_task_name, "auto_ablations.yaml")),
            allowed_run_keys=allowed_run_keys,
            selected_task_name=selected_task_name,
        )
        auto_ablation_experiments = selected_auto_ablation_experiments(
            train_cfg["auto_ablations"],
            auto_ablations_cfg,
        )

    if bool(train_cfg["auto_exp_1"]):
        auto_exp1_entries = build_auto_exp1_entries(
            selected_task_name=selected_task_name,
            requested_names=requested_names if requested_names else None,
        )
        auto_exp1_cfg = build_auto_exp1_config(
            train_cfg=train_cfg,
            selected_task_name=selected_task_name,
            model_entries=auto_exp1_entries,
        )

    if bool(train_cfg["auto_exp_2"]):
        auto_exp2_entries = build_auto_exp2_entries(
            selected_task_name=selected_task_name,
            requested_names=requested_names if requested_names else None,
            skip_names=train_cfg.get("auto_exp_2_skip_models", []),
        )
        auto_exp2_cfg = build_auto_exp2_config(
            train_cfg=train_cfg,
            selected_task_name=selected_task_name,
            model_entries=auto_exp2_entries,
        )

    if bool(train_cfg["auto_exp_3"]):
        auto_exp3_entries = build_auto_exp3_entries(
            selected_task_name=selected_task_name,
            requested_names=requested_names if requested_names else None,
        )
        auto_exp3_cfg = build_auto_exp3_config(
            train_cfg=train_cfg,
            selected_task_name=selected_task_name,
            model_entries=auto_exp3_entries,
        )

    if bool(train_cfg["auto_exp_4"]):
        auto_exp4_entries = build_auto_exp4_entries(
            selected_task_name=selected_task_name,
            requested_names=requested_names if requested_names else None,
        )
        auto_exp4_cfg = build_auto_exp4_config(
            train_cfg=train_cfg,
            selected_task_name=selected_task_name,
            model_entries=auto_exp4_entries,
        )

    if bool(train_cfg["auto_exp_5"]):
        auto_exp5_entries = build_auto_exp5_entries(
            selected_task_name=selected_task_name,
            requested_names=requested_names if requested_names else None,
        )
        auto_exp5_cfg = build_auto_exp5_config(
            train_cfg=train_cfg,
            selected_task_name=selected_task_name,
            model_entries=auto_exp5_entries,
        )

    if bool(train_cfg["auto_exp_6"]):
        auto_exp6_entries = build_auto_exp6_entries(
            selected_task_name=selected_task_name,
            requested_names=requested_names if requested_names else None,
        )
        auto_exp6_cfg = build_auto_exp6_config(
            train_cfg=train_cfg,
            selected_task_name=selected_task_name,
            model_entries=auto_exp6_entries,
        )

    if bool(train_cfg["auto_exp_7"]):
        auto_exp7_entries = build_auto_exp7_entries(
            selected_task_name=selected_task_name,
            requested_names=requested_names if requested_names else None,
        )
        auto_exp7_cfg = build_auto_exp7_config(
            train_cfg=train_cfg,
            selected_task_name=selected_task_name,
            model_entries=auto_exp7_entries,
        )

    if bool(train_cfg["auto_exp_8"]):
        auto_exp8_entries = build_auto_exp8_entries(
            selected_task_name=selected_task_name,
            requested_names=requested_names if requested_names else None,
            base_seed=seed,
        )
        auto_exp8_cfg = build_auto_exp8_config(
            train_cfg=train_cfg,
            selected_task_name=selected_task_name,
            model_entries=auto_exp8_entries,
        )

    if bool(train_cfg["auto_exp_8_mm_ablation"]):
        auto_exp8_mm_ablation_entries = build_auto_exp8_mm_ablation_entries(
            selected_task_name=selected_task_name,
            requested_names=requested_names if requested_names else None,
            base_seed=seed,
        )
        auto_exp8_mm_ablation_cfg = build_auto_exp8_mm_ablation_config(
            train_cfg=train_cfg,
            selected_task_name=selected_task_name,
            model_entries=auto_exp8_mm_ablation_entries,
        )

    if bool(train_cfg["auto_exp_9_ablation"]):
        auto_exp9_ablation_entries = build_auto_exp9_ablation_entries(
            selected_task_name=selected_task_name,
            requested_names=requested_names if requested_names else None,
            base_seed=seed,
        )
        auto_exp9_ablation_cfg = build_auto_exp9_ablation_config(
            train_cfg=train_cfg,
            selected_task_name=selected_task_name,
            model_entries=auto_exp9_ablation_entries,
        )

    if bool(train_cfg["auto_exp_11_module_ablation"]):
        auto_exp11_module_ablation_entries = build_auto_exp11_module_ablation_entries(
            selected_task_name=selected_task_name,
            requested_names=requested_names if requested_names else None,
            base_seed=seed,
        )
        auto_exp11_module_ablation_cfg = build_auto_exp11_module_ablation_config(
            train_cfg=train_cfg,
            selected_task_name=selected_task_name,
            model_entries=auto_exp11_module_ablation_entries,
        )

    if auto_ablations_cfg is not None and requested_names:
        raise ValueError("auto_ablations 模式当前不支持 --models，请通过 train.yaml 的 auto_ablations 选择实验")

    if (
        auto_baselines_cfg is not None
        or auto_sotas_cfg is not None
        or auto_ablations_cfg is not None
        or auto_exp1_cfg is not None
        or auto_exp2_cfg is not None
        or auto_exp3_cfg is not None
        or auto_exp4_cfg is not None
        or auto_exp5_cfg is not None
        or auto_exp6_cfg is not None
        or auto_exp7_cfg is not None
        or auto_exp8_cfg is not None
        or auto_exp8_mm_ablation_cfg is not None
        or auto_exp9_ablation_cfg is not None
        or auto_exp11_module_ablation_cfg is not None
    ):
        selected_names = [item["name"] for item in auto_baseline_entries] + [item["name"] for item in auto_sota_entries]
        selected_names += [item["name"] for item in auto_exp1_entries]
        selected_names += [item["name"] for item in auto_exp2_entries]
        selected_names += [item["name"] for item in auto_exp3_entries]
        selected_names += [item["name"] for item in auto_exp4_entries]
        selected_names += [item["name"] for item in auto_exp5_entries]
        selected_names += [item["name"] for item in auto_exp6_entries]
        selected_names += [item["name"] for item in auto_exp7_entries]
        selected_names += [item["name"] for item in auto_exp8_entries]
        selected_names += [item["name"] for item in auto_exp8_mm_ablation_entries]
        selected_names += [item["name"] for item in auto_exp9_ablation_entries]
        selected_names += [item["name"] for item in auto_exp11_module_ablation_entries]
        selected_task_model_names = (
            [resolve_series_entry_model_name(item) for item in auto_baseline_entries]
            + [resolve_series_entry_model_name(item) for item in auto_sota_entries]
            + [resolve_series_entry_model_name(item) for item in auto_exp1_entries]
            + [resolve_series_entry_model_name(item) for item in auto_exp2_entries]
            + [resolve_series_entry_model_name(item) for item in auto_exp3_entries]
            + [resolve_series_entry_model_name(item) for item in auto_exp4_entries]
            + [resolve_series_entry_model_name(item) for item in auto_exp5_entries]
            + [resolve_series_entry_model_name(item) for item in auto_exp6_entries]
            + [resolve_series_entry_model_name(item) for item in auto_exp7_entries]
            + [resolve_series_entry_model_name(item) for item in auto_exp8_entries]
            + [resolve_series_entry_model_name(item) for item in auto_exp8_mm_ablation_entries]
            + [resolve_series_entry_model_name(item) for item in auto_exp9_ablation_entries]
            + [resolve_series_entry_model_name(item) for item in auto_exp11_module_ablation_entries]
            + [
                resolve_series_entry_model_name(entry)
                for experiment in auto_ablation_experiments
                for entry in experiment["models"]
                if entry["enabled"]
            ]
        )
        selected_names.extend(
            [
                f"{experiment['name']}:{entry['name']}"
                for experiment in auto_ablation_experiments
                for entry in experiment["models"]
                if entry["enabled"]
            ]
        )
        if requested_names and auto_ablations_cfg is None:
            unresolved = [name for name in requested_names if name not in set(selected_names)]
            if unresolved:
                raise ValueError(f"当前自动批量配置中不存在这些已启用模型：{unresolved}")
        if not selected_names:
            raise ValueError("自动批量模式下没有可运行的模型，请检查配置文件中的 enabled 字段")
    else:
        selected_names = selected_model_names(args.models, train_cfg)
        selected_task_model_names = list(selected_names)

    required_tasks = resolve_required_tasks(selected_task_model_names, selected_task_name)
    print("=" * 72)
    print("训练配置")
    print(f"模型={','.join(selected_names)}")
    print(f"任务={selected_task_name}")
    print(f"epoch={max_epochs} patience={patience} seed={seed}")
    for line in build_training_config_summary_lines(
        train_cfg["default_run"],
        image_size=image_size,
        num_workers=num_workers,
    ):
        print(line)
    training_context = prepare_training_context(
        path_cfg=path_cfg,
        train_cfg=train_cfg,
        seed=seed,
        max_exams_per_task=max_exams_per_task,
        required_tasks=required_tasks,
    )
    for task_name in sorted(required_tasks):
        task_stats = training_context.get("task_stats", {}).get(task_name, {})
        print(
            f"{format_task_display_name(task_name)}样本 "
            f"总样本={task_stats.get('total_records', 0)} "
            f"train/val/test={task_stats.get('train_size', 0)}/{task_stats.get('val_size', 0)}/{task_stats.get('test_size', 0)}"
        )
        added_records = int(task_stats.get("class_balance_added_records", 0) or 0)
        if added_records > 0:
            print(
                f"{format_task_display_name(task_name)}类别平衡 "
                f"训练集原始={task_stats.get('train_original_size', 0)} "
                f"新增虚拟bag={added_records} "
                f"平衡后训练集={task_stats.get('train_size', 0)}"
            )

    if auto_baselines_cfg is not None:
        run_auto_baselines(
            train_cfg=train_cfg,
            auto_baselines_cfg=auto_baselines_cfg,
            model_cfg=model_cfg,
            training_context=training_context,
            model_entries=auto_baseline_entries,
            seed=seed,
            max_epochs=max_epochs,
            patience=patience,
            image_size=image_size,
            num_workers=num_workers,
            pretrained=pretrained,
            use_multi_gpu=use_multi_gpu,
            active_gpu_count=active_gpu_count,
        )
    if auto_sotas_cfg is not None:
        run_auto_sotas(
            train_cfg=train_cfg,
            auto_sotas_cfg=auto_sotas_cfg,
            model_cfg=model_cfg,
            training_context=training_context,
            model_entries=auto_sota_entries,
            seed=seed + 10000,
            max_epochs=max_epochs,
            patience=patience,
            image_size=image_size,
            num_workers=num_workers,
            pretrained=pretrained,
            use_multi_gpu=use_multi_gpu,
            active_gpu_count=active_gpu_count,
        )
    if auto_ablations_cfg is not None:
        run_auto_ablations(
            train_cfg=train_cfg,
            auto_ablations_experiments=auto_ablation_experiments,
            model_cfg=model_cfg,
            training_context=training_context,
            seed=seed + 20000,
            max_epochs=max_epochs,
            patience=patience,
            image_size=image_size,
            num_workers=num_workers,
            pretrained=pretrained,
            use_multi_gpu=use_multi_gpu,
            active_gpu_count=active_gpu_count,
        )
    if auto_exp1_cfg is not None:
        run_auto_exp1(
            train_cfg=train_cfg,
            auto_exp1_cfg=auto_exp1_cfg,
            model_cfg=model_cfg,
            training_context=training_context,
            model_entries=auto_exp1_entries,
            seed=seed + 30000,
            max_epochs=max_epochs,
            patience=patience,
            image_size=image_size,
            num_workers=num_workers,
            pretrained=pretrained,
            use_multi_gpu=use_multi_gpu,
            active_gpu_count=active_gpu_count,
        )
    if auto_exp2_cfg is not None:
        run_auto_exp2(
            train_cfg=train_cfg,
            auto_exp2_cfg=auto_exp2_cfg,
            model_cfg=model_cfg,
            training_context=training_context,
            model_entries=auto_exp2_entries,
            seed=seed + 40000,
            max_epochs=max_epochs,
            patience=patience,
            image_size=image_size,
            num_workers=num_workers,
            pretrained=pretrained,
            use_multi_gpu=use_multi_gpu,
            active_gpu_count=active_gpu_count,
        )
    if auto_exp3_cfg is not None:
        run_auto_exp3(
            train_cfg=train_cfg,
            auto_exp3_cfg=auto_exp3_cfg,
            model_cfg=model_cfg,
            training_context=training_context,
            model_entries=auto_exp3_entries,
            seed=seed + 50000,
            max_epochs=max_epochs,
            patience=patience,
            image_size=image_size,
            num_workers=num_workers,
            pretrained=pretrained,
            use_multi_gpu=use_multi_gpu,
            active_gpu_count=active_gpu_count,
        )
    if auto_exp4_cfg is not None:
        run_auto_exp4(
            train_cfg=train_cfg,
            auto_exp4_cfg=auto_exp4_cfg,
            model_cfg=model_cfg,
            training_context=training_context,
            model_entries=auto_exp4_entries,
            seed=seed + 60000,
            max_epochs=max_epochs,
            patience=patience,
            image_size=image_size,
            num_workers=num_workers,
            pretrained=pretrained,
            use_multi_gpu=use_multi_gpu,
            active_gpu_count=active_gpu_count,
        )
    if auto_exp5_cfg is not None:
        run_auto_exp5(
            train_cfg=train_cfg,
            auto_exp5_cfg=auto_exp5_cfg,
            model_cfg=model_cfg,
            training_context=training_context,
            model_entries=auto_exp5_entries,
            seed=seed + 70000,
            max_epochs=max_epochs,
            patience=patience,
            image_size=image_size,
            num_workers=num_workers,
            pretrained=pretrained,
            use_multi_gpu=use_multi_gpu,
            active_gpu_count=active_gpu_count,
        )
    if auto_exp6_cfg is not None:
        run_auto_exp6(
            train_cfg=train_cfg,
            auto_exp6_cfg=auto_exp6_cfg,
            model_cfg=model_cfg,
            training_context=training_context,
            model_entries=auto_exp6_entries,
            seed=seed + 80000,
            max_epochs=max_epochs,
            patience=patience,
            image_size=image_size,
            num_workers=num_workers,
            pretrained=pretrained,
            use_multi_gpu=use_multi_gpu,
            active_gpu_count=active_gpu_count,
        )
    if auto_exp7_cfg is not None:
        run_auto_exp7(
            train_cfg=train_cfg,
            auto_exp7_cfg=auto_exp7_cfg,
            model_cfg=model_cfg,
            training_context=training_context,
            model_entries=auto_exp7_entries,
            seed=seed + 90000,
            max_epochs=max_epochs,
            patience=patience,
            image_size=image_size,
            num_workers=num_workers,
            pretrained=pretrained,
            use_multi_gpu=use_multi_gpu,
            active_gpu_count=active_gpu_count,
        )
    if auto_exp8_cfg is not None:
        run_auto_exp8(
            train_cfg=train_cfg,
            auto_exp8_cfg=auto_exp8_cfg,
            model_cfg=model_cfg,
            training_context=training_context,
            model_entries=auto_exp8_entries,
            seed=seed,
            max_epochs=max_epochs,
            patience=patience,
            image_size=image_size,
            num_workers=num_workers,
            pretrained=pretrained,
            use_multi_gpu=use_multi_gpu,
            active_gpu_count=active_gpu_count,
        )
    if auto_exp8_mm_ablation_cfg is not None:
        run_auto_exp8_mm_ablation(
            train_cfg=train_cfg,
            auto_exp8_mm_ablation_cfg=auto_exp8_mm_ablation_cfg,
            model_cfg=model_cfg,
            training_context=training_context,
            model_entries=auto_exp8_mm_ablation_entries,
            seed=seed,
            max_epochs=max_epochs,
            patience=patience,
            image_size=image_size,
            num_workers=num_workers,
            pretrained=pretrained,
            use_multi_gpu=use_multi_gpu,
            active_gpu_count=active_gpu_count,
        )
    if auto_exp9_ablation_cfg is not None:
        run_auto_exp9_ablation(
            train_cfg=train_cfg,
            auto_exp9_ablation_cfg=auto_exp9_ablation_cfg,
            model_cfg=model_cfg,
            training_context=training_context,
            model_entries=auto_exp9_ablation_entries,
            seed=seed,
            max_epochs=max_epochs,
            patience=patience,
            image_size=image_size,
            num_workers=num_workers,
            pretrained=pretrained,
            use_multi_gpu=use_multi_gpu,
            active_gpu_count=active_gpu_count,
        )
    if auto_exp11_module_ablation_cfg is not None:
        run_auto_exp11_module_ablation(
            train_cfg=train_cfg,
            auto_exp11_module_ablation_cfg=auto_exp11_module_ablation_cfg,
            model_cfg=model_cfg,
            training_context=training_context,
            model_entries=auto_exp11_module_ablation_entries,
            seed=seed,
            max_epochs=max_epochs,
            patience=patience,
            image_size=image_size,
            num_workers=num_workers,
            pretrained=pretrained,
            use_multi_gpu=use_multi_gpu,
            active_gpu_count=active_gpu_count,
        )
    if (
        auto_baselines_cfg is not None
        or auto_sotas_cfg is not None
        or auto_ablations_cfg is not None
        or auto_exp1_cfg is not None
        or auto_exp2_cfg is not None
        or auto_exp3_cfg is not None
        or auto_exp4_cfg is not None
        or auto_exp5_cfg is not None
        or auto_exp6_cfg is not None
        or auto_exp7_cfg is not None
        or auto_exp8_cfg is not None
        or auto_exp8_mm_ablation_cfg is not None
        or auto_exp9_ablation_cfg is not None
        or auto_exp11_module_ablation_cfg is not None
    ):
        return

    auto_explore_cfg: dict[str, Any] | None = None
    if bool(train_cfg["auto_explore"]):
        auto_explore_cfg = load_auto_explore_config(
            Path(args.auto_explore_config)
            if str(args.auto_explore_config).strip()
            else Path(resolve_default_config_path(selected_task_name, "auto_explore.yaml")),
            allowed_run_keys=allowed_run_keys,
        )
        for model_offset, model_name in enumerate(selected_names, start=1):
            run_auto_explore(
                train_cfg=train_cfg,
                auto_explore_cfg=auto_explore_cfg,
                model_cfg=model_cfg,
                training_context=training_context,
                model_name=model_name,
                seed=seed + model_offset * 1000,
                base_max_epochs=max_epochs,
                base_patience=patience,
                image_size=image_size,
                num_workers=num_workers,
                pretrained=pretrained,
                use_multi_gpu=use_multi_gpu,
                active_gpu_count=active_gpu_count,
            )
        return

    run_dirs: list[Path] = []
    for model_offset, model_name in enumerate(selected_names, start=1):
        run_dir, run_meta = allocate_task_run_dir(
            training_context["output_root"],
            train_cfg,
            model_name,
            is_auto_explore=False,
        )
        resume_path = run_meta.get("resume_path")
        if resume_path:
            print(f"训练目录={run_dir}")
            print(f"检测到未完成训练，使用 last.ckpt 继续：{resume_path}")
        else:
            print(f"训练目录={run_dir}")
        run_model_job(
            model_name=model_name,
            run_dir=run_dir,
            train_cfg=train_cfg,
            model_cfg=model_cfg,
            training_context=training_context,
            seed=seed + model_offset,
            max_epochs=max_epochs,
            patience=patience,
            image_size=image_size,
            num_workers=num_workers,
            pretrained=pretrained,
            use_multi_gpu=use_multi_gpu,
            active_gpu_count=active_gpu_count,
            run_test=True,
            run_overrides=None,
            resume_path=resume_path,
        )
        run_dirs.append(run_dir)

    print("\n训练完成。")
    for run_dir in run_dirs:
        print(f"训练目录：{run_dir}")


if __name__ == "__main__":
    main()
