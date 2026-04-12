#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import math
import os
import random
import re
import shutil
import sys
from dataclasses import asdict
from datetime import datetime
from pathlib import Path
from typing import Any, Callable

import numpy as np
import torch
import yaml
from torch.utils.data import DataLoader

from ablation_exp import ABLATION_EXPERIMENT_NAMES, build_all_ablation_experiments
from baselines.colon_baseline import ColonoscopyMILBaseline
from baselines.gastro_baseline import (
    GASTRO_BASELINE_CLASS_REGISTRY,
    GASTRO_BASELINE_MODEL_NAMES,
    build_gastro_baseline,
)
from model import GastroLabelGraphMIL
from sotas.gastro_sota import (
    GASTRO_SOTA_CLASS_REGISTRY,
    GASTRO_SOTA_MODEL_NAMES,
    build_gastro_sota,
)
from training import (
    COLO_BINARY_CLASS_NAMES,
    GASTRO_LABEL_NAMES,
    InstanceAwareBatchSampler,
    MILBagDataset,
    Trainer,
    TrainerConfig,
    build_task_records,
    mil_collate_fn,
    split_records,
    to_builtin_type,
)


sys.dont_write_bytecode = True
os.environ.setdefault("PYTHONDONTWRITEBYTECODE", "1")

MODEL_SEQUENCE = (
    "gastro_baseline",
    "gastro_label_graph_mil",
    "colonoscopy_baseline",
)
SUPPORTED_MODEL_NAMES = tuple(dict.fromkeys(MODEL_SEQUENCE + GASTRO_BASELINE_MODEL_NAMES))
SUPPORTED_MODEL_NAMES = tuple(dict.fromkeys(SUPPORTED_MODEL_NAMES + GASTRO_SOTA_MODEL_NAMES))

TRACKER_ALIAS_TO_META = {
    "best_macro_f1": {"metric_name": "macro_f1", "mode": "max"},
    "best_micro_f1": {"metric_name": "micro_f1", "mode": "max"},
    "best_val_loss": {"metric_name": "val_loss", "mode": "min"},
}
SERIES_TRACKER_ALIASES = tuple(TRACKER_ALIAS_TO_META.keys())

GASTRO_TASK_META = {
    "task_name": "gastro_multilabel",
    "task_dir_name": "gastro_multilabel_task",
    "run_prefix": "gastro",
}
MODEL_TASK_META = {
    model_name: dict(GASTRO_TASK_META)
    for model_name in (
        "gastro_baseline",
        "gastro_label_graph_mil",
        *GASTRO_BASELINE_MODEL_NAMES,
        *GASTRO_SOTA_MODEL_NAMES,
    )
}
MODEL_TASK_META["colonoscopy_baseline"] = {
    "task_name": "colonoscopy_binary",
    "task_dir_name": "colonoscopy_binary_task",
    "run_prefix": "colonoscopy",
}

AUTO_BASELINE_ALLOWED_MODEL_NAMES = tuple(
    name
    for name in SUPPORTED_MODEL_NAMES
    if name in GASTRO_BASELINE_CLASS_REGISTRY or name == "colonoscopy_baseline"
)
AUTO_SOTA_ALLOWED_MODEL_NAMES = tuple(
    name
    for name in SUPPORTED_MODEL_NAMES
    if name in GASTRO_SOTA_CLASS_REGISTRY
)
AUTO_ABLATION_ALLOWED_MODEL_NAMES = ("gastro_label_graph_mil",)

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


def resolve_model_task_meta(model_name: str) -> dict[str, str]:
    if model_name not in MODEL_TASK_META:
        raise ValueError(f"未知模型名: {model_name}")
    return dict(MODEL_TASK_META[model_name])


def resolve_required_tasks(model_names: list[str]) -> set[str]:
    return {resolve_model_task_meta(model_name)["task_name"] for model_name in model_names}


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
    if task_name == "gastro_multilabel":
        return "胃镜"
    if task_name == "colonoscopy_binary":
        return "肠镜"
    return task_name


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


def allocate_task_run_dir(
    output_root: Path,
    train_cfg: dict[str, Any],
    model_name: str,
    *,
    is_auto_explore: bool,
) -> tuple[Path, dict[str, Any]]:
    task_meta = resolve_model_task_meta(model_name)
    task_dir = output_root / train_cfg["train_run_dir_name"] / task_meta["task_dir_name"]
    task_dir.mkdir(parents=True, exist_ok=True)

    run_index = next_run_index(task_dir, task_meta["run_prefix"])
    suffix = "_para_auto" if is_auto_explore else ""
    run_dir = task_dir / f"{task_meta['run_prefix']}_{run_index}{suffix}"
    run_dir.mkdir(parents=True, exist_ok=True)

    return run_dir, {
        **task_meta,
        "task_dir": str(task_dir),
        "run_index": run_index,
        "is_auto_explore": is_auto_explore,
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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="训练胃镜与肠镜保留模型")
    parser.add_argument("--config", type=str, default="configs/path.yaml", help="路径配置文件")
    parser.add_argument("--train-config", type=str, default="configs/train.yaml", help="训练配置文件")
    parser.add_argument("--model-config", type=str, default="configs/model.yaml", help="模型配置文件")
    parser.add_argument(
        "--auto-explore-config",
        type=str,
        default="configs/auto_explore.yaml",
        help="自动探索配置文件",
    )
    parser.add_argument(
        "--auto-baselines-config",
        type=str,
        default="configs/auto_baselines.yaml",
        help="自动 baseline 配置文件",
    )
    parser.add_argument(
        "--auto-sotas-config",
        type=str,
        default="configs/auto_sotas.yaml",
        help="自动 SOTA 配置文件",
    )
    parser.add_argument(
        "--auto-ablations-config",
        type=str,
        default="configs/auto_ablations.yaml",
        help="自动消融实验配置文件",
    )
    parser.add_argument("--models", type=str, default="", help="仅运行指定模型，使用逗号分隔")
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
    return resolved


def _load_ratio(payload: dict[str, Any], key: str) -> tuple[float, float, float]:
    raw = payload.get(key, [0.6, 0.2, 0.2])
    if not isinstance(raw, list) or len(raw) != 3:
        raise ValueError(f"{key} 必须是长度为 3 的列表")
    ratios = tuple(float(item) for item in raw)
    if abs(sum(ratios) - 1.0) > 1e-8:
        raise ValueError(f"{key} 的和必须为 1")
    return ratios


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
    if not isinstance(enabled_models_raw, list) or not enabled_models_raw:
        raise ValueError("enabled_models 必须是非空列表")

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

    return {
        "gpu_ids": [int(item) for item in gpu_ids_raw],
        "num_workers": int(payload.get("num_workers", 6)),
        "seed": int(payload.get("seed", 42)),
        "image_size": int(payload.get("image_size", 224)),
        "max_epochs": int(payload.get("max_epochs", 30)),
        "patience": int(payload.get("patience", 30)),
        "max_exams_per_task": int(payload.get("max_exams_per_task", 0)),
        "min_instances": int(payload.get("min_instances", 1)),
        "split_ratio": _load_ratio(payload, "split_ratio"),
        "train_sampling_strategy": str(payload.get("train_sampling_strategy", "random")),
        "eval_sampling_strategy": str(payload.get("eval_sampling_strategy", "uniform")),
        "task_selection_dir_name": str(payload.get("task_selection_dir_name", "task_data")),
        "train_run_dir_name": str(payload.get("train_run_dir_name", "train_runs")),
        "run_dir_prefix": str(payload.get("run_dir_prefix", "run")),
        "remark_metric_alias": str(payload.get("remark_metric_alias", "best_macro_f1")),
        "remark_metric_name": str(payload.get("remark_metric_name", "macro_f1")),
        "enabled_models": enabled_models,
        "default_run": default_run,
        "auto_explore": auto_explore_enabled,
        "auto_baselines": auto_baselines_enabled,
        "auto_sotas": auto_sotas_enabled,
        "auto_ablations": auto_ablations_selection,
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

        task_meta = resolve_model_task_meta(model_name)
        normalized.append(
            {
                "name": model_name,
                "display_name": str(item.get("display_name", model_name)).strip() or model_name,
                "enabled": bool(item.get("enabled", True)),
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


def load_auto_baselines_config(config_path: Path, allowed_run_keys: set[str]) -> dict[str, Any]:
    return load_auto_model_series_config(
        config_path,
        allowed_run_keys,
        config_prefix="auto_baselines",
        allowed_model_names=AUTO_BASELINE_ALLOWED_MODEL_NAMES,
        default_output_dir_name="gastro_baselines",
    )


def load_auto_sotas_config(config_path: Path, allowed_run_keys: set[str]) -> dict[str, Any]:
    return load_auto_model_series_config(
        config_path,
        allowed_run_keys,
        config_prefix="auto_sotas",
        allowed_model_names=AUTO_SOTA_ALLOWED_MODEL_NAMES,
        default_output_dir_name="gastro_sotas",
    )


def normalize_auto_ablation_entries(
    raw_models: Any,
    allowed_run_keys: set[str],
    *,
    config_prefix: str,
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

        task_meta = resolve_model_task_meta(base_model_name)
        normalized.append(
            {
                "name": entry_name,
                "base_model_name": base_model_name,
                "display_name": str(item.get("display_name", entry_name)).strip() or entry_name,
                "enabled": bool(item.get("enabled", True)),
                "model_params": model_params,
                "run_overrides": run_overrides,
                "task_name": task_meta["task_name"],
                "task_dir_name": task_meta["task_dir_name"],
            }
        )
        seen_names.add(entry_name)

    return normalized


def load_auto_ablations_config(config_path: Path, allowed_run_keys: set[str]) -> dict[str, Any]:
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

    output_root_dir_name = str(payload.get("output_root_dir_name", "gastro_ablations")).strip() or "gastro_ablations"
    common_goal = str(payload.get("goal", "")).strip()

    remark_raw = payload.get("remark", {})
    if remark_raw is None:
        remark_raw = {}
    if not isinstance(remark_raw, dict):
        raise ValueError("auto_ablations.remark 配置格式错误")

    registry = {item["name"]: item for item in build_all_ablation_experiments()}
    raw_experiments = payload.get("experiments")
    if raw_experiments is None:
        raw_experiments = [{"name": name, "enabled": True} for name in ABLATION_EXPERIMENT_NAMES]
    if not isinstance(raw_experiments, list) or not raw_experiments:
        raise ValueError("auto_ablations.experiments 必须是非空列表")

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
) -> tuple[dict[str, list[dict[str, Any]]], str]:
    if not records:
        return {"train": [], "val": [], "test": []}, "empty"

    regular_split = split_records(records, seed=seed, ratios=ratios)
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
    if task_name == "gastro_multilabel":
        return "cache_gastro_multilabel_image"
    if task_name == "colonoscopy_binary":
        return "colonoscopy_binary_image_cache"
    raise ValueError(f"未知 task_name: {task_name}")


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
    image_cache_warmup: bool,
    memory_cache_size: int,
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
        memory_cache_size=memory_cache_size,
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
        memory_cache_size=memory_cache_size,
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
        memory_cache_size=memory_cache_size,
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
    (run_dir / "config.yaml").write_text(
        yaml.safe_dump(to_builtin_type(config_payload), allow_unicode=True, sort_keys=False),
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


def resolve_run_cfg(train_cfg: dict[str, Any], model_name: str) -> dict[str, Any]:
    del model_name
    return dict(train_cfg["default_run"])


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
    model_name: str,
) -> tuple[dict[str, list[dict[str, Any]]], list[float]]:
    task_name = resolve_model_task_meta(model_name)["task_name"]
    if task_name == "gastro_multilabel":
        return training_context["gastro_split"], training_context["gastro_pos_weight"]
    if task_name == "colonoscopy_binary":
        return training_context["colon_split"], training_context["colon_pos_weight"]
    raise ValueError(f"未知任务名: {task_name}")


def build_model_bundle(
    model_name: str,
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
            num_labels=3,
            dropout=float(model_param_cfg.get("dropout", 0.2)),
            topk=int(model_param_cfg.get("topk", 4)),
            num_heads=int(model_param_cfg.get("num_heads", 8)),
            num_layers=int(model_param_cfg.get("num_layers", 2)),
        )
        trainer_cfg = TrainerConfig(
            task_type="gastro_multilabel",
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
        return model, trainer_cfg, "gastro_multilabel", GASTRO_LABEL_NAMES, []

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
            num_labels=3,
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
            task_type="gastro_multilabel",
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
        return model, trainer_cfg, "gastro_multilabel", GASTRO_LABEL_NAMES, []

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
            num_labels=3,
            dropout=float(model_param_cfg.get("dropout", 0.2)),
        )
        trainer_cfg = TrainerConfig(
            task_type="gastro_multilabel",
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
        return model, trainer_cfg, "gastro_multilabel", GASTRO_LABEL_NAMES, []

    if model_name == "colonoscopy_baseline":
        monitor_metric, monitor_mode = resolve_monitor_settings(
            run_cfg,
            default_metric="auc",
            default_mode="max",
        )
        model = ColonoscopyMILBaseline(
            backbone_name="resnet50",
            pretrained=pretrained,
            freeze_stages=1,
            feature_dim=512,
            attn_dim=256,
            dropout=0.2,
        )
        trainer_cfg = TrainerConfig(
            task_type="colonoscopy_binary",
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
            loss_name=str(run_cfg.get("loss_name", "focal")),
            pos_weight=pos_weight,
            aux_loss_weights={},
            use_multi_gpu=use_multi_gpu,
            resume_path=resume_path,
            run_test=run_test,
        )
        return model, trainer_cfg, "colonoscopy_binary", [], COLO_BINARY_CLASS_NAMES

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
) -> dict[str, Any]:
    train_batch_size = normalize_batch_size(int(run_cfg.get("batch_size", 3)), active_gpu_count)
    eval_batch_size = normalize_batch_size(int(run_cfg.get("eval_batch_size", 3)), active_gpu_count)
    raw_cache_dir = str(run_cfg.get("image_cache_dir", "")).strip()
    resolved_cache_root_dir: Path | None = None
    resolved_cache_dir: Path | None = None
    if raw_cache_dir:
        candidate_cache_dir = Path(raw_cache_dir).expanduser()
        resolved_cache_root_dir = (
            candidate_cache_dir.resolve()
            if candidate_cache_dir.is_absolute()
            else (cache_root_dir / candidate_cache_dir).resolve()
        )
        resolved_cache_dir = resolved_cache_root_dir / task_image_cache_dir_name(task_name)

    effective_run_cfg = dict(run_cfg)
    if resolved_cache_root_dir is not None:
        effective_run_cfg["resolved_image_cache_root_dir"] = str(resolved_cache_root_dir)
    if resolved_cache_dir is not None:
        effective_run_cfg["resolved_image_cache_dir"] = str(resolved_cache_dir)
        effective_run_cfg["image_cache_task_scope"] = task_name

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
        image_cache_warmup=bool(run_cfg.get("image_cache_warmup", False)),
        memory_cache_size=int(run_cfg.get("memory_cache_size", 0)),
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
    resume_path: str | None = None,
    on_validation_epoch_end: Callable[[int, float, dict[str, Any]], None] | None = None,
) -> dict[str, Any]:
    run_cfg = resolve_run_cfg(train_cfg, model_name)
    if run_overrides:
        run_cfg.update(run_overrides)
    model_param_cfg = dict(model_cfg["models"].get(model_name, {}))
    if model_param_override:
        model_param_cfg.update(model_param_override)

    split_data, pos_weight = resolve_task_training_payload(training_context, model_name)

    seed_everything(seed)
    model, trainer_cfg, task_name, label_names, class_names = build_model_bundle(
        model_name=model_name,
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

    return run_single_model(
        model_name=model_name,
        model=model,
        trainer_cfg=trainer_cfg,
        split_data=split_data,
        task_name=task_name,
        image_size=image_size,
        num_workers=num_workers,
        run_dir=run_dir,
        seed=seed,
        run_cfg=run_cfg,
        model_param_cfg=model_param_cfg,
        min_instances=train_cfg["min_instances"],
        train_sampling=train_cfg["train_sampling_strategy"],
        eval_sampling=train_cfg["eval_sampling_strategy"],
        active_gpu_count=active_gpu_count,
        label_names=label_names,
        class_names=class_names,
        cache_root_dir=Path(training_context["task_selection_dir"]).resolve(),
        on_validation_epoch_end=on_validation_epoch_end,
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
    gastro_csv = task_selection_dir / "gastro_multilabel_task_datalist.csv"
    colon_csv = task_selection_dir / "colonoscopy_binary_task_datalist.csv"

    dataset_root = path_cfg.get("dataset_root")
    context = {
        "output_root": output_root,
        "task_selection_dir": str(task_selection_dir),
        "task_stats": {},
    }

    if "gastro_multilabel" in required_tasks:
        if not gastro_csv.is_file():
            raise FileNotFoundError(
                "未找到胃镜任务筛选 CSV，请先运行 `python task_data_selection.py` 生成："
                f"\n- {gastro_csv}"
            )
        gastro_records = build_task_records(
            task_csv_path=gastro_csv,
            task_name="gastro_multilabel",
            min_instances=train_cfg["min_instances"],
            dataset_root=dataset_root,
        )
        gastro_records = maybe_limit_records(gastro_records, max_num=max_exams_per_task, seed=seed)
        gastro_split, _ = build_compatible_split(
            gastro_records,
            seed=seed,
            ratios=train_cfg["split_ratio"],
        )
        gastro_pos_weight = (
            compute_multilabel_pos_weight(gastro_split["train"])
            if gastro_split["train"]
            else [1.0 for _ in GASTRO_LABEL_NAMES]
        )
        context["gastro_split"] = gastro_split
        context["gastro_pos_weight"] = gastro_pos_weight
        context["task_stats"]["gastro_multilabel"] = {
            "total_records": len(gastro_records),
            "train_size": len(gastro_split["train"]),
            "val_size": len(gastro_split["val"]),
            "test_size": len(gastro_split["test"]),
        }

    if "colonoscopy_binary" in required_tasks:
        if not colon_csv.is_file():
            raise FileNotFoundError(
                "未找到肠镜任务筛选 CSV，请先运行 `python task_data_selection.py` 生成："
                f"\n- {colon_csv}"
            )
        colon_records = build_task_records(
            task_csv_path=colon_csv,
            task_name="colonoscopy_binary",
            min_instances=train_cfg["min_instances"],
            dataset_root=dataset_root,
        )
        colon_records = maybe_limit_records(colon_records, max_num=max_exams_per_task, seed=seed + 1)
        colon_split, _ = build_compatible_split(
            colon_records,
            seed=seed + 1,
            ratios=train_cfg["split_ratio"],
        )
        colon_pos_weight = compute_binary_pos_weight(colon_split["train"]) if colon_split["train"] else [1.0]
        context["colon_split"] = colon_split
        context["colon_pos_weight"] = colon_pos_weight
        context["task_stats"]["colonoscopy_binary"] = {
            "total_records": len(colon_records),
            "train_size": len(colon_split["train"]),
            "val_size": len(colon_split["val"]),
            "test_size": len(colon_split["test"]),
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

    for model_index, model_name in enumerate(names_to_run, start=1):
        print(f"\n[{model_index}/{len(names_to_run)}] 开始训练 {model_name}")
        run_cfg = resolve_run_cfg(train_cfg, model_name)
        if run_overrides:
            run_cfg.update(run_overrides)
        model_param_cfg = dict(model_cfg["models"].get(model_name, {}))

        split_data, pos_weight = resolve_task_training_payload(training_context, model_name)

        model_seed = seed + model_index
        seed_everything(model_seed)
        model, trainer_cfg, task_name, label_names, class_names = build_model_bundle(
            model_name=model_name,
            run_cfg=run_cfg,
            model_param_cfg=model_param_cfg,
            pretrained=pretrained,
            max_epochs=max_epochs,
            patience=patience,
            pos_weight=pos_weight,
            use_multi_gpu=use_multi_gpu,
            run_test=run_test,
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
            train_sampling=train_cfg["train_sampling_strategy"],
            eval_sampling=train_cfg["eval_sampling_strategy"],
            active_gpu_count=active_gpu_count,
            label_names=label_names,
            class_names=class_names,
            cache_root_dir=Path(training_context["task_selection_dir"]).resolve(),
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
    task_dir.mkdir(parents=True, exist_ok=True)
    session_dir = task_dir / auto_series_cfg["output_dir_name"]
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
) -> None:
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

        record = build_auto_series_record(entry, run_dir)
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
                    seed=seed + run_index,
                    max_epochs=max_epochs,
                    patience=patience,
                    image_size=image_size,
                    num_workers=num_workers,
                    pretrained=pretrained,
                    use_multi_gpu=use_multi_gpu,
                    active_gpu_count=active_gpu_count,
                    run_test=bool(auto_series_cfg["run_test"]),
                    run_overrides=entry["run_overrides"],
                    model_param_override=entry["model_params"],
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

    for experiment_index, experiment_cfg in enumerate(auto_ablations_experiments, start=1):
        enabled_entries = [item for item in experiment_cfg["models"] if item["enabled"]]
        print(
            f"\n[自动 Ablations] {experiment_index}/{len(auto_ablations_experiments)} "
            f"| experiment={experiment_cfg['name']} | {experiment_cfg['display_name']}"
        )
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

    path_cfg = load_path_config(Path(args.config))
    train_cfg = load_train_config(Path(args.train_config))
    model_cfg = load_model_config(Path(args.model_config))
    allowed_run_keys = set(train_cfg["default_run"].keys()) - {"monitor_metric", "monitor_mode"}

    if bool(train_cfg["auto_explore"]) and (
        bool(train_cfg["auto_baselines"])
        or bool(train_cfg["auto_sotas"])
        or bool(train_cfg["auto_ablations"])
    ):
        raise ValueError("auto_explore 与 auto_baselines/auto_sotas/auto_ablations 不能同时开启，请二选一")

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
    if torch.cuda.is_available():
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.benchmark = True

    visible_gpu_count = torch.cuda.device_count() if torch.cuda.is_available() else 0
    use_multi_gpu = (not args.disable_multi_gpu) and visible_gpu_count > 1
    active_gpu_count = visible_gpu_count if use_multi_gpu else (1 if visible_gpu_count > 0 else 0)
    pretrained = not args.no_pretrained

    auto_baselines_cfg: dict[str, Any] | None = None
    auto_baseline_entries: list[dict[str, Any]] = []
    auto_sotas_cfg: dict[str, Any] | None = None
    auto_sota_entries: list[dict[str, Any]] = []
    auto_ablations_cfg: dict[str, Any] | None = None
    auto_ablation_experiments: list[dict[str, Any]] = []
    requested_names = [item.strip() for item in args.models.split(",") if item.strip()]

    if bool(train_cfg["auto_baselines"]):
        auto_baselines_cfg = load_auto_baselines_config(
            Path(args.auto_baselines_config),
            allowed_run_keys=allowed_run_keys,
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
            Path(args.auto_sotas_config),
            allowed_run_keys=allowed_run_keys,
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
            Path(args.auto_ablations_config),
            allowed_run_keys=allowed_run_keys,
        )
        auto_ablation_experiments = selected_auto_ablation_experiments(
            train_cfg["auto_ablations"],
            auto_ablations_cfg,
        )

    if auto_ablations_cfg is not None and requested_names:
        raise ValueError("auto_ablations 模式当前不支持 --models，请通过 train.yaml 的 auto_ablations 选择实验")

    if auto_baselines_cfg is not None or auto_sotas_cfg is not None or auto_ablations_cfg is not None:
        selected_names = [item["name"] for item in auto_baseline_entries] + [item["name"] for item in auto_sota_entries]
        selected_task_model_names = (
            [resolve_series_entry_model_name(item) for item in auto_baseline_entries]
            + [resolve_series_entry_model_name(item) for item in auto_sota_entries]
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

    required_tasks = resolve_required_tasks(selected_task_model_names)
    print("=" * 72)
    print("训练配置")
    print(f"模型={','.join(selected_names)}")
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
    if auto_baselines_cfg is not None or auto_sotas_cfg is not None or auto_ablations_cfg is not None:
        return

    auto_explore_cfg: dict[str, Any] | None = None
    if bool(train_cfg["auto_explore"]):
        auto_explore_cfg = load_auto_explore_config(
            Path(args.auto_explore_config),
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
        )
        run_dirs.append(run_dir)

    print("\n训练完成。")
    for run_dir in run_dirs:
        print(f"训练目录：{run_dir}")


if __name__ == "__main__":
    main()
