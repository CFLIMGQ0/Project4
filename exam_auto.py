#!/usr/bin/env python3
from __future__ import annotations

import argparse
import gc
import inspect
import os
import shutil
import sys
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any


sys.dont_write_bytecode = True
os.environ.setdefault("PYTHONDONTWRITEBYTECODE", "1")

PROJECT_ROOT = Path(__file__).resolve().parent


def _ensure_project_runtime_python() -> None:
    import importlib.util

    if importlib.util.find_spec("torch") is not None:
        return
    if os.environ.get("PROJECT4_EXAM_AUTO_REEXEC") == "1":
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

        os.environ["PROJECT4_EXAM_AUTO_REEXEC"] = "1"
        print(
            f"[exam_auto.py] 当前解释器 {current_python} 缺少 torch，自动切换到 {candidate}",
            file=sys.stderr,
        )
        os.execv(str(candidate), [str(candidate), *sys.argv])


_ensure_project_runtime_python()

MEMORY_KEYS = {
    "batch_size",
    "eval_batch_size",
    "train_max_instances",
    "eval_max_instances",
    "train_max_batch_instances",
    "eval_max_batch_instances",
}


@dataclass
class ExamEntry:
    name: str
    base_model_name: str
    task_name: str
    model_params: dict[str, Any]
    run_overrides: dict[str, Any]
    source: str


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="快速检查 python train.py 当前默认自动训练计划中的模型是否存在显存/运行中断风险"
    )
    parser.add_argument("--task", type=str, default="", help="任务名，默认读取 train.yaml 内的 task_name")
    parser.add_argument("--config", type=str, default="", help="路径配置文件，默认 configs/<task>/path.yaml")
    parser.add_argument("--train-config", type=str, default="", help="训练配置文件，默认 configs/<task>/train.yaml")
    parser.add_argument("--model-config", type=str, default="", help="模型配置文件，默认 configs/<task>/model.yaml")
    parser.add_argument("--auto-explore-config", type=str, default="", help="auto_explore 配置文件")
    parser.add_argument("--auto-baselines-config", type=str, default="", help="auto_baselines 配置文件")
    parser.add_argument("--auto-sotas-config", type=str, default="", help="auto_sotas 配置文件")
    parser.add_argument("--auto-ablations-config", type=str, default="", help="auto_ablations 配置文件")
    parser.add_argument("--models", type=str, default="", help="只检查指定模型，多个模型用英文逗号分隔")
    parser.add_argument("--disable-multi-gpu", action="store_true", help="与 train.py 一致的多卡禁用开关")
    parser.add_argument("--no-pretrained", action="store_true", help="跳过预训练权重加载，仅用于更快定位结构问题")
    parser.add_argument("--skip-eval", action="store_true", help="只检查训练前向/反向，不额外检查 eval batch")
    parser.add_argument("--skip-data-pipeline", action="store_true", help="跳过真实数据管线首批次检查")
    parser.add_argument(
        "--memory-safety-ratio",
        type=float,
        default=0.90,
        help="单卡峰值 reserved 显存不能超过检查开始时空闲显存的比例，默认 0.90",
    )
    parser.add_argument(
        "--min-host-memory-gb",
        type=float,
        default=12.0,
        help="启动检查前要求的主机可用内存下限，默认 12GB",
    )
    parser.add_argument(
        "--host-memory-floor-gb",
        type=float,
        default=4.0,
        help="检查过程中主机可用内存不能低于该值，默认 4GB",
    )
    return parser.parse_args()


def project_path(raw_path: str | Path) -> Path:
    path = Path(raw_path).expanduser()
    if path.is_absolute():
        return path
    return (PROJECT_ROOT / path).resolve()


def read_yaml_dict(path: Path) -> dict[str, Any]:
    import yaml

    if not path.is_file():
        raise FileNotFoundError(f"未找到配置文件: {path}")
    payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    if payload is None:
        return {}
    if not isinstance(payload, dict):
        raise ValueError(f"配置文件格式错误: {path}")
    return payload


def bytes_to_gb(value: int | float | None) -> float:
    if value is None:
        return 0.0
    return float(value) / (1024.0 ** 3)


def read_meminfo_bytes() -> dict[str, int]:
    result: dict[str, int] = {}
    try:
        for line in Path("/proc/meminfo").read_text(encoding="utf-8").splitlines():
            if ":" not in line:
                continue
            key, value = line.split(":", 1)
            fields = value.strip().split()
            if not fields:
                continue
            numeric = int(fields[0])
            if len(fields) >= 2 and fields[1].lower() == "kb":
                numeric *= 1024
            result[key] = numeric
    except Exception:
        return {}
    return result


def host_memory_state() -> dict[str, float]:
    meminfo = read_meminfo_bytes()
    return {
        "mem_total_gb": bytes_to_gb(meminfo.get("MemTotal")),
        "mem_available_gb": bytes_to_gb(meminfo.get("MemAvailable")),
        "swap_total_gb": bytes_to_gb(meminfo.get("SwapTotal")),
        "swap_free_gb": bytes_to_gb(meminfo.get("SwapFree")),
    }


def top_memory_process_lines(limit: int = 8) -> list[str]:
    try:
        import subprocess

        completed = subprocess.run(
            ["ps", "-eo", "pid,ppid,comm,rss,args", "--sort=-rss"],
            check=False,
            capture_output=True,
            text=True,
            timeout=3,
        )
    except Exception:
        return []
    if completed.returncode != 0:
        return []
    lines = completed.stdout.splitlines()
    return lines[: max(1, limit + 1)]


def format_host_memory_state(state: dict[str, float]) -> str:
    return (
        f"MemAvailable={state['mem_available_gb']:.2f}GB/"
        f"{state['mem_total_gb']:.2f}GB, "
        f"SwapFree={state['swap_free_gb']:.2f}GB/"
        f"{state['swap_total_gb']:.2f}GB"
    )


def assert_host_memory(stage: str, min_available_gb: float) -> None:
    if min_available_gb <= 0:
        return
    state = host_memory_state()
    available = float(state.get("mem_available_gb", 0.0))
    if available >= min_available_gb:
        return

    lines = "\n".join(top_memory_process_lines(limit=10))
    raise RuntimeError(
        f"{stage} 主机可用内存不足：{format_host_memory_state(state)}，"
        f"要求至少 {min_available_gb:.2f}GB。\n"
        f"当前最占内存进程：\n{lines}"
    )


def initial_train_config_path(args: argparse.Namespace) -> Path:
    if str(args.train_config).strip():
        return project_path(args.train_config)
    task_name = str(args.task).strip() or "task2"
    return project_path(Path("configs") / task_name / "train.yaml")


def preload_cuda_env(train_config_path: Path) -> None:
    payload = read_yaml_dict(train_config_path)
    gpu_ids = payload.get("gpu_ids", [0])
    if not isinstance(gpu_ids, list) or not gpu_ids:
        return
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = ",".join(str(int(item)) for item in gpu_ids)


def cleanup_stale_temp_dirs() -> None:
    temp_root = Path(tempfile.gettempdir())
    now = time.time()
    for candidate in temp_root.glob("project4_exam_auto_*"):
        try:
            if not candidate.is_dir():
                continue
            if now - candidate.stat().st_mtime < 24 * 3600:
                continue
            shutil.rmtree(candidate, ignore_errors=True)
        except Exception:
            continue


def set_process_temp_cache(temp_root: Path) -> None:
    os.environ["TORCH_HOME"] = str(temp_root / "torch_home")
    os.environ["XDG_CACHE_HOME"] = str(temp_root / "xdg_cache")
    os.environ["MPLCONFIGDIR"] = str(temp_root / "matplotlib")
    os.environ["HF_HOME"] = str(temp_root / "hf_home")


def resolve_config_path(raw_path: str, default_path: str) -> Path:
    if str(raw_path).strip():
        return project_path(raw_path)
    return project_path(default_path)


def requested_model_names(raw_models: str) -> list[str]:
    return [item.strip() for item in str(raw_models).split(",") if item.strip()]


def validate_auto_mode_conflicts(train_module: Any, train_cfg: dict[str, Any]) -> None:
    if bool(train_cfg["auto_explore"]) and (
        bool(train_cfg["auto_baselines"])
        or bool(train_cfg["auto_sotas"])
        or bool(train_cfg["auto_ablations"])
        or bool(train_cfg["auto_exp_1"])
        or bool(train_cfg["auto_exp_2"])
        or bool(train_cfg["auto_exp_3"])
        or bool(train_cfg.get("auto_exp_4", False))
    ):
        raise ValueError("auto_explore 与其他自动批量模式不能同时开启")

    auto_mode_count = int(bool(train_cfg["auto_baselines"]))
    auto_mode_count += int(bool(train_cfg["auto_sotas"]))
    auto_mode_count += int(bool(train_cfg["auto_exp_1"]))
    auto_mode_count += int(bool(train_cfg["auto_exp_2"]))
    auto_mode_count += int(bool(train_cfg["auto_exp_3"]))
    auto_mode_count += int(bool(train_cfg.get("auto_exp_4", False)))
    auto_mode_count += int(bool(train_cfg["auto_ablations"]))
    if auto_mode_count > 1:
        raise ValueError("auto_baselines、auto_sotas、auto_ablations、auto_exp_1、auto_exp_2、auto_exp_3、auto_exp_4 只能开启一个")

    del train_module


def stress_overrides_from_search_space(search_space: dict[str, Any]) -> dict[str, Any]:
    overrides: dict[str, Any] = {}
    for key, spec in search_space.items():
        if key not in MEMORY_KEYS:
            continue
        if not isinstance(spec, dict) or not bool(spec.get("enabled", False)):
            continue

        values = spec.get("values")
        if isinstance(values, list) and values:
            numeric_values = [item for item in values if isinstance(item, (int, float))]
            if numeric_values:
                overrides[key] = max(numeric_values)
            continue

        if "max" in spec and isinstance(spec["max"], (int, float)):
            overrides[key] = spec["max"]
        elif "high" in spec and isinstance(spec["high"], (int, float)):
            overrides[key] = spec["high"]
    return overrides


def entry_from_series_item(item: dict[str, Any], source: str, train_module: Any) -> ExamEntry:
    base_model_name = train_module.resolve_series_entry_model_name(item)
    return ExamEntry(
        name=str(item.get("name", base_model_name)).strip(),
        base_model_name=base_model_name,
        task_name=str(item["task_name"]),
        model_params=dict(item.get("model_params", {}) or {}),
        run_overrides=dict(item.get("run_overrides", {}) or {}),
        source=source,
    )


def resolve_exam_entries(
    *,
    args: argparse.Namespace,
    train_module: Any,
    train_cfg: dict[str, Any],
    selected_task_name: str,
) -> list[ExamEntry]:
    validate_auto_mode_conflicts(train_module, train_cfg)
    requested_names = requested_model_names(args.models)
    allowed_run_keys = set(train_cfg["default_run"].keys()) - {"monitor_metric", "monitor_mode"}

    if bool(train_cfg["auto_baselines"]):
        cfg = train_module.load_auto_baselines_config(
            resolve_config_path(
                args.auto_baselines_config,
                train_module.resolve_default_config_path(selected_task_name, "auto_baselines.yaml"),
            ),
            allowed_run_keys=allowed_run_keys,
            selected_task_name=selected_task_name,
        )
        selected = train_module.selected_auto_baseline_model_entries(args.models, cfg)
        return [entry_from_series_item(item, "auto_baselines", train_module) for item in selected]

    if bool(train_cfg["auto_sotas"]):
        cfg = train_module.load_auto_sotas_config(
            resolve_config_path(
                args.auto_sotas_config,
                train_module.resolve_default_config_path(selected_task_name, "auto_sotas.yaml"),
            ),
            allowed_run_keys=allowed_run_keys,
            selected_task_name=selected_task_name,
        )
        selected = train_module.selected_auto_sota_model_entries(args.models, cfg)
        return [entry_from_series_item(item, "auto_sotas", train_module) for item in selected]

    if bool(train_cfg["auto_ablations"]):
        if requested_names:
            raise ValueError("exam_auto.py 在 auto_ablations 模式下不支持 --models，请通过 train.yaml 选择实验")
        cfg = train_module.load_auto_ablations_config(
            resolve_config_path(
                args.auto_ablations_config,
                train_module.resolve_default_config_path(selected_task_name, "auto_ablations.yaml"),
            ),
            allowed_run_keys=allowed_run_keys,
            selected_task_name=selected_task_name,
        )
        experiments = train_module.selected_auto_ablation_experiments(train_cfg["auto_ablations"], cfg)
        entries: list[ExamEntry] = []
        for experiment in experiments:
            for item in experiment["models"]:
                if not bool(item.get("enabled", True)):
                    continue
                entry = entry_from_series_item(item, f"auto_ablations/{experiment['name']}", train_module)
                entry.name = f"{experiment['name']}:{entry.name}"
                entries.append(entry)
        return entries

    if bool(train_cfg["auto_exp_1"]):
        entries = train_module.build_auto_exp1_entries(
            selected_task_name=selected_task_name,
            requested_names=requested_names if requested_names else None,
        )
        return [entry_from_series_item(item, "auto_exp_1", train_module) for item in entries]

    if bool(train_cfg["auto_exp_2"]):
        entries = train_module.build_auto_exp2_entries(
            selected_task_name=selected_task_name,
            requested_names=requested_names if requested_names else None,
            skip_names=train_cfg.get("auto_exp_2_skip_models", []),
        )
        return [entry_from_series_item(item, "auto_exp_2", train_module) for item in entries]

    if bool(train_cfg["auto_exp_3"]):
        entries = train_module.build_auto_exp3_entries(
            selected_task_name=selected_task_name,
            requested_names=requested_names if requested_names else None,
        )
        return [entry_from_series_item(item, "auto_exp_3", train_module) for item in entries]

    if bool(train_cfg.get("auto_exp_4", False)):
        entries = train_module.build_auto_exp4_entries(
            selected_task_name=selected_task_name,
            requested_names=requested_names if requested_names else None,
        )
        return [entry_from_series_item(item, "auto_exp_4", train_module) for item in entries]

    if bool(train_cfg["auto_explore"]):
        cfg = train_module.load_auto_explore_config(
            resolve_config_path(
                args.auto_explore_config,
                train_module.resolve_default_config_path(selected_task_name, "auto_explore.yaml"),
            ),
            allowed_run_keys=allowed_run_keys,
        )
        names = train_module.selected_model_names(args.models, train_cfg)
        run_overrides = stress_overrides_from_search_space(cfg.get("search_space", {}))
        return [
            ExamEntry(
                name=name,
                base_model_name=name,
                task_name=train_module.resolve_model_task_meta(name, selected_task_name)["task_name"],
                model_params={},
                run_overrides=dict(run_overrides),
                source="auto_explore",
            )
            for name in names
        ]

    names = train_module.selected_model_names(args.models, train_cfg)
    return [
        ExamEntry(
            name=name,
            base_model_name=name,
            task_name=train_module.resolve_model_task_meta(name, selected_task_name)["task_name"],
            model_params={},
            run_overrides={},
            source="enabled_models",
        )
        for name in names
    ]


def normalize_run_batch(value: Any, active_gpu_count: int, train_module: Any) -> int:
    return train_module.normalize_batch_size(int(value), active_gpu_count)


def make_batch(
    *,
    batch_size: int,
    max_instances: int,
    image_size: int,
    num_labels: int,
    device: Any,
    torch_module: Any,
) -> dict[str, Any]:
    images = torch_module.zeros(
        (batch_size, max_instances, 3, image_size, image_size),
        dtype=torch_module.float32,
        device=device,
    )
    mask = torch_module.ones((batch_size, max_instances), dtype=torch_module.bool, device=device)
    labels = torch_module.zeros((batch_size, num_labels), dtype=torch_module.float32, device=device)
    if num_labels > 0:
        rows = torch_module.arange(batch_size, device=device)
        labels[rows, rows % num_labels] = 1.0
        if num_labels > 1:
            labels[rows, (rows * 3 + 1) % num_labels] = (rows % 2 == 0).float()
    return {"images": images, "mask": mask, "labels": labels}


def synthetic_instance_count(run_cfg: dict[str, Any], phase: str, batch_size: int) -> int:
    raw_key = f"{phase}_max_instances"
    batch_key = f"{phase}_max_batch_instances"
    max_instances = int(run_cfg.get(raw_key, 1))
    if max_instances > 0:
        return max_instances
    max_batch_instances = int(run_cfg.get(batch_key, batch_size))
    return max(1, max_batch_instances // max(1, int(batch_size)))


def forward_model(model: Any, raw_model: Any, batch: dict[str, Any], current_epoch: float) -> dict[str, Any]:
    params = inspect.signature(raw_model.forward).parameters
    kwargs: dict[str, Any] = {
        "images": batch["images"],
        "mask": batch["mask"],
    }
    if "labels" in params:
        kwargs["labels"] = batch["labels"]
    if "current_epoch" in params:
        kwargs["current_epoch"] = float(current_epoch)
    outputs = model(**kwargs)
    if not isinstance(outputs, dict) or "logits" not in outputs:
        raise RuntimeError("模型 forward 必须返回包含 logits 的 dict")
    return outputs


def build_optimizer(model: Any, trainer_cfg: Any, torch_module: Any) -> Any:
    optimizer_name = str(trainer_cfg.optimizer_name).strip().lower()
    if optimizer_name == "adamw":
        return torch_module.optim.AdamW(
            model.parameters(),
            lr=float(trainer_cfg.lr),
            weight_decay=float(trainer_cfg.weight_decay),
        )
    if optimizer_name == "adam":
        return torch_module.optim.Adam(
            model.parameters(),
            lr=float(trainer_cfg.lr),
            weight_decay=float(trainer_cfg.weight_decay),
        )
    if optimizer_name == "sgd":
        return torch_module.optim.SGD(
            model.parameters(),
            lr=float(trainer_cfg.lr),
            weight_decay=float(trainer_cfg.weight_decay),
            momentum=0.9,
            nesterov=True,
        )
    raise ValueError(f"未知 optimizer_name: {trainer_cfg.optimizer_name}")


def build_criterion(trainer_cfg: Any, device: Any, torch_module: Any) -> Any:
    from training.losses import build_binary_criterion, build_multilabel_criterion

    pos_weight_tensor = None
    if trainer_cfg.pos_weight is not None:
        pos_weight_tensor = torch_module.tensor(trainer_cfg.pos_weight, dtype=torch_module.float32, device=device)
    if trainer_cfg.task_type == "gastro_multilabel":
        return build_multilabel_criterion(loss_name=trainer_cfg.loss_name, pos_weight=pos_weight_tensor)
    if trainer_cfg.task_type == "colonoscopy_binary":
        return build_binary_criterion(loss_name=trainer_cfg.loss_name, pos_weight=pos_weight_tensor)
    raise ValueError(f"未知 task_type: {trainer_cfg.task_type}")


def primary_loss(
    *,
    raw_model: Any,
    criterion: Any,
    outputs: dict[str, Any],
    labels: Any,
    current_epoch: float,
    train_mode: bool,
) -> Any:
    custom_loss_fn = getattr(raw_model, "compute_loss", None)
    if callable(custom_loss_fn):
        loss = custom_loss_fn(
            outputs=outputs,
            labels=labels.float(),
            criterion=criterion,
            current_epoch=float(current_epoch),
            train_mode=bool(train_mode),
        )
        return loss.mean() if hasattr(loss, "ndim") and loss.ndim > 0 else loss
    loss = criterion(outputs["logits"], labels.float())
    return loss.mean() if loss.ndim > 0 else loss


def aux_loss(outputs: dict[str, Any], trainer_cfg: Any, device: Any, torch_module: Any) -> Any:
    values = outputs.get("aux_losses", {})
    total = torch_module.zeros((), device=device)
    if not isinstance(values, dict):
        return total
    for key, value in values.items():
        if not torch_module.is_tensor(value):
            continue
        current = value.mean() if value.ndim > 0 else value
        total = total + float(trainer_cfg.aux_loss_weights.get(key, 1.0)) * current
    return total


def reset_cuda_peak(torch_module: Any) -> None:
    if not torch_module.cuda.is_available():
        return
    for device_index in range(torch_module.cuda.device_count()):
        torch_module.cuda.reset_peak_memory_stats(device_index)


def cuda_free_memory(torch_module: Any) -> list[int]:
    if not torch_module.cuda.is_available():
        return []
    free_values: list[int] = []
    for device_index in range(torch_module.cuda.device_count()):
        free_bytes, _ = torch_module.cuda.mem_get_info(device_index)
        free_values.append(int(free_bytes))
    return free_values


def cuda_peak_text(torch_module: Any, free_before: list[int], memory_safety_ratio: float) -> tuple[str, bool]:
    if not torch_module.cuda.is_available():
        return "CPU", True

    parts: list[str] = []
    ok = True
    for device_index in range(torch_module.cuda.device_count()):
        props = torch_module.cuda.get_device_properties(device_index)
        peak_reserved = int(torch_module.cuda.max_memory_reserved(device_index))
        peak_allocated = int(torch_module.cuda.max_memory_allocated(device_index))
        total = int(props.total_memory)
        free_ref = free_before[device_index] if device_index < len(free_before) else total
        reserved_ratio = peak_reserved / max(1, free_ref)
        if reserved_ratio > memory_safety_ratio:
            ok = False
        parts.append(
            f"cuda:{device_index} allocated={peak_allocated / 1024 ** 3:.2f}GiB "
            f"reserved={peak_reserved / 1024 ** 3:.2f}GiB/{free_ref / 1024 ** 3:.2f}GiB "
            f"({reserved_ratio * 100:.1f}%)"
        )
    return "; ".join(parts), ok


def clear_runtime_memory(torch_module: Any | None = None) -> None:
    gc.collect()
    if torch_module is not None and torch_module.cuda.is_available():
        torch_module.cuda.empty_cache()
        try:
            torch_module.cuda.ipc_collect()
        except Exception:
            pass


def run_data_pipeline_probe(
    *,
    args: argparse.Namespace,
    entries: list[ExamEntry],
    train_module: Any,
    train_cfg: dict[str, Any],
    path_cfg: dict[str, Any],
    selected_task_name: str,
    active_gpu_count: int,
) -> None:
    if bool(args.skip_data_pipeline):
        print("[数据管线] 已跳过真实 DataLoader 首批次检查")
        return
    if not entries:
        return

    first_entry = entries[0]
    required_tasks = train_module.resolve_required_tasks(
        [entry.base_model_name for entry in entries],
        selected_task_name,
    )
    old_report_env = os.environ.get("PROJECT4_SUPPRESS_CLASS_BALANCE_REPORT")
    old_cache_env = os.environ.get("PROJECT4_DISABLE_DISK_CACHE_WRITE")
    os.environ["PROJECT4_SUPPRESS_CLASS_BALANCE_REPORT"] = "1"
    os.environ["PROJECT4_DISABLE_DISK_CACHE_WRITE"] = "1"

    try:
        print("[数据管线] 构建训练上下文：会读取真实 datalist、执行同款划分和类别平衡，但不写报告文件")
        training_context = train_module.prepare_training_context(
            path_cfg=path_cfg,
            train_cfg=train_cfg,
            seed=int(train_cfg["seed"]),
            max_exams_per_task=int(train_cfg["max_exams_per_task"]),
            required_tasks=required_tasks,
        )

        run_cfg = train_module.resolve_run_cfg(train_cfg, first_entry.base_model_name)
        run_cfg.update(first_entry.run_overrides)
        entry_num_workers = int(first_entry.run_overrides.get("num_workers", train_cfg["num_workers"]))
        if entry_num_workers < 0:
            entry_num_workers = int(train_cfg["num_workers"])

        task_meta = train_module.resolve_model_task_meta(first_entry.base_model_name, selected_task_name)
        split_data, _ = train_module.resolve_task_training_payload(training_context, task_meta["task_name"])
        resolved_cache_root_dir, resolved_cache_dir, legacy_cache_dirs = train_module.resolve_image_cache_directories(
            task_name=task_meta["task_name"],
            cache_root_dir=Path(training_context["task_selection_dir"]).resolve(),
            run_cfg=run_cfg,
        )
        del resolved_cache_root_dir

        print(
            "[数据管线] 检查首个真实训练 batch："
            f"model={first_entry.name}, num_workers={entry_num_workers}, "
            f"cache_mode={run_cfg.get('image_cache_mode')}, cache_warmup={bool(run_cfg.get('image_cache_warmup', False))}"
        )
        train_loader, val_loader, test_loader = train_module.build_loaders(
            split_data=split_data,
            task_name=task_meta["task_name"],
            image_size=int(train_cfg["image_size"]),
            num_workers=entry_num_workers,
            train_batch_size=train_module.normalize_batch_size(int(run_cfg.get("batch_size", 3)), active_gpu_count),
            eval_batch_size=train_module.normalize_batch_size(int(run_cfg.get("eval_batch_size", 3)), active_gpu_count),
            train_max_instances=int(run_cfg.get("train_max_instances", 32)),
            eval_max_instances=int(run_cfg.get("eval_max_instances", 32)),
            min_instances=int(train_cfg["min_instances"]),
            train_sampling=str(run_cfg.get("train_sampling_strategy", train_cfg["train_sampling_strategy"])),
            eval_sampling=str(run_cfg.get("eval_sampling_strategy", train_cfg["eval_sampling_strategy"])),
            random_instance_dropout=float(run_cfg.get("random_instance_dropout", 0.0)),
            train_max_batch_instances=int(run_cfg.get("train_max_batch_instances", 96)),
            eval_max_batch_instances=int(run_cfg.get("eval_max_batch_instances", 96)),
            seed=int(train_cfg["seed"]),
            pin_memory=bool(run_cfg.get("pin_memory", True)),
            persistent_workers=bool(run_cfg.get("persistent_workers", True)),
            loader_prefetch_factor=int(run_cfg.get("loader_prefetch_factor", 2)),
            image_cache_mode=str(run_cfg.get("image_cache_mode", "none")),
            image_cache_dir=resolved_cache_dir,
            image_cache_manifest=str(run_cfg.get("image_cache_manifest", "")).strip() or None,
            legacy_image_cache_dirs=legacy_cache_dirs,
            image_cache_warmup=bool(run_cfg.get("image_cache_warmup", False)),
            memory_cache_size=int(run_cfg.get("memory_cache_size", 0)),
        )
        iterator = iter(train_loader)
        batch = next(iterator)
        images_shape = tuple(batch["images"].shape)
        labels_shape = tuple(batch["labels"].shape)
        del batch, iterator, train_loader, val_loader, test_loader, training_context
        clear_runtime_memory()
        assert_host_memory("真实数据管线检查后", float(args.host_memory_floor_gb))
        print(f"[数据管线] 通过：首批次 images={images_shape}, labels={labels_shape}")
    finally:
        if old_report_env is None:
            os.environ.pop("PROJECT4_SUPPRESS_CLASS_BALANCE_REPORT", None)
        else:
            os.environ["PROJECT4_SUPPRESS_CLASS_BALANCE_REPORT"] = old_report_env
        if old_cache_env is None:
            os.environ.pop("PROJECT4_DISABLE_DISK_CACHE_WRITE", None)
        else:
            os.environ["PROJECT4_DISABLE_DISK_CACHE_WRITE"] = old_cache_env


def run_memory_probe(
    *,
    model: Any,
    trainer_cfg: Any,
    image_size: int,
    run_cfg: dict[str, Any],
    label_count: int,
    active_gpu_count: int,
    skip_eval: bool,
    memory_safety_ratio: float,
    train_module: Any,
    torch_module: Any,
) -> str:
    from torch.cuda.amp import GradScaler, autocast
    from torch import nn

    device = torch_module.device("cuda" if torch_module.cuda.is_available() else "cpu")
    free_before = cuda_free_memory(torch_module)
    reset_cuda_peak(torch_module)

    model = model.to(device)
    if bool(trainer_cfg.use_multi_gpu) and torch_module.cuda.device_count() > 1:
        model = nn.DataParallel(model)
    raw_model = model.module if isinstance(model, nn.DataParallel) else model
    criterion = build_criterion(trainer_cfg, device, torch_module)
    optimizer = build_optimizer(model, trainer_cfg, torch_module)
    scaler = GradScaler(enabled=bool(trainer_cfg.amp) and device.type == "cuda")

    train_batch_size = normalize_run_batch(run_cfg.get("batch_size", 1), active_gpu_count, train_module)
    train_instances = synthetic_instance_count(run_cfg, "train", train_batch_size)
    train_batch = make_batch(
        batch_size=train_batch_size,
        max_instances=train_instances,
        image_size=image_size,
        num_labels=label_count,
        device=device,
        torch_module=torch_module,
    )

    model.train()
    optimizer.zero_grad(set_to_none=True)
    with autocast(enabled=bool(trainer_cfg.amp) and device.type == "cuda"):
        outputs = forward_model(model, raw_model, train_batch, current_epoch=1.0)
        loss_main = primary_loss(
            raw_model=raw_model,
            criterion=criterion,
            outputs=outputs,
            labels=train_batch["labels"],
            current_epoch=1.0,
            train_mode=True,
        )
        loss = loss_main + aux_loss(outputs, trainer_cfg, device, torch_module)
    scaler.scale(loss).backward()
    scaler.step(optimizer)
    scaler.update()
    optimizer.zero_grad(set_to_none=True)
    if torch_module.cuda.is_available():
        torch_module.cuda.synchronize()

    if not skip_eval:
        eval_batch_size = normalize_run_batch(
            run_cfg.get("eval_batch_size", train_batch_size),
            active_gpu_count,
            train_module,
        )
        eval_instances = synthetic_instance_count(run_cfg, "eval", eval_batch_size)
        eval_batch = make_batch(
            batch_size=eval_batch_size,
            max_instances=eval_instances,
            image_size=image_size,
            num_labels=label_count,
            device=device,
            torch_module=torch_module,
        )
        model.eval()
        with torch_module.no_grad():
            with autocast(enabled=bool(trainer_cfg.amp) and device.type == "cuda"):
                forward_model(model, raw_model, eval_batch, current_epoch=1.0)
        if torch_module.cuda.is_available():
            torch_module.cuda.synchronize()

    peak_text, has_margin = cuda_peak_text(torch_module, free_before, memory_safety_ratio)
    if not has_margin:
        raise RuntimeError(
            f"显存峰值超过安全阈值 {memory_safety_ratio:.0%}。峰值: {peak_text}"
        )
    return peak_text


def check_one_entry(
    *,
    entry: ExamEntry,
    entry_index: int,
    total_entries: int,
    args: argparse.Namespace,
    train_module: Any,
    torch_module: Any,
    train_cfg: dict[str, Any],
    model_cfg: dict[str, Any],
    selected_task_name: str,
    base_use_multi_gpu: bool,
    base_active_gpu_count: int,
) -> None:
    run_cfg = train_module.resolve_run_cfg(train_cfg, entry.base_model_name)
    run_cfg.update(entry.run_overrides)

    entry_use_multi_gpu = base_use_multi_gpu
    if bool(run_cfg.get("disable_multi_gpu", False)):
        entry_use_multi_gpu = False
    elif "use_multi_gpu" in run_cfg:
        entry_use_multi_gpu = bool(run_cfg["use_multi_gpu"])
    entry_active_gpu_count = base_active_gpu_count if entry_use_multi_gpu else (1 if torch_module.cuda.is_available() else 0)

    model_param_cfg = dict(model_cfg["models"].get(entry.base_model_name, {}))
    model_param_cfg.update(entry.model_params)
    task_meta = train_module.resolve_model_task_meta(entry.base_model_name, selected_task_name)
    label_count = int(task_meta["num_labels"])
    pos_weight = [1.0] * label_count

    train_module.seed_everything(int(train_cfg["seed"]) + 90000 + entry_index)
    model, trainer_cfg, _, label_names, _ = train_module.build_model_bundle(
        model_name=entry.base_model_name,
        task_name=entry.task_name,
        run_cfg=run_cfg,
        model_param_cfg=model_param_cfg,
        pretrained=not args.no_pretrained,
        max_epochs=int(train_cfg["max_epochs"]),
        patience=int(train_cfg["patience"]),
        pos_weight=pos_weight,
        use_multi_gpu=entry_use_multi_gpu,
        run_test=True,
    )

    train_shape = (
        normalize_run_batch(run_cfg.get("batch_size", 1), entry_active_gpu_count, train_module),
        0,
    )
    train_shape = (train_shape[0], synthetic_instance_count(run_cfg, "train", train_shape[0]))
    eval_shape = (
        normalize_run_batch(run_cfg.get("eval_batch_size", train_shape[0]), entry_active_gpu_count, train_module),
        0,
    )
    eval_shape = (eval_shape[0], synthetic_instance_count(run_cfg, "eval", eval_shape[0]))
    multi_gpu_text = "multi-gpu" if entry_use_multi_gpu else "single-gpu/cpu"
    print(
        f"[检查] {entry_index}/{total_entries} {entry.name} "
        f"(base={entry.base_model_name}, {entry.source}, {multi_gpu_text}) "
        f"train={train_shape[0]}x{train_shape[1]} eval={eval_shape[0]}x{eval_shape[1]}"
    )

    try:
        peak_text = run_memory_probe(
            model=model,
            trainer_cfg=trainer_cfg,
            image_size=int(train_cfg["image_size"]),
            run_cfg=run_cfg,
            label_count=len(label_names),
            active_gpu_count=entry_active_gpu_count,
            skip_eval=bool(args.skip_eval),
            memory_safety_ratio=float(args.memory_safety_ratio),
            train_module=train_module,
            torch_module=torch_module,
        )
        print(f"[通过] {entry.name} | {peak_text}")
    finally:
        del model
        clear_runtime_memory(torch_module)


def main() -> int:
    args = parse_args()
    train_config_path = initial_train_config_path(args)
    preload_cuda_env(train_config_path)

    import train
    import torch

    cleanup_stale_temp_dirs()
    train_cfg = train.load_train_config(train_config_path)
    selected_task_name = train.resolve_train_task_name(str(args.task).strip() or None, train_cfg)
    train_cfg["task_name"] = selected_task_name
    path_config_path = (
        project_path(args.config)
        if str(args.config).strip()
        else project_path(train.resolve_default_config_path(selected_task_name, "path.yaml"))
    )
    path_cfg = train.load_path_config(path_config_path)
    model_config_path = (
        project_path(args.model_config)
        if str(args.model_config).strip()
        else project_path(train.resolve_default_config_path(selected_task_name, "model.yaml"))
    )
    model_cfg = train.load_model_config(model_config_path)

    torch.set_float32_matmul_precision("medium")
    if torch.cuda.is_available():
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.benchmark = True

    visible_gpu_count = torch.cuda.device_count() if torch.cuda.is_available() else 0
    base_use_multi_gpu = (not args.disable_multi_gpu) and visible_gpu_count > 1
    base_active_gpu_count = visible_gpu_count if base_use_multi_gpu else (1 if visible_gpu_count > 0 else 0)
    assert_host_memory("exam_auto 启动前", float(args.min_host_memory_gb))

    entries = resolve_exam_entries(
        args=args,
        train_module=train,
        train_cfg=train_cfg,
        selected_task_name=selected_task_name,
    )
    if not entries:
        raise ValueError("没有需要检查的模型")

    print("=" * 72)
    print("exam_auto 快速显存检查")
    print(f"path_config={path_config_path}")
    print(f"train_config={train_config_path}")
    print(f"model_config={model_config_path}")
    print(f"task={selected_task_name}")
    print(f"models={','.join(entry.name for entry in entries)}")
    print(f"cuda_visible_devices={os.environ.get('CUDA_VISIBLE_DEVICES', '') or '未设置'}")
    print(f"visible_gpu_count={visible_gpu_count} pretrained={not args.no_pretrained}")
    print(f"host_memory={format_host_memory_state(host_memory_state())}")
    print("说明：会先检查真实数据管线首批次，再做合成 bag 的前向、反向和优化器一步；不进入正式训练。")
    print("=" * 72)

    with tempfile.TemporaryDirectory(prefix="project4_exam_auto_") as temp_dir:
        set_process_temp_cache(Path(temp_dir))
        try:
            run_data_pipeline_probe(
                args=args,
                entries=entries,
                train_module=train,
                train_cfg=train_cfg,
                path_cfg=path_cfg,
                selected_task_name=selected_task_name,
                active_gpu_count=base_active_gpu_count,
            )
            for index, entry in enumerate(entries, start=1):
                check_one_entry(
                    entry=entry,
                    entry_index=index,
                    total_entries=len(entries),
                    args=args,
                    train_module=train,
                    torch_module=torch,
                    train_cfg=train_cfg,
                    model_cfg=model_cfg,
                    selected_task_name=selected_task_name,
                    base_use_multi_gpu=base_use_multi_gpu,
                    base_active_gpu_count=base_active_gpu_count,
                )
                assert_host_memory(f"{entry.name} 检查后", float(args.host_memory_floor_gb))
        finally:
            clear_runtime_memory(torch)

    print("=" * 72)
    print("exam_auto 检查完成：所有模型均通过当前配置下的快速显存试运行。")
    print("=" * 72)
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except KeyboardInterrupt:
        print("\nexam_auto 已中断，临时过程文件会在退出时清理。", file=sys.stderr)
        raise SystemExit(130)
    except RuntimeError as exc:
        message = str(exc)
        if "out of memory" in message.lower() or "cuda oom" in message.lower():
            print(f"\n[失败] 检测到 CUDA OOM 或显存不足：{message}", file=sys.stderr)
        else:
            print(f"\n[失败] 试运行失败：{message}", file=sys.stderr)
        raise SystemExit(1)
    except Exception as exc:
        print(f"\n[失败] 试运行失败：{exc}", file=sys.stderr)
        raise SystemExit(1)
