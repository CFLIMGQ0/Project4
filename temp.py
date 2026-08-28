#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import gc
import json
import os
import re
from datetime import datetime
from pathlib import Path
from typing import Any

import yaml

try:
    from tqdm import tqdm
except Exception:  # pragma: no cover
    tqdm = None


PROJECT_ROOT = Path(__file__).resolve().parent
DEFAULT_PATH_CONFIG = PROJECT_ROOT / "configs" / "task1" / "path.yaml"
DEFAULT_TRAIN_CONFIG = PROJECT_ROOT / "configs" / "task1" / "train.yaml"
DEFAULT_MODEL_CONFIG = PROJECT_ROOT / "configs" / "task1" / "model.yaml"
CHECKPOINT_ALIASES = ("best_macro_f1", "best_micro_f1", "best_val_loss")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="复测最新自动探索目录：对每个 train_xxx 评估 3 个最佳 checkpoint。"
    )
    parser.add_argument("--config", type=Path, default=DEFAULT_PATH_CONFIG, help="路径配置文件")
    parser.add_argument("--train-config", type=Path, default=DEFAULT_TRAIN_CONFIG, help="训练配置文件")
    parser.add_argument("--model-config", type=Path, default=DEFAULT_MODEL_CONFIG, help="模型配置文件")
    parser.add_argument(
        "--auto-dir",
        type=Path,
        default=None,
        help="指定自动探索目录；默认自动选择最新的 *_para_auto 目录",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="汇总输出目录；默认写入 paths.output_dir/temp",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="覆盖用于重建原始 train/val/test 切分的基础 seed",
    )
    parser.add_argument(
        "--max-exams-per-task",
        type=int,
        default=None,
        help="覆盖用于重建原始数据切分的 max_exams_per_task",
    )
    parser.add_argument(
        "--num-workers",
        type=int,
        default=None,
        help="覆盖评估阶段 DataLoader worker 数；默认使用每个 trial 保存的配置",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=0,
        help="仅评估前 N 个 train_xxx；默认 0 表示全部",
    )
    parser.add_argument(
        "--disable-multi-gpu",
        action="store_true",
        help="禁用多卡评估",
    )
    parser.add_argument(
        "--pretrained",
        action="store_true",
        help="初始化模型时使用预训练权重；评估已有 checkpoint 默认不需要",
    )
    parser.add_argument(
        "--fail-fast",
        action="store_true",
        help="遇到单个目录评估失败时立即停止",
    )
    return parser.parse_args()


def load_yaml_file(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(f"未找到 YAML 文件: {path}")
    payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"YAML 文件格式错误: {path}")
    return payload


def load_json_file(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(f"未找到 JSON 文件: {path}")
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"JSON 文件格式错误: {path}")
    return payload


def parse_generated_at(raw_value: Any, fallback_path: Path) -> datetime:
    text = str(raw_value or "").strip()
    if text:
        try:
            return datetime.fromisoformat(text)
        except Exception:
            pass
    return datetime.fromtimestamp(fallback_path.stat().st_mtime)


def discover_latest_auto_dir(output_root: Path) -> Path:
    candidates: list[tuple[datetime, Path]] = []
    for notes_path in output_root.rglob("notes.json"):
        session_dir = notes_path.parent
        if not session_dir.is_dir() or not session_dir.name.endswith("_para_auto"):
            continue
        try:
            notes_payload = load_json_file(notes_path)
        except Exception:
            continue
        candidates.append((parse_generated_at(notes_payload.get("generated_at"), notes_path), session_dir))

    if not candidates:
        raise FileNotFoundError(f"在 {output_root} 下未找到自动探索目录")
    candidates.sort(key=lambda item: (item[0], str(item[1])))
    return candidates[-1][1]


def extract_trial_index(name: str) -> int:
    match = re.fullmatch(r"train_(\d+)", name)
    if not match:
        raise ValueError(f"非法训练目录名: {name}")
    return int(match.group(1))


def list_trial_dirs(auto_dir: Path, limit: int = 0) -> list[Path]:
    trial_dirs = [path for path in auto_dir.iterdir() if path.is_dir() and re.fullmatch(r"train_\d+", path.name)]
    trial_dirs.sort(key=lambda path: extract_trial_index(path.name))
    if limit > 0:
        return trial_dirs[:limit]
    return trial_dirs


def resolve_summary_dir(path_cfg: dict[str, str], output_dir: Path | None) -> Path:
    if output_dir is not None:
        return output_dir.expanduser().resolve()
    return (Path(path_cfg["output_dir"]).resolve() / "temp").resolve()


def infer_base_split_seed(
    *,
    trial_cfg: dict[str, Any],
    trial_index: int,
    train_cfg: dict[str, Any],
    override_seed: int | None,
) -> int:
    if override_seed is not None:
        return int(override_seed)

    model_name = str(trial_cfg.get("model_name", "")).strip()
    trial_seed = int(trial_cfg.get("seed", train_cfg["seed"]))
    enabled_models = list(train_cfg.get("enabled_models", []))
    if model_name in enabled_models:
        model_offset = enabled_models.index(model_name) + 1
        inferred_seed = trial_seed - trial_index - model_offset * 1000
        return int(inferred_seed)
    return int(train_cfg["seed"])


def summarize_split_sizes(split_data: dict[str, list[dict[str, Any]]]) -> dict[str, int]:
    return {key: len(value) for key, value in split_data.items()}


def validate_rebuilt_split(
    trial_cfg: dict[str, Any],
    *,
    training_context: dict[str, Any],
) -> None:
    task_name = str(trial_cfg.get("task_name", "")).strip()
    expected = {str(key): int(value) for key, value in dict(trial_cfg.get("split_stats", {})).items()}
    if task_name == "gastro_multilabel":
        actual = summarize_split_sizes(training_context["gastro_split"])
    elif task_name == "colonoscopy_binary":
        actual = summarize_split_sizes(training_context["colon_split"])
    else:
        raise ValueError(f"未知 task_name: {task_name}")

    if expected != actual:
        raise RuntimeError(
            "重建出的数据切分与 trial 保存配置不一致。"
            f"\n期望 split_stats={expected}"
            f"\n实际 split_stats={actual}"
            "\n请检查当前 train-config 是否与自动探索运行时一致，"
            "或通过 --seed / --max-exams-per-task 指定正确参数。"
        )


def resolve_cache_dir(run_cfg: dict[str, Any], *, task_name: str, training_context: dict[str, Any]) -> Path | None:
    resolved_path = str(run_cfg.get("resolved_image_cache_dir", "")).strip()
    if resolved_path:
        return Path(resolved_path).expanduser().resolve()

    raw_cache_dir = str(run_cfg.get("image_cache_dir", "")).strip()
    if not raw_cache_dir:
        return None

    from train import task_image_cache_dir_name

    candidate = Path(raw_cache_dir).expanduser()
    if candidate.is_absolute():
        return (candidate.resolve() / task_image_cache_dir_name(task_name)).resolve()
    return (Path(training_context["task_selection_dir"]).resolve() / candidate / task_image_cache_dir_name(task_name)).resolve()


def metric_value(metrics: dict[str, Any], key: str) -> Any:
    if not isinstance(metrics, dict):
        return float("nan")
    return metrics.get(key, float("nan"))


def build_summary_rows(
    *,
    trial_dir: Path,
    trial_record: dict[str, Any],
    result: dict[str, Any],
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    test_results = result.get("test_results", {})
    for alias in CHECKPOINT_ALIASES:
        payload = test_results.get(alias, {})
        if not isinstance(payload, dict) or not payload:
            continue
        metrics = payload.get("metrics", {}) if isinstance(payload.get("metrics"), dict) else {}
        rows.append(
            {
                "train_dir": trial_dir.name,
                "trial_index": extract_trial_index(trial_dir.name),
                "trial_status": str(trial_record.get("status", "")),
                "trial_objective_score": trial_record.get("objective_score", float("nan")),
                "trial_best_score": trial_record.get("best_score", float("nan")),
                "trial_evaluation": str(trial_record.get("evaluation", "")),
                "stable_convergence": bool(trial_record.get("stable_convergence", False)),
                "checkpoint_alias": alias,
                "checkpoint_path": str(payload.get("checkpoint_path", "")),
                "best_epoch": int(payload.get("best_epoch", -1)),
                "selection_metric": str(payload.get("selection_metric", "")),
                "selection_value": payload.get("selection_value", float("nan")),
                "test_loss": payload.get("test_loss", float("nan")),
                "macro_f1": metric_value(metrics, "macro_f1"),
                "micro_f1": metric_value(metrics, "micro_f1"),
                "macro_auc": metric_value(metrics, "macro_auc"),
                "macro_ap": metric_value(metrics, "macro_ap"),
                "subset_accuracy": metric_value(metrics, "subset_accuracy"),
                "accuracy": metric_value(metrics, "accuracy"),
                "hamming_loss": metric_value(metrics, "hamming_loss"),
                "result_dir": str(payload.get("result_dir", "")),
                "params_json": json.dumps(trial_record.get("params", {}), ensure_ascii=False, sort_keys=True),
            }
        )
    return rows


def write_summary_csv(csv_path: Path, rows: list[dict[str, Any]]) -> None:
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        csv_path.write_text("", encoding="utf-8")
        return

    fieldnames: list[str] = []
    for row in rows:
        for key in row.keys():
            if key not in fieldnames:
                fieldnames.append(key)

    with csv_path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in fieldnames})


def sort_rows_desc(rows: list[dict[str, Any]], key: str) -> list[dict[str, Any]]:
    valid_rows = [row for row in rows if isinstance(row.get(key), (int, float))]
    return sorted(valid_rows, key=lambda item: float(item.get(key, float("nan"))), reverse=True)


def build_text_summary(
    *,
    auto_dir: Path,
    split_seed: int,
    max_exams_per_task: int,
    rows: list[dict[str, Any]],
    failures: list[dict[str, str]],
    notes_payload: dict[str, Any],
) -> str:
    lines = [
        "自动探索批量复测摘要",
        f"自动探索目录: {auto_dir}",
        f"使用切分 seed: {split_seed}",
        f"使用 max_exams_per_task: {max_exams_per_task}",
        (
            "自动探索完成情况: "
            f"configured={notes_payload.get('counts', {}).get('configured_trials', 0)}, "
            f"completed={notes_payload.get('counts', {}).get('completed_trials', 0)}, "
            f"successful={notes_payload.get('counts', {}).get('successful_trials', 0)}, "
            f"pruned={notes_payload.get('counts', {}).get('pruned_trials', 0)}"
        ),
        f"本次完成测试目录数: {len({row['train_dir'] for row in rows})}",
        f"失败目录数: {len(failures)}",
    ]

    macro_top = sort_rows_desc(rows, "macro_f1")[:10]
    micro_top = sort_rows_desc(rows, "micro_f1")[:10]

    if macro_top:
        lines.append("按测试 macro_f1 排名前 10 的 checkpoint:")
        for item in macro_top:
            lines.append(
                f"- {item['train_dir']} / {item['checkpoint_alias']} | "
                f"macro_f1={float(item['macro_f1']):.4f} | micro_f1={float(item['micro_f1']):.4f} | "
                f"test_loss={float(item['test_loss']):.4f}"
            )

    if micro_top:
        lines.append("按测试 micro_f1 排名前 10 的 checkpoint:")
        for item in micro_top:
            lines.append(
                f"- {item['train_dir']} / {item['checkpoint_alias']} | "
                f"micro_f1={float(item['micro_f1']):.4f} | macro_f1={float(item['macro_f1']):.4f} | "
                f"test_loss={float(item['test_loss']):.4f}"
            )

    if failures:
        lines.append("失败目录:")
        for item in failures:
            lines.append(f"- {item['train_dir']} | {item['error_message']}")

    return "\n".join(lines) + "\n"


def build_trial_record_map(notes_payload: dict[str, Any]) -> dict[str, dict[str, Any]]:
    mapping: dict[str, dict[str, Any]] = {}
    for item in notes_payload.get("trial_records", []):
        if not isinstance(item, dict):
            continue
        train_dir = str(item.get("train_dir", "")).strip()
        if not train_dir:
            continue
        mapping[train_dir] = item
    return mapping


def prepare_runtime(train_cfg: dict[str, Any], *, disable_multi_gpu: bool) -> tuple[bool, int]:
    import torch

    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = ",".join(str(item) for item in train_cfg["gpu_ids"])

    torch.set_float32_matmul_precision("medium")
    if torch.cuda.is_available():
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.benchmark = True

    visible_gpu_count = torch.cuda.device_count() if torch.cuda.is_available() else 0
    use_multi_gpu = (not disable_multi_gpu) and visible_gpu_count > 1
    active_gpu_count = visible_gpu_count if use_multi_gpu else (1 if visible_gpu_count > 0 else 0)
    return use_multi_gpu, active_gpu_count


def build_trainer_for_trial(
    *,
    trial_dir: Path,
    trial_cfg: dict[str, Any],
    train_cfg: dict[str, Any],
    training_context: dict[str, Any],
    use_multi_gpu: bool,
    active_gpu_count: int,
    pretrained: bool,
    num_workers_override: int | None,
):
    import torch

    from train import (
        build_loaders,
        build_model_bundle,
        normalize_batch_size,
        seed_everything,
    )
    from training.trainer import Trainer

    model_name = str(trial_cfg["model_name"])
    task_name = str(trial_cfg["task_name"])
    trial_seed = int(trial_cfg["seed"])
    image_size = int(trial_cfg["image_size"])
    num_workers = int(num_workers_override) if num_workers_override is not None else int(trial_cfg["num_workers"])
    trainer_saved = dict(trial_cfg.get("trainer", {}))
    run_cfg = dict(trial_cfg.get("run", {}))
    model_param_cfg = dict(trial_cfg.get("model_params", {}))

    if task_name == "gastro_multilabel":
        split_data = training_context["gastro_split"]
    elif task_name == "colonoscopy_binary":
        split_data = training_context["colon_split"]
    else:
        raise ValueError(f"未知 task_name: {task_name}")

    pos_weight = trainer_saved.get("pos_weight")
    seed_everything(trial_seed)
    model, trainer_cfg, resolved_task_name, label_names, class_names = build_model_bundle(
        model_name=model_name,
        run_cfg=run_cfg,
        model_param_cfg=model_param_cfg,
        pretrained=pretrained,
        max_epochs=0,
        patience=0,
        pos_weight=pos_weight,
        use_multi_gpu=use_multi_gpu,
        run_test=True,
    )
    if resolved_task_name != task_name:
        raise RuntimeError(f"task_name 不一致: config={task_name}, bundle={resolved_task_name}")

    last_ckpt = trial_dir / "checkpoints" / "last.ckpt"
    if not last_ckpt.is_file():
        raise FileNotFoundError(f"未找到 last.ckpt: {last_ckpt}")

    trainer_cfg.max_epochs = 0
    trainer_cfg.patience = 0
    trainer_cfg.resume_path = str(last_ckpt)
    trainer_cfg.run_test = True

    cache_dir = resolve_cache_dir(run_cfg, task_name=task_name, training_context=training_context)
    train_batch_size = normalize_batch_size(int(run_cfg.get("batch_size", 3)), active_gpu_count)
    eval_batch_size = normalize_batch_size(int(run_cfg.get("eval_batch_size", 3)), active_gpu_count)
    train_loader, val_loader, test_loader = build_loaders(
        split_data=split_data,
        task_name=task_name,
        image_size=image_size,
        num_workers=num_workers,
        train_batch_size=train_batch_size,
        eval_batch_size=eval_batch_size,
        train_max_instances=int(run_cfg.get("train_max_instances", 32)),
        eval_max_instances=int(run_cfg.get("eval_max_instances", 32)),
        min_instances=int(train_cfg["min_instances"]),
        train_sampling=str(train_cfg["train_sampling_strategy"]),
        eval_sampling=str(train_cfg["eval_sampling_strategy"]),
        random_instance_dropout=float(run_cfg.get("random_instance_dropout", 0.0)),
        train_max_batch_instances=int(run_cfg.get("train_max_batch_instances", 96)),
        eval_max_batch_instances=int(run_cfg.get("eval_max_batch_instances", 96)),
        seed=trial_seed,
        pin_memory=bool(run_cfg.get("pin_memory", True)),
        persistent_workers=bool(run_cfg.get("persistent_workers", True)),
        loader_prefetch_factor=int(run_cfg.get("loader_prefetch_factor", 2)),
        image_cache_mode=str(run_cfg.get("image_cache_mode", "none")),
        image_cache_dir=cache_dir,
        image_cache_manifest=str(run_cfg.get("image_cache_manifest", "")).strip() or None,
        legacy_image_cache_dirs=[],
        image_cache_warmup=False,
        memory_cache_size=int(run_cfg.get("memory_cache_size", 0)),
    )

    trainer = Trainer(
        model=model,
        cfg=trainer_cfg,
        run_dir=trial_dir,
        train_loader=train_loader,
        val_loader=val_loader,
        test_loader=test_loader,
        label_names=label_names,
        class_names=class_names,
        seed=trial_seed,
    )
    return trainer


def evaluate_trial_dir(
    *,
    trial_dir: Path,
    trial_cfg: dict[str, Any],
    trial_record: dict[str, Any],
    train_cfg: dict[str, Any],
    training_context: dict[str, Any],
    use_multi_gpu: bool,
    active_gpu_count: int,
    pretrained: bool,
    num_workers_override: int | None,
) -> dict[str, Any]:
    import torch

    trainer = build_trainer_for_trial(
        trial_dir=trial_dir,
        trial_cfg=trial_cfg,
        train_cfg=train_cfg,
        training_context=training_context,
        use_multi_gpu=use_multi_gpu,
        active_gpu_count=active_gpu_count,
        pretrained=pretrained,
        num_workers_override=num_workers_override,
    )
    result = trainer.fit()
    result["model_name"] = str(trial_cfg["model_name"])
    result["train_dir"] = str(trial_dir)
    result["train_dir_name"] = trial_dir.name
    del trial_record

    del trainer
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return result


def main() -> None:
    args = parse_args()

    try:
        from train import load_model_config, load_path_config, load_train_config, prepare_training_context
    except ModuleNotFoundError as exc:
        if exc.name == "torch":
            raise SystemExit(
                "当前 Python 环境缺少 torch，无法执行复测。"
                "\n请使用已安装训练依赖的解释器，例如："
                "\n/home/Lim/anaconda3/envs/myenv/bin/python temp.py"
            ) from exc
        raise

    path_cfg = load_path_config(args.config)
    train_cfg = load_train_config(args.train_config)
    _ = load_model_config(args.model_config)

    auto_dir = args.auto_dir.expanduser().resolve() if args.auto_dir is not None else discover_latest_auto_dir(Path(path_cfg["output_dir"]).resolve())
    if not auto_dir.is_dir():
        raise FileNotFoundError(f"自动探索目录不存在: {auto_dir}")

    notes_path = auto_dir / "notes.json"
    notes_payload = load_json_file(notes_path)
    trial_dirs = list_trial_dirs(auto_dir, limit=max(0, int(args.limit)))
    if not trial_dirs:
        raise FileNotFoundError(f"在 {auto_dir} 下未找到 train_xxx 目录")

    trial_cfg_map = {trial_dir.name: load_yaml_file(trial_dir / "config.yaml") for trial_dir in trial_dirs}
    first_trial_dir = trial_dirs[0]
    first_trial_cfg = trial_cfg_map[first_trial_dir.name]

    split_seed = infer_base_split_seed(
        trial_cfg=first_trial_cfg,
        trial_index=extract_trial_index(first_trial_dir.name),
        train_cfg=train_cfg,
        override_seed=args.seed,
    )
    max_exams_per_task = (
        int(args.max_exams_per_task)
        if args.max_exams_per_task is not None
        else int(train_cfg["max_exams_per_task"])
    )
    required_tasks = {str(cfg["task_name"]) for cfg in trial_cfg_map.values()}
    training_context = prepare_training_context(
        path_cfg=path_cfg,
        train_cfg=train_cfg,
        seed=split_seed,
        max_exams_per_task=max_exams_per_task,
        required_tasks=required_tasks,
    )
    validate_rebuilt_split(first_trial_cfg, training_context=training_context)

    summary_dir = resolve_summary_dir(path_cfg, args.output_dir)
    summary_dir.mkdir(parents=True, exist_ok=True)
    summary_csv_path = summary_dir / f"{auto_dir.name}_test_summary.csv"
    summary_json_path = summary_dir / f"{auto_dir.name}_test_summary.json"
    summary_txt_path = summary_dir / f"{auto_dir.name}_test_summary.txt"

    use_multi_gpu, active_gpu_count = prepare_runtime(
        train_cfg,
        disable_multi_gpu=bool(args.disable_multi_gpu),
    )

    print("=" * 72)
    print("自动探索批量复测")
    print(f"自动探索目录: {auto_dir}")
    print(f"trial 数量: {len(trial_dirs)}")
    print(f"使用切分 seed: {split_seed}")
    print(f"使用 max_exams_per_task: {max_exams_per_task}")
    print(f"多卡评估: {use_multi_gpu}")
    print(f"汇总目录: {summary_dir}")
    print("=" * 72)

    trial_record_map = build_trial_record_map(notes_payload)
    rows: list[dict[str, Any]] = []
    detailed_results: dict[str, Any] = {
        "auto_dir": str(auto_dir),
        "notes_path": str(notes_path),
        "summary_generated_at": datetime.now().isoformat(timespec="seconds"),
        "split_seed": split_seed,
        "max_exams_per_task": max_exams_per_task,
        "trial_count": len(trial_dirs),
        "results": {},
        "failures": [],
    }

    iterator = trial_dirs
    progress = None
    if tqdm is not None:
        progress = tqdm(trial_dirs, total=len(trial_dirs), desc=f"{auto_dir.name}-test", dynamic_ncols=True)
        iterator = progress

    for trial_dir in iterator:
        trial_name = trial_dir.name
        trial_cfg = trial_cfg_map[trial_name]
        trial_record = trial_record_map.get(trial_name, {})

        if progress is not None:
            progress.set_postfix_str(trial_name, refresh=False)
        print(f"\n[{trial_name}] 开始复测 3 个最佳 checkpoint")

        try:
            result = evaluate_trial_dir(
                trial_dir=trial_dir,
                trial_cfg=trial_cfg,
                trial_record=trial_record,
                train_cfg=train_cfg,
                training_context=training_context,
                use_multi_gpu=use_multi_gpu,
                active_gpu_count=active_gpu_count,
                pretrained=bool(args.pretrained),
                num_workers_override=args.num_workers,
            )
            trial_rows = build_summary_rows(
                trial_dir=trial_dir,
                trial_record=trial_record,
                result=result,
            )
            rows.extend(trial_rows)
            detailed_results["results"][trial_name] = {
                "status": "success",
                "trial_record": trial_record,
                "result": result,
            }
        except Exception as exc:
            failure_payload = {
                "train_dir": trial_name,
                "error_message": str(exc),
            }
            detailed_results["failures"].append(failure_payload)
            print(f"[{trial_name}] 复测失败: {exc}")
            if args.fail_fast:
                raise

    if progress is not None:
        progress.close()

    write_summary_csv(summary_csv_path, rows)
    summary_txt = build_text_summary(
        auto_dir=auto_dir,
        split_seed=split_seed,
        max_exams_per_task=max_exams_per_task,
        rows=rows,
        failures=list(detailed_results["failures"]),
        notes_payload=notes_payload,
    )
    summary_txt_path.write_text(summary_txt, encoding="utf-8")
    summary_json_path.write_text(
        json.dumps(detailed_results, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    print("\n复测完成。")
    print(f"CSV 汇总: {summary_csv_path}")
    print(f"TXT 汇总: {summary_txt_path}")
    print(f"JSON 汇总: {summary_json_path}")
    if detailed_results["failures"]:
        print(f"失败目录数: {len(detailed_results['failures'])}")
        raise SystemExit(1)


if __name__ == "__main__":
    main()
