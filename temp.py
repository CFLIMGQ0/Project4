#!/usr/bin/env python3
from __future__ import annotations

import argparse
import os
import shutil
import time
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parent
DEFAULT_PATH_CONFIG = PROJECT_ROOT / "configs" / "path.yaml"
DEFAULT_TRAIN_CONFIG = PROJECT_ROOT / "configs" / "train.yaml"
DEFAULT_MODEL_CONFIG = PROJECT_ROOT / "configs" / "model.yaml"
DEFAULT_TEMP_NUM_WORKERS = 8
DEFAULT_TEMP_BATCH_SIZE = 12
DEFAULT_TEMP_EVAL_BATCH_SIZE = 12
DEFAULT_TEMP_TRAIN_MAX_INSTANCES = 32
DEFAULT_TEMP_EVAL_MAX_INSTANCES = 32
DEFAULT_TEMP_TRAIN_MAX_BATCH_INSTANCES = 384
DEFAULT_TEMP_EVAL_MAX_BATCH_INSTANCES = 384
SUPPORTED_MODELS = (
    "gastro_label_graph_mil",
    "gastro_baseline",
    "colonoscopy_baseline",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="快速缓存训练入口：只跑一次使用缓存的训练，并默认执行测试。")
    parser.add_argument("--config", type=Path, default=DEFAULT_PATH_CONFIG, help="路径配置文件")
    parser.add_argument("--train-config", type=Path, default=DEFAULT_TRAIN_CONFIG, help="训练配置文件")
    parser.add_argument("--model-config", type=Path, default=DEFAULT_MODEL_CONFIG, help="模型配置文件")
    parser.add_argument(
        "--model",
        type=str,
        default="gastro_label_graph_mil",
        choices=SUPPORTED_MODELS,
        help="本次快速训练要运行的模型",
    )
    parser.add_argument("--epochs", type=int, default=2, help="快速训练 epoch 数")
    parser.add_argument("--patience", type=int, default=2, help="快速训练早停 patience")
    parser.add_argument("--max-exams-per-task", type=int, default=96, help="每个任务最多样本数")
    parser.add_argument("--num-workers", type=int, default=DEFAULT_TEMP_NUM_WORKERS, help="DataLoader worker 数")
    parser.add_argument("--image-size", type=int, default=None, help="覆盖图像尺寸")
    parser.add_argument("--seed", type=int, default=None, help="覆盖随机种子")
    parser.add_argument("--batch-size", type=int, default=DEFAULT_TEMP_BATCH_SIZE, help="覆盖训练 batch size")
    parser.add_argument(
        "--eval-batch-size",
        type=int,
        default=DEFAULT_TEMP_EVAL_BATCH_SIZE,
        help="覆盖验证/测试 batch size",
    )
    parser.add_argument(
        "--train-max-instances",
        type=int,
        default=DEFAULT_TEMP_TRAIN_MAX_INSTANCES,
        help="覆盖训练每个 bag 最大实例数",
    )
    parser.add_argument(
        "--eval-max-instances",
        type=int,
        default=DEFAULT_TEMP_EVAL_MAX_INSTANCES,
        help="覆盖验证每个 bag 最大实例数",
    )
    parser.add_argument(
        "--train-max-batch-instances",
        type=int,
        default=DEFAULT_TEMP_TRAIN_MAX_BATCH_INSTANCES,
        help="覆盖训练每个 batch 最大实例总数",
    )
    parser.add_argument(
        "--eval-max-batch-instances",
        type=int,
        default=DEFAULT_TEMP_EVAL_MAX_BATCH_INSTANCES,
        help="覆盖验证每个 batch 最大实例总数",
    )
    parser.add_argument(
        "--cache-mode",
        type=str,
        default="disk",
        choices=("memory", "disk", "memory_and_disk"),
        help="本次训练使用的缓存模式",
    )
    parser.add_argument(
        "--disable-cache-warmup",
        action="store_true",
        help="关闭训练前缓存预热",
    )
    parser.add_argument(
        "--clear-task-cache",
        action="store_true",
        help="训练前删除当前任务缓存目录，便于重新观察缓存构建效果",
    )
    parser.add_argument("--no-pretrained", action="store_true", help="禁用预训练权重")
    parser.add_argument("--disable-multi-gpu", action="store_true", help="禁用多卡训练")
    return parser.parse_args()


def task_image_cache_dir_name(task_name: str) -> str:
    if task_name == "gastro_multilabel":
        return "cache_gastro_multilabel_image"
    if task_name == "colonoscopy_binary":
        return "colonoscopy_binary_image_cache"
    raise ValueError(f"未知 task_name: {task_name}")


def resolve_cache_root_dir(task_selection_dir: Path, raw_cache_dir: str) -> Path:
    candidate = Path(raw_cache_dir).expanduser()
    if candidate.is_absolute():
        return candidate.resolve()
    return (task_selection_dir / candidate).resolve()


def count_cache_files(cache_dir: Path) -> int:
    if not cache_dir.is_dir():
        return 0
    return sum(1 for _ in cache_dir.rglob("*.npy"))


def build_run_overrides(args: argparse.Namespace) -> dict[str, int | str | bool]:
    run_overrides: dict[str, int | str | bool] = {
        "image_cache_mode": str(args.cache_mode),
        "image_cache_warmup": not args.disable_cache_warmup,
    }
    if args.batch_size is not None:
        run_overrides["batch_size"] = int(args.batch_size)
    if args.eval_batch_size is not None:
        run_overrides["eval_batch_size"] = int(args.eval_batch_size)
    if args.train_max_instances is not None:
        run_overrides["train_max_instances"] = int(args.train_max_instances)
    if args.eval_max_instances is not None:
        run_overrides["eval_max_instances"] = int(args.eval_max_instances)
    if args.train_max_batch_instances is not None:
        run_overrides["train_max_batch_instances"] = int(args.train_max_batch_instances)
    if args.eval_max_batch_instances is not None:
        run_overrides["eval_max_batch_instances"] = int(args.eval_max_batch_instances)
    return run_overrides


def main() -> None:
    args = parse_args()

    import torch

    from train import (
        allocate_task_run_dir,
        build_training_config_summary_lines,
        format_task_display_name,
        load_model_config,
        load_path_config,
        load_train_config,
        prepare_training_context,
        resolve_model_task_meta,
        run_model_job,
    )

    path_cfg = load_path_config(args.config)
    train_cfg = load_train_config(args.train_config)
    model_cfg = load_model_config(args.model_config)

    train_cfg["auto_explore"] = False
    train_cfg["enabled_models"] = [args.model]

    seed = args.seed if args.seed is not None else int(train_cfg["seed"])
    image_size = args.image_size if args.image_size is not None else int(train_cfg["image_size"])
    max_epochs = max(1, int(args.epochs))
    patience = max(1, min(int(args.patience), max_epochs))
    num_workers = max(0, int(args.num_workers))
    max_exams_per_task = max(1, int(args.max_exams_per_task))

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

    task_meta = resolve_model_task_meta(args.model)
    task_name = str(task_meta["task_name"])
    run_overrides = build_run_overrides(args)
    effective_run_cfg = dict(train_cfg["default_run"])
    effective_run_cfg.update(run_overrides)
    print("=" * 72)
    print("训练配置")
    print(f"模型={args.model} 任务={format_task_display_name(task_name)}")
    print(f"epoch={max_epochs} patience={patience} seed={seed}")
    for line in build_training_config_summary_lines(
        effective_run_cfg,
        image_size=image_size,
        num_workers=num_workers,
    ):
        print(line)
    training_context = prepare_training_context(
        path_cfg=path_cfg,
        train_cfg=train_cfg,
        seed=seed,
        max_exams_per_task=max_exams_per_task,
        required_tasks={task_name},
    )
    split_data = (
        training_context["gastro_split"]
        if task_name == "gastro_multilabel"
        else training_context["colon_split"]
    )

    task_selection_dir = Path(str(training_context["task_selection_dir"])).resolve()
    raw_cache_dir = str(effective_run_cfg.get("image_cache_dir", "")).strip()
    task_cache_dir = resolve_cache_root_dir(task_selection_dir, raw_cache_dir) / task_image_cache_dir_name(task_name)

    if args.clear_task_cache and task_cache_dir.exists():
        print(f"清理当前任务缓存目录：{task_cache_dir}")
        shutil.rmtree(task_cache_dir)

    cache_files_before = count_cache_files(task_cache_dir)
    train_size = len(split_data["train"])
    val_size = len(split_data["val"])
    test_size = len(split_data["test"])

    run_dir, run_meta = allocate_task_run_dir(
        Path(training_context["output_root"]).resolve(),
        train_cfg,
        args.model,
        is_auto_explore=False,
    )
    del run_meta
    task_stats = training_context.get("task_stats", {}).get(task_name, {})
    print(
        f"{format_task_display_name(task_name)}样本 总样本={task_stats.get('total_records', train_size + val_size + test_size)} "
        f"train/val/test={train_size}/{val_size}/{test_size}"
    )
    print(f"缓存目录={task_cache_dir} 文件数={cache_files_before}")
    print(f"训练目录={run_dir}")
    print("=" * 72)

    started_at = time.perf_counter()
    result = run_model_job(
        model_name=args.model,
        run_dir=run_dir,
        train_cfg=train_cfg,
        model_cfg=model_cfg,
        training_context=training_context,
        seed=seed + 1,
        max_epochs=max_epochs,
        patience=patience,
        image_size=image_size,
        num_workers=num_workers,
        pretrained=pretrained,
        use_multi_gpu=use_multi_gpu,
        active_gpu_count=active_gpu_count,
        run_test=True,
        run_overrides=run_overrides,
    )
    elapsed_seconds = time.perf_counter() - started_at

    cache_files_after = count_cache_files(task_cache_dir)
    new_cache_files = cache_files_after - cache_files_before

    print(f"\n总耗时：{elapsed_seconds:.2f} 秒")
    print(f"缓存结果：文件数={cache_files_after} 新增={new_cache_files}")


if __name__ == "__main__":
    main()
