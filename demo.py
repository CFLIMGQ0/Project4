#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import sys
from dataclasses import asdict
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import torch
import yaml
from torch.utils.data import DataLoader

# 尽量避免在源码目录下产生 pyc 文件。
sys.dont_write_bytecode = True
os.environ.setdefault("PYTHONDONTWRITEBYTECODE", "1")

from model import (
    DemoColoCountAwareDebiasMIL,
    DemoColoMILBaseline,
    DemoGastroMILBaseline,
    DemoGastroProtoMoEFormer,
)
from model.demo_data import (
    COLO_BINARY_CLASS_NAMES,
    GASTRO_LABEL_NAMES,
    DemoMILBagDataset,
    build_task_records,
    demo_mil_collate_fn,
    split_records,
)
from model.demo_trainer import DemoTrainer, TrainerConfig


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="一键运行 4 个 MIL 模型 demo")
    parser.add_argument("--config", type=str, default="configs/path.yaml", help="路径配置文件")
    parser.add_argument("--seed", type=int, default=42, help="随机种子")
    parser.add_argument("--epochs", type=int, default=3, help="每个模型训练轮数")
    parser.add_argument("--patience", type=int, default=2, help="早停耐心轮数")
    parser.add_argument("--image-size", type=int, default=224, help="输入图像尺寸")
    parser.add_argument("--num-workers", type=int, default=2, help="DataLoader 进程数（默认 2，降低 OOM 风险）")
    parser.add_argument("--max-exams-per-task", type=int, default=0, help="每个任务最多样本数，0 表示不限制")
    parser.add_argument("--no-pretrained", action="store_true", help="禁用 ImageNet 预训练")
    parser.add_argument("--disable-multi-gpu", action="store_true", help="禁用 DataParallel")
    return parser.parse_args()


def load_path_config(config_path: Path) -> dict[str, str]:
    if not config_path.is_file():
        raise FileNotFoundError(f"未找到配置文件: {config_path}")
    payload = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict) or "paths" not in payload:
        raise ValueError("配置文件缺少 paths 字段")
    paths = payload["paths"]
    if not isinstance(paths, dict):
        raise ValueError("paths 字段格式错误")

    required = ["valid_dicts_report_csv", "output_dir"]
    for k in required:
        if k not in paths:
            raise ValueError(f"paths 缺少字段: {k}")

    return {
        "valid_dicts_report_csv": str(paths["valid_dicts_report_csv"]),
        "output_dir": str(paths["output_dir"]),
    }


def maybe_limit_records(records: list[dict[str, Any]], max_num: int, seed: int) -> list[dict[str, Any]]:
    if max_num <= 0 or len(records) <= max_num:
        return records
    rng = np.random.default_rng(seed)
    idx = np.arange(len(records))
    rng.shuffle(idx)
    keep = idx[:max_num]
    return [records[int(i)] for i in keep]


def compute_multilabel_pos_weight(train_records: list[dict[str, Any]]) -> list[float]:
    y = np.array([r["labels"] for r in train_records], dtype=np.float32)
    pos = y.sum(axis=0)
    neg = y.shape[0] - pos
    pw = (neg + 1.0) / (pos + 1.0)
    return pw.astype(np.float32).tolist()


def compute_binary_pos_weight(train_records: list[dict[str, Any]]) -> list[float]:
    y = np.array([r["label"] for r in train_records], dtype=np.int64)
    pos = float((y == 1).sum())
    neg = float((y == 0).sum())
    pw = (neg + 1.0) / (pos + 1.0)
    return [float(pw)]


def build_loaders(
    split_data: dict[str, list[dict[str, Any]]],
    task_name: str,
    image_size: int,
    num_workers: int,
    train_batch_size: int,
    eval_batch_size: int,
    train_max_instances: int,
    val_max_instances: int,
    test_max_instances: int,
    min_instances: int,
    train_sampling: str,
    eval_sampling: str,
    random_instance_dropout: float,
) -> tuple[DataLoader, DataLoader, DataLoader]:
    pin_memory = torch.cuda.is_available()

    train_ds = DemoMILBagDataset(
        records=split_data["train"],
        task=task_name,
        max_instances=train_max_instances,
        min_instances=min_instances,
        bag_sampling_strategy=train_sampling,
        is_train=True,
        image_size=image_size,
        random_instance_dropout=random_instance_dropout,
    )
    val_ds = DemoMILBagDataset(
        records=split_data["val"],
        task=task_name,
        max_instances=val_max_instances,
        min_instances=min_instances,
        bag_sampling_strategy=eval_sampling,
        is_train=False,
        image_size=image_size,
        random_instance_dropout=0.0,
    )
    test_ds = DemoMILBagDataset(
        records=split_data["test"],
        task=task_name,
        max_instances=test_max_instances,
        min_instances=min_instances,
        bag_sampling_strategy=eval_sampling,
        is_train=False,
        image_size=image_size,
        random_instance_dropout=0.0,
    )

    loader_kwargs = {
        "num_workers": num_workers,
        "pin_memory": pin_memory,
        "collate_fn": demo_mil_collate_fn,
        "persistent_workers": num_workers > 0,
    }

    train_loader = DataLoader(train_ds, batch_size=train_batch_size, shuffle=True, drop_last=False, **loader_kwargs)
    val_loader = DataLoader(val_ds, batch_size=eval_batch_size, shuffle=False, drop_last=False, **loader_kwargs)
    test_loader = DataLoader(test_ds, batch_size=eval_batch_size, shuffle=False, drop_last=False, **loader_kwargs)
    return train_loader, val_loader, test_loader


def run_single_model(
    model_name: str,
    model,
    trainer_cfg: TrainerConfig,
    split_data: dict[str, list[dict[str, Any]]],
    task_name: str,
    image_size: int,
    num_workers: int,
    run_dir: Path,
    seed: int,
    dl_cfg: dict[str, Any],
    label_names: list[str],
    class_names: list[str],
) -> dict[str, Any]:
    train_loader, val_loader, test_loader = build_loaders(
        split_data=split_data,
        task_name=task_name,
        image_size=image_size,
        num_workers=num_workers,
        train_batch_size=dl_cfg["train_batch_size"],
        eval_batch_size=dl_cfg["eval_batch_size"],
        train_max_instances=dl_cfg["train_max_instances"],
        val_max_instances=dl_cfg["val_max_instances"],
        test_max_instances=dl_cfg["test_max_instances"],
        min_instances=dl_cfg["min_instances"],
        train_sampling=dl_cfg["train_sampling"],
        eval_sampling=dl_cfg["eval_sampling"],
        random_instance_dropout=dl_cfg["random_instance_dropout"],
    )

    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "run_config.json").write_text(
        json.dumps(
            {
                "model_name": model_name,
                "trainer_cfg": asdict(trainer_cfg),
                "dataloader_cfg": dl_cfg,
                "split_stats": {k: len(v) for k, v in split_data.items()},
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )

    trainer = DemoTrainer(
        model=model,
        cfg=trainer_cfg,
        run_dir=run_dir,
        train_loader=train_loader,
        val_loader=val_loader,
        test_loader=test_loader,
        label_names=label_names,
        class_names=class_names,
        seed=seed,
    )
    result = trainer.fit()
    return result


def main() -> None:
    args = parse_args()
    torch.set_float32_matmul_precision("medium")

    cfg = load_path_config(Path(args.config))
    report_csv = Path(cfg["valid_dicts_report_csv"])
    output_root = Path(cfg["output_dir"]) / "demo"
    output_root.mkdir(parents=True, exist_ok=True)

    print("=" * 80)
    print("开始构建 demo 数据")
    print(f"报告 CSV: {report_csv}")
    print(f"输出目录: {output_root}")
    print("=" * 80)

    gastro_records, colo_records = build_task_records(report_csv_path=report_csv, min_instances=1)
    gastro_records = maybe_limit_records(gastro_records, args.max_exams_per_task, args.seed)
    colo_records = maybe_limit_records(colo_records, args.max_exams_per_task, args.seed)

    if len(gastro_records) < 10:
        raise RuntimeError("胃镜可用样本过少，无法训练")
    if len(colo_records) < 10:
        raise RuntimeError("肠镜可用样本过少，无法训练")

    gastro_split = split_records(gastro_records, seed=args.seed, ratios=(0.6, 0.2, 0.2))
    colo_split = split_records(colo_records, seed=args.seed, ratios=(0.6, 0.2, 0.2))

    print(f"胃镜样本数: train={len(gastro_split['train'])}, val={len(gastro_split['val'])}, test={len(gastro_split['test'])}")
    print(f"肠镜样本数: train={len(colo_split['train'])}, val={len(colo_split['val'])}, test={len(colo_split['test'])}")

    session_dir = output_root / datetime.now().strftime("run_%Y%m%d_%H%M%S")
    session_dir.mkdir(parents=True, exist_ok=True)

    pretrained = not args.no_pretrained
    use_multi_gpu = (not args.disable_multi_gpu)

    gastro_pos_weight = compute_multilabel_pos_weight(gastro_split["train"])
    colo_pos_weight = compute_binary_pos_weight(colo_split["train"])

    all_results: dict[str, Any] = {
        "session_dir": str(session_dir),
        "gastro_split": {k: len(v) for k, v in gastro_split.items()},
        "colo_split": {k: len(v) for k, v in colo_split.items()},
        "models": {},
    }

    # 1) 胃镜 baseline
    print("\n[1/4] 训练 demo_gastro_mil_baseline")
    model_1 = DemoGastroMILBaseline(
        backbone_name="resnet50",
        pretrained=pretrained,
        freeze_stages=1,
        feature_dim=512,
        attn_dim=256,
        num_labels=3,
        dropout=0.2,
    )
    trainer_cfg_1 = TrainerConfig(
        task_type="gastro_multilabel",
        model_family="gastro_baseline",
        num_classes=3,
        num_labels=3,
        max_epochs=args.epochs,
        patience=args.patience,
        lr=2e-4,
        weight_decay=1e-4,
        warmup_ratio=0.1,
        grad_accum_steps=2,
        amp=True,
        monitor_metric="macro_auc",
        monitor_mode="max",
        topk_evidence=5,
        loss_name="asymmetric",
        pos_weight=gastro_pos_weight,
        aux_loss_weights={},
        use_multi_gpu=use_multi_gpu,
        resume_path=None,
    )
    dl_cfg_1 = {
        "train_batch_size": 1,
        "eval_batch_size": 1,
        "train_max_instances": 32,
        "val_max_instances": 48,
        "test_max_instances": 48,
        "min_instances": 8,
        "train_sampling": "random",
        "eval_sampling": "uniform",
        "random_instance_dropout": 0.05,
    }
    result_1 = run_single_model(
        model_name="demo_gastro_mil_baseline",
        model=model_1,
        trainer_cfg=trainer_cfg_1,
        split_data=gastro_split,
        task_name="gastro_multilabel",
        image_size=args.image_size,
        num_workers=args.num_workers,
        run_dir=session_dir / "01_demo_gastro_mil_baseline",
        seed=args.seed,
        dl_cfg=dl_cfg_1,
        label_names=GASTRO_LABEL_NAMES,
        class_names=[],
    )
    all_results["models"]["demo_gastro_mil_baseline"] = result_1

    # 2) 胃镜 advanced
    print("\n[2/4] 训练 demo_gastro_proto_moe_former")
    model_2 = DemoGastroProtoMoEFormer(
        backbone_name="resnet50",
        pretrained=pretrained,
        freeze_stages=1,
        feature_dim=512,
        attn_dim=256,
        num_labels=3,
        num_experts=4,
        proto_per_label=8,
        relation_type="transformer",
        relation_layers=2,
        dropout=0.2,
    )
    trainer_cfg_2 = TrainerConfig(
        task_type="gastro_multilabel",
        model_family="gastro_advanced",
        num_classes=3,
        num_labels=3,
        max_epochs=args.epochs,
        patience=args.patience,
        lr=1.5e-4,
        weight_decay=1e-4,
        warmup_ratio=0.1,
        grad_accum_steps=4,
        amp=True,
        monitor_metric="macro_auc",
        monitor_mode="max",
        topk_evidence=5,
        loss_name="asymmetric",
        pos_weight=gastro_pos_weight,
        aux_loss_weights={
            "proto": 0.3,
            "consistency": 0.2,
            "expert_balance": 0.05,
        },
        use_multi_gpu=use_multi_gpu,
        resume_path=None,
    )
    dl_cfg_2 = {
        "train_batch_size": 1,
        "eval_batch_size": 1,
        "train_max_instances": 24,
        "val_max_instances": 36,
        "test_max_instances": 36,
        "min_instances": 8,
        "train_sampling": "random",
        "eval_sampling": "uniform",
        "random_instance_dropout": 0.08,
    }
    result_2 = run_single_model(
        model_name="demo_gastro_proto_moe_former",
        model=model_2,
        trainer_cfg=trainer_cfg_2,
        split_data=gastro_split,
        task_name="gastro_multilabel",
        image_size=args.image_size,
        num_workers=args.num_workers,
        run_dir=session_dir / "02_demo_gastro_proto_moe_former",
        seed=args.seed,
        dl_cfg=dl_cfg_2,
        label_names=GASTRO_LABEL_NAMES,
        class_names=[],
    )
    all_results["models"]["demo_gastro_proto_moe_former"] = result_2

    # 3) 肠镜 baseline
    print("\n[3/4] 训练 demo_colo_mil_baseline")
    model_3 = DemoColoMILBaseline(
        backbone_name="resnet50",
        pretrained=pretrained,
        freeze_stages=1,
        feature_dim=512,
        attn_dim=256,
        num_classes=2,
        dropout=0.2,
    )
    trainer_cfg_3 = TrainerConfig(
        task_type="colo_binary",
        model_family="colo_baseline",
        num_classes=2,
        num_labels=1,
        max_epochs=args.epochs,
        patience=args.patience,
        lr=2e-4,
        weight_decay=1e-4,
        warmup_ratio=0.1,
        grad_accum_steps=2,
        amp=True,
        monitor_metric="auc",
        monitor_mode="max",
        topk_evidence=5,
        loss_name="focal",
        pos_weight=colo_pos_weight,
        aux_loss_weights={},
        use_multi_gpu=use_multi_gpu,
        resume_path=None,
    )
    dl_cfg_3 = {
        "train_batch_size": 1,
        "eval_batch_size": 1,
        "train_max_instances": 32,
        "val_max_instances": 48,
        "test_max_instances": 48,
        "min_instances": 8,
        "train_sampling": "random",
        "eval_sampling": "uniform",
        "random_instance_dropout": 0.03,
    }
    result_3 = run_single_model(
        model_name="demo_colo_mil_baseline",
        model=model_3,
        trainer_cfg=trainer_cfg_3,
        split_data=colo_split,
        task_name="colo_binary",
        image_size=args.image_size,
        num_workers=args.num_workers,
        run_dir=session_dir / "03_demo_colo_mil_baseline",
        seed=args.seed,
        dl_cfg=dl_cfg_3,
        label_names=[],
        class_names=COLO_BINARY_CLASS_NAMES,
    )
    all_results["models"]["demo_colo_mil_baseline"] = result_3

    # 4) 肠镜 advanced
    print("\n[4/4] 训练 demo_colo_count_aware_debias_mil")
    model_4 = DemoColoCountAwareDebiasMIL(
        backbone_name="resnet50",
        pretrained=pretrained,
        freeze_stages=1,
        feature_dim=512,
        topk_lesion=8,
        topk_context=8,
        prototype_k=8,
        binary_num_classes=2,
        dropout=0.2,
    )
    trainer_cfg_4 = TrainerConfig(
        task_type="colo_binary",
        model_family="colo_advanced",
        num_classes=2,
        num_labels=1,
        max_epochs=args.epochs,
        patience=args.patience,
        lr=1.5e-4,
        weight_decay=1e-4,
        warmup_ratio=0.1,
        grad_accum_steps=4,
        amp=True,
        monitor_metric="auc",
        monitor_mode="max",
        topk_evidence=5,
        loss_name="bce",
        pos_weight=colo_pos_weight,
        aux_loss_weights={
            "count": 0.2,
            "proto": 0.25,
            "hard_negative": 0.2,
            "consistency": 0.1,
        },
        use_multi_gpu=use_multi_gpu,
        resume_path=None,
    )
    dl_cfg_4 = {
        "train_batch_size": 1,
        "eval_batch_size": 1,
        "train_max_instances": 24,
        "val_max_instances": 36,
        "test_max_instances": 36,
        "min_instances": 8,
        "train_sampling": "random",
        "eval_sampling": "uniform",
        "random_instance_dropout": 0.08,
    }
    result_4 = run_single_model(
        model_name="demo_colo_count_aware_debias_mil",
        model=model_4,
        trainer_cfg=trainer_cfg_4,
        split_data=colo_split,
        task_name="colo_binary",
        image_size=args.image_size,
        num_workers=args.num_workers,
        run_dir=session_dir / "04_demo_colo_count_aware_debias_mil",
        seed=args.seed,
        dl_cfg=dl_cfg_4,
        label_names=[],
        class_names=COLO_BINARY_CLASS_NAMES,
    )
    all_results["models"]["demo_colo_count_aware_debias_mil"] = result_4

    summary_path = session_dir / "all_models_summary.json"
    summary_path.write_text(json.dumps(all_results, ensure_ascii=False, indent=2), encoding="utf-8")

    print("\n" + "=" * 80)
    print("4 个模型已完成训练/验证/测试")
    print(f"总输出目录: {session_dir}")
    print(f"汇总文件: {summary_path}")
    print("=" * 80)


if __name__ == "__main__":
    main()
