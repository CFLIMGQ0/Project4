#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import os
import random
import sys
from dataclasses import asdict
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import torch
import yaml
from torch.utils.data import DataLoader, Sampler

# 尽量避免在源码目录下产生 pyc 文件。
sys.dont_write_bytecode = True
os.environ.setdefault("PYTHONDONTWRITEBYTECODE", "1")

from models import (
    DemoColoCountAwareDebiasMIL,
    DemoColoMILBaseline,
    DemoGastroMILBaseline,
    DemoGastroProtoMoEFormer,
)
from models.demo_data import (
    COLO_BINARY_CLASS_NAMES,
    GASTRO_LABEL_NAMES,
    DemoMILBagDataset,
    build_task_records,
    demo_mil_collate_fn,
    split_records,
)
from models.demo_trainer import DemoTrainer, TrainerConfig


MODEL_KEYS = (
    "demo_gastro_mil_baseline",
    "demo_gastro_proto_moe_former",
    "demo_colo_mil_baseline",
    "demo_colo_count_aware_debias_mil",
)


class InstanceAwareBatchSampler(Sampler[list[int]]):
    """按实例总量限制批次，避免检查目录图像数差异大导致内存峰值。"""

    def __init__(
        self,
        records: list[dict[str, Any]],
        max_instances_per_bag: int,
        min_instances_per_bag: int,
        batch_size: int,
        max_instances_per_batch: int,
        shuffle: bool,
        seed: int,
    ) -> None:
        self.records = records
        self.max_instances_per_bag = max(1, int(max_instances_per_bag))
        self.min_instances_per_bag = max(1, int(min_instances_per_bag))
        self.batch_size = max(1, int(batch_size))
        self.max_instances_per_batch = max(1, int(max_instances_per_batch))
        self.shuffle = shuffle
        self.seed = int(seed)
        self._iter_count = 0

        self.instance_counts: list[int] = []
        for record in self.records:
            n = len(record.get("image_paths", []))
            n = max(1, n)
            n = min(n, self.max_instances_per_bag)
            n = max(n, self.min_instances_per_bag)
            self.instance_counts.append(n)

    def __iter__(self):
        indices = list(range(len(self.records)))
        if self.shuffle:
            rng = random.Random(self.seed + self._iter_count)
            rng.shuffle(indices)
        self._iter_count += 1

        batch: list[int] = []
        batch_instances = 0

        for idx in indices:
            n_inst = self.instance_counts[idx]

            need_flush = False
            if len(batch) >= self.batch_size:
                need_flush = True
            elif batch and (batch_instances + n_inst > self.max_instances_per_batch):
                need_flush = True

            if need_flush:
                yield batch
                batch = []
                batch_instances = 0

            batch.append(idx)
            batch_instances += n_inst

        if batch:
            yield batch

    def __len__(self) -> int:
        if not self.records:
            return 0
        return int(math.ceil(len(self.records) / float(self.batch_size)))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="一键运行 4 个 MIL 模型 demo")
    parser.add_argument("--config", type=str, default="configs/path.yaml", help="路径配置文件")
    parser.add_argument("--demo-config", type=str, default="configs/demo.yaml", help="demo 运行参数配置")
    parser.add_argument("--seed", type=int, default=42, help="随机种子")
    parser.add_argument("--epochs", type=int, default=3, help="每个模型训练轮数")
    parser.add_argument("--patience", type=int, default=2, help="早停耐心轮数")
    parser.add_argument("--image-size", type=int, default=224, help="输入图像尺寸")
    parser.add_argument("--num-workers", type=int, default=-1, help="覆盖 demo.yaml 中的 num_workers；-1 表示不覆盖")
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


def _load_per_model_int_map(payload: dict[str, Any], key: str, default: int) -> dict[str, int]:
    raw = payload.get(key, {})
    if isinstance(raw, int):
        return {k: int(raw) for k in MODEL_KEYS}
    if not isinstance(raw, dict):
        raise ValueError(f"{key} 必须是整数或字典")

    out: dict[str, int] = {}
    for k in MODEL_KEYS:
        value = int(raw.get(k, default))
        if value <= 0:
            raise ValueError(f"{key}.{k} 必须 > 0")
        out[k] = value
    return out


def _load_per_model_float_map(payload: dict[str, Any], key: str, default: float) -> dict[str, float]:
    raw = payload.get(key, {})
    if isinstance(raw, (int, float)):
        v = float(raw)
        return {k: v for k in MODEL_KEYS}
    if not isinstance(raw, dict):
        raise ValueError(f"{key} 必须是数字或字典")

    out: dict[str, float] = {}
    for k in MODEL_KEYS:
        value = float(raw.get(k, default))
        if value < 0.0:
            raise ValueError(f"{key}.{k} 必须 >= 0")
        out[k] = value
    return out


def load_demo_config(config_path: Path) -> dict[str, Any]:
    if not config_path.is_file():
        raise FileNotFoundError(f"未找到 demo 配置文件: {config_path}")

    payload = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("demo 配置文件格式错误")

    gpu_ids_raw = payload.get("gpu_ids", [0, 1, 2])
    if not isinstance(gpu_ids_raw, list) or not gpu_ids_raw:
        raise ValueError("gpu_ids 必须是非空列表")
    gpu_ids = [int(x) for x in gpu_ids_raw]

    num_workers = int(payload.get("num_workers", 6))
    if num_workers < 0:
        raise ValueError("num_workers 必须 >= 0")

    batch_size = _load_per_model_int_map(payload, "batch_size", default=3)
    eval_batch_size = _load_per_model_int_map(payload, "eval_batch_size", default=3)
    train_max_instances = _load_per_model_int_map(payload, "train_max_instances", default=24)
    val_max_instances = _load_per_model_int_map(payload, "val_max_instances", default=24)
    test_max_instances = _load_per_model_int_map(payload, "test_max_instances", default=24)
    train_max_batch_instances = _load_per_model_int_map(payload, "train_max_batch_instances", default=72)
    eval_max_batch_instances = _load_per_model_int_map(payload, "eval_max_batch_instances", default=72)
    random_instance_dropout = _load_per_model_float_map(payload, "random_instance_dropout", default=0.05)

    min_instances = int(payload.get("min_instances", 1))
    if min_instances <= 0:
        raise ValueError("min_instances 必须 > 0")

    train_sampling = str(payload.get("train_sampling_strategy", "random"))
    eval_sampling = str(payload.get("eval_sampling_strategy", "uniform"))

    return {
        "gpu_ids": gpu_ids,
        "num_workers": num_workers,
        "batch_size": batch_size,
        "eval_batch_size": eval_batch_size,
        "train_max_instances": train_max_instances,
        "val_max_instances": val_max_instances,
        "test_max_instances": test_max_instances,
        "train_max_batch_instances": train_max_batch_instances,
        "eval_max_batch_instances": eval_max_batch_instances,
        "random_instance_dropout": random_instance_dropout,
        "min_instances": min_instances,
        "train_sampling_strategy": train_sampling,
        "eval_sampling_strategy": eval_sampling,
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


def ceil_to_multiple(value: int, divisor: int) -> int:
    if divisor <= 0:
        return value
    return ((value + divisor - 1) // divisor) * divisor


def normalize_batch_size(value: int, active_gpu_count: int) -> int:
    v = max(1, int(value))
    if active_gpu_count <= 1:
        return v
    return ceil_to_multiple(max(v, active_gpu_count), active_gpu_count)


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
    train_max_batch_instances: int,
    eval_max_batch_instances: int,
    seed: int,
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
        max_instances_per_bag=val_max_instances,
        min_instances_per_bag=min_instances,
        batch_size=eval_batch_size,
        max_instances_per_batch=eval_max_batch_instances,
        shuffle=False,
        seed=seed + 1,
    )
    test_sampler = InstanceAwareBatchSampler(
        records=split_data["test"],
        max_instances_per_bag=test_max_instances,
        min_instances_per_bag=min_instances,
        batch_size=eval_batch_size,
        max_instances_per_batch=eval_max_batch_instances,
        shuffle=False,
        seed=seed + 2,
    )

    loader_kwargs: dict[str, Any] = {
        "num_workers": num_workers,
        "pin_memory": pin_memory,
        "collate_fn": demo_mil_collate_fn,
        "persistent_workers": num_workers > 0,
    }
    if num_workers > 0:
        loader_kwargs["prefetch_factor"] = 2

    train_loader = DataLoader(train_ds, batch_sampler=train_sampler, **loader_kwargs)
    val_loader = DataLoader(val_ds, batch_sampler=val_sampler, **loader_kwargs)
    test_loader = DataLoader(test_ds, batch_sampler=test_sampler, **loader_kwargs)
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
        train_max_batch_instances=dl_cfg["train_max_batch_instances"],
        eval_max_batch_instances=dl_cfg["eval_max_batch_instances"],
        seed=seed,
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

    path_cfg = load_path_config(Path(args.config))
    demo_cfg = load_demo_config(Path(args.demo_config))

    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = ",".join(str(i) for i in demo_cfg["gpu_ids"])

    torch.set_float32_matmul_precision("medium")

    if torch.cuda.is_available():
        torch.backends.cudnn.benchmark = True

    visible_gpu_count = torch.cuda.device_count() if torch.cuda.is_available() else 0
    use_multi_gpu = (not args.disable_multi_gpu) and visible_gpu_count > 1
    active_gpu_count = visible_gpu_count if use_multi_gpu else (1 if visible_gpu_count > 0 else 0)

    cfg_workers = int(demo_cfg["num_workers"])
    requested_workers = cfg_workers if args.num_workers < 0 else int(args.num_workers)
    cpu_cap = max(1, (os.cpu_count() or 8) - 2)
    effective_workers = max(0, min(requested_workers, cpu_cap))

    report_csv = Path(path_cfg["valid_dicts_report_csv"])
    output_root = Path(path_cfg["output_dir"]) / "demo"
    output_root.mkdir(parents=True, exist_ok=True)

    print("=" * 80)
    print("开始构建 demo 数据")
    print(f"报告 CSV: {report_csv}")
    print(f"输出目录: {output_root}")
    print(
        "硬件配置: "
        f"visible_gpu_count={visible_gpu_count}, "
        f"active_gpu_count={active_gpu_count}, "
        f"num_workers={effective_workers}, "
        f"gpu_ids={demo_cfg['gpu_ids']}"
    )
    print("=" * 80)

    gastro_records, colo_records = build_task_records(
        report_csv_path=report_csv,
        min_instances=demo_cfg["min_instances"],
    )
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

    gastro_pos_weight = compute_multilabel_pos_weight(gastro_split["train"])
    colo_pos_weight = compute_binary_pos_weight(colo_split["train"])

    all_results: dict[str, Any] = {
        "session_dir": str(session_dir),
        "gastro_split": {k: len(v) for k, v in gastro_split.items()},
        "colo_split": {k: len(v) for k, v in colo_split.items()},
        "models": {},
    }

    # 1) 胃镜 baseline
    key_1 = "demo_gastro_mil_baseline"
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
        grad_accum_steps=1 if use_multi_gpu else 2,
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
        "train_batch_size": normalize_batch_size(demo_cfg["batch_size"][key_1], active_gpu_count),
        "eval_batch_size": normalize_batch_size(demo_cfg["eval_batch_size"][key_1], active_gpu_count),
        "train_max_instances": demo_cfg["train_max_instances"][key_1],
        "val_max_instances": demo_cfg["val_max_instances"][key_1],
        "test_max_instances": demo_cfg["test_max_instances"][key_1],
        "train_max_batch_instances": demo_cfg["train_max_batch_instances"][key_1],
        "eval_max_batch_instances": demo_cfg["eval_max_batch_instances"][key_1],
        "min_instances": demo_cfg["min_instances"],
        "train_sampling": demo_cfg["train_sampling_strategy"],
        "eval_sampling": demo_cfg["eval_sampling_strategy"],
        "random_instance_dropout": demo_cfg["random_instance_dropout"][key_1],
    }
    result_1 = run_single_model(
        model_name=key_1,
        model=model_1,
        trainer_cfg=trainer_cfg_1,
        split_data=gastro_split,
        task_name="gastro_multilabel",
        image_size=args.image_size,
        num_workers=effective_workers,
        run_dir=session_dir / "01_demo_gastro_mil_baseline",
        seed=args.seed,
        dl_cfg=dl_cfg_1,
        label_names=GASTRO_LABEL_NAMES,
        class_names=[],
    )
    all_results["models"][key_1] = result_1

    # 2) 胃镜 advanced
    key_2 = "demo_gastro_proto_moe_former"
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
        grad_accum_steps=1 if use_multi_gpu else 4,
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
        "train_batch_size": normalize_batch_size(demo_cfg["batch_size"][key_2], active_gpu_count),
        "eval_batch_size": normalize_batch_size(demo_cfg["eval_batch_size"][key_2], active_gpu_count),
        "train_max_instances": demo_cfg["train_max_instances"][key_2],
        "val_max_instances": demo_cfg["val_max_instances"][key_2],
        "test_max_instances": demo_cfg["test_max_instances"][key_2],
        "train_max_batch_instances": demo_cfg["train_max_batch_instances"][key_2],
        "eval_max_batch_instances": demo_cfg["eval_max_batch_instances"][key_2],
        "min_instances": demo_cfg["min_instances"],
        "train_sampling": demo_cfg["train_sampling_strategy"],
        "eval_sampling": demo_cfg["eval_sampling_strategy"],
        "random_instance_dropout": demo_cfg["random_instance_dropout"][key_2],
    }
    result_2 = run_single_model(
        model_name=key_2,
        model=model_2,
        trainer_cfg=trainer_cfg_2,
        split_data=gastro_split,
        task_name="gastro_multilabel",
        image_size=args.image_size,
        num_workers=effective_workers,
        run_dir=session_dir / "02_demo_gastro_proto_moe_former",
        seed=args.seed,
        dl_cfg=dl_cfg_2,
        label_names=GASTRO_LABEL_NAMES,
        class_names=[],
    )
    all_results["models"][key_2] = result_2

    # 3) 肠镜 baseline
    key_3 = "demo_colo_mil_baseline"
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
        grad_accum_steps=1 if use_multi_gpu else 2,
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
        "train_batch_size": normalize_batch_size(demo_cfg["batch_size"][key_3], active_gpu_count),
        "eval_batch_size": normalize_batch_size(demo_cfg["eval_batch_size"][key_3], active_gpu_count),
        "train_max_instances": demo_cfg["train_max_instances"][key_3],
        "val_max_instances": demo_cfg["val_max_instances"][key_3],
        "test_max_instances": demo_cfg["test_max_instances"][key_3],
        "train_max_batch_instances": demo_cfg["train_max_batch_instances"][key_3],
        "eval_max_batch_instances": demo_cfg["eval_max_batch_instances"][key_3],
        "min_instances": demo_cfg["min_instances"],
        "train_sampling": demo_cfg["train_sampling_strategy"],
        "eval_sampling": demo_cfg["eval_sampling_strategy"],
        "random_instance_dropout": demo_cfg["random_instance_dropout"][key_3],
    }
    result_3 = run_single_model(
        model_name=key_3,
        model=model_3,
        trainer_cfg=trainer_cfg_3,
        split_data=colo_split,
        task_name="colo_binary",
        image_size=args.image_size,
        num_workers=effective_workers,
        run_dir=session_dir / "03_demo_colo_mil_baseline",
        seed=args.seed,
        dl_cfg=dl_cfg_3,
        label_names=[],
        class_names=COLO_BINARY_CLASS_NAMES,
    )
    all_results["models"][key_3] = result_3

    # 4) 肠镜 advanced
    key_4 = "demo_colo_count_aware_debias_mil"
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
        grad_accum_steps=1 if use_multi_gpu else 4,
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
        "train_batch_size": normalize_batch_size(demo_cfg["batch_size"][key_4], active_gpu_count),
        "eval_batch_size": normalize_batch_size(demo_cfg["eval_batch_size"][key_4], active_gpu_count),
        "train_max_instances": demo_cfg["train_max_instances"][key_4],
        "val_max_instances": demo_cfg["val_max_instances"][key_4],
        "test_max_instances": demo_cfg["test_max_instances"][key_4],
        "train_max_batch_instances": demo_cfg["train_max_batch_instances"][key_4],
        "eval_max_batch_instances": demo_cfg["eval_max_batch_instances"][key_4],
        "min_instances": demo_cfg["min_instances"],
        "train_sampling": demo_cfg["train_sampling_strategy"],
        "eval_sampling": demo_cfg["eval_sampling_strategy"],
        "random_instance_dropout": demo_cfg["random_instance_dropout"][key_4],
    }
    result_4 = run_single_model(
        model_name=key_4,
        model=model_4,
        trainer_cfg=trainer_cfg_4,
        split_data=colo_split,
        task_name="colo_binary",
        image_size=args.image_size,
        num_workers=effective_workers,
        run_dir=session_dir / "04_demo_colo_count_aware_debias_mil",
        seed=args.seed,
        dl_cfg=dl_cfg_4,
        label_names=[],
        class_names=COLO_BINARY_CLASS_NAMES,
    )
    all_results["models"][key_4] = result_4

    summary_path = session_dir / "all_models_summary.json"
    summary_path.write_text(json.dumps(all_results, ensure_ascii=False, indent=2), encoding="utf-8")

    print("\n" + "=" * 80)
    print("4 个模型已完成训练/验证/测试")
    print(f"总输出目录: {session_dir}")
    print(f"汇总文件: {summary_path}")
    print("=" * 80)


if __name__ == "__main__":
    main()
