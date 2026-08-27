#!/usr/bin/env python3
"""汇总指定模型的五折测试预测并绘制 one-vs-rest 混淆矩阵。"""

from __future__ import annotations

import argparse
import csv
import inspect
import json
import os
import shutil
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any


def _ensure_project_runtime_python() -> None:
    try:
        import torch  # noqa: F401
        return
    except ImportError:
        pass
    candidate = Path("/xmlg/Lim/conda/envs/myenv/bin/python")
    if candidate.is_file() and Path(sys.executable).resolve() != candidate.resolve():
        os.execv(str(candidate), [str(candidate), *sys.argv])


_ensure_project_runtime_python()

import matplotlib.pyplot as plt
import numpy as np
import torch
import yaml
from sklearn.metrics import confusion_matrix, f1_score
from torch.utils.data import DataLoader
from tqdm import tqdm

ROOT = Path(__file__).resolve().parents[1]
PROJECT_ROOT = ROOT.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from exp_10.models import build_text_classifier
from exp_10.train_text_classification import build_dataset_class, predict, tune_thresholds
from scripts.task1_distinct_significance import (
    build_test_loader,
    extract_probabilities,
    load_run_config,
    move_batch_to_device,
)
from scripts.task3_main_model_5fold import LABEL_NAMES, apply_watch_mask
from scripts.task3_table2_5fold import as_text_record
from sotas.task2.multimodal_sotas import (
    TASK2_MULTIMODAL_SOTA_CLASS_REGISTRY,
    build_task2_multimodal_sota,
)
from train import build_model_bundle, load_train_config


LABEL_DISPLAY_NAMES = (
    "Esophageal submucosal tumor",
    "Esophageal mucosal lesion",
    "Gastritis",
)


@dataclass(frozen=True)
class Selection:
    key: str
    dataset: str
    dataset_display: str
    model_display: str
    modality: str
    run_root: Path
    split_root: Path
    expected_macro_f1: float
    run_subdir: str = ""

    def fold_run_dir(self, fold: int) -> Path:
        base = self.run_root / self.dataset / f"fold_{fold}"
        return base / self.run_subdir if self.run_subdir else base

    def fold_manifest(self, fold: int) -> Path:
        return self.split_root / self.dataset / f"fold_{fold}" / "split_manifest.csv"


def selections() -> dict[str, Selection]:
    output_root = PROJECT_ROOT / "outputs/train_runs/task3"
    sota_root = output_root / "t3_multimodal_sotas_5fold"
    text_root = output_root / "t3_table2_5fold"
    return {
        "mmtf_wle": Selection(
            key="mmtf_wle",
            dataset="regular_white_light",
            dataset_display="WLE",
            model_display="MMTF",
            modality="image_text",
            run_root=sota_root / "image",
            split_root=sota_root / "data_splits",
            run_subdir="task2_mmtf_2025",
            expected_macro_f1=0.9149,
        ),
        "promef_chromoscopic": Selection(
            key="promef_chromoscopic",
            dataset="chromoscopic",
            dataset_display="Chromoscopic gastroscopy",
            model_display="ProMEF-MIL (ours; w/o mass conservation)",
            modality="reported_test_counts",
            run_root=output_root / "t3_apro_cope_ablation/apro_no_conservation",
            split_root=output_root / "t3_apro_cope_ablation/apro_no_conservation",
            expected_macro_f1=0.8877,
        ),
        "textcnn_surgical": Selection(
            key="textcnn_surgical",
            dataset="surgical",
            dataset_display="Surgical gastroscopy",
            model_display="TextCNN",
            modality="text",
            run_root=text_root / "text",
            split_root=text_root / "data_splits",
            run_subdir="textcnn_encoder",
            # 由当前保留的五个 best_model.pt 及其验证集阈值复算。
            expected_macro_f1=0.8756,
        ),
        "mmfnet_eus": Selection(
            key="mmfnet_eus",
            dataset="ultrasound",
            dataset_display="EUS",
            model_display="MMFNet",
            modality="image_text",
            run_root=sota_root / "image",
            split_root=sota_root / "data_splits",
            run_subdir="task2_mmfnet_2024",
            expected_macro_f1=0.9064,
        ),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--only",
        default="",
        help="逗号分隔的任务键；默认运行全部四组",
    )
    parser.add_argument("--device", default="auto", help="auto、cpu、cuda 或 cuda:N")
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--force", action="store_true", help="忽略已有折级预测缓存并重新推理")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=PROJECT_ROOT / "outputs/figures/task3_oof_confusion",
    )
    parser.add_argument(
        "--paper-copy",
        type=Path,
        default=ROOT / "figs/selected_models_oof_confusion_matrices.pdf",
        help="完成全部四组后复制一份 PDF 作为论文插图；设为空字符串可禁用",
    )
    return parser.parse_args()


def resolve_device(raw: str) -> torch.device:
    value = str(raw).strip().lower()
    if value == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(value)


def read_manifest(path: Path) -> dict[str, list[str]]:
    if not path.is_file():
        raise FileNotFoundError(f"缺少五折划分清单：{path}")
    result = {"train": [], "val": [], "test": []}
    with path.open("r", encoding="utf-8-sig", newline="") as file:
        for row in csv.DictReader(file):
            split = str(row["split"])
            if split in result:
                result[split].append(str(row["exam_dir"]))
    return result


def load_records() -> dict[str, dict[str, Any]]:
    cache_path = PROJECT_ROOT / "outputs/train_runs/task3/t3_main_model/records_cache.json"
    payload = json.loads(cache_path.read_text(encoding="utf-8"))
    records = payload.get("records", [])
    if not records:
        raise RuntimeError(f"样本缓存为空：{cache_path}")
    apply_watch_mask(records, True)
    return {str(record["exam_dir"]): record for record in records}


def split_records(
    selection: Selection,
    fold: int,
    records_by_exam: dict[str, dict[str, Any]],
) -> dict[str, list[dict[str, Any]]]:
    manifest = read_manifest(selection.fold_manifest(fold))
    split: dict[str, list[dict[str, Any]]] = {}
    for split_name, exam_dirs in manifest.items():
        missing = [exam_dir for exam_dir in exam_dirs if exam_dir not in records_by_exam]
        if missing:
            raise KeyError(f"{selection.key} fold_{fold} 的清单中有 {len(missing)} 个检查未命中样本缓存")
        split[split_name] = [records_by_exam[exam_dir] for exam_dir in exam_dirs]
    return split


def resolve_checkpoint(run_dir: Path) -> Path:
    candidates = (
        run_dir / "checkpoints/best_macro_f1.ckpt",
        run_dir / "best_macro_f1.ckpt",
    )
    for path in candidates:
        if path.is_file():
            return path
    raise FileNotFoundError(f"缺少 best_macro_f1 checkpoint：{run_dir}")


def load_image_model(
    run_dir: Path,
    train_cfg: dict[str, Any],
    device: torch.device,
) -> tuple[torch.nn.Module, dict[str, Any], int, int, bool]:
    model_name, run_cfg, model_params, seed, image_size = load_run_config(run_dir, train_cfg)
    cache_root = PROJECT_ROOT / "datasets/image_cache"
    run_cfg.update(
        {
            "image_cache_mode": "disk",
            "image_cache_dir": str(cache_root / "shared"),
            "resolved_image_cache_dir": str(cache_root / "shared"),
            "image_cache_manifest": str(cache_root / "task3_cache_manifest.jsonl.gz"),
            "resolved_legacy_image_cache_dirs": [
                str(cache_root / "task1"),
                str(cache_root / "task2"),
            ],
            "image_cache_warmup": False,
            "eval_batch_size": 1,
        }
    )
    config = yaml.safe_load((run_dir / "config.yaml").read_text(encoding="utf-8"))
    trainer_cfg = dict(config.get("trainer", {}))
    pos_weight = trainer_cfg.get("pos_weight") or [1.0] * len(LABEL_NAMES)
    if model_name in TASK2_MULTIMODAL_SOTA_CLASS_REGISTRY:
        # 当前训练入口新增了仅供部分模型使用的蒸馏参数，而旧版 SOTA
        # checkpoint 的基类并不接收该参数；按保存配置直接重建原始模型。
        model = build_task2_multimodal_sota(
            model_name=model_name,
            backbone_name=str(model_params.get("backbone_name", "convnext_tiny")),
            pretrained=False,
            freeze_stages=int(model_params.get("freeze_stages", 1)),
            feature_dim=int(model_params.get("feature_dim", 512)),
            attn_dim=int(model_params.get("attn_dim", 256)),
            hidden_dim=int(model_params.get("hidden_dim", 1024)),
            num_labels=len(LABEL_NAMES),
            dropout=float(model_params.get("dropout", 0.2)),
            encoder_chunk_size=int(model_params.get("encoder_chunk_size", 16)),
            text_vocab_size=int(model_params.get("text_vocab_size", 8192)),
            text_embed_dim=int(model_params.get("text_embed_dim", 128)),
            textcnn_kernel_sizes=tuple(model_params.get("textcnn_kernel_sizes", (2, 3, 4))),
            num_heads=int(model_params.get("num_heads", 4)),
            num_layers=int(model_params.get("num_layers", 2)),
            correlation_threshold=float(model_params.get("correlation_threshold", 0.5)),
            alignment_temperature=float(model_params.get("alignment_temperature", 1.0)),
            contrast_temperature=float(model_params.get("contrast_temperature", 0.07)),
            contrast_queue_size=int(model_params.get("contrast_queue_size", 256)),
            window_size=int(model_params.get("window_size", 8)),
        )
    else:
        model, _, _, _, _ = build_model_bundle(
            model_name=model_name,
            task_name="task2",
            run_cfg=run_cfg,
            model_param_cfg=model_params,
            pretrained=False,
            max_epochs=int(trainer_cfg.get("max_epochs", 30)),
            patience=int(trainer_cfg.get("patience", 30)),
            pos_weight=pos_weight,
            use_multi_gpu=False,
            run_test=False,
        )
    checkpoint_path = resolve_checkpoint(run_dir)
    try:
        checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    except TypeError:
        checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint.get("model_state", checkpoint), strict=True)
    model.to(device).eval()
    return model, run_cfg, seed, image_size, bool(trainer_cfg.get("amp", True))


def forward_image_model(model: torch.nn.Module, batch: dict[str, Any]) -> dict[str, Any]:
    parameters = inspect.signature(model.forward).parameters
    kwargs: dict[str, Any] = {"images": batch["images"], "mask": batch["mask"]}
    optional_keys = (
        "labels",
        "instance_types",
        "pseudo_region_labels",
        "pseudo_relevance",
        "instance_indices",
        "original_image_counts",
        "structured_categorical",
        "structured_numeric",
        "structured_mask",
        "text_token_ids",
        "text_token_mask",
        "watch_token_ids",
        "watch_token_mask",
        "guided_text_token_ids",
        "guided_text_token_mask",
    )
    for key in optional_keys:
        if key in parameters and key in batch:
            kwargs[key] = batch[key]
    if "current_epoch" in parameters:
        kwargs["current_epoch"] = 0.0
    output = model(**kwargs)
    if torch.is_tensor(output):
        return {"logits": output}
    if not isinstance(output, dict):
        raise TypeError(f"模型输出类型不支持：{type(output)}")
    return output


def infer_image_fold(
    selection: Selection,
    fold: int,
    split: dict[str, list[dict[str, Any]]],
    train_cfg: dict[str, Any],
    device: torch.device,
    num_workers: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, list[str]]:
    run_dir = selection.fold_run_dir(fold)
    model, run_cfg, seed, image_size, amp_enabled = load_image_model(run_dir, train_cfg, device)
    loader = build_test_loader(
        split_data=split,
        task_name="task2",
        image_size=image_size,
        num_workers=num_workers,
        run_cfg=run_cfg,
        seed=seed,
        min_instances=1,
    )
    true_parts: list[np.ndarray] = []
    prob_parts: list[np.ndarray] = []
    exam_dirs: list[str] = []
    iterator = tqdm(loader, desc=f"{selection.dataset_display} {selection.model_display} fold {fold}", dynamic_ncols=True)
    with torch.inference_mode():
        for batch_cpu in iterator:
            batch = move_batch_to_device(batch_cpu, device)
            with torch.autocast(
                device_type=device.type,
                dtype=torch.float16,
                enabled=amp_enabled and device.type == "cuda",
            ):
                output = forward_image_model(model, batch)
                probabilities = extract_probabilities(output)
            true_parts.append(batch["labels"].detach().cpu().numpy().astype(np.int64))
            prob_parts.append(probabilities.detach().float().cpu().numpy())
            exam_dirs.extend(str(value) for value in batch_cpu["exam_dirs"])
    y_true = np.concatenate(true_parts)
    y_prob = np.concatenate(prob_parts)
    y_pred = (y_prob >= 0.5).astype(np.int64)
    del model
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return y_true, y_prob, y_pred, exam_dirs


def infer_text_fold(
    selection: Selection,
    fold: int,
    split: dict[str, list[dict[str, Any]]],
    device: torch.device,
    num_workers: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, list[str], np.ndarray]:
    run_dir = selection.fold_run_dir(fold)
    config = json.loads((run_dir / "config.json").read_text(encoding="utf-8"))
    vocabulary = json.loads((run_dir / "vocabulary.json").read_text(encoding="utf-8"))
    text_split = {
        name: [as_text_record(record) for record in records]
        for name, records in split.items()
    }
    data_cfg = config["data"]
    training_cfg = config["training"]
    dataset_cls = build_dataset_class()

    def loader(name: str) -> DataLoader:
        dataset = dataset_cls(
            text_split[name],
            "textcnn_encoder",
            vocabulary,
            int(data_cfg["hash_vocab_size"]),
            int(data_cfg["max_length"]),
        )
        return DataLoader(
            dataset,
            batch_size=int(training_cfg["batch_size"]),
            shuffle=False,
            num_workers=num_workers,
            pin_memory=device.type == "cuda",
        )

    model = build_text_classifier(
        "textcnn_encoder",
        vocabulary_size=len(vocabulary),
        hash_vocab_size=int(data_cfg["hash_vocab_size"]),
        num_labels=len(LABEL_NAMES),
        max_length=int(data_cfg["max_length"]),
        model_config=config["model"],
    )
    try:
        state = torch.load(run_dir / "best_model.pt", map_location=device, weights_only=True)
    except TypeError:
        state = torch.load(run_dir / "best_model.pt", map_location=device)
    model.load_state_dict(state, strict=True)
    model.to(device).eval()
    val_prob, val_true = predict(model, loader("val"), device)
    thresholds = tune_thresholds(val_true, val_prob, list(training_cfg["threshold_grid"]))
    test_prob, test_true = predict(model, loader("test"), device)
    test_pred = (test_prob >= thresholds.reshape(1, -1)).astype(np.int64)
    exam_dirs = [record.exam_dir for record in text_split["test"]]
    del model
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return test_true.astype(np.int64), test_prob, test_pred, exam_dirs, thresholds


def recover_reported_test_counts(
    selection: Selection,
    fold: int,
    split: dict[str, list[dict[str, Any]]],
) -> tuple[np.ndarray, np.ndarray, np.ndarray, list[str], np.ndarray]:
    """从保留的逐标签测试指标精确恢复 one-vs-rest 计数。

    APro-CoPE 消融运行保留了每折 test_result.csv，但未保留逐样本概率。
    真实阳性数来自该折测试 manifest；recall 与 specificity 对应的整数计数
    可据此无歧义恢复。合成数组只用于统一后续逐标签混淆矩阵计算。
    """

    result_path = selection.fold_run_dir(fold) / "test_result.csv"
    with result_path.open("r", encoding="utf-8-sig", newline="") as file:
        rows = list(csv.DictReader(file))
    row = next(item for item in rows if item["checkpoint_alias"] == "best_macro_f1")
    test_records = split["test"]
    sample_count = len(test_records)
    y_true = np.zeros((sample_count, len(LABEL_NAMES)), dtype=np.int64)
    y_pred = np.zeros_like(y_true)
    thresholds = np.empty(len(LABEL_NAMES), dtype=np.float64)
    for label_index, label in enumerate(LABEL_NAMES):
        positives = sum(int(record["labels"][label_index]) for record in test_records)
        negatives = sample_count - positives
        recall = float(row[f"recall_{label}"])
        specificity = float(row[f"specificity_{label}"])
        true_positive = int(round(recall * positives))
        true_negative = int(round(specificity * negatives))
        false_positive = negatives - true_negative
        y_true[:positives, label_index] = 1
        y_pred[:true_positive, label_index] = 1
        y_pred[positives : positives + false_positive, label_index] = 1
        thresholds[label_index] = float(row[f"threshold_{label}"])
    # 这里只恢复逐标签汇总计数，不能把合成行误写成某位患者的逐样本预测。
    exam_dirs = [f"aggregate-count-slot/fold-{fold}/{index}" for index in range(sample_count)]
    return y_true, y_pred.astype(np.float64), y_pred, exam_dirs, thresholds


def write_fold_predictions(
    path: Path,
    fold: int,
    exam_dirs: list[str],
    y_true: np.ndarray,
    y_prob: np.ndarray,
    y_pred: np.ndarray,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8-sig", newline="") as file:
        fieldnames = ["fold", "exam_dir"]
        for label in LABEL_NAMES:
            fieldnames.extend((f"true_{label}", f"prob_{label}", f"pred_{label}"))
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        for index, exam_dir in enumerate(exam_dirs):
            row: dict[str, Any] = {"fold": fold, "exam_dir": exam_dir}
            for label_index, label in enumerate(LABEL_NAMES):
                row[f"true_{label}"] = int(y_true[index, label_index])
                row[f"prob_{label}"] = float(y_prob[index, label_index])
                row[f"pred_{label}"] = int(y_pred[index, label_index])
            writer.writerow(row)


def load_cached_fold(path: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray, list[str]]:
    rows: list[dict[str, str]]
    with path.open("r", encoding="utf-8-sig", newline="") as file:
        rows = list(csv.DictReader(file))
    y_true = np.asarray([[int(row[f"true_{label}"]) for label in LABEL_NAMES] for row in rows], dtype=np.int64)
    y_prob = np.asarray([[float(row[f"prob_{label}"]) for label in LABEL_NAMES] for row in rows], dtype=np.float64)
    y_pred = np.asarray([[int(row[f"pred_{label}"]) for label in LABEL_NAMES] for row in rows], dtype=np.int64)
    return y_true, y_prob, y_pred, [row["exam_dir"] for row in rows]


def collect_selection(
    selection: Selection,
    records_by_exam: dict[str, dict[str, Any]],
    train_cfg: dict[str, Any],
    output_dir: Path,
    device: torch.device,
    num_workers: int,
    force: bool,
) -> dict[str, Any]:
    true_parts: list[np.ndarray] = []
    prob_parts: list[np.ndarray] = []
    pred_parts: list[np.ndarray] = []
    fold_summaries: list[dict[str, Any]] = []
    for fold in range(1, 6):
        cache_path = output_dir / "predictions" / selection.key / f"fold_{fold}.csv"
        thresholds: np.ndarray | None = None
        if cache_path.is_file() and not force:
            y_true, y_prob, y_pred, exam_dirs = load_cached_fold(cache_path)
        else:
            split = split_records(selection, fold, records_by_exam)
            if selection.modality == "text":
                y_true, y_prob, y_pred, exam_dirs, thresholds = infer_text_fold(
                    selection, fold, split, device, num_workers
                )
            elif selection.modality == "reported_test_counts":
                y_true, y_prob, y_pred, exam_dirs, thresholds = recover_reported_test_counts(
                    selection, fold, split
                )
            else:
                y_true, y_prob, y_pred, exam_dirs = infer_image_fold(
                    selection, fold, split, train_cfg, device, num_workers
                )
            write_fold_predictions(cache_path, fold, exam_dirs, y_true, y_prob, y_pred)
        macro_f1 = float(f1_score(y_true, y_pred, average="macro", zero_division=0))
        fold_summaries.append(
            {
                "fold": fold,
                "samples": int(y_true.shape[0]),
                "macro_f1": macro_f1,
                "thresholds": thresholds.tolist() if thresholds is not None else [0.5] * len(LABEL_NAMES),
            }
        )
        true_parts.append(y_true)
        prob_parts.append(y_prob)
        pred_parts.append(y_pred)
    y_true_all = np.concatenate(true_parts)
    y_prob_all = np.concatenate(prob_parts)
    y_pred_all = np.concatenate(pred_parts)
    fold_mean = float(np.mean([row["macro_f1"] for row in fold_summaries]))
    if abs(fold_mean - selection.expected_macro_f1) > 5e-4:
        raise RuntimeError(
            f"{selection.key} 复算五折 Macro F1={fold_mean:.4f}，"
            f"与目标值 {selection.expected_macro_f1:.4f} 不一致"
        )
    matrices = [
        confusion_matrix(y_true_all[:, index], y_pred_all[:, index], labels=[0, 1])
        for index in range(len(LABEL_NAMES))
    ]
    return {
        "selection": selection,
        "y_true": y_true_all,
        "y_prob": y_prob_all,
        "y_pred": y_pred_all,
        "folds": fold_summaries,
        "fold_macro_f1_mean": fold_mean,
        "fold_macro_f1_std": float(np.std([row["macro_f1"] for row in fold_summaries], ddof=1)),
        "matrices": matrices,
    }


def save_summary(results: list[dict[str, Any]], output_dir: Path) -> None:
    payload: list[dict[str, Any]] = []
    csv_rows: list[dict[str, Any]] = []
    for result in results:
        selection: Selection = result["selection"]
        item = {
            "key": selection.key,
            "dataset": selection.dataset,
            "dataset_display": selection.dataset_display,
            "model": selection.model_display,
            "prediction_provenance": selection.modality,
            "actual_run_root": str(selection.run_root),
            "samples": int(result["y_true"].shape[0]),
            "fold_macro_f1_mean": result["fold_macro_f1_mean"],
            "fold_macro_f1_std": result["fold_macro_f1_std"],
            "folds": result["folds"],
            "confusion_matrices": {},
        }
        for label, matrix in zip(LABEL_NAMES, result["matrices"]):
            tn, fp, fn, tp = (int(value) for value in matrix.ravel())
            item["confusion_matrices"][label] = {"tn": tn, "fp": fp, "fn": fn, "tp": tp}
            csv_rows.append(
                {
                    "dataset": selection.dataset_display,
                    "model": selection.model_display,
                    "label": label,
                    "tn": tn,
                    "fp": fp,
                    "fn": fn,
                    "tp": tp,
                }
            )
        payload.append(item)
    (output_dir / "oof_confusion_summary.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    with (output_dir / "oof_confusion_counts.csv").open("w", encoding="utf-8-sig", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=list(csv_rows[0]))
        writer.writeheader()
        writer.writerows(csv_rows)


def plot_results(results: list[dict[str, Any]], output_dir: Path) -> Path:
    plt.rcParams.update(
        {
            "font.family": "serif",
            "font.serif": ["Times New Roman", "TeX Gyre Termes", "DejaVu Serif"],
            "font.size": 10,
            "axes.titlesize": 11,
        }
    )
    figure, axes = plt.subplots(len(results), len(LABEL_NAMES), figsize=(10.4, 9.2))
    if len(results) == 1:
        axes = np.asarray([axes])
    image = None
    for row_index, result in enumerate(results):
        selection: Selection = result["selection"]
        for col_index, (label_display, matrix) in enumerate(zip(LABEL_DISPLAY_NAMES, result["matrices"])):
            axis = axes[row_index, col_index]
            row_totals = matrix.sum(axis=1, keepdims=True)
            normalized = np.divide(
                matrix,
                row_totals,
                out=np.zeros_like(matrix, dtype=float),
                where=row_totals != 0,
            )
            image = axis.imshow(normalized, cmap="Blues", vmin=0.0, vmax=1.0)
            for true_index in range(2):
                for pred_index in range(2):
                    value = int(matrix[true_index, pred_index])
                    percent = 100.0 * normalized[true_index, pred_index]
                    color = "white" if normalized[true_index, pred_index] >= 0.55 else "black"
                    axis.text(
                        pred_index,
                        true_index,
                        f"{value}\n({percent:.1f}%)",
                        ha="center",
                        va="center",
                        color=color,
                        fontsize=10,
                    )
            axis.set_xticks((0, 1), labels=("Negative", "Positive"))
            axis.set_yticks((0, 1), labels=("Negative", "Positive"))
            axis.set_xlabel("Predicted class")
            if col_index == 0:
                axis.set_ylabel(
                    f"{selection.dataset_display}\n\nTrue class",
                    labelpad=9,
                )
            else:
                axis.set_ylabel("True class")
            axis.set_title(label_display)
    figure.suptitle("Pooled out-of-fold confusion matrices across five test folds", fontsize=14, y=0.995)
    figure.subplots_adjust(left=0.12, right=0.91, top=0.95, bottom=0.06, hspace=0.42, wspace=0.18)
    if image is not None:
        color_axis = figure.add_axes((0.93, 0.12, 0.018, 0.75))
        colorbar = figure.colorbar(image, cax=color_axis)
        colorbar.set_label("Row-normalized proportion")
    output_dir.mkdir(parents=True, exist_ok=True)
    pdf_path = output_dir / "selected_models_oof_confusion_matrices.pdf"
    png_path = output_dir / "selected_models_oof_confusion_matrices.png"
    svg_path = output_dir / "selected_models_oof_confusion_matrices.svg"
    figure.savefig(pdf_path, bbox_inches="tight")
    figure.savefig(png_path, dpi=300, bbox_inches="tight")
    figure.savefig(svg_path, bbox_inches="tight")
    plt.close(figure)
    return pdf_path


def main() -> None:
    args = parse_args()
    available = selections()
    selected_keys = [value.strip() for value in args.only.split(",") if value.strip()] or list(available)
    unknown = sorted(set(selected_keys) - set(available))
    if unknown:
        raise ValueError(f"未知任务键：{unknown}；允许值={list(available)}")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    device = resolve_device(args.device)
    print(f"[OOF confusion] 推理设备：{device}")
    records_by_exam = load_records()
    train_cfg = load_train_config(ROOT / "configs/task2/train.yaml")
    results = [
        collect_selection(
            available[key],
            records_by_exam,
            train_cfg,
            args.output_dir,
            device,
            max(0, int(args.num_workers)),
            bool(args.force),
        )
        for key in selected_keys
    ]
    save_summary(results, args.output_dir)
    if len(results) == len(available):
        pdf_path = plot_results(results, args.output_dir)
        if str(args.paper_copy):
            args.paper_copy.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(pdf_path, args.paper_copy)
            svg_path = pdf_path.with_suffix(".svg")
            paper_svg = args.paper_copy.with_suffix(".svg")
            shutil.copy2(svg_path, paper_svg)
            png_path = pdf_path.with_suffix(".png")
            paper_png = args.paper_copy.with_suffix(".png")
            shutil.copy2(png_path, paper_png)
            print(f"[OOF confusion] 论文插图副本：{args.paper_copy}")
            print(f"[OOF confusion] SVG 插图副本：{paper_svg}")
            print(f"[OOF confusion] PNG 插图副本：{paper_png}")
        print(f"[OOF confusion] 图表：{pdf_path}")
    else:
        print("[OOF confusion] 当前仅完成部分任务；集齐四组预测后再生成组合图。")
    print(f"[OOF confusion] 汇总：{args.output_dir / 'oof_confusion_summary.json'}")


if __name__ == "__main__":
    main()
