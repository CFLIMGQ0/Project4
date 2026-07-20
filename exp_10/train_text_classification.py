from __future__ import annotations

import argparse
import csv
import json
import random
from pathlib import Path
from typing import Any

import numpy as np
import yaml
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    cohen_kappa_score,
    f1_score,
    hamming_loss,
    precision_recall_fscore_support,
    roc_auc_score,
)
from tqdm import tqdm

from . import MODEL_NAMES
from .data import (
    TextRecord,
    build_mask_audit,
    build_train_vocabulary,
    encode_text,
    labels_array,
    load_masked_records,
    save_json,
    split_records_by_patient,
)


DEFAULT_CONFIG_PATH = Path(__file__).resolve().parents[1] / "configs/task2/exp10_text_classification.yaml"


def _import_torch():
    try:
        import torch
        import torch.nn as nn
        from torch.utils.data import DataLoader, Dataset
    except ImportError as error:
        raise RuntimeError("exp10 的五个神经文本编码器需要在已安装 PyTorch 的项目训练环境中运行") from error
    return torch, nn, DataLoader, Dataset


def tune_thresholds(y_true: np.ndarray, probabilities: np.ndarray, grid: list[float]) -> np.ndarray:
    thresholds = np.full(y_true.shape[1], 0.5, dtype=np.float64)
    for label_index in range(y_true.shape[1]):
        best_f1 = -1.0
        for threshold in grid:
            prediction = probabilities[:, label_index] >= threshold
            score = f1_score(y_true[:, label_index], prediction, zero_division=0)
            if score > best_f1:
                best_f1 = float(score)
                thresholds[label_index] = threshold
    return thresholds


def calculate_metrics(
    y_true: np.ndarray,
    probabilities: np.ndarray,
    thresholds: np.ndarray,
    label_names: list[str],
) -> dict[str, Any]:
    predictions = (probabilities >= thresholds.reshape(1, -1)).astype(np.int64)
    precision, recall, f1, support = precision_recall_fscore_support(
        y_true, predictions, average=None, zero_division=0
    )
    result: dict[str, Any] = {
        "macro_f1": float(f1_score(y_true, predictions, average="macro", zero_division=0)),
        "micro_f1": float(f1_score(y_true, predictions, average="micro", zero_division=0)),
        "macro_roc_auc": float(roc_auc_score(y_true, probabilities, average="macro")),
        "macro_pr_auc": float(average_precision_score(y_true, probabilities, average="macro")),
        "subset_accuracy": float(accuracy_score(y_true, predictions)),
        "hamming_loss": float(hamming_loss(y_true, predictions)),
        "kappa": float(cohen_kappa_score(y_true.reshape(-1), predictions.reshape(-1))),
        "thresholds": dict(zip(label_names, thresholds.astype(float).tolist())),
        "per_label": {},
    }
    for index, label_name in enumerate(label_names):
        result["per_label"][label_name] = {
            "precision": float(precision[index]),
            "recall": float(recall[index]),
            "f1": float(f1[index]),
            "support": int(support[index]),
            "roc_auc": float(roc_auc_score(y_true[:, index], probabilities[:, index])),
            "pr_auc": float(average_precision_score(y_true[:, index], probabilities[:, index])),
        }
    return result


def build_dataset_class():
    torch, _, _, Dataset = _import_torch()

    class Exp10TextDataset(Dataset):
        def __init__(
            self,
            records: list[TextRecord],
            encoder_name: str,
            vocabulary: dict[str, int],
            hash_vocab_size: int,
            max_length: int,
        ) -> None:
            self.records = records
            self.labels = labels_array(records).astype(np.float32)
            self.ids = np.zeros((len(records), max_length), dtype=np.int64)
            self.mask = np.zeros((len(records), max_length), dtype=np.bool_)
            for index, record in enumerate(records):
                self.ids[index], self.mask[index] = encode_text(
                    record.masked_text,
                    encoder_name=encoder_name,
                    vocabulary=vocabulary,
                    hash_vocab_size=hash_vocab_size,
                    max_length=max_length,
                )

        def __len__(self) -> int:
            return len(self.records)

        def __getitem__(self, index: int):
            return (
                torch.from_numpy(self.ids[index]),
                torch.from_numpy(self.mask[index]),
                torch.from_numpy(self.labels[index]),
            )

    return Exp10TextDataset


def predict(model, loader, device) -> tuple[np.ndarray, np.ndarray]:
    torch, _, _, _ = _import_torch()
    model.eval()
    probabilities: list[np.ndarray] = []
    labels: list[np.ndarray] = []
    with torch.no_grad():
        for token_ids, token_mask, targets in loader:
            logits = model(token_ids.to(device), token_mask.to(device))
            probabilities.append(torch.sigmoid(logits).cpu().numpy())
            labels.append(targets.numpy())
    return np.concatenate(probabilities), np.concatenate(labels)


def save_predictions(
    path: Path,
    records: list[TextRecord],
    probabilities: np.ndarray,
    metrics: dict[str, Any],
    label_names: list[str],
) -> None:
    with path.open("w", encoding="utf-8-sig", newline="") as file:
        fieldnames = ["patient_id", "exam_dir"]
        for label_name in label_names:
            fieldnames.extend([f"true_{label_name}", f"prob_{label_name}", f"pred_{label_name}"])
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        for record, sample_probabilities in zip(records, probabilities):
            row: dict[str, Any] = {"patient_id": record.patient_id, "exam_dir": record.exam_dir}
            for index, label_name in enumerate(label_names):
                probability = float(sample_probabilities[index])
                row[f"true_{label_name}"] = int(record.labels[index])
                row[f"prob_{label_name}"] = probability
                row[f"pred_{label_name}"] = int(probability >= metrics["thresholds"][label_name])
            writer.writerow(row)


def train_one_model(
    model_name: str,
    config: dict[str, Any],
    splits: dict[str, list[TextRecord]],
    vocabulary: dict[str, int],
    output_dir: Path,
) -> dict[str, Any]:
    torch, nn, DataLoader, _ = _import_torch()
    from .models import build_text_classifier

    seed = int(config["seed"])
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    data_cfg = config["data"]
    training_cfg = config["training"]
    label_names = list(data_cfg["label_names"])
    max_length = int(data_cfg["max_length"])
    hash_vocab_size = int(data_cfg["hash_vocab_size"])
    dataset_cls = build_dataset_class()
    datasets = {
        split: dataset_cls(records, model_name, vocabulary, hash_vocab_size, max_length)
        for split, records in splits.items()
    }
    loaders = {
        split: DataLoader(
            dataset,
            batch_size=int(training_cfg["batch_size"]),
            shuffle=split == "train",
            num_workers=int(training_cfg["num_workers"]),
            pin_memory=torch.cuda.is_available(),
        )
        for split, dataset in datasets.items()
    }

    model = build_text_classifier(
        model_name,
        vocabulary_size=len(vocabulary),
        hash_vocab_size=hash_vocab_size,
        num_labels=len(label_names),
        max_length=max_length,
        model_config=config["model"],
    )
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    train_labels = labels_array(splits["train"]).astype(np.float32)
    positives = train_labels.sum(axis=0)
    negatives = len(train_labels) - positives
    pos_weight = torch.tensor(negatives / np.maximum(positives, 1.0), dtype=torch.float32, device=device)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(training_cfg["learning_rate"]),
        weight_decay=float(training_cfg["weight_decay"]),
    )

    run_dir = output_dir / model_name
    run_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path = run_dir / "best_model.pt"
    best_val_f1 = -1.0
    best_epoch = 0
    stale_epochs = 0
    history: list[dict[str, float | int]] = []
    for epoch in range(1, int(training_cfg["max_epochs"]) + 1):
        model.train()
        losses: list[float] = []
        for token_ids, token_mask, targets in tqdm(
            loaders["train"], desc=f"训练 {model_name} epoch {epoch}", leave=False
        ):
            optimizer.zero_grad(set_to_none=True)
            logits = model(token_ids.to(device), token_mask.to(device))
            loss = criterion(logits, targets.to(device))
            loss.backward()
            optimizer.step()
            losses.append(float(loss.detach().cpu()))
        val_probabilities, val_targets = predict(model, loaders["val"], device)
        thresholds = tune_thresholds(val_targets, val_probabilities, list(training_cfg["threshold_grid"]))
        val_f1 = float(
            f1_score(val_targets, val_probabilities >= thresholds, average="macro", zero_division=0)
        )
        history.append(
            {"epoch": epoch, "train_loss": float(np.mean(losses)), "val_macro_f1": val_f1}
        )
        print(
            f"[{model_name}] epoch={epoch:03d} "
            f"train_loss={np.mean(losses):.4f} val_macro_f1={val_f1:.4f}"
        )
        if val_f1 > best_val_f1:
            best_val_f1 = val_f1
            best_epoch = epoch
            stale_epochs = 0
            torch.save(model.state_dict(), checkpoint_path)
        else:
            stale_epochs += 1
        if stale_epochs >= int(training_cfg["patience"]):
            break

    try:
        state_dict = torch.load(checkpoint_path, map_location=device, weights_only=True)
    except TypeError:
        state_dict = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(state_dict)
    val_probabilities, val_targets = predict(model, loaders["val"], device)
    thresholds = tune_thresholds(val_targets, val_probabilities, list(training_cfg["threshold_grid"]))
    test_probabilities, test_targets = predict(model, loaders["test"], device)
    metrics = calculate_metrics(test_targets, test_probabilities, thresholds, label_names)
    metrics.update(
        {
            "experiment_name": "exp10_text_classification",
            "model_name": model_name,
            "text_field": "watch",
            "answer_masking": True,
            "best_epoch": best_epoch,
            "best_val_macro_f1": best_val_f1,
        }
    )
    save_json(run_dir / "history.json", history)
    save_json(run_dir / "test_metrics.json", metrics)
    save_predictions(run_dir / "test_predictions.csv", splits["test"], test_probabilities, metrics, label_names)
    return metrics


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="exp10：五种掩码报告文本编码器 + MLP 三标签分类")
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG_PATH)
    parser.add_argument("--models", default="", help="逗号分隔的编码器名；默认运行配置中的五个模型")
    parser.add_argument("--audit-only", action="store_true", help="只检查遮蔽、划分和词表，不启动训练")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    with args.config.open("r", encoding="utf-8") as file:
        config = yaml.safe_load(file)
    records = load_masked_records(config)
    splits = split_records_by_patient(records, int(config["seed"]), list(config["data"]["split_ratio"]))
    label_names = list(config["data"]["label_names"])
    output_dir = Path(config["paths"]["output_dir"]).expanduser()
    output_dir.mkdir(parents=True, exist_ok=True)

    audit = build_mask_audit(splits, label_names, str(config["data"]["mask_token"]))
    save_json(output_dir / "mask_audit.json", audit)
    if any(audit["patient_overlap"].values()):
        raise RuntimeError(f"患者划分存在交叉: {audit['patient_overlap']}")
    if any(item["residual_answer_rows"] for item in audit["splits"].values()):
        raise RuntimeError("遮蔽后仍有答案词残留")

    vocabulary = build_train_vocabulary(
        splits["train"],
        max_vocab_size=int(config["data"]["vocab_size"]),
        min_frequency=int(config["data"]["min_token_frequency"]),
    )
    save_json(output_dir / "vocabulary.json", vocabulary)
    print(f"数据审计完成，训练集词表大小：{len(vocabulary)}")
    if args.audit_only:
        return

    requested = [item.strip() for item in args.models.split(",") if item.strip()]
    model_names = requested or list(config["models"])
    unknown = sorted(set(model_names) - set(MODEL_NAMES))
    if unknown:
        raise ValueError(f"未知 exp10 编码器: {unknown}；可选值为 {list(MODEL_NAMES)}")
    if len(set(config["models"])) != 5:
        raise ValueError("exp10 默认配置必须且只能包含五种文本编码器")

    summary = [train_one_model(name, config, splits, vocabulary, output_dir) for name in model_names]
    save_json(output_dir / "summary.json", summary)


if __name__ == "__main__":
    main()
