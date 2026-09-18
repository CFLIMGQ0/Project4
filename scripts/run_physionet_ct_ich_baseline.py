#!/usr/bin/env python3
"""在 PhysioNet CT-ICH 上运行检查级纯图像均值池化基线。"""

from __future__ import annotations

import argparse
import csv
import json
import random
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    f1_score,
    precision_recall_fscore_support,
    roc_auc_score,
)
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, Dataset
from torchvision import models, transforms
from tqdm import tqdm


DEFAULT_LABELS = ("Intraparenchymal", "Epidural", "Fracture_Yes_No")
DISPLAY_NAMES = {
    "Intraparenchymal": "IPH",
    "Epidural": "EDH",
    "Fracture_Yes_No": "Fracture",
}


class SliceDataset(Dataset):
    def __init__(self, rows: list[dict[str, object]], image_size: int) -> None:
        self.rows = rows
        self.transform = transforms.Compose(
            [
                transforms.Resize((image_size, image_size), antialias=True),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=(0.485, 0.456, 0.406),
                    std=(0.229, 0.224, 0.225),
                ),
            ]
        )

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, index: int):
        row = self.rows[index]
        with Image.open(str(row["image_path"])) as image:
            tensor = self.transform(image.convert("RGB"))
        return tensor, int(row["patient_id"]), int(row["slice_number"])


def parse_args() -> argparse.Namespace:
    project_root = Path(__file__).resolve().parents[2]
    default_data_root = (
        project_root
        / "datasets"
        / "physionet_ct_ich"
        / "computed-tomography-images-for-intracranial-hemorrhage-detection-and-segmentation-1.0.0"
    )
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, default=default_data_root)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=project_root / "outputs" / "physionet_ct_ich" / "mean_pool_baseline",
    )
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--num-workers", type=int, default=8)
    parser.add_argument("--image-size", type=int, default=224)
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--recompute-features", action="store_true")
    return parser.parse_args()


def load_slice_rows(data_root: Path, labels: tuple[str, ...]) -> list[dict[str, object]]:
    csv_path = data_root / "hemorrhage_diagnosis.csv"
    image_root = data_root / "Patients_CT"
    rows: list[dict[str, object]] = []
    with csv_path.open("r", encoding="utf-8-sig", newline="") as file:
        for row in tqdm(csv.DictReader(file), desc="读取切片标签"):
            patient_id = int(row["PatientNumber"])
            slice_number = int(row["SliceNumber"])
            image_path = image_root / f"{patient_id:03d}" / "brain" / f"{slice_number}.jpg"
            if not image_path.is_file():
                raise FileNotFoundError(f"缺少脑窗切片：{image_path}")
            rows.append(
                {
                    "patient_id": patient_id,
                    "slice_number": slice_number,
                    "image_path": str(image_path),
                    "labels": [int(row[label]) for label in labels],
                }
            )
    rows.sort(key=lambda item: (int(item["patient_id"]), int(item["slice_number"])))
    return rows


def build_feature_extractor(device: torch.device, project_root: Path) -> torch.nn.Module:
    torch.hub.set_dir(str(project_root / "pre_weights"))
    model = models.convnext_tiny(weights=models.ConvNeXt_Tiny_Weights.IMAGENET1K_V1)
    extractor = torch.nn.Sequential(
        model.features,
        model.avgpool,
        model.classifier[0],
        torch.nn.Flatten(1),
    )
    extractor.eval().to(device)
    for parameter in extractor.parameters():
        parameter.requires_grad_(False)
    return extractor


def extract_features(
    rows: list[dict[str, object]],
    cache_path: Path,
    args: argparse.Namespace,
) -> dict[str, np.ndarray]:
    if cache_path.is_file() and not args.recompute_features:
        loaded = np.load(cache_path)
        return {key: loaded[key] for key in loaded.files}

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    project_root = Path(__file__).resolve().parents[2]
    extractor = build_feature_extractor(device=device, project_root=project_root)
    dataset = SliceDataset(rows=rows, image_size=args.image_size)
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
        persistent_workers=args.num_workers > 0,
    )

    feature_batches: list[np.ndarray] = []
    patient_batches: list[np.ndarray] = []
    slice_batches: list[np.ndarray] = []
    with torch.inference_mode():
        for images, patient_ids, slice_numbers in tqdm(loader, desc="提取ConvNeXt切片特征"):
            images = images.to(device, non_blocking=True)
            with torch.autocast(device_type=device.type, enabled=device.type == "cuda"):
                features = extractor(images)
            feature_batches.append(features.float().cpu().numpy())
            patient_batches.append(patient_ids.numpy())
            slice_batches.append(slice_numbers.numpy())

    payload = {
        "features": np.concatenate(feature_batches).astype(np.float32),
        "patient_ids": np.concatenate(patient_batches).astype(np.int16),
        "slice_numbers": np.concatenate(slice_batches).astype(np.int16),
    }
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(cache_path, **payload)
    return payload


def build_patient_bags(
    rows: list[dict[str, object]],
    feature_payload: dict[str, np.ndarray],
    labels: tuple[str, ...],
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    patient_ids = sorted({int(row["patient_id"]) for row in rows})
    slice_labels: dict[int, np.ndarray] = {
        patient_id: np.zeros(len(labels), dtype=np.int64) for patient_id in patient_ids
    }
    for row in rows:
        patient_id = int(row["patient_id"])
        slice_labels[patient_id] = np.maximum(
            slice_labels[patient_id], np.asarray(row["labels"], dtype=np.int64)
        )

    bag_features: list[np.ndarray] = []
    bag_labels: list[np.ndarray] = []
    cached_patient_ids = feature_payload["patient_ids"].astype(np.int64)
    for patient_id in tqdm(patient_ids, desc="构建检查级图像包"):
        patient_features = feature_payload["features"][cached_patient_ids == patient_id]
        if patient_features.size == 0:
            raise RuntimeError(f"患者 {patient_id} 没有缓存特征")
        bag_features.append(patient_features.mean(axis=0))
        bag_labels.append(slice_labels[patient_id])
    return (
        np.asarray(patient_ids, dtype=np.int64),
        np.stack(bag_features).astype(np.float32),
        np.stack(bag_labels).astype(np.int64),
    )


def multilabel_folds(y: np.ndarray, n_splits: int, seed: int) -> list[np.ndarray]:
    """以贪心方式平衡各折的标签阳性数和样本数。"""
    rng = random.Random(seed)
    label_frequency = y.sum(axis=0).clip(min=1)
    positive_indices = [index for index in range(len(y)) if y[index].sum() > 0]
    negative_indices = [index for index in range(len(y)) if y[index].sum() == 0]
    rng.shuffle(positive_indices)
    positive_indices.sort(
        key=lambda index: (
            -float((y[index] / label_frequency).sum()),
            -int(y[index].sum()),
        )
    )

    folds: list[list[int]] = [[] for _ in range(n_splits)]
    fold_label_counts = np.zeros((n_splits, y.shape[1]), dtype=np.float64)
    desired_labels = y.sum(axis=0) / float(n_splits)
    desired_size = len(y) / float(n_splits)

    for index in positive_indices:
        active = y[index].astype(bool)
        scores = []
        for fold_index in range(n_splits):
            label_deficit = (desired_labels[active] - fold_label_counts[fold_index, active]).sum()
            size_deficit = desired_size - len(folds[fold_index])
            scores.append((float(label_deficit), float(size_deficit), rng.random()))
        best_fold = max(range(n_splits), key=lambda fold_index: scores[fold_index])
        folds[best_fold].append(index)
        fold_label_counts[best_fold] += y[index]

    rng.shuffle(negative_indices)
    for index in negative_indices:
        smallest_size = min(len(fold) for fold in folds)
        candidates = [i for i, fold in enumerate(folds) if len(fold) == smallest_size]
        chosen = rng.choice(candidates)
        folds[chosen].append(index)

    return [np.asarray(sorted(fold), dtype=np.int64) for fold in folds]


def metric_dict(y_true: np.ndarray, probabilities: np.ndarray) -> dict[str, object]:
    predictions = (probabilities >= 0.5).astype(np.int64)
    precision, recall, per_label_f1, support = precision_recall_fscore_support(
        y_true,
        predictions,
        average=None,
        zero_division=0,
    )
    result: dict[str, object] = {
        "macro_f1": float(f1_score(y_true, predictions, average="macro", zero_division=0)),
        "micro_f1": float(f1_score(y_true, predictions, average="micro", zero_division=0)),
        "exact_match": float(accuracy_score(y_true, predictions)),
        "per_label_precision": precision.tolist(),
        "per_label_recall": recall.tolist(),
        "per_label_f1": per_label_f1.tolist(),
        "per_label_support": support.tolist(),
    }
    try:
        result["macro_auroc"] = float(roc_auc_score(y_true, probabilities, average="macro"))
    except ValueError:
        result["macro_auroc"] = None
    try:
        result["macro_auprc"] = float(average_precision_score(y_true, probabilities, average="macro"))
    except ValueError:
        result["macro_auprc"] = None
    return result


def run_cross_validation(
    patient_ids: np.ndarray,
    x: np.ndarray,
    y: np.ndarray,
    labels: tuple[str, ...],
    args: argparse.Namespace,
) -> dict[str, object]:
    fold_indices = multilabel_folds(y=y, n_splits=args.folds, seed=args.seed)
    oof_probabilities = np.zeros_like(y, dtype=np.float64)
    fold_results: list[dict[str, object]] = []
    all_indices = np.arange(len(y))

    for fold_number, test_indices in enumerate(tqdm(fold_indices, desc="五折患者级评估"), start=1):
        train_indices = np.setdiff1d(all_indices, test_indices)
        scaler = StandardScaler()
        x_train = scaler.fit_transform(x[train_indices])
        x_test = scaler.transform(x[test_indices])
        fold_probabilities = np.zeros((len(test_indices), len(labels)), dtype=np.float64)

        for label_index in range(len(labels)):
            classifier = LogisticRegression(
                C=1.0,
                class_weight="balanced",
                max_iter=2000,
                random_state=args.seed + fold_number * 10 + label_index,
                solver="liblinear",
            )
            classifier.fit(x_train, y[train_indices, label_index])
            fold_probabilities[:, label_index] = classifier.predict_proba(x_test)[:, 1]

        oof_probabilities[test_indices] = fold_probabilities
        fold_metrics = metric_dict(y[test_indices], fold_probabilities)
        fold_metrics.update(
            {
                "fold": fold_number,
                "patient_ids": patient_ids[test_indices].tolist(),
                "test_size": int(len(test_indices)),
                "test_positive_counts": y[test_indices].sum(axis=0).tolist(),
            }
        )
        fold_results.append(fold_metrics)

    overall = metric_dict(y, oof_probabilities)
    fold_macro_f1 = np.asarray([float(row["macro_f1"]) for row in fold_results])
    fold_micro_f1 = np.asarray([float(row["micro_f1"]) for row in fold_results])
    overall["fold_macro_f1_mean"] = float(fold_macro_f1.mean())
    overall["fold_macro_f1_std"] = float(fold_macro_f1.std(ddof=1))
    overall["fold_micro_f1_mean"] = float(fold_micro_f1.mean())
    overall["fold_micro_f1_std"] = float(fold_micro_f1.std(ddof=1))

    prediction_rows = []
    binary_predictions = (oof_probabilities >= 0.5).astype(np.int64)
    for row_index, patient_id in enumerate(patient_ids):
        row: dict[str, object] = {"patient_id": int(patient_id)}
        for label_index, label in enumerate(labels):
            name = DISPLAY_NAMES[label]
            row[f"true_{name}"] = int(y[row_index, label_index])
            row[f"prob_{name}"] = float(oof_probabilities[row_index, label_index])
            row[f"pred_{name}"] = int(binary_predictions[row_index, label_index])
        prediction_rows.append(row)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    prediction_path = args.output_dir / "oof_predictions.csv"
    with prediction_path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=list(prediction_rows[0].keys()))
        writer.writeheader()
        writer.writerows(prediction_rows)

    return {
        "experiment": "ConvNeXt-Tiny ImageNet frozen features + examination mean pooling + logistic head",
        "labels": [DISPLAY_NAMES[label] for label in labels],
        "num_patients": int(len(patient_ids)),
        "positive_patients": y.sum(axis=0).tolist(),
        "folds": fold_results,
        "overall": overall,
        "oof_predictions": str(prediction_path),
    }


def main() -> None:
    args = parse_args()
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)
    labels = DEFAULT_LABELS
    args.output_dir.mkdir(parents=True, exist_ok=True)

    rows = load_slice_rows(data_root=args.data_root, labels=labels)
    feature_payload = extract_features(
        rows=rows,
        cache_path=args.output_dir / "convnext_tiny_slice_features.npz",
        args=args,
    )
    patient_ids, bag_features, bag_labels = build_patient_bags(
        rows=rows,
        feature_payload=feature_payload,
        labels=labels,
    )
    results = run_cross_validation(
        patient_ids=patient_ids,
        x=bag_features,
        y=bag_labels,
        labels=labels,
        args=args,
    )
    result_path = args.output_dir / "results.json"
    result_path.write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")

    overall = results["overall"]
    print(f"结果已保存：{result_path}")
    print(
        "Macro-F1="
        f"{overall['fold_macro_f1_mean']:.4f} ± {overall['fold_macro_f1_std']:.4f}，"
        f"OOF Macro-F1={overall['macro_f1']:.4f}，"
        f"OOF Micro-F1={overall['micro_f1']:.4f}"
    )
    for label_name, f1_value in zip(results["labels"], overall["per_label_f1"]):
        print(f"{label_name} F1={f1_value:.4f}")


if __name__ == "__main__":
    main()
