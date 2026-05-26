from __future__ import annotations

from typing import Any

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    cohen_kappa_score,
    confusion_matrix,
    f1_score,
    hamming_loss,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_curve,
    roc_auc_score,
)


def _safe_auc(y_true: np.ndarray, y_prob: np.ndarray) -> float:
    if len(np.unique(y_true)) < 2:
        return float("nan")
    return float(roc_auc_score(y_true, y_prob))


def _safe_ap(y_true: np.ndarray, y_prob: np.ndarray) -> float:
    if y_true.sum() == 0:
        return float("nan")
    return float(average_precision_score(y_true, y_prob))


def _safe_kappa(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    if y_true.size == 0:
        return float("nan")
    if len(np.unique(y_true)) < 2 and len(np.unique(y_pred)) < 2:
        return 1.0 if np.array_equal(y_true, y_pred) else 0.0
    score = float(cohen_kappa_score(y_true, y_pred))
    return 0.0 if np.isnan(score) else score


def _safe_roc_curve(y_true: np.ndarray, y_prob: np.ndarray) -> dict[str, Any]:
    if len(np.unique(y_true)) < 2:
        return {"available": False, "fpr": [], "tpr": []}
    fpr, tpr, _ = roc_curve(y_true, y_prob)
    return {
        "available": True,
        "fpr": fpr.tolist(),
        "tpr": tpr.tolist(),
    }


def _safe_pr_curve(y_true: np.ndarray, y_prob: np.ndarray) -> dict[str, Any]:
    if y_true.sum() == 0:
        return {"available": False, "recall": [], "precision": []}
    precision, recall, _ = precision_recall_curve(y_true, y_prob)
    return {
        "available": True,
        "recall": recall.tolist(),
        "precision": precision.tolist(),
    }


def _nanmean(values: list[float]) -> float:
    if not values:
        return float("nan")
    arr = np.asarray(values, dtype=np.float64)
    if np.isnan(arr).all():
        return float("nan")
    return float(np.nanmean(arr))


def _confusion_details(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, int]:
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    if cm.shape != (2, 2):
        return {"tn": 0, "fp": 0, "fn": 0, "tp": 0}
    tn, fp, fn, tp = cm.ravel()
    return {
        "tn": int(tn),
        "fp": int(fp),
        "fn": int(fn),
        "tp": int(tp),
    }


def _binary_label_metrics(y_true: np.ndarray, y_prob: np.ndarray, y_pred: np.ndarray) -> dict[str, Any]:
    confusion = _confusion_details(y_true, y_pred)
    tn = confusion["tn"]
    fp = confusion["fp"]
    fn = confusion["fn"]
    tp = confusion["tp"]

    recall = float(tp / (tp + fn)) if (tp + fn) > 0 else 0.0
    precision = float(tp / (tp + fp)) if (tp + fp) > 0 else 0.0
    specificity = float(tn / (tn + fp)) if (tn + fp) > 0 else 0.0
    f1 = float(f1_score(y_true, y_pred, zero_division=0))
    accuracy = float(accuracy_score(y_true, y_pred))
    roc_auc = _safe_auc(y_true, y_prob)
    pr_auc = _safe_ap(y_true, y_prob)

    return {
        "label_wise_acc": accuracy,
        "recall": recall,
        "precision": precision,
        "specificity": specificity,
        "f1": f1,
        "roc_auc": roc_auc,
        "pr_auc": pr_auc,
        "confusion": confusion,
        "roc_curve": _safe_roc_curve(y_true, y_prob),
        "pr_curve": _safe_pr_curve(y_true, y_prob),
    }


def compute_multilabel_metrics(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    label_names: list[str],
    threshold: float | list[float] | np.ndarray = 0.5,
) -> dict[str, Any]:
    if np.isscalar(threshold):
        threshold_array = np.full((y_prob.shape[1],), float(threshold), dtype=np.float32)
    else:
        threshold_array = np.asarray(threshold, dtype=np.float32).reshape(-1)
        if threshold_array.shape[0] != y_prob.shape[1]:
            raise ValueError("多标签阈值数量必须与标签数量一致")

    y_pred = (y_prob >= threshold_array.reshape(1, -1)).astype(np.int64)

    acc_list: list[float] = []
    recall_list: list[float] = []
    precision_list: list[float] = []
    specificity_list: list[float] = []
    f1_list: list[float] = []
    auc_list: list[float] = []
    ap_list: list[float] = []
    metrics: dict[str, Any] = {}
    confusion_by_label: dict[str, Any] = {}
    roc_curve_by_label: dict[str, Any] = {}
    pr_curve_by_label: dict[str, Any] = {}

    for index, label_name in enumerate(label_names):
        label_true = y_true[:, index]
        label_prob = y_prob[:, index]
        label_pred = y_pred[:, index]

        label_metrics = _binary_label_metrics(label_true, label_prob, label_pred)

        acc_list.append(label_metrics["label_wise_acc"])
        recall_list.append(label_metrics["recall"])
        precision_list.append(label_metrics["precision"])
        specificity_list.append(label_metrics["specificity"])
        f1_list.append(label_metrics["f1"])
        auc_list.append(label_metrics["roc_auc"])
        ap_list.append(label_metrics["pr_auc"])

        metrics[f"label_wise_acc_{label_name}"] = label_metrics["label_wise_acc"]
        metrics[f"recall_{label_name}"] = label_metrics["recall"]
        metrics[f"precision_{label_name}"] = label_metrics["precision"]
        metrics[f"specificity_{label_name}"] = label_metrics["specificity"]
        metrics[f"f1_{label_name}"] = label_metrics["f1"]
        metrics[f"roc_auc_{label_name}"] = label_metrics["roc_auc"]
        metrics[f"pr_auc_{label_name}"] = label_metrics["pr_auc"]
        metrics[f"threshold_{label_name}"] = float(threshold_array[index])

        confusion_by_label[label_name] = label_metrics["confusion"]
        roc_curve_by_label[label_name] = label_metrics["roc_curve"]
        pr_curve_by_label[label_name] = label_metrics["pr_curve"]

    flat_true = y_true.reshape(-1)
    flat_pred = y_pred.reshape(-1)

    metrics["label_wise_acc_mean"] = _nanmean(acc_list)
    metrics["macro_recall"] = _nanmean(recall_list)
    metrics["macro_precision"] = _nanmean(precision_list)
    metrics["macro_specificity"] = _nanmean(specificity_list)
    metrics["macro_f1"] = _nanmean(f1_list)
    metrics["macro_roc_auc"] = _nanmean(auc_list)
    metrics["macro_pr_auc"] = _nanmean(ap_list)
    metrics["macro_auc"] = metrics["macro_roc_auc"]
    metrics["macro_ap"] = metrics["macro_pr_auc"]
    metrics["micro_recall"] = float(recall_score(flat_true, flat_pred, zero_division=0))
    metrics["micro_precision"] = float(precision_score(flat_true, flat_pred, zero_division=0))
    metrics["micro_f1"] = float(f1_score(flat_true, flat_pred, zero_division=0))
    metrics["subset_accuracy"] = float(np.mean(np.all(y_pred == y_true, axis=1)))
    metrics["hamming_loss"] = float(hamming_loss(y_true, y_pred))
    metrics["kappa"] = _safe_kappa(flat_true, flat_pred)
    metrics["threshold_mean"] = float(np.mean(threshold_array))
    metrics["confusion_matrix_by_label"] = confusion_by_label
    metrics["roc_curve_by_label"] = roc_curve_by_label
    metrics["pr_curve_by_label"] = pr_curve_by_label
    return metrics


def compute_binary_metrics(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    class_names: list[str] | None = None,
    threshold: float = 0.5,
) -> dict[str, Any]:
    resolved_class_names = class_names or ["negative", "positive"]
    if len(resolved_class_names) != 2:
        raise ValueError("二分类任务的 class_names 必须包含 2 个类别名")

    y_pred = (y_prob >= threshold).astype(np.int64)
    prob_by_class = np.stack([1.0 - y_prob, y_prob], axis=1)
    metrics: dict[str, Any] = {}

    acc_list: list[float] = []
    recall_list: list[float] = []
    precision_list: list[float] = []
    specificity_list: list[float] = []
    f1_list: list[float] = []
    auc_list: list[float] = []
    ap_list: list[float] = []
    confusion_by_label: dict[str, Any] = {}
    roc_curve_by_label: dict[str, Any] = {}
    pr_curve_by_label: dict[str, Any] = {}

    for class_index, class_name in enumerate(resolved_class_names):
        class_true = (y_true == class_index).astype(np.int64)
        class_pred = (y_pred == class_index).astype(np.int64)
        class_prob = prob_by_class[:, class_index]

        class_metrics = _binary_label_metrics(class_true, class_prob, class_pred)
        acc_list.append(class_metrics["label_wise_acc"])
        recall_list.append(class_metrics["recall"])
        precision_list.append(class_metrics["precision"])
        specificity_list.append(class_metrics["specificity"])
        f1_list.append(class_metrics["f1"])
        auc_list.append(class_metrics["roc_auc"])
        ap_list.append(class_metrics["pr_auc"])

        metrics[f"label_wise_acc_{class_name}"] = class_metrics["label_wise_acc"]
        metrics[f"recall_{class_name}"] = class_metrics["recall"]
        metrics[f"precision_{class_name}"] = class_metrics["precision"]
        metrics[f"specificity_{class_name}"] = class_metrics["specificity"]
        metrics[f"f1_{class_name}"] = class_metrics["f1"]
        metrics[f"roc_auc_{class_name}"] = class_metrics["roc_auc"]
        metrics[f"pr_auc_{class_name}"] = class_metrics["pr_auc"]

        confusion_by_label[class_name] = class_metrics["confusion"]
        roc_curve_by_label[class_name] = class_metrics["roc_curve"]
        pr_curve_by_label[class_name] = class_metrics["pr_curve"]

    accuracy = float(accuracy_score(y_true, y_pred))
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    tn, fp, fn, tp = cm.ravel() if cm.shape == (2, 2) else (0, 0, 0, 0)
    positive_name = resolved_class_names[1]

    metrics["label_wise_acc_mean"] = _nanmean(acc_list)
    metrics["macro_recall"] = _nanmean(recall_list)
    metrics["macro_precision"] = _nanmean(precision_list)
    metrics["macro_specificity"] = _nanmean(specificity_list)
    metrics["macro_f1"] = _nanmean(f1_list)
    metrics["macro_roc_auc"] = _nanmean(auc_list)
    metrics["macro_pr_auc"] = _nanmean(ap_list)
    metrics["macro_auc"] = metrics["macro_roc_auc"]
    metrics["macro_ap"] = metrics["macro_pr_auc"]
    metrics["micro_recall"] = float(recall_score(y_true, y_pred, average="micro", zero_division=0))
    metrics["micro_precision"] = float(precision_score(y_true, y_pred, average="micro", zero_division=0))
    metrics["micro_f1"] = float(f1_score(y_true, y_pred, average="micro", zero_division=0))
    metrics["accuracy"] = accuracy
    metrics["auc"] = metrics[f"roc_auc_{positive_name}"]
    metrics["ap"] = metrics[f"pr_auc_{positive_name}"]
    metrics["f1"] = metrics[f"f1_{positive_name}"]
    metrics["sensitivity"] = metrics[f"recall_{positive_name}"]
    metrics["specificity"] = metrics[f"specificity_{positive_name}"]
    metrics["hamming_loss"] = float(hamming_loss(y_true, y_pred))
    metrics["kappa"] = _safe_kappa(y_true, y_pred)
    metrics["confusion_matrix"] = {
        "tn": int(tn),
        "fp": int(fp),
        "fn": int(fn),
        "tp": int(tp),
    }
    metrics["confusion_matrix_overall"] = {
        "class_names": resolved_class_names,
        "matrix": cm.astype(int).tolist(),
    }
    metrics["confusion_matrix_by_label"] = confusion_by_label
    metrics["roc_curve_by_label"] = roc_curve_by_label
    metrics["pr_curve_by_label"] = pr_curve_by_label

    return metrics


def to_builtin_type(obj: Any) -> Any:
    if isinstance(obj, dict):
        return {key: to_builtin_type(value) for key, value in obj.items()}
    if isinstance(obj, list):
        return [to_builtin_type(value) for value in obj]
    if isinstance(obj, tuple):
        return [to_builtin_type(value) for value in obj]
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, np.generic):
        return obj.item()
    return obj
