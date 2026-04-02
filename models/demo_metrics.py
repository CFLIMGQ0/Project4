from __future__ import annotations

from typing import Any

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    confusion_matrix,
    f1_score,
    hamming_loss,
    matthews_corrcoef,
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


def _nanmean(values: list[float]) -> float:
    if not values:
        return float("nan")
    arr = np.array(values, dtype=np.float64)
    if np.isnan(arr).all():
        return float("nan")
    return float(np.nanmean(arr))


def compute_multilabel_metrics(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    label_names: list[str],
    threshold: float = 0.5,
) -> dict[str, Any]:
    y_pred = (y_prob >= threshold).astype(np.int64)

    per_label_auc: dict[str, float] = {}
    per_label_ap: dict[str, float] = {}
    per_label_f1: dict[str, float] = {}
    auc_list: list[float] = []
    ap_list: list[float] = []
    f1_list: list[float] = []

    for idx, label in enumerate(label_names):
        yt = y_true[:, idx]
        yp = y_prob[:, idx]
        yhat = y_pred[:, idx]
        auc = _safe_auc(yt, yp)
        ap = _safe_ap(yt, yp)
        f1 = float(f1_score(yt, yhat, zero_division=0))

        per_label_auc[label] = auc
        per_label_ap[label] = ap
        per_label_f1[label] = f1
        auc_list.append(auc)
        ap_list.append(ap)
        f1_list.append(f1)

    metrics: dict[str, Any] = {
        "per_label_auc": per_label_auc,
        "per_label_ap": per_label_ap,
        "per_label_f1": per_label_f1,
        "macro_auc": _nanmean(auc_list),
        "macro_ap": _nanmean(ap_list),
        "macro_f1": _nanmean(f1_list),
        "subset_accuracy": float(np.mean(np.all(y_pred == y_true, axis=1))),
        "hamming_loss": float(hamming_loss(y_true, y_pred)),
    }

    # 同时给出扁平化字段，便于 CSV 记录。
    for label in label_names:
        metrics[f"auc_{label}"] = per_label_auc[label]
        metrics[f"ap_{label}"] = per_label_ap[label]
        metrics[f"f1_{label}"] = per_label_f1[label]

    return metrics


def compute_binary_metrics(y_true: np.ndarray, y_prob: np.ndarray, threshold: float = 0.5) -> dict[str, Any]:
    y_pred = (y_prob >= threshold).astype(np.int64)
    auc = _safe_auc(y_true, y_prob)
    ap = _safe_ap(y_true, y_prob)
    f1 = float(f1_score(y_true, y_pred, zero_division=0))
    acc = float(accuracy_score(y_true, y_pred))

    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    if cm.shape == (2, 2):
        tn, fp, fn, tp = cm.ravel()
    else:
        tn = fp = fn = tp = 0

    sensitivity = float(tp / (tp + fn)) if (tp + fn) > 0 else 0.0
    specificity = float(tn / (tn + fp)) if (tn + fp) > 0 else 0.0
    mcc = float(matthews_corrcoef(y_true, y_pred)) if len(np.unique(y_true)) > 1 else 0.0

    return {
        "auc": auc,
        "ap": ap,
        "f1": f1,
        "accuracy": acc,
        "sensitivity": sensitivity,
        "specificity": specificity,
        "mcc": mcc,
        "confusion_matrix": {
            "tn": int(tn),
            "fp": int(fp),
            "fn": int(fn),
            "tp": int(tp),
        },
    }


def compute_multiclass_metrics(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    class_names: list[str],
) -> dict[str, Any]:
    y_pred = np.argmax(y_prob, axis=1)
    acc = float(accuracy_score(y_true, y_pred))
    macro_f1 = float(f1_score(y_true, y_pred, average="macro", zero_division=0))
    weighted_f1 = float(f1_score(y_true, y_pred, average="weighted", zero_division=0))

    metrics: dict[str, Any] = {
        "accuracy": acc,
        "macro_f1": macro_f1,
        "weighted_f1": weighted_f1,
    }

    for idx, cname in enumerate(class_names):
        yt = (y_true == idx).astype(np.int64)
        yp = y_prob[:, idx]
        yhat = (y_pred == idx).astype(np.int64)
        metrics[f"auc_{cname}"] = _safe_auc(yt, yp)
        metrics[f"ap_{cname}"] = _safe_ap(yt, yp)
        metrics[f"f1_{cname}"] = float(f1_score(yt, yhat, zero_division=0))

    return metrics


def to_builtin_type(obj: Any) -> Any:
    if isinstance(obj, dict):
        return {k: to_builtin_type(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [to_builtin_type(v) for v in obj]
    if isinstance(obj, np.generic):
        return obj.item()
    return obj
