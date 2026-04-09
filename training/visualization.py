from __future__ import annotations

import math
from pathlib import Path
from typing import Any

import numpy as np

try:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
except Exception:  # pragma: no cover
    matplotlib = None
    plt = None


def _ensure_parent(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)


def _available() -> bool:
    return plt is not None


def save_loss_curve(history: list[dict[str, Any]], path: Path) -> bool:
    if not _available() or not history:
        return False

    epochs = [int(item["epoch"]) for item in history]
    train_loss = [float(item["train_loss"]) for item in history]
    val_loss = [float(item["val_loss"]) for item in history]

    _ensure_parent(path)
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(epochs, train_loss, marker="o", linewidth=2, label="Train Loss")
    ax.plot(epochs, val_loss, marker="o", linewidth=2, label="Val Loss")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss")
    ax.set_title("Loss Curve")
    ax.grid(alpha=0.3, linestyle=":")
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)
    return True


def _render_single_confusion(ax, matrix: np.ndarray, labels: list[str], title: str) -> None:
    im = ax.imshow(matrix, cmap="Blues")
    ax.set_xticks(range(len(labels)))
    ax.set_yticks(range(len(labels)))
    ax.set_xticklabels(labels)
    ax.set_yticklabels(labels)
    ax.set_xlabel("Pred")
    ax.set_ylabel("True")
    ax.set_title(title)

    threshold = matrix.max() / 2.0 if matrix.size > 0 else 0.0
    for row in range(matrix.shape[0]):
        for col in range(matrix.shape[1]):
            color = "white" if matrix[row, col] > threshold else "black"
            ax.text(col, row, str(int(matrix[row, col])), ha="center", va="center", color=color, fontsize=10)

    return im


def save_confusion_matrix(metrics: dict[str, Any], path: Path, title: str) -> bool:
    if not _available():
        return False

    overall = metrics.get("confusion_matrix_overall")
    by_label = metrics.get("confusion_matrix_by_label", {})
    _ensure_parent(path)

    if isinstance(overall, dict) and isinstance(overall.get("matrix"), list):
        matrix = np.asarray(overall["matrix"], dtype=np.int64)
        labels = [str(item) for item in overall.get("class_names", ["0", "1"])]
        fig, ax = plt.subplots(figsize=(5.5, 4.8))
        _render_single_confusion(ax, matrix, labels, title)
        fig.tight_layout()
        fig.savefig(path, dpi=180)
        plt.close(fig)
        return True

    if not isinstance(by_label, dict) or not by_label:
        return False

    names = list(by_label.keys())
    cols = min(3, max(1, len(names)))
    rows = math.ceil(len(names) / cols)
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 4.6, rows * 4.2))
    axes_arr = np.asarray(axes).reshape(-1)

    for ax in axes_arr[len(names) :]:
        ax.axis("off")

    for ax, name in zip(axes_arr, names):
        current = by_label.get(name, {})
        matrix = np.asarray(
            [
                [int(current.get("tn", 0)), int(current.get("fp", 0))],
                [int(current.get("fn", 0)), int(current.get("tp", 0))],
            ],
            dtype=np.int64,
        )
        _render_single_confusion(ax, matrix, ["Neg", "Pos"], f"{title}\n{name}")

    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)
    return True


def _plot_curve(ax, x: list[float], y: list[float], label: str) -> None:
    if not x or not y:
        return
    ax.plot(x, y, linewidth=2, label=label)


def save_roc_curve(metrics: dict[str, Any], path: Path, title: str) -> bool:
    if not _available():
        return False

    curves = metrics.get("roc_curve_by_label")
    if not isinstance(curves, dict) or not curves:
        return False

    _ensure_parent(path)
    fig, ax = plt.subplots(figsize=(6.4, 5.2))
    plotted = False

    for label_name, curve in curves.items():
        if not isinstance(curve, dict) or not curve.get("available"):
            continue
        auc_value = metrics.get(f"roc_auc_{label_name}", float("nan"))
        legend = f"{label_name} (AUC={auc_value:.4f})" if not np.isnan(auc_value) else f"{label_name} (AUC=nan)"
        _plot_curve(ax, curve.get("fpr", []), curve.get("tpr", []), legend)
        plotted = True

    if not plotted:
        plt.close(fig)
        return False

    ax.plot([0.0, 1.0], [0.0, 1.0], linestyle="--", color="gray", linewidth=1)
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title(title)
    ax.grid(alpha=0.3, linestyle=":")
    ax.legend(fontsize=9)
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)
    return True


def save_pr_curve(metrics: dict[str, Any], path: Path, title: str) -> bool:
    if not _available():
        return False

    curves = metrics.get("pr_curve_by_label")
    if not isinstance(curves, dict) or not curves:
        return False

    _ensure_parent(path)
    fig, ax = plt.subplots(figsize=(6.4, 5.2))
    plotted = False

    for label_name, curve in curves.items():
        if not isinstance(curve, dict) or not curve.get("available"):
            continue
        ap_value = metrics.get(f"pr_auc_{label_name}", float("nan"))
        legend = f"{label_name} (AP={ap_value:.4f})" if not np.isnan(ap_value) else f"{label_name} (AP=nan)"
        _plot_curve(ax, curve.get("recall", []), curve.get("precision", []), legend)
        plotted = True

    if not plotted:
        plt.close(fig)
        return False

    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.set_title(title)
    ax.grid(alpha=0.3, linestyle=":")
    ax.legend(fontsize=9)
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)
    return True
