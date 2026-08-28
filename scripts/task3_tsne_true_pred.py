#!/usr/bin/env python3
"""将 TASK3 标签级融合特征绘制为真实分布图和预测分布图。"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.preprocessing import StandardScaler


DATASET_DISPLAY_NAMES = {
    "regular_white_light": "White-light endoscopy",
    "chromoscopic": "Chromoscopic endoscopy",
    "surgical": "Surgical endoscopy",
    "ultrasound": "Ultrasound endoscopy",
}
LABEL_DISPLAY_NAMES = (
    "Esophageal SMT",
    "Esophageal mucosal lesion/tumor",
    "Gastritis",
)
LABEL_COLORS = ("#4C78A8", "#E45756", "#54A24B")
STATUS_MARKERS = {0: "X", 1: "o"}
STATUS_NAMES = {0: "Negative", 1: "Positive"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--feature-dir",
        type=Path,
        required=True,
        help="包含四个 *_fold1_features.npz 文件的目录",
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--fold", type=int, default=1)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--max-iter", type=int, default=2000)
    return parser.parse_args()


def compute_label_tsne(
    label_features: np.ndarray,
    *,
    seed: int,
    max_iter: int,
) -> tuple[np.ndarray, dict[str, Any]]:
    if label_features.ndim != 3 or label_features.shape[1] != len(LABEL_DISPLAY_NAMES):
        raise ValueError(
            "标签级融合特征应为 [样本数, 3, 特征维度]，"
            f"实际形状为 {label_features.shape}"
        )
    flattened = label_features.reshape(-1, label_features.shape[-1])
    standardized = StandardScaler().fit_transform(flattened)
    pca_components = min(50, standardized.shape[1], standardized.shape[0] - 1)
    reduced = PCA(n_components=pca_components, random_state=seed).fit_transform(standardized)
    perplexity = min(30.0, max(5.0, float((len(reduced) - 1) // 3)))
    embedding = TSNE(
        n_components=2,
        perplexity=perplexity,
        learning_rate="auto",
        init="pca",
        max_iter=int(max_iter),
        random_state=int(seed),
    ).fit_transform(reduced)
    return embedding, {
        "points": int(len(flattened)),
        "feature_dim": int(flattened.shape[1]),
        "pca_components": int(pca_components),
        "perplexity": float(perplexity),
        "max_iter": int(max_iter),
        "seed": int(seed),
    }


def plot_distribution(
    *,
    dataset_name: str,
    embedding: np.ndarray,
    statuses: np.ndarray,
    distribution_name: str,
    output_path: Path,
) -> None:
    label_indices = np.tile(np.arange(len(LABEL_DISPLAY_NAMES)), len(statuses))
    flat_statuses = statuses.astype(int).reshape(-1)
    if embedding.shape[0] != len(flat_statuses):
        raise ValueError(
            f"坐标与状态数量不一致：embedding={embedding.shape[0]}，status={len(flat_statuses)}"
        )

    fig, ax = plt.subplots(figsize=(9.0, 7.6), constrained_layout=True)
    for label_index, color in enumerate(LABEL_COLORS):
        for status in (0, 1):
            selected = (label_indices == label_index) & (flat_statuses == status)
            marker = STATUS_MARKERS[status]
            ax.scatter(
                embedding[selected, 0],
                embedding[selected, 1],
                s=44 if status == 1 else 48,
                c=color,
                marker=marker,
                alpha=0.76,
                linewidths=0.45,
                edgecolors="white" if status == 1 else color,
                rasterized=True,
            )

    class_handles = [
        Line2D(
            [0],
            [0],
            marker="o",
            linestyle="",
            markerfacecolor=color,
            markeredgecolor="white",
            markersize=8,
            label=label_name,
        )
        for label_name, color in zip(LABEL_DISPLAY_NAMES, LABEL_COLORS)
    ]
    status_handles = [
        Line2D(
            [0],
            [0],
            marker=STATUS_MARKERS[status],
            linestyle="",
            markerfacecolor="#666666",
            markeredgecolor="#666666",
            markersize=8,
            label=STATUS_NAMES[status],
        )
        for status in (1, 0)
    ]
    class_legend = ax.legend(
        handles=class_handles,
        title="Label class (color)",
        loc="upper left",
        fontsize=9,
        title_fontsize=9,
        frameon=True,
    )
    ax.add_artist(class_legend)
    ax.legend(
        handles=status_handles,
        title="Status (marker)",
        loc="upper right",
        fontsize=9,
        title_fontsize=9,
        frameon=True,
    )

    ax.set_title(
        f"{DATASET_DISPLAY_NAMES[dataset_name]}\n{distribution_name}",
        fontsize=15,
        fontweight="bold",
        pad=12,
    )
    ax.set_xlabel("t-SNE 1", fontsize=11)
    ax.set_ylabel("t-SNE 2", fontsize=11)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.grid(False)
    for spine in ax.spines.values():
        spine.set_color("#CFCFCF")
        spine.set_linewidth(0.8)
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    args = parse_args()
    feature_dir = args.feature_dir.expanduser().resolve()
    output_dir = args.output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    metadata: dict[str, Any] = {
        "fold": int(args.fold),
        "coordinate_definition": (
            "每个检查的三个标签级融合特征分别作为一个点；"
            "同一数据集的真实分布图与预测分布图共用完全相同的t-SNE坐标。"
        ),
        "color_definition": {
            LABEL_DISPLAY_NAMES[index]: LABEL_COLORS[index]
            for index in range(len(LABEL_DISPLAY_NAMES))
        },
        "marker_definition": {
            STATUS_NAMES[status]: STATUS_MARKERS[status]
            for status in (0, 1)
        },
        "prediction_threshold": 0.5,
        "datasets": {},
    }

    for dataset_name in DATASET_DISPLAY_NAMES:
        feature_path = feature_dir / f"{dataset_name}_fold{args.fold}_features.npz"
        if not feature_path.is_file():
            raise FileNotFoundError(f"缺少融合特征文件：{feature_path}")
        with np.load(feature_path) as loaded:
            required = {"label_features", "labels", "probabilities"}
            missing = required - set(loaded.files)
            if missing:
                raise ValueError(f"{feature_path} 缺少字段：{sorted(missing)}")
            label_features = loaded["label_features"]
            labels = loaded["labels"].astype(int)
            probabilities = loaded["probabilities"]

        if labels.shape != probabilities.shape or labels.shape[1] != len(LABEL_DISPLAY_NAMES):
            raise ValueError(
                f"{dataset_name} 标签或预测形状错误："
                f"labels={labels.shape}，probabilities={probabilities.shape}"
            )
        predictions = (probabilities >= 0.5).astype(int)
        embedding, tsne_metadata = compute_label_tsne(
            label_features,
            seed=int(args.seed),
            max_iter=int(args.max_iter),
        )

        true_path = output_dir / f"{dataset_name}_true_distribution.png"
        prediction_path = output_dir / f"{dataset_name}_predicted_distribution.png"
        plot_distribution(
            dataset_name=dataset_name,
            embedding=embedding,
            statuses=labels,
            distribution_name="Ground-truth distribution",
            output_path=true_path,
        )
        plot_distribution(
            dataset_name=dataset_name,
            embedding=embedding,
            statuses=predictions,
            distribution_name="Model-predicted distribution",
            output_path=prediction_path,
        )
        metadata["datasets"][dataset_name] = {
            "samples": int(labels.shape[0]),
            "points": int(labels.size),
            "true_positive_counts": {
                LABEL_DISPLAY_NAMES[index]: int(labels[:, index].sum())
                for index in range(len(LABEL_DISPLAY_NAMES))
            },
            "predicted_positive_counts": {
                LABEL_DISPLAY_NAMES[index]: int(predictions[:, index].sum())
                for index in range(len(LABEL_DISPLAY_NAMES))
            },
            "tsne": tsne_metadata,
            "true_plot": str(true_path),
            "predicted_plot": str(prediction_path),
        }
        print(
            f"[t-SNE] {dataset_name}: 样本={labels.shape[0]}，点数={labels.size}，"
            f"已生成真实分布图和预测分布图"
        )

    (output_dir / "plot_metadata.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(f"[t-SNE] 新版8张图已生成：{output_dir}")


if __name__ == "__main__":
    main()
