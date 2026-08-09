#!/usr/bin/env python3
"""统计并绘制 TASK3 四个数据集的标签共现矩阵。"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import yaml
from tqdm import tqdm


SRC_ROOT = Path(__file__).resolve().parents[1]
PROJECT_ROOT = SRC_ROOT.parent
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from train import build_multilabel_minority_balance


DEFAULT_DATALIST = PROJECT_ROOT / "datasets/task_data/task2/gastro_multilabel_task_datalist.csv"
DEFAULT_CONFIG = SRC_ROOT / "configs/task3/t3_main_model.yaml"
DEFAULT_OUTPUT_DIR = SRC_ROOT / "figs"
DEFAULT_FOLD_ROOT = PROJECT_ROOT / "outputs/train_runs/task3/t3_main_model"

LABELS = (
    ("Esophageal SMT", "label_esophageal_smt"),
    ("Esophageal mucosal lesion", "label_esophageal_mucosal_or_tumor"),
    ("Gastritis", "label_gastritis"),
)

DATASET_TITLES = {
    "regular_white_light": "WLE",
    "chromoscopic": "Chromoendoscopy",
    "surgical": "Surgical Gastroscopy",
    "ultrasound": "Endoscopic Ultrasonography",
}

DATASET_SHORT_TITLES = {
    "regular_white_light": "WLE",
    "chromoscopic": "Chromoendoscopy",
    "surgical": "Surgical Gastroscopy",
    "ultrasound": "EUS",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--datalist", type=Path, default=DEFAULT_DATALIST)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--fold-root", type=Path, default=DEFAULT_FOLD_ROOT)
    return parser.parse_args()


def read_yaml(path: Path) -> dict[str, Any]:
    payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"配置文件顶层必须为字典：{path}")
    return payload


def load_dataset_labels(
    datalist_path: Path,
    config_path: Path,
) -> dict[str, np.ndarray]:
    config = read_yaml(config_path)
    dataset_config = config.get("datasets", {})
    missing_datasets = set(DATASET_TITLES) - set(dataset_config)
    if missing_datasets:
        raise ValueError(f"TASK3配置缺少数据集：{sorted(missing_datasets)}")

    title_to_dataset: dict[str, str] = {}
    for dataset_name in DATASET_TITLES:
        for report_title in dataset_config[dataset_name]["report_titles"]:
            normalized = str(report_title).strip()
            if normalized in title_to_dataset:
                raise ValueError(f"检查标题被重复分配：{normalized}")
            title_to_dataset[normalized] = dataset_name

    with datalist_path.open("r", encoding="utf-8-sig", newline="") as file:
        reader = csv.DictReader(file)
        required = {"reportTitle", *(column for _, column in LABELS)}
        missing_columns = required - set(reader.fieldnames or [])
        if missing_columns:
            raise ValueError(f"数据表缺少字段：{sorted(missing_columns)}")
        rows = list(reader)

    grouped: dict[str, list[list[int]]] = {name: [] for name in DATASET_TITLES}
    for row in tqdm(rows, desc="统计四个数据集标签", unit="次检查"):
        dataset_name = title_to_dataset.get(str(row["reportTitle"]).strip())
        if dataset_name is None:
            continue
        grouped[dataset_name].append(
            [int(float(row[column])) for _, column in LABELS]
        )

    arrays: dict[str, np.ndarray] = {}
    for dataset_name, values in grouped.items():
        if not values:
            raise ValueError(f"数据集没有匹配记录：{dataset_name}")
        array = np.asarray(values, dtype=np.int64)
        if not np.isin(array, (0, 1)).all():
            raise ValueError(f"数据集标签不是二值：{dataset_name}")
        arrays[dataset_name] = array
    return arrays


def compute_cooccurrence(label_array: np.ndarray) -> np.ndarray:
    return label_array.T @ label_array


def load_balanced_training_labels(
    fold_root: Path,
    config_path: Path,
) -> tuple[dict[str, np.ndarray], dict[str, list[np.ndarray]]]:
    config = read_yaml(config_path)
    class_balance = dict(config["class_balance"])
    label_names = tuple(column for _, column in LABELS)
    grouped: dict[str, list[np.ndarray]] = {name: [] for name in DATASET_TITLES}

    jobs = [
        (dataset_name, fold)
        for dataset_name in DATASET_TITLES
        for fold in range(1, int(config["folds"]) + 1)
    ]
    for dataset_name, fold in tqdm(jobs, desc="重建五折重平衡训练曝光", unit="折"):
        run_dir = fold_root / dataset_name / f"fold_{fold}"
        manifest_path = run_dir / "split_manifest.csv"
        report_path = run_dir / "class_balance_report.json"
        with manifest_path.open("r", encoding="utf-8-sig", newline="") as file:
            rows = [row for row in csv.DictReader(file) if row["split"] == "train"]
        train_records = [
            {
                "patient_id": row["patient_id"],
                "exam_dir": row["exam_dir"],
                "labels": [int(float(row[column])) for _, column in LABELS],
            }
            for row in rows
        ]
        balanced_records, generated_report = build_multilabel_minority_balance(
            train_records=train_records,
            label_names=label_names,
            cfg=class_balance,
            seed=int(config["seed"]) + fold,
        )
        saved_report = json.loads(report_path.read_text(encoding="utf-8"))
        for key in ("original_train_size", "balanced_train_size", "added_records", "after"):
            if generated_report[key] != saved_report[key]:
                raise RuntimeError(
                    f"重建结果与训练报告不一致：{dataset_name}/fold_{fold}/{key}"
                )
        grouped[dataset_name].append(
            np.asarray([record["labels"] for record in balanced_records], dtype=np.int64)
        )

    cumulative = {
        dataset_name: np.concatenate(fold_arrays, axis=0)
        for dataset_name, fold_arrays in grouped.items()
    }
    return cumulative, grouped


def save_matrix_csv(
    matrix: np.ndarray,
    total: int,
    output_path: Path,
    total_name: str = "total_examinations",
) -> None:
    with output_path.with_suffix(".csv").open("w", encoding="utf-8", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(["label", *[name for name, _ in LABELS]])
        for label_name, row in zip([name for name, _ in LABELS], matrix.tolist()):
            writer.writerow([label_name, *row])
        writer.writerow([total_name, total, "", ""])


def plot_matrix(
    matrix: np.ndarray,
    total: int,
    figure_title: str,
    output_path: Path,
    total_name: str = "total_examinations",
) -> None:
    display_labels = ["Esophageal\nSMT", "Esophageal mucosal\nlesion", "Gastritis"]
    fig, ax = plt.subplots(figsize=(7.6, 6.2), dpi=300)
    image = ax.imshow(matrix, cmap="Greens", vmin=0, vmax=max(1, int(matrix.max())))

    ax.set_title(
        figure_title,
        fontsize=17,
        fontweight="bold",
        pad=16,
    )
    ax.set_xticks(np.arange(len(display_labels)))
    ax.set_yticks(np.arange(len(display_labels)))
    ax.set_xticklabels(display_labels, fontsize=11)
    ax.set_yticklabels(display_labels, fontsize=11)
    ax.set_xlabel("Label", fontsize=12, labelpad=10)
    ax.set_ylabel("Label", fontsize=12, labelpad=10)

    for row_index in range(matrix.shape[0]):
        for column_index in range(matrix.shape[1]):
            count = int(matrix[row_index, column_index])
            percent = 100.0 * count / total
            descriptor = "Positive" if row_index == column_index else "Co-positive"
            color = "white" if count > matrix.max() * 0.55 else "black"
            ax.text(
                column_index,
                row_index,
                f"{count}\n({percent:.1f}%)\n{descriptor}",
                ha="center",
                va="center",
                fontsize=10,
                fontweight="bold",
                color=color,
            )

    ax.set_xticks(np.arange(-0.5, matrix.shape[1], 1), minor=True)
    ax.set_yticks(np.arange(-0.5, matrix.shape[0], 1), minor=True)
    ax.grid(which="minor", color="white", linestyle="-", linewidth=2.0)
    ax.tick_params(which="minor", bottom=False, left=False)
    ax.tick_params(axis="both", length=0)

    colorbar = fig.colorbar(image, ax=ax, fraction=0.046, pad=0.04)
    colorbar.set_label("Number of examinations", fontsize=11)
    colorbar.ax.tick_params(labelsize=10)

    fig.tight_layout()
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)
    save_matrix_csv(matrix, total, output_path, total_name=total_name)


def plot_overview(
    original_labels: dict[str, np.ndarray],
    balanced_labels: dict[str, np.ndarray],
    output_path: Path,
) -> None:
    dataset_order = tuple(DATASET_TITLES.keys())
    column_titles = ("WLE", "Chromoendoscopy", "Surgical", "EUS")
    row_payloads = (("Original", original_labels), ("Rebalanced", balanced_labels))
    tick_labels = ("SMT", "EML", "Gastritis")

    fig, axes = plt.subplots(2, 4, figsize=(15, 8), dpi=300)
    image = None
    for row_index, (row_title, payload) in enumerate(row_payloads):
        for column_index, (dataset_name, column_title) in enumerate(
            zip(dataset_order, column_titles)
        ):
            ax = axes[row_index, column_index]
            labels = payload[dataset_name]
            matrix = compute_cooccurrence(labels)
            percentages = matrix.astype(np.float64) * 100.0 / labels.shape[0]
            image = ax.imshow(percentages, cmap="Greens", vmin=0, vmax=100)
            if row_index == 0:
                ax.set_title(column_title, fontsize=18, fontweight="bold", pad=10)
            ax.set_xticks(np.arange(3))
            ax.set_yticks(np.arange(3))
            ax.set_xticklabels(tick_labels, fontsize=13, rotation=35, ha="right")
            ax.set_yticklabels(tick_labels if column_index == 0 else ("", "", ""), fontsize=13)
            for matrix_row in range(3):
                for matrix_column in range(3):
                    count = int(matrix[matrix_row, matrix_column])
                    percent = float(percentages[matrix_row, matrix_column])
                    ax.text(
                        matrix_column,
                        matrix_row,
                        f"{count}\n{percent:.1f}%",
                        ha="center",
                        va="center",
                        fontsize=13,
                        fontweight="bold",
                        color="white" if percent > 48 else "black",
                    )
            ax.set_xticks(np.arange(-0.5, 3, 1), minor=True)
            ax.set_yticks(np.arange(-0.5, 3, 1), minor=True)
            ax.grid(which="minor", color="white", linewidth=1.5)
            ax.tick_params(which="minor", bottom=False, left=False)
            ax.tick_params(axis="both", length=0)
        axes[row_index, 0].set_ylabel(row_title, fontsize=18, fontweight="bold", labelpad=12)

    if image is None:
        raise RuntimeError("没有可绘制的共现矩阵")
    colorbar_axis = fig.add_axes([0.945, 0.18, 0.012, 0.64])
    colorbar = fig.colorbar(image, cax=colorbar_axis)
    colorbar.set_label("Percentage of examinations or training exposures (%)", fontsize=14)
    colorbar.ax.tick_params(labelsize=12)
    fig.subplots_adjust(left=0.07, right=0.925, bottom=0.12, top=0.92, wspace=0.08, hspace=0.20)
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)


def plot_fold_overview(
    original_labels: dict[str, np.ndarray],
    balanced_fold_labels: dict[str, list[np.ndarray]],
    output_path: Path,
) -> None:
    dataset_order = tuple(DATASET_TITLES.keys())
    column_titles = ("WLE", "Chromoendoscopy", "Surgical", "EUS")
    tick_labels = ("SMT", "EML", "Gastritis")
    row_titles = ("Original", "Fold 1", "Fold 2", "Fold 3", "Fold 4", "Fold 5")
    fig, axes = plt.subplots(6, 4, figsize=(15, 20), dpi=240)
    image = None
    csv_rows: list[list[Any]] = []

    for row_index, row_title in enumerate(row_titles):
        for column_index, (dataset_name, column_title) in enumerate(
            zip(dataset_order, column_titles)
        ):
            labels = (
                original_labels[dataset_name]
                if row_index == 0
                else balanced_fold_labels[dataset_name][row_index - 1]
            )
            matrix = compute_cooccurrence(labels)
            percentages = matrix.astype(np.float64) * 100.0 / labels.shape[0]
            ax = axes[row_index, column_index]
            image = ax.imshow(percentages, cmap="Greens", vmin=0, vmax=100)
            title = f"{column_title}\n$n={labels.shape[0]}$" if row_index == 0 else f"$n={labels.shape[0]}$"
            ax.set_title(title, fontsize=13, fontweight="bold", pad=4)
            ax.set_xticks(np.arange(3))
            ax.set_yticks(np.arange(3))
            ax.set_xticklabels(
                tick_labels if row_index == len(row_titles) - 1 else ("", "", ""),
                fontsize=11,
                rotation=35,
                ha="right",
            )
            ax.set_yticklabels(
                tick_labels if column_index == 0 else ("", "", ""),
                fontsize=11,
            )
            for matrix_row in range(3):
                for matrix_column in range(3):
                    count = int(matrix[matrix_row, matrix_column])
                    percent = float(percentages[matrix_row, matrix_column])
                    ax.text(
                        matrix_column,
                        matrix_row,
                        f"{count}\n{percent:.1f}%",
                        ha="center",
                        va="center",
                        fontsize=10,
                        fontweight="bold",
                        color="white" if percent > 48 else "black",
                    )
            ax.set_xticks(np.arange(-0.5, 3, 1), minor=True)
            ax.set_yticks(np.arange(-0.5, 3, 1), minor=True)
            ax.grid(which="minor", color="white", linewidth=1.2)
            ax.tick_params(which="minor", bottom=False, left=False)
            ax.tick_params(axis="both", length=0)
            csv_rows.append(
                [dataset_name, row_title, int(labels.shape[0]), *matrix.reshape(-1).tolist()]
            )
        axes[row_index, 0].set_ylabel(
            row_title if row_index == 0 else f"{row_title}\nRebalanced",
            fontsize=14,
            fontweight="bold",
            labelpad=10,
        )

    if image is None:
        raise RuntimeError("没有可绘制的逐折共现矩阵")
    colorbar_axis = fig.add_axes([0.945, 0.16, 0.012, 0.68])
    colorbar = fig.colorbar(image, cax=colorbar_axis)
    colorbar.set_label("Percentage of examinations or training exposures (%)", fontsize=13)
    colorbar.ax.tick_params(labelsize=11)
    fig.suptitle(
        "Original and Fold-wise Rebalanced Label Co-occurrence Matrices",
        fontsize=19,
        fontweight="bold",
        y=0.985,
    )
    fig.subplots_adjust(left=0.08, right=0.925, bottom=0.045, top=0.95, wspace=0.10, hspace=0.22)
    fig.savefig(output_path, bbox_inches="tight")
    fig.savefig(output_path.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)

    with output_path.with_suffix(".csv").open("w", encoding="utf-8", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(
            [
                "dataset",
                "distribution",
                "total",
                "smt_smt",
                "smt_eml",
                "smt_gastritis",
                "eml_smt",
                "eml_eml",
                "eml_gastritis",
                "gastritis_smt",
                "gastritis_eml",
                "gastritis_gastritis",
            ]
        )
        writer.writerows(csv_rows)


def main() -> None:
    args = parse_args()
    output_dir = args.output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    grouped_labels = load_dataset_labels(
        args.datalist.expanduser().resolve(),
        args.config.expanduser().resolve(),
    )

    original_labels = dict(grouped_labels)

    for dataset_name, label_array in grouped_labels.items():
        matrix = compute_cooccurrence(label_array)
        output_path = output_dir / f"label_cooccurrence_{dataset_name}.png"
        plot_matrix(
            matrix,
            total=int(label_array.shape[0]),
            figure_title=f"{DATASET_TITLES[dataset_name]} Label Co-occurrence Matrix",
            output_path=output_path,
        )
        print(f"[TASK3] {dataset_name}: n={label_array.shape[0]}, matrix={matrix.tolist()}")
        print(f"[TASK3] 已保存：{output_path}")

    balanced_labels, balanced_fold_labels = load_balanced_training_labels(
        args.fold_root.expanduser().resolve(),
        args.config.expanduser().resolve(),
    )
    for dataset_name in DATASET_TITLES:
        label_array = balanced_labels[dataset_name]
        matrix = compute_cooccurrence(label_array)
        output_path = output_dir / f"label_cooccurrence_balanced_{dataset_name}.png"
        plot_matrix(
            matrix,
            total=int(label_array.shape[0]),
            figure_title=(
                f"{DATASET_SHORT_TITLES[dataset_name]} Rebalanced Training\n"
                "Label Co-occurrence Matrix"
            ),
            output_path=output_path,
            total_name="total_five_fold_training_exposures",
        )
        print(
            f"[TASK3-BALANCED] {dataset_name}: exposures={label_array.shape[0]}, "
            f"matrix={matrix.tolist()}"
        )
        print(f"[TASK3-BALANCED] 已保存：{output_path}")

    overview_path = output_dir / "label_cooccurrence_overview.png"
    plot_overview(original_labels, balanced_labels, overview_path)
    print(f"[TASK3] 已保存8矩阵总览：{overview_path}")

    fold_overview_path = output_dir / "label_cooccurrence_foldwise_24panel.png"
    plot_fold_overview(original_labels, balanced_fold_labels, fold_overview_path)
    print(f"[TASK3] 已保存24矩阵逐折大图：{fold_overview_path}")


if __name__ == "__main__":
    main()
