from __future__ import annotations

import argparse
import csv
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


DEFAULT_DATALIST = Path("/home/Lim/Project4/datasets/task_data/task1/gastro_multilabel_task_datalist.csv")
DEFAULT_OUTPUT = Path("/home/Lim/Project4/src/figs/label_cooccurrence_matrix.png")

LABELS = [
    ("Esophageal SMT", "label_esophageal_smt"),
    ("Esophageal mucosal lesion", "label_esophageal_mucosal_or_tumor"),
    ("Gastritis", "label_gastritis"),
]


def load_labels(csv_path: Path) -> np.ndarray:
    rows: list[list[int]] = []
    with csv_path.open("r", encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        missing = [column for _, column in LABELS if column not in (reader.fieldnames or [])]
        if missing:
            raise ValueError(f"Missing label columns in {csv_path}: {missing}")
        for row in reader:
            rows.append([int(float(row[column])) for _, column in LABELS])
    if not rows:
        raise ValueError(f"No rows found in {csv_path}")
    return np.asarray(rows, dtype=np.int64)


def compute_cooccurrence(label_array: np.ndarray) -> np.ndarray:
    return label_array.T @ label_array


def save_matrix_csv(matrix: np.ndarray, output_path: Path) -> None:
    csv_path = output_path.with_suffix(".csv")
    with csv_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["label"] + [name for name, _ in LABELS])
        for label_name, row in zip([name for name, _ in LABELS], matrix.tolist()):
            writer.writerow([label_name] + row)


def plot_cooccurrence(matrix: np.ndarray, total: int, output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)

    display_labels = ["Esophageal\nSMT", "Esophageal mucosal\nlesion", "Gastritis"]
    fig, ax = plt.subplots(figsize=(7.6, 6.2), dpi=300)
    image = ax.imshow(matrix, cmap="Greens", vmin=0, vmax=int(matrix.max()))

    ax.set_title("Label Co-occurrence Matrix", fontsize=17, fontweight="bold", pad=16)
    ax.set_xticks(np.arange(len(display_labels)))
    ax.set_yticks(np.arange(len(display_labels)))
    ax.set_xticklabels(display_labels, fontsize=11)
    ax.set_yticklabels(display_labels, fontsize=11)
    ax.set_xlabel("Label", fontsize=12, labelpad=10)
    ax.set_ylabel("Label", fontsize=12, labelpad=10)

    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]):
            count = int(matrix[i, j])
            percent = 100.0 * count / total
            descriptor = "Positive" if i == j else "Co-positive"
            color = "white" if count > matrix.max() * 0.55 else "black"
            ax.text(
                j,
                i,
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
    save_matrix_csv(matrix, output_path)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate label co-occurrence matrix figure.")
    parser.add_argument("--datalist", type=Path, default=DEFAULT_DATALIST, help="Datalist CSV path.")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT, help="Output PNG path.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    label_array = load_labels(args.datalist)
    matrix = compute_cooccurrence(label_array)
    plot_cooccurrence(matrix, total=label_array.shape[0], output_path=args.output)
    print(f"Saved co-occurrence figure: {args.output}")
    print(f"Saved co-occurrence matrix CSV: {args.output.with_suffix('.csv')}")


if __name__ == "__main__":
    main()
