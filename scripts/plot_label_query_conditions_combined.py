#!/usr/bin/env python3
"""Plot the three label-query conditions after pooling all four test datasets."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import FormatStrFormatter


ROOT = Path(__file__).resolve().parents[1]
PROJECT_ROOT = ROOT.parent
STATS_ROOT = (
    PROJECT_ROOT
    / "outputs/quick_finetune/label_query_discrimination_full"
)
OUTPUT = ROOT / "figs/label_query_retrieval_specificity_four_datasets.png"

CONDITIONS = (
    ("Correct", "correct", "#C23B3B"),
    ("Cross-label", "cross_label", "#4C78A8"),
    ("Pooled", "pooled", "#999999"),
)


def main() -> None:
    stats_paths = sorted(STATS_ROOT.glob("*/fold_1/stats.json"))
    if len(stats_paths) != 4:
        raise RuntimeError(f"Expected four dataset statistics, found {len(stats_paths)}")
    rows = [json.loads(path.read_text(encoding="utf-8")) for path in stats_paths]
    total = sum(row["after_overall"]["positive_label_observations"] for row in rows)
    values = []
    for _, key, _ in CONDITIONS:
        weighted_sum = sum(
            row["after_overall"][key]
            * row["after_overall"]["positive_label_observations"]
            for row in rows
        )
        values.append(weighted_sum / total)

    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 11,
            "axes.labelsize": 12,
            "xtick.labelsize": 11,
            "ytick.labelsize": 10,
            "savefig.facecolor": "white",
        }
    )
    figure, axis = plt.subplots(figsize=(6.2, 4.4))
    x = np.arange(len(CONDITIONS))
    axis.plot(
        x,
        values,
        color="#333333",
        linewidth=2.5,
        zorder=2,
    )
    axis.scatter(
        x,
        values,
        s=110,
        color=[item[2] for item in CONDITIONS],
        edgecolor="white",
        linewidth=1.2,
        zorder=3,
    )

    lower = 0.8
    upper = 0.9
    axis.set_ylim(lower, upper)
    axis.set_ylabel("Mean positive-label confidence")
    axis.set_xticks(x, [item[0] for item in CONDITIONS])
    axis.yaxis.set_major_formatter(FormatStrFormatter("%.2f"))
    axis.grid(axis="y", color="#E2E2E2", linewidth=0.8, zorder=0)
    axis.spines[["top", "right"]].set_visible(False)
    axis.spines[["left", "bottom"]].set_color("#777777")

    offset = (upper - lower) * 0.025
    for x_position, value in zip(x, values):
        axis.text(
            x_position,
            value + offset,
            f"{value:.4f}",
            ha="center",
            va="bottom",
            fontsize=11,
            fontweight="bold",
            color="#222222",
        )

    figure.tight_layout()
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(OUTPUT, dpi=300, bbox_inches="tight")
    plt.close(figure)
    print(OUTPUT.resolve())


if __name__ == "__main__":
    main()
