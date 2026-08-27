#!/usr/bin/env python3
"""Combine the four OOF distribution analyses into a 4-row by 3-column figure."""

from __future__ import annotations

import json
import string
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D

ROOT = Path(__file__).resolve().parents[1]
PROJECT_ROOT = ROOT.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts import plot_apro_mechanism_preview as base


DATASETS = (
    ("wle", "WLE"),
    ("chromoscopic", "Chromoscopic"),
    ("surgical", "Surgical"),
    ("eus", "EUS"),
)
VARIANTS = ("original_pe", "apro_full")
VARIANT_LABELS = ("Sampled-slot PE", "APro-CoPE")
PANELS = (
    (
        "label_accuracy",
        "Label-wise accuracy",
        "Proportion correct\n(higher is better)",
    ),
    (
        "true_class_confidence",
        "True-class confidence",
        "True-class probability\n(higher is better)",
    ),
    (
        "brier_score",
        "Brier score",
        "Squared probability error\n(lower is better)",
    ),
)


def load_examinations(output_dir: Path, short_name: str) -> list[dict]:
    path = output_dir / f"figure2_oof_distribution_{short_name}.json"
    payload = json.loads(path.read_text(encoding="utf-8"))
    return payload["examinations"]


def metric_values(examinations: list[dict], variant: str, metric: str) -> np.ndarray:
    return np.asarray(
        [exam["metrics"][variant][metric] for exam in examinations],
        dtype=float,
    )


def padded_limits(all_values: list[np.ndarray], metric: str) -> tuple[float, float]:
    values = np.concatenate(all_values)
    lower = float(values.min())
    upper = float(values.max())
    span = max(upper - lower, 0.05)
    if metric in {"label_accuracy", "true_class_confidence"}:
        return max(0.0, lower - 0.06 * span), min(1.02, upper + 0.035 * span)
    return max(0.0, lower - 0.025 * span), upper + 0.055 * span


def draw_panel(
    axis: plt.Axes,
    values: list[np.ndarray],
    colors: tuple[str, str],
    rng: np.random.Generator,
) -> None:
    violin = axis.violinplot(
        values,
        positions=(1, 2),
        widths=0.80,
        showmeans=False,
        showmedians=False,
        showextrema=False,
        bw_method=0.22,
    )
    for body, color in zip(violin["bodies"], colors):
        body.set_facecolor(color)
        body.set_edgecolor(color)
        body.set_alpha(0.18)
        body.set_linewidth(1.0)

    box = axis.boxplot(
        values,
        positions=(1, 2),
        widths=0.27,
        patch_artist=True,
        showfliers=False,
        medianprops={"color": "#202020", "linewidth": 1.35},
        whiskerprops={"color": "#303030", "linewidth": 0.9},
        capprops={"color": "#303030", "linewidth": 0.9},
    )
    for patch, color in zip(box["boxes"], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.27)
        patch.set_edgecolor(color)
        patch.set_linewidth(1.1)

    for position, value_set, color in zip((1, 2), values, colors):
        jitter = rng.uniform(-0.105, 0.105, len(value_set))
        axis.scatter(
            np.full(len(value_set), position) + jitter,
            value_set,
            s=5,
            color=color,
            alpha=0.20,
            linewidth=0,
            rasterized=True,
            zorder=2,
        )
        axis.scatter(
            position,
            float(value_set.mean()),
            marker="D",
            s=34,
            facecolor=color,
            edgecolor="white",
            linewidth=0.7,
            zorder=4,
        )

    axis.set_xlim(0.48, 2.52)
    axis.set_xticks((1, 2))
    axis.grid(axis="y", color="#E8E8E8", linewidth=0.65)
    axis.tick_params(axis="both", labelsize=8.5)


def main() -> None:
    output_dir = PROJECT_ROOT / "temp_img"
    base.configure_style()
    colors = (base.COLORS["original_pe"], base.COLORS["apro_full"])
    examinations_by_dataset = {
        short_name: load_examinations(output_dir, short_name)
        for short_name, _ in DATASETS
    }

    column_limits = []
    for metric, _, _ in PANELS:
        all_values = [
            metric_values(examinations_by_dataset[short_name], variant, metric)
            for short_name, _ in DATASETS
            for variant in VARIANTS
        ]
        column_limits.append(padded_limits(all_values, metric))

    fig, axes = plt.subplots(4, 3, figsize=(13.2, 11.8), sharey="col")
    rng = np.random.default_rng(2026)
    panel_letters = iter(string.ascii_lowercase)

    for row, (short_name, display_name) in enumerate(DATASETS):
        examinations = examinations_by_dataset[short_name]
        for column, (metric, title, subtitle) in enumerate(PANELS):
            axis = axes[row, column]
            values = [
                metric_values(examinations, variant, metric)
                for variant in VARIANTS
            ]
            draw_panel(axis, values, colors, rng)
            axis.set_ylim(*column_limits[column])
            axis.text(
                0.015,
                0.975,
                next(panel_letters),
                transform=axis.transAxes,
                ha="left",
                va="top",
                fontsize=11.5,
                fontweight="bold",
            )
            if row == 0:
                axis.set_title(
                    title,
                    fontsize=11.5,
                    fontweight="bold",
                    pad=5,
                )
            mean_labels = [
                rf"$\bar{{x}}={float(value_set.mean()):.4f}$"
                for value_set in values
            ]
            axis.set_xticklabels(mean_labels, rotation=0)

        axes[row, 0].set_ylabel(
            f"{display_name}\n(n={len(examinations)})",
            fontsize=10.5,
            fontweight="bold",
            labelpad=15,
        )

    legend_handles = [
        Line2D(
            [0],
            [0],
            marker="o",
            linestyle="none",
            markerfacecolor=color,
            markeredgecolor=color,
            markersize=8,
            label=label,
        )
        for color, label in zip(colors, VARIANT_LABELS)
    ]
    fig.legend(
        handles=legend_handles,
        loc="upper center",
        bbox_to_anchor=(0.5, 0.954),
        ncol=2,
        frameon=False,
        columnspacing=2.2,
        handletextpad=0.5,
    )
    fig.suptitle(
        "Effect of APro-CoPE on examination-level prediction quality",
        fontsize=15,
        fontweight="bold",
        y=0.992,
    )
    fig.subplots_adjust(
        left=0.085,
        right=0.992,
        bottom=0.055,
        top=0.880,
        wspace=0.18,
        hspace=0.22,
    )
    base.save_figure(
        fig,
        output_dir / "apro_cope_examination_level_prediction_distributions",
    )
    print("Saved combined 4-row by 3-column OOF distribution figure")


if __name__ == "__main__":
    main()
