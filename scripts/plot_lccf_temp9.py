#!/usr/bin/env python3
"""Circular LCCF evidence map with overlap rings and label-impact bars."""
from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Circle, Patch, Wedge
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "outputs/lccf_public_evidence_20261001"
SOURCE = OUT / "figure_source_data.csv"
sys.path.insert(0, "/home/Lim/.agents/skills/nature-figure/scripts")
from audit_panel_alignment import require_matplotlib_panel_alignment

DATASETS = {
    "ct_rate": ("CT-RATE", ["Emph.", "Atel.", "Fibrotic"]),
    "mr_rate": ("MR-RATE-1K", ["Uns.", "Neuro.", "Cerebro.", "Neo."]),
    "amos_mm": ("AMOS-MM", ["L", "K", "G", "S", "B", "P", "T"]),
}
SHARED = "#7D8992"
LCCF = "#C95742"
INK = "#273946"
PALE = "#EEF1F3"
GRID = "#E2E8EC"


def load_source() -> dict[str, dict[str, object]]:
    rows = list(csv.DictReader(SOURCE.open(encoding="utf-8-sig")))
    result: dict[str, dict[str, object]] = {}
    for dataset, (title, labels) in DATASETS.items():
        overlap_rows = [
            row for row in rows
            if row["dataset"] == dataset and row["figure"] == "temp3"
            and row["condition"] in {"a_shared", "full"}
        ]
        overlap = {row["condition"]: 100.0 * float(row["value"]) for row in overlap_rows}
        if set(overlap) != {"a_shared", "full"}:
            raise RuntimeError(f"Missing overlap values for {dataset}")
        matrices = []
        for method in ("Shared control", "Label-wise attention"):
            selected = [
                row for row in rows
                if row["dataset"] == dataset and row["figure"] == "temp4"
                and row["condition"] == method
            ]
            n = len(labels)
            if len(selected) != n * n:
                raise RuntimeError(f"Missing temp4 matrix for {dataset}/{method}")
            matrix = np.zeros((n, n), dtype=float)
            for row in selected:
                matrix[labels.index(row["group"]), labels.index(row["target"])] = float(row["value"])
            if not np.allclose(matrix.sum(axis=1), 100.0, atol=1e-3):
                raise RuntimeError(f"temp4 matrix is not row-normalized for {dataset}/{method}")
            matrices.append(matrix)
        result[dataset] = {
            "title": title, "labels": labels, "overlap": overlap,
            "matrices": matrices,
        }
    return result


def ring(ax, radius: float, width: float, value: float, color: str) -> None:
    ax.add_patch(Wedge((0, 0), radius, 0, 360, width=width,
                       facecolor=PALE, edgecolor="none", zorder=1))
    ax.add_patch(Wedge((0, 0), radius, 90 - 3.6 * value, 90,
                       width=width, facecolor=color, edgecolor="none", zorder=2))


def polar_xy(angle: float, radius: float) -> tuple[float, float]:
    radians = np.deg2rad(angle)
    return radius * np.cos(radians), radius * np.sin(radians)


def draw_panel(ax, data: dict[str, object], letter: str) -> None:
    title = str(data["title"])
    labels = list(data["labels"])
    overlap = data["overlap"]
    matrices = data["matrices"]
    n = len(labels)
    ax.set_xlim(-2.30, 2.30)
    ax.set_ylim(-2.30, 2.30)
    ax.set_aspect("equal")
    ax.axis("off")
    ax.text(-2.18, 2.17, letter, fontsize=9.5, fontweight="bold", color=INK)
    ax.text(0, 2.17, title, ha="center", va="bottom", fontsize=9.5,
            fontweight="bold", color=INK)

    # Outer radial bars: diagonal share of the row-normalized intervention matrix.
    base, scale = 1.05, 0.0108
    for value in (20, 40, 60):
        ax.add_patch(Circle((0, 0), base + scale * value, facecolor="none",
                            edgecolor=GRID, linewidth=0.55, zorder=0))
    angles = 90 - np.arange(n) * 360 / n
    bar_width = min(13.0, 40.0 / n)
    offset = bar_width * 0.64
    for index, (label, angle) in enumerate(zip(labels, angles)):
        for value, color, delta in (
            (float(matrices[0][index, index]), SHARED, offset),
            (float(matrices[1][index, index]), LCCF, -offset),
        ):
            ax.add_patch(Wedge(
                (0, 0), base + scale * value,
                angle + delta - bar_width / 2,
                angle + delta + bar_width / 2,
                width=scale * value, facecolor=color,
                edgecolor="white", linewidth=0.45, zorder=3,
            ))
        x, y = polar_xy(float(angle), 2.06)
        cosine = np.cos(np.deg2rad(angle))
        ha = "left" if cosine > 0.22 else "right" if cosine < -0.22 else "center"
        ax.text(x, y, label, ha=ha, va="center", fontsize=5.7,
                fontweight="bold", color=INK, zorder=4)

    # Inner rings summarize image-selection overlap; only Shared and LCCF remain.
    ring(ax, 0.90, 0.10, float(overlap["a_shared"]), SHARED)
    ring(ax, 0.70, 0.10, float(overlap["full"]), LCCF)
    ax.add_patch(Circle((0, 0), 0.56, facecolor="white", edgecolor="none", zorder=5))
    ax.text(0, 0.10, f"{float(overlap['a_shared']):.1f}%", ha="center",
            va="center", fontsize=5.6, color=SHARED, fontweight="bold", zorder=6)
    ax.text(0, -0.10, f"{float(overlap['full']):.1f}%", ha="center",
            va="center", fontsize=5.6, color=LCCF, fontweight="bold", zorder=6)


def main() -> None:
    plt.rcParams.update({
        "font.family": "sans-serif",
        "font.sans-serif": ["Arial", "DejaVu Sans", "Liberation Sans"],
        "font.size": 7.2,
        "pdf.fonttype": 42,
        "svg.fonttype": "none",
        "savefig.facecolor": "white",
    })
    data = load_source()
    fig, axes = plt.subplots(1, 3, figsize=(7.2, 3.45))
    fig.subplots_adjust(left=0.025, right=0.975, bottom=0.16, top=0.79, wspace=0.10)
    for ax, (dataset, values), letter in zip(axes, data.items(), "abc"):
        draw_panel(ax, values, letter)
    fig.suptitle("Label-specific image selection", fontsize=10.2,
                 fontweight="bold", color=INK, y=0.965)
    fig.legend(
        handles=[Patch(facecolor=SHARED, label="Shared"),
                 Patch(facecolor=LCCF, label="LCCF")],
        loc="upper center", bbox_to_anchor=(0.5, 0.885), ncol=2,
        frameon=False, fontsize=7.0, handlelength=1.2, columnspacing=1.6,
    )
    fig.text(0.5, 0.035,
             "Rings: pairwise Top-5 overlap (lower is more distinct); bars: diagonal intervention share.",
             ha="center", fontsize=6.3, color="#536570")
    fig.canvas.draw()
    require_matplotlib_panel_alignment(
        fig, axes=list(axes), strict=True, tolerance_pt=1.5,
        gutter_tolerance_pt=1.5,
        json_out=OUT / "temp9.alignment.json",
        overlay_svg=OUT / "temp9.alignment.svg",
    )
    fig.savefig(OUT / "temp9.png", dpi=600, bbox_inches="tight")
    fig.savefig(OUT / "temp9.pdf", bbox_inches="tight")
    fig.savefig(OUT / "temp9.svg", bbox_inches="tight")
    fig.savefig(OUT / "temp9.tiff", dpi=600, bbox_inches="tight",
                pil_kwargs={"compression": "tiff_lzw"})
    (ROOT / "temp9.png").write_bytes((OUT / "temp9.png").read_bytes())
    (OUT / "temp9_contract.json").write_text(json.dumps({
        "claim": "LCCF selects distinct image evidence and concentrates intervention impact on corresponding labels.",
        "rings": "Shared and LCCF pairwise Top-5 Jaccard overlap; lower is more distinct",
        "bars": "Shared and LCCF diagonal shares of the row-normalized intervention matrix",
        "conditions": ["Shared", "LCCF"],
        "datasets": [str(values["title"]) for values in data.values()],
        "source": str(SOURCE.relative_to(ROOT)),
    }, ensure_ascii=False, indent=2))
    plt.close(fig)


if __name__ == "__main__":
    main()
