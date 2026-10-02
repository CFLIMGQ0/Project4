#!/usr/bin/env python3
"""Circular LCCF evidence map with multimodal baselines."""
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
OVERLAP_SOURCE = OUT / "multimodal_overlap_v2.csv"
MATRIX_SOURCE = OUT / "multimodal_intervention_summary_v2.csv"
sys.path.insert(0, "/home/Lim/.agents/skills/nature-figure/scripts")
from audit_panel_alignment import require_matplotlib_panel_alignment

DATASETS = {
    "ct_rate": ("CT-RATE", ["Emph.", "Atel.", "Fibrotic"]),
    "mr_rate": ("MR-RATE-1K", ["Uns.", "Neuro.", "Cerebro.", "Neo."]),
    "amos_mm": ("AMOS-MM", ["L", "K", "G", "S", "B", "P", "T"]),
    "aa_mini": ("AA-Mini", ["Liver", "Pancreas", "Kidney"]),
}
METHODS = [
    "a_shared", "mmfnet", "radfuse", "saif", "mmtf", "camchex",
    "med3dvlm", "m3fm", "unified_mm", "adaptive_fusion", "full",
]
NAMES = {
    "a_shared": "Shared", "mmfnet": "MMF", "radfuse": "RadF.",
    "saif": "SAIF", "mmtf": "MMTF", "camchex": "CaM",
    "med3dvlm": "Med3D", "m3fm": "M3FM", "unified_mm": "U-MM",
    "adaptive_fusion": "A-Fus.", "full": "LCCF",
}
COLORS = {
    "a_shared": "#7D8992", "mmfnet": "#2F6B9A", "radfuse": "#4C78A8",
    "saif": "#168C8C", "mmtf": "#36A39A", "camchex": "#6A9F58",
    "med3dvlm": "#8C6BB1", "m3fm": "#A06A95", "unified_mm": "#D08B3A",
    "adaptive_fusion": "#E27A31", "full": "#C95742",
}
INK = "#273946"
PALE = "#EEF1F3"
GRID = "#E2E8EC"


def load_data() -> dict[str, dict[str, object]]:
    overlap_rows = list(csv.DictReader(OVERLAP_SOURCE.open(encoding="utf-8-sig")))
    mm_rows = list(csv.DictReader(MATRIX_SOURCE.open(encoding="utf-8")))
    result = {}
    for dataset, (title, labels) in DATASETS.items():
        overlap = {
            row["method"]: float(row["value"])
            for row in overlap_rows
            if row["dataset"] == dataset and row["method"] in METHODS
        }
        if set(overlap) != set(METHODS):
            raise RuntimeError(f"Missing overlap values for {dataset}: {set(METHODS)-set(overlap)}")
        matrices = {}
        for method in METHODS:
            selected = [r for r in mm_rows if r["dataset"] == dataset and r["method"] == method]
            n = len(labels); mat = np.zeros((n, n), dtype=float)
            for row in selected:
                mat[labels.index(row["source"]), labels.index(row["target"])] = float(row["value"])
            if len(selected) != n * n or not np.allclose(mat.sum(1), 100.0, atol=1e-3):
                raise RuntimeError(f"Bad multimodal matrix for {dataset}/{method}")
            matrices[method] = mat
        result[dataset] = {"title": title, "labels": labels, "overlap": overlap, "matrices": matrices}
    return result


def polar_xy(angle: float, radius: float) -> tuple[float, float]:
    radians = np.deg2rad(angle)
    return radius * np.cos(radians), radius * np.sin(radians)


def draw_panel(ax, data: dict[str, object], letter: str, show_header: bool = True) -> None:
    title = str(data["title"]); labels = list(data["labels"])
    overlap = data["overlap"]; matrices = data["matrices"]
    n = len(labels)
    ax.set_xlim(-2.55, 2.55); ax.set_ylim(-2.55, 2.55)
    ax.set_aspect("equal"); ax.axis("off")
    if show_header:
        ax.text(-2.30, 2.30, letter, fontsize=9.2, fontweight="bold", color=INK)
        ax.text(0, 2.30, title, ha="center", va="bottom", fontsize=9.2,
                fontweight="bold", color=INK)

    # One slim radial bar per method and label.  The bar height is the
    # diagonal share of the row-normalized intervention matrix.
    base, scale = 1.18, 0.0110
    for value in (20, 40, 60, 80):
        guide_radius = base + scale * value
        ax.add_patch(Circle((0, 0), guide_radius, facecolor="none",
                            edgecolor=GRID, linewidth=.48, zorder=0))
        gx, gy = polar_xy(132.0, guide_radius + .30)
        ax.text(gx, gy, f"{value}%", ha="right", va="center", fontsize=5.0,
                color="#7B8790", fontweight="bold",
                bbox=dict(facecolor="white", edgecolor="none", alpha=.82, pad=.12),
                zorder=2)
    angles = 90 - np.arange(n) * 360 / n
    bar_width = min(4.4, 27.0 / n)
    offsets = np.linspace(-17.5, 17.5, len(METHODS))
    for index, (label, angle) in enumerate(zip(labels, angles)):
        for method, delta in zip(METHODS, offsets):
            value = float(matrices[method][index, index])
            if not np.isfinite(value):
                continue
            ax.add_patch(Wedge(
                (0, 0), base + scale * value,
                angle + delta - bar_width / 2, angle + delta + bar_width / 2,
                width=scale * value, facecolor=COLORS[method],
                edgecolor="white", linewidth=.23, zorder=3,
            ))
        x, y = polar_xy(float(angle), 2.45)
        ax.text(x, y, label, ha="center", va="center", fontsize=5.2,
                fontweight="bold", color=INK, zorder=4)

    # Concentric arcs encode overlap for all conditions.  The white core is
    # used for a compact, color-matched method key so the ring colors remain
    # interpretable without adding a second legend to every panel.
    ring_outer, ring_width, ring_gap = 1.16, .045, .010
    for idx, method in enumerate(METHODS):
        radius = ring_outer - idx * (ring_width + ring_gap)
        value = float(overlap[method])
        ax.add_patch(Wedge((0, 0), radius, 0, 360, width=ring_width,
                           facecolor=PALE, edgecolor="none", linewidth=0, zorder=1))
        ax.add_patch(Wedge((0, 0), radius, 90 - 3.6 * value, 90,
                           width=ring_width, facecolor=COLORS[method],
                           edgecolor="none", zorder=2))
        # Mark the occupancy encoded by each ring at the end of its colored arc.
        # The labels are deliberately compact because the color legend identifies
        # the corresponding method names below the composite figure.
    # Keep the innermost LCCF ring visible; the previous core radius covered
    # most of that ring because eleven method rings are drawn concentrically.
    ax.add_patch(Circle((0, 0), .50, facecolor="white", edgecolor="none", zorder=5))
    ax.text(0, .095, "Shared: 100%", ha="center", va="center", fontsize=5.5,
            color=COLORS["a_shared"], fontweight="bold", zorder=6)
    ax.text(0, -.095, f"LCCF: {float(overlap['full']):.1f}%", ha="center", va="center",
            fontsize=5.5, color=COLORS["full"], fontweight="bold", zorder=6)


def main() -> None:
    plt.rcParams.update({
        "font.family": "sans-serif", "font.sans-serif": ["Arial", "DejaVu Sans", "Liberation Sans"],
        "font.size": 7.0, "pdf.fonttype": 42, "svg.fonttype": "none", "savefig.facecolor": "white",
    })
    data = load_data()
    fig, axes = plt.subplots(1, 4, figsize=(11.3, 4.05))
    fig.subplots_adjust(left=.01, right=.99, bottom=.19, top=.77, wspace=-.04)
    for ax, (dataset, values), letter in zip(axes, data.items(), "abcd"):
        draw_panel(ax, values, letter)
    handles = [Patch(facecolor=COLORS[m], label=NAMES[m]) for m in METHODS]
    fig.legend(handles=handles, loc="lower right", bbox_to_anchor=(.99, .06), ncol=len(METHODS),
               frameon=False, fontsize=5.9, handlelength=.95, columnspacing=1.0,
               handletextpad=.32)
    fig.canvas.draw()
    require_matplotlib_panel_alignment(
        fig, axes=list(axes), strict=True, tolerance_pt=1.5, gutter_tolerance_pt=1.5,
        json_out=OUT / "temp9_multimodal.alignment.json",
        overlay_svg=OUT / "temp9_multimodal.alignment.svg",
    )
    for suffix, kwargs in [("png", {"dpi": 600, "bbox_inches": "tight"}),
                           ("pdf", {"bbox_inches": "tight"}),
                           ("svg", {"bbox_inches": "tight"})]:
        fig.savefig(OUT / f"temp9_multimodal.{suffix}", **kwargs)
    # Keep explicit vector-export calls for static preflight tools.
    fig.savefig(OUT / "temp9_multimodal.pdf", bbox_inches="tight")
    fig.savefig(OUT / "temp9_multimodal.svg", bbox_inches="tight")
    fig.savefig(OUT / "temp9_multimodal.tiff", dpi=600, bbox_inches="tight",
                pil_kwargs={"compression": "tiff_lzw"})
    (ROOT / "temp9_multimodal.png").write_bytes((OUT / "temp9_multimodal.png").read_bytes())
    (OUT / "temp9_multimodal_contract.json").write_text(json.dumps({
        "claim": "Multimodal baselines and LCCF differ in image-selection overlap and label-specific intervention concentration.",
        "rings": "Pairwise Top-5 Jaccard overlap; lower is more distinct",
        "bars": "Diagonal shares of row-normalized absolute decision impact after top-5 intervention",
        "conditions": METHODS, "datasets": [x[0] for x in DATASETS.values()],
        "source": [str(OVERLAP_SOURCE.relative_to(ROOT)), str(MATRIX_SOURCE.relative_to(ROOT))],
    }, ensure_ascii=False, indent=2))
    plt.close(fig)


if __name__ == "__main__":
    main()
