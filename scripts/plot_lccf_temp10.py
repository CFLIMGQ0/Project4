#!/usr/bin/env python3
"""Plot the measured temp10 label-coupling intervention results."""
from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D

SKILL = Path("/home/Lim/.agents/skills/nature-figure/scripts")
sys.path.insert(0, str(SKILL))
from audit_panel_alignment import require_matplotlib_panel_alignment

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "outputs" / "lccf_temp10_20261001"
RAW = OUT / "raw"

DATASETS = [
    ("ct_rate", "CT-RATE", "#C44748"),
    ("mr_rate", "MR-RATE-1K", "#4C78A8"),
    ("amos_mm", "AMOS-MM", "#2A9D8F"),
    ("aa_mini", "AA-Mini", "#7B61A8"),
]
CONDITIONS = [
    ("No reasoning", "b_none"),
    ("Ordinary graph", "b_graph"),
    ("Hypergraph E=1", "b_hyper1"),
    ("Hypergraph E=2", "full"),
    ("Hypergraph E=3", "b_hyper3"),
    ("Hypergraph E=4", "b_hyper4"),
    ("Hypergraph E=5", "b_hyper5"),
]


def style() -> None:
    plt.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans", "sans-serif"],
            "font.size": 7.6,
            "axes.titlesize": 9.2,
            "axes.labelsize": 8.0,
            "xtick.labelsize": 6.4,
            "ytick.labelsize": 7.0,
            "legend.fontsize": 6.8,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "svg.fonttype": "none",
            "axes.linewidth": 0.75,
            "savefig.facecolor": "white",
        }
    )


def read_summary(dataset: str, variant: str) -> tuple[float, float, list[float]]:
    rows = []
    for path in sorted((RAW / dataset / variant).glob("fold_*.json")):
        row = json.loads(path.read_text())
        if row.get("protocol_version") != "full_suffix_case_mean_v2":
            raise RuntimeError(f"旧协议或缺失协议标记: {path}")
        rows.append(float(row["coexistence_coupling_ratio"]))
    if len(rows) != 5:
        raise RuntimeError(f"{dataset}/{variant}: 需要五折结果，当前为 {len(rows)} 折")
    values = np.asarray(rows, dtype=float)
    return float(values.mean()), float(values.std(ddof=1)), rows


def write_source_csv(values: dict) -> None:
    path = OUT / "temp10_source_data.csv"
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.writer(stream)
        writer.writerow(["dataset", "display_dataset", "condition", "variant", "fold", "coupling_ratio"])
        for dataset, display, _ in DATASETS:
            for condition, variant in CONDITIONS:
                _, _, folds = values[dataset][variant]
                for fold, value in enumerate(folds, start=1):
                    writer.writerow([dataset, display, condition, variant, fold, f"{value:.10f}"])


def main() -> None:
    style()
    values = {
        dataset: {variant: read_summary(dataset, variant) for _, variant in CONDITIONS}
        for dataset, _, _ in DATASETS
    }
    write_source_csv(values)

    fig, axes = plt.subplots(1, 4, figsize=(9.6, 3.25), sharey=True)
    fig.subplots_adjust(left=0.075, right=0.99, bottom=0.28, top=0.70, wspace=0.16)
    x = np.arange(len(CONDITIONS), dtype=float)

    for index, (dataset, display, color) in enumerate(DATASETS):
        ax = axes[index]
        means = np.asarray([values[dataset][variant][0] for _, variant in CONDITIONS])
        sds = np.asarray([values[dataset][variant][1] for _, variant in CONDITIONS])
        ax.axhline(1.0, color="#9AA0A6", linewidth=0.9, linestyle=(0, (3, 2)), zorder=1)
        # The ordinary graph is a pairwise reference; hypergraph E=1..5 are
        # connected because E is the ordered sweep variable.
        ax.errorbar(x[0], means[0], yerr=sds[0], fmt="s", ms=5.5,
                    color="#6C737B", mfc="white", mec="#6C737B", mew=1.2,
                    capsize=2.2, capthick=0.8, zorder=4)
        ax.errorbar(x[1], means[1], yerr=sds[1], fmt="D", ms=5.2,
                    color="#6C737B", mfc="white", mec="#6C737B", mew=1.2,
                    capsize=2.2, capthick=0.8, zorder=4)
        ax.errorbar(x[2:], means[2:], yerr=sds[2:], fmt="o-",
                    ms=5.6, color=color, mfc="white", mec=color, mew=1.5,
                    linewidth=2.0, capsize=2.2, capthick=0.8, zorder=5)
        ax.fill_between(x[2:], 1.0, means[2:], color=color, alpha=0.07, zorder=2)
        ax.set_title(display, loc="left", fontweight="bold", pad=7)
        ax.set_xticks(x, ["None", "Graph", "E=1", "E=2", "E=3", "E=4", "E=5"])
        ax.set_xlim(-0.35, 6.35)
        ax.set_ylim(0.55, 4.55)
        ax.set_yticks([1, 2, 3, 4])
        ax.spines[["top", "right"]].set_visible(False)
        ax.spines[["left", "bottom"]].set_color("#8A9098")
        ax.grid(axis="y", color="#E2E5E9", linewidth=0.65)
        ax.set_axisbelow(True)
        ax.tick_params(axis="x", length=3, pad=3)
        if index:
            ax.tick_params(labelleft=False)

    axes[0].set_ylabel("Coexistence coupling ratio")
    fig.text(0.53, 0.095, "Label-reasoning configuration", ha="center", fontsize=8.0)
    handles = [
        Line2D([0], [0], marker="s", color="#6C737B", markerfacecolor="white",
               markersize=5.5, linestyle="None", label="No reasoning"),
        Line2D([0], [0], marker="D", color="#6C737B", markerfacecolor="white",
               markersize=5.2, linestyle="None", label="Ordinary graph"),
        Line2D([0], [0], marker="o", color="#4C78A8", markerfacecolor="white",
               markersize=5.5, linewidth=2, label="Hypergraph"),
    ]
    fig.legend(handles=handles, loc="upper center", bbox_to_anchor=(0.53, 0.875),
               ncol=3, frameon=False, handlelength=1.8, columnspacing=1.4)
    fig.canvas.draw()
    require_matplotlib_panel_alignment(
        fig, axes=list(axes), tolerance_pt=1.5, gutter_tolerance_pt=1.5,
        strict=True, json_out=OUT / "temp10.alignment.json",
        overlay_svg=OUT / "temp10.alignment.svg",
    )
    fig.savefig(OUT / "temp10.pdf", bbox_inches="tight")
    fig.savefig(OUT / "temp10.svg", bbox_inches="tight")
    fig.savefig(OUT / "temp10.png", dpi=300, bbox_inches="tight")
    fig.savefig(OUT / "temp10.tiff", dpi=600, bbox_inches="tight")
    (ROOT / "temp10.png").write_bytes((OUT / "temp10.png").read_bytes())
    contract = {
        "status": "measured_five_fold_results",
        "metric": "mean absolute cross-label logit change for co-positive pairs divided by source-positive/target-absent pairs; case means then fold means",
        "x_axis": [condition for condition, _ in CONDITIONS],
        "error_bars": "sample standard deviation across five folds",
        "datasets": [display for _, display, _ in DATASETS],
        "source_data": "temp10_source_data.csv",
        "source_protocol": "evaluate_lccf_temp10.py with protocol full_suffix_case_mean_v2",
        "interpretation": "The ratio is an unsigned coupling diagnostic; it does not determine whether a label relation is supportive or suppressive.",
    }
    (OUT / "temp10.contract.json").write_text(json.dumps(contract, ensure_ascii=False, indent=2))
    plt.close(fig)


if __name__ == "__main__":
    main()
