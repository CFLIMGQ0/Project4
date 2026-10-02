#!/usr/bin/env python3
"""Render the four-panel temp6 retrieval diagnostic, including AA-Mini."""
from __future__ import annotations

import csv
import string
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
from scripts import evaluate_lccf_public_figures as evaluator
from scripts import plot_lccf_public_figures as plotter

OUT = ROOT / "outputs/lccf_public_evidence_20261001"


def decorate(ax):
    ax.set_axisbelow(True)
    ax.grid(axis="y", color="#E2E5E9", linewidth=.65)
    ax.spines[["top", "right"]].set_visible(False)


def load_full_only(datasets):
    """Load only the full-model raw arrays; AA-Mini has no baseline raw files."""
    result = {}
    for ds in datasets:
        parts = []
        for fold in range(1, 6):
            p = OUT / "raw" / ds / "full" / f"fold_{fold}.npz"
            meta = __import__("json").loads(p.with_suffix(".json").read_text())
            assert meta["test_reference_passed"]
            with np.load(p, allow_pickle=False) as z:
                parts.append({k: z[k] for k in z.files})
        result[ds] = {k: np.concatenate([part[k] for part in parts])
                      for k in parts[0]}
    return result


def main() -> None:
    evaluator.DATASETS["aa_mini"] = ("AA-Mini", "aa_mini", 3)
    plotter.DATASETS = evaluator.DATASETS
    raw = load_full_only(list(plotter.DATASETS))
    plt.rcParams.update({"font.family": "sans-serif",
                         "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
                         "font.size": 9, "axes.titlesize": 10,
                         "axes.labelsize": 9, "xtick.labelsize": 8,
                         "ytick.labelsize": 8, "pdf.fonttype": 42,
                         "ps.fonttype": 42, "svg.fonttype": "none",
                         "axes.linewidth": .75, "savefig.facecolor": "white"})

    datasets = list(plotter.DATASETS)
    conditions = [("prob", "Correct", "#C44748"),
                  ("cross_prob", "Cross-label", "#4C78A8"),
                  ("pooled_prob", "Pooled", "#999999")]
    collection = {}
    all_bounds = []
    for ds in datasets:
        collection[ds] = [plotter.fold_means(raw[ds], field,
                                              "All positive examinations")[0]
                           for field, _, _ in conditions]
        for values in collection[ds]:
            all_bounds.extend([values.mean() - values.std(ddof=1),
                               values.mean() + values.std(ddof=1)])
    lo = max(0., min(all_bounds) - .04)
    hi = min(1.03, max(all_bounds) + .065)

    fig, axes = plt.subplots(1, 4, figsize=(11.2, 3.3), sharey=True)
    fig.subplots_adjust(left=.065, right=.99, top=.82, bottom=.24, wspace=.16)
    source_rows = []
    for di, (ax, ds) in enumerate(zip(axes, datasets)):
        values = collection[ds]
        means = [v.mean() for v in values]
        ax.plot(range(3), means, color="#333333", linewidth=1.4, zorder=2)
        for j, (v, (_, name, color)) in enumerate(zip(values, conditions)):
            mean, sd = v.mean(), v.std(ddof=1)
            ax.errorbar(j, mean, yerr=sd, fmt="o", color=color, markersize=7,
                        capsize=4, elinewidth=1.0, markeredgecolor="white",
                        markeredgewidth=.7, zorder=3)
            for fold, value in enumerate(v, 1):
                source_rows.append({"figure": "temp6", "dataset": ds,
                                    "condition": name,
                                    "group": "positive examinations",
                                    "fold": fold, "value": float(value)})
        ax.set_title(f"{string.ascii_lowercase[di]}  {plotter.DATASETS[ds][0]}",
                     loc="left", fontweight="bold", pad=12)
        ax.set_xticks(range(3), [name for _, name, _ in conditions])
        ax.set_xlim(-.35, 2.35); ax.set_ylim(lo, hi); decorate(ax)
    axes[0].set_ylabel("Mean positive-label confidence")
    fig.canvas.draw()
    skill = Path("/home/Lim/.agents/skills/nature-figure/scripts")
    sys.path.insert(0, str(skill))
    from audit_panel_alignment import require_matplotlib_panel_alignment
    require_matplotlib_panel_alignment(fig, axes=list(axes), tolerance_pt=1.5,
                                       gutter_tolerance_pt=1.5, strict=True,
                                       json_out=OUT / "temp6.alignment.json")
    fig.savefig(OUT / "temp6.pdf", bbox_inches="tight")
    fig.savefig(OUT / "temp6.svg", bbox_inches="tight")
    fig.savefig(OUT / "temp6.png", dpi=300, bbox_inches="tight")
    fig.savefig(OUT / "temp6.tiff", dpi=600, bbox_inches="tight",
                pil_kwargs={"compression": "tiff_lzw"})
    (ROOT / "temp6.png").write_bytes((OUT / "temp6.png").read_bytes())
    with (OUT / "temp6_aamini_source_data.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=["figure", "dataset", "condition", "group", "fold", "value"])
        writer.writeheader(); writer.writerows(source_rows)
    plt.close(fig)


if __name__ == "__main__":
    main()
