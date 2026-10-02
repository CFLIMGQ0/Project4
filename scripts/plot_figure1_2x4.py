#!/usr/bin/env python3
"""Figure 1：四数据集、两指标的 2×4 ACPE–Original PE 离散删片增益曲面。"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path
import sys

import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm
from matplotlib.cm import ScalarMappable
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
SKILL = Path("/home/Lim/.agents/skills/nature-figure/scripts")
sys.path.insert(0, str(SKILL))
from audit_panel_alignment import require_matplotlib_panel_alignment

SOURCE = ROOT / "outputs/paper_results/figure1_summary/summary.csv"
OUT = ROOT / "outputs/paper_results/figure1_2x4"
DATASETS = (("ct_rate", "CT-RATE"), ("amos_mm", "AMOS-MM"),
            ("mr_rate_1k", "MR-RATE-1K"), ("merlin_1k", "Merlin-1K"))
METRICS = (("Delta_MacroF1_Mean", r"$\Delta$Macro-F1"),
           ("Delta_Acc005_Mean", r"$\Delta$Acc@0.05"))
RATIOS = tuple(range(0, 81, 10))
BLOCKS = tuple(range(8, 0, -1))


def matrix(rows: list[dict], dataset: str, metric: str) -> np.ndarray | None:
    lookup = {(int(row["DeletionRatio"]), int(row["Blocks"])): row
              for row in rows if row["Dataset"] == dataset}
    result = np.empty((len(RATIOS), len(BLOCKS)), dtype=float)
    for i, ratio in enumerate(RATIOS):
        for j, blocks in enumerate(BLOCKS):
            row = lookup.get((ratio, 1 if ratio == 0 else blocks))
            if not row or row["Status"] != "COMPLETE" or row[metric] == "":
                return None
            result[i, j] = float(row[metric])
    if not np.isfinite(result).all():
        raise ValueError(f"非有限网格值：{dataset}/{metric}")
    assert np.allclose(result[0], result[0, 0])  # 0% 删除只测一次，所有 B 等价
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--allow-withheld-merlin", action="store_true",
                        help="Merlin 未审定时保留两块明确标记的空白子图")
    args = parser.parse_args()
    with SOURCE.open(encoding="utf-8-sig", newline="") as stream:
        rows = list(csv.DictReader(stream))
    arrays = {(dataset, metric): matrix(rows, dataset, metric)
              for dataset, _ in DATASETS for metric, _ in METRICS}
    missing = [(dataset, metric) for (dataset, metric), value in arrays.items() if value is None]
    permitted = {("merlin_1k", metric) for metric, _ in METRICS} if args.allow_withheld_merlin else set()
    unexpected = set(missing) - permitted
    if unexpected:
        raise RuntimeError(f"Figure 1 网格数据未完成，不输出图：{sorted(unexpected)}")
    if missing and not args.allow_withheld_merlin:
        raise RuntimeError(f"Figure 1 网格数据未完成：{missing}")

    mpl.rcParams.update({"font.family": "sans-serif", "font.sans-serif": ["Arial", "DejaVu Sans"],
                         "pdf.fonttype": 42, "svg.fonttype": "none", "font.size": 5.7,
                         "axes.linewidth": .5, "savefig.transparent": False})
    fig = plt.figure(figsize=(7.15, 4.02), constrained_layout=False)
    grid = fig.add_gridspec(2, 4, left=.055, right=.915, bottom=.10, top=.91,
                            wspace=.04, hspace=.02)
    axes = []
    cmap = mpl.colormaps["RdBu"]  # 正增益：蓝；负增益：红
    ranges = []
    for metric, _ in METRICS:
        values = [np.abs(arrays[dataset, metric]).max() for dataset, _ in DATASETS
                  if arrays[dataset, metric] is not None]
        ranges.append(max(.05, np.ceil(max(values)/.05)*.05))
    x, y = np.meshgrid(np.asarray(BLOCKS), np.asarray(RATIOS))
    for i, (metric, row_title) in enumerate(METRICS):
        norm = TwoSlopeNorm(vmin=-ranges[i], vcenter=0, vmax=ranges[i])
        for j, (dataset, title) in enumerate(DATASETS):
            ax = fig.add_subplot(grid[i, j], projection="3d")
            axes.append(ax)
            z = arrays[dataset, metric]
            if z is None:
                ax.set_axis_off()
                ax.text2D(.5, .48, "Report-derived labels\nresult withheld",
                          ha="center", va="center", fontsize=6.2, color="#505050",
                          transform=ax.transAxes)
            else:
                ax.plot_surface(x, y, z, facecolors=cmap(norm(z)), shade=False,
                                edgecolor="#404040", linewidth=.13, antialiased=True,
                                rstride=1, cstride=1)
                ax.scatter(x.ravel(), y.ravel(), z.ravel(), s=.5, color="#303030", depthshade=False)
                ax.set_xlim(8, 1)
                ax.set_ylim(0, 80)
                ax.set_zlim(-ranges[i], ranges[i])
                ax.set_xticks((8, 4, 1), labels=("B8", "B4", "B1"))
                ax.set_yticks((0, 40, 80), labels=("0", "40", "80"))
                ax.set_zticks((-ranges[i], 0, ranges[i]),
                              labels=("", "0", f"{ranges[i]:g}"))
                ax.tick_params(labelsize=5.0, pad=-3, width=.35)
                ax.view_init(elev=25, azim=-58)
                ax.set_box_aspect((1.0, .80, .55), zoom=.85)
                for axis in (ax.xaxis, ax.yaxis, ax.zaxis):
                    axis.pane.fill = False
                    axis.line.set_linewidth(.35)
                ax.grid(False)
            if i == 0:
                ax.set_title(title, fontsize=6.7, pad=0)
            ax.text2D(.02, .96, chr(ord("a")+i*4+j), transform=ax.transAxes,
                      fontsize=7.4, fontweight="bold", va="top")
        colorbar_axis = fig.add_axes((.932, .535 if i == 0 else .115, .012, .33))
        bar = fig.colorbar(ScalarMappable(norm=norm, cmap=cmap), cax=colorbar_axis,
                           ticks=(-ranges[i], 0, ranges[i]))
        bar.ax.tick_params(labelsize=5.2, length=2, pad=1)
        bar.outline.set_linewidth(.4)
        fig.text(.004, .704 if i == 0 else .295, row_title, rotation=90,
                 rotation_mode="anchor",
                 ha="left", va="center", fontsize=7.0)
    fig.text(.49, .025, "Deletion blocks (B8 to B1)   ·   Deletion ratio (0–80%)",
             ha="center", va="center", fontsize=6.0)
    OUT.mkdir(parents=True, exist_ok=True)
    stem = OUT / "acpe_deletion_2x4"
    require_matplotlib_panel_alignment(fig, axes=axes, panel_ids=list("abcdefgh"),
        json_out=str(stem)+".alignment.json", overlay_svg=str(stem)+".alignment.svg",
        tolerance_pt=1.5, gutter_tolerance_pt=1.5, strict=True)
    fig.savefig(stem.with_suffix(".pdf"), bbox_inches="tight")
    fig.savefig(stem.with_suffix(".svg"), bbox_inches="tight")
    fig.savefig(stem.with_suffix(".png"), dpi=600, bbox_inches="tight")
    fig.savefig(stem.with_suffix(".tiff"), dpi=600, bbox_inches="tight")
    plt.close(fig)
    print(f"输出 {stem}.pdf；缺失子图：{missing}", flush=True)


if __name__ == "__main__":
    main()
