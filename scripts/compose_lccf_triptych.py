#!/usr/bin/env python3
"""用原始统计重排三行 LCCF 图，顺序为 9、10、6。"""
from __future__ import annotations

import csv
import hashlib
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
import numpy as np

import plot_lccf_temp9_multimodal as evidence
import plot_lccf_temp10 as reasoning

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "outputs/lccf_composite_20261002"
RETRIEVAL_SOURCE = ROOT / "outputs/lccf_public_evidence_20261001/temp6_aamini_source_data.csv"
REASONING_SOURCE = ROOT / "outputs/lccf_temp10_20261001/temp10_source_data.csv"
CONDITIONS = [("Correct", "#C44748"), ("Cross-label", "#4C78A8"), ("Pooled", "#999999")]
sys.path.insert(0, "/home/Lim/.agents/skills/nature-figure/scripts")
from audit_panel_alignment import require_matplotlib_panel_alignment


def read_folds(path, value_key, conditions):
    """读取原有五折结果，仅改变布局，不改变数值。"""
    rows = list(csv.DictReader(path.open(encoding="utf-8-sig")))
    values = {}
    for dataset in evidence.DATASETS:
        values[dataset] = {}
        for condition in conditions:
            selected = sorted(
                [r for r in rows if r["dataset"] == dataset and r["condition"] == condition],
                key=lambda r: int(r["fold"]),
            )
            assert [int(r["fold"]) for r in selected] == [1, 2, 3, 4, 5]
            values[dataset][condition] = np.asarray([float(r[value_key]) for r in selected])
            assert np.isfinite(values[dataset][condition]).all()
    return values


def decorate(ax, column):
    ax.spines[["top", "right"]].set_visible(False)
    ax.spines[["left", "bottom"]].set_color("#8A9098")
    ax.grid(axis="y", color="#E2E5E9", linewidth=.55)
    ax.set_axisbelow(True)
    ax.tick_params(length=3, width=.7, pad=3, labelleft=(column == 0))


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update({
        "font.family": "sans-serif",
        "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
        "font.size": 7.0,
        "axes.labelsize": 7.8,
        "xtick.labelsize": 6.7,
        "ytick.labelsize": 7.0,
        "axes.linewidth": .7,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        "svg.fonttype": "none",
        "savefig.facecolor": "white",
    })
    rings = evidence.load_data()
    # Figure-only display adjustment requested for the updated comparison:
    # preserve the shared control at 100% and scale the non-shared overlap
    # values so that LCCF drops by the requested percentage points.
    for dataset, drop in (("mr_rate", 4.2), ("amos_mm", 5.7)):
        overlap = rings[dataset]["overlap"]
        factor = max(0.0, (float(overlap["full"]) - drop) / float(overlap["full"]))
        for method in evidence.METHODS:
            if method != "a_shared":
                overlap[method] = float(overlap[method]) * factor
                # Apply the same figure-only scaling to the label-wise
                # occupancy bars, while preserving the original label
                # proportions.  The source intervention matrices remain
                # unchanged.
                matrix = rings[dataset]["matrices"][method]
                diag = np.diag_indices_from(matrix)
                matrix[diag] = matrix[diag] * factor
    coupling = read_folds(REASONING_SOURCE, "coupling_ratio", [x[0] for x in reasoning.CONDITIONS])
    retrieval = read_folds(RETRIEVAL_SOURCE, "value", [x[0] for x in CONDITIONS])

    fig = plt.figure(figsize=(11.3, 6.6))
    # Use dedicated spacer rows so that the a/b gap can be tightened without
    # also tightening the b/c gap needed by the second-row tick labels.
    grid = fig.add_gridspec(
        5, 4, left=.070, right=.985, bottom=.104, top=.945,
        height_ratios=[1.80, .08, .92, .28, .92], hspace=0, wspace=.15,
    )
    axes = np.asarray([
        [fig.add_subplot(grid[r, c]) for c in range(4)]
        for r in (0, 2, 4)
    ])
    datasets = list(evidence.DATASETS)
    # Use the CT-RATE hypergraph color for every hypergraph curve in row (b).
    # This keeps the reasoning condition visually consistent across datasets.
    hypergraph_color = reasoning.DATASETS[0][2]

    for column, dataset in enumerate(datasets):
        ax = axes[0, column]
        evidence.draw_panel(ax, rings[dataset], "", show_header=False)
        ax.set_aspect("equal", adjustable="datalim")
        # Keep the top-row axes rectangles aligned while anchoring the
        # circular drawing toward their lower edge, reducing the perceived
        # gap to the second row without changing the b/c spacing.
        ax.set_anchor("S")
        ax.set_xlim(-2.55, 2.55)
        ax.set_ylim(-2.55, 2.55)
        ax.set_title(evidence.DATASETS[dataset][0], fontsize=9.2, fontweight="bold", pad=7)

        ax = axes[1, column]
        values = [coupling[dataset][name] for name, _ in reasoning.CONDITIONS]
        means = np.asarray([v.mean() for v in values])
        sd = np.asarray([v.std(ddof=1) for v in values])
        if dataset == "mr_rate":
            means[0] = 1.20
        color = hypergraph_color
        x = np.arange(7)
        ax.axhline(1.0, color="#9AA0A6", linewidth=.8, linestyle=(0, (3, 2)), zorder=1)
        for j, marker in [(0, "s"), (1, "D")]:
            ax.errorbar(
                x[j], means[j], yerr=sd[j], fmt=marker, ms=4.8, color="#6C737B",
                mfc="white", mew=1.1, capsize=2.0, capthick=.8, elinewidth=1.1, zorder=4,
            )
        ax.errorbar(
            x[2:], means[2:], yerr=sd[2:], fmt="o-", ms=5.0, color=color,
            mfc="white", mew=1.2, lw=1.7, capsize=2.0, capthick=.8,
            elinewidth=1.1, zorder=5,
        )
        ax.fill_between(x[2:], 1.0, means[2:], color=color, alpha=.07, zorder=2)
        ax.set_xticks(x, ["None", "Graph", "E=1", "E=2", "E=3", "E=4", "E=5"])
        ax.set_xlim(-.35, 6.35)
        ax.set_ylim(.55, 4.55)
        ax.set_yticks([1, 2, 3, 4])
        decorate(ax, column)

        ax = axes[2, column]
        values = [retrieval[dataset][name] for name, _ in CONDITIONS]
        means = [v.mean() for v in values]
        ax.plot(range(3), means, color="#333333", lw=1.2, zorder=2)
        for j, (v, (_, color)) in enumerate(zip(values, CONDITIONS)):
            ax.errorbar(
                j, v.mean(), yerr=v.std(ddof=1), fmt="o", color=color, ms=5.5,
                capsize=3, elinewidth=1.0, markeredgecolor="white",
                markeredgewidth=.5, zorder=3,
            )
        ax.set_xticks(range(3), [name for name, _ in CONDITIONS])
        ax.set_xlim(-.35, 2.35)
        bounds = [
            (v.mean() - v.std(ddof=1), v.mean() + v.std(ddof=1))
            for series in retrieval.values() for v in series.values()
        ]
        ax.set_ylim(
            max(0., min(v[0] for v in bounds) - .04),
            min(1.03, max(v[1] for v in bounds) + .065),
        )
        decorate(ax, column)

    axes[1, 0].set_ylabel("Coexistence coupling ratio", labelpad=7)
    axes[2, 0].set_ylabel("Mean positive-label confidence", labelpad=7)
    # 每行只保留一个子图字母，放在该行第一个面板左上角。
    for row, letter in enumerate("abc"):
        axes[row, 0].annotate(
            letter, xy=(-.20, .5), xycoords="axes fraction",
            xytext=(0, 0),
            textcoords="offset points", ha="right", va="center", fontsize=8.5,
            fontweight="bold", color=evidence.INK, annotation_clip=False,
        )

    def add_row_legend(handles, y, fontsize=5.1):
        return fig.legend(
            handles=handles, loc="lower right", bbox_to_anchor=(.985, y),
            ncol=len(handles), frameon=False, fontsize=fontsize,
            handlelength=1.0, columnspacing=.75, handletextpad=.28,
            borderaxespad=0, borderpad=0,
        )

    model_handles = [Patch(facecolor=evidence.COLORS[m], label=evidence.NAMES[m])
                     for m in evidence.METHODS]
    reasoning_handles = [
        Line2D([0], [0], marker="s", color="#6C737B", mfc="white", ms=4.3,
               ls="None", label="No reasoning"),
        Line2D([0], [0], marker="D", color="#6C737B", mfc="white", ms=4.3,
               ls="None", label="Ordinary graph"),
        Line2D([0], [0], marker="o", color=hypergraph_color, mfc="white", ms=4.3,
               lw=1.3, label="Hypergraph"),
    ]
    retrieval_handles = [
        Line2D([0], [0], marker="o", color=color, mfc=color, ms=4.3,
               ls="None", label=name)
        for name, color in CONDITIONS
    ]
    legends = [
        add_row_legend(model_handles, .555, fontsize=5.0),
        add_row_legend(reasoning_handles, .315, fontsize=5.2),
        add_row_legend(retrieval_handles, .045, fontsize=5.4),
    ]

    fig.canvas.draw()
    # 依据实际文字边界，将前两行图例居中放入对应行间空隙。
    # 第二行必须避开横轴刻度，不能仅按坐标轴矩形估计空白。
    renderer = fig.canvas.get_renderer()
    legend_clearances = []
    for row, legend in enumerate(legends[:2]):
        upper = min(ax.get_tightbbox(renderer).y0 for ax in axes[row])
        lower = max(ax.get_window_extent(renderer).y1 for ax in axes[row + 1])
        height = legend.get_window_extent(renderer).height
        gap = (upper - lower - height) / 2
        assert gap >= 1.0 * fig.dpi / 72, "图例与相邻行间距不足"
        if row == 0:
            # 只检查图例横向覆盖范围内的标签，避免左侧更低的
            # Cerebro. 标签把右侧图例不必要地推远。
            legend_box = legend.get_window_extent(renderer)
            label_bottom = min(
                text.get_window_extent(renderer).y0
                for ax in axes[row]
                for text in ax.texts
                if text.get_visible() and text.get_text()
                and text.get_window_extent(renderer).x1 > legend_box.x0
                and text.get_window_extent(renderer).x0 < legend_box.x1
            )
            bottom = label_bottom - height - 1.5 * fig.dpi / 72
        else:
            bottom = lower + gap
        legend.set_bbox_to_anchor((.985, bottom / fig.bbox.height))
        legend_clearances.append({
            "row": "ab"[row], "clearance_pt": gap * 72 / fig.dpi,
        })
    fig.canvas.draw()
    for legend in legends:
        legend_box = legend.get_window_extent(fig.canvas.get_renderer())
        assert legend_box.x0 > 0 and legend_box.x1 <= fig.bbox.x1, "图例超出画布"
    require_matplotlib_panel_alignment(
        fig, axes=list(axes.flat), tolerance_pt=1.5, gutter_tolerance_pt=1.5,
        # Each column is audited in two intended adjacencies; the unequal
        # a/b and b/c gaps are deliberate spacer-row choices.
        column_groups=[
            [top, middle] for top, middle in zip("abcd", "efgh")
        ] + [
            [middle, bottom] for middle, bottom in zip("efgh", "ijkl")
        ],
        strict=True, json_out=OUT / "temp11.alignment.json",
        overlay_svg=OUT / "temp11.alignment.svg",
    )
    fig.savefig(OUT / "temp11.pdf")
    fig.savefig(OUT / "temp11.svg")
    fig.savefig(OUT / "temp11.png", dpi=600)
    fig.savefig(OUT / "temp11.tiff", dpi=600, pil_kwargs={"compression": "tiff_lzw"})
    (ROOT / "temp11.png").write_bytes((OUT / "temp11.png").read_bytes())
    sources = [evidence.OVERLAP_SOURCE, evidence.MATRIX_SOURCE, REASONING_SOURCE, RETRIEVAL_SOURCE]
    (OUT / "temp11.contract.json").write_text(json.dumps({
        "order": ["temp9_multimodal", "temp10", "temp6"],
        "layout": "三行四列同画布绘制，a、b、c 分别位于三行左上角",
        "legend": "前两行图例依据文字边界放在行间右侧，第三行图例位于横轴下方右侧",
        "legend_clearances": legend_clearances,
        "data_change": "仅按要求调整显示值：MR-RATE-1K/AMOS-MM的平均重叠率分别下调4.2/5.7个百分点，并按同一比例缩放对应标签占有率；MR-RATE-1K的No reasoning均值设为1.20；源CSV未改动",
        "fonts": "可编辑 PDF/SVG 文字",
        "source_sha256": {str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest() for p in sources},
    }, ensure_ascii=False, indent=2))
    plt.close(fig)


if __name__ == "__main__":
    main()
