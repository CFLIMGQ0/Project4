#!/usr/bin/env python3
"""从既有CT-RATE逐切片结果绘制单病例位置分布三联图。"""
from __future__ import annotations

import argparse
import csv
from pathlib import Path
import random
import sys

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np


FIGURE_SKILL_SCRIPTS = Path("/home/Lim/.agents/skills/nature-figure/scripts")
if str(FIGURE_SKILL_SCRIPTS) not in sys.path:
    sys.path.insert(0, str(FIGURE_SKILL_SCRIPTS))


ROOT = Path(__file__).resolve().parents[2]
SOURCE = ROOT / "outputs/ct_rate_680/position_baselines/position_recovery"
FRACTIONS = (0.25, 0.50, 0.75)
SEED = 2026
CONNECTOR_POSITIONS = (0, 8, 16, 24, 32, 40, 48, 56, 63)


def load_rows():
    with (SOURCE / "per_slice_results.csv").open(encoding="utf-8-sig", newline="") as stream:
        return list(csv.DictReader(stream))


def load_repeat_metrics():
    with (SOURCE / "repeat_metrics.csv").open(encoding="utf-8-sig", newline="") as stream:
        return list(csv.DictReader(stream))


def choose_case(rows, selection_seed):
    case_ids = sorted({int(row["case_index"]) for row in rows})
    if not case_ids:
        raise ValueError("逐切片结果中没有可选病例")
    selector = random.Random(selection_seed)
    return int(selector.choice(case_ids))


def select_condition(rows, case_index, fraction):
    selected = [
        row for row in rows
        if int(row["case_index"]) == case_index
        and np.isclose(float(row["deletion_fraction"]), fraction)
        and int(row["seed"]) == SEED
    ]
    selected.sort(key=lambda row: int(row["selected_position"]))
    if len(selected) != 64:
        raise ValueError(f"case={case_index}, deletion={fraction:.0%}没有完整64张切片记录")
    positions = [int(row["selected_position"]) for row in selected]
    if positions != list(range(64)):
        raise ValueError(f"case={case_index}, deletion={fraction:.0%}输入顺序不完整")
    for row in selected:
        for field in ("true_position", "baseline_position", "model_position"):
            if not np.isfinite(float(row[field])):
                raise ValueError(f"位置数值异常：{field}, case={case_index}, deletion={fraction:.0%}")
    return selected


def get_metrics(repeat_rows, case_index, fraction):
    matches = [
        row for row in repeat_rows
        if int(row["case_index"]) == case_index
        and np.isclose(float(row["deletion_fraction"]), fraction)
        and int(row["seed"]) == SEED
    ]
    if len(matches) != 1:
        raise ValueError(f"case={case_index}, deletion={fraction:.0%}没有唯一指标记录")
    row = matches[0]
    return {
        "original_pre": float(row["baseline_PRE"]),
        "original_gre": float(row["baseline_GRE"]),
        "acpe_pre": float(row["model_PRE"]),
        "acpe_gre": float(row["model_GRE"]),
    }


def write_csv(path, condition_rows, metrics, case_index, patient_id, selection_seed):
    fields = [
        "case_index", "patient_id", "selection_seed", "deletion_fraction", "actual_delete_fraction",
        "sampling_seed", "selected_position", "raw_index", "true_position", "original_pe_position",
        "acpe_position", "acpe_raw_context_coordinate", "connector_position",
        "original_pe_PRE", "original_pe_GRE", "acpe_PRE", "acpe_GRE",
    ]
    with path.open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for group, group_metrics in zip(condition_rows, metrics):
            for row in group:
                writer.writerow({
                    "case_index": case_index,
                    "patient_id": patient_id,
                    "selection_seed": selection_seed,
                    "deletion_fraction": row["deletion_fraction"],
                    "actual_delete_fraction": row["actual_delete_fraction"],
                    "sampling_seed": row["seed"],
                    "selected_position": row["selected_position"],
                    "raw_index": row["raw_index"],
                    "true_position": row["true_position"],
                    "original_pe_position": row["baseline_position"],
                    "acpe_position": row["model_position"],
                    "acpe_raw_context_coordinate": row["model_context_coordinate"],
                    "connector_position": int(row["selected_position"]) in CONNECTOR_POSITIONS,
                    "original_pe_PRE": group_metrics["original_pre"],
                    "original_pe_GRE": group_metrics["original_gre"],
                    "acpe_PRE": group_metrics["acpe_pre"],
                    "acpe_GRE": group_metrics["acpe_gre"],
                })


def draw(output_png, output_pdf, condition_rows, metrics, case_index, patient_id, selection_seed):
    from audit_panel_alignment import require_matplotlib_panel_alignment

    mpl.rcParams.update({
        "font.family": "sans-serif",
        "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans", "sans-serif"],
        "svg.fonttype": "none",
        "pdf.fonttype": 42,
        "font.size": 7,
        "axes.spines.right": False,
        "axes.spines.top": False,
        "axes.linewidth": 0.8,
    })
    colors = {"truth": "#333333", "original": "#4477AA", "acpe": "#CC6677"}
    figure, axes = plt.subplots(1, 3, figsize=(7.2, 3.15), sharex=True, sharey=True)
    y_values = {"truth": 2.0, "original": 1.0, "acpe": 0.0}
    y_labels = ["Ground truth", "Original PE", "ACPE"]
    patient_label = f"case_index={case_index}; patient_id={patient_id}"

    for panel_index, (axis, fraction, rows) in enumerate(zip(axes, FRACTIONS, condition_rows)):
        x_truth = np.asarray([float(row["true_position"]) for row in rows])
        x_original = np.asarray([float(row["baseline_position"]) for row in rows])
        x_acpe = np.asarray([float(row["model_position"]) for row in rows])
        selected = np.asarray([int(row["selected_position"]) for row in rows])
        connector_mask = np.isin(selected, CONNECTOR_POSITIONS)

        for index in np.flatnonzero(connector_mask):
            axis.plot(
                [x_truth[index], x_original[index], x_acpe[index]],
                [y_values["truth"], y_values["original"], y_values["acpe"]],
                color="#B8B8B8", linewidth=0.55, alpha=0.65, zorder=1,
            )
        axis.scatter(x_truth, np.full_like(x_truth, y_values["truth"]), s=13,
                     color=colors["truth"], marker="o", label="Ground truth", zorder=3)
        axis.scatter(x_original, np.full_like(x_original, y_values["original"]), s=14,
                     color=colors["original"], marker="s", label="Original PE", zorder=3)
        axis.scatter(x_acpe, np.full_like(x_acpe, y_values["acpe"]), s=16,
                     color=colors["acpe"], marker="^", label="ACPE", zorder=3)
        axis.axhline(y_values["truth"], color="#E2E2E2", linewidth=0.7, zorder=0)
        axis.axhline(y_values["original"], color="#E2E2E2", linewidth=0.7, zorder=0)
        axis.axhline(y_values["acpe"], color="#E2E2E2", linewidth=0.7, zorder=0)
        axis.set_xlim(0.0, 1.0)
        axis.set_ylim(-0.45, 2.45)
        axis.set_xticks(np.linspace(0, 1, 6))
        axis.set_yticks([2, 1, 0])
        axis.set_yticklabels(y_labels)
        axis.set_xlabel("Normalized position")
        axis.set_title(
            f"{fraction:.0%} deletion\n"
            f"Original PE: PRE {metrics[panel_index]['original_pre']:.3f}; GRE {metrics[panel_index]['original_gre']:.3f}\n"
            f"ACPE: PRE {metrics[panel_index]['acpe_pre']:.3f}; GRE {metrics[panel_index]['acpe_gre']:.3f}",
            fontsize=7.2, pad=8,
        )
        axis.text(-0.12, 1.04, chr(ord("a") + panel_index), transform=axis.transAxes,
                  fontsize=9, fontweight="bold", va="bottom", ha="left")
        axis.grid(axis="x", color="#EEEEEE", linewidth=0.6)
        axis.tick_params(axis="both", length=2.5, width=0.6, pad=2)
        axis.set_axisbelow(True)

    figure.suptitle(f"CT-RATE position distribution: {patient_label}", fontsize=8.5, y=1.10)
    figure.text(0.5, 1.045, f"selection seed={selection_seed}; sampling seed=2026",
                ha="center", va="center", fontsize=7)
    handles, labels = axes[0].get_legend_handles_labels()
    figure.legend(handles, labels, loc="upper center", bbox_to_anchor=(0.5, 1.015),
                  ncol=3, frameon=False, handletextpad=0.4, columnspacing=1.0)
    figure.subplots_adjust(left=0.12, right=0.99, bottom=0.18, top=0.76, wspace=0.18)

    require_matplotlib_panel_alignment(
        figure,
        json_out="/tmp/ct_position_case_figure.alignment.json",
        overlay_svg="/tmp/ct_position_case_figure.alignment.svg",
        tolerance_pt=1.5,
        gutter_tolerance_pt=1.5,
        require_panel_labels=True,
        strict=True,
    )
    figure.savefig(output_png, dpi=600, bbox_inches="tight")
    figure.savefig(output_pdf, bbox_inches="tight")
    plt.close(figure)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=SOURCE)
    parser.add_argument("--selection-seed", type=int, default=2026)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    rows = load_rows()
    repeat_rows = load_repeat_metrics()
    case_index = choose_case(rows, args.selection_seed)
    condition_rows = [select_condition(rows, case_index, fraction) for fraction in FRACTIONS]
    patient_ids = {row["patient_id"] for group in condition_rows for row in group}
    if len(patient_ids) != 1:
        raise ValueError("三个删除比例没有使用同一病例")
    patient_id = next(iter(patient_ids))
    metrics = [get_metrics(repeat_rows, case_index, fraction) for fraction in FRACTIONS]
    output_stem = args.output_dir / f"selected_case_{case_index:04d}_position_distribution"
    write_csv(output_stem.with_suffix(".csv"), condition_rows, metrics,
              case_index, patient_id, args.selection_seed)
    draw(output_stem.with_suffix(".png"), output_stem.with_suffix(".pdf"),
         condition_rows, metrics, case_index, patient_id, args.selection_seed)
    print(f"case_index={case_index}; patient_id={patient_id}")
    print(f"png={output_stem.with_suffix('.png')}")
    print(f"pdf={output_stem.with_suffix('.pdf')}")
    print(f"csv={output_stem.with_suffix('.csv')}")


if __name__ == "__main__":
    main()
