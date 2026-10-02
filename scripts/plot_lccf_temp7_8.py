#!/usr/bin/env python3
"""Render two additional LCCF mechanism figures from the audited public results.

temp7 combines visual self-impact and label--text matching diagnostics.
temp8 combines subgroup F1 for coexistence reasoning with a text-intervention
F1 diagnostic.  It only reads the existing five-fold NPZ files and does not
train models or modify the manuscript.
"""
from __future__ import annotations

import csv
import json
import string
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "outputs/lccf_public_evidence_20261001"
sys.path.insert(0, str(ROOT / "src/scripts"))
from evaluate_lccf_public_figures import DATASETS
from plot_lccf_public_figures import load, style, decorate

SKILL = Path("/home/Lim/.agents/skills/nature-figure/scripts")
sys.path.insert(0, str(SKILL))
from audit_panel_alignment import require_matplotlib_panel_alignment

SOURCE: list[dict] = []


def collapse(case: np.ndarray, values: np.ndarray) -> np.ndarray:
    return np.stack([values[case == c].mean(axis=0) for c in np.unique(case)])


def positive_confidence(prob: np.ndarray, labels: np.ndarray) -> np.ndarray:
    return (prob * labels).sum(1) / np.maximum(labels.sum(1), 1)


def selected(labels: np.ndarray, group: str) -> np.ndarray:
    count = labels.sum(1)
    if group == "Single-positive":
        return count == 1
    if group == "Co-positive":
        return count >= 2
    return count > 0


def fold_values(d: dict, field: str, group: str) -> np.ndarray:
    values = positive_confidence(d[field], d["labels"])
    keep = selected(d["labels"], group)
    return np.array([values[keep & (d["fold"] == f)].mean() for f in range(1, 6)])


def fold_f1(d: dict, field: str, group: str) -> np.ndarray:
    keep = selected(d["labels"], group)
    y = d["labels"]
    p = d[field] >= 0.5
    result = []
    for f in range(1, 6):
        use = keep & (d["fold"] == f)
        scores = []
        for label in range(y.shape[1]):
            yt, yp = y[use, label].astype(bool), p[use, label].astype(bool)
            tp = np.sum(yt & yp)
            fp = np.sum(~yt & yp)
            fn = np.sum(yt & ~yp)
            denom = 2 * tp + fp + fn
            scores.append(2 * tp / denom if denom else 0.0)
        result.append(np.mean(scores))
    return np.asarray(result)


def self_impact(d: dict, route: int) -> np.ndarray:
    """Per-examination mean diagonal share after deleting source-label evidence."""
    delta = np.abs(d["deletion_signed"][:, route])  # [case, source, target]
    labels = d["labels"] > 0
    values = []
    for i in range(len(delta)):
        source = np.flatnonzero(labels[i])
        if not len(source) or not np.isfinite(delta[i]).all():
            values.append(np.nan)
            continue
        shares = []
        for s in source:
            total = delta[i, s].sum()
            shares.append(delta[i, s, s] / total if total > 1e-12 else np.nan)
        values.append(np.nanmean(shares))
    return np.asarray(values)


def fold_metric(values: np.ndarray, folds: np.ndarray) -> np.ndarray:
    return np.array([np.nanmean(values[folds == f]) for f in range(1, 6)])


def save_figure(fig: plt.Figure, axes: np.ndarray, name: str) -> None:
    fig.canvas.draw()
    require_matplotlib_panel_alignment(
        fig, axes=list(np.asarray(axes).flat), tolerance_pt=1.5,
        gutter_tolerance_pt=1.5, strict=True,
        json_out=OUT / f"{name}.alignment.json")
    fig.savefig(OUT / f"{name}.pdf", dpi=300)
    fig.savefig(OUT / f"{name}.svg", dpi=300)
    fig.savefig(OUT / f"{name}.png", dpi=300)
    (ROOT / f"{name}.png").write_bytes((OUT / f"{name}.png").read_bytes())
    plt.close(fig)


def fig_temp7(raw: dict) -> None:
    """Evidence selectivity: visual self-impact and label--text matching."""
    fig, axes = plt.subplots(2, 3, figsize=(10.2, 6.3), sharex=False)
    fig.subplots_adjust(left=.085, right=.995, top=.86, bottom=.15, wspace=.20, hspace=.58)
    colors = ["#999999", "#C44748"]
    labels = ["Shared control", "Label-wise LCCF"]
    for col, ds in enumerate(DATASETS):
        d = raw[ds]["full"]
        # Top: own-label impact share after removing the source label's top evidence.
        top = []
        for route, name in enumerate(labels):
            vals = self_impact(d, route)
            fold = fold_metric(vals, d["fold"])
            top.append(fold)
            SOURCE.append({"figure": "temp7", "panel": "visual self-impact",
                           "dataset": ds, "condition": name,
                           "mean": float(fold.mean()), "sd": float(fold.std(ddof=1))})
        ax = axes[0, col]
        means = [v.mean() * 100 for v in top]
        errs = [v.std(ddof=1) * 100 for v in top]
        ax.bar(np.arange(2), means, yerr=errs, capsize=3, color=colors,
               edgecolor="white", linewidth=.5, error_kw={"elinewidth": .9})
        ax.set_ylim(0, 100); ax.set_yticks(np.arange(0, 101, 20)); decorate(ax)
        ax.set_title(f"{string.ascii_lowercase[col]}  {DATASETS[ds][0]}", loc="left", fontweight="bold", pad=10)
        ax.set_xticks(range(2), ["Shared", "Label-wise"])
        # Bottom: text evidence intervention, retaining both single- and co-positive cases.
        conditions = [("prob", "Correct", "#C44748"), ("pooled_prob", "Pooled", "#999999"),
                      ("cross_prob", "Cross-label", "#4C78A8")]
        ax = axes[1, col]
        width = .23
        for j, (field, name, color) in enumerate(conditions):
            means, errs = [], []
            for k, group in enumerate(["Single-positive", "Co-positive"]):
                fold = fold_values(d, field, group)
                means.append(fold.mean()); errs.append(fold.std(ddof=1))
                SOURCE.append({"figure": "temp7", "panel": "text matching", "dataset": ds,
                               "condition": name, "group": group,
                               "mean": float(fold.mean()), "sd": float(fold.std(ddof=1))})
            ax.bar(np.arange(2) + (j - 1) * width, means, width, yerr=errs,
                   capsize=2.5, color=color, edgecolor="white", linewidth=.5,
                   error_kw={"elinewidth": .8}, label=name)
        ax.set_ylim(0, 1.0); ax.set_yticks(np.arange(0, 1.01, .2)); decorate(ax)
        ax.set_xticks(range(2), ["Single-positive", "Co-positive"])
    axes[0, 0].set_ylabel("Own-label decision impact (%)")
    axes[1, 0].set_ylabel("Positive-label confidence")
    handles, leg = axes[1, 0].get_legend_handles_labels()
    fig.legend(handles, leg, loc="upper center", bbox_to_anchor=(.55, .985), ncol=3, frameon=False)
    fig.text(.52, .055, "Top: fraction of decision change remaining on the source label; bottom: fixed visual pathway with text evidence interventions.",
             ha="center", fontsize=7.6, color="#555555")
    save_figure(fig, axes, "temp7")


def fig_temp8(raw: dict) -> None:
    """Coexistence recognition and selective fusion as F1 diagnostics."""
    fig, axes = plt.subplots(2, 3, figsize=(10.2, 6.3), sharex=False)
    fig.subplots_adjust(left=.085, right=.995, top=.86, bottom=.15, wspace=.20, hspace=.58)
    methods = [("b_none", "No reasoning", "#999999"), ("b_graph", "Ordinary graph", "#E5A619"),
               ("full", "Label hypergraph", "#C44748")]
    groups = ["Single-positive", "Co-positive"]
    for col, ds in enumerate(DATASETS):
        ax = axes[0, col]
        width = .23
        for j, (method, name, color) in enumerate(methods):
            means, errs = [], []
            d = raw[ds][method]
            for group in groups:
                fold = fold_f1(d, "prob", group)
                means.append(fold.mean() * 100); errs.append(fold.std(ddof=1) * 100)
                SOURCE.append({"figure": "temp8", "panel": "coexistence F1", "dataset": ds,
                               "condition": name, "group": group,
                               "mean": float(fold.mean()), "sd": float(fold.std(ddof=1))})
            ax.bar(np.arange(2) + (j - 1) * width, means, width, yerr=errs,
                   capsize=2.5, color=color, edgecolor="white", linewidth=.5,
                   error_kw={"elinewidth": .8}, label=name)
        ax.set_ylim(0, 100); ax.set_yticks(np.arange(0, 101, 20)); decorate(ax)
        ax.set_xticks(range(2), groups)
        ax.set_title(f"{string.ascii_lowercase[col]}  {DATASETS[ds][0]}", loc="left", fontweight="bold", pad=10)
        # Bottom: F1 after replacing each label's text evidence by pooled or cross-label evidence.
        ax = axes[1, col]
        conditions = [("prob", "Correct", "#C44748"), ("pooled_prob", "Pooled", "#999999"),
                      ("cross_prob", "Cross-label", "#4C78A8")]
        means, errs = [], []
        d = raw[ds]["full"]
        for field, name, color in conditions:
            fold = fold_f1(d, field, "All-positive")
            means.append(fold.mean() * 100); errs.append(fold.std(ddof=1) * 100)
            SOURCE.append({"figure": "temp8", "panel": "text intervention F1", "dataset": ds,
                           "condition": name, "group": "All-positive",
                           "mean": float(fold.mean()), "sd": float(fold.std(ddof=1))})
        x = np.arange(3)
        ax.bar(x, means, .52, yerr=errs, capsize=3,
               color=[c for _, _, c in conditions], edgecolor="white", linewidth=.5,
               error_kw={"elinewidth": .9})
        ax.set_ylim(0, 100); ax.set_yticks(np.arange(0, 101, 20)); decorate(ax)
        ax.set_xticks(x, [n for _, n, _ in conditions])
    axes[0, 0].set_ylabel("Subgroup Macro-F1 @0.5 (%)")
    axes[1, 0].set_ylabel("Macro-F1 @0.5 (%)")
    handles, leg = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, leg, loc="upper center", bbox_to_anchor=(.53, .985), ncol=3, frameon=False)
    fig.text(.52, .055, "Top: single- versus co-positive examinations; bottom: label-specific text evidence interventions.",
             ha="center", fontsize=7.8, color="#555555")
    save_figure(fig, axes, "temp8")


def main() -> None:
    style()
    raw = load()
    fig_temp7(raw)
    fig_temp8(raw)
    keys = sorted(set(k for row in SOURCE for k in row))
    with (OUT / "temp7_temp8_source_data.csv").open("w", newline="", encoding="utf-8-sig") as stream:
        writer = csv.DictWriter(stream, fieldnames=keys)
        writer.writeheader(); writer.writerows(SOURCE)
    (OUT / "temp7_temp8_completed.json").write_text(json.dumps({
        "figures": ["temp7.png", "temp8.png"], "datasets": list(DATASETS), "folds": 5,
        "source_data": "temp7_temp8_source_data.csv",
        "note": "仅使用已审计的公开五折NPZ；未训练、未修改论文。"}, ensure_ascii=False, indent=2) + "\n")
    print("temp7/temp8 已生成：", OUT)


if __name__ == "__main__":
    main()
