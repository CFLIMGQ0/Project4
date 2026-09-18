#!/usr/bin/env python3
"""在同一批 AMOS-MM 检查上分别绘制三类及全部七类器官异常的共现矩阵。"""

from __future__ import annotations

import argparse
from collections import Counter
import csv
import hashlib
import json
from pathlib import Path
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap, Normalize
import numpy as np
from tqdm import tqdm


ROOT = Path(__file__).resolve().parents[2]
AUDIT = ROOT / "outputs/amos_mm/leavs_label_audit"
OUTPUT = ROOT / "outputs/figures/amos_mm_label_cooccurrence"
KEYS = ["liver", "kidney", "gallbladder", "spleen", "bowel", "pancreas", "stomach"]
DISPLAY = ["Liver", "Kidney", "Gallbladder", "Spleen", "Bowel", "Pancreas", "Stomach"]
ZH = ["肝脏", "肾脏", "胆囊", "脾脏", "肠道", "胰腺", "胃"]


def read_csv(path):
    with path.open(encoding="utf-8-sig", newline="") as stream:
        return list(csv.DictReader(stream))


def write_json(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")


def load_and_verify():
    source = AUDIT / "hybrid_1687_organ_states.csv"
    rows = read_csv(source)
    ids = [row["scan_id"] for row in rows]
    assert len(ids) == len(set(ids)) == 1687
    states = np.array([[int(row[key]) for key in KEYS] for row in rows], dtype=np.int64)
    assert set(np.unique(states)) <= {-3, -2, -1, 0, 1}
    # 此指示矩阵仅表示“有明确阳性记录”，其余状态不是临床阴性标签。
    positive_indicator = (states == 1).astype(np.int64)
    matrix = positive_indicator.T @ positive_indicator
    direct = np.array([[np.count_nonzero((states[:, i] == 1) & (states[:, j] == 1))
                        for j in range(7)] for i in range(7)], dtype=np.int64)
    assert np.array_equal(matrix, direct) and np.array_equal(matrix, matrix.T)
    summary_path = AUDIT / "summary.json"
    expected = json.loads(summary_path.read_text())["cohorts"]["hybrid_1687"]
    for index, key in enumerate(KEYS):
        assert int(matrix[index, index]) == expected["organ_stats"][key]["positive"]
    for pair in expected["pairs"]:
        i, j = [KEYS.index(key) for key in pair["labels"]]
        assert int(matrix[i, j]) == pair["both_positive"]
    human = read_csv(AUDIT / "human_test_200_organ_states.csv")
    development = read_csv(AUDIT / "auto_development_1487_organ_states.csv")
    human_ids, development_ids = {r["scan_id"] for r in human}, {r["scan_id"] for r in development}
    assert len(human_ids) == 200 and len(development_ids) == 1487
    assert not (human_ids & development_ids) and human_ids | development_ids == set(ids)
    combined = {row["scan_id"]: row for row in development + human}
    assert all(all(row[key] == combined[row["scan_id"]][key] for key in KEYS) for row in rows)
    evidence_path = AUDIT / "hybrid_1687_evidence_states.csv"
    evidence = read_csv(evidence_path)
    evidence_by_id = {row["scan_id"]: row for row in evidence}
    assert set(evidence_by_id) == set(ids)
    assert all(all((int(row[key]) == 1) == (evidence_by_id[row["scan_id"]][key] == "positive")
                   for key in KEYS) for row in rows)
    verified_path = ROOT / "outputs/amos_mm/download/verified_files.jsonl"
    verified = {json.loads(line)["scan_id"] for line in verified_path.read_text().splitlines() if line.strip()}
    assert set(ids) == verified
    submatrix = positive_indicator[:, :3].T @ positive_indicator[:, :3]
    assert np.array_equal(submatrix, matrix[:3, :3])
    result = {
        "dataset": "AMOS-MM", "examinations": len(rows), "counting_unit": "唯一 CT 检查编号",
        "label_source": {"automatic_development": 1487, "human_majority_test": 200},
        "organ_groups": KEYS, "organ_names_zh": ZH,
        "grouping": "双肾合并、大小肠合并；全部为此前定义的七类器官异常",
        "matrix_counts": matrix.tolist(), "matrix_percent": (100 * matrix / len(rows)).tolist(),
        "three_label_counts": submatrix.tolist(), "three_label_percent": (100 * submatrix / len(rows)).tolist(),
        "three_label_is_exact_principal_submatrix": True,
        "co_positive_cases_three_labels": int(np.count_nonzero(positive_indicator[:, :3].sum(1) >= 2)),
        "co_positive_cases_seven_labels": int(np.count_nonzero(positive_indicator.sum(1) >= 2)),
        "raw_state_counts": {key: dict(Counter(str(int(v)) for v in states[:, i])) for i, key in enumerate(KEYS)},
        "evidence_state_counts": {key: dict(Counter(row[key] for row in evidence)) for key in KEYS},
        "unknown_as_negative": False, "sampled": False, "excluded_examinations": 0,
        "percent_definition": "明确阳性或明确共阳性检查数 / 全部 1687 检查；不推断未知记录的真实临床状态",
        "previous_overview_difference": "此前八数据集总览中 AMOS-MM 使用 1211 例完整三标签候选；本次两图统一使用全部1687例",
        "checks": {"independent_cell_counts": True, "source_marginals_and_pairs": True,
                   "annotation_source_join": True, "downloaded_case_ids_match": True},
        "source_sha256": {str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest()
                          for p in [source, summary_path, evidence_path,
                                    AUDIT / "human_test_200_organ_states.csv", AUDIT / "auto_development_1487_organ_states.csv",
                                    verified_path]},
    }
    return matrix, result


def color_and_contrast(rgb):
    def luminance(color):
        x = np.asarray(color)
        linear = np.where(x <= 0.04045, x / 12.92, ((x + 0.055) / 1.055) ** 2.4)
        return float(linear @ [0.2126, 0.7152, 0.0722])
    background = luminance(rgb)
    dark = luminance([21 / 255, 54 / 255, 77 / 255])
    choices = [("#15364d", (background + 0.05) / (dark + 0.05)),
               ("white", 1.05 / (background + 0.05)), ("black", (background + 0.05) / 0.05)]
    return next((color, ratio) for color, ratio in choices if ratio >= 4.5)


def draw(matrix, n, prefix, output, audit_dir):
    sys.path.insert(0, str(audit_dir))
    from audit_panel_alignment import require_matplotlib_panel_alignment

    plt.rcParams.update({"font.family": "sans-serif", "font.sans-serif": ["DejaVu Sans", "Arial", "Helvetica"],
                         "font.size": 10, "pdf.fonttype": 42, "svg.fonttype": "none",
                         "axes.spines.top": False, "axes.spines.right": False, "savefig.facecolor": "white"})
    # 两张单独图采用相同物理画布、矩阵外框及统一色标。
    fig, ax = plt.subplots(figsize=(7.2, 6.7), dpi=300)
    fig.subplots_adjust(left=0.175, right=0.835, bottom=0.15, top=0.965)
    count = len(matrix)
    percentages = matrix * 100 / n
    assert (percentages >= 0).all() and (percentages <= 60).all()
    cmap = LinearSegmentedColormap.from_list("amos_cooccurrence_blue", ["#f5f9fd", "#c3dcef", "#76afd2", "#347dab", "#14466c"])
    norm = Normalize(vmin=0, vmax=60)
    artist = ax.imshow(percentages, cmap=cmap, norm=norm, interpolation="nearest", aspect="equal")
    ax.set_xticks(range(count), DISPLAY[:count], fontsize=9.5)
    ax.set_yticks(range(count), DISPLAY[:count], fontsize=9.5)
    ax.set_xlabel("Organ abnormality", fontsize=10, labelpad=12)
    ax.set_ylabel("Organ abnormality", fontsize=10, labelpad=12)
    ax.tick_params(axis="both", which="both", length=0, pad=7)
    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.set_xticks(np.arange(-0.5, count, 1), minor=True)
    ax.set_yticks(np.arange(-0.5, count, 1), minor=True)
    ax.grid(which="minor", color="white", linewidth=1.3)
    contrasts = []
    for i in range(count):
        for j in range(count):
            color, ratio = color_and_contrast(cmap(norm(percentages[i, j]))[:3])
            contrasts.append(ratio)
            ax.text(j, i - 0.10, str(int(matrix[i, j])), ha="center", va="center", color=color,
                    fontsize=16 if count == 3 else 10, fontweight="bold")
            ax.text(j, i + 0.18, f"({percentages[i, j]:.1f}%)", ha="center", va="center", color=color,
                    fontsize=11 if count == 3 else 8)
    bar_ax = fig.add_axes([0.86, 0.24, 0.018, 0.62])
    bar = fig.colorbar(artist, cax=bar_ax, ticks=[0, 10, 20, 30, 40, 50, 60])
    bar.ax.set_yticklabels([f"{value}%" for value in [0, 10, 20, 30, 40, 50, 60]])
    bar.ax.tick_params(length=2.5, width=0.6, labelsize=9)
    bar.outline.set_visible(False)
    bar.set_label("Recorded-positive examinations (%)", fontsize=9, labelpad=9)
    qa = output / "qa"
    qa.mkdir(exist_ok=True)
    fig.canvas.draw()
    require_matplotlib_panel_alignment(fig, axes=[ax], json_out=qa / f"{prefix}.alignment.json",
                                       require_panel_labels=False, tolerance_pt=1.5, strict=True)
    # 内部矢量文件用于自动文字校验；正式交付仅 PNG。
    fig.savefig(qa / f"{prefix}.svg")
    fig.savefig(qa / f"{prefix}.pdf", dpi=300)
    fig.savefig(output / f"{prefix}.png", dpi=300)
    plt.close(fig)
    write_json(qa / f"{prefix}.contrast.json", {"minimum_ratio": min(contrasts), "required_minimum": 4.5,
                                               "cells": len(contrasts), "passed": min(contrasts) >= 4.5})


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=OUTPUT)
    parser.add_argument("--audit-dir", type=Path, required=True)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    matrix, summary = load_and_verify()
    write_json(args.output_dir / "summary.json", summary)
    variants = [("amos_mm_three_label_cooccurrence", matrix[:3, :3]),
                ("amos_mm_all_seven_label_cooccurrence", matrix)]
    for prefix, values in tqdm(variants, desc="绘制 AMOS-MM 共现矩阵", unit="张"):
        with (args.output_dir / f"{prefix}.csv").open("w", encoding="utf-8-sig", newline="") as stream:
            writer = csv.writer(stream)
            writer.writerow(["row_label", "column_label", "count", "percent_of_1687", "cell_type"])
            for i in range(len(values)):
                for j in range(len(values)):
                    writer.writerow([KEYS[i], KEYS[j], int(values[i, j]), f"{100 * values[i, j] / 1687:.8f}",
                                     "marginal_positive" if i == j else "pairwise_co_positive"])
        draw(values, summary["examinations"], prefix, args.output_dir, args.audit_dir)
    print("已生成两张 PNG；两图使用同一批 1687 例检查，且三标签矩阵与七标签相应子矩阵一致。")
    print(json.dumps({k: summary[k] for k in ["three_label_counts", "matrix_counts"]}, ensure_ascii=False))


if __name__ == "__main__":
    main()
