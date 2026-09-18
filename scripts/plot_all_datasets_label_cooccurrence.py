#!/usr/bin/env python3
"""绘制八个检查级三标签任务的共现矩阵；只读任务标签，不使用预测概率。"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import sys
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap, Normalize
import numpy as np
from tqdm import tqdm
import yaml


ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "outputs/figures/all_datasets_label_cooccurrence"
GASTRO_LABELS = ["label_esophageal_smt", "label_esophageal_mucosal_or_tumor", "label_gastritis"]
GASTRO_NAMES = ["Esophageal SMT", "Esophageal mucosal lesion", "Gastritis"]
GASTRO_TICKS = ["Esophageal\nSMT", "Esophageal\nmucosal lesion", "Gastritis"]


@dataclass
class Cohort:
    key: str
    name: str
    labels: list[str]
    ticks: list[str]
    y: np.ndarray
    sources: list[Path]
    notes: dict


def read_csv(path):
    with path.open(encoding="utf-8-sig", newline="") as stream:
        return list(csv.DictReader(stream))


def binary_labels(rows, columns):
    values = np.array([[float(row[c]) for c in columns] for row in rows])
    if values.ndim != 2 or values.shape[1] != 3 or not np.isin(values, [0, 1]).all():
        raise ValueError("本图只接受三项完整二值标签，不能将未知值强行归零")
    return values.astype(np.int64)


def assert_unique(values, name):
    if len(values) != len(set(values)):
        raise ValueError(f"{name} 检查编号重复")


def compare_oof_truth(actual, path, columns):
    rows = read_csv(path)
    assert_unique([int(r["patient_id"]) for r in rows], path.parent.name)
    saved = {int(r["patient_id"]): y for r, y in zip(rows, binary_labels(rows, columns))}
    if set(actual) != set(saved) or any(not np.array_equal(actual[k], saved[k]) for k in actual):
        raise ValueError("重新计算的检查标签与既有实验的真实标签不符")


def load_cohorts():
    cohorts = []
    source = ROOT / "datasets/task_data/task2/gastro_multilabel_task_datalist.csv"
    config_path = ROOT / "src/configs/task3/t3_main_model.yaml"
    config = yaml.safe_load(config_path.read_text())
    rows = read_csv(source)
    known_titles = {t for v in config["datasets"].values() for t in v["report_titles"]}
    outside = [r for r in rows if r["reportTitle"].strip() not in known_titles]
    names = [("regular_white_light", "WLE", 977), ("chromoscopic", "Chromoscopic", 340),
             ("surgical", "Surgical", 965), ("ultrasound", "EUS", 579)]
    for key, name, expected in tqdm(names, desc="核对胃镜检查标签", unit="数据集"):
        selected = [r for r in rows if r["reportTitle"].strip() in config["datasets"][key]["report_titles"]]
        assert_unique([r["exam_dir"] for r in selected], name)
        if len(selected) != expected:
            raise ValueError(f"{name} 的任务样本数改变，请重新确认")
        notes = {"unit": "唯一检查目录", "source_total_rows": len(rows), "scope": "原始任务分布，未重平衡",
                 "unique_patient_ids": len({r["patient_id"] for r in selected}),
                 "outside_four_task_records": len(outside),
                 "outside_four_task_titles": dict(Counter(r["reportTitle"] for r in outside))}
        cohorts.append(Cohort(key, name, GASTRO_NAMES, GASTRO_TICKS, binary_labels(selected, GASTRO_LABELS),
                              [source, config_path], notes))

    source = ROOT / "datasets/physionet_ct_ich/computed-tomography-images-for-intracranial-hemorrhage-detection-and-segmentation-1.0.0/hemorrhage_diagnosis.csv"
    rows = read_csv(source)
    labels = binary_labels(rows, ["Intraparenchymal", "Epidural", "Fracture_Yes_No"])
    assert_unique([(r["PatientNumber"], r["SliceNumber"]) for r in rows], "CT-ICH 切片")
    grouped = {}
    for row, target in zip(rows, labels):
        key = int(row["PatientNumber"])
        grouped[key] = np.maximum(grouped.get(key, np.zeros(3, dtype=np.int64)), target)
    comparison = ROOT / "outputs/physionet_ct_ich/table2_image_baselines/mean_pooling_oof_predictions.csv"
    compare_oof_truth(grouped, comparison, ["true_IPH", "true_EDH", "true_Fracture"])
    if len(grouped) != 82:
        raise ValueError("CT-ICH 检查数量改变")
    cohorts.append(Cohort("ct_ich", "CT-ICH", ["IPH", "EDH", "Fracture"], ["IPH", "EDH", "Fracture"],
                          np.stack([grouped[k] for k in sorted(grouped)]), [source, comparison],
                          {"unit": "患者的一次检查", "source_slice_rows": len(rows),
                           "label_aggregation": "患者内全部切片按标签取逻辑或", "oof_true_label_check": "逐例一致"}))

    source = ROOT / "datasets/cq500/raw/reads.csv"
    rows = read_csv(source)
    assert_unique([r["name"] for r in rows], "CQ500")
    target_columns = ["IPH", "MassEffect", "MidlineShift"]
    readers = np.stack([binary_labels(rows, [f"R{i}:{c}" for c in target_columns]) for i in (1, 2, 3)])
    labels = (readers.sum(axis=0) >= 2).astype(np.int64)
    grouped = {int(r["name"].rsplit("-", 1)[-1]): y for r, y in zip(rows, labels)}
    comparison = ROOT / "outputs/cq500/table2_image_baselines/mean_pooling_oof_predictions.csv"
    compare_oof_truth(grouped, comparison, ["true_IPH", "true_Mass Effect", "true_Midline Shift"])
    if len(grouped) != 491:
        raise ValueError("CQ500 检查数量改变")
    cohorts.append(Cohort("cq500", "CQ500", ["IPH", "Mass effect", "Midline shift"],
                          ["IPH", "Mass effect", "Midline shift"], labels, [source, comparison],
                          {"unit": "CT 检查", "label_aggregation": "三位阅片者至少两位判阳性", "oof_true_label_check": "逐例一致"}))

    source = ROOT / "datasets/ct_rate_680/samples_680.csv"
    rows = read_csv(source)
    assert_unique([r["patient_id"] for r in rows], "CT-RATE")
    names = ["Emphysema", "Atelectasis", "Pulmonary fibrotic sequela"]
    labels = binary_labels(rows, names)
    if len(rows) != 680 or labels.sum(axis=0).tolist() != [148, 169, 188] or (labels.sum(1) >= 2).sum() != 116:
        raise ValueError("CT-RATE 任务统计与既有协议不符")
    cohorts.append(Cohort("ct_rate", "CT-RATE", names, ["Emphysema", "Atelectasis", "Fibrotic\nsequela"],
                          labels, [source], {"unit": "患者的一次检查", "scope": "既有 680 例人工标签任务子集"}))

    source = ROOT / "outputs/amos_mm/leavs_label_audit/完整三标签候选.csv"
    rows = read_csv(source)
    assert_unique([r["scan_id"] for r in rows], "AMOS-MM")
    labels = binary_labels(rows, ["liver", "kidney", "gallbladder"])
    if len(rows) != 1211 or (labels.sum(1) >= 2).sum() != 535:
        raise ValueError("AMOS-MM 标签完整候选规模改变")
    notes = {"unit": "CT 检查编号", "scope": "完整标签候选子集，不是全部下载检查",
             "downloaded_exam_count": 1687, "excluded_incomplete_labels": 476,
             "label_sources": dict(Counter(r["label_source"] for r in rows)),
             "unknown_as_negative": False, "patient_uniqueness_verified": False}
    cohorts.append(Cohort("amos_mm", "AMOS-MM (subset)", ["Liver abnormality", "Kidney abnormality", "Gallbladder abnormality"],
                          ["Liver\nabnormality", "Kidney\nabnormality", "Gallbladder\nabnormality"], labels, [source], notes))
    return cohorts


def summarize(cohort):
    y = cohort.y
    if y.shape[1] != 3 or y.shape[0] == 0 or not np.isin(y, [0, 1]).all():
        raise ValueError("空数据或非法标签")
    matrix = y.T @ y
    # 逐格独立计数复核，防止整数溢出、转置错误和对角线口径错误。
    brute = np.array([[np.count_nonzero((y[:, i] == 1) & (y[:, j] == 1)) for j in range(3)] for i in range(3)])
    assert np.array_equal(matrix, brute) and np.array_equal(matrix, matrix.T)
    assert np.array_equal(np.diag(matrix), y.sum(axis=0))
    hist = np.bincount(y.sum(axis=1), minlength=4)
    assert int(np.triu(matrix, k=1).sum()) == int(hist[2] + 3 * hist[3])
    return {"dataset": cohort.key, "display_name": cohort.name, "labels": cohort.labels,
            "examinations": len(y), "positive_counts": y.sum(axis=0).tolist(),
            "matrix_counts": matrix.tolist(), "matrix_percent": (100 * matrix / len(y)).tolist(),
            "zero_one_two_three_positive_counts": hist.tolist(), "co_positive_cases": int(hist[2] + hist[3]),
            "scope": cohort.notes, "source_sha256": {
                str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest() for p in cohort.sources}}


def write_source_data(payloads, output):
    (output / "summary.json").write_text(json.dumps(payloads, ensure_ascii=False, indent=2), encoding="utf-8")
    with (output / "cooccurrence_source_data.csv").open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(["dataset", "examinations", "row_label", "column_label", "cell_type", "count", "percent"])
        for p in payloads:
            for i, row in enumerate(p["matrix_counts"]):
                for j, count in enumerate(row):
                    writer.writerow([p["dataset"], p["examinations"], p["labels"][i], p["labels"][j],
                                     "marginal_positive" if i == j else "pairwise_co_positive", count,
                                     format(p["matrix_percent"][i][j], ".8f")])
    text = ["# 八个数据集共现矩阵\n", "格内数字为检查例数，括号内为占该面板全部检查的比例；颜色使用共同的 0–100% 色标。",
            "对角线为标签边际阳性（含共阳性），其余格子为两标签共阳性（含三阳性）。每例检查仅计一次，无抽样、无重平衡。\n",
            "| 数据集 | 检查数 | 零阳性 | 单阳性 | 恰好两阳性 | 三阳性 | 至少两阳性 |",
            "|---|---:|---:|---:|---:|---:|---:|"]
    for p in payloads:
        text.append("| " + " | ".join(map(str, [p["display_name"], p["examinations"], *p["zero_one_two_three_positive_counts"], p["co_positive_cases"]])) + " |")
    text.extend(["\nAMOS-MM 面板为刚提出的 1,211 例完整标签候选（1,011 自动开发 + 200 人工测试），下载总量为 1,687 例；476 例标签不完整的开发检查未纳入此面板。",
                 "胃镜任务表的 2,872 条中有 11 条不属于既有四场景配置（静脉曲张手术 2 条、超声胃镜下手术 9 条），其余 2,861 条全部统计；同一患者的不同检查仍是不同检查级样本。",
                 "CT-ICH 按全部切片标签取并集；CQ500 按三位阅片者多数票。两者逐例匹配既有 OOF 文件中的真实标签，不使用任何预测值。",
                 "SMT：食管黏膜下肿物；IPH：脑实质内出血；EDH：硬膜外出血。Fibrotic sequela：肺纤维化后遗改变。AMOS-MM 为器官异常标签，不是三个特定疾病诊断。",
                 "本图仅反映所选任务的标签分布；不同任务的疾病定义、标签来源和筛选规则分别记录于 summary.json。",
                 "没有更改论文、原始数据、旧图或启动训练。"])
    (output / "统计说明.md").write_text("\n\n".join(text) + "\n", encoding="utf-8")


def draw(cohorts, payloads, output, audit_dir):
    sys.path.insert(0, str(audit_dir))
    from audit_panel_alignment import require_matplotlib_panel_alignment

    plt.rcParams.update({
        "font.family": "sans-serif", "font.sans-serif": ["DejaVu Sans", "Arial", "Helvetica"],
        "font.size": 9, "axes.labelsize": 9, "xtick.labelsize": 8.5, "ytick.labelsize": 8.5,
        "pdf.fonttype": 42, "svg.fonttype": "none", "savefig.facecolor": "white",
        "axes.spines.top": False, "axes.spines.right": False,
    })
    # 屏幕审阅用宽幅定量网格；不直接按窄栏缩放为期刊最终版面。
    fig, axes = plt.subplots(2, 4, figsize=(14.4, 7.2), dpi=300)
    fig.subplots_adjust(left=0.076, right=0.912, bottom=0.085, top=0.922, wspace=0.43, hspace=0.40)
    cmap = LinearSegmentedColormap.from_list("cooccurrence_blue", ["#f5f9fd", "#c3dcef", "#76afd2", "#347dab", "#14466c"])
    norm = Normalize(vmin=0, vmax=100)
    contrast_checks = []

    def relative_luminance(rgb):
        rgb = np.asarray(rgb)
        linear = np.where(rgb <= 0.04045, rgb / 12.92, ((rgb + 0.055) / 1.055) ** 2.4)
        return float(linear @ [0.2126, 0.7152, 0.0722])

    for ax, cohort, values in zip(axes.flat, cohorts, payloads):
        percent = np.array(values["matrix_percent"])
        counts = np.array(values["matrix_counts"])
        artist = ax.imshow(percent, cmap=cmap, norm=norm, interpolation="nearest", aspect="equal")
        ax.set_xticks(range(3), cohort.ticks)
        ax.set_yticks(range(3), cohort.ticks)
        ax.tick_params(axis="both", which="both", length=0, pad=6)
        for spine in ax.spines.values():
            spine.set_visible(False)
        ax.set_xticks(np.arange(-0.5, 3, 1), minor=True)
        ax.set_yticks(np.arange(-0.5, 3, 1), minor=True)
        ax.grid(which="minor", color="white", linewidth=1.3)
        # 名称用于识别数据集；不放总标题、面板字母、n=角标或底部注释。
        ax.text(0.5, 1.058, cohort.name, transform=ax.transAxes, ha="center", va="bottom",
                fontsize=11, fontweight="bold", color="#243746")
        for i in range(3):
            for j in range(3):
                rgb = cmap(norm(percent[i, j]))[:3]
                luminance = relative_luminance(rgb)
                dark = relative_luminance([21 / 255, 54 / 255, 77 / 255])
                choices = [("#15364d", (luminance + 0.05) / (dark + 0.05)),
                           ("white", 1.05 / (luminance + 0.05)),
                           ("black", (luminance + 0.05) / 0.05)]
                color, contrast = next((color, ratio) for color, ratio in choices if ratio >= 4.5)
                contrast_checks.append(contrast)
                ax.text(j, i - 0.10, str(int(counts[i, j])), ha="center", va="center",
                        fontsize=11, fontweight="bold", color=color)
                ax.text(j, i + 0.17, f"({percent[i, j]:.1f}%)", ha="center", va="center",
                        fontsize=8.0, color=color)
    color_ax = fig.add_axes([0.935, 0.258, 0.009, 0.48])
    colorbar = fig.colorbar(artist, cax=color_ax, ticks=[0, 20, 40, 60, 80, 100])
    colorbar.ax.set_yticklabels(["0%", "20%", "40%", "60%", "80%", "100%"])
    colorbar.ax.tick_params(length=2.5, width=0.6, labelsize=8.0)
    colorbar.outline.set_visible(False)
    colorbar.set_label("Examinations (%)", fontsize=9, labelpad=7)
    qa = output / "qa"
    qa.mkdir(exist_ok=True)
    (qa / "annotation_contrast.json").write_text(json.dumps({
        "minimum_ratio": min(contrast_checks), "required_minimum": 4.5,
        "checked_cells": len(contrast_checks), "passed": min(contrast_checks) >= 4.5,
    }, indent=2), encoding="utf-8")
    fig.canvas.draw()
    require_matplotlib_panel_alignment(
        fig, json_out=qa / "panel_alignment.json", axes=list(axes.flat),
        panel_ids=[p["dataset"] for p in payloads], tolerance_pt=1.5,
        gutter_tolerance_pt=1.5, require_panel_labels=False, strict=True,
    )
    # PDF 仅用于自动碰撞检查；对用户交付和后续论文使用均为 PNG。
    fig.savefig(qa / "all_datasets_label_cooccurrence.svg")
    fig.savefig(qa / "all_datasets_label_cooccurrence.pdf", dpi=300)
    fig.savefig(output / "all_datasets_label_cooccurrence.png", dpi=300)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=OUT)
    parser.add_argument("--audit-dir", type=Path, required=True, help="图形对齐校验工具所在目录")
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    cohorts = load_cohorts()
    payloads = [summarize(c) for c in tqdm(cohorts, desc="逐格校验共现计数", unit="数据集")]
    write_source_data(payloads, args.output_dir)
    draw(cohorts, payloads, args.output_dir, args.audit_dir)
    for p in payloads:
        print(f"{p['display_name']}: {p['examinations']} 例；共阳性 {p['co_positive_cases']} 例；矩阵 {p['matrix_counts']}")
    print("PNG 已生成；请完成 PDF 文字尺寸与碰撞检查后交付。")


if __name__ == "__main__":
    main()
