#!/usr/bin/env python3
"""用可追溯的五折汇总填入 temp.tex；未完成和待审计的单元格保留 --。"""

from __future__ import annotations

import csv
from pathlib import Path

from build_paper_experiment_inventory import (ROOT, DATASETS, PUBLIC, IMAGE, TEXT, MULTI,
                                               POSITION, TRANSITION, ROUTES)
from aggregate_paper_results import DATASET_NAMES, DISPLAY, POSITION_NAMES

TABLES = ROOT / "outputs/paper_tables"
TEX = ROOT / "temp.tex"
MODEL_TEX = {
    "topk_mil": r"Top-$k$ MIL",
    "task2_mmfnet_2024": r"MMFNet (2024)$^{\dagger}$",
    "task2_radfuse_2025": r"RadFuse (2025)$^{\dagger}$",
    "task2_saif_2025": r"SAIF (2025)$^{\dagger}$",
    "task2_mmtf_2025": r"MMTF (2026)$^{\dagger}$",
    "task2_camchex_adapted": r"CaMCheX (2025)$^{\dagger}$",
    "task2_med3dvlm_adapted": r"Med3DVLM (2025)$^{\dagger}$",
    "task2_m3fm_adapted": r"M3FM (2025)$^{\dagger}$",
    "amef_multimodal": r"\textbf{ALM-MIL (Ours)}",
    "unified_multimodal_framework_2026": r"Unified MM (2026)$^{\dagger}$",
    "adaptive_multimodal_fusion_2026": r"Adaptive Fusion (2026)$^{\dagger\ddagger}$",
    "gcf_net_2026": r"GCF-Net (2026)$^{\dagger}$",
}


def read(name: str) -> list[dict]:
    with (TABLES / name).open(encoding="utf-8-sig", newline="") as stream:
        return list(csv.DictReader(stream))


def item(value: dict | None, *, mean: str, std: str, bold: bool = False,
         allowed: tuple[str, ...] = ("COMPLETE",)) -> str:
    if not value or value.get("Status") not in allowed or not value.get(mean):
        return "--"
    shown = f'{float(value[mean]):.4f} $\\pm$ {float(value[std]):.4f}'
    return rf"\textbf{{{shown}}}" if bold else shown


def replace_body(source: str, label: str, body: str) -> str:
    pivot = source.index(r"\label{" + label + "}")
    begin = source.index(r"\begin{tabular}", pivot)
    first_rule = source.index(r"\midrule", begin) + len(r"\midrule")
    last_rule = source.index(r"\bottomrule", first_rule)
    return source[:first_rule] + "\n\n" + body.strip() + "\n\n" + source[last_rule:]


def table1(source: str) -> str:
    values = {(row["Method_Key"], row["Dataset"]): row for row in read("table1_summary.csv")}
    best = {DATASET_NAMES[dataset]: max(
        (float(value["Mean_MacroF1"]) for (method, name), value in values.items()
         if name == DATASET_NAMES[dataset]
         and value["Status"] in ("COMPLETE", "SUMMARY_ONLY") and value["Mean_MacroF1"]),
        default=float("-inf")) for dataset in DATASETS}
    rows = []
    for group in (IMAGE, TEXT, MULTI[:7], MULTI[8:], MULTI[7:8]):
        if rows:
            rows.append(r"\midrule")
        for method in group:
            label = MODEL_TEX.get(method, DISPLAY[method])
            image = method not in TEXT
            report = method in TEXT or (method in MULTI and method != "gcf_net_2026")
            cells = [label, r"$\checkmark$" if image else r"$\times$",
                     r"$\checkmark$" if report else r"$\times$"]
            for dataset in DATASETS:
                value = values.get((method, DATASET_NAMES[dataset]))
                if (value and value["Status"] in ("COMPLETE", "SUMMARY_ONLY")
                        and value["Mean_MacroF1"]):
                    shown = f'{100*float(value["Mean_MacroF1"]):.1f}$\\pm${100*float(value["Std_MacroF1"]):.1f}'
                    if abs(float(value["Mean_MacroF1"])-best[DATASET_NAMES[dataset]]) < 1e-8:
                        shown = rf"\textbf{{{shown}}}"
                    cells.append(shown)
                else:
                    cells.append("--")
            rows.append(" & ".join(cells) + r" \\")
    return replace_body(source, "tab:overall_comparison", "\n\n".join(rows))


def table2(source: str) -> str:
    values = {(row["Method"], row["Dataset"], int(row["DeletionRatio"])): row
              for row in read("table2_summary.csv")}
    best = {(DATASET_NAMES[dataset], ratio): max(
        (float(value["Mean"]) for (method, name, deletion), value in values.items()
         if name == DATASET_NAMES[dataset] and deletion == ratio
         and value["Status"] == "COMPLETE" and value["Mean"]), default=float("-inf"))
        for dataset in PUBLIC for ratio in (0,25,50,75)}
    rows = []
    for variant in POSITION:
        if variant == "acpe":
            rows.append(r"\midrule")
        label = POSITION_NAMES[variant] if variant != "acpe" else r"\textbf{ACPE (Ours)}"
        for ratio_index, ratio in enumerate((0, 25, 50, 75)):
            cells = [rf"\multirow{{4}}{{*}}{{{label}}}" if ratio_index == 0 else "", rf"{ratio}\%"]
            for dataset in PUBLIC:
                value = values.get((POSITION_NAMES[variant], DATASET_NAMES[dataset], ratio))
                cells.append(item(value, mean="Mean", std="Std",
                    bold=bool(value and value["Mean"] and
                              abs(float(value["Mean"])-best[DATASET_NAMES[dataset], ratio]) < 1e-8))
                    if dataset != "merlin_1k" else "--")
            rows.append(" & ".join(cells) + r" \\")
        if variant != "acpe":
            rows.append(r"\addlinespace[1pt]")
    return replace_body(source, "tab:position_mechanism_comparison", "\n\n".join(rows))


def table3(source: str) -> str:
    values = {(row["Configuration"], row["Dataset"]): row for row in read("table3_summary.csv")}
    best = {DATASET_NAMES[dataset]: max(
        (float(value["Mean_MacroF1"]) for (variant, name), value in values.items()
         if name == DATASET_NAMES[dataset] and value["Status"] == "COMPLETE"
         and value["Mean_MacroF1"]), default=float("-inf")) for dataset in PUBLIC}
    marks = ((1,0,0,0,0), (1,0,0,0,1), (1,1,0,0,1), (1,0,1,0,1),
             (1,1,0,1,1), (1,0,1,1,1), (1,1,1,1,1))
    rows = []
    for variant, flags in zip(TRANSITION, marks):
        if variant == "123456+1":
            rows.append(r"\midrule")
        cells = [r"$\checkmark$" if flag else r"$\times$" for flag in flags]
        for dataset in PUBLIC:
            value = values.get((variant, DATASET_NAMES[dataset]))
            cells.append(item(value, mean="Mean_MacroF1", std="Std_MacroF1",
                              bold=bool(value and value["Mean_MacroF1"] and
                                        abs(float(value["Mean_MacroF1"])-best[DATASET_NAMES[dataset]]) < 1e-8))
                         if dataset != "merlin_1k" else "--")
        rows.append(" & ".join(cells) + r" \\")
    return replace_body(source, "tab:transition_mlp_ablation", "\n\n".join(rows))


def table4(source: str) -> str:
    values = {(row["Configuration"], row["Dataset"]): row for row in read("table4_summary.csv")}
    best = {DATASET_NAMES[dataset]: max(
        (float(value["Mean_MacroF1"]) for (variant, name), value in values.items()
         if name == DATASET_NAMES[dataset] and value["Status"] == "COMPLETE"
         and value["Mean_MacroF1"]), default=float("-inf")) for dataset in PUBLIC}
    rows = []
    for variant, flags in zip(ROUTES, ((1,0), (0,1), (1,1))):
        if variant == "acpe":
            rows.append(r"\midrule")
        cells = [r"$\checkmark$" if flag else r"$\times$" for flag in flags]
        for dataset in PUBLIC:
            value = values.get((variant, DATASET_NAMES[dataset]))
            cells.append(item(value, mean="Mean_MacroF1", std="Std_MacroF1",
                              bold=bool(value and value["Mean_MacroF1"] and
                                        abs(float(value["Mean_MacroF1"])-best[DATASET_NAMES[dataset]]) < 1e-8))
                         if dataset != "merlin_1k" else "--")
        rows.append(" & ".join(cells) + r" \\")
    return replace_body(source, "tab:position_route_ablation", "\n\n".join(rows))


def main() -> None:
    source = TEX.read_text(encoding="utf-8")
    source = table1(source)
    source = table2(source)
    source = table3(source)
    source = table4(source)
    source = source.replace("Values are Macro-F1, reported as", "Values are Macro-F1 (\\%), reported as")
    source = source.replace(r"\fontsize{5.2}{5.7}\selectfont", r"\fontsize{6.2}{6.8}\selectfont")
    source = source.replace("Merlin-1K results are left\nblank until the corresponding five-fold experiments are completed.",
        "Merlin report-input results are withheld after a label-leakage audit.\nNew private baseline cells use frozen ConvNeXt features from the existing RGB cache.\n$^{\\ddagger}$ marks a preprint.")
    source = source.replace("The new baselines lack private image data on this host.",
                            "New private baseline cells use frozen ConvNeXt features from the existing RGB cache.")
    source = source.replace("New private baseline cells use the existing RGB image cache when available.",
                            "New private baseline cells use frozen ConvNeXt features from the existing RGB cache.")
    if "Unified MM and Adaptive Fusion abbreviate" not in source:
        source = source.replace("$^{\\ddagger}$ marks a preprint.",
            "$^{\\ddagger}$ marks a preprint. Unified MM and Adaptive Fusion abbreviate\n"
            "Unified Multimodal Framework and Adaptive Multimodal Fusion.")
    TEX.write_text(source, encoding="utf-8")
    print(f"已更新 {TEX}；未完成/待审计项保留 --", flush=True)


if __name__ == "__main__":
    main()
