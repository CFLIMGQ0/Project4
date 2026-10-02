#!/usr/bin/env python3
"""盘点论文表格与鲁棒性图所需的现有折次，不修改原始实验结果。"""

from __future__ import annotations

import csv
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "outputs/paper_results"
DATASETS = ("wle", "chromoscopic", "surgical", "eus", "ct_rate", "amos_mm", "mr_rate_1k", "merlin_1k")
PUBLIC = DATASETS[4:]
IMAGE = ("attention_mil", "mean_pooling", "transformer_context_mil", "topk_mil", "max_pooling",
         "transmil", "dsmil", "dtfd_mil", "clam_mb", "clam_sb")
TEXT = ("hashed_mean_encoder", "vocab_attention_encoder", "textcnn_encoder", "bigru_encoder",
        "transformer_encoder")
MULTI = ("task2_mmfnet_2024", "task2_radfuse_2025", "task2_saif_2025", "task2_mmtf_2025",
         "task2_camchex_adapted", "task2_med3dvlm_adapted", "task2_m3fm_adapted", "amef_multimodal",
         "unified_multimodal_framework_2026", "adaptive_multimodal_fusion_2026", "gcf_net_2026")
POSITION = ("original_pe", "comrope_ap", "videorope", "path", "dape_v2_kerple", "acpe")
TRANSITION = ("1", "1+1", "123+1", "145+1", "1236+1", "1456+1", "123456+1")
ROUTES = ("apro_absolute_only", "apro_relative_only", "acpe")


def base(dataset: str) -> Path:
    return ROOT / "outputs" / {"ct_rate": "ct_rate_680", "amos_mm": "amos_mm"}.get(dataset, dataset)


def model_folder(dataset: str, method: str, fold: int) -> Path | None:
    if dataset not in PUBLIC:
        if method in {"unified_multimodal_framework_2026", "adaptive_multimodal_fusion_2026", "gcf_net_2026"}:
            private_name = {"wle": "regular_white_light", "eus": "ultrasound"}.get(dataset, dataset)
            return ROOT / "outputs/paper_results/new_baselines" / private_name / f"fold_{fold}" / method
        return None
    if method in {"unified_multimodal_framework_2026", "adaptive_multimodal_fusion_2026", "gcf_net_2026"}:
        folder = ROOT / "outputs/paper_results/new_baselines" / dataset
        if dataset == "amos_mm": folder /= "7_labels"
        return folder / f"fold_{fold}" / method
    if dataset == "merlin_1k":
        return base(dataset) / "table1_all_models_fivefold" / f"fold_{fold}" / method
    if method in ("task2_camchex_adapted", "task2_med3dvlm_adapted", "task2_m3fm_adapted"):
        name = {"ct_rate": "ct_rate_680", "amos_mm": "amos_mm_7", "mr_rate_1k": "mr_rate_1k"}[dataset]
        folder = ROOT / "outputs/task_adapted_vlms" / name
        if dataset == "amos_mm": folder /= "7_labels"
        return folder / f"fold_{fold}" / method
    if dataset == "amos_mm":
        return base(dataset) / "all_models/7_labels" / f"fold_{fold}" / method
    return base(dataset) / "all_models_fivefold" / f"fold_{fold}" / method


def variant_folder(dataset: str, kind: str, variant: str, fold: int) -> Path:
    if kind == "position" and variant == "acpe" and dataset == "ct_rate":
        return base(dataset) / "position_baselines/apro_full" / f"fold_{fold}" / "amef_multimodal"
    if variant in ("acpe", "123456+1"):
        return model_folder(dataset, "amef_multimodal", fold)  # type: ignore[return-value]
    root = base(dataset)
    if kind == "position":
        if dataset == "ct_rate": root /= "position_baselines"
        elif dataset == "amos_mm": root /= "position_replacements_7_labels"
        else: root /= "position_replacements"
    elif kind == "transition":
        if dataset == "ct_rate": root /= "acpe_transition_groups_fivefold"
        elif dataset == "amos_mm": root /= "transition_groups_7_labels"
        else: root /= "transition_groups"
    elif kind == "route":
        if dataset == "ct_rate": root /= f"amef_{variant}_fivefold"
        elif dataset == "amos_mm": root = root / "position_replacements_7_labels" / variant
        else: root = root / "position_replacements" / variant
    if kind != "route": root /= variant
    if dataset == "amos_mm": root /= "7_labels"
    return root / f"fold_{fold}" / "amef_multimodal"


def inspect(folder: Path | None, experiment: str, dataset: str, configuration: str, fold: int,
            action: str = "") -> dict[str, str]:
    if folder is None:
        return dict(Experiment=experiment, Dataset=dataset, Configuration=configuration, Fold=str(fold),
                    Checkpoint_exists="no", Inference_result_exists="summary only", Required_metrics_exist="summary only",
                    Status="PARTIAL", Existing_output_path="temp.tex", Action_needed="核验原五折来源；保留现有汇总，不重训")
    ckpt = folder / "best_model.pt"
    result = next((p for p in (folder / "test_metrics.json", folder / "result.json") if p.exists()), None)
    prediction = next((p for p in (folder / "test_predictions.csv", folder / "test_predictions.npz") if p.exists()), None)
    metric_ok = False
    if result:
        try:
            value = json.loads(result.read_text())
            metric_ok = isinstance(value.get("macro_f1"), (int, float))
        except (ValueError, OSError):
            pass
    status = "COMPLETE" if metric_ok and prediction else "PARTIAL" if ckpt.exists() or result else "MISSING"
    if not action:
        action = ("直接复用" if status == "COMPLETE" else
                  "仅用现有checkpoint补推理" if ckpt.exists() else "补训练及推理")
    return dict(Experiment=experiment, Dataset=dataset, Configuration=configuration, Fold=str(fold),
                Checkpoint_exists="yes" if ckpt.exists() else "no",
                Inference_result_exists="yes" if prediction else "no",
                Required_metrics_exist="yes" if metric_ok else "no", Status=status,
                Existing_output_path=str(folder.relative_to(ROOT)), Action_needed=action)


def inspect_table2_inference(dataset: str, variant: str, fold: int) -> dict[str, str]:
    if dataset == "ct_rate":
        path = OUT / "table2_ct_block3" / f"fold_{fold}.json"
        present = set(json.loads(path.read_text()).get("results", {}).get(variant, {})) if path.exists() else set()
        complete = {"0", "25", "50", "75"} <= present
    elif dataset in ("amos_mm", "mr_rate_1k"):
        native = base(dataset)
        if variant in ("acpe", "original_pe"):
            path = native / "deletion_classification_block3_seed42" / variant / "folds" / f"fold_{fold}.json"
        else:
            path = OUT / "table2_block3" / ("mr_rate" if dataset == "mr_rate_1k" else dataset) / variant / f"fold_{fold}.json"
        fractions = ({int(round(item["delete_fraction"] * 100))
                      for item in json.loads(path.read_text()).get("metrics", [])
                      if isinstance(item.get("macro_f1"), (int, float))} if path.exists() else set())
        complete = {0, 25, 50, 75} <= fractions
    else:
        path, complete = base(dataset), False
    blocked = dataset == "merlin_1k"
    return dict(Experiment="Table 2 inference", Dataset=dataset, Configuration=variant, Fold=str(fold),
                Checkpoint_exists="yes" if (variant_folder(dataset, "position", variant, fold) / "best_model.pt").exists() else "no",
                Inference_result_exists="yes" if path.is_file() else "no",
                Required_metrics_exist="yes" if complete else "no",
                Status="BLOCKED" if blocked else "COMPLETE" if complete else "PARTIAL",
                Existing_output_path=str(path.relative_to(ROOT)),
                Action_needed=("报告派生标签泄漏，暂停多模态位置比较" if blocked else
                               "直接复用" if complete else "按统一 Block3 协议补四档删片推理"))


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    records: list[dict[str, str]] = []
    for dataset in DATASETS:
        for method in (*IMAGE, *TEXT, *MULTI):
            for fold in range(1, 6):
                row = inspect(model_folder(dataset, method, fold), "Table 1", dataset, method, fold)
                if dataset in DATASETS[:4] and method in MULTI[-3:] and row["Status"] != "COMPLETE":
                    row["Status"] = "PARTIAL"
                    row["Action_needed"] = "原始 JPG 缺失，但完整 336px 缓存和患者级折存在；冻结特征准备后运行适配基线"
                if dataset == "merlin_1k" and method in (*TEXT, *MULTI[:-1]):
                    row["Status"] = "BLOCKED"
                    row["Action_needed"] = "标签源于同份报告，且 MASKTARGET 位置残留标签线索；参见 merlin_text_leakage_audit.md"
                records.append(row)
    for dataset in PUBLIC:
        for method in POSITION:
            for fold in range(1, 6):
                folder = variant_folder(dataset, "position", method, fold)
                row = inspect(folder, "Table 2 training", dataset, method, fold)
                if dataset == "merlin_1k":
                    row["Status"] = "BLOCKED"
                    row["Action_needed"] = "配对报告现有掩码泄漏风险，暂停多模态位置比较"
                if row["Status"] == "COMPLETE":
                    row["Action_needed"] = "复用训练checkpoint；核对0/25/50/75%统一删除推理"
                records.append(row)
                records.append(inspect_table2_inference(dataset, method, fold))
        for variant in TRANSITION:
            for fold in range(1, 6):
                folder = variant_folder(dataset, "transition", variant, fold)
                row = inspect(folder, "Table 3", dataset, variant, fold)
                if dataset == "merlin_1k":
                    row["Status"] = "BLOCKED"
                    row["Action_needed"] = "配对报告现有掩码泄漏风险，暂停多模态消融"
                records.append(row)
        for variant in ROUTES:
            for fold in range(1, 6):
                folder = variant_folder(dataset, "route", variant, fold)
                row = inspect(folder, "Table 4", dataset, variant, fold)
                if dataset == "merlin_1k":
                    row["Status"] = "BLOCKED"
                    row["Action_needed"] = "配对报告现有掩码泄漏风险，暂停多模态消融"
                records.append(row)
        for variant in ("original_pe", "acpe"):
            for fold in range(1, 6):
                folder = variant_folder(dataset, "position", variant, fold)
                row = inspect(folder, "Figure 1", dataset, variant, fold)
                grid = OUT / "figure1_grid" / dataset / f"fold_{fold}.json"
                completed_grid = (len(json.loads(grid.read_text()).get("results", {})) == 65
                                  if grid.exists() else False)
                row["Inference_result_exists"] = "yes" if grid.exists() else "no"
                row["Required_metrics_exist"] = "yes" if completed_grid else "no"
                row["Status"] = ("COMPLETE" if completed_grid else
                                 "PARTIAL" if row["Checkpoint_exists"] == "yes" else "MISSING")
                row["Existing_output_path"] = str(grid.relative_to(ROOT)) if grid.exists() else row["Existing_output_path"]
                row["Action_needed"] = ("直接复用" if completed_grid else
                                         "复用checkpoint；补B1–B8×0–80%配对删除推理与位置恢复" if
                                         row["Checkpoint_exists"] == "yes" else "缺少正常训练checkpoint，先补训练")
                if dataset == "merlin_1k":
                    row["Status"] = "BLOCKED"
                    row["Action_needed"] = "配对报告现有掩码泄漏风险，暂停多模态鲁棒性图"
                records.append(row)

    fields = list(records[0])
    for name, selected in (("experiment_inventory.csv", records),
                           ("missing_jobs.csv", [r for r in records if r["Status"] != "COMPLETE"]),
                           ("completed_jobs.csv", [r for r in records if r["Status"] == "COMPLETE"])):
        with (OUT / name).open("w", encoding="utf-8-sig", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=fields); writer.writeheader(); writer.writerows(selected)
    from collections import Counter
    counts = Counter((r["Experiment"], r["Status"]) for r in records)
    lines = ["# 论文实验盘点", "", "按已有折次文件检查 checkpoint、预测文件及 Macro-F1。完整五折在后续汇总时按原值复用；有数据完整性问题的结果记为 BLOCKED。", "",
             "| 任务 | COMPLETE | PARTIAL | MISSING | BLOCKED |", "|---|---:|---:|---:|---:|"]
    for task in ("Table 1", "Figure 1", "Table 2 training", "Table 2 inference", "Table 3", "Table 4"):
        lines.append(f"| {task} | {counts[task, 'COMPLETE']} | {counts[task, 'PARTIAL']} | {counts[task, 'MISSING']} | {counts[task, 'BLOCKED']} |")
    lines += ["", "## 已确认的配置边界", "",
              "- Merlin-1K 当前 manifest 只有 study_id，现有五折是 study-level，不能称为 patient-level；标签源于同份报告，MASKTARGET 的位置仍泄漏部分目标类别，报告输入结果暂停进入论文表格（见 outputs/paper_results/merlin_text_leakage_audit.md）。",
              "- Figure 1 的 B1–B8、0–80% 网格已在 CT-RATE、AMOS-MM 和 MR-RATE-1K 的正常训练 checkpoint 上补做固定参数推理；Merlin-1K 因报告标签泄漏暂缓。",
              "- Table 2 的训练状态与删除推理状态分开记录；三个可审定数据集的 0/25/50/75% 均采用统一删除协议。",
              "- GCF-Net 原方法使用图像和标签嵌入，未使用患者报告；表 1 的 Txt. 栏应如实标注。",
              "- 私有数据集在 temp.tex 有汇总值，但本盘点尚未逐一关联五折原始文件，因此记为 PARTIAL，不启动重训。",
              "- 四个私有数据集的原始 JPG 目录 datasets/main_data 在本机缺失，但 datasets/image_cache/shared 和原患者级 split_manifest 完整；新增三基线由缓存构建冻结特征后运行。",
              "", "逐折明细见 `experiment_inventory.csv`；待补项见 `missing_jobs.csv`。", ""]
    (OUT / "experiment_inventory.md").write_text("\n".join(lines), encoding="utf-8")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
