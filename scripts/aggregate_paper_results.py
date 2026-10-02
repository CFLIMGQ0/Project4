#!/usr/bin/env python3
"""从已有逐折结果生成论文表格；缺失折次留空，绝不推断或补值。"""

from __future__ import annotations

import csv
from functools import lru_cache
import json
import re
import statistics
from pathlib import Path

from build_paper_experiment_inventory import (ROOT, DATASETS, PUBLIC, IMAGE, TEXT, MULTI,
                                               POSITION, TRANSITION, ROUTES, model_folder,
                                               variant_folder)

OUT = ROOT / "outputs/paper_tables"
DISPLAY = {
    **dict(zip(IMAGE, ("Attention MIL", "Mean pooling", "Transformer-context MIL", "Top-k MIL",
                        "Max pooling", "TransMIL", "DSMIL", "DTFD-MIL", "CLAM-MB", "CLAM-SB"))),
    **dict(zip(TEXT, ("Hashed mean", "Vocabulary attention", "TextCNN", "BiGRU", "Transformer"))),
    **dict(zip(MULTI, ("MMFNet", "RadFuse", "SAIF", "MMTF", "CaMCheX", "Med3DVLM", "M3FM",
                         "ALM-MIL", "Unified Multimodal Framework", "Adaptive Multimodal Fusion",
                         "GCF-Net"))),
}
DATASET_NAMES = dict(zip(DATASETS, ("WLE", "Chromoscopic", "Surgical", "EUS", "CT-RATE",
                                     "AMOS-MM", "MR-RATE-1K", "Merlin-1K")))
POSITION_NAMES = dict(zip(POSITION, ("Original PE", "ComRoPE-AP", "VideoRoPE", "PaTH",
                                       "DAPE V2-KERPLE", "ACPE")))


def write_csv(name: str, rows: list[dict]) -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    if not rows:
        return
    with (OUT / name).open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def fold_metric(folder: Path | None):
    if folder is None:
        return None
    for name in ("test_metrics.json", "result.json"):
        path = folder / name
        if path.exists():
            value = json.loads(path.read_text(encoding="utf-8"))
            if isinstance(value.get("macro_f1"), (float, int)):
                return float(value["macro_f1"]), path.relative_to(ROOT).as_posix()
    return None


@lru_cache(maxsize=16)
def read_payload(path: Path) -> dict:
    return json.loads(path.read_text())


def summarize(values: list[float]) -> tuple[str, str]:
    if len(values) != 5:
        return "", ""
    return repr(statistics.mean(values)), repr(statistics.stdev(values))


def private_table_values() -> dict[tuple[str, str], tuple[str, str]]:
    """旧稿仅保存四个私有数据集汇总；保留来源与原精度限制。"""
    archive_csv = ROOT / "outputs/paper_results/private_historical_summary.csv"
    if archive_csv.exists():
        with archive_csv.open(encoding="utf-8-sig", newline="") as stream:
            archived = list(csv.DictReader(stream))
        inverse = {DATASET_NAMES[dataset]: dataset for dataset in DATASETS[:4]}
        result = {(inverse[row["Dataset"]], row["Method_Key"]):
                  (row["Mean_MacroF1"], row["Std_MacroF1"]) for row in archived}
        if len(result) != len(IMAGE + TEXT + MULTI[:8]) * 4:
            raise ValueError(f"私有历史汇总缺单元格：{archive_csv}")
        return result
    source = (ROOT / "temp.tex").read_text(encoding="utf-8")
    chunk = source.split("\\label{tab:overall_comparison}", 1)[1].split("\\bottomrule", 1)[0]
    rows = {}
    names = list(IMAGE + TEXT + MULTI[:8])
    for key in names:
        display = DISPLAY[key]
        pattern = (r"Top-\$k\$ MIL" if key == "topk_mil" else
                   re.escape(display).replace(r"\ ", r"\s*"))
        match = re.search(r"^\s*(?:\\textbf\{)?" + pattern +
                          r"(?:\s*\([^)]*\))?(?:\$\^\{\\dagger\}\$)?\}?\s*(?:\n|&)(.*?)(?=\\\\)",
                          chunk, re.DOTALL | re.MULTILINE | re.IGNORECASE)
        if not match:
            continue
        numbers = re.findall(r"(\d+\.\d+)\s*\$\\pm\$\s*(\d+\.\d+)", match.group(1))
        if len(numbers) >= 4:
            for dataset, pair in zip(DATASETS[:4], numbers[:4]):
                rows[dataset, key] = pair
    if not all((dataset, "amef_multimodal") in rows for dataset in DATASETS[:4]):
        # 旧 temp.tex 的 ALM-MIL 私有四列另存于根目录 table1.md；
        # 仅恢复其原有四位小数，不将其冒充逐折原始记录。
        archive = ROOT / "table1.md"
        if archive.exists():
            match = re.search(r"^\|\s*\*\*ALM-MIL \(Ours\)\*\*[^\n]*", archive.read_text(), re.MULTILINE)
            if match:
                numbers = re.findall(r"(\d+\.\d+)\s*±\s*(\d+\.\d+)", match.group())
                if len(numbers) >= 4:
                    for dataset, pair in zip(DATASETS[:4], numbers[:4]):
                        rows[dataset, "amef_multimodal"] = pair
    return rows


def table1() -> None:
    private = private_table_values()
    fold_rows, summary = [], []
    for method in IMAGE + TEXT + MULTI:
        img = "no" if method in TEXT else "yes"
        txt = "yes" if method in TEXT or (method in MULTI and method != "gcf_net_2026") else "no"
        for dataset in DATASETS:
            values = []
            for fold in range(1, 6):
                found = fold_metric(model_folder(dataset, method, fold))
                if found:
                    metric, source = found
                    values.append(metric)
                    fold_rows.append({"Method": DISPLAY[method], "Method_Key": method,
                                      "Dataset": DATASET_NAMES[dataset], "Fold": fold,
                                      "MacroF1": repr(metric), "Source": source})
            mean, std = summarize(values)
            status = "COMPLETE" if len(values) == 5 else "PARTIAL" if values else "MISSING"
            source = "fold-level JSON" if values else ""
            if not values and (dataset, method) in private:
                mean, std = private[dataset, method]
                status, source = ("SUMMARY_ONLY",
                                  "outputs/paper_results/private_historical_summary.csv "
                                  "(archived from earlier temp.tex; 4 decimal places)")
            if dataset in DATASETS[:4] and method in MULTI[-3:] and len(values) != 5:
                status = "PENDING_PRIVATE_CACHE_FEATURES"
            if dataset == "merlin_1k" and method in (*TEXT, *MULTI[:-1]):
                status = "BLOCKED_MASK_LEAKAGE_AUDIT"
            summary.append({"Method": DISPLAY[method], "Method_Key": method,
                            "Img": img, "Txt": txt, "Dataset": DATASET_NAMES[dataset],
                            "Mean_MacroF1": mean, "Std_MacroF1": std,
                            "Num_Folds": len(values) if values else (5 if status == "SUMMARY_ONLY" else 0),
                            "Status": status, "Source": source})
    write_csv("table1_fold_results.csv", fold_rows)
    write_csv("table1_summary.csv", summary)
    headers = [DATASET_NAMES[d] for d in DATASETS]
    lines = ["# Table 1：总体性能比较", "", "逐折 JSON 优先；已有私有基线仅能追溯到 temp.tex 四位小数汇总，暂记 SUMMARY_ONLY。新增三基线利用完整的私有 336px 图像缓存运行；未完成单元格留空。Merlin-1K 报告派生标签与掩码位置存在标签线索，报告输入数值待审计。", "",
             "| 方法 | Img. | Txt. | " + " | ".join(headers) + " |", "|---|:---:|:---:|" + "|".join(["---:"]*8) + "|"]
    for method in IMAGE + TEXT + MULTI:
        records = [next(r for r in summary if r["Method_Key"] == method and r["Dataset"] == DATASET_NAMES[d]) for d in DATASETS]
        cells = ["待审" if r["Status"] == "BLOCKED_MASK_LEAKAGE_AUDIT" else
                 "运行中" if r["Status"] == "PENDING_PRIVATE_CACHE_FEATURES" else
                 f'{float(r["Mean_MacroF1"]):.4f} ± {float(r["Std_MacroF1"]):.4f}' if r["Mean_MacroF1"] else "—" for r in records]
        lines.append(f'| {DISPLAY[method]} | {"✓" if records[0]["Img"] == "yes" else "×"} | {"✓" if records[0]["Txt"] == "yes" else "×"} | ' + " | ".join(cells) + " |")
    (OUT / "table1.md").write_text("\n".join(lines)+"\n", encoding="utf-8")


def table_simple(number: int, variants: tuple[str, ...], kind: str) -> None:
    folds, summary = [], []
    for variant in variants:
        for dataset in PUBLIC:
            values = []
            for fold in range(1, 6):
                found = fold_metric(variant_folder(dataset, kind, variant, fold))
                if found:
                    metric, source = found
                    values.append(metric)
                    folds.append({"Configuration": variant, "Dataset": DATASET_NAMES[dataset],
                                  "Fold": fold, "MacroF1": repr(metric), "Source": source})
            mean, std = summarize(values)
            summary.append({"Configuration": variant, "Dataset": DATASET_NAMES[dataset],
                            "Mean_MacroF1": mean, "Std_MacroF1": std,
                            "Num_Folds": len(values),
                            "Status": "BLOCKED_MASK_LEAKAGE_AUDIT" if dataset == "merlin_1k" else
                                      "COMPLETE" if len(values) == 5 else "PARTIAL" if values else "MISSING"})
    write_csv(f"table{number}_fold_results.csv", folds)
    write_csv(f"table{number}_summary.csv", summary)
    lines = [f"# Table {number}", "", "| 配置 | " + " | ".join(DATASET_NAMES[d] for d in PUBLIC) + " |",
             "|---|" + "|".join(["---:"]*4) + "|"]
    for variant in variants:
        cells = []
        for dataset in PUBLIC:
            record = next(r for r in summary if r["Configuration"] == variant and r["Dataset"] == DATASET_NAMES[dataset])
            cells.append("待审" if dataset == "merlin_1k" else
                         f'{float(record["Mean_MacroF1"]):.4f} ± {float(record["Std_MacroF1"]):.4f}' if record["Mean_MacroF1"] else "—")
        lines.append(f'| {variant} | ' + " | ".join(cells) + " |")
    (OUT / f"table{number}.md").write_text("\n".join(lines)+"\n", encoding="utf-8")


def table2() -> None:
    folds, summary = [], []
    for variant in POSITION:
        for dataset in PUBLIC:
            for ratio in (0, 25, 50, 75):
                values = []
                for fold in range(1, 6):
                    found = None
                    if dataset in ("amos_mm", "mr_rate_1k"):
                        native = "amos_mm" if dataset == "amos_mm" else "mr_rate_1k"
                        test_name = "amos_mm" if dataset == "amos_mm" else "mr_rate"
                        if variant in ("acpe", "original_pe"):
                            path = ROOT / "outputs" / native / "deletion_classification_block3_seed42" / variant / "folds" / f"fold_{fold}.json"
                        else:
                            path = ROOT / "outputs/paper_results/table2_block3" / test_name / variant / f"fold_{fold}.json"
                        if path.exists():
                            payload = read_payload(path)
                            baseline = ROOT / "outputs" / native / "deletion_classification_block3_seed42/acpe/folds" / f"fold_{fold}.json"
                            if ratio == 0 and baseline.exists() and variant != "acpe":
                                reference = read_payload(baseline)
                                if payload["test_indices"] != reference["test_indices"] or payload["manifest"] != reference["manifest"]:
                                    raise ValueError(f"Block3 病例或掩码不匹配：{path}")
                            item = next((m for m in payload["metrics"] if int(round(m["delete_fraction"]*100)) == ratio), None)
                            if item is not None:
                                found = float(item["macro_f1"]), path.relative_to(ROOT).as_posix(), int(item["test_cases"])
                    elif dataset == "ct_rate":
                        path = ROOT / "outputs/paper_results/table2_ct_block3" / f"fold_{fold}.json"
                        if path.exists():
                            payload = read_payload(path)
                            item = payload.get("results", {}).get(variant, {}).get(str(ratio))
                            if item is not None:
                                found = (float(item["macro_f1"]), path.relative_to(ROOT).as_posix(),
                                         int(item["num_cases"]))
                    elif ratio == 0:
                        metric = fold_metric(variant_folder(dataset, "position", variant, fold))
                        if metric:
                            found = (*metric, -1)
                    if found:
                        metric, source, cases = found
                        values.append(metric)
                        folds.append({"Method": POSITION_NAMES[variant], "Dataset": DATASET_NAMES[dataset],
                                      "DeletionRatio": ratio, "Fold": fold,
                                      "MacroF1": repr(metric), "NumCases": cases,
                                      "Source": source})
                mean, std = summarize(values)
                summary.append({"Method": POSITION_NAMES[variant], "Dataset": DATASET_NAMES[dataset],
                                "DeletionRatio": ratio, "Mean": mean, "Std": std,
                                "NumFolds": len(values),
                                "Status": "BLOCKED_MASK_LEAKAGE_AUDIT" if dataset == "merlin_1k" else
                                          "COMPLETE" if len(values) == 5 else
                                          "PARTIAL" if values else "MISSING"})
    write_csv("table2_fold_results.csv", folds)
    write_csv("table2_summary.csv", summary)
    lines = ["# Table 2：位置方法比较", "", "CT-RATE、AMOS-MM 与 MR-RATE-1K 使用完整序列 Block3 固定删片协议；每个数据集的 0/25/50/75% 在相同可评估病例上推理。CT-RATE 仅包含 75% 删除后仍能采样 64 张的病例。Merlin-1K 的报告派生标签与掩码线索待审计。", "",
             "| 方法 | " + " | ".join(DATASET_NAMES[d] for d in PUBLIC) + " |",
             "|---|" + "|".join(["---:"]*4) + "|"]
    for variant in POSITION:
        cells=[]
        for dataset in PUBLIC:
            records=[next(r for r in summary if r["Method"]==POSITION_NAMES[variant] and r["Dataset"]==DATASET_NAMES[dataset] and r["DeletionRatio"]==ratio) for ratio in (0,25,50,75)]
            cells.append("待审" if dataset == "merlin_1k" else
                         ' / '.join(f'{float(r["Mean"]):.4f} ± {float(r["Std"]):.4f}' if r["Mean"] else '—' for r in records))
        lines.append(f'| {POSITION_NAMES[variant]} | '+" | ".join(cells)+" |")
    (OUT/"table2.md").write_text("\n".join(lines)+"\n",encoding="utf-8")


if __name__ == "__main__":
    table1()
    table2()
    table_simple(3, TRANSITION, "transition")
    table_simple(4, ROUTES, "route")
    print(f"已生成 {OUT.relative_to(ROOT)} 的 Table 1–4 可追溯逐折汇总。")
