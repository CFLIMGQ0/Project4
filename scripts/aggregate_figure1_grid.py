#!/usr/bin/env python3
"""汇总 Figure 1 五折网格；缺少任何折或条件时不生成虚构的完整图。"""

from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np

from paper_block_deletion import BLOCKS, RATIOS

ROOT = Path(__file__).resolve().parents[2]
SOURCE = ROOT / "outputs/paper_results/figure1_grid"
OUT = ROOT / "outputs/paper_results/figure1_summary"
DATASETS = ("ct_rate", "amos_mm", "mr_rate_1k", "merlin_1k")


def write_csv(path: Path, rows: list[dict], columns: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    fold_rows: list[dict] = []
    summary_rows: list[dict] = []
    for dataset in DATASETS:
        for ratio in RATIOS:
            for blocks in ((1,) if ratio == 0 else BLOCKS):
                key = f"{ratio:02d}_B{blocks}"
                found = []
                for fold in range(1, 6):
                    path = SOURCE / dataset / f"fold_{fold}.json"
                    if not path.exists():
                        continue
                    payload = json.loads(path.read_text())
                    item = payload["results"].get(key)
                    if not item or item.get("status"):
                        continue
                    cases = payload.get("common_case_indices")
                    if not cases or item["case_indices"] != cases:
                        raise ValueError(f"Figure 1 病例不一致：{path}, {key}")
                    row = {"Dataset": dataset, "Fold": fold, "DeletionRatio": ratio,
                           "Blocks": blocks, "NumCases": item["num_cases"],
                           "Seed": item["seed"],
                           "ACPE_MacroF1": item["acpe_f1"],
                           "Original_MacroF1": item["original_f1"],
                           "Delta_MacroF1": item["delta_f1"],
                           "ACPE_Acc005": item["acpe_acc005"],
                           "Original_Acc005": item["original_acc005"],
                           "Delta_Acc005": item["delta_acc005"],
                           "Source": str(path.relative_to(ROOT))}
                    fold_rows.append(row)
                    found.append(row)
                out = {"Dataset": dataset, "DeletionRatio": ratio, "Blocks": blocks,
                       "NumFolds": len(found),
                       "Status": "BLOCKED_MASK_LEAKAGE_AUDIT" if dataset == "merlin_1k" else
                                 "COMPLETE" if len(found) == 5 else "PARTIAL" if found else "MISSING",
                       "CasesPerFold": "/".join(str(row["NumCases"]) for row in found)}
                for metric in ("Delta_MacroF1", "Delta_Acc005"):
                    values = np.asarray([row[metric] for row in found], dtype=np.float64)
                    out[f"{metric}_Mean"] = float(values.mean()) if len(values) else ""
                    out[f"{metric}_Std"] = float(values.std(ddof=1)) if len(values) > 1 else ""
                summary_rows.append(out)
    fold_cols = ["Dataset", "Fold", "DeletionRatio", "Blocks", "Seed", "NumCases",
                 "ACPE_MacroF1", "Original_MacroF1", "Delta_MacroF1",
                 "ACPE_Acc005", "Original_Acc005", "Delta_Acc005", "Source"]
    summary_cols = ["Dataset", "DeletionRatio", "Blocks", "NumFolds", "Status", "CasesPerFold",
                    "Delta_MacroF1_Mean", "Delta_MacroF1_Std",
                    "Delta_Acc005_Mean", "Delta_Acc005_Std"]
    write_csv(OUT / "fold_results.csv", fold_rows, fold_cols)
    write_csv(OUT / "summary.csv", summary_rows, summary_cols)
    export_columns = ("dataset", "fold", "ratio", "B", "seed", "num_cases",
                      "original_f1", "acpe_f1", "delta_f1", "original_acc005",
                      "acpe_acc005", "delta_acc005", "source")
    for dataset in DATASETS:
        selected = [row for row in fold_rows if row["Dataset"] == dataset]
        export_rows = [dict(zip(export_columns, (row["Dataset"], row["Fold"],
            row["DeletionRatio"], row["Blocks"], row["Seed"], row["NumCases"],
            row["Original_MacroF1"], row["ACPE_MacroF1"], row["Delta_MacroF1"],
            row["Original_Acc005"], row["ACPE_Acc005"], row["Delta_Acc005"],
            row["Source"]))) for row in selected]
        write_csv(OUT / f"figure1_{dataset}.csv", export_rows, list(export_columns))
        write_csv(OUT / f"figure1_{dataset}_summary.csv",
                  [row for row in summary_rows if row["Dataset"] == dataset], summary_cols)
    statuses = {dataset: sum(row["Status"] == "COMPLETE" for row in summary_rows
                             if row["Dataset"] == dataset) for dataset in DATASETS}
    (OUT / "status.json").write_text(json.dumps({"complete_grid_cells_per_dataset": statuses,
                                                  "expected_cells_per_dataset": 1 + (len(RATIOS)-1)*len(BLOCKS),
                                                  "zero_ratio_note": "0% 仅测一次，绘图跨 B 复制同一数值",
                                                  "fold_rows": len(fold_rows)},
                                                 ensure_ascii=False, indent=2)+"\n", encoding="utf-8")
    print(json.dumps(statuses, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
