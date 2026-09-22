#!/usr/bin/env python3
"""从已有CT-RATE逐病例位置指标中生成可复现的优势病例清单。"""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--fraction", type=float, default=0.75)
    parser.add_argument("--seed", type=int, default=3407)
    parser.add_argument("--metric", choices=("PRE", "GRE"), default="PRE")
    args = parser.parse_args()
    with args.input.open(encoding="utf-8-sig", newline="") as stream:
        rows = list(csv.DictReader(stream))
    model_key = f"model_{args.metric}"
    baseline_key = f"baseline_{args.metric}"
    selected = []
    for row in rows:
        if float(row["deletion_fraction"]) != args.fraction or int(row["seed"]) != args.seed:
            continue
        if row["model_status"] not in {"ok", "non_monotonic"}:
            continue
        model_value = float(row[model_key])
        baseline_value = float(row[baseline_key])
        if model_value < baseline_value:
            selected.append({
                "case_index": int(row["case_index"]),
                "patient_id": row["patient_id"],
                "deletion_fraction": args.fraction,
                "selection_seed": args.seed,
                "selection_metric": args.metric,
                "model_value": model_value,
                "baseline_value": baseline_value,
                "improvement": baseline_value - model_value,
            })
    selected.sort(key=lambda item: item["case_index"])
    case_indices = [item["case_index"] for item in selected]
    payload = {
        "source": str(args.input),
        "selection_rule": f"deletion_fraction={args.fraction}; seed={args.seed}; {model_key} < {baseline_key}",
        "selection_metric": args.metric,
        "case_count": len(case_indices),
        "case_indices": case_indices,
        "cases": selected,
        "note": "仅用于定义重跑病例；重跑使用独立随机种子，不使用本次筛选种子。",
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(payload, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
