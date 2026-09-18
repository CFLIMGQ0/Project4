#!/usr/bin/env python3
"""合并分组生成结果，核对逐例覆盖并导出单段影像描述。"""

import argparse
import csv
import json
from pathlib import Path

from generate_cq500_image_descriptions import ROOT, quality_flags, save_json


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dirs", type=Path, nargs="+", required=True)
    parser.add_argument("--output-dir", type=Path, default=ROOT / "outputs/cq500/image_descriptions")
    parser.add_argument("--expected-cases", type=int, default=491)
    args = parser.parse_args()
    reports = {}
    for directory in args.input_dirs:
        for path in sorted(directory.glob("case_*.json")):
            item = json.loads(path.read_text())
            patient_id = item["scan_id"]
            if patient_id in reports:
                raise ValueError(f"病例{patient_id}有多份描述，请核对生成版本")
            assert item["status"] == "generated" and item["text"], path
            assert item["ground_truth_read"] is False, path
            assert item["synthetic"] is True and item["review_status"] == "not_reviewed", path
            item["quality_flags"] = quality_flags(item["text"])
            reports[patient_id] = item
    missing = sorted(set(range(args.expected_cases)) - reports.keys())
    extra = sorted(reports.keys() - set(range(args.expected_cases)))
    if missing or extra:
        raise ValueError(f"覆盖不完整：已完成{len(reports)}例，缺少{missing}，多出{extra}")
    unsuitable = {i: r["quality_flags"] for i, r in reports.items()
                  if "diagnostic_or_label_wording" in r["quality_flags"]
                  or "unsupported_numeric_measurement" in r["quality_flags"]}
    if unsuitable:
        raise ValueError(f"以下病例仍有诊断/分类措辞或无依据数值，需先核对再导出：{unsuitable}")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    csv_path = args.output_dir / f"descriptions_{args.expected_cases}.csv"
    with csv_path.open("w", encoding="utf-8-sig", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(["patient_id", "image_description"])
        writer.writerows((i, reports[i]["text"]) for i in sorted(reports))
    md_path = args.output_dir / f"descriptions_{args.expected_cases}.md"
    md_path.write_text(
        "# 逐例影像所见\n\n图像派生的 AI 描述草稿；阅片复核状态另见审计文件。\n\n"
        + "\n\n".join(f"## 病例 {i:03d}\n\n{reports[i]['text']}" for i in sorted(reports)) + "\n",
        encoding="utf-8",
    )
    save_json(args.output_dir / "description_audit.json", {
        "expected_cases": args.expected_cases, "completed_cases": len(reports),
        "ground_truth_read": False, "prediction_files_read": False,
        "synthetic": True, "review_status": "not_reviewed",
        "observed_frames": sum(r["observed_frames"] for r in reports.values()),
        "cases_with_source_decode_failures": [i for i, r in reports.items() if r["failed_files"]],
        "cases_with_fewer_than_ten_frames": [i for i, r in reports.items() if r["observed_frames"] < 10],
        "format_flagged_cases": {i: r["quality_flags"] for i, r in reports.items() if r["quality_flags"]},
        "reports": [reports[i] for i in sorted(reports)],
    })
    print(f"汇总完成：{len(reports)}例；文件：{csv_path}", flush=True)


if __name__ == "__main__":
    main()
