#!/usr/bin/env python3
"""筛选 valid_dicts_report.csv 中 operationValue=胃镜检查 的肠镜报告。"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path


def normalize(value: str | None) -> str:
    return (value or "").strip()


def is_colonoscopy_report(report_title: str) -> bool:
    title = normalize(report_title)
    return "肠镜" in title or "结肠镜" in title


def count_rows(csv_path: Path, encoding: str = "utf-8-sig") -> int:
    with csv_path.open("r", encoding=encoding, newline="") as f:
        reader = csv.DictReader(f)
        return sum(1 for _ in reader)


def print_progress(current: int, total: int, width: int = 30) -> None:
    ratio = (current / total) if total else 1.0
    filled = int(width * ratio)
    bar = "#" * filled + "-" * (width - filled)
    print(f"\r进度 [{bar}] {current}/{total} ({ratio * 100:6.2f}%)", end="", flush=True)


def find_mixed_reports(csv_path: Path, encoding: str = "utf-8-sig") -> tuple[int, list[tuple[int, dict[str, str]]]]:
    total = count_rows(csv_path, encoding=encoding)
    matched_rows: list[tuple[int, dict[str, str]]] = []

    with csv_path.open("r", encoding=encoding, newline="") as f:
        reader = csv.DictReader(f)

        for idx, row in enumerate(reader, start=1):
            if idx == 1 or idx == total or idx % 200 == 0:
                print_progress(idx, total)

            report_title = normalize(row.get("reportTitle"))
            operation_value = normalize(row.get("operationValue"))

            if is_colonoscopy_report(report_title) and operation_value == "胃镜检查":
                matched_rows.append((idx, row))

    print()
    return total, matched_rows


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="找出 operationValue 为胃镜检查的肠镜报告，并输出到终端。"
    )
    parser.add_argument(
        "--csv",
        type=Path,
        default=Path("../datasets/valid_dicts_report.csv"),
        help="CSV 文件路径（默认: ../datasets/valid_dicts_report.csv）",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    csv_path: Path = args.csv

    if not csv_path.exists():
        raise FileNotFoundError(f"未找到 CSV 文件: {csv_path}")

    total, matched_rows = find_mixed_reports(csv_path)

    print(f"CSV 文件: {csv_path}")
    print(f"总记录数: {total}")
    print(f"命中数量: {len(matched_rows)}")

    if not matched_rows:
        print("未发现 operationValue=胃镜检查 的肠镜报告。")
        return

    print("\n命中明细（row_index 为数据行序号，从 1 开始，不含表头）:")
    for i, (row_index, row) in enumerate(matched_rows, start=1):
        print(
            f"[{i}] row_index={row_index}, "
            f"admissionNo={normalize(row.get('admissionNo'))}, "
            f"namePatient={normalize(row.get('namePatient'))}, "
            f"checkTime={normalize(row.get('checkTime'))}, "
            f"reportTitle={normalize(row.get('reportTitle'))}, "
            f"operationValue={normalize(row.get('operationValue'))}, "
            f"exam_dir={normalize(row.get('exam_dir'))}"
        )


if __name__ == "__main__":
    main()