#!/usr/bin/env python3
"""检查 valid_dicts_report.csv 相对 valid_dicts_report_original.csv 的修改情况。

输出仅包含不一致项，按“键 -> 取值变化 -> 次数”压缩展示。
示例：
    hp键：
    空值 -> 未检 *100
"""

from __future__ import annotations

import argparse
import csv
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, Iterable, List


EMPTY_LABEL = "空值"
MISSING_ROW_LABEL = "<缺失行>"


def normalize_value(value: str | None) -> str:
    if value is None:
        return EMPTY_LABEL
    text = str(value).strip()
    return text if text else EMPTY_LABEL


def render_progress(current: int, total: int, width: int = 30) -> None:
    if total <= 0:
        return
    ratio = current / total
    filled = int(width * ratio)
    bar = "#" * filled + "-" * (width - filled)
    sys.stderr.write(f"\r处理进度 [{bar}] {current}/{total}")
    if current >= total:
        sys.stderr.write("\n")
    sys.stderr.flush()


def read_csv_rows(csv_path: Path) -> List[Dict[str, str]]:
    with csv_path.open("r", encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        return list(reader)


def compare_rows(
    original_rows: List[Dict[str, str]],
    modified_rows: List[Dict[str, str]],
    keys: Iterable[str],
) -> Dict[str, Counter]:
    transition_by_key: Dict[str, Counter] = defaultdict(Counter)

    total = len(list(keys))
    key_list = list(keys)
    total = len(key_list)

    for idx, key in enumerate(key_list, start=1):
        render_progress(idx, total)
        max_len = max(len(original_rows), len(modified_rows))
        for row_i in range(max_len):
            origin_row = original_rows[row_i] if row_i < len(original_rows) else None
            mod_row = modified_rows[row_i] if row_i < len(modified_rows) else None

            origin_raw = origin_row.get(key) if origin_row is not None else MISSING_ROW_LABEL
            mod_raw = mod_row.get(key) if mod_row is not None else MISSING_ROW_LABEL

            origin_val = normalize_value(origin_raw)
            mod_val = normalize_value(mod_raw)

            if origin_val != mod_val:
                transition_by_key[key][(origin_val, mod_val)] += 1

    return transition_by_key


def print_result(transition_by_key: Dict[str, Counter]) -> None:
    for key in sorted(transition_by_key.keys()):
        transitions = transition_by_key[key]
        if not transitions:
            continue
        print(f"{key}键：")
        for (src, dst), count in sorted(
            transitions.items(), key=lambda x: (-x[1], x[0][0], x[0][1])
        ):
            print(f"{src} -> {dst} *{count}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "检查数据集根目录中的 valid_dicts_report.csv 与 "
            "valid_dicts_report_original.csv 差异并按值变化次数压缩输出"
        )
    )
    parser.add_argument(
        "dataset_root",
        nargs="?",
        default=".",
        help="数据集根目录（默认当前目录）",
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()
    dataset_root = Path(args.dataset_root).expanduser().resolve()

    modified_path = dataset_root / "valid_dicts_report.csv"
    original_path = dataset_root / "valid_dicts_report_original.csv"

    for p in (modified_path, original_path):
        if not p.exists():
            raise FileNotFoundError(f"未找到文件: {p}")

    modified_rows = read_csv_rows(modified_path)
    original_rows = read_csv_rows(original_path)

    all_keys = sorted(
        set().union(*(row.keys() for row in original_rows))
        | set().union(*(row.keys() for row in modified_rows))
    )

    transitions = compare_rows(original_rows, modified_rows, all_keys)
    print_result(transitions)


if __name__ == "__main__":
    main()
