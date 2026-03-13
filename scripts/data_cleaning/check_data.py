"""核验 all_patients_raw.xlsx 与患者目录的姓名一致性。"""

from __future__ import annotations

import argparse
import re
from pathlib import Path
from typing import Iterable, Set

import pandas as pd


NAME_COLUMN_CANDIDATES = [
    "patient_name",
    "name",
    "patient",
    "姓名",
    "患者姓名",
    "病人姓名",
    "患者",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="检查 all_patients_raw.xlsx 中患者姓名是否与数据集目录一致"
    )
    parser.add_argument(
        "--dataset-root",
        type=Path,
        default=None,
        help="数据集根目录（每个患者一个子目录）",
    )
    parser.add_argument(
        "--excel-path",
        type=Path,
        default=None,
        help="all_patients_raw.xlsx 文件路径",
    )
    parser.add_argument(
        "--name-column",
        type=str,
        default=None,
        help="手动指定 Excel 中患者姓名列名（不传则自动识别）",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("scripts/data_cleaning/output"),
        help="输出报告目录",
    )
    return parser.parse_args()


def _choose_existing_path(candidates: list[Path]) -> Path | None:
    for candidate in candidates:
        expanded = candidate.expanduser()
        if expanded.exists():
            return expanded
    return None


def resolve_input_paths(args: argparse.Namespace) -> tuple[Path, Path]:
    dataset_root = args.dataset_root.expanduser() if args.dataset_root else None
    excel_path = args.excel_path.expanduser() if args.excel_path else None

    if dataset_root is None:
        dataset_root = _choose_existing_path(
            [
                Path("data/raw/eus_dataset"),
                Path("data/eus_dataset"),
                Path("~/.cache/kagglehub/datasets/eus_dataset"),
            ]
        )

    if excel_path is None:
        excel_path = _choose_existing_path(
            [
                Path("data/raw/all_patients_raw.xlsx"),
                Path("data/all_patients_raw.xlsx"),
                Path("all_patients_raw.xlsx"),
            ]
        )

    missing_flags = []
    if dataset_root is None:
        missing_flags.append("--dataset-root")
    if excel_path is None:
        missing_flags.append("--excel-path")

    if missing_flags:
        joined = "、".join(missing_flags)
        raise ValueError(
            f"缺少必要参数：{joined}。\n"
            "请显式传参，示例：\n"
            "python scripts/data_cleaning/check_data.py "
            "--dataset-root /path/to/eus_dataset "
            "--excel-path /path/to/all_patients_raw.xlsx"
        )

    return dataset_root, excel_path


def normalize_name(name: str) -> str:
    text = re.sub(r"\s+", "", str(name))
    return text.strip(";；")


def extract_name_from_folder(folder_name: str) -> str:
    # 目录格式示例：2022-02-08; 陈慧; 1195061
    parts = [p.strip() for p in re.split(r"[;；]", folder_name) if p.strip()]
    if len(parts) >= 2:
        return normalize_name(parts[1])
    return normalize_name(folder_name)


def collect_folder_names(dataset_root: Path) -> Set[str]:
    names = set()
    for path in dataset_root.iterdir():
        if path.is_dir():
            parsed = extract_name_from_folder(path.name)
            if parsed:
                names.add(parsed)
    return names


def detect_name_column(df: pd.DataFrame, manual_column: str | None) -> str:
    if manual_column:
        if manual_column not in df.columns:
            raise ValueError(f"指定列 {manual_column} 不存在，当前列：{list(df.columns)}")
        return manual_column

    lower_map = {str(col).lower(): col for col in df.columns}
    for candidate in NAME_COLUMN_CANDIDATES:
        if candidate.lower() in lower_map:
            return lower_map[candidate.lower()]

    raise ValueError(
        "自动识别失败：未找到患者姓名列。请通过 --name-column 手动指定。"
    )


def collect_excel_names(series: Iterable[object]) -> Set[str]:
    names = set()
    for value in series:
        if pd.isna(value):
            continue
        parsed = normalize_name(str(value))
        if parsed:
            names.add(parsed)
    return names


def main() -> None:
    args = parse_args()
    dataset_root, excel_path = resolve_input_paths(args)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    if not dataset_root.exists():
        raise FileNotFoundError(f"dataset-root 不存在：{dataset_root}")
    if not excel_path.exists():
        raise FileNotFoundError(f"excel-path 不存在：{excel_path}")

    df = pd.read_excel(excel_path)
    name_column = detect_name_column(df, args.name_column)

    folder_names = collect_folder_names(dataset_root)
    excel_names = collect_excel_names(df[name_column])

    only_in_excel = sorted(excel_names - folder_names)
    only_in_folders = sorted(folder_names - excel_names)
    in_both = sorted(excel_names & folder_names)

    report_path = args.output_dir / "name_check_report.md"
    lines = [
        "# 患者姓名一致性核验报告",
        "",
        f"- 数据集目录：`{dataset_root}`",
        f"- Excel 文件：`{excel_path}`",
        f"- Excel 姓名列：`{name_column}`",
        f"- 目录姓名数量：{len(folder_names)}",
        f"- Excel 姓名数量：{len(excel_names)}",
        f"- 匹配成功数量：{len(in_both)}",
        f"- 仅 Excel 中存在：{len(only_in_excel)}",
        f"- 仅目录中存在：{len(only_in_folders)}",
        "",
        "## 仅 Excel 中存在的姓名",
    ]
    lines.extend([f"- {name}" for name in only_in_excel] or ["- 无"])
    lines.append("")
    lines.append("## 仅目录中存在的姓名")
    lines.extend([f"- {name}" for name in only_in_folders] or ["- 无"])

    report_path.write_text("\n".join(lines), encoding="utf-8")

    print(f"核验完成，报告已输出：{report_path}")


if __name__ == "__main__":
    main()
