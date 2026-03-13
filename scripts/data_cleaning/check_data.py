"""核验 all_patients.xlsx 与患者目录的一致性（姓名与图像数量）。"""

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
        description="检查 all_patients.xlsx 中患者姓名、WLS/EUS 数量是否与数据集目录一致"
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
        help="all_patients.xlsx 文件路径",
    )
    parser.add_argument(
        "--name-column",
        type=str,
        default=None,
        help="手动指定 Excel 中患者姓名列名（不传则自动识别）",
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

    if excel_path is None and dataset_root is not None:
        excel_path = _choose_existing_path(
            [
                dataset_root / "all_patients.xlsx",
                dataset_root / "all_patients_raw.xlsx",
            ]
        )

    if excel_path is None:
        excel_path = _choose_existing_path(
            [
                Path("data/raw/all_patients.xlsx"),
                Path("data/all_patients.xlsx"),
                Path("all_patients.xlsx"),
                Path("data/raw/all_patients_raw.xlsx"),
                Path("data/all_patients_raw.xlsx"),
                Path("all_patients_raw.xlsx"),
                Path("~/.cache/kagglehub/datasets/eus_dataset/all_patients.xlsx"),
                Path("~/.cache/kagglehub/datasets/eus_dataset/all_patients_raw.xlsx"),
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
            "--excel-path /path/to/all_patients.xlsx\n"
            "或将 Excel 放在 dataset-root 下并命名为 all_patients.xlsx"
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


def _safe_int(value: object) -> int | None:
    if pd.isna(value):
        return None
    text = str(value).strip()
    if not text:
        return None
    try:
        return int(float(text))
    except ValueError:
        return None


def _detect_count_column(df: pd.DataFrame, candidates: list[str]) -> str:
    lower_map = {str(col).lower(): col for col in df.columns}
    for candidate in candidates:
        if candidate.lower() in lower_map:
            return lower_map[candidate.lower()]
    raise ValueError(f"自动识别失败：未找到列 {candidates}。")


def _classify_image_type(path: Path) -> str | None:
    stem = path.stem.lower()
    if any(k in stem for k in ["wls", "wle", "white"]):
        return "wls"
    if "eus" in stem:
        return "eus"
    return None


def count_images_in_patient_folder(folder: Path) -> tuple[int, int, int]:
    """返回 (wls_count, eus_count, unknown_count)。"""
    image_exts = {".png", ".jpg", ".jpeg", ".bmp", ".webp", ".tif", ".tiff"}
    image_paths = [
        p for p in folder.iterdir() if p.is_file() and p.suffix.lower() in image_exts
    ]

    wls_count = 0
    eus_count = 0
    unknown_paths: list[Path] = []

    for image_path in image_paths:
        # 1.png 视作报告图，不纳入 WLS/EUS 计数。
        if image_path.stem == "1":
            continue
        image_type = _classify_image_type(image_path)
        if image_type == "wls":
            wls_count += 1
        elif image_type == "eus":
            eus_count += 1
        else:
            unknown_paths.append(image_path)

    # 对无法通过文件名判断的图片，按数字序号兼容历史规则：2 记为 WLS，其余记为 EUS。
    for image_path in sorted(unknown_paths, key=lambda p: p.name):
        if image_path.stem.isdigit() and int(image_path.stem) == 2:
            wls_count += 1
        else:
            eus_count += 1

    return wls_count, eus_count, len(unknown_paths)


def build_folder_count_map(dataset_root: Path) -> dict[str, tuple[int, int, int]]:
    result: dict[str, tuple[int, int, int]] = {}
    for path in dataset_root.iterdir():
        if path.is_dir():
            name = extract_name_from_folder(path.name)
            if name:
                result[name] = count_images_in_patient_folder(path)
    return result


def main() -> None:
    args = parse_args()
    dataset_root, excel_path = resolve_input_paths(args)

    if not dataset_root.exists():
        raise FileNotFoundError(f"dataset-root 不存在：{dataset_root}")
    if not excel_path.exists():
        raise FileNotFoundError(f"excel-path 不存在：{excel_path}")

    df = pd.read_excel(excel_path)
    name_column = detect_name_column(df, args.name_column)
    wls_column = _detect_count_column(df, ["wls_count", "wls", "wle_count", "wle"])
    eus_column = _detect_count_column(df, ["eus_count", "eus"])

    folder_count_map = build_folder_count_map(dataset_root)
    folder_names = set(folder_count_map)
    excel_names = collect_excel_names(df[name_column])

    only_in_excel = sorted(excel_names - folder_names)
    only_in_folders = sorted(folder_names - excel_names)
    in_both = sorted(excel_names & folder_names)

    print("患者姓名一致性核验结果")
    print(f"- 数据集目录：{dataset_root}")
    print(f"- Excel 文件：{excel_path}")
    print(f"- Excel 姓名列：{name_column}")
    print(f"- Excel WLS 计数列：{wls_column}")
    print(f"- Excel EUS 计数列：{eus_column}")
    print(f"- 目录姓名数量：{len(folder_names)}")
    print(f"- Excel 姓名数量：{len(excel_names)}")
    print(f"- 匹配成功数量：{len(in_both)}")
    print(f"- 仅 Excel 中存在：{len(only_in_excel)}")
    print(f"- 仅目录中存在：{len(only_in_folders)}")

    print("\n仅 Excel 中存在的姓名：")
    if only_in_excel:
        for name in only_in_excel:
            print(f"- {name}")
    else:
        print("- 无")

    count_mismatches = []
    bad_rows = 0
    for _, row in df.iterrows():
        name = normalize_name(str(row.get(name_column, "")))
        if not name or name not in folder_count_map:
            continue

        excel_wls = _safe_int(row.get(wls_column))
        excel_eus = _safe_int(row.get(eus_column))
        if excel_wls is None or excel_eus is None:
            bad_rows += 1
            continue

        folder_wls, folder_eus, unknown_count = folder_count_map[name]
        if excel_wls != folder_wls or excel_eus != folder_eus:
            count_mismatches.append(
                {
                    "name": name,
                    "excel_wls": excel_wls,
                    "real_wls": folder_wls,
                    "excel_eus": excel_eus,
                    "real_eus": folder_eus,
                    "unknown_count": unknown_count,
                }
            )

    print("\nWLS/EUS 数量核验结果：")
    print(f"- 可用于计数核验的患者数：{len(in_both)}")
    print(f"- Excel 计数字段为空/非法的患者数：{bad_rows}")
    print(f"- 数量不一致患者数：{len(count_mismatches)}")

    if count_mismatches:
        print("\n数量不一致明细：")
        for item in count_mismatches:
            print(
                "- {name}: "
                "WLS(excel={excel_wls}, folder={real_wls}), "
                "EUS(excel={excel_eus}, folder={real_eus}), "
                "未显式标注类型图片数={unknown_count}".format(**item)
            )
    else:
        print("- 所有可核验患者的 WLS/EUS 数量均一致。")

    print("\n仅目录中存在的姓名：")
    if only_in_folders:
        for name in only_in_folders:
            print(f"- {name}")
    else:
        print("- 无")


if __name__ == "__main__":
    main()
