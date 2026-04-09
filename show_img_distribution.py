#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import yaml

from training.data import IMAGE_EXTENSIONS, resolve_exam_dir

PROJECT_ROOT = Path(__file__).resolve().parent
DEFAULT_CONFIG_PATH = PROJECT_ROOT / "configs" / "path.yaml"

try:
    from tqdm import tqdm
except Exception:  # pragma: no cover
    tqdm = None


@dataclass
class PathConfig:
    dataset_root: Path
    report_csv_path: Path


def build_progress(iterable: Iterable, desc: str):
    if tqdm is not None:
        return tqdm(iterable, desc=desc, dynamic_ncols=True)
    return iterable


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="统计检查目录中的 img 数量分布")
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG_PATH, help="路径配置文件，默认 configs/path.yaml")
    parser.add_argument(
        "--source",
        choices=["report", "dataset"],
        default="report",
        help="统计来源：report 表示按 valid_dicts_report_csv 中的检查目录统计；dataset 表示扫描 dataset_root 下所有检查目录",
    )
    parser.add_argument("--report-csv", type=Path, default=None, help="可选：覆盖配置中的 valid_dicts_report_csv")
    parser.add_argument("--dataset-root", type=Path, default=None, help="可选：覆盖配置中的 dataset_root")
    return parser.parse_args()


def resolve_config_path(raw_path: str | Path, config_path: Path) -> Path:
    path = Path(raw_path).expanduser()
    if path.is_absolute():
        return path
    return (config_path.parent / path).resolve()


def load_path_config(config_path: Path, report_csv_override: Path | None, dataset_root_override: Path | None) -> PathConfig:
    config_path = config_path.expanduser().resolve()
    payload = yaml.safe_load(config_path.read_text(encoding="utf-8")) or {}
    paths = payload.get("paths", {})
    if not isinstance(paths, dict):
        raise ValueError(f"{config_path} 中缺少 paths 配置")

    if "dataset_root" not in paths and dataset_root_override is None:
        raise KeyError(f"{config_path} 中缺少 paths.dataset_root")
    if "valid_dicts_report_csv" not in paths and report_csv_override is None:
        raise KeyError(f"{config_path} 中缺少 paths.valid_dicts_report_csv")

    dataset_root = (
        dataset_root_override.expanduser().resolve()
        if dataset_root_override is not None
        else resolve_config_path(paths["dataset_root"], config_path)
    )
    report_csv_path = (
        report_csv_override.expanduser().resolve()
        if report_csv_override is not None
        else resolve_config_path(paths["valid_dicts_report_csv"], config_path)
    )
    return PathConfig(dataset_root=dataset_root, report_csv_path=report_csv_path)


def iter_exam_dirs_from_dataset(dataset_root: Path) -> list[Path]:
    if not dataset_root.exists():
        raise FileNotFoundError(f"dataset_root 不存在：{dataset_root}")
    if not dataset_root.is_dir():
        raise NotADirectoryError(f"dataset_root 不是目录：{dataset_root}")

    patient_dirs = sorted(path for path in dataset_root.iterdir() if path.is_dir())
    exam_dirs: list[Path] = []
    for patient_dir in build_progress(patient_dirs, desc="扫描患者目录"):
        exam_dirs.extend(sorted(path for path in patient_dir.iterdir() if path.is_dir()))
    return exam_dirs


def iter_exam_dirs_from_report(report_csv_path: Path, dataset_root: Path) -> tuple[list[Path], int, int]:
    if not report_csv_path.is_file():
        raise FileNotFoundError(f"未找到报告 CSV：{report_csv_path}")

    with report_csv_path.open("r", encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        rows = list(reader)
        fieldnames = reader.fieldnames or []

    if "exam_dir" not in fieldnames:
        raise KeyError(f"{report_csv_path} 中缺少 exam_dir 字段")

    seen: set[str] = set()
    exam_dirs: list[Path] = []
    duplicate_count = 0
    missing_count = 0

    for row in build_progress(rows, desc="解析检查目录"):
        raw_exam_dir = str(row.get("exam_dir", "")).strip()
        if not raw_exam_dir:
            missing_count += 1
            continue

        exam_dir, _ = resolve_exam_dir(raw_exam_dir, dataset_root=dataset_root)
        if not exam_dir.is_dir():
            missing_count += 1
            continue

        key = str(exam_dir.resolve())
        if key in seen:
            duplicate_count += 1
            continue

        seen.add(key)
        exam_dirs.append(exam_dir)

    return exam_dirs, duplicate_count, missing_count


def count_images_for_exam(exam_dir: Path) -> int:
    img_dir = exam_dir / "img"
    if not img_dir.is_dir():
        return 0
    return sum(1 for path in img_dir.rglob("*") if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS)


def build_distribution(exam_dirs: list[Path]) -> Counter[int]:
    distribution: Counter[int] = Counter()
    for exam_dir in build_progress(exam_dirs, desc="统计 img 数量"):
        distribution[count_images_for_exam(exam_dir)] += 1
    return distribution


def print_distribution(distribution: Counter[int], total_exam_dirs: int, source: str, duplicate_count: int = 0, missing_count: int = 0) -> None:
    source_text = "valid_dicts_report_csv（去重后）" if source == "report" else "dataset_root"
    print(f"统计来源：{source_text}")
    print(f"检查目录总数：{total_exam_dirs}")
    if source == "report":
        print(f"重复检查目录条目数：{duplicate_count}")
        print(f"缺失或无效检查目录条目数：{missing_count}")
    print("检查目录中的img数量分布：")
    for img_count in sorted(distribution):
        print(f"img数量为{img_count}：{distribution[img_count]}")


def main() -> None:
    args = parse_args()
    path_cfg = load_path_config(args.config, args.report_csv, args.dataset_root)

    if args.source == "dataset":
        exam_dirs = iter_exam_dirs_from_dataset(path_cfg.dataset_root)
        distribution = build_distribution(exam_dirs)
        print_distribution(distribution, total_exam_dirs=len(exam_dirs), source="dataset")
        return

    exam_dirs, duplicate_count, missing_count = iter_exam_dirs_from_report(
        report_csv_path=path_cfg.report_csv_path,
        dataset_root=path_cfg.dataset_root,
    )
    distribution = build_distribution(exam_dirs)
    print_distribution(
        distribution,
        total_exam_dirs=len(exam_dirs),
        source="report",
        duplicate_count=duplicate_count,
        missing_count=missing_count,
    )


if __name__ == "__main__":
    main()
