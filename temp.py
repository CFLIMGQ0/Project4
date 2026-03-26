from __future__ import annotations

import argparse
import csv
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from shutil import get_terminal_size
from typing import Iterable

from check_pdf import CONFIG_PATH, load_yaml_config

try:
    from tqdm import tqdm  # type: ignore
except ImportError:
    tqdm = None


@dataclass
class PathConfig:
    dataset_root: Path
    report_csv_path: Path


class SimpleProgressBar:
    def __init__(self, total: int, desc: str) -> None:
        self.total = max(total, 1)
        self.current = 0
        self.desc = desc

    def update(self, step: int = 1) -> None:
        self.current = min(self.total, self.current + step)
        width = min(40, max(10, get_terminal_size((80, 20)).columns - 42))
        ratio = self.current / self.total
        done = int(width * ratio)
        bar = '=' * done + '-' * (width - done)
        print(f'\r{self.desc}: [{bar}] {self.current}/{self.total} ({ratio * 100:5.1f}%)', end='', flush=True)
        if self.current >= self.total:
            print()

    def close(self) -> None:
        return


def build_progress(total: int, desc: str):
    if tqdm is not None:
        return tqdm(total=total, desc=desc, unit='项')
    return SimpleProgressBar(total=total, desc=desc)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description='按“患者目录 -> 检查目录 -> reportTitle”顺序执行一致性检查'
    )
    parser.add_argument('--config', type=Path, default=CONFIG_PATH, help='路径配置文件，默认 configs/path.yaml')
    parser.add_argument('--dataset-root', type=Path, default=None, help='可选：覆盖配置中的 dataset_root')
    parser.add_argument(
        '--report-csv',
        type=Path,
        default=None,
        help='可选：覆盖配置中的 valid_dicts_report.csv 路径（默认 dataset_base_root/valid_dicts_report.csv）',
    )
    return parser.parse_args()


def build_path_config(config_path: Path, dataset_root_override: Path | None, report_csv_override: Path | None) -> PathConfig:
    payload = load_yaml_config(config_path.expanduser())
    paths_payload = payload.get('paths')
    if not isinstance(paths_payload, dict):
        raise ValueError('path.yaml 必须包含 paths 分组')

    config_dir = config_path.expanduser().resolve().parent

    def resolve_path(raw_path: str) -> Path:
        path = Path(raw_path).expanduser()
        if path.is_absolute():
            return path
        return (config_dir.parent / path).resolve()

    dataset_root = (
        dataset_root_override.expanduser().resolve()
        if dataset_root_override is not None
        else resolve_path(str(paths_payload['dataset_root']))
    )
    dataset_base_root = resolve_path(str(paths_payload['dataset_base_root']))
    report_csv_path = (
        report_csv_override.expanduser().resolve()
        if report_csv_override is not None
        else (dataset_base_root / 'valid_dicts_report.csv').resolve()
    )
    return PathConfig(dataset_root=dataset_root, report_csv_path=report_csv_path)


def iter_patient_dirs(dataset_root: Path) -> list[Path]:
    return sorted(path for path in dataset_root.iterdir() if path.is_dir())


def iter_exam_dirs(patient_dir: Path) -> list[Path]:
    return sorted(path for path in patient_dir.iterdir() if path.is_dir())


def is_exam_name_consistent(patient_name: str, exam_name: str) -> bool:
    # 兼容“完全相同”与“检查目录以患者名开头（如加后缀）”两种情况。
    return exam_name == patient_name or exam_name.startswith(f'{patient_name}_') or exam_name.startswith(patient_name)


def check_exam_patient_consistency(patient_dirs: list[Path]) -> list[tuple[str, str]]:
    mismatches: list[tuple[str, str]] = []
    progress = build_progress(total=len(patient_dirs), desc='检查患者目录与检查目录名一致性')
    try:
        for patient_dir in patient_dirs:
            patient_name = patient_dir.name
            exam_dirs = iter_exam_dirs(patient_dir)
            for exam_dir in exam_dirs:
                exam_name = exam_dir.name
                if not is_exam_name_consistent(patient_name, exam_name):
                    mismatches.append((patient_name, exam_name))
            progress.update(1)
    finally:
        progress.close()
    return mismatches


def find_duplicate_patient_names(patient_dirs: list[Path]) -> dict[str, list[str]]:
    grouped: dict[str, list[str]] = defaultdict(list)
    progress = build_progress(total=len(patient_dirs), desc='检查患者重名')
    try:
        for patient_dir in patient_dirs:
            grouped[patient_dir.name].append(str(patient_dir))
            progress.update(1)
    finally:
        progress.close()
    return {name: paths for name, paths in grouped.items() if len(paths) > 1}


def load_report_titles(report_csv_path: Path) -> set[str]:
    if not report_csv_path.is_file():
        raise FileNotFoundError(f'未找到报告汇总文件：{report_csv_path}')

    with report_csv_path.open('r', encoding='utf-8-sig', newline='') as f:
        reader = csv.DictReader(f)
        rows = list(reader)

    if not rows:
        return set()

    if 'reportTitle' not in reader.fieldnames:
        raise KeyError(f'{report_csv_path} 中不存在 reportTitle 字段')

    titles: set[str] = set()
    progress = build_progress(total=len(rows), desc='统计 reportTitle 类型')
    try:
        for row in rows:
            title = str(row.get('reportTitle', '')).strip()
            if title:
                titles.add(title)
            progress.update(1)
    finally:
        progress.close()
    return titles


def print_mismatch(mismatches: Iterable[tuple[str, str]]) -> None:
    records = sorted(set(mismatches))
    print('发现患者目录与检查目录名不一致，详情如下：')
    for idx, (patient_name, exam_name) in enumerate(records, start=1):
        print(f'{idx}. 患者目录名：{patient_name} | 检查目录名：{exam_name}')


def print_duplicates(duplicates: dict[str, list[str]]) -> None:
    print('发现重名患者目录，详情如下：')
    for idx, (name, paths) in enumerate(sorted(duplicates.items()), start=1):
        print(f'{idx}. 患者名：{name}（共 {len(paths)} 个目录）')
        for path in sorted(paths):
            print(f'   - {path}')


def print_report_titles(titles: set[str]) -> None:
    sorted_titles = sorted(titles)
    print(f'共发现 {len(sorted_titles)} 种 reportTitle：')
    for idx, title in enumerate(sorted_titles, start=1):
        print(f'{idx}. {title}')


def main() -> None:
    args = parse_args()
    config = build_path_config(args.config, args.dataset_root, args.report_csv)

    if not config.dataset_root.is_dir():
        raise NotADirectoryError(f'dataset_root 不存在或不是目录：{config.dataset_root}')

    patient_dirs = iter_patient_dirs(config.dataset_root)
    print(f'患者目录总数：{len(patient_dirs)}')

    mismatches = check_exam_patient_consistency(patient_dirs)
    if mismatches:
        print_mismatch(mismatches)
        return

    print('步骤1通过：所有检查目录的病人名字与患者目录一致。')
    duplicates = find_duplicate_patient_names(patient_dirs)
    if duplicates:
        print_duplicates(duplicates)
        return

    print('步骤2通过：未发现患者重名。')
    titles = load_report_titles(config.report_csv_path)
    print_report_titles(titles)


if __name__ == '__main__':
    main()
