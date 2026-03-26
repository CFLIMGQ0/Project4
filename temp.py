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
        help='可选：覆盖配置中的 valid_dicts_report_csv 路径（未配置时默认 dataset_base_root/valid_dicts_report.csv）',
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
    report_csv_config = paths_payload.get('valid_dicts_report_csv')
    report_csv_path = (
        report_csv_override.expanduser().resolve()
        if report_csv_override is not None
        else resolve_path(str(report_csv_config)) if report_csv_config else (dataset_base_root / 'valid_dicts_report.csv').resolve()
    )
    return PathConfig(dataset_root=dataset_root, report_csv_path=report_csv_path)


def iter_patient_dirs(dataset_root: Path) -> list[Path]:
    return sorted(path for path in dataset_root.iterdir() if path.is_dir())


def normalize_patient_name(name: str) -> str:
    return name.strip().replace(' ', '')


def load_report_rows(report_csv_path: Path) -> list[dict[str, str]]:
    if not report_csv_path.is_file():
        raise FileNotFoundError(f'未找到报告汇总文件：{report_csv_path}')

    with report_csv_path.open('r', encoding='utf-8-sig', newline='') as f:
        reader = csv.DictReader(f)
        rows = list(reader)
        fieldnames = reader.fieldnames or []

    required = {'exam_dir', 'namePatient', 'reportTitle'}
    missing = required - set(fieldnames)
    if missing:
        missing_fields = '、'.join(sorted(missing))
        raise KeyError(f'{report_csv_path} 中缺少必需字段：{missing_fields}')
    return rows


def check_exam_patient_consistency(rows: list[dict[str, str]], patient_dirs: list[Path]) -> tuple[list[tuple[str, str, str]], dict[str, str]]:
    mismatches: list[tuple[str, str, str]] = []
    patient_name_map: dict[str, str] = {}
    patient_row_map: dict[str, list[tuple[str, str]]] = defaultdict(list)
    patient_scope = {str(path.resolve()) for path in patient_dirs}
    progress = build_progress(total=len(rows), desc='检查同一患者目录下不同检查目录的 namePatient 是否一致')

    try:
        for row in rows:
            exam_dir_raw = str(row.get('exam_dir', '')).strip()
            name_patient = normalize_patient_name(str(row.get('namePatient', '')).strip())
            if not exam_dir_raw:
                progress.update(1)
                continue

            exam_dir = Path(exam_dir_raw).expanduser().resolve()
            patient_dir = str(exam_dir.parent)
            if patient_scope and patient_dir not in patient_scope:
                progress.update(1)
                continue

            patient_row_map[patient_dir].append((exam_dir.name, name_patient))
            progress.update(1)
    finally:
        progress.close()

    for patient_dir, records in patient_row_map.items():
        non_empty_names = [name for _, name in records if name]
        if non_empty_names:
            patient_name_map[patient_dir] = non_empty_names[0]

        if len(records) <= 1:
            continue

        unique_names = set(non_empty_names)
        if len(unique_names) > 1:
            for exam_name, name_patient in records:
                mismatches.append((patient_dir, exam_name, name_patient or '（空）'))

    return mismatches, patient_name_map


def find_duplicate_patient_names(patient_name_map: dict[str, str]) -> dict[str, list[str]]:
    grouped: dict[str, list[str]] = defaultdict(list)
    progress = build_progress(total=len(patient_name_map), desc='检查患者重名')
    try:
        for patient_dir, patient_name in patient_name_map.items():
            grouped[patient_name].append(patient_dir)
            progress.update(1)
    finally:
        progress.close()
    return {name: paths for name, paths in grouped.items() if len(paths) > 1}


def load_report_titles(rows: list[dict[str, str]]) -> set[str]:
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


def print_mismatch(mismatches: Iterable[tuple[str, str, str]]) -> None:
    records = sorted(set(mismatches))
    print('发现同一患者目录下检查目录中的病人名字不一致，详情如下：')
    for idx, (patient_dir, exam_name, extracted_name) in enumerate(records, start=1):
        print(f'{idx}. 患者目录：{patient_dir} | 检查目录：{exam_name} | 解析病人名：{extracted_name}')


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

    rows = load_report_rows(config.report_csv_path)
    print(f'报告记录总数：{len(rows)}')

    mismatches, patient_name_map = check_exam_patient_consistency(rows, patient_dirs)
    if mismatches:
        print_mismatch(mismatches)
        return

    print('步骤1通过：同一患者目录下不同检查目录对应 PDF 的 namePatient 一致。')
    duplicates = find_duplicate_patient_names(patient_name_map)
    if duplicates:
        print_duplicates(duplicates)
        return

    print('步骤2通过：未发现患者重名。')
    titles = load_report_titles(rows)
    print_report_titles(titles)


if __name__ == '__main__':
    main()
