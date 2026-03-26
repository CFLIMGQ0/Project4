from __future__ import annotations

import argparse
import csv
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from shutil import get_terminal_size

from check_pdf import CONFIG_PATH, load_yaml_config

try:
    from tqdm import tqdm  # type: ignore
except ImportError:
    tqdm = None

IMAGE_SUFFIXES = {'.jpg', '.jpeg', '.png', '.bmp', '.gif', '.tif', '.tiff', '.webp'}


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
    parser = argparse.ArgumentParser(description='统计每个检查目录中的 img 图片数量分布（按数量升序输出）')
    parser.add_argument('--config', type=Path, default=CONFIG_PATH, help='路径配置文件，默认 configs/path.yaml')
    parser.add_argument(
        '--report-csv',
        type=Path,
        default=None,
        help='可选：覆盖配置中的 valid_dicts_report_csv 路径（未配置时默认 dataset_base_root/valid_dicts_report.csv）',
    )
    return parser.parse_args()


def build_path_config(config_path: Path, report_csv_override: Path | None) -> PathConfig:
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

    dataset_root = resolve_path(str(paths_payload['dataset_root']))
    dataset_base_root = resolve_path(str(paths_payload['dataset_base_root']))
    report_csv_config = paths_payload.get('valid_dicts_report_csv')
    report_csv_path = (
        report_csv_override.expanduser().resolve()
        if report_csv_override is not None
        else resolve_path(str(report_csv_config)) if report_csv_config else (dataset_base_root / 'valid_dicts_report.csv').resolve()
    )
    return PathConfig(dataset_root=dataset_root, report_csv_path=report_csv_path)


def load_exam_dirs(report_csv_path: Path) -> list[str]:
    if not report_csv_path.is_file():
        raise FileNotFoundError(f'未找到报告汇总文件：{report_csv_path}')

    with report_csv_path.open('r', encoding='utf-8-sig', newline='') as f:
        reader = csv.DictReader(f)
        rows = list(reader)
        fieldnames = reader.fieldnames or []

    if 'exam_dir' not in fieldnames:
        raise KeyError(f'{report_csv_path} 中缺少必需字段：exam_dir')

    exam_dirs: list[str] = []
    for row in rows:
        raw = str(row.get('exam_dir', '')).strip()
        if raw:
            exam_dirs.append(raw)
    return exam_dirs


def resolve_exam_dir(raw_exam_dir: str, dataset_root: Path) -> Path:
    exam_dir_path = Path(raw_exam_dir).expanduser()
    if exam_dir_path.is_absolute():
        return exam_dir_path
    return (dataset_root / exam_dir_path).resolve()


def count_images_in_img_dir(exam_dir: Path) -> int:
    img_dir = exam_dir / 'img'
    if not img_dir.is_dir():
        return 0
    return sum(1 for item in img_dir.iterdir() if item.is_file() and item.suffix.lower() in IMAGE_SUFFIXES)


def build_distribution(exam_dirs: list[str], dataset_root: Path) -> tuple[Counter[int], int]:
    unique_exam_dirs = sorted(set(exam_dirs))
    distribution: Counter[int] = Counter()

    progress = build_progress(total=len(unique_exam_dirs), desc='统计检查目录图片数量')
    try:
        for raw_exam_dir in unique_exam_dirs:
            exam_dir = resolve_exam_dir(raw_exam_dir, dataset_root)
            img_count = count_images_in_img_dir(exam_dir)
            distribution[img_count] += 1
            progress.update(1)
    finally:
        progress.close()

    return distribution, len(unique_exam_dirs)


def print_distribution(distribution: Counter[int], total_exam_dirs: int) -> None:
    print(f'检查目录总数：{total_exam_dirs}')
    print('图片数量分布（按 img 数量从小到大）：')
    for img_count in sorted(distribution.keys()):
        case_count = distribution[img_count]
        print(f'img数量为{img_count}：{case_count}例')


def main() -> None:
    args = parse_args()
    config = build_path_config(args.config, args.report_csv)
    exam_dirs = load_exam_dirs(config.report_csv_path)
    distribution, total_exam_dirs = build_distribution(exam_dirs=exam_dirs, dataset_root=config.dataset_root)
    print_distribution(distribution=distribution, total_exam_dirs=total_exam_dirs)


if __name__ == '__main__':
    main()
