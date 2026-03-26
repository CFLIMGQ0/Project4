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


@dataclass
class PathConfig:
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
        description='仅统计报告中的 reportTitle 内容'
    )
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

    dataset_base_root = resolve_path(str(paths_payload['dataset_base_root']))
    report_csv_config = paths_payload.get('valid_dicts_report_csv')
    report_csv_path = (
        report_csv_override.expanduser().resolve()
        if report_csv_override is not None
        else resolve_path(str(report_csv_config)) if report_csv_config else (dataset_base_root / 'valid_dicts_report.csv').resolve()
    )
    return PathConfig(report_csv_path=report_csv_path)


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


def load_report_title_counts(rows: list[dict[str, str]]) -> Counter[str]:
    title_counts: Counter[str] = Counter()
    progress = build_progress(total=len(rows), desc='统计 reportTitle 类型')
    try:
        for row in rows:
            title = str(row.get('reportTitle', '')).strip()
            if title:
                title_counts[title] += 1
            progress.update(1)
    finally:
        progress.close()
    return title_counts


def print_report_title_counts(title_counts: Counter[str]) -> None:
    sorted_title_counts = sorted(title_counts.items(), key=lambda item: (-item[1], item[0]))
    print(f'共发现 {len(sorted_title_counts)} 种 reportTitle（按出现次数降序）：')
    for idx, (title, count) in enumerate(sorted_title_counts, start=1):
        print(f'{idx}. {title}: {count}')


def main() -> None:
    args = parse_args()
    config = build_path_config(args.config, args.report_csv)

    rows = load_report_rows(config.report_csv_path)
    print(f'报告记录总数：{len(rows)}')
    title_counts = load_report_title_counts(rows)
    print_report_title_counts(title_counts)


if __name__ == '__main__':
    main()
