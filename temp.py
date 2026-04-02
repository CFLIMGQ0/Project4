from __future__ import annotations

import argparse
import csv
import re
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from shutil import get_terminal_size


@dataclass
class PathConfig:
    dataset_base_root: Path
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


def parse_simple_yaml_mapping(yaml_path: Path) -> dict[str, str]:
    text = yaml_path.read_text(encoding='utf-8')
    in_paths = False
    result: dict[str, str] = {}
    for raw_line in text.splitlines():
        line = raw_line.split('#', 1)[0].rstrip()
        if not line.strip():
            continue
        if re.match(r'^\s*paths\s*:\s*$', line):
            in_paths = True
            continue
        if not in_paths:
            continue
        if re.match(r'^\S', raw_line):
            break

        m = re.match(r'^\s{2,}([A-Za-z0-9_]+)\s*:\s*(.+?)\s*$', line)
        if not m:
            continue
        key, value = m.group(1), m.group(2)
        value = value.strip().strip('"').strip("'")
        result[key] = value
    if not result:
        raise ValueError(f'无法从 {yaml_path} 解析 paths 配置')
    return result


def resolve_path(raw_path: str, config_dir: Path) -> Path:
    path = Path(raw_path).expanduser()
    if path.is_absolute():
        return path
    return (config_dir.parent / path).resolve()


def build_path_config(config_path: Path, report_csv_override: Path | None) -> PathConfig:
    config_path = config_path.expanduser().resolve()
    paths_payload = parse_simple_yaml_mapping(config_path)
    dataset_base_root = resolve_path(paths_payload['dataset_base_root'], config_path.parent)
    if report_csv_override is not None:
        report_csv_path = report_csv_override.expanduser().resolve()
    else:
        report_csv_raw = paths_payload.get('valid_dicts_report_csv', 'valid_dicts_report.csv')
        report_csv_path = resolve_path(report_csv_raw, config_path.parent)
    return PathConfig(dataset_base_root=dataset_base_root, report_csv_path=report_csv_path)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description='统计胃镜/肠镜 badness、hp 与 operationValue 类型次数')
    parser.add_argument('--config', type=Path, default=Path('configs/path.yaml'), help='路径配置文件，默认 configs/path.yaml')
    parser.add_argument('--report-csv', type=Path, default=None, help='可选：覆盖 valid_dicts_report_csv')
    return parser.parse_args()


def load_rows(report_csv_path: Path) -> list[dict[str, str]]:
    if not report_csv_path.is_file():
        raise FileNotFoundError(f'未找到报告汇总文件：{report_csv_path}')

    with report_csv_path.open('r', encoding='utf-8-sig', newline='') as f:
        reader = csv.DictReader(f)
        rows = list(reader)
        fieldnames = set(reader.fieldnames or [])

    required = {'reportTitle', 'badness', 'hp', 'operationValue'}
    missing = sorted(required - fieldnames)
    if missing:
        raise KeyError(f'CSV 缺少字段：{"、".join(missing)}')
    return rows


def classify_organ(report_title: str) -> str | None:
    has_stomach = '胃' in report_title and '肠' not in report_title
    has_intestine = '肠' in report_title
    if has_stomach:
        return '胃镜'
    if has_intestine:
        return '肠镜'
    return None


def normalize_value(value: str, empty_label: str = '空值') -> str:
    cleaned = value.replace('\u3000', ' ').strip()
    return cleaned if cleaned else empty_label


def count_badness_hp(rows: list[dict[str, str]]) -> dict[str, dict[str, Counter[str]]]:
    stats = {
        '胃镜': {'badness': Counter(), 'hp': Counter(), 'operationValue': Counter()},
        '肠镜': {'badness': Counter(), 'hp': Counter(), 'operationValue': Counter()},
    }

    progress = SimpleProgressBar(total=len(rows), desc='统计 badness/hp/operationValue')
    try:
        for row in rows:
            report_title = str(row.get('reportTitle', '')).strip()
            organ = classify_organ(report_title)
            if organ is None:
                progress.update(1)
                continue

            badness = normalize_value(str(row.get('badness', '')))
            hp = normalize_value(str(row.get('hp', '')))
            operation_value = normalize_value(str(row.get('operationValue', '')))
            stats[organ]['badness'][badness] += 1
            stats[organ]['hp'][hp] += 1
            stats[organ]['operationValue'][operation_value] += 1
            progress.update(1)
    finally:
        progress.close()

    return stats


def print_type_counts(counter: Counter[str]) -> None:
    if not counter:
        print('无：0')
        return
    for name, count in counter.most_common():
        print(f'{name}：{count}')


def print_stats(stats: dict[str, dict[str, Counter[str]]]) -> None:
    for organ in ['胃镜', '肠镜']:
        print(f'\n{organ}：')
        print('badness的类型：')
        print_type_counts(stats[organ]['badness'])
        print('\nhp的类型：')
        print_type_counts(stats[organ]['hp'])
        print('\noperationValue的类型：')
        print_type_counts(stats[organ]['operationValue'])


def main() -> None:
    args = parse_args()
    cfg = build_path_config(args.config, args.report_csv)
    rows = load_rows(cfg.report_csv_path)
    print(f'报告记录总数：{len(rows)}')
    print(f'统计文件：{cfg.report_csv_path}')

    stats = count_badness_hp(rows)
    print_stats(stats)


if __name__ == '__main__':
    main()
