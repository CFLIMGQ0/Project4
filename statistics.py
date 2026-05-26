from __future__ import annotations

import argparse
import csv
import re
import sys
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from shutil import get_terminal_size
from unicodedata import combining, east_asian_width


@dataclass
class PathConfig:
    dataset_base_root: Path
    report_csv_path: Path


class SimpleProgressBar:
    def __init__(self, total: int, desc: str) -> None:
        self.total = max(total, 1)
        self.current = 0
        self.desc = desc
        self._is_tty = sys.stdout.isatty()
        self._last_reported_percent = -1
        self._last_rendered_width = 0

    @staticmethod
    def _display_width(text: str) -> int:
        width = 0
        for char in text:
            if combining(char):
                continue
            width += 2 if east_asian_width(char) in {'W', 'F'} else 1
        return width

    @classmethod
    def _truncate_to_width(cls, text: str, max_width: int) -> str:
        if max_width <= 0:
            return ''
        if cls._display_width(text) <= max_width:
            return text

        ellipsis = '...'
        ellipsis_width = cls._display_width(ellipsis)
        if max_width <= ellipsis_width:
            return '.' * max_width

        kept_chars: list[str] = []
        kept_width = 0
        target_width = max_width - ellipsis_width
        for char in text:
            char_width = cls._display_width(char)
            if kept_width + char_width > target_width:
                break
            kept_chars.append(char)
            kept_width += char_width
        return ''.join(kept_chars) + ellipsis

    def _build_line(self, ratio: float) -> str:
        terminal_width = max(20, get_terminal_size((80, 20)).columns)
        progress_text = f'{self.current}/{self.total}'
        percent_text = f'({ratio * 100:5.1f}%)'

        compact_line = f'{progress_text} {percent_text}'
        if self._display_width(compact_line) >= terminal_width:
            return compact_line

        suffix = f'] {progress_text} {percent_text}'
        max_bar_width = 40

        desc_limit = terminal_width - self._display_width(': [') - self._display_width(suffix) - 10
        desc = self._truncate_to_width(self.desc, desc_limit)
        prefix = f'{desc}: [' if desc else '['
        bar_width = min(
            max_bar_width,
            max(1, terminal_width - self._display_width(prefix) - self._display_width(suffix)),
        )
        if bar_width >= 10:
            done = int(bar_width * ratio)
            bar = '=' * done + '-' * (bar_width - done)
            return f'{prefix}{bar}{suffix}'

        fallback_desc = self._truncate_to_width(
            self.desc,
            terminal_width - self._display_width(f': {progress_text} {percent_text}'),
        )
        if fallback_desc:
            return f'{fallback_desc}: {progress_text} {percent_text}'
        return compact_line

    def _render_tty(self, ratio: float) -> None:
        line = self._build_line(ratio)
        line_width = self._display_width(line)
        padding = max(0, self._last_rendered_width - line_width)
        print(f'\r{line}{" " * padding}', end='', flush=True)
        self._last_rendered_width = line_width

    def update(self, step: int = 1) -> None:
        self.current = min(self.total, self.current + step)
        ratio = self.current / self.total
        percent = int(ratio * 100)

        if not self._is_tty:
            if percent > self._last_reported_percent and (percent % 10 == 0 or self.current >= self.total):
                print(f'{self.desc}: {self.current}/{self.total} ({ratio * 100:5.1f}%)')
                self._last_reported_percent = percent
            return

        self._render_tty(ratio)
        if self.current >= self.total:
            print()
            self._last_rendered_width = 0

    def close(self) -> None:
        if self._is_tty and self._last_rendered_width > 0:
            print()
            self._last_rendered_width = 0


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
    parser.add_argument('--config', type=Path, default=Path('configs/task1/path.yaml'), help='路径配置文件，默认 configs/task1/path.yaml')
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


def normalize_operation_item(value: str) -> str:
    cleaned = value.replace('\u3000', ' ').strip()
    cleaned = re.sub(r'\s*[（(]\d[0-9A-Za-z.xX.-]*[)）]\s*$', '', cleaned)
    return cleaned.strip()


def split_operation_values(value: str) -> list[str]:
    cleaned = value.replace('\u3000', ' ').strip()
    if not cleaned:
        return ['空值']

    parts = [normalize_operation_item(part) for part in re.split(r'[|，,]', cleaned)]
    values = [part for part in parts if part]
    return values if values else ['空值']


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
            operation_values = split_operation_values(str(row.get('operationValue', '')))
            stats[organ]['badness'][badness] += 1
            stats[organ]['hp'][hp] += 1
            for operation_value in operation_values:
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
