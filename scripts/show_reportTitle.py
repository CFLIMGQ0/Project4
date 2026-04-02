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

PROJECT_ROOT = Path(__file__).resolve().parent.parent
CONFIG_PATH = PROJECT_ROOT / 'configs' / 'path.yaml'

try:
    from tqdm import tqdm  # type: ignore
except ImportError:
    tqdm = None


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
        desc_limit = terminal_width - self._display_width(': [') - self._display_width(suffix) - 10
        desc = self._truncate_to_width(self.desc, desc_limit)
        prefix = f'{desc}: [' if desc else '['
        bar_width = min(
            40,
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


def build_progress(total: int, desc: str):
    if tqdm is not None:
        return tqdm(total=total, desc=desc, unit='项')
    return SimpleProgressBar(total=total, desc=desc)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description='仅统计报告中的 reportTitle 内容')
    parser.add_argument('--config', type=Path, default=CONFIG_PATH, help='路径配置文件，默认 configs/path.yaml')
    parser.add_argument(
        '--report-csv',
        type=Path,
        default=None,
        help='可选：覆盖配置中的 valid_dicts_report_csv 路径（未配置时默认 dataset_base_root/valid_dicts_report.csv）',
    )
    return parser.parse_args()


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

        match = re.match(r'^\s{2,}([A-Za-z0-9_]+)\s*:\s*(.+?)\s*$', line)
        if not match:
            continue
        key, value = match.group(1), match.group(2)
        result[key] = value.strip().strip('"').strip("'")

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
    report_csv_config = paths_payload.get('valid_dicts_report_csv')
    report_csv_path = (
        report_csv_override.expanduser().resolve()
        if report_csv_override is not None
        else resolve_path(report_csv_config, config_path.parent) if report_csv_config else (dataset_base_root / 'valid_dicts_report.csv').resolve()
    )
    return PathConfig(dataset_base_root=dataset_base_root, report_csv_path=report_csv_path)


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


IMAGE_EXTENSIONS = {'.jpg', '.jpeg', '.png', '.bmp', '.gif', '.tif', '.tiff', '.webp', '.dcm'}


def count_images_for_exam(exam_dir: Path) -> int:
    img_dir = exam_dir / 'img'
    if not img_dir.is_dir():
        return 0
    return sum(1 for path in img_dir.iterdir() if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS)


def load_report_title_image_counts(rows: list[dict[str, str]], dataset_base_root: Path) -> Counter[str]:
    image_counts: Counter[str] = Counter()
    progress = build_progress(total=len(rows), desc='统计各 reportTitle 的图像总数')
    try:
        for row in rows:
            title = str(row.get('reportTitle', '')).strip()
            if not title:
                progress.update(1)
                continue

            exam_dir_raw = str(row.get('exam_dir', '')).strip()
            exam_dir = Path(exam_dir_raw).expanduser()
            if not exam_dir.is_absolute():
                exam_dir = (dataset_base_root / exam_dir).resolve()
            image_counts[title] += count_images_for_exam(exam_dir)
            progress.update(1)
    finally:
        progress.close()
    return image_counts


def group_title_counts_by_digestive_part(title_counts: Counter[str]) -> dict[str, Counter[str]]:
    grouped: dict[str, Counter[str]] = {'胃': Counter(), '肠': Counter()}
    for title, count in title_counts.items():
        if '胃' in title:
            grouped['胃'][title] = count
        elif '肠' in title:
            grouped['肠'][title] = count
    return grouped


def print_report_title_counts_by_group(grouped_counts: dict[str, Counter[str]], image_counts: Counter[str]) -> None:
    for group_name in ('胃', '肠'):
        group_counter = grouped_counts.get(group_name, Counter())
        sorted_items = sorted(group_counter.items(), key=lambda item: (-item[1], item[0]))
        print(f'\n【{group_name}】共 {len(sorted_items)} 种 reportTitle（按报告数量降序）')
        if not sorted_items:
            print('无匹配记录')
            continue
        for idx, (title, count) in enumerate(sorted_items, start=1):
            print(f'{idx}. {title}: 报告 {count} 条，图像 {image_counts.get(title, 0)} 张')


def main() -> None:
    args = parse_args()
    config = build_path_config(args.config, args.report_csv)

    rows = load_report_rows(config.report_csv_path)
    print(f'报告记录总数：{len(rows)}')
    title_counts = load_report_title_counts(rows)
    image_counts = load_report_title_image_counts(rows, config.dataset_base_root)
    grouped_counts = group_title_counts_by_digestive_part(title_counts)
    print_report_title_counts_by_group(grouped_counts, image_counts)


if __name__ == '__main__':
    main()
