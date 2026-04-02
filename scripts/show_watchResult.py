from __future__ import annotations

import argparse
import csv
import re
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from shutil import get_terminal_size

SPLIT_PATTERN = re.compile(r'[，,；;。、“”\n\r、]+')


@dataclass
class PathConfig:
    dataset_base_root: Path
    report_csv_path: Path
    output_dir: Path


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
    output_dir_raw = paths_payload.get('output_dir', str(dataset_base_root))
    output_dir = resolve_path(output_dir_raw, config_path.parent)
    return PathConfig(dataset_base_root=dataset_base_root, report_csv_path=report_csv_path, output_dir=output_dir)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description='统计胃/肠数据中的 watchResult 类型及出现次数（按中文标点细分）')
    parser.add_argument('--config', type=Path, default=Path('configs/path.yaml'), help='路径配置文件，默认 configs/path.yaml')
    parser.add_argument('--report-csv', type=Path, default=None, help='可选：覆盖 valid_dicts_report_csv')
    parser.add_argument('--output-csv', type=Path, default=None, help='可选：覆盖统计汇总 CSV 输出路径')
    return parser.parse_args()


def load_rows(report_csv_path: Path) -> list[dict[str, str]]:
    if not report_csv_path.is_file():
        raise FileNotFoundError(f'未找到报告汇总文件：{report_csv_path}')

    with report_csv_path.open('r', encoding='utf-8-sig', newline='') as f:
        reader = csv.DictReader(f)
        rows = list(reader)
        fieldnames = set(reader.fieldnames or [])

    required = {'reportTitle', 'watchResult'}
    missing = sorted(required - fieldnames)
    if missing:
        raise KeyError(f'CSV 缺少字段：{"、".join(missing)}')
    return rows


def split_watch_result(value: str) -> list[str]:
    if not value:
        return []
    normalized = value.replace('\u3000', ' ').strip()
    parts = [p.strip() for p in SPLIT_PATTERN.split(normalized)]
    return [p for p in parts if p]


def classify_organ(report_title: str) -> str | None:
    has_stomach = '胃' in report_title and '肠' not in report_title
    has_intestine = '肠' in report_title
    if has_stomach:
        return '胃'
    if has_intestine:
        return '肠'
    return None


def count_watch_results(rows: list[dict[str, str]]) -> tuple[Counter[str], Counter[str]]:
    gastric_counter: Counter[str] = Counter()
    intestinal_counter: Counter[str] = Counter()

    progress = SimpleProgressBar(total=len(rows), desc='拆分并统计 watchResult')
    try:
        for row in rows:
            report_title = str(row.get('reportTitle', '')).strip()
            organ = classify_organ(report_title)
            if organ is None:
                progress.update(1)
                continue

            watch_result = str(row.get('watchResult', '')).strip()
            parts = split_watch_result(watch_result)
            if organ == '胃':
                gastric_counter.update(parts)
            else:
                intestinal_counter.update(parts)
            progress.update(1)
    finally:
        progress.close()

    return gastric_counter, intestinal_counter


def print_counter(title: str, counter: Counter[str]) -> None:
    print(f'\n==================== {title} ====================')
    print(f'类型总数：{len(counter)}')
    if not counter:
        print('无可统计结果。')
        return
    for idx, (item, count) in enumerate(counter.most_common(), start=1):
        print(f'{idx:>4}. {item}: {count}')


def dump_summary_csv(output_csv_path: Path, gastric_counter: Counter[str], intestinal_counter: Counter[str]) -> None:
    output_csv_path.parent.mkdir(parents=True, exist_ok=True)
    with output_csv_path.open('w', encoding='utf-8-sig', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=['organ', 'watchResult', 'count'])
        writer.writeheader()
        for item, count in gastric_counter.most_common():
            writer.writerow({'organ': '胃', 'watchResult': item, 'count': count})
        for item, count in intestinal_counter.most_common():
            writer.writerow({'organ': '肠', 'watchResult': item, 'count': count})


def main() -> None:
    args = parse_args()
    cfg = build_path_config(args.config, args.report_csv)
    rows = load_rows(cfg.report_csv_path)
    print(f'报告记录总数：{len(rows)}')
    print(f'统计文件：{cfg.report_csv_path}')

    gastric_counter, intestinal_counter = count_watch_results(rows)
    print_counter('胃镜 watchResult 细分结果（按次数降序）', gastric_counter)
    print_counter('肠镜 watchResult 细分结果（按次数降序）', intestinal_counter)
    output_csv_path = (args.output_csv.expanduser().resolve() if args.output_csv else cfg.output_dir / 'watch_result_summary.csv')
    dump_summary_csv(output_csv_path, gastric_counter, intestinal_counter)
    print(f'\n统计汇总 CSV 已保存：{output_csv_path}')


if __name__ == '__main__':
    main()
