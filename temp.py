from __future__ import annotations

import argparse
import csv
import re
from dataclasses import dataclass
from pathlib import Path
from shutil import get_terminal_size


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
    if report_csv_override is not None:
        report_csv_path = report_csv_override.expanduser().resolve()
    else:
        report_csv_raw = paths_payload.get('valid_dicts_report_csv', 'valid_dicts_report.csv')
        report_csv_path = resolve_path(report_csv_raw, config_path.parent)
    return PathConfig(report_csv_path=report_csv_path)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description='修正 valid_dicts_report.csv 中胃镜/肠镜 hp 的空值与待确认值')
    parser.add_argument('--config', type=Path, default=Path('configs/path.yaml'), help='路径配置文件，默认 configs/path.yaml')
    parser.add_argument('--report-csv', type=Path, default=None, help='可选：覆盖 valid_dicts_report_csv')
    return parser.parse_args()


def normalize_text(value: str) -> str:
    return value.replace('\u3000', ' ').strip()


def classify_organ(report_title: str) -> str | None:
    title = normalize_text(report_title)
    has_stomach = '胃' in title and '肠' not in title
    has_intestine = '肠' in title
    if has_stomach:
        return '胃镜'
    if has_intestine:
        return '肠镜'
    return None


def should_set_unchecked(organ: str | None, hp_value: str) -> bool:
    hp_norm = normalize_text(hp_value)
    is_empty = hp_norm == ''
    if organ == '胃镜':
        return is_empty or hp_norm == '待确认'
    if organ == '肠镜':
        return is_empty
    return False


def load_rows(report_csv_path: Path) -> tuple[list[dict[str, str]], list[str]]:
    if not report_csv_path.is_file():
        raise FileNotFoundError(f'未找到报告汇总文件：{report_csv_path}')

    with report_csv_path.open('r', encoding='utf-8-sig', newline='') as f:
        reader = csv.DictReader(f)
        rows = list(reader)
        fieldnames = list(reader.fieldnames or [])

    required = {'reportTitle', 'hp'}
    missing = sorted(required - set(fieldnames))
    if missing:
        raise KeyError(f'CSV 缺少字段：{"、".join(missing)}')

    return rows, fieldnames


def rewrite_hp(rows: list[dict[str, str]]) -> tuple[list[dict[str, str]], dict[str, int]]:
    stats = {
        '胃镜_待确认改未检': 0,
        '胃镜_空值改未检': 0,
        '肠镜_空值改未检': 0,
        '总修改数': 0,
    }

    progress = SimpleProgressBar(total=len(rows), desc='重写 hp 字段')
    try:
        for row in rows:
            report_title = str(row.get('reportTitle', ''))
            hp_raw = str(row.get('hp', ''))
            organ = classify_organ(report_title)

            if should_set_unchecked(organ, hp_raw):
                hp_norm = normalize_text(hp_raw)
                if organ == '胃镜' and hp_norm == '待确认':
                    stats['胃镜_待确认改未检'] += 1
                elif organ == '胃镜' and hp_norm == '':
                    stats['胃镜_空值改未检'] += 1
                elif organ == '肠镜' and hp_norm == '':
                    stats['肠镜_空值改未检'] += 1

                row['hp'] = '未检'
                stats['总修改数'] += 1

            progress.update(1)
    finally:
        progress.close()

    return rows, stats


def write_rows(report_csv_path: Path, rows: list[dict[str, str]], fieldnames: list[str]) -> None:
    tmp_path = report_csv_path.with_suffix(report_csv_path.suffix + '.tmp')
    with tmp_path.open('w', encoding='utf-8-sig', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    tmp_path.replace(report_csv_path)


def main() -> None:
    args = parse_args()
    cfg = build_path_config(args.config, args.report_csv)

    rows, fieldnames = load_rows(cfg.report_csv_path)
    print(f'读取记录数：{len(rows)}')
    print(f'目标文件：{cfg.report_csv_path}')

    new_rows, stats = rewrite_hp(rows)

    write_rows(cfg.report_csv_path, new_rows, fieldnames)
    print('已完成覆盖写回。')
    print('修改统计：')
    for k, v in stats.items():
        print(f'- {k}：{v}')


if __name__ == '__main__':
    main()
