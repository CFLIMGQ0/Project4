from __future__ import annotations

import argparse
import csv
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

SUPPORTED_SUFFIXES = {'.json', '.jsonl', '.csv'}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description='递归检查目录内 latest_archive_time 的所有取值。')
    parser.add_argument(
        '--target-dir',
        type=Path,
        default=Path('/home/Lim/datasets/project4/main_data/ZS17239199/ZS0044501964'),
        help='待检查目录（默认使用本次排查路径）',
    )
    parser.add_argument(
        '--show-files',
        action='store_true',
        help='输出每个取值出现在哪些文件中',
    )
    return parser.parse_args()


def normalize_value(value: Any) -> str:
    if value is None:
        return ''
    return str(value).strip()


def extract_from_json_obj(obj: Any, values: list[str]) -> None:
    if isinstance(obj, dict):
        for key, val in obj.items():
            if key == 'latest_archive_time':
                values.append(normalize_value(val))
            extract_from_json_obj(val, values)
    elif isinstance(obj, list):
        for item in obj:
            extract_from_json_obj(item, values)


def scan_json_file(path: Path) -> list[str]:
    try:
        data = json.loads(path.read_text(encoding='utf-8'))
    except UnicodeDecodeError:
        data = json.loads(path.read_text(encoding='utf-8-sig'))
    values: list[str] = []
    extract_from_json_obj(data, values)
    return values


def scan_jsonl_file(path: Path) -> list[str]:
    values: list[str] = []
    with path.open('r', encoding='utf-8') as file:
        for line_no, line in enumerate(file, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f'JSONL 解析失败（{path}:{line_no}）: {exc}') from exc
            extract_from_json_obj(record, values)
    return values


def scan_csv_file(path: Path) -> list[str]:
    values: list[str] = []
    with path.open('r', encoding='utf-8-sig', newline='') as file:
        reader = csv.DictReader(file)
        if not reader.fieldnames or 'latest_archive_time' not in reader.fieldnames:
            return values
        for row in reader:
            values.append(normalize_value(row.get('latest_archive_time')))
    return values


def scan_file(path: Path) -> list[str]:
    suffix = path.suffix.lower()
    if suffix == '.json':
        return scan_json_file(path)
    if suffix == '.jsonl':
        return scan_jsonl_file(path)
    if suffix == '.csv':
        return scan_csv_file(path)
    return []


def main() -> None:
    args = parse_args()
    target_dir = args.target_dir.expanduser()

    if not target_dir.exists():
        print(f'目录不存在：{target_dir}')
        return
    if not target_dir.is_dir():
        print(f'目标不是目录：{target_dir}')
        return

    candidate_files = [
        path
        for path in sorted(target_dir.rglob('*'))
        if path.is_file() and path.suffix.lower() in SUPPORTED_SUFFIXES
    ]

    if not candidate_files:
        print(f'未找到可扫描文件（支持：{", ".join(sorted(SUPPORTED_SUFFIXES))}）。')
        return

    counter: Counter[str] = Counter()
    value_to_files: dict[str, set[str]] = defaultdict(set)
    scanned_files = 0
    found_files = 0

    for file_path in candidate_files:
        scanned_files += 1
        try:
            values = scan_file(file_path)
        except Exception as exc:
            print(f'跳过文件（解析失败）：{file_path}，原因：{exc}')
            continue

        if not values:
            continue

        found_files += 1
        rel_path = str(file_path.relative_to(target_dir))
        for val in values:
            counter[val] += 1
            value_to_files[val].add(rel_path)

    print(f'扫描目录：{target_dir}')
    print(f'扫描文件数：{scanned_files}')
    print(f'包含 latest_archive_time 的文件数：{found_files}')
    print(f'latest_archive_time 总计出现次数：{sum(counter.values())}')
    print(f'latest_archive_time 不同取值数：{len(counter)}')

    if not counter:
        print('未找到 latest_archive_time 字段。')
        return

    print('\n取值统计（按出现次数降序）：')
    for idx, (value, times) in enumerate(counter.most_common(), start=1):
        show_value = value if value else '<空字符串>'
        print(f'{idx}. {show_value} -> {times} 次')
        if args.show_files:
            files = sorted(value_to_files[value])
            print(f'   文件数：{len(files)}')
            for one_file in files:
                print(f'   - {one_file}')


if __name__ == '__main__':
    main()
