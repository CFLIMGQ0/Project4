from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from shutil import get_terminal_size
from typing import Iterable

CSV_PATH_DEFAULT = Path('/home/Lim/outputs/project4/cache_solve_conflicted_pdfs/valid_dicts_pdf_round3.csv')
JSONL_NAME_DEFAULT = 'solve_conflicted_pdfs_round3.jsonl'

try:
    from tqdm import tqdm  # type: ignore
except ImportError:
    tqdm = None


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
        return tqdm(total=total, desc=desc, unit='条')
    return SimpleProgressBar(total=total, desc=desc)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description='按检查目录输出 suggest/watch 冲突内容')
    parser.add_argument('--csv-path', type=Path, default=CSV_PATH_DEFAULT, help='第三轮汇总 CSV 路径')
    parser.add_argument(
        '--jsonl-path',
        type=Path,
        default=None,
        help='第三轮缓存 JSONL 路径（默认自动取 csv 同目录下的 solve_conflicted_pdfs_round3.jsonl）',
    )
    return parser.parse_args()


def _split_conflict_types(text: str) -> set[str]:
    return {part.strip() for part in text.split('|') if part.strip()}


def load_conflict_exam_dirs(csv_path: Path) -> tuple[set[str], set[str]]:
    suggest_dirs: set[str] = set()
    watch_dirs: set[str] = set()

    with csv_path.open('r', encoding='utf-8-sig', newline='') as f:
        reader = csv.DictReader(f)
        rows = list(reader)

    progress = build_progress(total=len(rows), desc='读取 CSV')
    try:
        for row in rows:
            exam_dir = str(row.get('exam_dir', '')).strip()
            conflict_types = _split_conflict_types(str(row.get('conflict_key_types', '')))
            if not exam_dir or not conflict_types:
                progress.update(1)
                continue

            if 'suggest' in conflict_types:
                suggest_dirs.add(exam_dir)
            if 'watch' in conflict_types:
                watch_dirs.add(exam_dir)
            progress.update(1)
    finally:
        progress.close()

    return suggest_dirs, watch_dirs


def load_conflict_values(jsonl_path: Path) -> dict[str, dict[str, list[str]]]:
    values_by_exam: dict[str, dict[str, list[str]]] = {}
    lines = jsonl_path.read_text(encoding='utf-8').splitlines()

    progress = build_progress(total=len(lines), desc='读取 JSONL')
    try:
        for line in lines:
            text = line.strip()
            if not text:
                progress.update(1)
                continue

            payload = json.loads(text)
            exam_dir = str(payload.get('exam_dir', '')).strip()
            if not exam_dir:
                progress.update(1)
                continue

            field_values = payload.get('field_values', {})
            suggest_values = sorted({str(v).strip() for v in field_values.get('suggest', []) if str(v).strip()})
            watch_values = sorted({str(v).strip() for v in field_values.get('watch', []) if str(v).strip()})
            values_by_exam[exam_dir] = {
                'suggest': suggest_values,
                'watch': watch_values,
            }
            progress.update(1)
    finally:
        progress.close()

    return values_by_exam


def _print_conflict_block(title: str, target_dirs: Iterable[str], values_by_exam: dict[str, dict[str, list[str]]], key: str) -> None:
    print(f'{title}：')
    sorted_dirs = sorted(set(target_dirs))
    if not sorted_dirs:
        print('（无）')
        print()
        return

    for idx, exam_dir in enumerate(sorted_dirs, start=1):
        print(f'{idx}.{exam_dir}')
        values = values_by_exam.get(exam_dir, {}).get(key, [])
        if not values:
            print(f'{key}1：未在 JSONL 中找到冲突内容')
            continue

        for val_idx, value in enumerate(values, start=1):
            print(f'{key}{val_idx}：{value}')
    print()


def main() -> None:
    args = parse_args()
    csv_path = args.csv_path.expanduser().resolve()
    if not csv_path.is_file():
        raise FileNotFoundError(f'CSV 文件不存在：{csv_path}')

    jsonl_path = args.jsonl_path.expanduser().resolve() if args.jsonl_path else (csv_path.parent / JSONL_NAME_DEFAULT).resolve()
    if not jsonl_path.is_file():
        raise FileNotFoundError(
            f'JSONL 文件不存在：{jsonl_path}\n'
            '说明：valid_dicts_pdf_round3.csv 只包含冲突类型，不包含具体冲突文本；需要搭配 solve_conflicted_pdfs_round3.jsonl。'
        )

    suggest_dirs, watch_dirs = load_conflict_exam_dirs(csv_path)
    values_by_exam = load_conflict_values(jsonl_path)

    _print_conflict_block('suggest冲突', suggest_dirs, values_by_exam, key='suggest')
    _print_conflict_block('watch冲突', watch_dirs, values_by_exam, key='watch')


if __name__ == '__main__':
    main()
