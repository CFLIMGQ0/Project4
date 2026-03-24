from __future__ import annotations

import argparse
import importlib.util
from collections import Counter
from pathlib import Path
from typing import Iterable

if importlib.util.find_spec('pypdf') is not None:
    from pypdf import PdfReader
else:
    PdfReader = None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description='检查指定目录下 pdf 子目录内所有 PDF 的 archiveTime 字段取值。')
    parser.add_argument(
        '--target-dir',
        type=Path,
        default=Path('/home/Lim/datasets/project4/main_data/ZS17239199/ZS0044501964'),
        help='检查目录（默认使用本次排查目录）',
    )
    parser.add_argument(
        '--show-missing',
        action='store_true',
        help='同时输出没有 archiveTime 字段的 PDF 文件',
    )
    return parser.parse_args()


def normalize_value(value: object) -> str:
    if value is None:
        return ''
    return str(value).strip()


def iter_pdf_dirs(target_dir: Path) -> Iterable[Path]:
    if target_dir.name.lower() == 'pdf' and target_dir.is_dir():
        yield target_dir
    for path in sorted(target_dir.rglob('*')):
        if path.is_dir() and path.name.lower() == 'pdf':
            yield path


def collect_pdf_files(target_dir: Path) -> list[Path]:
    pdf_files: list[Path] = []
    seen: set[Path] = set()
    for pdf_dir in iter_pdf_dirs(target_dir):
        for file_path in sorted(pdf_dir.rglob('*.pdf')):
            resolved = file_path.resolve()
            if file_path.is_file() and resolved not in seen:
                seen.add(resolved)
                pdf_files.append(file_path)
    return pdf_files


def extract_archive_time(pdf_path: Path) -> str:
    if PdfReader is None:
        raise RuntimeError('当前环境未安装 pypdf，请先安装后再运行。')

    reader = PdfReader(str(pdf_path))
    fields = reader.get_fields() or {}
    if not fields:
        return ''

    for key, item in fields.items():
        if str(key).strip().lower() != 'archivetime':
            continue

        if hasattr(item, 'get'):
            return normalize_value(item.get('/V'))

        return normalize_value(item)
    return ''


def main() -> None:
    args = parse_args()
    target_dir = args.target_dir.expanduser()

    if not target_dir.exists():
        print(f'目录不存在：{target_dir}')
        return
    if not target_dir.is_dir():
        print(f'目标不是目录：{target_dir}')
        return

    pdf_files = collect_pdf_files(target_dir)
    if not pdf_files:
        print('未找到任何 pdf 目录或 PDF 文件。')
        return

    value_counter: Counter[str] = Counter()
    missing_files: list[Path] = []

    print(f'扫描目录：{target_dir}')
    print(f'发现 PDF 文件数：{len(pdf_files)}')
    print('\n每个 PDF 的 archiveTime：')

    for file_path in pdf_files:
        try:
            value = extract_archive_time(file_path)
        except Exception as exc:
            print(f'- {file_path} -> 读取失败：{exc}')
            continue

        if value:
            value_counter[value] += 1
            print(f'- {file_path} -> {value}')
        else:
            missing_files.append(file_path)
            print(f'- {file_path} -> <未找到 archiveTime>')

    print('\narchiveTime 取值统计：')
    if value_counter:
        for idx, (value, count) in enumerate(value_counter.most_common(), start=1):
            print(f'{idx}. {value} -> {count} 个 PDF')
    else:
        print('未从任何 PDF 解析到 archiveTime。')

    print(f'\n存在 archiveTime 的 PDF 数：{sum(value_counter.values())}')
    print(f'未找到 archiveTime 的 PDF 数：{len(missing_files)}')

    if args.show_missing and missing_files:
        print('\n未找到 archiveTime 的 PDF 列表：')
        for file_path in missing_files:
            print(f'- {file_path}')


if __name__ == '__main__':
    main()
