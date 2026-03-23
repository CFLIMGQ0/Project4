from __future__ import annotations

import argparse
import csv
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

try:
    import yaml  # type: ignore
except ImportError:
    yaml = None

CONFIG_PATH = Path(__file__).resolve().parent / 'configs' / 'path.yaml'
DEFAULT_OUTPUT_NAME = 'broken_pdf_files.csv'
EOF_SCAN_BYTES = 4096


@dataclass
class PathConfig:
    dataset_root: Path
    output_dir: Path


@dataclass
class BrokenPdfRecord:
    pdf_path: Path
    reason: str
    size_bytes: int
    check_method: str


class PdfIntegrityError(Exception):
    """PDF 完整性检查失败。"""


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description='检查所有患者目录中的 PDF 是否完整，并导出损坏文件 CSV')
    parser.add_argument('--config', type=Path, default=CONFIG_PATH, help='路径配置文件，默认使用 configs/path.yaml')
    parser.add_argument('--input-dir', type=Path, default=None, help='可选：覆盖配置中的 dataset_root')
    parser.add_argument('--output-dir', type=Path, default=None, help='可选：覆盖配置中的 output_dir')
    parser.add_argument('--output-name', type=str, default=DEFAULT_OUTPUT_NAME, help='输出 CSV 文件名，默认 broken_pdf_files.csv')
    parser.add_argument('--max-files', type=int, default=None, help='可选：仅检查前 N 个 PDF，用于快速抽查')
    return parser.parse_args()


def load_yaml_config(config_path: Path) -> dict[str, Any]:
    if not config_path.exists():
        raise FileNotFoundError(f'路径配置文件不存在：{config_path}')

    if yaml is not None:
        payload = yaml.safe_load(config_path.read_text(encoding='utf-8'))
        if not isinstance(payload, dict):
            raise ValueError(f'路径配置文件格式错误：{config_path}')
        return payload

    lines = config_path.read_text(encoding='utf-8').splitlines()
    payload: dict[str, Any] = {}
    current_section: str | None = None
    for raw_line in lines:
        if not raw_line.strip() or raw_line.lstrip().startswith('#'):
            continue

        indent = len(raw_line) - len(raw_line.lstrip(' '))
        line = raw_line.strip()
        if line.endswith(':'):
            current_section = line[:-1]
            payload[current_section] = {}
            continue

        key, sep, value = line.partition(':')
        if not sep:
            raise ValueError(f'无法解析路径配置行：{raw_line}')

        cleaned_value = value.strip().strip('"').strip("'")
        if indent == 0:
            payload[key.strip()] = cleaned_value
            current_section = None
            continue

        if current_section is None:
            raise ValueError(f'发现未归属分组的缩进行：{raw_line}')
        payload[current_section][key.strip()] = cleaned_value
    return payload


def build_path_config(config_path: Path, input_dir: Path | None, output_dir: Path | None) -> PathConfig:
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

    dataset_root = input_dir.expanduser().resolve() if input_dir is not None else resolve_path(str(paths_payload['dataset_root']))
    final_output_dir = output_dir.expanduser().resolve() if output_dir is not None else resolve_path(str(paths_payload['output_dir']))
    return PathConfig(dataset_root=dataset_root, output_dir=final_output_dir)


def iter_pdf_files(dataset_root: Path) -> Iterable[Path]:
    return sorted(path for path in dataset_root.rglob('*.pdf') if path.is_file())


def check_pdf_integrity(pdf_path: Path) -> tuple[str, int]:
    try:
        file_size = pdf_path.stat().st_size
    except OSError as exc:
        raise PdfIntegrityError(f'无法读取文件信息：{exc}') from exc

    if file_size <= 0:
        raise PdfIntegrityError('文件大小为 0 字节')

    try:
        with pdf_path.open('rb') as file:
            head = file.read(8)
            if not head.startswith(b'%PDF-'):
                raise PdfIntegrityError('文件头缺少 %PDF- 标识')

            tail_size = min(file_size, EOF_SCAN_BYTES)
            file.seek(-tail_size, 2)
            tail = file.read(tail_size)
    except OSError as exc:
        raise PdfIntegrityError(f'读取文件失败：{exc}') from exc

    if b'%%EOF' not in tail:
        raise PdfIntegrityError('文件尾缺少 %%EOF 标识')

    startxref_index = tail.rfind(b'startxref')
    if startxref_index == -1:
        raise PdfIntegrityError('文件尾缺少 startxref 标识')

    startxref_block = tail[startxref_index:].splitlines()
    if len(startxref_block) < 2:
        raise PdfIntegrityError('startxref 后缺少偏移量')

    offset_line = startxref_block[1].strip()
    if not offset_line.isdigit():
        raise PdfIntegrityError('startxref 偏移量不是数字')

    xref_offset = int(offset_line)
    if xref_offset < 0 or xref_offset >= file_size:
        raise PdfIntegrityError('startxref 偏移量超出文件范围')

    return 'header+eof+startxref', file_size


def render_progress(current: int, total: int, width: int = 40) -> str:
    if total <= 0:
        return '[{}] 100.0% (0/0)'.format('#' * width)
    filled = int(width * current / total)
    bar = '#' * filled + '-' * (width - filled)
    percent = current / total * 100
    return f'[{bar}] {percent:5.1f}% ({current}/{total})'


def write_progress(current: int, total: int) -> None:
    message = '\r总进度：' + render_progress(current, total)
    end = '\n' if current >= total else ''
    sys.stdout.write(message + end)
    sys.stdout.flush()


def export_broken_records(output_csv: Path, records: list[BrokenPdfRecord], dataset_root: Path) -> None:
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    with output_csv.open('w', encoding='utf-8-sig', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(['pdf_path', 'relative_path', 'reason', 'size_bytes', 'check_method'])
        for record in records:
            try:
                relative_path = record.pdf_path.relative_to(dataset_root)
            except ValueError:
                relative_path = record.pdf_path
            writer.writerow([
                str(record.pdf_path),
                str(relative_path),
                record.reason,
                record.size_bytes,
                record.check_method,
            ])


def main() -> None:
    args = parse_args()
    path_config = build_path_config(args.config, args.input_dir, args.output_dir)
    dataset_root = path_config.dataset_root
    output_dir = path_config.output_dir
    output_csv = output_dir / args.output_name

    if not dataset_root.exists():
        print(f'输入目录不存在：{dataset_root}')
        return
    if not dataset_root.is_dir():
        print(f'输入路径不是目录：{dataset_root}')
        return

    pdf_files = list(iter_pdf_files(dataset_root))
    if args.max_files is not None and args.max_files > 0:
        pdf_files = pdf_files[:args.max_files]

    if not pdf_files:
        output_dir.mkdir(parents=True, exist_ok=True)
        export_broken_records(output_csv, [], dataset_root)
        print(f'未找到 PDF 文件，已输出空结果：{output_csv}')
        return

    broken_records: list[BrokenPdfRecord] = []
    checked_count = 0

    print(f'开始检查 PDF，共 {len(pdf_files)} 个文件。')
    write_progress(0, len(pdf_files))

    for pdf_path in pdf_files:
        try:
            check_method, file_size = check_pdf_integrity(pdf_path)
        except PdfIntegrityError as exc:
            size_bytes = pdf_path.stat().st_size if pdf_path.exists() else -1
            broken_records.append(
                BrokenPdfRecord(
                    pdf_path=pdf_path,
                    reason=str(exc),
                    size_bytes=size_bytes,
                    check_method='header+eof+startxref',
                )
            )
        else:
            _ = file_size
            _ = check_method

        checked_count += 1
        write_progress(checked_count, len(pdf_files))

    export_broken_records(output_csv, broken_records, dataset_root)

    print('检查完成。')
    print(f'PDF 总数：{len(pdf_files)}')
    print(f'损坏数量：{len(broken_records)}')
    print(f'结果 CSV：{output_csv}')


if __name__ == '__main__':
    main()
