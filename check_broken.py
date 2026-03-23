from __future__ import annotations

import argparse
import binascii
import csv
import struct
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

try:
    import yaml  # type: ignore
except ImportError:
    yaml = None

try:
    from PIL import Image, UnidentifiedImageError  # type: ignore
except ImportError:
    Image = None

    class UnidentifiedImageError(Exception):
        """Pillow 不可用时的占位异常。"""

CONFIG_PATH = Path(__file__).resolve().parent / 'configs' / 'path.yaml'
DEFAULT_OUTPUT_NAME = 'broken_files.csv'
EOF_SCAN_BYTES = 4096
IMAGE_EXTENSIONS = {
    '.jpg',
    '.jpeg',
    '.png',
    '.bmp',
    '.gif',
    '.tif',
    '.tiff',
    '.webp',
}


@dataclass
class PathConfig:
    dataset_root: Path
    output_dir: Path


@dataclass
class BrokenFileRecord:
    file_path: Path
    file_type: str
    reason: str
    size_bytes: int
    check_method: str


class FileIntegrityError(Exception):
    """文件完整性检查失败。"""


class PdfIntegrityError(FileIntegrityError):
    """PDF 完整性检查失败。"""


class ImageIntegrityError(FileIntegrityError):
    """图片完整性检查失败。"""


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description='检查所有患者目录中的 PDF 与图片是否完整，并导出损坏文件 CSV')
    parser.add_argument('--config', type=Path, default=CONFIG_PATH, help='路径配置文件，默认使用 configs/path.yaml')
    parser.add_argument('--input-dir', type=Path, default=None, help='可选：覆盖配置中的 dataset_root')
    parser.add_argument('--output-dir', type=Path, default=None, help='可选：覆盖配置中的 output_dir')
    parser.add_argument('--output-name', type=str, default=DEFAULT_OUTPUT_NAME, help='输出 CSV 文件名，默认 broken_files.csv')
    parser.add_argument('--max-files', type=int, default=None, help='可选：仅检查前 N 个文件，用于快速抽查')
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


def iter_target_files(dataset_root: Path) -> Iterable[Path]:
    return sorted(
        path for path in dataset_root.rglob('*')
        if path.is_file() and (path.suffix.lower() == '.pdf' or path.suffix.lower() in IMAGE_EXTENSIONS)
    )


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


def _check_png_bytes(data: bytes) -> str:
    signature = b'\x89PNG\r\n\x1a\n'
    if not data.startswith(signature):
        raise ImageIntegrityError('PNG 文件头不合法')

    offset = len(signature)
    seen_iend = False
    while offset + 8 <= len(data):
        chunk_length = struct.unpack('>I', data[offset:offset + 4])[0]
        chunk_type = data[offset + 4:offset + 8]
        chunk_data_start = offset + 8
        chunk_data_end = chunk_data_start + chunk_length
        chunk_crc_end = chunk_data_end + 4
        if chunk_crc_end > len(data):
            raise ImageIntegrityError('PNG 数据块长度超出文件范围')

        chunk_data = data[chunk_data_start:chunk_data_end]
        expected_crc = struct.unpack('>I', data[chunk_data_end:chunk_crc_end])[0]
        actual_crc = binascii.crc32(chunk_type)
        actual_crc = binascii.crc32(chunk_data, actual_crc) & 0xffffffff
        if actual_crc != expected_crc:
            raise ImageIntegrityError(f'PNG 数据块 CRC 校验失败：{chunk_type.decode("ascii", errors="ignore") or "unknown"}')

        offset = chunk_crc_end
        if chunk_type == b'IEND':
            seen_iend = True
            break

    if not seen_iend:
        raise ImageIntegrityError('PNG 文件缺少 IEND 结束块')
    if offset != len(data):
        extra = data[offset:]
        if extra.strip(b'\x00\r\n\t '):
            raise ImageIntegrityError('PNG 文件在 IEND 后仍包含异常数据')
    return 'signature+chunks+crc'


def _read_jpeg_segment_length(data: bytes, offset: int) -> tuple[int, int]:
    if offset + 2 > len(data):
        raise ImageIntegrityError('JPEG 段长度字段不完整')
    segment_length = struct.unpack('>H', data[offset:offset + 2])[0]
    if segment_length < 2:
        raise ImageIntegrityError('JPEG 段长度非法')
    return segment_length, offset + 2


def _check_jpeg_bytes(data: bytes) -> str:
    if len(data) < 4 or data[:2] != b'\xff\xd8':
        raise ImageIntegrityError('JPEG 文件头不合法')
    if data[-2:] != b'\xff\xd9':
        raise ImageIntegrityError('JPEG 文件尾缺少 EOI 标识')

    offset = 2
    seen_sos = False
    while offset + 1 < len(data):
        if data[offset] != 0xFF:
            if seen_sos:
                break
            raise ImageIntegrityError('JPEG 标记前缺少 0xFF 前缀')

        while offset < len(data) and data[offset] == 0xFF:
            offset += 1
        if offset >= len(data):
            raise ImageIntegrityError('JPEG 标记异常结束')

        marker = data[offset]
        offset += 1

        if marker == 0xD9:
            return 'signature+segments+eoi'

        if marker in {0x01} or 0xD0 <= marker <= 0xD8:
            continue

        segment_length, _ = _read_jpeg_segment_length(data, offset)
        segment_end = offset + segment_length
        if segment_end > len(data):
            raise ImageIntegrityError('JPEG 段长度超出文件范围')

        if marker == 0xDA:
            seen_sos = True
            break

        offset = segment_end

    if not seen_sos:
        raise ImageIntegrityError('JPEG 缺少 SOS 扫描段')
    return 'signature+sos+eoi'


def _check_gif_bytes(data: bytes) -> str:
    if len(data) < 14 or data[:6] not in {b'GIF87a', b'GIF89a'}:
        raise ImageIntegrityError('GIF 文件头不合法')
    if data[-1:] != b'\x3b':
        raise ImageIntegrityError('GIF 文件尾缺少结束符')
    return 'signature+trailer'


def _check_bmp_bytes(data: bytes) -> str:
    if len(data) < 26 or data[:2] != b'BM':
        raise ImageIntegrityError('BMP 文件头不合法')
    declared_size = struct.unpack('<I', data[2:6])[0]
    if declared_size != len(data):
        raise ImageIntegrityError('BMP 文件大小与头信息不一致')
    pixel_offset = struct.unpack('<I', data[10:14])[0]
    dib_header_size = struct.unpack('<I', data[14:18])[0]
    if pixel_offset >= len(data) or dib_header_size < 12:
        raise ImageIntegrityError('BMP 头信息非法')
    return 'signature+size+header'


def _check_tiff_bytes(data: bytes) -> str:
    if len(data) < 8:
        raise ImageIntegrityError('TIFF 文件过小')
    byte_order = data[:2]
    if byte_order == b'II':
        endian = '<'
    elif byte_order == b'MM':
        endian = '>'
    else:
        raise ImageIntegrityError('TIFF 字节序标识不合法')
    magic = struct.unpack(endian + 'H', data[2:4])[0]
    if magic != 42:
        raise ImageIntegrityError('TIFF 魔数不正确')
    first_ifd_offset = struct.unpack(endian + 'I', data[4:8])[0]
    if first_ifd_offset >= len(data):
        raise ImageIntegrityError('TIFF 首个 IFD 偏移量超出文件范围')
    return 'signature+ifd-offset'


def _check_webp_bytes(data: bytes) -> str:
    if len(data) < 16:
        raise ImageIntegrityError('WEBP 文件过小')
    if data[:4] != b'RIFF' or data[8:12] != b'WEBP':
        raise ImageIntegrityError('WEBP RIFF 头不合法')
    declared_size = struct.unpack('<I', data[4:8])[0] + 8
    if declared_size != len(data):
        raise ImageIntegrityError('WEBP 文件大小与 RIFF 头不一致')
    return 'riff+size'


def check_image_integrity(image_path: Path) -> tuple[str, int]:
    try:
        file_size = image_path.stat().st_size
    except OSError as exc:
        raise ImageIntegrityError(f'无法读取文件信息：{exc}') from exc

    if file_size <= 0:
        raise ImageIntegrityError('文件大小为 0 字节')

    try:
        data = image_path.read_bytes()
    except OSError as exc:
        raise ImageIntegrityError(f'读取文件失败：{exc}') from exc

    if Image is not None:
        try:
            with Image.open(image_path) as image:
                image.verify()
            with Image.open(image_path) as image:
                image.load()
            return 'pillow-verify+load', file_size
        except (UnidentifiedImageError, OSError, ValueError):
            pass

    suffix = image_path.suffix.lower()
    if suffix in {'.png'}:
        return _check_png_bytes(data), file_size
    if suffix in {'.jpg', '.jpeg'}:
        return _check_jpeg_bytes(data), file_size
    if suffix == '.gif':
        return _check_gif_bytes(data), file_size
    if suffix == '.bmp':
        return _check_bmp_bytes(data), file_size
    if suffix in {'.tif', '.tiff'}:
        return _check_tiff_bytes(data), file_size
    if suffix == '.webp':
        return _check_webp_bytes(data), file_size

    raise ImageIntegrityError(f'暂不支持的图片类型：{suffix or "无扩展名"}')


def check_file_integrity(file_path: Path) -> tuple[str, str, int]:
    suffix = file_path.suffix.lower()
    if suffix == '.pdf':
        check_method, file_size = check_pdf_integrity(file_path)
        return 'pdf', check_method, file_size
    if suffix in IMAGE_EXTENSIONS:
        check_method, file_size = check_image_integrity(file_path)
        return 'image', check_method, file_size
    raise FileIntegrityError(f'暂不支持的文件类型：{suffix or "无扩展名"}')


def infer_check_method(file_path: Path) -> str:
    suffix = file_path.suffix.lower()
    if suffix == '.pdf':
        return 'header+eof+startxref'
    if Image is not None:
        return 'pillow-verify+load'
    if suffix == '.png':
        return 'signature+chunks+crc'
    if suffix in {'.jpg', '.jpeg'}:
        return 'signature+sos+eoi'
    if suffix == '.gif':
        return 'signature+trailer'
    if suffix == '.bmp':
        return 'signature+size+header'
    if suffix in {'.tif', '.tiff'}:
        return 'signature+ifd-offset'
    if suffix == '.webp':
        return 'riff+size'
    return 'unknown'


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


def export_broken_records(output_csv: Path, records: list[BrokenFileRecord], dataset_root: Path) -> None:
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    with output_csv.open('w', encoding='utf-8-sig', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(['file_path', 'relative_path', 'file_type', 'reason', 'size_bytes', 'check_method'])
        for record in records:
            try:
                relative_path = record.file_path.relative_to(dataset_root)
            except ValueError:
                relative_path = record.file_path
            writer.writerow([
                str(record.file_path),
                str(relative_path),
                record.file_type,
                record.reason,
                record.size_bytes,
                record.check_method,
            ])


def prompt_delete_broken_files(records: list[BrokenFileRecord]) -> bool:
    if not records:
        return False

    print('发现以下损坏文件：')
    for index, record in enumerate(records, start=1):
        print(f'{index}. [{record.file_type}] {record.file_path}（原因：{record.reason}）')

    while True:
        answer = input('是否删除以上损坏文件？请输入 y/yes 确认，其他输入将保留文件：').strip().lower()
        if answer in {'y', 'yes'}:
            return True
        if answer in {'', 'n', 'no'}:
            return False
        print('输入无效，请输入 y/yes 或 n/no。')


def delete_broken_files(records: list[BrokenFileRecord]) -> tuple[int, list[tuple[Path, str]]]:
    deleted_count = 0
    failed_records: list[tuple[Path, str]] = []

    for record in records:
        try:
            record.file_path.unlink()
            deleted_count += 1
        except OSError as exc:
            failed_records.append((record.file_path, str(exc)))

    return deleted_count, failed_records


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

    target_files = list(iter_target_files(dataset_root))
    if args.max_files is not None and args.max_files > 0:
        target_files = target_files[:args.max_files]

    if not target_files:
        output_dir.mkdir(parents=True, exist_ok=True)
        export_broken_records(output_csv, [], dataset_root)
        print(f'未找到 PDF 或图片文件，已输出空结果：{output_csv}')
        return

    broken_records: list[BrokenFileRecord] = []
    checked_count = 0

    print(f'开始检查文件，共 {len(target_files)} 个（PDF + 图片）。')
    write_progress(0, len(target_files))

    for file_path in target_files:
        try:
            file_type, check_method, file_size = check_file_integrity(file_path)
        except FileIntegrityError as exc:
            size_bytes = file_path.stat().st_size if file_path.exists() else -1
            broken_records.append(
                BrokenFileRecord(
                    file_path=file_path,
                    file_type='pdf' if file_path.suffix.lower() == '.pdf' else 'image',
                    reason=str(exc),
                    size_bytes=size_bytes,
                    check_method=infer_check_method(file_path),
                )
            )
        else:
            _ = file_size
            _ = check_method
            _ = file_type

        checked_count += 1
        write_progress(checked_count, len(target_files))

    export_broken_records(output_csv, broken_records, dataset_root)

    total_pdf_count = sum(1 for path in target_files if path.suffix.lower() == '.pdf')
    total_image_count = len(target_files) - total_pdf_count
    broken_pdf_count = sum(1 for record in broken_records if record.file_type == 'pdf')
    broken_image_count = sum(1 for record in broken_records if record.file_type == 'image')

    print('检查完成。')
    print(f'文件总数：{len(target_files)}')
    print(f'PDF 数量：{total_pdf_count}，其中损坏 {broken_pdf_count} 个')
    print(f'图片数量：{total_image_count}，其中损坏 {broken_image_count} 个')
    print(f'损坏总数：{len(broken_records)}')
    print(f'结果 CSV：{output_csv}')

    if not broken_records:
        print('未发现损坏文件，无需清理。')
        return

    if not prompt_delete_broken_files(broken_records):
        print('已保留所有损坏文件。')
        return

    deleted_count, failed_records = delete_broken_files(broken_records)
    print(f'已删除 {deleted_count} 个损坏文件。')
    if failed_records:
        print('以下文件删除失败：')
        for failed_path, reason in failed_records:
            print(f'- {failed_path}（原因：{reason}）')


if __name__ == '__main__':
    main()
