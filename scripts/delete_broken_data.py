from __future__ import annotations

import argparse
import binascii
import csv
import json
import shutil
import struct
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Sequence

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


PROJECT_ROOT = Path(__file__).resolve().parent.parent
CONFIG_PATH = PROJECT_ROOT / 'configs' / 'path.yaml'
DEFAULT_OUTPUT_NAME = 'broken_files.csv'
DELETE_BROKEN_DATA_JSON_NAME = 'delete_broken_data.json'
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
    '.dcm',
}
BROKEN_FILE_SCAN_EXTENSIONS = IMAGE_EXTENSIONS - {'.dcm'}
PDF_EXTENSIONS = {'.pdf'}
PREVIEW_LIMIT = 20


@dataclass
class PathConfig:
    dataset_root: Path
    dataset_base_root: Path
    output_dir: Path


@dataclass
class BrokenFileRecord:
    file_path: Path
    file_type: str
    reason: str
    size_bytes: int
    check_method: str


@dataclass
class InvalidSubdirRecord:
    exam_dir: Path
    target_dir: Path
    target_type: str
    reason: str


@dataclass
class ManualDeleteCandidate:
    exam_relative_path: Path
    reason: str


@dataclass
class DatasetStats:
    patient_count: int = 0
    exam_count: int = 0
    image_count: int = 0
    report_count: int = 0


MANUAL_DELETE_CANDIDATES = [
    ManualDeleteCandidate(exam_relative_path=Path('ZS19332084') / 'ZS0046078433', reason='检查为图片数量不全且损坏'),
]


class FileIntegrityError(Exception):
    """文件完整性检查失败。"""


class PdfIntegrityError(FileIntegrityError):
    """PDF 完整性检查失败。"""


class ImageIntegrityError(FileIntegrityError):
    """图片完整性检查失败。"""


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description='统一执行损坏文件、空子目录、不完整检查目录、手动目录与空患者目录清理'
    )
    parser.add_argument('--config', type=Path, default=CONFIG_PATH, help='路径配置文件，默认使用 configs/path.yaml')
    parser.add_argument('--input-dir', type=Path, default=None, help='可选：覆盖配置中的 dataset_root')
    parser.add_argument('--output-dir', type=Path, default=None, help='可选：覆盖配置中的 output_dir')
    parser.add_argument('--output-name', type=str, default=DEFAULT_OUTPUT_NAME, help='损坏文件 CSV 文件名')
    parser.add_argument('--max-files', type=int, default=None, help='可选：仅检查前 N 个可校验文件，用于快速抽查')
    parser.add_argument('--yes', action='store_true', help='跳过交互确认，直接执行所有可删除步骤')
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
    dataset_base_root = resolve_path(str(paths_payload['dataset_base_root']))
    final_output_dir = output_dir.expanduser().resolve() if output_dir is not None else resolve_path(str(paths_payload['output_dir']))
    return PathConfig(dataset_root=dataset_root, dataset_base_root=dataset_base_root, output_dir=final_output_dir)


def render_progress(current: int, total: int, width: int = 40) -> str:
    if total <= 0:
        return '[{}] 100.0% (0/0)'.format('#' * width)
    filled = int(width * current / total)
    bar = '#' * filled + '-' * (width - filled)
    percent = current / total * 100
    return f'[{bar}] {percent:5.1f}% ({current}/{total})'


def write_progress(label: str, current: int, total: int) -> None:
    message = f'\r{label}：' + render_progress(current, total)
    end = '\n' if current >= total else ''
    sys.stdout.write(message + end)
    sys.stdout.flush()


def update_progress(label: str, current: int, total: int, last_percent: int) -> int:
    percent = 100 if total <= 0 else int(current * 100 / total)
    if percent != last_percent or current in {0, total}:
        write_progress(label, current, total)
        return percent
    return last_percent


def iter_patient_dirs(dataset_root: Path) -> list[Path]:
    return sorted(path for path in dataset_root.iterdir() if path.is_dir())


def iter_exam_dirs(patient_dir: Path) -> list[Path]:
    return sorted(path for path in patient_dir.iterdir() if path.is_dir())


def count_files_in_dir(target_dir: Path, extensions: set[str]) -> int:
    if not target_dir.exists() or not target_dir.is_dir():
        return 0
    return sum(1 for path in target_dir.rglob('*') if path.is_file() and path.suffix.lower() in extensions)


def collect_dataset_stats(dataset_root: Path) -> DatasetStats:
    patient_dirs = iter_patient_dirs(dataset_root)
    stats = DatasetStats(patient_count=len(patient_dirs))

    print(f'开始统计数据集概览，共 {len(patient_dirs)} 名患者。')
    last_percent = update_progress('统计进度', 0, len(patient_dirs), -1)

    for index, patient_dir in enumerate(patient_dirs, start=1):
        exam_dirs = iter_exam_dirs(patient_dir)
        stats.exam_count += len(exam_dirs)

        for exam_dir in exam_dirs:
            stats.image_count += count_files_in_dir(exam_dir / 'img', IMAGE_EXTENSIONS)
            stats.report_count += count_files_in_dir(exam_dir / 'pdf', PDF_EXTENSIONS)

        last_percent = update_progress('统计进度', index, len(patient_dirs), last_percent)

    return stats


def print_dataset_stats(title: str, dataset_root: Path, stats: DatasetStats) -> None:
    print()
    print(title)
    print(f'- 数据目录：{dataset_root}')
    print(f'- 患者数量：{stats.patient_count}')
    print(f'- 检查目录数量：{stats.exam_count}')
    print(f'- 图片数量：{stats.image_count}')
    print(f'- 报告数量：{stats.report_count}')


def print_dataset_delta(before: DatasetStats, after: DatasetStats) -> None:
    print()
    print('统计变化')
    print(f'- 患者数量变化：{after.patient_count - before.patient_count}')
    print(f'- 检查目录数量变化：{after.exam_count - before.exam_count}')
    print(f'- 图片数量变化：{after.image_count - before.image_count}')
    print(f'- 报告数量变化：{after.report_count - before.report_count}')


def build_stats_payload(stats: DatasetStats) -> dict[str, int]:
    return {
        'patient_count': stats.patient_count,
        'exam_count': stats.exam_count,
        'image_count': stats.image_count,
        'report_count': stats.report_count,
    }


def write_delete_broken_data_json(output_path: Path, stats: DatasetStats) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    payload = build_stats_payload(stats)
    output_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')


def print_step_header(step_no: int, title: str) -> None:
    print()
    print(f'========== 第 {step_no} 步：{title} ==========')


def confirm_action(prompt: str, auto_yes: bool) -> bool:
    if auto_yes:
        print(f'{prompt} 自动确认：yes')
        return True

    while True:
        answer = input(f'{prompt} 请输入 y/yes 确认，其他输入取消：').strip().lower()
        if answer in {'y', 'yes'}:
            return True
        if answer in {'', 'n', 'no'}:
            return False
        print('输入无效，请输入 y/yes 或 n/no。')


def print_preview(title: str, lines: Sequence[str], limit: int = PREVIEW_LIMIT) -> None:
    if not lines:
        return

    print(title)
    for index, line in enumerate(lines[:limit], start=1):
        print(f'{index}. {line}')

    remaining = len(lines) - limit
    if remaining > 0:
        print(f'... 其余 {remaining} 项已省略。')


def iter_target_files(dataset_root: Path) -> Iterable[Path]:
    return sorted(
        path for path in dataset_root.rglob('*')
        if path.is_file() and (path.suffix.lower() in PDF_EXTENSIONS or path.suffix.lower() in BROKEN_FILE_SCAN_EXTENSIONS)
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
        actual_crc = binascii.crc32(chunk_data, actual_crc) & 0xFFFFFFFF
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
    if len(data) < 4 or data[:2] != b'\xFF\xD8':
        raise ImageIntegrityError('JPEG 文件头不合法')
    if data[-2:] != b'\xFF\xD9':
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
    if data[-1:] != b'\x3B':
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

    suffix = image_path.suffix.lower()
    if suffix == '.png':
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

    if Image is not None:
        try:
            with Image.open(image_path) as image:
                image.verify()
            with Image.open(image_path) as image:
                image.load()
            return 'pillow-verify+load', file_size
        except (UnidentifiedImageError, OSError, ValueError) as exc:
            raise ImageIntegrityError(f'Pillow 校验失败：{exc}') from exc

    raise ImageIntegrityError(f'暂不支持的图片类型：{suffix or "无扩展名"}')


def check_file_integrity(file_path: Path) -> tuple[str, str, int]:
    suffix = file_path.suffix.lower()
    if suffix == '.pdf':
        check_method, file_size = check_pdf_integrity(file_path)
        return 'pdf', check_method, file_size
    if suffix in BROKEN_FILE_SCAN_EXTENSIONS:
        check_method, file_size = check_image_integrity(file_path)
        return 'image', check_method, file_size
    raise FileIntegrityError(f'暂不支持的文件类型：{suffix or "无扩展名"}')


def infer_check_method(file_path: Path) -> str:
    suffix = file_path.suffix.lower()
    if suffix == '.pdf':
        return 'header+eof+startxref'
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
    if Image is not None:
        return 'pillow-verify+load'
    return 'unknown'


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


def inspect_broken_files(
    dataset_root: Path,
    output_csv: Path,
    max_files: int | None,
) -> tuple[list[BrokenFileRecord], int, int, int]:
    target_files = list(iter_target_files(dataset_root))
    if max_files is not None and max_files > 0:
        target_files = target_files[:max_files]

    if not target_files:
        export_broken_records(output_csv, [], dataset_root)
        return [], 0, 0, 0

    broken_records: list[BrokenFileRecord] = []
    total_pdf_count = sum(1 for path in target_files if path.suffix.lower() in PDF_EXTENSIONS)
    total_image_count = len(target_files) - total_pdf_count

    print(f'开始检查可校验文件，共 {len(target_files)} 个。')
    last_percent = update_progress('文件检查进度', 0, len(target_files), -1)

    for index, file_path in enumerate(target_files, start=1):
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
            _ = file_type
            _ = check_method
            _ = file_size

        last_percent = update_progress('文件检查进度', index, len(target_files), last_percent)

    export_broken_records(output_csv, broken_records, dataset_root)
    return broken_records, len(target_files), total_pdf_count, total_image_count


def delete_broken_files(records: Sequence[BrokenFileRecord]) -> tuple[int, list[tuple[Path, str]]]:
    deleted_count = 0
    failed_records: list[tuple[Path, str]] = []

    for record in records:
        if not record.file_path.exists() or not record.file_path.is_file():
            continue
        try:
            record.file_path.unlink()
            deleted_count += 1
        except OSError as exc:
            failed_records.append((record.file_path, str(exc)))

    return deleted_count, failed_records


def has_target_file(target_dir: Path, extensions: set[str]) -> bool:
    for file_path in target_dir.iterdir():
        if file_path.is_file() and file_path.suffix.lower() in extensions:
            return True
    return False


def inspect_exam_dirs(dataset_root: Path) -> list[InvalidSubdirRecord]:
    invalid_records: list[InvalidSubdirRecord] = []
    patient_dirs = iter_patient_dirs(dataset_root)

    print(f'开始检查 img/pdf 子目录是否为空，共 {len(patient_dirs)} 名患者。')
    last_percent = update_progress('子目录检查进度', 0, len(patient_dirs), -1)

    for index, patient_dir in enumerate(patient_dirs, start=1):
        exam_dirs = iter_exam_dirs(patient_dir)
        for exam_dir in exam_dirs:
            img_dir = exam_dir / 'img'
            pdf_dir = exam_dir / 'pdf'

            if not img_dir.exists():
                invalid_records.append(
                    InvalidSubdirRecord(exam_dir=exam_dir, target_dir=img_dir, target_type='img', reason='目录不存在')
                )
            elif not img_dir.is_dir():
                invalid_records.append(
                    InvalidSubdirRecord(exam_dir=exam_dir, target_dir=img_dir, target_type='img', reason='路径存在但不是目录')
                )
            elif not has_target_file(img_dir, IMAGE_EXTENSIONS):
                invalid_records.append(
                    InvalidSubdirRecord(exam_dir=exam_dir, target_dir=img_dir, target_type='img', reason='未找到图片文件')
                )

            if not pdf_dir.exists():
                invalid_records.append(
                    InvalidSubdirRecord(exam_dir=exam_dir, target_dir=pdf_dir, target_type='pdf', reason='目录不存在')
                )
            elif not pdf_dir.is_dir():
                invalid_records.append(
                    InvalidSubdirRecord(exam_dir=exam_dir, target_dir=pdf_dir, target_type='pdf', reason='路径存在但不是目录')
                )
            elif not has_target_file(pdf_dir, PDF_EXTENSIONS):
                invalid_records.append(
                    InvalidSubdirRecord(exam_dir=exam_dir, target_dir=pdf_dir, target_type='pdf', reason='未找到 PDF 文件')
                )

        last_percent = update_progress('子目录检查进度', index, len(patient_dirs), last_percent)

    return invalid_records


def collect_deletable_invalid_dirs(invalid_records: Sequence[InvalidSubdirRecord]) -> tuple[list[Path], list[str], int]:
    reasons_by_dir: dict[Path, set[str]] = {}
    non_deletable_issue_count = 0

    for record in invalid_records:
        if record.target_dir.exists() and record.target_dir.is_dir():
            reasons_by_dir.setdefault(record.target_dir, set()).add(f'{record.target_type}：{record.reason}')
        else:
            non_deletable_issue_count += 1

    deletable_dirs = sorted(reasons_by_dir)
    preview_lines = [
        f'{target_dir} | 原因：{"；".join(sorted(reasons_by_dir[target_dir]))}'
        for target_dir in deletable_dirs
    ]
    return deletable_dirs, preview_lines, non_deletable_issue_count


def delete_dirs(target_dirs: Sequence[Path]) -> tuple[int, list[tuple[Path, str]]]:
    deleted_count = 0
    failed_records: list[tuple[Path, str]] = []

    for target_dir in target_dirs:
        if not target_dir.exists() or not target_dir.is_dir():
            continue
        try:
            shutil.rmtree(target_dir)
            deleted_count += 1
        except OSError as exc:
            failed_records.append((target_dir, str(exc)))

    return deleted_count, failed_records


def find_incomplete_exam_dirs(dataset_root: Path) -> list[Path]:
    incomplete_dirs: list[Path] = []
    patient_dirs = iter_patient_dirs(dataset_root)

    print(f'开始检查不完整检查目录，共 {len(patient_dirs)} 名患者。')
    last_percent = update_progress('检查目录检查进度', 0, len(patient_dirs), -1)

    for index, patient_dir in enumerate(patient_dirs, start=1):
        for exam_dir in iter_exam_dirs(patient_dir):
            has_img = (exam_dir / 'img').is_dir()
            has_pdf = (exam_dir / 'pdf').is_dir()
            if has_img ^ has_pdf:
                incomplete_dirs.append(exam_dir)

        last_percent = update_progress('检查目录检查进度', index, len(patient_dirs), last_percent)

    return incomplete_dirs


def find_empty_patients(dataset_root: Path) -> tuple[list[Path], int]:
    patient_dirs = iter_patient_dirs(dataset_root)
    empty_patients: list[Path] = []

    print(f'开始检查空患者目录，共 {len(patient_dirs)} 名患者。')
    last_percent = update_progress('空患者检查进度', 0, len(patient_dirs), -1)

    for index, patient_dir in enumerate(patient_dirs, start=1):
        if not any(patient_dir.iterdir()):
            empty_patients.append(patient_dir)
        last_percent = update_progress('空患者检查进度', index, len(patient_dirs), last_percent)

    return empty_patients, len(patient_dirs)


def run_step_delete_broken_files(dataset_root: Path, output_csv: Path, max_files: int | None, auto_yes: bool) -> None:
    print_step_header(1, '删除损坏文件')

    broken_records, total_target_count, total_pdf_count, total_image_count = inspect_broken_files(
        dataset_root=dataset_root,
        output_csv=output_csv,
        max_files=max_files,
    )

    print(f'- 已检查文件数量：{total_target_count}')
    print(f'- 已检查图片数量：{total_image_count}')
    print(f'- 已检查报告数量：{total_pdf_count}')
    print(f'- 待删除损坏文件数量：{len(broken_records)}')
    print(f'- 损坏文件清单 CSV：{output_csv}')

    if not broken_records:
        print('第 1 步未发现损坏文件，直接进入下一步。')
        return

    preview_lines = [
        f'[{record.file_type}] {record.file_path} | 原因：{record.reason}'
        for record in broken_records
    ]
    print_preview('损坏文件预览：', preview_lines)

    if not confirm_action(f'是否执行第 1 步，删除以上 {len(broken_records)} 个损坏文件？', auto_yes):
        print('第 1 步已跳过，损坏文件保持不变。')
        return

    deleted_count, failed_records = delete_broken_files(broken_records)
    print(f'第 1 步已删除 {deleted_count} 个损坏文件。')

    if failed_records:
        print('以下文件删除失败：')
        for failed_path, reason in failed_records:
            print(f'- {failed_path}（原因：{reason}）')


def run_step_delete_empty_dicts(dataset_root: Path, auto_yes: bool) -> None:
    print_step_header(2, '删除空 img/pdf 子目录')

    invalid_records = inspect_exam_dirs(dataset_root)
    deletable_dirs, preview_lines, non_deletable_issue_count = collect_deletable_invalid_dirs(invalid_records)

    print(f'- 检查到的不合规 img/pdf 记录数量：{len(invalid_records)}')
    print(f'- 实际可删除子目录数量：{len(deletable_dirs)}')
    print(f'- 仅用于提示、当前无需删除的异常记录数量：{non_deletable_issue_count}')

    if not deletable_dirs:
        print('第 2 步没有需要删除的实际子目录，直接进入下一步。')
        return

    print_preview('待删除子目录预览：', preview_lines)

    if not confirm_action(f'是否执行第 2 步，删除以上 {len(deletable_dirs)} 个空或异常子目录？', auto_yes):
        print('第 2 步已跳过，子目录保持不变。')
        return

    deleted_count, failed_records = delete_dirs(deletable_dirs)
    print(f'第 2 步已删除 {deleted_count} 个子目录。')

    if failed_records:
        print('以下子目录删除失败：')
        for target_dir, reason in failed_records:
            print(f'- {target_dir}（原因：{reason}）')


def run_step_delete_broken_dicts(dataset_root: Path, auto_yes: bool) -> None:
    print_step_header(3, '删除不完整检查目录')

    incomplete_dirs = find_incomplete_exam_dirs(dataset_root)
    print(f'- 待删除不完整检查目录数量：{len(incomplete_dirs)}')

    if not incomplete_dirs:
        print('第 3 步未发现不完整检查目录，直接进入下一步。')
        return

    preview_lines = [str(exam_dir) for exam_dir in incomplete_dirs]
    print_preview('待删除检查目录预览：', preview_lines)

    if not confirm_action(f'是否执行第 3 步，删除以上 {len(incomplete_dirs)} 个不完整检查目录？', auto_yes):
        print('第 3 步已跳过，检查目录保持不变。')
        return

    deleted_count, failed_records = delete_dirs(incomplete_dirs)
    print(f'第 3 步已删除 {deleted_count} 个不完整检查目录。')

    if failed_records:
        print('以下检查目录删除失败：')
        for exam_dir, reason in failed_records:
            print(f'- {exam_dir}（原因：{reason}）')


def run_step_manual_delete(dataset_root: Path, auto_yes: bool) -> None:
    print_step_header(4, '手动删除指定目录')

    existing_candidates = [
        candidate for candidate in MANUAL_DELETE_CANDIDATES if (dataset_root / candidate.exam_relative_path).exists()
    ]

    print(f'- 手动候选目录数量：{len(existing_candidates)}')
    if not existing_candidates:
        print('第 4 步没有需要手动处理的现存目录，直接进入下一步。')
        return

    preview_lines = [
        f'{(dataset_root / candidate.exam_relative_path).as_posix()} | 原因：{candidate.reason}'
        for candidate in existing_candidates
    ]
    print_preview('手动删除候选目录：', preview_lines, limit=len(preview_lines))

    deleted_count = 0
    failed_records: list[tuple[Path, str]] = []
    skipped_paths: list[Path] = []

    for candidate in existing_candidates:
        target_dir = dataset_root / candidate.exam_relative_path
        prompt = f'是否删除目录 {target_dir.as_posix()}？原因：{candidate.reason}。'
        if not confirm_action(prompt, auto_yes):
            skipped_paths.append(target_dir)
            continue

        try:
            shutil.rmtree(target_dir)
            deleted_count += 1
        except OSError as exc:
            failed_records.append((target_dir, str(exc)))

    print(f'第 4 步已删除 {deleted_count} 个手动指定目录。')

    if skipped_paths:
        print('以下手动目录已保留：')
        for target_dir in skipped_paths:
            print(f'- {target_dir}')

    if failed_records:
        print('以下手动目录删除失败：')
        for target_dir, reason in failed_records:
            print(f'- {target_dir}（原因：{reason}）')


def run_step_delete_empty_patients(dataset_root: Path, auto_yes: bool) -> None:
    print_step_header(5, '删除空患者目录')

    empty_patients, total_patients = find_empty_patients(dataset_root)
    print(f'- 当前患者目录总数：{total_patients}')
    print(f'- 待删除空患者目录数量：{len(empty_patients)}')

    if not empty_patients:
        print('第 5 步未发现空患者目录，清理流程结束。')
        return

    preview_lines = [str(patient_dir) for patient_dir in empty_patients]
    print_preview('待删除空患者目录预览：', preview_lines)

    if not confirm_action(f'是否执行第 5 步，删除以上 {len(empty_patients)} 个空患者目录？', auto_yes):
        print('第 5 步已跳过，空患者目录保持不变。')
        return

    deleted_count, failed_records = delete_dirs(empty_patients)
    print(f'第 5 步已删除 {deleted_count} 个空患者目录。')

    if failed_records:
        print('以下患者目录删除失败：')
        for patient_dir, reason in failed_records:
            print(f'- {patient_dir}（原因：{reason}）')


def main() -> None:
    args = parse_args()
    path_config = build_path_config(args.config, args.input_dir, args.output_dir)
    dataset_root = path_config.dataset_root
    dataset_base_root = path_config.dataset_base_root
    output_csv = path_config.output_dir / args.output_name
    stats_json_path = dataset_base_root / DELETE_BROKEN_DATA_JSON_NAME

    if not dataset_root.exists():
        print(f'输入目录不存在：{dataset_root}')
        return
    if not dataset_root.is_dir():
        print(f'输入路径不是目录：{dataset_root}')
        return

    before_stats = collect_dataset_stats(dataset_root)
    print_dataset_stats('运行前统计', dataset_root, before_stats)

    run_step_delete_broken_files(dataset_root, output_csv, args.max_files, args.yes)
    run_step_delete_empty_dicts(dataset_root, args.yes)
    run_step_delete_broken_dicts(dataset_root, args.yes)
    run_step_manual_delete(dataset_root, args.yes)
    run_step_delete_empty_patients(dataset_root, args.yes)

    after_stats = collect_dataset_stats(dataset_root)
    print_dataset_stats('运行结束后统计', dataset_root, after_stats)
    print_dataset_delta(before_stats, after_stats)
    write_delete_broken_data_json(stats_json_path, after_stats)
    print(f'- 当前数据集统计文件：{stats_json_path}')


if __name__ == '__main__':
    main()
