from __future__ import annotations

import argparse
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Any

try:
    import yaml  # type: ignore
except ImportError:
    yaml = None

PROJECT_ROOT = Path(__file__).resolve().parent.parent
CONFIG_PATH = PROJECT_ROOT / 'configs' / 'path.yaml'
IMAGE_EXTENSIONS = {'.jpg', '.jpeg', '.png', '.bmp', '.gif', '.tif', '.tiff', '.webp', '.dcm'}
PDF_EXTENSIONS = {'.pdf'}


@dataclass
class PathConfig:
    dataset_root: Path


@dataclass
class InvalidSubdirRecord:
    exam_dir: Path
    target_dir: Path
    target_type: str
    reason: str


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description='检查每个检查目录下 img/pdf 子目录是否包含对应文件，并可选择删除不合规目录'
    )
    parser.add_argument('--config', type=Path, default=CONFIG_PATH, help='路径配置文件，默认使用 configs/path.yaml')
    parser.add_argument('--input-dir', type=Path, default=None, help='可选：覆盖配置中的 dataset_root')
    parser.add_argument('--yes', action='store_true', help='跳过交互确认，直接删除不合规目录')
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


def build_path_config(config_path: Path, input_dir: Path | None) -> PathConfig:
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
    return PathConfig(dataset_root=dataset_root)


def has_target_file(target_dir: Path, extensions: set[str]) -> bool:
    for file_path in target_dir.iterdir():
        if file_path.is_file() and file_path.suffix.lower() in extensions:
            return True
    return False


def inspect_exam_dirs(dataset_root: Path) -> list[InvalidSubdirRecord]:
    invalid_records: list[InvalidSubdirRecord] = []

    patient_dirs = sorted(path for path in dataset_root.iterdir() if path.is_dir())
    for patient_dir in patient_dirs:
        exam_dirs = sorted(path for path in patient_dir.iterdir() if path.is_dir())
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

    return invalid_records


def print_summary(invalid_records: list[InvalidSubdirRecord]) -> None:
    print('检查完成。以下是 img/pdf 不合规目录统计：')
    print(f'- 不合规目录总数：{len(invalid_records)}')

    img_count = sum(1 for record in invalid_records if record.target_type == 'img')
    pdf_count = sum(1 for record in invalid_records if record.target_type == 'pdf')
    print(f'- img 不合规目录：{img_count}')
    print(f'- pdf 不合规目录：{pdf_count}')

    for index, record in enumerate(invalid_records, start=1):
        print(
            f'{index}. 检查目录：{record.exam_dir} | 目标目录：{record.target_dir} | 类型：{record.target_type} | 原因：{record.reason}'
        )


def confirm_delete(auto_yes: bool) -> bool:
    if auto_yes:
        return True

    while True:
        answer = input('是否删除以上不合规目录中“实际存在且为目录”的项？请输入 y/yes 确认，其他输入取消：').strip().lower()
        if answer in {'y', 'yes'}:
            return True
        if answer in {'', 'n', 'no'}:
            return False
        print('输入无效，请输入 y/yes 或 n/no。')


def delete_invalid_dirs(invalid_records: list[InvalidSubdirRecord]) -> tuple[int, list[tuple[Path, str]]]:
    deleted_count = 0
    failed_records: list[tuple[Path, str]] = []

    deduplicated_dirs = sorted({record.target_dir for record in invalid_records})
    for target_dir in deduplicated_dirs:
        if not target_dir.exists() or not target_dir.is_dir():
            continue

        try:
            shutil.rmtree(target_dir)
            deleted_count += 1
        except OSError as exc:
            failed_records.append((target_dir, str(exc)))

    return deleted_count, failed_records


def main() -> None:
    args = parse_args()
    path_config = build_path_config(args.config, args.input_dir)
    dataset_root = path_config.dataset_root

    if not dataset_root.exists():
        print(f'输入目录不存在：{dataset_root}')
        return
    if not dataset_root.is_dir():
        print(f'输入路径不是目录：{dataset_root}')
        return

    invalid_records = inspect_exam_dirs(dataset_root)
    if not invalid_records:
        print('未发现不合规 img/pdf 目录。')
        return

    print_summary(invalid_records)

    if not confirm_delete(args.yes):
        print('已取消删除，目录保持不变。')
        return

    deleted_count, failed_records = delete_invalid_dirs(invalid_records)
    print(f'已删除 {deleted_count} 个不合规目录。')

    if failed_records:
        print('以下目录删除失败：')
        for target_dir, reason in failed_records:
            print(f'- {target_dir}（原因：{reason}）')


if __name__ == '__main__':
    main()
