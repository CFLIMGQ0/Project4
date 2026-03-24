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


@dataclass
class PathConfig:
    dataset_root: Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description='查找空目录患者并按需删除')
    parser.add_argument('--config', type=Path, default=CONFIG_PATH, help='路径配置文件，默认使用 configs/path.yaml')
    parser.add_argument('--input-dir', type=Path, default=None, help='可选：覆盖配置中的 dataset_root')
    parser.add_argument('--yes', action='store_true', help='跳过交互确认，直接删除空目录患者')
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


def is_empty_patient_dir(patient_dir: Path) -> bool:
    return not any(patient_dir.iterdir())


def find_empty_patients(dataset_root: Path) -> tuple[list[Path], int]:
    patient_dirs = sorted(path for path in dataset_root.iterdir() if path.is_dir())
    empty_patients = [patient_dir for patient_dir in patient_dirs if is_empty_patient_dir(patient_dir)]
    return empty_patients, len(patient_dirs)


def print_summary(total_patients: int, empty_patients: list[Path]) -> None:
    empty_count = len(empty_patients)
    print('检查完成：')
    print(f'- 患者目录总数：{total_patients}')
    print(f'- 空目录患者数：{empty_count}')
    print(f'- 非空目录患者数：{total_patients - empty_count}')

    if not empty_patients:
        return

    print('以下为空目录患者：')
    for index, patient_dir in enumerate(empty_patients, start=1):
        print(f'{index}. {patient_dir}')


def confirm_delete(auto_yes: bool) -> bool:
    if auto_yes:
        return True

    while True:
        answer = input('是否删除以上空目录患者？请输入 y/yes 确认，其他输入取消：').strip().lower()
        if answer in {'y', 'yes'}:
            return True
        if answer in {'', 'n', 'no'}:
            return False
        print('输入无效，请输入 y/yes 或 n/no。')


def delete_patients(empty_patients: list[Path]) -> tuple[int, list[tuple[Path, str]]]:
    deleted_count = 0
    failed_records: list[tuple[Path, str]] = []

    for patient_dir in empty_patients:
        if not patient_dir.exists() or not patient_dir.is_dir():
            continue

        try:
            shutil.rmtree(patient_dir)
            deleted_count += 1
        except OSError as exc:
            failed_records.append((patient_dir, str(exc)))

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

    empty_patients, total_patients = find_empty_patients(dataset_root)
    print_summary(total_patients, empty_patients)

    if not empty_patients:
        print('未发现空目录患者，无需删除。')
        return

    if not confirm_delete(args.yes):
        print('已取消删除，目录保持不变。')
        return

    deleted_count, failed_records = delete_patients(empty_patients)
    print(f'已删除 {deleted_count} 个空目录患者。')

    if failed_records:
        print('以下目录删除失败：')
        for patient_dir, reason in failed_records:
            print(f'- {patient_dir}（原因：{reason}）')


if __name__ == '__main__':
    main()
