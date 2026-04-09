from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any

try:
    import yaml  # type: ignore
except ImportError:
    yaml = None


PROJECT_ROOT = Path(__file__).resolve().parent
CONFIG_PATH = PROJECT_ROOT / 'configs' / 'path.yaml'
DEFAULT_TARGET_NAME = 'ZS19332084'


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description='在数据集中查找指定目录名的位置')
    parser.add_argument('--config', type=Path, default=CONFIG_PATH, help='路径配置文件，默认使用 configs/path.yaml')
    parser.add_argument('--dataset-root', type=Path, default=None, help='可选：覆盖配置中的 dataset_root')
    parser.add_argument('--target-name', type=str, default=DEFAULT_TARGET_NAME, help='要查找的目录名')
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


def resolve_dataset_root(config_path: Path, dataset_root: Path | None) -> Path:
    if dataset_root is not None:
        return dataset_root.expanduser().resolve()

    payload = load_yaml_config(config_path.expanduser())
    paths_payload = payload.get('paths')
    if not isinstance(paths_payload, dict):
        raise ValueError('path.yaml 必须包含 paths 分组')

    raw_path = paths_payload.get('dataset_root')
    if not isinstance(raw_path, str) or not raw_path.strip():
        raise ValueError('path.yaml 中缺少有效的 paths.dataset_root')

    resolved_path = Path(raw_path).expanduser()
    if resolved_path.is_absolute():
        return resolved_path.resolve()

    config_dir = config_path.expanduser().resolve().parent
    return (config_dir.parent / resolved_path).resolve()


def render_progress(current: int, total: int, width: int = 40) -> str:
    if total <= 0:
        return '[{}] 100.0% (0/0)'.format('#' * width)
    filled = int(width * current / total)
    bar = '#' * filled + '-' * (width - filled)
    percent = current / total * 100
    return f'[{bar}] {percent:5.1f}% ({current}/{total})'


def write_progress(current: int, total: int) -> None:
    message = '\r搜索进度：' + render_progress(current, total)
    end = '\n' if current >= total else ''
    sys.stdout.write(message + end)
    sys.stdout.flush()


def update_progress(current: int, total: int, last_percent: int) -> int:
    percent = 100 if total <= 0 else int(current * 100 / total)
    if percent != last_percent or current in {0, total}:
        write_progress(current, total)
        return percent
    return last_percent


def find_target_dirs(dataset_root: Path, target_name: str) -> list[Path]:
    direct_match = dataset_root / target_name
    if direct_match.is_dir():
        print(f'已在数据集一级目录中找到目标目录：{direct_match}')
        return [direct_match]

    patient_dirs = sorted(path for path in dataset_root.iterdir() if path.is_dir())
    print(f'一级目录未命中，开始递归搜索，共需检查 {len(patient_dirs)} 个一级目录。')
    last_percent = update_progress(0, len(patient_dirs), -1)

    matched_dirs: list[Path] = []
    for index, patient_dir in enumerate(patient_dirs, start=1):
        if patient_dir.name == target_name:
            matched_dirs.append(patient_dir)

        for subdir in patient_dir.rglob('*'):
            if subdir.is_dir() and subdir.name == target_name:
                matched_dirs.append(subdir)

        last_percent = update_progress(index, len(patient_dirs), last_percent)

    return matched_dirs


def main() -> None:
    args = parse_args()
    dataset_root = resolve_dataset_root(args.config, args.dataset_root)

    if not dataset_root.exists():
        print(f'数据集目录不存在：{dataset_root}')
        return
    if not dataset_root.is_dir():
        print(f'数据集路径不是目录：{dataset_root}')
        return

    print(f'数据集目录：{dataset_root}')
    print(f'目标目录名：{args.target_name}')

    matched_dirs = find_target_dirs(dataset_root, args.target_name)
    if not matched_dirs:
        print('未找到目标目录。')
        return

    print(f'共找到 {len(matched_dirs)} 个匹配目录：')
    for index, matched_dir in enumerate(matched_dirs, start=1):
        print(f'{index}. {matched_dir}')


if __name__ == '__main__':
    main()
