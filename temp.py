from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from shutil import get_terminal_size
from typing import Any

try:
    import yaml  # type: ignore
except ImportError:
    yaml = None

try:
    from tqdm import tqdm  # type: ignore
except ImportError:
    tqdm = None

PROJECT_ROOT = Path(__file__).resolve().parent
DEFAULT_CONFIG_PATH = PROJECT_ROOT / 'configs' / 'path.yaml'
ROUND2_CACHE_FILE_NAME = 'solve_conflicted_pdfs_round2.jsonl'
DEFAULT_PROCESS_CACHE_DIR_NAME = 'cache_solve_conflicted_pdfs'
NON_IMPORTANT_EFFECTIVE_KEYS = {
    'archiveTime',
    'checkTime',
    'roomName',
    'anesthesiologistName',
    'narcosisType',
    'doctorName',
    'endoscopeName',
    'applyDeptName',
    'applyNo',
    'bedId',
    'hisPatientId',
    'patientAreaName',
    'admissionNo',
    'patientType',
}


@dataclass
class PathConfig:
    output_dir: Path
    process_cache_dir_name: str


class SimpleProgressBar:
    def __init__(self, total: int, desc: str) -> None:
        self.total = max(total, 1)
        self.current = 0
        self.desc = desc

    def update(self, step: int = 1) -> None:
        self.current = min(self.total, self.current + step)
        width = min(40, max(10, get_terminal_size((80, 20)).columns - 40))
        ratio = self.current / self.total
        done = int(width * ratio)
        bar = '=' * done + '-' * (width - done)
        print(f'\r{self.desc}: [{bar}] {self.current}/{self.total} ({ratio * 100:5.1f}%)', end='', flush=True)
        if self.current >= self.total:
            print()


def build_progress(total: int, desc: str):
    if tqdm is not None:
        return tqdm(total=total, desc=desc, unit='record')
    return SimpleProgressBar(total=total, desc=desc)


def normalize_text(value: Any) -> str:
    return ' '.join(str(value).strip().split())


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description='检查第二轮唯一性确认后是否仍有非重要键冲突')
    parser.add_argument('--config', type=Path, default=DEFAULT_CONFIG_PATH, help='路径配置文件，默认 configs/path.yaml')
    parser.add_argument('--round2-cache', type=Path, default=None, help='可选：直接指定 solve_conflicted_pdfs_round2.jsonl')
    parser.add_argument('--max-examples', type=int, default=20, help='最多展示的冲突目录样例数，默认 20')
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


def build_path_config(config_path: Path) -> PathConfig:
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

    output_dir = resolve_path(str(paths_payload['output_dir']))
    process_cache_dir_name = normalize_text(paths_payload.get('process_cache_dir_name', DEFAULT_PROCESS_CACHE_DIR_NAME)).strip('/\\')
    if not process_cache_dir_name:
        raise ValueError('process_cache_dir_name 不能为空')

    return PathConfig(output_dir=output_dir, process_cache_dir_name=process_cache_dir_name)


def check_round2_non_important_conflicts(round2_cache_path: Path, max_examples: int) -> int:
    lines = round2_cache_path.read_text(encoding='utf-8').splitlines()
    progress = build_progress(total=len(lines), desc='扫描第二轮缓存')

    remaining_non_important_conflicts: dict[str, list[str]] = {}
    non_important_conflict_stats: dict[str, int] = {}

    try:
        for line in lines:
            text = line.strip()
            if not text:
                progress.update(1)
                continue

            payload = json.loads(text)
            exam_dir = str(payload.get('exam_dir', ''))
            conflict_keys = [str(item) for item in payload.get('conflict_keys', [])]

            hit_keys = sorted({key for key in conflict_keys if key in NON_IMPORTANT_EFFECTIVE_KEYS})
            if hit_keys:
                remaining_non_important_conflicts[exam_dir] = hit_keys
                for key in hit_keys:
                    non_important_conflict_stats[key] = non_important_conflict_stats.get(key, 0) + 1

            progress.update(1)
    finally:
        if hasattr(progress, 'close'):
            progress.close()

    print('\n=== 第二轮唯一性确认后：非重要键冲突检查结果 ===')
    print(f'- 读取文件：{round2_cache_path}')
    print(f'- 检查目录总数：{len(lines)}')
    print(f'- 仍含非重要键冲突的目录数：{len(remaining_non_important_conflicts)}')

    if not remaining_non_important_conflicts:
        print('- 结论：第二轮后未发现剩余非重要键冲突。')
        return 0

    print('- 剩余冲突键计数（按目录数降序）：')
    for key, count in sorted(non_important_conflict_stats.items(), key=lambda item: (-item[1], item[0])):
        print(f'  - {key}: {count}')

    if max_examples > 0:
        print(f'- 冲突目录样例（最多 {max_examples} 条）：')
        for idx, (exam_dir, keys) in enumerate(sorted(remaining_non_important_conflicts.items()), start=1):
            if idx > max_examples:
                break
            print(f'  {idx}. {exam_dir}')
            print(f'     冲突键: {", ".join(keys)}')

    return 1


def main() -> None:
    args = parse_args()
    if args.max_examples < 0:
        raise ValueError('--max-examples 不能为负数')

    config = build_path_config(args.config)
    if args.round2_cache is not None:
        round2_cache_path = args.round2_cache.expanduser().resolve()
    else:
        round2_cache_path = config.output_dir / config.process_cache_dir_name / ROUND2_CACHE_FILE_NAME

    if not round2_cache_path.is_file():
        raise FileNotFoundError(
            f'第二轮缓存文件不存在：{round2_cache_path}\n'
            '请先运行 scripts/solve_conflicted_pdfs.py 生成第二轮结果，或通过 --round2-cache 显式指定路径。'
        )

    exit_code = check_round2_non_important_conflicts(round2_cache_path=round2_cache_path, max_examples=args.max_examples)
    raise SystemExit(exit_code)


if __name__ == '__main__':
    main()
