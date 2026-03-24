from __future__ import annotations

import argparse
import csv
import datetime as dt
import json
import sys
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

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from statistics import extract_pdf_fields  # noqa: E402

CONFIG_PATH = PROJECT_ROOT / 'configs' / 'path.yaml'
CACHE_FILE_NAME = 'conflicted_dicts.csv'
CONFLICT_VALUE_TYPE_SUMMARY_FILE_NAME = 'conflict_value_type_summary.csv'
CONFLICT_VALUE_DETAILS_FILE_NAME = 'conflict_key_value_details.json'
IGNORE_CONFLICT_KEY = 'archiveTime'
TARGET_DETAIL_KEYS = [
    'specimen',
    'doctorName',
    'operationValue',
    'endoscopeName',
    'narcosisType',
    'anesthesiologistName',
    'checkTime',
    'hp',
    'roomName',
    'score',
    'badness',
]


@dataclass
class PathConfig:
    dataset_root: Path
    dataset_base_root: Path


@dataclass
class ConflictExamRecord:
    patient_id: str
    exam_id: str
    exam_dir: Path
    pdf_count: int
    parsed_pdf_count: int
    conflict_keys: list[str]
    conflict_value_map: dict[str, list[str]]
    latest_archive_time: str


@dataclass
class PdfFieldSnapshot:
    pdf_path: Path
    fields: dict[str, str]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description='统计并缓存冲突检查目录，archiveTime 按最晚值处理')
    parser.add_argument('--config', type=Path, default=CONFIG_PATH, help='路径配置文件，默认使用 configs/path.yaml')
    parser.add_argument('--input-dir', type=Path, default=None, help='可选：覆盖配置中的 dataset_root')
    parser.add_argument('--dataset-base-root', type=Path, default=None, help='可选：覆盖配置中的 dataset_base_root')
    parser.add_argument('--force-rescan', action='store_true', help='忽略已有 conflicted_dicts.csv，强制重新扫描')
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


def build_path_config(config_path: Path, input_dir: Path | None, dataset_base_root: Path | None) -> PathConfig:
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

    resolved_dataset_root = input_dir.expanduser().resolve() if input_dir is not None else resolve_path(str(paths_payload['dataset_root']))
    resolved_dataset_base_root = (
        dataset_base_root.expanduser().resolve()
        if dataset_base_root is not None
        else resolve_path(str(paths_payload['dataset_base_root']))
    )
    return PathConfig(dataset_root=resolved_dataset_root, dataset_base_root=resolved_dataset_base_root)


def iter_exam_dirs(dataset_root: Path) -> list[Path]:
    exam_dirs: list[Path] = []
    for patient_dir in sorted(path for path in dataset_root.iterdir() if path.is_dir()):
        exam_dirs.extend(sorted(path for path in patient_dir.iterdir() if path.is_dir()))
    return exam_dirs


def iter_pdf_files(exam_dir: Path) -> list[Path]:
    pdf_dir = exam_dir / 'pdf'
    if not pdf_dir.is_dir():
        return []
    return sorted(path for path in pdf_dir.rglob('*.pdf') if path.is_file())


def normalize_text(value: str) -> str:
    return ' '.join(str(value).strip().split())


def parse_archive_time(value: str) -> dt.datetime | None:
    text = normalize_text(value)
    if not text:
        return None

    candidates = [
        '%Y-%m-%d %H:%M:%S',
        '%Y/%m/%d %H:%M:%S',
        '%Y-%m-%d %H:%M',
        '%Y/%m/%d %H:%M',
        '%Y-%m-%d',
        '%Y/%m/%d',
        '%Y%m%d%H%M%S',
        '%Y%m%d',
    ]
    for fmt in candidates:
        try:
            return dt.datetime.strptime(text, fmt)
        except ValueError:
            continue
    return None


def pick_latest_archive_time(values: set[str]) -> str:
    normalized = [normalize_text(value) for value in values if normalize_text(value)]
    if not normalized:
        return ''

    parsed_values: list[tuple[dt.datetime, str]] = []
    for value in normalized:
        parsed = parse_archive_time(value)
        if parsed is not None:
            parsed_values.append((parsed, value))

    if parsed_values:
        parsed_values.sort(key=lambda item: (item[0], item[1]))
        return parsed_values[-1][1]

    return sorted(set(normalized))[-1]


def find_conflict_keys(pdf_snapshots: list[PdfFieldSnapshot]) -> tuple[list[str], dict[str, list[str]], str]:
    key_to_values: dict[str, set[str]] = {}
    for snapshot in pdf_snapshots:
        for key, raw_value in snapshot.fields.items():
            value = normalize_text(raw_value)
            if not value:
                continue
            key_to_values.setdefault(key, set()).add(value)

    archive_values = key_to_values.get(IGNORE_CONFLICT_KEY, set())
    latest_archive_time = pick_latest_archive_time(archive_values)

    conflict_value_map = {
        key: sorted(values)
        for key, values in key_to_values.items()
        if key != IGNORE_CONFLICT_KEY and len(values) > 1
    }
    conflict_keys = sorted(conflict_value_map.keys())
    return conflict_keys, conflict_value_map, latest_archive_time


def summarize_conflict_value_types(conflict_records: list[ConflictExamRecord]) -> list[tuple[str, int]]:
    key_to_values: dict[str, set[str]] = {}
    for record in conflict_records:
        for key, values in record.conflict_value_map.items():
            if not values:
                continue
            key_to_values.setdefault(key, set()).update(values)
    return sorted(
        ((key, len(values)) for key, values in key_to_values.items()),
        key=lambda item: (-item[1], item[0]),
    )


def collect_key_value_details(conflict_records: list[ConflictExamRecord], target_keys: list[str]) -> dict[str, list[str]]:
    key_to_values: dict[str, set[str]] = {key: set() for key in target_keys}
    for record in conflict_records:
        for key in target_keys:
            values = record.conflict_value_map.get(key)
            if not values:
                continue
            key_to_values[key].update(values)
    return {key: sorted(values) for key, values in key_to_values.items()}


def write_key_value_details_json(output_path: Path, payload: dict[str, list[str]]) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding='utf-8')


def has_conflict_value_details(conflict_records: list[ConflictExamRecord]) -> bool:
    has_conflict = False
    for record in conflict_records:
        if record.conflict_keys:
            has_conflict = True
        if record.conflict_value_map:
            return True
    return not has_conflict


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
        message = f'\r{self.desc}: [{bar}] {self.current}/{self.total} ({ratio * 100:5.1f}%)'
        print(message, end='', flush=True)
        if self.current >= self.total:
            print()


def build_progress(total: int, desc: str):
    if tqdm is not None:
        return tqdm(total=total, desc=desc, unit='exam')
    return SimpleProgressBar(total=total, desc=desc)


def scan_conflicted_exam_dirs(dataset_root: Path, exam_dirs: list[Path] | None = None) -> list[ConflictExamRecord]:
    conflict_records: list[ConflictExamRecord] = []
    target_exam_dirs = exam_dirs if exam_dirs is not None else iter_exam_dirs(dataset_root)
    progress = build_progress(total=len(target_exam_dirs), desc='扫描检查目录')

    try:
        for exam_dir in target_exam_dirs:
            pdf_files = iter_pdf_files(exam_dir)
            if len(pdf_files) < 2:
                progress.update(1)
                continue

            pdf_snapshots: list[PdfFieldSnapshot] = []
            for pdf_path in pdf_files:
                try:
                    fields = extract_pdf_fields(pdf_path)
                except Exception:
                    continue
                pdf_snapshots.append(PdfFieldSnapshot(pdf_path=pdf_path, fields=fields))

            if len(pdf_snapshots) < 2:
                progress.update(1)
                continue

            conflict_keys, conflict_value_map, latest_archive_time = find_conflict_keys(pdf_snapshots)
            if not conflict_keys:
                progress.update(1)
                continue

            patient_id = exam_dir.parent.name
            exam_id = exam_dir.name
            conflict_records.append(
                ConflictExamRecord(
                    patient_id=patient_id,
                    exam_id=exam_id,
                    exam_dir=exam_dir,
                    pdf_count=len(pdf_files),
                    parsed_pdf_count=len(pdf_snapshots),
                    conflict_keys=conflict_keys,
                    conflict_value_map=conflict_value_map,
                    latest_archive_time=latest_archive_time,
                )
            )
            progress.update(1)
    finally:
        if hasattr(progress, 'close'):
            progress.close()

    return conflict_records


def write_conflict_cache(cache_path: Path, conflict_records: list[ConflictExamRecord]) -> None:
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    with cache_path.open('w', encoding='utf-8-sig', newline='') as csv_file:
        writer = csv.DictWriter(
            csv_file,
            fieldnames=[
                'patient_id',
                'exam_id',
                'exam_dir',
                'pdf_count',
                'parsed_pdf_count',
                'conflict_key_count',
                'conflict_keys',
                'conflict_value_map',
                'latest_archive_time',
            ],
        )
        writer.writeheader()
        for record in conflict_records:
            writer.writerow(
                {
                    'patient_id': record.patient_id,
                    'exam_id': record.exam_id,
                    'exam_dir': str(record.exam_dir),
                    'pdf_count': record.pdf_count,
                    'parsed_pdf_count': record.parsed_pdf_count,
                    'conflict_key_count': len(record.conflict_keys),
                    'conflict_keys': '|'.join(record.conflict_keys),
                    'conflict_value_map': json.dumps(record.conflict_value_map, ensure_ascii=False, sort_keys=True),
                    'latest_archive_time': record.latest_archive_time,
                }
            )


def load_conflict_cache(cache_path: Path) -> list[ConflictExamRecord]:
    records: list[ConflictExamRecord] = []
    with cache_path.open('r', encoding='utf-8-sig', newline='') as csv_file:
        reader = csv.DictReader(csv_file)
        for row in reader:
            conflict_keys = [key for key in (row.get('conflict_keys') or '').split('|') if key]
            raw_conflict_value_map = (row.get('conflict_value_map') or '').strip()
            conflict_value_map: dict[str, list[str]] = {}
            if raw_conflict_value_map:
                try:
                    payload = json.loads(raw_conflict_value_map)
                    if isinstance(payload, dict):
                        for key, values in payload.items():
                            if not isinstance(key, str):
                                continue
                            if isinstance(values, list):
                                normalized_values = [normalize_text(str(value)) for value in values if normalize_text(str(value))]
                                if normalized_values:
                                    conflict_value_map[key] = sorted(set(normalized_values))
                except json.JSONDecodeError:
                    conflict_value_map = {}
            records.append(
                ConflictExamRecord(
                    patient_id=(row.get('patient_id') or '').strip(),
                    exam_id=(row.get('exam_id') or '').strip(),
                    exam_dir=Path((row.get('exam_dir') or '').strip()),
                    pdf_count=int((row.get('pdf_count') or '0').strip() or 0),
                    parsed_pdf_count=int((row.get('parsed_pdf_count') or '0').strip() or 0),
                    conflict_keys=conflict_keys,
                    conflict_value_map=conflict_value_map,
                    latest_archive_time=(row.get('latest_archive_time') or '').strip(),
                )
            )
    return records


def resolve_cached_exam_dirs(conflict_records: list[ConflictExamRecord], dataset_root: Path) -> list[Path]:
    resolved_exam_dirs: list[Path] = []
    seen_paths: set[Path] = set()
    for record in conflict_records:
        candidate = record.exam_dir
        if not candidate.is_absolute():
            candidate = (dataset_root / record.patient_id / record.exam_id).resolve()
        else:
            candidate = candidate.resolve()
        if not candidate.is_dir() or candidate in seen_paths:
            continue
        seen_paths.add(candidate)
        resolved_exam_dirs.append(candidate)
    return resolved_exam_dirs


def print_summary(
    cache_path: Path,
    summary_path: Path,
    details_path: Path,
    conflict_records: list[ConflictExamRecord],
    from_cache: bool,
) -> None:
    conflict_value_type_summary = summarize_conflict_value_types(conflict_records)
    key_value_details = collect_key_value_details(conflict_records, TARGET_DETAIL_KEYS)
    write_key_value_details_json(details_path, key_value_details)

    print('冲突检查目录处理完成。')
    print(f'- 数据来源：{"缓存文件" if from_cache else "重新扫描"}')
    print(f'- 冲突检查目录数量：{len(conflict_records)}')
    print(f'- 缓存文件路径：{cache_path}')
    print(f'- 冲突值类型统计文件：{summary_path}')
    print(f'- 指定键冲突值明细 JSON：{details_path}')
    print(f'- 冲突判定忽略键：{IGNORE_CONFLICT_KEY}')
    print('- archiveTime 处理规则：同一检查目录内取最晚值（已写入 latest_archive_time 列）')
    if conflict_value_type_summary:
        print('- 冲突键值类型统计（键 -> 不同冲突值数量）：')
        for key, value_type_count in conflict_value_type_summary:
            print(f'  - {key}: {value_type_count} 种')
    else:
        print('- 冲突键值类型统计：无')

    print('- 指定键冲突值明细（键 -> 具体值列表）：')
    for key in TARGET_DETAIL_KEYS:
        values = key_value_details.get(key, [])
        print(f'  - {key}: {len(values)} 种')
        for value in values:
            print(f'    - {value}')


def main() -> None:
    args = parse_args()
    path_config = build_path_config(args.config, args.input_dir, args.dataset_base_root)

    if not path_config.dataset_root.exists() or not path_config.dataset_root.is_dir():
        print(f'输入路径不是有效目录：{path_config.dataset_root}')
        return

    cache_path = path_config.dataset_base_root / CACHE_FILE_NAME
    summary_path = path_config.dataset_base_root / CONFLICT_VALUE_TYPE_SUMMARY_FILE_NAME
    details_path = path_config.dataset_base_root / CONFLICT_VALUE_DETAILS_FILE_NAME

    if cache_path.exists() and not args.force_rescan:
        conflict_records = load_conflict_cache(cache_path)
        if has_conflict_value_details(conflict_records):
            print_summary(cache_path, summary_path, details_path, conflict_records, from_cache=True)
            return

        cached_exam_dirs = resolve_cached_exam_dirs(conflict_records, path_config.dataset_root)
        print('检测到缓存缺少 conflict_value_map 明细，将只扫描 conflicted_dicts.csv 中记录的检查目录。')
        conflict_records = scan_conflicted_exam_dirs(path_config.dataset_root, exam_dirs=cached_exam_dirs)
        write_conflict_cache(cache_path, conflict_records)
        print_summary(cache_path, summary_path, details_path, conflict_records, from_cache=False)
        return

    conflict_records = scan_conflicted_exam_dirs(path_config.dataset_root)
    write_conflict_cache(cache_path, conflict_records)
    print_summary(cache_path, summary_path, details_path, conflict_records, from_cache=False)


if __name__ == '__main__':
    main()
