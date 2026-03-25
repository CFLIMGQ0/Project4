from __future__ import annotations

import argparse
import csv
import json
import re
import sys
from datetime import datetime, timezone
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
ROUND1_VALID_DICTS_SUMMARY_FILE_NAME = 'valid_dicts_pdf_round1.csv'
ROUND1_VALID_DICTS_REPORT_FILE_NAME = 'valid_dicts_report_round1.csv'
ROUND1_CACHE_FILE_NAME = 'solve_conflicted_pdfs_round1.jsonl'
ROUND2_VALID_DICTS_SUMMARY_FILE_NAME = 'valid_dicts_pdf_round2.csv'
ROUND2_VALID_DICTS_REPORT_FILE_NAME = 'valid_dicts_report_round2.csv'
ROUND2_CACHE_FILE_NAME = 'solve_conflicted_pdfs_round2.jsonl'
ROUND3_VALID_DICTS_SUMMARY_FILE_NAME = 'valid_dicts_pdf_round3.csv'
ROUND3_VALID_DICTS_REPORT_FILE_NAME = 'valid_dicts_report_round3.csv'
ROUND3_CACHE_FILE_NAME = 'solve_conflicted_pdfs_round3.jsonl'
ROUND4_VALID_DICTS_SUMMARY_FILE_NAME = 'valid_dicts_pdf_round4.csv'
ROUND4_VALID_DICTS_REPORT_FILE_NAME = 'valid_dicts_report_round4.csv'
ROUND4_CACHE_FILE_NAME = 'solve_conflicted_pdfs_round4.jsonl'
PROCESS_CACHE_DIR_NAME = 'cache_solve_conflicted_pdfs'
LEGACY_VALID_DICTS_SUMMARY_FILE_NAME = 'valid_dicts_pdf.csv'
LEGACY_VALID_DICTS_REPORT_FILE_NAME = 'valid_dicts_report.csv'
HP_PRIORITY = ['阳性', '阴性', '待确认', '未检']
DIGIT_PATTERN = re.compile(r'\d')
IMPORTANT_EFFECTIVE_KEYS = {
    'badness',
    'hp',
    'score',
    'operationValue',
    'specimen',
    'watchResult',
}
NON_IMPORTANT_EFFECTIVE_KEYS = {
    'archiveTime',
    'checkTime',
    'roomName',
    'anesthesiologistName',
    'narcosisType',
    'doctorName',
    'endoscopeName',
}
@dataclass
class PathConfig:
    dataset_root: Path
    dataset_base_root: Path
    output_dir: Path
    process_cache_dir_name: str


@dataclass
class ExamScanResult:
    exam_dir: Path
    is_valid: bool
    conflict_keys: list[str]
    merged_valid_fields: dict[str, str]
    field_values: dict[str, list[str]]


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

    def close(self) -> None:
        return None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description='扫描检查目录 PDF，输出有效目录与冲突键统计')
    parser.add_argument('--config', type=Path, default=CONFIG_PATH, help='路径配置文件，默认使用 configs/path.yaml')
    parser.add_argument('--input-dir', type=Path, default=None, help='可选：覆盖配置中的 dataset_root')
    parser.add_argument('--dataset-base-root', type=Path, default=None, help='可选：覆盖配置中的 dataset_base_root')
    parser.add_argument('--output-dir', type=Path, default=None, help='可选：覆盖配置中的 output_dir')
    parser.add_argument(
        '--process-cache-dir-name',
        type=str,
        default=None,
        help='可选：覆盖配置中的 process_cache_dir_name（过程文件子目录名）',
    )
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


def build_path_config(
    config_path: Path,
    input_dir: Path | None,
    dataset_base_root: Path | None,
    output_dir: Path | None,
    process_cache_dir_name: str | None,
) -> PathConfig:
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
    resolved_output_dir = output_dir.expanduser().resolve() if output_dir is not None else resolve_path(str(paths_payload['output_dir']))
    configured_process_cache_dir_name = process_cache_dir_name or str(
        paths_payload.get('process_cache_dir_name', PROCESS_CACHE_DIR_NAME)
    )
    cleaned_process_cache_dir_name = normalize_text(configured_process_cache_dir_name).strip('/\\')
    if not cleaned_process_cache_dir_name:
        raise ValueError('process_cache_dir_name 不能为空')
    return PathConfig(
        dataset_root=resolved_dataset_root,
        dataset_base_root=resolved_dataset_base_root,
        output_dir=resolved_output_dir,
        process_cache_dir_name=cleaned_process_cache_dir_name,
    )


def build_progress(total: int, desc: str):
    if tqdm is not None:
        return tqdm(total=total, desc=desc, unit='exam')
    return SimpleProgressBar(total=total, desc=desc)


def normalize_text(value: str) -> str:
    return ' '.join(str(value).strip().split())


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


def extract_non_empty_fields(pdf_path: Path) -> dict[str, str]:
    fields = extract_pdf_fields(pdf_path)
    normalized_fields: dict[str, str] = {}
    for key, raw_value in fields.items():
        key_text = normalize_text(key)
        value_text = normalize_text(raw_value)
        if not key_text or not value_text:
            continue
        normalized_fields[key_text] = value_text
    return normalized_fields


def verify_exam_uniqueness(exam_dir: Path) -> ExamScanResult:
    pdf_files = iter_pdf_files(exam_dir)
    merged_fields: dict[str, str] = {}
    conflict_keys: set[str] = set()
    field_values: dict[str, list[str]] = {}

    for pdf_path in pdf_files:
        try:
            current_fields = extract_non_empty_fields(pdf_path)
        except Exception:
            continue

        for key, current_value in current_fields.items():
            field_values.setdefault(key, []).append(current_value)
            previous_value = merged_fields.get(key)
            if previous_value is None:
                merged_fields[key] = current_value
                continue
            if previous_value != current_value:
                conflict_keys.add(key)

    sorted_conflict_keys = sorted(conflict_keys)
    return ExamScanResult(
        exam_dir=exam_dir,
        is_valid=not sorted_conflict_keys,
        conflict_keys=sorted_conflict_keys,
        merged_valid_fields=merged_fields,
        field_values=field_values,
    )


def parse_archive_time(value: str) -> datetime | None:
    text = normalize_text(value)
    if not text:
        return None

    unix_candidate = text.replace('.', '', 1)
    if unix_candidate.isdigit():
        try:
            number = float(text)
            return datetime.fromtimestamp(number)
        except (OverflowError, OSError, ValueError):
            pass

    normalized = text.replace('Z', '+00:00').replace('/', '-')
    if ' ' in normalized and 'T' not in normalized:
        normalized = normalized.replace(' ', 'T', 1)

    try:
        return datetime.fromisoformat(normalized)
    except ValueError:
        pass

    known_formats = (
        '%Y-%m-%d',
        '%Y-%m-%d %H:%M:%S',
        '%Y-%m-%d %H:%M',
        '%Y%m%d',
        '%Y%m%d%H%M%S',
        '%Y%m%d%H%M',
    )
    for fmt in known_formats:
        try:
            return datetime.strptime(text, fmt)
        except ValueError:
            continue
    return None


def to_utc_timestamp(value: datetime) -> float:
    if value.tzinfo is None:
        value = value.replace(tzinfo=timezone.utc)
    return value.timestamp()


def choose_latest_time_value(values: set[str]) -> str | None:
    if not values:
        return None

    dated_candidates: list[tuple[float, str]] = []
    for item in values:
        parsed = parse_archive_time(item)
        if parsed is None:
            continue
        dated_candidates.append((to_utc_timestamp(parsed), item))

    if not dated_candidates:
        return None

    dated_candidates.sort(key=lambda x: x[0])
    return dated_candidates[-1][1]


def choose_hp_value(values: set[str]) -> str | None:
    if not values:
        return None
    normalized_values = {normalize_text(value) for value in values if normalize_text(value)}
    for candidate in HP_PRIORITY:
        if candidate in normalized_values:
            return candidate
    return sorted(normalized_values, key=lambda x: (len(x), x), reverse=True)[0] if normalized_values else None


def choose_doctor_name(values: set[str]) -> str:
    valid_values = [value for value in values if not DIGIT_PATTERN.search(value)]
    if not valid_values:
        return ''
    sorted_candidates = sorted(valid_values, key=lambda x: (len(x), x), reverse=True)
    return sorted_candidates[0]


def choose_endoscope_name(values: set[str]) -> str | None:
    if not values:
        return None

    candidates: set[str] = set()
    for raw_value in values:
        pieces = re.split(r'[,，]', raw_value)
        for piece in pieces:
            normalized_piece = normalize_text(piece)
            if normalized_piece:
                candidates.add(normalized_piece)

    if not candidates:
        return None

    sorted_candidates = sorted(candidates, key=lambda x: (-len(x), x))
    selected: list[str] = []
    for candidate in sorted_candidates:
        has_digit = bool(DIGIT_PATTERN.search(candidate))
        if (not has_digit) and any(candidate != existing and candidate in existing for existing in selected):
            continue
        selected.append(candidate)

    return ','.join(selected)


def choose_score_value(values: set[str]) -> str | None:
    if not values:
        return None
    scored_candidates: list[tuple[int, str]] = []
    for item in values:
        numbers = [int(match) for match in re.findall(r'\d+', item)]
        if not numbers:
            continue
        scored_candidates.append((max(numbers), item))
    if not scored_candidates:
        return None
    scored_candidates.sort(key=lambda x: (x[0], len(x[1]), x[1]))
    return scored_candidates[-1][1]


def choose_union_text_value(values: set[str]) -> str | None:
    if not values:
        return None

    merged: list[str] = []
    seen: set[str] = set()
    for raw_value in sorted(values):
        pieces = re.split(r'[,，]', raw_value)
        for piece in pieces:
            normalized_piece = normalize_text(piece)
            if not normalized_piece or normalized_piece in seen:
                continue
            seen.add(normalized_piece)
            merged.append(normalized_piece)

    if not merged:
        return None
    return '，'.join(merged)


def split_specimen_items(raw_value: str) -> list[str]:
    return [normalize_text(item) for item in re.split(r'[,，；;、\n]+', raw_value) if normalize_text(item)]


def parse_specimen_item(item: str) -> tuple[str, int | None]:
    matched = re.match(r'^(.*?)[xX＊*]\s*(\d+)$', item)
    if matched is None:
        return item, None
    return normalize_text(matched.group(1)), int(matched.group(2))


def choose_specimen_value(values: set[str]) -> str | None:
    if not values:
        return None

    specimen_count_map: dict[str, int] = {}
    specimen_text_map: dict[str, str] = {}
    for raw_value in values:
        for item in split_specimen_items(raw_value):
            specimen_name, specimen_count = parse_specimen_item(item)
            if not specimen_name:
                continue
            specimen_text_map.setdefault(specimen_name, specimen_name)
            if specimen_count is None:
                continue
            previous_count = specimen_count_map.get(specimen_name)
            if previous_count is None or specimen_count > previous_count:
                specimen_count_map[specimen_name] = specimen_count

    if not specimen_text_map:
        return None

    merged_items: list[str] = []
    for specimen_name in sorted(specimen_text_map):
        if specimen_name in specimen_count_map:
            merged_items.append(f"{specimen_name}*{specimen_count_map[specimen_name]}")
        else:
            merged_items.append(specimen_text_map[specimen_name])
    return '，'.join(merged_items)


def apply_classified_round_rules(
    results: list[ExamScanResult],
    target_keys: set[str],
    stats_template: dict[str, int],
) -> tuple[list[ExamScanResult], dict[str, int]]:
    stats = dict(stats_template)
    patched_results: list[ExamScanResult] = []

    for item in results:
        if not item.conflict_keys:
            patched_results.append(item)
            continue

        new_merged_fields = dict(item.merged_valid_fields)
        new_conflict_keys = list(item.conflict_keys)

        for conflict_key in list(item.conflict_keys):
            if conflict_key not in target_keys:
                continue
            values = item.field_values.get(conflict_key, [])
            if conflict_key in {'archiveTime', 'checkTime'}:
                chosen_value = choose_latest_time_value(values)
                if chosen_value is not None:
                    new_merged_fields[conflict_key] = chosen_value
                    new_conflict_keys = [key for key in new_conflict_keys if key != conflict_key]
                    stats[conflict_key] += 1
            elif conflict_key == 'badness':
                new_merged_fields['badness'] = '有'
                new_conflict_keys = [key for key in new_conflict_keys if key != 'badness']
                stats['badness'] += 1
            elif conflict_key == 'roomName':
                new_merged_fields['roomName'] = ''
                new_conflict_keys = [key for key in new_conflict_keys if key != 'roomName']
                stats['roomName'] += 1
            elif conflict_key == 'anesthesiologistName':
                new_merged_fields['anesthesiologistName'] = ''
                new_conflict_keys = [key for key in new_conflict_keys if key != 'anesthesiologistName']
                stats['anesthesiologistName'] += 1
            elif conflict_key == 'narcosisType':
                new_merged_fields['narcosisType'] = ''
                new_conflict_keys = [key for key in new_conflict_keys if key != 'narcosisType']
                stats['narcosisType'] += 1
            elif conflict_key == 'hp':
                chosen_hp = choose_hp_value(values)
                if chosen_hp is not None:
                    new_merged_fields['hp'] = chosen_hp
                    new_conflict_keys = [key for key in new_conflict_keys if key != 'hp']
                    stats['hp'] += 1
            elif conflict_key == 'score':
                chosen_score = choose_score_value(values)
                if chosen_score is not None:
                    new_merged_fields['score'] = chosen_score
                    new_conflict_keys = [key for key in new_conflict_keys if key != 'score']
                    stats['score'] += 1
            elif conflict_key == 'operationValue':
                chosen_operation_value = choose_union_text_value(values)
                if chosen_operation_value is not None:
                    new_merged_fields['operationValue'] = chosen_operation_value
                    new_conflict_keys = [key for key in new_conflict_keys if key != 'operationValue']
                    stats['operationValue'] += 1
            elif conflict_key == 'specimen':
                chosen_specimen = choose_specimen_value(values)
                if chosen_specimen is not None:
                    new_merged_fields['specimen'] = chosen_specimen
                    new_conflict_keys = [key for key in new_conflict_keys if key != 'specimen']
                    stats['specimen'] += 1
            elif conflict_key == 'watchResult':
                chosen_union_value = choose_union_text_value(values)
                if chosen_union_value is not None:
                    new_merged_fields[conflict_key] = chosen_union_value
                    new_conflict_keys = [key for key in new_conflict_keys if key != conflict_key]
                    stats[conflict_key] += 1
            elif conflict_key == 'doctorName':
                new_merged_fields['doctorName'] = choose_doctor_name(values)
                new_conflict_keys = [key for key in new_conflict_keys if key != 'doctorName']
                stats['doctorName'] += 1
            elif conflict_key == 'endoscopeName':
                chosen_endoscope = choose_endoscope_name(values)
                if chosen_endoscope is not None:
                    new_merged_fields['endoscopeName'] = chosen_endoscope
                    new_conflict_keys = [key for key in new_conflict_keys if key != 'endoscopeName']
                    stats['endoscopeName'] += 1

        patched_results.append(
            ExamScanResult(
                exam_dir=item.exam_dir,
                is_valid=not new_conflict_keys,
                conflict_keys=new_conflict_keys,
                merged_valid_fields=new_merged_fields,
                field_values=item.field_values,
            )
        )
    return patched_results, stats


def apply_second_class_uniqueness_rules(results: list[ExamScanResult]) -> tuple[list[ExamScanResult], dict[str, int]]:
    stats_template = {
        'archiveTime': 0,
        'checkTime': 0,
        'roomName': 0,
        'anesthesiologistName': 0,
        'narcosisType': 0,
        'doctorName': 0,
        'endoscopeName': 0,
    }
    return apply_classified_round_rules(results, NON_IMPORTANT_EFFECTIVE_KEYS, stats_template)


def apply_third_class_uniqueness_rules(results: list[ExamScanResult]) -> tuple[list[ExamScanResult], dict[str, int]]:
    stats_template = {
        'badness': 0,
        'hp': 0,
        'score': 0,
        'operationValue': 0,
        'specimen': 0,
        'watchResult': 0,
    }
    return apply_classified_round_rules(results, IMPORTANT_EFFECTIVE_KEYS, stats_template)


def scan_all_exam_dirs(dataset_root: Path) -> list[ExamScanResult]:
    exam_dirs = iter_exam_dirs(dataset_root)
    results: list[ExamScanResult] = []
    progress = build_progress(total=len(exam_dirs), desc='扫描检查目录')

    try:
        for exam_dir in exam_dirs:
            results.append(verify_exam_uniqueness(exam_dir))
            progress.update(1)
    finally:
        if hasattr(progress, 'close'):
            progress.close()

    return results


def serialize_result(item: ExamScanResult) -> dict[str, Any]:
    return {
        'exam_dir': str(item.exam_dir),
        'is_valid': item.is_valid,
        'conflict_keys': item.conflict_keys,
        'merged_valid_fields': item.merged_valid_fields,
        'field_values': {key: list(values) for key, values in item.field_values.items()},
    }


def deserialize_result(payload: dict[str, Any]) -> ExamScanResult:
    return ExamScanResult(
        exam_dir=Path(str(payload.get('exam_dir', ''))),
        is_valid=bool(payload.get('is_valid', False)),
        conflict_keys=[str(key) for key in payload.get('conflict_keys', [])],
        merged_valid_fields={
            str(key): str(value)
            for key, value in dict(payload.get('merged_valid_fields', {})).items()
        },
        field_values={
            str(key): [str(value) for value in values]
            for key, values in dict(payload.get('field_values', {})).items()
        },
    )


def save_cached_results(cache_path: Path, results: list[ExamScanResult]) -> None:
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    with cache_path.open('w', encoding='utf-8') as cache_file:
        for item in results:
            cache_file.write(json.dumps(serialize_result(item), ensure_ascii=False) + '\n')


def load_cached_results(cache_path: Path) -> list[ExamScanResult]:
    results: list[ExamScanResult] = []
    with cache_path.open('r', encoding='utf-8') as cache_file:
        for line in cache_file:
            text = line.strip()
            if not text:
                continue
            results.append(deserialize_result(json.loads(text)))
    return results


def write_valid_dicts_pdf(output_path: Path, results: list[ExamScanResult]) -> None:
    def get_conflict_num(item: ExamScanResult, key_name: str) -> int:
        if key_name not in item.conflict_keys:
            return 1
        values = [
            normalize_text(value)
            for value in item.field_values.get(key_name, [])
            if normalize_text(value)
        ]
        return max(len(values), 1)

    def build_conflict_types_with_num(item: ExamScanResult, suggest_num: int, watch_num: int) -> str:
        expanded_keys: list[str] = []
        for key in item.conflict_keys:
            if key == 'suggest':
                expanded_keys.extend(['suggest'] * suggest_num)
            elif key == 'watch':
                expanded_keys.extend(['watch'] * watch_num)
            else:
                expanded_keys.append(key)
        return '|'.join(expanded_keys)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open('w', encoding='utf-8-sig', newline='') as csv_file:
        writer = csv.writer(csv_file)
        writer.writerow(
            [
                'exam_dir',
                'is_valid',
                'conflict_key_count',
                'conflict_instance_count',
                'conflict_key_types',
                'suggest_num',
                'watch_num',
            ]
        )
        for item in results:
            suggest_num = get_conflict_num(item, 'suggest')
            watch_num = get_conflict_num(item, 'watch')
            conflict_types = build_conflict_types_with_num(item, suggest_num, watch_num)
            suggest_conflict_instance = suggest_num if 'suggest' in item.conflict_keys else 0
            watch_conflict_instance = watch_num if 'watch' in item.conflict_keys else 0
            conflict_instance_count = (
                len(item.conflict_keys)
                - int('suggest' in item.conflict_keys)
                - int('watch' in item.conflict_keys)
                + suggest_conflict_instance
                + watch_conflict_instance
            )
            writer.writerow(
                [
                    str(item.exam_dir),
                    1 if item.is_valid else 0,
                    len(item.conflict_keys),
                    conflict_instance_count,
                    conflict_types,
                    suggest_num,
                    watch_num,
                ]
            )


def write_valid_dicts_report(output_path: Path, results: list[ExamScanResult]) -> None:
    valid_results = [item for item in results if item.is_valid]
    all_keys = sorted({key for item in valid_results for key in item.merged_valid_fields.keys()})

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open('w', encoding='utf-8-sig', newline='') as csv_file:
        writer = csv.writer(csv_file)
        writer.writerow(['exam_dir', *all_keys])
        for item in valid_results:
            row = [str(item.exam_dir)]
            row.extend(item.merged_valid_fields.get(key, '') for key in all_keys)
            writer.writerow(row)


def apply_round4_suggest_watch_rules(results: list[ExamScanResult]) -> tuple[list[ExamScanResult], dict[str, int]]:
    patched_results: list[ExamScanResult] = []
    stats = {'suggest_dir': 0, 'watch_dir': 0, 'suggest_conflict_num': 0, 'watch_conflict_num': 0}

    for item in results:
        for key_name in ('suggest', 'watch'):
            if key_name not in item.conflict_keys:
                continue
            values = [normalize_text(value) for value in item.field_values.get(key_name, []) if normalize_text(value)]
            conflict_num = max(len(values), 1)
            stats[f'{key_name}_conflict_num'] += conflict_num
            stats[f'{key_name}_dir'] += 1

        patched_results.append(
            ExamScanResult(
                exam_dir=item.exam_dir,
                is_valid=item.is_valid,
                conflict_keys=list(item.conflict_keys),
                merged_valid_fields=dict(item.merged_valid_fields),
                field_values=item.field_values,
            )
        )
    return patched_results, stats


def print_summary(
    summary_path: Path, report_path: Path, results: list[ExamScanResult], title: str, include_output_paths: bool = True
) -> None:
    total_count = len(results)
    valid_count = sum(1 for item in results if item.is_valid)
    invalid_count = total_count - valid_count

    print(f'{title}统计完成。')
    print(f'- 检查目录总数：{total_count}')
    print(f'- 有效检查目录数：{valid_count}')
    print(f'- 无效检查目录数：{invalid_count}')
    if include_output_paths:
        print(f'- 唯一性汇总文件：{summary_path}')
        print(f'- 有效目录键值报告文件：{report_path}')
        print('- valid_dicts_pdf.csv 第二列规则：有效=1，无效=0')


def main() -> None:
    args = parse_args()
    path_config = build_path_config(
        args.config,
        args.input_dir,
        args.dataset_base_root,
        args.output_dir,
        args.process_cache_dir_name,
    )

    if not path_config.dataset_root.exists() or not path_config.dataset_root.is_dir():
        print(f'输入路径不是有效目录：{path_config.dataset_root}')
        return

    output_dir = path_config.output_dir
    process_output_dir = output_dir / path_config.process_cache_dir_name
    round1_summary_path = process_output_dir / ROUND1_VALID_DICTS_SUMMARY_FILE_NAME
    round1_report_path = process_output_dir / ROUND1_VALID_DICTS_REPORT_FILE_NAME
    round1_cache_path = process_output_dir / ROUND1_CACHE_FILE_NAME

    round2_summary_path = process_output_dir / ROUND2_VALID_DICTS_SUMMARY_FILE_NAME
    round2_report_path = process_output_dir / ROUND2_VALID_DICTS_REPORT_FILE_NAME
    round2_cache_path = process_output_dir / ROUND2_CACHE_FILE_NAME
    round3_summary_path = process_output_dir / ROUND3_VALID_DICTS_SUMMARY_FILE_NAME
    round3_report_path = process_output_dir / ROUND3_VALID_DICTS_REPORT_FILE_NAME
    round3_cache_path = process_output_dir / ROUND3_CACHE_FILE_NAME
    round4_summary_path = process_output_dir / ROUND4_VALID_DICTS_SUMMARY_FILE_NAME
    round4_report_path = process_output_dir / ROUND4_VALID_DICTS_REPORT_FILE_NAME
    round4_cache_path = process_output_dir / ROUND4_CACHE_FILE_NAME
    legacy_summary_path = path_config.dataset_base_root / LEGACY_VALID_DICTS_SUMMARY_FILE_NAME
    legacy_report_path = path_config.dataset_base_root / LEGACY_VALID_DICTS_REPORT_FILE_NAME

    round1_ready = round1_cache_path.exists() and round1_summary_path.exists() and round1_report_path.exists()
    if round1_ready:
        round1_results = load_cached_results(round1_cache_path)
        print(f'检测到第一轮确认结果，跳过第一轮计算：{round1_cache_path}')
    else:
        round1_results = scan_all_exam_dirs(path_config.dataset_root)
        save_cached_results(round1_cache_path, round1_results)
        write_valid_dicts_pdf(round1_summary_path, round1_results)
        write_valid_dicts_report(round1_report_path, round1_results)
        print(f'第一轮缓存与结果已生成：{process_output_dir}')

    if not round1_summary_path.exists() or not round1_report_path.exists():
        write_valid_dicts_pdf(round1_summary_path, round1_results)
        write_valid_dicts_report(round1_report_path, round1_results)

    print_summary(round1_summary_path, round1_report_path, round1_results, title='第一轮唯一性确认')

    round2_ready = round2_cache_path.exists() and round2_summary_path.exists() and round2_report_path.exists()
    if round2_ready:
        round2_results = load_cached_results(round2_cache_path)
        second_class_stats = None
        print(f'检测到第二类确认结果，跳过第二类计算：{round2_cache_path}')
    else:
        round2_results, second_class_stats = apply_second_class_uniqueness_rules(round1_results)
        save_cached_results(round2_cache_path, round2_results)
        write_valid_dicts_pdf(round2_summary_path, round2_results)
        write_valid_dicts_report(round2_report_path, round2_results)
        print(f'第二类缓存与结果已生成：{process_output_dir}')

    if not round2_summary_path.exists() or not round2_report_path.exists():
        write_valid_dicts_pdf(round2_summary_path, round2_results)
        write_valid_dicts_report(round2_report_path, round2_results)

    print_summary(round2_summary_path, round2_report_path, round2_results, title='第二类唯一性确认（非重要有效键冲突处理）')
    if second_class_stats is not None:
        print(f"- 第二类按最晚时间消解 archiveTime 冲突目录数：{second_class_stats['archiveTime']}")
        print(f"- 第二类按最晚时间消解 checkTime 冲突目录数：{second_class_stats['checkTime']}")
        print(f"- 第二类清空 roomName 的目录数：{second_class_stats['roomName']}")
        print(f"- 第二类清空 anesthesiologistName 的目录数：{second_class_stats['anesthesiologistName']}")
        print(f"- 第二类清空 narcosisType 的目录数：{second_class_stats['narcosisType']}")
        print(f"- 第二类按规则处理 doctorName 的目录数：{second_class_stats['doctorName']}")
        print(f"- 第二类按合并去重规则处理 endoscopeName 的目录数：{second_class_stats['endoscopeName']}")

    round3_ready = round3_cache_path.exists() and round3_summary_path.exists() and round3_report_path.exists()
    if round3_ready:
        round3_results = load_cached_results(round3_cache_path)
        third_class_stats = None
        print(f'检测到第三类确认结果，跳过第三类计算：{round3_cache_path}')
    else:
        round3_results, third_class_stats = apply_third_class_uniqueness_rules(round2_results)
        save_cached_results(round3_cache_path, round3_results)
        write_valid_dicts_pdf(round3_summary_path, round3_results)
        write_valid_dicts_report(round3_report_path, round3_results)
        write_valid_dicts_pdf(legacy_summary_path, round3_results)
        write_valid_dicts_report(legacy_report_path, round3_results)
        print(f'第三类缓存与结果已生成：{process_output_dir}')

    if not round3_summary_path.exists() or not round3_report_path.exists():
        write_valid_dicts_pdf(round3_summary_path, round3_results)
        write_valid_dicts_report(round3_report_path, round3_results)
    print_summary(round3_summary_path, round3_report_path, round3_results, title='第三类唯一性确认（重要有效键冲突处理）')
    if third_class_stats is not None:
        print(f"- 第三类设置 badness='有' 的目录数：{third_class_stats['badness']}")
        print(f"- 第三类按优先级处理 hp 的目录数：{third_class_stats['hp']}")
        print(f"- 第三类按最大分数处理 score 的目录数：{third_class_stats['score']}")
        print(f"- 第三类按逗号拆分合并去重处理 operationValue 的目录数：{third_class_stats['operationValue']}")
        print(f"- 第三类按部位合并处理 specimen 的目录数：{third_class_stats['specimen']}")
        print(f"- 第三类按逗号拆分合并处理 watchResult 的目录数：{third_class_stats['watchResult']}")

    round4_ready = round4_cache_path.exists() and round4_summary_path.exists() and round4_report_path.exists()
    if round4_ready:
        round4_results = load_cached_results(round4_cache_path)
        round4_stats = None
        print(f'检测到第四轮确认结果，跳过第四轮计算：{round4_cache_path}')
    else:
        round4_results, round4_stats = apply_round4_suggest_watch_rules(round3_results)
        print('第四轮统计 suggest/watch 冲突规模，保留这两个字段的冲突。')
        save_cached_results(round4_cache_path, round4_results)
        write_valid_dicts_pdf(round4_summary_path, round4_results)
        write_valid_dicts_report(round4_report_path, round4_results)
        print(f'第四轮缓存与结果已生成：{process_output_dir}')

    if not round4_summary_path.exists() or not round4_report_path.exists():
        write_valid_dicts_pdf(round4_summary_path, round4_results)
        write_valid_dicts_report(round4_report_path, round4_results)
    print_summary(round4_summary_path, round4_report_path, round4_results, title='第四轮唯一性确认（统计 suggest/watch 冲突并保留）')
    if round4_stats is not None:
        print(f"- 第四轮 suggest 冲突目录数：{round4_stats['suggest_dir']}")
        print(f"- 第四轮 suggest 冲突项数量：{round4_stats['suggest_conflict_num']}")
        print(f"- 第四轮 watch 冲突目录数：{round4_stats['watch_dir']}")
        print(f"- 第四轮 watch 冲突项数量：{round4_stats['watch_conflict_num']}")

    write_valid_dicts_pdf(legacy_summary_path, round4_results)
    write_valid_dicts_report(legacy_report_path, round4_results)
    print(f'第四轮完成，已更新数据集根目录兼容输出文件：{legacy_summary_path}、{legacy_report_path}')

    unresolved_round4_keys = sorted({key for item in round4_results for key in item.conflict_keys})
    for unresolved_key in unresolved_round4_keys:
        print(f'{unresolved_key}冲突未完全解决')


if __name__ == '__main__':
    main()
