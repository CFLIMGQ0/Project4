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
LEGACY_VALID_DICTS_SUMMARY_FILE_NAME = 'valid_dicts_pdf.csv'
LEGACY_VALID_DICTS_REPORT_FILE_NAME = 'valid_dicts_report.csv'
HP_PRIORITY = ['阳性', '阴性', '待确认', '未检']
DIGIT_PATTERN = re.compile(r'\d')


@dataclass
class PathConfig:
    dataset_root: Path
    dataset_base_root: Path
    output_dir: Path


@dataclass
class ExamScanResult:
    exam_dir: Path
    is_valid: bool
    conflict_keys: list[str]
    merged_valid_fields: dict[str, str]
    field_values: dict[str, set[str]]


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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description='扫描检查目录 PDF，输出有效目录与冲突键统计')
    parser.add_argument('--config', type=Path, default=CONFIG_PATH, help='路径配置文件，默认使用 configs/path.yaml')
    parser.add_argument('--input-dir', type=Path, default=None, help='可选：覆盖配置中的 dataset_root')
    parser.add_argument('--dataset-base-root', type=Path, default=None, help='可选：覆盖配置中的 dataset_base_root')
    parser.add_argument('--output-dir', type=Path, default=None, help='可选：覆盖配置中的 output_dir')
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
    return PathConfig(
        dataset_root=resolved_dataset_root,
        dataset_base_root=resolved_dataset_base_root,
        output_dir=resolved_output_dir,
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


def verify_exam_validity(exam_dir: Path) -> ExamScanResult:
    pdf_files = iter_pdf_files(exam_dir)
    merged_fields: dict[str, str] = {}
    conflict_keys: set[str] = set()
    field_values: dict[str, set[str]] = {}

    for pdf_path in pdf_files:
        try:
            current_fields = extract_non_empty_fields(pdf_path)
        except Exception:
            continue

        for key, current_value in current_fields.items():
            field_values.setdefault(key, set()).add(current_value)
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


def apply_second_round_rules(results: list[ExamScanResult]) -> tuple[list[ExamScanResult], dict[str, int]]:
    stats = {
        'archiveTime': 0,
        'checkTime': 0,
        'badness': 0,
        'roomName': 0,
        'hp': 0,
        'anesthesiologistName': 0,
        'doctorName': 0,
    }
    patched_results: list[ExamScanResult] = []

    for item in results:
        if not item.conflict_keys:
            patched_results.append(item)
            continue

        new_merged_fields = dict(item.merged_valid_fields)
        new_conflict_keys = list(item.conflict_keys)

        for conflict_key in list(item.conflict_keys):
            values = item.field_values.get(conflict_key, set())
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
            elif conflict_key == 'hp':
                chosen_hp = choose_hp_value(values)
                if chosen_hp is not None:
                    new_merged_fields['hp'] = chosen_hp
                    new_conflict_keys = [key for key in new_conflict_keys if key != 'hp']
                    stats['hp'] += 1
            elif conflict_key == 'anesthesiologistName':
                new_merged_fields['anesthesiologistName'] = ''
                new_conflict_keys = [key for key in new_conflict_keys if key != 'anesthesiologistName']
                stats['anesthesiologistName'] += 1
            elif conflict_key == 'doctorName':
                new_merged_fields['doctorName'] = choose_doctor_name(values)
                new_conflict_keys = [key for key in new_conflict_keys if key != 'doctorName']
                stats['doctorName'] += 1

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


def scan_all_exam_dirs(dataset_root: Path) -> list[ExamScanResult]:
    exam_dirs = iter_exam_dirs(dataset_root)
    results: list[ExamScanResult] = []
    progress = build_progress(total=len(exam_dirs), desc='扫描检查目录')

    try:
        for exam_dir in exam_dirs:
            results.append(verify_exam_validity(exam_dir))
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
        'field_values': {key: sorted(values) for key, values in item.field_values.items()},
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
            str(key): {str(value) for value in values}
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
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open('w', encoding='utf-8-sig', newline='') as csv_file:
        writer = csv.writer(csv_file)
        writer.writerow(['exam_dir', 'is_valid', 'conflict_key_count', 'conflict_key_types'])
        for item in results:
            writer.writerow(
                [
                    str(item.exam_dir),
                    1 if item.is_valid else 0,
                    len(item.conflict_keys),
                    '|'.join(item.conflict_keys),
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
        print(f'- 有效性汇总文件：{summary_path}')
        print(f'- 有效目录键值报告文件：{report_path}')
        print('- valid_dicts_pdf.csv 第二列规则：有效=1，无效=0')


def main() -> None:
    args = parse_args()
    path_config = build_path_config(args.config, args.input_dir, args.dataset_base_root, args.output_dir)

    if not path_config.dataset_root.exists() or not path_config.dataset_root.is_dir():
        print(f'输入路径不是有效目录：{path_config.dataset_root}')
        return

    output_dir = path_config.output_dir
    round1_summary_path = output_dir / ROUND1_VALID_DICTS_SUMMARY_FILE_NAME
    round1_report_path = output_dir / ROUND1_VALID_DICTS_REPORT_FILE_NAME
    round1_cache_path = output_dir / ROUND1_CACHE_FILE_NAME

    round2_summary_path = output_dir / ROUND2_VALID_DICTS_SUMMARY_FILE_NAME
    round2_report_path = output_dir / ROUND2_VALID_DICTS_REPORT_FILE_NAME
    round2_cache_path = output_dir / ROUND2_CACHE_FILE_NAME

    legacy_summary_path = output_dir / LEGACY_VALID_DICTS_SUMMARY_FILE_NAME
    legacy_report_path = output_dir / LEGACY_VALID_DICTS_REPORT_FILE_NAME

    if round1_cache_path.exists():
        round1_results = load_cached_results(round1_cache_path)
        print(f'检测到第一轮缓存，直接读取：{round1_cache_path}')
    else:
        round1_results = scan_all_exam_dirs(path_config.dataset_root)
        save_cached_results(round1_cache_path, round1_results)
        write_valid_dicts_pdf(round1_summary_path, round1_results)
        write_valid_dicts_report(round1_report_path, round1_results)
        print(f'第一轮缓存与结果已生成：{output_dir}')

    print_summary(round1_summary_path, round1_report_path, round1_results, title='第一轮有效性确认')

    if round2_cache_path.exists():
        round2_results = load_cached_results(round2_cache_path)
        second_round_stats = None
        print(f'检测到第二轮缓存，直接读取：{round2_cache_path}')
    else:
        round2_results, second_round_stats = apply_second_round_rules(round1_results)
        save_cached_results(round2_cache_path, round2_results)
        write_valid_dicts_pdf(round2_summary_path, round2_results)
        write_valid_dicts_report(round2_report_path, round2_results)
        write_valid_dicts_pdf(legacy_summary_path, round2_results)
        write_valid_dicts_report(legacy_report_path, round2_results)
        print(f'第二轮缓存与结果已生成：{output_dir}')

    if not round2_summary_path.exists() or not round2_report_path.exists():
        write_valid_dicts_pdf(round2_summary_path, round2_results)
        write_valid_dicts_report(round2_report_path, round2_results)
    if not legacy_summary_path.exists() or not legacy_report_path.exists():
        write_valid_dicts_pdf(legacy_summary_path, round2_results)
        write_valid_dicts_report(legacy_report_path, round2_results)

    print_summary(round2_summary_path, round2_report_path, round2_results, title='第二轮有效性确认（按冲突键定制规则处理）')
    if second_round_stats is not None:
        print(f"- 第二轮按最晚时间消解 archiveTime 冲突目录数：{second_round_stats['archiveTime']}")
        print(f"- 第二轮按最晚时间消解 checkTime 冲突目录数：{second_round_stats['checkTime']}")
        print(f"- 第二轮设置 badness='有' 的目录数：{second_round_stats['badness']}")
        print(f"- 第二轮清空 roomName 的目录数：{second_round_stats['roomName']}")
        print(f"- 第二轮按优先级处理 hp 的目录数：{second_round_stats['hp']}")
        print(f"- 第二轮清空 anesthesiologistName 的目录数：{second_round_stats['anesthesiologistName']}")
        print(f"- 第二轮按规则处理 doctorName 的目录数：{second_round_stats['doctorName']}")


if __name__ == '__main__':
    main()
