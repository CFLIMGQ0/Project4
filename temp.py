from __future__ import annotations

import argparse
import csv
import datetime as dt
from dataclasses import dataclass
from pathlib import Path

from statistics import extract_pdf_fields


@dataclass
class PdfSnapshot:
    pdf_path: Path
    fields: dict[str, str]
    archive_time_raw: str
    archive_time_parsed: dt.datetime | None
    non_empty_count: int


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description='检查 archiveTime 空值，并在冲突目录中比较最晚报告与其他报告的非空键数量。'
    )
    parser.add_argument(
        '--dataset-root',
        type=Path,
        default=Path('/home/Lim/datasets/project4/main_data'),
        help='主数据根目录（默认 /home/Lim/datasets/project4/main_data）',
    )
    parser.add_argument(
        '--conflicted-csv',
        type=Path,
        default=Path('/home/Lim/datasets/project4/conflicted_dicts.csv'),
        help='冲突检查目录 CSV（默认 /home/Lim/datasets/project4/conflicted_dicts.csv）',
    )
    return parser.parse_args()


def normalize_text(value: object) -> str:
    if value is None:
        return ''
    return ' '.join(str(value).strip().split())


def parse_archive_time(value: str) -> dt.datetime | None:
    text = normalize_text(value)
    if not text:
        return None

    formats = [
        '%Y-%m-%d %H:%M:%S',
        '%Y/%m/%d %H:%M:%S',
        '%Y-%m-%d %H:%M',
        '%Y/%m/%d %H:%M',
        '%Y-%m-%d',
        '%Y/%m/%d',
        '%Y%m%d%H%M%S',
        '%Y%m%d',
    ]
    for fmt in formats:
        try:
            return dt.datetime.strptime(text, fmt)
        except ValueError:
            continue
    return None


def iter_exam_dirs(dataset_root: Path) -> list[Path]:
    exam_dirs: list[Path] = []
    if not dataset_root.is_dir():
        return exam_dirs
    for patient_dir in sorted(path for path in dataset_root.iterdir() if path.is_dir()):
        exam_dirs.extend(sorted(path for path in patient_dir.iterdir() if path.is_dir()))
    return exam_dirs


def iter_pdf_files(exam_dir: Path) -> list[Path]:
    pdf_dir = exam_dir / 'pdf'
    if not pdf_dir.is_dir():
        return []
    return sorted(path for path in pdf_dir.rglob('*.pdf') if path.is_file())


def collect_pdf_snapshots(exam_dir: Path) -> tuple[list[PdfSnapshot], list[tuple[Path, str]]]:
    snapshots: list[PdfSnapshot] = []
    parse_errors: list[tuple[Path, str]] = []

    for pdf_path in iter_pdf_files(exam_dir):
        try:
            fields = extract_pdf_fields(pdf_path)
        except Exception as exc:
            parse_errors.append((pdf_path, str(exc)))
            continue

        archive_time_raw = normalize_text(fields.get('archiveTime', ''))
        snapshots.append(
            PdfSnapshot(
                pdf_path=pdf_path,
                fields=fields,
                archive_time_raw=archive_time_raw,
                archive_time_parsed=parse_archive_time(archive_time_raw),
                non_empty_count=sum(1 for value in fields.values() if normalize_text(value)),
            )
        )

    return snapshots, parse_errors


def choose_truth_pdf(snapshots: list[PdfSnapshot]) -> PdfSnapshot | None:
    if not snapshots:
        return None

    # 规则：
    # 1) 先取 archiveTime 最新的一组（可解析时间优先，无法解析时退化为原始字符串排序）；
    # 2) 若“最新 archiveTime”有多个，取非空键数量最多者；
    # 3) 再按路径兜底，保证结果稳定。
    best_time_key = max(
        (
            1 if item.archive_time_parsed is not None else 0,
            item.archive_time_parsed or dt.datetime.min,
            item.archive_time_raw,
        )
        for item in snapshots
    )
    latest_snapshots = [
        item
        for item in snapshots
        if (
            1 if item.archive_time_parsed is not None else 0,
            item.archive_time_parsed or dt.datetime.min,
            item.archive_time_raw,
        )
        == best_time_key
    ]
    return max(
        latest_snapshots,
        key=lambda item: (
            item.non_empty_count,
            str(item.pdf_path),
        ),
    )


def load_conflicted_exam_dirs(conflicted_csv: Path, dataset_root: Path) -> tuple[list[Path], list[str]]:
    errors: list[str] = []
    exam_dirs: list[Path] = []
    seen: set[Path] = set()

    if not conflicted_csv.is_file():
        errors.append(f'冲突 CSV 不存在：{conflicted_csv}')
        return exam_dirs, errors

    with conflicted_csv.open('r', encoding='utf-8-sig', newline='') as csv_file:
        reader = csv.DictReader(csv_file)
        if not reader.fieldnames:
            errors.append(f'冲突 CSV 表头为空：{conflicted_csv}')
            return exam_dirs, errors

        for idx, row in enumerate(reader, start=2):
            exam_dir_text = normalize_text(row.get('exam_dir', ''))
            patient_id = normalize_text(row.get('patient_id', ''))
            exam_id = normalize_text(row.get('exam_id', ''))

            candidate: Path | None = None
            if exam_dir_text:
                raw_path = Path(exam_dir_text).expanduser()
                candidate = raw_path if raw_path.is_absolute() else (dataset_root / raw_path)
            elif patient_id and exam_id:
                candidate = dataset_root / patient_id / exam_id

            if candidate is None:
                errors.append(f'第 {idx} 行无法解析检查目录（缺少 exam_dir 或 patient_id/exam_id）')
                continue

            resolved = candidate.resolve()
            if resolved in seen:
                continue
            seen.add(resolved)
            exam_dirs.append(resolved)

    return exam_dirs, errors


def check_archive_time_missing(dataset_root: Path) -> tuple[int, list[str], int, int]:
    exam_dirs = iter_exam_dirs(dataset_root)
    all_empty_exam_dirs: list[str] = []
    parse_error_count = 0
    total_pdf_count = 0

    for exam_dir in exam_dirs:
        snapshots, parse_errors = collect_pdf_snapshots(exam_dir)
        parse_error_count += len(parse_errors)
        total_pdf_count += len(snapshots)
        if not snapshots:
            continue

        if all(not snapshot.archive_time_raw for snapshot in snapshots):
            all_empty_exam_dirs.append(str(exam_dir))

    return len(exam_dirs), all_empty_exam_dirs, parse_error_count, total_pdf_count


def check_conflicted_non_empty_count(conflicted_exam_dirs: list[Path]) -> tuple[int, list[str], int, int]:
    exceed_locations: list[str] = []
    parse_error_count = 0
    checked_exam_count = 0

    for exam_dir in conflicted_exam_dirs:
        if not exam_dir.is_dir():
            exceed_locations.append(f'{exam_dir} （目录不存在）')
            continue

        snapshots, parse_errors = collect_pdf_snapshots(exam_dir)
        parse_error_count += len(parse_errors)
        if len(snapshots) < 2:
            continue

        truth_snapshot = choose_truth_pdf(snapshots)
        if truth_snapshot is None:
            continue

        checked_exam_count += 1
        for snapshot in snapshots:
            if snapshot.pdf_path == truth_snapshot.pdf_path:
                continue
            if snapshot.non_empty_count > truth_snapshot.non_empty_count:
                exceed_locations.append(
                    ' | '.join(
                        [
                            f'检查目录: {exam_dir}',
                            f'真值PDF: {truth_snapshot.pdf_path} (archiveTime={truth_snapshot.archive_time_raw or "<空>"}, 非空键={truth_snapshot.non_empty_count})',
                            f'对比PDF: {snapshot.pdf_path} (archiveTime={snapshot.archive_time_raw or "<空>"}, 非空键={snapshot.non_empty_count})',
                        ]
                    )
                )

    return checked_exam_count, exceed_locations, parse_error_count, len(conflicted_exam_dirs)


def main() -> None:
    args = parse_args()
    dataset_root = args.dataset_root.expanduser().resolve()
    conflicted_csv = args.conflicted_csv.expanduser().resolve()

    print('==== 任务1：检查所有检查目录的 archiveTime 是否非空 ====')
    if not dataset_root.is_dir():
        print(f'数据目录不存在：{dataset_root}')
    else:
        exam_count, all_empty_exam_dirs, parse_errors, total_pdf_count = check_archive_time_missing(dataset_root)
        print(f'数据根目录：{dataset_root}')
        print(f'检查目录数量：{exam_count}')
        print(f'成功解析 PDF 数量：{total_pdf_count}')
        print(f'“目录内所有 PDF 的 archiveTime 都为空”的检查目录数量：{len(all_empty_exam_dirs)}')
        print(f'PDF 解析失败数量：{parse_errors}')
        if all_empty_exam_dirs:
            print('\n符合条件的检查目录路径：')
            for location in all_empty_exam_dirs:
                print(f'- {location}')

    print('\n==== 任务2：冲突目录中比较真值报告与其他报告的非空键数量 ====')
    conflicted_exam_dirs, csv_errors = load_conflicted_exam_dirs(conflicted_csv, dataset_root)
    print(f'冲突 CSV：{conflicted_csv}')
    print(f'冲突检查目录条目数（去重后）：{len(conflicted_exam_dirs)}')

    if csv_errors:
        print('冲突 CSV 解析告警：')
        for error in csv_errors:
            print(f'- {error}')

    checked_exam_count, exceed_locations, parse_errors, _ = check_conflicted_non_empty_count(conflicted_exam_dirs)
    print(f'进入比较流程的检查目录数（至少2份可解析报告）：{checked_exam_count}')
    print(f'发现“其他报告非空键数 > 真值报告非空键数”的次数：{len(exceed_locations)}')
    print(f'PDF 解析失败数量：{parse_errors}')

    if exceed_locations:
        print('\n具体位置：')
        for location in exceed_locations:
            print(f'- {location}')


if __name__ == '__main__':
    main()
