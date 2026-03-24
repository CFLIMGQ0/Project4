from __future__ import annotations

import argparse
import csv
import json
from dataclasses import dataclass
from pathlib import Path


@dataclass
class BadnessConflict:
    exam_dir: str
    badness_values: list[str]
    archive_times: list[str]
    latest_archive_time: str


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description='读取 conflicted_dicts.csv 并提取 badness 冲突目录及对应 archiveTime')
    parser.add_argument(
        '--csv-path',
        type=Path,
        default=Path('/home/Lim/datasets/project4/conflicted_dicts.csv'),
        help='conflicted_dicts.csv 路径',
    )
    return parser.parse_args()


def _normalize_values(values: object) -> list[str]:
    if not isinstance(values, list):
        return []
    cleaned: list[str] = []
    for value in values:
        text = ' '.join(str(value).strip().split())
        if text and text not in cleaned:
            cleaned.append(text)
    return cleaned


def load_badness_conflicts(csv_path: Path) -> list[BadnessConflict]:
    conflicts: list[BadnessConflict] = []
    with csv_path.open('r', encoding='utf-8-sig', newline='') as csv_file:
        reader = csv.DictReader(csv_file)
        for row in reader:
            raw_map = (row.get('conflict_value_map') or '').strip()
            if not raw_map:
                continue

            try:
                conflict_value_map = json.loads(raw_map)
            except json.JSONDecodeError:
                continue

            if not isinstance(conflict_value_map, dict):
                continue

            badness_values = _normalize_values(conflict_value_map.get('badness'))
            if len(badness_values) < 2:
                continue

            archive_times = _normalize_values(conflict_value_map.get('archiveTime'))
            conflicts.append(
                BadnessConflict(
                    exam_dir=(row.get('exam_dir') or '').strip(),
                    badness_values=badness_values,
                    archive_times=archive_times,
                    latest_archive_time=(row.get('latest_archive_time') or '').strip(),
                )
            )
    return conflicts


def main() -> None:
    args = parse_args()
    csv_path = args.csv_path.expanduser()
    if not csv_path.exists():
        print(f'文件不存在：{csv_path}')
        return

    conflicts = load_badness_conflicts(csv_path)
    print(f'文件：{csv_path}')
    print(f'badness 冲突目录数：{len(conflicts)}')

    if not conflicts:
        print('没有找到 badness 冲突目录。')
        return

    print('\nbadness 冲突目录详情：')
    for idx, item in enumerate(conflicts, start=1):
        print(f'{idx}. 目录：{item.exam_dir}')
        print(f'   - badness 冲突值：{", ".join(item.badness_values)}')
        if item.archive_times:
            print(f'   - 对应 archiveTime：{", ".join(item.archive_times)}')
        else:
            print('   - 对应 archiveTime：无（CSV 明细里没有 archiveTime 冲突值）')
        print(f'   - latest_archive_time：{item.latest_archive_time or "无"}')


if __name__ == '__main__':
    main()
