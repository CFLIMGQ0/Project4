from __future__ import annotations

import argparse
import csv
import re
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from shutil import get_terminal_size

IMAGE_EXTENSIONS = {'.jpg', '.jpeg', '.png', '.bmp', '.gif', '.tif', '.tiff', '.webp', '.dcm'}

# 目标统计清单：报告标题 -> (期望报告数, 期望图像数)
TARGETS: dict[str, dict[str, dict[str, tuple[int, int]]]] = {
    '胃镜': {
        '常规白光胃镜': {
            '无痛胃镜检查报告': (784, 65056),
            '胃镜检查报告': (186, 13800),
            '一诊疗无痛胃镜报告': (5, 381),
            '职工体检胃镜(无痛)报告': (1, 82),
            '急诊胃镜下取异物报告': (2, 202),
        },
        '染色胃镜': {
            '放大染色胃镜精查报告': (291, 25184),
            '无痛胃镜(含色素内镜)报告': (51, 4293),
            '国际部无痛胃镜检查（含色素内镜）报告': (2, 182),
            '国际部胃镜检查（含色素内镜）报告': (1, 31),
        },
        '手术胃镜': {
            '胃镜手术(住院)报告': (842, 38602),
            '胃镜下切除手术报告': (124, 5100),
            '胃镜下其他手术报告': (7, 246),
            '急诊胃镜报告': (2, 49),
        },
        '超声胃镜': {
            '超声胃镜检查报告': (446, 12844),
            '无痛超声胃镜报告': (142, 6248),
        },
        '其他': {
            '胃镜下静脉曲张手术报告': (2, 92),
            '超声胃镜下手术报告': (12, 534),
        },
    },
    '肠镜': {
        '常规白光肠镜': {
            '无痛肠镜检查报告': (429, 33134),
            '肠镜检查报告': (31, 2224),
            '一诊疗无痛肠镜报告': (3, 290),
        },
        '染色肠镜': {
            '无痛肠镜(含色素内镜)报告': (3, 183),
            '国际部无痛肠镜检查（含色素内镜）报告': (1, 96),
            '国际部肠镜检查（含色素内镜）报告': (1, 45),
        },
        '手术肠镜': {
            '肠镜手术(住院)报告': (72, 5470),
            '肠镜下手术报告': (6, 466),
        },
        '超声肠镜': {
            '无痛超声肠镜报告': (1, 37),
            '超声肠镜检查报告': (1, 13),
        },
        '其他': {
            '十二指肠镜检查报告': (1, 35),
        },
    },
}


@dataclass
class PathConfig:
    dataset_base_root: Path
    report_csv_path: Path


class SimpleProgressBar:
    def __init__(self, total: int, desc: str) -> None:
        self.total = max(total, 1)
        self.current = 0
        self.desc = desc

    def update(self, step: int = 1) -> None:
        self.current = min(self.total, self.current + step)
        width = min(40, max(10, get_terminal_size((80, 20)).columns - 42))
        ratio = self.current / self.total
        done = int(width * ratio)
        bar = '=' * done + '-' * (width - done)
        print(f'\r{self.desc}: [{bar}] {self.current}/{self.total} ({ratio * 100:5.1f}%)', end='', flush=True)
        if self.current >= self.total:
            print()

    def close(self) -> None:
        return


def parse_simple_yaml_mapping(yaml_path: Path) -> dict[str, str]:
    text = yaml_path.read_text(encoding='utf-8')
    in_paths = False
    result: dict[str, str] = {}
    for raw_line in text.splitlines():
        line = raw_line.split('#', 1)[0].rstrip()
        if not line.strip():
            continue
        if re.match(r'^\s*paths\s*:\s*$', line):
            in_paths = True
            continue
        if not in_paths:
            continue
        if re.match(r'^\S', raw_line):
            break

        m = re.match(r'^\s{2,}([A-Za-z0-9_]+)\s*:\s*(.+?)\s*$', line)
        if not m:
            continue
        key, value = m.group(1), m.group(2)
        value = value.strip().strip('"').strip("'")
        result[key] = value
    if not result:
        raise ValueError(f'无法从 {yaml_path} 解析 paths 配置')
    return result


def resolve_path(raw_path: str, config_dir: Path) -> Path:
    path = Path(raw_path).expanduser()
    if path.is_absolute():
        return path
    return (config_dir.parent / path).resolve()


def build_path_config(config_path: Path, report_csv_override: Path | None) -> PathConfig:
    config_path = config_path.expanduser().resolve()
    paths_payload = parse_simple_yaml_mapping(config_path)
    dataset_base_root = resolve_path(paths_payload['dataset_base_root'], config_path.parent)
    if report_csv_override is not None:
        report_csv_path = report_csv_override.expanduser().resolve()
    else:
        report_csv_raw = paths_payload.get('valid_dicts_report_csv', 'valid_dicts_report.csv')
        report_csv_path = resolve_path(report_csv_raw, config_path.parent)
    return PathConfig(dataset_base_root=dataset_base_root, report_csv_path=report_csv_path)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description='按指定清单统计胃镜/肠镜报告数和图像数，并与目标值对比')
    parser.add_argument('--config', type=Path, default=Path('configs/path.yaml'), help='路径配置文件，默认 configs/path.yaml')
    parser.add_argument('--report-csv', type=Path, default=None, help='可选：覆盖 valid_dicts_report_csv')
    return parser.parse_args()


def load_rows(report_csv_path: Path) -> list[dict[str, str]]:
    if not report_csv_path.is_file():
        raise FileNotFoundError(f'未找到报告汇总文件：{report_csv_path}')
    with report_csv_path.open('r', encoding='utf-8-sig', newline='') as f:
        reader = csv.DictReader(f)
        rows = list(reader)
        fieldnames = set(reader.fieldnames or [])
    need = {'exam_dir', 'reportTitle'}
    miss = sorted(need - fieldnames)
    if miss:
        raise KeyError(f'CSV 缺少字段：{"、".join(miss)}')
    return rows


def count_exam_images(exam_dir: Path) -> int:
    img_dir = exam_dir / 'img'
    if not img_dir.is_dir():
        return 0
    return sum(1 for p in img_dir.iterdir() if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS)


def flatten_targets() -> dict[str, tuple[str, str, int, int]]:
    flat: dict[str, tuple[str, str, int, int]] = {}
    for organ, groups in TARGETS.items():
        for group, items in groups.items():
            for title, (exp_reports, exp_images) in items.items():
                flat[title] = (organ, group, exp_reports, exp_images)
    return flat


def compute_stats(rows: list[dict[str, str]], dataset_base_root: Path) -> dict[str, tuple[int, int]]:
    flat = flatten_targets()
    report_counter: Counter[str] = Counter()
    image_counter: Counter[str] = Counter()
    image_cache: dict[Path, int] = {}

    progress = SimpleProgressBar(total=len(rows), desc='统计清单内 reportTitle')
    try:
        for row in rows:
            title = str(row.get('reportTitle', '')).strip()
            if title not in flat:
                progress.update(1)
                continue
            report_counter[title] += 1

            exam_dir_raw = str(row.get('exam_dir', '')).strip()
            exam_dir = Path(exam_dir_raw).expanduser()
            if not exam_dir.is_absolute():
                exam_dir = (dataset_base_root / exam_dir).resolve()
            if exam_dir not in image_cache:
                image_cache[exam_dir] = count_exam_images(exam_dir)
            image_counter[title] += image_cache[exam_dir]
            progress.update(1)
    finally:
        progress.close()

    result: dict[str, tuple[int, int]] = {}
    for title in flat:
        result[title] = (report_counter.get(title, 0), image_counter.get(title, 0))
    return result


def print_comparison(stats: dict[str, tuple[int, int]]) -> None:
    for organ, groups in TARGETS.items():
        print(f'\n==================== {organ} ====================')
        organ_exp_r = organ_exp_i = organ_act_r = organ_act_i = 0
        for group, items in groups.items():
            print(f'\n【{group}】')
            group_exp_r = group_exp_i = group_act_r = group_act_i = 0
            for title, (exp_r, exp_i) in items.items():
                act_r, act_i = stats.get(title, (0, 0))
                dr = act_r - exp_r
                di = act_i - exp_i
                print(f'- {title}\n  期望: 报告 {exp_r}，图像 {exp_i} | 实际: 报告 {act_r}，图像 {act_i} | 差值: 报告 {dr:+d}，图像 {di:+d}')
                group_exp_r += exp_r
                group_exp_i += exp_i
                group_act_r += act_r
                group_act_i += act_i

            print(
                f'  小计 -> 期望: 报告 {group_exp_r}，图像 {group_exp_i} | '
                f'实际: 报告 {group_act_r}，图像 {group_act_i} | '
                f'差值: 报告 {group_act_r - group_exp_r:+d}，图像 {group_act_i - group_exp_i:+d}'
            )
            organ_exp_r += group_exp_r
            organ_exp_i += group_exp_i
            organ_act_r += group_act_r
            organ_act_i += group_act_i

        print(
            f'\n{organ}总计 -> 期望: 报告 {organ_exp_r}，图像 {organ_exp_i} | '
            f'实际: 报告 {organ_act_r}，图像 {organ_act_i} | '
            f'差值: 报告 {organ_act_r - organ_exp_r:+d}，图像 {organ_act_i - organ_exp_i:+d}'
        )


def main() -> None:
    args = parse_args()
    cfg = build_path_config(args.config, args.report_csv)
    rows = load_rows(cfg.report_csv_path)
    print(f'报告记录总数：{len(rows)}')
    print(f'统计文件：{cfg.report_csv_path}')
    stats = compute_stats(rows, cfg.dataset_base_root)
    print_comparison(stats)


if __name__ == '__main__':
    main()
