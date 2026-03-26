from __future__ import annotations

import argparse
import csv
import re
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from shutil import copy2, get_terminal_size, rmtree

from check_pdf import CONFIG_PATH, load_yaml_config

try:
    from tqdm import tqdm  # type: ignore
except ImportError:
    tqdm = None

IMAGE_SUFFIXES = {'.jpg', '.jpeg', '.png', '.bmp', '.gif', '.tif', '.tiff', '.webp'}


@dataclass
class PathConfig:
    dataset_root: Path
    report_csv_path: Path
    output_dir: Path


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


def build_progress(total: int, desc: str):
    if tqdm is not None:
        return tqdm(total=total, desc=desc, unit='项')
    return SimpleProgressBar(total=total, desc=desc)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description='按 reportTitle 类型抽取图像样本（每类最多 10 张）')
    parser.add_argument('--config', type=Path, default=CONFIG_PATH, help='路径配置文件，默认 configs/path.yaml')
    parser.add_argument(
        '--report-csv',
        type=Path,
        default=None,
        help='可选：覆盖配置中的 valid_dicts_report_csv 路径（未配置时默认 dataset_base_root/valid_dicts_report.csv）',
    )
    parser.add_argument('--output-dir', type=Path, default=None, help='可选：覆盖配置中的 output_dir')
    parser.add_argument('--max-per-type', type=int, default=10, help='每类 reportTitle 最多抽取图像数量，默认 10')
    parser.add_argument(
        '--run-dir-name',
        type=str,
        default='report_title_samples',
        help='输出运行目录名称（会创建在 output_dir 下），默认 report_title_samples',
    )
    parser.add_argument('--clear-output', action='store_true', help='若输出运行目录已存在，先清空再重新生成')
    return parser.parse_args()


def build_path_config(config_path: Path, report_csv_override: Path | None, output_dir_override: Path | None) -> PathConfig:
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

    dataset_root = resolve_path(str(paths_payload['dataset_root']))
    dataset_base_root = resolve_path(str(paths_payload['dataset_base_root']))
    report_csv_config = paths_payload.get('valid_dicts_report_csv')
    report_csv_path = (
        report_csv_override.expanduser().resolve()
        if report_csv_override is not None
        else resolve_path(str(report_csv_config)) if report_csv_config else (dataset_base_root / 'valid_dicts_report.csv').resolve()
    )
    output_dir = (
        output_dir_override.expanduser().resolve()
        if output_dir_override is not None
        else resolve_path(str(paths_payload['output_dir']))
    )
    return PathConfig(dataset_root=dataset_root, report_csv_path=report_csv_path, output_dir=output_dir)


def load_report_rows(report_csv_path: Path) -> list[dict[str, str]]:
    if not report_csv_path.is_file():
        raise FileNotFoundError(f'未找到报告汇总文件：{report_csv_path}')

    with report_csv_path.open('r', encoding='utf-8-sig', newline='') as f:
        reader = csv.DictReader(f)
        rows = list(reader)
        fieldnames = reader.fieldnames or []

    required = {'exam_dir', 'reportTitle'}
    missing = required - set(fieldnames)
    if missing:
        missing_fields = '、'.join(sorted(missing))
        raise KeyError(f'{report_csv_path} 中缺少必需字段：{missing_fields}')
    return rows


def sanitize_dir_name(name: str) -> str:
    cleaned = re.sub(r'[\\/:*?"<>|]+', '_', name.strip())
    cleaned = re.sub(r'\s+', ' ', cleaned)
    return cleaned[:120] if cleaned else '未命名类型'


def resolve_exam_dir(raw_exam_dir: str, dataset_root: Path) -> Path:
    exam_dir_path = Path(raw_exam_dir).expanduser()
    if exam_dir_path.is_absolute():
        return exam_dir_path
    return (dataset_root / exam_dir_path).resolve()


def iter_images(img_dir: Path) -> list[Path]:
    if not img_dir.is_dir():
        return []
    return sorted(path for path in img_dir.iterdir() if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES)


def collect_images_by_title(rows: list[dict[str, str]], dataset_root: Path, max_per_type: int) -> tuple[dict[str, list[Path]], list[str]]:
    images_by_title: dict[str, list[Path]] = defaultdict(list)
    warnings: list[str] = []

    progress = build_progress(total=len(rows), desc='按 reportTitle 收集图片')
    try:
        for row in rows:
            title = str(row.get('reportTitle', '')).strip()
            if not title or len(images_by_title[title]) >= max_per_type:
                progress.update(1)
                continue

            exam_dir = resolve_exam_dir(str(row.get('exam_dir', '')), dataset_root)
            img_dir = exam_dir / 'img'
            image_paths = iter_images(img_dir)
            if not image_paths:
                warnings.append(f'未找到图片：{img_dir}')
                progress.update(1)
                continue

            for image_path in image_paths:
                if len(images_by_title[title]) >= max_per_type:
                    break
                if image_path not in images_by_title[title]:
                    images_by_title[title].append(image_path)
            progress.update(1)
    finally:
        progress.close()

    return images_by_title, warnings


def ensure_run_dir(run_dir: Path, clear_output: bool) -> None:
    if run_dir.exists() and clear_output:
        rmtree(run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)


def export_images(images_by_title: dict[str, list[Path]], run_dir: Path) -> tuple[int, int]:
    copied_count = 0
    dir_count = 0
    titles = sorted(images_by_title.keys())

    progress = build_progress(total=len(titles), desc='导出图片样本')
    try:
        for title in titles:
            image_paths = images_by_title[title]
            if not image_paths:
                progress.update(1)
                continue

            title_dir = run_dir / sanitize_dir_name(title)
            title_dir.mkdir(parents=True, exist_ok=True)
            dir_count += 1

            for idx, image_path in enumerate(image_paths, start=1):
                target_name = f'{idx:02d}_{image_path.name}'
                copy2(image_path, title_dir / target_name)
                copied_count += 1
            progress.update(1)
    finally:
        progress.close()
    return dir_count, copied_count


def print_summary(run_dir: Path, images_by_title: dict[str, list[Path]], warnings: list[str]) -> None:
    sampled_titles = sorted((title, len(paths)) for title, paths in images_by_title.items() if paths)
    print(f'输出运行目录：{run_dir}')
    print(f'共导出 {len(sampled_titles)} 个 reportTitle 类型。')
    for idx, (title, count) in enumerate(sampled_titles, start=1):
        print(f'{idx}. {title}: {count} 张')

    if warnings:
        print(f'\n提示：共有 {len(warnings)} 条记录未找到图片目录或图片文件，仅展示前 20 条：')
        for item in warnings[:20]:
            print(f'- {item}')


def main() -> None:
    args = parse_args()
    if args.max_per_type <= 0:
        raise ValueError('--max-per-type 必须大于 0')

    config = build_path_config(args.config, args.report_csv, args.output_dir)
    run_dir = config.output_dir / args.run_dir_name

    rows = load_report_rows(config.report_csv_path)
    print(f'报告记录总数：{len(rows)}')

    ensure_run_dir(run_dir=run_dir, clear_output=args.clear_output)
    images_by_title, warnings = collect_images_by_title(rows=rows, dataset_root=config.dataset_root, max_per_type=args.max_per_type)
    dir_count, copied_count = export_images(images_by_title=images_by_title, run_dir=run_dir)

    print(f'实际生成目录数：{dir_count}，复制图片总数：{copied_count}')
    print_summary(run_dir=run_dir, images_by_title=images_by_title, warnings=warnings)


if __name__ == '__main__':
    main()
