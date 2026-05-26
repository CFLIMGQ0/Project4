from __future__ import annotations

import argparse
import csv
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

try:
    import yaml  # type: ignore
except ImportError:
    yaml = None


PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_CONFIG_PATH = PROJECT_ROOT / "configs" / "task2" / "path.yaml"
DEFAULT_OUTPUT_CSV = "invalid_image_names.csv"

# 要求格式：xxxx_xxxx未命名(xxx).jpg
# 示例：1722212672972_胃肠镜检查未命名(35).jpg
NAME_PATTERN = re.compile(r"^\d+_.+未命名\(\d+\)\.jpg$", re.IGNORECASE)
IMAGE_EXTENSIONS = {
    ".jpg",
    ".jpeg",
    ".png",
    ".bmp",
    ".gif",
    ".tif",
    ".tiff",
    ".webp",
}


@dataclass
class InvalidNameRecord:
    file_path: Path
    file_name: str
    reason: str


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="检查图片命名是否符合 xxxx_xxxx未命名(xxx).jpg 格式")
    parser.add_argument(
        "--config",
        type=Path,
        default=DEFAULT_CONFIG_PATH,
        help="路径配置文件，默认 configs/task2/path.yaml",
    )
    parser.add_argument(
        "--input-dir",
        type=Path,
        default=None,
        help="可选：覆盖配置中的 dataset_root",
    )
    parser.add_argument(
        "--output-csv",
        type=Path,
        default=None,
        help="可选：不合规文件输出 CSV 路径，默认输出到当前目录",
    )
    parser.add_argument(
        "--max-files",
        type=int,
        default=0,
        help="可选：仅检查前 N 张图片（0 表示全部）",
    )
    return parser.parse_args()


def load_yaml_payload(config_path: Path) -> dict[str, Any]:
    if not config_path.exists():
        raise FileNotFoundError(f"配置文件不存在: {config_path}")

    if yaml is not None:
        payload = yaml.safe_load(config_path.read_text(encoding="utf-8"))
        if not isinstance(payload, dict):
            raise ValueError(f"配置文件格式错误: {config_path}")
        return payload

    # 在未安装 PyYAML 时提供最小兼容解析（仅支持 paths: 下简单键值）
    payload: dict[str, Any] = {}
    lines = config_path.read_text(encoding="utf-8").splitlines()
    current_section: str | None = None
    for raw_line in lines:
        if not raw_line.strip() or raw_line.lstrip().startswith("#"):
            continue
        indent = len(raw_line) - len(raw_line.lstrip(" "))
        line = raw_line.strip()
        if line.endswith(":"):
            current_section = line[:-1]
            payload[current_section] = {}
            continue

        key, sep, value = line.partition(":")
        if not sep:
            continue

        cleaned_value = value.strip().strip('"').strip("'")
        if indent == 0:
            payload[key.strip()] = cleaned_value
            current_section = None
            continue

        if current_section is not None and isinstance(payload.get(current_section), dict):
            payload[current_section][key.strip()] = cleaned_value

    return payload


def resolve_dataset_root(config_path: Path, override_input_dir: Path | None) -> Path:
    if override_input_dir is not None:
        return override_input_dir.expanduser().resolve()

    payload = load_yaml_payload(config_path.expanduser())
    paths = payload.get("paths")
    if not isinstance(paths, dict) or "dataset_root" not in paths:
        raise ValueError("配置文件缺少 paths.dataset_root")

    dataset_root_raw = str(paths["dataset_root"])
    dataset_root = Path(dataset_root_raw).expanduser()
    if dataset_root.is_absolute():
        return dataset_root.resolve()

    config_dir = config_path.expanduser().resolve().parent
    return (config_dir.parent / dataset_root).resolve()


def render_progress(current: int, total: int, width: int = 40) -> str:
    if total <= 0:
        return f"[{'#' * width}] 100.0% (0/0)"
    filled = int(width * current / total)
    bar = "#" * filled + "-" * (width - filled)
    percent = current / total * 100
    return f"[{bar}] {percent:5.1f}% ({current}/{total})"


def write_progress(label: str, current: int, total: int) -> None:
    message = f"\r{label}: {render_progress(current, total)}"
    end = "\n" if current >= total else ""
    sys.stdout.write(message + end)
    sys.stdout.flush()


def update_progress(label: str, current: int, total: int, last_percent: int) -> int:
    percent = 100 if total <= 0 else int(current * 100 / total)
    if percent != last_percent or current in {0, total}:
        write_progress(label, current, total)
        return percent
    return last_percent


def collect_image_files(dataset_root: Path, max_files: int = 0) -> list[Path]:
    image_files: list[Path] = []
    for path in dataset_root.rglob("*"):
        if not path.is_file():
            continue
        if path.suffix.lower() not in IMAGE_EXTENSIONS:
            continue
        image_files.append(path)
        if max_files > 0 and len(image_files) >= max_files:
            break
    return image_files


def validate_image_name(file_path: Path) -> InvalidNameRecord | None:
    filename = file_path.name
    if not NAME_PATTERN.match(filename):
        reason = "不匹配模式: ^\\d+_.+未命名\\(\\d+\\)\\.jpg$"
        return InvalidNameRecord(file_path=file_path, file_name=filename, reason=reason)
    return None


def write_invalid_csv(records: list[InvalidNameRecord], output_csv: Path) -> None:
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    with output_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["file_path", "file_name", "reason"])
        for record in records:
            writer.writerow([str(record.file_path), record.file_name, record.reason])


def main() -> None:
    args = parse_args()
    dataset_root = resolve_dataset_root(args.config, args.input_dir)

    if not dataset_root.exists() or not dataset_root.is_dir():
        raise FileNotFoundError(f"数据目录不存在或不是目录: {dataset_root}")

    output_csv = args.output_csv.expanduser().resolve() if args.output_csv else Path.cwd() / DEFAULT_OUTPUT_CSV

    print(f"扫描目录: {dataset_root}")
    print("正在收集图片文件...")
    image_files = collect_image_files(dataset_root=dataset_root, max_files=args.max_files)

    total = len(image_files)
    print(f"待检查图片总数: {total}")
    if total == 0:
        print("未发现图片文件，检查结束。")
        return

    invalid_records: list[InvalidNameRecord] = []
    last_percent = update_progress("命名检查进度", 0, total, -1)
    for idx, image_path in enumerate(image_files, start=1):
        invalid = validate_image_name(image_path)
        if invalid is not None:
            invalid_records.append(invalid)
        last_percent = update_progress("命名检查进度", idx, total, last_percent)

    valid_count = total - len(invalid_records)
    print("\n检查完成。")
    print(f"符合格式数量: {valid_count}")
    print(f"不符合格式数量: {len(invalid_records)}")
    print(f"符合率: {valid_count / total * 100:.2f}%")

    write_invalid_csv(invalid_records, output_csv)
    print(f"不合规明细已保存: {output_csv}")

    if invalid_records:
        preview_count = min(20, len(invalid_records))
        print(f"\n前 {preview_count} 条不合规样例:")
        for record in invalid_records[:preview_count]:
            print(f"- {record.file_path}")


if __name__ == "__main__":
    main()
