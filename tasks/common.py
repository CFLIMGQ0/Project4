from __future__ import annotations

import csv
import re
import unicodedata
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from shutil import get_terminal_size
from typing import Any, Iterable

try:
    from tqdm import tqdm  # type: ignore
except ImportError:  # pragma: no cover
    tqdm = None


REQUIRED_COLUMNS = {"reportTitle", "watchResult"}
LEGACY_PATH_ALIASES: dict[str, str] = {
    "valid_dicts_report_for_task2.csv": "valid_dicts_report_for task2.csv",
    "valid_dicts_report_for task2.csv": "valid_dicts_report_for_task2.csv",
}
OPTIONAL_ALIASES: dict[str, list[str]] = {
    "patient_id": ["patient_id", "patientId", "pid", "patient"],
    "exam_dir": ["exam_dir", "examDir", "exam_directory", "check_dir"],
    "reportTitle": ["reportTitle", "report_title", "title"],
    "watchResult": ["watchResult", "watch_result", "diagnosis", "result"],
    "img_num": ["img_num", "imgNum", "image_num", "image_count"],
    "watch": ["watch", "watch_text"],
    "specimen": ["specimen", "biopsy", "specimen_text"],
    "hp": ["hp", "hp_status"],
}

COMMON_OUTPUT_FIELDS = [
    "patient_id",
    "exam_dir",
    "reportTitle",
    "watchResult",
    "watch",
    "specimen",
    "hp",
    "img_num",
]

GASTROSCOPY_HINTS = [
    "胃镜",
    "食管",
    "胃窦",
    "胃体",
    "胃角",
    "贲门",
    "幽门",
    "十二指肠",
    "无痛胃镜",
    "超声胃镜",
]
COLONOSCOPY_HINTS = [
    "肠镜",
    "结肠",
    "直肠",
    "回盲",
    "乙状结肠",
    "升结肠",
    "降结肠",
    "横结肠",
    "盲肠",
]
UNCERTAIN_TOKENS = ["待病理", "性质待定", "考虑", "可能", "？", "?", "待定"]
NORMAL_PATTERNS = [
    "无异常发现",
    "未见异常",
    "未见明显异常",
    "检查无异常发现",
    "结肠镜检查未见明显异常",
    "结肠镜检查无异常发现",
]
POLYP_PATTERNS = ["息肉", "结肠息肉", "直肠息肉", "结直肠息肉"]
LESION_TOKENS = ["息肉", "肿物", "炎", "憩室", "溃疡", "糜烂", "癌", "出血", "狭窄", "病变"]


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
        bar = "=" * done + "-" * (width - done)
        print(f"\r{self.desc}: [{bar}] {self.current}/{self.total} ({ratio * 100:5.1f}%)", end="", flush=True)
        if self.current >= self.total:
            print()

    def close(self) -> None:
        return


def build_progress(total: int, desc: str):
    if tqdm is not None:
        return tqdm(total=total, desc=desc, unit="条")
    return SimpleProgressBar(total=total, desc=desc)


def detect_column(fieldnames: list[str], aliases: list[str]) -> str | None:
    for alias in aliases:
        if alias in fieldnames:
            return alias
    return None


def resolve_columns(fieldnames: list[str]) -> dict[str, str | None]:
    resolved: dict[str, str | None] = {}
    for canonical_name, aliases in OPTIONAL_ALIASES.items():
        resolved[canonical_name] = detect_column(fieldnames, aliases)
    for required in REQUIRED_COLUMNS:
        if resolved.get(required) is None:
            raise KeyError(f"输入 CSV 缺少必需字段：{required}（可接受别名：{OPTIONAL_ALIASES[required]}）")
    return resolved


def resolve_compatible_path(path: Path) -> Path:
    alias_name = LEGACY_PATH_ALIASES.get(path.name)
    if path.is_file() or alias_name is None:
        return path
    alias_path = path.with_name(alias_name)
    return alias_path if alias_path.is_file() else path


def load_rows(report_csv_path: Path) -> tuple[list[dict[str, str]], dict[str, str | None]]:
    resolved_path = resolve_compatible_path(report_csv_path)
    if not resolved_path.is_file():
        raise FileNotFoundError(f"未找到报告总表：{report_csv_path}")
    with resolved_path.open("r", encoding="utf-8-sig", newline="") as file:
        reader = csv.DictReader(file)
        rows = list(reader)
        fieldnames = reader.fieldnames or []
    return rows, resolve_columns(fieldnames)


def write_csv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8-sig", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        if rows:
            writer.writerows(rows)


def normalize_text(text: str) -> str:
    normalized = unicodedata.normalize("NFKC", text or "")
    normalized = normalized.replace("\u3000", " ")
    normalized = re.sub(r"\s+", "", normalized)
    normalized = normalized.replace("（", "(").replace("）", ")")
    normalized = normalized.replace("【", "[").replace("】", "]")
    normalized = normalized.replace("，", ",").replace("；", ";").replace("：", ":")
    return normalized.lower()


def contains_any(text: str, patterns: Iterable[str]) -> bool:
    return any(pattern in text for pattern in patterns)


def to_int(value: Any) -> int:
    if value is None:
        return 0
    cleaned = re.sub(r"[^0-9]", "", str(value))
    return int(cleaned) if cleaned else 0


def get_value(row: dict[str, str], col_name: str | None) -> str:
    if col_name is None:
        return ""
    return str(row.get(col_name, "")).strip()


def derive_patient_id_from_exam_dir(exam_dir: str) -> str:
    exam_path = Path(str(exam_dir).strip())
    if not exam_path.name:
        return ""
    if exam_path.parent.name:
        return exam_path.parent.name
    return exam_path.name


def is_gastroscopy_record(report_title_norm: str, watch_result_norm: str) -> bool:
    has_gastro = contains_any(report_title_norm, GASTROSCOPY_HINTS) or contains_any(watch_result_norm, GASTROSCOPY_HINTS)
    has_colon = contains_any(report_title_norm, COLONOSCOPY_HINTS) or contains_any(watch_result_norm, COLONOSCOPY_HINTS)
    return has_gastro and not has_colon


def is_colonoscopy_record(report_title_norm: str, watch_result_norm: str) -> bool:
    has_colon = contains_any(report_title_norm, COLONOSCOPY_HINTS) or contains_any(watch_result_norm, COLONOSCOPY_HINTS)
    has_gastro = contains_any(report_title_norm, GASTROSCOPY_HINTS) or contains_any(watch_result_norm, GASTROSCOPY_HINTS)
    return has_colon and not has_gastro


def build_common_output_row(
    *,
    row: dict[str, str],
    columns: dict[str, str | None],
) -> dict[str, Any]:
    exam_dir = get_value(row, columns.get("exam_dir"))
    patient_id = get_value(row, columns.get("patient_id")) or derive_patient_id_from_exam_dir(exam_dir)
    return {
        "patient_id": patient_id,
        "exam_dir": exam_dir,
        "reportTitle": get_value(row, columns.get("reportTitle")),
        "watchResult": get_value(row, columns.get("watchResult")),
        "watch": get_value(row, columns.get("watch")),
        "specimen": get_value(row, columns.get("specimen")),
        "hp": get_value(row, columns.get("hp")),
        "img_num": to_int(get_value(row, columns.get("img_num"))),
    }


@dataclass
class SelectionResult:
    rows: list[dict[str, Any]]
    fieldnames: list[str]
    total_candidates: int
    selected_count: int
    selected_image_sum: int
    positive_counter: Counter[str]
    exclude_counter: Counter[str]
