from __future__ import annotations

import argparse
import csv
import re
import unicodedata
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from shutil import get_terminal_size
from typing import Any


SCRIPT_DIR = Path(__file__).resolve().parent


def resolve_default_config_path() -> Path:
    candidates = (
        SCRIPT_DIR / "configs" / "path.yaml",
        SCRIPT_DIR.parent / "configs" / "path.yaml",
    )
    for candidate in candidates:
        if candidate.exists():
            return candidate
    return candidates[0]


CONFIG_PATH = resolve_default_config_path()
PROJECT_ROOT = CONFIG_PATH.parent.parent if CONFIG_PATH.exists() else SCRIPT_DIR

try:
    from tqdm import tqdm  # type: ignore
except ImportError:  # pragma: no cover
    tqdm = None


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


def load_yaml_config(config_path: Path) -> dict[str, Any]:
    if not config_path.exists():
        raise FileNotFoundError(f"路径配置文件不存在：{config_path}")

    lines = config_path.read_text(encoding="utf-8").splitlines()
    payload: dict[str, Any] = {}
    current_section: str | None = None
    for raw_line in lines:
        line_without_comment = raw_line.split("#", 1)[0]
        if not line_without_comment.strip():
            continue

        indent = len(line_without_comment) - len(line_without_comment.lstrip(" "))
        line = line_without_comment.strip()
        if line.endswith(":"):
            current_section = line[:-1]
            payload[current_section] = {}
            continue

        key, sep, value = line.partition(":")
        if not sep:
            raise ValueError(f"无法解析配置行：{raw_line}")

        cleaned_value = value.strip().strip('"').strip("'")
        if indent == 0:
            payload[key.strip()] = cleaned_value
            current_section = None
            continue

        if current_section is None:
            raise ValueError(f"发现未归属分组的缩进行：{raw_line}")
        section_payload = payload.setdefault(current_section, {})
        if not isinstance(section_payload, dict):
            raise ValueError(f"配置分组格式错误：{current_section}")
        section_payload[key.strip()] = cleaned_value
    return payload


@dataclass
class PathConfig:
    report_csv_path: Path
    output_dir: Path


@dataclass
class SelectionSummary:
    input_row_count: int
    report_csv_path: Path
    output_dir: Path
    gastro_csv: Path
    colon_binary_csv: Path
    gastro_total: int
    gastro_selected: int
    gastro_img_sum: int
    gastro_label_counter: Counter[str]
    gastro_exclude_counter: Counter[str]
    colon_total: int
    colon_selected: int
    colon_img_sum: int
    colon_binary_counter: Counter[str]
    colon_binary_exclude_counter: Counter[str]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="生成胃镜多标签与肠镜二分类任务筛选 CSV")
    parser.add_argument("--config", type=Path, default=CONFIG_PATH, help="路径配置文件，默认自动查找 configs/path.yaml")
    parser.add_argument("--report-csv", type=Path, default=None, help="可选：覆盖配置中的 valid_dicts_report_csv")
    parser.add_argument("--output-dir", type=Path, default=None, help="可选：覆盖输出目录，默认写入数据集根目录下的 task_data")
    return parser.parse_args()


def build_path_config(config_path: Path, report_csv_override: Path | None, output_dir_override: Path | None) -> PathConfig:
    payload = load_yaml_config(config_path.expanduser())
    paths_payload = payload.get("paths")
    if not isinstance(paths_payload, dict):
        raise ValueError("path.yaml 必须包含 paths 分组")

    config_dir = config_path.expanduser().resolve().parent

    def resolve_path(raw_path: str) -> Path:
        path = Path(raw_path).expanduser()
        if path.is_absolute():
            return path
        return (config_dir.parent / path).resolve()

    report_csv_config = paths_payload.get("valid_dicts_report_csv")
    if report_csv_config is None and report_csv_override is None:
        raise ValueError("path.yaml 缺少 paths.valid_dicts_report_csv")

    report_csv_path = (
        report_csv_override.expanduser().resolve()
        if report_csv_override is not None
        else resolve_path(str(report_csv_config))
    )

    if output_dir_override is not None:
        output_dir = output_dir_override.expanduser().resolve()
    else:
        dataset_base_root_config = paths_payload.get("dataset_base_root")
        if dataset_base_root_config:
            output_dir = resolve_path(str(dataset_base_root_config)) / "task_data"
        else:
            output_dir = PROJECT_ROOT / "datasets" / "task_data"

    return PathConfig(report_csv_path=report_csv_path, output_dir=output_dir)


REQUIRED_COLUMNS = {"reportTitle", "watchResult"}
OPTIONAL_ALIASES: dict[str, list[str]] = {
    "exam_dir": ["exam_dir", "examDir", "exam_directory", "check_dir"],
    "reportTitle": ["reportTitle", "report_title", "title"],
    "watchResult": ["watchResult", "watch_result", "diagnosis", "result"],
    "img_num": ["img_num", "imgNum", "image_num", "image_count"],
}


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


def normalize_text(text: str) -> str:
    normalized = unicodedata.normalize("NFKC", text or "")
    normalized = normalized.replace("\u3000", " ")
    normalized = re.sub(r"\s+", "", normalized)
    normalized = normalized.replace("（", "(").replace("）", ")")
    normalized = normalized.replace("【", "[").replace("】", "]")
    normalized = normalized.replace("，", ",").replace("；", ";").replace("：", ":")
    return normalized.lower()


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

GASTRO_LABEL_RULES: dict[str, list[str]] = {
    "label_esophageal_smt": [
        "食管smt",
        "食管黏膜下隆起",
        "食管隆起性病变",
        "食管smt(来源于黏膜肌层)",
        "食管smt(来源于固有肌层)",
        "食管smt(来源于黏膜下层)",
        "食管黏膜下肿物",
    ],
    "label_esophageal_mucosal_or_tumor": [
        "食管黏膜病变",
        "食管肿物",
        "食管黏膜病变(待病理)",
        "食管黏膜病变(性质待定)",
        "食管肿物(待病理)",
        "食管占位",
        "食管新生物",
    ],
    "label_gastritis": [
        "慢性胃炎",
        "慢性非活动性胃炎",
        "慢性活动性胃炎",
        "萎缩性胃炎",
        "糜烂性胃炎",
        "浅表性胃炎",
        "胆汁反流性胃炎",
        "胃炎",
        "c1",
        "c2",
        "c3",
        "o1",
        "o2",
        "o3",
    ],
}

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

TASK_DATA_MD_PATH = SCRIPT_DIR / "TASK_DATA.md"

GASTRO_LABEL_DISPLAY = {
    "label_esophageal_smt": "食管 SMT",
    "label_esophageal_mucosal_or_tumor": "食管黏膜病变 / 食管肿物",
    "label_gastritis": "胃炎类",
}

EXCLUDE_REASON_DISPLAY = {
    "empty_watchResult": "`watchResult` 为空",
    "uncertain_without_target_label": "仅出现不确定表述，未命中目标标签",
    "no_target_label": "未命中目标标签",
    "conflicting_findings": "同时出现“正常”和病变语义，存在冲突",
}

EXCLUDE_REASON_ORDER = [
    "empty_watchResult",
    "uncertain_without_target_label",
    "no_target_label",
    "conflicting_findings",
]


def contains_any(text: str, patterns: list[str]) -> bool:
    return any(pattern in text for pattern in patterns)


def match_rule_map(text: str, rule_map: dict[str, list[str]]) -> dict[str, int]:
    return {label: int(contains_any(text, keywords)) for label, keywords in rule_map.items()}


def to_int(value: str | None) -> int:
    if value is None:
        return 0
    cleaned = re.sub(r"[^0-9]", "", str(value))
    return int(cleaned) if cleaned else 0


def load_rows(report_csv_path: Path) -> tuple[list[dict[str, str]], dict[str, str | None]]:
    if not report_csv_path.is_file():
        raise FileNotFoundError(f"未找到报告总表：{report_csv_path}")
    with report_csv_path.open("r", encoding="utf-8-sig", newline="") as file:
        reader = csv.DictReader(file)
        rows = list(reader)
        fieldnames = reader.fieldnames or []
    return rows, resolve_columns(fieldnames)


def get_value(row: dict[str, str], col_name: str | None) -> str:
    if col_name is None:
        return ""
    return str(row.get(col_name, "")).strip()


def is_gastroscopy_record(report_title_norm: str, watch_result_norm: str) -> bool:
    has_gastro = contains_any(report_title_norm, GASTROSCOPY_HINTS) or contains_any(watch_result_norm, GASTROSCOPY_HINTS)
    has_colon = contains_any(report_title_norm, COLONOSCOPY_HINTS) or contains_any(watch_result_norm, COLONOSCOPY_HINTS)
    return has_gastro and not has_colon


def is_colonoscopy_record(report_title_norm: str, watch_result_norm: str) -> bool:
    has_colon = contains_any(report_title_norm, COLONOSCOPY_HINTS) or contains_any(watch_result_norm, COLONOSCOPY_HINTS)
    has_gastro = contains_any(report_title_norm, GASTROSCOPY_HINTS) or contains_any(watch_result_norm, GASTROSCOPY_HINTS)
    return has_colon and not has_gastro


def write_csv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str]) -> None:
    with path.open("w", encoding="utf-8-sig", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        if rows:
            writer.writerows(rows)


def order_counter_items(counter: Counter[str]) -> list[tuple[str, int]]:
    ordered_keys = [key for key in EXCLUDE_REASON_ORDER if key in counter]
    remaining_keys = sorted(key for key in counter if key not in EXCLUDE_REASON_ORDER)
    return [(key, counter[key]) for key in ordered_keys + remaining_keys]


def build_counter_lines(counter: Counter[str]) -> list[str]:
    if not counter:
        return ["- 无"]

    lines: list[str] = []
    for key, value in order_counter_items(counter):
        display_name = EXCLUDE_REASON_DISPLAY.get(key, key)
        lines.append(f"- {display_name}：`{value}`")
    return lines


def build_task_data_markdown(summary: SelectionSummary) -> str:
    gastro_label_lines = [
        (
            f"- {GASTRO_LABEL_DISPLAY.get(label_name, label_name)}（字段：`{label_name}`）："
            f"`{summary.gastro_label_counter.get(label_name, 0)}`"
        )
        for label_name in GASTRO_LABEL_DISPLAY
    ]
    colon_label_lines = [
        f"- 正常（`0`）：`{summary.colon_binary_counter.get('正常', 0)}`",
        f"- 息肉（`1`）：`{summary.colon_binary_counter.get('息肉', 0)}`",
    ]

    lines = [
        "# 任务数据筛选结果",
        "",
        "## 胃镜三标签多标签任务",
        "",
        (
            f"本次共识别出 `{summary.gastro_total}` 条胃镜记录，其中 "
            f"`{summary.gastro_selected}` 条被纳入多标签任务，累计图像 "
            f"`{summary.gastro_img_sum}` 张。"
        ),
        "",
        "标签分布：",
        "",
        *gastro_label_lines,
        "",
        "主要剔除原因：",
        "",
        *build_counter_lines(summary.gastro_exclude_counter),
        "",
        "## 肠镜二分类任务",
        "",
        (
            f"本次共识别出 `{summary.colon_total}` 条肠镜记录，其中 "
            f"`{summary.colon_selected}` 条被纳入二分类任务，累计图像 "
            f"`{summary.colon_img_sum}` 张。"
        ),
        "",
        "类别分布：",
        "",
        *colon_label_lines,
        "",
        "主要剔除原因：",
        "",
        *build_counter_lines(summary.colon_binary_exclude_counter),
        "",
        "## 输出文件",
        "",
        f"生成文件位于：`{summary.output_dir}`",
        "",
        f"- `{summary.gastro_csv.name}`",
        f"- `{summary.colon_binary_csv.name}`",
        "",
    ]
    return "\n".join(lines)


def write_task_data_markdown(summary: SelectionSummary, markdown_path: Path) -> None:
    markdown_path.write_text(build_task_data_markdown(summary), encoding="utf-8")


def build_outputs(rows: list[dict[str, str]], columns: dict[str, str | None], output_dir: Path, report_csv_path: Path) -> SelectionSummary:
    output_dir.mkdir(parents=True, exist_ok=True)

    gastro_rows: list[dict[str, Any]] = []
    colon_binary_rows: list[dict[str, Any]] = []

    gastro_label_counter = Counter()
    colon_binary_counter = Counter()
    gastro_exclude_counter = Counter()
    colon_binary_exclude_counter = Counter()

    gastro_total = 0
    colon_total = 0
    gastro_img_sum = 0
    colon_img_sum = 0

    progress = build_progress(total=len(rows), desc="筛选任务数据")
    try:
        for row in rows:
            exam_dir = get_value(row, columns.get("exam_dir"))
            report_title = get_value(row, columns.get("reportTitle"))
            watch_result = get_value(row, columns.get("watchResult"))
            img_num = to_int(get_value(row, columns.get("img_num")))

            report_title_norm = normalize_text(report_title)
            watch_result_norm = normalize_text(watch_result)

            is_gastro = is_gastroscopy_record(report_title_norm, watch_result_norm)
            is_colon = is_colonoscopy_record(report_title_norm, watch_result_norm)

            if is_gastro:
                gastro_total += 1
            if is_colon:
                colon_total += 1

            if is_gastro:
                label_map = match_rule_map(watch_result_norm, GASTRO_LABEL_RULES)
                label_sum = sum(label_map.values())
                if not watch_result_norm:
                    gastro_exclude_counter["empty_watchResult"] += 1
                elif label_sum <= 0:
                    if contains_any(watch_result_norm, UNCERTAIN_TOKENS):
                        gastro_exclude_counter["uncertain_without_target_label"] += 1
                    else:
                        gastro_exclude_counter["no_target_label"] += 1
                else:
                    gastro_rows.append(
                        {
                            "exam_dir": exam_dir,
                            "reportTitle": report_title,
                            "watchResult": watch_result,
                            "img_num": img_num,
                            "label_esophageal_smt": label_map["label_esophageal_smt"],
                            "label_esophageal_mucosal_or_tumor": label_map["label_esophageal_mucosal_or_tumor"],
                            "label_gastritis": label_map["label_gastritis"],
                            "label_sum": label_sum,
                        }
                    )
                    gastro_img_sum += img_num
                    for label_name, label_value in label_map.items():
                        if label_value == 1:
                            gastro_label_counter[label_name] += 1

            if is_colon:
                has_normal = contains_any(watch_result_norm, NORMAL_PATTERNS)
                has_polyp = contains_any(watch_result_norm, POLYP_PATTERNS)
                has_other_lesion = contains_any(watch_result_norm, LESION_TOKENS)

                if not watch_result_norm:
                    colon_binary_exclude_counter["empty_watchResult"] += 1
                elif has_polyp:
                    colon_binary_rows.append(
                        {
                            "exam_dir": exam_dir,
                            "reportTitle": report_title,
                            "watchResult": watch_result,
                            "img_num": img_num,
                            "binary_label": 1,
                            "binary_label_name": "息肉",
                        }
                    )
                    colon_binary_img_sum_increment = img_num
                    colon_img_sum += colon_binary_img_sum_increment
                    colon_binary_counter["息肉"] += 1
                elif has_normal and not has_other_lesion:
                    colon_binary_rows.append(
                        {
                            "exam_dir": exam_dir,
                            "reportTitle": report_title,
                            "watchResult": watch_result,
                            "img_num": img_num,
                            "binary_label": 0,
                            "binary_label_name": "正常",
                        }
                    )
                    colon_img_sum += img_num
                    colon_binary_counter["正常"] += 1
                elif has_normal and has_other_lesion:
                    colon_binary_exclude_counter["conflicting_findings"] += 1
                else:
                    colon_binary_exclude_counter["no_target_label"] += 1

            progress.update(1)
    finally:
        progress.close()

    gastro_csv = output_dir / "gastro_multilabel_task_datalist.csv"
    colon_binary_csv = output_dir / "colonoscopy_binary_task_datalist.csv"

    write_csv(
        gastro_csv,
        gastro_rows,
        [
            "exam_dir",
            "reportTitle",
            "watchResult",
            "img_num",
            "label_esophageal_smt",
            "label_esophageal_mucosal_or_tumor",
            "label_gastritis",
            "label_sum",
        ],
    )
    write_csv(
        colon_binary_csv,
        colon_binary_rows,
        [
            "exam_dir",
            "reportTitle",
            "watchResult",
            "img_num",
            "binary_label",
            "binary_label_name",
        ],
    )

    print("\n=== 任务筛选结果 ===")
    print(f"胃镜总记录数：{gastro_total}")
    print(f"胃镜多标签纳入记录数：{len(gastro_rows)}")
    print(f"胃镜多标签总图像数：{gastro_img_sum}")
    print(f"胃镜标签-食管SMT：{gastro_label_counter.get('label_esophageal_smt', 0)}")
    print(f"胃镜标签-食管黏膜病变/食管肿物：{gastro_label_counter.get('label_esophageal_mucosal_or_tumor', 0)}")
    print(f"胃镜标签-胃炎类：{gastro_label_counter.get('label_gastritis', 0)}")
    print(f"胃镜剔除原因：{dict(gastro_exclude_counter)}")
    print(f"肠镜总记录数：{colon_total}")
    print(f"肠镜二分类纳入记录数：{len(colon_binary_rows)}")
    print(f"肠镜二分类总图像数：{colon_img_sum}")
    print(f"肠镜二分类正常：{colon_binary_counter.get('正常', 0)}")
    print(f"肠镜二分类息肉：{colon_binary_counter.get('息肉', 0)}")
    print(f"肠镜二分类剔除原因：{dict(colon_binary_exclude_counter)}")
    print("输出文件路径：")
    print(f"- {gastro_csv}")
    print(f"- {colon_binary_csv}")
    return SelectionSummary(
        input_row_count=len(rows),
        report_csv_path=report_csv_path,
        output_dir=output_dir,
        gastro_csv=gastro_csv,
        colon_binary_csv=colon_binary_csv,
        gastro_total=gastro_total,
        gastro_selected=len(gastro_rows),
        gastro_img_sum=gastro_img_sum,
        gastro_label_counter=gastro_label_counter,
        gastro_exclude_counter=gastro_exclude_counter,
        colon_total=colon_total,
        colon_selected=len(colon_binary_rows),
        colon_img_sum=colon_img_sum,
        colon_binary_counter=colon_binary_counter,
        colon_binary_exclude_counter=colon_binary_exclude_counter,
    )


def main() -> None:
    args = parse_args()
    path_config = build_path_config(
        config_path=args.config,
        report_csv_override=args.report_csv,
        output_dir_override=args.output_dir,
    )
    rows, columns = load_rows(path_config.report_csv_path)
    summary = build_outputs(
        rows=rows,
        columns=columns,
        output_dir=path_config.output_dir,
        report_csv_path=path_config.report_csv_path,
    )
    write_task_data_markdown(summary=summary, markdown_path=TASK_DATA_MD_PATH)
    print(f"TASK_DATA 文档已更新：{TASK_DATA_MD_PATH}")


if __name__ == "__main__":
    main()
