from __future__ import annotations

import argparse
import json
import re
import sys
import unicodedata
from collections import OrderedDict
from dataclasses import dataclass
from pathlib import Path

from check_pdf import (
    CONFIG_PATH,
    build_path_config,
    iter_pdf_files,
    process_single_pdf,
)

LABEL_PATTERN = re.compile(r"([\u4e00-\u9fffA-Za-z][\u4e00-\u9fffA-Za-z0-9]{1,20})：")
DEFAULT_OUTPUT_JSON = Path(__file__).resolve().parent / "key_match.json"
IGNORED_TEXT_PREFIXES = ("[第",)


@dataclass
class LabelEntry:
    label: str
    value: str
    order_index: int


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="根据 PDF 页面实际内容生成 key_match.json")
    parser.add_argument(
        "--config",
        type=Path,
        default=CONFIG_PATH,
        help="路径配置文件，默认使用 configs/path.yaml",
    )
    parser.add_argument(
        "--input-dir",
        type=Path,
        default=None,
        help="可选：覆盖配置中的 dataset_root，支持传入任意 PDF 根目录",
    )
    parser.add_argument(
        "--preview-chars",
        type=int,
        default=12000,
        help="全文文字预览字符数，默认 12000，尽量避免关键信息被截断",
    )
    parser.add_argument(
        "--raw-text",
        type=str,
        default="",
        help="可选：直接传入终端文本进行解析，无需访问原始 PDF",
    )
    parser.add_argument(
        "--output-json",
        type=Path,
        default=DEFAULT_OUTPUT_JSON,
        help="输出的 key_match.json 路径，默认写入仓库根目录",
    )
    return parser.parse_args()


def clean_value(value: str) -> str:
    return value.strip().replace("（空值）", "")


def normalize_text(text: str) -> str:
    text = unicodedata.normalize("NFKC", text or "")
    text = text.replace("\x00", "")
    text = re.sub(r"\s+", "", text)
    text = re.sub(r"[：:，,。；;、（）()【】\[\]<>《》\"'“”‘’]", "", text)
    return text.strip()


def parse_check_pdf_output(raw_text: str) -> tuple[OrderedDict[str, str], str]:
    fields = OrderedDict()
    full_text_lines: list[str] = []
    section = ""

    for line in raw_text.splitlines():
        stripped = line.strip()
        if stripped == "【表单字段】":
            section = "fields"
            continue
        if stripped == "【全文文字预览】":
            section = "full_text"
            continue
        if stripped.startswith("处理完成") or (stripped.startswith("{") and section == "full_text"):
            break

        if section == "fields":
            if not stripped or stripped.startswith("【") or ":" not in stripped:
                continue
            key, value = stripped.split(":", 1)
            fields[key.strip()] = clean_value(value)
        elif section == "full_text":
            full_text_lines.append(line.rstrip())

    return fields, "\n".join(full_text_lines).strip()


def load_input_data(args: argparse.Namespace) -> tuple[str, OrderedDict[str, str], str, str, str]:
    raw_text = args.raw_text.strip()
    if not raw_text and not sys.stdin.isatty():
        raw_text = sys.stdin.read().strip()

    if raw_text:
        if "【表单字段】" not in raw_text or "【全文文字预览】" not in raw_text:
            raise ValueError("--raw-text 需要传入 check_pdf.py 的完整终端输出，必须同时包含【表单字段】和【全文文字预览】。")
        fields, full_text = parse_check_pdf_output(raw_text)
        return "raw_text", fields, full_text, "未知", "来自 check_pdf.py 终端输出"

    path_config = build_path_config(args.config, args.input_dir)
    pdf_files = list(iter_pdf_files(path_config.dataset_root))
    if not pdf_files:
        return "empty", OrderedDict(), "", "", ""

    pdf_path = pdf_files[0]
    fields, full_text, page_count, strategy = process_single_pdf(pdf_path, args.preview_chars)
    return str(pdf_path), OrderedDict(fields), full_text, str(page_count if page_count != "" else "未知"), strategy


def extract_label_entries(full_text: str) -> list[LabelEntry]:
    entries: list[LabelEntry] = []
    current_entry: LabelEntry | None = None
    order_index = 0

    for raw_line in full_text.splitlines():
        line = raw_line.strip()
        if not line:
            continue
        if any(line.startswith(prefix) and line.endswith("页]") for prefix in IGNORED_TEXT_PREFIXES):
            continue

        matches = list(LABEL_PATTERN.finditer(line))
        if not matches:
            if current_entry is not None:
                current_entry.value = f"{current_entry.value}\n{line}".strip()
            continue

        for index, match in enumerate(matches):
            label = re.sub(r"\s+", "", match.group(1).strip())
            start = match.end()
            end = matches[index + 1].start() if index + 1 < len(matches) else len(line)
            value = line[start:end].strip()
            value = re.sub(r"^[：:]+", "", value).strip()
            current_entry = LabelEntry(label=label, value=value, order_index=order_index)
            entries.append(current_entry)
            order_index += 1

    return entries


def score_label_match(field_value: str, label_value: str) -> tuple[int, int] | None:
    normalized_field = normalize_text(field_value)
    normalized_label_value = normalize_text(label_value)

    if not normalized_field or not normalized_label_value:
        return None

    if normalized_field == normalized_label_value:
        return (3, len(normalized_field))
    if normalized_field in normalized_label_value:
        return (2, len(normalized_field))
    if normalized_label_value in normalized_field:
        return (1, len(normalized_label_value))
    return None


def build_key_match_json(ordered_fields: OrderedDict[str, str], label_entries: list[LabelEntry]) -> OrderedDict[str, str]:
    matched_rows: list[tuple[int, int, str, str]] = []

    for original_index, (field_key, field_value) in enumerate(ordered_fields.items()):
        cleaned_field_value = clean_value(field_value)
        if not cleaned_field_value:
            continue

        best_match: tuple[int, int, int, str] | None = None
        for entry in label_entries:
            score = score_label_match(cleaned_field_value, entry.value)
            if score is None:
                continue

            candidate = (score[0], score[1], -entry.order_index, entry.label)
            if best_match is None or candidate > best_match:
                best_match = candidate

        if best_match is None:
            continue

        _, _, negative_order_index, label = best_match
        matched_rows.append((-negative_order_index, original_index, field_key, label))

    matched_rows.sort(key=lambda item: (item[0], item[1]))

    result = OrderedDict()
    for _, _, field_key, label in matched_rows:
        result[field_key] = label
    return result


def write_key_match_json(output_path: Path, payload: OrderedDict[str, str]) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def main() -> None:
    args = parse_args()
    input_source, ordered_fields, full_text, page_count, strategy = load_input_data(args)

    if input_source == "empty":
        print("未找到目标 PDF，且未通过 --raw-text 或标准输入提供 check_pdf.py 输出，无法生成 key_match.json。")
        return

    if not ordered_fields:
        print("未提取到英文键，无法生成 key_match.json。")
        return

    label_entries = extract_label_entries(full_text)
    if not label_entries:
        print("未从 PDF 页面文字中解析到“中文键：值”的结构，无法生成可靠的 key_match.json。")
        return

    key_match_payload = build_key_match_json(ordered_fields, label_entries)
    output_json = args.output_json.expanduser().resolve()
    write_key_match_json(output_json, key_match_payload)

    print("=" * 120)
    print("key_match.json 已生成")
    if input_source == "raw_text":
        print("输入来源：check_pdf.py 终端文本")
        print("原始文件：未直接访问 PDF")
    else:
        print(f"输入文件：{Path(input_source).name}")
        print(f"输入路径：{input_source}")
    print(f"页数：{page_count}")
    print(f"提取方式：{strategy}")
    print(f"英文键总数：{len(ordered_fields)}")
    print(f"页面标签总数：{len(label_entries)}")
    print(f"最终保留映射数：{len(key_match_payload)}")
    print(f"输出文件：{output_json}")
    print("说明：仅保留“英文键有值”且“值能在页面中文标签处匹配到”的映射项；空值键和无法确认的键都会被丢弃。")
    print("-" * 120)
    print(json.dumps(key_match_payload, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
