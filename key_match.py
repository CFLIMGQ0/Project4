from __future__ import annotations

import argparse
import json
import re
import sys
from collections import OrderedDict
from pathlib import Path

from check_pdf import (
    CONFIG_PATH,
    build_path_config,
    iter_pdf_files,
    process_single_pdf,
)

LABEL_PATTERN = re.compile(r"([\u4e00-\u9fffA-Za-z][\u4e00-\u9fffA-Za-z0-9]{1,20})：")
LEGACY_SECTION_PATTERN = re.compile(r"^【(.+?)】$")
CANDIDATE_MARK = "※"
DEFAULT_OUTPUT_JSON = Path(__file__).resolve().parent / "key_match.json"

ENGLISH_TO_CHINESE_CANDIDATES = OrderedDict(
    [
        ("date", ["检查日期", "报告日期", "日期"]),
        ("dactor", ["报告医师", "麻醉医生", "医生"]),
        ("project", ["操作名称", "检查项目", "项目"]),
        ("department", ["科室"]),
        ("bed", ["病床号"]),
        ("outpatient", ["门诊号"]),
        ("inspect", ["检查号", "检查项目"]),
        ("patient", ["姓名", "患者"]),
        ("endoscopy", ["内镜号", "镜号"]),
        ("proposal", ["注意事项", "建议"]),
        ("pathology", ["病理"]),
        ("Signal", ["指征"]),
        ("biopsy", ["活检部位"]),
        ("under_diagnosis", ["诊断"]),
        ("under_see", ["内镜所见"]),
        ("photo", ["图片", "照片"]),
        ("inspect_time", ["检查日期", "检查时间"]),
        ("project_name", ["操作名称", "项目名称"]),
        ("pics", ["图片"]),
        ("sex", ["性别"]),
        ("watch", ["内镜所见"]),
        ("doctorName", ["报告医师", "医生"]),
        ("bedId", ["病床号"]),
        ("patientAreaName", ["病区"]),
        ("checkTime", ["检查日期", "检查时间"]),
        ("archiveTime", ["报告日期", "归档时间"]),
        ("markImage", ["标记图", "图片"]),
        ("codeImage", ["二维码", "条码"]),
        ("age", ["年龄"]),
        ("reportTitle", ["报告标题", "报告名称"]),
        ("applyDeptName", ["科室"]),
        ("hisPatientId", ["内镜号", "患者编号"]),
        ("admissionNo", ["住院号"]),
        ("applyNo", ["检查号", "申请单号"]),
        ("endoscopeName", ["镜号", "内镜型号"]),
        ("narcosisType", ["麻醉方式"]),
        ("patientType", ["门诊号", "患者类型"]),
        ("specimen", ["活检部位", "标本"]),
        ("watchResult", ["操作结果", "诊断"]),
        ("suggest", ["注意事项", "建议"]),
        ("namePatient", ["姓名"]),
        ("anesthesiologistName", ["麻醉医生"]),
        ("operation", ["操作过程"]),
        ("conditionRemark", ["患者一般情况备注"]),
        ("operationRemark", ["操作过程备注"]),
        ("badnessRemark", ["不良反应备注"]),
        ("condition", ["患者一般情况"]),
        ("badness", ["不良反应"]),
        ("operationValue", ["操作名称"]),
        ("roomName", ["诊间"]),
    ]
)

EXTRA_VISIBLE_LABELS = [
    "姓名",
    "性别",
    "年龄",
    "内镜号",
    "科室",
    "门诊号",
    "病区",
    "病床号",
    "检查号",
    "检查日期",
    "报告日期",
    "诊间",
    "镜号",
    "麻醉方式",
    "麻醉医生",
    "操作步骤",
    "内镜所见",
    "操作名称",
    "操作结果",
    "活检部位",
    "诊断",
    "操作过程",
    "患者一般情况",
    "不良反应",
    "注意事项",
    "报告医师",
    "地址",
    "电话",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="生成英文字段到中文键名的 key_match.json 映射文件")
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
        default=4000,
        help="全文文字预览字符数，默认 4000，尽量保证可提取到更多中文标签",
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


def extract_visible_labels(full_text: str) -> list[str]:
    ordered_positions: list[tuple[int, str]] = []

    for match in LABEL_PATTERN.finditer(full_text):
        label = re.sub(r"\s+", "", match.group(1).strip())
        if 2 <= len(label) <= 20:
            ordered_positions.append((match.start(), label))

    for label in EXTRA_VISIBLE_LABELS:
        match = re.search(re.escape(label), full_text)
        if match:
            ordered_positions.append((match.start(), label))

    ordered_positions.sort(key=lambda item: (item[0], item[1]))

    deduped = OrderedDict()
    for _, label in ordered_positions:
        deduped.setdefault(label, True)
    return list(deduped.keys())


def parse_check_pdf_output(raw_text: str) -> tuple[OrderedDict[str, str], list[str]]:
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

    return fields, extract_visible_labels("\n".join(full_text_lines).strip())


def parse_legacy_key_match_output(raw_text: str) -> tuple[OrderedDict[str, str], list[str]]:
    fields = OrderedDict()
    visible_labels: list[str] = []
    section = ""

    for line in raw_text.splitlines():
        stripped = line.strip()
        section_match = LEGACY_SECTION_PATTERN.match(stripped)
        if section_match:
            section = section_match.group(1)
            continue

        if section == "提取到的英文键":
            if not stripped.startswith("- ") or ":" not in stripped:
                continue
            body = stripped[2:]
            key, value = body.split(":", 1)
            fields[key.strip()] = clean_value(value)
        elif section == "提取到的中文键":
            if stripped.startswith("- "):
                label = stripped[2:].strip()
                if label and label not in visible_labels:
                    visible_labels.append(label)

    return fields, visible_labels


def load_input_data(args: argparse.Namespace) -> tuple[str, OrderedDict[str, str], list[str], str, str]:
    raw_text = args.raw_text.strip()
    if not raw_text and not sys.stdin.isatty():
        raw_text = sys.stdin.read().strip()

    if raw_text:
        if "【表单字段】" in raw_text and "【全文文字预览】" in raw_text:
            fields, visible_labels = parse_check_pdf_output(raw_text)
            return "raw_text", fields, visible_labels, "未知", "来自 check_pdf.py 终端输出"
        fields, visible_labels = parse_legacy_key_match_output(raw_text)
        return "raw_text", fields, visible_labels, "未知", "来自键名分析终端输出"

    path_config = build_path_config(args.config, args.input_dir)
    pdf_files = list(iter_pdf_files(path_config.dataset_root))
    if not pdf_files:
        return "empty", OrderedDict(), [], "", ""

    pdf_path = pdf_files[0]
    fields, full_text, page_count, strategy = process_single_pdf(pdf_path, args.preview_chars)
    visible_labels = extract_visible_labels(full_text)
    return str(pdf_path), OrderedDict(fields), visible_labels, str(page_count if page_count != "" else "未知"), strategy


def resolve_label(key: str, visible_label_to_index: dict[str, int]) -> tuple[str, int, bool]:
    candidates = ENGLISH_TO_CHINESE_CANDIDATES.get(key, [])
    direct_matches = [label for label in candidates if label in visible_label_to_index]

    if direct_matches:
        best_label = min(direct_matches, key=lambda label: (visible_label_to_index[label], candidates.index(label)))
        return best_label, visible_label_to_index[best_label], False

    if candidates:
        return f"{CANDIDATE_MARK}{candidates[0]}", len(visible_label_to_index) + 1000, True

    return f"{CANDIDATE_MARK}未配置候选中文键", len(visible_label_to_index) + 2000, True


def build_key_match_json(ordered_fields: OrderedDict[str, str], visible_labels: list[str]) -> OrderedDict[str, str]:
    visible_label_to_index = {label: index for index, label in enumerate(visible_labels)}
    sortable_rows: list[tuple[int, int, str, str]] = []

    for original_index, key in enumerate(ordered_fields.keys()):
        resolved_label, label_index, used_candidate = resolve_label(key, visible_label_to_index)
        sort_index = label_index if not used_candidate else label_index + original_index
        sortable_rows.append((sort_index, original_index, key, resolved_label))

    sortable_rows.sort(key=lambda item: (item[0], item[1]))

    result = OrderedDict()
    for _, _, key, resolved_label in sortable_rows:
        result[key] = resolved_label
    return result


def write_key_match_json(output_path: Path, payload: OrderedDict[str, str]) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def main() -> None:
    args = parse_args()
    input_source, ordered_fields, visible_labels, page_count, strategy = load_input_data(args)

    if input_source == "empty":
        print("未找到目标 PDF，且未通过 --raw-text 或标准输入提供终端输出，无法生成 key_match.json。")
        return

    if not ordered_fields:
        print("未提取到英文键，无法生成 key_match.json。")
        return

    key_match_payload = build_key_match_json(ordered_fields, visible_labels)
    output_json = args.output_json.expanduser().resolve()
    write_key_match_json(output_json, key_match_payload)

    print("=" * 120)
    print("key_match.json 已生成")
    if input_source == "raw_text":
        print("输入来源：终端文本")
        print("原始文件：未直接访问 PDF")
    else:
        print(f"输入文件：{Path(input_source).name}")
        print(f"输入路径：{input_source}")
    print(f"页数：{page_count}")
    print(f"提取方式：{strategy}")
    print(f"中文键数量：{len(visible_labels)}")
    print(f"英文键数量：{len(ordered_fields)}")
    print(f"输出文件：{output_json}")
    print("说明：以 '※' 开头的中文键表示页面上未直接找到该标签，当前使用候选中文键占位。")
    print("-" * 120)
    print(json.dumps(key_match_payload, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
