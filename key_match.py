from __future__ import annotations

import argparse
import re
import sys
from collections import OrderedDict
from pathlib import Path

from check_pdf import (
    CONFIG_PATH,
    TARGET_EXAM_ID,
    TARGET_PATIENT_ID,
    build_path_config,
    iter_pdf_files,
    process_single_pdf,
)

LABEL_PATTERN = re.compile(r"([\u4e00-\u9fffA-Za-z][\u4e00-\u9fffA-Za-z0-9]{1,20})：")

ENGLISH_TO_CHINESE_CANDIDATES = OrderedDict(
    [
        ("date", ["日期", "检查日期", "报告日期"]),
        ("dactor", ["医生", "报告医师", "麻醉医生"]),
        ("project", ["项目", "检查项目", "操作名称"]),
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
        ("project_name", ["项目名称", "操作名称"]),
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
        ("watchResult", ["诊断", "操作结果"]),
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
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="输出 PDF 英文表单键与中文可见标签的对应关系，仅在终端打印")
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
        help="可选：直接传入 check_pdf.py 的终端输出文本进行解析，无需访问原始 PDF",
    )
    return parser.parse_args()



def extract_visible_labels(full_text: str) -> list[str]:
    found = OrderedDict()
    for label in EXTRA_VISIBLE_LABELS:
        if label in full_text:
            found[label] = True
    for match in LABEL_PATTERN.finditer(full_text):
        label = re.sub(r"\s+", "", match.group(1).strip())
        if 2 <= len(label) <= 20:
            found[label] = True
    return list(found.keys())



def find_best_matches(candidates: list[str], visible_labels: set[str]) -> list[str]:
    return [label for label in candidates if label in visible_labels]


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
        if stripped.startswith("处理完成") or stripped.startswith("{") and section == "full_text":
            break

        if section == "fields":
            if not stripped or stripped.startswith("【"):
                continue
            if ":" not in stripped:
                continue
            key, value = stripped.split(":", 1)
            fields[key.strip()] = value.strip().replace("（空值）", "")
        elif section == "full_text":
            full_text_lines.append(line.rstrip())

    full_text = "\n".join(full_text_lines).strip()
    return fields, full_text


def load_input_data(args: argparse.Namespace) -> tuple[str, OrderedDict[str, str], str, str, str]:
    raw_text = args.raw_text.strip()
    if not raw_text and not sys.stdin.isatty():
        raw_text = sys.stdin.read().strip()

    if raw_text:
        fields, full_text = parse_check_pdf_output(raw_text)
        return "raw_text", fields, full_text, "未知", "来自 check_pdf.py 终端输出"

    path_config = build_path_config(args.config, args.input_dir)
    pdf_files = list(iter_pdf_files(path_config.dataset_root))
    if not pdf_files:
        return "empty", OrderedDict(), "", "", ""

    pdf_path = pdf_files[0]
    fields, full_text, page_count, strategy = process_single_pdf(pdf_path, args.preview_chars)
    return str(pdf_path), OrderedDict(fields), full_text, str(page_count if page_count != "" else "未知"), strategy



def main() -> None:
    args = parse_args()
    input_source, ordered_fields, full_text, page_count, strategy = load_input_data(args)

    if input_source == "empty":
        print("未找到目标 PDF，且未通过 --raw-text 或标准输入提供 check_pdf.py 输出，无法进行键名匹配。")
        return

    visible_labels = extract_visible_labels(full_text)
    visible_label_set = set(visible_labels)

    matched_visible_labels: set[str] = set()
    unmatched_english: list[str] = []

    print("=" * 120)
    print("英文键与中文键匹配分析（仅终端输出）")
    if input_source == "raw_text":
        print("文件：来自终端粘贴文本")
        print("路径：未直接访问原始 PDF")
    else:
        pdf_name = Path(input_source).name
        print(f"文件：{pdf_name}")
        print(f"路径：{input_source}")
    print(f"页数：{page_count}")
    print(f"提取方式：{strategy}")
    print("=" * 120)

    print("\n【说明】")
    print("1. 英文键来自 PDF 表单字段名，它是模板内部键，不保证与页面上可见中文标签一一对应。")
    print("2. 中文键来自 PDF 页面可见文字中的标签，如“姓名：”“检查号：”“麻醉医生：”。")
    print("3. 若英文键找不到中文键，通常表示它是内部存储字段、隐藏字段、图片字段、备注字段，或者该中文标签未直接印在页面上。")
    print("4. 若中文键找不到英文键，通常表示它是固定排版文字、多个字段合并显示，或者值来自别的英文字段。")

    print("\n【提取到的英文键】")
    for key, value in ordered_fields.items():
        print(f"- {key}: {value if value else '（空值）'}")

    print("\n【提取到的中文键】")
    for label in visible_labels:
        print(f"- {label}")

    print("\n【英文键 -> 中文键对应】")
    for key in ordered_fields:
        candidates = ENGLISH_TO_CHINESE_CANDIDATES.get(key, [])
        matches = find_best_matches(candidates, visible_label_set)
        if matches:
            matched_visible_labels.update(matches)
            print(f"- {key} -> {', '.join(matches)}")
        else:
            unmatched_english.append(key)
            candidate_text = "、".join(candidates) if candidates else "（未配置候选中文键）"
            print(f"- {key} -> 未找到；候选中文键：{candidate_text}")

    unmatched_chinese = [label for label in visible_labels if label not in matched_visible_labels]

    print("\n【英文键缺少对应中文键】")
    if unmatched_english:
        for key in unmatched_english:
            print(f"- {key}")
    else:
        print("- 无")

    print("\n【中文键缺少对应英文键】")
    if unmatched_chinese:
        for label in unmatched_chinese:
            print(f"- {label}")
    else:
        print("- 无")


if __name__ == "__main__":
    main()
