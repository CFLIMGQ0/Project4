#!/usr/bin/env python3
"""患者级数据预处理脚本。

功能概述：
1. 扫描患者目录并生成匿名 patient_id。
2. 基于图像内容启发式规则识别 report / wle / eus 模态。
3. 对报告图进行 OCR（优先使用 pytesseract，失败时容错）。
4. 从 OCR 文本中抽取结构化字段并解析四分类标签。
5. 在每位患者目录下生成“纵向键值” report.csv（字段+取值两列）。
6. 生成全局 patient_summary.csv（每位患者一行）。

OCR 依赖说明：
- Python 包：pytesseract、Pillow
- 系统依赖：tesseract-ocr（建议安装中文语言包 chi_sim）
- 若依赖缺失或 OCR 失败，流程会继续执行，并标记 needs_manual_review。
"""

from __future__ import annotations

import argparse
import csv
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

try:
    from PIL import Image, ImageStat
except Exception:  # pragma: no cover
    Image = None
    ImageStat = None

try:
    import pytesseract
except Exception:  # pragma: no cover
    pytesseract = None


PNG_SUFFIXES = {".png", ".PNG"}
UNCERTAIN_TERMS = ("可能", "考虑", "待排", "不除外", "？", "?")
LABEL_GIST = "间质瘤"
LABEL_GIST_POSSIBLE = "可能间质瘤"
LABEL_LEIOMYOMA_POSSIBLE = "可能平滑肌瘤"
LABEL_LEIOMYOMA = "平滑肌瘤"


@dataclass
class ImageStats:
    width: int
    height: int
    mean_r: float
    mean_g: float
    mean_b: float
    std_r: float
    std_g: float
    std_b: float
    gray_std: float
    dark_ratio: float


def safe_read_image_stats(image_path: Path) -> Optional[ImageStats]:
    """读取图像统计信息，用于模态启发式识别。"""
    if Image is None or ImageStat is None:
        return None
    try:
        with Image.open(image_path) as img:
            rgb = img.convert("RGB")
            gray = img.convert("L")
            width, height = rgb.size
            rgb_stat = ImageStat.Stat(rgb)
            gray_stat = ImageStat.Stat(gray)
            mean_r, mean_g, mean_b = rgb_stat.mean
            std_r, std_g, std_b = rgb_stat.stddev
            gray_std = gray_stat.stddev[0]

            pixels = list(gray.getdata())
            dark_ratio = sum(1 for p in pixels if p < 40) / max(len(pixels), 1)
            return ImageStats(
                width=width,
                height=height,
                mean_r=mean_r,
                mean_g=mean_g,
                mean_b=mean_b,
                std_r=std_r,
                std_g=std_g,
                std_b=std_b,
                gray_std=gray_std,
                dark_ratio=dark_ratio,
            )
    except Exception:
        return None


def extract_report_text(image_path: Path, ocr_lang: str = "chi_sim+eng") -> str:
    """从报告图中提取文本，OCR 异常时返回空串。"""
    if pytesseract is None or Image is None:
        return ""
    try:
        with Image.open(image_path) as img:
            # 适当放大有助于小字体 OCR
            resized = img.convert("L").resize((img.width * 2, img.height * 2))
            text = pytesseract.image_to_string(resized, lang=ocr_lang)
            return text.strip()
    except Exception:
        return ""


def normalize_text(text: str) -> str:
    return re.sub(r"\s+", " ", text or "").strip()


def find_with_patterns(text: str, patterns: Sequence[str]) -> Optional[str]:
    for pattern in patterns:
        m = re.search(pattern, text, flags=re.IGNORECASE)
        if m:
            return m.group(1).strip(" ：:;；，,。")
    return None


def parse_structured_report_fields(report_text_raw: str) -> Dict[str, Optional[str]]:
    """从 OCR 文本中抽取结构化字段（字段级识别）。"""
    text = normalize_text(report_text_raw)

    fields = {
        "report_name": find_with_patterns(text, [r"姓名\s*[:：]\s*([^\s]+)"]),
        "report_sex": find_with_patterns(text, [r"性别\s*[:：]\s*([男女])"]),
        "report_age": find_with_patterns(text, [r"年龄\s*[:：]\s*([0-9]{1,3})\s*岁?"]),
        "report_exam_no": find_with_patterns(text, [r"检查号\s*[:：]\s*([A-Za-z0-9\-]+)"]),
        "report_ward_bed": find_with_patterns(
            text,
            [r"病区\s*[-－—]\s*床号\s*[:：]\s*([^\s]+)", r"病区床号\s*[:：]\s*([^\s]+)", r"床号\s*[:：]\s*([^\s]+)"],
        ),
        "report_outpatient_no": find_with_patterns(text, [r"门诊号\s*[:：]\s*([A-Za-z0-9\-]+)"]),
        "report_medical_record_no": find_with_patterns(text, [r"病案号\s*[:：]\s*([A-Za-z0-9\-]+)"]),
        "report_exam_date": find_with_patterns(text, [r"检查日期\s*[:：]\s*([0-9]{4}[\-/\.][0-9]{1,2}[\-/\.][0-9]{1,2})"]),
        "report_department": find_with_patterns(text, [r"申请科室\s*[:：]\s*([^\s]+)"]),
        "report_machine_model": find_with_patterns(text, [r"机型\s*[:：]\s*([^\s]+)", r"设备型号\s*[:：]\s*([^\s]+)"]),
        "report_description": find_with_patterns(text, [r"诊断描述\s*[:：]\s*(.+?)(?:镜下诊断|检查图像|$)"]),
        "report_endoscopic_impression": find_with_patterns(text, [r"镜下诊断\s*[:：]\s*(.+?)(?:检查图像|$)"]),
        "report_exam_images": find_with_patterns(text, [r"检查图[象像]\s*[:：]\s*(.+)$"]),
    }

    summary_parts = []
    for key in [
        "report_name",
        "report_sex",
        "report_age",
        "report_exam_date",
        "report_department",
        "report_endoscopic_impression",
        "report_description",
    ]:
        value = fields.get(key)
        if value:
            summary_parts.append(f"{key}={value}")
    fields["report_summary_normalized"] = " | ".join(summary_parts)
    return fields


def parse_label_from_report(report_endoscopic_impression: str, report_description: str) -> Tuple[Optional[str], bool, bool]:
    """解析四分类标签，返回 (label_4class, is_uncertain_label, needs_manual_review)。"""
    impression = normalize_text(report_endoscopic_impression)
    description = normalize_text(report_description)
    combined = f"{impression} {description}".strip()

    is_uncertain = any(term in combined for term in UNCERTAIN_TERMS)

    has_gist = "间质瘤" in combined
    has_leiomyoma = "平滑肌瘤" in combined

    label_4class: Optional[str] = None
    needs_manual_review = False

    if has_gist and has_leiomyoma:
        needs_manual_review = True
    elif has_gist:
        label_4class = LABEL_GIST_POSSIBLE if is_uncertain else LABEL_GIST
    elif has_leiomyoma:
        label_4class = LABEL_LEIOMYOMA_POSSIBLE if is_uncertain else LABEL_LEIOMYOMA
    else:
        needs_manual_review = True

    if not combined:
        needs_manual_review = True

    return label_4class, is_uncertain, needs_manual_review


def detect_image_modality(image_path: Path, ocr_lang: str) -> str:
    """识别单张图片模态：report / wle / eus。"""
    # 1) 文件名辅助（非唯一依据）
    name_lower = image_path.name.lower()
    if any(k in name_lower for k in ("report", "报告", "检查单")):
        return "report"
    if "eus" in name_lower:
        return "eus"
    if any(k in name_lower for k in ("wle", "gastroscopy", "胃镜", "内镜")):
        return "wle"

    # 2) 内容规则：先 OCR 少量文字判断是否为报告版式
    ocr_text = extract_report_text(image_path, ocr_lang=ocr_lang)
    report_keywords = ("姓名", "性别", "年龄", "检查号", "镜下诊断", "诊断描述", "申请科室")
    if sum(1 for k in report_keywords if k in ocr_text) >= 2:
        return "report"

    # 3) 图像统计启发式
    stats = safe_read_image_stats(image_path)
    if stats is None:
        return "eus"

    # 报告页通常长宽比接近纸张并且亮底+文字
    ratio = stats.width / max(stats.height, 1)
    color_std = (stats.std_r + stats.std_g + stats.std_b) / 3.0

    if 0.55 <= ratio <= 0.85 and stats.mean_r > 150 and stats.mean_g > 150 and stats.mean_b > 150 and color_std > 20:
        return "report"

    # EUS 通常偏灰度、暗部比例更高
    rgb_gap = max(abs(stats.mean_r - stats.mean_g), abs(stats.mean_g - stats.mean_b), abs(stats.mean_r - stats.mean_b))
    if rgb_gap < 18 and stats.dark_ratio > 0.2:
        return "eus"

    # 其余默认 WLE
    return "wle"


def list_patient_folders(dataset_root: Path) -> List[Path]:
    return sorted([p for p in dataset_root.iterdir() if p.is_dir()])


def collect_png_images(patient_dir: Path) -> List[Path]:
    return sorted([p for p in patient_dir.iterdir() if p.is_file() and p.suffix in PNG_SUFFIXES])


def join_paths(paths: Sequence[Path]) -> str:
    return "|".join(str(p) for p in paths)


def build_patient_record(patient_id: str, patient_dir: Path, ocr_lang: str) -> Dict[str, object]:
    folder_name = patient_dir.name
    png_files = collect_png_images(patient_dir)

    report_paths: List[Path] = []
    wle_paths: List[Path] = []
    eus_paths: List[Path] = []

    for img_path in png_files:
        # 优先使用常见数据命名约定：1.png=报告，2.png=WLE，3~n.png=EUS
        # 该规则用于弥补 OCR 对低清晰截图或版式变形报告页的识别不足。
        stem = img_path.stem.strip()
        modality: Optional[str] = None
        if stem.isdigit():
            image_idx = int(stem)
            if image_idx == 1:
                modality = "report"
            elif image_idx == 2:
                modality = "wle"
            elif image_idx >= 3:
                modality = "eus"

        if modality is None:
            modality = detect_image_modality(img_path, ocr_lang=ocr_lang)

        if modality == "report":
            report_paths.append(img_path)
        elif modality == "wle":
            wle_paths.append(img_path)
        else:
            eus_paths.append(img_path)

    # 兜底：若完全未识别出报告，则按“OCR 报告关键词命中最多”的图片回填为 report。
    # 这样可避免 report 被误归类导致字段几乎全空。
    if not report_paths and png_files:
        best_path: Optional[Path] = None
        best_score = -1
        report_keywords = ("姓名", "性别", "年龄", "检查号", "镜下诊断", "诊断描述", "申请科室")
        for img_path in png_files:
            text = extract_report_text(img_path, ocr_lang=ocr_lang)
            score = sum(1 for k in report_keywords if k in text)
            if score > best_score:
                best_score = score
                best_path = img_path

        # 至少命中 1 个关键词才进行回填，避免把普通内镜图误判为报告。
        if best_path is not None and best_score >= 1:
            report_paths.append(best_path)
            if best_path in wle_paths:
                wle_paths.remove(best_path)
            if best_path in eus_paths:
                eus_paths.remove(best_path)

    report_text_raw = ""
    if report_paths:
        texts = [extract_report_text(path, ocr_lang=ocr_lang) for path in report_paths]
        report_text_raw = "\n".join([t for t in texts if t]).strip()

    structured = parse_structured_report_fields(report_text_raw)
    label_4class, is_uncertain_label, needs_manual_review = parse_label_from_report(
        structured.get("report_endoscopic_impression") or "",
        structured.get("report_description") or "",
    )

    if not report_paths or not report_text_raw:
        needs_manual_review = True

    has_report = len(report_paths) > 0
    has_wle = len(wle_paths) > 0
    has_eus = len(eus_paths) > 0

    present = []
    if has_report:
        present.append("report")
    if has_wle:
        present.append("wle")
    if has_eus:
        present.append("eus")

    record: Dict[str, object] = {
        "patient_id": patient_id,
        "folder_name": folder_name,
        "folder_path": str(patient_dir),
        "report_paths": join_paths(report_paths),
        "wle_paths": join_paths(wle_paths),
        "eus_paths": join_paths(eus_paths),
        "num_report": len(report_paths),
        "num_wle": len(wle_paths),
        "num_eus": len(eus_paths),
        "has_report": has_report,
        "has_wle": has_wle,
        "has_eus": has_eus,
        "modalities_present": "|".join(present),
        "report_text_raw": report_text_raw,
        "label_4class": label_4class or "",
        "is_uncertain_label": is_uncertain_label,
        "needs_manual_review": needs_manual_review,
    }
    record.update(structured)
    return record


def write_csv(file_path: Path, rows: Sequence[Dict[str, object]], fieldnames: Sequence[str]) -> None:
    file_path.parent.mkdir(parents=True, exist_ok=True)
    with file_path.open("w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({k: row.get(k, "") for k in fieldnames})


def bool_str(v: object) -> str:
    return "True" if bool(v) else "False"


PATIENT_REPORT_ITEMS: List[Tuple[str, str]] = [
    ("姓名", "report_name"),
    ("性别", "report_sex"),
    ("年龄", "report_age"),
    ("检查号", "report_exam_no"),
    ("病区-床号", "report_ward_bed"),
    ("门诊号", "report_outpatient_no"),
    ("病案号", "report_medical_record_no"),
    ("检查日期", "report_exam_date"),
    ("申请科室", "report_department"),
    ("机型", "report_machine_model"),
    ("诊断描述", "report_description"),
    ("镜下诊断", "report_endoscopic_impression"),
    ("检查图象", "report_exam_images"),
    ("类别（四分类）", "label_4class"),
    ("是否有WLS", "has_wle"),
    ("WLS图像数", "num_wle"),
    ("是否有EUS", "has_eus"),
    ("EUS图像数", "num_eus"),
]


def write_patient_report_csv(record: Dict[str, object], overwrite: bool) -> None:
    patient_dir = Path(str(record["folder_path"]))
    out_file = patient_dir / "report.csv"
    if out_file.exists() and not overwrite:
        return

    rows: List[Dict[str, object]] = []
    for display_key, src_key in PATIENT_REPORT_ITEMS:
        value = record.get(src_key, "")
        if src_key in {"has_wle", "has_eus"}:
            value = bool_str(value)
        rows.append({"字段": display_key, "值": value if value is not None else ""})

    write_csv(out_file, rows, fieldnames=["字段", "值"])


ALL_FIELDS = [
    "patient_id",
    "folder_name",
    "folder_path",
    "report_paths",
    "wle_paths",
    "eus_paths",
    "num_report",
    "num_wle",
    "num_eus",
    "has_report",
    "has_wle",
    "has_eus",
    "modalities_present",
    "report_text_raw",
    "report_name",
    "report_sex",
    "report_age",
    "report_exam_no",
    "report_ward_bed",
    "report_outpatient_no",
    "report_medical_record_no",
    "report_exam_date",
    "report_department",
    "report_machine_model",
    "report_description",
    "report_endoscopic_impression",
    "report_exam_images",
    "report_summary_normalized",
    "label_4class",
    "is_uncertain_label",
    "needs_manual_review",
]

SUMMARY_FIELDS = [
    "patient_id",
    "folder_name",
    "label_4class",
    "report_name",
    "report_sex",
    "report_age",
    "report_exam_no",
    "report_ward_bed",
    "report_outpatient_no",
    "report_medical_record_no",
    "report_exam_date",
    "report_department",
    "report_machine_model",
    "report_description",
    "report_endoscopic_impression",
    "report_exam_images",
    "has_wle",
    "num_wle",
    "has_eus",
    "num_eus",
]


def run_pipeline(dataset_root: Path, output_dir: Path, overwrite_report_csv: bool, ocr_lang: str) -> Tuple[Path, int]:
    patient_folders = list_patient_folders(dataset_root)

    records: List[Dict[str, object]] = []
    for idx, patient_dir in enumerate(patient_folders, start=1):
        patient_id = f"P{idx:06d}"
        try:
            record = build_patient_record(patient_id=patient_id, patient_dir=patient_dir, ocr_lang=ocr_lang)
        except Exception:
            record = {
                "patient_id": patient_id,
                "folder_name": patient_dir.name,
                "folder_path": str(patient_dir),
                "report_paths": "",
                "wle_paths": "",
                "eus_paths": "",
                "num_report": 0,
                "num_wle": 0,
                "num_eus": 0,
                "has_report": False,
                "has_wle": False,
                "has_eus": False,
                "modalities_present": "",
                "report_text_raw": "",
                "report_name": "",
                "report_sex": "",
                "report_age": "",
                "report_exam_no": "",
                "report_ward_bed": "",
                "report_outpatient_no": "",
                "report_medical_record_no": "",
                "report_exam_date": "",
                "report_department": "",
                "report_machine_model": "",
                "report_description": "",
                "report_endoscopic_impression": "",
                "report_exam_images": "",
                "report_summary_normalized": "",
                "label_4class": "",
                "is_uncertain_label": False,
                "needs_manual_review": True,
            }

        records.append(record)
        write_patient_report_csv(record, overwrite=overwrite_report_csv)

    summary_path = output_dir / "patient_summary.csv"
    write_csv(summary_path, records, fieldnames=SUMMARY_FIELDS)
    return summary_path, len(records)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="患者级数据预处理脚本")
    parser.add_argument(
        "--dataset-root",
        type=Path,
        default=Path("/home/Lim/.cache/kagglehub/datasets/eus_dataset"),
        help="原始数据集根目录（患者文件夹一级目录）",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("/home/Lim/projects/eus-gist-leiomyoma/data/manifests"),
        help="全局输出目录（patient_summary.csv）",
    )
    parser.add_argument(
        "--overwrite-report-csv",
        action="store_true",
        help="若患者目录已存在 report.csv，是否覆盖写入",
    )
    parser.add_argument(
        "--ocr-lang",
        type=str,
        default="chi_sim+eng",
        help="pytesseract OCR 语言参数，例如 chi_sim+eng",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    dataset_root: Path = args.dataset_root
    output_dir: Path = args.output_dir

    if not dataset_root.exists() or not dataset_root.is_dir():
        raise SystemExit(f"dataset_root 不存在或不是目录: {dataset_root}")

    summary_path, patient_count = run_pipeline(
        dataset_root=dataset_root,
        output_dir=output_dir,
        overwrite_report_csv=args.overwrite_report_csv,
        ocr_lang=args.ocr_lang,
    )

    print("预处理完成")
    print(f"患者数量: {patient_count}")
    print(f"汇总文件: {summary_path}")


if __name__ == "__main__":
    main()
