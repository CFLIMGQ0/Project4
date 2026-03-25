from __future__ import annotations

import argparse
from collections import defaultdict
from pathlib import Path

from statistics import extract_pdf_fields

try:
    from pypdf import PdfReader
except Exception:  # noqa: BLE001
    PdfReader = None

EFFECTIVE_KEYS = {
    "admissionNo",
    "age",
    "anesthesiologistName",
    "applyDeptName",
    "applyNo",
    "archiveTime",
    "badness",
    "bedId",
    "checkTime",
    "condition",
    "doctorName",
    "endoscopeName",
    "hisPatientId",
    "hp",
    "namePatient",
    "narcosisType",
    "operation",
    "operationRemark",
    "operationValue",
    "patientAreaName",
    "patientType",
    "reportTitle",
    "roomName",
    "score",
    "sex",
    "specimen",
    "suggest",
    "watch",
    "watchResult",
}


class ProgressTracker:
    def __init__(self, total: int, width: int = 30, prefix: str = "处理进度") -> None:
        self.total = total
        self.width = width
        self.prefix = prefix
        self.current = 0

    def update(self, step: int = 1) -> None:
        if self.total <= 0:
            return
        self.current = min(self.total, self.current + step)
        ratio = self.current / self.total
        filled = min(self.width, int(ratio * self.width))
        bar = "#" * filled + "-" * (self.width - filled)
        print(
            f"\r{self.prefix}：[{bar}] {self.current}/{self.total} ({ratio:.0%})",
            end="",
            flush=True,
        )

    def close(self) -> None:
        if self.total <= 0:
            return
        self.update(0)
        print()


def normalize_text(value: object) -> str:
    if value is None:
        return ""
    return " ".join(str(value).strip().split())


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "扫描数据集 PDF，输出有效英文键对应的中文键名称；"
            "若同一英文键对应多个中文键，会打上[多值]标记。"
        )
    )
    parser.add_argument(
        "--dataset-root",
        type=Path,
        default=Path("/home/Lim/datasets/project4/main_data"),
        help="数据根目录（默认 /home/Lim/datasets/project4/main_data）",
    )
    return parser.parse_args()


def iter_pdf_files(dataset_root: Path):
    for patient_dir in sorted(path for path in dataset_root.iterdir() if path.is_dir()):
        for exam_dir in sorted(path for path in patient_dir.iterdir() if path.is_dir()):
            pdf_dir = exam_dir / "pdf"
            if not pdf_dir.is_dir():
                continue
            for pdf_path in sorted(path for path in pdf_dir.rglob("*.pdf") if path.is_file()):
                yield pdf_path


def extract_cn_labels_from_pdf(pdf_path: Path) -> dict[str, set[str]]:
    mapping: dict[str, set[str]] = defaultdict(set)
    if PdfReader is None:
        return mapping

    reader = PdfReader(str(pdf_path))
    fields = reader.get_fields() or {}
    for field_name, field in fields.items():
        english_key = normalize_text(field.get("/T") or field_name)
        if english_key not in EFFECTIVE_KEYS:
            continue

        chinese_key = normalize_text(field.get("/TU"))
        if chinese_key:
            mapping[english_key].add(chinese_key)

    return mapping


def merge_mapping(target: dict[str, set[str]], source: dict[str, set[str]]) -> None:
    for key, values in source.items():
        target[key].update(values)


def enrich_with_field_values(mapping: dict[str, set[str]], fields: dict[str, str]) -> None:
    """兜底：某些 PDF 取不到 /TU 时，保留该英文键，中文先记为未知。"""
    normalized = {normalize_text(k): normalize_text(v) for k, v in fields.items()}
    for key in EFFECTIVE_KEYS:
        if key in normalized and key not in mapping:
            mapping[key].add("（未提取到中文键名）")


def main() -> None:
    args = parse_args()
    dataset_root = args.dataset_root.expanduser().resolve()
    if not dataset_root.is_dir():
        return

    key_to_cn_names: dict[str, set[str]] = defaultdict(set)
    total_pdfs = sum(1 for _ in iter_pdf_files(dataset_root))
    progress = ProgressTracker(total=total_pdfs, prefix="遍历 PDF")

    for pdf_path in iter_pdf_files(dataset_root):
        try:
            merge_mapping(key_to_cn_names, extract_cn_labels_from_pdf(pdf_path))
            fields = extract_pdf_fields(pdf_path)
            enrich_with_field_values(key_to_cn_names, fields)
        except Exception:  # noqa: BLE001
            continue
        finally:
            progress.update()

    progress.close()

    for key in sorted(EFFECTIVE_KEYS):
        cn_names = sorted(name for name in key_to_cn_names.get(key, set()) if name)
        if not cn_names:
            print(f"{key} -> （未找到中文键名）")
            continue

        marker = " [多值]" if len(cn_names) > 1 else ""
        print(f"{key} -> {' / '.join(cn_names)}{marker}")


if __name__ == "__main__":
    main()
