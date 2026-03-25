from __future__ import annotations

import argparse
from pathlib import Path

from statistics import extract_pdf_fields

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


def normalize_text(value: object) -> str:
    if value is None:
        return ""
    return " ".join(str(value).strip().split())


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="找到一个所有有效键都非空的 PDF，并仅输出该 PDF 路径。"
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


def is_all_effective_keys_non_empty(fields: dict[str, str]) -> bool:
    normalized = {normalize_text(k): normalize_text(v) for k, v in fields.items()}
    return all(normalized.get(key, "") for key in EFFECTIVE_KEYS)


def main() -> None:
    args = parse_args()
    dataset_root = args.dataset_root.expanduser().resolve()
    if not dataset_root.is_dir():
        return

    for pdf_path in iter_pdf_files(dataset_root):
        try:
            fields = extract_pdf_fields(pdf_path)
        except Exception:  # noqa: BLE001
            continue

        if is_all_effective_keys_non_empty(fields):
            print(pdf_path)
            return


if __name__ == "__main__":
    main()
