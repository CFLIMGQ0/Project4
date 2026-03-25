from __future__ import annotations

import argparse
from dataclasses import dataclass
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


@dataclass
class ProgressTracker:
    total: int
    width: int = 30
    prefix: str = "扫描进度"
    current: int = 0

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


def parse_target_keys(raw_target_keys: str) -> list[str]:
    target_keys = [normalize_text(key) for key in raw_target_keys.split(",")]
    target_keys = [key for key in target_keys if key]
    if not target_keys:
        raise ValueError("--target-keys 不能为空，至少提供 1 个键")

    invalid_keys = [key for key in target_keys if key not in EFFECTIVE_KEYS]
    if invalid_keys:
        raise ValueError(
            "以下键不在有效键清单内：" + ", ".join(sorted(set(invalid_keys)))
        )

    return list(dict.fromkeys(target_keys))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "按给定英文键查找“字段值非空”的首个 PDF，并输出该 PDF 路径；"
            "一旦所有目标键都命中，立即停止扫描。"
        )
    )
    parser.add_argument(
        "--dataset-root",
        type=Path,
        default=Path("/home/Lim/datasets/project4/main_data"),
        help="数据根目录（默认 /home/Lim/datasets/project4/main_data）",
    )
    parser.add_argument(
        "--target-keys",
        type=str,
        required=True,
        help="待查找的英文键，使用英文逗号分隔，例如 age,operation,watch",
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


def collect_first_hits(dataset_root: Path, target_keys: list[str]) -> dict[str, tuple[Path, str]]:
    remaining = set(target_keys)
    hits: dict[str, tuple[Path, str]] = {}

    total_pdfs = sum(1 for _ in iter_pdf_files(dataset_root))
    progress = ProgressTracker(total=total_pdfs)

    for pdf_path in iter_pdf_files(dataset_root):
        if not remaining:
            break
        try:
            fields = extract_pdf_fields(pdf_path)
        except Exception:  # noqa: BLE001
            progress.update()
            continue

        normalized_fields = {
            normalize_text(key): normalize_text(value) for key, value in fields.items()
        }

        for key in tuple(remaining):
            value = normalized_fields.get(key, "")
            if value:
                hits[key] = (pdf_path, value)
                remaining.remove(key)

        progress.update()

    progress.close()
    return hits


def print_result(target_keys: list[str], hits: dict[str, tuple[Path, str]]) -> None:
    print("\n查找结果：")
    for key in target_keys:
        matched = hits.get(key)
        if matched is None:
            print(f"- {key}: 未找到非空值")
            continue

        pdf_path, value = matched
        print(f"- {key}: {pdf_path.as_uri()}")
        print(f"  示例值: {value}")


def main() -> None:
    args = parse_args()
    dataset_root = args.dataset_root.expanduser().resolve()
    if not dataset_root.is_dir():
        raise FileNotFoundError(f"数据根目录不存在：{dataset_root}")

    target_keys = parse_target_keys(args.target_keys)
    hits = collect_first_hits(dataset_root, target_keys)
    print_result(target_keys, hits)


if __name__ == "__main__":
    main()
