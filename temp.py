from __future__ import annotations

import argparse
import random
from dataclasses import dataclass
from pathlib import Path

from statistics import extract_pdf_fields


@dataclass
class PdfParseResult:
    pdf_path: Path
    fields: dict[str, str]


@dataclass
class ExamConflictResult:
    exam_dir: Path
    pdf_results: list[PdfParseResult]
    specimen_values: set[str]


def normalize_text(value: object) -> str:
    if value is None:
        return ""
    return " ".join(str(value).strip().split())


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="随机抽取 specimen 键冲突的检查目录，并输出对应 PDF 表单解析结果。"
    )
    parser.add_argument(
        "--dataset-root",
        type=Path,
        default=Path("/home/Lim/datasets/project4/main_data"),
        help="数据根目录（默认 /home/Lim/datasets/project4/main_data）",
    )
    parser.add_argument(
        "--sample-size",
        type=int,
        default=3,
        help="随机抽取目录数，默认 3",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="随机种子（可选），用于复现实验",
    )
    return parser.parse_args()


def iter_exam_dirs(dataset_root: Path) -> list[Path]:
    if not dataset_root.is_dir():
        return []

    exam_dirs: list[Path] = []
    for patient_dir in sorted(path for path in dataset_root.iterdir() if path.is_dir()):
        exam_dirs.extend(sorted(path for path in patient_dir.iterdir() if path.is_dir()))
    return exam_dirs


def iter_pdf_files(exam_dir: Path) -> list[Path]:
    pdf_dir = exam_dir / "pdf"
    if not pdf_dir.is_dir():
        return []
    return sorted(path for path in pdf_dir.rglob("*.pdf") if path.is_file())


def collect_exam_pdf_results(exam_dir: Path) -> list[PdfParseResult]:
    results: list[PdfParseResult] = []
    for pdf_path in iter_pdf_files(exam_dir):
        try:
            fields = extract_pdf_fields(pdf_path)
        except Exception as exc:  # noqa: BLE001
            print(f"[警告] 解析失败，已跳过：{pdf_path} | {exc}")
            continue
        results.append(PdfParseResult(pdf_path=pdf_path, fields=fields))
    return results


def find_specimen_conflicted_exam_dirs(dataset_root: Path) -> list[ExamConflictResult]:
    conflicted: list[ExamConflictResult] = []
    for exam_dir in iter_exam_dirs(dataset_root):
        pdf_results = collect_exam_pdf_results(exam_dir)
        if len(pdf_results) < 2:
            continue

        specimen_values: set[str] = set()
        for item in pdf_results:
            specimen_value = normalize_text(item.fields.get("specimen", ""))
            if specimen_value:
                specimen_values.add(specimen_value)

        if len(specimen_values) >= 2:
            conflicted.append(
                ExamConflictResult(
                    exam_dir=exam_dir,
                    pdf_results=pdf_results,
                    specimen_values=specimen_values,
                )
            )
    return conflicted


def print_pdf_fields(fields: dict[str, str]) -> None:
    if not fields:
        print("    （无可解析字段）")
        return

    for key in sorted(fields):
        value = normalize_text(fields.get(key, ""))
        shown_value = value if value else "<空>"
        print(f"    - {key}: {shown_value}")


def main() -> None:
    args = parse_args()
    dataset_root = args.dataset_root.expanduser().resolve()

    if not dataset_root.is_dir():
        print(f"数据目录不存在：{dataset_root}")
        return

    conflicted_exam_dirs = find_specimen_conflicted_exam_dirs(dataset_root)
    print(f"数据根目录：{dataset_root}")
    print(f"找到 specimen 键冲突的检查目录数量：{len(conflicted_exam_dirs)}")

    if not conflicted_exam_dirs:
        print("未发现 specimen 键冲突目录。")
        return

    sample_size = max(1, args.sample_size)
    sample_size = min(sample_size, len(conflicted_exam_dirs))
    rng = random.Random(args.seed)
    selected = rng.sample(conflicted_exam_dirs, sample_size)

    print(f"\n随机抽取 {sample_size} 个检查目录（seed={args.seed}）：")
    for index, exam in enumerate(selected, start=1):
        print(f"\n[{index}] 检查目录：{exam.exam_dir}")
        print(f"  specimen 冲突值：{sorted(exam.specimen_values)}")
        print(f"  可解析 PDF 数量：{len(exam.pdf_results)}")

        for pdf_index, result in enumerate(exam.pdf_results, start=1):
            specimen_value = normalize_text(result.fields.get("specimen", "")) or "<空>"
            print(f"  ({pdf_index}) PDF：{result.pdf_path}")
            print(f"      specimen: {specimen_value}")
            print("      表单字段解析：")
            print_pdf_fields(result.fields)


if __name__ == "__main__":
    main()
