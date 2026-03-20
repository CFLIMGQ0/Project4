from __future__ import annotations

import argparse
import json
import re
from collections import Counter
from dataclasses import asdict, dataclass
from pathlib import Path


DEFAULT_DATASET_ROOT = Path("/home/Lim/datasets/eus-gist-leiomyoma")
PATIENT_DIR_PATTERN = re.compile(r"^ZS\w+$", re.IGNORECASE)


@dataclass
class ExamIssue:
    patient_id: str
    exam_dir: str
    has_img: bool
    has_pdf: bool


@dataclass
class DatasetSummary:
    dataset_root: str
    patient_count: int
    total_exam_count: int
    empty_patient_count: int
    invalid_patient_name_count: int
    invalid_patient_name_examples: list[str]
    missing_img_or_pdf_exam_count: int
    missing_img_or_pdf_examples: list[ExamIssue]
    exam_count_distribution: dict[int, int]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="统计患者数量与检查次数分布")
    parser.add_argument(
        "--dataset-root",
        type=Path,
        default=DEFAULT_DATASET_ROOT,
        help="数据集根目录，默认使用 /home/Lim/datasets/eus-gist-leiomyoma/",
    )
    parser.add_argument(
        "--save-json",
        type=Path,
        default=None,
        help="可选：将统计结果保存为 JSON 文件",
    )
    parser.add_argument(
        "--show-all-issues",
        action="store_true",
        help="打印全部缺失 img/pdf 的检查目录，否则仅展示前 20 条",
    )
    return parser.parse_args()


def is_patient_dir(path: Path) -> bool:
    return path.is_dir()


def is_valid_patient_name(name: str) -> bool:
    return bool(PATIENT_DIR_PATTERN.match(name))


def collect_exam_dirs(patient_dir: Path) -> list[Path]:
    return sorted([path for path in patient_dir.iterdir() if path.is_dir()])


def summarize_dataset(dataset_root: Path) -> DatasetSummary:
    if not dataset_root.exists():
        raise FileNotFoundError(f"数据集根目录不存在：{dataset_root}")
    if not dataset_root.is_dir():
        raise NotADirectoryError(f"数据集根路径不是目录：{dataset_root}")

    patient_dirs = sorted([path for path in dataset_root.iterdir() if is_patient_dir(path)])
    exam_distribution: Counter[int] = Counter()
    invalid_patient_names: list[str] = []
    missing_structure_issues: list[ExamIssue] = []
    empty_patient_count = 0
    total_exam_count = 0

    for patient_dir in patient_dirs:
        if not is_valid_patient_name(patient_dir.name):
            invalid_patient_names.append(patient_dir.name)

        exam_dirs = collect_exam_dirs(patient_dir)
        exam_count = len(exam_dirs)
        exam_distribution[exam_count] += 1
        total_exam_count += exam_count

        if exam_count == 0:
            empty_patient_count += 1

        for exam_dir in exam_dirs:
            img_dir = exam_dir / "img"
            pdf_dir = exam_dir / "pdf"
            has_img = img_dir.is_dir()
            has_pdf = pdf_dir.is_dir()
            if not (has_img and has_pdf):
                missing_structure_issues.append(
                    ExamIssue(
                        patient_id=patient_dir.name,
                        exam_dir=exam_dir.name,
                        has_img=has_img,
                        has_pdf=has_pdf,
                    )
                )

    return DatasetSummary(
        dataset_root=str(dataset_root),
        patient_count=len(patient_dirs),
        total_exam_count=total_exam_count,
        empty_patient_count=empty_patient_count,
        invalid_patient_name_count=len(invalid_patient_names),
        invalid_patient_name_examples=invalid_patient_names[:20],
        missing_img_or_pdf_exam_count=len(missing_structure_issues),
        missing_img_or_pdf_examples=missing_structure_issues[:20],
        exam_count_distribution=dict(sorted(exam_distribution.items())),
    )


def print_summary(summary: DatasetSummary, show_all_issues: bool = False) -> None:
    print("数据集结构统计结果")
    print("=" * 60)
    print(f"数据集根目录：{summary.dataset_root}")
    print(f"患者总数：{summary.patient_count}")
    print(f"检查总次数：{summary.total_exam_count}")
    print(f"空患者目录数：{summary.empty_patient_count}")
    print(f"命名异常患者目录数：{summary.invalid_patient_name_count}")
    print(f"缺少 img/pdf 的检查目录数：{summary.missing_img_or_pdf_exam_count}")

    print("\n患者检查次数分布：")
    if not summary.exam_count_distribution:
        print("- 无可统计患者目录")
    else:
        for exam_count, patient_count in summary.exam_count_distribution.items():
            print(f"- {exam_count} 次检查：{patient_count} 个患者")

    if summary.invalid_patient_name_examples:
        print("\n命名异常患者目录样例（最多 20 条）：")
        for name in summary.invalid_patient_name_examples:
            print(f"- {name}")

    issues = summary.missing_img_or_pdf_examples
    if show_all_issues and summary.missing_img_or_pdf_exam_count > len(issues):
        print(
            "\n提示：当前输出对象中默认仅保留前 20 条结构异常样例；"
            "如需完整结果，可自行扩展脚本保存全部异常明细。"
        )

    if issues:
        print("\n缺少 img/pdf 的检查目录样例：")
        for issue in issues:
            print(
                f"- 患者 {issue.patient_id} / 检查 {issue.exam_dir} "
                f"(img={issue.has_img}, pdf={issue.has_pdf})"
            )


def save_summary_json(summary: DatasetSummary, output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    payload = asdict(summary)
    payload["missing_img_or_pdf_examples"] = [
        asdict(issue) for issue in summary.missing_img_or_pdf_examples
    ]
    output_path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def main() -> None:
    args = parse_args()
    summary = summarize_dataset(args.dataset_root.expanduser())
    print_summary(summary, show_all_issues=args.show_all_issues)

    if args.save_json is not None:
        save_summary_json(summary, args.save_json.expanduser())
        print(f"\n统计结果已保存到：{args.save_json.expanduser()}")


if __name__ == "__main__":
    main()
