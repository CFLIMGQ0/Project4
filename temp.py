from __future__ import annotations

import argparse
import csv
import importlib.util
import re
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any

if importlib.util.find_spec("yaml") is not None:
    import yaml
else:
    yaml = None

CONFIG_PATH = Path(__file__).resolve().parent / "configs" / "path.yaml"
PATIENT_DIR_PATTERN = re.compile(r"^ZS\w+$", re.IGNORECASE)


@dataclass
class PathConfig:
    dataset_root: Path
    output_dir: Path
    patient_validity_table: Path
    cleaning_report: Path


@dataclass
class ExamIssue:
    patient_id: str
    exam_dir: str
    has_img: bool
    has_pdf: bool


@dataclass
class PatientValidity:
    patient_id: str
    is_valid: int
    invalid_reason: str


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


@dataclass
class CleaningRoundResult:
    round_name: str
    removed_exam_count: int
    zero_exam_patient_count: int


@dataclass
class DatasetReport:
    original_summary: DatasetSummary
    cleaned_summary: DatasetSummary
    patient_validity_rows: list[PatientValidity]
    cleaning_rounds: list[CleaningRoundResult]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="统计患者数量与检查次数分布")
    parser.add_argument(
        "--config",
        type=Path,
        default=CONFIG_PATH,
        help="路径配置文件，默认使用 configs/path.yaml",
    )
    parser.add_argument(
        "--dataset-root",
        type=Path,
        default=None,
        help="可选：覆盖 path.yaml 中的 dataset_root",
    )
    parser.add_argument(
        "--save-table",
        type=Path,
        default=None,
        help="可选：覆盖 path.yaml 中的 patient_validity_table 输出路径",
    )
    parser.add_argument(
        "--save-cleaning-report",
        type=Path,
        default=None,
        help="可选：覆盖 path.yaml 中的 cleaning_report 输出路径",
    )
    parser.add_argument(
        "--show-all-issues",
        action="store_true",
        help="打印全部缺失 img/pdf 的检查目录，否则仅展示前 20 条",
    )
    return parser.parse_args()


def load_yaml_config(config_path: Path) -> dict[str, Any]:
    if not config_path.exists():
        raise FileNotFoundError(f"路径配置文件不存在：{config_path}")

    if yaml is not None:
        payload = yaml.safe_load(config_path.read_text(encoding="utf-8"))
        if not isinstance(payload, dict):
            raise ValueError(f"路径配置文件格式错误：{config_path}")
        return payload

    lines = config_path.read_text(encoding="utf-8").splitlines()
    payload: dict[str, Any] = {}
    current_section: str | None = None
    for raw_line in lines:
        if not raw_line.strip() or raw_line.lstrip().startswith("#"):
            continue

        indent = len(raw_line) - len(raw_line.lstrip(" "))
        line = raw_line.strip()
        if line.endswith(":"):
            current_section = line[:-1]
            payload[current_section] = {}
            continue

        key, _, value = line.partition(":")
        if not _:
            raise ValueError(f"无法解析路径配置行：{raw_line}")
        cleaned_value = value.strip().strip('"').strip("'")
        if indent == 0:
            payload[key.strip()] = cleaned_value
            current_section = None
            continue

        if current_section is None:
            raise ValueError(f"发现未归属分组的缩进行：{raw_line}")
        payload[current_section][key.strip()] = cleaned_value
    return payload


def build_path_config(
    config_path: Path,
    dataset_root: Path | None,
    save_table: Path | None,
    save_cleaning_report: Path | None,
) -> PathConfig:
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

    resolved_dataset_root = dataset_root.expanduser().resolve() if dataset_root is not None else resolve_path(str(paths_payload["dataset_root"]))
    resolved_output_dir = resolve_path(str(paths_payload["output_dir"]))
    resolved_patient_validity_table = save_table.expanduser().resolve() if save_table is not None else resolve_path(str(paths_payload["patient_validity_table"]))
    resolved_cleaning_report = save_cleaning_report.expanduser().resolve() if save_cleaning_report is not None else resolve_path(str(paths_payload["cleaning_report"]))

    return PathConfig(
        dataset_root=resolved_dataset_root,
        output_dir=resolved_output_dir,
        patient_validity_table=resolved_patient_validity_table,
        cleaning_report=resolved_cleaning_report,
    )


def is_patient_dir(path: Path) -> bool:
    return path.is_dir()


def is_valid_patient_name(name: str) -> bool:
    return bool(PATIENT_DIR_PATTERN.match(name))


def collect_exam_dirs(patient_dir: Path) -> list[Path]:
    return sorted([path for path in patient_dir.iterdir() if path.is_dir()])


def is_valid_exam_dir(exam_dir: Path) -> bool:
    return (exam_dir / "img").is_dir() and (exam_dir / "pdf").is_dir()


def build_summary(
    dataset_root: Path,
    patient_dirs: list[Path],
    exam_dirs_by_patient: dict[str, list[Path]],
    missing_structure_issues: list[ExamIssue],
) -> DatasetSummary:
    exam_distribution: Counter[int] = Counter()
    invalid_patient_names: list[str] = []
    empty_patient_count = 0
    total_exam_count = 0

    for patient_dir in patient_dirs:
        if not is_valid_patient_name(patient_dir.name):
            invalid_patient_names.append(patient_dir.name)

        exam_count = len(exam_dirs_by_patient[patient_dir.name])
        exam_distribution[exam_count] += 1
        total_exam_count += exam_count

        if exam_count == 0:
            empty_patient_count += 1

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


def build_patient_validity_rows(
    patient_dirs: list[Path],
    cleaned_exam_dirs_by_patient: dict[str, list[Path]],
) -> list[PatientValidity]:
    rows: list[PatientValidity] = []
    for patient_dir in patient_dirs:
        patient_id = patient_dir.name
        cleaned_exam_count = len(cleaned_exam_dirs_by_patient[patient_id])
        is_valid = 0 if cleaned_exam_count == 0 else 1
        invalid_reason = "第 1 轮清洗后该患者检查次数为 0" if is_valid == 0 else ""
        rows.append(
            PatientValidity(
                patient_id=patient_id,
                is_valid=is_valid,
                invalid_reason=invalid_reason,
            )
        )
    return rows


def summarize_dataset(dataset_root: Path) -> DatasetReport:
    if not dataset_root.exists():
        raise FileNotFoundError(f"数据集根目录不存在：{dataset_root}")
    if not dataset_root.is_dir():
        raise NotADirectoryError(f"数据集根路径不是目录：{dataset_root}")

    patient_dirs = sorted([path for path in dataset_root.iterdir() if is_patient_dir(path)])
    all_exam_dirs_by_patient: dict[str, list[Path]] = {}
    cleaned_exam_dirs_by_patient: dict[str, list[Path]] = {}
    missing_structure_issues: list[ExamIssue] = []

    for patient_dir in patient_dirs:
        exam_dirs = collect_exam_dirs(patient_dir)
        all_exam_dirs_by_patient[patient_dir.name] = exam_dirs

        valid_exam_dirs: list[Path] = []
        for exam_dir in exam_dirs:
            img_dir = exam_dir / "img"
            pdf_dir = exam_dir / "pdf"
            has_img = img_dir.is_dir()
            has_pdf = pdf_dir.is_dir()
            if is_valid_exam_dir(exam_dir):
                valid_exam_dirs.append(exam_dir)
                continue

            issue = ExamIssue(
                patient_id=patient_dir.name,
                exam_dir=exam_dir.name,
                has_img=has_img,
                has_pdf=has_pdf,
            )
            missing_structure_issues.append(issue)

        cleaned_exam_dirs_by_patient[patient_dir.name] = valid_exam_dirs

    original_summary = build_summary(
        dataset_root=dataset_root,
        patient_dirs=patient_dirs,
        exam_dirs_by_patient=all_exam_dirs_by_patient,
        missing_structure_issues=missing_structure_issues,
    )
    cleaned_summary = build_summary(
        dataset_root=dataset_root,
        patient_dirs=patient_dirs,
        exam_dirs_by_patient=cleaned_exam_dirs_by_patient,
        missing_structure_issues=[],
    )
    patient_validity_rows = build_patient_validity_rows(
        patient_dirs=patient_dirs,
        cleaned_exam_dirs_by_patient=cleaned_exam_dirs_by_patient,
    )
    first_round_removed_exam_count = sum(
        len(all_exam_dirs_by_patient[patient_dir.name]) - len(cleaned_exam_dirs_by_patient[patient_dir.name])
        for patient_dir in patient_dirs
    )
    first_round_zero_exam_patient_count = sum(
        1 for patient_dir in patient_dirs if len(cleaned_exam_dirs_by_patient[patient_dir.name]) == 0
    )
    cleaning_rounds = [
        CleaningRoundResult(
            round_name="第 1 轮：去掉缺失 img 或 pdf 的检查目录",
            removed_exam_count=first_round_removed_exam_count,
            zero_exam_patient_count=first_round_zero_exam_patient_count,
        )
    ]
    return DatasetReport(
        original_summary=original_summary,
        cleaned_summary=cleaned_summary,
        patient_validity_rows=patient_validity_rows,
        cleaning_rounds=cleaning_rounds,
    )


def print_single_summary(
    title: str,
    summary: DatasetSummary,
    show_issues: bool = False,
    show_all_issues: bool = False,
) -> None:
    print(title)
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

    if not show_issues:
        return

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


def print_report(report: DatasetReport, show_all_issues: bool = False) -> None:
    print_single_summary(
        title="原始统计结果",
        summary=report.original_summary,
        show_issues=True,
        show_all_issues=show_all_issues,
    )
    print()
    print_single_summary(
        title="第 1 轮清洗后的统计",
        summary=report.cleaned_summary,
    )
    print("\n清洗轮次摘要：")
    for round_result in report.cleaning_rounds:
        print(f"- {round_result.round_name}")
        print(f"  去掉的检查目录数：{round_result.removed_exam_count}")
        print(f"  清洗后检查次数为 0 的患者数：{round_result.zero_exam_patient_count}")


def save_patient_validity_table(rows: list[PatientValidity], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8-sig", newline="") as csv_file:
        writer = csv.writer(csv_file)
        writer.writerow(["patient_dir", "is_valid", "invalid_reason"])
        for row in rows:
            writer.writerow([row.patient_id, row.is_valid, row.invalid_reason])


def save_cleaning_report(report: DatasetReport, output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        f"原始数据：{report.original_summary.patient_count}名患者，{report.original_summary.total_exam_count}次检查",
    ]
    for round_result in report.cleaning_rounds:
        lines.append(round_result.round_name)
        lines.append(f"- 去掉的检查目录数：{round_result.removed_exam_count}")
        lines.append(f"- 清洗后检查次数为 0 的患者数：{round_result.zero_exam_patient_count}")
    lines.append(
        f"第 1 轮清洗后保留：{report.cleaned_summary.patient_count}名患者，{report.cleaned_summary.total_exam_count}次检查"
    )
    output_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    args = parse_args()
    path_config = build_path_config(
        config_path=args.config,
        dataset_root=args.dataset_root,
        save_table=args.save_table,
        save_cleaning_report=args.save_cleaning_report,
    )
    path_config.output_dir.mkdir(parents=True, exist_ok=True)

    report = summarize_dataset(path_config.dataset_root)
    print_report(report, show_all_issues=args.show_all_issues)

    save_patient_validity_table(report.patient_validity_rows, path_config.patient_validity_table)
    print(f"\n患者有效性表格已保存到：{path_config.patient_validity_table}")

    save_cleaning_report(report, path_config.cleaning_report)
    print(f"清洗统计报告已保存到：{path_config.cleaning_report}")


if __name__ == "__main__":
    main()
