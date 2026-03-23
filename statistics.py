from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass
class ProgressTracker:
    total: int
    prefix: str = "处理 PDF"
    width: int = 30
    current: int = 0

    def update(self, step: int = 1) -> None:
        self.current = min(self.total, self.current + step)
        self.render()

    def render(self) -> None:
        if self.total <= 0:
            return
        ratio = self.current / self.total
        filled = min(self.width, int(ratio * self.width))
        bar = "#" * filled + "-" * (self.width - filled)
        print(
            f"\r{self.prefix}进度：[{bar}] {self.current}/{self.total} ({ratio:.0%})",
            end="",
            flush=True,
        )

    def close(self) -> None:
        if self.total <= 0:
            return
        self.render()
        print()

from check_pdf import (
    CONFIG_PATH,
    MAX_PDF_SIZE_MB,
    PdfProcessError,
    PdfReader,
    clean_inline,
    extract_form_fields,
    load_yaml_config,
    normalize_value,
    options_to_dict,
    parse_pdf_objects,
)


@dataclass
class PathConfig:
    dataset_root: Path


@dataclass
class PdfStat:
    patient_id: str
    exam_id: str
    pdf_path: Path
    fields: dict[str, str]

    @property
    def field_count(self) -> int:
        return len(self.fields)

    @property
    def non_empty_field_count(self) -> int:
        return sum(1 for value in self.fields.values() if value)


@dataclass
class ExamDedupResult:
    patient_id: str
    exam_id: str
    pdf_count: int
    parsed_pdf_count: int
    status: str
    representative_pdf: Path | None = None
    representative_non_empty_count: int = 0
    representative_field_count: int = 0
    conflict_keys: list[str] | None = None
    skipped_reason: str = ""


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="统计全部患者 PDF 表单键及患者级去重结果")
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
        "--max-patients",
        type=int,
        default=None,
        help="可选：仅处理前 N 个患者，便于快速抽查",
    )
    parser.add_argument(
        "--max-examples",
        type=int,
        default=10,
        help="终端展示异常样例的最大条数，默认 10",
    )
    parser.add_argument(
        "--output-json",
        type=Path,
        default=None,
        help="可选：将完整统计结果保存为 JSON 文件",
    )
    return parser.parse_args()


def build_path_config(config_path: Path, dataset_root: Path | None) -> PathConfig:
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
    return PathConfig(dataset_root=resolved_dataset_root)


def iter_patient_dirs(dataset_root: Path) -> list[Path]:
    return sorted(path for path in dataset_root.iterdir() if path.is_dir())


def iter_exam_dirs(patient_dir: Path) -> list[Path]:
    return sorted(path for path in patient_dir.iterdir() if path.is_dir())


def iter_pdf_files(exam_dir: Path) -> list[Path]:
    pdf_dir = exam_dir / "pdf"
    if not pdf_dir.is_dir():
        return []
    return sorted(path for path in pdf_dir.rglob("*.pdf") if path.is_file())


def extract_pdf_fields(pdf_path: Path) -> dict[str, str]:
    if PdfReader is not None:
        reader = PdfReader(str(pdf_path))
        fields = reader.get_fields() or {}
        extracted: dict[str, str] = {}
        for field_name, field in fields.items():
            value = clean_inline(field.get("/V"))
            opt_map = options_to_dict(field.get("/Opt"))
            display_value = opt_map.get(value, value) if opt_map else value
            extracted[field_name] = normalize_value(display_value)
        return extracted

    pdf_size_mb = pdf_path.stat().st_size / (1024 * 1024)
    if pdf_size_mb > MAX_PDF_SIZE_MB:
        raise PdfProcessError(
            f"文件大小为 {pdf_size_mb:.1f} MB，超过限制 {MAX_PDF_SIZE_MB} MB，已跳过以避免内存占用过高"
        )

    pdf_bytes = pdf_path.read_bytes()
    if not pdf_bytes.startswith(b"%PDF"):
        raise PdfProcessError("文件头不是有效的 PDF 标识")

    objects = parse_pdf_objects(pdf_bytes)
    if not objects:
        raise PdfProcessError("未解析到 PDF 对象，可能是加密或结构异常文件")

    return {key: normalize_value(value) for key, value in extract_form_fields(objects)}


def collect_pdf_stats(
    dataset_root: Path,
    max_patients: int | None = None,
) -> tuple[list[str], dict[tuple[str, str], int], list[PdfStat], list[dict[str, str]]]:
    patient_dirs = iter_patient_dirs(dataset_root)
    if max_patients is not None and max_patients > 0:
        patient_dirs = patient_dirs[:max_patients]

    patient_ids = [path.name for path in patient_dirs]
    exam_pdf_totals: dict[tuple[str, str], int] = {}
    pdf_stats: list[PdfStat] = []
    errors: list[dict[str, str]] = []
    exam_targets: list[tuple[Path, Path, list[Path]]] = []

    for patient_dir in patient_dirs:
        for exam_dir in iter_exam_dirs(patient_dir):
            pdf_files = iter_pdf_files(exam_dir)
            exam_pdf_totals[(patient_dir.name, exam_dir.name)] = len(pdf_files)
            exam_targets.append((patient_dir, exam_dir, pdf_files))

    total_pdf_count = sum(len(pdf_files) for _, _, pdf_files in exam_targets)
    progress = ProgressTracker(total=total_pdf_count)
    if total_pdf_count > 0:
        progress.render()

    for patient_dir, exam_dir, pdf_files in exam_targets:
        for pdf_path in pdf_files:
            try:
                fields = extract_pdf_fields(pdf_path)
                pdf_stats.append(
                    PdfStat(
                        patient_id=patient_dir.name,
                        exam_id=exam_dir.name,
                        pdf_path=pdf_path,
                        fields=fields,
                    )
                )
            except Exception as exc:
                errors.append(
                    {
                        "patient_id": patient_dir.name,
                        "exam_id": exam_dir.name,
                        "pdf_path": str(pdf_path),
                        "reason": str(exc),
                    }
                )
            finally:
                progress.update()

    progress.close()
    return patient_ids, exam_pdf_totals, pdf_stats, errors


def summarize_keys(pdf_stats: list[PdfStat]) -> list[dict[str, int | str]]:
    total_counter: Counter[str] = Counter()
    non_empty_counter: Counter[str] = Counter()

    for stat in pdf_stats:
        for key, value in stat.fields.items():
            total_counter[key] += 1
            if value:
                non_empty_counter[key] += 1

    return [
        {
            "key": key,
            "total_count": total_counter[key],
            "non_empty_count": non_empty_counter[key],
        }
        for key in sorted(total_counter)
    ]


def group_pdfs_by_exam(pdf_stats: list[PdfStat]) -> dict[tuple[str, str], list[PdfStat]]:
    grouped: dict[tuple[str, str], list[PdfStat]] = defaultdict(list)
    for stat in pdf_stats:
        grouped[(stat.patient_id, stat.exam_id)].append(stat)
    return grouped


def find_conflict_keys(pdf_stats: list[PdfStat]) -> list[str]:
    key_to_values: dict[str, set[str]] = defaultdict(set)
    for stat in pdf_stats:
        for key, value in stat.fields.items():
            if value:
                key_to_values[key].add(value)
    return sorted(key for key, values in key_to_values.items() if len(values) > 1)


def choose_representative(pdf_stats: list[PdfStat]) -> PdfStat:
    return max(
        pdf_stats,
        key=lambda item: (
            item.non_empty_field_count,
            item.field_count,
            item.pdf_path.name,
        ),
    )


def build_exam_dedup_results(
    exam_pdf_totals: dict[tuple[str, str], int],
    pdf_stats: list[PdfStat],
    errors: list[dict[str, str]],
) -> list[ExamDedupResult]:
    grouped_stats = group_pdfs_by_exam(pdf_stats)
    error_count_by_exam: Counter[tuple[str, str]] = Counter((item["patient_id"], item["exam_id"]) for item in errors)
    all_exam_keys = sorted(set(exam_pdf_totals) | set(grouped_stats) | set(error_count_by_exam))
    results: list[ExamDedupResult] = []

    for patient_id, exam_id in all_exam_keys:
        exam_pdf_stats = sorted(grouped_stats.get((patient_id, exam_id), []), key=lambda item: item.pdf_path.name)
        pdf_count = exam_pdf_totals.get((patient_id, exam_id), len(exam_pdf_stats) + error_count_by_exam[(patient_id, exam_id)])

        if pdf_count == 0:
            results.append(
                ExamDedupResult(
                    patient_id=patient_id,
                    exam_id=exam_id,
                    pdf_count=0,
                    parsed_pdf_count=0,
                    status="skipped",
                    skipped_reason="该检查目录下没有 PDF 文件",
                )
            )
            continue

        if not exam_pdf_stats:
            results.append(
                ExamDedupResult(
                    patient_id=patient_id,
                    exam_id=exam_id,
                    pdf_count=pdf_count,
                    parsed_pdf_count=0,
                    status="skipped",
                    skipped_reason="该检查目录下 PDF 全部解析失败",
                )
            )
            continue

        conflict_keys = find_conflict_keys(exam_pdf_stats)
        if conflict_keys:
            results.append(
                ExamDedupResult(
                    patient_id=patient_id,
                    exam_id=exam_id,
                    pdf_count=pdf_count,
                    parsed_pdf_count=len(exam_pdf_stats),
                    status="failed",
                    conflict_keys=conflict_keys,
                )
            )
            continue

        representative = choose_representative(exam_pdf_stats)
        results.append(
            ExamDedupResult(
                patient_id=patient_id,
                exam_id=exam_id,
                pdf_count=pdf_count,
                parsed_pdf_count=len(exam_pdf_stats),
                status="success",
                representative_pdf=representative.pdf_path,
                representative_non_empty_count=representative.non_empty_field_count,
                representative_field_count=representative.field_count,
                skipped_reason="该检查目录存在部分 PDF 解析失败，但保留已成功解析结果" if error_count_by_exam[(patient_id, exam_id)] else "",
            )
        )

    return results


def summarize_patients(patient_ids: list[str], exam_results: list[ExamDedupResult]) -> dict[str, Any]:
    patient_to_results: dict[str, list[ExamDedupResult]] = defaultdict(list)
    for result in exam_results:
        patient_to_results[result.patient_id].append(result)

    success_patients: list[str] = []
    failed_patients: list[str] = []
    skipped_patients: list[str] = []

    for patient_id in sorted(patient_ids):
        results = patient_to_results.get(patient_id, [])
        if not results:
            skipped_patients.append(patient_id)
            continue
        statuses = {item.status for item in results}
        if "failed" in statuses:
            failed_patients.append(patient_id)
        elif statuses == {"success"}:
            success_patients.append(patient_id)
        else:
            skipped_patients.append(patient_id)

    return {
        "patient_count": len(patient_ids),
        "dedup_success_patient_count": len(success_patients),
        "dedup_failed_patient_count": len(failed_patients),
        "dedup_skipped_patient_count": len(skipped_patients),
        "dedup_success_patients": success_patients,
        "dedup_failed_patients": failed_patients,
        "dedup_skipped_patients": skipped_patients,
    }


def build_summary(
    dataset_root: Path,
    patient_ids: list[str],
    pdf_stats: list[PdfStat],
    errors: list[dict[str, str]],
    exam_results: list[ExamDedupResult],
) -> dict[str, Any]:
    patient_summary = summarize_patients(patient_ids, exam_results)
    return {
        "dataset_root": str(dataset_root),
        "parsed_pdf_count": len(pdf_stats),
        "pdf_parse_error_count": len(errors),
        "exam_count": len(exam_results),
        "dedup_success_exam_count": sum(1 for item in exam_results if item.status == "success"),
        "dedup_failed_exam_count": sum(1 for item in exam_results if item.status == "failed"),
        "dedup_skipped_exam_count": sum(1 for item in exam_results if item.status == "skipped"),
        **patient_summary,
    }


def to_jsonable_exam_results(exam_results: list[ExamDedupResult]) -> list[dict[str, Any]]:
    payload: list[dict[str, Any]] = []
    for item in exam_results:
        payload.append(
            {
                "patient_id": item.patient_id,
                "exam_id": item.exam_id,
                "pdf_count": item.pdf_count,
                "parsed_pdf_count": item.parsed_pdf_count,
                "status": item.status,
                "representative_pdf": str(item.representative_pdf) if item.representative_pdf else "",
                "representative_non_empty_count": item.representative_non_empty_count,
                "representative_field_count": item.representative_field_count,
                "conflict_keys": item.conflict_keys or [],
                "skipped_reason": item.skipped_reason,
            }
        )
    return payload


def print_key_stats(key_stats: list[dict[str, int | str]]) -> None:
    print("\n一、全部 PDF 表单键统计")
    print("=" * 80)
    if not key_stats:
        print("未解析到任何表单键。")
        return

    for item in key_stats:
        print(f"- {item['key']}（出现 {item['total_count']} 次）：非空 {item['non_empty_count']} 次")


def print_exam_stats(exam_results: list[ExamDedupResult], max_examples: int) -> None:
    print("\n二、检查目录去重统计")
    print("=" * 80)
    success_count = sum(1 for item in exam_results if item.status == "success")
    failed_count = sum(1 for item in exam_results if item.status == "failed")
    skipped_count = sum(1 for item in exam_results if item.status == "skipped")
    print(f"- 去重成功的检查目录数：{success_count}")
    print(f"- 去重失败的检查目录数：{failed_count}")
    print(f"- 跳过的检查目录数：{skipped_count}")

    failed_examples = [item for item in exam_results if item.status == "failed"][:max_examples]
    if failed_examples:
        print("\n去重失败样例：")
        for item in failed_examples:
            print(
                f"- 患者 {item.patient_id} / 检查 {item.exam_id}：冲突键 {', '.join(item.conflict_keys or [])}"
            )

    skipped_examples = [item for item in exam_results if item.status == "skipped"][:max_examples]
    if skipped_examples:
        print("\n跳过样例：")
        for item in skipped_examples:
            print(
                f"- 患者 {item.patient_id} / 检查 {item.exam_id}：{item.skipped_reason or '无可用 PDF'}"
            )


def print_patient_stats(summary: dict[str, Any], exam_results: list[ExamDedupResult], max_examples: int) -> None:
    print("\n三、患者级去重结果")
    print("=" * 80)
    print(f"- 参与统计的患者数：{summary['patient_count']}")
    print(f"- 去重成功患者数：{summary['dedup_success_patient_count']}")
    print(f"- 去重失败患者数：{summary['dedup_failed_patient_count']}")
    print(f"- 跳过患者数：{summary['dedup_skipped_patient_count']}")

    success_examples = [item for item in exam_results if item.status == "success"][:max_examples]
    if success_examples:
        print("\n去重成功样例（不展示代表 PDF 文件名）：")
        for item in success_examples:
            print(
                f"- 患者 {item.patient_id} / 检查 {item.exam_id}："
                f"代表结果非空键数={item.representative_non_empty_count}，"
                f"总键数={item.representative_field_count}"
            )


def print_error_stats(errors: list[dict[str, str]], max_examples: int) -> None:
    print("\n四、PDF 解析异常")
    print("=" * 80)
    print(f"- 解析失败 PDF 数：{len(errors)}")
    for item in errors[:max_examples]:
        print(
            f"- 患者 {item['patient_id']} / 检查 {item['exam_id']} / 文件 {Path(item['pdf_path']).name}：{item['reason']}"
        )


def save_json(
    output_json: Path,
    summary: dict[str, Any],
    key_stats: list[dict[str, int | str]],
    exam_results: list[ExamDedupResult],
    errors: list[dict[str, str]],
) -> None:
    output_json.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "summary": summary,
        "key_stats": key_stats,
        "exam_results": to_jsonable_exam_results(exam_results),
        "pdf_errors": errors,
    }
    output_json.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def main() -> None:
    args = parse_args()
    path_config = build_path_config(args.config, args.dataset_root)

    if not path_config.dataset_root.exists():
        print(f"数据集根目录不存在：{path_config.dataset_root}")
        return
    if not path_config.dataset_root.is_dir():
        print(f"数据集根路径不是目录：{path_config.dataset_root}")
        return

    patient_ids, exam_pdf_totals, pdf_stats, errors = collect_pdf_stats(
        path_config.dataset_root,
        max_patients=args.max_patients,
    )
    key_stats = summarize_keys(pdf_stats)
    exam_results = build_exam_dedup_results(exam_pdf_totals, pdf_stats, errors)
    summary = build_summary(path_config.dataset_root, patient_ids, pdf_stats, errors, exam_results)

    print("统计完成。")
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    print_key_stats(key_stats)
    print_exam_stats(exam_results, args.max_examples)
    print_patient_stats(summary, exam_results, args.max_examples)
    print_error_stats(errors, args.max_examples)

    if args.output_json is not None:
        save_json(args.output_json.expanduser().resolve(), summary, key_stats, exam_results, errors)
        print(f"\n完整 JSON 结果已保存到：{args.output_json.expanduser().resolve()}")


if __name__ == "__main__":
    main()
