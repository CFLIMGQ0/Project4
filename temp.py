from __future__ import annotations

import argparse
from pathlib import Path

from statistics import (
    CONFIG_PATH,
    build_exam_dedup_results,
    build_path_config,
    collect_pdf_stats,
    load_valid_patient_ids,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="抽取含冲突 PDF 的患者样本（最多 10 人）"
    )
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
        "--patient-validity-table",
        type=Path,
        default=None,
        help="可选：覆盖 path.yaml 中的 patient_validity_table",
    )
    parser.add_argument(
        "--max-patients",
        type=int,
        default=None,
        help="可选：限制参与统计的患者数，用于快速抽样",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=10,
        help="输出患者数量上限，默认 10",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    path_config = build_path_config(
        config_path=args.config,
        dataset_root=args.dataset_root,
        patient_validity_table=args.patient_validity_table,
    )
    valid_patient_ids = load_valid_patient_ids(path_config.patient_validity_table)

    patient_ids, exam_pdf_totals, pdf_stats, errors = collect_pdf_stats(
        dataset_root=path_config.dataset_root,
        valid_patient_ids=valid_patient_ids,
        max_patients=args.max_patients,
    )

    exam_results = build_exam_dedup_results(
        exam_pdf_totals=exam_pdf_totals,
        pdf_stats=pdf_stats,
        errors=errors,
    )

    conflict_results = [item for item in exam_results if item.status == "failed"]
    patient_to_conflicts: dict[str, list] = {}
    for item in conflict_results:
        patient_to_conflicts.setdefault(item.patient_id, []).append(item)

    selected_patient_ids = sorted(patient_to_conflicts)[: max(args.limit, 0)]

    print(f"有效患者总数：{len(patient_ids)}")
    print(f"含冲突检查的患者数：{len(patient_to_conflicts)}")
    print(f"展示患者数（limit={args.limit}）：{len(selected_patient_ids)}")

    if not selected_patient_ids:
        print("未找到包含冲突 PDF 的患者。")
        return

    print("\n样例患者（每人展示含冲突的检查目录与冲突键）：")
    for patient_id in selected_patient_ids:
        print(f"- 患者 {patient_id}")
        for item in sorted(patient_to_conflicts[patient_id], key=lambda x: x.exam_id):
            conflict_key_text = ", ".join(item.conflict_keys or [])
            print(
                f"  - 检查 {item.exam_id} | pdf_count={item.pdf_count} | "
                f"parsed_pdf_count={item.parsed_pdf_count} | 冲突键：{conflict_key_text}"
            )


if __name__ == "__main__":
    main()
