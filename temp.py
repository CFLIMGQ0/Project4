from __future__ import annotations

import argparse
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path

from statistics import (
    CONFIG_PATH,
    PdfProcessError,
    build_path_config,
    extract_pdf_fields,
    filter_valid_patient_dirs,
    iter_exam_dirs,
    iter_patient_dirs,
    iter_pdf_files,
    load_valid_patient_ids,
)


@dataclass
class ConflictDetail:
    patient_id: str
    exam_id: str
    key: str
    value_to_pdfs: dict[str, list[str]] = field(default_factory=dict)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="按扫描顺序提取前 5 个冲突键的详细过程")
    parser.add_argument("--config", type=Path, default=CONFIG_PATH, help="路径配置文件")
    parser.add_argument("--dataset-root", type=Path, default=None, help="可选：覆盖 dataset_root")
    parser.add_argument(
        "--patient-validity-table",
        type=Path,
        default=None,
        help="可选：覆盖 patient_validity_table",
    )
    parser.add_argument(
        "--max-patients",
        type=int,
        default=None,
        help="可选：限制扫描患者数（按目录排序后截断）",
    )
    parser.add_argument("--limit", type=int, default=5, help="需要提取的冲突键数量，默认 5")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    limit = max(0, args.limit)

    path_config = build_path_config(
        config_path=args.config,
        dataset_root=args.dataset_root,
        patient_validity_table=args.patient_validity_table,
    )
    valid_patient_ids = load_valid_patient_ids(path_config.patient_validity_table)

    patient_dirs = filter_valid_patient_dirs(iter_patient_dirs(path_config.dataset_root), valid_patient_ids)
    if args.max_patients is not None and args.max_patients > 0:
        patient_dirs = patient_dirs[: args.max_patients]

    conflict_details: list[ConflictDetail] = []
    recorded_keys: set[tuple[str, str, str]] = set()
    scanned_exam_count = 0
    scanned_pdf_count = 0

    for patient_dir in patient_dirs:
        for exam_dir in iter_exam_dirs(patient_dir):
            scanned_exam_count += 1
            key_value_to_pdfs: dict[str, dict[str, list[str]]] = defaultdict(lambda: defaultdict(list))

            for pdf_path in iter_pdf_files(exam_dir):
                scanned_pdf_count += 1
                try:
                    fields = extract_pdf_fields(pdf_path)
                except (PdfProcessError, Exception):
                    continue

                for key, value in fields.items():
                    if not value:
                        continue
                    key_value_to_pdfs[key][value].append(pdf_path.name)

                    value_map = key_value_to_pdfs[key]
                    if len(value_map) > 1:
                        marker = (patient_dir.name, exam_dir.name, key)
                        if marker in recorded_keys:
                            continue

                        conflict_details.append(
                            ConflictDetail(
                                patient_id=patient_dir.name,
                                exam_id=exam_dir.name,
                                key=key,
                                value_to_pdfs={k: sorted(v) for k, v in value_map.items()},
                            )
                        )
                        recorded_keys.add(marker)

                        if len(conflict_details) >= limit:
                            break

                if len(conflict_details) >= limit:
                    break

            if len(conflict_details) >= limit:
                break

        if len(conflict_details) >= limit:
            break

    print(f"扫描患者数：{len(patient_dirs)}")
    print(f"已扫描检查目录数：{scanned_exam_count}")
    print(f"已扫描 PDF 数：{scanned_pdf_count}")
    print(f"命中冲突键数量：{len(conflict_details)}（limit={limit}）")

    if not conflict_details:
        print("未找到冲突键。")
        return

    print("\n冲突键详细过程（按扫描顺序）：")
    for idx, detail in enumerate(conflict_details, start=1):
        print(f"{idx}. 患者={detail.patient_id} | 检查={detail.exam_id} | 键={detail.key}")
        for value, pdf_names in sorted(detail.value_to_pdfs.items(), key=lambda x: x[0]):
            print(f"   - 值：{value}")
            print(f"     来源 PDF：{', '.join(pdf_names)}")


if __name__ == "__main__":
    main()
