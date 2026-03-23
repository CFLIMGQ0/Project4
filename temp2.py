from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
from typing import Any

from check_pdf import CONFIG_PATH, PdfReader
from statiscs import build_path_config, iter_exam_dirs, iter_patient_dirs, iter_pdf_files

TARGET_WARNING_PATTERNS = (
    "Ignoring wrong pointing object",
    "Object ID",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="定位会触发 pypdf 解析告警或解析失败的 PDF")
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
    return parser.parse_args()


class LogCaptureHandler(logging.Handler):
    def __init__(self) -> None:
        super().__init__(level=logging.WARNING)
        self.messages: list[str] = []

    def emit(self, record: logging.LogRecord) -> None:
        self.messages.append(record.getMessage())


def inspect_pdf_with_pypdf(pdf_path: Path) -> tuple[list[str], str | None]:
    if PdfReader is None:
        return [], "当前环境未安装 pypdf，无法定位解析告警来源"

    handler = LogCaptureHandler()
    logger_names = ("pypdf", "PyPDF2")
    original_states: list[tuple[logging.Logger, list[logging.Handler], int, bool]] = []
    try:
        for logger_name in logger_names:
            logger = logging.getLogger(logger_name)
            original_states.append((logger, list(logger.handlers), logger.level, logger.propagate))
            logger.handlers = [handler]
            logger.setLevel(logging.WARNING)
            logger.propagate = False

        reader = PdfReader(str(pdf_path))
        reader.get_fields()
        len(reader.pages)
    except Exception as exc:
        return handler.messages, str(exc)
    finally:
        for logger, handlers, level, propagate in original_states:
            logger.handlers = handlers
            logger.setLevel(level)
            logger.propagate = propagate

    return handler.messages, None


def is_target_warning(message: str) -> bool:
    return any(pattern in message for pattern in TARGET_WARNING_PATTERNS)


def main() -> None:
    args = parse_args()
    path_config = build_path_config(args.config, args.dataset_root)

    if not path_config.dataset_root.exists():
        print(f"数据集根目录不存在：{path_config.dataset_root}")
        return
    if not path_config.dataset_root.is_dir():
        print(f"数据集根路径不是目录：{path_config.dataset_root}")
        return

    patient_dirs = iter_patient_dirs(path_config.dataset_root)
    if args.max_patients is not None and args.max_patients > 0:
        patient_dirs = patient_dirs[:args.max_patients]

    warning_records: list[dict[str, Any]] = []
    error_records: list[dict[str, str]] = []
    total_pdf_count = 0

    for patient_dir in patient_dirs:
        for exam_dir in iter_exam_dirs(patient_dir):
            for pdf_path in iter_pdf_files(exam_dir):
                total_pdf_count += 1
                warning_messages, error_message = inspect_pdf_with_pypdf(pdf_path)
                matched_warnings = [message for message in warning_messages if is_target_warning(message)]

                if matched_warnings:
                    warning_records.append(
                        {
                            "patient_id": patient_dir.name,
                            "exam_id": exam_dir.name,
                            "pdf_path": str(pdf_path),
                            "warnings": matched_warnings,
                        }
                    )

                if error_message is not None:
                    error_records.append(
                        {
                            "patient_id": patient_dir.name,
                            "exam_id": exam_dir.name,
                            "pdf_path": str(pdf_path),
                            "reason": error_message,
                        }
                    )

    print(f"扫描 PDF 总数：{total_pdf_count}")
    print(f"触发解析告警的 PDF 数：{len(warning_records)}")
    print(f"解析失败 PDF 数：{len(error_records)}")

    if warning_records:
        print("\n触发解析告警的 PDF 明细：")
        print(json.dumps(warning_records, ensure_ascii=False, indent=2))
        print("\n告警 PDF 路径列表：")
        for index, item in enumerate(warning_records, start=1):
            print(f"{index}. {item['pdf_path']}")
            print(f"   患者：{item['patient_id']}")
            print(f"   检查：{item['exam_id']}")
            for warning_index, warning in enumerate(item["warnings"], start=1):
                print(f"   告警{warning_index}：{warning}")
    else:
        print("没有发现触发目标解析告警的 PDF。")

    if error_records:
        print("\n解析失败 PDF 明细：")
        print(json.dumps(error_records, ensure_ascii=False, indent=2))
        print("\n失败 PDF 路径列表：")
        for index, item in enumerate(error_records, start=1):
            print(f"{index}. {item['pdf_path']}")
            print(f"   患者：{item['patient_id']}")
            print(f"   检查：{item['exam_id']}")
            print(f"   原因：{item['reason']}")
    else:
        print("没有发现解析失败的 PDF。")


if __name__ == "__main__":
    main()
