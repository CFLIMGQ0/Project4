from __future__ import annotations

import argparse
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

from statistics import extract_pdf_fields

EFFECTIVE_KEY_CN_MAP = {
    "reportTitle": "页面标题",
    "age": "年龄",
    "anesthesiologistName": "麻醉医生",
    "applyDeptName": "科室",
    "applyNo": "检查号",
    "badness": "不良反应",
    "bedId": "病床号",
    "checkTime": "检查日期",
    "condition": "患者一般情况",
    "doctorName": "报告医师",
    "endoscopeName": "镜号",
    "hisPatientId": "内镜号",
    "namePatient": "姓名",
    "narcosisType": "麻醉方式",
    "operation": "操作过程",
    "operationValue": "操作名称",
    "patientAreaName": "病区",
    "roomName": "诊间",
    "sex": "性别",
    "suggest": "注意事项",
    "watch": "内镜所见",
    "watchResult": "诊断",
    "archiveTime": "报告日期",
    "specimen": "活检部位",
    "admissionNo": "住院号",
    "hp": "HP(幽门螺旋杆菌)",
    "operationRemark": "操作过程备注",
    "patientType": "patientType",
    "score": "score",
}

DEFAULT_PDF_FILES = [
    "/home/Lim/datasets/project4/main_data/ZS09036474/ZS0049122068/pdf/ZS-W48202412064f8cce47f7924946a2e570c154a5c8c3.pdf",
    "/home/Lim/datasets/project4/main_data/ZS08004085/ZS0048989885/pdf/ZS-W482024122030b9595e531f4c31b4b0416f4f3323dc.pdf",
    "/home/Lim/datasets/project4/main_data/ZS09020808/ZS0053136773/pdf/ZS-C4820250822481f01beb4484321b99392ad340a423c.pdf",
]


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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="读取指定 PDF 表单信息，并统计 patientType/score/operation（限定 operationRemark 非空）。"
    )
    parser.add_argument(
        "--dataset-root",
        type=Path,
        default=Path("/home/Lim/datasets/project4/main_data"),
        help="数据根目录（默认 /home/Lim/datasets/project4/main_data）",
    )
    parser.add_argument(
        "--pdf-files",
        nargs="+",
        default=DEFAULT_PDF_FILES,
        help="要展示表单信息的 PDF 路径列表",
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


def print_selected_pdf_fields(pdf_files: list[Path]) -> None:
    print("\n=== 指定 PDF 的表单信息 ===")
    for pdf_path in pdf_files:
        print(f"\n文件：{pdf_path}")
        if not pdf_path.is_file():
            print("- 文件不存在")
            continue
        try:
            fields = extract_pdf_fields(pdf_path)
        except Exception as exc:  # noqa: BLE001
            print(f"- 读取失败：{exc}")
            continue

        if not fields:
            print("- 未解析到表单字段")
            continue

        for key in sorted(fields):
            value = normalize_text(fields.get(key, ""))
            cn_name = EFFECTIVE_KEY_CN_MAP.get(key)
            if cn_name:
                print(f"- {key}（{cn_name}）：{value}")
            else:
                print(f"- {key}：{value}")


def summarize_counter(counter: Counter[str], title: str) -> None:
    print(f"\n=== {title} ===")
    print(f"类型数量：{len(counter)}")
    if not counter:
        print("- 无非空值")
        return

    for value, count in sorted(counter.items(), key=lambda item: (-item[1], item[0])):
        print(f"- {value} * {count}")


def collect_dataset_statistics(dataset_root: Path) -> tuple[Counter[str], Counter[str], Counter[str]]:
    patient_type_counter: Counter[str] = Counter()
    score_counter: Counter[str] = Counter()
    operation_counter_when_remark_non_empty: Counter[str] = Counter()

    pdf_paths = list(iter_pdf_files(dataset_root))
    progress = ProgressTracker(total=len(pdf_paths), prefix="全量 PDF 扫描")

    for pdf_path in pdf_paths:
        try:
            fields = extract_pdf_fields(pdf_path)
        except Exception:  # noqa: BLE001
            progress.update()
            continue

        patient_type = normalize_text(fields.get("patientType", ""))
        if patient_type:
            patient_type_counter[patient_type] += 1

        score = normalize_text(fields.get("score", ""))
        if score:
            score_counter[score] += 1

        operation_remark = normalize_text(fields.get("operationRemark", ""))
        operation = normalize_text(fields.get("operation", ""))
        if operation_remark and operation:
            operation_counter_when_remark_non_empty[operation] += 1

        progress.update()

    progress.close()
    return patient_type_counter, score_counter, operation_counter_when_remark_non_empty


def main() -> None:
    args = parse_args()
    dataset_root = args.dataset_root.expanduser().resolve()
    if not dataset_root.is_dir():
        raise FileNotFoundError(f"数据根目录不存在：{dataset_root}")

    pdf_files = [Path(path).expanduser().resolve() for path in args.pdf_files]
    print_selected_pdf_fields(pdf_files)

    patient_type_counter, score_counter, operation_counter = collect_dataset_statistics(
        dataset_root
    )

    summarize_counter(patient_type_counter, "patientType 的类型数量与全部值")
    summarize_counter(score_counter, "score 的类型数量与全部值")
    summarize_counter(
        operation_counter,
        "operationRemark 非空时，operation（操作过程）的类型数量与全部值",
    )


if __name__ == "__main__":
    main()
