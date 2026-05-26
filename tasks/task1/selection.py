from __future__ import annotations

from collections import Counter

from ..base import TaskSpec
from ..common import (
    COMMON_OUTPUT_FIELDS,
    SelectionResult,
    UNCERTAIN_TOKENS,
    build_common_output_row,
    build_progress,
    contains_any,
    is_gastroscopy_record,
    normalize_text,
)

TASK1_LABEL_RULES: dict[str, list[str]] = {
    "label_esophageal_smt": [
        "食管smt",
        "食管黏膜下隆起",
        "食管隆起性病变",
        "食管smt(来源于黏膜肌层)",
        "食管smt(来源于固有肌层)",
        "食管smt(来源于黏膜下层)",
        "食管黏膜下肿物",
    ],
    "label_esophageal_mucosal_or_tumor": [
        "食管黏膜病变",
        "食管肿物",
        "食管黏膜病变(待病理)",
        "食管黏膜病变(性质待定)",
        "食管肿物(待病理)",
        "食管占位",
        "食管新生物",
    ],
    "label_gastritis": [
        "慢性胃炎",
        "慢性非活动性胃炎",
        "慢性活动性胃炎",
        "萎缩性胃炎",
        "糜烂性胃炎",
        "浅表性胃炎",
        "胆汁反流性胃炎",
        "胃炎",
        "c1",
        "c2",
        "c3",
        "o1",
        "o2",
        "o3",
    ],
}

TASK1_SPEC = TaskSpec(
    name="task1",
    display_name="TASK1 胃镜三标签",
    family="gastro",
    task_type="gastro_multilabel",
    task_dir_name="task1",
    run_prefix="task1",
    data_subdir="task1",
    datalist_filename="gastro_multilabel_task_datalist.csv",
    label_names=tuple(TASK1_LABEL_RULES.keys()),
    class_names=(),
    default_report_csv="valid_dicts_report.csv",
)


def _match_rule_map(text: str) -> dict[str, int]:
    return {label: int(any(keyword in text for keyword in keywords)) for label, keywords in TASK1_LABEL_RULES.items()}


def build_selection_result(rows: list[dict[str, str]], columns: dict[str, str | None]) -> SelectionResult:
    selected_rows: list[dict[str, object]] = []
    positive_counter: Counter[str] = Counter()
    exclude_counter: Counter[str] = Counter()
    total_candidates = 0
    image_sum = 0

    progress = build_progress(total=len(rows), desc="筛选 task1")
    try:
        for row in rows:
            common = build_common_output_row(row=row, columns=columns)
            report_title_norm = normalize_text(str(common["reportTitle"]))
            watch_result_norm = normalize_text(str(common["watchResult"]))
            if not is_gastroscopy_record(report_title_norm, watch_result_norm):
                progress.update(1)
                continue

            total_candidates += 1
            label_map = _match_rule_map(watch_result_norm)
            label_sum = sum(label_map.values())
            if not watch_result_norm:
                exclude_counter["empty_watchResult"] += 1
            elif label_sum <= 0:
                if contains_any(watch_result_norm, UNCERTAIN_TOKENS):
                    exclude_counter["uncertain_without_target_label"] += 1
                else:
                    exclude_counter["no_target_label"] += 1
            else:
                output_row = dict(common)
                output_row.update(label_map)
                output_row["label_sum"] = label_sum
                selected_rows.append(output_row)
                image_sum += int(common["img_num"])
                for label_name, label_value in label_map.items():
                    if label_value == 1:
                        positive_counter[label_name] += 1
            progress.update(1)
    finally:
        progress.close()

    fieldnames = [*COMMON_OUTPUT_FIELDS, *TASK1_SPEC.label_names, "label_sum"]
    return SelectionResult(
        rows=selected_rows,
        fieldnames=fieldnames,
        total_candidates=total_candidates,
        selected_count=len(selected_rows),
        selected_image_sum=image_sum,
        positive_counter=positive_counter,
        exclude_counter=exclude_counter,
    )
