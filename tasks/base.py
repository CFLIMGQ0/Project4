from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class TaskSpec:
    name: str
    display_name: str
    family: str
    task_type: str
    task_dir_name: str
    run_prefix: str
    data_subdir: str
    datalist_filename: str
    label_names: tuple[str, ...] = ()
    class_names: tuple[str, ...] = ()
    default_report_csv: str | None = None

    @property
    def is_multilabel(self) -> bool:
        return self.task_type == "gastro_multilabel"

    @property
    def is_binary(self) -> bool:
        return self.task_type == "colonoscopy_binary"

    @property
    def num_labels(self) -> int:
        if self.is_multilabel:
            return len(self.label_names)
        if self.is_binary:
            return 1
        raise ValueError(f"未知 task_type: {self.task_type}")
