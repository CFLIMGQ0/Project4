from __future__ import annotations

from .base import TaskSpec
from .task1 import TASK1_SPEC
from .task2 import TASK2_SPEC

TASK_REGISTRY: dict[str, TaskSpec] = {
    TASK1_SPEC.name: TASK1_SPEC,
    TASK2_SPEC.name: TASK2_SPEC,
}

TASK_ALIAS_MAP = {
    "task1_gastro3": TASK1_SPEC.name,
    "task2_gastro3": TASK2_SPEC.name,
}

DEFAULT_GASTRO_TASK_NAME = TASK2_SPEC.name


def normalize_task_name(task_name: str) -> str:
    normalized = str(task_name).strip()
    if not normalized:
        raise ValueError("任务名不能为空")
    normalized = TASK_ALIAS_MAP.get(normalized, normalized)
    if normalized not in TASK_REGISTRY:
        raise ValueError(f"未知任务名: {task_name}")
    return normalized

def get_task_spec(task_name: str) -> TaskSpec:
    return TASK_REGISTRY[normalize_task_name(task_name)]


def list_task_specs() -> list[TaskSpec]:
    return list(TASK_REGISTRY.values())


def is_gastro_task(task_name: str) -> bool:
    return get_task_spec(task_name).family == "gastro"
