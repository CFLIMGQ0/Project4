from __future__ import annotations

from tasks import DEFAULT_GASTRO_TASK_NAME, get_task_spec

from .task1 import TASK1_ABLATION_EXPERIMENT_REGISTRY
from .task2 import TASK2_ABLATION_EXPERIMENT_REGISTRY

TASK_ABLATION_EXPERIMENT_REGISTRY = {
    "task1": TASK1_ABLATION_EXPERIMENT_REGISTRY,
    "task2": TASK2_ABLATION_EXPERIMENT_REGISTRY,
}


def get_ablation_experiment_registry(task_name: str = DEFAULT_GASTRO_TASK_NAME) -> dict[str, object]:
    spec = get_task_spec(task_name)
    return TASK_ABLATION_EXPERIMENT_REGISTRY.get(spec.name, {})


def list_ablation_experiment_names(task_name: str = DEFAULT_GASTRO_TASK_NAME) -> tuple[str, ...]:
    return tuple(get_ablation_experiment_registry(task_name).keys())


ABLATION_EXPERIMENT_NAMES = list_ablation_experiment_names(DEFAULT_GASTRO_TASK_NAME)
ABLATION_EXPERIMENT_REGISTRY = get_ablation_experiment_registry(DEFAULT_GASTRO_TASK_NAME)


def build_ablation_experiment(experiment_name: str, task_name: str = DEFAULT_GASTRO_TASK_NAME) -> dict:
    registry = get_ablation_experiment_registry(task_name)
    if experiment_name not in registry:
        raise ValueError(f"任务 {get_task_spec(task_name).name} 不支持消融实验: {experiment_name}")
    return registry[experiment_name]()


def build_all_ablation_experiments(task_name: str = DEFAULT_GASTRO_TASK_NAME) -> list[dict]:
    return [build_ablation_experiment(name, task_name=task_name) for name in list_ablation_experiment_names(task_name)]


__all__ = [
    "TASK_ABLATION_EXPERIMENT_REGISTRY",
    "ABLATION_EXPERIMENT_REGISTRY",
    "ABLATION_EXPERIMENT_NAMES",
    "get_ablation_experiment_registry",
    "list_ablation_experiment_names",
    "build_ablation_experiment",
    "build_all_ablation_experiments",
]
