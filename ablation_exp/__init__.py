from __future__ import annotations

from .attn_dim_ablation import EXPERIMENT_NAME as ATTN_DIM_ABLATION_NAME
from .attn_dim_ablation import build_experiment as build_attn_dim_ablation
from .bag_size_ablation import EXPERIMENT_NAME as BAG_SIZE_ABLATION_NAME
from .bag_size_ablation import build_experiment as build_bag_size_ablation

ABLATION_EXPERIMENT_REGISTRY = {
    BAG_SIZE_ABLATION_NAME: build_bag_size_ablation,
    ATTN_DIM_ABLATION_NAME: build_attn_dim_ablation,
}

ABLATION_EXPERIMENT_NAMES = tuple(ABLATION_EXPERIMENT_REGISTRY.keys())


def build_ablation_experiment(experiment_name: str) -> dict:
    if experiment_name not in ABLATION_EXPERIMENT_REGISTRY:
        raise ValueError(f"未知消融实验名: {experiment_name}")
    return ABLATION_EXPERIMENT_REGISTRY[experiment_name]()


def build_all_ablation_experiments() -> list[dict]:
    return [build_ablation_experiment(name) for name in ABLATION_EXPERIMENT_NAMES]


__all__ = [
    "ABLATION_EXPERIMENT_REGISTRY",
    "ABLATION_EXPERIMENT_NAMES",
    "build_ablation_experiment",
    "build_all_ablation_experiments",
]

