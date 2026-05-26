from __future__ import annotations

from .task1_model_ablation import LABEL_GRAPH_ABLATION_NAME
from .task1_model_ablation import build_label_graph_ablation

TASK1_ABLATION_EXPERIMENT_REGISTRY = {
    LABEL_GRAPH_ABLATION_NAME: build_label_graph_ablation,
}

TASK1_ABLATION_EXPERIMENT_NAMES = tuple(TASK1_ABLATION_EXPERIMENT_REGISTRY.keys())

__all__ = [
    "TASK1_ABLATION_EXPERIMENT_REGISTRY",
    "TASK1_ABLATION_EXPERIMENT_NAMES",
]
