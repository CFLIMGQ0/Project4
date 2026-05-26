from __future__ import annotations

from baselines.task1.gastro_baseline import (
    GASTRO_BASELINE_CLASS_REGISTRY,
    GASTRO_BASELINE_MODEL_NAMES,
    GastroAttentionMILBaseline,
    GastroMILBaseline,
    GastroMaxPoolBaseline,
    GastroMeanPoolBaseline,
    GastroTopKMILBaseline,
    GastroTransformerMILBaseline,
    build_gastro_baseline,
)

TASK2_BASELINE_CLASS_REGISTRY: dict[str, object] = dict(GASTRO_BASELINE_CLASS_REGISTRY)
TASK2_BASELINE_MODEL_NAMES: tuple[str, ...] = tuple(GASTRO_BASELINE_MODEL_NAMES)

__all__ = [
    "TASK2_BASELINE_CLASS_REGISTRY",
    "TASK2_BASELINE_MODEL_NAMES",
    "GastroAttentionMILBaseline",
    "GastroMILBaseline",
    "GastroMeanPoolBaseline",
    "GastroMaxPoolBaseline",
    "GastroTransformerMILBaseline",
    "GastroTopKMILBaseline",
    "build_gastro_baseline",
]
