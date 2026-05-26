from __future__ import annotations

from sotas.task1.gastro_sota import (
    GASTRO_SOTA_CLASS_REGISTRY,
    GASTRO_SOTA_MODEL_NAMES,
    GastroCLAMMBSOTA,
    GastroCLAMSBSOTA,
    GastroDSMILSOTA,
    GastroDTFDMILSOTA,
    GastroTransMILSOTA,
    build_gastro_sota,
)

TASK2_SOTA_CLASS_REGISTRY: dict[str, object] = dict(GASTRO_SOTA_CLASS_REGISTRY)
TASK2_SOTA_MODEL_NAMES: tuple[str, ...] = tuple(GASTRO_SOTA_MODEL_NAMES)

__all__ = [
    "TASK2_SOTA_CLASS_REGISTRY",
    "TASK2_SOTA_MODEL_NAMES",
    "GastroCLAMSBSOTA",
    "GastroCLAMMBSOTA",
    "GastroDSMILSOTA",
    "GastroTransMILSOTA",
    "GastroDTFDMILSOTA",
    "build_gastro_sota",
]
