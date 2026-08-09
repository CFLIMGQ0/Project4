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
from .multimodal_sotas import (
    HasanImageTextFusion2024,
    MMFNet2024,
    MMTF2025,
    RadFuse2025,
    SAIF2025,
    TASK2_MULTIMODAL_SOTA_CLASS_REGISTRY,
    TASK2_MULTIMODAL_SOTA_MODEL_NAMES,
    build_task2_multimodal_sota,
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
    "TASK2_MULTIMODAL_SOTA_CLASS_REGISTRY",
    "TASK2_MULTIMODAL_SOTA_MODEL_NAMES",
    "HasanImageTextFusion2024",
    "MMFNet2024",
    "SAIF2025",
    "MMTF2025",
    "RadFuse2025",
    "build_task2_multimodal_sota",
]
