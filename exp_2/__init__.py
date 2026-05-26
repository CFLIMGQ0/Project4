from .baseline import (
    CTURSMILModel,
    EQLV2MILModel,
    EXP2_BASELINE_CLASS_REGISTRY,
    EXP2_BASELINE_MODEL_NAMES,
    LDAMMILModel,
    LogitNormMILModel,
    RALMILModel,
    build_exp2_baseline,
)
from .idea import (
    AnchorMILModel,
    CVARMILModel,
    EntmaxMILModel,
    EXP2_IDEA_CLASS_REGISTRY,
    EXP2_IDEA_MODEL_NAMES,
    FDRMILModel,
    RPCAMILModel,
    build_exp2_idea,
)

EXP2_CLASS_REGISTRY = {**EXP2_IDEA_CLASS_REGISTRY, **EXP2_BASELINE_CLASS_REGISTRY}
EXP2_MODEL_NAMES = EXP2_IDEA_MODEL_NAMES + EXP2_BASELINE_MODEL_NAMES


def build_exp2_model(model_name: str, **kwargs):
    if model_name in EXP2_IDEA_CLASS_REGISTRY:
        return build_exp2_idea(model_name, **kwargs)
    if model_name in EXP2_BASELINE_CLASS_REGISTRY:
        return build_exp2_baseline(model_name, **kwargs)
    raise ValueError(f"未知 exp2 模型名: {model_name}")


__all__ = [
    "AnchorMILModel",
    "CTURSMILModel",
    "CVARMILModel",
    "EQLV2MILModel",
    "EXP2_BASELINE_CLASS_REGISTRY",
    "EXP2_BASELINE_MODEL_NAMES",
    "EXP2_CLASS_REGISTRY",
    "EXP2_IDEA_CLASS_REGISTRY",
    "EXP2_IDEA_MODEL_NAMES",
    "EXP2_MODEL_NAMES",
    "EntmaxMILModel",
    "FDRMILModel",
    "LDAMMILModel",
    "LogitNormMILModel",
    "RALMILModel",
    "RPCAMILModel",
    "build_exp2_baseline",
    "build_exp2_idea",
    "build_exp2_model",
]
