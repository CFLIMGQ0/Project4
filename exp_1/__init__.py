from .idea import (
    EXP1_IDEA_CLASS_REGISTRY,
    EXP1_IDEA_MODEL_NAMES,
    OTMILModel,
    IBMILModel,
    HypMILModel,
    SDMILModel,
    EDLMILModel,
    build_exp1_idea,
)
from .baseline import (
    EXP1_BASELINE_CLASS_REGISTRY,
    EXP1_BASELINE_MODEL_NAMES,
    DBMILModel,
    ASLTMILModel,
    LACAMILModel,
    CLMILModel,
    CSMLMILModel,
    build_exp1_baseline,
)

EXP1_CLASS_REGISTRY = {**EXP1_IDEA_CLASS_REGISTRY, **EXP1_BASELINE_CLASS_REGISTRY}
EXP1_MODEL_NAMES = EXP1_IDEA_MODEL_NAMES + EXP1_BASELINE_MODEL_NAMES


def build_exp1_model(model_name: str, **kwargs):
    if model_name in EXP1_IDEA_CLASS_REGISTRY:
        return build_exp1_idea(model_name, **kwargs)
    if model_name in EXP1_BASELINE_CLASS_REGISTRY:
        return build_exp1_baseline(model_name, **kwargs)
    raise ValueError(f"未知 exp1 模型名: {model_name}")


__all__ = [
    "EXP1_CLASS_REGISTRY",
    "EXP1_MODEL_NAMES",
    "EXP1_IDEA_CLASS_REGISTRY",
    "EXP1_IDEA_MODEL_NAMES",
    "EXP1_BASELINE_CLASS_REGISTRY",
    "EXP1_BASELINE_MODEL_NAMES",
    "OTMILModel",
    "IBMILModel",
    "HypMILModel",
    "SDMILModel",
    "EDLMILModel",
    "DBMILModel",
    "ASLTMILModel",
    "LACAMILModel",
    "CLMILModel",
    "CSMLMILModel",
    "build_exp1_model",
    "build_exp1_idea",
    "build_exp1_baseline",
]
