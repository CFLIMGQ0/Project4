import inspect

from .cturs_mil import CTURSMILModel
from .eqlv2_mil import EQLV2MILModel
from .ldam_mil import LDAMMILModel
from .logitnorm_mil import LogitNormMILModel
from .ral_mil import RALMILModel

EXP2_BASELINE_CLASS_REGISTRY = {
    "cturs_mil": CTURSMILModel,
    "eqlv2_mil": EQLV2MILModel,
    "ldam_mil": LDAMMILModel,
    "logitnorm_mil": LogitNormMILModel,
    "ral_mil": RALMILModel,
}

EXP2_BASELINE_MODEL_NAMES = (
    "cturs_mil",
    "eqlv2_mil",
    "ldam_mil",
    "logitnorm_mil",
    "ral_mil",
)


def build_exp2_baseline(model_name: str, **kwargs):
    if model_name not in EXP2_BASELINE_CLASS_REGISTRY:
        raise ValueError(f"未知 exp2 baseline 模型名: {model_name}")
    model_cls = EXP2_BASELINE_CLASS_REGISTRY[model_name]
    valid_kwargs = {
        key: value
        for key, value in kwargs.items()
        if key in inspect.signature(model_cls.__init__).parameters
    }
    return model_cls(**valid_kwargs)


__all__ = [
    "CTURSMILModel",
    "EQLV2MILModel",
    "EXP2_BASELINE_CLASS_REGISTRY",
    "EXP2_BASELINE_MODEL_NAMES",
    "LDAMMILModel",
    "LogitNormMILModel",
    "RALMILModel",
    "build_exp2_baseline",
]
