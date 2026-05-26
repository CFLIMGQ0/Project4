import inspect

from .db_mil import DBMILModel
from .aslt_mil import ASLTMILModel
from .laca_mil import LACAMILModel
from .cl_mil import CLMILModel
from .csml_mil import CSMLMILModel

EXP1_BASELINE_CLASS_REGISTRY = {
    "db_mil": DBMILModel,
    "aslt_mil": ASLTMILModel,
    "laca_mil": LACAMILModel,
    "cl_mil": CLMILModel,
    "csml_mil": CSMLMILModel,
}

EXP1_BASELINE_MODEL_NAMES = (
    "db_mil",
    "aslt_mil",
    "laca_mil",
    "cl_mil",
    "csml_mil",
)


def build_exp1_baseline(model_name: str, **kwargs):
    if model_name not in EXP1_BASELINE_CLASS_REGISTRY:
        raise ValueError(f"未知 exp1 baseline 模型名: {model_name}")
    model_cls = EXP1_BASELINE_CLASS_REGISTRY[model_name]
    valid_kwargs = {
        key: value
        for key, value in kwargs.items()
        if key in inspect.signature(model_cls.__init__).parameters
    }
    return model_cls(**valid_kwargs)


__all__ = [
    "EXP1_BASELINE_CLASS_REGISTRY",
    "EXP1_BASELINE_MODEL_NAMES",
    "DBMILModel",
    "ASLTMILModel",
    "LACAMILModel",
    "CLMILModel",
    "CSMLMILModel",
    "build_exp1_baseline",
]
