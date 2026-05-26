import inspect

from .anchor_mil import AnchorMILModel
from .cvar_mil import CVARMILModel
from .entmax_mil import EntmaxMILModel
from .fdr_mil import FDRMILModel
from .rpca_mil import RPCAMILModel

EXP2_IDEA_CLASS_REGISTRY = {
    "entmax_mil": EntmaxMILModel,
    "cvar_mil": CVARMILModel,
    "rpca_mil": RPCAMILModel,
    "fdr_mil": FDRMILModel,
    "anchor_mil": AnchorMILModel,
}

EXP2_IDEA_MODEL_NAMES = (
    "entmax_mil",
    "cvar_mil",
    "rpca_mil",
    "fdr_mil",
    "anchor_mil",
)


def build_exp2_idea(model_name: str, **kwargs):
    if model_name not in EXP2_IDEA_CLASS_REGISTRY:
        raise ValueError(f"未知 exp2 idea 模型名: {model_name}")
    model_cls = EXP2_IDEA_CLASS_REGISTRY[model_name]
    valid_kwargs = {
        key: value
        for key, value in kwargs.items()
        if key in inspect.signature(model_cls.__init__).parameters
    }
    return model_cls(**valid_kwargs)


__all__ = [
    "EXP2_IDEA_CLASS_REGISTRY",
    "EXP2_IDEA_MODEL_NAMES",
    "AnchorMILModel",
    "CVARMILModel",
    "EntmaxMILModel",
    "FDRMILModel",
    "RPCAMILModel",
    "build_exp2_idea",
]
