import inspect

from .ot_mil import OTMILModel
from .ib_mil import IBMILModel
from .hyp_mil import HypMILModel
from .sd_mil import SDMILModel
from .edl_mil import EDLMILModel

EXP1_IDEA_CLASS_REGISTRY = {
    "ot_mil": OTMILModel,
    "ib_mil": IBMILModel,
    "hyp_mil": HypMILModel,
    "sd_mil": SDMILModel,
    "edl_mil": EDLMILModel,
}

EXP1_IDEA_MODEL_NAMES = (
    "ot_mil",
    "ib_mil",
    "hyp_mil",
    "sd_mil",
    "edl_mil",
)


def build_exp1_idea(model_name: str, **kwargs):
    if model_name not in EXP1_IDEA_CLASS_REGISTRY:
        raise ValueError(f"未知 exp1 idea 模型名: {model_name}")
    model_cls = EXP1_IDEA_CLASS_REGISTRY[model_name]
    valid_kwargs = {
        key: value
        for key, value in kwargs.items()
        if key in inspect.signature(model_cls.__init__).parameters
    }
    return model_cls(**valid_kwargs)


__all__ = [
    "EXP1_IDEA_CLASS_REGISTRY",
    "EXP1_IDEA_MODEL_NAMES",
    "OTMILModel",
    "IBMILModel",
    "HypMILModel",
    "SDMILModel",
    "EDLMILModel",
    "build_exp1_idea",
]
