import inspect

from .clam_mb_sota import GastroCLAMMBSOTA
from .clam_sb_sota import GastroCLAMSBSOTA
from .dsmil_sota import GastroDSMILSOTA
from .dtfd_mil_sota import GastroDTFDMILSOTA
from .transmil_sota import GastroTransMILSOTA

GASTRO_SOTA_CLASS_REGISTRY = {
    "gastro_clam_sb_sota": GastroCLAMSBSOTA,
    "gastro_clam_mb_sota": GastroCLAMMBSOTA,
    "gastro_dsmil_sota": GastroDSMILSOTA,
    "gastro_transmil_sota": GastroTransMILSOTA,
    "gastro_dtfd_mil_sota": GastroDTFDMILSOTA,
}

GASTRO_SOTA_MODEL_NAMES = (
    "gastro_clam_sb_sota",
    "gastro_clam_mb_sota",
    "gastro_dsmil_sota",
    "gastro_transmil_sota",
    "gastro_dtfd_mil_sota",
)


def build_gastro_sota(model_name: str, **kwargs):
    if model_name not in GASTRO_SOTA_CLASS_REGISTRY:
        raise ValueError(f"未知胃镜 SOTA 模型名: {model_name}")
    model_cls = GASTRO_SOTA_CLASS_REGISTRY[model_name]
    valid_kwargs = {
        key: value
        for key, value in kwargs.items()
        if key in inspect.signature(model_cls.__init__).parameters
    }
    return model_cls(**valid_kwargs)


__all__ = [
    "GASTRO_SOTA_CLASS_REGISTRY",
    "GASTRO_SOTA_MODEL_NAMES",
    "GastroCLAMSBSOTA",
    "GastroCLAMMBSOTA",
    "GastroDSMILSOTA",
    "GastroTransMILSOTA",
    "GastroDTFDMILSOTA",
    "build_gastro_sota",
]
