import inspect

from .gastro_max_pool_baseline import GastroMaxPoolBaseline
from .gastro_mean_pool_baseline import GastroMeanPoolBaseline
from .gastro_mil_baseline import GastroAttentionMILBaseline, GastroMILBaseline
from .gastro_topk_mil_baseline import GastroTopKMILBaseline
from .gastro_transformer_mil_baseline import GastroTransformerMILBaseline

GASTRO_BASELINE_CLASS_REGISTRY = {
    "gastro_baseline": GastroAttentionMILBaseline,
    "gastro_attention_mil_baseline": GastroAttentionMILBaseline,
    "gastro_mean_pool_baseline": GastroMeanPoolBaseline,
    "gastro_max_pool_baseline": GastroMaxPoolBaseline,
    "gastro_transformer_mil_baseline": GastroTransformerMILBaseline,
    "gastro_topk_mil_baseline": GastroTopKMILBaseline,
}

GASTRO_BASELINE_MODEL_NAMES = (
    "gastro_attention_mil_baseline",
    "gastro_mean_pool_baseline",
    "gastro_max_pool_baseline",
    "gastro_transformer_mil_baseline",
    "gastro_topk_mil_baseline",
)


def build_gastro_baseline(model_name: str, **kwargs):
    if model_name not in GASTRO_BASELINE_CLASS_REGISTRY:
        raise ValueError(f"未知胃镜 baseline 模型名: {model_name}")
    model_cls = GASTRO_BASELINE_CLASS_REGISTRY[model_name]
    valid_kwargs = {
        key: value
        for key, value in kwargs.items()
        if key in inspect.signature(model_cls.__init__).parameters
    }
    return model_cls(**valid_kwargs)


__all__ = [
    "GASTRO_BASELINE_CLASS_REGISTRY",
    "GASTRO_BASELINE_MODEL_NAMES",
    "GastroAttentionMILBaseline",
    "GastroMILBaseline",
    "GastroMeanPoolBaseline",
    "GastroMaxPoolBaseline",
    "GastroTransformerMILBaseline",
    "GastroTopKMILBaseline",
    "build_gastro_baseline",
]
