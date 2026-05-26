from __future__ import annotations


def _build_task2_rg_hmil_entry(
    *,
    name: str,
    display_name: str,
    use_text_guidance: bool,
    use_region_grouping: bool,
    use_conditional_graph: bool,
    use_hierarchy: bool,
) -> dict:
    return {
        "name": name,
        "display_name": display_name,
        "enabled": True,
        "base_model_name": "rg_hmil",
        "model_params": {
            "backbone_name": "convnext_tiny",
            "freeze_stages": 1,
            "feature_dim": 512,
            "attn_dim": 256,
            "num_regions": 6,
            "condition_dim": 128,
            "dropout": 0.2,
            "use_text_guidance": use_text_guidance,
            "use_region_grouping": use_region_grouping,
            "use_conditional_graph": use_conditional_graph,
            "use_hierarchy": use_hierarchy,
            "region_cls_weight": 0.5,
            "relevance_weight": 0.5,
        },
        "run_overrides": {},
    }


def build_ablation_static_lgr() -> dict:
    return {
        "name": "ablation_static_lgr",
        "display_name": "静态基线消融",
        "output_dir_name": "ablation_static_lgr",
        "goal": "验证关闭文本引导、区域分组和条件图后，RG-HMIL 退化配置的表现。",
        "models": [
            _build_task2_rg_hmil_entry(
                name="ablation_static_lgr",
                display_name="Static LGR baseline",
                use_text_guidance=False,
                use_region_grouping=False,
                use_conditional_graph=False,
                use_hierarchy=False,
            )
        ],
    }


def build_ablation_text_guidance() -> dict:
    return {
        "name": "ablation_text_guidance",
        "display_name": "文本引导消融",
        "output_dir_name": "ablation_text_guidance",
        "goal": "验证仅开启文本引导相关模块时的增益。",
        "models": [
            _build_task2_rg_hmil_entry(
                name="ablation_text_guidance",
                display_name="Text guidance only",
                use_text_guidance=True,
                use_region_grouping=False,
                use_conditional_graph=False,
                use_hierarchy=False,
            )
        ],
    }


def build_ablation_region_grouping() -> dict:
    return {
        "name": "ablation_region_grouping",
        "display_name": "区域分组消融",
        "output_dir_name": "ablation_region_grouping",
        "goal": "验证在文本引导基础上加入区域分组后的增益。",
        "models": [
            _build_task2_rg_hmil_entry(
                name="ablation_region_grouping",
                display_name="Text guidance + region grouping",
                use_text_guidance=True,
                use_region_grouping=True,
                use_conditional_graph=False,
                use_hierarchy=False,
            )
        ],
    }


def build_ablation_conditional_graph() -> dict:
    return {
        "name": "ablation_conditional_graph",
        "display_name": "条件图消融",
        "output_dir_name": "ablation_conditional_graph",
        "goal": "验证在区域分组基础上加入条件图推理但关闭层次门控时的增益。",
        "models": [
            _build_task2_rg_hmil_entry(
                name="ablation_conditional_graph",
                display_name="Text guidance + region grouping + conditional graph",
                use_text_guidance=True,
                use_region_grouping=True,
                use_conditional_graph=True,
                use_hierarchy=False,
            )
        ],
    }


def build_rg_hmil_full() -> dict:
    return {
        "name": "rg_hmil_full",
        "display_name": "完整 RG-HMIL",
        "output_dir_name": "rg_hmil_full",
        "goal": "验证开启全部模块后的完整 RG-HMIL 表现。",
        "models": [
            _build_task2_rg_hmil_entry(
                name="rg_hmil_full",
                display_name="Full RG-HMIL",
                use_text_guidance=True,
                use_region_grouping=True,
                use_conditional_graph=True,
                use_hierarchy=True,
            )
        ],
    }


TASK2_ABLATION_EXPERIMENT_REGISTRY: dict[str, object] = {
    "ablation_static_lgr": build_ablation_static_lgr,
    "ablation_text_guidance": build_ablation_text_guidance,
    "ablation_region_grouping": build_ablation_region_grouping,
    "ablation_conditional_graph": build_ablation_conditional_graph,
    "rg_hmil_full": build_rg_hmil_full,
}
TASK2_ABLATION_EXPERIMENT_NAMES: tuple[str, ...] = tuple(TASK2_ABLATION_EXPERIMENT_REGISTRY.keys())

__all__ = [
    "TASK2_ABLATION_EXPERIMENT_REGISTRY",
    "TASK2_ABLATION_EXPERIMENT_NAMES",
    "build_ablation_static_lgr",
    "build_ablation_text_guidance",
    "build_ablation_region_grouping",
    "build_ablation_conditional_graph",
    "build_rg_hmil_full",
]
