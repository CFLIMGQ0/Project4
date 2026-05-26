from __future__ import annotations


LABEL_GRAPH_ABLATION_NAME = "exp_task1_ablation1"

BASE_GRAPH_PARAMS = {
    "backbone_name": "convnext_tiny",
    "freeze_stages": 1,
    "feature_dim": 512,
    "attn_dim": 256,
    "dropout": 0.2,
}

SUMMARY_ORDER = {
    "exp_task1_ablation1_full_label_graph": 1,
    "exp_task1_ablation1_wo_label_graph": 2,
    "exp_task1_ablation1_label_self_attention": 3,
    "exp_task1_ablation1_static_gcn": 4,
    "exp_task1_ablation1_dynamic_gat": 5,
    "exp_task1_ablation1_label_transformer": 6,
    "exp_task1_ablation1_low_rank_graph": 7,
    "exp_task1_ablation1_cosine_graph": 8,
    "exp_task1_ablation1_label_mlp_mixer": 9,
    "exp_task1_ablation1_label_hypergraph": 10,
}


def _metadata(
    *,
    experiment_name: str,
    display_name: str,
) -> dict:
    return {
        "experiment_name": experiment_name,
        "summary_name": display_name,
        "seed_group_name": experiment_name,
        "summary_order": SUMMARY_ORDER.get(experiment_name, 99),
    }


def _entry(
    *,
    name: str,
    display_name: str,
    model_params: dict,
    base_model_name: str = "gastro_label_graph_mil",
) -> dict:
    return {
        "name": name,
        "base_model_name": base_model_name,
        "display_name": display_name,
        "enabled": True,
        "metadata": _metadata(
            experiment_name=name,
            display_name=display_name,
        ),
        "model_params": model_params,
        "run_overrides": {},
    }


def full_label_graph_params(backbone_name: str = "convnext_tiny") -> dict:
    return {
        **BASE_GRAPH_PARAMS,
        "backbone_name": backbone_name,
        "use_label_graph": True,
        "label_graph_type": "learnable",
        "use_label_wise_attention": True,
        "attention_type": "label_specific",
        "pooling_type": "label_attention",
    }


def without_label_graph_params() -> dict:
    return {
        **BASE_GRAPH_PARAMS,
        "use_label_graph": False,
        "label_graph_type": "learnable",
        "use_label_wise_attention": True,
        "attention_type": "label_specific",
        "pooling_type": "label_attention",
    }


def self_attention_reasoner_params() -> dict:
    return {
        **BASE_GRAPH_PARAMS,
        "use_label_graph": True,
        "label_graph_type": "self_attention",
        "use_label_wise_attention": True,
        "attention_type": "label_specific",
        "pooling_type": "label_attention",
    }


def static_gcn_reasoner_params() -> dict:
    return {
        **BASE_GRAPH_PARAMS,
        "use_label_graph": True,
        "label_graph_type": "static_gcn",
        "use_label_wise_attention": True,
        "attention_type": "label_specific",
        "pooling_type": "label_attention",
    }


def dynamic_gat_reasoner_params() -> dict:
    return {
        **BASE_GRAPH_PARAMS,
        "use_label_graph": True,
        "label_graph_type": "dynamic_gat",
        "use_label_wise_attention": True,
        "attention_type": "label_specific",
        "pooling_type": "label_attention",
    }


def label_transformer_reasoner_params() -> dict:
    return {
        **BASE_GRAPH_PARAMS,
        "use_label_graph": True,
        "label_graph_type": "label_transformer",
        "label_graph_heads": 4,
        "use_label_wise_attention": True,
        "attention_type": "label_specific",
        "pooling_type": "label_attention",
    }


def low_rank_graph_reasoner_params() -> dict:
    return {
        **BASE_GRAPH_PARAMS,
        "use_label_graph": True,
        "label_graph_type": "low_rank_graph",
        "label_graph_rank": 2,
        "use_label_wise_attention": True,
        "attention_type": "label_specific",
        "pooling_type": "label_attention",
    }


def cosine_graph_reasoner_params() -> dict:
    return {
        **BASE_GRAPH_PARAMS,
        "use_label_graph": True,
        "label_graph_type": "cosine_graph",
        "use_label_wise_attention": True,
        "attention_type": "label_specific",
        "pooling_type": "label_attention",
    }


def label_mlp_mixer_reasoner_params() -> dict:
    return {
        **BASE_GRAPH_PARAMS,
        "use_label_graph": True,
        "label_graph_type": "label_mlp_mixer",
        "use_label_wise_attention": True,
        "attention_type": "label_specific",
        "pooling_type": "label_attention",
    }


def label_hypergraph_reasoner_params() -> dict:
    return {
        **BASE_GRAPH_PARAMS,
        "use_label_graph": True,
        "label_graph_type": "label_hypergraph",
        "label_hypergraph_edges": 2,
        "use_label_wise_attention": True,
        "attention_type": "label_specific",
        "pooling_type": "label_attention",
    }


def build_label_graph_ablation() -> dict:
    return {
        "name": LABEL_GRAPH_ABLATION_NAME,
        "display_name": "TASK1 Label Graph Reasoner 消融",
        "output_dir_name": "exp_task1_ablation1",
        "goal": "只验证 TASK1 主模型中 label graph reasoner 的作用：保留、移除，以及替换为多种常见标签关系建模结构。",
        "models": [
            _entry(
                name="exp_task1_ablation1_full_label_graph",
                display_name="Full Label Graph Reasoner",
                model_params=full_label_graph_params(),
            ),
            _entry(
                name="exp_task1_ablation1_wo_label_graph",
                display_name="w/o Label Graph Reasoner",
                model_params=without_label_graph_params(),
            ),
            _entry(
                name="exp_task1_ablation1_label_self_attention",
                display_name="Label Self-Attention Reasoner",
                model_params=self_attention_reasoner_params(),
            ),
            _entry(
                name="exp_task1_ablation1_static_gcn",
                display_name="Static Co-occurrence GCN Reasoner",
                model_params=static_gcn_reasoner_params(),
            ),
            _entry(
                name="exp_task1_ablation1_dynamic_gat",
                display_name="Dynamic Label GAT Reasoner",
                model_params=dynamic_gat_reasoner_params(),
            ),
            _entry(
                name="exp_task1_ablation1_label_transformer",
                display_name="Label Transformer Reasoner",
                model_params=label_transformer_reasoner_params(),
            ),
            _entry(
                name="exp_task1_ablation1_low_rank_graph",
                display_name="Low-Rank Label Graph Reasoner",
                model_params=low_rank_graph_reasoner_params(),
            ),
            _entry(
                name="exp_task1_ablation1_cosine_graph",
                display_name="Cosine Dynamic Graph Reasoner",
                model_params=cosine_graph_reasoner_params(),
            ),
            _entry(
                name="exp_task1_ablation1_label_mlp_mixer",
                display_name="Label MLP-Mixer Reasoner",
                model_params=label_mlp_mixer_reasoner_params(),
            ),
            _entry(
                name="exp_task1_ablation1_label_hypergraph",
                display_name="Label Hypergraph Reasoner",
                model_params=label_hypergraph_reasoner_params(),
            ),
        ],
    }
