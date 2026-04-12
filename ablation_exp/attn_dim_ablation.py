from __future__ import annotations


EXPERIMENT_NAME = "attn_dim_ablation"


def build_experiment() -> dict:
    base_model_name = "gastro_label_graph_mil"
    return {
        "name": EXPERIMENT_NAME,
        "display_name": "注意力维度消融",
        "output_dir_name": "attn_dim_ablation",
        "goal": "考察多标签注意力模块的 attn_dim 对胃镜标签图 MIL 表现的影响",
        "models": [
            {
                "name": "attn_dim_064",
                "base_model_name": base_model_name,
                "display_name": "attn_dim=64",
                "enabled": True,
                "model_params": {
                    "attn_dim": 64,
                },
                "run_overrides": {},
            },
            {
                "name": "attn_dim_128",
                "base_model_name": base_model_name,
                "display_name": "attn_dim=128",
                "enabled": True,
                "model_params": {
                    "attn_dim": 128,
                },
                "run_overrides": {},
            },
            {
                "name": "attn_dim_256",
                "base_model_name": base_model_name,
                "display_name": "attn_dim=256",
                "enabled": True,
                "model_params": {
                    "attn_dim": 256,
                },
                "run_overrides": {},
            },
            {
                "name": "attn_dim_384",
                "base_model_name": base_model_name,
                "display_name": "attn_dim=384",
                "enabled": True,
                "model_params": {
                    "attn_dim": 384,
                },
                "run_overrides": {},
            },
            {
                "name": "attn_dim_512",
                "base_model_name": base_model_name,
                "display_name": "attn_dim=512",
                "enabled": True,
                "model_params": {
                    "attn_dim": 512,
                },
                "run_overrides": {},
            },
        ],
    }

