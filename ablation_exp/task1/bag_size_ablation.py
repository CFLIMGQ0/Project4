from __future__ import annotations


EXPERIMENT_NAME = "bag_size_ablation"


def build_experiment() -> dict:
    base_model_name = "gastro_label_graph_mil"
    return {
        "name": EXPERIMENT_NAME,
        "display_name": "Bag 数量消融",
        "output_dir_name": "bag_size_ablation",
        "goal": "考察训练阶段每个 bag 的实例数量变化对胃镜标签图 MIL 的影响",
        "models": [
            {
                "name": "train_max_instances_08",
                "base_model_name": base_model_name,
                "display_name": "bag=8, batch_instances=96",
                "enabled": True,
                "model_params": {},
                "run_overrides": {
                    "train_max_instances": 8,
                    "train_max_batch_instances": 96,
                },
            },
            {
                "name": "train_max_instances_12",
                "base_model_name": base_model_name,
                "display_name": "bag=12, batch_instances=144",
                "enabled": True,
                "model_params": {},
                "run_overrides": {
                    "train_max_instances": 12,
                    "train_max_batch_instances": 144,
                },
            },
            {
                "name": "train_max_instances_16",
                "base_model_name": base_model_name,
                "display_name": "bag=16, batch_instances=192",
                "enabled": True,
                "model_params": {},
                "run_overrides": {
                    "train_max_instances": 16,
                    "train_max_batch_instances": 192,
                },
            },
            {
                "name": "train_max_instances_20",
                "base_model_name": base_model_name,
                "display_name": "bag=20, batch_instances=240",
                "enabled": True,
                "model_params": {},
                "run_overrides": {
                    "train_max_instances": 20,
                    "train_max_batch_instances": 240,
                },
            },
            {
                "name": "train_max_instances_24",
                "base_model_name": base_model_name,
                "display_name": "bag=24, batch_instances=288",
                "enabled": True,
                "model_params": {},
                "run_overrides": {
                    "train_max_instances": 24,
                    "train_max_batch_instances": 288,
                },
            },
        ],
    }

