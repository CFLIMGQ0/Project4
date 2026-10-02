#!/usr/bin/env python3
"""在三个公开数据集上筛选 ACPE 与标签证据聚合的最小交互。"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys

import numpy as np
import torch
import yaml

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "src/scripts"))

from exp_8.acpe_interactions import LabelPositionInteractionMixin, LabelPositionInteractionModel
from run_physionet_ct_ich_table2_baselines import CachedFeatureBackbone
from scripts.task3_apro_cope_ablation_scheduler import base_model_params
from amos_mm_model_adapters import PartialLabelAMEF

MODEL_KEY = "amef_multimodal"
VARIANTS = ("centroid", "attention_bias")
DATASETS = ("ct_rate", "amos_mm", "mr_rate_1k")
OUT = ROOT / "outputs/acpe_autoresearch/training"


class PartialLabelPositionInteraction(LabelPositionInteractionMixin, PartialLabelAMEF):
    """AMOS-MM 部分标签监督对应的交互变体。"""


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def parameters(variant: str) -> dict:
    main = yaml.safe_load((ROOT / "src/configs/task3/t3_main_model.yaml").read_text())
    params = base_model_params("apro_full")
    params["label_query_consistency_weight"] = float(
        main["model"]["params"]["label_query_consistency_weight"]
    )
    params["image_aux_weight"] = 0.0
    params["interaction_variant"] = variant
    return params


def output_root(dataset: str, variant: str) -> Path:
    return OUT / variant / dataset


def protocol(dataset: str, variant: str) -> dict:
    paths = [
        ROOT / "src/scripts/run_acpe_system_interaction.py",
        ROOT / "src/exp_8/acpe_interactions.py",
        ROOT / "src/exp_8/models.py",
        ROOT / "src/scripts/task3_apro_cope_ablation_scheduler.py",
        ROOT / "src/scripts/run_cq500_table2_multimodal_baselines.py",
        ROOT / "src/training/losses.py",
        ROOT / "src/configs/task3/t3_main_model.yaml",
    ]
    dataset_paths = {
        "ct_rate": ["src/scripts/run_ctrate_680_all_models.py", "outputs/ct_rate_680/experiment/patient_folds.json", "outputs/ct_rate_680/experiment/preparation_protocol.json"],
        "amos_mm": ["src/scripts/run_amos_mm_all_models.py", "src/scripts/amos_mm_model_adapters.py", "outputs/amos_mm/experiment/splits.json", "outputs/amos_mm/experiment/preparation_protocol.json"],
        "mr_rate_1k": ["src/scripts/run_mrrate_1k_amef.py", "outputs/mr_rate_1k/experiment/patient_folds.json", "outputs/mr_rate_1k/experiment/preparation_protocol.json"],
    }
    paths.extend(ROOT / item for item in dataset_paths[dataset])
    result = {
        "dataset": dataset,
        "variant": variant,
        "model": MODEL_KEY,
        "parameters": parameters(variant),
        "training": "沿用现有三数据集各自的完整30轮训练、原划分、验证集选模及阈值；本轮先执行第1折筛选",
        "purpose": "仅在原ACPE后给标签级视觉聚合加入一个轻量位置交互；不改报告输入或标签来源",
        "sources": {str(path.relative_to(ROOT)): sha256(path) for path in paths},
    }
    result["protocol_sha256"] = hashlib.sha256(
        json.dumps(result, sort_keys=True, ensure_ascii=False).encode()
    ).hexdigest()
    return result


def initialize(dataset: str, variant: str) -> str:
    folder = output_root(dataset, variant)
    folder.mkdir(parents=True, exist_ok=True)
    value = protocol(dataset, variant)
    saved = folder / "protocol.json"
    if saved.exists():
        existing = json.loads(saved.read_text())
        if existing != value:
            raise RuntimeError(f"协议或源码已变化，拒绝混用结果：{saved}")
    else:
        saved.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n")
    return value["protocol_sha256"]


def build_model(dataset: str, variant: str, count: int):
    params = dict(parameters(variant))
    auxiliary = {key.removesuffix("_weight"): params.pop(key)
                 for key in list(params) if key.endswith("_weight")}
    klass = PartialLabelPositionInteraction if dataset == "amos_mm" else LabelPositionInteractionModel
    model = klass(**params, pretrained=False, num_labels=count)
    model.instance_encoder.backbone = CachedFeatureBackbone(
        input_dim=768, output_dim=params["feature_dim"], dropout=params["dropout"]
    )
    return model, auxiliary


def run_ct(variant: str, fold: int, digest: str) -> None:
    import run_ctrate_680_all_models as core

    core.MODELS = {MODEL_KEY: f"ACPE + {variant}"}
    core.IMAGE_MODELS = {}
    core.TEXT_MODELS = {}
    core.FUSION_MODELS = {}
    core.parameters = lambda: {MODEL_KEY: parameters(variant)}
    core.protocol = lambda: protocol("ct_rate", variant)
    core.build_model = lambda key, _params: build_model("ct_rate", variant, 3)
    core.configure_adapters()
    args = argparse.Namespace(
        output_dir=output_root("ct_rate", variant),
        models=[MODEL_KEY], devices=list(range(5)), worker_index=fold - 1,
    )
    assert digest == core.protocol()["protocol_sha256"]
    core.worker(args)


def run_amos(variant: str, fold: int, digest: str) -> None:
    import run_amos_mm_all_models as core

    core.OUT = output_root("amos_mm", variant)
    core.INPUT = ROOT / "outputs/amos_mm/experiment"
    core.MODELS = {MODEL_KEY: f"ACPE + {variant}"}
    core.IMAGE_MODELS = {}
    core.TEXT_MODELS = {}
    core.fusion_base.MODELS = {}
    core.parameters = lambda _key: parameters(variant)
    core.build_model = lambda key, count, vocabulary_size=8192: build_model("amos_mm", variant, count)
    rows = json.loads((core.INPUT / "samples.json").read_text())
    splits = json.loads((core.INPUT / "splits.json").read_text())
    bags = core.load_features(rows)
    job = {"labels": 7, "model": MODEL_KEY, "fold": fold - 1}
    core.train_job(job, rows, splits, bags, digest, f"acpe_{variant}_{fold}")


def run_mr(variant: str, fold: int, digest: str) -> None:
    import run_mrrate_1k_amef as core

    core.OUTPUT_ROOT = output_root("mr_rate_1k", variant)
    core.INPUT = ROOT / "outputs/mr_rate_1k/experiment"
    core.MODELS = {MODEL_KEY: f"ACPE + {variant}"}
    core.model_parameters = lambda: parameters(variant)
    core.build_model = lambda key, _params: build_model("mr_rate_1k", variant, 4)
    core.configure_base()
    rows, folds, targets = core.read_inputs()
    bags = core.load_features(rows)
    ids = np.arange(len(rows), dtype=np.int64)
    text_ids, text_mask = core.fusion_base.encode_descriptions(
        {row["case_index"]: row["findings_masked"] for row in rows}, ids
    )
    args = argparse.Namespace(output_dir=core.OUTPUT_ROOT)
    core.fusion_base.train_fold(
        args, MODEL_KEY, fold - 1, ids, bags, targets, folds,
        text_ids, text_mask, core.model_parameters(), digest,
    )


def smoke(dataset: str, variant: str) -> None:
    torch.set_num_threads(2)
    torch.backends.mha.set_fastpath_enabled(False)
    count = {"ct_rate": 3, "amos_mm": 7, "mr_rate_1k": 4}[dataset]
    model, _ = build_model(dataset, variant, count)
    model.cuda().train()
    images = torch.randn(2, 8, 768, 1, 1, device="cuda")
    mask = torch.ones(2, 8, dtype=torch.bool, device="cuda")
    token_ids = torch.randint(1, 128, (2, 32), device="cuda")
    positions = torch.arange(8, device="cuda").repeat(2, 1)
    targets = torch.randint(0, 2, (2, count), device="cuda").float()
    labels = (targets, torch.ones_like(targets, dtype=torch.bool)) if dataset == "amos_mm" else targets
    out = model(
        images=images, mask=mask, labels=labels,
        watch_token_ids=token_ids, watch_token_mask=torch.ones_like(token_ids, dtype=torch.bool),
        instance_indices=positions, original_image_counts=torch.full((2,), 8, device="cuda"),
    )
    loss = torch.nn.functional.binary_cross_entropy_with_logits(out["logits"], targets)
    loss.backward()
    if not torch.isfinite(loss):
        raise FloatingPointError("诊断损失非有限")
    print(f"{dataset}/{variant} 前向及反向检查通过，loss={float(loss):.4f}", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", choices=DATASETS, required=True)
    parser.add_argument("--variant", choices=VARIANTS, required=True)
    parser.add_argument("--fold", type=int, choices=range(1, 6), default=1)
    parser.add_argument("--initialize-only", action="store_true")
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()
    digest = initialize(args.dataset, args.variant)
    if args.initialize_only:
        print(digest)
        return
    if args.smoke:
        smoke(args.dataset, args.variant)
        return
    torch.set_num_threads(4)
    torch.backends.mha.set_fastpath_enabled(False)
    {"ct_rate": run_ct, "amos_mm": run_amos, "mr_rate_1k": run_mr}[args.dataset](
        args.variant, args.fold, digest
    )


if __name__ == "__main__":
    main()
