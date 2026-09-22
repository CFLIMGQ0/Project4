#!/usr/bin/env python3
"""训练 AMOS-MM 七标签与 MR-RATE 四标签的位置编码替换模型。"""
from __future__ import annotations

import argparse
import csv
import json
import os
import shutil
import statistics
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

VARIANTS = {
    "original_pe": "Original PE",
    "apro_absolute_only": "AMEF APro absolute-only",
    "apro_relative_only": "AMEF APro relative-only",
    "comrope_ap": "ComRoPE-AP",
    "videorope": "VideoRoPE",
    "path": "PaTH",
    "dape_v2_kerple": "DAPE V2-Kerple",
}
DATASETS = {"amos_mm", "mr_rate"}
MODEL_KEY = "amef_multimodal"
OUTPUTS = {
    "amos_mm": ROOT / "outputs/amos_mm/position_replacements_7_labels",
    "mr_rate": ROOT / "outputs/mr_rate_1k/position_replacements",
}


def save_json(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + f".{os.getpid()}.tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")
    temporary.replace(path)


def stable_hash(path: Path) -> str:
    import hashlib

    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024**2), b""):
            digest.update(block)
    return digest.hexdigest()


def stable_sources(dataset: str) -> dict[str, str]:
    relative = [
        "src/scripts/run_amos_mrrate_position_replacements.py",
        "src/scripts/run_amos_mm_all_models.py",
        "src/scripts/amos_mm_model_adapters.py",
        "src/scripts/run_mrrate_1k_amef.py",
        "src/scripts/run_cq500_table2_multimodal_baselines.py",
        "src/scripts/task3_apro_cope_ablation_scheduler.py",
        "src/exp_8/models.py",
        "src/exp_8/position_baselines.py",
        "src/training/losses.py",
    ]
    if dataset == "amos_mm":
        relative.extend([
            "src/scripts/prepare_amos_mm_experiments.py",
            "outputs/amos_mm/experiment/preparation_protocol.json",
            "outputs/amos_mm/experiment/samples.json",
            "outputs/amos_mm/experiment/splits.json",
            "outputs/amos_mm/experiment/image_exclusions.json",
        ])
    else:
        relative.extend([
            "src/scripts/prepare_mrrate_1k_experiment.py",
            "outputs/mr_rate_1k/experiment/preparation_protocol.json",
            "outputs/mr_rate_1k/experiment/samples.json",
            "outputs/mr_rate_1k/experiment/patient_folds.json",
        ])
    return {item: stable_hash(ROOT / item) for item in relative}


def params_for(variant: str) -> dict:
    from scripts.task3_apro_cope_ablation_scheduler import base_model_params

    if variant in {"apro_absolute_only", "apro_relative_only"}:
        params = base_model_params(variant)
    else:
        params = base_model_params("original_pe" if variant == "original_pe" else "no_pe")
    if variant not in {"original_pe", "apro_absolute_only", "apro_relative_only"}:
        params["ct_attention_variant"] = variant
    import yaml

    main = yaml.safe_load((ROOT / "src/configs/task3/t3_main_model.yaml").read_text())
    params["label_query_consistency_weight"] = float(
        main["model"]["params"]["label_query_consistency_weight"]
    )
    params["image_aux_weight"] = 0.0
    return params


def protocol(dataset: str, variant: str) -> dict:
    if dataset == "amos_mm":
        labels = ["liver", "kidney", "gallbladder", "spleen", "bowel", "pancreas", "stomach"]
        split = "AMOS-MM固定开发/人工测试协议；七标签；五个开发折轮换验证，固定200例人工测试集"
        input_note = "沿用AMOS-MM部分标签掩码、冻结轴位视觉特征、masked-ASL和完整图文路径"
    else:
        labels = [
            "PP_Unspecific_bucket", "PP_Neurodegenerative",
            "PP_Cerebrovascular", "PP_Neoplastic",
        ]
        split = "MR-RATE-1K固定五折；test=k，validation=(k+1)%5，其余三折训练"
        input_note = "沿用MR-RATE-1K冻结视觉特征、findings掩码文本和64实例上限"
    value = {
        "dataset": dataset,
        "labels": labels,
        "model": {MODEL_KEY: VARIANTS[variant]},
        "position_variant": variant,
        "position_module": "原生APro-CoPE分支" if variant in {"apro_absolute_only", "apro_relative_only"} else ("Original PE" if variant == "original_pe" else VARIANTS[variant]),
        "position_input": "所有分支只使用删除后重新编号的序列槽位；原始索引和总数只用于真值评估",
        "split": split,
        "input": input_note,
        "model_parameters": params_for(variant),
        "training": "沿用对应数据集AMEF训练循环；不使用测试标签选择模型，不新增位置回归头",
        "source_sha256": stable_sources(dataset),
    }
    value["protocol_sha256"] = stable_hash_bytes(value)
    return value


def stable_hash_bytes(value: dict) -> str:
    import hashlib

    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False).encode()).hexdigest()


def output_dir(dataset: str, variant: str) -> Path:
    return OUTPUTS[dataset] / variant


def configure_amos(variant: str):
    import run_amos_mm_all_models as core
    from exp_8.position_baselines import replace_slice_attention

    core.torch.backends.mha.set_fastpath_enabled(False)
    core.OUT = output_dir("amos_mm", variant)
    core.INPUT = ROOT / "outputs/amos_mm/experiment"
    core.MODELS = {MODEL_KEY: VARIANTS[variant]}
    core.IMAGE_MODELS = {}
    core.TEXT_MODELS = {}
    core.fusion_base.MODELS = {}
    core.parameters = lambda key: params_for(variant)

    def build_model(key, count, vocabulary_size=8192):
        if key != MODEL_KEY:
            raise ValueError(key)
        params = dict(params_for(variant))
        params.pop("ct_attention_variant", None)
        weights = {name.removesuffix("_weight"): params.pop(name)
                   for name in list(params) if name.endswith("_weight")}
        model = core.PartialLabelAMEF(**params, pretrained=False, num_labels=count)
        model.instance_encoder.backbone = core.image_base.CachedFeatureBackbone(
            768, params["feature_dim"], params["dropout"]
        )
        if variant not in {"original_pe", "apro_absolute_only", "apro_relative_only"}:
            replace_slice_attention(model, variant)
        return model, weights

    core.build_model = build_model
    return core


def configure_mr(variant: str):
    import run_mrrate_1k_amef as core
    from exp_8.position_baselines import replace_slice_attention

    core.OUTPUT_ROOT = output_dir("mr_rate", variant)
    core.MODELS = {MODEL_KEY: VARIANTS[variant]}
    core.INPUT = ROOT / "outputs/mr_rate_1k/experiment"
    core.model_parameters = lambda: params_for(variant)

    def build_model(key, parameters):
        import torch
        from exp_8.models import Exp12AProCoPEWatchCrossAttentionTextCNNModel

        if key != MODEL_KEY:
            raise ValueError(key)
        params = dict(parameters)
        params.pop("ct_attention_variant", None)
        auxiliary = {name.removesuffix("_weight"): params.pop(name)
                      for name in list(params) if name.endswith("_weight")}
        model = Exp12AProCoPEWatchCrossAttentionTextCNNModel(
            **params, pretrained=False, num_labels=len(core.LABELS)
        )
        model.instance_encoder.backbone = core.CachedFeatureBackbone(
            input_dim=768, output_dim=params["feature_dim"], dropout=params["dropout"]
        )
        if variant not in {"original_pe", "apro_absolute_only", "apro_relative_only"}:
            replace_slice_attention(model, variant)
        torch.backends.mha.set_fastpath_enabled(False)
        return model, auxiliary

    core.build_model = build_model
    core.configure_base()
    return core


def amos_paths(variant: str, fold: int) -> list[str]:
    base = f"outputs/amos_mm/position_replacements_7_labels/{variant}/7_labels/fold_{fold}/amef_multimodal"
    return [f"{base}/{name}" for name in ("result.json", "best_model.pt", "test_predictions.npz", "history.json", "config.json")]


def mr_paths(variant: str, fold: int) -> list[str]:
    base = f"outputs/mr_rate_1k/position_replacements/{variant}/fold_{fold}/amef_multimodal"
    return [f"{base}/{name}" for name in ("completed.json", "best_model.pt", "test_metrics.json", "test_predictions.csv", "validation_predictions.csv", "history.json", "config.json")]


def initialize(dataset: str, variant: str) -> None:
    folder = output_dir(dataset, variant)
    folder.mkdir(parents=True, exist_ok=True)
    save_json(folder / "protocol.json", protocol(dataset, variant))
    jobs = [{"dataset": dataset, "variant": variant, "fold": fold} for fold in range(1, 6)]
    save_json(folder / "jobs.json", jobs)


def check_protocol(folder: Path, dataset: str, variant: str) -> str:
    current = protocol(dataset, variant)
    saved = json.loads((folder / "protocol.json").read_text())
    if saved != current:
        raise RuntimeError(f"协议不一致：{folder}")
    return current["protocol_sha256"]


def run_amos(variant: str, fold: int) -> None:
    core = configure_amos(variant)
    folder = output_dir("amos_mm", variant)
    digest = check_protocol(folder, "amos_mm", variant)
    job_folder = folder / "7_labels" / f"fold_{fold}" / MODEL_KEY
    result_path = job_folder / "result.json"
    if result_path.exists():
        recorded = json.loads(result_path.read_text())
        if recorded.get("protocol_sha256") == digest:
            return
        shutil.rmtree(job_folder)
    rows = json.loads((core.INPUT / "samples.json").read_text())
    splits = json.loads((core.INPUT / "splits.json").read_text())
    bags = core.load_features(rows)
    job = {"labels": 7, "model": MODEL_KEY, "fold": fold - 1}
    core.train_job(job, rows, splits, bags, digest, f"{variant}_{fold}")


def run_mr(variant: str, fold: int) -> None:
    core = configure_mr(variant)
    folder = output_dir("mr_rate", variant)
    digest = check_protocol(folder, "mr_rate", variant)
    marker = folder / f"fold_{fold}" / MODEL_KEY / "completed.json"
    if marker.exists():
        recorded = json.loads((marker.parent / "test_metrics.json").read_text())
        if recorded.get("protocol_sha256") == digest:
            return
        shutil.rmtree(marker.parent)
    rows, folds, targets = core.read_inputs()
    bags = core.load_features(rows)
    ids = np.arange(len(rows), dtype=np.int64)
    text_ids, text_mask = core.fusion_base.encode_descriptions(
        {row["case_index"]: row["findings_masked"] for row in rows}, ids
    )
    args = argparse.Namespace(output_dir=folder)
    core.fusion_base.train_fold(
        args, MODEL_KEY, fold - 1, ids, bags, targets, folds,
        text_ids, text_mask, core.model_parameters(), digest
    )


def aggregate_amos(variant: str) -> None:
    folder = output_dir("amos_mm", variant)
    digest = check_protocol(folder, "amos_mm", variant)
    results = []
    for fold in range(1, 6):
        path = folder / "7_labels" / f"fold_{fold}" / MODEL_KEY / "result.json"
        if path.exists():
            item = json.loads(path.read_text())
            results.append(item)
    if len(results) != 5:
        raise RuntimeError(f"AMOS {variant}仅完成{len(results)}/5折")
    accepted_protocols = sorted({digest, *(item["protocol_sha256"] for item in results)})
    summary = {
        "dataset": "AMOS-MM",
        "labels": 7,
        "position_variant": variant,
        "model": VARIANTS[variant],
        "completed_folds": 5,
        "macro_f1_mean": statistics.mean(item["macro_f1"] for item in results),
        "macro_f1_std": statistics.stdev(item["macro_f1"] for item in results),
        "folds": results,
        "protocol_sha256": digest,
        "accepted_protocol_hashes": accepted_protocols,
    }
    save_json(folder / "summary.json", summary)
    with (folder / "summary.csv").open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=["position_variant", "macro_f1_mean", "macro_f1_std", "completed_folds"])
        writer.writeheader()
        writer.writerow({k: summary[k] for k in writer.fieldnames})


def aggregate_mr(variant: str) -> None:
    core = configure_mr(variant)
    folder = output_dir("mr_rate", variant)
    digest = check_protocol(folder, "mr_rate", variant)
    accepted = {digest}
    for fold in range(1, 6):
        path = folder / f"fold_{fold}" / MODEL_KEY / "test_metrics.json"
        if path.exists():
            accepted.add(json.loads(path.read_text())["protocol_sha256"])
    args = argparse.Namespace(output_dir=folder, accepted_protocols=accepted)
    if core.aggregate(args, digest) != 5:
        raise RuntimeError(f"MR-RATE {variant}五折尚未完成")


def smoke_test(dataset: str, variant: str) -> None:
    import torch
    from training.losses import AsymmetricLossMultiLabel

    core = configure_amos(variant) if dataset == "amos_mm" else configure_mr(variant)
    count = 7 if dataset == "amos_mm" else 4
    params = params_for(variant)
    model, weights = (core.build_model(MODEL_KEY, count, 8192)
                      if dataset == "amos_mm" else core.build_model(MODEL_KEY, params))
    model.cuda().train()
    images = torch.randn(2, 8, 768, 1, 1, device="cuda")
    mask = torch.ones(2, 8, dtype=torch.bool, device="cuda")
    tokens = torch.randint(1, 128, (2, 64), device="cuda")
    token_mask = torch.ones(2, 64, dtype=torch.bool, device="cuda")
    positions = torch.arange(8, device="cuda").repeat(2, 1)
    counts = torch.full((2,), 8, dtype=torch.long, device="cuda")
    targets = torch.randint(0, 2, (2, count), device="cuda").float()
    known = torch.ones_like(targets, dtype=torch.bool)
    labels = (targets, known) if dataset == "amos_mm" else targets
    output = model(images=images, mask=mask, watch_token_ids=tokens, watch_token_mask=token_mask,
                   instance_indices=positions, original_image_counts=counts, labels=labels)
    loss = AsymmetricLossMultiLabel()(output["logits"], targets)
    for name, weight in weights.items():
        if weight:
            loss = loss + float(weight) * output["aux_losses"][name]
    if not torch.isfinite(loss):
        raise RuntimeError(f"{dataset}/{variant}损失非有限")
    loss.backward()
    gradients = [parameter.grad for parameter in model.parameters() if parameter.grad is not None]
    if not gradients or not all(torch.isfinite(gradient).all() for gradient in gradients):
        raise RuntimeError(f"{dataset}/{variant}梯度非有限")
    print(json.dumps({"dataset": dataset, "variant": variant, "finite_loss": True,
                      "finite_gradients": True, "loss": float(loss.detach())}, ensure_ascii=False))
    del model, output, loss
    torch.cuda.empty_cache()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", choices=sorted(DATASETS), required=True)
    parser.add_argument("--variant", choices=sorted(VARIANTS), required=True)
    parser.add_argument("--fold", type=int, choices=range(1, 6))
    parser.add_argument("--initialize", action="store_true")
    parser.add_argument("--aggregate", action="store_true")
    parser.add_argument("--audit-only", action="store_true")
    parser.add_argument("--smoke-test", action="store_true")
    args = parser.parse_args()
    initialize(args.dataset, args.variant)
    if args.audit_only:
        print(json.dumps(protocol(args.dataset, args.variant), ensure_ascii=False, indent=2))
    elif args.smoke_test:
        smoke_test(args.dataset, args.variant)
    elif args.aggregate:
        (aggregate_amos if args.dataset == "amos_mm" else aggregate_mr)(args.variant)
    elif args.fold:
        (run_amos if args.dataset == "amos_mm" else run_mr)(args.variant, args.fold)
    elif not args.initialize:
        parser.error("请选择 --fold、--aggregate、--audit-only 或 --initialize")


if __name__ == "__main__":
    main()
