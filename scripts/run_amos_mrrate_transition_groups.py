#!/usr/bin/env python3
"""AMOS-MM七标签与MR-RATE-1K四标签的ACPE过渡输入组合五折实验。"""
from __future__ import annotations

import argparse
import csv
import json
import statistics
from pathlib import Path

import run_amos_mrrate_position_replacements as base

ROOT = base.ROOT
MODEL_KEY = base.MODEL_KEY
DATASETS = base.DATASETS
VARIANTS = {
    "1": ([1], False),
    "1+1": ([1], True),
    "123+1": ([1, 2, 3], True),
    "145+1": ([1, 4, 5], True),
    "1236+1": ([1, 2, 3, 6], True),
    "1456+1": ([1, 4, 5, 6], True),
}
DISPLAY_NAMES = {key: f"ACPE transition {key}" for key in VARIANTS}
OUTPUTS = {
    "amos_mm": ROOT / "outputs/amos_mm/transition_groups_7_labels",
    "mr_rate": ROOT / "outputs/mr_rate_1k/transition_groups",
}
ORIGINAL_STABLE_SOURCES = base.stable_sources


def params_for(variant: str) -> dict:
    from scripts.task3_apro_cope_ablation_scheduler import base_model_params
    import yaml

    groups, include_gap = VARIANTS[variant]
    params = base_model_params("apro_full")
    params.update(
        apro_transition_dim=64,
        apro_fourier_mlp_layers=2,
        apro_transition_groups=groups,
        apro_transition_include_gap=include_gap,
        image_aux_weight=0.0,
    )
    main = yaml.safe_load((ROOT / "src/configs/task3/t3_main_model.yaml").read_text())
    params["label_query_consistency_weight"] = float(
        main["model"]["params"]["label_query_consistency_weight"]
    )
    return params


def stable_sources(dataset: str) -> dict[str, str]:
    sources = ORIGINAL_STABLE_SOURCES(dataset)
    sources["src/scripts/run_amos_mrrate_transition_groups.py"] = base.stable_hash(Path(__file__))
    return sources


def protocol(dataset: str, variant: str) -> dict:
    labels = (
        ["liver", "kidney", "gallbladder", "spleen", "bowel", "pancreas", "stomach"]
        if dataset == "amos_mm"
        else [
            "PP_Unspecific_bucket",
            "PP_Neurodegenerative",
            "PP_Cerebrovascular",
            "PP_Neoplastic",
        ]
    )
    value = {
        "dataset": dataset,
        "labels": labels,
        "model": {MODEL_KEY: DISPLAY_NAMES[variant]},
        "position_variant": "apro_full",
        "transition_ablation": {
            "variant": variant,
            "groups": VARIANTS[variant][0],
            "include_gap_in_mlp": VARIANTS[variant][1],
            "input_dim": len(VARIANTS[variant][0]) * 64 + int(VARIANTS[variant][1]),
            "transition_dim": 64,
            "hidden_dim": 64,
            "output_dim": 1,
            "selection": "仅拼接被选的原生六组过渡输入，不改变各组计算定义",
            "other_modules": "原始双支ACPE、Fourier双层投影、alpha=1.5、position_dim=64",
        },
        "position_input": "沿用公开数据集原训练接口的完整序列槽位；不新增输入或校准",
        "split": (
            "AMOS-MM七标签固定开发折与固定200例人工测试集"
            if dataset == "amos_mm"
            else "MR-RATE-1K固定五折；test=k，validation=(k+1)%5，其余三折训练"
        ),
        "training": "沿用现有AMEF训练循环、30 epochs、seed=42+100*fold；测试集不参与选择",
        "model_parameters": params_for(variant),
        "source_sha256": stable_sources(dataset),
    }
    value["protocol_sha256"] = base.stable_hash_bytes(value)
    return value


def output_dir(dataset: str, variant: str) -> Path:
    return OUTPUTS[dataset] / variant


def configure_amos(variant: str):
    import run_amos_mm_all_models as core

    core.torch.backends.mha.set_fastpath_enabled(False)
    core.OUT = output_dir("amos_mm", variant)
    core.INPUT = ROOT / "outputs/amos_mm/experiment"
    core.MODELS = {MODEL_KEY: DISPLAY_NAMES[variant]}
    core.IMAGE_MODELS = {}
    core.TEXT_MODELS = {}
    core.fusion_base.MODELS = {}
    core.parameters = lambda key: params_for(variant)

    def build_model(key, count, vocabulary_size=8192):
        if key != MODEL_KEY:
            raise ValueError(key)
        params = dict(params_for(variant))
        weights = {
            name.removesuffix("_weight"): params.pop(name)
            for name in list(params)
            if name.endswith("_weight")
        }
        model = core.PartialLabelAMEF(**params, pretrained=False, num_labels=count)
        model.instance_encoder.backbone = core.image_base.CachedFeatureBackbone(
            768, params["feature_dim"], params["dropout"]
        )
        return model, weights

    core.build_model = build_model
    return core


def configure_mr(variant: str):
    import run_mrrate_1k_amef as core
    import torch
    from exp_8.models import Exp12AProCoPEWatchCrossAttentionTextCNNModel

    core.OUTPUT_ROOT = output_dir("mr_rate", variant)
    core.MODELS = {MODEL_KEY: DISPLAY_NAMES[variant]}
    core.INPUT = ROOT / "outputs/mr_rate_1k/experiment"
    core.model_parameters = lambda: params_for(variant)

    def build_model(key, parameters):
        if key != MODEL_KEY:
            raise ValueError(key)
        params = dict(parameters)
        auxiliary = {
            name.removesuffix("_weight"): params.pop(name)
            for name in list(params)
            if name.endswith("_weight")
        }
        model = Exp12AProCoPEWatchCrossAttentionTextCNNModel(
            **params, pretrained=False, num_labels=len(core.LABELS)
        )
        model.instance_encoder.backbone = core.CachedFeatureBackbone(
            input_dim=768, output_dim=params["feature_dim"], dropout=params["dropout"]
        )
        torch.backends.mha.set_fastpath_enabled(False)
        return model, auxiliary

    core.build_model = build_model
    core.configure_base()
    return core


def install_base_hooks() -> None:
    base.VARIANTS = DISPLAY_NAMES
    base.OUTPUTS = OUTPUTS
    base.params_for = params_for
    base.stable_sources = stable_sources
    base.protocol = protocol
    base.output_dir = output_dir
    base.configure_amos = configure_amos
    base.configure_mr = configure_mr


def aggregate_all() -> None:
    rows = []
    original_check_protocol = base.check_protocol
    base.check_protocol = lambda folder, dataset, variant: json.loads(
        (folder / "protocol.json").read_text()
    )["protocol_sha256"]
    try:
        for dataset in ("amos_mm", "mr_rate"):
            for variant in VARIANTS:
                if dataset == "amos_mm":
                    base.aggregate_amos(variant)
                    summary = json.loads((output_dir(dataset, variant) / "summary.json").read_text())
                else:
                    base.aggregate_mr(variant)
                    summary = json.loads((output_dir(dataset, variant) / "summary.json").read_text())[0]
                groups, include_gap = VARIANTS[variant]
                rows.append({
                    "dataset": "AMOS-MM" if dataset == "amos_mm" else "MR-RATE-1K",
                    "variant": variant,
                    "input_dim": len(groups) * 64 + int(include_gap),
                    "macro_f1_mean": summary["macro_f1_mean"],
                    "macro_f1_std": summary["macro_f1_std"],
                    "macro_f1_fixed_0_5_mean": summary.get("macro_f1_fixed_0_5_mean"),
                    "macro_f1_fixed_0_5_std": summary.get("macro_f1_fixed_0_5_std"),
                    "oof_macro_f1": summary.get("oof_macro_f1"),
                })
    finally:
        base.check_protocol = original_check_protocol
    root = ROOT / "outputs/public_transition_groups"
    root.mkdir(parents=True, exist_ok=True)
    with (root / "comparison.csv").open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    lines = [
        "# AMOS-MM 与 MR-RATE-1K ACPE transition 输入组合五折结果",
        "",
        "六种配置均使用现有公开数据集、现有五折划分和现有AMEF训练循环；均值±标准差为五个测试折的样本标准差（ddof=1）。",
        "",
        "| 数据集 | 输入组合 | 拼接维度 | Macro-F1 | 固定0.5 Macro-F1 | OOF Macro-F1 |",
        "|---|---|---:|---:|---:|---:|",
    ]
    for row in rows:
        fixed = "NA" if row["macro_f1_fixed_0_5_mean"] is None else (
            f"{row['macro_f1_fixed_0_5_mean']:.4f} ± {row['macro_f1_fixed_0_5_std']:.4f}"
        )
        oof = "NA" if row["oof_macro_f1"] is None else f"{row['oof_macro_f1']:.4f}"
        lines.append(
            f"| {row['dataset']} | {row['variant']} | {row['input_dim']} | "
            f"{row['macro_f1_mean']:.4f} ± {row['macro_f1_std']:.4f} | "
            f"{fixed} | {oof} |"
        )
    lines.extend([
        "",
        "①当前投影特征；②前向差分；③后向差分；④②绝对值；⑤③绝对值；⑥②与③逐元素乘积。",
        "`+1`表示把原始归一化采集间隔输入transition MLP；其余ACPE位置路径保持不变。",
    ])
    (root / "results.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines), flush=True)


def main() -> None:
    install_base_hooks()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", choices=sorted(DATASETS))
    parser.add_argument("--variant", choices=sorted(VARIANTS))
    parser.add_argument("--fold", type=int, choices=range(1, 6))
    parser.add_argument("--initialize", action="store_true")
    parser.add_argument("--aggregate-all", action="store_true")
    parser.add_argument("--audit-all", action="store_true")
    parser.add_argument("--smoke-test", action="store_true")
    args = parser.parse_args()
    if args.audit_all:
        for dataset in sorted(DATASETS):
            for variant in VARIANTS:
                base.initialize(dataset, variant)
                print(f"{dataset}/{variant}: {protocol(dataset, variant)['protocol_sha256']}")
        return
    if args.aggregate_all:
        aggregate_all()
        return
    if not args.dataset or not args.variant:
        parser.error("请指定--dataset和--variant")
    base.initialize(args.dataset, args.variant)
    if args.initialize:
        return
    if args.smoke_test:
        base.smoke_test(args.dataset, args.variant)
    elif args.fold:
        (base.run_amos if args.dataset == "amos_mm" else base.run_mr)(args.variant, args.fold)
    else:
        parser.error("请指定--fold、--initialize或--smoke-test")


if __name__ == "__main__":
    main()
