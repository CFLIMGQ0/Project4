#!/usr/bin/env python3
"""在CT-RATE标准协议和固定缺失协议下训练四种ACPE替换模块。"""
from __future__ import annotations

import argparse
import csv
import gc
import json
from pathlib import Path

import numpy as np
import torch
import yaml

import run_ctrate_680_all_models as core
import run_cq500_amef_multimodal as amef
import run_ctrate_slice_missing as missing
from exp_8.position_baselines import replace_slice_attention
from scripts.task3_apro_cope_ablation_scheduler import base_model_params


ROOT = core.ROOT
MODEL_KEY = "amef_multimodal"
VARIANTS = {
    "comrope_ap": "ComRoPE-AP（CT-1D适配）",
    "videorope": "VideoRoPE（CT-1×1适配）",
    "path": "PaTH（CT双向适配）",
    "dape_v2_kerple": "DAPE V2-Kerple（CT双向1×3适配）",
}
CONDITIONS = {
    "standard": {
        "source": ROOT / "outputs/ct_rate_680/experiment",
        "output": ROOT / "outputs/ct_rate_680/position_baselines_standard",
        "reference": (
            ROOT / "outputs/ct_rate_680/all_models_fivefold/summary.json",
            ROOT / "outputs/ct_rate_680/amef_original_pe_fivefold/summary.json",
        ),
    },
    "missing25": {
        "source": ROOT / "outputs/ct_rate_680/missing25_raw_hidden_fivefold",
        "output": ROOT / "outputs/ct_rate_680/missing25_all_positions_fivefold",
    },
    "missing50": {
        "source": ROOT / "outputs/ct_rate_680/missing50_raw_hidden_fivefold",
        "output": ROOT / "outputs/ct_rate_680/missing50_all_positions_fivefold",
    },
    "missing75": {
        "source": ROOT / "outputs/ct_rate_680/missing75_raw_hidden_fivefold",
        "output": ROOT / "outputs/ct_rate_680/missing75_all_positions_fivefold",
    },
}

STANDARD_INPUT = core.INPUT
STANDARD_SETTINGS = dict(core.SETTINGS)
STANDARD_READ_INPUTS = core.read_inputs
STANDARD_LOAD_FEATURES = core.load_features
STANDARD_FORWARD = core.forward_batch
STANDARD_PROTOCOL = core.protocol

CONDITION = "standard"
VARIANT = "comrope_ap"
SOURCE = CONDITIONS[CONDITION]["source"]
OUT = CONDITIONS[CONDITION]["output"]


def parameters():
    configuration = yaml.safe_load((ROOT / "src/configs/task3/t3_main_model.yaml").read_text())
    params = base_model_params("no_pe")
    params.update(image_aux_weight=0.0, ct_attention_variant=VARIANT)
    params["label_query_consistency_weight"] = configuration["model"]["params"]["label_query_consistency_weight"]
    return {MODEL_KEY: params}


def build_model(key, params):
    if key != MODEL_KEY:
        raise ValueError(f"未知模型：{key}")
    params = dict(params)
    variant = params.pop("ct_attention_variant")
    model, auxiliary = amef.build_model(key, params)
    replace_slice_attention(model, variant)
    return model, auxiliary


def protocol():
    value = missing.stable_paths(STANDARD_PROTOCOL())
    source_files = [
        Path(__file__),
        ROOT / "src/exp_8/position_baselines.py",
        ROOT / "src/exp_8/models.py",
        ROOT / "src/model/gastro_label_graph_mil/modules.py",
        ROOT / "src/training/data.py",
        ROOT / "src/scripts/task3_apro_cope_ablation_scheduler.py",
        ROOT / "outputs/ct_rate_680/position_baselines/upstream_manifest.json",
    ]
    if CONDITION == "standard":
        source_files.extend([
            SOURCE / "samples.json",
            SOURCE / "patient_folds.json",
            SOURCE / "preparation_protocol.json",
        ])
        condition = {
            "dataset": "CT-RATE固定680例标准无删除协议",
            "position": "保持原680例实验输入：原始切片索引和原CT长度传入模型；替换模块自身仅使用其定义支持的信息",
            "missingness": "无预先随机删除；沿用原训练设置instance_dropout=0.25",
            "reference_results": {
                "apro_full": "outputs/ct_rate_680/all_models_fivefold/summary.json",
                "original_pe": "outputs/ct_rate_680/amef_original_pe_fivefold/summary.json",
            },
        }
    else:
        source_files.extend([
            SOURCE / "data/samples.json",
            SOURCE / "data/patient_folds.json",
            SOURCE / "data/preparation_protocol.json",
            SOURCE / "data/sampling.json",
            SOURCE / "data/feature_integrity.json",
        ])
        manifest = json.loads((SOURCE / "data/sampling.json").read_text())
        condition = {
            "dataset": f"CT-RATE完整CT删除{manifest['delete_fraction']:.0%}后的固定子集",
            "position": "与已有ACPE/Original PE缺失实验一致：原始索引、坐标、间距和原CT长度均不输入模型",
            "missingness": "共用已有固定缺失抽样及冻结特征；instance_dropout=0",
            "delete_fraction": manifest["delete_fraction"],
            "sampling_seed": manifest["seed"],
            "reference_results": str((SOURCE / "comparison.json").relative_to(ROOT)),
        }
    for path in source_files:
        value["source_sha256"][str(path.relative_to(ROOT))] = core.sha256(path)
    value.update(
        models={MODEL_KEY: VARIANTS[VARIANT]},
        model_parameters=parameters(),
        position_variant=VARIANT,
        runner="src/scripts/run_ctrate_position_replacements.py",
        adaptation=json.loads((ROOT / "outputs/ct_rate_680/position_baselines/upstream_manifest.json").read_text()),
        scope="四种位置机制是明确边界的CT切片适配，不宣称复现原论文完整架构或原数据集指标",
        prediction_coordinates="四种模块均无原生标量上下文坐标c_t，严格位置恢复指标不可计算",
        **condition,
    )
    value.pop("protocol_sha256", None)
    value["protocol_sha256"] = missing.json_digest(value)
    return value


def configure(condition, variant):
    global CONDITION, VARIANT, SOURCE, OUT
    CONDITION = condition
    VARIANT = variant
    SOURCE = CONDITIONS[condition]["source"]
    OUT = CONDITIONS[condition]["output"]
    core.MODELS = {MODEL_KEY: VARIANTS[variant]}
    core.IMAGE_MODELS, core.TEXT_MODELS, core.FUSION_MODELS = {}, {}, {}
    core.parameters = parameters
    core.protocol = protocol
    core.build_model = build_model
    if condition == "standard":
        core.INPUT = STANDARD_INPUT
        core.SETTINGS = dict(STANDARD_SETTINGS)
        core.read_inputs = STANDARD_READ_INPUTS
        core.load_features = STANDARD_LOAD_FEATURES
        core.forward_batch = STANDARD_FORWARD
    else:
        missing.OUT = SOURCE
        preparation = json.loads((SOURCE / "data/preparation_protocol.json").read_text())
        core.INPUT = SOURCE / "data"
        core.SETTINGS = {
            **STANDARD_SETTINGS,
            "max_instances": int(preparation["slices_per_case"]),
            "instance_dropout": 0.0,
        }
        core.read_inputs = missing.read_inputs
        core.load_features = missing.load_features
        core.forward_batch = missing.forward_hidden
    core.configure_adapters()


def arguments(condition, variant, fold=None):
    return argparse.Namespace(
        output_dir=CONDITIONS[condition]["output"] / variant,
        models=None,
        devices=list(range(5)),
        worker_index=None if fold is None else fold - 1,
    )


def audit(condition):
    reports = {}
    for variant in VARIANTS:
        configure(condition, variant)
        rows, folds, targets = core.read_inputs()
        bags = core.load_features(rows)
        shapes = sorted({tuple(np.asarray(bag).shape) for bag in bags})
        current = protocol()
        missing.ensure_json(OUT / variant / "protocol.json", current)
        reports[variant] = {
            "protocol_sha256": current["protocol_sha256"],
            "cases": len(rows),
            "fold_sizes": list(map(len, folds["folds"])),
            "feature_shapes": [list(shape) for shape in shapes],
            "targets_shape": list(targets.shape),
        }
    core.save_json(OUT / "audit.json", reports)
    print(json.dumps(reports, ensure_ascii=False, indent=2), flush=True)


def smoke(condition):
    from training.losses import AsymmetricLossMultiLabel

    torch.set_num_threads(2)
    torch.backends.mha.set_fastpath_enabled(False)
    reports = {}
    for variant in VARIANTS:
        configure(condition, variant)
        rows, _, targets = core.read_inputs()
        rows, targets = rows[:16], targets[:16]
        bags = core.load_features(rows)
        tokens, masks = core.fusion_base.encode_descriptions(
            {row["case_index"]: row["findings_masked"] for row in rows}, np.arange(len(rows))
        )
        batch = next(iter(amef.loader(bags, targets, np.arange(len(rows)), False, 42)))
        torch.manual_seed(42)
        model, weights = build_model(MODEL_KEY, parameters()[MODEL_KEY])
        model._ct_model_key = MODEL_KEY
        model = model.cuda()
        optimizer = torch.optim.AdamW(model.parameters(), lr=2e-4)
        optimizer.zero_grad(set_to_none=True)
        output = core.forward_batch(model, batch, tokens, masks, "cuda", training=True)
        loss = AsymmetricLossMultiLabel()(output["logits"], batch[2].cuda())
        for name, weight in weights.items():
            if weight:
                loss = loss + weight * output["aux_losses"][name]
        if not torch.isfinite(loss):
            raise ValueError(f"{condition}/{variant}损失异常")
        loss.backward()
        gradients = [parameter.grad for parameter in model.parameters() if parameter.grad is not None]
        if not gradients or not all(torch.isfinite(gradient).all() for gradient in gradients):
            raise ValueError(f"{condition}/{variant}梯度异常")
        if variant != "videorope" and not any(
            parameter.grad is not None and parameter.grad.abs().sum() > 0
            for name, parameter in model.named_parameters()
            if "position_module" in name
        ):
            raise ValueError(f"{condition}/{variant}位置参数未参与分类学习")
        optimizer.step()
        reports[variant] = {
            "finite_loss_and_gradients": True,
            "loss": float(loss.item()),
            "peak_reserved_mib": torch.cuda.max_memory_reserved() / 1024**2,
            "position_variant": getattr(model, "position_baseline", None),
            "apro_disabled": model.apro_positioner is None and model.position_variant == "no_pe",
        }
        del model, optimizer, output, loss
        gc.collect()
        torch.cuda.empty_cache()
    core.save_json(OUT / "smoke_audit.json", reports)
    print(json.dumps(reports, ensure_ascii=False, indent=2), flush=True)


def reference_summaries(condition):
    if condition == "standard":
        all_models = json.loads(CONDITIONS[condition]["reference"][0].read_text())
        apro = next(item for item in all_models if item["model_key"] == MODEL_KEY)
        original = json.loads(CONDITIONS[condition]["reference"][1].read_text())[0]
        return [
            {**apro, "position_variant": "apro_full", "result_source": "原标准680例五折"},
            {**original, "position_variant": "original_pe", "result_source": "原标准680例五折"},
        ]
    values = json.loads((CONDITIONS[condition]["source"] / "comparison.json").read_text())
    return [{**item, "result_source": "原固定缺失配对五折"} for item in values]


def aggregate(condition):
    summaries = reference_summaries(condition)
    for variant in VARIANTS:
        configure(condition, variant)
        args = arguments(condition, variant)
        if core.aggregate(args, protocol()["protocol_sha256"]) != 5:
            raise RuntimeError(f"{condition}/{variant}五折尚未完成")
        summary = json.loads((args.output_dir / "summary.json").read_text())[0]
        summaries.append({**summary, "position_variant": variant, "result_source": "本次替换模块五折"})
    order = {name: index for index, name in enumerate(("apro_full", "original_pe", *VARIANTS))}
    summaries.sort(key=lambda item: order[item["position_variant"]])
    out = CONDITIONS[condition]["output"]
    core.save_json(out / "comparison.json", summaries)
    fields = [
        "position_variant", "model", "macro_f1_mean", "macro_f1_std",
        "macro_f1_fixed_0_5_mean", "macro_f1_fixed_0_5_std", "oof_macro_f1", "result_source",
    ]
    with (out / "comparison.csv").open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(summaries)
    title = "标准无删除" if condition == "standard" else f"随机删除{condition.removeprefix('missing')}%"
    report = (
        f"# CT-RATE {title}位置模块分类对照\n\n"
        "主指标为五个测试折Macro-F1的均值±样本标准差（ddof=1）；阈值只由各折验证集确定。\n\n"
        "| 位置模块 | Macro-F1 | 固定0.5 Macro-F1 | OOF Macro-F1 | 来源 |\n"
        "|---|---:|---:|---:|---|\n"
    )
    for item in summaries:
        report += (
            f"| {item['model']} | {item['macro_f1_mean']:.4f} ± {item['macro_f1_std']:.4f} | "
            f"{item['macro_f1_fixed_0_5_mean']:.4f} ± {item['macro_f1_fixed_0_5_std']:.4f} | "
            f"{item['oof_macro_f1']:.4f} | {item['result_source']} |\n"
        )
    if condition == "standard":
        report += "\n本表沿用原标准680例协议：ACPE为原0.8134结果，训练期instance_dropout=0.25，且传入原始切片索引与原CT长度。\n"
    else:
        report += "\n六种方法共用已有固定删除样本、冻结视觉特征、掩码文本及原五折；隐藏原始位置并关闭额外instance dropout。\n"
    report += "四种替换模块没有原生标量上下文坐标c_t，因此严格位置恢复PRE/GRE/Acc不可计算，未用注意力或额外回归头伪造。\n"
    (out / "comparison.md").write_text(report, encoding="utf-8")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--condition", choices=CONDITIONS, required=True)
    parser.add_argument("--variant", choices=VARIANTS, default="comrope_ap")
    parser.add_argument("--fold", type=int, choices=range(1, 6))
    parser.add_argument("--audit-only", action="store_true")
    parser.add_argument("--smoke-test", action="store_true")
    parser.add_argument("--aggregate", action="store_true")
    args = parser.parse_args()
    configure(args.condition, args.variant)
    if args.audit_only:
        audit(args.condition)
    elif args.smoke_test:
        smoke(args.condition)
    elif args.aggregate:
        aggregate(args.condition)
    elif args.fold:
        core.worker(arguments(args.condition, args.variant, args.fold))
    else:
        parser.error("请选择审计、冒烟检查、汇总或指定fold")


if __name__ == "__main__":
    main()
