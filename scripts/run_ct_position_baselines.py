#!/usr/bin/env python3
"""CT-RATE原五折分类：四个位置机制适配与同协议ACPE、Original PE对照。"""
from __future__ import annotations

import argparse
import csv
import gc
import json
from pathlib import Path

import run_ctrate_680_all_models as core
import run_cq500_amef_multimodal as amef
from run_ctrate_slice_missing import forward_hidden, stable_paths, json_digest, ensure_json
from scripts.task3_apro_cope_ablation_scheduler import base_model_params
from exp_8.position_baselines import replace_slice_attention
import numpy as np
import torch
import yaml

ROOT = core.ROOT
OUT = ROOT / "outputs/ct_rate_680/position_baselines"
ORIGINAL_PROTOCOL = core.protocol
VARIANTS = {"apro_full": "ACPE（等距输入）", "original_pe": "Original PE",
            "comrope_ap": "ComRoPE-AP（CT-1D适配）", "videorope": "VideoRoPE（CT-1×1适配）",
            "path": "PaTH（CT双向适配）", "dape_v2_kerple": "DAPE V2-Kerple（CT双向1×3适配）"}
VARIANT = "apro_full"
MODEL_KEY = "amef_multimodal"


def parameters():
    configuration = yaml.safe_load((ROOT / "src/configs/task3/t3_main_model.yaml").read_text())
    params = base_model_params(VARIANT if VARIANT in ("apro_full", "original_pe") else "no_pe")
    params.update(image_aux_weight=0.0, ct_attention_variant=VARIANT)
    params["label_query_consistency_weight"] = configuration["model"]["params"]["label_query_consistency_weight"]
    return {MODEL_KEY: params}


def build_model(key, params):
    params = dict(params)
    variant = params.pop("ct_attention_variant")
    model, auxiliary = amef.build_model(key, params)
    if variant not in ("apro_full", "original_pe"):
        replace_slice_attention(model, variant)
    return model, auxiliary


def protocol():
    result = stable_paths(ORIGINAL_PROTOCOL())
    for relative in ("src/scripts/run_ct_position_baselines.py", "src/exp_8/position_baselines.py",
                     "src/scripts/run_ctrate_slice_missing.py", "src/scripts/audit_ct_position_baselines.py",
                     "src/exp_4/models.py", "src/model/gastro_label_graph_mil/modules.py",
                     "src/training/data.py", "src/scripts/task3_apro_cope_ablation_scheduler.py",
                     "outputs/ct_rate_680/position_baselines/upstream_manifest.json"):
        result["source_sha256"][relative] = core.sha256(ROOT / relative)
    result.update(position_variant=VARIANT, runner="src/scripts/run_ct_position_baselines.py",
                  position="全部方法只输入当前槽位；原始索引、间距、原CT长度全部屏蔽；无额外切片丢弃",
                  image="复用原680例每例64张冻结三窗ConvNeXt特征；原患者五折；训练验证测试不改变抽样",
                  scope="CT模块适配，不宣称复现原论文完整架构或其数据集指标；同协议重新训练ACPE/Original PE对照",
                  prediction_coordinates="仅ACPE原生支持标量上下文坐标；四个新模块无此接口，位置恢复指标N/A",
                  adaptation=json.loads((OUT / "upstream_manifest.json").read_text()),
                  missingness="本组分类未随机删除原CT切片，instance_dropout=0；与50%/75%子集实验分开")
    result.pop("protocol_sha256", None)
    result["protocol_sha256"] = json_digest(result)
    return result


def configure(variant):
    global VARIANT
    VARIANT = variant
    preparation = json.loads((core.INPUT / "preparation_protocol.json").read_text())
    core.MODELS = {MODEL_KEY: VARIANTS[variant]}
    core.IMAGE_MODELS, core.TEXT_MODELS, core.FUSION_MODELS = {}, {}, {}
    core.SETTINGS = {**core.SETTINGS, "max_instances": int(preparation["slices_per_case"]), "instance_dropout": 0.0}
    core.parameters, core.protocol = parameters, protocol
    core.build_model, core.forward_batch = build_model, forward_hidden
    core.configure_adapters()


def arguments(variant, fold=None):
    return argparse.Namespace(output_dir=OUT / variant, models=None, devices=list(range(5)),
                              worker_index=None if fold is None else fold - 1)


def smoke():
    from training.losses import AsymmetricLossMultiLabel
    torch.set_num_threads(2)
    rows, _, targets = core.read_inputs()
    rows, targets = rows[:16], targets[:16]
    bags = core.load_features(rows)
    tokens, masks = core.fusion_base.encode_descriptions(
        {row["case_index"]: row["findings_masked"] for row in rows}, np.arange(len(rows)))
    batch = next(iter(amef.loader(bags, targets, np.arange(16), False, 42)))
    report = {}
    for variant in VARIANTS:
        configure(variant)
        torch.manual_seed(42)
        model, weights = build_model(MODEL_KEY, parameters()[MODEL_KEY])
        model = model.cuda()
        criterion = AsymmetricLossMultiLabel()
        optimizer = torch.optim.AdamW(model.parameters(), lr=2e-4)
        torch.cuda.reset_peak_memory_stats()
        for step in range(2):
            model.train()
            optimizer.zero_grad(set_to_none=True)
            output = forward_hidden(model, batch, tokens, masks, "cuda", training=True)
            loss = criterion(output["logits"], batch[2].cuda())
            for name, weight in weights.items():
                if weight:
                    loss = loss + weight * output["aux_losses"][name]
            if not torch.isfinite(loss):
                raise ValueError(f"{variant}训练损失异常")
            loss.backward()
            if not all(torch.isfinite(parameter.grad).all() for parameter in model.parameters() if parameter.grad is not None):
                raise ValueError(f"{variant}梯度异常")
            if variant not in ("apro_full", "original_pe", "videorope"):
                if not any(parameter.grad is not None and parameter.grad.abs().sum() > 0
                           for name, parameter in model.named_parameters() if "position_module" in name):
                    raise ValueError("位置参数未参与分类学习")
            optimizer.step()
        model.eval()
        with torch.no_grad():
            expected = forward_hidden(model, batch, tokens, masks, "cuda")
            poisoned = (*batch[:4], torch.full_like(batch[4], 987654), torch.full_like(batch[5], 1000000))
            actual = forward_hidden(model, poisoned, tokens, masks, "cuda")
            torch.testing.assert_close(actual["logits"], expected["logits"], atol=0, rtol=0)
            if variant not in ("apro_full", "original_pe"):
                assert model.apro_positioner is None
                assert "apro_context_coordinates" not in actual
            if variant == "apro_full":
                coordinates = actual["apro_raw_coordinates"]
                torch.testing.assert_close(coordinates, torch.linspace(0, 1, coordinates.shape[1], device="cuda").expand_as(coordinates))
        report[variant] = {"finite_loss_gradients": True, "metadata_poisoning_no_effect": True,
                           "peak_reserved_mib": torch.cuda.max_memory_reserved() / 1024**2,
                           "smoke_steps": 2, "loss": loss.item()}
        del model, optimizer, output, actual, expected, loss
        gc.collect()
        torch.cuda.empty_cache()
    core.save_json(OUT / "smoke_audit.json", report)
    print(json.dumps(report, ensure_ascii=False, indent=2))


def aggregate():
    summaries = []
    for variant in VARIANTS:
        configure(variant)
        args = arguments(variant)
        if core.aggregate(args, protocol()["protocol_sha256"]) != 5:
            raise RuntimeError(f"{variant}五折尚未完成")
        summaries.append({"position_variant": variant, **json.loads((args.output_dir / "summary.json").read_text())[0]})
    core.save_json(OUT / "comparison.json", summaries)
    with (OUT / "comparison.csv").open("w", encoding="utf-8-sig", newline="") as stream:
        fields = ["position_variant", "macro_f1_mean", "macro_f1_std", "macro_f1_fixed_0_5_mean", "macro_f1_fixed_0_5_std", "oof_macro_f1"]
        writer = csv.DictWriter(stream, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(summaries)
    text = "# CT-RATE位置模块分类对照\n\n原680例、原五折、等距输入、关闭额外切片丢弃；均值±五个测试折样本标准差（ddof=1）。\n\n"
    text += "| 模型 | Macro-F1（验证集调阈值） | Macro-F1（固定0.5） | OOF Macro-F1 |\n|---|---:|---:|---:|\n"
    for record in summaries:
        text += (f"| {record['model']} | {record['macro_f1_mean']:.4f} ± {record['macro_f1_std']:.4f} | "
                 f"{record['macro_f1_fixed_0_5_mean']:.4f} ± {record['macro_f1_fixed_0_5_std']:.4f} | {record['oof_macro_f1']:.4f} |\n")
    text += "\n四个新模块是有明确边界的CT切片适配，不是原论文完整架构复现；详见upstream_manifest.json。VideoRoPE的1×1空间网格退化为等间隔RoPE。禁止与不同位置输入或丢弃设置的旧实验直接归因比较。\n"
    (OUT / "comparison.md").write_text(text)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--variant", choices=VARIANTS, default="apro_full")
    parser.add_argument("--fold", type=int, choices=range(1, 6))
    parser.add_argument("--audit-only", action="store_true")
    parser.add_argument("--smoke-test", action="store_true")
    parser.add_argument("--aggregate", action="store_true")
    args = parser.parse_args()
    configure(args.variant)
    if args.smoke_test:
        smoke()
    elif args.audit_only:
        for variant in VARIANTS:
            configure(variant)
            ensure_json(OUT / variant / "protocol.json", protocol())
        print("六种模型的独立协议已固定。")
    elif args.aggregate:
        aggregate()
    elif args.fold:
        core.worker(arguments(args.variant, args.fold))
    else:
        parser.error("请选择审计、冒烟检查、汇总或指定测试折")


if __name__ == "__main__":
    main()
