#!/usr/bin/env python3
"""CT-RATE原始ACPE的六种过渡输入组合五折消融，保留所有旧结果。"""
from __future__ import annotations

import argparse
import csv
import fcntl
import hashlib
import json
from pathlib import Path

import yaml

import run_ctrate_680_all_models as core

ROOT = core.ROOT
OUTPUT = ROOT / "outputs/ct_rate_680/acpe_transition_groups_fivefold"
MODEL_KEY = "amef_multimodal"
VARIANTS = {
    "1": ([1], False),
    "1+1": ([1], True),
    "123+1": ([1, 2, 3], True),
    "145+1": ([1, 4, 5], True),
    "1236+1": ([1, 2, 3, 6], True),
    "1456+1": ([1, 4, 5, 6], True),
}
BASE_PROTOCOL = core.protocol
VARIANT = "1"
BASELINE = ROOT / "outputs/ct_rate_680/amef_original_acpe_rerun_fivefold_v2"


def parameters():
    from scripts.task3_apro_cope_ablation_scheduler import base_model_params

    groups, include_gap = VARIANTS[VARIANT]
    main = yaml.safe_load((ROOT / "src/configs/task3/t3_main_model.yaml").read_text())
    model = base_model_params("apro_full")
    model.update(
        apro_transition_dim=64,
        apro_fourier_mlp_layers=2,
        apro_transition_groups=groups,
        apro_transition_include_gap=include_gap,
        label_query_consistency_weight=main["model"]["params"]["label_query_consistency_weight"],
        image_aux_weight=0.0,
    )
    return {MODEL_KEY: model}


def stable_paths(item):
    prefix = str(ROOT.resolve()) + "/"
    if isinstance(item, dict):
        return {key.removeprefix(prefix): stable_paths(value) for key, value in item.items()}
    if isinstance(item, list):
        return [stable_paths(value) for value in item]
    return item.removeprefix(prefix) if isinstance(item, str) else item


def protocol():
    value = stable_paths(BASE_PROTOCOL())
    groups, include_gap = VARIANTS[VARIANT]
    value["transition_ablation"] = {
        "variant": VARIANT,
        "groups": groups,
        "include_gap_in_mlp": include_gap,
        "input_dim": len(groups) * 64 + int(include_gap),
        "hidden_dim": 64,
        "output_dim": 1,
        "selection": "仅拼接被选组，不用零填充占位；不改变各组原来的计算定义",
        "gap_scope": "去掉+1只去掉MLP输入中的gap；采集锚定、gap*exp(eta)和长度守恒仍保留",
        "other_modules": "原始双支ACPE，Fourier双层投影，alpha=1.5，position_dim=64",
    }
    value["seed_rule"] = "沿用原始CT五折：base_seed=42，训练seed=42+100*fold，即142/242/342/442/542"
    for path in (Path(__file__).resolve(), ROOT / "src/scripts/task3_apro_cope_ablation_scheduler.py"):
        value["source_sha256"][str(path.relative_to(ROOT))] = core.sha256(path)
    value["runner"] = str(Path(__file__).resolve().relative_to(ROOT))
    value.pop("protocol_sha256", None)
    value["protocol_sha256"] = hashlib.sha256(
        json.dumps(value, sort_keys=True, ensure_ascii=False).encode()
    ).hexdigest()
    return value


def configure(variant):
    global VARIANT
    VARIANT = variant
    core.MODELS = {MODEL_KEY: f"ACPE（过渡输入{variant}）"}
    core.IMAGE_MODELS = {}
    core.TEXT_MODELS = {}
    core.FUSION_MODELS = {}
    core.parameters = parameters
    core.protocol = protocol
    core.configure_adapters()


def arguments(output_root, variant, fold=None):
    return argparse.Namespace(
        output_dir=output_root / variant, models=[MODEL_KEY], devices=list(range(5)),
        worker_index=None if fold is None else fold - 1,
    )


def audit(output_root, variant):
    configure(variant)
    core.read_inputs()
    folder = output_root / variant
    folder.mkdir(parents=True, exist_ok=True)
    current = protocol()
    with (folder / "run.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        saved = folder / "protocol.json"
        if saved.exists() and json.loads(saved.read_text()) != current:
            raise ValueError(f"{variant}协议改变，拒绝混用原有结果")
        core.save_json(saved, current)
    model = current["model_parameters"][MODEL_KEY].copy()
    for name in ("apro_transition_dim", "apro_fourier_mlp_layers", "apro_transition_groups", "apro_transition_include_gap"):
        model.pop(name)
    baseline = json.loads((BASELINE / "fold_1" / MODEL_KEY / "config.json").read_text())
    if model != baseline["model"] or core.SETTINGS != baseline["settings"]:
        raise ValueError("消融以外配置与原始ACPE不一致")
    print(f"{variant} 协议通过；输入维度{current['transition_ablation']['input_dim']}；{current['protocol_sha256']}", flush=True)


def aggregate(output_root):
    rows = []
    baseline = next(row for row in json.loads((BASELINE / "summary.json").read_text())
                    if row["model_key"] == MODEL_KEY)
    for variant in VARIANTS:
        configure(variant)
        args = arguments(output_root, variant)
        digest = json.loads((args.output_dir / "protocol.json").read_text())["protocol_sha256"]
        if digest != protocol()["protocol_sha256"]:
            raise ValueError(f"{variant}汇总时源码或配置变化")
        if core.aggregate(args, digest) != 5:
            raise RuntimeError(f"{variant}尚未完成五折，不能生成完整对比表")
        summary = json.loads((args.output_dir / "summary.json").read_text())[0]
        for fold in range(1, 6):
            folder = args.output_dir / f"fold_{fold}" / MODEL_KEY
            reference = BASELINE / f"fold_{fold}" / MODEL_KEY
            if json.loads((folder / "split_ids.json").read_text()) != json.loads((reference / "split_ids.json").read_text()):
                raise ValueError(f"{variant}折{fold}病例划分不匹配")
            config = json.loads((folder / "config.json").read_text())
            if config["seed"] != 42 + 100 * fold or config["model"] != parameters()[MODEL_KEY]:
                raise ValueError(f"{variant}折{fold}种子或模型参数不匹配")
        groups, include_gap = VARIANTS[variant]
        rows.append({
            "variant": variant, "input_dim": len(groups) * 64 + int(include_gap),
            "completed_folds": 5,
            **{name: summary[name] for name in (
                "macro_f1_mean", "macro_f1_std", "macro_f1_fixed_0_5_mean",
                "macro_f1_fixed_0_5_std", "oof_macro_f1", "oof_micro_f1",
            )},
            "delta_macro_f1": summary["macro_f1_mean"] - baseline["macro_f1_mean"],
        })
    core.save_json(output_root / "comparison.json", {"variants": rows, "baseline": baseline,
                   "baseline_source": str(BASELINE.relative_to(ROOT)), "std_ddof": 1})
    with (output_root / "comparison.csv").open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    report = [
        "# CT-RATE ACPE前后分析输入组合消融", "",
        "固定680例患者五折，每种配置408训练/136验证/136测试，共6×5=30个训练任务；不重划分、不按测试结果筛选配置。",
        "只有transition MLP输入拼接和相应输入层尺寸变化；projector为512→64，MLP隐藏层64、输出eta为1；Fourier为原双层，绝对与相对路径均保留。",
        "种子沿用原始对照：base_seed=42，各折为142/242/342/442/542；主阈值仅在验证集选择。均值±样本标准差来自五个测试折，ddof=1。", "",
        "| 输入组合 | 拼接维度 | 五折Macro-F1 | 固定0.5阈值Macro-F1 | 相对原始F1差值 |",
        "|---|---:|---:|---:|---:|",
        f"| 原始123456+1（已有重跑对照） | 385 | {baseline['macro_f1_mean']:.4f} ± {baseline['macro_f1_std']:.4f} | {baseline['macro_f1_fixed_0_5_mean']:.4f} ± {baseline['macro_f1_fixed_0_5_std']:.4f} | — |",
    ]
    for row in rows:
        report.append(f"| {row['variant']} | {row['input_dim']} | {row['macro_f1_mean']:.4f} ± {row['macro_f1_std']:.4f} | {row['macro_f1_fixed_0_5_mean']:.4f} ± {row['macro_f1_fixed_0_5_std']:.4f} | {row['delta_macro_f1']:+.4f} |")
    report.extend([
        "", "①当前特征；②前向差分；③后向差分；④②的绝对值；⑤③的绝对值；⑥②与③的逐元素乘积。",
        "`+1`仅表示是否把gap输入transition MLP。无`+1`不移除采集锚定、gap乘权、长度守恒或其它路径的位置输入，不是no_pe。",
        "本实验为原始分类口径（传入原采样索引及总层数），不是隐藏位置的恢复实验；更高测试均值不等于已经证明显著性或可据测试集调参。",
        "", "运行：`python src/scripts/run_ctrate_acpe_transition_groups_pool.py`。",
    ])
    (output_root / "results.md").write_text("\n".join(report) + "\n", encoding="utf-8")
    print("\n".join(report), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", type=Path, default=OUTPUT)
    parser.add_argument("--variant", choices=VARIANTS)
    parser.add_argument("--fold", type=int, choices=range(1, 6))
    parser.add_argument("--audit-all", action="store_true")
    parser.add_argument("--aggregate-all", action="store_true")
    args = parser.parse_args()
    args.output_root = args.output_root.resolve()
    if args.audit_all:
        for variant in VARIANTS:
            audit(args.output_root, variant)
    elif args.aggregate_all:
        aggregate(args.output_root)
    elif args.variant is not None and args.fold is not None:
        configure(args.variant)
        core.worker(arguments(args.output_root, args.variant, args.fold))
    else:
        parser.error("请指定--audit-all、--aggregate-all，或同时指定--variant和--fold")


if __name__ == "__main__":
    main()
