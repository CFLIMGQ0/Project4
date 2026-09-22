#!/usr/bin/env python3
"""在CQ500和PhysioNet CT-ICH上运行AMEF位置编码替换五折实验。"""
from __future__ import annotations

import argparse
import csv
import gc
import hashlib
import json
from pathlib import Path
import statistics

import numpy as np
import torch
import yaml

import run_cq500_amef_multimodal as cq_amef
import run_cq500_table2_multimodal_baselines as cq_core
import run_physionet_ct_ich_table2_baselines as ich_core
from exp_8.position_baselines import replace_slice_attention
from scripts.task3_apro_cope_ablation_scheduler import base_model_params
from training.losses import AsymmetricLossMultiLabel


ROOT = Path(__file__).resolve().parents[2]
MODEL_KEY = "amef_multimodal"
ICH_MODEL_KEY = "amef_image_branch"
VARIANTS = {
    "original_pe": "Original PE",
    "comrope_ap": "ComRoPE-AP（CT-1D适配）",
    "videorope": "VideoRoPE（CT-1×1适配）",
    "path": "PaTH（CT双向适配）",
    "dape_v2_kerple": "DAPE V2-Kerple（CT双向1×3适配）",
}
OUTPUTS = {
    "cq500": ROOT / "outputs/cq500/position_baselines",
    "physionet_ct_ich": ROOT / "outputs/physionet_ct_ich/position_baselines",
}

CQ_ORIGINAL_BUILD = cq_amef.build_model
CQ_ORIGINAL_PROTOCOL = cq_amef.protocol
DATASET = "cq500"
VARIANT = "original_pe"


def save_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def stable_paths(value):
    prefix = str(ROOT.resolve()) + "/"
    if isinstance(value, dict):
        return {
            key.removeprefix(prefix) if isinstance(key, str) else key: stable_paths(item)
            for key, item in value.items()
        }
    if isinstance(value, list):
        return [stable_paths(item) for item in value]
    if isinstance(value, str):
        return value.removeprefix(prefix)
    return value


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False).encode()).hexdigest()


def ensure_json(path, value):
    if path.exists() and json.loads(path.read_text()) != value:
        raise ValueError(f"已有协议不一致，禁止混用：{path}")
    save_json(path, value)


def cq_arguments(variant, fold=None):
    return argparse.Namespace(
        text_csv=ROOT / "outputs/cq500/image_descriptions/uniform64/descriptions_draft.csv",
        reads=ROOT / "datasets/cq500/raw/reads.csv",
        folds_json=ROOT / "outputs/cq500/table2_image_baselines/patient_folds.json",
        feature_cache=ROOT / "outputs/cq500/convnext_tiny_scan_features.npz",
        config=ROOT / "src/configs/task3/t3_main_model.yaml",
        output_dir=OUTPUTS["cq500"] / variant,
        devices=list(range(5)),
        worker_index=None if fold is None else fold - 1,
    )


def cq_parameters(args):
    configuration = yaml.safe_load(args.config.read_text())
    position_variant = VARIANT if VARIANT == "original_pe" else "no_pe"
    parameters = base_model_params(position_variant)
    parameters["label_query_consistency_weight"] = float(
        configuration["model"]["params"]["label_query_consistency_weight"]
    )
    parameters["image_aux_weight"] = 0.0
    parameters["ct_attention_variant"] = VARIANT
    return {MODEL_KEY: parameters}


def cq_build_model(key, parameters):
    values = dict(parameters)
    variant = values.pop("ct_attention_variant")
    model, auxiliary = CQ_ORIGINAL_BUILD(key, values)
    if variant != "original_pe":
        replace_slice_attention(model, variant)
    return model, auxiliary


def cq_protocol(args):
    value = stable_paths(CQ_ORIGINAL_PROTOCOL(args))
    tracked = [
        Path(__file__),
        ROOT / "src/exp_8/position_baselines.py",
        ROOT / "src/exp_8/models.py",
        ROOT / "src/model/gastro_label_graph_mil/modules.py",
    ]
    value["source_sha256"].update({str(path.relative_to(ROOT)): sha256(path) for path in tracked})
    value.update(
        models={MODEL_KEY: VARIANTS[VARIANT]},
        model_parameters=cq_parameters(args),
        dataset="CQ500固定491例多模态五折",
        position_variant=VARIANT,
        position_protocol=(
            "沿用AMEF-MIL缓存轴位序列索引与长度；Original PE使用槽位编码，四种替换模块使用各自CT适配定义"
        ),
        scope="只替换AMEF切片上下文编码中的位置机制；视觉缓存、AI影像描述、划分、损失和训练设置不变",
        runner="src/scripts/run_public_ct_position_replacements.py",
    )
    value.pop("protocol_sha256", None)
    value["protocol_sha256"] = digest(value)
    return value


def configure_cq(variant):
    global DATASET, VARIANT
    DATASET = "cq500"
    VARIANT = variant
    cq_amef.MODELS = {MODEL_KEY: VARIANTS[variant]}
    cq_amef.model_parameters = cq_parameters
    cq_amef.build_model = cq_build_model
    cq_amef.protocol = cq_protocol
    cq_amef.configure_base()


def ich_arguments(variant):
    return argparse.Namespace(
        data_root=ROOT / "datasets/physionet_ct_ich/computed-tomography-images-for-intracranial-hemorrhage-detection-and-segmentation-1.0.0",
        feature_cache=ROOT / "outputs/physionet_ct_ich/mean_pool_baseline/convnext_tiny_slice_features.npz",
        output_dir=OUTPUTS["physionet_ct_ich"] / variant,
        device="cuda:0",
        folds=5,
        epochs=30,
        batch_size=8,
        lr=2e-4,
        weight_decay=0.02,
        warmup_ratio=0.2,
        max_instances=64,
        instance_dropout=0.25,
        seed=42,
    )


def ich_build_model(model_key):
    if model_key != ICH_MODEL_KEY:
        raise ValueError(f"未知模型：{model_key}")
    model = ich_core.AMEFImageOnlyBranch()
    model.core.apro_positioner = None
    if VARIANT == "original_pe":
        model.core.position_variant = "original_pe"
    else:
        model.core.position_variant = "no_pe"
        replace_slice_attention(model.core, VARIANT)
    return model


def ich_protocol(args):
    tracked = [
        Path(__file__),
        Path(ich_core.__file__),
        ROOT / "src/exp_8/position_baselines.py",
        ROOT / "src/exp_8/models.py",
        ROOT / "src/model/gastro_label_graph_mil/modules.py",
        args.data_root / "hemorrhage_diagnosis.csv",
        args.feature_cache,
        ROOT / "outputs/physionet_ct_ich/table2_image_baselines/patient_folds.json",
        ROOT / "outputs/physionet_ct_ich/table2_image_baselines/amef_image_branch.json",
    ]
    value = {
        "dataset": "PhysioNet CT-ICH固定82例患者五折",
        "labels": list(ich_core.LABEL_NAMES),
        "models": {ICH_MODEL_KEY: VARIANTS[VARIANT]},
        "position_variant": VARIANT,
        "settings": {
            "folds": args.folds,
            "epochs": args.epochs,
            "batch_size": args.batch_size,
            "learning_rate": args.lr,
            "weight_decay": args.weight_decay,
            "warmup_ratio": args.warmup_ratio,
            "max_instances": args.max_instances,
            "instance_dropout": args.instance_dropout,
            "seed": args.seed,
        },
        "split": "严格复用原AMEF图像分支确定性患者五折；测试k、验证k+1、其余三折训练",
        "image_input": "复用冻结ImageNet ConvNeXt-Tiny切片特征；无文本分支",
        "position_protocol": "原实验未传入DICOM绝对坐标；各方法只使用有序缓存序列及自身定义支持的信息",
        "selection": "最低验证分类损失选择checkpoint；固定0.5阈值测试",
        "scope": "只替换AMEF图像分支切片上下文编码中的位置机制",
        "runner": "src/scripts/run_public_ct_position_replacements.py",
        "source_sha256": {str(path.relative_to(ROOT)): sha256(path) for path in tracked},
    }
    value["protocol_sha256"] = digest(value)
    return value


def configure_ich(variant):
    global DATASET, VARIANT
    DATASET = "physionet_ct_ich"
    VARIANT = variant
    ich_core.MODEL_SPECS[ICH_MODEL_KEY] = {
        "display_name": VARIANTS[variant],
        "registry": "ours",
        "model_name": "exp12_apro_cope_watch_cross_attn_textcnn",
        "params": {},
    }
    ich_core.build_model = ich_build_model


def audit_cq():
    reports = {}
    for variant in VARIANTS:
        configure_cq(variant)
        args = cq_arguments(variant)
        texts, labels, folds = cq_core.load_inputs(args)
        patient_ids, bags, targets = __import__("run_cq500_table2_baselines").load_bags(
            args.reads, args.feature_cache
        )
        if patient_ids.tolist() != sorted(texts):
            raise ValueError("CQ500文本和图像患者不一致")
        if not np.array_equal(targets, [labels[int(patient)] for patient in patient_ids]):
            raise ValueError("CQ500标签不一致")
        current = cq_protocol(args)
        ensure_json(args.output_dir / "protocol.json", current)
        reports[variant] = {
            "protocol_sha256": current["protocol_sha256"],
            "cases": len(patient_ids),
            "fold_sizes": [len(group) for group in folds["folds"]],
            "bag_length_range": [min(map(len, bags)), max(map(len, bags))],
        }
    save_json(OUTPUTS["cq500"] / "audit.json", reports)
    print(json.dumps(reports, ensure_ascii=False, indent=2), flush=True)


def ich_inputs(args):
    patient_ids, bags, targets = ich_core.load_bags(args.data_root, args.feature_cache)
    folds = ich_core.multilabel_folds(targets, args.folds, args.seed)
    saved = json.loads((ROOT / "outputs/physionet_ct_ich/table2_image_baselines/patient_folds.json").read_text())
    actual = [patient_ids[group].tolist() for group in folds]
    if actual != saved["folds"]:
        raise ValueError("PhysioNet CT-ICH患者五折与原实验不一致")
    return patient_ids, bags, targets, folds


def audit_ich():
    reports = {}
    for variant in VARIANTS:
        configure_ich(variant)
        args = ich_arguments(variant)
        patient_ids, bags, targets, folds = ich_inputs(args)
        current = ich_protocol(args)
        ensure_json(args.output_dir / "protocol.json", current)
        reports[variant] = {
            "protocol_sha256": current["protocol_sha256"],
            "cases": len(patient_ids),
            "fold_sizes": list(map(len, folds)),
            "bag_length_range": [min(map(len, bags)), max(map(len, bags))],
            "targets_shape": list(targets.shape),
        }
    save_json(OUTPUTS["physionet_ct_ich"] / "audit.json", reports)
    print(json.dumps(reports, ensure_ascii=False, indent=2), flush=True)


def smoke_cq():
    reports = {}
    for variant in VARIANTS:
        configure_cq(variant)
        args = cq_arguments(variant)
        texts, labels, _ = cq_core.load_inputs(args)
        patient_ids, bags, targets = __import__("run_cq500_table2_baselines").load_bags(
            args.reads, args.feature_cache
        )
        text_ids, text_mask = cq_core.encode_descriptions(texts, patient_ids)
        batch = next(iter(cq_amef.loader(bags, targets, np.arange(16), False, 42)))
        model, weights = cq_build_model(MODEL_KEY, cq_parameters(args)[MODEL_KEY])
        model = model.cuda()
        output = cq_amef.forward_batch(model, batch, text_ids, text_mask, "cuda", training=True)
        loss = AsymmetricLossMultiLabel()(output["logits"], batch[2].cuda())
        for name, weight in weights.items():
            if weight:
                loss = loss + weight * output["aux_losses"][name]
        loss.backward()
        validate_smoke(model, loss, variant)
        reports[variant] = {"finite_loss_and_gradients": True, "loss": float(loss.item())}
        del model, output, loss
        gc.collect()
        torch.cuda.empty_cache()
    save_json(OUTPUTS["cq500"] / "smoke_audit.json", reports)
    print(json.dumps(reports, ensure_ascii=False, indent=2), flush=True)


def smoke_ich():
    reports = {}
    for variant in VARIANTS:
        configure_ich(variant)
        args = ich_arguments(variant)
        _, bags, targets, _ = ich_inputs(args)
        dataset = ich_core.FeatureBagDataset(bags, targets, np.arange(16), 64, 0.0, False)
        loader = torch.utils.data.DataLoader(dataset, batch_size=8, collate_fn=ich_core.collate_bags)
        features, mask, batch_targets, _ = next(iter(loader))
        model = ich_build_model(ICH_MODEL_KEY).cuda()
        output = model(features.cuda(), mask.cuda())
        loss = AsymmetricLossMultiLabel()(output["logits"], batch_targets.cuda())
        loss.backward()
        validate_smoke(model.core, loss, variant)
        reports[variant] = {"finite_loss_and_gradients": True, "loss": float(loss.item())}
        del model, output, loss
        gc.collect()
        torch.cuda.empty_cache()
    save_json(OUTPUTS["physionet_ct_ich"] / "smoke_audit.json", reports)
    print(json.dumps(reports, ensure_ascii=False, indent=2), flush=True)


def validate_smoke(model, loss, variant):
    if not torch.isfinite(loss):
        raise ValueError(f"{DATASET}/{variant}损失异常")
    gradients = [parameter.grad for parameter in model.parameters() if parameter.grad is not None]
    if not gradients or not all(torch.isfinite(gradient).all() for gradient in gradients):
        raise ValueError(f"{DATASET}/{variant}梯度异常")
    if variant not in ("original_pe", "videorope") and not any(
        parameter.grad is not None and parameter.grad.abs().sum() > 0
        for name, parameter in model.named_parameters()
        if "position_module" in name
    ):
        raise ValueError(f"{DATASET}/{variant}位置参数未参与训练")


def worker_cq(variant, fold):
    configure_cq(variant)
    cq_core.worker(cq_arguments(variant, fold))


def worker_ich(variant, fold):
    configure_ich(variant)
    args = ich_arguments(variant)
    current = ich_protocol(args)
    saved = json.loads((args.output_dir / "protocol.json").read_text())
    if saved["protocol_sha256"] != current["protocol_sha256"]:
        raise ValueError("PhysioNet协议哈希不一致")
    patient_ids, bags, targets, folds = ich_inputs(args)
    test_index = fold - 1
    validation_index = fold % 5
    test_indices = folds[test_index]
    validation_indices = folds[validation_index]
    train_indices = np.concatenate([
        group for index, group in enumerate(folds) if index not in (test_index, validation_index)
    ])
    metrics, ordered_indices, probabilities = ich_core.train_one_fold(
        ICH_MODEL_KEY,
        fold,
        bags,
        targets,
        train_indices,
        validation_indices,
        test_indices,
        args,
    )
    metrics.update(
        model=VARIANTS[variant],
        model_key=ICH_MODEL_KEY,
        position_variant=variant,
        protocol_sha256=current["protocol_sha256"],
    )
    folder = args.output_dir / f"fold_{fold}" / ICH_MODEL_KEY
    folder.mkdir(parents=True, exist_ok=True)
    with (folder / "test_predictions.csv").open("w", encoding="utf-8-sig", newline="") as stream:
        fields = ["patient_id"]
        for label in ich_core.LABEL_NAMES:
            fields.extend([f"true_{label}", f"prob_{label}", f"pred_{label}"])
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for index, probability in zip(ordered_indices, probabilities):
            row = {"patient_id": int(patient_ids[index])}
            for label_index, label in enumerate(ich_core.LABEL_NAMES):
                row.update({
                    f"true_{label}": int(targets[index, label_index]),
                    f"prob_{label}": float(probability[label_index]),
                    f"pred_{label}": int(probability[label_index] >= 0.5),
                })
            writer.writerow(row)
    save_json(folder / "test_metrics.json", metrics)
    save_json(folder / "completed.json", {
        "protocol_sha256": current["protocol_sha256"],
        "fold": fold,
        "model": ICH_MODEL_KEY,
    })


def aggregate_cq():
    summaries = [
        {
            **json.loads((ROOT / "outputs/cq500/amef_multimodal_uniform64/summary.json").read_text())[0],
            "position_variant": "apro_full",
            "result_source": "既有AMEF-MIL五折",
        }
    ]
    for variant in VARIANTS:
        configure_cq(variant)
        args = cq_arguments(variant)
        current = cq_protocol(args)
        if cq_core.aggregate(args, current["protocol_sha256"]) != 5:
            raise RuntimeError(f"CQ500/{variant}五折尚未完成")
        summaries.append({
            **json.loads((args.output_dir / "summary.json").read_text())[0],
            "position_variant": variant,
            "result_source": "本次位置替换五折",
        })
    write_comparison("cq500", summaries, tuned=True)


def aggregate_ich_variant(variant):
    configure_ich(variant)
    args = ich_arguments(variant)
    current = ich_protocol(args)
    metrics = []
    predictions = []
    for fold in range(1, 6):
        folder = args.output_dir / f"fold_{fold}" / ICH_MODEL_KEY
        completed = json.loads((folder / "completed.json").read_text())
        if completed["protocol_sha256"] != current["protocol_sha256"]:
            raise ValueError(f"PhysioNet/{variant}/fold{fold}协议不一致")
        metrics.append(json.loads((folder / "test_metrics.json").read_text()))
        with (folder / "test_predictions.csv").open(encoding="utf-8-sig", newline="") as stream:
            predictions.extend(csv.DictReader(stream))
    predictions.sort(key=lambda row: int(row["patient_id"]))
    patient_ids, _, targets, _ = ich_inputs(args)
    if [int(row["patient_id"]) for row in predictions] != patient_ids.tolist():
        raise ValueError(f"PhysioNet/{variant} OOF患者不完整")
    probabilities = np.asarray([
        [float(row[f"prob_{label}"]) for label in ich_core.LABEL_NAMES] for row in predictions
    ])
    if not np.isfinite(probabilities).all():
        raise ValueError(f"PhysioNet/{variant}概率异常")
    result = {
        "model_key": ICH_MODEL_KEY,
        "model": VARIANTS[variant],
        "position_variant": variant,
        "completed_folds": 5,
        "num_patients": len(patient_ids),
        "macro_f1_mean": statistics.mean(item["macro_f1"] for item in metrics),
        "macro_f1_std": statistics.stdev(item["macro_f1"] for item in metrics),
        "micro_f1_mean": statistics.mean(item["micro_f1"] for item in metrics),
        "oof": ich_core.metric_dict(targets, probabilities),
        "folds": metrics,
        "result_source": "本次位置替换五折",
    }
    save_json(args.output_dir / "summary.json", [result])
    return result


def aggregate_ich():
    reference = next(
        item for item in json.loads(
            (ROOT / "outputs/physionet_ct_ich/table2_image_baselines/summary.json").read_text()
        )
        if item["model_key"] == ICH_MODEL_KEY
    )
    summaries = [{
        **reference,
        "macro_f1_mean": reference["fold_macro_f1_mean"],
        "macro_f1_std": reference["fold_macro_f1_std"],
        "position_variant": "apro_full",
        "result_source": "既有AMEF图像分支五折",
    }]
    summaries.extend(aggregate_ich_variant(variant) for variant in VARIANTS)
    write_comparison("physionet_ct_ich", summaries, tuned=False)


def write_comparison(dataset, summaries, tuned):
    output = OUTPUTS[dataset]
    save_json(output / "comparison.json", summaries)
    fields = ["position_variant", "model", "macro_f1_mean", "macro_f1_std", "oof_macro_f1", "result_source"]
    with (output / "comparison.csv").open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        for item in summaries:
            row = dict(item)
            row["oof_macro_f1"] = item.get("oof_macro_f1", item.get("oof", {}).get("macro_f1"))
            writer.writerow(row)
    title = "CQ500多模态" if dataset == "cq500" else "PhysioNet CT-ICH纯图像"
    threshold = "验证集逐标签调阈值" if tuned else "固定0.5阈值"
    report = (
        f"# {title}位置编码替换五折结果\n\n"
        f"主指标为{threshold}的测试Macro-F1；均值±五个测试折样本标准差（ddof=1）。\n\n"
        "| 位置模块 | Macro-F1 | OOF Macro-F1 |\n|---|---:|---:|\n"
    )
    for item in summaries:
        oof = item.get("oof_macro_f1", item.get("oof", {}).get("macro_f1"))
        report += f"| {item['model']} | {item['macro_f1_mean']:.4f} ± {item['macro_f1_std']:.4f} | {oof:.4f} |\n"
    report += "\n四种新方法为CT切片适配版本，只替换位置机制；不宣称复现原论文完整架构或原数据集指标。\n"
    (output / "comparison.md").write_text(report, encoding="utf-8")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", choices=OUTPUTS, required=True)
    parser.add_argument("--variant", choices=VARIANTS, default="original_pe")
    parser.add_argument("--fold", type=int, choices=range(1, 6))
    parser.add_argument("--audit-only", action="store_true")
    parser.add_argument("--smoke-test", action="store_true")
    parser.add_argument("--aggregate", action="store_true")
    args = parser.parse_args()
    torch.set_num_threads(2)
    torch.backends.mha.set_fastpath_enabled(False)
    if args.audit_only:
        audit_cq() if args.dataset == "cq500" else audit_ich()
    elif args.smoke_test:
        smoke_cq() if args.dataset == "cq500" else smoke_ich()
    elif args.aggregate:
        aggregate_cq() if args.dataset == "cq500" else aggregate_ich()
    elif args.fold:
        if args.dataset == "cq500":
            worker_cq(args.variant, args.fold)
        else:
            worker_ich(args.variant, args.fold)
    else:
        parser.error("请选择审计、冒烟检查、汇总或指定fold")


if __name__ == "__main__":
    main()
