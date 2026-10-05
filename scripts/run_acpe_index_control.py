#!/usr/bin/env python3
"""四公开数据集的位置索引/变形四条件五折对照；结果独立保存。"""
from __future__ import annotations

import argparse
import csv
import fcntl
import hashlib
import json
import os
from pathlib import Path
import statistics
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "src/scripts"))

OUT = ROOT / "outputs/acpe_index_control_20261003"
MODEL = "amef_multimodal"
DATASETS = {"ct_rate": "CT-RATE", "mr_rate": "MR-RATE-1K",
            "amos_mm": "AMOS-MM", "aa_mini": "AA-Mini"}
VARIANTS = {"no_pe": "No PE", "uniform_pe": "Uniform PE (Original PE)",
            "acquisition_index_pe": "Acquisition-index PE (c=r, eta=0)", "acpe": "ACPE"}


def save_json(path, obj):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + f".{os.getpid()}.tmp")
    tmp.write_text(json.dumps(obj, ensure_ascii=False, indent=2) + "\n")
    tmp.replace(path)


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def folder(dataset, variant, fold=None):
    path = OUT / dataset / variant
    if fold is not None:
        if dataset == "amos_mm":
            path /= "7_labels"
        path = path / f"fold_{fold}" / MODEL
    return path


def result_path(dataset, variant, fold):
    return folder(dataset, variant, fold) / ("result.json" if dataset == "amos_mm" else "test_metrics.json")


def complete(dataset, variant, fold):
    path = folder(dataset, variant, fold)
    return (path / "verified.json").exists()


def parameters(variant):
    import yaml
    from task3_apro_cope_ablation_scheduler import base_model_params
    params = base_model_params("apro_full")
    params["label_query_consistency_weight"] = float(yaml.safe_load(
        (ROOT / "src/configs/task3/t3_main_model.yaml").read_text())["model"]["params"]["label_query_consistency_weight"])
    params["index_control_variant"] = variant
    return params


def builder(dataset, variant):
    def build(key, supplied, vocabulary_size=8192):
        import torch
        from exp_8.models import Exp12AProCoPEWatchCrossAttentionTextCNNModel
        from exp_8.acpe_index_control import install_control
        from amos_mm_model_adapters import PartialLabelAMEF
        from run_physionet_ct_ich_table2_baselines import CachedFeatureBackbone
        assert key == MODEL
        values = dict(parameters(variant) if dataset == "amos_mm" else supplied)
        assert values.pop("index_control_variant") == variant
        weights = {k.removesuffix("_weight"): values.pop(k)
                   for k in list(values) if k.endswith("_weight")}
        cls = PartialLabelAMEF if dataset == "amos_mm" else Exp12AProCoPEWatchCrossAttentionTextCNNModel
        model = cls(**values, num_labels={"ct_rate": 3, "mr_rate": 4, "amos_mm": 7, "aa_mini": 3}[dataset], pretrained=False)
        model.instance_encoder.backbone = CachedFeatureBackbone(768, values["feature_dim"], values["dropout"])
        install_control(model, variant)
        model._ct_model_key = key
        torch.backends.mha.set_fastpath_enabled(False)

        # 真实首批次核对：不允许新对照因漏传元数据而退化为均匀位置。
        checked = set()
        def audit(module, args, kwargs, output):
            phase = "train" if module.training else "eval"
            if phase in checked:
                return
            mask = kwargs["mask"]
            indices, counts = kwargs["instance_indices"], kwargs["original_image_counts"]
            assert indices is not None and counts is not None
            assert ((indices >= 0) & (indices < counts[:, None]))[mask].all()
            pairs = mask[:, 1:] & mask[:, :-1]
            assert (indices[:, 1:] - indices[:, :-1])[pairs].gt(0).all()
            payload = {"真实采集索引已传入": True, "有效索引严格递增": True,
                       "原序列长度范围": [int(counts.min()), int(counts.max())]}
            if variant in ("acquisition_index_pe", "acpe"):
                raw, context = output["apro_raw_coordinates"], output["apro_context_coordinates"]
                expected = indices.clamp_min(0).float() / (counts[:, None] - 1).clamp_min(1)
                torch.testing.assert_close(raw[mask], expected[mask])
                payload["上下文偏移最大绝对值"] = float((context - raw)[mask].abs().max())
                if variant == "acquisition_index_pe":
                    assert torch.equal(raw, context)
                    assert torch.count_nonzero(output["apro_transition_eta"]) == 0
                    payload.update(c_equals_r=True, eta_exactly_zero=True)
            if CURRENT_FOLDER is not None:
                save_json(CURRENT_FOLDER / f"input_audit_{phase}.json", payload)
            checked.add(phase)
        model.register_forward_hook(audit, with_kwargs=True)
        return model, weights
    return build


CURRENT_FOLDER = None


def configure(dataset, variant):
    build = builder(dataset, variant)
    if dataset == "ct_rate":
        import run_ctrate_680_all_models as core
        core.MODELS = {MODEL: VARIANTS[variant]}
        core.IMAGE_MODELS, core.TEXT_MODELS, core.FUSION_MODELS = {}, {}, {}
        core.parameters = lambda: {MODEL: parameters(variant)}
        core.build_model = build
        core.configure_adapters()
    elif dataset == "mr_rate":
        import run_mrrate_1k_amef as core
        core.MODELS = {MODEL: VARIANTS[variant]}
        core.model_parameters = lambda: parameters(variant)
        core.build_model = build
        core.configure_base()
    elif dataset == "amos_mm":
        import run_amos_mm_all_models as core
        core.OUT = folder(dataset, variant)
        core.MODELS = {MODEL: VARIANTS[variant]}
        core.IMAGE_MODELS, core.TEXT_MODELS = {}, {}
        core.parameters = lambda key: parameters(variant)
        core.build_model = build
    else:
        import run_abdomenatlas3_anonymous_table1 as anonymous
        core = anonymous.base
        core.INPUT = anonymous.INPUT
        core.OUT = folder(dataset, variant)
        core.MODELS = {MODEL: VARIANTS[variant]}
        core.IMAGE_MODELS, core.TEXT_MODELS, core.FUSION_MODELS = {}, {}, {}
        core.parameters = lambda: {MODEL: parameters(variant)}
        core.build_model = build
        core.configure()
    return core


def protocol(dataset, variant, core):
    source_files = ["src/scripts/run_acpe_index_control.py", "src/exp_8/acpe_index_control.py",
                    "src/exp_8/models.py", "src/exp_4/models.py", "src/training/losses.py",
                    "src/scripts/run_cq500_table2_multimodal_baselines.py",
                    "src/scripts/run_physionet_ct_ich_table2_baselines.py",
                    "src/scripts/task3_apro_cope_ablation_scheduler.py", str(Path(core.__file__).relative_to(ROOT))]
    input_files = [core.INPUT / "samples.json", core.INPUT / "preparation_protocol.json",
                   core.INPUT / ("splits.json" if dataset == "amos_mm" else "patient_folds.json")]
    if dataset == "amos_mm":
        input_files.append(core.INPUT / "image_exclusions.json")
        source_files.append("src/scripts/amos_mm_model_adapters.py")
    if dataset == "aa_mini":
        input_files.append(core.INPUT / "masking_protocol.json")
        source_files.append("src/scripts/run_abdomenatlas3_anonymous_table1.py")
    obj = {"dataset": dataset, "variant": variant, "model_parameters": parameters(variant),
           "settings": core.SETTINGS, "fold_seeds": [142, 242, 342, 442, 542],
           "selection": "沿用原训练器：最低验证分类损失选模，验证集调标签阈值，另存固定0.5结果",
           "split": "AMOS-MM为五个开发折和固定人工测试集" if dataset == "amos_mm" else "原患者五折：三折训练、一折验证、一折测试",
           "controls": {"no_pe": "保留视觉Transformer，不注入位置",
                        "uniform_pe": "历史Original PE：均匀槽位标量MLP绝对位置编码",
                        "acquisition_index_pe": "严格c=r、eta=0；保留完整ACPE的Fourier/绝对/相对注入；冻结变形网络",
                        "acpe": "原始六组视觉描述子含u_it加间距，双路径、无额外位置预热"},
           "comparison": "关键配对acquisition_index_pe与acpe只改变deformation；共有参数初始化及dropout随机数一致",
           "scope": "四条件独立重训，不从旧表格数值拼接结果；不修改论文或历史实验",
           "source_sha256": {f: sha(ROOT / f) for f in sorted(set(source_files))},
           "input_sha256": {str(f.relative_to(ROOT)): sha(f) for f in input_files}}
    obj["protocol_sha256"] = hashlib.sha256(json.dumps(obj, sort_keys=True, ensure_ascii=False).encode()).hexdigest()
    path = folder(dataset, variant) / "protocol.json"
    if path.exists():
        assert json.loads(path.read_text()) == obj, f"拒绝混合协议：{path}"
    else:
        save_json(path, obj)
    return obj


def train(dataset, variant, fold):
    global CURRENT_FOLDER
    import torch
    import numpy as np
    torch.set_num_threads(2)
    torch.backends.mha.set_fastpath_enabled(False)
    CURRENT_FOLDER = folder(dataset, variant, fold)
    CURRENT_FOLDER.mkdir(parents=True, exist_ok=True)
    with (CURRENT_FOLDER / "run.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        core = configure(dataset, variant)
        p = protocol(dataset, variant, core)
        digest = p["protocol_sha256"]
        if complete(dataset, variant, fold):
            assert json.loads((CURRENT_FOLDER / "verified.json").read_text())["protocol_sha256"] == digest
            return
        if not result_path(dataset, variant, fold).exists():
            if dataset == "amos_mm":
                rows = json.loads((core.INPUT / "samples.json").read_text())
                splits = json.loads((core.INPUT / "splits.json").read_text())
                bags = core.load_features(rows)
                core.train_job({"labels": 7, "model": MODEL, "fold": fold-1}, rows, splits, bags, digest, f"p0_{variant}_{fold}")
            else:
                rows, folds, y = core.read_inputs()
                if dataset == "aa_mini":
                    core.image_fold(MODEL, fold-1, rows, folds, y, digest)
                else:
                    bags = core.load_features(rows)
                    ids = np.arange(len(rows), dtype=np.int64)
                    tokens, masks = core.fusion_base.encode_descriptions({r["case_index"]: r["findings_masked"] for r in rows}, ids)
                    core.fusion_base.train_fold(argparse.Namespace(output_dir=folder(dataset, variant)), MODEL, fold-1,
                                               ids, bags, y, folds, tokens, masks, parameters(variant), digest)
        verify(dataset, variant, fold, p)


def verify(dataset, variant, fold, p):
    import numpy as np
    from sklearn.metrics import f1_score
    target = folder(dataset, variant, fold)
    result = json.loads(result_path(dataset, variant, fold).read_text())
    assert result["protocol_sha256"] == p["protocol_sha256"]
    assert (target / "best_model.pt").exists()
    for phase in ("train", "eval"):
        assert (target / f"input_audit_{phase}.json").exists()
    if dataset == "amos_mm":
        with np.load(target / "test_predictions.npz") as values:
            y, probs, thresholds = values["labels"], values["probabilities"], values["thresholds"]
    else:
        with (target / "test_predictions.csv").open(encoding="utf-8-sig") as stream:
            rows = list(csv.DictReader(stream))
        labels = list(result["thresholds"])
        y = np.asarray([[int(float(row["true_"+k])) for k in labels] for row in rows])
        probs = np.asarray([[float(row["prob_"+k]) for k in labels] for row in rows])
        thresholds = np.asarray([result["thresholds"][k] for k in labels])
    score = float(f1_score(y, probs >= thresholds, average="macro", zero_division=0))
    assert abs(score-result["macro_f1"]) < 1e-7
    save_json(target / "verified.json", {"dataset": dataset, "variant": variant, "fold": fold,
              "macro_f1": score, "macro_f1_fixed_0_5": float(f1_score(y, probs >= .5, average="macro", zero_division=0)),
              "protocol_sha256": p["protocol_sha256"], "prediction_recalculation": True,
              "checkpoint_sha256": sha(target / "best_model.pt")})


def summarize():
    OUT.mkdir(parents=True, exist_ok=True)
    with (OUT / "summary.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        records = {}
        for dataset in DATASETS:
            for variant in VARIANTS:
                vals = [json.loads((folder(dataset, variant, k) / "verified.json").read_text())
                        for k in range(1, 6) if complete(dataset, variant, k)]
                item = {"dataset": dataset, "variant": variant, "completed": len(vals), "folds": vals}
                if len(vals) == 5:
                    assert len({v["protocol_sha256"] for v in vals}) == 1
                    item.update(mean=statistics.mean(v["macro_f1"] for v in vals), std=statistics.stdev(v["macro_f1"] for v in vals))
                records[dataset, variant] = item
        lines = ["# ACPE：采集索引与上下文变形的四条件对照", "", "独立重训；只汇报完整五折。F1使用百分数，标准差为样本标准差。论文与历史表格不自动修改。", "",
                 "| 位置配置 | " + " | ".join(DATASETS.values()) + " |", "|---|" + "---:|"*4]
        for variant, name in VARIANTS.items():
            cells = []
            for dataset in DATASETS:
                item = records[dataset, variant]
                cells.append(f"{item['mean']*100:.2f} ± {item['std']*100:.2f}" if item["completed"] == 5 else f"待完成（{item['completed']}/5）")
            lines.append("| " + name + " | " + " | ".join(cells) + " |")
        lines += ["", "关键比较：Acquisition-index PE保留ACPE的双路径注入，只设置c=r、η=0；ACPE保留u_it。Uniform PE沿用Original PE的均匀槽位编码，其注入结构与双路径不同，因此Uniform→Acquisition的差异不能全部归因于坐标来源。", "", "所有条件在同一数据集内保持输入、划分、种子、训练设置一致，使用既有缓存和掩码文本；AMOS-MM沿用五开发折和固定200例人工测试集。", "", "## ACPE − Acquisition-index PE（百分点）", "", "| 数据集 | 五个配对折差值 | 平均差值 |", "|---|---|---:|"]
        for dataset, name in DATASETS.items():
            a, b = records[dataset, "acpe"], records[dataset, "acquisition_index_pe"]
            if a["completed"] == b["completed"] == 5:
                diffs = [100*(x["macro_f1"]-y["macro_f1"]) for x, y in zip(a["folds"], b["folds"])]
                lines.append(f"| {name} | " + ", ".join(f"{d:+.2f}" for d in diffs) + f" | {statistics.mean(diffs):+.2f} |")
        save_json(OUT / "summary.json", list(records.values()))
        (OUT / "results.md").write_text("\n".join(lines) + "\n")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", choices=DATASETS)
    parser.add_argument("--variant", choices=VARIANTS)
    parser.add_argument("--fold", type=int, choices=range(1, 6))
    parser.add_argument("--initialize", action="store_true")
    parser.add_argument("--summarize", action="store_true")
    args = parser.parse_args()
    if args.summarize:
        summarize()
    elif args.initialize:
        core = configure(args.dataset, args.variant)
        protocol(args.dataset, args.variant, core)
    elif args.fold:
        train(args.dataset, args.variant, args.fold)
        summarize()
    else:
        parser.error("请选择初始化、训练或汇总")


if __name__ == "__main__":
    main()
