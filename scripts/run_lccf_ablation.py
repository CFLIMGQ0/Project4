#!/usr/bin/env python3
"""LCCF单因素消融：复用三个公开数据集的原训练循环和患者划分。"""
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
OUTPUT = ROOT / "outputs/lccf_ablation"
DATASETS = {"ct_rate": "CT-RATE", "amos_mm": "AMOS-MM七标签", "mr_rate": "MR-RATE-1K"}
INPUTS = {"ct_rate": "ct_rate_680", "amos_mm": "amos_mm", "mr_rate": "mr_rate_1k"}
MODEL_KEY = "amef_multimodal"
VARIANTS = {
    "full": {},
    "a_mean": {"lccf_pooling": "mean"},
    "a_max": {"lccf_pooling": "max"},
    "a_shared": {"lccf_pooling": "shared_attention"},
    "b_none": {"lccf_reasoning": "none"},
    "b_graph": {"lccf_reasoning": "ordinary_graph"},
    "b_hyper1": {"lccf_hypergraph_edges": 1},
    "b_hyper3": {"lccf_hypergraph_edges": 3},
    "b_hyper4": {"lccf_hypergraph_edges": 4},
    "b_hyper5": {"lccf_hypergraph_edges": 5},
    "c_mean": {"lccf_text_retrieval": "shared_mean"},
    "c_shared": {"lccf_text_retrieval": "shared_query"},
    "c_no_identity": {"lccf_text_retrieval": "label_query_no_identity"},
    "d_add": {"lccf_fusion": "direct"},
    "d_fixed": {"lccf_fusion": "fixed"},
    "d_label": {"lccf_fusion": "label_constant"},
    "d_exam": {"lccf_fusion": "exam_shared"},
}
TABLE = [
    ("A", "Mean pooling", "a_mean"), ("A", "Max pooling", "a_max"),
    ("A", "Shared attention", "a_shared"), ("A", "Label-wise attention ★", "full"),
    ("B", "No reasoning", "b_none"), ("B", "Ordinary label graph", "b_graph"),
    ("B", "Hypergraph E=1", "b_hyper1"), ("B", "Hypergraph E=2 ★", "full"),
    ("B", "Hypergraph E=3", "b_hyper3"), ("B", "Hypergraph E=4", "b_hyper4"),
    ("B", "Hypergraph E=5", "b_hyper5"), ("C", "Shared mean pooling", "c_mean"),
    ("C", "Shared-query cross-attention", "c_shared"),
    ("C", "Label-wise query, w/o identity", "c_no_identity"),
    ("C", "Label-wise query + identity ★", "full"),
    ("D", "Direct addition", "d_add"), ("D", "Fixed gate=0.5", "d_fixed"),
    ("D", "Label-wise constant gate", "d_label"),
    ("D", "Examination-wise shared gate", "d_exam"),
    ("D", "Examination- and label-wise gate ★", "full"),
]


def save_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(f".{os.getpid()}.tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")
    temporary.replace(path)


def params_for(variant):
    import yaml
    from task3_apro_cope_ablation_scheduler import base_model_params
    params = base_model_params("apro_full")
    config = yaml.safe_load((ROOT / "src/configs/task3/t3_main_model.yaml").read_text())
    params["label_query_consistency_weight"] = config["model"]["params"]["label_query_consistency_weight"]
    params.update(lccf_pooling="label_wise_attention", lccf_reasoning="hypergraph",
                  lccf_hypergraph_edges=2, lccf_text_retrieval="label_query_identity", lccf_fusion="exam_label")
    params.update(VARIANTS[variant])
    return params


def build_model(variant, count):
    import torch
    from exp_8.models import Exp13LCCFAblationModel
    from run_physionet_ct_ich_table2_baselines import CachedFeatureBackbone
    params = params_for(variant)
    weights = {name.removesuffix("_weight"): params.pop(name) for name in list(params) if name.endswith("_weight")}
    model = Exp13LCCFAblationModel(**params, pretrained=False, num_labels=count)
    model.instance_encoder.backbone = CachedFeatureBackbone(768, params["feature_dim"], params["dropout"])
    torch.backends.mha.set_fastpath_enabled(False)
    return model, weights


def configure(dataset, variant):
    folder = OUTPUT / dataset / variant
    if dataset == "amos_mm":
        import run_amos_mm_all_models as core
        core.OUT = folder
        core.MODELS = {MODEL_KEY: variant}
        core.parameters = lambda key: params_for(variant)
        core.build_model = lambda key, count, vocabulary_size=8192: build_model(variant, count)
    elif dataset == "ct_rate":
        import run_ctrate_680_all_models as core
        core.MODELS = {MODEL_KEY: variant}
        core.build_model = lambda key, params: build_model(variant, 3)
        core.configure_adapters()
    else:
        import run_mrrate_1k_amef as core
        core.MODELS = {MODEL_KEY: variant}
        core.build_model = lambda key, params: build_model(variant, 4)
        core.configure_base()
    return core


def protocol(dataset, variant):
    core = configure(dataset, variant)
    inputs = ROOT / "outputs" / INPUTS[dataset] / "experiment"
    paths = [Path(__file__), Path(core.__file__), ROOT / "src/exp_8/models.py",
             ROOT / "src/exp_4/models.py", ROOT / "src/model/common/pooling.py",
             ROOT / "src/model/gastro_label_graph_mil/modules.py", ROOT / "src/training/losses.py",
             ROOT / "src/scripts/run_cq500_table2_multimodal_baselines.py",
             ROOT / "src/scripts/run_cq500_amef_multimodal.py",
             ROOT / "src/scripts/amos_mm_model_adapters.py",
             ROOT / "src/configs/task3/t3_main_model.yaml",
             inputs / "samples.json", inputs / "preparation_protocol.json",
             inputs / ("splits.json" if dataset == "amos_mm" else "patient_folds.json")]
    if dataset == "amos_mm":
        paths.append(inputs / "image_exclusions.json")
    value = {
        "dataset": dataset, "variant": variant, "model_parameters": params_for(variant),
        "settings": core.SETTINGS, "labels": core.LABELS[:7],
        "source_sha256": {str(path.relative_to(ROOT)): hashlib.sha256(path.read_bytes()).hexdigest() for path in paths},
        "scope": "只改一个LCCF因素；ACPE双支、视觉缓存、TextCNN、分类器与原训练循环保持不变",
        "split": "AMOS七标签沿用五个开发折与固定200例人工测试；CT/MR沿用患者五折，测试k、验证(k+1)%5",
        "seed": "沿用原五折配置base_seed=42，训练种子142/242/342/442/542；同折所有变体相同",
        "selection": "最低验证分类损失选checkpoint；只在验证集选阈值；不按测试指标选择配置",
        "lqd": "沿用现有label_query_consistency及权重0.01；仅已知阳性/缺失标签配对；现有代码重算反事实gate，不是原稿的固定gate",
        "std": "五折Macro-F1的样本标准差ddof=1；AMOS为固定测试集上五次训练，不是OOF",
        "shared_control": "四个星号共用full五折；每数据集17种唯一配置、85个训练任务",
    }
    value["protocol_sha256"] = hashlib.sha256(json.dumps(value, ensure_ascii=False, sort_keys=True).encode()).hexdigest()
    return value


def initialize(dataset, variant):
    value = protocol(dataset, variant)
    path = OUTPUT / dataset / variant / "protocol.json"
    if path.exists():
        if json.loads(path.read_text()) != value:
            folder = path.parent
            completed = list(folder.rglob("result.json")) + list(folder.rglob("test_metrics.json"))
            if completed:
                raise ValueError(f"{path}协议变化，拒绝混合结果")
            save_json(path, value)
    else:
        save_json(path, value)
    return value["protocol_sha256"]


def result_path(dataset, variant, fold):
    folder = OUTPUT / dataset / variant
    if dataset == "amos_mm":
        return folder / "7_labels" / f"fold_{fold}" / MODEL_KEY / "result.json"
    return folder / f"fold_{fold}" / MODEL_KEY / "test_metrics.json"


def run_fold(dataset, variant, fold):
    import numpy as np
    import torch
    torch.set_num_threads(1)
    torch.backends.mha.set_fastpath_enabled(False)
    digest = initialize(dataset, variant)
    core = configure(dataset, variant)
    target = result_path(dataset, variant, fold)
    target.parent.mkdir(parents=True, exist_ok=True)
    with (target.parent / "lccf.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        if target.exists():
            assert json.loads(target.read_text())["protocol_sha256"] == digest
            return
        if dataset == "amos_mm":
            rows = json.loads((core.INPUT / "samples.json").read_text())
            splits = json.loads((core.INPUT / "splits.json").read_text())
            bags = core.load_features(rows)
            core.train_job({"labels": 7, "model": MODEL_KEY, "fold": fold - 1}, rows, splits,
                           bags, digest, f"{variant}_{fold}")
        else:
            rows, folds, targets = core.read_inputs()
            bags = core.load_features(rows)
            indices = np.arange(len(rows), dtype=np.int64)
            texts = {row["case_index"]: row["findings_masked"] for row in rows}
            tokens, masks = core.fusion_base.encode_descriptions(texts, indices)
            args = argparse.Namespace(output_dir=OUTPUT / dataset / variant)
            core.fusion_base.train_fold(args, MODEL_KEY, fold - 1, indices, bags, targets,
                                        folds, tokens, masks, params_for(variant), digest)
        save_json(target.parent / "gpu_peak.json", {
            "allocated_mib": torch.cuda.max_memory_allocated() / 1024**2,
            "reserved_mib": torch.cuda.max_memory_reserved() / 1024**2,
            "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
        })


def aggregate(datasets):
    for dataset in datasets:
        rows = []
        for part, label, variant in TABLE:
            metrics = []
            path = OUTPUT / dataset / variant / "protocol.json"
            if not path.exists():
                continue
            digest = json.loads(path.read_text())["protocol_sha256"]
            for fold in range(1, 6):
                path = result_path(dataset, variant, fold)
                if path.exists():
                    value = json.loads(path.read_text())
                    assert value["protocol_sha256"] == digest
                    metrics.append(value["macro_f1"])
            rows.append({"part": part, "configuration": label, "variant": variant,
                         "completed_folds": len(metrics),
                         "macro_f1_mean": statistics.mean(metrics) if len(metrics) == 5 else None,
                         "macro_f1_std": statistics.stdev(metrics) if len(metrics) == 5 else None})
        if not rows:
            continue
        folder = OUTPUT / dataset
        with (folder / "summary.csv").open("w", encoding="utf-8-sig", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
        lines = [f"# {DATASETS[dataset]} LCCF消融", "", "每行仅变动一个因素；四个星号共用同一full结果。",
                 "均值±样本标准差来自五折（ddof=1）；AMOS为固定200例测试集上的五次训练。未完成五折不填正式均值。",
                 "", "| 部分 | 配置 | 完成折数 | Macro-F1 ↑ |", "|---|---|---:|---:|"]
        for row in rows:
            metric = (f"{row['macro_f1_mean']:.4f} ± {row['macro_f1_std']:.4f}"
                      if row["completed_folds"] == 5 else "待完成")
            lines.append(f"| {row['part']} | {row['configuration']} | {row['completed_folds']}/5 | {metric} |")
        (folder / "results.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def smoke(device):
    import torch
    from tqdm import tqdm
    from amos_mm_model_adapters import masked_asl
    torch.set_num_threads(1)
    torch.manual_seed(42)
    records = []
    for variant in tqdm(VARIANTS, desc="LCCF前后向自检"):
        model, weights = build_model(variant, 7)
        model.to(device).train()
        images = torch.randn(16, 64, 768, 1, 1, device=device)
        mask = torch.ones(16, 64, dtype=torch.bool, device=device)
        mask[1, 32:] = False
        tokens = torch.randint(1, 8192, (16, 512), device=device)
        text_mask = torch.ones_like(tokens, dtype=torch.bool)
        text_mask[-1] = False
        targets = torch.randint(0, 2, (16, 7), device=device).float()
        known = torch.ones_like(targets, dtype=torch.bool)
        known[:, -1] = False
        positions = torch.arange(64, device=device).expand(16, -1)
        counts = torch.full((16,), 64, device=device)
        if device.startswith("cuda"):
            torch.cuda.reset_peak_memory_stats()
        optimizer = torch.optim.AdamW(model.parameters(), lr=2e-4)
        for step in range(2):
            optimizer.zero_grad(set_to_none=True)
            output = model(images, mask, (targets, known), tokens, text_mask, positions, counts)
            assert output["logits"].shape == (16, 7)
            assert output["watch_text_gate"].shape == (16, 7)
            assert torch.equal(output["logits"][-1], output["image_only_logits"][-1])
            loss = masked_asl(output["logits"], targets, known)
            for name, weight in weights.items():
                loss = loss + weight * output["aux_losses"][name]
            loss.backward()
            assert torch.isfinite(loss) and all(torch.isfinite(parameter.grad).all() for parameter in model.parameters() if parameter.grad is not None)
            optimizer.step()
        model.eval()
        with torch.no_grad():
            output = model(images, mask, None, tokens, text_mask, positions, counts)
            assert torch.isfinite(output["logits"]).all()
        records.append({"variant": variant, "finite_gradient": True, "missing_text_visual_fallback": True,
                        "peak_reserved_mib": torch.cuda.max_memory_reserved() / 1024**2 if device.startswith("cuda") else None})
        del model, output, loss, optimizer
        if device.startswith("cuda"):
            torch.cuda.empty_cache()
    save_json(OUTPUT / "smoke.json", records)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--datasets", nargs="+", choices=DATASETS, default=["ct_rate", "amos_mm"])
    parser.add_argument("--variant", choices=VARIANTS)
    parser.add_argument("--fold", type=int, choices=range(1, 6))
    parser.add_argument("--audit", action="store_true")
    parser.add_argument("--aggregate", action="store_true")
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    if args.audit:
        for dataset in args.datasets:
            for variant in VARIANTS:
                initialize(dataset, variant)
        aggregate(args.datasets)
        print(f"协议核对通过：{args.datasets}，每数据集85任务")
    elif args.aggregate:
        aggregate(args.datasets)
    elif args.smoke:
        smoke(args.device)
    elif args.variant and args.fold and len(args.datasets) == 1:
        run_fold(args.datasets[0], args.variant, args.fold)
    else:
        parser.error("指定--audit、--aggregate、--smoke，或单数据集+variant+fold")


if __name__ == "__main__":
    main()
