#!/usr/bin/env python3
"""AbdomenAtlas 3.0 Mini 表1：固定22个模型、三标签、互斥五折。"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
from pathlib import Path
import random
import statistics
import sys
import time
from types import SimpleNamespace

import numpy as np
import torch
import yaml
from tqdm import tqdm

from prepare_abdomenatlas3_table1 import ROOT, OUT as FEATURE_INPUT, LABELS, save_json, sha256
sys.path.insert(0, str(ROOT / "src"))
import run_merlin_1k_all_models as core
from run_paper2026_baselines import common_params, adjacency
from sotas.task2.paper2026_adapted import REGISTRY, build_paper2026_adapted

INPUT = ROOT / "outputs/abdomenatlas3_mini/experiment_answer_mask_v2"
OUT = ROOT / "outputs/abdomenatlas3_mini/table1_fivefold_original_uit_answer_mask_v2"
IMAGE_KEYS = ("attention_mil", "transmil", "dsmil", "dtfd_mil", "clam_mb", "clam_sb")
IMAGE_MODELS = {key: core.image_base.MODEL_SPECS[key]["display_name"] for key in IMAGE_KEYS}
IMAGE_MODELS["gcf_net_2026"] = "GCF-Net"
TEXT_MODELS = dict(core.TEXT_MODELS)
FUSION_MODELS = {"task2_mmfnet_2024": "MMFNet", "task2_radfuse_2025": "RadFuse",
                 "task2_saif_2025": "SAIF", "task2_mmtf_2025": "MMTF",
                 "task2_camchex_adapted": "CaMCheX", "task2_med3dvlm_adapted": "Med3DVLM",
                 "task2_m3fm_adapted": "M3FM", "unified_multimodal_framework_2026": "Unified MM",
                 "adaptive_multimodal_fusion_2026": "Adaptive Fusion"}
MODELS = {**IMAGE_MODELS, **TEXT_MODELS, **FUSION_MODELS, "amef_multimodal": "ALM-MIL"}
SETTINGS = {**core.SETTINGS, "max_instances": 64, "text_max_length": 512}
assert len(MODELS) == 22


def read_inputs():
    rows = json.loads((INPUT / "samples.json").read_text())
    folds = json.loads((INPUT / "patient_folds.json").read_text())
    assert [r["case_index"] for r in rows] == list(range(len(rows)))
    assert len({r["exam_id"] for r in rows}) == len(rows)
    assert sorted(i for g in folds["folds"] for i in g) == list(range(len(rows)))
    assert folds["labels"] == LABELS
    y = np.asarray([r["labels"] for r in rows], dtype=np.int64)
    for group, count in zip(folds["folds"], folds["fold_positive_counts"]):
        assert y[group].sum(0).tolist() == count
    return rows, folds, y


def parameters():
    from scripts.task3_apro_cope_ablation_scheduler import base_model_params
    config = yaml.safe_load((ROOT / "src/configs/task2/model.yaml").read_text())
    values = {k: config["models"][k] for k in FUSION_MODELS if k not in REGISTRY}
    values.update({k: common_params(k, None) for k in REGISTRY})
    values.update({k: core.image_base.MODEL_SPECS[k] for k in IMAGE_KEYS})
    values["amef_multimodal"] = base_model_params("apro_full")
    main = yaml.safe_load((ROOT / "src/configs/task3/t3_main_model.yaml").read_text())
    values["amef_multimodal"]["label_query_consistency_weight"] = main["model"]["params"]["label_query_consistency_weight"]
    assert values["amef_multimodal"]["image_aux_weight"] == 0
    return values


def text_config(fold):
    cfg = yaml.safe_load((ROOT / "src/configs/task2/exp10_text_classification.yaml").read_text())
    cfg["seed"] = SETTINGS["base_seed"] + 100 * (fold+1)
    cfg["data"].update(label_names=LABELS, text_field="findings_masked", max_length=512)
    cfg["training"]["num_workers"] = 0
    cfg["experiment_name"] = "abdomenatlas3_table1"
    cfg["paths"] = {"data_json": str(INPUT / "samples.json")}
    return cfg


def protocol():
    sources = {Path(__file__), Path(core.__file__), Path(core.image_base.__file__),
               Path(core.fusion_base.__file__), Path(core.amef_base.__file__),
               ROOT / "src/scripts/task3_apro_cope_ablation_scheduler.py",
               ROOT / "src/scripts/run_paper2026_baselines.py",
               ROOT / "src/scripts/prepare_abdomenatlas3_table1.py",
               ROOT / "src/scripts/mask_abdomenatlas3_answers.py"}
    for directory in ("exp_4", "exp_8", "exp_10", "model", "baselines/task1", "sotas/task1", "sotas/task2", "training"):
        sources.update((ROOT / "src" / directory).rglob("*.py"))
    sources.update(INPUT / name for name in ("samples.json", "patient_folds.json", "preparation_protocol.json", "masking_protocol.json"))
    value = {"dataset": "AbdomenAtlas 3.0 Mini", "labels": LABELS, "models": MODELS,
             "settings": SETTINGS, "text_config": text_config(0), "model_parameters": parameters(),
             "source_sha256": {str(p.relative_to(ROOT)): sha256(p) for p in sorted(sources)},
             "split": "所有模型共用BDMAP检查ID五折，3训练/1验证/1测试；无跨检查患者ID，不能额外声称患者互斥",
             "model": "ALM-MIL原始完整ACPE：保留u_it、前后转变、幅值、交互项和间距；无r051置零或位置预热",
             "position": "原完整CT的轴位索引及原层数，六组64维加一个间距；绝对和相对双路径",
             "image": "沿用此前公开数据集表1：冻结ImageNet ConvNeXt-Tiny 768维，共用64层三窗特征；训练投影和下游模型，非端到端微调",
             "report": "仅官方合成narrative report所见部分；答案掩码v2遮蔽诊断词、同义词和明确有无病灶短语；不输入结论、实例计数或其他标签字段",
             "report_limit": "报告与病灶标签共享标注来源，结果属于合成报告辅助分类，不作为独立临床诊断证据",
             "selection": "沿用已有公开数据集协议：图像/多模态验证ASL最小；文本验证F1最大、8轮早停",
             "threshold": "仅验证集调逐标签阈值；同时保存固定0.5结果；测试集只用于最终评分",
             "gcf_graph": "各折仅用训练标签计算共现邻接矩阵",
             "baselines": "复用当前项目的任务适配实现，不宣称是原论文官方完整模型",
             "completion": "22个模型×5折=110任务；只汇总完整五折，保留所有逐例预测与checkpoint"}
    value["protocol_sha256"] = hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False).encode()).hexdigest()
    return value


def initialize():
    read_inputs()
    value = protocol()
    path = OUT / "protocol.json"
    if path.exists():
        assert json.loads(path.read_text()) == value, "训练协议已变化，禁止混合结果"
    save_json(path, value)
    save_json(OUT / "jobs.json", [{"model": k, "fold": fold} for fold in range(1, 6) for k in MODELS])
    return value


def build_model(key, params):
    if key in REGISTRY:
        values = dict(params)
        weights = {n.removesuffix("_weight"): values.pop(n) for n in list(values) if n.endswith("_weight")}
        model = build_paper2026_adapted(key, **values, pretrained=False, num_labels=3)
        model.instance_encoder.backbone = core.image_base.CachedFeatureBackbone(768, values["feature_dim"], values["dropout"])
    elif key in IMAGE_KEYS:
        model, weights = core.image_base.build_model(key), params.get("aux_weights", {})
    elif key == "amef_multimodal":
        model, weights = core.amef_base.build_model(key, params)
        acpe = [m for m in model.modules() if hasattr(m, "transition_groups")]
        assert len(acpe) == 1 and tuple(acpe[0].transition_groups) == (1, 2, 3, 4, 5, 6)
        assert acpe[0].transition_include_gap and acpe[0].transition_mlp[1].in_features == 385
    else:
        model, weights = core.ORIGINAL_FUSION_BUILDER(key, params)
    model._ct_model_key = key
    return model, weights


def configure():
    core.INPUT, core.LABELS = INPUT, LABELS
    core.IMAGE_MODELS, core.TEXT_MODELS, core.FUSION_MODELS = IMAGE_MODELS, TEXT_MODELS, FUSION_MODELS
    core.MODELS, core.SETTINGS = MODELS, SETTINGS
    core.configure_adapters()
    core.fusion_base.build_model = build_model


def text_fold(key, fold, rows, folds, digest):
    from exp_10.data import TextRecord, build_train_vocabulary
    from exp_10.train_text_classification import train_one_model
    from sklearn.metrics import f1_score
    folder = OUT / f"fold_{fold+1}" / key
    selected = {"train": [i for j, g in enumerate(folds["folds"]) if j not in (fold, (fold+1)%5) for i in g],
                "val": folds["folds"][(fold+1)%5], "test": folds["folds"][fold]}
    records = [TextRecord(str(r["case_index"]), r["exam_id"], r["findings_masked"],
                         np.asarray(r["labels"]), tuple(r["mask_hits"])) for r in rows]
    splits = {k: [records[i] for i in indices] for k, indices in selected.items()}
    cfg = text_config(fold)
    vocab = build_train_vocabulary(splits["train"], cfg["data"]["vocab_size"], cfg["data"]["min_token_frequency"])
    save_json(folder / "config.json", cfg)
    save_json(folder / "vocabulary.json", vocab)
    save_json(folder / "split_ids.json", selected)
    begin = time.time()
    metrics = train_one_model(key, cfg, splits, vocab, folder.parent)
    with (folder / "test_predictions.csv").open(encoding="utf-8-sig") as stream:
        pred = list(csv.DictReader(stream))
    assert [int(r["patient_id"]) for r in pred] == selected["test"]
    y = np.asarray([[int(r[f"true_{l}"]) for l in LABELS] for r in pred])
    prob = np.asarray([[float(r[f"prob_{l}"]) for l in LABELS] for r in pred])
    assert np.isfinite(prob).all() and ((prob >= 0) & (prob <= 1)).all()
    metrics.update(model=MODELS[key], model_key=key, fold=fold+1, protocol_sha256=digest,
                   text_source="AbdomenAtlas3.0Mini narrative findings masked",
                   macro_f1_fixed_0_5=float(f1_score(y, prob >= .5, average="macro", zero_division=0)),
                   wall_seconds=time.time()-begin)
    save_json(folder / "test_metrics.json", metrics)
    save_json(folder / "completed.json", {"protocol_sha256": digest, "model": key, "fold": fold+1})


def atomic_torch(path, value):
    tmp = path.with_suffix(".tmp.pt")
    torch.save(value, tmp)
    tmp.replace(path)


def image_fold(key, fold, rows, folds, y, digest):
    from training.losses import AsymmetricLossMultiLabel
    from exp_10.train_text_classification import tune_thresholds, calculate_metrics
    base = core.fusion_base
    folder = OUT / f"fold_{fold+1}" / key
    seed = SETTINGS["base_seed"] + 100 * (fold+1)
    core.image_base.seed_everything(seed)
    split = {"train": [i for j, g in enumerate(folds["folds"]) if j not in (fold, (fold+1)%5) for i in g],
             "val": folds["folds"][(fold+1)%5], "test": folds["folds"][fold]}
    assert set(split["train"]).isdisjoint(split["val"] + split["test"])
    assert set(split["val"]).isdisjoint(split["test"])
    params = parameters()[key]
    if key == "gcf_net_2026":
        params["label_adjacency"] = adjacency(y, split["train"])
    save_json(folder / "split_ids.json", split)
    save_json(folder / "config.json", {"settings": SETTINGS, "model": params, "seed": seed, "protocol_sha256": digest})
    # 文本规则更新不影响CT特征；按原图像协议核验并复用原缓存。
    prep = json.loads((INPUT / "preparation_protocol.json").read_text())
    original = json.loads((FEATURE_INPUT / "preparation_protocol.json").read_text())
    assert prep["feature_preparation_sha256"] == original["preparation_sha256"]
    core.INPUT = FEATURE_INPUT
    try:
        bags = core.load_features(rows)
    finally:
        core.INPUT = INPUT
    ids = np.arange(len(rows))
    token_ids, token_mask = base.encode_descriptions({r["case_index"]: r["findings_masked"] for r in rows}, ids)
    loaders = {name: base.loader(bags, y, indices, name == "train", seed) for name, indices in split.items()}
    device = torch.device("cuda:0")
    model, auxiliary = build_model(key, params)
    model.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=SETTINGS["learning_rate"], weight_decay=SETTINGS["weight_decay"])
    total = len(loaders["train"]) * SETTINGS["epochs"]
    warmup = int(total * SETTINGS["warmup_ratio"])
    def factor(step):
        if step < warmup:
            return (step+1)/max(1, warmup)
        return .5 * (1+math.cos(math.pi*(step-warmup)/max(1, total-warmup)))
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, factor)
    scaler = torch.amp.GradScaler("cuda")
    criterion = AsymmetricLossMultiLabel()
    best_loss, best_epoch, history, first = math.inf, 0, [], 1
    last = folder / "last.pt"
    if last.exists():
        saved = torch.load(last, map_location="cpu", weights_only=False)
        assert saved["protocol_sha256"] == digest
        model.load_state_dict(saved["model"])
        optimizer.load_state_dict(saved["optimizer"])
        scheduler.load_state_dict(saved["scheduler"])
        scaler.load_state_dict(saved["scaler"])
        best_loss, best_epoch, history = saved["best_loss"], saved["best_epoch"], saved["history"]
        first = saved["epoch"]+1
        random.setstate(saved["python_rng"])
        np.random.set_state(saved["numpy_rng"])
        torch.set_rng_state(saved["torch_rng"])
        torch.cuda.set_rng_state_all(saved["cuda_rng"])
        loaders["train"].generator.set_state(saved["loader_rng"])
        del saved
    begin = time.time()
    for epoch in tqdm(range(first, SETTINGS["epochs"]+1), desc=f"{MODELS[key]} 折{fold+1}"):
        model.train()
        losses = []
        for batch in loaders["train"]:
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast("cuda"):
                output = base.forward_batch(model, batch, token_ids, token_mask, device, training=True)
                loss = criterion(output["logits"], batch[2].to(device))
                for name, weight in auxiliary.items():
                    loss = loss + float(weight) * output["aux_losses"][name]
            assert torch.isfinite(loss), (key, fold, epoch, "非有限损失")
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()
            losses.append(float(loss.detach()))
        val_loss, _, _, _ = base.evaluate(model, loaders["val"], token_ids, token_mask, device)
        history.append({"epoch": epoch, "train_loss": statistics.mean(losses), "val_loss": val_loss})
        if val_loss < best_loss:
            best_loss, best_epoch = val_loss, epoch
            atomic_torch(folder / "best_model.pt", {"state_dict": {k: v.detach().cpu().clone() for k,v in model.state_dict().items()},
                          "epoch": epoch, "model_key": key, "model_parameters": params, "protocol_sha256": digest})
        save_json(folder / "history.json", history)
        atomic_torch(last, {"model": model.state_dict(), "optimizer": optimizer.state_dict(), "scheduler": scheduler.state_dict(),
                     "scaler": scaler.state_dict(), "epoch": epoch, "best_epoch": best_epoch, "best_loss": best_loss,
                     "history": history, "protocol_sha256": digest, "python_rng": random.getstate(),
                     "numpy_rng": np.random.get_state(), "torch_rng": torch.get_rng_state(),
                     "cuda_rng": torch.cuda.get_rng_state_all(), "loader_rng": loaders["train"].generator.get_state()})
        print(f"{MODELS[key]} 折{fold+1} epoch={epoch} val_loss={val_loss:.6f}", flush=True)
    best = torch.load(folder / "best_model.pt", map_location="cpu", weights_only=False)
    model.load_state_dict(best["state_dict"])
    _, val_y, val_p, val_ids = base.evaluate(model, loaders["val"], token_ids, token_mask, device)
    thresholds = tune_thresholds(val_y, val_p, SETTINGS["threshold_grid"])
    _, test_y, test_p, test_ids = base.evaluate(model, loaders["test"], token_ids, token_mask, device)
    assert val_ids.tolist() == split["val"] and test_ids.tolist() == split["test"]
    metrics = calculate_metrics(test_y, test_p, thresholds, LABELS)
    from sklearn.metrics import f1_score
    metrics.update(model=MODELS[key], model_key=key, fold=fold+1, protocol_sha256=digest,
                   best_epoch=best_epoch, best_val_loss=best_loss, seed=seed,
                   macro_f1_fixed_0_5=float(f1_score(test_y, test_p >= .5, average="macro", zero_division=0)),
                   wall_seconds=time.time()-begin, split_sizes={k: len(v) for k,v in split.items()})
    base.write_predictions(folder / "validation_predictions.csv", val_ids, val_y, val_p, thresholds)
    base.write_predictions(folder / "test_predictions.csv", test_ids, test_y, test_p, thresholds)
    best["thresholds"] = thresholds.tolist()
    atomic_torch(folder / "best_model.pt", best)
    save_json(folder / "test_metrics.json", metrics)
    save_json(folder / "completed.json", {"protocol_sha256": digest, "model": key, "fold": fold+1})


def aggregate():
    p = json.loads((OUT / "protocol.json").read_text())
    summary, progress = [], []
    for key, name in MODELS.items():
        records = []
        for fold in range(1, 6):
            folder = OUT / f"fold_{fold}" / key
            if (folder / "completed.json").exists():
                record = json.loads((folder / "test_metrics.json").read_text())
                assert record["protocol_sha256"] == p["protocol_sha256"]
                records.append(record)
        progress.append({"model": name, "completed_folds": len(records)})
        if len(records) == 5:
            summary.append({"model": name, "model_key": key,
                "macro_f1_mean": statistics.mean(r["macro_f1"] for r in records),
                "macro_f1_std": statistics.stdev(r["macro_f1"] for r in records),
                "fixed_0_5_mean": statistics.mean(r["macro_f1_fixed_0_5"] for r in records),
                "fixed_0_5_std": statistics.stdev(r["macro_f1_fixed_0_5"] for r in records)})
    save_json(OUT / "summary.json", summary)
    save_json(OUT / "progress.json", progress)
    with (OUT / "summary.csv").open("w", newline="", encoding="utf-8-sig") as stream:
        writer = csv.DictWriter(stream, fieldnames=["model", "model_key", "macro_f1_mean", "macro_f1_std", "fixed_0_5_mean", "fixed_0_5_std"])
        writer.writeheader()
        writer.writerows(summary)
    lines = ["# AbdomenAtlas 3.0 Mini 表1实验", "", "ALM-MIL保留原始u_it；仅完整五折进入此汇总，F1以百分数显示。", "",
             "| 模型 | 验证集阈值 F1 | 固定0.5阈值 F1 |", "|---|---:|---:|"]
    for r in summary:
        lines.append(f"| {r['model']} | {100*r['macro_f1_mean']:.2f} ± {100*r['macro_f1_std']:.2f} | {100*r['fixed_0_5_mean']:.2f} ± {100*r['fixed_0_5_std']:.2f} |")
    lines += ["", "具体划分、掩码、冻结编码器及任务适配范围见 protocol.json；这里不自动写入论文。"]
    (OUT / "results.md").write_text("\n".join(lines)+"\n", encoding="utf-8")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", choices=list(MODELS))
    parser.add_argument("--fold", type=int, choices=range(1, 6))
    parser.add_argument("--initialize", action="store_true")
    parser.add_argument("--aggregate", action="store_true")
    args = parser.parse_args()
    if args.initialize:
        initialize()
        return
    if args.aggregate:
        aggregate()
        return
    assert args.model and args.fold
    torch.set_num_threads(2)
    torch.cuda.set_per_process_memory_fraction(.42)
    torch.backends.mha.set_fastpath_enabled(False)
    configure()
    p = protocol()
    assert json.loads((OUT / "protocol.json").read_text()) == p, "源码或数据协议改变"
    rows, folds, y = read_inputs()
    marker = OUT / f"fold_{args.fold}" / args.model / "completed.json"
    if marker.exists():
        assert json.loads(marker.read_text())["protocol_sha256"] == p["protocol_sha256"]
        return
    if args.model in TEXT_MODELS:
        text_fold(args.model, args.fold-1, rows, folds, p["protocol_sha256"])
    else:
        image_fold(args.model, args.fold-1, rows, folds, y, p["protocol_sha256"])


if __name__ == "__main__":
    main()
