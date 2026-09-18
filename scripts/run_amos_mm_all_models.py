#!/usr/bin/env python3
"""AMOS-MM 三/七标签：表2全部基线和AMEF，五折开发与固定人工测试集。"""
from __future__ import annotations

import argparse
import csv
import fcntl
import hashlib
import json
import math
import os
import shutil
from pathlib import Path
import sys
import time
import traceback

import numpy as np
import torch
import yaml
from sklearn.metrics import f1_score, precision_score, recall_score, roc_auc_score
from tqdm import tqdm

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
from prepare_amos_mm_experiments import OUT as INPUT, LABELS, save_json, sha256
import run_physionet_ct_ich_table2_baselines as image_base
import run_cq500_table2_multimodal_baselines as fusion_base
from run_cq500_table2_text_baselines import MODELS as TEXT_MODELS
from amos_mm_model_adapters import PartialLabelAMEF, masked_asl, masked_bce, forward_image, forward_fusion
from exp_10.data import TextRecord, build_train_vocabulary, encode_text
from exp_10.models import build_text_classifier
from scripts.task3_apro_cope_ablation_scheduler import base_model_params

OUT = ROOT / "outputs/amos_mm/all_models"
IMAGE_EXCLUSIONS = INPUT / "image_exclusions.json"
IMAGE_MODELS = {k: v["display_name"] for k, v in image_base.MODEL_SPECS.items() if k != "amef_image_branch"}
MODELS = {**IMAGE_MODELS, **TEXT_MODELS, **fusion_base.MODELS, "amef_multimodal": "AMEF-MIL"}
GRID = np.round(np.arange(.1, .901, .05), 2)
TEXT_CFG = yaml.safe_load((ROOT / "src/configs/task2/exp10_text_classification.yaml").read_text())
SETTINGS = {"epochs": 30, "text_epochs": 50, "text_patience": 8, "batch_size": 16,
            "text_batch_size": 64, "lr": .0002, "weight_decay": .02, "text_weight_decay": .0001,
            "warmup_ratio": .2, "instance_dropout": .25, "text_max_length": 512,
            "text_vocab_size": 8192, "seed": 42, "threshold_grid": GRID.tolist(),
            "precision": "图像/多模态基线AMP；文本与AMEF使用FP32"}


def category(key):
    return "纯图像" if key in IMAGE_MODELS else "纯文本" if key in TEXT_MODELS else "AMEF-MIL" if key == "amef_multimodal" else "多模态基线"


def parameters(key):
    if key in IMAGE_MODELS:
        return image_base.MODEL_SPECS[key]
    if key in TEXT_MODELS:
        return TEXT_CFG["model"]
    if key == "amef_multimodal":
        p = base_model_params("apro_full")
        main = yaml.safe_load((ROOT / "src/configs/task3/t3_main_model.yaml").read_text())
        p["label_query_consistency_weight"] = main["model"]["params"]["label_query_consistency_weight"]
        return p
    return yaml.safe_load((ROOT / "src/configs/task2/model.yaml").read_text())["models"][key]


def image_exclusion_indices(rows):
    manifest = json.loads(IMAGE_EXCLUSIONS.read_text())
    assert manifest["schema_version"] == 1
    excluded = set()
    for item in manifest["excluded"]:
        index = int(item["case_index"])
        assert item["applies_to"] == ["pure_image", "multimodal"]
        assert rows[index]["case_index"] == index
        assert rows[index]["scan_id"] == item["scan_id"]
        assert rows[index]["label_source"] == item["label_source"] == "automatic_development"
        excluded.add(index)
    assert len(excluded) == len(manifest["excluded"])
    return excluded


def protocol():
    sources = [Path(__file__), ROOT / "src/scripts/amos_mm_model_adapters.py",
               ROOT / "src/scripts/prepare_amos_mm_experiments.py", Path(image_base.__file__),
               ROOT / "src/scripts/task3_apro_cope_ablation_scheduler.py",
               INPUT / "preparation_protocol.json", INPUT / "samples.json", INPUT / "splits.json",
               IMAGE_EXCLUSIONS]
    for rel in ("exp_4", "exp_8", "exp_10", "model", "baselines/task1", "sotas/task1", "sotas/task2", "training"):
        sources.extend(sorted((ROOT / "src" / rel).rglob("*.py")))
    p = {"models": MODELS, "tasks": {str(c): LABELS[:c] for c in (3, 7)}, "settings": SETTINGS,
         "model_parameters": {k: parameters(k) for k in MODELS},
         "source_sha256": {str(p): sha256(p) for p in sources},
         "split": "纯文本使用1487例中的四个开发折训练、另一折验证；图像与多模态从开发折中排除1例官方损坏CT后使用1486例；固定200例人工标签测试不变",
         "image_exclusion": "amos_5964仅从纯图像/多模态训练与验证排除；官方AMOS-MM条目自身为截断gzip，禁止补零或换用不同CT",
         "selection": "图像与多模态按验证masked-ASL最小；文本按验证已知标签macro-F1最大并8轮早停",
         "threshold": "仅开发验证集的已知标签；网格0.10至0.90；无阳性或无阴性时固定0.5",
         "primary_metric": "同一200例人工测试集上五次训练的macro-F1均值及样本标准差；不是互斥测试折OOF",
         "secondary_metric": "逐标签F1、micro-F1、固定0.5阈值F1、共阳性子集F1及检查级完全匹配率",
         "amef": "全图文预测路径，ASL+0.01查询判别损失；不启动另行图像分支蒸馏",
         "partial_supervision": "主分类、CLAM实例、DTFD伪包、MMFNet分支和AMEF查询辅助监督均屏蔽未知标签",
         "label_query_extension": "C标签的C-1个循环置换，三标签与原两个置换一致；同一0.01权重",
         "clam_padding": "本次CLAM实例辅助监督只选择有效CT层，避免可变长序列填充进入底部证据",
         "report_scope": "诊断词固定掩码的官方所见；标签也来自报告，属于报告辅助的检查级器官异常共存分类",
         "test_use": "每个模型每次训练完成及阈值冻结后评估一次测试集，保留全部结果"}
    p["protocol_sha256"] = hashlib.sha256(json.dumps(p, sort_keys=True, ensure_ascii=False).encode()).hexdigest()
    return p


def initialize():
    OUT.mkdir(parents=True, exist_ok=True)
    p = protocol()
    path = OUT / "protocol.json"
    if path.exists() and json.loads(path.read_text()) != p:
        raise ValueError("正式训练协议变化，禁止混合结果；请使用新的实验输出目录")
    save_json(path, p)
    order = [*TEXT_MODELS, *IMAGE_MODELS, *fusion_base.MODELS, "amef_multimodal"]
    jobs = [{"labels": c, "model": k, "fold": f} for f in range(5) for k in order for c in (3, 7)]
    save_json(OUT / "jobs.json", jobs)
    return p


def build_model(key, count, vocabulary_size=8192):
    if key in TEXT_MODELS:
        return build_text_classifier(key, vocabulary_size=vocabulary_size, hash_vocab_size=8192,
                                    num_labels=count, max_length=512, model_config=parameters(key)), {}
    if key in IMAGE_MODELS:
        image_base.LABEL_NAMES = LABELS[:count]
        return image_base.build_model(key), parameters(key).get("aux_weights", {})
    p = dict(parameters(key))
    weights = {k.removesuffix("_weight"): p.pop(k) for k in list(p) if k.endswith("_weight")}
    if key == "amef_multimodal":
        model = PartialLabelAMEF(**p, pretrained=False, num_labels=count)
    else:
        from sotas.task2 import build_task2_multimodal_sota
        model = build_task2_multimodal_sota(key, **p, pretrained=False, num_labels=count)
    model.instance_encoder.backbone = image_base.CachedFeatureBackbone(768, p["feature_dim"], p["dropout"])
    return model, weights


def load_features(rows):
    digest = json.loads((INPUT / "preparation_protocol.json").read_text())["preparation_sha256"]
    excluded = image_exclusion_indices(rows)
    result = []
    for row in tqdm(rows, desc="加载共用图像缓存"):
        if row["case_index"] in excluded:
            result.append(None)
            continue
        with np.load(INPUT / "features" / f"{row['case_index']:04d}.npz", allow_pickle=False) as c:
            assert str(c["scan_id"]) == row["scan_id"] and str(c["preparation_sha256"]) == digest
            x, pos, length = c["features"].copy(), c["slice_indices"].copy(), int(c["original_count"])
            assert x.shape == (len(pos), 768) and np.isfinite(x).all()
            assert np.all(np.diff(pos) > 0) and 0 <= pos[0] <= pos[-1] < length
            result.append((x, pos, length))
    return result


def tokens_for(key, rows, train, count):
    records = [TextRecord(rows[i]["scan_id"], "", rows[i]["findings_masked"],
                          np.asarray(rows[i]["labels"][:count]), ()) for i in train]
    vocabulary = build_train_vocabulary(records, max_vocab_size=8192, min_frequency=1)
    if key in TEXT_MODELS:
        encoded = [encode_text(r["findings_masked"], encoder_name=key, vocabulary=vocabulary,
                               hash_vocab_size=8192, max_length=512) for r in rows]
    else:
        from training.data import _encode_text_fields
        encoded = [_encode_text_fields({"watch": r["findings_masked"]}, ("watch",), max_length=512,
                                        vocab_size=8192) for r in rows]
    ids = torch.as_tensor(np.stack([np.asarray(v[0]) for v in encoded]), dtype=torch.long)
    mask = torch.as_tensor(np.stack([np.asarray(v[1]) for v in encoded]), dtype=torch.bool)
    assert mask.any(1).all()
    return ids, mask, vocabulary


def pack(ids, bags, training):
    picked = []
    for index in ids:
        bag = bags[int(index)]
        assert bag is not None, f"影像排除病例进入了图像批次：{int(index)}"
        x, pos, count = bag
        indices = np.arange(len(x))
        if training:
            indices = np.sort(np.random.choice(len(x), max(1, round(len(x) * .75)), replace=False))
        picked.append((x[indices], pos[indices], count))
    n = max(len(v[0]) for v in picked)
    images = torch.zeros(len(ids), n, 768, device="cuda")
    mask = torch.zeros(len(ids), n, dtype=torch.bool, device="cuda")
    positions = torch.full((len(ids), n), -1, dtype=torch.long, device="cuda")
    counts = torch.tensor([v[2] for v in picked], device="cuda")
    for j, (x, p, _) in enumerate(picked):
        images[j, :len(x)] = torch.as_tensor(x, device="cuda")
        mask[j, :len(x)] = True
        positions[j, :len(x)] = torch.as_tensor(p, device="cuda")
    # 旧图像编码器的接口为 B,N,C,H,W；缓存使用 C=768,H=W=1。
    return images[..., None, None], mask, positions, counts


def forward(model, key, ids, bags, token_ids, token_mask, y, known, training):
    t, k = y[ids].cuda(), known[ids].cuda()
    if key in TEXT_MODELS:
        return {"logits": model(token_ids[ids].cuda(), token_mask[ids].cuda()), "aux_losses": {}}
    images, mask, positions, counts = pack(ids, bags, training)
    if key in IMAGE_MODELS:
        return forward_image(model, key, images, mask, t, k, training)
    return forward_fusion(model, key, images, mask, token_ids[ids].cuda(), token_mask[ids].cuda(),
                          positions, counts, t, k, training)


def tune(y, probabilities, known):
    thresholds = np.full(y.shape[1], .5)
    for j in range(y.shape[1]):
        obs = known[:, j]
        if len(np.unique(y[obs, j])) < 2:
            continue
        scores = [f1_score(y[obs, j], probabilities[obs, j] >= t, zero_division=0) for t in GRID]
        thresholds[j] = GRID[int(np.argmax(scores))]
    return thresholds


def known_f1(y, p, known, thresholds):
    return float(np.mean([f1_score(y[known[:, j], j], p[known[:, j], j] >= thresholds[j], zero_division=0)
                          for j in range(y.shape[1]) if known[:, j].any()]))


def test_metrics(y, p, thresholds):
    pred = p >= thresholds
    copositive = y.sum(1) >= 2
    return {"macro_f1": float(f1_score(y, pred, average="macro", zero_division=0)),
            "micro_f1": float(f1_score(y, pred, average="micro", zero_division=0)),
            "macro_f1_at_0.5": float(f1_score(y, p >= .5, average="macro", zero_division=0)),
            "per_label_f1": f1_score(y, pred, average=None, zero_division=0).tolist(),
            "per_label_precision": precision_score(y, pred, average=None, zero_division=0).tolist(),
            "per_label_recall": recall_score(y, pred, average=None, zero_division=0).tolist(),
            "macro_auroc": float(roc_auc_score(y, p, average="macro")),
            "exact_match": float((pred == y).all(1).mean()),
            "copositive_n": int(copositive.sum()),
            "copositive_macro_f1": float(f1_score(y[copositive], pred[copositive], average="macro", zero_division=0)),
            "copositive_exact_match": float((pred[copositive] == y[copositive]).all(1).mean())}


def train_job(job, rows, splits, bags, digest, worker):
    c, key, fold = job["labels"], job["model"], job["fold"]
    is_text = key in TEXT_MODELS
    folder = job_folder(job)
    folder.mkdir(parents=True, exist_ok=True)
    seed = 42 + 100 * (fold + 1)
    image_base.seed_everything(seed)
    val = np.asarray(splits["validation_folds"][fold])
    train = np.asarray(sorted(set(splits["development"]) - set(val)))
    test = np.asarray(splits["test"])
    assert not (set(train) & set(val) or set(train) & set(test) or set(val) & set(test))
    excluded = image_exclusion_indices(rows)
    assert not excluded.intersection(test), "影像排除清单不得影响固定人工测试集"
    applied_exclusions = []
    if not is_text:
        train = np.asarray([index for index in train if index not in excluded])
        val = np.asarray([index for index in val if index not in excluded])
        applied_exclusions = [rows[index]["scan_id"] for index in sorted(excluded)]
        assert not excluded.intersection(train) and not excluded.intersection(val)
    y = torch.tensor([r["labels"][:c] for r in rows], dtype=torch.float32)
    known = torch.tensor([r["known_mask"][:c] for r in rows], dtype=torch.bool)
    assert known[test].all()
    token_ids, token_mask, vocabulary = tokens_for(key, rows, train, c)
    model, weights = build_model(key, c, len(vocabulary))
    model.cuda()
    use_amp = not is_text and key != "amef_multimodal"
    batch_size = SETTINGS["text_batch_size"] if is_text else SETTINGS["batch_size"]
    epochs = SETTINGS["text_epochs"] if is_text else SETTINGS["epochs"]
    pos = (y[train] * known[train]).sum(0)
    neg = ((1 - y[train]) * known[train]).sum(0)
    pos_weight = (neg / pos.clamp_min(1)).cuda()
    optimizer = torch.optim.AdamW(model.parameters(), lr=SETTINGS["lr"],
                                 weight_decay=SETTINGS["text_weight_decay"] if is_text else SETTINGS["weight_decay"])
    total_steps = math.ceil(len(train) / batch_size) * epochs
    warmup = round(total_steps * SETTINGS["warmup_ratio"])

    def rate(step):
        return (step + 1) / max(1, warmup) if step < warmup else .5 * (1 + math.cos(math.pi * (step - warmup) / max(1, total_steps - warmup)))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, rate) if not is_text else None
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)

    def criterion(logits, targets, obs):
        return masked_bce(logits, targets, obs, pos_weight) if is_text else masked_asl(logits, targets, obs)

    def evaluate(indices):
        model.eval()
        probabilities, numerator, denominator = [], 0., 0
        with torch.inference_mode():
            for start in range(0, len(indices), batch_size):
                ids = indices[start:start + batch_size]
                with torch.autocast("cuda", enabled=use_amp):
                    out = forward(model, key, ids, bags, token_ids, token_mask, y, known, False)
                    loss = criterion(out["logits"], y[ids].cuda(), known[ids].cuda())
                assert torch.isfinite(out["logits"]).all() and torch.isfinite(loss)
                observations = int(known[ids].sum())
                numerator += float(loss) * observations
                denominator += observations
                probabilities.append(out["logits"].float().sigmoid().cpu().numpy())
        return np.concatenate(probabilities), numerator / max(1, denominator)

    save_json(folder / "config.json", {"job": job, "seed": seed, "parameters": parameters(key), "settings": SETTINGS,
              "protocol_sha256": digest, "training_positive_counts": pos.tolist(), "training_negative_counts": neg.tolist(),
              "train": train.tolist(), "validation": val.tolist(), "test": test.tolist(),
              "image_exclusions_applied": applied_exclusions, "vocabulary": vocabulary if is_text else "fixed_hash"})
    begin = time.monotonic()
    best, best_epoch, history, state, stale = math.inf, 0, [], None, 0
    print(f"开始 {c}标签 {MODELS[key]} 开发折{fold + 1}，训练/验证/人工测试={len(train)}/{len(val)}/{len(test)}", flush=True)
    for epoch in range(1, epochs + 1):
        model.train()
        losses, order = [], np.random.permutation(train)
        for start in range(0, len(order), batch_size):
            ids = order[start:start + batch_size]
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast("cuda", enabled=use_amp):
                output = forward(model, key, ids, bags, token_ids, token_mask, y, known, True)
                loss = criterion(output["logits"], y[ids].cuda(), known[ids].cuda())
                for name, weight in weights.items():
                    if weight:
                        loss = loss + float(weight) * output["aux_losses"][name]
            if not torch.isfinite(loss):
                raise ValueError(f"{key}出现非有限损失，停止该任务")
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            # AMP溢出由GradScaler跳过该步并降低缩放；FP32的非有限梯度直接报错。
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5., error_if_nonfinite=not use_amp)
            scaler.step(optimizer)
            scaler.update()
            if scheduler:
                scheduler.step()
            losses.append(float(loss.detach()))
        val_p, val_loss = evaluate(val)
        thresholds = tune(y[val].numpy(), val_p, known[val].numpy())
        val_f1 = known_f1(y[val].numpy(), val_p, known[val].numpy(), thresholds)
        score = -val_f1 if is_text else val_loss
        if score < best:
            best, best_epoch, stale = score, epoch, 0
            state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        else:
            stale += 1
        row = {"epoch": epoch, "train_loss": float(np.mean(losses)), "val_loss": val_loss,
               "val_known_macro_f1": val_f1, "elapsed_seconds": time.monotonic() - begin}
        history.append(row)
        save_json(folder / "history.json", history)
        save_json(OUT / f"worker_{worker}_status.json", {"state": "training", "job": job, **row, "updated_at": time.time()})
        print(f"{c}标签 {key} 折{fold + 1} 轮{epoch}/{epochs} loss={row['train_loss']:.5f} val_loss={val_loss:.5f} val_F1={val_f1:.4f}", flush=True)
        if is_text and stale >= SETTINGS["text_patience"]:
            break
    assert state is not None
    model.load_state_dict(state)
    val_p, _ = evaluate(val)
    thresholds = tune(y[val].numpy(), val_p, known[val].numpy())
    # 在任何读取测试图像/文本预测之前，先将模型和阈值冻结到磁盘。
    temp = folder / "best_model.tmp.pt"
    torch.save({"state_dict": state, "job": job, "best_epoch": best_epoch, "thresholds": thresholds,
                "protocol_sha256": digest}, temp)
    temp.replace(folder / "best_model.pt")
    test_p, _ = evaluate(test)
    result = {"job": job, "display_name": MODELS[key], "category": category(key), "labels": LABELS[:c],
              "best_epoch": best_epoch, "thresholds": thresholds.tolist(), "protocol_sha256": digest,
              "seconds": time.monotonic() - begin, "test_n": len(test),
              "image_exclusions_applied": applied_exclusions, **test_metrics(y[test].numpy(), test_p, thresholds)}
    np.savez_compressed(folder / "test_predictions.npz", scan_ids=np.asarray([rows[i]["scan_id"] for i in test]),
                        labels=y[test].numpy(), probabilities=test_p, thresholds=thresholds)
    save_json(folder / "result.json", result)
    print(f"完成 {c}标签 {key} 折{fold + 1} 测试macro-F1={result['macro_f1']:.4f}", flush=True)
    del model, optimizer, state
    torch.cuda.empty_cache()


def job_folder(job):
    return OUT / f"{job['labels']}_labels" / f"fold_{job['fold'] + 1}" / job["model"]


def aggregate():
    with (OUT / "aggregate.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        results = [json.loads(p.read_text()) for p in OUT.glob("*_labels/fold_*/*/result.json")]
        summary = []
        for c in (3, 7):
            for key in MODELS:
                selected = [r for r in results if r["job"]["labels"] == c and r["job"]["model"] == key]
                if not selected:
                    continue
                values = [r["macro_f1"] for r in selected]
                summary.append({"labels": c, "model": MODELS[key], "category": category(key),
                                "completed_runs": len(selected), "macro_f1_mean": float(np.mean(values)),
                                "macro_f1_std": float(np.std(values, ddof=1)) if len(values) > 1 else None,
                                "copositive_macro_f1_mean": float(np.mean([r["copositive_macro_f1"] for r in selected]))})
        save_json(OUT / "summary.json", {"completed": len(results), "total": 200, "models": summary,
                  "note": "纯文本使用1487例开发数据；图像/多模态因官方amos_5964影像损坏使用1486例开发数据；固定200例人工测试集不变。少于五次的条目是中间进度。",
                  "updated_at": time.time()})
        if summary:
            path = OUT / "summary.csv"
            temp = path.with_suffix(".tmp.csv")
            with temp.open("w", encoding="utf-8-sig", newline="") as stream:
                writer = csv.DictWriter(stream, fieldnames=list(summary[0]))
                writer.writeheader()
                writer.writerows(summary)
            temp.replace(path)


def validate_feature_exclusions():
    rows = json.loads((INPUT / "samples.json").read_text())
    digest = json.loads((INPUT / "preparation_protocol.json").read_text())["preparation_sha256"]
    excluded = image_exclusion_indices(rows)
    available = 0
    for row in tqdm(rows, desc="核验图像特征与排除清单"):
        path = INPUT / "features" / f"{row['case_index']:04d}.npz"
        if row["case_index"] in excluded:
            assert not path.exists(), f"排除病例不应使用缓存特征：{path}"
            continue
        with np.load(path, allow_pickle=False) as cache:
            assert str(cache["scan_id"]) == row["scan_id"]
            assert str(cache["preparation_sha256"]) == digest
            features = cache["features"]
            positions = cache["slice_indices"]
            count = int(cache["original_count"])
            assert features.shape == (len(positions), 768) and np.isfinite(features).all()
            assert np.all(np.diff(positions) > 0) and 0 <= positions[0] <= positions[-1] < count
        available += 1
    assert available + len(excluded) == len(rows)
    save_json(INPUT / "feature_status.json", {"state": "complete", "completed": available,
              "available": available, "excluded": len(excluded), "total": len(rows), "missing": [],
              "excluded_cases": [rows[index]["scan_id"] for index in sorted(excluded)],
              "preparation_sha256": digest, "updated_at": time.time()})
    print(f"图像特征核验通过：可用{available}例，按协议排除{len(excluded)}例。", flush=True)


def migrate_image_exclusion_protocol():
    OUT.mkdir(parents=True, exist_ok=True)
    current_path = OUT / "protocol.json"
    current = json.loads(current_path.read_text())
    updated = protocol()
    if current == updated:
        print("影像排除协议已经迁移。", flush=True)
        return
    old_digest = current["protocol_sha256"]
    new_digest = updated["protocol_sha256"]
    result_paths = sorted(OUT.glob("*_labels/fold_*/*/result.json"))
    assert result_paths, "没有可迁移的既有结果"
    jobs = []
    for path in result_paths:
        result = json.loads(path.read_text())
        assert result["job"]["model"] in TEXT_MODELS, "已有影像结果时禁止元数据迁移，必须另建输出目录"
        assert result["protocol_sha256"] in {old_digest, new_digest}
        jobs.append(result["job"])
    backup = OUT / "protocol_before_image_exclusion.json"
    if backup.exists():
        assert json.loads(backup.read_text())["protocol_sha256"] == old_digest
    else:
        save_json(backup, current)
    migration = {"from_protocol_sha256": old_digest, "to_protocol_sha256": new_digest,
                 "migrated_completed_jobs": len(result_paths), "jobs": jobs,
                 "scope": "Only completed pure-text metadata is migrated; their reports, splits, model weights, predictions and metrics are unchanged.",
                 "checkpoint_note": "Completed text checkpoints retain the original protocol digest as a historical record of the protocol used at training time.",
                 "reason": "The new manifest affects only future pure-image and multimodal train/validation indices.",
                 "time": time.time()}
    for path in result_paths:
        result = json.loads(path.read_text())
        result["protocol_sha256"] = new_digest
        result["protocol_migration"] = {"from": old_digest, "training_inputs_unchanged": True}
        save_json(path, result)
        config_path = path.with_name("config.json")
        config = json.loads(config_path.read_text())
        assert config["protocol_sha256"] in {old_digest, new_digest}
        config["protocol_sha256"] = new_digest
        config["protocol_migration"] = {"from": old_digest, "training_inputs_unchanged": True}
        save_json(config_path, config)
    save_json(OUT / "protocol_migration_image_exclusion.json", migration)
    save_json(current_path, updated)
    aggregate()
    print(f"已迁移{len(result_paths)}个不受影响的纯文本结果：{old_digest[:12]} -> {new_digest[:12]}", flush=True)


def worker(index):
    torch.set_num_threads(2)
    torch.backends.mha.set_fastpath_enabled(False)
    torch.cuda.set_per_process_memory_fraction(.40)
    recorded = json.loads((OUT / "protocol.json").read_text())
    assert protocol() == recorded, "代码或实验输入变化，停止混用"
    digest = recorded["protocol_sha256"]
    rows = json.loads((INPUT / "samples.json").read_text())
    splits = json.loads((INPUT / "splits.json").read_text())
    jobs = json.loads((OUT / "jobs.json").read_text())
    bags = None
    while True:
        if shutil.disk_usage(OUT).free < 8 * 1024**3:
            save_json(OUT / f"worker_{index}_status.json", {"state": "waiting_for_disk", "required_free_gib": 8, "updated_at": time.time()})
            time.sleep(30)
            continue
        completed, failed, busy, claimed = 0, 0, 0, False
        ready = (INPUT / "feature_status.json").exists() and json.loads((INPUT / "feature_status.json").read_text())["state"] == "complete"
        for job in jobs:
            folder = job_folder(job)
            if (folder / "result.json").exists():
                completed += 1
                continue
            if (folder / "error.json").exists():
                failed += 1
                continue
            if job["model"] not in TEXT_MODELS and not ready:
                continue
            folder.mkdir(parents=True, exist_ok=True)
            with (folder / "run.lock").open("a") as lock:
                try:
                    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
                except BlockingIOError:
                    busy += 1
                    continue
                if (folder / "result.json").exists() or (folder / "error.json").exists():
                    continue
                claimed = True
                try:
                    if job["model"] not in TEXT_MODELS and bags is None:
                        bags = load_features(rows)
                    train_job(job, rows, splits, bags, digest, index)
                except Exception:
                    error = traceback.format_exc()
                    print(error, flush=True)
                    save_json(folder / "error.json", {"job": job, "traceback": error, "time": time.time()})
                    torch.cuda.empty_cache()
                aggregate()
                break
        if not claimed:
            state = "complete" if completed == 200 else "finished_with_errors" if completed + failed == 200 else "waiting"
            save_json(OUT / f"worker_{index}_status.json", {"state": state, "completed": completed, "failed": failed,
                      "busy": busy, "features_ready": ready, "updated_at": time.time()})
            if completed + failed == 200:
                break
            time.sleep(15)


def smoke():
    from training.losses import AsymmetricLossMultiLabel
    torch.set_num_threads(2)
    torch.backends.mha.set_fastpath_enabled(False)
    torch.cuda.set_per_process_memory_fraction(.38)
    z = torch.randn(4, 7, device="cuda", requires_grad=True)
    y = torch.randint(0, 2, (4, 7), device="cuda").float()
    obs = torch.rand(4, 7, device="cuda") > .4
    loss = masked_asl(z, y, obs) + masked_bce(z, y, obs)
    loss.backward()
    assert torch.equal(z.grad[~obs], torch.zeros_like(z.grad[~obs]))
    assert torch.allclose(masked_asl(z, y, torch.ones_like(obs)), AsymmetricLossMultiLabel()(z, y))
    assert float(masked_asl(z, y, torch.zeros_like(obs)).detach()) == 0
    tests = []
    for c in (3, 7):
        targets = torch.tensor([[1, 0, 0, 1, 0, 0, 1], [0, 1, 0, 0, 1, 0, 0]], dtype=torch.float32)[:, :c]
        known = torch.ones_like(targets, dtype=torch.bool)
        known[:, 2] = False
        bags = [(np.random.randn(8, 768).astype(np.float32), np.arange(8) * 3, 25),
                (np.random.randn(5, 768).astype(np.float32), np.arange(5) * 3, 16)]
        ids, mask = torch.randint(1, 90, (2, 512)), torch.ones(2, 512, dtype=torch.bool)
        mask[1, 300:] = False
        ids[~mask] = 0
        for key in tqdm(MODELS, desc=f"{c}标签全模型前后向检查"):
            model, weights = build_model(key, c, 100)
            model.cuda().train()
            amp = key not in TEXT_MODELS and key != "amef_multimodal"
            with torch.autocast("cuda", enabled=amp):
                output = forward(model, key, np.asarray([0, 1]), bags, ids, mask, targets, known, True)
                assert output["logits"].shape == (2, c)
                loss = masked_asl(output["logits"], targets.cuda(), known.cuda())
                for name, weight in weights.items():
                    if weight:
                        loss = loss + weight * output["aux_losses"][name]
            assert torch.isfinite(loss)
            loss.backward()
            assert all(torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None)
            model.eval()
            with torch.inference_mode(), torch.autocast("cuda", enabled=amp):
                evaluation = forward(model, key, np.asarray([0, 1]), bags, ids, mask, targets, known, False)
                assert torch.isfinite(evaluation["logits"]).all()
            tests.append({"labels": c, "model": key, "loss": float(loss.detach()), "finite_gradient": True, "finite_eval": True})
            del model, output, evaluation, loss
            torch.cuda.empty_cache()
    save_json(OUT / "smoke_tests.json", {"masked_loss_unknown_gradient_zero": True, "all_known_asl_equivalent": True,
              "all_unknown_zero": True, "models": tests, "note": "合成输入仅用于实现检查，不作为模型性能结果"})
    print("40组前向、反向与推理检查通过。", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--initialize", action="store_true")
    parser.add_argument("--worker", type=int)
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--aggregate", action="store_true")
    parser.add_argument("--validate-feature-exclusions", action="store_true")
    parser.add_argument("--migrate-image-exclusion-protocol", action="store_true")
    args = parser.parse_args()
    if args.initialize:
        initialize()
    elif args.smoke:
        smoke()
    elif args.aggregate:
        aggregate()
    elif args.validate_feature_exclusions:
        validate_feature_exclusions()
    elif args.migrate_image_exclusion_protocol:
        migrate_image_exclusion_protocol()
    elif args.worker is not None:
        worker(args.worker)
    else:
        parser.error("请选择初始化、完整实现检查、工作进程或汇总")
