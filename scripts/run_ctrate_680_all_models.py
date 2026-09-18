#!/usr/bin/env python3
"""CT-RATE680例统一患者五折：表2各类基线、AMEF多模态及图像分支。"""

from __future__ import annotations

import argparse
import csv
import fcntl
import gc
import hashlib
import json
import os
from pathlib import Path
import statistics
import subprocess
import sys
import time
import traceback

import numpy as np
from tqdm import tqdm

from prepare_ctrate_680_experiment import OUT as INPUT, LABELS, ROOT, save_json, sha256

sys.path.insert(0, str(ROOT / "src"))
import run_physionet_ct_ich_table2_baselines as image_base
import run_cq500_table2_multimodal_baselines as fusion_base
import run_cq500_amef_multimodal as amef_base
from run_cq500_table2_text_baselines import MODELS as TEXT_MODELS

IMAGE_MODELS = {k: v["display_name"] for k, v in image_base.MODEL_SPECS.items()}
FUSION_MODELS = dict(fusion_base.MODELS)
MODELS = {**IMAGE_MODELS, **TEXT_MODELS, **FUSION_MODELS, "amef_multimodal": "AMEF-MIL"}
ORIGINAL_FUSION_BUILDER = fusion_base.build_model
SOURCE_SLICE_INDICES, SOURCE_COUNTS = [], []
SETTINGS = {**fusion_base.SETTINGS, "max_instances": 64, "text_max_length": 512}


def category(key):
    if key in IMAGE_MODELS:
        return "纯图像" if key != "amef_image_branch" else "我们的纯图像分支"
    return "纯文本" if key in TEXT_MODELS else "我们的多模态模型" if key == "amef_multimodal" else "多模态基线"


def read_inputs():
    rows = json.loads((INPUT / "samples.json").read_text())
    folds = json.loads((INPUT / "patient_folds.json").read_text())
    assert len(rows) == len({r["patient_id"] for r in rows}) == 680
    assert [r["case_index"] for r in rows] == list(range(680))
    assert sorted(i for group in folds["folds"] for i in group) == list(range(680))
    assert folds["labels"] == LABELS
    y = np.asarray([r["labels"] for r in rows], dtype=np.int64)
    for group, counts in zip(folds["folds"], folds["fold_positive_counts"]):
        assert y[group].sum(0).tolist() == counts
    return rows, folds, y


def parameters():
    import yaml
    from scripts.task3_apro_cope_ablation_scheduler import base_model_params
    cfg = yaml.safe_load((ROOT / "src/configs/task2/model.yaml").read_text())
    values = {k: cfg["models"][k] for k in FUSION_MODELS}
    main = yaml.safe_load((ROOT / "src/configs/task3/t3_main_model.yaml").read_text())
    values["amef_multimodal"] = base_model_params("apro_full")
    values["amef_multimodal"]["label_query_consistency_weight"] = main["model"]["params"]["label_query_consistency_weight"]
    values.update({k: v for k, v in image_base.MODEL_SPECS.items()})
    return values


def text_config(fold):
    import yaml
    cfg = yaml.safe_load((ROOT / "src/configs/task2/exp10_text_classification.yaml").read_text())
    cfg["seed"] = SETTINGS["base_seed"] + 100 * (fold+1)
    cfg["data"].update(label_names=LABELS, text_field="Findings_EN_masked", max_length=512)
    cfg["training"]["num_workers"] = 0
    cfg["experiment_name"] = "ct_rate_680_table2_fivefold"
    cfg["paths"] = {"data_json": str(INPUT / "samples.json")}
    return cfg


def protocol():
    paths = [Path(__file__), ROOT / "src/scripts/prepare_ctrate_680_experiment.py",
             Path(image_base.__file__), Path(fusion_base.__file__), Path(amef_base.__file__),
             ROOT / "src/exp_10/models.py", ROOT / "src/exp_10/data.py",
             ROOT / "src/exp_10/train_text_classification.py", ROOT / "src/exp_8/models.py",
             ROOT / "src/sotas/task2/multimodal_sotas.py", ROOT / "src/training/losses.py",
             ROOT / "src/configs/task2/model.yaml", ROOT / "src/configs/task3/t3_main_model.yaml",
             ROOT / "src/configs/task2/exp10_text_classification.yaml",
             INPUT / "samples.json", INPUT / "patient_folds.json", INPUT / "preparation_protocol.json"]
    value = {"dataset": "CT-RATE fixed 680 patients", "labels": LABELS, "models": MODELS,
             "settings_image_and_multimodal": SETTINGS, "text_config": text_config(0),
             "model_parameters": parameters(), "source_sha256": {str(p): sha256(p) for p in paths},
             "split": "各模型复用同一患者五折：408训练/136验证/136测试；每例仅一次测试",
             "text": "临床Findings_EN统一目标词掩码；不输入诊断印象、标签或其他报告段",
             "image": "冻结ImageNet ConvNeXt-Tiny，均匀64层三窗；所有图像模型共用缓存",
             "amef": "完整主要多模态结构，ASL+0.01标签查询判别损失；额外图像分支单独训练，未加入教师蒸馏",
             "position": "AMEF传入完整标准化轴位序列的原层索引和原长度",
             "selection": "图像/多模态：最低验证分类损失；文本：沿用原验证F1与早停协议",
             "threshold": "逐折验证集选择；统一额外保存固定0.5结果；测试集不参与选择",
             "scope": "新数据集上重新训练的跨任务五折评价，不是胃镜权重零样本外部测试",
             "primary_metric": "五折测试Macro-F1均值和样本标准差；另报汇总OOF、每标签及共阳性子集",
             "result_selection": "预先固定所有模型和全部五折，保留全部结果，不按测试分数筛选模型或配置"}
    value["protocol_sha256"] = hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False).encode()).hexdigest()
    return value


def load_features(rows):
    global SOURCE_SLICE_INDICES, SOURCE_COUNTS
    bags, SOURCE_SLICE_INDICES, SOURCE_COUNTS = [], [], []
    digest = json.loads((INPUT / "preparation_protocol.json").read_text())["preparation_sha256"]
    for row in tqdm(rows, desc="读取680例共用特征缓存"):
        with np.load(INPUT / "features" / f"{row['case_index']:04d}.npz", allow_pickle=False) as c:
            assert str(c["patient_id"]) == row["patient_id"] and str(c["preparation_sha256"]) == digest
            features, indices = c["features"].astype(np.float32), c["slice_indices"].copy()
            count = int(c["original_count"])
            assert features.shape == (len(indices), 768) and 1 <= len(indices) <= 64
            assert np.isfinite(features).all() and np.all(np.diff(indices) > 0)
            assert 0 <= indices[0] <= indices[-1] < count
            bags.append(features)
            SOURCE_SLICE_INDICES.append(indices)
            SOURCE_COUNTS.append(count)
    return bags


def build_model(key, params):
    if key in IMAGE_MODELS:
        return image_base.build_model(key), image_base.MODEL_SPECS[key].get("aux_weights", {})
    if key == "amef_multimodal":
        return amef_base.build_model(key, params)
    return ORIGINAL_FUSION_BUILDER(key, params)


def forward_batch(model, batch, text_ids, text_mask, device, training=False):
    import torch
    features, mask, labels, ids, cached_positions, _ = batch
    key = model._ct_model_key
    if key in ("amef_image_branch", "amef_multimodal"):
        positions = torch.full_like(cached_positions, -1)
        for j, case in enumerate(ids.tolist()):
            valid = cached_positions[j] >= 0
            positions[j, valid] = torch.from_numpy(SOURCE_SLICE_INDICES[case])[cached_positions[j, valid]]
        counts = torch.tensor([SOURCE_COUNTS[i] for i in ids.tolist()], dtype=torch.long, device=device)
        with torch.autocast(device_type=torch.device(device).type, enabled=False):
            if key == "amef_image_branch":
                context, embeds, attention, extra = model.core.encode_long_mil(
                    features.to(device), mask.to(device), positions.to(device), counts)
                return model.core.build_outputs(logits=model.core.classify(embeds), attention=attention,
                                                features=context, extra_outputs=extra)
            kwargs = {"images": features.to(device), "mask": mask.to(device),
                      "watch_token_ids": text_ids[ids].to(device), "watch_token_mask": text_mask[ids].to(device),
                      "instance_indices": positions.to(device), "original_image_counts": counts}
            if training:
                kwargs["labels"] = labels.to(device)
            return model(**kwargs)
    if key in IMAGE_MODELS:
        return image_base.forward_model(model, key, features.to(device), mask.to(device),
                                        labels.to(device) if training else None)
    kwargs = {"images": features.to(device), "mask": mask.to(device),
              "watch_token_ids": text_ids[ids].to(device), "watch_token_mask": text_mask[ids].to(device)}
    if training:
        kwargs["labels"] = labels.to(device)
    return model(**kwargs)


def configured_builder(key, params):
    model, aux = build_model(key, params)
    model._ct_model_key = key
    return model, aux


def configure_adapters():
    image_base.LABEL_NAMES = LABELS
    fusion_base.LABEL_NAMES = LABELS
    fusion_base.MODELS = MODELS
    fusion_base.SETTINGS = SETTINGS
    fusion_base.loader = amef_base.loader
    fusion_base.build_model = configured_builder
    fusion_base.forward_batch = forward_batch


def text_fold(args, key, fold, rows, folds, digest):
    from exp_10.data import TextRecord, build_train_vocabulary
    from exp_10.train_text_classification import train_one_model
    from sklearn.metrics import f1_score
    begin = time.monotonic()
    folder = args.output_dir / f"fold_{fold+1}" / key
    selected = {"train": [i for j, group in enumerate(folds["folds"]) if j not in (fold, (fold+1)%5) for i in group],
                "val": folds["folds"][(fold+1)%5], "test": folds["folds"][fold]}
    records = [TextRecord(str(r["case_index"]), r["patient_id"], r["findings_masked"],
                          np.asarray(r["labels"], dtype=np.int64), tuple(r["mask_hits"])) for r in rows]
    splits = {k: [records[i] for i in indices] for k, indices in selected.items()}
    cfg = text_config(fold)
    vocab = build_train_vocabulary(splits["train"], cfg["data"]["vocab_size"], cfg["data"]["min_token_frequency"])
    save_json(folder / "config.json", cfg)
    save_json(folder / "vocabulary.json", vocab)
    save_json(folder / "split_ids.json", selected)
    metrics = train_one_model(key, cfg, splits, vocab, folder.parent)
    with (folder / "test_predictions.csv").open(encoding="utf-8-sig", newline="") as stream:
        pred_rows = list(csv.DictReader(stream))
    assert [int(r["patient_id"]) for r in pred_rows] == selected["test"]
    y = np.asarray([[int(r[f"true_{label}"]) for label in LABELS] for r in pred_rows])
    prob = np.asarray([[float(r[f"prob_{label}"]) for label in LABELS] for r in pred_rows])
    assert np.isfinite(prob).all() and ((prob >= 0) & (prob <= 1)).all()
    metrics.update(model=MODELS[key], model_key=key, fold=fold+1, protocol_sha256=digest,
                   text_field="Findings_EN_masked", text_source="official_clinical_report",
                   macro_f1_fixed_0_5=float(f1_score(y, prob >= .5, average="macro", zero_division=0)),
                   micro_f1_fixed_0_5=float(f1_score(y, prob >= .5, average="micro", zero_division=0)),
                   wall_seconds=time.monotonic()-begin)
    save_json(folder / "test_metrics.json", metrics)
    save_json(folder / "completed.json", {"protocol_sha256": digest, "fold": fold+1, "model": key})


def selected_models(args):
    return list(MODELS) if args.models is None else args.models


def worker(args):
    import torch
    torch.set_num_threads(4)
    torch.backends.mha.set_fastpath_enabled(False)
    configure_adapters()
    current = protocol()
    digest = current["protocol_sha256"]
    assert json.loads((args.output_dir / "protocol.json").read_text())["protocol_sha256"] == digest
    rows, folds, y = read_inputs()
    ids = np.arange(680, dtype=np.int64)
    bags, tokens, masks = None, None, None
    params = parameters()
    jobs = [(key, fold) for key in selected_models(args) for fold in range(5)]
    for index, (key, fold) in enumerate(jobs):
        if index % len(args.devices) != args.worker_index:
            continue
        marker = args.output_dir / f"fold_{fold+1}" / key / "completed.json"
        if marker.exists():
            assert json.loads(marker.read_text())["protocol_sha256"] == digest
            continue
        try:
            if key in TEXT_MODELS:
                text_fold(args, key, fold, rows, folds, digest)
            else:
                if bags is None:
                    bags = load_features(rows)
                    texts = {r["case_index"]: r["findings_masked"] for r in rows}
                    tokens, masks = fusion_base.encode_descriptions(texts, ids)
                fusion_base.train_fold(args, key, fold, ids, bags, y, folds, tokens, masks, params[key], digest)
            gc.collect()
            torch.cuda.empty_cache()
        except Exception:
            save_json(args.output_dir / "errors" / f"{key}_fold_{fold+1}.json", {"traceback": traceback.format_exc()})
            raise


def aggregate(args, digest):
    from sklearn.metrics import f1_score, accuracy_score
    rows, _, targets = read_inputs()
    summaries, completed = [], 0
    for key in selected_models(args):
        metrics, predictions = [], []
        for fold in range(1, 6):
            folder = args.output_dir / f"fold_{fold}" / key
            if not (folder / "completed.json").exists():
                continue
            metric = json.loads((folder / "test_metrics.json").read_text())
            assert metric["protocol_sha256"] == digest
            metrics.append(metric)
            with (folder / "test_predictions.csv").open(encoding="utf-8-sig", newline="") as stream:
                predictions.extend({**r, "fold": fold} for r in csv.DictReader(stream))
        completed += len(metrics)
        if len(metrics) != 5:
            continue
        predictions.sort(key=lambda r: int(r["patient_id"]))
        assert [int(r["patient_id"]) for r in predictions] == list(range(680))
        y = np.asarray([[int(r[f"true_{n}"]) for n in LABELS] for r in predictions])
        p = np.asarray([[float(r[f"prob_{n}"]) for n in LABELS] for r in predictions])
        pred = np.asarray([[int(r[f"pred_{n}"]) for n in LABELS] for r in predictions])
        assert np.array_equal(y, targets) and np.isfinite(p).all() and ((p >= 0) & (p <= 1)).all()
        for fold, metric in enumerate(metrics, 1):
            which = np.asarray([r["fold"] == fold for r in predictions])
            thresholds = np.asarray([metric["thresholds"][n] for n in LABELS])
            assert np.array_equal(pred[which], p[which] >= thresholds)
            assert np.isclose(f1_score(y[which], pred[which], average="macro", zero_division=0), metric["macro_f1"])
        co = y.sum(1) >= 2
        summary = {"model_key": key, "model": MODELS[key], "category": category(key), "completed_folds": 5,
                   "macro_f1_mean": statistics.mean(m["macro_f1"] for m in metrics),
                   "macro_f1_std": statistics.stdev(m["macro_f1"] for m in metrics),
                   "macro_f1_fixed_0_5_mean": statistics.mean(m["macro_f1_fixed_0_5"] for m in metrics),
                   "macro_f1_fixed_0_5_std": statistics.stdev(m["macro_f1_fixed_0_5"] for m in metrics),
                   "oof_macro_f1": float(f1_score(y, pred, average="macro", zero_division=0)),
                   "oof_micro_f1": float(f1_score(y, pred, average="micro", zero_division=0)),
                   "oof_exact_match": float(accuracy_score(y, pred)),
                   "oof_per_label_f1": dict(zip(LABELS, f1_score(y, pred, average=None, zero_division=0).tolist())),
                   "copositive_cases": int(co.sum()),
                   "copositive_macro_f1": float(f1_score(y[co], pred[co], average="macro", zero_division=0)),
                   "copositive_exact_match": float(accuracy_score(y[co], pred[co])), "folds": metrics}
        summaries.append(summary)
        for row in predictions:
            row["source_patient_id"] = rows[int(row["patient_id"])]["patient_id"]
        with (args.output_dir / f"{key}_oof_predictions.csv").open("w", encoding="utf-8-sig", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(predictions[0]))
            writer.writeheader(); writer.writerows(predictions)
    save_json(args.output_dir / "summary.json", summaries)
    fields = ["category", "model", "macro_f1_mean", "macro_f1_std", "macro_f1_fixed_0_5_mean", "macro_f1_fixed_0_5_std",
              "oof_macro_f1", "oof_micro_f1", "copositive_macro_f1", "copositive_exact_match"]
    with (args.output_dir / "summary.csv").open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields, extrasaction="ignore")
        writer.writeheader(); writer.writerows(summaries)
    report = "# CT-RATE680例五折三标签分类\n\n标签：肺气肿、肺不张、肺纤维化后遗改变。所有方法共用患者划分和掩码所见；所有图像方法共用冻结ConvNeXt特征。\n\n| 类别 | 模型 | 五折Macro-F1 | 固定0.5阈值Macro-F1 |\n|---|---|---:|---:|\n"
    for r in summaries:
        report += f"| {r['category']} | {r['model']} | {r['macro_f1_mean']:.4f} ± {r['macro_f1_std']:.4f} | {r['macro_f1_fixed_0_5_mean']:.4f} ± {r['macro_f1_fixed_0_5_std']:.4f} |\n"
    report += "\n仅完整五折进入汇总；标准差采用样本标准差。主阈值仅来自验证集。文本保留原文本基线优化/早停协议，图像与多模态保留原缓存实验的30轮ASL训练。此结果属于新任务重训练评价；额外AMEF图像分支为单独训练，未运行教师蒸馏。\n"
    (args.output_dir / "results.md").write_text(report, encoding="utf-8")
    save_json(args.output_dir / "progress.json", {"completed_fold_jobs": completed,
              "expected_fold_jobs": 5*len(selected_models(args)), "completed_models": len(summaries)})
    return completed


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=ROOT / "outputs/ct_rate_680/all_models_fivefold")
    parser.add_argument("--devices", type=int, nargs="+", default=[0, 1, 2, 3])
    parser.add_argument("--models", nargs="+", choices=list(MODELS))
    parser.add_argument("--worker-index", type=int)
    parser.add_argument("--audit-only", action="store_true")
    args = parser.parse_args()
    if args.worker_index is not None:
        worker(args)
        return
    args.output_dir.mkdir(parents=True, exist_ok=True)
    with (args.output_dir / "run.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        read_inputs()
        current = protocol()
        saved = args.output_dir / "protocol.json"
        if saved.exists() and json.loads(saved.read_text())["protocol_sha256"] != current["protocol_sha256"]:
            raise ValueError("协议或源码改变，禁止与已有正式结果混用")
        save_json(saved, current)
        if args.audit_only:
            print(f"协议核对通过：{len(selected_models(args))}个模型，{5*len(selected_models(args))}个折次。", flush=True)
            return
        handles, workers, started = [], [], time.time()
        for index, device in enumerate(args.devices):
            env = os.environ.copy()
            env.update(CUDA_VISIBLE_DEVICES=str(device), OMP_NUM_THREADS="4", TOKENIZERS_PARALLELISM="false")
            cmd = [sys.executable, "-u", str(Path(__file__).resolve()), "--worker-index", str(index),
                   "--devices", *map(str, args.devices), "--output-dir", str(args.output_dir)]
            if args.models:
                cmd += ["--models", *args.models]
            handle = (args.output_dir / f"worker_{index}.log").open("a")
            handles.append(handle)
            workers.append(subprocess.Popen(cmd, env=env, stdout=handle, stderr=subprocess.STDOUT))
        state = {"status": "running", "supervisor_pid": os.getpid(), "worker_pids": [p.pid for p in workers],
                 "started_unix": started, "models": selected_models(args)}
        save_json(args.output_dir / "run_state.json", state)
        while any(p.poll() is None for p in workers):
            completed = aggregate(args, current["protocol_sha256"])
            print(f"训练完成：{completed}/{5*len(selected_models(args))}个模型折次", flush=True)
            time.sleep(15)
        completed = aggregate(args, current["protocol_sha256"])
        codes = [p.returncode for p in workers]
        state.update(status="complete" if completed == 5*len(selected_models(args)) and not any(codes) else "incomplete",
                     completed_fold_jobs=completed, worker_exit_codes=codes,
                     finished_unix=time.time(), wall_seconds=time.time()-started)
        save_json(args.output_dir / "run_state.json", state)
        for handle in handles:
            handle.close()
        if state["status"] != "complete":
            raise SystemExit(1)


if __name__ == "__main__":
    main()
