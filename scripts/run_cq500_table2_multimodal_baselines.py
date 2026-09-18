#!/usr/bin/env python3
"""在CQ500的491例图像缓存与AI影像描述上运行表2四个多模态基线。"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
from pathlib import Path
import statistics
import subprocess
import sys
import time
import traceback

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
from run_cq500_table2_text_baselines import LABEL_NAMES, load_inputs, save_json

MODELS = {
    "task2_mmfnet_2024": "MMFNet",
    "task2_radfuse_2025": "RadFuse",
    "task2_saif_2025": "SAIF",
    "task2_mmtf_2025": "MMTF",
}
SETTINGS = {
    "epochs": 30, "batch_size": 16, "learning_rate": 2e-4,
    "weight_decay": 0.02, "warmup_ratio": 0.2, "max_instances": 28,
    "instance_dropout": 0.25, "base_seed": 42,
    "text_max_length": 128, "text_vocab_size": 8192,
    "threshold_grid": [round(0.1 + 0.05 * i, 2) for i in range(17)],
}


def model_parameters(args):
    import yaml
    config = yaml.safe_load(args.config.read_text())
    return {key: config["models"][key] for key in MODELS}


def protocol(args):
    tracked = {
        "descriptions": args.text_csv, "labels": args.reads, "folds": args.folds_json,
        "feature_cache": args.feature_cache, "model_config": args.config,
        "model_implementation": ROOT / "src/sotas/task2/multimodal_sotas.py",
        "text_encoder": ROOT / "src/exp_8/models.py",
        "base_model": ROOT / "src/exp_4/models.py",
        "instance_encoder": ROOT / "src/model/gastro_label_graph_mil/modules.py",
        "tokenizer": ROOT / "src/training/data.py",
        "loss": ROOT / "src/training/losses.py",
        "image_adapter": ROOT / "src/scripts/run_physionet_ct_ich_table2_baselines.py",
        "image_loader": ROOT / "src/scripts/run_cq500_table2_baselines.py",
        "input_validator": ROOT / "src/scripts/run_cq500_table2_text_baselines.py",
        "runner": Path(__file__),
    }
    value = {
        "models": MODELS, "labels": LABEL_NAMES, "settings": SETTINGS,
        "model_parameters": model_parameters(args),
        "source_paths": {k: str(v) for k, v in tracked.items()},
        "source_sha256": {k: hashlib.sha256(v.read_bytes()).hexdigest() for k, v in tracked.items()},
        "split": "复用CQ500图像实验；test=k，validation=(k+1)%5，其他三折训练",
        "image_input": "已缓存的冻结ImageNet ConvNeXt-Tiny特征；原轴位序列最多28层",
        "text_input": "uniform64图像派生AI试稿；原文不改写、不额外遮蔽；沿用watch哈希编码接口",
        "model_scope": "复用表2任务适配融合实现与TextCNN；仅以原CQ500缓存适配器替换视觉backbone",
        "training": "沿用CQ500图像基线训练配置；ASL及各模型原辅助损失；不重采样病例",
        "checkpoint_selection": "最低验证集分类损失；测试集仅在选定权重后评分",
        "threshold_selection": "选定权重后仅在验证集逐标签调阈值；另存固定0.5结果",
        "primary_metric": "五折测试Macro-F1均值和样本标准差；验证集调阈值",
        "note": "文本由图像生成，属于图像派生辅助文本实验，并非独立采集临床报告。",
    }
    value["protocol_sha256"] = hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False).encode()).hexdigest()
    return value


def build_model(key, parameters):
    from sotas.task2.multimodal_sotas import build_task2_multimodal_sota
    from run_physionet_ct_ich_table2_baselines import CachedFeatureBackbone
    parameters = dict(parameters)
    auxiliary = {k.removesuffix("_weight"): parameters.pop(k)
                 for k in list(parameters) if k.endswith("_weight")}
    model = build_task2_multimodal_sota(key, **parameters, pretrained=False, num_labels=3)
    model.instance_encoder.backbone = CachedFeatureBackbone(
        input_dim=768, output_dim=parameters["feature_dim"], dropout=parameters["dropout"])
    return model, auxiliary


def encode_descriptions(texts, patient_ids):
    import torch
    from training.data import _encode_text_fields
    rows = [_encode_text_fields({"watch": texts[int(i)]}, ("watch",),
                               max_length=SETTINGS["text_max_length"],
                               vocab_size=SETTINGS["text_vocab_size"]) for i in patient_ids]
    ids, masks = (torch.stack([r[j] for r in rows]) for j in (0, 1))
    assert masks.any(dim=1).all()
    return ids, masks


def loader(bags, labels, indices, training, seed):
    import numpy as np
    import torch
    from torch.utils.data import DataLoader
    from run_physionet_ct_ich_table2_baselines import FeatureBagDataset, collate_bags
    dataset = FeatureBagDataset(bags, labels, np.asarray(indices), SETTINGS["max_instances"],
                                SETTINGS["instance_dropout"] if training else 0, training)
    return DataLoader(dataset, batch_size=SETTINGS["batch_size"], shuffle=training,
                      num_workers=0, collate_fn=collate_bags,
                      generator=torch.Generator().manual_seed(seed))


def forward_batch(model, batch, text_ids, text_mask, device, training=False):
    features, image_mask, targets, indices = batch
    kwargs = {"images": features.to(device), "mask": image_mask.to(device),
              "watch_token_ids": text_ids[indices].to(device),
              "watch_token_mask": text_mask[indices].to(device)}
    if training:
        kwargs["labels"] = targets.to(device)
    return model(**kwargs)


def evaluate(model, data_loader, text_ids, text_mask, device):
    import numpy as np
    import torch
    from training.losses import AsymmetricLossMultiLabel
    model.eval()
    targets, probabilities, indices = [], [], []
    total_loss = 0.0
    with torch.inference_mode():
        for batch in data_loader:
            output = forward_batch(model, batch, text_ids, text_mask, device)
            total_loss += float(AsymmetricLossMultiLabel()(output["logits"], batch[2].to(device))) * len(batch[2])
            targets.append(batch[2].numpy().astype(np.int64))
            probabilities.append(torch.sigmoid(output["logits"]).cpu().numpy())
            indices.append(batch[3].numpy())
    y, p, idx = map(np.concatenate, (targets, probabilities, indices))
    assert np.isfinite(p).all() and ((p >= 0) & (p <= 1)).all()
    return total_loss / len(y), y, p, idx


def write_predictions(path, patient_ids, y, probabilities, thresholds):
    fields = ["patient_id"]
    for label in LABEL_NAMES:
        fields.extend([f"true_{label}", f"prob_{label}", f"pred_{label}", f"pred_0.5_{label}"])
    with path.open("w", encoding="utf-8-sig", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fields)
        writer.writeheader()
        for patient, targets, probs in zip(patient_ids, y, probabilities):
            row = {"patient_id": int(patient)}
            for j, label in enumerate(LABEL_NAMES):
                row.update({f"true_{label}": int(targets[j]), f"prob_{label}": float(probs[j]),
                            f"pred_{label}": int(probs[j] >= thresholds[j]),
                            f"pred_0.5_{label}": int(probs[j] >= 0.5)})
            writer.writerow(row)


def train_fold(args, key, fold, patient_ids, bags, targets, folds, text_ids, text_mask, parameters, digest):
    import numpy as np
    import torch
    from sklearn.metrics import f1_score
    from tqdm import tqdm
    from training.losses import AsymmetricLossMultiLabel
    from exp_10.train_text_classification import tune_thresholds, calculate_metrics
    from run_physionet_ct_ich_table2_baselines import seed_everything

    start = time.monotonic()
    folder = args.output_dir / f"fold_{fold+1}" / key
    seed = SETTINGS["base_seed"] + 100 * (fold + 1)
    seed_everything(seed)
    validation_fold = (fold + 1) % 5
    split_ids = {
        "train": [i for j, group in enumerate(folds["folds"]) if j not in (fold, validation_fold) for i in group],
        "val": folds["folds"][validation_fold], "test": folds["folds"][fold],
    }
    mapping = {int(patient): j for j, patient in enumerate(patient_ids)}
    splits = {name: [mapping[i] for i in ids] for name, ids in split_ids.items()}
    assert set(split_ids["train"]).isdisjoint(split_ids["val"] + split_ids["test"])
    assert set(split_ids["val"]).isdisjoint(split_ids["test"])
    save_json(folder / "split_ids.json", split_ids)
    save_json(folder / "config.json", {"settings": SETTINGS, "model": parameters, "seed": seed, "protocol_sha256": digest})
    loaders = {name: loader(bags, targets, indices, name == "train", seed) for name, indices in splits.items()}
    device = torch.device("cuda:0")
    model, aux_weights = build_model(key, parameters)
    model.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=SETTINGS["learning_rate"], weight_decay=SETTINGS["weight_decay"])
    total_steps = len(loaders["train"]) * SETTINGS["epochs"]
    warmup = int(total_steps * SETTINGS["warmup_ratio"])

    def lr_factor(step):
        if step < warmup:
            return (step + 1) / max(1, warmup)
        return 0.5 * (1 + math.cos(math.pi * (step - warmup) / max(1, total_steps - warmup)))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_factor)
    scaler = torch.amp.GradScaler("cuda")
    criterion = AsymmetricLossMultiLabel()
    best_loss, best_state, best_epoch, history = math.inf, None, 0, []
    print(f"开始{MODELS[key]}折{fold+1}，训练/验证/测试={list(map(len,split_ids.values()))}", flush=True)
    for epoch in tqdm(range(1, SETTINGS["epochs"]+1), desc=f"{MODELS[key]} 折{fold+1}"):
        model.train()
        losses, auxiliary_totals = [], {k: [] for k in aux_weights}
        for batch in loaders["train"]:
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device_type="cuda"):
                output = forward_batch(model, batch, text_ids, text_mask, device, training=True)
                loss = criterion(output["logits"], batch[2].to(device))
                for name, weight in aux_weights.items():
                    aux = output["aux_losses"][name]
                    loss = loss + float(weight) * aux
                    auxiliary_totals[name].append(float(aux.detach()))
            if not torch.isfinite(loss):
                raise RuntimeError(f"{key}折{fold+1}轮{epoch}损失非有限数")
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()
            losses.append(float(loss.detach()))
        val_loss, _, _, _ = evaluate(model, loaders["val"], text_ids, text_mask, device)
        history.append({"epoch": epoch, "train_loss": statistics.mean(losses), "val_loss": val_loss,
                        "aux_losses": {k: statistics.mean(v) for k,v in auxiliary_totals.items()}})
        if val_loss < best_loss:
            best_loss, best_epoch = val_loss, epoch
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        save_json(folder / "history.json", history)
    assert best_state is not None
    model.load_state_dict(best_state)
    _, val_y, val_p, val_indices = evaluate(model, loaders["val"], text_ids, text_mask, device)
    thresholds = tune_thresholds(val_y, val_p, SETTINGS["threshold_grid"])
    _, test_y, test_p, test_indices = evaluate(model, loaders["test"], text_ids, text_mask, device)
    assert test_indices.tolist() == splits["test"] and val_indices.tolist() == splits["val"]
    metrics = calculate_metrics(test_y, test_p, thresholds, LABEL_NAMES)
    metrics.update({"model": MODELS[key], "model_key": key, "fold": fold+1,
                    "best_epoch": best_epoch, "best_val_loss": best_loss, "seed": seed,
                    "thresholds": dict(zip(LABEL_NAMES, thresholds.tolist())),
                    "macro_f1_fixed_0_5": float(f1_score(test_y, test_p >= 0.5, average="macro", zero_division=0)),
                    "micro_f1_fixed_0_5": float(f1_score(test_y, test_p >= 0.5, average="micro", zero_division=0)),
                    "protocol_sha256": digest, "split_sizes": {k: len(v) for k,v in splits.items()},
                    "wall_seconds": time.monotonic()-start, "auxiliary_loss_weights": aux_weights})
    write_predictions(folder / "validation_predictions.csv", patient_ids[val_indices], val_y, val_p, thresholds)
    write_predictions(folder / "test_predictions.csv", patient_ids[test_indices], test_y, test_p, thresholds)
    torch.save({"state_dict": best_state, "model_key": key, "model_parameters": parameters,
                "epoch": best_epoch, "thresholds": thresholds.tolist(), "protocol_sha256": digest}, folder / "best_model.pt")
    save_json(folder / "test_metrics.json", metrics)
    save_json(folder / "completed.json", {"protocol_sha256": digest, "fold": fold+1, "model": key})
    print(f"完成{MODELS[key]}折{fold+1}：Macro-F1={metrics['macro_f1']:.4f}；固定0.5={metrics['macro_f1_fixed_0_5']:.4f}", flush=True)


def worker(args):
    import numpy as np
    import torch
    from run_cq500_table2_baselines import load_bags
    torch.set_num_threads(4)
    torch.backends.mha.set_fastpath_enabled(False)
    current = protocol(args)
    assert current["protocol_sha256"] == json.loads((args.output_dir / "protocol.json").read_text())["protocol_sha256"]
    texts, labels, folds = load_inputs(args)
    patient_ids, bags, targets = load_bags(args.reads, args.feature_cache)
    assert patient_ids.tolist() == sorted(texts)
    assert np.array_equal(targets, [labels[int(i)] for i in patient_ids])
    assert all(len(b) and b.shape[1] == 768 and np.isfinite(b).all() for b in bags)
    text_ids, text_mask = encode_descriptions(texts, patient_ids)
    parameters = model_parameters(args)
    for index, (key, fold) in enumerate((k,f) for k in MODELS for f in range(5)):
        if index % len(args.devices) != args.worker_index:
            continue
        marker = args.output_dir / f"fold_{fold+1}" / key / "completed.json"
        if marker.exists():
            assert json.loads(marker.read_text())["protocol_sha256"] == current["protocol_sha256"]
            continue
        try:
            train_fold(args, key, fold, patient_ids, bags, targets, folds, text_ids, text_mask,
                       parameters[key], current["protocol_sha256"])
            torch.cuda.empty_cache()
        except Exception as exc:
            save_json(args.output_dir / "errors" / f"{key}_fold_{fold+1}.json",
                      {"error": str(exc), "traceback": traceback.format_exc()})
            raise


def aggregate(args, digest):
    from sklearn.metrics import f1_score
    summaries, completed = [], 0
    for key, display in MODELS.items():
        metrics, predictions = [], []
        for fold in range(1, 6):
            folder = args.output_dir / f"fold_{fold}" / key
            if not (folder / "completed.json").exists():
                continue
            item = json.loads((folder / "test_metrics.json").read_text())
            assert item["protocol_sha256"] == digest
            metrics.append(item)
            with (folder / "test_predictions.csv").open(encoding="utf-8-sig", newline="") as file:
                predictions.extend({**r, "fold": fold} for r in csv.DictReader(file))
        completed += len(metrics)
        if len(metrics) != 5:
            continue
        predictions.sort(key=lambda r: int(r["patient_id"]))
        assert [int(r["patient_id"]) for r in predictions] == list(range(491))
        y = [[int(r[f"true_{n}"]) for n in LABEL_NAMES] for r in predictions]
        pred = [[int(r[f"pred_{n}"]) for n in LABEL_NAMES] for r in predictions]
        summary = {
            "model_key": key, "model": display, "completed_folds": 5, "oof_cases": 491,
            "macro_f1_mean": statistics.mean(r["macro_f1"] for r in metrics),
            "macro_f1_std": statistics.stdev(r["macro_f1"] for r in metrics),
            "micro_f1_mean": statistics.mean(r["micro_f1"] for r in metrics),
            "oof_macro_f1": float(f1_score(y, pred, average="macro", zero_division=0)),
            "oof_per_label_f1": dict(zip(LABEL_NAMES, f1_score(y, pred, average=None, zero_division=0).tolist())),
            "macro_f1_fixed_0_5_mean": statistics.mean(r["macro_f1_fixed_0_5"] for r in metrics),
            "macro_f1_fixed_0_5_std": statistics.stdev(r["macro_f1_fixed_0_5"] for r in metrics),
            "folds": metrics,
        }
        summaries.append(summary)
        with (args.output_dir / f"{key}_oof_predictions.csv").open("w", encoding="utf-8-sig", newline="") as file:
            writer = csv.DictWriter(file, fieldnames=list(predictions[0]))
            writer.writeheader(); writer.writerows(predictions)
    save_json(args.output_dir / "summary.json", summaries)
    with (args.output_dir / "summary.csv").open("w", encoding="utf-8-sig", newline="") as file:
        fields = ["model", "macro_f1_mean", "macro_f1_std", "micro_f1_mean", "oof_macro_f1", "macro_f1_fixed_0_5_mean", "macro_f1_fixed_0_5_std"]
        writer = csv.DictWriter(file, fieldnames=fields, extrasaction="ignore")
        writer.writeheader(); writer.writerows(summaries)
    report = "# CQ500四个多模态基线\n\n491例、IPH/Mass Effect/Midline Shift三标签；沿用既有患者五折划分。图像使用已有冻结特征，文本使用uniform64图像派生AI试稿。\n\n| 模型 | 验证集调阈值Macro-F1 | 固定0.5阈值Macro-F1 |\n|---|---:|---:|\n"
    report += "\n".join(f"| {r['model']} | {r['macro_f1_mean']:.4f} ± {r['macro_f1_std']:.4f} | {r['macro_f1_fixed_0_5_mean']:.4f} ± {r['macro_f1_fixed_0_5_std']:.4f} |" for r in summaries)
    report += "\n\n表中为五折均值±样本标准差。与刚完成的文本基线对应时看调阈值列；与先前固定阈值的图像基线对应时看固定0.5列。仅阈值口径和病例划分一致不代表训练配置完全相同，详见protocol.json。文本并非独立采集临床报告。\n"
    (args.output_dir / "results.md").write_text(report, encoding="utf-8")
    save_json(args.output_dir / "progress.json", {"completed_fold_jobs": completed, "expected_fold_jobs": 20, "completed_models": len(summaries)})
    return completed


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--text-csv", type=Path, default=ROOT / "outputs/cq500/image_descriptions/uniform64/descriptions_draft.csv")
    parser.add_argument("--reads", type=Path, default=ROOT / "datasets/cq500/raw/reads.csv")
    parser.add_argument("--folds-json", type=Path, default=ROOT / "outputs/cq500/table2_image_baselines/patient_folds.json")
    parser.add_argument("--feature-cache", type=Path, default=ROOT / "outputs/cq500/convnext_tiny_scan_features.npz")
    parser.add_argument("--config", type=Path, default=ROOT / "src/configs/task2/model.yaml")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "outputs/cq500/table2_multimodal_baselines_uniform64")
    parser.add_argument("--devices", type=int, nargs="+", default=[0, 1, 2, 3])
    parser.add_argument("--worker-index", type=int)
    parser.add_argument("--audit-only", action="store_true")
    args = parser.parse_args()
    if args.worker_index is not None:
        worker(args); return
    texts, _, folds = load_inputs(args)
    current = protocol(args)
    saved = args.output_dir / "protocol.json"
    if saved.exists() and json.loads(saved.read_text())["protocol_sha256"] != current["protocol_sha256"]:
        raise ValueError("输出目录已有其他协议结果，请另选目录")
    save_json(saved, current)
    save_json(args.output_dir / "data_audit.json", {"cases": len(texts), "unique_texts": len(set(texts.values())),
              "positive_counts": folds["positive_counts"], "fold_sizes": [len(f) for f in folds["folds"]],
              "text_modified": False, "folds_modified": False})
    if args.audit_only:
        print("491例文本、标签及五折对应关系核对通过", flush=True); return
    handles, workers, started = [], [], time.time()
    for index, device in enumerate(args.devices):
        env = os.environ.copy()
        env.update({"CUDA_VISIBLE_DEVICES": str(device), "OMP_NUM_THREADS": "4", "TOKENIZERS_PARALLELISM": "false"})
        command = [sys.executable, "-u", str(Path(__file__).resolve()), "--worker-index", str(index)]
        for name in ("text_csv", "reads", "folds_json", "feature_cache", "config", "output_dir"):
            command.extend(["--" + name.replace("_", "-"), str(getattr(args, name))])
        command.extend(["--devices", *map(str, args.devices)])
        handle = (args.output_dir / f"worker_{index}.log").open("a")
        handles.append(handle)
        workers.append(subprocess.Popen(command, env=env, stdout=handle, stderr=subprocess.STDOUT))
    state = {"status": "running", "supervisor_pid": os.getpid(), "worker_pids": [p.pid for p in workers], "started_unix": started}
    save_json(args.output_dir / "run_state.json", state)
    while any(p.poll() is None for p in workers):
        completed = aggregate(args, current["protocol_sha256"])
        print(f"已完成{completed}/20个模型折次", flush=True)
        time.sleep(15)
    completed = aggregate(args, current["protocol_sha256"])
    codes = [p.returncode for p in workers]
    state.update({"status": "complete" if completed == 20 and not any(codes) else "incomplete",
                  "completed_fold_jobs": completed, "worker_exit_codes": codes,
                  "finished_unix": time.time(), "wall_seconds": time.time()-started})
    save_json(args.output_dir / "run_state.json", state)
    for handle in handles:
        handle.close()
    print(f"四模型五折实验结束：{state['status']}", flush=True)
    if state["status"] != "complete":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
