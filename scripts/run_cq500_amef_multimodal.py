#!/usr/bin/env python3
"""在CQ500原五折上补跑AMEF-MIL的主要多模态预测路径。"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time

import run_cq500_table2_multimodal_baselines as base

ROOT = base.ROOT
MODEL_KEY = "amef_multimodal"
MODELS = {MODEL_KEY: "AMEF-MIL"}
_baseline_protocol = base.protocol
_baseline_aggregate = base.aggregate


def model_parameters(args):
    import yaml
    from task3_apro_cope_ablation_scheduler import base_model_params
    parameters = base_model_params("apro_full")
    main_config = yaml.safe_load(args.config.read_text())
    parameters["label_query_consistency_weight"] = float(
        main_config["model"]["params"]["label_query_consistency_weight"])
    assert parameters["image_aux_weight"] == 0.0
    return {MODEL_KEY: parameters}


def build_model(key, parameters):
    import torch
    from exp_8.models import Exp12AProCoPEWatchCrossAttentionTextCNNModel
    from run_physionet_ct_ich_table2_baselines import CachedFeatureBackbone
    assert key == MODEL_KEY
    torch.backends.mha.set_fastpath_enabled(False)
    parameters = dict(parameters)
    auxiliary = {k.removesuffix("_weight"): parameters.pop(k)
                 for k in list(parameters) if k.endswith("_weight")}
    model = Exp12AProCoPEWatchCrossAttentionTextCNNModel(**parameters, pretrained=False, num_labels=3)
    model.instance_encoder.backbone = CachedFeatureBackbone(
        input_dim=768, output_dim=parameters["feature_dim"], dropout=parameters["dropout"])
    return model, auxiliary


def loader(bags, labels, indices, training, seed):
    import numpy as np
    import torch
    from torch.utils.data import DataLoader, Dataset
    from run_physionet_ct_ich_table2_baselines import collate_bags

    class OrderedFeatureBags(Dataset):
        def __len__(self):
            return len(indices)

        def __getitem__(self, item):
            index = int(indices[item])
            bag = bags[index]
            original_count = len(bag)
            selected = np.arange(original_count)
            maximum = base.SETTINGS["max_instances"]
            if len(selected) > maximum:
                if training:
                    selected = np.sort(np.random.choice(len(bag), maximum, replace=False))
                else:
                    selected = np.linspace(0, len(bag)-1, maximum).round().astype(np.int64)
            dropout = base.SETTINGS["instance_dropout"]
            if training and dropout > 0 and len(selected) > 1:
                keep = max(1, int(round(len(selected) * (1-dropout))))
                chosen = np.sort(np.random.choice(len(selected), keep, replace=False))
                selected = selected[chosen]
            return (torch.from_numpy(bag[selected]), torch.tensor(labels[index], dtype=torch.float32),
                    index, torch.from_numpy(selected), original_count)

    def collate(records):
        features, mask, targets, item_ids = collate_bags([r[:3] for r in records])
        reference_indices = torch.full(mask.shape, -1, dtype=torch.long)
        for j, record in enumerate(records):
            reference_indices[j, :len(record[3])] = record[3]
        counts = torch.tensor([r[4] for r in records], dtype=torch.long)
        return features, mask, targets, item_ids, reference_indices, counts

    return DataLoader(OrderedFeatureBags(), batch_size=base.SETTINGS["batch_size"], shuffle=training,
                      num_workers=0, collate_fn=collate,
                      generator=torch.Generator().manual_seed(seed))


def forward_batch(model, batch, text_ids, text_mask, device, training=False):
    import torch
    features, mask, targets, indices, reference_indices, counts = batch
    kwargs = {
        "images": features.to(device), "mask": mask.to(device),
        "watch_token_ids": text_ids[indices].to(device),
        "watch_token_mask": text_mask[indices].to(device),
        "instance_indices": reference_indices.to(device), "original_image_counts": counts.to(device),
    }
    if training:
        kwargs["labels"] = targets.to(device)
    # 沿用CQ500纯图像AMEF路径的FP32计算，保护APro-CoPE小间隔归一化。
    with torch.autocast(device_type=torch.device(device).type, enabled=False):
        return model(**kwargs)


def protocol(args):
    value = _baseline_protocol(args)
    value.pop("protocol_sha256")
    extras = {
        "amef_runner": Path(__file__),
        "apro_model_config_builder": ROOT / "src/scripts/task3_apro_cope_ablation_scheduler.py",
        "comparison_protocol": ROOT / "outputs/cq500/table2_multimodal_baselines_uniform64/protocol.json",
    }
    for name, path in extras.items():
        value["source_paths"][name] = str(path)
        value["source_sha256"][name] = hashlib.sha256(path.read_bytes()).hexdigest()
    value.update({
        "model_scope": "AMEF-MIL主要多模态路径：完整APro-CoPE、标签注意力、标签超图、TextCNN、标签查询及门控残差融合",
        "training": "ASL多模态分类损失+0.01标签查询判别损失；本次不运行面向纯图像输出的另行蒸馏实验",
        "prediction_output": "logits为图文融合输出；不是image_only_logits",
        "precision": "AMEF前向FP32，沿用此前CQ500图像AMEF的数值精度；四个多模态基线采用AMP",
        "position_reference": "缓存的有序轴位序列为CT参考序列，保留实例丢弃前的序列索引和长度；不是全量DICOM绝对索引",
    })
    comparison = json.loads(extras["comparison_protocol"].read_text())
    assert value["settings"] == comparison["settings"]
    for name in ("descriptions", "labels", "folds", "feature_cache"):
        assert value["source_sha256"][name] == comparison["source_sha256"][name]
    value["protocol_sha256"] = hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False).encode()).hexdigest()
    return value


def aggregate(args, digest):
    completed = _baseline_aggregate(args, digest)
    summary = json.loads((args.output_dir / "summary.json").read_text())
    base.save_json(args.output_dir / "progress.json", {
        "completed_fold_jobs": completed, "expected_fold_jobs": 5, "completed_models": len(summary)})
    report = "# CQ500 AMEF-MIL完整多模态实验\n\n491例CT图像缓存＋uniform64生成文本，IPH、Mass Effect、Midline Shift三标签；沿用先前五折划分。统计图文融合logits的预测。\n\n| 模型 | 验证集调阈值Macro-F1 | 固定0.5阈值Macro-F1 |\n|---|---:|---:|\n"
    comparison_path = ROOT / "outputs/cq500/table2_multimodal_baselines_uniform64/summary.json"
    comparisons = json.loads(comparison_path.read_text())
    for row in comparisons + summary:
        report += f"| {row['model']} | {row['macro_f1_mean']:.4f} ± {row['macro_f1_std']:.4f} | {row['macro_f1_fixed_0_5_mean']:.4f} ± {row['macro_f1_fixed_0_5_std']:.4f} |\n"
    report += "\n均值±样本标准差。模型选择及阈值仅使用验证集；每例在测试中恰好出现一次。图像为冻结特征，文本为图像派生AI描述。AMEF保留缓存序列在增强前的位置，以FP32计算；其余共同训练配置与四个基线一致。论文及输入数据未修改。\n"
    (args.output_dir / "results.md").write_text(report, encoding="utf-8")
    return completed


def configure_base():
    """仅在本进程注入适配器；不改动此前四基线的源码或结果。"""
    base.MODELS = MODELS
    base.model_parameters = model_parameters
    base.build_model = build_model
    base.loader = loader
    base.forward_batch = forward_batch
    base.protocol = protocol


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--text-csv", type=Path, default=ROOT / "outputs/cq500/image_descriptions/uniform64/descriptions_draft.csv")
    parser.add_argument("--reads", type=Path, default=ROOT / "datasets/cq500/raw/reads.csv")
    parser.add_argument("--folds-json", type=Path, default=ROOT / "outputs/cq500/table2_image_baselines/patient_folds.json")
    parser.add_argument("--feature-cache", type=Path, default=ROOT / "outputs/cq500/convnext_tiny_scan_features.npz")
    parser.add_argument("--config", type=Path, default=ROOT / "src/configs/task3/t3_main_model.yaml")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "outputs/cq500/amef_multimodal_uniform64")
    parser.add_argument("--devices", type=int, nargs="+", default=[0, 1, 2, 3])
    parser.add_argument("--worker-index", type=int)
    parser.add_argument("--audit-only", action="store_true")
    return parser.parse_args()


def main():
    args = parse_args()
    configure_base()
    if args.worker_index is not None:
        base.worker(args)
        return
    texts, _, folds = base.load_inputs(args)
    current = protocol(args)
    saved = args.output_dir / "protocol.json"
    if saved.exists() and json.loads(saved.read_text())["protocol_sha256"] != current["protocol_sha256"]:
        raise ValueError("输出目录存在其他协议，禁止混用结果")
    base.save_json(saved, current)
    base.save_json(args.output_dir / "data_audit.json", {
        "cases": len(texts), "positive_counts": folds["positive_counts"],
        "fold_sizes": [len(f) for f in folds["folds"]], "text_modified": False, "folds_modified": False})
    if args.audit_only:
        print("491例输入与五折划分核对通过", flush=True)
        return
    started, handles, workers = time.time(), [], []
    for index, device in enumerate(args.devices):
        env = os.environ.copy()
        env.update({"CUDA_VISIBLE_DEVICES": str(device), "OMP_NUM_THREADS": "4"})
        command = [sys.executable, "-u", str(Path(__file__).resolve()), "--worker-index", str(index)]
        for name in ("text_csv", "reads", "folds_json", "feature_cache", "config", "output_dir"):
            command.extend(["--" + name.replace("_", "-"), str(getattr(args, name))])
        command.extend(["--devices", *map(str, args.devices)])
        handle = (args.output_dir / f"worker_{index}.log").open("a")
        handles.append(handle)
        workers.append(subprocess.Popen(command, env=env, stdout=handle, stderr=subprocess.STDOUT))
    state = {"status": "running", "supervisor_pid": os.getpid(),
             "worker_pids": [p.pid for p in workers], "started_unix": started}
    base.save_json(args.output_dir / "run_state.json", state)
    while any(p.poll() is None for p in workers):
        completed = aggregate(args, current["protocol_sha256"])
        print(f"AMEF-MIL已完成{completed}/5折", flush=True)
        time.sleep(10)
    completed = aggregate(args, current["protocol_sha256"])
    codes = [p.returncode for p in workers]
    state.update({"status": "complete" if completed == 5 and not any(codes) else "incomplete",
                  "completed_fold_jobs": completed, "worker_exit_codes": codes,
                  "finished_unix": time.time(), "wall_seconds": time.time()-started})
    base.save_json(args.output_dir / "run_state.json", state)
    for handle in handles:
        handle.close()
    print(f"AMEF-MIL五折训练结束：{state['status']}", flush=True)
    if state["status"] != "complete":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
