#!/usr/bin/env python3
"""在已有私有图像缓存上运行三种 2026 基础适配基线，沿用患者级五折。"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
from pathlib import Path
import statistics  # 避免 src/statistics.py 遮蔽标准库。
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "src/scripts"))

from exp_10.masking import mask_answer_terms
from run_paper2026_baselines import DISPLAY, adjacency, common_params, digest, save_json
from sotas.task2.paper2026_adapted import REGISTRY, build_paper2026_adapted
from run_physionet_ct_ich_table2_baselines import CachedFeatureBackbone
from prepare_private_paper_features import feature_path

DATASETS = ("regular_white_light", "chromoscopic", "surgical", "ultrasound")
SPLIT_ROOT = ROOT / "outputs/train_runs/task3/t3_main_model"
CSV = ROOT / "datasets/task_data/task2/gastro_multilabel_task_datalist.csv"
OUT = ROOT / "outputs/paper_results/new_baselines"
LABELS = ("label_esophageal_smt", "label_esophageal_mucosal_or_tumor", "label_gastritis")
SETTINGS = {"epochs": 30, "batch_size": 16, "learning_rate": 2e-4,
            "weight_decay": .02, "warmup_ratio": .2, "max_instances": 64,
            "instance_dropout": 0., "base_seed": 2026,
            "text_max_length": 512, "text_vocab_size": 8192,
            "threshold_grid": [round(.1 + .05*i, 2) for i in range(17)]}


def source_rows(dataset: str):
    path = SPLIT_ROOT / dataset / "fold_1/split_manifest.csv"
    with path.open(encoding="utf-8-sig", newline="") as stream:
        first = list(csv.DictReader(stream))
    exams = [row["exam_dir"] for row in first]
    if len(exams) != len(set(exams)):
        raise ValueError(f"检查目录重复：{dataset}")
    exam_to_index = {exam: index for index, exam in enumerate(exams)}
    rows = {row["exam_dir"]: row for row in first}
    fold_groups = []
    for fold in range(1, 6):
        with (SPLIT_ROOT / dataset / f"fold_{fold}/split_manifest.csv").open(
                encoding="utf-8-sig", newline="") as stream:
            current = list(csv.DictReader(stream))
        if set(row["exam_dir"] for row in current) != set(exams):
            raise ValueError(f"患者折病例全集不一致：{dataset}/fold{fold}")
        for row in current:
            if any(row[label] != rows[row["exam_dir"]][label] for label in LABELS):
                raise ValueError(f"病例标签跨折不一致：{row['exam_dir']}")
        fold_groups.append([exam_to_index[row["exam_dir"]] for row in current if row["split"] == "test"])
    if sorted(index for group in fold_groups for index in group) != list(range(len(exams))):
        raise ValueError(f"测试折未形成检查分区：{dataset}")
    for fold in range(1, 6):
        with (SPLIT_ROOT / dataset / f"fold_{fold}/split_manifest.csv").open(
                encoding="utf-8-sig", newline="") as stream:
            current = list(csv.DictReader(stream))
        val = {exam_to_index[row["exam_dir"]] for row in current if row["split"] == "val"}
        if val != set(fold_groups[fold % 5]):
            raise ValueError(f"验证折不是下一测试折：{dataset}/fold{fold}")
        split_patient = {name: {row["patient_id"] for row in current if row["split"] == name}
                         for name in ("train", "val", "test")}
        if any(split_patient[a] & split_patient[b] for a, b in (("train", "val"),
                                                                ("train", "test"), ("val", "test"))):
            raise ValueError(f"患者跨集合泄漏：{dataset}/fold{fold}")
    labels = np.asarray([[int(rows[exam][label]) for label in LABELS] for exam in exams], dtype=np.int64)
    with CSV.open(encoding="utf-8-sig", newline="") as stream:
        report_rows = {row["exam_dir"]: row for row in csv.DictReader(stream)}
    if any(exam not in report_rows for exam in exams):
        raise ValueError(f"缺少配对报告：{dataset}")
    texts = [mask_answer_terms(report_rows[exam]["watch"])[0] for exam in exams]
    if any(not value for value in texts):
        raise ValueError(f"掩码后报告为空：{dataset}")
    return exams, labels, fold_groups, texts


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", choices=DATASETS, required=True)
    parser.add_argument("--method", choices=tuple(REGISTRY), required=True)
    parser.add_argument("--fold", type=int, choices=range(1, 6), required=True)
    args = parser.parse_args()
    import torch
    from types import SimpleNamespace
    from training.data import _encode_text_fields
    import run_cq500_table2_multimodal_baselines as fusion

    torch.set_num_threads(2)
    torch.backends.mha.set_fastpath_enabled(False)
    if torch.cuda.is_available():
        torch.cuda.set_per_process_memory_fraction(float(os.environ.get("PAPER_MEMORY_FRACTION", ".4")))
    exams, targets, groups, texts = source_rows(args.dataset)
    bags = []
    for exam in exams:
        path = feature_path(args.dataset, exam)
        if not path.exists():
            raise FileNotFoundError(f"先提取私有缓存特征：{path}")
        with np.load(path, allow_pickle=False) as cache:
            if str(cache["exam_dir"]) != exam:
                raise ValueError(f"特征病例不匹配：{path}")
            bags.append(cache["features"].astype(np.float32))
    patient_ids = np.arange(len(exams), dtype=np.int64)
    pairs = [_encode_text_fields({"watch": value}, ("watch",),
                                 max_length=SETTINGS["text_max_length"],
                                 vocab_size=SETTINGS["text_vocab_size"]) for value in texts]
    tokens = torch.stack([pair[0] for pair in pairs])
    masks = torch.stack([pair[1] for pair in pairs])
    if not masks.any(1).all():
        raise ValueError("掩码报告编码后为空")
    index = args.fold - 1
    train = [case for k, group in enumerate(groups) if k not in (index, (index+1)%5)
             for case in group]
    graph = adjacency(targets, train) if args.method == "gcf_net_2026" else None
    params = common_params(args.method, graph)
    sources = [Path(__file__), ROOT / "src/sotas/task2/paper2026_adapted.py", CSV,
               ROOT / "src/exp_10/masking.py",
               ROOT / "datasets/image_cache/task3_cache_manifest.jsonl.gz"]
    sources += [SPLIT_ROOT / args.dataset / f"fold_{fold}/split_manifest.csv" for fold in range(1, 6)]
    protocol = {"dataset": args.dataset, "method": args.method, "fold": args.fold,
                "scope": "基础任务适配；复用完整缓存图像并冻结 ImageNet ConvNeXt-Tiny，非原论文完整架构复现",
                "train_indices": train, "validation_indices": groups[(index+1)%5],
                "test_indices": groups[index], "num_exams": len(exams),
                "image_input": "现有 336px RGB 缓存，每检查按采集文件名顺序均匀采样至多64张，冻结768维视觉特征",
                "text_input": "none" if args.method == "gcf_net_2026" else "watch 字段统一目标词典掩码，中文字符哈希编码",
                "settings": SETTINGS, "parameters": params,
                "source_sha256": {str(path.relative_to(ROOT)): digest(path) for path in sources}}
    protocol["protocol_sha256"] = hashlib.sha256(json.dumps(protocol, sort_keys=True).encode()).hexdigest()
    folder = OUT / args.dataset / f"fold_{args.fold}" / args.method
    existing = folder / "protocol.json"
    if existing.exists() and json.loads(existing.read_text()) != protocol and (folder / "completed.json").exists():
        raise RuntimeError(f"已有实验协议不同：{folder}")
    save_json(existing, protocol)
    if (folder / "completed.json").exists():
        return

    def build(key: str, model_params: dict):
        values = dict(model_params)
        auxiliary = {name.removesuffix("_weight"): values.pop(name) for name in list(values)
                     if name.endswith("_weight")}
        model = build_paper2026_adapted(key, **values, pretrained=False, num_labels=3)
        model.instance_encoder.backbone = CachedFeatureBackbone(768, values["feature_dim"], values["dropout"])
        return model, auxiliary

    fusion.MODELS = {args.method: DISPLAY[args.method]}
    fusion.SETTINGS = SETTINGS
    fusion.LABEL_NAMES = LABELS
    fusion.build_model = build
    fusion.train_fold(SimpleNamespace(output_dir=OUT / args.dataset), args.method, index,
                      patient_ids, bags, targets, {"folds": groups}, tokens, masks,
                      params, protocol["protocol_sha256"])
    result = folder / "test_metrics.json"
    if not result.exists() or not (folder / "test_predictions.csv").exists():
        raise RuntimeError(f"训练结束但文件不全：{folder}")
    print(f"{args.dataset} {args.method} fold{args.fold}: Macro-F1="
          f"{json.loads(result.read_text())['macro_f1']:.5f}", flush=True)


if __name__ == "__main__":
    main()
