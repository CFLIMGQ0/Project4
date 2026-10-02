#!/usr/bin/env python3
"""按私有队列原训练流程补齐表1三种基线，保留历史冻结特征实验。"""
from __future__ import annotations

import argparse
import copy
import csv
import fcntl
import gzip
import hashlib
import json
import os
from pathlib import Path
import statistics
import sys

import numpy as np
from tqdm import tqdm
import yaml

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
OUT = ROOT / "outputs/private_table1_matched_20260930"
SPLITS = ROOT / "outputs/train_runs/task3/t3_main_model"
CACHE = ROOT / "datasets/image_cache"
MANIFEST = CACHE / "task3_cache_manifest.jsonl.gz"
CSV = ROOT / "datasets/task_data/task2/gastro_multilabel_task_datalist.csv"
DATASETS = ("chromoscopic", "ultrasound", "surgical", "regular_white_light")
NAMES = dict(chromoscopic="Chromoscopic", ultrasound="EUS", surgical="Surgical", regular_white_light="WLE")
METHODS = {"gcf_net_2026": "GCF-Net", "unified_multimodal_framework_2026": "Unified MM",
           "adaptive_multimodal_fusion_2026": "Adaptive Fusion"}
LABELS = ["label_esophageal_smt", "label_esophageal_mucosal_or_tumor", "label_gastritis"]


def save(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + f".{os.getpid()}.tmp")
    tmp.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n")
    tmp.replace(path)


def read_csv(path):
    with path.open(encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def digest(path):
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def prepare():
    from exp_10.masking import mask_answer_terms

    OUT.mkdir(parents=True, exist_ok=True)
    cfg_path = ROOT / "src/configs/task3/t3_multimodal_sotas_5fold.yaml"
    experiment_cfg = yaml.safe_load(cfg_path.read_text())
    reports = {row["exam_dir"]: row for row in read_csv(CSV)}
    listing = {}
    with gzip.open(MANIFEST, "rt") as handle:
        for line in tqdm(handle, desc="核对私有RGB缓存索引"):
            row = json.loads(line)
            assert row["cache_image_size"] == 336
            assert (CACHE / "shared" / row["cache_relpath"]).is_file(), row["cache_relpath"]
            exam = str(Path(row["image_path"]).parent.parent)
            listing.setdefault(exam, []).append(row["image_path"])
    sources = [Path(__file__), cfg_path, CSV, MANIFEST,
               ROOT / "pre_weights/checkpoints/convnext_tiny-983f1562.pth"]
    for directory in [ROOT / "src/training", ROOT / "src/model", ROOT / "src/exp_4", ROOT / "src/exp_8"]:
        sources.extend(sorted(directory.rglob("*.py")))
    sources += [ROOT / name for name in ["src/train.py", "src/exp_10/masking.py",
        "src/scripts/task1_table_5fold.py", "src/scripts/run_paper2026_baselines.py",
        "src/sotas/task2/paper2026_adapted.py", "src/sotas/task2/multimodal_sotas.py"]]
    inputs = {}
    for dataset in tqdm(DATASETS, desc="核对四队列患者级五折"):
        records = {}
        folds = []
        for fold in range(1, 6):
            source = SPLITS / dataset / f"fold_{fold}"
            sources.extend([source / "split_manifest.csv", source / "config.yaml"])
            rows = read_csv(source / "split_manifest.csv")
            parts = {name: [row["exam_dir"] for row in rows if row["split"] == name]
                     for name in ("train", "val", "test")}
            patients = {name: {row["patient_id"] for row in rows if row["split"] == name}
                        for name in parts}
            for a, b in [("train", "val"), ("train", "test"), ("val", "test")]:
                assert patients[a].isdisjoint(patients[b]), (dataset, fold)
            assert len({row["exam_dir"] for row in rows}) == len(rows)
            for row in rows:
                exam = row["exam_dir"]
                report = reports[exam]
                images = sorted(listing[exam])
                assert len(images) == int(row["img_num"]), (dataset, exam)
                labels = [int(row[key]) for key in LABELS]
                assert labels == [int(report[key]) for key in LABELS]
                masked, _ = mask_answer_terms(report["watch"])
                assert masked.strip()
                record = {"exam_dir": exam, "patient_id": row["patient_id"],
                          "image_paths": images, "img_num": len(images), "labels": labels,
                          "report_title": row["report_title"], "watch": masked,
                          "text_raw": {"watch": masked}}
                if exam in records:
                    assert records[exam] == record
                records[exam] = record
            old = yaml.safe_load((source / "config.yaml").read_text())
            assert old["seed"] == 2026 + fold
            folds.append({"parts": parts, "reference": old})
        assert sorted(exam for fold in folds for exam in fold["parts"]["test"]) == sorted(records)
        inputs[dataset] = {"records": records, "folds": folds}
    save(OUT / "inputs.json", inputs)
    protocol = {"说明": "只补齐三种2026适配基线在四个私有队列上的五折。直接读取336px RGB缓存并沿用224px图像变换、ConvNeXt-Tiny部分微调、患者划分、仅训练集重采样、统一报告掩码及原训练器；不使用冻结768维特征。",
                "selection_alias": "best_macro_f1", "datasets": DATASETS, "methods": METHODS,
                "class_balance": experiment_cfg["class_balance"],
                "memory_optimization": "骨干按原16图分块并使用保留随机状态的激活检查点，单GPU进程显存上限45%",
                "contrastive_note": "Unified MM保留原有配对对比项；沿用历史batch_size=1时该项按既有适配实现为零，梯度累积不改变其批内定义。",
                "source_sha256": {str(p.relative_to(ROOT)): digest(p) for p in sources},
                "inputs_sha256": digest(OUT / "inputs.json")}
    protocol["protocol_sha256"] = hashlib.sha256(json.dumps(protocol, sort_keys=True, ensure_ascii=False).encode()).hexdigest()
    path = OUT / "protocol.json"
    if path.exists():
        assert json.loads(path.read_text()) == protocol, "补跑协议已改变，禁止混用结果"
    else:
        save(path, protocol)
    jobs = []
    for fold in range(1, 6):
        for dataset in DATASETS:
            for method in METHODS:
                key = f"private_matched_{dataset}_{method}_fold_{fold}"
                jobs.append({"kind": "external", "key": key, "dataset": dataset, "model": method,
                    "fold": fold, "protocol_sha256": protocol["protocol_sha256"],
                    "completion_path": str(OUT / dataset / f"fold_{fold}" / method / "completed.json"),
                    "arguments": [str(Path(__file__).resolve()), "--dataset", dataset,
                                  "--method", method, "--fold", str(fold)]})
    save(OUT / "jobs.json", jobs)
    summarize()
    print(f"已准备{len(jobs)}个折次，输入图像与五折清单核对完成。", flush=True)


def summarize():
    with (OUT / "summary.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        summaries = []
        for method in METHODS:
            for dataset in DATASETS:
                results = [json.loads(p.read_text()) for p in sorted((OUT / dataset).glob(f"fold_*/{method}/completed.json"))]
                row = {"dataset": dataset, "model": method, "completed_folds": len(results), "folds": results}
                if len(results) == 5:
                    row.update(macro_f1_mean=statistics.mean(r["macro_f1"] for r in results),
                               macro_f1_std=statistics.stdev(r["macro_f1"] for r in results))
                summaries.append(row)
        save(OUT / "summary.json", summaries)


def run(dataset, method, fold):
    import torch
    from torch.utils.checkpoint import checkpoint
    import train as core
    from scripts.task1_table_5fold import build_fold_context
    from run_paper2026_baselines import adjacency, common_params
    from sotas.task2.paper2026_adapted import build_paper2026_adapted
    from training import TrainerConfig, to_builtin_type

    os.environ["PROJECT4_DISABLE_DISK_CACHE_WRITE"] = "1"
    torch.set_num_threads(2)
    torch.cuda.set_per_process_memory_fraction(.45)
    torch.backends.mha.set_fastpath_enabled(False)
    torch.set_float32_matmul_precision("medium")
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.benchmark = True
    protocol = json.loads((OUT / "protocol.json").read_text())
    for name, value in protocol["source_sha256"].items():
        assert digest(ROOT / name) == value, name
    assert digest(OUT / "inputs.json") == protocol["inputs_sha256"]
    data = json.loads((OUT / "inputs.json").read_text())[dataset]
    row = data["folds"][fold - 1]
    old = row["reference"]
    raw = {key: [copy.deepcopy(data["records"][exam]) for exam in exams]
           for key, exams in row["parts"].items()}
    seed = old["seed"]
    folder = OUT / dataset / f"fold_{fold}" / method
    folder.mkdir(parents=True, exist_ok=True)
    with (folder / "exclusive.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        if (folder / "completed.json").exists():
            assert json.loads((folder / "completed.json").read_text())["protocol_sha256"] == protocol["protocol_sha256"]
            return
        context = build_fold_context(base_context={"output_root": OUT, "task_selection_dir": str(CSV.parent.parent)},
            task_csv=CSV, split_data=raw,
            train_cfg={"seed": seed, "class_balance": protocol["class_balance"]}, task_name="task2")
        payload = context["tasks"]["task2"]
        balance = payload["balance_report"]
        prior_balance = old["run"]["class_balance_report"]
        for key in ["original_train_size", "balanced_train_size", "added_records", "before", "after", "repeat_histogram"]:
            assert balance[key] == prior_balance[key], (dataset, fold, key)
        graph = adjacency(np.array([r["labels"] for r in raw["train"]]), list(range(len(raw["train"])))) if method == "gcf_net_2026" else None
        params = common_params(method, graph)
        values = dict(params)
        aux = {key.removesuffix("_weight"): values.pop(key) for key in list(values) if key.endswith("_weight")}
        core.seed_everything(seed)
        model = build_paper2026_adapted(method, pretrained=True, num_labels=3, **values)
        assert any(p.requires_grad for p in model.instance_encoder.backbone.parameters())
        forward = model.instance_encoder.backbone.forward
        def memory_efficient_forward(images):
            if model.training and torch.is_grad_enabled():
                return checkpoint(forward, images, use_reentrant=False, preserve_rng_state=True)
            return forward(images)
        model.instance_encoder.backbone.forward = memory_efficient_forward
        trainer = dict(old["trainer"])
        trainer.update(pos_weight=payload["pos_weight"], aux_loss_weights=aux, use_multi_gpu=False,
            data_parallel_device_ids=None, resume_path=core.auto_series_resume_checkpoint(folder),
            test_result_metadata={"dataset": dataset, "fold": fold, "method": method,
                "protocol_sha256": protocol["protocol_sha256"],
                "inference_inputs": "image" if method == "gcf_net_2026" else "image+masked_watch"})
        run_cfg = dict(old["run"])
        run_cfg.update(image_cache_dir=str(CACHE), image_cache_manifest=str(MANIFEST),
                       image_cache_warmup=False, class_balance_report=balance)
        if method == "gcf_net_2026":
            run_cfg.update(modality_level="image", modality_fields="image", inference_inputs="image", leakage_note="不输入报告")
            for records in payload["split"].values():
                for record in records:
                    record["watch"], record["text_raw"] = "", {}
        save(folder / "class_balance_report.json", balance)
        save(folder / "split_ids.json", row["parts"])
        save(folder / "protocol.json", {"protocol_sha256": protocol["protocol_sha256"],
            "dataset": dataset, "method": method, "fold": fold, "parameters": params,
            "reference_config": str(SPLITS / dataset / f"fold_{fold}/config.yaml")})
        print(f"开始表1补跑：{dataset}/{method}/fold{fold}，使用RGB图像微调，最多30轮。", flush=True)
        result = core.run_single_model(model_name=method, model=model, trainer_cfg=TrainerConfig(**trainer),
            split_data=payload["split"], task_name="task2", image_size=old["image_size"], num_workers=old["num_workers"],
            run_dir=folder, seed=seed, run_cfg=run_cfg, model_param_cfg=params, min_instances=1,
            train_sampling=run_cfg["train_sampling_strategy"], eval_sampling=run_cfg["eval_sampling_strategy"],
            active_gpu_count=1, label_names=LABELS, class_names=LABELS, cache_root_dir=CACHE)
        save(folder / "result.json", to_builtin_type(result))
        chosen = result["test_results"][protocol["selection_alias"]]
        save(folder / "completed.json", {"protocol_sha256": protocol["protocol_sha256"],
            "dataset": dataset, "method": method, "fold": fold,
            "macro_f1": chosen["metrics"]["macro_f1"], "best_epoch": chosen["best_epoch"],
            "source": str(folder), "selection_alias": protocol["selection_alias"]})
    summarize()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prepare", action="store_true")
    parser.add_argument("--dataset", choices=DATASETS)
    parser.add_argument("--method", choices=METHODS)
    parser.add_argument("--fold", type=int, choices=range(1, 6))
    args = parser.parse_args()
    if args.prepare:
        prepare()
    else:
        assert args.dataset and args.method and args.fold
        run(args.dataset, args.method, args.fold)


if __name__ == "__main__":
    main()
