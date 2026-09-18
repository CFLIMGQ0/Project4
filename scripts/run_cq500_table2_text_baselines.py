#!/usr/bin/env python3
"""使用论文表2的五种文本编码器，在CQ500图像派生文本上复用原五折划分。"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
from pathlib import Path
import re
import statistics
import subprocess
import sys
import time
import traceback

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
MODELS = {
    "hashed_mean_encoder": "Hashed mean",
    "vocab_attention_encoder": "Vocabulary attention",
    "textcnn_encoder": "TextCNN",
    "bigru_encoder": "BiGRU",
    "transformer_encoder": "Transformer",
}
LABEL_COLUMNS = ["IPH", "MassEffect", "MidlineShift"]
LABEL_NAMES = ["IPH", "Mass Effect", "Midline Shift"]


def save_json(path: Path, obj: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(obj, ensure_ascii=False, indent=2), encoding="utf-8")
    temporary.replace(path)


def load_inputs(args):
    with args.text_csv.open(encoding="utf-8-sig", newline="") as file:
        text_rows = list(csv.DictReader(file))
    texts = {int(r["patient_id"]): r["image_description"] for r in text_rows}
    if len(text_rows) != 491 or len(texts) != 491 or set(texts) != set(range(491)):
        raise ValueError("输入必须含491例，且每例只有一段文本")
    if not all(text.strip() for text in texts.values()):
        raise ValueError("输入文本存在空值")
    labels = {}
    with args.reads.open(encoding="utf-8-sig", newline="") as file:
        for row in csv.DictReader(file):
            scan_id = int(re.search(r"(\d+)$", row["name"]).group(1))
            if scan_id in labels:
                raise ValueError("标签文件存在重复病例")
            labels[scan_id] = [int(sum(int(row[f"R{r}:{name}"]) for r in (1, 2, 3)) >= 2)
                               for name in LABEL_COLUMNS]
    folds = json.loads(args.folds_json.read_text())
    flattened = [i for fold in folds["folds"] for i in fold]
    assert len(folds["folds"]) == 5 and len(flattened) == 491
    assert len(set(flattened)) == 491 and set(flattened) == set(texts) == set(labels)
    assert folds["labels"] == LABEL_NAMES
    assert [sum(labels[i][j] for i in texts) for j in range(3)] == folds["positive_counts"]
    for indices, expected in zip(folds["folds"], folds["fold_positive_counts"]):
        assert [sum(labels[i][j] for i in indices) for j in range(3)] == expected
    return texts, labels, folds


def configuration(args):
    import yaml
    cfg = yaml.safe_load(args.config.read_text())
    cfg["data"].update({"label_names": LABEL_NAMES, "text_field": "image_description"})
    # 词表和token数组很小，主进程直接取批次，避免每轮派生数据加载进程。
    cfg["training"]["num_workers"] = 0
    cfg["paths"] = {"data_csv": str(args.text_csv), "output_dir": str(args.output_dir)}
    cfg["experiment_name"] = "cq500_table2_text_baselines_uniform64"
    return cfg


def protocol(args):
    tracked = {
        "text_csv": args.text_csv, "label_csv": args.reads,
        "patient_folds": args.folds_json, "base_config": args.config,
        "text_models_source": ROOT / "src/exp_10/models.py",
        "text_training_source": ROOT / "src/exp_10/train_text_classification.py",
        "runner_source": Path(__file__),
    }
    value = {
        "source_sha256": {key: hashlib.sha256(path.read_bytes()).hexdigest() for key, path in tracked.items()},
        "models": MODELS, "labels": LABEL_NAMES,
        "config": configuration(args), "input": "uniform64_image_derived_AI_drafts",
        "text_rewriting": False, "additional_category_masking": False,
        "split": "reuse_image_experiment_folds; test=k, validation=(k+1)%5, train=other_three",
        "vocabulary": "training_fold_text_only", "threshold_source": "validation_only",
        "class_balance": "training_fold_negative_positive_ratio_in_BCE_pos_weight; no_oversampling",
        "primary_metric": "mean_and_sample_std_of_five_test_fold_macro_f1",
        "extra_metric": "pooled_OOF_F1_and_fixed_0.5_threshold_F1",
        "note": "此实验使用图像派生AI试稿，不代表独立采集的临床文本外部验证。真实标签仅作为分类监督与评分目标，不拼入文本。",
    }
    value["protocol_sha256"] = hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False).encode()).hexdigest()
    return value


def worker(args):
    import numpy as np
    import torch
    from sklearn.metrics import f1_score
    from exp_10.data import TextRecord, build_train_vocabulary
    from exp_10.train_text_classification import train_one_model

    torch.set_num_threads(4)
    current_protocol = protocol(args)
    recorded = json.loads((args.output_dir / "protocol.json").read_text())
    if current_protocol["protocol_sha256"] != recorded["protocol_sha256"]:
        raise ValueError("运行协议发生变化，禁止混用结果")
    texts, labels, folds = load_inputs(args)
    records = {i: TextRecord(str(i), f"cq500_case_{i:03d}", texts[i], np.asarray(labels[i], dtype=np.int64), ())
               for i in texts}
    jobs = [(model, fold) for model in MODELS for fold in range(5)]
    for job_index, (model, fold) in enumerate(jobs):
        if job_index % len(args.devices) != args.worker_index:
            continue
        run_dir = args.output_dir / f"fold_{fold+1}" / model
        marker = run_dir / "completed.json"
        if marker.exists():
            previous = json.loads(marker.read_text())
            if previous["protocol_sha256"] == current_protocol["protocol_sha256"]:
                print(f"跳过已完成：{MODELS[model]}，折{fold+1}", flush=True)
                continue
            raise ValueError("已有结果采用不同协议")
        try:
            start = time.monotonic()
            validation_fold = (fold + 1) % 5
            selected = {
                "train": [i for j, indices in enumerate(folds["folds"]) if j not in (fold, validation_fold) for i in indices],
                "val": folds["folds"][validation_fold], "test": folds["folds"][fold],
            }
            assert not (set(selected["train"]) & set(selected["val"]))
            assert not (set(selected["train"]) & set(selected["test"]))
            assert not (set(selected["val"]) & set(selected["test"]))
            splits = {name: [records[i] for i in indices] for name, indices in selected.items()}
            cfg = configuration(args)
            cfg["seed"] += fold + 1
            vocabulary = build_train_vocabulary(splits["train"], cfg["data"]["vocab_size"], cfg["data"]["min_token_frequency"])
            save_json(run_dir / "config.json", cfg)
            save_json(run_dir / "vocabulary.json", vocabulary)
            save_json(run_dir / "split_ids.json", selected)
            print(f"开始{MODELS[model]}，折{fold+1}；训练/验证/测试={list(map(len,selected.values()))}", flush=True)
            metrics = train_one_model(model, cfg, splits, vocabulary, run_dir.parent)
            with (run_dir / "test_predictions.csv").open(encoding="utf-8-sig", newline="") as file:
                predictions = list(csv.DictReader(file))
            assert [int(r["patient_id"]) for r in predictions] == selected["test"]
            targets = np.asarray([[int(r[f"true_{n}"]) for n in LABEL_NAMES] for r in predictions])
            probs = np.asarray([[float(r[f"prob_{n}"]) for n in LABEL_NAMES] for r in predictions])
            thresholds = np.asarray([metrics["thresholds"][n] for n in LABEL_NAMES])
            assert np.isfinite(probs).all() and ((0 <= probs) & (probs <= 1)).all()
            assert np.isclose(f1_score(targets, probs >= thresholds, average="macro", zero_division=0), metrics["macro_f1"])
            metrics.update({
                "experiment_name": cfg["experiment_name"], "text_field": "image_description",
                "answer_masking": False, "text_source": "image_derived_AI_drafts",
                "fold": fold+1, "model_display": MODELS[model],
                "protocol_sha256": current_protocol["protocol_sha256"],
                "split_sizes": {name:len(ids) for name,ids in selected.items()},
                "macro_f1_fixed_0_5": float(f1_score(targets, probs >= 0.5, average="macro", zero_division=0)),
                "micro_f1_fixed_0_5": float(f1_score(targets, probs >= 0.5, average="micro", zero_division=0)),
                "wall_seconds": round(time.monotonic()-start,3),
            })
            save_json(run_dir / "test_metrics.json", metrics)
            save_json(marker, {"protocol_sha256": current_protocol["protocol_sha256"], "fold":fold+1, "model":model})
            print(f"完成{MODELS[model]}，折{fold+1}：测试Macro-F1={metrics['macro_f1']:.4f}", flush=True)
            torch.cuda.empty_cache()
        except Exception as exc:
            save_json(args.output_dir / "errors" / f"{model}_fold_{fold+1}.json",
                      {"error": str(exc), "traceback": traceback.format_exc()})
            raise


def aggregate(args, digest):
    import numpy as np
    from sklearn.metrics import f1_score
    summaries, completed = [], 0
    for model, display in MODELS.items():
        metrics, predictions = [], []
        for fold in range(1,6):
            folder = args.output_dir / f"fold_{fold}" / model
            if not (folder / "completed.json").exists():
                continue
            row = json.loads((folder / "test_metrics.json").read_text())
            assert row["protocol_sha256"] == digest
            metrics.append(row)
            with (folder / "test_predictions.csv").open(encoding="utf-8-sig", newline="") as file:
                predictions.extend({**r,"fold":fold} for r in csv.DictReader(file))
        completed += len(metrics)
        if len(metrics) != 5:
            continue
        predictions.sort(key=lambda r:int(r["patient_id"]))
        assert len(predictions)==491 and {int(r['patient_id']) for r in predictions}==set(range(491))
        y = np.asarray([[int(r[f"true_{n}"]) for n in LABEL_NAMES] for r in predictions])
        pred = np.asarray([[int(r[f"pred_{n}"]) for n in LABEL_NAMES] for r in predictions])
        prob = np.asarray([[float(r[f"prob_{n}"]) for n in LABEL_NAMES] for r in predictions])
        summary = {
            "model_key": model, "model":display, "completed_folds":5, "oof_cases":491,
            "macro_f1_mean": statistics.mean(r["macro_f1"] for r in metrics),
            "macro_f1_std": statistics.stdev(r["macro_f1"] for r in metrics),
            "micro_f1_mean": statistics.mean(r["micro_f1"] for r in metrics),
            "oof_macro_f1": float(f1_score(y,pred,average="macro",zero_division=0)),
            "oof_micro_f1": float(f1_score(y,pred,average="micro",zero_division=0)),
            "oof_per_label_f1": dict(zip(LABEL_NAMES,f1_score(y,pred,average=None,zero_division=0).tolist())),
            "macro_f1_fixed_0_5_mean":statistics.mean(r["macro_f1_fixed_0_5"] for r in metrics),
            "macro_f1_fixed_0_5_std":statistics.stdev(r["macro_f1_fixed_0_5"] for r in metrics),
            "oof_macro_f1_fixed_0_5":float(f1_score(y,prob>=0.5,average="macro",zero_division=0)),
            "folds":metrics,
        }
        summaries.append(summary)
        with (args.output_dir/f"{model}_oof_predictions.csv").open("w",encoding="utf-8-sig",newline="") as file:
            writer=csv.DictWriter(file,fieldnames=list(predictions[0])); writer.writeheader(); writer.writerows(predictions)
    save_json(args.output_dir/"summary.json",summaries)
    with (args.output_dir/"summary.csv").open("w",encoding="utf-8-sig",newline="") as file:
        fields=["model","macro_f1_mean","macro_f1_std","micro_f1_mean","oof_macro_f1","oof_micro_f1","macro_f1_fixed_0_5_mean","macro_f1_fixed_0_5_std"]
        writer=csv.DictWriter(file,fieldnames=fields,extrasaction="ignore"); writer.writeheader(); writer.writerows(summaries)
    report = "# CQ500 五种文本基线\n\n491例图像派生AI文本，沿用图像实验五折划分；词表、模型选择及阈值均在训练/验证数据上确定。输入原文不额外掩码或改写。\n\n| 模型 | 五折Macro-F1（均值±标准差） | 汇总OOF Macro-F1 |\n|---|---:|---:|\n"
    report += "\n".join(f"| {r['model']} | {r['macro_f1_mean']:.4f} ± {r['macro_f1_std']:.4f} | {r['oof_macro_f1']:.4f} |" for r in summaries)
    report += "\n\n主结果使用各折验证集确定的标签阈值。固定0.5阈值结果另见summary.csv，不能将两种阈值结果混为同一比较。该文本由图像生成，不是独立临床报告。\n"
    (args.output_dir/"results.md").write_text(report,encoding="utf-8")
    save_json(args.output_dir/"progress.json",{"completed_fold_jobs":completed,"expected_fold_jobs":25,"completed_models":len(summaries)})
    return completed


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--text-csv",type=Path,default=ROOT/"outputs/cq500/image_descriptions/uniform64/descriptions_draft.csv")
    parser.add_argument("--reads",type=Path,default=ROOT/"datasets/cq500/raw/reads.csv")
    parser.add_argument("--folds-json",type=Path,default=ROOT/"outputs/cq500/table2_image_baselines/patient_folds.json")
    parser.add_argument("--config",type=Path,default=ROOT/"src/configs/task2/exp10_text_classification.yaml")
    parser.add_argument("--output-dir",type=Path,default=ROOT/"outputs/cq500/table2_text_baselines_uniform64")
    parser.add_argument("--devices",type=int,nargs="+",default=[0,1,2,3])
    parser.add_argument("--worker-index",type=int)
    parser.add_argument("--audit-only",action="store_true")
    args=parser.parse_args()
    if args.worker_index is not None:
        worker(args); return
    texts,labels,folds=load_inputs(args)
    current=protocol(args); args.output_dir.mkdir(parents=True,exist_ok=True)
    saved=args.output_dir/"protocol.json"
    if saved.exists() and json.loads(saved.read_text())["protocol_sha256"]!=current["protocol_sha256"]:
        raise ValueError("输出目录已采用其他协议，请另选目录")
    save_json(saved,current)
    save_json(args.output_dir/"data_audit.json",{"cases":len(texts),"unique_texts":len(set(texts.values())),"label_names":LABEL_NAMES,"positive_counts":folds["positive_counts"],"fold_sizes":[len(f) for f in folds["folds"]],"patient_overlap":0,"text_edited":False})
    if args.audit_only:
        print("491例文本、标签、五折对应关系核对通过",flush=True); return
    handles,workers=[],[]; started=time.time()
    for index,device in enumerate(args.devices):
        environment=os.environ.copy(); environment.update({"CUDA_VISIBLE_DEVICES":str(device),"OMP_NUM_THREADS":"4","TOKENIZERS_PARALLELISM":"false"})
        command=[sys.executable,"-u",str(Path(__file__).resolve()),"--worker-index",str(index),"--text-csv",str(args.text_csv),"--reads",str(args.reads),"--folds-json",str(args.folds_json),"--config",str(args.config),"--output-dir",str(args.output_dir),"--devices",*map(str,args.devices)]
        handle=(args.output_dir/f"worker_{index}.log").open("a"); handles.append(handle)
        workers.append(subprocess.Popen(command,env=environment,stdout=handle,stderr=subprocess.STDOUT))
    save_json(args.output_dir/"run_state.json",{"status":"running","supervisor_pid":os.getpid(),"worker_pids":[p.pid for p in workers],"started_unix":started})
    while any(p.poll() is None for p in workers):
        completed=aggregate(args,current["protocol_sha256"])
        print(f"已完成{completed}/25个模型折次",flush=True); time.sleep(15)
    completed=aggregate(args,current["protocol_sha256"]); codes=[p.returncode for p in workers]
    status="complete" if completed==25 and not any(codes) else "incomplete"
    save_json(args.output_dir/"run_state.json",{"status":status,"completed_fold_jobs":completed,"worker_exit_codes":codes,"started_unix":started,"finished_unix":time.time(),"wall_seconds":time.time()-started})
    for h in handles: h.close()
    print(f"五折文本实验结束：{status}",flush=True)
    if status!="complete": raise SystemExit(1)


if __name__=="__main__":
    main()
