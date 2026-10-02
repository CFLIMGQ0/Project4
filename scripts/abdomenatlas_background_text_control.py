#!/usr/bin/env python3
"""固定四器官基础所见的诊断对照；仅训练/验证，不访问测试折。"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import random
import re
import statistics
import sys
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import f1_score
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
from exp_10.data import TextRecord, build_train_vocabulary, encode_text, tokenize
from exp_10.models import build_text_classifier
from exp_10.train_text_classification import tune_thresholds
from prepare_abdomenatlas3_table1 import save_json

ORIGINAL = ROOT / "outputs/abdomenatlas3_mini/experiment_answer_mask_v2"
REFERENCE = ROOT / "outputs/abdomenatlas3_mini/table1_fivefold_original_uit_answer_mask_v2/fold_1"
OUT = ROOT / "outputs/abdomenatlas3_mini/background_findings_control"
ORGANS = ["Liver", "Pancreas", "Kidney", "Spleen"]
LABELS = ["liver_lesion", "pancreatic_lesion", "kidney_lesion"]


def prepare():
    rows = json.loads((ORIGINAL / "samples.json").read_text())
    with (ROOT / "datasets/abdomenatlas3/AbdomenAtlas3.0MiniWithMeta.csv").open(newline="") as stream:
        official = {r["BDMAP ID"]: r for r in csv.DictReader(stream)}
    lengths = []
    for row in tqdm(rows, desc="提取固定四器官基础所见"):
        blocks = []
        for organ in ORGANS:
            text = official[row["exam_id"]]["structured report"]
            match = re.search(r"(?mi)^" + organ + r":[ \t]*$", text)
            assert match, (row["exam_id"], organ)
            block = re.split(r"\n\s*\n", text[match.end():].lstrip("\r\n"), maxsplit=1)[0].strip()
            assert not re.search(r"lesion|tumor|mass\b|location|contact|stage|resect|segment|\bhead\b|\btail\b|\bx\b", block, re.I)
            blocks.append(organ + ": " + re.sub(r"\s+", " ", block))
        row["findings_masked"] = " ".join(blocks)
        row["mask_hits"] = []
        row["text_source"] = "官方structured report固定四器官基础所见段落"
        row["mask_version"] = "background_findings_control"
        lengths.append(len(tokenize(row["findings_masked"])))
    split = json.loads((REFERENCE / "textcnn_encoder/split_ids.json").read_text())
    assert set(split["train"]).isdisjoint(split["val"] + split["test"])
    save_json(OUT / "samples.json", rows)
    save_json(OUT / "split_ids.json", split)
    save_json(OUT / "protocol.json", {"scope": "诊断对照，改变报告输入范围，不自动替代正式表1实验",
        "selection": "所有病例固定抽取Liver/Pancreas/Kidney/Spleen基础所见段，按同一顺序排列；不依据个体标签筛选文字",
        "retained": "器官大小、器官体积、整体密度和HU等基础信息",
        "excluded": "全部病灶专属段落、病灶尺寸、定位、血管接触、分期和IMPRESSION结论；不插入条件性掩码占位符",
        "evaluation": "仅第一折的训练与验证集；保持原模型超参数和验证选模规则；不读取测试折做本轮决策",
        "case_count": len(rows), "token_quantiles": np.quantile(lengths, [0,.5,.9,1]).tolist(),
        "source_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest()})
    (OUT / "examples.md").write_text("# 固定基础所见示例\n\n"+"\n\n".join(
        f"## {r['exam_id']}\n\n{r['findings_masked']}" for r in rows[:5])+"\n", encoding="utf-8")


def run(key):
    cfg = json.loads((REFERENCE / key / "config.json").read_text())
    rows = json.loads((OUT / "samples.json").read_text())
    split = json.loads((OUT / "split_ids.json").read_text())
    folder = OUT / key
    folder.mkdir(parents=True, exist_ok=True)
    seed = cfg["seed"]
    torch.set_num_threads(2)
    torch.cuda.set_per_process_memory_fraction(.42)
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)
    records = [TextRecord(str(i), rows[i]["exam_id"], rows[i]["findings_masked"], np.array(rows[i]["labels"]), ()) for i in split["train"]]
    vocab = build_train_vocabulary(records, 8192, 1)
    loaders = {}
    for name in ("train", "val"):
        encoded = [encode_text(rows[i]["findings_masked"], encoder_name=key, vocabulary=vocab,
                               hash_vocab_size=8192, max_length=512) for i in split[name]]
        ids = torch.tensor(np.stack([r[0] for r in encoded]), dtype=torch.long)
        masks = torch.tensor(np.stack([r[1] for r in encoded]), dtype=torch.bool)
        labels = torch.tensor(np.array([rows[i]["labels"] for i in split[name]]), dtype=torch.float32)
        loaders[name] = DataLoader(TensorDataset(ids, masks, labels), batch_size=64, shuffle=name == "train", num_workers=0)
    model = build_text_classifier(key, vocabulary_size=len(vocab), hash_vocab_size=8192, num_labels=3,
                                  max_length=512, model_config=cfg["model"]).cuda()
    target = np.array([r.labels for r in records]); positive = target.sum(0)
    criterion = torch.nn.BCEWithLogitsLoss(pos_weight=torch.tensor((len(target)-positive)/positive, dtype=torch.float32, device="cuda"))
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg["training"]["learning_rate"], weight_decay=cfg["training"]["weight_decay"])
    history, best, stale = [], -1., 0
    save_json(folder / "config.json", cfg)
    save_json(folder / "vocabulary.json", vocab)
    for epoch in tqdm(range(1, cfg["training"]["max_epochs"]+1), desc=key):
        model.train(); losses=[]
        for ids, mask, labels in loaders["train"]:
            optimizer.zero_grad(set_to_none=True)
            loss = criterion(model(ids.cuda(), mask.cuda()), labels.cuda())
            assert torch.isfinite(loss)
            loss.backward();optimizer.step();losses.append(float(loss.detach()))
        model.eval(); probabilities=[]; truth=[]
        with torch.inference_mode():
            for ids, mask, labels in loaders["val"]:
                probabilities.append(model(ids.cuda(), mask.cuda()).sigmoid().cpu().numpy());truth.append(labels.numpy())
        p,y=map(np.concatenate,(probabilities,truth))
        thresholds=tune_thresholds(y,p,cfg["training"]["threshold_grid"])
        score=float(f1_score(y,p>=thresholds,average="macro",zero_division=0))
        history.append({"epoch":epoch,"loss":statistics.mean(losses),"validation_f1":score})
        if score>best:
            best=score;stale=0
            torch.save(model.state_dict(),folder/"best_model.pt")
            np.savez_compressed(folder/"best_validation_predictions.npz",case_indices=np.array(split['val']),labels=y,probabilities=p,thresholds=thresholds)
            result={"model":key,"best_epoch":epoch,"validation_macro_f1":score,
                    "validation_fixed_0_5_f1":float(f1_score(y,p>=.5,average="macro",zero_division=0)),
                    "per_label_f1":dict(zip(LABELS,f1_score(y,p>=thresholds,average=None,zero_division=0).tolist())),
                    "validation_cases":len(y),"thresholds":thresholds.tolist()}
            save_json(folder/"validation_metrics.json",result)
        else:stale+=1
        save_json(folder/"history.json",history)
        print(f"{key} epoch={epoch} validation_f1={score:.4f} best={best:.4f}",flush=True)
        if stale>=cfg['training']['patience']:break
    reference=json.loads((REFERENCE/key/"test_metrics.json").read_text())
    result.update(reference_validation_f1=reference['best_val_macro_f1'],status='complete',
                  validation_f1_change_pp=100*(result['validation_macro_f1']-reference['best_val_macro_f1']))
    save_json(folder/"validation_metrics.json",result)
    save_json(folder/"completed.json",result)
    print(json.dumps(result,ensure_ascii=False),flush=True)


if __name__ == "__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model',choices=['textcnn_encoder','vocab_attention_encoder'])
    args=parser.parse_args()
    if args.model:run(args.model)
    else:prepare()
