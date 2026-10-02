#!/usr/bin/env python3
"""仅用第一折训练/验证集审计文本预测线索，不改变掩码或正在训练的任务。"""
from __future__ import annotations
import csv
import hashlib
import json
import statistics
import sys
from collections import Counter
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import f1_score
from tqdm import tqdm

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
from exp_10.data import tokenize
from exp_10.models import build_text_classifier

INPUT = ROOT / "outputs/abdomenatlas3_mini/experiment_answer_mask_v2"
RUN = ROOT / "outputs/abdomenatlas3_mini/table1_fivefold_original_uit_answer_mask_v2/fold_1"
OUT = ROOT / "outputs/abdomenatlas3_mini/text_cue_audit_v2"
LABELS = ["liver_lesion", "pancreatic_lesion", "kidney_lesion"]


def save(name, value):
    (OUT / name).write_text(json.dumps(value, ensure_ascii=False, indent=2)+"\n", encoding="utf-8")


def load_model(key):
    folder = RUN / key
    assert (folder / "completed.json").exists()
    cfg = json.loads((folder / "config.json").read_text())
    vocab = json.loads((folder / "vocabulary.json").read_text())
    model = build_text_classifier(key, vocabulary_size=len(vocab), hash_vocab_size=8192,
                                  num_labels=3, max_length=512, model_config=cfg["model"])
    model.load_state_dict(torch.load(folder / "best_model.pt", map_location="cpu", weights_only=True))
    model.eval()
    model.requires_grad_(False)
    thresholds = json.loads((folder / "test_metrics.json").read_text())["thresholds"]
    return model, vocab, np.asarray([thresholds[l] for l in LABELS])


def predict(model, vocab, tokens, removed=frozenset()):
    result = []
    with torch.inference_mode():
        for start in range(0, len(tokens), 32):
            ids = [[vocab.get(t, 1) for t in text if t not in removed] or [1] for text in tokens[start:start+32]]
            n = max(4, max(map(len, ids)))
            x = torch.zeros(len(ids), n, dtype=torch.long)
            mask = torch.zeros_like(x, dtype=torch.bool)
            for i, values in enumerate(ids):
                x[i, :len(values)] = torch.tensor(values)
                mask[i, :len(values)] = True
            result.append(model(x, mask).sigmoid().numpy())
    return np.concatenate(result)


def main():
    torch.set_num_threads(6)
    OUT.mkdir(parents=True, exist_ok=True)
    rows = json.loads((INPUT / "samples.json").read_text())
    split = json.loads((RUN / "vocab_attention_encoder/split_ids.json").read_text())
    train, val = split["train"], split["val"]
    assert set(train).isdisjoint(val) and set(val).isdisjoint(split["test"])
    tokens = [tokenize(rows[i]["findings_masked"])[:512] for i in val]
    y = np.asarray([rows[i]["labels"] for i in val])
    train_y = np.asarray([rows[i]["labels"] for i in train])
    train_counts = [Counter() for _ in LABELS]
    train_negative = [Counter() for _ in LABELS]
    for i in tqdm(train, desc="统计训练集词语关联"):
        seen = set(tokenize(rows[i]["findings_masked"])[:512])
        for j in range(3):
            (train_counts[j] if rows[i]["labels"][j] else train_negative[j]).update(seen)
    attention, vocab, threshold = load_model("vocab_attention_encoder")
    counts = torch.zeros(len(val), len(vocab))
    for i, text in enumerate(tokens):
        for token, count in Counter(text).items():
            counts[i, vocab.get(token, 1)] += count
    # 注意力编码器没有上下文交互，可以精确计算删除某词全部出现位置后的池化向量。
    with torch.inference_mode():
        feature = attention.encoder.token_projection(attention.encoder.embedding(torch.arange(len(vocab))))
        score = attention.encoder.attention(feature).squeeze(1)
        weight = (score-score.max()).exp()
        denominator = (counts * weight).sum(1)
        numerator = counts @ (feature * weight[:, None])
        pooled = numerator / denominator[:, None]
        logits = attention.classifier(pooled)
        base = logits.sigmoid().numpy()
    direct = predict(attention, vocab, tokens)
    assert np.max(np.abs(base-direct)) < 5e-5
    base_f1 = f1_score(y, base >= threshold, average=None, zero_division=0)
    frequency = (counts > 0).sum(0)
    candidates = [(word, index) for word, index in vocab.items() if index > 1 and frequency[index] >= 15]
    rankings = []
    for word, index in tqdm(candidates, desc="逐词删除并重新预测"):
        which = torch.where(counts[:, index] > 0)[0]
        with torch.inference_mode():
            deleted_weight = counts[which, index] * weight[index]
            new_pool = (numerator[which] - deleted_weight[:, None]*feature[index]) / (denominator[which]-deleted_weight).clamp_min(1e-12)[:, None]
            changed_logit = attention.classifier(new_pool)
            changed_p = changed_logit.sigmoid().numpy()
        remaining = base.copy()
        remaining[which.numpy()] = changed_p
        scores = f1_score(y, remaining >= threshold, average=None, zero_division=0)
        for j, label in enumerate(LABELS):
            positive = y[which.numpy(), j] == 1
            number = int(positive.sum())
            delta = base[which.numpy(), j] - changed_p[:, j]
            total_positive = int(y[:, j].sum())
            rankings.append({"label": label, "word": word,
                "train_positive_present": train_counts[j][word], "train_positive_total": int(train_y[:,j].sum()),
                "train_negative_present": train_negative[j][word], "train_negative_total": int((1-train_y[:,j]).sum()),
                "validation_present": len(which), "validation_positive_present": number,
                "positive_probability_drop_pp": float(100*delta[positive].sum()/total_positive),
                "positive_probability_drop_when_present_pp": float(100*delta[positive].mean()) if number else 0.,
                "positive_logit_drop_when_present": float((logits[which,j].numpy()-changed_logit[:,j].numpy())[positive].mean()) if number else 0.,
                "label_f1_drop_pp": float(100*(base_f1[j]-scores[j]))})
    save("word_rankings.json", rankings)
    with (OUT / "word_rankings.csv").open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rankings[0]));writer.writeheader();writer.writerows(rankings)
    top = {label: sorted([r for r in rankings if r["label"] == label and r["validation_positive_present"] >= 10],
                         key=lambda r: r["positive_probability_drop_pp"], reverse=True)[:15] for label in LABELS}
    save("top_words.json", top)
    for label, records in top.items():
        print(label, [(r["word"], round(r["positive_probability_drop_pp"],2)) for r in records[:8]], flush=True)

    groups = {
        "掩码占位符": {"masktarget"},
        "肝脏分段位置": {"hepatic", "segment", "segments", "lobe", "lobes"},
        "胰腺血管关系": {"contact", "vein", "veins", "portal", "aorta", "smv", "sma", "cha", "ivc", "mesenteric",
                      "superior", "artery", "arteries", "vena", "cava", "encasement", "encases", "abutment", "abuts", "vascular", "invasion", "involvement"},
        "胰腺部位": {"head", "body", "tail", "uncinate"},
        "肾脏及左右位置": {"renal", "kidney", "kidneys", "left", "right", "bilateral", "cortex", "cortical", "pole"},
        "密度描述": {"hypoattenuating", "hyperattenuating", "isoattenuating", "hypodense", "hyperdense", "isodense", "attenuation"},
        "病灶描述模板": {"largest", "smaller", "smallest", "located", "identified", "present", "multiple", "several", "surrounding", "tissue"},
    }
    groups["合并上述线索"] = set.union(*groups.values())
    group_results = []
    for key in ("vocab_attention_encoder", "textcnn_encoder"):
        model, vocabulary, thresholds = load_model(key)
        baseline = predict(model, vocabulary, tokens)
        original_scores = f1_score(y, baseline >= thresholds, average=None, zero_division=0)
        for name, removed in tqdm(groups.items(), desc=f"{key}词组敏感性"):
            changed = predict(model, vocabulary, tokens, removed)
            scores = f1_score(y, changed >= thresholds, average=None, zero_division=0)
            result = {"model": key, "group": name, "words": sorted(removed),
                "baseline_macro_f1": float(original_scores.mean()), "removed_macro_f1": float(scores.mean()),
                "macro_f1_drop_pp": float(100*(original_scores.mean()-scores.mean())),
                "baseline_label_f1": dict(zip(LABELS, original_scores.tolist())),
                "removed_label_f1": dict(zip(LABELS, scores.tolist())),
                "label_f1_drop_pp": dict(zip(LABELS, (100*(original_scores-scores)).tolist()))}
            group_results.append(result)
            save("group_deletion.json", group_results)
            print(key,name,'F1下降',round(result['macro_f1_drop_pp'],2),flush=True)
    save("protocol.json", {"scope": "第一折训练集词频关联、第一折验证集逐词/词组删除；不修改训练或掩码，不使用测试集选词",
        "validation_cases": len(val), "training_cases": len(train), "labels": LABELS,
        "checkpoints": {k: hashlib.sha256((RUN/k/"best_model.pt").read_bytes()).hexdigest() for k in ("vocab_attention_encoder","textcnn_encoder")},
        "thresholds": "保持各模型原验证集阈值固定，不重新调参", "execution": "仅CPU",
        "limit": "删词敏感性说明模型依赖，不等价于临床因果；相关词可能是合法影像所见或合成模板线索"})
    lines = ["# AbdomenAtlas 文本预测线索审计", "", "第一折训练集统计词与标签的关联，验证集检查固定模型删词后的预测变化。没有修改掩码、训练、论文或测试结果。", ""]
    for label, records in top.items():
        lines += [f"## {label}", "", "| 词 | 训练阳性中出现 | 训练阴性中出现 | 删除后阳性平均概率下降（百分点） |", "|---|---:|---:|---:|"]
        for r in records[:10]:
            lines.append(f"| {r['word']} | {r['train_positive_present']}/{r['train_positive_total']} | {r['train_negative_present']}/{r['train_negative_total']} | {r['positive_probability_drop_pp']:.2f} |")
        lines.append("")
    lines += ["## 词组删除", "", "| 模型 | 删除线索 | 原验证F1 | 删除后F1 | 差值（百分点） |", "|---|---|---:|---:|---:|"]
    for r in group_results:
        lines.append(f"| {r['model']} | {r['group']} | {r['baseline_macro_f1']*100:.2f} | {r['removed_macro_f1']*100:.2f} | {-r['macro_f1_drop_pp']:.2f} |")
    lines += ["", "这些结果说明当前模型对词语和模板结构的敏感性，不证明单个词必然决定所有病例。尺寸、密度及位置也可能属于合理的诊断证据。"]
    (OUT / "report.md").write_text("\n".join(lines)+"\n", encoding="utf-8")


if __name__ == "__main__":
    main()
