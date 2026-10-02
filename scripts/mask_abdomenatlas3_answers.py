#!/usr/bin/env python3
"""为肝、胰、肾病灶分类统一遮蔽直接答案词及明确的阴性诊断表达。"""
from __future__ import annotations

import collections
import csv
import hashlib
import json
from pathlib import Path
import re
import shutil

import numpy as np
from tqdm import tqdm

from prepare_abdomenatlas3_table1 import ROOT, DATA, OUT as ORIGINAL, save_json, sha256, IMPRESSION_PATTERN

OUT = ROOT / "outputs/abdomenatlas3_mini/experiment_answer_mask_v2"
# 词干列举覆盖诊断同义词，不屏蔽正常器官名称、密度、大小、形态或HU值。
DIAGNOSIS = (r"(?:lesions?|masses|mass|cysts?|cystic|tumou?rs?|neoplas(?:m|ms|tic)|"
             r"(?:adeno|hepato(?:cellular)?|cholangiocellular|cholangio|renal.cell)?carcinomas?|"
             r"cancers?|malignan(?:cy|cies|t)|benign|metasta(?:sis|ses|tic)|"
             r"nodules?|hemangiomas?|haemangiomas?|adenomas?|"
             r"abnormalit(?:y|ies)|patholog(?:y|ies|ic|ical))")
ANSWER_WORDS = re.compile(r"\b" + DIAGNOSIS + r"\b", re.I)
ORGAN = r"(?:(?:left|right|bilateral)\s+)?(?:liver|hepatic|pancreas|pancreatic|kidneys?|renal)"
COUNT = r"(?:\d+|one|two|three|four|five|six|seven|eight|nine|ten|eleven|twelve|multiple|several|numerous|many)"
QUALIFIER = r"(?:focal|any|other|significant|obvious|suspicious|solid|cystic|discrete|new|additional|detectable|visible|hepatic|pancreatic|renal|liver|kidney)"
# 明确否定病灶存在的整段短语一并遮蔽，避免留下 no MASKTARGET 直接提示阴性。
NEGATIVE = re.compile(
    r"\b(?:no(?:\s+(?:evidence|signs?))?(?:\s+of)?|without(?:\s+(?:evidence|signs?))?(?:\s+of)?|"
    r"free\s+(?:of|from)|absence\s+of|negative\s+for)\s+"
    r"(?:" + QUALIFIER + r"\s+){0,5}" + DIAGNOSIS +
    r"(?:\s*(?:,|or|and)\s*(?:" + QUALIFIER + r"\s+){0,4}" + DIAGNOSIS + r")*"
    r"(?:\s+(?:are|were|is|was|have\s+been))?(?:\s+(?:identified|seen|noted|detected|present|found))?", re.I)
DIAGNOSIS_NEGATED = re.compile(
    r"\b(?:" + ORGAN + r"\s+)?" + DIAGNOSIS +
    r"\s+(?:(?:is|are|was|were)\s+)?(?:absent|not\s+(?:seen|identified|noted|detected|present|found))\b", re.I)
DIAGNOSIS_POSITIVE = re.compile(
    r"\b(?:" + ORGAN + r"\s+)?" + DIAGNOSIS +
    r"\s+(?:is|are|was|were)\s+(?:present|identified|seen|noted|detected|found)\b", re.I)
COUNTED_DIAGNOSIS = re.compile(
    r"\b(?:(?:a\s+)?total\s+of\s+)?" + COUNT + r"\s+(?:" + ORGAN + r"\s+)?" + DIAGNOSIS + r"\b", re.I)
ORGAN_DIAGNOSIS = re.compile(r"\b" + ORGAN + r"\s+" + DIAGNOSIS + r"\b", re.I)
# 补充包含并列器官或修饰词的否定形式，限定在同一语句内，不能跨越句号。
NEGATIVE_COMPLEX = re.compile(
    r"\b(?:no|without|absence\s+of|free\s+(?:of|from)|negative\s+for)\s+"
    r"(?:(?!" + DIAGNOSIS + r"\b)[A-Za-z-]+(?:\s+|,\s*)){0,10}" + DIAGNOSIS +
    r"(?:\s*(?:,|or|and)\s*(?:" + QUALIFIER + r"\s+){0,4}" + DIAGNOSIS + r")*"
    r"(?:\s+(?:are|were|is|was))?(?:\s+(?:identified|seen|noted|detected|present|found))?", re.I)
RULES = [("阴性诊断短语", NEGATIVE), ("复杂阴性诊断短语", NEGATIVE_COMPLEX), ("后置阴性判断", DIAGNOSIS_NEGATED),
         ("明确阳性判断", DIAGNOSIS_POSITIVE), ("病灶数量及诊断词", COUNTED_DIAGNOSIS),
         ("器官与诊断词组合", ORGAN_DIAGNOSIS), ("诊断词及同义词", ANSWER_WORDS)]


def mask_findings(text):
    findings = IMPRESSION_PATTERN.split(text, maxsplit=1)[0].strip()
    hits = []
    for name, pattern in RULES:
        def replace(match):
            hits.append({"rule": name, "text": match.group(0)})
            return " MASKTARGET "
        findings = pattern.sub(replace, findings)
    findings = re.sub(r"(?:\bMASKTARGET\s*){2,}", "MASKTARGET ", findings)
    return re.sub(r"\s+", " ", findings).strip(), hits


def main():
    original_rows = json.loads((ORIGINAL / "samples.json").read_text())
    with (DATA / "AbdomenAtlas3.0MiniWithMeta.csv").open(newline="") as stream:
        raw = {r["BDMAP ID"]: r for r in csv.DictReader(stream)}
    rows, examples, counts, words, lengths = [], [], collections.Counter(), collections.Counter(), []
    changed = 0
    for old in tqdm(original_rows, desc="遮蔽直接答案表达"):
        text, hits = mask_findings(raw[old["exam_id"]]["narrative report"])
        assert text and not ANSWER_WORDS.search(text), old["exam_id"]
        assert len(re.findall(r"[A-Za-z]+", text.replace("MASKTARGET", ""))) >= 10
        rows.append({**old, "findings_masked": text, "mask_hits": [h["text"] for h in hits],
                     "mask_version": "answer_words_v2"})
        changed += text != old["findings_masked"]
        counts.update(h["rule"] for h in hits)
        words.update(h["text"].lower() for h in hits)
        lengths.append(len(text.split()))
        if len(examples) < 12 and text != old["findings_masked"]:
            examples.append({"exam_id": old["exam_id"], "before": old["findings_masked"],
                             "after": text, "masked_spans": hits})
    assert [(r["exam_id"], r["labels"]) for r in rows] == [(r["exam_id"], r["labels"]) for r in original_rows]
    mask_protocol = {"version": "answer_words_v2", "rules": {name: p.pattern for name,p in RULES},
        "remove_impression": True, "mask_uses_individual_labels": False,
        "preserve": "尺寸、HU、密度、形态、器官结构及位置等所见信息",
        "scope": "直接诊断词、同义词、器官诊断词组合、明确病灶有无和数量短语",
        "limit": "文字掩码不保证消除由合成报告模板及标注来源产生的全部间接标签线索",
        "original_samples_sha256": sha256(ORIGINAL / "samples.json"), "source_sha256": sha256(Path(__file__))}
    mask_protocol["masking_sha256"] = hashlib.sha256(json.dumps(mask_protocol, sort_keys=True).encode()).hexdigest()
    oldprep = json.loads((ORIGINAL / "preparation_protocol.json").read_text())
    prep = {**oldprep, "mask_pattern": ANSWER_WORDS.pattern, "masking_protocol": mask_protocol,
            "feature_input_dir": str(ORIGINAL), "feature_preparation_sha256": oldprep["preparation_sha256"]}
    prep.pop("preparation_sha256")
    prep["preparation_sha256"] = hashlib.sha256(json.dumps(prep, sort_keys=True).encode()).hexdigest()
    if (OUT / "samples.json").exists():
        assert json.loads((OUT / "samples.json").read_text()) == rows, "已有v2文本不同，禁止覆盖"
    save_json(OUT / "samples.json", rows)
    save_json(OUT / "preparation_protocol.json", prep)
    save_json(OUT / "masking_protocol.json", mask_protocol)
    shutil.copy2(ORIGINAL / "patient_folds.json", OUT / "patient_folds.json")
    save_json(OUT / "mask_audit.json", {"cases": len(rows), "changed_cases": changed,
        "rule_hits": dict(counts), "frequent_masked_spans": words.most_common(40),
        "remaining_direct_answer_words": 0, "word_length_quantiles": np.quantile(lengths, [0,.5,.9,1]).tolist()})
    save_json(OUT / "mask_examples.json", examples)
    lines = ["# 答案词掩码示例", "", "规则统一应用于全部报告，不依赖单例标签；图像、标签和五折划分保持一致。", ""]
    for e in examples[:5]:
        lines += [f"## {e['exam_id']}", "", "**掩码前（上一版输入）**", e["before"], "", "**掩码后**", e["after"], ""]
    (OUT / "mask_examples.md").write_text("\n".join(lines)+"\n", encoding="utf-8")
    print(f"更新{changed}/{len(rows)}份报告；直接答案词扫描残留0，保留原图像缓存和五折。", flush=True)


if __name__ == "__main__":
    main()
