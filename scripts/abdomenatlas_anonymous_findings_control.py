#!/usr/bin/env python3
"""病灶所见去除器官归属的独立对照，保留客观尺寸和密度，不改正式训练输入。"""
from __future__ import annotations
import argparse
import csv
import hashlib
import json
import re
from pathlib import Path

import numpy as np
from tqdm import tqdm

import abdomenatlas_background_text_control as base
from prepare_abdomenatlas3_table1 import save_json

BACKGROUND = base.OUT
OUT = base.ROOT / "outputs/abdomenatlas3_mini/anonymous_findings_control"
NUMBER = r"[+-]?\d+(?:\.\d+)?"
HEAD = re.compile(r"(?mi)^(?:Liver|Pancreas|Kidney) lesion \d+:[ \t]*$")


def prepare():
    rows = json.loads((BACKGROUND / "samples.json").read_text())
    with (base.ROOT / "datasets/abdomenatlas3/AbdomenAtlas3.0MiniWithMeta.csv").open(newline="") as stream:
        official = {r["BDMAP ID"]: r for r in csv.DictReader(stream)}
    lengths, counts = [], []
    for row in tqdm(rows, desc="构建无器官归属的病灶所见"):
        text = official[row["exam_id"]]["structured report"]
        findings = []
        for match in HEAD.finditer(text):
            block = re.split(r"\n\s*\n", text[match.end():].lstrip("\r\n"), maxsplit=1)[0].strip()
            size = re.search(r"Size:\s*("+NUMBER+r")\s*x\s*("+NUMBER+r")\s*cm", block, re.I)
            volume = re.search(r"Volume:\s*("+NUMBER+r")\s*cc", block, re.I)
            density = re.search(r"Enhancement relative to [^:]+:\s*(\w+)\s*\(HU value is\s*("+NUMBER+r")\s*\+/-\s*("+NUMBER+r")\)", block, re.I)
            assert size and volume and density, (row["exam_id"], block)
            description = (f"Focal finding: dimensions {size[1]} by {size[2]} cm; volume {volume[1]} cc; "
                           f"attenuation {density[1].lower()}; mean HU {density[2]} with standard deviation {density[3]}.")
            # 根据病灶本身的测量排序，去掉原报告按器官分组的固定顺序。
            key = (max(float(size[1]), float(size[2])), float(volume[1]), description)
            findings.append((key, description))
        findings.sort(reverse=True)
        # 最多六条观测，以相同规则控制文本长度；不读取个体标签决定保留内容。
        selected = [description for _,description in findings[:6]]
        row["findings_masked"] += " " + " ".join(selected)
        row["findings_masked"] = row["findings_masked"].strip()
        row["text_source"] = "固定基础所见加无器官归属的客观病灶描述"
        row["mask_version"] = "anonymous_findings_control"
        lengths.append(len(base.tokenize(row["findings_masked"])))
        counts.append(len(findings))
    assert max(lengths) <= 512, max(lengths)
    save_json(OUT / "samples.json", rows)
    save_json(OUT / "split_ids.json", json.loads((BACKGROUND / "split_ids.json").read_text()))
    save_json(OUT / "protocol.json", {"scope": "独立诊断对照，不自动替代正式表1结果",
        "source": "官方structured report；固定基础所见与病灶尺寸、体积、密度、HU",
        "removed": "病灶器官归属、左右位置、肝分段、胰头体尾、图像编号、血管关系、分期、诊断结论",
        "order": "四器官基础所见统一顺序；病灶不分器官，按最大径、体积、描述排序，最多六条",
        "label_usage": "规则不依据个体标签；标签仅用于监督。匿名化保留病灶存在与客观影像特征，不声称去掉全部标签相关信息。",
        "case_count": len(rows), "token_quantiles": np.quantile(lengths,[0,.5,.9,1]).tolist(),
        "evaluation": "原第一折训练/验证；不读取测试折进行本轮决策；超参数与前面对照一致",
        "source_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest()})
    (OUT / "examples.md").write_text("# 无器官归属的病灶所见示例\n\n"+"\n\n".join(
        f"## {r['exam_id']}\n\n{r['findings_masked']}" for r in rows[:5])+"\n", encoding="utf-8")
    print(f"已构建{len(rows)}例，文本长度分位数{np.quantile(lengths,[0,.5,.9,1]).tolist()}", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model', choices=['textcnn_encoder','vocab_attention_encoder'])
    args = parser.parse_args()
    if args.model:
        base.OUT = OUT
        base.run(args.model)
    else:
        prepare()
