#!/usr/bin/env python3
"""仅利用公开目录元数据，测量选定680例CT-RATE压缩影像体积。"""

import csv
import hashlib
import io
import json
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import requests

ROOT = Path(__file__).resolve().parents[2]
REVISION = "deeca4d89e9f978d4d1bccd88a55071ddbb146bb"
LABELS = ["Emphysema", "Atelectasis", "Pulmonary fibrotic sequela"]
OUTPUT = ROOT / "outputs/dataset_candidates/ct_rate_680_file_sizes.json"


def load_annotation(name):
    url = "https://raw.githubusercontent.com/ibrahimethemhamamci/CT-CLIP/main/text_classifier/data/" + name
    response = requests.get(url, timeout=30)
    response.raise_for_status()
    return name, {
        "rows": list(csv.DictReader(io.StringIO(response.text))),
        "url": url, "sha256": hashlib.sha256(response.content).hexdigest(),
    }


def list_sizes(split):
    """公开tree接口只返回文件名和字节数，不访问受控影像或标签内容。"""
    session = requests.Session()
    url = (f"https://huggingface.co/api/datasets/ibrahimhamamci/CT-RATE/tree/{REVISION}/"
           f"dataset/{split}_fixed?recursive=true&limit=1000")
    files, page = {}, 0
    while url:
        response = session.get(url, timeout=45)
        response.raise_for_status()
        for item in response.json():
            if item["type"] == "file" and item["path"].endswith(".nii.gz"):
                files[item["path"]] = item["size"]
        page += 1
        url = response.links.get("next", {}).get("url")
        if page % 20 == 0 or not url:
            print(f"{split}公开清单：{page}页，已核对{len(files)}个文件的大小", flush=True)
    return files


def main():
    names = ["train.csv", "val.csv", "map_train_classifier.csv", "map_val_classifier.csv"]
    with ThreadPoolExecutor(4) as pool:
        annotations = dict(pool.map(load_annotation, names))
    by_exam = {}
    for split in ("train", "val"):
        mapping = {r["AccessionNo"]: r["NameinCTRATE"]
                   for r in annotations[f"map_{split}_classifier.csv"]["rows"]}
        for row in annotations[split + ".csv"]["rows"]:
            exam = mapping[row["AccessionNo"]]
            if exam.lower() == "not in ct-rate":
                continue
            targets = [int(float(row[label])) for label in LABELS]
            if exam in by_exam:
                assert by_exam[exam] == targets
            by_exam[exam] = targets
    by_patient = {}
    for exam, targets in sorted(by_exam.items()):
        patient = "_".join(exam.split("_")[:2])
        by_patient.setdefault(patient, (exam, targets))
    assert len(by_patient) == 680
    positive_counts = [sum(r[1][j] for r in by_patient.values()) for j in range(3)]
    assert positive_counts == [148, 169, 188]
    print("已确认同一680例名单及三标签阳性数", positive_counts, flush=True)
    files = {}
    with ThreadPoolExecutor(2) as pool:
        for group in pool.map(list_sizes, ("train", "valid")):
            files.update(group)
    selected = []
    for patient, (exam, targets) in by_patient.items():
        split = exam.split("_")[0]
        path = f"dataset/{split}_fixed/{patient}/{exam}/{exam}_1.nii.gz"
        if path not in files:
            raise ValueError(f"选定病例缺少重建1：{exam}")
        selected.append({"patient_id": patient, "exam_id": exam, "labels": targets,
                         "file_path": path, "bytes": files[path]})
    total = sum(r["bytes"] for r in selected)
    result = {
        "repository": "ibrahimhamamci/CT-RATE", "revision": REVISION,
        "label_names": LABELS, "positive_counts": positive_counts,
        "patients": len(selected), "files": len(selected), "reconstruction": 1,
        "bytes": total, "GB": total / 1e9, "GiB": total / 2**30,
        "mean_file_MB": total / len(selected) / 1e6,
        "min_file_MB": min(r["bytes"] for r in selected) / 1e6,
        "max_file_MB": max(r["bytes"] for r in selected) / 1e6,
        "co_positive_counts": {str(k): sum(sum(r["labels"]) == k for r in selected) for k in range(4)},
        "scope": "680名患者，每人一项已人工标注检查，重建1；仅压缩CT文件，不含缓存和训练输出",
        "images_downloaded": False,
        "annotation_sources": {name: {k: v for k, v in item.items() if k != "rows"}
                               for name, item in annotations.items()},
        "selected_files": selected,
    }
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({k: v for k, v in result.items() if k not in ("selected_files", "annotation_sources")},
                     ensure_ascii=False, indent=2), flush=True)
    print(f"逐例文件清单已保存：{OUTPUT}", flush=True)


if __name__ == "__main__":
    main()
