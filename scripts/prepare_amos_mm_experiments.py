#!/usr/bin/env python3
"""AMOS-MM 三/七标签共用数据协议、已知标签掩码和冻结轴位特征。"""
from __future__ import annotations

import argparse
import collections
import csv
import hashlib
import json
import os
from pathlib import Path
import re
import sys
import time

import numpy as np
from tqdm import tqdm

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
DATA = ROOT / "datasets/amos_mm"
AUDIT = ROOT / "outputs/amos_mm/leavs_label_audit"
OUT = ROOT / "outputs/amos_mm/experiment"
LABELS = ["liver", "kidney", "gallbladder", "spleen", "bowel", "pancreas", "stomach"]
WINDOWS = [(40., 400.), (60., 150.), (40., 800.)]
# 事先固定，适用于全部病例和两种任务；保留位置、形态及密度等所见信息。
DIAGNOSIS = re.compile(
    r"\b(?:cysts?|cystic|tumou?rs?|carcinoma\w*|cancer\w*|neoplasm\w*|"
    r"metasta\w*|cirrho\w*|steatosis|fatty liver|hepatitis|hepatomegaly|"
    r"splenomegaly|pancreatitis|cholecystitis|cholelithiasis|choledocholithiasis|"
    r"nephrolithiasis|urolithiasis|hydronephrosis|gastritis|enteritis|colitis|"
    r"diverticul\w*|hemangioma\w*|adenoma\w*|calculi|calculus|stones?|"
    r"cholecystectomy|nephrectomy|splenectomy|gastrectomy|pancreatectomy)\b", re.I)


def save_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(f".{os.getpid()}.tmp")
    temp.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")
    temp.replace(path)


def sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024**2), b""):
            digest.update(block)
    return digest.hexdigest()


def table(name):
    with (AUDIT / name).open(encoding="utf-8-sig", newline="") as stream:
        rows = list(csv.DictReader(stream))
    assert len(rows) == len({r["scan_id"] for r in rows})
    return {r["scan_id"]: r for r in rows}


def prepare():
    from run_physionet_ct_ich_table2_baselines import multilabel_folds
    source = DATA / "report_generation_train_val.json"
    official = json.loads(source.read_text())
    records = {Path(r["image"]).name.split(".")[0]: r
               for split in ("training", "validation") for r in official[split]}
    evidence = table("hybrid_1687_evidence_states.csv")
    human = table("human_test_200_evidence_states.csv")
    automatic = table("auto_development_1487_evidence_states.csv")
    assert len(records) == len(evidence) == 1687 and len(human) == 200
    assert set(human).isdisjoint(automatic) and set(human) | set(automatic) == set(records) == set(evidence)
    verified = {r["scan_id"]: r for r in map(json.loads,
                (ROOT / "outputs/amos_mm/download/verified_files.jsonl").read_text().splitlines())}
    rows, image_hashes, report_hashes, hits = [], collections.defaultdict(list), collections.defaultdict(list), collections.Counter()
    for index, scan in enumerate(tqdm(sorted(records), desc="核对检查、报告及已知标签")):
        record, states = records[scan], evidence[scan]
        assert all(states[k] in {"positive", "negative_supported", "uncertain", "not_stated"} for k in LABELS)
        assert all(states[k] == (human if scan in human else automatic)[scan][k] for k in LABELS)
        findings = record["labels"]["report"]["findings"]
        assert isinstance(findings, dict) and set(findings) <= {"chest", "abdomen", "pelvis"}
        text = " ".join(f"{region.capitalize()}: {findings[region]}" for region in ("chest", "abdomen", "pelvis")
                        if isinstance(findings.get(region), str) and findings[region].strip())
        text = re.sub(r"\s+", " ", text).strip()
        assert text and (DATA / record["image"]).is_file()
        matched = [m.group().lower() for m in DIAGNOSIS.finditer(text)]
        hits.update(matched)
        masked = DIAGNOSIS.sub("MASKTARGET", text)
        assert DIAGNOSIS.search(masked) is None
        rows.append({"case_index": index, "scan_id": scan, "file_path": record["image"],
                     "label_source": "human_test" if scan in human else "automatic_development",
                     "labels": [int(states[k] == "positive") for k in LABELS],
                     "known_mask": [states[k] in {"positive", "negative_supported"} for k in LABELS],
                     "evidence_states": [states[k] for k in LABELS],
                     "findings_masked": masked, "mask_hits": matched})
        image_hashes[verified[scan]["sha256"]].append(scan)
        report_hashes[hashlib.sha256(text.encode()).hexdigest()].append(scan)
    duplicate_images = [v for v in image_hashes.values() if len(v) > 1]
    duplicate_reports = [v for v in report_hashes.values() if len(v) > 1]
    if duplicate_images or duplicate_reports:
        raise ValueError("发现完全重复检查或原始所见；须先确定分组划分，停止自动训练")
    y = np.asarray([r["labels"] for r in rows], dtype=np.int64)
    known = np.asarray([r["known_mask"] for r in rows], dtype=bool)
    dev = np.asarray([r["case_index"] for r in rows if r["label_source"] == "automatic_development"])
    test = np.asarray([r["case_index"] for r in rows if r["label_source"] == "human_test"])
    assert len(dev) == 1487 and known[test].all() and (y[~known] == 0).all()
    # 正阳性与明确阴性同时分层；同一开发折服务三标签及七标签实验。
    indicators = np.concatenate([y[dev], known[dev] & (y[dev] == 0)], axis=1)
    groups = [dev[group].tolist() for group in multilabel_folds(indicators, 5, 42)]
    assert sorted(i for g in groups for i in g) == dev.tolist()
    inputs = [source, AUDIT / "hybrid_1687_evidence_states.csv",
              AUDIT / "human_test_200_evidence_states.csv", AUDIT / "auto_development_1487_evidence_states.csv"]
    protocol = {"cases": 1687, "labels": LABELS, "tasks": {"3": LABELS[:3], "7": LABELS},
                "development": 1487, "test": 200, "seed": 42,
                "split": "1487例开发数据分五折，每次四折训练一折验证；200例人工标注固定独立测试",
                "unit": "检查ID；公开资料未提供可核实的跨检查患者ID",
                "unknown_policy": "uncertain和not_stated按标签屏蔽，包括辅助监督；绝不填充为阴性监督",
                "text": "只使用官方findings的chest/abdomen/pelvis；不输入impression、QA或标签表",
                "text_limit": 512, "diagnosis_pattern": DIAGNOSIS.pattern,
                "text_caveat": "标签源于同份报告，掩码降低直接诊断词捷径但不等价于独立标签泄漏审计",
                "slices": 64, "orientation": "RAS，轴位从下至上均匀取样，保留原层索引及原长度",
                "windows_level_width": WINDOWS, "image_size": 224,
                "backbone": "冻结ImageNet ConvNeXt-Tiny的768维特征；所有图像方法共用",
                "weights_sha256": sha256(ROOT / "pre_weights/checkpoints/convnext_tiny-983f1562.pth"),
                "source_sha256": {str(p): sha256(p) for p in inputs},
                "preparation_source_sha256": sha256(Path(__file__)),
                "duplicate_image_sha256_groups": duplicate_images, "duplicate_findings_groups": duplicate_reports}
    protocol["preparation_sha256"] = hashlib.sha256(json.dumps(protocol, sort_keys=True).encode()).hexdigest()
    path = OUT / "preparation_protocol.json"
    if path.exists() and json.loads(path.read_text()) != protocol:
        raise ValueError("准备协议不同，禁止覆盖既有正式实验")
    save_json(path, protocol)
    save_json(OUT / "samples.json", rows)
    save_json(OUT / "splits.json", {"development": dev.tolist(), "test": test.tolist(), "validation_folds": groups,
              "validation_positive_counts": [y[g].sum(0).tolist() for g in groups],
              "validation_negative_counts": [(known[g] & (y[g] == 0)).sum(0).tolist() for g in groups]})
    save_json(OUT / "audit.json", {"labels": LABELS, "positive_counts": y.sum(0).tolist(),
              "known_counts": known.sum(0).tolist(), "development_positive_counts": y[dev].sum(0).tolist(),
              "test_positive_counts": y[test].sum(0).tolist(), "mask_terms": dict(hits),
              "test_copositive_counts": {str(c): int((y[test, :c].sum(1) >= 2).sum()) for c in (3, 7)},
              "development_no_known_targets": {str(c): int((known[dev, :c].sum(1) == 0).sum()) for c in (3, 7)},
              "findings_over_512_tokens": sum(len(re.findall(r"[a-z0-9]+", r["findings_masked"].lower())) > 512 for r in rows),
              "fold_sizes": [len(g) for g in groups]})
    print("1687例检查配对及两个任务的统一五折开发协议已固定。", flush=True)


def features(limit=None):
    import nibabel as nib
    import torch
    import torch.nn.functional as F
    from torchvision import models
    torch.set_num_threads(2)
    torch.hub.set_dir(str(ROOT / "pre_weights"))
    model = models.convnext_tiny(weights=models.ConvNeXt_Tiny_Weights.IMAGENET1K_V1)
    encoder = torch.nn.Sequential(model.features, model.avgpool, model.classifier[0], torch.nn.Flatten(1)).eval().cuda()
    del model
    mean = torch.tensor([.485, .456, .406], device="cuda")[None, :, None, None]
    std = torch.tensor([.229, .224, .225], device="cuda")[None, :, None, None]
    digest = json.loads((OUT / "preparation_protocol.json").read_text())["preparation_sha256"]
    rows = json.loads((OUT / "samples.json").read_text())
    if limit:
        rows = rows[:limit]
    folder = OUT / "features"
    folder.mkdir(parents=True, exist_ok=True)
    begin = time.monotonic()
    for step, row in enumerate(tqdm(rows, desc="提取AMOS-MM全量64层特征"), 1):
        path = folder / f"{row['case_index']:04d}.npz"
        if path.exists():
            with np.load(path, allow_pickle=False) as c:
                assert str(c["preparation_sha256"]) == digest and str(c["scan_id"]) == row["scan_id"]
                assert c["features"].shape == (len(c["slice_indices"]), 768) and np.isfinite(c["features"]).all()
        else:
            image = nib.load(str(DATA / row["file_path"]))
            assert len(image.shape) == 3 and np.isfinite(image.affine).all()
            data = np.asarray(image.dataobj, dtype=np.float32)
            orient = nib.orientations.ornt_transform(nib.orientations.io_orientation(image.affine),
                                                    nib.orientations.axcodes2ornt(("R", "A", "S")))
            data = nib.orientations.apply_orientation(data, orient)
            total = data.shape[2]
            indices = np.unique(np.linspace(0, total - 1, min(64, total)).round().astype(np.int64))
            slices = np.ascontiguousarray(data[::-1, ::-1, indices].transpose(2, 1, 0))
            del data
            assert np.isfinite(slices).all()
            values_out = []
            with torch.inference_mode():
                for start in range(0, len(slices), 16):
                    values = torch.from_numpy(slices[start:start + 16]).cuda()[:, None]
                    channels = torch.cat([((values - (l - w / 2)) / w).clamp(0, 1) for l, w in WINDOWS], dim=1)
                    channels = F.interpolate(channels, (224, 224), mode="bilinear", align_corners=False, antialias=True)
                    with torch.autocast("cuda"):
                        feat = encoder((channels - mean) / std)
                    values_out.append(feat.float().cpu().numpy())
            result = np.concatenate(values_out)
            assert result.shape == (len(indices), 768) and np.isfinite(result).all()
            temp = path.with_suffix(".tmp.npz")
            np.savez_compressed(temp, features=result, slice_indices=indices, original_count=total,
                                scan_id=row["scan_id"], preparation_sha256=digest,
                                hu_quantiles=np.quantile(slices, [0, .01, .5, .99, 1]))
            temp.replace(path)
        save_json(OUT / "feature_status.json", {"state": "running", "completed": step, "total": len(rows),
                  "scan_id": row["scan_id"], "elapsed_seconds": time.monotonic() - begin, "updated_at": time.time()})
    save_json(OUT / "feature_status.json", {"state": "complete" if not limit else "smoke_complete",
              "completed": len(rows), "total": 1687, "elapsed_seconds": time.monotonic() - begin,
              "updated_at": time.time(), "preparation_sha256": digest})


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--features", action="store_true")
    parser.add_argument("--limit", type=int)
    args = parser.parse_args()
    features(args.limit) if args.features else prepare()
