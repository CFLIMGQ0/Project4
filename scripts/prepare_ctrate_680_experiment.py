#!/usr/bin/env python3
"""固定CT-RATE三标签、所见掩码、患者五折与64层冻结视觉特征。"""

from __future__ import annotations

import argparse
import collections
import csv
import hashlib
import json
import os
from pathlib import Path
import re
import subprocess
import sys
import time

import numpy as np
from tqdm import tqdm

ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "datasets/ct_rate_680"
OUT = ROOT / "outputs/ct_rate_680/experiment"
LABELS = ["Emphysema", "Atelectasis", "Pulmonary fibrotic sequela"]
# 词典在训练前固定，对所有报告采用相同规则，不依据该例标签决定掩码。
TARGET_PATTERN = re.compile(
    r"\b(?:emphysema\w*|emphysematous|atelecta\w*|fibro[- ]?atelecta\w*|"
    r"fibrosis|fibroses|fibrotic|fibro[- ]?sequel\w*)\b", re.I)
WINDOWS = [(-600.0, 1500.0), (40.0, 400.0), (-600.0, 700.0)]


def save_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")
    tmp.replace(path)


def sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024**2), b""):
            digest.update(block)
    return digest.hexdigest()


def read_samples():
    with (DATA / "samples_680.csv").open(encoding="utf-8-sig", newline="") as stream:
        samples = list(csv.DictReader(stream))
    assert len(samples) == len({r["patient_id"] for r in samples}) == 680
    return sorted(samples, key=lambda r: r["patient_id"])


def prepare_tables():
    from run_physionet_ct_ich_table2_baselines import multilabel_folds
    samples = read_samples()
    y = np.asarray([[int(r[label]) for label in LABELS] for r in samples], dtype=np.int64)
    assert y.sum(0).tolist() == [148, 169, 188]
    assert [int((y.sum(1) == n).sum()) for n in range(4)] == [305, 259, 102, 14]
    groups = multilabel_folds(y, 5, 42)
    rows, hits = [], collections.Counter()
    for index, row in enumerate(tqdm(samples, desc="配对报告并掩码目标词")):
        text = re.sub(r"\s+", " ", row["Findings_EN"]).strip()
        matched = [m.group(0).lower() for m in TARGET_PATTERN.finditer(text)]
        hits.update(matched)
        masked = TARGET_PATTERN.sub(" MASKTARGET ", text)
        masked = re.sub(r"\s+", " ", masked).strip()
        assert masked and TARGET_PATTERN.search(masked) is None
        rows.append({"case_index": index, "patient_id": row["patient_id"],
                     "exam_id": row["exam_id"], "file_path": row["file_path"],
                     "labels": y[index].tolist(), "findings_masked": masked,
                     "mask_hits": matched})
    assert len(set(np.concatenate(groups))) == 680
    payload = {"labels": LABELS, "patient_ids": list(range(680)),
               "patient_id_mapping": {str(r["case_index"]): r["patient_id"] for r in rows},
               "positive_counts": y.sum(0).tolist(),
               "folds": [group.tolist() for group in groups],
               "fold_positive_counts": [y[group].sum(0).tolist() for group in groups],
               "fold_co_positive_counts": [int((y[group].sum(1) >= 2).sum()) for group in groups],
               "seed": 42, "split": "测试第k折，验证第k+1折，其余三折训练"}
    parameters = {"cases": 680, "labels": LABELS, "slices_per_case": 64,
                  "sampling": "完整轴位序列从下至上均匀采样，保留原层索引和原长度",
                  "orientation": "RAS标准化后按固定轴位显示方向取图",
                  "windows_level_width": WINDOWS, "image_size": 224,
                  "backbone": "ImageNet ConvNeXt-Tiny；冻结768维特征，共用于所有图像方法",
                  "weights_sha256": sha256(ROOT / "pre_weights/checkpoints/convnext_tiny-983f1562.pth"),
                  "intensity": "读取fixed版本NIfTI已修正强度，不重复应用DICOM截距",
                  "text_field": "Findings_EN", "excluded_fields": ["Impressions_EN", "ClinicalInformation_EN", "Technique_EN"],
                  "target_pattern": TARGET_PATTERN.pattern, "mask_token": "MASKTARGET",
                  "text_max_length": 512, "raw_reports_modified": False,
                  "samples_sha256": sha256(DATA / "samples_680.csv"),
                  "preparation_source_sha256": sha256(Path(__file__))}
    digest = hashlib.sha256(json.dumps(parameters, sort_keys=True).encode()).hexdigest()
    parameters["preparation_sha256"] = digest
    prior = OUT / "preparation_protocol.json"
    if prior.exists() and json.loads(prior.read_text())["preparation_sha256"] != digest:
        raise ValueError("已有缓存采用其他准备协议，停止以避免覆盖")
    save_json(prior, parameters)
    save_json(OUT / "samples.json", rows)
    save_json(OUT / "patient_folds.json", payload)
    lengths = [len(re.findall(r"[a-z0-9]+", r["findings_masked"].lower())) for r in rows]
    save_json(OUT / "text_audit.json", {"cases": 680, "input_field": "Findings_EN",
              "mask_hit_cases": sum(bool(r["mask_hits"]) for r in rows),
              "mask_terms": dict(hits), "residual_target_pattern_hits": 0,
              "token_length_quantiles": np.quantile(lengths, [0, .5, .9, 1]).tolist(),
              "over_512_tokens": sum(n > 512 for n in lengths),
              "mask_rule_uses_patient_labels": False, "clinical_reports_are_independent_of_image_models": True,
              "label_provenance": "人工报告标注；所见掩码用于降低直接类别词泄漏，不代表完成独立影像标签审计"})
    print("680例报告掩码及统一患者五折已准备。", flush=True)


def feature_worker(index, devices, limit=None):
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
    protocol = json.loads((OUT / "preparation_protocol.json").read_text())
    digest = protocol["preparation_sha256"]
    rows = json.loads((OUT / "samples.json").read_text())
    jobs = [row for row in rows if row["case_index"] % len(devices) == index]
    if limit is not None:
        jobs = jobs[:limit]
    folder = OUT / "features"
    folder.mkdir(parents=True, exist_ok=True)
    for step, row in enumerate(tqdm(jobs, desc=f"GPU{devices[index]}提取64层特征"), 1):
        path = folder / f"{row['case_index']:04d}.npz"
        if path.exists():
            with np.load(path, allow_pickle=False) as cache:
                assert str(cache["preparation_sha256"]) == digest
                assert cache["features"].shape == (len(cache["slice_indices"]), 768)
                assert np.isfinite(cache["features"]).all()
            continue
        begin = time.monotonic()
        image = nib.load(str(DATA / row["file_path"]))
        if len(image.shape) != 3 or not np.isfinite(image.affine).all():
            raise ValueError(f"NIfTI维度或空间矩阵异常：{row['patient_id']}")
        data = np.asarray(image.dataobj, dtype=np.float32)
        orient = nib.orientations.ornt_transform(nib.orientations.io_orientation(image.affine),
                                                nib.orientations.axcodes2ornt(("R", "A", "S")))
        data = nib.orientations.apply_orientation(data, orient)
        total = data.shape[2]
        indices = np.unique(np.linspace(0, total-1, min(64, total)).round().astype(np.int64))
        slices = np.ascontiguousarray(data[::-1, ::-1, indices].transpose(2, 1, 0))
        del data
        if not np.isfinite(slices).all():
            raise ValueError(f"影像强度存在非有限数：{row['patient_id']}")
        features = []
        with torch.inference_mode():
            for start in range(0, len(slices), 16):
                values = torch.from_numpy(slices[start:start+16]).cuda()[:, None]
                # 先在原分辨率窗变换，再缩放；三窗通道对所有模型保持一致。
                channels = torch.cat([((values-(level-width/2))/width).clamp(0, 1)
                                      for level, width in WINDOWS], dim=1)
                channels = F.interpolate(channels, size=(224, 224), mode="bilinear", align_corners=False, antialias=True)
                with torch.autocast("cuda"):
                    feat = encoder((channels-mean)/std)
                features.append(feat.float().cpu().numpy())
        features = np.concatenate(features)
        assert features.shape == (len(indices), 768) and np.isfinite(features).all()
        temp = path.with_suffix(".tmp.npz")
        np.savez_compressed(temp, features=features, slice_indices=indices,
                            original_count=total, patient_id=row["patient_id"],
                            preparation_sha256=digest, hu_quantiles=np.quantile(slices, [0, .01, .5, .99, 1]))
        temp.replace(path)
        save_json(OUT / f"feature_worker_{index}.json", {"status": "running", "done": step,
                  "total": len(jobs), "last_patient": row["patient_id"],
                  "last_seconds": time.monotonic()-begin})
    save_json(OUT / f"feature_worker_{index}.json", {"status": "complete", "done": len(jobs), "total": len(jobs)})


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tables-only", action="store_true")
    parser.add_argument("--devices", type=int, nargs="+", default=[0, 1, 2, 3])
    parser.add_argument("--worker-index", type=int)
    parser.add_argument("--limit", type=int)
    args = parser.parse_args()
    if args.worker_index is not None:
        feature_worker(args.worker_index, args.devices, args.limit)
        return
    prepare_tables()
    if args.tables_only:
        return
    handles, workers = [], []
    for index, device in enumerate(args.devices):
        env = os.environ.copy()
        env.update(CUDA_VISIBLE_DEVICES=str(device), OMP_NUM_THREADS="2")
        command = [sys.executable, "-u", str(Path(__file__).resolve()), "--worker-index", str(index),
                   "--devices", *map(str, args.devices)]
        if args.limit is not None:
            command += ["--limit", str(args.limit)]
        handle = (OUT / f"feature_worker_{index}.log").open("a")
        handles.append(handle)
        workers.append(subprocess.Popen(command, env=env, stdout=handle, stderr=subprocess.STDOUT))
    while any(p.poll() is None for p in workers):
        count = len(list((OUT / "features").glob("*.npz")))
        state = {"status": "running", "completed_cases": count, "expected_cases": 680,
                 "worker_pids": [p.pid for p in workers], "updated_unix": time.time()}
        save_json(OUT / "feature_state.json", state)
        print(f"已生成特征缓存：{count}/680", flush=True)
        time.sleep(10)
    codes = [p.returncode for p in workers]
    count = len(list((OUT / "features").glob("*.npz")))
    state.update(status="complete" if count == 680 and not any(codes) else "incomplete",
                 completed_cases=count, worker_exit_codes=codes, finished_unix=time.time())
    save_json(OUT / "feature_state.json", state)
    for handle in handles:
        handle.close()
    if any(codes):
        raise SystemExit(1)
    print(f"特征准备结束：{count}/680", flush=True)


if __name__ == "__main__":
    main()
