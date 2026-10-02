#!/usr/bin/env python3
"""准备 AbdomenAtlas 3.0 Mini 三标签队列，流式读取压缩包并生成共用特征。"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
from pathlib import Path
import queue
import re
import shutil
import sys
import tarfile
import tempfile
import threading
import time
import traceback

import numpy as np
from tqdm import tqdm

ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "datasets/abdomenatlas3"
OUT = ROOT / "outputs/abdomenatlas3_mini/experiment"
LABELS = ["liver_lesion", "pancreatic_lesion", "kidney_lesion"]
COUNT_FIELDS = [f"number of {name} lesion instances" for name in ("liver", "pancreatic", "kidney")]
WINDOWS = [(-600., 1500.), (40., 400.), (-600., 700.)]
# 对全部病例应用同一词典；保留形态、尺寸、密度及解剖描述，不读取个体标签来决定掩码。
TARGET_PATTERN = re.compile(r"\b(?:lesions?|masses|mass|cysts?|cystic|tumou?rs?|neoplasms?|neoplastic|"
                            r"carcinomas?|cancers?|metastases|metastasis|metastatic|nodules?|"
                            r"hemangiomas?|haemangiomas?|adenomas?)\b", re.I)
IMPRESSION_PATTERN = re.compile(r"(?:\*\*)?\b(?:IMPRESSION|CONCLUSION|SUMMARY)\s*(?:\*\*)?\s*:", re.I)


def save_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + f".{os.getpid()}.tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def prepare_tables():
    sys.path.insert(0, str(ROOT / "src"))
    from run_physionet_ct_ich_table2_baselines import multilabel_folds
    metadata = DATA / "AbdomenAtlas3.0MiniWithMeta.csv"
    with metadata.open(newline="") as stream:
        raw = list(csv.DictReader(stream))
    manifest = json.loads((ROOT / "outputs/abdomenatlas_download_20260929/manifest.json").read_text())
    verified = json.loads((ROOT / "outputs/abdomenatlas_download_20260929/verified.json").read_text())
    archives = []
    for item in manifest:
        path = DATA / item["path"]
        stat = path.stat()
        v = verified[item["name"]]
        assert stat.st_size == item["size"] == v["size"]
        assert stat.st_mtime_ns == v["mtime_ns"] and item["sha256"] == v["sha256"]
        ids = re.findall(r"BDMAP_(\d{8})", item["name"])
        assert len(ids) == 2
        archives.append({"name": item["name"], "first": int(ids[0]), "last": int(ids[1]),
                         "size": item["size"], "sha256": item["sha256"]})
    rows, excluded, tokens = [], [], []
    for r in tqdm(raw, desc="构建检查级图文队列"):
        exam = r["BDMAP ID"]
        text = r["narrative report"].strip()
        if not text:
            excluded.append({"exam_id": exam, "reason": "缺少 narrative report"})
            continue
        findings = IMPRESSION_PATTERN.split(text, maxsplit=1)[0].strip()
        assert findings, exam
        hits = TARGET_PATTERN.findall(findings)
        masked = re.sub(r"\s+", " ", TARGET_PATTERN.sub(" MASKTARGET ", findings)).strip()
        assert not TARGET_PATTERN.search(masked)
        residual = re.findall(r"[A-Za-z]+", masked.replace("MASKTARGET", ""))
        assert len(residual) >= 10, (exam, "掩码后报告信息不足")
        counts = [float(r[k]) for k in COUNT_FIELDS]
        assert all(np.isfinite(v) and v >= 0 and v == int(v) for v in counts)
        number = int(exam.rsplit("_", 1)[1])
        matches = [a for a in archives if a["first"] <= number <= a["last"]]
        assert len(matches) == 1, exam
        rows.append({"case_index": len(rows), "exam_id": exam, "patient_id": exam,
                     "archive": matches[0]["name"], "file_path": exam + "/ct.nii.gz",
                     "labels": [int(v > 0) for v in counts], "lesion_instance_counts": counts,
                     "findings_masked": masked, "mask_hits": hits,
                     "label_source": "官方病灶实例计数", "text_source": "narrative report 的所见部分",
                     "metadata_shape": r["shape"], "metadata_spacing": r["spacing"]})
        tokens.append(len(masked.split()))
    assert len({r["exam_id"] for r in rows}) == len(rows)
    y = np.asarray([r["labels"] for r in rows], dtype=np.int64)
    groups = [v.tolist() for v in multilabel_folds(y, 5, 42)]
    folds = {"labels": LABELS, "folds": groups, "seed": 42,
             "group_key": "BDMAP ID；元数据未提供可核验的跨检查患者标识",
             "split": "每轮三折训练、一折验证、一折测试；每例恰好进入一次测试",
             "fold_positive_counts": [y[g].sum(0).tolist() for g in groups]}
    protocol = {"dataset": "AbdomenAtlas3.0Mini", "case_count": len(rows), "labels": LABELS,
                "metadata_sha256": sha256(metadata), "archives": archives,
                "label_rule": "各器官官方病灶实例数大于零；不从输入报告提取标签",
                "text_field": "narrative report", "remove_impression": True,
                "mask_pattern": TARGET_PATTERN.pattern, "mask_uses_individual_labels": False,
                "report_provenance": "官方合成叙述报告，病灶描述与标签共享原始标注来源；不视为独立临床诊断验证",
                "image": {"orientation": "RAS轴位，按原层索引递增", "max_slices": 64,
                          "size": [224, 224], "windows_level_width": WINDOWS,
                          "encoder": "冻结 ImageNet ConvNeXt-Tiny 768维；沿用已有公开数据集流程",
                          "weights_sha256": sha256(ROOT / "pre_weights/checkpoints/convnext_tiny-983f1562.pth")},
                "source_sha256": sha256(Path(__file__))}
    protocol["preparation_sha256"] = hashlib.sha256(json.dumps(protocol, sort_keys=True).encode()).hexdigest()
    existing = OUT / "preparation_protocol.json"
    if existing.exists():
        assert json.loads(existing.read_text()) == protocol, "已有准备协议不同，禁止混用缓存"
    save_json(existing, protocol)
    save_json(OUT / "samples.json", rows)
    save_json(OUT / "patient_folds.json", folds)
    save_json(OUT / "cohort_audit.json", {"metadata_cases": len(raw), "included": len(rows),
              "excluded": excluded, "positive_counts": y.sum(0).tolist(),
              "copositive_count": int((y.sum(1) >= 2).sum()), "all_negative": int((y.sum(1) == 0).sum()),
              "token_length_quantiles": np.quantile(tokens, [0, .5, .9, 1]).tolist(),
              "over_512_words": sum(t > 512 for t in tokens), "fold_sizes": list(map(len, groups))})
    print(f"队列已固定：{len(rows)}例，阳性数{y.sum(0).tolist()}，共阳性{(y.sum(1)>=2).sum()}例", flush=True)


def validate_cache(path, row, digest):
    with np.load(path, allow_pickle=False) as c:
        assert str(c["patient_id"]) == row["patient_id"]
        assert str(c["preparation_sha256"]) == digest
        x, pos, count = c["features"], c["slice_indices"], int(c["original_count"])
        assert x.shape == (len(pos), 768) and 1 <= len(pos) <= 64 and np.isfinite(x).all()
        assert np.all(np.diff(pos) > 0) and 0 <= pos[0] <= pos[-1] < count


def feature_worker(index, workers):
    import nibabel as nib
    import torch
    import torch.nn.functional as F
    from torchvision import models
    torch.set_num_threads(2)
    torch.cuda.set_per_process_memory_fraction(.42)
    torch.hub.set_dir(str(ROOT / "pre_weights"))
    network = models.convnext_tiny(weights=models.ConvNeXt_Tiny_Weights.IMAGENET1K_V1)
    encoder = torch.nn.Sequential(network.features, network.avgpool, network.classifier[0],
                                  torch.nn.Flatten(1)).eval().cuda()
    del network
    mean = torch.tensor([.485, .456, .406], device="cuda")[None, :, None, None]
    std = torch.tensor([.229, .224, .225], device="cuda")[None, :, None, None]
    protocol = json.loads((OUT / "preparation_protocol.json").read_text())
    digest = protocol["preparation_sha256"]
    rows = json.loads((OUT / "samples.json").read_text())
    archives = [a for i, a in enumerate(protocol["archives"]) if i % workers == index]
    assigned = {r["exam_id"]: r for r in rows if r["archive"] in {a["name"] for a in archives}}
    folder = OUT / "features"
    folder.mkdir(parents=True, exist_ok=True)
    work, results = queue.Queue(), queue.Queue(maxsize=2)
    for archive in archives:
        work.put(archive)
    started, done = time.time(), 0
    # 每卡两个CPU解码线程，逐病例处理，不展开整个570GB数据集。
    def producer():
        try:
            with tempfile.TemporaryDirectory(prefix=f"atlas_{index}_", dir=OUT) as temporary:
                while True:
                    try:
                        archive = work.get_nowait()
                    except queue.Empty:
                        break
                    expected = {r["exam_id"] for r in assigned.values() if r["archive"] == archive["name"]}
                    if all((folder / f"{assigned[e]['case_index']:04d}.npz").exists() for e in expected):
                        for exam in sorted(expected):
                            row = assigned[exam]
                            validate_cache(folder / f"{row['case_index']:04d}.npz", row, digest)
                            results.put(("cached", row))
                        continue
                    seen = set()
                    with tarfile.open(DATA / "image_only" / archive["name"], "r|gz") as tar:
                        for member in tar:
                            parts = Path(member.name).parts
                            if not member.isfile() or len(parts) < 2 or parts[-1] != "ct.nii.gz":
                                continue
                            exam = parts[-2]
                            if exam not in expected:
                                continue
                            assert exam not in seen, f"归档重复病例：{exam}"
                            seen.add(exam)
                            row = assigned[exam]
                            path = folder / f"{row['case_index']:04d}.npz"
                            if path.exists():
                                validate_cache(path, row, digest)
                                results.put(("cached", row))
                                continue
                            nii = Path(temporary) / "ct.nii.gz"
                            with tar.extractfile(member) as source, nii.open("wb") as dest:
                                shutil.copyfileobj(source, dest, 8 << 20)
                            image = nib.load(nii)
                            assert len(image.shape) == 3 and np.isfinite(image.affine).all(), exam
                            orientation = nib.orientations.ornt_transform(nib.orientations.io_orientation(image.affine),
                                            nib.orientations.axcodes2ornt(("R", "A", "S")))
                            volume = nib.orientations.apply_orientation(np.asarray(image.dataobj, dtype=np.float32), orientation)
                            count = volume.shape[2]
                            pos = np.unique(np.linspace(0, count - 1, min(64, count)).round().astype(np.int64))
                            slices = np.ascontiguousarray(volume[::-1, ::-1, pos].transpose(2, 1, 0))
                            assert np.isfinite(slices).all(), exam
                            del volume, image
                            nii.unlink()
                            results.put(("image", row, slices, pos, count))
                    assert seen == expected, (archive["name"], sorted(expected - seen))
        except BaseException:
            results.put(("error", traceback.format_exc()))
        finally:
            results.put(("end",))

    threads = [threading.Thread(target=producer, daemon=True) for _ in range(2)]
    for thread in threads:
        thread.start()
    ended = 0
    with tqdm(total=len(assigned), desc=f"特征任务{index}", mininterval=10) as bar:
        while ended < len(threads):
            item = results.get()
            if item[0] == "end":
                ended += 1
                continue
            if item[0] == "error":
                raise RuntimeError(item[1])
            row = item[1]
            if item[0] == "image":
                _, row, slices, pos, count = item
                features = []
                with torch.inference_mode():
                    for start in range(0, len(slices), 16):
                        values = torch.from_numpy(slices[start:start+16]).cuda()[:, None]
                        channels = torch.cat([((values - (level-width/2))/width).clamp(0, 1)
                                              for level, width in WINDOWS], dim=1)
                        channels = F.interpolate(channels, (224, 224), mode="bilinear", align_corners=False, antialias=True)
                        with torch.autocast("cuda"):
                            feature = encoder((channels-mean)/std)
                        features.append(feature.float().cpu().numpy())
                feature = np.concatenate(features)
                assert feature.shape == (len(pos), 768) and np.isfinite(feature).all()
                path = folder / f"{row['case_index']:04d}.npz"
                temporary = path.with_suffix(".tmp.npz")
                np.savez_compressed(temporary, features=feature, slice_indices=pos, original_count=count,
                                    patient_id=row["patient_id"], preparation_sha256=digest,
                                    hu_quantiles=np.quantile(slices, [0, .01, .5, .99, 1]))
                temporary.replace(path)
                del slices, features, feature, item
            done += 1
            bar.update()
            save_json(OUT / f"feature_worker_{index}.json", {"status": "running", "pid": os.getpid(),
                       "done": done, "total": len(assigned), "last_exam": row["exam_id"],
                       "elapsed_seconds": time.time()-started, "updated_unix": time.time()})
    assert done == len(assigned)
    save_json(OUT / f"feature_worker_{index}.json", {"status": "complete", "done": done, "total": len(assigned)})


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--worker-index", type=int)
    parser.add_argument("--workers", type=int, default=2)
    args = parser.parse_args()
    if args.worker_index is None:
        prepare_tables()
    else:
        feature_worker(args.worker_index, args.workers)


if __name__ == "__main__":
    main()
