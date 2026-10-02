#!/usr/bin/env python3
"""从现有 336px 私有图像缓存提取 64 张/检查的冻结视觉特征，无需原始 JPG。"""

from __future__ import annotations

import argparse
import csv
import gzip
import hashlib
import json
from collections import defaultdict
from pathlib import Path
import sys

import numpy as np
from tqdm import tqdm

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src/scripts"))
from prepare_public_block3_features import FrozenEncoder

MANIFEST = ROOT / "datasets/image_cache/task3_cache_manifest.jsonl.gz"
CACHE = ROOT / "datasets/image_cache/shared"
SPLITS = ROOT / "outputs/train_runs/task3/t3_main_model"
OUT = ROOT / "outputs/paper_results/private_features"
DATASETS = ("regular_white_light", "chromoscopic", "surgical", "ultrasound")


def manifest() -> dict[str, list[tuple[str, str]]]:
    mapping = defaultdict(list)
    with gzip.open(MANIFEST, "rt", encoding="utf-8") as stream:
        for line in stream:
            row = json.loads(line)
            exam = str(Path(row["image_path"]).parent.parent)
            mapping[exam].append((row["image_path"], row["cache_relpath"]))
    return {exam: sorted(values) for exam, values in mapping.items()}


def exam_rows(dataset: str) -> list[dict]:
    path = SPLITS / dataset / "fold_1/split_manifest.csv"
    with path.open(encoding="utf-8-sig", newline="") as stream:
        rows = list(csv.DictReader(stream))
    if len(rows) != len({row["exam_dir"] for row in rows}):
        raise ValueError(f"折内有重复检查：{dataset}")
    return rows


def feature_path(dataset: str, exam_dir: str) -> Path:
    name = hashlib.sha256(exam_dir.encode()).hexdigest()[:20]
    return OUT / dataset / f"{name}.npz"


def process(dataset: str, row: dict, listing: list[tuple[str, str]], encoder: FrozenEncoder) -> None:
    import torch

    exam = row["exam_dir"]
    output = feature_path(dataset, exam)
    count = len(listing)
    if count < 1:
        raise ValueError(f"检查图像缓存为空：{exam}")
    keep = min(64, count)
    selected = np.asarray([round(k * (count - 1) / (keep - 1)) for k in range(keep)] if keep > 1 else [0],
                          dtype=np.int64)
    if len(np.unique(selected)) != len(selected):
        raise ValueError(f"均匀选片出现重复：{exam}")
    if output.exists():
        with np.load(output, allow_pickle=False) as cache:
            if (str(cache["exam_dir"]) == exam and int(cache["original_count"]) == count
                    and np.array_equal(cache["source_indices"], selected)
                    and cache["features"].shape == (keep, 768)
                    and np.isfinite(cache["features"]).all()):
                return
        raise RuntimeError(f"已有私有特征缓存协议不符：{output}")
    values = []
    with torch.inference_mode():
        for begin in range(0, keep, 16):
            image_list = []
            for index in selected[begin:begin+16]:
                source = CACHE / listing[int(index)][1]
                pixels = np.load(source, allow_pickle=False)
                if pixels.shape != (336, 336, 3) or pixels.dtype != np.uint8:
                    raise ValueError(f"缓存尺寸/类型不符：{source}")
                image_list.append(pixels)
            batch = torch.from_numpy(np.stack(image_list)).to(encoder.device)
            batch = batch.permute(0, 3, 1, 2).float().div_(255)
            batch = encoder.F.interpolate(batch, (224, 224), mode="bilinear",
                                          align_corners=False, antialias=True)
            with torch.autocast("cuda"):
                encoded = encoder.model((batch-encoder.mean)/encoder.std)
            values.append(encoded.float().cpu().numpy())
    features = np.concatenate(values)
    if features.shape != (keep, 768) or not np.isfinite(features).all():
        raise ValueError(f"特征异常：{exam}")
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(".tmp.npz")
    np.savez_compressed(temporary, features=features, source_indices=selected,
                        original_count=np.int64(count), exam_dir=np.asarray(exam),
                        patient_id=np.asarray(row["patient_id"]))
    temporary.replace(output)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", choices=DATASETS, required=True)
    parser.add_argument("--worker-index", type=int, required=True)
    parser.add_argument("--workers", type=int, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--limit", type=int, default=0)
    args = parser.parse_args()
    listings = manifest()
    rows = exam_rows(args.dataset)
    for row in rows:
        if row["exam_dir"] not in listings:
            raise FileNotFoundError(f"缓存 manifest 缺检查：{row['exam_dir']}")
    jobs = [row for index, row in enumerate(rows) if index % args.workers == args.worker_index]
    if args.limit:
        jobs = jobs[:args.limit]
    encoder = FrozenEncoder(args.device)
    for row in tqdm(jobs, desc=f"{args.dataset} worker{args.worker_index} 私有缓存特征"):
        process(args.dataset, row, listings[row["exam_dir"]], encoder)
    print(f"完成 {args.dataset} worker{args.worker_index}: {len(jobs)} 个检查", flush=True)


if __name__ == "__main__":
    main()
