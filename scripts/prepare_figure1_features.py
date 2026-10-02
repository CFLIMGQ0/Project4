#!/usr/bin/env python3
"""只为 Figure 1 的实际删片网格补提取原始序列冻结视觉特征。"""

from __future__ import annotations

import argparse
import gzip
import hashlib
import json
from pathlib import Path
import sys

import numpy as np
from tqdm import tqdm

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "src/scripts"))
from prepare_public_block3_features import FrozenEncoder, prepare_amos, prepare_mr

INPUTS = {"ct_rate": "ct_rate_680", "amos_mm": "amos_mm", "mr_rate_1k": "mr_rate_1k",
          "merlin_1k": "merlin_1k"}
RAWS = {"ct_rate": ROOT / "datasets/ct_rate_680", "merlin_1k": ROOT / "merlin_selected_1000"}
MASKS = ROOT / "outputs/paper_results/figure1_masks"
OUT = ROOT / "outputs/paper_results/figure1_features"
WINDOWS = ((-600., 1500.), (40., 400.), (-600., 700.))


def encode_ct_slices(row: dict, indices: np.ndarray, encoder: FrozenEncoder, dataset: str):
    import nibabel as nib
    import torch

    path = RAWS[dataset] / row["file_path"]
    image = nib.load(str(path))
    data = np.asarray(image.dataobj, dtype=np.float32)
    transform = nib.orientations.ornt_transform(
        nib.orientations.io_orientation(image.affine), nib.orientations.axcodes2ornt(("R", "A", "S")))
    data = nib.orientations.apply_orientation(data, transform)
    total = int(data.shape[2])
    if len(indices) == 0:
        return total, np.empty((0, 768), np.float32)
    if int(indices[0]) < 0 or int(indices[-1]) >= total:
        raise ValueError(f"原始切片索引越界：{path}")
    slices = np.ascontiguousarray(data[::-1, ::-1, indices].transpose(2, 1, 0))
    del data
    if not np.isfinite(slices).all():
        raise ValueError(f"非有限图像值：{path}")
    values = []
    with torch.inference_mode():
        for start in range(0, len(slices), 16):
            batch = torch.from_numpy(slices[start:start+16]).to(encoder.device)[:, None]
            channels = torch.cat([((batch-(level-width/2))/width).clamp(0, 1)
                                  for level, width in WINDOWS], dim=1)
            channels = encoder.F.interpolate(channels, (224, 224), mode="bilinear",
                                             align_corners=False, antialias=True)
            with torch.autocast("cuda"):
                encoded = encoder.model((channels-encoder.mean)/encoder.std)
            values.append(encoded.float().cpu().numpy())
    features = np.concatenate(values)
    if features.shape != (len(indices), 768) or not np.isfinite(features).all():
        raise ValueError(f"提取特征异常：{path}")
    return total, features


def prepare_case(dataset: str, case: int, rows: list[dict], encoder: FrozenEncoder) -> None:
    manifest_path = MASKS / dataset / f"{case:04d}.json.gz"
    if not manifest_path.exists():
        raise FileNotFoundError(manifest_path)
    output = OUT / dataset / f"{case:04d}.npz"
    source_hash = hashlib.sha256(manifest_path.read_bytes()).hexdigest()
    with gzip.open(manifest_path, "rt", encoding="utf-8") as stream:
        manifest = json.load(stream)
    union = np.asarray(sorted({int(index) for sample in manifest["configurations"].values()
                               for index in sample["selected_raw_indices"]}), dtype=np.int64)
    if output.exists():
        with np.load(output, allow_pickle=False) as cached:
            if (str(cached["mask_manifest_sha256"]) == source_hash and
                np.array_equal(cached["source_indices"], union) and
                cached["features"].shape == (len(union), 768) and
                np.isfinite(cached["features"]).all()):
                return
        raise RuntimeError(f"现有 Figure 1 特征缓存协议不同：{output}")
    initial = ROOT / "outputs" / INPUTS[dataset] / "experiment/features" / f"{case:04d}.npz"
    with np.load(initial, allow_pickle=False) as cache:
        old_indices = cache["slice_indices"].astype(np.int64)
        old_features = cache["features"].astype(np.float32)
        old_total = int(cache["original_count"])
    if old_total != manifest["source_count"]:
        raise ValueError(f"原始序列长度不一致：{dataset}/{case}")
    missing = np.setdiff1d(union, old_indices, assume_unique=True)
    if len(missing) == 0:
        actual_total, new_features = old_total, np.empty((0, 768), np.float32)
    elif dataset in ("ct_rate", "merlin_1k"):
        actual_total, new_features = encode_ct_slices(rows[case], missing, encoder, dataset)
    elif dataset == "amos_mm":
        actual_total, new_features = prepare_amos(rows[case], missing, encoder)
    else:
        actual_total, new_features = prepare_mr(rows[case], missing, encoder)
    if actual_total != old_total:
        raise ValueError(f"原始图像长度与缓存不一致：{dataset}/{case}")
    old_map = {int(index): feature for index, feature in zip(old_indices, old_features) if index in union}
    new_map = {int(index): feature for index, feature in zip(missing, new_features)}
    features = np.stack([old_map[int(index)] if int(index) in old_map else new_map[int(index)]
                         for index in union]).astype(np.float32)
    if features.shape != (len(union), 768) or not np.isfinite(features).all():
        raise ValueError(f"合并特征异常：{dataset}/{case}")
    output.parent.mkdir(parents=True, exist_ok=True)
    temp = output.with_suffix(".tmp.npz")
    np.savez_compressed(temp, features=features, source_indices=union,
                        original_count=np.int64(old_total), case_index=np.int64(case),
                        mask_manifest_sha256=np.asarray(source_hash),
                        reused_initial_features=np.int64(len(old_map)),
                        newly_encoded_features=np.int64(len(missing)))
    temp.replace(output)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", choices=tuple(INPUTS), required=True)
    parser.add_argument("--worker-index", type=int, required=True)
    parser.add_argument("--workers", type=int, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--limit", type=int)
    args = parser.parse_args()
    root = MASKS / args.dataset
    rows = json.loads((ROOT / "outputs" / INPUTS[args.dataset] / "experiment/samples.json").read_text())
    cases = sorted(int(path.stem.split(".")[0]) for path in root.glob("*.json.gz"))
    cases = [case for case in cases if case % args.workers == args.worker_index]
    if args.limit is not None:
        cases = cases[:args.limit]
    encoder = FrozenEncoder(args.device)
    for case in tqdm(cases, desc=f"{args.dataset} GPU-worker{args.worker_index} 原始切片特征"):
        prepare_case(args.dataset, case, rows, encoder)
    print(f"完成 {args.dataset} worker{args.worker_index}: {len(cases)} 病例", flush=True)


if __name__ == "__main__":
    main()
