#!/usr/bin/env python3
"""补齐 CT-RATE Block3 25/75% 在 Figure 1 特征并集之外的少量切片。"""

from __future__ import annotations

import json
import argparse
from pathlib import Path
import sys

import numpy as np
from tqdm import tqdm

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src/scripts"))
from paper_block_deletion import block_sampling
from prepare_public_block3_features import FrozenEncoder
from prepare_figure1_features import encode_ct_slices

OUT = ROOT / "outputs/paper_results/table2_ct_extra_features"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--available-only", action="store_true",
                        help="只处理已经完成 Figure 1 特征缓存的病例，适合与特征提取并行")
    args = parser.parse_args()
    input_dir = ROOT / "outputs/ct_rate_680/experiment"
    rows = json.loads((input_dir / "samples.json").read_text())
    encoder = None
    eligible = 0
    pending = 0
    total_missing = 0
    for case, row in enumerate(tqdm(rows, desc="CT-RATE 表2补充切片")):
        original = input_dir / "features" / f"{case:04d}.npz"
        with np.load(original, allow_pickle=False) as cache:
            total = int(cache["original_count"])
        if total < 256:
            continue
        eligible += 1
        figure_cache = ROOT / "outputs/paper_results/figure1_features/ct_rate" / f"{case:04d}.npz"
        if not figure_cache.exists():
            if args.available_only:
                pending += 1
                continue
            raise FileNotFoundError(figure_cache)
        with np.load(figure_cache, allow_pickle=False) as cache:
            union = cache["source_indices"].astype(np.int64)
        selected = np.asarray(sorted({int(index) for ratio in (0,25,50,75)
                                      for index in block_sampling(total, ratio, 1 if ratio == 0 else 3, case)["selected_raw_indices"]}),
                              dtype=np.int64)
        missing = np.setdiff1d(selected, union, assume_unique=True)
        if not len(missing):
            continue
        path = OUT / f"{case:04d}.npz"
        if path.exists():
            with np.load(path, allow_pickle=False) as cache:
                if np.array_equal(cache["source_indices"], missing) and cache["features"].shape == (len(missing),768):
                    total_missing += len(missing)
                    continue
            raise RuntimeError(f"现有补充缓存不同：{path}")
        if encoder is None:
            encoder = FrozenEncoder("cuda:0")
        actual, features = encode_ct_slices(row, missing, encoder, "ct_rate")
        if actual != total:
            raise ValueError(f"原始长度不符：case={case}")
        OUT.mkdir(parents=True, exist_ok=True)
        temporary = path.with_suffix(".tmp.npz")
        np.savez_compressed(temporary, source_indices=missing, features=features,
                            original_count=np.int64(total))
        temporary.replace(path)
        total_missing += len(missing)
    print(f"CT-RATE 表2可评估病例{eligible}；等待 Figure 1 特征缓存{pending}；补充切片{total_missing}", flush=True)


if __name__ == "__main__":
    main()
