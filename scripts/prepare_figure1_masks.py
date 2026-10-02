#!/usr/bin/env python3
"""Figure 1 四数据集共享的 B1–B8 × 0–80% 可复现删片清单。"""

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
sys.path.insert(0, str(ROOT / "src/scripts"))
from paper_block_deletion import RATIOS, BLOCKS, block_sampling

OUT = ROOT / "outputs/paper_results/figure1_masks"
INPUTS = {"ct_rate": "ct_rate_680", "amos_mm": "amos_mm", "mr_rate_1k": "mr_rate_1k",
          "merlin_1k": "merlin_1k"}


def test_folds(dataset: str, input_dir: Path):
    if dataset == "amos_mm":
        split = json.loads((input_dir / "splits.json").read_text())
        return [split["test"] for _ in range(5)]
    return json.loads((input_dir / "patient_folds.json").read_text())["folds"]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", choices=tuple(INPUTS), required=True)
    args = parser.parse_args()
    input_dir = ROOT / "outputs" / INPUTS[args.dataset] / "experiment"
    output_dir = OUT / args.dataset
    output_dir.mkdir(parents=True, exist_ok=True)
    rows = json.loads((input_dir / "samples.json").read_text())
    folds = test_folds(args.dataset, input_dir)
    cases = sorted({int(case) for group in folds for case in group})
    counts = {str(ratio): {str(blocks): [0]*5 for blocks in ((1,) if ratio == 0 else BLOCKS)}
              for ratio in RATIOS}
    skipped = []
    for case in tqdm(cases, desc=f"{args.dataset} 生成Figure 1共享删片清单"):
        cache_path = input_dir / "features" / f"{case:04d}.npz"
        if not cache_path.exists():
            skipped.append({"case_index": case, "reason": "feature cache missing"})
            continue
        with np.load(cache_path, allow_pickle=False) as cache:
            total = int(cache["original_count"])
        manifest = {"dataset": args.dataset, "case_index": case,
                    "source_count": total, "seed": 42, "configurations": {}}
        for ratio in RATIOS:
            for blocks in ((1,) if ratio == 0 else BLOCKS):
                try:
                    value = block_sampling(total, ratio, blocks, case)
                except ValueError as error:
                    skipped.append({"case_index": case, "ratio": ratio, "blocks": blocks,
                                    "reason": str(error)})
                    continue
                manifest["configurations"][f"{ratio:02d}_B{blocks}"] = value
                for fold, group in enumerate(folds):
                    if case in group:
                        counts[str(ratio)][str(blocks)][fold] += 1
        path = output_dir / f"{case:04d}.json.gz"
        if path.exists():
            with gzip.open(path, "rt", encoding="utf-8") as stream:
                if json.load(stream) != manifest:
                    raise RuntimeError(f"已存在的删片清单不同：{path}")
        else:
            temp = path.with_suffix(".tmp")
            with gzip.open(temp, "wt", encoding="utf-8") as stream:
                json.dump(manifest, stream, ensure_ascii=False, separators=(",", ":"))
            temp.replace(path)
    protocol = {"dataset": args.dataset, "seed": 42, "ratios": RATIOS, "blocks": BLOCKS,
                "zero_ratio_run_once": True, "target_max_instances": 64,
                "short_sequence": "不足64时保留全部有效图像并以图像掩码补齐",
                "sampling": "与原Block3相同的随机区段算法；适用病例逐配置计数",
                "source_sha256": {str(path.relative_to(ROOT)): hashlib.sha256(path.read_bytes()).hexdigest()
                                  for path in [Path(__file__), ROOT / "src/scripts/paper_block_deletion.py",
                                               input_dir / "samples.json"]},
                "cases": len(cases), "processed": len(list(output_dir.glob("*.json.gz"))),
                "fold_eligible_counts": counts, "skipped": skipped}
    (output_dir / "summary.json").write_text(json.dumps(protocol, ensure_ascii=False, indent=2))
    print(f"{args.dataset}: 已生成{protocol['processed']}/{len(cases)}病例删片清单；80% B8每折可评估数={counts['80']['8']}")


if __name__ == "__main__":
    main()
