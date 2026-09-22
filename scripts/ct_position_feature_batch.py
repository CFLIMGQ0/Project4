#!/usr/bin/env python3
"""按任务清单复用原始CT缺失特征提取器，可指定临时原图目录。"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import run_ctrate_slice_missing as experiment


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--spec", type=Path, required=True)
    parser.add_argument("--raw-root", type=Path)
    args = parser.parse_args()
    spec = json.loads(args.spec.read_text())
    if args.raw_root is not None:
        experiment.DATA = args.raw_root.resolve()
    original_reader = experiment.read_inputs
    original_save = experiment.save_json
    for group in spec["groups"]:
        experiment.OUT = experiment.ROOT / group["output"]
        rows, folds, targets = original_reader()
        indices = set(group["case_indices"])
        selected = [row for row in rows if row["case_index"] in indices]
        if len(selected) != len(indices):
            raise ValueError("特征任务中存在无效病例")
        experiment.read_inputs = lambda: (selected, folds, targets)
        experiment.save_json = lambda path, value: original_save(
            args.spec.parent / f"{args.spec.stem}_progress.json", value)
        experiment.extract(0, 1)


if __name__ == "__main__":
    main()
