#!/usr/bin/env python3
"""为 CT-RATE 位置恢复生成可复用的完整切片视觉特征缓存。"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys

import numpy as np
from tqdm import tqdm


ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "src"
EXPERIMENT = ROOT / "outputs" / "ct_rate_680" / "experiment"
DATA = ROOT / "datasets" / "ct_rate_680"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024**2), b""):
            digest.update(block)
    return digest.hexdigest()


def load_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--case-indices-file", type=Path, required=True)
    parser.add_argument("--cache-dir", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--data-root", type=Path, default=DATA)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    import torch

    if str(SRC) not in sys.path:
        sys.path.insert(0, str(SRC))
    from evaluate_ctrate_amef_position_recovery import FrozenConvNeXt, oriented_volume

    selection = load_json(args.case_indices_file)
    case_indices = [int(value) for value in selection["case_indices"]]
    rows = {int(row["case_index"]): row for row in load_json(EXPERIMENT / "samples.json")}
    preparation = load_json(EXPERIMENT / "preparation_protocol.json")
    evaluator = SRC / "scripts" / "evaluate_ctrate_amef_position_recovery.py"
    cache_protocol = hashlib.sha256(
        (preparation["preparation_sha256"] + sha256(evaluator)).encode("utf-8")
    ).hexdigest()
    args.cache_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device)
    extractor = FrozenConvNeXt(device)
    for case_index in tqdm(case_indices, desc="生成CT位置特征缓存"):
        row = rows[case_index]
        output = args.cache_dir / f"{case_index:04d}.npz"
        if output.is_file():
            with np.load(output, allow_pickle=False) as cached:
                if str(cached["cache_protocol"]) == cache_protocol:
                    continue
        data_root = args.data_root.resolve()
        original_data_root = __import__("evaluate_ctrate_amef_position_recovery").DATA
        __import__("evaluate_ctrate_amef_position_recovery").DATA = data_root
        slices, positions, indices, equidistant = oriented_volume(row)
        features = extractor.encode(slices)
        temporary = output.with_suffix(".tmp.npz")
        np.savez_compressed(
            temporary,
            features=features,
            true_positions=positions,
            slice_indices=indices,
            equidistant=np.asarray(int(equidistant), dtype=np.int8),
            cache_protocol=cache_protocol,
            patient_id=row["patient_id"],
        )
        temporary.replace(output)
        __import__("evaluate_ctrate_amef_position_recovery").DATA = original_data_root
    print(f"特征缓存完成：{len(case_indices)}例，目录：{args.cache_dir}")


if __name__ == "__main__":
    main()
