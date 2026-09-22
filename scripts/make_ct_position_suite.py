#!/usr/bin/env python3
"""生成50%与75%缺失配对五折的可续跑六GPU任务图。"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from ct_six_gpu_queue import ROOT, DEFAULT_QUEUE, save


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--batch-cases", type=int, default=4)
    args = parser.parse_args()
    if args.batch_cases < 1:
        parser.error("每批病例数量必须为正")
    if DEFAULT_QUEUE.exists():
        raise FileExistsError("已有队列不可覆盖；直接续跑调度器")
    folders = {rate: f"outputs/ct_rate_680/missing{rate}_raw_hidden_fivefold" for rate in (50, 75)}
    rows = {rate: json.loads((ROOT / folder / "data/samples.json").read_text()) for rate, folder in folders.items()}
    mapping = {rate: {row["source_case_index"]: row for row in values} for rate, values in rows.items()}
    sources = sorted(mapping[50], key=lambda index: (index not in mapping[75], index))
    dependencies = {rate: [] for rate in folders}
    jobs = []
    for start in range(0, len(sources), args.batch_cases):
        selected = sources[start:start + args.batch_cases]
        name = f"features_{start // args.batch_cases:03d}"
        groups, artifacts = [], []
        for rate, folder in folders.items():
            indices = [mapping[rate][index]["case_index"] for index in selected if index in mapping[rate]]
            if indices:
                groups.append({"output": folder, "case_indices": indices})
                artifacts += [f"{folder}/data/features/{index:04d}.npz" for index in indices]
                dependencies[rate].append(name)
        spec = DEFAULT_QUEUE.parent / "feature_specs" / f"{name}.json"
        save(spec, {"groups": groups})
        jobs.append({"id": name, "phase": "features", "memory_mib": 2500,
                     "command": ["src/scripts/ct_position_feature_batch.py", "--spec", str(spec.relative_to(ROOT))],
                     "sync_inputs": [str(spec.relative_to(ROOT))], "artifacts": artifacts,
                     "raw_files": [mapping[50][index]["file_path"] for index in selected], "timeout_seconds": 3600})
    for rate, folder in folders.items():
        command = ["src/scripts/run_ctrate_slice_missing.py", "--output-dir", folder]
        audit = f"missing{rate}_audit"
        smoke = f"missing{rate}_smoke"
        jobs.append({"id": audit, "phase": "audit", "local_only": True, "memory_mib": 500,
                     "command": [*command, "--audit-only"], "depends_on": dependencies[rate],
                     "artifacts": [f"{folder}/data_audit.json", f"{folder}/data/feature_integrity.json",
                                   *[f"{folder}/{variant}/protocol.json" for variant in ("apro_full", "original_pe")]]})
        jobs.append({"id": smoke, "phase": "smoke", "local_only": True, "memory_mib": 2000,
                     "command": [*command, "--smoke-test"], "depends_on": [audit],
                     "artifacts": [f"{folder}/smoke_audit.json"]})
        trained = []
        for fold in range(1, 6):
            for variant in ("apro_full", "original_pe"):
                name = f"missing{rate}_{variant}_fold{fold}"
                trained.append(name)
                results = f"{folder}/{variant}/fold_{fold}/amef_multimodal"
                jobs.append({"id": name, "phase": "training", "memory_mib": 2500,
                             "command": [*command, "--variant", variant, "--fold", str(fold)],
                             "depends_on": [smoke],
                             "sync_inputs": [f"{folder}/data", f"{folder}/{variant}/protocol.json"],
                             "artifacts": [f"{results}/{file}" for file in (
                                 "completed.json", "test_metrics.json", "test_predictions.csv", "best_model.pt")],
                             "sync_output_dirs": [results]})
        jobs.append({"id": f"missing{rate}_aggregate", "phase": "aggregate", "local_only": True,
                     "memory_mib": 500, "command": [*command, "--aggregate"], "depends_on": trained,
                     "artifacts": [f"{folder}/comparison.csv", f"{folder}/comparison.md", f"{folder}/comparison.json"]})
    stage_inputs = ["pre_weights/checkpoints/convnext_tiny-983f1562.pth"]
    stage_inputs += [f"{folder}/data" for folder in folders.values()]
    save(DEFAULT_QUEUE, {"description": "两种原CT删除比例；固定原五折；同卡允许多个任务；仅临时传输当前批次原CT",
                         "stage_inputs": stage_inputs, "jobs": jobs})
    print(f"已生成{len(jobs)}个任务：678例原CT，20个训练折，按75%子集优先准备。")


if __name__ == "__main__":
    main()
