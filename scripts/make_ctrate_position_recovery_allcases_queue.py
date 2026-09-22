#!/usr/bin/env python3
"""为 CT-RATE block3 全测试集生成病例级六卡动态队列。"""

from __future__ import annotations

import argparse
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
EXPERIMENT = ROOT / "outputs" / "ct_rate_680" / "experiment"
DATA = ROOT / "datasets" / "ct_rate_680"
CHECKPOINT = "outputs/ct_rate_680/all_models_fivefold/fold_4/amef_multimodal/best_model.pt"
OUTPUT = "outputs/ct_rate_680/position_recovery_block3_all136"
QUEUE = "outputs/ct_rate_680/position_recovery_block3_all136_queue/queue.json"
SEEDS = [42, 2026, 3407, 7919, 104729]
ARTIFACT_NAMES = (
    "metadata.json",
    "sampling_manifest.jsonl",
    "per_slice_results.csv",
    "interval_results.csv",
    "repeat_metrics.csv",
    "ct_summary.csv",
    "summary.csv",
    "paper_table.csv",
    "paper_table.md",
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", default=OUTPUT)
    parser.add_argument("--queue", default=QUEUE)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--nonzero-seeds", type=int, nargs="+", default=[42])
    args = parser.parse_args()

    rows = json.loads((EXPERIMENT / "samples.json").read_text(encoding="utf-8"))
    split = json.loads(
        (ROOT / "outputs/ct_rate_680/all_models_fivefold/fold_4/amef_multimodal/split_ids.json").read_text(
            encoding="utf-8"
        )
    )
    test_ids = [int(value) for value in split["test"]]
    by_case = {int(row["case_index"]): row for row in rows}
    if sorted(test_ids) != test_ids or len(test_ids) != 136:
        raise ValueError(f"fold4测试集不是预期的136例：{len(test_ids)}")
    if set(test_ids) - set(by_case):
        raise ValueError("测试集存在未在samples.json中的病例")

    output = Path(args.output)
    queue_path = Path(args.queue)
    case_dir = output / "case_files"
    case_dir.mkdir(parents=True, exist_ok=True)
    all_cases_file = case_dir / "all_case_indices.json"
    all_cases_file.write_text(json.dumps({"case_indices": test_ids}, indent=2) + "\n", encoding="utf-8")

    jobs = []
    for case_index in test_ids:
        row = by_case[case_index]
        case_name = f"case_{case_index:04d}"
        case_file = case_dir / f"{case_name}.json"
        case_file.write_text(json.dumps({"case_indices": [case_index]}, indent=2) + "\n", encoding="utf-8")
        shard = f"{output.as_posix()}/shards/{case_name}"
        artifacts = [f"{shard}/{name}" for name in ARTIFACT_NAMES]
        jobs.append({
            "id": case_name,
            "phase": "position_recovery",
            "memory_mib": 2500,
            "timeout_seconds": 3600,
            "command": [
                "src/scripts/evaluate_ctrate_amef_position_recovery.py",
                "--checkpoint", CHECKPOINT,
                "--mode", "block3",
                "--allow-short-inputs",
                "--case-indices-file", f"{case_dir.as_posix()}/{case_name}.json",
                "--output-dir", shard,
                "--raw-cache-dir", f"{shard}/raw_feature_cache",
                "--cleanup-raw-cache",
                "--nonzero-seeds", *[str(seed) for seed in args.nonzero_seeds],
                "--skip-plot",
                "--overwrite",
            ],
            "sync_inputs": [f"{case_dir.as_posix()}/{case_name}.json"],
            "raw_files": [row["file_path"]],
            "artifacts": artifacts,
        })

    queue = {
        "description": "CT-RATE fold4固定测试集136例；block3全病例；不足T时使用实际剩余切片；每个非零比例使用命令行指定种子；204四卡与202两卡动态显存队列",
        "stage_inputs": [
            "outputs/ct_rate_680/experiment",
            "outputs/ct_rate_680/all_models_fivefold/fold_4/amef_multimodal",
            "pre_weights/checkpoints/convnext_tiny-983f1562.pth",
        ],
        "expected_case_indices": test_ids,
        "nonzero_seeds": [int(seed) for seed in args.nonzero_seeds],
        "output_dir": output.as_posix(),
        "jobs": jobs,
    }
    queue_path.parent.mkdir(parents=True, exist_ok=True)
    if queue_path.exists() and not args.overwrite:
        raise FileExistsError(f"队列已存在，若确认重建请加--overwrite：{queue_path}")
    queue_path.write_text(json.dumps(queue, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"queue": str(queue_path), "jobs": len(jobs), "test_cases": len(test_ids)}, ensure_ascii=False))


if __name__ == "__main__":
    main()
