#!/usr/bin/env python3
"""合并 CT-RATE 位置恢复分组任务并重新生成全测试集汇总。"""

from __future__ import annotations

import argparse
import csv
import json
import time
from collections import Counter
from pathlib import Path

from evaluate_ctrate_amef_position_recovery import aggregate_results, plot_curves, save_json, write_csv


CSV_NAMES = ("per_slice_results.csv", "interval_results.csv", "repeat_metrics.csv")


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8-sig", newline="") as stream:
        return list(csv.DictReader(stream))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--shards", nargs="+", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    output = args.output.resolve()
    if output.exists() and any(output.iterdir()) and not args.overwrite:
        raise FileExistsError(f"合并目录非空，请使用新目录或加--overwrite：{output}")
    output.mkdir(parents=True, exist_ok=True)

    slice_rows: list[dict[str, str]] = []
    interval_rows: list[dict[str, str]] = []
    repeat_rows: list[dict[str, str]] = []
    manifest_rows: list[dict] = []
    case_ids: list[int] = []
    metadata = None
    shard_metadata = []
    for shard in args.shards:
        shard = shard.resolve()
        required = [shard / name for name in CSV_NAMES] + [shard / "metadata.json"]
        missing = [str(path) for path in required if not path.is_file()]
        if missing:
            raise FileNotFoundError(f"分片不完整：{missing}")
        shard_meta = json.loads((shard / "metadata.json").read_text(encoding="utf-8"))
        shard_metadata.append(shard_meta)
        shard_cases = set(int(value) for value in shard_meta.get("selected_case_indices") or [])
        if not shard_cases:
            shard_cases = {int(row["case_index"]) for row in read_csv(shard / "repeat_metrics.csv")}
        if not shard_cases:
            raise ValueError(f"分片没有病例：{shard}")
        case_ids.extend(shard_cases)
        if metadata is None:
            metadata = dict(shard_meta)
        slice_rows.extend(read_csv(shard / "per_slice_results.csv"))
        interval_rows.extend(read_csv(shard / "interval_results.csv"))
        repeat_rows.extend(read_csv(shard / "repeat_metrics.csv"))
        manifest_path = shard / "sampling_manifest.jsonl"
        with manifest_path.open(encoding="utf-8") as stream:
            manifest_rows.extend(json.loads(line) for line in stream if line.strip())

    if len(case_ids) != len(set(case_ids)):
        raise ValueError("分片病例集合存在重复")
    if metadata is None:
        raise RuntimeError("没有可合并的元数据")

    for name, rows in (
        ("per_slice_results.csv", slice_rows),
        ("interval_results.csv", interval_rows),
        ("repeat_metrics.csv", repeat_rows),
    ):
        fields = list(rows[0]) if rows else ["case_index", "patient_id"]
        write_csv(output / name, rows, fields)
    with (output / "sampling_manifest.jsonl").open("w", encoding="utf-8") as stream:
        for row in manifest_rows:
            stream.write(json.dumps(row, ensure_ascii=False) + "\n")
    aggregate_results(repeat_rows, output)
    plot_curves(output, slice_rows, sorted(case_ids)[:6])
    test_case_indices = set()
    total_test_cases = int(metadata.get("test_case_count_before_exclusion", 0))
    split_path = Path(str(metadata.get("split_path", "")))
    if split_path.is_file():
        test_case_indices = set(json.loads(split_path.read_text(encoding="utf-8"))["test"])
        total_test_cases = len(test_case_indices)
    excluded_case_indices = sorted(test_case_indices - set(case_ids)) if test_case_indices else []
    metadata.update({
        "output_type": "merged_recovery_shards",
        "test_case_count_before_exclusion": total_test_cases,
        "selected_case_indices": sorted(case_ids),
        "eligible_case_count": len(case_ids),
        "excluded_case_count": len(excluded_case_indices),
        "excluded_cases": excluded_case_indices,
        "processed_case_count": len(case_ids),
        "repeat_count": len(repeat_rows),
        "slice_result_count": len(slice_rows),
        "interval_result_count": len(interval_rows),
        "shard_count": len(args.shards),
        "shard_metadata": shard_metadata,
        "abnormal_model_status_counts": dict(Counter(row["model_status"] for row in repeat_rows)),
        "finished_unix": time.time(),
    })
    save_json(output / "metadata.json", metadata)
    print(f"合并完成：{len(case_ids)}例，输出目录：{output}")


if __name__ == "__main__":
    main()
