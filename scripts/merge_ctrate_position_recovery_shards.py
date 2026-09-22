#!/usr/bin/env python3
"""合并 CT-RATE 位置恢复病例级任务，并生成全测试集汇总。"""

from __future__ import annotations

import argparse
import csv
import json
import time
from pathlib import Path

from evaluate_ctrate_amef_position_recovery import (
    aggregate_results,
    plot_block3_position_distribution,
    save_json,
    write_csv,
)


CSV_NAMES = ("per_slice_results.csv", "interval_results.csv", "repeat_metrics.csv")


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8-sig", newline="") as stream:
        return list(csv.DictReader(stream))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--queue", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--plot-seed", type=int, default=42)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    queue = json.loads(args.queue.read_text(encoding="utf-8"))
    expected = [int(value) for value in queue["expected_case_indices"]]
    shard_root = Path(queue["output_dir"]) / "shards"
    shard_dirs = [shard_root / f"case_{case_index:04d}" for case_index in expected]
    missing = [str(path) for path in shard_dirs if not (path / "metadata.json").is_file()]
    if missing:
        raise FileNotFoundError(f"仍有病例任务未完成：{missing[:10]}")

    output = args.output.resolve()
    if output.exists() and any(output.iterdir()) and not args.overwrite:
        raise FileExistsError(f"合并目录非空，请加--overwrite：{output}")
    output.mkdir(parents=True, exist_ok=True)

    all_case_ids: list[int] = []
    slice_rows: list[dict[str, str]] = []
    interval_rows: list[dict[str, str]] = []
    repeat_rows: list[dict[str, str]] = []
    manifest_rows: list[dict] = []
    metadata = None
    shard_metadata = []
    for shard_dir in shard_dirs:
        shard_meta = json.loads((shard_dir / "metadata.json").read_text(encoding="utf-8"))
        shard_metadata.append(shard_meta)
        case_ids = {int(value) for value in shard_meta.get("selected_case_indices") or []}
        if not case_ids:
            case_ids = {int(row["case_index"]) for row in read_csv(shard_dir / "repeat_metrics.csv")}
        if len(case_ids) != 1:
            raise ValueError(f"病例分片不唯一：{shard_dir} -> {case_ids}")
        all_case_ids.extend(case_ids)
        if metadata is None:
            metadata = dict(shard_meta)
        for name, target in (
            ("per_slice_results.csv", slice_rows),
            ("interval_results.csv", interval_rows),
            ("repeat_metrics.csv", repeat_rows),
        ):
            target.extend(read_csv(shard_dir / name))
        with (shard_dir / "sampling_manifest.jsonl").open(encoding="utf-8") as stream:
            manifest_rows.extend(json.loads(line) for line in stream if line.strip())

    if sorted(all_case_ids) != sorted(expected) or len(all_case_ids) != len(set(all_case_ids)):
        raise ValueError("病例分片集合与fold4完整测试集不一致")
    if metadata is None:
        raise RuntimeError("没有可合并的分片")

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
    plot_case_index = plot_block3_position_distribution(
        output,
        slice_rows,
        repeat_rows,
        manifest_rows,
        expected,
        args.plot_seed,
    )

    metadata.update({
        "output_type": "merged_full_test_set",
        "test_case_count_before_exclusion": len(expected),
        "eligible_case_count": len(expected),
        "excluded_case_count": 0,
        "excluded_cases": [],
        "processed_case_count": len(expected),
        "repeat_count": len(repeat_rows),
        "slice_result_count": len(slice_rows),
        "interval_result_count": len(interval_rows),
        "plot_case_index": plot_case_index,
        "shard_count": len(shard_dirs),
        "shard_metadata": shard_metadata,
        "finished_unix": time.time(),
        "full_test_case_policy": "all fold4 test cases; when remaining_count<T, use K=remaining_count without duplication or interpolation",
    })
    save_json(output / "metadata.json", metadata)
    print(f"合并完成：{len(expected)}例，输出目录：{output}")


if __name__ == "__main__":
    main()
