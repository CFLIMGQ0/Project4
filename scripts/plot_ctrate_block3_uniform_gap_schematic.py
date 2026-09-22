#!/usr/bin/env python3
"""Plot the CT-RATE block3 sampling-versus-slot-position schematic."""
from __future__ import annotations

import argparse
import csv
import json
import random
from pathlib import Path
import sys

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import FancyArrowPatch


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_BLOCK3_ROOT = ROOT / "outputs/ct_rate_680/position_recovery_block3_all136_seed42"
DEFAULT_OUTPUT = ROOT / "outputs/ct_rate_680/position_recovery_block3_uniform_gap_schematic_final"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--block3-root", type=Path, default=DEFAULT_BLOCK3_ROOT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--case-index", type=int)
    parser.add_argument("--selection-seed", type=int, default=42)
    parser.add_argument("--deletion-seed", type=int, default=42)
    return parser.parse_args()


def load_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def available_cases(root: Path, deletion_seed: int) -> list[int]:
    index_file = root / "case_files" / "all_case_indices.json"
    case_indices = json.loads(index_file.read_text(encoding="utf-8"))["case_indices"]
    available = []
    for case_index in sorted(int(value) for value in case_indices):
        manifest_path = root / "shards" / f"case_{case_index:04d}" / "sampling_manifest.jsonl"
        if not manifest_path.is_file():
            continue
        rows = load_jsonl(manifest_path)
        if any(abs(float(row["requested_delete_fraction"]) - 0.5) < 1e-9 and int(row["seed"]) == deletion_seed for row in rows):
            available.append(case_index)
    return available


def choose_case(root: Path, requested_case: int | None, selection_seed: int, deletion_seed: int) -> tuple[int, list[int]]:
    cases = available_cases(root, deletion_seed)
    if not cases:
        raise RuntimeError("已有 block3 结果中没有可用的 50%/指定删除种子病例")
    if requested_case is not None:
        if requested_case not in cases:
            raise ValueError(f"case_index={requested_case} 没有可用的 50%/seed={deletion_seed} 结果")
        return requested_case, cases
    selector = random.Random(selection_seed)
    return int(selector.choice(cases)), cases


def load_case_data(root: Path, case_index: int, deletion_seed: int) -> tuple[dict, list[dict], list[dict]]:
    shard = root / "shards" / f"case_{case_index:04d}"
    manifests = load_jsonl(shard / "sampling_manifest.jsonl")
    manifest = next(
        row
        for row in manifests
        if abs(float(row["requested_delete_fraction"]) - 0.5) < 1e-9 and int(row["seed"]) == deletion_seed
    )
    with (shard / "per_slice_results.csv").open(encoding="utf-8-sig", newline="") as stream:
        slices = [
            row
            for row in csv.DictReader(stream)
            if abs(float(row["deletion_fraction"]) - 0.5) < 1e-9 and int(row["seed"]) == deletion_seed
        ]
    with (shard / "interval_results.csv").open(encoding="utf-8-sig", newline="") as stream:
        intervals = [
            row
            for row in csv.DictReader(stream)
            if abs(float(row["deletion_fraction"]) - 0.5) < 1e-9 and int(row["seed"]) == deletion_seed
        ]
    return manifest, slices, intervals


def validate_inputs(manifest: dict, slices: list[dict], intervals: list[dict]) -> None:
    total = len(manifest["deleted_mask"])
    selected = np.asarray(manifest["selected_raw_indices"], dtype=np.int64)
    if manifest["requested_delete_count"] != manifest["actual_deleted_count"]:
        raise ValueError("删除数量不一致")
    if len(manifest["deleted_segments"]) != 3:
        raise ValueError("block3 没有恰好三个连续删除区段")
    if len(selected) != int(manifest["input_instances"]):
        raise ValueError("采样数量不一致")
    if not np.array_equal(selected, np.unique(selected)) or not np.all(np.diff(selected) > 0):
        raise ValueError("采样索引不是严格递增且无重复")
    if selected[0] != 0 or selected[-1] != total - 1:
        raise ValueError("采样没有保留完整序列首尾")
    if len(slices) != len(selected) or len(intervals) != len(selected) - 1:
        raise ValueError("逐切片或逐间隔结果数量不一致")
    baseline = np.asarray([float(row["baseline_position"]) for row in slices])
    expected = np.linspace(0.0, 1.0, len(slices))
    if not np.allclose(baseline, expected, atol=1e-6):
        raise ValueError("Original PE 不是等距槽位")
    crossing_count = sum(row["crossing_gap"] == "True" for row in intervals)
    if crossing_count < 1:
        raise ValueError("该病例没有跨删除区段的相邻输入间隔")


def write_source_csv(output_dir: Path, manifest: dict, slices: list[dict], intervals: list[dict], case_index: int) -> None:
    rows: list[dict] = []
    total = len(manifest["deleted_mask"])
    segment_ids = manifest["deleted_segment_ids"]
    for raw_index, deleted in enumerate(manifest["deleted_mask"]):
        rows.append({
            "layer": "full_sequence",
            "input_position": "",
            "raw_index": raw_index,
            "normalized_position": raw_index / (total - 1),
            "slot_position": "",
            "is_deleted": bool(deleted),
            "deleted_segment_index": segment_ids[raw_index] if deleted else "",
            "crossing_gap_to_next": "",
        })
    for row in slices:
        rows.append({
            "layer": "sampled_true_position",
            "input_position": int(row["selected_position"]),
            "raw_index": int(row["raw_index"]),
            "normalized_position": float(row["true_position"]),
            "slot_position": float(row["baseline_position"]),
            "is_deleted": False,
            "deleted_segment_index": "",
            "crossing_gap_to_next": "",
        })
    for row in slices:
        rows.append({
            "layer": "original_pe_slot",
            "input_position": int(row["selected_position"]),
            "raw_index": int(row["raw_index"]),
            "normalized_position": "",
            "slot_position": float(row["baseline_position"]),
            "is_deleted": False,
            "deleted_segment_index": "",
            "crossing_gap_to_next": "",
        })
    crossing_by_left = {int(row["left_selected_position"]): row["crossing_gap"] == "True" for row in intervals}
    for row in rows:
        if row["layer"] in {"sampled_true_position", "original_pe_slot"}:
            row["crossing_gap_to_next"] = crossing_by_left.get(int(row["input_position"]), False)
    fieldnames = list(rows[0])
    with (output_dir / f"case_{case_index:04d}_block3_uniform_gap_schematic.csv").open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def add_double_arrow(axis, left: float, right: float, y: float, color: str, text: str, text_y: float) -> None:
    axis.add_patch(FancyArrowPatch(
        (left, y), (right, y), arrowstyle="<->", mutation_scale=8,
        linewidth=1.0, color=color, zorder=5,
    ))
    axis.text((left + right) / 2.0, text_y, text, ha="center", va="bottom", fontsize=7, color=color)


def plot_figure(output_dir: Path, manifest: dict, slices: list[dict], intervals: list[dict], case_index: int, selection_seed: int, available_count: int) -> None:
    total = len(manifest["deleted_mask"])
    raw_x = np.arange(total, dtype=float) / float(total - 1)
    selected = np.asarray([int(row["raw_index"]) for row in slices], dtype=np.int64)
    true_x = np.asarray([float(row["true_position"]) for row in slices])
    slot_x = np.asarray([float(row["baseline_position"]) for row in slices])
    deleted = np.asarray(manifest["deleted_mask"], dtype=bool)
    segment_ids = np.asarray(manifest["deleted_segment_ids"], dtype=np.int64)

    mpl.rcParams.update({
        "font.family": "sans-serif",
        "font.sans-serif": ["Noto Sans CJK SC", "Arial", "Helvetica", "DejaVu Sans", "sans-serif"],
        "pdf.fonttype": 42,
        "svg.fonttype": "none",
        "font.size": 8,
        "axes.spines.right": False,
        "axes.spines.top": False,
        "axes.linewidth": 0.8,
    })
    figure, axis = plt.subplots(figsize=(7.2, 4.8))
    axis.set_xlim(-0.015, 1.015)
    axis.set_ylim(-0.55, 2.55)
    axis.axis("off")

    y_full, y_true, y_slot = 2.0, 1.0, 0.0
    axis.hlines([y_full, y_true, y_slot], 0.0, 1.0, color="#c8cdd2", linewidth=0.8, zorder=0)
    for segment in manifest["deleted_segments"]:
        start = segment["start_raw_index"] / float(total - 1)
        end = segment["end_raw_index_inclusive"] / float(total - 1)
        axis.axvspan(start, end, ymin=0.72, ymax=0.93, color="#f4b6a6", alpha=0.68, zorder=0)
        axis.plot([start, start], [1.79, 2.21], color="#c44e52", linewidth=0.8, zorder=2)
        axis.plot([end, end], [1.79, 2.21], color="#c44e52", linewidth=0.8, zorder=2)
        axis.text((start + end) / 2, 2.34, f"删除 {segment['start_raw_index']}–{segment['end_raw_index_inclusive']}",
                  ha="center", va="bottom", fontsize=7, color="#a33f43")

    axis.vlines(raw_x[~deleted], y_full - 0.10, y_full + 0.10, color="#557a95", linewidth=0.45, alpha=0.75)
    axis.vlines(raw_x[deleted], y_full - 0.12, y_full + 0.12, color="#c44e52", linewidth=0.65, alpha=0.95)
    axis.scatter(raw_x[~deleted], np.full((~deleted).sum(), y_full), s=5, color="#557a95", zorder=3)
    axis.scatter(raw_x[deleted], np.full(deleted.sum(), y_full), s=5, color="#c44e52", zorder=3)
    axis.text(-0.012, y_full, "完整 CT 序列", ha="right", va="center", fontsize=9, fontweight="bold", color="#28343b")

    axis.vlines(true_x, y_true - 0.10, y_true + 0.10, color="#2166ac", linewidth=0.65, alpha=0.65)
    axis.scatter(true_x, np.full(len(true_x), y_true), s=24, color="#2166ac", edgecolor="white", linewidth=0.35, zorder=4)
    axis.text(-0.012, y_true, "采样切片真实位置", ha="right", va="center", fontsize=9, fontweight="bold", color="#28343b")
    axis.text(0.01, 1.17, "先删去连续区段，再对剩余序列均匀采样 T=64 张", ha="left", va="bottom", fontsize=7.5, color="#2166ac")

    axis.vlines(slot_x, y_slot - 0.10, y_slot + 0.10, color="#d27b2d", linewidth=0.65, alpha=0.65)
    axis.scatter(slot_x, np.full(len(slot_x), y_slot), s=24, color="#d27b2d", edgecolor="white", linewidth=0.35, zorder=4)
    axis.text(-0.012, y_slot, "Original PE 槽位", ha="right", va="center", fontsize=9, fontweight="bold", color="#28343b")

    connector_indices = sorted(set(np.linspace(0, len(slices) - 1, 9, dtype=int).tolist()))
    crossing_indices = [int(row["left_selected_position"]) for row in intervals if row["crossing_gap"] == "True"]
    connector_indices = sorted(set(connector_indices + crossing_indices + [index + 1 for index in crossing_indices]))
    for index in connector_indices:
        axis.plot([true_x[index], slot_x[index]], [y_true, y_slot], color="#9aa0a6", linewidth=0.35, alpha=0.45, zorder=1)

    largest = max(
        (row for row in intervals if row["crossing_gap"] == "True"),
        key=lambda row: float(row["true_gap"]),
    )
    left_index = int(largest["left_selected_position"])
    right_index = int(largest["right_selected_position"])
    true_left, true_right = true_x[left_index], true_x[right_index]
    slot_left, slot_right = slot_x[left_index], slot_x[right_index]
    axis.scatter([true_left, true_right], [y_true, y_true], s=42, facecolors="none", edgecolors="#b2182b", linewidths=1.2, zorder=6)
    axis.scatter([slot_left, slot_right], [y_slot, y_slot], s=42, facecolors="none", edgecolors="#b2182b", linewidths=1.2, zorder=6)
    add_double_arrow(axis, true_left, true_right, 1.30, "#b2182b", f"真实间距 ≈ {float(largest['true_gap']):.3f}", 1.34)
    add_double_arrow(axis, slot_left, slot_right, -0.29, "#b2182b", f"槽位间距 = 1/63 ≈ {float(largest['baseline_gap']):.3f}", -0.25)
    axis.text(
        0.70, 1.52,
        "相邻输入切片跨过连续删除区段，\n但 Original PE 仍把它们放在相邻等距槽位",
        ha="center", va="center", fontsize=8, color="#7f1d1d",
    )

    axis.set_xticks([0.0, 0.25, 0.5, 0.75, 1.0])
    axis.set_xticklabels(["0", "0.25", "0.50", "0.75", "1.0"], fontsize=8)
    axis.tick_params(axis="x", length=3, color="#6e7780")
    axis.text(0.5, -0.49, "归一化位置（完整序列端点为 0 和 1）", ha="center", va="top", fontsize=8.5, color="#28343b")
    axis.set_title(
        f"CT-RATE block3 问题示意：50% 连续删片后均匀采样不等于真实空间等距\n"
        f"case_index={case_index} · patient_id={manifest['patient_id']} · N={total} · D={int(manifest['actual_deleted_count'])} · T={len(slices)} · 删除种子={manifest['seed']} · 选择种子={selection_seed} · 可选病例={available_count}",
        fontsize=11, fontweight="bold", pad=16, color="#1f2933",
    )
    figure.tight_layout(pad=1.2)
    skill_scripts = Path("/home/Lim/.agents/skills/nature-figure/scripts")
    sys.path.insert(0, str(skill_scripts))
    from audit_panel_alignment import require_matplotlib_panel_alignment

    stem = f"case_{case_index:04d}_block3_uniform_gap_schematic"
    require_matplotlib_panel_alignment(
        figure,
        json_out=output_dir / f"{stem}.alignment.json",
        tolerance_pt=1.5,
        gutter_tolerance_pt=1.5,
        strict=True,
    )
    figure.savefig(output_dir / f"{stem}.svg", bbox_inches="tight")
    figure.savefig(output_dir / f"{stem}.pdf", bbox_inches="tight")
    figure.savefig(output_dir / f"{stem}.png", dpi=600, bbox_inches="tight")
    figure.savefig(output_dir / f"{stem}.tiff", dpi=600, bbox_inches="tight")
    plt.close(figure)


def main() -> None:
    args = parse_args()
    root = args.block3_root.resolve()
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    case_index, cases = choose_case(root, args.case_index, args.selection_seed, args.deletion_seed)
    manifest, slices, intervals = load_case_data(root, case_index, args.deletion_seed)
    validate_inputs(manifest, slices, intervals)
    write_source_csv(output_dir, manifest, slices, intervals, case_index)
    selection = {
        "selection_seed": args.selection_seed,
        "deletion_seed": args.deletion_seed,
        "case_index": case_index,
        "patient_id": manifest["patient_id"],
        "deletion_fraction": manifest["requested_delete_fraction"],
        "total_slices": len(manifest["deleted_mask"]),
        "deleted_count": manifest["actual_deleted_count"],
        "input_instances": manifest["input_instances"],
        "deleted_segments": manifest["deleted_segments"],
        "available_completed_cases_for_50_percent": cases,
        "uses_acpe": False,
        "retraining": False,
    }
    (output_dir / "selection.json").write_text(json.dumps(selection, indent=2, ensure_ascii=False), encoding="utf-8")
    plot_figure(output_dir, manifest, slices, intervals, case_index, args.selection_seed, len(cases))
    print(json.dumps({"case_index": case_index, "patient_id": manifest["patient_id"], "output_dir": str(output_dir)}, ensure_ascii=False))


if __name__ == "__main__":
    main()
