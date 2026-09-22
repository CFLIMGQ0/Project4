#!/usr/bin/env python3
"""绘制三个高斯加权随机删除簇的采样位置示意图。"""
from __future__ import annotations

import argparse
import csv
import gzip
import json
import re
import sys
from collections import deque
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
from matplotlib.patches import FancyArrowPatch, Polygon
from matplotlib.transforms import Bbox
from tqdm import tqdm


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_SOURCE = ROOT / "outputs/ct_rate_680/position_recovery_block3_uniform_gap_schematic_final/selection.json"
DEFAULT_OUTPUT = ROOT / "outputs/ct_rate_680/position_recovery_gaussian_clusters"
DEFAULT_GASTRO_MANIFEST = ROOT / "datasets/image_cache/task3_cache_manifest.jsonl.gz"
DEFAULT_GASTRO_CACHE_ROOT = ROOT / "datasets/image_cache/shared"
DEFAULT_GASTRO_EXAM = "/home/Lim/Project4/datasets/main_data/ZS25014124/ZS0049665016"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-selection", type=Path, default=DEFAULT_SOURCE)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--random-seed", type=int, default=42)
    parser.add_argument("--extra-random-deletions", type=int, default=8)
    parser.add_argument("--layout", choices=["temp1", "temp2"], default="temp1")
    parser.add_argument("--top-row", choices=["markers", "gastroscopy"], default="markers")
    parser.add_argument("--gastro-manifest", type=Path, default=DEFAULT_GASTRO_MANIFEST)
    parser.add_argument("--gastro-cache-root", type=Path, default=DEFAULT_GASTRO_CACHE_ROOT)
    parser.add_argument("--gastro-exam", default=DEFAULT_GASTRO_EXAM)
    parser.add_argument("--gastro-source-start", type=int, default=0)
    parser.add_argument("--gastro-source-end", type=int, default=140)
    return parser.parse_args()


def build_variant(selection: dict, random_seed: int, extra_random_deletions: int, layout: str) -> dict:
    total = int(selection["total_slices"])
    target_instances = int(selection["input_instances"])
    source_lengths = [int(segment["length"]) for segment in selection["deleted_segments"]]
    if len(source_lengths) != 3 or extra_random_deletions < 0:
        raise ValueError("需要三个删除簇，额外随机删除数量不能为负")
    selector = np.random.default_rng(random_seed)
    if layout == "temp2":
        center_fractions = np.asarray([0.28, 0.50, 0.72])
        candidate_windows = [(1, 73), (78, 132), (137, total - 2)]
        sigmas = np.asarray([0.08, 0.045, 0.12]) * (total - 1)
    else:
        center_fractions = np.asarray([0.18, 0.50, 0.83])
        candidate_windows = None
        sigmas = np.asarray([0.08, 0.025, 0.08]) * (total - 1)
    centers = center_fractions * (total - 1)
    if candidate_windows is None:
        candidate_groups = np.array_split(np.arange(1, total - 1), 3)
    else:
        candidate_groups = [np.arange(start, end + 1) for start, end in candidate_windows]
    segments = []
    groups = []
    mask = np.zeros(total, dtype=bool)
    segment_ids = np.full(total, -1, dtype=np.int64)
    group_ids = np.full(total, -1, dtype=np.int64)
    for group_index, (candidates, center, count, sigma) in enumerate(
        zip(candidate_groups, centers, source_lengths, sigmas)
    ):
        weights = np.exp(-0.5 * ((candidates - center) / sigma) ** 2)
        weights /= weights.sum()
        deleted_indices = np.sort(selector.choice(candidates, size=count, replace=False, p=weights))
        mask[deleted_indices] = True
        group_ids[deleted_indices] = group_index
        runs = np.split(deleted_indices, np.flatnonzero(np.diff(deleted_indices) > 1) + 1)
        segment_indices = []
        for part_index, run in enumerate(runs):
            start, end = int(run[0]), int(run[-1])
            segment_indices.append(len(segments))
            segment_ids[run] = len(segments)
            segments.append({
                "segment_index": len(segments),
                "group_index": group_index,
                "part_index": part_index,
                "start_raw_index": start,
                "end_raw_index_inclusive": end,
                "length": int(len(run)),
            })
        groups.append({
            "group_index": group_index,
            "center_raw_index": float(center),
            "sigma_raw_slices": float(sigma),
            "candidate_start": int(candidates[0]),
            "candidate_end": int(candidates[-1]),
            "start_raw_index": int(deleted_indices[0]),
            "end_raw_index_inclusive": int(deleted_indices[-1]),
            "deleted_raw_indices": deleted_indices.tolist(),
            "segment_indices": segment_indices,
        })
    candidates = np.flatnonzero(~mask)[1:-1]
    if layout == "temp2":
        candidates = candidates[candidates > groups[0]["end_raw_index_inclusive"]]
    random_deletions = np.sort(selector.choice(candidates, size=extra_random_deletions, replace=False)).tolist()
    mask[random_deletions] = True
    remaining = np.flatnonzero(~mask).astype(np.int64)
    if len(remaining) < target_instances:
        raise ValueError("删除后切片不足，不能无重复采样")
    selected_slots = np.rint(np.linspace(0, len(remaining) - 1, target_instances)).astype(np.int64)
    selected = remaining[selected_slots]
    if selected[0] != 0 or selected[-1] != total - 1 or len(np.unique(selected)) != target_instances:
        raise ValueError("变体采样没有保留端点或产生重复")
    selected_positions = selected.astype(float) / float(total - 1)
    dense_count = min(5, target_instances)
    dense_spans = selected_positions[dense_count - 1:] - selected_positions[:1 - dense_count]
    dense_start = int(np.argmin(dense_spans))
    gap_values = np.diff(selected)
    large_gap_index = int(np.argmax(gap_values))
    return {
        "case_index": int(selection["case_index"]),
        "patient_id": selection["patient_id"],
        "random_seed": int(random_seed),
        "extra_random_deletions": int(extra_random_deletions),
        "layout": layout,
        "total_slices": total,
        "target_instances": target_instances,
        "concentration_groups": groups,
        "block_segments": segments,
        "random_deleted_raw_indices": random_deletions,
        "deleted_count": int(mask.sum()),
        "actual_delete_fraction": float(mask.mean()),
        "deleted_mask": mask.tolist(),
        "deleted_segment_ids": segment_ids.tolist(),
        "deleted_group_ids": group_ids.tolist(),
        "remaining_raw_count": int(len(remaining)),
        "selected_raw_indices": selected.tolist(),
        "selected_remaining_slots": selected_slots.tolist(),
        "dense_window_input_positions": list(range(dense_start, dense_start + dense_count)),
        "dense_window_raw_indices": selected[dense_start:dense_start + dense_count].tolist(),
        "dense_window_true_span": float(dense_spans[dense_start]),
        "large_gap_input_positions": [large_gap_index, large_gap_index + 1],
        "large_gap_raw_indices": selected[large_gap_index:large_gap_index + 2].tolist(),
        "large_gap_true_span": float(gap_values[large_gap_index] / float(total - 1)),
        "first_22_input_positions": [0, min(21, target_instances - 1)],
        "first_22_raw_indices": selected[[0, min(21, target_instances - 1)]].tolist(),
        "first_22_true_span": float(selected_positions[min(21, target_instances - 1)]),
        "first_22_slot_span": float(min(21, target_instances - 1) / float(target_instances - 1)),
        "first_22_interpretation": "用户指定第1和第22个采样节点；原序列相对间距较小，不代表相邻序号或已测量的胃镜物理邻近。",
        "source_block_lengths": source_lengths,
        "deletion_method": "truncated_gaussian_weighted_without_replacement_plus_uniform",
        "note": "示意图：按三个截断高斯权重逐片无放回随机删除，连片和散点自然产生；不是原始block3实验mask。",
        "figure_claim": "对剩余序列均匀采样不等于真实空间等距；不展示ACPE或实验性能。",
        "schematic_only": True,
        "uses_acpe": False,
        "retraining": False,
    }


def write_source_csv(output_dir: Path, variant: dict) -> None:
    total = variant["total_slices"]
    selected = np.asarray(variant["selected_raw_indices"], dtype=np.int64)
    true_positions = selected.astype(float) / float(total - 1)
    slots = np.linspace(0.0, 1.0, variant["target_instances"])
    random_indices = set(variant["random_deleted_raw_indices"])
    rows: list[dict] = []
    for raw_index, deleted in enumerate(variant["deleted_mask"]):
        rows.append({
            "layer": "full_sequence",
            "input_position": "",
            "raw_index": raw_index,
            "normalized_position": raw_index / float(total - 1),
            "slot_position": "",
            "delete_type": "random_sparse" if raw_index in random_indices else ("gaussian_cluster" if deleted else "retained"),
            "block_index": variant["deleted_segment_ids"][raw_index] if deleted and raw_index not in random_indices else "",
        })
    for input_position, (raw_index, true_position, slot) in enumerate(zip(selected, true_positions, slots)):
        rows.append({
            "layer": "sampled_true_position",
            "input_position": input_position,
            "raw_index": int(raw_index),
            "normalized_position": float(true_position),
            "slot_position": float(slot),
            "delete_type": "sampled_retained",
            "block_index": "",
        })
        rows.append({
            "layer": "original_pe_slot",
            "input_position": input_position,
            "raw_index": int(raw_index),
            "normalized_position": "",
            "slot_position": float(slot),
            "delete_type": "same_sampled_slice",
            "block_index": "",
        })
    with (output_dir / f"case_{variant['case_index']:04d}_gaussian_clusters.csv").open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def add_arrow(axis, left: float, right: float, y: float, color: str, linewidth: float) -> None:
    axis.add_patch(FancyArrowPatch(
        (left, y), (right, y), arrowstyle="<->", mutation_scale=4,
        shrinkA=0, shrinkB=0, linewidth=linewidth, color=color, zorder=5,
    ))


def find_clear_region_label(true_x: np.ndarray, slot_x: np.ndarray,
                            left_index: int, right_index: int) -> tuple[float, float]:
    best: tuple[float, float, float] | None = None
    for fraction in np.linspace(0.25, 0.75, 51):
        left = (1.0 - fraction) * slot_x[left_index] + fraction * true_x[left_index]
        right = (1.0 - fraction) * slot_x[right_index] + fraction * true_x[right_index]
        low, high = min(left, right), max(left, right)
        if high - low < 0.02:
            continue
        line_x = (1.0 - fraction) * slot_x + fraction * true_x
        candidates = np.linspace(low + 0.006, high - 0.006, 512)
        clearance = np.min(np.abs(candidates[:, None] - line_x[None, :]), axis=1)
        candidate_index = int(np.argmax(clearance))
        score = float(clearance[candidate_index])
        if best is None or score > best[0]:
            best = (score, float(candidates[candidate_index]), float(fraction))
    if best is None:
        fraction = 0.5
        return (float((true_x[left_index] + true_x[right_index]
                      + slot_x[left_index] + slot_x[right_index]) / 4.0), fraction)
    return best[1], best[2]


def draw_region_letter(axis, letter: str, x: float, y: float, color: str) -> None:
    axis.text(x, y, letter, ha="center", va="center", fontsize=11,
              fontweight="bold", color=color, rasterized=True, zorder=5)


def load_gastroscopy_frames(args: argparse.Namespace, variant: dict) -> list[np.ndarray]:
    rows = []
    with gzip.open(args.gastro_manifest, "rt", encoding="utf-8") as stream:
        for line in tqdm(stream, desc="读取胃镜缓存索引", unit="行"):
            row = json.loads(line)
            if row["image_path"].rsplit("/img/", 1)[0] == args.gastro_exam.rstrip("/"):
                rows.append(row)
    rows.sort(key=lambda row: [int(part) if part.isdigit() else part
                              for part in re.split(r"(\d+)", Path(row["image_path"]).name)])
    target_instances = variant["target_instances"]
    start = max(0, int(args.gastro_source_start))
    end = min(len(rows) - 1, int(args.gastro_source_end))
    if end < start or end - start + 1 < target_instances:
        raise ValueError(
            f"胃镜连续帧范围无效：start={start}, end={end}, 需要至少 {target_instances} 张"
        )
    source_rows = rows[start:end + 1]
    selected_indices = start + np.rint(
        np.linspace(0, len(source_rows) - 1, target_instances)
    ).astype(np.int64)
    if len(np.unique(selected_indices)) != target_instances:
        raise ValueError("胃镜示例图像不能重复")
    frames = []
    provenance = []
    for input_position, source_index in enumerate(tqdm(selected_indices, desc="加载真实胃镜帧", unit="张")):
        row = rows[int(source_index)]
        cache_path = args.gastro_cache_root / row["cache_relpath"]
        frame = np.load(cache_path, allow_pickle=False)
        if frame.ndim != 3 or frame.shape[-1] != 3 or frame.dtype != np.uint8:
            raise ValueError(f"不是 uint8 RGB 图像：{cache_path}")
        frames.append(frame)
        provenance.append({
            "sampled_input_position": input_position,
            "schematic_raw_index": variant["selected_raw_indices"][input_position],
            "node_position": variant["selected_raw_indices"][input_position] / float(variant["total_slices"] - 1),
            "source_sequence_index": int(source_index),
            "image_path": row["image_path"],
            "cache_path": str(cache_path.resolve()),
            "displayed": True,
        })
    variant["gastroscopy"] = {
        "exam_dir": args.gastro_exam,
        "manifest": str(args.gastro_manifest.resolve()),
        "available_frames": len(rows),
        "source_frame_range": [start, end],
        "frame_selection": "在同一检查的连续暖色帧区间内按时间戳均匀选择 T 张，无重复；每帧与一个采样位置节点一一对应。",
        "image_processing": "真实RGB缓存，只做侧向投影与黑色边框，不合成组织纹理，不调色。",
        "occlusion": "左侧靠前图像覆盖右侧靠后图像",
        "frames": provenance,
    }
    variant["figure_claim"] = "示意：剩余序列均匀采样不等于原序列等距；图像位置不代表胃镜物理空间坐标。"
    return frames


def make_endoscopy_card(frame: np.ndarray) -> np.ndarray:
    image = Image.fromarray(frame)
    width, height = image.size
    crop_left = max(1, int(round(width * 0.10)))
    crop_top = max(1, int(round(height * 0.05)))
    crop_bottom = max(crop_top + 1, int(round(height * 0.10)))
    image = image.crop((crop_left, crop_top, width, height - crop_bottom))
    rgba = np.asarray(image.convert("RGBA")).copy()
    luminance = rgba[..., :3].mean(axis=2)
    dark = luminance < 10.0
    edge_dark = np.zeros(dark.shape, dtype=bool)
    queue = deque()
    height, width = dark.shape
    for x in range(width):
        if dark[0, x]:
            edge_dark[0, x] = True
            queue.append((0, x))
        if dark[height - 1, x]:
            edge_dark[height - 1, x] = True
            queue.append((height - 1, x))
    for y in range(height):
        if dark[y, 0]:
            edge_dark[y, 0] = True
            queue.append((y, 0))
        if dark[y, width - 1]:
            edge_dark[y, width - 1] = True
            queue.append((y, width - 1))
    while queue:
        y, x = queue.popleft()
        for delta_y, delta_x in ((-1, -1), (-1, 0), (-1, 1), (0, -1),
                                 (0, 1), (1, -1), (1, 0), (1, 1)):
            next_y, next_x = y + delta_y, x + delta_x
            if (0 <= next_y < height and 0 <= next_x < width
                    and dark[next_y, next_x] and not edge_dark[next_y, next_x]):
                edge_dark[next_y, next_x] = True
                queue.append((next_y, next_x))
    rgba[edge_dark, 3] = 0
    bordered = Image.fromarray(rgba)
    width, height = bordered.size
    rise = int(round(height * 0.18))
    projected = bordered.transform(
        (width, height + rise), Image.Transform.AFFINE,
        (1.0, 0.0, 0.0, rise / float(width - 1), 1.0, -rise),
        resample=Image.Resampling.BICUBIC, fillcolor=(0, 0, 0, 0),
    )
    return np.asarray(projected)


def draw_endoscopy_sequence(axis, variant: dict, raw_x: np.ndarray, y_node: float,
                            frames: list[np.ndarray]) -> None:
    total = variant["total_slices"]
    selected = np.asarray(variant["selected_raw_indices"], dtype=np.int64)
    if len(frames) != len(selected):
        raise ValueError("胃镜图像数量必须与采样位置节点一致")
    card_width = 0.036
    bottom = y_node + 0.20
    top = bottom + 0.64
    axis.vlines(raw_x[selected], y_node, bottom + 0.05,
                color="#557a95", linewidth=0.6, alpha=0.85, zorder=2)
    for input_position in reversed(range(len(selected))):
        raw_index = int(selected[input_position])
        axis.imshow(
            make_endoscopy_card(frames[input_position]),
            extent=(raw_x[raw_index] - card_width / 2.0,
                    raw_x[raw_index] + card_width / 2.0, bottom, top),
            origin="upper", aspect="auto", interpolation="bilinear",
            zorder=3.0 + (total - raw_index) / total,
        )


def draw(output_dir: Path, variant: dict, top_row: str,
         gastro_frames: list[np.ndarray] | None = None) -> None:
    total = variant["total_slices"]
    selected = np.asarray(variant["selected_raw_indices"], dtype=np.int64)
    raw_x = np.arange(total, dtype=float) / float(total - 1)
    true_x = selected.astype(float) / float(total - 1)
    slot_x = np.linspace(0.0, 1.0, variant["target_instances"])
    deleted = np.asarray(variant["deleted_mask"], dtype=bool)

    mpl.rcParams.update({
        "font.family": "sans-serif",
        "font.sans-serif": ["Noto Sans CJK SC", "Arial", "Helvetica", "DejaVu Sans", "sans-serif"],
        "pdf.fonttype": 42,
        "svg.fonttype": "none",
        "axes.spines.right": False,
        "axes.spines.top": False,
        "axes.linewidth": 0.8,
    })
    figure, axis = plt.subplots(figsize=(7.2, 4.1 if top_row == "gastroscopy" else 4.8))
    axis.set_xlim(-0.025 if top_row == "gastroscopy" else -0.015,
                  1.025 if top_row == "gastroscopy" else 1.015)
    axis.set_ylim(-0.68, 2.12 if top_row == "gastroscopy" else 2.55)
    axis.axis("off")
    y_full, y_true, y_slot = 2.0, 1.0, 0.0
    track_heights = [y_true, y_slot] if top_row == "gastroscopy" else [y_full, y_true, y_slot]
    axis.hlines(track_heights, 0.0, 1.0, color="#d2d6da", linewidth=0.8, zorder=0)

    if top_row == "gastroscopy":
        if gastro_frames is None:
            raise ValueError("胃镜模式必须使用真实缓存图像")
        draw_endoscopy_sequence(axis, variant, raw_x, y_true, gastro_frames)
    else:
        for group in variant["concentration_groups"]:
            axis.text(group["center_raw_index"] / float(total - 1), 2.34,
                      f"高斯删除簇 {group['group_index'] + 1}",
                      ha="center", va="bottom", fontsize=7, color="#a33f43")

        deleted_indices = np.flatnonzero(deleted)
        deleted_runs = np.split(deleted_indices, np.flatnonzero(np.diff(deleted_indices) > 1) + 1)
        for run in deleted_runs:
            if not len(run):
                continue
            start = (int(run[0]) - 0.4) / float(total - 1)
            end = (int(run[-1]) + 0.4) / float(total - 1)
            axis.fill_between([start, end], 1.76, 2.24,
                              color="#efb1a3", alpha=0.68, linewidth=0, zorder=0)

        retained = ~deleted
        axis.vlines(raw_x[retained], y_full - 0.10, y_full + 0.10, color="#557a95", linewidth=0.5, alpha=0.75)
        axis.vlines(raw_x[deleted], y_full - 0.12, y_full + 0.12, color="#c44e52", linewidth=0.7, alpha=0.95)
        axis.scatter(raw_x[retained], np.full(retained.sum(), y_full), s=5, color="#557a95", zorder=3)
        axis.scatter(raw_x[deleted], np.full(deleted.sum(), y_full), s=5, color="#c44e52", zorder=3)
        axis.text(0.99, 2.53, "红色：删除切片",
                  ha="right", va="top", fontsize=7, color="#4b5563")

    axis.vlines(true_x, y_true - 0.10, y_true + 0.10, color="#2166ac", linewidth=0.65, alpha=0.7)
    axis.scatter(true_x, np.full(len(true_x), y_true), s=23, color="#2166ac", edgecolor="white", linewidth=0.35, zorder=4)
    axis.vlines(slot_x, y_slot - 0.10, y_slot + 0.10, color="#d27b2d", linewidth=0.65, alpha=0.7)
    axis.scatter(slot_x, np.full(len(slot_x), y_slot), s=23, color="#d27b2d", edgecolor="white", linewidth=0.35, zorder=4)

    connector_indices = range(len(selected))
    for index in connector_indices:
        axis.plot([true_x[index], slot_x[index]], [y_true, y_slot], color="#a5abb0", linewidth=0.35, alpha=0.45, zorder=1)

    annotation_color = "#b2182b"
    first_position, second_position = variant["first_22_input_positions"]
    first_true, second_true = true_x[first_position], true_x[second_position]
    first_slot, second_slot = slot_x[first_position], slot_x[second_position]
    axis.add_patch(Polygon(
        [(first_true, y_true), (second_true, y_true),
         (second_slot, y_slot), (first_slot, y_slot)],
        closed=True, facecolor="#f2a3a3", edgecolor="none", alpha=0.24, zorder=0.25,
    ))
    label_x = (first_true + second_true + first_slot + second_slot) / 4.0
    draw_region_letter(axis, "A", label_x, 0.50, annotation_color)
    for input_position in (first_position, second_position):
        axis.plot([true_x[input_position], slot_x[input_position]], [y_true, y_slot],
                  color=annotation_color, linewidth=0.8, alpha=0.8, zorder=2)
    axis.scatter([first_true, second_true], [y_true, y_true], s=39,
                 facecolors="none", edgecolors=annotation_color, linewidths=1.2, zorder=7)
    axis.scatter([first_slot, second_slot], [y_slot, y_slot], s=39,
                 facecolors="none", edgecolors=annotation_color, linewidths=1.2, zorder=7)
    distance_arrow_y = -0.18
    add_arrow(axis, first_slot, second_slot, distance_arrow_y, annotation_color, 1.0)
    axis.text((first_slot + second_slot) / 2, distance_arrow_y - 0.04,
              "0.336",
              ha="center", va="top", fontsize=7, color=annotation_color)

    true_distance_arrow_y = 1.10 if top_row == "gastroscopy" else 1.28
    add_arrow(axis, first_true, second_true, true_distance_arrow_y, annotation_color, 1.0)

    largest_index = int(variant["large_gap_input_positions"][0])
    axis.add_patch(Polygon(
        [(true_x[largest_index], y_true), (true_x[largest_index + 1], y_true),
         (slot_x[largest_index + 1], y_slot), (slot_x[largest_index], y_slot)],
        closed=True, facecolor="#8ecae6", edgecolor="none", alpha=0.25, zorder=0.25,
    ))
    label_x = (true_x[largest_index] + true_x[largest_index + 1]
               + slot_x[largest_index] + slot_x[largest_index + 1]) / 4.0
    draw_region_letter(axis, "B", label_x, 0.50, "#2166ac")
    for input_position in (largest_index, largest_index + 1):
        axis.plot([true_x[input_position], slot_x[input_position]], [y_true, y_slot],
                  color=annotation_color, linewidth=0.8, alpha=0.8, zorder=2)
    gap_arrow_height = 1.10 if top_row == "gastroscopy" else 1.28
    add_arrow(axis, true_x[largest_index], true_x[largest_index + 1], gap_arrow_height, annotation_color, 1.0)
    add_arrow(axis, slot_x[largest_index], slot_x[largest_index + 1], distance_arrow_y, annotation_color, 1.0)
    axis.scatter([true_x[largest_index], true_x[largest_index + 1]], [y_true, y_true], s=39,
                 facecolors="none", edgecolors=annotation_color, linewidths=1.2, zorder=6)
    axis.scatter([slot_x[largest_index], slot_x[largest_index + 1]], [y_slot, y_slot], s=39,
                 facecolors="none", edgecolors=annotation_color, linewidths=1.2, zorder=6)
    axis.text((slot_x[largest_index] + slot_x[largest_index + 1]) / 2, distance_arrow_y - 0.04,
              "0.016",
              ha="center", va="top", fontsize=7, color=annotation_color)

    label_position = -0.028 if top_row == "gastroscopy" else -0.012
    axis.text(label_position, y_true + 0.44 if top_row == "gastroscopy" else y_full,
              "Sampled endoscopic\nimages" if top_row == "gastroscopy" else "Complete CT\nsequence",
              ha="right", va="center", fontsize=8,
              fontweight="bold", color="#28343b")
    true_label = ("Sample positions\n(physical space)" if top_row == "gastroscopy"
                  else "Sampled-slice positions\n(real space)")
    axis.text(label_position, y_true, true_label, ha="right", va="center", fontsize=8,
              fontweight="bold", color="#28343b")
    axis.text(label_position, y_slot, "Original PE\npositions",
              ha="right", va="center", fontsize=8,
              fontweight="bold", color="#28343b")

    for tick in [0.0, 0.25, 0.5, 0.75, 1.0]:
        axis.text(tick, -0.34, f"{tick:g}", ha="center", va="top", fontsize=7, color="#6e7780")
    figure.tight_layout(pad=1.2)
    skill_scripts = Path("/home/Lim/.agents/skills/nature-figure/scripts")
    sys.path.insert(0, str(skill_scripts))
    from audit_panel_alignment import require_matplotlib_panel_alignment

    stem = f"case_{variant['case_index']:04d}_gaussian_clusters"
    require_matplotlib_panel_alignment(
        figure,
        json_out=output_dir / f"{stem}.alignment.json",
        tolerance_pt=1.5,
        gutter_tolerance_pt=1.5,
        strict=True,
    )
    export_bbox = "tight"
    if top_row == "gastroscopy":
        original_dpi = figure.dpi
        figure.set_dpi(600)
        figure.canvas.draw()
        canvas = np.asarray(figure.canvas.buffer_rgba())
        content = np.any(canvas[..., :3] < 250, axis=2)
        content_rows = np.flatnonzero(content.any(axis=1))
        content_columns = np.flatnonzero(content.any(axis=0))
        export_bbox = Bbox.from_extents(
            content_columns[0] / figure.dpi,
            (canvas.shape[0] - content_rows[-1] - 1) / figure.dpi,
            (content_columns[-1] + 1) / figure.dpi,
            (canvas.shape[0] - content_rows[0]) / figure.dpi,
        ).padded(0.04)
        figure.set_dpi(original_dpi)
    figure.savefig(output_dir / f"{stem}.svg", bbox_inches=export_bbox)
    figure.savefig(output_dir / f"{stem}.pdf", bbox_inches=export_bbox)
    figure.savefig(output_dir / f"{stem}.png", dpi=600, bbox_inches=export_bbox)
    figure.savefig(output_dir / f"{stem}.tiff", dpi=600, bbox_inches=export_bbox)
    plt.close(figure)


def main() -> None:
    args = parse_args()
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    source = json.loads(args.source_selection.resolve().read_text(encoding="utf-8"))
    variant = build_variant(source, args.random_seed, args.extra_random_deletions, args.layout)
    variant["top_row"] = args.top_row
    gastro_frames = load_gastroscopy_frames(args, variant) if args.top_row == "gastroscopy" else None
    write_source_csv(output_dir, variant)
    (output_dir / "selection.json").write_text(json.dumps(variant, indent=2, ensure_ascii=False), encoding="utf-8")
    draw(output_dir, variant, args.top_row, gastro_frames)
    print(json.dumps({
        "case_index": variant["case_index"],
        "patient_id": variant["patient_id"],
        "deleted_count": variant["deleted_count"],
        "output_dir": str(output_dir),
    }, ensure_ascii=False))


if __name__ == "__main__":
    main()
