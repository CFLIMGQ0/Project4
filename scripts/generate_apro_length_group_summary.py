#!/usr/bin/env python3
"""Generate the 2-by-4 APro-CoPE qualitative mechanism summary figure."""

from __future__ import annotations

import gc
import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib.lines import Line2D

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.scripts import generate_apro_dataset_mechanism_figures as generator
from src.scripts import plot_apro_mechanism_preview as preview


OUTPUT_STEM = PROJECT_ROOT / "temp_img/apro_cope_mechanism_length_groups_four_datasets"
DATASETS = ("wle", "chromoscopic", "surgical", "eus")
COLUMN_NAMES = ("WLE", "Chromoendoscopy", "Surgical gastroscopy", "EUS")
GROUPS = ("gt64", "le64")


def load_case(short_name: str, group: str, device: torch.device) -> dict[str, object]:
    metadata_path = PROJECT_ROOT / "temp_img" / f"apro_cope_mechanism_{short_name}_{group}.json"
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    dataset = str(metadata["dataset"])
    fold = int(metadata["fold"])
    records = preview.load_test_records(dataset, fold)
    target = str(metadata["exam_dir"])
    matches = [record for record in records if str(record["exam_dir"]) == target]
    if not matches:
        tail = "/".join(Path(target).parts[-2:])
        matches = [record for record in records if str(record["exam_dir"]).endswith(tail)]
    if len(matches) != 1:
        raise RuntimeError(f"Could not uniquely resolve examination: {target}")

    source_record = matches[0]
    record, _, _ = generator.select_analysis_segment(
        source_record,
        int(metadata["analysis_segment_start_index"]),
        int(metadata["analysis_segment_end_index"]),
    )
    indices = [int(value) for value in metadata["sampled_indices_within_segment"]]
    cache_dataset = preview.build_cache_dataset(records)
    model = preview.load_model("apro_full", dataset, fold, device)
    batch, arrays = preview.make_batch(record, indices, cache_dataset)
    output = preview.infer(model, batch, device, capture_raw_features=True)
    del model
    gc.collect()
    if device.type == "cuda":
        torch.cuda.empty_cache()

    raw = output["apro_raw_coordinates"][: len(indices)]
    context = output["apro_context_coordinates"][: len(indices)]
    acquisition_gap = np.diff(raw)
    slot = np.linspace(0.0, 1.0, len(indices))
    slot_mismatch = 100.0 * (
        np.diff(slot) / np.clip(acquisition_gap, 1e-8, None) - 1.0
    )
    contextual_change = 100.0 * (
        np.diff(context) / np.clip(acquisition_gap, 1e-8, None) - 1.0
    )
    return {
        "metadata": metadata,
        "arrays": arrays,
        "raw": raw,
        "gap_midpoints": 0.5 * (raw[:-1] + raw[1:]),
        "feature_change": preview.normalized_feature_change(
            output["raw_features"][: len(indices)]
        ),
        "slot_mismatch": slot_mismatch,
        "contextual_change": contextual_change,
    }


def sequence_annotation(case: dict[str, object], group: str) -> str:
    metadata = case["metadata"]
    count = int(metadata["analysis_segment_image_count"])
    sampled = int(metadata["sampled_image_count"])
    is_segment = (
        int(metadata["analysis_segment_start_index"]) != 0
        or int(metadata["analysis_segment_end_index"])
        != int(metadata["original_examination_image_count"]) - 1
    )
    noun = "frame segment" if is_segment else "image sequence"
    if group == "gt64":
        return f"{count}-{noun} → {sampled} sampled"
    return f"{count}-{noun}; all retained"


def main() -> None:
    preview.configure_style()
    plt.rcParams.update(
        {
            "font.size": 10.0,
            "axes.labelsize": 10.0,
            "xtick.labelsize": 8.5,
            "ytick.labelsize": 8.5,
        }
    )
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    cases = {
        (group, dataset): load_case(dataset, group, device)
        for group in GROUPS
        for dataset in DATASETS
    }
    image_stacks = {
        key: preview.build_horizontal_slice_stack(case["arrays"])
        for key, case in cases.items()
    }
    stack_height = max(stack.shape[0] for stack in image_stacks.values())
    stack_width = max(stack.shape[1] for stack in image_stacks.values())

    fig = plt.figure(figsize=(21.5, 11.8), facecolor="white")
    grid = fig.add_gridspec(
        7,
        5,
        width_ratios=(0.27, 1.0, 1.0, 1.0, 1.0),
        height_ratios=(0.70, 0.72, 0.96, 0.30, 0.70, 0.72, 0.96),
        left=0.025,
        right=0.994,
        bottom=0.064,
        top=0.965,
        wspace=0.10,
        hspace=0.08,
    )

    data_grid_rows = (0, 1, 2, 4, 5, 6)
    row_label_axes = [fig.add_subplot(grid[row, 0]) for row in data_grid_rows]
    for axis in row_label_axes:
        axis.axis("off")

    image_row_labels = (
        (r"$N>64$", "(64 sampled)"),
        (r"$N\leq64$", "(All retained)"),
    )
    for row, (condition, action) in zip((0, 3), image_row_labels):
        axis = row_label_axes[row]
        axis.text(
            0.02,
            0.98,
            f"{condition}\n{action}",
            ha="left",
            va="top",
            fontsize=12.5,
            fontweight="normal",
        )

    row_titles = (
        "Ordered image\nsequences",
        "Adjacent-image\nfeature changes",
        "Relative position-\ninterval changes",
        "Ordered image\nsequences",
        "Adjacent-image\nfeature changes",
        "Relative position-\ninterval changes",
    )
    for row, (axis, title) in enumerate(zip(row_label_axes, row_titles)):
        is_image_row = row in (0, 3)
        axis.text(
            0.98 if is_image_row else 0.50,
            0.40 if is_image_row else 0.50,
            title,
            ha="right" if is_image_row else "center",
            va="center",
            fontsize=10.8,
            fontweight="normal",
            linespacing=1.10,
        )

    row_axes: list[list[plt.Axes]] = [[] for _ in range(6)]
    for group_index, group in enumerate(GROUPS):
        logical_row_base = group_index * 3
        grid_row_base = group_index * 4
        for column_index, dataset in enumerate(DATASETS):
            case = cases[(group, dataset)]
            raw = np.asarray(case["raw"])
            gap_midpoints = np.asarray(case["gap_midpoints"])
            column = column_index + 1

            image_axis = fig.add_subplot(grid[grid_row_base, column])
            feature_axis = fig.add_subplot(grid[grid_row_base + 1, column])
            interval_axis = fig.add_subplot(grid[grid_row_base + 2, column])
            row_axes[logical_row_base].append(image_axis)
            row_axes[logical_row_base + 1].append(feature_axis)
            row_axes[logical_row_base + 2].append(interval_axis)

            image_stack = image_stacks[(group, dataset)]
            image_height, image_width = image_stack.shape[:2]
            image_left = 0.5 * (stack_width - image_width)
            image_axis.imshow(
                image_stack,
                aspect="equal",
                interpolation="lanczos",
                extent=(
                    image_left,
                    image_left + image_width,
                    image_height,
                    0.0,
                ),
            )
            image_axis.set_xlim(0.0, float(stack_width))
            image_axis.set_ylim(float(stack_height), 0.0)
            image_axis.set_anchor("S")
            image_axis.axis("off")
            image_axis.set_title(
                sequence_annotation(case, group),
                fontsize=8.8,
                fontweight="normal",
                pad=1.5,
            )

            feature_axis.plot(
                raw,
                case["feature_change"],
                color=preview.COLORS["change"],
                linewidth=1.75,
            )
            feature_axis.fill_between(
                raw,
                0.0,
                case["feature_change"],
                color=preview.COLORS["change"],
                alpha=0.18,
            )
            feature_axis.set_xlim(0.0, 1.0)
            feature_axis.set_ylim(0.0, 1.30)
            feature_axis.tick_params(axis="x", labelbottom=False)
            feature_axis.grid(True, color="#E7E7E7", linewidth=0.65)

            interval_axis.axhline(0.0, color="#555555", linestyle="--", linewidth=0.85)
            interval_axis.plot(
                gap_midpoints,
                case["slot_mismatch"],
                color=preview.COLORS["original_pe"],
                linewidth=1.45,
                linestyle=(0, (5, 2)),
            )
            interval_axis.plot(
                gap_midpoints,
                case["contextual_change"],
                color=preview.COLORS["apro_full"],
                linewidth=1.70,
            )
            interval_axis.set_xlim(0.0, 1.0)
            interval_axis.set_ylim(-200.0, 200.0)
            interval_axis.set_xticks((0.0, 0.25, 0.5, 0.75, 1.0))
            interval_axis.grid(True, color="#E7E7E7", linewidth=0.65)

            if column_index == 0:
                feature_axis.tick_params(axis="y", labelleft=True)
                interval_axis.tick_params(axis="y", labelleft=True)
            else:
                feature_axis.tick_params(axis="y", labelleft=False)
                interval_axis.tick_params(axis="y", labelleft=False)

    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    for column_name, axis in zip(COLUMN_NAMES, row_axes[0]):
        position = axis.get_position()
        title_box = axis.title.get_window_extent(renderer=renderer).transformed(
            fig.transFigure.inverted()
        )
        fig.text(
            0.5 * (position.x0 + position.x1),
            title_box.y1 + 0.005,
            column_name,
            ha="center",
            va="bottom",
            fontsize=14.5,
            fontweight="bold",
        )

    legend_handles = (
        Line2D(
            [0],
            [0],
            color=preview.COLORS["original_pe"],
            linewidth=1.8,
            linestyle=(0, (5, 2)),
            label="Sampled-slot/acquisition mismatch",
        ),
        Line2D(
            [0],
            [0],
            color=preview.COLORS["apro_full"],
            linewidth=2.0,
            label="APro-CoPE contextual deformation",
        ),
    )
    fig.legend(
        handles=legend_handles,
        loc="lower right",
        bbox_to_anchor=(0.995, 0.008),
        ncol=2,
        frameon=False,
        fontsize=10.5,
    )
    OUTPUT_STEM.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(
        OUTPUT_STEM.with_suffix(".png"),
        dpi=300,
        facecolor="white",
        bbox_inches="tight",
        pad_inches=0.035,
    )
    fig.savefig(
        OUTPUT_STEM.with_suffix(".pdf"),
        facecolor="white",
        bbox_inches="tight",
        pad_inches=0.035,
    )
    plt.close(fig)
    print(OUTPUT_STEM.with_suffix(".png"))


if __name__ == "__main__":
    main()
