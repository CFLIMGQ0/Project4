#!/usr/bin/env python3
"""Generate matched APro-CoPE mechanism figures for one gastroscopy dataset."""

from __future__ import annotations

import argparse
import copy
import gc
import json
import sys
from pathlib import Path

import numpy as np
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.scripts import plot_apro_mechanism_preview as preview


FILE_NAMES = {
    "regular_white_light": "wle",
    "chromoscopic": "chromoscopic",
    "surgical": "surgical",
    "ultrasound": "eus",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", choices=tuple(FILE_NAMES), required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--candidates-per-fold", type=int, default=12)
    parser.add_argument(
        "--exam-dir",
        type=str,
        default=None,
        help="Use one exact examination instead of automatic candidate selection.",
    )
    parser.add_argument(
        "--segment-start",
        type=int,
        default=0,
        help="Zero-based first frame retained from an explicitly selected examination.",
    )
    parser.add_argument(
        "--segment-end",
        type=int,
        default=None,
        help="Zero-based inclusive last frame retained from an explicitly selected examination.",
    )
    parser.add_argument(
        "--output-suffix",
        type=str,
        default="",
        help="Optional suffix appended to the dataset-specific output stems.",
    )
    parser.add_argument(
        "--sampled-indices",
        type=str,
        default=None,
        help="Optional comma-separated zero-based indices within the selected segment.",
    )
    parser.add_argument("--output-dir", type=Path, default=preview.PROJECT_ROOT / "temp_img")
    parser.add_argument("--seed", type=int, default=2026)
    return parser.parse_args()


def mechanism_score(output: dict[str, np.ndarray], indices: list[int]) -> tuple[float, dict[str, float]]:
    raw = output["apro_raw_coordinates"][: len(indices)]
    context = output["apro_context_coordinates"][: len(indices)]
    acquisition_gap = np.diff(raw)
    valid = acquisition_gap > 1e-8
    deformation = 100.0 * (
        np.diff(context)[valid] / np.clip(acquisition_gap[valid], 1e-8, None) - 1.0
    )
    changes = preview.normalized_feature_change(output["raw_features"][: len(indices)])
    if deformation.size == 0:
        return -float("inf"), {}

    abs_p90 = float(np.percentile(np.abs(deformation), 90))
    deformation_range = float(np.percentile(deformation, 95) - np.percentile(deformation, 5))
    two_sided = float(min(max(float(deformation.max()), 0.0), max(float(-deformation.min()), 0.0)))
    change_spread = float(np.percentile(changes, 95) - np.percentile(changes, 20))
    score = abs_p90 + 0.30 * deformation_range + 0.20 * two_sided + 20.0 * change_spread
    return score, {
        "absolute_deformation_p90_pct": abs_p90,
        "deformation_p5_to_p95_range_pct": deformation_range,
        "two_sided_deformation_pct": two_sided,
        "feature_change_spread": change_spread,
    }


def select_analysis_segment(
    record: dict[str, object], start: int, end: int | None
) -> tuple[dict[str, object], int, int]:
    """Return a record restricted to one inclusive acquisition-ordered segment."""
    image_paths = list(record["image_paths"])
    if not image_paths:
        raise ValueError("The selected examination contains no images.")
    segment_start = int(start)
    segment_end = len(image_paths) - 1 if end is None else int(end)
    if segment_start < 0 or segment_end < segment_start or segment_end >= len(image_paths):
        raise ValueError(
            f"Invalid segment [{segment_start}, {segment_end}] for "
            f"an examination with {len(image_paths)} images."
        )

    segmented = copy.deepcopy(record)
    segmented["image_paths"] = image_paths[segment_start : segment_end + 1]
    segmented["img_num"] = len(segmented["image_paths"])
    return segmented, segment_start, segment_end


def main() -> None:
    args = parse_args()
    preview.configure_style()
    device = torch.device(args.device)
    output_dir = args.output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    best: dict[str, object] | None = None
    if args.exam_dir:
        target = str(args.exam_dir)
        for fold in range(1, 6):
            records = preview.load_test_records(args.dataset, fold)
            matches = [record for record in records if str(record["exam_dir"]) == target]
            if not matches:
                continue
            source_record = matches[0]
            record, segment_start, segment_end = select_analysis_segment(
                source_record, args.segment_start, args.segment_end
            )
            cache_dataset = preview.build_cache_dataset(records)
            model = preview.load_model("apro_full", args.dataset, fold, device)
            if args.sampled_indices:
                indices = [
                    int(value.strip())
                    for value in args.sampled_indices.split(",")
                    if value.strip()
                ]
                if (
                    not indices
                    or indices != sorted(set(indices))
                    or indices[0] < 0
                    or indices[-1] >= len(record["image_paths"])
                ):
                    raise ValueError(
                        "--sampled-indices must be a nonempty, strictly increasing list "
                        "within the selected segment."
                    )
            else:
                indices = preview.uniform_indices(len(record["image_paths"]), 64)
            batch, arrays = preview.make_batch(record, indices, cache_dataset)
            output = preview.infer(model, batch, device, capture_raw_features=True)
            score, components = mechanism_score(output, indices)
            best = {
                "score": score,
                "score_components": components,
                "fold": fold,
                "record": record,
                "source_record": source_record,
                "segment_start": segment_start,
                "segment_end": segment_end,
                "indices": indices,
                "arrays": arrays,
                "output": output,
            }
            del model
            gc.collect()
            if device.type == "cuda":
                torch.cuda.empty_cache()
            break

        if best is None:
            raise RuntimeError(f"Examination not found in {args.dataset} test folds: {target}")

    else:
        for fold in range(1, 6):
            records = preview.load_test_records(args.dataset, fold)
            eligible = [record for record in records if len(record["image_paths"]) >= 64]
            candidates = sorted(
                eligible, key=lambda record: len(record["image_paths"]), reverse=True
            )[: max(1, int(args.candidates_per_fold))]
            cache_dataset = preview.build_cache_dataset(records)
            model = preview.load_model("apro_full", args.dataset, fold, device)

            for case_index, record in enumerate(candidates):
                rng = np.random.default_rng(args.seed + fold * 1009 + case_index * 53)
                indices = preview.stratified_jitter_indices(
                    len(record["image_paths"]), 64, rng
                )
                batch, arrays = preview.make_batch(record, indices, cache_dataset)
                output = preview.infer(model, batch, device, capture_raw_features=True)
                score, components = mechanism_score(output, indices)
                if best is None or score > float(best["score"]):
                    best = {
                        "score": score,
                        "score_components": components,
                        "fold": fold,
                        "record": record,
                        "indices": indices,
                        "arrays": arrays,
                        "output": output,
                    }

            del model
            gc.collect()
            if device.type == "cuda":
                torch.cuda.empty_cache()

    if best is None:
        raise RuntimeError(f"No eligible examination found for {args.dataset}")

    short_name = FILE_NAMES[args.dataset]
    suffix = str(args.output_suffix).strip()
    if suffix and not suffix.startswith("_"):
        suffix = f"_{suffix}"
    record = best["record"]
    sequence_name = "EUS" if short_name == "eus" else short_name
    image_panel_title = (
        f"(a) {len(best['indices'])} images sampled from a "
        f"{len(record['image_paths'])}-image {sequence_name} sequence"
    )
    if len(record["image_paths"]) > 64:
        image_panel_title += " (N > 64)"
    preview.plot_figure3b_combined_with_image_strip(
        best["output"],
        best["arrays"],
        best["indices"],
        output_dir,
        output_stem=f"apro_cope_mechanism_{short_name}{suffix}",
        stack_stem=f"apro_cope_image_stack_{short_name}{suffix}",
        image_panel_title=image_panel_title,
    )
    source_record = best.get("source_record", record)
    segment_start = int(best.get("segment_start", 0))
    segment_end = int(best.get("segment_end", len(source_record["image_paths"]) - 1))
    output = best["output"]
    relative_indices = [int(value) for value in best["indices"]]
    metadata = {
        "dataset": args.dataset,
        "dataset_display_name": preview.DATASET_NAMES[args.dataset],
        "length_group": "gt64" if len(record["image_paths"]) > 64 else "le64",
        "length_condition": "N > 64" if len(record["image_paths"]) > 64 else "N <= 64",
        "fold": int(best["fold"]),
        "exam_dir": str(source_record["exam_dir"]),
        "original_examination_image_count": len(source_record["image_paths"]),
        "analysis_segment_start_index": segment_start,
        "analysis_segment_end_index": segment_end,
        "analysis_segment_image_count": len(record["image_paths"]),
        "sampled_image_count": len(best["indices"]),
        "sampled_indices_within_segment": relative_indices,
        "sampled_indices_in_original_examination": [
            segment_start + value for value in relative_indices
        ],
        "labels": [int(value) for value in record["labels"]],
        "selection_score": float(best["score"]),
        "selection_score_components": best["score_components"],
        "raw_coordinate_first": float(output["apro_raw_coordinates"][0]),
        "raw_coordinate_last": float(
            output["apro_raw_coordinates"][len(relative_indices) - 1]
        ),
        "context_coordinate_first": float(output["apro_context_coordinates"][0]),
        "context_coordinate_last": float(
            output["apro_context_coordinates"][len(relative_indices) - 1]
        ),
        "selection_scope": (
            f"explicit acquisition-ordered segment selected after frame-level "
            f"{preview.DATASET_NAMES[args.dataset]} image verification"
            if args.exam_dir
            else f"top {args.candidates_per_fold} longest eligible test examinations per fold across five folds"
        ),
    }
    (output_dir / f"apro_cope_mechanism_{short_name}{suffix}.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(metadata, ensure_ascii=False))


if __name__ == "__main__":
    main()
