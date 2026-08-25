#!/usr/bin/env python3
"""生成 APro-CoPE 机制验证预览图，不修改论文正文。

图1和图2读取真实检查点、测试折与离线图像缓存；图4使用论文表3中的
五折汇总结果。图2仅用于版式与分析流程预览，正式入稿前应扩展到全部
折外测试检查并增加重采样次数。
"""

from __future__ import annotations

import argparse
import copy
import csv
import json
import math
import random
import sys
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image, ImageDraw
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from tqdm import tqdm

ROOT = Path(__file__).resolve().parents[1]
PROJECT_ROOT = ROOT.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from exp_8 import build_exp8_model
from scripts.task3_main_model_5fold import apply_watch_mask
from training.data import MILBagDataset, mil_collate_fn


LABEL_NAMES = (
    "Esophageal SMT",
    "Esophageal mucosal lesion",
    "Gastritis",
)
DATASET_NAMES = {
    "regular_white_light": "WLE",
    "chromoscopic": "Chromoscopic",
    "surgical": "Surgical",
    "ultrasound": "EUS",
}
POSITION_ROOT = PROJECT_ROOT / "outputs/train_runs/task3/t3_apro_cope_ablation"
RECORDS_CACHE = PROJECT_ROOT / "outputs/train_runs/task3/t3_main_model/records_cache.json"
CACHE_ROOT = PROJECT_ROOT / "datasets/image_cache"
CACHE_DIR = CACHE_ROOT / "shared"
CACHE_MANIFEST = CACHE_ROOT / "task3_cache_manifest.jsonl.gz"


COLORS = {
    "no_pe": "#7A7A7A",
    "original_pe": "#3B6FB6",
    "apro_full": "#D14B40",
    "raw": "#202020",
    "change": "#D6A23A",
    "positive": "#2C8C6B",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=PROJECT_ROOT / "temp_img")
    parser.add_argument("--dataset", choices=tuple(DATASET_NAMES), default="chromoscopic")
    parser.add_argument("--fold", type=int, default=1)
    parser.add_argument("--device", default="cuda:3")
    parser.add_argument("--preview-cases", type=int, default=12)
    parser.add_argument("--resamples", type=int, default=8)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--skip-inference", action="store_true")
    return parser.parse_args()


def configure_style() -> None:
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 10.5,
            "axes.titlesize": 12,
            "axes.labelsize": 10.5,
            "xtick.labelsize": 9,
            "ytick.labelsize": 9,
            "legend.fontsize": 9,
            "axes.linewidth": 0.8,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "savefig.facecolor": "white",
        }
    )


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def load_model(variant: str, dataset: str, fold: int, device: torch.device) -> torch.nn.Module:
    run_root = POSITION_ROOT / variant
    cfg = read_json(run_root / "experiment_config.json")
    params = dict(cfg["model"]["params"])
    params.pop("image_aux_weight", None)
    model = build_exp8_model(
        model_name=str(cfg["model"]["model_name"]),
        num_labels=len(LABEL_NAMES),
        pretrained=False,
        **params,
    )
    checkpoint_path = run_root / dataset / f"fold_{fold}" / "checkpoints/best_macro_f1.ckpt"
    try:
        checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    except TypeError:
        checkpoint = torch.load(checkpoint_path, map_location="cpu")
    missing, unexpected = model.load_state_dict(checkpoint["model_state"], strict=False)
    if missing or unexpected:
        raise RuntimeError(
            f"检查点不匹配：{variant}, missing={list(missing)}, unexpected={list(unexpected)}"
        )
    model.to(device).eval()
    return model


def load_test_records(dataset: str, fold: int) -> list[dict[str, Any]]:
    payload = read_json(RECORDS_CACHE)
    records = copy.deepcopy(payload["records"])
    apply_watch_mask(records, enabled=True)
    record_map = {str(record["exam_dir"]): record for record in records}
    manifest = POSITION_ROOT / "apro_full" / dataset / f"fold_{fold}" / "split_manifest.csv"
    selected: list[dict[str, Any]] = []
    with manifest.open("r", encoding="utf-8-sig", newline="") as file:
        for row in csv.DictReader(file):
            if str(row.get("split", "")).lower() != "test":
                continue
            exam_dir = str(row["exam_dir"])
            if exam_dir not in record_map:
                raise KeyError(f"测试检查未出现在记录缓存中：{exam_dir}")
            selected.append(record_map[exam_dir])
    return selected


def build_cache_dataset(records: list[dict[str, Any]]) -> MILBagDataset:
    return MILBagDataset(
        records=records,
        task_name="task2",
        max_instances=64,
        min_instances=1,
        bag_sampling_strategy="uniform",
        is_train=False,
        image_size=224,
        random_instance_dropout=0.0,
        image_cache_mode="disk",
        image_cache_dir=CACHE_DIR,
        image_cache_manifest=CACHE_MANIFEST,
        memory_cache_size=256,
        split_name="test",
    )


def uniform_indices(num_images: int, keep: int = 64) -> list[int]:
    if num_images <= keep:
        return list(range(num_images))
    return np.rint(np.linspace(0, num_images - 1, keep)).astype(int).tolist()


def stratified_jitter_indices(num_images: int, keep: int, rng: np.random.Generator) -> list[int]:
    if num_images <= keep:
        return list(range(num_images))
    edges = np.linspace(0, num_images, keep + 1)
    indices: list[int] = []
    for left_raw, right_raw in zip(edges[:-1], edges[1:]):
        left = int(math.floor(left_raw))
        right = max(left + 1, int(math.ceil(right_raw)))
        indices.append(int(rng.integers(left, min(right, num_images))))
    indices[0] = 0
    indices[-1] = num_images - 1
    indices = sorted(set(indices))
    if len(indices) < keep:
        remaining = [value for value in range(num_images) if value not in set(indices)]
        rng.shuffle(remaining)
        indices.extend(remaining[: keep - len(indices)])
    return sorted(indices[:keep])


def make_batch(
    record: dict[str, Any],
    indices: list[int],
    cache_dataset: MILBagDataset,
) -> tuple[dict[str, Any], list[np.ndarray]]:
    arrays: list[np.ndarray] = []
    images: list[torch.Tensor] = []
    paths = [record["image_paths"][index] for index in indices]
    for path in paths:
        array = cache_dataset._load_image_array(path)
        arrays.append(array)
        images.append(cache_dataset.transform(Image.fromarray(array, mode="RGB")))
    item: dict[str, Any] = {
        "images": torch.stack(images),
        "label": torch.tensor(record["labels"], dtype=torch.float32),
        "exam_dir": record["exam_dir"],
        "image_paths": paths,
        "instance_indices": torch.tensor(indices, dtype=torch.long),
        "original_image_count": len(record["image_paths"]),
        "report_title": record.get("report_title", ""),
        "img_num": int(record.get("img_num", len(record["image_paths"]))),
        "meta": {},
    }
    text_tensors = cache_dataset._text_tensors_for_record(record)
    if text_tensors:
        item.update(text_tensors)
    return mil_collate_fn([item]), arrays


def move_batch(batch: dict[str, Any], device: torch.device) -> dict[str, Any]:
    return {
        key: value.to(device, non_blocking=True) if isinstance(value, torch.Tensor) else value
        for key, value in batch.items()
    }


@torch.inference_mode()
def infer(
    model: torch.nn.Module,
    batch: dict[str, Any],
    device: torch.device,
    capture_raw_features: bool = False,
) -> dict[str, np.ndarray]:
    batch_device = move_batch(batch, device)
    captured: dict[str, torch.Tensor] = {}
    hook_handle = None
    if capture_raw_features and getattr(model, "apro_positioner", None) is not None:
        def capture_input(module: torch.nn.Module, inputs: tuple[torch.Tensor, ...]) -> None:
            del module
            captured["raw_features"] = inputs[0].detach()

        hook_handle = model.apro_positioner.transition_projector.register_forward_pre_hook(capture_input)

    autocast_enabled = device.type == "cuda"
    with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=autocast_enabled):
        context_features, label_embeds, attention, extras = model.encode_long_mil(
            batch_device["images"],
            batch_device["mask"],
            batch_device["instance_indices"],
            batch_device["original_image_counts"],
        )
        outputs = model._build_watch_cross_attention_outputs(
            images=batch_device["images"],
            label_embeds=label_embeds,
            attention=attention,
            features=context_features,
            extra_outputs=extras,
            labels=batch_device["labels"],
            watch_token_ids=batch_device.get("watch_token_ids"),
            watch_token_mask=batch_device.get("watch_token_mask"),
            use_gate=True,
        )
    if hook_handle is not None:
        hook_handle.remove()

    result = {
        "probabilities": torch.sigmoid(outputs["logits"]).float().cpu().numpy()[0],
        "image_probabilities": torch.sigmoid(outputs["image_only_logits"]).float().cpu().numpy()[0],
        "label_embeddings": label_embeds.float().cpu().numpy()[0],
        "attention": outputs["attention"].float().cpu().numpy()[0],
        "context_features": context_features.float().cpu().numpy()[0],
    }
    for key in ("apro_raw_coordinates", "apro_context_coordinates", "apro_transition_eta"):
        if key in outputs:
            result[key] = outputs[key].float().cpu().numpy()[0]
    if "raw_features" in captured:
        result["raw_features"] = captured["raw_features"].float().cpu().numpy()[0]
    return result


def normalized_feature_change(features: np.ndarray) -> np.ndarray:
    normalized = features / np.clip(np.linalg.norm(features, axis=1, keepdims=True), 1e-8, None)
    changes = np.zeros(len(normalized), dtype=float)
    if len(normalized) > 1:
        changes[1:] = 1.0 - np.sum(normalized[1:] * normalized[:-1], axis=1)
    upper = float(np.quantile(changes[1:], 0.95)) if len(changes) > 2 else float(changes.max())
    return np.clip(changes / max(upper, 1e-8), 0.0, 1.25)


def crop_valid_square(image: np.ndarray) -> np.ndarray:
    """去除胃镜图像外围近黑区域，并以有效视野中心裁成1:1。"""
    if image.ndim != 3 or image.shape[0] < 2 or image.shape[1] < 2:
        return image
    brightness = np.max(image.astype(np.float32), axis=2)
    valid = brightness > 18.0
    valid_rows = np.flatnonzero(valid.mean(axis=1) >= 0.15)
    valid_columns = np.flatnonzero(valid.mean(axis=0) >= 0.15)
    if len(valid_rows) < 2 or len(valid_columns) < 2:
        side = min(image.shape[:2])
        y_start = (image.shape[0] - side) // 2
        x_start = (image.shape[1] - side) // 2
        return image[y_start : y_start + side, x_start : x_start + side]

    y_min, y_max = int(valid_rows[0]), int(valid_rows[-1]) + 1
    x_min, x_max = int(valid_columns[0]), int(valid_columns[-1]) + 1
    side = min(y_max - y_min, x_max - x_min)
    center_y = 0.5 * (y_min + y_max)
    center_x = 0.5 * (x_min + x_max)
    y_start = int(round(center_y - 0.5 * side))
    x_start = int(round(center_x - 0.5 * side))
    y_start = min(max(y_start, 0), image.shape[0] - side)
    x_start = min(max(x_start, 0), image.shape[1] - side)
    return image[y_start : y_start + side, x_start : x_start + side]


def _perspective_coefficients(
    output_points: list[tuple[float, float]],
    input_points: list[tuple[float, float]],
) -> tuple[float, ...]:
    """Return Pillow output-to-input perspective coefficients."""
    rows: list[list[float]] = []
    targets: list[float] = []
    for (x_out, y_out), (x_in, y_in) in zip(output_points, input_points):
        rows.append([x_out, y_out, 1.0, 0.0, 0.0, 0.0, -x_in * x_out, -x_in * y_out])
        targets.append(x_in)
        rows.append([0.0, 0.0, 0.0, x_out, y_out, 1.0, -y_in * x_out, -y_in * y_out])
        targets.append(y_in)
    return tuple(np.linalg.solve(np.asarray(rows), np.asarray(targets)).tolist())


def build_horizontal_slice_stack(arrays: list[np.ndarray]) -> np.ndarray:
    """Compose a left-to-right stack of flat, perspective-projected image planes."""
    card_width = 118
    card_height = 360
    layer_step = 22
    canvas_width = card_width + layer_step * max(len(arrays) - 1, 0)
    canvas_height = card_height
    canvas = Image.new("RGBA", (canvas_width, canvas_height), (255, 255, 255, 0))

    # This is the supplied vertical stack geometry rotated by 90 degrees: every
    # image is a tall, narrow parallelogram, and opposite sides remain parallel.
    output_quad = [(112.0, 4.0), (112.0, 312.0), (4.0, 356.0), (4.0, 48.0)]
    source_quad = [(0.0, 0.0), (511.0, 0.0), (511.0, 511.0), (0.0, 511.0)]
    coefficients = _perspective_coefficients(output_quad, source_quad)

    # Later images are drawn first. Earlier images on the left therefore cover
    # the images to their right, while a narrow edge of every layer remains visible.
    for position in range(len(arrays) - 1, -1, -1):
        square = crop_valid_square(arrays[position])
        source = Image.fromarray(np.rot90(square, k=1), mode="RGB").resize(
            (512, 512), Image.Resampling.LANCZOS
        )
        warped = source.convert("RGBA").transform(
            (card_width, card_height),
            Image.Transform.PERSPECTIVE,
            coefficients,
            resample=Image.Resampling.BICUBIC,
            fillcolor=(0, 0, 0, 0),
        )

        polygon_mask = Image.new("L", (card_width, card_height), 0)
        ImageDraw.Draw(polygon_mask).polygon(output_quad, fill=255)
        warped.putalpha(polygon_mask)

        x_offset = position * layer_step
        canvas.alpha_composite(warped, (x_offset, 0))

    return np.asarray(canvas)


def choose_case(
    records: list[dict[str, Any]],
    cache_dataset: MILBagDataset,
    original_model: torch.nn.Module,
    apro_model: torch.nn.Module,
    device: torch.device,
    max_candidates: int = 20,
) -> tuple[dict[str, Any], dict[str, Any]]:
    candidates = sorted(records, key=lambda record: len(record["image_paths"]), reverse=True)[:max_candidates]
    best_score = -float("inf")
    best_record: dict[str, Any] | None = None
    best_meta: dict[str, Any] | None = None
    for record in tqdm(candidates, desc="筛选图1代表性检查", dynamic_ncols=True):
        indices = uniform_indices(len(record["image_paths"]), 64)
        batch, _ = make_batch(record, indices, cache_dataset)
        original = infer(original_model, batch, device)
        apro = infer(apro_model, batch, device)
        labels = np.asarray(record["labels"], dtype=float)
        direction = 2.0 * labels - 1.0
        margin_gain = float(np.sum(direction * (apro["probabilities"] - original["probabilities"])))
        original_correct = int(np.sum((original["probabilities"] >= 0.5) == labels))
        apro_correct = int(np.sum((apro["probabilities"] >= 0.5) == labels))
        correction_gain = apro_correct - original_correct
        score = 3.0 * correction_gain + margin_gain + 0.001 * len(record["image_paths"])
        if score > best_score:
            best_score = score
            best_record = record
            best_meta = {
                "uniform_original_probabilities": original["probabilities"].tolist(),
                "uniform_apro_probabilities": apro["probabilities"].tolist(),
                "selection_score": score,
            }
    if best_record is None or best_meta is None:
        raise RuntimeError("无法选择图1代表性检查")
    return best_record, best_meta


def plot_figure1(
    record: dict[str, Any],
    original: dict[str, np.ndarray],
    apro: dict[str, np.ndarray],
    arrays: list[np.ndarray],
    indices: list[int],
    output_dir: Path,
    metadata: dict[str, Any],
) -> None:
    raw = apro["apro_raw_coordinates"][: len(indices)]
    context = apro["apro_context_coordinates"][: len(indices)]
    slot = np.linspace(0.0, 1.0, len(indices))
    changes = normalized_feature_change(apro["raw_features"][: len(indices)])

    attention = apro["attention"][:, : len(indices)]
    attention_score = attention.max(axis=0)
    candidate_order = np.argsort(-(changes + attention_score / max(attention_score.max(), 1e-8)))
    thumbnail_positions = sorted(candidate_order[:8].tolist(), key=lambda value: raw[value])

    fig = plt.figure(figsize=(15.6, 10.2), constrained_layout=True)
    outer = fig.add_gridspec(4, 2, height_ratios=(0.82, 1.35, 1.15, 1.0))
    thumb_grid = outer[0, :].subgridspec(1, 8, wspace=0.04)
    for column, position in enumerate(thumbnail_positions):
        axis = fig.add_subplot(thumb_grid[0, column])
        axis.imshow(arrays[position])
        axis.set_title(f"$r_t$={raw[position]:.2f}", fontsize=9, pad=3)
        axis.set_xticks([])
        axis.set_yticks([])
        for spine in axis.spines.values():
            spine.set_linewidth(1.5)
            spine.set_color(COLORS["apro_full"] if changes[position] > 0.55 else "#B5B5B5")

    ax_without = fig.add_subplot(outer[1, 0])
    ax_with = fig.add_subplot(outer[1, 1], sharex=ax_without, sharey=ax_without)
    ax_without.plot(raw, raw, color=COLORS["raw"], linewidth=1.8, linestyle="--", label="True acquisition anchor $r_t$")
    ax_without.plot(raw, slot, color=COLORS["original_pe"], linewidth=2.3, marker="o", markersize=3.0, label="Sampled-slot coordinate $q_t$")
    ax_without.fill_between(raw, 0, changes, color=COLORS["change"], alpha=0.14, transform=ax_without.get_xaxis_transform())
    ax_without.set_title("a  Without APro-CoPE: sampled slots replace acquisition positions", loc="left", fontweight="bold")
    ax_without.set_ylabel("Position coordinate")
    ax_without.legend(loc="upper left", frameon=False)

    ax_with.plot(raw, raw, color=COLORS["raw"], linewidth=1.8, linestyle="--", label="Acquisition anchor $r_t$")
    ax_with.plot(raw, context, color=COLORS["apro_full"], linewidth=2.5, marker="o", markersize=3.0, label="Context coordinate $c_t$")
    ax_with.fill_between(raw, 0, changes, color=COLORS["change"], alpha=0.14, transform=ax_with.get_xaxis_transform())
    ax_with.set_title("b  With APro-CoPE: acquisition-anchored contextual deformation", loc="left", fontweight="bold")
    ax_with.legend(loc="upper left", frameon=False)
    for axis in (ax_without, ax_with):
        axis.set_xlim(-0.02, 1.02)
        axis.set_ylim(-0.03, 1.03)
        axis.set_xlabel("Original acquisition position")
        axis.grid(True, color="#E8E8E8", linewidth=0.7)

    ax_error = fig.add_subplot(outer[2, 0])
    ax_error.plot(raw, slot - raw, color=COLORS["original_pe"], linewidth=2.1)
    ax_error.axhline(0.0, color="#555555", linewidth=0.9, linestyle="--")
    ax_error.fill_between(raw, 0, slot - raw, color=COLORS["original_pe"], alpha=0.18)
    ax_error.set_title("c  Slot-coordinate distortion", loc="left", fontweight="bold")
    ax_error.set_ylabel("$q_t-r_t$")
    ax_error.set_xlabel("Original acquisition position")
    ax_error.grid(True, color="#E8E8E8", linewidth=0.7)

    ax_warp = fig.add_subplot(outer[2, 1], sharex=ax_error)
    ax_warp.plot(raw, warp_ratio, color=COLORS["apro_full"], linewidth=2.2, label="Gap deformation $\\Delta c_t/\\Delta r_t$")
    ax_warp.axhline(1.0, color="#555555", linewidth=0.9, linestyle="--")
    ax_warp.fill_between(raw, 1.0, warp_ratio, where=warp_ratio >= 1.0, color=COLORS["apro_full"], alpha=0.18, interpolate=True)
    ax_warp.fill_between(raw, 1.0, warp_ratio, where=warp_ratio < 1.0, color=COLORS["original_pe"], alpha=0.13, interpolate=True)
    ax_warp.set_title("d  Content-adaptive expansion and compression", loc="left", fontweight="bold")
    ax_warp.set_ylabel("Warp ratio")
    ax_warp.set_xlabel("Original acquisition position")
    ax_warp.grid(True, color="#E8E8E8", linewidth=0.7)

    ax_change = fig.add_subplot(outer[3, :])
    ax_change.plot(raw, changes, color=COLORS["change"], linewidth=2.0, label="Adjacent-image feature change")
    ax_change.fill_between(raw, 0.0, changes, color=COLORS["change"], alpha=0.22)
    ax_change.set_ylim(0.0, max(1.05, float(changes.max()) * 1.08))
    ax_change.set_xlabel("Original acquisition position")
    ax_change.set_ylabel("Normalized change")
    ax_change.set_title("e  Visual transition intensity used by the learned deformation", loc="left", fontweight="bold")
    ax_change.grid(True, color="#E8E8E8", linewidth=0.7)
    ax_change.legend(loc="upper right", frameon=False)

    labels = np.asarray(record["labels"], dtype=int)
    fig.suptitle(
        "Mechanism preview: sampled-slot distortion versus APro-CoPE contextual coordinates\n"
        f"{DATASET_NAMES[metadata['dataset']]} test examination · {len(record['image_paths'])} acquired images · labels={labels.tolist()}",
        fontsize=15,
        fontweight="bold",
    )
    save_figure(fig, output_dir / "figure1_position_deformation_preview")

    metadata["figure1"] = {
        "exam_dir": str(record["exam_dir"]),
        "original_image_count": len(record["image_paths"]),
        "sampled_indices": indices,
        "labels": labels.tolist(),
        "sampled_slot_position_mae": float(np.mean(np.abs(slot - raw))),
        "context_deformation_mae": float(np.mean(np.abs(context - raw))),
        "original_probabilities": original["probabilities"].tolist(),
        "apro_probabilities": apro["probabilities"].tolist(),
    }


def plot_figure1_merged(
    record: dict[str, Any],
    apro: dict[str, np.ndarray],
    arrays: list[np.ndarray],
    indices: list[int],
    output_dir: Path,
    metadata: dict[str, Any],
) -> None:
    """将原图1的a/b与c/d分别合并，生成更紧凑的双面板预览。"""
    raw = apro["apro_raw_coordinates"][: len(indices)]
    context = apro["apro_context_coordinates"][: len(indices)]
    slot = np.linspace(0.0, 1.0, len(indices))
    changes = normalized_feature_change(apro["raw_features"][: len(indices)])
    raw_gap = np.diff(raw, prepend=raw[0])
    context_gap = np.diff(context, prepend=context[0])
    warp_ratio = np.ones_like(raw)
    valid = raw_gap > 1e-8
    warp_ratio[valid] = context_gap[valid] / raw_gap[valid]

    attention = apro["attention"][:, : len(indices)]
    attention_score = attention.max(axis=0)
    candidate_order = np.argsort(
        -(changes + attention_score / max(attention_score.max(), 1e-8))
    )
    thumbnail_positions = sorted(candidate_order[:8].tolist(), key=lambda value: raw[value])

    fig = plt.figure(figsize=(13.8, 7.8), constrained_layout=True)
    outer = fig.add_gridspec(3, 1, height_ratios=(0.78, 1.45, 1.35))
    thumb_grid = outer[0].subgridspec(1, 8, wspace=0.04)
    for column, position in enumerate(thumbnail_positions):
        axis = fig.add_subplot(thumb_grid[0, column])
        axis.imshow(arrays[position])
        axis.set_title(f"$r_t$={raw[position]:.2f}", fontsize=9, pad=3)
        axis.set_xticks([])
        axis.set_yticks([])
        for spine in axis.spines.values():
            spine.set_linewidth(1.3)
            spine.set_color(
                COLORS["apro_full"] if changes[position] > 0.55 else "#B5B5B5"
            )

    ax_coordinates = fig.add_subplot(outer[1])
    ax_coordinates.plot(
        raw,
        raw,
        color=COLORS["raw"],
        linewidth=1.8,
        linestyle="--",
        label="Acquisition anchor $r_t$",
    )
    ax_coordinates.plot(
        raw,
        slot,
        color=COLORS["original_pe"],
        linewidth=2.1,
        marker="s",
        markersize=3.2,
        label="Without APro-CoPE: sampled-slot coordinate $q_t$",
    )
    ax_coordinates.plot(
        raw,
        context,
        color=COLORS["apro_full"],
        linewidth=2.3,
        marker="o",
        markersize=3.0,
        label="With APro-CoPE: contextual coordinate $c_t$",
    )
    ax_coordinates.set_xlim(-0.02, 1.02)
    ax_coordinates.set_ylim(-0.03, 1.03)
    ax_coordinates.set_xlabel("Original acquisition position")
    ax_coordinates.set_ylabel("Position coordinate")
    ax_coordinates.set_title(
        "a  Acquisition anchors, sampled-slot positions, and contextual coordinates",
        loc="left",
        fontweight="bold",
    )
    ax_coordinates.grid(True, color="#E8E8E8", linewidth=0.7)
    ax_coordinates.legend(loc="upper left", frameon=False, ncol=3)

    ax_change = fig.add_subplot(outer[2], sharex=ax_coordinates)
    slot_gap = np.diff(slot)
    acquisition_gap = np.diff(raw)
    contextual_gap = np.diff(context)
    valid_gap = acquisition_gap > 1e-8
    gap_positions = raw[1:][valid_gap]
    slot_change_pct = 100.0 * (slot_gap[valid_gap] / acquisition_gap[valid_gap] - 1.0)
    apro_change_pct = 100.0 * (
        contextual_gap[valid_gap] / acquisition_gap[valid_gap] - 1.0
    )
    ax_change.plot(
        gap_positions,
        slot_change_pct,
        color=COLORS["original_pe"],
        linewidth=2.1,
        label="Sampled-slot interval distortion",
    )
    ax_change.fill_between(
        gap_positions,
        0.0,
        slot_change_pct,
        color=COLORS["original_pe"],
        alpha=0.13,
    )
    ax_change.plot(
        gap_positions,
        apro_change_pct,
        color=COLORS["apro_full"],
        linewidth=2.2,
        label="With APro-CoPE: contextual interval change",
    )
    ax_change.fill_between(
        gap_positions,
        0.0,
        apro_change_pct,
        color=COLORS["apro_full"],
        alpha=0.10,
    )
    ax_change.axhline(0.0, color="#444444", linewidth=1.0, linestyle="--")
    ax_change.set_xlabel("Original acquisition position")
    ax_change.set_ylabel("Relative interval change (%)")
    ax_change.grid(True, color="#E8E8E8", linewidth=0.7)
    ax_change.set_title(
        "b  Local interval changes relative to original acquisition spacing",
        loc="left",
        fontweight="bold",
    )
    ax_change.legend(loc="upper left", frameon=False, ncol=2)

    labels = np.asarray(record["labels"], dtype=int)
    fig.suptitle(
        "Compact mechanism preview: position distortion and contextual deformation\n"
        f"{DATASET_NAMES[metadata['dataset']]} test examination · "
        f"{len(record['image_paths'])} acquired images · labels={labels.tolist()}",
        fontsize=14.5,
        fontweight="bold",
    )
    save_figure(fig, output_dir / "figure1_position_deformation_merged_preview")


def pairwise_cosine_consistency(values: np.ndarray) -> float:
    normalized = values / np.clip(np.linalg.norm(values, axis=-1, keepdims=True), 1e-8, None)
    similarities = np.einsum("rld,sld->rsl", normalized, normalized)
    upper = np.triu_indices(values.shape[0], k=1)
    return float(similarities[upper[0], upper[1], :].mean())


def attention_histogram(attention: np.ndarray, raw_positions: np.ndarray, bins: int = 32) -> np.ndarray:
    # attention: [labels, instances]
    result = np.zeros((attention.shape[0], bins), dtype=float)
    bin_indices = np.clip(np.floor(raw_positions * bins).astype(int), 0, bins - 1)
    for label_index in range(attention.shape[0]):
        np.add.at(result[label_index], bin_indices, attention[label_index])
    result /= np.clip(result.sum(axis=1, keepdims=True), 1e-8, None)
    return result


def collect_sampling_stability(
    records: list[dict[str, Any]],
    cache_dataset: MILBagDataset,
    models: dict[str, torch.nn.Module],
    device: torch.device,
    preview_cases: int,
    resamples: int,
    seed: int,
) -> tuple[dict[str, dict[str, list[float]]], dict[str, list[float]], list[str]]:
    eligible = [record for record in records if len(record["image_paths"]) >= 96]
    eligible = sorted(eligible, key=lambda record: len(record["image_paths"]), reverse=True)
    selected = eligible[: min(preview_cases, len(eligible))]
    metrics = {
        variant: {"representation": [], "prediction_sd": [], "attention": []}
        for variant in models
    }
    resample_probabilities = {
        variant: [[] for _ in range(resamples)]
        for variant in models
    }
    all_labels: list[np.ndarray] = []

    for record_index, record in enumerate(tqdm(selected, desc="计算图2采样稳定性", dynamic_ncols=True)):
        all_labels.append(np.asarray(record["labels"], dtype=int))
        index_sets = [
            stratified_jitter_indices(
                len(record["image_paths"]),
                64,
                np.random.default_rng(seed + record_index * 1009 + repeat * 53),
            )
            for repeat in range(resamples)
        ]
        for variant, model in models.items():
            label_embeddings: list[np.ndarray] = []
            probabilities: list[np.ndarray] = []
            attention_maps: list[np.ndarray] = []
            for repeat, indices in enumerate(index_sets):
                batch, _ = make_batch(record, indices, cache_dataset)
                output = infer(model, batch, device)
                label_embeddings.append(output["label_embeddings"])
                probabilities.append(output["probabilities"])
                raw_positions = np.asarray(indices, dtype=float) / max(len(record["image_paths"]) - 1, 1)
                attention_maps.append(attention_histogram(output["attention"], raw_positions))
                resample_probabilities[variant][repeat].append(output["probabilities"])

            label_embeddings_array = np.stack(label_embeddings)
            probabilities_array = np.stack(probabilities)
            attention_array = np.stack(attention_maps)
            metrics[variant]["representation"].append(
                pairwise_cosine_consistency(label_embeddings_array)
            )
            metrics[variant]["prediction_sd"].append(float(probabilities_array.std(axis=0).mean()))
            metrics[variant]["attention"].append(pairwise_cosine_consistency(attention_array))

    labels_array = np.stack(all_labels)
    f1_by_resample: dict[str, list[float]] = {}
    for variant in models:
        f1_values: list[float] = []
        for repeat in range(resamples):
            probs = np.stack(resample_probabilities[variant][repeat])
            predictions = (probs >= 0.5).astype(int)
            label_f1: list[float] = []
            for label_index in range(labels_array.shape[1]):
                true = labels_array[:, label_index]
                pred = predictions[:, label_index]
                tp = int(np.sum((true == 1) & (pred == 1)))
                fp = int(np.sum((true == 0) & (pred == 1)))
                fn = int(np.sum((true == 1) & (pred == 0)))
                denom = 2 * tp + fp + fn
                label_f1.append((2 * tp / denom) if denom else 0.0)
            f1_values.append(float(np.mean(label_f1)))
        f1_by_resample[variant] = f1_values
    return metrics, f1_by_resample, [str(record["exam_dir"]) for record in selected]


def plot_figure2(
    metrics: dict[str, dict[str, list[float]]],
    output_dir: Path,
    dataset: str,
    fold: int,
    num_cases: int,
    resamples: int,
) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(12.8, 4.4), constrained_layout=True)
    variants = ("original_pe", "apro_full")
    labels = ("Sampled-slot PE", "APro-CoPE")
    colors = (COLORS["original_pe"], COLORS["apro_full"])
    panels = (
        ("representation", "Label-representation consistency", "Pairwise cosine similarity", "higher is better"),
        ("prediction_sd", "Prediction variability", "Mean probability SD", "lower is better"),
        ("attention", "Attention-map consistency", "Pairwise cosine similarity", "higher is better"),
    )
    for panel_index, (axis, (metric_key, title, ylabel, direction)) in enumerate(
        zip(axes[:3], panels)
    ):
        values = [metrics[variant][metric_key] for variant in variants]
        box = axis.boxplot(
            values,
            positions=(1, 2),
            widths=0.55,
            patch_artist=True,
            showfliers=False,
            medianprops={"color": "#202020", "linewidth": 1.5},
            whiskerprops={"linewidth": 1.1},
            capprops={"linewidth": 1.1},
        )
        for patch, color in zip(box["boxes"], colors):
            patch.set_facecolor(color)
            patch.set_alpha(0.22)
            patch.set_edgecolor(color)
            patch.set_linewidth(1.5)
        rng = np.random.default_rng(103)
        for position, value_set, color in zip((1, 2), values, colors):
            jitter = rng.normal(0.0, 0.045, len(value_set))
            axis.scatter(np.full(len(value_set), position) + jitter, value_set, s=24, color=color, alpha=0.78, edgecolor="white", linewidth=0.35, zorder=3)
        axis.set_xticks((1, 2), labels, rotation=12)
        axis.set_ylabel(ylabel)
        axis.set_title(
            f"{chr(97 + panel_index)}  {title}\n({direction})",
            loc="left",
            fontweight="bold",
        )
        axis.grid(axis="y", color="#E8E8E8", linewidth=0.7)

    fig.suptitle(
        f"Sampling-stability pilot · {DATASET_NAMES[dataset]} fold-{fold} test examinations "
        f"(n={num_cases}, {resamples} stratified-jittered samples per examination)",
        fontsize=14,
        fontweight="bold",
    )
    save_figure(fig, output_dir / "figure2_sampling_stability_preview")


def plot_figure3_case_study(
    record: dict[str, Any],
    no_pe: dict[str, np.ndarray],
    original: dict[str, np.ndarray],
    apro: dict[str, np.ndarray],
    arrays: list[np.ndarray],
    indices: list[int],
    output_dir: Path,
) -> None:
    """绘制单病例机制链预览：内容变化、坐标变形、注意力与预测。"""
    labels = np.asarray(record["labels"], dtype=int)
    raw = apro["apro_raw_coordinates"][: len(indices)]
    context = apro["apro_context_coordinates"][: len(indices)]
    changes = normalized_feature_change(apro["raw_features"][: len(indices)])
    acquisition_gap = np.diff(raw)
    contextual_gap = np.diff(context)
    sampled_slot = np.linspace(0.0, 1.0, len(indices))
    sampled_slot_gap = np.diff(sampled_slot)
    sampled_slot_change_pct = 100.0 * (
        sampled_slot_gap / np.clip(acquisition_gap, 1e-8, None) - 1.0
    )
    deformation_pct = 100.0 * (
        contextual_gap / np.clip(acquisition_gap, 1e-8, None) - 1.0
    )
    gap_midpoints = 0.5 * (raw[1:] + raw[:-1])

    original_attention = original["attention"][:, : len(indices)]
    apro_attention = apro["attention"][:, : len(indices)]
    original_predictions = original["probabilities"] >= 0.5
    apro_predictions = apro["probabilities"] >= 0.5
    corrected = np.flatnonzero((original_predictions != labels) & (apro_predictions == labels))
    target_label = int(corrected[0]) if len(corrected) else int(np.argmax(np.abs(apro["probabilities"] - original["probabilities"])))

    # 选择覆盖完整流程且包含高内容变化/高目标标签注意力的代表帧。
    attention_scale = apro_attention[target_label] / max(float(apro_attention[target_label].max()), 1e-8)
    difference_scale = np.abs(apro_attention[target_label] - original_attention[target_label])
    difference_scale /= max(float(difference_scale.max()), 1e-8)
    relevance = changes + 0.65 * attention_scale + 0.35 * difference_scale
    selected_positions: list[int] = [0, len(indices) - 1]
    for candidate in np.argsort(-relevance):
        candidate = int(candidate)
        if all(abs(float(raw[candidate] - raw[existing])) >= 0.055 for existing in selected_positions):
            selected_positions.append(candidate)
        if len(selected_positions) >= 10:
            break
    if len(selected_positions) < 10:
        for candidate in np.rint(np.linspace(0, len(indices) - 1, 10)).astype(int):
            if int(candidate) not in selected_positions:
                selected_positions.append(int(candidate))
            if len(selected_positions) >= 10:
                break
    selected_positions = sorted(selected_positions[:10], key=lambda value: raw[value])

    fig = plt.figure(figsize=(14.2, 8.15), constrained_layout=True)
    outer = fig.add_gridspec(3, 1, height_ratios=(1.02, 1.62, 1.0))

    image_grid = outer[0].subgridspec(2, len(selected_positions), height_ratios=(0.16, 1.0), hspace=0.02, wspace=0.035)
    image_header = fig.add_subplot(image_grid[0, :])
    image_header.axis("off")
    image_header.set_title("a  Representative images in acquisition order", loc="left", fontweight="bold", pad=1)
    for column, position in enumerate(selected_positions):
        axis = fig.add_subplot(image_grid[1, column])
        axis.imshow(arrays[position])
        axis.set_title(
            f"$r_t$={raw[position]:.2f}\n$c_t$={context[position]:.2f}",
            fontsize=8.6,
            pad=2.5,
        )
        axis.set_xticks([])
        axis.set_yticks([])
        border_color = COLORS["apro_full"] if position == selected_positions[np.argmax([relevance[p] for p in selected_positions])] else "#B8B8B8"
        for spine in axis.spines.values():
            spine.set_color(border_color)
            spine.set_linewidth(1.6 if border_color == COLORS["apro_full"] else 0.8)
    middle = outer[1].subgridspec(3, 2, width_ratios=(1.72, 1.0), hspace=0.11, wspace=0.24)
    ax_change = fig.add_subplot(middle[0, 0])
    ax_slot_change = fig.add_subplot(middle[1, 0], sharex=ax_change)
    ax_deform = fig.add_subplot(middle[2, 0], sharex=ax_change)
    ax_probability = fig.add_subplot(middle[:, 1])

    ax_change.plot(raw, changes, color=COLORS["change"], linewidth=2.1)
    ax_change.fill_between(raw, 0.0, changes, color=COLORS["change"], alpha=0.18)
    ax_change.set_ylabel("Feature change\n(normalized)")
    ax_change.tick_params(axis="x", labelbottom=False)
    ax_change.grid(True, color="#E8E8E8", linewidth=0.7)
    ax_change.set_title("b  Content change and position-interval comparison", loc="left", fontweight="bold")

    ax_slot_change.axhline(0.0, color="#444444", linestyle="--", linewidth=1.0)
    ax_slot_change.plot(gap_midpoints, sampled_slot_change_pct, color=COLORS["original_pe"], linewidth=1.9)
    ax_slot_change.fill_between(gap_midpoints, 0.0, sampled_slot_change_pct, color=COLORS["original_pe"], alpha=0.13)
    ax_slot_change.set_ylabel("Sampled-slot interval\ndistortion (%)")
    ax_slot_change.tick_params(axis="x", labelbottom=False)
    ax_slot_change.grid(True, color="#E8E8E8", linewidth=0.7)

    ax_deform.axhline(0.0, color="#444444", linestyle="--", linewidth=1.1)
    ax_deform.plot(gap_midpoints, deformation_pct, color=COLORS["apro_full"], linewidth=2.0)
    ax_deform.fill_between(gap_midpoints, 0.0, deformation_pct, color=COLORS["apro_full"], alpha=0.14)
    ax_deform.set_xlabel("Original acquisition position")
    ax_deform.set_ylabel("APro-CoPE interval\nchange (%)")
    ax_deform.grid(True, color="#E8E8E8", linewidth=0.7)

    y_positions = np.arange(len(LABEL_NAMES), dtype=float)
    ax_probability.axvline(0.5, color="#555555", linestyle="--", linewidth=1.2, label="Decision threshold")
    probability_series = (
        (no_pe["probabilities"], y_positions - 0.16, COLORS["no_pe"], "^", "No PE"),
        (original["probabilities"], y_positions, COLORS["original_pe"], "s", "Sampled-slot PE"),
        (apro["probabilities"], y_positions + 0.16, COLORS["apro_full"], "o", "APro-CoPE"),
    )
    ax_probability.axhspan(target_label - 0.32, target_label + 0.32, color=COLORS["change"], alpha=0.08, zorder=0)
    for row in range(len(LABEL_NAMES)):
        row_values = [float(series[0][row]) for series in probability_series]
        ax_probability.hlines(row, min(row_values), max(row_values), color="#C6C6C6", linewidth=1.5, zorder=1)
    for probabilities, offsets, color, marker, name in probability_series:
        ax_probability.scatter(probabilities, offsets, marker=marker, s=55, color=color, label=name, zorder=3)
        value = float(probabilities[target_label])
        y_value = float(offsets[target_label])
        ax_probability.annotate(
            f"{value:.2f}",
            (value, y_value),
            xytext=(0, 9 if name != "APro-CoPE" else -10),
            textcoords="offset points",
            ha="center",
            va="center",
            fontsize=7.4,
            color=color,
        )
    tick_labels = [f"{name}  (GT {int(value)})" for name, value in zip(LABEL_NAMES, labels)]
    ax_probability.set_yticks(y_positions, tick_labels)
    ax_probability.set_xlim(-0.03, 1.03)
    ax_probability.set_ylim(len(LABEL_NAMES) - 0.45, -0.55)
    ax_probability.set_xlabel("Predicted probability")
    ax_probability.set_title("c  Label-level prediction", loc="left", fontweight="bold", pad=24)
    ax_probability.grid(axis="x", color="#E8E8E8", linewidth=0.7)
    ax_probability.legend(
        loc="center left",
        bbox_to_anchor=(1.01, 0.5),
        ncol=1,
        frameon=False,
        fontsize=7.2,
        handletextpad=0.45,
        columnspacing=0.9,
    )

    bottom = outer[2].subgridspec(1, 2, width_ratios=(1.0, 0.025), wspace=0.055)
    attention_axis = fig.add_subplot(bottom[0, 0])
    color_axis = fig.add_subplot(bottom[0, 1])
    relative_attention = []
    no_pe_attention = no_pe["attention"][:, : len(indices)]
    for attention in (no_pe_attention, original_attention, apro_attention):
        relative_attention.append(attention / np.clip(attention.mean(axis=1, keepdims=True), 1e-8, None))
    common_vmax = float(np.quantile(np.concatenate([value.ravel() for value in relative_attention]), 0.98))
    combined_attention = np.concatenate(relative_attention, axis=0)
    image = attention_axis.imshow(
        combined_attention,
        aspect="auto",
        interpolation="nearest",
        cmap="magma",
        vmin=0.0,
        vmax=max(common_vmax, 1.05),
        extent=(float(raw[0]), float(raw[-1]), combined_attention.shape[0] - 0.5, -0.5),
    )
    heatmap_labels = [
        f"No PE · {name}" for name in LABEL_NAMES
    ] + [
        f"Slot PE · {name}" for name in LABEL_NAMES
    ] + [
        f"APro-CoPE · {name}" for name in LABEL_NAMES
    ]
    attention_axis.set_yticks(np.arange(len(heatmap_labels)), heatmap_labels)
    attention_axis.axhline(len(LABEL_NAMES) - 0.5, color="white", linewidth=2.2)
    attention_axis.axhline(2 * len(LABEL_NAMES) - 0.5, color="white", linewidth=2.2)
    attention_axis.set_xlim(float(raw[0]), float(raw[-1]))
    attention_axis.set_xlabel("Original acquisition position")
    attention_axis.set_title("d  Label-wise MIL attention over the original acquisition sequence", loc="left", fontweight="bold")
    colorbar = fig.colorbar(image, cax=color_axis)
    colorbar.set_label("Relative attention")

    fig.suptitle(
        "Representative examination: content-adaptive position deformation, attention, and prediction\n"
        f"{len(record['image_paths'])} acquired images · ground truth={labels.tolist()}",
        fontsize=14.2,
        fontweight="bold",
    )
    save_figure(fig, output_dir / "figure3_case_mechanism_preview")


def plot_figure3b_position_interval_comparison(
    record: dict[str, Any],
    apro: dict[str, np.ndarray],
    indices: list[int],
    output_dir: Path,
) -> None:
    """将图3(b)独立绘制为内容变化与位置间隔变化对照图。"""
    raw = apro["apro_raw_coordinates"][: len(indices)]
    context = apro["apro_context_coordinates"][: len(indices)]
    changes = normalized_feature_change(apro["raw_features"][: len(indices)])

    acquisition_gap = np.diff(raw)
    sampled_slot = np.linspace(0.0, 1.0, len(indices))
    sampled_slot_change_pct = 100.0 * (
        np.diff(sampled_slot) / np.clip(acquisition_gap, 1e-8, None) - 1.0
    )
    apro_change_pct = 100.0 * (
        np.diff(context) / np.clip(acquisition_gap, 1e-8, None) - 1.0
    )
    gap_midpoints = 0.5 * (raw[1:] + raw[:-1])

    # 两个百分比面板采用完全相同的纵轴范围，以便直接比较变形幅度。
    interval_limit = 1.08 * float(
        np.max(np.abs(np.concatenate((sampled_slot_change_pct, apro_change_pct))))
    )
    interval_limit = max(interval_limit, 5.0)

    fig, axes = plt.subplots(
        3,
        1,
        figsize=(10.8, 6.8),
        sharex=True,
        constrained_layout=True,
        gridspec_kw={"height_ratios": (0.92, 1.0, 1.0)},
    )
    ax_change, ax_slot, ax_apro = axes

    ax_change.plot(raw, changes, color=COLORS["change"], linewidth=2.35)
    ax_change.fill_between(raw, 0.0, changes, color=COLORS["change"], alpha=0.20)
    ax_change.set_ylabel("Feature change\n(normalized)")
    ax_change.set_ylim(bottom=0.0)
    ax_change.set_title("(a) Adjacent-image feature change", loc="left", fontweight="bold")

    interval_panels = (
        (
            ax_slot,
            sampled_slot_change_pct,
            COLORS["original_pe"],
            "(b) Sampled-slot interval distortion",
            "Interval distortion (%)",
        ),
        (
            ax_apro,
            apro_change_pct,
            COLORS["apro_full"],
            "(c) APro-CoPE",
            "Interval change (%)",
        ),
    )
    for axis, values, color, title, ylabel in interval_panels:
        axis.axhline(0.0, color="#444444", linestyle="--", linewidth=1.05)
        axis.plot(gap_midpoints, values, color=color, linewidth=2.15)
        axis.fill_between(gap_midpoints, 0.0, values, color=color, alpha=0.16)
        axis.set_ylim(-interval_limit, interval_limit)
        axis.set_ylabel(ylabel)
        axis.set_title(title, loc="left", fontweight="bold")

    for axis in axes:
        axis.grid(True, color="#E7E7E7", linewidth=0.75)
        axis.set_xlim(float(raw[0]), float(raw[-1]))
    ax_apro.set_xlabel("Original acquisition position")
    save_figure(fig, output_dir / "figure3b_position_interval_comparison_preview")


def plot_figure3b_combined_position_interval_comparison(
    apro: dict[str, np.ndarray],
    indices: list[int],
    output_dir: Path,
) -> None:
    """绘制图3(b)的紧凑版本，将两种位置间隔变化叠加比较。"""
    raw = apro["apro_raw_coordinates"][: len(indices)]
    context = apro["apro_context_coordinates"][: len(indices)]
    changes = normalized_feature_change(apro["raw_features"][: len(indices)])

    acquisition_gap = np.diff(raw)
    sampled_slot = np.linspace(0.0, 1.0, len(indices))
    sampled_slot_change_pct = 100.0 * (
        np.diff(sampled_slot) / np.clip(acquisition_gap, 1e-8, None) - 1.0
    )
    apro_change_pct = 100.0 * (
        np.diff(context) / np.clip(acquisition_gap, 1e-8, None) - 1.0
    )
    gap_midpoints = 0.5 * (raw[1:] + raw[:-1])
    interval_limit = 1.08 * float(
        np.max(np.abs(np.concatenate((sampled_slot_change_pct, apro_change_pct))))
    )
    interval_limit = max(interval_limit, 5.0)

    fig, (ax_change, ax_interval) = plt.subplots(
        2,
        1,
        figsize=(10.8, 4.9),
        sharex=True,
        constrained_layout=True,
        gridspec_kw={"height_ratios": (0.86, 1.15)},
    )

    ax_change.plot(raw, changes, color=COLORS["change"], linewidth=2.35)
    ax_change.fill_between(raw, 0.0, changes, color=COLORS["change"], alpha=0.20)
    ax_change.set_ylabel("Feature change\n(normalized)")
    ax_change.set_ylim(bottom=0.0)
    ax_change.set_title("(a) Adjacent-image feature change", loc="left", fontweight="bold")

    ax_interval.axhline(0.0, color="#444444", linestyle="--", linewidth=1.05)
    ax_interval.plot(
        gap_midpoints,
        sampled_slot_change_pct,
        color=COLORS["original_pe"],
        linewidth=2.05,
        linestyle=(0, (5, 2)),
        label="Sampled-slot interval distortion",
    )
    ax_interval.plot(
        gap_midpoints,
        apro_change_pct,
        color=COLORS["apro_full"],
        linewidth=2.20,
        label="APro-CoPE",
    )
    ax_interval.set_ylim(-interval_limit, interval_limit)
    ax_interval.set_ylabel("Interval distortion (%)")
    ax_interval.set_xlabel("Original acquisition position")
    ax_interval.set_title("(b) Position-interval comparison", loc="left", fontweight="bold")
    ax_interval.legend(
        loc="lower right",
        bbox_to_anchor=(1.0, 1.015),
        frameon=False,
        ncol=2,
        borderaxespad=0.0,
    )

    for axis in (ax_change, ax_interval):
        axis.grid(True, color="#E7E7E7", linewidth=0.75)
        axis.set_xlim(float(raw[0]), float(raw[-1]))

    save_figure(fig, output_dir / "figure3b_combined_position_interval_comparison_preview")


def plot_figure3b_combined_with_image_strip(
    apro: dict[str, np.ndarray],
    arrays: list[np.ndarray],
    indices: list[int],
    output_dir: Path,
    output_stem: str = "figure3b_combined_with_images_preview",
    stack_stem: str = "figure3a_horizontal_image_stack_preview",
    image_panel_title: str = "(a) Sampled images in acquisition order",
) -> None:
    """在合并版图3(b)上方增加按采集顺序层叠的64张旋转图像。"""
    raw = apro["apro_raw_coordinates"][: len(indices)]
    context = apro["apro_context_coordinates"][: len(indices)]
    changes = normalized_feature_change(apro["raw_features"][: len(indices)])

    acquisition_gap = np.diff(raw)
    sampled_slot = np.linspace(0.0, 1.0, len(indices))
    sampled_slot_change_pct = 100.0 * (
        np.diff(sampled_slot) / np.clip(acquisition_gap, 1e-8, None) - 1.0
    )
    apro_change_pct = 100.0 * (
        np.diff(context) / np.clip(acquisition_gap, 1e-8, None) - 1.0
    )
    gap_midpoints = 0.5 * (raw[1:] + raw[:-1])
    interval_limit = 1.08 * float(
        np.max(np.abs(np.concatenate((sampled_slot_change_pct, apro_change_pct))))
    )
    interval_limit = max(interval_limit, 5.0)

    fig = plt.figure(figsize=(10.8, 9.2), constrained_layout=True)
    grid = fig.add_gridspec(3, 1, height_ratios=(1.45, 0.82, 1.15))
    ax_images = fig.add_subplot(grid[0])
    ax_change = fig.add_subplot(grid[1])
    ax_interval = fig.add_subplot(grid[2], sharex=ax_change)

    ax_images.set_xlim(0.0, 1.0)
    ax_images.set_ylim(0.0, 1.0)
    ax_images.axis("off")
    ax_images.set_title(
        image_panel_title,
        loc="left",
        fontweight="bold",
        pad=2,
    )

    # Build the stack as one perspective-composited image. Each source image is
    # first cropped to 1:1, projected onto the same shallow plane, and then offset
    # to the right. The left layer is composited last so it covers the right layer.
    image_stack = build_horizontal_slice_stack(arrays)
    stack_rgba = Image.fromarray(image_stack, mode="RGBA")
    stack_background = Image.new("RGBA", stack_rgba.size, (255, 255, 255, 255))
    stack_background.alpha_composite(stack_rgba)
    stack_background.convert("RGB").save(
        output_dir / f"{stack_stem}.png", quality=95
    )
    ax_images.imshow(
        image_stack,
        aspect="equal",
        interpolation="lanczos",
        clip_on=True,
    )
    ax_images.set_xlim(-0.5, image_stack.shape[1] - 0.5)
    ax_images.set_ylim(image_stack.shape[0] - 0.5, -0.5)

    ax_change.plot(raw, changes, color=COLORS["change"], linewidth=2.35)
    ax_change.fill_between(raw, 0.0, changes, color=COLORS["change"], alpha=0.20)
    ax_change.set_ylabel("Feature change\n(normalized)")
    ax_change.set_ylim(bottom=0.0)
    ax_change.set_title("(b) Adjacent-image feature change", loc="left", fontweight="bold")

    ax_interval.axhline(0.0, color="#444444", linestyle="--", linewidth=1.05)
    ax_interval.plot(
        gap_midpoints,
        sampled_slot_change_pct,
        color=COLORS["original_pe"],
        linewidth=2.05,
        linestyle=(0, (5, 2)),
        label="Sampled-slot interval distortion",
    )
    ax_interval.plot(
        gap_midpoints,
        apro_change_pct,
        color=COLORS["apro_full"],
        linewidth=2.20,
        label="APro-CoPE",
    )
    ax_interval.set_ylim(-interval_limit, interval_limit)
    ax_interval.set_ylabel("Interval distortion (%)")
    ax_interval.set_xlabel("Original acquisition position")
    ax_interval.set_title("(c) Position-interval comparison", loc="left", fontweight="bold")
    ax_interval.legend(
        loc="lower right",
        bbox_to_anchor=(1.0, 1.015),
        frameon=False,
        ncol=2,
        borderaxespad=0.0,
    )

    for axis in (ax_change, ax_interval):
        axis.grid(True, color="#E7E7E7", linewidth=0.75)
        axis.set_xlim(float(raw[0]), float(raw[-1]))

    save_figure(fig, output_dir / output_stem)


def plot_figure4(output_dir: Path) -> None:
    budgets = np.asarray([16, 32, 48, 64, 80, 96])
    values = {
        "WLE": {
            "no_pe": ([0.8673, 0.8743, 0.8720, 0.8816, 0.8814, 0.8836], [0.0115, 0.0078, 0.0092, 0.0086, 0.0076, 0.0091]),
            "original_pe": ([0.8819, 0.8840, 0.8872, 0.8979, 0.8864, 0.8930], [0.0132, 0.0141, 0.0148, 0.0140, 0.0106, 0.0117]),
            "apro_full": ([0.8986, 0.9068, 0.9119, 0.9149, 0.9158, 0.9165], [0.0131, 0.0133, 0.0142, 0.0114, 0.0102, 0.0090]),
        },
        "Chromoscopic": {
            "no_pe": ([0.7478, 0.7535, 0.7635, 0.7795, 0.7818, 0.7856], [0.0513, 0.0341, 0.0438, 0.0351, 0.0426, 0.0397]),
            "original_pe": ([0.7789, 0.7978, 0.8249, 0.8592, 0.8665, 0.8706], [0.0683, 0.0794, 0.0690, 0.0761, 0.0508, 0.0658]),
            "apro_full": ([0.8134, 0.8460, 0.8732, 0.8877, 0.8906, 0.8929], [0.0368, 0.0297, 0.0240, 0.0168, 0.0214, 0.0177]),
        },
        "Surgical": {
            "no_pe": ([0.7824, 0.7891, 0.7898, 0.8024, 0.8041, 0.8068], [0.0837, 0.0676, 0.0762, 0.0882, 0.0864, 0.0828]),
            "original_pe": ([0.8012, 0.8125, 0.8221, 0.8243, 0.8312, 0.8345], [0.0919, 0.0679, 0.0865, 0.0418, 0.0369, 0.0410]),
            "apro_full": ([0.8168, 0.8304, 0.8408, 0.8456, 0.8480, 0.8495], [0.0235, 0.0215, 0.0185, 0.0224, 0.0178, 0.0197]),
        },
        "EUS": {
            "no_pe": ([0.8443, 0.8537, 0.8594, 0.8694, 0.8715, 0.8731], [0.0803, 0.0758, 0.0926, 0.0974, 0.1026, 0.1034]),
            "original_pe": ([0.8712, 0.8836, 0.8915, 0.8999, 0.9021, 0.9042], [0.0663, 0.0749, 0.0672, 0.0617, 0.0598, 0.0635]),
            "apro_full": ([0.8879, 0.8979, 0.9030, 0.9064, 0.9089, 0.9103], [0.0635, 0.0564, 0.0434, 0.0474, 0.0452, 0.0442]),
        },
    }
    names = {
        "no_pe": "No PE",
        "original_pe": "Sampled-slot PE",
        "apro_full": "APro-CoPE",
    }
    markers = {"no_pe": "o", "original_pe": "s", "apro_full": "D"}

    fig, axes = plt.subplots(2, 2, figsize=(11.8, 8.4), constrained_layout=True, sharex=True)
    all_mean_values = [mean for dataset_values in values.values() for mean, _ in dataset_values.values()]
    y_min = min(min(series) for series in all_mean_values) - 0.025
    y_max = max(max(series) for series in all_mean_values) + 0.018
    for panel_index, (axis, (dataset_name, dataset_values)) in enumerate(zip(axes.flat, values.items())):
        for variant in ("no_pe", "original_pe", "apro_full"):
            means = np.asarray(dataset_values[variant][0])
            stds = np.asarray(dataset_values[variant][1])
            axis.plot(
                budgets,
                means,
                marker=markers[variant],
                markersize=5.4,
                linewidth=2.25 if variant == "apro_full" else 1.9,
                color=COLORS[variant],
                label=names[variant],
                zorder=3,
            )
            axis.fill_between(
                budgets,
                means - stds,
                means + stds,
                color=COLORS[variant],
                alpha=0.10 if variant == "apro_full" else 0.07,
                linewidth=0,
            )
        axis.set_title(f"{chr(97 + panel_index)}  {dataset_name}", loc="left", fontweight="bold")
        axis.set_xticks(budgets)
        axis.set_ylim(y_min, y_max)
        axis.grid(True, color="#E8E8E8", linewidth=0.7)
        if panel_index // 2 == 1:
            axis.set_xlabel("Number of sampled images")
        if panel_index % 2 == 0:
            axis.set_ylabel("Macro-F1")

    legend_handles = [
        Line2D([0], [0], color=COLORS[key], marker=markers[key], linewidth=2.1, label=names[key])
        for key in ("no_pe", "original_pe", "apro_full")
    ]
    fig.legend(handles=legend_handles, loc="upper center", bbox_to_anchor=(0.5, 1.025), ncol=3, frameon=False)
    fig.suptitle(
        "Long-sequence utilization with and without procedure-aware position modeling",
        fontsize=15,
        fontweight="bold",
        y=1.06,
    )
    save_figure(fig, output_dir / "figure4_long_sequence_utilization_preview")


def save_figure(fig: plt.Figure, stem: Path) -> None:
    stem.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(stem.with_suffix(".png"), dpi=300, bbox_inches="tight")
    fig.savefig(stem.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    args = parse_args()
    output_dir = args.output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    configure_style()
    plot_figure4(output_dir)

    metadata: dict[str, Any] = {
        "purpose": "APro-CoPE机制验证版式与分析流程预览",
        "dataset": args.dataset,
        "fold": int(args.fold),
        "seed": int(args.seed),
        "figure4_source": "论文表3中的五折均值与标准差",
    }
    if args.skip_inference:
        metadata["inference_skipped"] = True
        (output_dir / "apro_mechanism_preview_metadata.json").write_text(
            json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        return

    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("当前环境无法使用CUDA")
    records = load_test_records(args.dataset, args.fold)
    cache_dataset = build_cache_dataset(records)
    original_model = load_model("original_pe", args.dataset, args.fold, device)
    apro_model = load_model("apro_full", args.dataset, args.fold, device)

    representative, selection_meta = choose_case(
        records,
        cache_dataset,
        original_model,
        apro_model,
        device,
    )
    rng = np.random.default_rng(args.seed + 7919)
    nonuniform_indices = stratified_jitter_indices(len(representative["image_paths"]), 64, rng)
    batch, arrays = make_batch(representative, nonuniform_indices, cache_dataset)
    original_output = infer(original_model, batch, device)
    apro_output = infer(apro_model, batch, device, capture_raw_features=True)
    plot_figure1(
        representative,
        original_output,
        apro_output,
        arrays,
        nonuniform_indices,
        output_dir,
        metadata,
    )
    plot_figure1_merged(
        representative,
        apro_output,
        arrays,
        nonuniform_indices,
        output_dir,
        metadata,
    )
    metadata["figure1_selection"] = selection_meta

    models = {"original_pe": original_model, "apro_full": apro_model}
    stability_metrics, f1_by_resample, selected_exams = collect_sampling_stability(
        records,
        cache_dataset,
        models,
        device,
        preview_cases=max(2, int(args.preview_cases)),
        resamples=max(3, int(args.resamples)),
        seed=int(args.seed),
    )
    plot_figure2(
        stability_metrics,
        output_dir,
        args.dataset,
        int(args.fold),
        len(selected_exams),
        max(3, int(args.resamples)),
    )
    metadata["figure2"] = {
        "scope": "单折小规模机制预览；正式分析需扩展至全部折外测试检查",
        "num_examinations": len(selected_exams),
        "resamples_per_examination": max(3, int(args.resamples)),
        "sampling": "64-image stratified jittered sampling with endpoints retained",
        "selected_exam_dirs": selected_exams,
        "metrics": stability_metrics,
        "macro_f1_by_resample": f1_by_resample,
    }
    (output_dir / "apro_mechanism_preview_metadata.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(f"预览图已生成：{output_dir}")


if __name__ == "__main__":
    main()
