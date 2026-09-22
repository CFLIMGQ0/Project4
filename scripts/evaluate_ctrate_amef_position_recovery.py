#!/usr/bin/env python3
"""评估 CT-RATE 上 AMEF-MIL 的 APro-CoPE 位置恢复能力。"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import sys
import time
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np
from tqdm import tqdm


ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "src"
DATA = ROOT / "datasets" / "ct_rate_680"
EXPERIMENT = ROOT / "outputs" / "ct_rate_680" / "experiment"
DEFAULT_CHECKPOINT_ROOT = ROOT / "outputs" / "ct_rate_680" / "all_models_fivefold"
DEFAULT_OUTPUT = ROOT / "outputs" / "ct_rate_680" / "position_recovery"
DELETION_FRACTIONS = (0.0, 0.25, 0.5, 0.75)
NONZERO_SEEDS = (42, 2026, 3407, 7919, 104729)


def save_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
    temporary.replace(path)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024**2), b""):
            digest.update(block)
    return digest.hexdigest()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, default=None)
    parser.add_argument("--checkpoint-root", type=Path, default=DEFAULT_CHECKPOINT_ROOT)
    parser.add_argument("--mode", choices=("random", "block3"), default="random")
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--data-root", "--raw-root", dest="data_root", type=Path, default=None)
    parser.add_argument("--raw-cache-dir", type=Path, default=None)
    parser.add_argument(
        "--allow-legacy-cache",
        action="store_true",
        help="显式复用旧位置恢复特征缓存，并在metadata中记录其协议值",
    )
    parser.add_argument("--cleanup-raw-cache", action="store_true")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--case-indices-file", type=Path, default=None)
    parser.add_argument(
        "--skip-eligibility-check",
        action="store_true",
        help="使用已由本地主机原始CT检查过的病例清单，远端只读缓存时跳过NIfTI检查",
    )
    parser.add_argument("--nonzero-seeds", type=int, nargs="+", default=list(NONZERO_SEEDS))
    parser.add_argument(
        "--allow-short-inputs",
        action="store_true",
        help="允许删除后少于训练配置T的病例，按实际剩余数量输入（最多T张）",
    )
    parser.add_argument("--num-curves", type=int, default=6)
    parser.add_argument("--plot-seed", type=int, default=42)
    parser.add_argument("--skip-plot", action="store_true")
    parser.add_argument("--debug-eta-zero", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--audit-only", action="store_true")
    return parser.parse_args()


def resolve_checkpoint(root: Path, explicit: Path | None) -> tuple[Path, int, list[dict[str, Any]]]:
    if explicit is not None:
        checkpoint = explicit.resolve()
        if not checkpoint.is_file():
            raise FileNotFoundError(f"checkpoint不存在：{checkpoint}")
        try:
            fold = int(checkpoint.parents[1].name.removeprefix("fold_"))
        except ValueError as error:
            raise ValueError(f"无法从checkpoint路径识别fold：{checkpoint}") from error
        return checkpoint, fold, []

    candidates: list[dict[str, Any]] = []
    for fold in range(1, 6):
        folder = root / f"fold_{fold}" / "amef_multimodal"
        checkpoint = folder / "best_model.pt"
        metrics_path = folder / "test_metrics.json"
        if not checkpoint.is_file() or not metrics_path.is_file():
            raise FileNotFoundError(f"五折候选不完整：{folder}")
        metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
        value = float(metrics["best_val_loss"])
        if not math.isfinite(value):
            raise ValueError(f"验证损失非有限：{metrics_path}")
        candidates.append({
            "fold": fold,
            "checkpoint": str(checkpoint.resolve()),
            "best_val_loss": value,
            "best_epoch": int(metrics["best_epoch"]),
            "test_macro_f1_not_used": float(metrics["macro_f1"]),
        })
    selected = min(candidates, key=lambda item: (item["best_val_loss"], item["fold"]))
    return Path(selected["checkpoint"]), int(selected["fold"]), candidates


def load_rows_and_test_split(split_path: Path) -> tuple[list[dict[str, Any]], list[int]]:
    rows = json.loads((EXPERIMENT / "samples.json").read_text(encoding="utf-8"))
    split_ids = json.loads(split_path.read_text(encoding="utf-8"))
    test_ids = [int(value) for value in split_ids["test"]]
    by_case = {int(row["case_index"]): row for row in rows}
    if sorted(by_case) != list(range(len(rows))):
        raise ValueError("samples.json 的 case_index 不是连续编号")
    selected_rows = [by_case[index] for index in test_ids]
    if len(selected_rows) != len(set(test_ids)):
        raise ValueError("测试集包含重复病例")
    return rows, test_ids


def target_delete_count(total: int, fraction: float) -> int:
    if total < 2:
        return 0
    requested = int(math.floor(float(fraction) * total + 0.5))
    return min(max(0, requested), total - 2)


def build_sampling(
    total: int,
    fraction: float,
    seed: int,
    target_instances: int,
    allow_short_inputs: bool = False,
) -> dict[str, Any]:
    delete_count = target_delete_count(total, fraction)
    if fraction == 0.0:
        deleted = np.empty(0, dtype=np.int64)
    else:
        rng = np.random.default_rng(seed)
        internal = np.arange(1, total - 1, dtype=np.int64)
        deleted = np.sort(rng.choice(internal, size=delete_count, replace=False))
    keep_mask = np.ones(total, dtype=bool)
    keep_mask[deleted] = False
    remaining = np.flatnonzero(keep_mask).astype(np.int64)
    if len(remaining) < target_instances and not allow_short_inputs:
        raise ValueError("删除后剩余切片数量不足T")
    input_instances = min(target_instances, len(remaining)) if allow_short_inputs else target_instances
    if input_instances < 2:
        raise ValueError("删除后剩余切片数量不足两个端点")
    if len(remaining) == input_instances:
        selected = remaining.copy()
        selected_slots = np.arange(input_instances, dtype=np.int64)
    else:
        selected_slots = np.rint(np.linspace(0, len(remaining) - 1, input_instances)).astype(np.int64)
        if len(np.unique(selected_slots)) != input_instances:
            raise ValueError("均匀采样产生重复槽位")
        selected = remaining[selected_slots]
    if selected[0] != 0 or selected[-1] != total - 1:
        raise ValueError("均匀采样未保留完整序列首尾")
    return {
        "requested_delete_fraction": float(fraction),
        "requested_delete_count": int(math.floor(float(fraction) * total + 0.5)),
        "actual_deleted_count": int(len(deleted)),
        "actual_delete_fraction": float(len(deleted) / total),
        "seed": int(seed),
        "deleted_raw_indices": deleted.tolist(),
        "deleted_mask": (~keep_mask).tolist(),
        "deleted_segment_ids": [-1] * total,
        "deleted_segments": [],
        "remaining_raw_count": int(len(remaining)),
        "input_instances": int(input_instances),
        "selected_raw_indices": selected.tolist(),
        "selected_remaining_slots": selected_slots.tolist(),
        "sampling_mode": "random_sparse",
    }


def block3_delete_count(total: int, fraction: float) -> int:
    if total < 2:
        return 0
    requested = int(math.floor(float(total) * float(fraction)))
    return min(max(0, requested), total - 2)


def build_block3_sampling(
    total: int,
    fraction: float,
    seed: int,
    target_instances: int,
    allow_short_inputs: bool = False,
) -> dict[str, Any]:
    delete_count = block3_delete_count(total, fraction)
    deleted_mask = np.zeros(total, dtype=bool)
    segment_ids = np.full(total, -1, dtype=np.int64)
    segments: list[dict[str, int]] = []
    rng = np.random.default_rng(seed)
    if delete_count:
        if delete_count < 3:
            raise ValueError("block3非零删除数量不足以构造三个正长度区段")
        remaining_count = total - delete_count
        if remaining_count < 4:
            raise ValueError("block3删除后剩余切片不足以构造四个正长度保留区段")
        base, remainder = divmod(delete_count, 3)
        delete_lengths = [base + int(index < remainder) for index in range(3)]
        rng.shuffle(delete_lengths)
        split_points = np.sort(rng.choice(np.arange(1, remaining_count), size=3, replace=False))
        keep_lengths = np.diff(np.concatenate(([0], split_points, [remaining_count]))).astype(int).tolist()
        cursor = 0
        for segment_index, delete_length in enumerate(delete_lengths):
            cursor += keep_lengths[segment_index]
            start = cursor
            end = start + int(delete_length) - 1
            deleted_mask[start:end + 1] = True
            segment_ids[start:end + 1] = segment_index
            segments.append({
                "segment_index": int(segment_index),
                "start_raw_index": int(start),
                "end_raw_index_inclusive": int(end),
                "length": int(delete_length),
            })
            cursor = end + 1
        cursor += keep_lengths[3]
        if cursor != total:
            raise ValueError("block3区段构造未覆盖完整序列")

    remaining = np.flatnonzero(~deleted_mask).astype(np.int64)
    if len(remaining) < target_instances and not allow_short_inputs:
        raise ValueError("block3删除后剩余切片数量不足T")
    input_instances = min(target_instances, len(remaining)) if allow_short_inputs else target_instances
    selected_slots = np.rint(np.linspace(0, len(remaining) - 1, input_instances)).astype(np.int64)
    if len(np.unique(selected_slots)) != input_instances:
        raise ValueError("block3均匀采样产生重复槽位")
    selected = remaining[selected_slots]
    if selected[0] != 0 or selected[-1] != total - 1:
        raise ValueError("block3均匀采样未保留完整序列首尾")
    if int(deleted_mask.sum()) != delete_count:
        raise ValueError("block3删除数量不等于floor(N*fraction)")
    for left, right in zip(segments, segments[1:]):
        if right["start_raw_index"] <= left["end_raw_index_inclusive"] + 1:
            raise ValueError("block3删除区段相邻或重叠")
    return {
        "requested_delete_fraction": float(fraction),
        "requested_delete_count": int(delete_count),
        "actual_deleted_count": int(deleted_mask.sum()),
        "actual_delete_fraction": float(deleted_mask.mean()),
        "seed": int(seed),
        "deleted_raw_indices": np.flatnonzero(deleted_mask).astype(np.int64).tolist(),
        "deleted_mask": deleted_mask.tolist(),
        "deleted_segment_ids": segment_ids.tolist(),
        "deleted_segments": segments,
        "remaining_raw_count": int(len(remaining)),
        "input_instances": int(input_instances),
        "selected_raw_indices": selected.tolist(),
        "selected_remaining_slots": selected_slots.tolist(),
        "sampling_mode": "block3",
    }


def validate_block3_sampling(sampling: dict[str, Any], total: int, target_instances: int) -> None:
    mask = np.asarray(sampling["deleted_mask"], dtype=bool)
    selected = np.asarray(sampling["selected_raw_indices"], dtype=np.int64)
    if mask.shape != (total,) or int(mask.sum()) != int(sampling["requested_delete_count"]):
        raise ValueError("block3完整删除mask与删除数量不一致")
    input_instances = int(sampling["input_instances"])
    if mask[0] or mask[-1] or len(selected) != input_instances:
        raise ValueError("block3端点或最终T数量校验失败")
    if not np.array_equal(selected, np.unique(selected)) or not np.all(np.diff(selected) > 0):
        raise ValueError("block3最终切片存在重复或乱序")
    segments = sampling["deleted_segments"]
    if sampling["requested_delete_count"] and len(segments) != 3:
        raise ValueError("block3非零删除未生成恰好三个区段")
    for segment in segments:
        start = int(segment["start_raw_index"])
        end = int(segment["end_raw_index_inclusive"])
        if start < 1 or end >= total - 1 or end - start + 1 != int(segment["length"]):
            raise ValueError("block3区段边界或长度异常")
        if not mask[start:end + 1].all() or mask[max(0, start - 1)] or mask[min(total - 1, end + 1)]:
            raise ValueError("block3区段不是连续且不相邻的删除段")


def spacing_ratio(values: np.ndarray) -> float:
    gaps = np.diff(np.asarray(values, dtype=np.float64))
    if len(gaps) == 0 or not np.isfinite(gaps).all() or np.any(gaps <= 0):
        return float("nan")
    return float(gaps.max() / gaps.min())


def crossing_gap_values(
    selected: np.ndarray,
    true_positions: np.ndarray,
    baseline_prediction: np.ndarray,
    model_prediction: np.ndarray,
    deleted_mask: np.ndarray,
    segment_ids: np.ndarray,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    true_gaps = np.diff(true_positions)
    baseline_gaps = np.diff(baseline_prediction)
    model_gaps = np.diff(model_prediction)
    interval_rows: list[dict[str, Any]] = []
    crossing = []
    for interval_index, (left, right) in enumerate(zip(selected[:-1], selected[1:])):
        deleted_between = deleted_mask[int(left) + 1:int(right)]
        ids = sorted({int(value) for value in segment_ids[int(left) + 1:int(right)] if int(value) >= 0})
        is_crossing = bool(deleted_between.any())
        crossing.append(is_crossing)
        interval_rows.append({
            "interval_index": int(interval_index),
            "left_selected_position": int(interval_index),
            "right_selected_position": int(interval_index + 1),
            "left_raw_index": int(left),
            "right_raw_index": int(right),
            "true_gap": float(true_gaps[interval_index]),
            "baseline_gap": float(baseline_gaps[interval_index]),
            "model_gap": float(model_gaps[interval_index]),
            "crossing_gap": is_crossing,
            "crossing_segment_indices": json.dumps(ids, ensure_ascii=False),
        })
    crossing_mask = np.asarray(crossing, dtype=bool)
    values = {
        "crossing_gap_count": int(crossing_mask.sum()),
        "model_CrossingGapMAE": float(np.mean(np.abs(model_gaps[crossing_mask] - true_gaps[crossing_mask])))
        if crossing_mask.any() and np.isfinite(model_gaps[crossing_mask]).all() else float("nan"),
        "baseline_CrossingGapMAE": float(np.mean(np.abs(baseline_gaps[crossing_mask] - true_gaps[crossing_mask])))
        if crossing_mask.any() else float("nan"),
    }
    return values, interval_rows


def oriented_volume(row: dict[str, Any]) -> tuple[np.ndarray, np.ndarray, np.ndarray, bool]:
    import nibabel as nib

    image_path = DATA / row["file_path"]
    if not image_path.is_file():
        raise FileNotFoundError(f"NIfTI不存在：{image_path}")
    image = nib.load(str(image_path))
    if len(image.shape) != 3 or not np.isfinite(image.affine).all():
        raise ValueError(f"NIfTI维度或空间矩阵异常：{row['patient_id']}")
    data = np.asarray(image.dataobj, dtype=np.float32)
    orientation = nib.orientations.ornt_transform(
        nib.orientations.io_orientation(image.affine),
        nib.orientations.axcodes2ornt(("R", "A", "S")),
    )
    oriented = nib.orientations.apply_orientation(data, orientation)
    transformed_affine = image.affine @ nib.orientations.inv_ornt_aff(orientation, image.shape)
    axis_vector = transformed_affine[:3, 2]
    axis_norm = float(np.linalg.norm(axis_vector))
    if axis_norm <= 0 or not math.isfinite(axis_norm):
        raise ValueError(f"切片轴空间向量异常：{row['patient_id']}")
    axis_direction = axis_vector / axis_norm
    indices = np.arange(oriented.shape[2], dtype=np.float64)
    homogeneous = np.zeros((4, len(indices)), dtype=np.float64)
    homogeneous[2] = indices
    homogeneous[3] = 1.0
    world = (transformed_affine @ homogeneous)[:3].T
    positions = (world - world[0]) @ axis_direction
    positions = positions.astype(np.float64)
    if not np.isfinite(positions).all() or len(positions) < 2:
        raise ValueError(f"真实切片位置异常：{row['patient_id']}")
    span = float(positions[-1] - positions[0])
    if abs(span) <= 1e-12:
        raise ValueError(f"真实切片位置跨度为零：{row['patient_id']}")
    normalized = (positions - positions[0]) / span
    differences = np.diff(positions)
    equidistant = bool(np.allclose(differences, differences[0], rtol=1e-5, atol=1e-5))
    slices = np.ascontiguousarray(oriented[::-1, ::-1, :].transpose(2, 1, 0))
    del data, oriented, world, homogeneous
    if not np.isfinite(slices).all():
        raise ValueError(f"影像强度存在非有限数：{row['patient_id']}")
    return slices, normalized, np.arange(len(normalized), dtype=np.int64), equidistant


class FrozenConvNeXt:
    def __init__(self, device: Any) -> None:
        import torch
        from torchvision import models

        self.torch = torch
        self.device = device
        torch.hub.set_dir(str(ROOT / "pre_weights"))
        backbone = models.convnext_tiny(weights=models.ConvNeXt_Tiny_Weights.IMAGENET1K_V1)
        self.encoder = torch.nn.Sequential(
            backbone.features,
            backbone.avgpool,
            backbone.classifier[0],
            torch.nn.Flatten(1),
        ).to(device).eval()
        del backbone
        self.mean = torch.tensor([.485, .456, .406], device=device)[None, :, None, None]
        self.std = torch.tensor([.229, .224, .225], device=device)[None, :, None, None]

    def encode(self, slices: np.ndarray) -> np.ndarray:
        torch = self.torch
        values = []
        with torch.inference_mode():
            for start in range(0, len(slices), 16):
                batch = torch.from_numpy(slices[start:start + 16]).to(self.device)[:, None]
                channels = torch.cat(
                    [((batch - level + width / 2) / width).clamp(0, 1)
                     for level, width in ((-600.0, 1500.0), (40.0, 400.0), (-600.0, 700.0))],
                    dim=1,
                )
                channels = torch.nn.functional.interpolate(
                    channels, size=(224, 224), mode="bilinear", align_corners=False, antialias=True
                )
                with torch.autocast("cuda", enabled=self.device.type == "cuda"):
                    feature = self.encoder((channels - self.mean) / self.std)
                values.append(feature.float().cpu().numpy())
        result = np.concatenate(values, axis=0).astype(np.float32, copy=False)
        if result.shape != (len(slices), 768) or not np.isfinite(result).all():
            raise ValueError("ConvNeXt特征形状或数值异常")
        return result


def load_or_extract_case(
    row: dict[str, Any],
    extractor: FrozenConvNeXt,
    cache_dir: Path,
    cache_protocol: str,
    accepted_cache_protocols: set[str] | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, bool]:
    cache_dir.mkdir(parents=True, exist_ok=True)
    path = cache_dir / f"{int(row['case_index']):04d}.npz"
    if path.is_file():
        with np.load(path, allow_pickle=False) as cached:
            cached_protocol = str(cached["cache_protocol"])
            if cached_protocol != cache_protocol and cached_protocol not in (accepted_cache_protocols or set()):
                raise ValueError(f"原始特征缓存协议不一致：{path}")
            features = cached["features"].astype(np.float32, copy=False)
            positions = cached["true_positions"].astype(np.float64, copy=False)
            indices = cached["slice_indices"].astype(np.int64, copy=False)
            equidistant = bool(int(cached["equidistant"]))
            if features.shape != (len(indices), 768) or not np.isfinite(features).all():
                raise ValueError(f"原始特征缓存异常：{path}")
            if len(indices) != len(positions) or not np.array_equal(indices, np.arange(len(indices))):
                raise ValueError(f"原始位置缓存异常：{path}")
            return features, positions, indices, equidistant

    slices, positions, indices, equidistant = oriented_volume(row)
    features = extractor.encode(slices)
    temporary = path.with_suffix(".tmp.npz")
    np.savez_compressed(
        temporary,
        features=features,
        true_positions=positions,
        slice_indices=indices,
        equidistant=np.asarray(int(equidistant), dtype=np.int8),
        cache_protocol=cache_protocol,
        patient_id=row["patient_id"],
    )
    temporary.replace(path)
    return features, positions, indices, equidistant


def normalize_context(context: np.ndarray) -> tuple[np.ndarray, str, bool]:
    if context.ndim != 1 or len(context) < 2:
        return np.full(len(context), np.nan), "invalid_context_shape", False
    if not np.isfinite(context).all():
        return np.full(len(context), np.nan), "nonfinite_context", False
    monotonic = bool(np.all(np.diff(context) >= -1e-7))
    span = float(context[-1] - context[0])
    if not math.isfinite(span) or abs(span) <= 1e-12:
        return np.full(len(context), np.nan), "invalid_context_span", monotonic
    prediction = (context - context[0]) / span
    if not np.isfinite(prediction).all():
        return np.full(len(context), np.nan), "nonfinite_prediction", monotonic
    return prediction, "ok" if monotonic else "non_monotonic", monotonic


def metric_values(prediction: np.ndarray, truth: np.ndarray) -> dict[str, float]:
    if not np.isfinite(prediction).all():
        return {"PRE": float("nan"), "GRE": float("nan"), "Acc@0.02": float("nan"), "Acc@0.05": float("nan")}
    errors = np.abs(prediction - truth)
    internal = errors[1:-1]
    true_gaps = np.diff(truth)
    predicted_gaps = np.diff(prediction)
    return {
        "PRE": float(internal.mean()),
        "GRE": float(0.5 * np.abs(predicted_gaps - true_gaps).sum()),
        "Acc@0.02": float((internal <= 0.02).mean()),
        "Acc@0.05": float((internal <= 0.05).mean()),
    }


def build_model(checkpoint_path: Path, device: Any, debug_eta_zero: bool = False) -> Any:
    import torch

    if str(SRC) not in sys.path:
        sys.path.insert(0, str(SRC))
    scripts_root = str(SRC / "scripts")
    if scripts_root not in sys.path:
        sys.path.insert(0, scripts_root)
    import run_cq500_amef_multimodal as amef_base

    try:
        checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    except TypeError:
        checkpoint = torch.load(checkpoint_path, map_location="cpu")
    if checkpoint.get("model_key") != "amef_multimodal":
        raise ValueError(f"checkpoint不是AMEF-MIL多模态模型：{checkpoint.get('model_key')}")
    parameters = dict(checkpoint["model_parameters"])
    model, _ = amef_base.build_model("amef_multimodal", parameters)
    model.load_state_dict(checkpoint["state_dict"], strict=True)
    model.to(device).eval()
    torch.backends.mha.set_fastpath_enabled(False)
    position_variant = getattr(model, "position_variant", None)
    if position_variant not in {"apro_full", "apro_absolute_only", "apro_relative_only"}:
        raise ValueError(
            "checkpoint的位置模块不是可恢复位置的APro-CoPE变体："
            f"{position_variant}"
        )
    if getattr(model, "apro_positioner", None) is None:
        raise ValueError("checkpoint未初始化APro-CoPE位置模块")
    if debug_eta_zero:
        positioner = model.apro_positioner
        if not hasattr(positioner, "transition_mode"):
            raise ValueError("APro-CoPE实现缺少transition_mode，无法执行eta=0自检")
        positioner.transition_mode = "none"
    model.eval()
    return model, checkpoint, parameters


def position_forward(
    model: Any, features: np.ndarray, target_instances: int, device: Any
) -> tuple[np.ndarray, np.ndarray, str, bool]:
    import torch

    image_tensor = torch.from_numpy(features)[None, :, :, None, None].to(device=device, dtype=torch.float32)
    mask = torch.ones((1, target_instances), dtype=torch.bool, device=device)
    uniform_slots = torch.arange(target_instances, dtype=torch.long, device=device)[None]
    uniform_count = torch.tensor([target_instances], dtype=torch.long, device=device)
    expected = torch.arange(target_instances, dtype=torch.float32, device=device)
    expected = expected / float(target_instances - 1)
    with torch.inference_mode():
        instance_features, _ = model.encode_instances(image_tensor, mask)
        _, _, diagnostics = model._encode_position(
            instance_features, mask, uniform_slots, uniform_count
        )
    raw_coordinates = diagnostics.get("apro_raw_coordinates")
    context_coordinates = diagnostics.get("apro_context_coordinates")
    if raw_coordinates is None or context_coordinates is None:
        raise RuntimeError("APro-CoPE未返回原始/上下文标量坐标")
    if not torch.allclose(raw_coordinates[0], expected, atol=1e-6, rtol=1e-6):
        raise RuntimeError("位置模块未按要求接收等距槽位，可能存在原始位置泄漏")
    context = context_coordinates[0].detach().float().cpu().numpy()
    prediction, status, monotonic = normalize_context(context)
    return context, prediction, status, monotonic


def write_csv(path: Path, rows: list[dict[str, Any]], fields: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def finite_std(values: list[float]) -> float:
    if len(values) < 2:
        return float("nan")
    return float(np.std(np.asarray(values, dtype=np.float64), ddof=1))


def format_table_metric(mean: Any, std: Any) -> str:
    mean_value = float(mean)
    std_value = float(std)
    if not math.isfinite(mean_value):
        return "—"
    if not math.isfinite(std_value):
        return f"{mean_value:.4f}"
    return f"{mean_value:.4f} ± {std_value:.4f}"


def write_paper_table(summary_rows: list[dict[str, Any]], output_dir: Path) -> None:
    metric_columns = (
        ("PRE", "PRE ↓"),
        ("GRE", "GRE ↓"),
        ("Acc@0.02", "Acc@0.02 ↑"),
        ("Acc@0.05", "Acc@0.05 ↑"),
    )
    table_rows: list[dict[str, Any]] = []
    for summary in summary_rows:
        fraction = float(summary["deletion_fraction"])
        if fraction == 0.0:
            continue
        deletion_label = f"{int(round(fraction * 100))}%"
        for method, prefix in (("等距位置基线", "baseline"), ("ACPE", "model")):
            row = {"删除比例": deletion_label, "方法": method}
            for metric, label in metric_columns:
                row[label] = format_table_metric(
                    summary[f"{prefix}_{metric}_mean"],
                    summary[f"{prefix}_{metric}_std"],
                )
            table_rows.append(row)
    fields = ["删除比例", "方法"] + [label for _, label in metric_columns]
    write_csv(output_dir / "paper_table.csv", table_rows, fields)
    lines = [
        "| 删除比例 | 方法 | PRE ↓ | GRE ↓ | Acc@0.02 ↑ | Acc@0.05 ↑ |",
        "| --- | --- | ---: | ---: | ---: | ---: |",
    ]
    for row in table_rows:
        lines.append(
            f"| {row['删除比例']} | {row['方法']} | {row['PRE ↓']} | {row['GRE ↓']} | "
            f"{row['Acc@0.02 ↑']} | {row['Acc@0.05 ↑']} |"
        )
    lines.extend([
        "",
        "注：表中为均值 ± 标准差；先对每个 CT 的随机重复取均值，再对测试 CT 等权汇总，标准差为 CT 级均值的样本标准差（ddof=1）。",
        "ACPE 为论文命名；代码中复用的真实位置模块名称为 APro-CoPE。",
    ])
    (output_dir / "paper_table.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def aggregate_results(repeat_rows: list[dict[str, Any]], output_dir: Path) -> None:
    metrics = ("PRE", "GRE", "Acc@0.02", "Acc@0.05", "CrossingGapMAE")
    by_condition_case: dict[tuple[float, int], list[dict[str, Any]]] = defaultdict(list)
    for row in repeat_rows:
        by_condition_case[(float(row["deletion_fraction"]), int(row["case_index"]))].append(row)
    ct_rows: list[dict[str, Any]] = []
    for (fraction, case_index), rows in sorted(by_condition_case.items()):
        ct_row: dict[str, Any] = {
            "case_index": case_index,
            "patient_id": rows[0]["patient_id"],
            "deletion_fraction": fraction,
            "actual_delete_fraction_mean": float(np.mean([float(r["actual_delete_fraction"]) for r in rows])),
            "repeat_count": len(rows),
            "crossing_gap_count_mean": float(np.mean([float(r["crossing_gap_count"]) for r in rows])),
            "model_invalid_repeats": sum(r["model_status"] not in {"ok", "non_monotonic"} for r in rows),
            "model_non_monotonic_repeats": sum(r["model_status"] == "non_monotonic" for r in rows),
        }
        for metric in metrics:
            model_values = [float(r[f"model_{metric}"]) for r in rows]
            baseline_values = [float(r[f"baseline_{metric}"]) for r in rows]
            ct_row[f"model_{metric}"] = float(np.mean(model_values)) if all(math.isfinite(v) for v in model_values) else float("nan")
            ct_row[f"baseline_{metric}"] = float(np.mean(baseline_values)) if all(math.isfinite(v) for v in baseline_values) else float("nan")
        ct_rows.append(ct_row)

    summary_rows: list[dict[str, Any]] = []
    for fraction in DELETION_FRACTIONS:
        rows = [row for row in ct_rows if float(row["deletion_fraction"]) == fraction]
        summary: dict[str, Any] = {
            "deletion_fraction": fraction,
            "test_ct_total": len(rows),
            "model_invalid_repeat_count": sum(int(row["model_invalid_repeats"]) for row in rows),
            "model_non_monotonic_repeat_count": sum(int(row["model_non_monotonic_repeats"]) for row in rows),
            "std_definition": "sample SD (ddof=1) over per-CT means; each CT is first averaged over its random repeats",
        }
        for metric in metrics:
            for name in ("model", "baseline"):
                values = [float(row[f"{name}_{metric}"]) for row in rows if math.isfinite(float(row[f"{name}_{metric}"]))]
                summary[f"{name}_{metric}_mean"] = float(np.mean(values)) if values else float("nan")
                summary[f"{name}_{metric}_std"] = finite_std(values)
                summary[f"{name}_{metric}_n_ct"] = len(values)
        summary_rows.append(summary)

    ct_fields = list(ct_rows[0]) if ct_rows else ["case_index", "patient_id", "deletion_fraction"]
    summary_fields = list(summary_rows[0]) if summary_rows else ["deletion_fraction"]
    write_csv(output_dir / "ct_summary.csv", ct_rows, ct_fields)
    write_csv(output_dir / "summary.csv", summary_rows, summary_fields)
    write_paper_table(summary_rows, output_dir)


def plot_curves(
    output_dir: Path,
    slice_rows: list[dict[str, Any]],
    curve_cases: list[int],
) -> None:
    if not curve_cases:
        return
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    curve_dir = output_dir / "position_curves"
    curve_dir.mkdir(parents=True, exist_ok=True)
    for case_index in curve_cases:
        selected = [
            row for row in slice_rows
            if int(row["case_index"]) == case_index and int(row["seed"]) == 0
        ]
        if not selected:
            selected = [
                row for row in slice_rows
                if int(row["case_index"]) == case_index and int(row["seed"]) == NONZERO_SEEDS[0]
            ]
        if not selected:
            continue
        fractions = sorted({float(row["deletion_fraction"]) for row in selected})
        figure, axes = plt.subplots(2, 2, figsize=(11, 8), sharex=True, sharey=True)
        for axis, fraction in zip(axes.flat, fractions):
            rows = sorted(
                [row for row in selected if float(row["deletion_fraction"]) == fraction],
                key=lambda row: int(row["selected_position"]),
            )
            x = [int(row["selected_position"]) for row in rows]
            axis.plot(x, [float(row["true_position"]) for row in rows], label="truth", linewidth=2)
            axis.plot(x, [float(row["baseline_position"]) for row in rows], label="uniform", linestyle="--")
            model_values = [float(row["model_position"]) for row in rows]
            if all(math.isfinite(value) for value in model_values):
                axis.plot(x, model_values, label="AMEF/APro-CoPE")
            axis.set_title(f"delete={fraction:g}")
            axis.grid(alpha=0.25)
        axes[0, 0].legend()
        figure.supxlabel("input slice order")
        figure.supylabel("normalized position")
        figure.suptitle(f"CT-RATE case {case_index}")
        figure.tight_layout()
        figure.savefig(curve_dir / f"case_{case_index:04d}.png", dpi=160)
        plt.close(figure)


def plot_block3_position_distribution(
    output_dir: Path,
    slice_rows: list[dict[str, Any]],
    repeat_rows: list[dict[str, Any]],
    manifest_rows: list[dict[str, Any]],
    case_indices: list[int],
    selection_seed: int,
) -> int | None:
    if not case_indices:
        return None
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    rng = np.random.default_rng(selection_seed)
    case_index = int(rng.choice(np.asarray(sorted(case_indices), dtype=np.int64)))
    available_seeds = sorted({int(row["seed"]) for row in repeat_rows if int(row["case_index"]) == case_index and int(row["seed"]) != 0})
    if not available_seeds:
        return case_index
    chosen_seed = available_seeds[0]
    fractions = (0.25, 0.5, 0.75)
    figure, axes = plt.subplots(1, 3, figsize=(15, 4.8), sharex=True, sharey=True)
    if not isinstance(axes, np.ndarray):
        axes = np.asarray([axes])
    colors = {"truth": "#2166ac", "baseline": "#666666", "model": "#b2182b"}
    y_positions = {"truth": 2.0, "baseline": 1.0, "model": 0.0}
    label_names = {"truth": "Ground truth", "baseline": "Original PE", "model": "ACPE"}
    for axis, fraction in zip(axes, fractions):
        rows = sorted(
            [
                row for row in slice_rows
                if int(row["case_index"]) == case_index
                and int(row["seed"]) == chosen_seed
                and float(row["deletion_fraction"]) == fraction
            ],
            key=lambda row: int(row["selected_position"]),
        )
        repeat = next(
            row for row in repeat_rows
            if int(row["case_index"]) == case_index
            and int(row["seed"]) == chosen_seed
            and float(row["deletion_fraction"]) == fraction
        )
        manifest = next(
            row for row in manifest_rows
            if int(row["case_index"]) == case_index
            and int(row["seed"]) == chosen_seed
            and float(row["requested_delete_fraction"]) == fraction
        )
        total = len(manifest["deleted_mask"])
        for segment in manifest.get("deleted_segments", []):
            start = float(segment["start_raw_index"]) / float(total - 1)
            end = float(segment["end_raw_index_inclusive"]) / float(total - 1)
            axis.axvspan(start, end, color="#fddbc7", alpha=0.55, linewidth=0)
            axis.axvline(start, color="#d6604d", linewidth=0.7, alpha=0.8)
            axis.axvline(end, color="#d6604d", linewidth=0.7, alpha=0.8)
        x_values = {
            "truth": np.asarray([float(row["true_position"]) for row in rows]),
            "baseline": np.asarray([float(row["baseline_position"]) for row in rows]),
            "model": np.asarray([float(row["model_position"]) for row in rows]),
        }
        for index in range(0, len(rows), max(1, len(rows) // 12)):
            axis.plot(
                [x_values["truth"][index], x_values["baseline"][index], x_values["model"][index]],
                [y_positions["truth"], y_positions["baseline"], y_positions["model"]],
                color="#999999",
                linewidth=0.45,
                alpha=0.45,
                zorder=1,
            )
        for name in ("truth", "baseline", "model"):
            axis.scatter(
                x_values[name],
                np.full(len(rows), y_positions[name]),
                s=13,
                color=colors[name],
                label=label_names[name],
                alpha=0.85,
                zorder=2,
            )
        axis.set_title(
            f"{int(round(fraction * 100))}%\n"
            f"PRE: {float(repeat['baseline_PRE']):.4f} / {float(repeat['model_PRE']):.4f}\n"
            f"GRE: {float(repeat['baseline_GRE']):.4f} / {float(repeat['model_GRE']):.4f}"
        )
        axis.set_xlim(0.0, 1.0)
        axis.set_ylim(-0.6, 2.6)
        axis.set_yticks([2.0, 1.0, 0.0], ["Ground truth", "Original PE", "ACPE"])
        axis.set_xlabel("normalized position")
        axis.grid(axis="x", alpha=0.2)
    axes[0].legend(loc="upper left", fontsize=8)
    figure.suptitle(f"CT-RATE block3 position distribution: case {case_index}, seed {chosen_seed}")
    figure.tight_layout()
    figure.savefig(output_dir / f"block3_position_distribution_case_{case_index:04d}.png", dpi=200)
    figure.savefig(output_dir / f"block3_position_distribution_case_{case_index:04d}.pdf")
    plt.close(figure)
    save_json(
        output_dir / "block3_plot_selection.json",
        {"selection_seed": int(selection_seed), "case_index": case_index, "deletion_seed": chosen_seed},
    )
    return case_index


def main() -> None:
    args = parse_args()
    global DATA
    if args.data_root is not None:
        DATA = args.data_root.resolve()
    checkpoint_path, fold, candidates = resolve_checkpoint(args.checkpoint_root, args.checkpoint)
    config_path = checkpoint_path.with_name("config.json")
    config = json.loads(config_path.read_text(encoding="utf-8")) if config_path.is_file() else {}
    split_path = checkpoint_path.with_name("split_ids.json")
    rows, test_ids = load_rows_and_test_split(split_path)
    selected_case_indices = None
    if args.case_indices_file is not None:
        selection = json.loads(args.case_indices_file.read_text(encoding="utf-8"))
        selected_case_indices = [int(value) for value in selection["case_indices"]]
        if len(selected_case_indices) != len(set(selected_case_indices)):
            raise ValueError("case-indices-file包含重复病例")
        test_set = set(test_ids)
        unknown = sorted(set(selected_case_indices) - test_set)
        if unknown:
            raise ValueError(f"筛选病例不在固定测试集：{unknown}")
        test_ids = [case for case in test_ids if case in set(selected_case_indices)]
    rows_by_case = {int(row["case_index"]): row for row in rows}
    target_instances = int(config.get("settings", {}).get("max_instances", 0))
    if target_instances < 2:
        raise ValueError("无法从训练配置读取有效T：settings.max_instances")
    if args.limit is not None and args.limit <= 0:
        raise ValueError("--limit必须为正数")

    if args.audit_only:
        print(json.dumps({
            "mode": args.mode,
            "checkpoint": str(checkpoint_path),
            "selected_fold": fold,
            "test_cases": len(test_ids),
            "T": target_instances,
            "candidates": candidates,
        }, ensure_ascii=False, indent=2))
        return

    default_output = DEFAULT_OUTPUT if args.mode == "random" else DEFAULT_OUTPUT.parent / "position_recovery_block3"
    output_dir = (args.output_dir or default_output).resolve()
    if output_dir.exists() and any(output_dir.iterdir()) and not args.overwrite:
        raise FileExistsError(f"输出目录非空，若确认重跑请加--overwrite：{output_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)
    import torch

    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError(f"请求的CUDA设备不可用：{device}")
    if str(SRC) not in sys.path:
        sys.path.insert(0, str(SRC))
    protocol_path = EXPERIMENT / "preparation_protocol.json"
    preparation_protocol = json.loads(protocol_path.read_text(encoding="utf-8"))
    cache_protocol = hashlib.sha256(
        (preparation_protocol["preparation_sha256"] + sha256(Path(__file__))).encode("utf-8")
    ).hexdigest()
    raw_cache_dir = (args.raw_cache_dir or output_dir / "raw_feature_cache").resolve()
    accepted_cache_protocols: set[str] = set()
    legacy_cache_protocols: list[str] = []
    if args.allow_legacy_cache:
        legacy_metadata_path = ROOT / "outputs" / "ct_rate_680" / "position_recovery" / "metadata.json"
        if not legacy_metadata_path.is_file():
            raise FileNotFoundError(f"旧缓存元数据不存在：{legacy_metadata_path}")
        legacy_metadata = json.loads(legacy_metadata_path.read_text(encoding="utf-8"))
        legacy_protocol = str(legacy_metadata.get("raw_feature_cache_protocol", ""))
        if not legacy_protocol:
            raise ValueError("旧缓存元数据缺少raw_feature_cache_protocol")
        accepted_cache_protocols.add(legacy_protocol)
        legacy_cache_protocols.append(legacy_protocol)
    manifest_path = output_dir / "sampling_manifest.jsonl"
    if manifest_path.exists():
        manifest_path.unlink()
    model, checkpoint, parameters = build_model(
        checkpoint_path,
        device,
        debug_eta_zero=args.debug_eta_zero,
    )
    extractor = FrozenConvNeXt(device)

    eligible: list[int] = []
    excluded: list[dict[str, Any]] = []
    if args.skip_eligibility_check:
        if args.case_indices_file is None:
            raise ValueError("--skip-eligibility-check必须配合--case-indices-file")
        eligible = list(test_ids)
    elif args.mode == "block3" and not args.allow_short_inputs:
        import nibabel as nib

        for case_index in test_ids:
            row = rows_by_case[case_index]
            total = int(nib.load(str(DATA / row["file_path"])).shape[2])
            remaining = total - block3_delete_count(total, 0.75)
            if remaining < target_instances:
                excluded.append({"case_index": case_index, "patient_id": row["patient_id"], "original_count": total, "remaining_at_75": remaining})
            else:
                eligible.append(case_index)
    elif args.allow_short_inputs:
        eligible = list(test_ids)
    else:
        for case_index in test_ids:
            row = rows_by_case[case_index]
            image_path = DATA / row["file_path"]
            import nibabel as nib

            shape = nib.load(str(image_path)).shape
            total = int(shape[2])
            remaining = total - target_delete_count(total, 0.75)
            if remaining < target_instances:
                excluded.append({"case_index": case_index, "patient_id": row["patient_id"], "original_count": total, "remaining_at_75": remaining})
            else:
                eligible.append(case_index)
    if args.limit is not None:
        eligible = eligible[:args.limit]

    diagnostic_alpha = float(parameters.get("apro_warp_alpha", getattr(model.apro_positioner, "warp_alpha", float("nan"))))
    diagnostic_bound = float(math.exp(2.0 * diagnostic_alpha))
    metadata = {
        "script": str(Path(__file__).resolve()),
        "script_sha256": sha256(Path(__file__).resolve()),
        "mode": args.mode,
        "checkpoint": str(checkpoint_path.resolve()),
        "checkpoint_sha256": sha256(checkpoint_path),
        "checkpoint_selection": "minimum best_val_loss across five AMEF-MIL folds; test macro-F1 is recorded but not used",
        "selected_fold": fold,
        "fold_candidates": candidates,
        "checkpoint_payload": {key: value for key, value in checkpoint.items() if key != "state_dict"},
        "config_path": str(config_path.resolve()),
        "config": config,
        "dataset": "CT-RATE existing experiment test split only",
        "data_root": str(DATA),
        "split_path": str(split_path.resolve()),
        "test_case_count_before_exclusion": len(test_ids),
        "selected_case_indices": selected_case_indices,
        "selection_manifest": str(args.case_indices_file.resolve()) if args.case_indices_file else None,
        "eligible_case_count": len(eligible),
        "excluded_case_count": len(excluded),
        "excluded_cases": excluded,
        "T": target_instances,
        "allow_short_inputs": bool(args.allow_short_inputs),
        "skip_eligibility_check": bool(args.skip_eligibility_check),
        "raw_cache_dir": str(raw_cache_dir),
        "cleanup_raw_cache": bool(args.cleanup_raw_cache),
        "allow_legacy_cache": bool(args.allow_legacy_cache),
        "accepted_cache_protocols": legacy_cache_protocols,
        "deletion_fractions": list(DELETION_FRACTIONS),
        "nonzero_random_seeds": list(args.nonzero_seeds),
        "zero_deletion_seed": 0,
        "sampling": (
            "three random non-adjacent contiguous deletion segments; delete first, then uniformly sample remaining sequence to "
            + ("min(T, remaining_count) when needed" if args.allow_short_inputs else "T")
            if args.mode == "block3"
            else (
                "random deletion from internal raw sequence only; endpoints retained; remaining sequence uniformly sampled "
                "without replacement to T"
                if not args.allow_short_inputs
                else "random deletion from internal raw sequence only; endpoints retained; remaining sequence uniformly sampled "
                "without replacement to min(T, remaining_count)"
            )
        ),
        "deletion_count_rule": (
            "block3 uses floor(N*fraction), split as three near-equal positive lengths; four positive retained segments "
            "make deletion segments non-overlapping and non-adjacent"
            if args.mode == "block3"
            else "round-half-up fraction*N, capped at N-2 to retain endpoints; actual count and fraction are saved per repeat"
        ),
        "input_position": (
            "model receives instance_indices=0..K-1 and original_image_counts=K, yielding u_t=t/(K-1), "
            "where K=min(T, remaining_count) for allow-short-inputs; raw indices and physical positions never enter model inputs"
            if args.allow_short_inputs
            else "model receives instance_indices=0..T-1 and original_image_counts=T only, yielding u_t=t/(T-1); raw indices and physical positions never enter model inputs"
        ),
        "position_module": "existing exp_8.models.AProCoPE via apro_context_coordinates; no model structure or default behavior modified",
        "coordinate_source": "RAS-oriented NIfTI affine projection along oriented slice axis; normalized endpoints are 0 and 1",
        "equidistance_check": "reported per CT; NIfTI affine coordinates are used even when the check is false",
        "feature_preprocessing": preparation_protocol,
        "raw_feature_cache_protocol": cache_protocol,
        "device": str(device),
        "metric_std": "sample SD (ddof=1) over per-CT means; random repeats are averaged within each CT first",
        "range_diagnostic": {
            "alpha": diagnostic_alpha,
            "exp_2alpha": diagnostic_bound,
            "interpretation": "diagnostic only; no alpha change and no case exclusion",
        },
        "debug_eta_zero": bool(args.debug_eta_zero),
        "classification_evaluation": False,
        "no_grad_eval": True,
        "started_unix": time.time(),
    }
    save_json(output_dir / "metadata.json", metadata)

    slice_rows: list[dict[str, Any]] = []
    interval_rows: list[dict[str, Any]] = []
    repeat_rows: list[dict[str, Any]] = []
    manifest_rows: list[dict[str, Any]] = []
    abnormal_counts: dict[str, int] = defaultdict(int)
    curve_cases = eligible[:max(0, args.num_curves)]
    for case_index in tqdm(eligible, desc="评估CT-RATE位置恢复"):
        row = rows_by_case[case_index]
        features, true_positions, raw_indices, equidistant = load_or_extract_case(
            row, extractor, raw_cache_dir, cache_protocol, accepted_cache_protocols
        )
        total = len(raw_indices)
        if not np.array_equal(raw_indices, np.arange(total)):
            raise ValueError(f"原始索引不连续：{row['patient_id']}")
        true_full_spacing_ratio = spacing_ratio(true_positions)
        alpha = float(parameters.get("apro_warp_alpha", getattr(model.apro_positioner, "warp_alpha", float("nan"))))
        spacing_ratio_bound = float(math.exp(2.0 * alpha))
        for fraction in DELETION_FRACTIONS:
            seeds = (0,) if fraction == 0.0 else tuple(args.nonzero_seeds)
            for seed in seeds:
                if args.mode == "block3":
                    sampling = build_block3_sampling(
                        total,
                        fraction,
                        seed,
                        target_instances,
                        allow_short_inputs=args.allow_short_inputs,
                    )
                    validate_block3_sampling(sampling, total, target_instances)
                else:
                    sampling = build_sampling(
                        total,
                        fraction,
                        seed,
                        target_instances,
                        allow_short_inputs=args.allow_short_inputs,
                    )
                selected = np.asarray(sampling["selected_raw_indices"], dtype=np.int64)
                selected_features = features[selected]
                truth = true_positions[selected]
                input_instances = int(sampling["input_instances"])
                baseline = np.linspace(0.0, 1.0, input_instances, dtype=np.float64)
                model_context, model_prediction, model_status, monotonic = position_forward(
                    model, selected_features, input_instances, device
                )
                model_metrics = metric_values(model_prediction, truth)
                baseline_metrics = metric_values(baseline, truth)
                deleted_mask = np.asarray(sampling["deleted_mask"], dtype=bool) if "deleted_mask" in sampling else np.zeros(total, dtype=bool)
                segment_ids = np.asarray(sampling.get("deleted_segment_ids", [-1] * total), dtype=np.int64)
                crossing_values, current_interval_rows = crossing_gap_values(
                    selected,
                    truth,
                    baseline,
                    model_prediction,
                    deleted_mask,
                    segment_ids,
                )
                model_selected_spacing_ratio = spacing_ratio(model_prediction)
                baseline_selected_spacing_ratio = spacing_ratio(baseline)
                abnormal_counts[model_status] += 1
                repeat = {
                    "case_index": case_index,
                    "patient_id": row["patient_id"],
                    "deletion_fraction": float(fraction),
                    "actual_delete_fraction": sampling["actual_delete_fraction"],
                    "input_instances": input_instances,
                    "seed": seed,
                    "equidistant_coordinate": equidistant,
                    "model_status": model_status,
                    "model_monotonic": monotonic,
                    "model_invalid": model_status not in {"ok", "non_monotonic"},
                    "crossing_gap_count": crossing_values["crossing_gap_count"],
                    "true_full_spacing_ratio": true_full_spacing_ratio,
                    "spacing_ratio_bound": spacing_ratio_bound,
                    "true_spacing_ratio_exceeds_bound": bool(
                        math.isfinite(true_full_spacing_ratio) and true_full_spacing_ratio > spacing_ratio_bound
                    ),
                    "model_selected_spacing_ratio": model_selected_spacing_ratio,
                    "baseline_selected_spacing_ratio": baseline_selected_spacing_ratio,
                    "model_spacing_ratio_exceeds_bound": bool(
                        math.isfinite(model_selected_spacing_ratio) and model_selected_spacing_ratio > spacing_ratio_bound
                    ),
                }
                for metric in model_metrics:
                    repeat[f"model_{metric}"] = model_metrics[metric]
                    repeat[f"baseline_{metric}"] = baseline_metrics[metric]
                repeat["model_CrossingGapMAE"] = crossing_values["model_CrossingGapMAE"]
                repeat["baseline_CrossingGapMAE"] = crossing_values["baseline_CrossingGapMAE"]
                if args.debug_eta_zero:
                    if not np.allclose(model_prediction, baseline, atol=2e-5, rtol=2e-5):
                        raise ValueError("eta=0自检失败：ACPE上下文坐标未回到等距槽位")
                    for metric in model_metrics:
                        if not math.isclose(model_metrics[metric], baseline_metrics[metric], abs_tol=2e-5, rel_tol=2e-5):
                            raise ValueError(f"eta=0自检失败：{metric}未与Original PE一致")
                repeat_rows.append(repeat)
                for interval_row in current_interval_rows:
                    interval_rows.append({
                        "case_index": case_index,
                        "patient_id": row["patient_id"],
                        "deletion_fraction": float(fraction),
                        "actual_delete_fraction": sampling["actual_delete_fraction"],
                        "seed": seed,
                        **interval_row,
                        "model_status": model_status,
                    })
                manifest_rows.append({**sampling, "case_index": case_index, "patient_id": row["patient_id"], "equidistant_coordinate": equidistant})
                with manifest_path.open("a", encoding="utf-8") as manifest:
                    manifest.write(json.dumps(manifest_rows[-1], ensure_ascii=False) + "\n")
                for position, (raw_index, truth_value, baseline_value, model_value, context_value) in enumerate(
                    zip(selected, truth, baseline, model_prediction, model_context)
                ):
                    slice_rows.append({
                        "case_index": case_index,
                        "patient_id": row["patient_id"],
                        "deletion_fraction": float(fraction),
                        "actual_delete_fraction": sampling["actual_delete_fraction"],
                        "input_instances": input_instances,
                        "seed": seed,
                        "selected_position": position,
                        "raw_index": int(raw_index),
                        "true_position": float(truth_value),
                        "baseline_position": float(baseline_value),
                        "model_context_coordinate": float(context_value),
                        "model_position": float(model_value),
                        "model_status": model_status,
                        "model_monotonic": monotonic,
                        "equidistant_coordinate": equidistant,
                    })

    slice_fields = list(slice_rows[0]) if slice_rows else ["case_index", "patient_id"]
    repeat_fields = list(repeat_rows[0]) if repeat_rows else ["case_index", "patient_id"]
    write_csv(output_dir / "per_slice_results.csv", slice_rows, slice_fields)
    interval_fields = list(interval_rows[0]) if interval_rows else ["case_index", "patient_id"]
    write_csv(output_dir / "interval_results.csv", interval_rows, interval_fields)
    write_csv(output_dir / "repeat_metrics.csv", repeat_rows, repeat_fields)
    aggregate_results(repeat_rows, output_dir)
    plot_case_index = None
    if args.mode == "block3" and not args.skip_plot:
        plot_case_index = plot_block3_position_distribution(
            output_dir,
            slice_rows,
            repeat_rows,
            manifest_rows,
            eligible,
            args.plot_seed,
        )
    else:
        plot_curves(output_dir, slice_rows, curve_cases)
    if args.cleanup_raw_cache:
        for cache_file in raw_cache_dir.glob("*.npz"):
            cache_file.unlink()
        for temporary_file in raw_cache_dir.glob("*.tmp.npz"):
            temporary_file.unlink()
        try:
            raw_cache_dir.rmdir()
        except OSError:
            pass
    metadata.update({
        "finished_unix": time.time(),
        "processed_case_count": len(eligible),
        "repeat_count": len(repeat_rows),
        "slice_result_count": len(slice_rows),
        "interval_result_count": len(interval_rows),
        "plot_case_index": plot_case_index,
        "skip_plot": bool(args.skip_plot),
        "abnormal_model_status_counts": dict(abnormal_counts),
    })
    save_json(output_dir / "metadata.json", metadata)
    print(f"位置恢复评估完成：{len(eligible)}例，输出目录：{output_dir}")


if __name__ == "__main__":
    main()
