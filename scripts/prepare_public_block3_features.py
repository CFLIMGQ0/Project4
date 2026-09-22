#!/usr/bin/env python3
"""为 AMOS-MM/MR-RATE-1K 按完整原序列准备 block3 最终切片特征。"""
from __future__ import annotations

import argparse
import json
import math
import sys
import tempfile
import zipfile
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "src/scripts"))

FRACTIONS = (0.0, 0.25, 0.5, 0.75)
SEED = 42
TARGET = 64
DATASET_INFO = {
    "amos_mm": {
        "data": ROOT / "datasets/amos_mm",
        "experiment": ROOT / "outputs/amos_mm/experiment",
        "output": ROOT / "outputs/amos_mm/block3_full_features",
    },
    "mr_rate": {
        "data": ROOT / "datasets/mr_rate_1k",
        "experiment": ROOT / "outputs/mr_rate_1k/experiment",
        "output": ROOT / "outputs/mr_rate_1k/block3_full_features",
    },
}


def block3_sampling(total: int, fraction: float, case_index: int) -> dict:
    delete_count = min(max(int(math.floor(total * fraction)), 0), max(0, total - 2))
    deleted_mask = np.zeros(total, dtype=bool)
    segment_ids = np.full(total, -1, dtype=np.int64)
    segments = []
    if delete_count:
        if delete_count < 3 or total - delete_count < 4:
            raise ValueError(f"无法构造block3：N={total}, D={delete_count}")
        rng = np.random.default_rng(np.random.SeedSequence([SEED, case_index, int(fraction * 100), 3]))
        base, remainder = divmod(delete_count, 3)
        lengths = [base + int(i < remainder) for i in range(3)]
        rng.shuffle(lengths)
        cuts = np.sort(rng.choice(np.arange(1, total - delete_count), size=3, replace=False))
        keeps = np.diff(np.concatenate(([0], cuts, [total - delete_count]))).astype(int)
        cursor = 0
        for segment_index, length in enumerate(lengths):
            cursor += int(keeps[segment_index])
            start = cursor
            end = start + int(length) - 1
            deleted_mask[start:end + 1] = True
            segment_ids[start:end + 1] = segment_index
            segments.append({
                "segment_index": int(segment_index),
                "start_raw_index": int(start),
                "end_raw_index_inclusive": int(end),
                "length": int(length),
            })
            cursor = end + 1
        if cursor + int(keeps[3]) != total:
            raise ValueError("block3区段未覆盖完整序列")
    remaining = np.flatnonzero(~deleted_mask).astype(np.int64)
    if len(remaining) < TARGET:
        raise ValueError(f"删除后不足T={TARGET}: N={total}, fraction={fraction}")
    slots = np.rint(np.linspace(0, len(remaining) - 1, TARGET)).astype(np.int64)
    if len(np.unique(slots)) != TARGET:
        raise ValueError("block3均匀采样产生重复槽位")
    selected = remaining[slots]
    if selected[0] != 0 or selected[-1] != total - 1:
        raise ValueError("block3未保留完整序列首尾")
    if int(deleted_mask.sum()) != delete_count:
        raise ValueError("block3删除数量不精确")
    for left, right in zip(segments, segments[1:]):
        if right["start_raw_index"] <= left["end_raw_index_inclusive"] + 1:
            raise ValueError("block3区段相邻或重叠")
    return {
        "requested_delete_fraction": float(fraction),
        "requested_delete_count": int(delete_count),
        "actual_deleted_count": int(deleted_mask.sum()),
        "actual_delete_fraction": float(deleted_mask.mean()),
        "seed": SEED,
        "deleted_raw_indices": np.flatnonzero(deleted_mask).astype(np.int64).tolist(),
        "deleted_mask": deleted_mask.tolist(),
        "deleted_segment_ids": segment_ids.tolist(),
        "deleted_segments": segments,
        "remaining_raw_count": int(len(remaining)),
        "input_instances": TARGET,
        "selected_raw_indices": selected.tolist(),
        "selected_remaining_slots": slots.tolist(),
        "sampling_mode": "block3",
    }


def selected_union(total: int, case_index: int) -> tuple[dict[str, dict], np.ndarray]:
    samplings = {}
    union = set()
    for fraction in FRACTIONS:
        item = block3_sampling(total, fraction, case_index)
        samplings[str(fraction)] = item
        union.update(item["selected_raw_indices"])
    return samplings, np.asarray(sorted(union), dtype=np.int64)


class FrozenEncoder:
    def __init__(self, device: str):
        import torch
        from torchvision import models

        self.torch = torch
        import torch.nn.functional as F
        self.F = F
        torch.set_num_threads(2)
        torch.hub.set_dir(str(ROOT / "pre_weights"))
        base = models.convnext_tiny(weights=models.ConvNeXt_Tiny_Weights.IMAGENET1K_V1)
        self.model = torch.nn.Sequential(
            base.features, base.avgpool, base.classifier[0], torch.nn.Flatten(1)
        ).eval().to(device)
        del base
        self.device = torch.device(device)
        self.mean = torch.tensor([.485, .456, .406], device=self.device)[None, :, None, None]
        self.std = torch.tensor([.229, .224, .225], device=self.device)[None, :, None, None]

    def encode(self, images: list[np.ndarray], mode: str) -> np.ndarray:
        chunks = []
        with self.torch.inference_mode():
            for start in range(0, len(images), 16):
                batch = np.stack(images[start:start + 16]).astype(np.float32)
                values = self.torch.from_numpy(batch).to(self.device)[:, None]
                if mode == "amos":
                    windows = ((40.0, 400.0), (60.0, 150.0), (40.0, 800.0))
                    channels = self.torch.cat(
                        [((values - (level - width / 2)) / width).clamp(0, 1)
                         for level, width in windows], dim=1
                    )
                else:
                    channels = values.repeat(1, 3, 1, 1)
                channels = self.F.interpolate(
                    channels, (224, 224), mode="bilinear", align_corners=False, antialias=True
                )
                with self.torch.autocast("cuda"):
                    feature = self.model((channels - self.mean) / self.std)
                chunks.append(feature.float().cpu().numpy())
        result = np.concatenate(chunks, axis=0)
        if result.shape != (len(images), 768) or not np.isfinite(result).all():
            raise ValueError("ConvNeXt特征形状或数值异常")
        return result


def prepare_amos(row: dict, source_indices: np.ndarray, encoder: FrozenEncoder):
    import nibabel as nib

    image = nib.load(str(DATASET_INFO["amos_mm"]["data"] / row["file_path"]))
    data = np.asarray(image.dataobj, dtype=np.float32)
    orient = nib.orientations.ornt_transform(
        nib.orientations.io_orientation(image.affine),
        nib.orientations.axcodes2ornt(("R", "A", "S")),
    )
    data = nib.orientations.apply_orientation(data, orient)
    total = int(data.shape[2])
    if source_indices[-1] >= total:
        raise ValueError(f"AMOS原始索引越界: {row['scan_id']}")
    slices = np.ascontiguousarray(
        data[::-1, ::-1, source_indices].transpose(2, 1, 0)
    )
    features = encoder.encode([item for item in slices], "amos")
    return total, features


def prepare_mr(row: dict, source_indices: np.ndarray, encoder: FrozenEncoder):
    import nibabel as nib
    from prepare_mrrate_1k_experiment import plane_axis, robust_percentiles, slice_2d
    import torch
    import torch.nn.functional as F

    zip_path = DATASET_INFO["mr_rate"]["data"] / row["zip_path"]
    if not zip_path.is_file():
        raise FileNotFoundError(zip_path)
    wanted = set(int(x) for x in source_indices.tolist())
    images: dict[int, np.ndarray] = {}
    with tempfile.TemporaryDirectory(prefix=f"block3_{row['study_uid']}_") as temp:
        temp = Path(temp)
        with zipfile.ZipFile(zip_path) as archive:
            members = [name for name in archive.namelist() if "/img/" in name and name.endswith(".nii.gz")]
            prefix = row["study_uid"] + "_"
            by_series = {
                Path(name).name[len(prefix):-7]: name
                for name in members
                if Path(name).name.startswith(prefix)
            }
            ordered = []
            for meta in row["series"]:
                member = by_series.get(meta["series_id"])
                if member is not None:
                    archive.extract(member, temp)
                    ordered.append((meta, temp / member))
        descriptors = []
        total = 0
        for series_index, (meta, path) in enumerate(ordered):
            image = nib.as_closest_canonical(nib.load(str(path)))
            if len(image.shape) != 3:
                continue
            axis = plane_axis(meta["acquisition_plane"], image.header.get_zooms()[:3])
            count = int(image.shape[axis])
            descriptors.append({"meta": meta, "path": path, "axis": axis,
                                "count": count, "offset": total, "series_index": series_index})
            total += count
        if not descriptors or total < 64:
            raise ValueError(f"MR原始序列不足T={TARGET}: {row['study_uid']}, N={total}")
        for descriptor in descriptors:
            start = descriptor["offset"]
            end = start + descriptor["count"]
            local = [int(index - start) for index in source_indices if start <= index < end]
            if not local:
                continue
            image = nib.as_closest_canonical(nib.load(str(descriptor["path"])))
            data = np.asarray(image.dataobj, dtype=np.float32)
            lo, hi = robust_percentiles(data)
            for local_index in local:
                image_2d = slice_2d(data, descriptor["axis"], local_index)
                image_2d = np.nan_to_num(image_2d, nan=lo, posinf=hi, neginf=lo)
                image_2d = np.clip((image_2d - lo) / (hi - lo), 0.0, 1.0).astype(np.float32)
                image_2d = F.interpolate(
                    torch.from_numpy(image_2d)[None, None], (224, 224),
                    mode="bilinear", align_corners=False, antialias=True,
                ).squeeze().numpy()
                images[start + local_index] = image_2d
            del data
    if set(images) != wanted:
        missing = sorted(wanted.difference(images))[:10]
        raise RuntimeError(f"MR存在未提取的原始切片: {row['study_uid']} {missing}")
    features = encoder.encode([images[int(index)] for index in source_indices], "mr")
    return total, features


def prepare_case(dataset: str, case_index: int, device: str, force: bool = False) -> None:
    info = DATASET_INFO[dataset]
    rows = json.loads((info["experiment"] / "samples.json").read_text(encoding="utf-8"))
    row = rows[case_index]
    output = info["output"] / f"{case_index:04d}.npz"
    if output.is_file() and not force:
        with np.load(output, allow_pickle=False) as cache:
            if np.isfinite(cache["features"]).all() and cache["case_index"] == case_index:
                print(f"已存在 {output}", flush=True)
                return
    encoder = FrozenEncoder(device)
    if dataset == "amos_mm":
        import nibabel as nib
        image = nib.load(str(info["data"] / row["file_path"]))
        orient = nib.orientations.ornt_transform(
            nib.orientations.io_orientation(image.affine), nib.orientations.axcodes2ornt(("R", "A", "S"))
        )
        total = int(nib.orientations.apply_orientation(np.empty(image.shape, dtype=np.uint8), orient).shape[2])
    else:
        total = None
        # MR的总长度和最终特征在同一次ZIP解析中确定；先用表元数据建立候选，随后严格核对。
        import nibabel as nib
        from prepare_mrrate_1k_experiment import plane_axis
        with zipfile.ZipFile(info["data"] / row["zip_path"]) as archive:
            prefix = row["study_uid"] + "_"
            members = {Path(name).name[len(prefix):-7]: name for name in archive.namelist()
                       if "/img/" in name and name.endswith(".nii.gz") and Path(name).name.startswith(prefix)}
            with tempfile.TemporaryDirectory(prefix=f"block3_header_{row['study_uid']}_") as temp:
                temp = Path(temp)
                total = 0
                for meta in row["series"]:
                    member = members.get(meta["series_id"])
                    if member is None:
                        continue
                    archive.extract(member, temp)
                    image = nib.as_closest_canonical(nib.load(str(temp / member)))
                    if len(image.shape) == 3:
                        total += int(image.shape[plane_axis(meta["acquisition_plane"], image.header.get_zooms()[:3])])
    exclusion = info["output"] / f"{case_index:04d}.excluded.json"
    if total is None or total < TARGET:
        info["output"].mkdir(parents=True, exist_ok=True)
        exclusion.write_text(json.dumps({
            "dataset": dataset, "case_index": case_index, "original_count": total,
            "reason": f"完整序列少于T={TARGET}", "seed": SEED,
        }, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"排除 {dataset} case={case_index}: N={total} < T={TARGET}", flush=True)
        return
    try:
        samplings, source_indices = selected_union(total, case_index)
    except ValueError as error:
        if "不足T" not in str(error):
            raise
        info["output"].mkdir(parents=True, exist_ok=True)
        exclusion.write_text(json.dumps({
            "dataset": dataset, "case_index": case_index, "original_count": total,
            "reason": str(error), "seed": SEED,
        }, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"统一排除 {dataset} case={case_index}: {error}", flush=True)
        return
    if dataset == "amos_mm":
        total_check, features = prepare_amos(row, source_indices, encoder)
    else:
        total_check, features = prepare_mr(row, source_indices, encoder)
    if total_check != total:
        raise ValueError(f"原始序列长度前后不一致: {total} != {total_check}")
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(".tmp.npz")
    np.savez_compressed(
        temporary, features=features.astype(np.float32), source_indices=source_indices,
        original_count=np.asarray(total, dtype=np.int64), case_index=np.asarray(case_index, dtype=np.int64),
        sampling_json=np.asarray(json.dumps(samplings, ensure_ascii=False)),
        scan_id=np.asarray(row.get("scan_id", row.get("study_uid", str(case_index)))),
        protocol=np.asarray("full_original_sequence -> block3 delete -> uniform T=64; only final slices encoded"),
    )
    temporary.replace(output)
    print(f"完成 {dataset} case={case_index}, N={total}, 编码={len(source_indices)}: {output}", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", choices=sorted(DATASET_INFO), required=True)
    parser.add_argument("--case-index", type=int, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    prepare_case(args.dataset, args.case_index, args.device, args.force)


if __name__ == "__main__":
    main()
