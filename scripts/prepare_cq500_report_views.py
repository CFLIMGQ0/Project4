#!/usr/bin/env python3
"""逐例转换全部可解码 CT 图像，并保存用于标签盲化描述的阅片输入。"""

from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
import hashlib
import json
import os
from pathlib import Path
import re

import numpy as np
from PIL import Image, ImageDraw
import pydicom
from tqdm import tqdm


ROOT = Path(__file__).resolve().parents[2]
PATTERN = re.compile(r"CQ500[-_ ]?CT[-_ ]?(\d+)", re.I)
# 三个独立灰度窗：脑窗、硬膜下窗、骨窗。窗宽和窗位按此顺序指定。
WINDOWS = ((40.0, 80.0), (80.0, 200.0), (500.0, 3000.0))


def write_json(path: Path, value: object) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")
    tmp.replace(path)


def discover(root: Path) -> dict[int, list[str]]:
    patients: dict[int, list[str]] = {}
    for directory, dirs, names in os.walk(root):
        dirs.sort()
        match = PATTERN.search(directory)
        if match:
            for name in sorted(names):
                if name.lower().endswith(".dcm"):
                    patients.setdefault(int(match.group(1)), []).append(str(Path(directory) / name))
    return patients


def finite_values(value: object, length: int) -> list[float] | None:
    try:
        arr = np.asarray(value, dtype=float)
        return arr.tolist() if arr.shape == (length,) and np.isfinite(arr).all() else None
    except (TypeError, ValueError):
        return None


def render(values: np.ndarray, dataset: object, side: int) -> np.ndarray:
    # 仅使用像素、HU 变换和像素间距；不呈现患者信息、自由文本或诊断元数据。
    values = values.astype(np.float32)
    values *= float(getattr(dataset, "RescaleSlope", 1.0))
    values += float(getattr(dataset, "RescaleIntercept", 0.0))
    spacing = finite_values(getattr(dataset, "PixelSpacing", None), 2) or [1.0, 1.0]
    height, width = values.shape
    scale = side / max(height * spacing[0], width * spacing[1])
    dest = (max(1, round(width * spacing[1] * scale)), max(1, round(height * spacing[0] * scale)))
    panels = []
    for level, window in WINDOWS:
        pixels = np.rint(np.clip((values - level + window / 2) / window, 0, 1) * 255).astype(np.uint8)
        picture = Image.fromarray(pixels).resize(dest, Image.Resampling.LANCZOS)
        panel = Image.new("L", (side, side))
        panel.paste(picture, ((side - dest[0]) // 2, (side - dest[1]) // 2))
        panels.append(np.asarray(panel))
    return np.concatenate(panels, axis=1)


def prepare_case(task: tuple[int, list[str], str, int]) -> dict:
    scan_id, paths, output, side = task
    destination = Path(output) / f"case_{scan_id:03d}"
    destination.mkdir(parents=True, exist_ok=True)
    manifest_path = destination / "manifest.json"
    fingerprint = hashlib.sha256("\n".join(sorted(paths)).encode()).hexdigest()
    if manifest_path.exists():
        cached = json.loads(manifest_path.read_text())
        if cached.get("source_fingerprint") == fingerprint and cached.get("side") == side:
            if all((destination / s["cache"]).exists() for s in cached["series"]):
                return cached
    seen = set()
    series: dict[str, list[dict]] = {}
    failures, duplicates = [], []
    for path in sorted(paths):
        try:
            header = pydicom.dcmread(path, stop_before_pixels=True, force=True)
            uid = str(getattr(header, "SOPInstanceUID", ""))
            key = uid or hashlib.sha256(Path(path).read_bytes()).hexdigest()
            if key in seen:
                duplicates.append(path)
                continue
            seen.add(key)
            orientation = finite_values(getattr(header, "ImageOrientationPatient", None), 6)
            position = finite_values(getattr(header, "ImagePositionPatient", None), 3)
            series_uid = str(getattr(header, "SeriesInstanceUID", Path(path).parent))
            # 不同方向的图像分开，避免同序列内混合平面。
            grouping = series_uid + str([round(x, 3) for x in orientation] if orientation else "unknown")
            series.setdefault(grouping, []).append({
                "path": path, "sop_uid": uid,
                "instance": int(getattr(header, "InstanceNumber", 0)),
                "orientation": orientation, "position": position,
                "number_of_frames": int(getattr(header, "NumberOfFrames", 1)),
            })
        except Exception as exc:
            failures.append({"path": path, "stage": "header", "error": str(exc)[:500]})
    manifest_series = []
    rendered_count = 0
    for series_no, (_, records) in enumerate(sorted(series.items())):
        first_orientation = records[0]["orientation"]
        normal = np.cross(first_orientation[:3], first_orientation[3:]) if first_orientation else None
        def ordering(row: dict) -> tuple:
            loc = float(np.dot(normal, row["position"])) if normal is not None and row["position"] else row["instance"]
            return loc, row["instance"], row["path"]
        records.sort(key=ordering)
        frames, sources = [], []
        for row in records:
            try:
                ds = pydicom.dcmread(row["path"], force=True)
                values = ds.pixel_array
                if values.ndim == 2:
                    values = values[None]
                if values.ndim != 3 or getattr(ds, "SamplesPerPixel", 1) != 1:
                    raise ValueError(f"不支持的像素形状 {values.shape}")
                for frame_no, frame in enumerate(values):
                    frames.append(render(frame, ds, side))
                    sources.append({**row, "frame": frame_no})
            except Exception as exc:
                failures.append({"path": row["path"], "stage": "pixels", "error": str(exc)[:500]})
        if not frames:
            continue
        name = f"series_{series_no:03d}.npz"
        np.savez_compressed(destination / name, frames=np.stack(frames))
        axial = bool(normal is not None and abs(float(normal[2])) > 0.85)
        standard_left_right = bool(first_orientation and first_orientation[0] > 0.85)
        manifest_series.append({
            "series_index": series_no, "cache": name, "frame_count": len(frames),
            "axial": axial, "standard_left_right": standard_left_right,
            "orientation": first_orientation, "sources": sources,
        })
        rendered_count += len(frames)
    result = {
        "scan_id": scan_id, "source_fingerprint": fingerprint,
        "source_files": len(paths), "duplicate_files": len(duplicates),
        "unique_instances": len(seen), "rendered_frames": rendered_count,
        "side": side, "windows": WINDOWS, "series": manifest_series,
        "failed_files": failures, "duplicate_paths": duplicates,
        "ground_truth_read": False, "frame_sampling": "none",
        "synthetic_report": True, "review_status": "not_reviewed",
    }
    write_json(manifest_path, result)
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dicom-root", type=Path, default=ROOT / "datasets/cq500/dicom")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "outputs/cq500/image_descriptions/views")
    parser.add_argument("--case-ids", type=int, nargs="+")
    parser.add_argument("--workers", type=int, default=12)
    parser.add_argument("--side", type=int, default=224)
    args = parser.parse_args()
    assert args.side % 32 == 0
    patients = discover(args.dicom_root)
    selected = args.case_ids if args.case_ids is not None else sorted(patients)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    tasks = [(i, patients[i], str(args.output_dir), args.side) for i in selected]
    summaries = []
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        pending = {pool.submit(prepare_case, task): task[0] for task in tasks}
        for future in tqdm(as_completed(pending), total=len(pending), desc="转换患者全部影像"):
            result = future.result()
            summaries.append({k: v for k, v in result.items() if k not in {"series", "duplicate_paths"}})
            write_json(args.output_dir / "preparation_progress.json", {
                "requested_cases": len(tasks), "completed_cases": len(summaries),
                "rendered_frames": sum(x["rendered_frames"] for x in summaries),
                "failed_files": sum(len(x["failed_files"]) for x in summaries),
            })
    write_json(args.output_dir / "preparation_summary.json", sorted(summaries, key=lambda x: x["scan_id"]))
    print(f"转换完成：{len(summaries)}例，{sum(x['rendered_frames'] for x in summaries)}帧", flush=True)


if __name__ == "__main__":
    main()
