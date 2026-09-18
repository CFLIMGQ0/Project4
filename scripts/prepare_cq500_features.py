#!/usr/bin/env python3
"""Select one axial series per CQ500 examination and cache ConvNeXt slice features."""

from __future__ import annotations

import argparse
import json
import os
import re
from pathlib import Path
from typing import Any

import numpy as np
import pydicom
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import models, transforms
from tqdm import tqdm


SCAN_PATTERN = re.compile(r"CQ500[-_ ]?CT[-_ ]?(\d+)", re.IGNORECASE)
WINDOWS = ((40.0, 80.0), (500.0, 3000.0), (175.0, 50.0))
HEADER_TAGS = [
    "Modality",
    "SeriesInstanceUID",
    "SeriesDescription",
    "ProtocolName",
    "ConvolutionKernel",
    "ImageType",
    "SliceThickness",
    "ImagePositionPatient",
    "InstanceNumber",
    "Rows",
    "Columns",
]


def parse_args() -> argparse.Namespace:
    project_root = Path(__file__).resolve().parents[2]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dicom-root", type=Path, default=project_root / "datasets" / "cq500" / "dicom")
    parser.add_argument(
        "--feature-cache",
        type=Path,
        default=project_root / "outputs" / "cq500" / "convnext_tiny_scan_features.npz",
    )
    parser.add_argument(
        "--manifest",
        type=Path,
        default=project_root / "outputs" / "cq500" / "selected_series.json",
    )
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--slices-per-scan", type=int, default=28)
    parser.add_argument("--image-size", type=int, default=224)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--num-workers", type=int, default=8)
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def scan_id_from_path(path: Path) -> int | None:
    match = SCAN_PATTERN.search(str(path))
    return int(match.group(1)) if match else None


def first_dicom_header(files: list[Path]) -> Any | None:
    for path in files[: min(8, len(files))]:
        try:
            header = pydicom.dcmread(
                str(path), stop_before_pixels=True, force=True, specific_tags=HEADER_TAGS
            )
            if int(getattr(header, "Rows", 0)) > 0 and int(getattr(header, "Columns", 0)) > 0:
                return header
        except Exception:
            continue
    return None


def describe_header(header: Any) -> str:
    image_type = getattr(header, "ImageType", "")
    if isinstance(image_type, (list, tuple)):
        image_type = " ".join(str(value) for value in image_type)
    return " ".join(
        str(getattr(header, field, ""))
        for field in ("SeriesDescription", "ProtocolName", "ConvolutionKernel")
    ) + f" {image_type}"


def series_score(files: list[Path], header: Any) -> float:
    count = len(files)
    if str(getattr(header, "Modality", "CT")).upper() != "CT" or count < 10:
        return -1e9
    description = describe_header(header).lower()
    score = min(count, 100) * 0.5
    excluded = ("localizer", "scout", "surview", "topogram", "coronal", "sagittal", "mpr")
    if any(word in description for word in excluded):
        score -= 1000.0
    if "bone" in description:
        score -= 250.0
    if any(word in description for word in ("plain", "pre contrast", "non contrast", "axial")):
        score += 60.0
    if any(word in description for word in ("standard", "soft", "brain")):
        score += 30.0
    try:
        thickness = float(header.SliceThickness)
    except Exception:
        thickness = 0.0
    if thickness > 0:
        score += 180.0 - 25.0 * abs(thickness - 5.0)
        if 2.5 <= thickness <= 6.0:
            score += 100.0
    return score


def discover_series(dicom_root: Path) -> dict[int, list[dict[str, Any]]]:
    candidates: dict[int, list[dict[str, Any]]] = {}
    for root, _, names in tqdm(os.walk(dicom_root), desc="发现CQ500序列"):
        if not names:
            continue
        directory = Path(root)
        scan_id = scan_id_from_path(directory)
        if scan_id is None:
            continue
        files = [directory / name for name in names if not name.startswith(".")]
        header = first_dicom_header(files)
        if header is None:
            continue
        row = {
            "directory": directory,
            "files": files,
            "count": len(files),
            "description": describe_header(header),
            "thickness": float(getattr(header, "SliceThickness", 0.0) or 0.0),
            "score": series_score(files, header),
        }
        candidates.setdefault(scan_id, []).append(row)
    return candidates


def ordered_headers(files: list[Path]) -> list[tuple[Path, float, float]]:
    rows: list[tuple[Path, float, float]] = []
    for fallback_order, path in enumerate(files):
        try:
            header = pydicom.dcmread(
                str(path), stop_before_pixels=True, force=True, specific_tags=HEADER_TAGS
            )
            position = getattr(header, "ImagePositionPatient", None)
            z = float(position[2]) if position is not None and len(position) >= 3 else float("nan")
            instance = float(getattr(header, "InstanceNumber", fallback_order))
            rows.append((path, z, instance))
        except Exception:
            continue
    if rows and sum(np.isfinite(row[1]) for row in rows) >= max(2, len(rows) // 2):
        rows.sort(key=lambda row: (np.nan_to_num(row[1], nan=1e20), row[2]))
        deduplicated: list[tuple[Path, float, float]] = []
        seen: set[float] = set()
        for row in rows:
            key = round(row[1], 3) if np.isfinite(row[1]) else row[2]
            if key not in seen:
                deduplicated.append(row)
                seen.add(key)
        return deduplicated
    return sorted(rows, key=lambda row: row[2])


def select_resampled_slices(
    headers: list[tuple[Path, float, float]], count: int
) -> list[tuple[Path, float, float]]:
    if len(headers) <= count:
        return headers
    positions = np.asarray([row[1] for row in headers], dtype=np.float64)
    if np.all(np.isfinite(positions)) and positions[-1] - positions[0] >= 5.0 * (count - 1):
        center = 0.5 * (positions[0] + positions[-1])
        targets = center + (np.arange(count) - (count - 1) / 2.0) * 5.0
        indices = [int(np.argmin(np.abs(positions - target))) for target in targets]
        indices = sorted(set(indices))
        if len(indices) == count:
            return [headers[index] for index in indices]
    start = (len(headers) - count) // 2
    return headers[start : start + count]


def select_scan_slices(
    candidates: dict[int, list[dict[str, Any]]], slices_per_scan: int
) -> tuple[list[tuple[int, int, Path]], list[dict[str, Any]]]:
    selected_rows: list[tuple[int, int, Path]] = []
    manifest: list[dict[str, Any]] = []
    for scan_id in sorted(candidates):
        best = max(candidates[scan_id], key=lambda row: float(row["score"]))
        headers = ordered_headers(best["files"])
        chosen = select_resampled_slices(headers, slices_per_scan)
        for order, (path, _, _) in enumerate(chosen):
            selected_rows.append((scan_id, order, path))
        manifest.append(
            {
                "scan_id": scan_id,
                "series_directory": str(best["directory"]),
                "description": best["description"],
                "slice_thickness": best["thickness"],
                "available_slices": len(headers),
                "selected_slices": len(chosen),
                "score": best["score"],
            }
        )
    return selected_rows, manifest


def window_channel(values: np.ndarray, level: float, width: float) -> np.ndarray:
    low = level - width / 2.0
    channel = np.clip((values - low) / width, 0.0, 1.0)
    return np.rint(channel * 255.0).astype(np.uint8)


class DicomSliceDataset(Dataset):
    def __init__(self, rows: list[tuple[int, int, Path]], image_size: int) -> None:
        self.rows = rows
        self.transform = transforms.Compose(
            [
                transforms.Resize((image_size, image_size), antialias=True),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=(0.485, 0.456, 0.406),
                    std=(0.229, 0.224, 0.225),
                ),
            ]
        )

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, int, int]:
        scan_id, order, path = self.rows[index]
        dataset = pydicom.dcmread(str(path), force=True)
        values = dataset.pixel_array.astype(np.float32)
        values = values * float(getattr(dataset, "RescaleSlope", 1.0))
        values = values + float(getattr(dataset, "RescaleIntercept", 0.0))
        channels = [window_channel(values, level, width) for level, width in WINDOWS]
        image = np.stack(channels, axis=-1)
        if str(getattr(dataset, "PhotometricInterpretation", "MONOCHROME2")) == "MONOCHROME1":
            image = 255 - image
        return self.transform(Image.fromarray(image, mode="RGB")), scan_id, order


def build_feature_extractor(device: torch.device, project_root: Path) -> torch.nn.Module:
    torch.hub.set_dir(str(project_root / "pre_weights"))
    model = models.convnext_tiny(weights=models.ConvNeXt_Tiny_Weights.IMAGENET1K_V1)
    extractor = torch.nn.Sequential(
        model.features,
        model.avgpool,
        model.classifier[0],
        torch.nn.Flatten(1),
    )
    extractor.eval().to(device)
    for parameter in extractor.parameters():
        parameter.requires_grad_(False)
    return extractor


def main() -> None:
    args = parse_args()
    if args.feature_cache.is_file() and not args.force:
        print(f"特征缓存已存在：{args.feature_cache}")
        return
    candidates = discover_series(args.dicom_root)
    rows, manifest = select_scan_slices(candidates, args.slices_per_scan)
    if len(manifest) != 491:
        raise RuntimeError(f"应发现491例CQ500检查，实际发现{len(manifest)}例")
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    project_root = Path(__file__).resolve().parents[2]
    extractor = build_feature_extractor(device, project_root)
    loader = DataLoader(
        DicomSliceDataset(rows, args.image_size),
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
        persistent_workers=args.num_workers > 0,
    )
    features: list[np.ndarray] = []
    patient_ids: list[np.ndarray] = []
    slice_orders: list[np.ndarray] = []
    with torch.inference_mode():
        for images, scan_ids, orders in tqdm(loader, desc="提取CQ500 ConvNeXt特征"):
            images = images.to(device, non_blocking=True)
            with torch.autocast(device_type=device.type, enabled=device.type == "cuda"):
                batch_features = extractor(images)
            features.append(batch_features.float().cpu().numpy())
            patient_ids.append(scan_ids.numpy())
            slice_orders.append(orders.numpy())
    args.feature_cache.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        args.feature_cache,
        features=np.concatenate(features).astype(np.float32),
        patient_ids=np.concatenate(patient_ids).astype(np.int16),
        slice_numbers=np.concatenate(slice_orders).astype(np.int16),
    )
    args.manifest.parent.mkdir(parents=True, exist_ok=True)
    args.manifest.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"完成：{len(manifest)}例检查，{len(rows)}张切片，缓存至{args.feature_cache}")


if __name__ == "__main__":
    main()
