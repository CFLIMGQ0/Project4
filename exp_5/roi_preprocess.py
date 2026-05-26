from __future__ import annotations

import csv
import hashlib
import json
import math
import os
import subprocess
import sys
import urllib.request
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
from PIL import Image

try:
    from tqdm import tqdm
except Exception:  # pragma: no cover
    tqdm = None


SAM2_LARGE_CHECKPOINT_URL = "https://dl.fbaipublicfiles.com/segment_anything_2/092824/sam2.1_hiera_large.pt"
SAM2_LARGE_REPO_ID = "facebook/sam2.1-hiera-large"
SAM2_LARGE_CHECKPOINT_NAME = "sam2.1_hiera_large.pt"
SAM2_LARGE_CONFIG_NAME = "configs/sam2.1/sam2.1_hiera_l.yaml"


@dataclass(frozen=True)
class Exp5ROIConfig:
    enabled: bool = True
    model_id: str = SAM2_LARGE_REPO_ID
    checkpoint_name: str = SAM2_LARGE_CHECKPOINT_NAME
    checkpoint_url: str = SAM2_LARGE_CHECKPOINT_URL
    model_cfg: str = SAM2_LARGE_CONFIG_NAME
    auto_install_package: bool = True
    force_rebuild: bool = False
    max_images: int = 0
    num_shards: int = 1
    shard_gpus: tuple[int, ...] = ()
    top_k_per_image: int = 2
    crop_margin_ratio: float = 0.08
    save_mask_png: bool = True
    pred_iou_thresh: float = 0.88
    stability_score_thresh: float = 0.92
    min_area_ratio: float = 0.003
    max_area_ratio: float = 0.45
    min_bbox_side_ratio: float = 0.035
    nms_iou_thresh: float = 0.80
    points_per_side: int = 32
    crop_n_layers: int = 0
    crop_n_points_downscale_factor: int = 2
    min_mask_region_area: int = 100
    output_dir_name: str = "roi/task2/sam2_1_hiera_large"


def _progress(iterable, total: int | None = None, desc: str = "处理中"):
    if tqdm is not None:
        return tqdm(iterable, total=total, desc=desc, dynamic_ncols=True)
    return iterable


def _parse_int_tuple(value: Any) -> tuple[int, ...]:
    if value is None:
        return ()
    if isinstance(value, str):
        parts = [part.strip() for part in value.split(",")]
        return tuple(int(part) for part in parts if part)
    if isinstance(value, int):
        return (int(value),)
    if isinstance(value, (list, tuple)):
        return tuple(int(item) for item in value)
    return ()


def _sha1_text(value: str) -> str:
    return hashlib.sha1(value.encode("utf-8")).hexdigest()


def _source_key(image_path: str | Path) -> str:
    return str(Path(image_path).expanduser().resolve())


def _download_file(url: str, target_path: Path) -> None:
    target_path.parent.mkdir(parents=True, exist_ok=True)
    temp_path = target_path.with_name(f"{target_path.name}.{os.getpid()}.tmp")
    try:
        with urllib.request.urlopen(url) as response, temp_path.open("wb") as file:
            total = int(response.headers.get("Content-Length", "0") or 0)
            progress = tqdm(total=total, unit="B", unit_scale=True, desc=f"下载 {target_path.name}") if tqdm and total > 0 else None
            while True:
                chunk = response.read(1024 * 1024)
                if not chunk:
                    break
                file.write(chunk)
                if progress is not None:
                    progress.update(len(chunk))
            if progress is not None:
                progress.close()
        os.replace(temp_path, target_path)
    finally:
        if temp_path.exists():
            temp_path.unlink(missing_ok=True)


def _ensure_sam2_package(auto_install: bool) -> None:
    try:
        __import__("sam2")
        return
    except Exception:
        if not auto_install:
            raise RuntimeError(
                "未安装 sam2。请先安装 facebookresearch/sam2，或在 exp_5 ROI 配置中启用 auto_install_package。"
            )

    print("[EXP5 ROI] 当前环境未检测到 sam2，开始自动安装 facebookresearch/sam2。")
    env = os.environ.copy()
    env.setdefault("SAM2_BUILD_CUDA", "0")
    subprocess.check_call(
        [
            sys.executable,
            "-m",
            "pip",
            "install",
            "hydra-core",
            "omegaconf",
            "iopath",
        ]
    )
    # 不让 pip 顺手升级 torch/torchvision；当前项目训练环境已经固定可用。
    subprocess.check_call(
        [
            sys.executable,
            "-m",
            "pip",
            "install",
            "--no-build-isolation",
            "--no-deps",
            "git+https://github.com/facebookresearch/sam2.git",
        ],
        env=env,
    )


def _resolve_checkpoint_path(project_root: Path, cfg: Exp5ROIConfig) -> Path:
    checkpoint_dir = project_root / "pre_weights" / "sam2"
    checkpoint_path = checkpoint_dir / cfg.checkpoint_name
    if checkpoint_path.is_file():
        return checkpoint_path

    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    try:
        from huggingface_hub import hf_hub_download

        print(f"[EXP5 ROI] 下载 SAM2 权重: {cfg.model_id}/{cfg.checkpoint_name}")
        downloaded_path = hf_hub_download(
            repo_id=cfg.model_id,
            filename=cfg.checkpoint_name,
            local_dir=str(checkpoint_dir),
        )
        return Path(downloaded_path)
    except Exception as exc:
        print(f"[EXP5 ROI] HuggingFace 下载失败，改用 Meta 公共地址: {exc}")
        _download_file(cfg.checkpoint_url, checkpoint_path)
        return checkpoint_path


def _build_mask_generator(cfg: Exp5ROIConfig, checkpoint_path: Path, device_override: str | None = None):
    _ensure_sam2_package(cfg.auto_install_package)
    from sam2.automatic_mask_generator import SAM2AutomaticMaskGenerator
    from sam2.build_sam import build_sam2

    warnings.filterwarnings(
        "ignore",
        message=r".*Skipping the post-processing step due to the error above.*",
        category=UserWarning,
    )
    warnings.filterwarnings(
        "ignore",
        message=r".*profile_node.*",
        category=UserWarning,
    )
    warnings.filterwarnings(
        "ignore",
        message=r".*The given NumPy array is not writable.*",
        category=UserWarning,
    )
    warnings.filterwarnings(
        "ignore",
        message=r".*The default value of the antialias parameter.*",
        category=UserWarning,
    )

    device = device_override or ("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[EXP5 ROI] 加载 SAM2: cfg={cfg.model_cfg}, checkpoint={checkpoint_path}, device={device}")
    sam2_model = build_sam2(cfg.model_cfg, str(checkpoint_path), device=device, apply_postprocessing=False)
    if int(cfg.min_mask_region_area) > 0:
        print("[EXP5 ROI] 关闭 SAM2 CUDA 小区域后处理；后续仍使用 exp_5 自己的面积和 bbox 过滤。")
    return SAM2AutomaticMaskGenerator(
        sam2_model,
        points_per_side=int(cfg.points_per_side),
        pred_iou_thresh=float(cfg.pred_iou_thresh),
        stability_score_thresh=float(cfg.stability_score_thresh),
        crop_n_layers=int(cfg.crop_n_layers),
        crop_n_points_downscale_factor=int(cfg.crop_n_points_downscale_factor),
        min_mask_region_area=0,
        output_mode="binary_mask",
    )


def _bbox_iou_xywh(a: list[float], b: list[float]) -> float:
    ax1, ay1, aw, ah = a
    bx1, by1, bw, bh = b
    ax2, ay2 = ax1 + aw, ay1 + ah
    bx2, by2 = bx1 + bw, by1 + bh
    inter_w = max(0.0, min(ax2, bx2) - max(ax1, bx1))
    inter_h = max(0.0, min(ay2, by2) - max(ay1, by1))
    inter = inter_w * inter_h
    union = max(aw * ah + bw * bh - inter, 1e-8)
    return float(inter / union)


def _filter_and_rank_masks(
    raw_masks: list[dict[str, Any]],
    image_width: int,
    image_height: int,
    cfg: Exp5ROIConfig,
) -> list[dict[str, Any]]:
    image_area = max(1, int(image_width) * int(image_height))
    min_side = float(cfg.min_bbox_side_ratio) * float(min(image_width, image_height))
    candidates: list[dict[str, Any]] = []
    for mask in raw_masks:
        bbox = [float(item) for item in mask.get("bbox", [0, 0, 0, 0])]
        if len(bbox) != 4:
            continue
        area = float(mask.get("area", 0.0))
        area_ratio = area / float(image_area)
        if area_ratio < float(cfg.min_area_ratio) or area_ratio > float(cfg.max_area_ratio):
            continue
        if bbox[2] < min_side or bbox[3] < min_side:
            continue
        predicted_iou = float(mask.get("predicted_iou", 0.0))
        stability_score = float(mask.get("stability_score", 0.0))
        score = predicted_iou * stability_score * math.sqrt(max(area_ratio, 1e-8))
        candidates.append(
            {
                "bbox": bbox,
                "area": area,
                "area_ratio": area_ratio,
                "predicted_iou": predicted_iou,
                "stability_score": stability_score,
                "score": score,
                "segmentation": mask.get("segmentation"),
            }
        )

    candidates.sort(key=lambda item: float(item["score"]), reverse=True)
    selected: list[dict[str, Any]] = []
    for candidate in candidates:
        if any(_bbox_iou_xywh(candidate["bbox"], item["bbox"]) > float(cfg.nms_iou_thresh) for item in selected):
            continue
        selected.append(candidate)
        if len(selected) >= int(cfg.top_k_per_image):
            break
    return selected


def _crop_bbox_with_margin(
    image: Image.Image,
    bbox: list[float],
    margin_ratio: float,
) -> tuple[Image.Image, list[int]]:
    image_width, image_height = image.size
    x, y, w, h = bbox
    margin = float(margin_ratio) * max(float(w), float(h))
    left = max(0, int(math.floor(x - margin)))
    top = max(0, int(math.floor(y - margin)))
    right = min(image_width, int(math.ceil(x + w + margin)))
    bottom = min(image_height, int(math.ceil(y + h + margin)))
    if right <= left or bottom <= top:
        left, top, right, bottom = 0, 0, image_width, image_height
    return image.crop((left, top, right, bottom)), [left, top, right - left, bottom - top]


def _write_image_roi_outputs(
    *,
    source_image_path: str,
    image: Image.Image,
    masks: list[dict[str, Any]],
    digest: str,
    roi_root: Path,
    cfg: Exp5ROIConfig,
) -> list[dict[str, Any]]:
    crop_dir = roi_root / "crops" / digest[:2]
    mask_dir = roi_root / "masks" / digest[:2]
    crop_dir.mkdir(parents=True, exist_ok=True)
    if cfg.save_mask_png:
        mask_dir.mkdir(parents=True, exist_ok=True)

    entries: list[dict[str, Any]] = []
    for rank, mask in enumerate(masks, start=1):
        crop_path = crop_dir / f"{digest}_roi{rank:02d}.jpg"
        crop, crop_bbox = _crop_bbox_with_margin(
            image,
            [float(item) for item in mask["bbox"]],
            float(cfg.crop_margin_ratio),
        )
        if not crop_path.is_file() or cfg.force_rebuild:
            crop.save(crop_path, quality=92)

        mask_path = ""
        segmentation = mask.get("segmentation")
        if cfg.save_mask_png and segmentation is not None:
            mask_array = np.asarray(segmentation).astype(np.uint8) * 255
            resolved_mask_path = mask_dir / f"{digest}_roi{rank:02d}.png"
            if not resolved_mask_path.is_file() or cfg.force_rebuild:
                Image.fromarray(mask_array, mode="L").save(resolved_mask_path)
            mask_path = str(resolved_mask_path)

        entries.append(
            {
                "source_image_path": source_image_path,
                "crop_path": str(crop_path),
                "mask_path": mask_path,
                "rank": rank,
                "score": float(mask["score"]),
                "predicted_iou": float(mask["predicted_iou"]),
                "stability_score": float(mask["stability_score"]),
                "area": float(mask["area"]),
                "area_ratio": float(mask["area_ratio"]),
                "bbox_x": int(round(float(mask["bbox"][0]))),
                "bbox_y": int(round(float(mask["bbox"][1]))),
                "bbox_w": int(round(float(mask["bbox"][2]))),
                "bbox_h": int(round(float(mask["bbox"][3]))),
                "crop_x": int(crop_bbox[0]),
                "crop_y": int(crop_bbox[1]),
                "crop_w": int(crop_bbox[2]),
                "crop_h": int(crop_bbox[3]),
            }
        )
    return entries


def _metadata_path_for_image(roi_root: Path, digest: str) -> Path:
    return roi_root / "metadata" / digest[:2] / f"{digest}.json"


def _read_cached_metadata(metadata_path: Path) -> list[dict[str, Any]]:
    payload = json.loads(metadata_path.read_text(encoding="utf-8"))
    entries = payload.get("entries", [])
    if not isinstance(entries, list):
        return []
    return [entry for entry in entries if isinstance(entry, dict)]


def _write_cached_metadata(metadata_path: Path, payload: dict[str, Any]) -> None:
    metadata_path.parent.mkdir(parents=True, exist_ok=True)
    temp_path = metadata_path.with_name(f"{metadata_path.name}.{os.getpid()}.tmp")
    temp_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    os.replace(temp_path, metadata_path)


def _collect_unique_image_paths(training_context: dict[str, Any], task_name: str) -> list[str]:
    task_payload = training_context.get("tasks", {}).get(task_name, {})
    split_data = task_payload.get("split", {})
    paths: dict[str, None] = {}
    for split_name in ("train", "val", "test"):
        for record in split_data.get(split_name, []):
            for image_path in record.get("image_paths", []):
                paths[_source_key(image_path)] = None
    return list(paths.keys())


def _write_roi_index(index_path: Path, rows: list[dict[str, Any]]) -> None:
    fieldnames = [
        "source_image_path",
        "crop_path",
        "mask_path",
        "rank",
        "score",
        "predicted_iou",
        "stability_score",
        "area",
        "area_ratio",
        "bbox_x",
        "bbox_y",
        "bbox_w",
        "bbox_h",
        "crop_x",
        "crop_y",
        "crop_w",
        "crop_h",
    ]
    index_path.parent.mkdir(parents=True, exist_ok=True)
    temp_path = index_path.with_name(f"{index_path.name}.{os.getpid()}.tmp")
    with temp_path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key, "") for key in fieldnames})
    os.replace(temp_path, index_path)


def _read_roi_index(index_path: Path) -> list[dict[str, Any]]:
    if not index_path.is_file():
        return []
    with index_path.open("r", encoding="utf-8", newline="") as file:
        return [dict(row) for row in csv.DictReader(file)]


def _process_roi_image_paths(
    *,
    image_paths: list[str],
    cfg: Exp5ROIConfig,
    roi_root: Path,
    mask_generator_factory,
    progress_desc: str,
    shard_index_path: Path | None = None,
) -> dict[str, Any]:
    index_rows: list[dict[str, Any]] = []
    processed = 0
    reused = 0
    failed = 0
    empty = 0
    mask_generator = None

    iterator = _progress(image_paths, total=len(image_paths), desc=progress_desc)
    for image_path in iterator:
        source_path = _source_key(image_path)
        digest = _sha1_text(source_path)
        metadata_path = _metadata_path_for_image(roi_root, digest)
        if metadata_path.is_file() and not cfg.force_rebuild:
            try:
                cached_entries = _read_cached_metadata(metadata_path)
                index_rows.extend(cached_entries)
                reused += 1
                continue
            except Exception:
                metadata_path.unlink(missing_ok=True)

        try:
            if mask_generator is None:
                mask_generator = mask_generator_factory()
            with Image.open(source_path) as pil_image:
                image = pil_image.convert("RGB")
                image_array = np.array(image, copy=True)
                raw_masks = mask_generator.generate(image_array)
                selected_masks = _filter_and_rank_masks(
                    raw_masks=raw_masks,
                    image_width=image.width,
                    image_height=image.height,
                    cfg=cfg,
                )
                entries = _write_image_roi_outputs(
                    source_image_path=source_path,
                    image=image,
                    masks=selected_masks,
                    digest=digest,
                    roi_root=roi_root,
                    cfg=cfg,
                )
            if not entries:
                empty += 1
            index_rows.extend(entries)
            _write_cached_metadata(
                metadata_path,
                {
                    "source_image_path": source_path,
                    "digest": digest,
                    "model_id": cfg.model_id,
                    "checkpoint_name": cfg.checkpoint_name,
                    "entries": entries,
                },
            )
            processed += 1
        except Exception as exc:
            failed += 1
            _write_cached_metadata(
                metadata_path,
                {
                    "source_image_path": source_path,
                    "digest": digest,
                    "error": str(exc),
                    "entries": [],
                },
            )

        if hasattr(iterator, "set_postfix"):
            iterator.set_postfix(
                {
                    "新建": processed,
                    "复用": reused,
                    "空": empty,
                    "失败": failed,
                },
                refresh=False,
            )

    result = {
        "image_count": len(image_paths),
        "roi_count": len(index_rows),
        "processed_images": processed,
        "reused_images": reused,
        "empty_images": empty,
        "failed_images": failed,
    }
    if shard_index_path is not None:
        _write_roi_index(shard_index_path, index_rows)
        result["shard_index_path"] = str(shard_index_path)
    else:
        result["index_rows"] = index_rows
    return result


def _process_roi_shard(payload: dict[str, Any]) -> dict[str, Any]:
    cfg: Exp5ROIConfig = payload["cfg"]
    roi_root = Path(payload["roi_root"])
    checkpoint_path = Path(payload["checkpoint_path"])
    shard_id = int(payload["shard_id"])
    num_shards = int(payload["num_shards"])
    gpu_id = payload.get("gpu_id")
    image_paths = [str(path) for path in payload.get("image_paths", [])]
    shard_index_path = roi_root / "shards" / f"roi_index_shard_{shard_id:02d}.csv"

    if not image_paths:
        _write_roi_index(shard_index_path, [])
        return {
            "shard_id": shard_id,
            "num_shards": num_shards,
            "gpu_id": gpu_id,
            "image_count": 0,
            "roi_count": 0,
            "processed_images": 0,
            "reused_images": 0,
            "empty_images": 0,
            "failed_images": 0,
            "shard_index_path": str(shard_index_path),
        }

    device_override = None
    if gpu_id is not None and torch.cuda.is_available():
        gpu_id = int(gpu_id)
        torch.cuda.set_device(gpu_id)
        device_override = f"cuda:{gpu_id}"

    print(
        f"[EXP5 ROI] shard {shard_id + 1}/{num_shards} "
        f"使用 {'GPU ' + str(gpu_id) if gpu_id is not None else 'CPU'}，图片数={len(image_paths)}"
    )

    def build_worker_mask_generator():
        return _build_mask_generator(cfg, checkpoint_path, device_override=device_override)

    result = _process_roi_image_paths(
        image_paths=image_paths,
        cfg=cfg,
        roi_root=roi_root,
        mask_generator_factory=build_worker_mask_generator,
        progress_desc=f"EXP5-SAM2 shard{shard_id + 1}",
        shard_index_path=shard_index_path,
    )
    result.update({"shard_id": shard_id, "num_shards": num_shards, "gpu_id": gpu_id})
    return result


def _resolve_shard_gpus(cfg: Exp5ROIConfig, num_shards: int) -> tuple[int | None, ...]:
    if num_shards <= 1:
        return (None,)
    if not torch.cuda.is_available():
        raise ValueError("EXP5 ROI 多 shard 分割需要 CUDA；当前环境未检测到可用 GPU")
    raw_gpus = cfg.shard_gpus or tuple(range(num_shards))
    if len(raw_gpus) < num_shards:
        raise ValueError(f"EXP5 ROI num_shards={num_shards}，但 shard_gpus 只有 {len(raw_gpus)} 个")
    shard_gpus = tuple(int(gpu_id) for gpu_id in raw_gpus[:num_shards])
    if torch.cuda.is_available():
        gpu_count = torch.cuda.device_count()
        invalid_gpus = [gpu_id for gpu_id in shard_gpus if gpu_id < 0 or gpu_id >= gpu_count]
        if invalid_gpus:
            raise ValueError(f"EXP5 ROI shard_gpus 包含不可用 GPU: {invalid_gpus}，当前可用数量={gpu_count}")
    return shard_gpus


def normalize_exp5_roi_config(raw_cfg: dict[str, Any] | None) -> Exp5ROIConfig:
    raw_cfg = raw_cfg or {}
    return Exp5ROIConfig(
        enabled=bool(raw_cfg.get("enabled", True)),
        model_id=str(raw_cfg.get("model_id", SAM2_LARGE_REPO_ID)).strip() or SAM2_LARGE_REPO_ID,
        checkpoint_name=str(raw_cfg.get("checkpoint_name", SAM2_LARGE_CHECKPOINT_NAME)).strip()
        or SAM2_LARGE_CHECKPOINT_NAME,
        checkpoint_url=str(raw_cfg.get("checkpoint_url", SAM2_LARGE_CHECKPOINT_URL)).strip()
        or SAM2_LARGE_CHECKPOINT_URL,
        model_cfg=str(raw_cfg.get("model_cfg", SAM2_LARGE_CONFIG_NAME)).strip() or SAM2_LARGE_CONFIG_NAME,
        auto_install_package=bool(raw_cfg.get("auto_install_package", True)),
        force_rebuild=bool(raw_cfg.get("force_rebuild", False)),
        max_images=int(raw_cfg.get("max_images", 0)),
        num_shards=max(1, int(raw_cfg.get("num_shards", 1))),
        shard_gpus=_parse_int_tuple(raw_cfg.get("shard_gpus", ())),
        top_k_per_image=int(raw_cfg.get("top_k_per_image", 2)),
        crop_margin_ratio=float(raw_cfg.get("crop_margin_ratio", 0.08)),
        save_mask_png=bool(raw_cfg.get("save_mask_png", True)),
        pred_iou_thresh=float(raw_cfg.get("pred_iou_thresh", 0.88)),
        stability_score_thresh=float(raw_cfg.get("stability_score_thresh", 0.92)),
        min_area_ratio=float(raw_cfg.get("min_area_ratio", 0.003)),
        max_area_ratio=float(raw_cfg.get("max_area_ratio", 0.45)),
        min_bbox_side_ratio=float(raw_cfg.get("min_bbox_side_ratio", 0.035)),
        nms_iou_thresh=float(raw_cfg.get("nms_iou_thresh", 0.80)),
        points_per_side=int(raw_cfg.get("points_per_side", 32)),
        crop_n_layers=int(raw_cfg.get("crop_n_layers", 0)),
        crop_n_points_downscale_factor=int(raw_cfg.get("crop_n_points_downscale_factor", 2)),
        min_mask_region_area=int(raw_cfg.get("min_mask_region_area", 100)),
        output_dir_name=str(raw_cfg.get("output_dir_name", "roi/task2/sam2_1_hiera_large")).strip()
        or "roi/task2/sam2_1_hiera_large",
    )


def prepare_exp5_roi_cache(
    *,
    training_context: dict[str, Any],
    task_name: str,
    roi_cfg: dict[str, Any] | None = None,
) -> dict[str, Any]:
    cfg = normalize_exp5_roi_config(roi_cfg)
    if not cfg.enabled:
        return {"enabled": False, "roi_index_path": "", "roi_root_dir": ""}

    output_root = Path(training_context["output_root"]).resolve()
    project_root = output_root.parent
    dataset_base_root = Path(training_context["task_selection_dir"]).resolve().parent
    roi_root = dataset_base_root / cfg.output_dir_name
    roi_root.mkdir(parents=True, exist_ok=True)
    index_path = roi_root / "roi_index.csv"
    summary_path = roi_root / "summary.json"

    if index_path.is_file() and not bool(cfg.force_rebuild):
        summary: dict[str, Any] = {}
        if summary_path.is_file():
            try:
                loaded_summary = json.loads(summary_path.read_text(encoding="utf-8"))
                if isinstance(loaded_summary, dict):
                    summary = loaded_summary
            except Exception:
                summary = {}
        summary.update(
            {
                "enabled": True,
                "task_name": task_name,
                "roi_root_dir": str(roi_root),
                "roi_index_path": str(index_path),
                "reused_existing_index": True,
            }
        )
        print(f"[EXP5 ROI] 复用已有 ROI 索引: {index_path}")
        return summary

    image_paths = _collect_unique_image_paths(training_context, task_name)
    if cfg.max_images > 0:
        image_paths = image_paths[: int(cfg.max_images)]
    print(f"[EXP5 ROI] 待处理图片数: {len(image_paths)}")
    print(f"[EXP5 ROI] ROI 输出目录: {roi_root}")

    checkpoint_path = _resolve_checkpoint_path(project_root, cfg)
    _ensure_sam2_package(cfg.auto_install_package)

    configured_shards = max(1, int(cfg.num_shards))
    num_shards = min(configured_shards, len(image_paths)) if image_paths else 1
    if num_shards > 1 and not torch.cuda.is_available():
        print("[EXP5 ROI] 当前环境未检测到 CUDA，多 shard ROI 分割自动降级为单进程 CPU。")
        num_shards = 1
    shard_summaries: list[dict[str, Any]] = []
    index_rows: list[dict[str, Any]] = []

    if num_shards > 1:
        import multiprocessing as mp

        shard_gpus = _resolve_shard_gpus(cfg, num_shards)
        print(f"[EXP5 ROI] 启用多 GPU ROI 分割: shards={num_shards}, gpus={list(shard_gpus)}")
        payloads = []
        for shard_id in range(num_shards):
            payloads.append(
                {
                    "cfg": cfg,
                    "roi_root": str(roi_root),
                    "checkpoint_path": str(checkpoint_path),
                    "shard_id": shard_id,
                    "num_shards": num_shards,
                    "gpu_id": shard_gpus[shard_id],
                    "image_paths": image_paths[shard_id::num_shards],
                }
            )
        ctx = mp.get_context("spawn")
        with ctx.Pool(processes=num_shards) as pool:
            shard_summaries = pool.map(_process_roi_shard, payloads)
        for shard_summary in sorted(shard_summaries, key=lambda item: int(item["shard_id"])):
            index_rows.extend(_read_roi_index(Path(str(shard_summary["shard_index_path"]))))
    else:
        print("[EXP5 ROI] 使用单进程 ROI 分割")

        def build_mask_generator():
            return _build_mask_generator(cfg, checkpoint_path)

        result = _process_roi_image_paths(
            image_paths=image_paths,
            cfg=cfg,
            roi_root=roi_root,
            mask_generator_factory=build_mask_generator,
            progress_desc="EXP5-SAM2分割",
        )
        index_rows = list(result.pop("index_rows", []))
        shard_summaries = [result]

    _write_roi_index(index_path, index_rows)
    processed = sum(int(item.get("processed_images", 0)) for item in shard_summaries)
    reused = sum(int(item.get("reused_images", 0)) for item in shard_summaries)
    empty = sum(int(item.get("empty_images", 0)) for item in shard_summaries)
    failed = sum(int(item.get("failed_images", 0)) for item in shard_summaries)
    summary = {
        "enabled": True,
        "task_name": task_name,
        "roi_root_dir": str(roi_root),
        "roi_index_path": str(index_path),
        "checkpoint_path": str(checkpoint_path),
        "image_count": len(image_paths),
        "roi_count": len(index_rows),
        "processed_images": processed,
        "reused_images": reused,
        "empty_images": empty,
        "failed_images": failed,
        "num_shards": num_shards,
        "shard_gpus": list(_resolve_shard_gpus(cfg, num_shards)) if num_shards > 1 else [],
        "shards": shard_summaries,
        "config": cfg.__dict__,
    }
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(
        "[EXP5 ROI] 完成: "
        f"ROI={len(index_rows)} 新建={processed} 复用={reused} 空={empty} 失败={failed}"
    )
    return summary
