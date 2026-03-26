from __future__ import annotations

import argparse
import csv
import json
import logging
import math
import random
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from PIL import Image, ImageOps, UnidentifiedImageError
from sklearn.cluster import AgglomerativeClustering
from sklearn.decomposition import PCA
from torch.utils.data import DataLoader, Dataset
from torchvision import models, transforms

try:
    from tqdm import tqdm
except ImportError:
    tqdm = None

try:
    import matplotlib.pyplot as plt
    import seaborn as sns

    PLOT_AVAILABLE = True
except ImportError:
    PLOT_AVAILABLE = False

IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"}


@dataclass
class ImageRecord:
    patient_id: str
    image_path: Path


class SafeImageDataset(Dataset):
    """安全读取图片，读取失败不抛异常，交给上层统一记录日志。"""

    def __init__(self, records: list[ImageRecord], img_size: int) -> None:
        self.records = records
        self.transform = transforms.Compose(
            [
                transforms.Resize((img_size, img_size)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ]
        )

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, index: int) -> dict[str, Any]:
        rec = self.records[index]
        try:
            with Image.open(rec.image_path) as im:
                im = ImageOps.exif_transpose(im)
                im = im.convert("RGB")
                if im.width < 8 or im.height < 8:
                    raise ValueError(f"图片尺寸过小: {im.width}x{im.height}")
                tensor = self.transform(im)
            return {
                "ok": True,
                "patient_id": rec.patient_id,
                "image_path": str(rec.image_path),
                "tensor": tensor,
            }
        except (UnidentifiedImageError, OSError, ValueError) as exc:
            return {
                "ok": False,
                "patient_id": rec.patient_id,
                "image_path": str(rec.image_path),
                "error": str(exc),
            }


def collate_fn(batch: list[dict[str, Any]]) -> dict[str, Any]:
    ok_items = [item for item in batch if item["ok"]]
    bad_items = [item for item in batch if not item["ok"]]

    if ok_items:
        tensors = torch.stack([item["tensor"] for item in ok_items], dim=0)
        patient_ids = [item["patient_id"] for item in ok_items]
        image_paths = [item["image_path"] for item in ok_items]
    else:
        tensors = None
        patient_ids = []
        image_paths = []

    return {
        "tensors": tensors,
        "patient_ids": patient_ids,
        "image_paths": image_paths,
        "bad_items": bad_items,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="患者目录内部图像分布一致性快速检查工具（无需训练）")
    parser.add_argument("--root_dir", type=Path, required=True, help="患者级目录根路径")
    parser.add_argument("--output_dir", type=Path, required=True, help="输出目录")
    parser.add_argument("--model_name", type=str, default="resnet18", choices=["resnet18", "resnet50"], help="预训练模型")
    parser.add_argument("--batch_size", type=int, default=64, help="特征提取批大小")
    parser.add_argument("--num_workers", type=int, default=4, help="DataLoader 线程数")
    parser.add_argument("--device", type=str, default="auto", choices=["auto", "cpu", "cuda"], help="运行设备")
    parser.add_argument("--img_size", type=int, default=224, help="输入图像尺寸")
    parser.add_argument("--min_cluster_ratio", type=float, default=0.8, help="最大簇占比阈值")
    parser.add_argument("--max_outlier_ratio", type=float, default=0.2, help="异常比例阈值")
    parser.add_argument("--default_sim_thr_2", type=float, default=0.8, help="两张图时默认相似度阈值")
    parser.add_argument("--save_contact_sheet", action="store_true", help="是否保存疑似不一致患者拼图")
    parser.add_argument("--save_umap", action="store_true", help="是否尝试保存全局 UMAP/PCA 图")
    parser.add_argument("--top_k_suspicious", type=int, default=20, help="导出前 K 个最可疑患者")
    parser.add_argument("--recursive", action="store_true", help="是否递归搜索患者目录")
    parser.add_argument("--sim_quantile", type=float, default=0.1, help="相似度阈值分位数（低分位）")
    parser.add_argument("--disp_quantile", type=float, default=0.9, help="离散度阈值分位数（高分位）")
    parser.add_argument("--min_group_size", type=int, default=8, help="分组自适应阈值最小样本量")
    parser.add_argument("--max_umap_points", type=int, default=5000, help="UMAP/PCA 最大采样点数")
    parser.add_argument("--seed", type=int, default=42, help="随机种子")
    return parser.parse_args()


def setup_logger(output_dir: Path) -> logging.Logger:
    output_dir.mkdir(parents=True, exist_ok=True)
    logger = logging.getLogger("check_data")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()

    formatter = logging.Formatter("%(asctime)s | %(levelname)s | %(message)s")

    fh = logging.FileHandler(output_dir / "run.log", encoding="utf-8")
    fh.setFormatter(formatter)
    logger.addHandler(fh)

    sh = logging.StreamHandler()
    sh.setFormatter(formatter)
    logger.addHandler(sh)
    return logger


def resolve_device(device: str) -> torch.device:
    if device == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device == "cuda" and not torch.cuda.is_available():
        return torch.device("cpu")
    return torch.device(device)


def build_feature_extractor(model_name: str, device: torch.device) -> nn.Module:
    if model_name == "resnet18":
        model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
    elif model_name == "resnet50":
        model = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
    else:
        raise ValueError(f"不支持模型: {model_name}")

    extractor = nn.Sequential(*list(model.children())[:-1])
    extractor.eval().to(device)
    return extractor


def iter_patient_dirs(root_dir: Path) -> list[Path]:
    if not root_dir.is_dir():
        raise NotADirectoryError(f"root_dir 不是目录: {root_dir}")
    return sorted([p for p in root_dir.iterdir() if p.is_dir()])


def collect_patient_images(root_dir: Path, recursive: bool, logger: logging.Logger) -> tuple[dict[str, list[Path]], list[str]]:
    patient_dirs = iter_patient_dirs(root_dir)
    patient_to_images: dict[str, list[Path]] = {}
    empty_patients: list[str] = []

    iterator = patient_dirs
    if tqdm is not None:
        iterator = tqdm(patient_dirs, desc="扫描患者目录", unit="患者")

    for patient_dir in iterator:
        patient_id = patient_dir.name
        if recursive:
            files = [p for p in patient_dir.rglob("*") if p.is_file() and p.suffix.lower() in IMAGE_SUFFIXES]
        else:
            files = [p for p in patient_dir.iterdir() if p.is_file() and p.suffix.lower() in IMAGE_SUFFIXES]

        files = sorted(files)
        if len(files) == 0:
            empty_patients.append(patient_id)
            logger.warning("患者目录为空，已跳过: %s", patient_id)
            continue
        patient_to_images[patient_id] = files

    return patient_to_images, empty_patients


def extract_embeddings(
    patient_to_images: dict[str, list[Path]],
    model_name: str,
    batch_size: int,
    num_workers: int,
    img_size: int,
    device: torch.device,
    logger: logging.Logger,
) -> tuple[dict[str, list[np.ndarray]], dict[str, list[str]], list[dict[str, str]], np.ndarray]:
    records: list[ImageRecord] = []
    for patient_id, image_list in patient_to_images.items():
        for image_path in image_list:
            records.append(ImageRecord(patient_id=patient_id, image_path=image_path))

    dataset = SafeImageDataset(records=records, img_size=img_size)
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        collate_fn=collate_fn,
        pin_memory=device.type == "cuda",
    )

    model = build_feature_extractor(model_name=model_name, device=device)
    patient_embeddings: dict[str, list[np.ndarray]] = defaultdict(list)
    patient_images: dict[str, list[str]] = defaultdict(list)
    broken_images: list[dict[str, str]] = []
    all_embeddings: list[np.ndarray] = []

    iterator = dataloader
    if tqdm is not None:
        iterator = tqdm(dataloader, desc="提取图像特征", unit="batch")

    with torch.no_grad():
        for batch in iterator:
            for bad in batch["bad_items"]:
                broken_images.append(
                    {
                        "patient_id": bad["patient_id"],
                        "image_path": bad["image_path"],
                        "error": bad["error"],
                    }
                )
                logger.warning("坏图跳过 | patient=%s | image=%s | error=%s", bad["patient_id"], bad["image_path"], bad["error"])

            tensors = batch["tensors"]
            if tensors is None:
                continue

            tensors = tensors.to(device, non_blocking=True)
            feats = model(tensors).flatten(1)
            feats = nn.functional.normalize(feats, p=2, dim=1)
            feats_np = feats.detach().cpu().numpy()

            for idx, emb in enumerate(feats_np):
                pid = batch["patient_ids"][idx]
                img_path = batch["image_paths"][idx]
                patient_embeddings[pid].append(emb)
                patient_images[pid].append(img_path)
                all_embeddings.append(emb)

    global_emb = np.vstack(all_embeddings) if all_embeddings else np.empty((0, 1), dtype=np.float32)
    return patient_embeddings, patient_images, broken_images, global_emb


def pairwise_similarity_stats(emb: np.ndarray) -> dict[str, float]:
    n = emb.shape[0]
    if n <= 1:
        return {
            "mean_similarity": 1.0,
            "median_similarity": 1.0,
            "min_similarity": 1.0,
            "max_similarity": 1.0,
            "similarity_std": 0.0,
        }
    sim = emb @ emb.T
    tri = sim[np.triu_indices(n, k=1)]
    return {
        "mean_similarity": float(np.mean(tri)),
        "median_similarity": float(np.median(tri)),
        "min_similarity": float(np.min(tri)),
        "max_similarity": float(np.max(tri)),
        "similarity_std": float(np.std(tri)),
    }


def centroid_dispersion_stats(emb: np.ndarray) -> tuple[dict[str, float], np.ndarray]:
    centroid = np.mean(emb, axis=0)
    centroid_norm = centroid / (np.linalg.norm(centroid) + 1e-12)
    distances = 1.0 - np.clip(emb @ centroid_norm, -1.0, 1.0)
    stats = {
        "centroid_dispersion_mean": float(np.mean(distances)),
        "centroid_dispersion_median": float(np.median(distances)),
        "centroid_dispersion_max": float(np.max(distances)),
    }
    return stats, distances


def detect_outliers_by_mad(distances: np.ndarray, mad_k: float = 3.5) -> np.ndarray:
    if len(distances) <= 2:
        return np.zeros(len(distances), dtype=bool)
    med = np.median(distances)
    mad = np.median(np.abs(distances - med))
    if mad < 1e-8:
        return distances > (med + 1e-6)
    robust_sigma = 1.4826 * mad
    thr = med + mad_k * robust_sigma
    return distances > thr


def cluster_structure(emb: np.ndarray, n: int, distance_threshold: float = 0.2) -> tuple[int, float, list[int]]:
    if n <= 2:
        labels = [0] * n
        return 1, 1.0, labels

    try:
        clusterer = AgglomerativeClustering(
            n_clusters=None,
            distance_threshold=distance_threshold,
            metric="cosine",
            linkage="average",
        )
        labels_arr = clusterer.fit_predict(emb)
    except TypeError:
        clusterer = AgglomerativeClustering(
            n_clusters=None,
            distance_threshold=distance_threshold,
            affinity="cosine",
            linkage="average",
        )
        labels_arr = clusterer.fit_predict(emb)
    except Exception:
        labels_arr = np.zeros(n, dtype=int)

    counts = Counter(labels_arr.tolist())
    cluster_count = len(counts)
    largest_cluster_ratio = max(counts.values()) / max(n, 1)
    return int(cluster_count), float(largest_cluster_ratio), labels_arr.tolist()


def group_key(image_count: int) -> str:
    if image_count <= 4:
        return str(image_count)
    if 5 <= image_count <= 9:
        return "5-9"
    return "10+"


def adaptive_quantile(values: list[float], q: float, default_val: float) -> float:
    if len(values) == 0:
        return default_val
    return float(np.quantile(np.array(values, dtype=np.float32), q))


def compute_group_thresholds(
    rows: list[dict[str, Any]],
    sim_quantile: float,
    disp_quantile: float,
    min_group_size: int,
    default_sim_thr_2: float,
) -> dict[str, Any]:
    two_sim = [r["median_similarity"] for r in rows if r["image_count"] == 2]
    if len(two_sim) >= min_group_size:
        sim_thr_2 = adaptive_quantile(two_sim, sim_quantile, default_sim_thr_2)
        sim_thr_2_source = "2图组自适应分位数"
    elif len(two_sim) > 0:
        sim_thr_2 = float(max(default_sim_thr_2, np.quantile(two_sim, sim_quantile)))
        sim_thr_2_source = "2图组样本较少，保守回退"
    else:
        sim_thr_2 = default_sim_thr_2
        sim_thr_2_source = "2图组无样本，使用默认阈值"

    rows_ge3 = [r for r in rows if r["image_count"] >= 3]
    global_sim_pool = [r["median_similarity"] for r in rows_ge3]
    global_disp_pool = [r["centroid_dispersion_median"] for r in rows_ge3]

    group_thresholds: dict[str, dict[str, float | str]] = {}
    for key in ["3", "4", "5-9", "10+"]:
        members = [r for r in rows_ge3 if group_key(r["image_count"]) == key]
        sim_pool = [r["median_similarity"] for r in members]
        disp_pool = [r["centroid_dispersion_median"] for r in members]

        if len(sim_pool) >= min_group_size and len(disp_pool) >= min_group_size:
            sim_thr = adaptive_quantile(sim_pool, sim_quantile, 0.75)
            disp_thr = adaptive_quantile(disp_pool, disp_quantile, 0.25)
            source = "组内自适应"
        elif len(global_sim_pool) >= min_group_size and len(global_disp_pool) >= min_group_size:
            sim_thr = adaptive_quantile(global_sim_pool, sim_quantile, 0.75)
            disp_thr = adaptive_quantile(global_disp_pool, disp_quantile, 0.25)
            source = "全局回退"
        else:
            sim_thr = 0.75
            disp_thr = 0.25
            source = "默认回退"

        group_thresholds[key] = {
            "similarity_threshold": float(sim_thr),
            "dispersion_threshold": float(disp_thr),
            "source": source,
            "group_size": len(members),
        }

    return {
        "sim_thr_2": float(sim_thr_2),
        "sim_thr_2_source": sim_thr_2_source,
        "group_thresholds": group_thresholds,
    }


def score_for_two(similarity: float, threshold: float) -> float:
    low = threshold - 0.2
    high = min(1.0, threshold + 0.15)
    score = (similarity - low) / (high - low + 1e-12)
    return float(np.clip(score, 0.0, 1.0) * 100.0)


def score_for_ge3(
    row: dict[str, Any],
    sim_thr: float,
    disp_thr: float,
    min_cluster_ratio: float,
    max_outlier_ratio: float,
) -> float:
    sim_norm = np.clip((row["median_similarity"] - sim_thr) / (1.0 - sim_thr + 1e-12), 0.0, 1.0)
    disp_norm = np.clip((disp_thr - row["centroid_dispersion_median"]) / (disp_thr + 1e-12), 0.0, 1.0)
    cluster_ratio_norm = np.clip((row["largest_cluster_ratio"] - min_cluster_ratio) / (1.0 - min_cluster_ratio + 1e-12), 0.0, 1.0)
    outlier_norm = np.clip((max_outlier_ratio - row["outlier_ratio"]) / (max_outlier_ratio + 1e-12), 0.0, 1.0)
    cluster_count_norm = 1.0 if row["cluster_count"] <= 1 else float(max(0.0, 1.0 - (row["cluster_count"] - 1) / 3.0))

    score = (
        0.30 * sim_norm
        + 0.25 * disp_norm
        + 0.20 * cluster_ratio_norm
        + 0.15 * outlier_norm
        + 0.10 * cluster_count_norm
    )
    return float(np.clip(score, 0.0, 1.0) * 100.0)


def evaluate_consistency(
    rows: list[dict[str, Any]],
    thresholds: dict[str, Any],
    min_cluster_ratio: float,
    max_outlier_ratio: float,
) -> None:
    sim_thr_2 = thresholds["sim_thr_2"]
    group_thr = thresholds["group_thresholds"]

    for row in rows:
        n = row["image_count"]
        reasons: list[str] = []

        if n == 1:
            row["is_consistent"] = True
            row["consistency_score"] = 100.0
            reasons.append("仅有1张图，默认判为一致（证据较弱）")
            row["reason"] = "；".join(reasons)
            continue

        if n == 2:
            sim_val = row["median_similarity"]
            ok = sim_val >= sim_thr_2
            row["is_consistent"] = bool(ok)
            row["consistency_score"] = score_for_two(sim_val, sim_thr_2)
            reasons.append(f"2图相似度={sim_val:.4f}，阈值={sim_thr_2:.4f}")
            reasons.append("通过" if ok else "未通过")
            row["reason"] = "；".join(reasons)
            continue

        gk = group_key(n)
        sim_thr = group_thr[gk]["similarity_threshold"]
        disp_thr = group_thr[gk]["dispersion_threshold"]

        cond_sim = row["median_similarity"] >= sim_thr
        cond_disp = row["centroid_dispersion_median"] <= disp_thr
        cond_cluster_ratio = row["largest_cluster_ratio"] >= min_cluster_ratio
        cond_outlier = row["outlier_ratio"] <= max_outlier_ratio
        cond_cluster_count = (row["cluster_count"] <= 1) or (row["cluster_count"] == 2 and row["largest_cluster_ratio"] >= 0.9)

        row["is_consistent"] = bool(cond_sim and cond_disp and cond_cluster_ratio and cond_outlier and cond_cluster_count)
        row["consistency_score"] = score_for_ge3(
            row=row,
            sim_thr=sim_thr,
            disp_thr=disp_thr,
            min_cluster_ratio=min_cluster_ratio,
            max_outlier_ratio=max_outlier_ratio,
        )

        reasons.append(f"median_similarity={row['median_similarity']:.4f} (阈值≥{sim_thr:.4f}) {'✓' if cond_sim else '✗'}")
        reasons.append(f"centroid_dispersion_median={row['centroid_dispersion_median']:.4f} (阈值≤{disp_thr:.4f}) {'✓' if cond_disp else '✗'}")
        reasons.append(f"largest_cluster_ratio={row['largest_cluster_ratio']:.4f} (阈值≥{min_cluster_ratio:.2f}) {'✓' if cond_cluster_ratio else '✗'}")
        reasons.append(f"outlier_ratio={row['outlier_ratio']:.4f} (阈值≤{max_outlier_ratio:.2f}) {'✓' if cond_outlier else '✗'}")
        reasons.append(f"cluster_count={row['cluster_count']} (要求单簇或次簇极小) {'✓' if cond_cluster_count else '✗'}")
        row["reason"] = "；".join(reasons)


def save_summary_by_image_count(df: pd.DataFrame, output_path: Path) -> pd.DataFrame:
    grouped = (
        df.groupby("image_count", as_index=False)
        .agg(total_cases=("patient_id", "count"), consistent_cases=("is_consistent", "sum"))
        .sort_values("image_count")
    )
    grouped["inconsistent_cases"] = grouped["total_cases"] - grouped["consistent_cases"]
    grouped["consistent_rate"] = grouped["consistent_cases"] / grouped["total_cases"].clip(lower=1)
    grouped.to_csv(output_path, index=False, encoding="utf-8-sig")
    return grouped


def render_plots(df: pd.DataFrame, output_dir: Path, logger: logging.Logger) -> None:
    if not PLOT_AVAILABLE:
        logger.warning("未安装 matplotlib/seaborn，跳过图表输出")
        return

    sns.set_style("whitegrid")

    plt.figure(figsize=(10, 5))
    sns.countplot(data=df, x="image_count", color="#4c72b0")
    plt.title("患者图像数量分布")
    plt.xlabel("每患者图像数")
    plt.ylabel("患者数")
    plt.tight_layout()
    plt.savefig(output_dir / "plot_image_count_distribution.png", dpi=150)
    plt.close()

    temp = (
        df.groupby(["image_count", "is_consistent"], as_index=False)
        .size()
        .rename(columns={"size": "count"})
    )
    plt.figure(figsize=(10, 5))
    sns.barplot(data=temp, x="image_count", y="count", hue="is_consistent")
    plt.title("不同图像数量下的一致/不一致数量")
    plt.xlabel("每患者图像数")
    plt.ylabel("患者数")
    plt.tight_layout()
    plt.savefig(output_dir / "plot_consistency_by_image_count.png", dpi=150)
    plt.close()

    plt.figure(figsize=(10, 5))
    sns.histplot(df["consistency_score"], bins=30, kde=True, color="#55a868")
    plt.title("一致性分数分布")
    plt.xlabel("consistency_score")
    plt.ylabel("频数")
    plt.tight_layout()
    plt.savefig(output_dir / "plot_consistency_score_hist.png", dpi=150)
    plt.close()


def save_embedding_scatter(
    all_embeddings: np.ndarray,
    all_meta: list[tuple[str, bool]],
    output_dir: Path,
    save_umap: bool,
    max_points: int,
    seed: int,
    logger: logging.Logger,
) -> str:
    if not PLOT_AVAILABLE:
        return "跳过：缺少 matplotlib/seaborn"

    if all_embeddings.shape[0] < 3:
        return "跳过：有效图像不足3张"

    rng = np.random.default_rng(seed)
    n = all_embeddings.shape[0]
    idx = np.arange(n)
    if n > max_points:
        idx = rng.choice(idx, size=max_points, replace=False)

    x = all_embeddings[idx]
    labels = np.array([1 if all_meta[i][1] else 0 for i in idx])

    method_used = "PCA"
    coords = None

    if save_umap:
        try:
            import umap

            reducer = umap.UMAP(n_components=2, random_state=seed)
            coords = reducer.fit_transform(x)
            method_used = "UMAP"
        except Exception as exc:
            logger.warning("UMAP 生成失败，自动回退到 PCA: %s", exc)

    if coords is None:
        pca = PCA(n_components=2, random_state=seed)
        coords = pca.fit_transform(x)

    plt.figure(figsize=(8, 7))
    plt.scatter(coords[:, 0], coords[:, 1], c=labels, s=10, alpha=0.7, cmap="coolwarm")
    plt.title(f"全局特征散点图（{method_used}）")
    plt.xlabel("dim-1")
    plt.ylabel("dim-2")
    plt.tight_layout()
    out = output_dir / f"global_embedding_{method_used.lower()}.png"
    plt.savefig(out, dpi=160)
    plt.close()
    return f"已保存：{out.name}"


def save_contact_sheet_for_patient(
    patient_id: str,
    image_paths: list[str],
    suspicious_set: set[str],
    output_dir: Path,
    thumb_size: int = 160,
    cols: int = 6,
) -> Path | None:
    if len(image_paths) == 0:
        return None

    thumbs: list[Image.Image] = []
    for p in image_paths:
        try:
            with Image.open(p) as im:
                im = ImageOps.exif_transpose(im).convert("RGB")
                im.thumbnail((thumb_size, thumb_size))
                canvas = Image.new("RGB", (thumb_size, thumb_size), (255, 255, 255))
                x_off = (thumb_size - im.width) // 2
                y_off = (thumb_size - im.height) // 2
                canvas.paste(im, (x_off, y_off))

                if Path(p).name in suspicious_set:
                    border = Image.new("RGB", (thumb_size, thumb_size), (220, 60, 60))
                    border.paste(canvas, (4, 4))
                    canvas = border
                thumbs.append(canvas)
        except Exception:
            continue

    if not thumbs:
        return None

    rows = math.ceil(len(thumbs) / cols)
    sheet = Image.new("RGB", (cols * thumb_size, rows * thumb_size), (245, 245, 245))
    for i, thumb in enumerate(thumbs):
        r = i // cols
        c = i % cols
        sheet.paste(thumb, (c * thumb_size, r * thumb_size))

    out = output_dir / f"contact_sheet_{patient_id}.jpg"
    sheet.save(out, quality=90)
    return out


def main() -> None:
    args = parse_args()
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    output_dir: Path = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    logger = setup_logger(output_dir)

    device = resolve_device(args.device)
    logger.info("使用设备: %s", device)

    patient_to_images_raw, empty_patients = collect_patient_images(args.root_dir, args.recursive, logger)
    if not patient_to_images_raw:
        logger.error("没有可处理的患者图像，程序结束")
        return

    patient_embeddings, patient_images_valid, broken_images, all_embeddings = extract_embeddings(
        patient_to_images=patient_to_images_raw,
        model_name=args.model_name,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        img_size=args.img_size,
        device=device,
        logger=logger,
    )

    rows: list[dict[str, Any]] = []
    dropped_patients: list[str] = []

    for patient_id in sorted(patient_to_images_raw.keys()):
        emb_list = patient_embeddings.get(patient_id, [])
        img_paths = patient_images_valid.get(patient_id, [])
        n = len(emb_list)

        if n == 0:
            dropped_patients.append(patient_id)
            logger.warning("患者全部为坏图或读取失败，已跳过: %s", patient_id)
            continue

        emb = np.vstack(emb_list).astype(np.float32)
        sim_stats = pairwise_similarity_stats(emb)
        disp_stats, centroid_dist = centroid_dispersion_stats(emb)
        outlier_mask = detect_outliers_by_mad(centroid_dist, mad_k=3.5)
        outlier_ratio = float(np.mean(outlier_mask)) if n > 0 else 0.0

        cluster_count, largest_cluster_ratio, labels = cluster_structure(emb, n=n, distance_threshold=0.2)

        suspicious_images = [Path(img_paths[i]).name for i, is_out in enumerate(outlier_mask.tolist()) if is_out]

        row = {
            "patient_id": patient_id,
            "image_count": n,
            "is_consistent": False,
            "consistency_score": 0.0,
            **sim_stats,
            **disp_stats,
            "cluster_count": int(cluster_count),
            "largest_cluster_ratio": float(largest_cluster_ratio),
            "outlier_ratio": float(outlier_ratio),
            "suspicious_images": "|".join(suspicious_images),
            "reason": "",
            "cluster_labels": "|".join(map(str, labels)),
        }
        rows.append(row)

    if not rows:
        logger.error("没有可计算指标的患者，程序结束")
        return

    thresholds = compute_group_thresholds(
        rows=rows,
        sim_quantile=args.sim_quantile,
        disp_quantile=args.disp_quantile,
        min_group_size=args.min_group_size,
        default_sim_thr_2=args.default_sim_thr_2,
    )

    evaluate_consistency(
        rows=rows,
        thresholds=thresholds,
        min_cluster_ratio=args.min_cluster_ratio,
        max_outlier_ratio=args.max_outlier_ratio,
    )

    df = pd.DataFrame(rows).sort_values(["consistency_score", "image_count", "patient_id"], ascending=[True, True, True])

    patient_csv = output_dir / "patient_level_metrics.csv"
    df_out = df.drop(columns=["cluster_labels"])
    df_out.to_csv(patient_csv, index=False, encoding="utf-8-sig", quoting=csv.QUOTE_MINIMAL)

    summary_df = save_summary_by_image_count(df_out, output_dir / "summary_by_image_count.csv")

    suspicious_df = df_out.sort_values("consistency_score", ascending=True).head(args.top_k_suspicious)
    suspicious_df.to_csv(output_dir / "top_suspicious_patients.csv", index=False, encoding="utf-8-sig")

    render_plots(df_out, output_dir, logger)

    patient_consistent_map = {r["patient_id"]: bool(r["is_consistent"]) for r in rows}
    all_meta: list[tuple[str, bool]] = []
    for pid, paths in patient_images_valid.items():
        for _ in paths:
            all_meta.append((pid, patient_consistent_map.get(pid, False)))
    emb_msg = save_embedding_scatter(
        all_embeddings=all_embeddings,
        all_meta=all_meta,
        output_dir=output_dir,
        save_umap=args.save_umap,
        max_points=args.max_umap_points,
        seed=args.seed,
        logger=logger,
    )

    contact_sheet_paths: list[str] = []
    if args.save_contact_sheet:
        sheet_dir = output_dir / "contact_sheets"
        sheet_dir.mkdir(parents=True, exist_ok=True)
        for _, row in suspicious_df.iterrows():
            pid = str(row["patient_id"])
            suspicious_set = set(str(row["suspicious_images"]).split("|")) if str(row["suspicious_images"]) else set()
            out = save_contact_sheet_for_patient(
                patient_id=pid,
                image_paths=patient_images_valid.get(pid, []),
                suspicious_set=suspicious_set,
                output_dir=sheet_dir,
            )
            if out is not None:
                contact_sheet_paths.append(str(out))

    total_patients = len(df_out)
    total_images = int(df_out["image_count"].sum())
    consistent_count = int(df_out["is_consistent"].sum())
    inconsistent_count = total_patients - consistent_count
    consistency_rate = consistent_count / max(total_patients, 1)

    global_summary = {
        "total_patients": total_patients,
        "total_images": total_images,
        "consistent_patients": consistent_count,
        "inconsistent_patients": inconsistent_count,
        "consistency_rate": consistency_rate,
        "empty_patients_skipped": empty_patients,
        "all_bad_image_patients_skipped": dropped_patients,
        "broken_image_count": len(broken_images),
        "thresholds": thresholds,
        "top_k_suspicious": suspicious_df.to_dict(orient="records"),
        "embedding_plot_status": emb_msg,
        "contact_sheets": contact_sheet_paths,
    }

    with (output_dir / "global_summary.json").open("w", encoding="utf-8") as f:
        json.dump(global_summary, f, ensure_ascii=False, indent=2)

    if broken_images:
        pd.DataFrame(broken_images).to_csv(output_dir / "broken_images.csv", index=False, encoding="utf-8-sig")

    print("\n==================== 结果汇总 ====================")
    print(f"总患者数：{total_patients}")
    print(f"总图像数：{total_images}")
    print("\n按患者图像数量统计：")
    for _, r in summary_df.iterrows():
        print(f"img数量为{int(r['image_count'])}：{int(r['total_cases'])}例（一致性数量{int(r['consistent_cases'])}）")

    print(f"\n一致患者总数：{consistent_count}")
    print(f"不一致患者总数：{inconsistent_count}")
    print(f"一致率：{consistency_rate:.2%}")
    print(f"阈值说明（2图）：{thresholds['sim_thr_2']:.4f}，来源：{thresholds['sim_thr_2_source']}")
    print(f"全局嵌入图：{emb_msg}")

    print(f"\n最可疑的前{args.top_k_suspicious}个患者（按分数升序）：")
    for _, r in suspicious_df.iterrows():
        print(
            f"- {r['patient_id']} | score={r['consistency_score']:.2f} | n={int(r['image_count'])} | "
            f"median_sim={r['median_similarity']:.4f} | disp={r['centroid_dispersion_median']:.4f} | "
            f"clusters={int(r['cluster_count'])} | largest_ratio={r['largest_cluster_ratio']:.2f} | "
            f"outlier_ratio={r['outlier_ratio']:.2f}"
        )


if __name__ == "__main__":
    main()
