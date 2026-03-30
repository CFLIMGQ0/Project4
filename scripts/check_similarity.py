"""
不同 reportTitle 类型间的图像同质性分析工具。

使用预训练 ResNet 提取图像深度特征，通过三种统计算法分别对胃和肠的
reportTitle 类型生成相似度 / 距离邻接矩阵，帮助判断不同报告类型间
的图像是否本质相同（同质）。

算法：
  1. 质心余弦相似度（Centroid Cosine Similarity）
  2. FID（Fréchet Inception Distance）—— 业界最广泛使用的分布距离指标
  3. MMD（Maximum Mean Discrepancy，高斯 RBF 核）—— 基于核方法的两样本检验

输出（写入 paths.output_dir/check_data/）：
  - gastric/ 与 intestinal/ 各自包含 3 组热力图（PNG）+ 邻接矩阵（CSV）
  - group_stats.csv（每种 reportTitle 的图像总量与采样量）
  - summary.json（全局摘要）
  - run.log（执行日志）
"""
from __future__ import annotations

import argparse
import csv
import json
import logging
import random
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn as nn
from PIL import Image, ImageOps, UnidentifiedImageError
from scipy.linalg import sqrtm
from sklearn.metrics import pairwise_distances
from sklearn.metrics.pairwise import rbf_kernel
from torch.utils.data import DataLoader, Dataset
from torchvision import models, transforms

from check_pdf import CONFIG_PATH, load_yaml_config

try:
    from tqdm import tqdm
except ImportError:
    tqdm = None

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import matplotlib.font_manager as fm

    PLOT_AVAILABLE = True
except ImportError:
    PLOT_AVAILABLE = False

# ═══════════════════════════════════════════════════════════════════════════
# 常量
# ═══════════════════════════════════════════════════════════════════════════
IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"}
GASTRIC_KW = "胃镜"
INTESTINAL_KW = "肠镜"
ORGAN_ZH = {"gastric": "胃", "intestinal": "肠"}
FID_EPS = 1e-6  # FID 协方差矩阵正则化


# ═══════════════════════════════════════════════════════════════════════════
# 数据结构
# ═══════════════════════════════════════════════════════════════════════════
@dataclass
class PathConfig:
    dataset_root: Path
    report_csv: Path
    output_dir: Path


@dataclass
class ImageRecord:
    report_type: str
    image_path: Path


# ═══════════════════════════════════════════════════════════════════════════
# 参数解析
# ═══════════════════════════════════════════════════════════════════════════
def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="不同 reportTitle 类型间的图像同质性分析（生成邻接矩阵）",
    )
    p.add_argument("--config", type=Path, default=CONFIG_PATH,
                   help="路径配置文件（默认 configs/path.yaml）")
    p.add_argument("--report-csv", type=Path, default=None,
                   help="覆盖 valid_dicts_report_csv 路径")
    p.add_argument("--output-dir", type=Path, default=None,
                   help="覆盖输出目录")
    p.add_argument("--model-name", default="resnet18",
                   choices=["resnet18", "resnet50"],
                   help="预训练特征提取模型（默认 resnet18）")
    p.add_argument("--batch-size", type=int, default=64,
                   help="特征提取批大小（默认 64）")
    p.add_argument("--num-workers", type=int, default=4,
                   help="DataLoader 工作线程数（默认 4）")
    p.add_argument("--device", default="auto",
                   choices=["auto", "cpu", "cuda"],
                   help="运行设备（默认 auto）")
    p.add_argument("--img-size", type=int, default=224,
                   help="输入图像尺寸（默认 224）")
    p.add_argument("--max-samples", type=int, default=300,
                   help="每种 reportTitle 最大采样图片数（默认 300）")
    p.add_argument("--min-samples", type=int, default=20,
                   help="FID / MMD 所需最少样本数（默认 20）")
    p.add_argument("--seed", type=int, default=42,
                   help="随机种子（默认 42）")
    return p.parse_args()


# ═══════════════════════════════════════════════════════════════════════════
# 配置加载
# ═══════════════════════════════════════════════════════════════════════════
def build_config(args: argparse.Namespace) -> PathConfig:
    payload = load_yaml_config(args.config.expanduser())
    paths = payload.get("paths")
    if not isinstance(paths, dict):
        raise ValueError("path.yaml 需包含 paths 分组")

    cfg_dir = args.config.expanduser().resolve().parent

    def res(raw: str) -> Path:
        p = Path(raw).expanduser()
        return p if p.is_absolute() else (cfg_dir.parent / p).resolve()

    ds_root = res(str(paths["dataset_root"]))
    base_root = res(str(paths["dataset_base_root"]))

    report_csv = (
        args.report_csv.expanduser().resolve()
        if args.report_csv
        else res(str(paths["valid_dicts_report_csv"]))
        if paths.get("valid_dicts_report_csv")
        else base_root / "valid_dicts_report.csv"
    )
    out = (
        args.output_dir.expanduser().resolve()
        if args.output_dir
        else res(str(paths["output_dir"]))
    )
    return PathConfig(
        dataset_root=ds_root,
        report_csv=report_csv,
        output_dir=out / "check_data",
    )


# ═══════════════════════════════════════════════════════════════════════════
# 日志
# ═══════════════════════════════════════════════════════════════════════════
def setup_logger(out_dir: Path) -> logging.Logger:
    out_dir.mkdir(parents=True, exist_ok=True)
    lg = logging.getLogger("check_data")
    lg.setLevel(logging.INFO)
    lg.handlers.clear()

    fmt = logging.Formatter("%(asctime)s | %(levelname)s | %(message)s")
    fh = logging.FileHandler(out_dir / "run.log", encoding="utf-8")
    fh.setFormatter(fmt)
    lg.addHandler(fh)

    sh = logging.StreamHandler()
    sh.setFormatter(fmt)
    lg.addHandler(sh)
    return lg


# ═══════════════════════════════════════════════════════════════════════════
# 数据加载与分组
# ═══════════════════════════════════════════════════════════════════════════
def load_report_rows(csv_path: Path) -> list[dict[str, str]]:
    """读取 valid_dicts_report.csv，返回行列表。"""
    if not csv_path.is_file():
        raise FileNotFoundError(f"未找到报告 CSV: {csv_path}")
    with csv_path.open("r", encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        rows = list(reader)
        flds = set(reader.fieldnames or [])
    need = {"exam_dir", "reportTitle"}
    if not need.issubset(flds):
        raise KeyError(f"CSV 缺少字段: {need - flds}")
    return rows


def classify_organ(title: str) -> str | None:
    """将 reportTitle 归类为 gastric / intestinal，无法归类返回 None。"""
    if GASTRIC_KW in title:
        return "gastric"
    if INTESTINAL_KW in title:
        return "intestinal"
    return None


def collect_images_by_type(
    rows: list[dict[str, str]],
    ds_root: Path,
    lg: logging.Logger,
) -> dict[str, dict[str, list[Path]]]:
    """
    遍历报告 CSV，收集每种 reportTitle 对应的全部图片路径。

    返回 ``{organ: {reportTitle: [image_paths]}}``。
    """
    out: dict[str, dict[str, list[Path]]] = {
        "gastric": defaultdict(list),
        "intestinal": defaultdict(list),
    }
    skip: set[str] = set()

    iterator = tqdm(rows, desc="扫描图像目录", unit="行") if tqdm else rows
    for row in iterator:
        title = row.get("reportTitle", "").strip()
        organ = classify_organ(title) if title else None
        if organ is None:
            if title:
                skip.add(title)
            continue

        raw = row.get("exam_dir", "").strip()
        if not raw:
            continue
        ep = Path(raw).expanduser()
        if not ep.is_absolute():
            ep = (ds_root / ep).resolve()

        img_dir = ep / "img"
        if not img_dir.is_dir():
            continue
        for f in img_dir.iterdir():
            if f.is_file() and f.suffix.lower() in IMAGE_SUFFIXES:
                out[organ][title].append(f)

    if skip:
        lg.warning("无法归类的 reportTitle（已跳过）: %s", skip)
    return out


def sample_per_type(
    images: dict[str, list[Path]],
    max_n: int,
    rng: random.Random,
) -> dict[str, list[Path]]:
    """对每种 reportTitle 做随机采样，不超过 max_n。"""
    return {
        t: rng.sample(ps, max_n) if len(ps) > max_n else list(ps)
        for t, ps in images.items()
    }


# ═══════════════════════════════════════════════════════════════════════════
# Dataset & 特征提取
# ═══════════════════════════════════════════════════════════════════════════
class ImgDataset(Dataset):
    """安全读取图片，读取失败不抛异常。"""

    def __init__(self, recs: list[ImageRecord], size: int) -> None:
        self.recs = recs
        self.tf = transforms.Compose([
            transforms.Resize((size, size)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ])

    def __len__(self) -> int:
        return len(self.recs)

    def __getitem__(self, i: int) -> dict[str, Any]:
        r = self.recs[i]
        try:
            with Image.open(r.image_path) as im:
                im = ImageOps.exif_transpose(im)
                im = im.convert("RGB")
                return {
                    "ok": True,
                    "type": r.report_type,
                    "tensor": self.tf(im),
                    "path": str(r.image_path),
                }
        except (UnidentifiedImageError, OSError, ValueError) as e:
            return {
                "ok": False,
                "type": r.report_type,
                "path": str(r.image_path),
                "error": str(e),
            }


def _collate(batch: list[dict[str, Any]]) -> dict[str, Any]:
    ok = [b for b in batch if b["ok"]]
    bad = [b for b in batch if not b["ok"]]
    return {
        "tensors": torch.stack([b["tensor"] for b in ok]) if ok else None,
        "types": [b["type"] for b in ok],
        "bad": bad,
    }


def resolve_device(s: str) -> torch.device:
    if s == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(s)


def build_extractor(name: str, dev: torch.device) -> nn.Module:
    """构建去掉最后分类层的预训练 ResNet 特征提取器。"""
    if name == "resnet18":
        m = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
    else:
        m = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
    ext = nn.Sequential(*list(m.children())[:-1])
    return ext.eval().to(dev)


def extract_features(
    imgs: dict[str, list[Path]],
    model_name: str,
    batch_size: int,
    num_workers: int,
    img_size: int,
    dev: torch.device,
    lg: logging.Logger,
) -> tuple[dict[str, np.ndarray], int]:
    """
    提取每种 reportTitle 的 L2 归一化深度特征。

    返回 ``{reportTitle: (N, D) ndarray}`` 和损坏图像数量。
    """
    recs = [ImageRecord(t, p) for t, ps in imgs.items() for p in ps]
    if not recs:
        return {}, 0

    ds = ImgDataset(recs, img_size)
    dl = DataLoader(
        ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        collate_fn=_collate,
        pin_memory=(dev.type == "cuda"),
    )

    model = build_extractor(model_name, dev)
    embs: dict[str, list[np.ndarray]] = defaultdict(list)
    bad_count = 0

    it = tqdm(dl, desc="提取图像特征", unit="batch") if tqdm else dl
    with torch.no_grad():
        for batch in it:
            bad_count += len(batch["bad"])
            for b in batch["bad"]:
                lg.warning("坏图跳过: %s | %s", b["path"], b.get("error", ""))
            if batch["tensors"] is None:
                continue
            feats = model(batch["tensors"].to(dev, non_blocking=True)).flatten(1)
            feats = nn.functional.normalize(feats, p=2, dim=1).cpu().numpy()
            for idx, tp in enumerate(batch["types"]):
                embs[tp].append(feats[idx])

    result: dict[str, np.ndarray] = {}
    for t, vecs in embs.items():
        if vecs:
            result[t] = np.vstack(vecs).astype(np.float32)
    return result, bad_count


# ═══════════════════════════════════════════════════════════════════════════
# 算法 1：质心余弦相似度（Centroid Cosine Similarity）
# ═══════════════════════════════════════════════════════════════════════════
def algo_centroid_cosine(
    embs: dict[str, np.ndarray],
    types: list[str],
) -> np.ndarray:
    """
    计算每种 reportTitle 特征质心之间的余弦相似度。

    返回 (T, T) 对称矩阵，取值 [0, 1]。
    """
    centroids = []
    for t in types:
        c = embs[t].mean(axis=0)
        c = c / (np.linalg.norm(c) + 1e-12)
        centroids.append(c)
    C = np.vstack(centroids)  # (T, D)
    return np.clip(C @ C.T, 0.0, 1.0)


# ═══════════════════════════════════════════════════════════════════════════
# 算法 2：FID（Fréchet Inception Distance）
# ═══════════════════════════════════════════════════════════════════════════
def _fid_pair(
    mu1: np.ndarray, sigma1: np.ndarray,
    mu2: np.ndarray, sigma2: np.ndarray,
) -> float:
    """计算两组高斯分布之间的 FID。"""
    diff = mu1 - mu2
    covmean, _ = sqrtm(sigma1 @ sigma2, disp=False)
    if np.iscomplexobj(covmean):
        covmean = covmean.real
    fid = float(diff @ diff + np.trace(sigma1 + sigma2 - 2.0 * covmean))
    return max(0.0, fid)


def algo_fid(
    embs: dict[str, np.ndarray],
    types: list[str],
    min_n: int,
) -> tuple[np.ndarray, list[bool]]:
    """
    FID 邻接矩阵。

    对样本数 < min_n 的类型标记为不可用（矩阵对应行列为 NaN）。
    返回 (T, T) 矩阵和每种类型的可用标记列表。
    """
    T = len(types)
    stats: dict[str, tuple[np.ndarray, np.ndarray] | None] = {}
    valid: list[bool] = []

    for t in types:
        X = embs[t]
        if X.shape[0] >= min_n:
            mu = X.mean(axis=0)
            sigma = np.cov(X, rowvar=False) + FID_EPS * np.eye(X.shape[1])
            stats[t] = (mu, sigma)
            valid.append(True)
        else:
            stats[t] = None
            valid.append(False)

    mat = np.full((T, T), np.nan)
    pairs = [
        (i, j) for i in range(T) for j in range(i + 1, T)
        if valid[i] and valid[j]
    ]
    it = tqdm(pairs, desc="计算 FID", unit="pair") if tqdm else pairs

    for i in range(T):
        if valid[i]:
            mat[i, i] = 0.0

    for i, j in it:
        mu1, s1 = stats[types[i]]  # type: ignore[misc]
        mu2, s2 = stats[types[j]]  # type: ignore[misc]
        v = _fid_pair(mu1, s1, mu2, s2)
        mat[i, j] = v
        mat[j, i] = v

    return mat, valid


# ═══════════════════════════════════════════════════════════════════════════
# 算法 3：MMD（Maximum Mean Discrepancy，高斯 RBF 核）
# ═══════════════════════════════════════════════════════════════════════════
def _mmd_pair(X: np.ndarray, Y: np.ndarray, gamma: float) -> float:
    """计算两组样本之间的 MMD^2（RBF 核）。"""
    return max(0.0, float(
        rbf_kernel(X, X, gamma).mean()
        + rbf_kernel(Y, Y, gamma).mean()
        - 2.0 * rbf_kernel(X, Y, gamma).mean()
    ))


def algo_mmd(
    embs: dict[str, np.ndarray],
    types: list[str],
    min_n: int,
) -> tuple[np.ndarray, list[bool]]:
    """
    MMD 邻接矩阵（高斯 RBF 核 + 中位数启发式带宽）。

    返回 (T, T) 矩阵和可用标记。
    """
    T = len(types)

    # ── 自适应带宽：中位数启发式 ──
    all_data = np.vstack([embs[t] for t in types])
    rng = np.random.default_rng(42)
    n_s = min(2000, all_data.shape[0])
    sample = all_data[rng.choice(all_data.shape[0], n_s, replace=False)]
    dists = pairwise_distances(sample, metric="sqeuclidean")
    med = float(np.median(dists[np.triu_indices(n_s, k=1)]))
    gamma = 1.0 / (med + 1e-12)

    valid = [embs[t].shape[0] >= min_n for t in types]
    mat = np.full((T, T), np.nan)

    pairs = [
        (i, j) for i in range(T) for j in range(i + 1, T)
        if valid[i] and valid[j]
    ]
    it = tqdm(pairs, desc="计算 MMD", unit="pair") if tqdm else pairs

    for i in range(T):
        if valid[i]:
            mat[i, i] = 0.0

    for i, j in it:
        v = _mmd_pair(embs[types[i]], embs[types[j]], gamma)
        mat[i, j] = v
        mat[j, i] = v

    return mat, valid


# ═══════════════════════════════════════════════════════════════════════════
# 可视化工具
# ═══════════════════════════════════════════════════════════════════════════
def setup_chinese_font() -> str | None:
    """尝试为 matplotlib 配置中文字体，返回找到的字体名或 None。"""
    if not PLOT_AVAILABLE:
        return None
    candidates = [
        "WenQuanYi Micro Hei", "WenQuanYi Zen Hei",
        "Noto Sans CJK SC", "Noto Sans CJK",
        "SimHei", "Microsoft YaHei", "PingFang SC",
    ]
    for name in candidates:
        try:
            path = fm.findfont(
                fm.FontProperties(family=name), fallback_to_default=False,
            )
            if path and Path(path).exists():
                plt.rcParams["font.sans-serif"] = [name]
                plt.rcParams["axes.unicode_minus"] = False
                return name
        except Exception:
            continue
    plt.rcParams["axes.unicode_minus"] = False
    return None


def _shorten(s: str, n: int = 14) -> str:
    return s if len(s) <= n else s[: n - 1] + "…"


def save_heatmap(
    mat: np.ndarray,
    labels: list[str],
    display_labels: list[str],
    title: str,
    path: Path,
    cmap: str = "RdYlGn",
    vmin: float | None = None,
    vmax: float | None = None,
) -> None:
    """保存邻接矩阵热力图。"""
    if not PLOT_AVAILABLE:
        return
    T = len(labels)
    size = max(8, T * 0.8)
    fig, ax = plt.subplots(figsize=(size, size))

    masked = np.ma.array(mat, mask=np.isnan(mat))
    im = ax.imshow(masked, cmap=cmap, interpolation="nearest",
                   vmin=vmin, vmax=vmax)
    fig.colorbar(im, ax=ax, shrink=0.8)

    # 标注数值
    vmax_real = np.nanmax(mat) if not np.all(np.isnan(mat)) else 1.0
    vmin_real = np.nanmin(mat) if not np.all(np.isnan(mat)) else 0.0
    mid = (vmax_real + vmin_real) / 2.0
    cell_font = max(5, 9 - T // 5)
    for i in range(T):
        for j in range(T):
            if not np.isnan(mat[i, j]):
                color = "white" if mat[i, j] > mid else "black"
                ax.text(j, i, f"{mat[i, j]:.3f}", ha="center", va="center",
                        fontsize=cell_font, color=color)

    tick_font = max(6, 9 - T // 6)
    ax.set_xticks(range(T))
    ax.set_yticks(range(T))
    ax.set_xticklabels(display_labels, rotation=45, ha="right",
                       fontsize=tick_font)
    ax.set_yticklabels(display_labels, fontsize=tick_font)
    ax.set_title(title, fontsize=12, pad=10)
    fig.tight_layout()
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def save_matrix_csv(mat: np.ndarray, labels: list[str], path: Path) -> None:
    """将邻接矩阵保存为 CSV（行列标签为完整 reportTitle）。"""
    with path.open("w", encoding="utf-8-sig", newline="") as f:
        w = csv.writer(f)
        w.writerow([""] + labels)
        for i, lb in enumerate(labels):
            vals = [
                "" if np.isnan(mat[i, j]) else f"{mat[i, j]:.6f}"
                for j in range(len(labels))
            ]
            w.writerow([lb] + vals)


def save_label_legend(types: list[str], display: list[str], path: Path) -> None:
    """当热力图使用缩写标签时，保存完整标签映射。"""
    with path.open("w", encoding="utf-8-sig", newline="") as f:
        w = csv.writer(f)
        w.writerow(["display_label", "reportTitle"])
        for d, t in zip(display, types):
            w.writerow([d, t])


# ═══════════════════════════════════════════════════════════════════════════
# 单器官类型完整分析
# ═══════════════════════════════════════════════════════════════════════════
def analyze_organ(
    organ: str,
    type_imgs: dict[str, list[Path]],
    args: argparse.Namespace,
    dev: torch.device,
    base_dir: Path,
    lg: logging.Logger,
    has_cjk_font: bool,
) -> dict[str, Any]:
    """对一个器官类型执行完整的三算法分析。"""
    zh = ORGAN_ZH[organ]
    lg.info("开始分析「%s」（共 %d 种 reportTitle）", zh, len(type_imgs))

    od = base_dir / organ
    od.mkdir(parents=True, exist_ok=True)

    # ── 采样 ──
    rng = random.Random(args.seed)
    sampled = sample_per_type(type_imgs, args.max_samples, rng)

    # 按总图像数降序排列
    types = sorted(sampled, key=lambda t: len(type_imgs[t]), reverse=True)

    # ── 保存分组统计 ──
    with (od / "group_stats.csv").open("w", encoding="utf-8-sig", newline="") as f:
        w = csv.DictWriter(
            f, fieldnames=["reportTitle", "total_images", "sampled_images"],
        )
        w.writeheader()
        for t in types:
            w.writerow({
                "reportTitle": t,
                "total_images": len(type_imgs[t]),
                "sampled_images": len(sampled[t]),
            })

    # ── 特征提取 ──
    embs, bad = extract_features(
        sampled, args.model_name, args.batch_size,
        args.num_workers, args.img_size, dev, lg,
    )
    avail = [t for t in types if t in embs and embs[t].shape[0] > 0]
    if len(avail) < 2:
        lg.warning("「%s」可用类型 < 2，跳过矩阵计算", zh)
        return {"organ": organ, "status": "skipped", "reason": "可用类型不足"}

    T = len(avail)

    # 热力图标签：有中文字体用缩写中文，否则用序号
    if has_cjk_font:
        disp = [_shorten(t) for t in avail]
    else:
        disp = [f"T{i + 1}" for i in range(T)]
        save_label_legend(avail, disp, od / "label_legend.csv")

    # ── 算法 1：质心余弦相似度 ──
    lg.info("[%s] 质心余弦相似度 %dx%d", zh, T, T)
    cos_mat = algo_centroid_cosine(embs, avail)
    save_matrix_csv(cos_mat, avail, od / "centroid_cosine_similarity.csv")
    save_heatmap(cos_mat, avail, disp,
                 title=f"质心余弦相似度 — {zh}",
                 path=od / "centroid_cosine_similarity.png",
                 cmap="RdYlGn", vmin=0.0, vmax=1.0)

    # ── 算法 2：FID ──
    lg.info("[%s] FID %dx%d", zh, T, T)
    fid_mat, fid_ok = algo_fid(embs, avail, args.min_samples)
    save_matrix_csv(fid_mat, avail, od / "fid_matrix.csv")
    save_heatmap(fid_mat, avail, disp,
                 title=f"FID（Fréchet Inception Distance）— {zh}",
                 path=od / "fid_matrix.png",
                 cmap="RdYlGn_r")

    # ── 算法 3：MMD ──
    lg.info("[%s] MMD %dx%d", zh, T, T)
    mmd_mat, mmd_ok = algo_mmd(embs, avail, args.min_samples)
    save_matrix_csv(mmd_mat, avail, od / "mmd_matrix.csv")
    save_heatmap(mmd_mat, avail, disp,
                 title=f"MMD（Maximum Mean Discrepancy）— {zh}",
                 path=od / "mmd_matrix.png",
                 cmap="RdYlGn_r")

    return {
        "organ": organ,
        "zh": zh,
        "status": "done",
        "types": avail,
        "sample_counts": {t: int(embs[t].shape[0]) for t in avail},
        "broken": bad,
        "fid_skip": [avail[i] for i in range(T) if not fid_ok[i]],
        "mmd_skip": [avail[i] for i in range(T) if not mmd_ok[i]],
        "cos_mat": cos_mat,
        "fid_mat": fid_mat,
        "mmd_mat": mmd_mat,
        "dir": str(od),
    }


# ═══════════════════════════════════════════════════════════════════════════
# 终端输出工具
# ═══════════════════════════════════════════════════════════════════════════
def print_group_table(
    types: list[str],
    type_imgs: dict[str, list[Path]],
    counts: dict[str, int],
    zh: str,
) -> None:
    """在终端打印该器官类型的 reportTitle 分组总览。"""
    print(f"\n{'=' * 70}")
    print(f"  {zh}组 reportTitle（共 {len(types)} 种）")
    print(f"{'=' * 70}")
    for i, t in enumerate(types, 1):
        total = len(type_imgs[t])
        sampled = counts.get(t, 0)
        print(f"  {i:>2d}. {t}  |  总图像 {total}  |  采样 {sampled}")


def print_top_pairs(
    mat: np.ndarray,
    labels: list[str],
    name: str,
    k: int = 5,
    ascending: bool = True,
) -> None:
    """打印邻接矩阵中最相似的 Top-K 对。"""
    pairs: list[tuple[float, str, str]] = []
    T = len(labels)
    for i in range(T):
        for j in range(i + 1, T):
            if not np.isnan(mat[i, j]):
                pairs.append((mat[i, j], labels[i], labels[j]))
    pairs.sort(key=lambda x: x[0], reverse=not ascending)

    tag = "最小（最相似）" if ascending else "最大（最相似）"
    n = min(k, len(pairs))
    print(f"\n  {name} — {tag} Top-{n}：")
    for r, (v, a, b) in enumerate(pairs[:n], 1):
        a2 = a if len(a) <= 22 else a[:21] + "…"
        b2 = b if len(b) <= 22 else b[:21] + "…"
        print(f"    {r}. {a2}  <->  {b2} : {v:.4f}")


def print_interpretation_guide(min_samples: int) -> None:
    """在终端打印三种指标的解读说明。"""
    print(f"\n{'=' * 70}")
    print("  指标解读指南")
    print(f"{'=' * 70}")
    print(f"""
  [算法 1] 质心余弦相似度（Centroid Cosine Similarity）
  ─────────────────────────────────────────────────────
  原理：对每种 reportTitle 的全部图像特征取均值（质心），再计算
        两两质心间的余弦相似度。反映两组图像在特征空间中「方向」
        是否一致。
  取值范围：0 ~ 1（越大越相似）
  参考阈值：
    > 0.95       两类图像高度同质，可视为本质相同
    0.85 ~ 0.95  较为相似，存在较大重叠
    0.70 ~ 0.85  有一定差异，但仍具有可比性
    < 0.70       差异明显，属于不同类型的图像
  查看方式：
    热力图  <organ>/centroid_cosine_similarity.png （绿色 = 相似）
    矩阵值  <organ>/centroid_cosine_similarity.csv

  [算法 2] FID（Fréchet Inception Distance）
  ─────────────────────────────────────────────────────
  原理：假设两组深度特征各服从多维高斯分布，计算两个高斯分布间
        的 Fréchet 距离。同时考虑分布的均值差与协方差差异。
        FID 是图像生成与对比领域最广泛使用的分布距离指标。
  取值范围：0 ~ +∞（越小越相似）
  参考阈值：
    < 5          分布几乎一致，两类图像高度同质
    5 ~ 20       分布较为接近，存在可量化差异
    20 ~ 50      明显不同，但可能共享部分视觉特征
    > 50         差异较大，属于不同分布
  注意：样本数 < {min_samples} 的类型标记为 NaN，不参与计算。
  查看方式：
    热力图  <organ>/fid_matrix.png （绿色 = 相似）
    矩阵值  <organ>/fid_matrix.csv

  [算法 3] MMD（Maximum Mean Discrepancy，高斯 RBF 核）
  ─────────────────────────────────────────────────────
  原理：将样本映射到再生核希尔伯特空间（RKHS），比较两组样本
        在该空间中均值嵌入（mean embedding）的差异。
        使用中位数启发式（median heuristic）自适应选择核带宽。
        MMD 是非参数两样本检验的经典方法，不假设数据分布形式。
  取值范围：0 ~ +∞（越小越相似）
  参考阈值（因核带宽而异，建议以矩阵内相对大小为主）：
    < 0.001      两组分布极为接近，几乎不可区分
    0.001 ~ 0.01 存在微小差异，总体相似
    0.01 ~ 0.05  有可检测差异
    > 0.05       差异显著
  查看方式：
    热力图  <organ>/mmd_matrix.png （绿色 = 相似）
    矩阵值  <organ>/mmd_matrix.csv

  综合判断建议
  ─────────────────────────────────────────────────────
  • 三个指标从不同维度刻画分布相似性，建议综合参考。
  • 若三者一致指向「相似」，可较有把握认为两类图像同质。
  • 存在分歧时，优先参考 FID（业界最广泛认可的分布距离指标）。
  • 样本量较少的类型结果可靠性较低，请关注 group_stats.csv。
  • 上文 <organ> 指 gastric（胃）或 intestinal（肠）子目录。
""")


# ═══════════════════════════════════════════════════════════════════════════
# 主入口
# ═══════════════════════════════════════════════════════════════════════════
def main() -> None:
    args = parse_args()
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    cfg = build_config(args)
    lg = setup_logger(cfg.output_dir)
    dev = resolve_device(args.device)
    lg.info("设备=%s  模型=%s  每类最大采样=%d  FID/MMD 最少样本=%d",
            dev, args.model_name, args.max_samples, args.min_samples)

    # ── 字体 ──
    has_cjk = False
    if PLOT_AVAILABLE:
        font = setup_chinese_font()
        has_cjk = font is not None
        lg.info("中文字体: %s", font if font else "未找到（热力图将使用序号标签）")
    else:
        lg.warning("matplotlib 不可用，将跳过热力图输出（仅生成 CSV）")

    # ── 加载数据 ──
    rows = load_report_rows(cfg.report_csv)
    lg.info("加载 %d 条报告记录（来源: %s）", len(rows), cfg.report_csv)

    lg.info("扫描各检查目录下的图像文件……")
    grouped = collect_images_by_type(rows, cfg.dataset_root, lg)

    for organ in ("gastric", "intestinal"):
        zh = ORGAN_ZH[organ]
        ti = grouped[organ]
        total_imgs = sum(len(v) for v in ti.values())
        lg.info("「%s」：%d 种 reportTitle，共 %d 张图像", zh, len(ti), total_imgs)

    # ── 逐器官分析 ──
    results: list[dict[str, Any]] = []
    for organ in ("gastric", "intestinal"):
        ti = grouped[organ]
        if len(ti) < 2:
            lg.warning("「%s」可用 reportTitle 类型 < 2，跳过", ORGAN_ZH[organ])
            results.append({"organ": organ, "status": "skipped",
                            "reason": "类型不足 2 种"})
            continue

        r = analyze_organ(organ, ti, args, dev, cfg.output_dir, lg, has_cjk)
        results.append(r)

    # ── 保存全局摘要 ──
    def _jsonable(r: dict[str, Any]) -> dict[str, Any]:
        return {k: v for k, v in r.items() if not isinstance(v, np.ndarray)}

    summary = {
        "report_csv": str(cfg.report_csv),
        "output_dir": str(cfg.output_dir),
        "model": args.model_name,
        "max_samples": args.max_samples,
        "min_samples": args.min_samples,
        "seed": args.seed,
        "results": [_jsonable(r) for r in results],
    }
    sp = cfg.output_dir / "summary.json"
    sp.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")

    # ══════════════════════════════════════════════════════════════════════
    # 终端汇总
    # ══════════════════════════════════════════════════════════════════════
    print(f"\n{'#' * 70}")
    print("  图像同质性分析完成")
    print(f"{'#' * 70}")

    for r in results:
        if r["status"] != "done":
            print(f"\n  [{ORGAN_ZH.get(r['organ'], r['organ'])}] "
                  f"跳过：{r.get('reason', '')}")
            continue

        organ = r["organ"]
        zh = r["zh"]
        types = r["types"]
        counts = r["sample_counts"]

        print_group_table(types, grouped[organ], counts, zh)

        print_top_pairs(r["cos_mat"], types, "质心余弦相似度",
                        k=5, ascending=False)
        print_top_pairs(r["fid_mat"], types, "FID", k=5, ascending=True)
        print_top_pairs(r["mmd_mat"], types, "MMD", k=5, ascending=True)

        if r["fid_skip"]:
            print(f"\n  FID 因样本不足跳过的类型: "
                  f"{', '.join(r['fid_skip'])}")
        if r["mmd_skip"]:
            print(f"  MMD 因样本不足跳过的类型: "
                  f"{', '.join(r['mmd_skip'])}")

    print_interpretation_guide(args.min_samples)

    print(f"  输出目录: {cfg.output_dir}")
    print(f"  摘要文件: {sp}\n")


if __name__ == "__main__":
    main()
