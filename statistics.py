from __future__ import annotations

import argparse
import shutil
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp", ".gif"}


@dataclass
class DetectConfig:
    input_dir: Path
    output_dir: Path
    black_threshold: int
    morph_kernel: int
    min_valid_area_ratio: float
    min_bbox_width: int
    min_bbox_height: int
    min_aspect_ratio: float
    max_aspect_ratio: float
    min_gray_std: float
    min_edge_density: float


@dataclass
class ProgressBar:
    total: int
    width: int = 32
    current: int = 0

    def update(self, step: int = 1) -> None:
        self.current = min(self.total, self.current + step)
        self.render()

    def render(self) -> None:
        if self.total <= 0:
            return
        ratio = self.current / self.total
        filled = int(self.width * ratio)
        bar = "#" * filled + "-" * (self.width - filled)
        print(f"\r处理进度: [{bar}] {self.current}/{self.total} ({ratio:.0%})", end="", flush=True)

    def close(self) -> None:
        if self.total > 0:
            self.render()
            print()


def parse_args() -> DetectConfig:
    parser = argparse.ArgumentParser(description="批量检测异常内镜图像（纯规则，不训练模型）")
    parser.add_argument("--input-dir", type=Path, required=True, help="输入图像目录（递归扫描）")
    parser.add_argument("--output-dir", type=Path, required=True, help="输出根目录")
    parser.add_argument("--black-threshold", type=int, default=22, help="黑背景阈值，B/G/R 均小于该值视为背景")
    parser.add_argument("--morph-kernel", type=int, default=5, help="形态学核大小")
    parser.add_argument("--min-valid-area-ratio", type=float, default=0.03, help="最小有效区域面积占比")
    parser.add_argument("--min-bbox-width", type=int, default=80, help="候选区域最小宽度")
    parser.add_argument("--min-bbox-height", type=int, default=80, help="候选区域最小高度")
    parser.add_argument("--min-aspect-ratio", type=float, default=0.35, help="候选区域最小宽高比 w/h")
    parser.add_argument("--max-aspect-ratio", type=float, default=3.2, help="候选区域最大宽高比 w/h")
    parser.add_argument("--min-gray-std", type=float, default=12.0, help="候选区域灰度标准差阈值")
    parser.add_argument("--min-edge-density", type=float, default=0.015, help="候选区域边缘密度阈值")

    args = parser.parse_args()
    return DetectConfig(
        input_dir=args.input_dir.expanduser().resolve(),
        output_dir=args.output_dir.expanduser().resolve(),
        black_threshold=int(args.black_threshold),
        morph_kernel=max(1, int(args.morph_kernel)),
        min_valid_area_ratio=float(args.min_valid_area_ratio),
        min_bbox_width=int(args.min_bbox_width),
        min_bbox_height=int(args.min_bbox_height),
        min_aspect_ratio=float(args.min_aspect_ratio),
        max_aspect_ratio=float(args.max_aspect_ratio),
        min_gray_std=float(args.min_gray_std),
        min_edge_density=float(args.min_edge_density),
    )


def collect_images(input_dir: Path) -> list[Path]:
    if not input_dir.is_dir():
        raise FileNotFoundError(f"输入目录不存在或不可读: {input_dir}")
    return sorted(
        p for p in input_dir.rglob("*") if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS
    )


def build_foreground_mask(image: np.ndarray, cfg: DetectConfig) -> np.ndarray:
    # 背景定义：像素三个通道都低于黑阈值
    bg_mask = np.all(image < cfg.black_threshold, axis=2).astype(np.uint8) * 255
    fg_mask = cv2.bitwise_not(bg_mask)

    k = cfg.morph_kernel
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))
    fg_mask = cv2.morphologyEx(fg_mask, cv2.MORPH_OPEN, kernel)
    fg_mask = cv2.morphologyEx(fg_mask, cv2.MORPH_CLOSE, kernel)
    return fg_mask


def compute_component_features(
    labels: np.ndarray,
    stats: np.ndarray,
    centroids: np.ndarray,
    comp_idx: int,
    image: np.ndarray,
    gray: np.ndarray,
    edge_map: np.ndarray,
) -> dict[str, float]:
    x = int(stats[comp_idx, cv2.CC_STAT_LEFT])
    y = int(stats[comp_idx, cv2.CC_STAT_TOP])
    w = int(stats[comp_idx, cv2.CC_STAT_WIDTH])
    h = int(stats[comp_idx, cv2.CC_STAT_HEIGHT])
    area = int(stats[comp_idx, cv2.CC_STAT_AREA])

    component_mask = (labels[y : y + h, x : x + w] == comp_idx)
    if not np.any(component_mask):
        gray_std = 0.0
        edge_density = 0.0
    else:
        gray_values = gray[y : y + h, x : x + w][component_mask]
        gray_std = float(np.std(gray_values)) if gray_values.size else 0.0

        edge_values = edge_map[y : y + h, x : x + w][component_mask]
        edge_density = float(np.mean(edge_values > 0)) if edge_values.size else 0.0

    return {
        "x_min": x,
        "y_min": y,
        "x_max": x + w - 1,
        "y_max": y + h - 1,
        "bbox_width": w,
        "bbox_height": h,
        "area": area,
        "aspect_ratio": float(w / h) if h > 0 else 0.0,
        "centroid_x": float(centroids[comp_idx, 0]),
        "centroid_y": float(centroids[comp_idx, 1]),
        "gray_std": gray_std,
        "edge_density": edge_density,
    }


def evaluate_component_rules(features: dict[str, float], image_area: int, cfg: DetectConfig) -> tuple[bool, list[str]]:
    reasons: list[str] = []
    area_ratio = float(features["area"] / image_area) if image_area > 0 else 0.0

    if area_ratio < cfg.min_valid_area_ratio:
        reasons.append("area_too_small")
    if features["bbox_width"] < cfg.min_bbox_width or features["bbox_height"] < cfg.min_bbox_height:
        reasons.append("bbox_too_small")
    if features["aspect_ratio"] < cfg.min_aspect_ratio or features["aspect_ratio"] > cfg.max_aspect_ratio:
        reasons.append("elongated_text_like_region")
    if features["gray_std"] < cfg.min_gray_std:
        reasons.append("texture_too_low")
    if features["edge_density"] < cfg.min_edge_density:
        reasons.append("edge_density_too_low")

    return len(reasons) == 0, reasons


def detect_single_image(image_path: Path, cfg: DetectConfig) -> tuple[dict[str, object], list[str]]:
    image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
    if image is None:
        row = {
            "image_path": str(image_path.resolve()),
            "width": np.nan,
            "height": np.nan,
            "has_valid_region": False,
            "valid_component_count": 0,
            "selected_component_area": np.nan,
            "selected_area_ratio": np.nan,
            "x_min": np.nan,
            "y_min": np.nan,
            "x_max": np.nan,
            "y_max": np.nan,
            "bbox_width": np.nan,
            "bbox_height": np.nan,
            "centroid_x": np.nan,
            "centroid_y": np.nan,
            "gray_std": np.nan,
            "edge_density": np.nan,
            "abnormal_reason": "image_read_failed",
        }
        return row, ["image_read_failed"]

    height, width = image.shape[:2]
    image_area = int(width * height)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    edge_map = cv2.Canny(gray, 50, 150)

    fg_mask = build_foreground_mask(image, cfg)
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(fg_mask, connectivity=8)

    if num_labels <= 1:
        row = {
            "image_path": str(image_path.resolve()),
            "width": width,
            "height": height,
            "has_valid_region": False,
            "valid_component_count": 0,
            "selected_component_area": np.nan,
            "selected_area_ratio": np.nan,
            "x_min": np.nan,
            "y_min": np.nan,
            "x_max": np.nan,
            "y_max": np.nan,
            "bbox_width": np.nan,
            "bbox_height": np.nan,
            "centroid_x": np.nan,
            "centroid_y": np.nan,
            "gray_std": np.nan,
            "edge_density": np.nan,
            "abnormal_reason": "no_foreground_component",
        }
        return row, ["no_foreground_component"]

    valid_components: list[dict[str, float]] = []
    fail_reasons_counter: Counter[str] = Counter()

    for comp_idx in range(1, num_labels):
        features = compute_component_features(labels, stats, centroids, comp_idx, image, gray, edge_map)
        passed, reasons = evaluate_component_rules(features, image_area, cfg)
        if passed:
            features["area_ratio"] = float(features["area"] / image_area) if image_area > 0 else 0.0
            valid_components.append(features)
        else:
            fail_reasons_counter.update(reasons)

    if not valid_components:
        # 选出现次数最多的失败原因，帮助后续人工核对
        top_reasons = [reason for reason, _count in fail_reasons_counter.most_common(3)]
        abnormal_reason = "no_component_passes_rules"
        if top_reasons:
            abnormal_reason += ";" + ";".join(top_reasons)

        row = {
            "image_path": str(image_path.resolve()),
            "width": width,
            "height": height,
            "has_valid_region": False,
            "valid_component_count": 0,
            "selected_component_area": np.nan,
            "selected_area_ratio": np.nan,
            "x_min": np.nan,
            "y_min": np.nan,
            "x_max": np.nan,
            "y_max": np.nan,
            "bbox_width": np.nan,
            "bbox_height": np.nan,
            "centroid_x": np.nan,
            "centroid_y": np.nan,
            "gray_std": np.nan,
            "edge_density": np.nan,
            "abnormal_reason": abnormal_reason,
        }
        return row, top_reasons or ["no_component_passes_rules"]

    selected = max(valid_components, key=lambda x: x["area"])
    row = {
        "image_path": str(image_path.resolve()),
        "width": width,
        "height": height,
        "has_valid_region": True,
        "valid_component_count": len(valid_components),
        "selected_component_area": int(selected["area"]),
        "selected_area_ratio": float(selected["area_ratio"]),
        "x_min": int(selected["x_min"]),
        "y_min": int(selected["y_min"]),
        "x_max": int(selected["x_max"]),
        "y_max": int(selected["y_max"]),
        "bbox_width": int(selected["bbox_width"]),
        "bbox_height": int(selected["bbox_height"]),
        "centroid_x": float(selected["centroid_x"]),
        "centroid_y": float(selected["centroid_y"]),
        "gray_std": float(selected["gray_std"]),
        "edge_density": float(selected["edge_density"]),
        "abnormal_reason": "",
    }
    return row, []


def safe_copy_with_dedup(src_path: Path, dst_dir: Path) -> Path:
    dst_dir.mkdir(parents=True, exist_ok=True)
    dst_path = dst_dir / src_path.name
    if not dst_path.exists():
        shutil.copy2(src_path, dst_path)
        return dst_path

    stem = src_path.stem
    suffix = src_path.suffix
    idx = 1
    while True:
        candidate = dst_dir / f"{stem}_{idx}{suffix}"
        if not candidate.exists():
            shutil.copy2(src_path, candidate)
            return candidate
        idx += 1


def write_markdown_summary(
    summary_path: Path,
    cfg: DetectConfig,
    total: int,
    normal_count: int,
    abnormal_count: int,
    abnormal_paths: list[str],
    abnormal_reason_stats: Counter[str],
) -> None:
    abnormal_ratio = (abnormal_count / total) if total > 0 else 0.0

    lines = [
        "# 图像有效区域检测汇总报告",
        "",
        "## 方法说明",
        "- 本脚本不训练模型、不下载外部数据、不调用远程 API。",
        "- 使用规则流程：黑背景检测 → 非背景掩膜 → 形态学清理 → 连通域分析 → 几何与纹理规则判定。",
        "",
        "## 背景检测策略",
        f"- 黑背景阈值：像素 B/G/R 三通道均小于 `{cfg.black_threshold}` 视为背景。",
        f"- 形态学核大小：`{cfg.morph_kernel}`（开运算+闭运算）。",
        "",
        "## 连通域分析策略",
        "- 在非背景掩膜上进行 8 邻域连通域分析。",
        "- 对每个连通域提取面积、外接矩形、宽高比、质心、灰度标准差、边缘密度。",
        "",
        "## 有效区域判定规则",
        f"- 面积占比 >= `{cfg.min_valid_area_ratio}`",
        f"- 外接框宽度 >= `{cfg.min_bbox_width}` 且高度 >= `{cfg.min_bbox_height}`",
        f"- 宽高比范围：`[{cfg.min_aspect_ratio}, {cfg.max_aspect_ratio}]`",
        f"- 灰度标准差 >= `{cfg.min_gray_std}`",
        f"- 边缘密度 >= `{cfg.min_edge_density}`",
        "- 若多个连通域均满足规则，选择面积最大的作为有效区域。",
        "",
        "## 批量统计结果",
        f"- 总图像数：{total}",
        f"- 正常图像数：{normal_count}",
        f"- 异常图像数：{abnormal_count}",
        f"- 异常占比：{abnormal_ratio:.2%}",
        "",
        "## 异常类型统计",
    ]

    if abnormal_reason_stats:
        for reason, count in abnormal_reason_stats.most_common():
            lines.append(f"- {reason}: {count}")
    else:
        lines.append("- 无")

    lines.extend(["", "## 异常图像路径列表"])
    if abnormal_paths:
        lines.extend([f"- {p}" for p in abnormal_paths])
    else:
        lines.append("- 无")

    summary_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    cfg = parse_args()
    image_paths = collect_images(cfg.input_dir)

    abnormal_dir = cfg.output_dir / "abnormal_images"
    abnormal_dir.mkdir(parents=True, exist_ok=True)
    csv_path = abnormal_dir / "image_valid_region_check.csv"
    md_path = abnormal_dir / "image_valid_region_summary.md"

    total = len(image_paths)
    rows: list[dict[str, object]] = []
    abnormal_paths: list[str] = []
    abnormal_reason_stats: Counter[str] = Counter()

    print(f"总图像数: {total}")
    progress = ProgressBar(total=total)

    for image_path in image_paths:
        row, reasons = detect_single_image(image_path, cfg)
        rows.append(row)

        if not bool(row["has_valid_region"]):
            copied_path = safe_copy_with_dedup(image_path, abnormal_dir)
            abnormal_paths.append(str(image_path.resolve()))
            abnormal_reason_stats.update(reasons if reasons else ["unknown"])
            print(f"\n异常图像: {image_path.resolve()} -> {copied_path.resolve()}")

        progress.update(1)

    progress.close()

    df = pd.DataFrame(rows)
    if df.empty:
        df = pd.DataFrame(
            columns=[
                "image_path",
                "width",
                "height",
                "has_valid_region",
                "valid_component_count",
                "selected_component_area",
                "selected_area_ratio",
                "x_min",
                "y_min",
                "x_max",
                "y_max",
                "bbox_width",
                "bbox_height",
                "centroid_x",
                "centroid_y",
                "gray_std",
                "edge_density",
                "abnormal_reason",
            ]
        )

    df.to_csv(csv_path, index=False, encoding="utf-8-sig")

    abnormal_count = int((~df["has_valid_region"]).sum()) if len(df) else 0
    normal_count = int(df["has_valid_region"].sum()) if len(df) else 0

    write_markdown_summary(
        summary_path=md_path,
        cfg=cfg,
        total=total,
        normal_count=normal_count,
        abnormal_count=abnormal_count,
        abnormal_paths=abnormal_paths,
        abnormal_reason_stats=abnormal_reason_stats,
    )

    print(f"检测到有效区域的图像数: {normal_count}")
    print(f"异常图像数: {abnormal_count}")
    print(f"异常图像复制目录: {abnormal_dir.resolve()}")
    print(f"CSV 已生成: {csv_path.resolve()}")
    print(f"Markdown 已生成: {md_path.resolve()}")


if __name__ == "__main__":
    main()
