from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import pandas as pd

try:
    import yaml  # type: ignore
except ImportError:
    yaml = None

PROJECT_ROOT = Path(__file__).resolve().parent
CONFIG_PATH = PROJECT_ROOT / "configs" / "path.yaml"
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp", ".gif", ".dcm"}


@dataclass
class AnalyzeConfig:
    output_csv: Path
    output_md: Path
    black_threshold: int
    morph_kernel: int
    min_area_ratio: float
    centroid_std_ratio_threshold: float
    zscore_threshold: float


@dataclass
class PathConfig:
    dataset_base_root: Path
    dataset_root: Path
    report_csv_path: Path


@dataclass
class ProgressBar:
    total: int
    width: int = 30
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


def parse_args() -> tuple[argparse.Namespace, AnalyzeConfig]:
    parser = argparse.ArgumentParser(description="统计无痛胃镜检查报告对应图像的有效区域稳定性")
    parser.add_argument("--config", type=Path, default=CONFIG_PATH, help="路径配置文件，默认 configs/path.yaml")
    parser.add_argument("--report-csv", type=Path, default=None, help="可选：覆盖配置中的 valid_dicts_report_csv")
    parser.add_argument(
        "--report-title",
        type=str,
        default="无痛胃镜检查报告",
        help="要筛选的 reportTitle，默认仅处理‘无痛胃镜检查报告’",
    )
    parser.add_argument(
        "--input-dir",
        type=Path,
        default=None,
        help="可选：直接指定图像目录（指定后将跳过 reportTitle 过滤逻辑）",
    )
    parser.add_argument("--output-csv", type=Path, default=Path("effective_region_per_image.csv"), help="逐图像明细 CSV")
    parser.add_argument("--output-md", type=Path, default=Path("effective_region_summary.md"), help="汇总统计 Markdown")
    parser.add_argument("--black-threshold", type=int, default=20, help="背景黑区阈值（0~255）")
    parser.add_argument("--morph-kernel", type=int, default=5, help="形态学核尺寸")
    parser.add_argument("--min-area-ratio", type=float, default=0.05, help="最大连通域面积占比下限")
    parser.add_argument(
        "--centroid-std-ratio-threshold",
        type=float,
        default=0.03,
        help="质心稳定阈值（归一化后 x/y 标准差最大值阈值）",
    )
    parser.add_argument("--zscore-threshold", type=float, default=3.0, help="离群检测 z-score 阈值")

    args = parser.parse_args()
    cfg = AnalyzeConfig(
        output_csv=args.output_csv,
        output_md=args.output_md,
        black_threshold=args.black_threshold,
        morph_kernel=max(1, int(args.morph_kernel)),
        min_area_ratio=float(args.min_area_ratio),
        centroid_std_ratio_threshold=float(args.centroid_std_ratio_threshold),
        zscore_threshold=float(args.zscore_threshold),
    )
    return args, cfg


def load_yaml_config(config_path: Path) -> dict[str, Any]:
    if not config_path.exists():
        raise FileNotFoundError(f"路径配置文件不存在：{config_path}")

    if yaml is not None:
        payload = yaml.safe_load(config_path.read_text(encoding="utf-8"))
        if not isinstance(payload, dict):
            raise ValueError(f"路径配置文件格式错误：{config_path}")
        return payload

    lines = config_path.read_text(encoding="utf-8").splitlines()
    payload: dict[str, Any] = {}
    current_section: str | None = None
    for raw_line in lines:
        if not raw_line.strip() or raw_line.lstrip().startswith("#"):
            continue

        indent = len(raw_line) - len(raw_line.lstrip(" "))
        line = raw_line.strip()
        if line.endswith(":"):
            current_section = line[:-1]
            payload[current_section] = {}
            continue

        key, sep, value = line.partition(":")
        if not sep:
            raise ValueError(f"无法解析路径配置行：{raw_line}")

        cleaned_value = value.strip().strip('"').strip("'")
        if indent == 0:
            payload[key.strip()] = cleaned_value
            current_section = None
            continue

        if current_section is None:
            raise ValueError(f"发现未归属分组的缩进行：{raw_line}")
        payload[current_section][key.strip()] = cleaned_value
    return payload


def build_path_config(config_path: Path, report_csv_override: Path | None) -> PathConfig:
    payload = load_yaml_config(config_path.expanduser())
    paths_payload = payload.get("paths")
    if not isinstance(paths_payload, dict):
        raise ValueError("path.yaml 必须包含 paths 分组")

    config_dir = config_path.expanduser().resolve().parent

    def resolve_path(raw_path: str) -> Path:
        path = Path(raw_path).expanduser()
        if path.is_absolute():
            return path
        return (config_dir.parent / path).resolve()

    dataset_base_root = resolve_path(str(paths_payload["dataset_base_root"]))
    dataset_root = resolve_path(str(paths_payload["dataset_root"]))
    report_csv_config = paths_payload.get("valid_dicts_report_csv")
    report_csv_path = (
        report_csv_override.expanduser().resolve()
        if report_csv_override is not None
        else resolve_path(str(report_csv_config)) if report_csv_config else (dataset_base_root / "valid_dicts_report.csv").resolve()
    )

    return PathConfig(
        dataset_base_root=dataset_base_root,
        dataset_root=dataset_root,
        report_csv_path=report_csv_path,
    )


def resolve_exam_dir(raw_exam_dir: str, path_cfg: PathConfig) -> Path:
    exam_dir = Path(raw_exam_dir).expanduser()
    if exam_dir.is_absolute():
        return exam_dir

    candidate1 = (path_cfg.dataset_base_root / exam_dir).resolve()
    if candidate1.exists():
        return candidate1

    candidate2 = (path_cfg.dataset_root / exam_dir).resolve()
    if candidate2.exists():
        return candidate2

    return candidate1


def collect_image_paths_from_report(path_cfg: PathConfig, report_title: str) -> list[Path]:
    if not path_cfg.report_csv_path.is_file():
        raise FileNotFoundError(f"未找到报告汇总文件：{path_cfg.report_csv_path}")

    with path_cfg.report_csv_path.open("r", encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        rows = list(reader)
        fieldnames = reader.fieldnames or []

    required = {"exam_dir", "reportTitle"}
    missing = required - set(fieldnames)
    if missing:
        missing_fields = "、".join(sorted(missing))
        raise KeyError(f"{path_cfg.report_csv_path} 中缺少必需字段：{missing_fields}")

    image_paths: list[Path] = []
    matched_rows = 0
    exam_dir_missing = 0

    progress = ProgressBar(total=len(rows))
    for row in rows:
        title = str(row.get("reportTitle", "")).strip()
        if title != report_title:
            progress.update(1)
            continue

        matched_rows += 1
        exam_dir_raw = str(row.get("exam_dir", "")).strip()
        exam_dir = resolve_exam_dir(exam_dir_raw, path_cfg)
        img_dir = exam_dir / "img"
        if not img_dir.is_dir():
            exam_dir_missing += 1
            progress.update(1)
            continue

        for path in sorted(img_dir.iterdir()):
            if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS:
                image_paths.append(path)
        progress.update(1)

    progress.close()
    print(f"reportTitle=‘{report_title}’ 的检查记录数: {matched_rows}")
    if exam_dir_missing > 0:
        print(f"警告：有 {exam_dir_missing} 条记录未找到 img 目录，已跳过。")

    return image_paths


def collect_image_paths_from_input_dir(input_dir: Path) -> list[Path]:
    if not input_dir.exists() or not input_dir.is_dir():
        raise FileNotFoundError(f"输入目录不存在或不可读: {input_dir}")

    image_paths = sorted(
        path
        for path in input_dir.rglob("*")
        if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
    )
    print("已启用 --input-dir，跳过 reportTitle 过滤逻辑。")
    return image_paths


def safe_zscore(values: pd.Series) -> pd.Series:
    std = float(values.std(ddof=0))
    if std < 1e-12:
        return pd.Series(np.zeros(len(values)), index=values.index, dtype=float)
    mean = float(values.mean())
    return (values - mean) / std


def analyze_single_image(image_path: Path, cfg: AnalyzeConfig) -> dict[str, object]:
    image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
    if image is None:
        return {
            "image_path": str(image_path.resolve()),
            "width": np.nan,
            "height": np.nan,
            "x_min": np.nan,
            "y_min": np.nan,
            "x_max": np.nan,
            "y_max": np.nan,
            "bbox_width": np.nan,
            "bbox_height": np.nan,
            "component_area": 0,
            "image_area": 0,
            "area_ratio": 0.0,
            "centroid_x": np.nan,
            "centroid_y": np.nan,
            "is_abnormal": True,
            "abnormal_reason": "图像读取失败",
        }

    height, width = image.shape[:2]
    image_area = int(width * height)

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    bg_mask = (gray <= cfg.black_threshold).astype(np.uint8) * 255
    non_bg_mask = cv2.bitwise_not(bg_mask)

    k = cfg.morph_kernel
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))
    non_bg_mask = cv2.morphologyEx(non_bg_mask, cv2.MORPH_OPEN, kernel)
    non_bg_mask = cv2.morphologyEx(non_bg_mask, cv2.MORPH_CLOSE, kernel)

    num_labels, _labels, stats, centroids = cv2.connectedComponentsWithStats(non_bg_mask, connectivity=8)

    if num_labels <= 1:
        return {
            "image_path": str(image_path.resolve()),
            "width": width,
            "height": height,
            "x_min": np.nan,
            "y_min": np.nan,
            "x_max": np.nan,
            "y_max": np.nan,
            "bbox_width": np.nan,
            "bbox_height": np.nan,
            "component_area": 0,
            "image_area": image_area,
            "area_ratio": 0.0,
            "centroid_x": np.nan,
            "centroid_y": np.nan,
            "is_abnormal": True,
            "abnormal_reason": "未检测到有效连通域",
        }

    areas = stats[1:, cv2.CC_STAT_AREA]
    largest_idx = int(np.argmax(areas)) + 1

    x = int(stats[largest_idx, cv2.CC_STAT_LEFT])
    y = int(stats[largest_idx, cv2.CC_STAT_TOP])
    w = int(stats[largest_idx, cv2.CC_STAT_WIDTH])
    h = int(stats[largest_idx, cv2.CC_STAT_HEIGHT])
    component_area = int(stats[largest_idx, cv2.CC_STAT_AREA])
    area_ratio = component_area / image_area if image_area > 0 else 0.0
    centroid_x = float(centroids[largest_idx, 0])
    centroid_y = float(centroids[largest_idx, 1])

    reasons: list[str] = []
    if area_ratio < cfg.min_area_ratio:
        reasons.append(f"最大连通域面积占比过小(<{cfg.min_area_ratio:.3f})")

    return {
        "image_path": str(image_path.resolve()),
        "width": width,
        "height": height,
        "x_min": x,
        "y_min": y,
        "x_max": x + w - 1,
        "y_max": y + h - 1,
        "bbox_width": w,
        "bbox_height": h,
        "component_area": component_area,
        "image_area": image_area,
        "area_ratio": area_ratio,
        "centroid_x": centroid_x,
        "centroid_y": centroid_y,
        "is_abnormal": len(reasons) > 0,
        "abnormal_reason": "；".join(reasons),
    }


def aggregate_stats(df: pd.DataFrame) -> dict[str, dict[str, float]]:
    metrics = ["x_min", "y_min", "x_max", "y_max", "area_ratio", "centroid_x", "centroid_y"]
    stats_summary: dict[str, dict[str, float]] = {}
    for m in metrics:
        vals = df[m].dropna()
        stats_summary[m] = {
            "mean": float(vals.mean()) if len(vals) else float("nan"),
            "std": float(vals.std(ddof=0)) if len(vals) else float("nan"),
        }
    return stats_summary


def apply_outlier_rules(df: pd.DataFrame, cfg: AnalyzeConfig) -> pd.DataFrame:
    df = df.copy()
    valid_mask = (~df["x_min"].isna()) & (~df["centroid_x"].isna())
    valid_df = df.loc[valid_mask]

    if len(valid_df) < 2:
        return df

    columns = ["x_min", "y_min", "x_max", "y_max", "area_ratio", "centroid_x", "centroid_y"]
    zscores = {col: safe_zscore(valid_df[col]) for col in columns}

    for idx in valid_df.index:
        reasons = []
        for col in ["x_min", "y_min", "x_max", "y_max"]:
            if abs(zscores[col].loc[idx]) > cfg.zscore_threshold:
                reasons.append(f"{col}离群(|z|>{cfg.zscore_threshold:.1f})")
        if abs(zscores["centroid_x"].loc[idx]) > cfg.zscore_threshold or abs(zscores["centroid_y"].loc[idx]) > cfg.zscore_threshold:
            reasons.append(f"质心离群(|z|>{cfg.zscore_threshold:.1f})")
        if abs(zscores["area_ratio"].loc[idx]) > cfg.zscore_threshold:
            reasons.append(f"面积占比离群(|z|>{cfg.zscore_threshold:.1f})")

        if reasons:
            old = str(df.at[idx, "abnormal_reason"]).strip()
            merged = "；".join([r for r in ([old] + reasons) if r and r != "nan"])
            df.at[idx, "abnormal_reason"] = merged
            df.at[idx, "is_abnormal"] = True

    return df


def centroid_stability(df: pd.DataFrame, cfg: AnalyzeConfig) -> tuple[bool, float, float]:
    valid = df.dropna(subset=["centroid_x", "centroid_y", "width", "height"])
    if len(valid) == 0:
        return False, float("nan"), float("nan")

    cx_norm_std = float((valid["centroid_x"] / valid["width"]).std(ddof=0))
    cy_norm_std = float((valid["centroid_y"] / valid["height"]).std(ddof=0))
    is_stable = max(cx_norm_std, cy_norm_std) <= cfg.centroid_std_ratio_threshold
    return is_stable, cx_norm_std, cy_norm_std


def write_summary_markdown(
    output_md: Path,
    source_label: str,
    cfg: AnalyzeConfig,
    df: pd.DataFrame,
    stats_summary: dict[str, dict[str, float]],
    stable: bool,
    cx_norm_std: float,
    cy_norm_std: float,
) -> None:
    abnormal_df = df[df["is_abnormal"] == True]
    abnormal_paths = abnormal_df["image_path"].tolist()

    lines = [
        "# 内镜图像有效区域稳定性统计报告",
        "",
        "## 数据来源",
        f"- {source_label}",
        "",
        "## 方法说明",
        "- 目标：在不训练模型的前提下，基于图像预处理 + 连通域分析提取每张图像的有效区域。",
        "- 流程：背景黑区识别 → 非背景掩膜构建 → 形态学增强 → 连通域分析 → 最大连通域作为有效区域。",
        "",
        "## 背景黑区识别策略",
        f"- 灰度阈值法：gray <= {cfg.black_threshold} 判定为背景黑区。",
        "- 为减小压缩噪声影响，非背景掩膜执行开运算 + 闭运算。",
        f"- 形态学核大小：{cfg.morph_kernel}x{cfg.morph_kernel}。",
        "",
        "## 连通域分析策略",
        "- 在非背景二值图上执行 8 邻域连通域分析。",
        "- 取面积最大的连通域作为有效图像区域。",
        f"- 若最大连通域面积占比 < {cfg.min_area_ratio:.3f}，记为异常。",
        "",
        "## 质心稳定性判定规则",
        "- 对每张图像计算最大连通域质心 (centroid_x, centroid_y)。",
        "- 计算归一化标准差：std(centroid_x/width)、std(centroid_y/height)。",
        f"- 若 max(上述两者) <= {cfg.centroid_std_ratio_threshold:.4f}，判定“质心稳定”；否则“不稳定”。",
        f"- 本批次判定结果：**{'稳定' if stable else '不稳定'}**。",
        f"  - std(centroid_x/width) = {cx_norm_std:.6f}",
        f"  - std(centroid_y/height) = {cy_norm_std:.6f}",
        "",
        "## 坐标与面积统计（均值 ± 标准差）",
    ]

    for key in ["x_min", "y_min", "x_max", "y_max", "area_ratio", "centroid_x", "centroid_y"]:
        mean = stats_summary[key]["mean"]
        std = stats_summary[key]["std"]
        lines.append(f"- {key}: {mean:.6f} ± {std:.6f}")

    lines.extend(
        [
            "",
            "## 异常图像统计",
            f"- 异常图像数量：{len(abnormal_df)} / {len(df)}",
            f"- 离群检测 z-score 阈值：{cfg.zscore_threshold:.2f}",
            "",
            "## 异常图像路径列表",
        ]
    )

    if abnormal_paths:
        lines.extend([f"- {p}" for p in abnormal_paths])
    else:
        lines.append("- 无")

    output_md.write_text("\n".join(lines) + "\n", encoding="utf-8")


def build_empty_df() -> pd.DataFrame:
    return pd.DataFrame(
        columns=[
            "image_path",
            "width",
            "height",
            "x_min",
            "y_min",
            "x_max",
            "y_max",
            "bbox_width",
            "bbox_height",
            "component_area",
            "image_area",
            "area_ratio",
            "centroid_x",
            "centroid_y",
            "is_abnormal",
            "abnormal_reason",
        ]
    )


def main() -> None:
    args, cfg = parse_args()

    if args.input_dir is not None:
        image_paths = collect_image_paths_from_input_dir(args.input_dir.expanduser().resolve())
        source_label = f"输入目录：{args.input_dir.expanduser().resolve()}"
    else:
        path_cfg = build_path_config(args.config, args.report_csv)
        image_paths = collect_image_paths_from_report(path_cfg, args.report_title)
        source_label = f"报告筛选：{args.report_title}（来源 CSV：{path_cfg.report_csv_path}）"

    total = len(image_paths)
    print(f"总图像数: {total}")

    rows: list[dict[str, object]] = []
    progress = ProgressBar(total=total)
    for image_path in image_paths:
        rows.append(analyze_single_image(image_path, cfg))
        progress.update(1)
    progress.close()

    df = pd.DataFrame(rows) if rows else build_empty_df()

    if len(df) > 0:
        df = apply_outlier_rules(df, cfg)

    stats_summary = aggregate_stats(df) if len(df) > 0 else {
        k: {"mean": float("nan"), "std": float("nan")}
        for k in ["x_min", "y_min", "x_max", "y_max", "area_ratio", "centroid_x", "centroid_y"]
    }
    stable, cx_norm_std, cy_norm_std = centroid_stability(df, cfg) if len(df) > 0 else (False, float("nan"), float("nan"))

    cfg.output_csv.parent.mkdir(parents=True, exist_ok=True)
    cfg.output_md.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(cfg.output_csv, index=False, encoding="utf-8-sig")
    write_summary_markdown(cfg.output_md, source_label, cfg, df, stats_summary, stable, cx_norm_std, cy_norm_std)

    abnormal_df = df[df["is_abnormal"] == True] if len(df) > 0 else pd.DataFrame()
    success_count = int((~df["x_min"].isna()).sum()) if len(df) > 0 else 0

    print(f"成功检测数: {success_count}")
    print(f"异常图像数: {len(abnormal_df)}")
    print("异常图像路径:")
    if len(abnormal_df) == 0:
        print("- 无")
    else:
        for p in abnormal_df["image_path"].tolist():
            print(f"- {p}")

    print(f"CSV 已生成: {cfg.output_csv.resolve()}")
    print(f"Markdown 已生成: {cfg.output_md.resolve()}")


if __name__ == "__main__":
    main()
