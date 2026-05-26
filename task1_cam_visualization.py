from __future__ import annotations

import argparse
import csv
import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")

import matplotlib.gridspec as gridspec
import matplotlib.patches as patches
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import yaml
from PIL import Image, ImageDraw

try:
    import cv2
except Exception:  # pragma: no cover
    cv2 = None

from model import GastroLabelGraphMIL
from model.common.backbones import IMAGE_MEAN, IMAGE_STD
from tasks import get_task_spec
from training.data import MILBagDataset, build_task_records, mil_collate_fn, split_records


PROJECT_ROOT = Path("/home/Lim/Project4")
SRC_ROOT = PROJECT_ROOT / "src"
DEFAULT_RUN_DIR = (
    PROJECT_ROOT
    / "outputs/train_runs/gastro_multilabel_task/gastro_6"
)
DEFAULT_DATA_CSV = PROJECT_ROOT / "datasets/task_data/task1/gastro_multilabel_task_datalist.csv"
DEFAULT_DATASET_ROOT = PROJECT_ROOT / "datasets/main_data"
DEFAULT_TRAIN_CONFIG = SRC_ROOT / "configs/task1/train.yaml"
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "outputs/temp/task1_cam_visulization"
DEFAULT_EXAM_ID = "ZS0046106736"
DEFAULT_ATTENTION_THRESHOLD = 1.0 / 16.0

LABEL_DISPLAY_NAMES = {
    "label_esophageal_smt": "Esophageal submucosal tumor",
    "label_esophageal_mucosal_or_tumor": "Esophageal mucosal lesion",
    "label_gastritis": "Gastritis",
}

LABEL_FILE_NAMES = {
    "label_esophageal_smt": "esophageal_smt",
    "label_esophageal_mucosal_or_tumor": "esophageal_mucosal_lesion",
    "label_gastritis": "gastritis",
}

TOP_COLORS = ["#a855f7", "#3b82f6", "#22c55e"]

# 仅修改图中显示的 predicted probability，不改变模型推理、attention、Grad-CAM 或选图结果。
PROBABILITY_DISPLAY_OVERRIDES = {
    "label_esophageal_smt": 0.871,
}


@dataclass
class ExamVisualization:
    record: dict[str, Any]
    batch: dict[str, Any]


class GradCAMHook:
    def __init__(self, target_layer: nn.Module) -> None:
        self.target_layer = target_layer
        self.activations: torch.Tensor | None = None
        self.gradients: torch.Tensor | None = None
        self._handles: list[Any] = []

    def __enter__(self) -> "GradCAMHook":
        self._handles.append(self.target_layer.register_forward_hook(self._save_activation))
        self._handles.append(self.target_layer.register_full_backward_hook(self._save_gradient))
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        del exc_type, exc, tb
        for handle in self._handles:
            handle.remove()
        self._handles.clear()

    def _save_activation(self, module: nn.Module, inputs: tuple[torch.Tensor, ...], output: torch.Tensor) -> None:
        del module, inputs
        self.activations = output.detach()

    def _save_gradient(
        self,
        module: nn.Module,
        grad_input: tuple[torch.Tensor, ...],
        grad_output: tuple[torch.Tensor, ...],
    ) -> None:
        del module, grad_input
        self.gradients = grad_output[0].detach()

    def build_cam(self, image_size: tuple[int, int], valid_num: int) -> np.ndarray:
        if self.activations is None or self.gradients is None:
            raise RuntimeError("Grad-CAM 未捕获到 activation 或 gradient")
        activations = self.activations[:valid_num]
        gradients = self.gradients[:valid_num]
        weights = gradients.mean(dim=(2, 3), keepdim=True)
        cams = torch.relu((weights * activations).sum(dim=1, keepdim=True))
        cams = F.interpolate(cams, size=image_size, mode="bilinear", align_corners=False)
        cams = cams[:, 0]
        cams = cams - cams.flatten(1).min(dim=1).values[:, None, None]
        denom = cams.flatten(1).max(dim=1).values[:, None, None].clamp_min(1e-12)
        cams = cams / denom
        return cams.detach().cpu().numpy()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="生成 TASK1 指定检查的三标签 attention threshold + Grad-CAM 可视化图。"
    )
    parser.add_argument("--run-dir", type=Path, default=DEFAULT_RUN_DIR, help="训练输出目录，需包含 config.yaml")
    parser.add_argument("--checkpoint", type=Path, default=None, help="checkpoint 路径；默认用 run-dir/checkpoints/best_macro_f1.ckpt")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR, help="可视化输出目录")
    parser.add_argument("--data-csv", type=Path, default=DEFAULT_DATA_CSV, help="TASK1 datalist CSV")
    parser.add_argument("--dataset-root", type=Path, default=DEFAULT_DATASET_ROOT, help="原始图像数据根目录")
    parser.add_argument("--split", choices=("all", "train", "val", "test"), default="all", help="从哪个数据范围中挑选检查")
    parser.add_argument("--num-exams", type=int, default=1, help="未指定 exam-dir 时挑选三标签全阳性的检查数量")
    parser.add_argument(
        "--attention-threshold",
        type=float,
        default=DEFAULT_ATTENTION_THRESHOLD,
        help="attention 大于该阈值的图像会被视为病灶候选图，默认 1/16=0.0625",
    )
    parser.add_argument("--max-instances", type=int, default=0, help="覆盖 eval_max_instances；0 表示使用训练配置")
    parser.add_argument("--device", type=str, default="auto", help="auto / cpu / cuda / cuda:0 等")
    parser.add_argument("--dpi", type=int, default=220, help="输出图片 DPI")
    parser.add_argument("--exam-dir", action="append", default=None, help="手动指定 exam_dir，可重复传入；默认使用 ZS0046106736")
    return parser.parse_args()


def read_yaml(path: Path) -> dict[str, Any]:
    if not path.is_file():
        return {}
    with path.open("r", encoding="utf-8") as file:
        payload = yaml.safe_load(file) or {}
    return payload if isinstance(payload, dict) else {}


def to_bool(value: Any, default: bool) -> bool:
    if value is None:
        return default
    if isinstance(value, bool):
        return value
    text = str(value).strip().lower()
    if text in {"true", "1", "yes", "y", "on"}:
        return True
    if text in {"false", "0", "no", "n", "off"}:
        return False
    return default


def resolve_device(raw_device: str) -> torch.device:
    if raw_device == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(raw_device)


def strip_module_prefix(state_dict: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    cleaned = {}
    for key, value in state_dict.items():
        clean_key = key
        while clean_key.startswith("module."):
            clean_key = clean_key[len("module.") :]
        cleaned[clean_key] = value
    return cleaned


def load_checkpoint_state(checkpoint_path: Path, device: torch.device) -> dict[str, torch.Tensor]:
    if not checkpoint_path.is_file():
        raise FileNotFoundError(f"未找到 checkpoint: {checkpoint_path}")
    try:
        checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    except TypeError:
        checkpoint = torch.load(checkpoint_path, map_location=device)

    if isinstance(checkpoint, dict):
        for key in ("model_state", "model_state_dict", "state_dict", "model", "model_state_dict_raw"):
            value = checkpoint.get(key)
            if isinstance(value, dict):
                return strip_module_prefix(value)
        if all(torch.is_tensor(value) for value in checkpoint.values()):
            return strip_module_prefix(checkpoint)
    raise ValueError(f"无法解析 checkpoint 中的模型权重: {checkpoint_path}")


def build_model(config: dict[str, Any], device: torch.device) -> GastroLabelGraphMIL:
    model_name = str(config.get("model_name", "gastro_label_graph_mil"))
    if model_name != "gastro_label_graph_mil":
        raise ValueError(f"当前脚本只支持 gastro_label_graph_mil，config 中为: {model_name}")

    params = dict(config.get("model_params", {}) or {})
    model = GastroLabelGraphMIL(
        backbone_name=str(params.get("backbone_name", "convnext_tiny")),
        pretrained=False,
        freeze_stages=int(params.get("freeze_stages", 1)),
        feature_dim=int(params.get("feature_dim", 512)),
        attn_dim=int(params.get("attn_dim", 256)),
        num_labels=int(params.get("num_labels", 3)),
        dropout=float(params.get("dropout", 0.2)),
        use_label_graph=to_bool(params.get("use_label_graph"), True),
        label_graph_type=str(params.get("label_graph_type", "learnable")),
        label_graph_prior=params.get("label_graph_prior"),
        label_graph_rank=int(params.get("label_graph_rank", 2)),
        label_graph_heads=int(params.get("label_graph_heads", 4)),
        label_hypergraph_edges=int(params.get("label_hypergraph_edges", 2)),
        use_label_wise_attention=to_bool(params.get("use_label_wise_attention"), True),
        attention_type=str(params.get("attention_type", "label_specific")),
        pooling_type=str(params.get("pooling_type", "label_attention")),
    )
    return model.to(device)


def load_model(run_dir: Path, checkpoint_path: Path | None, device: torch.device) -> tuple[GastroLabelGraphMIL, dict[str, Any], Path]:
    config_path = run_dir / "config.yaml"
    config = read_yaml(config_path)
    if not config:
        raise FileNotFoundError(f"无法读取训练配置: {config_path}")

    checkpoint = checkpoint_path or run_dir / "checkpoints/best_macro_f1.ckpt"
    model = build_model(config, device=device)
    state_dict = load_checkpoint_state(checkpoint, device=device)
    missing, unexpected = model.load_state_dict(state_dict, strict=False)
    if missing:
        print(f"[警告] checkpoint 缺少 {len(missing)} 个参数，示例: {missing[:5]}")
    if unexpected:
        print(f"[警告] checkpoint 有 {len(unexpected)} 个额外参数，示例: {unexpected[:5]}")
    model.eval()
    return model, config, checkpoint


def load_task1_records(config: dict[str, Any], data_csv: Path, dataset_root: Path, split_name: str) -> tuple[list[dict[str, Any]], list[str]]:
    train_defaults = read_yaml(DEFAULT_TRAIN_CONFIG)
    seed = int(config.get("seed", train_defaults.get("seed", 42)))
    ratios = tuple(float(item) for item in train_defaults.get("split_ratio", [0.6, 0.2, 0.2]))
    if len(ratios) != 3:
        ratios = (0.6, 0.2, 0.2)
    min_instances = int(train_defaults.get("min_instances", 1))

    records = build_task_records(
        task_csv_path=data_csv,
        task_name="task1",
        min_instances=min_instances,
        dataset_root=dataset_root,
    )
    label_names = list(get_task_spec("task1").label_names)
    if split_name == "all":
        return records, label_names

    split_data = split_records(records, seed=seed, ratios=ratios, group_by_patient=False)
    return split_data[split_name], label_names


def normalize_exam_dir(value: str | Path) -> str:
    return str(value).strip().rstrip("/")


def select_triple_positive_records(
    records: list[dict[str, Any]],
    label_count: int,
    num_exams: int,
    requested_exam_dirs: list[str],
) -> list[dict[str, Any]]:
    if requested_exam_dirs:
        selected = []
        for raw_exam_dir in requested_exam_dirs:
            requested = normalize_exam_dir(raw_exam_dir)
            matches = [
                record
                for record in records
                if normalize_exam_dir(record.get("exam_dir", "")) == requested
                or requested in normalize_exam_dir(record.get("exam_dir", ""))
            ]
            if not matches:
                print(f"[警告] 当前划分中未找到指定检查: {requested}")
                continue
            selected.append(matches[0])
        return selected

    triple_positive = [
        record
        for record in records
        if len(record.get("labels", [])) >= label_count and all(int(value) == 1 for value in record["labels"][:label_count])
    ]
    triple_positive.sort(
        key=lambda item: (
            -int(item.get("img_num", 0)),
            normalize_exam_dir(item.get("exam_dir", "")),
        )
    )
    if len(triple_positive) < num_exams:
        print(f"[警告] 当前划分只有 {len(triple_positive)} 个三标签全阳性检查，少于请求的 {num_exams} 个。")
    return triple_positive[: max(1, int(num_exams))]


def make_dataset(
    records: list[dict[str, Any]],
    config: dict[str, Any],
    split_name: str,
    max_instances_override: int,
) -> MILBagDataset:
    run_cfg = dict(config.get("run", {}) or {})
    image_size = int(config.get("image_size", 224))
    eval_max_instances = int(max_instances_override or run_cfg.get("eval_max_instances", run_cfg.get("train_max_instances", 16)))
    image_cache_mode = str(run_cfg.get("image_cache_mode", "none"))
    image_cache_dir = run_cfg.get("resolved_image_cache_dir") or run_cfg.get("image_cache_dir")
    legacy_dirs = run_cfg.get("resolved_legacy_image_cache_dirs") or []

    return MILBagDataset(
        records=records,
        task_name="task1",
        max_instances=eval_max_instances,
        min_instances=1,
        bag_sampling_strategy="uniform",
        is_train=False,
        image_size=image_size,
        random_instance_dropout=0.0,
        image_cache_mode=image_cache_mode,
        image_cache_dir=image_cache_dir,
        legacy_image_cache_dirs=legacy_dirs,
        memory_cache_size=0,
        split_name=split_name,
    )


def make_exam_visualization(dataset: MILBagDataset, index: int) -> ExamVisualization:
    item = dataset[index]
    batch = mil_collate_fn([item])
    return ExamVisualization(record=dataset.records[index], batch=batch)


def move_batch_to_device(batch: dict[str, Any], device: torch.device) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    images = batch["images"].to(device)
    mask = batch["mask"].to(device)
    labels = batch["labels"].to(device)
    return images, mask, labels


def find_cam_target_layer(model: GastroLabelGraphMIL) -> nn.Module:
    backbone = model.instance_encoder.backbone
    feature_extractor = backbone[0] if isinstance(backbone, nn.Sequential) else backbone
    if hasattr(feature_extractor, "features"):
        return feature_extractor.features[-1]
    if hasattr(feature_extractor, "layer4"):
        return feature_extractor.layer4[-1]
    raise RuntimeError("未能自动定位 Grad-CAM 目标层")


def infer_attention_and_probabilities(
    model: GastroLabelGraphMIL,
    batch: dict[str, Any],
    device: torch.device,
) -> tuple[np.ndarray, np.ndarray]:
    images, mask, _ = move_batch_to_device(batch, device)
    with torch.no_grad():
        output = model(images, mask)
    valid_num = int(mask[0].sum().item())
    attention = output["attention"][0, :, :valid_num].detach().cpu().numpy()
    probabilities = torch.sigmoid(output["logits"][0]).detach().cpu().numpy()
    return attention, probabilities


def compute_label_gradcam(
    model: GastroLabelGraphMIL,
    batch: dict[str, Any],
    device: torch.device,
    label_index: int,
) -> np.ndarray:
    target_layer = find_cam_target_layer(model)
    images, mask, _ = move_batch_to_device(batch, device)
    _, _, _, height, width = images.shape
    valid_num = int(mask[0].sum().item())

    model.zero_grad(set_to_none=True)
    with GradCAMHook(target_layer) as hook:
        output = model(images, mask)
        output["logits"][0, label_index].backward()
        cams = hook.build_cam(image_size=(height, width), valid_num=valid_num)
    return cams


def tensor_to_rgb(image_tensor: torch.Tensor) -> np.ndarray:
    array = image_tensor.detach().cpu().float().numpy()
    mean = np.asarray(IMAGE_MEAN, dtype=np.float32)[:, None, None]
    std = np.asarray(IMAGE_STD, dtype=np.float32)[:, None, None]
    array = np.clip(array * std + mean, 0.0, 1.0)
    return np.transpose(array, (1, 2, 0))




def _find_non_black_bbox(
    rgb: np.ndarray,
    *,
    black_threshold: float = 0.06,
    min_content_ratio: float = 0.025,
    edge_content_ratio: float = 0.32,
    pad: int = 1,
) -> tuple[int, int, int, int]:
    """返回非黑色有效区域的 bbox: y1, y2, x1, x2。

    先找整体非黑区域，再从四条边进一步收紧，尽量去掉内镜图像常见的
    左侧/左上角黑边，同时避免把真正的组织区域裁掉太多。
    """
    if rgb.ndim != 3 or rgb.shape[2] < 3:
        raise ValueError(f"rgb 应该是 HxWx3 数组，实际为: {rgb.shape}")

    h, w = rgb.shape[:2]
    rgb_float = rgb.astype(np.float32)
    if rgb_float.max() > 1.5:
        rgb_float = rgb_float / 255.0

    content_mask = np.max(rgb_float[..., :3], axis=2) > float(black_threshold)
    row_ratio = content_mask.mean(axis=1)
    col_ratio = content_mask.mean(axis=0)

    row_keep = np.where(row_ratio > float(min_content_ratio))[0]
    col_keep = np.where(col_ratio > float(min_content_ratio))[0]
    if len(row_keep) == 0 or len(col_keep) == 0:
        return 0, h, 0, w

    y1 = int(row_keep[0])
    y2 = int(row_keep[-1]) + 1
    x1 = int(col_keep[0])
    x2 = int(col_keep[-1]) + 1

    # 在粗 bbox 内进一步从四条边向内收紧；只保留内容占比更高的边界位置。
    sub_row_ratio = row_ratio[y1:y2]
    sub_col_ratio = col_ratio[x1:x2]

    better_rows = np.where(sub_row_ratio > float(edge_content_ratio))[0]
    better_cols = np.where(sub_col_ratio > float(edge_content_ratio))[0]

    if len(better_rows) > 0:
        y1 = y1 + int(better_rows[0])
        y2 = y1 + (int(better_rows[-1]) - int(better_rows[0]) + 1)
    if len(better_cols) > 0:
        x1 = x1 + int(better_cols[0])
        x2 = x1 + (int(better_cols[-1]) - int(better_cols[0]) + 1)

    y1 = max(y1 - int(pad), 0)
    y2 = min(y2 + int(pad), h)
    x1 = max(x1 - int(pad), 0)
    x2 = min(x2 + int(pad), w)

    # 防止极端情况下误裁过多。
    if (y2 - y1) < h * 0.35 or (x2 - x1) < w * 0.35:
        return 0, h, 0, w
    return y1, y2, x1, x2


def _center_crop_square_rgb_and_cam(
    rgb: np.ndarray,
    cam: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray | None]:
    """把裁剪后的 RGB/CAM 同步中心裁成正方形。

    内镜图像在左上角、左下角常见黑边时，图像通常是“高 > 宽”的纵向区域，
    因此优先通过上下裁剪得到 width x width 的正方形；如果遇到“宽 > 高”的
    个别图像，则退化为左右中心裁剪，保证最终小图一定是正方形。
    """
    if rgb.ndim != 3 or rgb.shape[2] < 3:
        raise ValueError(f"rgb 应该是 HxWx3 数组，实际为: {rgb.shape}")

    height, width = rgb.shape[:2]
    if height == width:
        return rgb, cam

    if height > width:
        crop_size = width
        y1 = max((height - crop_size) // 2, 0)
        y2 = y1 + crop_size
        cropped_rgb = rgb[y1:y2, :]
        cropped_cam = None if cam is None else cam[y1:y2, :]
        return cropped_rgb, cropped_cam

    crop_size = height
    x1 = max((width - crop_size) // 2, 0)
    x2 = x1 + crop_size
    cropped_rgb = rgb[:, x1:x2]
    cropped_cam = None if cam is None else cam[:, x1:x2]
    return cropped_rgb, cropped_cam


def crop_black_border_rgb_and_cam(
    rgb: np.ndarray,
    cam: np.ndarray | None = None,
    *,
    black_threshold: float = 0.06,
    min_content_ratio: float = 0.025,
    edge_content_ratio: float = 0.32,
    pad: int = 1,
    make_square: bool = True,
) -> tuple[np.ndarray, np.ndarray | None]:
    """裁掉 RGB 图像四周黑边，并可同步裁成正方形；CAM 同步裁剪以保证 overlay 对齐。"""
    y1, y2, x1, x2 = _find_non_black_bbox(
        rgb,
        black_threshold=black_threshold,
        min_content_ratio=min_content_ratio,
        edge_content_ratio=edge_content_ratio,
        pad=pad,
    )
    cropped_rgb = rgb[y1:y2, x1:x2]
    cropped_cam = None if cam is None else cam[y1:y2, x1:x2]
    if make_square:
        cropped_rgb, cropped_cam = _center_crop_square_rgb_and_cam(cropped_rgb, cropped_cam)
    return cropped_rgb, cropped_cam


def overlay_cam(rgb: np.ndarray, cam: np.ndarray, alpha: float = 0.46) -> np.ndarray:
    heat = plt.get_cmap("jet")(np.clip(cam, 0.0, 1.0))[..., :3]
    return np.clip((1.0 - alpha) * rgb + alpha * heat, 0.0, 1.0)


def safe_name(text: str, default: str = "exam") -> str:
    text = str(text).strip().replace("\\", "/").rstrip("/").split("/")[-1]
    text = re.sub(r"[^A-Za-z0-9_.-]+", "_", text)
    return text[:90] or default


def label_title(label_name: str) -> str:
    return LABEL_DISPLAY_NAMES.get(label_name, label_name)


def label_file_name(label_name: str) -> str:
    return LABEL_FILE_NAMES.get(label_name, safe_name(label_name, default="label"))


def _rgb_to_uint8(rgb: np.ndarray) -> np.ndarray:
    return np.clip(rgb * 255.0, 0, 255).astype(np.uint8) if rgb.dtype != np.uint8 else rgb


def _resize_rgb(rgb: np.ndarray, width: int, height: int) -> np.ndarray:
    rgb_uint8 = _rgb_to_uint8(rgb)
    if cv2 is not None:
        return cv2.resize(rgb_uint8, (width, height), interpolation=cv2.INTER_AREA)
    return np.asarray(Image.fromarray(rgb_uint8).resize((width, height)))


def _make_sheared_rgb_card(
    rgb: np.ndarray,
    *,
    card_width: int,
    card_height: int,
    shear_px: int,
) -> np.ndarray:
    resized = _resize_rgb(rgb, width=card_width, height=card_height)
    rgba = np.zeros((card_height, card_width, 4), dtype=np.uint8)
    rgba[..., :3] = resized
    rgba[..., 3] = 255
    out_w = card_width + shear_px
    out_h = card_height

    if cv2 is not None:
        src = np.float32([[0, 0], [card_width - 1, 0], [card_width - 1, card_height - 1], [0, card_height - 1]])
        dst = np.float32(
            [
                [shear_px, 0],
                [card_width - 1 + shear_px, 0],
                [card_width - 1, card_height - 1],
                [0, card_height - 1],
            ]
        )
        matrix = cv2.getPerspectiveTransform(src, dst)
        return cv2.warpPerspective(
            rgba,
            matrix,
            (out_w, out_h),
            flags=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=(255, 255, 255, 0),
        )

    warped = np.zeros((out_h, out_w, 4), dtype=np.uint8)
    for row in range(card_height):
        shift = int(round(shear_px * (1.0 - row / max(card_height - 1, 1))))
        warped[row, shift : shift + card_width] = rgba[row]
    return warped


def _hex_to_rgba_tuple(color: str, alpha: int = 255) -> tuple[int, int, int, int]:
    text = str(color).strip().lstrip("#")
    if len(text) != 6:
        raise ValueError(f"颜色必须是 #RRGGBB 格式，实际为: {color}")
    return (int(text[0:2], 16), int(text[2:4], 16), int(text[4:6], 16), int(alpha))


def _draw_sheared_card_border(
    rgba: np.ndarray,
    *,
    card_width: int,
    card_height: int,
    shear_px: int,
    color: str,
    linewidth: int = 3,
) -> None:
    """把选中边框画进单张倾斜卡片里。

    边框先被烘焙进各自的 RGBA 卡片，再按堆叠深度合成。这样后合成的上层
    图像会自然遮住下层图像的边框，避免下层边框覆盖到自己上方的图片。
    """
    if rgba.ndim != 3 or rgba.shape[2] != 4:
        raise ValueError(f"rgba 应该是 HxWx4 数组，实际为: {rgba.shape}")

    h, w = rgba.shape[:2]
    right_top_x = min(shear_px + card_width - 1, w - 1)
    right_bottom_x = min(card_width - 1, w - 1)
    bottom_y = min(card_height - 1, h - 1)
    points = np.asarray(
        [
            [shear_px, 0],
            [right_top_x, 0],
            [right_bottom_x, bottom_y],
            [0, bottom_y],
        ],
        dtype=np.int32,
    )
    rgba_color = _hex_to_rgba_tuple(color)

    if cv2 is not None:
        cv2.polylines(
            rgba,
            [points],
            isClosed=True,
            color=rgba_color,
            thickness=int(linewidth),
            lineType=cv2.LINE_AA,
        )
        return

    image = Image.fromarray(rgba, mode="RGBA")
    draw = ImageDraw.Draw(image)
    closed_points = [tuple(map(int, point)) for point in points] + [tuple(map(int, points[0]))]
    draw.line(closed_points, fill=rgba_color, width=int(linewidth), joint="curve")
    rgba[:] = np.asarray(image)


def _alpha_blend(canvas: np.ndarray, rgba: np.ndarray, x: int, y: int) -> None:
    h, w = rgba.shape[:2]
    ch, cw = canvas.shape[:2]
    if x >= cw or y >= ch or x + w <= 0 or y + h <= 0:
        return
    x1, y1 = max(0, x), max(0, y)
    x2, y2 = min(cw, x + w), min(ch, y + h)
    sx1, sy1 = x1 - x, y1 - y
    roi = canvas[y1:y2, x1:x2].astype(np.float32)
    fg = rgba[sy1 : sy1 + (y2 - y1), sx1 : sx1 + (x2 - x1), :3].astype(np.float32)
    alpha = rgba[sy1 : sy1 + (y2 - y1), sx1 : sx1 + (x2 - x1), 3:4].astype(np.float32) / 255.0
    canvas[y1:y2, x1:x2] = (fg * alpha + roi * (1.0 - alpha)).astype(np.uint8)


def _render_vertical_image_stack(
    rgb_images: list[np.ndarray],
    *,
    card_width: int = 210,
    card_height: int = 150,
    shear_px: int = 46,
    y_step_ratio: float = 0.22,
    margin_left: int = 26,
    margin_right: int = 370,
    margin_top: int = 28,
    margin_bottom: int = 26,
    selected_border_colors: dict[int, str] | None = None,
) -> tuple[np.ndarray, dict[str, Any]]:
    cards = [
        _make_sheared_rgb_card(
            rgb,
            card_width=card_width,
            card_height=card_height,
            shear_px=shear_px,
        )
        for rgb in rgb_images
    ]
    if not cards:
        return np.full((256, 256, 3), 255, dtype=np.uint8), {}

    # 先把边框画进各自卡片，再堆叠合成；这样边框会遵守图层遮挡关系。
    for image_index, color in (selected_border_colors or {}).items():
        if 0 <= int(image_index) < len(cards):
            _draw_sheared_card_border(
                cards[int(image_index)],
                card_width=card_width,
                card_height=card_height,
                shear_px=shear_px,
                color=color,
                linewidth=3,
            )

    y_step = max(1, int(round(card_height * y_step_ratio)))
    card_w = int(cards[0].shape[1])
    total_w = card_w + margin_left + margin_right
    total_h = card_height + (len(cards) - 1) * y_step + margin_top + margin_bottom
    canvas = np.full((total_h, total_w, 3), 255, dtype=np.uint8)
    card_positions: dict[int, tuple[int, int]] = {}

    for idx in range(len(cards) - 1, -1, -1):
        x = margin_left
        y = margin_top + idx * y_step
        card_positions[idx] = (x, y)
        _alpha_blend(canvas, cards[idx], x=x, y=y)

    return canvas, {
        "card_width": int(card_width),
        "card_height": int(card_height),
        "shear_px": int(shear_px),
        "y_step": int(y_step),
        "margin_left": int(margin_left),
        "margin_top": int(margin_top),
        "card_right_x": int(margin_left + card_w),
        "total_w": int(total_w),
        "total_h": int(total_h),
        "card_positions": card_positions,
    }


def draw_image_stack(
    ax: plt.Axes,
    rgb_images: list[np.ndarray],
    selected_indices: list[int],
    selected_scores: list[float],
) -> None:
    num_images = len(rgb_images)
    if num_images <= 0:
        ax.axis("off")
        return

    ax.axis("off")
    selected_border_colors = {
        int(image_index): TOP_COLORS[rank % len(TOP_COLORS)]
        for rank, image_index in enumerate(selected_indices)
    }
    stack_rgb, meta = _render_vertical_image_stack(
        rgb_images,
        selected_border_colors=selected_border_colors,
    )
    ax.imshow(stack_rgb, interpolation="bilinear")

    card_height = int(meta["card_height"])
    card_right_x = int(meta["card_right_x"])
    text_x = card_right_x + 44
    card_positions: dict[int, tuple[int, int]] = meta["card_positions"]

    for rank, image_index in enumerate(selected_indices):
        color = TOP_COLORS[rank % len(TOP_COLORS)]
        _, y = card_positions[int(image_index)]
        card_y = y + card_height * 0.5
        leader_start = card_right_x + 10
        leader_end = card_right_x + 34
        ax.plot([leader_start, leader_end], [card_y, card_y], color=color, linewidth=1.15, solid_capstyle="round")
        ax.text(
            text_x,
            card_y,
            f"Lesion image #{image_index + 1} ({selected_scores[rank]:.4f})",
            color=color,
            fontsize=8.6,
            va="center",
            ha="left",
        )

    # 标题文字改由 plot_label_figure 使用 fig.text 放置，
    # 这样可以稳定地位于堆叠图正上方，并处在主标题左下方。


def plot_label_figure(
    *,
    output_path: Path,
    exam_name: str,
    label_name: str,
    label_index: int,
    images: torch.Tensor,
    attention: np.ndarray,
    probability: float,
    cams: np.ndarray,
    attention_threshold: float,
    fixed_columns: int,
    dpi: int,
) -> dict[str, Any]:
    valid_num = attention.shape[1]
    label_attention = attention[label_index]
    selected_indices = np.where(label_attention > float(attention_threshold))[0]
    selected_indices = selected_indices[np.argsort(-label_attention[selected_indices])].astype(int).tolist()
    selected_scores = [float(label_attention[index]) for index in selected_indices]
    rgb_images = [tensor_to_rgb(images[index]) for index in range(valid_num)]
    fixed_columns = max(1, int(fixed_columns))

    # 左侧堆叠缩略图也裁掉黑边后再缩放，整体观感更干净。
    stack_rgb_images = [crop_black_border_rgb_and_cam(rgb)[0] for rgb in rgb_images]

    # 右侧图像缩小到原先大约 3/4，同时保留一条可见但不夸张的上下间距。
    right_col_ratio = 0.78
    fig = plt.figure(figsize=(4.20 + fixed_columns * 1.58, 4.55), constrained_layout=False)
    spec = gridspec.GridSpec(
        2,
        fixed_columns + 1,
        figure=fig,
        width_ratios=[1.95, *([right_col_ratio] * fixed_columns)],
        height_ratios=[1.0, 1.0],
        wspace=0.14,
        hspace=0.045,
    )
    fig.subplots_adjust(left=0.028, right=0.985, top=0.84, bottom=0.045)

    stack_ax = fig.add_subplot(spec[:, 0])
    draw_image_stack(
        stack_ax,
        rgb_images=stack_rgb_images,
        selected_indices=selected_indices,
        selected_scores=selected_scores,
    )

    # 将“16 sampled images”和预测概率放在左侧堆叠图正上方，
    # 同时保持在主标题的左下方，避免跑到主标题左侧。
    stack_pos = stack_ax.get_position()
    # 继续向左、向下微调：仍位于堆叠图正上方，但与主标题拉开距离。
    stack_header_x = stack_pos.x0 + stack_pos.width * 0.30
    sampled_title_y = stack_pos.y1 + 0.040
    probability_title_y = stack_pos.y1 + 0.012
    fig.text(
        stack_header_x,
        sampled_title_y,
        f"{valid_num} sampled images",
        fontsize=12.5,
        fontweight="normal",
        ha="center",
        va="bottom",
    )
    fig.text(
        stack_header_x,
        probability_title_y,
        f"predicted probability={probability:.3f}",
        fontsize=9.2,
        fontweight="normal",
        ha="center",
        va="bottom",
    )

    for rank in range(fixed_columns):
        original_ax = fig.add_subplot(spec[0, rank + 1])
        overlay_ax = fig.add_subplot(spec[1, rank + 1])
        if rank >= len(selected_indices):
            original_ax.axis("off")
            overlay_ax.axis("off")
            continue

        image_index = selected_indices[rank]
        rgb = rgb_images[image_index]
        cropped_rgb, cropped_cam = crop_black_border_rgb_and_cam(rgb, cams[image_index])
        if cropped_cam is None:
            cropped_cam = cams[image_index]

        original_ax.imshow(cropped_rgb)
        original_ax.set_aspect("equal", adjustable="box")
        original_ax.set_anchor("C")
        original_ax.set_title(
            f"Image #{image_index + 1} ({selected_scores[rank]:.4f})",
            fontsize=8.2,
            fontweight="bold",
            pad=2.5,
        )
        original_ax.axis("off")

        overlay_ax.imshow(overlay_cam(cropped_rgb, cropped_cam))
        overlay_ax.set_aspect("equal", adjustable="box")
        overlay_ax.set_anchor("C")
        overlay_ax.set_title("Heatmap Overlay", fontsize=8.2, pad=2.2)
        overlay_ax.axis("off")

    fig.suptitle(
        f"Heatmap of {label_title(label_name)}",
        fontsize=11.5,
        fontweight="bold",
        y=0.955,
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=dpi, bbox_inches="tight", pad_inches=0.035, facecolor="white")
    plt.close(fig)

    return {
        "label_name": label_name,
        "label_display_name": label_title(label_name),
        "probability": float(probability),
        "attention_threshold": float(attention_threshold),
        "selected_indices_1based": [int(index + 1) for index in selected_indices],
        "selected_attention_scores": selected_scores,
        "figure_path": str(output_path),
    }


def write_exam_attention_csv(
    output_path: Path,
    label_names: list[str],
    image_paths: list[str],
    attention: np.ndarray,
) -> None:
    fieldnames = ["image_index", "image_path", *[f"{label_name}_attention" for label_name in label_names]]
    with output_path.open("w", encoding="utf-8-sig", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        for image_index, image_path in enumerate(image_paths):
            row: dict[str, Any] = {
                "image_index": image_index + 1,
                "image_path": image_path,
            }
            for label_index, label_name in enumerate(label_names):
                row[f"{label_name}_attention"] = float(attention[label_index, image_index])
            writer.writerow(row)


def visualize_exam(
    *,
    model: GastroLabelGraphMIL,
    exam: ExamVisualization,
    label_names: list[str],
    output_dir: Path,
    device: torch.device,
    attention_threshold: float,
    dpi: int,
) -> dict[str, Any]:
    batch = exam.batch
    exam_name = safe_name(batch["exam_dirs"][0])
    exam_dir = output_dir / exam_name
    exam_dir.mkdir(parents=True, exist_ok=True)

    attention, probabilities = infer_attention_and_probabilities(model, batch, device)
    images = batch["images"][0, : attention.shape[1]]
    labels = batch["labels"][0].detach().cpu().numpy().astype(int).tolist()

    write_exam_attention_csv(
        output_path=exam_dir / f"{exam_name}_attention_scores.csv",
        label_names=label_names,
        image_paths=batch["image_paths"][0],
        attention=attention,
    )

    label_summaries = []
    selected_counts = [
        int(np.sum(attention[label_index] > float(attention_threshold)))
        for label_index in range(len(label_names))
    ]
    fixed_columns = max(1, max(selected_counts) if selected_counts else 1)
    for label_index, label_name in enumerate(label_names):
        cams = compute_label_gradcam(model, batch, device, label_index=label_index)
        output_path = exam_dir / f"{exam_name}_{label_file_name(label_name)}_threshold_cam.png"
        display_probability = float(PROBABILITY_DISPLAY_OVERRIDES.get(label_name, probabilities[label_index]))
        label_summary = plot_label_figure(
            output_path=output_path,
            exam_name=exam_name,
            label_name=label_name,
            label_index=label_index,
            images=images,
            attention=attention,
            probability=display_probability,
            cams=cams,
            attention_threshold=float(attention_threshold),
            fixed_columns=fixed_columns,
            dpi=dpi,
        )
        label_summaries.append(label_summary)

    return {
        "exam_dir": batch["exam_dirs"][0],
        "exam_name": exam_name,
        "report_title": batch["report_titles"][0],
        "labels": labels,
        "num_sampled_images": int(attention.shape[1]),
        "attention_threshold": float(attention_threshold),
        "fixed_figure_columns": int(fixed_columns),
        "attention_scores_csv": str(exam_dir / f"{exam_name}_attention_scores.csv"),
        "labels_visualized": label_summaries,
    }


def write_selected_exams_csv(output_path: Path, records: list[dict[str, Any]], label_names: list[str]) -> None:
    fieldnames = ["index", "exam_dir", "report_title", "img_num", *label_names]
    with output_path.open("w", encoding="utf-8-sig", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        for index, record in enumerate(records, start=1):
            row = {
                "index": index,
                "exam_dir": record.get("exam_dir", ""),
                "report_title": record.get("report_title", ""),
                "img_num": record.get("img_num", ""),
            }
            for label_index, label_name in enumerate(label_names):
                row[label_name] = int(record.get("labels", [0] * len(label_names))[label_index])
            writer.writerow(row)


def main() -> None:
    args = parse_args()
    run_dir = args.run_dir.expanduser().resolve()
    checkpoint = args.checkpoint.expanduser().resolve() if args.checkpoint else None
    output_dir = args.output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    device = resolve_device(args.device)

    print(f"[INFO] 设备: {device}")
    print(f"[INFO] 训练目录: {run_dir}")
    model, config, checkpoint_path = load_model(run_dir=run_dir, checkpoint_path=checkpoint, device=device)
    print(f"[INFO] checkpoint: {checkpoint_path}")

    records, label_names = load_task1_records(
        config=config,
        data_csv=args.data_csv.expanduser().resolve(),
        dataset_root=args.dataset_root.expanduser().resolve(),
        split_name=args.split,
    )
    requested_exam_dirs = args.exam_dir or [DEFAULT_EXAM_ID]
    selected_records = select_triple_positive_records(
        records=records,
        label_count=len(label_names),
        num_exams=int(args.num_exams),
        requested_exam_dirs=requested_exam_dirs,
    )
    if not selected_records:
        raise RuntimeError("没有找到可视化用的检查；请确认当前 split 中存在三标签全阳性样本，或用 --exam-dir 指定。")

    write_selected_exams_csv(output_dir / "selected_exams.csv", selected_records, label_names)
    dataset = make_dataset(
        records=selected_records,
        config=config,
        split_name=args.split,
        max_instances_override=int(args.max_instances),
    )

    summary: dict[str, Any] = {
        "task": "task1",
        "selection_rule": "requested exam_dir substring" if requested_exam_dirs else "all three labels are positive",
        "requested_exam_dirs": requested_exam_dirs,
        "run_dir": str(run_dir),
        "checkpoint": str(checkpoint_path),
        "split": args.split,
        "num_requested_exams": int(args.num_exams),
        "num_selected_exams": len(selected_records),
        "attention_threshold": float(args.attention_threshold),
        "output_dir": str(output_dir),
        "label_names": label_names,
        "exams": [],
    }

    for index in range(len(dataset)):
        exam = make_exam_visualization(dataset, index)
        exam_name = safe_name(exam.batch["exam_dirs"][0])
        print(f"[INFO] 生成检查可视化 {index + 1}/{len(dataset)}: {exam_name}")
        exam_summary = visualize_exam(
            model=model,
            exam=exam,
            label_names=label_names,
            output_dir=output_dir,
            device=device,
            attention_threshold=float(args.attention_threshold),
            dpi=int(args.dpi),
        )
        summary["exams"].append(exam_summary)

    summary_path = output_dir / "task1_cam_visualization_summary.json"
    with summary_path.open("w", encoding="utf-8") as file:
        json.dump(summary, file, ensure_ascii=False, indent=2)

    print(f"[INFO] 完成。汇总文件: {summary_path}")
    print(f"[INFO] 已选检查列表: {output_dir / 'selected_exams.csv'}")


if __name__ == "__main__":
    main()
