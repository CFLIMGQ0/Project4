#!/usr/bin/env python3
"""使用 TASK3 主模型 Fold 1 测试集融合特征绘制 t-SNE 图。"""

from __future__ import annotations

import argparse
import csv
import json
import os
import sys
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import yaml
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader
from tqdm import tqdm

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from exp_8 import build_exp8_model
from scripts.task3_main_model_5fold import apply_watch_mask
from training.data import InstanceAwareBatchSampler, MILBagDataset, mil_collate_fn


LABEL_NAMES = (
    "label_esophageal_smt",
    "label_esophageal_mucosal_or_tumor",
    "label_gastritis",
)
LABEL_DISPLAY_NAMES = (
    "Esophageal SMT",
    "Esophageal mucosal lesion/tumor",
    "Gastritis",
)
DATASET_DISPLAY_NAMES = {
    "regular_white_light": "White-light",
    "chromoscopic": "Chromoscopic",
    "surgical": "Surgical",
    "ultrasound": "Ultrasound",
}
DATASET_ORDER = tuple(DATASET_DISPLAY_NAMES)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        type=Path,
        default=ROOT / "configs/task3/t3_main_model.yaml",
        help="TASK3 主模型配置",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=ROOT.parent / "outputs/t_sne",
        help="t-SNE 输出目录",
    )
    parser.add_argument("--fold", type=int, default=1, help="固定用于可视化的折号")
    parser.add_argument(
        "--datasets",
        default=",".join(DATASET_ORDER),
        help="逗号分隔的数据集键",
    )
    parser.add_argument(
        "--device",
        default="auto",
        help="推理设备，例如 auto、cpu、cuda:2",
    )
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--max-iter", type=int, default=2000)
    parser.add_argument(
        "--reuse-features",
        action="store_true",
        help="存在已提取特征时跳过模型推理",
    )
    parser.add_argument(
        "--validate-only",
        action="store_true",
        help="仅校验数据、检查点与模型权重，不执行推理",
    )
    return parser.parse_args()


def read_yaml(path: Path) -> dict[str, Any]:
    payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"配置文件顶层必须是字典：{path}")
    return payload


def to_builtin(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): to_builtin(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [to_builtin(item) for item in value]
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Path):
        return str(value)
    return value


def resolve_device(raw_device: str) -> torch.device:
    requested = str(raw_device).strip().lower()
    if requested != "auto":
        device = torch.device(requested)
        if device.type == "cuda" and not torch.cuda.is_available():
            raise RuntimeError("请求使用 CUDA，但当前环境无法识别 GPU")
        return device
    if not torch.cuda.is_available():
        return torch.device("cpu")
    free_memory = []
    for index in range(torch.cuda.device_count()):
        free_bytes, _ = torch.cuda.mem_get_info(index)
        free_memory.append((int(free_bytes), index))
    return torch.device(f"cuda:{max(free_memory)[1]}")


def load_masked_records(records_cache: Path) -> dict[str, dict[str, Any]]:
    payload = json.loads(records_cache.read_text(encoding="utf-8"))
    records = payload.get("records")
    if not isinstance(records, list):
        raise ValueError(f"样本缓存格式错误：{records_cache}")
    apply_watch_mask(records, enabled=True)
    record_map: dict[str, dict[str, Any]] = {}
    for record in records:
        exam_dir = str(record.get("exam_dir", ""))
        if not exam_dir:
            continue
        if exam_dir in record_map:
            raise ValueError(f"样本缓存中出现重复检查目录：{exam_dir}")
        record_map[exam_dir] = record
    return record_map


def load_test_records(
    manifest_path: Path,
    record_map: dict[str, dict[str, Any]],
) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    with manifest_path.open("r", encoding="utf-8-sig", newline="") as file:
        for row in csv.DictReader(file):
            if str(row.get("split", "")).strip().lower() != "test":
                continue
            exam_dir = str(row.get("exam_dir", ""))
            record = record_map.get(exam_dir)
            if record is None:
                raise KeyError(f"测试清单中的检查未出现在样本缓存：{exam_dir}")
            manifest_labels = [int(row[label_name]) for label_name in LABEL_NAMES]
            record_labels = [int(value) for value in record["labels"]]
            if manifest_labels != record_labels:
                raise ValueError(
                    f"测试清单与样本缓存标签不一致：{exam_dir}，"
                    f"manifest={manifest_labels}，cache={record_labels}"
                )
            records.append(record)
    if len({str(record["exam_dir"]) for record in records}) != len(records):
        raise ValueError(f"测试清单存在重复检查：{manifest_path}")
    if len(records) < 4:
        raise ValueError(f"测试样本过少，无法绘制 t-SNE：{manifest_path}")
    return records


def build_model(cfg: dict[str, Any], checkpoint_path: Path, device: torch.device) -> torch.nn.Module:
    model_name = str(cfg["model"]["model_name"])
    params = dict(cfg["model"]["params"])
    params.pop("image_aux_weight", None)
    model = build_exp8_model(
        model_name=model_name,
        num_labels=len(LABEL_NAMES),
        pretrained=False,
        **params,
    )
    try:
        checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    except TypeError:
        checkpoint = torch.load(checkpoint_path, map_location="cpu")
    missing, unexpected = model.load_state_dict(checkpoint["model_state"], strict=False)
    if missing or unexpected:
        raise RuntimeError(
            f"检查点与模型结构不一致：missing={list(missing)}，unexpected={list(unexpected)}"
        )
    model.to(device)
    model.eval()
    return model


def build_test_loader(
    records: list[dict[str, Any]],
    cfg: dict[str, Any],
    num_workers: int,
    seed: int,
) -> DataLoader:
    run_cfg = cfg["training"]["run_overrides"]
    max_instances = int(run_cfg["eval_max_instances"])
    dataset = MILBagDataset(
        records=records,
        task_name="task2",
        max_instances=max_instances,
        min_instances=1,
        bag_sampling_strategy=str(run_cfg["eval_sampling_strategy"]),
        is_train=False,
        image_size=int(cfg["training"]["image_size"]),
        random_instance_dropout=0.0,
        image_cache_mode=str(run_cfg["image_cache_mode"]),
        image_cache_dir=Path(run_cfg["image_cache_dir"]),
        memory_cache_size=0,
        split_name="test",
    )
    sampler = InstanceAwareBatchSampler(
        records=records,
        max_instances_per_bag=max_instances,
        min_instances_per_bag=1,
        batch_size=1,
        max_instances_per_batch=int(run_cfg["eval_max_batch_instances"]),
        shuffle=False,
        seed=seed,
    )
    loader_kwargs: dict[str, Any] = {
        "batch_sampler": sampler,
        "num_workers": max(0, int(num_workers)),
        "pin_memory": torch.cuda.is_available(),
        "collate_fn": mil_collate_fn,
        "persistent_workers": False,
    }
    if int(num_workers) > 0:
        loader_kwargs["prefetch_factor"] = 1
    return DataLoader(dataset, **loader_kwargs)


def extract_fused_label_embeddings(
    model: torch.nn.Module,
    images: torch.Tensor,
    mask: torch.Tensor,
    watch_token_ids: torch.Tensor,
    watch_token_mask: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """复现主模型融合路径，返回 label-wise 融合特征及预测概率。"""
    _, label_embeds, _, _ = model.encode_long_mil(images, mask)
    text_tokens, text_mask, _, text_active = model.text_encoder(
        watch_token_ids,
        watch_token_mask,
        batch_size=images.shape[0],
        device=images.device,
    )
    safe_mask = text_mask.bool().clone()
    empty_rows = ~safe_mask.any(dim=1)
    if empty_rows.any():
        safe_mask[empty_rows, 0] = True
        text_tokens = text_tokens.clone()
        text_tokens[empty_rows, 0] = 0.0
    text_label_embeds, _ = model.text_cross_attn(
        label_embeds,
        text_tokens,
        text_tokens,
        key_padding_mask=~safe_mask,
        need_weights=False,
    )
    active = text_active.view(-1, 1, 1).to(dtype=text_label_embeds.dtype)
    text_label_embeds = text_label_embeds * active
    gates = torch.sigmoid(model.text_gate(torch.cat([label_embeds, text_label_embeds], dim=-1)))
    gates = gates * active
    fused_label_embeds = label_embeds + gates * text_label_embeds
    probabilities = torch.sigmoid(model.classify(fused_label_embeds))
    return fused_label_embeds, probabilities


def extract_dataset_features(
    *,
    dataset_name: str,
    model: torch.nn.Module,
    loader: DataLoader,
    device: torch.device,
) -> dict[str, np.ndarray]:
    global_features: list[np.ndarray] = []
    label_features: list[np.ndarray] = []
    labels: list[np.ndarray] = []
    probabilities: list[np.ndarray] = []
    exam_dirs: list[str] = []
    report_titles: list[str] = []

    with torch.inference_mode():
        progress = tqdm(loader, total=len(loader), desc=f"{dataset_name} 融合特征")
        for batch in progress:
            images = batch["images"].to(device, non_blocking=True)
            mask = batch["mask"].to(device, non_blocking=True)
            watch_ids = batch["watch_token_ids"].to(device, non_blocking=True)
            watch_mask = batch["watch_token_mask"].to(device, non_blocking=True)
            fused, probs = extract_fused_label_embeddings(
                model,
                images,
                mask,
                watch_ids,
                watch_mask,
            )
            label_features.append(fused.float().cpu().numpy())
            global_features.append(fused.mean(dim=1).float().cpu().numpy())
            labels.append(batch["labels"].float().cpu().numpy())
            probabilities.append(probs.float().cpu().numpy())
            exam_dirs.extend(str(value) for value in batch["exam_dirs"])
            report_titles.extend(str(value) for value in batch["report_titles"])

    return {
        "global_features": np.concatenate(global_features, axis=0),
        "label_features": np.concatenate(label_features, axis=0),
        "labels": np.concatenate(labels, axis=0),
        "probabilities": np.concatenate(probabilities, axis=0),
        "exam_dirs": np.asarray(exam_dirs, dtype=str),
        "report_titles": np.asarray(report_titles, dtype=str),
    }


def compute_tsne(features: np.ndarray, seed: int, max_iter: int) -> tuple[np.ndarray, dict[str, Any]]:
    standardized = StandardScaler().fit_transform(features)
    pca_components = min(50, standardized.shape[1], standardized.shape[0] - 1)
    reduced = PCA(n_components=pca_components, random_state=seed).fit_transform(standardized)
    perplexity = min(30.0, max(5.0, float((len(reduced) - 1) // 3)))
    embedding = TSNE(
        n_components=2,
        perplexity=perplexity,
        learning_rate="auto",
        init="pca",
        max_iter=int(max_iter),
        random_state=seed,
    ).fit_transform(reduced)
    return embedding, {
        "samples": int(len(features)),
        "original_dim": int(features.shape[1]),
        "pca_components": int(pca_components),
        "perplexity": float(perplexity),
        "max_iter": int(max_iter),
        "seed": int(seed),
    }


def scatter_binary(ax: plt.Axes, embedding: np.ndarray, labels: np.ndarray) -> None:
    colors = {0: "#4C78A8", 1: "#E45756"}
    names = {0: "Negative", 1: "Positive"}
    for value in (0, 1):
        selected = labels.astype(int) == value
        ax.scatter(
            embedding[selected, 0],
            embedding[selected, 1],
            s=23,
            c=colors[value],
            marker="o",
            alpha=0.78,
            linewidths=0.25,
            edgecolors="white",
            label=f"{names[value]} (n={int(selected.sum())})",
            rasterized=True,
        )
    ax.set_xticks([])
    ax.set_yticks([])
    ax.grid(False)
    for spine in ax.spines.values():
        spine.set_color("#D0D0D0")
        spine.set_linewidth(0.7)


def save_dataset_plot(
    *,
    dataset_name: str,
    embedding: np.ndarray,
    labels: np.ndarray,
    output_path: Path,
) -> None:
    fig, axes = plt.subplots(1, len(LABEL_NAMES), figsize=(16.2, 5.2), constrained_layout=True)
    for label_index, ax in enumerate(axes):
        scatter_binary(ax, embedding, labels[:, label_index])
        ax.set_title(LABEL_DISPLAY_NAMES[label_index], fontsize=12)
        ax.legend(loc="best", fontsize=8, frameon=True)
    fig.suptitle(
        f"{DATASET_DISPLAY_NAMES[dataset_name]} test set: fused multimodal features",
        fontsize=15,
        fontweight="bold",
    )
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def save_combined_plot(
    results: dict[str, dict[str, np.ndarray]],
    output_path: Path,
) -> None:
    dataset_names = list(results)
    fig, axes = plt.subplots(
        len(dataset_names),
        len(LABEL_NAMES),
        figsize=(17.5, 4.6 * len(dataset_names)),
        constrained_layout=True,
    )
    axes = np.atleast_2d(axes)
    for row_index, dataset_name in enumerate(dataset_names):
        embedding = results[dataset_name]["embedding"]
        labels = results[dataset_name]["labels"]
        for label_index in range(len(LABEL_NAMES)):
            ax = axes[row_index, label_index]
            scatter_binary(ax, embedding, labels[:, label_index])
            if row_index == 0:
                ax.set_title(LABEL_DISPLAY_NAMES[label_index], fontsize=12, fontweight="bold")
            if label_index == 0:
                ax.set_ylabel(
                    DATASET_DISPLAY_NAMES[dataset_name],
                    fontsize=12,
                    fontweight="bold",
                    labelpad=12,
                )
            ax.legend(loc="best", fontsize=7.5, frameon=True)
    fig.suptitle(
        "t-SNE visualization of fused multimodal representations (Fold 1 test sets)",
        fontsize=16,
        fontweight="bold",
    )
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def save_metadata_csv(
    *,
    dataset_name: str,
    payload: dict[str, np.ndarray],
    output_path: Path,
) -> None:
    fieldnames = [
        "dataset",
        "fold",
        "exam_dir",
        "report_title",
        "tsne_x",
        "tsne_y",
        *LABEL_NAMES,
        *(f"prob_{name}" for name in LABEL_NAMES),
    ]
    with output_path.open("w", encoding="utf-8-sig", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        for index in range(len(payload["labels"])):
            row: dict[str, Any] = {
                "dataset": dataset_name,
                "fold": 1,
                "exam_dir": payload["exam_dirs"][index],
                "report_title": payload["report_titles"][index],
                "tsne_x": float(payload["embedding"][index, 0]),
                "tsne_y": float(payload["embedding"][index, 1]),
            }
            for label_index, label_name in enumerate(LABEL_NAMES):
                row[label_name] = int(payload["labels"][index, label_index])
                row[f"prob_{label_name}"] = float(payload["probabilities"][index, label_index])
            writer.writerow(row)


def main() -> None:
    args = parse_args()
    cfg = read_yaml(args.config.expanduser().resolve())
    output_dir = args.output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("PROJECT4_DISABLE_DISK_CACHE_WRITE", "1")

    requested_datasets = [value.strip() for value in args.datasets.split(",") if value.strip()]
    unknown = [value for value in requested_datasets if value not in cfg["datasets"]]
    if unknown:
        raise ValueError(f"未知数据集：{unknown}")
    if not 1 <= int(args.fold) <= int(cfg["folds"]):
        raise ValueError(f"折号超出范围：{args.fold}")

    train_output_dir = Path(cfg["paths"]["output_dir"]).expanduser().resolve()
    record_map = load_masked_records(train_output_dir / "records_cache.json")
    device = resolve_device(args.device)
    print(f"[t-SNE] 推理设备：{device}")

    results: dict[str, dict[str, np.ndarray]] = {}
    run_metadata: dict[str, Any] = {
        "source_config": str(args.config.expanduser().resolve()),
        "source_experiment": str(train_output_dir),
        "output_dir": str(output_dir),
        "fold": int(args.fold),
        "checkpoint_alias": "best_macro_f1",
        "feature": "mean(fused_label_embeds, dim=labels)",
        "datasets": {},
    }

    for dataset_name in requested_datasets:
        fold_dir = train_output_dir / dataset_name / f"fold_{args.fold}"
        manifest_path = fold_dir / "split_manifest.csv"
        checkpoint_path = fold_dir / "checkpoints/best_macro_f1.ckpt"
        for required_path in (manifest_path, checkpoint_path):
            if not required_path.is_file():
                raise FileNotFoundError(f"缺少必要文件：{required_path}")
        test_records = load_test_records(manifest_path, record_map)
        positive_counts = np.asarray([record["labels"] for record in test_records], dtype=int).sum(axis=0)
        print(
            f"[t-SNE] {dataset_name}: 测试样本={len(test_records)}，"
            f"三标签阳性数={positive_counts.tolist()}"
        )

        model = build_model(cfg, checkpoint_path, device)
        if args.validate_only:
            del model
            if device.type == "cuda":
                torch.cuda.empty_cache()
            continue

        feature_path = output_dir / f"{dataset_name}_fold{args.fold}_features.npz"
        if args.reuse_features and feature_path.is_file():
            loaded = np.load(feature_path)
            payload = {key: loaded[key] for key in loaded.files}
            print(f"[t-SNE] 复用特征：{feature_path}")
        else:
            loader = build_test_loader(
                test_records,
                cfg,
                num_workers=int(args.num_workers),
                seed=int(args.seed),
            )
            payload = extract_dataset_features(
                dataset_name=dataset_name,
                model=model,
                loader=loader,
                device=device,
            )
            np.savez_compressed(feature_path, **payload)
            print(f"[t-SNE] 特征已保存：{feature_path}")
        del model
        if device.type == "cuda":
            torch.cuda.empty_cache()

        embedding, tsne_cfg = compute_tsne(
            payload["global_features"],
            seed=int(args.seed),
            max_iter=int(args.max_iter),
        )
        payload["embedding"] = embedding
        results[dataset_name] = payload
        run_metadata["datasets"][dataset_name] = {
            "display_name": DATASET_DISPLAY_NAMES.get(dataset_name, dataset_name),
            "manifest": str(manifest_path),
            "checkpoint": str(checkpoint_path),
            "test_samples": len(test_records),
            "positive_counts": {
                LABEL_NAMES[index]: int(positive_counts[index])
                for index in range(len(LABEL_NAMES))
            },
            "tsne": tsne_cfg,
        }

        dataset_plot_path = output_dir / f"{dataset_name}_fold{args.fold}_tsne.png"
        save_dataset_plot(
            dataset_name=dataset_name,
            embedding=embedding,
            labels=payload["labels"],
            output_path=dataset_plot_path,
        )
        save_metadata_csv(
            dataset_name=dataset_name,
            payload=payload,
            output_path=output_dir / f"{dataset_name}_fold{args.fold}_tsne.csv",
        )
        print(f"[t-SNE] 子图已保存：{dataset_plot_path}")

    if args.validate_only:
        print("[t-SNE] 数据、检查点与模型权重校验通过")
        return

    combined_path = output_dir / f"task3_main_model_fold{args.fold}_tsne_4x3.png"
    save_combined_plot(results, combined_path)
    metadata_path = output_dir / "run_metadata.json"
    metadata_path.write_text(
        json.dumps(to_builtin(run_metadata), ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(f"[t-SNE] 组合图已保存：{combined_path}")
    print(f"[t-SNE] 运行元数据已保存：{metadata_path}")


if __name__ == "__main__":
    main()
