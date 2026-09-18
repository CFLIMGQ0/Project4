#!/usr/bin/env python3
"""在 PhysioNet CT-ICH 上运行论文表2中的纯图像 MIL 基线。"""

from __future__ import annotations

import argparse
import csv
import gc
import json
import math
import random
import sys
from copy import deepcopy
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import accuracy_score, average_precision_score, f1_score, roc_auc_score
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm


SRC_ROOT = Path(__file__).resolve().parents[1]
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from baselines.task1.gastro_baseline import build_gastro_baseline  # noqa: E402
from exp_8.models import Exp12AProCoPEWatchCrossAttentionTextCNNModel  # noqa: E402
from sotas.task1.gastro_sota import build_gastro_sota  # noqa: E402
from training.losses import AsymmetricLossMultiLabel  # noqa: E402


LABEL_COLUMNS = ("Intraparenchymal", "Epidural", "Fracture_Yes_No")
LABEL_NAMES = ("IPH", "EDH", "Fracture")
MODEL_SPECS: dict[str, dict[str, Any]] = {
    "attention_mil": {
        "display_name": "Attention MIL",
        "registry": "baseline",
        "model_name": "gastro_attention_mil_baseline",
        "params": {"attn_dim": 256},
    },
    "mean_pooling": {
        "display_name": "Mean pooling",
        "registry": "baseline",
        "model_name": "gastro_mean_pool_baseline",
        "params": {"hidden_dim": 512},
    },
    "transformer_context_mil": {
        "display_name": "Transformer-context MIL",
        "registry": "baseline",
        "model_name": "gastro_transformer_mil_baseline",
        "params": {"attn_dim": 256, "num_heads": 8, "num_layers": 2},
    },
    "topk_mil": {
        "display_name": "Top-k MIL",
        "registry": "baseline",
        "model_name": "gastro_topk_mil_baseline",
        "params": {"hidden_dim": 256, "topk": 4},
    },
    "max_pooling": {
        "display_name": "Max pooling",
        "registry": "baseline",
        "model_name": "gastro_max_pool_baseline",
        "params": {"hidden_dim": 256},
    },
    "transmil": {
        "display_name": "TransMIL",
        "registry": "sota",
        "model_name": "gastro_transmil_sota",
        "params": {"hidden_dim": 256, "num_heads": 8, "num_layers": 2},
    },
    "dsmil": {
        "display_name": "DSMIL",
        "registry": "sota",
        "model_name": "gastro_dsmil_sota",
        "params": {"hidden_dim": 256},
    },
    "dtfd_mil": {
        "display_name": "DTFD-MIL",
        "registry": "sota",
        "model_name": "gastro_dtfd_mil_sota",
        "params": {"attn_dim": 256, "hidden_dim": 256, "num_groups": 4},
        "aux_weights": {"pseudo_bag": 0.2},
    },
    "clam_mb": {
        "display_name": "CLAM-MB",
        "registry": "sota",
        "model_name": "gastro_clam_mb_sota",
        "params": {"attn_dim": 256, "hidden_dim": 256, "instance_topk": 4},
        "aux_weights": {
            "attention_entropy": 0.05,
            "attention_diversity": 0.05,
            "instance_clustering": 0.2,
        },
    },
    "clam_sb": {
        "display_name": "CLAM-SB",
        "registry": "sota",
        "model_name": "gastro_clam_sb_sota",
        "params": {"hidden_dim": 256, "instance_topk": 4},
        "aux_weights": {"attention_entropy": 0.05, "instance_clustering": 0.2},
    },
    "amef_image_branch": {
        "display_name": "AMEF-MIL image-only branch",
        "registry": "ours",
        "model_name": "exp12_apro_cope_watch_cross_attn_textcnn",
        "params": {},
    },
}


def parse_args() -> argparse.Namespace:
    project_root = Path(__file__).resolve().parents[2]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--data-root",
        type=Path,
        default=project_root
        / "datasets"
        / "physionet_ct_ich"
        / "computed-tomography-images-for-intracranial-hemorrhage-detection-and-segmentation-1.0.0",
    )
    parser.add_argument(
        "--feature-cache",
        type=Path,
        default=project_root
        / "outputs"
        / "physionet_ct_ich"
        / "mean_pool_baseline"
        / "convnext_tiny_slice_features.npz",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=project_root / "outputs" / "physionet_ct_ich" / "table2_image_baselines",
    )
    parser.add_argument("--models", nargs="+", default=list(MODEL_SPECS))
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--weight-decay", type=float, default=0.02)
    parser.add_argument("--warmup-ratio", type=float, default=0.2)
    parser.add_argument("--max-instances", type=int, default=64)
    parser.add_argument("--instance-dropout", type=float, default=0.25)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def multilabel_folds(y: np.ndarray, n_splits: int, seed: int) -> list[np.ndarray]:
    """以确定性的贪心分配平衡各折的多标签阳性数。"""
    rng = random.Random(seed)
    label_frequency = y.sum(axis=0).clip(min=1)
    positives = [index for index in range(len(y)) if y[index].sum() > 0]
    negatives = [index for index in range(len(y)) if y[index].sum() == 0]
    rng.shuffle(positives)
    positives.sort(
        key=lambda index: (-float((y[index] / label_frequency).sum()), -int(y[index].sum()))
    )
    folds: list[list[int]] = [[] for _ in range(n_splits)]
    fold_label_counts = np.zeros((n_splits, y.shape[1]), dtype=np.float64)
    desired_labels = y.sum(axis=0) / float(n_splits)
    desired_size = len(y) / float(n_splits)
    for index in positives:
        active = y[index].astype(bool)
        scores = []
        for fold_index in range(n_splits):
            label_deficit = (desired_labels[active] - fold_label_counts[fold_index, active]).sum()
            size_deficit = desired_size - len(folds[fold_index])
            scores.append((float(label_deficit), float(size_deficit), rng.random()))
        chosen = max(range(n_splits), key=lambda fold_index: scores[fold_index])
        folds[chosen].append(index)
        fold_label_counts[chosen] += y[index]
    rng.shuffle(negatives)
    for index in negatives:
        smallest = min(len(fold) for fold in folds)
        candidates = [i for i, fold in enumerate(folds) if len(fold) == smallest]
        folds[rng.choice(candidates)].append(index)
    return [np.asarray(sorted(fold), dtype=np.int64) for fold in folds]


def load_bags(
    data_root: Path,
    feature_cache: Path,
) -> tuple[np.ndarray, list[np.ndarray], np.ndarray]:
    cached = np.load(feature_cache)
    features = cached["features"].astype(np.float32)
    cached_patient_ids = cached["patient_ids"].astype(np.int64)
    cached_slice_numbers = cached["slice_numbers"].astype(np.int64)

    slice_targets: dict[tuple[int, int], np.ndarray] = {}
    with (data_root / "hemorrhage_diagnosis.csv").open(
        "r", encoding="utf-8-sig", newline=""
    ) as file:
        for row in tqdm(csv.DictReader(file), desc="读取切片标签"):
            key = (int(row["PatientNumber"]), int(row["SliceNumber"]))
            slice_targets[key] = np.asarray([int(row[column]) for column in LABEL_COLUMNS])

    patient_ids = np.asarray(sorted(np.unique(cached_patient_ids)), dtype=np.int64)
    bags: list[np.ndarray] = []
    targets: list[np.ndarray] = []
    for patient_id in tqdm(patient_ids, desc="构建患者级图像包"):
        indices = np.flatnonzero(cached_patient_ids == patient_id)
        indices = indices[np.argsort(cached_slice_numbers[indices])]
        bags.append(features[indices])
        patient_slice_targets = [
            slice_targets[(int(patient_id), int(cached_slice_numbers[index]))] for index in indices
        ]
        targets.append(np.max(np.stack(patient_slice_targets), axis=0))
    return patient_ids, bags, np.stack(targets).astype(np.int64)


class FeatureBagDataset(Dataset):
    def __init__(
        self,
        bags: list[np.ndarray],
        targets: np.ndarray,
        indices: np.ndarray,
        max_instances: int,
        instance_dropout: float,
        training: bool,
    ) -> None:
        self.bags = bags
        self.targets = targets
        self.indices = indices
        self.max_instances = max_instances
        self.instance_dropout = instance_dropout
        self.training = training

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, item: int) -> tuple[torch.Tensor, torch.Tensor, int]:
        index = int(self.indices[item])
        bag = self.bags[index]
        if len(bag) > self.max_instances:
            if self.training:
                selected = np.sort(np.random.choice(len(bag), self.max_instances, replace=False))
            else:
                selected = np.linspace(0, len(bag) - 1, self.max_instances).round().astype(np.int64)
            bag = bag[selected]
        if self.training and self.instance_dropout > 0 and len(bag) > 1:
            keep_count = max(1, int(round(len(bag) * (1.0 - self.instance_dropout))))
            selected = np.sort(np.random.choice(len(bag), keep_count, replace=False))
            bag = bag[selected]
        return torch.from_numpy(bag), torch.from_numpy(self.targets[index].astype(np.float32)), index


def collate_bags(
    batch: list[tuple[torch.Tensor, torch.Tensor, int]],
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    max_instances = max(row[0].shape[0] for row in batch)
    feature_dim = batch[0][0].shape[-1]
    features = torch.zeros(len(batch), max_instances, feature_dim, 1, 1, dtype=torch.float32)
    mask = torch.zeros(len(batch), max_instances, dtype=torch.bool)
    targets = torch.stack([row[1] for row in batch])
    indices = torch.tensor([row[2] for row in batch], dtype=torch.long)
    for batch_index, (bag, _, _) in enumerate(batch):
        count = bag.shape[0]
        features[batch_index, :count, :, 0, 0] = bag
        mask[batch_index, :count] = True
    return features, mask, targets, indices


class CachedFeatureBackbone(nn.Module):
    """复现原模型backbone末端的768到512维投影。"""

    def __init__(self, input_dim: int = 768, output_dim: int = 512, dropout: float = 0.2) -> None:
        super().__init__()
        self.projector = nn.Sequential(
            nn.Flatten(1),
            nn.Linear(input_dim, output_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.projector(x)


class AMEFImageOnlyBranch(nn.Module):
    """AMEF-MIL的APro-CoPE、标签注意力与标签超图纯图像路径。"""

    def __init__(self) -> None:
        super().__init__()
        # PyTorch的MHA推理快速路径无法稳定处理APro-CoPE生成的逐头三维偏置。
        torch.backends.mha.set_fastpath_enabled(False)
        self.core = Exp12AProCoPEWatchCrossAttentionTextCNNModel(
            backbone_name="convnext_tiny",
            pretrained=False,
            freeze_stages=1,
            feature_dim=512,
            attn_dim=256,
            hidden_dim=1024,
            num_labels=len(LABEL_NAMES),
            dropout=0.2,
            encoder_chunk_size=16,
            num_heads=4,
            num_layers=2,
            use_label_graph=True,
            label_graph_type="label_hypergraph",
            label_hypergraph_edges=2,
            text_vocab_size=8192,
            text_embed_dim=128,
            textcnn_kernel_sizes=(2, 3, 4),
            position_variant="apro_full",
            apro_position_dim=64,
            apro_warp_alpha=1.5,
            apro_fourier_frequencies=8,
        )
        self.core.instance_encoder.backbone = CachedFeatureBackbone()

    def forward(self, images: torch.Tensor, mask: torch.Tensor) -> dict[str, torch.Tensor]:
        context_features, label_embeds, attention, extra_outputs = self.core.encode_long_mil(
            images,
            mask,
        )
        logits = self.core.classify(label_embeds)
        return self.core.build_outputs(
            logits=logits,
            attention=attention,
            features=context_features,
            extra_outputs=extra_outputs,
        )


def build_model(model_key: str) -> nn.Module:
    spec = MODEL_SPECS[model_key]
    if spec["registry"] == "ours":
        return AMEFImageOnlyBranch()
    common = {
        "backbone_name": "convnext_tiny",
        "pretrained": False,
        "freeze_stages": 1,
        "feature_dim": 512,
        "num_labels": len(LABEL_NAMES),
        "dropout": 0.2,
        **spec.get("params", {}),
    }
    if spec["registry"] == "baseline":
        model = build_gastro_baseline(spec["model_name"], **common)
        model.encoder = CachedFeatureBackbone()
    else:
        model = build_gastro_sota(spec["model_name"], **common)
        model.instance_encoder.backbone = CachedFeatureBackbone()
    return model


def forward_model(
    model: nn.Module,
    model_key: str,
    features: torch.Tensor,
    mask: torch.Tensor,
    targets: torch.Tensor | None,
) -> dict[str, torch.Tensor]:
    if targets is not None and model_key in {"dtfd_mil", "clam_mb", "clam_sb"}:
        return model(features, mask, labels=targets)
    return model(features, mask)


def metric_dict(targets: np.ndarray, probabilities: np.ndarray) -> dict[str, Any]:
    predictions = (probabilities >= 0.5).astype(np.int64)
    result: dict[str, Any] = {
        "macro_f1": float(f1_score(targets, predictions, average="macro", zero_division=0)),
        "micro_f1": float(f1_score(targets, predictions, average="micro", zero_division=0)),
        "exact_match": float(accuracy_score(targets, predictions)),
        "per_label_f1": f1_score(targets, predictions, average=None, zero_division=0).tolist(),
    }
    try:
        result["macro_auroc"] = float(roc_auc_score(targets, probabilities, average="macro"))
    except ValueError:
        result["macro_auroc"] = None
    try:
        result["macro_auprc"] = float(average_precision_score(targets, probabilities, average="macro"))
    except ValueError:
        result["macro_auprc"] = None
    return result


@torch.inference_mode()
def evaluate(
    model: nn.Module,
    model_key: str,
    loader: DataLoader,
    device: torch.device,
) -> tuple[float, np.ndarray, np.ndarray, np.ndarray]:
    model.eval()
    criterion = AsymmetricLossMultiLabel()
    losses: list[float] = []
    targets_list: list[np.ndarray] = []
    probabilities_list: list[np.ndarray] = []
    indices_list: list[np.ndarray] = []
    for features, mask, targets, indices in loader:
        features = features.to(device, non_blocking=True)
        mask = mask.to(device, non_blocking=True)
        targets_device = targets.to(device, non_blocking=True)
        output = forward_model(model, model_key, features, mask, None)
        losses.append(float(criterion(output["logits"], targets_device).item()))
        targets_list.append(targets.numpy().astype(np.int64))
        probabilities_list.append(torch.sigmoid(output["logits"]).cpu().numpy())
        indices_list.append(indices.numpy())
    return (
        float(np.mean(losses)),
        np.concatenate(targets_list),
        np.concatenate(probabilities_list),
        np.concatenate(indices_list),
    )


def train_one_fold(
    model_key: str,
    fold_number: int,
    bags: list[np.ndarray],
    targets: np.ndarray,
    train_indices: np.ndarray,
    val_indices: np.ndarray,
    test_indices: np.ndarray,
    args: argparse.Namespace,
) -> tuple[dict[str, Any], np.ndarray, np.ndarray]:
    fold_seed = args.seed + fold_number * 100
    seed_everything(fold_seed)
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    model = build_model(model_key).to(device)
    criterion = AsymmetricLossMultiLabel()
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    train_loader = DataLoader(
        FeatureBagDataset(
            bags, targets, train_indices, args.max_instances, args.instance_dropout, True
        ),
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=0,
        collate_fn=collate_bags,
        generator=torch.Generator().manual_seed(fold_seed),
    )
    val_loader = DataLoader(
        FeatureBagDataset(bags, targets, val_indices, args.max_instances, 0.0, False),
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=0,
        collate_fn=collate_bags,
    )
    test_loader = DataLoader(
        FeatureBagDataset(bags, targets, test_indices, args.max_instances, 0.0, False),
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=0,
        collate_fn=collate_bags,
    )

    total_steps = max(1, len(train_loader) * args.epochs)
    warmup_steps = int(total_steps * args.warmup_ratio)

    def lr_lambda(step: int) -> float:
        if step < warmup_steps:
            return float(step + 1) / float(max(1, warmup_steps))
        progress = float(step - warmup_steps) / float(max(1, total_steps - warmup_steps))
        return 0.5 * (1.0 + math.cos(math.pi * progress))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
    amp_enabled = device.type == "cuda" and model_key != "amef_image_branch"
    scaler = torch.amp.GradScaler("cuda", enabled=amp_enabled)
    best_val_loss = float("inf")
    best_epoch = 0
    best_state: dict[str, torch.Tensor] | None = None
    aux_weights = MODEL_SPECS[model_key].get("aux_weights", {})
    epoch_bar = tqdm(range(1, args.epochs + 1), desc=f"{MODEL_SPECS[model_key]['display_name']} 折{fold_number}")
    for epoch in epoch_bar:
        model.train()
        train_losses: list[float] = []
        for features, mask, batch_targets, _ in train_loader:
            features = features.to(device, non_blocking=True)
            mask = mask.to(device, non_blocking=True)
            batch_targets = batch_targets.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device_type=device.type, enabled=amp_enabled):
                output = forward_model(model, model_key, features, mask, batch_targets)
                loss = criterion(output["logits"], batch_targets)
                for loss_name, loss_weight in aux_weights.items():
                    if loss_name in output.get("aux_losses", {}):
                        loss = loss + float(loss_weight) * output["aux_losses"][loss_name]
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()
            train_losses.append(float(loss.item()))

        val_loss, _, _, _ = evaluate(model, model_key, val_loader, device)
        epoch_bar.set_postfix(train=f"{np.mean(train_losses):.4f}", val=f"{val_loss:.4f}")
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_epoch = epoch
            best_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}

    if best_state is None:
        raise RuntimeError("训练结束后未获得有效checkpoint")
    model.load_state_dict(best_state)
    _, test_targets, test_probabilities, ordered_indices = evaluate(
        model, model_key, test_loader, device
    )
    metrics = metric_dict(test_targets, test_probabilities)
    metrics.update(
        {
            "fold": fold_number,
            "best_epoch": best_epoch,
            "best_val_loss": best_val_loss,
            "train_size": int(len(train_indices)),
            "val_size": int(len(val_indices)),
            "test_size": int(len(test_indices)),
            "test_positive_counts": test_targets.sum(axis=0).tolist(),
        }
    )
    del model, optimizer, scaler
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return metrics, ordered_indices, test_probabilities


def run_model(
    model_key: str,
    patient_ids: np.ndarray,
    bags: list[np.ndarray],
    targets: np.ndarray,
    folds: list[np.ndarray],
    args: argparse.Namespace,
) -> dict[str, Any]:
    output_path = args.output_dir / f"{model_key}.json"
    oof_probabilities = np.full(targets.shape, np.nan, dtype=np.float64)
    fold_results: list[dict[str, Any]] = []
    for test_fold_index in range(args.folds):
        val_fold_index = (test_fold_index + 1) % args.folds
        test_indices = folds[test_fold_index]
        val_indices = folds[val_fold_index]
        train_indices = np.concatenate(
            [
                folds[index]
                for index in range(args.folds)
                if index not in {test_fold_index, val_fold_index}
            ]
        )
        metrics, ordered_indices, probabilities = train_one_fold(
            model_key,
            test_fold_index + 1,
            bags,
            targets,
            train_indices,
            val_indices,
            test_indices,
            args,
        )
        oof_probabilities[ordered_indices] = probabilities
        fold_results.append(metrics)
        temporary = {
            "model": MODEL_SPECS[model_key]["display_name"],
            "status": "running",
            "completed_folds": len(fold_results),
            "folds": fold_results,
        }
        output_path.write_text(json.dumps(temporary, ensure_ascii=False, indent=2), encoding="utf-8")

    if np.isnan(oof_probabilities).any():
        raise RuntimeError(f"{model_key}的OOF预测不完整")
    fold_macro = np.asarray([row["macro_f1"] for row in fold_results], dtype=np.float64)
    fold_micro = np.asarray([row["micro_f1"] for row in fold_results], dtype=np.float64)
    result = {
        "model_key": model_key,
        "model": MODEL_SPECS[model_key]["display_name"],
        "status": "complete",
        "labels": list(LABEL_NAMES),
        "num_patients": int(len(patient_ids)),
        "positive_patients": targets.sum(axis=0).tolist(),
        "protocol": {
            "split": "patient-grouped five-fold; three folds train, one fold validation, one fold test",
            "features": "frozen ImageNet ConvNeXt-Tiny slice features",
            "max_instances": args.max_instances,
            "epochs": args.epochs,
            "learning_rate": args.lr,
            "loss": "asymmetric multilabel loss",
            "threshold": 0.5,
        },
        "folds": fold_results,
        "fold_macro_f1_mean": float(fold_macro.mean()),
        "fold_macro_f1_std": float(fold_macro.std(ddof=1)),
        "fold_micro_f1_mean": float(fold_micro.mean()),
        "fold_micro_f1_std": float(fold_micro.std(ddof=1)),
        "oof": metric_dict(targets, oof_probabilities),
    }
    output_path.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")

    prediction_path = args.output_dir / f"{model_key}_oof_predictions.csv"
    with prediction_path.open("w", encoding="utf-8", newline="") as file:
        fields = ["patient_id"]
        for label_name in LABEL_NAMES:
            fields.extend([f"true_{label_name}", f"prob_{label_name}", f"pred_{label_name}"])
        writer = csv.DictWriter(file, fieldnames=fields)
        writer.writeheader()
        for row_index, patient_id in enumerate(patient_ids):
            row: dict[str, Any] = {"patient_id": int(patient_id)}
            for label_index, label_name in enumerate(LABEL_NAMES):
                probability = float(oof_probabilities[row_index, label_index])
                row[f"true_{label_name}"] = int(targets[row_index, label_index])
                row[f"prob_{label_name}"] = probability
                row[f"pred_{label_name}"] = int(probability >= 0.5)
            writer.writerow(row)
    print(
        f"{result['model']}: Macro-F1={result['fold_macro_f1_mean']:.4f} "
        f"± {result['fold_macro_f1_std']:.4f}"
    )
    return result


def write_summary(results: list[dict[str, Any]], output_dir: Path) -> None:
    results.sort(key=lambda row: float(row["fold_macro_f1_mean"]), reverse=True)
    summary_json = output_dir / "summary.json"
    summary_json.write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")
    summary_csv = output_dir / "summary.csv"
    with summary_csv.open("w", encoding="utf-8", newline="") as file:
        fields = [
            "model",
            "macro_f1_mean",
            "macro_f1_std",
            "micro_f1_mean",
            "micro_f1_std",
            "oof_macro_f1",
            "oof_micro_f1",
            "oof_macro_auroc",
            "oof_macro_auprc",
        ]
        writer = csv.DictWriter(file, fieldnames=fields)
        writer.writeheader()
        for result in results:
            writer.writerow(
                {
                    "model": result["model"],
                    "macro_f1_mean": result["fold_macro_f1_mean"],
                    "macro_f1_std": result["fold_macro_f1_std"],
                    "micro_f1_mean": result["fold_micro_f1_mean"],
                    "micro_f1_std": result["fold_micro_f1_std"],
                    "oof_macro_f1": result["oof"]["macro_f1"],
                    "oof_micro_f1": result["oof"]["micro_f1"],
                    "oof_macro_auroc": result["oof"]["macro_auroc"],
                    "oof_macro_auprc": result["oof"]["macro_auprc"],
                }
            )


def main() -> None:
    args = parse_args()
    unknown = [model for model in args.models if model not in MODEL_SPECS]
    if unknown:
        raise ValueError(f"未知模型：{unknown}；可选模型：{list(MODEL_SPECS)}")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    seed_everything(args.seed)
    patient_ids, bags, targets = load_bags(args.data_root, args.feature_cache)
    folds = multilabel_folds(targets, args.folds, args.seed)
    split_payload = {
        "patient_ids": patient_ids.tolist(),
        "labels": list(LABEL_NAMES),
        "folds": [patient_ids[indices].tolist() for indices in folds],
        "fold_positive_counts": [targets[indices].sum(axis=0).tolist() for indices in folds],
    }
    (args.output_dir / "patient_folds.json").write_text(
        json.dumps(split_payload, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    results = []
    for model_key in args.models:
        results.append(run_model(model_key, patient_ids, bags, targets, folds, args))
    write_summary(results, args.output_dir)


if __name__ == "__main__":
    main()
