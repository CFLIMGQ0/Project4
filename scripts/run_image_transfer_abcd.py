#!/usr/bin/env python3
"""AMEF-MIL纯图像知识迁移A/B/C/D训练与五折评价。"""
from __future__ import annotations

import argparse
import csv
import hashlib
import fcntl
import math
import json
import os
from pathlib import Path
import random
import sys
import time

import numpy as np
import torch
import yaml
from sklearn.metrics import f1_score
from tqdm import tqdm

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
from training.image_transfer import (
    FrozenMultimodalTeacher,
    ImageTransferStudent,
    VARIANTS,
    state_digest,
    transfer_losses,
)

CONFIG_PATH = ROOT / "src/configs/task3/image_transfer_abcd.yaml"
CONFIG = yaml.safe_load(CONFIG_PATH.read_text())
DATASETS = tuple(CONFIG["datasets"])
if set(DATASETS) != {"ct_rate", "amos_mm", "mr_rate"} or CONFIG["variants"] != VARIANTS:
    raise ValueError("仅允许指定的三个数据集和A/B/C/D四组")
OUTPUT = ROOT / CONFIG["output_root"]


def source_protocol(dataset):
    return json.loads((ROOT / CONFIG["datasets"][dataset]["settings_source"]).read_text())


def label_names(dataset):
    return source_protocol(dataset)["labels"]


def train_settings(dataset):
    settings = dict(source_protocol(dataset)["settings"])
    settings["learning_rate"] = settings.get("learning_rate", settings.get("lr"))
    settings["base_seed"] = settings.get("base_seed", settings.get("seed"))
    return settings


def save_json(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")
    temporary.replace(path)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024**2), b""):
            digest.update(block)
    return digest.hexdigest()


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def model_parameters(dataset: str) -> dict:
    protocol = source_protocol(dataset)
    parameters = dict(protocol["model_parameters"])
    return {key: value for key, value in parameters.items() if not key.endswith("_weight")}


def load_context(dataset: str):
    input_root = ROOT / CONFIG["datasets"][dataset]["input_root"]
    if dataset == "ct_rate":
        import run_ctrate_680_all_models as core
        rows, folds, labels = core.read_inputs()
        split_groups = folds["folds"]
    elif dataset == "mr_rate":
        import run_mrrate_1k_amef as core
        rows, folds, labels = core.read_inputs()
        split_groups = folds["folds"]
    else:
        import run_amos_mm_all_models as core
        rows = json.loads((input_root / "samples.json").read_text())
        split_data = json.loads((input_root / "splits.json").read_text())
        split_groups = split_data["validation_folds"]
        labels = np.asarray([row["labels"][:len(label_names(dataset))] for row in rows], dtype=np.int64)
    bags = []
    excluded = set()
    if dataset == "amos_mm":
        excluded = {
            int(item["case_index"])
            for item in json.loads((input_root / "image_exclusions.json").read_text())["excluded"]
        }
    preparation = json.loads((input_root / "preparation_protocol.json").read_text())["preparation_sha256"]
    for row in tqdm(rows, desc=f"加载{dataset}特征缓存"):
        case_index = int(row["case_index"])
        if case_index in excluded:
            bags.append(None)
            continue
        with np.load(input_root / "features" / f"{case_index:04d}.npz", allow_pickle=False) as cache:
            features = cache["features"].astype(np.float32)
            indices = cache["slice_indices"].astype(np.int64)
            count = int(cache["original_count"])
            identity = CONFIG["datasets"][dataset]["identity_field"]
            cache_identity = "study_uid" if dataset == "mr_rate" else identity
            if cache_identity in cache:
                assert str(cache[cache_identity]) == row[cache_identity]
            assert str(cache["preparation_sha256"]) == preparation
            assert features.shape == (len(indices), 768)
            assert np.isfinite(features).all() and np.all(np.diff(indices) > 0)
            assert 0 <= indices[0] <= indices[-1] < count
            bags.append((features, indices, count))
    if dataset == "amos_mm":
        known = np.asarray([row["known_mask"][:len(label_names(dataset))] for row in rows], dtype=bool)
        ids = [row["scan_id"] for row in rows]
    elif dataset == "ct_rate":
        known = np.ones_like(labels, dtype=bool)
        ids = [row["exam_id"] for row in rows]
    else:
        known = np.ones_like(labels, dtype=bool)
        ids = [row["study_uid"] for row in rows]
    return core, rows, labels, known, split_groups, bags, ids


def text_inputs(dataset: str, core, rows, train_indices: list[int], count: int):
    if dataset == "amos_mm":
        token_ids, token_mask, _ = core.tokens_for("amef_multimodal", rows, train_indices, count)
        return token_ids, token_mask
    core.fusion_base.SETTINGS = train_settings(dataset)
    indices = np.arange(len(rows), dtype=np.int64)
    texts = {int(row["case_index"]): row["findings_masked"] for row in rows}
    return core.fusion_base.encode_descriptions(texts, indices)


def make_splits(dataset: str, split_groups, fold: int, rows, excluded: set[int]):
    if dataset == "amos_mm":
        validation = list(split_groups[fold - 1])
        development = list(json.loads((ROOT / "outputs/amos_mm/experiment/splits.json").read_text())["development"])
        train = sorted(index for index in development if index not in validation and index not in excluded)
        validation = [index for index in validation if index not in excluded]
        test = list(json.loads((ROOT / "outputs/amos_mm/experiment/splits.json").read_text())["test"])
        return train, validation, test
    validation = list(split_groups[(fold) % 5])
    test = list(split_groups[fold - 1])
    train = [index for group_index, group in enumerate(split_groups) if group_index not in (fold - 1, fold % 5) for index in group]
    return train, validation, test


def make_batch(indices, bags, labels, known, token_ids, token_mask, training, device, settings, rng=None):
    selected_rows = []
    for index in indices:
        item = bags[int(index)]
        if item is None:
            raise ValueError(f"病例{index}没有有效图像缓存")
        features, positions, original_count = item
        selected = np.arange(len(features), dtype=np.int64)
        if training and len(selected) > 1:
            keep = max(1, int(round(len(selected) * (1.0 - settings["instance_dropout"]))))
            selected = np.sort(rng.choice(selected, keep, replace=False))
        selected_rows.append((features[selected], positions[selected], original_count))
    maximum = max(len(item[0]) for item in selected_rows)
    batch_size = len(indices)
    images = torch.zeros(batch_size, maximum, 768, 1, 1, device=device)
    mask = torch.zeros(batch_size, maximum, dtype=torch.bool, device=device)
    instance_indices = torch.full((batch_size, maximum), -1, dtype=torch.long, device=device)
    counts = torch.tensor([item[2] for item in selected_rows], dtype=torch.long, device=device)
    for row_index, (features, positions, _) in enumerate(selected_rows):
        length = len(features)
        images[row_index, :length, :, 0, 0] = torch.from_numpy(features).to(device)
        mask[row_index, :length] = True
        instance_indices[row_index, :length] = torch.from_numpy(positions).to(device)
    batch = {
        "images": images,
        "mask": mask,
        "instance_indices": instance_indices,
        "original_image_counts": counts,
        "labels": torch.as_tensor(labels[indices], dtype=torch.float32, device=device),
        "known": torch.as_tensor(known[indices], dtype=torch.bool, device=device),
    }
    if token_ids is not None:
        batch["watch_token_ids"] = token_ids[indices].to(device)
        batch["watch_token_mask"] = token_mask[indices].to(device)
    return batch


def build_model(dataset: str, parameters: dict, variant: str, device: torch.device):
    from run_physionet_ct_ich_table2_baselines import CachedFeatureBackbone
    params = dict(parameters)
    torch.backends.mha.set_fastpath_enabled(False)
    model = ImageTransferStudent(**params, pretrained=False, num_labels=len(label_names(dataset)))
    model.instance_encoder.backbone = CachedFeatureBackbone(768, params["feature_dim"], params["dropout"])
    model._safe_attention_mask_path = True
    model.configure_supervision(variant)
    model.to(device)
    return model


def build_teacher(dataset: str, parameters: dict, fold: int, device: torch.device):
    from exp_8.models import Exp13LCCFAblationModel
    from run_physionet_ct_ich_table2_baselines import CachedFeatureBackbone
    torch.backends.mha.set_fastpath_enabled(False)
    teacher = Exp13LCCFAblationModel(**parameters, pretrained=False, num_labels=len(label_names(dataset)))
    teacher.instance_encoder.backbone = CachedFeatureBackbone(768, parameters["feature_dim"], parameters["dropout"])
    teacher._safe_attention_mask_path = True
    checkpoint_dir = OUTPUT.parent / "lccf_ablation" / dataset / "full"
    if dataset == "amos_mm":
        checkpoint_dir = checkpoint_dir / "7_labels"
    checkpoint = checkpoint_dir / f"fold_{fold}" / "amef_multimodal" / "best_model.pt"
    payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
    teacher.load_state_dict(payload["state_dict"], strict=True)
    teacher.to(device)
    frozen = FrozenMultimodalTeacher(teacher)
    frozen.eval()
    return frozen, checkpoint


def teacher_checkpoint_path(dataset: str, fold: int) -> Path:
    checkpoint_dir = OUTPUT.parent / "lccf_ablation" / dataset / "full"
    if dataset == "amos_mm":
        checkpoint_dir = checkpoint_dir / "7_labels"
    return checkpoint_dir / f"fold_{fold}" / "amef_multimodal" / "best_model.pt"


def audit_protocol() -> None:
    audit = {
        "scope": list(DATASETS),
        "seed": 42,
        "variants": VARIANTS,
        "student": {
            "model": "ImageTransferStudent",
            "shared_visual_path": "AMEF ACPE/APro-CoPE + Transformer + LCCF visual aggregation/reasoning",
            "image_head": "independent C_img",
            "initialization": "random, same fold seed across A/B/C/D",
            "evaluation_output": "s_img only",
        },
        "teacher": {
            "model": "Exp13LCCFAblationModel",
            "checkpoint_source": "outputs/lccf_ablation/*/full",
            "selection_source": "existing LCCF full checkpoints selected by validation classification loss",
            "frozen_eval_no_grad": True,
            "independent_from_student": True,
        },
        "datasets": {},
        "notes": [
            "本实验只覆盖CT-RATE、AMOS-MM、MR-RATE-1K，不包含其他数据集。",
            "现有教师的模型结构、标签顺序和视觉缓存沿用对应LCCF full协议；教师选择指标是验证分类损失，不是本实验的纯图像Macro-F1。",
            "safe_attention_mask_path仅显式展开原TransformerEncoderLayer计算，避免当前PyTorch自定义3D mask评估态偶发NaN，不改变参数或ACPE/LCCF结构。",
        ],
    }
    for dataset in DATASETS:
        input_root = ROOT / CONFIG["datasets"][dataset]["input_root"]
        preparation_path = input_root / "preparation_protocol.json"
        source_protocol_path = OUTPUT.parent / "lccf_ablation" / dataset / "full" / "protocol.json"
        preparation = json.loads(preparation_path.read_text(encoding="utf-8"))
        source_protocol = json.loads(source_protocol_path.read_text(encoding="utf-8"))
        parameters = model_parameters(dataset)
        settings = train_settings(dataset)
        if source_protocol["model_parameters"]["position_variant"] != "apro_full":
            raise RuntimeError(f"{dataset}教师不是默认ACPE/APro-CoPE：{source_protocol['model_parameters']['position_variant']}")
        fold_records = []
        for fold in range(1, 6):
            checkpoint = teacher_checkpoint_path(dataset, fold)
            if not checkpoint.is_file():
                raise FileNotFoundError(checkpoint)
            payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
            state = payload["state_dict"]
            finite = all(torch.isfinite(value).all().item() for value in state.values() if torch.is_floating_point(value))
            if not finite:
                raise ValueError(f"教师checkpoint包含非有限参数：{checkpoint}")
            teacher, _ = build_teacher(dataset, parameters, fold, torch.device("cpu"))
            if teacher.model.num_labels != len(label_names(dataset)) or any(parameter.requires_grad for parameter in teacher.parameters()):
                raise ValueError(f"教师冻结或标签维度不符合协议：{checkpoint}")
            fold_records.append({
                "fold": fold,
                "checkpoint": str(checkpoint.relative_to(ROOT)),
                "checkpoint_sha256": sha256(checkpoint),
                "state_finite": finite,
                "labels": len(label_names(dataset)),
                "frozen": True,
            })
            del teacher
        audit["datasets"][dataset] = {
            "input_root": str(input_root.relative_to(ROOT)),
            "preparation_sha256": preparation.get("preparation_sha256"),
            "preparation_labels": preparation.get("labels"),
            "expected_labels": len(label_names(dataset)),
            "source_protocol": str(source_protocol_path.relative_to(ROOT)),
            "model_parameters": parameters,
            "folds": fold_records,
        }
    save_json(OUTPUT / "protocol_audit.json", audit)
    print(json.dumps(audit, ensure_ascii=False, indent=2))


def tune_thresholds(labels, probabilities, known, dataset):
    thresholds = np.full(labels.shape[1], 0.5, dtype=np.float64)
    for label_index in range(labels.shape[1]):
        valid = known[:, label_index]
        if valid.sum() == 0:
            continue
        if dataset == "amos_mm" and len(np.unique(labels[valid, label_index])) < 2:
            continue
        scores = [f1_score(labels[valid, label_index], probabilities[valid, label_index] >= threshold, zero_division=0)
                  for threshold in train_settings(dataset)["threshold_grid"]]
        thresholds[label_index] = train_settings(dataset)["threshold_grid"][int(np.argmax(scores))]
    return thresholds


def evaluate(dataset, student, indices, bags, labels, known, device, batch_size):
    student.eval()
    probabilities, targets, observed, case_indices = [], [], [], []
    with torch.no_grad():
        for start in range(0, len(indices), batch_size):
            batch_indices = indices[start:start + batch_size]
            batch = make_batch(batch_indices, bags, labels, known, None, None, False, device, {}, None)
            output = student(batch["images"], batch["mask"], batch["instance_indices"], batch["original_image_counts"])
            if not torch.isfinite(output["s_img"]).all():
                raise FloatingPointError("纯图像评估logits非有限，拒绝写入结果")
            probabilities.append(torch.sigmoid(output["s_img"]).cpu().numpy())
            targets.append(labels[batch_indices])
            observed.append(known[batch_indices])
            case_indices.extend(batch_indices)
    return np.concatenate(targets), np.concatenate(probabilities), np.concatenate(observed), case_indices


def masked_macro_f1(labels, probabilities, known, thresholds):
    values = []
    for label_index in range(labels.shape[1]):
        valid = known[:, label_index]
        if valid.any():
            values.append(f1_score(labels[valid, label_index], probabilities[valid, label_index] >= thresholds[label_index], zero_division=0))
    return float(np.mean(values)) if values else 0.0


def write_predictions(path, ids, labels, probabilities, known, thresholds):
    names = ["label_" + str(index) for index in range(labels.shape[1])]
    fields = ["case_index", "case_id"]
    for name in names:
        fields.extend([f"true_{name}", f"known_{name}", f"prob_{name}", f"pred_{name}"])
    with path.open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for row_index, case_index in enumerate(ids["indices"]):
            row = {"case_index": int(case_index), "case_id": ids["values"][row_index]}
            for label_index, name in enumerate(names):
                row.update({f"true_{name}": int(labels[row_index, label_index]),
                            f"known_{name}": int(known[row_index, label_index]),
                            f"prob_{name}": float(probabilities[row_index, label_index]),
                            f"pred_{name}": int(probabilities[row_index, label_index] >= thresholds[label_index])})
            writer.writerow(row)


def train_fold(dataset: str, variant: str, fold: int, device: torch.device) -> None:
    settings = train_settings(dataset)
    fold_seed = int(settings["base_seed"]) + 100 * fold
    seed_everything(fold_seed)
    core, rows, labels, known, split_groups, bags, case_ids = load_context(dataset)
    excluded = {index for index, item in enumerate(bags) if item is None}
    train, validation, test = make_splits(dataset, split_groups, fold, rows, excluded)
    parameters = model_parameters(dataset)
    output_dir = OUTPUT / dataset / variant / f"fold_{fold}"
    output_dir.mkdir(parents=True, exist_ok=True)
    model = build_model(dataset, parameters, variant, device)
    initial_digest = state_digest(model.state_dict())
    teacher, teacher_path = (None, None)
    if VARIANTS[variant]["lambda_kd"]:
        teacher, teacher_path = build_teacher(dataset, parameters, fold, device)
    text_ids, text_mask = (None, None)
    if VARIANTS[variant]["lambda_mm"] or VARIANTS[variant]["lambda_kd"]:
        text_ids, text_mask = text_inputs(dataset, core, rows, train, len(label_names(dataset)))
    optimizer = torch.optim.AdamW([parameter for parameter in model.parameters() if parameter.requires_grad],
                                  lr=settings["learning_rate"], weight_decay=settings["weight_decay"])
    total_steps = math.ceil(len(train) / settings["batch_size"]) * settings["epochs"]
    warmup_steps = int(total_steps * settings["warmup_ratio"])

    def lr_factor(step):
        if step < warmup_steps:
            return (step + 1) / max(1, warmup_steps)
        return 0.5 * (1.0 + math.cos(math.pi * (step - warmup_steps) / max(1, total_steps - warmup_steps)))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_factor)
    save_json(output_dir / "config.json", {
        "dataset": dataset, "variant": variant, "fold": fold, "seed": fold_seed,
        "model_parameters": parameters, "supervision": VARIANTS[variant], "training_settings": settings,
        "lambda_q": 0.01, "tau": 2.0, "student_initialization": "random_same_seed",
        "student_initial_state_sha256": initial_digest,
        "teacher_checkpoint": str(teacher_path.relative_to(ROOT)) if teacher_path else None,
        "teacher_checkpoint_sha256": sha256(teacher_path) if teacher_path else None,
        "teacher_selection": "existing LCCF full checkpoint; validation classification loss",
        "split_sizes": {"train": len(train), "validation": len(validation), "test": len(test)},
        "evaluation": "validation checkpoint selection and threshold tuning use s_img only; test uses s_img only",
    })
    seed_everything(fold_seed)
    best_state, best_score, best_epoch, history = None, -1.0, 0, []
    begin = time.monotonic()
    random_generator = np.random.default_rng(fold_seed)
    for epoch in range(1, settings["epochs"] + 1):
        model.train()
        if teacher is not None:
            teacher.eval()
        losses = {"l_img": [], "l_mm": [], "l_kd": [], "l_total": []}
        order = random_generator.permutation(train)
        for start in range(0, len(order), settings["batch_size"]):
            batch_indices = order[start:start + settings["batch_size"]]
            batch = make_batch(batch_indices, bags, labels, known, text_ids, text_mask, True, device, settings, random_generator)
            optimizer.zero_grad(set_to_none=True)
            _, values = transfer_losses(model, batch, variant, teacher, lambda_q=0.01, tau=2.0)
            values["l_total"].backward()
            optimizer.step()
            scheduler.step()
            for name in losses:
                if values[name] is not None:
                    losses[name].append(float(values[name].detach()))
        val_labels, val_probabilities, val_known, _ = evaluate(dataset, model, validation, bags, labels, known, device, settings["batch_size"])
        thresholds = tune_thresholds(val_labels, val_probabilities, val_known, dataset)
        val_score = masked_macro_f1(val_labels, val_probabilities, val_known, thresholds)
        row = {"epoch": epoch, "validation_macro_f1_img": val_score, "elapsed_seconds": time.monotonic() - begin}
        for name, values in losses.items():
            row[name] = float(np.mean(values)) if values else 0.0
        history.append(row)
        save_json(output_dir / "history.json", history)
        if val_score > best_score:
            best_score, best_epoch = val_score, epoch
            best_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
            best_thresholds = thresholds.copy()
    if best_state is None:
        raise RuntimeError("没有保存最佳学生checkpoint")
    model.load_state_dict(best_state)
    test_labels, test_probabilities, test_known, test_indices = evaluate(dataset, model, test, bags, labels, known, device, settings["batch_size"])
    test_score = masked_macro_f1(test_labels, test_probabilities, test_known, best_thresholds)
    val_labels, val_probabilities, val_known, val_indices = evaluate(dataset, model, validation, bags, labels, known, device, settings["batch_size"])
    write_predictions(output_dir / "test_predictions.csv", {"indices": test_indices, "values": [case_ids[index] for index in test_indices]}, test_labels, test_probabilities, test_known, best_thresholds)
    write_predictions(output_dir / "validation_predictions.csv", {"indices": val_indices, "values": [case_ids[index] for index in val_indices]}, val_labels, val_probabilities, val_known, best_thresholds)
    torch.save({"state_dict": best_state, "variant": variant, "fold": fold, "thresholds": best_thresholds.tolist(), "teacher_checkpoint": str(teacher_path) if teacher_path else None}, output_dir / "best_model.pt")
    save_json(output_dir / "test_metrics.json", {"dataset": dataset, "variant": variant, "fold": fold,
        "best_epoch": best_epoch, "best_validation_macro_f1_img": best_score, "test_macro_f1_img": test_score,
        "thresholds": best_thresholds.tolist(), "wall_seconds": time.monotonic() - begin,
        "teacher_checkpoint": str(teacher_path) if teacher_path else None})


def aggregate_results() -> None:
    rows = []
    missing = []
    for dataset in DATASETS:
        for variant in VARIANTS:
            for fold in range(1, 6):
                path = OUTPUT / dataset / variant / f"fold_{fold}" / "test_metrics.json"
                if not path.is_file():
                    missing.append(str(path.relative_to(ROOT)))
                    continue
                value = json.loads(path.read_text(encoding="utf-8"))
                rows.append({
                    "dataset": dataset,
                    "variant": variant,
                    "fold": fold,
                    "best_epoch": value["best_epoch"],
                    "validation_macro_f1_img": value["best_validation_macro_f1_img"],
                    "test_macro_f1_img": value["test_macro_f1_img"],
                })
    if missing:
        raise RuntimeError(f"仍缺少{len(missing)}个折结果，拒绝汇总：{missing[:5]}")
    fold_path = OUTPUT / "fold_results.csv"
    fold_path.parent.mkdir(parents=True, exist_ok=True)
    with fold_path.open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    summary = []
    for dataset in DATASETS:
        for variant in VARIANTS:
            values = [row["test_macro_f1_img"] for row in rows
                      if row["dataset"] == dataset and row["variant"] == variant]
            summary.append({"dataset": dataset, "variant": variant, "folds": len(values),
                            "mean_macro_f1": float(np.mean(values)),
                            "std_macro_f1": float(np.std(values, ddof=1))})
    with (OUTPUT / "summary.csv").open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(summary[0]))
        writer.writeheader()
        writer.writerows(summary)
    effects = []
    for dataset in DATASETS:
        by_variant = {
            variant: [row["test_macro_f1_img"] for row in rows
                      if row["dataset"] == dataset and row["variant"] == variant]
            for variant in VARIANTS
        }
        for left, right, name in (("B", "A", "B-A"), ("C", "A", "C-A"),
                                  ("D", "A", "D-A"), ("D", "B", "D-B"), ("D", "C", "D-C")):
            values = np.asarray(by_variant[left]) - np.asarray(by_variant[right])
            effects.append({"dataset": dataset, "difference": name, "folds": len(values),
                            "mean_macro_f1_delta": float(values.mean()),
                            "std_macro_f1_delta": float(values.std(ddof=1)),
                            "mean_percentage_points": float(values.mean() * 100.0),
                            "std_percentage_points": float(values.std(ddof=1) * 100.0)})
    with (OUTPUT / "effects.csv").open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(effects[0]))
        writer.writeheader()
        writer.writerows(effects)
    lines = [
        "# 纯图像知识迁移 A/B/C/D 五折结果",
        "",
        "仅统计 CT-RATE、AMOS-MM、MR-RATE-1K；均值和标准差按五个折的测试 Macro-F1 计算，标准差为样本标准差（ddof=1）。",
        "",
        "| 数据集 | 配置 | 折数 | Macro-F1 均值 | 标准差 |",
        "|---|---:|---:|---:|---:|",
    ]
    for row in summary:
        lines.append(f"| {row['dataset']} | {row['variant']} | {row['folds']} | {row['mean_macro_f1']:.4f} | {row['std_macro_f1']:.4f} |")
    lines.extend(["", "差值 CSV：`effects.csv`；差值的百分点列已乘100。"])
    (OUTPUT / "summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    save_json(OUTPUT / "aggregate_manifest.json", {
        "scope": list(DATASETS),
        "variants": list(VARIANTS),
        "folds": 5,
        "standard_deviation": "sample_std_ddof_1",
        "fold_results": str(fold_path.relative_to(ROOT)),
        "summary": str((OUTPUT / "summary.csv").relative_to(ROOT)),
        "effects": str((OUTPUT / "effects.csv").relative_to(ROOT)),
    })
    print((OUTPUT / "summary.md").read_text(encoding="utf-8"))


def smoke(device_name: str) -> None:
    device = torch.device(device_name)
    seed_everything(42)
    parameters = model_parameters("ct_rate")
    images = torch.randn(4, 8, 768, 1, 1, device=device)
    mask = torch.ones(4, 8, dtype=torch.bool, device=device)
    positions = torch.arange(8, device=device).expand(4, -1)
    counts = torch.full((4,), 8, dtype=torch.long, device=device)
    labels = torch.randint(0, 2, (4, 3), device=device).float()
    known = torch.ones_like(labels, dtype=torch.bool)
    tokens = torch.randint(1, 8192, (4, 16), device=device)
    token_mask = torch.ones_like(tokens, dtype=torch.bool)
    batch = {"images": images, "mask": mask, "instance_indices": positions, "original_image_counts": counts,
             "labels": labels, "known": known, "watch_token_ids": tokens, "watch_token_mask": token_mask}
    records = []
    teacher_state = None
    teacher = None
    initial_digest = None
    for variant in "ABCD":
        seed_everything(42)
        student = build_model("ct_rate", parameters, variant, device).train()
        current_initial_digest = state_digest(student.state_dict())
        if initial_digest is None:
            initial_digest = current_initial_digest
        assert current_initial_digest == initial_digest
        if variant in "BD":
            seed_before = torch.get_rng_state()
            from exp_8.models import Exp13LCCFAblationModel
            from run_physionet_ct_ich_table2_baselines import CachedFeatureBackbone
            teacher_core = Exp13LCCFAblationModel(
                **parameters, pretrained=False, num_labels=3
            )
            teacher_core.instance_encoder.backbone = CachedFeatureBackbone(
                768, parameters["feature_dim"], parameters["dropout"]
            )
            teacher_core._safe_attention_mask_path = True
            teacher_core.to(device).eval()
            teacher = FrozenMultimodalTeacher(teacher_core).eval()
            teacher_state = state_digest(teacher.model.state_dict())
            torch.set_rng_state(seed_before)
        output, values = transfer_losses(student, batch, variant, teacher, lambda_q=0.01, tau=2.0)
        values["l_total"].backward()
        shared_gradient = student.context_encoder.layers[0].self_attn.in_proj_weight.grad
        image_gradient = student.image_classifiers[0].weight.grad
        assert shared_gradient is not None and torch.isfinite(shared_gradient).all()
        assert image_gradient is not None and torch.isfinite(image_gradient).all()
        if variant in "CD":
            assert student.classifiers[0].weight.grad is not None
        else:
            assert student.classifiers[0].weight.grad is None
        if variant in "BD":
            assert all(parameter.grad is None for parameter in teacher.parameters())
            assert state_digest(teacher.model.state_dict()) == teacher_state
            student_parameter_ids = {id(parameter) for parameter in student.parameters()}
            assert student_parameter_ids.isdisjoint(id(parameter) for parameter in teacher.parameters())
        student.eval()
        with torch.no_grad():
            image_only = student(images, mask, positions, counts)["s_img"]
            report_ignored = student(images, mask, positions, counts, watch_token_ids=tokens, watch_token_mask=token_mask)["s_img"]
        if not torch.isfinite(image_only).all() or not torch.isfinite(report_ignored).all():
            raise AssertionError(f"纯图像前向出现非有限值：{variant}")
        assert torch.allclose(image_only, report_ignored, rtol=0.0, atol=1e-6)
        records.append({"variant": variant, "finite_loss": bool(torch.isfinite(values["l_total"])),
                        "shared_visual_gradient": True, "image_head_gradient": True,
                        "teacher_frozen": variant not in "BD" or all(parameter.grad is None for parameter in teacher.parameters()),
                        "same_student_initialization": current_initial_digest == initial_digest,
                        "image_only_report_invariant": True})
        del student, output, values
        if teacher is not None:
            del teacher
            teacher = None
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    save_json(OUTPUT / "smoke_abcd.json", records)
    print(json.dumps(records, ensure_ascii=False, indent=2))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", choices=DATASETS)
    parser.add_argument("--variant", choices=tuple(VARIANTS))
    parser.add_argument("--fold", type=int, choices=range(1, 6))
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--audit", action="store_true")
    parser.add_argument("--aggregate", action="store_true")
    args = parser.parse_args()
    if args.smoke:
        smoke(args.device)
        return
    if args.audit:
        audit_protocol()
        return
    if getattr(args, "aggregate", False):
        aggregate_results()
        return
    if not all((args.dataset, args.variant, args.fold)):
        parser.error("训练时必须同时指定--dataset、--variant和--fold")
    train_fold(args.dataset, args.variant, args.fold, torch.device(args.device))


if __name__ == "__main__":
    main()
