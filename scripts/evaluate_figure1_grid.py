#!/usr/bin/env python3
"""固定 ACPE/Original PE 权重，在共享 B-block 掩码下计算分类及位置恢复。"""

from __future__ import annotations

import argparse
import gzip
import hashlib
import json
from pathlib import Path
import sys

import numpy as np
import torch
from sklearn.metrics import f1_score

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "src/scripts"))

from exp_8.models import Exp12AProCoPEWatchCrossAttentionTextCNNModel
from run_physionet_ct_ich_table2_baselines import CachedFeatureBackbone
from evaluate_public_block3 import normalize_context, position_metrics
from paper_block_deletion import RATIOS, BLOCKS

INPUTS = {"ct_rate": "ct_rate_680", "amos_mm": "amos_mm", "mr_rate_1k": "mr_rate_1k"}
MASK_ROOT = ROOT / "outputs/paper_results/figure1_masks"
FEATURE_ROOT = ROOT / "outputs/paper_results/figure1_features"
OUT = ROOT / "outputs/paper_results/figure1_grid"


def fold_cases(dataset: str, fold: int) -> tuple[list[int], list[int]]:
    input_dir = ROOT / "outputs" / INPUTS[dataset] / "experiment"
    if dataset == "amos_mm":
        split = json.loads((input_dir / "splits.json").read_text())
        return list(map(int, split["test"])), list(map(int, split["validation_folds"][fold-1]))
    groups = json.loads((input_dir / "patient_folds.json").read_text())["folds"]
    return list(map(int, groups[fold-1])), list(map(int, groups[fold % 5]))


def model_folder(dataset: str, variant: str, fold: int) -> Path:
    if dataset == "ct_rate":
        folder = ROOT / "outputs/ct_rate_680/position_baselines" / ("apro_full" if variant == "acpe" else "original_pe")
    elif dataset == "amos_mm":
        folder = ROOT / "outputs/amos_mm"
        folder /= "all_models/7_labels" if variant == "acpe" else "position_replacements_7_labels/original_pe/7_labels"
    else:
        folder = ROOT / "outputs/mr_rate_1k"
        folder /= "all_models_fivefold" if variant == "acpe" else "position_replacements/original_pe"
    return folder / f"fold_{fold}" / "amef_multimodal"


def load_model(dataset: str, variant: str, fold: int, device: torch.device):
    folder = model_folder(dataset, variant, fold)
    path = folder / "best_model.pt"
    config = json.loads((folder / "config.json").read_text())
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    params = dict(config.get("model", config.get("parameters", {})))
    params.pop("ct_attention_variant", None)
    for key in list(params):
        if key.endswith("_weight"):
            params.pop(key)
    model = Exp12AProCoPEWatchCrossAttentionTextCNNModel(
        **params, pretrained=False, num_labels={"ct_rate": 3, "amos_mm": 7, "mr_rate_1k": 4}[dataset])
    model.instance_encoder.backbone = CachedFeatureBackbone(768, params["feature_dim"], params["dropout"])
    model.load_state_dict(checkpoint["state_dict"], strict=True)
    model.to(device).eval()
    thresholds = checkpoint.get("thresholds")
    if isinstance(thresholds, dict):
        thresholds = list(thresholds.values())
    return model, np.asarray(thresholds, dtype=np.float32), str(path.relative_to(ROOT)), hashlib.sha256(path.read_bytes()).hexdigest()


def tokenize(dataset: str, rows: list[dict], validation: list[int]):
    if dataset == "amos_mm":
        from run_amos_mm_all_models import tokens_for
        return tokens_for("amef_multimodal", rows, np.asarray(validation, dtype=np.int64), 7)[:2]
    from run_cq500_table2_multimodal_baselines import SETTINGS, encode_descriptions
    SETTINGS["text_max_length"] = 512
    return encode_descriptions({row["case_index"]: row["findings_masked"] for row in rows},
                               np.arange(len(rows), dtype=np.int64))


def load_cases(dataset: str, case_indices: list[int]) -> dict[int, dict]:
    records = {}
    for case in case_indices:
        mask_path = MASK_ROOT / dataset / f"{case:04d}.json.gz"
        feature_path = FEATURE_ROOT / dataset / f"{case:04d}.npz"
        if not feature_path.exists():
            raise FileNotFoundError(f"缺少完整网格冻结特征：{feature_path}")
        with gzip.open(mask_path, "rt", encoding="utf-8") as stream:
            manifest = json.load(stream)
        with np.load(feature_path, allow_pickle=False) as cache:
            if str(cache["mask_manifest_sha256"]) != hashlib.sha256(mask_path.read_bytes()).hexdigest():
                raise ValueError(f"特征与删片清单不匹配：{feature_path}")
            records[case] = {"features": cache["features"].astype(np.float32),
                             "indices": cache["source_indices"].astype(np.int64),
                             "source_count": int(cache["original_count"]),
                             "configurations": manifest["configurations"]}
    return records


def batch_for(cases: list[int], selected: dict[int, np.ndarray], records: dict[int, dict], device: torch.device):
    length = max(len(selected[case]) for case in cases)
    images = torch.zeros(len(cases), length, 768, 1, 1, dtype=torch.float32)
    mask = torch.zeros(len(cases), length, dtype=torch.bool)
    positions = torch.full((len(cases), length), -1, dtype=torch.long)
    counts = torch.zeros(len(cases), dtype=torch.long)
    for j, case in enumerate(cases):
        raw = selected[case]
        source = records[case]["indices"]
        local = np.searchsorted(source, raw)
        if not np.array_equal(source[local], raw):
            raise ValueError(f"源特征缺少选中切片：case={case}")
        n = len(raw)
        images[j, :n, :, 0, 0] = torch.from_numpy(records[case]["features"][local])
        mask[j, :n] = True
        positions[j, :n] = torch.arange(n)
        counts[j] = n
    return images.to(device), mask.to(device), positions.to(device), counts.to(device)


def save_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(".tmp")
    temp.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    temp.replace(path)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", choices=tuple(INPUTS), required=True)
    parser.add_argument("--fold", type=int, choices=range(1, 6), required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--limit-cases", type=int, default=0)
    parser.add_argument("--only-ratio", type=int, choices=RATIOS)
    parser.add_argument("--only-block", type=int, choices=BLOCKS)
    parser.add_argument("--output-root", type=Path, default=OUT)
    args = parser.parse_args()
    torch.set_num_threads(2)
    torch.backends.mha.set_fastpath_enabled(False)
    device = torch.device(args.device)
    test, validation = fold_cases(args.dataset, args.fold)
    if args.limit_cases:
        test = test[:args.limit_cases]
    rows = json.loads((ROOT / "outputs" / INPUTS[args.dataset] / "experiment/samples.json").read_text())
    labels = np.asarray([row["labels"] for row in rows], dtype=np.int64)
    records = load_cases(args.dataset, test)
    keys = [f"{ratio:02d}_B{blocks}" for ratio in RATIOS
            for blocks in ((1,) if ratio == 0 else BLOCKS)]
    common_cases = [case for case in test
                    if all(key in records[case]["configurations"] for key in keys)]
    if not common_cases:
        raise RuntimeError(f"{args.dataset}/fold{args.fold} 没有全网格共同可评估病例")
    token_ids, token_mask = tokenize(args.dataset, rows, validation)
    models = {variant: load_model(args.dataset, variant, args.fold, device)
              for variant in ("acpe", "original_pe")}
    output = args.output_root / args.dataset / f"fold_{args.fold}.json"
    existing = json.loads(output.read_text()) if output.exists() else None
    checkpoint_sources = {key: {"path": value[2], "sha256": value[3],
                                "thresholds": value[1].tolist()} for key, value in models.items()}
    if existing:
        if (existing["checkpoints"] != checkpoint_sources
                or existing["test_indices"] != test
                or existing.get("common_case_indices") != common_cases):
            raise RuntimeError(f"已有 Figure 1 结果协议不同：{output}")
        results = existing["results"]
    else:
        results = {}
    for ratio in RATIOS:
        if args.only_ratio is not None and ratio != args.only_ratio:
            continue
        for blocks in ((1,) if ratio == 0 else BLOCKS):
            if args.only_block is not None and blocks != args.only_block:
                continue
            key = f"{ratio:02d}_B{blocks}"
            if key in results:
                continue
            eligible = common_cases
            if not eligible:
                results[key] = {"ratio": ratio, "B": blocks, "num_cases": 0,
                                "status": "NO_ELIGIBLE_CASES"}
                continue
            selected = {case: np.asarray(records[case]["configurations"][key]["selected_raw_indices"], dtype=np.int64)
                        for case in eligible}
            probabilities = {variant: [] for variant in models}
            accuracies = {variant: [] for variant in models}
            valid_recovery = {variant: 0 for variant in models}
            with torch.inference_mode():
                for start in range(0, len(eligible), args.batch_size):
                    cases = eligible[start:start+args.batch_size]
                    images, mask, positions, counts = batch_for(cases, selected, records, device)
                    for variant, (model, _, _, _) in models.items():
                        output_values = model(images=images, mask=mask,
                            watch_token_ids=token_ids[cases].to(device),
                            watch_token_mask=token_mask[cases].to(device),
                            instance_indices=positions, original_image_counts=counts)
                        logits = output_values["logits"]
                        if not torch.isfinite(logits).all():
                            raise FloatingPointError(f"非有限分类输出：{args.dataset}/{args.fold}/{key}/{variant}")
                        probabilities[variant].append(logits.sigmoid().float().cpu().numpy())
                        context = output_values.get("apro_context_coordinates") if variant == "acpe" else None
                        for j, case in enumerate(cases):
                            raw = selected[case]
                            truth = raw.astype(np.float64) / float(records[case]["source_count"]-1)
                            if variant == "acpe":
                                if context is None:
                                    raise RuntimeError("ACPE 输出缺少 contextual coordinate")
                                prediction, status, _ = normalize_context(context[j, :len(raw)].float().cpu().numpy())
                            else:
                                prediction = np.linspace(0., 1., len(raw), dtype=np.float64)
                                status = "original_pe_uniform"
                            acc = position_metrics(prediction, truth)["Acc@0.05"]
                            accuracies[variant].append(acc)
                            valid_recovery[variant] += int(np.isfinite(acc))
            targets = labels[eligible]
            f1, acc005 = {}, {}
            for variant, (_, thresholds, _, _) in models.items():
                prob = np.concatenate(probabilities[variant])
                f1[variant] = float(f1_score(targets, prob >= thresholds[None], average="macro", zero_division=0))
                acc005[variant] = float(np.nanmean(accuracies[variant]))
            results[key] = {"ratio": ratio, "B": blocks, "num_cases": len(eligible),
                            "case_indices": eligible, "seed": 42,
                            "acpe_f1": f1["acpe"], "original_f1": f1["original_pe"],
                            "delta_f1": f1["acpe"]-f1["original_pe"],
                            "acpe_acc005": acc005["acpe"], "original_acc005": acc005["original_pe"],
                            "delta_acc005": acc005["acpe"]-acc005["original_pe"],
                            "valid_recovery": valid_recovery,
                            "case_acc005": {variant: dict(zip(eligible, accuracies[variant])) for variant in models}}
            save_json(output, {"dataset": args.dataset, "fold": args.fold, "test_indices": test,
                               "common_case_indices": common_cases,
                               "validation_indices": validation, "checkpoints": checkpoint_sources,
                               "protocol": "仅评估该折全65种网格条件共同可评估病例；完整原序列先按同一B-block清单删片，后均匀采样至多64张；短序列掩码补齐；两个模型均只输入重编号槽位。Acc@0.05沿用内部切片绝对归一化误差≤0.05的病例均值，折内取nanmean。",
                               "results": results})
            print(f"{args.dataset} fold{args.fold} {key}: N={len(eligible)} ΔF1={results[key]['delta_f1']:.4f} ΔAcc={results[key]['delta_acc005']:.4f}", flush=True)


if __name__ == "__main__":
    main()
