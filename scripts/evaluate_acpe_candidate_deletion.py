#!/usr/bin/env python3
"""在同一病例和固定删片清单上比较 ACPE 研究候选与原模型。"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import numpy as np
import torch
from sklearn.metrics import f1_score

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "src/scripts"))

from evaluate_figure1_grid import INPUTS, batch_for, fold_cases, load_cases, tokenize
from exp_8.models import Exp12AProCoPEWatchCrossAttentionTextCNNModel
from run_physionet_ct_ich_table2_baselines import CachedFeatureBackbone
from evaluate_public_block3 import normalize_context, position_metrics

CONDITIONS = ("00_B1", "50_B1", "80_B1", "80_B8")
VARIANTS = ("centroid", "attention_bias", "confidence_gate", "confidence_half", "eta_scale", "block_aug")
BASE_ROOTS = {
    "ct_rate": ROOT / "outputs/ct_rate_680/all_models_fivefold",
    "amos_mm": ROOT / "outputs/amos_mm/all_models",
    "mr_rate_1k": ROOT / "outputs/mr_rate_1k/all_models_fivefold",
}
COUNTS = {"ct_rate": 3, "amos_mm": 7, "mr_rate_1k": 4}


def model_folder(dataset: str, fold: int, variant: str):
    root = BASE_ROOTS[dataset] if variant == "base" else ROOT / "outputs/acpe_autoresearch/training" / variant / dataset
    return root / ("7_labels" if dataset == "amos_mm" else "") / f"fold_{fold}" / "amef_multimodal"


def load_model(dataset: str, fold: int, variant: str, device: torch.device):
    folder = model_folder(dataset, fold, variant)
    config = json.loads((folder / "config.json").read_text())
    params = dict(config.get("model", config.get("parameters", {})))
    params.pop("ct_attention_variant", None)
    for key in list(params):
        if key.endswith("_weight"):
            params.pop(key)
    if variant == "base":
        model = Exp12AProCoPEWatchCrossAttentionTextCNNModel(
            **params, pretrained=False, num_labels=COUNTS[dataset]
        )
        model.instance_encoder.backbone = CachedFeatureBackbone(768, params["feature_dim"], params["dropout"])
    elif variant in {"confidence_gate", "confidence_half"}:
        from run_acpe_confidence_gate import confidence_build
        model, _ = confidence_build(dataset, variant, COUNTS[dataset])
    elif variant == "eta_scale":
        from run_acpe_eta_scale import eta_build
        model, _ = eta_build(dataset, variant, COUNTS[dataset])
    elif variant == "block_aug":
        from run_acpe_block_augmentation import block_build
        model, _ = block_build(dataset, variant, COUNTS[dataset])
    else:
        from run_acpe_system_interaction import build_model
        model, _ = build_model(dataset, variant, COUNTS[dataset])
    checkpoint = torch.load(folder / "best_model.pt", map_location="cpu", weights_only=False)
    model.load_state_dict(checkpoint["state_dict"], strict=True)
    thresholds = checkpoint["thresholds"]
    if isinstance(thresholds, dict):
        thresholds = list(thresholds.values())
    return model.to(device).eval(), np.asarray(thresholds, dtype=np.float32)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", choices=tuple(INPUTS), required=True)
    parser.add_argument("--fold", type=int, choices=range(1, 6), required=True)
    parser.add_argument("--variant", choices=VARIANTS, required=True)
    parser.add_argument("--index-protocol", choices=("reindexed", "original"), default="reindexed")
    parser.add_argument("--batch-size", type=int, default=32)
    args = parser.parse_args()

    torch.set_num_threads(2)
    torch.backends.mha.set_fastpath_enabled(False)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    test, validation = fold_cases(args.dataset, args.fold)
    rows = json.loads((ROOT / "outputs" / INPUTS[args.dataset] / "experiment/samples.json").read_text())
    labels = np.asarray([row["labels"] for row in rows], dtype=np.int64)
    records = load_cases(args.dataset, test)
    cases = [case for case in test if all(key in records[case]["configurations"] for key in CONDITIONS)]
    token_ids, token_mask = tokenize(args.dataset, rows, validation)
    models = {name: load_model(args.dataset, args.fold, name, device) for name in ("base", args.variant)}
    output = {
        "dataset": args.dataset,
        "fold": args.fold,
        "variant": args.variant,
        "index_protocol": args.index_protocol,
        "case_indices": cases,
        "results": {},
    }
    for condition in CONDITIONS:
        selected = {case: np.asarray(records[case]["configurations"][condition]["selected_raw_indices"], dtype=np.int64)
                    for case in cases}
        measurements = {}
        for name, (model, thresholds) in models.items():
            probabilities, accuracies, gates, scales = [], [], [], []
            with torch.inference_mode():
                for start in range(0, len(cases), args.batch_size):
                    batch_cases = cases[start : start + args.batch_size]
                    images, mask, indices, counts = batch_for(batch_cases, selected, records, device)
                    if args.index_protocol == "original":
                        for row, case in enumerate(batch_cases):
                            count = len(selected[case])
                            indices[row, :count] = torch.as_tensor(selected[case], device=device)
                            counts[row] = records[case]["source_count"]
                    values = model(
                        images=images, mask=mask,
                        watch_token_ids=token_ids[batch_cases].to(device),
                        watch_token_mask=token_mask[batch_cases].to(device),
                        instance_indices=indices, original_image_counts=counts,
                    )
                    probabilities.append(values["logits"].sigmoid().float().cpu().numpy())
                    if "acpe_confidence_gate" in values:
                        gates.extend(values["acpe_confidence_gate"].float().cpu().tolist())
                    if "acpe_eta_scale" in values:
                        scales.extend(values["acpe_eta_scale"].float().cpu().tolist())
                    context = values["apro_context_coordinates"]
                    for row, case in enumerate(batch_cases):
                        truth = selected[case].astype(np.float64) / (records[case]["source_count"] - 1)
                        prediction, _, _ = normalize_context(context[row, :len(selected[case])].float().cpu().numpy())
                        accuracies.append(position_metrics(prediction, truth)["Acc@0.05"])
            prob = np.concatenate(probabilities)
            measurements[name] = {
                "macro_f1": float(f1_score(labels[cases], prob >= thresholds[None, :], average="macro", zero_division=0)),
                "acc005": float(np.nanmean(accuracies)),
                "mean_gate": float(np.mean(gates)) if gates else None,
                "mean_eta_scale": float(np.mean(scales)) if scales else None,
                "thresholds": thresholds.tolist(),
                "probabilities": prob.tolist(),
            }
        measurements["delta_macro_f1"] = measurements[args.variant]["macro_f1"] - measurements["base"]["macro_f1"]
        measurements["delta_acc005"] = measurements[args.variant]["acc005"] - measurements["base"]["acc005"]
        output["results"][condition] = measurements
        print(f"{args.dataset} fold{args.fold} {args.variant} {args.index_protocol} {condition}: "
              f"ΔF1={measurements['delta_macro_f1']:+.4f}, ΔAcc={measurements['delta_acc005']:+.4f}", flush=True)

    destination = ROOT / "outputs/acpe_autoresearch/deletion" / args.index_protocol / args.variant / args.dataset / f"fold_{args.fold}.json"
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(".tmp")
    temporary.write_text(json.dumps(output, ensure_ascii=False, indent=2) + "\n")
    temporary.replace(destination)


if __name__ == "__main__":
    main()
