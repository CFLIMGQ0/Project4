#!/usr/bin/env python3
"""在已有权重和固定删片清单上诊断 ACPE 坐标是否影响分类。"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import numpy as np
import torch
from sklearn.metrics import f1_score

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src/scripts"))

from evaluate_figure1_grid import (  # noqa: E402
    INPUTS,
    batch_for,
    fold_cases,
    load_cases,
    load_model,
    tokenize,
)

CONDITIONS = ("00_B1", "50_B1", "80_B1", "80_B8")
INTERVENTIONS = ("normal", "zero", "reverse", "double", "shuffle")


def replace_context(positioner, intervention: str) -> None:
    original = positioner._contextual_coordinates

    def contextual(features, mask, raw):
        context, eta = original(features, mask, raw)
        if intervention == "normal":
            return context, eta
        residual = context - raw
        if intervention == "zero":
            result = raw
        elif intervention == "reverse":
            result = raw - residual
        elif intervention == "double":
            result = raw + 2.0 * residual
        elif intervention == "shuffle":
            result = raw.clone()
            # 仅打乱有效内部图像的调整量；端点和补齐槽位保持原样。
            for row in range(mask.shape[0]):
                count = int(mask[row].sum())
                if count > 3:
                    perm = torch.arange(count - 2, device=raw.device).flip(0)
                    result[row, 1 : count - 1] += residual[row, 1 : count - 1][perm]
        else:
            raise ValueError(intervention)
        return result * mask.to(result.dtype), eta

    positioner._contextual_coordinates = contextual


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", choices=tuple(INPUTS), required=True)
    parser.add_argument("--fold", type=int, choices=range(1, 6), required=True)
    parser.add_argument("--index-protocol", choices=("reindexed", "original"), default="reindexed")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--limit-cases", type=int, default=0)
    parser.add_argument("--output-root", type=Path, default=ROOT / "outputs/acpe_autoresearch/counterfactual")
    args = parser.parse_args()

    torch.set_num_threads(2)
    torch.backends.mha.set_fastpath_enabled(False)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    test, validation = fold_cases(args.dataset, args.fold)
    if args.limit_cases:
        test = test[: args.limit_cases]
    rows = json.loads((ROOT / "outputs" / INPUTS[args.dataset] / "experiment/samples.json").read_text())
    labels = np.asarray([row["labels"] for row in rows], dtype=np.int64)
    records = load_cases(args.dataset, test)
    eligible = [case for case in test if all(key in records[case]["configurations"] for key in CONDITIONS)]
    if not eligible:
        raise RuntimeError("没有可用于全部诊断条件的病例")
    token_ids, token_mask = tokenize(args.dataset, rows, validation)
    model, thresholds, checkpoint_path, checkpoint_sha256 = load_model(args.dataset, "acpe", args.fold, device)
    if model.apro_positioner is None:
        raise RuntimeError("加载的模型不包含 ACPE")
    positioner = model.apro_positioner
    original_method = positioner._contextual_coordinates
    output = {
        "dataset": args.dataset,
        "fold": args.fold,
        "index_protocol": args.index_protocol,
        "checkpoint": checkpoint_path,
        "checkpoint_sha256": checkpoint_sha256,
        "thresholds": thresholds.tolist(),
        "case_indices": eligible,
        "interventions": list(INTERVENTIONS),
        "conditions": list(CONDITIONS),
        "results": {},
    }

    for condition in CONDITIONS:
        selected = {
            case: np.asarray(records[case]["configurations"][condition]["selected_raw_indices"], dtype=np.int64)
            for case in eligible
        }
        condition_results = {}
        for intervention in INTERVENTIONS:
            positioner._contextual_coordinates = original_method
            replace_context(positioner, intervention)
            probabilities = []
            displacement = []
            with torch.inference_mode():
                for start in range(0, len(eligible), args.batch_size):
                    cases = eligible[start : start + args.batch_size]
                    images, mask, indices, counts = batch_for(cases, selected, records, device)
                    if args.index_protocol == "original":
                        for row, case in enumerate(cases):
                            count = len(selected[case])
                            indices[row, :count] = torch.as_tensor(selected[case], device=device)
                            counts[row] = records[case]["source_count"]
                    values = model(
                        images=images,
                        mask=mask,
                        watch_token_ids=token_ids[cases].to(device),
                        watch_token_mask=token_mask[cases].to(device),
                        instance_indices=indices,
                        original_image_counts=counts,
                    )
                    logits = values["logits"]
                    if not torch.isfinite(logits).all():
                        raise FloatingPointError(f"非有限分类输出：{args.dataset}/{args.fold}/{condition}/{intervention}")
                    probabilities.append(logits.sigmoid().float().cpu().numpy())
                    if intervention == "normal":
                        raw = values["apro_raw_coordinates"]
                        context = values["apro_context_coordinates"]
                        displacement.append((context - raw).abs().masked_select(mask).float().cpu().numpy())
            prob = np.concatenate(probabilities)
            target = labels[eligible]
            pred = prob >= thresholds[None, :]
            condition_results[intervention] = {
                "macro_f1": float(f1_score(target, pred, average="macro", zero_division=0)),
                "per_label_f1": f1_score(target, pred, average=None, zero_division=0).tolist(),
                "probabilities": prob.tolist(),
            }
            if intervention == "normal":
                condition_results[intervention]["mean_absolute_coordinate_adjustment"] = float(
                    np.concatenate(displacement).mean()
                )
        baseline = condition_results["normal"]["macro_f1"]
        for intervention in INTERVENTIONS[1:]:
            condition_results[intervention]["delta_macro_f1"] = (
                condition_results[intervention]["macro_f1"] - baseline
            )
        output["results"][condition] = condition_results
        print(
            f"{args.dataset} fold{args.fold} {args.index_protocol} {condition}: "
            + " ".join(f"{name}={condition_results[name]['macro_f1']:.4f}" for name in INTERVENTIONS),
            flush=True,
        )

    destination = args.output_root / args.index_protocol / args.dataset / f"fold_{args.fold}.json"
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(".tmp")
    temporary.write_text(json.dumps(output, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    temporary.replace(destination)
    print(f"已保存：{destination}", flush=True)


if __name__ == "__main__":
    main()
