#!/usr/bin/env python3
"""使用已有五折权重评估 temp10；保留逐病例、逐标签对的干预前后输出。

在聚合层屏蔽源标签的 Top-5 图像证据，然后重新执行标签推理、文本检索、
门控融合与分类。每个病例内先对同组标签对求平均，再对病例求平均。
该诊断描述预测敏感性，不把共阳性等同于正相关，也不把阴性标签当作无关标签。
"""
from __future__ import annotations

import argparse
import gc
import json
from pathlib import Path

import numpy as np
import torch
from tqdm import tqdm

from evaluate_lccf_public_figures import (
    DATASETS,
    folder_for,
    load_inputs,
    load_model,
    pack,
    reference,
)

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "outputs" / "lccf_temp10_20261001"
PROTOCOL_VERSION = "full_suffix_case_mean_v2"
VARIANTS = {
    "No reasoning": "b_none",
    "Ordinary graph": "b_graph",
    "Hypergraph E=1": "b_hyper1",
    "Hypergraph E=2": "full",
    "Hypergraph E=3": "b_hyper3",
    "Hypergraph E=4": "b_hyper4",
    "Hypergraph E=5": "b_hyper5",
}


@torch.inference_mode()
def evaluate_fold(ds: str, variant: str, fold: int, inputs, device: torch.device, batch_size: int):
    rows, all_y, known, tokens, text_masks, bags = inputs
    folder = folder_for(ds, variant, fold)
    ids, ref_y, ref_p, _ = reference(folder, ds, rows)
    assert np.array_equal(all_y[ids], ref_y) and known[ids].all()
    common_ids, _, _, _ = reference(folder_for(ds, "full", fold), ds, rows)
    assert np.array_equal(ids, common_ids)
    model, digest, config = load_model(folder, ds, variant, device)
    model.eval()

    gathered = {k: [] for k in ("logits", "visual_logits", "changed_logits", "changed_visual_logits", "image_count", "top_indices")}
    for start in range(0, len(ids), batch_size):
        case = ids[start : start + batch_size]
        x, mask, pos, counts, token, text_mask = pack(
            case, bags, tokens, text_masks, device
        )
        context, base_embeds, attention, _ = model.encode_long_mil(x, mask, pos, counts)
        text, tm, _, active = model.text_encoder(token, text_mask, batch_size=len(case), device=device)

        def suffix(embeds):
            retrieved, _ = model._lccf_retrieve_text(embeds, text, tm)
            retrieved = retrieved * active[:, None, None].to(retrieved.dtype)
            fused, _ = model._lccf_fuse(embeds, retrieved, active)
            return model.classify(fused)

        base_logits = suffix(base_embeds)
        visual_logits = model.classify(base_embeds)
        pooled = torch.einsum("bln,bnd->bld", attention, context)
        replay_embeds, _ = model._lccf_reason(pooled)
        torch.testing.assert_close(replay_embeds, base_embeds, atol=1e-5, rtol=1e-5)
        batch_size_actual, n_labels, n_instances = attention.shape
        changed_all, changed_visual_all, top_all = [], [], []
        for source in range(n_labels):
            changed = attention.clone()
            indices = torch.full((batch_size_actual, 5), -1, dtype=torch.long, device=device)
            for bi in range(batch_size_actual):
                k = min(5, int(mask[bi].sum()) - 1)
                if k <= 0:
                    continue
                score = attention[bi, source].masked_fill(~mask[bi], -torch.inf)
                idx = torch.argsort(score, descending=True, stable=True)[:k]
                indices[bi, :k] = idx
                changed[bi, :, idx] = 0
            changed /= changed.sum(-1, keepdim=True).clamp_min(1e-12)
            changed_bag = torch.einsum("bln,bnd->bld", changed, context)
            changed_embeds, _ = model._lccf_reason(changed_bag)
            changed_logits = suffix(changed_embeds)
            changed_visual = model.classify(changed_embeds)
            changed_logits[mask.sum(1) < 2] = torch.nan
            changed_visual[mask.sum(1) < 2] = torch.nan
            changed_all.append(changed_logits)
            changed_visual_all.append(changed_visual)
            top_all.append(indices)
        values = {
            "logits": base_logits, "visual_logits": visual_logits,
            "changed_logits": torch.stack(changed_all, 1),
            "changed_visual_logits": torch.stack(changed_visual_all, 1),
            "image_count": mask.sum(1), "top_indices": torch.stack(top_all, 1),
        }
        for key, value in values.items():
            gathered[key].append(value.cpu().numpy())

    data = {k: np.concatenate(v) for k, v in gathered.items()}
    data.update(case=ids, labels=ref_y)
    replay_p = 1 / (1 + np.exp(-data["logits"]))
    max_error = float(np.abs(replay_p - ref_p).max())
    if max_error > 5e-4:
        raise RuntimeError(f"{ds}/{variant}/{fold}: 原预测复核失败 {max_error}")
    changes = np.abs(data["logits"][:, None, :] - data["changed_logits"])
    y = ref_y.astype(bool)
    off_diagonal = ~np.eye(y.shape[1], dtype=bool)[None]
    co_mask = y[:, :, None] & y[:, None, :] & off_diagonal & np.isfinite(changes)
    non_mask = y[:, :, None] & ~y[:, None, :] & off_diagonal & np.isfinite(changes)
    def case_average(pair_mask):
        counts = pair_mask.sum((1, 2))
        return np.divide(np.where(pair_mask, changes, 0).sum((1, 2)), counts,
                         out=np.full(len(ids), np.nan), where=counts > 0)
    co_cases, non_cases = case_average(co_mask), case_average(non_mask)
    data.update(co_case_mean=co_cases, nonco_case_mean=non_cases)
    co, non = float(np.nanmean(co_cases)), float(np.nanmean(non_cases))
    if not (np.isfinite(co) and np.isfinite(non) and non > 1e-12):
        raise RuntimeError(f"{ds}/{variant}/{fold}: 比值分母或分组样本无效")
    ratio = co / non
    metadata = {
        "dataset": ds,
        "variant": variant,
        "fold": fold,
        "checkpoint": str((folder / "best_model.pt").relative_to(ROOT)),
        "checkpoint_bytes": (folder / "best_model.pt").stat().st_size,
        "checkpoint_mtime_ns": (folder / "best_model.pt").stat().st_mtime_ns,
        "training_protocol_sha256": digest,
        "cases": len(ids),
        "protocol_version": PROTOCOL_VERSION,
        "cases_used": int((data["image_count"] >= 2).sum()),
        "co_cases": int(np.isfinite(co_cases).sum()),
        "nonco_cases": int(np.isfinite(non_cases).sum()),
        "co_pairs": int(co_mask.sum()),
        "nonco_pairs": int(non_mask.sum()),
        "reference_max_probability_error": max_error,
        "test_reference_passed": True,
        "mean_abs_logit_change_coexisting": co,
        "mean_abs_logit_change_noncoexisting": non,
        "coexistence_coupling_ratio": ratio,
        "device": str(device),
        "metric": "共阳性标签对与源阳性/目标阴性标签对的平均绝对 logit 变化之比；先按病例平均，再按组平均",
        "intervention": "固定上下文特征，在全部聚合权重行屏蔽源标签 Top-5 图像并归一化，重新执行标签推理、文本检索、门控和分类",
    }
    output = OUT / "raw" / ds / variant
    output.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(output / f"fold_{fold}.npz", **data)
    (output / f"fold_{fold}.json").write_text(json.dumps(metadata, ensure_ascii=False, indent=2))
    del model
    gc.collect()
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return metadata


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--datasets", nargs="+", choices=list(DATASETS), default=list(DATASETS))
    parser.add_argument("--variants", nargs="+", choices=list(VARIANTS.values()), default=list(VARIANTS.values()))
    parser.add_argument("--folds", nargs="+", type=int, default=[1, 2, 3, 4, 5])
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--threads", type=int, default=2)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    torch.set_num_threads(args.threads)
    torch.backends.mha.set_fastpath_enabled(False)
    torch.manual_seed(2026)
    device = torch.device(args.device)
    OUT.mkdir(parents=True, exist_ok=True)
    all_rows = []
    for ds in args.datasets:
        inputs = load_inputs(ds)
        for variant in args.variants:
            for fold in tqdm(args.folds, desc=f"{ds}/{variant}", mininterval=5):
                target = OUT / "raw" / ds / variant / f"fold_{fold}.json"
                if (args.resume and target.exists() and target.with_suffix(".npz").exists()
                    and json.loads(target.read_text()).get("protocol_version") == PROTOCOL_VERSION):
                    metadata = json.loads(target.read_text())
                else:
                    metadata = evaluate_fold(ds, variant, fold, inputs, device, args.batch_size)
                all_rows.append(metadata)
                print(
                    f"{ds}/{variant}/fold{fold}: "
                    f"co={metadata['mean_abs_logit_change_coexisting']:.6f}, "
                    f"non={metadata['mean_abs_logit_change_noncoexisting']:.6f}, "
                    f"ratio={metadata['coexistence_coupling_ratio']:.4f}",
                    flush=True,
                )
    # 每个工作进程独立写汇总，最终绘图读取全部原始折记录。
    summary = {}
    for ds in args.datasets:
        summary[ds] = {}
        for label, variant in VARIANTS.items():
            rows = [r for r in all_rows if r["dataset"] == ds and r["variant"] == variant]
            if not rows:
                continue
            vals = np.array([r["coexistence_coupling_ratio"] for r in rows], dtype=float)
            co = np.array([r["mean_abs_logit_change_coexisting"] for r in rows], dtype=float)
            non = np.array([r["mean_abs_logit_change_noncoexisting"] for r in rows], dtype=float)
            summary[ds][label] = {
                "variant": variant,
                "mean": float(np.nanmean(vals)),
                "std": float(np.nanstd(vals, ddof=1)) if len(vals) > 1 else 0.0,
                "fold_values": vals.tolist(),
                "co_mean": float(np.nanmean(co)),
                "nonco_mean": float(np.nanmean(non)),
                "folds": len(rows),
            }
    (OUT / ("summary_" + "_".join(args.datasets) + ".json")).write_text(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
