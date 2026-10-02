#!/usr/bin/env python3
"""三种 2026 论文的基础适配；逐数据集/逐折训练并保留原有评价协议。"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import statistics  # 标准库须先于 src/statistics.py 进入 sys.modules。
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "src/scripts"))

from sotas.task2.paper2026_adapted import REGISTRY, build_paper2026_adapted
from run_physionet_ct_ich_table2_baselines import CachedFeatureBackbone

DISPLAY = {
    "unified_multimodal_framework_2026": "Unified Multimodal Framework (2026), adapted",
    "adaptive_multimodal_fusion_2026": "Adaptive Multimodal Fusion (2026), adapted",
    "gcf_net_2026": "GCF-Net (2026), adapted",
}
OUT = ROOT / "outputs/paper_results/new_baselines"


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def save_json(path: Path, data: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


def adjacency(y: np.ndarray, train: list[int]) -> list[list[float]]:
    """只使用训练折构造标签共现关系，阈值与高斯映射不访问验证/测试标签。"""
    observed = y[train].astype(np.float64)
    counts = observed.T @ observed
    diag = np.diag(counts)
    cosine = counts / np.sqrt(np.maximum(diag[:, None] * diag[None, :], 1.))
    filtered = np.where(cosine >= .1, np.exp(-((1 - cosine)**2) / (2 * .5**2)), 0.)
    np.fill_diagonal(filtered, 1.)
    return filtered.tolist()


def common_params(method: str, graph: list[list[float]] | None) -> dict:
    params = {"backbone_name": "convnext_tiny", "freeze_stages": 1, "feature_dim": 512,
              "attn_dim": 256, "hidden_dim": 1024, "dropout": .2, "encoder_chunk_size": 16,
              "text_vocab_size": 8192, "text_embed_dim": 128,
              "textcnn_kernel_sizes": [2, 3, 4], "num_heads": 4}
    if method == "unified_multimodal_framework_2026":
        params.update(contrast_temperature=.07, contrastive_weight=.1)
    if method == "gcf_net_2026":
        params["label_adjacency"] = graph
    return params


def inputs(dataset: str):
    if dataset == "ct_rate":
        import run_ctrate_680_all_models as adapter
        adapter.configure_adapters()
        return adapter, adapter, "findings_masked"
    if dataset == "mr_rate_1k":
        import run_mrrate_1k_all_models as adapter
        adapter.configure()
        return adapter, adapter.core, "findings_masked"
    if dataset == "merlin_1k":
        import run_merlin_1k_all_models as adapter
        adapter.configure_adapters()
        return adapter, adapter, "findings_masked"
    raise ValueError(dataset)


def run_public(args: argparse.Namespace) -> None:
    import torch
    from types import SimpleNamespace

    torch.set_num_threads(2)
    torch.backends.mha.set_fastpath_enabled(False)
    if torch.cuda.is_available():
        torch.cuda.set_per_process_memory_fraction(float(os.environ.get("PAPER_MEMORY_FRACTION", ".4")))
    adapter, core, text_field = inputs(args.dataset)
    fusion = core.fusion_base
    rows, folds, y = adapter.read_inputs()
    fold = args.fold - 1
    train = [i for k, group in enumerate(folds["folds"]) if k not in (fold, (fold + 1) % 5)
             for i in group]
    graph = adjacency(y, train) if args.method == "gcf_net_2026" else None
    params = common_params(args.method, graph)
    out = OUT / args.dataset
    folder = out / f"fold_{args.fold}" / args.method
    sources = [Path(__file__), ROOT / "src/sotas/task2/paper2026_adapted.py",
               ROOT / "outputs" / {"ct_rate": "ct_rate_680", "mr_rate_1k": "mr_rate_1k",
                                   "merlin_1k": "merlin_1k"}[args.dataset] / "experiment" / "samples.json",
               ROOT / "outputs" / {"ct_rate": "ct_rate_680", "mr_rate_1k": "mr_rate_1k",
                                   "merlin_1k": "merlin_1k"}[args.dataset] / "experiment" / "patient_folds.json"]
    protocol = {"dataset": args.dataset, "method": args.method, "scope": "基础任务适配，非原论文完整架构复现",
                "fold": args.fold, "train_indices": train, "test_indices": folds["folds"][fold],
                "validation_indices": folds["folds"][(fold+1)%5],
                "text_input": "none" if args.method == "gcf_net_2026" else text_field,
                "parameters": params, "training_settings": adapter.SETTINGS,
                "source_sha256": {str(p.relative_to(ROOT)): digest(p) for p in sources},
                "primary_metric": "test Macro-F1; validation-tuned thresholds; 5 folds, sample SD"}
    protocol["protocol_sha256"] = hashlib.sha256(json.dumps(protocol, sort_keys=True).encode()).hexdigest()
    saved = folder / "protocol.json"
    if saved.exists() and json.loads(saved.read_text()) != protocol and (folder / "completed.json").exists():
        raise RuntimeError(f"现有折次协议不同，禁止覆盖：{folder}")
    save_json(saved, protocol)
    if (folder / "completed.json").exists():
        return

    def build(key, model_params):
        values = dict(model_params)
        weights = {name.removesuffix("_weight"): values.pop(name) for name in list(values)
                   if name.endswith("_weight")}
        model = build_paper2026_adapted(key, **values, pretrained=False, num_labels=y.shape[1])
        model.instance_encoder.backbone = CachedFeatureBackbone(768, values["feature_dim"], values["dropout"])
        model._ct_model_key = key
        return model, weights

    fusion.MODELS = {args.method: DISPLAY[args.method]}
    fusion.SETTINGS = adapter.SETTINGS
    fusion.LABEL_NAMES = adapter.LABELS
    fusion.build_model = build
    fusion.forward_batch = core.forward_batch
    bags = adapter.load_features(rows)
    ids = np.arange(len(rows), dtype=np.int64)
    tokens, masks = fusion.encode_descriptions({r["case_index"]: r[text_field] for r in rows}, ids)
    fusion.train_fold(SimpleNamespace(output_dir=out), args.method, fold, ids, bags, y, folds,
                      tokens, masks, params, protocol["protocol_sha256"])
    result = folder / "test_metrics.json"
    if not result.exists() or not (folder / "test_predictions.csv").exists():
        raise RuntimeError(f"训练结束但结果不全：{folder}")
    print(f"完成 {args.dataset} {args.method} fold {args.fold}: "
          f"Macro-F1={json.loads(result.read_text())['macro_f1']:.5f}", flush=True)


def run_amos(args: argparse.Namespace) -> None:
    import run_amos_mm_all_models as core
    import torch

    torch.set_num_threads(2)
    torch.backends.mha.set_fastpath_enabled(False)
    if torch.cuda.is_available():
        torch.cuda.set_per_process_memory_fraction(float(os.environ.get("PAPER_MEMORY_FRACTION", ".4")))
    rows = json.loads((core.INPUT / "samples.json").read_text())
    splits = json.loads((core.INPUT / "splits.json").read_text())
    fold = args.fold - 1
    validation = set(splits["validation_folds"][fold])
    excluded = core.image_exclusion_indices(rows)
    train = sorted(set(splits["development"]) - validation - excluded)
    labels = np.asarray([r["labels"][:7] for r in rows], dtype=np.int64)
    known = np.asarray([r["known_mask"][:7] for r in rows], dtype=bool)
    graph = adjacency(np.where(known, labels, 0), train) if args.method == "gcf_net_2026" else None
    params = common_params(args.method, graph)
    out = OUT / "amos_mm"
    folder = out / "7_labels" / f"fold_{args.fold}" / args.method
    sources = [Path(__file__), ROOT / "src/sotas/task2/paper2026_adapted.py",
               core.INPUT / "samples.json", core.INPUT / "splits.json",
               core.INPUT / "image_exclusions.json"]
    protocol = {"dataset": "amos_mm", "method": args.method,
                "scope": "基础任务适配，非原论文完整架构复现", "fold": args.fold,
                "train_indices": train, "validation_indices": sorted(validation - excluded),
                "test_indices": splits["test"], "text_input": "none" if args.method == "gcf_net_2026" else "masked report",
                "parameters": params, "training_settings": core.SETTINGS,
                "source_sha256": {str(p.relative_to(ROOT)): digest(p) for p in sources},
                "metric": "fixed 200-case manual test Macro-F1; development-fold validation thresholds"}
    protocol["protocol_sha256"] = hashlib.sha256(json.dumps(protocol, sort_keys=True).encode()).hexdigest()
    saved = folder / "protocol.json"
    if saved.exists() and json.loads(saved.read_text()) != protocol and (folder / "result.json").exists():
        raise RuntimeError(f"现有折次协议不同，禁止覆盖：{folder}")
    save_json(saved, protocol)
    if (folder / "result.json").exists() and (folder / "test_predictions.npz").exists():
        return

    def build(key, count, vocabulary_size=8192):
        del vocabulary_size
        values = dict(params)
        weights = {name.removesuffix("_weight"): values.pop(name) for name in list(values)
                   if name.endswith("_weight")}
        model = build_paper2026_adapted(key, **values, pretrained=False, num_labels=count)
        model.instance_encoder.backbone = CachedFeatureBackbone(768, values["feature_dim"], values["dropout"])
        return model, weights

    core.OUT = out
    core.MODELS = {args.method: DISPLAY[args.method]}
    core.IMAGE_MODELS = {}
    core.TEXT_MODELS = {}
    core.category = lambda _: "纯图像" if args.method == "gcf_net_2026" else "多模态基线"
    core.parameters = lambda _: params
    core.build_model = build
    bags = core.load_features(rows)
    core.train_job({"labels": 7, "model": args.method, "fold": fold}, rows, splits, bags,
                   protocol["protocol_sha256"], f"paper_{args.method}_{args.fold}")
    result = folder / "result.json"
    if not result.exists() or not (folder / "test_predictions.npz").exists():
        raise RuntimeError(f"训练结束但结果不全：{folder}")
    print(f"完成 AMOS-MM {args.method} fold {args.fold}: "
          f"Macro-F1={json.loads(result.read_text())['macro_f1']:.5f}", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", choices=("ct_rate", "amos_mm", "mr_rate_1k", "merlin_1k"), required=True)
    parser.add_argument("--method", choices=tuple(REGISTRY), required=True)
    parser.add_argument("--fold", type=int, choices=range(1, 6), required=True)
    args = parser.parse_args()
    if args.dataset == "amos_mm":
        run_amos(args)
    else:
        run_public(args)


if __name__ == "__main__":
    main()
