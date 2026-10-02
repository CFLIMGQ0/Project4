#!/usr/bin/env python3
"""AA-Mini 表2--8的统一五折训练入口。

每次进程只运行一个配置和一个折次，以便外部调度器在显存允许时并行
放置任务。输入、标签、病例划分和原始特征完全复用匿名所见版本的表1。
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src/scripts"))

import run_abdomenatlas3_anonymous_table1 as anonymous

CORE = anonymous.base
INPUT = anonymous.INPUT
OUT = ROOT / "outputs/abdomenatlas3_mini/table2_8_fivefold_anonymous_findings"
MODEL_KEY = "amef_multimodal"

# Full ACPE/LCCF is already present in the anonymous-findings Table-1 run.
# The other entries are trained here; names match the rows in Tables 2--8.
VARIANTS = {
    "position_original": {"position_variant": "original_pe"},
    "position_comrope": {"position_variant": "comrope_ap"},
    "position_videorope": {"position_variant": "videorope"},
    "position_path": {"position_variant": "path"},
    "position_dape": {"position_variant": "dape_v2_kerple"},
    "position_absolute": {"position_variant": "apro_absolute_only"},
    "position_relative": {"position_variant": "apro_relative_only"},
    "desc_u": {"apro_transition_groups": [1], "apro_transition_include_gap": False},
    "desc_u_gap": {"apro_transition_groups": [1], "apro_transition_include_gap": True},
    "desc_signed_gap": {"apro_transition_groups": [1, 2, 3], "apro_transition_include_gap": True},
    "desc_mag_gap": {"apro_transition_groups": [1, 4, 5], "apro_transition_include_gap": True},
    "desc_signed_inter_gap": {"apro_transition_groups": [1, 2, 3, 6], "apro_transition_include_gap": True},
    "desc_mag_inter_gap": {"apro_transition_groups": [1, 4, 5, 6], "apro_transition_include_gap": True},
    "desc_no_u": {"apro_transition_groups": [2, 3, 4, 5, 6], "apro_transition_include_gap": True},
    "agg_mean": {"lccf_pooling": "mean"},
    "agg_max": {"lccf_pooling": "max"},
    "agg_shared": {"lccf_pooling": "shared_attention"},
    "reason_none": {"lccf_reasoning": "none"},
    "reason_graph": {"lccf_reasoning": "ordinary_graph"},
    "reason_h1": {"lccf_hypergraph_edges": 1},
    "reason_h3": {"lccf_hypergraph_edges": 3},
    "reason_h4": {"lccf_hypergraph_edges": 4},
    "reason_h5": {"lccf_hypergraph_edges": 5},
    "retrieval_mean": {"lccf_text_retrieval": "shared_mean"},
    "retrieval_shared": {"lccf_text_retrieval": "shared_query"},
    "retrieval_no_identity": {"lccf_text_retrieval": "label_query_no_identity"},
    "fusion_direct": {"lccf_fusion": "direct"},
    "fusion_fixed": {"lccf_fusion": "fixed"},
    "fusion_label": {"lccf_fusion": "label_constant"},
    "fusion_exam": {"lccf_fusion": "exam_shared"},
}

DISPLAY = {
    "position_original": "Original PE", "position_comrope": "ComRoPE-AP",
    "position_videorope": "VideoRoPE", "position_path": "PaTH",
    "position_dape": "DAPE V2-KERPLE", "desc_u": "u only",
    "position_absolute": "Absolute route only", "position_relative": "Relative route only",
    "desc_u_gap": "u + gap", "desc_signed_gap": "u + signed + gap",
    "desc_mag_gap": "u + magnitude + gap", "desc_signed_inter_gap": "u + signed + interaction + gap",
    "desc_mag_inter_gap": "u + magnitude + interaction + gap", "desc_no_u": "w/o u",
    "agg_mean": "Mean pooling", "agg_max": "Max pooling", "agg_shared": "Shared attention",
    "reason_none": "No reasoning", "reason_graph": "Ordinary label graph",
    "reason_h1": "Hypergraph E=1", "reason_h3": "Hypergraph E=3",
    "reason_h4": "Hypergraph E=4", "reason_h5": "Hypergraph E=5",
    "retrieval_mean": "Shared mean pooling", "retrieval_shared": "Shared-query cross-attention",
    "retrieval_no_identity": "Label-wise query, w/o identity",
    "fusion_direct": "Direct addition", "fusion_fixed": "Fixed gate = 0.5",
    "fusion_label": "Label-wise constant gate", "fusion_exam": "Examination-wise shared gate",
}


def params_for(variant: str) -> dict:
    from task3_apro_cope_ablation_scheduler import base_model_params
    from yaml import safe_load

    params = base_model_params("apro_full")
    if variant.startswith(("agg_", "reason_", "retrieval_", "fusion_")):
        params.update(
            lccf_pooling="label_wise_attention", lccf_reasoning="hypergraph",
            lccf_hypergraph_edges=2, lccf_text_retrieval="label_query_identity",
            lccf_fusion="exam_label",
        )
    main = safe_load((ROOT / "src/configs/task3/t3_main_model.yaml").read_text())
    params["label_query_consistency_weight"] = main["model"]["params"]["label_query_consistency_weight"]
    params.update(VARIANTS[variant])
    # The alternative positional baselines are implemented by replacing the
    # slice self-attention after constructing the common no-PE backbone.
    if variant.startswith("position_"):
        params["position_variant"] = "original_pe" if variant == "position_original" else "no_pe"
    return params


def protocol(variant: str) -> dict:
    paths = [Path(__file__), Path(anonymous.__file__), Path(CORE.__file__),
             ROOT / "src/exp_8/models.py", ROOT / "src/exp_8/position_baselines.py",
             ROOT / "src/exp_4/models.py", INPUT / "samples.json",
             INPUT / "patient_folds.json", INPUT / "preparation_protocol.json",
             INPUT / "masking_protocol.json"]
    value = {
        "dataset": "AbdomenAtlas 3.0 Mini",
        "variant": variant, "display": DISPLAY[variant], "labels": CORE.LABELS,
        "model_parameters": params_for(variant),
        "scope": "仅改变表2--8对应的一个ACPE或LCCF配置；输入、病例划分、训练循环和评价口径固定",
        "text": "anonymous_findings固定文本；不输入答案、结论或其他标签字段",
        "split": "沿用AA-Mini匿名所见表1的五折，训练3折、验证1折、测试1折",
        "selection": "验证集分类损失选择checkpoint，测试集只用于最终评分",
        "seed": "沿用表1五折种子142/242/342/442/542",
        "source_sha256": {str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest() for p in paths},
    }
    value["protocol_sha256"] = hashlib.sha256(json.dumps(value, ensure_ascii=False, sort_keys=True).encode()).hexdigest()
    return value


def replace_position_attention(model, variant: str):
    from exp_8.position_baselines import replace_slice_attention
    mapping = {
        "position_comrope": "comrope_ap", "position_videorope": "videorope",
        "position_path": "path", "position_dape": "dape_v2_kerple",
    }
    if variant in mapping:
        replace_slice_attention(model, mapping[variant])
    return model


def build_model(key, params):
    import torch
    from exp_8.models import Exp12AProCoPEWatchCrossAttentionTextCNNModel, Exp13LCCFAblationModel
    from run_physionet_ct_ich_table2_baselines import CachedFeatureBackbone

    assert key == MODEL_KEY
    values = dict(params)
    weights = {name.removesuffix("_weight"): values.pop(name) for name in list(values) if name.endswith("_weight")}
    variant = CURRENT_VARIANT
    if variant.startswith("agg_") or variant.startswith("reason_") or variant.startswith("retrieval_") or variant.startswith("fusion_"):
        model = Exp13LCCFAblationModel(**values, pretrained=False, num_labels=3)
    else:
        model = Exp12AProCoPEWatchCrossAttentionTextCNNModel(**values, pretrained=False, num_labels=3)
    model.instance_encoder.backbone = CachedFeatureBackbone(768, values["feature_dim"], values["dropout"])
    model = replace_position_attention(model, variant)
    model._ct_model_key = key
    torch.backends.mha.set_fastpath_enabled(False)
    return model, weights


CURRENT_VARIANT = ""


def configure(variant: str):
    global CURRENT_VARIANT
    CURRENT_VARIANT = variant
    CORE.INPUT = INPUT
    CORE.LABELS = anonymous.base.LABELS
    CORE.OUT = OUT / variant
    CORE.MODELS = {MODEL_KEY: DISPLAY[variant]}
    CORE.IMAGE_MODELS, CORE.TEXT_MODELS, CORE.FUSION_MODELS = {}, {}, {}
    CORE.parameters = lambda: {MODEL_KEY: params_for(variant)}
    CORE.build_model = build_model
    CORE.protocol = lambda: protocol(variant)
    CORE.configure()
    CORE.core.fusion_base.build_model = build_model


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--variant", choices=sorted(VARIANTS), required=True)
    parser.add_argument("--fold", type=int, choices=range(1, 6), required=True)
    parser.add_argument("--initialize", action="store_true")
    args = parser.parse_args()
    configure(args.variant)
    CORE.OUT.mkdir(parents=True, exist_ok=True)
    if args.initialize:
        CORE.initialize()
        return
    # Use the original, audited five-fold training/evaluation implementation.
    CORE.main_args = None
    import torch
    torch.set_num_threads(2)
    torch.backends.mha.set_fastpath_enabled(False)
    CORE.main = CORE.main  # keep the imported function visible for protocol checks
    # Calling the fold body directly avoids spawning a second worker.
    rows, folds, y = CORE.read_inputs()
    digest = protocol(args.variant)["protocol_sha256"]
    path = CORE.OUT / "protocol.json"
    if path.exists():
        assert json.loads(path.read_text()) == protocol(args.variant), "训练协议已变化"
    else:
        anonymous.save_json(path, protocol(args.variant))
    marker = CORE.OUT / f"fold_{args.fold}" / MODEL_KEY / "completed.json"
    if marker.exists():
        return
    CORE.image_fold(MODEL_KEY, args.fold - 1, rows, folds, y, digest)


if __name__ == "__main__":
    main()
