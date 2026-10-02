#!/usr/bin/env python3
"""Evaluate AA-Mini evidence diagnostics used by temp6, temp9 and temp10.

This adapter reuses the audited public-figure evaluators with the AA-Mini
anonymous-findings folds and the already trained checkpoints.  It does not
train models or modify the manuscript.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
from tqdm import tqdm

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
from scripts import evaluate_lccf_public_figures as base
from scripts import evaluate_lccf_temp10 as temp10
from scripts import evaluate_multimodal_interventions as temp9

AA_TABLE1 = ROOT / "outputs/abdomenatlas3_mini/table1_fivefold_original_uit_anonymous_findings"
AA_TABLE28 = ROOT / "outputs/abdomenatlas3_mini/table2_8_fivefold_anonymous_findings"
AA_EXPERIMENT = ROOT / "outputs/abdomenatlas3_mini/experiment"
AA_ANON = ROOT / "outputs/abdomenatlas3_mini/experiment_anonymous_findings"

AA_METHOD_DIRS = {
    "mmfnet": "task2_mmfnet_2024",
    "radfuse": "task2_radfuse_2025",
    "saif": "task2_saif_2025",
    "mmtf": "task2_mmtf_2025",
    "camchex": "task2_camchex_adapted",
    "med3dvlm": "task2_med3dvlm_adapted",
    "m3fm": "task2_m3fm_adapted",
    "unified_mm": "unified_multimodal_framework_2026",
    "adaptive_fusion": "adaptive_multimodal_fusion_2026",
}


def aa_folder(ds: str, variant: str, fold: int) -> Path:
    if ds != "aa_mini":
        return ORIGINAL_FOLDER_FOR(ds, variant, fold)
    if variant == "full":
        name = "amef_multimodal"
        return AA_TABLE1 / f"fold_{fold}" / name
    if variant in {"b_none", "b_graph", "b_hyper1", "b_hyper3", "b_hyper4", "b_hyper5"}:
        name = {
            "b_none": "reason_none", "b_graph": "reason_graph",
            "b_hyper1": "reason_h1", "b_hyper3": "reason_h3",
            "b_hyper4": "reason_h4", "b_hyper5": "reason_h5",
        }[variant]
        return AA_TABLE28 / name / f"fold_{fold}" / "amef_multimodal"
    if variant in AA_METHOD_DIRS:
        return AA_TABLE1 / f"fold_{fold}" / AA_METHOD_DIRS[variant]
    raise KeyError((ds, variant))


def aa_load_inputs(ds: str):
    assert ds == "aa_mini"
    rows = json.loads((AA_ANON / "samples.json").read_text())
    y = np.asarray([row["labels"][:3] for row in rows], dtype=int)
    known = np.asarray([row.get("known_mask", [True] * 3)[:3] for row in rows], dtype=bool)
    from training.data import _encode_text_fields
    tokens_masks = [
        _encode_text_fields({"watch": row["findings_masked"]}, ("watch",), max_length=512, vocab_size=8192)
        for row in rows
    ]
    tokens = torch.stack([x[0] for x in tokens_masks])
    text_masks = torch.stack([x[1] for x in tokens_masks])
    bags = {}
    for fold in range(1, 6):
        ids, _, _, _ = base.reference(aa_folder("aa_mini", "full", fold), "aa_mini", rows)
        for index in ids:
            index = int(index)
            if index in bags:
                continue
            with np.load(AA_EXPERIMENT / "features" / f"{index:04d}.npz", allow_pickle=False) as z:
                feat = z["features"].astype(np.float32)
                pos = z["slice_indices"].astype(np.int64)
                total = int(z["original_count"])
            assert 1 <= len(feat) <= 64 and feat.shape == (len(pos), 768)
            assert np.isfinite(feat).all() and np.all(np.diff(pos) > 0)
            bags[index] = (feat, pos, total)
    return rows, y, known, tokens, text_masks, bags


ORIGINAL_FOLDER_FOR = base.folder_for
ORIGINAL_DATASETS = dict(base.DATASETS)


def configure() -> None:
    base.DATASETS = dict(ORIGINAL_DATASETS)
    base.DATASETS["aa_mini"] = ("AA-Mini", "aa_mini", 3)
    base.folder_for = aa_folder
    base.load_inputs = aa_load_inputs
    # temp10 imported these functions by name, so patch its module namespace.
    temp10.DATASETS = [("aa_mini", "AA-Mini", "#7B61A8")]
    temp10.folder_for = aa_folder
    temp10.load_inputs = aa_load_inputs
    temp10.load_model = base.load_model
    temp10.pack = base.pack
    temp10.reference = base.reference
    temp9.base.folder_for = aa_folder
    temp9.base.load_inputs = aa_load_inputs
    temp9.base.DATASETS = base.DATASETS
    temp9.base.load_model = base.load_model
    temp9.base.pack = base.pack
    temp9.base.reference = base.reference


def run_temp10(device: str, batch_size: int, resume: bool) -> None:
    configure()
    inputs = aa_load_inputs("aa_mini")
    out = temp10.OUT / "raw" / "aa_mini"
    out.mkdir(parents=True, exist_ok=True)
    device_obj = torch.device(device)
    for label, variant in temp10.VARIANTS.items():
        for fold in tqdm([1, 2, 3, 4, 5], desc=f"AA-Mini/temp10/{variant}", mininterval=5):
            target = out / variant / f"fold_{fold}.json"
            if resume and target.exists():
                try:
                    if json.loads(target.read_text()).get("protocol_version") == temp10.PROTOCOL_VERSION:
                        continue
                except json.JSONDecodeError:
                    pass
            temp10.evaluate_fold("aa_mini", variant, fold, inputs, device_obj, batch_size)


def run_temp9(device: str, batch_size: int, resume: bool) -> None:
    configure()
    device_obj = torch.device(device)
    inputs = aa_load_inputs("aa_mini")
    variants = (*base.MULTIMODAL, "full")
    out = ROOT / "outputs/lccf_public_evidence_20261001/multimodal_interventions_v2"
    for variant in variants:
        for fold in tqdm([1, 2, 3, 4, 5], desc=f"AA-Mini/temp9/{variant}", mininterval=5):
            target = out / "raw" / "aa_mini" / variant / f"fold_{fold}.json"
            if resume and target.exists():
                try:
                    if json.loads(target.read_text()).get("protocol_version") == temp9.VERSION:
                        continue
                except json.JSONDecodeError:
                    pass
            temp9.evaluate("aa_mini", variant, fold, inputs, device_obj, batch_size, out)


def run_temp6(device: str, batch_size: int, resume: bool) -> None:
    """Evaluate the full AA-Mini model for the retrieval diagnostic."""
    configure()
    inputs = aa_load_inputs("aa_mini")
    out = base.OUT / "raw" / "aa_mini" / "full"
    out.mkdir(parents=True, exist_ok=True)
    device_obj = torch.device(device)
    for fold in tqdm([1, 2, 3, 4, 5], desc="AA-Mini/temp6/full", mininterval=5):
        target = out / f"fold_{fold}.json"
        if resume and target.exists():
            try:
                if json.loads(target.read_text()).get("test_reference_passed"):
                    continue
            except json.JSONDecodeError:
                pass
        base.evaluate("aa_mini", "full", fold, inputs, device_obj, batch_size)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage", choices=["temp6", "temp9", "temp10", "all"], default="all")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    torch.set_num_threads(2)
    torch.backends.mha.set_fastpath_enabled(False)
    torch.manual_seed(2026)
    if args.stage in {"temp6", "all"}:
        run_temp6(args.device, args.batch_size, args.resume)
    if args.stage in {"temp10", "all"}:
        run_temp10(args.device, args.batch_size, args.resume)
    if args.stage in {"temp9", "all"}:
        run_temp9(args.device, args.batch_size, args.resume)


if __name__ == "__main__":
    main()
