#!/usr/bin/env python3
"""用已有单折权重补测 r051 的完整原序列删除网格；只评估验证集。"""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import gc
import gzip
import hashlib
import json
import os
from pathlib import Path
import sys
import time

import numpy as np
import torch
from tqdm import tqdm

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "src/scripts"))
from auto_research.model import ResearchModel
from auto_research.train import DATASETS, collate, label_f1, write_json
from training.data import _encode_text_fields
from paper_block_deletion import block_sampling, RATIOS, BLOCKS
from evaluate_public_block3 import normalize_context, position_metrics
import prepare_figure1_features as feature_builder
from prepare_public_block3_features import FrozenEncoder

RUNS = ROOT / "outputs/acpe_research_200_20260928/runs"
OUT = ROOT / "outputs/r051_3d_20260928"
MODELS = {"r051": "r051_depth_descriptor_2c1ff", "original_pe": "r033_original_pe"}
GRID = [(r, b) for r in RATIOS for b in ([1] if r == 0 else BLOCKS)]


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def get_cases(dataset):
    protocols = {name: json.loads((RUNS / (rid + "_" + dataset) / "protocol.json").read_text())
                 for name, rid in MODELS.items()}
    reference = protocols["r051"]
    assert reference["validation_ids"] == protocols["original_pe"]["validation_ids"]
    assert reference["train_ids"] == protocols["original_pe"]["train_ids"]
    cases = reference["validation_ids"]
    assert not set(cases) & set(reference["train_ids"])
    assert not set(cases) & set(reference["excluded_test_ids"])
    folder = ROOT / "outputs" / DATASETS[dataset] / "experiment"
    rows_list = json.loads((folder / "samples.json").read_text())
    assert all(row["case_index"] == i for i, row in enumerate(rows_list))
    assert sha(folder / "samples.json") == reference["input_hashes"]["samples.json"]
    records, excluded, missing = {}, [], []
    for case in tqdm(cases, desc=dataset + " 核验完整网格", mininterval=3):
        initial = folder / "features" / f"{case:04d}.npz"
        with np.load(initial, allow_pickle=False) as cache:
            count = int(cache["original_count"])
        try:
            selections = {f"{r:02d}_B{b}": block_sampling(count, r, b, case) for r, b in GRID}
        except ValueError as error:
            excluded.append(dict(case_index=case, source_count=count, reason=str(error)))
            continue
        old = ROOT / "outputs/paper_results/figure1_features" / dataset / f"{case:04d}.npz"
        own = OUT / "features" / dataset / f"{case:04d}.npz"
        mask_path = OUT / "masks" / dataset / f"{case:04d}.json.gz"
        mask_path.parent.mkdir(parents=True, exist_ok=True)
        manifest = dict(dataset=dataset, case_index=case, source_count=count, seed=42,
                        configurations=selections)
        if not mask_path.exists():
            with gzip.open(mask_path, "wt", encoding="utf-8") as stream:
                json.dump(manifest, stream, ensure_ascii=False, separators=(",", ":"))
        else:
            with gzip.open(mask_path, "rt", encoding="utf-8") as stream:
                assert json.load(stream) == manifest
        source = old if old.exists() else own
        records[case] = dict(source_count=count, configurations=selections, path=source,
                             mask_path=mask_path)
        if not source.exists():
            missing.append(case)
    if not records:
        raise RuntimeError("验证集没有满足全部65条件的共同病例")
    write_json(OUT / dataset / "eligibility.json", dict(validation_ids=cases,
               eligible_ids=sorted(records), excluded=excluded,
               rule="完整原始序列允许全部65种非空非相邻B-block删除；各条件使用同一病例集"))
    return records, rows_list, missing, protocols


def prepare_features(dataset, records, rows, missing, device):
    if missing:
        feature_builder.MASKS = OUT / "masks"
        feature_builder.OUT = OUT / "features"
        encoder = FrozenEncoder(str(device))
        # 冻结模型只读；每线程的 inference_mode/autocast 由 encode 内部建立。
        # 用一个 CUDA 进程重叠病例解压和提取，避免每个病例的磁盘等待串行阻塞。
        with ThreadPoolExecutor(max_workers=4) as pool:
            futures = [pool.submit(feature_builder.prepare_case, dataset, case, rows, encoder) for case in missing]
            for future in tqdm(as_completed(futures), total=len(futures), desc=dataset + " 补提冻结特征", mininterval=4):
                future.result()
        del encoder
        gc.collect()
        torch.cuda.empty_cache()
    hashes = {}
    for case, record in tqdm(records.items(), desc=dataset + " 装载网格特征", mininterval=3):
        path = record["path"]
        with np.load(path, allow_pickle=False) as cache:
            assert int(cache["original_count"]) == record["source_count"]
            source_indices = cache["source_indices"].astype(np.int64)
            features = cache["features"].astype(np.float32)
        assert features.shape == (len(source_indices), 768) and np.isfinite(features).all()
        for selection in record["configurations"].values():
            indices = np.asarray(selection["selected_raw_indices"])
            slots = np.searchsorted(source_indices, indices)
            assert np.all(slots < len(source_indices)) and np.array_equal(source_indices[slots], indices)
        # 复用原始缓存时，核验重合切片特征确实来自同一冻结编码结果。
        initial = ROOT / "outputs" / DATASETS[dataset] / "experiment/features" / f"{case:04d}.npz"
        with np.load(initial, allow_pickle=False) as cache:
            common, ia, ib = np.intersect1d(source_indices, cache["slice_indices"], return_indices=True)
            assert len(common) and np.array_equal(features[ia], cache["features"].astype(np.float32)[ib])
        record.update(indices=source_indices, features=features)
        hashes[str(case)] = dict(path=str(path.relative_to(ROOT)), sha256=sha(path),
                                mask_sha256=sha(record["mask_path"]))
    write_json(OUT / dataset / "feature_manifest.json", hashes)


def load_models(dataset, protocols, labels, device):
    models, sources = {}, {}
    for name, rid in MODELS.items():
        folder = RUNS / (rid + "_" + dataset)
        result = json.loads((folder / "result.json").read_text())
        for relative in ("auto_research/model.py", "exp_8/models.py"):
            assert sha(ROOT / "src" / relative) == protocols[name]["code_hashes"][relative]
        ckpt = torch.load(folder / "best.pt", map_location="cpu", weights_only=True)
        config = protocols[name]["spec"]["config"]
        assert ckpt["config"] == config
        model = ResearchModel(labels, config)
        model.load_state_dict(ckpt["model"], strict=True)
        model.epoch = int(ckpt["epoch"]) - 1
        model.to(device).eval()
        models[name] = model
        sources[name] = dict(checkpoint=str((folder / "best.pt").relative_to(ROOT)),
                             sha256=sha(folder / "best.pt"), config=config,
                             best_epoch=ckpt["epoch"], thresholds=result["thresholds"],
                             position_scale=min(1., ckpt["epoch"] / config.get("position_warmup", 1)),
                             seed=protocols[name]["spec"]["seed"], fold=protocols[name]["spec"]["fold"])
    return models, sources


def run(dataset, device, batch_size):
    started = time.time()
    records, rows, missing, protocols = get_cases(dataset)
    prepare_features(dataset, records, rows, missing, device)
    cases = sorted(records)
    tokens = {case: _encode_text_fields({"watch": rows[case]["findings_masked"]}, ("watch",),
               max_length=512, vocab_size=8192) for case in cases}
    models, sources = load_models(dataset, protocols, len(rows[cases[0]]["labels"]), device)
    output = OUT / dataset / "grid.json"
    header = dict(dataset=dataset, evaluation_split="validation", fold=0, seed=42,
                  index_protocol="original_acquisition_indices", source_sequence="full_before_deletion",
                  thresholds="saved_clean_validation_thresholds_no_retuning", num_cases=len(cases),
                  case_indices=cases, checkpoints=sources, grids=65,
                  autocast="float16", training=False,
                  coordinate_metric="内部切片对原始采集索引的Acc@0.05，先病例内平均再病例间平均；Original PE为均匀槽位坐标；非物理位置恢复")
    results = {}
    if output.exists():
        old = json.loads(output.read_text())
        for key, value in header.items():
            assert old[key] == value, f"已有评估协议不同：{key}"
        results = old["results"]
    for ratio, blocks in tqdm(GRID, desc=dataset + " 65条件配对评估", mininterval=3):
        key = f"{ratio:02d}_B{blocks}"
        if key in results:
            continue
        probabilities = {name: [] for name in models}
        accuracies = {name: [] for name in models}
        labels, knowns, selected_all = [], [], []
        with torch.inference_mode():
            for start in range(0, len(cases), batch_size):
                selected_cases = cases[start:start + batch_size]
                entries, originals = [], []
                for case in selected_cases:
                    record = records[case]
                    selected = np.asarray(record["configurations"][key]["selected_raw_indices"], dtype=np.int64)
                    local = np.searchsorted(record["indices"], selected)
                    row = rows[case]
                    entries.append((record["features"][local], selected, record["source_count"],
                                    *tokens[case], np.asarray(row["labels"], np.float32),
                                    np.asarray(row.get("known_mask", [True]*len(row["labels"])), bool), case))
                    originals.append(selected)
                    selected_all.append(selected.tolist())
                batch = collate(entries)
                labels.append(batch["targets"].numpy());knowns.append(batch["known"].numpy())
                inputs = {k:v.to(device) for k,v in batch.items() if k not in {"targets","known","ids"}}
                for name, model in models.items():
                    with torch.autocast("cuda", dtype=torch.float16):
                        values = model(**inputs)
                    logits = values["logits"].float()
                    assert torch.isfinite(logits).all(), "分类输出非有限"
                    probabilities[name].append(logits.sigmoid().cpu().numpy())
                    context = values.get("apro_context_coordinates")
                    for j, case in enumerate(selected_cases):
                        selected = originals[j]
                        truth = selected.astype(np.float64) / (records[case]["source_count"]-1)
                        if name == "r051":
                            pred, status, monotonic = normalize_context(context[j,:len(selected)].float().cpu().numpy())
                            assert monotonic and status == "ok", "上下文坐标异常"
                        else:
                            pred = np.linspace(0.,1.,len(selected))
                        acc = position_metrics(pred,truth)["Acc@0.05"]
                        assert np.isfinite(acc)
                        accuracies[name].append(acc)
        y, known = np.concatenate(labels), np.concatenate(knowns)
        metrics = {}
        arrays = dict(case_indices=np.asarray(cases), labels=y, known_mask=known)
        for name in models:
            probability = np.concatenate(probabilities[name])
            thresholds = sources[name]["thresholds"]
            f1 = float(label_f1(y, probability, known, thresholds).mean())
            acc = float(np.mean(accuracies[name]))
            metrics[name] = dict(macro_f1=f1, acc005=acc)
            arrays[name+"_probabilities"] = probability
            arrays[name+"_case_acc005"] = np.asarray(accuracies[name])
        predictions = OUT / dataset / "predictions" / (key + ".npz")
        predictions.parent.mkdir(exist_ok=True)
        np.savez_compressed(predictions, **arrays)
        results[key] = dict(ratio=ratio, blocks=blocks, num_cases=len(cases), status="COMPLETE",
                            **metrics, delta_f1=metrics["r051"]["macro_f1"]-metrics["original_pe"]["macro_f1"],
                            delta_acc005=metrics["r051"]["acc005"]-metrics["original_pe"]["acc005"],
                            prediction_file=str(predictions.relative_to(ROOT)), prediction_sha256=sha(predictions))
        write_json(output, dict(**header, results=results, updated=time.time(), seconds=time.time()-started))
        print(f"{dataset} {key}: N={len(cases)} ΔF1={results[key]['delta_f1']:+.4f} ΔAcc={results[key]['delta_acc005']:+.4f}", flush=True)
    assert len(results) == 65
    del models, records
    gc.collect();torch.cuda.empty_cache()


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--datasets",nargs="+",choices=list(DATASETS),required=True)
    parser.add_argument("--batch-size",type=int,default=16)
    args=parser.parse_args()
    if os.environ.get("CUDA_VISIBLE_DEVICES") not in {"2","3"}:
        raise RuntimeError("本次评估仅允许204主机GPU2或GPU3，每卡一个进程；GPU0禁用")
    torch.set_num_threads(2);torch.set_num_interop_threads(1)
    torch.backends.mha.set_fastpath_enabled(False)
    for dataset in args.datasets:
        run(dataset,torch.device("cuda:0"),args.batch_size)


if __name__ == "__main__":main()
