#!/usr/bin/env python3
"""完整CT删除部分切片后保留T层、隐藏原位置的配对五折训练。"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import sys
import time

import numpy as np
from tqdm import tqdm
import yaml

ROOT = Path(__file__).resolve().parents[2]
import run_ctrate_680_all_models as core
import run_cq500_amef_multimodal as amef
from evaluate_ctrate_amef_position_recovery import build_sampling, target_delete_count, FrozenConvNeXt
from prepare_ctrate_680_experiment import DATA, OUT as SOURCE, save_json, sha256
from scripts.task3_apro_cope_ablation_scheduler import base_model_params

DEFAULT_OUTPUT = ROOT / "outputs/ct_rate_680/missing25_raw_hidden_fivefold"
VARIANTS = {"apro_full": "AMEF-MIL（ACPE）", "original_pe": "AMEF-MIL（Original PE）"}
MODEL_KEY = "amef_multimodal"
BASE_PROTOCOL = core.protocol
OUT = DEFAULT_OUTPUT
VARIANT = "apro_full"


def json_digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False).encode()).hexdigest()


def ensure_json(path, value):
    if path.exists() and json.loads(path.read_text()) != value:
        raise ValueError(f"已有协议/数据不一致，禁止混用：{path}")
    save_json(path, value)


def prepare(seed, fraction):
    original = json.loads((SOURCE / "preparation_protocol.json").read_text())
    target = int(original["slices_per_case"])
    rows = json.loads((SOURCE / "samples.json").read_text())
    folds = json.loads((SOURCE / "patient_folds.json").read_text())
    retained, excluded, sampling = [], [], []
    for row in tqdm(rows, desc=f"筛选完整CT删除{fraction:.0%}后的病例并固定抽样"):
        source_index = row["case_index"]
        with np.load(SOURCE / "features" / f"{source_index:04d}.npz", allow_pickle=False) as cache:
            assert str(cache["patient_id"]) == row["patient_id"]
            assert str(cache["preparation_sha256"]) == original["preparation_sha256"]
            count = int(cache["original_count"])
        if count - target_delete_count(count, fraction) < target:
            excluded.append({"source_case_index": source_index, "patient_id": row["patient_id"],
                             "original_count": count, "reason": "删除后不足训练配置要求的T张"})
            continue
        case_seed = int(np.random.SeedSequence([seed, source_index]).generate_state(1)[0])
        selected = build_sampling(count, fraction, case_seed, target)
        selected.update(case_index=len(retained), source_case_index=source_index,
                        patient_id=row["patient_id"], original_count=count)
        sampling.append(selected)
        retained.append({**row, "case_index": len(retained), "source_case_index": source_index})
    mapping = {row["source_case_index"]: row["case_index"] for row in retained}
    groups = [[mapping[index] for index in group if index in mapping] for group in folds["folds"]]
    targets = np.asarray([row["labels"] for row in retained])
    if any(not group or np.any(targets[group].sum(0) == 0) or
           np.any(targets[group].sum(0) == len(group)) for group in groups):
        raise ValueError("筛选后的某一折缺少某标签正例或负例，不能直接运行原五折")
    subset_folds = {"labels": core.LABELS, "folds": groups,
                   "source_fold_case_indices": [[retained[index]["source_case_index"] for index in group]
                                                for group in groups],
                   "fold_positive_counts": [targets[group].sum(0).tolist() for group in groups],
                   "split": "仅过滤原五折，不重新划分；测试k、验证k+1、其余训练"}
    manifest = {"seed": seed, "delete_fraction": fraction, "target_instances": target,
                "cases": sampling, "excluded": excluded,
                "source_cases": len(rows), "retained_cases": len(retained), "excluded_cases": len(excluded),
                "sampling": "完整CT先随机删除指定比例，保留首尾，在剩余序列中按序号均匀采样T张；无重复或补齐",
                "repeats": 1, "mask_scope": "每例固定一次抽样，所有折、两个模型、训练验证测试共用"}
    prep = {"slices_per_case": target, "source_preparation": original,
            "source_samples_sha256": sha256(SOURCE / "samples.json"),
            "source_folds_sha256": sha256(SOURCE / "patient_folds.json"),
            "sampling_sha256": json_digest(manifest),
            "extractor_source_sha256": sha256(Path(__file__).with_name("evaluate_ctrate_amef_position_recovery.py")),
            "position_input": "只用当前输入序列的等距槽位；原始位置、间距和长度均不输入模型"}
    prep["preparation_sha256"] = json_digest(prep)
    folder = OUT / "data"
    for name, value in (("samples.json", retained), ("patient_folds.json", subset_folds),
                        ("sampling.json", manifest), ("preparation_protocol.json", prep)):
        ensure_json(folder / name, value)
    print(f"保留{len(retained)}/{len(rows)}例，排除{len(excluded)}例；五折数量{list(map(len, groups))}", flush=True)


def read_inputs():
    rows = json.loads((OUT / "data/samples.json").read_text())
    folds = json.loads((OUT / "data/patient_folds.json").read_text())
    targets = np.asarray([row["labels"] for row in rows], dtype=np.int64)
    assert len(rows) == len({row["patient_id"] for row in rows})
    assert [row["case_index"] for row in rows] == list(range(len(rows)))
    assert sorted(index for group in folds["folds"] for index in group) == list(range(len(rows)))
    assert folds["labels"] == core.LABELS
    for group, originals, counts in zip(folds["folds"], folds["source_fold_case_indices"],
                                       folds["fold_positive_counts"]):
        assert [rows[index]["source_case_index"] for index in group] == originals
        assert targets[group].sum(0).tolist() == counts
    return rows, folds, targets


def extract(worker, workers):
    import nibabel as nib
    import torch

    torch.set_num_threads(2)
    rows, _, _ = read_inputs()
    manifest = json.loads((OUT / "data/sampling.json").read_text())
    digest = json.loads((OUT / "data/preparation_protocol.json").read_text())["preparation_sha256"]
    encoder = FrozenConvNeXt(torch.device("cuda:0"))
    folder = OUT / "data/features"
    folder.mkdir(parents=True, exist_ok=True)
    selected_rows = [row for row in rows if row["case_index"] % workers == worker]
    started = time.time()
    for step, row in enumerate(tqdm(selected_rows, desc=f"抽样后提取特征：工作进程{worker}"), 1):
        case = row["case_index"]
        record = manifest["cases"][case]
        indices = np.asarray(record["selected_raw_indices"], dtype=np.int64)
        path = folder / f"{case:04d}.npz"
        if path.exists():
            with np.load(path, allow_pickle=False) as cached:
                assert str(cached["preparation_sha256"]) == digest
                assert str(cached["patient_id"]) == row["patient_id"]
                assert np.array_equal(cached["slice_indices"], indices)
                assert cached["features"].shape == (len(indices), 768)
                assert np.isfinite(cached["features"]).all()
            continue
        image = nib.load(str(DATA / row["file_path"]))
        if len(image.shape) != 3 or not np.isfinite(image.affine).all():
            raise ValueError(f"NIfTI维度或空间矩阵异常：{row['patient_id']}")
        orientation = nib.orientations.ornt_transform(nib.orientations.io_orientation(image.affine),
                                                      nib.orientations.axcodes2ornt(("R", "A", "S")))
        volume = nib.orientations.apply_orientation(np.asarray(image.dataobj, dtype=np.float32), orientation)
        if volume.shape[2] != record["original_count"]:
            raise ValueError(f"实际完整CT长度和原缓存不一致：{row['patient_id']}")
        slices = np.ascontiguousarray(volume[::-1, ::-1, indices].transpose(2, 1, 0))
        del volume, image
        if not np.isfinite(slices).all():
            raise ValueError(f"图像数值异常：{row['patient_id']}")
        features = encoder.encode(slices)
        temporary = path.with_suffix(".tmp.npz")
        np.savez_compressed(temporary, features=features, slice_indices=indices,
                            true_positions=indices / (record["original_count"] - 1),
                            original_count=record["original_count"], patient_id=row["patient_id"],
                            preparation_sha256=digest)
        temporary.replace(path)
        save_json(OUT / f"feature_worker_{worker}.json", {"completed": step, "total": len(selected_rows),
                  "wall_seconds": time.time() - started, "last_case": case})


def load_features(rows):
    manifest = json.loads((OUT / "data/sampling.json").read_text())
    prep = json.loads((OUT / "data/preparation_protocol.json").read_text())
    bags = []
    for row in tqdm(rows, desc="读取两模型共用的切片缺失特征"):
        case = row["case_index"]
        with np.load(OUT / "data/features" / f"{case:04d}.npz", allow_pickle=False) as cached:
            assert str(cached["patient_id"]) == row["patient_id"]
            assert str(cached["preparation_sha256"]) == prep["preparation_sha256"]
            assert np.array_equal(cached["slice_indices"], manifest["cases"][case]["selected_raw_indices"])
            features = cached["features"].astype(np.float32)
            assert features.shape == (prep["slices_per_case"], 768) and np.isfinite(features).all()
            bags.append(features)
    return bags


def parameters():
    main = yaml.safe_load((ROOT / "src/configs/task3/t3_main_model.yaml").read_text())
    model = base_model_params(VARIANT)
    model["label_query_consistency_weight"] = main["model"]["params"]["label_query_consistency_weight"]
    model["image_aux_weight"] = 0.0
    return {MODEL_KEY: model}


def forward_hidden(model, batch, text_ids, text_mask, device, training=False):
    import torch

    features, mask, labels, case_ids, _, _ = batch
    kwargs = {"images": features.to(device), "mask": mask.to(device),
              "watch_token_ids": text_ids[case_ids].to(device),
              "watch_token_mask": text_mask[case_ids].to(device),
              "instance_indices": None, "original_image_counts": None}
    if training:
        kwargs["labels"] = labels.to(device)
    with torch.autocast(device_type=torch.device(device).type, enabled=False):
        return model(**kwargs)


def stable_paths(value):
    prefix = str(ROOT.resolve()) + "/"
    if isinstance(value, dict):
        return {key.removeprefix(prefix): stable_paths(item) for key, item in value.items()}
    if isinstance(value, list):
        return [stable_paths(item) for item in value]
    return value.removeprefix(prefix) if isinstance(value, str) else value


def protocol():
    value = stable_paths(BASE_PROTOCOL())
    for path in (Path(__file__), ROOT / "src/exp_4/models.py",
                 ROOT / "src/model/gastro_label_graph_mil/modules.py",
                 ROOT / "src/training/data.py",
                 ROOT / "src/scripts/task3_apro_cope_ablation_scheduler.py",
                 OUT / "data/sampling.json", OUT / "data/feature_integrity.json"):
        value["source_sha256"][str(path.relative_to(ROOT))] = sha256(path)
    manifest = json.loads((OUT / "data/sampling.json").read_text())
    value.update(dataset=f"CT-RATE原680例中符合完整CT删除{manifest['delete_fraction']:.0%}后仍有T层的子集",
                 split="仅过滤原五折归属，测试k、验证k+1、其余三折训练；不重新分组",
                 image="完整CT删除指定比例后均匀采样T层；保留首尾，两个模型完全共用冻结ConvNeXt特征",
                 position="原始索引、坐标、间距、完整CT长度均不输入；ACPE使用等距fallback，Original PE使用既有TimePositionEncoding",
                 missingness="单个固定缺失随机种子；每例抽样固定，训练验证测试相同缺失条件；额外instance_dropout=0",
                 scope="两个位置变体从头训练分类头与聚合模块，不加载旧分类checkpoint；视觉编码器冻结",
                 amef="只改变position_variant；ASL+0.01标签查询损失，保留相同掩码临床文本，非纯视觉评估",
                 runner="src/scripts/run_ctrate_slice_missing.py", position_variant=VARIANT,
                 delete_fraction=manifest["delete_fraction"])
    value.pop("protocol_sha256", None)
    value["protocol_sha256"] = json_digest(value)
    return value


def configure(variant):
    global VARIANT
    VARIANT = variant
    prep = json.loads((OUT / "data/preparation_protocol.json").read_text())
    core.INPUT = OUT / "data"
    core.MODELS = {MODEL_KEY: VARIANTS[variant]}
    core.IMAGE_MODELS, core.TEXT_MODELS, core.FUSION_MODELS = {}, {}, {}
    core.SETTINGS = {**core.SETTINGS, "max_instances": prep["slices_per_case"], "instance_dropout": 0.0}
    core.read_inputs = read_inputs
    core.parameters = parameters
    core.protocol = protocol
    core.build_model = amef.build_model
    core.load_features = load_features
    core.forward_batch = forward_hidden
    core.configure_adapters()


def arguments_for_variant(variant, fold=None):
    return argparse.Namespace(output_dir=OUT / variant, models=None, devices=list(range(5)),
                              worker_index=None if fold is None else fold - 1)


def audit():
    rows, folds, _ = read_inputs()
    load_features(rows)
    integrity = {f"{row['case_index']:04d}.npz": sha256(OUT / "data/features" / f"{row['case_index']:04d}.npz")
                 for row in tqdm(rows, desc="核对共用特征文件哈希")}
    ensure_json(OUT / "data/feature_integrity.json", integrity)
    for variant in VARIANTS:
        configure(variant)
        ensure_json(OUT / variant / "protocol.json", protocol())
    save_json(OUT / "data_audit.json", {"cases": len(rows), "fold_sizes": list(map(len, folds["folds"])),
              "identical_features_both_models": True, "additional_instance_dropout": 0.0,
              "source_positions_hidden": True, "excluded_cases": 680-len(rows)})
    print("抽样、特征完整性和两模型协议审计通过", flush=True)


def smoke():
    import gc
    import torch
    from training.losses import AsymmetricLossMultiLabel

    assert core.statistics.mean([0.0, 1.0]) == 0.5
    assert core.fusion_base.statistics.stdev([0.0, 1.0]) > 0
    torch.set_num_threads(2)
    torch.backends.mha.set_fastpath_enabled(False)
    rows, _, targets = read_inputs()
    rows, targets = rows[:16], targets[:16]
    bags = load_features(rows)
    tokens, masks = core.fusion_base.encode_descriptions(
        {row["case_index"]: row["findings_masked"] for row in rows}, np.arange(len(rows)))
    results = {}
    for variant in VARIANTS:
        configure(variant)
        batch = next(iter(amef.loader(bags, targets, np.arange(len(rows)), True, 142)))
        assert batch[1].all() and batch[1].shape[1] == core.SETTINGS["max_instances"]
        model, weights = amef.build_model(MODEL_KEY, parameters()[MODEL_KEY])
        assert (model.apro_positioner is not None) == (variant == "apro_full")
        model.cuda().train()
        optimizer = torch.optim.AdamW(model.parameters(), lr=core.SETTINGS["learning_rate"])
        torch.cuda.reset_peak_memory_stats()
        for _ in range(2):
            optimizer.zero_grad(set_to_none=True)
            output = forward_hidden(model, batch, tokens, masks, "cuda:0", training=True)
            loss = AsymmetricLossMultiLabel()(output["logits"], batch[2].cuda())
            for name, weight in weights.items():
                loss = loss + weight * output["aux_losses"][name]
            assert torch.isfinite(loss)
            loss.backward()
            assert all(torch.isfinite(parameter.grad).all() for parameter in model.parameters()
                       if parameter.grad is not None)
            optimizer.step()
        model.eval()
        with torch.no_grad():
            original = forward_hidden(model, batch, tokens, masks, "cuda:0")
            poisoned = (*batch[:4], torch.full_like(batch[4], 9999), torch.full_like(batch[5], 10000))
            changed = forward_hidden(model, poisoned, tokens, masks, "cuda:0")
            torch.testing.assert_close(original["logits"], changed["logits"], rtol=0, atol=0)
            if variant == "apro_full":
                raw = original["apro_raw_coordinates"]
                expected = torch.linspace(0, 1, raw.shape[1], device=raw.device).expand_as(raw)
                torch.testing.assert_close(raw, expected)
        results[variant] = {"finite_loss_and_gradients": True, "metadata_poisoning_no_effect": True,
                            "peak_reserved_mib": torch.cuda.max_memory_reserved() / 1024**2,
                            "statistics_module": core.statistics.__file__}
        del model, optimizer, output, loss, original, changed
        gc.collect()
        torch.cuda.empty_cache()
    save_json(OUT / "smoke_audit.json", results)
    print(json.dumps(results, ensure_ascii=False, indent=2), flush=True)


def aggregate():
    summaries = []
    for variant in VARIANTS:
        configure(variant)
        args = arguments_for_variant(variant)
        if core.aggregate(args, protocol()["protocol_sha256"]) != 5:
            raise ValueError(f"{variant}尚未完成五折")
        summary = json.loads((args.output_dir / "summary.json").read_text())[0]
        summaries.append({**summary, "position_variant": variant})
    save_json(OUT / "comparison.json", summaries)
    import csv

    fields = ["position_variant", "macro_f1_mean", "macro_f1_std", "macro_f1_fixed_0_5_mean",
              "macro_f1_fixed_0_5_std", "oof_macro_f1"]
    with (OUT / "comparison.csv").open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(summaries)
    manifest = json.loads((OUT / "data/sampling.json").read_text())
    report = (f"# CT-RATE完整CT缺失{manifest['delete_fraction']:.0%}：ACPE与Original PE配对五折\n\n"
              f"保留{manifest['retained_cases']}例、排除{manifest['excluded_cases']}例，过滤原五折而不重新划分。"
              f"缺失随机种子{manifest['seed']}（每例种子从原case_index派生），只做一次缺失抽样。\n\n"
              "| 模型 | 验证集调阈值Macro-F1 | 固定0.5 Macro-F1 | OOF Macro-F1 |\n|---|---:|---:|---:|\n")
    for summary in summaries:
        report += (f"| {summary['model']} | {summary['macro_f1_mean']:.4f} ± {summary['macro_f1_std']:.4f} | "
                   f"{summary['macro_f1_fixed_0_5_mean']:.4f} ± {summary['macro_f1_fixed_0_5_std']:.4f} | "
                   f"{summary['oof_macro_f1']:.4f} |\n")
    report += ("\n均值±五个测试折的样本标准差（ddof=1），不是多次缺失抽样的标准差。"
               "完整CT删除指定比例后从剩余序列采样训练配置指定的T张，两模型同图同文本；隐藏真实坐标，关闭额外切片丢弃。"
               "保留原三标签和掩码临床所见，视觉编码器冻结，分类与聚合模块重新训练30轮，测试不用于选模型或阈值。"
               "这不是纯视觉位置恢复评估；与此前实验的缺失条件、位置输入及额外丢弃设置不同，"
               "应优先比较本次完全相同条件下的两模型，不能将与旧实验的差值全部归因于位置模块。\n")
    (OUT / "comparison.md").write_text(report, encoding="utf-8")


def main():
    global OUT
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--prepare-only", action="store_true")
    parser.add_argument("--sampling-seed", type=int, default=42)
    parser.add_argument("--delete-fraction", type=float, choices=(0.25, 0.5, 0.75), default=0.25)
    parser.add_argument("--feature-worker", type=int)
    parser.add_argument("--feature-workers", type=int, default=1)
    parser.add_argument("--audit-only", action="store_true")
    parser.add_argument("--smoke-test", action="store_true")
    parser.add_argument("--variant", choices=list(VARIANTS))
    parser.add_argument("--fold", type=int, choices=range(1, 6))
    parser.add_argument("--aggregate", action="store_true")
    args = parser.parse_args()
    OUT = args.output_dir.resolve()
    if args.prepare_only:
        prepare(args.sampling_seed, args.delete_fraction)
    elif args.feature_worker is not None:
        extract(args.feature_worker, args.feature_workers)
    elif args.audit_only:
        audit()
    elif args.smoke_test:
        configure("apro_full")
        smoke()
    elif args.aggregate:
        aggregate()
    elif args.variant is not None and args.fold is not None:
        configure(args.variant)
        core.worker(arguments_for_variant(args.variant, args.fold))
    else:
        parser.error("请选择准备、提取、审计、冒烟检查、汇总，或指定variant和fold")


if __name__ == "__main__":
    main()
