#!/usr/bin/env python3
"""新分类实验完成后复用严格位置恢复评估，并对无标量坐标的方法明确报告N/A。"""
from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import run_ct_position_baselines as training
import evaluate_ctrate_amef_position_recovery as evaluation
import numpy as np
import torch

OUT = training.OUT / "position_recovery"
ORIGINAL_CACHE_READER = evaluation.load_or_extract_case


def plot_curves(output_dir, slice_rows, curve_cases):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plot

    folder = output_dir / "position_curves"
    folder.mkdir(parents=True, exist_ok=True)
    audit = {}
    for case in curve_cases:
        figure, axes = plot.subplots(2, 2, figsize=(11, 8), sharex=True, sharey=True)
        records = []
        for axis, fraction in zip(axes.flat, evaluation.DELETION_FRACTIONS):
            seed = 0 if fraction == 0 else evaluation.NONZERO_SEEDS[0]
            rows = sorted([row for row in slice_rows if int(row["case_index"]) == case
                           and float(row["deletion_fraction"]) == fraction and int(row["seed"]) == seed],
                          key=lambda row: int(row["selected_position"]))
            if len(rows) < 2:
                raise ValueError(f"曲线病例{case}在缺失{fraction}、种子{seed}没有完整切片记录")
            order = [int(row["selected_position"]) for row in rows]
            axis.plot(order, [float(row["true_position"]) for row in rows], label="True position", linewidth=2)
            axis.plot(order, [float(row["baseline_position"]) for row in rows], label="Uniform", linestyle="--")
            positions = [float(row["model_position"]) for row in rows]
            if np.isfinite(positions).all():
                axis.plot(order, positions, label="ACPE")
            else:
                axis.text(0.5, 0.5, "Invalid ACPE coordinates", transform=axis.transAxes, ha="center")
            axis.set_title(f"Delete {fraction:.0%}; seed {seed}")
            axis.grid(alpha=0.25)
            records.append({"deletion_fraction": fraction, "seed": seed, "slice_count": len(rows)})
        axes[0, 0].legend()
        figure.supxlabel("Input slice order")
        figure.supylabel("Normalized position")
        figure.suptitle(f"CT-RATE case {case}")
        figure.tight_layout()
        figure.savefig(folder / f"case_{case:04d}.png", dpi=160)
        plot.close(figure)
        audit[str(case)] = records
    evaluation.save_json(output_dir / "curve_selection_audit.json", audit)


def redraw_existing():
    with (OUT / "per_slice_results.csv").open(encoding="utf-8-sig", newline="") as stream:
        rows = list(csv.DictReader(stream))
    cases = sorted({int(row["case_index"]) for row in rows})[:4]
    plot_curves(OUT, rows, cases)


def load_model(checkpoint_path, device):
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    if checkpoint["model_key"] != training.MODEL_KEY:
        raise ValueError("不是AMEF分类checkpoint")
    model, _ = training.build_model(training.MODEL_KEY, checkpoint["model_parameters"])
    model.load_state_dict(checkpoint["state_dict"], strict=True)
    model.to(device).eval()
    torch.backends.mha.set_fastpath_enabled(False)
    return model, checkpoint, checkpoint["model_parameters"]


def reuse_features(row, extractor, cache_dir, cache_protocol):
    old = training.ROOT / "outputs/ct_rate_680/position_recovery/raw_feature_cache"
    if (old / f"{row['case_index']:04d}.npz").is_file():
        with np.load(old / f"{row['case_index']:04d}.npz", allow_pickle=False) as data:
            if str(data["patient_id"]) != row["patient_id"]:
                raise ValueError("旧原始特征缓存患者ID不一致")
        return ORIGINAL_CACHE_READER(row, extractor, old, cache_protocol)
    return ORIGINAL_CACHE_READER(row, extractor, cache_dir, cache_protocol)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plots-only", action="store_true")
    args = parser.parse_args()
    if args.plots_only:
        redraw_existing()
        metadata = json.loads((OUT / "metadata.json").read_text())
        metadata.setdefault("initial_wrapper_sha256", metadata["wrapper_sha256"])
        metadata["wrapper_sha256"] = evaluation.sha256(Path(__file__))
        metadata["plot_fix"] = "按缺失条件分别选择0或42种子；仅修复空白面板，不改指标与抽样结果"
        evaluation.save_json(OUT / "metadata.json", metadata)
        return
    torch.set_num_threads(2)
    checkpoint, selected_fold, _ = evaluation.resolve_checkpoint(training.OUT / "apro_full", None)
    saved = OUT / "metadata.json"
    prior = json.loads(saved.read_text()) if saved.exists() else {}
    if prior and prior["checkpoint_sha256"] != evaluation.sha256(checkpoint):
        raise ValueError("已有位置评估使用不同checkpoint，禁止覆盖混用")
    if not prior.get("finished_unix"):
        evaluation.build_model = load_model
        evaluation.load_or_extract_case = reuse_features
        evaluation.plot_curves = plot_curves
        sys.argv = [str(Path(evaluation.__file__)), "--checkpoint-root", str(training.OUT / "apro_full"),
                    "--output-dir", str(OUT), "--overwrite", "--num-curves", "4"]
        evaluation.main()
    metadata = json.loads(saved.read_text())
    metadata.update(wrapper="src/scripts/evaluate_ct_position_suite.py",
                    wrapper_sha256=evaluation.sha256(Path(__file__)),
                    coordinate_limitation="本数据为规则NIfTI体积，只能审计RAS仿射坐标；缺少逐层DICOM位置，不能据此排除源采集非均匀间距。",
                    new_modules_without_scalar="ComRoPE/VideoRoPE/PaTH/DAPE V2均无原生标量位置，未新增回归头或校准")
    evaluation.save_json(saved, metadata)
    audit = {}
    for variant in ("comrope_ap", "videorope", "path", "dape_v2_kerple"):
        path, fold, candidates = evaluation.resolve_checkpoint(training.OUT / variant, None)
        model, payload, params = load_model(path, "cuda:0")
        config = json.loads(path.with_name("config.json").read_text())
        target = config["settings"]["max_instances"]
        split = json.loads(path.with_name("split_ids.json").read_text())
        case = split["test"][0]
        with np.load(training.core.INPUT / "features" / f"{case:04d}.npz", allow_pickle=False) as features:
            images = torch.from_numpy(features["features"]).cuda()[None, :, :, None, None]
        mask = torch.ones(1, target, dtype=torch.bool, device="cuda")
        with torch.inference_mode():
            _, _, _, diagnostics = model.encode_long_mil(images, mask, instance_indices=None, original_image_counts=None)
        if "apro_context_coordinates" in diagnostics or model.apro_positioner is not None:
            raise ValueError("新位置模块不应残留ACPE或伪造其坐标")
        audit[variant] = {"checkpoint": str(path.relative_to(training.ROOT)), "checkpoint_sha256": evaluation.sha256(path),
                          "selected_fold": fold, "fold_candidates": candidates, "T": target,
                          "native_scalar_position": False, "status": "N/A", "evaluated_test_case": case,
                          "output_keys": sorted(diagnostics), "reason": "原生输出不是标量c_t；禁止将矩阵、相位或注意力权重当作位置"}
        del model, payload, images
        torch.cuda.empty_cache()
    evaluation.save_json(OUT / "native_coordinate_audit.json", audit)
    with (OUT / "summary.csv").open(encoding="utf-8-sig", newline="") as stream:
        summaries = list(csv.DictReader(stream))
    metrics = ("PRE", "GRE", "Acc@0.02", "Acc@0.05")
    result = []
    table = "# CT-RATE位置恢复：可计算性与结果\n\n"
    table += f"ACPE按五折最低验证损失选择折{selected_fold}；只用其测试CT，各缺失条件共用满足75%后仍有T层的病例。\n\n"
    table += "| 删除比例 | 方法 | PRE↓ | GRE↓ | Acc@0.02↑ | Acc@0.05↑ |\n|---|---|---:|---:|---:|---:|\n"
    for summary in summaries:
        fraction = float(summary["deletion_fraction"])
        for name, prefix in (("等距位置基线", "baseline"), ("ACPE", "model")):
            row = {"deletion_fraction": fraction, "method": name, "status": "computed"}
            displayed = []
            for metric in metrics:
                mean, std = summary[f"{prefix}_{metric}_mean"], summary[f"{prefix}_{metric}_std"]
                row[f"{metric}_mean"], row[f"{metric}_std"] = mean, std
                displayed.append(f"{float(mean):.4f} ± {float(std):.4f}")
            result.append(row)
            table += f"| {fraction:.0%} | {name} | " + " | ".join(displayed) + " |\n"
        for variant in audit:
            result.append({"deletion_fraction": fraction, "method": training.VARIANTS[variant], "status": "N/A:no_native_scalar_position"})
            table += f"| {fraction:.0%} | {training.VARIANTS[variant]} | N/A | N/A | N/A | N/A |\n"
    fields = ["deletion_fraction", "method", "status", *[f"{metric}_{statistic}" for metric in metrics for statistic in ("mean", "std")]]
    evaluation.write_csv(OUT / "all_methods_summary.csv", result, fields)
    table += "\n每个CT先对非零比例的五组随机重复平均，再对CT等权汇总；标准差为CT间样本标准差（ddof=1）。N/A不是0分、未完成或模型失效，而是原方法没有标量位置输出；没有引入额外位置回归头。\n"
    (OUT / "all_methods_table.md").write_text(table)
    print("位置恢复评估与四种方法原生坐标可计算性审计完成。")


if __name__ == "__main__":
    main()
