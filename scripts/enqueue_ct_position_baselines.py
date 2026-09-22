#!/usr/bin/env python3
"""核对冒烟结果后将位置机制五折和后续位置评估加入正在运行的六卡队列。"""
from __future__ import annotations

import json

from ct_six_gpu_queue import ROOT, DEFAULT_QUEUE, save


def main():
    folder = "outputs/ct_rate_680/position_baselines"
    variants = ("apro_full", "original_pe", "comrope_ap", "videorope", "path", "dape_v2_kerple")
    smoke = json.loads((ROOT / folder / "smoke_audit.json").read_text())
    audit = json.loads((ROOT / folder / "implementation_audit.json").read_text())
    if not all(smoke[name]["finite_loss_gradients"] for name in variants):
        raise ValueError("模型冒烟检查未全部通过")
    if not all(audit[name]["padding_invariance"] for name in variants[2:]):
        raise ValueError("位置实现审计未全部通过")
    queue = json.loads(DEFAULT_QUEUE.read_text())
    if any(job["id"].startswith("position_") for job in queue["jobs"]):
        raise FileExistsError("位置模型任务已存在，不重复追加")
    inputs = ["src/exp_8/position_baselines.py", "src/scripts/run_ct_position_baselines.py",
              "src/scripts/audit_ct_position_baselines.py", f"{folder}/upstream_manifest.json",
              "outputs/ct_rate_680/experiment"]
    jobs = []
    for fold in range(1, 6):
        for variant in variants:
            destination = f"{folder}/{variant}/fold_{fold}/amef_multimodal"
            jobs.append({"id": f"position_{variant}_fold{fold}", "phase": "training", "memory_mib": 2500,
                         "command": ["src/scripts/run_ct_position_baselines.py", "--variant", variant, "--fold", str(fold)],
                         "sync_inputs": [*inputs, f"{folder}/{variant}/protocol.json"],
                         "artifacts": [f"{destination}/{name}" for name in (
                             "completed.json", "test_metrics.json", "test_predictions.csv", "best_model.pt", "config.json", "split_ids.json")],
                         "sync_output_dirs": [destination]})
    jobs.append({"id": "position_aggregate", "phase": "aggregate", "local_only": True, "memory_mib": 500,
                 "command": ["src/scripts/run_ct_position_baselines.py", "--aggregate"],
                 "depends_on": [job["id"] for job in jobs],
                 "artifacts": [f"{folder}/comparison.csv", f"{folder}/comparison.json", f"{folder}/comparison.md"]})
    jobs.append({"id": "position_recovery", "phase": "evaluation", "local_only": True, "memory_mib": 2500,
                 "command": ["src/scripts/evaluate_ct_position_suite.py"], "depends_on": ["position_aggregate"],
                 "artifacts": [f"{folder}/position_recovery/{name}" for name in (
                     "all_methods_summary.csv", "all_methods_table.md", "native_coordinate_audit.json",
                     "per_slice_results.csv", "repeat_metrics.csv", "sampling_manifest.jsonl", "metadata.json")],
                 "timeout_seconds": 7200})
    queue["jobs"].extend(jobs)
    save(DEFAULT_QUEUE, queue)
    print("新增30个配对分类折任务、1个汇总及1个严格位置评估/可计算性审计任务。")


if __name__ == "__main__":
    main()
