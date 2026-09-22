#!/usr/bin/env python3
"""Run task-adapted CaMCheX, Med3DVLM and M3FM baselines.

The official projects target different input modalities/tasks.  This runner
therefore reports an explicit task-adapted reproduction on the project's
existing cached study features and report tokens; it never overwrites the
previous table-2 outputs.
"""
from __future__ import annotations

import argparse
import csv
import fcntl
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import traceback
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src/scripts"))

ADAPTED_MODELS = {
    "task2_camchex_adapted": "CaMCheX-adapted",
    "task2_med3dvlm_adapted": "Med3DVLM-adapted",
    "task2_m3fm_adapted": "M3FM-adapted",
}
DEFAULT_OUTPUTS = {
    "ctrate": ROOT / "outputs/task_adapted_vlms/ct_rate_680",
    "amos": ROOT / "outputs/task_adapted_vlms/amos_mm_7",
    "mrrate": ROOT / "outputs/task_adapted_vlms/mr_rate_1k",
}


def _patch_ct(output_dir: Path):
    import run_ctrate_680_all_models as core

    core.IMAGE_MODELS = {}
    core.TEXT_MODELS = {}
    core.FUSION_MODELS = dict(ADAPTED_MODELS)
    core.MODELS = dict(ADAPTED_MODELS)
    core.configure_adapters()
    return core


def _patch_mr(output_dir: Path):
    import run_mrrate_1k_all_models as core

    core.IMAGE_MODELS = {}
    core.TEXT_MODELS = {}
    core.FUSION_MODELS = dict(ADAPTED_MODELS)
    core.MODELS = dict(ADAPTED_MODELS)
    core.RECOVERY = ROOT / "outputs/task_adapted_vlms/no_compatible_recovery_protocol.json"
    core.configure()
    return core


def _patch_amos(output_dir: Path):
    import run_amos_mm_all_models as core

    core.OUT = output_dir
    core.IMAGE_MODELS = {}
    core.TEXT_MODELS = {}
    core.MODELS = dict(ADAPTED_MODELS)
    core.fusion_base.MODELS = dict(ADAPTED_MODELS)
    return core


def _target(dataset: str, output_dir: Path):
    if dataset == "ctrate":
        return _patch_ct(output_dir)
    if dataset == "mrrate":
        return _patch_mr(output_dir)
    if dataset == "amos":
        return _patch_amos(output_dir)
    raise ValueError(dataset)


def _protocol(dataset: str, core):
    if dataset == "amos":
        return core.protocol()
    return core.protocol()


def _save_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False, default=str), encoding="utf-8")
    temporary.replace(path)


def _write_manifest(dataset: str, output_dir: Path, protocol: dict) -> None:
    reference_files = {
        "camchex_ml_decoder": ROOT / "src/third_party/camchex_reference/ml_decoder.py",
        "camchex_model_reference": ROOT / "src/third_party/camchex_reference/models.py",
        "med3dvlm_projector": ROOT / "src/third_party/med3dvlm_reference/mlp.py",
        "m3fm_task_encoder": ROOT / "src/third_party/m3fm_reference/task_encoder.py",
        "m3fm_model_reference": ROOT / "src/third_party/m3fm_reference/m3fm.py",
    }
    manifest = {
        "schema_version": 1,
        "dataset": {"ctrate": "CT-RATE", "amos": "AMOS-MM (7 labels)", "mrrate": "MR-RATE-1K"}[dataset],
        "models": ADAPTED_MODELS,
        "reproduction_mode": "task-adapted",
        "official_model_scope": {
            "CaMCheX": "multi-view chest radiographs plus structured clinical context",
            "Med3DVLM": "3D vision-language pretraining/VQA/report generation",
            "M3FM": "medical 3D foundation model pretraining and report generation",
        },
        "adaptation": {
            "input": "existing cached ordered study-level 768-d visual features plus existing masked report token ids",
            "output": "dataset-specific multilabel logits",
            "CaMCheX-adapted": "two learned complementary ordered-study view slots plus report context",
            "Med3DVLM-adapted": "dual visual pooling streams and a three-token MLP-Mixer-style fusion",
            "M3FM-adapted": "global/half-study multi-scale visual tokens with report cross-attention and gated residual",
        },
        "not_claimed": [
            "not an official checkpoint reproduction",
            "not zero-shot inference from the official projects",
            "not a new raw-image backbone pretraining run",
        ],
        "official_reference_sha256": {
            name: hashlib.sha256(path.read_bytes()).hexdigest()
            for name, path in reference_files.items()
        },
        "adaptation_implementation_sha256": {
            "task_adapted_vlms.py": hashlib.sha256(
                (ROOT / "src/sotas/task2/task_adapted_vlms.py").read_bytes()
            ).hexdigest(),
            "model.yaml": hashlib.sha256(
                (ROOT / "src/configs/task2/model.yaml").read_bytes()
            ).hexdigest(),
        },
        "fairness": "same existing patient-level folds, cached features, report tokenizer, ASL training loop and validation-loss checkpoint selection as the current task-2 multimodal baselines",
        "protocol_sha256": protocol["protocol_sha256"],
    }
    _save_json(output_dir / "task_adaptation_manifest.json", manifest)


def _initialize(dataset: str, output_dir: Path, selected_models: list[str]) -> dict:
    core = _target(dataset, output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    if dataset == "amos":
        protocol = _protocol(dataset, core)
        jobs = [
            {"labels": 7, "model": key, "fold": fold}
            for fold in range(5)
            for key in selected_models
        ]
        _save_json(output_dir / "jobs.json", jobs)
    else:
        core.read_inputs()
        protocol = _protocol(dataset, core)
    saved = output_dir / "protocol.json"
    if saved.exists() and json.loads(saved.read_text(encoding="utf-8")) != protocol:
        raise RuntimeError(f"{output_dir} 已存在不同协议，拒绝混合旧结果")
    _save_json(saved, protocol)
    _write_manifest(dataset, output_dir, protocol)
    return protocol


def _args(output_dir: Path, devices: list[int], worker_index: int, selected_models: list[str]):
    return SimpleNamespace(
        output_dir=output_dir,
        devices=devices,
        worker_index=worker_index,
        models=selected_models,
        compatible_protocols={},
    )


def _amos_worker(output_dir: Path, devices: list[int], worker_index: int, selected_models: list[str]) -> None:
    import fcntl
    import json
    import shutil
    import time
    import torch

    core = _patch_amos(output_dir)
    torch.set_num_threads(2)
    torch.backends.mha.set_fastpath_enabled(False)
    torch.cuda.set_per_process_memory_fraction(float(os.environ.get("TASK_ADAPTED_MEMORY_FRACTION", "0.40")))
    recorded = json.loads((output_dir / "protocol.json").read_text(encoding="utf-8"))
    assert core.protocol() == recorded
    rows = json.loads((core.INPUT / "samples.json").read_text(encoding="utf-8"))
    splits = json.loads((core.INPUT / "splits.json").read_text(encoding="utf-8"))
    jobs = json.loads((output_dir / "jobs.json").read_text(encoding="utf-8"))
    bags = None
    total = len(jobs)
    while True:
        completed = sum((output_dir / f"7_labels" / f"fold_{j['fold'] + 1}" / j["model"] / "result.json").exists() for j in jobs)
        failed = sum((output_dir / f"7_labels" / f"fold_{j['fold'] + 1}" / j["model"] / "error.json").exists() for j in jobs)
        if completed + failed >= total:
            _save_json(output_dir / f"worker_{worker_index}_status.json", {
                "state": "complete" if failed == 0 else "finished_with_errors",
                "completed": completed, "failed": failed, "total": total, "updated_at": time.time(),
            })
            return
        if shutil.disk_usage(output_dir).free < 8 * 1024**3:
            _save_json(output_dir / f"worker_{worker_index}_status.json", {
                "state": "waiting_for_disk", "free_bytes": shutil.disk_usage(output_dir).free,
                "updated_at": time.time(),
            })
            time.sleep(30)
            continue
        claimed = False
        for index, job in enumerate(jobs):
            if index % len(devices) != worker_index:
                continue
            folder = output_dir / "7_labels" / f"fold_{job['fold'] + 1}" / job["model"]
            if (folder / "result.json").exists() or (folder / "error.json").exists():
                continue
            folder.mkdir(parents=True, exist_ok=True)
            with (folder / "run.lock").open("a") as lock:
                try:
                    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
                except BlockingIOError:
                    continue
                if (folder / "result.json").exists() or (folder / "error.json").exists():
                    continue
                claimed = True
                try:
                    if bags is None:
                        bags = core.load_features(rows)
                    core.train_job(job, rows, splits, bags, recorded["protocol_sha256"], worker_index)
                except Exception:
                    error = traceback.format_exc()
                    print(error, flush=True)
                    _save_json(folder / "error.json", {"job": job, "traceback": error, "time": time.time()})
                core.aggregate()
                break
        if not claimed:
            time.sleep(10)


def _worker(dataset: str, output_dir: Path, devices: list[int], worker_index: int, selected_models: list[str]) -> None:
    if dataset == "amos":
        _amos_worker(output_dir, devices, worker_index, selected_models)
        return
    core = _target(dataset, output_dir)
    args = _args(output_dir, devices, worker_index, selected_models)
    core.worker(args)


def _aggregate(dataset: str, output_dir: Path, selected_models: list[str]) -> int:
    core = _target(dataset, output_dir)
    if dataset == "amos":
        core.aggregate()
        result_count = len(list(output_dir.glob("7_labels/fold_*/*/result.json")))
    else:
        args = _args(output_dir, list(range(6)), 0, selected_models)
        digest = json.loads((output_dir / "protocol.json").read_text(encoding="utf-8"))["protocol_sha256"]
        if dataset == "mrrate":
            args.compatible_protocols = {}
        result_count = core.aggregate(args, digest)
    _write_results_report(dataset, output_dir)
    return int(result_count)


def _write_results_report(dataset: str, output_dir: Path) -> None:
    summary_path = output_dir / "summary.csv"
    if not summary_path.exists():
        return
    with summary_path.open(encoding="utf-8-sig", newline="") as stream:
        rows = list(csv.DictReader(stream))
    if dataset == "ctrate":
        title = "# CT-RATE 680例五折三标签分类"
        protocol = "五折 Macro-F1；标准差为五折样本标准差。"
        columns = ("model", "macro_f1_mean", "macro_f1_std", "macro_f1_fixed_0_5_mean", "macro_f1_fixed_0_5_std")
        header = "| 模型 | 五折 Macro-F1 | 固定0.5阈值 Macro-F1 |\n|---|---:|---:|"
    elif dataset == "mrrate":
        title = "# MR-RATE-1K 五折多标签分类"
        protocol = "五折 Macro-F1；标准差为五折样本标准差。"
        columns = ("model", "macro_f1_mean", "macro_f1_std", "macro_f1_fixed_0_5_mean", "macro_f1_fixed_0_5_std")
        header = "| 模型 | 五折 Macro-F1 | 固定0.5阈值 Macro-F1 |\n|---|---:|---:|"
    else:
        title = "# AMOS-MM 七标签多模态分类"
        protocol = "5次固定开发运行；标准差为运行样本标准差，固定200例人工测试集口径沿用现有 AMOS-MM 流程。"
        columns = ("model", "macro_f1_mean", "macro_f1_std", "copositive_macro_f1_mean")
        header = "| 模型 | Macro-F1 | 共阳性 Macro-F1 |\n|---|---:|---:|"
    lines = [title, "", protocol, "", header]
    for row in rows:
        model = row[columns[0]]
        if dataset == "amos":
            lines.append(
                f"| {model} | {float(row[columns[1]]):.4f} ± {float(row[columns[2]]):.4f} | "
                f"{float(row[columns[3]]):.4f} |"
            )
        else:
            lines.append(
                f"| {model} | {float(row[columns[1]]):.4f} ± {float(row[columns[2]]):.4f} | "
                f"{float(row[columns[3]]):.4f} ± {float(row[columns[4]]):.4f} |"
            )
    lines.extend([
        "",
        "结果属于任务适配重训练，不是官方 checkpoint 复现；不覆盖旧表2结果。",
    ])
    (output_dir / "results.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def _launch(dataset: str, output_dir: Path, devices: list[int], selected_models: list[str]) -> None:
    protocol = _initialize(dataset, output_dir, selected_models)
    jobs = 5 * len(selected_models)
    if dataset == "amos":
        jobs = len(json.loads((output_dir / "jobs.json").read_text(encoding="utf-8")))
    handles = []
    processes = []
    started = time.time()
    for worker_index, device in enumerate(devices):
        env = os.environ.copy()
        env.update(
            CUDA_VISIBLE_DEVICES=str(device),
            OMP_NUM_THREADS="2",
            TOKENIZERS_PARALLELISM="false",
        )
        command = [
            sys.executable, "-u", str(Path(__file__).resolve()),
            "--dataset", dataset,
            "--output-dir", str(output_dir),
            "--devices", *map(str, devices),
            "--worker-index", str(worker_index),
            "--models", *selected_models,
        ]
        handle = (output_dir / f"worker_{worker_index}.log").open("a", encoding="utf-8")
        handles.append(handle)
        processes.append(subprocess.Popen(command, env=env, stdout=handle, stderr=subprocess.STDOUT))
    _save_json(output_dir / "run_state.json", {
        "status": "running", "dataset": dataset, "models": selected_models,
        "devices": devices, "worker_pids": [p.pid for p in processes],
        "expected_fold_jobs": jobs, "protocol_sha256": protocol["protocol_sha256"],
        "started_unix": started,
    })
    while any(process.poll() is None for process in processes):
        try:
            completed = _aggregate(dataset, output_dir, selected_models)
            print(f"{dataset}: {completed}/{jobs} 个折次完成", flush=True)
        except Exception:
            print(traceback.format_exc(), flush=True)
        time.sleep(20)
    completed = _aggregate(dataset, output_dir, selected_models)
    codes = [process.returncode for process in processes]
    _save_json(output_dir / "run_state.json", {
        "status": "complete" if completed == jobs and not any(codes) else "incomplete",
        "dataset": dataset, "models": selected_models, "devices": devices,
        "worker_pids": [p.pid for p in processes], "expected_fold_jobs": jobs,
        "completed_fold_jobs": completed, "worker_exit_codes": codes,
        "protocol_sha256": protocol["protocol_sha256"], "started_unix": started,
        "finished_unix": time.time(), "wall_seconds": time.time() - started,
    })
    for handle in handles:
        handle.close()
    if completed != jobs or any(codes):
        raise SystemExit(1)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", choices=tuple(DEFAULT_OUTPUTS), required=True)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--devices", type=int, nargs="+", default=[0, 1, 2, 3, 4, 5])
    parser.add_argument("--models", nargs="+", choices=list(ADAPTED_MODELS), default=list(ADAPTED_MODELS))
    parser.add_argument("--worker-index", type=int)
    parser.add_argument("--aggregate", action="store_true")
    parser.add_argument("--audit-only", action="store_true")
    args = parser.parse_args()
    output_dir = args.output_dir or DEFAULT_OUTPUTS[args.dataset]
    if args.worker_index is not None:
        _worker(args.dataset, output_dir, args.devices, args.worker_index, args.models)
        return
    if args.aggregate:
        count = _aggregate(args.dataset, output_dir, args.models)
        print(f"已汇总 {count} 个折次。", flush=True)
        return
    protocol = _initialize(args.dataset, output_dir, args.models)
    if args.audit_only:
        print(json.dumps({"dataset": args.dataset, "models": args.models,
                          "protocol_sha256": protocol["protocol_sha256"]}, ensure_ascii=False), flush=True)
        return
    with (output_dir / "run.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        _launch(args.dataset, output_dir, args.devices, args.models)


if __name__ == "__main__":
    main()
