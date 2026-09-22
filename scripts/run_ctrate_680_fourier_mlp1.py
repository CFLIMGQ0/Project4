#!/usr/bin/env python3
"""CT-RATE五折Fourier投影一层MLP消融训练入口。"""
from __future__ import annotations

import argparse
import fcntl
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time

import yaml

ROOT = Path(__file__).resolve().parents[2]
SCRIPTS = ROOT / "src/scripts"
sys.path.insert(0, str(SCRIPTS))
import run_ctrate_680_amef_no_pe as core

MODEL_KEY = "amef_multimodal"
POSITION_VARIANT = "apro_full"
MODELS = {MODEL_KEY: "AMEF-MIL (Fourier后单层MLP)"}
LABELS = core.LABELS
INPUT = core.INPUT
SETTINGS = core.SETTINGS
ORIGINAL_PROTOCOL = core.ORIGINAL_PROTOCOL


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def parameters():
    from scripts.task3_apro_cope_ablation_scheduler import base_model_params

    main = yaml.safe_load((ROOT / "src/configs/task3/t3_main_model.yaml").read_text())
    model = base_model_params(POSITION_VARIANT)
    model["apro_fourier_mlp_layers"] = 1
    model["label_query_consistency_weight"] = main["model"]["params"]["label_query_consistency_weight"]
    model["image_aux_weight"] = 0.0
    return {MODEL_KEY: model}


def protocol():
    value = ORIGINAL_PROTOCOL()
    root_prefix = str(ROOT.resolve()) + "/"

    def stable_paths(item):
        if isinstance(item, dict):
            return {
                (key.removeprefix(root_prefix) if isinstance(key, str) and key.startswith(root_prefix) else key):
                stable_paths(child)
                for key, child in item.items()
            }
        if isinstance(item, list):
            return [stable_paths(child) for child in item]
        if isinstance(item, str) and item.startswith(root_prefix):
            return item.removeprefix(root_prefix)
        return item

    value = stable_paths(value)
    value["models"] = MODELS
    value["model_parameters"] = parameters()
    value["position_ablation"] = (
        "position_variant=apro_full; absolute visual and relative attention routes retained; "
        "all Fourier projections use one Linear layer instead of Linear-GELU-Linear"
    )
    value["runner"] = "src/scripts/run_ctrate_680_fourier_mlp1.py"
    value["source_sha256"][value["runner"]] = sha256(Path(__file__).resolve())
    value.pop("protocol_sha256", None)
    value["protocol_sha256"] = hashlib.sha256(
        json.dumps(value, sort_keys=True, ensure_ascii=False).encode()
    ).hexdigest()
    return value


def build_model(key, params):
    if key != MODEL_KEY:
        raise ValueError(key)
    return core.amef_base.build_model(key, params)


def configure():
    core.POSITION_VARIANT = POSITION_VARIANT
    core.INPUT = INPUT
    core.LABELS = LABELS
    core.MODELS = MODELS
    core.SETTINGS = SETTINGS
    core.parameters = parameters
    core.protocol = protocol
    core.build_model = build_model
    core.configure()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path,
                        default=ROOT / "outputs/ct_rate_680/amef_fourier_mlp1_fivefold")
    parser.add_argument("--devices", type=int, nargs="+", default=[0, 1, 2, 3])
    parser.add_argument("--models", nargs="+", choices=[MODEL_KEY], default=None)
    parser.add_argument("--worker-index", type=int)
    parser.add_argument("--audit-only", action="store_true")
    parser.add_argument("--aggregate", action="store_true")
    args = parser.parse_args()
    configure()
    if args.worker_index is not None:
        core.core.worker(args)
        return
    args.output_dir.mkdir(parents=True, exist_ok=True)
    with (args.output_dir / "run.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        current = protocol()
        saved = args.output_dir / "protocol.json"
        if saved.exists() and json.loads(saved.read_text())["protocol_sha256"] != current["protocol_sha256"]:
            raise ValueError("Fourier单层MLP实验协议或源码改变，禁止混用已有结果")
        core.core.save_json(saved, current)
        if args.audit_only:
            print(json.dumps({"protocol_sha256": current["protocol_sha256"], "fold_jobs": 5,
                              "position_variant": POSITION_VARIANT, "fourier_mlp_layers": 1}, ensure_ascii=False))
            return
        if args.aggregate:
            print(f"已汇总{core.core.aggregate(args, current['protocol_sha256'])}/5个折次。", flush=True)
            return
        handles, processes = [], []
        started = time.time()
        for index, _device in enumerate(args.devices):
            environment = os.environ.copy()
            environment.update(CUDA_VISIBLE_DEVICES=str(_device), OMP_NUM_THREADS="4",
                               TOKENIZERS_PARALLELISM="false")
            command = [sys.executable, "-u", str(Path(__file__).resolve()),
                       "--worker-index", str(index), "--devices", *map(str, args.devices),
                       "--output-dir", str(args.output_dir)]
            handle = (args.output_dir / f"worker_{index}.log").open("a", encoding="utf-8")
            handles.append(handle)
            processes.append(subprocess.Popen(command, cwd=SCRIPTS, env=environment,
                                              stdout=handle, stderr=subprocess.STDOUT))
        core.core.save_json(args.output_dir / "run_state.json", {"status": "running",
                      "worker_pids": [p.pid for p in processes], "devices": args.devices,
                      "started_unix": started, "fourier_mlp_layers": 1})
        while any(process.poll() is None for process in processes):
            completed = core.core.aggregate(args, current["protocol_sha256"])
            print(f"CT-RATE Fourier单层MLP完成：{completed}/5", flush=True)
            time.sleep(15)
        completed = core.core.aggregate(args, current["protocol_sha256"])
        codes = [process.returncode for process in processes]
        status = "complete" if completed == 5 and not any(codes) else "incomplete"
        core.core.save_json(args.output_dir / "run_state.json", {"status": status,
                      "completed_fold_jobs": completed, "expected_fold_jobs": 5,
                      "worker_exit_codes": codes, "started_unix": started,
                      "finished_unix": time.time(), "wall_seconds": time.time() - started,
                      "fourier_mlp_layers": 1})
        for handle in handles:
            handle.close()
        if status != "complete":
            raise SystemExit(1)


if __name__ == "__main__":
    main()
