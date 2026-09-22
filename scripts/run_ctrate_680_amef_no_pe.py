#!/usr/bin/env python3
"""CT-RATE 680例 AMEF-MIL 位置变体五折实验。"""
from __future__ import annotations

import argparse
import copy
import fcntl
import json
import os
from pathlib import Path
import subprocess
import sys
import time

import yaml


ROOT = Path(__file__).resolve().parents[2]
SCRIPTS = ROOT / "src" / "scripts"
OUTPUT = ROOT / "outputs" / "ct_rate_680" / "amef_no_pe_fivefold"
MR_STATUS = ROOT / "outputs" / "mr_rate_1k" / "all_models_fivefold" / "gpu_pool_status.json"
MODEL_KEY = "amef_multimodal"
POSITION_VARIANT = "no_pe"
MODELS = {MODEL_KEY: "AMEF-MIL (no APro/PE)"}

sys.path.insert(0, str(SCRIPTS))
import run_ctrate_680_all_models as core
import run_cq500_amef_multimodal as amef_base


ORIGINAL_PROTOCOL = core.protocol
ORIGINAL_PARAMETERS = core.parameters
LABELS = core.LABELS
INPUT = core.INPUT
SETTINGS = {**core.SETTINGS, "max_instances": 64, "text_max_length": 512}


def parameters():
    from scripts.task3_apro_cope_ablation_scheduler import base_model_params

    main = yaml.safe_load((ROOT / "src/configs/task3/t3_main_model.yaml").read_text())
    model = base_model_params(POSITION_VARIANT)
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
    value["position_ablation"] = {
        "no_pe": "position_variant=no_pe; APro-CoPE and all positional encoding removed",
        "original_pe": "position_variant=original_pe; APro-CoPE removed, standard positional encoding retained",
        "apro_absolute_only": "position_variant=apro_absolute_only; APro-CoPE absolute visual-feature route only",
        "apro_relative_only": "position_variant=apro_relative_only; APro-CoPE relative attention-bias route only",
    }[POSITION_VARIANT]
    value["runner"] = "src/scripts/run_ctrate_680_amef_no_pe.py"
    value["source_sha256"]["src/scripts/run_ctrate_680_amef_no_pe.py"] = core.sha256(Path(__file__).resolve())
    value.pop("protocol_sha256", None)
    value["protocol_sha256"] = core.hashlib.sha256(
        json.dumps(value, sort_keys=True, ensure_ascii=False).encode()
    ).hexdigest()
    return value


def build_model(key, params):
    if key != MODEL_KEY:
        raise ValueError(f"未知模型：{key}")
    return amef_base.build_model(key, params)


def configure():
    core.INPUT = INPUT
    core.LABELS = LABELS
    core.MODELS = MODELS
    core.IMAGE_MODELS = {}
    core.TEXT_MODELS = {}
    core.FUSION_MODELS = {}
    core.SETTINGS = SETTINGS
    core.amef_base = amef_base
    core.parameters = parameters
    core.protocol = protocol
    core.build_model = build_model
    core.configure_adapters()


def configure_variant(position_variant):
    global MODELS, POSITION_VARIANT
    POSITION_VARIANT = position_variant
    display_name = {
        "no_pe": "AMEF-MIL (no APro/PE)",
        "original_pe": "AMEF-MIL (original PE, no APro)",
        "apro_absolute_only": "AMEF-MIL (APro absolute visual route only)",
        "apro_relative_only": "AMEF-MIL (APro relative attention route only)",
    }[position_variant]
    MODELS = {MODEL_KEY: display_name}


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--position-variant",
        choices=("no_pe", "original_pe", "apro_absolute_only", "apro_relative_only"),
        default="no_pe",
    )
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--devices", type=int, nargs="+", default=[0, 1, 2, 3])
    parser.add_argument("--models", nargs="+", default=None)
    parser.add_argument("--worker-index", type=int)
    parser.add_argument("--audit-only", action="store_true")
    parser.add_argument("--aggregate", action="store_true")
    parser.add_argument("--wait-for-mr", action="store_true")
    return parser.parse_args()


def wait_for_mr():
    while True:
        state = "missing"
        if MR_STATUS.is_file():
            try:
                state = json.loads(MR_STATUS.read_text()).get("state", "unknown")
            except json.JSONDecodeError:
                state = "unreadable"
        if state != "running":
            print(f"MR-RATE状态为{state}，开始CT-RATE {POSITION_VARIANT}实验。", flush=True)
            return
        print(f"MR-RATE仍在运行，CT-RATE {POSITION_VARIANT}实验排队等待。", flush=True)
        time.sleep(60)


def available_devices(devices: list[int], minimum_free_mib: int = 7500) -> list[int]:
    result = subprocess.run(
        ["nvidia-smi", "--query-gpu=index,memory.free", "--format=csv,noheader,nounits"],
        check=True,
        capture_output=True,
        text=True,
    )
    free = {
        int(index.strip()): int(value.strip())
        for index, value in (line.split(",", 1) for line in result.stdout.splitlines() if line.strip())
    }
    return [device for device in devices if free.get(device, 0) >= minimum_free_mib]


def wait_for_devices(devices: list[int]) -> list[int]:
    while True:
        selected = available_devices(devices)
        if selected:
            print(f"CT-RATE {POSITION_VARIANT}使用本机GPU：{selected}。", flush=True)
            return selected
        print("暂时没有显存达到7.5GB的本机GPU，继续等待。", flush=True)
        time.sleep(30)


def main():
    args = parse_args()
    configure_variant(args.position_variant)
    if args.output_dir is None:
        args.output_dir = ROOT / "outputs" / "ct_rate_680" / f"amef_{POSITION_VARIANT}_fivefold"
    configure()
    if args.wait_for_mr and args.worker_index is None and not args.audit_only and not args.aggregate:
        wait_for_mr()
        args.devices = wait_for_devices(args.devices)
    if args.worker_index is not None:
        core.worker(args)
        return

    args.output_dir.mkdir(parents=True, exist_ok=True)
    with (args.output_dir / "run.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        current = protocol()
        saved = args.output_dir / "protocol.json"
        if saved.exists() and json.loads(saved.read_text())["protocol_sha256"] != current["protocol_sha256"]:
            raise ValueError("no_pe协议或源码改变，禁止混用已有结果")
        core.save_json(saved, current)
        if args.audit_only:
            print(json.dumps({"protocol_sha256": current["protocol_sha256"], "fold_jobs": 5,
                              "position_variant": POSITION_VARIANT, "labels": LABELS}, ensure_ascii=False, indent=2))
            return
        if args.aggregate:
            print(f"已汇总{core.aggregate(args, current['protocol_sha256'])}/5个折次。", flush=True)
            return

        handles, processes = [], []
        started = time.time()
        for index, device in enumerate(args.devices):
            environment = os.environ.copy()
            environment.update(CUDA_VISIBLE_DEVICES=str(device), OMP_NUM_THREADS="4",
                               TOKENIZERS_PARALLELISM="false")
            command = [sys.executable, "-u", str(Path(__file__).resolve()),
                       "--worker-index", str(index), "--devices", *map(str, args.devices),
                       "--output-dir", str(args.output_dir), "--position-variant", POSITION_VARIANT]
            handle = (args.output_dir / f"worker_{index}.log").open("a", encoding="utf-8")
            handles.append(handle)
            processes.append(subprocess.Popen(command, cwd=SCRIPTS, env=environment,
                                              stdout=handle, stderr=subprocess.STDOUT))
        core.save_json(args.output_dir / "run_state.json", {
            "status": "running", "worker_pids": [p.pid for p in processes],
            "devices": args.devices, "started_unix": started,
            "position_variant": POSITION_VARIANT, "expected_fold_jobs": 5,
        })
        while any(process.poll() is None for process in processes):
            completed = core.aggregate(args, current["protocol_sha256"])
            print(f"CT-RATE {POSITION_VARIANT}完成：{completed}/5个折次", flush=True)
            time.sleep(15)
        completed = core.aggregate(args, current["protocol_sha256"])
        codes = [process.returncode for process in processes]
        state = "complete" if completed == 5 and not any(codes) else "incomplete"
        core.save_json(args.output_dir / "run_state.json", {
            "status": state, "completed_fold_jobs": completed, "expected_fold_jobs": 5,
            "worker_exit_codes": codes, "started_unix": started, "finished_unix": time.time(),
            "wall_seconds": time.time() - started, "position_variant": POSITION_VARIANT,
        })
        for handle in handles:
            handle.close()
        if state != "complete":
            raise SystemExit(1)


if __name__ == "__main__":
    main()
