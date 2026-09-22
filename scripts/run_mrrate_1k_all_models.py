#!/usr/bin/env python3
"""MR-RATE-1K论文表2全模型五折实验。"""
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

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src/scripts"))
import run_ctrate_680_all_models as core
import run_mrrate_1k_amef as amef_base
from prepare_mrrate_1k_experiment import LABELS, OUT as INPUT, save_json, sha256

IMAGE_MODELS = {k: v["display_name"] for k, v in core.image_base.MODEL_SPECS.items()}
TEXT_MODELS = dict(core.TEXT_MODELS)
FUSION_MODELS = dict(core.fusion_base.MODELS)
MODELS = {**IMAGE_MODELS, **TEXT_MODELS, **FUSION_MODELS, "amef_multimodal": "AMEF-MIL"}
SETTINGS = {**core.fusion_base.SETTINGS, "max_instances": 64, "text_max_length": 512,
            "epochs": 30, "batch_size": 16, "learning_rate": 2e-4, "weight_decay": 0.02,
            "base_seed": 42}
RECOVERY = ROOT / "outputs/mr_rate_1k/all_models_fivefold/recovery_protocol.json"


def result_compatibility():
    if not RECOVERY.exists():
        return {}
    recovered = json.loads(RECOVERY.read_text())["compatible_protocols"]
    if "amef_multimodal" in recovered or not set(recovered).issubset(MODELS):
        raise ValueError("恢复白名单只能包含未受四标签文本查询损失修复影响的基线")
    return recovered


def normalize_rows():
    rows = json.loads((INPUT / "samples.json").read_text())
    for row in rows:
        row["patient_id"] = row["case_index"]
        row["Findings_EN_masked"] = row["findings_masked"]
    return rows


def read_inputs():
    rows = normalize_rows()
    folds = json.loads((INPUT / "patient_folds.json").read_text())
    assert len(rows) == len({r["patient_id"] for r in rows}) == 1000
    assert [r["case_index"] for r in rows] == list(range(1000))
    assert sorted(i for group in folds["folds"] for i in group) == list(range(1000))
    assert folds["labels"] == LABELS and [len(group) for group in folds["folds"]] == [200] * 5
    y = np.asarray([r["labels"] for r in rows], dtype=np.int64)
    for group, expected in zip(folds["folds"], folds["fold_positive_counts"]):
        assert y[group].sum(0).tolist() == expected
    return rows, folds, y


def parameters():
    import yaml
    from scripts.task3_apro_cope_ablation_scheduler import base_model_params
    config = yaml.safe_load((ROOT / "src/configs/task2/model.yaml").read_text())
    values = {k: config["models"][k] for k in FUSION_MODELS}
    main = yaml.safe_load((ROOT / "src/configs/task3/t3_main_model.yaml").read_text())
    values["amef_multimodal"] = base_model_params("apro_full")
    values["amef_multimodal"]["label_query_consistency_weight"] = main["model"]["params"]["label_query_consistency_weight"]
    values["amef_multimodal"]["image_aux_weight"] = 0.0
    values.update({k: v for k, v in core.image_base.MODEL_SPECS.items()})
    return values


def text_config(fold):
    import yaml
    config = yaml.safe_load((ROOT / "src/configs/task2/exp10_text_classification.yaml").read_text())
    config["seed"] = SETTINGS["base_seed"] + 100 * (fold + 1)
    config["data"].update(label_names=LABELS, text_field="findings_masked", max_length=512)
    config["training"]["num_workers"] = 0
    config["experiment_name"] = "mrrate_1k_table2_fivefold"
    config["paths"] = {"data_json": "outputs/mr_rate_1k/experiment/samples.json"}
    return config


def protocol():
    tracked = [Path(__file__), Path(core.__file__), Path(amef_base.__file__),
               ROOT / "src/scripts/prepare_mrrate_1k_experiment.py",
               ROOT / "src/scripts/run_cq500_table2_multimodal_baselines.py",
               ROOT / "src/scripts/run_physionet_ct_ich_table2_baselines.py",
               ROOT / "src/scripts/task3_apro_cope_ablation_scheduler.py",
               ROOT / "src/exp_10/models.py", ROOT / "src/exp_10/data.py",
               ROOT / "src/exp_10/train_text_classification.py", ROOT / "src/exp_8/models.py",
               ROOT / "src/sotas/task2/multimodal_sotas.py", ROOT / "src/training/losses.py",
               ROOT / "src/configs/task2/model.yaml", ROOT / "src/configs/task2/exp10_text_classification.yaml",
               ROOT / "src/configs/task3/t3_main_model.yaml", INPUT / "samples.json",
               INPUT / "patient_folds.json", INPUT / "preparation_protocol.json"]
    value = {"dataset": "MR-RATE-1K", "cases": 1000, "labels": LABELS, "models": MODELS,
             "settings_image_and_multimodal": SETTINGS, "text_config": text_config(0),
             "model_parameters": parameters(), "source_sha256": {str(p.relative_to(ROOT)): sha256(p) for p in tracked},
             "split": "固定患者五折：test=k，validation=(k+1)%5，其余三折训练；600/200/200",
             "text": "仅官方findings掩码文本，不输入impression、clinical_information或technique",
             "image": "全部MRI series组成有序序列，最多64实例，冻结ConvNeXt-Tiny特征",
             "amef": "APro-CoPE、label-wise attention、label hypergraph、TextCNN、查询交叉注意力和门控融合",
             "selection": "图像/多模态按验证损失选最优；文本按验证Macro-F1和原早停协议；测试集只用于最终评估",
             "primary_metric": "五折测试Macro-F1均值±样本标准差，另报OOF、逐标签及共阳性结果"}
    if RECOVERY.exists():
        value["recovery"] = json.loads(RECOVERY.read_text())
        value["source_sha256"][str(RECOVERY.relative_to(ROOT))] = sha256(RECOVERY)
    value["protocol_sha256"] = hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False).encode()).hexdigest()
    return value


def load_features(rows):
    core.SOURCE_SLICE_INDICES, core.SOURCE_COUNTS = [], []
    amef_base.SOURCE_POSITIONS, amef_base.SOURCE_COUNTS = [], []
    digest = json.loads((INPUT / "preparation_protocol.json").read_text())["preparation_sha256"]
    bags = []
    for row in core.tqdm(rows, desc="读取MR-RATE-1K共用特征"):
        path = INPUT / "features" / f"{row['case_index']:04d}.npz"
        if not path.exists():
            raise FileNotFoundError(path)
        with np.load(path, allow_pickle=False) as cache:
            assert str(cache["study_uid"]) == row["study_uid"]
            assert str(cache["preparation_sha256"]) == digest
            features = cache["features"].astype(np.float32)
            positions = cache["slice_indices"].astype(np.int64)
            count = int(cache["original_count"])
            assert features.shape == (len(positions), 768) and 1 <= len(positions) <= 64
            assert np.isfinite(features).all() and np.all(np.diff(positions) > 0)
            assert 0 <= positions[0] <= positions[-1] < count
            bags.append(features)
            core.SOURCE_SLICE_INDICES.append(positions)
            core.SOURCE_COUNTS.append(count)
            amef_base.SOURCE_POSITIONS.append(positions)
            amef_base.SOURCE_COUNTS.append(count)
    return bags


def build_model(key, params):
    if key in IMAGE_MODELS:
        return core.image_base.build_model(key), core.image_base.MODEL_SPECS[key].get("aux_weights", {})
    if key == "amef_multimodal":
        return amef_base.build_model(key, params)
    from run_physionet_ct_ich_table2_baselines import CachedFeatureBackbone
    from sotas.task2.multimodal_sotas import build_task2_multimodal_sota

    parameters = dict(params)
    auxiliary = {name.removesuffix("_weight"): parameters.pop(name)
                 for name in list(parameters) if name.endswith("_weight")}
    model = build_task2_multimodal_sota(
        key,
        **parameters,
        pretrained=False,
        num_labels=len(LABELS),
    )
    model.instance_encoder.backbone = CachedFeatureBackbone(
        input_dim=768,
        output_dim=parameters["feature_dim"],
        dropout=parameters["dropout"],
    )
    return model, auxiliary


def configure():
    core.INPUT, core.LABELS, core.ROOT = INPUT, LABELS, ROOT
    core.IMAGE_MODELS, core.TEXT_MODELS = IMAGE_MODELS, TEXT_MODELS
    core.FUSION_MODELS, core.MODELS = FUSION_MODELS, MODELS
    core.SETTINGS = SETTINGS
    core.amef_base = amef_base
    core.read_inputs = read_inputs
    core.parameters = parameters
    core.text_config = text_config
    core.protocol = protocol
    core.load_features = load_features
    core.build_model = build_model
    core.configure_adapters()


def worker(args):
    import torch
    torch.set_num_threads(2)
    torch.backends.mha.set_fastpath_enabled(False)
    torch.cuda.set_per_process_memory_fraction(float(os.environ.get("MRRATE_MEMORY_FRACTION", "0.35")))
    configure()
    args.compatible_protocols = result_compatibility()
    core.worker(args)


def aggregate(args, digest):
    configure()
    args.compatible_protocols = result_compatibility()
    return core.aggregate(args, digest)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=ROOT / "outputs/mr_rate_1k/all_models_fivefold")
    parser.add_argument("--devices", type=int, nargs="+", default=[0, 1, 2, 3, 4, 5])
    parser.add_argument("--models", nargs="+", choices=list(MODELS))
    parser.add_argument("--worker-index", type=int)
    parser.add_argument("--audit-only", action="store_true")
    parser.add_argument("--aggregate", action="store_true")
    args = parser.parse_args()
    configure()
    if args.worker_index is not None:
        worker(args)
        return
    args.output_dir.mkdir(parents=True, exist_ok=True)
    with (args.output_dir / "run.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        read_inputs()
        current = protocol()
        saved = args.output_dir / "protocol.json"
        if saved.exists() and json.loads(saved.read_text())["protocol_sha256"] != current["protocol_sha256"]:
            raise ValueError("MR-RATE表2协议或源码发生变化，禁止混用已有结果")
        save_json(saved, current)
        if args.audit_only:
            features = len(list((INPUT / "features").glob("*.npz")))
            print(f"协议核对通过：{len(MODELS)}个模型、{len(MODELS)*5}个折次、特征{features}/1000。", flush=True)
            return
        if args.aggregate:
            print(f"已汇总{aggregate(args, current['protocol_sha256'])}/{len(MODELS) * 5}个折次。", flush=True)
            return
        handles, processes = [], []
        jobs = len(MODELS) * 5
        for index, device in enumerate(args.devices):
            env = os.environ.copy()
            env.update(CUDA_VISIBLE_DEVICES=str(device), OMP_NUM_THREADS="2", TOKENIZERS_PARALLELISM="false")
            command = [sys.executable, "-u", str(Path(__file__).resolve()), "--worker-index", str(index),
                       "--devices", *map(str, args.devices), "--output-dir", str(args.output_dir)]
            if args.models:
                command += ["--models", *args.models]
            handle = (args.output_dir / f"worker_{index}.log").open("a")
            handles.append(handle)
            processes.append(subprocess.Popen(command, env=env, stdout=handle, stderr=subprocess.STDOUT))
        state = {"status": "running", "worker_pids": [p.pid for p in processes], "devices": args.devices,
                 "models": list(args.models or MODELS), "started_unix": time.time()}
        save_json(args.output_dir / "run_state.json", state)
        while any(p.poll() is None for p in processes):
            completed = aggregate(args, current["protocol_sha256"])
            print(f"MR-RATE表2完成：{completed}/{jobs}个模型折次", flush=True)
            time.sleep(20)
        completed = aggregate(args, current["protocol_sha256"])
        codes = [p.returncode for p in processes]
        state.update(status="complete" if completed == jobs and not any(codes) else "incomplete",
                     completed_fold_jobs=completed, expected_fold_jobs=jobs, worker_exit_codes=codes,
                     finished_unix=time.time())
        save_json(args.output_dir / "run_state.json", state)
        for handle in handles:
            handle.close()
        if state["status"] != "complete":
            raise SystemExit(1)


if __name__ == "__main__":
    main()
