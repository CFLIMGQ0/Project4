#!/usr/bin/env python3
"""两主机任务级并行：本机持有任务锁，远端核验同版代码后计算并回传结果。"""
from __future__ import annotations

import argparse
import fcntl
import json
import os
from pathlib import Path
import shlex
import subprocess
import sys
import time
import traceback

ROOT = Path(__file__).resolve().parents[2]
POOL = ROOT / "outputs/amos_mm/compute_pool"
HOST = "Lim@172.16.170.202"
REMOTE = "/home/Lim/Project4/outputs/amos_mm/compute_pool_20260915"
SSH = ["ssh", "-i", "/home/Lim/.ssh/id_ed25519_project4_pool", "-o", "IdentitiesOnly=yes",
       "-o", "BatchMode=yes", "-o", "ConnectTimeout=8", "-o", "ServerAliveInterval=15", "-o", "ServerAliveCountMax=3"]
MINIMUM_FREE_MIB = 12000


def command(args, **kwargs):
    return subprocess.run(args, check=True, text=True, **kwargs)


def sync(source, target, *options):
    command(["rsync", "-a", "--checksum", "-e", shlex.join(SSH), *options, str(source), target])


def stage():
    import run_amos_mm_all_models as run
    POOL.mkdir(parents=True, exist_ok=True)
    with (POOL / "stage.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        p = json.loads((run.OUT / "protocol.json").read_text())
        files = {str(Path(v).relative_to(ROOT)) for v in p["source_sha256"]}
        for name in ("tasks", "baselines", "sotas", "model", "exp_4", "exp_8", "exp_10", "training"):
            files.update(str(v.relative_to(ROOT)) for v in (ROOT / "src" / name).rglob("*.py"))
        files.update({"src/scripts/run_cq500_table2_multimodal_baselines.py", "src/scripts/run_cq500_table2_text_baselines.py",
                      "src/scripts/amos_mm_remote_pool.py", "src/configs/task2/exp10_text_classification.yaml",
                      "src/configs/task2/model.yaml", "src/configs/task3/t3_main_model.yaml",
                      "outputs/amos_mm/all_models/protocol.json"})
        manifest = POOL / "snapshot_files.txt"
        manifest.write_text("\n".join(sorted(files)) + "\n", encoding="utf-8")
        snapshot = {"protocol_sha256": p["protocol_sha256"], "files": {name: run.sha256(ROOT / name) for name in files},
                    "versions": {"torch": "2.12.1+cu126", "torchvision": "0.27.1+cu126", "numpy": "2.4.4",
                                 "sklearn": "1.9.0", "scipy": "1.18.0"}}
        run.save_json(POOL / "snapshot_manifest.json", snapshot)
        command([*SSH, HOST, shlex.join(["mkdir", "-p", REMOTE + "/outputs/amos_mm/compute_pool", REMOTE + "/runtime"])])
        sync(str(ROOT) + "/", f"{HOST}:{REMOTE}/", "--files-from=" + str(manifest))
        sync(POOL / "snapshot_manifest.json", f"{HOST}:{REMOTE}/outputs/amos_mm/compute_pool/snapshot_manifest.json")
        wheels = list((POOL / "wheels").glob("numpy-2.4.4-*.whl"))
        assert len(wheels) == 1, "需要准备与本机完全相同的NumPy轮子"
        sync(wheels[0], f"{HOST}:{REMOTE}/runtime/")
        # 仅解包已固定的wheel到任务私有路径，避免依赖远端损坏的pip或更改全局环境。
        install = ["/home/Lim/conda/envs/myenv/bin/python", "-m", "zipfile", "-e",
                   REMOTE + "/runtime/" + wheels[0].name, REMOTE + "/runtime/python"]
        command([*SSH, HOST, shlex.join(install)])
        run.save_json(POOL / "staged.json", {"protocol_sha256": p["protocol_sha256"], "remote_root": REMOTE,
                                            "time": time.time(), "note": "独立运行副本，不修改远端旧项目和全局环境"})


def remote_command(gpu, *args):
    return [*SSH, HOST, "cd " + shlex.quote(REMOTE + "/src") + " && " + shlex.join([
        "env", f"CUDA_VISIBLE_DEVICES={gpu}", "OMP_NUM_THREADS=2", "OPENBLAS_NUM_THREADS=2",
        "LD_LIBRARY_PATH=/home/Lim/conda/envs/myenv/lib", "PYTHONNOUSERSITE=1",
        "PYTHONPATH=" + REMOTE + "/runtime/python", "/home/Lim/conda/envs/myenv/bin/python", "-u",
        "scripts/amos_mm_remote_pool.py", *args])]


def verify_remote():
    import importlib
    import run_amos_mm_all_models as run
    snapshot = json.loads((POOL / "snapshot_manifest.json").read_text())
    for name, digest in snapshot["files"].items():
        assert run.sha256(ROOT / name) == digest, f"远端文件不一致：{name}"
    versions = {k: importlib.import_module(k).__version__ for k in snapshot["versions"]}
    assert versions == snapshot["versions"], versions
    p = json.loads((run.OUT / "protocol.json").read_text())
    assert p["protocol_sha256"] == snapshot["protocol_sha256"]
    assert json.dumps(run.SETTINGS, sort_keys=True) == json.dumps(p["settings"], sort_keys=True)
    assert json.dumps({k: run.parameters(k) for k in run.MODELS}, sort_keys=True) == json.dumps(p["model_parameters"], sort_keys=True)
    run.save_json(POOL / "runtime_verified.json", {"versions": versions, "files": len(snapshot["files"]),
                  "protocol_sha256": p["protocol_sha256"], "time": time.time()})
    print("远端源码、输入、超参数及运行库版本核验一致。", flush=True)
    return p


def remote_worker(request_path):
    import run_amos_mm_all_models as run
    p = verify_remote()
    run.torch.set_num_threads(2)
    run.torch.backends.mha.set_fastpath_enabled(False)
    run.torch.cuda.set_per_process_memory_fraction(.40)
    request = json.loads(Path(request_path).read_text())
    assert request["protocol_sha256"] == p["protocol_sha256"]
    rows = json.loads((run.INPUT / "samples.json").read_text())
    splits = json.loads((run.INPUT / "splits.json").read_text())
    bags = run.load_features(rows)
    run.train_job(request["job"], rows, splits, bags, p["protocol_sha256"], request["worker"])


def dispatch(gpu, worker):
    import run_amos_mm_all_models as run
    import numpy as np
    run.torch.set_num_threads(2)
    POOL.mkdir(parents=True, exist_ok=True)
    jobs = json.loads((run.OUT / "jobs.json").read_text())
    p = json.loads((run.OUT / "protocol.json").read_text())
    copied = False
    while True:
        remaining = [j for j in jobs if not (run.job_folder(j) / "result.json").exists()
                     and not (run.job_folder(j) / "error.json").exists()]
        if not remaining:
            run.save_json(run.OUT / f"worker_{worker}_status.json", {"state": "complete", "host": HOST, "gpu": gpu, "updated_at": time.time()})
            break
        try:
            ready = json.loads((run.INPUT / "feature_status.json").read_text())["state"] == "complete"
            free = int(command([*SSH, HOST, f"nvidia-smi -i {gpu} --query-gpu=memory.free --format=csv,noheader,nounits"],
                               capture_output=True).stdout.strip())
            if not ready or free < MINIMUM_FREE_MIB:
                run.save_json(run.OUT / f"worker_{worker}_status.json", {"state": "waiting_for_features" if not ready else "waiting_for_gpu",
                              "host": HOST, "gpu": gpu, "free_mib": free, "minimum_free_mib": MINIMUM_FREE_MIB, "updated_at": time.time()})
                time.sleep(20)
                continue
            if not copied:
                with (POOL / "feature_sync.lock").open("a") as lock:
                    fcntl.flock(lock, fcntl.LOCK_EX)
                    command([*SSH, HOST, shlex.join(["mkdir", "-p", REMOTE + "/outputs/amos_mm/experiment/features"])])
                    sync(str(run.INPUT / "features") + "/", f"{HOST}:{REMOTE}/outputs/amos_mm/experiment/features/",
                         "--include=*.npz", "--exclude=*")
                    copied = True
            claimed = False
            for job in remaining:
                folder = run.job_folder(job)
                folder.mkdir(parents=True, exist_ok=True)
                with (folder / "run.lock").open("a") as lock:
                    try:
                        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
                    except BlockingIOError:
                        continue
                    if (folder / "result.json").exists() or (folder / "error.json").exists():
                        continue
                    claimed = True
                    request = POOL / f"request_{worker}.json"
                    run.save_json(request, {"job": job, "worker": worker, "protocol_sha256": p["protocol_sha256"]})
                    remote_request = REMOTE + f"/outputs/amos_mm/compute_pool/request_{worker}.json"
                    sync(request, f"{HOST}:{remote_request}")
                    run.save_json(run.OUT / f"worker_{worker}_status.json", {"state": "remote_training", "host": HOST,
                                  "gpu": gpu, "job": job, "updated_at": time.time()})
                    try:
                        command(remote_command(gpu, "--remote-worker", remote_request))
                        remote_folder = REMOTE + "/" + str(folder.relative_to(ROOT))
                        sync(f"{HOST}:{remote_folder}/", str(folder) + "/", "--exclude=run.lock", "--exclude=result.json")
                        temp = folder / f"remote_result_{worker}.json"
                        sync(f"{HOST}:{remote_folder}/result.json", str(temp))
                        result = json.loads(temp.read_text())
                        assert result["protocol_sha256"] == p["protocol_sha256"] and result["job"] == job
                        with np.load(folder / "test_predictions.npz", allow_pickle=False) as v:
                            recomputed = run.test_metrics(v["labels"], v["probabilities"], v["thresholds"])
                        assert abs(recomputed["macro_f1"] - result["macro_f1"]) < 1e-12
                        run.save_json(folder / "execution_host.json", {"host": HOST, "gpu": gpu, "memory_fraction": .40,
                                      "source_snapshot": str(POOL / "snapshot_manifest.json"), "metric_recomputed_locally": True})
                        run.save_json(folder / "result.json", result)
                    except Exception:
                        error = traceback.format_exc()
                        print(error, flush=True)
                        run.save_json(folder / "error.json", {"job": job, "host": HOST, "traceback": error, "time": time.time()})
                    run.aggregate()
                    break
            if not claimed:
                time.sleep(20)
        except Exception:
            print(traceback.format_exc(), flush=True)
            time.sleep(30)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", action="store_true")
    parser.add_argument("--verify", action="store_true")
    parser.add_argument("--remote-worker")
    parser.add_argument("--dispatch", type=int)
    parser.add_argument("--worker", type=int, default=5)
    args = parser.parse_args()
    if args.stage:
        stage()
    elif args.verify:
        verify_remote()
    elif args.remote_worker:
        remote_worker(args.remote_worker)
    elif args.dispatch is not None:
        dispatch(args.dispatch, args.worker)
    else:
        parser.error("请选择准备副本、验证、远端计算或本机调度")
