#!/usr/bin/env python3
"""MR-RATE-1K六卡动态表2任务队列，按模型/折次重试并汇总远端结果。"""
from __future__ import annotations

import fcntl
import json
import os
from pathlib import Path
import shlex
import signal
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "src"
OUT = ROOT / "outputs/mr_rate_1k/all_models_fivefold"
INPUT = ROOT / "outputs/mr_rate_1k/experiment"
SCRIPT = SRC / "scripts/run_mrrate_1k_all_models.py"
REMOTE_HOST = "Lim@172.16.170.202"
REMOTE_ROOT = "/home/Lim/mr_rate_1k_pool_20260919"
REMOTE_OUT = REMOTE_ROOT + "/outputs/mr_rate_1k/all_models_fivefold"
SSH = ["ssh", "-i", "/home/Lim/.ssh/id_ed25519_project4_pool", "-o", "IdentitiesOnly=yes",
       "-o", "BatchMode=yes", "-o", "ConnectTimeout=8", "-o", "ServerAliveInterval=15",
       "-o", "ServerAliveCountMax=3"]
LOCAL_GPUS = (0, 1, 2, 3)
REMOTE_GPUS = (0, 1)
DEVICES = (0, 1, 2, 3, 4, 5)
MINIMUM_FREE_MIB = 7500
POLL_SECONDS = 20
MAX_RESTARTS = 3


def save_json(path, value):
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


def command(args, **kwargs):
    return subprocess.run(args, check=True, text=True, **kwargs)


def local_free():
    output = command(["nvidia-smi", "--query-gpu=index,memory.free", "--format=csv,noheader,nounits"],
                     capture_output=True).stdout
    return {int(row.split(",")[0]): int(row.split(",")[1]) for row in output.strip().splitlines()}


def remote_script(script, capture_output=False):
    return command([*SSH, REMOTE_HOST, script], capture_output=capture_output)


def remote_free():
    output = remote_script("nvidia-smi --query-gpu=index,memory.free --format=csv,noheader,nounits", capture_output=True).stdout
    return {int(row.split(",")[0]): int(row.split(",")[1]) for row in output.strip().splitlines()}


def rsync(source, target):
    command(["rsync", "-a", "--partial", "-e", shlex.join(SSH), str(source), target])


def stage_remote():
    if len(list((INPUT / "features").glob("*.npz"))) != 1000:
        raise RuntimeError("MR-RATE特征未完成，禁止启动六卡表2训练")
    remote_script("mkdir -p " + shlex.quote(REMOTE_ROOT + "/src") + " " +
                  shlex.quote(REMOTE_ROOT + "/outputs/mr_rate_1k/experiment") + " " +
                  shlex.quote(REMOTE_OUT))
    rsync(str(SRC) + "/", f"{REMOTE_HOST}:{REMOTE_ROOT}/src/")
    rsync(str(INPUT) + "/", f"{REMOTE_HOST}:{REMOTE_ROOT}/outputs/mr_rate_1k/experiment/")
    rsync(OUT / "protocol.json", f"{REMOTE_HOST}:{REMOTE_OUT}/protocol.json")
    if (OUT / "recovery_protocol.json").exists():
        rsync(OUT / "recovery_protocol.json", f"{REMOTE_HOST}:{REMOTE_OUT}/recovery_protocol.json")
    remote_script("cd " + shlex.quote(REMOTE_ROOT) + " && /home/Lim/conda/envs/myenv/bin/python "
                  "src/scripts/run_mrrate_1k_all_models.py --audit-only")
    save_json(OUT / "remote_stage.json", {"host": REMOTE_HOST, "remote_root": REMOTE_ROOT,
              "remote_output": REMOTE_OUT, "features": 1000, "staged_unix": time.time()})


def launch_local(gpu, model, fold, log_path):
    environment = os.environ.copy()
    environment.update({"CUDA_VISIBLE_DEVICES": str(gpu), "OMP_NUM_THREADS": "2",
                        "TOKENIZERS_PARALLELISM": "false", "MRRATE_MEMORY_FRACTION": "0.35"})
    log_path.parent.mkdir(parents=True, exist_ok=True)
    stream = log_path.open("a", encoding="utf-8")
    stream.write(f"\n[{time.strftime('%F %T')}] 本机GPU{gpu}启动{model}折{fold}\n")
    stream.flush()
    process = subprocess.Popen([sys.executable, "-u", str(SCRIPT), "--worker-index", str(fold-1),
                                "--devices", "0", "1", "2", "3", "4", "--models", model,
                                "--output-dir", str(OUT)],
                               cwd=ROOT, env=environment, stdout=stream, stderr=subprocess.STDOUT)
    stream.close()
    return process


def launch_remote(gpu, model, fold, log_path):
    remote_command = "cd {root} && exec env CUDA_VISIBLE_DEVICES={gpu} OMP_NUM_THREADS=2 " \
                     "TOKENIZERS_PARALLELISM=false MRRATE_MEMORY_FRACTION=0.35 " \
                     "/home/Lim/conda/envs/myenv/bin/python -u src/scripts/run_mrrate_1k_all_models.py " \
                     "--worker-index {worker} --devices 0 1 2 3 4 --models {model} --output-dir {out}".format(
                         root=shlex.quote(REMOTE_ROOT), gpu=gpu, worker=fold-1, model=shlex.quote(model),
                         out=shlex.quote(REMOTE_OUT))
    log_path.parent.mkdir(parents=True, exist_ok=True)
    stream = log_path.open("a", encoding="utf-8")
    stream.write(f"\n[{time.strftime('%F %T')}] 202主机GPU{gpu}启动{model}折{fold}\n")
    stream.flush()
    process = subprocess.Popen([*SSH, REMOTE_HOST, remote_command], stdout=stream, stderr=subprocess.STDOUT)
    stream.close()
    return process


def remote_completed():
    result = remote_script("find " + shlex.quote(REMOTE_OUT) + " -path '*/completed.json' -type f | wc -l",
                           capture_output=True)
    return int(result.stdout.strip())


def local_completed():
    return len(list(OUT.glob("fold_*/*/completed.json")))


def write_report():
    summary = json.loads((OUT / "summary.json").read_text())
    lines = ["# MR-RATE-1K 表2全模型五折结果", "", "- 样本：1000 名患者、4 个报告派生病理类别",
             "- 划分：固定患者五折，600/200/200；测试折不参与模型选择", "",
             "| 类别 | 模型 | 五折 Macro-F1 | 固定0.5 Macro-F1 | OOF Macro-F1 |", "|---|---|---:|---:|---:|"]
    for row in summary:
        lines.append(f"| {row.get('category', '')} | {row['model']} | {row['macro_f1_mean']:.4f} ± {row['macro_f1_std']:.4f} | "
                     f"{row['macro_f1_fixed_0_5_mean']:.4f} ± {row['macro_f1_fixed_0_5_std']:.4f} | {row['oof_macro_f1']:.4f} |")
    (OUT.parent / "table2_results.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main():
    import shutil
    import traceback
    import run_mrrate_1k_all_models as runner

    OUT.mkdir(parents=True, exist_ok=True)
    with (OUT / "gpu_pool.lock").open("a") as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as error:
            raise SystemExit("已有MR-RATE GPU池调度器") from error
        if shutil.disk_usage(OUT).free < 3 * 1024**3:
            raise RuntimeError("本机结果盘剩余不足3GiB，停止新增任务")
        remote_available = remote_script("df -Pk /home/Lim | tail -1", capture_output=True).stdout
        if int(remote_available.split()[3]) * 1024 < 3 * 1024**3:
            raise RuntimeError("202主机结果盘剩余不足3GiB，停止新增任务")
        stage_remote()
        stopping = False

        def stop(_signal, _frame):
            nonlocal stopping
            stopping = True

        signal.signal(signal.SIGINT, stop)
        signal.signal(signal.SIGTERM, stop)
        digest = json.loads((OUT / "protocol.json").read_text())["protocol_sha256"]
        compatible = runner.result_compatibility()
        jobs = [(model, fold) for model in runner.MODELS for fold in range(1, 6)]

        def done(job):
            model, fold = job
            folder = OUT / f"fold_{fold}" / model
            marker = folder / "completed.json"
            if not marker.exists():
                return False
            payload = json.loads(marker.read_text())
            if payload["protocol_sha256"] not in {digest, *compatible.get(model, [])}:
                raise ValueError(f"结果协议不在审核白名单内：{marker}")
            if not all((folder / name).exists() for name in ("test_metrics.json", "test_predictions.csv")):
                raise ValueError(f"完成标记对应的结果不完整：{folder}")
            return True

        pending = [job for job in jobs if not done(job)]
        active, attempts, failed, history = {}, {}, {}, []
        session_tag = time.strftime("%Y%m%d_%H%M%S")
        started = time.time()
        try:
            while (pending or active) and not stopping:
                for name, running in list(active.items()):
                    process = running["process"]
                    if process.poll() is None:
                        continue
                    model, fold = running["job"]
                    if running["host"] == "remote" and process.returncode == 0:
                        relative = f"fold_{fold}/{model}"
                        destination = OUT / relative
                        destination.mkdir(parents=True, exist_ok=True)
                        rsync(f"{REMOTE_HOST}:{REMOTE_OUT}/{relative}/", str(destination) + "/")
                    success = process.returncode == 0 and done(running["job"])
                    event = {key: value for key, value in running.items() if key != "process"}
                    event.update(exit_code=process.returncode, success=success, finished_unix=time.time())
                    history.append(event)
                    save_json(OUT / f"gpu_job_history_{session_tag}.json", history)
                    del active[name]
                    if not success:
                        if attempts[name] < MAX_RESTARTS:
                            pending.append(running["job"])
                        else:
                            failed[name] = running["log"]
                memories = {"local": local_free(), "remote": remote_free()}
                budgets = {host: dict(values) for host, values in memories.items()}
                for running in active.values():
                    if time.time() - running["started_unix"] < 30:
                        budgets[running["host"]][running["gpu"]] -= MINIMUM_FREE_MIB
                for job in list(pending):
                    candidates = [(free, host, gpu) for host, values in budgets.items() for gpu, free in values.items()
                                  if free >= MINIMUM_FREE_MIB]
                    if not candidates:
                        break
                    _, host, gpu = max(candidates)
                    model, fold = job
                    name = f"{model}_fold_{fold}"
                    attempts[name] = attempts.get(name, 0) + 1
                    log_path = OUT / "gpu_pool_logs" / f"{session_tag}_{name}_{host}_gpu{gpu}_attempt{attempts[name]}.log"
                    process = (launch_local(gpu, model, fold, log_path) if host == "local"
                               else launch_remote(gpu, model, fold, log_path))
                    active[name] = {"job": job, "host": host, "gpu": gpu, "process": process,
                                    "pid": process.pid, "started_unix": time.time(), "log": str(log_path)}
                    pending.remove(job)
                    budgets[host][gpu] -= MINIMUM_FREE_MIB
                    print(f"分配{name}到{host} GPU{gpu}，尝试{attempts[name]}", flush=True)
                completed = sum(done(job) for job in jobs)
                save_json(OUT / "gpu_pool_status.json", {"state": "running" if active else "waiting_for_memory",
                          "completed": completed, "total": len(jobs), "pending_jobs": len(pending),
                          "minimum_free_mib": MINIMUM_FREE_MIB, "max_per_gpu": None,
                          "memory": memories, "failed_jobs": failed,
                          "active": {name: {key: value for key, value in running.items() if key != "process"}
                                     for name, running in active.items()}, "updated_unix": time.time()})
                print(f"MR-RATE已完成{completed}/{len(jobs)}；排队{len(pending)}；失败{len(failed)}", flush=True)
                if pending or active:
                    time.sleep(POLL_SECONDS)
            command([sys.executable, str(SCRIPT), "--aggregate", "--output-dir", str(OUT)], cwd=ROOT)
            completed = sum(done(job) for job in jobs)
            status = "complete" if completed == len(jobs) else "incomplete"
            save_json(OUT / "gpu_pool_status.json", {"state": status, "completed": completed, "total": len(jobs),
                      "failed_jobs": failed, "wall_seconds": time.time()-started, "updated_unix": time.time()})
            write_report()
        except Exception:
            save_json(OUT / "gpu_pool_status.json", {"state": "failed", "traceback": traceback.format_exc(),
                      "active": {name: {key: value for key, value in running.items() if key != "process"}
                                 for name, running in active.items()}, "updated_unix": time.time()})
            raise
        finally:
            if stopping:
                for running in active.values():
                    if running["process"].poll() is None:
                        running["process"].terminate()


if __name__ == "__main__":
    main()
