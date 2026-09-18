#!/usr/bin/env python3
"""动态运行AMOS-MM：本机四卡加远端两卡，每卡最多一个本实验任务。"""
from __future__ import annotations

import fcntl
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "src"
OUT = ROOT / "outputs/amos_mm/all_models"
POOL = ROOT / "outputs/amos_mm/compute_pool"
TEMP = ROOT / "temp.md"
RUNNER = SRC / "scripts/run_amos_mm_all_models.py"
REMOTE_POOL = SRC / "scripts/amos_mm_remote_pool.py"
LOCAL_GPUS = (0, 1, 2, 3)
REMOTE_GPUS = (0, 1)
MINIMUM_FREE_MIB = 12000
POLL_SECONDS = 15


def save_json(path, value):
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


def local_free_memory():
    output = subprocess.run(
        ["nvidia-smi", "--query-gpu=index,memory.free", "--format=csv,noheader,nounits"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout
    return {int(line.split(",")[0]): int(line.split(",")[1]) for line in output.strip().splitlines()}


def counts():
    completed = len(list(OUT.glob("*_labels/fold_*/*/result.json")))
    failed = len(list(OUT.glob("*_labels/fold_*/*/error.json")))
    return completed, failed


def launch(command, log_path, environment=None):
    log_path.parent.mkdir(parents=True, exist_ok=True)
    stream = log_path.open("a", encoding="utf-8")
    stream.write(f"\n[{time.strftime('%F %T')}] launch: {' '.join(map(str, command))}\n")
    stream.flush()
    process = subprocess.Popen(
        command,
        cwd=SRC,
        env=environment,
        stdout=stream,
        stderr=subprocess.STDOUT,
    )
    stream.close()
    return process


def write_final_results():
    summary = json.loads((OUT / "summary.json").read_text())
    assert summary["completed"] == summary["total"] == 200
    lines = [
        "# AMOS-MM 三标签与七标签实验结果",
        "",
        f"- 完成任务：{summary['completed']}/{summary['total']}",
        "- 评估：五次开发划分训练，固定 200 例人工标签测试集",
        "- 数据说明：纯文本使用 1,487 例开发报告；影像/多模态因官方 `amos_5964` CT 损坏使用 1,486 例开发数据",
        "",
        "| 标签数 | 类别 | 模型 | 完成折数 | Macro-F1 均值 | Macro-F1 标准差 | 共阳性 Macro-F1 |",
        "|---:|---|---|---:|---:|---:|---:|",
    ]
    for row in sorted(summary["models"], key=lambda item: (item["labels"], item["category"], item["model"])):
        standard_deviation = row["macro_f1_std"]
        lines.append(
            f"| {row['labels']} | {row['category']} | {row['model']} | {row['completed_runs']} | "
            f"{row['macro_f1_mean']:.4f} | {standard_deviation:.4f} | {row['copositive_macro_f1_mean']:.4f} |"
        )
    temporary = TEMP.with_suffix(".tmp.md")
    temporary.write_text("\n".join(lines) + "\n", encoding="utf-8")
    temporary.replace(TEMP)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    lock_path = OUT / "gpu_pool.lock"
    with lock_path.open("a") as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as error:
            raise SystemExit("已有AMOS-MM GPU池调度器正在运行") from error
        protocol = json.loads((OUT / "protocol.json").read_text())
        staged = json.loads((POOL / "staged.json").read_text())
        assert staged["protocol_sha256"] == protocol["protocol_sha256"], "远端代码快照不是当前协议，请先重新执行 --stage"
        completed, failed = counts()
        if completed < 200:
            TEMP.write_text("", encoding="utf-8")
        stopping = False

        def stop(_signal, _frame):
            nonlocal stopping
            stopping = True

        signal.signal(signal.SIGTERM, stop)
        signal.signal(signal.SIGINT, stop)
        local_processes = {}
        remote_processes = {}
        try:
            while not stopping:
                completed, failed = counts()
                if completed + failed >= 200:
                    break
                memory = local_free_memory()
                for gpu, process in list(local_processes.items()):
                    if process.poll() is not None:
                        del local_processes[gpu]
                for gpu, process in list(remote_processes.items()):
                    if process.poll() is not None:
                        del remote_processes[gpu]
                for gpu in LOCAL_GPUS:
                    if gpu in local_processes or memory.get(gpu, 0) < MINIMUM_FREE_MIB:
                        continue
                    environment = os.environ.copy()
                    environment.update({"CUDA_VISIBLE_DEVICES": str(gpu), "OMP_NUM_THREADS": "2", "OPENBLAS_NUM_THREADS": "2"})
                    local_processes[gpu] = launch(
                        [sys.executable, "-u", str(RUNNER.relative_to(SRC)), "--worker", str(gpu)],
                        OUT / "gpu_pool_logs" / f"local_gpu_{gpu}.log",
                        environment,
                    )
                for gpu in REMOTE_GPUS:
                    if gpu in remote_processes:
                        continue
                    worker = 4 + gpu
                    remote_processes[gpu] = launch(
                        [sys.executable, "-u", str(REMOTE_POOL.relative_to(SRC)), "--dispatch", str(gpu), "--worker", str(worker)],
                        OUT / "gpu_pool_logs" / f"remote_gpu_{gpu}.log",
                    )
                save_json(OUT / "gpu_pool_status.json", {
                    "state": "running",
                    "completed": completed,
                    "failed": failed,
                    "minimum_free_mib": MINIMUM_FREE_MIB,
                    "local": {str(gpu): {"free_mib": memory.get(gpu), "pid": local_processes[gpu].pid if gpu in local_processes else None}
                              for gpu in LOCAL_GPUS},
                    "remote": {str(gpu): {"pid": remote_processes[gpu].pid if gpu in remote_processes else None}
                               for gpu in REMOTE_GPUS},
                    "updated_at": time.time(),
                })
                time.sleep(POLL_SECONDS)
        finally:
            for process in [*local_processes.values(), *remote_processes.values()]:
                if process.poll() is None:
                    process.terminate()
            for process in [*local_processes.values(), *remote_processes.values()]:
                try:
                    process.wait(timeout=10)
                except subprocess.TimeoutExpired:
                    process.kill()
            completed, failed = counts()
            state = "complete" if completed == 200 and failed == 0 else "finished_with_errors" if completed + failed >= 200 else "stopped"
            save_json(OUT / "gpu_pool_status.json", {"state": state, "completed": completed, "failed": failed,
                      "minimum_free_mib": MINIMUM_FREE_MIB, "updated_at": time.time()})
            if state == "complete":
                subprocess.run([sys.executable, str(RUNNER), "--aggregate"], cwd=SRC, check=True)
                write_final_results()


if __name__ == "__main__":
    main()
