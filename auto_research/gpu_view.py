#!/usr/bin/env python3
"""按实验组合并显示 GPU 任务；保留真实显存、进程数和可展开的任务明细。"""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
import csv
from datetime import datetime
import io
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import unicodedata

EXPERIMENT = "acpe_research_200_20260928"
SSH = ["ssh", "-i", "/home/Lim/.ssh/id_ed25519_project4_pool", "-o", "BatchMode=yes",
       "-o", "ConnectTimeout=5", "Lim@172.16.170.202"]
PYTHON = "/home/Lim/conda/envs/myenv/bin/python"


def query(fields, kind):
    result = subprocess.run(["nvidia-smi", f"--query-{kind}={fields}",
                             "--format=csv,noheader,nounits"], capture_output=True, text=True,
                            check=True, timeout=8)
    return [[value.strip() for value in row] for row in csv.reader(io.StringIO(result.stdout)) if row]


def numeric(value):
    try:
        return int(value)
    except ValueError:
        return None


def collect(root):
    GPUs = query("index,uuid,memory.total,memory.used,utilization.gpu", "gpu")
    processes = query("gpu_uuid,pid,used_gpu_memory", "compute-apps")
    metadata = {}
    for _, pid, _ in processes:
        if pid in metadata:
            continue
        proc = Path("/proc") / pid
        item = dict(pid=int(pid), research=False)
        try:
            args = (proc / "cmdline").read_bytes().decode(errors="replace").split("\0")
            worker = any(arg.endswith(f"/{EXPERIMENT}/workspace/src/auto_research/gpu_worker.py") for arg in args)
            item["research"] = worker or any(arg.endswith(f"/{EXPERIMENT}/workspace/src/auto_research/train.py") for arg in args)
            if worker:
                folder = Path(args[args.index("--queue-dir") + 1])
                try:
                    state = json.loads((folder / "worker_state.json").read_text())
                    item["tasks"] = list(state["tasks"].values())
                    item["queued"] = state["pending"]
                except (FileNotFoundError, json.JSONDecodeError):
                    item["tasks"] = []
                item["mode"] = "single_process_per_gpu"
            elif item["research"]:
                folder = Path(args[args.index("--output") + 1])
                item["job"] = folder.name
                try:
                    history = json.loads((folder / "history.json").read_text())
                    item["epoch"] = history[-1]["epoch"] if history else 0
                except (FileNotFoundError, json.JSONDecodeError):
                    item["epoch"] = 0
                item["tasks"] = [dict(job_id=item["job"], epoch=item["epoch"])]
        except (FileNotFoundError, PermissionError, ValueError):
            item["identity_unavailable"] = True
        metadata[pid] = item
    result = []
    for index, uuid, total, used, utilization in GPUs:
        assigned = []
        for gpu_uuid, pid, memory in processes:
            if gpu_uuid == uuid:
                assigned.append(dict(metadata[pid], memory_mib=numeric(memory)))
        ours = [p for p in assigned if p["research"]]
        # 不把看不到身份或显存的进程误认成本项目，也不隐去它们占用的全卡显存。
        grouped_memory = sum(p["memory_mib"] or 0 for p in ours)
        result.append(dict(index=int(index), uuid=uuid, total_mib=numeric(total),
                           used_mib=numeric(used), utilization=numeric(utilization),
                           group_memory_mib=grouped_memory,
                           group_memory_complete=all(p["memory_mib"] is not None for p in ours),
                           research_jobs=ours, other_compute_processes=len(assigned)-len(ours)))
    return dict(gpus=result, sampled_at=time.time())


def remote_collect():
    script = Path(__file__).read_text()
    result = subprocess.run(SSH + [f"{PYTHON} - --probe --root /home/Lim/Project4"],
                            input=script, capture_output=True, text=True, timeout=15, check=True)
    return json.loads(result.stdout)


def snapshot(root, hosts):
    results = {}
    with ThreadPoolExecutor(max_workers=2) as pool:
        futures = {host: pool.submit(collect, root) if host == "204" else pool.submit(remote_collect)
                   for host in hosts}
        for host, future in futures.items():
            try:
                results[host] = future.result()
            except (subprocess.SubprocessError, json.JSONDecodeError, OSError) as error:
                results[host] = dict(error=str(error))
    return results


def show(root, data, details):
    output = root / "outputs" / EXPERIMENT
    policy = {}
    try:
        policy = json.loads((output / "resource_policy.json").read_text()).get("excluded_gpus", {})
    except (FileNotFoundError, json.JSONDecodeError):
        pass
    print("ALM-MIL Auto Research  |  " + datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
    try:
        state = json.loads((output / "state.json").read_text())
        completed = sum(r["status"] == "completed" for r in state["rounds"])
        mode = "每卡一个训练进程，内部调度多个实验" if state.get("execution_mode") == "single_process_per_gpu" else "每卡合并显示本轮实验"
        print(f"已完成方案 {completed}/{state['target_rounds']}；{mode}。")
    except (FileNotFoundError, json.JSONDecodeError):
        pass
    widths = (5, 4, 8, 19, 43, 12)
    def table_row(values):
        cells = []
        for value, width in zip(values, widths):
            text = str(value)
            length = sum(2 if unicodedata.east_asian_width(char) in "WF" else 1 for char in text)
            cells.append(text + " " * max(1, width - length))
        return "".join(cells).rstrip()
    print(table_row(("主机", "GPU", "利用率", "全卡显存", "本轮实验组", "其他计算进程")))
    for host, result in data.items():
        if "error" in result:
            print(f"{host}: 读取失败，{result['error']}")
            continue
        for gpu in result["gpus"]:
            processes = len(gpu["research_jobs"])
            count = sum(len(p.get("tasks", [])) for p in gpu["research_jobs"])
            group = (f"ALM-MIL [{processes}进程/{count}任务, {gpu['group_memory_mib']/1024:.1f}GiB]" if processes
                     else "—")
            if not gpu["group_memory_complete"]:
                group += "（部分显存未知）"
            if gpu["index"] in policy.get(host, []):
                group += "（实验禁用）"
            percent = f"{gpu['utilization']}%" if gpu["utilization"] is not None else "未知"
            memory = f"{gpu['used_mib']/1024:4.1f}/{gpu['total_mib']/1024:.1f}GiB"
            print(table_row((host, gpu["index"], percent, memory, group, gpu["other_compute_processes"])))
            if details:
                for process in gpu["research_jobs"]:
                    print(f"    PID {process['pid']} | 进程显存 {process['memory_mib']}MiB")
                    for task in process.get("tasks", []):
                        print(f"        epoch {str(task.get('epoch', '—')):>2}/30 | {task.get('phase', 'training')} | {task['job_id']}")
    print("利用率和全卡显存包含其他程序；任务数为进程内部已装载的实验数。--details 可展开任务。")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path)
    parser.add_argument("--hosts", nargs="+", choices=["204", "202"], default=["204", "202"])
    parser.add_argument("-i", "--interval", nargs="?", type=float, const=2., default=0., help="自动刷新，默认2秒")
    parser.add_argument("--details", action="store_true", help="展开内部任务 PID 和进度")
    parser.add_argument("--json", action="store_true")
    parser.add_argument("--probe", action="store_true", help=argparse.SUPPRESS)
    args = parser.parse_args()
    root = args.root or Path(__file__).resolve().parents[2]
    if args.probe:
        print(json.dumps(collect(root)))
        return
    if args.interval < 0:
        parser.error("刷新间隔不能为负数")
    try:
        while True:
            started = time.monotonic()
            data = snapshot(root, args.hosts)
            if args.json:
                print(json.dumps(data, ensure_ascii=False), flush=True)
            else:
                if args.interval and sys.stdout.isatty():
                    print("\033[2J\033[H", end="")
                show(root, data, args.details)
            if not args.interval:
                return
            time.sleep(max(.1, args.interval - (time.monotonic()-started)))
    except KeyboardInterrupt:
        return


if __name__ == "__main__":
    main()
