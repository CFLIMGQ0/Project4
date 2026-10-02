"""每张卡一个 CUDA 进程，按批次交错推进多个独立实验。"""
from __future__ import annotations
import argparse
from contextlib import redirect_stdout, redirect_stderr
import fcntl
import gc
import json
import os
from pathlib import Path
import random
import signal
import sys
import time
import traceback
from types import SimpleNamespace

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from auto_research.train import run_job, write_json


class RandomState:
    """协作切换时保存所有训练采样源；不使用共享的交错随机序列。"""
    def __init__(self, cuda=True):
        self.python = random.getstate()
        self.numpy = np.random.get_state()
        self.torch = torch.get_rng_state()
        self.cuda = torch.cuda.get_rng_state() if cuda else None

    def restore(self):
        random.setstate(self.python)
        np.random.set_state(self.numpy)
        torch.set_rng_state(self.torch)
        if self.cuda is not None:
            torch.cuda.set_rng_state(self.cuda)


class Task:
    def __init__(self, request, folder, shared_cache):
        self.request = request
        self.ticket = request["ticket"]
        self.folder = folder
        Path(request["log"]).parent.mkdir(parents=True, exist_ok=True)
        self.log = Path(request["log"]).open("a", buffering=1)
        self.generator = run_job(SimpleNamespace(root=Path(request["root"]), job=Path(request["job"]),
                                 output=Path(request["output"]), execution_mode="single_process_per_gpu"),
                                 shared_cache=shared_cache)
        self.rng = None
        self.started = time.time()
        self.progress = dict(phase="starting")
        self.stream = torch.cuda.Stream()
        self.allocated_before = 0.
        self.footprint_mib = 0.

    def advance(self):
        if self.rng is not None:
            self.rng.restore()
        before = torch.cuda.memory_allocated() / 2**20
        with redirect_stdout(self.log), redirect_stderr(self.log), torch.cuda.stream(self.stream):
            try:
                self.progress = next(self.generator)
                # 任务切换只在完整 batch / 评估 batch 边界，不跨越 autograd 上下文。
                self.rng = RandomState()
                after = torch.cuda.memory_allocated() / 2**20
                self.footprint_mib += after - before
                return False
            except StopIteration:
                return True

    def finish(self, status, error=None):
        payload = dict(ticket=self.ticket, job_id=self.request["job_id"], status=status,
                       pid=os.getpid(), started=self.started, finished=time.time(), error=error)
        write_json(self.folder / "done" / (self.ticket + ".json"), payload)
        self.generator.close()
        self.log.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--queue-dir", type=Path, required=True)
    parser.add_argument("--host", choices=["202", "204"], required=True)
    parser.add_argument("--gpu", type=int, required=True)
    parser.add_argument("--max-resident", type=int, default=64)
    args = parser.parse_args()
    if args.gpu == 0:
        raise RuntimeError("用户已暂停 GPU0，拒绝启动训练 worker")
    folder = args.queue_dir
    folder.mkdir(parents=True, exist_ok=True)
    lock = (folder / "worker.lock").open("w")
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    lock.write(str(os.getpid())); lock.flush()
    for name in ("inbox", "claimed", "done"):
        (folder / name).mkdir(exist_ok=True)
    torch.set_num_threads(2)
    torch.set_num_interop_threads(1)
    torch.cuda.init()
    torch.backends.mha.set_fastpath_enabled(False)
    active, shared_cache = {}, {}
    finishing = False
    def request_finish(_sig, _frame):
        nonlocal finishing
        finishing = True
    signal.signal(signal.SIGTERM, request_finish)
    signal.signal(signal.SIGINT, request_finish)
    estimate, headroom = 768., 512.
    last_publish = 0.
    seen = {p.stem for p in (folder / "done").glob("*.json")}
    print(f"单进程 GPU worker 启动 host={args.host} GPU={args.gpu} PID={os.getpid()}", flush=True)
    while True:
        control = folder / "control.json"
        if control.exists():
            config = json.loads(control.read_text())
            finishing = finishing or config.get("stop_when_idle", False)
        free, total = torch.cuda.mem_get_info()
        recyclable = max(0, torch.cuda.memory_reserved()-torch.cuda.memory_allocated())
        available = (free+recyclable)/2**20
        loading = any(t.progress["phase"] in {"starting", "loading", "initialized"} for t in active.values())
        # 新任务完成第一个真实训练 batch 后才用其常驻显存估算继续装载，不做短训练。
        if not finishing and not loading and len(active) < args.max_resident and available > estimate+headroom:
            for path in sorted((folder / "inbox").glob("*.json"), key=lambda p:p.stat().st_mtime):
                if path.stem in seen or path.stem in active:
                    continue
                request = json.loads(path.read_text())
                if request["ticket"] != path.stem:
                    raise ValueError("任务票据与文件名不一致")
                result = Path(request["output"]) / "result.json"
                if result.exists():
                    write_json(folder/"done"/(path.stem+".json"), dict(ticket=path.stem,job_id=request["job_id"],status="completed",pid=os.getpid(),reused=True))
                    seen.add(path.stem)
                    continue
                claimed = folder / "claimed" / path.name
                path.replace(claimed)
                try:
                    task = Task(request, folder, shared_cache)
                    active[path.stem] = task
                    print("接收内部任务 " + request["job_id"], flush=True)
                except Exception:
                    write_json(folder/"done"/path.name, dict(ticket=path.stem,job_id=request["job_id"],
                               status="failed",pid=os.getpid(),error=traceback.format_exc()))
                    seen.add(path.stem)
                break
        for ticket, task in list(active.items()):
            try:
                complete = task.advance()
                if task.progress["phase"] == "training":
                    estimate = max(estimate, task.footprint_mib + 384.)
                if complete:
                    task.finish("completed")
                    seen.add(ticket)
                    del active[ticket]
                    del task
                    gc.collect()
                    print("内部任务完成 " + ticket, flush=True)
            except Exception as error:
                message = traceback.format_exc()
                task.log.write(message)
                task.finish("failed", message)
                seen.add(ticket)
                del active[ticket]
                del task
                gc.collect()
                torch.cuda.empty_cache()
                if isinstance(error, torch.OutOfMemoryError):
                    estimate = max(estimate*1.3, 1024.)
                    headroom = min(headroom+256, 2048.)
                print("内部任务失败 " + ticket + ": " + str(error), flush=True)
        if time.time()-last_publish >= 2:
            free,total = torch.cuda.mem_get_info()
            recyclable = max(0,torch.cuda.memory_reserved()-torch.cuda.memory_allocated())
            capacity = max(0, int(((free+recyclable)/2**20-headroom)/estimate))
            payload = dict(pid=os.getpid(),host=args.host,gpu=args.gpu,updated=time.time(),
                           mode="single_process_per_gpu",finishing=finishing,free_mib=free/2**20,
                           reserved_mib=torch.cuda.memory_reserved()/2**20,
                           allocated_mib=torch.cuda.memory_allocated()/2**20,
                           admission_estimate_mib=estimate,additional_capacity=capacity,
                           pending=len(list((folder/"inbox").glob("*.json"))),
                           tasks={ticket:dict(job_id=t.request["job_id"],started=t.started,**t.progress)
                                  for ticket,t in active.items()})
            write_json(folder/"worker_state.json",payload)
            last_publish=time.time()
        if finishing and not active:
            print("全部内部任务已结束，退出 worker。",flush=True)
            break
        if not active:
            time.sleep(.5)


if __name__ == "__main__":
    main()
