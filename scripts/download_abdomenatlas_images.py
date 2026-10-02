#!/usr/bin/env python3
"""并发、可续传地下载 AbdomenAtlas 3.0 CT 影像包，并逐包校验 SHA-256。"""

import argparse
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
import fcntl
import hashlib
import json
import os
from pathlib import Path
import shutil
import signal
import subprocess
import threading
import time


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=4)
    args = parser.parse_args()
    args.data_dir.mkdir(parents=True, exist_ok=True)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    lock_file = (args.data_dir / ".download.lock").open("w")
    try:
        fcntl.flock(lock_file, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        raise SystemExit("同一目录已有下载器运行，退出以避免重复写入。")
    lock_file.write(str(os.getpid()))
    lock_file.flush()
    items = json.loads(args.manifest.read_text())
    total = sum(item["size"] for item in items)
    stop = threading.Event()
    mutex = threading.Lock()
    states = {item["name"]: {"status": "queued", "attempt": 0} for item in items}
    processes = {}
    last_progress = [time.monotonic(), 0]
    started = datetime.now().isoformat(timespec="seconds")
    verification_file = args.output_dir / "verified.json"
    verified = json.loads(verification_file.read_text()) if verification_file.exists() else {}

    def log(message):
        print(f"[{datetime.now():%Y-%m-%d %H:%M:%S}] {message}", flush=True)

    def atomic_json(path, value):
        tmp = path.with_suffix(path.suffix + ".tmp")
        tmp.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n")
        tmp.replace(path)

    def update(name, **kwargs):
        with mutex:
            states[name].update(kwargs)

    def fingerprint(path):
        st = path.stat()
        return {"size": st.st_size, "mtime_ns": st.st_mtime_ns}

    def checksum(path):
        digest = hashlib.sha256()
        with path.open("rb") as handle:
            for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
                if stop.is_set():
                    return None
                digest.update(chunk)
        return digest.hexdigest()

    def check_file(path, item):
        name = item["name"]
        if not path.exists() or path.stat().st_size != item["size"]:
            return False
        with mutex:
            saved = verified.get(name)
        if path.name == name and saved == {**fingerprint(path), "sha256": item["sha256"]}:
            return True
        update(name, status="verifying")
        actual = checksum(path)
        if actual is None:
            return False
        if actual != item["sha256"]:
            backup = path.with_name(path.name + f".sha256_mismatch_{int(time.time())}")
            path.rename(backup)
            log(f"校验不匹配，文件保留为 {backup.name}，重新下载 {name}")
            return False
        if path.name != name:
            path.replace(args.data_dir / name)
            path = args.data_dir / name
        with mutex:
            verified[name] = {**fingerprint(path), "sha256": actual}
            atomic_json(verification_file, verified)
        return True

    def worker(item):
        name = item["name"]
        final = args.data_dir / name
        part = args.data_dir / (name + ".part")
        if final.exists():
            if check_file(final, item):
                update(name, status="complete")
                log(f"复用已校验文件：{name}")
                return
            if final.exists() and final.stat().st_size < item["size"] and not part.exists():
                final.rename(part)
            elif final.exists():
                raise RuntimeError(f"已有目标文件大小不符，保留原文件并停止：{name}")
        for attempt in range(1, 21):
            if stop.is_set():
                update(name, status="paused")
                return
            if check_file(part, item):
                update(name, status="complete")
                log(f"下载并校验完成：{name}")
                return
            if part.exists() and part.stat().st_size > item["size"]:
                raise RuntimeError(f"部分文件超出预期大小，保留文件并停止：{name}")
            if shutil.disk_usage(args.data_dir).free < 80 * 1024**3:
                raise RuntimeError("可用空间低于 80 GiB，停止下载并保留断点。")
            update(name, status="downloading", attempt=attempt)
            log(f"开始/续传：{name}（第 {attempt} 次）")
            command = [
                "curl", "--noproxy", "*", "--location", "--fail", "--silent", "--show-error",
                "--connect-timeout", "20", "--speed-time", "120", "--speed-limit", "1024",
                "--continue-at", "-", "--output", str(part), item["url"],
            ]
            with (args.output_dir / (name + ".curl.log")).open("a") as error_log:
                process = subprocess.Popen(command, stdout=error_log, stderr=error_log)
                with mutex:
                    processes[name] = process
                code = process.wait()
                with mutex:
                    processes.pop(name, None)
            if stop.is_set():
                update(name, status="paused")
                return
            if code == 0 and check_file(part, item):
                update(name, status="complete")
                log(f"下载并校验完成：{name}")
                return
            update(name, status="retry_wait", last_exit_code=code)
            log(f"下载未完成，保留断点后重试：{name}，curl={code}")
            stop.wait(min(60, attempt * 5))
        raise RuntimeError(f"连续 20 次未完成：{name}")

    def progress():
        present = 0
        lengths = {}
        for item in items:
            final = args.data_dir / item["name"]
            part = args.data_dir / (item["name"] + ".part")
            try:
                size = final.stat().st_size if final.exists() else part.stat().st_size
            except FileNotFoundError:
                size = 0
            lengths[item["name"]] = min(size, item["size"])
            present += lengths[item["name"]]
        now = time.monotonic()
        rate = max(0, present - last_progress[1]) / max(now - last_progress[0], 0.001)
        if last_progress[1] == 0:
            rate = 0
        last_progress[:] = now, present
        with mutex:
            records = {name: {**state, "downloaded_bytes": lengths[name]} for name, state in states.items()}
        done = sum(s["status"] == "complete" for s in records.values())
        active = sum(s["status"] == "downloading" for s in records.values())
        free = shutil.disk_usage(args.data_dir).free
        atomic_json(args.output_dir / "status.json", {
            "pid": os.getpid(), "started": started, "updated": datetime.now().isoformat(timespec="seconds"),
            "total_bytes": total, "downloaded_bytes": present, "complete_files": done,
            "total_files": len(items), "active_downloads": active, "rate_bytes_per_second": rate,
            "remaining_seconds_estimate": (total - present) / rate if rate > 0 else None,
            "disk_free_bytes": free, "files": records,
        })
        cells = int(present / total * 24)
        bar = "#" * cells + "-" * (24 - cells)
        log(f"进度 [{bar}] {present / 1e9:.2f}/{total / 1e9:.2f} GB；"
            f"已校验 {done}/{len(items)} 包；正在传输 {active} 包；近段速度 {rate / 1e6:.2f} MB/s")
        if free < 80 * 1024**3:
            log("可用空间低于 80 GiB，停止下载并保留断点。")
            stop.set()

    def request_stop(signum, _frame):
        log(f"收到信号 {signum}，停止下载并保留断点。")
        stop.set()

    signal.signal(signal.SIGTERM, request_stop)
    signal.signal(signal.SIGINT, request_stop)
    log(f"使用镜像直连，{args.workers} 路并发，共 {len(items)} 包。")
    error = False
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = [pool.submit(worker, item) for item in items]
        while not all(f.done() for f in futures):
            progress()
            if stop.is_set():
                with mutex:
                    active = list(processes.values())
                for process in active:
                    if process.poll() is None:
                        process.terminate()
                for f in futures:
                    f.cancel()
            time.sleep(5)
        for future in futures:
            if future.cancelled():
                continue
            try:
                future.result()
            except Exception as exc:
                error = True
                log(f"任务失败：{exc}")
    progress()
    log("全部影像包下载并校验完成。" if all(s["status"] == "complete" for s in states.values())
        else "本轮下载结束，未完成的文件可在下次运行时续传。")
    raise SystemExit(1 if error else 0)


if __name__ == "__main__":
    main()
