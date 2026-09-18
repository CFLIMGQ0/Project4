#!/usr/bin/env python3
"""将同一版本Qwen权重补齐到内存缓存；只下载缺少的字节区间。"""

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import os
from pathlib import Path
import shutil
import time
import urllib.request


REVISION = "0c351dd01ed87e9c1b53cbc748cba10e6187ff3b"


def fetch(task):
    name, start, end, total, descriptor = task
    url = f"https://huggingface.co/Qwen/Qwen3-VL-8B-Instruct/resolve/{REVISION}/{name}?download=true&cq500range={start}"
    for attempt in range(4):
        try:
            request = urllib.request.Request(url, headers={"Range": f"bytes={start}-{end}"})
            with urllib.request.urlopen(request, timeout=60) as response:
                expected = f"bytes {start}-{end}/{total}"
                if response.status != 206 or response.headers.get("Content-Range") != expected:
                    raise RuntimeError(f"下载范围不一致：{name}，{start}-{end}")
                offset = start
                while block := response.read(1024 * 1024):
                    if offset + len(block) > end + 1:
                        raise RuntimeError("服务器返回内容超出范围")
                    view = memoryview(block)
                    while view:
                        written = os.pwrite(descriptor, view, offset)
                        if written <= 0:
                            raise OSError("写入权重缓存失败")
                        offset += written
                        view = view[written:]
                if offset != end + 1:
                    raise RuntimeError("下载内容被截断")
            return end - start + 1
        except Exception:
            if attempt == 3:
                raise
            time.sleep(attempt + 1)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--destination", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=16)
    args = parser.parse_args()
    args.destination.mkdir(parents=True, exist_ok=True)
    files, jobs = [], []
    chunk = 32 * 1024 * 1024
    try:
        for source in sorted(args.source.iterdir()):
            if not source.is_file():
                continue
            destination = args.destination / source.name
            total = source.stat().st_size
            if destination.exists() and destination.stat().st_size == total:
                continue
            if source.suffix != ".safetensors":
                shutil.copy2(source, destination)
                continue
            partial = destination.with_suffix(destination.suffix + ".partial")
            if partial.exists():
                raise RuntimeError(f"检测到旧的分段下载文件，请核对后另选缓存目录：{partial}")
            offset = destination.stat().st_size if destination.exists() else 0
            if offset > total:
                raise ValueError("已有缓存大于原始文件")
            if destination.exists():
                destination.rename(partial)
            descriptor = os.open(partial, os.O_CREAT | os.O_RDWR, 0o600)
            files.append((descriptor, partial, destination, total))
            # 临时文件预分配不会被生成进程误认为已完成的正式权重。
            os.ftruncate(descriptor, total)
            jobs.extend((source.name, start, min(total - 1, start + chunk - 1), total, descriptor)
                        for start in range(offset, total, chunk))
        print(f"需要补齐{sum(j[2]-j[1]+1 for j in jobs)/1024**3:.2f}GiB，共{len(jobs)}段", flush=True)
        completed = 0
        started = time.monotonic()
        with ThreadPoolExecutor(max_workers=args.workers) as pool:
            futures = [pool.submit(fetch, job) for job in jobs]
            for index, future in enumerate(as_completed(futures), 1):
                completed += future.result()
                if index % 8 == 0 or index == len(jobs):
                    print(f"补齐权重 {index}/{len(jobs)}段，{completed/1024**3:.2f}GiB，"
                          f"平均{completed/1024**2/max(1,time.monotonic()-started):.1f}MiB/s", flush=True)
        for descriptor, partial, destination, total in files:
            os.fsync(descriptor)
            assert partial.stat().st_size == total
            partial.replace(destination)
        print("同版本权重内存缓存补齐完成", flush=True)
    finally:
        for descriptor, *_ in files:
            os.close(descriptor)


if __name__ == "__main__":
    main()
