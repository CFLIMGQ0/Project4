#!/usr/bin/env python3
"""204主机 CT-RATE 完整测试集位置特征缓存的满载GPU任务池。"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import time


ROOT = Path(__file__).resolve().parents[2]
BUILDER = ROOT / "src" / "scripts" / "build_ctrate_position_feature_cache.py"
SPLIT = ROOT / "outputs" / "ct_rate_680" / "amef_apro_absolute_only_fivefold" / "fold_4" / "amef_multimodal" / "split_ids.json"
CACHE = ROOT / "outputs" / "ct_rate_680" / "position_recovery_apro_path_full136_seed42" / "raw_feature_cache"
QUEUE = ROOT / "outputs" / "ct_rate_680" / "apro_path_full136_cache_queue"
LOG_ROOT = ROOT / "outputs" / "ct_rate_680" / "apro_path_full136_cache_logs"
GROUP_COUNT = 24
RESERVED_JOB_MIB = 2000
HEADROOM_MIB = 500


def save_json(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


def gpu_free() -> dict[int, int]:
    result = subprocess.run(
        ["nvidia-smi", "--query-gpu=index,memory.free", "--format=csv,noheader,nounits"],
        check=True, capture_output=True, text=True,
    )
    return {int(index.strip()): int(value.strip()) for index, value in (
        line.split(",", 1) for line in result.stdout.splitlines() if line.strip()
    )}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reserved-job-mib", type=int, default=RESERVED_JOB_MIB)
    parser.add_argument("--poll-seconds", type=int, default=5)
    args = parser.parse_args()
    test_ids = [int(value) for value in json.loads(SPLIT.read_text())["test"]]
    groups = [test_ids[index::GROUP_COUNT] for index in range(GROUP_COUNT)]
    queue_files = []
    for index, group in enumerate(groups):
        path = QUEUE / f"group_{index:02d}.json"
        save_json(path, {"case_indices": group, "group": index, "total_test_cases": len(test_ids)})
        queue_files.append(path)
    CACHE.mkdir(parents=True, exist_ok=True)
    pending = list(range(GROUP_COUNT))
    active: dict[str, dict] = {}
    retries = {index: 0 for index in pending}
    started = time.time()
    while pending or active:
        free = gpu_free()
        for key, payload in list(active.items()):
            process = payload["process"]
            if process.poll() is None:
                continue
            del active[key]
            group = payload["group"]
            if process.returncode != 0:
                retries[group] += 1
                if retries[group] <= 2:
                    pending.append(group)
                    print(f"缓存任务失败重排：group_{group:02d}", flush=True)
                else:
                    raise RuntimeError(f"缓存任务连续失败：group_{group:02d}")
        reservations: dict[int, int] = {}
        for payload in active.values():
            reservations[payload["gpu"]] = reservations.get(payload["gpu"], 0) + 1
        candidates = []
        for gpu, memory in free.items():
            slots = reservations.get(gpu, 0)
            while memory >= (slots + 1) * args.reserved_job_mib + HEADROOM_MIB:
                candidates.append((memory - slots * args.reserved_job_mib, gpu))
                slots += 1
        candidates.sort(reverse=True)
        while pending and candidates:
            _, gpu = candidates.pop(0)
            group = pending.pop(0)
            log_path = LOG_ROOT / f"gpu_{gpu}_group_{group:02d}.log"
            log_path.parent.mkdir(parents=True, exist_ok=True)
            stream = log_path.open("a", encoding="utf-8")
            stream.write(f"\n[{time.strftime('%F %T')}] GPU{gpu}启动group_{group:02d}\n")
            stream.flush()
            environment = os.environ.copy()
            environment.update({"CUDA_VISIBLE_DEVICES": str(gpu), "OMP_NUM_THREADS": "2"})
            command = [sys.executable, "-u", str(BUILDER), "--case-indices-file", str(queue_files[group]),
                       "--cache-dir", str(CACHE), "--device", "cuda:0"]
            process = subprocess.Popen(command, cwd=ROOT, env=environment,
                                       stdout=stream, stderr=subprocess.STDOUT)
            stream.close()
            active[f"group_{group:02d}:gpu_{gpu}:{process.pid}"] = {
                "group": group, "gpu": gpu, "pid": process.pid,
                "process": process, "started_unix": time.time(),
            }
        save_json(ROOT / "outputs" / "ct_rate_680" / "apro_path_full136_cache_status.json", {
            "state": "running", "pending": pending,
            "active": {key: {k: v for k, v in value.items() if k != "process"}
                       for key, value in active.items()},
            "free_memory_mib": free, "reserved_job_mib": args.reserved_job_mib,
            "started_unix": started,
        })
        print(f"缓存任务活动{len(active)}个、排队{len(pending)}个；显存{free}", flush=True)
        if pending or active:
            time.sleep(args.poll_seconds)
    save_json(ROOT / "outputs" / "ct_rate_680" / "apro_path_full136_cache_status.json", {
        "state": "complete", "case_count": len(test_ids), "cache_files": len(list(CACHE.glob("*.npz"))),
        "finished_unix": time.time(), "wall_seconds": time.time() - started,
    })


if __name__ == "__main__":
    main()
