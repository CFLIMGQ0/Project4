#!/usr/bin/env python3
"""AMOS-MM无人值守监控：校验结果、重试失败、恢复调度并在低磁盘时远端转存。"""
from __future__ import annotations

import fcntl
import hashlib
import json
import math
import os
from pathlib import Path
import shlex
import shutil
import signal
import subprocess
import sys
import time
import traceback

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "outputs/amos_mm/all_models"
STATUS = OUT / "unattended_monitor_status.json"
LOG = OUT / "unattended_monitor.log"
HOST = "Lim@172.16.170.202"
SSH = ["ssh", "-i", "/home/Lim/.ssh/id_ed25519_project4_pool", "-o", "IdentitiesOnly=yes",
       "-o", "BatchMode=yes", "-o", "ConnectTimeout=8", "-o", "ServerAliveInterval=15",
       "-o", "ServerAliveCountMax=3"]
REMOTE_SPILLS = ("/home/Lim/AMOS_MM_checkpoint_spill_20260916",
                 "/new_data/Lim/AMOS_MM_checkpoint_spill_20260916")
TEXT_MODELS = {"hashed_mean_encoder", "vocab_attention_encoder", "textcnn_encoder",
               "bigru_encoder", "transformer_encoder"}
POLL_SECONDS = 60
MAX_ATTEMPTS = 3
LOCAL_STALE_SECONDS = 30 * 60
REMOTE_STALE_SECONDS = 6 * 60 * 60
PROJECT_SPILL_THRESHOLD = 30 * 1024**3
PROJECT_SPILL_TARGET = 40 * 1024**3
REMOTE_RESERVE = 15 * 1024**3


def save_json(path, value):
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


def log(message):
    line = f"[{time.strftime('%F %T')}] {message}"
    print(line, flush=True)
    with LOG.open("a", encoding="utf-8") as stream:
        stream.write(line + "\n")


def sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def run(args, **kwargs):
    return subprocess.run(args, check=True, text=True, **kwargs)


def ssh(script, capture_output=False):
    return run([*SSH, HOST, script], capture_output=capture_output)


def result_and_error_paths():
    results = sorted(OUT.glob("*_labels/fold_*/*/result.json"))
    errors = sorted(OUT.glob("*_labels/fold_*/*/error.json"))
    return results, errors


def expected_job(path):
    return {
        "labels": int(path.parents[2].name.removesuffix("_labels")),
        "fold": int(path.parents[1].name.removeprefix("fold_")) - 1,
        "model": path.parent.name,
    }


def validate_result(path, protocol_digest):
    result = json.loads(path.read_text())
    config = json.loads(path.with_name("config.json").read_text())
    job = expected_job(path)
    assert result["job"] == config["job"] == job
    assert result["protocol_sha256"] == config["protocol_sha256"] == protocol_digest
    assert result["test_n"] == len(config["test"]) == 200
    assert all(math.isfinite(float(result[name])) for name in
               ("macro_f1", "micro_f1", "macro_f1_at_0.5", "macro_auroc", "exact_match"))
    if job["model"] not in TEXT_MODELS:
        assert result["image_exclusions_applied"] == config["image_exclusions_applied"] == ["amos_5964"]
        assert 781 not in config["train"] and 781 not in config["validation"] and 781 not in config["test"]
    with np.load(path.with_name("test_predictions.npz"), allow_pickle=False) as predictions:
        assert predictions["labels"].shape == predictions["probabilities"].shape == (200, job["labels"])
        assert predictions["thresholds"].shape == (job["labels"],)
        assert len(predictions["scan_ids"]) == 200
        assert np.isfinite(predictions["probabilities"]).all()
    checkpoint = path.with_name("best_model.pt")
    pointer = path.with_name("best_model.pt.remote.json")
    assert checkpoint.is_file() or pointer.is_file()


def quarantine_invalid_result(path, error):
    with path.with_name("run.lock").open("a") as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            return False
        if not path.exists():
            return False
        destination = path.parent / "invalid_results"
        destination.mkdir(exist_ok=True)
        stamp = time.strftime("%Y%m%d_%H%M%S")
        path.replace(destination / f"result_{stamp}.json")
        save_json(path.with_name("error.json"), {"job": expected_job(path), "time": time.time(),
                  "validation_error": str(error), "traceback": traceback.format_exc()})
        log(f"结果校验失败并进入自动重试：{path.parent}: {error}")
        return True


def validate_results(protocol_digest):
    invalid = []
    for path in result_and_error_paths()[0]:
        try:
            validate_result(path, protocol_digest)
        except Exception as error:
            invalid.append(str(path))
            quarantine_invalid_result(path, error)
    return invalid


def retry_errors():
    retried = []
    final = []
    for path in result_and_error_paths()[1]:
        with path.with_name("run.lock").open("a") as lock:
            try:
                fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError:
                continue
            if not path.exists():
                continue
            attempts = path.parent / "attempt_errors"
            attempts.mkdir(exist_ok=True)
            previous = sorted(attempts.glob("attempt_*.json"))
            attempt_number = len(previous) + 1
            if attempt_number >= MAX_ATTEMPTS:
                final.append(str(path.parent))
                continue
            archived = attempts / f"attempt_{attempt_number}_{time.strftime('%Y%m%d_%H%M%S')}.json"
            path.replace(archived)
            save_json(path.parent / "retry_status.json", {"next_attempt": attempt_number + 1,
                      "maximum_attempts": MAX_ATTEMPTS, "last_error": str(archived), "updated_at": time.time()})
            retried.append(str(path.parent))
            log(f"任务失败，已安排第{attempt_number + 1}/{MAX_ATTEMPTS}次尝试：{path.parent}")
    return retried, final


def disk_snapshot():
    local = {}
    for name, path in (("local_root", Path("/")), ("local_project", Path("/xmlg"))):
        usage = shutil.disk_usage(path)
        local[name] = {"path": str(path), "total": usage.total, "used": usage.used, "free": usage.free}
    code = (
        "import json,shutil; "
        "print(json.dumps({p: dict(zip(('total','used','free'), shutil.disk_usage(p))) "
        "for p in ('/home/Lim','/new_data')}))"
    )
    remote = json.loads(ssh("python3 -c " + shlex.quote(code), capture_output=True).stdout)
    return {"local": local, "remote": remote}


def rsync_to_remote(source, target):
    run(["rsync", "-a", "--partial", "-e", shlex.join(SSH), str(source), f"{HOST}:{target}"])


def spill_checkpoints(disks):
    project_free = disks["local"]["local_project"]["free"]
    if project_free >= PROJECT_SPILL_THRESHOLD:
        return []
    candidates = [(path, disks["remote"]["/home/Lim" if path.startswith("/home") else "/new_data"]["free"])
                  for path in REMOTE_SPILLS]
    candidates = [item for item in candidates if item[1] > REMOTE_RESERVE]
    assert candidates, "项目盘空间不足，且两块远端磁盘都低于安全余量"
    moved = []
    checkpoints = sorted(path for path in OUT.glob("*_labels/fold_*/*/best_model.pt")
                         if path.is_file() and path.with_name("result.json").exists())
    for index, checkpoint in enumerate(checkpoints):
        if shutil.disk_usage("/xmlg").free >= PROJECT_SPILL_TARGET:
            break
        root = sorted(candidates, key=lambda item: item[1], reverse=True)[index % len(candidates)][0]
        relative = checkpoint.relative_to(OUT)
        target = f"{root}/{relative}"
        temporary = target + f".tmp.{os.getpid()}"
        ssh("mkdir -p " + shlex.quote(str(Path(target).parent)))
        rsync_to_remote(checkpoint, temporary)
        local_digest = sha256(checkpoint)
        remote_digest = ssh("sha256sum " + shlex.quote(temporary), capture_output=True).stdout.split()[0]
        assert local_digest == remote_digest
        ssh("mv " + shlex.quote(temporary) + " " + shlex.quote(target))
        save_json(checkpoint.with_name("best_model.pt.remote.json"), {"host": HOST, "path": target,
                  "sha256": local_digest, "bytes": checkpoint.stat().st_size, "moved_at": time.time()})
        checkpoint.unlink()
        moved.append(str(relative))
        log(f"项目盘低空间，checkpoint已校验后转存：{relative} -> {HOST}:{target}")
    return moved


def process_exists(pattern):
    result = subprocess.run(["pgrep", "-f", pattern], capture_output=True, text=True)
    return result.returncode == 0


def terminate(pid, reason):
    try:
        os.kill(int(pid), signal.SIGTERM)
        log(f"终止无心跳进程 PID={pid}：{reason}")
    except ProcessLookupError:
        pass


def recover_stale_workers(pool):
    now = time.time()
    actions = []
    for worker in range(4):
        path = OUT / f"worker_{worker}_status.json"
        if not path.exists():
            continue
        status = json.loads(path.read_text())
        pid = pool.get("local", {}).get(str(worker), {}).get("pid")
        if pid and status.get("state") == "training" and now - status.get("updated_at", now) > LOCAL_STALE_SECONDS:
            terminate(pid, f"本机GPU {worker}超过30分钟无epoch心跳")
            actions.append(f"local_worker_{worker}")
    for worker in (4, 5):
        path = OUT / f"worker_{worker}_status.json"
        if not path.exists():
            continue
        status = json.loads(path.read_text())
        gpu = worker - 4
        pid = pool.get("remote", {}).get(str(gpu), {}).get("pid")
        if pid and status.get("state") == "remote_training" and now - status.get("updated_at", now) > REMOTE_STALE_SECONDS:
            request = f"request_{worker}.json"
            ssh("pkill -TERM -f " + shlex.quote(f"amos_mm_remote_pool.py --remote-worker .*{request}") + " || true")
            terminate(pid, f"远端GPU {gpu}单任务超过6小时")
            actions.append(f"remote_worker_{worker}")
    return actions


def restart_scheduler_if_needed(pool):
    heartbeat_age = time.time() - pool.get("updated_at", 0)
    if process_exists("src/scripts/run_amos_mm_gpu_pool.py"):
        return False
    if heartbeat_age < 3 * POLL_SECONDS:
        return False
    for section in ("local", "remote"):
        for slot in pool.get(section, {}).values():
            if slot.get("pid"):
                terminate(slot["pid"], "GPU池调度器已退出，清理孤儿子进程")
    ssh("pkill -TERM -f 'amos_mm_remote_pool.py --remote-worker .*request_[45].json' || true")
    subprocess.run(["tmux", "kill-session", "-t", "amos_mm_gpu_pool"], capture_output=True)
    command = (f"cd {shlex.quote(str(ROOT))} && exec {shlex.quote(sys.executable)} -u "
               "src/scripts/run_amos_mm_gpu_pool.py >> outputs/amos_mm/all_models/gpu_pool_supervisor.log 2>&1")
    run(["tmux", "new-session", "-d", "-s", "amos_mm_gpu_pool", command])
    log("检测到GPU池调度器退出，已清理孤儿进程并自动重启。")
    return True


def checkpoint_forecast(completed):
    checkpoints = [path for path in OUT.glob("*_labels/fold_*/*/best_model.pt") if path.is_file()]
    total = sum(path.stat().st_size for path in checkpoints)
    average = total / max(1, len(checkpoints))
    return {"local_checkpoint_count": len(checkpoints), "local_checkpoint_bytes": total,
            "average_checkpoint_bytes": average,
            "conservative_remaining_bytes": int(max(0, 200 - completed) * max(average, 128 * 1024**2))}


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    with (OUT / "unattended_monitor.lock").open("a") as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as error:
            raise SystemExit("已有AMOS-MM无人值守监控进程") from error
        protocol_digest = json.loads((OUT / "protocol.json").read_text())["protocol_sha256"]
        previous = json.loads(STATUS.read_text()) if STATUS.exists() else {}
        stopping = False

        def stop(_signal, _frame):
            nonlocal stopping
            stopping = True

        signal.signal(signal.SIGINT, stop)
        signal.signal(signal.SIGTERM, stop)
        log("无人值守监控启动：60秒巡检、最多3次任务尝试、磁盘低水位自动远端转存。")
        while not stopping:
            try:
                invalid = validate_results(protocol_digest)
                retried, final_errors = retry_errors()
                results, errors = result_and_error_paths()
                completed = len(results)
                disks = disk_snapshot()
                moved = spill_checkpoints(disks)
                pool = json.loads((OUT / "gpu_pool_status.json").read_text())
                stale_actions = recover_stale_workers(pool)
                restarted = restart_scheduler_if_needed(pool)
                now = time.time()
                last_progress_at = now if completed > previous.get("completed", -1) else previous.get("last_progress_at", now)
                state = "complete" if completed == 200 and not errors else "final_errors" if completed + len(errors) >= 200 else "running"
                current = {"state": state, "completed": completed, "total": 200, "current_errors": len(errors),
                           "final_error_jobs": final_errors, "retried_this_cycle": retried,
                           "invalid_results_this_cycle": invalid, "stale_workers_restarted": stale_actions,
                           "scheduler_restarted": restarted, "checkpoints_spilled_this_cycle": moved,
                           "last_progress_at": last_progress_at, "no_progress_seconds": now - last_progress_at,
                           "disk": disks, "checkpoint_forecast": checkpoint_forecast(completed),
                           "protocol_sha256": protocol_digest, "updated_at": now}
                save_json(STATUS, current)
                previous = current
                if state in {"complete", "final_errors"}:
                    log(f"监控结束：state={state}, completed={completed}, errors={len(errors)}")
                    break
            except Exception:
                error = traceback.format_exc()
                log("巡检异常，将在下一周期重试：\n" + error)
                save_json(STATUS, {"state": "monitor_cycle_error", "traceback": error, "updated_at": time.time()})
            time.sleep(POLL_SECONDS)


if __name__ == "__main__":
    main()
