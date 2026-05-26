from __future__ import annotations

import json
import os
import signal
import subprocess
import sys
import threading
import time
from collections import deque
from datetime import datetime
from pathlib import Path
from typing import Any


def _extract_cli_option(argv: list[str], option_name: str) -> str:
    for index, token in enumerate(argv):
        if token == option_name and index + 1 < len(argv):
            return argv[index + 1].strip()
        prefix = f"{option_name}="
        if token.startswith(prefix):
            return token[len(prefix):].strip()
    return ""


def _guess_task_name(argv: list[str], cwd: Path) -> str:
    explicit_task = _extract_cli_option(argv, "--task")
    if explicit_task:
        return explicit_task

    for option_name in ("--train-config", "--config", "--model-config"):
        raw_path = _extract_cli_option(argv, option_name)
        if not raw_path:
            continue
        path = Path(raw_path).expanduser()
        if not path.is_absolute():
            path = (cwd / path).resolve()
        parts_lower = {part.lower() for part in path.parts}
        if "task1" in parts_lower:
            return "task1"
        if "task2" in parts_lower:
            return "task2"
    return "task2"


def _resolve_path_config_path(argv: list[str], cwd: Path) -> Path:
    raw_path = _extract_cli_option(argv, "--config")
    if raw_path:
        config_path = Path(raw_path).expanduser()
        return config_path if config_path.is_absolute() else (cwd / config_path).resolve()
    task_name = _guess_task_name(argv, cwd)
    return (cwd / "configs" / task_name / "path.yaml").resolve()


def _load_output_root(argv: list[str], cwd: Path) -> Path:
    default_output_root = (cwd.parent / "outputs").resolve()
    config_path = _resolve_path_config_path(argv, cwd)
    if not config_path.is_file():
        return default_output_root

    try:
        import yaml

        payload = yaml.safe_load(config_path.read_text(encoding="utf-8"))
        paths = payload.get("paths", {}) if isinstance(payload, dict) else {}
        raw_output_dir = str(paths.get("output_dir", "")).strip()
        if not raw_output_dir:
            return default_output_root
        output_root = Path(raw_output_dir).expanduser()
        if not output_root.is_absolute():
            output_root = (config_path.parent / output_root).resolve()
        return output_root.resolve()
    except Exception:
        return default_output_root


def _read_meminfo() -> dict[str, int]:
    meminfo: dict[str, int] = {}
    try:
        for line in Path("/proc/meminfo").read_text(encoding="utf-8").splitlines():
            if ":" not in line:
                continue
            key, value = line.split(":", 1)
            fields = value.strip().split()
            if not fields:
                continue
            numeric = int(fields[0])
            if len(fields) >= 2 and fields[1].lower() == "kb":
                numeric *= 1024
            meminfo[key] = numeric
    except Exception:
        return {}
    return meminfo


def _read_process_status(pid: int) -> dict[str, int]:
    status_path = Path(f"/proc/{pid}/status")
    if not status_path.is_file():
        return {}
    result: dict[str, int] = {}
    try:
        for line in status_path.read_text(encoding="utf-8").splitlines():
            if ":" not in line:
                continue
            key, value = line.split(":", 1)
            fields = value.strip().split()
            if not fields:
                continue
            if key in {"VmRSS", "VmHWM", "VmSize"}:
                numeric = int(fields[0])
                if len(fields) >= 2 and fields[1].lower() == "kb":
                    numeric *= 1024
                result[key] = numeric
            elif key == "Threads":
                result[key] = int(fields[0])
    except Exception:
        return {}
    return result


def _query_gpu_stats() -> list[dict[str, Any]]:
    try:
        completed = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=index,name,memory.total,memory.used,utilization.gpu,utilization.memory",
                "--format=csv,noheader,nounits",
            ],
            check=False,
            capture_output=True,
            text=True,
            timeout=3,
        )
    except Exception:
        return []

    if completed.returncode != 0:
        return []

    gpus: list[dict[str, Any]] = []
    for line in completed.stdout.splitlines():
        parts = [part.strip() for part in line.split(",")]
        if len(parts) < 6:
            continue
        try:
            gpus.append(
                {
                    "index": int(parts[0]),
                    "name": parts[1],
                    "memory_total_mb": int(parts[2]),
                    "memory_used_mb": int(parts[3]),
                    "utilization_gpu_pct": int(parts[4]),
                    "utilization_memory_pct": int(parts[5]),
                }
            )
        except Exception:
            continue
    return gpus


def _query_recent_oom_messages() -> list[str]:
    commands = [
        ["dmesg", "-T", "--color=never"],
        ["dmesg"],
    ]
    keywords = ("out of memory", "oom-killer", "oom killer", "killed process")

    for command in commands:
        try:
            completed = subprocess.run(
                command,
                check=False,
                capture_output=True,
                text=True,
                timeout=3,
            )
        except Exception:
            continue

        if completed.returncode != 0 and not completed.stdout:
            continue
        lines = [
            line.strip()
            for line in completed.stdout.splitlines()
            if any(keyword in line.lower() for keyword in keywords)
        ]
        if lines:
            return lines[-20:]
    return []


def _bytes_to_gb(value: int | float | None) -> float | None:
    if value is None:
        return None
    return round(float(value) / (1024.0 ** 3), 3)


def _sample_runtime_state(child_pid: int) -> dict[str, Any]:
    meminfo = _read_meminfo()
    process_status = _read_process_status(child_pid)
    return {
        "ts": datetime.now().isoformat(timespec="seconds"),
        "host_memory": {
            "mem_total_gb": _bytes_to_gb(meminfo.get("MemTotal")),
            "mem_available_gb": _bytes_to_gb(meminfo.get("MemAvailable")),
            "swap_total_gb": _bytes_to_gb(meminfo.get("SwapTotal")),
            "swap_free_gb": _bytes_to_gb(meminfo.get("SwapFree")),
        },
        "process": {
            "pid": child_pid,
            "rss_gb": _bytes_to_gb(process_status.get("VmRSS")),
            "peak_rss_gb": _bytes_to_gb(process_status.get("VmHWM")),
            "vmsize_gb": _bytes_to_gb(process_status.get("VmSize")),
            "threads": process_status.get("Threads"),
        },
        "gpus": _query_gpu_stats(),
    }


def _build_kill_diagnosis(returncode: int, samples: list[dict[str, Any]], oom_lines: list[str]) -> dict[str, Any]:
    signal_number = -returncode if returncode < 0 else None
    signal_name = signal.Signals(signal_number).name if signal_number is not None and signal_number in signal.Signals._value2member_map_ else ""

    min_mem_available = min(
        (
            sample.get("host_memory", {}).get("mem_available_gb")
            for sample in samples
            if sample.get("host_memory", {}).get("mem_available_gb") is not None
        ),
        default=None,
    )
    min_swap_free = min(
        (
            sample.get("host_memory", {}).get("swap_free_gb")
            for sample in samples
            if sample.get("host_memory", {}).get("swap_free_gb") is not None
        ),
        default=None,
    )
    peak_rss = max(
        (
            sample.get("process", {}).get("peak_rss_gb") or 0.0
            for sample in samples
        ),
        default=0.0,
    )
    peak_gpu_usage_ratio = 0.0
    peak_gpu_snapshot: dict[str, Any] | None = None
    for sample in samples:
        for gpu in sample.get("gpus", []):
            total_mb = float(gpu.get("memory_total_mb") or 0.0)
            used_mb = float(gpu.get("memory_used_mb") or 0.0)
            if total_mb <= 0:
                continue
            ratio = used_mb / total_mb
            if ratio > peak_gpu_usage_ratio:
                peak_gpu_usage_ratio = ratio
                peak_gpu_snapshot = gpu

    probable_cause = "未知，需要结合训练日志进一步判断"
    evidence: list[str] = []
    if oom_lines:
        probable_cause = "高概率是系统 OOM killer 触发的内存杀进程"
        evidence.append("检测到最近的 dmesg 中存在 OOM / Killed process 记录")
    elif signal_number == signal.SIGKILL:
        if min_mem_available is not None and min_mem_available <= 1.5:
            probable_cause = "高概率是主机内存不足触发的 SIGKILL"
            evidence.append(f"采样期间主机可用内存最低只有 {min_mem_available:.3f} GB")
            if min_swap_free is not None:
                evidence.append(f"采样期间主机可用 Swap 最低约 {min_swap_free:.3f} GB")
        elif peak_gpu_usage_ratio >= 0.95 and peak_gpu_snapshot is not None:
            probable_cause = "较大概率是 GPU 显存打满后被外层环境强制杀掉"
            evidence.append(
                f"GPU{peak_gpu_snapshot.get('index')} 峰值显存占用约 {peak_gpu_usage_ratio * 100:.1f}%"
            )
        else:
            probable_cause = "收到 SIGKILL，可能是外部 kill -9、调度器回收，或资源限制导致"
            if peak_rss:
                evidence.append(f"进程峰值 RSS 约 {peak_rss:.3f} GB")
    elif signal_number == signal.SIGTERM:
        probable_cause = "收到 SIGTERM，通常是外部终止、会话关闭或调度器超时"
    elif returncode >= 128:
        probable_cause = f"进程以非零退出码 {returncode} 结束，可能经历了外层包装后的信号退出"

    return {
        "returncode": returncode,
        "signal_number": signal_number,
        "signal_name": signal_name,
        "probable_cause": probable_cause,
        "evidence": evidence,
        "min_mem_available_gb": min_mem_available,
        "min_swap_free_gb": min_swap_free,
        "peak_process_rss_gb": peak_rss,
        "peak_gpu_usage_ratio": round(peak_gpu_usage_ratio, 4),
        "peak_gpu_snapshot": peak_gpu_snapshot,
        "recent_oom_messages": oom_lines,
    }


def _write_kill_report(
    *,
    diagnostics_dir: Path,
    command: list[str],
    child_pid: int,
    returncode: int,
    samples: list[dict[str, Any]],
) -> Path:
    diagnostics_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    oom_lines = _query_recent_oom_messages()
    diagnosis = _build_kill_diagnosis(returncode, samples, oom_lines)

    payload = {
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "cwd": str(Path.cwd()),
        "command": command,
        "child_pid": child_pid,
        "diagnosis": diagnosis,
        "sample_count": len(samples),
        "recent_samples": samples[-60:],
    }
    report_path = diagnostics_dir / f"train_kill_report_{timestamp}.json"
    report_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")

    markdown_lines = [
        "# 训练异常退出诊断",
        "",
        f"- 生成时间：{payload['generated_at']}",
        f"- 工作目录：`{payload['cwd']}`",
        f"- 子进程 PID：`{child_pid}`",
        f"- 返回码：`{returncode}`",
        f"- 推测原因：{diagnosis['probable_cause']}",
    ]
    if diagnosis["signal_name"]:
        markdown_lines.append(f"- 信号：`{diagnosis['signal_name']}`")
    if diagnosis["evidence"]:
        markdown_lines.append("- 关键证据：")
        markdown_lines.extend([f"  - {item}" for item in diagnosis["evidence"]])
    if diagnosis["recent_oom_messages"]:
        markdown_lines.append("- 最近 OOM / Killed process 日志：")
        markdown_lines.extend([f"  - `{item}`" for item in diagnosis["recent_oom_messages"]])
    markdown_lines.extend(
        [
            "",
            "## 文件",
            "",
            f"- JSON 详情：`{report_path}`",
        ]
    )
    report_md_path = diagnostics_dir / f"train_kill_report_{timestamp}.md"
    report_md_path.write_text("\n".join(markdown_lines) + "\n", encoding="utf-8")
    return report_path


def supervise_train_invocation_if_needed() -> None:
    if os.environ.get("PROJECT4_TRAIN_CHILD") == "1":
        return
    if os.environ.get("PROJECT4_DISABLE_KILL_DIAG") == "1":
        return

    cwd = Path.cwd()
    output_root = _load_output_root(sys.argv[1:], cwd)
    diagnostics_dir = output_root / "runtime_diagnostics" / "train_kill_reports"

    child_env = os.environ.copy()
    child_env["PROJECT4_TRAIN_CHILD"] = "1"
    child_env.setdefault("PYTHONUNBUFFERED", "1")
    child = subprocess.Popen([sys.executable, *sys.argv], env=child_env)

    stop_event = threading.Event()
    samples: deque[dict[str, Any]] = deque(maxlen=240)

    def sampler() -> None:
        while not stop_event.is_set():
            samples.append(_sample_runtime_state(child.pid))
            stop_event.wait(5.0)

    sampler_thread = threading.Thread(target=sampler, name="train-kill-sampler", daemon=True)
    sampler_thread.start()

    try:
        while True:
            try:
                returncode = child.wait(timeout=1.0)
                break
            except subprocess.TimeoutExpired:
                continue
    except KeyboardInterrupt:
        try:
            child.send_signal(signal.SIGINT)
            returncode = child.wait(timeout=10.0)
        except Exception:
            child.kill()
            returncode = child.wait()
    finally:
        stop_event.set()
        sampler_thread.join(timeout=1.0)

    abnormal = returncode != 0
    signal_exit = returncode < 0
    likely_wrapped_signal = returncode in {137, 143}

    if abnormal and (signal_exit or likely_wrapped_signal):
        report_path = _write_kill_report(
            diagnostics_dir=diagnostics_dir,
            command=[sys.executable, *sys.argv],
            child_pid=child.pid,
            returncode=returncode,
            samples=list(samples),
        )
        print(
            f"[train.py] 检测到训练进程异常退出，诊断报告已写入: {report_path}",
            file=sys.stderr,
        )

    if returncode < 0:
        os._exit(128 + (-returncode))
    raise SystemExit(returncode)


__all__ = ["supervise_train_invocation_if_needed"]
