#!/usr/bin/env python3
"""恢复因磁盘争用而暂停的私有缓存特征提取进程。"""

from __future__ import annotations

import json
import os
from pathlib import Path
import signal

ROOT = Path(__file__).resolve().parents[2]
STATE = ROOT / "outputs/paper_results/private_feature_logs/paused_state.json"


def main() -> None:
    if not STATE.exists():
        print("没有待恢复的特征进程", flush=True)
        return
    data = json.loads(STATE.read_text())
    for row in data["paused_workers"]:
        pid = int(row["pid"])
        cmdline = Path(f"/proc/{pid}/cmdline")
        if not cmdline.exists() or b"prepare_private_paper_features.py" not in cmdline.read_bytes():
            raise RuntimeError(f"PID {pid} 不再是预期的特征进程")
        os.kill(pid, signal.SIGCONT)
        print(f"已恢复 {row['dataset']} worker{row['worker_index']} pid={pid}", flush=True)
    data["resumed"] = True
    STATE.write_text(json.dumps(data, ensure_ascii=False, indent=2)+"\n", encoding="utf-8")


if __name__ == "__main__":
    main()
