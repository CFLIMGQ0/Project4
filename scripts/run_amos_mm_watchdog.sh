#!/usr/bin/env bash
set -u

ROOT=/xmlg/Lim/Project4
PYTHON=/xmlg/Lim/conda/envs/myenv/bin/python
LOG="$ROOT/outputs/amos_mm/all_models/unattended_monitor_console.log"

cd "$ROOT" || exit 1
while true; do
  "$PYTHON" -u src/scripts/monitor_amos_mm_unattended.py >>"$LOG" 2>&1
  read -r completed errors < <("$PYTHON" - <<'PY'
from pathlib import Path
out = Path("outputs/amos_mm/all_models")
print(len(list(out.glob("*_labels/fold_*/*/result.json"))),
      len(list(out.glob("*_labels/fold_*/*/error.json"))))
PY
  )
  if (( completed + errors >= 200 )); then
    exit 0
  fi
  printf '[%s] watchdog exited; restarting in 30 seconds\n' "$(date '+%F %T')" >>"$LOG"
  sleep 30
done
