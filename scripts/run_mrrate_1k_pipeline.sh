#!/usr/bin/env bash
set -u

ROOT=/xmlg/Lim/Project4
PYTHON=/xmlg/Lim/conda/envs/myenv/bin/python
INPUT="$ROOT/outputs/mr_rate_1k/experiment"
LOG="$ROOT/outputs/mr_rate_1k/pipeline.log"

cd "$ROOT" || exit 1
printf '[%s] MR-RATE pipeline watcher started\n' "$(date '+%F %T')" >>"$LOG"
while true; do
  count=$(find "$INPUT/features" -name '*.npz' 2>/dev/null | wc -l)
  state=$("$PYTHON" -c 'import json; print(json.load(open("outputs/mr_rate_1k/experiment/feature_state.json"))["status"])' 2>/dev/null || true)
  printf '[%s] features=%s state=%s\n' "$(date '+%F %T')" "$count" "$state" >>"$LOG"
  if [ "$count" -ge 1000 ] && [ "$state" = complete ]; then
    printf '[%s] features complete; launching six-GPU table2 pool\n' "$(date '+%F %T')" >>"$LOG"
    exec "$PYTHON" -u src/scripts/run_mrrate_1k_gpu_pool.py >>"$LOG" 2>&1
  fi
  sleep 60
done
