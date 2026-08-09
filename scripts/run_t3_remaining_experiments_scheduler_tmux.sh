#!/usr/bin/env bash
set -euo pipefail

SESSION="t3_remaining_experiments"
ROOT_DIR="/xmlg/Lim/Project4/src"
PYTHON_BIN="/xmlg/Lim/conda/envs/myenv/bin/python"
SCHEDULER="$ROOT_DIR/scripts/task3_remaining_experiments_scheduler.py"
OUTPUT_DIR="/xmlg/Lim/Project4/outputs/train_runs/task3/t3_remaining_experiments"
LOG_FILE="$OUTPUT_DIR/scheduler.log"

if tmux has-session -t "$SESSION" 2>/dev/null; then
  echo "tmux会话已存在：$SESSION"
  exit 0
fi

mkdir -p "$OUTPUT_DIR"
command="cd '$ROOT_DIR'; '$PYTHON_BIN' '$SCHEDULER' --gpus 0,1,2,3 --max-per-gpu 2 --estimated-memory-mb 6500 --min-headroom-mb 1000 --poll-seconds 30 --cache-wait-seconds 60 2>&1 | tee -a '$LOG_FILE'; exec bash"
tmux new-session -d -s "$SESSION" -n scheduler "$command"

echo "已启动tmux会话：$SESSION"
echo "调度日志：$LOG_FILE"
echo "状态文件：$OUTPUT_DIR/scheduler_state.json"
