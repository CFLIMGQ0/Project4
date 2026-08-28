#!/usr/bin/env bash
set -euo pipefail

SESSION_NAME="t3_gpu_pool"
SRC_ROOT="/xmlg/Lim/Project4/src"
PYTHON_BIN="/xmlg/Lim/conda/envs/myenv/bin/python"
SCHEDULER="$SRC_ROOT/scripts/task3_gpu_pool_scheduler.py"
LOG_FILE="/xmlg/Lim/Project4/outputs/train_runs/task3/t3_remaining_experiments/pool_scheduler.log"

mkdir -p "$(dirname "$LOG_FILE")"

if tmux has-session -t "$SESSION_NAME" 2>/dev/null; then
  echo "tmux会话已存在：$SESSION_NAME"
  exit 0
fi

tmux new-session -d -s "$SESSION_NAME" -n scheduler \
  "cd '$SRC_ROOT'; '$PYTHON_BIN' '$SCHEDULER' \
    --groups apro,distill,table3,table4 \
    --local-gpus 0,1,2,3 \
    --remote-gpus 0,1 \
    --remote-name xmlg202 \
    --remote-target Lim@172.16.170.202 \
    --remote-project-root /home/Lim/Project4 \
    --remote-python /home/Lim/conda/envs/myenv/bin/python \
    --ssh-key /home/Lim/.ssh/id_ed25519_project4_pool \
    --max-per-gpu 2 \
    --estimated-memory-mb 6500 \
    --min-headroom-mb 1000 \
    --poll-seconds 30 \
    2>&1 | tee -a '$LOG_FILE'; exec bash"

echo "已启动六卡池tmux会话：$SESSION_NAME"
echo "日志：$LOG_FILE"
