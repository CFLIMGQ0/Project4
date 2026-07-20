#!/usr/bin/env bash
set -euo pipefail

SESSION_NAME="exp11_module_ablation"
PROJECT_DIR="/home/Lim/Project4/src"
OUTPUT_DIR="/home/Lim/Project4/outputs/train_runs/task2/exp11_module_ablation"
LOG_PATH="${OUTPUT_DIR}/tmux_exp11_module_ablation.log"
PYTHON_BIN="/home/Lim/anaconda3/envs/myenv/bin/python"

if tmux has-session -t "${SESSION_NAME}" 2>/dev/null; then
  echo "tmux 会话 ${SESSION_NAME} 已存在。"
  echo "查看命令：tmux attach -t ${SESSION_NAME}"
  exit 0
fi

mkdir -p "${OUTPUT_DIR}"

tmux new-session -d -s "${SESSION_NAME}" \
  "cd '${PROJECT_DIR}' && MPLCONFIGDIR=/tmp/matplotlib-exp11 '${PYTHON_BIN}' train.py 2>&1 | tee -a '${LOG_PATH}'"

echo "已启动 exp11_module_ablation 的10组组合实验。"
echo "tmux 会话：${SESSION_NAME}"
echo "查看命令：tmux attach -t ${SESSION_NAME}"
echo "日志文件：${LOG_PATH}"
