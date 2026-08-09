#!/usr/bin/env bash
set -euo pipefail

PROJECT_DIR="/home/Lim/Project4/src"
PYTHON_BIN="/home/Lim/anaconda3/envs/myenv/bin/python"
OUTPUT_DIR="/home/Lim/Project4/outputs/train_runs/task2/table2_multimodal_sotas"
LOG_DIR="${OUTPUT_DIR}/logs"
SESSION_NAME="t2_multimodal_sotas"

if ! nvidia-smi >/dev/null 2>&1; then
  echo "GPU驱动当前不可用：nvidia-smi 执行失败，未启动训练。"
  exit 1
fi

GPU_COUNT="$("${PYTHON_BIN}" -c 'import torch; print(torch.cuda.device_count())')"
if [[ "${GPU_COUNT}" -lt 3 ]]; then
  echo "至少需要3张可见GPU，当前PyTorch仅检测到 ${GPU_COUNT} 张，未启动训练。"
  exit 1
fi

if tmux has-session -t "${SESSION_NAME}" 2>/dev/null; then
  echo "tmux会话 ${SESSION_NAME} 已存在；为保护断点，不重复创建。"
  echo "查看命令：tmux attach -t ${SESSION_NAME}"
  exit 0
fi

mkdir -p "${LOG_DIR}"

run_one() {
  local gpu="$1"
  local model="$2"
  local config="${PROJECT_DIR}/configs/task2/t2_multimodal_sotas_gpu${gpu}.yaml"
  local log="${LOG_DIR}/${model}.log"
  echo "[$(date '+%F %T')] GPU=${gpu} 启动 ${model}" | tee -a "${log}"
  "${PYTHON_BIN}" "${PROJECT_DIR}/train.py" \
    --task task2 \
    --train-config "${config}" \
    --model-config "${PROJECT_DIR}/configs/task2/model.yaml" \
    --models "${model}" 2>&1 | tee -a "${log}"
}

export -f run_one
export PROJECT_DIR PYTHON_BIN OUTPUT_DIR LOG_DIR

tmux new-session -d -s "${SESSION_NAME}" -n gpu0 \
  "bash -lc 'set -euo pipefail; run_one 0 task2_hasan_itf_2024; run_one 0 task2_radfuse_2025; exec bash'"
tmux new-window -t "${SESSION_NAME}" -n gpu1 \
  "bash -lc 'set -euo pipefail; sleep 3; run_one 1 task2_mmfnet_2024; run_one 1 task2_mmtf_2025; exec bash'"
tmux new-window -t "${SESSION_NAME}" -n gpu2 \
  "bash -lc 'set -euo pipefail; sleep 6; run_one 2 task2_saif_2025; exec bash'"
tmux new-window -t "${SESSION_NAME}" -n finalize \
  "bash -lc '${PROJECT_DIR}/scripts/finalize_t2_multimodal_sotas.sh; exec bash'"

echo "已启动tmux会话：${SESSION_NAME}"
echo "查看全部窗口：tmux list-windows -t ${SESSION_NAME}"
echo "进入会话：tmux attach -t ${SESSION_NAME}"
echo "日志目录：${LOG_DIR}"
