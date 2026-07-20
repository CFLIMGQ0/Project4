#!/usr/bin/env bash
set -euo pipefail

SESSION_NAME="t3_main_model"
ROOT_DIR="/home/Lim/Project4/src"
PYTHON_BIN="/home/Lim/anaconda3/envs/myenv/bin/python"
RUNNER="${ROOT_DIR}/scripts/task3_main_model_5fold.py"
OUTPUT_DIR="/home/Lim/Project4/outputs/train_runs/task3/t3_main_model"
LOG_DIR="${OUTPUT_DIR}/logs"

mkdir -p "${LOG_DIR}"

if tmux has-session -t "${SESSION_NAME}" 2>/dev/null; then
    echo "tmux会话已存在：${SESSION_NAME}"
    exit 0
fi

worker_command() {
    local gpu_id="$1"
    local datasets="$2"
    local log_name="$3"
    printf '%q ' bash -lc "while ! nvidia-smi -i ${gpu_id} >/dev/null 2>&1; do echo \"[TASK3] GPU ${gpu_id} 尚不可用，30秒后重试\"; sleep 30; done; cd '${ROOT_DIR}'; CUDA_VISIBLE_DEVICES=${gpu_id} '${PYTHON_BIN}' '${RUNNER}' --datasets '${datasets}' 2>&1 | tee -a '${LOG_DIR}/${log_name}.log'"
}

tmux new-session -d -s "${SESSION_NAME}" -n gpu0 "$(worker_command 0 regular_white_light gpu0_regular_white_light)"
tmux new-window -t "${SESSION_NAME}" -n gpu1 "$(worker_command 1 chromoscopic,ultrasound gpu1_chromoscopic_ultrasound)"
tmux new-window -t "${SESSION_NAME}" -n gpu2 "$(worker_command 2 surgical gpu2_surgical)"

echo "已创建tmux会话：${SESSION_NAME}"
echo "GPU0：常规白光胃镜5折"
echo "GPU1：染色胃镜5折，完成后继续超声胃镜5折"
echo "GPU2：手术胃镜5折"
