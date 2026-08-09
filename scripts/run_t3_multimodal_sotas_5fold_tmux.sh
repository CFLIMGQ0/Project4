#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="/home/Lim/Project4/src"
PYTHON_BIN="/home/Lim/anaconda3/envs/myenv/bin/python"
RUNNER="$ROOT_DIR/scripts/task3_table2_5fold.py"
CONFIG="$ROOT_DIR/configs/task3/t3_multimodal_sotas_5fold.yaml"
OUTPUT_DIR="/home/Lim/Project4/outputs/train_runs/task3/t3_multimodal_sotas_5fold"
LOG_DIR="$OUTPUT_DIR/logs"
SESSION="t3_multimodal_sotas_5fold"
RUN_ID="$(date '+%Y%m%d_%H%M%S')"
STATUS_DIR="$OUTPUT_DIR/status/$RUN_ID"

if ! nvidia-smi >/dev/null 2>&1; then
  echo "GPU驱动不可用，未启动TASK3图文五折实验。"
  exit 1
fi

GPU_COUNT="$("$PYTHON_BIN" -c 'import torch; print(torch.cuda.device_count())')"
if [[ "$GPU_COUNT" -lt 3 ]]; then
  echo "至少需要3张可见GPU，当前PyTorch仅检测到 $GPU_COUNT 张，未启动实验。"
  exit 1
fi

if tmux has-session -t "$SESSION" 2>/dev/null; then
  echo "tmux会话已存在：$SESSION"
  echo "查看命令：tmux attach -t $SESSION"
  exit 0
fi

mkdir -p "$LOG_DIR" "$STATUS_DIR"

# 先进行缓存、文本遮蔽、四类数据归属和患者级五折泄漏审计。
"$PYTHON_BIN" "$RUNNER" --config "$CONFIG" --modality image --prepare-only \
  2>&1 | tee -a "$LOG_DIR/prepare.log"

for shard in 0 1 2; do
  command="set -o pipefail; cd '$ROOT_DIR'; if CUDA_VISIBLE_DEVICES=$shard '$PYTHON_BIN' '$RUNNER' --config '$CONFIG' --modality image --num-shards 3 --shard-index $shard 2>&1 | tee -a '$LOG_DIR/gpu${shard}.log'; then touch '$STATUS_DIR/gpu${shard}.done'; else touch '$STATUS_DIR/gpu${shard}.failed'; fi; exec bash"
  if [[ "$shard" == "0" ]]; then
    tmux new-session -d -s "$SESSION" -n "gpu0" "$command"
  else
    tmux new-window -t "$SESSION" -n "gpu${shard}" "$command"
  fi
done

controller="set -o pipefail; cd '$ROOT_DIR'; \
while true; do \
  if compgen -G '$STATUS_DIR/*.failed' >/dev/null; then echo '[TASK3-MM] 检测到失败分片，停止自动汇总'; exit 1; fi; \
  if [[ -f '$STATUS_DIR/gpu0.done' && -f '$STATUS_DIR/gpu1.done' && -f '$STATUS_DIR/gpu2.done' ]]; then break; fi; \
  echo '[TASK3-MM] 训练尚未全部完成，60秒后复查'; sleep 60; \
done; \
'$PYTHON_BIN' '$RUNNER' --config '$CONFIG' --summarize-only 2>&1 | tee -a '$LOG_DIR/summary.log'; \
echo '[TASK3-MM] 100次训练全部完成并已汇总'; exec bash"
tmux new-window -t "$SESSION" -n "finalize" "$controller"
tmux select-window -t "$SESSION:gpu0"

echo "已启动tmux会话：$SESSION"
echo "任务总数：5个模型 × 4个数据集 × 5折 = 100次训练"
echo "GPU分配：三个单卡分片，预计分别执行34、33、33次"
echo "查看命令：tmux attach -t $SESSION"
echo "日志目录：$LOG_DIR"
