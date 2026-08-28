#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="/home/Lim/Project4/src"
PYTHON_BIN="/home/Lim/anaconda3/envs/myenv/bin/python"
RUNNER="$ROOT_DIR/scripts/task3_table2_5fold.py"
OUTPUT_DIR="/home/Lim/Project4/outputs/train_runs/task3/t3_table2_5fold"
LOG_DIR="$OUTPUT_DIR/logs"
STATUS_DIR="$OUTPUT_DIR/status"
SESSION="t3_table2_5fold"

mkdir -p "$LOG_DIR" "$STATUS_DIR"
rm -f "$STATUS_DIR"/text_shard_*.failed "$STATUS_DIR/image.failed"

if tmux has-session -t "$SESSION" 2>/dev/null; then
  echo "tmux会话已存在：$SESSION"
  exit 1
fi

for shard in 0 1 2; do
  command="set -o pipefail; cd '$ROOT_DIR'; if CUDA_VISIBLE_DEVICES=$shard '$PYTHON_BIN' '$RUNNER' --modality text --num-shards 3 --shard-index $shard 2>&1 | tee -a '$LOG_DIR/text_shard_${shard}.log'; then touch '$STATUS_DIR/text_shard_${shard}.done'; else touch '$STATUS_DIR/text_shard_${shard}.failed'; fi"
  if [[ "$shard" == "0" ]]; then
    tmux new-session -d -s "$SESSION" -n "text0" "$command"
  else
    tmux new-window -t "$SESSION" -n "text${shard}" "$command"
  fi
done

controller="set -o pipefail; cd '$ROOT_DIR'; \
while true; do \
  if compgen -G '$STATUS_DIR/text_shard_*.failed' >/dev/null; then echo '[TASK3-T2] 文本阶段存在失败任务，图像阶段未启动'; exit 1; fi; \
  if [[ -f '$STATUS_DIR/text_shard_0.done' && -f '$STATUS_DIR/text_shard_1.done' && -f '$STATUS_DIR/text_shard_2.done' ]]; then break; fi; \
  echo '[TASK3-T2] 等待三个文本分片完成，60秒后复查'; sleep 60; \
done; \
echo '[TASK3-T2] 文本阶段完成，开始三卡图像模型阶段'; \
if CUDA_VISIBLE_DEVICES=0,1,2 '$PYTHON_BIN' '$RUNNER' --modality image 2>&1 | tee -a '$LOG_DIR/image_all_models.log'; then \
  '$PYTHON_BIN' '$RUNNER' --summarize-only 2>&1 | tee -a '$LOG_DIR/summary.log'; \
else touch '$STATUS_DIR/image.failed'; exit 1; fi"
tmux new-window -t "$SESSION" -n "image" "$controller"
tmux select-window -t "$SESSION:text0"

echo "已启动tmux会话：$SESSION"
echo "查看：tmux attach -t $SESSION"
echo "文本阶段由3张卡各运行一个分片；完成后图像模型自动使用3张卡联合训练。"
