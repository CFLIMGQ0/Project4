#!/usr/bin/env bash
set -euo pipefail

SESSION="t3_image_branch_distill"
ROOT_DIR="/home/Lim/Project4/src"
PYTHON_BIN="/home/Lim/anaconda3/envs/myenv/bin/python"
RUNNER="$ROOT_DIR/scripts/task3_main_model_5fold.py"
CONFIG="$ROOT_DIR/configs/task3/t3_image_branch_distill.yaml"
OUTPUT_DIR="/home/Lim/Project4/outputs/train_runs/task3/t3_image_branch_distill"
LOG_DIR="$OUTPUT_DIR/logs"
STATUS_DIR="$OUTPUT_DIR/status"

if tmux has-session -t "$SESSION" 2>/dev/null; then
  echo "tmux会话已存在：$SESSION"
  exit 0
fi
mkdir -p "$LOG_DIR" "$STATUS_DIR"

gpu_is_idle() {
  local gpu_id="$1"
  local processes
  processes="$(nvidia-smi -i "$gpu_id" --query-compute-apps=pid --format=csv,noheader 2>/dev/null || true)"
  [[ -z "${processes//[[:space:]]/}" ]]
}

for shard in 0 1 2; do
  command="set -o pipefail; cd '$ROOT_DIR'; while ! nvidia-smi -i '$shard' >/dev/null 2>&1 || ! bash -lc 'processes=\"\$(nvidia-smi -i $shard --query-compute-apps=pid --format=csv,noheader 2>/dev/null || true)\"; [[ -z \"\${processes//[[:space:]]/}\" ]]'; do echo '[T3-DISTILL] GPU $shard仍在使用，60秒后重试'; sleep 60; done; if CUDA_VISIBLE_DEVICES=$shard '$PYTHON_BIN' '$RUNNER' --config '$CONFIG' --num-shards 3 --shard-index $shard 2>&1 | tee -a '$LOG_DIR/gpu${shard}.log'; then touch '$STATUS_DIR/gpu${shard}.done'; else touch '$STATUS_DIR/gpu${shard}.failed'; fi; exec bash"
  if [[ "$shard" -eq 0 ]]; then
    tmux new-session -d -s "$SESSION" -n "gpu0" "$command"
  else
    tmux new-window -t "$SESSION" -n "gpu${shard}" "$command"
  fi
done

controller="set -o pipefail; cd '$ROOT_DIR'; while true; do if compgen -G '$STATUS_DIR/*.failed' >/dev/null; then echo '[T3-DISTILL] 检测到失败分片'; exit 1; fi; if [[ -f '$STATUS_DIR/gpu0.done' && -f '$STATUS_DIR/gpu1.done' && -f '$STATUS_DIR/gpu2.done' ]]; then break; fi; echo '[T3-DISTILL] 训练尚未全部完成，60秒后复查'; sleep 60; done; '$PYTHON_BIN' '$RUNNER' --config '$CONFIG' --summarize-only 2>&1 | tee -a '$LOG_DIR/summary.log'; echo '[T3-DISTILL] 20折训练与汇总全部完成'; exec bash"
tmux new-window -t "$SESSION" -n finalize "$controller"
tmux select-window -t "$SESSION:gpu0"

echo "已启动tmux会话：$SESSION"
echo "输出目录：$OUTPUT_DIR"
