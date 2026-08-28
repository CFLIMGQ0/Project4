#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="/home/Lim/Project4/src"
PYTHON_BIN="/home/Lim/anaconda3/envs/myenv/bin/python"
RUNNER="$ROOT_DIR/scripts/task3_table2_5fold.py"
UPDATER="$ROOT_DIR/scripts/update_t3_multimodal_sotas_table.py"
CONFIG="$ROOT_DIR/configs/task3/t3_multimodal_sotas_5fold.yaml"
OUTPUT_DIR="/home/Lim/Project4/outputs/train_runs/task3/t3_multimodal_sotas_5fold"
LOG_FILE="$OUTPUT_DIR/logs/finalize_table.log"

mkdir -p "$(dirname "$LOG_FILE")"
cd "$ROOT_DIR"

while true; do
  completed="$(find "$OUTPUT_DIR/image" -path '*/test_macro_f1/metrics.json' -type f | wc -l)"
  echo "[$(date '+%F %T')] 已完成 $completed/100 折" | tee -a "$LOG_FILE"
  if [[ "$completed" -eq 100 ]]; then
    break
  fi
  if [[ "$completed" -gt 100 ]]; then
    echo "完成文件数异常：$completed，大于预期100，停止自动落表。" | tee -a "$LOG_FILE"
    exit 1
  fi
  sleep 60
done

"$PYTHON_BIN" "$RUNNER" --config "$CONFIG" --summarize-only 2>&1 | tee -a "$LOG_FILE"
"$PYTHON_BIN" "$UPDATER" --check-only 2>&1 | tee -a "$LOG_FILE"
"$PYTHON_BIN" "$UPDATER" 2>&1 | tee -a "$LOG_FILE"

if rg -n '38/100|358/420|不足5折|暂定值|0/5|暂无可汇总' "$ROOT_DIR/table.md" | tee -a "$LOG_FILE"; then
  echo "table.md仍含临时结果标记，停止并报告错误。" | tee -a "$LOG_FILE"
  exit 1
fi

echo "[$(date '+%F %T')] 100/100折已汇总，table.md更新并复核完成。" | tee -a "$LOG_FILE"
