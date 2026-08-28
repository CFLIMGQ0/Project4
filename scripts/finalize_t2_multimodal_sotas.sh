#!/usr/bin/env bash
set -euo pipefail

PROJECT_DIR="/home/Lim/Project4/src"
PYTHON_BIN="/home/Lim/anaconda3/envs/myenv/bin/python"
OUTPUT_DIR="/home/Lim/Project4/outputs/train_runs/task2/table2_multimodal_sotas"
FINALIZE_LOG="${OUTPUT_DIR}/logs/finalize.log"
EXPECTED_MODELS=5

mkdir -p "$(dirname "${FINALIZE_LOG}")"
echo "[$(date '+%F %T')] 开始等待 ${EXPECTED_MODELS} 个图文SOTA结果。" | tee -a "${FINALIZE_LOG}"

while true; do
  RESULT_COUNT="$(
    "${PYTHON_BIN}" "${PROJECT_DIR}/scripts/task2_multimodal_sotas_table.py" \
      --output-dir "${OUTPUT_DIR}" | sed -n 's#^已完成 \([0-9][0-9]*\)/.*#\1#p'
  )"
  RESULT_COUNT="${RESULT_COUNT:-0}"
  echo "[$(date '+%F %T')] 当前完成 ${RESULT_COUNT}/${EXPECTED_MODELS}。" | tee -a "${FINALIZE_LOG}"

  if [[ "${RESULT_COUNT}" -eq "${EXPECTED_MODELS}" ]]; then
    "${PYTHON_BIN}" "${PROJECT_DIR}/scripts/task2_multimodal_sotas_table.py" \
      --output-dir "${OUTPUT_DIR}" \
      --table "${PROJECT_DIR}/table.md" \
      --update-table 2>&1 | tee -a "${FINALIZE_LOG}"
    echo "[$(date '+%F %T')] 五个结果齐全，table.md已更新。" | tee -a "${FINALIZE_LOG}"
    exit 0
  fi

  if ! pgrep -f 'train.py.*task2_(hasan_itf|mmfnet|saif|mmtf|radfuse)' >/dev/null; then
    echo "[$(date '+%F %T')] 未检测到训练进程但结果不完整，请检查各模型日志。" | tee -a "${FINALIZE_LOG}"
    exit 1
  fi
  sleep 60
done
