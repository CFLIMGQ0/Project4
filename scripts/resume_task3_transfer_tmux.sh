#!/usr/bin/env bash
set -euo pipefail

SESSION="transfer"
SOURCE_ROOT="/home/Lim/Project4"
TARGET="Lim@100.84.245.7:/xmlg/Lim/Project4/"
MANIFEST="/home/Lim/task3_transfer_files.txt"
LOG_FILE="/home/Lim/transfer.log"

if tmux has-session -t "$SESSION" 2>/dev/null; then
  echo "tmux会话已存在：$SESSION"
  exit 0
fi
if [[ ! -s "$MANIFEST" ]]; then
  echo "传输清单不存在或为空：$MANIFEST"
  exit 1
fi

command="cd '$SOURCE_ROOT'; \
printf '\n===== TASK3 transfer start %s -> $TARGET =====\n' \"\$(date '+%F %T')\" | tee -a '$LOG_FILE'; \
set -o pipefail; \
rsync -ah --info=progress2 --partial --append-verify --stats \
  --files-from='$MANIFEST' \
  -e 'ssh -o ServerAliveInterval=30 -o ServerAliveCountMax=20 -o TCPKeepAlive=yes' \
  ./ '$TARGET' 2>&1 | tee -a '$LOG_FILE'; \
status=\${PIPESTATUS[0]}; \
printf '\n===== TASK3 transfer exit=%s at %s =====\n' \"\$status\" \"\$(date '+%F %T')\" | tee -a '$LOG_FILE'; \
exec bash"

tmux new-session -d -s "$SESSION" -n rsync "$command"
echo "已启动tmux会话：$SESSION"
echo "日志：$LOG_FILE"
