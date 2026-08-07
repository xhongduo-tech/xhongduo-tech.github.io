#!/bin/bash
# 为「所有还有 `- [ ]` 条目的专题」各启动一个独立写手 session（1M 上下文，headless）。
# 用 nohup 完全脱离主会话：即使本 master session 关闭，worker 也会继续写盘。
# 已由 run-topic-session.sh 自写 logs/workers/<tier>-<key>.log（tee）。
#
# 用法：scripts/launch-all-topics.sh [MAX_TOPICS]   # 可选：最多启动前 N 个专题（按剩余数降序）
# 输出：logs/workers/launched.pids  记录 "tier/key PID"
set -euo pipefail
mkdir -p logs/workers
PIDFILE="logs/workers/launched.pids"
: > "$PIDFILE"

# 收集还有剩余条目的专题，按剩余数降序
LIST=$(mktemp)
for idx in docs/posts/*/*/index.md; do
  remaining=$(grep -c -- '- \[ \]' "$idx" || true)
  [ "${remaining:-0}" -gt 0 ] || continue
  tier=$(echo "$idx" | sed 's#docs/posts/##; s#/.*##')
  key=$(echo "$idx" | sed 's#docs/posts/[^/]*/##; s#/index.md##')
  name=$(grep -m1 '^# ' "$idx" | sed 's/^# //')
  echo "$remaining $tier $key $name" >> "$LIST"
done
sort -rn "$LIST" > "$LIST.sorted"

MAX="${1:-}"
count=0
while read -r remaining tier key name; do
  if [ -n "$MAX" ] && [ "$count" -ge "$MAX" ]; then break; fi
  # 该专题已有活 worker 则跳过（防止重复启动）
  if pgrep -f "run-topic-session.sh $tier $key " >/dev/null; then
    echo "skip（已在跑）: $tier/$key"
    continue
  fi
  nohup bash scripts/run-topic-session.sh "$tier" "$key" "$name" > /dev/null 2>&1 &
  echo "$tier/$key $!" >> "$PIDFILE"
  count=$((count+1))
  sleep 0.5   # 轻微错峰，避免瞬时进程风暴
done < "$LIST.sorted"
rm -f "$LIST" "$LIST.sorted"
echo "本轮已启动 $count 个专题 worker，PID 记录于 $PIDFILE"
