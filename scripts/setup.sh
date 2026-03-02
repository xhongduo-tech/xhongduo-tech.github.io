#!/bin/bash
# setup.sh — 一键安装自动发布系统
# 用法：bash scripts/setup.sh

set -e
REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
SCRIPTS_DIR="$REPO_ROOT/scripts"
LAUNCH_AGENTS_DIR="$HOME/Library/LaunchAgents"

echo "=== 自动发布系统安装 ==="

# 1. 检查 claude CLI
echo "→ 检查 Claude Code CLI..."
CLAUDE_BIN="$HOME/.local/bin/claude"
if [ ! -x "$CLAUDE_BIN" ]; then
  echo "  ✗ 未找到 claude CLI（$CLAUDE_BIN）"
  echo "  请先安装 Claude Code：https://claude.ai/code"
  exit 1
fi
CLAUDE_VERSION=$("$CLAUDE_BIN" --version 2>/dev/null || echo "unknown")
echo "  ✓ claude 已就绪：$CLAUDE_VERSION"

# 2. 创建日志目录
mkdir -p "$SCRIPTS_DIR/logs"
echo "→ 日志目录已创建：scripts/logs/"

# 3. 注册 launchd 任务
echo "→ 注册定时任务..."
mkdir -p "$LAUNCH_AGENTS_DIR"

# 注销旧任务（1am/7am）
for OLD in 1am 7am; do
  OLD_DEST="$LAUNCH_AGENTS_DIR/com.xhd.blog.autopost-${OLD}.plist"
  if [ -f "$OLD_DEST" ]; then
    launchctl unload "$OLD_DEST" 2>/dev/null || true
    rm -f "$OLD_DEST"
    echo "  已移除旧任务：${OLD}"
  fi
done

# 注册新的 2am 任务
PLIST_SRC="$SCRIPTS_DIR/com.xhd.blog.autopost-2am.plist"
PLIST_DEST="$LAUNCH_AGENTS_DIR/com.xhd.blog.autopost-2am.plist"
cp "$PLIST_SRC" "$PLIST_DEST"
launchctl unload "$PLIST_DEST" 2>/dev/null || true
launchctl load "$PLIST_DEST"
echo "  已注册：2am 任务（每天凌晨 2:00，生成 6 篇）"

echo ""
echo "=== 安装完成 ==="
echo ""
echo "定时任务："
echo "  凌晨 2:00 → 生成 6 篇文章并推送"
echo ""
echo "常用命令："
echo "  预览下 3 篇主题：python3 scripts/auto_post.py --dry-run"
echo "  立即生成 6 篇：  python3 scripts/auto_post.py --count 6"
echo "  重试失败推送：   python3 scripts/auto_post.py --retry"
echo "  查看日志：       tail -f scripts/logs/auto_post_$(date +%Y%m%d).log"
echo "  查看队列状态：   python3 -c \\"
echo "    import json; q=json.load(open('scripts/topic_queue.json'));"
echo "    done=sum(1 for t in q if t['status']=='done');"
echo "    pending=sum(1 for t in q if t['status']=='pending');"
echo "    print(f'已完成 {done}，待发布 {pending}')"
