#!/bin/bash
# setup.sh — 一键安装自动发布系统
# 用法：bash scripts/setup.sh

set -e
REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
SCRIPTS_DIR="$REPO_ROOT/scripts"
LAUNCH_AGENTS_DIR="$HOME/Library/LaunchAgents"

echo "=== 自动发布系统安装 ==="

# 1. 安装 Python 依赖
echo "→ 安装 Python 依赖..."
pip3 install anthropic --quiet
echo "  anthropic 已安装"

# 2. 配置 .env
if [ ! -f "$SCRIPTS_DIR/.env" ]; then
  cp "$SCRIPTS_DIR/.env.template" "$SCRIPTS_DIR/.env"
  echo ""
  echo "  ⚠️  请编辑 scripts/.env，填入你的 ANTHROPIC_API_KEY："
  echo "     open $SCRIPTS_DIR/.env"
  echo ""
  read -p "  填好后按回车继续..." _
fi

# 3. 创建日志目录
mkdir -p "$SCRIPTS_DIR/logs"
echo "→ 日志目录已创建：scripts/logs/"

# 4. 注册 launchd 任务
echo "→ 注册定时任务..."
mkdir -p "$LAUNCH_AGENTS_DIR"

for PLIST in 1am 7am; do
  SRC="$SCRIPTS_DIR/com.xhd.blog.autopost-${PLIST}.plist"
  DEST="$LAUNCH_AGENTS_DIR/com.xhd.blog.autopost-${PLIST}.plist"
  cp "$SRC" "$DEST"
  launchctl unload "$DEST" 2>/dev/null || true
  launchctl load "$DEST"
  echo "  已注册：凌晨 1:00 任务（$PLIST）"
done

echo ""
echo "=== 安装完成 ==="
echo ""
echo "定时任务："
echo "  凌晨 1:00 → 生成 3 篇文章并推送"
echo "  早上 7:00 → 生成 3 篇文章并推送"
echo ""
echo "常用命令："
echo "  预览下 3 篇主题：python3 scripts/auto_post.py --dry-run"
echo "  立即生成 1 篇：  python3 scripts/auto_post.py --count 1"
echo "  重试失败推送：   python3 scripts/auto_post.py --retry"
echo "  查看日志：       tail -f scripts/logs/auto_post_$(date +%Y%m%d).log"
echo "  查看队列状态：   python3 -c \\"
echo "    import json; q=json.load(open('scripts/topic_queue.json'));"
echo "    done=sum(1 for t in q if t['status']=='done');"
echo "    pending=sum(1 for t in q if t['status']=='pending');"
echo "    print(f'已完成 {done}，待发布 {pending}')"
