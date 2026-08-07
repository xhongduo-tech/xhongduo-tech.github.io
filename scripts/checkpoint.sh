#!/bin/bash
# 主控检查点：progress 重生成 → 红线扫描 → 全量构建 → 提交推送。
# 任一环节失败即中止（红线/构建不过不许提交）。
# 用法：scripts/checkpoint.sh "<commit message 前缀>"
set -euo pipefail
MSG="${1:-feat: 批量写作 checkpoint}"

echo "== 1/4 gen-progress =="
node scripts/gen-progress.mjs

echo "== 2/4 autofix（尽力清理红线共性问题）=="
node scripts/autofix.mjs

echo "== 3/4 docs:build（硬门禁，build 失败则中止不提交）=="
npm run docs:build

echo "== 4/4 commit + push =="
git add -A
git commit -m "$MSG $(date '+%Y-%m-%d %H:%M')" || echo "（无改动可提交）"
git push
echo "✅ checkpoint 完成"
