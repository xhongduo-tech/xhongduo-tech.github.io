#!/bin/bash
# 启动一个「一专题一 session」的独立写手会话（headless claude -p）。
# 由主 session（master）调用：每个 worker 负责一个专题，写到该专题无 `- [ ]` 为止（或达到 MAX_POSTS 上限）。
#
# 用法：
#   scripts/run-topic-session.sh <tier> <key> <名称> [MAX_POSTS]
# 例：
#   scripts/run-topic-session.sh foundations physics 基础物理 8     # 试点：只写 8 篇
#   scripts/run-topic-session.sh foundations chemistry 化学        # 完整：写到专题写完
#
# 输出：日志写入 logs/workers/<tier>-<key>.log；stdout 打印该 worker 的报告（含剩余条目数）。

set -euo pipefail
TIER="$1"; KEY="$2"; NAME="$3"; MAX_POSTS="${4:-}"
LOG="logs/workers/${TIER}-${KEY}.log"
mkdir -p logs/workers

if [ -n "$MAX_POSTS" ]; then
  BOUND="本轮任务：连续撰写本专题接下来最多 ${MAX_POSTS} 篇仍为 \`- [ ]\` 的条目（按 index.md 出现顺序），
写满 ${MAX_POSTS} 篇或条目耗尽即停止，返回简短报告。"
else
  BOUND="本轮任务：连续撰写本专题**所有**仍为 \`- [ ]\` 的条目，直到 index.md 里没有任何 \`- [ ]\` 为止。
上下文接近上限时靠自动压缩续写；若自动压缩后仍无法继续，就在安全断点停下并如实报告剩余数。"
fi

PROMPT=$(cat <<EOF
你是「从极限到大模型」博客写作流水线的独立写手 session。本 session 只负责一个专题：${NAME}（${TIER}/${KEY}）。

## 一次性设置（按序通读，勿重复读）
1. .claude/writing-charter.md（编辑章程，最高约束）
2. .claude/agents/${TIER}-${KEY}.md（小组简报）
3. docs/posts/foundations/math/set-concept.md（风格范本，只读一次）
4. docs/posts/${TIER}/${KEY}/index.md（选题规划，写作中随勾选更新）

## ${BOUND}

## 逐篇写作要求
- 每篇写入 docs/posts/${TIER}/${KEY}/<slug>.md（slug 用英文 kebab-case，与条目标题对应）。
- 写完一篇立即：把 index.md 中该条目改为 \`- [x] [<标题>](./<slug>)\`，然后写下一篇（保持上下文，勿重读设置文件）。
- frontmatter date: 2026-08-07；byline「第X级 · ${NAME} ｜ <教材> <章节> ｜ 2026-08-07」。
- 严格遵循编辑章程：结构模板、公式解析（有公式的主题）、重点加粗、易错辨析、≥2 条 marginnote、小结、结尾引子。
- 配图吞吐优先：仅当示意图显著提升理解才配 SVG（≤1/篇，遵循章程第4节）。
- 每篇 ≥120 行、≥3 编号分节；纯概念主题用核心对比表替代公式解析并标注。
- **marginnote/sidenote 里禁止 markdown \`**\`**：需要强调一律用 \`<strong>…</strong>\` 包裹。
- **表格行内代码里的管道要写 \`\\|\`**；裸记号（如 \`<eos>\`、\`<where>\`、\`<digit>\`）必须用反引号包裹。
- **HTML span 必须成对闭合**：每开一个 \`<span …>\` 就配一个 \`</span>\`；禁止游离 \`</span>\`、\`</p>\`；换行用 \`<br/>\` 而非 \`</br>\`。写完后自查一遍再交。

## 红线（违反即失败）
- 不改 docs/.vitepress/data/progress.json。
- 不执行任何 git 命令、不 commit、不 push。
- 不启动子代理/多 agent；就你一个写手。
- 不重读设置文件（只读一次上面 4 个）。

## 返回
结束时返回简短报告：本次写了 N 篇（列出标题/slug/行数），剩余条目数，以及你停下的原因（写完 / 到达上限 / 上下文受限）。
EOF
)

echo "== worker ${TIER}/${KEY} (${NAME}) 启动 $(date '+%H:%M:%S')，日志：${LOG} =="
claude -p "$PROMPT" --dangerously-skip-permissions --output-format text 2>&1 | tee "$LOG"
