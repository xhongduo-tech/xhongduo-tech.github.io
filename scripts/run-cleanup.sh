#!/bin/bash
# 内容修复 worker：把指定文件里的 `undefined` 占位符替换为真实专业内容。
# 由主 session 组织批处理调用：每个 worker 处理一批文件。
# 用法：scripts/run-cleanup.sh <文件1> <文件2> ...
set -euo pipefail
LOG="logs/workers/cleanup-$(date +%s).log"
mkdir -p logs/workers

FILES=$(printf '%s\n' "$@" | tr '\n' ' ')
PROMPT=$(cat <<EOF
你是「从极限到大模型」博客的内容修复 worker。你的任务：修复以下文件里的 **\`undefined\` 占位符**。

背景：之前批量写作时，部分 worker 在需要写代码样例、特殊符号、术语名的地方输出了 \`undefined\` 占位符（LLM 常见 artifact）。你负责把它们替换成真实、专业、正确的内容。

要修复的文件（共 $# 个）：
$FILES

## 对每个文件
1. 用 Read 通读全文，理解该篇主题与语境。
2. 找出所有 \`undefined\` 占位符。
3. 逐个判断它应该是什么，并替换为正确内容：
   - **代码样例位置**（如 C++/Python/SQL/伪代码）：写出与该专题既有风格一致、语法正确、可读的代码/语句。
   - **特殊符号**（如 SMILES 的 \`>>\`、\`A*\` 搜索、数学记号）：写出正确符号（必要时用反引号包裹）。
   - **术语/节点名**（如 AST 的 \`Block\`/\`Expr\`、SQL 的 \`SELECT\`、变量名）：写出该语境下正确的术语。
   - 若 \`undefined\` 出现在反引号代码里，说明原本该处是某段代码，补一段真实代码。
4. 用 Write/Edit 写回。**保持文章其余内容、标题、marginnote、格式不变**——只修 \`undefined\`。
5. 不要动 frontmatter、index.md；不执行 git。

## 返回
逐文件报告：修复了几处 \`undefined\`、分别补成了什么类型的内容（代码/符号/术语）。若有拿不准的，按语境最合理的补全。
EOF
)

echo "== cleanup worker 启动 $(date '+%H:%M:%S')，${#} 个文件，日志：${LOG} =="
claude -p "$PROMPT" --dangerously-skip-permissions --output-format text 2>&1 | tee "$LOG"
