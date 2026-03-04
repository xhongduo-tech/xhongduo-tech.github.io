#!/usr/bin/env bash
# auto_post.sh — 每次随机取 10 篇，生成后逐篇推送
#
# 用法：
#   bash scripts/auto_post.sh            # 随机取 10 篇生成并推送
#   bash scripts/auto_post.sh --dry-run  # 预览将生成的 10 篇，不实际生成

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
SCRIPTS_DIR="$REPO_ROOT/scripts"
POSTS_DIR="$REPO_ROOT/posts"
QUEUE_FILE="$SCRIPTS_DIR/topic_queue.json"
POSTS_JSON="$POSTS_DIR/posts.json"
LOGS_DIR="$SCRIPTS_DIR/logs"
CLAUDE_BIN="/opt/homebrew/bin/claude"
BATCH_SIZE=10

mkdir -p "$LOGS_DIR"
LOG_FILE="$LOGS_DIR/auto_post_$(date +%Y%m%d_%H%M%S).log"

# ── 日志 ──────────────────────────────────────────────────────────────────────
log()  { echo "$(date '+%Y-%m-%d %H:%M:%S') [INFO]  $*" | tee -a "$LOG_FILE"; }
err()  { echo "$(date '+%Y-%m-%d %H:%M:%S') [ERROR] $*" | tee -a "$LOG_FILE" >&2; }
warn() { echo "$(date '+%Y-%m-%d %H:%M:%S') [WARN]  $*" | tee -a "$LOG_FILE"; }

# ── 参数 ──────────────────────────────────────────────────────────────────────
DRY_RUN=false
[[ "${1:-}" == "--dry-run" ]] && DRY_RUN=true

# ── 系统提示 ───────────────────────────────────────────────────────────────────
SYSTEM_PROMPT='你是一位顶级 AI 技术博客作者，目标读者是具备 Python 和深度学习基础的工程师。

核心原则：**Show, don'\''t tell** — 用代码、数值、公式证明观点，而非泛泛而谈。

═══════════════════════════════════════════════════════════════════════════════
【文章结构模板】（必须严格遵循）
═══════════════════════════════════════════════════════════════════════════════

## 概念定义
精确定义 + 1 句直觉解释 + 为什么这个问题重要

---

## 数学原理
- 从第一性原理出发，完整推导
- **每个公式后必须有具体数值示例**：代入小数值（如 n=3, d=64）展示计算过程
- 标注所有符号的含义、维度、取值范围

---

## 代码实现
```python
# 必须满足：
# 1. 完整可运行（复制粘贴即可执行）
# 2. 包含 assert 验证输出正确性
# 3. 打印中间结果展示计算过程
# 4. 不用 placeholder 或 "..."
```

---

## 工程细节与性能分析
- 时间复杂度（具体到操作次数）
- 空间复杂度（具体到字节数，如 "缓存 n×d×4 字节"）
- 实际性能瓶颈在哪
- 常见坑点与调优技巧

---

## 局限性与替代方案
- 这个方法不能解决什么问题
- 什么时候不该用它
- 替代方案对比（表格呈现）

---

## 参考资料
1. [标题](链接) - 一句话说明为什么推荐
2. ...
（至少 3 条高质量参考）

═══════════════════════════════════════════════════════════════════════════════
【强制要求】
═══════════════════════════════════════════════════════════════════════════════

✓ 代码：完整、可运行、有输出、有验证
✓ 数值：每个数学概念都用具体数字演示
✓ 维度：每个张量标注 shape，如 x.shape = (batch, seq, dim)
✓ 对比：用表格比较不同方法的优劣
✓ 参考：arXiv 论文、官方文档、权威博客

✗ 禁止：
  - 感叹号、夸张形容词
  - "接下来我们看看" 等过渡句
  - 空洞总结 "希望对你有帮助"
  - 代码片段用 "..." 或 "此处省略"
  - 只讲概念不给数值示例
  - 调包代码（如 `nn.Linear`）而不展示内部实现

═══════════════════════════════════════════════════════════════════════════════
【格式规范】
═══════════════════════════════════════════════════════════════════════════════

- 从 ## 开始，不写 # 标题
- 数学：行内 $...$，块级 $$...$$
- 代码：```python 并注明语言
- 章节间用 --- 分隔
- 段落 ≤ 5 行
- 文末单独一行输出：{"summary": "不超过60字的核心摘要"}'

# ── 按技术依赖顺序取 N 个 pending 主题的 slug 列表 ────────────────────────────
# 队列 topic_queue.json 已按技术依赖关系由浅入深排序（ID 越小越基础）。
# 策略：从最前面的 pending 主题中取前 N 个，确保基础技术优先覆盖。
pick_random_slugs() {
    python3 - "$QUEUE_FILE" "$BATCH_SIZE" << 'PYEOF'
import json, sys
with open(sys.argv[1]) as f:
    data = json.load(f)
# 保留原始顺序（已按技术依赖排列），只取前 N 个 pending
pending = [d['slug'] for d in data if d.get('status') == 'pending']
for s in pending[:int(sys.argv[2])]:
    print(s)
PYEOF
}

# ── 根据 slug 获取完整 topic JSON ──────────────────────────────────────────────
get_topic() {
    local slug="$1"
    jq --arg s "$slug" 'first(.[] | select(.slug == $s))' "$QUEUE_FILE"
}

# ── 生成文章 ──────────────────────────────────────────────────────────────────
# 返回值：0=成功，非0=失败
generate_article() {
    local title="$1" brief="$2" depth_hint="$3"
    local tmp_out tmp_err
    tmp_out=$(mktemp /tmp/auto_post_out_XXXXXX)
    tmp_err=$(mktemp /tmp/auto_post_err_XXXXXX)

    local prompt="${SYSTEM_PROMPT}

请写一篇技术博文，主题如下：

标题：${title}
核心要点：${brief}
深度方向：${depth_hint}

请严格遵循写作规范，在文章末尾单独输出 summary JSON。"

    unset CLAUDECODE CLAUDE_CODE_ENTRYPOINT 2>/dev/null || true

    local exit_code=0
    "$CLAUDE_BIN" -p "$prompt" --model claude-opus-4-6 \
        > "$tmp_out" 2>"$tmp_err" || exit_code=$?

    local stderr_content
    stderr_content=$(cat "$tmp_err")
    rm -f "$tmp_err"

    if [[ "$exit_code" -ne 0 ]]; then
        err "  claude 退出码 ${exit_code}: $(echo "$stderr_content" | head -3)"
        rm -f "$tmp_out"
        return 1
    fi

    cat "$tmp_out"
    rm -f "$tmp_out"
}

# ── 写入文件 + 更新 posts.json ─────────────────────────────────────────────────
publish_article() {
    local slug="$1" title="$2" tags_json="$3" output="$4"

    # 提取 summary（最后一行 {"summary":...}）
    local summary
    summary=$(printf '%s' "$output" \
        | grep -o '{"summary":"[^"]*"}' | tail -1 \
        | jq -r '.summary // empty' 2>/dev/null || true)
    [[ -z "$summary" ]] && summary="$title"

    # 去掉 summary 行，保留正文
    local content
    content=$(printf '%s' "$output" | grep -v '^{"summary":')

    # 写 .md
    printf '%s\n' "$content" > "$POSTS_DIR/${slug}.md"

    # 更新 posts.json（若 slug 已存在则跳过）
    local exists
    exists=$(jq --arg s "$slug" '[.[] | select(.slug==$s)] | length' "$POSTS_JSON")
    if [[ "$exists" -eq 0 ]]; then
        local entry
        entry=$(jq -n \
            --arg title   "$title" \
            --arg slug    "$slug" \
            --arg date    "$(date +%Y-%m-%d)" \
            --arg summary "$summary" \
            --argjson tags "$tags_json" \
            '{title:$title,slug:$slug,date:$date,author:"both",tags:$tags,summary:$summary}')
        jq --argjson e "$entry" '. = [$e] + .' "$POSTS_JSON" \
            > /tmp/posts_tmp_$$ && mv /tmp/posts_tmp_$$ "$POSTS_JSON"
    else
        warn "  posts.json 中 ${slug} 已存在，跳过插入"
    fi
}

# ── Git 提交推送 ───────────────────────────────────────────────────────────────
git_commit_push() {
    local slug="$1"
    cd "$REPO_ROOT"
    git add "posts/${slug}.md" posts/posts.json
    if git diff --cached --quiet; then
        warn "  无暂存变更，跳过 commit"
        return 0
    fi
    git commit -m "$(printf 'post: %s\n\nCo-Authored-By: Claude Opus 4.6 <noreply@anthropic.com>' "$slug")"
    git push origin main
    log "  ✓ 已推送：${slug}"
}

# ── 更新队列状态（无时间戳）──────────────────────────────────────────────────
mark_queue() {
    local slug="$1" status="$2"
    jq --arg slug "$slug" --arg s "$status" \
        'map(if .slug == $slug then .status = $s else . end)' \
        "$QUEUE_FILE" > /tmp/queue_tmp_$$ && mv /tmp/queue_tmp_$$ "$QUEUE_FILE"
}

# ── Dry-run ───────────────────────────────────────────────────────────────────
if $DRY_RUN; then
    echo "本次将随机生成以下 ${BATCH_SIZE} 篇："
    echo ""
    slugs=()
    while IFS= read -r line; do slugs+=("$line"); done < <(pick_random_slugs)
    for slug in "${slugs[@]}"; do
        topic=$(get_topic "$slug")
        echo "$topic" | jq -r '"  [\(.tags[0])] \(.title)\n  slug: \(.slug)\n"'
    done
    pending_count=$(jq '[.[] | select(.status == "pending")] | length' "$QUEUE_FILE")
    done_count=$(jq '[.[] | select(.status == "done")] | length' "$QUEUE_FILE")
    echo "  待生成：${pending_count}  已完成：${done_count}"
    exit 0
fi

# ── 主逻辑 ────────────────────────────────────────────────────────────────────
log "=== auto_post.sh 启动，随机取 ${BATCH_SIZE} 篇生成 ==="

# 取本批次 slug 列表
slugs=()
while IFS= read -r line; do slugs+=("$line"); done < <(pick_random_slugs)

if [[ "${#slugs[@]}" -eq 0 ]]; then
    log "队列为空，所有主题已完成。"
    exit 0
fi

log "本批次共 ${#slugs[@]} 篇：$(IFS=', '; echo "${slugs[*]}")"

for slug in "${slugs[@]}"; do
    topic=$(get_topic "$slug")
    title=$(     jq -r '.title'                                                  <<< "$topic")
    brief=$(     jq -r '.brief'                                                  <<< "$topic")
    depth_hint=$(jq -r '.depth_hint // "从基础原理到工程实现，包含完整数学推导和可运行代码"' <<< "$topic")
    tags_json=$( jq    '.tags'                                                   <<< "$topic")

    log "──────────────────────────────────────────"
    log "生成中：${title}"
    log "slug  ：${slug}"

    output=""
    rc=0
    output=$(generate_article "$title" "$brief" "$depth_hint") || rc=$?

    if [[ "$rc" -ne 0 ]]; then
        err "生成失败，立即停止。已完成主题已标记为 done。"
        exit 1
    fi

    publish_article "$slug" "$title" "$tags_json" "$output"
    mark_queue "$slug" "done"
    git_commit_push "$slug"

    sleep 3
done

log "=== 本批次 ${#slugs[@]} 篇全部完成 ==="
