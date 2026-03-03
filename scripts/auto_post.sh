#!/usr/bin/env bash
# auto_post.sh — 按队列顺序逐篇生成博文并推送
#
# 用法：
#   bash scripts/auto_post.sh            # 运行直到队列清空
#   bash scripts/auto_post.sh --dry-run  # 预览下一篇，不实际生成
#
# 触发速率限制时自动等待 4 小时后继续。

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
SCRIPTS_DIR="$REPO_ROOT/scripts"
POSTS_DIR="$REPO_ROOT/posts"
QUEUE_FILE="$SCRIPTS_DIR/topic_queue.json"
POSTS_JSON="$POSTS_DIR/posts.json"
LOGS_DIR="$SCRIPTS_DIR/logs"
CLAUDE_BIN="/opt/homebrew/bin/claude"
RATE_LIMIT_WAIT=14400  # 4 小时（秒）

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
SYSTEM_PROMPT='你是一位专注于 AI / 大模型技术的技术博客作者。

【目标读者】
有基础编程经验（会写 Python），但刚开始接触 AI/ML 的开发者。
目标：让初学者真正看懂，同时让有经验的读者也能学到深度内容。做到"入门友好，深度不妥协"。

【写作规范】

1. 开篇先用 1-2 句话建立直觉：说清楚"这是什么、解决什么问题"，可以用类比辅助，
   但类比之后必须立刻给出精确定义，不能停留在比喻层面。
   例：KV Cache 是推理时的备忘录——避免重复计算已经处理过的 token。
       数学上，它缓存的是每层 Attention 的 K、V 矩阵：K ∈ ℝ^{t×d_k}，V ∈ ℝ^{t×d_v}。

2. 推进顺序：直觉 → 精确定义 → 数学公式 → 代码示例 → 工程细节。
   每一步承接上一步，读者能跟上推进节奏。

3. 数学公式必须解释每个符号：含义、维度、取值范围。
   不写"其中各参数含义如上"之类的省略。

4. 代码示例用 Python，注释说明关键步骤，代码应当可以直接运行或稍加修改可运行。

5. 每个 ## 大节之间用 --- 分隔。单段落不超过 5 行。

6. 禁止：感叹号、"非常重要/令人惊讶"等夸张词、空洞过渡句（"接下来我们看看……"）、
   无内容的总结段（"本文介绍了……希望对你有帮助"）。

7. 指出局限性和工程注意事项（坑点、性能 trade-off、适用范围）。

8. 文章末尾附参考资料（arXiv 链接或官方文档），不少于 2 条。

【格式要求】
- 从 ## 开始写正文，不写 # 标题（系统会自动渲染标题）
- 数学公式：行内 $...$，块级 $$...$$
- 代码块注明语言：```python
- 目标长度：2000-4000 字，内容为主，不凑字数

【输出格式】
正文结束后另起一行输出（不放在代码块内）：
{"summary": "一两句摘要，不超过60字，说明文章核心内容"}'

# ── 生成文章 ──────────────────────────────────────────────────────────────────
# 返回值：0=成功，1=失败，2=速率限制
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

    # 检测速率限制
    if [[ "$exit_code" -ne 0 ]] && \
       echo "$stderr_content" | grep -qiE "rate.limit|too many|429|quota|overload|exceeded|please wait"; then
        rm -f "$tmp_out"
        return 2
    fi

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
            '{title:$title,slug:$slug,date:$date,author:"claude",tags:$tags,summary:$summary}')
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

# ── 更新队列状态 ───────────────────────────────────────────────────────────────
mark_queue() {
    local slug="$1" status="$2"
    jq --arg slug "$slug" --arg s "$status" --arg ts "$(date -u +%Y-%m-%dT%H:%M:%S)" \
        'map(if .slug == $slug then .status = $s | .done_at = $ts else . end)' \
        "$QUEUE_FILE" > /tmp/queue_tmp_$$ && mv /tmp/queue_tmp_$$ "$QUEUE_FILE"
}

# ── Dry-run ───────────────────────────────────────────────────────────────────
if $DRY_RUN; then
    next=$(jq 'first(.[] | select(.status == "pending")) // empty' "$QUEUE_FILE")
    if [[ -z "$next" ]]; then
        echo "队列为空，所有主题已完成。"
    else
        echo "下一篇将生成："
        echo "$next" | jq -r '"  标题：\(.title)\n  slug： \(.slug)\n  分类：\(.tags[0])\n  brief：\(.brief[:100])..."'
    fi
    pending_count=$(jq '[.[] | select(.status == "pending")] | length' "$QUEUE_FILE")
    done_count=$(jq '[.[] | select(.status == "done")] | length' "$QUEUE_FILE")
    echo ""
    echo "  待生成：${pending_count}  已完成：${done_count}"
    exit 0
fi

# ── 主循环 ────────────────────────────────────────────────────────────────────
log "=== auto_post.sh 启动，逐篇生成直到队列清空 ==="

while true; do
    # 取第一个 pending 主题
    topic=$(jq 'first(.[] | select(.status == "pending")) // empty' "$QUEUE_FILE")

    if [[ -z "$topic" ]]; then
        log "所有主题已生成完毕！"
        break
    fi

    slug=$(      jq -r '.slug'                                                  <<< "$topic")
    title=$(     jq -r '.title'                                                 <<< "$topic")
    brief=$(     jq -r '.brief'                                                 <<< "$topic")
    depth_hint=$(jq -r '.depth_hint // "从基础概念到工程实践，包含数学推导和可运行代码"' <<< "$topic")
    tags_json=$( jq    '.tags'                                                  <<< "$topic")

    log "──────────────────────────────────────────"
    log "生成中：${title}"
    log "slug  ：${slug}"

    output=""
    rc=0
    output=$(generate_article "$title" "$brief" "$depth_hint") || rc=$?

    if [[ "$rc" -ne 0 ]]; then
        err "生成失败（退出码 ${rc}），标记回 pending 并停止：${title}"
        mark_queue "$slug" "pending"
        exit 1
    fi

    publish_article "$slug" "$title" "$tags_json" "$output"
    mark_queue "$slug" "done"
    git_commit_push "$slug"

    # 避免请求过于密集
    sleep 3
done

log "=== 全部完成 ==="
