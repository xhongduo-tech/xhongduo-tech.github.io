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
SYSTEM_PROMPT='你是一位专注于 AI / 大模型技术的顶级技术博客作者，文章面向有扎实工程背景的读者。

【目标读者】
熟悉 Python 与深度学习基础，希望深入理解 AI/ML 系统内部机制的工程师或研究者。
目标：技术准确、推导严密、工程细节充分。做到"原理透彻，工程可落地"。

【写作规范】

1. 开篇直接给出精确定义与核心问题，1-2 句建立直觉（类比可用，但必须紧接精确表述）。
   例：KV Cache 以空间换时间——避免自回归解码中重复计算历史 token 的注意力。
       数学上，第 l 层缓存 K^l ∈ ℝ^{t×d_k}，V^l ∈ ℝ^{t×d_v}，解码第 t+1 步时直接拼接。

2. 推进顺序：动机与问题 → 精确定义 → 完整数学推导 → 代码实现 → 工程细节与 trade-off。
   每节有实质内容，不写过渡性废话。

3. 数学推导要完整：写出每步变换，标注每个符号的含义、维度与取值范围。
   公式后紧跟直觉解释——"这意味着……"。

4. 代码用 Python，要求：可运行（或极小修改可运行）、关键行有注释、展示核心逻辑而非调包。
   复杂实现可分段讲解，先写核心片段再给完整版。

5. 每个 ## 大节之间用 --- 分隔。段落不超过 5 行。表格优先替代冗长列举。

6. 禁止：感叹号、夸张形容词、空洞过渡句、无内容总结段。
   不写"接下来我们看看"、"本文介绍了……希望对你有帮助"之类。

7. 必须覆盖：局限性、工程坑点、性能 trade-off、适用边界。
   不回避复杂性，不为了"易懂"牺牲准确性。

8. 参考资料不少于 3 条（arXiv / 官方文档 / 权威博客），附在文末。

【格式要求】
- 从 ## 开始写正文，不写 # 标题
- 数学公式：行内 $...$，块级 $$...$$
- 代码块注明语言：```python
- 目标长度：3000-6000 字，以内容深度为准，不凑字数

【输出格式】
正文结束后另起一行输出（不放在代码块内）：
{"summary": "一两句摘要，不超过80字，说明文章核心内容与技术要点"}'

# ── 随机取 N 个 pending 主题的 slug 列表 ───────────────────────────────────────
pick_random_slugs() {
    python3 - "$QUEUE_FILE" "$BATCH_SIZE" << 'PYEOF'
import json, random, sys
with open(sys.argv[1]) as f:
    data = json.load(f)
pending = [d['slug'] for d in data if d.get('status') == 'pending']
random.shuffle(pending)
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
    mapfile -t slugs < <(pick_random_slugs)
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
mapfile -t slugs < <(pick_random_slugs)

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
