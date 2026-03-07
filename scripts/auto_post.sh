#!/usr/bin/env bash
# auto_post.sh — 每次按顺序取前 N 篇 pending，Codex 生成并校验改写后推送
#
# 用法：
#   bash scripts/auto_post.sh            # 按顺序取前 10 篇，Codex 生成并校验改写后推送
#   BATCH_SIZE=1 bash scripts/auto_post.sh
#   bash scripts/auto_post.sh --dry-run  # 预览将生成的前 10 篇，不实际生成

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
SCRIPTS_DIR="$REPO_ROOT/scripts"
POSTS_DIR="$REPO_ROOT/posts"
QUEUE_FILE="$SCRIPTS_DIR/topic_queue.json"
POSTS_JSON="$POSTS_DIR/posts.json"
LOGS_DIR="$SCRIPTS_DIR/logs"
FAILED_OUTPUT_DIR="$POSTS_DIR/_failed"
CODEX_BIN="${CODEX_BIN:-$(command -v codex 2>/dev/null || true)}"
[[ -z "$CODEX_BIN" ]] && CODEX_BIN="/Applications/Codex.app/Contents/Resources/codex"
# 成本优化默认：草稿走 mini，校验改写走 5.4（可用环境变量覆盖）
CODEX_DRAFT_MODEL="${CODEX_DRAFT_MODEL:-${CODEX_MODEL:-gpt-5-codex-mini}}"
CODEX_DRAFT_REASONING_EFFORT="${CODEX_DRAFT_REASONING_EFFORT:-${CODEX_REASONING_EFFORT:-medium}}"
CODEX_REFINE_MODEL="${CODEX_REFINE_MODEL:-${CODEX_MODEL:-gpt-5.4}}"
CODEX_REFINE_REASONING_EFFORT="${CODEX_REFINE_REASONING_EFFORT:-${CODEX_REASONING_EFFORT:-medium}}"
BATCH_SIZE="${BATCH_SIZE:-10}"

mkdir -p "$LOGS_DIR" "$FAILED_OUTPUT_DIR"
LOG_FILE="$LOGS_DIR/auto_post_$(date +%Y%m%d_%H%M%S).log"
RUN_DRAFT_DIR="$(mktemp -d /tmp/auto_post_drafts_XXXXXX)"
trap 'rm -rf "$RUN_DRAFT_DIR"' EXIT

# ── 日志 ──────────────────────────────────────────────────────────────────────
log()  { echo "$(date '+%Y-%m-%d %H:%M:%S') [INFO]  $*" | tee -a "$LOG_FILE"; }
err()  { echo "$(date '+%Y-%m-%d %H:%M:%S') [ERROR] $*" | tee -a "$LOG_FILE" >&2; }
warn() { echo "$(date '+%Y-%m-%d %H:%M:%S') [WARN]  $*" | tee -a "$LOG_FILE"; }

# ── 参数 ──────────────────────────────────────────────────────────────────────
DRY_RUN=false
[[ "${1:-}" == "--dry-run" ]] && DRY_RUN=true

# ── 系统提示 ───────────────────────────────────────────────────────────────────
SYSTEM_PROMPT=$(cat <<'EOF'
你是一位面向资深工程师的 AI 技术博客作者。输出要可复核、可运行、可比较，避免空话。

只输出文章正文，不要输出任何前言、说明、道歉、执行过程、权限/文件/工具/网络状态描述。

严格使用以下章节顺序：
## 核心结论
## 问题定义与边界
## 核心机制与推导
## 代码实现
## 工程权衡与常见坑
## 替代方案与适用边界
## 参考资料

要求：
- 从 `##` 开始，不写 `#`
- 章节之间使用 `---`
- 至少包含一个可运行的 `python` 代码块，代码里有 `assert` 或明确校验输出
- 至少包含一个 Markdown 表格
- 至少包含一个带具体数字的推演或容量估算
- 使用统一的 running example 贯穿机制、代码和工程分析
- 优先引用论文、官方文档、源码或高质量技术博客
- 文末最后一行单独输出：{"summary":"不超过60字的核心摘要"}
EOF
)

# ── 按技术依赖顺序取 N 个 pending 主题的 slug 列表 ────────────────────────────
# 队列 topic_queue.json 已按技术依赖关系由浅入深排序（ID 越小越基础）。
# 策略：从最前面的 pending 主题中取前 N 个，确保基础技术优先覆盖。
pick_next_slugs() {
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

# ── 保存失败稿，避免内容丢失 ──────────────────────────────────────────────────
save_failed_article() {
    local slug="$1" stage="$2" output="$3"
    local failed_file="$FAILED_OUTPUT_DIR/${slug}.${stage}.md"

    [[ -z "${output//[[:space:]]/}" ]] && return 0

    printf '%s\n' "$output" > "$failed_file"
    warn "  已保存失败稿：$failed_file"
}

draft_file_for_slug() {
    local slug="$1"
    printf '%s/%s.md' "$RUN_DRAFT_DIR" "$slug"
}

render_progress() {
    local current="$1" total="$2" label="$3"
    local width=24 filled empty bar
    filled=$(( current * width / total ))
    empty=$(( width - filled ))
    printf -v bar '%*s' "$filled" ''
    bar=${bar// /#}
    printf -v pad '%*s' "$empty" ''
    pad=${pad// /-}
    printf '\r进度 [%s%s] %d/%d %s' "$bar" "$pad" "$current" "$total" "$label"
    if [[ "$current" -ge "$total" ]]; then
        printf '\n'
    fi
}

# ── 根据 slug 获取完整 topic JSON ──────────────────────────────────────────────
get_topic() {
    local slug="$1"
    jq --arg s "$slug" 'first(.[] | select(.slug == $s))' "$QUEUE_FILE"
}

# ── 解析前置阅读标题 ────────────────────────────────────────────────────────────
resolve_prereq_titles() {
    local topic_json="$1"
    local prereq_json
    prereq_json=$(jq '.prerequisites // []' <<< "$topic_json")
    jq -r --argjson req "$prereq_json" '
        if ($req | length) == 0 then
            ""
        else
            [.[] | select(.slug as $slug | $req | index($slug)) | .title] | join(" / ")
        end
    ' "$QUEUE_FILE"
}

# ── 调用 Codex 生成正文 ────────────────────────────────────────────────────────
run_codex_prompt() {
    local prompt="$1"
    local tmp_out tmp_err
    tmp_out=$(mktemp /tmp/auto_post_out_XXXXXX)
    tmp_err=$(mktemp /tmp/auto_post_err_XXXXXX)

    if [[ ! -x "$CODEX_BIN" ]]; then
        err "未找到可执行的 codex CLI: $CODEX_BIN"
        rm -f "$tmp_out" "$tmp_err"
        return 1
    fi

    local exit_code=0
    "$CODEX_BIN" exec \
        --model "$CODEX_DRAFT_MODEL" \
        -c "model_reasoning_effort=\"${CODEX_DRAFT_REASONING_EFFORT}\"" \
        --skip-git-repo-check \
        --color never \
        -s workspace-write \
        --output-last-message "$tmp_out" \
        -C "$REPO_ROOT" \
        "$prompt" \
        > /dev/null 2>"$tmp_err" || exit_code=$?

    local stderr_content
    stderr_content=$(cat "$tmp_err")
    rm -f "$tmp_err"

    if [[ "$exit_code" -ne 0 ]]; then
        if grep -Eiq 'permission|readonly|Operation not permitted|websocket|shell_snapshot|state_db' <<< "$stderr_content"; then
            err "  codex 调用失败（环境噪声已省略），请检查 codex 登录状态或网络连接。"
        else
            err "  codex 退出码 ${exit_code}: $(echo "$stderr_content" | head -3)"
        fi
        rm -f "$tmp_out"
        return 1
    fi

    cat "$tmp_out"
    rm -f "$tmp_out"
}

# ── 生成文章 ──────────────────────────────────────────────────────────────────
generate_article() {
    local article_context="$1"
    local prompt="${SYSTEM_PROMPT}

请直接写出合格终稿，写作上下文如下：

${article_context}

额外要求：
- 默认读者已经掌握“前置阅读”里的内容，不要重复讲基础定义；只在必要处用 1-2 句承接
- 用同一个 running example 贯穿公式、代码和工程分析
- 如果是工程或部署主题，不要伪造学术化推导；请用复杂度、容量、调度或协议模型解释
- 如果是理论、训练或微调主题，公式必须可推导、符号必须可复核
- 如果工具、文件、权限、网络有任何异常，都不要写进正文
- 最后一行必须单独输出 summary JSON
"

    run_codex_prompt "$prompt"
}

sanitize_article_output() {
    local article_file="$1"
    python3 - "$article_file" <<'PYEOF'
from pathlib import Path
import re
import sys

path = Path(sys.argv[1])
text = path.read_text()
lines = text.splitlines()

start = None
for idx, line in enumerate(lines):
    if line.strip().startswith("## "):
        start = idx
        break

if start is not None:
    lines = lines[start:]

noise_patterns = [
    r"看起来.*权限.*",
    r".*权限被拒绝.*",
    r".*文件读取.*",
    r".*无法读取文件.*",
    r".*tool.*failed.*",
    r".*network.*error.*",
]

cleaned = []
for line in lines:
    stripped = line.strip()
    if stripped and any(re.search(pattern, stripped, flags=re.I) for pattern in noise_patterns):
        continue
    cleaned.append(line)

path.write_text("\n".join(cleaned).strip() + "\n")
PYEOF
}

stage_draft_article() {
    local slug="$1" output="$2"
    local draft_file
    draft_file=$(draft_file_for_slug "$slug")
    printf '%s\n' "$output" > "$draft_file"
    sanitize_article_output "$draft_file"
}

# ── 质量校验 ──────────────────────────────────────────────────────────────────
validate_article_output() {
    local article_file="$1" require_summary="${2:-1}"
    python3 - "$article_file" "$require_summary" <<'PYEOF'
import json, re, sys
from pathlib import Path

text = Path(sys.argv[1]).read_text()
require_summary = sys.argv[2] == "1"
issues = []
required_sections = [
    "## 核心结论",
    "## 问题定义与边界",
    "## 核心机制与推导",
    "## 代码实现",
    "## 工程权衡与常见坑",
    "## 替代方案与适用边界",
    "## 参考资料",
]

for section in required_sections:
    if section not in text:
        issues.append(f"缺少章节：{section}")

if "```python" not in text:
    issues.append("缺少可运行的 python 代码块")
if "|" not in text:
    issues.append("缺少 Markdown 表格")
if "$$" not in text and not re.search(r"\$[^$\n]+\$", text):
    issues.append("缺少公式或数学记号")

refs_body = ""
refs_match = re.search(r"^## 参考资料\s*$([\s\S]*)", text, flags=re.M)
if refs_match:
    refs_body = refs_match.group(1)
    refs_lines = []
    for raw_line in refs_body.splitlines():
        line = raw_line.strip()
        if line.startswith('{"summary"'):
            break
        refs_lines.append(raw_line)
    refs_body = "\n".join(refs_lines).strip()

list_items = re.findall(r"^\s*(?:\d+\.\s+|[-*]\s+).+", refs_body, flags=re.M)
markdown_links = re.findall(r"\[[^\]]+\]\((https?://[^)]+)\)", refs_body)
bare_urls = re.findall(r"https?://\S+", refs_body)
ref_count = max(len(list_items), len(set(markdown_links)), len(set(bare_urls)))
if ref_count < 3:
    issues.append("参考资料少于 3 条")

if require_summary:
    summary_ok = False
    for line in reversed(text.splitlines()):
        line = line.strip()
        if not line:
            continue
        if not line.startswith("{"):
            break
        try:
            obj = json.loads(line)
        except Exception:
            break
        summary = obj.get("summary", "")
        if isinstance(summary, str) and summary.strip():
            summary_ok = True
            break
    if not summary_ok:
        issues.append("缺少末尾 summary JSON")

if issues:
    print("\n".join(issues))
    sys.exit(1)
PYEOF
}

# ── 写入文件 + 更新 posts.json ─────────────────────────────────────────────────
publish_article() {
    local slug="$1" title="$2" tags_json="$3" output="$4"

    # 提取 summary（末尾 JSON 行）
    local summary
    summary=$(printf '%s' "$output" | python3 -c '
import json, sys
lines = sys.stdin.read().splitlines()
for line in reversed(lines):
    line = line.strip()
    if not line:
        continue
    if not line.startswith("{"):
        break
    try:
        obj = json.loads(line)
    except Exception:
        break
    summary = obj.get("summary", "")
    if isinstance(summary, str) and summary.strip():
        print(summary.strip())
        break
')
    [[ -z "$summary" ]] && summary="$title"

    # 去掉 summary 行，保留正文
    local content
    content=$(printf '%s' "$output" | python3 -c '
import json, sys
lines = sys.stdin.read().splitlines()
drop_idx = None
for idx in range(len(lines) - 1, -1, -1):
    line = lines[idx].strip()
    if not line:
        continue
    if not line.startswith("{"):
        break
    try:
        obj = json.loads(line)
    except Exception:
        break
    if isinstance(obj.get("summary"), str):
        drop_idx = idx
        break

if drop_idx is None:
    print("\n".join(lines))
else:
    print("\n".join(lines[:drop_idx] + lines[drop_idx + 1:]))
')

    # 写 .md
    printf '%s\n' "$content" > "$POSTS_DIR/${slug}.md"
    log "  已写入正文：$POSTS_DIR/${slug}.md"

    # 更新 posts.json（已存在则覆盖，不存在则插入）
    local entry exists
    entry=$(jq -n \
        --arg title   "$title" \
        --arg slug    "$slug" \
        --arg date    "$(date +%Y-%m-%d)" \
        --arg summary "$summary" \
        --argjson tags "$tags_json" \
        '{title:$title,slug:$slug,date:$date,author:"both",tags:$tags,summary:$summary}')
    exists=$(jq --arg s "$slug" 'any(.[]; .slug == $s)' "$POSTS_JSON")
    if [[ "$exists" == "true" ]]; then
        jq --argjson e "$entry" 'map(if .slug == $e.slug then $e else . end)' "$POSTS_JSON" \
            > /tmp/posts_tmp_$$ && mv /tmp/posts_tmp_$$ "$POSTS_JSON"
    else
        jq --argjson e "$entry" '. = [$e] + .' "$POSTS_JSON" \
            > /tmp/posts_tmp_$$ && mv /tmp/posts_tmp_$$ "$POSTS_JSON"
    fi
}

validate_posts_json_entries() {
    local slug count
    if ! jq empty "$POSTS_JSON" >/dev/null 2>&1; then
        err "posts/posts.json 不是合法 JSON"
        return 1
    fi

    for slug in "$@"; do
        count=$(jq --arg s "$slug" '[.[] | select(.slug == $s)] | length' "$POSTS_JSON")
        if [[ "$count" -ne 1 ]]; then
            err "posts/posts.json 中 slug=${slug} 的条目数量异常：${count}"
            return 1
        fi
    done
}

# ── 调用 Codex 批量校验并改写草稿 ──────────────────────────────────────────────
refine_batch_with_codex() {
    local -a batch_slugs=("$@")
    local tmp_out tmp_err prompt file_refs slug
    tmp_out=$(mktemp /tmp/auto_post_codex_out_XXXXXX)
    tmp_err=$(mktemp /tmp/auto_post_codex_err_XXXXXX)
    file_refs=""

    if [[ ! -x "$CODEX_BIN" ]]; then
        err "未找到可执行的 codex CLI: $CODEX_BIN"
        rm -f "$tmp_out" "$tmp_err"
        return 1
    fi

    for slug in "${batch_slugs[@]}"; do
        file_refs="${file_refs}
- ${RUN_DRAFT_DIR}/${slug}.md"
    done

    prompt=$(cat <<EOF
请对本批次技术博客草稿逐篇执行“校验 + 改写润色”，并直接在工作区原地修改草稿文件。

只允许修改以下文件：
${file_refs}

任务要求：
- 你需要先自行检查章节完整性、表格、公式/数学记号、代码可运行性、参考资料数量、summary JSON，再按缺陷直接改写到合格
- 在保持主题、章节顺序、核心结论和整体篇幅级别基本稳定的前提下，润色表达并修正技术问题
- 保持每篇文章的标题、slug、主题边界、章节顺序和主要结论不变
- 不要删除必须章节，不要新增与主题无关的内容，不要改动未列出的文件
- 不要写任何关于权限、文件访问、工具调用、网络环境的描述
- 每篇草稿末尾必须保留 summary JSON，供后续发布脚本提取摘要

完成后只输出一句简短中文说明。
EOF
)

    local exit_code=0
    "$CODEX_BIN" exec \
        --model "$CODEX_REFINE_MODEL" \
        -c "model_reasoning_effort=\"${CODEX_REFINE_REASONING_EFFORT}\"" \
        --skip-git-repo-check \
        --color never \
        -s workspace-write \
        --output-last-message "$tmp_out" \
        -C "$REPO_ROOT" \
        "$prompt" \
        > /dev/null 2>"$tmp_err" || exit_code=$?

    local stderr_content
    stderr_content=$(cat "$tmp_err")
    rm -f "$tmp_err"

    if [[ "$exit_code" -ne 0 ]]; then
        if grep -Eiq 'permission|readonly|Operation not permitted|websocket|shell_snapshot|state_db' <<< "$stderr_content"; then
            err "  codex 调用失败（环境噪声已省略），请检查 codex 登录状态或网络连接。"
        else
            err "  codex 退出码 ${exit_code}: $(echo "$stderr_content" | head -3)"
        fi
        rm -f "$tmp_out"
        return 1
    fi

    cat "$tmp_out"
    rm -f "$tmp_out"
}

# ── Git 提交推送 ───────────────────────────────────────────────────────────────
git_commit_push_batch() {
    local -a batch_slugs=("$@")
    cd "$REPO_ROOT"
    git add posts/posts.json scripts/topic_queue.json
    local slug
    for slug in "${batch_slugs[@]}"; do
        git add "posts/${slug}.md"
    done
    if git diff --cached --quiet; then
        warn "  无暂存变更，跳过 commit"
        return 0
    fi
    {
        printf 'post: batch %s (%d articles)\n\n' "$(date +%Y-%m-%d)" "${#batch_slugs[@]}"
        printf 'Generated with Codex (%s) and validated/polished with Codex (%s).\n\n' "$CODEX_DRAFT_MODEL" "$CODEX_REFINE_MODEL"
        printf 'Articles:\n'
        printf '%s\n' "${batch_slugs[@]}"
        printf '\nCo-Authored-By: Codex <noreply@openai.com>\n'
    } | git commit -F -
    git push origin main
    log "  ✓ 已推送本批次 ${#batch_slugs[@]} 篇文章"
}

# ── 更新队列状态（无时间戳）──────────────────────────────────────────────────
mark_queue() {
    local slug="$1" status="$2"
    jq --arg slug "$slug" --arg s "$status" \
        'map(if .slug == $slug then .status = $s else . end)' \
        "$QUEUE_FILE" > /tmp/queue_tmp_$$ && mv /tmp/queue_tmp_$$ "$QUEUE_FILE"
}

mark_queue_batch() {
    local status="$1"
    shift
    local slug
    for slug in "$@"; do
        mark_queue "$slug" "$status"
    done
}

# ── Dry-run ───────────────────────────────────────────────────────────────────
if $DRY_RUN; then
    echo "本次将按顺序生成以下 ${BATCH_SIZE} 篇："
    echo ""
    slugs=()
    while IFS= read -r line; do slugs+=("$line"); done < <(pick_next_slugs)
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
log "=== auto_post.sh 启动，按顺序取前 ${BATCH_SIZE} 篇生成 ==="
log "模型配置：draft=${CODEX_DRAFT_MODEL}(${CODEX_DRAFT_REASONING_EFFORT}) refine=${CODEX_REFINE_MODEL}(${CODEX_REFINE_REASONING_EFFORT})"

# 取本批次 slug 列表
slugs=()
while IFS= read -r line; do slugs+=("$line"); done < <(pick_next_slugs)

if [[ "${#slugs[@]}" -eq 0 ]]; then
    log "队列为空，所有主题已完成。"
    exit 0
fi

log "本批次共 ${#slugs[@]} 篇：$(IFS=', '; echo "${slugs[*]}")"

generated_slugs=()
total_articles="${#slugs[@]}"
total_steps=$(( total_articles * 2 + 1 ))
progress_step=0

for slug in "${slugs[@]}"; do
    topic=$(get_topic "$slug")
    title=$(     jq -r '.title'                                                  <<< "$topic")
    brief=$(     jq -r '.brief'                                                  <<< "$topic")
    depth_hint=$(jq -r '.depth_hint // "从基础原理到工程实现，包含完整数学推导和可运行代码"' <<< "$topic")
    blog_category=$(jq -r '.blog_category // (.tags[0] // "工程实践")'           <<< "$topic")
    tags_text=$(   jq -r '(.tags // []) | join(" / ")'                           <<< "$topic")
    prereq_titles=$(resolve_prereq_titles "$topic")
    tags_json=$( jq '
        if .blog_category then
            ([.blog_category] + (.tags // []))
            | reduce .[] as $tag ([]; if index($tag) then . else . + [$tag] end)
        else
            (.tags // [])
        end
    ' <<< "$topic")
    article_context=$(cat <<EOF
标题：${title}
博客分类：${blog_category}
主题标签：${tags_text}
前置阅读：${prereq_titles:-无}
核心要点：${brief}
深度方向：${depth_hint}
EOF
)

    log "──────────────────────────────────────────"
    log "生成中：${title}"
    log "slug  ：${slug}"
    log "分类  ：${blog_category}"
    progress_step=$((progress_step + 1))
    render_progress "$progress_step" "$total_steps" "生成 ${slug}"

    output=""
    rc=0
    output=$(generate_article "$article_context") || rc=$?

    if [[ "$rc" -ne 0 ]]; then
        save_failed_article "$slug" "generation-error" "$output"
        err "生成失败，立即停止。本批次已生成内容会保留在本地，且不会推送 git。"
        exit 1
    fi

    stage_draft_article "$slug" "$output"
    draft_file=$(draft_file_for_slug "$slug")
    if [[ ! -s "$draft_file" ]]; then
        save_failed_article "$slug" "empty-draft" "$output"
        err "Codex 生成的草稿为空，立即停止。"
        exit 1
    fi

    generated_slugs+=("$slug")

    sleep 3
done

log "Codex 草稿生成完成，开始调用 Codex 逐篇校验并改写本批次 ${#generated_slugs[@]} 篇文章"
progress_step=$((progress_step + 1))
render_progress "$progress_step" "$total_steps" "Codex 校验改写"

codex_result=""
rc=0
codex_result=$(refine_batch_with_codex "${generated_slugs[@]}") || rc=$?
if [[ "$rc" -ne 0 ]]; then
    err "Codex 批量校验改写失败，已保留草稿，未推送 git。"
    exit 1
fi
[[ -n "${codex_result//[[:space:]]/}" ]] && log "Codex：$(echo "$codex_result" | tr '\n' ' ' | sed 's/[[:space:]]\+/ /g')"

for slug in "${generated_slugs[@]}"; do
    draft_file=$(draft_file_for_slug "$slug")
    progress_step=$((progress_step + 1))
    render_progress "$progress_step" "$total_steps" "发布 ${slug}"
    sanitize_article_output "$draft_file"
    validation_failed=false
    issues=$(validate_article_output "$draft_file" 1) || validation_failed=true
    if $validation_failed; then
        save_failed_article "$slug" "codex-validation-failed" "$(cat "$draft_file")"
        err "Codex 改写后文章校验失败：${slug}"
        while IFS= read -r issue; do
            [[ -n "$issue" ]] && err "  - $issue"
        done <<< "$issues"
        exit 1
    fi

    output=$(cat "$draft_file")
    topic=$(get_topic "$slug")
    title=$(     jq -r '.title'                                        <<< "$topic")
    tags_json=$( jq '
        if .blog_category then
            ([.blog_category] + (.tags // []))
            | reduce .[] as $tag ([]; if index($tag) then . else . + [$tag] end)
        else
            (.tags // [])
        end
    ' <<< "$topic")
    publish_article "$slug" "$title" "$tags_json" "$output"
    validation_failed=false
    issues=$(validate_article_output "$POSTS_DIR/${slug}.md" 0) || validation_failed=true
    if $validation_failed; then
        err "发布后的正文校验失败：${slug}"
        while IFS= read -r issue; do
            [[ -n "$issue" ]] && err "  - $issue"
        done <<< "$issues"
        exit 1
    fi
done

validate_posts_json_entries "${generated_slugs[@]}" || exit 1

mark_queue_batch "done" "${generated_slugs[@]}"
git_commit_push_batch "${generated_slugs[@]}"

log "=== 本批次 ${#generated_slugs[@]} 篇全部完成 ==="
