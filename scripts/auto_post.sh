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
SYSTEM_PROMPT=$(cat <<'EOF'
你是一位面向资深工程师的 AI 技术博客作者。文章目标不是泛科普，而是输出可以复核、可以实现、可以比较的技术分析。

总原则：
- 先结论后展开，第一节直接回答“它解决什么问题、核心判断是什么、适用边界在哪里”
- 全文围绕一条主线展开：问题定义 -> 机制/推导 -> 数值例子 -> 代码实现 -> 工程权衡 -> 替代方案
- 术语、符号、变量命名必须前后一致；同一个 running example 要贯穿公式、代码与工程分析
- 理论/训练/微调主题要有实打实的数学推导；部署/系统/工程主题可以用复杂度、容量、协议或成本模型代替纯数学推导，但不能硬凑空洞公式
- 只使用第一手或高可信参考：论文、官方文档、源码、权威技术博客

必须严格遵循以下章节顺序：

## 核心结论
用 2-4 句话给出定义、价值、核心判断和适用边界。

---

## 问题定义与边界
明确输入输出、符号、shape、假设条件与问题约束；如果有前置概念，只做最短承接，不重复铺垫。

---

## 核心机制与推导
讲清机制。理论主题做公式推导，工程主题做复杂度/容量/协议模型分析。
要求：
- 每个关键公式后都给一个具体数值例子
- 每个张量或缓存都标注 shape / 尺寸 / 字节级估算
- 至少给一个 Markdown 表格比较方案差异

---

## 代码实现
必须给完整、可运行、可验证的 `python` 代码。
要求：
- 复制即可执行
- 含 `assert` 或明确的输出校验
- 打印关键中间结果
- 不允许出现 `...`、placeholder、伪代码式空壳

---

## 工程权衡与常见坑
从延迟、吞吐、显存、复杂度、可维护性、失败模式几个角度展开，说明真实瓶颈在哪里、最容易踩的坑是什么。

---

## 替代方案与适用边界
明确什么时候不该用它，并和至少两种替代方案做对比。

---

## 参考资料
至少 3 条，优先论文 / 官方文档 / 源码 / 高质量技术博客；每条都说明推荐理由。

强制质量门槛：
- 至少一个完整 `python` 代码块
- 至少一个 Markdown 表格
- 至少一个带具体数字的推演
- 不能出现“这里只给思路”“实现略”“读者自行补全”
- 不要使用夸张语气、口语化比喻、读者心理模拟
- 文末最后一行单独输出：{"summary":"不超过60字的核心摘要"}

格式要求：
- 从 `##` 开始，不写 `#`
- 章节间用 `---`
- 数学用 `$...$` 和 `$$...$$`
- 段落尽量短，每段不超过 5 行
EOF
)

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

# ── 调用 Claude ───────────────────────────────────────────────────────────────
run_claude_prompt() {
    local prompt="$1"
    local tmp_out tmp_err
    tmp_out=$(mktemp /tmp/auto_post_out_XXXXXX)
    tmp_err=$(mktemp /tmp/auto_post_err_XXXXXX)

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

# ── 生成文章 ──────────────────────────────────────────────────────────────────
generate_article() {
    local article_context="$1"
    local prompt="${SYSTEM_PROMPT}

请写一篇技术博文，写作上下文如下：

${article_context}

额外要求：
- 默认读者已经掌握“前置阅读”里的内容，不要重复讲基础定义；只在必要处用 1-2 句承接
- 用同一个 running example 贯穿公式、代码和工程分析
- 如果是工程或部署主题，不要伪造学术化推导；请用复杂度、容量、调度或协议模型解释
- 如果是理论、训练或微调主题，公式必须可推导、符号必须可复核
- 最后一行必须单独输出 summary JSON
"

    run_claude_prompt "$prompt"
}

# ── 修订文章 ──────────────────────────────────────────────────────────────────
repair_article() {
    local article_context="$1" draft="$2" issues="$3"
    local prompt="${SYSTEM_PROMPT}

下面是一篇技术博客初稿，但它未通过质量门槛。请根据缺失项直接重写为合格的完整终稿。

写作上下文：
${article_context}

未通过项：
${issues}

初稿如下：
${draft}

请只输出修订后的完整正文，保持章节顺序不变，最后一行继续输出 summary JSON，不要解释修改过程。
"

    run_claude_prompt "$prompt"
}

# ── 质量校验 ──────────────────────────────────────────────────────────────────
validate_article_output() {
    local article_file="$1"
    python3 - "$article_file" <<'PYEOF'
import json, re, sys
from pathlib import Path

text = Path(sys.argv[1]).read_text()
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
if len(re.findall(r"^\d+\.\s+\[", text, flags=re.M)) < 3:
    issues.append("参考资料少于 3 条")

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

    output=""
    rc=0
    output=$(generate_article "$article_context") || rc=$?

    if [[ "$rc" -ne 0 ]]; then
        err "生成失败，立即停止。已完成主题已标记为 done。"
        exit 1
    fi

    validate_file=$(mktemp /tmp/auto_post_validate_XXXXXX)
    printf '%s' "$output" > "$validate_file"
    validation_failed=false
    issues=$(validate_article_output "$validate_file") || validation_failed=true

    if $validation_failed; then
        warn "  首轮生成未通过质量门槛，执行一次修订"
        while IFS= read -r issue; do
            [[ -n "$issue" ]] && warn "    - $issue"
        done <<< "$issues"

        rc=0
        output=$(repair_article "$article_context" "$output" "$issues") || rc=$?
        if [[ "$rc" -ne 0 ]]; then
            rm -f "$validate_file"
            err "修订失败，立即停止。"
            exit 1
        fi

        printf '%s' "$output" > "$validate_file"
        validation_failed=false
        issues=$(validate_article_output "$validate_file") || validation_failed=true
        if $validation_failed; then
            rm -f "$validate_file"
            err "修订后仍未通过质量门槛："
            while IFS= read -r issue; do
                [[ -n "$issue" ]] && err "  - $issue"
            done <<< "$issues"
            exit 1
        fi
    fi
    rm -f "$validate_file"

    publish_article "$slug" "$title" "$tags_json" "$output"
    mark_queue "$slug" "done"
    git_commit_push "$slug"

    sleep 3
done

log "=== 本批次 ${#slugs[@]} 篇全部完成 ==="
