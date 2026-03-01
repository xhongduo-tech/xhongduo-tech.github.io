#!/usr/bin/env python3
"""
auto_post.py — 自动生成技术博文并推送到 GitHub Pages

用法：
  python3 scripts/auto_post.py --count 3    # 生成 3 篇并推送（默认）
  python3 scripts/auto_post.py --retry      # 仅重试失败的推送，不生成新文章
  python3 scripts/auto_post.py --dry-run    # 打印将要写的主题，不实际生成

依赖：
  pip3 install anthropic
  设置环境变量 ANTHROPIC_API_KEY 或在 scripts/.env 写入
"""

import json
import os
import sys
import fcntl
import subprocess
import argparse
import logging
from datetime import date, datetime
from pathlib import Path

import anthropic

# ── 路径配置 ───────────────────────────────────────────────────────────────────
REPO_ROOT   = Path(__file__).parent.parent
SCRIPTS_DIR = REPO_ROOT / "scripts"
POSTS_DIR   = REPO_ROOT / "posts"
QUEUE_FILE  = SCRIPTS_DIR / "topic_queue.json"
POSTS_JSON  = POSTS_DIR / "posts.json"
LOGS_DIR    = SCRIPTS_DIR / "logs"
FAILED_LOG  = LOGS_DIR / "failed_push.log"
LOCK_FILE   = SCRIPTS_DIR / ".auto_post.lock"
ENV_FILE    = SCRIPTS_DIR / ".env"

LOGS_DIR.mkdir(parents=True, exist_ok=True)

# ── 读取 .env（供 launchd 使用，shell 启动时不加载 .zshrc）────────────────────
if ENV_FILE.exists():
    for raw in ENV_FILE.read_text().splitlines():
        raw = raw.strip()
        if raw and not raw.startswith("#") and "=" in raw:
            k, v = raw.split("=", 1)
            os.environ.setdefault(k.strip(), v.strip())

# ── 日志 ──────────────────────────────────────────────────────────────────────
log_file = LOGS_DIR / f"auto_post_{datetime.now().strftime('%Y%m%d')}.log"
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.FileHandler(log_file, encoding="utf-8"),
        logging.StreamHandler(sys.stdout),
    ],
)
log = logging.getLogger(__name__)

# ── 系统提示 ───────────────────────────────────────────────────────────────────
SYSTEM_PROMPT = """你是一位专注于大模型技术的技术博客作者，面向技术从业者（开发者、研究者）写作。

【写作风格规范 — 必须严格遵守】

1. 开篇直接给出定义或核心结论。第一段必须回答"这是什么/这解决什么问题"。
   禁止：情感铺垫、"你一定遇到过"、"大家好今天来聊聊"等。

2. 先结论后展开：先给精确定义，再展开机制细节。读者应在最早位置获得核心信息。

3. 使用精确技术术语。首次出现给中英文对照，例如"注意力机制（Attention）"。

4. 数据和代码胜过描述：能用公式说清的不用文字绕，能用代码演示的不用段落描述，
   表格优于列举，图表数据优于"非常有效"等形容词。

5. 禁止：感叹号、"非常重要/令人惊讶/完美"等夸张修辞、口语化比喻替代定义、
   重复强调同一要点、空洞过渡句（"接下来我们看看……"）。

6. 每个 ## 大节之间用 --- 分隔。段落不超过 5 行。

7. 文章末尾附参考资料（论文 arXiv 链接或官方文档）。

【格式要求】
- 从 ## 开始写（不写 # 标题）
- 数学公式：行内用 $...$，块级用 $$...$$
- 代码块注明语言
- 目标长度：1500-3000 字，以技术内容为主，不凑字数

【技术深度要求】
- 面向技术从业者，不解释基本编程概念
- 包含数学推导过程（不只是结论）
- 给出具体数值、实验数据、代码示例
- 指出该技术的局限性和工程注意事项

【输出格式】
在文章正文结束后，另起一行输出如下 JSON（不要放在代码块里）：
{"summary": "一两句摘要，不超过60字，技术事实陈述"}
"""


# ── 工具函数 ───────────────────────────────────────────────────────────────────

def read_queue() -> list[dict]:
    with open(QUEUE_FILE, encoding="utf-8") as f:
        return json.load(f)


def write_queue(queue: list[dict]) -> None:
    with open(QUEUE_FILE, "w", encoding="utf-8") as f:
        json.dump(queue, f, ensure_ascii=False, indent=2)


def read_posts() -> list[dict]:
    with open(POSTS_JSON, encoding="utf-8") as f:
        return json.load(f)


def write_posts(posts: list[dict]) -> None:
    with open(POSTS_JSON, "w", encoding="utf-8") as f:
        json.dump(posts, f, ensure_ascii=False, indent=2)


# ── 文章生成 ───────────────────────────────────────────────────────────────────

def generate_article(client: anthropic.Anthropic, topic: dict) -> tuple[str, str]:
    """调用 claude-opus-4-6 + extended thinking 生成文章，返回 (markdown, summary)。"""
    user_prompt = (
        f"请写一篇技术博文，主题如下：\n\n"
        f"标题：{topic['title']}\n"
        f"核心要点：{topic['brief']}\n"
        f"深度方向：{topic.get('depth_hint', '深入原理，包含数学推导和可运行代码示例')}\n\n"
        "请严格遵循系统提示的风格规范，在文章末尾输出 summary JSON。"
    )

    log.info(f"  正在生成：{topic['title']}")

    response = client.messages.create(
        model="claude-opus-4-6",
        max_tokens=16000,
        thinking={
            "type": "enabled",
            "budget_tokens": 10000,
        },
        system=SYSTEM_PROMPT,
        messages=[{"role": "user", "content": user_prompt}],
    )

    # 只取 text block，忽略 thinking block
    full_text = "".join(
        block.text for block in response.content if block.type == "text"
    ).strip()

    # 解析尾部 summary JSON
    summary = topic["title"]  # fallback
    lines = full_text.splitlines()
    for line in reversed(lines):
        stripped = line.strip()
        if stripped.startswith('{"summary":'):
            try:
                summary = json.loads(stripped)["summary"]
                # 从正文中移除这行
                idx = full_text.rfind(stripped)
                full_text = full_text[:idx].rstrip()
            except Exception:
                pass
            break

    return full_text, summary


def publish_article(topic: dict, markdown: str, summary: str) -> Path:
    """写入 .md 文件，更新 posts.json，返回 .md 路径。"""
    slug = topic["slug"]
    today = date.today().isoformat()

    md_path = POSTS_DIR / f"{slug}.md"
    md_path.write_text(markdown, encoding="utf-8")

    posts = read_posts()
    if any(p["slug"] == slug for p in posts):
        log.warning(f"  posts.json 已存在 slug={slug}，跳过插入")
        return md_path

    posts.insert(0, {
        "title":   topic["title"],
        "slug":    slug,
        "date":    today,
        "author":  "claude",
        "tags":    topic["tags"],
        "summary": summary,
    })
    write_posts(posts)
    return md_path


# ── Git 操作 ───────────────────────────────────────────────────────────────────

def git_push(slugs: list[str]) -> bool:
    """add + commit + push，返回是否成功。"""
    files = [f"posts/{s}.md" for s in slugs] + ["posts/posts.json"]
    commit_msg = (
        "post: " + " / ".join(slugs) + "\n\n"
        "Co-Authored-By: Claude Opus 4.6 <noreply@anthropic.com>"
    )
    try:
        subprocess.run(
            ["git", "add"] + files,
            cwd=REPO_ROOT, check=True, capture_output=True,
        )
        # 检查是否有内容可 commit
        status = subprocess.run(
            ["git", "diff", "--cached", "--quiet"],
            cwd=REPO_ROOT, capture_output=True,
        )
        if status.returncode == 0:
            log.info("  没有暂存变更，跳过 commit")
            return True

        subprocess.run(
            ["git", "commit", "-m", commit_msg],
            cwd=REPO_ROOT, check=True, capture_output=True,
        )
        result = subprocess.run(
            ["git", "push", "origin", "main"],
            cwd=REPO_ROOT, capture_output=True, text=True,
        )
        if result.returncode != 0:
            log.error(f"  push 失败：{result.stderr.strip()}")
            return False
        log.info(f"  push 成功：{' / '.join(slugs)}")
        return True
    except subprocess.CalledProcessError as e:
        log.error(f"  git 操作异常：{e.stderr.decode() if e.stderr else e}")
        return False


def log_failed_push(slugs: list[str]) -> None:
    record = json.dumps({
        "time": datetime.now().isoformat(),
        "slugs": slugs,
    }, ensure_ascii=False)
    with open(FAILED_LOG, "a", encoding="utf-8") as f:
        f.write(record + "\n")
    log.warning(f"  已记录失败推送：{slugs}")


# ── 重试失败推送 ───────────────────────────────────────────────────────────────

def retry_failed_pushes() -> None:
    if not FAILED_LOG.exists():
        return
    lines = [l for l in FAILED_LOG.read_text(encoding="utf-8").splitlines() if l.strip()]
    if not lines:
        return

    log.info(f"发现 {len(lines)} 条待重试推送记录...")
    remaining = []
    for line in lines:
        try:
            record = json.loads(line)
        except Exception:
            continue
        slugs = record["slugs"]
        missing = [s for s in slugs if not (POSTS_DIR / f"{s}.md").exists()]
        if missing:
            log.warning(f"  文件缺失 {missing}，跳过该记录")
            continue
        if git_push(slugs):
            log.info(f"  重推成功：{slugs}")
        else:
            remaining.append(record)

    FAILED_LOG.write_text(
        "\n".join(json.dumps(r, ensure_ascii=False) for r in remaining)
        + ("\n" if remaining else ""),
        encoding="utf-8",
    )


# ── 主流程 ─────────────────────────────────────────────────────────────────────

def run(count: int) -> None:
    client = anthropic.Anthropic()  # 自动读取 ANTHROPIC_API_KEY

    queue = read_queue()
    pending = [t for t in queue if t.get("status") == "pending"]
    if not pending:
        log.info("主题队列为空，没有待生成文章。请在 scripts/topic_queue.json 中添加主题。")
        return

    batch = pending[:count]
    log.info(f"本次生成 {len(batch)} 篇：{[t['title'] for t in batch]}")

    published_slugs = []
    for topic in batch:
        try:
            markdown, summary = generate_article(client, topic)
            publish_article(topic, markdown, summary)
            for t in queue:
                if t["slug"] == topic["slug"]:
                    t["status"] = "done"
                    t["done_at"] = datetime.now().isoformat()
                    break
            published_slugs.append(topic["slug"])
            log.info(f"  写入成功：posts/{topic['slug']}.md")
        except Exception as e:
            log.error(f"  生成失败 [{topic['title']}]：{e}", exc_info=True)
            for t in queue:
                if t["slug"] == topic["slug"]:
                    t["status"] = "error"
                    t["error"] = str(e)
                    break

    write_queue(queue)

    if not published_slugs:
        log.warning("本批次没有成功生成的文章，跳过推送")
        return

    if not git_push(published_slugs):
        log_failed_push(published_slugs)


def dry_run(count: int) -> None:
    queue = read_queue()
    pending = [t for t in queue if t.get("status") == "pending"]
    if not pending:
        print("队列为空")
        return
    print(f"接下来将生成 {min(count, len(pending))} 篇：")
    for t in pending[:count]:
        print(f"  [{t['tags'][0]}] {t['title']}")
        print(f"    slug: {t['slug']}")
        print(f"    brief: {t['brief'][:80]}...")
        print()


def main():
    parser = argparse.ArgumentParser(description="自动生成技术博文并推送")
    parser.add_argument("--count",   type=int, default=3, help="本次生成文章数量（默认 3）")
    parser.add_argument("--retry",   action="store_true",  help="仅重试失败推送，不生成新文章")
    parser.add_argument("--dry-run", action="store_true",  help="预览待生成主题，不实际调用 API")
    args = parser.parse_args()

    if args.__dict__["dry_run"]:
        dry_run(args.count)
        return

    # 互斥锁：防止凌晨 1 点和早上 7 点任务同时运行
    lock_fd = open(LOCK_FILE, "w")
    try:
        fcntl.flock(lock_fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except OSError:
        log.warning("另一个 auto_post 进程正在运行，本次跳过")
        sys.exit(0)

    try:
        retry_failed_pushes()
        if not args.retry:
            run(args.count)
    finally:
        fcntl.flock(lock_fd, fcntl.LOCK_UN)
        lock_fd.close()
        LOCK_FILE.unlink(missing_ok=True)


if __name__ == "__main__":
    main()
