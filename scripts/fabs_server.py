"""
FABS — Fully Automated Blog-post System
FastAPI + WebSocket backend, Anthropic SDK pipeline.
Run: python scripts/fabs_server.py
Open: http://127.0.0.1:8765
"""

from __future__ import annotations

import asyncio
import fcntl
import json
import os
import re
import shutil
import sys
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

import anthropic
from fastapi import FastAPI, Request, WebSocket, WebSocketDisconnect
from fastapi.responses import HTMLResponse

# Path to the 'claude' CLI (Claude Code) — used in CLI mode
CLI_BINARY: str = shutil.which("claude") or "claude"

# ──────────────────────────────────────────────────────────────
# Paths
# ──────────────────────────────────────────────────────────────
REPO_ROOT = Path(__file__).parent.parent
QUEUE_FILE = REPO_ROOT / "scripts" / "topic_queue.json"
POSTS_DIR = REPO_ROOT / "posts"
POSTS_JSON = POSTS_DIR / "posts.json"
FAILED_DIR = POSTS_DIR / "_failed"
DASHBOARD_HTML = Path(__file__).parent / "fabs_dashboard.html"
CONFIG_FILE = Path(__file__).parent / "fabs_config.json"

# ──────────────────────────────────────────────────────────────
# Model catalogue
# ──────────────────────────────────────────────────────────────
AVAILABLE_MODELS: list[tuple[str, str]] = [
    ("claude-haiku-4-5-20251001", "Haiku 4.5 (Fast)"),
    ("claude-sonnet-4-6", "Sonnet 4.6 (Balanced)"),
    ("claude-opus-4-6", "Opus 4.6 (Best)"),
]

MODEL_COSTS: dict[str, tuple[float, float]] = {
    "claude-haiku-4-5-20251001": (0.00000025, 0.00000125),
    "claude-sonnet-4-6": (0.000003, 0.000015),
    "claude-opus-4-6": (0.000015, 0.000075),
}

DEFAULT_MODELS = {
    "research": "claude-haiku-4-5-20251001",
    "outline": "claude-haiku-4-5-20251001",
    "write": "claude-sonnet-4-6",
    "review": "claude-sonnet-4-6",
    "refine": "claude-sonnet-4-6",
}

STAGES = ["research", "outline", "write", "review", "refine", "publish"]

# ──────────────────────────────────────────────────────────────
# Config file helpers
# ──────────────────────────────────────────────────────────────
def load_config() -> dict:
    config: dict = {
        "anthropic_api_key": os.environ.get("ANTHROPIC_API_KEY", ""),
        "batch_size": 0,  # 0 = unlimited
        "use_cli": False,  # True = Claude Pro subscription via CLI
    }
    if CONFIG_FILE.exists():
        try:
            saved = json.loads(CONFIG_FILE.read_text(encoding="utf-8"))
            config.update(saved)
        except Exception:
            pass
    # env var always overrides saved key if present
    env_key = os.environ.get("ANTHROPIC_API_KEY", "")
    if env_key:
        config["anthropic_api_key"] = env_key
    return config


def save_config(config: dict) -> None:
    # Never write env key to disk if it came from environment
    to_save = dict(config)
    CONFIG_FILE.write_text(json.dumps(to_save, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


# ──────────────────────────────────────────────────────────────
# Prompt templates
# ──────────────────────────────────────────────────────────────
RESEARCH_PROMPT = """\
Topic: {title}
Brief: {brief}
Category: {blog_category}

Research and provide a technical summary covering:
1. Core definition/theorem (2-3 precise sentences, no warm-up)
2. Mathematical foundations — unify notation, state assumptions, sketch key proof steps
3. Step-by-step mechanism (numbered, concrete); include a minimal numerical example
4. Typical use cases and constraints; compare at least two related approaches on complexity/accuracy/stability
5. Common pitfalls with root causes
6. Related work / alternatives with at least 3 specific citations (papers, docs, or source code)

Output in Chinese. Be precise and technical. No filler phrases."""

OUTLINE_PROMPT = """\
为以下文章创建详细大纲：{title}

研究摘要：
{research}

必须包含以下章节（顺序固定，一个不能少）：
1. ## 核心结论
2. ## 问题定义与边界
3. ## 核心机制与推导
4. ## 代码实现
5. ## 工程权衡与常见坑
6. ## 替代方案与适用边界
7. ## 参考资料

对每个章节说明：内容要点（2-3条）、所需元素（公式/表格/代码）、预计篇幅。
选择一个贯穿全文的 running example 用于章节 2-5。"""

WRITE_PROMPT = """\
你是一位面向资深工程师的 AI 技术博客作者。输出要可复核、可运行、可比较，避免空话。

只输出文章正文，不要输出任何前言、说明、道歉或过程描述。

严格使用以下章节顺序：
## 核心结论
## 问题定义与边界
## 核心机制与推导
## 代码实现
## 工程权衡与常见坑
## 替代方案与适用边界
## 参考资料

写作方法论：
- 先统一符号与假设，再完成核心推导；用最小数值实验验证公式，最后对比至少两种近似方法在复杂度/误差界/稳定性上的差异
- 每节选一个 running example 贯穿机制推导、代码实现、工程分析

格式要求：
- 从 `##` 开始，不写 `#`（H1 由系统自动生成）
- 章节之间使用 `---`
- 至少包含一个可运行的 `python` 代码块，含 `assert` 或明确校验输出
- 至少包含一个 Markdown 表格
- 至少包含数学公式 $...$ 或 $$...$$
- 优先引用论文、官方文档、源码或高质量技术博客
- 文末最后一行单独输出：{{"summary":"不超过60字的核心摘要"}}

---

标题：{title}
博客分类：{blog_category}
主题标签：{tags_text}
核心要点：{brief}

大纲：
{outline}

研究摘要：
{research}"""

REVIEW_PROMPT = """\
Review this blog post for quality. Return JSON only, no other text.

{article_text}

Check:
1. All 7 required sections present: 核心结论/问题定义与边界/核心机制与推导/代码实现/工程权衡与常见坑/替代方案与适用边界/参考资料
2. Python code block with runnable assertions
3. Markdown table exists (| column |)
4. Math formula ($...$ or $$...$$)
5. At least 3 references in 参考资料
6. JSON summary at end: {{"summary": "..."}}

Return exactly: {{"pass": true, "score": 85, "issues": [], "summary": "extracted summary"}}"""

REFINE_PROMPT = """\
Improve this technical blog article. Keep structure, title, main conclusions, and section order identical.

Fix any issues: missing sections, non-runnable code, missing table, missing math, thin references.
Improve: technical precision, example consistency, clarity.

Title: {title}

Article:
{article_text}

Output the complete improved article starting from ## with the same JSON summary line at the end."""


# ──────────────────────────────────────────────────────────────
# State
# ──────────────────────────────────────────────────────────────
@dataclass
class PostEntry:
    slug: str
    title: str
    status: str
    category: str = ""
    error: str = ""

    def to_dict(self) -> dict:
        return {"slug": self.slug, "title": self.title, "status": self.status,
                "category": self.category, "error": self.error}


@dataclass
class FABSState:
    running: bool = False
    paused: bool = False
    current_slug: Optional[str] = None
    current_stage: Optional[str] = None
    in_progress: list[dict] = field(default_factory=list)
    completed: list[dict] = field(default_factory=list)
    failed: list[dict] = field(default_factory=list)
    pending_preview: list[dict] = field(default_factory=list)   # first 30 from queue
    model_config: dict = field(default_factory=lambda: dict(DEFAULT_MODELS, refine_enabled=False))
    api_key_configured: bool = False
    use_cli: bool = False          # True = generating via Claude Pro CLI
    cli_configured: bool = False   # True = 'claude' binary found in PATH
    queue_loaded: bool = False     # True once topics have been loaded (from file or upload)
    stats: dict = field(default_factory=lambda: {
        "total_tokens": 0,
        "total_cost_usd": 0.0,
        "elapsed_seconds": 0,
        "pending_count": 0,
        "writing_count": 0,
        "published_count": 0,
    })
    start_time: Optional[str] = None

    def to_dict(self) -> dict:
        return {
            "running": self.running,
            "paused": self.paused,
            "current_slug": self.current_slug,
            "current_stage": self.current_stage,
            "in_progress": self.in_progress,
            "completed": self.completed,
            "failed": self.failed,
            "pending_preview": self.pending_preview,
            "model_config": self.model_config,
            "api_key_configured": self.api_key_configured,
            "use_cli": self.use_cli,
            "cli_configured": self.cli_configured,
            "queue_loaded": self.queue_loaded,
            "stats": self.stats,
            "start_time": self.start_time,
        }


# ──────────────────────────────────────────────────────────────
# FABS Manager
# ──────────────────────────────────────────────────────────────
class FABSManager:
    def __init__(self) -> None:
        self.state = FABSState()
        self._config = load_config()
        self._pending: list[dict] = []
        self._pause_event = asyncio.Event()
        self._pause_event.set()
        self._stop_flag = False
        self._pipeline_task: Optional[asyncio.Task] = None
        self._ws_clients: set[WebSocket] = set()
        self._lock = asyncio.Lock()
        self._init_client()
        self._load_queue()

    def _init_client(self) -> None:
        api_key = self._config.get("anthropic_api_key", "")
        self._client = anthropic.AsyncAnthropic(api_key=api_key or "placeholder")
        self.state.api_key_configured = bool(api_key and len(api_key) > 10)
        self.state.cli_configured = shutil.which("claude") is not None
        self.state.use_cli = bool(self._config.get("use_cli", False))

    # ── Config & Settings ──────────────────────────────────────
    async def update_settings(self, settings: dict) -> None:
        changed = False
        if "anthropic_api_key" in settings:
            self._config["anthropic_api_key"] = settings["anthropic_api_key"].strip()
            changed = True
        if "batch_size" in settings:
            self._config["batch_size"] = max(0, int(settings.get("batch_size", 0)))
        if "use_cli" in settings:
            self._config["use_cli"] = bool(settings["use_cli"])
            changed = True
        save_config(self._config)
        if changed:
            self._init_client()
        await self._broadcast_state()

    def get_settings(self) -> dict:
        key = self._config.get("anthropic_api_key", "")
        return {
            "anthropic_api_key": key,
            "anthropic_api_key_masked": ("sk-ant-..." + key[-4:]) if len(key) > 10 else "",
            "batch_size": self._config.get("batch_size", 0),
            "api_key_configured": self.state.api_key_configured,
            "use_cli": self._config.get("use_cli", False),
            "cli_configured": self.state.cli_configured,
            "cli_binary": CLI_BINARY if self.state.cli_configured else None,
        }

    # ── Queue ──────────────────────────────────────────────────
    def _load_queue(self) -> None:
        """Try to load queue from topic_queue.json; silently skip if file is missing."""
        if not QUEUE_FILE.exists():
            self._pending = []
            self.state.queue_loaded = False
            return
        try:
            data: list[dict] = json.loads(QUEUE_FILE.read_text(encoding="utf-8"))
            self._pending = [t for t in data if t.get("status") == "pending"]
            self.state.stats["pending_count"] = len(self._pending)
            self.state.queue_loaded = bool(self._pending)
            self._refresh_preview()
        except Exception as exc:
            print(f"Warning: could not load {QUEUE_FILE}: {exc}")
            self._pending = []
            self.state.queue_loaded = False

    async def load_queue_from_upload(self, raw: list) -> tuple[bool, str]:
        """Validate and load topics from a browser-uploaded JSON array."""
        if not isinstance(raw, list):
            return False, "JSON must be an array of topic objects"
        valid = [t for t in raw if isinstance(t, dict) and t.get("slug") and t.get("title")]
        if not valid:
            return False, "No valid items found — each item must have at least 'slug' and 'title'"
        # Treat all non-done items as pending; if there's no 'status' field treat as pending
        pending = [t for t in valid if t.get("status", "pending") not in ("done",)]
        if not pending:
            return False, "All items already have status 'done' — nothing to process"
        # Normalise: ensure status == "pending" so pipeline doesn't skip them
        for t in pending:
            t["status"] = "pending"
        self._pending = pending
        self.state.stats["pending_count"] = len(pending)
        self.state.queue_loaded = True
        self._refresh_preview()
        await self._broadcast_state()
        await self._broadcast_log("info", f"Queue loaded from upload: {len(pending)} topics ready.")
        return True, f"Loaded {len(pending)} topics"

    def _refresh_preview(self) -> None:
        self.state.pending_preview = [
            {
                "id": t.get("id", 0),
                "slug": t["slug"],
                "title": t["title"],
                "category": t.get("blog_category", ""),
                "tags": t.get("tags", [])[:3],
            }
            for t in self._pending[:30]
        ]

    def _update_queue_status(self, slug: str, status: str) -> None:
        try:
            with open(QUEUE_FILE, "r+", encoding="utf-8") as f:
                fcntl.flock(f, fcntl.LOCK_EX)
                data = json.load(f)
                for item in data:
                    if item.get("slug") == slug:
                        item["status"] = status
                        break
                f.seek(0)
                json.dump(data, f, ensure_ascii=False, indent=2)
                f.write("\n")
                f.truncate()
                fcntl.flock(f, fcntl.LOCK_UN)
        except Exception as e:
            asyncio.create_task(self._broadcast_log("warn", f"Queue update failed for {slug}: {e}"))

    # ── WebSocket broadcast ────────────────────────────────────
    async def _broadcast(self, msg: dict) -> None:
        dead: set[WebSocket] = set()
        for ws in list(self._ws_clients):
            try:
                await ws.send_json(msg)
            except Exception:
                dead.add(ws)
        self._ws_clients -= dead

    async def _broadcast_state(self) -> None:
        await self._broadcast({"type": "state_update", "data": self.state.to_dict()})

    async def _broadcast_log(self, level: str, msg: str) -> None:
        ts = datetime.now().strftime("%H:%M:%S")
        await self._broadcast({"type": "log", "level": level, "msg": f"[{ts}] {msg}"})

    async def _broadcast_chunk(self, stage: str, text: str) -> None:
        await self._broadcast({"type": "stream_chunk", "stage": stage, "text": text})

    async def _broadcast_stage_start(self, stage: str, slug: str) -> None:
        await self._broadcast({"type": "stage_start", "stage": stage, "slug": slug})

    # ── Controls ───────────────────────────────────────────────
    async def control(self, action: str) -> None:
        if action == "start" and not self.state.running:
            if self._config.get("use_cli", False):
                if not self.state.cli_configured:
                    await self._broadcast_log("error", "Claude CLI not found in PATH. Install Claude Code or switch to API mode in Settings.")
                    return
            else:
                if not self.state.api_key_configured:
                    await self._broadcast_log("error", "API key not configured. Open Settings (S) to set ANTHROPIC_API_KEY, or enable Claude Pro mode.")
                    return
            self._stop_flag = False
            self._pause_event.set()
            self.state.running = True
            self.state.paused = False
            self.state.start_time = datetime.now(timezone.utc).isoformat()
            self._pipeline_task = asyncio.create_task(self._pipeline_loop())
        elif action == "pause" and self.state.running and not self.state.paused:
            self._pause_event.clear()
            self.state.paused = True
            await self._broadcast_log("info", "Paused — will stop after current stage completes.")
        elif action == "resume" and self.state.running and self.state.paused:
            self._pause_event.set()
            self.state.paused = False
            await self._broadcast_log("info", "Resumed.")
        elif action == "stop":
            self._stop_flag = True
            self._pause_event.set()
            if self._pipeline_task:
                self._pipeline_task.cancel()
            self.state.running = False
            self.state.paused = False
            await self._broadcast_log("info", "Stopped.")
        await self._broadcast_state()

    # ── Pipeline ───────────────────────────────────────────────
    async def _check_pause_or_stop(self) -> None:
        await self._pause_event.wait()
        if self._stop_flag:
            raise asyncio.CancelledError("FABS stopped by user")

    async def _pipeline_loop(self) -> None:
        batch_size = self._config.get("batch_size", 0)
        processed = 0
        try:
            while not self._stop_flag:
                await self._check_pause_or_stop()
                if not self._pending:
                    await self._broadcast_log("info", "Queue exhausted. All pending topics done.")
                    break
                if batch_size > 0 and processed >= batch_size:
                    await self._broadcast_log("info", f"Batch of {batch_size} posts completed.")
                    break

                topic = self._pending[0]
                slug = topic["slug"]
                entry = PostEntry(slug=slug, title=topic["title"], status="queued",
                                  category=topic.get("blog_category", ""))
                self.state.in_progress.insert(0, entry.to_dict())
                self.state.current_slug = slug
                self.state.stats["writing_count"] = 1
                await self._broadcast_state()

                try:
                    await self._run_topic(topic)
                    self._pending.pop(0)
                    self.state.in_progress = [p for p in self.state.in_progress if p["slug"] != slug]
                    entry.status = "done"
                    self.state.completed.insert(0, entry.to_dict())
                    self.state.stats["published_count"] += 1
                    self.state.stats["pending_count"] = len(self._pending)
                    self._refresh_preview()
                    self._update_queue_status(slug, "done")
                    await self._broadcast_log("info", f"Published: {topic['title']}")
                    processed += 1
                except asyncio.CancelledError:
                    raise
                except Exception as exc:
                    await self._broadcast_log("error", f"Failed [{slug}]: {exc}")
                    self._pending.pop(0)
                    self.state.in_progress = [p for p in self.state.in_progress if p["slug"] != slug]
                    entry.status = "failed"
                    entry.error = str(exc)
                    self.state.failed.insert(0, entry.to_dict())
                    self._update_queue_status(slug, "failed")

                self.state.current_slug = None
                self.state.current_stage = None
                self.state.stats["writing_count"] = 0
                await self._broadcast_state()
        except asyncio.CancelledError:
            pass
        finally:
            self.state.running = False
            self.state.paused = False
            self.state.current_slug = None
            self.state.current_stage = None
            self.state.stats["writing_count"] = 0
            await self._broadcast_state()

    async def _run_topic(self, topic: dict) -> None:
        slug = topic["slug"]
        partial: dict[str, str] = {}

        async def set_stage(stage: str) -> None:
            self.state.current_stage = stage
            self.state.in_progress = [
                {**p, "status": stage} if p["slug"] == slug else p
                for p in self.state.in_progress
            ]
            await self._broadcast_stage_start(stage, slug)
            await self._broadcast_state()
            await self._check_pause_or_stop()

        try:
            await set_stage("research")
            research = await self._stream_call(
                self.state.model_config.get("research", DEFAULT_MODELS["research"]),
                RESEARCH_PROMPT.format(
                    title=topic["title"],
                    brief=topic.get("brief", ""),
                    blog_category=topic.get("blog_category", ""),
                ), "research", 2048,
            )
            partial["research"] = research

            await set_stage("outline")
            outline = await self._stream_call(
                self.state.model_config.get("outline", DEFAULT_MODELS["outline"]),
                OUTLINE_PROMPT.format(title=topic["title"], research=research),
                "outline", 1024,
            )
            partial["outline"] = outline

            await set_stage("write")
            article = await self._stream_call(
                self.state.model_config.get("write", DEFAULT_MODELS["write"]),
                WRITE_PROMPT.format(
                    title=topic["title"],
                    blog_category=topic.get("blog_category", ""),
                    tags_text=" / ".join(topic.get("tags", [])),
                    brief=topic.get("brief", ""),
                    outline=outline,
                    research=research,
                ), "write", 8192,
            )
            partial["article"] = article

            await set_stage("review")
            review = await self._run_review(article)
            if not review.get("pass", False):
                issues = review.get("issues", [])
                await self._broadcast_log("warn", f"Review issues [{slug}]: {', '.join(issues)}")

            if self.state.model_config.get("refine_enabled", False):
                await set_stage("refine")
                article = await self._stream_call(
                    self.state.model_config.get("refine", DEFAULT_MODELS["refine"]),
                    REFINE_PROMPT.format(title=topic["title"], article_text=article),
                    "refine", 8192,
                )
                partial["article_refined"] = article

            await set_stage("publish")
            await self._publish(topic, article, review.get("summary", ""))

        except asyncio.CancelledError:
            raise
        except Exception:
            FAILED_DIR.mkdir(parents=True, exist_ok=True)
            for stage_name, content in partial.items():
                (FAILED_DIR / f"{slug}.{stage_name}.md").write_text(content, encoding="utf-8")
            raise

    # ── Streaming dispatch ─────────────────────────────────────
    async def _stream_call(self, model: str, prompt: str, stage: str, max_tokens: int) -> str:
        """Dispatch to CLI or API streaming based on current mode."""
        if self._config.get("use_cli", False):
            return await self._stream_call_cli(model, prompt, stage)
        return await self._stream_call_api(model, prompt, stage, max_tokens)

    # ── Anthropic API streaming ────────────────────────────────
    async def _stream_call_api(self, model: str, prompt: str, stage: str, max_tokens: int) -> str:
        chunks: list[str] = []
        async with self._client.messages.stream(
            model=model, max_tokens=max_tokens,
            messages=[{"role": "user", "content": prompt}],
        ) as stream:
            async for text in stream.text_stream:
                if self._stop_flag:
                    raise asyncio.CancelledError("Stopped during stream")
                chunks.append(text)
                await self._broadcast_chunk(stage, text)
            final = await stream.get_final_message()
            inp, out = final.usage.input_tokens, final.usage.output_tokens

        cost = self._token_cost(model, inp, out)
        async with self._lock:
            self.state.stats["total_tokens"] += inp + out
            self.state.stats["total_cost_usd"] += cost

        return "".join(chunks)

    # ── Claude Pro CLI streaming ───────────────────────────────
    async def _stream_call_cli(self, model: str, prompt: str, stage: str) -> str:
        """Call the 'claude' CLI (Claude Code) subprocess and stream its stdout.

        Uses the user's Claude Pro subscription — no API tokens consumed.
        Model name date-suffixes are stripped (e.g. claude-haiku-4-5-20251001
        → claude-haiku-4-5) for CLI compatibility.
        """
        cli_model = re.sub(r"-\d{8}$", "", model)  # strip date suffix
        # Unset CLAUDECODE so the CLI doesn't refuse to run inside another Claude session
        cli_env = {k: v for k, v in os.environ.items() if k != "CLAUDECODE"}
        proc = await asyncio.create_subprocess_exec(
            CLI_BINARY, "-p", prompt, "--model", cli_model,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
            env=cli_env,
        )
        assert proc.stdout is not None
        assert proc.stderr is not None
        chunks: list[str] = []
        while True:
            chunk = await proc.stdout.read(512)
            if not chunk:
                break
            if self._stop_flag:
                proc.kill()
                await proc.wait()
                raise asyncio.CancelledError("Stopped during CLI stream")
            text = chunk.decode("utf-8", errors="replace")
            chunks.append(text)
            await self._broadcast_chunk(stage, text)
        await proc.wait()
        if proc.returncode != 0:
            err_bytes = await proc.stderr.read()
            err_text = err_bytes.decode("utf-8", errors="replace")[:400]
            raise RuntimeError(f"claude CLI exited {proc.returncode}: {err_text}")
        return "".join(chunks)

    async def _run_review(self, article_text: str) -> dict:
        model = self.state.model_config.get("review", DEFAULT_MODELS["review"])
        raw = await self._stream_call(model, REVIEW_PROMPT.format(article_text=article_text), "review", 1024)
        match = re.search(r'\{[^{}]*"pass"[^{}]*\}', raw, re.DOTALL)
        if match:
            try:
                return json.loads(match.group())
            except json.JSONDecodeError:
                pass
        return {"pass": False, "score": 0, "issues": ["Review parse failed"], "summary": ""}

    # ── Publish ────────────────────────────────────────────────
    async def _publish(self, topic: dict, article_text: str, review_summary: str) -> None:
        slug = topic["slug"]
        summary, content = self._extract_summary(article_text)
        if not summary:
            summary = review_summary or topic["title"]
        if len(summary) > 60:
            summary = summary[:59] + "…"

        raw_tags: list[str] = topic.get("tags", [])
        category: str = topic.get("blog_category", raw_tags[0] if raw_tags else "工程实践")
        tags = [category] + [t for t in raw_tags if t != category]

        POSTS_DIR.mkdir(parents=True, exist_ok=True)
        (POSTS_DIR / f"{slug}.md").write_text(content.strip() + "\n", encoding="utf-8")

        entry = {
            "title": topic["title"], "slug": slug,
            "date": datetime.now().strftime("%Y-%m-%d"),
            "author": "both", "tags": tags, "summary": summary,
        }
        posts: list[dict] = json.loads(POSTS_JSON.read_text(encoding="utf-8"))
        posts = [p for p in posts if p.get("slug") != slug]
        posts.insert(0, entry)
        POSTS_JSON.write_text(json.dumps(posts, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    @staticmethod
    def _extract_summary(text: str) -> tuple[str, str]:
        lines = text.splitlines()
        for i in range(len(lines) - 1, max(len(lines) - 10, -1), -1):
            line = lines[i].strip()
            if not line:
                continue
            if not line.startswith("{"):
                break
            try:
                obj = json.loads(line)
                if isinstance(obj.get("summary"), str):
                    remaining = lines[:i] + [l for l in lines[i + 1:] if l.strip()]
                    return obj["summary"].strip(), "\n".join(remaining)
            except json.JSONDecodeError:
                break
        return "", text

    def _resolve_prereqs(self, prereqs: list[str]) -> str:
        if not prereqs:
            return ""
        lookup = {t["slug"]: t.get("title", t["slug"]) for t in self._pending}
        return "、".join(lookup.get(s, s) for s in prereqs)

    @staticmethod
    def _token_cost(model: str, inp: int, out: int) -> float:
        rates = MODEL_COSTS.get(model, MODEL_COSTS["claude-sonnet-4-6"])
        return inp * rates[0] + out * rates[1]

    # ── Timer tick ─────────────────────────────────────────────
    async def tick(self) -> None:
        while True:
            await asyncio.sleep(1)
            if self.state.running and self.state.start_time:
                try:
                    start = datetime.fromisoformat(self.state.start_time)
                    now = datetime.now(timezone.utc)
                    if start.tzinfo is None:
                        start = start.replace(tzinfo=timezone.utc)
                    self.state.stats["elapsed_seconds"] = int((now - start).total_seconds())
                    await self._broadcast({"type": "tick", "elapsed": self.state.stats["elapsed_seconds"]})
                except Exception:
                    pass


# ──────────────────────────────────────────────────────────────
# FastAPI app
# ──────────────────────────────────────────────────────────────
app = FastAPI(title="FABS")
manager: FABSManager


@app.on_event("startup")
async def startup_event() -> None:
    global manager
    manager = FABSManager()
    asyncio.create_task(manager.tick())
    key_status = "configured" if manager.state.api_key_configured else "NOT SET — open Settings in the dashboard"
    cli_status = f"found at {CLI_BINARY}" if manager.state.cli_configured else "NOT FOUND"
    pend = len(manager._pending)
    print(f"FABS ready. Open http://127.0.0.1:8765")
    print(f"API key:      {key_status}")
    print(f"Claude CLI:   {cli_status}")
    print(f"Mode:         {'Claude Pro (CLI)' if manager.state.use_cli else 'Anthropic API'}")
    if pend:
        print(f"Pending topics: {pend} (loaded from {QUEUE_FILE.name})")
    else:
        print(f"Pending topics: 0 — upload a topic_queue.json via the dashboard Queue panel")


@app.get("/")
async def dashboard() -> HTMLResponse:
    return HTMLResponse(
        DASHBOARD_HTML.read_text(encoding="utf-8"),
        headers={"Cache-Control": "no-store, no-cache, must-revalidate, max-age=0"},
    )


@app.post("/api/control/{action}")
async def control(action: str) -> dict:
    if action not in ("start", "pause", "resume", "stop"):
        return {"error": "Invalid action"}
    await manager.control(action)
    return {"ok": True}


@app.get("/api/queue")
async def get_queue() -> dict:
    return manager.state.to_dict()


@app.get("/api/settings")
async def get_settings() -> dict:
    return manager.get_settings()


@app.post("/api/settings")
async def post_settings(payload: dict) -> dict:
    await manager.update_settings(payload)
    return {"ok": True, "api_key_configured": manager.state.api_key_configured}


@app.post("/api/upload-queue")
async def upload_queue(request: Request) -> dict:
    """Accept JSON body (array) and replace the in-memory pending queue.

    Client sends: Content-Type: application/json, body = raw JSON array.
    No python-multipart required.
    """
    try:
        raw = await request.json()
    except Exception as exc:
        return {"ok": False, "error": f"Invalid JSON body: {exc}"}
    ok, msg = await manager.load_queue_from_upload(raw)
    if ok:
        return {"ok": True, "message": msg}
    return {"ok": False, "error": msg}


@app.post("/api/model")
async def set_model(payload: dict) -> dict:
    stage = payload.get("stage")
    model = payload.get("model")
    valid_models = [m for m, _ in AVAILABLE_MODELS]
    if stage and model and model in valid_models:
        manager.state.model_config[stage] = model
    if "refine_enabled" in payload:
        manager.state.model_config["refine_enabled"] = bool(payload["refine_enabled"])
    await manager._broadcast_state()
    return {"ok": True}


@app.websocket("/ws")
async def websocket_endpoint(ws: WebSocket) -> None:
    await ws.accept()
    manager._ws_clients.add(ws)
    await ws.send_json({"type": "state_update", "data": manager.state.to_dict()})
    try:
        while True:
            data = await ws.receive_json()
            action = data.get("action")
            if action in ("start", "pause", "resume", "stop"):
                await manager.control(action)
            elif action == "set_model":
                stage, model = data.get("stage"), data.get("model")
                valid = [m for m, _ in AVAILABLE_MODELS]
                if stage and model and model in valid:
                    manager.state.model_config[stage] = model
                await manager._broadcast_state()
            elif action == "toggle_refine":
                manager.state.model_config["refine_enabled"] = bool(data.get("enabled", False))
                await manager._broadcast_state()
            elif action == "save_settings":
                await manager.update_settings(data.get("settings", {}))
    except WebSocketDisconnect:
        manager._ws_clients.discard(ws)
    except Exception:
        manager._ws_clients.discard(ws)


# ──────────────────────────────────────────────────────────────
# Entry point
# ──────────────────────────────────────────────────────────────
if __name__ == "__main__":
    import uvicorn
    uvicorn.run("fabs_server:app", host="127.0.0.1", port=8765, reload=False)
