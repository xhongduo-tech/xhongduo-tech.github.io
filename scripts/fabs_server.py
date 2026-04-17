"""
FABS — Fully Automated Blog-post System
FastAPI + WebSocket backend, Codex CLI pipeline.
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
import tempfile
from contextlib import asynccontextmanager
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

from fastapi import FastAPI, Request, WebSocket, WebSocketDisconnect
from fastapi.responses import HTMLResponse

# Path to codex CLI
CODEX_BINARY: str = shutil.which("codex") or "codex"

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
    ("gpt-5.4-mini", "GPT-5.4 Mini (Cost Saver)"),
    ("gpt-5.4", "GPT-5.4 (High Quality)"),
]

DEFAULT_MODELS = {
    "research": "gpt-5.4-mini",
    "outline": "gpt-5.4-mini",
    "write": "gpt-5.4",
    "review": "gpt-5.4-mini",
    "refine": "gpt-5.4",
}

VALID_REASONING_EFFORTS = {"low", "medium", "high"}
STAGES = ["research", "outline", "write", "review", "refine", "publish"]
MODEL_STAGES = ["research", "outline", "write", "refine"]


# ──────────────────────────────────────────────────────────────
# Config file helpers
# ──────────────────────────────────────────────────────────────
def load_config() -> dict:
    config: dict = {
        "batch_size": 0,  # 0 = unlimited
        "codex_reasoning_effort": "medium",
        "auto_git_push": True,
        "git_push_remote": "origin",
        "git_push_ref": "HEAD",
        "min_content_chars": 2200,
        "concurrent_workers": 1,
    }
    if CONFIG_FILE.exists():
        try:
            saved = json.loads(CONFIG_FILE.read_text(encoding="utf-8"))
            config.update(saved)
        except Exception:
            pass

    try:
        config["batch_size"] = max(0, int(config.get("batch_size", 0)))
    except Exception:
        config["batch_size"] = 0

    effort = str(config.get("codex_reasoning_effort", "medium")).strip().lower()
    config["codex_reasoning_effort"] = effort if effort in VALID_REASONING_EFFORTS else "medium"
    config["auto_git_push"] = bool(config.get("auto_git_push", True))
    config["git_push_remote"] = str(config.get("git_push_remote", "origin")).strip() or "origin"
    config["git_push_ref"] = str(config.get("git_push_ref", "HEAD")).strip() or "HEAD"
    try:
        config["min_content_chars"] = max(1200, int(config.get("min_content_chars", 2200)))
    except Exception:
        config["min_content_chars"] = 2200

    for stage in MODEL_STAGES:
        model = config.get(stage)
        valid = [m for m, _ in AVAILABLE_MODELS]
        if model in valid:
            continue
        config[stage] = DEFAULT_MODELS[stage]

    config["refine_enabled"] = bool(config.get("refine_enabled", False))
    try:
        config["concurrent_workers"] = max(1, int(config.get("concurrent_workers", 1)))
    except Exception:
        config["concurrent_workers"] = 1
    return config


def save_config(config: dict) -> None:
    to_save = dict(config)
    CONFIG_FILE.write_text(json.dumps(to_save, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


# ──────────────────────────────────────────────────────────────
# Prompt templates
# ──────────────────────────────────────────────────────────────
RESEARCH_PROMPT = """\
Topic: {title}
Brief: {brief}
Category: {blog_category}

输出中文研究笔记（控制在 8 条以内）：
1) 一句话核心定义
2) 面向新手的直观解释（2-3 句）
3) 关键公式/机制（符号统一）
4) 一个最小数值例子
5) 一个真实工程场景
6) 常见坑与规避
7) 参考来源（至少 3 条，优先论文/官方文档/源码）

要求：短句、可复核、避免空话。"""

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

对每章输出：
- 关键点（2 条）
- 必须示例（至少 1 个，含新手可理解版本）
- 必要元素（公式/表格/代码）

总要求：由浅入深，避免术语堆砌。"""

WRITE_PROMPT = """\
你是一位技术博客作者，目标读者是“零基础到初级工程师”，但技术结论必须准确。

只输出文章正文，不输出前言、解释、道歉、过程说明。

严格使用以下章节顺序：
## 核心结论
## 问题定义与边界
## 核心机制与推导
## 代码实现
## 工程权衡与常见坑
## 替代方案与适用边界
## 参考资料

格式要求：
- 从 `##` 开始，不写 `#`（H1 由系统自动生成）
- 章节之间使用 `---`
- 至少包含一个可运行的 `python` 代码块（含 `assert`）
- 至少包含一个 Markdown 表格
- 至少包含数学公式 $...$ 或 $$...$$
- 至少包含 1 个“玩具例子”与 1 个“真实工程例子”
- 术语首次出现时用一句白话解释
- 正文（不含参考资料）建议 2200-4200 中文字
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

REFINE_PROMPT = """\
在保持结构、标题、章节顺序和核心结论不变的前提下，改写并补全文章。

重点修复：
1) 章节缺失、代码不可运行、表格/公式/参考资料不足
2) 篇幅与深度不够
3) 对新手不友好（术语堆砌、例子不足）

Title: {title}

Article:
{article_text}

输出完整改写稿，从 `##` 开始，保留末尾 summary JSON。"""


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
        return {
            "slug": self.slug,
            "title": self.title,
            "status": self.status,
            "category": self.category,
            "error": self.error,
        }


@dataclass
class FABSState:
    running: bool = False
    paused: bool = False
    current_slug: Optional[str] = None
    current_stage: Optional[str] = None
    in_progress: list[dict] = field(default_factory=list)
    completed: list[dict] = field(default_factory=list)
    failed: list[dict] = field(default_factory=list)
    pending_preview: list[dict] = field(default_factory=list)
    workers: list[dict] = field(default_factory=list)  # active concurrent workers
    model_config: dict = field(default_factory=lambda: dict(DEFAULT_MODELS, refine_enabled=False))
    codex_configured: bool = False
    queue_loaded: bool = False
    stats: dict = field(
        default_factory=lambda: {
            "total_tokens": 0,
            "total_cost_usd": 0.0,
            "elapsed_seconds": 0,
            "pending_count": 0,
            "writing_count": 0,
            "published_count": 0,
        }
    )
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
            "workers": self.workers,
            "model_config": self.model_config,
            "codex_configured": self.codex_configured,
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
        self._worker_tasks: set[asyncio.Task] = set()
        self._ws_clients: set[WebSocket] = set()
        self._sync_runtime_state()
        self._load_queue()

    def _sync_runtime_state(self) -> None:
        self.state.codex_configured = shutil.which("codex") is not None
        self.state.model_config = dict(
            DEFAULT_MODELS,
            refine_enabled=bool(self._config.get("refine_enabled", False)),
        )
        for stage in MODEL_STAGES:
            model = self._config.get(stage, DEFAULT_MODELS[stage])
            valid = [m for m, _ in AVAILABLE_MODELS]
            self.state.model_config[stage] = model if model in valid else DEFAULT_MODELS[stage]

        effort = str(self._config.get("codex_reasoning_effort", "medium")).strip().lower()
        if effort not in VALID_REASONING_EFFORTS:
            effort = "medium"
        self._config["codex_reasoning_effort"] = effort
        self.state.model_config["concurrent_workers"] = max(1, int(self._config.get("concurrent_workers", 1)))

    async def close(self) -> None:
        self._stop_flag = True
        self._pause_event.set()
        for t in list(self._worker_tasks):
            t.cancel()
        if self._pipeline_task:
            self._pipeline_task.cancel()
            try:
                await self._pipeline_task
            except Exception:
                pass

    # ── Config & Settings ──────────────────────────────────────
    async def update_settings(self, settings: dict) -> None:
        if "batch_size" in settings:
            try:
                self._config["batch_size"] = max(0, int(settings.get("batch_size", 0)))
            except Exception:
                self._config["batch_size"] = 0

        if "codex_reasoning_effort" in settings:
            effort = str(settings.get("codex_reasoning_effort", "medium")).strip().lower()
            self._config["codex_reasoning_effort"] = effort if effort in VALID_REASONING_EFFORTS else "medium"

        if "auto_git_push" in settings:
            self._config["auto_git_push"] = bool(settings["auto_git_push"])
        if "git_push_remote" in settings:
            remote = str(settings.get("git_push_remote", "origin")).strip()
            self._config["git_push_remote"] = remote or "origin"
        if "git_push_ref" in settings:
            ref = str(settings.get("git_push_ref", "HEAD")).strip()
            self._config["git_push_ref"] = ref or "HEAD"
        if "min_content_chars" in settings:
            try:
                self._config["min_content_chars"] = max(1200, int(settings.get("min_content_chars", 2200)))
            except Exception:
                self._config["min_content_chars"] = 2200

        valid_models = [m for m, _ in AVAILABLE_MODELS]
        for stage in MODEL_STAGES:
            if stage in settings and settings[stage] in valid_models:
                self._config[stage] = settings[stage]
                self.state.model_config[stage] = settings[stage]

        if "refine_enabled" in settings:
            enabled = bool(settings["refine_enabled"])
            self._config["refine_enabled"] = enabled
            self.state.model_config["refine_enabled"] = enabled

        if "concurrent_workers" in settings:
            try:
                self._config["concurrent_workers"] = max(1, int(settings["concurrent_workers"]))
            except Exception:
                self._config["concurrent_workers"] = 1

        save_config(self._config)
        self._sync_runtime_state()
        await self._broadcast_state()

    def get_settings(self) -> dict:
        payload = {
            "batch_size": self._config.get("batch_size", 0),
            "codex_reasoning_effort": self._config.get("codex_reasoning_effort", "medium"),
            "codex_configured": self.state.codex_configured,
            "codex_binary": CODEX_BINARY if self.state.codex_configured else None,
            "model_config": self.state.model_config,
            "refine_enabled": bool(self.state.model_config.get("refine_enabled", False)),
            "auto_git_push": bool(self._config.get("auto_git_push", True)),
            "git_push_remote": self._config.get("git_push_remote", "origin"),
            "git_push_ref": self._config.get("git_push_ref", "HEAD"),
            "min_content_chars": int(self._config.get("min_content_chars", 2200)),
            "concurrent_workers": int(self._config.get("concurrent_workers", 1)),
        }
        for stage in MODEL_STAGES:
            payload[stage] = self.state.model_config.get(stage, DEFAULT_MODELS[stage])
        return payload

    # ── Queue ──────────────────────────────────────────────────
    def _load_queue(self) -> None:
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
        if not isinstance(raw, list):
            return False, "JSON must be an array of topic objects"
        valid = [t for t in raw if isinstance(t, dict) and t.get("slug") and t.get("title")]
        if not valid:
            return False, "No valid items found — each item must have at least 'slug' and 'title'"
        pending = [t for t in valid if t.get("status", "pending") not in ("done",)]
        if not pending:
            return False, "All items already have status 'done' — nothing to process"
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
        except Exception as exc:
            asyncio.create_task(self._broadcast_log("warn", f"Queue update failed for {slug}: {exc}"))

    async def _run_git_cmd(self, *args: str) -> tuple[int, str, str]:
        proc = await asyncio.create_subprocess_exec(
            "git",
            *args,
            cwd=str(REPO_ROOT),
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        out_b, err_b = await proc.communicate()
        out = out_b.decode("utf-8", errors="replace").strip()
        err = err_b.decode("utf-8", errors="replace").strip()
        return proc.returncode, out, err

    async def _git_commit_push_single(self, slug: str) -> None:
        await self._broadcast_log("info", f"[git] committing {slug}")
        add_rc, _, add_err = await self._run_git_cmd("add", "posts/posts.json", "scripts/topic_queue.json", f"posts/{slug}.md")
        if add_rc != 0:
            raise RuntimeError(f"git add failed: {add_err}")

        diff_rc, _, diff_err = await self._run_git_cmd("diff", "--cached", "--quiet")
        if diff_rc == 0:
            await self._broadcast_log("warn", f"[git] no staged changes for {slug}, skip commit/push")
            return
        if diff_rc not in (0, 1):
            raise RuntimeError(f"git diff --cached failed: {diff_err}")

        msg = (
            f"post: {slug}\n\n"
            "Generated with Codex staged pipeline.\n\n"
            "Co-Authored-By: Codex <noreply@openai.com>\n"
        )
        msg_file = tempfile.NamedTemporaryFile(prefix="fabs_git_msg_", suffix=".txt", delete=False)
        msg_file.write(msg.encode("utf-8"))
        msg_file.close()
        msg_path = Path(msg_file.name)

        try:
            commit_rc, _, commit_err = await self._run_git_cmd("commit", "-F", str(msg_path))
            if commit_rc != 0:
                raise RuntimeError(f"git commit failed: {commit_err}")

            remote = self._config.get("git_push_remote", "origin")
            ref = self._config.get("git_push_ref", "HEAD")
            push_rc, _, push_err = await self._run_git_cmd("push", remote, ref)
            if push_rc != 0:
                raise RuntimeError(f"git push {remote} {ref} failed: {push_err}")
        finally:
            try:
                msg_path.unlink(missing_ok=True)
            except Exception:
                pass

        await self._broadcast_log("info", f"[git] pushed {slug}")

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

    async def _broadcast_chunk(self, stage: str, text: str, worker_id: int = 0) -> None:
        await self._broadcast({"type": "stream_chunk", "stage": stage, "text": text, "worker_id": worker_id})

    async def _broadcast_stage_start(self, stage: str, slug: str, worker_id: int = 0) -> None:
        await self._broadcast({"type": "stage_start", "stage": stage, "slug": slug, "worker_id": worker_id})

    # ── Controls ───────────────────────────────────────────────
    async def control(self, action: str) -> None:
        if action == "start" and not self.state.running:
            if not self.state.codex_configured:
                await self._broadcast_log("error", "Codex CLI not found in PATH. Install Codex CLI first.")
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
            for t in list(self._worker_tasks):
                t.cancel()
            if self._pipeline_task:
                self._pipeline_task.cancel()
            self.state.running = False
            self.state.paused = False
            self.state.workers = []
            await self._broadcast_log("info", "Stopped.")
        await self._broadcast_state()

    # ── Pipeline ───────────────────────────────────────────────
    async def _check_pause_or_stop(self) -> None:
        await self._pause_event.wait()
        if self._stop_flag:
            raise asyncio.CancelledError("FABS stopped by user")

    # ── Worker state helpers ────────────────────────────────────
    def _set_worker(self, worker_id: int, slug: str, title: str, category: str, stage: str) -> None:
        self.state.workers = [w for w in self.state.workers if w["worker_id"] != worker_id]
        self.state.workers.append({"worker_id": worker_id, "slug": slug, "title": title, "category": category, "stage": stage})

    def _update_worker_stage(self, worker_id: int, stage: str) -> None:
        for w in self.state.workers:
            if w["worker_id"] == worker_id:
                w["stage"] = stage

    def _clear_worker(self, worker_id: int) -> None:
        self.state.workers = [w for w in self.state.workers if w["worker_id"] != worker_id]

    async def _pipeline_loop(self) -> None:
        n_workers = max(1, int(self._config.get("concurrent_workers", 1)))
        batch_size = self._config.get("batch_size", 0)
        total_processed = 0
        # Assign worker IDs 1..n_workers in a cyclic pool
        next_worker_id = 1

        async def run_worker(topic: dict, worker_id: int) -> None:
            nonlocal total_processed
            slug = topic["slug"]
            entry = PostEntry(
                slug=slug,
                title=topic["title"],
                status="queued",
                category=topic.get("blog_category", ""),
            )
            self.state.in_progress.insert(0, entry.to_dict())
            self.state.current_slug = slug
            self._set_worker(worker_id, slug, topic["title"], topic.get("blog_category", ""), "queued")
            self.state.stats["writing_count"] = len(self._worker_tasks)
            await self._broadcast_state()
            try:
                await self._run_topic(topic, worker_id)
                self._update_queue_status(slug, "done")
                if self._config.get("auto_git_push", True):
                    await self._git_commit_push_single(slug)
                self.state.in_progress = [p for p in self.state.in_progress if p["slug"] != slug]
                entry.status = "done"
                self.state.completed.insert(0, entry.to_dict())
                self.state.stats["published_count"] += 1
                self.state.stats["pending_count"] = len(self._pending)
                self._refresh_preview()
                total_processed += 1
                await self._broadcast_log("info", f"Published: {topic['title']}")
            except asyncio.CancelledError:
                raise
            except Exception as exc:
                await self._broadcast_log("error", f"Failed [{slug}]: {exc}")
                self.state.in_progress = [p for p in self.state.in_progress if p["slug"] != slug]
                entry.status = "failed"
                entry.error = str(exc)
                self.state.failed.insert(0, entry.to_dict())
                self._update_queue_status(slug, "failed")
            finally:
                self._clear_worker(worker_id)
                self._worker_tasks.discard(asyncio.current_task())  # type: ignore[arg-type]
                self.state.stats["writing_count"] = len(self._worker_tasks)
                if not self.state.workers:
                    self.state.current_slug = None
                    self.state.current_stage = None
                await self._broadcast_state()

        try:
            while not self._stop_flag:
                await self._check_pause_or_stop()

                # Clean up done tasks
                self._worker_tasks = {t for t in self._worker_tasks if not t.done()}

                if not self._pending:
                    if self._worker_tasks:
                        await asyncio.gather(*self._worker_tasks, return_exceptions=True)
                    await self._broadcast_log("info", "Queue exhausted. All pending topics done.")
                    break

                if batch_size > 0 and total_processed >= batch_size:
                    if self._worker_tasks:
                        await asyncio.gather(*self._worker_tasks, return_exceptions=True)
                    await self._broadcast_log("info", f"Batch of {batch_size} posts completed.")
                    break

                if len(self._worker_tasks) >= n_workers:
                    # Wait for any one worker to free up, then re-loop
                    await asyncio.wait(self._worker_tasks, return_when=asyncio.FIRST_COMPLETED)
                    continue

                topic = self._pending.pop(0)
                worker_id = next_worker_id
                next_worker_id = (next_worker_id % n_workers) + 1
                task = asyncio.create_task(run_worker(topic, worker_id), name=f"fabs-worker-{worker_id}")
                self._worker_tasks.add(task)

        except asyncio.CancelledError:
            for t in list(self._worker_tasks):
                t.cancel()
            if self._worker_tasks:
                await asyncio.gather(*self._worker_tasks, return_exceptions=True)
        finally:
            self._worker_tasks.clear()
            self.state.running = False
            self.state.paused = False
            self.state.current_slug = None
            self.state.current_stage = None
            self.state.workers = []
            self.state.stats["writing_count"] = 0
            await self._broadcast_state()

    async def _run_topic(self, topic: dict, worker_id: int = 0) -> None:
        slug = topic["slug"]
        partial: dict[str, str] = {}

        async def set_stage(stage: str) -> None:
            self.state.current_stage = stage
            self.state.in_progress = [
                {**p, "status": stage} if p["slug"] == slug else p
                for p in self.state.in_progress
            ]
            self._update_worker_stage(worker_id, stage)
            await self._broadcast_stage_start(stage, slug, worker_id)
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
                ),
                "research",
                2048,
                worker_id,
            )
            partial["research"] = research

            await set_stage("outline")
            outline = await self._stream_call(
                self.state.model_config.get("outline", DEFAULT_MODELS["outline"]),
                OUTLINE_PROMPT.format(title=topic["title"], research=research),
                "outline",
                1024,
                worker_id,
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
                ),
                "write",
                8192,
                worker_id,
            )
            partial["article"] = article

            await set_stage("review")
            review = self._run_review_local(article)
            if not review.get("pass", False):
                issues = review.get("issues", [])
                await self._broadcast_log("warn", f"Review issues [{slug}]: {', '.join(issues)}")

            needs_refine = (not review.get("pass", False)) or self.state.model_config.get("refine_enabled", False)
            if needs_refine:
                await set_stage("refine")
                article = await self._stream_call(
                    self.state.model_config.get("refine", DEFAULT_MODELS["refine"]),
                    REFINE_PROMPT.format(title=topic["title"], article_text=article),
                    "refine",
                    8192,
                    worker_id,
                )
                partial["article_refined"] = article
                review = self._run_review_local(article)
                if not review.get("pass", False):
                    raise RuntimeError(f"Refine 后仍不达标: {', '.join(review.get('issues', []))}")

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
    async def _stream_call(self, model: str, prompt: str, stage: str, max_tokens: int, worker_id: int = 0) -> str:
        _ = max_tokens  # retained for signature compatibility
        return await self._stream_call_codex(model, prompt, stage, worker_id)

    # ── Codex CLI call ─────────────────────────────────────────
    async def _stream_call_codex(self, model: str, prompt: str, stage: str, worker_id: int = 0) -> str:
        effort = self._config.get("codex_reasoning_effort", "medium")
        tmp_output = tempfile.NamedTemporaryFile(prefix="fabs_codex_", suffix=".txt", delete=False)
        tmp_output.close()
        out_path = Path(tmp_output.name)

        cmd = [
            CODEX_BINARY,
            "exec",
            "--model",
            model,
            "-c",
            f'model_reasoning_effort="{effort}"',
            "--skip-git-repo-check",
            "--color",
            "never",
            "-s",
            "workspace-write",
            "-C",
            str(REPO_ROOT),
            "--output-last-message",
            str(out_path),
            prompt,
        ]

        proc = await asyncio.create_subprocess_exec(
            *cmd,
            stdout=asyncio.subprocess.DEVNULL,
            stderr=asyncio.subprocess.PIPE,
        )
        assert proc.stderr is not None

        async def _drain_stderr() -> str:
            chunks: list[bytes] = []
            while True:
                chunk = await proc.stderr.read(4096)
                if not chunk:
                    break
                chunks.append(chunk)
            return b"".join(chunks).decode("utf-8", errors="replace").strip()

        stderr_task = asyncio.create_task(_drain_stderr())

        while proc.returncode is None:
            if self._stop_flag:
                proc.kill()
                await proc.wait()
                stderr_task.cancel()
                raise asyncio.CancelledError("Stopped during codex call")
            await asyncio.sleep(0.15)

        stderr_text = await stderr_task

        try:
            if proc.returncode != 0:
                short = "\n".join(stderr_text.splitlines()[:4]) if stderr_text else "unknown error"
                raise RuntimeError(f"codex exited {proc.returncode}: {short}")

            text = out_path.read_text(encoding="utf-8").strip()
            if not text:
                raise RuntimeError("codex returned empty output")
            await self._broadcast_chunk(stage, text, worker_id)
            return text
        finally:
            try:
                out_path.unlink(missing_ok=True)
            except Exception:
                pass

    def _run_review_local(self, article_text: str) -> dict:
        issues: list[str] = []
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
            if section not in article_text:
                issues.append(f"缺少章节：{section}")

        if "```python" not in article_text:
            issues.append("缺少可运行的 python 代码块")
        if "|" not in article_text:
            issues.append("缺少 Markdown 表格")
        if "$$" not in article_text and not re.search(r"\$[^$\n]+\$", article_text):
            issues.append("缺少公式或数学记号")

        refs_body = ""
        refs_match = re.search(r"^## 参考资料\s*$([\s\S]*)", article_text, flags=re.M)
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

        summary = ""
        for line in reversed(article_text.splitlines()):
            line = line.strip()
            if not line:
                continue
            if not line.startswith("{"):
                break
            try:
                obj = json.loads(line)
            except Exception:
                break
            if isinstance(obj.get("summary"), str) and obj["summary"].strip():
                summary = obj["summary"].strip()
                break
        if not summary:
            issues.append("缺少末尾 summary JSON")

        char_count = len(re.sub(r"\s+", "", article_text))
        min_chars = int(self._config.get("min_content_chars", 2200))
        if char_count < min_chars:
            issues.append(f"正文篇幅偏短（当前约 {char_count} 字符，要求至少 {min_chars}）")

        score = max(0, 100 - 10 * len(issues))
        return {"pass": not issues, "score": score, "issues": issues, "summary": summary}

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
            "title": topic["title"],
            "slug": slug,
            "date": datetime.now().strftime("%Y-%m-%d"),
            "author": "both",
            "tags": tags,
            "summary": summary,
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
                    remaining = lines[:i] + [l for l in lines[i + 1 :] if l.strip()]
                    return obj["summary"].strip(), "\n".join(remaining)
            except json.JSONDecodeError:
                break
        return "", text

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
# FastAPI app — lifespan (startup + graceful shutdown)
# ──────────────────────────────────────────────────────────────
manager: FABSManager


@asynccontextmanager
async def lifespan(app: FastAPI):
    global manager
    manager = FABSManager()
    asyncio.create_task(manager.tick())
    codex_status = f"found at {CODEX_BINARY}" if manager.state.codex_configured else "NOT FOUND"
    pend = len(manager._pending)
    print("FABS ready. Open http://127.0.0.1:8765")
    print(f"Codex CLI:    {codex_status}")
    print("Mode:         Codex staged models (mini + 5.4)")
    print(f"Git push:     {'enabled' if manager._config.get('auto_git_push', True) else 'disabled'}")
    if pend:
        print(f"Pending topics: {pend} (loaded from {QUEUE_FILE.name})")
    else:
        print("Pending topics: 0 — upload a topic_queue.json via the dashboard Queue panel")
    yield
    await manager.close()


app = FastAPI(title="FABS", lifespan=lifespan)


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


@app.get("/api/host-info")
async def host_info() -> dict:
    cpu = os.cpu_count() or 1
    ram_gb: Optional[float] = None
    try:
        import resource
        ram_gb = None  # resource module doesn't give total RAM
    except Exception:
        pass
    try:
        # macOS / Linux: read total memory
        with open("/proc/meminfo") as f:
            for line in f:
                if line.startswith("MemTotal:"):
                    ram_gb = int(line.split()[1]) / (1024 ** 2)
                    break
    except Exception:
        pass
    if ram_gb is None:
        try:
            import subprocess
            out = subprocess.check_output(["sysctl", "-n", "hw.memsize"], text=True).strip()
            ram_gb = int(out) / (1024 ** 3)
        except Exception:
            pass
    suggested = max(1, min(cpu - 1, 4))
    return {"cpu_cores": cpu, "ram_gb": round(ram_gb, 1) if ram_gb else None, "suggested": suggested}


@app.get("/api/queue")
async def get_queue() -> dict:
    return manager.state.to_dict()


@app.get("/api/settings")
async def get_settings() -> dict:
    return manager.get_settings()


@app.post("/api/settings")
async def post_settings(payload: dict) -> dict:
    await manager.update_settings(payload)
    return {"ok": True, "codex_configured": manager.state.codex_configured}


@app.post("/api/upload-queue")
async def upload_queue(request: Request) -> dict:
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
        manager._config[stage] = model
    if "refine_enabled" in payload:
        enabled = bool(payload["refine_enabled"])
        manager.state.model_config["refine_enabled"] = enabled
        manager._config["refine_enabled"] = enabled
    save_config(manager._config)
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
                    manager._config[stage] = model
                    save_config(manager._config)
                await manager._broadcast_state()
            elif action == "toggle_refine":
                enabled = bool(data.get("enabled", False))
                manager.state.model_config["refine_enabled"] = enabled
                manager._config["refine_enabled"] = enabled
                save_config(manager._config)
                await manager._broadcast_state()
            elif action == "save_settings":
                await manager.update_settings(data.get("settings", {}))
            elif action == "set_concurrent":
                try:
                    n = max(1, int(data.get("workers", 1)))
                except Exception:
                    n = 1
                manager._config["concurrent_workers"] = n
                manager.state.model_config["concurrent_workers"] = n
                save_config(manager._config)
                await manager._broadcast_state()
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
