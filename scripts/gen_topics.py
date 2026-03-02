#!/usr/bin/env python3
"""
gen_topics.py — 批量生成博文主题并追加到 topic_queue.json

用法：
  python3 scripts/gen_topics.py              # 生成直到队列达到 540 个
  python3 scripts/gen_topics.py --target 200  # 指定目标数量
  python3 scripts/gen_topics.py --dry-run     # 预览将要生成的批次
"""

import json
import os
import re
import subprocess
import sys
import argparse
from pathlib import Path

REPO_ROOT   = Path(__file__).parent.parent
QUEUE_FILE  = REPO_ROOT / "scripts" / "topic_queue.json"
CLAUDE_BIN  = "/Users/xuhongduo/.local/bin/claude"

# 每个批次：(主分类, 子话题描述, 数量)
BATCH_SPECS = [
    ("底层原理",   "FlashAttention IO感知算法、稀疏注意力（Longformer/BigBird）、线性注意力（Performer/Mamba）", 30),
    ("底层原理",   "混合专家模型MoE：路由机制、负载均衡、Switch Transformer、Mixtral设计", 30),
    ("底层原理",   "模型并行与流水线并行：Tensor Parallel、PP、ZeRO优化器、Megatron-LM", 30),
    ("底层原理",   "扩展定律（Chinchilla定律）、涌现能力（Emergence）、上下文学习机制", 30),
    ("底层原理",   "对比学习（SimCLR/SimCSE）、知识蒸馏、表示对齐、多模态Embedding", 30),
    ("模型解析",   "Claude系列（Claude 1/2/3/3.5）架构特点与Constitutional AI", 30),
    ("模型解析",   "Gemini系列（Gemini 1/1.5/2）、多模态架构与长上下文技术", 30),
    ("模型解析",   "DeepSeek系列（V1/V2/V3/R1）、MoE设计与推理强化训练", 30),
    ("模型解析",   "Mistral系列、Phi系列小模型、Falcon、Yi、Qwen技术细节对比", 30),
    ("模型解析",   "代码大模型（Codex/CodeLlama/DeepSeekCoder）、多语言模型、数学推理模型", 30),
    ("智能体",     "ReAct框架、思维链（CoT）、自洽性（Self-Consistency）、ToT树状思维", 30),
    ("智能体",     "工具调用（Function Calling）、代码执行、浏览器控制、计算机使用", 30),
    ("智能体",     "Agent记忆系统（短期/长期/语义记忆）、向量数据库集成、外部知识管理", 30),
    ("智能体",     "多Agent系统：角色分工、通信协议、协调机制、AutoGen/CrewAI架构", 30),
    ("智能体",     "Agent评测（GAIA/SWE-bench/AgentBench）、自我反思、错误恢复、安全边界", 30),
    ("工程实践",   "vLLM PagedAttention、TensorRT-LLM、连续批处理、推理服务优化", 30),
    ("工程实践",   "生产部署：灰度发布、AB测试、延迟监控、成本核算、模型版本管理", 30),
    ("工程实践",   "合成数据生成、数据清洗Pipeline、评估指标设计、SFT数据配比策略", 30),
]

PROMPT_TEMPLATE = """你是技术博客主题策划编辑，专注于大模型技术（面向开发者和研究者）。

请生成 {count} 个技术博文主题，聚焦方向：{focus}

要求：
- 每个主题是一个具体技术点，能写成1500-3000字的深度文章
- 主分类必须是：{category}
- 标题简洁（中文，不超过20字）
- 与以下已有主题不重复：
{existing_sample}

直接输出纯 JSON 数组（不要加代码块标记），每个元素格式：
{{
  "title": "文章标题",
  "slug": "url-friendly-english-slug",
  "tags": ["{category}", "子标签1", "子标签2"],
  "brief": "核心要点：具体技术内容，包含关键公式或数值，100-150字",
  "depth_hint": "需要包含的具体推导、代码或实验数据，30-60字"
}}

只输出 JSON 数组，不要任何说明文字。"""


def read_queue() -> list[dict]:
    with open(QUEUE_FILE, encoding="utf-8") as f:
        return json.load(f)


def write_queue(queue: list[dict]) -> None:
    with open(QUEUE_FILE, "w", encoding="utf-8") as f:
        json.dump(queue, f, ensure_ascii=False, indent=2)


def call_claude(prompt: str) -> str:
    env = os.environ.copy()
    env.pop("CLAUDECODE", None)          # 允许在 Claude Code 会话内嵌套调用
    env.pop("CLAUDE_CODE_ENTRYPOINT", None)
    result = subprocess.run(
        [CLAUDE_BIN, "-p", prompt, "--model", "claude-opus-4-6"],
        capture_output=True, text=True, timeout=300, env=env,
    )
    if result.returncode != 0:
        raise RuntimeError(f"claude 调用失败: {result.stderr.strip()[:200]}")
    return result.stdout.strip()


def parse_topics(output: str) -> list[dict]:
    """从 claude 输出中提取 JSON 数组。"""
    match = re.search(r'\[[\s\S]*\]', output)
    if not match:
        raise ValueError(f"未找到 JSON 数组，输出前200字:\n{output[:200]}")
    return json.loads(match.group())


def generate_batch(category: str, focus: str, count: int, existing_titles: list[str]) -> list[dict]:
    sample = "\n".join(f"- {t}" for t in existing_titles[-40:])
    prompt = PROMPT_TEMPLATE.format(
        count=count,
        focus=focus,
        category=category,
        existing_sample=sample,
    )
    output = call_claude(prompt)
    return parse_topics(output)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--target",  type=int, default=540, help="目标主题总数（默认540）")
    parser.add_argument("--dry-run", action="store_true",  help="预览批次，不实际调用 API")
    args = parser.parse_args()

    queue = read_queue()
    needed = args.target - len(queue)

    if needed <= 0:
        print(f"队列已有 {len(queue)} 个主题，已达目标 {args.target}，无需生成。")
        return

    print(f"当前 {len(queue)} 个，目标 {args.target}，需生成 {needed} 个。")

    if args.dry_run:
        print("\n将按以下批次生成：")
        total = 0
        for i, (cat, focus, n) in enumerate(BATCH_SPECS, 1):
            actual = min(n, needed - total)
            if actual <= 0:
                break
            print(f"  批次{i}: [{cat}] {focus[:50]}... × {actual}")
            total += actual
        return

    existing_slugs  = {t["slug"] for t in queue}
    existing_titles = [t["title"] for t in queue]
    next_id         = max(t["id"] for t in queue) + 1
    generated       = 0

    for batch_idx, (category, focus, count) in enumerate(BATCH_SPECS, 1):
        remaining = needed - generated
        if remaining <= 0:
            break

        actual_count = min(count, remaining)
        print(f"\n[批次 {batch_idx}/{len(BATCH_SPECS)}] [{category}] {focus[:40]}... → 生成 {actual_count} 个")

        try:
            topics = generate_batch(category, focus, actual_count, existing_titles)
        except Exception as e:
            print(f"  生成失败: {e}")
            continue

        added = 0
        for t in topics:
            slug = t.get("slug", "").strip().lower()
            if not slug or slug in existing_slugs:
                continue
            # 确保 tags[0] 是主分类
            tags = t.get("tags", [category])
            if not tags or tags[0] != category:
                tags = [category] + [x for x in tags if x != category]

            queue.append({
                "id":         next_id,
                "title":      t.get("title", "").strip(),
                "slug":       slug,
                "tags":       tags,
                "brief":      t.get("brief", "").strip(),
                "depth_hint": t.get("depth_hint", "").strip(),
                "status":     "pending",
            })
            existing_slugs.add(slug)
            existing_titles.append(t.get("title", ""))
            next_id += 1
            added   += 1

        write_queue(queue)
        generated += added
        print(f"  已添加 {added} 个，队列总计 {len(queue)} 个（目标 {args.target}）")

    print(f"\n完成！生成 {generated} 个新主题，队列共 {len(queue)} 个。")


if __name__ == "__main__":
    main()
