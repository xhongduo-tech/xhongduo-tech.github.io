## 核心结论

Agent 的短期记忆，本质上就是**当前还能放进模型上下文窗口的 token 集合**。这里的 token 可以先粗略理解成“模型读取、计费、推理时使用的最小文本单位”。窗口一旦装满，系统不会自动把旧内容变成高质量记忆，而是只能做三件事：**丢弃、压缩、重组**。

对工程实现来说，主流只有三条路：

| 方案 | 做法 | 优点 | 主要代价 | 适合场景 |
| --- | --- | --- | --- | --- |
| 无压缩 | 全量保留历史 | 实现最简单，行为最直观 | 很快超限，成本随轮数近似线性上升 | 极短会话、一次性问答 |
| 滑动窗口 | 只保留最近 N 轮 | 延迟低，逻辑稳定，容易调试 | 早期目标、约束、偏好容易被忘 | 客服、短问答、轻工具调用 |
| 递归摘要 + 预算压缩 | 旧历史先压成摘要，再按类别分配 token | 能跨多轮保留目标、约束、决策链 | 需要额外设计、监控和异步处理 | 长任务、多工具、多 Agent 协作 |

一个直观的玩具估算是：假设每轮新增 1K token，15 轮就是 15K token；如果每 3 轮再插入一次 4K token 的工具日志，那么总量会更快逼近上限。很多工程文章都会提醒，标称 128K 的窗口，真正安全可用的常常不是 128K，而是大约 100K 左右，因为你还要给模型的输出预留空间。DEV Community 上一篇面向工程实践的文章就给出了类似预算思路，把“系统提示词、工具定义、历史对话、工具结果、回复预留”拆开单独管理，而不是把它们当成一个无差别的大字符串处理。[来源](https://dev.to/gantz/why-your-agent-forgets-after-15-turns-and-how-to-fix-it-2f8k)

因此，短期记忆管理不是“让模型记得更多”，而是**在固定窗口内，尽量让最重要的信息持续可见**。工程上的关键不在“摘要写得像不像人类笔记”，而在于两件事：

1. 旧历史是否先被抽取成结构化事实，再做压缩。
2. 每一类信息是否有明确 token 预算，而不是谁长谁占满窗口。

经验上，如果摘要前先抽取“实体、目标、约束、决策、未完成事项”，再生成摘要，关键事实保留率会明显高于直接把长对话压成一段自然语言概述。这个保留率是工程指标，不是数学定理，它取决于任务类型、摘要提示词、工具输出形态，以及你是否做了回归评测。

---

## 问题定义与边界

这里讨论的“短期记忆”，只指**会话内、当前 prompt 直接可见**的那部分内容，不讨论向量数据库、知识库检索、文件系统快照、数据库状态表这类外部长期记忆。

先把三个容易混淆的层级拆开：

| 对象 | 白话解释 | 典型内容 | 生命周期 |
| --- | --- | --- | --- |
| conversation | 最近对话本身 | 用户消息、模型回复、追问与澄清 | 几轮到几十轮 |
| session | 当前任务状态 | 目标、限制、计划、完成进度、待办 | 一次任务内 |
| event / episodic memory | 值得长期保留的事件 | 关键工具结果、异常、决策、风险提示 | 可跨多轮保留 |

很多新手第一次做 Agent，会把这三类内容都直接塞进同一个 `messages` 数组里，然后在超限时从头删消息。这样能跑，但很快会出问题，因为这三类信息的重要性并不一样：

- 最近对话强调“局部连贯”。
- session 强调“当前任务不能跑偏”。
- episodic memory 强调“历史关键事实不能被冲掉”。

把上下文想成一个“token 罐子”，更容易理解：

```text
[system prompt][tool schema][session memory][episodic summary][recent messages][fresh tool outputs][response reserve]
```

每轮都会往里继续塞东西：用户消息、模型回复、工具调用参数、工具返回结果、错误栈、网页抓取内容。窗口一满，最早的内容不会自动升华成高质量记忆，只会被截断，或者被一次糟糕的摘要粗暴压扁。

因此可以把第 $t$ 轮真正进入 prompt 的记忆写成：

$$
M_t = f(S_{t-1}, M_{t-1}, C)
$$

其中：

- $M_t$：第 $t$ 轮实际进入上下文窗口的记忆集合
- $S_{t-1}$：上一轮新增状态，例如用户新要求、最新工具结果、异常
- $M_{t-1}$：上一轮保留下来的内容
- $C$：压缩与调度策略，包括截断、摘要、预算分配、去重规则

如果再把 token 预算显式写出来，可以得到一个更工程化的表示：

$$
B = B_{sys} + B_{tool} + B_{sess} + B_{epis} + B_{conv} + B_{fresh} + B_{resp}
$$

这里：

- $B$：模型总上下文预算
- $B_{sys}$：系统提示词预算
- $B_{tool}$：工具定义预算
- $B_{sess}$：session 记忆预算
- $B_{epis}$：事件记忆预算
- $B_{conv}$：最近对话预算
- $B_{fresh}$：新工具结果预算
- $B_{resp}$：输出预留预算

这个公式的意义很朴素：**总预算是固定的，你多留一类信息，就必须少留另一类信息。**

边界也要说清楚：

1. 滑动窗口解决的是“最近上下文可见性”，不是“长期知识存储”。
2. 摘要压缩解决的是“有限窗口下保留关键事实”，不是“无损还原全部历史”。
3. token 预算解决的是“谁优先占空间”的调度问题，不保证推理一定正确。
4. 多轮退化不只来自窗口不够大，也来自对话本身把任务拆散了。微软研究在六类生成任务上观察到，多轮对话相对单轮完整指令平均性能下降约 39%。这说明“信息分散在多轮里”本身就是额外损耗，不是单纯把窗口做大就能完全解决。[来源](https://www.microsoft.com/en-us/research/publication/llms-get-lost-in-multi-turn-conversation/)

---

## 核心机制与推导

短期记忆管理可以拆成两个动作：**先压缩，再分配**。前者回答“旧内容怎么变短”，后者回答“变短后谁能留下”。

先看一个可落地的预算表示例：

| 类别 | 预算 token | 用途 | 为什么不能省略 |
| --- | --- | --- | --- |
| conversation | 600 | 最近若干轮自然语言对话 | 保证当前语义连贯 |
| session | 300 | 当前任务目标、约束、计划 | 防止模型跑偏 |
| episodic | 800 | 历史决策、实体、关键结论 | 防止忘掉重要事实 |
| tool result fresh | 1200 | 本轮新工具输出 | 给最新证据留位置 |
| response reserve | 2000 | 预留给模型生成回复 | 防止输入塞满后无输出空间 |

这个表不是为了追求“预算数字绝对精确”，而是强制系统回答一个工程问题：**什么信息比什么信息更值得留在窗口里。**

### 1. 滑动窗口

滑动窗口可以写成一个截断函数：

$$
W_t = \text{truncate}(H_t, B_w)
$$

其中：

- $H_t$：截至第 $t$ 轮的历史消息序列
- $B_w$：分配给最近对话的预算

它的优点是实现极简单：从后往前保留，直到预算耗尽为止。它的问题也非常稳定：**越早的重要信息，越容易被无差别删掉。**

例如，一个故障排查 Agent 在第 1 轮明确了“不能改数据库结构”，到第 12 轮只剩最近几轮日志分析，模型就可能提出“新增字段记录重试状态”这种看似合理、实际上违反约束的方案。这不是推理能力不够，而是约束已经不在窗口里了。

### 2. 摘要压缩

摘要压缩是另一种映射：

$$
S_t = g(H_{old}, B_s)
$$

其中：

- $H_{old}$：准备被移出最近窗口的旧历史
- $B_s$：摘要预算
- $g(\cdot)$：压缩函数，可以是规则、模板、模型调用，或者三者组合

很多实现失败，不是因为摘要太短，而是因为摘要只剩一段“自然语言概述”。自然语言概述对人类阅读友好，但对后续 Agent 推理未必足够稳定。更稳妥的做法通常是两步：

1. 先抽取结构化事实。
2. 再把结构化事实组织成摘要。

可抽取的字段一般包括：

| 字段 | 例子 | 作用 |
| --- | --- | --- |
| 实体 | 服务名、订单号、用户角色、机器 ID | 保留对象关系 |
| 目标 | 修复支付超时、完成部署、生成周报 | 保留任务方向 |
| 约束 | 不能改表结构、不能重启主库、只能读权限 | 保留边界 |
| 决策 | 先查幂等，再查队列延迟 | 保留推理路径 |
| 结论 | 回调重试异常、缓存击穿不是主因 | 保留已验证事实 |
| 未完成事项 | 还需验证超时阈值与重试次数配置 | 保留下一步行动 |

如果这一步做得好，后续摘要即使只有几百 token，仍能保留“结构”；如果这一步缺失，摘要往往只剩“聊过支付、讨论了回调、准备继续排查”这类低密度文本。

### 3. 递归摘要

当会话很长时，连“摘要本身”也会越来越长，因此还需要递归摘要：

$$
R_k = g(R_{k-1} \cup H_k, B_s)
$$

它的含义是：上一轮摘要 $R_{k-1}$ 与新的一批历史 $H_k$ 合并后，再次压缩到固定预算 $B_s$ 内。这样，旧记忆的总长度不会无限增长。

假设累计旧历史有 100K token，每层都压到 500 token，那么最终进入新 prompt 的旧记忆不再是 100K，而是一个大约 500 token 的递归摘要。压缩比可以写成：

$$
\text{compression ratio} = \frac{100000}{500} = 200
$$

但压缩比越高，不代表系统越好。因为真实可用的信息保留率通常不是线性下降，而是先缓慢下降，再在某个压缩强度后快速恶化。可以把它理解成下面这个关系：

$$
\text{retention} = h(\text{compression ratio}, \text{structure quality})
$$

这里真正可控的变量不是“压了多少倍”，而是 `structure quality`，也就是你在压缩前是否把实体、约束、决策链提取干净。

### 4. 为什么多 Agent 更依赖压缩

单 Agent 会话里，旧上下文顶多是“自己和用户说过的话”；多 Agent 链路里，问题会变成“每一级都把上一级所有原始上下文继续往下传”。这样 token 成本会指数式地接近不可控。

CloudThinker 的一篇工程文章给了一个很典型的例子：上游 Agent 先处理 50K+ token 的 CloudWatch 输出，然后把结论压成约 200 token 的 handoff 发给下游专家，下游只继承“诊断结论与待验证假设”，而不是整个原始日志。文章中给出的例子显示，这种基于上下文隔离和 handoff 摘要的做法，在多级链路里能把总处理 token 明显压下来。[来源](https://cloudthinker.ai/blogs/cloudthinker-agentic-orchestration-and-context-optimization)

结论不是“所有工具结果都只留 200 token”，而是：**下游 Agent 需要的是任务相关证据，不是上游的全部工作痕迹。**

---

## 代码实现

下面给出一个最小可运行版本，演示三件事：

1. 新消息持续写入。
2. 当历史过长时，把旧消息压成结构化摘要。
3. 按预算组装最终上下文。

代码不依赖第三方库。`count_tokens` 仍然是玩具实现，只做近似计数，但这版代码可以直接运行，便于先把机制跑通。

```python
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List
import re


TOKEN_BUDGET = {
    "conversation": 600,
    "session": 300,
    "episodic": 800,
    "tool_fresh": 1200,
    "reserve": 2000,
}

SUMMARY_TRIGGER_TOKENS = 1600


def count_tokens(text: str) -> int:
    """
    玩具实现：用“中文单字 / 英文单词 / 数字 / 符号块”粗略近似 token。
    真正接入生产时，应替换为目标模型对应 tokenizer。
    """
    parts = re.findall(r"[\u4e00-\u9fff]|[A-Za-z_]+|\d+|[^\s]", text)
    return len(parts)


@dataclass
class Message:
    role: str
    content: str
    kind: str = "conversation"  # conversation / tool / decision


@dataclass
class ContextBundle:
    session: List[str]
    episodic: List[str]
    conversation: List[str]
    tool_fresh: List[str]

    def total_tokens(self) -> int:
        items = self.session + self.episodic + self.conversation + self.tool_fresh
        return sum(count_tokens(x) for x in items)


@dataclass
class TokenBuffer:
    messages: List[Message] = field(default_factory=list)
    episodic_summaries: List[str] = field(default_factory=list)

    def append(self, msg: Message) -> None:
        self.messages.append(msg)

    def message_tokens(self) -> int:
        return sum(count_tokens(m.content) for m in self.messages)

    def summary_tokens(self) -> int:
        return sum(count_tokens(s) for s in self.episodic_summaries)

    def total_tokens(self) -> int:
        return self.message_tokens() + self.summary_tokens()

    def should_compress(self) -> bool:
        return self.message_tokens() > SUMMARY_TRIGGER_TOKENS and len(self.messages) > 6

    def compress_old_segments(self, keep_last: int = 6) -> None:
        """
        压缩较老的消息，仅保留最近 keep_last 条原始消息。
        摘要前先抽取结构化事实，避免只剩模糊自然语言。
        """
        if len(self.messages) <= keep_last:
            return

        old_messages = self.messages[:-keep_last]
        recent_messages = self.messages[-keep_last:]

        goals: List[str] = []
        constraints: List[str] = []
        decisions: List[str] = []
        tool_findings: List[str] = []
        open_items: List[str] = []

        for msg in old_messages:
            lines = [line.strip() for line in msg.content.splitlines() if line.strip()]
            for line in lines:
                if line.startswith("目标:"):
                    goals.append(line[3:].strip())
                elif line.startswith("约束:"):
                    constraints.append(line[3:].strip())
                elif line.startswith("决策:"):
                    decisions.append(line[3:].strip())
                elif line.startswith("结论:"):
                    tool_findings.append(line[3:].strip())
                elif line.startswith("待办:"):
                    open_items.append(line[3:].strip())

            if msg.kind == "tool" and not lines:
                tool_findings.append(msg.content.strip())
            if msg.kind == "decision" and "决策:" not in msg.content:
                decisions.append(msg.content.strip())

        def dedupe_keep_order(items: List[str], limit: int) -> List[str]:
            seen = set()
            result = []
            for item in items:
                normalized = item.strip()
                if not normalized or normalized in seen:
                    continue
                seen.add(normalized)
                result.append(normalized)
                if len(result) >= limit:
                    break
            return result

        goals = dedupe_keep_order(goals, 4)
        constraints = dedupe_keep_order(constraints, 4)
        decisions = dedupe_keep_order(decisions, 6)
        tool_findings = dedupe_keep_order(tool_findings, 6)
        open_items = dedupe_keep_order(open_items, 4)

        summary_parts = ["历史摘要:"]
        if goals:
            summary_parts.append("目标:")
            summary_parts.extend(f"- {x}" for x in goals)
        if constraints:
            summary_parts.append("约束:")
            summary_parts.extend(f"- {x}" for x in constraints)
        if decisions:
            summary_parts.append("关键决策:")
            summary_parts.extend(f"- {x}" for x in decisions)
        if tool_findings:
            summary_parts.append("关键结论:")
            summary_parts.extend(f"- {x}" for x in tool_findings)
        if open_items:
            summary_parts.append("未完成事项:")
            summary_parts.extend(f"- {x}" for x in open_items)

        if len(summary_parts) == 1:
            summary_parts.append("- 旧历史中未抽取到明确结构化字段，保留主题与当前结论。")

        merged_summary = "\n".join(summary_parts)

        # 只保留最近几份摘要，避免摘要区继续无界增长
        self.episodic_summaries.append(merged_summary)
        self.episodic_summaries = self.episodic_summaries[-3:]
        self.messages = recent_messages

        # 如果摘要区超预算，再做一次极简折叠
        self._trim_episodic_to_budget()

    def _trim_episodic_to_budget(self) -> None:
        while sum(count_tokens(x) for x in self.episodic_summaries) > TOKEN_BUDGET["episodic"]:
            if len(self.episodic_summaries) == 1:
                text = self.episodic_summaries[0]
                lines = text.splitlines()
                self.episodic_summaries[0] = "\n".join(lines[:12])
                break
            self.episodic_summaries.pop(0)

    def build_context(self, session_state: str) -> ContextBundle:
        if count_tokens(session_state) > TOKEN_BUDGET["session"]:
            raise ValueError("session_state exceeds session budget")

        conversation_items: List[str] = []
        tool_fresh_items: List[str] = []

        # 从新到旧收集，再反转，保证最终顺序仍是时间正序
        for msg in reversed(self.messages):
            target = tool_fresh_items if msg.kind == "tool" else conversation_items
            budget_key = "tool_fresh" if msg.kind == "tool" else "conversation"
            current_tokens = sum(count_tokens(x) for x in target)
            msg_tokens = count_tokens(msg.content)

            if current_tokens + msg_tokens <= TOKEN_BUDGET[budget_key]:
                target.append(f"{msg.role}: {msg.content}")

        conversation_items.reverse()
        tool_fresh_items.reverse()

        episodic_items: List[str] = []
        used = 0
        for summary in reversed(self.episodic_summaries):
            t = count_tokens(summary)
            if used + t > TOKEN_BUDGET["episodic"]:
                continue
            episodic_items.append(summary)
            used += t
        episodic_items.reverse()

        bundle = ContextBundle(
            session=[session_state],
            episodic=episodic_items,
            conversation=conversation_items,
            tool_fresh=tool_fresh_items,
        )
        return bundle


def demo() -> None:
    buf = TokenBuffer()

    for i in range(1, 9):
        buf.append(
            Message(
                role="user",
                kind="conversation",
                content=(
                    f"第{i}轮用户消息\n"
                    "目标: 修复支付回调超时\n"
                    "约束: 不能改数据库结构\n"
                    "待办: 确认重试次数与超时阈值配置"
                ),
            )
        )
        buf.append(
            Message(
                role="assistant",
                kind="decision",
                content=(
                    f"第{i}轮分析\n"
                    "决策: 优先检查网关重试、幂等逻辑与消息堆积\n"
                    "结论: 当前没有证据表明主库性能是主因"
                ),
            )
        )
        if i % 3 == 0:
            buf.append(
                Message(
                    role="tool",
                    kind="tool",
                    content=(
                        "CloudWatch 摘要: callback_timeout_p95 升高; "
                        "worker_queue_lag 持续增长; db_cpu 正常"
                    ),
                )
            )

    if buf.should_compress():
        buf.compress_old_segments(keep_last=6)

    context = buf.build_context(
        session_state=(
            "当前会话状态:\n"
            "- 任务: 排查支付回调超时\n"
            "- 必守约束: 不能改数据库结构\n"
            "- 当前策略: 优先保留目标、约束、关键决策和最新工具证据"
        )
    )

    assert "排查支付回调超时" in context.session[0]
    assert len(context.conversation) > 0
    assert len(context.episodic) <= 3
    assert context.total_tokens() > 0

    print("context assembled")
    print("conversation tokens =", sum(count_tokens(x) for x in context.conversation))
    print("episodic tokens =", sum(count_tokens(x) for x in context.episodic))
    print("tool_fresh tokens =", sum(count_tokens(x) for x in context.tool_fresh))
    print("total tokens =", context.total_tokens())


if __name__ == "__main__":
    demo()
```

这段代码对应的工程含义可以直接拆成四步：

| 步骤 | 代码位置 | 做的事 | 作用 |
| --- | --- | --- | --- |
| 写入消息 | `append()` | 持续收集用户、助手、工具输出 | 建立原始历史 |
| 判断是否压缩 | `should_compress()` | 根据阈值判断是否需要整理旧历史 | 防止等到超限再应急处理 |
| 压缩旧历史 | `compress_old_segments()` | 先抽取字段，再生成结构化摘要 | 保留高价值事实 |
| 组装上下文 | `build_context()` | 按预算拼接 session、episodic、recent、tool_fresh | 控制最终 prompt 长度 |

如果用伪代码概括，就是：

```text
append(new_message)

if should_compress():
    compress_old_segments()

context = build_context(session_state)
call_model(context)
```

真实工程里还会再补两层：

1. 用真实 tokenizer 替换玩具 `count_tokens()`。
2. 把压缩过程放到异步线程或队列，避免主请求链路阻塞。

一个更接近生产的流程通常是：

```text
主线程收到新消息
-> 先写原始消息缓冲区
-> 如果达到阈值，投递后台压缩任务
-> 当前轮先用“最近窗口 + 上一次缓存摘要”继续工作
-> 后台刷新新的摘要缓存
-> 下一轮消费更新后的摘要
```

这样做的核心收益不是“更优雅”，而是把“可用性”和“记忆质量”分开：当前轮先保证能回应，下一轮再吃到更好的摘要。

---

## 工程权衡与常见坑

最容易犯的错误，不是“不会做摘要”，而是**在错误的位置做摘要，或者在错误的时机才开始压缩**。

| 坑 | 典型表现 | 为什么会发生 | 规避方式 |
| --- | --- | --- | --- |
| 只做滑动窗口 | 模型忘掉早期目标和硬约束 | 最近消息天然挤掉更早但更重要的内容 | 至少保留结构化 session memory |
| 摘要不保留结构 | 摘要看起来通顺，但后续决策明显跑偏 | 只压成自然语言段落，没有实体和决策链 | 先抽取字段，再生成摘要 |
| 工具输出原样回填 | 日志、JSON、HTML 快速撑爆窗口 | 工具结果通常比自然语言长一个数量级 | 大结果先摘要，再按字段回填 |
| 同步压缩 | 每轮都明显变慢 | 额外增加一次模型调用或重规则处理 | 放到后台异步，主线程读缓存 |
| 不预留回复空间 | 输入能发出，输出阶段截断或失败 | 把标称窗口误当成可全用预算 | 保留 5% 到 10% response reserve |
| 压缩触发太晚 | 一旦超限，只能临时砍历史 | 没有提前整理缓冲区 | 在 60% 到 70% 用量启动整理 |
| 摘要只保留“结论” | 模型知道做什么，不知道为什么 | 缺失推理前提与约束来源 | 摘要中同时保留目标、约束、决策 |
| 摘要不断叠摘要 | 摘要越来越抽象，最终失真 | 多轮压缩没有回看原始证据 | 关键事实单独做结构化字段，不反复改写 |

### 为什么“只保留结论”很危险

新手常见误区是：“我只要把最终结论记下来就够了。”这在很多任务里不成立，因为后续动作依赖的不只是结论，还依赖**结论成立的条件**。

例如：

| 丢失的信息 | 结果 |
| --- | --- |
| 只记得“先修支付超时” | 不知道前提是“不能改数据库结构” |
| 只记得“怀疑幂等逻辑异常” | 不知道这是假设，尚未验证 |
| 只记得“数据库不是主因” | 不知道这是基于某次采样，不是永久事实 |

因此，摘要里至少要区分三种语义：

1. 已确认事实
2. 当前假设
3. 必守约束

如果三者混在一起，模型会把“猜测”当事实，把“旧结论”当新证据。

### 延迟和精度怎么取舍

延迟和精度是硬折衷。

- 同步压缩的优点是当前轮上下文质量高。
- 同步压缩的代价是当前轮会多一次处理，增加额外延迟。
- 异步压缩的优点是交互流畅。
- 异步压缩的代价是当前轮可能还在使用“旧摘要”。

这类折衷没有统一标准，一般取决于任务类型：

| 任务类型 | 更优先的目标 | 常见选择 |
| --- | --- | --- |
| 聊天、客服 | 响应快 | 轻量滑窗 + 异步摘要 |
| 代码修改、故障排查 | 约束不能丢 | 结构化摘要 + 提前压缩 |
| 多 Agent 调度 | 成本和链路稳定性 | handoff 摘要 + 上下文隔离 |
| 高风险流程 | 可审计和一致性 | 外部状态存储 + 会话内摘要辅助 |

一个典型失真例子是：第 1 轮定义“订单状态必须与支付状态一致”，第 8 轮摘要里只剩“修复支付问题”。模型此时仍然可能生成一套看起来很完整的方案，但它已经偏离了原始业务约束。比“完全忘记”更危险的是，**模型给出了自洽但错误的方案**。

---

## 替代方案与适用边界

滑动窗口、摘要压缩、预算分配，解决的是**短期记忆调度**。一旦任务超出“会话内信息整理”的范围，就应该考虑外部记忆，而不是继续把摘要当数据库用。

先看一个简单映射：

| 需求特征 | 更适合的方案 | 原因 |
| --- | --- | --- |
| 5 轮以内短对话 | 纯滑动窗口 | 足够简单，额外设计不划算 |
| 需要跨十几轮记住目标与约束 | 滑窗 + 结构化摘要 | 能兼顾最近对话和长期任务状态 |
| 多工具、多 Agent、长链路任务 | 递归摘要 + token 预算 | 控制复制扩散，避免链路越传越长 |
| 大规模知识查找 | RAG / 检索式记忆 | 目标是找知识，不是保存会话过程 |
| 跨天协作、需要审计 | 外部状态存储 + 新会话重载 | 需要稳定、可追溯、可查询的事实源 |

可以用一个更直接的决策规则：

1. 如果任务短、上下文浅，优先滑动窗口，不要为了“看起来高级”过度设计。
2. 如果用户目标会跨十几轮持续存在，至少加入 session summary，把目标、约束、待办单独存放。
3. 如果工具输出远大于自然语言对话，先压工具结果，而不是先删用户消息。
4. 如果历史事实必须稳定、可审计、可跨天复用，直接上外部状态存储或检索，不要指望摘要承担数据库职责。
5. 如果摘要已经被压到几十 token 还嫌长，说明问题通常不是“摘要不够强”，而是“你在让会话窗口承担长期存储的工作”。

这里再强调一个初学者容易误判的点：**摘要不是长期记忆系统，它只是短期记忆的整理器。**

对比来看：

- “只保留最近 5 轮”适合问答，不适合持续排查线上故障。
- “摘要 + 预算保留结构化目标”更适合协作式任务，因为它保留了“为什么这样做”。
- 当预算已经紧到摘要都只能剩几十 token 时，继续压缩往往收益很差，此时更合理的是把历史写入外部存储，再按相关性检索回当前部分。

一句话概括边界：**短期记忆负责把当前任务继续做下去，长期记忆负责把任务世界的事实稳定保存下来。**

---

## 参考资料

- [Why Your Agent Forgets After 15 Turns (And How to Fix It)](https://dev.to/gantz/why-your-agent-forgets-after-15-turns-and-how-to-fix-it-2f8k)  
  重要性：给出了上下文预算、实际可用窗口、滑动窗口与摘要组合的工程化写法，适合作为入门实现框架。文中的“128K 实际常按约 100K 可用处理”属于工程预留经验，不是统一标准。

- [CloudThinker Agentic Orchestration and Context Optimization](https://cloudthinker.ai/blogs/cloudthinker-agentic-orchestration-and-context-optimization)  
  重要性：提供多 Agent handoff、上下文隔离、摘要交接与 token 成本控制案例，适合说明“为什么多级链路必须避免原始上下文逐级复制”。

- [The Forgetting Agent: Why Multi-Turn Conversations Collapse](https://blakecrosley.com/en/blog/agent-memory-degradation)  
  重要性：用工程视角解释多轮退化常见症状，包括压缩损失、上下文漂移、协作失真，适合作为问题现象的补充阅读。

- [LLMs Get Lost In Multi-Turn Conversation](https://www.microsoft.com/en-us/research/publication/llms-get-lost-in-multi-turn-conversation/)  
  重要性：研究层面的核心证据来源。微软研究在六类生成任务上报告，多轮对话相对单轮完整指令平均性能下降约 39%，说明多轮交互本身就是任务完成质量的重要干扰项。

- [Multi-Agent Negotiation Simulations with LLMs](https://next.gr/ai/autonomous-systems/multi-agent-negotiation-simulations-with-llms)  
  重要性：可作为“动态上下文裁剪、重要性权衡、显式策略分配”的补充案例，用来说明预算分配不是拍脑袋，而是可以外显成策略函数。
