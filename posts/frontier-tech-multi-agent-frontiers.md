## 核心结论

多智能体协作的前沿，不是把一个任务丢给更多代理，而是把任务拆成更清晰的角色、协议和阶段。这里的“代理”就是能接收消息、调用模型或工具、再产出结果的执行单元；“协议”就是它们如何传话、何时接棒、何时停下来的规则。

如果一个任务本来就只有一步问答，单代理通常更便宜、更稳定。多智能体真正有价值的场景，是长流程、异构工具、多阶段产物的任务：先有人规划，再有人执行，再有人验证，最后把结果汇总。AutoGen 一类框架把这种流程抽象成可对话的 Agent；IoA 一类系统进一步把“找队友、组队、路由消息、管理状态”做成统一基础设施。

可以把它先理解成一条流水线，而不是一群人同时乱说话。决定收益的核心，不是代理数量，而是三件事是否成立：

| 要素 | 白话解释 | 做对了的结果 |
|---|---|---|
| 角色拆分 | 每个代理只做一类工作 | 减少一个代理包办全部时的失误 |
| 消息协议 | 规定消息格式、接棒条件、结束条件 | 减少重复沟通和状态混乱 |
| 任务分解 | 把大任务切成能独立验证的小任务 | 让每一步都有明确产物 |

最小判断标准可以写成：

$$
C_{\text{total}} = C_{\text{agents}} + C_{\text{comm}}
$$

其中，$C_{\text{agents}}$ 是各角色完成子任务的成本，$C_{\text{comm}}$ 是代理之间来回传消息的成本。多智能体只有在质量提升或流程收益足以覆盖通信开销时才成立。

---

## 问题定义与边界

“多智能体协作”在工程上不是哲学问题，而是一个调度问题：一个复杂任务是否值得拆给多个角色处理。这里的“调度”就是决定谁先做、谁后做、谁来检查、什么时候结束。

边界先说清楚：

| 任务类型 | 是否适合多智能体 | 原因 |
|---|---|---|
| 单轮问答 | 不适合 | 通信成本几乎纯浪费 |
| 少量工具调用 | 通常不适合 | 单代理加工具已经够用 |
| 长流程原型开发 | 适合 | 规划、实现、验证能明显分开 |
| 多数据源分析 | 适合 | 不同代理可并行查找、汇总、交叉核验 |
| 多阶段交付物 | 适合 | 草稿、实现、测试、评审天然分层 |

玩具例子先看一个。假设需求是“给博客加搜索页”。单代理要自己理解需求、设计字段、写代码、检查边界。多智能体则可以拆成：

1. Planner 先列出步骤和验收标准。
2. Worker 按步骤实现搜索逻辑。
3. Reviewer 检查空关键词、大小写、无结果提示。
4. QA 再跑一轮样例输入。

这个例子里，多智能体的收益不在“更聪明”，而在“把遗漏暴露成可检查的中间产物”。

真实工程例子更典型。比如做“产品需求 -> 原型代码 -> 自动化测试 -> 评审意见 -> 修订提交”的流水线。这里有文档解析、代码生成、终端执行、浏览器测试、结果总结五类动作。单代理需要在很长上下文里来回切换角色，很容易忘掉前面的约束；多智能体可以把这些职责拆给不同代理，通过统一消息流传递阶段产物。这正是 AutoGen、IoA 这类系统重点解决的问题。

所以边界很明确：多智能体不是单代理的升级版，而是复杂流程的结构化编排器。简单问题不要上复杂系统。

---

## 核心机制与推导

多智能体系统真正难的地方，不是“让两个模型互相聊天”，而是让聊天有状态、有进展、能收敛。

第一层机制是角色分工。AutoGen 把代理建模成可对话对象，任何代理都能收发消息。常见角色可以简化成下面三类：

| 角色 | 职责 | 交付物 |
|---|---|---|
| Planner | 分解任务、记录约束、指定下一步 | ledger 或任务清单 |
| Worker | 调工具、写代码、产出中间结果 | 代码、文档、数据 |
| Reviewer / QA | 检查逻辑、执行验证、决定是否返工 | 评审意见、测试结论 |

这里的“ledger”可以理解成工作台账，也就是当前任务的结构化记忆：哪些事实已确认，哪些需要查询，哪些需要计算，哪些只是暂时猜测。微软研究团队在介绍 AutoGen 复杂任务流程时，明确把 ledger 当成工作记忆来组织多代理协作。它的重要性在于，系统不是每轮都从自然语言大段重述开始，而是围绕同一份台账推进。

第二层机制是状态机。IoA 把群聊状态形式化为有限状态机，至少包含讨论、同步分派、异步分派、暂停等待、结论几个状态。这里的“状态机”就是系统只能在少数几个明确状态之间切换，不能无限制乱跳。这样做的意义是：每轮对话不仅要生成内容，还要决定现在处于什么阶段，下一位说话者是谁，是否需要等待某个异步任务完成。

可以把推进条件写成一个简单判断：

$$
\text{继续协作} \iff \Delta Q - C_{\text{comm}} > 0
$$

其中 $\Delta Q$ 表示多一个角色参与后带来的质量增益，$C_{\text{comm}}$ 表示新增加的通信成本。这个公式不是论文里的严格优化目标，但它准确表达了工程判断：新增代理只有在带来足够增益时才值得存在。

再看成本数据。IoA 在开放任务基准上的分析显示，单体代理接入系统后，子任务成本可能下降，但通信成本会显著增加：

| 设置 | 每任务平均成本（美元） |
|---|---|
| AutoGPT（Standalone） | 0.39 |
| Open Interpreter（Standalone） | 0.16 |
| AutoGPT（in IoA） | 0.33 |
| Open Interpreter（in IoA） | 0.13 |
| IoA Communication | 0.53 |
| IoA Communication（Dedup.） | 0.28 |
| IoA Overall | 0.99 |
| IoA Overall（Dedup.） | 0.74 |

这组数字说明一个核心事实：任务拆分可能让执行更便宜，但如果消息层失控，整体成本仍会上升。IoA 论文还指出，很多高通信成本来自重复转述旧信息，尤其在异步分派后最明显。

所以前沿不只是“更多代理”，而是三种基础能力：

1. 结构化记忆：用 ledger、任务表、结论表代替长篇复述。
2. 明确状态：规定何时讨论、何时分派、何时等待、何时结束。
3. 通信压缩：只传新信息，不反复转发旧上下文。

---

## 代码实现

下面给一个可运行的玩具实现。它不依赖任何外部框架，只模拟三个角色如何通过消息总线协作，以及消息去重如何降低通信成本。这里的“消息总线”就是所有角色都往同一个消息列表写入结构化消息。

```python
from dataclasses import dataclass
from typing import List, Dict, Set


@dataclass
class Message:
    sender: str
    role: str
    content: str


def planner(task: str) -> List[Message]:
    return [
        Message("planner", "ledger", "goal: 为博客增加搜索功能"),
        Message("planner", "ledger", "steps: 1.建立索引 2.实现过滤 3.处理空关键词"),
        Message("planner", "handoff", f"task: {task}"),
    ]


def worker(messages: List[Message]) -> List[Message]:
    seen_goal = any("搜索功能" in m.content for m in messages)
    assert seen_goal
    return [
        Message("worker", "code", "def search(posts, q): return [p for p in posts if q.lower() in p.lower()]"),
        Message("worker", "note", "need: 空关键词时返回全部文章"),
    ]


def reviewer(messages: List[Message]) -> List[Message]:
    code_msg = next(m for m in messages if m.role == "code")
    assert "def search" in code_msg.content
    return [
        Message("reviewer", "qa", "check: 大小写一致"),
        Message("reviewer", "qa", "check: 空关键词返回全部"),
        Message("reviewer", "decision", "status: revise_once"),
    ]


def dedup(messages: List[Message]) -> List[Message]:
    unique: List[Message] = []
    seen: Set[str] = set()
    for m in messages:
        key = f"{m.role}:{m.content}"
        if key not in seen:
            seen.add(key)
            unique.append(m)
    return unique


def cost(messages: List[Message], token_price: float = 0.001) -> float:
    # 用字符数近似 token 数，只做教学示意
    total_chars = sum(len(m.content) for m in messages)
    return round(total_chars * token_price, 3)


bus = []
bus.extend(planner("实现搜索并补充边界处理"))
bus.extend(worker(bus))
bus.extend(reviewer(bus))

raw_cost = cost(bus)

# 模拟重复转发旧消息
bus_with_dup = bus + [bus[0], bus[1], bus[1]]
dup_cost = cost(bus_with_dup)
dedup_cost = cost(dedup(bus_with_dup))

assert raw_cost <= dup_cost
assert dedup_cost < dup_cost
assert any(m.role == "decision" for m in bus)

print("raw_cost =", raw_cost)
print("dup_cost =", dup_cost)
print("dedup_cost =", dedup_cost)
```

这个例子表达的是机制，不是产品代码。它说明三件事：

1. Planner 先产出 ledger，而不是直接让 Worker 盲做。
2. Worker 只负责实现，不负责定义验收标准。
3. Reviewer 通过结构化检查项回传，而不是写一段泛泛评论。

如果切到真实工程，代码通常会换成框架实现。以 AutoGen 的思路看，`AssistantAgent` 可以承担规划或执行角色，`UserProxyAgent` 可以承接代码执行、工具调用或人工确认。启动后，任务由某个发起者 `initiate_chat`，中间再靠自动回复、工具调用和群聊管理推进。

真实工程例子可以这样理解：你要做一个“从 PRD 自动生成原型页面并跑回归测试”的系统。

1. Planner 读取 PRD，生成 ledger：页面列表、字段约束、验收点。
2. Worker 调代码模型生成 HTML/CSS/JS。
3. Browser Agent 打开页面截图、检查交互。
4. QA Agent 读取日志和截图，输出缺陷列表。
5. Planner 根据缺陷决定返工还是结束。

这类流程的关键，不在单个代理能力有多强，而在每轮 hand-off 都要附带最小但完整的上下文，比如任务编号、输入约束、上一步产物位置、验收结论。

---

## 工程权衡与常见坑

多智能体系统最常见的失败，不是模型答错，而是系统层面失控。这里的“系统层面”就是消息、状态、角色和成本之间的耦合出了问题。

| 陷阱 | 现象 | 规避策略 |
|---|---|---|
| 上下文重复 | 代理不断复述已有内容 | 用 JSON 消息、摘要字段、dedup 模块 |
| 协调开销过高 | 任务没做多少，沟通轮次先爆炸 | 设最大轮次和最大团队规模 |
| 责任模糊 | 大家都在提建议，没人对结果负责 | 每个角色绑定唯一输出物 |
| 状态失真 | 系统不知道该继续讨论还是等待任务 | 明确状态机和结束条件 |
| 返工循环 | Reviewer 一直打回，Worker 一直重写 | 给出可执行验收标准，而不是主观意见 |

最值得强调的是通信重复。IoA 的成本分析显示，手工去重后通信成本从 0.53 美元降到 0.28 美元，整体成本从 0.99 美元降到 0.74 美元。这个量级已经说明：很多多智能体系统不是输在“不会做”，而是输在“太会说”。

因此工程上应该优先做的往往不是再加一个代理，而是先补三条规则：

1. 只有新信息才允许进入共享上下文。
2. 每次 hand-off 必须包含固定字段，如 `goal`、`task_id`、`artifact`、`done_criteria`。
3. 超过若干轮无进展时，回到 ledger 重排任务，而不是继续空转。

对初级工程师来说，可以记一个简单原则：先把单代理流程跑通，再把最容易出错的那一步拆成独立角色。不要一开始就设计“五代理自治社会”。

---

## 替代方案与适用边界

多智能体不是默认答案。很多情况下，更好的方案是“单代理 + 工具调用 + 明确提示模板”。如果任务短、上下文稳定、工具少，这种方案更容易维护。

可以用下面这张表做选型：

| 方案 | 适合场景 | 优点 | 缺点 |
|---|---|---|---|
| 单代理 + Prompt | 单步问答、轻量生成 | 成本低、实现快 | 长流程容易漏步骤 |
| 单代理 + Tools | 少量外部工具调用 | 架构简单、可控 | 角色混合，状态管理弱 |
| 单代理 + Orchestrator | 有流程但角色不多 | 易落地 | 扩展到异构代理较难 |
| 多智能体协作 | 长流程、异构工具、多阶段产物 | 分工清晰、可并行、便于校验 | 通信成本高、调度复杂 |

如果只需要“查资料 -> 写摘要 -> 输出结果”，单代理通常够用。若任务升级为“拆需求 -> 写代码 -> 执行测试 -> 看截图 -> 回归修改”，多智能体才开始显示结构优势。

所以适用边界可以压缩成一句话：当任务天然包含不同能力层、不同工具链、不同验收阶段时，多智能体才是合理设计；否则它只是把简单问题包装成复杂系统。

---

## 参考资料

- [AutoGen 文档：Multi-agent Conversation Framework](https://autogenhub.github.io/autogen/docs/Use-Cases/agent_chat/)
- [Microsoft Research: AutoGen: Enabling Next-Gen LLM Applications via Multi-Agent Conversation Framework](https://www.microsoft.com/en-us/research/publication/autogen-enabling-next-gen-llm-applications-via-multi-agent-conversation-framework/)
- [ICLR 2025 论文：Internet of Agents](https://proceedings.iclr.cc/paper_files/paper/2025/file/59c27bf8d56d3d50c7aeaf7535dee975-Paper-Conference.pdf)
