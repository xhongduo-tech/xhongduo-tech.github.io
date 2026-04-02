## 核心结论

写作助手 Agent 不是“会续写的聊天机器人”，而是一个把**计划**、**检索**、**执行**、**反馈**串成闭环的系统。计划，白话说就是先决定要分几步做；检索，就是补足当前写作需要的材料；执行，就是生成提纲、段落、改写或校对；反馈，就是检查结果是否还满足最初目标。

它为什么成立，核心不在“提示词写得多漂亮”，而在于把写作过程拆成可控阶段，并且在每个阶段只提供必要信息。对新手最重要的理解是：写作助手 Agent 不是一次性吐出全文，而是像一名受约束的协作者，先确认目标，再进入不同工序，最后对结果做回看。

一个实用的设计框架可以从五个维度看：

| 维度 | 关注问题 | 典型设计项 | 对写作助手的直接影响 |
| --- | --- | --- | --- |
| 任务 | 写什么、写到哪一步 | 调研、提纲、初稿、改写、校对 | 决定 Agent 当前阶段 |
| 用户 | 谁在使用、能力如何 | 新手/熟手、领域知识、控制偏好 | 决定解释深度和交互方式 |
| 技术 | 用哪些模型和工具 | 检索、重写、事实校验、格式检查 | 决定能力边界和成本 |
| 交互 | 人和系统怎么来回协作 | 建议式、问答式、批注式、自动执行 | 决定可控性与打断成本 |
| 生态 | 放在什么系统里运行 | 编辑器、知识库、PR Bot、后台治理 | 决定权限、审计和责任划分 |

玩具例子很直观。用户说：“写一段关于分布式链路追踪的技术原理。”一个像样的写作助手 Agent 不会立刻输出 800 字，而会先把任务拆成四段：调研术语、组织结构、起草正文、检查术语是否偏题。这样做的结果是，输出更像“技术原理说明”，而不是泛泛的科普介绍。

---

## 问题定义与边界

写作助手 Agent 最容易失败的地方，不是模型不会写，而是**目标没有边界**。边界，白话说就是“做到什么程度就算完成，最多花多少资源”。如果没有边界，Agent 会不断追加材料、不断重写、不断调用工具，最后成本上升，结果却未必更好。

最常见的失控句式是“写更多”“再展开一点”“你继续优化”。这种指令没有终止条件，系统只能把“继续”理解为重复迭代。解决方法不是换模型，而是补全请求定义。比如把需求改成：“完成一个 3 段说明，最多 400 字，面向初级工程师，解释核心原理，不展开历史背景。”这时目标、长度、受众、主题边界都清楚了。

工程上可以用一个边界检查表先把请求压实：

| 检查项 | 必须回答的问题 | 没回答会怎样 |
| --- | --- | --- |
| 目标 | 最终产物是什么 | 输出形态漂移 |
| 终止条件 | 写到什么程度停止 | 无限迭代 |
| 预算 | 最多多少轮、多少 token、多少工具调用 | 成本失控 |
| 阶段 | 当前是调研、提纲、初稿还是校对 | 工具乱用 |
| 工具门控 | 哪个阶段允许调用哪个工具 | 错误阶段做错误动作 |

这里还有一个常被忽略的边界：**上下文包**。上下文包，白话说就是“本轮真正发给模型看的那一小包信息”。如果把所有聊天记录、所有检索结果、所有草稿版本都塞进去，模型并不会因此更稳，反而会把真正重要的约束淹没掉。

所以问题定义不只是“用户想写什么”，还包括“系统本轮只该知道什么”。对写作助手 Agent 来说，这一步决定了后续能否收敛。

---

## 核心机制与推导

写作助手 Agent 的核心机制可以压缩成一个预算公式：

$$
|I| + |S| + |E| + |H| \le B
$$

其中：

- $I$ 是 Instruction，指令，白话说就是用户当前明确要求。
- $S$ 是 State，状态，白话说就是系统记住的阶段、约束和变量。
- $E$ 是 Evidence，证据，白话说就是检索到的资料、规则和外部结果。
- $H$ 是 History，历史，白话说就是之前对话和之前输出的摘要。
- $B$ 是预算上限，也就是本轮上下文最大容量。

这个公式的意义不是数学炫技，而是说明一个事实：模型每次决策都在竞争有限注意力。你多塞一点证据，就必须少放一点历史；你保留更多历史，就必须压缩状态。真正的工程能力，是决定哪部分信息值得保留。

看一个最小数值例子。假设 $B=900$，我们分配：

| 项目 | 预算 |
| --- | --- |
| 指令 $I$ | 100 |
| 状态 $S$ | 150 |
| 证据 $E$ | 400 |
| 历史 $H$ | 250 |
| 总计 | 900 |

这时预算刚好打满。如果新检索结果又增加 200 个 token，那么总量变成 1100，系统必须裁剪。若不裁剪，模型常见的失败模式是忘掉“字数限制”或“语气约束”，因为这些通常放在指令或状态区，而长证据更容易抢占注意力。

这里可以得到一个直接推导：对写作助手 Agent，**信息排序比提示词润色更关键**。因为在预算固定时，错误不是“模型没看到信息”，而是“模型看到太多无关信息，没把关键约束当成优先项”。

更进一步，预算不该只有总量，还应有子预算。比如状态最多 300、证据最多 800、历史最多 250。这样做的作用类似限流。限流，白话说就是“防止某一类信息突然占满整个通道”。在写作场景里，最典型的是检索结果暴涨，把结构约束埋掉。

因此，写作助手 Agent 的机制不是“多轮对话自动更聪明”，而是“每轮都重新编译上下文，并显式控制预算分布”。

---

## 代码实现

实现上最关键的是一个“上下文编译器”。上下文编译器，白话说就是把目标、状态、证据、历史重新整理成一份可发给模型的输入包。它不是简单拼字符串，而是按阶段、预算和工具权限做筛选。

下面给出一个可运行的最小实现。它没有接真实模型，但完整体现了预算校验、历史裁剪和阶段门控。

```python
from dataclasses import dataclass, field

def prune_history(history, max_chars):
    kept = []
    total = 0
    for item in reversed(history):
        if total + len(item) > max_chars:
            break
        kept.append(item)
        total += len(item)
    return list(reversed(kept))

def build_context(goal, state, evidence, history, budget, sub_budget):
    instruction = goal
    state_text = f"stage={state['stage']}; constraints={state['constraints']}"
    evidence_text = "\n".join(evidence)
    history_list = prune_history(history, sub_budget["H"])
    history_text = "\n".join(history_list)

    parts = {
        "I": instruction[:sub_budget["I"]],
        "S": state_text[:sub_budget["S"]],
        "E": evidence_text[:sub_budget["E"]],
        "H": history_text[:sub_budget["H"]],
    }

    total = sum(len(v) for v in parts.values())
    if total > budget:
        overflow = total - budget
        parts["H"] = parts["H"][:-overflow] if overflow < len(parts["H"]) else ""
    assert sum(len(v) for v in parts.values()) <= budget
    return parts

def allowed_tools(stage):
    gates = {
        "research": {"search", "outline"},
        "draft": {"outline", "draft", "refine"},
        "review": {"lint", "fact_check", "refine"},
    }
    return gates.get(stage, set())

@dataclass
class AgentState:
    goal: str
    stage: str
    constraints: str
    evidence: list = field(default_factory=list)
    history: list = field(default_factory=list)

def can_use_tool(agent_state, tool_name):
    return tool_name in allowed_tools(agent_state.stage)

agent = AgentState(
    goal="写一段关于分布式链路追踪技术原理的说明，300字内",
    stage="draft",
    constraints="正式语气；解释原理；避免营销措辞",
    evidence=["链路追踪用于记录请求在多服务间的传播路径。", "Trace 由多个 Span 组成。"],
    history=["先解释定义。", "再说明为什么需要上下文传播。"]
)

ctx = build_context(
    goal=agent.goal,
    state={"stage": agent.stage, "constraints": agent.constraints},
    evidence=agent.evidence,
    history=agent.history,
    budget=220,
    sub_budget={"I": 80, "S": 60, "E": 50, "H": 30},
)

assert sum(len(v) for v in ctx.values()) <= 220
assert can_use_tool(agent, "refine") is True
assert can_use_tool(agent, "search") is False
```

这段代码里有两个工程点值得注意。

第一，阶段门控。门控，白话说就是“当前阶段只允许做当前阶段该做的事”。例如 `draft` 阶段允许改写和起草，但不应该再大规模检索，否则系统会在写作途中不断跳回调研阶段，形成工具环路。

第二，历史剪枝。剪枝，白话说就是“保留最有用的一小部分，删掉其余部分”。写作历史不能无限累积，否则系统会越来越像在阅读日志，而不是执行当前任务。

真实工程例子可以这样理解。一个团队把写作助手接进内部编辑器，要求它为技术设计文档生成“背景”“方案”“风险”三节。这个系统通常不会让单个 Agent 同时负责检索、起草、审批和发布，而是把职责拆开：编辑器内 Agent 负责建议和初稿，校验器负责格式与术语检查，后台治理系统负责记录调用日志和失败原因。这样做不是保守，而是为了防止一个 Agent 同时拥有太多权限，导致失控后无法定位责任。

---

## 工程权衡与常见坑

写作助手 Agent 的第一类权衡是“灵活性”与“可控性”。越开放的目标，越容易显得聪明；但越开放，也越难收敛。对生产系统而言，通常宁可少一点自由，也要换来可预测的完成路径。

常见坑可以整理成一张表：

| 陷阱 | 现象 | 根因 | 规避方式 |
| --- | --- | --- | --- |
| 目标无界 | 一直改写不停止 | 没有终止条件 | 设轮数上限、字数上限、完成判定 |
| 上下文膨胀 | 忘记原始格式约束 | 证据和历史挤占预算 | 分层预算、编译上下文包 |
| 工具误用 | 草稿阶段反复检索 | 没有阶段门控 | 工具白名单按阶段开放 |
| 约束漂移 | 语气、受众、格式逐轮偏掉 | 状态没有固定保留 | 把关键约束放入状态区并重复校验 |
| 黑盒失败 | 结果不好但不知道为何 | 没有日志和失败分类 | 记录每轮输入、工具调用、校验结果 |

一个典型失败模式是**context drift**。漂移，白话说就是“系统一开始懂你的意思，后面慢慢偏掉”。例如用户要求“正式语气、面向初级工程师、解释原理不讲商业价值”，Agent 初稿做对了，但在第三轮改写时开始加入营销口吻和产品优势。这不是单次生成能力不足，而是系统没有在每轮继续检查约束。

所以成熟实现一般会加一个“失败架构”。失败架构，白话说就是“先假设系统会错，并把错误变成可观察事件”。例如每轮输出后跑一次格式校验、术语校验、长度校验，若失败就记录原因，不直接继续抽样。这样你才能知道问题是预算分配不对、工具调用顺序错了，还是检索证据本身不可靠。

对初级工程师最实用的结论是：不要把 prompt 当成总开关。写作助手 Agent 的大部分失败，都来自流程设计失控，而不是某一句提示词少了两个限定词。

---

## 替代方案与适用边界

并不是所有写作任务都值得上完整 Agent。对于线性、重复、低风险任务，固定模板加规则校验往往更稳。模板，白话说就是预先定义输出结构；规则校验，就是用确定性程序检查长度、标题、术语或字段是否齐全。

可以用对照表快速判断：

| 方案 | 适合场景 | 优点 | 缺点 |
| --- | --- | --- | --- |
| 编辑器内写作 Agent | 需要交互式起草、改写、补全 | 体验自然、辅助强 | 需要严格门控 |
| PR Bot 审核 | 文档提交后的统一检查 | 审计清楚、可批量治理 | 交互不及时 |
| 后台报告系统 | 做全局质量监控和日志分析 | 便于追责和优化 | 不直接产出内容 |
| 模板 + Policy Layer | 固定格式、低风险文档 | 成本低、稳定性高 | 适应复杂任务较弱 |

真实工程里，一个更稳妥的架构通常是分层的。编辑器 Agent 只负责内容建议和语言润色；PR Bot 负责检查术语、链接、格式和敏感信息；后台报告系统统计失败类型、成本和回滚率。这样即使写作 Agent 失误，也不会直接把不合规内容送到最终出口。

因此，写作助手 Agent 的适用边界很清楚：当任务需要多阶段协作、上下文理解和交互式修订时，它有价值；当任务高度重复、格式固定、风险敏感时，确定性流程往往更好。不要把“更像智能体”误当成“更适合工程”。

---

## 参考资料

- Lee 等，《A Design Space for Intelligent and Interactive Writing Assistants》，CHI 2024 相关综述，讨论写作助手的任务、用户、技术、交互、生态五维设计空间。  
  https://www.researchgate.net/publication/380520055_A_Design_Space_for_Intelligent_and_Interactive_Writing_Assistants
- Chier Hu，《Context Engineering in Agent》，讨论上下文包、预算分配、子预算与工具治理。  
  https://medium.com/agenticais/context-engineering-in-agent-982cb4d36293
- Then New Stack，《Why Agentic LLM Systems Fail: Control, Cost, and Reliability》，讨论目标无边界、成本失控与可靠性问题。  
  https://thenewstack.io/why-agentic-llm-systems-fail-control-cost-and-reliability/
- Jaymin West，《Debugging Agents》，讨论把失败做成可诊断事件的工程实践。  
  https://www.jayminwest.com/agentic-engineering-book/7-practices/1-debugging-agents
- Apptension，《AI-Assisted Development in 2026》，讨论编辑器、PR、后台三层协作与治理思路。  
  https://apptension.com/blog/ai-assisted-development-in-2026
- Maisum Hashim，《Why Prompt Engineering Won’t Fix AI Agent Architecture》，讨论为什么结构设计比提示词堆叠更关键。  
  https://www.maisumhashim.com/blog/why-prompt-engineering-wont-fix-ai-agent-architecture
