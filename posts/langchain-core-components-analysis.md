## 核心结论

LangChain 的核心价值，不是“帮你多调一次模型”，而是把模型调用组织成一套可控工作流。这个工作流通常由 5 个部件构成：

| 组件 | 直白解释 | 主要职责 | 典型输入 | 典型输出 |
| --- | --- | --- | --- | --- |
| Chain | 把步骤串起来的执行器 | 顺序执行、分支选择、变量传递 | 用户输入、上游结果 | 结构化结果或文本 |
| Agent | 会自己决定下一步做什么的循环控制器 | 选工具、执行、再判断是否继续 | 当前状态、可用工具 | 最终答案或中间动作 |
| Memory | 给系统“留痕”的状态层 | 读取历史、保存新状态 | 本轮输入、历史上下文 | 注入后的上下文变量 |
| Callback | 监听运行过程的钩子 | 记录日志、计时、追踪错误 | 运行事件 | 日志、指标、追踪数据 |
| 扩展点 | 留给工程方接入自定义能力的接口 | 自定义 Chain、Tool、Memory、Tracer | 框架约定对象 | 定制行为 |

对新手可以先记一句话：`Chain 负责排步骤，Agent 负责选动作，Memory 负责记历史，Callback 负责看过程。`

从执行模型上看，顺序 Chain 满足：

$$
I_{i+1}=O_i
$$

也就是第 $i$ 步的输出，直接成为第 $i+1$ 步的输入。Agent 则不是一次走完，而是一个循环：

$$
a_t=\pi_{\text{LLM}}(s_t),\quad
o_t=\text{Tool}(a_t),\quad
s_{t+1}=U(s_t,a_t,o_t)
$$

直到停止条件成立：

$$
stop(s_t)=\text{True}
$$

这套抽象的工程意义在于，你可以在三个层级上控系统：单个步骤怎么跑，整条链怎么连，整个多轮决策怎么停。源码层面看，LangChain 把“执行”“状态”“观察”拆开了，所以可测试性、可监控性、可复用性都比手写一大段 prompt glue code 更强。

---

## 问题定义与边界

本文讨论的问题是：LangChain 如何把一次模型调用扩展成“可串联、可分支、可记忆、可观测”的完整工作流。

更具体地说，目标不是分析底层大模型参数，也不是分析某个具体搜索工具，而是分析以下交互关系：

| 组件 | 作用 | 输入边界 | 输出边界 |
| --- | --- | --- | --- |
| Chain | 组织固定流程 | 接收结构化输入字典 | 返回结构化输出字典 |
| Agent | 执行决策循环 | 接收当前状态和工具集合 | 产生动作或最终答案 |
| Memory | 提供跨轮状态 | 在调用前后读写变量 | 不直接决定流程 |
| Callback | 监听生命周期 | 接收 start/end/error 等事件 | 不改业务结果，只做旁路处理 |

如果把它画成一个最简流程，可以理解成：

`用户输入 -> Chain/Agent -> Tool/LLM -> 输出`

同时有两条侧边线：

`Memory` 在执行前加载上下文、执行后保存结果  
`Callback` 在每个生命周期节点接收事件

这里有两个边界必须说清：

1. Chain 偏“确定流程”。如果你已经知道步骤顺序，用 Chain 比 Agent 更稳定。
2. Agent 偏“开放决策”。如果让模型自己选工具，就必须额外设计 stop 条件、最大轮数、权限边界。

一个新手常见问题是：“我要让模型查文档、算数字、记住上文，LangChain 怎么保证每步都在控制里？”答案是：控制并不来自模型本身，而来自这套运行时分层。模型只负责部分决策，不负责整个工程系统。

另外要注意版本边界。经典 `Chain`、`SequentialChain`、`AgentExecutor` 仍然是理解源码结构的好入口；而 LangChain v1 的现代 Agent 运行时已经明显向 LangGraph 靠拢，状态、图执行、持久化更强。理解本文时，可以把前者看成“经典抽象”，把后者看成“新运行时实现”。

---

## 核心机制与推导

先看最简单的玩具例子。假设输入是数字 $x$：

- 第一步加 2，得到 $O_1=x+2$
- 第二步乘 3，得到 $O_2=3\cdot O_1$

若 $x=5$，则：

$$
O_1=5+2=7,\quad O_2=3\times7=21
$$

这就是最标准的顺序 Chain。它的关键不是“加法”和“乘法”，而是“上一步输出自动喂给下一步”。源码里 `SequentialChain` 的核心思想正是维护一个变量字典，把每步返回的新键合并进去，再交给后续步骤使用。

分支机制则是在中间插入一个条件函数 $f(I_k)$。例如：

- 若问题里包含“计算”，走算术工具链
- 若问题里包含“总结”，走文档总结链

形式化写成：

$$
C_{next}=
\begin{cases}
C_{math}, & f(I_k)=\text{math} \\
C_{summary}, & f(I_k)=\text{summary}
\end{cases}
$$

再看 Agent。Agent 不是把路径提前写死，而是每轮让模型根据当前状态决定动作。这里的“状态”可以简单理解成“到目前为止系统知道的全部信息”，包括消息历史、工具输出、额外变量。它通常至少包含：

- 当前用户请求
- 历史对话
- 已执行过的动作和观察
- 终止条件相关信息，例如迭代次数

下面这个对照表可以帮助理解四个组件在同一轮中的分工：

| 时刻 | Chain/Agent 在做什么 | Memory 在做什么 | Callback 在做什么 |
| --- | --- | --- | --- |
| 开始前 | 接收输入，准备执行 | `load` 历史变量 | `on_*_start` 记录开始 |
| 中间步 | 调用模型或工具 | 一般不改长期状态 | 记录 token、耗时、工具调用 |
| 结束后 | 整理最终输出 | `save` 输入输出 | `on_*_end` 或 `on_*_error` |

真实工程例子更能说明问题。假设你做一个企业内部文档问答系统：

1. Chain A 先做检索，把知识库里相关文档取出来。
2. Chain B 把检索结果总结成结构化摘要。
3. Agent 根据摘要决定是否还需要调用数据库工具、日历工具、权限检查工具。
4. Memory 记住这个用户偏好“回答要简洁，偏技术风格”。
5. Callback 把每一步的模型耗时、工具耗时、错误码都发到追踪系统。

这个系统一旦出错，你能知道是检索没命中、工具超时、还是 Agent 在循环里反复选错工具。这就是 Callback 和状态拆分的意义。没有这层拆分，系统只能表现为“模型答错了”，但你根本不知道错在哪一环。

---

## 代码实现

先用一个不依赖 LangChain 的极简 Python 实现，模拟 `SequentialChain + Agent + Memory + Callback` 的核心思路。代码能直接运行，重点是把源码机制看清楚。

```python
from dataclasses import dataclass, field

@dataclass
class SimpleMemory:
    store: dict = field(default_factory=dict)

    def load(self):
        return dict(self.store)

    def save(self, inputs, outputs):
        self.store["last_input"] = inputs
        self.store["last_output"] = outputs

class LoggerCallback:
    def __init__(self):
        self.events = []

    def on_start(self, name, payload):
        self.events.append(("start", name, payload))

    def on_end(self, name, payload):
        self.events.append(("end", name, payload))

def add_two_step(ctx):
    return {"value": ctx["value"] + 2}

def multiply_three_step(ctx):
    return {"value": ctx["value"] * 3}

def sequential_chain(value, memory, cb):
    ctx = {"value": value}
    ctx.update(memory.load())

    cb.on_start("chain", {"input": value})
    out1 = add_two_step(ctx)
    ctx.update(out1)

    out2 = multiply_three_step(ctx)
    ctx.update(out2)
    cb.on_end("chain", {"output": ctx["value"]})

    memory.save({"value": value}, {"value": ctx["value"]})
    return ctx["value"]

def calculator_tool(x):
    return x * 2

def simple_agent(question, memory, cb, max_steps=3):
    state = {
        "question": question,
        "history": memory.load(),
        "steps": 0,
        "answer": None,
    }

    cb.on_start("agent", {"question": question})

    while state["steps"] < max_steps:
        state["steps"] += 1

        if "double" in question:
            observation = calculator_tool(6)
            state["answer"] = f"tool_result={observation}"
            break
        else:
            state["answer"] = "no_tool_needed"
            break

    cb.on_end("agent", {"answer": state["answer"], "steps": state["steps"]})
    memory.save({"question": question}, {"answer": state["answer"]})
    return state["answer"]

memory = SimpleMemory()
cb = LoggerCallback()

result = sequential_chain(5, memory, cb)
assert result == 21

answer = simple_agent("please double the number", memory, cb)
assert answer == "tool_result=12"

assert cb.events[0][0] == "start"
assert cb.events[-1][0] == "end"
```

这段代码对应 LangChain 源码里的几个关键点：

| 抽象 | 直白解释 | 典型类/方法 |
| --- | --- | --- |
| `Chain` | 一个能接收输入并产出输出的可组合单元 | `Chain`, `invoke`, `run` |
| `SequentialChain` | 维护中间变量并顺序传递 | `SequentialChain` |
| `AgentExecutor` / `create_agent` 运行时 | 控制“模型决策 -> 工具执行 -> 状态更新”循环 | `AgentExecutor`, `create_agent` |
| `BaseMemory` | 规定 memory 必须实现加载和保存 | `load_memory_variables`, `save_context` |
| `BaseCallbackHandler` / tracer | 监听生命周期事件 | `on_chain_start`, `on_chain_end`, `on_tool_start` |

如果写成伪码，经典执行路径大致是：

```text
inputs = merge(user_inputs, memory.load())
callbacks.on_chain_start(inputs)

for chain in chains:
    outputs = chain.invoke(inputs)
    inputs.update(outputs)

memory.save(inputs, outputs)
callbacks.on_chain_end(outputs)
return outputs
```

Agent 则更像：

```text
state = init(user_input, memory.load())
while not stop(state):
    action = llm_plan(state)
    observation = run_tool(action)
    state = update(state, action, observation)

memory.save(user_input, state)
return final_answer(state)
```

这里最重要的源码设计，不是某个类名，而是两个接口约定：

1. Memory 不负责“决定流程”，只负责“提供和持久化状态”。
2. Callback 不负责“修改业务值”，只负责“观察运行过程”。

这两个约定让主流程保持清晰，也让日志、追踪、A/B 测试、告警可以独立演化。

---

## 工程权衡与常见坑

源码抽象好理解，但落到工程里最容易出问题的是边界控制。

| 坑 | 影响 | 规避策略 |
| --- | --- | --- |
| 不加 Memory | 多轮对话断裂，系统无法引用上文 | 至少保存消息历史或用户偏好 |
| Memory 无裁剪 | 长对话把上下文窗口撑爆 | 做窗口截断、摘要化、分层记忆 |
| Agent 无 stop 条件 | 反复调用工具，形成死循环 | 设最大步数、最终答案判定、超时 |
| 工具权限过大 | 可能执行危险操作 | 做白名单、参数校验、人审 |
| Callback 复用不隔离 | 并发请求日志串线 | 每次运行传入 run-specific callbacks / tags / metadata |
| 把简单流程也写成 Agent | 成本高、可测性差 | 固定路径优先用 Chain |

新手最容易忽略的是：Memory 不是“越多越好”。如果把所有对话原文都拼进提示词，模型会越来越慢，也越来越容易被无关历史干扰。更合理的做法通常是：

- 短期 memory 存最近若干轮消息
- 长期 memory 只存用户偏好、实体信息、任务摘要
- 真正需要检索的历史，放到向量库或外部存储里

另一个常见坑是 Agent 的停止条件。理论上模型会在觉得“答完了”时停，但工程上不能只靠这个假设。你至少要有一种硬约束，例如最大迭代次数 $T_{max}$：

$$
stop(s_t)=\mathbf{1}(t \ge T_{max}\ \text{or final\_answer found})
$$

否则只要模型持续输出“我再查一下”，系统就会一直烧 token。

真实工程里还有一个隐蔽问题：并发追踪。假设同一时刻 200 个请求都在跑，如果所有请求共用一套全局 logger，而没有 run id、tags、metadata，你看到的日志就会混在一起，排障几乎不可能。Callback 的正确用法不是“有日志就行”，而是“每次运行都能唯一定位”。

---

## 替代方案与适用边界

不是所有任务都需要完整的 LangChain 运行时。选型原则应该是“控制需求决定抽象层级”。

| 方案 | 何时用 | 何时省略 |
| --- | --- | --- |
| 单次模型调用 | 问答一次出结果 | 不需要多步流程 |
| 顺序 Chain | 步骤固定、顺序明确 | 不需要模型自主决策 |
| 分支 Chain / Router | 路径有限但要做条件选择 | 不需要开放式工具选择 |
| Agent | 需要动态选工具、多轮决策 | 流程能提前写死时不要上 |
| Agent + Memory + Callback | 多轮、可观测、要上线生产 | Demo 阶段可先简化 |
| 自定义扩展点 | 需要接企业日志、权限、监控、存储 | 单机实验可暂不做 |

可以用一句规则概括：

- 只需一步答案，用单 Chain 或直接模型调用。
- 需要固定多步处理，用顺序 Chain。
- 需要模型自己挑工具，用 Agent。
- 需要跨轮记偏好、记上下文，加 Memory。
- 需要生产排障和成本分析，加 Callback 或 LangSmith tracing。
- 需要接公司内部系统，再写自定义扩展点。

从源码分析角度看，LangChain 的设计并不神秘。它做的事，本质上是把一个复杂系统拆成五个稳定接口：执行、决策、状态、观察、扩展。真正的工程质量，不在于你用了多少组件，而在于你是否把该固定的地方固定住，把该开放的地方隔离好。

---

## 参考资料

| 资料 | 涵盖内容 | 读者收益 |
| --- | --- | --- |
| [LangChain `chains` 总览](https://api.python.langchain.com/en/latest/langchain/chains.html) | Chain 抽象、可组合性、Memory/Callbacks 入口 | 建立整体组件视图 |
| [SequentialChain 文档](https://api.python.langchain.com/en/latest/langchain/chains/langchain.chains.sequential.SequentialChain.html) | 顺序链、输入输出传递、运行时 callbacks | 理解固定流程的源码设计 |
| [Chain 基类文档](https://api.python.langchain.com/en/latest/langchain/chains/langchain.chains.base.Chain.html) | `run`/`invoke`、memory、callbacks 参数 | 看清公共执行接口 |
| [AgentExecutor 文档](https://api.python.langchain.com/en/latest/langchain/agents/langchain.agents.agent.AgentExecutor.html) | 经典 Agent 执行器、工具循环、回调传播 | 理解决策循环控制 |
| [LangChain Agents 文档](https://docs.langchain.com/oss/python/langchain-agents) | v1 Agent、LangGraph 运行时、状态 schema | 区分新旧运行时抽象 |
| [Memory overview](https://docs.langchain.com/oss/python/concepts/memory) | 短期记忆、长期记忆、状态持久化 | 理解 Memory 的工程边界 |
| [BaseMemory 文档](https://api.python.langchain.com/en/latest/core/memory/langchain_core.memory.BaseMemory.html) | `load_memory_variables`、`save_context` | 对齐自定义 Memory 接口 |
| [Callbacks 文档](https://api.python.langchain.com/en/latest/core/callbacks.html) | Callback handler 与 manager 类型 | 理解事件监听入口 |
| [LangSmith Observability](https://docs.langchain.com/oss/python/langchain/observability) | tracing、tags、metadata、可观测性 | 把 Callback 思路接到生产监控 |
