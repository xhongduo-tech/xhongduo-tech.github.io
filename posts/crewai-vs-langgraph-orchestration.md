## 核心结论

CrewAI 与 LangGraph 的差别，核心不在“能不能做多 Agent”，而在“你把系统当什么来建”。

CrewAI 更像“流程中心”的编排框架。流程中心的意思是，你先定义角色、任务、执行顺序，再让框架按顺序或按管理层级去推进。这样写出来的代码接近“一个团队如何分工”，对初学者更直观，尤其适合 Planner、Researcher、Reviewer 这类角色清晰的协作任务。

LangGraph 更像“状态中心”的编排框架。状态中心的意思是，系统的核心对象不是任务列表，而是一份显式的共享状态 `State`，每个节点只负责读取状态、产出状态更新，边负责决定下一步走向。它更适合分支多、可恢复、可审计、需要重放执行过程的复杂流程。

可以先看三维对比：

| 维度 | CrewAI | LangGraph |
|---|---|---|
| 编排范式 | 声明式团队协作：`Agent + Task + Crew + Process` | 图式编排：`StateGraph + Node + Edge + Reducer` |
| 状态管理 | 偏隐式，主要靠任务上下文、消息传递、memory | 显式，状态字段、更新规则、合并规则都写出来 |
| 控制流 | `Process.sequential` / `Process.hierarchical` 为主 | 普通边、条件边、循环、并行 fan-out/fan-in |
| 调试方式 | 看任务链、回调、观察 manager 行为 | 看状态快照、节点输出、checkpoint、重放 |
| 适合场景 | 角色驱动任务、快速原型、开放式协作 | 复杂状态流转、合规流程、长链路恢复 |
| 主要代价 | 分支可控性弱于显式图 | 前期设计成本和心智负担更高 |

一个直接判断标准是：

- 如果你脑中先出现的是“谁来做什么”，优先想 CrewAI。
- 如果你脑中先出现的是“状态有哪些字段、在哪些分叉点变化”，优先想 LangGraph。

玩具例子可以帮助建立直觉。

假设你要写一篇文章。用 CrewAI 的想法是：作者起草，审稿人检查，编辑润色，按队列推进。用 LangGraph 的想法是：状态里有 `draft`、`review_notes`、`approved`、`final_text` 四个字段，每个节点更新其中一部分，只有 `approved=True` 才能进入发布节点。前者像“团队协作表”，后者像“流程状态机”。状态机就是“系统在有限个状态之间按规则跳转”的模型。

---

## 问题定义与边界

多 Agent 编排不是简单地把多个大模型放在一起。它至少要解决四件事：任务分工、上下文传递、状态一致性、错误恢复。

“编排”这个词第一次出现时可以理解成一句白话：规定多个 Agent 什么时候执行、看什么输入、把结果交给谁。框架的设计差异，最终都会落到这四件事上。

本文只比较两个框架在三条主线上的差异：

| 本文讨论 | 具体问题 |
|---|---|
| 编排范式 | 你是先定义“角色和任务”，还是先定义“图和状态” |
| 状态管理 | 中间结果是隐式流动，还是显式建模 |
| 控制流 | 分支、循环、并行、回退由谁控制 |

本文不展开比较模型兼容性、商业版本功能、前端可视化、部署价格，也不讨论单 Agent 框架。

从系统边界看，CrewAI 与 LangGraph 都不是模型本身，而是“运行时外壳”。运行时就是负责执行程序的那层机制。它们都可以接 LLM、工具、检索、记忆，但对“系统如何前进”的理解不同：

- CrewAI 假设你更关心角色协作。
- LangGraph 假设你更关心状态演化。

可以把这两者放进一个统一架构里理解：

$$
\text{Multi-Agent System} = \text{Agents} + \text{Tools} + \text{Memory} + \text{Orchestrator}
$$

其中 `Orchestrator` 就是编排器。CrewAI 和 LangGraph 都在实现最后这一层，但实现方式不同。

真实工程里，这个差别会非常大。比如“旅行规划”任务，需求常常是开放的：先搜资料，再出方案，再复核，允许中途补充。这里角色协作通常比严格状态图更自然。反过来，像“合规审批”或“城市交通信号控制”，系统必须知道每个决策点之前的状态、谁改了什么、失败后从哪里恢复，这时显式状态图就更可靠。

---

## 核心机制与推导

### 1. CrewAI：过程流

CrewAI 的最小心智模型是：

1. 定义 Agent。
2. 定义 Task。
3. 用 `Crew` 把它们装起来。
4. 选择 `Process` 决定怎么执行。

`Process.sequential` 的语义最清楚：任务按列表顺序推进，前一个任务的输出成为后一个任务的上下文。上下文就是“后续步骤可读取的前置结果”。

如果抽象成公式，CrewAI 更接近：

$$
C_{i+1} = f(T_i, C_i)
$$

这里 $C_i$ 是第 $i$ 步已有上下文，$T_i$ 是第 $i$ 个任务。也就是“任务推动上下文往前流”。

`Process.hierarchical` 再加一个 manager。manager 的职责是规划、委派、校验。但注意，这种控制更依赖 manager 的提示词和框架的内部策略，而不是开发者完全显式定义的边结构。所以它可读，但不如图式控制那样强约束。

### 2. LangGraph：状态流

LangGraph 的最小心智模型是：

1. 先定义共享状态 `State`。
2. 每个节点读取旧状态，返回一个“状态更新”。
3. 通过边和条件边决定下一个节点。
4. 用 reducer 合并多个更新。

reducer 第一次出现时可以理解成一句白话：当多个节点都想改同一个字段时，规定怎么合并，而不是直接覆盖。

它的抽象公式可以写成：

$$
S_{next} = \operatorname{Reducer}(S_{current}, \Delta S_{node})
$$

如果存在并行分支，那么一个 superstep 内会有多个更新：

$$
S_{next} = \operatorname{Reducer}(S_{current}, \Delta S_1, \Delta S_2, \ldots, \Delta S_n)
$$

superstep 可以白话理解为“一轮同时执行并一起提交的节点集合”。官方文档强调这一轮提交具有事务性。事务性就是“要么这一轮都成功并提交，要么这一轮不提交”。这正是 LangGraph 能做恢复、重放、checkpoint 的基础。

### 3. 玩具例子：A → B/C → D

假设状态只有一个列表 `aggregate=[]`。

- 节点 A 输出 `["A"]`
- 然后并行走到 B 和 C
- B 输出 `["B"]`
- C 输出 `["C"]`
- D 汇总

如果 reducer 是列表拼接，那么状态变化是：

$$
[] \xrightarrow{A} [A] \xrightarrow{B,C} [A,B,C]
$$

这里最重要的不是“最后得到什么”，而是“为什么不会乱”。因为合并规则是显式定义的，不靠节点之间私下传话。

这就是两种框架最本质的区别：

- CrewAI 主要问：任务交给谁做。
- LangGraph 主要问：状态如何变化。

---

## 代码实现

先给一个可运行的玩具实现。它不依赖 CrewAI 或 LangGraph，但分别模拟“过程流”和“状态流”的核心差别。

```python
from typing import Dict, List

def crewai_like_pipeline(topic: str) -> Dict[str, str]:
    context = {"topic": topic}

    def planner(ctx):
        return {"outline": f"{ctx['topic']} 的三段式提纲"}

    def writer(ctx):
        return {"draft": f"根据《{ctx['outline']}》写出的初稿"}

    def reviewer(ctx):
        passed = "三段式" in ctx["outline"] and "初稿" in ctx["draft"]
        return {"approved": str(passed)}

    for task in (planner, writer, reviewer):
        context.update(task(context))
    return context

def langgraph_like_pipeline(topic: str) -> Dict[str, object]:
    state = {
        "topic": topic,
        "aggregate": [],
        "approved": False,
    }

    def reducer_list(old: List[str], update: List[str]) -> List[str]:
        return old + update

    def node_a(s):
        return {"aggregate": ["A"]}

    def node_b(s):
        return {"aggregate": ["B"]}

    def node_c(s):
        return {"aggregate": ["C"]}

    def node_d(s):
        ok = s["aggregate"] == ["A", "B", "C"]
        return {"approved": ok}

    for update in (node_a(state), node_b(state), node_c(state)):
        if "aggregate" in update:
            state["aggregate"] = reducer_list(state["aggregate"], update["aggregate"])

    state.update(node_d(state))
    return state

crew_result = crewai_like_pipeline("多Agent编排")
graph_result = langgraph_like_pipeline("多Agent编排")

assert crew_result["approved"] == "True"
assert graph_result["aggregate"] == ["A", "B", "C"]
assert graph_result["approved"] is True
```

这段代码反映了两个核心差异：

- `crewai_like_pipeline` 通过 `context.update(...)` 让任务链自然前进。
- `langgraph_like_pipeline` 通过 reducer 控制共享状态如何合并。

下面是接近真实框架 API 形态的最小示例。

CrewAI 风格：

```python
from crewai import Agent, Task, Crew, Process

planner = Agent(role="Planner", goal="拆解需求")
writer = Agent(role="Writer", goal="完成正文")
reviewer = Agent(role="Reviewer", goal="检查事实和结构")

tasks = [
    Task(description="先给出文章提纲", agent=planner),
    Task(description="根据提纲写初稿", agent=writer),
    Task(description="检查初稿并修正", agent=reviewer),
]

crew = Crew(
    agents=[planner, writer, reviewer],
    tasks=tasks,
    process=Process.sequential,
    memory=True,
)

result = crew.kickoff()
```

LangGraph 风格：

```python
from typing_extensions import TypedDict, Annotated
from operator import add
from langgraph.graph import StateGraph, START, END

class State(TypedDict):
    aggregate: Annotated[list[str], add]
    approved: bool

def a(state: State):
    return {"aggregate": ["A"]}

def b(state: State):
    return {"aggregate": ["B"]}

def c(state: State):
    return {"aggregate": ["C"]}

def d(state: State):
    return {"approved": state["aggregate"] == ["A", "B", "C"]}

builder = StateGraph(State)
builder.add_node("a", a)
builder.add_node("b", b)
builder.add_node("c", c)
builder.add_node("d", d)

builder.add_edge(START, "a")
builder.add_edge("a", "b")
builder.add_edge("a", "c")
builder.add_edge("b", "d")
builder.add_edge("c", "d")
builder.add_edge("d", END)

graph = builder.compile()
result = graph.invoke({"aggregate": [], "approved": False})
```

同样是“四步任务”，CrewAI 代码强调“谁做什么”，LangGraph 代码强调“状态如何过图”。

---

## 工程权衡与常见坑

### 1. CrewAI 的优势和坑

CrewAI 的优势是低门槛。对初学者来说，`Agent`、`Task`、`Crew` 这组概念和真实团队分工高度一致，代码也短。

但它的主要风险是：复杂分支往往不够显式。

官方文档明确给出的是 `sequential` 和 `hierarchical` 两类 `Process`。这适合“有主线”的任务，但当你需要细粒度条件边、回退、部分重试、分支结果合并时，如果仍只靠 manager 提示词控制，很容易出现三类问题：

| 常见坑 | 原因 | 规避方式 |
|---|---|---|
| 不该执行的任务也被执行 | 分支逻辑写在提示词里，不在显式边里 | 把强条件前置为代码判断，减少纯提示词路由 |
| 后一步覆盖前一步结果 | 结果合并规则不显式 | 为任务输出设计结构化 schema，不只传自然语言 |
| 调试困难 | 中间状态分散在消息和 memory 中 | 开启 step/task callback，保留每步输入输出 |

社区案例显示，`hierarchical` 在一些场景下需要非常明确的 manager 指令，才能避免“所有任务都跑一遍”或“后任务覆盖前任务”的问题。这里要把它理解为工程事实：拟人化协作越强，可预测性通常越弱。

### 2. LangGraph 的优势和坑

LangGraph 的优势是强控制。它把状态、边、分支、checkpoint 都放在显式模型里，因此可以做：

- 中断后恢复
- 失败后从 checkpoint 重启
- 查看历史状态快照
- 并行分支后按 reducer 合并
- 时间回放和审计

但代价也清楚：设计复杂。

你必须提前想清楚三件事：

1. `State` 到底有哪些字段。
2. 每个字段是覆盖、追加还是自定义合并。
3. 哪些节点之间允许转移。

如果前面没设计好，后面改图会很痛苦。尤其是字段一旦混乱，图会迅速退化成“复杂但不可维护的状态泥团”。

### 3. 真实工程例子

真实工程里，一个典型对比是：

- 旅行策划助手：用户目标模糊，允许探索和反复修正。Planner 负责拆解城市与时间安排，Researcher 查机酒与签证，Reviewer 做一致性检查。这里 CrewAI 更自然。
- 智能交通控制：每个路口信号策略都依赖实时状态、历史状态、异常状态，并且每次决策必须可追踪、可恢复、可审计。这里 LangGraph 更自然，因为每个分支点都可以被显式建成节点和条件边。

一句话概括：开放式协作偏 CrewAI，严格状态流转偏 LangGraph。

---

## 替代方案与适用边界

选择不是二选一，而是三选一：只用 CrewAI、只用 LangGraph、混合使用。

可以用下表做决策：

| 需求特征 | 更适合 CrewAI | 更适合 LangGraph | 更适合混合 |
|---|---|---|---|
| 目标模糊、需要角色协作 | 是 | 否 | 可选 |
| 强分支、强状态一致性 | 否 | 是 | 可选 |
| 需要 checkpoint / replay / 审计 | 弱 | 强 | 强 |
| 快速原型 | 强 | 中 | 中 |
| 合规审批、长流程恢复 | 弱 | 强 | 强 |
| 开放性子任务嵌入到大流程 | 中 | 中 | 强 |

混合方案在工程上很常见，也最实用：

1. 用 LangGraph 固定主流程，例如“审核 → 执行 → 复核 → 归档”。
2. 把“执行”节点做成一个 CrewAI crew。
3. 让 CrewAI 在这个节点内部完成 Planner、Executor、Reviewer 的开放式协作。
4. 协作结果再回写到 LangGraph 的显式状态里。

这样做的好处是：

- 全局控制流仍然是确定的。
- 局部开放任务仍然保留角色协作的灵活性。

适用边界也要说清楚。

如果你的系统必须回答“为什么这一步会走到这里、失败后如何从第 17 步恢复、谁改了状态字段 `risk_level`”，那就不要把主控制流放在 CrewAI 的隐式协作上。反过来，如果你的系统本质是“让多个专业角色一起讨论出一个还不错的答案”，那就没必要一开始就上完整状态图。

---

## 参考资料

- [CrewAI 官方文档：Processes](https://docs.crewai.com/en/concepts/processes)
- [CrewAI 官方文档：Sequential Process](https://docs.crewai.com/en/learn/sequential-process)
- [CrewAI 官方文档：Crews](https://docs.crewai.com/en/concepts/crews)
- [CrewAI 官方文档：Flows](https://docs.crewai.com/en/concepts/flows)
- [LangGraph 官方文档：Use the graph API（Python）](https://docs.langchain.com/oss/python/langgraph/use-graph-api)
- [LangGraph 官方文档：Persistence（Python）](https://docs.langchain.com/oss/python/langgraph/persistence)
- [LangGraph 官方文档：Checkpointer integrations](https://docs.langchain.com/oss/python/integrations/checkpointers)
- [ZenML：LangGraph vs CrewAI](https://www.zenml.io/blog/langgraph-vs-crewai)
- [Preprints：Multi-Agent AI Systems for Biological and Clinical Data Analysis](https://www.preprints.org/manuscript/202512.2602/v1)
- [BARD AI：Why CrewAI’s Manager-Employee Architecture Fails](https://bardai.ai/2025/11/25/why-crewais-manager-employee-architecture-fails-and-the-best-way-to-fix-it/)
