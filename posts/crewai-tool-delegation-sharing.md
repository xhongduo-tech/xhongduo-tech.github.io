## 核心结论

在 CrewAI 里，`allow_delegation=True` 的作用不是“让 Agent 更聪明”，而是**给 Agent 增加两类协作工具**：一个用于把子任务交给队友，一个用于向队友提问。工具是程序里可被模型调用的能力入口。换句话说，打开这个开关后，Agent 不只是会自己做事，还会把“其他 Agent 的专业能力”当成可调用资源。

可以把这个机制写成一个最小公式：

$$
\text{DelegationCapability}(A)=
\begin{cases}
\{\text{DelegateWorkTool}, \text{AskQuestionTool}\}, & allow\_delegation=True \\
\varnothing, & allow\_delegation=False
\end{cases}
$$

这说明委托能力不是模糊行为，而是显式能力集合。CrewAI 文档当前的公开说明是：当 `allow_delegation=True` 时，Agent 会自动获得 `Delegate work to coworker(task, context, coworker)` 和 `Ask question to coworker(question, context, coworker)` 两个工具。

对初学者最重要的结论有三条：

| 结论 | 白话解释 | 工程含义 |
|---|---|---|
| `allow_delegation` 是工具注入开关 | 打开后才能“找同事办事” | 没开就不会发生自动委派 |
| 协作工具和自定义工具走同一套规范 | 输入格式、执行方式都像普通 Tool | 便于统一监控、校验、调试 |
| 工具共享不是“全员全权” | 共享的是可见能力，不等于无限授权 | 需要 `allowed_agents`、层级设计和最小权限控制 |

一个新手版直观理解是：Researcher 和 Writer 都开启委派后，模型在推理时会看到“我既能自己做，也能把写作相关子任务交给 Writer，或者向 Researcher 追问事实”。这就是“把队友当工具”使用。

---

## 问题定义与边界

这里要解决的问题，不是“如何把多个 Agent 串起来”，而是**如何让某个 Agent 在执行中，把更适合的子任务交给更合适的队友**。串联是预先写死流程，委派是运行时决策。前者像固定流水线，后者像团队协作。

边界要先说清楚：

1. `allow_delegation` 控制的是**是否允许发起协作工具调用**。
2. 它不等于“自动共享所有底层工具权限”。
3. 它也不保证委派一定成功，模型仍要根据上下文决定是否调用。
4. 如果没有进一步限制，默认委派范围可能过宽，容易引发循环和选择混乱。

一个最小玩具例子是两人团队：

- `Researcher`：擅长查资料
- `Writer`：擅长整理成文

如果两者都设置 `allow_delegation=True`，那么流程可能是：

1. Researcher 收到“写一篇市场分析”的总任务。
2. 它先完成资料收集。
3. 它把“把材料整理成读者可读的结构化文章”委派给 Writer。
4. Writer 在信息不足时，再通过提问工具向 Researcher 追问某个数据点。

这个例子里，协作是合法的，因为双方都能调用协作工具。

但在工程里，合法不等于合理。下面这个表更重要：

| 角色 | `allow_delegation` | `allowed_agents` | 能否继续委派 | 典型用途 |
|---|---|---|---|---|
| Manager | True | `["Researcher", "Writer"]` | 能 | 统一调度 |
| Researcher | True | `["Writer"]` | 受限能 | 只把整理任务下发给 Writer |
| Writer | False | 无 | 不能 | 专注产出，避免回传循环 |

这里的 `allowed_agents` 可以理解为**允许委派的白名单**，也就是“这个 Agent 可以把任务交给谁”。白名单是安全里很常见的做法，意思是默认不放行，只有明确列出的目标才允许访问。

所以本文讨论的边界是：**同一 Crew 内的本地协作与工具共享**。如果你要跨服务、跨团队、跨部署环境委派，就已经进入 A2A 协议的适用范围，不属于单纯 `allow_delegation` 的边界。

---

## 核心机制与推导

CrewAI 这套设计成立，关键在于：**协作能力被做成 Tool，而不是做成特殊分支逻辑**。Tool 就是供大模型按名称和参数调用的函数式接口。这样做有两个直接结果：

1. 模型的决策入口统一。
2. 输入校验和执行生命周期统一。

自定义工具的规范通常是：

- 继承 `BaseTool`
- 声明 `args_schema`
- 实现 `_run`
- 可选实现 `_arun`

`args_schema` 可以理解为“参数合同”，也就是调用这个工具时，输入必须满足什么结构。`_run` 是同步执行入口，`_arun` 是异步执行入口。

因此，协作工具本质上也能抽象成下面这个统一模型：

$$
\text{ToolCall} = (\text{name}, \text{args\_schema}, \text{executor})
$$

把委派机制代入后，有：

$$
\text{DelegateWorkTool}=(\text{name}, \{task, context, coworker\}, \_run)
$$

$$
\text{AskQuestionTool}=(\text{name}, \{question, context, coworker\}, \_run)
$$

这意味着模型做的事情其实很朴素：根据当前任务状态，构造一个符合 schema 的参数对象，然后调用对应工具。工具内部再把请求转给指定 coworker。

可以把执行流简化成伪代码：

```text
if agent.allow_delegation:
    agent.tools += [DelegateWorkTool, AskQuestionTool]

llm_decide_next_action()

if action == "Delegate work to coworker":
    validate(task, context, coworker)
    send_subtask_to(coworker)

if action == "Ask question to coworker":
    validate(question, context, coworker)
    send_question_to(coworker)
```

这个统一机制解释了两个常见疑问。

第一，为什么说“跨 Agent 工具共享”不是直接把别人的 Python 函数塞给当前 Agent？  
因为共享的是**协作入口**，不是原始执行句柄。当前 Agent 不是直接拿到对方所有内部实现，而是通过“委派给某角色”这个受控接口访问其能力。

第二，为什么要强调 schema？  
因为如果没有明确输入约束，模型容易产生模糊调用，例如只说“帮我处理一下”，却不提供任务上下文。参数合同越清晰，调用成功率越高，监控也越稳定。

---

## 代码实现

下面先给一个简化但可运行的 Python 例子。它不是 CrewAI 源码，而是一个最小模型，用来演示“打开开关后自动获得两个协作工具”和“通过白名单限制委派”的核心机制。

```python
from dataclasses import dataclass, field
from typing import Callable, Dict, List


@dataclass
class Tool:
    name: str
    run: Callable[..., str]


@dataclass
class Agent:
    role: str
    allow_delegation: bool = False
    allowed_agents: List[str] = field(default_factory=list)
    tools: Dict[str, Tool] = field(default_factory=dict)

    def inject_collaboration_tools(self, registry: Dict[str, "Agent"]) -> None:
        if not self.allow_delegation:
            return

        def delegate_work(task: str, context: str, coworker: str) -> str:
            assert task.strip(), "task 不能为空"
            assert coworker in registry, "coworker 不存在"
            if self.allowed_agents:
                assert coworker in self.allowed_agents, "目标不在允许白名单"
            return f"{self.role} -> {coworker}: DELEGATE [{task}] with {context}"

        def ask_question(question: str, context: str, coworker: str) -> str:
            assert question.strip(), "question 不能为空"
            assert coworker in registry, "coworker 不存在"
            if self.allowed_agents:
                assert coworker in self.allowed_agents, "目标不在允许白名单"
            return f"{self.role} -> {coworker}: ASK [{question}] with {context}"

        self.tools["delegate"] = Tool("Delegate work to coworker", delegate_work)
        self.tools["ask"] = Tool("Ask question to coworker", ask_question)


researcher = Agent(role="Researcher", allow_delegation=True, allowed_agents=["Writer"])
writer = Agent(role="Writer", allow_delegation=False)

crew = {"Researcher": researcher, "Writer": writer}

for agent in crew.values():
    agent.inject_collaboration_tools(crew)

assert "delegate" in researcher.tools
assert "ask" in researcher.tools
assert "delegate" not in writer.tools

msg = researcher.tools["delegate"].run(
    task="整理调研结论为文章结构",
    context="主题是 CrewAI 工具委托",
    coworker="Writer",
)
assert "Researcher -> Writer" in msg

try:
    researcher.tools["delegate"].run(
        task="把任务交给不存在的人",
        context="bad case",
        coworker="Manager",
    )
    raise AssertionError("这里应该失败")
except AssertionError:
    pass
```

这个例子说明了三件事：

| 观察点 | 结果 | 含义 |
|---|---|---|
| `allow_delegation=True` | 自动注入 `delegate` 和 `ask` | 开关决定是否拥有协作工具 |
| `allow_delegation=False` | 没有协作工具 | 角色不能继续分发任务 |
| `allowed_agents=["Writer"]` | 只能委派给 Writer | 白名单限制委派范围 |

如果换成更接近 CrewAI 的写法，结构会像这样：

```python
from typing import Type
from pydantic import BaseModel, Field
from crewai.tools import BaseTool

class DelegateInput(BaseModel):
    task: str = Field(..., description="需要委派的子任务")
    context: str = Field(..., description="执行所需上下文")
    coworker: str = Field(..., description="目标同事角色名")

class DelegateWorkTool(BaseTool):
    name: str = "Delegate work to coworker"
    description: str = "将任务委派给指定队友"
    args_schema: Type[BaseModel] = DelegateInput

    def _run(self, task: str, context: str, coworker: str) -> str:
        return f"delegate {task} to {coworker} with {context}"
```

这里的重点不是“你要手写 CrewAI 内置协作工具”，而是理解：**CrewAI 的内置协作工具与自定义工具遵循同一种对象模型**。因此你在排查问题时，可以把它们当作同类实体看待，比如统一观察：

- 工具描述是否清晰
- 参数 schema 是否足够约束输入
- `_run` 或 `_arun` 是否有副作用
- 工具调用日志是否可追踪

真实工程例子可以设成一个内容生产团队：

- `Manager`：接收业务目标，拆分任务
- `Researcher`：检索资料，验证事实
- `Writer`：写草稿，整理结构

推荐配置是：

- `Manager.allow_delegation=True`
- `Researcher.allow_delegation=True`
- `Writer.allow_delegation=False`
- `Researcher.allowed_agents=["Writer"]`
- `Manager.allowed_agents=["Researcher", "Writer"]`

这样做的执行节奏是：Manager 只做调度，Researcher 在研究完成后把成文任务交给 Writer，Writer 不再继续向上或横向乱委派。路径短，权限清晰，成本可控。

---

## 工程权衡与常见坑

最常见的错误不是不会开委派，而是**把委派开成无边界广播**。这会同时带来正确性问题和成本问题。

下面这个风险表最实用：

| 问题 | 典型症状 | 根因 | 规避措施 |
|---|---|---|---|
| 委派循环 | A 交给 B，B 又交还给 A | 全员都能继续委派 | 只让上层角色开委派，或限制 `allowed_agents` |
| 选择瘫痪 | 模型反复犹豫选谁 | 可选目标过多 | 用白名单缩小候选范围 |
| 上下文丢失 | 队友收到任务但无法执行 | `context` 不完整 | 把任务目标、输入、约束写入 context |
| 权限扩散 | 低权限任务间接触发高风险工具 | 工具可见性与委派路径未隔离 | 最小权限、白名单、敏感工具隔离 |
| 成本失控 | Token 和调用次数异常增长 | 无节制多跳协作 | 限制层级、监控 step callback、减少 re-delegation |

其中“权限扩散”最容易被忽视。因为从抽象层看，当前 Agent 只是“让别人做事”，但从系统层看，这可能意味着**通过委派链间接触达更多工具**。如果某个队友挂着高风险 MCP 工具，比如能访问数据库、执行命令、改外部系统状态，那么委派就不只是组织协作问题，而是安全问题。

CrewAI 的 MCP 安全文档强调了几个原则，翻成工程语言就是：

1. 只连接可信 MCP 服务。
2. 对外部工具元数据保持警惕，因为工具描述本身也可能成为提示注入入口。
3. 对敏感工具做严格访问控制。
4. 委派链一长，审计和权限边界就必须更清楚。

一个简单判断标准是：**协作工具解决的是“谁来做”，不是“谁都能做”**。如果系统设计让低价值 Agent 能通过多跳委派触发高风险工具，那设计已经偏了。

还有一个实践坑是把 backstory 当权限系统。backstory 是提示词背景，只能影响模型偏好，不能替代硬约束。真正的硬约束应该来自：

- `allow_delegation`
- `allowed_agents`
- 工具可见性控制
- MCP/平台侧鉴权与 RBAC

---

## 替代方案与适用边界

`allow_delegation` 适合的是**同一 Crew 内部的本地协作**。如果问题超出这个边界，就要换方案。

| 方案 | 适用场景 | 优点 | 限制 |
|---|---|---|---|
| `allow_delegation` 本地协作 | 同一 Crew 内多角色分工 | 配置简单，自动协作 | 主要解决本地团队内委派 |
| `allowed_agents` + 层级管理 | 任务链明确，需要管控 | 层级清晰，避免循环 | 灵活性低于完全自由协作 |
| A2A 协议 | 跨服务、跨组织、远端 Agent | 能接远端专家或远端系统 | 配置更复杂，需要网络与认证 |

A2A 可以理解为“把委派目标从本地同事，扩展成远端代理服务”。CrewAI 当前文档说明里，Agent 在配置了 A2A 能力后，可以在本地执行和远端委派之间自主选择。这就适合下面的真实场景：

- 本地团队负责内容组织
- 远端有一个专门做法规检索的 Agent 服务
- 当前任务涉及法规引用
- 本地 Agent 发现自己没有足够工具，就把任务委派给远端 A2A Agent

这时你不该再把问题看成“Crew 内共享工具”，而应看成“跨系统代理协作”。

因此可以用一句话区分三者：

- 同一 Crew、同一运行边界内协作，用 `allow_delegation`
- 同一 Crew 但需要清晰上下级和白名单，用 `allowed_agents` + 层级流程
- 跨部署边界、跨组织或远端服务协作，用 A2A

---

## 参考资料

- CrewAI Collaboration 文档：`allow_delegation=True` 时自动获得两个协作工具  
  https://docs.crewai.com/en/concepts/collaboration

- CrewAI Create Custom Tools 文档：`BaseTool`、`args_schema`、`_run`、`_arun` 规范  
  https://docs.crewai.com/learn/create-custom-tools

- CrewAI Tools 概念文档：自定义工具与 `BaseTool` 的基础结构  
  https://docs.crewai.com/en/concepts/tools

- CrewAI GitHub PR #2068：`allowed_agents` 的层级委派与白名单思路  
  https://github.com/crewAIInc/crewAI/pull/2068

- CrewAI MCP Security 文档：可信 MCP、工具元数据注入风险、最小权限与访问控制  
  https://docs.crewai.com/mcp/security

- CrewAI A2A Agent Delegation 文档：远端 Agent 委派与跨边界协作  
  https://docs.crewai.com/en/learn/a2a-agent-delegation
