## 核心结论

多 Agent 工作流的 DSL，核心不是“换一种语法写流程”，而是把协作系统抽象成一张**状态图**。状态图的白话解释是：把每个参与工作的角色画成节点，把“谁把什么信息交给谁”画成边，再给边加上“满足什么条件才走”这一层规则。这样，节点、消息、条件分支、循环结构就能放进同一个模型里。

对初级工程师来说，最重要的结论有三点：

| 视角 | 优点 | 典型用途 |
| --- | --- | --- |
| 图结构视角 | 节点、边、条件、循环统一建模 | 设计和推演协作流程 |
| DSL 表达视角 | 把图写成可执行配置或代码 | 落地成实际系统 |
| 工程实现视角 | 调度、重试、观测、测试更容易接入 | 生产环境运行 |

第一，**把多 Agent 工作流当作 StateGraph 最稳妥**。Agent 是节点，消息与上下文流是边，`If`、`Else`、`Foreach`、`BreakLoop` 是图上的路由规则。这种抽象不会绑死在某个具体框架上。

第二，**DSL 有三种常见形态，各有侧重**。YAML 声明式适合让流程清晰可读；Python 装饰器或链式 DSL 适合把流程嵌进代码里，便于调试与测试；图可视化适合跨角色协作评审，让产品、算法、工程都能看懂流程。

第三，**不要把 DSL 理解成“替代代码”**。更准确的理解是：DSL 负责描述稳定结构，代码负责承载复杂逻辑。简单路由、固定节点依赖、有限循环，可以放进声明式 DSL；复杂判断、自定义调度、动态 Agent 选择，通常要退回到 Python 这类可编程 DSL。

玩具例子可以先这样理解：有三个角色，“采集”“校验”“执行”。白板上画三个方框，再画箭头：采集产出数据给校验；如果值不够就回到采集；如果满足条件就交给执行。每条箭头上写“满足条件才走”或“继续循环”，这已经是一个最小可用的工作流 DSL 模型。

---

## 问题定义与边界

这里的问题不是“如何写很多 Agent”，而是“如何让多个 Agent 的协作关系**可描述、可执行、可观察**”。

**Agent** 的白话解释是：一个能接收输入、处理任务、再输出结果的执行单元。它可以是一个 LLM 调用，也可以是普通函数、检索器、规则引擎，甚至是人工审批节点。

**DSL** 的白话解释是：为了某类问题专门设计的一套小语言。它不追求像通用编程语言那样什么都能做，而是追求把某个领域的结构表达得更直接。

多 Agent 工作流 DSL 的目标通常有四个：

1. 明确节点定义：每个 Agent 做什么、接收什么、输出什么。
2. 明确边定义：消息、共享状态、上下文如何传递。
3. 明确控制流：条件分支、循环、失败回退、终止条件如何表示。
4. 明确执行语义：调度顺序、并发限制、重试策略、可观测性如何落地。

边界也要说清楚。DSL 不是无限表达力的语言。一个工程可用的工作流 DSL，至少要明确下面几件事：

| 控制结构 | 作用 | 常见限制 |
| --- | --- | --- |
| `actions` | 声明节点及执行顺序 | 难表达复杂依赖计算 |
| `If` | 条件分支 | 条件表达式往往受限 |
| `Foreach` | 遍历集合或重复执行 | 动态集合来源要提前定义 |
| `BreakLoop` | 提前终止循环 | 终止条件与状态更新要一致 |

新手版理解可以非常直接：假设 Agent A 先运行，把结果发给 B；B 再决定要不要让 C 运行。这里最重要的不是“B 里写了多少代码”，而是要把“节点是角色，边是消息”清楚分开。这样你才知道问题到底出在节点逻辑，还是出在路由设计。

工程里最常见的边界冲突有两个。

第一，**静态声明和动态逻辑的边界**。如果你希望流程文件能让非程序员读懂，就会偏向 YAML；但 YAML 对复杂表达式支持弱，一旦条件依赖运行时上下文的复杂组合，就会开始变形。

第二，**共享状态模型的边界**。共享状态的白话解释是：所有节点都能读取或更新的一份公共上下文。如果这份状态没有明确定义字段、读写规则和生命周期，流程很快会变成隐式耦合，最后谁都不敢改。

所以，DSL 设计的真正难点，不是语法长什么样，而是：哪些信息必须在图上显式声明，哪些逻辑允许藏在代码里。

---

## 核心机制与推导

把多 Agent 工作流建模为：

$$
G=(N,E,\Sigma)
$$

其中：

- $N$ 是节点集合，每个节点对应一个 Agent。
- $E$ 是边集合，每条边定义从哪个节点流向哪个节点。
- $\Sigma$ 是系统状态集合，表示共享上下文、消息缓存、循环计数、执行结果等。

如果把共享状态记作 $S \in \Sigma$，那么每个节点 $n_i$ 本质上执行一个状态变换函数：

$$
f_i: S \rightarrow S
$$

白话解释是：节点拿到当前上下文，处理后写回一个新上下文。

每条边可以表示为：

$$
e_j=(n_{src}, n_{dst}, cond, msg)
$$

其中：

- $n_{src}$ 是起点节点。
- $n_{dst}$ 是目标节点。
- $cond: S \rightarrow \{\text{true}, \text{false}\}$ 是条件函数。
- $msg$ 表示要传递的消息或上下文字段。

当节点 $n_{src}$ 执行完得到新状态 $S'$ 后，如果 $cond(S')=\text{true}$，调度器就沿这条边走向 $n_{dst}$。否则走另一条备用边，或直接结束。

这个模型的价值在于，它把几种看起来不同的控制结构统一了：

| 图上的概念 | DSL 中常见写法 | 本质 |
| --- | --- | --- |
| 条件边 | `If` | 基于状态选择后继节点 |
| 默认边 | `Else` | 条件不满足时的兜底路径 |
| 回环边 | `Foreach` / loop | 节点再次进入前序节点 |
| 终止边 | `BreakLoop` / `End` | 满足条件后退出图 |

玩具例子可以写成这样：

- 节点 A：采集数据，输出一个值 $v$。
- 节点 B：校验 $v$ 是否足够大。
- 节点 C：执行正式动作。

初始时有 $loop\_cnt=0$。A 每次产出：

$$
v = 2 + loop\_cnt
$$

B 的条件函数为：

$$
cond(S)= [v \ge 5]
$$

如果条件不满足，B 走回环边回到 A，并把 $loop\_cnt$ 加一；如果满足，就流向 C。这个过程说明一件事：**循环不是特殊结构，本质上只是指回前序节点的边**。

可以把图简化成下面这种手写表示：

```text
[A:采集] --v--> [B:校验]
   ^              |
   | v<5          | v>=5
   +--------------+
                  \
                   -> [C:执行]
```

真实工程例子更接近下面这种模式：

- `planner` 节点：根据用户目标生成任务计划。
- `executor` 节点：执行当前计划步骤。
- `reviewer` 节点：检查结果是否满足目标。
- 条件边：如果 `plan_complete=true`，结束；否则回到 `planner` 或继续到 `executor`。

这类系统经常会把计划、执行结果、错误信息写入共享状态，再由条件边决定下一跳。也就是说，Agent 之间真正传递的并不只是自然语言消息，而是一份结构化状态。

因此，DSL 设计要回答的核心问题只有一个：**你打算用什么语法，把这张状态图完整、明确、可执行地表达出来。**

---

## 代码实现

先看最小的 YAML 声明式例子。它适合描述“先做什么，再判断什么，最后做什么”这种结构稳定的流程。

```yaml
workflow:
  name: collect-validate-execute

state:
  loop_cnt: 0
  value: 0
  max_loop: 3

actions:
  - id: collect
    agent: collector
    next: validate

  - id: validate
    agent: validator
    if:
      condition: "state.value >= 5"
      then: execute
      else: collect

  - id: execute
    agent: executor
    next: end
```

这个片段和图的映射关系非常直接：

- `actions` 里的每个条目就是一个节点。
- `next` 表示默认边。
- `if.then` 和 `if.else` 表示条件边。
- `state` 是共享状态的初始值。

新手版理解可以概括成一句话：这个文件只在写一件事，先做 A，再判断是否进入 B 或回到 A，最后做 C。

但 YAML 只能描述结构，不能直接执行。下面给一个可运行的 Python 玩具实现，用最少代码模拟“节点 + 条件边 + 循环”的调度逻辑：

```python
from dataclasses import dataclass, field

@dataclass
class State:
    loop_cnt: int = 0
    value: int = 0
    history: list[str] = field(default_factory=list)

def collect(state: State) -> State:
    state.value = 2 + state.loop_cnt
    state.history.append(f"collect(value={state.value})")
    return state

def validate(state: State) -> str:
    state.history.append(f"validate(value={state.value})")
    if state.value >= 4 or state.loop_cnt >= 2:
        return "execute"
    state.loop_cnt += 1
    return "collect"

def execute(state: State) -> State:
    state.history.append("execute()")
    return state

def run_workflow() -> State:
    state = State()
    current = "collect"

    while current != "end":
        if current == "collect":
            state = collect(state)
            current = "validate"
        elif current == "validate":
            current = validate(state)
        elif current == "execute":
            state = execute(state)
            current = "end"
        else:
            raise ValueError(f"unknown node: {current}")

    return state

result = run_workflow()
assert result.value >= 4
assert result.history[-1] == "execute()"
assert result.history.count("execute()") == 1
print(result.history)
```

这个例子故意保持简单，但已经包含了 DSL 的几个核心语义：

- `current` 是当前节点指针。
- 每个函数是一个节点执行器。
- `validate` 返回下一个节点名，相当于边路由。
- `State` 是共享状态。
- `assert` 用来验证流程终态是否正确。

如果流程复杂度上升，很多团队会改用 Python 装饰器或链式 DSL。链式 DSL 的白话解释是：用连续的方法调用，把图关系一行一行串出来。示意写法如下：

```python
workflow = (
    Workflow("ops-pipeline")
    .agent("planner", run=plan_task)
    .agent("executor", run=execute_task)
    .agent("reviewer", run=review_result)
    .edge("planner", "executor")
    .if_("reviewer", cond=lambda s: s["done"], then="end", else_="planner")
)
```

这种写法的优点不在“更短”，而在“更可编程”：

- 条件函数可以直接写 Python。
- 单个 Agent 可以被 mock，白话解释是：在测试里用假的实现替换真实实现。
- 可以插断点、打日志、写单元测试。
- 可以把复杂路由逻辑拆成普通函数。

真实工程例子通常会再多一层：先用 YAML 声明大致拓扑，再在 Python 中注册节点实现。例如：

- YAML 定义：`planner -> executor -> reviewer`
- Python 注册：`planner` 用 LLM 生成计划，`executor` 调工具执行，`reviewer` 校验结果
- 运行时：调度器根据状态决定走哪条边

这是一种很实用的折中。结构可读，逻辑可编程。

---

## 工程权衡与常见坑

三种 DSL 形态没有绝对优劣，问题只在于你的流程复杂度和团队协作方式。

先看常见坑：

| 常见坑 | 具体表现 | 规避策略 |
| --- | --- | --- |
| 动态条件过复杂 | YAML 中塞大量表达式，可读性迅速下降 | 把条件下沉到 Python 函数 |
| 循环次数失控 | 回环条件不完整，流程可能卡死 | 在状态里显式维护计数与上限 |
| 共享状态污染 | 多节点随意改字段，行为难预测 | 定义状态 schema 和写入边界 |
| 可视化拥挤 | 节点一多，图变成线团 | 分层、子图、按阶段折叠 |
| 图与代码不同步 | 配置改了，执行器没同步更新 | 用单一事实来源生成另一侧产物 |

第一类权衡是**可读性和表达力的冲突**。纯 YAML 好读，但一旦你要写“如果过去三轮评分均值低于阈值且当前错误类型属于可重试类，再切换到备用 Agent”这种规则，声明式配置很快就会变成另一种难读代码。这个时候应该降级为代码控制，而不是继续硬塞进 DSL。

第二类权衡是**结构稳定和执行灵活的冲突**。如果你的工作流经常新增节点、改路由、由非工程角色参与评审，DSL 很有价值；如果你的流程本质上是运行时动态搜索，比如根据检索结果临时生成 20 个 Agent 子任务，纯声明式图就不一定合适。

第三类权衡是**可视化友好和系统规模的冲突**。图在 3 到 10 个节点时很清晰，到了 50 个节点往往会拥挤。解决办法通常不是“画更大的图”，而是引入层级：

- 顶层图：只显示阶段，如“计划”“执行”“校验”。
- 子图：展开某个阶段内部的具体 Agent。
- 代码同步：图从 DSL 或代码自动生成，避免手工维护两份定义。

新手常见误区还有一个：把“Agent 节点”设计得太粗。比如把检索、推理、校验、执行全塞进一个超级 Agent。这样图上虽然节点少了，但可观测性和可测试性都变差。一般来说，一个节点至少应该满足“职责单一、输入输出明确”。

什么时候应该明确退回纯代码控制？经验上有三个信号：

1. 条件分支已经依赖复杂计算或外部副作用。
2. 节点集合在运行时才确定，静态图无法提前列出。
3. 你需要细粒度控制并发、超时、熔断、补偿逻辑。

出现这些信号时，不要强行坚持“所有东西都必须是 DSL”。

---

## 替代方案与适用边界

如果不用工作流 DSL，还可以有三类方案：

| 方式 | 适用场景 | 局限 |
| --- | --- | --- |
| YAML/声明式 DSL | 流程稳定、需要评审、需要可视化 | 复杂逻辑表达弱 |
| Python 代码 DSL | 需要调试、测试、动态逻辑 | 非工程人员不易直接阅读 |
| 纯图谱工具/可视化编排 | 跨职能协作、演示、低门槛配置 | 与代码同步和版本控制较难 |

第一类替代方案是**纯代码**。例如直接用 Python 协程、线程或事件循环来调度 Agent。它的优点是灵活，几乎没有表达力上限；缺点是透明度差，流程结构藏在控制流里，不适合多人评审，也不利于快速回答“当前为什么走到这里”。

第二类替代方案是**纯图形化工具**。优点是非程序员也能参与设计，节点和边一眼可见；缺点是版本管理、代码复用、测试自动化通常弱于代码库。特别是在 Agent 数量很大、状态频繁变化时，图形界面往往会变成“展示层”，真正逻辑还是要回到代码里。

第三类是**混合方案**，也是最常见的工程落点：

- 用 YAML 或 JSON 定义稳定拓扑。
- 用 Python 注册节点实现与复杂条件。
- 用图可视化作为观测和评审界面。
- 用统一状态模型串起日志、重试、回放和测试。

新手版对比可以这样理解：

- 写一整段 Python 控制流，最灵活，但别人读起来成本高。
- 用 DSL 图形化配置，结构更清楚，适合团队一起 review。
- 纯图谱工具最直观，但复杂项目里常常不够精确。

所以，DSL 的适用边界不是“只要是多 Agent 就该上 DSL”，而是：**当你需要把协作结构从执行细节里抽离出来，并让它可审查、可复用、可观测时，DSL 才真正有价值。**

---

## 参考资料

| 资源名称 | 所属主题 | 作用简述 | 访问用途 |
| --- | --- | --- | --- |
| LangGraph / 状态图编排相关文章 | 图/FSM 工作流 | 用状态图理解多 Agent 的计划、执行、校验闭环 | 用来参考图式建模与条件边设计 |
| Microsoft Agent Framework Declarative Workflows | YAML 声明式工作流 | 提供 `actions`、`If`、`Foreach`、`BreakLoop` 等控制结构 | 用来参考声明式 DSL 语法与边界 |
| Kubiya Workflow DSL Overview | Python 链式 DSL | 展示链式或图式 API 如何把工作流嵌入代码 | 用来参考可测试、可调试 DSL 设计 |

- LangGraph 相关思路：把多 Agent 系统建模为图，适合表达计划-执行-校验这类闭环。
- Microsoft Agent Framework Declarative Workflows：强调 YAML 声明式控制流，适合让流程结构更易读。
- Kubiya Workflow DSL：强调链式 API 的可编程性，适合把 DSL 与工程代码、测试体系结合。

参考链接：
- https://healthark.ai/orchestrating-multi-agent-systems-with-lang-graph-mcp/
- https://learn.microsoft.com/en-us/agent-framework/workflows/declarative
- https://docs.kubiya.ai/sdk/workflow-dsl-overview
