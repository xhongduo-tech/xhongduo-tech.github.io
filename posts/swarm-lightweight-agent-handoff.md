## 核心结论

Swarm 的核心设计不是“让很多 Agent 一起开会”，而是把多 Agent 协作收缩成一个更小的问题：当前 Agent 在需要时，显式把控制权交给下一个 Agent。这里的 `handoff` 可以理解为“转接函数”，也就是当前 Agent 通过一次工具调用，告诉客户端后续应该由谁继续说话。

这带来三个直接结论：

1. Swarm 是客户端编排。白话说，真正驱动流程的是你的应用代码，不是框架内部一个长期运行的调度中心。
2. Swarm 是无状态的。白话说，框架自己不替你记住会话历史；下一轮还能接着聊，前提是客户端把历史消息和上下文再次传回去。
3. Swarm 的 Agent 切换是按需发生的。没有切换时，它几乎就等价于普通的单 Agent Chat Completions 调用；只有模型决定调用 `handoff` 时，才会进入“换人处理”这一步。

一个最小玩具例子是客服路由。`triage` Agent 负责分流，用户说“我要退款”，它不需要自己解决，而是调用 `transfer_to_sales()`，返回 `sales_agent`。接下来同一段消息历史交给 `sales` 继续处理。这里没有额外的总控 Agent，也没有复杂的工作流引擎。

可以把 Swarm 的运行链路压缩成下表：

| 步骤 | 做什么 | 和普通单 Agent 的关系 |
|---|---|---|
| completion | 当前 Agent 先生成回复或工具调用 | 完全一致 |
| 工具 | 如果模型调用函数，就执行函数 | 完全一致 |
| handoff | 如果函数返回另一个 Agent，就切换发言者 | Swarm 新增的关键一步 |
| context | 更新消息和上下文，再决定是否继续循环 | 本质仍是客户端维护状态 |

从工程角度看，Swarm 的价值不在“功能最多”，而在“最少多加一层机制就能支持多 Agent 切换”。如果你的需求是单次路由、轻量分工、客户端本来就掌握状态，那么它的复杂度通常明显低于 AutoGen 和 CrewAI。

---

## 问题定义与边界

多 Agent 框架真正要解决的，不是“Agent 数量不够”，而是“谁来决定下一步由谁做、状态放哪里、切换怎么发生”。这三个问题如果都交给框架，系统通常会变重；如果都交给业务方，开发成本又会上升。Swarm 的选择是：框架只处理最小闭环，把“切换”做轻，把“状态”留给客户端。

因此，Swarm 主要解决的是这样一类问题：

- 一个对话里可能需要多个角色分工。
- 角色之间的切换路径比较清晰。
- 客户端愿意自己维护 `messages` 和 `context_variables`。
- 不需要一个常驻的中心调度器来长期统筹全局。

它的边界也同样清楚。Swarm 并不负责以下事情：

- 不替你保存会话历史。
- 不替你做复杂流程恢复。
- 不天然适合长周期、强状态依赖、多人并发协作的任务。
- 不内建一个“总指挥”来根据全局策略决定所有 Agent 的发言顺序。

这和 AutoGen、CrewAI 的取舍不同。AutoGen 常见模式里，会有 `GroupChatManager`、selector 或 team 机制决定谁下一步发言。这里的“selector”可以理解为“选人规则”，也就是一个中心化策略层。CrewAI 则把流程拆成 Flow 与 Crew。Flow 可以理解为“状态化工作流”，它负责事件推进、状态传递和分支控制；Crew 更像一组协作执行者。

对比后更容易看清 Swarm 的边界：

| 框架 | 调度层是谁 | 状态主要由谁承担 | 典型特征 |
|---|---|---|---|
| Swarm | 客户端 + 当前 Agent 的 handoff | 客户端 | 轻量、无状态、按需切换 |
| AutoGen | Manager / selector / team | 框架与运行时共同承担 | 中心化策略更强，可扩展到更复杂协作 |
| CrewAI | Flow + Crew | Flow 状态层 | 工作流更重，适合明确状态机和事件流 |

对新手来说，可以用一句话记住边界：Swarm 不是“帮你托管整个多 Agent 系统”，而是“给你一个最薄的 Agent 转接机制”。如果你在下一轮调用 `client.run()` 时没有把完整消息历史重新传入，那么系统不会“记得刚才已经转给 sales 了”，因为它从设计上就不记忆。

---

## 核心机制与推导

Swarm 的机制可以抽象成一个循环。这个循环之所以好理解，是因为它没有偏离普通 LLM 工具调用范式，只是在“工具返回值”里额外允许返回一个 Agent。

设当前 Agent 为 $A_t$，消息历史为 $M_t$，上下文变量为 $C_t$。一次运行后，系统会产生新的消息历史和可能的新 Agent：

$$
(A_t, M_t, C_t) \xrightarrow{\text{completion/tool}} (A_{t+1}, M_{t+1}, C_{t+1})
$$

其中：

- 如果没有发生 handoff，那么 $A_{t+1} = A_t$。
- 如果某个工具函数返回了另一个 Agent，那么 $A_{t+1} \neq A_t$。
- 历史消息 $M$ 和上下文 $C$ 由客户端继续持有，并在下一轮再次传入。

这个过程可以写成接近官方思路的伪代码：

```text
current_agent = agent
messages = input_messages
context = context_variables

loop:
  response = model(current_agent, messages, context)

  append response to messages

  if response has tool calls:
      for each tool call:
          result = execute(tool)
          append tool result to messages

          if result is Agent:
              current_agent = result
          if result updates context:
              merge into context

      continue loop

  return current_agent, messages, context
```

这里最关键的一点是：`handoff` 不是框架外的一次强制跳转，而是模型像调用普通工具一样，自己决定是否调用。也就是说，Swarm 的“切换”被压缩成了“特殊工具返回值”。

### 玩具例子

假设有两个 Agent：

- `triage`：分流客服请求。
- `sales`：处理退款与订单问题。

用户输入：“我想退款。”

推导过程是：

1. 当前 Agent 是 `triage`。
2. 模型发现这是退款类问题。
3. `triage` 调用 `transfer_to_sales()`。
4. 函数返回 `sales_agent`。
5. 客户端把当前 Agent 更新为 `sales_agent`。
6. 下一步回复由 `sales_agent` 基于同一份消息历史继续生成。

这比中心化调度少了一层判断。不是“调度器看到退款，所以把发言权给 sales”，而是“triage 自己说：这个问题该交给 sales”。

### 真实工程例子

在真实客服系统中，通常会有 `triage -> refund / tech / feedback` 三类路由：

- 用户问退款进度，切给 `refund_agent`
- 用户问接口报错，切给 `tech_agent`
- 用户提产品建议，切给 `feedback_agent`

如果用 Swarm，业务代码的责任主要是两件事：

1. 在客户端保存完整会话历史和用户上下文。
2. 给分流 Agent 定义可转接目标与工具函数。

如果用 AutoGen，同类场景往往还要定义 selector 或 manager 的选人逻辑；如果用 CrewAI，则经常需要设计 Flow 状态、节点、事件与 Crew 内部协作。三者不是“谁高级谁低级”的关系，而是把复杂度放在不同位置。Swarm 的核心推导是：把复杂度从“中央编排”挪到“局部 handoff + 外部状态管理”。

---

## 代码实现

下面先给一个不依赖外部库的 Python 玩具实现，用来把 `handoff` 机制讲透。它不是官方 SDK，但运行逻辑和 Swarm 的核心思想一致。

```python
from dataclasses import dataclass
from typing import Callable, List, Dict, Optional


@dataclass
class Agent:
    name: str
    decide: Callable[[List[Dict[str, str]], Dict[str, str]], Optional[str]]


def transfer_to_sales() -> str:
    return "sales"


def triage_decide(messages, context):
    last_user_text = messages[-1]["content"]
    if "退款" in last_user_text:
        return transfer_to_sales()
    return None


def sales_decide(messages, context):
    return None


AGENTS = {
    "triage": Agent(name="triage", decide=triage_decide),
    "sales": Agent(name="sales", decide=sales_decide),
}


def run(agent_name: str, messages: List[Dict[str, str]], context: Dict[str, str]):
    current = AGENTS[agent_name]

    while True:
        next_agent_name = current.decide(messages, context)
        if next_agent_name is None:
            return current.name
        current = AGENTS[next_agent_name]


messages = [{"role": "user", "content": "我想退款"}]
context = {"user_id": "u_123"}

final_agent = run("triage", messages, context)

assert final_agent == "sales"
print(final_agent)
```

这个例子里，`triage` 不负责给出退款政策，它只负责判断“该不该转接”。这就是轻量级 Agent 切换的核心：先解决“谁来处理”，再解决“怎么处理”。

再看接近 Swarm 用法的示意代码。重点不在 API 细节，而在结构关系：

```python
from swarm import Swarm, Agent

client = Swarm()

def transfer_to_sales():
    return sales_agent

sales_agent = Agent(
    name="Sales Agent",
    instructions="处理退款、订单和支付问题。"
)

triage_agent = Agent(
    name="Triage Agent",
    instructions="先判断问题类别；如果是退款或订单问题，转给 Sales Agent。",
    handoffs=[sales_agent],
)

messages = [
    {"role": "user", "content": "我要退款，订单号是 A12345"}
]

response = client.run(
    agent=triage_agent,
    messages=messages,
    context_variables={"user_id": "u_123", "locale": "zh-CN"},
    execute_tools=True,
)

print(response.agent.name)
assert response.agent.name in {"Triage Agent", "Sales Agent"}
```

这段代码要注意三点：

1. `handoff` 只在需要转接时发生，不需要转接时行为和单 Agent 对话接近。
2. `messages` 是客户端输入，不是框架内部自动持久化。
3. `response.agent` 代表当前回合结束后，后续最适合继续对话的 Agent。

如果把真实工程例子补全，可以形成这样的职责划分：

| Agent | 职责 | 何时 handoff |
|---|---|---|
| `triage` | 问题分类 | 判断用户意图后转给具体领域 Agent |
| `refund` | 退款规则、退款进度 | 需要核对订单时调用业务工具 |
| `tech` | 技术报错、接口问题 | 需要日志、工单系统时调用业务工具 |
| `feedback` | 收集建议、分类意见 | 需要沉淀标签时写入反馈库 |

所以，Swarm 并没有发明一种全新的执行范式，而是把“多 Agent 协作”重写成“一个普通 LLM 工具循环 + 一个特殊的工具返回值”。

---

## 工程权衡与常见坑

Swarm 的轻量来自少做事，但少做事意味着有些责任明确落到应用层。实际落地时，最常见的问题不是“不会 handoff”，而是“以为 handoff 之后框架会自动兜底”。

### 常见坑一：并行工具调用导致多个 handoff 冲突

如果底层模型支持并行工具调用，而当前 Agent 一次回复里同时触发多个函数，其中两个函数都可能返回不同 Agent，那么下一位到底是谁就会变得不可控。对新手来说，可以理解成“一个客服同时把电话转给销售和技术，线路冲突了”。

官方文档和相关实现经验通常建议在这类场景中关闭并行工具调用，即设置 `parallel_tool_calls=False`。因为对 Swarm 而言，handoff 最好是单一、确定、顺序发生的事件。

### 常见坑二：忘记回传完整消息历史

Swarm 无状态，意思不是“没有上下文”，而是“上下文由你带着走”。如果你这轮把用户历史、工具结果、context variables 丢了，下一轮即便仍然指定同一个 Agent，它看到的也是一段被截断的世界。

可以把这个约束写成一个简单条件：

$$
\text{连续对话成立} \Rightarrow \text{客户端必须保留并重传 } M_t, C_t
$$

否则就会出现“上一轮已经识别为退款，下一轮又重新问一遍订单号”的退化行为。

### 常见坑三：把 Swarm 当成工作流引擎

Swarm 适合“当前谁处理”这类路由问题，不适合直接承载复杂审批流、长事务恢复、多步骤状态机。如果一个任务有大量中间状态，例如“申请 -> 审批 -> 审核失败回退 -> 人工复核 -> 财务确认”，那就不该只靠 handoff 组织全部逻辑。

### 常见坑四：Agent 指令边界不清

如果 `triage` 的职责写成“既分类、又解释退款政策、又收集技术日志”，它就会退化成一个大而全的总控 Agent，handoff 的收益会下降。轻量切换成立的前提是角色边界清晰。

下面用表格汇总常见坑：

| 坑 | 后果 | 规避措施 |
|---|---|---|
| 并行工具触发多个 handoff | 下一位 Agent 不确定 | 关闭并行工具调用，保证 handoff 串行 |
| 没有重传消息与上下文 | 会话像“失忆”一样重置 | 每轮都回传完整 `messages/context_variables` |
| 把 Swarm 当流程引擎 | 复杂状态难维护，代码补丁化 | 只用它做路由，把长状态流程放到外部系统 |
| Agent 职责重叠 | 分流失效，提示词互相污染 | 明确每个 Agent 的输入边界和输出边界 |

工程上可以这样理解取舍：Swarm 省掉的是框架层复杂度，不是业务层复杂度。框架越轻，你越需要明确“状态在我这里”“切换由我兜底”“流程不能无限膨胀”。

---

## 替代方案与适用边界

如果问题只是“一个用户请求来了，先分类，再交给对应专家处理”，Swarm 往往是够用且更直接的。如果问题升级为“多个 Agent 要长期协作、存在中心决策、可能分布式运行、还要恢复中断状态”，那它通常不是首选。

下面把三者放在同一坐标系里看：

| 方案 | 调度方式 | 状态存储 | 工程复杂度 | 更适合什么 |
|---|---|---|---|---|
| Swarm | Agent 通过 handoff 自主转接 | 客户端外部维护 | 低到中 | 轻量路由、单次切换、客户端掌控状态 |
| AutoGen | Manager / selector / team 决定下一位 | 框架与运行时共同承担 | 中到高 | 需要中心策略、复杂协作、可扩展运行时 |
| CrewAI | Flow 事件驱动，Crew 内部协同 | Flow 持有流程状态 | 中到高 | 明确状态机、工作流较长、业务节点多 |

### Swarm 适用边界

适用：

- 客服分流
- 售前/售后路由
- 一个入口 Agent 转若干专家 Agent
- 你已经有自己的数据库、会话存储和业务状态层

不太适用：

- 长周期任务编排
- 多 Agent 多轮协商后再统一决策
- 复杂事件流和审批流
- 强依赖持久化状态恢复的系统

### 新手对比例子

还是同一个客服路由问题。

如果用 Swarm，你可能只写：

- 一个 `triage` Agent
- 三个目标 Agent：`refund`、`tech`、`feedback`
- 三个 handoff 函数
- 客户端每轮回传消息历史

如果用 AutoGen，你往往要额外思考：

- 谁是 manager
- selector 怎么决定下一位
- 广播给哪些 Agent
- 是否需要更复杂的 team 运行机制

如果用 CrewAI，你往往要思考：

- Flow 的状态字段怎么设计
- 节点之间如何转移
- Crew 在哪个节点执行
- 事件失败后怎么回退

所以，本质差异不是“Swarm 功能更少”这么简单，而是“Swarm 把问题定义为一次次局部转接，AutoGen 和 CrewAI 更愿意把问题定义为整体编排”。当你的问题本来就只是轻量路由时，Swarm 的定义往往更贴近任务本身。

---

## 参考资料

1. OpenAI Swarm GitHub 文档：https://github.com/openai/swarm  
   关键贡献：说明 Swarm 的目标、`client.run()` 循环、handoff 示例，以及“客户端运行、无状态”的基本设计。

2. AutoGen Swarm / GroupChat 文档：https://microsoft.github.io/autogen/stable/user-guide/agentchat-user-guide/swarm.html  
   关键贡献：解释中心化管理、handoff 在 AutoGen 中的组织方式，以及并行工具调用可能带来的切换不确定性。

3. AutoGen 项目文档主页：https://microsoft.github.io/autogen/  
   关键贡献：补充 GroupChat、selector、team 等多 Agent 编排思路，便于和 Swarm 的轻量机制对照。

4. CrewAI 架构与 Flow 文档：https://docs.crewai.com/  
   关键贡献：说明 Flow 与 Crew 的分层职责，帮助理解其“状态化工作流 + Agent 协作”的工程定位。

5. MCP Agent Swarm 工作流说明：https://docs.mcp-agent.com/workflows/swarm  
   关键贡献：从客服路由等具体场景说明 Swarm 式切换在工程实现上的简化价值。
