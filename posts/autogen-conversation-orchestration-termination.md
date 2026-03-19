## 核心结论

AutoGen 的多 Agent 会话编排，本质上是在解决两个问题：下一句该由谁说，以及什么时候必须停。`speaker_selection_method` 负责前者，`max_round`、`is_termination_msg` 和更细粒度的 termination 条件负责后者。两类机制必须一起设计，否则系统很容易进入“能继续说，但已经没有新信息”的空转状态。

可以把 `GroupChat` 理解成“会议记录本”，也就是保存参与者和历史消息的容器；把 `GroupChatManager` 理解成“主持人”，也就是根据当前上下文决定下一位发言者。对初学者最重要的结论是：发言顺序不是界面参数，而是影响收敛速度、调用成本和是否死循环的核心控制面。

一个最小可解释例子是“三个同学轮流汇报进度”。如果设置 `speaker_selection_method="round_robin"`，顺序就是固定轮换；如果 `max_round=5`，整场会议最多进行 5 轮；如果 `is_termination_msg` 检测到消息里包含 `TERMINATE`，则提前结束。这样即使某个同学一直补充细节，也不会让会议无限延长。

| 策略 | 选择方式 | 收敛特性 | 适合场景 |
| --- | --- | --- | --- |
| `round_robin` | 固定轮换 | 最稳定、最可预测 | 流程化协作 |
| `random` | 随机抽取 | 多样性高，收敛波动大 | 探索式生成 |
| `manual` | 人工指定 | 最可控，吞吐最低 | 审核型流程 |
| `auto` | 模型依据上下文选择 | 灵活，但成本最高、最容易空转 | 复杂动态协作 |

---

## 问题定义与边界

多 Agent 系统不是“把多个模型放在一起”这么简单。它的真正难点在于，多个 Agent 会互相引用上一条消息，从而形成反馈回路。反馈回路的白话解释是：上一轮输出会成为下一轮输入，所以一旦规则不严，对话就会自己把自己续下去。

如果没有调度规则，多个 Agent 很可能出现两类问题。第一类是循环发言，例如 Coder 一直改代码，Critic 一直提小问题，二者都没有明确的停机信号。第二类是资源失控，例如 `auto` 模式为了“选择最合适的下一位发言者”，反复调用模型做判断，本身也会消耗 token 和时间。

课堂讨论是一个贴近现实的类比。没有主持人时，谁都可以继续说；没有时间限制时，任何细节都能无限展开。多 Agent 协作也是一样，`speaker_selection_method` 相当于主持规则，`max_round`、关键词终止、消息数终止则相当于“本节课到点必须下课”。

从边界上看，本文讨论的是经典 AutoGen `GroupChat` / `GroupChatManager` 风格的会话编排，以及它与 termination 组合的设计思路，不展开更高层的工作流编排框架。核心约束可以写成：

$$
\text{终止} = \text{轮次上限} \;\lor\; \text{终止关键词触发} \;\lor\; \text{消息数/Token/超时边界触发}
$$

这里的 $\lor$ 表示“任一条件满足即可停止”。工程上通常不是“必须全部满足才停”，而是“任何一个安全阈值先到就停”，因为多 Agent 的首要目标是有限资源内收敛，而不是把每个 Agent 都说到没话说。

---

## 核心机制与推导

经典流程可以写成一条箭头链：

`GroupChat 存消息 -> GroupChatManager 读上下文 -> 按 speaker_selection_method 选 speaker -> Agent 回复 -> 终止判定 -> 未终止则继续`

其中，`GroupChat` 主要保存三个东西：成员列表、消息历史、轮次状态。`GroupChatManager` 不直接“解决业务问题”，而是负责根据规则调度下一位说话者。调度完成后，被选中的 Agent 基于历史消息生成回复，回复再被追加回 `GroupChat`。

玩具例子可以这样看。假设有三个角色：`Planner` 负责拆任务，`Coder` 负责写方案，`Reviewer` 负责挑错。若采用 `round_robin`，顺序固定为 Planner -> Coder -> Reviewer -> Planner。这样做的优点是稳定，缺点是有时 Reviewer 明明暂时不需要发言，也会被强制轮到。若采用 `auto`，Manager 会根据上下文决定“这一步最该谁说”，比如发现 Planner 已经给出完整拆解，就直接让 Coder 接着做。但这种灵活性依赖模型判断，因此更贵，也更可能误判。

为什么终止判定要在“每条回复后”执行？因为多 Agent 的状态变化是离散的，每一条消息都可能已经满足完成条件。例如 Reviewer 可能回复：“测试通过，TERMINATE。”如果系统只在每 3 轮后检查，就会白白多跑两轮。更一般地说，若第 $t$ 条消息后满足终止条件，则继续执行到第 $t+1$ 条只会增加成本，不会提升正确性。

真实工程例子是“三 Agent 代码迭代回路”。设有 `Coder` 写代码，`Critic` 做代码审查，`UserProxy` 执行代码并反馈运行结果。一个稳定设计通常是：

1. `Coder` 根据需求输出实现。
2. `Critic` 只指出缺陷，不直接改代码。
3. `UserProxy` 负责运行、收集结果，并在任务完成时输出 `TERMINATE`。
4. `GroupChatManager` 每轮后检查是否达到 `max_round`，或是否有终止消息。

这个设计的关键不是“谁更聪明”，而是角色职责和停机边界清楚。职责清楚，发言才有方向；停机边界清楚，迭代才不会失控。

---

## 代码实现

下面先用一个不依赖 AutoGen 的玩具实现，演示“选 speaker + 检查终止”的核心逻辑。它能直接运行，目的是让机制先可见，再映射到框架配置。

```python
from dataclasses import dataclass

@dataclass
class Message:
    role: str
    content: str

def is_termination_msg(msg: Message) -> bool:
    return "TERMINATE" in msg.content

def next_speaker(agents, last_index, method="round_robin"):
    if method != "round_robin":
        raise ValueError("toy example only supports round_robin")
    return (last_index + 1) % len(agents)

def run_chat(agents, scripted_outputs, max_round=5):
    messages = []
    speaker_idx = -1

    for round_no in range(max_round):
        speaker_idx = next_speaker(agents, speaker_idx, method="round_robin")
        role = agents[speaker_idx]
        content = scripted_outputs[role].pop(0)
        msg = Message(role=role, content=content)
        messages.append(msg)

        if is_termination_msg(msg):
            return messages, "terminated"

    return messages, "max_round_reached"

agents = ["Planner", "Coder", "Reviewer"]
scripted = {
    "Planner": ["拆解任务"],
    "Coder": ["实现完成"],
    "Reviewer": ["检查通过，TERMINATE"],
}

msgs, status = run_chat(agents, scripted, max_round=5)
assert status == "terminated"
assert msgs[-1].role == "Reviewer"
assert "TERMINATE" in msgs[-1].content
```

映射到 AutoGen 的经典配置时，思想基本一致。`GroupChat(max_round=12)` 负责轮次上限；`speaker_selection_method="round_robin"` 或 `"auto"` 负责选人；`is_termination_msg=lambda msg: "TERMINATE" in msg["content"]` 负责识别提前结束信号。

```python
# 伪代码，展示关键配置思路
groupchat = GroupChat(
    agents=[coder, critic, user_proxy],
    messages=[],
    max_round=12,
    speaker_selection_method="round_robin",
)

manager = GroupChatManager(
    groupchat=groupchat,
    llm_config=llm_config,
    is_termination_msg=lambda msg: "TERMINATE" in msg.get("content", "")
)

user_proxy.initiate_chat(
    manager,
    message="实现一个带单元测试的排序函数"
)
```

如果使用新版 AgentChat 风格的 termination 组合，常见做法是把多个终止器做“或”组合，例如“最多 20 条消息”或“出现 `TERMINATE` 就停止”。这与上面的数学表达式是一致的：任何一个边界先触发，都应该立刻退出。

---

## 工程权衡与常见坑

最常见的坑不是模型不够强，而是停机条件不完整。很多初学者看到 `auto` 很灵活，就把 speaker 选择交给模型，但忘了再加硬边界。结果是系统虽然表面上“还在工作”，实际却在重复确认、重复批评、重复改写。

下面这个对比在工程上很典型：

| 配置方式 | 结果 |
| --- | --- |
| 只有 `auto`，无 `max_round`，无关键词终止 | 最容易无限循环，成本不可控 |
| `auto` + `MaxMessageTermination` | 能停，但可能在任务未完成时被动截断 |
| `round_robin` + `max_round` + `TERMINATE` | 收敛稳定，便于排查 |
| `manual` + 终止器 | 最安全，但需要人工介入 |

一个常见失败例子是：开发者只设置 `auto + MaxMessageTermination(20)`。看上去已经有上限，但如果任务本身需要 24 条消息才能完成，系统会在第 20 条被迫截断，表现为“并没有死循环，但也没有做完”。这说明“能停”不等于“合理地停”。

另一个常见问题是关键词设计太脆弱。若把终止条件写成“只要消息中出现 `done` 就停止”，那么 Agent 在说“not done yet”时也可能误触发。关键词终止的白话解释是：用显式口令作为停机信号，所以口令必须足够专用，例如 `TERMINATE`、`<TASK_DONE>` 这种不容易在自然语言中误出现的标记。

真实工程里，更稳定的做法通常是双层甚至三层终止：

1. `max_round` 控制最坏情况。
2. `is_termination_msg` 或 `TextMentionTermination("TERMINATE")` 控制正常完成。
3. `MaxMessageTermination`、`TokenTermination` 或 `TimeoutTermination` 控制资源上限。

还有一个坑是把 `random` 当成“更智能”。随机选择的价值是增加探索，不是提高确定性。如果任务是固定流程，比如“写代码 -> 审查 -> 执行 -> 汇总”，随机打乱顺序通常只会让上下文更混乱。相反，`round_robin` 虽然机械，但对新手最友好，因为你几乎可以直接预测下一步系统会做什么。

---

## 替代方案与适用边界

`speaker_selection_method` 没有统一最优解，只有与任务结构是否匹配。

| 方案 | 优点 | 缺点 | 推荐场景 |
| --- | --- | --- | --- |
| `round_robin` | 可预测、易调试、易复现 | 可能让不该发言的 Agent 也发言 | 结构化流水线 |
| `random` | 引入探索，减少固定套路 | 收敛慢，结果不稳定 | 头脑风暴、创意生成 |
| `manual` | 人工把关，风险最低 | 人力成本高，吞吐低 | 高风险审核、演示 |
| `auto` | 上下文自适应，灵活性强 | 成本高，需更强终止设计 | 动态任务、多分支协作 |

如果任务偏探索，例如多种方案 brainstorming，`random` 有时比 `round_robin` 更有价值，因为它能打破固定路径依赖。但它不适合强约束流程。若场景要求人工确认，例如医疗建议复核、上线审批、合同审阅，`manual` 更合适，因为你需要把“谁继续说”这个控制权交给人，而不是交给模型。

终止机制也可以按资源约束扩展：

| 终止方式 | 控制对象 | 适用边界 |
| --- | --- | --- |
| `MaxMessageTermination` | 消息条数 | 控制交互步数 |
| `TextMentionTermination` | 指定文本口令 | 显式完成信号 |
| `TokenTermination` | token 消耗 | 大模型成本控制 |
| `TimeoutTermination` | 时间上限 | 在线服务 SLA |
| `HandoffTermination` | 交接事件 | 多阶段工作流切换 |

高成本任务通常不应只靠 `TERMINATE`。原因很简单：如果某个 Agent 忘了输出终止口令，系统就会继续消耗。此时 `TimeoutTermination` 和 `TokenTermination` 是必要的保险丝。它们的白话解释分别是“时间到了必须停”和“预算花到上限必须停”。

因此可以给出一个实用选择原则：流程固定时优先 `round_robin`；探索优先时考虑 `random`；需要人工责任链时用 `manual`；只有在上下文变化大、固定顺序明显不够用时，再用 `auto`，并且一定同时配置硬终止条件。

---

## 参考资料

1. AutoGen 官方 Termination 教程：介绍 `MaxMessageTermination`、`TextMentionTermination`、`TokenTermination`、`TimeoutTermination`、`HandoffTermination` 的组合方式。https://microsoft.github.io/autogen/stable/user-guide/agentchat-user-guide/tutorial/termination.html  
2. AutoGen 官方 GroupChat 相关示例：展示经典 `GroupChat`、`GroupChatManager`、`is_termination_msg`、`max_round` 的配置思路。https://microsoft.github.io/autogen/0.2/docs/topics/groupchat/resuming_groupchat/  
3. ByteZoneX《Autogen GroupChat 实战》：用 Coder/Critic/UserProxy 多 Agent 回路解释 `speaker_selection_method` 与 `TERMINATE` 的实战组织方式。https://www.bytezonex.com/archives/GIU6Z2oB.html
