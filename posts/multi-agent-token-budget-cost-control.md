## 核心结论

多 Agent 系统的 Token 成本，通常不是“模型单次调用太贵”，而是“多个 Agent 同时带着长上下文互相转发信息”，导致同一批内容被重复计费。真正有效的控制方法，不是只调低模型参数，而是在三个层面同时设限：全局预算池、每个 Agent 的硬上限、共享信息的摘要压缩。

可以先用一句白话定义术语。Token 是模型处理文本时计费的最小单位，大致可以理解为“被模型读入和输出的一小段文本成本”。上下文窗口是模型当前这次调用能看到的历史内容范围。多 Agent 指多个分工不同的模型节点共同完成一项任务，比如规划、检索、执行、审核分别交给不同 Agent。

工程上最稳的策略通常是：

| 层级 | 作用 | 典型做法 |
|---|---|---|
| 全局预算 `B_global` | 防止系统总成本失控 | 给单个任务或单次会话设置总 Token 上限 |
| 单 Agent 预算 `B_i` | 防止某个 Agent 抢占资源 | 每个 Agent 默认只拿固定额度，如 800 token |
| 摘要层 `S_summary` | 减少重复传递 | 旧对话、共享事实先压缩再传播 |

对新手可以直接这样理解：给每个 Agent 先发 800 token，再准备一个“公共钱包”给关键节点临时加钱；旧对话不要原样转发，而是先做摘要，把重复付费的部分砍掉。这样通常比“所有 Agent 都带完整历史”便宜得多，也更稳定。

---

## 问题定义与边界

问题的本质是上下文乘法叠加。单 Agent 系统里，一段历史只被一个模型读取一次；多 Agent 系统里，同一段历史可能被规划 Agent、工具 Agent、审核 Agent 分别读取一遍，成本立刻变成多份。

一个玩具例子很直观。假设有 3 个 Agent 处理同一张工单，每个 Agent 都带着 600 token 的历史和 200 token 的当前任务说明。若不做预算管理，总输入大约是：

$$
3 \times (600 + 200) = 2400
$$

如果这 600 token 历史里有 400 token 是所有 Agent 都需要的共享事实，那么实际上你为这 400 token 付了 3 次钱。这就是冗余传递。

因此，多 Agent Token 预算问题要回答三个边界：

| 边界问题 | 要不要管 | 原因 |
|---|---|---|
| 单步、一次性的小任务 | 可以弱化管理 | 任务短，重复传播有限 |
| 多轮对话、长链路任务 | 必须管理 | 历史会快速累积 |
| 核心事实、约束条件 | 必须保留 | 丢了会导致错误推理 |
| 低频背景、旧中间结果 | 优先摘要或外存 | 重复传递价值低 |
| VIP/关键 Agent | 可动态加额 | 质量损失代价更高 |
| 普通 Agent | 维持硬上限 | 避免预算抢占 |

可以用一个近似式描述总消耗边界：

$$
T_{total} \approx \sum_i \min(B_i, Context_i) + \max(0, T_{shared} - S_{summary})
$$

这里：

- $B_i$ 是第 $i$ 个 Agent 的上限，也就是它最多能花多少。
- $Context_i$ 是它原本想带入的上下文长度。
- $T_{shared}$ 是多个 Agent 都会重复读取的共享信息。
- $S_{summary}$ 是把共享信息压缩后的长度。

这个式子表达的含义很直接：每个 Agent 先受自己的预算约束，共享部分再尽量通过摘要缩小。边界外的内容，比如低价值的历史闲聊，不应该继续进入上下文。

---

## 核心机制与推导

核心机制是分层预算和动态调度。

第一层是全局预算池 `B_global`。它控制一个任务整体最多能花多少 Token，相当于总闸门。第二层是单 Agent 预算 `B_i`，它控制每个节点默认只能用多少。第三层是摘要层，用来缩减所有节点共享的重复信息。

如果没有第三层，前两层只能“硬砍”；硬砍虽然能省钱，但容易损失关键信息。摘要层的作用，是先做信息压缩，再做预算分配。

目标可以写成：

$$
S_{summary} \ll T_{shared}
$$

意思是摘要后的共享信息应远小于原始共享信息。比如共享内容原本 400 token，摘要后变成 160 token，那么压缩率就是：

$$
\frac{160}{400} = 40\%
$$

这意味着重复传播部分减少了 60%。

再引入动态优先级函数 `Priority_i(t)`。优先级就是“当前这轮谁更值得花钱”的评分。它通常由任务重要性、截止时间、失败代价、用户等级等因素组成。一个简化写法是：

$$
Extra_i(t) = \frac{Priority_i(t)}{\sum_j Priority_j(t)} \times Pool_{remain}
$$

含义是：公共预算池剩下多少，就按优先级比例发给需要额外 Token 的 Agent。

继续用最小数值例子。假设：

- 3 个 Agent，每个默认 800 token
- 共享信息原本 400 token
- 摘要后剩 160 token

那么总消耗从：

$$
3 \times 800 + 400 = 2800
$$

变为：

$$
3 \times 800 + 160 = 2560
$$

直接少了 240 token，约节省 8.6%。如果这 240 token 不浪费掉，而是回收到全局池，再临时分给最关键的 Agent，就能同时提升质量和稳定性。

真实工程里，节省幅度往往更大，因为实际系统的重复内容远多于这个玩具例子。客服、代码代理、工作流编排这类场景，经常会把同一工单背景、工具结果、状态说明在多个节点之间重复传播。只要摘要器能稳定保留“任务目标、关键事实、约束、未决问题”四类信息，压缩收益通常非常明显。

从流程上看，可以理解为：

1. 原始消息先进入调度器。
2. 调度器检查当前任务的全局预算剩余。
3. 每个 Agent 先拿默认额度 `B_i`。
4. 若上下文超限，先对旧历史和共享事实做摘要。
5. 若仍不够，再按 `Priority_i(t)` 从公共池分配额外 Token。
6. 调用结束后，把本轮消耗写入监控，并更新下一轮预算状态。

---

## 代码实现

下面给一个可运行的 Python 玩具实现。它不依赖真实模型，只模拟“统计 token 数、触发摘要、按优先级调额”的过程。这里把“token 数”简化为文本按空格切分后的词数，目的是让逻辑可运行、可验证。

```python
from dataclasses import dataclass
from typing import List, Dict

def estimate_tokens(text: str) -> int:
    return len(text.split())

def summarize_messages(messages: List[str], max_tokens: int) -> str:
    # 玩具摘要器：保留每条消息前若干词，再拼接；真实工程中这里会调用摘要模型
    words = []
    for msg in messages:
        words.extend(msg.split()[:8])
        if len(words) >= max_tokens:
            break
    return " ".join(words[:max_tokens])

@dataclass
class Agent:
    name: str
    base_budget: int
    priority: float
    context: List[str]

def allocate_budget(
    agents: List[Agent],
    global_pool: int,
    shared_messages: List[str],
    shared_summary_cap: int = 40
) -> Dict[str, int]:
    shared_raw = " ".join(shared_messages)
    raw_shared_tokens = estimate_tokens(shared_raw)
    shared_summary = summarize_messages(shared_messages, shared_summary_cap)
    summary_tokens = estimate_tokens(shared_summary)

    extra_pool = max(global_pool - summary_tokens, 0)
    total_priority = sum(a.priority for a in agents)

    allocation = {}
    for agent in agents:
        raw_context = " ".join(agent.context)
        context_tokens = estimate_tokens(raw_context)

        # 先给硬上限
        allowed = min(agent.base_budget, context_tokens)

        # 如果上下文超出硬上限，再看能否从公共池调剂
        overflow = max(context_tokens - agent.base_budget, 0)
        extra = 0
        if overflow > 0 and total_priority > 0 and extra_pool > 0:
            share = int(extra_pool * (agent.priority / total_priority))
            extra = min(overflow, share)

        allocation[agent.name] = allowed + extra

    allocation["_raw_shared_tokens"] = raw_shared_tokens
    allocation["_summary_tokens"] = summary_tokens
    return allocation

agents = [
    Agent(
        name="planner",
        base_budget=20,
        priority=3.0,
        context=["user asks for refund process", "need check policy and order history", "vip case urgent response needed"]
    ),
    Agent(
        name="retriever",
        base_budget=15,
        priority=1.0,
        context=["policy says refund within 7 days unopened items only", "order created 3 days ago item not shipped"]
    ),
    Agent(
        name="reviewer",
        base_budget=15,
        priority=2.0,
        context=["final answer must mention timeline risk compliance and exception handling"]
    ),
]

shared_messages = [
    "customer is vip and order id 12345 created three days ago",
    "product not shipped yet and warehouse status is pending",
    "policy requires confirmation of payment channel and refund window"
]

result = allocate_budget(agents, global_pool=30, shared_messages=shared_messages, shared_summary_cap=12)

assert result["_summary_tokens"] <= result["_raw_shared_tokens"]
assert result["planner"] >= 20
assert "retriever" in result and "reviewer" in result

print(result)
```

这段代码体现了四个关键点：

| 代码点 | 含义 |
|---|---|
| `base_budget` | 每个 Agent 的硬上限 |
| `global_pool` | 公共预算池 |
| `summarize_messages` | 共享消息先压缩 |
| `priority` | 谁有资格临时拿更多 Token |

如果你更习惯读伪代码，可以把调度逻辑概括为：

```python
def schedule(agent, context, shared):
    if tokens(shared) > shared_cap:
        shared = summarize(shared)

    usable = min(agent.base_budget, tokens(context))

    if tokens(context) > agent.base_budget:
        extra = borrow_from_global_pool(agent.priority)
        usable += extra

    log_usage(agent.name, usable)
    return usable
```

真实工程例子可以看客服平台。假设一个工单由 4 个节点处理：分类、知识检索、回复生成、审核。做法不是让 4 个节点都带 20 条完整历史，而是：

1. 最近 3 到 5 条消息原样保留。
2. 更早历史压成结构化摘要，只保留用户诉求、订单状态、异常点、已承诺动作。
3. 检索结果先去重，再只传摘要版本给下游。
4. VIP 工单提高 `priority`，允许回复生成节点临时多拿预算。
5. 所有节点的输入、输出、摘要前后长度都打点到监控面板。

这样做的结果通常不是“每次都最省”，而是“长期可控”。成本控制的关键不是单次最优，而是系统不会因为少数长对话或异常工单失控。

---

## 工程权衡与常见坑

第一个常见坑是不设 per-agent 硬上限。结果往往是最上游 Agent 把预算吃光，下游真正负责输出的 Agent 反而没有足够上下文。对新手可以这样理解：Agent A 连续塞进 5 条完整历史，Agent B 本该根据最新信息回复用户，却已经没有 Token 读取新消息了。

第二个坑是不做摘要，只做截断。截断就是超了就硬删，白话说是“直接把老内容砍掉”。这样确实省钱，但很容易把关键事实一起删掉，导致后续 Agent 重复追问，最终反而增加调用次数和总成本。

第三个坑是摘要质量差。摘要如果没保留订单号、用户约束、已执行动作，下游 Agent 会重新检索、重新询问、重新推理，成本和延迟一起上升。

第四个坑是只看单轮成本，不看会话累计成本。多 Agent 系统真正危险的地方在于多轮叠加。某一轮多花 500 token 可能不明显，但 30 轮以后就是结构性浪费。

下面这张表更适合工程排查：

| 坑 | 触发条件 | 后果 | 规避方式 |
|---|---|---|---|
| 无 per-agent 上限 | 默认让所有 Agent 自由带历史 | 某节点抢占预算 | 给每个 Agent 设硬上限 |
| 无共享摘要 | 原样转发完整历史 | 重复计费 | 共享事实统一摘要后广播 |
| 摘要丢关键事实 | 摘要模板过短或无结构 | 下游反复询问 | 摘要固定保留目标、约束、状态、未决项 |
| 只做截断不做压缩 | 超限时直接删旧消息 | 语义断裂 | 最近消息保留，老消息摘要 |
| 无监控 | 不记录各节点消耗 | 爆表后才发现 | 建立每 Agent Token 仪表盘与告警 |
| 优先级静态不变 | 所有任务同权重 | 紧急任务拿不到资源 | 按时效、价值、失败代价动态调度 |

这里有一个重要权衡：实时性和成本通常互相拉扯。摘要会带来额外一次模型调用，也就是额外延迟。如果任务极短、历史极少，做摘要未必划算；但一旦进入多轮协作，摘要几乎总是值得。实际经验通常是：当共享上下文会被 2 个以上 Agent 重复读取，或者历史超过固定阈值时，就应触发摘要。

---

## 替代方案与适用边界

不是所有场景都需要“动态预算 + 摘要 + 公共池”这一整套方案。更简单的方法有时更合适。

第一种替代方案是静态预算。也就是每个 Agent 固定上限，不做动态调额。这种方案实现简单，适合短任务、低风险流程，比如单轮分类、短链路审核。问题是遇到复杂任务时不够灵活，关键节点可能被卡死。

第二种是外部长期记忆。长期记忆就是把历史存在数据库、向量库或状态存储里，需要时再取，不把所有历史都塞进上下文。它的优点是长期成本低，适合客服、Copilot、长期项目协作；缺点是召回质量决定效果，如果召回不到关键事实，模型一样会出错。

第三种是低频摘要加召回。不是每轮都摘要，而是到达阈值才做一次，并把摘要存起来，下次优先读摘要。它适合中等复杂度系统，工程成本低于实时动态调度。

对比可以放在表里：

| 方案 | 优点 | 缺点 | 适用边界 |
|---|---|---|---|
| 静态预算 | 简单、稳定、容易上线 | 弹性差，关键任务可能受限 | 短任务、低复杂度协作 |
| 外部长期记忆 | 长会话最省 Token | 依赖召回质量与存储设计 | 长期对话、客服、知识助手 |
| 低频摘要 + 召回 | 实现成本适中 | 时效性不如实时摘要 | 中等长度流程 |
| 动态预算 + 摘要 + 公共池 | 质量和成本平衡最好 | 调度和监控更复杂 | 多 Agent、高频协作、成本敏感系统 |

再看一个真实工程对比。客服平台如果直接把最近 20 条消息塞进上下文，4 个子 Agent 都读一遍，那么 20 条历史会被重复计费 4 次。更省的做法是把完整历史放在外部数据库，只把最近 3 条原文和一份 100 token 左右的结构化摘要送进上下文，需要细节时再查询数据库。这种设计通常比“上下文里携带 20 条历史”便宜得多，也更容易控制延迟。

所以适用边界很明确：

- 任务短、链路短，用静态预算即可。
- 会话长、历史重，用外部记忆优先。
- 多 Agent 频繁协作、共享信息多，用动态预算加摘要最合适。

---

## 参考资料

1. AICosts，*AI Agent Cost Crisis: Budget Disaster Prevention Guide*，2025。贡献点：给出了多 Agent 成本失控的工程视角、预算池思路，以及将长历史压缩后显著降本的案例，适合做成本建模起点。  
2. Medium，*Multi-Agent AI Systems: Architecture, Implementation Challenges, and Practical Insights*，2026。贡献点：强调多 Agent 架构里的上下文叠加、角色分工与重复传递问题，适合理解为什么单 Agent 优化方法在多 Agent 下常常失效。  
3. EZClaws，*How to Reduce AI Agent Costs*，近年实操文章。贡献点：总结了摘要、上下文裁剪、低频信息外存等落地手段，适合把抽象预算策略变成可执行的工程措施。  

对新手最值得抓住的一点是：参考资料里的真实场景并不是“模型更便宜了”，而是“同样的信息不再被多个 Agent 反复阅读”。例如把 20 条对话压缩成约 100 token 的摘要，再配合每个 Agent 的硬上限和全局预算池，系统总成本会从不可控变成可预测。
