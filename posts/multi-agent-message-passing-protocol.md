## 核心结论

多 Agent 通信协议的核心，不是先决定“用黑板、点对点还是发布订阅”，而是先定义统一消息信封。所谓消息信封，就是消息外层那组所有 Agent 都看得懂的标准字段。一个足够通用的定义可以写成：

$$
M=\{sender,\ receiver,\ content,\ metadata\}
$$

其中：

- `sender` 是发件方，白话说就是“这条消息是谁发的”。
- `receiver` 是目标方，白话说就是“这条消息打算给谁”。
- `content` 是正文，白话说就是“真正要处理的数据”。
- `metadata` 是附表，白话说就是“帮助路由、重试、追踪、限流的控制信息”。

关键点有两个。

第一，统一信封后，三种通信范式可以共存。黑板适合共享中间状态，点对点适合明确委派，发布订阅适合广播事件。它们不必各自发明一套字段，只要底层都收同一种消息，基础设施就能按 `receiver`、`topic`、`priority`、`ttl` 做路由。

第二，真正决定系统可扩展性的不是 `content`，而是 `metadata`。新增 Agent 时，通常不需要和所有旧 Agent 两两协商协议；它只要理解标准元数据，就能接入现有队列、订阅主题、读取黑板快照，并参与同一条任务链路。

一个给新手的玩具例子是“标准信封邮局”。邮局不关心信里写了什么，它先看信封：发件人是谁、收件人是谁、优先级高不高、多久过期。这样无论是普通信、挂号信还是群发通知，都能走同一套分拣系统。多 Agent 协议设计也是同一个逻辑。

| metadata 字段 | 作用 | 路由/调度行为 |
|---|---|---|
| `message_id` | 消息唯一 ID，用来去重 | 避免重复消费 |
| `correlation_id` | 同一任务链路 ID，用来把请求和响应串起来 | 聚合多个 Agent 的结果 |
| `timestamp` | 发送时间 | 计算等待时长、做监控 |
| `ttl` | 生存时间，超过即失效 | 丢弃过期消息，避免旧结果污染 |
| `priority` | 优先级 | 高优先级先出队 |
| `topic` | 主题标签 | 发布订阅时决定哪些 Agent 收到 |
| `trace_id` | 追踪 ID | 端到端观测与审计 |

---

## 问题定义与边界

多 Agent 系统的通信问题，本质上是“异构执行单元如何在不同时序、不同耦合度下交换可执行信息”。异构执行单元，白话说就是“职责不同、实现不同、速度也不同的一组 Agent”。

常见的三种通信范式如下。

| 范式 | 典型角色 | 耦合度 | 适用场景 | 切换条件 |
|---|---|---:|---|---|
| 共享黑板 | planner、critic、cleaner | 中 | 共享中间结果、反复迭代求解 | 需要公共上下文时使用 |
| 点对点消息 | orchestrator -> tool-agent | 高 | 指定某个 Agent 执行明确任务 | 需要确认接收方和回复时使用 |
| 发布订阅 | event bus -> observers | 低 | 广播状态变化、触发观察者 | 发送方不想知道接收方是谁时使用 |

给新手的直观理解是：团队协作里，有人把信息写在公告板上，这是黑板；有人直接发邮件给某个人，这是点对点；有人在群组里发“新版本已发布”，谁订阅谁收到，这是发布订阅。

边界也要说清楚。本文讨论的是“面向工程落地的 Agent 间消息协议”，重点在：

- 如何统一消息格式。
- 如何在三种通信范式之间切换。
- 如何用队列处理异步协作中的缓冲和背压。

本文不展开：

- LLM 提示词设计。
- 复杂一致性协议，比如 Raft、Paxos。
- 强事务数据库语义。
- 跨数据中心网络容灾。

也就是说，这里解决的是“消息怎么定义、怎么流动、峰值怎么不炸”，不是“所有 Agent 内部推理怎么做”。

---

## 核心机制与推导

统一信封后，通信基础设施就可以把“消息内容”和“消息控制”分开处理。内容交给业务 Agent，控制交给队列、路由器和调度器。

### 1. 统一信封如何兼容三种范式

如果 `receiver` 是具体 Agent ID，就是点对点。  
如果 `receiver` 是 `blackboard`，就是向共享黑板写入。  
如果 `metadata.topic` 有值并由多个消费者订阅，就是发布订阅。

这意味着三种范式不是三套基础设施，而是同一条消息在不同路由规则下的三种投递方式。

### 2. metadata 为什么决定可扩展性

`message_id` 解决幂等。幂等，白话说就是“重复处理同一条消息，结果不能越做越错”。  
`correlation_id` 解决链路追踪。  
`ttl` 解决旧消息污染。  
`priority` 解决紧急任务插队。  
`timestamp` 解决监控和延迟计算。

新增一个 `reviewer-agent` 时，它不需要重新定义“如何关联上游研究结果”，直接读取 `correlation_id` 即可；也不需要发明“紧急复核”这类私有字段，直接复用 `priority=high` 即可。

### 3. 队列为什么是异步协作的缓冲层

队列的作用不是“把消息存起来”这么简单，而是把生产速度和消费速度解耦。解耦，白话说就是“发送方不用等接收方立刻处理完”。

一个简化公式可以写成：

$$
throughput = \min(total\_supply,\ queue\_capacity)
$$

这里 `total_supply` 是生产速率，`queue_capacity` 可以理解为当前系统可稳定处理的吞吐上限。若供给高于处理能力，多出来的部分只能进入积压，积压继续上升就会增加等待延迟。

玩具例子：

- `research`、`filter`、`review` 三个 Agent，各自产生 8 条消息/秒。
- 总供给为 $8+8+8=24$ 条/秒。
- 队列稳定吞吐是 20 条/秒。
- 那么系统每秒净积压 4 条。

真实工程例子：

- 正常流量下，系统入口 20 条/秒，消费者也能处理 20 条/秒，P95 延迟稳定在 180ms。
- 某次批量任务触发后，入口瞬时升到 40 条/秒。
- 如果不做背压，队列深度会持续上涨，延迟很快突破 250ms。
- 如果设置阈值 `queue_depth > 25` 触发背压，把普通 Agent 降到 5 条/秒，并优先处理 `priority=high` 的复核消息，系统可回落到可控区间。

下面用一个简化表说明这个过程：

| 阶段 | Research 频率 | Filter 频率 | Review 频率 | 队列状态 | 估计延迟 |
|---|---:|---:|---:|---|---:|
| 平稳期 | 8/s | 8/s | 8/s | 轻度积压 | 180ms |
| 峰值期 | 14/s | 14/s | 12/s | 深度快速上升 | >300ms |
| 背压触发后 | 5/s | 5/s | 10/s | 高优先级先清空 | 220-250ms |
| 恢复期 | 6/s | 6/s | 8/s | 深度下降 | <200ms |

一个常见但有效的背压策略是：

```text
if queue_depth > threshold:
    reduce send_rate
    enqueue by priority
    drop expired messages by ttl
```

这个机制的工程含义是：先保命，再保全量。系统先保证关键任务不超时，再决定普通任务是延后、降频还是过期丢弃。

---

## 代码实现

下面给一个可运行的 Python 简化实现。它展示三件事：

1. 用统一 `Message` 类表示消息。
2. 用 `priority` 和 `ttl` 做调度。
3. 当 `queue_depth >= 25` 时触发背压，把发送速率降到 5 条/秒。

```python
from dataclasses import dataclass, field, asdict
from typing import Any, Dict, Optional, List
from collections import deque
import time
import uuid

@dataclass
class Message:
    sender: str
    receiver: str
    content: Any
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        self.metadata.setdefault("message_id", str(uuid.uuid4()))
        self.metadata.setdefault("correlation_id", self.metadata["message_id"])
        self.metadata.setdefault("timestamp", time.time())
        self.metadata.setdefault("ttl", 10.0)  # seconds
        self.metadata.setdefault("priority", "normal")
        self.metadata.setdefault("topic", None)

    def is_expired(self, now: Optional[float] = None) -> bool:
        now = now or time.time()
        return now - self.metadata["timestamp"] > self.metadata["ttl"]

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Message":
        return cls(**data)


class Broker:
    PRIORITY_ORDER = {"high": 0, "normal": 1, "low": 2}

    def __init__(self):
        self.queue: List[Message] = []
        self.blackboard: List[Dict[str, Any]] = []
        self.subscriptions: Dict[str, List[str]] = {}
        self.send_rate = 8

    def subscribe(self, topic: str, agent_id: str):
        self.subscriptions.setdefault(topic, []).append(agent_id)

    def enqueue_message(self, msg: Message):
        if not msg.is_expired():
            self.queue.append(msg)
            self.queue.sort(key=lambda m: self.PRIORITY_ORDER[m.metadata["priority"]])

    def apply_backpressure(self):
        queue_depth = len(self.queue)
        self.send_rate = 5 if queue_depth >= 25 else 8
        return self.send_rate

    def dispatch_by_metadata(self):
        delivered = []
        while self.queue:
            msg = self.queue.pop(0)
            if msg.is_expired():
                continue
            if msg.receiver == "blackboard":
                self.blackboard.append({
                    "sender": msg.sender,
                    "content": msg.content,
                    "version": len(self.blackboard) + 1,
                    "message_id": msg.metadata["message_id"],
                })
                delivered.append(("blackboard", msg.content))
            elif msg.metadata.get("topic"):
                for agent_id in self.subscriptions.get(msg.metadata["topic"], []):
                    delivered.append((agent_id, msg.content))
            else:
                delivered.append((msg.receiver, msg.content))
        return delivered


broker = Broker()
broker.subscribe("task.updated", "observer-a")
broker.subscribe("task.updated", "observer-b")

msg1 = Message(
    sender="researcher",
    receiver="blackboard",
    content={"finding": "need review"},
    metadata={"priority": "high", "ttl": 5}
)

msg2 = Message(
    sender="planner",
    receiver="router",
    content={"event": "task updated"},
    metadata={"topic": "task.updated", "priority": "normal"}
)

broker.enqueue_message(msg1)
broker.enqueue_message(msg2)

for i in range(25):
    broker.enqueue_message(Message(sender="loadgen", receiver="worker", content=i))

assert broker.apply_backpressure() == 5

result = broker.dispatch_by_metadata()
assert any(target == "blackboard" for target, _ in result)
assert any(target == "observer-a" for target, _ in result)
assert any(target == "observer-b" for target, _ in result)

data = msg1.to_dict()
msg1_copy = Message.from_dict(data)
assert msg1_copy.sender == "researcher"
assert msg1_copy.metadata["priority"] == "high"
```

上面这段代码不是生产实现，但结构是对的。生产实现通常会把内存队列换成 Kafka、RabbitMQ、Azure Service Bus 一类消息中间件，把 `blackboard` 换成带版本和审计的持久化存储。

关键函数可以概括如下：

| 函数 | 作用 |
|---|---|
| `enqueue_message` | 入队并按优先级排序 |
| `apply_backpressure` | 根据队列深度调整发送速率 |
| `dispatch_by_metadata` | 根据 `receiver/topic` 决定投递到黑板、点对点或订阅者 |
| `is_expired` | 检查消息是否过期 |
| `subscribe` | 维护主题订阅关系 |

一个简化流程可以理解为：

1. Agent 生成统一格式消息。
2. Broker 读取 `metadata` 决定是否入队。
3. 队列深度过高时触发背压。
4. Router 根据 `receiver` 或 `topic` 投递。
5. 写黑板时记录版本、来源和审计信息。

---

## 工程权衡与常见坑

黑板、点对点、发布订阅都不是“最好”的方案，它们只是“在某类约束下更合适”的方案。

最大的工程风险通常不是消息发不出去，而是消息“看起来发对了，实际含义已经错了”。

| 常见坑 | 问题表现 | 规避措施 |
|---|---|---|
| 黑板污染 | 错误中间结果被所有 Agent 继续引用 | 写入权限、版本号、签名、审计链 |
| 元数据不统一 | 新 Agent 接入困难，路由规则越来越多 | 统一 envelope 和字段命名 |
| 缺乏背压 | 队列积压后延迟失控 | 设深度阈值、降频、限流 |
| 不做幂等 | 重试造成重复执行 | 用 `message_id` 去重 |
| 不设 `ttl` | 老消息在高峰后继续污染状态 | 过期即丢弃 |
| 过度依赖单一范式 | 要么共享状态过多，要么广播过多 | 混合使用三种范式 |

黑板尤其危险，因为它本质上是共享可变状态。共享可变状态，白话说就是“很多人都能改同一份东西”。一旦没有权限控制，任一 Agent 都可以写入“任务完成”，后面的 Agent 看到这个状态就可能提前结束流程。

下面是一个简短策略片段：

```python
def can_write_blackboard(agent_id: str, role: str) -> bool:
    allowed_roles = {"planner", "critic", "cleaner"}
    return role in allowed_roles and agent_id.startswith("agent-")

def audit_record(message: Message, action: str) -> dict:
    return {
        "message_id": message.metadata["message_id"],
        "sender": message.sender,
        "action": action,
        "timestamp": message.metadata["timestamp"],
        "priority": message.metadata["priority"],
    }
```

另一个高频坑是“队列已经 25 条了还不降频”。很多系统一开始只看吞吐，不看等待时间；但对 Agent 协作来说，排队太久会让上下文失效，特别是带 `ttl` 的消息会成批过期，最后吞吐看似还在，真正有用的结果却下降。

---

## 替代方案与适用边界

如果任务低耦合、主要是广播通知，那么只用发布订阅就够了。发送方只发“文档已更新”“任务已完成”，谁关心谁订阅。

如果任务需要明确责任归属和结果确认，点对点更合适。比如 orchestrator 指定 `retriever-agent` 去抓数据，然后等待它返回特定结果。

如果任务需要反复共享中间状态，比如 planner、critic、cleaner 围绕同一问题多轮协作，黑板更合适。bMAS 这类系统的价值就在这里：Agent 不是互相私聊，而是围绕同一块公共上下文迭代。

可以用下表快速选择：

| 条件 | 推荐方案 |
|---|---|
| 耦合度低，观察者多 | 发布订阅 |
| 需要确认接收方和回复 | 点对点 |
| 需要共享中间状态 | 黑板 |
| 需要强审计、可回放 | 消息队列 + 持久化事件流 |
| 担心黑板被随意修改 | 队列写入，控制单元再落板 |

一个常见替代方案是“不要让 Agent 直接写黑板，而是先写入消息队列，由控制单元审核后再写黑板”。这样做的好处是有回放能力。回放，白话说就是“系统可以按历史消息重新复原当时发生了什么”。

伪代码如下：

```text
agent -> queue.append(message)
controller <- queue.consume()
if validate(message) and not expired(message):
    blackboard.write(versioned_entry)
    audit_log.append(message)
else:
    dead_letter_queue.append(message)
```

这个方案牺牲了一点实时性，换来更强的可审计性和污染隔离能力。对金融、医疗、客服质检这类需要追责的系统，更适合走这条路线。

---

## 参考资料

1. Han, Bochen; Zhang, Songmao. *Exploring Advanced LLM Multi-Agent Systems Based on Blackboard Architecture*，arXiv:2507.01701，2025。用途：说明黑板式多 Agent 中 control unit、planner、critic、cleaner 的协作方式，以及在 commonsense 和数学任务上的效果。  
   链接：[https://www.emergentmind.com/papers/2507.01701](https://www.emergentmind.com/papers/2507.01701)

2. Microsoft Multi-Agent Reference Architecture, *Message-Driven Communication*，2025-06-30 更新。用途：说明消息驱动通信中的异步解耦、相关 ID、可观测性、幂等和队列缓冲。  
   链接：[https://microsoft.github.io/multi-agent-reference-architecture/docs/agents-communication/Message-Driven.html](https://microsoft.github.io/multi-agent-reference-architecture/docs/agents-communication/Message-Driven.html)

3. SW4RM Agentic Protocol, *Overview* 与 *Protocol Specification*。用途：说明标准化消息信封、ACK 生命周期、流控与可互操作协议字段。  
   链接：[https://sw4rm.ai/overview/](https://sw4rm.ai/overview/)  
   链接：[https://sw4rm.ai/protocol/](https://sw4rm.ai/protocol/)

4. Ranjan Kumar, *Why Asynchronous Processing Queues Are the Backbone of Agentic AI*，2025。用途：说明异步队列在缓冲、限流、优先级调度、并发 Agent 协作中的工程价值。  
   链接：[https://ranjankumar.in/why-asynchronous-processing-queues-are-the-backbone-of-agentic-ai/](https://ranjankumar.in/why-asynchronous-processing-queues-are-the-backbone-of-agentic-ai/)

5. Emergent Mind, *Blackboard LLM Multi-Agent System (bMAS)*。用途：作为 bMAS 架构摘要，帮助快速理解“所有交互经由黑板”这一设计。  
   链接：[https://www.emergentmind.com/topics/blackboard-based-llm-multi-agent-system-bmas](https://www.emergentmind.com/topics/blackboard-based-llm-multi-agent-system-bmas)
