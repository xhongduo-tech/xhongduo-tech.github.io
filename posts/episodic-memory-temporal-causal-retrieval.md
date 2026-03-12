## 核心结论

情景记忆是 Agent 对“自己经历过什么”的长期记录。这里的“经历”不是抽象知识，而是一次对话、一次工具调用、一次决策、一次失败及其后果。要让这类记忆可用，检索层不能只做向量相似度，还必须显式处理两个维度：

1. 时序维度：回答“什么时候发生”“上次是什么时候”“在 Q3 之前发生了什么”。
2. 因果维度：回答“为什么这样做”“哪个事件触发了这次失败”。

公开结果已经说明这一点。Mastra 在 LongMemEval 上仅做语义召回时，`temporal-reasoning` 最高约为 75.2%，总体准确率约 80.05%；Hindsight 这类带结构化时间与图检索的系统，公开报道中 `temporal reasoning` 从 31.6% 提升到 79.7%，总体准确率达到 91.4%。这说明单纯“像不像”不足以回答“何时”和“为何”。

| 方案 | temporal reasoning | overall accuracy | 能否直接回答“上次失败是什么时候” | 能否直接回答“为什么选 A” |
| --- | ---: | ---: | --- | --- |
| 纯向量/工作记忆基线 | 29.3% - 31.6% | 35.4% 起 | 弱 | 弱 |
| 语义召回 RAG | 61.7% - 75.2% | 73.98% - 80.05% | 中 | 中 |
| 时间+结构化因果记忆 | 79.7% | 91.4% | 强 | 强 |

结论可以压缩成一句话：情景记忆要从“相似文本搜索”升级为“带时间轴和因果边的事件检索”。

---

## 问题定义与边界

情景记忆第一次出现时可以先这样理解：它就是 Agent 的“经历日志”，类似程序的运行日志，但每条记录不仅有文本，还有结构化上下文。

这类系统解决的是事件级问题，不是百科问答。它更关心：

- 某件事是否真的发生过
- 发生在什么时间区间
- 这次结果由哪些前置事件触发
- 同一线程或跨线程里，哪些记录属于同一条决策链

边界也要先说清楚。不是所有记忆都适合做时序索引。

| 字段 | 含义 | 主要用途 |
| --- | --- | --- |
| `threadId` | 会话或任务链标识，白话说就是“这段经历属于哪条线” | 限定局部上下文 |
| `occurrence_time` | 事件实际发生时间 | 回答“什么时候发生” |
| `mention_time` | 记录被写入系统的时间 | 回答“最近聊过什么” |
| `cause_link` | 指向触发该事件的上游事件 | 回答“为什么” |
| `tags` | 主题标签或事件类型 | 粗筛选 |
| `embedding` | 语义向量，白话说就是“文本内容的数学表示” | 语义召回 |

最容易被忽略的是双时间。Nicolò Boschi 对这个问题讲得很直白：`occurrence` 是“事情何时发生”，`mention` 是“系统何时知道这件事”。两者不一样。

玩具例子：

- 记忆内容：“Alice 去年 6 月结婚了。”
- `occurrence_time = 2024-06`
- `mention_time = 2025-01-15`

这样两个问题才能同时答对：

- “2024 年 6 月 Alice 发生了什么？”查 `occurrence_time`
- “我们最近聊过 Alice 什么事？”查 `mention_time`

如果你只有一个时间戳，两个问题里必然有一个会错。

---

## 核心机制与推导

向量检索第一次出现时可以理解为“按语义像不像排序”。它擅长找“主题相近”的记忆，但不天然理解“先后顺序”和“因果链”。

因此检索分数通常要变成多信号融合。工程上可写成：

$$
score(m \mid q)=\alpha \cdot s_{sem}(q,m)+\beta \cdot s_{time}(q,m)+\gamma \cdot s_{cause}(q,m)
$$

其中：

- $s_{sem}$：语义相似度，通常来自 cosine similarity。
- $s_{time}$：时间一致性，白话说就是“这条记忆在不在用户问的时间范围里，而且离目标时间近不近”。
- $s_{cause}$：因果一致性，白话说就是“这条记忆是不是决策链上的触发节点或关键中间节点”。

如果查询是“上次哪个操作失败了”，则 $\beta$ 要高，因为时间是主约束；如果查询是“为什么选方案 A”，则 $\gamma$ 要高，因为你要优先找到触发决策的边。

一种实用做法是分三步，而不是一次性把所有分数揉在一起：

1. 语义召回候选：先用 embedding 取 Top-K。
2. 时间重排：再用 `threadId`、时间区间、最近性排序过滤。
3. 因果补链：沿 `cause_link` 或图边回溯 1 到 3 跳，把原因链补齐。

TEMPR 这类系统的思路更进一步：语义、关键词、图谱、时间并行跑，再用 RRF 融合。RRF 第一次出现时可以理解为“多路检索投票式合并”，谁在多条检索路线上都排名靠前，谁最终得分就高。

玩具例子：

用户问：“为什么上周报警触发了？”

系统的正确流程不是只搜“报警”，而是：

1. 用向量找到与“报警、触发、配置”相关的候选事件。
2. 把时间范围收缩到“上周”。
3. 沿因果链追到：
   - 配置改为 X
   - 导致负载激增
   - 超过阈值
   - 触发报警

没有第 2 步，会混入历史报警；没有第 3 步，只会返回“报警发生了”，答不出“为什么”。

---

## 代码实现

下面先给一个最小可运行的 Python 玩具实现。它不依赖向量库，只演示“语义分 + 时间分 + 因果分”的基本思路。

```python
from math import exp
from datetime import datetime

memories = [
    {
        "id": "e1",
        "text": "deploy failed because env VAR_X was removed",
        "semantic": 0.72,
        "thread_id": "t1",
        "occurred_at": "2026-03-01T10:00:00",
        "cause_link": None,
        "type": "failure",
    },
    {
        "id": "e2",
        "text": "rollback executed after failed deploy",
        "semantic": 0.68,
        "thread_id": "t1",
        "occurred_at": "2026-03-01T10:05:00",
        "cause_link": "e1",
        "type": "action",
    },
    {
        "id": "e3",
        "text": "deploy failed because image tag was wrong",
        "semantic": 0.70,
        "thread_id": "t1",
        "occurred_at": "2026-02-20T09:00:00",
        "cause_link": None,
        "type": "failure",
    },
]

query_time = datetime.fromisoformat("2026-03-01T11:00:00")
target_thread = "t1"

def time_weight(occurred_at: str, now: datetime) -> float:
    delta_hours = abs((now - datetime.fromisoformat(occurred_at)).total_seconds()) / 3600
    return exp(-delta_hours / 24)  # 24小时衰减

def cause_weight(memory: dict) -> float:
    return 1.0 if memory["cause_link"] is not None else 0.3

def score(memory: dict, alpha=0.5, beta=0.3, gamma=0.2) -> float:
    if memory["thread_id"] != target_thread:
        return -1
    return (
        alpha * memory["semantic"]
        + beta * time_weight(memory["occurred_at"], query_time)
        + gamma * cause_weight(memory)
    )

ranked = sorted(memories, key=score, reverse=True)

assert ranked[0]["id"] == "e2"
assert ranked[1]["id"] == "e1"
assert score(ranked[0]) > score(ranked[1]) > score(ranked[2])

print([m["id"] for m in ranked])
```

这个例子里，`e2` 的语义分不一定最高，但它更接近目标时间，而且挂在失败事件之后，因此更适合回答“上次失败后做了什么”。

真实工程里，数据通常落在支持向量索引和过滤查询的存储上。以 Azure Cosmos DB 为例，官方推荐的模式是“一次 turn 一条文档”，核心字段包括 `threadId`、`timestamp`、`embedding`、`metrics`。这样可以先拿最近上下文，再做向量或混合查询。

按时间取最近记录：

```sql
SELECT TOP @k c.content, c.timestamp
FROM c
WHERE c.threadId = @threadId
  AND c.timestamp BETWEEN @start AND @end
ORDER BY c.timestamp DESC
```

按语义取候选：

```sql
SELECT TOP @k c.content, c.timestamp, VectorDistance(c.embedding, @queryVector) AS dist
FROM c
WHERE c.threadId = @threadId
ORDER BY VectorDistance(c.embedding, @queryVector)
```

做混合检索：

```sql
SELECT TOP @k c.content, c.timestamp, VectorDistance(c.embedding, @queryVector) AS dist
FROM c
WHERE c.threadId = @threadId
ORDER BY RANK RRF(
  VectorDistance(c.embedding, @queryVector),
  FullTextScore(c.content, @searchString)
)
```

真实工程例子：

一次部署助手的记忆文档可以长这样：

- `threadId = prod-release-2026-03-01`
- `timestamp = 2026-03-01T10:05:00Z`
- `content = rollback executed after failed deploy`
- `tags = ["deploy", "rollback", "failure"]`
- `cause_link = deploy-step-17`
- `metrics.latencyMs = 177`

当用户问“上次哪个操作失败了，为什么失败”，应用层的流程通常是：

1. 先在当前 `threadId` 下找最近的 `failure`。
2. 再回溯它的 `cause_link`。
3. 最后把“失败事件 + 原因节点 + 处理动作”一起喂给模型生成答案。

---

## 工程权衡与常见坑

第一个坑是只存一个时间戳。这会把“发生时间”和“写入时间”混掉，导致历史事实无法按事件时间检索。

第二个坑是默认只按 ingestion 排序。ingestion 第一次出现时可以理解为“写库时间”。它方便做最近记录，但会把“去年发生、今天补录”的事件排到很前面，误导时序推理。

第三个坑是没有因果链。没有 `cause_link` 时，模型只能从语义近似片段里猜“为什么”，结果通常是把背景信息、后果、修复动作混成一个解释。

| 常见坑 | 结果 | 规避策略 |
| --- | --- | --- |
| 只有 `mention_time` | “去年发生的事”查不准 | 同时存 `occurrence_time` 与 `mention_time` |
| 默认按写入时间排序 | 最近补录的旧事件被误判成最近发生 | 查询时显式区分两种时间 |
| 没有 `cause_link` | “为什么”回答不连贯 | 给决策、工具调用、报警建立边 |
| 只有向量检索 | 时间题和解释题精度差 | 先召回，再过滤，再补链 |
| 跨线程无边界 | 其他任务的相似文本混入 | 先按 `threadId` 或 tenant 限定 |
| 事件粒度过粗 | 一条文档里塞完整线程，无法精确定位原因 | 一次 turn 或一次事件一条文档 |

真实工程里，医疗、金融、运维这类系统尤其需要因果链，因为“为什么报警”“为什么拒绝交易”“为什么改配置”都要可追溯。这里的可追溯不是把全部历史都塞进 prompt，而是先用结构化索引把关键链路缩到几条高置信事件，再交给模型做语言表达。

---

## 替代方案与适用边界

不是所有 Agent 都需要完整的时间因果记忆。

如果你的场景只是单线程问答助手，且问题多为“帮我找一下刚才说过什么”，那么“向量 + 最近时间排序”已经足够，成本低，延迟也低。

如果你的场景有明显的主题模式，例如固定领域客服、固定模板工作流，那么 RAG + 模板化摘要仍然有效。Mastra 的公开结果说明，只做语义召回也能把总体准确率推到约 78.59% 到 80.05%，对很多产品已经够用。

但一旦问题进入下面三类，就该升级：

1. 明确时间边界：如“Q3 之前”“上周”“上次失败”。
2. 明确解释链：如“为什么选 A”“什么导致报警”。
3. 多跳回溯：如“在 Alice 晋升前，团队在做什么”。

| 方案 | 成本 | 时序能力 | 因果能力 | 适用场景 |
| --- | --- | --- | --- | --- |
| 纯向量检索 | 低 | 弱 | 弱 | 单会话、轻量助手 |
| RAG + template | 中 | 中 | 中 | 垂直领域助手、固定问题分布 |
| 时间+因果记忆 | 中到高 | 强 | 强 | 多线程 Agent、审计、规划、复盘 |

一个简单判断标准是：如果你的用户会经常问“上次”“之前”“为什么”，那就不要只做向量库。

---

## 参考资料

- Azure Cosmos DB, Agent memories in Azure Cosmos DB for NoSQL: https://learn.microsoft.com/en-us/azure/cosmos-db/gen-ai/agentic-memories
- VentureBeat, With 91% accuracy, open source Hindsight agentic memory provides 20/20 vision for AI agents stuck on failing RAG: https://venturebeat.com/data/with-91-accuracy-open-source-hindsight-agentic-memory-provides-20-20-vision
- Mastra Research, Yes, you can use RAG for agent memory: https://mastra.ai/research/use-rag-for-agent-memory
- Nicolò Boschi, Temporal reasoning: "when it happened" vs "when you learned it": https://nicoloboschi.com/posts/20251223/
- Mnemosyne, We Built the First AI Agent Memory System With Zero LLM Calls — Here's the Architecture: https://dev.to/mnemosybrain/we-built-the-first-ai-agent-memory-system-with-zero-llm-calls-heres-the-architecture-5hgc
