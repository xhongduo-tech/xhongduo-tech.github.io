## 核心结论

Agent 记忆管理的目标，不是把历史全部存下来，而是把少量真正有用的信息，在真正需要的时刻放回上下文窗口。这里的“上下文窗口”可以先理解成模型当前能直接看到、并据此生成回复的工作区。

最实用的工程划分是三层记忆：短期上下文 $C_t$、长期语义记忆 $S$、情节记忆 $E$。短期上下文负责当前会话内的临时状态，长期语义记忆负责跨会话稳定事实与偏好，情节记忆负责带时间顺序的事件轨迹。新手可以先把它们理解成：短期记忆像工作台，长期记忆像档案柜，情节记忆像带时间戳的日志。

难点不在“能不能存”，而在四件事：写入、检索、压缩、遗忘。记忆系统如果设计不当，会出现三类直接后果：把无关历史塞满上下文，导致记忆污染；把所有东西长期保留，导致成本失控；把过时偏好继续拿来用，导致人格漂移或决策错误。

一个统一且可解释的检索公式可以写成：

$$
score(m_i, q) = \alpha r_i + \beta i_i + \gamma v_i,\quad \alpha+\beta+\gamma=1
$$

这里不是“记忆存在就用”，而是“记忆值得进入上下文才用”。客服 agent 就是典型例子：当前工单内容进入短期上下文，用户套餐与禁用项进入长期语义记忆，每次处理步骤和结论进入情节记忆。回复前只召回最相关、最新、最重要的少量条目，而不是把全部聊天记录重新贴给模型。

| 记忆层 | 保存什么 | 保存多久 | 何时读取 | 何时清理 |
|---|---|---|---|---|
| 短期上下文 $C_t$ | 当前轮次任务、最近消息、临时状态 | 分钟到单次会话 | 每次生成前直接读取 | 会话结束、摘要完成后清理 |
| 长期语义记忆 $S$ | 稳定事实、偏好、规则、用户画像 | 天到月，甚至更久 | 跨会话决策前检索 | TTL 到期、版本冲突、人工修订时 |
| 情节记忆 $E$ | 事件序列、操作轨迹、处理结果 | 视任务价值保留 | 需要追溯过程或相似案例时 | 归档、压缩、低价值淘汰时 |

---

## 问题定义与边界

这篇文章讨论的不是“数据库怎么存”，也不是“向量库怎么查”，而是 Agent 内部的记忆管理。更准确地说，记忆管理是控制哪些信息被写入、何时被取回、如何被压缩、何时被遗忘，从而影响后续上下文窗口的过程。

可以先给出一句定义：

**记忆管理 = 写入 + 检索 + 压缩 + 遗忘**

这一定义很重要，因为很多系统把 prompt、RAG、日志、缓存都叫“记忆”，最后导致架构边界不清。Prompt 是当前直接喂给模型的输入模板。RAG 是检索增强生成，本质上是从外部知识源取材料。日志是系统记录，主要用于审计和排障。它们都可能参与记忆系统，但它们本身不等于记忆管理。

玩具例子可以用“和客服连续聊 20 轮”来说明边界：
- “你现在要退货”属于短期上下文，因为它只对当前任务直接生效。
- “你是会员、默认收货地址、禁用某种配送方式”属于长期语义记忆，因为它们跨会话稳定存在。
- “刚刚已经升级到人工坐席、等待了 8 分钟、上一次系统给出补偿方案”属于情节记忆，因为它们描述的是事件顺序和处理轨迹。

| 对象 | 它是什么 | 主要作用 | 是否等于记忆 |
|---|---|---|---|
| 短期上下文 | 当前工作区中的临时信息 | 支撑当前回复 | 是 |
| 长期语义记忆 | 稳定事实和偏好 | 支撑跨会话一致性 | 是 |
| 情节记忆 | 带时间顺序的事件记录 | 支撑回溯与经验复用 | 是 |
| Prompt | 当前输入模板与指令 | 约束模型行为 | 否 |
| RAG | 从外部资料检索内容 | 补充知识 | 否 |
| 日志 | 系统行为记录 | 审计、排障、分析 | 否 |

真正的边界判断标准不是“它存在哪里”，而是“它是否在影响未来上下文的选择”。只要一个系统负责决定“哪些过去信息要再次进入模型视野”，它就在做记忆管理。

---

## 核心机制与推导

工程上最好按一条主线理解：写入 -> 评分 -> 召回 -> 注入上下文 -> 压缩/归档/过期。先把流程讲清楚，再讨论算法细节。

一个常见流程如下：

1. 用户输入到来，先进入短期上下文。
2. 主链路 `hot path` 只做轻量写入，例如保存当前消息、更新会话状态。
3. 后台 `background` 异步判断哪些信息值得升级为长期语义记忆或情节记忆。
4. 下次生成前，根据当前查询 $q$ 检索候选记忆。
5. 用统一打分函数排序，只有满足阈值 $score>\theta$ 的条目才允许注入上下文。
6. 过长会话被压缩成摘要，低价值记忆过期或归档。

这里最关键的是评分，而不是“有没有查到”。统一公式可写成：

$$
score(m_i, q) = \alpha r_i + \beta i_i + \gamma v_i
$$

其中：
- $r_i = e^{-\lambda \Delta t_i}$ 表示新近度。意思是越近的记忆分数越高，时间越久衰减越明显。
- $i_i \in [0,1]$ 表示重要性。意思是这条记忆对风险、约束、偏好、任务结果有多关键。
- $v_i = \cos(e(m_i), e(q))$ 表示相关性。这里的 embedding 可以先理解成“把文本变成向量后的语义坐标”。

| 变量 | 含义 | 典型范围 | 工程作用 |
|---|---|---|---|
| $r_i$ | 新近度 | $[0,1]$ | 防止旧信息长期霸占上下文 |
| $i_i$ | 重要性 | $[0,1]$ | 提升高风险、高价值信息优先级 |
| $v_i$ | 相关性 | $[-1,1]$ 或近似 $[0,1]$ | 保证召回与当前问题语义对齐 |

玩具例子最能说明联合评分的必要性。设 $\alpha=0.4,\beta=0.3,\gamma=0.3$。

- 记忆 A：刚刚发生，重要，但文字上不太像当前查询。设 $r=0.8,i=0.9,v=0.2$，则 $score=0.65$。
- 记忆 B：语义很像当前问题，但很旧，也不够关键。设 $r=0.3,i=0.4,v=0.9$，则 $score=0.51$。

若阈值 $\theta=0.6$，A 进入上下文，B 不进入。结论是：最近发生且高风险的信息，可能应该压过“更像但不重要”的旧信息。很多系统召回不准，不是检索库太差，而是只按相似度排，忽略了时间和重要性。

真实工程例子可以看客服系统。用户昨天把配送偏好改成“不要电话联系，只接受短信通知”。今天他来催单，当前查询文本可能和“短信通知”并不相似，但这是一条高重要性的约束。如果只按语义相似度召回，它可能根本进不了上下文；如果联合新近度与重要性，它就会被保留，避免后续流程再次触发禁用行为。

---

## 代码实现

实现上建议拆成三个模块：写入模块、检索模块、整理模块。这样可以把实时响应和后台整理分开，避免把延迟打进主链路。复杂模型不是起点，数据结构和策略接口才是起点。

下面给出一个可运行的最小 Python 骨架，包含记忆项、打分、阈值筛选、TTL 过期和简单摘要合并逻辑：

```python
from dataclasses import dataclass
from math import exp, sqrt
from typing import List, Optional
import time

@dataclass
class MemoryItem:
    id: str
    type: str  # 'semantic' or 'episodic'
    content: str
    embedding: List[float]
    importance: float   # 0..1
    created_at: float
    ttl_days: Optional[float] = None
    version: int = 1

def cosine(a: List[float], b: List[float]) -> float:
    dot = sum(x * y for x, y in zip(a, b))
    na = sqrt(sum(x * x for x in a))
    nb = sqrt(sum(y * y for y in b))
    return 0.0 if na == 0 or nb == 0 else dot / (na * nb)

def recency_score(created_at: float, now: float, lam: float = 0.01) -> float:
    delta_days = (now - created_at) / 86400.0
    return exp(-lam * delta_days)

def expired(item: MemoryItem, now: float) -> bool:
    if item.ttl_days is None:
        return False
    return (now - item.created_at) / 86400.0 > item.ttl_days

def score(item: MemoryItem, query_embedding: List[float], now: float,
          alpha: float = 0.4, beta: float = 0.3, gamma: float = 0.3) -> float:
    r = recency_score(item.created_at, now)
    i = item.importance
    v = cosine(item.embedding, query_embedding)
    return alpha * r + beta * i + gamma * v

def retrieve(memories: List[MemoryItem], query_embedding: List[float], now: float,
             theta: float = 0.6, top_k: int = 5) -> List[MemoryItem]:
    candidates = []
    for m in memories:
        if expired(m, now):
            continue
        s = score(m, query_embedding, now)
        if s > theta:
            candidates.append((s, m))
    candidates.sort(key=lambda x: x[0], reverse=True)
    return [m for _, m in candidates[:top_k]]

def merge_summary(items: List[MemoryItem]) -> str:
    # 最小示例：真实系统通常会调用模型做压缩摘要
    return " | ".join(m.content for m in items[-3:])

now = time.time()
a = MemoryItem("A", "episodic", "刚升级到人工坐席", [1.0, 0.0], 0.9, now - 1 * 86400, ttl_days=30)
b = MemoryItem("B", "semantic", "用户喜欢深夜回电", [0.8, 0.2], 0.4, now - 90 * 86400, ttl_days=60)
q = [1.0, 0.1]

result = retrieve([a, b], q, now, theta=0.6)
assert len(result) == 1
assert result[0].id == "A"
assert expired(b, now) is True
assert "人工坐席" in merge_summary(result)
```

这个骨架有几个关键点。

第一，写入模块不要把每条原始对话都直接升格为长期记忆。更合理的做法是：当前消息先进入短期缓存，只有满足白名单条件的信息才进入候选池，例如稳定偏好、禁用项、决策结果、任务结论。

第二，检索模块要对“召回多少”做硬限制。就算阈值通过，也不能把几十条记忆全部注入上下文。大多数系统在这里需要 `top_k`、分层上限、去重和冲突检查。

第三，整理模块要异步运行。它负责摘要合并、重复项折叠、TTL 检查、版本升级和冲突处理。比如“用户偏好已从 A 改为 B”不能简单新增一条，而应把旧版本标记为失效，避免两条记忆互相打架。

---

## 工程权衡与常见坑

最大的坑是写太多、召回太满、上下文被污染。长期记忆不是聊天记录仓库，而是稳定事实、偏好、约束、决策和结果的仓库。把整段闲聊原文写进去，只会拉高检索噪声，最后让真正重要的内容被淹没。

第二个坑是只按相似度召回。语义相似度很有用，但它只回答“像不像”，不回答“新不新”和“重不重要”。工程里最容易出事故的，往往不是最像的问题，而是刚发生、影响大的约束变化。

第三个坑是没有失效机制。用户三天前说“以后都发邮箱”，今天又明确改口“只要短信”。如果系统保留两条长期记忆、又没有版本号或 TTL，它就可能继续用旧偏好回复，造成错误决策。

| 风险 | 典型表现 | 规避策略 |
|---|---|---|
| 延迟高 | 每轮都做重写入、重摘要 | 后台整理、批量合并、热路径最小化 |
| 记忆污染 | 无关聊天进入长期库 | 白名单写入、摘要后再入库 |
| 召回不准 | 只找最相似文本 | 联合评分：新近度 + 重要性 + 相关性 |
| 冲突失效 | 旧偏好与新偏好并存 | TTL、版本号、冲突检测、人工复核 |

还有一个常被忽略的问题是“人格漂移”。这里的人格漂移不是拟人化表达，而是指系统长期累积了互相矛盾、过时、低质量的信息，导致后续行为风格和决策标准不断偏移。解决思路不是继续加更多记忆，而是提升写入门槛和清理质量。

---

## 替代方案与适用边界

不是所有系统都值得上完整三层记忆。对话很短、任务简单、状态很少的场景，纯短期上下文就够了。只有当记忆会影响未来多轮决策、跨会话服务和长期个性化时，长期语义记忆和情节记忆才真正值得引入。

一次性问答机器人就是最简单的反例。用户提一个问题，系统查资料，返回答案，然后会话结束。这里没有必要建设长期记忆，因为历史不会影响未来决策。相反，长期陪伴型客服 agent、销售 agent、编程协作 agent 都会跨会话复用状态，这时分层记忆才有明显收益。

还要明确：向量库不是完整的记忆管理。它只是检索基础设施之一。真正的系统仍然要回答“什么该写入、何时该失效、如何去冲突、什么时候压缩”。

| 方案 | 实现成本 | 查询速度 | 跨会话能力 | 可解释性 | 适用任务 |
|---|---|---|---|---|---|
| 纯 Prompt | 低 | 快 | 弱 | 高 | 单轮、短对话 |
| RAG | 中 | 中 | 取决于外部知识源 | 中 | 知识问答、文档检索 |
| 外部数据库字段 | 中 | 快 | 强 | 高 | 结构化用户资料 |
| 事件日志 | 低 | 中 | 强 | 高 | 审计、追溯、分析 |
| 分层记忆系统 | 高 | 中 | 强 | 中 | 长任务、跨会话 Agent |
| 图记忆 | 高 | 中到慢 | 强 | 中 | 复杂关系推理 |

一个简单决策规则是：
- 如果记忆只影响单轮回答，不必引入长期记忆。
- 如果记忆会影响未来多轮决策，就需要分层设计。
- 如果系统更需要“查知识”，优先考虑 RAG。
- 如果系统更需要“记住这个用户和这次过程”，优先考虑记忆架构。

---

## 参考资料

概念来源与工程实现来源需要分开理解。本文的三层记忆划分借鉴了生成式智能体、状态化 agent、记忆层级与认知架构相关资料；上文的评分公式是对常见工程做法的统一抽象，不是某一篇论文的唯一原始表述。

1. [Generative Agents: Interactive Simulacra of Human Behavior](https://research.google/pubs/generative-agents-interactive-simulacra-of-human-behavior/) 用途：支持“情节记忆、反思、行为轨迹”这条线。
2. [Letta Docs: Stateful Agents / Memory Hierarchy](https://docs.letta.com/guides/core-concepts/stateful-agents) 用途：支持“分层记忆”与状态化 agent 的工程表达。
3. [Letta Docs: Archival Memory](https://docs.letta.com/guides/core-concepts/memory/archival-memory) 用途：支持“归档记忆”与长期存储的工程实践。
4. [LangChain Docs: Memory Overview](https://docs.langchain.com/oss/javascript/concepts/memory) 用途：支持“短期/长期/对话记忆”的产品化语境。
5. [LangChain Docs: Short-term Memory](https://docs.langchain.com/oss/javascript/langchain/short-term-memory) 用途：支持短期上下文管理的实现思路。
6. [Cognitive Architectures for Language Agents](https://collaborate.princeton.edu/en/publications/cognitive-architectures-for-language-agents/) 用途：支持把记忆放进整体认知架构中理解。
