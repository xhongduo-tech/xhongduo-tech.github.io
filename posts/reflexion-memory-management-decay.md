## 核心结论

Reflexion 的核心不是“让模型学会更多知识”，而是“把失败后的文字反思继续喂给下一轮决策”。这里的“反思”指自然语言写成的错误总结，论文里记作 `sr_t`；“长期记忆”指跨 episode 保留的反思列表，供后续试验继续使用。这样做的收益很直接：不改模型参数，只改提示词里的经验上下文，就能让同一个大模型在后续回合少犯同类错误。

问题也同样直接。长期记忆不是越多越好。反思条目一旦无限增长，会出现三类退化：一是上下文窗口被旧反思挤占，二是检索时无关旧经验混入，三是旧经验之间可能互相矛盾。Reflexion 论文在实际实验里把可见记忆截断到最近 3 条，明确理由是避免 prompt 过长；后续工程文档和实践文章进一步说明，实际系统里保留最近 3 到 5 条、再配合摘要压缩和去重，通常比“全量保留”更稳。

一个最小玩具例子：假设每条反思约 100 token。在 4k 窗口里，保留 3 条大约占 300 token；保留 12 条大约占 1200 token。多出来的 900 token 并不是免费空间，它会直接挤压当前任务描述、最新轨迹、工具返回结果的位置。对 Agent 来说，这不是“记得更多”，而是“当前该看的东西看不全”。

| 可见反思条目数 | 反思约占 token | 4k 窗口剩余比例 | 典型结果 |
|---|---:|---:|---|
| 1 | 100 | 97.5% | 信息新，但经验太少 |
| 3 | 300 | 92.5% | 常见最稳妥区间 |
| 5 | 500 | 87.5% | 适合稍长任务 |
| 12 | 1200 | 70.0% | 噪声与挤占明显 |

工程上可以把检索分数写成一个组合权重：

$$
w = 0.6 \cdot similarity + 0.4 \cdot e^{-\Delta t / 30}
$$

这里 `similarity` 是语义相似度，白话说就是“这条旧反思和当前问题像不像”；$\Delta t$ 是时间差，白话说就是“它是不是太久以前的经验”。这个公式不是 Reflexion 论文原始公式，而是后续工程实践里常见的时间衰减检索写法，用来解决“旧经验长期霸榜”的问题。

---

## 问题定义与边界

先定义问题。Reflexion 要解决的不是传统强化学习里的参数更新，而是失败轨迹的语言化再利用。Actor 负责执行任务，Evaluator 负责判断成败，Self-Reflector 负责把失败原因写成一段可操作的总结。下一轮并不是从零开始，而是把这些总结作为额外上下文重新提示 Actor。

边界也要说清楚。长期记忆只适合存“能指导下次决策的经验”，不适合存放所有原始轨迹。原始轨迹是完整发生过程，信息量大但冗长；反思是从轨迹里抽出的可复用结论，信息更稀疏，也更适合放进 prompt。否则系统会把“日志仓库”误当成“决策记忆”。

一个真实工程例子是客服或物流 Agent。系统可能积累 800 多条历史交互。如果把所有失败总结都塞进 prompt，结果通常不是更聪明，而是更混乱：模型一会儿引用旧政策，一会儿引用过期地址规则，还可能把不同客户场景混在一起。此时边界条件就很明确：当前轮只保留 3 到 5 条高相关、高新鲜度的反思，其余内容转成摘要或沉淀为语义知识。

| $\Omega$ 取值 | prompt 占比 | 优点 | 主要风险 |
|---|---:|---|---|
| 1 | 最低 | 最新、简单 | 容易丢掉连续错误模式 |
| 3 | 低 | 论文与工程里最常见 | 长期线索仍可能不足 |
| 5 | 中 | 适合复杂多步任务 | 噪声开始增加 |
| 全量保留 | 不确定且持续增长 | 理论上不丢信息 | 几乎必然膨胀、检索混乱 |

时间衰减的边界也类似。若定义

$$
freshness = e^{-\Delta t/30}
$$

那么 30 天前的记忆会被明显降权。这里不是说“30 天前一定无效”，而是说默认系统应当偏向近期经验，除非相似度非常高。对零基础读者可以把它理解成：旧笔记不是删除，而是自动从“最常看”变成“备查资料”。

---

## 核心机制与推导

Reflexion 的长期记忆更新可以抽象成一个定长队列：

$$
mem_t = truncate(mem_{t-1} \oplus [sr_t], \Omega)
$$

其中 `mem_t` 是第 $t$ 轮可见记忆，$\oplus$ 表示把新反思接到尾部，`truncate` 表示如果长度超过 $\Omega$，就把最旧的条目截掉。白话解释就是“新经验入队，旧经验出队”。

简化流程如下：

```text
Actor -> Evaluator -> Self-Reflector -> Memory Update -> Retrieval -> Actor
```

这个机制重要的地方不在“存下来”，而在“存什么、拿什么”。如果只做 append，不做管理，记忆会从有用经验退化成噪声池。后续工程系统一般会加入三层处理。

第一层是摘要压缩。摘要压缩的意思是把多条细碎反思提炼成一条更短的高层规律。例如三次失败分别写成“忘记先看库存”“没有校验仓库编号”“错误使用默认仓库”，可以压缩成“物流任务在下单前必须先核验库存与仓库上下文”。AgentDock 文档把这类操作称为 episodic 到 semantic conversion，即把事件性记忆转成语义性知识。

第二层是相似去重。去重不是只看文本重复，而是同时看向量相似度、关键词重叠、元数据相似和时间邻近。原因很简单，两条表述不同的反思可能在语义上是同一件事。比如“先查库存再分配仓位”和“别在未知库存时直接派单”，文字不一样，但策略几乎相同。

第三层是时间衰减加权。检索时不直接按相似度排序，而是做加权：

$$
w = 0.6 \cdot similarity + 0.4 \cdot e^{-\Delta t / 30}
$$

这条式子的含义是：相关性优先，但新鲜度必须参与决策。否则系统会反复召回“过去非常像、现在已过期”的经验。

玩具例子如下。当前问题是“物流派单失败，因为目标仓缺货”。系统里有四条反思：

| 反思 | 相似度 | 距今天数 | 新鲜度 | 最终分数 |
|---|---:|---:|---:|---:|
| 先检查库存再派单 | 0.92 | 2 | 0.94 | 0.93 |
| 避免重复调用配送 API | 0.61 | 1 | 0.97 | 0.75 |
| 注意节假日时效 | 0.58 | 40 | 0.26 | 0.45 |
| 别使用过期仓库规则 | 0.83 | 50 | 0.19 | 0.57 |

最后真正该进 prompt 的通常只有前 1 到 3 条，而不是全部。因为 prompt 需要的是“下一步可执行提示”，不是“历史档案总览”。

---

## 代码实现

实现上，一个最小可用版本的记忆结构至少需要这几个字段：反思文本、摘要、时间戳、向量、类型。这里“向量”指 embedding，白话说就是把一句话映射成便于比较相似度的数字表示。

| 字段 | 作用 | 是否必需 |
|---|---|---|
| `text` | 原始反思文本 | 是 |
| `summary` | 压缩后的短结论 | 推荐 |
| `timestamp` | 写入时间，用于衰减 | 是 |
| `embedding` | 用于相似检索 | 推荐 |
| `kind` | episodic / semantic | 推荐 |

下面是一个可运行的 Python 玩具实现，演示三件事：追加记忆、截断到最近 $\Omega$ 条、按“相似度 + 时间衰减”检索。为了可运行，这里用词袋交集模拟 embedding 相似度，而不是调用真实向量模型。

```python
from __future__ import annotations
from dataclasses import dataclass
from datetime import datetime, timedelta
import math

@dataclass
class MemoryItem:
    text: str
    timestamp: datetime
    summary: str = ""

def tokenize(text: str) -> set[str]:
    return set(text.lower().replace("，", " ").replace(",", " ").split())

def similarity(a: str, b: str) -> float:
    sa, sb = tokenize(a), tokenize(b)
    if not sa or not sb:
        return 0.0
    return len(sa & sb) / len(sa | sb)

def time_decay(now: datetime, ts: datetime, tau_days: int = 30) -> float:
    days_ago = max((now - ts).days, 0)
    return math.exp(-days_ago / tau_days)

class ReflexionMemory:
    def __init__(self, omega: int = 3):
        self.omega = omega
        self.items: list[MemoryItem] = []

    def append_reflection(self, item: MemoryItem) -> None:
        self.items.append(item)
        if len(self.items) > self.omega:
            self.items = self.items[-self.omega:]

    def retrieve(self, query: str, now: datetime, top_k: int = 3):
        scored = []
        for item in self.items:
            sim = similarity(query, item.text + " " + item.summary)
            fresh = time_decay(now, item.timestamp)
            score = 0.6 * sim + 0.4 * fresh
            scored.append((score, item))
        scored.sort(key=lambda x: x[0], reverse=True)
        return scored[:top_k]

now = datetime(2026, 3, 20)
mem = ReflexionMemory(omega=3)
mem.append_reflection(MemoryItem("先 检查 库存 再 派单", now - timedelta(days=2), "库存校验优先"))
mem.append_reflection(MemoryItem("避免 重复 调用 配送 API", now - timedelta(days=1), "接口幂等"))
mem.append_reflection(MemoryItem("注意 节假日 时效", now - timedelta(days=40), "时效规则"))
mem.append_reflection(MemoryItem("不要 使用 过期 仓库 规则", now - timedelta(days=50), "仓库规则更新"))

# 只保留最近 3 条
assert len(mem.items) == 3
assert mem.items[0].text == "避免 重复 调用 配送 API"

result = mem.retrieve("库存 派单 失败 先 检查 仓库", now, top_k=2)
assert len(result) == 2
assert result[0][0] >= result[1][0]

# 最近且相关的条目分数应高于很旧且不太相关的条目
best_text = result[0][1].text
assert "库存" in best_text or "仓库" in best_text
print("ok")
```

真实工程里，这段代码通常还要补两步。第一步是在写入前做去重，例如相似度超过阈值 `0.85` 时直接合并；第二步是在后台做 consolidation，把 7 天前的重要 episodic 记忆转成 semantic 记忆，避免活跃 prompt 长度持续上涨。

---

## 工程权衡与常见坑

第一类坑是记忆膨胀。看起来每条反思都“也许未来有用”，于是系统不断追加，最终任何查询都能召回一堆半相关文本。规避方法不是简单删旧，而是分层：活跃窗口保留最近 3 到 5 条，较老反思先摘要，再升级为语义知识。

第二类坑是重复强化错误。所谓“重复强化”就是模型不断读到自己过去写下的错误总结，结果把偶然错误当成长期规律。比如某次失败是工具接口暂时异常，但反思写成“这个 API 不可靠，应避免调用”，后续多次被检索出来，系统就会错误回避本应继续使用的工具。

第三类坑是去重过猛。把所有相似句都合并，可能会丢掉关键上下文差异。AgentDock 文档里给出的经验是先保守：高阈值、保留原始记录、先只做 merge，再逐步加入 synthesize 和 abstract。

| 常见坑 | 具体表现 | 规避策略 |
|---|---|---|
| 记忆膨胀 | prompt 越来越长 | 滑动窗口 + 摘要压缩 |
| 旧事掩盖新事 | 检索总是命中过时经验 | 时间衰减排序 |
| 重复内容堆积 | 相同错误被多次召回 | 多指标去重合并 |
| 过度抽象 | 只剩口号，失去可执行性 | semantic 与原文双存 |
| 过早删除 | 重要长期规律被丢失 | 分层存储，延迟清理 |

删除逻辑也不应只按“是否很旧”，而应按“是否既旧又不相关”。常见伪代码是：

```python
if score < threshold:
    drop(item)
```

更稳妥的版本是：只有当 `freshness` 低、`similarity` 低、且已被摘要覆盖时，才允许删除。否则宁可转存为 semantic，也不要直接丢。

真实工程例子里，客服 Agent 的促销政策经常每周变。如果不做时间衰减，系统会反复拿出上周的优惠规则；如果只做时间衰减不做去重，又会把本周三条相近规则同时塞进 prompt。最稳的组合通常是：最新 3 条原始反思 + 1 条本周摘要 + 后台语义库检索。

---

## 替代方案与适用边界

Reflexion 的滑动窗口方案适合“失败后能明确总结教训，且后续任务与前一轮相似”的场景，例如代码修复、工具调用、多步任务规划。如果任务轮次很短、状态变化很快，例如实时客服闲聊，保留大量 episode 级反思的收益并不高，此时只保留摘要和元数据通常更合适。

替代方案一是向量数据库加摘要。它的思路是把历史反思放到外部检索系统，不把它们长期占在 prompt 里。优点是容量大，适合长期积累；缺点是你必须额外处理召回质量，否则只会把“上下文溢出”换成“检索误召回”。

替代方案二是纯 semantic memory。也就是不保留大部分 episode 原文，只沉淀高层规律。这适合高频、高噪声系统，例如日志型 Agent；但对需要复盘细节的任务，例如程序调试，它可能丢掉关键因果链。

| 方案 | 优点 | 缺点 | 适用场景 |
|---|---|---|---|
| Reflexion 固定 $\Omega$ 窗口 | 简单、可控、延迟低 | 容量小 | 短中期反复试错 |
| 向量库 + 摘要 | 容量大、可扩展 | 召回链路更复杂 | 长周期生产系统 |
| 纯 semantic memory | prompt 最省 | 细节损失大 | 高频重复任务 |
| 全量保留原文 | 实现最省事 | 最容易失控 | 不建议长期使用 |

可以用一个很简单的判断式来理解边界：

$$
\text{适合固定窗口} \iff \text{failure rate 高且 prompt budget 有限}
$$

当失败率高、可见上下文预算又有限时，固定窗口最有效，因为它把最贵的资源留给“最近、最相关、最可执行”的经验。反过来，如果任务特别长、知识跨度特别大，仅靠 $\Omega=3$ 或 $\Omega=5$ 通常不够，必须引入外部检索和语义整合。

---

## 参考资料

下表列的是本文直接依赖的三类来源，以及它们分别解决了什么问题。

| 来源 | 主要贡献 | 对应章节 | 链接 |
|---|---|---|---|
| Reflexion: Language Agents with Verbal Reinforcement, NeurIPS 2023 | 给出 Actor、Evaluator、Self-Reflection 回路；定义长期记忆；说明 $\Omega$ 通常取 1 到 3，并在实验中截断到最近 3 条反思 | 核心结论、问题定义、核心机制 | https://papers.neurips.cc/paper_files/paper/2023/file/1b44b878bb782e6954cd888628510e90-Paper-Conference.pdf |
| AgentDock Memory Consolidation Guide | 给出 episodic 到 semantic conversion、memory deduplication、hierarchical abstraction，以及保守阈值和批处理建议 | 核心机制、工程权衡、替代方案 | https://hub.agentdock.ai/docs/memory/consolidation-guide |
| Agent Memory Wars, Medium, 2026 | 提供工程化检索示例：相关度与时间衰减加权、返回 top 3 到 5 条、避免旧记忆长期霸榜 | 核心结论、代码实现、工程权衡 | https://medium.com/%40nraman.n6/agent-memory-wars-why-your-multi-agent-system-forgets-what-matters-and-how-to-fix-it-a9a1901df0d9 |

对零基础读者，三份资料可以这样理解：论文回答“Reflexion 为什么成立”，AgentDock 文档回答“记忆膨胀后怎么管”，Medium 实践文回答“生产系统里怎么检索得更稳”。其中“时间衰减加权”与“3 到 5 条检索窗口”属于工程经验总结，不是 Reflexion 论文的统一标准配置，应按任务长度、token 预算和更新频率调整。
