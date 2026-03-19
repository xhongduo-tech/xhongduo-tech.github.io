## 核心结论

Agent 记忆系统的评测，不能只看“有没有检索到”，必须看“检索到了以后，能不能在后续任务里真正改变结果”。这也是端到端评测框架的核心：把记忆模块当成一个独立子系统，同时把它放回完整的 Agent 执行闭环里检查。

更具体地说，一个可用的评测框架至少要覆盖四个维度：

| 维度 | 白话解释 | 典型指标 | 关注的问题 |
| --- | --- | --- | --- |
| 检索精度 | 该拿出来的记忆，能不能被拿出来 | Recall@k、MRR | 找不找得到 |
| 信息保持率 | 多轮对话以后，关键信息还记不记得住 | 长程事实命中率 | 会不会越聊越忘 |
| 存储效率 | 花了多少 token 或字节存这些记忆 | Token/条目、压缩比 | 成本是否可控 |
| 任务贡献度 | 开启记忆后，任务成功率到底涨没涨 | $\Delta S=S_{\text{on}}-S_{\text{off}}$ | 记忆是否真的有用 |

这四个维度分别回答四个不同问题：能否取回、能否保留、代价多大、是否值得。缺少其中任何一个，评测都会失真。只测 Recall，可能出现“检索很好但任务仍失败”；只测成功率，又无法定位问题出在检索、写入还是更新策略。

对零基础读者，可以先记住一个最直观的结论：记忆系统的质量，决定了 Agent 在跨轮次任务里的稳定性。比如一个旅行规划 Agent，第一轮已经知道“用户要 3 天行程、同行有父母、预算中等”，如果后面推荐了深夜转机和高强度徒步路线，问题通常不在生成模型不会说话，而在记忆链路没有把早期约束正确带到后续决策中。

---

## 问题定义与边界

这里讨论的“Agent 记忆系统”，不是单纯把更长的聊天记录塞进上下文窗口。上下文窗口就是模型当前一次能看到的文本范围；而记忆系统是把历史信息写入外部存储，在未来按需检索并影响行为的机制。

因此，本文的评测边界是：

1. Agent 会在多轮交互中持续产生信息。
2. 这些信息会被写入某种记忆存储。
3. 后续任务会触发检索。
4. 检索结果会改变 Agent 的决策、工具调用或最终答案。

这意味着，单轮问答、静态知识库 QA、甚至一次性长上下文推理，都不等价于记忆评测。它们最多测“模型当下能不能读懂材料”，而不是“系统能不能在未来重新用上历史材料”。

可以用一个玩具例子说明边界。

玩具例子：学生档案助手  
第一轮用户说：“我叫小林，偏好 Python，准备投后端实习。”  
第五轮用户问：“你建议我优先补哪类项目？”  
如果系统能回答“优先补 Python 后端项目，比如 Web API、数据库和部署”，说明它把前面的偏好保留下来了。  
如果它只回答“建议做你感兴趣的项目”，那不是语言能力差，而是记忆没有参与后续决策。

真实工程例子：多人旅行规划  
一个 group travel planner 在早期轮次收集每个人的预算、饮食禁忌、年龄结构、出发地和时间窗口，后面再做酒店、交通、景点和日程安排。这里的约束分散在不同轮次、不同人、不同子任务里。如果系统没有稳定记忆，就会出现局部正确、全局冲突：酒店便宜但离集合点太远，航班时间合适但老人无法承受红眼航班，餐厅评分高但不满足忌口要求。

从场景上看，端到端评测通常需要覆盖多类任务，因为不同任务依赖的记忆类型不同：

| 场景 | 依赖的记忆类型 | 评估重点 |
| --- | --- | --- |
| 网页购物 | 即时偏好、商品约束 | 检索是否命中用户要求 |
| 渐进搜索 | 中间发现、查询轨迹 | 历史信息能否指导下一步搜索 |
| 旅行规划 | 多角色事实、长期约束 | 长程一致性与冲突处理 |
| 数学推理 | 中间结论、已证步骤 | 推理链是否能复用 |
| 物理推理 | 条件、公式选择、单位约束 | 历史前提是否持续生效 |

所以本文的边界很明确：评测的是 Memory→Agent→Environment 这个循环，而不是模型单次回答能力。

---

## 核心机制与推导

端到端评测的难点在于，记忆系统不是一个单一分数能描述的对象。比较稳妥的做法，是把记忆拆成“写入、检索、保留、贡献”四条链路，再分别定义指标。

先看检索精度。

设相关记忆总数为 $N_{\text{rel}}$，成功检索出的相关记忆数为 $N_{\text{hit}}$)，则总体召回率为：

$$
R=\frac{N_{\text{hit}}}{N_{\text{rel}}}
$$

Recall@k 是“前 $k$ 条结果里找回了多少相关项”的比例。MRR 是 Mean Reciprocal Rank，白话说就是“第一个正确结果排得靠不靠前”的平均分，定义为：

$$
\text{MRR}=\frac{1}{Q}\sum_{i=1}^{Q}\frac{1}{\text{rank}_i}
$$

其中 $Q$ 是查询数，$\text{rank}_i$ 是第 $i$ 个查询中首个正确结果的位置。首个正确结果越靠前，MRR 越大。

再看信息保持率。它衡量的是：经过很多轮写入、更新、覆盖以后，系统还能不能回忆出仍然有效的事实。可写成：

$$
A_{\text{long}}=\frac{N_{\text{fact-correct}}}{N_{\text{fact-query}}}
$$

这里 $N_{\text{fact-query}}$ 是长程事实查询总数，$N_{\text{fact-correct}}$ 是回答正确的数量。这个指标常常比 Recall 更难，因为它不只考察检索，还考察记忆是否被错误覆盖、是否过期、是否在压缩时丢失关键条件。

存储效率衡量成本。一个简单定义是：

$$
E=\frac{T_{\text{store}}}{N_{\text{retrievable}}}
$$

其中 $T_{\text{store}}$ 是用于存储的 token 数，$N_{\text{retrievable}}$ 是可被有效检索的记忆条目数。$E$ 越低，表示单位有效记忆的成本越低。这个指标的意义很直接：如果一个系统能记住 100 条事实，但要消耗极大上下文和索引成本，它未必适合真实生产环境。

最后是任务贡献度：

$$
\Delta S=S_{\text{memory on}}-S_{\text{memory off}}
$$

这里 $S$ 可以是任务成功率、平均得分、完成步数下降比例等。这个指标最重要，因为它直接回答：记忆系统有没有把“记住”转化成“做成”。

可以把四个指标的作用概括如下：

| 指标 | 测什么 | 常见失败信号 |
| --- | --- | --- |
| Recall@k | 候选记忆是否被找回 | 相关事实根本不在前 k 项 |
| MRR | 正确记忆排位是否足够靠前 | 正确条目被噪声压到后面 |
| 长程事实命中率 | 多轮后是否仍记得关键信息 | 旧约束被遗忘或覆盖 |
| $\Delta S$ | 记忆是否提升任务结果 | 检索有命中但结果无提升 |

一个简单推导是：只有当“命中率高”“排位靠前”“事实未失真”“结果被使用”同时成立时，$\Delta S$ 才会显著为正。用链式视角表示，就是：

$$
P(\text{任务成功})
\approx
P(\text{写入正确})
\cdot
P(\text{检索命中})
\cdot
P(\text{排序靠前})
\cdot
P(\text{注入后被采纳})
$$

这不是严格概率模型，但很适合作为工程分析框架。任何一项接近 0，最终任务贡献都会很弱。

---

## 代码实现

实现一个最小可用的端到端评测框架，建议拆成三段：记忆写入与检索、长程事实检查、任务贡献对照实验。

下面先用一个可运行的玩具评测器说明核心逻辑。它不依赖外部服务，重点是把指标链路跑通。

```python
from math import isclose

memories = [
    {"id": "m1", "text": "用户要3天行程，同行有父母", "score": 0.95, "relevant": True},
    {"id": "m2", "text": "用户预算中等，偏好高铁", "score": 0.89, "relevant": True},
    {"id": "m3", "text": "用户喜欢夜生活", "score": 0.40, "relevant": False},
    {"id": "m4", "text": "推荐红眼航班更便宜", "score": 0.35, "relevant": False},
]

def recall_at_k(items, k):
    topk = items[:k]
    total_relevant = sum(1 for x in items if x["relevant"])
    hit = sum(1 for x in topk if x["relevant"])
    return hit / total_relevant if total_relevant else 0.0

def mrr(items):
    for idx, item in enumerate(items, start=1):
        if item["relevant"]:
            return 1.0 / idx
    return 0.0

def long_term_accuracy(preds, labels):
    correct = sum(int(p == y) for p, y in zip(preds, labels))
    return correct / len(labels) if labels else 0.0

def efficiency(total_tokens, retrievable_items):
    return total_tokens / retrievable_items if retrievable_items else float("inf")

def delta_success(with_memory, without_memory):
    return with_memory - without_memory

# 检索结果按 score 从高到低排列
ranked = sorted(memories, key=lambda x: x["score"], reverse=True)

r2 = recall_at_k(ranked, 2)
mrr_score = mrr(ranked)
long_acc = long_term_accuracy(
    preds=["3天", "有父母", "预算中等", "高铁"],
    labels=["3天", "有父母", "预算中等", "高铁"]
)
e = efficiency(total_tokens=120, retrievable_items=4)
delta = delta_success(with_memory=0.8, without_memory=0.5)

assert isclose(r2, 1.0)
assert isclose(mrr_score, 1.0)
assert isclose(long_acc, 1.0)
assert isclose(e, 30.0)
assert isclose(delta, 0.3)

print("Recall@2 =", r2)
print("MRR =", mrr_score)
print("Long-term Accuracy =", long_acc)
print("Efficiency =", e)
print("Delta Success =", delta)
```

这个玩具例子表达了四件事：

1. 相关记忆必须真的排进前列。
2. 第一个正确记忆最好尽早出现。
3. 长对话后，关键事实仍要答对。
4. 最终必须比较开关记忆前后的任务结果。

接着看更接近真实工程的最小骨架。假设我们基于 `datasets` 加载多轮任务数据，然后对每条样本同时跑 `memory-on` 和 `memory-off` 两个版本。

```python
from datasets import load_dataset

def build_memory(history):
    store = []
    for turn in history:
        text = turn.get("text", "")
        if "预算" in text or "偏好" in text or "约束" in text:
            store.append(text)
    return store

def retrieve(store, query, topk=5):
    # 这里只是占位实现；真实系统应替换为 embedding + reranker
    scored = []
    for item in store:
        overlap = sum(1 for token in ["预算", "偏好", "父母", "3天", "高铁"] if token in item and token in query)
        scored.append((overlap, item))
    scored.sort(reverse=True)
    return [x[1] for x in scored[:topk] if x[0] > 0]

def run_agent(task, retrieved_memories):
    target = task.get("target_constraints", [])
    joined = " ".join(retrieved_memories)
    hit = sum(1 for c in target if c in joined)
    return 1 if hit >= max(1, len(target) // 2) else 0

# 示例配置名以数据集实际提供为准
ds = load_dataset("ZexueHe/memoryarena", "group_travel_planner", split="test")

success_on = 0
success_off = 0

for task in ds.select(range(min(20, len(ds)))):
    history = task.get("conversation", [])
    query = task.get("final_query", "")
    store = build_memory(history)
    memories = retrieve(store, query, topk=5)

    success_on += run_agent(task, memories)
    success_off += run_agent(task, [])

delta_s = success_on / 20 - success_off / 20
print("Delta Success =", delta_s)
```

这段代码对应真实评测框架里的三个子环节：

| 代码步骤 | 对应评测能力 | 应记录的数据 |
| --- | --- | --- |
| `build_memory` | 写入策略 | 写入条数、平均长度、压缩率 |
| `retrieve` | 检索与排序 | Recall@k、MRR、阈值命中率 |
| `run_agent` | 任务贡献 | success rate、得分变化、步数变化 |

真实工程里，建议把日志设计成“每轮一条记录”。至少保存：查询、写入内容、检索候选、最终注入内容、模型输出、任务得分。这样当 $\Delta S$ 很低时，才能准确定位是写入问题、排序问题，还是 Agent 根本没用上记忆。

---

## 工程权衡与常见坑

端到端评测最大的价值，不是给出一个漂亮均分，而是暴露“记忆系统为什么失效”。常见问题通常集中在三类：记错、过时、太多。

| 常见坑 | 典型症状 | 推荐对策 |
| --- | --- | --- |
| Hallucination Memory | 检索出表面相似但实际错误的条目 | 加 reranker、设相似度阈值 |
| Stale Memory | 系统反复使用过时事实 | 时间衰减、版本号、显式更新 |
| Context Overload | 注入记忆过多，模型忽略关键项 | top-K 限流、摘要压缩、字段化注入 |

先看 hallucination。这里的意思不是模型凭空捏造，而是记忆层把“像对的但其实不对的”内容送进了上下文。比如用户说“带父母出行，需要白天到达”，系统却因为“便宜”检索到一条旧记忆“优先考虑红眼航班”，后续规划就会被误导。解决思路通常有两步：先用向量检索召回，再用更精细的 reranker 重排；同时设置阈值，低于阈值的候选直接丢弃。

再看 stale memory，也就是陈旧记忆。记忆不是越久越可靠。真实系统里，票价、库存、政策、联系人状态都会变化。比如今天是 2026-03-19，而一条价格记忆来自 2025-01-05，如果系统不做时间衰减，就会把历史价格当当前价格。工程上要么给每条记忆带时间戳和 TTL，要么把“易过时信息”设计成必须二次验证后才能使用。

第三类是 context overload。把检索到的 50 条记忆一次性塞进提示词，看上去像“信息更全”，实际常常更差。因为模型会被大量低价值文本稀释注意力。更稳妥的做法是只注入最相关的 5 到 10 条，再把其余信息压缩成摘要或结构化字段。

这里有一个现实权衡：压缩会降低 token 成本，但也可能损失细节；保留原文更安全，但成本更高。经验上，事实型记忆适合结构化存储，推理轨迹适合摘要存储，工具操作日志适合事件流存储。不同类型不要混成一种格式，否则检索质量和后续利用率都会下降。

还有一个常见误区是把“有提升”误当成“系统稳定”。如果某次实验里 $\Delta S$ 从 0.40 提到 0.55，看上去涨了 15 个点，但如果方差很大、不同场景表现差异极大，那这个系统仍然不稳定。端到端评测最好按任务类型、轮次长度、记忆条数分桶统计，而不是只报单一平均值。

---

## 替代方案与适用边界

端到端评测框架不是唯一方案，它更像总控台。不同记忆架构在不同任务上适用边界不同。

| 方案 | 核心思路 | 适用场景 | 工程投入 | 主要指标 |
| --- | --- | --- | --- | --- |
| MemoryArena 风格端到端评测 | 在完整 Agent 循环里测记忆 | 多会话、多子任务 Agent | 中到高 | Recall、长程准确率、$\Delta S$ |
| AWM | 把工作流模式写成可复用记忆 | Web navigation、多步网页操作 | 中 | 任务成功率、步骤效率 |
| Memory-first 架构 | 混合检索、schema、知识图谱 | 生产级长期记忆系统 | 高 | 召回率、时序一致性、成本 |

如果你刚开始做一个购物助手或网页自动化代理，AWM 这类 workflow memory 很有吸引力。它关注的是“以前成功过的操作流程能不能迁移到新任务”，适合网页点击、表单填写、站点导航这类结构接近的任务。它不一定擅长复杂个人事实记忆，但对重复工作流很有效。

如果你在做的是长期陪伴、企业知识助手或跨会话规划系统，memory-first 架构更常见。它会把记忆拆成 schema、知识图谱、向量索引、时间信息等多层结构。好处是召回和更新更可控，坏处是工程复杂度明显上升。

本文提出的四维端到端评测框架，最适合以下边界：

1. 任务不是一次性完成，而是跨轮次展开。
2. 历史信息会显著影响未来决策。
3. 需要比较不同记忆实现方案。
4. 需要把“效果”和“成本”同时纳入评估。

它不太适合的边界也要说清楚：如果任务几乎全是单轮问答，或者所有关键信息都能直接放进当前上下文，那单独建设复杂记忆系统的收益可能不高。此时做简单的 prompt engineering 或长上下文基线，反而更划算。

所以，替代方案不是“谁更先进”，而是谁更匹配任务结构。评测框架的作用，是把这个选择变成可量化决策，而不是凭感觉下注。

---

## 参考资料

- MemoryAgentBench：提出面向 Agent 记忆能力的统一评测协议，强调准确检索、测试期学习、长程理解与选择性忘记等维度。  
- MemoryArena：构造多会话、长轨迹的 Memory-Agent-Environment 闭环任务，用于测量记忆对真实任务完成的影响。  
- Hugging Face `ZexueHe/memoryarena`：提供数据集加载方式与不同任务切片。  
- Cortex 关于 LLM memory recall 的实践文章：给出 Recall、MRR、长程准确率、存储效能与任务差分等常用指标。  
- OrbitalAI memory management 指南：总结记忆幻觉、陈旧事实、上下文过载等常见工程问题及缓解策略。  
- AWM 相关资料：展示 workflow memory 在 WebArena、Mind2Web 一类网页任务中的适用性。
