## 核心结论

召回层的效果评估，本质上是在回答一个工程问题：**这个检索模块有没有把“后续能用”的候选，按可接受的成本送到排序层或生成层手里。**

只看 `Recall@K` 不够，因为它只回答“有没有尽量找全”，不回答另外两个关键问题：

| 维度 | 代表指标 | 要解决的问题 |
| --- | --- | --- |
| 命中能力 | Recall@K、HitRate@K | 相关内容有没有被找到 |
| 排序友好度 | MRR、首个相关位置、排序后上屏贡献率 | 找到的内容是不是容易被后续模型利用 |
| 候选质量 | 覆盖率、平均流行度、新颖性 | 候选是不是过度集中在热门内容 |
| 系统代价 | 每请求候选成本、延迟、reranker 计算量 | 找这些候选值不值得 |

对推荐系统、搜索系统、RAG 都一样：召回层不是独立 KPI，它是下游链路的供给层。高召回但低可排序性，等于把大量“脏候选”交给排序器，既浪费算力，也会压低最终上屏质量。

玩具例子可以直接说明这个问题。假设某个请求有 5 个真正相关的文档，系统返回前 4 个候选时命中了其中 3 个，那么 `Recall@4 = 3/5 = 0.6`。如果第一个相关结果排在第 2 位，那么 `HitRate@4 = 1`，`RR = 1/2 = 0.5`。这说明系统“找到了不少”，但“排得还不够靠前”。对排序层来说，这两种状态的价值完全不同。

---

## 问题定义与边界

先定义术语。**召回层**，就是先从大规模候选池里快速筛出一小批可能相关内容的模块。白话讲，它像第一轮粗筛，不负责最终决策，只负责把可能有用的东西先捞出来。

本文讨论的“召回层效果评估”，边界很明确：

1. 只评估召回阶段输出的候选集合。
2. 关注召回是否为后续排序或生成提供可用输入。
3. 不展开讨论排序模型内部结构，也不讨论 LLM 生成答案的语言质量。
4. 可以把“排序后上屏贡献率”作为召回价值的外部验证，因为召回的目标本来就是服务下游。

常见基础指标定义如下。

设某次请求的标注相关集合为 $R$，召回返回前 $K$ 个候选为 $T_K$，则：

$$
Recall@K = \frac{|T_K \cap R|}{|R|}
$$

它表示“全部相关内容里，前 $K$ 个候选找回了多少”。

设一组请求总数为 $N$，若第 $i$ 个请求在前 $K$ 个结果中至少命中 1 个相关项，则记为 1，否则为 0，则：

$$
HitRate@K = \frac{1}{N}\sum_{i=1}^{N}\mathbf{1}(|T_K^{(i)} \cap R^{(i)}| > 0)
$$

它表示“有多少请求至少没完全扑空”。

若某次请求第一个相关结果出现在位置 $rank_i$，则倒数排名 `Reciprocal Rank` 为：

$$
RR_i = \frac{1}{rank_i}
$$

所有请求平均后得到：

$$
MRR = \frac{1}{N}\sum_{i=1}^{N} RR_i
$$

它表示“第一个有用结果出现得够不够早”。

覆盖率也很重要。设一段时间内被召回过的唯一 item 数为 $U$，全量候选池大小为 $C$，则：

$$
Coverage = \frac{|U|}{|C|}
$$

它表示召回是不是总在少数热门 item 上打转。

成本可以粗略估算为：

$$
Cost\ per\ request \approx K \times \frac{tokens\_per\_candidate}{1000} \times price_{1k}
$$

这个公式的意义不是精确计费，而是提醒你：候选数 $K$ 往上加时，reranker 和 LLM 的成本通常近似线性放大。

---

## 核心机制与推导

召回层评估要同时回答三个问题。

第一，**能不能命中**。这由 `Recall@K` 和 `HitRate@K` 回答。`Recall@K` 看找全程度，`HitRate@K` 看是否至少找到一个。前者适合评估“漏召回”，后者适合评估“是否完全失手”。

第二，**命中的东西是否容易被下游利用**。这由 `MRR` 和“排序后上屏贡献率”回答。排序后上屏贡献率，指的是最终展示或喂给生成模型的结果里，有多少来自某个召回通道。白话讲，它衡量“你找回来的候选，最后到底有没有真的用上”。

第三，**这种命中是否划算**。这由覆盖率、平均流行度、新颖性、每请求成本共同回答。平均流行度高，通常说明系统偏热门内容；新颖性高，通常说明系统能给出更多长尾结果。二者没有绝对好坏，要看业务目标。

下面给一个玩具例子。

假设请求 `q1` 的真实相关集合是 `{A, B, C, D, E}`，召回前 4 个是 `[X, B, Y, C]`。

| 位置 | 候选 | 是否相关 |
| --- | --- | --- |
| 1 | X | 否 |
| 2 | B | 是 |
| 3 | Y | 否 |
| 4 | C | 是 |

则：

- 命中相关数为 2
- 总相关数为 5
- 所以 $Recall@4 = 2/5 = 0.4$
- 因为前 4 个里至少有 1 个相关，所以 $HitRate@4 = 1$
- 第一个相关在第 2 位，所以 $RR = 1/2 = 0.5$

这个结果代表：系统没有完全失败，但也没有把足够多的相关项拉进来，而且首个相关出现得偏后。若排序层只能处理前 3 个候选，那么实际上只有 `B` 能参与后续阶段，`C` 的价值就未必兑现。

真实工程里，更常见的是两阶段流程：

`候选库 -> 召回 Top40 -> reranker 重排 Top12 -> 最终上屏 Top3 或送入 LLM`

这个链路里，召回层的 `K` 不是越大越好。因为 `K` 上去以后：

1. `Recall@K` 往往继续提升，但边际收益会下降。
2. reranker 的计算量几乎按候选数线性增加。
3. 送入 LLM 的上下文长度会变长，成本与延迟上升。
4. 如果新增候选大多是“低质但勉强相关”，排序器会更难分辨，最终上屏不一定更好。

所以实际推导不是“把 recall 做到最高”，而是找一个 Pareto 边界：在同等成本下尽量提升下游收益，或者在同等收益下尽量降低成本。

---

## 代码实现

下面给一个可运行的 Python 示例，演示如何在离线评估中同时计算命中、质量、成本和排序后贡献。

```python
from collections import defaultdict

queries = [
    {
        "qid": "q1",
        "relevant": {"A", "B", "C", "D", "E"},
        "retrieved": ["X", "B", "Y", "C", "Z"],
        "served_after_rerank": ["B", "X"],
    },
    {
        "qid": "q2",
        "relevant": {"M", "N"},
        "retrieved": ["P", "Q", "N", "R", "S"],
        "served_after_rerank": ["N", "Q"],
    },
]

popularity = {
    "A": 100, "B": 80, "C": 60, "D": 40, "E": 20,
    "M": 50, "N": 30, "X": 200, "Y": 150, "Z": 120,
    "P": 90, "Q": 70, "R": 65, "S": 55,
}

catalog_size = 1000
K = 4
tokens_per_candidate = 120
price_per_1k_tokens = 0.002

def recall_at_k(relevant, retrieved, k):
    topk = retrieved[:k]
    return len(set(topk) & set(relevant)) / len(relevant) if relevant else 0.0

def hitrate_at_k(relevant, retrieved, k):
    topk = retrieved[:k]
    return 1.0 if set(topk) & set(relevant) else 0.0

def reciprocal_rank(relevant, retrieved, k):
    for idx, item in enumerate(retrieved[:k], start=1):
        if item in relevant:
            return 1.0 / idx
    return 0.0

def avg_popularity(retrieved, k, popularity_map):
    topk = retrieved[:k]
    return sum(popularity_map.get(x, 0) for x in topk) / len(topk)

def candidate_cost(k, tokens_per_candidate, price_per_1k_tokens):
    return k * (tokens_per_candidate / 1000.0) * price_per_1k_tokens

recalls, hits, rrs, pops = [], [], [], []
unique_retrieved = set()
retrieval_contrib = 0
served_total = 0

for q in queries:
    recalls.append(recall_at_k(q["relevant"], q["retrieved"], K))
    hits.append(hitrate_at_k(q["relevant"], q["retrieved"], K))
    rrs.append(reciprocal_rank(q["relevant"], q["retrieved"], K))
    pops.append(avg_popularity(q["retrieved"], K, popularity))
    unique_retrieved.update(q["retrieved"][:K])

    retrieved_topk = set(q["retrieved"][:K])
    served_total += len(q["served_after_rerank"])
    retrieval_contrib += sum(1 for x in q["served_after_rerank"] if x in retrieved_topk and x in q["relevant"])

metrics = {
    "Recall@K": sum(recalls) / len(recalls),
    "HitRate@K": sum(hits) / len(hits),
    "MRR@K": sum(rrs) / len(rrs),
    "Coverage": len(unique_retrieved) / catalog_size,
    "AvgPopularity": sum(pops) / len(pops),
    "CostPerRequest": candidate_cost(K, tokens_per_candidate, price_per_1k_tokens),
    "ServedContribution": retrieval_contrib / served_total if served_total else 0.0,
}

assert round(metrics["Recall@K"], 2) == 0.45
assert round(metrics["HitRate@K"], 2) == 1.00
assert round(metrics["MRR@K"], 2) == 0.42
assert metrics["CostPerRequest"] > 0

print(metrics)
```

这个例子里，`ServedContribution` 的含义是：最终排序后真正上屏的结果中，有多少既来自当前召回集合，又确实相关。它不是标准学术指标，但在工程里非常有用，因为它直接连接召回与最终收益。

常见参数可以这样管理：

| 参数 | 含义 | 典型作用 |
| --- | --- | --- |
| `K` | 每次召回候选数 | 控制 recall 与成本的平衡 |
| `tokens_per_candidate` | 单候选平均文本长度 | 估算 LLM 或 reranker 成本 |
| `price_per_1k_tokens` | 每千 token 价格 | 估算单请求费用 |
| `catalog_size` | 全量候选池大小 | 计算覆盖率 |
| `popularity_source` | 流行度统计口径 | 判断热门偏置 |
| `served_after_rerank` | 排序后实际使用结果 | 计算上屏贡献率 |

真实工程例子：一个知识库问答服务，向量召回先取 `Top40`，BM25 再补 `Top20`，合并去重后交给 cross-encoder 重排，最后只把前 `Top8` 文档切片送入 LLM。此时离线评估不应只看“Top40 是否命中答案文档”，还要看：

1. 合并召回后 `Recall@40` 提升了多少。
2. 重排后前 `Top8` 里真正相关文档占比多少。
3. 多加的候选是否显著提高了最终答案质量。
4. token 成本和延迟是否还能接受。

如果 `Recall@40` 从 0.82 提升到 0.87，但排序后上屏贡献只增加 1%，而 reranker 成本增加 60%，这通常不是好优化。

---

## 工程权衡与常见坑

最常见的误区，是把召回层当成单一命中问题，而不是供给质量问题。

| 常见坑 | 现象 | 规避方式 |
| --- | --- | --- |
| 只追 `Recall@K` | 候选越来越多，排序和生成成本暴涨 | 同时看 `MRR`、上屏贡献率、单请求成本 |
| 忽视覆盖率 | 总是召回热门内容，长尾内容难曝光 | 加入覆盖率、新颖性、流行度约束 |
| 只看离线命中 | 离线很好，线上上屏收益很差 | 增加 reranker 后指标和线上实验 |
| `K` 固定过大 | 高峰期延迟抖动，资源浪费 | 做分层 K、early stop、token cap |
| 标注集太小 | 指标看起来稳定，实际偏差很大 | 补充难例、长尾 query、冷启动样本 |

这里最值得强调的是“高 recall 低贡献”。这类问题在推荐系统和 RAG 都非常常见。

例如某召回通道新增了很多弱相关候选，导致：

- `Recall@100` 提升明显；
- 但这些候选在 reranker 中几乎排不到前面；
- 最终上屏点击、转化或答案正确率没有提升；
- 反而因为候选变多，cross-encoder 和 LLM 成本显著上升。

这说明新增候选虽然“相关性擦边”，但“可排序性差”。白话讲，就是它们理论上不算完全错，但对最终决策帮助不大。

成本增长也必须显式建模。若每个候选平均 150 tokens，价格为每千 tokens 0.002 美元，那么：

- `K=20` 时，成本约为 $20 \times 150 / 1000 \times 0.002 = 0.006$
- `K=80` 时，成本约为 $0.024$

如果还要叠加 reranker 前向计算和多路召回合并，这个差距会继续放大。对高 QPS 系统，这不是小数点问题，而是资源预算问题。

---

## 替代方案与适用边界

召回评估不一定总是以 `Recall + HitRate + MRR + Cost` 为中心。不同系统可以换主指标，但前提是你清楚业务目标。

| 方案 | 适用场景 | 优点 | 边界 |
| --- | --- | --- | --- |
| `Recall@K + HitRate + Cost` | 冷启动、高吞吐、先保证别漏掉 | 简单直接，便于快速调参 | 不足以判断下游收益 |
| 加 `MRR` 和首个相关位置 | 排序层较强，关心前排质量 | 能反映候选可排序性 | 仍未直接覆盖最终业务指标 |
| 加上屏贡献率 | 两阶段架构成熟 | 能直接衡量召回对下游的增量 | 依赖排序日志和曝光日志 |
| 用 `NDCG` 或排序损失反馈 | 排序精度优先的系统 | 更贴近最终展示质量 | 对召回漏检不够敏感 |
| 强化学习或策略学习选候选 | 候选选择成本高、反馈闭环强 | 可直接优化长期收益 | 工程复杂度高，调试成本大 |

什么时候优先用哪套指标，可以这样判断：

1. 如果系统还在早期，相关内容经常根本找不到，那么先把 `Recall@K`、`HitRate@K`、成本打稳。
2. 如果系统已经能稳定命中，但排序层吃不消，就把重点转到 `MRR`、首个相关位置、上屏贡献率。
3. 如果你已经有稳定的排序反馈和曝光反馈，那么单独优化召回 recall 的价值会下降，此时更适合把召回评估绑定到 `NDCG`、点击收益或答案正确率。
4. 如果业务强依赖长尾内容，例如文档问答、商品搜索、内容分发，就必须长期看覆盖率与流行度偏置，避免系统越来越“只会推热门”。

换句话说，`Recall@K` 是必要指标，但很少是充分指标。召回层最终不是为了“找回来”，而是为了“找回来且值得继续处理”。

---

## 参考资料

- [PremAI, RAG Evaluation Metrics & Frameworks Testing, 2026](https://blog.premai.io/rag-evaluation-metrics-frameworks-testing-2026/?utm_source=openai)：总结 RAG 检索、排序、生成多个层次的评估指标。
- [LayerLens, RAG Evaluation Framework for Production AI Systems](https://layerlens.ai/blog/rag-evaluation-framework-for-production-ai-systems?utm_source=openai)：强调生产环境中检索与 reranker、LLM 的联动评估。
- [Pinecone, Offline Evaluation for Search and Retrieval](https://www.pinecone.io/learn/offline-evaluation/?utm_source=openai)：给出 Recall、HitRate、MRR 等基础定义与离线评估方法。
- [TensorOpt, Measuring Search Quality](https://tensoropt.ai/blog/measuring-search-quality?utm_source=openai)：讨论搜索质量、成本、流行度偏置等工程问题。
- [Weaviate Retrieval Evaluation Metrics](https://weaviate.io/)：可扩展阅读，适合理解向量检索与混合检索的指标口径。
- [Practical RAG Evaluation 相关论文与工业实践](https://arxiv.org/)：适合继续追踪成本、延迟、答案质量联合评估方法。
