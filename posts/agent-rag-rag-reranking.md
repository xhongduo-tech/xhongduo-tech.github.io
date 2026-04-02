## 核心结论

RAG 的重排序，本质上是“先多找，再精排”的二阶段检索。RAG 是“检索增强生成”，白话说就是先去知识库找资料，再把资料喂给大模型回答。重排序不是替代检索，而是修正“初检相关但不够准”的问题。

第一阶段通常用向量检索或 BM25 做粗排。向量检索是“按语义相似度找文本”，BM25 是“按关键词匹配强度打分”。它们的共同优点是快，适合在大规模文档库里先召回几十到几百条候选；共同缺点是 top-1 往往不够可靠，尤其遇到编号、术语、产品名、流程步骤这类精确信息时，语义相似不等于答案正确。

第二阶段用 reranker 做细排。reranker 是“重新给候选文档打分的模型”，常见实现是 cross-encoder。cross-encoder 是“把 query 和文档拼在一起联合理解”的模型，因此比只看各自向量相似度的 bi-encoder 更擅长判断“这段文字是不是在回答这个问题”。代价是慢，所以只能用于少量候选，而不能全库扫描。

这套设计成立的原因很直接：第一阶段解决召回，第二阶段解决精度，最后生成阶段只消费被确认过的少量上下文。对真实系统而言，这比“只做一次向量检索然后直接喂给 LLM”稳定得多。

| 阶段 | 模型 | 作用 |
| --- | --- | --- |
| 第一轮检索 | 向量/BM25 | 召回 >70 条候选 |
| 第二轮 rerank | cross-encoder | 评分 + 排序，取出 top 5 |
| 生成 | LLM | 仅消费已确认高相关的上下文 |

一个新手版场景很典型。客户问：“发票编号 6832 的退款步骤是什么？”向量检索可能找到很多“退款”“发票”“售后”的相似段落，但不一定把真正包含“6832”的那条排在最前。reranker 看到 query 与文档联合内容后，会更重视“6832”和“退款步骤”同时出现的证据，把那条真正可回答的问题文档推到前面。最终进入 LLM 上下文的，不是“看起来像答案”的几段，而是“真的能回答”的那一段。

---

## 问题定义与边界

要理解重排序，先要明确它解决的不是“检索有没有结果”，而是“进入生成阶段的上下文是否足够可信”。

纯向量检索容易出现一种常见错误：语义相近，但任务不对。比如用户问“保单 #B-78 的赔付流程”，系统却返回“保单通用说明”“赔付政策概览”“理赔材料清单”。这些文本都和“保单”“赔付”相关，但没有回答“B-78 的具体流程”。LLM 再强，也只能基于拿到的上下文生成，于是会产出一段看起来流畅、实则偏题的“标准说明”。

重排序的边界也必须说清。它只能在“候选集合里已经有正确答案”的前提下发挥作用。如果第一阶段没有把真正相关文档召回出来，reranker 没有办法从不存在的候选里“创造答案”。所以系统设计上，必须先保证召回，再谈重排。

第二个边界是规模。cross-encoder 需要对 query 和每条候选逐一联合推理，复杂度明显高于向量相似度计算。因此它通常只用于 top-N 候选，常见范围是 20 到 100，极少超过 200。超过这个范围，延迟会迅速恶化，本来响应很快的问答系统会被细排拖慢。

下面这个表可以把故障模式看得更清楚：

| 故障模式 | 描述 | 边界 |
| --- | --- | --- |
| 召回不足 | 向量漏 keywords/编号 | 引入 hybrid 搜索（BM25+向量） |
| 排序失真 | top-1 文档与意图不符 | 只 rerank top-100，避免整体推理延迟 |
| 上下文噪声 | 过多无关 chunk 破坏 LLM recall | 先 rerank 再截取 top-K |

“上下文噪声”是另一个必须强调的问题。LLM 的上下文窗口不是无限有效的。即使 token 放得下，相关信息被一堆边缘片段包围时，模型也可能抓错重点。这也是为什么 RAG 系统不能简单理解成“多塞一点总没错”。很多时候，少而准的 5 段，比杂而多的 30 段更可靠。

所以重排序的作用边界可以总结成一句话：它不是全能检索器，而是“把已召回的候选压缩成高置信上下文”的一道质量闸门。

---

## 核心机制与推导

RAG 重排序最常见的机制，是把多个阶段的信号融合成最终排序分数。一个常见写法是：

$$
S_{final} = w_{vec}\cdot norm(S_{vec}) + w_{lex}\cdot norm(S_{lex}) + w_{rerank}\cdot norm(S_{rerank})
$$

这里的 $S_{vec}$ 是向量检索分数，$S_{lex}$ 是词法检索分数，通常来自 BM25，$S_{rerank}$ 是重排序模型分数。`norm` 是归一化，白话说就是“把不同量纲的分数拉到可比较的区间”，因为 BM25、embedding 相似度、cross-encoder 输出原本不是一个刻度。$w_{vec}, w_{lex}, w_{rerank}$ 是权重，表示业务上你更信哪一种信号。

为什么要融合，而不是直接只看 reranker？因为第一阶段信号本身携带重要信息。比如 query 里包含明确编号时，BM25 对精确词匹配特别有价值；而 query 比较抽象时，向量检索又更容易召回语义近邻。reranker 虽然更准，但它是在一个已经被初筛过的集合里工作，因此更适合作为“强判别器”，而不是唯一信号。

看一个玩具例子。假设 query 是“发票 6832 的退款步骤”，初检后有 3 条候选 A/B/C：

- A：向量得分 0.81，内容是“退款政策总览”
- B：向量得分 0.75，内容是“发票 6832 的退款处理说明”
- C：向量得分 0.70，内容是“售后系统工单处理”

如果只按初检排序，A 会排第一。但 reranker 联合看 query 和文本后，发现 B 明显更精确，于是得到新分数：

- A：0.65
- B：0.88
- C：0.52

如果最终排序更看重 reranker，那么 B 就会上升到第一。这个变化的意义，不是数学上“名次变了”，而是工程上“真正能回答问题的证据进入了 prompt”。

用白话说，向量检索擅长判断“像不像同类问题”，cross-encoder 更擅长判断“是不是这个问题的答案”。

再看真实工程例子。企业客服知识库里，用户问“保单 #B-78 的人工复核流程”。第一阶段 hybrid search 先召回 100 条候选，其中既有“保单复核总则”，也有“B-78 赔付例外条款”，还有“人工审核 SLA”。如果不做细排，LLM 很可能综合这些片段拼出一段“看起来合理”的流程说明。但 reranker 会把真正同时包含“B-78”“人工复核”“流程步骤”的 chunk 提到前 3，LLM 只读这些证据，答案就从“泛化说明”变成“具体流程”。

延迟为什么可控？因为 reranker 只跑 top-N。假设每条候选平均 300 token，rerank top-50 与 rerank top-500 的成本不是一个量级。实际系统里，把额外延迟控制在 100 到 400ms 的关键，不是优化某个神奇参数，而是严格限制细排候选规模。

---

## 代码实现

实现上可以把 RAG 重排序理解成 4 个动作：先召回候选，再给候选打细粒度分数，再重排，最后只取最小必要的 top-K 给 LLM。

下面先给一个可运行的最小 Python 版本。它不依赖深度学习库，而是用“关键词命中 + 编号命中”模拟 rerank 思路，重点是把流程讲清楚：

```python
from dataclasses import dataclass

@dataclass
class Doc:
    doc_id: str
    content: str
    vec_score: float
    bm25_score: float

def minmax_norm(values):
    lo, hi = min(values), max(values)
    if hi == lo:
        return [1.0 for _ in values]
    return [(v - lo) / (hi - lo) for v in values]

def simple_rerank_score(query, content):
    score = 0.0
    for token in ["退款", "步骤", "发票", "6832"]:
        if token in query and token in content:
            score += 1.0
    if "6832" in query and "6832" in content:
        score += 2.0
    return score

def rerank(query, docs, w_vec=0.2, w_lex=0.2, w_rerank=0.6):
    vecs = minmax_norm([d.vec_score for d in docs])
    lexs = minmax_norm([d.bm25_score for d in docs])
    reranks = minmax_norm([simple_rerank_score(query, d.content) for d in docs])

    scored = []
    for d, sv, sl, sr in zip(docs, vecs, lexs, reranks):
        final_score = w_vec * sv + w_lex * sl + w_rerank * sr
        scored.append((final_score, d.doc_id, d.content))

    return sorted(scored, key=lambda x: x[0], reverse=True)

query = "发票编号6832的退款步骤是什么"
docs = [
    Doc("A", "退款政策总览，适用于一般订单。", 0.81, 0.30),
    Doc("B", "发票编号6832退款步骤：提交申请，财务复核，原路退回。", 0.75, 0.85),
    Doc("C", "售后工单处理与客服升级路径。", 0.70, 0.20),
]

ranked = rerank(query, docs)
assert ranked[0][1] == "B"
assert "6832" in ranked[0][2]
print(ranked[0])
```

这个例子虽然是玩具实现，但结构和生产系统一致：

1. `retrieve(k=top_n)` 先多召回候选。
2. `rerank(query, candidates)` 对 query 与候选逐条细排。
3. 重新按分数排序。
4. 只把 top-K，例如 3 到 5 条，送进 LLM。

如果要用本地 cross-encoder，结构通常是这样：

```python
from sentence_transformers import CrossEncoder

class RerankedRetriever:
    def __init__(self, retriever, reranker_name="cross-encoder/ms-marco-MiniLM-L-6-v2"):
        self.retriever = retriever
        self.reranker = CrossEncoder(reranker_name)

    def query(self, text, k=5, rerank_top_n=20):
        candidates = self.retriever.retrieve(text, k=rerank_top_n)
        pairs = [(text, doc["content"]) for doc in candidates]
        scores = self.reranker.predict(pairs)
        top_docs = [candidates[i] for i in sorted(range(len(scores)), key=lambda i: -scores[i])[:k]]
        return top_docs
```

这段代码的关键不是模型名字，而是接口分层。`retriever` 负责“快”，`reranker` 负责“准”，两者职责分开后，后续替换向量库、BM25、托管 API 都比较容易。

如果用托管 rerank 服务，代码会更短。思路一般是“搜索 API 返回候选后，服务端直接附带 rerank 配置并给出重排结果”。它的优势是部署简单，缺点是成本、可控性和隐私边界要单独评估。对新团队来说，先把流程跑通，再决定本地化还是托管，通常更合理。

真实工程里还要多做一步：chunk 级排序后，再做文档级去重。否则同一篇文档的多个相邻 chunk 可能占满 top-K，结果把上下文多样性压没了。

---

## 工程权衡与常见坑

重排序不是“加上就一定更好”，而是“在延迟、成本、精度之间重新分配预算”。

第一类权衡是延迟。cross-encoder 的准确度通常高于 bi-encoder，但每次查询都要对每个候选逐一推理，所以延迟会随着 `rerank_top_n` 近似线性增长。生产里常见的经验值是只 rerank top-50 到 top-100；如果超过 500ms 仍不达标，优先减小候选规模、压缩 chunk 长度，或换更轻量的 reranker，而不是无上限堆机器。

第二类权衡是可解释性。embedding 分数、BM25 分数、reranker 分数来自不同模型体系，数值含义不同。你看到 `0.92` 和 `0.45`，不代表前者一定更强，只说明打分刻度不一样。如果不归一化，业务方、运营、审计都会问：为什么第二轮打了更低分，最后却排更前？所以排序系统要么统一归一化，要么在后台同时展示“原始分数 + 最终融合分数”。

第三类权衡是召回与精排的职责边界。很多系统失败，不是 reranker 不够强，而是第一阶段没把答案捞出来。比如仅靠向量检索处理“合同号 ZX-19 修订条款”，大概率会漏掉编号精确匹配，导致 reranker 根本没有机会。此时真正该做的是 hybrid 检索，而不是盲目换更大的 reranker。

下面是常见坑与规避方式：

| 常见坑 | 描述 | 规避 |
| --- | --- | --- |
| rerank 全量 | 1000+ 文档 rerank 拉垮 latency | 限制 rerank top 50~100 |
| 分数不一致 | embedding 0.92 vs rerank 0.45 难解释 | 归一化/双重展示 |
| 只用向量 | 丢失精确关键词/编号 | 引入 hybrid 检索 + rerank |

还有几个容易被忽略的问题。

一是训练域偏移。域偏移是“模型训练见过的文本分布，和你线上数据不一样”。通用 reranker 在 FAQ、网页问答上表现很好，但到了法律条款、保险批注、代码仓库文档，未必仍然稳定。解决办法通常不是立刻自训练，而是先离线评测：拿一批标注 query-doc 对，比较“只检索”和“检索+重排”的 NDCG、MRR、Recall@K，再决定是否值得上线。

二是 chunk 粒度。chunk 是“切分后的文档片段”。切太小，候选会丢失上下文；切太大，reranker 成本上升，还会把无关信息一起带进来。很多团队上线后发现重排序提升有限，根因其实是 chunk 设计不合理。

三是 top-K 设定。top-K 不是越大越安全。对大多数问答系统，进入生成阶段的证据控制在 3 到 8 条更常见。超过这个范围，LLM 的注意力会被稀释，最终答案反而变差。

---

## 替代方案与适用边界

重排序是当前 RAG 的主流强化手段，但不是唯一方案。

第一种替代方案是只用 hybrid search 加 rank fusion。rank fusion 是“把多个检索结果合并成一个总排序”，典型方法是 RRF。RRF 的好处是快、稳定、工程复杂度低，非常适合 FAQ 机器人、低延迟搜索助手、早期验证产品。在内容质量本身较高、问题形式较标准时，它已经能解决大部分问题。

第二种替代方案是 LLM rerank。也就是让大模型直接判断候选文档与 query 的匹配程度。它的优点是语义理解更强，尤其适合复杂推理、长上下文比较、多跳证据判断；缺点是贵、慢、稳定性受 prompt 和模型版本影响大。因此它更适合候选数小于 10、预算充足、对准确率要求极高的场景，而不是高并发在线服务的默认配置。

第三种替代方案是先不做 rerank，而是先优化数据。很多检索失败表面上像排序问题，实际是知识库结构问题，例如标题缺失、chunk 切分混乱、元数据不全、同义词映射缺失。若数据层没有整理好，过早上 reranker 只是掩盖问题。

一个典型的真实边界场景是低延迟 FAQ 机器人。假设系统 SLA 是 300ms 内返回答案，那么你很可能没有余量再加一个 200 到 300ms 的 cross-encoder。这时更实际的做法是：先用 BM25+向量做 hybrid 搜索，再用 RRF 融合，直接取前 5 条给 LLM。等监控显示满意度下降，或发现编号类、流程类问题频繁失败，再引入轻量 reranker 做升级。

方案对比如下：

| 方案 | 优点 | 适用边界 |
| --- | --- | --- |
| Cross-encoder rerank | 精度最高 | 候选数 ≤100、容忍额外 200~400ms |
| LLM rerank | 上下文理解最强 | 少量查询、预算充足 |
| RRF/Hybrid | 极低延迟 | 内容已高相关、初验阶段 |

所以适用边界可以归纳成一句话：如果你的问题主要是“候选里有答案，但排不准”，上 reranker；如果问题是“候选里根本没有答案”，先修召回；如果问题是“系统预算和时延非常紧”，先用 hybrid + fusion，把简单方案做到稳定。

---

## 参考资料

- Pinecone, “Rerankers and Two-Stage Retrieval”: https://www.pinecone.io/learn/series/rag/rerankers/
- Ailog, “Reranking for RAG”: https://app.ailog.fr/en/blog/guides/reranking
- Calibraint, “Retrieval Augmented Generation Failure Modes”: https://www.calibraint.com/blog/retrieval-augmented-generation-failure
- Ragaboutit, “Adaptive Retrieval Reranking”: https://ragaboutit.com/adaptive-retrieval-reranking-how-to-implement-cross-encoder-models-to-fix-enterprise-rag-ranking-failures/
- StackAI, “RAG Best Practices”: https://www.stackai.com/insights/retrieval-augmented-generation-%28rag%29-best-practices-for-enterprise-ai-chunking-embeddings-reranking-and-hybrid-search-optimization
- Pinecone Docs, “Rerank Search Results”: https://docs.pinecone.io/guides/search/rerank-results
