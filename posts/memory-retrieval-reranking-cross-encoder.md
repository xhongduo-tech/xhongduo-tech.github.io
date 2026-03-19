## 核心结论

Cross-Encoder 精排的作用不是“找到更多候选”，而是“把已经召回的候选重新排准”。在 Agent 记忆系统里，第一次召回通常由 Bi-Encoder 完成。Bi-Encoder 是把 query 和文档分别编码成向量再做相似度比较的模型，优点是快，缺点是排序里会混入“看起来像相关、实际上不够相关”的噪声。

Cross-Encoder 则把 query 和候选 chunk 拼成一个联合输入，让同一个 Transformer 同时看两者。这样它能直接判断“这个问句里的关键词、语义关系、限定条件，是否真的在这个候选里被满足”。结果通常不是把 recall 大幅拉高，而是把 top 10 里的错误顺序纠正掉。对 RAG 和记忆检索来说，这种纠正非常值钱，因为最终送进 LLM 的上下文通常只有 5 到 10 条，前几名错了，后面的答案就容易漂。

一个可操作的工程结论是：先用 BM25 或向量检索召回 50 到 100 条，再对其中 top 20 到 top 50 做 Cross-Encoder 精排，最后留下 top 5 到 top 10 注入上下文。这样通常能把质量提升控制在可接受延迟内。Mixpeek 文档给出的典型范围是：搜索阶段 <20ms，重排序 50 到 100ms，总体 70 到 120ms。Deepak 的生产案例中，50 到 10 的 cross-encoder rerank 增加约 28ms，整个链路 p50 为 66ms。

下表可以把“为什么要多一层精排”说清楚：

| 方案 | 主要职责 | 典型 nDCG@10 / Recall 表现 | 延迟特征 | 是否适合直接做最终排序 |
| --- | --- | --- | --- | --- |
| BM25 | 词项召回，擅长精确关键词 | 词匹配强，语义弱 | 很低 | 不建议单独承担 |
| Bi-Encoder | 语义召回，擅长大规模 ANN | 召回强，但 top 位次有噪声 | 很低到中等 | 通常不够稳 |
| Cross-Encoder | 精确相关性判断 | 前几名质量最高 | 中等到较高，按候选数线性增长 | 适合做最终精排 |
| 两阶段 Recall + Rerank | 先找全，再排准 | 通常最平衡 | 可控 | 最常见生产方案 |

玩具例子：query 是“量子安全迁移时间表”。Bi-Encoder 可能把“量子计算威胁 RSA”“NIST 后量子标准”“量子随机数生成”都排进前列，因为这些句子都和“量子安全”语义接近。Cross-Encoder 会进一步看到“迁移时间表”这个约束，于是把含有迁移节点、标准发布日期、实施窗口的文档排到最前面，把只有泛泛背景介绍的文档往后压。

---

## 问题定义与边界

问题定义可以写成一句话：在记忆检索中，第一次召回已经把“可能相关”的内容找出来了，但前几名的排序还不够可靠，导致高价值记忆没有进入最终上下文窗口。

这里的“记忆”不是数据库里的任意文本，而是 Agent 历史交互、任务状态、工具输出、用户偏好、外部资料切片等可复用上下文。它有三个边界。

第一，Cross-Encoder 只负责精排，不负责全库搜索。因为它要对每个 query-document 对单独跑一次前向计算，复杂度接近 $O(k)$，其中 $k$ 是候选数。对全库上百万 chunk 直接跑，成本不可接受。

第二，精排只解决“相关性排序”，不自动解决“记忆是否过期”。例如用户问“上周讨论的预算决策”，纯相关性很高的旧版本预算也可能被排前面。记忆系统还需要时间过滤、版本元数据、重要度加权等额外机制。

第三，精排收益依赖首次召回的上限。如果第一次召回的 top 100 根本没把正确 chunk 找进来，Cross-Encoder 无法“凭空创造相关结果”。它是 second-stage ranker，不是召回补救器。

下面这个表格更适合做系统设计时的预算约束：

| 阶段 | 输入规模 | 目标 | 常见模型 | 延迟预算 | 失败时 fallback |
| --- | --- | --- | --- | --- | --- |
| Recall | 全库到 top 50/100 | 找全候选 | BM25、BGE、DPR、ColBERT | 5-30ms | 另一套召回器或 RRF |
| Rerank | top 20/50 | 排准前几名 | Cohere Rerank、bge-reranker-v2、Jina Reranker | 50-100ms | 原始召回分数 |
| Context Pack | top 5/10 | 控制 token 预算 | 规则裁剪、贪心装箱 | 1-5ms | 截断到更少 chunk |

真实工程例子：一个客服 Agent 维护“用户历史工单 + 产品知识库 + 最近会话状态”三类记忆。用户问“你上次说的退款例外条件对企业套餐适用吗”。仅靠向量召回，系统可能把“退款政策总则”“企业套餐价格说明”“上次对话中的退款讨论”混在一起。加上 Cross-Encoder 后，问句里的“上次说的”“例外条件”“企业套餐适用”会共同参与判断，最终把真正能回答当前问题的 5 条记忆压到前面。

---

## 核心机制与推导

Cross-Encoder 的输入不是两个独立向量，而是一个联合序列：

$$
[q; d] = [CLS], q_1, ..., q_n, [SEP], d_1, ..., d_m, [SEP]
$$

这里的核心是 attention。attention 可以理解为“每个 token 在读别的 token 时，应该分多少注意力”。标准形式是：

$$
Attention(Q,K,V)=softmax\left(\frac{QK^\top}{\sqrt{d_k}}\right)V
$$

它的工程含义是：query 里的词能直接看到 document 里的词，document 里的词也能反过来影响 query 的表示。于是模型不仅知道“这两段话主题相近”，还知道“问句里的限制词是否被文档具体满足”。

最终相关性分数通常来自 `[CLS]` 向量，也就是整对输入的汇总表示：

$$
score(q,d)=W \cdot h_{[CLS]} + b
$$

如果系统本身就是混合检索，最终排序常见做法是先做一轮分数融合，再做精排。可以抽象成：

$$
final\_score=\alpha \cdot BM25 + \beta \cdot Dense + \gamma \cdot ColBERT + \delta \cdot CrossEncoder
$$

其中 $\alpha,\beta,\gamma,\delta$ 是权重。实际工程里更常见的做法是：BM25、Dense、ColBERT 先做召回或 RRF 融合，Cross-Encoder 不一定和前面同权加和，而是作为最后一层覆盖式重排器。

为什么 Bi-Encoder 会有排序噪声？因为它把整个 query 压缩成一个向量，把整个文档压缩成一个向量，再比较两者距离。压缩以后，细粒度约束容易丢失。比如 query 是“支持 Python 3.12 且不依赖 CUDA 的 embedding 服务”，两个文档都可能和“embedding 服务”很接近，但只有一个文档同时满足“Python 3.12”和“无 CUDA”。Cross-Encoder 能直接建模这些交叉条件，因此更适合 top-N 精排。

从指标角度看，nDCG@10 可以理解为“前 10 个结果里，高相关结果是否被尽量排在前面”的指标。它比单纯 recall 更适合评价精排层，因为精排关心的是前几名的质量，而不是更靠后的长尾候选。

---

## 代码实现

下面给一个可运行的最小实现。它模拟“两阶段检索 + 精排”的核心流程，不依赖真实模型，也能把排序逻辑跑通。

```python
import math
from typing import List, Dict

query = "企业套餐退款例外条件"

docs = [
    {"id": "a", "text": "退款政策总则，介绍标准退款时限。", "bm25": 0.62, "dense": 0.71},
    {"id": "b", "text": "企业套餐退款例外条件：签约后7天内可申请人工审核。", "bm25": 0.81, "dense": 0.77},
    {"id": "c", "text": "企业套餐价格说明，不含退款例外条款。", "bm25": 0.74, "dense": 0.69},
    {"id": "d", "text": "上次会话记录：用户询问企业套餐退款例外条件。", "bm25": 0.70, "dense": 0.80},
]

def fuse_score(doc: Dict[str, float]) -> float:
    return 0.5 * doc["bm25"] + 0.5 * doc["dense"]

def cross_encoder_score(query: str, text: str) -> float:
    # 玩具版精排器：按 query 关键词命中数重新打分
    keywords = ["企业套餐", "退款", "例外条件"]
    hits = sum(1 for kw in keywords if kw in text)
    base = hits / len(keywords)
    # 给明确规则描述更高分
    if "7天内" in text or "人工审核" in text:
        base += 0.2
    return min(base, 1.0)

# 第一阶段：融合召回分数
stage1 = sorted(docs, key=fuse_score, reverse=True)

# 第二阶段：对 top-3 精排
top_k = stage1[:3]
reranked = sorted(top_k, key=lambda d: cross_encoder_score(query, d["text"]), reverse=True)

assert stage1[0]["id"] in {"b", "d"}   # 初召回前列可能有噪声
assert reranked[0]["id"] == "b"        # 精排后把真正满足约束的文档排到第一

print("stage1:", [d["id"] for d in stage1])
print("reranked:", [d["id"] for d in reranked])
```

这段代码里最关键的不是玩具打分函数，而是流程顺序：

1. 先用便宜的分数把候选缩到较小集合。
2. 再把 query 和候选成对计算精排分数。
3. 只让精排参与最后一小段排序。

如果换成真实模型，常见 Python 形式如下：

```python
from sentence_transformers import CrossEncoder
import numpy as np

query = "企业套餐退款例外条件"
corpus = [
    "退款政策总则，介绍标准退款时限。",
    "企业套餐退款例外条件：签约后7天内可申请人工审核。",
    "企业套餐价格说明，不含退款例外条款。",
    "上次会话记录：用户询问企业套餐退款例外条件。"
]

bm25_scores = np.array([0.62, 0.81, 0.74, 0.70])
dense_scores = np.array([0.71, 0.77, 0.69, 0.80])

fusion_scores = 0.5 * bm25_scores + 0.5 * dense_scores
top_idx = np.argsort(fusion_scores)[::-1][:3]

model = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")
pairs = [[query, corpus[i]] for i in top_idx]
rerank_scores = model.predict(pairs)

final_idx = top_idx[np.argsort(rerank_scores)[::-1]]
assert len(final_idx) == 3
```

在真实工程中，还要再补三件事：

| 组件 | 最低要求 | 原因 |
| --- | --- | --- |
| 候选截断 | 只精排 top 20 或 top 50 | 控制线性成本 |
| 批处理 | 一次送多对 query-doc | 提高吞吐 |
| 超时与降级 | 超时后回退原始排序 | 避免 Agent 阻塞 |

---

## 工程权衡与常见坑

最常见的错误是把 Cross-Encoder 当作“质量越高越应该多跑”。这在工程上通常是错的。Cross-Encoder 的成本随候选数线性上升，所以真正的优化方向不是“多排一点”，而是“让第一次召回更稳，让精排输入更小”。

下面这些坑最常见：

| 常见坑 | 结果 | 规避方式 |
| --- | --- | --- |
| 对 top 200 甚至 top 1000 全量精排 | 延迟失控，吞吐骤降 | 限制到 top 20 或 top 50 |
| 召回阶段太弱 | 正确文档没进候选集 | 先优化 recall，不要指望精排补救 |
| 只看相关性，不看时间和版本 | 旧记忆排到前面 | 增加 recency filter、版本号、source metadata |
| reranker 超时无 fallback | 整个 Agent 卡住 | 回退到 RRF 或原始召回分数 |
| chunk 太长 | 模型截断后关键信息丢失 | 控制 chunk 长度，必要时窗口切分 |
| 线上不做指标拆分 | 不知道问题在 recall 还是 rerank | 分别统计 Recall@K、nDCG@10、p50/p95 延迟 |

一个经验公式是：如果你的上下文窗口最终只放 $M$ 条记忆，那么第一次召回至少应保证正确答案高概率进入 top $5M$ 到 top $10M$，然后再由精排压缩到最终窗口。比如最终只放 5 条，那么 recall 阶段常见配置就是 top 25 到 top 50。

还有一个常被忽略的点：Cross-Encoder 分数不一定适合跨 query 直接比较。它更适合在同一个 query 的候选集合内排序，而不是把不同 query 的分数拉到一起做全局决策。因此在线上日志、告警阈值、样本回放系统里，不要把“0.82 比 0.76 高很多”当成绝对语义，只能把它理解为“当前 query 下更相关”。

---

## 替代方案与适用边界

不是所有场景都必须上 Cross-Encoder。是否值得引入，取决于三个条件：候选规模、延迟预算、现有召回质量。

如果系统已经是低并发、高价值问答，且错误排序的代价高，比如企业知识库、合规问答、复杂 Agent 记忆注入，那么 Cross-Encoder 往往是高 ROI 组件。因为它直接提升前几名质量，而这正是最终答案最敏感的部分。

如果系统是高并发、超低延迟场景，比如实时推荐、毫秒级搜索补全，Cross-Encoder 往往太贵。这时更现实的方案是：

| 替代方案 | 适用场景 | 优点 | 缺点 |
| --- | --- | --- | --- |
| 仅用 Bi-Encoder | 极低延迟、高 QPS | 便宜、易扩展 | 前几名质量不够稳 |
| BM25 + Dense + RRF | 预算有限的通用检索 | 简单、稳健 | 精排能力有限 |
| ColBERT | 需要更强 token 级匹配，但不想全用 CE | 质量高于普通双编码器 | 索引和推理更复杂 |
| 轻量 reranker | 延迟敏感但仍想要精排 | 比标准 CE 更快 | 上限通常低一些 |
| 异步 rerank | 对首屏速度敏感，可接受二次刷新 | 首包快 | 交互复杂度上升 |

适用边界可以这样判断：

1. 如果首次召回的 nDCG@10 已经足够高，精排带来的提升可能不值得额外成本。
2. 如果 query 很短、意图简单、候选差异大，Bi-Encoder 往往已经够用。
3. 如果 query 有多重约束、否定条件、时间范围、角色关系，Cross-Encoder 的收益通常更明显。
4. 如果记忆检索直接影响工具调用、事务执行、合规回答，优先保质量，再谈多出的几十毫秒。

一个实用决策规则是：先做离线 A/B。固定 recall 层，只替换 rerank 层，比较 nDCG@10、Recall@K、最终答案正确率和 p95 延迟。如果前几名质量提升明显，且整体延迟仍满足 SLO，再把 Cross-Encoder 留在线上。

---

## 参考资料

- [Mixpeek Rerank 文档](https://docs.mixpeek.com/docs/retrieval/stages/rerank)：给出两阶段模式、`top_k=100` 后 `top_n=10` 的推荐配置，以及“100 docs 约 50-100ms”的延迟范围。
- [Deepak: Production-Grade Agent Memory for RAG Systems](https://medium.com/%40deepak23188/production-grade-agent-memory-for-rag-systems-optimizing-hybrid-ann-vector-retrieval-in-bab0e72bb10b)：给出 50 到 10 rerank 增加约 28ms、整链路 p50 66ms 的生产案例，并说明 reranker 失败时回退到 RRF。
- [Sentence Transformers CrossEncoder 文档](https://www.sbert.net/docs/package_reference/cross_encoder/cross_encoder.html)：Python 中最常见的 CrossEncoder 调用接口。
- [Emergent Mind: Cross-Encoder Reranking](https://www.emergentmind.com/topics/cross-encoder-reranking)：整理了 Cross-Encoder 的联合编码形式、$s(q,d)=W \cdot h_{[CLS]} + b$ 评分头和 attention 机制。
- [Johal: Retrieval Augmented Generation ColBERT BM25 Hybrid Dense Sparse Reranking 2026](https://johal.in/retrieval-augmented-generation-colbert-bm25-hybrid-dense-sparse-reranking-2026-3/)：提供混合检索与精排的示意 benchmark 和代码样例，适合理解工程拼装方式。
- [BAAI bge-reranker-v2 系列模型卡](https://huggingface.co/BAAI/bge-reranker-v2-minicpm-layerwise)：可用于了解 bge-reranker-v2 家族的部署取舍与模型定位。
