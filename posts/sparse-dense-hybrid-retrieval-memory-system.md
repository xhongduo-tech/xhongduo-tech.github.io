## 核心结论

Sparse-Dense 混合检索的目标很直接：把“语义理解”和“关键词命中”放进同一次排序。Dense，稠密检索，白话说就是“把一句话压成向量后按语义相似度找近邻”；Sparse，稀疏检索，白话说就是“按词项是否出现、出现几次、是否稀有来打分”。两者单独使用都不完整。

在 Agent 记忆系统里，这个不完整会直接变成错误上下文。纯 Dense 能理解“供应链中断”和“供货风险”接近，但可能忽略 `material adverse effect` 这种必须精确匹配的术语。纯 BM25 能抓住术语、编号、型号，但对改写表达、同义句、摘要式提问不敏感。把两者按线性方式融合：

$$
s_{\text{hybrid}}(q,d)=\alpha \cdot s_{\text{dense}}(q,d) + (1-\alpha)\cdot s_{\text{sparse}}(q,d)
$$

其中 $\alpha$ 是权重，白话说就是“语义分和关键词分谁更重要”。在公开实践里，$\alpha \approx 0.7$ 往往是记忆检索的有效起点，意思是语义主导、关键词兜底。Aimultiple 的实验里，混合检索把 MRR@10 从 0.410 提到 0.486，Recall 也同步提升，但带来约 201 ms 的平均额外延迟。这不是免费午餐，但通常值得，因为更高质量的召回会减少 LLM 读错材料、少走无效推理路径。

一个新手最容易理解的玩具例子是查询 `2025 iPhone 15 Pro Max`。Dense 会偏向“最新旗舰手机”“高端苹果设备”这类语义接近的段落，Sparse 会确保 `iPhone 15 Pro Max` 这个完整型号被明确命中。线性融合后，既不容易漏掉“同义表述”，也不容易把缺少精确型号的泛泛介绍排到前面。

下表可以作为工程复盘的起点：

| 系统 | MRR@10 | Recall@10 | 查询延迟 | LLM token 费用 |
| --- | --- | --- | --- | --- |
| Dense-only | 0.410 | 0.912 | 基线 | 基线 |
| Hybrid + BM25/SPLADE | 0.486 | 0.947 | +201 ms | 下降明显 |
| Hybrid + Rerank | 更高 | 0.971 | 取决于 rerank 阶段 | 可下降约 55% |

结论不是“混合检索永远最好”，而是：当记忆库里同时存在语义改写、专有术语、编号、型号、法规条款时，混合检索通常是最稳的默认方案。

---

## 问题定义与边界

问题定义可以表述为：给定一个用户查询 $q$，从记忆库中的文档片段集合 $D$ 里，找出最可能帮助当前 Agent 回答问题的前 $k$ 个片段。这里的“记忆库”不一定是聊天历史，也可以是财报切片、工单记录、规则文档、用户画像摘要、工具调用结果缓存。

边界也必须先说清楚。

第一，本文讨论的是“候选召回与初排”，不是最终答案生成。也就是说，重点是把对的片段排进前列，而不是直接让检索器给出最终自然语言回答。

第二，本文讨论的是文本记忆系统，不展开图片、表格结构体、多模态 embedding，也不展开长链路 Agent 的任务规划问题。即使在真实系统里这些常常同时存在，检索层仍然可以单独分析。

第三，本文默认记忆库规模已经大到“不能靠字符串扫描”解决，通常是几千到几百万个片段。在这个量级下，必须使用倒排索引、ANN 索引或二者组合。ANN，近似最近邻，白话说就是“牺牲一点精确度换更快地找相似向量”。

为什么纯 Dense 不够？因为它优化的是语义邻近，而不是术语完整命中。比如在金融风控问答里，用户问“10-K 中是否提到 `material adverse effect` 与 supply chain risk 的关联”，Dense 很可能找回大量“供应链风险”的相关段落，但把包含精确法务措辞的关键段落排得不够靠前。对问答系统来说，这种错不是“差一点”，而是证据缺失。

为什么纯 BM25 不够？因为它只看词项统计。BM25 的基本形式是：

$$
\mathrm{score}_{\mathrm{BM25}}(q,d)=\sum_{w\in q} \mathrm{IDF}(w)\cdot
\frac{f(w,d)(k_1+1)}{f(w,d)+k_1\left(1-b+b|d|/\mathrm{avgdl}\right)}
$$

这里 $f(w,d)$ 是词在文档中出现的次数，IDF 可以理解为“这个词越稀有，命中后越值钱”。它很适合抓法规编号、错误码、型号名、API 名称，但如果用户说“账号被冻结”而文档里写的是“账户被风控限制登录”，BM25 可能就抓不稳，因为关键词表面不一致。

真实工程例子更能说明边界。在一个面向金融材料的 Agent 记忆系统中，检索对象可能是 2,000+ 个切片后的 10-K/10-Q 段落。用户问题常常同时包含语义概念和法务术语。Dense 负责把“供应链中断”“上游履约风险”“物流瓶颈”收拢到一个语义簇，Sparse 负责把 `material adverse effect`、年份、公司名、节标题这类不能模糊的信号锁住。这个场景里，单一方案天然有盲点。

---

## 核心机制与推导

混合检索的核心机制只有三步：分别打分、对齐尺度、合并排序。

先看 Dense。文档和查询都被编码成向量，常见打分是 cosine similarity，也就是余弦相似度。白话说，它不关心两个向量有多长，而看它们方向是否接近：

$$
s_{\text{dense}}(q,d)=\cos(v_q,v_d)=\frac{v_q \cdot v_d}{\|v_q\|\|v_d\|}
$$

再看 Sparse。最常见是 BM25，也可以是 SPLADE。SPLADE 是学习型稀疏表示，白话说就是“模型自己学出哪些词应该被激活，并且激活得很稀疏”，这样既保留倒排索引式的高效检索，又比传统词频方法更有语义扩展能力。

然后是融合。最简单也最常用的是线性插值：

$$
s_{\text{hybrid}}(q,d)=\alpha \cdot s_{\text{dense}}(q,d) + (1-\alpha)\cdot s_{\text{sparse}}(q,d),\qquad \alpha \in [0.3,0.7]
$$

这个公式背后的推导不复杂。你可以把 Dense 和 Sparse 看成两个有偏但互补的估计器。Dense 对语义相关性估得较准，但对精确关键词敏感度不足；Sparse 对关键词精确性估得较准，但对改写表达泛化不足。线性加权的目的不是求数学上唯一正确的概率，而是在排序意义上减少单路偏差。

这里有一个很关键的工程前提：两个分数通常不在同一量纲上。Dense 可能是 0 到 1 的余弦分，BM25 可能是几分到十几分，SPLADE 的分布又不同。如果直接相加，数值大的那一路会“天然赢”。所以生产里经常要做 min-max 归一化、z-score 标准化，或者先转成 rank 再做融合。

看一个玩具例子。假设查询是“iPhone 15 Pro Max 电池续航”。三个候选文档的分数如下：

| 文档 | Dense | Sparse | $\alpha=0.7$ 后混合分 |
| --- | --- | --- | --- |
| A: 最新旗舰续航评测，但没写完整型号 | 0.82 | 0.10 | 0.604 |
| B: 明确写了 iPhone 15 Pro Max，但内容较短 | 0.60 | 0.32 | 0.516 |
| C: 旧款 iPhone 电池说明 | 0.55 | 0.05 | 0.400 |

如果只看 Dense，A 可能排第一；如果只看 Sparse，B 可能压过 A。但融合后，A 和 B 都保住高位，C 被甩开。这个结果更接近真实需求，因为用户既关心型号，也关心续航语义。

用户给出的数值例子也可以直接算：

$$
0.7 \times 0.60 + 0.3 \times 0.32 = 0.42 + 0.096 = 0.516
$$

这恰好说明一个常见误区：混合检索不是“把两个好结果简单叠加就一定更高”，而是“让排序更稳”。它真正改善的是 top-k 的相对次序，而不是让每个文档分数都显著变大。

很多系统还会在融合后再接 RRF。RRF，Reciprocal Rank Fusion，白话说就是“谁在多个候选列表里都靠前，谁就更可信”。公式一般是：

$$
\mathrm{RRF}(d)=\sum_i \frac{1}{k+r_i(d)}
$$

其中 $r_i(d)$ 是文档在第 $i$ 个列表中的排名。RRF 的优势是对分数量纲不敏感，缺点是会损失部分原始分差信息。所以常见做法不是“线性融合”和“RRF”二选一，而是先分别召回，再按场景选一种合并方式，或者线性融合做粗排、RRF 做补强覆盖。

---

## 代码实现

下面给出一个可运行的最小 Python 示例。它不依赖外部库，目的是把“并行打分、归一化、融合、排序”的逻辑讲清楚。示例里用手写分数代替真实向量检索和 BM25 检索。

```python
from math import isclose

def minmax_normalize(scores):
    if not scores:
        return {}
    values = list(scores.values())
    lo, hi = min(values), max(values)
    if isclose(lo, hi):
        return {k: 1.0 for k in scores}
    return {k: (v - lo) / (hi - lo) for k, v in scores.items()}

def hybrid_rank(dense_scores, sparse_scores, alpha=0.7):
    dense_norm = minmax_normalize(dense_scores)
    sparse_norm = minmax_normalize(sparse_scores)

    all_docs = set(dense_norm) | set(sparse_norm)
    fused = {}
    for doc_id in all_docs:
        d = dense_norm.get(doc_id, 0.0)
        s = sparse_norm.get(doc_id, 0.0)
        fused[doc_id] = alpha * d + (1 - alpha) * s

    ranked = sorted(fused.items(), key=lambda x: x[1], reverse=True)
    return ranked

dense_scores = {
    "A_latest_flagship_review": 0.82,
    "B_exact_model_note": 0.60,
    "C_old_iphone_battery": 0.55,
}
sparse_scores = {
    "A_latest_flagship_review": 2.0,
    "B_exact_model_note": 8.0,
    "C_old_iphone_battery": 1.0,
}

ranked = hybrid_rank(dense_scores, sparse_scores, alpha=0.7)

# A 语义最强，B 关键词最强，二者应排在前两位
assert ranked[0][0] in {"A_latest_flagship_review", "B_exact_model_note"}
assert ranked[1][0] in {"A_latest_flagship_review", "B_exact_model_note"}
assert ranked[-1][0] == "C_old_iphone_battery"

# 验证缺失某一路分数时仍可工作
ranked2 = hybrid_rank({"X": 0.9}, {"Y": 5.0}, alpha=0.7)
assert len(ranked2) == 2

print(ranked)
```

如果把它翻译成真实工程流水线，通常是下面这样：

```python
def search_memory(query, alpha=0.7, top_k=10):
    dense_hits = dense_index.search(encode_dense(query), top_k=100)
    sparse_hits = sparse_index.search(encode_sparse(query), top_k=100)

    dense_scores = normalize(dict(dense_hits))
    sparse_scores = normalize(dict(sparse_hits))

    fused = {}
    for doc_id in set(dense_scores) | set(sparse_scores):
        fused[doc_id] = (
            alpha * dense_scores.get(doc_id, 0.0)
            + (1 - alpha) * sparse_scores.get(doc_id, 0.0)
        )

    candidates = sorted(fused.items(), key=lambda x: x[1], reverse=True)[:50]
    reranked = cross_encoder.rerank(query, [doc_id for doc_id, _ in candidates])
    return reranked[:top_k]
```

这个伪代码里有四个工程点。

第一，并行。Dense 编码和 Sparse 查询最好并行发起，因为它们的依赖不同，串行会白白增加尾延迟。

第二，候选池。不要只取 top 10 后立即融合。更稳妥的做法是两边各取 top 50 或 top 100，再在并集上融合，否则一边召回早早把好文档截断，另一边就没机会补回来。

第三，归一化。直接拿余弦分和 BM25 分做线性组合是常见错误。

第四，二阶段排序。Cross-encoder，交叉编码器，白话说就是“把 query 和候选文档一起喂给模型做更贵但更准的相关性判断”。它通常不适合全库检索，但非常适合对前几十个候选做精排。

一个真实工程例子是金融风控 Agent。系统先把 10-K 切成 2,000+ 个片段，同时建 Int8 HNSW 向量索引和 BM25 索引。用户问“公司是否在 2025 年披露 material adverse effect related to supply chain risk”。Dense 路由召回“供应链风险”相关语义段落，BM25 路由锁定 `material adverse effect`、年份、公司名。融合后取前 50 个，再交给 cross-encoder 精排，最终送进 LLM 的只保留前 5 到 10 个片段。结果是 Recall@10 更高，送给 LLM 的无关文本更少，token 成本也明显下降。

---

## 工程权衡与常见坑

混合检索的第一大权衡是准确率和延迟。两路检索并行不代表“零成本”，因为你至少要多维护一套索引、多走一次查询路径，还可能增加归一化、融合、rerank 的计算。公开实践里约 201 ms 的增量并不夸张，问题是这 201 ms 换来的召回提升是否足够值钱。对离线分析、企业知识问答通常值得；对极低延迟客服首响，则要更谨慎。

第二大权衡是调参复杂度。$\alpha$ 不是拍脑袋定出来的。很多团队一开始会认为“关键词重要，那就把 $\alpha$ 降低”，结果很快发现 paraphrase 被大量打掉；或者认为“Dense 是大模型时代主流”，把 $\alpha$ 拉到 0.9，结果型号、错误码、法规编号开始漏检。正确做法是拿真实 query 集，在不同 $\alpha$ 下画 MRR@10、Recall@k、nDCG 曲线，而不是看几条样例就决定。

第三大权衡是 Sparse 组件本身。BM25 简单稳健，但词面能力上限固定；SPLADE 更强，但 posting list 可能更大，索引构建和推理成本也更复杂。posting list，倒排链表，白话说就是“一个词命中了哪些文档的列表”。稀疏模型如果激活词太多，查询延迟就会上升，甚至抵消它带来的召回收益。

常见坑有下面几类：

| 问题 | 现象 | 本质原因 | 处理方式 |
| --- | --- | --- | --- |
| 未归一化直接相加 | 某一路长期压制另一路 | 分数量纲不同 | 先做标准化或 rank 融合 |
| $\alpha$ 只凭经验设置 | 线上效果波动大 | 不同 query 类型差异大 | 用验证集画曲线 |
| top_k 过小 | 融合后提升不明显 | 候选在召回阶段已被截断 | 两路先取更大的候选池 |
| 只测离线不测端到端 | 离线分数高，线上回答仍差 | rerank 与上下文截断未纳入 | 联合评估检索与答案质量 |
| 稀疏索引过重 | 延迟飙升 | SPLADE 稀疏度失控 | 控制激活词数与缓存策略 |

一个典型坑是“以为混合检索只调 $\alpha$ 就够了”。实际上至少还要同时关注切片长度、候选池大小、归一化方式、rerank 截断阈值。比如你把长文切得太粗，Dense 可能因为主题覆盖更广而得分偏高，Sparse 则因为关键词密度被稀释而偏低，最后让融合失衡。

另一个典型坑来自评估口径。很多团队只看 Recall@100，觉得“都召回到了”，但 Agent 真正消耗的是前几个片段。对记忆系统，MRR@10、Recall@5、最终答案正确率，往往比大候选池指标更重要，因为 LLM 不会认真读完 100 个片段。

---

## 替代方案与适用边界

线性混合不是唯一方案，只是最容易落地的方案。什么时候该换别的方法，要看场景。

第一类替代方案是 RRF。它适合两路分数很难校准、但排名质量都还不错的情况。优点是稳定、实现简单；缺点是它只看名次，不看“第一名比第二名高很多还是只高一点”。

第二类是动态权重，也就是 query-aware $\alpha$。白话说就是“不同查询，语义权重和关键词权重不一样”。例如带明显编号、型号、法规条款的查询，可以降低 $\alpha$；纯自然语言问法则提高 $\alpha$。这类方法常用规则、轻量分类器，甚至 LLM 预测查询类型来决定权重。效果可能更高，但维护更复杂。

第三类是 Sparse-only + 强 rerank。如果你的场景天然关键词密集，例如法规条文、错误码检索、API 文档版本定位，那么仅用 BM25 或 SPLADE 做大候选召回，再用 cross-encoder 精排，可能已经足够。这里再引入 Dense 未必划算。

第四类是结构化检索或图检索。如果知识主要靠实体关系和字段过滤，例如“某客户在 30 天内是否同时触发规则 A 与规则 B”，那就不该把一切都压成文本语义问题，而应该把 metadata filter、图关系遍历和文本检索结合。

可以用一个矩阵来判断：

| 方案 | 适用情景 | 优点 | 缺点 |
| --- | --- | --- | --- |
| 线性 Hybrid | 语义与术语都重要的通用记忆系统 | 实现直接、收益稳定 | 需要调 $\alpha$ 与归一化 |
| RRF + BM25/Dense | 两路分数不可比，但排名各自有效 | 稳健 | 分差利用不足 |
| Query-aware α | 查询类型差异明显 | 潜在效果更高 | 维护复杂 |
| Sparse-only + rerank | 法规、编号、错误码密集场景 | 低延迟、低复杂度 | paraphrase 能力弱 |
| Structured/Graph + Text | 强关系型知识 | 精准约束强 | 建模和维护成本高 |

举一个只适合 Sparse-only 的例子。假设系统要回答“《证券法》第 193 条第 2 款是否涉及信息披露义务”。这类查询的决定性信号是法规编号和条款位置，不是抽象语义。此时用 BM25 抓 `193 条`、`第 2 款`、`信息披露义务`，再接一个强 reranker，通常就够了。强行加 Dense 不一定带来增益，反而增加索引和延迟成本。

所以适用边界可以概括为一句话：当系统同时面对“语义改写”和“精确词项”两类需求时，用混合检索；当其中一类需求压倒性占优时，优先用更简单、更便宜的单路方案。

---

## 参考资料

- Aimultiple, “Hybrid RAG: Boosting RAG Accuracy”. https://research.aimultiple.com/hybrid-rag//
- Deepak Kumar, “Production-Grade Agent Memory for RAG Systems: Optimizing Hybrid ANN + Vector Retrieval”. https://medium.com/%40deepak23188/production-grade-agent-memory-for-rag-systems-optimizing-hybrid-ann-vector-retrieval-in-bab0e72bb10b
- Emergent Mind, “Dense-Sparse Hybrid Retrieval”. https://www.emergentmind.com/topics/dense-sparse-hybrid-retrieval
- Emergent Mind, “Hybrid Retrieval Architectures”. https://www.emergentmind.com/topics/hybrid-retrieval-architectures
- Emergent Mind, “SPLADE-doc”. https://www.emergentmind.com/topics/splade-doc
- Naver SPLADE GitHub Repository. https://github.com/naver/splade
