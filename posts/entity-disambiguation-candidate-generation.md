## 核心结论

实体消歧的候选生成，不是直接判断 mention 最后对应哪个实体，而是先用尽量低的代价，把正确实体放进一个较小的候选集合 `Top-K`。这里的 mention 可以理解为“文本里出现、需要链接到知识库实体的一段名字”，比如“Apple”“Tesla”“乔丹”。

候选生成阶段真正要优化的指标，是 `recall@K`，即“正确实体有没有出现在前 `K` 个候选里”。如果真值没有被召回，后面的精排模型再强也没有用，因为它根本看不到正确答案。

$$
recall@K = \frac{\text{真值被包含在 Top-K 中的样本数}}{\text{总样本数}}
$$

一个最小玩具例子就能说明问题。mention 是 “Apple”，上下文是 “I bought a new iPhone and MacBook”。候选生成至少应该先捞出：

- `Apple Inc.`
- `Apple (fruit)`
- `Apple Records`

如果系统只保留 `K=1`，而先验误把“苹果水果”排在第一，那么真值会直接丢失；如果保留 `K=2`，`Apple Inc.` 仍有机会进入后续精排。候选生成的核心不是“一步命中”，而是“保住上限”。

候选生成和精排的职责不同，不能混淆：

| 模块 | 目标 | 优先指标 | 常见方法 | 成本特征 |
|---|---|---|---|---|
| 候选生成 | 高召回拿到 Top-K | `recall@K` | alias、redirect、BM25、向量召回、流行度 | 面向大规模检索，要求便宜 |
| 精排 | 从 Top-K 中选最终实体 | accuracy、MRR、NDCG | cross-encoder、特征模型、LLM 打分 | 对单条样本更贵 |

真实工程里，常见做法不是押注单一路径，而是把 `alias recall + dense retrieval` 组合起来，再用 BM25 和先验补洞。这样做的原因很直接：热门实体不能吞掉长尾实体，词面相似不能替代语义相似，而低延迟又要求不能把精排输入做得无限大。

---

## 问题定义与边界

实体消歧候选生成的输入，通常有三部分：

- `m`：mention，即待消歧的文本片段
- `x`：context，即 mention 所在上下文
- `E`：entity set，即知识库里的全部实体集合

输出是一个较小的候选集合 `C_K`：

$$
C_K = Top_K(s_{gen}(e \mid m, x)), \quad C_K \subset E
$$

这里的 `s_gen(e | m, x)` 可以理解为“候选生成阶段对实体 `e` 的召回分数”。它不需要非常精确，只要能把真值尽量送进 `Top-K` 即可。

几个核心字段可以先固定理解：

| 字段 | 含义 | 白话解释 |
|---|---|---|
| `mention` | 待链接文本 | 文本里那段名字 |
| `context` | mention 周围上下文 | 用来判断这段名字到底指谁 |
| `entity` | 知识库实体 | 系统最终要链接到的标准对象 |
| `candidate set` | 候选集合 | 一批“可能是对的实体” |
| `Top-K` | 前 `K` 个候选 | 控制后续精排成本的截断结果 |

边界也要说清楚。候选生成不负责做最终判断，它只负责解决两个问题：

1. 不要漏掉真值。
2. 不要让候选集大到拖垮后续精排。

这意味着候选生成不是“越多越好”。如果你把整个知识库都交给精排，理论上召回率最高，但计算成本会失控；如果你只留 3 个候选，延迟会很好，但很容易漏掉长尾真值。工程目标是找到一个可接受的平衡点。

看一个直观例子。新闻里出现 mention “Tesla”，上下文分别是：

- “Tesla released a new charging standard.”
- “He was inspired by Nikola Tesla’s experiments.”

这两个 mention 词面完全一样，但第一个更可能指 `Tesla, Inc.`，第二个更可能指 `Nikola Tesla`。这说明候选生成不能只看 mention 本身，还必须引入上下文，否则候选集会被高频实体主导。

再看一个真实工程例子。电商搜索日志里有一句：“苹果 16 什么时候降价”。这里的 “苹果” 不是一般百科语义，而是商品和品牌场景中的实体链接问题。候选生成至少要能召回：

- `Apple Inc.`
- `iPhone 16`
- `Apple (fruit)`

如果系统只靠静态别名字典，很可能因为“苹果”最常见含义太宽，把水果或泛品牌解释顶到前面；而上下文中的“16”“降价”其实是重要信号，它提示这是消费电子场景。候选生成的边界，就是把这些可能性捞出来并交给下一阶段，而不是在这一阶段就做最终裁决。

---

## 核心机制与推导

候选生成常用的是多路召回。多路召回的本质，是让不同信号各自负责一种覆盖能力，然后把它们叠加。因为任何单一路径都有盲区：

- alias 字典擅长别名映射，但不理解语义
- redirect 擅长规范化历史名称，但覆盖依赖知识库维护
- BM25 擅长词面匹配，但处理不了深层语义
- dense retrieval 擅长语义近邻，但有时会过召
- popularity prior 擅长热门实体兜底，但会压制长尾

一个常见融合打分形式是：

$$
s_{gen}(e \mid m, x) =
\lambda_1 \log p(e \mid m) +
\lambda_2 BM25(x, e) +
\lambda_3 \cos(f(m, x), g(e)) +
\lambda_4 pop(e)
$$

其中：

| 项 | 含义 | 作用 |
|---|---|---|
| $p(e \mid m)$ | mention-entity 先验 | 某个 mention 历史上最常链接到哪个实体 |
| $BM25(x, e)$ | 稀疏匹配分数 | 根据上下文词面与实体文档是否重合来召回 |
| $\cos(f(m,x), g(e))$ | 向量余弦相似度 | 比较 mention+上下文 与 实体表示 的语义接近程度 |
| $pop(e)$ | 流行度先验 | 对热门实体提供兜底偏置 |
| $\lambda_i$ | 权重 | 控制各路信号贡献大小 |

最后取前 `K` 个：

$$
C_K = \arg top_K \ s_{gen}(e \mid m, x)
$$

先看玩具例子。mention = “Apple”，上下文 = “iPhone, MacBook, chip”。

- alias 字典能召回 `Apple Inc.` 和 `Apple (fruit)`
- redirect 能补 `Apple Computer`
- BM25 会因为 “MacBook”“chip” 这些词，更偏向公司实体
- dense retrieval 会把 “iPhone, MacBook” 映射到消费电子语义邻域
- popularity prior 可能同时给公司和水果较高分，但不能单独决定结果

假设融合后得到：

| 实体 | 先验 | BM25 | 语义相似 | 流行度 | 总分 |
|---|---:|---:|---:|---:|---:|
| Apple Inc. | 0.62 | 1.10 | 0.91 | 0.83 | 0.82 |
| Apple (fruit) | 0.71 | 0.15 | 0.22 | 0.79 | 0.41 |
| Apple Records | 0.05 | 0.08 | 0.19 | 0.21 | 0.18 |

这里“水果”先验可能更高，因为很多通用语料里“apple”经常被解释成水果；但上下文中的产品词把公司实体拉了回来。这就是多路召回的意义：用其他信号修正单一路径的偏差。

再看真实工程例子。知识图谱问答系统中，用户输入：“马斯克旗下自动驾驶公司”。mention 可能是隐式的，甚至不完全等于实体表面名。此时：

- alias 路径未必有直接命中
- BM25 可以从实体简介里匹配到 “自动驾驶”“马斯克”
- dense retrieval 能把整句映射到 `Tesla, Inc.` 的语义邻域
- popularity prior 进一步把明显更常见的候选往前推

如果没有 dense retrieval，这类自由表达查询的召回会明显掉；如果只有 dense retrieval，又可能把“自动驾驶”“公司”语义相近但不正确的实体也拉进来。所以工程上通常不是替代，而是组合。

一个重要推导是：候选规模越大，理论召回率越高，但精排成本近似线性增长。若精排成本记为 `O(K)`，那么总系统成本常常受 `K` 直接控制。因此需要通过离线曲线选阈值，而不是拍脑袋定 `K=50` 或 `K=100`。典型做法是画 `recall@K` 曲线，寻找边际收益开始变平的位置。

---

## 代码实现

代码实现的关键，不是把公式照搬出来，而是把“多路召回、统一去重、融合打分、Top-K 截断”落成稳定可维护的流程。下面用一个可运行的 Python 玩具实现说明整体结构。

```python
from math import log
from difflib import SequenceMatcher

ENTITIES = {
    "apple_inc": {
        "title": "Apple Inc.",
        "aliases": ["apple", "apple computer", "aapl"],
        "redirects": ["apple computers"],
        "doc": "iphone macbook ipad ios chip company technology",
        "popularity": 0.83,
        "embedding_tags": {"iphone", "macbook", "ios", "technology", "company"},
    },
    "apple_fruit": {
        "title": "Apple (fruit)",
        "aliases": ["apple", "apples", "苹果"],
        "redirects": [],
        "doc": "fruit food nutrition tree sweet red green",
        "popularity": 0.79,
        "embedding_tags": {"fruit", "food", "nutrition", "tree"},
    },
    "apple_records": {
        "title": "Apple Records",
        "aliases": ["apple records", "apple"],
        "redirects": [],
        "doc": "music label beatles record company",
        "popularity": 0.21,
        "embedding_tags": {"music", "record", "label", "beatles"},
    },
}

MENTION_PRIOR = {
    "apple": {
        "apple_inc": 0.62,
        "apple_fruit": 0.71,
        "apple_records": 0.05,
    }
}

def normalize(text: str) -> str:
    return " ".join(text.lower().strip().split())

def tokens(text: str) -> set[str]:
    return set(normalize(text).replace(",", " ").split())

def prior_score(mention: str, entity_id: str) -> float:
    p = MENTION_PRIOR.get(normalize(mention), {}).get(entity_id, 1e-6)
    return log(p + 1e-9)

def alias_recall(mention: str) -> set[str]:
    m = normalize(mention)
    hits = set()
    for eid, meta in ENTITIES.items():
        names = meta["aliases"] + meta["redirects"]
        if m in [normalize(x) for x in names]:
            hits.add(eid)
        else:
            # 编辑距离近似：这里用字符串相似度做简化
            if max(SequenceMatcher(None, m, normalize(x)).ratio() for x in names) > 0.85:
                hits.add(eid)
    return hits

def bm25_like_score(context: str, entity_id: str) -> float:
    # 这里不是严格 BM25，只是演示“词面重合越多，分数越高”
    q = tokens(context)
    d = tokens(ENTITIES[entity_id]["doc"])
    overlap = len(q & d)
    return overlap / (len(q) + 1e-9)

def bm25_recall(context: str, topn: int = 3) -> set[str]:
    scored = [(eid, bm25_like_score(context, eid)) for eid in ENTITIES]
    scored.sort(key=lambda x: x[1], reverse=True)
    return {eid for eid, score in scored[:topn] if score > 0}

def dense_score(mention: str, context: str, entity_id: str) -> float:
    # 用标签交集模拟向量语义相似度
    q = tokens(mention) | tokens(context)
    e = ENTITIES[entity_id]["embedding_tags"]
    return len(q & e) / ((len(q) * len(e)) ** 0.5 + 1e-9)

def dense_recall(mention: str, context: str, topn: int = 3) -> set[str]:
    scored = [(eid, dense_score(mention, context, eid)) for eid in ENTITIES]
    scored.sort(key=lambda x: x[1], reverse=True)
    return {eid for eid, score in scored[:topn] if score > 0}

def popularity_score(entity_id: str) -> float:
    return ENTITIES[entity_id]["popularity"]

def generate_candidates(mention: str, context: str, k: int = 2):
    pool = set()
    pool |= alias_recall(mention)
    pool |= bm25_recall(context)
    pool |= dense_recall(mention, context)

    scored = []
    for eid in pool:
        score = (
            0.35 * prior_score(mention, eid) +
            0.25 * bm25_like_score(context, eid) +
            0.30 * dense_score(mention, context, eid) +
            0.10 * popularity_score(eid)
        )
        scored.append((eid, score))

    scored.sort(key=lambda x: x[1], reverse=True)
    return scored[:k]

# 玩具例子
result = generate_candidates("Apple", "I bought a new iPhone and MacBook", k=2)
top_ids = [eid for eid, _ in result]
assert "apple_inc" in top_ids
assert len(result) == 2

# 另一个例子
result2 = generate_candidates("Apple", "This fruit is sweet and rich in nutrition", k=2)
top_ids2 = [eid for eid, _ in result2]
assert "apple_fruit" in top_ids2

print(result)
print(result2)
```

这段代码故意做了几处简化，但流程是完整的：

1. `alias_recall` 负责别名、重定向和近似字符串召回。
2. `bm25_recall` 用词面重合模拟稀疏检索。
3. `dense_recall` 用标签相似模拟向量召回。
4. `generate_candidates` 汇总候选池并统一打分。
5. 最后按分数截断成 `Top-K`。

实现层面最重要的不是“单个打分函数多精致”，而是每路召回都要能单独评估。否则你只会看到总结果变差，却不知道是 alias 漏了，还是向量召回漂了。

不同召回源的工程成本和适用场景大致如下：

| 召回源 | 实现成本 | 在线延迟 | 优点 | 适用场景 |
|---|---|---|---|---|
| alias / redirect | 低 | 低 | 精确、便宜、可解释 | 规范实体名、别名稳定 |
| 编辑距离 | 低 | 低到中 | 可补拼写错误、变体 | 用户输入脏、缩写多 |
| BM25 倒排 | 中 | 低到中 | 词面检索强、成熟稳定 | 实体描述文本丰富 |
| dense retrieval | 中到高 | 中 | 语义召回强，适合自由表达 | 长查询、隐式 mention |
| popularity prior | 低 | 低 | 热门歧义兜底 | 高频实体密集场景 |

真实工程例子里，这段流程会换成更工业化的组件。例如：

- alias/redirect 放在 KV 或 trie 索引
- BM25 用 Lucene、Elasticsearch 或 Tantivy
- dense retrieval 用 ANN 索引，如 FAISS、HNSW
- 候选融合在服务层完成
- 精排只处理前 20 到 100 个候选

如果知识库每天更新，还要把索引构建和知识库版本绑定，否则线上召回会出现“候选里有旧实体、精排特征用的是新实体”的版本撕裂问题。

---

## 工程权衡与常见坑

候选生成最核心的工程矛盾，是召回率和候选规模之间的平衡。`K` 过小会损伤上限，`K` 过大则让精排计算成本线性上升。这个矛盾不能靠感觉解决，要靠离线评估和线上监控。

一个常见经验是画 `recall@K` 曲线。比如：

| K | recall@K |
|---:|---:|
| 1 | 0.72 |
| 5 | 0.88 |
| 10 | 0.93 |
| 20 | 0.96 |
| 50 | 0.975 |
| 100 | 0.981 |

如果从 `K=20` 到 `K=100` 只增加 2.1 个点召回，但精排成本涨了 5 倍，那就要认真判断这是否值得。不同业务答案不同：问答系统可能更重上限，在线推荐系统可能更重延迟。

最常见的坑大致有这些：

| 常见坑 | 现象 | 原因 | 规避方案 |
|---|---|---|---|
| 只靠 alias | 高频热门实体霸榜，长尾真值漏掉 | 字典只会“表面名匹配” | alias 和 dense retrieval 并联 |
| 只靠 BM25 | 缩写、同义表达召回差 | 稀疏检索依赖词面重合 | 补向量召回或别名字典 |
| prior 过强 | 总是推热门实体 | 历史频次被当成硬规则 | 把 prior 当特征，不做硬过滤 |
| `K` 过小 | 精排上限低 | 真值进不来 | 用 `recall@K` 曲线选 `K` |
| `K` 过大 | 精排延迟和成本上升 | 候选集无控制扩张 | 分层截断，先粗排再精排 |
| 索引和 KB 版本不一致 | 线上出现找不到实体、特征错位 | 发布链路未绑定版本 | 索引构建和实体库原子发布 |
| redirect 未维护 | 老名称、新名称互相断开 | 知识库规范化不足 | 建立别名与重定向同步流程 |

再看一个真实工程例子。百科型知识库中，mention “Jordan” 可能指：

- Michael Jordan
- Jordan（国家）
- Air Jordan
- Jordan River

如果只用流行度先验，体育明星往往会长期压过其他实体；但在地理上下文里，“Jordan borders Saudi Arabia” 明显应该是国家。这里的问题不是模型不会算分，而是候选生成如果没有把国家实体放进来，精排就无从修正。

另一个常见坑是每路召回没有单独监控。工程上至少应该记录：

- 每路召回命中率
- 每路召回贡献的独有真值比例
- 多路合并后的总 `recall@K`
- 候选池平均大小、P95 大小
- 精排输入大小与延迟关系

如果没有这些指标，系统变差时你只能知道“整体掉了”，但不知道是 alias 词表过期，还是向量索引更新后语义漂移。

可以用一个简单的统计函数记录每路覆盖：

```python
def route_coverage(sample_truths, route_outputs):
    # sample_truths: list[true_entity_id]
    # route_outputs: dict[str, list[set[str]]]
    metrics = {}
    total = len(sample_truths)
    for route, outputs in route_outputs.items():
        hit = sum(1 for y, cands in zip(sample_truths, outputs) if y in cands)
        metrics[route] = hit / total
    return metrics

truths = ["apple_inc", "apple_fruit"]
routes = {
    "alias": [{"apple_inc", "apple_fruit"}, {"apple_inc", "apple_fruit"}],
    "bm25": [{"apple_inc"}, {"apple_fruit"}],
}
m = route_coverage(truths, routes)
assert m["alias"] == 1.0
assert m["bm25"] == 1.0
```

这类监控不复杂，但非常关键。因为候选生成的问题，常常不是“平均效果差”，而是“某一路原本负责兜底的样本突然全丢了”。

---

## 替代方案与适用边界

候选生成没有唯一正确方案。不同数据规模、知识库质量、资源预算下，最优设计不同。关键不是追求“最先进”，而是选一个与你的业务边界匹配的方案。

先给一个对比表：

| 方案 | 优点 | 缺点 | 适用场景 |
|---|---|---|---|
| 字典召回 | 精确、便宜、实现快 | 覆盖依赖人工维护，泛化差 | 小型知识库、命名规范强 |
| BM25 | 成熟稳定、可解释、对文本实体库友好 | 依赖词面，缩写和语义变体弱 | 实体描述文本丰富 |
| 双塔向量召回 | 语义覆盖强，适合自由表达 | 训练和索引成本更高，可能过召 | 大规模开放域检索 |
| 流行度先验 | 简单有效，热门歧义处理好 | 压制长尾，容易形成偏置 | 高频实体集中场景 |
| 多路融合 | 召回最稳健，能互补 | 系统复杂度最高 | 追求高召回的生产系统 |

如果知识库很小，比如公司内部知识库只有几千个实体，而且名字规范、别名稳定，那么 `alias + redirect` 往往已经足够。因为这时人工维护字典的成本不高，dense retrieval 的额外复杂度未必有回报。

如果场景是中等规模百科或电商实体库，通常 `alias + BM25` 是更稳的起点。alias 负责精确映射，BM25 负责从实体描述、类目、摘要里补词面相关实体。这套方案不一定最强，但工程确定性高。

如果场景是开放域问答、学术搜索、跨语言实体链接，查询表达更自由、实体数量更大，那么只靠词面检索通常不够。这时 dense retrieval 的价值会明显上升，因为它能处理“查询不直接复述实体名字”的情况。

还要注意一个边界：如果你的知识库更新很慢，而且别名体系非常规范，那么字典召回收益高于向量召回；如果查询表达高度自由，经常使用缩写、隐喻、描述性短语，那么语义召回的收益更高。

可以把选择逻辑简化为下面的决策原则：

| 条件 | 优先方案 |
|---|---|
| 数据少、资源少、上线快 | 字典 + redirect |
| 文本描述丰富、需要稳定检索 | 字典 + BM25 |
| 查询自由、长尾多、追求高召回 | BM25 + dense retrieval + prior |
| 热门歧义特别多 | prior 作为特征加入融合 |
| 精排成本敏感、延迟严格 | 控制 `K`，先粗召回再分层截断 |

最后强调一点：流行度先验不是替代检索的方案，它更像一个便宜的偏置项。单独依靠 prior，会让系统越来越偏热门实体；但完全没有 prior，高频歧义又会损失明显。合理用法是把它放进融合打分，而不是拿它当硬过滤器。

---

## 参考资料

1. [BLINK: Zero-shot Entity Linking with Dense Entity Retrieval](https://arxiv.org/abs/1911.03814)
2. [BLINK GitHub Repository](https://github.com/facebookresearch/BLINK)
3. [Lucene BM25Similarity 官方文档](https://lucene.apache.org/core/8_11_2/core/org/apache/lucene/search/similarities/BM25Similarity.html)
4. [MediaWiki Redirects 官方文档](https://www.mediawiki.org/wiki/Help%3ARedirects/en)
5. [Entity Linking Meets Deep Learning: Techniques and Solutions](https://www.researchgate.net/publication/354888745_Entity_Linking_Meets_Deep_Learning_Techniques_and_Solutions)
6. [Neural Entity Linking: A Survey of Models Based on Deep Learning](https://journals.sagepub.com/doi/10.3233/SW-222986)
7. [Entity Linking for Biomedical Literature](https://pmc.ncbi.nlm.nih.gov/articles/PMC4460707/)
