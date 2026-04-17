## 核心结论

多模态记忆的核心不是“把图片和文字分别存起来”，而是把它们编码到同一向量空间。向量空间可以理解为“机器用坐标表示语义相近程度的地方”。像 CLIP 这样的双塔模型，会把图像和文本分别编码成同维度向量，常见是 512 维或 768 维，然后用相似度直接比较。

这件事对视觉 Agent 很重要。Agent 的长期记忆如果只存原始图片和原始文本，后续检索要分两套系统做；如果存联合向量，一套索引就能同时支持“图找图”“文找图”“图找文”“图文混合找记忆”。这比传统分离索引更适合交互式系统，因为查询接口统一，评价指标也统一。

工程上，联合索引通常能提升跨模态召回，但代价是构建更慢、占用更多离线算力。Amazon Visual Search 的公开结果表明，把文本信息纳入训练和索引阶段后，Recall@1/5/10 都有提升；而更激进的跨模态图索引还能继续追求高 Recall，但构建成本会明显上升。换句话说，多模态记忆系统本质上是在用更多离线计算，换更稳定的在线召回。

一个新手版的玩具例子是：把“红裙子图片”和“red dress”都映射成同一类条码。条码在这里就是向量。以后用户上传一张红裙照片，或者直接输入“red dress”，系统都能在同一索引里找到相近商品。检索质量就看正确结果能不能排进前 $K$ 个，也就是 Recall@K。

---

## 问题定义与边界

这里的“多模态记忆”指的是：系统把图像、文本，或者图文对，编码成可检索的向量，并在后续任务中把这些向量当成长期记忆单元。记忆单元可以理解为“机器可复用的最小知识块”。本文讨论的是检索型记忆，不讨论生成式记忆摘要，不讨论视频时序建模，也不讨论多轮对话状态管理。

边界需要先说清楚。

第一，本文默认图像和文本能被映射到同一语义空间。语义空间可以理解为“相同意思会靠近，不同意思会分开”的坐标系。CLIP 的训练目标正是让配对图文更近，不配对图文更远。

第二，本文讨论的是“检索质量”和“索引成本”的平衡，不把问题扩展到端到端任务成败。比如一个视觉 Agent 找错了历史图片，未必立刻导致最终回答错误，但它的记忆层已经失效。

第三，评估指标以 Recall@K 为主。Recall@K 可以理解为“前 K 个结果里找回了多少本该被找回的相关项”。其基本形式是：

$$
Recall = \frac{TP}{TP + FN}
$$

其中 $TP$ 是正确找回的相关项数量，$FN$ 是本该找回但没找回的数量。在多模态检索里，常见看 $Recall@1$、$Recall@5$、$Recall@10$。如果一个记忆系统在 top-10 里连该出现的历史样本都放不进去，后面再做重排、规则过滤，收益也很有限。

可以用一个图书馆类比这个边界：同一本书的封面图和文字简介，不应分在两套完全无关的目录里，而应该共享同一个检索入口。系统最终关心的不是“封面相似度”本身，而是“用户想找的那本书能不能在前几项里出现”。

---

## 核心机制与推导

CLIP 的核心机制是对比学习。对比学习可以理解为“让正确配对更近，让错误配对更远”的训练方式。训练后，图像编码器输出一个向量，文本编码器也输出一个同维度向量，二者可以直接做点积或余弦相似度比较。

为什么归一化很关键？归一化就是把向量长度缩放到 1。这样做以后，点积就等价于余弦相似度，比较的主要是方向而不是长度。方向更接近“语义是否相似”，长度往往会混入模型置信度或噪声，不适合作为主检索信号。

这使得多模态查询可以做代数运算。IMPA 论文给出的形式是：

$$
{\textbf q}=\sum_i \sigma_i w_i {\textbf e}_i
$$

其中，${\textbf e}_i$ 是某个图像或文本的嵌入向量，$\sigma_i \in \{-1,1\}$ 表示这个条件是“加强”还是“减去”，$w_i$ 是权重。直白地说，就是可以把多个意图叠加成一个查询向量。

一个玩具例子如下。

你有三段文字：
- “红色”
- “裙子”
- “去掉正式晚礼服风格”

系统先把三段文字都编码成向量，再做：

$$
q = 1.0 \cdot e(\text{红色}) + 1.0 \cdot e(\text{裙子}) - 0.6 \cdot e(\text{晚礼服})
$$

最后把 $q$ 归一化后去查图库。这样返回的结果会更偏“日常红裙”，而不是“红色礼服”。这个过程本质上不是关键词过滤，而是语义方向上的加减。

如果不想做真正的联合索引，也可以走分索引融合。IMPA 给出一个简单形式：

$$
d_i = \frac{\alpha d_i^{\mathcal V} + \beta d_i^{\mathcal T}}{2}
$$

这里 $d_i^{\mathcal V}$ 是视觉索引里的距离，$d_i^{\mathcal T}$ 是文本索引里的距离，$\alpha,\beta$ 控制两边权重。它的意义很直接：先分别查，再在结果层融合。这样实现更轻，但跨模态对齐通常不如端到端联合训练和联合索引稳定。

下面这张表把几种方案的机制和成本放在一起：

| 方案 | 向量空间是否统一 | 是否支持图文混合查询 | 索引构建成本 | 跨模态检索效果 |
|---|---|---:|---:|---:|
| 分离索引 | 否 | 弱，靠后融合 | 低 | 一般 |
| 统一编码 + 单索引 | 是 | 强 | 中 | 好 |
| 统一编码 + 联合训练 + 联合索引 | 是 | 很强 | 高 | 最好 |
| 高级图结构索引 | 是 | 很强 | 很高 | 高 Recall 场景最好 |

真实工程例子是电商视觉搜索。Amazon 公布的结果显示，把文本信号纳入模型和索引后，图到多模态、以及多模态搜索任务的 Recall@1/5/10 都有提升。例如表 3 中，ViT-g/14 + BERT(354M) 的 4-tower 模型在 multimodal retrieval 上达到 0.61 / 0.78 / 0.82，继续用 3-tower 权重初始化微调后可到 0.64 / 0.82 / 0.86。这说明图像查询不是只靠视觉纹理就够，标题、短文本重写、类目语义会显著影响召回。

---

## 代码实现

下面给一个可运行的最小实现。它不依赖真实 CLIP 权重，而是演示“联合向量存储 + 混合查询 + Recall@K 评估”的核心结构。思路和真实系统一致：离线把图像和文本编码成统一向量并入库，在线把查询编码成向量并做 top-k 检索。

```python
import math

def l2_norm(v):
    return math.sqrt(sum(x * x for x in v))

def normalize(v):
    n = l2_norm(v)
    assert n > 0
    return [x / n for x in v]

def dot(a, b):
    assert len(a) == len(b)
    return sum(x * y for x, y in zip(a, b))

def combine(weighted_vectors):
    acc = [0.0] * len(weighted_vectors[0][1])
    for w, vec in weighted_vectors:
        for i, x in enumerate(vec):
            acc[i] += w * x
    return normalize(acc)

def topk_search(query_vec, memory_db, k=3):
    scored = []
    for item_id, vec, payload in memory_db:
        scored.append((dot(query_vec, vec), item_id, payload))
    scored.sort(reverse=True)
    return scored[:k]

def recall_at_k(results, relevant_ids, k):
    top_ids = {item_id for _, item_id, _ in results[:k]}
    tp = len(top_ids & set(relevant_ids))
    fn = len(set(relevant_ids) - top_ids)
    return tp / (tp + fn) if (tp + fn) else 0.0

# 玩具“共享语义空间”
red = normalize([1.0, 0.0, 0.0])
dress = normalize([0.0, 1.0, 0.0])
formal = normalize([0.0, 0.0, 1.0])

memory_db = [
    ("img_red_dress_casual", combine([(1.0, red), (1.0, dress), (-0.2, formal)]), {"type": "image"}),
    ("img_red_dress_formal", combine([(1.0, red), (1.0, dress), (0.8, formal)]), {"type": "image"}),
    ("txt_red_dress", combine([(1.0, red), (1.0, dress)]), {"type": "text"}),
]

# 混合查询：红色 + 裙子 - 正式
query = combine([(1.0, red), (1.0, dress), (-0.6, formal)])
results = topk_search(query, memory_db, k=3)

assert results[0][1] in {"img_red_dress_casual", "txt_red_dress"}
r_at_2 = recall_at_k(results, relevant_ids={"img_red_dress_casual", "txt_red_dress"}, k=2)
assert abs(r_at_2 - 1.0) < 1e-9

print(results)
print("Recall@2 =", r_at_2)
```

把这段逻辑换成真实工程实现时，变化主要有三点。

第一，`combine` 前面的向量不再是手工写死，而是由图像编码器和文本编码器生成。真实系统常见做法是离线批量编码商品图、截图、OCR 文本、标题、摘要等，然后写入向量数据库或 OpenSearch 的 `knn_vector` 字段。

第二，`topk_search` 不再是全表扫描，而是近似最近邻索引。近似最近邻可以理解为“用更快但不保证绝对最优的搜索结构换速度”。HNSW 是常见选择，因为在线快、工程成熟。

第三，记忆单元不能只存向量，通常还要存 `id`、来源、时间戳、模态类型、权限信息、原始载荷地址。否则你只能找到“像什么”，不能知道“它是谁、能不能返回给用户”。

一个面向真实 Agent 的最小落地流程通常是：

| 阶段 | 输入 | 输出 | 关键动作 |
|---|---|---|---|
| 离线构建 | 图片、文本、元数据 | 向量索引 | 批量编码、归一化、入库 |
| 在线查询 | 文本、图片或两者都有 | top-k 候选记忆 | 编码查询、向量检索 |
| 在线后处理 | top-k 候选 | 最终结果 | 规则过滤、重排、去重 |
| 反馈回写 | 点击、采纳、纠错 | 更新权重或索引 | relevance feedback、增量刷新 |

---

## 工程权衡与常见坑

第一个坑是把“统一编码”误当成“问题已经解决”。统一编码只解决了比较接口统一，不代表召回已经足够好。很多系统上线后发现，图像相似但语义不对，比如都含“红色局部区域”却不是同一类商品。Amazon 的论文正是针对这类局部视觉误匹配，引入额外图文对齐来约束模型。

第二个坑是忽视索引构建成本。RoarGraph 这类跨模态图索引在高 Recall 区间很强，但论文里明确给出，构建时间可达 HNSW 的 4.8 到 17.5 倍。对频繁上新的商品库、媒体库、用户相册库，这不是小优化，而是部署节奏问题。你如果每天都重建全量索引，离线链路会先成为瓶颈。

第三个坑是把 Recall 指标看成单点数字。Recall@10 提升不等于业务一定收益更大，因为结果列表里可能混入重复项、近重复项，或者对用户无解释性的样本。多模态记忆系统最好同时跟踪：
- Recall@K
- 去重后的 Recall@K
- 首条命中率
- 查询延迟
- 索引更新时间

第四个坑是负向约束难做。公式里虽然可以写 $-e(\text{交通})$、$-e(\text{水印})$，但现实里负向语义不一定稳定。原因是训练时模型主要学“谁像谁”，不一定学得足够好“谁不该像谁”。所以减法查询更适合做细调，不适合替代硬过滤。

第五个坑是记忆污染。Agent 如果把低质量 OCR、错误标题、用户临时输入都直接并入长期记忆，会把向量空间拉偏。新手常犯的错误是“凡是出现过的都存”。更合理的做法是分层：
- 工作记忆：短期、可丢弃
- 长期记忆：高置信、已去噪、可检索
- 事实缓存：结构化字段，单独存

---

## 替代方案与适用边界

如果你的数据规模不大、更新很频繁，最实用的方案通常不是一步到位上联合图文索引，而是“分模态索引 + 结果融合”。这类方案的好处是简单、稳定、可灰度。图库走 HNSW，文本走 BM25 或文本向量索引，最后按权重融合。它不一定是最优，但很适合 MVP 和快速试错。

如果你的场景是高价值检索，比如电商主搜索、版权匹配、视频素材检索，联合编码和联合索引更值得做，因为一次召回失败的成本更高。这时可以接受更重的离线预处理，并配套定期重建、冷热分层、采样更新。

如果你的场景要求快速增量更新，而不是极致 Recall，可以优先选 HNSW 一类成熟 ANN 结构，必要时再叠加轻量重排。相比追求最强图结构，先把“能稳定更新、能稳定回滚、能监控退化”做好，收益往往更大。

最重要的边界是：联合向量存储适合“语义相关性”为主的记忆，不适合替代精确事实库。比如“这张图像像不像某件商品”适合向量检索；但“这个订单创建于哪一天”应该查结构化数据库。把两类问题混在一起，是很多 Agent 记忆系统后期难维护的根源。

---

## 参考资料

- CLIP 背景与共享向量空间：[Contrastive Language-Image Pre-training](https://en.wikipedia.org/wiki/Contrastive_Language-Image_Pre-training)
- Amazon KDD’24 论文：[Bringing multimodality to Amazon visual search system](https://assets.amazon.science/57/3b/f549e60d4bce8179c81bebdf03bb/bringing-multimodality-to-amazon-visual-search-system.pdf)
- 多模态检索综述与 Recall 指标：[A Survey of Full-Cycle Cross-Modal Retrieval](https://www.mdpi.com/2076-3417/13/7/4571)
- 向量代数组合与融合公式：[Multimodal video retrieval with CLIP: a user study](https://link.springer.com/article/10.1007/s10791-023-09425-2)
- 工程实现示例：[Implement unified text and image search with a CLIP model using Amazon SageMaker and Amazon OpenSearch Service](https://aws.amazon.com/blogs/machine-learning/implement-unified-text-and-image-search-with-a-clip-model-using-amazon-sagemaker-and-amazon-opensearch-service/)
- 高 Recall 图索引权衡：[RoarGraph VLDB 2024](https://kay21s.github.io/RoarGraph-VLDB2024.pdf)
- 跨模态检索综述说明：[Bridging Modalities: A Survey of Cross-Modal Image-Text Retrieval](https://www.icck.org/article/html/cjif.2024.361895)
