## 核心结论

冷启动物品的核心问题不是“没有 ID”，而是“没有交互历史”。如果推荐系统只依赖 `item_id -> embedding` 的查表式学习，新物品上线时即使有合法的 `item_id`，也无法学到可靠向量，因为这个 ID 没有点击、加购、购买、停留等行为信号。

Embedding ID 化的替代方案，是把物品向量 $e_i$ 的来源从“历史行为学出来的 ID 表”迁移到“上线前就能获得的信息”。这些信息包括内容特征、图结构邻居、预训练编码器输出。最终推荐打分仍然可以保持统一形式：

$$
s(u,i) = p_u^\top e_i
$$

其中 $p_u$ 是用户向量，$e_i$ 是物品向量。关键变化是：$e_i$ 不再只由 `item_id` 查表得到，而是可以由内容、邻居或预训练模型生成。

真实工程例子：电商新 SKU 上线首日没有点击，但有标题、类目、品牌、价格、主图。系统可以先用这些信息生成 $e_i$，让它进入召回池；等后续交互积累后，再让 ID Embedding 补充或逐步接管。

---

## 问题定义与边界

物品侧冷启动，是指新物品已经进入系统，但还没有足够用户交互，导致推荐模型无法从历史行为中学习它的表示。这里的“表示”就是 embedding，也就是一段定长数字向量，用来让模型计算相似度或打分。

本文讨论的前提是：新物品没有交互信号，但有侧信息。侧信息是指不依赖用户行为、上线前就能拿到的信息，例如标题、类目、品牌、价格、图片、商家、标签、上下架时间、相似商品关系。

本文不讨论三类问题：

| 问题类型 | 是否本文重点 | 原因 |
|---|---:|---|
| 物品侧冷启动 | 是 | 新物品有内容和属性，但没有行为 |
| 用户侧冷启动 | 否 | 用户缺少历史行为，需要建模用户画像或上下文 |
| 全系统冷启动 | 否 | 用户和物品都缺数据，需要种子数据、规则或人工运营 |
| 排序全链路优化 | 否 | 本文重点是新物品向量如何生成并进入召回 |

电商例子很典型：一个新商品有标题“男士轻量跑步鞋”、类目“运动鞋”、品牌、价格、主图，但没有点击、加购、购买记录。此时如果模型只会查 `item_id` 的 embedding，这个物品要么没有向量，要么只有随机初始化向量，召回质量很低。系统必须先依赖内容或结构信号生成表示。

几种输入来源的区别如下：

| 表示来源 | 依赖输入 | 适合场景 | 主要风险 |
|---|---|---|---|
| ID Embedding | 历史交互 | 成熟物品、行为充分 | 新物品没有可靠参数 |
| 内容特征 | 标题、类目、品牌、价格、图片 | 新 SKU 首发、内容完整 | 内容质量差会影响表示 |
| 图邻居 | 相似商品、同店商品、共现关系 | 商品关系丰富的平台 | 图泄漏、过度平滑 |
| 预训练向量 | 文本模型、图像模型、多模态模型 | 文本或图片语义强 | 推理成本高、空间不对齐 |

---

## 核心机制与推导

统一记号如下：

| 符号 | 含义 | 白话解释 |
|---|---|---|
| $x_i$ | 物品内容特征 | 标题、类目、品牌、价格等原始输入 |
| $N(i)$ | 物品 $i$ 的邻居集合 | 与该物品相似或相关的其他物品 |
| $m_i$ | 预训练模型输出 | 文本或图片模型产生的语义向量 |
| $e_i$ | 最终物品向量 | 推荐系统实际用于召回或排序的向量 |
| $p_u$ | 用户向量 | 用户兴趣的向量表示 |

三条主路线都在做同一件事：把不同输入映射到推荐空间。推荐空间是指用户向量和物品向量可以直接做内积、余弦相似度或其他打分的向量空间。

第一条路线是内容型：

$$
e_i = f_\theta(x_i)
$$

$f_\theta$ 是一个可训练编码器，可以是 MLP、Wide & Deep、Transformer 小模型，也可以是多个特征塔的组合。输入 $x_i$ 可以包括标题文本向量、类目 ID、品牌 ID、价格分桶、图片向量。它的目标是学习“什么样的内容容易被什么样的用户喜欢”。

第二条路线是图结构型：

$$
e_i = \sigma(W[x_i \Vert \operatorname{mean}_{j \in N(i)} e_j])
$$

$\Vert$ 表示向量拼接，$\sigma$ 是非线性函数，例如 ReLU。图结构型方法认为，一个新物品虽然没有自己的交互，但它可能有邻居：同类目商品、同品牌商品、同店商品、图片相似商品、运营维护的替代品。邻居平均可以补足局部语义。GraphSAGE 和 PinSage 都属于这类思路，区别在于采样、聚合和大规模工程实现不同。

第三条路线是预训练型：

$$
e_i = P(h_i^{LM})
$$

$h_i^{LM}$ 是语言模型或多模态模型输出的向量，$P$ 是投影层。投影层是一段小模型，用来把预训练语义空间对齐到推荐空间。预训练模型知道“轻量跑步鞋”和“缓震运动鞋”语义接近，但它并不知道平台用户的点击偏好，所以还需要 $P$ 做适配。

完整链路可以写成：

$$
\text{input}_i \rightarrow \text{encoder} \rightarrow e_i \rightarrow s(u,i)=p_u^\top e_i
$$

玩具例子：设新物品内容特征 $x=[1,0,1]$，内容映射矩阵为：

$$
W=
\begin{bmatrix}
0.5 & 0.1 & 0.0 \\
0.2 & 0.3 & 0.4
\end{bmatrix}
$$

则内容分支得到：

$$
e_c = Wx = [0.5, 0.6]
$$

图结构分支中，两个邻居向量分别是 $[0.3,0.9]$ 和 $[0.7,0.5]$，均值为：

$$
e_g = [0.5,0.7]
$$

预训练分支中，文本编码器输出 $h^{LM}=[0.48,0.64]$，投影后暂设不变：

$$
e_p=[0.48,0.64]
$$

用户向量 $p_u=[0.2,0.8]$，则三种打分为：

| 分支 | 物品向量 | 打分 |
|---|---|---:|
| 内容分支 | $[0.5,0.6]$ | $0.2*0.5+0.8*0.6=0.58$ |
| 图分支 | $[0.5,0.7]$ | $0.2*0.5+0.8*0.7=0.66$ |
| 预训练分支 | $[0.48,0.64]$ | $0.2*0.48+0.8*0.64=0.608$ |

这个例子说明：打分框架不变，变化的是 $e_i$ 的生成方式。

---

## 代码实现

工程实现要拆成三个层次：特征读取、向量生成、召回打分。不要把所有逻辑塞进一个大模型，因为冷启动链路通常需要离线预计算、缓存、回滚和灰度。

新物品上线时的最小流程是：

| 阶段 | 职责 | 输出 |
|---|---|---|
| 离线训练 | 用历史数据训练内容编码器、图聚合器、投影层 | 模型参数 |
| 离线预计算 | 对新物品批量生成内容向量、图向量、预训练向量 | 物品向量库 |
| 线上召回 | 读取用户向量，与候选物品向量做近邻检索或内积打分 | 候选列表 |
| 在线更新 | 新交互进入后更新 ID Embedding 或融合权重 | 更稳定的物品表示 |

输入字段处理可以按下表设计：

| 字段 | 处理方式 | 是否适合预计算 |
|---|---|---:|
| 标题 | tokenizer 后送文本编码器 | 是 |
| 类目 | 类目 ID embedding 或 one-hot | 是 |
| 品牌 | 品牌 ID embedding | 是 |
| 价格 | 标准化或分桶 | 是 |
| 主图 | 图像编码器提取向量 | 是 |
| 邻居 ID | 采样后聚合邻居向量 | 是 |

下面是一段可运行的简化 Python。它保留了三个核心函数：`encode_content`、`aggregate_neighbors`、`score`。

```python
import numpy as np

def encode_content(x, W):
    """把内容特征映射成物品向量。"""
    return W @ x

def aggregate_neighbors(neighbor_vectors):
    """用邻居平均生成图结构向量。"""
    if len(neighbor_vectors) == 0:
        return None
    return np.mean(np.array(neighbor_vectors, dtype=float), axis=0)

def score(user_vector, item_vector):
    """用户向量和物品向量做内积打分。"""
    return float(np.dot(user_vector, item_vector))

def fuse_vectors(content_vec, graph_vec=None, pretrained_vec=None, weights=None):
    vectors = [content_vec]
    default_weights = [1.0]

    if graph_vec is not None:
        vectors.append(graph_vec)
        default_weights.append(1.0)

    if pretrained_vec is not None:
        vectors.append(pretrained_vec)
        default_weights.append(1.0)

    if weights is None:
        weights = np.array(default_weights, dtype=float)
    else:
        weights = np.array(weights, dtype=float)

    weights = weights / weights.sum()
    output = np.zeros_like(content_vec, dtype=float)

    for w, v in zip(weights, vectors):
        output += w * v

    return output

x = np.array([1.0, 0.0, 1.0])
W = np.array([
    [0.5, 0.1, 0.0],
    [0.2, 0.3, 0.4],
])

content_vec = encode_content(x, W)
graph_vec = aggregate_neighbors([
    np.array([0.3, 0.9]),
    np.array([0.7, 0.5]),
])
pretrained_vec = np.array([0.48, 0.64])

user_vec = np.array([0.2, 0.8])

assert np.allclose(content_vec, np.array([0.5, 0.6]))
assert np.allclose(graph_vec, np.array([0.5, 0.7]))
assert abs(score(user_vec, content_vec) - 0.58) < 1e-9
assert abs(score(user_vec, graph_vec) - 0.66) < 1e-9
assert abs(score(user_vec, pretrained_vec) - 0.608) < 1e-9

final_vec = fuse_vectors(content_vec, graph_vec, pretrained_vec, weights=[0.4, 0.4, 0.2])
assert final_vec.shape == (2,)
assert score(user_vec, final_vec) > 0
```

真实工程中，标题编码、主图编码、图邻居聚合不应在每次线上请求里实时执行。更常见的做法是：新物品入库后触发离线任务，生成向量并写入向量库；线上召回只读取已经算好的 $e_i$。这样才能控制延迟和成本。

---

## 工程权衡与常见坑

冷启动方案不是越复杂越好，而是要覆盖足够信息，同时保证线上可用、可控、可缓存。一个昂贵但不可缓存的方案，可能离线指标很好，线上无法稳定服务。

常见问题如下：

| 问题 | 后果 | 规避 |
|---|---|---|
| 特征太稀疏 | 新物品向量缺少区分度 | 补齐标题、类目、品牌、图片等强特征 |
| 训练和线上清洗不一致 | 表示漂移，线上效果下降 | 统一 tokenizer、清洗规则、缺失值处理 |
| 图结构使用未来边 | 离线指标虚高 | 严格按时间截断构图 |
| 热门邻居占比过高 | 冷门物品被热门语义带偏 | 邻居采样、度归一化、残差连接 |
| 预训练模型实时推理 | 延迟高、成本高 | 离线预计算、蒸馏、小投影头、缓存 |
| 只看内容相似 | 推荐结果同质化 | 融合用户偏好、业务规则和探索策略 |

例子 1：只有标题没有主图时，文本编码器会变弱。标题“夏季新款”信息量很低，无法区分鞋、衣服、包。此时需要补齐类目、品牌、价格、主图特征。

例子 2：训练时使用清洗后的标题，线上却直接用商家原始脏标题，例如包含重复词、营销词、特殊符号，会导致编码分布变化。模型学到的是干净输入，线上看到的是另一个输入空间。

例子 3：图结构如果把未来点击边纳入邻居，离线评估会提前看到答案。比如用 4 月 10 日之后的共现关系去训练 4 月 1 日上线商品的冷启动表示，离线指标会虚高，线上无法复现。

配置上要显式约束时间切分、邻居采样和缓存策略：

```python
config = {
    "train_end_time": "2026-04-01 00:00:00",
    "graph_edge_max_time": "2026-04-01 00:00:00",
    "neighbor_sample_size": 20,
    "exclude_future_edges": True,
    "precompute_item_vectors": True,
    "vector_cache_ttl_seconds": 86400,
    "realtime_pretrained_encoding": False,
}

assert config["exclude_future_edges"] is True
assert config["graph_edge_max_time"] <= config["train_end_time"]
assert config["realtime_pretrained_encoding"] is False
```

性能约束必须提前设计。预训练文本模型和多模态模型通常延迟高，不能在每次召回请求里跑完整模型。工程上通常使用四种手段：离线预计算、向量缓存、蒸馏到小模型、只在线上运行轻量投影头。

---

## 替代方案与适用边界

Embedding ID 化不是唯一解，也不是要完全废弃 ID Embedding。更准确的说法是：冷启动阶段不能只依赖 ID Embedding；当交互充分后，ID Embedding 仍然是高效且强大的表示方式。

几种方案的适用边界如下：

| 方案 | 优点 | 缺点 | 适用阶段 |
|---|---|---|---|
| 纯 ID Embedding | 简单、高效、容易服务化 | 新物品无法学习可靠向量 | 成熟物品 |
| 内容向量召回 | 新物品上线即可生成表示 | 依赖内容质量 | 新 SKU 首发 |
| 图结构补全 | 能利用相似商品和局部关系 | 容易图泄漏、过度平滑 | 关系丰富的平台 |
| 预训练编码器 | 文本、图片语义强 | 成本高，需要对齐推荐空间 | 内容强、多模态强场景 |
| 混合方案 | 覆盖更全面，稳定性更好 | 系统复杂度更高 | 大多数生产系统 |

什么时候应该回退到 ID Embedding？当物品已经积累足够交互，并且行为信号稳定时，ID Embedding 可以更直接地刻画平台内真实偏好。内容上相似的两个商品，用户反馈可能完全不同，原因可能是价格、库存、评价、品牌心智、履约体验。纯内容向量很难捕捉这些行为差异。

什么时候应该融合而不是替代？在新物品从冷启动进入成熟阶段的过程中，内容向量、图向量、预训练向量和 ID Embedding 可以按权重融合。例如：

$$
e_i = \alpha e_i^{ID} + \beta e_i^{content} + \gamma e_i^{graph} + \delta e_i^{pretrain}
$$

早期让 $\beta,\gamma,\delta$ 更大；交互增加后逐步提高 $\alpha$。这样可以避免上线首日没有表示，也可以避免成熟后还被静态内容限制。

最佳实践通常是混合方案，而不是单一路线：内容向量解决“上线即有表示”，图结构解决“相似关系补全”，预训练解决“语义理解”，ID Embedding 解决“行为反馈沉淀”。

---

## 参考资料

1. [GraphSAGE: Inductive Representation Learning on Large Graphs](https://proceedings.neurips.cc/paper_files/paper/2017/hash/5dd9db5e033da9c6fb5ba83c7a7ebea9-Abstract.html)：图结构方向，用邻居采样和聚合为未见节点生成表示。
2. [Graph Convolutional Neural Networks for Web-Scale Recommender Systems](https://huggingface.co/papers/1806.01973)：PinSage 论文，用图卷积和随机游走解决大规模推荐召回中的物品表示问题。
3. [NFC: a deep and hybrid item-based model for item cold-start recommendation](https://link.springer.com/article/10.1007/s11257-021-09303-w)：内容编码方向，讨论混合式 item cold-start 推荐。
4. [Language-Model Prior Overcomes Cold-Start Items](https://dblp.org/rec/journals/corr/abs-2411-09065.html)：预训练方向，讨论语言模型先验如何缓解冷启动物品问题。
