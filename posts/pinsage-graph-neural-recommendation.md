## 核心结论

PinSage 是 Pinterest 为超大规模推荐图设计的图神经网络推荐方法。它的目标不是学习一个适用于所有图任务的通用表示，而是为推荐系统里的物品节点学习稳定的 item embedding，也就是把每个物品压成一个可用于相似度检索的向量。

在 Pinterest 场景里，一个 pin 的相似内容不是靠全图遍历得到，而是在 pin-board 二部图里根据“更容易从它走到哪些 pin”来决定表示。二部图是指节点分成两类，边只连接不同类型节点；这里一类是 pin，另一类是 board，pin 被用户收藏到 board 上，就形成一条边。

PinSage 的关键改造有两点：第一，不做全邻居扩张，而是用短随机游走找重要邻居；第二，不把邻居平均聚合，而是按随机游走访问次数加权聚合。随机游走是指从一个节点出发，按边随机移动若干步；访问次数越高，说明这个邻居在图结构上越重要。

| 对比项 | GraphSAGE | PinSage |
|---|---|---|
| 邻居选择方式 | 固定采样一批直接邻居或多跳邻居 | 对目标 pin 做短随机游走，按访问次数选 Top-T |
| 聚合方式 | mean、pooling、LSTM 等通用聚合 | Importance Pooling，按 visit count 加权 |
| 适用场景 | 一般图表示学习、节点分类、链路预测 | 超大规模 item 推荐召回 |
| 计算代价 | 多跳展开时容易邻域爆炸 | 固定重要邻居数量，计算更可控 |

PinSage 的总公式链可以概括为：

$$
\text{随机游走计数}
\rightarrow
\text{邻域筛选}
\rightarrow
\text{加权聚合}
\rightarrow
\text{表示更新}
\rightarrow
\text{排序训练}
$$

---

## 问题定义与边界

PinSage 解决的问题是：给定 pin-board 二部图、pin 的内容特征和历史共现关系，学习每个 pin 的向量表示，用于相关推荐。内容特征可以来自图片、文本、类别、描述等信息；历史共现关系来自用户把哪些 pin 收藏到同一个 board 上。

Pinterest 的真实工程例子是 Related Pins、Shopping、Ads 等场景。输入是图片、文本和共现关系，输出是 pin embedding；下游系统用这些 embedding 做近邻召回，再交给排序模型做更细的 rerank。近邻召回是指在海量候选里先找出向量最相近的一小批 item，rerank 是指再用更复杂模型重新排序。

| 项目 | 定义 |
|---|---|
| 输入 | pin-board 图结构、pin 内容特征、历史收藏或共现关系 |
| 输出 | 每个 pin 的向量表示 |
| 下游 | ANN 近邻召回、Related Pins、Shopping、Ads rerank |
| 不解决 | 完整推荐决策、复杂用户序列建模、纯内容之外的冷启动问题 |

二部图可以简化表示为：

```text
pin_a  --- board_1 --- pin_b
pin_a  --- board_2 --- pin_c
pin_d  --- board_2 --- pin_c
pin_e  --- board_3 --- pin_b
```

如果 `pin_a` 和 `pin_b` 经常出现在相同或相近的 board 上，它们在图上会更容易互相到达，因此更可能被认为语义相关。这个关系不是简单的文本相似，也不是单张图片相似，而是用户组织内容时留下的结构信号。

边界也要说清楚：PinSage 不是完整推荐系统。它主要解决“图驱动的 item 表示学习”。线上推荐还需要召回服务、特征服务、排序模型、过滤规则、业务约束和实时反馈。PinSage 输出的是可检索的向量，不是最终展示列表。

---

## 核心机制与推导

PinSage 的第一步是采样邻居。对目标节点 \(u\) 做多次短随机游走，统计每个候选 pin 被访问的次数 \(c_{uv}\)。然后把访问次数做 L1 归一化，得到邻居权重：

$$
\alpha_{uv}=\frac{c_{uv}}{\sum_w c_{uw}}
$$

这里 \(c_{uv}\) 是从 \(u\) 出发后访问到 \(v\) 的次数，\(\alpha_{uv}\) 是归一化后的重要性权重。L1 归一化是指让所有权重加起来等于 1。论文中说明，这种 visit count 的 L1 归一化在随机游走次数足够多时，近似 Personalized PageRank。Personalized PageRank 是一种从指定节点出发衡量其他节点相关性的图算法。

邻域筛选公式是：

$$
N(u)=\mathrm{TopT}_v(\alpha_{uv})
$$

也就是只保留权重最高的 \(T\) 个邻居，避免全邻居展开。Top-T 是指按分数排序后取前 \(T\) 个元素。

第二步是聚合邻居。先对邻居表示做线性变换和 ReLU，再按权重加权求和：

$$
n_u=\sum_{v\in N(u)}\alpha_{uv}\,\mathrm{ReLU}(Qh_v+q)
$$

其中 \(h_v\) 是邻居节点 \(v\) 的当前表示，\(Q\) 和 \(q\) 是可学习参数，ReLU 是把负数截断为 0 的非线性函数。

第三步是更新目标节点表示。把自身表示 \(z_u\) 和邻居聚合结果 \(n_u\) 拼接，再经过线性变换、激活函数和 L2 归一化：

$$
z'_u=\mathrm{norm}_2(\mathrm{ReLU}(W[z_u;n_u]+w))
$$

L2 归一化是指把向量长度缩放为 1，让后续点积或余弦相似度更稳定。

训练目标使用 margin loss，让目标节点 \(u\) 更接近正样本 \(i\)，远离负样本 \(n\)：

$$
\mathcal L=\mathbb E_{n\sim P_n(u)}\max(0,z_u\cdot z_n-z_u\cdot z_i+\Delta)
$$

其中 \(z_u\cdot z_i\) 是向量点积，表示相似度；\(\Delta\) 是间隔超参数。hard negative 是看起来相似但不是正样本的负例，它比随机负样本更能逼模型学到细粒度差异。

玩具例子如下。假设随机游走只访问到两个邻居，计数为 \(c=[3,1]\)，则权重为：

$$
\alpha=[0.75,0.25]
$$

若两个邻居变换后的特征分别是 \(2\) 和 \(4\)，聚合结果是：

$$
n_u=0.75\times2+0.25\times4=2.5
$$

若自身表示 \(z_u=1\)，示意性拼接为 \([1,2.5]\)，做 L2 归一化后得到约：

$$
[0.37,0.93]
$$

这说明更常被随机游走访问到的邻居，对最终表示影响更大。

为什么不是均匀采样？假设一个 pin 有 100 个候选邻居，其中 5 个来自高度相关的 board，95 个只是弱共现。如果均匀采样，弱相关邻居容易占多数，表示会被拉平。随机游走访问次数会让强相关邻居自然获得更高权重。

机制流程可以表示为：

```text
目标 pin
  -> 短随机游走
  -> 访问计数归一化
  -> Top-T 重要邻居
  -> Importance Pooling
  -> 表示更新与 L2 normalize
  -> 正负样本排序训练
```

---

## 代码实现

工程实现不需要一次复现全部论文细节，但要保持流水线完整：图采样、邻居聚合、训练目标。下面是一个可运行的极简 Python 例子，用小图模拟 PinSage 的核心逻辑。

```python
import random
import math
from collections import Counter

def random_walk_sampling(u, graph, num_walks, walk_length, seed=0):
    random.seed(seed)
    counts = Counter()
    for _ in range(num_walks):
        cur = u
        for _ in range(walk_length):
            neigh = graph.get(cur, [])
            if not neigh:
                break
            cur = random.choice(neigh)
            if cur.startswith("pin_") and cur != u:
                counts[cur] += 1
    return counts

def top_t_neighbors(counts, T):
    return dict(counts.most_common(T))

def normalize_counts(counts):
    total = sum(counts.values())
    return {k: v / total for k, v in counts.items()} if total else {}

def relu(x):
    return max(0.0, x)

def aggregate_neighbors(features, weights):
    # 简化版：邻居特征是一维数值，Q=1, q=0
    return sum(weights[v] * relu(features[v]) for v in weights)

def l2_normalize(vec):
    norm = math.sqrt(sum(x * x for x in vec))
    return [x / norm for x in vec]

def update_embedding(self_emb, neigh_emb):
    # 简化版：拼接后直接 ReLU + L2 normalize
    return l2_normalize([relu(self_emb), relu(neigh_emb)])

def dot(a, b):
    return sum(x * y for x, y in zip(a, b))

def margin_loss(anchor, positive, negatives, margin):
    losses = []
    for neg in negatives:
        losses.append(max(0.0, dot(anchor, neg) - dot(anchor, positive) + margin))
    return sum(losses) / len(losses)

graph = {
    "pin_a": ["board_1", "board_2"],
    "board_1": ["pin_a", "pin_b"],
    "board_2": ["pin_a", "pin_b", "pin_c"],
    "pin_b": ["board_1", "board_2"],
    "pin_c": ["board_2"],
}

features = {"pin_b": 2.0, "pin_c": 4.0}

counts = {"pin_b": 3, "pin_c": 1}
weights = normalize_counts(counts)
neigh = aggregate_neighbors(features, weights)
emb = update_embedding(1.0, neigh)

assert weights == {"pin_b": 0.75, "pin_c": 0.25}
assert abs(neigh - 2.5) < 1e-9
assert abs(emb[0] - 0.3713906763) < 1e-6
assert abs(emb[1] - 0.9284766909) < 1e-6

anchor = emb
positive = l2_normalize([0.4, 0.9])
negative = l2_normalize([0.9, 0.1])
loss = margin_loss(anchor, positive, [negative], margin=0.2)
assert loss >= 0.0
```

最小伪代码流程如下：

```python
for u, positive_i in training_batch:
    counts = random_walk_sampling(u, graph, num_walks, walk_length)
    neighbors = top_t_neighbors(counts, T)
    weights = normalize_counts(neighbors)

    neigh_emb = aggregate_neighbors(features, weights)
    z_u = update_embedding(self_emb[u], neigh_emb)

    negatives = sample_hard_negatives(u)
    loss = margin_loss(z_u, z_i, negatives, margin)
    optimizer.step(loss)
```

训练循环可以拆成：

| 阶段 | 工程动作 |
|---|---|
| 输入批次 | 取一批 anchor pin 和正样本 pin |
| 采样邻居 | 对每个 anchor 做短随机游走，取 Top-T |
| 前向传播 | 聚合邻居特征，更新 embedding |
| 计算损失 | 用正样本和 hard negatives 计算 margin loss |
| 反向更新 | 更新 GNN 参数和特征变换参数 |

真实系统通常不会每次线上请求都跑 GNN。更常见的做法是离线批量计算 pin embedding，写入向量库；线上请求时用 ANN 检索找近邻。ANN 是 approximate nearest neighbor，意思是用近似算法在大规模向量集合里快速找相似向量。这样可以把复杂训练留在离线，把线上延迟控制在毫秒级。

---

## 工程权衡与常见坑

PinSage 的核心工程权衡是：牺牲“完整使用所有邻居”的理论干净性，换取可训练、可扩展、可上线的系统。对亿级节点图来说，全 \(k\)-hop 扩张会让邻居数量指数增长。假设每个节点平均有 100 个邻居，两跳就是约 \(100^2\)，三跳就是约 \(100^3\)。这在显存、训练时间和数据加载上都不可接受。

固定 \(T\) 个重要邻居可以控制计算量。即使原图里一个 pin 连接到大量 board，模型每层也只处理有限邻居。这个设计让 GNN 从研究原型变成工程系统。

| 问题 | 后果 | 规避方式 |
|---|---|---|
| 全图或多跳爆炸 | 训练慢，显存大，批次无法稳定 | 用短随机游走和固定 Top-T 邻居 |
| 均匀采样失真 | 弱相关邻居占比高，表示变平 | 用 visit count 做 Importance Pooling |
| 没有 L2 normalize | 点积受向量模长影响，ANN 检索不稳定 | 输出 embedding 后做 L2 归一化 |
| 负样本不够难 | 模型只学到粗粒度区分 | 引入 hard negatives |
| 图更新频繁 | 离线 embedding 过期，召回质量下降 | 定期重算，增量更新，高频 item 优先刷新 |

一个常见错误是把 PinSage 理解成“随机采样版 GraphSAGE”。这个说法不够准确。PinSage 的采样不是简单随机抽邻居，而是通过随机游走访问次数估计重要性；聚合也不是平均，而是按重要性加权。

另一个坑是忽略图和内容特征的配合。PinSage 需要图结构提供协同信号，也需要 pin 内容特征提供泛化能力。如果只依赖图结构，新 item 很难获得好表示；如果只依赖内容特征，又损失了用户组织行为带来的语义信息。

还有一个线上坑是 embedding 版本管理。训练出的向量、ANN 索引、排序特征必须使用兼容版本。否则召回阶段使用新向量，排序阶段使用旧特征，指标波动会很难排查。

---

## 替代方案与适用边界

PinSage 适合“图结构强、item 数量大、召回规模大、内容特征丰富”的推荐场景。它解决的是图驱动的 item 表示学习，不是所有推荐问题的通解。

| 方法 | 优点 | 缺点 | 适用场景 |
|---|---|---|---|
| Matrix Factorization | 简单、稳定、训练成本低 | 对内容特征利用弱，冷启动困难 | 只有用户-物品点击矩阵的基础推荐 |
| 双塔召回 | 在线检索友好，用户塔和物品塔可分离 | 对复杂图结构建模较弱 | 用户行为丰富、需要低延迟召回 |
| 标准 GNN | 表达能力强，能建模多跳关系 | 大图上邻域爆炸，训练复杂 | 中小规模图、节点分类、链路预测 |
| 序列推荐模型 | 擅长捕捉行为顺序和短期兴趣 | 对 item-item 图结构利用不直接 | 新闻、短视频、电商会话推荐 |
| PinSage | 适合大规模 item 图，召回向量稳定 | 实现复杂，需要图采样和离线索引 | Pinterest 类内容组织图和相关推荐 |

选择时可以按问题结构判断：如果只有用户-物品点击，没有丰富图结构，先考虑 Matrix Factorization 或双塔模型；如果图很大、关系复杂、item 内容丰富，PinSage 更合适；如果主要目标是建模用户最近几次行为的顺序变化，序列推荐模型通常更直接。

当图很稀疏、特征很弱、或实时性要求极高时，PinSage 的收益可能不足以覆盖复杂度。尤其是强实时推荐场景，离线 embedding 可能跟不上用户兴趣变化，需要搭配实时特征、在线排序或短周期增量更新。

术语对照表：

| 术语 | 含义 |
|---|---|
| Pin | Pinterest 里的内容 item，可以是图片、商品或灵感内容 |
| Board | 用户收藏和组织 pin 的集合 |
| Embedding | 把对象表示成向量，便于相似度计算 |
| Random Walk | 从图上某个节点出发，沿边随机移动 |
| Top-T | 按分数选前 T 个候选 |
| Hard Negative | 与目标相似但不应匹配的负样本 |

---

## 参考资料

1. [PinSage: Graph Convolutional Neural Networks for Web-Scale Recommender Systems](https://arxiv.org/abs/1806.01973)
2. [KDD 2018 Accepted Paper: Graph Convolutional Neural Networks for Web-Scale Recommender Systems](https://www.kdd.org/kdd2018/accepted-papers/view/graph-convolutional-neural-networks-for-web-scale-recommender-systems)
3. [Pinterest Engineering: PinSage, a new graph convolutional neural network for web-scale recommender systems](https://medium.com/pinterest-engineering/pinsage-a-new-graph-convolutional-neural-network-for-web-scale-recommender-systems-88795a107f48)
4. [GraphSAGE 官方页](https://snap.stanford.edu/graphsage/)

建议阅读顺序：先看 Pinterest Engineering 博客理解工程动机，再看 arXiv 论文看公式和实验，最后看 GraphSAGE 官方页补背景。核心公式和工程直觉主要来自论文，随机游走加权采样是 PinSage 区别于普通 GraphSAGE 的关键点。
