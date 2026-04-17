## 核心结论

向量召回里的 ANN 索引，意思是“近似最近邻索引”，就是不再严格扫描全库，而是用更少计算量找到“足够接近”的候选。它的选择本质不是“谁最快”，而是同时在三件事之间做交换：召回率、延迟、内存。

生产环境里，最常见的误判是只看 QPS。QPS 只是吞吐量，表示一秒能处理多少请求，但它并不直接告诉你结果是否找对了、尾延迟是否失控、机器是否装得下数据。真正该盯的是三维指标：

| 指标 | 典型目标建议 | 说明 |
| --- | --- | --- |
| Recall@10 | ≥ 0.9 | 返回前 10 个结果里，至少 9 个接近精确搜索 |
| P95/P99 延迟 | 交互场景常见目标是 P95 < 80ms | 决定用户实际感知，尤其是慢请求尾部 |
| 字节/向量 | 常见预算 200–400B，压缩场景可更低 | 决定总容量与机器成本 |

如果内存充裕、目标是低延迟和高召回，HNSW 往往是优先选项。HNSW 是“分层图索引”，白话说就是先在稀疏大路上快速靠近目标，再在底层密集道路里精细找邻居。它查询快、召回高，但需要把大量图结构和原始向量放在内存里。

如果数据量已经到亿级甚至十亿级，IVF-PQ 更常见。IVF 是“倒排文件索引”，白话说就是先把向量粗分桶，只搜最可能命中的几个桶；PQ 是“乘积量化”，白话说就是把长向量压缩成短码，用少量字节近似表示原始向量。它们的组合能明显降低内存，但会引入量化误差，召回率通常低于高配 HNSW。

一个简单判断标准是：先定复合目标，再选索引。例如目标是 `Recall@10 >= 0.9`、`P95 <= 10ms`、`bytes per vector <= 250B`。如果 HNSW 满足前两项但内存超预算，就要考虑 IVF-PQ 或热冷分层，而不是继续只压榨单机吞吐。

---

## 问题定义与边界

向量检索的任务，是从大量向量里找出与查询向量最相近的前 $K$ 个结果。这里的“相近”通常用余弦相似度、内积或欧氏距离衡量。暴力精确搜索，意思是每次查询都和全库所有向量逐一比较，结果最准，但当库规模到 $10^7$、$10^8$ 甚至更高时，延迟和成本通常不可接受。

所以 ANN 的问题定义不是“怎么完全替代精确搜索”，而是“在可接受误差下，把检索成本降下来”。它的边界也要说清楚：

| 场景 | 更合适的选择 | 原因 |
| --- | --- | --- |
| 数据量不大，要求极高准确性 | 精确搜索或高配 HNSW | 没必要为压缩牺牲结果 |
| 数据量中等，追求低延迟 | HNSW | 通常能换到更高 Recall |
| 数据量超大，内存预算严格 | IVF / IVF-PQ | 先分桶再压缩，容量更强 |
| 离线候选生成、可接受二阶段重排 | IVF-PQ + rerank | 先快筛，再精排 |

Recall@K 是最核心的准确性指标。Recall 就是“找回了多少本该找到的结果”。如果精确搜索的前 10 个答案里，ANN 返回结果和它重合了 9 个，那么 Recall@10 就是 0.9。

数学上常写成：

$$
\text{Recall@K} = \frac{1}{Q} \sum_{i=1}^{Q} \frac{|\text{TopK}_{\text{approx},i} \cap \text{TopK}_{\text{exact},i}|}{K}
$$

其中 $Q$ 是查询数，$\text{TopK}_{\text{exact}}$ 表示精确搜索得到的前 $K$ 个结果，$\text{TopK}_{\text{approx}}$ 表示近似索引返回的前 $K$ 个结果。

玩具例子可以非常直观。假设只有 8 个商品向量，查询是“黑色跑鞋”。精确搜索的前 3 个结果是 `[A, B, C]`，ANN 返回 `[A, C, D]`。两者交集有 2 个，所以：

$$
\text{Recall@3} = \frac{2}{3}
$$

这说明 ANN 没有“完全错”，但漏掉了一个真正重要的候选。对推荐系统和召回系统来说，这种漏掉会直接影响后续排序模型的上限，因为后面的精排只能在已召回的候选里选。

---

## 核心机制与推导

HNSW、IVF、PQ 解决的是同一个问题，但方法不同。

HNSW 的全称是 Hierarchical Navigable Small World。它本质上是多层图。图可以理解成“节点和边组成的网络”。每个向量是一个节点，边连接到若干近邻。高层节点少、连接稀疏，适合快速接近目标；底层节点多、连接密集，适合精细搜索。查询时通常从高层入口点开始，做贪心下降，也就是每一步都走向当前看起来更接近查询的位置，直到最底层再做更充分的局部扩展。

它为什么快？因为不再检查所有节点，而是沿着图结构走一条相对短的路径。它为什么吃内存？因为要保存原始向量，还要保存每个节点的邻接关系。节点连接数越大，搜索质量通常越好，但内存也更高。

IVF 的全称是 Inverted File。它先做一次粗聚类，聚类就是把相似向量分成若干组。查询时先找最接近查询的几个聚类中心，再只在这些桶里搜索，而不是全库扫描。这里两个参数很关键：

| 参数 | 含义 | 调大后的典型效果 |
| --- | --- | --- |
| `nlist` | 粗聚类桶数 | 桶更细，训练和管理更复杂 |
| `nprobe` | 查询时访问的桶数 | Recall 上升，延迟也上升 |

PQ 的作用是进一步压缩。原始向量如果是 768 维 `float32`，每维 4 字节，总大小大约是：

$$
768 \times 4 = 3072 \text{ bytes} \approx 3 \text{ KB}
$$

1 亿条就是约 300GB，仅原始向量就已经很大。PQ 的思路是把向量切成 $M$ 段子向量，每段用一个小码本离散编码。码本可以理解成“这一段常见形状的字典”。原始向量：

$$
x = [x_1, x_2, \dots, x_M]
$$

编码后变成：

$$
\text{code}(x) = [q_1(x_1), q_2(x_2), \dots, q_M(x_M)]
$$

其中：

$$
q_m(x_m) = \arg\min_j \|x_m - c_{m,j}\|^2
$$

意思是：第 $m$ 段子向量，不再存原值，而是只存“它最接近第几个码本中心”的编号。如果每段用 256 个中心，那么一个编号只要 1 字节。假设把 768 维向量切成 48 段，那么每条向量只需 48 字节编码，压缩比大约是：

$$
\frac{3072}{48} = 64
$$

也就是约 64 倍压缩。

玩具例子：把一个 8 维向量切成 2 段，每段 4 维。原始向量是 `[0.1, 0.0, 0.2, 0.1, 0.8, 0.7, 0.9, 0.8]`。前 4 维可能被量化成“码字 3”，后 4 维量化成“码字 11”。存储时不再保留 8 个浮点数，而只保留 `[3, 11]` 这两个编号。这样内存下降很明显，但距离计算变成“近似距离”，不是精确距离，所以召回会受影响。

真实工程例子可以这样理解。假设一个推荐系统要在 2 亿图片 embedding 里做召回。若全部用 HNSW，查询延迟可能非常好，但内存会很重，扩容昂贵。若改成 IVF-PQ，先把候选压成短码，再用 `nprobe` 控制检索范围，可能把单向量成本压到几十字节到几百字节，但需要接受一定召回损失。此时系统设计重点就不再是“绝对最准”，而是“能否让粗召回足够好，给下游排序留出空间”。

---

## 代码实现

下面用纯 Python 写一个最小可运行示例，演示两个关键概念：

1. 如何计算 Recall@K  
2. 为什么“少搜一些桶”会导致召回下降

这个例子不是工业级 ANN 实现，而是帮助理解指标和参数的关系。

```python
import math

def l2(a, b):
    return math.sqrt(sum((x - y) ** 2 for x, y in zip(a, b)))

def exact_topk(query, vectors, k):
    pairs = [(idx, l2(query, vec)) for idx, vec in enumerate(vectors)]
    pairs.sort(key=lambda x: x[1])
    return [idx for idx, _ in pairs[:k]]

def recall_at_k(exact_ids, approx_ids, k):
    return len(set(exact_ids[:k]) & set(approx_ids[:k])) / k

# 两个“桶”，模拟 IVF 的粗分桶
bucket_0 = [
    [0.0, 0.0],
    [0.1, 0.0],
    [0.2, 0.1],
]

bucket_1 = [
    [1.0, 1.0],
    [1.1, 1.0],
    [0.9, 1.2],
]

vectors = bucket_0 + bucket_1
query = [0.15, 0.05]

# 精确搜索
exact = exact_topk(query, vectors, k=2)

# 近似搜索：只搜一个桶，模拟 nprobe 太小
approx_only_bucket_1 = exact_topk(query, bucket_1, k=2)
approx_only_bucket_1 = [i + len(bucket_0) for i in approx_only_bucket_1]

# 近似搜索：搜对了桶
approx_bucket_0 = exact_topk(query, bucket_0, k=2)

r_bad = recall_at_k(exact, approx_only_bucket_1, k=2)
r_good = recall_at_k(exact, approx_bucket_0, k=2)

assert exact == [1, 2]
assert r_bad == 0.0
assert r_good == 1.0

print("exact:", exact)
print("bad recall@2:", r_bad)
print("good recall@2:", r_good)
```

这个例子说明：如果 IVF 在粗聚类阶段就把查询导向了错误的桶，后面的精细搜索再快也没用，所以 `nprobe` 往往是第一批要 sweep 的参数。sweep 就是“系统性扫参数”，白话说就是把参数从小到大跑一遍，看曲线怎么变化。

如果使用 Faiss，IVF-PQ 的构建流程通常是“训练粗聚类 + 训练 PQ 码本 + 添加数据 + 设置查询参数”。最小示意如下：

```python
import faiss

def build_ivfpq(vectors, nlist=256, m=16, nbits=8):
    d = vectors.shape[1]
    quantizer = faiss.IndexFlatIP(d)
    index = faiss.IndexIVFPQ(quantizer, d, nlist, m, nbits)
    index.train(vectors[:100_000])
    index.add(vectors)
    index.nprobe = 16
    return index
```

这里几个参数要直观理解：

| 参数 | 作用 | 常见影响 |
| --- | --- | --- |
| `nlist` | 粗聚类数量 | 太小会让桶太粗，太大训练成本升高 |
| `m` | PQ 子空间数量 | 越大通常压缩更细，但训练和误差行为更复杂 |
| `nbits` | 每个子空间编码位数 | 位数越高，码本更细，内存也更高 |
| `nprobe` | 查询访问桶数 | 最直接影响 Recall/Latency 曲线 |

真实工程里，一个常见流程是：先用离线样本做 exact baseline，再对 `nprobe`、`m`、`nbits`、`nlist` 做网格试验。每组参数都记录 Recall@10、P95 latency、bytes/vector。最后不是选“最快”的，而是选满足 SLO 的最低成本方案。

例如在推荐系统召回层，线上可能会这样部署：最近 7 天热点内容放 HNSW，要求 5 到 10ms 内返回高质量候选；长尾历史内容放 IVF-PQ，允许 20 到 30ms 延迟，再把两路结果合并做重排。这比用单一索引覆盖全部数据更常见。

---

## 工程权衡与常见坑

第一类坑，是把“压缩率”误当成唯一目标。PQ 能把 768 维 `float32` 从约 3KB 压到几十字节，这很诱人，但压得越狠，量化误差越大。量化误差就是“压缩后表示和原始向量之间的偏差”。如果误差太大，近邻关系会被破坏，召回下降会非常明显。

第二类坑，是把单机 benchmark 误当成线上结论。离线压测里 QPS 很漂亮，不代表线上体验就好。真实流量下更关键的是 P95、P99 和冷启动表现。特别是 HNSW，这类图索引往往很依赖内存命中和页缓存预热。冷启动阶段，如果操作系统还没把热数据读入内存，前几百个请求的尾延迟会很难看。

第三类坑，是只看平均 Recall，不看分布漂移。漂移的意思是模型输入分布变了，比如 embedding 模型升级、商品库内容变化、文本长度结构变化。即使索引参数没变，Recall 也可能从 0.92 掉到 0.85。解决办法通常不是“肉眼看几个样本”，而是持续保留一小部分 shadow exact 查询。shadow exact 的意思是线上少量请求同时跑精确搜索，用来校验 ANN 是否偏离。

| 坑 | 影响 | 规避 |
| --- | --- | --- |
| PQ 压缩过猛 | Recall 明显下降，排序上限变低 | 离线 sweep，保留 exact baseline |
| `nprobe` 太小 | 延迟很好，但漏召回严重 | 画 Recall-Latency 曲线找拐点 |
| `nlist` 不合理 | 桶分布失衡，部分桶过大 | 检查聚类样本与桶大小分布 |
| HNSW 冷启动 | 前几百请求 P95 飙高 | 预热索引与页面缓存 |
| 模型分布漂移 | 老参数突然失效 | shadow exact + 蓝绿重建 |

真实工程例子：图片推荐系统把 PQ 参数从 `m=16` 调到 `m=32`，表面上每段编码更细，但训练样本不足、残差分布也不稳定，结果线上 Recall@10 从 0.91 掉到 0.86。此时如果只看机器成本，会误以为“压缩成功”；如果同时看业务点击率和精确对照，就会发现召回候选已经变差。正确做法是回退参数、补训练样本、先在蓝绿环境重建索引再切流。

这里有一个基本判断：ANN 索引不是“建完就不管”的静态资产，而是和 embedding 模型、数据分布、容量规划一起变化的工程组件。

---

## 替代方案与适用边界

并不是所有场景都该直接上 HNSW 或 IVF-PQ。索引选择最好从数据规模、延迟预算、更新频率、内存预算四个维度一起看。

| 向量规模 | 目标 | 推荐索引 |
| --- | --- | --- |
| < 10M | 极低延迟、频繁更新 | HNSW |
| 10M–500M | 希望兼顾容量与效果 | IVF 或 IVF-PQ |
| > 1B | 容量优先，允许近似误差 | IVF-PQ + 分片/GPU |
| 小库高精度场景 | 精度优先 | Flat 精确搜索 |

HNSW 更适合热数据。热数据就是最近经常被查、价值高、延迟要求严格的数据。例如 RAG 系统里最近更新的知识片段，或者推荐系统里最近活跃的内容候选。它的问题是内存成本高，更新也不是无限灵活。

IVF-PQ 更适合冷数据。冷数据就是容量大、访问频率低、但仍需要覆盖的数据。例如长尾商品、历史对话、老旧文档。它用更低成本保留大范围检索能力，但通常需要接受 Recall 不是最优。

一个很实用的替代方案是分层检索。分层的意思是把不同温度、不同价值的数据放到不同索引里。真实工程例子：客服知识库把最近 30 天高频工单放 HNSW，把历史归档问答放 IVF-PQ。查询先打热层，命中不足再打冷层，最后统一重排。这样既保留了热路径低延迟，也避免把所有历史数据都塞进昂贵内存。

另一个边界是“是否需要二阶段重排”。如果下游有强排序模型，召回层可以接受适度近似，因为后续还会重排；如果下游只是直接展示 TopK，那么召回层精度要求就更高，可能更偏向 HNSW 或较高 `nprobe` 的 IVF。

所以，索引选择不是抽象算法题，而是容量规划题。目标不是找到“最先进算法”，而是找到在当前业务约束下最合适的误差分配方式。

---

## 参考资料

| 来源 | 关注焦点 |
| --- | --- |
| [The Agentic Web: Vector Databases in Production](https://www.theagenticweb.dev/blog/vector-databases-in-production?utm_source=openai) | 生产指标设计，Recall、Latency、Cost 的联合观测 |
| [SystemOverflow: HNSW vs IVF-PQ at Billion Scale](https://www.systemoverflow.com/learn/ml-search-ranking/search-scalability/approximate-nearest-neighbor-search-hnsw-vs-ivf-pq-at-billion-scale?utm_source=openai) | HNSW 与 IVF-PQ 在大规模场景下的内存、召回与实践边界 |
| [SystemOverflow: Approximate Nearest Neighbor HNSW IVF PQ](https://www.systemoverflow.com/learn/ml-search-ranking/search-scalability/approximate-nearest-neighbor-hnsw-ivf-pq?utm_source=openai) | 常见 ANN 索引的机制与选型建议 |
| [APXML: Product Quantization Mechanics](https://apxml.com/courses/advanced-vector-search-llms/chapter-1-ann-algorithms/product-quantization-mechanics?utm_source=openai) | PQ 的编码公式、子空间划分与距离估计 |
| [APXML: IVF Variations](https://apxml.com/courses/advanced-vector-search-llms/chapter-1-ann-algorithms/ivf-variations?utm_source=openai) | IVF 系列索引与 `nprobe` 等参数含义 |
| [EngineersOfAI: ANN Algorithms](https://engineersofai.com/docs/ai-systems/vector-database-engineering/ann-algorithms?utm_source=openai) | Faiss 中 IVF-PQ 的工程用法与参数示意 |
