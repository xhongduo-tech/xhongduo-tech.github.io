## 核心结论

ANN 召回是在允许少量近似误差的前提下，从百万级向量里快速找 top-k 最近邻。

向量检索的精确目标可以写成：

$$
\operatorname{argmin}_{i} \ \delta(q, x_i)
$$

其中 $q$ 是查询向量，$x_i$ 是数据库里的第 $i$ 个向量，$\delta$ 是距离函数。ANN，也就是 Approximate Nearest Neighbor，中文通常叫“近似最近邻”，做的不是改变这个目标，而是用近似方式更快地逼近这个目标。

核心结论只有两点：

1. ANN 的本质不是“完全准确”，而是用可控误差换取可控的延迟、内存和吞吐。
2. 真正的工程目标不是“找到最先进算法”，而是在召回率、延迟、内存、构建成本、更新成本之间找到可用平衡。

玩具例子：有 10 个二维点，查询点是 `(2, 3)`，暴力搜索就是把查询点和 10 个点逐个算距离，然后排序。数据变成 1000 万条 128 维向量时，暴力搜索仍然要逐条计算距离，计算量大约是 $n \times d$。ANN 会先用索引结构快速筛出一小批候选，再对候选做更精确的排序。

真实工程例子：推荐系统里可能有 1 亿条 item embedding。embedding 是把商品、文章、视频或用户表示成一串数字的向量。在线请求通常不会直接对 1 亿条物品全量排序，而是先用 ANN 召回 top200，再用精确点积、深度排序模型或业务规则重排到 top20。ANN 负责把候选集从千万级或亿级压到几百条。

| 方法 | 精度 | 延迟 | 内存 | 适用维度 |
|---|---:|---:|---:|---|
| 暴力搜索 | 最高 | 高 | 低到中 | 任意维度，小数据可用 |
| KD-tree | 精确或近似 | 低维较低，高维退化 | 中 | 低维数据 |
| HNSW | 高 | 低 | 高 | 中高维稠密向量 |
| PQ/OPQ | 中到高 | 低 | 低 | 超大规模、高维、内存敏感 |

---

## 问题定义与边界

向量检索是在一个向量集合 $X=\{x_i\}_{i=1}^{n}$ 中，给定查询向量 $q$，找出距离最近的 $k$ 个向量：

$$
\operatorname{TopK}_{x_i \in X} \ -\delta(q, x_i)
$$

这里的 top-k 是“前 k 个结果”。如果距离越小越相似，就按距离升序取前 k 个；如果分数越大越相似，比如内积，就按分数降序取前 k 个。

距离度量需要先统一：

| 度量 | 公式 | 含义 | 常见场景 |
|---|---|---|---|
| L2 距离 | $\|q-x\|_2$ | 欧氏距离，越小越近 | 图像、通用 embedding |
| Cosine 相似度 | $\frac{q \cdot x}{\|q\|\|x\|}$ | 方向越接近越相似 | 文本语义向量 |
| Inner Product | $q \cdot x$ | 内积越大越相似 | 推荐召回、双塔模型 |

如果所有向量都做了 L2 归一化，也就是 $\|q\|=\|x\|=1$，那么 L2 距离和 cosine 相似度存在单调关系：

$$
\|q-x\|_2^2 = 2 - 2(q \cdot x)
$$

这意味着归一化之后，最大化内积和最小化 L2 距离在排序上等价。工程里最常见的坑之一，就是训练时用 cosine，索引里用 inner product，线上又忘了归一化，导致检索结果漂移。

ANN 的边界也要明确：ANN 解决的是候选召回，不是最终排序。召回是先从大池子里捞出可能相关的一小批候选；排序是对候选做更精细的打分。新手可以把它理解成：从海量商品里先挑出最像的 200 个，再从这 200 个里精确排序。

| 场景 | 是否需要 ANN | 原因 |
|---|---|---|
| 几千条向量 | 通常不需要 | 暴力扫描简单，索引开销不划算 |
| 低维、几十万条 | 可尝试 KD-tree | 空间剪枝还有机会有效 |
| 128 维或 768 维稠密向量 | 通常需要 HNSW、IVF、PQ | 暴力扫描延迟高，KD-tree 容易退化 |
| 最终排序 | 不应只依赖 ANN | ANN 有近似误差，需要 rerank |
| 高频实时更新 | 需要单独评估 | 有些索引构建快，有些训练和重建成本高 |

---

## 核心机制与推导

主流 ANN 加速思路可以分成三类：树方法、图方法、量化方法。

KD-tree 是树方法。它的核心思想是“空间划分”：每个节点按某个维度把空间切成两半，查询时优先进入更可能接近查询点的分支。KD-tree 是 k-dimensional tree，白话说就是按多个坐标轴递归切分空间的二叉树。

KD-tree 能剪枝，是因为每个节点代表一个空间区域。设节点区域到查询点的距离下界为 $LB(node,q)$，当前已经找到的最好距离为 $r^*$。如果：

$$
LB(node,q) > r^*
$$

那么这个节点下面所有点都不可能比当前结果更近，可以直接跳过。问题在于，高维空间里点与点之间的距离会变得不容易区分，很多节点的下界都不够大，剪枝效果下降。128 维、768 维 embedding 上，KD-tree 往往不是主力方案。

HNSW 是图方法。HNSW 全称 Hierarchical Navigable Small World，白话说是“分层的可导航近邻图”。它把向量组织成多层图：上层点少，用来快速找到大方向；下层点多，用来做精细搜索。查询时先从高层入口点开始，沿着更接近查询向量的邻居移动，再逐层下降，最后在底层扩展候选。

HNSW 常见参数如下：

| 参数 | 含义 | 调大后的影响 |
|---|---|---|
| `M` | 每个点最多连接的邻居数 | 召回可能升高，内存增加 |
| `ef_construction` | 构建索引时的搜索宽度 | 构建更慢，图质量更高 |
| `ef` | 查询时保留的候选宽度 | 召回通常升高，延迟升高 |

这里的 `ef` 是线上最常调的参数。它越大，搜索过程中保留的候选越多，越不容易错过真实近邻，但每次查询访问的节点也更多。

PQ 是量化方法。PQ 全称 Product Quantization，中文常叫“乘积量化”。量化的意思是用少量代表点近似原始连续向量。PQ 会把一个向量切成 $M$ 段：

$$
x=(x^{(1)},...,x^{(M)})
$$

每一段都有自己的码本。码本是若干个代表向量的集合，可以理解为“常见局部形状的表”。对第 $m$ 段，编码过程是找最近的码本中心：

$$
c_m = \operatorname{argmin}_j \|x^{(m)}-a_m[j]\|^2
$$

查询时不再直接拿完整原始向量算距离，而是查表计算近似距离：

$$
\hat{\delta}(q,x)=\sum_m \|q^{(m)}-a_m[c_m]\|^2
$$

玩具例子：取 $d=4$，$M=2$，把向量分成前后两段。查询 $q=(2,1,0,3)$。如果第一段码本中心是 $(3,1)$，第二段码本中心是 $(0,2)$，那么近似向量是 $(3,1,0,2)$，近似距离为：

$$
\|(2,1)-(3,1)\|^2+\|(0,3)-(0,2)\|^2=1+1=2
$$

如果真实向量是 $x=(1.8,1.2,0.1,2.7)$，精确距离约为 $0.18$。这说明 PQ 会引入量化误差，所以工程中经常先用 PQ 召回，再用原始向量精排。

OPQ 是 Optimized Product Quantization，白话说是“先旋转向量，再做 PQ”。它会学习一个旋转矩阵 $R$：

$$
x' = R x
$$

旋转后再切段量化，目的是让不同子向量的信息分布更均匀，减少 PQ 切段带来的误差。

| 方法 | 加速机制 | 主要牺牲 | 典型适用场景 |
|---|---|---|---|
| KD-tree | 空间划分与距离下界剪枝 | 高维剪枝失效 | 低维、小中规模数据 |
| HNSW | 分层图遍历，少量访问近邻节点 | 内存较高 | 高召回、低延迟在线检索 |
| PQ | 向量压缩与查表近似距离 | 距离有量化误差 | 超大规模、内存敏感 |
| OPQ | 旋转后量化，降低误差 | 训练更复杂 | 对 PQ 召回质量要求更高 |

---

## 代码实现

实现向量检索时，建议先写暴力基线，再替换成 ANN。暴力基线的价值是校验距离函数、归一化逻辑和 top-k 口径。没有基线，后面调 HNSW、IVFPQ、OPQ 时很难判断问题来自算法、参数还是数据处理。

下面是一个最小可运行的暴力 top-k 示例：

```python
import numpy as np

def l2_topk(query, items, k):
    distances = np.sum((items - query) ** 2, axis=1)
    idx = np.argsort(distances)[:k]
    return idx, distances[idx]

items = np.array([
    [0.0, 0.0],
    [1.0, 1.0],
    [2.0, 2.0],
    [10.0, 10.0],
], dtype=np.float32)

query = np.array([1.2, 1.1], dtype=np.float32)
idx, dist = l2_topk(query, items, k=2)

assert idx.tolist() == [1, 2]
assert dist[0] < dist[1]
print(idx, dist)
```

HNSW 的代码通常依赖 `hnswlib`。下面是典型结构：

```python
import numpy as np
import hnswlib

dim = 128
num_items = 10000
items = np.random.randn(num_items, dim).astype(np.float32)

# cosine 检索通常要求先归一化
items /= np.linalg.norm(items, axis=1, keepdims=True) + 1e-12

index = hnswlib.Index(space="cosine", dim=dim)
index.init_index(max_elements=num_items, ef_construction=200, M=16)
index.add_items(items, np.arange(num_items))
index.set_ef(100)

query = np.random.randn(1, dim).astype(np.float32)
query /= np.linalg.norm(query, axis=1, keepdims=True) + 1e-12

labels, distances = index.knn_query(query, k=10)
assert labels.shape == (1, 10)
assert distances.shape == (1, 10)
```

Faiss 的 IVFPQ 会先把向量分桶，再在桶内用 PQ 压缩。IVF 是 Inverted File Index，白话说是“先把向量分到若干粗粒度桶里，查询时只查其中一部分桶”。

```python
import numpy as np
import faiss

dim = 128
num_items = 50000
nlist = 256      # IVF 桶数量
m = 16           # PQ 子向量段数
nbits = 8        # 每段编码位数

items = np.random.randn(num_items, dim).astype("float32")
faiss.normalize_L2(items)

quantizer = faiss.IndexFlatIP(dim)
index = faiss.IndexIVFPQ(quantizer, dim, nlist, m, nbits)
index.train(items)
index.add(items)

index.nprobe = 16

query = np.random.randn(5, dim).astype("float32")
faiss.normalize_L2(query)

scores, ids = index.search(query, 20)
assert ids.shape == (5, 20)
assert scores.shape == (5, 20)
```

在线链路通常不是“ANN 查完直接返回”，而是召回后重排：

```python
import numpy as np

def exact_rerank(query, item_vectors, candidate_ids, topk):
    candidates = item_vectors[candidate_ids]
    scores = candidates @ query
    order = np.argsort(-scores)[:topk]
    return candidate_ids[order], scores[order]

item_vectors = np.random.randn(1000, 64).astype(np.float32)
item_vectors /= np.linalg.norm(item_vectors, axis=1, keepdims=True) + 1e-12

query = np.random.randn(64).astype(np.float32)
query /= np.linalg.norm(query) + 1e-12

candidate_ids = np.array([5, 9, 12, 300, 42, 77])
top_ids, top_scores = exact_rerank(query, item_vectors, candidate_ids, topk=3)

assert len(top_ids) == 3
assert np.all(top_scores[:-1] >= top_scores[1:])
```

| 步骤 | 输入 | 输出 | 说明 |
|---|---|---|---|
| 1. 向量生成 | 用户、query、item 特征 | embedding | 由召回模型生成 |
| 2. 归一化 | 原始 embedding | 单位向量 | 统一 cosine/IP 口径 |
| 3. ANN 召回 | query vector | top200 candidates | HNSW、IVFPQ、PQ |
| 4. 精确重排 | candidates + 原始向量 | top20 | 用精确距离或业务模型 |
| 5. 返回结果 | top20 | 页面展示 | 可叠加过滤、去重、多样性 |

评测至少要看三类指标：召回率、P95 延迟、内存占用。召回率是 ANN 结果和暴力搜索结果的重合比例；P95 延迟是 95% 请求都能低于的耗时；内存占用决定服务部署成本。

---

## 工程权衡与常见坑

ANN 参数不是越大越好。`ef`、`nprobe`、`M`、PQ 段数、码本大小都会改变召回、延迟、内存和构建时间。

新手常见误区是只看召回率。例如把 HNSW 的 `ef` 从 50 调到 500，召回可能明显上升，但 P95 延迟也可能超过线上 SLA。SLA 是服务等级目标，白话说就是线上接口必须满足的延迟和稳定性要求。如果推荐接口要求 50ms 内返回，单纯追求 99.9% 召回没有意义。

真实事故例子：某推荐系统用 PQ 压缩 1 亿条 item embedding。码本用三个月前的数据训练，后来召回模型升级，embedding 分布发生变化，但索引码本没有重训。结果离线评测还能勉强通过，线上点击率下降。原因不是 PQ 算法失效，而是码本和线上向量分布不一致，量化误差变大。

| 常见坑 | 表现 | 规避方式 |
|---|---|---|
| KD-tree 用在高维稠密向量 | 查询访问大量节点，接近暴力扫描 | 高维优先评估 HNSW、IVF、PQ |
| 只调 HNSW 的 `ef` | 召回上升但延迟变差 | 同时评估 `M`、`ef_construction`、`ef` |
| PQ 码本分布过旧 | 召回突然下降 | 用接近线上分布的数据重训码本 |
| L2、cosine、IP 混用 | 离线线上结果不一致 | 固定距离度量和归一化规范 |
| ANN 结果直接排序 | 最终结果不稳定 | 对候选做 exact rerank |
| 索引更新策略缺失 | 新物品不可见或旧物品残留 | 设计增量更新、删除和重建流程 |

| 目标 | 倾向选择 | 代价 |
|---|---|---|
| 高召回 | 增大 `ef`、`nprobe`、HNSW 图连接数 | 延迟和内存上升 |
| 低延迟 | 减少搜索宽度，减少候选数 | 召回可能下降 |
| 低内存 | PQ、OPQ、IVFPQ | 需要训练，距离有误差 |
| 高频更新 | HNSW 或支持增量的索引 | 内存更高，删除可能复杂 |
| 静态大库 | IVF、PQ、OPQ | 重建成本可接受 |

参数调优建议按顺序做。第一步，固定距离度量和归一化方式，确保暴力基线正确。第二步，选索引结构，比如 HNSW 或 IVFPQ。第三步，调离线构建参数，比如 `M`、`ef_construction`、`nlist`、PQ 段数。第四步，调线上查询参数，比如 `ef` 或 `nprobe`。最后用真实流量分布评估 P50、P95、P99 延迟，不要只用平均延迟。

---

## 替代方案与适用边界

ANN 不是所有向量检索问题的默认答案。数据量、维度、更新频率、内存预算、召回目标不同，方案会不同。

玩具例子：只有几千条 32 维向量，直接暴力扫描通常最简单。代码少、结果精确、没有训练和索引维护成本。此时上 HNSW 或 IVFPQ，可能只是把简单问题复杂化。

真实工程例子：1 亿条 128 维 item embedding，单机内存有限，且要求在线 50ms 内召回 top200。此时暴力扫描基本不可接受。若更看重高召回和更新便利，可以优先 HNSW；若内存紧张，可以优先 IVFPQ、PQ、OPQ；若业务天然有类目、地域、语言分桶，还可以先按业务规则路由，再在桶内做 ANN。

| 方案 | 数据量 | 维度 | 更新频率 | 内存限制 | 召回要求 |
|---|---|---|---|---|---|
| 暴力扫描 | 小到中 | 任意 | 任意 | 中 | 精确最高 |
| KD-tree | 小到中 | 低维 | 中 | 中 | 精确或近似 |
| HNSW | 中到大 | 中高维 | 中到较高 | 高 | 高召回 |
| IVF | 大 | 中高维 | 中 | 中 | 依赖 `nprobe` |
| IVFPQ | 大到超大 | 中高维 | 较低 | 低 | 中到高 |
| PQ/OPQ | 超大 | 高维 | 较低 | 很低 | 允许近似误差 |

决策规则可以简化为四条：

低维、小数据：优先暴力扫描或 KD-tree。系统简单比算法复杂更重要。

大数据、高召回：优先 HNSW。它通常能在召回和延迟之间给出很好的平衡，但要接受更高内存占用。

超大规模、内存敏感：优先 PQ、OPQ、IVFPQ。它们通过压缩向量降低内存和计算量，但要认真评估量化误差。

需要分桶路由：优先 IVF 系列或业务分桶加 ANN。IVF 的粗聚类桶适合先缩小搜索范围，但 `nprobe` 太小会漏召回，太大又会增加延迟。

最终选择不应只看算法名字，而要落到一张评测表：相同数据、相同 query 集、相同 top-k、相同距离度量下，对比 recall@k、P95 延迟、索引大小、构建时间和更新成本。ANN 是工程系统的一部分，不是单独的算法竞赛。

---

## 参考资料

- Bentley, J. L. 1975. Multidimensional Binary Search Trees Used for Associative Searching. Communications of the ACM. https://doi.org/10.1145/361002.361007
- Malkov, Y. A., Yashunin, D. A. 2018. Efficient and robust approximate nearest neighbor search using Hierarchical Navigable Small World graphs. https://arxiv.org/abs/1603.09320
- Jégou, H., Douze, M., Schmid, C. 2011. Product Quantization for Nearest Neighbor Search. IEEE TPAMI. https://doi.org/10.1109/TPAMI.2010.57
- Ge, T., He, K., Ke, Q., Sun, J. 2013. Optimized Product Quantization for Approximate Nearest Neighbor Search. https://www.microsoft.com/en-us/research/publication/optimized-product-quantization-for-approximate-nearest-neighbor-search/
- Faiss Wiki: Faiss indexes, including PQ and IVFPQ. https://github.com/facebookresearch/faiss/wiki/Faiss-indexes
- Faiss C++ API: OPQMatrix. https://faiss.ai/cpp_api/struct/structfaiss_1_1OPQMatrix.html
- hnswlib README. https://github.com/nmslib/hnswlib
