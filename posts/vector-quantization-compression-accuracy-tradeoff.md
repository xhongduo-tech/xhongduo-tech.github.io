## 核心结论

向量数据库里的量化压缩，本质是用“更短的编码”替代“完整的浮点向量”，用少量精度损失换取大幅存储下降。这里的“量化”可以理解为：不再保存每一维的精确数值，而是保存它更粗粒度的近似表示。

Product Quantization，简称 PQ，是最常见的工程方案。它把一个高维向量拆成 $m$ 段，每段分别做聚类，然后只保存“这一段最接近哪个聚类中心”的编号。这样，原来需要保存几千字节的 float32 向量，最后只需几十到一百多字节。以 1536 维向量为例，原始存储约为 $1536 \times 4 = 6144$ 字节，也就是 6 KB；如果压到 64 字节，压缩比接近 $96:1$。在实际系统里，考虑索引结构、候选重评分和参数配置，常见可接受压缩比往往落在 $24:1$ 到 $64:1$ 之间。

OPQ，Optimized Product Quantization，是 PQ 的增强版。它先学一个旋转矩阵，把原向量转到一个更适合分段量化的坐标系，再做 PQ。这里的“旋转矩阵”可以直白理解为：不改变向量整体几何关系，但重新安排信息分布，让每一段更容易压缩。它通常能把 PQ 带来的召回损失再收回来 1% 到 3%，在很多检索系统里足以决定是否上线。

对初级工程师，最重要的判断不是“PQ 是否无损”，而是“这点损失值不值得换成本”。结论通常很明确：当数据规模进入百万到亿级，PQ/OPQ 不再是优化项，而是让系统可部署的前提条件。

---

## 问题定义与边界

先把问题说清楚。向量检索的核心对象是 embedding，也就是“把文本、图片或行为压成一个固定长度数组的表示”。如果每条向量都用 float32 原样存储，成本会随维度和数据量线性增长。

下面用 1536 维向量做一个直接计算：

| 数据量 | 单条向量大小 | 仅向量原始存储 |
|---|---:|---:|
| 100 万 | 6 KB | 约 6 GB |
| 1000 万 | 6 KB | 约 60 GB |
| 1 亿 | 6 KB | 约 600 GB |

这还只是向量本身，不包含 HNSW 图、倒排结构、payload、冗余副本和操作系统缓存。真实工程里，内存预算通常要再乘一个安全系数。RAG 系统尤其明显，因为你不只要存文档块，还要支持持续增量写入和低延迟检索。

所以，问题不是“要不要压缩”，而是“压缩到什么程度还不至于把检索质量打穿”。这里的“检索质量”常用 Recall@k 衡量，可以直白理解为：真正应该被找回的答案，有多少还留在前 k 个候选里。

一个玩具例子可以帮助理解边界。假设数据库里只有 10 条向量，查询向量最相似的真实结果是 A、B、C。如果压缩后排序变成 A、C、D，那么 Recall@3 从 100% 下降到 $2/3$。在小数据集里，这种误差很刺眼；但在大规模系统里，只要先取更大的候选集，再用原始向量重排，最终前 10 的质量往往仍然可接受。

因此，PQ/OPQ 的适用边界很明确：

| 场景 | 是否优先考虑 PQ/OPQ | 原因 |
|---|---|---|
| 10 万以内，小内存压力 | 不一定 | 原始向量更简单，调参成本低 |
| 百万级，单机内存吃紧 | 是 | 压缩能显著降低资源门槛 |
| 千万到亿级，RAG/推荐/记忆库 | 基本必须 | 不压缩往往无法把成本压到可上线区间 |
| 极端高精度、低延迟且预算充足 | 谨慎 | 可能更适合原始向量 + HNSW |

---

## 核心机制与推导

PQ 的核心机制可以拆成三步。

第一步，分段。把 $d$ 维向量拆成 $m$ 个子向量，每段维度是 $d/m$。例如 1536 维拆成 48 段，每段 32 维。

第二步，每段单独聚类。通常每段训练 256 个质心。这里的“质心”可以直白理解为：这一小段向量的 256 个代表模板。因为 256 个编号刚好能用 1 个字节表示，所以每段最后只存 1 个字节。

第三步，拼接编码。如果一共有 $m$ 段，那么一条向量只需保存 $m$ 个编号。于是总编码长度是：

$$
\text{code\_bits} = m \times \text{code\_size}
$$

如果每段 8 bit，也就是 1 字节，那么：

$$
\text{code\_bytes} = m
$$

例如，1536 维拆成 64 段，每段 1 字节，总编码就是 64 字节。

为什么这种表示还能做近邻搜索？因为查询时不需要把整条向量完整重构出来，而是可以用 ADC，Asymmetric Distance Computation，非对称距离计算。这里的“非对称”可以理解为：查询向量保留原始精度，数据库向量用压缩码表示。

对于查询向量 $q$ 和数据库向量 $x$，PQ 用近似距离：

$$
distance(q, x) \approx \sum_{j=1}^{m} dist(q_j, c_{x,j})
$$

其中 $q_j$ 是查询向量第 $j$ 段，$c_{x,j}$ 是数据库向量在第 $j$ 段对应的质心。意思很直接：每一段查一次表，再把这 $m$ 段的距离加起来。

这比直接对 1536 个浮点维度做乘加便宜得多。原始做法要处理全部维度；PQ 做法只需要为每段预计算“查询段到 256 个质心的距离表”，然后按编码编号取值累加。

玩具例子如下。假设一个 4 维向量拆成两段，每段 2 维：

- 原始向量：$x=[0.2, 0.1, 3.1, 2.9]$
- 第一段码本质心：$[0,0]$、$[1,1]$
- 第二段码本质心：$[3,3]$、$[4,4]$

那么第一段最接近 $[0,0]$，第二段最接近 $[3,3]$，编码就是 `[0, 0]`。以后数据库里不再保存四个浮点，只保存两个编号。查询时，把查询向量也拆成两段，分别查“这段离质心 0 多远、离质心 1 多远”，最后累加。

OPQ 比 PQ 多一步旋转：

$$
x' = R x
$$

其中 $R$ 是学习得到的正交矩阵。它的作用不是降维，而是重新分配信息，让原来强相关的维度不要扎堆落在同一段里。直白看，PQ 假设“切开的每一段都能独立压缩”；但真实 embedding 往往维度之间高度相关，这个假设并不成立。OPQ 先旋转，再切段，等于先把数据“拌匀”，再压缩，结果通常更稳。

真实工程里，这带来的收益很具体。比如一个 768 维或 1536 维 embedding 库，普通 PQ 把 Recall@10 打掉 3% 左右，OPQ 往往能把损失压到 1% 以内，或者至少减少到 1% 到 2% 的区间。对问答检索、长上下文记忆和推荐召回，这个差距足够显著。

---

## 代码实现

下面先给一个最小可运行的 Python 例子，用非常简化的方式模拟 PQ 的“分段编码 + 查表近似距离”。它不是工业级实现，但可以帮助理解核心机制。

```python
import math

def l2(a, b):
    return math.sqrt(sum((x - y) ** 2 for x, y in zip(a, b)))

def encode_by_codebooks(vec, codebooks):
    # codebooks: list[list[centroid]]
    # 每段选最近质心的编号
    codes = []
    seg_dim = len(codebooks[0][0])
    for i, centroids in enumerate(codebooks):
        seg = vec[i * seg_dim:(i + 1) * seg_dim]
        best_id = min(range(len(centroids)), key=lambda j: l2(seg, centroids[j]))
        codes.append(best_id)
    return codes

def adc_distance(query, codes, codebooks):
    # 近似距离：每段查询向量到对应质心的距离之和
    seg_dim = len(codebooks[0][0])
    total = 0.0
    for i, code in enumerate(codes):
        seg = query[i * seg_dim:(i + 1) * seg_dim]
        centroid = codebooks[i][code]
        total += l2(seg, centroid)
    return total

# 两段，每段 2 维，每段两个质心
codebooks = [
    [[0.0, 0.0], [1.0, 1.0]],
    [[3.0, 3.0], [4.0, 4.0]],
]

x = [0.2, 0.1, 3.1, 2.9]
q = [0.1, 0.2, 3.2, 3.0]

codes = encode_by_codebooks(x, codebooks)
approx = adc_distance(q, codes, codebooks)
exact = l2(q, x)

assert codes == [0, 0]
assert approx >= 0
assert exact < 0.3
print("codes:", codes)
print("approx distance:", round(approx, 4))
print("exact distance:", round(exact, 4))
```

工业实现不会自己手写码本，而是直接使用向量数据库或 ANN 库。以 Qdrant 的思路看，重点有两个：

1. 建库时开启 PQ。
2. 查询时开启重评分。

示意代码如下：

```python
from qdrant_client import QdrantClient, models

client = QdrantClient(url="http://localhost:6333")

client.create_collection(
    collection_name="pq_demo",
    vectors_config=models.VectorParams(
        size=1536,
        distance=models.Distance.COSINE,
    ),
    quantization_config=models.ProductQuantization(
        product=models.ProductQuantizationConfig(
            compression=models.CompressionRatio.X32,
            always_ram=True,
        ),
    ),
)

# 查询时通常会结合更大的候选池和重评分
results = client.query_points(
    collection_name="pq_demo",
    query=[0.0] * 1536,
    limit=10,
    search_params=models.SearchParams(
        quantization=models.QuantizationSearchParams(
            rescore=True
        )
    ),
)

assert results is not None
```

真实工程例子是 RAG 记忆库。假设你要给一个智能体保存 1000 万条历史片段，每条 1536 维：

- 原始向量：约 60 GB
- 加上图索引和冗余：可能轻松破百 GB
- 开 PQ 后：主存只保留压缩码和必要结构
- 查询时：先在压缩码上找前 100 或前 200，再拿原始向量重排出前 10

这套流程的关键不是“只用 PQ 搜到底”，而是“PQ 用来筛候选，原始向量用来做最后裁决”。很多系统能把 Recall@10 的下降控制在 2% 到 5%，如果再引入 OPQ，损失还能进一步缩小。

---

## 工程权衡与常见坑

第一类坑是训练样本太少。PQ 不是固定规则压缩，而是“先学码本，再编码”。如果每个 shard 只有几千条样本，码本学不稳，某些段的质心会严重失真。结果不是压缩比不够，而是召回突然塌陷。工程上通常至少要让每个 shard 有 1 万到 10 万级训练样本，再开始训练更稳妥。

第二类坑是压缩比过高。段数太少、每段表达能力太弱时，会出现 crowding。这里可以直白理解为：本来彼此不同的向量，被压到同一个或相近编码桶里，导致排序分不开。表现形式通常是“前几个结果还行，但后面的结果抖动很大”。

第三类坑是只看压缩，不看重评分。PQ 近似检索一般要配合 overfetch，也就是“多拿一些候选”，再 rescore，也就是“用更准的方式重新排序”。如果你直接把 PQ 的 top10 当最终答案，精度损失会比预期大得多。很多线上系统真正稳定的配置不是 top10，而是“先取 top100 或 top200，再重排 top10”。

第四类坑是忽略向量分布。很多 embedding 维度相关性很强，普通 PQ 的分段方式会把高度耦合的信息切坏。OPQ 在这种场景下收益最明显，因此当你发现 PQ 的 Recall@10 掉得超出预期时，第一反应通常不是立刻放弃压缩，而是先评估 OPQ。

下面给一个常见参数判断表：

| 问题现象 | 可能原因 | 处理方式 |
|---|---|---|
| Recall 明显下降 | 训练样本不足 | 增加训练数据，再训练码本 |
| 前 1-3 名正常，后续结果很差 | 压缩过强，crowding | 增加段数，降低压缩比 |
| 查询快了，但答案不稳 | 没有 overfetch + rescore | 提高候选池并启用重评分 |
| 某类数据效果特别差 | 向量维度强相关 | 改用 OPQ 或重新选分段数 |

还有一个常见误区：把“压缩后字节数”直接当“总成本下降倍数”。实际上，系统总成本还受索引结构、冷热分层、磁盘页缓存、原始向量是否保留等因素影响。PQ 能显著降低主存压力，但不代表所有成本按同一比例缩小。

---

## 替代方案与适用边界

PQ/OPQ 不是唯一方案。实际工程里常见替代路线还有 Scalar Quantization、Binary Quantization，以及不做量化、直接靠 HNSW 扛精度。

Scalar Quantization，标量量化，可以理解为“按维度单独压缩”，实现简单，压缩比通常不如 PQ，但精度更稳，适合先做低风险降本。

Binary Quantization，二值量化，可以理解为“把向量压成位串”，极致省空间、速度也高，但对向量分布和模型特性更敏感，不是所有 embedding 都适合。

HNSW 不是量化方法，而是一种图索引结构。它的优势是高召回和低延迟，前提是向量大体放得进内存。如果数据量还在几百万以内，预算也足够，原始向量 + HNSW 往往是更省心的方案。

可以用下面的表做选择：

| 方案 | 压缩比 | 精度 | 复杂度 | 适用场景 |
|---|---:|---|---|---|
| 原始向量 + HNSW | 1x | 最高 | 中 | 中小规模，高精度优先 |
| SQ | 约 4x | 高 | 低 | 先做温和压缩 |
| PQ | 约 16x 到 64x | 中高 | 中 | 百万到亿级的主流折中 |
| OPQ + PQ | 约 16x 到 64x | 高于 PQ | 中高 | 压缩后仍要求较高召回 |
| BQ | 可很高 | 波动较大 | 中 | 极端成本敏感场景 |

真实工程上，常见成熟架构是冷热分层：

- 热数据：原始向量 + HNSW，保证高召回和低延迟
- 冷数据：IVF-PQ 或 PQ/OPQ，压缩存储，降低成本
- 查询路径：先查热层，再查冷层，最后合并重排

当数据规模进入 1B 级别，也就是十亿级，单纯要求“全量原始向量常驻内存”通常不现实，此时 IVF-PQ 或 OPQ 基本会进入候选列表。反过来，如果你的数据集只有几十万，且业务最怕精度波动，那么直接用原始向量往往更合理，没必要为未来可能永远不会到来的规模提前付出复杂度。

---

## 参考资料

- Weaviate 官方文档：Vector Quantization 与 PQ/OPQ 概念说明  
- Weaviate 官方文档：PQ Compression 配置与训练参数说明  
- Qdrant 官方文章：What is Vector Quantization  
- Qdrant 官方文档：Quantization Guide  
- FAISS / SystemOverflow：IVF-PQ 与 ADC 检索机制介绍  
- SystemOverflow：HNSW vs IVF-PQ at Billion Scale  
- Pinecone / FAISS 资料：OPQ 的旋转预处理思路
