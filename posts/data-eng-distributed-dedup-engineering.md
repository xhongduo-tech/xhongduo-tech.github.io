## 核心结论

TB 级文本去重不能靠“把所有文档两两比较”完成，因为全局成对比对的复杂度接近 $O(N^2)$。当文档数从百万增长到千万、上亿时，CPU 不是唯一瓶颈，内存保存候选对、网络传输中间结果、分布式任务的 shuffle 都会先失控。

可行做法是把原始文本先转换成集合表示，再用 MinHash 生成短签名，最后用 LSH 分桶，只在同桶内做局部精比对。这条链路的意义不是“把相似度计算做得更玄学”，而是把原本不可承受的全局比较，改造成大部分阶段都能线性扩展的分布式流水线。

MinHash 是“用少量哈希值近似集合相似度”的方法。更具体地说，它不直接保存完整 n-gram 集合，而是保留若干次“最小哈希抽样”结果，用较短签名近似原集合的 Jaccard 相似度。

LSH 是“让相似对象更容易落进同一桶”的索引方法。它不试图精确判定所有文档是否重复，而是先把“可能重复”的文档召回出来，再把计算预算留给这些候选。

对工程系统来说，真正重要的不是算法名词，而是流水线是否能控住三类成本：

1. 签名生成的 CPU 成本。
2. 分桶与重分发带来的 shuffle 成本。
3. 桶内候选爆炸带来的局部精比对成本。

只要这三件事能控住，TB 级去重就能从“理论可行但实际跑不动”，变成“分布式可落地”。

一个适合新手的理解顺序如下：

1. 把每篇文章拆成若干 n-gram 集合。
2. 用多组哈希函数抽样，得到固定长度的 MinHash 签名。
3. 把签名切成多个 band。
4. 每个 band 再算一个 bucket key。
5. 只有 band 完全相同、进入同一 bucket 的文档，才进入下一步精比对。
6. 把确认重复的文档连成图，最后每组只保留一个代表文档。

并行流水线可以概括成：

`数据分片 -> MinHash -> band 切分 -> bucket 分发 -> 桶内局部比对 -> 连通分量 -> 输出代表文档`

如果只记一句话，可以记这个：

> MinHash 负责把“大文本”压缩成“可比较的短签名”，LSH 负责把“全局比较”缩小成“局部候选比较”。

---

## 问题定义与边界

问题定义很具体：给定 TB 级文档集合，找出 Jaccard 相似度高于某个阈值的近重复文本，并从每组重复文档中保留一个代表记录。

Jaccard 相似度衡量两个集合的重叠程度：

$$
s=\frac{|A\cap B|}{|A\cup B|}
$$

其中：

- $A$ 和 $B$ 是两篇文档对应的 n-gram 集合。
- 分子是共有片段数。
- 分母是合并后去重的总片段数。
- $s$ 的取值范围是 $[0,1]$，越接近 1 表示越相似。

这里的集合一般不是“词集合”，而是字符 n-gram 或 token n-gram 集合。n-gram 是“长度为 n 的连续片段”；直白地说，就是把文本切成很多局部小块。这样即使原文有少量删改、插词、标点变化，两个文本仍会保留大量共同片段。

这个问题的边界需要先说清楚，否则很容易把“近重复去重”误解成“语义检索”或“精确去重”。

| 维度 | 内容 |
|---|---|
| 输入 | `doc_id + text`，通常先转成字符 n-gram 或 token n-gram 集合 |
| 输出 | 重复组、代表文档、被删除或标记的副本 |
| 目标规模 | 千万到亿级文档，TB 级原始文本 |
| 判定目标 | 找出字面上高度相似的近重复文本 |
| 关键瓶颈 | 全局成对比对不可行、shuffle 过大、热点 bucket、桶内候选爆炸 |
| 近似来源 | MinHash 与 LSH 都是近似方法，不保证 100% 精确召回 |
| 适用对象 | 网页正文、文章、评论、爬虫语料、代码片段等大规模近重复文本 |
| 不擅长对象 | 极短文本、强语义改写、跨语言重写、只保留主题但表述完全不同的文本 |

为了让边界更直观，先看几个最小例子。

### 例子 1：明显不相似

文档 A：`abcde`  
文档 B：`abXde`

取字符 3-gram：

- A 的集合：`{abc, bcd, cde}`
- B 的集合：`{abX, bXd, Xde}`

交集为空，因此：

$$
J(A,B)=\frac{0}{6}=0
$$

它们不会被视为近重复。

### 例子 2：局部修改但主体相似

文档 C：`机器学习系统设计`  
文档 D：`机器学习系统实现`

取字符 2-gram：

- C：`{机器, 器学, 学习, 习系, 系统, 统设, 设计}`
- D：`{机器, 器学, 学习, 习系, 系统, 统实, 实现}`

交集有 5 个，合集有 9 个，因此：

$$
J(C,D)=\frac{5}{9}\approx 0.556
$$

如果阈值设为 0.5，它们可能进入重复候选。

### 例子 3：为什么不用“整篇文本直接比较”

假设有 $N=10^8$ 篇文档。全局两两比较的对数约为：

$$
\frac{N(N-1)}{2}\approx 5\times10^{15}
$$

即使每次比较只花 $1$ 微秒，总时间也接近：

$$
5\times10^{15}\ \mu s = 5\times10^9\ s \approx 158\ 年
$$

这还没有算文本切分、集合构建、网络传输、调度开销。所以工程上第一件事不是“优化比较函数”，而是“避免绝大多数比较根本不会发生”。

因此，输入不是“整篇文本直接比较”，而是“文档先转成集合表示”；输出也不是“列出全部相似对”，而是“在候选集内做精比对后，输出重复簇与代表文档”。

---

## 核心机制与推导

### 1. MinHash 在近似什么

MinHash 的核心性质是：如果对集合做一次随机置换，两个集合最小元素相同的概率，等于它们的 Jaccard 相似度 $s$。

写成概率形式就是：

$$
P\bigl(\min(\pi(A))=\min(\pi(B))\bigr)=J(A,B)=s
$$

其中 $\pi$ 表示一次随机置换。

工程里当然不会真的对全集做随机置换，因为代价太大。实际做法是用多组独立哈希函数去近似这种随机抽样过程。每个哈希函数都会对集合中所有 token 取哈希值，并记录最小值，于是一次哈希函数对应签名中的一行。

如果做 $K$ 次采样，就得到长度为 $K$ 的签名：

$$
\text{sig}(A)=[m_1(A),m_2(A),...,m_K(A)]
$$

其中 $m_i(A)$ 表示第 $i$ 个哈希函数下集合 $A$ 的最小哈希值。

对任意一行，都有：

$$
P(m_i(A)=m_i(B))=s
$$

因此，签名中相等行数的比例，就是 Jaccard 的无偏估计：

$$
\hat{s}=\frac{1}{K}\sum_{i=1}^{K}\mathbf{1}[m_i(A)=m_i(B)]
$$

其中 $\mathbf{1}[\cdot]$ 是指示函数，条件成立记为 1，否则记为 0。

### 2. 为什么签名要足够长

签名太短，估计方差就大。把每一行是否相等看作一次伯努利试验，则：

$$
\mathbb{E}[\hat{s}] = s,\qquad \mathrm{Var}(\hat{s})=\frac{s(1-s)}{K}
$$

这说明：

- $K$ 越大，估计越稳定。
- $s$ 越接近 0.5，波动通常越明显。
- 短签名适合粗筛，不适合作为最终判定。

经验上：

| 签名长度 $K$ | 用途 | 特点 |
|---|---|---|
| 32 | 快速实验、玩具验证 | 误差较大，只适合感受流程 |
| 64 | 小规模粗筛 | 成本低，召回稳定性一般 |
| 128 | 常见工程起点 | 成本与效果较均衡 |
| 256+ | 高召回任务 | CPU 与存储成本更高 |

### 3. LSH 在过滤什么

即使签名只有 128 行，也不能让所有文档互相比 128 行。因为“所有文档互相比较”本身就已经爆炸了。LSH 的作用是把签名切块，只让可能相似的文档互相见面。

设签名被切成 $b$ 个 band，每个 band 有 $r$ 行，则：

$$
K=b\times r
$$

两个文档只要在某一个 band 上全部相同，就进入同一个候选桶。

先看单个 band。若文档相似度为 $s$，则该 band 内 $r$ 行全部相同的概率为：

$$
s^r
$$

那么“某个 band 不匹配”的概率是：

$$
1-s^r
$$

“所有 $b$ 个 band 都不匹配”的概率是：

$$
(1-s^r)^b
$$

因此至少有一个 band 匹配，也就是成为候选对的概率为：

$$
P(s)=1-(1-s^r)^b
$$

这就是 LSH 的 S 曲线来源。

### 4. 这条 S 曲线意味着什么

白话讲：

- 相似度低的文档，很难在整段 band 上完全一致，因此大概率不会进候选集。
- 相似度高的文档，至少有一个 band 完全一致的概率会迅速升高，因此大概率能被召回。

看一个具体参数：$b=20,\ r=5$。

| Jaccard 相似度 $s$ | $s^r$ | 候选概率 $P(s)=1-(1-s^r)^b$ |
|---|---:|---:|
| 0.2 | 0.00032 | 0.006 |
| 0.4 | 0.01024 | 0.186 |
| 0.6 | 0.07776 | 0.801 |
| 0.8 | 0.32768 | 0.9996 |
| 0.9 | 0.59049 | 接近 1 |

这张表说明两件事：

1. 低相似度文档会被自然滤掉。
2. 高相似度文档会以高概率进入候选集。

### 5. band 和 row 怎么理解

新手最容易困惑的是：为什么不是“越多 band 越好”或“每个 band 越长越好”。

它们是对冲关系：

- 增大 $b$：更多 band，意味着“给文档更多撞桶机会”，召回会上升，但候选数也可能上升。
- 增大 $r$：每个 band 更严格，意味着“必须更多连续签名行都相同才算撞桶”，精度上升，但召回可能下降。

可以用一个近似阈值来帮助理解：

$$
t \approx \left(\frac{1}{b}\right)^{1/r}
$$

这里的 $t$ 不是精确阈值，只是 S 曲线开始明显抬升的大致位置。

例如：

| $b$ | $r$ | 近似拐点 $t\approx (1/b)^{1/r}$ | 直观含义 |
|---:|---:|---:|---|
| 20 | 5 | 0.55 | 0.55 以上相似度更容易被召回 |
| 32 | 4 | 0.42 | 更宽松，候选更多 |
| 16 | 8 | 0.71 | 更严格，适合高相似去重 |

所以，LSH 参数不是越大越好，而是要围绕业务阈值调形状。你想抓住 $s \ge 0.8$ 的重复文本，就应让曲线在 0.8 附近快速抬升，而不是把大量 0.3、0.4 的弱相似文档也拉进来。

### 6. 一个更完整的玩具例子

签名长度 6，分成 2 个 band，每个 band 3 行：

- 文档 A：`[2,1,3 | 5,4,6]`
- 文档 B：`[2,2,3 | 5,1,6]`
- 文档 C：`[2,1,3 | 8,7,9]`

比较结果：

| 文档对 | band 1 | band 2 | 是否进候选 |
|---|---|---|---|
| A vs B | 不完全一致 | 不完全一致 | 否 |
| A vs C | 完全一致 | 不完全一致 | 是 |
| B vs C | 不完全一致 | 不完全一致 | 否 |

这就是 LSH 真正省成本的地方：不是“更快地比较所有文档”，而是“尽量不让低相似文档进入比较阶段”。

因此，“MinHash -> band -> bucket -> 候选”的链路，本质上是在做一层可调的粗筛，把全局二次复杂度改造成局部可控复杂度。

---

## 代码实现

下面先给一个可以直接运行的 Python 示例。它覆盖了完整链路：

1. 文本转 n-gram 集合。
2. 生成 MinHash 签名。
3. 按 band 分桶。
4. 生成候选对。
5. 在候选对上做精确 Jaccard 比对。
6. 用并查集汇总重复簇。
7. 为每个簇选代表文档。

这个版本只依赖 Python 标准库，可以直接运行。

```python
from collections import defaultdict
from itertools import combinations
import hashlib
import math


def ngrams(text, n=3):
    text = text.strip()
    if not text:
        return {""}
    if len(text) < n:
        return {text}
    return {text[i:i + n] for i in range(len(text) - n + 1)}


def stable_hash(value, seed):
    raw = f"{seed}:{value}".encode("utf-8")
    # 使用 blake2b 得到稳定、可复现的 64 位整数
    return int.from_bytes(hashlib.blake2b(raw, digest_size=8).digest(), "big")


def minhash_signature(tokens, num_perm=64):
    if not tokens:
        tokens = {""}
    sig = []
    for seed in range(num_perm):
        sig.append(min(stable_hash(token, seed) for token in tokens))
    return sig


def lsh_buckets(doc_sigs, bands=16, rows=4):
    if bands * rows <= 0:
        raise ValueError("bands * rows must be positive")

    buckets = defaultdict(list)
    for doc_id, sig in doc_sigs.items():
        if len(sig) != bands * rows:
            raise ValueError(f"signature length mismatch for {doc_id}")
        for band_id in range(bands):
            start = band_id * rows
            band_slice = tuple(sig[start:start + rows])
            bucket_key = stable_hash(str(band_slice), seed=band_id)
            buckets[(band_id, bucket_key)].append(doc_id)
    return buckets


def jaccard(a, b):
    union = a | b
    if not union:
        return 1.0
    return len(a & b) / len(union)


class UnionFind:
    def __init__(self, items):
        self.parent = {x: x for x in items}
        self.rank = {x: 0 for x in items}

    def find(self, x):
        parent = self.parent
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(self, a, b):
        ra = self.find(a)
        rb = self.find(b)
        if ra == rb:
            return
        if self.rank[ra] < self.rank[rb]:
            ra, rb = rb, ra
        self.parent[rb] = ra
        if self.rank[ra] == self.rank[rb]:
            self.rank[ra] += 1


def choose_representative(group, docs):
    # 示例策略：优先保留文本更长的；长度相同则按 doc_id 排序
    return sorted(group, key=lambda doc_id: (-len(docs[doc_id]), doc_id))[0]


def deduplicate(
    docs,
    n=2,
    num_perm=64,
    bands=16,
    rows=4,
    jaccard_threshold=0.5,
    max_bucket_size=1000,
):
    if bands * rows != num_perm:
        raise ValueError("num_perm must equal bands * rows")

    doc_sets = {doc_id: ngrams(text, n=n) for doc_id, text in docs.items()}
    doc_sigs = {
        doc_id: minhash_signature(tokens, num_perm=num_perm)
        for doc_id, tokens in doc_sets.items()
    }

    buckets = lsh_buckets(doc_sigs, bands=bands, rows=rows)

    candidate_pairs = set()
    skipped_hot_buckets = 0

    for (_, _), doc_ids in buckets.items():
        unique_ids = sorted(set(doc_ids))
        if len(unique_ids) < 2:
            continue
        if len(unique_ids) > max_bucket_size:
            skipped_hot_buckets += 1
            continue
        for a, b in combinations(unique_ids, 2):
            candidate_pairs.add((a, b))

    uf = UnionFind(docs.keys())
    exact_edges = []

    for a, b in sorted(candidate_pairs):
        score = jaccard(doc_sets[a], doc_sets[b])
        if score >= jaccard_threshold:
            uf.union(a, b)
            exact_edges.append((a, b, round(score, 4)))

    groups = defaultdict(list)
    for doc_id in docs:
        groups[uf.find(doc_id)].append(doc_id)

    dedup_groups = []
    representatives = {}

    for root, group in groups.items():
        group_sorted = sorted(group)
        dedup_groups.append(group_sorted)
        representatives[root] = choose_representative(group_sorted, docs)

    dedup_groups.sort()

    stats = {
        "num_docs": len(docs),
        "num_candidates": len(candidate_pairs),
        "num_exact_edges": len(exact_edges),
        "num_groups": len(dedup_groups),
        "skipped_hot_buckets": skipped_hot_buckets,
    }

    return {
        "groups": dedup_groups,
        "representatives": representatives,
        "exact_edges": exact_edges,
        "stats": stats,
    }


if __name__ == "__main__":
    docs = {
        "d1": "机器学习系统设计基础",
        "d2": "机器学习系统设计入门",
        "d3": "今天天气很好适合散步",
        "d4": "机器学习系统设计基础课程",
        "d5": "今天天气不错适合散步",
        "d6": "数据库事务与隔离级别",
    }

    result = deduplicate(
        docs,
        n=2,
        num_perm=64,
        bands=16,
        rows=4,
        jaccard_threshold=0.45,
        max_bucket_size=100,
    )

    print("groups:")
    for group in result["groups"]:
        print("  ", group)

    print("\nexact duplicate edges:")
    for edge in result["exact_edges"]:
        print("  ", edge)

    print("\nstats:")
    for k, v in result["stats"].items():
        print(f"  {k}: {v}")

    # 基本断言，保证示例可以运行并给出稳定输出
    groups_as_sets = [set(group) for group in result["groups"]]
    assert any({"d1", "d2", "d4"}.issubset(group) for group in groups_as_sets)
    assert any({"d3", "d5"}.issubset(group) for group in groups_as_sets)
    assert any(group == {"d6"} for group in groups_as_sets)
```

### 运行后会发生什么

这个示例里：

- `d1`、`d2`、`d4` 都是“机器学习系统设计...”的近重复文本。
- `d3`、`d5` 是“天气适合散步”的近重复文本。
- `d6` 与其他文本主题和片段都差异较大，通常会单独成组。

这个流程有几个工程意义：

| 步骤 | 作用 | 为什么不能省 |
|---|---|---|
| `ngrams` | 把原文变成可比较的集合 | 没有集合表示就无法用 Jaccard/MinHash |
| `minhash_signature` | 把大集合压缩成定长签名 | 否则存储和比较代价太高 |
| `lsh_buckets` | 只让可能相似的文档相遇 | 否则还是接近全局比较 |
| `jaccard` 精比对 | 修正近似召回带来的误报 | LSH 不是最终判定 |
| `UnionFind` | 把成对重复合并成重复簇 | 真实重复通常是成组出现，不只是成对 |

### 分布式伪代码

真实工程里，单机代码只是验证机制。生产环境通常会把“签名生成”和“分桶候选召回”放在 Spark、Flink 或 MapReduce 类系统中执行。下面给一个更接近实际任务的 Spark 风格伪代码：

```python
from pyspark.sql import functions as F, types as T

NUM_PERM = 128
BANDS = 32
ROWS = 4
JACCARD_THRESHOLD = 0.8

# 1. 读取原始数据，只保留必要字段
docs = (
    spark.read.parquet("s3://bucket/raw_docs")
    .select("doc_id", "text", "publish_time", "quality_score")
)

# 2. 文本 -> n-gram -> MinHash
@F.udf(returnType=T.ArrayType(T.LongType()))
def build_signature(text):
    # 内部可用 datasketch 或自定义稳定实现
    ...
    return signature

sig_df = docs.select(
    "doc_id",
    "publish_time",
    "quality_score",
    build_signature("text").alias("sig")
)

# 3. 拆 band，输出轻量键值
band_schema = T.ArrayType(
    T.StructType([
        T.StructField("band_id", T.IntegerType()),
        T.StructField("bucket_key", T.StringType())
    ])
)

@F.udf(returnType=band_schema)
def split_bands(sig):
    ...
    return [{"band_id": i, "bucket_key": "..."} for i in range(BANDS)]

bucket_df = (
    sig_df
    .select("doc_id", F.explode(split_bands("sig")).alias("band"))
    .select(
        "doc_id",
        F.col("band.band_id").alias("band_id"),
        F.col("band.bucket_key").alias("bucket_key")
    )
)

# 4. 只传递 (band_id, bucket_key, doc_id) 进入 shuffle
grouped = (
    bucket_df
    .groupBy("band_id", "bucket_key")
    .agg(F.collect_list("doc_id").alias("doc_ids"))
)

# 5. 热点 bucket 做限流或旁路
small_buckets = grouped.filter(F.size("doc_ids") <= 1000)
hot_buckets = grouped.filter(F.size("doc_ids") > 1000)

# 6. 桶内局部精比对
dup_edges = small_buckets.rdd.flatMap(compare_within_bucket).toDF(["src", "dst"])

# 7. 图连通分量，得到重复簇
components = run_connected_components(dup_edges)

# 8. 选择代表文档
result = select_representative(
    components=components,
    docs_meta=docs.select("doc_id", "publish_time", "quality_score")
)

result.write.mode("overwrite").parquet("s3://bucket/dedup_result")
```

### 真实工程里的最小可行数据流

假设你要清洗 10TB 网页正文数据，目标是去掉转载、镜像站、模板化重复页面，最小可行链路通常是：

1. 按分区读取原始文本，只保留 `doc_id`、`text`、少量保留策略字段。
2. 在 executor 本地完成 n-gram 与 MinHash，避免把中间集合对象发到网络。
3. 只把 `(band_id, bucket_key, doc_id)` 这类轻量记录发到 shuffle。
4. 在同 bucket 内做局部精比对。
5. 把通过阈值的重复边汇总成连通分量。
6. 每个簇按规则保留一条代表文档，比如最早发布时间、最高质量分或最规范 URL。

这里的关键不是“算法会不会”，而是“中间数据格式够不够瘦”。如果把全文、完整 n-gram 集合甚至原始 HTML 一起带进 shuffle，网络和磁盘 spill 往往会先成为瓶颈。

---

## 工程权衡与常见坑

真正把系统跑稳，靠的是参数、数据流和监控，不是公式本身。

先看常见问题表：

| 坑/症状 | 典型原因 | 缓解方式 |
|---|---|---|
| shuffle 字节数暴涨 | bucket 过多，或中间字段太胖 | 只传 `doc_id + band key`，不要把全文带进 shuffle |
| 某些 task 极慢 | 热点 bucket 过大，数据倾斜 | 调整 `b/r`，热点桶拆分，单独旁路处理 |
| 磁盘 spill 很高 | 桶内聚合超出内存 | 增加分区数，限制 bucket 大小，改成流式比对 |
| 候选对数量爆炸 | 参数过松，低相似文本也进候选 | 增大 `r`、减小 `b`，或提高精比对阈值 |
| 召回明显下降 | 参数过严，高相似文档没撞桶 | 增大 `b`，增加签名长度 |
| executor OOM | 单桶文档过多，或原文对象常驻内存 | 热点旁路、二次分桶、按需回源读取全文 |
| 去重结果不稳定 | 哈希实现不稳定，签名不可复现 | 使用固定 seed 与稳定哈希函数 |
| 代表文档选错 | 只做聚类，不做保留策略 | 单独设计 representative 规则 |

下面把几个最常见误区展开说清楚。

### 1. 误把 LSH 当成最终判定

不对。LSH 的职责是候选召回，不是最终判定。进入同一个 bucket，只能说明“可能重复”，不能说明“一定重复”。

工程上一般会有两层判断：

1. `LSH` 负责粗筛。
2. `Jaccard` 或更高精度相似度负责精判。

如果直接把“同桶”当成“重复”，会出现大量误报，尤其是在模板化文本、页脚、协议页、导航页很多的数据集里。

### 2. 忽略热点 bucket

现实数据里总会有一些高频公共结构，例如：

- 通用免责声明
- 电商模板页
- 页头页脚
- 协议、版权声明
- 机器生成的统一栏目文本

这些结构会产生大量相似 n-gram，导致某些 bucket 特别热。一个热点 bucket 就足以拖慢整批任务。

热点 bucket 常见处理方式如下：

| 方法 | 思路 | 适用场景 |
|---|---|---|
| 直接截断 | 单桶超过阈值就不在主流程处理 | 先保主流程稳定 |
| 二次分桶 | 对热点桶再按前缀或额外哈希切分 | 热点很多但仍想保召回 |
| 旁路任务 | 热点桶交给单独任务慢慢算 | 线上主链路有 SLA |
| 先做模板过滤 | 去掉高频 boilerplate 再签名 | 网页类数据特别有效 |

### 3. 把 band-row 参数写死

$b$ 和 $r$ 决定召回与成本平衡，没有通用最优配置。

不同数据类型通常差异很大：

| 数据类型 | 推荐关注点 |
|---|---|
| 长网页正文 | 模板噪声、热点 bucket、长尾 task |
| 短评论 | n-gram 稀疏，误判风险高 |
| 标题/标签 | 文本过短，LSH 往往不稳定 |
| 代码片段 | token 化方式更重要，字符 n-gram 未必最好 |
| 多语言语料 | 需要先做语言分流，否则特征噪声大 |

因此参数调优必须基于样本集做离线评估，而不是抄一份配置直接上线。

一个实用调参流程通常是：

1. 先人工标注一批正负样本对。
2. 固定 n-gram 方案，扫描多个 `num_perm / b / r` 组合。
3. 观察召回率、精确率、候选总数、桶分布。
4. 在召回可接受前提下，优先压低候选数与热点桶比例。
5. 最后再观察分布式资源指标是否能承受。

### 4. 桶内仍然做全量两两比较

即使局部桶比全局小，热点桶内部仍可能很大。桶内如果直接做 $O(m^2)$ 比较，还是会爆。

设某个桶里有 $m$ 篇文档，则两两比较数为：

$$
\frac{m(m-1)}{2}
$$

当 $m=10,000$ 时，候选对数约为：

$$
\frac{10000\times 9999}{2}\approx 5\times10^7
$$

单桶就五千万对，完全足以拖垮一个 task。

因此桶内需要 early pruning，也就是提前剪枝。常见规则如下：

| 剪枝规则 | 原因 |
|---|---|
| 长度差异过大直接跳过 | 高度重复文本长度通常不会相差过大 |
| 语言不一致直接跳过 | 不同语种共享 n-gram 很少 |
| 标题哈希完全不同且正文长度差大 | 快速过滤明显不相似项 |
| 高频模板片段占比过高 | 说明相似性可能由模板而非正文造成 |
| 先比更便宜的特征 | 用低成本特征减少精比对次数 |

### 5. 只关注算法正确，不关注保留策略

去重系统不是只输出“哪些重复”，还要回答“保留谁”。

常见保留策略包括：

| 策略 | 适用场景 |
|---|---|
| 最早发布时间优先 | 新闻、网页归档 |
| 质量分最高优先 | 搜索索引、内容聚合 |
| URL 最规范优先 | 同站镜像与参数页 |
| 文本最长优先 | 采集页与摘要页混杂 |
| 主站域名优先 | 抓取了转载与原站时 |

如果代表文档选择不合理，去重结果会伤害下游质量，例如删掉原始页面、保留低质量镜像页。

### 6. 新手最容易忽略的监控项

监控上至少要盯这些指标：

- `shuffle read bytes`
- `shuffle write bytes`
- `spill to disk`
- `spill to memory`
- `executor task duration`
- `records per bucket`
- `95/99 分位 bucket size`
- `候选对总数`
- `最终重复边数量`
- `去重率`
- `热点 bucket 占比`
- `每万篇文档平均候选数`

可以进一步把这些指标和问题对应起来：

| 指标异常 | 常见含义 |
|---|---|
| `shuffle write bytes` 突增 | 中间键值太胖，或 bucket 数暴涨 |
| `99 分位 bucket size` 过高 | 数据倾斜明显，有热点模板 |
| `候选对总数` 激增 | LSH 参数太松，粗筛失效 |
| `去重率` 断崖下跌 | 参数过严、哈希实现异常、数据预处理变了 |
| `spill to disk` 长期高位 | 内存与聚合策略失衡，不应只靠加机器 |

如果任务阶段中大量 executor 出现高比例 spill，说明桶内聚合策略已经失衡，此时应该先改分桶参数、热点策略和中间数据格式，而不是只做横向扩容。

新手可以先记住一个简单的稳定性策略：

> 当 bucket 内文档太多时，不要硬算。先截断、再细分、或旁路处理，让主流程先稳定。

---

## 替代方案与适用边界

LSH 不是唯一去重方案，它适合的是“大规模集合相似去重”，尤其是近重复文本，而不是所有重复问题。

| 方案 | 适用对象 | 相似度类型 | 优点 | 局限 | 硬件成本 |
|---|---|---|---|---|---|
| MinHash + LSH | 中长文本、网页、文档 | Jaccard 近似 | 易分布式扩展，适合 TB 级 | 参数敏感，短文本不稳定 | 中 |
| Bloom Filter | 精确键或短特征 | 是否出现过 | 极省内存，挡重高效 | 不能表达近似相似 | 低 |
| Trie / 倒排索引 | 固定字段、规则型匹配 | 精确或规则匹配 | 查询快，规则清晰 | 对自由文本近重复能力弱 | 低到中 |
| SimHash | 短文本、快速汉明距离过滤 | 位级近似 | 签名更短，比较快 | 对集合相似的解释性不如 MinHash 直接 | 低到中 |
| Embedding + ANN | 语义相似文本 | 向量相似度 | 能抓改写、释义、语义重复 | 推理和索引成本高，边界更难解释 | 高 |

### 1. Bloom Filter 解决的是“见过没有”

Bloom Filter 是“用位数组近似记录某元素是否出现过”的结构。它适合回答：

- 某个 URL 是否见过
- 某个 ID 是否重复
- 某个短 fingerprint 是否已存在

它不适合回答“这两篇文章像不像”，因为它不建模相似度，只建模成员存在性。

所以它适合精确键去重，不适合网页正文近重复。

### 2. Trie / 倒排索引解决的是规则匹配

如果你的重复定义是：

- 标题完全一致
- URL 前缀一致
- 某字段只差参数
- 某段固定模板相同

那么 Trie、倒排索引或规则系统可能更直接，而且更可解释。

但对自由文本的局部改写、删句、插句、改标题，这类方法通常不够稳健。

### 3. Embedding + ANN 解决的是“语义近似”

Embedding 是把文本编码成向量；ANN 是在大规模向量中快速找近邻。

这条路线更适合：

- 改写但意思一致
- 摘要式重写
- 语义接近但表述不同
- 跨模板改写内容

但它的成本更高，边界也更难定义。两个语义相近的文本是否应被视为“重复”，往往取决于业务规则，而不是单纯距离阈值。

### 4. SimHash 是什么位置

SimHash 常用于海量网页近重复检测，尤其在“快速指纹 + 汉明距离”场景中有实用价值。它和 MinHash 的区别可以简化理解为：

| 方法 | 更贴近哪类相似度 |
|---|---|
| MinHash | 集合重叠程度，天然对应 Jaccard |
| SimHash | 特征加权后的整体相近性，常用汉明距离衡量 |

如果你的系统已经围绕 Jaccard、n-gram 集合和 LSH 组织，MinHash 更自然。如果你追求更短签名和极快比较，SimHash 可以作为对照方案做实验。

### 5. 怎么选方案

可以用下面的决策表快速判断：

| 场景 | 更合适的方案 |
|---|---|
| 只判断某键是否出现过 | Bloom Filter |
| 网页正文、文章、代码片段近重复 | MinHash + LSH |
| 标题、URL、规则型字段去重 | Trie / 倒排 / 规则系统 |
| 改写、释义、语义重复 | Embedding + ANN |
| 文本极短，且只想要快速近似挡重 | SimHash 或规则方案 |

因此适用边界可以概括为：

- 如果你处理的是海量网页正文、文章、代码片段，且关注字面近重复，LSH 很合适。
- 如果你只是判断某个键是否出现过，Bloom Filter 更直接。
- 如果你要抓“改写但意思一样”的语义重复，Embedding + ANN 更有优势。
- 如果文本极短，比如标题、标签、两三句话，LSH 的集合表示会过于稀疏，效果可能不稳定。

---

## 参考资料

下面给出一组更适合工程实现的参考资料。它们的作用不是“背定义”，而是帮助你分别建立算法理解、分布式实现和参数调优三层认知。

| 资料 | 主要价值 | 适合阅读阶段 |
|---|---|---|
| Andrei Z. Broder, *On the Resemblance and Containment of Documents* | MinHash 与文档近重复的经典论文，建立 Jaccard/Resemblance 视角 | 理解原理 |
| Mining of Massive Datasets 中关于 Locality-Sensitive Hashing 的章节 | 对 MinHash、LSH、S 曲线推导讲得系统 | 理解公式与参数 |
| `datasketch` 官方文档 | Python 中 MinHash/LSH 的直接工程接口 | 快速原型 |
| Hugging Face / BigCode 的 near-dedup 方案说明 | 展示大规模数据集清洗中 near-dedup 的完整链路 | 看工程落地 |
| Spark MLlib 中 `MinHashLSH` 的官方文档 | 理解分布式 API 怎么组织输入、签名和近邻查询 | 看分布式实现 |
| 阿里云 EMR 或类似 Spark 实战文章 | 帮助把签名、分桶、shuffle、热点桶、图聚合串起来 | 看生产问题 |
| 图计算或 Union-Find 资料 | 解决“重复是成组出现，不是只成对出现”的问题 | 完成最后一跳 |

如果按新手阅读顺序，建议这样看：

1. 先看 `Jaccard + MinHash` 的基本性质，知道“为什么签名行相等概率等于相似度”。
2. 再看 `LSH` 的 band/row 与公式 $P(s)=1-(1-s^r)^b$，知道“为什么能减少候选”。
3. 然后看 `datasketch` 或 `Spark MinHashLSH` 的接口，先把小数据跑通。
4. 最后看 BigCode、Hugging Face 或云厂商实战文章，理解 shuffle、热点 bucket、连通分量这些工程问题。

一个更务实的结论是：

> 算法入门资料解决“为什么可行”，工程资料解决“为什么总是跑崩”。两者缺一不可。
