## 核心结论

MinHash LSH 的目标不是“直接判断两篇文档是否重复”，而是先用很便宜的方法筛出“可能重复”的候选，再对候选做精算。它解决的是大规模近似去重的算力问题。

先给定义。`k-shingle` 是把文本切成长度为 $k$ 的连续片段，本质上是“把文档变成集合特征”；`Jaccard 相似度` 是两个集合重合程度，公式是：

$$
J(A,B)=\frac{|A\cap B|}{|A\cup B|}
$$

MinHash 的关键性质是：如果对两个集合做足够多次独立随机哈希，那么“签名中相等的位置比例”会逼近它们的 Jaccard 相似度。LSH 的作用是把这些签名再切成若干段分桶，只让落入同桶的文档进入下一轮校验。这样，全量 $O(n^2)$ 比对会被改成“建索引 + 桶内精算”。

结论可以压缩成三句：

| 问题 | 方法 | 结果 |
|---|---|---|
| 文档太多，不能两两算 Jaccard | 用 MinHash 把大集合压成短签名 | 近似估计相似度 |
| 候选对还是太多 | 用 LSH 按签名分桶 | 只比较局部候选 |
| 近似方法会出错 | 对候选再算精确 Jaccard | 控制误判和漏判 |

一个最小玩具例子：假设签名长度为 4，文档 A 的签名是 `[10, 3, 8, 2]`，文档 B 的签名是 `[10, 7, 8, 9]`。二者有 2 个位置相等，所以 MinHash 估计相似度约为 $2/4=0.5$。如果把签名按每 2 个值分成 2 个 band，那么第一段 `(10,3)` 与 `(10,7)` 不同，第二段 `(8,2)` 与 `(8,9)` 也不同，于是它们不会成为候选。这说明 LSH 不是“精确找重复”，而是“用召回率换速度”。

---

## 问题定义与边界

问题很具体：在大规模文本、网页、日志、商品描述、知识库片段里，找出“近似重复”的内容。这里的“近似”通常指少量删改、格式变化、标点变化、局部插入，但整体词集合高度重合。

边界也要说清楚。MinHash LSH 主要适合“集合型相似度”问题，也就是你关心的是“有哪些 token 或 shingle 出现过”，而不是它们的语义向量距离，也不是严格的编辑距离。

先看为什么不能直接全量算。若有 10 万篇文档，两两组合大约是：

$$
\frac{100000\times 99999}{2}\approx 5\times 10^9
$$

这还没算集合交并操作本身的成本。哪怕单次比较只要几十微秒，总时间也会非常大。上百万文档时，全量比较基本不可接受。

精确去重和近似去重的差异可以用一张表说清：

| 方案 | 核心比较对象 | 理论规模 | 优点 | 缺点 |
|---|---|---|---|---|
| 全量 Jaccard | 每对文档的 shingle 集合 | $O(n^2)$ | 结果最准确 | 数据大时几乎不可用 |
| MinHash | 签名向量 | 建签名近似线性 | 可估计 Jaccard | 只是近似 |
| MinHash + LSH | 同桶候选的签名/集合 | 近似线性到次二次之间 | 大规模可运行 | 有漏检和误检 |

这里有个常见误解：MinHash LSH 不是替代精确判定，而是替代“全量候选生成”。真正工程里，常见流程是：

1. 文档切 shingle
2. 生成 MinHash 签名
3. 用 LSH 找候选
4. 对候选算精确 Jaccard 或做业务规则校验
5. 只删除确认重复的文档

真实工程例子是训练数据清洗。比如要处理上亿条网页文本，很多页面只是模板变化、广告插入、页脚不同。如果直接全对全比较不可行，就先对每条文本生成 128 维 MinHash 签名，再用 32 个 band 建分桶索引，最后只对候选对做精算。这样成本从“所有文档两两比较”变成“少量桶内候选比较”。

---

## 核心机制与推导

### 1. MinHash 为什么能估计 Jaccard

把文档转成 shingle 集合后，设集合为 $A$ 和 $B$。对全集做一个随机排列，两个集合各自取“排列后第一个出现的元素”。这个最小元素相等的概率，恰好等于 $J(A,B)$。MinHash 用多个独立哈希函数模拟这种随机排列。

术语解释：`签名` 就是“用很多个最小哈希值拼出来的短向量”，它替代原始大集合参与比较。

如果有 `num_perm = m` 个哈希函数，得到长度为 $m$ 的签名，则：

$$
\hat J(A,B)=\frac{1}{m}\sum_{i=1}^{m}\mathbf{1}[h_i^{min}(A)=h_i^{min}(B)]
$$

这里 $\mathbf{1}[\cdot]$ 是指示函数，条件成立记为 1，否则记为 0。$m$ 越大，估计方差越小，结果越稳定，但存储和计算也会增加。

### 2. LSH banding 为什么能筛候选

如果只靠 MinHash，虽然单次比较变便宜了，但还是要两两比签名，规模仍接近 $O(n^2)$。LSH 的做法是把长度为 $m$ 的签名切成 $b$ 个 band，每个 band 有 $r$ 行，因此：

$$
m=b\times r
$$

两个文档只要在任意一个 band 内“整段完全相同”，就进入候选集合。

术语解释：`band` 可以理解成“签名的一个分段”；`桶` 是“同一段哈希结果相同的文档集合”。

若两个文档的 MinHash 单位置匹配概率为 $s$，则一个 band 全部匹配的概率近似是 $s^r$。至少一个 band 匹配的概率为：

$$
P(\text{成为候选}) = 1-(1-s^r)^b
$$

这条公式决定了 LSH 的 S 曲线行为。相似度低于某阈值时，进入候选的概率很小；高于阈值时，概率迅速升高。经验阈值常写作：

$$
t\approx \left(\frac{1}{b}\right)^{1/r}
$$

它不是严格分界线，但足够指导调参。

### 3. 用数值看 trade-off

还是用玩具例子。两个签名分别是：

- A = `[2,1,3,5,4,6]`
- B = `[2,2,3,5,1,6]`

总共有 4/6 个位置相等，所以估计 Jaccard 约为 $0.667$。如果切成 2 个 band，每个 band 3 行：

- Band 1: A=`[2,1,3]`, B=`[2,2,3]`，不完全相同
- Band 2: A=`[5,4,6]`, B=`[5,1,6]`，不完全相同

于是它们不会进入候选。这个例子说明：LSH 的召回率不是 100%，即使相似度不低，也可能漏掉。

参数变化趋势可以直接看表：

| 调参方向 | 结果趋势 | 直观解释 |
|---|---|---|
| 增大 `num_perm` | MinHash 估计更稳定 | 签名更长，方差更小 |
| 增大 `b`、减小 `r` | recall 上升，precision 下降 | 更容易在某个 band 撞桶 |
| 减小 `b`、增大 `r` | precision 上升，recall 下降 | 必须更大段完全一致才进候选 |
| 阈值调高 | 候选更少 | 更偏保守 |
| 阈值调低 | 候选更多 | 更偏召回 |

所以 MinHash 解决的是“如何压缩集合并估相似”，LSH 解决的是“如何不做全量候选生成”。二者组合，才构成大规模近似去重系统。

---

## 代码实现

下面先写一个不依赖第三方库的最小可运行版本，目的是把机制讲透。这个版本不追求生产性能，但能跑通“shingle -> MinHash -> LSH 候选 -> 精确 Jaccard”。

```python
import hashlib
from collections import defaultdict
from itertools import combinations

def shingles(text: str, k: int = 3):
    text = text.lower().replace(" ", "")
    if len(text) < k:
        return {text}
    return {text[i:i+k] for i in range(len(text) - k + 1)}

def stable_hash(s: str, seed: int) -> int:
    data = f"{seed}:{s}".encode("utf-8")
    return int(hashlib.md5(data).hexdigest(), 16)

def minhash_signature(tokens, num_perm=32):
    sig = []
    for seed in range(num_perm):
        sig.append(min(stable_hash(token, seed) for token in tokens))
    return sig

def jaccard(a, b):
    return len(a & b) / len(a | b)

def lsh_candidates(signatures, bands=8):
    num_perm = len(next(iter(signatures.values())))
    assert num_perm % bands == 0
    rows = num_perm // bands

    buckets = defaultdict(set)
    for doc_id, sig in signatures.items():
        for band_id in range(bands):
            start = band_id * rows
            end = start + rows
            key = (band_id, tuple(sig[start:end]))
            buckets[key].add(doc_id)

    candidates = set()
    for docs in buckets.values():
        if len(docs) > 1:
            for a, b in combinations(sorted(docs), 2):
                candidates.add((a, b))
    return candidates

docs = {
    "d1": "the quick brown fox jumps over the lazy dog",
    "d2": "the quick brown fox jumps over a lazy dog",
    "d3": "distributed systems need careful engineering",
}

token_sets = {doc_id: shingles(text, k=3) for doc_id, text in docs.items()}
signatures = {doc_id: minhash_signature(tokens, num_perm=32) for doc_id, tokens in token_sets.items()}
cands = lsh_candidates(signatures, bands=8)

# d1 和 d2 很相近，d3 明显不同
assert jaccard(token_sets["d1"], token_sets["d2"]) > 0.6
assert jaccard(token_sets["d1"], token_sets["d3"]) < 0.2

# 候选对里应该更容易出现相近文本
assert ("d1", "d2") in cands or ("d2", "d1") in cands
```

这段代码里有三个核心参数：

| 参数 | 含义 | 调大后的效果 |
|---|---|---|
| `k` | shingle 长度 | 更严格，局部改动更敏感 |
| `num_perm` | MinHash 签名长度 | 估计更稳定，成本更高 |
| `bands` | 分桶段数 | 候选更多，召回通常更高 |

如果你要直接上工程，常见做法不是自己写 LSH，而是用 `datasketch`。它已经封装了 MinHash 和 MinHashLSH。思路如下：

```python
from datasketch import MinHash, MinHashLSH

def build_minhash(text: str, k: int = 5, num_perm: int = 128):
    m = MinHash(num_perm=num_perm)
    text = text.lower().replace(" ", "")
    shingles = {text[i:i+k] for i in range(max(1, len(text) - k + 1))}
    for s in shingles:
        m.update(s.encode("utf-8"))
    return m

docs = {
    "doc1": "minhash lsh is useful for near duplicate detection",
    "doc2": "minhash lsh is useful for near-duplicate detection",
    "doc3": "vector search solves a different problem",
}

lsh = MinHashLSH(threshold=0.5, num_perm=128)

store = {}
for doc_id, text in docs.items():
    mh = build_minhash(text, k=5, num_perm=128)
    store[doc_id] = mh
    lsh.insert(doc_id, mh)

result = set(lsh.query(store["doc1"]))
assert "doc2" in result
assert "doc3" not in result
```

真实工程里通常这样分层：

1. 离线任务把原始文档切成 shingle
2. 生成 MinHash 签名并落盘
3. 用 LSH 建索引，桶可以存在内存、Redis 或分布式 KV
4. 查询时只拿同桶文档做精确 Jaccard
5. 再结合业务规则去重，比如长度差、来源域名、发布时间窗口

例如训练语料清洗场景，可能会对网页正文按 5-word shingle 建 128 维签名，把签名切成 32 个 band。LSH 先筛出近似页，再对候选对计算精确 Jaccard；若超过 0.85 就认为重复，只保留质量更高的一份。这样能把海量重复模板页、转载页、采集页先清掉。

---

## 工程权衡与常见坑

第一类权衡是“召回率和精确率不能同时无限提高”。`recall` 是“真实重复里找回了多少”；`precision` 是“判成重复的里有多少真的重复”。MinHash LSH 的本质就是在这两者之间找平衡。

最常见的参数坑如下：

| 参数 | 调大后的主要影响 | 对 recall | 对 precision | 对资源 |
|---|---|---|---|---|
| `num_perm` | 签名更长 | 一般略升或更稳定 | 一般略升或更稳定 | CPU/内存增加 |
| `bands` | band 更多、每 band 更短 | 上升 | 下降 | 桶更多 |
| `rows_per_band` | 每 band 更长 | 下降 | 上升 | 桶更稀疏 |
| `k` | shingle 更长 | 常下降 | 常上升 | 特征更稀疏 |

几个常见坑：

1. `k` 选太小。字符 2-shingle 很容易把大量无关短文本混到一起，候选爆炸。
2. `k` 选太大。文本稍微改写就不共享 shingle，召回掉得很快。
3. 只增大 `num_perm`，不联动调 `bands` 和阈值。签名更稳定不代表系统一定更好，因为 LSH 分桶结构也变了。
4. 把 LSH 返回结果当最终重复。它只是候选集合，不是最终判定。
5. 没有验证集。没有小规模标注样本，你根本不知道参数是在提 precision，还是在牺牲 recall。

一个很典型的工程误区是：在 128 维签名、32 band 的系统里，发现误报多，于是只把 `num_perm` 提到 256，但 band 配置和判定阈值不变。结果桶变稀疏，候选结构改变，真正的重复对反而更难撞到一起，recall 下降。正确做法通常是一起看三件事：

1. 小验证集上的 precision-recall 曲线
2. 每个桶的文档数分布
3. 候选总量和最终精算成本

上线前的基本流程应该是：

1. 抽取一批人工标注的重复/非重复对
2. 固定 shingle 方案，网格搜索 `num_perm`、`bands`、阈值
3. 记录候选量、召回率、精确率、平均桶大小
4. 找到满足业务目标的拐点
5. 再放大到全量数据压测

如果数据是持续流入的，还要考虑索引存储。内存版实现简单，但全量大时可能要把桶写到 Redis。此时要注意桶热点、过期策略、批量插入和查询延迟，否则 LSH 自身会变成瓶颈。

---

## 替代方案与适用边界

MinHash LSH 不是唯一方案，它只是“集合重合型近似去重”里最经典的一类。

先看对比：

| 方案 | 核心特征 | 优点 | 缺点 | 适用场景 |
|---|---|---|---|---|
| MinHash + LSH | shingle 集合 | 能估 Jaccard，适合大规模候选筛选 | 有漏检误检，需要调参 | 文本近似去重、网页去重、训练语料清洗 |
| SimHash | 加权特征指纹 | 存储小，海明距离检索快 | 对集合交并不直观，语义和词权重影响大 | 搜索去重、短文本指纹 |
| 全量 Jaccard | 原始集合 | 最准确 | 无法扩展到大规模 | 小数据集、最终复核 |
| 倒排索引初筛 | 共享 token | 工程成熟，解释性强 | 很难直接映射 Jaccard 阈值 | 搜索系统、关键词重合筛选 |

适用边界可以简单记成一句话：如果你的相似性主要来自“共享片段很多”，MinHash LSH 很合适；如果你的相似性主要来自“表达不同但语义相近”，它就不是最佳方案。

举个对比：

- 如果两篇文章只是改了排版、删了几句、加了广告模块，MinHash 很稳，因为 shingle 集合还是高度重合。
- 如果两篇文章意思一样，但完全换了一套说法，同义替换很多，MinHash 效果会下降，因为它不理解语义。
- 如果变化主要体现在顺序扰动而不是词集合变化，SimHash 有时更方便，因为它更像一种全局指纹。
- 如果数据量不大，直接精确 Jaccard 反而最省心，因为没有近似误差，也不用调参。

所以 MinHash LSH 不是“更高级的相似度算法”，而是“在大规模集合相似判定上更划算的工程结构”。

---

## 参考资料

1. `datasketch` 官方文档中关于 `MinHash`、`MinHashLSH`、`threshold`、`num_perm` 和 Redis 存储配置的说明。
2. 面向向量数据库或大规模索引系统的 MinHash LSH 参数指南，重点看 band/rows 与吞吐、召回率之间的关系。
3. 讲解 MinHash、Jaccard 与 LSH S 曲线的近重复检测文章，适合理解概率模型和工程实践中的候选筛选流程。
