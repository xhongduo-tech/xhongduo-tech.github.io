## 核心结论

嵌入相似度指的是：给定查询向量 `q` 和候选向量 `x`，用一个分数或距离来决定谁排在前面。真正决定结果的，不只是“选内积还是欧氏距离”，而是 `训练目标 + 向量归一化 + 检索度量` 这三个因素是否一致。

同一批向量，训练时看起来很准，上线后却排序变了，最常见的原因是：训练时优化的是余弦相似度，线上召回却直接用未归一化内积，结果向量模长也参与了排序。

| 训练目标 | 向量是否归一化 | 检索度量 | 常见后果 |
|---|---|---|---|
| 优化 cosine | 训练后做 L2 归一化 | dot / cosine / squared L2 | 排序通常可互换 |
| 优化 dot | 不归一化 | dot | 模长会参与排序 |
| 优化 dot | 不归一化 | L2 | 可能与训练目标偏离 |
| KG 关系打分 | 实体可归一化也可不归一化 | 仅实体近邻 | 只能粗召回，不能替代关系推理 |

中心判断只有一句：离线训练、索引构建、线上检索三者一致，排序才稳定；三者不一致，指标会漂移。

---

## 问题定义与边界

本文只讨论“向量相似度如何影响排序”，不讨论嵌入模型本身怎么训练得更好，也不展开 ANN 索引的实现细节。

统一记号如下：

| 记号 | 含义 | 白话解释 |
|---|---|---|
| `q` | query vector | 你拿去检索别人的那个向量 |
| `x` | candidate vector | 候选库里被比较的向量 |
| `r` | relation vector | 在知识图谱里表示关系类型的向量 |

先给新手版结论：两个向量很近，不等于“语义相同”，只等于“在当前度量下接近”。度量换了，排序就可能变。

“相似度”和“距离”也不是同一个东西：

| 类型 | 典型形式 | 排序规则 |
|---|---|---|
| 相似度 | 越大越像，如 `dot`、`cosine` | 从大到小排 |
| 距离 | 越小越近，如 `L2` | 从小到大排 |

知识图谱里还要再多加一层区分：

| 概念 | 含义 |
|---|---|
| 实体相似 | 两个实体整体上像不像 |
| 关系可连接 | 在某个具体关系下能不能互相连通 |

边界例子很重要。`Apple` 和 `Microsoft` 可能在“科技公司”这个实体空间里很接近，但输入 `(Apple, acquired, ?)` 时，`Microsoft` 接近并不意味着它能替代真实的被收购对象。也就是说，实体近邻不是关系推理。

---

## 核心机制与推导

三种最常见度量分别是：

$$
s_{ip}(q, x) = q \cdot x
$$

$$
s_{cos}(q, x) = \frac{q \cdot x}{\|q\| \|x\|}
$$

$$
d^2(q, x) = \|q - x\|^2
$$

它们关注的信号不同：

| 度量 | 看方向 | 看模长 | 典型含义 |
|---|---|---|---|
| 内积 `dot` | 看 | 也看 | 方向一致且模长大的向量会被抬高 |
| 余弦 `cosine` | 主要看 | 基本不看 | 只比较夹角 |
| 平方欧氏距离 `squared L2` | 看 | 也看 | 直接比较几何位置差 |

先看一个玩具例子。令 `q = (1, 0)`，候选为 `a = (2, 0)`，`b = (0.9, 0.9)`。

| 向量 | `q·x` | `||q-x||^2` | 结论 |
|---|---:|---:|---|
| `a=(2,0)` | `2` | `1` | 按内积更优 |
| `b=(0.9,0.9)` | `0.9` | `0.82` | 按 L2 更优 |

这里排序反了。原因不是算法错，而是 `dot` 把 `a` 的模长当成了加分项，而 `L2` 更关心几何位置。

再把它们归一化。设 `a'=(1,0)`，`b'=(0.8,0.6)`，并且 `q` 本身已经是单位向量。此时：

| 向量 | `dot` | `cosine` | `||q-x||^2` |
|---|---:|---:|---:|
| `a'` | `1.0` | `1.0` | `0` |
| `b'` | `0.8` | `0.8` | `0.4` |

排序一致了。原因可以直接推导：

当 `\|q\|=\|x\|=1` 时，

$$
\|q-x\|^2 = \|q\|^2 + \|x\|^2 - 2 q \cdot x = 2 - 2(q \cdot x)
$$

又因为单位向量时 `q \cdot x = s_{cos}(q, x)`，所以：

$$
d^2(q, x) = 2 - 2s_{ip}(q, x) = 2 - 2s_{cos}(q, x)
$$

这说明在单位球面上，最大内积、最大余弦、最小平方欧氏距离的排序一致。注意关键词是“单位球面”。只要不满足这个条件，三者就不保证一致。

知识图谱进一步说明了“相似度不够用”。经典关系打分模型会把关系 `r` 显式纳入评分：

$$
\text{TransE}(h,r,t) = -\|h + r - t\|_p
$$

白话解释：如果头实体 `h` 经过关系 `r` 的平移后接近尾实体 `t`，这个三元组就更合理。

RotatE 的常见写法是：

$$
\text{RotatE}(h,r,t) = -\|h \circ r - t\|_1
$$

其中 `\circ` 是逐元素乘法。白话解释：关系被当成复空间里的“旋转”，不是简单的最近邻。

所以在 KG 场景里，`Apple` 和 `Microsoft` 向量接近，只能说明它们像，不说明 `(Apple, acquired, Microsoft)` 这个关系成立。

---

## 代码实现

下面给一个最小可运行版本，统一演示归一化、三种度量、排序，以及 KG 两阶段“召回 + 精排”的思路。

```python
import math

def l2_norm(v):
    return math.sqrt(sum(x * x for x in v))

def normalize(v):
    n = l2_norm(v)
    assert n > 0, "zero vector cannot be normalized"
    return [x / n for x in v]

def dot(a, b):
    return sum(x * y for x, y in zip(a, b))

def cosine(a, b):
    return dot(a, b) / (l2_norm(a) * l2_norm(b))

def squared_l2(a, b):
    return sum((x - y) ** 2 for x, y in zip(a, b))

def rank(query, candidates, metric, reverse=True):
    scored = [(name, metric(query, vec)) for name, vec in candidates.items()]
    return sorted(scored, key=lambda x: x[1], reverse=reverse)

# 玩具例子
q = [1.0, 0.0]
cands = {
    "a": [2.0, 0.0],
    "b": [0.9, 0.9],
}

by_dot = rank(q, cands, dot, reverse=True)
by_l2 = rank(q, cands, squared_l2, reverse=False)

assert by_dot[0][0] == "a"
assert by_l2[0][0] == "b"

# 归一化后排序一致
qn = normalize(q)
norm_cands = {k: normalize(v) for k, v in cands.items()}

by_dot_norm = rank(qn, norm_cands, dot, reverse=True)
by_cos_norm = rank(qn, norm_cands, cosine, reverse=True)
by_l2_norm = rank(qn, norm_cands, squared_l2, reverse=False)

assert [x[0] for x in by_dot_norm] == [x[0] for x in by_cos_norm]
assert [x[0] for x in by_dot_norm] == [x[0] for x in by_l2_norm]

# KG 两阶段示意：先按实体近邻召回，再按关系分数精排
def transe_score(h, r, t):
    return -sum(abs(h[i] + r[i] - t[i]) for i in range(len(h)))

head = [0.9, 0.1]
rel_acquired = [0.3, -0.2]
tails = {
    "Beats": [1.2, -0.1],
    "Microsoft": [0.95, 0.05],
    "Shazam": [1.25, -0.05],
}

# 阶段1：实体近邻召回
recall_topk = rank(head, tails, cosine, reverse=True)[:2]
assert len(recall_topk) == 2

# 阶段2：关系特定精排
reranked = sorted(
    [(name, transe_score(head, rel_acquired, tails[name])) for name, _ in recall_topk],
    key=lambda x: x[1],
    reverse=True,
)
assert reranked[0][0] in {"Beats", "Shazam"}
print("ok")
```

工程实现上有三个直接原则。

第一，如果向量已经做了 L2 归一化，优先用 `dot`。原因不是数学更对，而是它通常更省计算，且与 cosine 排序一致。

第二，如果系统里写的是 `L2`，必须确认库返回的是不是 `squared L2`。很多检索库为了省掉开方，直接返回平方距离。排序不变，但阈值解释会变。

第三，如果是知识图谱任务，尽量把“候选发现”和“关系打分”拆开。真实工程例子通常是：先用 ANN 从百万级实体库里召回 top-K 尾实体，再用 `TransE`、`RotatE` 或别的关系模型精排。前者解决速度，后者解决语义约束。

---

## 工程权衡与常见坑

理论上等价，不代表线上能随便替换。因为缓存内容、索引类型、训练 loss、归一化策略、阈值定义，都会让“看起来等价”的方案在工程上不等价。

一个常见真实例子是：文本检索模型训练时优化的是 cosine，相当于希望方向一致的向量排前；但线上向量库图省事，直接把未归一化向量用内积召回。结果高频文本、长文本或高范数样本被系统性提前，离线验证通过，线上点击排序却漂了。

| 坑 | 后果 | 规避 |
|---|---|---|
| 训练用 `cosine`，线上用未归一化 `dot` | 排序漂移 | 训练、建库、推理统一度量 |
| 把 `L2` 和 `squared L2` 混淆 | 阈值和监控解释错误 | 明确库返回值定义 |
| 向量已归一化，还重复算 cosine | 多余开销 | 归一化后直接用 `dot` |
| 忽略模长语义 | 丢失置信度或频率信息 | 先确认模长是否被模型当成信号 |
| 只做实体近邻，不做关系重排 | KG 错链路增多 | ANN 召回后接关系模型精排 |

可以用一条简单检查清单排查：

| 检查项 | 要问的问题 |
|---|---|
| 训练目标 | loss 优化的是 dot、cosine 还是某种关系分数？ |
| 归一化 | 训练前、入库前、查询前是否统一做了 L2 normalize？ |
| 索引度量 | 向量库到底按 inner product 还是 L2 检索？ |
| 返回值 | 是 similarity 还是 distance？是 L2 还是 squared L2？ |
| 业务解释 | 模长代表噪声、热度、置信度，还是纯副产物？ |

“模长什么时候是噪声，什么时候是信号”要单独说明。

如果模型训练时就把向量投到单位球面，模长基本是被消掉的，这时方向是主信号，余弦更自然。  
如果模型没有强制归一化，且样本频率、置信度、曝光强度会体现在模长里，那么内积保留了这些信息，不能随便换成余弦。  
如果你的任务要表达“半径内都算邻居”，比如地理坐标、物理位置或某些聚类后处理，欧氏距离会更直观。

---

## 替代方案与适用边界

不是所有场景都要追求“所有度量完全一致”。更现实的目标是：选一个与任务定义一致、与训练过程一致、与上线系统一致的度量。

| 场景判断 | 更合适的选择 | 原因 |
|---|---|---|
| 只关心方向，不关心模长 | cosine | 去掉模长干扰 |
| 模长包含置信度或强度 | dot | 保留范数信号 |
| 需要明确几何半径 | L2 / squared L2 | 距离解释直观 |
| 知识图谱链接预测 | 关系特定评分 | 相似不等于可连接 |

再映射到常见任务：

| 任务类型 | 常见度量 | 备注 |
|---|---|---|
| 语义检索 | cosine 或归一化后的 dot | 句向量常这样做 |
| 向量数据库召回 | dot / cosine / squared L2 | 取决于训练方式 |
| 推荐系统 MIPS | dot | 常保留模长信息 |
| 知识图谱补全 | TransE / RotatE / ComplEx 等 | 不能只靠实体近邻 |

什么时候不能只靠相似度？标准答案是：只要关系本身有方向、类型、组合规则，就不能只看实体近不近。`born_in`、`works_at`、`acquired`、`parent_of` 都属于这一类。实体空间最近邻可以帮你缩小候选集，但不能直接当最终判决器。

---

## 参考资料

1. [FAISS: MetricType and distances](https://github.com/facebookresearch/faiss/wiki/MetricType-and-distances)  
用于说明 `METRIC_L2` 返回平方欧氏距离，以及归一化后 cosine 可映射到 inner product / L2。

2. [Sentence Transformers: similarity](https://www.sbert.net/docs/package_reference/util/similarity.html)  
用于说明 `cos_sim`、`dot_score` 等相似度函数的定义和实际接口形式。

3. [Sentence Transformers: Semantic Textual Similarity](https://www.sbert.net/docs/sentence_transformer/usage/semantic_textual_similarity.html)  
用于说明语义相似度任务里默认常用 cosine，并展示实践层面的相似度切换方式。

4. [TransE: Translating Embeddings for Modeling Multi-relational Data](https://papers.nips.cc/paper/5071-translating-embeddings-for-modeling-multi-relational-data)  
用于说明知识图谱里关系可被建模为向量平移，而不是单纯实体最近邻。

5. [RotatE: Knowledge Graph Embedding by Relational Rotation in Complex Space](https://openreview.net/forum?id=HkgEQnRqYQ)  
用于说明关系也可以被建模为复空间旋转，从而处理对称、反对称、组合等模式。

6. [PyKEEN: Models](https://pykeen.readthedocs.io/en/stable/reference/models.html)  
用于说明工程中可用的 KG 嵌入模型族，便于把“召回 + 关系精排”落到现成框架。
