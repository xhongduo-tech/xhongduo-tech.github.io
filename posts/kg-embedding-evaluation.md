## 核心结论

知识图谱嵌入评估，核心不是判断一条三元组“真或假”，而是做**链接预测**。链接预测可以白话理解为：给定半条关系，让模型把所有可能答案排成一队，真实答案排得越靠前，模型越好。标准查询形式有两种：

- 补尾实体：$(h, r, ?)$
- 补头实体：$(?, r, t)$

评估时，模型会对全体实体打分，再根据真实实体的排名 $r_i$ 计算指标。最常见的指标是：

$$
MR = \frac{1}{|Q|}\sum_i r_i
$$

$$
MRR = \frac{1}{|Q|}\sum_i \frac{1}{r_i}
$$

$$
Hits@K = \frac{1}{|Q|}\sum_i \mathbf{1}[r_i \le K]
$$

其中，$Q$ 是查询集合，$r_i$ 是第 $i$ 个查询的真实答案名次。

有两个结论必须先记住。

第一，**MRR 比 MR 和 Hits@10 更适合作为主指标**。原因是 MRR 对头部排序更敏感，真实答案从第 2 名提升到第 1 名，MRR 会明显上升；而 Hits@10 往往看不出这种差别。

第二，**正式评估应优先使用 filtered setting，而不是 raw setting**。filtered 可以白话理解为：如果某个候选实体与当前查询也能组成已知真三元组，就把它从“错误候选”里移除，避免把“另一个真答案”错当负例。

| 指标 | 定义直觉 | 越大/越小更好 | 主要特点 |
|---|---|---:|---|
| MR | 平均名次 | 越小越好 | 易受极端坏样本影响 |
| MRR | 平均倒数名次 | 越大越好 | 对前几名最敏感 |
| Hits@1 | 第一名命中率 | 越大越好 | 最严格 |
| Hits@3 | 前三名命中率 | 越大越好 | 兼顾头部 |
| Hits@10 | 前十名命中率 | 越大越好 | 常用但较粗糙 |

---

## 问题定义与边界

知识图谱通常由三元组 $(h, r, t)$ 构成，分别表示头实体、关系、尾实体。实体可以理解为“图里的节点”，比如 `Paris`、`France`、`Einstein`；关系可以理解为“节点之间的边”，比如 `capital_of`、`born_in`。

知识图谱嵌入模型会把实体和关系映射到向量空间。向量空间可以白话理解为：用一组数字表示对象，让“语义接近”的对象在几何上也更接近。模型训练完成后，会定义一个打分函数 $f(h,r,t)$，分数越高或越低取决于具体模型，但本质都是为了区分更可能成立和不太可能成立的三元组。

评估对象不是自由生成答案，而是**实体排序**。例如：

- `(Paris, capital_of, ?)`：要在所有实体里找到最合理的尾实体
- `(?, born_in, Ulm)`：要在所有实体里找到最合理的头实体

这就是为什么知识图谱嵌入评估天然是 ranking，而不是生成式问答。

评估边界还包括两个关键点。

第一，通常要同时评估两个方向：

- `head prediction`：$(?, r, t)$
- `tail prediction`：$(h, r, ?)$

如果只报其中一个方向，结果会失真。某些关系天然更容易补头，另一些更容易补尾。例如 `(country, has_capital, ?)` 通常比 `(?, capital_of, city)` 更稳定，因为一个国家往往只有一个首都，而一个城市可能对应多种上下文关系。

第二，必须明确 **raw setting** 与 **filtered setting**。

| 设置 | 候选集 | 是否移除已知真三元组 | 典型用途 |
|---|---|---|---|
| raw | 全体实体 | 否 | 早期论文或对照实验 |
| filtered | 全体实体减去已知真候选 | 是 | 主流论文与工程评估 |

为什么 filtered 更合理？因为知识图谱本身可能是一对多、多对一、多对多关系。比如 `(Beijing, located_in, ?)` 可能既能连到 `China`，也能连到 `East Asia`，二者都是真。如果测试时只把其中一个当作标准答案，另一个却被当作负例，模型其实被冤枉了。

---

## 核心机制与推导

评估流程可以抽象成一条固定链路：

查询输入 → 对所有实体打分 → 按分数排序 → 找真实答案名次 → 聚合指标

设测试查询集合为 $Q=\{q_1,\dots,q_n\}$。对每个查询 $q_i$，模型都会产生一个真实答案名次 $r_i$。所有指标都只依赖这组名次。

先看一个玩具例子。

假设实体集合只有 `A, B, C, X, Y, Z`。有两条测试查询：

- `q1 = (h1, r1, ?)`，真实答案是 `B`
- `q2 = (h2, r2, ?)`，真实答案是 `Y`

对于 `q1`，模型排序结果前两名是：`C > B`。问题在于 `(h1, r1, C)` 其实也是知识图谱中的已知真三元组。

那么：

- 在 raw setting 下，`B` 排第 2，故 `r_1^{raw}=2`
- 在 filtered setting 下，`C` 会先被移除，`B` 变成第 1，故 `r_1^{filt}=1`

对于 `q2`，假设前两名都不是真答案，`Y` 排第 3，且前面没有其他已知真三元组。那么：

- `r_2^{raw}=3`
- `r_2^{filt}=3`

于是可得：

$$
MR_{raw} = \frac{2+3}{2} = 2.5
$$

$$
MR_{filt} = \frac{1+3}{2} = 2.0
$$

$$
MRR_{raw} = \frac{1/2+1/3}{2} \approx 0.417
$$

$$
MRR_{filt} = \frac{1+1/3}{2} \approx 0.667
$$

这个例子说明：filtered 的本质不是“给模型放水”，而是让评估目标与任务定义一致。任务要评估的是“是否把真答案排前面”，不是“是否把某个指定答案排在所有其他真答案之前”。

再看为什么 MRR 对头部排名更敏感。若两个模型在 100 个查询上表现相同，只差 1 个查询：

- 模型 A：真实答案排第 1，贡献 $1/1=1$
- 模型 B：真实答案排第 5，贡献 $1/5=0.2$

MRR 差值是 $0.8/100=0.008$，已经可见。若比较 Hits@10，这两个模型都记为命中，差异完全消失。这就是为什么 MRR 更适合比较“前几名是否更准”。

真实工程例子更直观。假设你在 `FB15k-237` 上训练 `TransE`、`ComplEx`、`RotatE`。你真正关心的问题不是“模型能否给某个三元组一个高分”，而是“对 `(h,r,?)` 的全库搜索时，正确实体是否排在足够前面”。因此评估时必须：

- 固定同一数据集切分
- 固定相同实体全集
- 固定 filtered 协议
- 同时报 `head`、`tail`、`both`
- 至少报 `MRR` 与 `Hits@1/3/10`

如果这些条件没对齐，分数没有横向可比性。

---

## 代码实现

实现评估时，核心只有四步：生成查询、对所有实体打分、做 filtered 过滤、计算排名并聚合指标。

下面给一个可运行的简化 Python 示例。它不依赖具体嵌入模型，只模拟“候选实体打分后如何计算 raw 与 filtered 指标”。

```python
from math import isclose

def rank_from_scores(scores, true_entity):
    true_score = scores[true_entity]
    higher = sum(1 for e, s in scores.items() if s > true_score)
    return 1 + higher

def filtered_rank_from_scores(scores, true_entity, filtered_entities):
    # filtered_entities: 需要从竞争者中移除的实体集合，不包含 true_entity
    filtered_scores = {
        e: s for e, s in scores.items()
        if e == true_entity or e not in filtered_entities
    }
    return rank_from_scores(filtered_scores, true_entity)

def metrics_from_ranks(ranks, hits_ks=(1, 3, 10)):
    n = len(ranks)
    mr = sum(ranks) / n
    mrr = sum(1.0 / r for r in ranks) / n
    hits = {k: sum(int(r <= k) for r in ranks) / n for k in hits_ks}
    return mr, mrr, hits

# 玩具例子 q1: raw rank = 2, filtered rank = 1
scores_q1 = {
    "C": 0.95,  # 也是已知真答案，filtered 时应移除
    "B": 0.90,  # 真实答案
    "A": 0.20,
    "X": 0.10,
}

raw_rank_q1 = rank_from_scores(scores_q1, "B")
filt_rank_q1 = filtered_rank_from_scores(scores_q1, "B", filtered_entities={"C"})

# 玩具例子 q2: raw rank = 3, filtered rank = 3
scores_q2 = {
    "X": 0.99,
    "Z": 0.95,
    "Y": 0.80,  # 真实答案
    "A": 0.10,
}

raw_rank_q2 = rank_from_scores(scores_q2, "Y")
filt_rank_q2 = filtered_rank_from_scores(scores_q2, "Y", filtered_entities=set())

assert raw_rank_q1 == 2
assert filt_rank_q1 == 1
assert raw_rank_q2 == 3
assert filt_rank_q2 == 3

mr_raw, mrr_raw, hits_raw = metrics_from_ranks([raw_rank_q1, raw_rank_q2])
mr_filt, mrr_filt, hits_filt = metrics_from_ranks([filt_rank_q1, filt_rank_q2])

assert isclose(mr_raw, 2.5)
assert isclose(mr_filt, 2.0)
assert isclose(mrr_raw, (1/2 + 1/3) / 2)
assert isclose(mrr_filt, (1 + 1/3) / 2)
assert isclose(hits_raw[1], 0.0)
assert isclose(hits_filt[1], 0.5)

print("raw:", mr_raw, mrr_raw, hits_raw)
print("filtered:", mr_filt, mrr_filt, hits_filt)
```

这段代码里最关键的一句是：

$$
rank = 1 + \#\{e \mid score(e) > score(true)\}
$$

也就是“真实答案前面有多少个分数更高的候选，它的名次就是多少加一”。

真实工程中，一般不会逐条用 Python 字典排序，而是把所有实体嵌入拼成矩阵，一次算完整个候选集分数，再做向量化排名。但逻辑完全一样。

| 代码模块 | 职责 | 易错点 |
|---|---|---|
| scoring | 对所有候选实体打分 | 不同模型分数方向可能相反 |
| filtering | 去掉其他已知真三元组 | 容易误删真实答案本身 |
| ranking | 计算真实实体名次 | 分数并列时规则要固定 |
| aggregation | 汇总 MR/MRR/Hits@K | 容易漏掉 head/tail 分开统计 |

如果要写成标准评估循环，伪代码通常是这样：

```python
for (h, r, t) in test_triples:
    # tail prediction
    scores_tail = score_all_tails(h, r)
    rank_tail = compute_rank(scores_tail, true=t, filtered=True)

    # head prediction
    scores_head = score_all_heads(r, t)
    rank_head = compute_rank(scores_head, true=h, filtered=True)

    update_metrics(rank_tail)
    update_metrics(rank_head)
```

这里还有一个实践点：filtered 时的“已知真三元组”通常要用 `train + valid + test` 的并集来构建查表结构，因为这些都代表图中已知为真的事实。

---

## 工程权衡与常见坑

评估协议必须固定，否则结果没有意义。知识图谱嵌入论文里很多“分数差异”，并不只来自模型能力，也可能来自评估协议不同。

最常见的坑有下面几类。

| 常见坑 | 问题本质 | 后果 | 规避方式 |
|---|---|---|---|
| raw / filtered 混报 | 协议不同 | 分数不可比 | 明确标注并优先 filtered |
| 只报单方向 | 只看 head 或 tail | 掩盖关系不对称性 | 至少报告 both，最好分开给 |
| 把另一个真三元组当负例 | 过滤逻辑错误 | 系统性低估性能 | 用全体已知真三元组做过滤 |
| 只看 MR | 长尾主导均值 | 对好模型也不稳定 | 同时报 MRR 与 Hits@K |
| 只看 Hits@10 | 指标太粗 | 看不出顶部细节 | 加报 Hits@1/3 与 MRR |
| 数据集逆关系泄漏 | 测试可走捷径 | 高分不代表强泛化 | 区分旧数据集与去泄漏版本 |

先说 MR。MR 是平均名次，看起来直观，但对极端值非常敏感。假设 99 条查询都排第 1，只有 1 条排第 10000，那么：

$$
MR = \frac{99 \times 1 + 10000}{100} = 100.99
$$

这个结果会让模型看起来很差，但其实大部分查询它都做得很好。MRR 就稳得多，因为倒数会压缩长尾影响。

再说 Hits@10。它能回答“前十名里有没有真实答案”，但它分不清第 1 名和第 10 名，也分不清第 11 名和第 1000 名。对于需要优化搜索结果顶部质量的系统，这个分辨率不够。

真实工程里还有一个大坑是**逆关系泄漏**。逆关系可以白话理解为：一条关系几乎可以由另一条关系反推，例如 `(parent_of)` 与 `(child_of)`。在早期数据集如 `FB15k`、`WN18` 中，这类模式较多，模型可能通过记忆对称或逆关系捷径拿到很高分，却没有真正学到复杂推理能力。因此后续更常用：

- `FB15k-237` 替代 `FB15k`
- `WN18RR` 替代 `WN18`

| 数据集 | 是否更容易有逆关系捷径 | 典型用途 |
|---|---|---|
| FB15k | 是 | 历史对照，不宜单独说明能力 |
| WN18 | 是 | 同上 |
| FB15k-237 | 较少 | 更常用于真实比较 |
| WN18RR | 较少 | 更常用于真实比较 |

因此，如果你看到两个模型一个报在 `FB15k` 上、一个报在 `FB15k-237` 上，即使都叫 “MRR=0.5”，也不能直接比较。这不是同一难度的任务。

---

## 替代方案与适用边界

本文讲的是知识图谱嵌入评估里的标准 ranking 协议，但它不是唯一评估方式。什么时候该用，什么时候不该用，边界必须清楚。

如果任务目标是“给定半条边，在全体实体里找最合理的补全实体”，那么 ranking 指标最自然：

- 适合：链接预测、知识补全、候选重排
- 主指标：MRR、Hits@K、MR

如果任务目标是“判断某条边是真是假”，那就更像二分类问题。二分类可以白话理解为：模型只需要回答“成立”或“不成立”，不要求把所有实体排队。这时更适合：

- Accuracy
- Precision / Recall / F1
- ROC-AUC / PR-AUC

如果任务目标是开放域检索式知识补全，例如先从超大候选库召回，再做重排，除了排名指标，还可能关注：

- Recall@K
- NDCG
- 下游问答效果
- 在线点击或人工审核通过率

| 评估方式 | 核心问题 | 常用指标 | 优点 | 局限 |
|---|---|---|---|---|
| ranking evaluation | 真实实体排第几 | MRR, MR, Hits@K | 贴合链接预测 | 依赖完整候选集 |
| binary classification | 三元组真假 | Accuracy, AUC, F1 | 更直接 | 负例构造影响很大 |
| retrieval-style evaluation | Top-K 检索质量 | Recall@K, NDCG | 适合大规模搜索 | 不完全等同标准 KG benchmark |

所以边界判断可以压缩成一句话：**当任务目标不是实体排序时，本文这些指标不应直接套用。**

还有一个常被忽略的点是负例采样。训练时常用负例采样，因为全体实体太大；但测试时标准 benchmark 通常要求全量排名或近似全量排名。不要把训练阶段的 sampled negative evaluation 误当成最终标准成绩。

---

## 参考资料

1. [Translating Embeddings for Modeling Multi-relational Data](https://papers.nips.cc/paper/5071-translating-embeddings-for-modeling-multi-relational-data.pdf)
2. [PyKEEN: Understanding the Evaluation](https://pykeen.readthedocs.io/en/stable/tutorial/understanding_evaluation.html)
3. [PyKEEN: InverseHarmonicMeanRank / MRR](https://pykeen.readthedocs.io/en/stable/api/pykeen.metrics.ranking.InverseHarmonicMeanRank.html)
4. [Observed versus latent features for knowledge base and text inference](https://aclanthology.org/W15-4007/)
5. [Microsoft 官方 FB15K-237 下载页](https://www.microsoft.com/en-hk/download/details.aspx?id=52312)
