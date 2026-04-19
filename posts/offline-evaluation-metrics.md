## 核心结论

离线评估指标体系，是在不接触线上流量的前提下，用历史日志或标注数据衡量推荐模型的结果质量。它的目标不是单独证明“模型好”，而是同时回答四类问题：排得准不准、覆盖广不广、列表单不单一、结果热不热。

一个推荐模型可以写成一个多目标评估框架：

$$
Eval(M)=f(Accuracy, Coverage, Diversity, Novelty, Position)
$$

其中 `Accuracy` 衡量排序是否接近真实偏好，`Coverage` 衡量系统是否只推荐少数物品，`Diversity` 衡量列表内部是否过于相似，`Novelty` 衡量结果是否过度集中在热门物品，`Position` 衡量命中是否出现在用户最容易看到的位置。

| 指标族 | 典型指标 | 回答的问题 | 新手理解 |
|---|---|---|---|
| 准确性 | AUC、GAUC | 模型是否把相关物品排在不相关物品前面 | 该排前的有没有排前 |
| 位置敏感性 | NDCG@K、MRR@K | 前几位是否命中 | 第 1 名命中比第 10 名命中更值钱 |
| 覆盖性 | Item Coverage、Catalog Coverage | 系统推荐了多少不同物品 | 有没有总盯着少数热门项 |
| 多样性 | ILS、Entropy | 列表或整体分布是否单一 | 推荐列表是不是太像 |
| 新颖性 | Novelty、平均流行度反向量 | 推荐是否过于热门 | 有没有探索长尾内容 |

真实工程里，单指标结论很容易失真。例如电商首页重排模型离线 `AUC` 上升，但 `Catalog Coverage` 和 `Entropy` 下降。这说明模型更会挑热门商品，但更不愿意探索长尾商品。短期点击率可能不差，但长期会让曝光继续集中，冷门商品更难获得机会。

---

## 问题定义与边界

离线评估，是用历史日志、样本标签或人工标注集合，在模型上线前评估推荐结果。它回答的是“排序是否合理”，不是“上线后业务一定提升”。

原因很直接：线上效果还受曝光机制、用户反馈回路、流量分配、探索策略、库存变化、活动干预影响。离线分数高，只能说明模型在当前评估口径下更接近历史数据或标注目标。

| 符号 | 含义 |
|---|---|
| $U$ | 用户集合 |
| $I$ | 物品集合 |
| $K$ | 推荐列表截断长度，例如 top-10 |
| $L_u^K$ | 用户 $u$ 的 top-K 推荐列表 |
| $R(u)$ | 用户 $u$ 的真实相关物品集合 |
| $rel_u(i)$ | 物品 $i$ 对用户 $u$ 的相关性，常见取值为 0 或 1 |
| $s_{ui}$ | 模型给用户 $u$ 和物品 $i$ 的预测分数 |
| $sim(a,b)$ | 物品 $a$ 与物品 $b$ 的相似度 |
| $p(x)$ | 物品、类目或其他统计单元 $x$ 的出现概率 |

新手可以先这样区分：准确率看命中没命中，覆盖率看有没有总推荐少数热门物品，多样性看一个列表是不是太像，新颖性看推荐是不是过于热门。

| 离线评估能回答 | 离线评估不能直接回答 |
|---|---|
| 当前模型在历史样本上排序是否更好 | 上线后 GMV、留存、转化一定提升 |
| top-K 位置命中是否改善 | 用户是否会因为新界面改变行为 |
| 推荐集合是否更集中或更分散 | 探索策略长期是否改变生态 |
| 长尾物品是否获得更多离线推荐机会 | 线上曝光、库存、价格、活动是否共同影响结果 |

---

## 核心机制与推导

排序类指标关注“相关物品是否排在不相关物品前面”。`AUC` 的白话解释是：把相关物品和不相关物品两两比较，看模型是否把相关物品打了更高分。

设 $P_u$ 是用户 $u$ 的正样本集合，$N_u$ 是负样本集合：

$$
AUC_u=\frac{1}{|P_u||N_u|}\sum_{p\in P_u}\sum_{n\in N_u}
\left(\mathbb{1}[s_{up}>s_{un}]+0.5\mathbb{1}[s_{up}=s_{un}]\right)
$$

`GAUC` 是按用户分组后的 AUC 加权平均：

$$
GAUC=\frac{\sum_u w_u AUC_u}{\sum_u w_u}
$$

它要按用户加权，是因为不同用户样本量不同。若直接把所有样本混在一起，大用户或高活跃用户会主导结论，低活跃用户的排序质量会被淹没。常见权重 $w_u$ 可以取用户曝光数、点击样本数或有效正负样本对数量，但实验之间必须固定。

位置敏感指标关注“命中是否在前排”。`NDCG` 的白话解释是：第 1 名命中和第 10 名命中不是一回事，前排命中更值钱。

$$
DCG_u@K=\sum_{i=1}^{K}\frac{2^{rel_u(i)}-1}{\log_2(i+1)}
$$

$$
NDCG_u@K=\frac{DCG_u@K}{IDCG_u@K}
$$

`IDCG` 是理想排序下的最大 `DCG`。`NDCG = DCG / IDCG` 的作用是归一化。不同用户的相关物品数量不同，直接比较 `DCG` 不公平；归一化后，每个用户的分数被压到可比较区间，通常在 0 到 1 之间。

`MRR` 只看第一个相关物品出现得有多早：

$$
MRR@K=\frac{1}{|U|}\sum_{u\in U}\frac{1}{rank_u}
$$

其中 $rank_u$ 是用户 $u$ 的 top-K 列表中第一个相关物品的名次；如果没有命中，通常记为 0。

覆盖类指标关注整个系统输出分布，不是看某个用户列表好不好。`Coverage` 的白话解释是：把所有用户的推荐结果加起来，系统到底推荐了多少不同物品。

$$
Item\ Coverage=\frac{|I_{pred}|}{|I|}
$$

$$
Catalog\ Coverage=\frac{|\bigcup_{u\in U}L_u^K|}{|I|}
$$

多样性指标关注列表内部或整体分布是否单一。`ILS` 是 intra-list similarity，意思是“列表内部相似度”：一个列表里两两物品越像，`ILS` 越高，多样性越低。

$$
ILS_u@K=\frac{2}{K(K-1)}\sum_{1\le i<j\le K}sim(x_{ui},x_{uj})
$$

$$
ILS@K=\frac{1}{|U|}\sum_{u\in U}ILS_u@K
$$

`Entropy` 是分布不确定性的度量。推荐结果越集中在少数物品或类目上，熵越低；分布越分散，熵越高。

$$
Shannon\ Entropy@K=-\sum_x p(x)\log p(x)
$$

| 指标 | 关注对象 | 是否位置敏感 | 是否全局指标 | 典型误区 |
|---|---|---:|---:|---|
| AUC | 正负样本对排序 | 否 | 否 | AUC 高不代表 top-K 一定好 |
| GAUC | 用户级 AUC 加权平均 | 否 | 是 | 权重定义不同不能横比 |
| NDCG@K | top-K 排名质量 | 是 | 可聚合 | 忘记固定 K |
| MRR@K | 第一个命中位置 | 是 | 可聚合 | 只适合重视首个答案的场景 |
| Coverage | 推荐物品覆盖范围 | 否 | 是 | 候选集变化导致误判 |
| ILS | 列表内部相似度 | 否 | 可聚合 | 相似度定义一换，数值就变 |
| Entropy | 推荐分布均匀程度 | 否 | 是 | 按物品算和按类目算不是一回事 |

玩具例子：物品集合为 `{A,B,C,D,E}`，某用户 top-3 是 `[A,B,C]`，真实相关为 `{A,C}`。第 1 位 `A` 命中，第 3 位 `C` 也命中，所以 `MRR@3=1`。如果理想排序是 `[A,C,B]`，当前排序把 `C` 放到第 3 位，`NDCG@3` 会低于 1，但仍然较高。

---

## 代码实现

工程实现要先统一口径：候选集、标签、`K` 值、相似度定义、流行度统计方式必须固定，否则不同实验不可比。推荐流程通常是：

```text
评分 -> 排序 -> 截断 top-K -> 逐用户评估 -> 全局聚合 -> 覆盖/多样性/新颖性统计
```

| 指标 | 输入 | 输出 | 依赖用户级标签 | 依赖物品相似度 |
|---|---|---|---:|---:|
| AUC/GAUC | 用户、物品、分数、标签 | 排序分数 | 是 | 否 |
| NDCG/MRR | top-K 列表、真实相关集合 | 位置敏感分数 | 是 | 否 |
| Coverage | top-K 列表、物品全集 | 覆盖比例 | 否 | 否 |
| ILS | top-K 列表、物品相似度 | 列表内部相似度 | 否 | 是 |
| Entropy | 推荐结果统计 | 分布熵 | 否 | 否 |

下面代码是一个可运行的最小实现，展示 `NDCG@K`、`MRR@K`、`Catalog Coverage`、`ILS@K` 和 `Entropy` 的统一计算方式：

```python
import math
from itertools import combinations
from collections import Counter

def dcg_at_k(items, relevant, k):
    score = 0.0
    for rank, item in enumerate(items[:k], start=1):
        rel = 1 if item in relevant else 0
        score += (2 ** rel - 1) / math.log2(rank + 1)
    return score

def ndcg_at_k(items, relevant, k):
    dcg = dcg_at_k(items, relevant, k)
    ideal_hits = min(len(relevant), k)
    ideal = sum(1 / math.log2(rank + 1) for rank in range(1, ideal_hits + 1))
    return 0.0 if ideal == 0 else dcg / ideal

def mrr_at_k(items, relevant, k):
    for rank, item in enumerate(items[:k], start=1):
        if item in relevant:
            return 1.0 / rank
    return 0.0

def catalog_coverage(recommendations, all_items, k):
    recommended = set()
    for items in recommendations.values():
        recommended.update(items[:k])
    return len(recommended) / len(all_items)

def ils_at_k(items, sim, k):
    topk = items[:k]
    pairs = list(combinations(topk, 2))
    if not pairs:
        return 0.0
    return sum(sim.get(tuple(sorted((a, b))), 0.0) for a, b in pairs) / len(pairs)

def entropy_at_k(recommendations, k):
    xs = []
    for items in recommendations.values():
        xs.extend(items[:k])
    total = len(xs)
    counts = Counter(xs)
    return -sum((c / total) * math.log(c / total) for c in counts.values())

recommendations = {
    "u1": ["A", "B", "C"],
    "u2": ["A", "D", "E"],
}
relevant = {
    "u1": {"A", "C"},
    "u2": {"E"},
}
all_items = {"A", "B", "C", "D", "E"}
sim = {
    ("A", "B"): 0.8, ("A", "C"): 0.2, ("B", "C"): 0.4,
    ("A", "D"): 0.1, ("A", "E"): 0.3, ("D", "E"): 0.7,
}

k = 3
ndcg = sum(ndcg_at_k(recommendations[u], relevant[u], k) for u in recommendations) / len(recommendations)
mrr = sum(mrr_at_k(recommendations[u], relevant[u], k) for u in recommendations) / len(recommendations)
coverage = catalog_coverage(recommendations, all_items, k)
ils = sum(ils_at_k(items, sim, k) for items in recommendations.values()) / len(recommendations)
entropy = entropy_at_k(recommendations, k)

assert round(ndcg, 3) == 0.710
assert round(mrr, 3) == 0.667
assert coverage == 1.0
assert round(ils, 3) == 0.417
assert round(entropy, 3) == 1.561
```

真实工程例子：电商首页重排模型上线前，通常会先固定一批离线曝光日志作为评估集，再对每个用户生成 top-20 商品。逐用户计算 `GAUC/NDCG@20/MRR@20`，全局统计 `Catalog Coverage/Entropy/ILS`。如果 `AUC` 上升但 `Entropy` 下降，说明模型更偏向少数高频商品，需要进一步检查长尾曝光和多样化重排。

---

## 工程权衡与常见坑

不同指标对应不同工程风险，不能把它们当成互相替代的分数。`AUC` 高不代表 top-K 一定好，`Coverage` 高也不代表排序一定准，`ILS` 低也不代表业务一定提升。

| 坑点 | 原因 | 规避方式 |
|---|---|---|
| 只看 AUC | AUC 不强调前排位置 | 同时看 NDCG@K、MRR@K |
| GAUC 权重随意变化 | 用户分组和权重会改变结论 | 固定 $w_u$ 定义 |
| K 值不一致 | top-10 和 top-50 不是同一问题 | 所有实验固定 K |
| 候选集变化 | 1 万候选和 10 万候选难度不同 | 固定召回池或分层报告 |
| ILS 相似度来源变化 | embedding、类目距离、文本相似度口径不同 | 固定相似度版本 |
| Entropy 粒度不清 | 按物品、品牌、类目统计结果不同 | 明确统计单元 |
| Novelty 和 Coverage 混用 | 新颖性看少见程度，覆盖率看范围 | 分开报告，不互相替代 |

新手版坑例：同一个模型，候选集从 1 万个物品变成 10 万个物品，`Coverage` 和 `AUC` 都可能变化。这不一定说明模型变差，可能只是评估难度和候选范围变了。

另一个常见坑是 `ILS` 的相似度定义。若原来用协同过滤 embedding 计算相似度，后来换成类目距离，数值会明显变化。此时不能把新旧 `ILS` 直接横向比较。

工程检查清单可以压缩成一个公式：

$$
ComparableEval = Fixed(K, CandidateSet, Label, Similarity, PopularityWindow)
$$

上线前至少检查：`K` 是否固定，候选集是否固定，标签定义是否固定，相似度版本是否固定，流行度统计窗口是否固定，是否按用户分层查看了冷启动用户、低活跃用户和高活跃用户。

---

## 替代方案与适用边界

不同任务应选择不同指标组合，不存在通用唯一指标。排序任务偏 `AUC/NDCG/MRR`，探索和生态任务更关注 `Coverage/ILS/Entropy/Novelty`。

| 任务场景 | 推荐指标组合 | 适用说明 |
|---|---|---|
| 只关心点击命中 | AUC + GAUC + NDCG@K | 适合点击率预估、商品排序 |
| 关心首个正确结果 | MRR@K + NDCG@K | 适合搜索、问答、单答案任务 |
| 关心首页内容不要太单一 | Coverage + ILS + Entropy | 适合信息流、视频流、首页推荐 |
| 关心冷启动或长尾 | Novelty + Coverage + 长尾曝光占比 | 适合内容生态和新商品扶持 |
| 关心整体业务增长 | 离线指标 + A/B 实验 + 留存指标 | 离线只能作为上线前筛选 |

决策顺序可以很简单：先确定目标，再选指标，再决定是否需要线上验证。若目标是“排序更准”，优先看准确性和位置敏感指标；若目标是“结果更丰富”，优先看覆盖性和多样性；若目标是“业务增长”，离线指标只能作为候选模型筛选，还必须结合线上 A/B、长期留存、用户研究和探索收益。

离线评估的边界是明确的：它能降低上线试错成本，但不能替代线上实验。推荐系统上线后会改变用户看到什么，用户反馈又会反过来改变训练数据，这就是反馈回路。只看历史日志，无法完整模拟这个过程。

---

## 参考资料

官方文档：

1. [scikit-learn roc_auc_score](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.roc_auc_score.html)
2. [scikit-learn ndcg_score](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.ndcg_score.html)
3. [RecBole Metrics](https://recbole.io/docs/v1.2.0/recbole/recbole.evaluator.metrics.html)
4. [RecBole Evaluation Settings](https://www.recbole.io/docs/user_guide/config/evaluation_settings.html)

论文与综述：

5. [Evaluating Recommender Systems](https://link.springer.com/chapter/10.1007/978-1-4899-7637-6_8)
6. [Intra-list similarity and human diversity perceptions of recommendations: the details matter](https://link.springer.com/article/10.1007/s11257-022-09351-w)
7. [Deep Interest Network for Click-Through Rate Prediction](https://dl.acm.org/doi/10.1145/3219819.3219823)

工程实现：

8. [Microsoft Recommenders Evaluation](https://microsoft-recommenders.readthedocs.io/en/latest/evaluation.html)
9. [Evidently Ranking and Recommendations Metrics](https://docs.evidentlyai.com/metrics/explainer_recsys)
10. [TorchMetrics Retrieval Metrics](https://lightning.ai/docs/torchmetrics/stable/retrieval/)

推荐阅读顺序：先看 scikit-learn 的 `AUC` 和 `NDCG` 定义，再看 RecBole 的评估配置，最后读多样性和推荐系统评估综述。
