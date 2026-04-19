## 核心结论

多目标排序融合，是把点击、转化、时长、互动等多个目标信号，按业务偏好合成一个最终排序分数，再据此输出统一榜单。

它的本质不是“同时优化很多目标”，而是先把多个目标变成可比较的分数，再用一个明确规则生成最终排序分数 $F_i$。如果没有这个统一分数，系统只能得到多张互相冲突的榜单：点击模型想把 A 放前面，转化模型想把 B 放前面，互动模型又想把 C 放前面。

通用写法是：

$$
F_i = \text{Fuse}(\tilde s_{i1}, \tilde s_{i2}, \ldots, \tilde s_{im}; w_1,w_2,\ldots,w_m)
$$

其中第 $i$ 个候选 item 有 $m$ 个目标分数，$\tilde s_{ij}$ 表示归一化后的第 $j$ 个目标分数，$w_j$ 表示该目标的业务权重。

同一批候选内容，如果同时有点击、转化、停留时长和互动率四类预测分，首页可以偏点击和停留，交易页可以偏转化，社区页可以偏互动。最终无论偏好如何变化，排序系统都必须输出一个统一的最终顺序。

| 方法 | 什么时候优先用 | 核心判断 |
|---|---|---|
| 加权求和 | 目标明确、分数可信、业务需要可解释 | 按业务权重合成总分 |
| 乘法融合 | 不希望某个目标太差 | 短板会明显拉低总分 |
| RRF | 多路召回或多排序器分值不可比 | 只看名次，不看原始分值 |
| LTR | 数据充足、目标复杂、人工权重难调 | 让模型学习融合函数 |

实际工程顺序通常是：先做分数校准，再做加权融合或排序融合，最后才考虑 Learning to Rank。LTR 是“排序学习”，意思是直接训练一个模型来预测 item 在某个上下文里的最终排序分。

---

## 问题定义与边界

目标，是系统想优化的行为指标，例如 CTR、CVR、停留时长、互动率。CTR 是点击率，表示曝光后被点击的概率；CVR 是转化率，表示点击或曝光后产生购买、下单、注册等转化行为的概率。融合，是把多个目标分数转换成一个可排序总分。

候选 item，是等待排序的内容、商品、视频、帖子或广告。目标数 $m$ 表示系统同时考虑多少个目标。归一化分数 $\tilde s_{ij}$ 是把不同模型输出调整到可比较区间后的分数。权重 $w_j$ 是业务对第 $j$ 个目标的偏好强度。

| 维度 | 定义 |
|---|---|
| 输入信号 | 多个模型或规则输出的目标分数，如 CTR、CVR、时长分、互动分 |
| 输出结果 | 每个候选 item 的最终分数 $F_i$，以及按 $F_i$ 排序后的列表 |
| 适用场景 | 推荐、搜索、广告、信息流、商品排序、多路召回融合 |
| 不适用场景 | 单目标模型训练、召回生成、内容安全审核、库存约束等完整业务策略替代 |

真实工程例子：用户进入电商首页时，系统可能同时拿到三个模型输出：点击分、转化分、停留分。首页更看重用户是否愿意点开和继续浏览，所以点击和停留权重较高。购物车页附近的推荐更接近交易环节，所以转化权重更高。同一件商品，在两个页面里的最终排序可能不同，因为业务目标不同。

需要明确边界：融合解决的是“已有多个打分结果如何合并”。它不负责训练单个 CTR 模型，也不替代价格过滤、库存过滤、内容安全、去重、多样性控制等业务策略。这些策略通常在召回、粗排、精排、重排等不同阶段共同工作。

---

## 核心机制与推导

最简单的方法是加权求和：

$$
F_i=\sum_{j=1}^{m} w_j \tilde s_{ij},\quad \sum_j w_j=1
$$

它直接、稳定、可解释。$w_j=0.5$ 表示该目标占最终分数的一半影响力。缺点是允许“偏科”：某个目标很高时，可能掩盖另一个目标很差的问题。

乘法融合的公式是：

$$
F_i=\prod_{j=1}^{m}(\tilde s_{ij}+\epsilon)^{w_j}
$$

其中 $\epsilon$ 是一个很小的正数，用来避免分数为 0 时整体结果直接变成 0。取对数后：

$$
\log F_i=\sum_j w_j\log(\tilde s_{ij}+\epsilon)
$$

这个形式说明，乘法融合本质上仍然是加权，但加权发生在对数空间。它会更强地惩罚短板，适合“点击、转化、质量都不能太差”的场景。

RRF 是 Reciprocal Rank Fusion，中文可理解为“倒数排名融合”。它不使用原始分值，只使用每个排序器给出的名次：

$$
F_i=\sum_{j=1}^{m}\frac{1}{k+r_{ij}}
$$

其中 $r_{ij}$ 是第 $j$ 个排序器给 item $i$ 的名次，$k$ 是平滑常数。$k$ 越大，头部名次差异被压得越平。

LTR 的统一写法是：

$$
F_i=f_\theta(q,x_i)
$$

$q$ 是请求上下文，例如用户、查询词、页面位置；$x_i$ 是候选 item 的特征；$f_\theta$ 是带参数的排序模型。它不再由人工写死融合公式，而是从样本中学习如何组合特征和目标。

玩具例子：两个 item，三个目标为点击、转化、时长，权重 $w=(0.5,0.3,0.2)$。

| item | 点击 | 转化 | 时长 | 加权和 |
|---|---:|---:|---:|---:|
| A | 0.9 | 0.2 | 0.4 | 0.57 |
| B | 0.6 | 0.8 | 0.3 | 0.63 |

加权和下，B 排在 A 前。若不带权重地看乘法，A 为 $0.9 \times 0.2 \times 0.4=0.072$，B 为 $0.6 \times 0.8 \times 0.3=0.144$，B 仍更靠前，而且 A 的低转化被明显惩罚。

如果使用 RRF，假设 A 在三个目标排序中的名次是 $(1,10,4)$，B 是 $(3,2,6)$，取 $k=60$：

| item | RRF 分数 |
|---|---:|
| A | $1/61+1/70+1/64 \approx 0.0463$ |
| B | $1/63+1/62+1/66 \approx 0.0472$ |

RRF 只关心名次，不关心“第 1 名比第 2 名高多少分”。

| 方法 | 输入 | 输出 | 优点 | 缺点 |
|---|---|---|---|---|
| 加权和 | 归一化分数、权重 | 总分 | 简单、可解释、易上线 | 依赖校准，允许偏科 |
| 乘法 | 归一化分数、权重 | 总分 | 强调多目标均衡 | 容易被低分目标拉垮 |
| RRF | 多个排序名次 | 总分 | 适合异构结果融合 | 丢失原始分值 |
| LTR | 特征、标签、上下文 | 模型预测分 | 表达能力强 | 需要训练数据和评估体系 |

---

## 代码实现

实现顺序通常是：先归一化，再融合，最后排序。生产环境还要把权重、融合模式、实验开关放进配置系统，便于 AB 测试和快速回滚。

```python
from math import prod

raw_scores = {
    "A": {"ctr": 0.90, "cvr": 0.20, "duration": 40},
    "B": {"ctr": 0.60, "cvr": 0.80, "duration": 30},
    "C": {"ctr": 0.30, "cvr": 0.50, "duration": 90},
}

weights = {"ctr": 0.5, "cvr": 0.3, "duration": 0.2}


def normalize(raw):
    keys = next(iter(raw.values())).keys()
    mins = {k: min(item[k] for item in raw.values()) for k in keys}
    maxs = {k: max(item[k] for item in raw.values()) for k in keys}

    result = {}
    for item_id, scores in raw.items():
        result[item_id] = {}
        for k, v in scores.items():
            if maxs[k] == mins[k]:
                result[item_id][k] = 0.0
            else:
                result[item_id][k] = (v - mins[k]) / (maxs[k] - mins[k])
    return result


def weighted_sum(scores, weights):
    return sum(scores[k] * weights[k] for k in weights)


def multiplicative(scores, weights, eps=1e-6):
    return prod((scores[k] + eps) ** weights[k] for k in weights)


def ranks_by_target(norm_scores):
    targets = next(iter(norm_scores.values())).keys()
    ranks = {item_id: {} for item_id in norm_scores}
    for target in targets:
        ordered = sorted(norm_scores, key=lambda x: norm_scores[x][target], reverse=True)
        for rank, item_id in enumerate(ordered, start=1):
            ranks[item_id][target] = rank
    return ranks


def rrf_score(item_ranks, k=60):
    return sum(1 / (k + rank) for rank in item_ranks.values())


def rank_items(norm_scores, mode):
    if mode == "weighted_sum":
        score_fn = lambda item: weighted_sum(norm_scores[item], weights)
    elif mode == "multiplicative":
        score_fn = lambda item: multiplicative(norm_scores[item], weights)
    elif mode == "rrf":
        ranks = ranks_by_target(norm_scores)
        score_fn = lambda item: rrf_score(ranks[item])
    else:
        raise ValueError(f"unknown mode: {mode}")

    return sorted(
        [(item, score_fn(item)) for item in norm_scores],
        key=lambda x: x[1],
        reverse=True,
    )


scores = normalize(raw_scores)

sum_rank = rank_items(scores, "weighted_sum")
mul_rank = rank_items(scores, "multiplicative")
rrf_rank = rank_items(scores, "rrf")

print("weighted_sum:", sum_rank)
print("multiplicative:", mul_rank)
print("rrf:", rrf_rank)

assert set(scores.keys()) == {"A", "B", "C"}
assert sum(weights.values()) == 1.0
assert len(sum_rank) == 3
assert sum_rank[0][1] >= sum_rank[-1][1]
```

主流程可以抽象成：

```python
scores = normalize(raw_scores)
final_score = fuse(scores, weights, mode="weighted_sum")
ranked_items = sorted(items, key=lambda x: x.final_score, reverse=True)
```

离线评估负责验证权重是否合理，例如看 NDCG、转化率估计、分桶表现。线上服务负责稳定执行排序逻辑，并保留实验开关、监控指标和回滚能力。

---

## 工程权衡与常见坑

工程问题通常不在公式本身，而在分数尺度、标签偏差、指标定义和线上反馈。

最典型的坑是分数没校准就直接相加。一个模型输出 $0$ 到 $1$ 的概率，另一个模型输出 $0$ 到 $100$ 的评分，如果直接相加，后者会压倒前者。即使 CTR 从 0.1 提升到 0.9，也抵不过另一个评分多 10 分，排序会失真。

| 问题表现 | 根因 | 修复办法 | 适用策略 |
|---|---|---|---|
| 某个目标完全主导排序 | 分数量纲不一致 | 归一化、校准、截断异常值 | 加权和、乘法 |
| 线上 KPI 下降但离线指标上升 | 离线标签不等于业务目标 | 增加 AB 测试和业务指标监控 | 全部策略 |
| 乘法结果大量接近 0 | 某些目标分过低或缺失 | 加 $\epsilon$、缺失值填充、换加权和 | 乘法 |
| RRF 后强模型优势消失 | 只保留名次，丢掉分值差距 | 对可信模型单独加权或用分值融合 | RRF |
| LTR 线上不稳定 | 样本少、特征漂移、标签噪声 | 增加数据、正则化、分桶监控 | LTR |
| 新内容永远排不上来 | 缺少探索流量 | 加探索机制、冷启动规则 | 全部策略 |

上线前检查清单：

| 检查项 | 说明 |
|---|---|
| 是否做了分数校准 | 不同目标分数必须可比较 |
| 是否有权重配置 | 权重不能硬编码在不可控位置 |
| 是否做了 AB 测试 | 最终权重应由线上效果确认 |
| 是否监控分桶指标 | 新老用户、品类、价格段、内容类型可能表现不同 |
| 是否支持回滚 | 排序变更影响面大，必须能快速切回 |
| 是否处理缺失分 | 新 item、冷启动 item、模型超时都可能没有完整分数 |

---

## 替代方案与适用边界

如果只是异构结果合并，优先考虑 RRF。异构结果，是指不同来源的候选列表分值体系不一样，例如文本检索、向量检索、规则召回、热门召回。它们的分数往往不可直接比较。

如果业务目标明确且分数经过校准，优先考虑加权和。电商精排里，CTR、CVR、客单价、停留等目标相对可量化，业务方也能解释“为什么转化权重要高”，加权和更容易协作和排查。

如果短板不能太差，考虑乘法融合。例如广告排序中，点击率高但质量分极低的广告不应排太前；内容推荐中，点击率高但负反馈风险高的内容也应被压制。

如果数据足够、目标复杂、上下文影响大，再考虑 LTR。LTR 可以学习非线性关系，例如“新用户更偏点击，老用户更偏转化”“低价商品点击高但利润低”“某些类目停留时长天然更长”。

| 方法 | 是否依赖原始分值 | 是否需要训练 | 是否可解释 | 适用边界 |
|---|---|---|---|---|
| 加权和 | 是 | 否 | 强 | 分数已校准，业务权重清楚 |
| 乘法融合 | 是 | 否 | 中 | 多目标都不能太差 |
| RRF | 否 | 否 | 中 | 多路召回、混合搜索、分值不可比 |
| LTR | 是 | 是 | 弱到中 | 数据充足、特征稳定、目标复杂 |

真实工程里这些方案不是互斥的。常见做法是：加权融合做基线，LTR 作为增强版本，RRF 作为检索召回融合补充。搜索场景中，文本召回和向量召回来源不同，直接拼接分值风险很高，RRF 通常更稳。电商排序中，多目标可比较且业务偏好明确时，加权和更直观，也更容易通过 AB 调权。

---

## 参考资料

基础理论：

1. [Reciprocal rank fusion outperforms condorcet and individual rank learning methods](https://research.google/pubs/reciprocal-rank-fusion-outperforms-condorcet-and-individual-rank-learning-methods/)：RRF 的原始论文，适合先理解排序融合为什么可以只看名次。
2. [Learning to Rank with Multiple Objective Functions](https://www.microsoft.com/en-us/research/publication/learning-to-rank-with-multiple-objective-functions/)：讨论多目标排序学习，说明多个目标如何合并为可学习的排序目标。

排序学习：

3. [XGBoost Learning to Rank](https://xgboost.readthedocs.io/en/stable/tutorials/learning_to_rank.html)：XGBoost 官方排序学习文档，适合理解 pairwise、NDCG 和 LambdaMART 的工程用法。
4. [TensorFlow Ranking Overview](https://www.tensorflow.org/ranking/overview)：TensorFlow Ranking 官方概览，适合了解可扩展 LTR 训练流程、损失函数和排序 pipeline。

工程实践：

5. [Azure AI Search: Relevance scoring in hybrid search using Reciprocal Rank Fusion](https://learn.microsoft.com/en-us/azure/search/hybrid-search-ranking)：Azure AI Search 官方文档，给出 RRF 在混合搜索中的工程化定义和公式。

建议阅读顺序：先看 RRF 论文理解排序融合，再看 XGBoost Ranking 或 TensorFlow Ranking 理解训练，最后看 Azure Search 文档理解工程化 RRF。
