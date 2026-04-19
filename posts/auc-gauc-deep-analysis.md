## 核心结论

AUC 衡量的是全局两两排序正确率：把所有样本混在一起，随机抽一个正样本和一个负样本，模型给正样本的分数高于负样本的概率。

GAUC 衡量的是用户内排序正确率：先在每个用户自己的样本里计算 AUC，再按用户权重做加权平均。它更接近推荐系统里的真实问题：同一个用户看到一批候选物品，模型能否把他更可能点击、购买、停留的物品排到前面。

| 指标 | 一句话定义 |
|---|---|
| AUC | 所有样本混在一起后，正样本排在负样本前面的概率 |
| GAUC | 每个用户内部先算 AUC，再按用户权重汇总 |
| 适用场景 | AUC 适合看全局区分能力；GAUC 更适合看推荐、广告里的个性化排序能力 |

一个关键差异是：AUC 会比较不同用户之间的样本分数，GAUC 通常只比较同一个用户内部的样本分数。

玩具例子：用户 A 的分数整体偏高，点击样本 0.90，未点击样本 0.80；用户 B 的分数整体偏低，点击样本 0.40，未点击样本 0.10。把两个人混起来看，A 的未点击样本 0.80 可能排在 B 的点击样本 0.40 前面，这会影响全局 AUC。但在真实推荐里，用户 A 的未点击物品通常不会和用户 B 的点击物品放在同一个候选列表里竞争。GAUC 避开了这种跨用户比较，更关注每个用户自己的排序是否正确。

真实工程例子：信息流 CTR 预估里，不同用户的活跃度、历史点击率、曝光量差异很大。重度用户可能每天有几百次曝光，轻度用户可能一周只有几次曝光。全局 AUC 很容易被样本量大的用户主导；GAUC 则先评估每个用户内部排序，再通过权重决定每类用户对总指标的影响。

---

## 问题定义与边界

本文讨论的评估对象是用户-物品打分排序。一个样本通常是 `(user_id, item_id, label, score)`：`label` 表示是否点击、购买或转化，`score` 是模型预测分数。这里不是分类准确率问题，也不是回归误差问题。排序评估关心的是分数的相对大小，而不是分数是否等于真实概率。

术语首次定义如下：正样本是 label 为 1 的样本，表示用户发生了目标行为；负样本是 label 为 0 的样本，表示用户没有发生目标行为；候选集是某个用户在一次推荐或一段统计窗口内被模型排序的一批物品。

| 符号 | 含义 |
|---|---|
| `U` | 用户集合 |
| `u` | 某一个用户 |
| `D_u` | 用户 `u` 的全部评估样本 |
| `P_u` | 用户 `u` 的正样本集合 |
| `N_u` | 用户 `u` 的负样本集合 |
| `s(x)` | 模型对样本 `x` 输出的分数 |
| `w_u` | 用户 `u` 在 GAUC 中的权重 |
| `U^*` | 可用于计算用户级 AUC 的用户集合 |

AUC 和 GAUC 都依赖正负样本同时存在。对某个用户来说，如果只有正样本，没有负样本，就无法判断“正样本是否排在负样本前面”；如果只有负样本，没有正样本，也同样无法计算用户内 AUC。因此 GAUC 通常只保留同时有正负样本的用户。

| 用户样本情况 | 是否纳入用户级 AUC | 原因 |
|---|---:|---|
| 同时有正样本和负样本 | 是 | 可以构造正负样本对 |
| 全正样本 | 否 | 没有负样本可比较 |
| 全负样本 | 否 | 没有正样本可比较 |
| 样本数为 0 或 1 | 否 | 无法形成有效正负样本对 |
| 权重为空或权重为 0 | 通常否 | 对加权平均没有贡献，且可能引入实现歧义 |

新手容易误解的一点是：两个用户的分数不能总是直接混着比。用户 A 的整体分数高，不代表模型对用户 A 的排序更好。可能只是用户 A 本来点击率高，模型分数整体抬高；用户 B 本来点击率低，模型分数整体偏低。推荐系统真正要解决的问题通常是：在用户 A 自己的候选物品里，哪些更应该排前面；在用户 B 自己的候选物品里，哪些更应该排前面。

---

## 核心机制与推导

全局 AUC 的 pairwise 解释是：随机抽一个正样本 $x^+$ 和一个负样本 $x^-$，如果 $s(x^+) > s(x^-)$，这对样本排序正确；如果分数相等，通常按半个正确处理；如果 $s(x^+) < s(x^-)$，排序错误。

对用户 `u`，用户内 AUC 定义为：

$$
AUC_u = Pr(s(x^+) > s(x^-)) + \frac{1}{2}Pr(s(x^+) = s(x^-))
$$

其中 $x^+ \in P_u$，$x^- \in N_u$。白话解释：只在同一个用户的样本里，统计正样本赢过负样本的比例；平分时算半次赢。

GAUC 的定义是：

$$
GAUC = \frac{\sum_{u \in U^*} w_u \cdot AUC_u}{\sum_{u \in U^*} w_u}
$$

其中 $U^*$ 是同时有正负样本、且权重有效的用户集合。权重 $w_u$ 必须提前固定，可以是曝光数、点击数、有效样本数，也可以是业务自定义权重。不同权重会得到不同 GAUC。

最小数值例子如下：

| 用户 | 样本 | label | score |
|---|---|---:|---:|
| u1 | a | 1 | 0.90 |
| u1 | b | 0 | 0.10 |
| u2 | c | 1 | 0.60 |
| u2 | d | 0 | 0.70 |
| u2 | e | 0 | 0.50 |

先算用户内 AUC。

| 用户 | 正样本 | 负样本 | 比较 | 结果 |
|---|---|---|---|---:|
| u1 | a: 0.90 | b: 0.10 | 0.90 > 0.10 | 1 |
| u2 | c: 0.60 | d: 0.70 | 0.60 < 0.70 | 0 |
| u2 | c: 0.60 | e: 0.50 | 0.60 > 0.50 | 1 |

所以：

$$
AUC_{u1} = 1.00
$$

$$
AUC_{u2} = \frac{0 + 1}{2} = 0.50
$$

如果使用样本数作为权重，$w_{u1}=2$，$w_{u2}=3$，则：

$$
GAUC = \frac{2 \times 1.00 + 3 \times 0.50}{2 + 3} = 0.70
$$

再看全局 AUC。全局有 2 个正样本 `a,c`，3 个负样本 `b,d,e`，共 $2 \times 3 = 6$ 个正负样本对。

| 正样本 | 负样本 | 比较 | 结果 |
|---|---|---|---:|
| a: 0.90 | b: 0.10 | 0.90 > 0.10 | 1 |
| a: 0.90 | d: 0.70 | 0.90 > 0.70 | 1 |
| a: 0.90 | e: 0.50 | 0.90 > 0.50 | 1 |
| c: 0.60 | b: 0.10 | 0.60 > 0.10 | 1 |
| c: 0.60 | d: 0.70 | 0.60 < 0.70 | 0 |
| c: 0.60 | e: 0.50 | 0.60 > 0.50 | 1 |

因此全局 AUC 为：

$$
AUC = \frac{5}{6} \approx 0.833
$$

同一批样本，GAUC 是 0.70，全局 AUC 是 0.833。差异来自跨用户比较：全局 AUC 把 `u1` 的正样本和 `u2` 的负样本、`u2` 的正样本和 `u1` 的负样本都放在一起比较；GAUC 只关心用户内部候选项的排序。

---

## 代码实现

实现 GAUC 的基本流程是：先按 `user_id` 分组，再在每组内计算 AUC，最后做加权平均。工程代码必须处理异常用户，包括全正、全负、样本数不足、权重为空、分数重复。

下面是一个可运行的 Python 实现。它不依赖第三方库，直接按正负样本对计算 AUC，适合教学和小规模验证。

```python
from collections import defaultdict

def binary_auc(labels, scores):
    positives = [s for y, s in zip(labels, scores) if y == 1]
    negatives = [s for y, s in zip(labels, scores) if y == 0]

    if not positives or not negatives:
        return None

    total = 0
    correct = 0.0

    for ps in positives:
        for ns in negatives:
            total += 1
            if ps > ns:
                correct += 1.0
            elif ps == ns:
                correct += 0.5

    return correct / total

def gauc(rows, weight_mode="sample_count"):
    groups = defaultdict(list)
    for row in rows:
        groups[row["user_id"]].append(row)

    weighted_sum = 0.0
    weight_sum = 0.0

    for user_id, user_rows in groups.items():
        labels = [r["label"] for r in user_rows]
        scores = [r["score"] for r in user_rows]
        auc_u = binary_auc(labels, scores)

        if auc_u is None:
            continue

        if weight_mode == "sample_count":
            weight_u = len(user_rows)
        elif weight_mode == "impression_weight":
            weight_u = sum(r.get("weight", 1.0) for r in user_rows)
        else:
            raise ValueError(f"unknown weight_mode: {weight_mode}")

        if weight_u <= 0:
            continue

        weighted_sum += weight_u * auc_u
        weight_sum += weight_u

    if weight_sum == 0:
        return None

    return weighted_sum / weight_sum

rows = [
    {"user_id": "u1", "label": 1, "score": 0.90, "weight": 1.0},
    {"user_id": "u1", "label": 0, "score": 0.10, "weight": 1.0},
    {"user_id": "u2", "label": 1, "score": 0.60, "weight": 1.0},
    {"user_id": "u2", "label": 0, "score": 0.70, "weight": 1.0},
    {"user_id": "u2", "label": 0, "score": 0.50, "weight": 1.0},
    {"user_id": "u3", "label": 1, "score": 0.80, "weight": 1.0},  # 全正用户，被过滤
]

assert binary_auc([1, 0], [0.90, 0.10]) == 1.0
assert binary_auc([1, 0, 0], [0.60, 0.70, 0.50]) == 0.5
assert abs(gauc(rows, "sample_count") - 0.7) < 1e-12
```

核心伪代码可以压缩成：

```python
for user in users:
    if user has both positive and negative samples:
        auc_u = compute_auc(user.labels, user.scores)
        total += weight_u * auc_u
        denom += weight_u

gauc = total / denom
```

实现细节需要固定口径：

| 细节 | 推荐做法 | 原因 |
|---|---|---|
| 权重来源 | 曝光数、有效样本数、点击数三选一并固定 | 权重不同，GAUC 结论可能不同 |
| 过滤规则 | 过滤全正、全负、样本数不足、权重非正用户 | 用户级 AUC 无定义或无贡献 |
| 是否支持 tie | 支持，分数相等按 0.5 处理 | 符合 AUC 的常见概率解释 |
| 采样口径 | 固定负采样比例和采样范围 | 采样变化会改变指标分布 |
| 时间窗口 | 离线评估窗口固定 | 避免训练数据和评估数据泄漏 |

生产环境通常不会用双重循环计算 AUC，因为复杂度是 $O(|P_u||N_u|)$。大规模数据会基于排序和秩统计计算，复杂度通常可降到 $O(n \log n)$。但无论实现如何优化，定义都应与上面的 pairwise 解释一致。

---

## 工程权衡与常见坑

GAUC 的结果强依赖权重定义。按曝光加权时，高曝光用户影响更大；按点击加权时，高点击用户影响更大；按用户等权时，轻度用户和重度用户的影响更接近。没有唯一正确的权重，只有是否匹配业务目标。

真实工程例子：某推荐模型上线前离线评估，重度用户 GAUC 从 0.74 提升到 0.78，轻度用户 GAUC 从 0.70 下降到 0.66。如果权重按曝光数计算，重度用户样本占 80%，总 GAUC 仍会上升。这个结果不能直接说明模型对所有用户都更好，它只说明在当前权重口径下加权平均更好。若业务目标是提升新用户留存，轻度用户下降可能是严重问题。

| 常见坑 | 问题 | 规避方式 |
|---|---|---|
| 全正/全负用户纳入计算 | 用户内没有正负样本对，AUC 无定义 | 统一过滤，并记录过滤用户占比 |
| 负采样后直接横比 | 负样本分布变了，AUC 和 GAUC 可能不可比 | 固定采样策略，变更后重新建立基线 |
| 权重口径不一致 | 今天按曝光，明天按点击，指标变化无法解释 | 在指标文档里写死权重定义 |
| 只报单一 GAUC | 总分掩盖分层退化 | 同时报全局 AUC、GAUC、分桶 AUC |
| 忽略 tie | 分数重复多时结果偏差明显 | 分数相等按 0.5 处理 |
| 只看离线 GAUC | 离线排序好不等于线上收益高 | 结合线上 A/B、CTR、转化率、留存看 |

推荐检查清单：

| 检查项 | 目的 |
|---|---|
| 按用户活跃度分层 | 判断重度、轻度用户是否方向一致 |
| 按冷启动程度分层 | 判断新用户、新物品是否退化 |
| 按曝光量分层 | 避免高曝光用户支配结论 |
| 按场景分层 | 首页、相关推荐、广告位可能分布不同 |
| 同时报过滤率 | 过滤用户过多时 GAUC 代表性下降 |
| 对比全局 AUC | 判断全局区分能力是否同步变化 |
| 对比线上指标 | 验证离线排序提升是否转化为业务收益 |

GAUC 不是越高越一定好。它只是一个离线排序指标，反映的是在指定样本、指定分组、指定权重下的用户内排序能力。模型是否更适合业务，还要看召回、粗排、精排、重排、探索策略和线上反馈链路。

---

## 替代方案与适用边界

AUC 仍然有价值。如果业务目标是全局区分正负样本，例如风控二分类、医学检测二分类、广告候选粗筛，AUC 能提供稳定的全局判别能力视角。GAUC 更适合每个用户内部候选集不同、且最终排序发生在用户内的任务，例如信息流推荐、猜你喜欢、广告 CTR 预估。

当业务关注 top-K 排名时，GAUC 只能作为辅助指标。原因是 GAUC 统计所有正负样本对，不特别关注前几个位置。推荐系统里，用户往往只看到前 5 个或前 10 个结果，排在第 1 位和第 50 位的业务价值差异很大。此时 NDCG、MRR、Recall@K 更直接。

| 指标 | 衡量对象 | 优点 | 局限 | 适用场景 |
|---|---|---|---|---|
| AUC | 全局正负样本排序 | 稳定、直观、可解释为概率 | 混合不同用户分数，可能受用户偏差影响 | 全局二分类、粗筛、整体判别能力评估 |
| GAUC | 用户内正负样本排序后加权平均 | 更贴近个性化推荐排序 | 依赖权重，过滤用户会影响代表性 | 推荐系统、广告 CTR 预估 |
| NDCG | 排名位置上的相关性收益 | 强调靠前位置，支持多级相关性 | 需要定义位置折扣和相关性等级 | 搜索排序、推荐 top-K 评估 |
| MRR | 第一个相关结果的位置 | 简单，关注首个命中 | 只关心第一个相关结果，忽略后续结果 | 问答检索、导航型搜索 |
| Recall@K | top-K 中召回了多少正样本 | 直接对应候选召回能力 | 不关心 top-K 内部顺序 | 召回阶段、候选生成评估 |

CTR 预估中，GAUC 很常用，因为每个用户看到的广告候选不同，目标是把该用户更可能点击的广告排到前面。搜索排序中，NDCG 往往更关键，因为搜索结果有明确位置，用户更关注前几条结果，并且相关性可能不是简单的 0/1，而是“完全相关、部分相关、不相关”等多级标签。

因此，指标选择应从业务问题反推：如果问题是“模型能否全局区分正负样本”，看 AUC；如果问题是“每个用户自己的候选项排序是否更好”，看 GAUC；如果问题是“前 K 个结果是否排得好”，看 NDCG、MRR、Recall@K。GAUC 不能替代所有排序指标，它只是推荐和广告评估里非常重要的一块。

---

## 参考资料

本文中 AUC 的概率解释主要参考 scikit-learn 文档与 Fawcett 的 ROC 论文；GAUC 的用户分组加权思想参考 DIN 论文；工程实现细节参考 RecBole 和 Elliot 的评估实现文档。

1. [scikit-learn: roc_auc_score](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.roc_auc_score.html)
2. [Tom Fawcett: An Introduction to ROC Analysis](https://doi.org/10.1016/j.patrec.2005.10.010)
3. [Deep Interest Network for Click-Through Rate Prediction](https://doi.org/10.1145/3219819.3219823)
4. [RecBole Evaluator Metrics Documentation](https://recbole.io/docs/v0.2.0/recbole/recbole.evaluator.metrics.html)
5. [Elliot GAUC Implementation Documentation](https://elliot.readthedocs.io/en/latest/_modules/elliot/evaluation/metrics/accuracy/AUC/gauc.html)
