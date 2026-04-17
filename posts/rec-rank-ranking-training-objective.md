## 核心结论

排序模型的训练目标，本质上是在回答一个问题：系统到底更关心“单条结果的绝对分数”，还是“结果之间的相对顺序”。

在工程上，主流方法分成三类：

| 方法 | 建模对象 | 直接优化什么 | 常见损失 | 优点 | 局限 |
|---|---|---|---|---|---|
| Pointwise | 单条样本 | 点击率、转化率等单点概率 | LogLoss、MSE | 训练稳定、实现简单、概率可校准 | 不直接对排序负责 |
| Pairwise | 样本对 | 哪个应排前 | BPR、Hinge Loss | 直接惩罚错排对 | 依赖负采样，训练集构造复杂 |
| Listwise | 整个列表 | NDCG、MAP 等排序指标 | LambdaRank、ListMLE、SoftRank | 最接近线上排序目标 | 训练复杂，对 session/slate 数据要求高 |

这里的“概率校准”可以理解为：模型输出的分数尽量接近真实发生概率，比如预测点击率是 0.2，长期看就应接近 20% 真会点击。

结论可以压缩成两句：

1. 如果业务主要关心单条内容会不会被点、会不会成交，Pointwise 往往是最稳的起点。  
2. 如果业务真正关心结果页前几名有没有排对，Pairwise 和 Listwise 更接近最终目标，尤其是 NDCG 这类位置敏感指标。

对初级工程师最重要的判断标准不是“哪种方法更高级”，而是“你的标注形态、数据规模、线上指标、训练成本是否支持这种目标”。很多线上系统并不是三选一，而是混合：先用 Pointwise 打稳基础，再用 Pairwise 或 Listwise 修正错排。

---

## 问题定义与边界

排序任务不是普通二分类。二分类只关心“这一条对不对”，排序还关心“这几条之间谁在前面”。

更严格地说，给定一次请求 $q$，模型需要对候选集合 $\{d_1, d_2, ..., d_n\}$ 输出得分 $\{s_1, s_2, ..., s_n\}$，再按得分从高到低排序。训练目标的差异，在于损失函数如何使用这些得分。

### 1. Pointwise 的边界

Pointwise 把每条文档独立看待。白话讲，就是“把排序问题拆成很多个单样本预测问题”。  
例如每条样本都有标签 $y \in \{0,1\}$，表示点击或未点击，那么模型只需学习：

$$
P(y=1|x)=\sigma(s)
$$

它不直接问“A 是否应该排在 B 前面”，只问“A 会不会被点击”。

这类方法适合：

- 候选召回后的粗排
- CTR/CVR 预估
- 需要概率解释性的场景
- 数据天然是单条曝光日志的场景

### 2. Pairwise 的边界

Pairwise 不看单条，而看两条样本之间的相对关系。白话讲，就是“模型学习谁应该赢过谁”。

例如同一个请求下，文档 A 被点击，文档 B 没被点击，那么训练时构造一对 $(A,B)$，要求：

$$
s_A > s_B
$$

这类方法更像“纠正错排”，尤其适合：

- 精排阶段
- 关注 Top-K 顺序质量
- 标注天然有偏好关系的数据
- 有能力做负采样的系统

### 3. Listwise 的边界

Listwise 把整个结果页当成一个训练对象。白话讲，就是“不再逐条或逐对判断，而是直接评价整页排得好不好”。

这类方法通常面向 NDCG、MAP 一类列表指标。它更贴近线上真实目标，但前提是你手里必须有 session 或 query 级别的数据，也就是知道“一次请求下整组候选和它们的相关性”。

### 4. 不同排序阶段的目标不同

真实工程里，一阶段和二阶段常常不该用同一种目标。原因很简单：阶段职责不同。

| 阶段 | 主要目标 | 典型数据形态 | 常见训练目标 |
|---|---|---|---|
| 召回/粗排 | 高覆盖、低延迟、基础概率估计 | 单条曝光样本多 | Pointwise |
| 精排 | 修正错排、提升前几位质量 | 同请求下候选集合 | Pairwise / Listwise |
| 重排 | 控制多样性、业务规则、位置收益 | 完整 slate | Listwise / 规则融合 |

玩具例子：  
一个搜索系统第一阶段从百万文档里筛出 500 个候选。此时最重要的是不要漏掉好内容，通常先用 Pointwise CTR 模型把“可能被点”的内容筛出来。  
第二阶段只剩 500 个候选，这时系统更关心前 10 名顺序是否合理，于是再用 Pairwise 或 Listwise 拉升 NDCG。

所以，训练目标不是抽象理论选择，而是阶段职责选择。

---

## 核心机制与推导

### 1. Pointwise：把排序拆成概率估计

Pointwise 最常见的是 LogLoss，也叫交叉熵。白话讲，就是“预测概率和真实标签差多远”。

对于单条样本 $(x_i, y_i)$，模型输出得分 $s_i$，经 sigmoid 后得到概率：

$$
\hat{p}_i = \sigma(s_i)=\frac{1}{1+e^{-s_i}}
$$

损失为：

$$
L_{\text{point}} = -\frac{1}{N}\sum_{i=1}^{N} \left[y_i \log \hat{p}_i + (1-y_i)\log(1-\hat{p}_i)\right]
$$

它的优点是明确、稳定、容易优化。缺点也很明确：如果两个样本都预测对了，但顺序错了，它未必会强烈惩罚。

### 2. Pairwise：直接惩罚错排对

Pairwise 的核心是：同一个请求下，若样本 $i$ 比样本 $j$ 更相关，就应满足 $s_i > s_j$。

BPR 的典型形式是：

$$
L_{\text{BPR}} = - \sum_{(u,i,j)} \log \sigma(s_{ui} - s_{uj})
$$

这里 $(u,i,j)$ 表示用户 $u$ 对物品 $i$ 的偏好高于物品 $j$。  
白话解释：如果正样本分数比负样本高很多，损失就小；如果排反了，损失就大。

另一种常见形式是 Hinge Loss：

$$
L_{\text{hinge}} = \sum_{(i,j)} \max(0, m - (s_i - s_j))
$$

其中 $m$ 是 margin，可以理解为“至少要拉开多少分差”。

### 3. Listwise：直接逼近排序指标

Listwise 的思想是：线上最终看的是 NDCG、MAP 等列表指标，训练就尽量直接朝这些指标优化。

NDCG 的核心是位置折损。白话讲，就是“排得越靠前，价值越大；同样的相关文档，排在第 1 位和第 10 位贡献不同”。

DCG 定义为：

$$
DCG@K = \sum_{r=1}^{K} \frac{2^{rel_r}-1}{\log_2(r+1)}
$$

NDCG 为：

$$
NDCG@K = \frac{DCG@K}{IDCG@K}
$$

其中 $IDCG@K$ 是理想排序下的最大 DCG。

LambdaRank 的关键做法是：如果交换两个文档的位置会导致较大的 $\Delta NDCG$，那这对样本的梯度权重就更大。

常见写法可概括为：

$$
\lambda_{ij} \propto \left|\Delta NDCG_{ij}\right| \cdot \sigma(-(s_i-s_j))
$$

白话讲：  
一对样本如果既排错了，又影响了前几名的质量，那么模型就应更用力地修正它。

### 4. 一个统一的玩具例子

假设一次查询只有两个结果 A、B：

- 真实标签：A=1，B=0
- 预测得分：$s_A=0.8,\ s_B=0.3$

#### Pointwise

若把 0.8、0.3 看成点击概率，则：

$$
L_A = -\log(0.8), \quad L_B = -\log(1-0.3)=-\log(0.7)
$$

平均损失约为：

$$
\frac{-\log(0.8)-\log(0.7)}{2} \approx 0.29
$$

如果把它们理解成 logit，再过 sigmoid，结果会略有不同，但结论不变：两条样本分别被独立惩罚。

#### Pairwise

若 margin 取 0.1，则：

$$
L_{\text{hinge}} = \max(0, 0.1-(0.8-0.3)) = 0
$$

因为 A 已明显高于 B，所以这对样本没有错排惩罚。

#### Listwise

只有两个文档时，如果 A 在前，NDCG=1；若交换成 B 在前，则 NDCG 明显下降。  
因此，Listwise 会认为“把这两个结果排反”是一个重要错误，并对这种交换给出较强训练信号。

可以用一个简单流程理解：

```text
同一请求下的候选集合
        |
        |-- Pointwise: 每条单独算损失
        |-- Pairwise: 每两条比较是否排反
        |-- Listwise: 整页一起看指标变化
```

真实工程例子：  
电商搜索里，用户搜“机械键盘”，结果页前两位如果一个是高相关商品、一个是低相关配件，Pointwise 可能只学到“两个都可能被点”；Pairwise 会强调“商品应在配件前”；Listwise 还会进一步强调“前两位排错对整体 NDCG 伤害最大”。

---

## 代码实现

下面用纯 Python 写一个可运行的最小示例，分别展示 Pointwise、Pairwise、Listwise 的核心计算。代码不是工业级训练框架，但足够说明机制。

```python
import math

def sigmoid(x: float) -> float:
    return 1.0 / (1.0 + math.exp(-x))

def pointwise_logloss(logits, labels):
    assert len(logits) == len(labels)
    loss = 0.0
    for s, y in zip(logits, labels):
        p = sigmoid(s)
        p = min(max(p, 1e-12), 1 - 1e-12)
        loss += -(y * math.log(p) + (1 - y) * math.log(1 - p))
    return loss / len(logits)

def bpr_loss(pos_score: float, neg_score: float) -> float:
    diff = pos_score - neg_score
    return -math.log(sigmoid(diff))

def hinge_loss(pos_score: float, neg_score: float, margin: float = 0.1) -> float:
    return max(0.0, margin - (pos_score - neg_score))

def dcg(rels):
    total = 0.0
    for rank, rel in enumerate(rels, start=1):
        total += (2 ** rel - 1) / math.log2(rank + 1)
    return total

def ndcg(rels):
    ideal = sorted(rels, reverse=True)
    best = dcg(ideal)
    if best == 0:
        return 0.0
    return dcg(rels) / best

def swap_delta_ndcg(rels, i, j):
    current = ndcg(rels)
    swapped = rels[:]
    swapped[i], swapped[j] = swapped[j], swapped[i]
    return abs(ndcg(swapped) - current)

def lambdarank_pair_weight(score_i, score_j, delta_ndcg):
    rho = 1.0 / (1.0 + math.exp(score_i - score_j))
    return delta_ndcg * rho

# 玩具例子：A 比 B 更相关
logits = [1.386294, -0.847298]   # sigmoid 后约为 0.8 和 0.3
labels = [1, 0]

pw = pointwise_logloss(logits, labels)
assert 0 < pw < 1

pos_score = 0.8
neg_score = 0.3
assert hinge_loss(pos_score, neg_score, margin=0.1) == 0.0
assert bpr_loss(pos_score, neg_score) > 0.0

rels = [1, 0]
delta = swap_delta_ndcg(rels, 0, 1)
assert delta > 0.0

weight = lambdarank_pair_weight(0.8, 0.3, delta)
assert weight > 0.0

print("pointwise_logloss =", round(pw, 4))
print("bpr_loss =", round(bpr_loss(pos_score, neg_score), 4))
print("delta_ndcg =", round(delta, 4))
print("lambda_weight =", round(weight, 4))
```

这段代码里有四个核心输入：

| 输入 | 含义 |
|---|---|
| `logits` / `scores` | 模型输出得分 |
| `labels` | 单条样本标签，常见是点击/未点击 |
| `pos_score, neg_score` | Pairwise 的正负样本得分 |
| `rels` | 一个 slate 内的相关性标签列表 |

### 一个更接近训练循环的伪代码

```python
for batch in dataloader:
    scores = model(batch.features)

    if objective == "pointwise":
        loss = pointwise_logloss(scores, batch.labels)

    elif objective == "pairwise":
        pos_scores, neg_scores = sample_pairs(scores, batch.labels, batch.query_ids)
        loss = mean(-log(sigmoid(pos_scores - neg_scores)))

    elif objective == "listwise":
        slate_scores, slate_labels = group_by_query(scores, batch.labels, batch.query_ids)
        loss = lambdarank_loss(slate_scores, slate_labels)

    loss.backward()
    optimizer.step()
```

这里最关键的差别不是数学公式，而是数据组织方式：

- Pointwise：按单条样本喂给模型即可。
- Pairwise：必须先构造正负样本对。
- Listwise：必须按 query/session 聚合成 slate。

真实工程例子：  
推荐系统精排训练时，通常先从曝光日志拿到 `query_id, item_id, clicked`。如果做 Pointwise，直接训练即可；如果做 Pairwise，要先在同一 `query_id` 内找点击样本和未点击样本组成训练对；如果做 Listwise，要把同一请求曝光出的所有 item 重新拼成一个列表，再计算列表级损失。

---

## 工程权衡与常见坑

训练目标的选择，真正难点不在“会不会写 loss”，而在“这个 loss 会把系统推向什么方向”。

### 1. Pointwise 常见问题：概率提升，但排序不一定提升

Pointwise 往往能把 AUC、LogLoss 做得很好，但线上 NDCG 不一定跟着涨。原因是它只关心每条样本是否预测准，不关心列表内部的相对位置。

一个常见 A/B 现象是：CTR 预估更准了，但前几位仍有错排，导致用户主观体验没改善。  
解决办法通常不是放弃 Pointwise，而是在精排阶段叠加 Pairwise 或 Listwise 辅助损失，例如：

$$
L = L_{\text{point}} + \alpha L_{\text{pair}}
$$

这里 $\alpha$ 是权重，表示“在保证概率建模稳定的同时，额外惩罚错排”。

### 2. Pairwise 常见问题：负采样决定了你学到什么

负采样可以理解为：从未点击或低相关样本里挑出“输家”来和正样本比较。  
问题在于，如果负样本总是热门商品、强曝光商品，模型学到的只是“这些东西常常输”，却不一定学到真正细粒度的排序规则。

例如一个推荐系统里，冷门内容曝光少，如果负采样几乎全来自热门内容，模型很难学会“如何在多个冷门候选之间排序”。

这会带来两类偏差：

| 坑 | 结果 | 常见应对 |
|---|---|---|
| 只采随机负样本 | 负样本太容易，训练信号弱 | 加难负样本（hard negative） |
| 只采热门负样本 | 学到曝光偏差，不是真偏好 | 按曝光概率加权或分层采样 |
| 正负比失衡 | 梯度不稳定 | 控制采样比例 |
| 跨 query 组对 | 学到无意义比较 | 只在同一 query/session 内组对 |

### 3. Listwise 常见问题：最接近目标，但成本最高

Listwise 很强，但代价很大：

- 需要 query/session 级数据
- 需要按 slate 组织 batch
- 训练速度通常更慢
- 小数据场景更容易过拟合
- 指标近似和梯度设计更复杂

尤其在推荐系统中，一次请求的候选数可能很大，若直接在超长列表上做 Listwise，训练和显存成本都很高。实践里常用 small slate，也就是只截取前若干候选训练。

### 4. 一个真实工程场景

搜索排序系统上线初期，团队通常先做 Pointwise CTR 模型，因为它最容易落地、最容易解释。  
上线后若发现“整体点击率预测更稳，但搜索结果第一页主观质量一般”，下一步往往不是推倒重来，而是：

1. 保留 Pointwise 主干，维持可解释性和概率校准。  
2. 在精排样本上加入 Pairwise BPR，专门修正“点击结果排在未点击结果后面”的问题。  
3. 如果 session 数据足够稳定，再逐步引入 Listwise 或 LambdaRank，重点优化前几位 NDCG。

这比一开始就全量上 Listwise 更现实。

---

## 替代方案与适用边界

现实里很少存在“最优的单一目标”，更多是“在当前约束下可落地的组合”。

### 1. 混合损失通常比单一损失更实用

常见做法是：

$$
L = \alpha L_{\text{point}} + \beta L_{\text{pair}} + \gamma L_{\text{list}}
$$

这不是为了数学好看，而是为了让系统同时满足三个目标：

- 分数有基本概率意义
- 错排对被显式惩罚
- 前几位排序指标得到直接关注

对新手来说，可以把它理解成“三层约束”：

- Pointwise 负责“每条结果别太离谱”
- Pairwise 负责“好结果别排到坏结果后面”
- Listwise 负责“整页前几名尽量最优”

### 2. 数据和算力不够时，先用代理目标

“代理目标”可以理解为：不能直接优化真实指标时，用一个更容易训练的近似目标代替。  
例如 ListMLE、SoftRank、Softmax Cross-Entropy over slate，都是常见的 Listwise 近似方案。

一个简单思路是，对 slate 内分数做 softmax，形成概率分布，再让高相关文档获得更高概率：

$$
P(d_i \mid q) = \frac{e^{s_i}}{\sum_j e^{s_j}}
$$

然后用相关性构造监督信号。这样虽然不是直接对 NDCG 求导，但比纯 Pointwise 更关注列表内部竞争。

### 3. 选择矩阵

| 条件 | 更适合 Pointwise | 更适合 Pairwise | 更适合 Listwise |
|---|---|---|---|
| 数据形态 | 单条曝光日志 | 可构造偏好对 | 有完整 slate/session |
| 线上目标 | CTR/CVR 预估 | 修正错排 | NDCG/MAP/Top-K 质量 |
| 训练复杂度 | 低 | 中 | 高 |
| 可解释性 | 高 | 中 | 相对低 |
| 采样复杂度 | 低 | 高 | 高 |
| 对位置敏感 | 弱 | 中 | 强 |

### 4. 一个分层落地方案

真实工程里，一个常见方案是：

- First-stage：只用 Pointwise CTR 模型，保证高吞吐、稳定和概率可解释。
- Second-stage：在同一请求内做 Pairwise BPR，修正明显错排。
- Re-rank：在较小 slate 上做 LambdaRank 或其他 Listwise 目标，专门优化 NDCG@10。

这个分层方案成立的原因不是“层数越多越高级”，而是每层看到的数据、延迟预算、业务职责都不同。

适用边界也要说清楚：

- 如果你只有离散标签和大规模单条样本，没有稳定 session，就不要硬上 Listwise。
- 如果你的业务需要明确概率输出做出价或预估，不能只用 Pairwise。
- 如果你的核心指标是首页前几位质量，单靠 Pointwise 往往不够。

---

## 参考资料

1. Ranking Objectives: Pointwise versus Pairwise versus Listwise Optimization  
   说明点式、对式、列表式三类目标的核心差异，适合先建立整体框架。

2. Listwise Ranking: Optimizing the Entire List with Metric-Aware Losses  
   重点解释 Listwise 为什么更接近 NDCG 这类指标，以及 metric-aware loss 的基本思想。

3. TensorFlow Recommenders Listwise Ranking Example  
   给出 listwise 排序实验和代码示例，适合对照实现路径理解工程落地方式。

4. Bayesian Personalized Ranking (BPR) Algorithm  
   解释 BPR 的 pairwise 建模方式和公式来源，适合复查为什么它本质上在优化“正样本胜过负样本”。

5. 关于曝光偏差与排序学习的讨论论文  
   说明为什么日志数据中的未点击不一定等于负反馈，以及负采样和曝光建模为何会影响训练目标的真实性。
