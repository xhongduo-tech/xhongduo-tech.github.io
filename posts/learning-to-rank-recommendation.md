## 核心结论

Learning to Rank，简称 LTR，是把“排序”当成监督学习问题来做：给定一个用户和一批候选商品，模型要做的不是判断每个商品“好不好”，而是决定“谁排前面”。

在推荐系统里，可以把用户 $u$ 看成信息检索里的 query，把候选物品集合 $D_u$ 看成 documents。模型对每个用户-物品对输出一个分数：

$$
s_i = f_\theta(u, i)
$$

其中 $f_\theta$ 是带参数的模型，$s_i$ 是物品 $i$ 的排序分数。最后按分数从高到低排序：

$$
\pi_u = sort(s)
$$

$\pi_u$ 表示用户 $u$ 看到的最终物品顺序。

推荐排序真正关心的是 top-K，也就是前 K 个展示结果。首页推荐、搜索结果、短视频流都不是平均地关心所有候选，而是更关心前几个位置是否排对。因此，相比普通分类，LTR 更应该对齐排序指标，例如 `NDCG@K`、`MRR`，或者它们的可优化近似目标。

| 任务 | 输入 | 模型输出 | 优化重点 | 典型指标 |
|---|---|---|---|---|
| 分类 | 单个样本 | 类别概率 | 单个样本是否预测正确 | AUC、LogLoss、Accuracy |
| 回归 | 单个样本 | 连续数值 | 单个数值是否接近标签 | MSE、MAE |
| 排序 | 一个候选列表 | 每个候选的分数 | 候选之间的相对顺序 | NDCG@K、MRR、MAP |

玩具例子：一个用户面前有 10 个候选结果，系统最在意的是前 3 个有没有排对，而不是每个结果分别打了几分。某个商品分数从 0.71 变成 0.73 不重要，重要的是它是否因此从第 8 位升到第 2 位。

真实工程例子：电商首页通常先由召回系统找出几百个可能相关商品，再由排序模型决定前 10 个展示什么。用户最终能看到的位置有限，所以排序模型的错误主要体现在“高价值商品被压到后面”或“低相关商品占据首屏”。

---

## 问题定义与边界

推荐排序问题可以定义为：对每个用户 $u$，给定候选集合 $D_u=\{i_1,i_2,\dots,i_n\}$，模型输出每个候选 item 的分数 $s_i$，再根据分数得到排序 $\pi_u$，目标是让高相关、高收益或高满意度的 item 排在前面。

术语说明：

| 符号 | 含义 |
|---|---|
| $u$ | 用户，也可以理解为一次推荐请求 |
| $i,j$ | 候选物品，可能是商品、视频、文章、广告 |
| $D_u$ | 用户 $u$ 的候选物品集合 |
| $y_i$ | 物品 $i$ 的监督标签，例如点击、购买、评分、相关性等级 |
| $s_i$ | 模型给物品 $i$ 输出的排序分数 |
| $\pi_u$ | 按分数排序后的物品顺序 |

| 对象 | 输入 | 输出 | 目标 |
|---|---|---|---|
| 用户 | 用户画像、历史行为、上下文 | 用户请求表示 | 表达当前意图 |
| 物品 | 物品属性、统计特征、内容特征 | 物品表示 | 表达候选价值 |
| 用户-物品对 | 用户特征、物品特征、交叉特征 | 分数 $s_i$ | 判断相对排序位置 |
| 候选列表 | 多个候选 item | 排序 $\pi_u$ | 优化 top-K 质量 |

LTR 不等于所有推荐问题。它最适合候选集已经比较小、需要精排或重排的阶段，不是召回的替代品。

| 推荐阶段 | 主要目标 | 常见规模 | LTR 适配度 |
|---|---|---:|---|
| 召回 | 找全可能相关候选 | 万级到百万级 | 低，重点是高召回和低延迟 |
| 粗排 | 快速过滤明显不合适的候选 | 千级到万级 | 中，可用轻量排序模型 |
| 精排 | 对候选做精细打分 | 百级到千级 | 高，适合 LTR |
| 重排 | 结合多样性、规则、业务约束调整顺序 | 十级到百级 | 高，但常和规则或约束优化结合 |

电商首页例子：召回阶段先找出 500 个候选商品，LTR 负责把这 500 个商品排成一个更符合用户意图的顺序。直白地说，先把可能相关的都找出来，再决定谁最该排前面。

边界要明确：如果系统连候选都找不到，问题是召回；如果候选已经确定，但前 10 个展示结果经常不合理，问题才更接近 LTR。

---

## 核心机制与推导

LTR 常见方法分为 Pointwise、Pairwise、Listwise。

Pointwise 是单点预测：把每个用户-物品对当成独立样本，预测点击率、评分或相关性分数。损失形式是：

$$
L_{pt} = \sum_i \ell(s_i, y_i)
$$

其中 $\ell$ 是单样本损失，例如交叉熵或均方误差。

Pairwise 是成对比较：不直接关心分数绝对值，而是关心“标签更高的 item 分数是否更高”。损失形式是：

$$
L_{pw} = \sum_{y_i > y_j} \log(1 + \exp(-(s_i - s_j)))
$$

如果 $y_i>y_j$，模型应该让 $s_i>s_j$。当 $s_i-s_j$ 越大，损失越小。

Listwise 是列表级学习：直接从整张候选列表的排序质量出发。常见指标是 DCG 和 NDCG。DCG 的意思是 Discounted Cumulative Gain，即带位置折扣的累计收益：

$$
DCG@K = \sum_{k=1}^{K} \frac{2^{y_{\pi_k}} - 1}{\log_2(k+1)}
$$

$y_{\pi_k}$ 表示排在第 $k$ 位的 item 标签。位置越靠后，分母越大，收益折扣越强。NDCG 是归一化 DCG：

$$
NDCG@K = \frac{DCG@K}{IDCG@K}
$$

IDCG 是理想排序下的 DCG。NDCG 的取值通常在 0 到 1 之间，越接近 1 表示排序越接近理想顺序。

| 方法 | 优化对象 | 优点 | 缺点 | 适用场景 |
|---|---|---|---|---|
| Pointwise | 单个 item 的标签 | 实现简单，可直接复用分类/回归模型 | 不直接建模相对顺序 | CTR 预估、评分预测、简单精排 |
| Pairwise | 两个 item 的相对顺序 | 更符合排序目标，适合隐式偏好 | 样本对数量多，采样影响大 | 偏好学习、BPR、RankNet |
| Listwise | 整个候选列表 | 更贴近 top-K 指标 | 实现复杂，对列表构造敏感 | 精排、搜索排序、重排 |
| LambdaLoss | 带指标权重的成对优化 | 能把梯度权重对齐到 NDCG | 理论和实现门槛更高 | LambdaRank、LambdaMART、工业排序 |

用同一个 3-item 玩具例子看三种方法。标签是 $y=[3,2,0]$，模型分数是 $s=[0.8,0.9,0.1]$。真实最相关的是第一个 item，但模型把第二个 item 排到了第一位。只看单个分数，很难直接看出排序错误的代价；看排序结果就很清楚：前两名反了。

在 Pairwise 里，错误主要来自样本对 $(1,2)$，因为 $y_1>y_2$，但 $s_1<s_2$。这对的损失是：

$$
\log(1+\exp(-(0.8-0.9))) \approx 0.74
$$

LambdaRank 进一步引入 NDCG 的变化量。对一对 item $(i,j)$，如果交换它们的位置会造成较大的 NDCG 变化，就应该给这对样本更大的训练权重：

$$
\lambda_{ij} = |\Delta NDCG_{ij}| \cdot \sigma(-(s_i - s_j))
$$

其中 $\sigma(t)=1/(1+e^{-t})$ 是 sigmoid 函数，意思是把数值压到 0 到 1 之间。$|\Delta NDCG_{ij}|$ 表示交换 $i,j$ 后 NDCG 的绝对变化。

推导直觉很直接：NDCG 的收益由标签和位置共同决定。高标签 item 如果被排到很靠后，或者两个高影响位置发生错误交换，DCG 变化会很大；尾部两个低相关 item 互换，DCG 变化很小。因此 $|\Delta NDCG_{ij}|$ 越大，说明这对样本越影响最终排序质量，训练时就应该被更强地修正。

---

## 代码实现

下面代码给出一个最小可运行版本：包含 pairwise loss、NDCG 评估，以及上面 3-item 玩具例子的断言。这里不用深度学习框架，目的是把排序损失和指标计算讲清楚。

```python
import math

def sigmoid(x):
    return 1.0 / (1.0 + math.exp(-x))

def pairwise_loss(scores, labels):
    loss = 0.0
    count = 0
    n = len(scores)
    for i in range(n):
        for j in range(n):
            if labels[i] > labels[j]:
                loss += math.log(1.0 + math.exp(-(scores[i] - scores[j])))
                count += 1
    return loss / max(count, 1)

def dcg_at_k(scores, labels, k):
    order = sorted(range(len(scores)), key=lambda idx: scores[idx], reverse=True)
    total = 0.0
    for rank, idx in enumerate(order[:k], start=1):
        gain = (2 ** labels[idx]) - 1
        discount = math.log2(rank + 1)
        total += gain / discount
    return total

def ndcg_at_k(scores, labels, k):
    dcg = dcg_at_k(scores, labels, k)
    ideal_scores = labels[:]
    idcg = dcg_at_k(ideal_scores, labels, k)
    return dcg / idcg if idcg > 0 else 0.0

labels = [3, 2, 0]
scores = [0.8, 0.9, 0.1]

loss = pairwise_loss(scores, labels)
ndcg = ndcg_at_k(scores, labels, 3)

assert round(loss, 2) == 0.50
assert 0.78 < ndcg < 0.80

better_scores = [0.9, 0.8, 0.1]
assert ndcg_at_k(better_scores, labels, 3) == 1.0
```

在真实 PyTorch 工程里，结构通常是下面这样：

```python
# user_features: 用户特征
# item_features: 候选物品特征
# context_features: 场景特征，例如时间、入口、设备

scores = model(user_features, item_features, context_features)
loss = pairwise_loss(scores, labels)  # 也可以换成 lambda_loss
ndcg = compute_ndcg(scores, labels, k=10)
```

Pairwise 的核心是构造样本对。假设某用户曝光了 5 个商品，其中点击商品是正样本，未点击但已曝光商品是可用负样本。模型输出每个 item 的分数后，把正样本和负样本两两比较：谁应该排前面，就拉开谁的分数差。

| 训练环节 | 输入 | 处理方式 | 输出 |
|---|---|---|---|
| 数据准备 | 曝光、点击、购买、停留时长 | 清洗日志，生成标签 | 用户级训练列表 |
| 候选构造 | 召回结果或曝光列表 | 保留同一次请求内候选 | $D_u$ |
| 损失计算 | 分数与标签 | Pointwise、Pairwise、LambdaLoss | 训练损失 |
| 评估 | 离线验证集 | 计算 NDCG@K、MRR、Recall@K | 排序质量 |
| 线上推理 | 实时用户、候选 item、上下文 | 输出分数并排序 | 展示列表 |

真实工程例子：短视频推荐精排会把用户近期观看、点赞、跳出、关注等行为做成用户特征，把视频作者、类目、时长、热度做成 item 特征，再加入时间、网络、入口等 context 特征。排序模型输出每个视频的分数，最后结合去重、多样性、风控和业务规则生成最终列表。

---

## 工程权衡与常见坑

推荐排序最常见的问题不是“模型不会学”，而是训练目标、样本构造、评估指标、线上流量不一致。这四者任何一个偏了，离线指标可能好看，线上效果仍然下降。

最典型的坑是把“未点击”直接当负样本。一个用户没点某商品，可能只是因为没看到，而不是不喜欢。没点过，不代表不想点；没曝光过，更不能当负反馈。

| 问题 | 表现 | 原因 | 规避方式 |
|---|---|---|---|
| 曝光偏差 | 模型学到“展示过什么”，而不是“用户喜欢什么” | 只有曝光 item 才有点击机会 | 只在曝光集采样，必要时做位置偏差修正 |
| 时间穿越 | 离线效果异常好，线上变差 | 训练特征使用了未来信息 | 按时间切分数据，检查特征生成时间 |
| 候选池漂移 | 离线排序有效，线上候选分布不同 | 训练候选和线上召回候选不一致 | 尽量复用线上召回链路构造训练样本 |
| 目标指标不一致 | LogLoss 降了，但 CTR 或 GMV 没涨 | 单点预测目标不等于排序目标 | top-K 场景优先对齐 NDCG@K、MRR |
| 位置偏差 | 排在前面的 item CTR 天然更高 | 用户更容易看到前排 item | 加入位置特征、反事实评估或随机桶校正 |

位置偏差的小例子：同样一个商品，出现在第 1 屏和第 5 屏，点击率不能直接当作同一分布。第 1 屏 CTR 高，可能只是因为它更容易被看到；第 5 屏 CTR 低，也可能不是商品差，而是用户根本没滑到那里。

| 处理规则 | 具体做法 |
|---|---|
| 只在曝光集采样 | 负样本来自“用户确实看过但没点”的 item |
| 按时间切分数据 | 用过去训练，用未来验证，避免随机切分导致泄漏 |
| 复用线上召回链路 | 训练列表尽量来自真实候选生成流程 |
| 对齐 top-K 指标 | 推荐首屏优先看 NDCG@K、MRR、Recall@K |
| 分层看指标 | 按新老用户、类目、流量入口分别评估 |

工程上还要权衡模型复杂度和延迟。LambdaMART、深度排序模型、交叉特征模型都可能提升排序质量，但精排通常有严格延迟预算。如果一次请求有 500 个候选，每个候选都要打分，模型每增加 1 毫秒都可能影响整体响应时间。因此工业系统常用多阶段结构：召回负责找全，粗排负责快速缩小范围，精排负责精细排序，重排负责多样性和规则约束。

---

## 替代方案与适用边界

LTR 不是所有推荐系统的唯一选择。它最适合“已经有候选集，需要把候选集排好”的阶段。如果任务是召回、探索、长期价值优化，就需要别的方法。

电商首页重排可以说明边界：如果系统只是想找出 500 个可能相关商品，LTR 不是最合适的；如果系统已经拿到 500 个候选，要决定前 10 个展示什么，LTR 更合适。也就是先找候选，再排顺序。LTR 解决的是排序，不是从海量库里找出所有可能相关对象。

| 方法 | 核心思想 | 适用场景 | 边界 |
|---|---|---|---|
| 分类 / 回归 | 预测点击率、转化率或评分 | 简单精排、CTR 预估 | 不直接优化相对顺序 |
| Pairwise ranking | 学习 item 之间谁更靠前 | 隐式偏好、稳定排序任务 | 样本对采样很关键 |
| Listwise ranking | 直接优化列表质量 | top-K 精排、搜索排序 | 实现复杂，对列表构造敏感 |
| BPR | 从隐式反馈中学习用户偏好 | 用户-物品推荐、协同过滤排序 | 不一定直接优化 NDCG |
| LambdaMART | GBDT + Lambda 梯度 | 表格特征强、工业排序 | 特征工程依赖较强 |
| 多目标排序 | 同时优化点击、转化、时长、收益 | 电商、广告、内容流 | 目标权重需要业务校准 |
| 强化学习 / bandit | 优化探索和长期收益 | 冷启动、探索、长期留存 | 训练和评估复杂度高 |

判断是否适合用 LTR，可以看四个问题：

| 判断问题 | 如果答案是“是” | 结论 |
|---|---|---|
| 是否有明确候选集 | 每次请求已有几十到几千个候选 | 适合排序模型 |
| 是否主要关注 top-K | 前几个位置决定主要收益 | 适合 NDCG、MRR、LambdaLoss |
| 是否有曝光日志 | 能知道用户看到了哪些 item | 可以构造更可靠的训练样本 |
| 是否需要对齐线上业务指标 | CTR、CVR、GMV、停留时长很重要 | 需要排序目标和业务目标联合设计 |

什么时候不用 LTR：召回阶段更适合向量检索、双塔模型、协同过滤或图召回；探索阶段更适合 bandit 或探索策略；长期收益建模可能需要强化学习、因果推断或用户生命周期模型。LTR 可以把候选排好，但不能替代整个推荐系统。

---

## 参考资料

1. [Learning to Rank using Gradient Descent / RankNet](https://www.microsoft.com/en-us/research/publication/learning-to-rank-using-gradient-descent/)：对应本文 Pairwise 排序损失部分，解释如何用成对比较学习排序。
2. [Learning to Rank: From Pairwise Approach to Listwise Approach](https://www.microsoft.com/en-us/research/publication/learning-to-rank-from-pairwise-approach-to-listwise-approach-2/)：对应本文 Pointwise、Pairwise、Listwise 方法对比部分。
3. [From RankNet to LambdaRank to LambdaMART: An Overview](https://www.microsoft.com/en-us/research/publication/from-ranknet-to-lambdarank-to-lambdamart-an-overview/)：对应本文 LambdaLoss、LambdaRank 和 LambdaMART 的机制说明。
4. [Direct Optimization of Evaluation Measures in Learning to Rank](https://www.microsoft.com/en-us/research/publication/direct-optimization-of-evaluation-measures-in-learning-to-rank/)：对应本文 NDCG 与排序指标直接优化部分。
5. [BPR: Bayesian Personalized Ranking from Implicit Feedback](https://arxiv.org/abs/1205.2618)：对应本文隐式反馈、Pairwise ranking 和替代方案边界部分。
