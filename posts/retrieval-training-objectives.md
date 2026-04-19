## 核心结论

召回模型的训练目标，本质是在学习一个表示空间：对同一个用户或请求，正样本的分数应该高于负样本。记用户或请求为 $u$，物品为 $i$，模型分数为 $s(u,i)$，温度系数为 $\tau$，Sigmoid 函数为 $\sigma(x)=1/(1+e^{-x})$。

四类常见目标的差异不在于“谁更高级”，而在于它们给模型的学习信号不同。

| 目标 | 学习对象 | 输入形式 | 适用场景 | 优点 | 缺点 |
|---|---|---|---|---|---|
| Pointwise | 单个样本是否相关 | $(u,i,y)$ | 点击率校准、二分类召回 | 稳定、直观、容易调试 | 不直接优化正负排序 |
| Pairwise / BPR | 正样本是否高于负样本 | $(u,i^+,i^-)$ | 隐式反馈排序 | 对排序偏好更直接 | 依赖负样本质量 |
| Listwise / Softmax | 正样本在候选集中的概率 | $(u,C_u,i^+)$ | 候选集内归一化训练 | 更贴近召回 top-K | 全量候选代价高 |
| Contrastive / InfoNCE | query 与正样本向量对齐 | $(z_u,z_i,B)$ | 两塔召回、大规模向量检索 | 适合 in-batch negatives | 对 batch 和假负样本敏感 |

玩具例子：同一组分数，正样本 $s^+=2$，负样本 $s_1^-=1$，$s_2^-=0$。Pointwise 会分别看每个样本像不像正负类；BPR 只看 $s^+-s^-$；Softmax 和 InfoNCE 会看正样本在候选集合里的归一化概率。因此同一组分数会产生不同损失，优化重点也不同。

真实工程例子：电商两塔召回中，商品量级可能是百万到亿级。训练时通常让用户塔输出 query embedding，让商品塔输出 item embedding，用 InfoNCE、sampled softmax 或 in-batch negatives 学向量空间；线上再用 ANN 检索 top-K 商品。这里模型首先要“可检索”，不只是输出一个校准后的点击概率。

---

## 问题定义与边界

召回阶段的任务不是最终排序。召回是从海量候选中快速找出几百或几千个“值得进入下一阶段”的 item。粗排和精排再使用更多特征、更复杂模型做精细打分。

| 阶段 | 输入规模 | 主要目标 | 常见模型 | 本文是否讨论 |
|---|---:|---|---|---|
| 召回 | 百万到亿级 | 快速找回相关候选 | 双塔、向量检索、规则召回 | 是 |
| 粗排 | 千到万级 | 初步排序和过滤 | 轻量 GBDT、DNN | 否 |
| 精排 | 百级 | 精细预估点击、转化、停留 | 多任务模型、特征交叉模型 | 否 |

本文只讨论召回模型的训练目标，不展开重排序特征工程、广告竞价、特征平台、在线 serving 架构。

需要区分两个目标：

| 目标 | 白话解释 | 更常见的训练目标 |
|---|---|---|
| 点击概率校准 | 分数要像真实点击概率 | Pointwise BCE |
| 向量空间可检索性 | 相似 query 和 item 要靠近 | BPR、Softmax、InfoNCE |

电商中，“用户是否点击某商品”可以建成二分类问题，用 Pointwise 学概率。但两塔召回更关心正样本是否在向量空间里离 query 更近，因此常用 InfoNCE 或 sampled softmax。

---

## 核心机制与推导

Pointwise 是单点学习。它把每个 $(u,i)$ 当作一个二分类样本，标签 $y_{ui}\in\{0,1\}$：

$$
\mathcal L_{pt}=-\sum_{(u,i)}[y_{ui}\log \sigma(s(u,i))+(1-y_{ui})\log(1-\sigma(s(u,i)))]
$$

这里 $y_{ui}=1$ 表示正样本，$y_{ui}=0$ 表示负样本。它关心“这个样本像不像正类”，不直接关心另一个 item 的分数是否更低。

Pairwise 是成对学习。BPR，Bayesian Personalized Ranking，白话说就是“让用户喜欢的 item 分数高于不喜欢或未交互的 item”：

$$
\mathcal L_{pr}=-\sum_{(u,i,j)}\log \sigma(s(u,i)-s(u,j))
$$

其中 $i$ 是正样本，$j$ 是负样本。它只看差值 $s(u,i)-s(u,j)$。如果正样本已经远高于负样本，损失会很小，梯度也会变弱。

Listwise 是候选集学习。Softmax 把一个候选集合 $C_u$ 里的分数归一化成概率：

$$
\mathcal L_{ls}=-\sum_u \log \frac{\exp(s(u,i^+)/\tau)}{\sum_{k\in C_u}\exp(s(u,k)/\tau)}
$$

$\tau$ 是温度，控制分布尖锐程度。$\tau$ 越小，高分样本越占优势，训练信号越集中。全量 softmax 要遍历所有 item，在大规模推荐中通常不可行，所以工程上会用 sampled softmax 或 in-batch negatives 近似分母。

Contrastive 是对比表示学习。InfoNCE，Noise Contrastive Estimation 的一种常见形式，白话说就是“把匹配的向量拉近，把同批其他向量推远”：

$$
\mathcal L_{nce}=-\log \frac{\exp(\mathrm{sim}(z_u,z_{i^+})/\tau)}{\sum_{k\in B}\exp(\mathrm{sim}(z_u,z_k)/\tau)}
$$

$z_u$ 和 $z_i$ 是 embedding，$\mathrm{sim}$ 通常是点积或余弦相似度，$B$ 是 batch 内 item 集合。in-batch negatives 指把同一个 batch 中其他样本的 item 当作当前 query 的负样本。

比较粒度可以这样理解：

| 目标 | 比较粒度 | 模型收到的主要信号 |
|---|---|---|
| Pointwise | 单点 | 这个 item 是否为正 |
| BPR | 成对 | 正样本是否高于一个负样本 |
| Softmax | 候选集 | 正样本是否压过集合内其他候选 |
| InfoNCE | 对比空间 | 正向量是否比其他向量更近 |

数值上，设 $s^+=2,s_1^-=1,s_2^-=0$。Pointwise 若只取正样本 2 和负样本 0，损失约为 $-\log\sigma(2)-\log(1-\sigma(0))=0.820$。BPR 若比较 $2$ 和 $0$，损失约为 $-\log\sigma(2)=0.127$。Softmax 候选集为 $\{2,1,0\}$ 时，损失约为 $-\log(e^2/(e^2+e^1+e^0))=0.408$。这些数值不能横向比较好坏，因为它们的归一化方式不同。

---

## 代码实现

先用一个可运行的 Python 例子复现上面的数值。这里不用深度学习框架，只计算 loss，便于检查公式。

```python
import math

def sigmoid(x):
    return 1 / (1 + math.exp(-x))

s_pos, s_neg1, s_neg2 = 2.0, 1.0, 0.0

pointwise = -math.log(sigmoid(s_pos)) - math.log(1 - sigmoid(s_neg2))
bpr = -math.log(sigmoid(s_pos - s_neg2))
softmax = -math.log(math.exp(s_pos) / (math.exp(s_pos) + math.exp(s_neg1) + math.exp(s_neg2)))

assert round(pointwise, 3) == 0.820
assert round(bpr, 3) == 0.127
assert round(softmax, 3) == 0.408
```

模型接口应先统一：query embedding、item embedding、score、negative samples 不要为每种 loss 重写一套模型。

```python
import torch
import torch.nn.functional as F

def score(query_emb, item_emb):
    return (query_emb * item_emb).sum(dim=-1)

q = torch.randn(4, 8)
pos = torch.randn(4, 8)
neg = torch.randn(4, 8)

pos_score = score(q, pos)
neg_score = score(q, neg)
labels = torch.cat([torch.ones(4), torch.zeros(4)])
logits = torch.cat([pos_score, neg_score])

pointwise_bce = F.binary_cross_entropy_with_logits(logits, labels)
assert pointwise_bce.ndim == 0
```

BPR 需要显式构造正负样本对。它不要求分数是概率，只要求正样本分数高于负样本。

```python
import torch
import torch.nn.functional as F

q = torch.randn(4, 8)
pos = torch.randn(4, 8)
neg = torch.randn(4, 8)

pos_score = (q * pos).sum(dim=-1)
neg_score = (q * neg).sum(dim=-1)

bpr_loss = -F.logsigmoid(pos_score - neg_score).mean()
assert bpr_loss.ndim == 0
```

InfoNCE 常用 batch 内负样本。对第 $r$ 个 query 来说，第 $r$ 个 item 是正样本，其他 item 是负样本。

```python
import torch
import torch.nn.functional as F

batch_size, dim, tau = 4, 8, 0.1
q = F.normalize(torch.randn(batch_size, dim), dim=-1)
item = F.normalize(torch.randn(batch_size, dim), dim=-1)

logits = q @ item.T / tau
target = torch.arange(batch_size)

info_nce = F.cross_entropy(logits, target)
assert logits.shape == (batch_size, batch_size)
assert info_nce.ndim == 0
```

batch 构造的关键是保证正样本对齐，并控制负样本来源。

```python
batch = [
    {"user_id": 1, "pos_item": 101},
    {"user_id": 2, "pos_item": 205},
    {"user_id": 3, "pos_item": 309},
]

# 两塔训练时：
# 1. user_id -> query tower -> q
# 2. pos_item -> item tower -> item
# 3. q @ item.T 得到 batch_size * batch_size 分数矩阵
# 4. 对角线是正样本，非对角线是 in-batch negatives

assert len(batch) == 3
assert batch[0]["pos_item"] != batch[1]["pos_item"]
```

---

## 工程权衡与常见坑

召回训练最容易出问题的地方不是公式，而是样本。未点击不等于负样本。用户从未曝光过某商品，可能只是没看到，不是讨厌它。把所有未点击都当负样本，会让模型学习到错误偏好。

| 常见坑 | 影响 | 规避方式 |
|---|---|---|
| 假负样本 | 把潜在喜欢的 item 推远 | 只从曝光未点中采负、加时间窗、降权 |
| 负样本过易 | BPR 梯度很弱 | 引入 hard negative、相似负样本、流行度采样 |
| batch 太小 | InfoNCE 分母负样本不足 | 增大 batch、跨 batch memory、额外采样 |
| 同批正样本冲突 | 一个用户的正样本可能是另一个用户的假负样本 | 去重、同会话合并、屏蔽相关 item |
| loss 数值不可横比 | 误判模型优劣 | 用 Recall@K、NDCG@K、线上指标比较 |

真实工程中，百万级商品两塔召回通常会混合多种负样本：一部分随机负样本保证覆盖，一部分热门负样本提升区分度，一部分 hard negative 逼模型学习细粒度差异。hard negative 指“看起来很像正样本但实际不是当前正样本”的负样本。

还要注意温度 $\tau$。温度太小，模型会过度关注少数最高分负样本，训练可能不稳定；温度太大，分布太平，区分信号变弱。实践中需要结合 batch size、embedding 归一化方式和离线 Recall@K 调参。

---

## 替代方案与适用边界

不是所有召回任务都必须用 InfoNCE。目标函数要服务于业务目标和数据条件。

| 任务类型 | 推荐目标函数 | 适用理由 |
|---|---|---|
| 点击概率校准 | Pointwise BCE | 输出接近概率，稳定直观 |
| 排序偏好学习 | BPR / Pairwise hinge | 直接优化正负相对顺序 |
| 大规模向量检索 | InfoNCE / sampled softmax | 贴近两塔召回和 ANN 检索 |
| 稀疏反馈 | BPR、加权 Pointwise | 能处理隐式反馈，但要谨慎采负 |
| 小规模内容推荐 | Pointwise 或 BPR | 候选不大，不必引入复杂对比训练 |

小规模内容推荐、标注充分、需要解释“点击概率是多少”时，Pointwise 更直接。比如内部知识库推荐，文章规模只有几千篇，样本标签质量较高，用 BCE 训练一个打分模型可能已经足够。

百万级商品召回、目标是线上向量检索时，应优先考虑 InfoNCE、sampled softmax 或其他 listwise 近似。因为线上检索本身就是“query 向量找最近 item 向量”，训练目标越贴近这个过程，embedding 通常越可用。

当曝光不完整、反馈噪声大、负样本质量差时，更强的目标不一定更好。InfoNCE 会把同批其他 item 当负样本，如果 batch 中存在大量用户也可能喜欢的 item，就会把相关 item 错误推远。此时需要样本去重、类目约束、曝光校正，或者使用更保守的 Pointwise / 加权 Pairwise 目标。

术语表：

| 术语 | 白话解释 |
|---|---|
| BPR | 让正样本分数高于负样本的排序损失 |
| InfoNCE | 拉近正样本向量、推远负样本向量的对比损失 |
| sampled softmax | 不遍历全量类别，只采一部分负样本近似 softmax |
| in-batch negatives | 把同一个 batch 中其他样本当作负样本 |

---

## 参考资料

1. [BPR: Bayesian Personalized Ranking from Implicit Feedback](https://www.cs.mcgill.ca/~uai2009/papers/UAI2009_0139_48141db02b9f0b02bc7158819ebfa2c7.pdf)  
这篇论文适合支持 Pairwise / BPR 的核心思想：隐式反馈中直接学习正负样本的相对顺序。

2. [A Simple Framework for Contrastive Learning of Visual Representations](https://arxiv.org/pdf/2002.05709.pdf)  
这篇论文适合支持 InfoNCE / NT-Xent 的对比学习形式，尤其是温度参数和 batch 内对比的机制。

3. [TensorFlow: tf.nn.sampled_softmax_loss](https://www.tensorflow.org/api_docs/python/tf/nn/sampled_softmax_loss)  
这份文档适合支持 sampled softmax 是大类别数 softmax 训练近似这一工程结论。

4. [On the Effectiveness of Sampled Softmax Loss for Item Recommendation](https://zlstwebsite.github.io/Zlst/publication/on-the-effectiveness-of-sampled-softmax-loss-for-item-recommendation/)  
这篇研究适合支持 sampled softmax 在 item recommendation 召回场景中的有效性讨论。

5. [Neural Collaborative Filtering](https://huggingface.co/papers/1708.05031)  
这篇论文适合支持 Pointwise 二分类目标在隐式反馈推荐建模中的典型用法。
