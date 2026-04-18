## 核心结论

Swish 召回模型本质上是“双塔召回”的轻量改造版：整体结构仍然是用户塔和物品塔，主要变化是把隐藏层里的 `ReLU` 换成 `Swish/SiLU`，并配合 `LayerNorm` 和对比学习，让模型更稳定地学到可检索向量。

双塔是指两个独立编码器：一个把用户特征编码成向量，另一个把物品特征编码成向量。向量是由数字组成的表示，例如 `[0.2, -0.1, 0.8]`，模型用它表达用户兴趣或物品语义。

新手版本可以这样理解：用户塔把用户历史行为变成一个向量，物品塔把商品信息变成一个向量。训练时让点击过的商品更接近用户，让没点过的商品更远。Swish 和 LayerNorm 的作用，就是让这个过程更平滑、更不容易训练崩。

核心公式有两个：

$$
Swish(x)=x \cdot \sigma(x)
$$

$$
s(u,v)=\frac{z_u^T z_v}{\tau}
$$

其中 $z_u$ 是用户向量，$z_v$ 是物品向量，$\tau$ 是温度系数。温度系数是控制 softmax 分布尖锐程度的超参数，值越小，模型越强调高相似度样本。

整体链路可以写成：

```text
输入特征 -> 双塔编码 -> LayerNorm -> Swish -> 向量归一化 -> 相似度 -> 对比学习损失 -> ANN 召回
```

ANN 是近似最近邻检索，意思是不精确遍历全部物品，而是用索引结构快速找到最相似的一批向量。Swish 双塔的价值不在“模型更复杂”，而在“结构简单、训练稳定、向量容易用于线上检索”。

---

## 问题定义与边界

推荐系统通常分为召回、排序、重排。召回阶段要从海量物品中快速找出少量候选，排序阶段再对这些候选做精细打分。

新手版本：如果首页有 100 万个商品，不可能一个个算分。召回模型先从中找出 200 个最像用户兴趣的商品，再交给精排模型做最终判断。

Swish 双塔解决的是召回问题，不是最终排序问题。它关注的是低延迟、广覆盖、可检索，而不是复杂特征交叉下的最高点击率预测。

| 阶段 | 目标 | 特点 | Swish 双塔是否适合 |
|---|---|---|---|
| 召回 | 快速缩小候选集 | 低延迟、粗筛 | 适合 |
| 精排 | 精细预测点击/转化 | 高精度、复杂特征交叉 | 不适合 |
| 重排 | 优化最终展示策略 | 多目标约束 | 不适合 |

几个术语需要先定义清楚。

正样本是用户真实点击、购买、收藏过的物品。负样本是训练时被当作“不匹配”的物品。对比学习是一种训练方法：它不只问“这个样本是不是点击”，而是让正样本比负样本更接近用户。

向量检索是把用户和物品都变成向量后，用点积或余弦相似度找最近的物品。点积是两个向量对应位置相乘再求和。余弦相似度衡量两个向量方向是否接近，通常会先做 L2 归一化。

这个边界很重要：双塔召回为了能离线建物品向量，通常不在用户和物品之间做复杂交叉。它牺牲一部分表达能力，换来大规模低延迟检索能力。

真实工程例子：电商首页召回中，用户侧输入最近点击、购买类目、价格偏好、地域和时间上下文；物品侧输入商品类目、标题向量、价格带、销量统计。离线先把全量商品编码成向量并写入 ANN 索引，线上只计算用户向量，再检索 TopK 商品。这个流程适合百万到亿级候选池。

---

## 核心机制与推导

Swish 双塔由四个关键部分组成：双塔结构、Swish 激活、LayerNorm 归一化、对比学习损失。

用户塔和物品塔分别表示为：

$$
z_u=f_u(x_u), \quad z_v=f_v(x_v)
$$

其中 $x_u$ 是用户特征，$x_v$ 是物品特征，$f_u$ 和 $f_v$ 是两个神经网络编码器。编码器是把原始特征变成固定长度向量的函数。

Swish 定义为：

$$
Swish(x)=x \cdot \sigma(x)
$$

$$
\sigma(x)=\frac{1}{1+e^{-x}}
$$

ReLU 的定义是 $ReLU(x)=max(0,x)$。它的问题是负半轴输出为 0，梯度也会被截断。Swish 在负区间不会直接清零，而是保留平滑变化，这让隐藏层更新更连续。

玩具例子：当 $x=1$ 时，$ReLU(1)=1$，$Swish(1)=1 \cdot \sigma(1) \approx 0.731$。当 $x=-1$ 时，$ReLU(-1)=0$，$Swish(-1)\approx -0.269$。这说明 Swish 不是简单丢掉负值，而是保留了一部分信息和梯度。

LayerNorm 的公式是：

$$
LN(h)=\gamma \odot \frac{h-\mu}{\sqrt{\sigma^2+\epsilon}}+\beta
$$

LayerNorm 是层归一化，白话解释就是：对同一个样本的一层输出做尺度调整，让数值分布更稳定。$\mu$ 是均值，$\sigma^2$ 是方差，$\gamma$ 和 $\beta$ 是可学习参数。

对比学习损失可以写成：

$$
L=-\log
\frac{
\exp(s(u,v^+)/\tau)
}{
\exp(s(u,v^+)/\tau)+\sum_i \exp(s(u,v_i^-)/\tau)
}
$$

其中 $v^+$ 是正样本物品，$v_i^-$ 是负样本物品。这个损失的含义是：正样本相似度越高，负样本相似度越低，损失越小。

最小数值例子如下。设用户向量 $z_u=[1,0.5]$，正样本 $z_p=[0.8,0.6]$，负样本 $z_n=[0.2,0.9]$，温度 $\tau=0.5$。

正样本点积为：

$$
1 \cdot 0.8 + 0.5 \cdot 0.6 = 1.1
$$

负样本点积为：

$$
1 \cdot 0.2 + 0.5 \cdot 0.9 = 0.65
$$

代入 softmax 后，正样本概率更高，损失更低。训练会继续推动 $z_u$ 靠近 $z_p$，远离 $z_n$。

小结：`ReLU` 的主要问题是负半轴梯度被截断，`Swish` 保留了更平滑的梯度；`LayerNorm` 稳定隐藏层尺度；对比学习把点击物品拉近、把负样本推远。三者共同作用，目标是让向量空间更适合 ANN 检索。

---

## 代码实现

下面是一个只依赖 Python 标准库的最小可运行版本，用来演示 `Swish + LayerNorm + 点积相似度 + 对比损失` 的核心计算。它不是完整训练框架，但公式链路完整，代码可以直接运行。

```python
import math

def sigmoid(x):
    return 1 / (1 + math.exp(-x))

def swish(x):
    return x * sigmoid(x)

def layer_norm(xs, eps=1e-5):
    mean = sum(xs) / len(xs)
    var = sum((x - mean) ** 2 for x in xs) / len(xs)
    return [(x - mean) / math.sqrt(var + eps) for x in xs]

def dot(a, b):
    return sum(x * y for x, y in zip(a, b))

def l2_normalize(xs, eps=1e-12):
    norm = math.sqrt(sum(x * x for x in xs))
    return [x / (norm + eps) for x in xs]

def contrastive_loss(user, pos, negs, tau=0.5):
    user = l2_normalize(user)
    pos = l2_normalize(pos)
    negs = [l2_normalize(n) for n in negs]

    pos_logit = dot(user, pos) / tau
    neg_logits = [dot(user, n) / tau for n in negs]
    logits = [pos_logit] + neg_logits

    m = max(logits)
    exps = [math.exp(x - m) for x in logits]
    prob_pos = exps[0] / sum(exps)
    return -math.log(prob_pos), prob_pos

hidden = [1.0, -1.0, 0.5]
encoded = [swish(x) for x in layer_norm(hidden)]

user = encoded
pos_item = [0.8, -0.2, 0.4]
neg_items = [[-0.3, 0.9, 0.1], [0.1, 0.2, -0.8]]

loss, prob = contrastive_loss(user, pos_item, neg_items)

assert len(encoded) == 3
assert loss > 0
assert 0 < prob < 1
assert abs(swish(1.0) - 0.7310585786) < 1e-6

print(round(loss, 4), round(prob, 4))
```

在真实工程里通常会用 PyTorch 或 TensorFlow 实现双塔。核心结构如下：

```python
class Tower(nn.Module):
    def __init__(self, in_dim, hidden_dim, out_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.SiLU(),   # Swish
            nn.Linear(hidden_dim, out_dim),
        )

    def forward(self, x):
        z = self.net(x)
        return F.normalize(z, dim=-1)

def contrastive_loss(u, pos, neg, tau=0.5):
    pos_score = (u * pos).sum(dim=-1) / tau
    neg_score = (u.unsqueeze(1) * neg).sum(dim=-1) / tau
    logits = torch.cat([pos_score.unsqueeze(1), neg_score], dim=1)
    labels = torch.zeros(u.size(0), dtype=torch.long, device=u.device)
    return F.cross_entropy(logits, labels)
```

实现上要分清训练和推理。

训练时，用户塔、物品塔同时更新。正样本来自点击、购买、收藏等行为；负样本可以来自 batch 内其他物品、跨 batch 记忆池、随机采样、热门物品采样或混合采样。

推理时，物品塔通常离线运行。系统把全量物品提前编码成向量，写入 Faiss、ScaNN、Milvus、HNSW 等 ANN 索引。线上请求到来时，只计算用户向量，然后从索引里取相似度最高的 TopK 物品。

| 模块 | 实现方式 | 作用 |
|---|---|---|
| 激活函数 | `SiLU` | 平滑梯度 |
| 归一化 | `LayerNorm` | 稳定尺度 |
| 相似度 | 点积 / 余弦 | 向量检索友好 |
| 负样本 | in-batch / cross-batch / mixed | 提升区分度 |

---

## 工程权衡与常见坑

Swish 不是“换个激活函数就一定涨点”。它的收益依赖初始化、学习率、归一化位置、负样本质量、batch 规模和 ANN 检索链路。

新手版本：如果你只把 `ReLU` 换成 `Swish`，但其他设置不变，结果可能没变好，甚至变差。因为模型稳定性还受学习率、归一化和样本质量影响。

| 常见坑 | 直接后果 | 规避方式 |
|---|---|---|
| 只改激活函数 | 收益不稳定 | 做 ablation |
| `LayerNorm` 放错位置 | 幅值信息被抹掉 | 放在 dense block 内部 |
| 假负例太多 | 误伤相似商品 | 过滤同主题/重复曝光 |
| batch 太小 | 负样本不足 | 用 cross-batch memory |
| 只看 AUC | 召回指标失真 | 看 Recall@K / NDCG@K / 覆盖率 |

假负例是最常见的问题。假负例是指训练中被标成负样本，但实际上用户可能喜欢的物品。例如用户点击了某款手机，另一个同品牌同价位手机没有被点击，并不代表用户不喜欢它。如果模型把这种商品强行推远，召回结果会变窄。

batch 太小也会影响效果。in-batch negative 是把同一个 batch 里其他用户的正样本当作当前用户的负样本。这个方法简单高效，但 batch 小时负样本数量不足，模型学到的区分边界会变弱。

评估也容易出错。AUC 衡量的是整体排序区分能力，但召回模型更关心 TopK 中是否覆盖用户可能点击的物品。因此要看 `Recall@K`、`NDCG@K`、覆盖率，以及 ANN 环境下的真实线上效果。

`Recall@K` 衡量真实正样本是否出现在前 K 个召回结果里。`NDCG@K` 衡量相关物品是否排在更靠前的位置。覆盖率衡量系统能召回多少不同物品，避免所有流量都集中在少数热门商品上。

真实工程中还要注意离线和在线的一致性。训练时用精确点积，线上用 ANN 近似检索，二者可能存在差距。模型离线指标提升，不等于线上 ANN 召回一定提升。必须在真实索引、真实延迟和真实候选池下验证。

---

## 替代方案与适用边界

Swish 双塔不是唯一方案。它的优势是轻量、稳定、易检索；缺点是用户和物品之间的复杂交叉表达较弱。

新手版本：如果你的目标是快速从百万商品里找候选，双塔是合适的；如果你要精细建模用户和商品之间的复杂交互，精排模型或交叉网络更合适。

| 方案 | 优点 | 缺点 | 适用场景 |
|---|---|---|---|
| Swish 双塔 | 简洁、稳定、检索友好 | 交互表达弱 | 大规模召回 |
| ReLU 双塔 | 便宜、常见 | 梯度截断更明显 | 基线 |
| 交叉网络 | 表达力强 | 训练/推理更重 | 精排或强特征场景 |
| 图召回 | 利用关系结构 | 依赖图质量 | 社交、电商关系链 |
| 协同过滤 / 近邻召回 | 简单、可解释 | 冷启动弱 | 行为数据充足场景 |

交叉网络是显式建模特征组合的模型，例如“用户年龄段 × 商品价格带 × 类目”。它比双塔表达力更强，但很难把全量物品提前编码成固定向量，因此更适合排序或小规模候选集。

图召回利用用户、物品、作者、店铺、标签之间的关系结构。它适合社交、电商、内容平台，但依赖图质量。如果关系图稀疏、噪声大或更新不及时，收益会受限。

协同过滤和近邻召回依赖用户行为相似性或物品共现关系。它们实现简单，常作为召回通道之一，但对新用户、新物品不友好。

Swish 双塔适合的边界很明确：候选量很大，延迟敏感，需要离线物品向量化，需要接 ANN 索引，不追求复杂交叉特征。它不适合直接替代精排，也不适合在强交叉特征决定结果的场景里单独承担最终决策。

合理的工程做法通常不是只选一个召回模型，而是多通道召回：Swish 双塔负责语义向量召回，协同过滤负责行为共现召回，图召回负责关系链扩展，热门召回负责兜底。最后由排序模型统一打分。

---

## 参考资料

| 主题 | 论文/资料 | 对应章节 |
|---|---|---|
| Swish | *Searching for Activation Functions* | 核心结论、核心机制、代码实现 |
| LayerNorm | *Layer Normalization* | 核心机制、工程权衡 |
| 对比学习 | *A Simple Framework for Contrastive Learning of Visual Representations* | 核心机制、代码实现 |
| Mixed Negative Sampling | *Mixed Negative Sampling for Learning Two-tower Neural Networks in Recommendations* | 代码实现、工程权衡 |
| Cross-Batch Negative Sampling | *Cross-Batch Negative Sampling for Training Two-Tower Recommenders* | 代码实现、工程权衡 |

Swish 的定义来自激活函数论文，LayerNorm 来自标准归一化论文，对比学习的损失形式可参考 SimCLR，双塔负采样实践可参考 mixed negative sampling 和 cross-batch negative sampling。

- Swish：https://research.google/pubs/searching-for-activation-functions/
- LayerNorm：https://arxiv.org/abs/1607.06450
- SimCLR：https://research.google/pubs/a-simple-framework-for-contrastive-learning-of-visual-representations/
- Mixed Negative Sampling：https://research.google/pubs/mixed-negative-sampling-for-learning-two-tower-neural-networks-in-recommendations/
- Cross-Batch Negative Sampling：https://arxiv.org/abs/2110.15154
