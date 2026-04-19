## 核心结论

长尾物品推荐，是在交互少、曝光稀疏的物品上，尽量学到可泛化表征，并把它们稳定放进候选集和排序结果的任务。

它的目标不是简单把尾部物品排到前面，而是在证据不足时减少模型对热门物品的结构性偏置。一个可用的总目标可以写成：

$$
L = L_{rec} + \lambda L_{cl}
$$

其中，$L_{rec}$ 是推荐任务本身的损失，$L_{cl}$ 是对比学习损失，$\lambda$ 控制两者的权重。对比学习是让同一个用户或物品在不同扰动视图下学到一致表示的方法，常用于缓解稀疏数据下 embedding 不稳定的问题。

真实工程例子是电商首页推荐。新品和小众 SKU 点击少，但并不等于没有价值。如果系统只根据历史点击训练，热门商品会获得更多曝光，更多曝光又带来更多点击，最后形成“越热越热、越冷越冷”的反馈回路。长尾策略要解决的是：让这些低交互物品至少有机会进入候选集，并在排序阶段得到合理评估。

主流方案本质上分三类：

| 方案 | 解决的问题 | 核心代价 |
| --- | --- | --- |
| 重采样 / 重加权 | 尾部样本训练信号弱 | 容易放大噪声 |
| 对比学习 | 稀疏物品表示不稳定 | 可能引入假负样本 |
| 图传播 | 尾部节点信息不足 | 层数过深会过平滑 |

工程上通常不是三选一，而是组合使用：先做流行度校正，再用轻量对比学习增强表示，如果用户-物品图足够密，再引入图传播。

---

## 问题定义与边界

长尾问题的核心不是“尾部物品没价值”，而是“尾部物品缺少可被模型学习的证据”。交互是用户点击、收藏、购买、观看等行为记录。推荐模型主要从交互中学习偏好，所以交互越少，物品 embedding 越容易不稳定。

头部物品是交互次数高的物品，长尾物品是交互次数低但数量巨大的物品。两者在训练中的差异很直接：

| 维度 | 头部物品 | 长尾物品 |
| --- | --- | --- |
| 交互次数 | 高 | 低 |
| 梯度信号 | 强 | 弱 |
| 曝光机会 | 多 | 少 |
| 图传播可见性 | 高 | 低 |
| 过拟合风险 | 较低 | 较高 |
| 主要问题 | 偏置累积 | 数据稀疏 |

这里的梯度信号，是模型参数根据损失函数更新的方向和强度。一个物品被训练样本反复命中，它对应的参数就会被多次更新；一个物品只出现几次，它的参数很难学稳。

优化边界必须明确：长尾推荐不是把所有冷门物品强行推给用户。系统仍然要守住整体 Recall、排序稳定性、转化率和用户体验。Recall@K 是前 K 个推荐结果中命中真实正反馈物品的比例，常用于衡量召回能力。长尾策略的合理目标是提高 LongTailRecall@K、Coverage@K 和新物品进入候选集的概率，同时避免主指标明显下降。

玩具例子：一个班级里有 100 本书，10 本热门书每本被借 100 次，90 本冷门书每本只被借 2 次。只按借阅次数推荐，系统几乎永远推荐那 10 本热门书。但某个学生喜欢冷门技术书，冷门书借阅少不代表不适合他，只代表系统证据少。

---

## 核心机制与推导

推荐模型通常先把用户和物品映射成向量。embedding 是用一组数字表示用户或物品特征的向量。用户 $u$ 和物品 $i$ 的匹配分数可以写成：

$$
s_{ui}=z_u^\top z_i
$$

其中，$z_u$ 是用户向量，$z_i$ 是物品向量，$z_u^\top z_i$ 是点积。点积越大，模型认为用户越可能喜欢该物品。

第一类方法是重加权。设 $p_i$ 是物品 $i$ 的交互次数，$\epsilon$ 是防止除零的小常数，$\alpha$ 是控制尾部放大强度的超参数，则：

$$
L_{rec}=\sum_{(u,i)\in D} w_i\,\ell(s_{ui}, y_{ui}),\qquad
w_i=(p_i+\epsilon)^{-\alpha}
$$

其中，$D$ 是训练集，$y_{ui}$ 是标签，$\ell$ 是基础损失函数，$w_i$ 是物品权重。$p_i$ 越小，$w_i$ 越大，尾部样本在训练中的影响就越大。

数值例子：头部物品 $p_h=1000$，尾部物品 $p_t=10$，取 $\alpha=0.5$，忽略 $\epsilon$：

$$
w_h=1000^{-0.5}\approx 0.0316,\qquad
w_t=10^{-0.5}\approx 0.316
$$

尾部权重大约是头部的 10 倍。如果基础损失都是 0.2，头部加权损失约为 0.0063，尾部加权损失约为 0.0632。这样模型不会只被大量头部样本主导。

第二类方法是对比学习。它给同一个节点构造两个视图，例如轻微 dropout 后的两份图表示，然后让同一节点的两个视图靠近，让不同节点远离：

$$
L_{cl}= -\log \frac{\exp(\mathrm{sim}(z_v^{(1)}, z_v^{(2)})/\tau)}
{\sum_{v'} \exp(\mathrm{sim}(z_v^{(1)}, z_{v'}^{(2)})/\tau)}
$$

其中，$v$ 可以是用户或物品，$z_v^{(1)}$ 和 $z_v^{(2)}$ 是两个视图下的表示，$\mathrm{sim}$ 是相似度函数，$\tau$ 是温度系数。温度系数控制分布的尖锐程度，越小越强调最相似样本。

第三类方法是图传播。用户-物品图是把用户和物品作为节点、交互作为边的图结构。图模型会让节点从邻居聚合信息：

$$
z_v^{(l+1)}=\sum_{x\in\mathcal N(v)} \frac{1}{\sqrt{d_v d_x}}\,z_x^{(l)}
$$

其中，$\mathcal N(v)$ 是节点 $v$ 的邻居集合，$d_v$ 和 $d_x$ 是节点度数，$l$ 是层数。节点度数就是一个节点连接了多少条边。这个公式的含义是：下一层表示来自邻居表示的加权平均。尾部物品交互少，但如果它连接到一些有代表性的用户，图传播可以把高阶偏好信息传过来。

---

## 代码实现

实现顺序应保持简单：数据切分、统计流行度、计算权重、训练模型、评估整体指标和长尾指标。不要先堆复杂模型，再补指标；否则无法判断提升来自哪里。

下面是一个可运行的最小 Python 例子，展示如何按物品流行度生成权重，并把权重乘进二分类损失。这里不用深度学习框架，只保留核心计算逻辑。

```python
import math
from collections import Counter

train_data = [
    ("u1", "head", 1, 0.9),
    ("u2", "head", 1, 0.7),
    ("u3", "head", 1, 0.8),
    ("u4", "tail", 1, 0.6),
]

def sigmoid(x):
    return 1 / (1 + math.exp(-x))

def bce_from_score(score, label):
    p = sigmoid(score)
    eps = 1e-12
    return -(label * math.log(p + eps) + (1 - label) * math.log(1 - p + eps))

def item_popularity(data):
    return Counter(item for _, item, _, _ in data)

def item_weight(pop, item, alpha=0.5, eps=1e-6):
    return (pop[item] + eps) ** (-alpha)

pop = item_popularity(train_data)

head_w = item_weight(pop, "head")
tail_w = item_weight(pop, "tail")

assert pop["head"] == 3
assert pop["tail"] == 1
assert tail_w > head_w

head_loss = head_w * bce_from_score(0.9, 1)
tail_loss = tail_w * bce_from_score(0.6, 1)

assert tail_loss > head_loss
print(round(head_w, 4), round(tail_w, 4), round(head_loss, 4), round(tail_loss, 4))
```

在真实训练代码中，结构通常是：

```python
# 1. 统计物品流行度
item_cnt = count_interactions(train_data)

# 2. 计算尾部权重
weight = {i: (item_cnt[i] + eps) ** (-alpha) for i in items}

# 3. 训练阶段
for batch in loader:
    score = model(user, item)
    rec_loss = weighted_bce(score, label, weight[item])
    cl_loss = contrastive_loss(view1, view2, tau)
    loss = rec_loss + lam * cl_loss
    loss.backward()
    optimizer.step()

# 4. 评估
metrics = {
    "Recall@K": recall_k(pred, truth, k),
    "LongTailRecall@K": tail_recall_k(pred, truth, tail_items, k),
    "Coverage@K": coverage_k(pred, all_items, k),
}
```

工程上要把头尾分桶、权重计算、训练损失和指标评估拆开。这样调 $\alpha$、$\lambda$、$\tau$、图层数时，才能知道是尾部权重生效、对比学习生效，还是图传播生效。

真实工程例子：短视频推荐中新作者作品互动少，若只按历史完播率训练，新作者很难进入召回。系统可以先把低曝光作品分桶，再给低曝光正反馈更高权重，同时用视频文本、封面、作者标签构造内容 embedding，最后用 LongTailRecall@K 和作者覆盖率共同评估。

---

## 工程权衡与常见坑

尾部权重不能无限放大。长尾样本少，噪声比例通常更高。一个偶然点击、误点或低质量内容，如果被过度加权，模型会把噪声当成偏好学习。

常见问题如下：

| 常见坑 | 典型现象 | 规避方式 |
| --- | --- | --- |
| 只看整体指标 | 整体 Recall 上升，尾部没改善 | 增加 tail Recall、coverage、novelty |
| 重采样过强 | 长尾物品被异常抬高 | 幂律权重加 clip |
| 对比增广过猛 | 表示语义失真 | 使用轻量 dropout，过滤假负样本 |
| 图层过深 | 物品表示趋同 | 控制层数，限制邻域扩散 |
| 只做随机切分 | 上线后效果回落 | 按时间切分验证 |

Coverage@K 是前 K 个推荐结果覆盖了多少不同物品，用来衡量推荐结果是否过度集中。Novelty 是新颖性，通常用来衡量推荐结果是否总是热门物品。

一个重要现象是：整体 Recall@K 提升不代表长尾策略有效。模型可能只是更准确地推荐热门物品，导致整体指标变好，但尾部 Recall@K 不涨，Coverage@K 下降。这说明系统更偏头部了。

对比学习也要谨慎。负样本是训练中被当作“不相关”的样本。假负样本是实际可能相关、但被错误当作负样本的样本。在推荐系统中，用户没有点击某个长尾物品，不代表不喜欢它，可能只是没见过。对比学习如果把这些物品强行推远，会伤害长尾泛化。

图传播的坑是过平滑。过平滑是多层聚合后，不同节点表示越来越相似，模型难以区分具体偏好。长尾物品本来信息少，如果被热门邻居反复覆盖，反而会失去自身特征。

---

## 替代方案与适用边界

长尾策略要按业务条件选，不应默认上复杂图模型。

| 场景 | 更适合的方案 | 不太适合的方案 |
| --- | --- | --- |
| 轻度头部偏置 | 重加权 + 轻量对比学习 | 深层图传播 |
| 极稀疏冷启动 | 内容特征 + 规则召回 | 纯协同过滤 |
| 图结构较强 | LightGCN / SGL / XSimGCL | 仅靠采样修正 |
| 强稳定性要求 | 小幅校正 + 严格监控 | 激进尾部放大 |

协同过滤是只根据用户-物品交互关系学习推荐的方法。它依赖历史行为，所以在极稀疏冷启动下天然受限。冷启动新品几乎没有交互，单靠长尾重加权没有足够样本可放大，通常还要引入内容特征，例如标题、类目、文本描述、图片向量、价格区间和作者信息。

如果业务更看重排序稳定性，例如金融产品、招聘推荐、B2B 采购推荐，尾部策略要保守。可以先在召回阶段提高尾部候选覆盖，再让排序阶段结合质量分、合规规则和用户意图过滤，而不是直接在最终排序里大幅抬高尾部。

如果数据量较小，优先做三件事：第一，按时间切分验证集；第二，建立长尾指标；第三，尝试带 clip 的流行度重加权。只有当用户-物品图足够丰富，并且基础指标已经稳定，再考虑图对比学习。

---

## 参考资料

| 资料 | 支撑内容 | 建议放置位置 |
| --- | --- | --- |
| SGL | 图自监督推荐机制 | 核心机制与推导 |
| XSimGCL | 简化图对比学习 | 核心机制与推导 |
| SELFRec | 推荐系统自监督实现参考 | 代码实现 |
| Popularity bias survey | 流行度偏置背景 | 问题定义与边界 |
| Popularity-weighted negative sampling | 采样和校正方法 | 工程权衡与常见坑 |

1. [Self-supervised Graph Learning for Recommendation](https://www.microsoft.com/en-us/research/publication/self-supervised-graph-learning-for-recommendation/)
2. [XSimGCL: Towards Extremely Simple Graph Contrastive Learning for Recommendation](https://research-repository.griffith.edu.au/items/9af19dff-132f-49f2-9242-520c2adb193f)
3. [SELFRec 开源实现](https://github.com/Coder-Yu/SELFRec)
4. [A survey on popularity bias in recommender systems](https://link.springer.com/article/10.1007/s11257-024-09406-0)
5. [Learning-to-rank debias with popularity-weighted negative sampling and popularity regularization](https://www.sciencedirect.com/science/article/abs/pii/S0925231224004521)
