## 核心结论

对比学习在召回中的核心价值，不是让模型更会分类，而是让表示空间更适合近邻检索。召回是推荐系统的第一阶段，目标是在百万或亿级候选中快速找出几百到几千个可能相关的物品；排序才负责更精细地比较这些候选。

对比学习的基本目标是：同一条数据的两个版本要像，别的数据要不像。术语上，正样本是应该被拉近的一对样本，负样本是应该被推远的样本。比如同一个用户行为序列可以做两种增强：一种随机遮住少量点击商品，另一种截取最近一段行为。它们都来自同一个用户兴趣状态，因此向量应该接近；别的用户序列通常代表不同兴趣，向量应该更远。

| 训练目标 | 学到的表示 | 召回收益 | 常见误区 |
|---|---|---|---|
| 拉近正样本、推远负样本 | 更稳定、更可分的用户/物品向量 | 更适合 ANN 近邻检索和双塔召回 | 以为它直接提升最终排序精度 |
| 让增强视图保持一致 | 对噪声和缺失更鲁棒的表示 | 用户行为不完整时仍能召回相关物品 | 增强越强越好 |
| 扩大正负样本间隔 | 更清晰的向量空间结构 | 减少无关候选进入召回池 | 把同兴趣样本误当负样本 |

---

## 问题定义与边界

召回任务里的基本对象包括：用户向量、物品向量、候选集合和 ANN 检索。ANN 是近似最近邻检索，白话说就是不用逐个计算所有物品分数，而是在向量索引里快速找出最接近的 top-K 物品。

本文只讨论召回阶段的表示学习，不讨论最终排序。召回关心“能不能从大量物品中先找对一批候选”，排序关心“这批候选内部谁排在前面”。电商场景里，召回阶段可能从一亿商品里找出 500 个候选；精排阶段再用更复杂的特征和模型预测点击率、转化率。先把候选找对，再谈排得准不准。

统一符号如下：

| 符号 | 含义 |
|---|---|
| $x_i$ | 第 $i$ 条原始样本，例如一个用户行为序列 |
| $\tilde x_i^a, \tilde x_i^b$ | 对 $x_i$ 做两次增强得到的两个视图 |
| $f_\theta$ | 编码器，把输入转成隐藏表示 |
| $g_\phi$ | 投影头，把隐藏表示映射到对比学习空间 |
| $z_i$ | 最终向量，$z_i=g_\phi(f_\theta(\tilde x_i))$ |
| $sim(a,b)$ | 相似度函数，常用余弦相似度 |

这里的“视图”是同一信息的不同观察方式。对用户序列来说，视图可以是裁剪、遮盖、重排后的序列；对商品图来说，视图可以是边 dropout 或节点 dropout 后的图结构。

---

## 核心机制与推导

InfoNCE 是对比学习最常用的损失之一。logits 是送进 softmax 的原始分数；softmax 是把多个分数转成概率的函数；loss 是训练时要最小化的错误程度。新手可以先理解为：更像的样本应该得到更高分，更不像的样本应该得到更低分。

InfoNCE 的形式是：

$$
\mathcal L_i
=
-\log
\frac{\exp(\mathrm{sim}(z_i,z_{i^+})/\tau)}
{\exp(\mathrm{sim}(z_i,z_{i^+})/\tau)+\sum_{j\in \mathcal N(i)} \exp(\mathrm{sim}(z_i,z_j)/\tau)}
$$

其中 $z_{i^+}$ 是正样本向量，$\mathcal N(i)$ 是负样本集合，$\tau$ 是温度系数。温度系数控制分布尖锐程度：$\tau$ 越小，相似度差异会被放大，模型对难样本更敏感，但训练也更容易不稳定。

玩具例子：anchor 与正样本、两个负样本的余弦相似度分别是 $0.9, 0.2, -0.1$，取 $\tau=0.5$。logits 是 $1.8, 0.4, -0.2$，softmax 后正样本概率约为 $0.724$，loss 约为 $0.323$。如果正样本相似度升到 $1.1$，负样本降到 $0.0, -0.3$，正样本概率会上升，loss 会下降。

```python
import math

def softmax(xs):
    m = max(xs)
    exps = [math.exp(x - m) for x in xs]
    s = sum(exps)
    return [x / s for x in exps]

def info_nce_loss(pos_sim, neg_sims, tau):
    logits = [pos_sim / tau] + [x / tau for x in neg_sims]
    p_pos = softmax(logits)[0]
    return -math.log(p_pos), p_pos

loss1, p1 = info_nce_loss(0.9, [0.2, -0.1], 0.5)
loss2, p2 = info_nce_loss(1.1, [0.0, -0.3], 0.5)

assert round(p1, 3) == 0.724
assert round(loss1, 3) == 0.323
assert p2 > p1
assert loss2 < loss1
```

MoCo 的关键是“动量编码器 + 队列”。动量编码器是一个慢更新的编码器，用来生成更稳定的 key 向量；队列是保存历史负样本向量的缓存。它的更新公式是：

$$
\theta_k \leftarrow m\theta_k + (1-m)\theta_q
$$

其中 $\theta_q$ 是当前查询编码器参数，$\theta_k$ 是 key 编码器参数，$m$ 是接近 1 的动量系数。MoCo 适合大规模召回，因为 batch 内负样本数量有限，而队列可以提供更多负样本。

CLIP 把图像和文本配对作为正样本，把同 batch 内其他图文组合作为负样本。这个思想可以迁移到多模态召回，例如商品图片和商品标题对齐、视频封面和搜索词对齐、用户文本查询和内容向量对齐。

| 方法 | 正样本来源 | 负样本来源 | 适用数据形态 |
|---|---|---|---|
| SimCLR | 同一样本的两种增强视图 | 同 batch 其他样本 | 图像、序列、结构化特征 |
| MoCo | query encoder 与 key encoder 的同源视图 | 队列中的历史 key | 大 batch 成本高、负样本需求大的场景 |
| CLIP | 配对的图像-文本或跨模态样本 | batch 内未配对样本 | 图文、多模态搜索、多模态推荐 |

真实工程例子：电商召回中，可以对用户点击序列做 crop、mask、reorder 增强。两个增强视图仍应表达同一段兴趣，比如“最近关注跑鞋”。模型通过对比学习预训练后，用户塔输出用户向量，物品塔输出物品向量，再用 ANN 从商品库中召回 top-K。

---

## 代码实现

最小训练流程包括四步：输入视图、编码、计算相似度、计算损失。下面是 PyTorch 风格伪代码，展示 batch 内正负样本构造。

```python
import torch
import torch.nn.functional as F

def info_nce_batch_loss(view_a, view_b, encoder, projector, tau=0.2):
    h_a = encoder(view_a)
    h_b = encoder(view_b)

    z_a = F.normalize(projector(h_a), dim=1)
    z_b = F.normalize(projector(h_b), dim=1)

    logits = z_a @ z_b.T / tau
    labels = torch.arange(z_a.size(0), device=z_a.device)

    loss_ab = F.cross_entropy(logits, labels)
    loss_ba = F.cross_entropy(logits.T, labels)
    return (loss_ab + loss_ba) / 2
```

这里 `logits[i][i]` 是第 $i$ 个样本两种视图的正样本分数，`logits[i][j]` 是第 $i$ 个样本与其他样本的负样本分数。

离线召回落地流程如下：

```python
def offline_build_index(user_tower, item_tower, users, items, ann_index):
    item_vectors = []
    for item in items:
        vec = item_tower(item)
        item_vectors.append(normalize(vec))

    ann_index.build(ids=[item.id for item in items], vectors=item_vectors)

    user_vectors = {}
    for user in users:
        user_vectors[user.id] = normalize(user_tower(user))

    return user_vectors, ann_index

def online_recall(user_id, user_vectors, ann_index, top_k=500):
    query = user_vectors[user_id]
    item_ids = ann_index.search(query, top_k=top_k)
    return item_ids
```

| 对象 | 输入 | 输出 | 张量形状 | 阶段 |
|---|---|---|---|---|
| 增强函数 | 原始序列 $x_i$ | 两个视图 $\tilde x_i^a,\tilde x_i^b$ | `[B, L]` | 训练 |
| 编码器 | 增强视图 | 隐藏表示 | `[B, H]` | 训练/推理 |
| 投影头 | 隐藏表示 | 对比向量 | `[B, D]` | 训练 |
| 相似度矩阵 | 两组向量 | batch 内两两相似度 | `[B, B]` | 训练 |
| 用户塔 | 用户特征/序列 | 用户向量 | `[D]` | 推理 |
| 物品塔 | 物品特征 | 物品向量 | `[D]` | 推理 |
| ANN 索引 | 物品向量库 | top-K 物品 id | `[K]` | 在线召回 |

---

## 工程权衡与常见坑

对比学习的工程难点在于多个参数互相耦合。增强强度、负样本质量、队列长度、温度 $\tau$、batch size 不能孤立调整。batch 太小会导致负样本不足；队列太长会引入陈旧向量；$\tau$ 太小会让模型过度关注少数难样本。

负样本污染是推荐召回里很常见的问题。比如用户正在看一双 Nike 跑鞋，batch 里另一双同品牌同价位跑鞋被当成负样本。模型会被迫把这两个本来应该相近的商品推远，最后召回效果可能变差。白话说，把本来就该相近的样本误当成对立面，会把模型学歪。

| 坑点 | 现象 | 原因 | 规避方法 |
|---|---|---|---|
| 增强过强 | loss 下降慢，召回相关性变差 | 正样本语义被破坏 | 从轻量 mask/crop 开始，观察召回指标 |
| 负样本污染 | 同兴趣物品被推远 | 同类目、同 session、同意图样本被当负样本 | 去同 session、去同类目，或使用软标签 |
| $\tau$ 过小 | 训练震荡，向量分布过尖 | softmax 过度放大相似度差异 | 与 batch size、负样本数量一起调参 |
| 队列过长 | 训练目标滞后 | 历史 key 与当前编码器不一致 | 控制队列长度，提高动量更新稳定性 |
| 队列过短 | 负样本不够，表示区分度弱 | 可比较样本数量不足 | 增大 batch 或队列，加入更难负样本 |
| 线上线下目标不一致 | 离线指标涨，线上点击不涨 | 预训练增强不符合真实查询分布 | 用线上召回日志校验增强策略 |

一个实用原则是：增强策略要贴近真实缺失和噪声。用户序列里的少量 mask 合理，因为线上用户行为本来就不完整；但把长序列随机打乱得过重，就可能破坏真实兴趣演化顺序。

---

## 替代方案与适用边界

对比学习不是唯一方案。它更适合数据有自然视图、能构造正负样本、目标是检索型表示的场景。如果问题本质是找相似项，对比学习通常更顺手；如果问题本质是精确排序，它未必最优。

| 方法 | 优点 | 缺点 | 适用场景 |
|---|---|---|---|
| 监督式双塔 | 目标直接，易上线 | 依赖点击、购买等强监督标签 | 有充足交互日志的召回系统 |
| Matrix Factorization | 简单稳定，成本低 | 表达能力有限，冷启动弱 | 用户-物品交互矩阵较稳定 |
| 纯自回归序列建模 | 擅长预测下一个行为 | 向量检索目标不一定最优 | 下一物品预测、强时序推荐 |
| 图表示学习 | 能利用用户-物品关系图 | 图构建和采样成本高 | 协同信号强、关系网络明显 |
| 对比学习 | 表征鲁棒，适合预训练和检索 | 正负样本构造要求高 | 有自然增强、多视图或跨模态数据 |

边界条件可以直接按四个问题判断：

| 判断问题 | 适合对比学习的答案 | 不适合时的倾向 |
|---|---|---|
| 是否有稳定增强 | 有，且不改变核心语义 | 增强难定义时优先监督学习 |
| 是否有足够负样本 | 有 batch、队列或可采样负例 | 负样本污染严重时谨慎使用 |
| 是否需要跨模态 | 需要图文、文搜图、搜商品等对齐 | 单模态强监督足够时可简化 |
| 是否存在强监督信号 | 弱监督不足或标签稀疏 | 强标签充足时双塔可能更直接 |

工程上常见的选择是组合使用：先用对比学习做用户/物品表征预训练，再用点击、收藏、购买等监督信号微调双塔。这样可以兼顾表示鲁棒性和线上目标一致性。

---

## 参考资料

| 论文名 | 解决的问题 | 适合阅读的阶段 |
|---|---|---|
| SimCLR | 用数据增强和 batch 内负样本学习视觉表征 | 先理解基础机制 |
| MoCo | 用动量编码器和队列扩大负样本规模 | 再理解大规模负样本 |
| CLIP | 用图文配对做跨模态对齐 | 理解多模态召回 |
| Contrastive Learning for Sequential Recommendation | 将对比学习迁移到序列推荐 | 看推荐场景应用 |
| Self-supervised Graph Learning for Recommendation | 将自监督图学习用于推荐 | 看图推荐应用 |

1. [SimCLR: A Simple Framework for Contrastive Learning of Visual Representations](https://arxiv.org/abs/2002.05709)
2. [MoCo: Momentum Contrast for Unsupervised Visual Representation Learning](https://ieeexplore.ieee.org/document/9157636)
3. [CLIP: Learning Transferable Visual Models From Natural Language Supervision](https://arxiv.org/abs/2103.00020)
4. [Contrastive Learning for Sequential Recommendation](https://doi.org/10.1109/ICDE53745.2022.00099)
5. [Self-supervised Graph Learning for Recommendation](https://www.microsoft.com/en-us/research/publication/self-supervised-graph-learning-for-recommendation/)
