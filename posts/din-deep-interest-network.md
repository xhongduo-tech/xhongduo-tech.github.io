## 核心结论

DIN（Deep Interest Network，深度兴趣网络）是一种面向 CTR 预估的候选感知注意力模型。CTR 预估指预测用户点击某个候选物品的概率，例如广告点击率或商品点击率。候选感知的意思是：模型在表示用户兴趣时，会先看当前候选物品是什么，再决定用户历史行为里哪些更重要。

DIN 的核心不是把用户历史行为压缩成一个固定用户向量，而是根据当前候选物品动态选择相关历史。用户历史中可能同时有“跑鞋”“跑步手表”“书籍”“耳机”。当候选物品是“跑鞋”时，运动相关历史更重要；当候选物品是“小说”时，阅读相关历史更重要。同一个用户，不同候选物品，对应的兴趣表示应该不同。

DIN 用 Local Activation Unit 完成这件事。Local Activation Unit 可以理解为一个逐条打分模块：它把每条历史行为嵌入 \(e_i\) 和候选物品嵌入 \(q\) 放在一起比较，得到相关性分数 \(s_i\)，再用注意力权重聚合历史行为。

核心公式如下：

$$
s_i = \mathrm{MLP}([e_i, q, e_i \odot q, e_i - q])
$$

$$
\alpha_i = \frac{\exp(s_i)}{\sum_{j=1}^{T}\exp(s_j)}
$$

$$
v = \sum_{i=1}^{T}\alpha_i e_i
$$

$$
\hat y = \sigma(\mathrm{MLP}([v, q, \text{其他特征}]))
$$

其中 MLP 是多层感知机，即由若干线性层和非线性激活函数组成的前馈神经网络；\(\sigma\) 是 sigmoid 函数，用来把输出压到 \(0\) 到 \(1\) 之间，表示点击概率。

| 方法 | 用户表示是否固定 | 是否看候选物品 | 对多兴趣的表达能力 |
|---|---:|---:|---:|
| 平均池化 | 是 | 否 | 弱 |
| DIN | 否 | 是 | 强 |

---

## 问题定义与边界

DIN 要解决的问题是：当用户有多条历史行为时，哪些历史和当前候选物品相关。它不是简单判断用户有没有历史，也不是把所有历史都平均看待。

传统 Embedding + MLP 排序模型常见做法是：先把用户点击过的物品 embedding 做 sum pooling 或 average pooling，再把得到的用户向量送入 MLP。embedding 是离散 ID 的稠密向量表示，例如把商品 ID 映射成一个可训练的 64 维向量。pooling 是把多个向量合并成一个向量的操作，例如平均或求和。

这种方式的问题是用户表示固定。一个用户看“跑鞋”和看“小说”时，模型拿到的是同一个历史聚合向量。如果用户历史里既有运动装备，也有图书，这个固定向量会混合多个兴趣，导致候选物品相关性被削弱。

玩具例子：用户历史是“运动鞋、跑步手表、书籍”。候选广告是“跑鞋”时，“运动鞋、跑步手表”更重要；候选广告是“小说”时，“书籍”更重要。DIN 关注的就是这种目标相关性。

真实工程例子：电商广告排序系统通常先由召回层拿到几百到几千个候选广告，再由排序模型逐个预测点击率。排序阶段候选物品已经确定，因此模型可以对每个候选广告分别计算用户兴趣。DIN 正适合这个阶段：同一个用户面对不同广告时，注意力权重不同，最终 CTR 也不同。

| 对比项 | 传统池化模型 | DIN |
|---|---|---|
| 是否依赖候选物品 | 不依赖 | 依赖 |
| 是否区分历史重要性 | 通常不区分 | 按候选物品动态区分 |
| 是否支持多兴趣 | 表达有限 | 更适合多兴趣 |
| 主要目标 | 得到固定用户向量 | 得到候选感知兴趣向量 |

DIN 适用于 CTR 预估、推荐排序、广告点击率预测等候选物品已知的任务。不适用于纯生成任务、无候选物品的用户画像建模，也不专门解决超长序列中的时间依赖问题。

---

## 核心机制与推导

设用户历史行为序列长度为 \(T\)，每条历史行为的 embedding 为 \(e_1,\dots,e_T\)，候选物品 embedding 为 \(q\)。DIN 的计算流程是：

```text
历史行为序列
    ↓
逐条与候选物品做交互
    ↓
Local Activation Unit 打分
    ↓
softmax 得到注意力权重
    ↓
加权汇聚得到兴趣向量
    ↓
拼接候选物品和其他特征
    ↓
输出 CTR
```

第一步，定义历史行为和候选物品：

$$
E = [e_1, e_2, \dots, e_T], \quad q \in \mathbb{R}^{D}
$$

其中 \(D\) 是 embedding 维度。

第二步，Local Activation Unit 对每条历史行为打分。它不只看 \(e_i\) 和 \(q\) 本身，还显式加入乘积项和差值项：

$$
s_i = \mathrm{MLP}([e_i, q, e_i \odot q, e_i - q])
$$

\(\odot\) 表示逐元素相乘。乘积项可以表达两个向量在各维度上的匹配程度，差值项可以表达距离或偏移。拼接后的向量再交给 MLP 学习非线性相关性。

第三步，用 softmax 把分数转成权重。softmax 是把一组实数转成概率分布的函数，所有输出非负且总和为 1：

$$
\alpha_i = \frac{\exp(s_i)}{\sum_{j=1}^{T}\exp(s_j)}
$$

第四步，对历史行为加权求和：

$$
v = \sum_{i=1}^{T}\alpha_i e_i
$$

这里的 \(v\) 就是候选感知兴趣向量。候选物品不同，\(s_i\) 不同，\(\alpha_i\) 不同，最终 \(v\) 也不同。

第五步，把 \(v\)、候选物品 \(q\)、用户特征、上下文特征等拼接后送入 CTR 网络：

$$
\hat y = \sigma(\mathrm{MLP}([v, q, \text{其他特征}]))
$$

符号含义如下：

| 符号 | 含义 |
|---|---|
| \(e_i\) | 第 \(i\) 条历史行为 embedding |
| \(q\) | 候选物品 embedding |
| \(s_i\) | 历史行为与候选物品的相关性分数 |
| \(\alpha_i\) | 注意力权重 |
| \(v\) | 候选感知兴趣向量 |
| \(\hat y\) | 预测点击概率 |

一个最小数值例子：设三条历史行为与候选物品的打分为 \([2,1,0]\)，softmax 后约为 \([0.665,0.245,0.090]\)。若

$$
e_1=[1,0],\quad e_2=[0,1],\quad e_3=[1,1]
$$

则

$$
v \approx 0.665[1,0] + 0.245[0,1] + 0.090[1,1] = [0.755,0.335]
$$

这说明第一条历史贡献最大，第二条和第三条贡献较小。历史行为不是一起平均，而是逐条判断“这条和当前候选物品是否相关”。

---

## 代码实现

实现 DIN 的重点不是写一个泛泛的 Attention 层，而是对历史序列中每个行为与候选物品逐条交互，再做 mask、softmax 和加权聚合。mask 是布尔掩码，用来标记哪些位置是真实历史，哪些位置是 padding。padding 是为了把不同长度序列补齐到同一长度而填充的空位。

简化 PyTorch 伪代码如下：

```python
# history_emb: [B, T, D]
# candidate_emb: [B, D]
# mask: [B, T]

q = candidate_emb.unsqueeze(1).expand(-1, T, -1)
x = torch.cat([history_emb, q, history_emb * q, history_emb - q], dim=-1)
scores = mlp_local_activation(x).squeeze(-1)

scores = scores.masked_fill(~mask, float("-inf"))
alpha = torch.softmax(scores, dim=1)
v = torch.sum(alpha.unsqueeze(-1) * history_emb, dim=1)

logits = ctr_mlp(torch.cat([v, candidate_emb, other_features], dim=-1))
```

张量形状对照：

| 名称 | 形状 | 说明 |
|---|---|---|
| `history_emb` | `[B, T, D]` | 批量历史行为 embedding |
| `candidate_emb` | `[B, D]` | 候选物品 embedding |
| `q` | `[B, T, D]` | 扩展后的候选物品 embedding |
| `x` | `[B, T, 4D]` | 逐条交互特征 |
| `scores` | `[B, T]` | 每条历史的相关性分数 |
| `alpha` | `[B, T]` | 注意力权重 |
| `v` | `[B, D]` | 聚合后的兴趣向量 |

下面是一个只依赖 Python 标准库的可运行玩具实现，用来验证 softmax 加权聚合逻辑：

```python
import math

def softmax(scores):
    m = max(scores)
    exps = [math.exp(x - m) for x in scores]
    total = sum(exps)
    return [x / total for x in exps]

def weighted_sum(weights, vectors):
    dim = len(vectors[0])
    out = [0.0] * dim
    for w, vec in zip(weights, vectors):
        for i in range(dim):
            out[i] += w * vec[i]
    return out

scores = [2.0, 1.0, 0.0]
history = [[1.0, 0.0], [0.0, 1.0], [1.0, 1.0]]

alpha = softmax(scores)
v = weighted_sum(alpha, history)

assert round(alpha[0], 3) == 0.665
assert round(alpha[1], 3) == 0.245
assert round(alpha[2], 3) == 0.090
assert round(v[0], 3) == 0.755
assert round(v[1], 3) == 0.335
```

训练输入通常包含：用户 ID、候选物品 ID、用户历史行为 ID 序列、历史长度、上下文特征、点击标签。线上推理输入通常没有点击标签，但必须保证用户历史、候选物品、字典映射、特征归一化方式和训练时一致。

| 字段 | 训练阶段 | 线上推理阶段 |
|---|---|---|
| 用户 ID | 有 | 有 |
| 候选物品 ID | 有 | 有 |
| 历史行为序列 | 有 | 有 |
| 点击标签 | 有 | 无 |
| 特征字典版本 | 必须固定 | 必须一致 |
| padding mask | 必须生成 | 必须生成 |

---

## 工程权衡与常见坑

DIN 的收益来自候选感知注意力，但工程效果高度依赖数据和特征一致性。模型结构本身不能修复脏数据、错位 ID 或线上线下不一致。

最常见的问题是 padding 不做 mask。若一个用户只有 3 条历史，但系统为了对齐补到 50 条，那么后 47 个位置不是有效行为。如果这些位置参与 softmax，模型会把空位也当成历史行为学习，注意力权重会被污染。

masked softmax 的核心逻辑是先把无效位置改成极小值，再做 softmax：

```python
# scores: [B, T]
# mask: True 表示有效历史，False 表示 padding
scores = scores.masked_fill(~mask, float("-inf"))
alpha = torch.softmax(scores, dim=1)
```

常见坑如下：

| 问题 | 后果 | 处理方式 |
|---|---|---|
| padding 不做 mask | 无效位置参与注意力 | mask 后再 softmax |
| 历史序列过长 | 噪声压过有效兴趣 | 截断最近 \(K\) 条或按 session 切分 |
| 训练和线上字典不一致 | embedding 对不上 | 固定字典版本并做一致性校验 |
| 随机切分验证集 | 高估离线效果 | 使用时间切分验证 |
| 只看离线 AUC | 无法判断线上收益 | 做在线 A/B 实验 |

历史太长时，DIN 不一定越看越准。用户几年前点过的商品可能已经和当前兴趣无关。工程上常用最近 \(K\) 条行为、按行为类型加权、按 session 保留高价值行为等方式控制噪声。

另一个坑是把 DIN 实现成“平均池化 + MLP”。如果先把历史平均成一个用户向量，再和候选物品拼接，候选物品已经无法影响历史内部的权重分配，这就丢掉了 DIN 的关键机制。

真实工程中还要关注特征时效性。例如广告排序系统中，用户历史行为可能来自实时日志，也可能来自离线特征库。如果训练用的是小时级更新特征，线上却用天级延迟特征，模型看到的分布会不同，CTR 预估会偏移。

---

## 替代方案与适用边界

DIN 不是推荐排序的通用最优解。它的优势是候选感知和行为级别局部相关性，弱点是对复杂时间顺序和长程依赖建模不足。

如果只需要一个稳定的用户总画像，平均池化、sum pooling 或简单 MLP 可能足够。它们实现简单、推理快、维护成本低，适合冷启动补充特征或轻量排序场景。

如果关心“这个候选物品对应哪几条历史”，DIN 更合适。它能让同一用户在不同候选物品下生成不同兴趣向量，适合电商广告、信息流推荐、商品排序等场景。

如果更关心“行为顺序如何影响兴趣变化”，DIN 就不够充分。此时可以考虑 DIEN、GRU、Transformer 类推荐模型。GRU 是一种循环神经网络结构，适合按时间更新隐藏状态；Transformer 是基于自注意力的序列建模结构，适合捕捉序列内部多位置关系。

| 方法 | 是否候选感知 | 是否显式建模顺序 | 是否适合长序列 | 实现复杂度 |
|---|---:|---:|---:|---:|
| 平均池化 | 否 | 否 | 一般 | 低 |
| DIN | 是 | 否 | 一般 | 中 |
| 序列模型 | 不一定 | 是 | 中 | 中 |
| Transformer 推荐 | 可设计为是 | 是 | 较强 | 高 |

DIN 强在 CTR 排序和候选相关历史选择，不擅长纯时序预测。它也不是“注意力越多越好”的证明。注意力机制只有在候选物品与历史行为存在明确相关性、且训练数据能支持这种相关性学习时，才会产生稳定收益。

---

## 参考资料

- 原论文 arXiv:1706.06978：https://arxiv.org/abs/1706.06978
- KDD 2018 / dblp 记录，含 DOI 10.1145/3219819.3219823：https://dblp.org/rec/conf/kdd/ZhouZSFZMYJLG18.html
- DeepCTR 的 DIN 文档：https://deepctr-doc.readthedocs.io/en/stable/Features.html
- 作者代码仓库 zhougr1993/DeepInterestNetwork：https://github.com/zhougr1993/DeepInterestNetwork

本文公式与机制以原论文为准，代码实现以开源仓库和工程实践为参考。
