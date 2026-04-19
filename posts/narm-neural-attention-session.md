## 核心结论

NARM（Neural Attentive Recommendation Machine，神经注意力推荐机）是一个面向匿名会话的 next-item 推荐模型。next-item 推荐指的是：给定用户当前会话里已经点击过的物品序列，预测下一次最可能点击哪个物品。它的目标不是“理解用户是谁”，而是“从当前 session 推断下一步最可能发生什么”。

NARM 的核心结构是 `GRU + Attention`。GRU（Gated Recurrent Unit，门控循环单元）是一种处理序列的神经网络，用来把点击顺序压缩成隐状态；Attention（注意力机制）是一种加权汇总方法，用来判断历史点击里哪些位置对当前预测更重要。

一个玩具例子：用户在一个会话里先看了“运动鞋首页”，又点了“跑步鞋详情”，中间夹杂一次“促销页浏览”。NARM 不会把最后一次点击直接当答案，也不会把整段历史平均处理，而是会更关注“跑步鞋详情”这类和当前目标更相关的点击，再结合整体顺序判断下一件更可能是“同系列鞋款”。

NARM 把会话表示拆成两部分：

| 表示 | 公式 | 保留的信息 | 作用 |
|---|---:|---|---|
| global | $c_g=h_t$ | 整体点击顺序压缩后的最终状态 | 判断 session 的整体演化方向 |
| local | $c_l=\sum_j \alpha_{tj}h_j$ | 被 attention 选中的关键历史点击 | 捕捉当前局部意图 |
| combined | $c_t=[c_g;c_l]$ | 全局顺序 + 当前意图 | 用于候选物品打分 |

完整流程可以写成：

```text
点击序列 -> GRU -> Attention -> 拼接表示 -> 候选打分
```

候选物品 $i$ 的打分通常写作：

$$
s_i=e_i^\top Bc_t
$$

其中 $e_i$ 是候选物品的向量，$B$ 是可学习参数矩阵，$c_t$ 是当前会话的混合意图表示。分数越高，表示模型认为这个物品越可能是下一次点击。

---

## 问题定义与边界

NARM 解决的是 session-based recommendation，也就是基于当前会话的推荐。session 是一段连续行为序列，例如一次打开电商 App 后连续点击的商品、类目页、搜索结果页。这个问题只使用当前会话内的点击序列，不依赖用户长期画像、历史购买记录或跨天偏好。

形式化地说，给定当前会话前缀：

$$
x_{1:t}=(x_1,x_2,\dots,x_t)
$$

模型要预测下一项：

$$
p(i\mid x_{1:t})
$$

也就是在所有候选物品 $i$ 中，哪个最可能成为 $x_{t+1}$。

训练样本必须按 session 前缀展开，而不是把整条 session 直接当成一个单标签分类任务。假设一条 session 是：

```text
A -> B -> C -> D
```

训练时应该拆成：

```text
([A] -> B)
([A, B] -> C)
([A, B, C] -> D)
```

而不是错误地构造成：

```text
([A, B, C, D] -> D)
```

后者把答案也放进了输入，属于数据泄漏。数据泄漏指训练时模型看到了预测时不应该看到的信息，会导致离线指标虚高、线上效果变差。

| 项目 | NARM 中的含义 |
|---|---|
| 输入 | 当前 session 的物品 ID 前缀，例如 `[x1, x2, x3]` |
| 输出 | 下一次点击的物品 ID，例如 `x4` |
| 不使用的信息 | 用户 ID、长期画像、跨 session 历史、人口属性 |
| 适用场景 | 匿名访问、电商短会话、新闻点击流、短视频连续浏览 |

边界也很明确：NARM 不擅长依赖长期用户身份、跨会话偏好迁移、强冷启动用户画像建模。冷启动指系统缺少足够历史行为时的推荐问题。如果任务是“这个用户下个月会不会复购某品牌”，只看当前 session 通常不够；这类问题更适合带用户特征、长期行为特征和上下文特征的推荐模型。

---

## 核心机制与推导

NARM 先把物品 ID 映射成向量，再交给 GRU。Embedding（嵌入）是把离散 ID 转成连续向量的表示方法，使模型可以学习物品之间的相似性。设输入序列为 $x_1,\dots,x_t$，GRU 输出每一步的隐状态：

$$
h_1,\dots,h_t=GRU(x_1,\dots,x_t)
$$

其中 $h_j$ 可以理解为“处理到第 $j$ 个点击时，模型记住的序列状态”。

全局编码器直接取最后一个隐状态：

$$
c_g=h_t
$$

它保留整段点击序列被 GRU 压缩后的结果。局部编码器对所有历史隐状态做 attention pooling。pooling 是把多个向量汇总成一个向量的操作；attention pooling 则是带权重的汇总：

$$
\alpha_{tj}=softmax(q(h_t,h_j))
$$

$$
c_l=\sum_{j=1}^{t}\alpha_{tj}h_j
$$

这里 $q(h_t,h_j)$ 是一个相关性打分函数，用来衡量当前位置 $j$ 的历史状态与当前最终状态 $h_t$ 有多相关。$\alpha_{tj}$ 是归一化后的注意力权重，所有位置的权重之和为 1。

最后拼接 global 和 local：

$$
c_t=[c_g;c_l]
$$

候选物品打分为：

$$
s_i=e_i^\top Bc_t
$$

再通过 softmax 得到概率：

$$
p(i\mid x_{1:t})=\frac{\exp(s_i)}{\sum_k\exp(s_k)}
$$

| 模块 | 职责 | 输出 |
|---|---|---|
| global | 用 GRU 最后状态压缩整体顺序 | $c_g$ |
| local | 用 attention 从历史点击中选择关键位置 | $c_l$ |
| attention | 计算每个历史点击对当前预测的重要性 | $\alpha_{tj}$ |
| candidate scoring | 计算每个候选物品与会话表示的匹配分 | $s_i$ |

新手例子：如果最近点了“手机壳”和“屏幕膜”，但前面还有一次“新手机页面”，那么 GRU 会记住整段顺序，Attention 会更偏向“新手机页面”和“手机壳”这类语义一致的点击，而不是平均看待所有历史点击。

用一个 3 步序列做数值例子。假设隐状态只有 1 维：

$$
h_1=0.2,\quad h_2=0.5,\quad h_3=0.9
$$

注意力权重已经归一化为：

$$
\alpha=[0.1,0.3,0.6]
$$

则：

$$
c_g=h_3=0.9
$$

$$
c_l=0.1\times0.2+0.3\times0.5+0.6\times0.9=0.68
$$

$$
c_t=[0.9;0.68]
$$

如果 $B=I$，候选 A 的向量是 $e_A=[1.0,0.8]$，候选 B 的向量是 $e_B=[0.4,1.0]$，则：

$$
s_A=1.0\times0.9+0.8\times0.68=1.444
$$

$$
s_B=0.4\times0.9+1.0\times0.68=1.04
$$

所以 A 排在 B 前面。

需要注意：论文公式和工程实现里的 attention 归一化方式可能不同。有的实现会显式 softmax，有的实现可能把注意力函数输出直接参与加权，或者在 mask 后再归一化。复现实验时要对齐论文、官方代码和所用框架的实现细节。

---

## 代码实现

实现 NARM 的核心流程是：

```text
Embedding -> GRUEncoder -> AttentionPooling -> concat -> ScoringLayer -> softmax / ranking loss
```

一个最小训练 batch 通常包含：

| 字段 | 形状 | 作用 |
|---|---:|---|
| `item_seq` | `B x T` | padding 后的物品 ID 序列 |
| `seq_len` | `B` | 每条序列的真实长度 |
| `target_item` | `B` | 下一次点击的真实物品 ID |
| `padding_mask` | `B x T` | 标记哪些位置是真实点击，哪些是 padding |
| `embedding` | `B x T x D` | 物品 ID 转成的向量 |
| `gru_output` | `B x T x H` | GRU 每一步输出的隐状态 |
| `session_vec` | `B x 2H` | 拼接后的会话表示 |

训练目标可以用全量 softmax + cross entropy：

$$
score \rightarrow softmax \rightarrow cross\ entropy
$$

也可以在大词表下使用 BPR（Bayesian Personalized Ranking，贝叶斯个性化排序损失）或 sampled softmax。BPR 是一种排序损失，目标是让正样本分数高于负样本分数。

下面是一个可运行的 Python 玩具实现，演示 attention pooling、拼接和候选打分。它不是完整训练代码，但覆盖了 NARM 的核心计算形状。

```python
import numpy as np

def softmax(x, mask=None):
    x = np.asarray(x, dtype=np.float64)
    if mask is not None:
        x = np.where(mask, x, -1e9)
    x = x - np.max(x)
    exp_x = np.exp(x)
    return exp_x / exp_x.sum()

# 假设这是 GRU 输出的 3 个历史隐状态，每个隐状态 2 维
H = np.array([
    [0.2, 0.1],
    [0.5, 0.4],
    [0.9, 0.7],
])

h_t = H[-1]                    # global: c_g = h_t
raw_attn = H @ h_t             # toy q(h_t, h_j): 点积相关性
alpha = softmax(raw_attn)
c_l = alpha @ H                # local: sum alpha_j * h_j
c_g = h_t
c_t = np.concatenate([c_g, c_l])

# 两个候选物品的向量，维度与 c_t 对齐；这里等价于 B=I
e_A = np.array([1.0, 0.8, 0.9, 0.7])
e_B = np.array([0.4, 0.3, 1.0, 0.9])

s_A = float(e_A @ c_t)
s_B = float(e_B @ c_t)

assert alpha.shape == (3,)
assert np.isclose(alpha.sum(), 1.0)
assert c_t.shape == (4,)
assert s_A > s_B
```

真实工程例子：电商匿名会话重排。系统先用召回模型从几十万商品中取出 100 到 1000 个候选，再把用户当前 session 的最近点击输入 NARM。NARM 输出每个候选商品的排序分，用于判断下一次更可能点击“同品牌跑鞋”“同系列鞋款”还是“促销页推荐商品”。

PyTorch 风格的模块通常拆成：

```python
# 伪代码
item_emb = Embedding(item_seq)                  # B x T x D
gru_out, last_hidden = GRUEncoder(item_emb)     # B x T x H, B x H
local_vec = AttentionPooling(gru_out, last_hidden, padding_mask)
global_vec = last_hidden
session_vec = concat(global_vec, local_vec)     # B x 2H
scores = ScoringLayer(session_vec, candidate_items)
loss = CrossEntropyLoss(scores, target_item)
```

这里最容易出错的不是 GRU 或 attention 公式本身，而是训练数据、padding mask、候选采样和 loss 形式。尤其是 padding 位置不能参与 attention，否则模型会把空白位置当成真实点击。

---

## 工程权衡与常见坑

NARM 在短 session、噪声不太强、当前意图比较集中的场景里很有效。但工程实现中，数据处理错误会比模型结构选择更致命。

| 问题 | 后果 | 处理方式 |
|---|---|---|
| 没有做前缀展开 | 模型训练目标错误，甚至出现数据泄漏 | 使用 `([x1]->x2), ([x1,x2]->x3)` 形式构造样本 |
| padding mask 未生效 | attention 分到 padding 位置 | softmax 前把 padding 位置设为极小值 |
| 长 session 不截断 | 噪声点击稀释 attention | 固定最大长度或按时间窗口裁剪 |
| 训练和线上截断策略不同 | 离线线上分布不一致 | 统一最近 N 次点击或统一时间窗口 |
| 全量 softmax 过大 | 训练和推理成本高 | 使用候选采样、sampled softmax 或召回后重排 |
| 只看最后点击 | 忽略前面关键意图 | 保留 attention 分支 |
| 只看 attention | 丢失整体顺序模式 | 保留 global 分支 |

简短流程应保持一致：

```text
raw session -> truncate -> pad -> mask -> model
```

代码注意点清单：

```text
- 前缀展开训练样本
- padding mask 必须生效
- attention 归一化要核对
- 长序列截断策略要一致
- target_item 不能出现在输入前缀之后
- 训练候选采样方式要和评估指标匹配
```

新手例子：如果用户在同一 session 里打开了很多无关页面，例如先看运动鞋，又点了客服页、优惠券页、物流说明页，再回到跑步鞋详情，attention 可能会被稀释。此时可以只保留最近 N 次点击，或者保留最近一段时间内的点击，减少噪声。

真实工程中，商品数十万级时，NARM 通常不直接对全量商品逐个打分。更常见的架构是：

```text
用户当前 session -> 召回 100~1000 个候选 -> NARM 重排 -> 业务规则融合 -> 展示
```

这样做的原因很直接：NARM 的会话建模能力适合排序，但全量候选打分成本高。召回层负责缩小搜索空间，NARM 负责在较小候选集里精排。

---

## 替代方案与适用边界

NARM 不是 session 推荐的唯一解。它的优势是结构清晰、双分支直观、attention 权重有一定解释性；限制是对长程依赖、复杂图结构和跨会话偏好建模并不充分。

| 模型 | 是否依赖用户 ID | 是否强调局部意图 | 是否擅长长序列 | 计算成本 | 可解释性 |
|---|---|---|---|---|---|
| GRU4Rec | 否 | 弱 | 中 | 低到中 | 中 |
| NARM | 否 | 强 | 中 | 中 | 较强 |
| SR-GNN | 否 | 中到强 | 中 | 中到高 | 中 |
| Transformer-based session models | 否或可选 | 强 | 较强 | 高 | 中 |
| 带用户画像的推荐模型 | 通常是 | 取决于特征设计 | 强，依赖历史特征 | 中到高 | 取决于模型 |

GRU4Rec 是较早的循环神经网络会话推荐模型，重点是用 GRU 建模序列状态。NARM 在此基础上增加 attention，更明确地区分整体状态和当前意图。SR-GNN 把 session 看成图，适合点击之间存在复杂转移关系的场景。Transformer-based session models 用自注意力处理序列，适合更长序列和复杂位置关系，但成本更高。带用户画像的推荐模型适合长期偏好建模，例如复购、会员推荐、跨天兴趣迁移。

新手例子：如果用户行为更像“按固定顺序浏览一组商品”，例如先看套餐页，再看配置页，再看支付页，那么只靠 attention 关注重点点击可能不够，顺序建模更强的模型可能更合适。如果用户行为是短会话里夹杂少量噪声，但主意图明显，例如围绕“跑步鞋”连续点击，NARM 更合适。

结论型选择标准：匿名会话短、噪声不大、主要目标是下一步点击排序时，选 NARM；如果要做跨天复购、长期偏好建模、用户画像驱动推荐，NARM 不应作为主力模型。

---

## 参考资料

| 来源 | 用途 | 可信度 |
|---|---|---|
| 原论文 | 确认模型定义、公式和实验设置 | 高 |
| DOI 页面 | 确认论文发表信息 | 高 |
| 官方代码 | 对齐实现细节，尤其是 attention 与训练流程 | 高 |
| RecBole 文档 | 查看框架化实现和输入输出约定 | 中到高 |

读论文时建议先看模型结构和公式，再看代码实现，最后看框架文档对输入输出格式的说明。正文中提到的“论文公式和实现可能不同”，主要来自论文描述、官方代码实现以及框架复现之间的细节差异，尤其是 attention 归一化、mask 处理和 loss 设置。

1. [Neural Attentive Session-based Recommendation PDF](https://renzhaochun.github.io/assets/pdf/1711.04725.pdf)
2. [Neural Attentive Session-based Recommendation DOI](https://doi.org/10.1145/3132847.3132926)
3. [NARM Official Code](https://github.com/lijingsdu/sessionRec_NARM)
4. [RecBole NARM Documentation](https://www.recbole.io/docs/user_guide/model/sequential/narm.html)
