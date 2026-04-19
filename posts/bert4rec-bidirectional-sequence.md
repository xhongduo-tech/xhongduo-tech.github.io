## 核心结论

BERT4Rec 是把用户历史行为序列当作 token 序列，用双向 Transformer 做 `Masked Item Prediction` 的序列推荐模型。

一句话定义：

`BERT4Rec = 双向 Transformer + 随机 Mask + 物品分类预测`

这里的 token 可以理解为“离散符号”，在推荐系统里通常就是 item id，例如商品 id、视频 id、文章 id。Masked Item Prediction 是“遮住一部分物品，让模型根据上下文把它猜回来”的训练任务。

新手版玩具例子：

用户序列是 `[A, B, C, D]`。训练时把 `B` 遮住，变成 `[A, MASK, C, D]`。BERT4Rec 会同时看左边的 `A` 和右边的 `C、D` 来猜 `B`。这和只看 `A` 来预测 `B` 不同。

核心区别是：

| 模型 | 建模方式 | 训练目标 | 是否看右侧上下文 | 更适合 |
|---|---|---|---|---|
| 传统 next-item 推荐 | 根据历史预测下一项 | $P(x_{t+1}\mid x_1,\dots,x_t)$ | 否 | 简单下一步预测 |
| SASRec | 单向 Self-Attention | 根据左侧历史预测下一项 | 否 | next-item 推荐、实时序列建模 |
| BERT4Rec | 双向 Transformer | 恢复被 mask 的 item | 是 | 序列补全、意图建模、重排 |

BERT4Rec 的输入表示是 item embedding 和 position embedding 相加：

$$
h_i^0 = e(x_i) + p_i
$$

其中，embedding 是“把离散 id 映射成向量”的方法；position embedding 是“告诉模型当前位置是第几个”的向量。

经过多层双向 Transformer 后，在被 mask 的位置做物品分类：

$$
P(x_i=v\mid X')=\mathrm{softmax}(W_o\,\mathrm{GELU}(W_p h_i^L+b_p)+b_o)_v
$$

这里的 softmax 是“把一组分数变成概率分布”的函数，$v$ 表示某个候选物品。

---

## 问题定义与边界

BERT4Rec 解决的问题是：给定一个用户按时间排序的历史行为序列，学习物品之间的上下文依赖关系，并预测被遮住的位置原本是什么物品。

设用户序列为：

$$
X=(x_1,x_2,\dots,x_n)
$$

其中 $x_i$ 是第 $i$ 个交互物品。随机选择一些位置组成 mask 集合 $M$，把这些位置替换成 `[MASK]`，得到训练输入 $X'$。模型的目标是恢复这些被遮住的真实 item。

| 符号 | 含义 | 新手解释 |
|---|---|---|
| $X=(x_1,\dots,x_n)$ | 用户历史序列 | 用户按时间点击、浏览、购买过的物品 |
| $M$ | 随机遮罩位置集合 | 哪些位置被盖住 |
| $X'$ | mask 后的训练输入 | 模型实际看到的序列 |
| $x_i^*$ | 第 $i$ 位真实 item | 被盖住位置的正确答案 |
| 目标 | 恢复被 mask 的 item | 做填空题 |

电商场景里的真实工程例子：

用户最近依次浏览了“手机”“手机壳”“无线耳机”“充电器”。BERT4Rec 要学的不是简单统计“手机壳之后最常见是什么”，而是学习这些物品共同构成的购买意图。例如，“手机”和“手机壳”一起出现时，可能强化“手机配件”意图；“无线耳机”和“充电器”又进一步说明用户可能处在新机配件采购阶段。

但 BERT4Rec 不是完整推荐系统，也不是直接替代召回层的万能模型。

| 场景 | 是否适合 BERT4Rec | 原因 |
|---|---:|---|
| 短中序列意图建模 | 适合 | 双向上下文能利用序列内部依赖 |
| 推荐重排 | 适合 | 候选集较小，可以精细打分 |
| 会话级兴趣理解 | 适合 | 行为集中，局部意图明显 |
| 极端冷启动用户 | 不适合 | 没有足够历史序列 |
| 只依赖超大召回覆盖的系统 | 不适合单独使用 | BERT4Rec 本身不解决候选生成 |
| 未裁剪的超长序列 | 不适合直接使用 | Transformer 计算复杂度较高 |

边界必须明确：BERT4Rec 学的是“序列建模与补全能力”。线上推荐通常还需要召回、过滤、规则、特征交叉、排序、业务约束和多目标优化。

---

## 核心机制与推导

BERT4Rec 的机制可以拆成五步：

1. 序列输入  
2. 随机 mask  
3. 加入位置编码  
4. 双向 Transformer 编码  
5. 对 mask 位做 softmax 分类  

新手版玩具例子：

原序列是 `[A, B, C, D]`，随机 mask 第 2 位和第 4 位，输入变成：

```text
[A, MASK, C, MASK]
```

模型分别预测第 2 位和第 4 位的真实物品。它不是从左到右生成整段序列，而是在固定位置做“填空”。

第一步是输入表示。模型不能直接理解 item id，所以要把每个 item 转成向量：

$$
h_i^0=e(x_i)+p_i
$$

其中 $e(x_i)$ 表示 item embedding，负责表达“这个物品是什么”；$p_i$ 表示 position embedding，负责表达“这个物品在序列中的第几个位置”。如果没有位置编码，`[A, B, C]` 和 `[C, B, A]` 在模型眼里会非常接近，顺序信息会丢失。

第二步是双向 Transformer 编码。Transformer 是一种基于 Attention 的序列模型，Attention 可以理解为“当前位置根据相关性从其他位置取信息”。双向 Attention 表示每个位置都可以看左侧和右侧的上下文。

第三步是训练目标。只在被 mask 的位置计算损失：

$$
L=-\frac{1}{|M|}\sum_{i\in M}\log P(x_i=x_i^*\mid X')
$$

交叉熵损失可以理解为“正确答案概率越高，损失越小”。如果模型给真实 item 的概率很低，损失就会变大。

最小数值例子：

| 位置 | 真实 item | 模型给真实 item 的概率 | 损失 |
|---|---|---:|---:|
| 第 2 位 | B | $P(B)=0.70$ | $-\log(0.70)=0.357$ |
| 第 4 位 | D | $P(D)=0.40$ | $-\log(0.40)=0.916$ |
| 平均 | - | - | $(0.357+0.916)/2=0.6365$ |

从这个例子可以看到，模型不是直接输出一个推荐列表，而是先学习“在上下文里某个位置应该是什么物品”。测试时，常见做法是在序列末尾追加 `[MASK]`，例如 `[A, B, C, MASK]`，让模型预测最后一个位置，从而得到下一件推荐物品的分数。

---

## 代码实现

实现 BERT4Rec 时，重点不是写一个很大的模型，而是把 mask、位置编码、输出层和损失函数接对。

代码结构通常拆成三块：数据预处理、模型前向、训练与评估。

| 模块 | 职责 | 容易出错点 |
|---|---|---|
| `Dataset.__getitem__` | 生成序列、随机 mask、保存标签 | label 必须是真实 item，不是 mask 后的值 |
| `collate_fn` | 对齐 batch 长度 | padding 位置不能参与损失 |
| `BERT4Rec.forward` | 输出每个位置的 hidden states | 需要加入 position embedding |
| `loss_fn` | 只在 mask 位置计算损失 | 不能对所有位置都算交叉熵 |

新手版伪代码：

```python
x, mask_pos, labels = sample_sequence(user_seq)
x_emb = item_emb(x) + pos_emb(range(len(x)))
h = transformer_encoder(x_emb)
logits = mlp(h[mask_pos])
loss = cross_entropy(logits, labels)
```

下面是一个可运行的最小 Python 例子，用来演示 mask 采样和损失计算。它不是完整 BERT4Rec，只保留训练目标的关键逻辑。

```python
import math
import random

MASK_ID = 0
PAD_ID = -1

def mask_sequence(seq, mask_ratio=0.5, seed=7):
    random.seed(seed)
    n = len(seq)
    mask_count = max(1, int(n * mask_ratio))
    mask_pos = sorted(random.sample(range(n), mask_count))

    x = list(seq)
    labels = []
    for pos in mask_pos:
        labels.append(seq[pos])
        x[pos] = MASK_ID

    return x, mask_pos, labels

def cross_entropy_from_true_probs(true_probs):
    return sum(-math.log(p) for p in true_probs) / len(true_probs)

seq = [101, 102, 103, 104]
x, mask_pos, labels = mask_sequence(seq, mask_ratio=0.5, seed=1)

assert len(mask_pos) == 2
assert len(labels) == 2
assert all(x[pos] == MASK_ID for pos in mask_pos)

# 假设模型在两个 mask 位置给真实 item 的概率分别是 0.70 和 0.40
loss = cross_entropy_from_true_probs([0.70, 0.40])
assert round(loss, 4) == 0.6365

print(x, mask_pos, labels, round(loss, 4))
```

真实工程里的训练流程通常是：

| 训练注意点 | 推荐做法 | 原因 |
|---|---|---|
| mask 比例 | 常从 15% 到 30% 试起 | 太低信号少，太高上下文不足 |
| 序列截断长度 | 保留最近 $N$ 个行为 | 控制计算成本，强化近期意图 |
| padding 位置 | 使用 attention mask 忽略 | padding 不是用户行为 |
| 测试构造 | 在末尾追加 `[MASK]` | 用最后位置预测 next item |
| 负样本评估 | 在候选集合上排序 | 全量 item 打分成本高 |

训练时，输入可能是 `[A, MASK, C, MASK]`；测试时，输入通常是 `[A, B, C, MASK]`。这个差异很重要，因为训练目标是补全任意位置，线上目标经常是预测下一项。

---

## 工程权衡与常见坑

BERT4Rec 的优势来自双向上下文，但工程落地时不能只看论文指标。

第一个常见坑是 mask 比例。mask 过高时，模型看到的上下文太少，任务会变成“盲猜”；mask 过低时，模型训练信号不足，学习效率低。对初始系统来说，先从 15% 或 20% 开始调参，比直接追求复杂策略更稳。

第二个常见坑是训练和测试目标不一致。训练时模型经常随机遮住中间位置，但上线时通常只预测“下一件商品”。新手版理解：如果平时练的是“填句子中间的空”，考试却只考“接下一句”，能力会相关，但不完全一致。因此工程上常加入 next-item 微调，或者在训练样本中增加“只 mask 最后一个位置”的样本。

| 常见坑 | 后果 | 规避方式 |
|---|---|---|
| `mask` 比例过高 | 上下文不足，预测不稳定 | 降低 mask 比例 |
| `mask` 比例过低 | 学习信号太少 | 提高 mask 数量或增加训练轮数 |
| 训练/测试不一致 | 线上 next-item 效果不稳定 | 加入 next-item 微调 |
| 序列太短 | 双向上下文优势不明显 | 改用轻量基线或混合特征 |
| 忘记位置编码 | 无法建模顺序 | 加入 position embedding |
| padding 参与损失 | 模型学习无意义位置 | padding 位置设为 ignore |
| 冷启动/长尾覆盖差 | 推荐覆盖不足 | 配合召回与内容特征 |

线上部署时，BERT4Rec 更常见的位置是重排层，而不是全量召回层。原因很直接：如果商品库有几百万个 item，让 Transformer 对全量 item 做实时打分成本很高。更可行的做法是先用召回系统拿到几百或几千个候选，再用 BERT4Rec 的用户序列表征对候选排序。

| 方案 | 优点 | 缺点 |
|---|---|---|
| 纯序列模型 | 实现链路清晰，能捕捉行为依赖 | 冷启动、长尾、候选覆盖弱 |
| 序列 + 召回 + 内容特征 | 覆盖更好，工程稳定性更强 | 系统复杂度更高 |
| 序列模型只做重排 | 成本可控，收益集中 | 依赖上游候选质量 |

一个真实工程例子：

电商首页“猜你喜欢”通常先由多路召回产生候选商品，例如相似商品召回、热门商品召回、协同过滤召回、内容召回。BERT4Rec 可以读取用户最近 20 到 100 次行为，输出用户当前兴趣表示，再对候选商品打分。这样它不需要承担“从全站商品里找候选”的职责，而是专注判断“这些候选中哪个更符合当前序列意图”。

---

## 替代方案与适用边界

不是所有序列推荐都应该用 BERT4Rec。模型选择要看序列长度、实时性、召回覆盖、冷启动程度和业务目标。

BERT4Rec 的目标是 mask 补全：

$$
P(x_i\mid X')
$$

SASRec 的目标更接近下一项预测：

$$
P(x_{t+1}\mid x_1,x_2,\dots,x_t)
$$

如果业务只需要“根据历史预测下一步点击”，且序列很长、实时性要求高，SASRec 或更轻量的 next-item 模型可能更合适。如果业务需要利用前后文补全用户意图，BERT4Rec 的双向上下文更有价值。

| 替代方案 | 核心思路 | 适用场景 |
|---|---|---|
| SASRec | 单向自注意力，根据左侧历史预测下一项 | next-item 推荐、实时预测 |
| GRU4Rec | 用 GRU 建模行为序列 | 早期系统、资源有限场景 |
| 会话推荐模型 | 重点建模短会话内兴趣 | 匿名用户、短期意图明显 |
| 内容增强模型 | 引入文本、图像、类目等内容特征 | 冷启动、长尾 item |
| BERT4Rec | 随机 mask，双向补全 | 序列补全、重排、意图建模 |

选择时可以按下面的条件判断：

| 条件 | 更倾向 BERT4Rec | 更倾向其他方案 |
|---|---|---|
| 序列长度 | 短中序列，已截断 | 超长序列且实时性极强 |
| 是否需要双向上下文 | 需要理解前后文关系 | 只关心下一步 |
| 实时性要求 | 可接受重排层延迟 | 毫秒级强约束 |
| 冷启动严重程度 | 有一定历史行为 | 新用户、新 item 很多 |
| 候选集质量 | 上游召回较稳定 | 召回覆盖本身很差 |

推荐系统工程里，模型不是越复杂越好。BERT4Rec 更像一个强序列编码器：当用户行为序列本身包含足够信号，并且系统能提供可靠候选集时，它能有效提升排序质量；当主要问题是冷启动、召回覆盖或业务规则约束时，单独上 BERT4Rec 很难解决核心问题。

---

## 参考资料

| 阅读顺序 | 资料 | 目的 |
|---:|---|---|
| 1 | BERT 原论文 | 理解 Masked Language Modeling 和双向 Transformer |
| 2 | BERT4Rec 论文 | 理解如何把 BERT 思路迁移到序列推荐 |
| 3 | SASRec 论文 | 对比单向自注意力推荐 |
| 4 | 官方代码 | 确认训练细节和实现方式 |

1. [BERT4Rec: Sequential Recommendation with Bidirectional Encoder Representations from Transformer](https://arxiv.org/pdf/1904.06690)
2. [Self-Attentive Sequential Recommendation](https://arxiv.org/abs/1808.09781)
3. [BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding](https://arxiv.org/abs/1810.04805)
4. [BERT4Rec 官方代码](https://github.com/FeiSun/BERT4Rec)
5. [session-aware-bert4rec 可复现实现](https://github.com/theeluwin/session-aware-bert4rec)
