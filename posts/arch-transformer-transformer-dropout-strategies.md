## 核心结论

Transformer 的 Dropout 不是“在模型里随便丢一些神经元”这么简单，而是按结构分层放置的随机正则化策略。正则化的白话解释是：故意在训练时制造一点扰动，防止模型把训练集记死。对 Transformer 来说，常见位置至少有四类：

| 类型 | 作用位置 | 主要抑制的问题 | 训练阶段 | 推理阶段 |
| --- | --- | --- | --- | --- |
| Embedding Dropout | 词向量与位置编码相加后 | 输入特征共适应 | 开启 | 关闭 |
| Attention Dropout | softmax 后的注意力权重 | 对少数邻居 token 过度依赖 | 开启 | 关闭 |
| Residual Dropout | 子层输出进入残差相加前 | 深层堆叠中的过拟合与不稳定 | 开启 | 关闭 |
| DropPath / Stochastic Depth | 整个残差分支 | 深层网络训练困难、层间依赖过强 | 随机跳过部分层 | 关闭随机性，保留全路径或做期望缩放 |

它们并不重复。Embedding Dropout 和 Attention Dropout 主要打断“特征维度”和“注意对象”的固定依赖；Residual Dropout 和 DropPath 主要打断“层级堆叠”的固定依赖。前者更像局部扰动，后者更像结构扰动。

最基础的 Dropout 写法是：

$$
y = \frac{x \odot m}{1-p}, \quad m \sim \text{Bernoulli}(1-p)
$$

这里的伯努利采样，白话解释是：每个位置都独立抛一次硬币，决定保留还是置零。除以 $1-p$ 的目的是保持输出期望不变，也就是训练和推理的平均量级一致。

玩具例子：一个 512 维的 embedding，若 Embedding Dropout 概率 $p=0.1$，训练时平均约有 $512 \times 0.1 \approx 51$ 个维度被置零，其余维度乘上 $\frac{1}{0.9}$。新手可以把它理解为：随机删掉 10% 的输入维度，但整体能量大体不变。

---

## 问题定义与边界

问题不是“Transformer 要不要 Dropout”，而是“在哪一层、以什么粒度、为了解决什么问题加 Dropout”。

传统全连接网络里的 Dropout，主要针对特征共适应。特征共适应的白话解释是：多个特征总是一起出现，模型开始偷懒，只靠固定搭配做判断。Transformer 比传统网络复杂得多，因为它同时有：

1. token 级输入表示
2. 多头注意力分布
3. 残差连接
4. 深层堆叠结构

因此，只在输入端打一次 Dropout 往往不够。一个常见误解是：“embedding 上打 0.1 的 dropout 就行。”这在浅模型或小任务上可能还能工作，但对标准 Transformer 来说，注意力权重和残差分支本身也是过拟合与不稳定的来源。

边界也要讲清楚：

| 问题 | 是否属于本文范围 | 说明 |
| --- | --- | --- |
| Dropout 在 Transformer 中放哪 | 是 | 本文核心 |
| 推理时 Dropout 是否开启 | 是 | 必须关闭随机性 |
| LayerNorm 与 BatchNorm 的统计差异 | 是 | 解释为何不会像 BN 一样漂移 |
| 数据增强、权重衰减、标签平滑 | 部分相关 | 属于更广义正则化，但不是本文主线 |
| MoE 路由噪声、专家丢弃 | 否 | 是更特殊的 Transformer 变体问题 |

还要区分训练与推理。训练阶段的目标是提升泛化，因此允许随机扰动；推理阶段的目标是稳定输出，因此这些随机开关必须关掉。这里一个容易混淆的点是 LayerNorm。LayerNorm 的白话解释是：对每个样本自己的特征做归一化，不依赖整个 batch 的均值和方差。正因为如此，Transformer 推理时不会像 BatchNorm 那样面临“训练统计和推理统计不一致”的问题，但这不代表 Dropout 可以继续开着。Dropout 的随机性不会被 LayerNorm 自动抵消。

---

## 核心机制与推导

可以把一个 Transformer 子层简化成下面这条链路：

```text
输入表示
  ↓
Embedding Dropout
  ↓
Multi-Head Attention
  ↓
Attention Softmax
  ↓
Attention Dropout
  ↓
线性投影
  ↓
Residual Dropout
  ↓
与残差相加
  ↓
LayerNorm
```

如果模型更深，还可能在“整个残差分支”上再包一层 DropPath。

### 1. Embedding Dropout

Embedding 是词向量与位置编码的和。位置编码的白话解释是：告诉模型“这个 token 在序列的第几个位置”。常见做法是在两者相加后做 Dropout，而不是只对词向量做。

原因很直接：模型输入一开始就容易形成固定维度依赖。如果某些 embedding 维度总是特别有用，模型会过早地把判断压到这些维度上。Embedding Dropout 会迫使模型把信息更分散地编码到多个维度。

设输入向量为 $e \in \mathbb{R}^d$，掩码为 $m \in \{0,1\}^d$，则输出为：

$$
\tilde{e} = \frac{e \odot m}{1-p}
$$

玩具例子：句子只有两个 token，embedding 维度只有 4。若第一个 token 的表示是 $[2, 1, 3, 4]$，$p=0.5$，一次采样后掩码可能是 $[1, 0, 1, 0]$，则输出是 $[4, 0, 6, 0]$。模型不能再总是依赖第 2 和第 4 维。

### 2. Attention Dropout

注意力权重来自 softmax。softmax 的白话解释是：把一组分数变成和为 1 的概率分布。Transformer 很容易出现“某几个邻居 token 权重过高”的现象，尤其在数据量不大或头数较多时更明显。

标准注意力是：

$$
\text{Attn}(Q,K,V)=\text{softmax}\left(\frac{QK^\top}{\sqrt{d_k}}\right)V
$$

Attention Dropout 作用在 softmax 输出后的权重矩阵 $A$ 上：

$$
\tilde{A} = \frac{A \odot M}{1-p}
$$

然后再做 $\tilde{A}V$。它的含义不是“丢掉值向量”，而是“随机让一部分注意连接失效”。模型因此不能永远盯着同一个上下文位置。

例如 decoder 在翻译时，如果当前词总是只看前一个词，Attention Dropout 会迫使它有时去参考更远的上下文，从而缓解局部依赖过强的问题。

### 3. Residual Dropout

残差连接的白话解释是：把原输入直接绕过子层，再与子层输出相加，避免深层网络训练困难。Transformer 每个子层几乎都带残差，因此子层输出的波动会层层传递。

若子层变换记为 $\mathcal{T}(x)$，Residual Dropout 通常写成：

$$
y = x + \text{Dropout}(\mathcal{T}(x))
$$

而不是：

$$
y = \text{Dropout}(x + \mathcal{T}(x))
$$

两者效果不同。前者只扰动新学到的增量分支，保留主干信息；后者会直接扰动整个和，训练更不稳定。标准 Transformer 实践中通常采用前者。

这类 Dropout 的目标不只是防过拟合，还包括稳定深层训练。因为每一层都必须学会：即使自己的增量分支有一部分随机失效，主干表示也要保持可用。

### 4. DropPath / Stochastic Depth

DropPath 也叫 Stochastic Depth。它不是丢掉若干元素，而是按样本随机跳过整个残差分支。白话说，就是这层“今天可能不工作”，直接让输入穿过去。

若第 $l$ 层变换为 $\mathcal{T}_l(H_l)$，可写为：

$$
H_{l+1} = H_l + \frac{z_l}{1-p_l}\mathcal{T}_l(H_l), \quad z_l \sim \text{Bernoulli}(1-p_l)
$$

也常见等价写法：

$$
H_{l+1} = (1-d_l)\mathcal{T}_l(H_l) + d_l H_l
$$

本质都是随机短路。短路的白话解释是：让某条计算路径临时断开，直接走捷径。

它特别适合深层 Transformer。层数越深，模型越容易依赖固定的层间协作模式；DropPath 迫使不同深度的子网络都能工作。很多实现还会让深层的丢弃率更高，例如从前层的 0 线性增长到最后一层的 0.2，因为越深的层越像“可选增强层”。

真实工程例子：做机器翻译或语音识别时，训练一个 24 层 Transformer，如果引入 LayerDrop 或 DropPath，那么部署阶段可以只保留前 18 层甚至 12 层做低延迟推理，而且不一定需要重新微调。这不是普通 Embedding Dropout 能做到的，因为它只影响特征，不提供“可裁剪深度”的鲁棒性。

---

## 代码实现

下面用一个最小可运行的 Python 例子，把四类策略的核心行为写清楚。这个例子不依赖深度学习框架，但逻辑和 PyTorch 实现一致。

```python
import random
from math import isclose

def dropout_vector(x, p, training=True):
    if not training or p == 0.0:
        return list(x)
    keep = 1.0 - p
    out = []
    for v in x:
        m = 1 if random.random() < keep else 0
        out.append(v * m / keep)
    return out

def attention_dropout(weights, p, training=True):
    if not training or p == 0.0:
        return list(weights)
    keep = 1.0 - p
    dropped = []
    for w in weights:
        m = 1 if random.random() < keep else 0
        dropped.append(w * m / keep)
    return dropped

def residual_add(x, sublayer_out, p, training=True):
    dropped = dropout_vector(sublayer_out, p, training=training)
    return [a + b for a, b in zip(x, dropped)]

def drop_path(x, branch_out, p, training=True):
    if not training or p == 0.0:
        return [a + b for a, b in zip(x, branch_out)]
    keep = 1.0 - p
    z = 1 if random.random() < keep else 0
    scaled_branch = [v * z / keep for v in branch_out]
    return [a + b for a, b in zip(x, scaled_branch)]

# 推理阶段应无随机性
x = [1.0, 2.0, 3.0]
assert dropout_vector(x, p=0.5, training=False) == x
assert attention_dropout([0.2, 0.3, 0.5], p=0.5, training=False) == [0.2, 0.3, 0.5]

# residual 形状与主干一致
res = residual_add([1, 1], [2, 2], p=0.5, training=True)
assert len(res) == 2

# drop_path 在推理阶段等价于直接保留分支
dp = drop_path([1, 2], [3, 4], p=0.9, training=False)
assert dp == [4, 6]

# 一个确定性检查：p=0 时训练与推理一致
assert residual_add([1, 2], [3, 4], p=0.0, training=True) == [4, 6]
assert isclose(sum([0.2, 0.3, 0.5]), 1.0)
```

如果换成 PyTorch，结构通常更接近这样：

```python
import torch
import torch.nn as nn
import torch.nn.functional as F

class TinyTransformerBlock(nn.Module):
    def __init__(self, d_model=8, p_embed=0.1, p_attn=0.1, p_res=0.1, p_path=0.0):
        super().__init__()
        self.embed_dropout = nn.Dropout(p_embed)
        self.attn_dropout = nn.Dropout(p_attn)
        self.res_dropout = nn.Dropout(p_res)
        self.proj = nn.Linear(d_model, d_model)
        self.norm = nn.LayerNorm(d_model)
        self.p_path = p_path

    def drop_path(self, x, branch):
        if (not self.training) or self.p_path == 0.0:
            return x + branch
        keep = 1.0 - self.p_path
        mask = torch.bernoulli(torch.full((x.size(0), 1, 1), keep, device=x.device))
        branch = branch * mask / keep
        return x + branch

    def forward(self, x, attn_scores):
        x = self.embed_dropout(x)

        attn = F.softmax(attn_scores, dim=-1)
        attn = self.attn_dropout(attn)

        branch = self.proj(x)
        branch = self.res_dropout(branch)

        x = self.drop_path(x, branch)
        x = self.norm(x)
        return x, attn

model = TinyTransformerBlock()
model.train()
x = torch.randn(2, 4, 8)
scores = torch.randn(2, 4, 4)
y, attn = model(x, scores)
assert y.shape == x.shape
assert attn.shape == scores.shape

model.eval()
y2, attn2 = model(x, scores)
assert y2.shape == x.shape
```

实现时有两个原则不能错：

1. `model.train()` 时才让 Dropout 生效。
2. `model.eval()` 时必须关闭 Embedding、Attention、Residual、DropPath 的随机行为。

---

## 工程权衡与常见坑

不同 Dropout 解决的问题不同，所以概率不能“一把梭”。经验上，输入层和残差层的 dropout 往往可以共存，但 attention dropout 过大时容易直接伤害建模能力，因为注意力分布本来就很稀疏。

| 常见坑 | 表现 | 原因 | 解决方案 |
| --- | --- | --- | --- |
| 推理输出不一致 | 同一输入多次结果不同 | 忘记 `model.eval()` | 推理前显式 `eval()` |
| Attention Dropout 过高 | 收敛慢、翻译漏词 | 注意连接被打断过多 | 降低到 0.1 或更低 |
| Residual Dropout 放错位置 | 深层训练不稳 | 把整个残差和一起 dropout | 只对分支输出做 dropout |
| DropPath 概率过大 | 深层几乎学不到东西 | 整个分支被频繁跳过 | 采用分层递增且上限保守 |
| 误以为 LayerNorm 能替代 eval | 推理仍随机 | LN 不会关闭 Dropout | 区分归一化与随机正则化 |

一个真实工程坑是机器翻译推理不稳定。训练好的 checkpoint 用同一条英文句子翻译三次，中文结果不一样，排查半天发现不是 beam search 问题，而是测试脚本没有 `model.eval()`。由于 attention dropout 和 residual dropout 仍在生效，每次前向图都不同，输出自然飘。

另一个权衡是深度与鲁棒性。Residual Dropout 更适合“保留所有层，但让每层学得别太死”；DropPath 更适合“允许某些层偶尔缺席，逼模型适应不同深度”。如果你明确有部署裁剪需求，比如服务器版 24 层、端侧版 12 层共用一套参数，那么 DropPath 或 LayerDrop 的价值就明显更高。

还有一个边界问题：在超大数据和超大模型下，有些现代 Transformer 会把标准 dropout 用得很轻，甚至某些位置接近不用。这不代表 Dropout 失效，而是大数据本身就有正则化效果，同时优化器、学习率调度、权重衰减等手段已足够强。对零基础到初级工程师来说，更合理的结论是：小中规模任务里，Dropout 仍然是默认应考虑的稳定组件。

---

## 替代方案与适用边界

Dropout 不是唯一选择。它属于“训练时加随机扰动”的一大类方法，但不同方法对 Transformer 的作用层次不同。

| 方法 | 作用层级 | 优点 | 缺点 | 适用场景 |
| --- | --- | --- | --- | --- |
| 标准 Dropout | 元素级 | 简单、便宜、框架原生支持 | 对结构级鲁棒性帮助有限 | 中小模型、常规训练 |
| Attention Dropout | 注意边级 | 抑制对固定邻居的依赖 | 过高会伤害对齐 | 编码器、翻译、理解任务 |
| Residual Dropout | 分支输出级 | 深层更稳，易与标准块集成 | 不能直接支持裁剪深度 | 标准 Transformer 堆叠 |
| DropPath / Stochastic Depth | 整层分支级 | 提升深层鲁棒性，支持深度裁剪 | 超参更敏感 | 深层网络、端侧部署 |
| LayerDrop | 层级 | 训练一套模型支持多深度推理 | 训练策略更复杂 | 需要动态深度的场景 |

LayerDrop 可以看成 DropPath 在 Transformer 层级上的更直接版本，尤其适合“训练时随机删层，推理时保留部分层”的设计。对于生成任务，比如摘要和对话，过高的 Attention Dropout 往往会伤害上下文连贯性，因此有时会更依赖 Residual Dropout 或较温和的 DropPath。对于分类或检索任务，attention 分布没有生成任务那么脆弱，Attention Dropout 通常更容易带来泛化收益。

一个新手能直接记住的判断规则是：

1. 你担心输入表示过拟合，用 Embedding Dropout。
2. 你担心模型总盯着固定上下文，用 Attention Dropout。
3. 你担心深层残差堆叠不稳，用 Residual Dropout。
4. 你还想让同一模型支持不同推理深度，用 DropPath 或 LayerDrop。

不要把这些策略理解成“越多越好”。它们的本质都是往训练里加噪声。噪声不足，模型容易记忆训练集；噪声过强，模型连应该学的模式也学不住。工程上最重要的不是把名词堆满，而是知道每个随机化手段究竟在打断哪一类依赖。

---

## 参考资料

- Vaswani et al., *Attention Is All You Need*. Transformer 原始论文，重点看模型结构图与残差、注意力计算公式。
- Fan et al., *Reducing Transformer Depth on Demand with Structured Dropout*. LayerDrop 代表性工作，重点看“单模型多深度部署”。
- ApX Machine Learning, *Transformer Regularization Techniques*. 汇总 Transformer 中常见正则化位置，适合建立全局图景。
- Aman.ai, *Transformer Primer*. 对注意力、残差、归一化等基础部件有系统解释，适合配合原论文阅读。
- next.gr, *Transformers Architecture Explained in Depth*. 对 Dropout、DropPath 等公式有较直观的实现解释。
- 关于 LayerNorm 与 BatchNorm 差异的技术博客与实现说明。阅读重点是“LayerNorm 不依赖 batch 统计，但不会替你关闭 Dropout”。
