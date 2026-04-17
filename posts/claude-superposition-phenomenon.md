## 核心结论

Superposition 可以直译为“叠加表征”。白话说法是：模型不会给每个概念都分配一个独立神经元或独立维度，而是让多个概念共用同一组方向，只要这些概念很少同时出现，就能把“超过维度数的特征”塞进有限空间。

Anthropic 在 2022 年的 toy model 中给出的核心结论是：当输入特征足够稀疏时，ReLU 这类非线性激活会让模型愿意接受少量干扰，用更高的表示容量换更低的维度开销。[Toy Models of Superposition](https://www.anthropic.com/news/toy-models-of-superposition) 与 [Transformer Circuits 原文](https://www.transformer-circuits.pub/2022/toy_model/) 都显示，模型会自发形成一些稳定几何结构，例如 antipodal pair、三角形、五边形，而不是只保留严格正交的方向。

这件事和 Claude 的关系在于：Anthropic 在 2024 年对 Claude 3 Sonnet 做特征提取时，明确指出“每个概念分布在许多神经元上，每个神经元也参与许多概念”，并借助 dictionary learning / sparse autoencoder 从中提取出数百万可解释特征。[Mapping the Mind of a Large Language Model](https://www.anthropic.com/research/mapping-mind-language-model) 说明，Claude 内部并不是“一神经元一概念”，而是先存在重叠表征，再通过稀疏特征分解把概念方向挖出来。

一个常用指标是

$$
D^*=\frac{m}{\lVert W\rVert_F^2}
$$

其中 $m$ 是隐藏层维度，$\lVert W\rVert_F^2$ 是权重矩阵的 Frobenius 范数平方。它可以理解为“每个特征平均占用多少维度”。$D^*$ 越小，说明单位维度里塞入的特征越多，superposition 越强。

---

## 问题定义与边界

先把符号讲清楚。设输入里有 $n$ 个潜在特征，隐藏层只有 $m$ 个维度，而且 $n \gg m$。这里“特征”不是输入列本身，而是模型内部想表示的语义方向，例如“Golden Gate Bridge”“代码 bug”“拍马屁语气”。“稀疏”指每个样本只会激活很少几个特征。

在 toy model 里，常把 $S$ 记作“一个特征不出现的概率”。那么单个特征出现概率是 $1-S$，平均激活特征数约为 $n(1-S)$。当

$$
n(1-S) \ll 1
$$

时，表示大多数样本里只有极少数特征同时打开，特征冲突很少，superposition 最容易出现。比起死板地记“$S<1/N$”，更稳妥的判断方式是看“平均同时活跃的特征数是否远小于 1 或至少远小于 $m$”。

下面这张表适合做快速判断：

| 条件 | 直观含义 | 更可能出现什么 |
| --- | --- | --- |
| $n \approx m$，且特征不稀疏 | 维度够用，很多特征常一起出现 | 正交表示或接近 PCA |
| $n \gg m$，但特征很密 | 维度不够，冲突又很多 | 强干扰，superposition 收益下降 |
| $n \gg m$，且 $n(1-S)\ll 1$ | 特征多但几乎不同时出现 | superposition 最明显 |
| 稀疏且有 ReLU / threshold | 可把小干扰“截掉” | 更容易稳定叠加表示 |

玩具例子：设 $n=5,\ m=2,\ S=0.999$。这表示有 5 个概念，但隐藏层只有 2 维；每个概念只在约 0.1% 的样本里出现。此时如果坚持“一维只服务一个特征”，最多可靠表示 2 个概念；但如果允许 5 个概念共享 2 维空间中的多个方向，只要它们很少同时激活，模型就能在大多数样本上正常重构，这就是 superposition。

边界也要说清楚。superposition 不是“越强越好”，它依赖三个前提：

| 变量 | 太小时 | 合适时 | 太大时 |
| --- | --- | --- | --- |
| 特征稀疏度 | 不足以产生收益 | 干扰低、容量高 | 无明显问题 |
| 特征/维度比 $n/m$ | 没必要叠加 | 有压缩动机 | 干扰可能过大 |
| 非线性阈值与 bias | 过滤不掉串扰 | 能截断弱冲突 | 过强会丢信息 |

所以，superposition 不是“模型一定会这样做”，而是“在维度紧张且特征稀疏时，常见的最优解之一”。

---

## 核心机制与推导

Anthropic toy model 的简化形式是：

$$
h = Wx
$$

$$
\hat{x}=\text{ReLU}(W^\top W x + b)
$$

这里 $x$ 是输入特征，$W$ 是编码矩阵，$h$ 是隐藏表示，$\hat{x}$ 是重构结果，$b$ 是偏置。ReLU 可以理解成“把负数直接截成 0 的开关”。

机制可以按四步理解。

1. 输入是稀疏的。多数样本只含少量特征，所以“特征 A 和特征 B 撞在一起”的情况不常见。
2. 编码矩阵 $W$ 不必给每个特征一条独占维度，而是把多个特征投影到有限维度中的不同方向。
3. 这些方向并不完全正交，会有内积，于是带来 interference，也就是“串扰”。
4. 解码时的 ReLU 和负 bias 会把很多弱负响应直接裁掉，于是模型容忍少量串扰，换来更高容量。

如果两个特征几乎互斥，那么把它们放在同一条轴的相反方向上尤其划算。Transformer Circuits 文章里把这种结构叫 antipodal pair。白话说法是：一维不再只存一个概念，而是“正方向表示特征 A，负方向表示特征 B”。因为 A 和 B 很少同时出现，这种复用在统计上是划算的。

为什么会出现三角形、五边形这类几何？因为当要塞入的特征更多时，模型要在“夹角尽量大”和“总共能塞更多方向”之间取平衡。二维空间里，若要放 5 个方向，均匀铺成近似五边形就比随便堆叠更稳定。这个结果不是人工写死的，而是梯度下降训练后自然出现的。

可以把推导链路简化成下面的流程：

`稀疏输入`  
`-> 同时激活特征少`  
`-> 允许少量方向重叠`  
`-> W 学到近似均匀分布的方向`  
`-> ReLU + 负 bias 过滤弱干扰`  
`-> 用 m 维表示超过 m 个特征`

$D^*$ 指标正是对这个过程的压缩度量。如果 $\lVert W\rVert_F^2$ 近似等于“已表示特征数”，那么

$$
D^*=\frac{m}{\lVert W\rVert_F^2}
$$

就近似表示“每个特征平均分到多少维度”。当 $D^*=1$ 时，可理解为“一特征约占一维”；当 $D^*=1/2$ 时，可理解为“两特征共享一维”的典型状态。Anthropic 在 toy model 中观察到 $D^*$ 会在若干分数附近出现“粘滞区”，对应特定几何构型，而不是平滑连续变化。

---

## 代码实现

下面先给一个可运行的玩具代码。它不做训练，而是直接构造“5 个特征压进 2 维”的情形，展示“单个特征能重构、多个特征同时出现时误差变大”的基本现象。

```python
import math
import numpy as np

def relu(x):
    return np.maximum(x, 0.0)

# 5 个特征，2 维隐藏空间
n_features = 5
m = 2

# 把 5 个特征均匀放在单位圆上，形成近似五边形方向
angles = np.linspace(0, 2 * math.pi, n_features, endpoint=False)
W = np.stack([np.cos(angles), np.sin(angles)], axis=0)  # shape: (2, 5)

# 负 bias：过滤弱串扰
b = np.full(n_features, -0.2)

def encode(x):
    return W @ x

def decode(h):
    return relu(W.T @ h + b)

def reconstruct(x):
    return decode(encode(x))

# 单特征样本：重构应较好
for i in range(n_features):
    x = np.zeros(n_features)
    x[i] = 1.0
    x_hat = reconstruct(x)
    assert x_hat[i] > 0.7, (i, x_hat)
    off_target = np.delete(x_hat, i)
    assert off_target.max() < 0.5, (i, x_hat)

# 两个特征同时激活：干扰会上升
x_dense = np.zeros(n_features)
x_dense[0] = 1.0
x_dense[1] = 1.0
x_hat_dense = reconstruct(x_dense)

single_err = np.mean([
    np.linalg.norm(reconstruct(np.eye(n_features)[i]) - np.eye(n_features)[i])
    for i in range(n_features)
])
dense_err = np.linalg.norm(x_hat_dense - x_dense)

assert dense_err > single_err
print("single_err =", round(float(single_err), 4))
print("dense_err =", round(float(dense_err), 4))
print("D* =", round(float(m / np.sum(W ** 2)), 4))
```

这个例子传达两个信息：

1. 当一次只激活一个特征时，共享方向问题不大。
2. 当多个相关特征同时点亮时，串扰迅速上升。

如果要做更接近工程实践的实现，一般会训练 sparse autoencoder。白话解释是：它先把高维激活压缩成少量稀疏特征，再尽量把原激活重构回来。一个最小 PyTorch 框架可以写成这样：

```python
import torch
import torch.nn as nn
import torch.nn.functional as F

class SparseAutoencoder(nn.Module):
    def __init__(self, n, m):
        super().__init__()
        self.W_enc = nn.Linear(n, m, bias=False)
        self.b_dec = nn.Parameter(torch.full((n,), -0.1))

    def forward(self, x):
        h = self.W_enc(x)
        x_hat = F.relu(F.linear(h, self.W_enc.weight.t(), self.b_dec))
        return h, x_hat

def loss_fn(x, h, x_hat, l1_coef=1e-3):
    recon = F.mse_loss(x_hat, x)
    sparse = h.abs().mean()
    return recon + l1_coef * sparse, recon, sparse
```

训练时建议同时跟踪三类指标：

| 指标 | 作用 | 典型现象 |
| --- | --- | --- |
| 重构误差 | 看信息是否丢失 | 太大说明压缩过头 |
| 激活稀疏率 | 看特征是否足够稀疏 | 太密会破坏 superposition 假设 |
| $D^*$ 或等价容量指标 | 看单位维度塞了多少特征 | 过低可能伴随更强串扰 |

真实工程例子是 Anthropic 对 Claude 3 Sonnet 的解释性研究。他们不是直接解释单个神经元，而是先训练出大量稀疏特征，再观察这些特征何时激活、被放大后会怎样影响模型输出。[Mapping the Mind](https://www.anthropic.com/research/mapping-mind-language-model) 中展示了如“Golden Gate Bridge”“代码错误”“拍马屁式称赞”等特征，并通过人工放大特征验证它们对行为有因果影响。这说明：在大型模型里，实用路径往往不是“消灭 superposition”，而是“先承认它存在，再用 SAE 把它拆开”。

---

## 工程权衡与常见坑

最常见的误解是：只要数据足够稀疏，superposition 就一定稳定出现。事实不是这样。优化路径、初始化、weight decay、学习率都会改变最终几何结构。相同超参数下，不同 seed 可能得到不同构型，甚至一个模型学出干净的可解释特征，另一个模型学出同样低损失但更难解释的混合方向。

下面是工程里更常见的坑：

| 问题 | 症状 | 调整方向 |
| --- | --- | --- |
| 只盯重构误差 | loss 很好，但特征不可解释 | 加强稀疏正则，关注 feature activation 分布 |
| bias 初始化不当 | 特征全开或全关 | 解码 bias 设为轻微负值 |
| weight decay 太弱 | 特征数量很多但串扰严重 | 适当增大约束，减少无效重叠 |
| 只跑一个 seed | 结论不稳，难复现 | 多 seed 比较几何和指标 |
| 把特征数当成唯一目标 | “挖出更多特征”但质量下降 | 联合评估单义性与可干预性 |

一个真实工程上的判断标准是：“不是特征越多越好，而是可解释、可控制的特征越多越好。”如果某次训练挖出 300 万个方向，但人工检查时大多混杂多个概念，它对后续 steering、监控和故障排查的价值，可能不如另一版只有 120 万个、但更接近 monosemantic 的特征。monosemantic 可以理解为“一个方向主要表达一个概念，而不是很多概念的混合”。

还要注意一句容易被说反的话：superposition 帮模型提升容量，但解释性研究的目标常常是从 superposition 中再提纯出更干净的特征。也就是说，模型训练和模型解释追求的目标并不完全一样。前者接受重叠，后者希望拆分重叠。

---

## 替代方案与适用边界

如果特征并不稀疏，或者 $n$ 和 $m$ 差不多，superposition 往往不是最合适的视角。此时传统的低维线性方法，例如 PCA，或者直接使用更宽的隐藏层，通常更稳健。

下面给一个方法对比：

| 方法 | 核心假设 | 优点 | 局限 | 适用边界 |
| --- | --- | --- | --- | --- |
| PCA / 正交表示 | 主要结构可由线性主方向解释 | 简单、稳定、易算 | 难处理稀疏非线性特征 | 特征较密、$n \approx m$ |
| 更大模型 / 更宽层 | 直接增加容量 | 干扰少、训练简单 | 成本高 | 预算充足、追求性能优先 |
| Superposition | 特征多且很少同时激活 | 压缩能力强 | 有串扰，解释更难 | $n \gg m$ 且稀疏 |
| Superposition + SAE | 承认重叠后再做特征分解 | 更利于解释和干预 | 训练与评估复杂 | 大模型可解释性分析 |

一个面向新手的判断规则是：

- 如果每条样本会激活 50% 的概念，那么“500 个概念压到 100 维”大概率要靠更大的模型或更强表示能力，而不是靠 superposition。
- 如果每条样本只激活 0.1% 的概念，那么让 500 个概念共享 100 维就有现实可能，因为冲突几乎不发生。

所以，superposition 不是替代所有表示学习方法的统一答案。它更像一种稀疏条件下的容量优化策略。在理解 Claude 这类模型时，它解释了“为什么单个神经元不等于单个概念”；在做工程时，它提醒我们“不要把 polysemantic neuron 误当成随机噪声”。

---

## 参考资料

| 来源 | 日期 | 焦点 |
| --- | --- | --- |
| [Anthropic: Toy Models of Superposition](https://www.anthropic.com/news/toy-models-of-superposition) | 2022-09-14 | superposition 的基本定义、稀疏输入、ReLU 过滤干扰 |
| [Transformer Circuits: Toy Models of Superposition](https://www.transformer-circuits.pub/2022/toy_model/) | 2022-09-14 | 完整公式、$D^*$ 指标、相变与几何结构 |
| [Anthropic: Superposition, Memorization, and Double Descent](https://www.anthropic.com/research/superposition-memorization-and-double-descent) | 2023-01-05 | superposition 与过拟合、记忆化之间的关系 |
| [Anthropic: Mapping the Mind of a Large Language Model](https://www.anthropic.com/research/mapping-mind-language-model) | 2024-05-21 | Claude 3 Sonnet 的特征提取、特征操控与因果验证 |

这些资料合起来支持一个较稳妥的结论：在稀疏 regime 下，大模型会把概念分散到重叠方向中；而解释性方法的任务，是从这种重叠表征里重新提取出尽量单义、可干预的特征。Claude 的可解释性研究并没有证明“所有内部表示都已被完全理解”，但它已经证明：superposition 不是噪声，而是理解现代模型内部结构时必须先面对的基本事实。
