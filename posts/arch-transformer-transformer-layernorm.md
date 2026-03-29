## 核心结论

LayerNorm 的作用，是把单个 token 的特征向量先“拉回到稳定尺度”，再交给后续线性层和注意力层处理。这里的“特征向量”可以理解为一个 token 在某一层里的所有通道值。它逐 token、沿最后一个特征维计算均值和方差：

$$
\mu=\frac{1}{d}\sum_{i=1}^{d}x_i,\qquad
\sigma^2=\frac{1}{d}\sum_{i=1}^{d}(x_i-\mu)^2
$$

$$
y_i=\gamma_i\cdot\frac{x_i-\mu}{\sqrt{\sigma^2+\epsilon}}+\beta_i
$$

其中 $\gamma,\beta\in\mathbb{R}^d$ 是可学习参数，分别负责缩放和偏移。白话说，LayerNorm 先把数值“摆正”，再允许模型自己决定要不要重新拉伸、整体平移。

RMSNorm 是 LayerNorm 的简化版。它保留“控制幅度”这件事，但去掉“减均值”这一步，也通常去掉 $\beta$：

$$
\mathrm{RMS}(x)=\sqrt{\frac{1}{d}\sum_{i=1}^{d}x_i^2+\epsilon},\qquad
y_i=\gamma_i\cdot \frac{x_i}{\mathrm{RMS}(x)}
$$

这意味着 RMSNorm 只关心向量有多大，不关心它是否以 0 为中心。工程上它更便宜，因为少了一次均值中心化和一组 bias 参数。在 LLaMA、Qwen 一类大模型中，RMSNorm 已经是标准做法。

可以先记住三个结论：

| 模块 | 归一化对象 | 是否减均值 | 是否有 $\beta$ | 常见场景 |
| --- | --- | --- | --- | --- |
| LayerNorm | 单个 token 的最后一维 | 是 | 是 | 通用 Transformer、小中型模型 |
| RMSNorm | 单个 token 的最后一维 | 否 | 通常否 | 大语言模型、pre-norm |
| QK-Norm | 注意力里的 Q/K 向量 | 常用 RMS 方式 | 通常否 | 长上下文、抑制注意力 logit 爆炸 |

如果只问“Transformer 里现在更常见的选择是什么”，答案通常是：主干 block 用 RMSNorm，注意力里的 Q/K 再额外做 QK-Norm，很多线性层不带 bias。

---

## 问题定义与边界

Transformer 的核心问题不是“能不能算出结果”，而是“深层网络能不能稳定训练”。稳定训练指的是：前向时激活值不要一层层越滚越大，反向时梯度不要忽大忽小。

假设某一层输出向量 $x\in\mathbb{R}^d$，如果它的幅度随样本、token 位置、上下文长度剧烈波动，那么下一层看到的输入尺度就不稳定。残差连接会把这种波动继续累积，注意力里的点积 $QK^\top$ 还会把它进一步放大。归一化层的任务，就是在进入下一步计算前，把这种尺度波动压住。

这里有三个边界需要分清：

| 问题 | LayerNorm/RMSNorm 是否直接解决 | 说明 |
| --- | --- | --- |
| 残差流中的激活尺度漂移 | 是 | 这是主任务 |
| 注意力 logits 过大 | 部分解决 | 主干 norm 有帮助，但通常还需要 QK-Norm |
| batch 间统计差异 | 否 | 它不依赖 batch 统计，不像 BatchNorm |

“逐 token 归一化”是 LayerNorm 和 RMSNorm 的关键边界。它们不是沿 batch 维统计，而是对每个 token 自己的特征向量做归一化。所以在 NLP 里它们比 BatchNorm 更自然，因为文本长度变化大、batch 统计不稳定。

再看一个典型问题。设 $q,k\in\mathbb{R}^d$，若每一维独立、方差近似相同，则点积

$$
q^\top k=\sum_{i=1}^{d} q_i k_i
$$

的方差会随 $d$ 近似线性增长。白话说，维度越大，注意力分数越容易变得过大。softmax 遇到极大的输入时会变得很尖，接近 one-hot，导致梯度难学。QK-Norm 的目标，就是在 Q、K 投影后先做归一化，再做点积，把分数范围控制住。

所以本文讨论的边界是：Transformer 中 token 级特征归一化、注意力前的 Q/K 归一化，以及它们在大模型中的工程配置；不讨论 BatchNorm、GroupNorm，也不讨论优化器层面的梯度裁剪。

---

## 核心机制与推导

先看一个玩具例子。某个 token 的隐藏状态是：

$$
x=[1,2,3]
$$

### 1. LayerNorm 怎么算

均值：

$$
\mu=\frac{1+2+3}{3}=2
$$

方差：

$$
\sigma^2=\frac{(1-2)^2+(2-2)^2+(3-2)^2}{3}=\frac{2}{3}
$$

标准差：

$$
\sigma=\sqrt{\frac{2}{3}}\approx 0.8165
$$

若先设 $\gamma=[1,1,1],\beta=[0,0,0]$，则输出约为：

$$
\left[\frac{-1}{0.8165},\frac{0}{0.8165},\frac{1}{0.8165}\right]\approx[-1.225,0,1.225]
$$

这说明 LayerNorm 做了两件事：

1. 中心化：把整体平移到 0 附近。
2. 标准化：把整体缩放到单位方差附近。

中心化的意义，是让每层输入分布更可控；标准化的意义，是让不同 token 的尺度接近。

### 2. RMSNorm 怎么算

同样对 $x=[1,2,3]$：

$$
\mathrm{RMS}(x)=\sqrt{\frac{1^2+2^2+3^2}{3}}=\sqrt{\frac{14}{3}}\approx 2.1602
$$

若 $\gamma=[1,1,1]$，输出约为：

$$
[0.463,0.926,1.389]
$$

它没有把均值移到 0，而是仅把“整体大小”压到稳定范围。白话说，RMSNorm 不在乎这组数是围绕 0 还是围绕 5，只在乎它们整体不要过大或过小。

### 3. 为什么 RMSNorm 还能工作

很多人第一次看到 RMSNorm 会觉得奇怪：不减均值，为什么还能稳定？

原因是，深层网络里最关键的常常不是“绝对中心在哪”，而是“幅度是否可控”。对于残差流和注意力计算，向量长度失控通常比均值偏移更危险。RMSNorm 保留了最重要的幅度控制，因此在大模型里常常够用。

从计算角度看，LayerNorm 需要：

1. 计算均值
2. 计算方差
3. 做中心化
4. 做仿射变换

RMSNorm 只需要：

1. 计算平方均值
2. 开平方或 rsqrt
3. 缩放

少掉中心化与 $\beta$，意味着更少的 reduce 操作和更少的参数同步，这就是它在大模型中受欢迎的根本原因。

### 4. 为什么要有 QK-Norm

注意力分数通常写作：

$$
\mathrm{Attn}(Q,K,V)=\mathrm{softmax}\left(\frac{QK^\top}{\sqrt{d}}\right)V
$$

虽然公式里已经有 $\sqrt{d}$ 缩放，但在长序列、大维度、混合精度训练下，Q 和 K 本身的范数仍可能波动很大。于是一些模型在 $Q,K$ 投影之后先做 RMSNorm，再送入点积。这样点积更接近“方向相似度”，而不是“方向相似度乘上长度放大”。

真实工程例子是 LLaMA、Qwen 一类大模型：主干 block 使用 pre-norm 的 RMSNorm，注意力部分进一步对 Q/K 做额外归一化。这样做的目标不是让表达能力变弱，而是让训练更稳、上下文更长时不容易数值爆炸。

---

## 代码实现

下面先给出一个不依赖 PyTorch 的可运行玩具实现，方便理解公式。它直接展示 LayerNorm 与 RMSNorm 的数值差别，并用 `assert` 验证结果性质。

```python
import math

def layer_norm(x, gamma=None, beta=None, eps=1e-5):
    d = len(x)
    mu = sum(x) / d
    var = sum((v - mu) ** 2 for v in x) / d
    denom = math.sqrt(var + eps)

    if gamma is None:
        gamma = [1.0] * d
    if beta is None:
        beta = [0.0] * d

    return [gamma[i] * ((x[i] - mu) / denom) + beta[i] for i in range(d)]

def rms_norm(x, gamma=None, eps=1e-6):
    d = len(x)
    rms = math.sqrt(sum(v * v for v in x) / d + eps)

    if gamma is None:
        gamma = [1.0] * d

    return [gamma[i] * (x[i] / rms) for i in range(d)]

x = [1.0, 2.0, 3.0]
ln = layer_norm(x)
rn = rms_norm(x)

# LayerNorm 输出均值应接近 0
assert abs(sum(ln) / len(ln)) < 1e-3

# RMSNorm 不强制零均值
assert sum(rn) / len(rn) > 0.8

# RMSNorm 的平方均值应接近 1
rms_sq = sum(v * v for v in rn) / len(rn)
assert abs(rms_sq - 1.0) < 1e-3

print("LayerNorm:", ln)
print("RMSNorm:", rn)
```

如果用 PyTorch，自定义 RMSNorm 通常也只需要几行。这里的 `rsqrt` 是“平方根的倒数”，常用于更高效地实现归一化。

```python
import torch
import torch.nn as nn

class RMSNorm(nn.Module):
    def __init__(self, dim, eps=1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim))
        self.eps = eps

    def forward(self, x):
        x_fp32 = x.float()
        scale = torch.rsqrt(x_fp32.pow(2).mean(dim=-1, keepdim=True) + self.eps)
        y = x_fp32 * scale
        return (y.to(dtype=x.dtype)) * self.weight
```

对应地，LayerNorm 直接使用官方模块即可：

```python
import torch.nn as nn

norm = nn.LayerNorm(embedding_dim, eps=1e-5, elementwise_affine=True)
```

在真实 Transformer block 中，常见写法是 pre-norm，也就是先归一化，再做注意力或 MLP：

```python
x = x + attention(rmsnorm1(x))
x = x + mlp(rmsnorm2(x))
```

这类结构的直觉很简单：残差主干尽量保持一条稳定通路，复杂变换放到归一化之后执行。

---

## 工程权衡与常见坑

工程上真正重要的不是“公式会不会背”，而是知道哪些配置一改就会出问题。

| 问题 | 现象 | 常见原因 | 处理方式 |
| --- | --- | --- | --- |
| `eps` 太小 | NaN、梯度爆炸 | 低精度下分母接近 0 | FP32 常用 `1e-5`，BF16/FP16 常用 `1e-6` |
| 大模型仍保留大量 bias | 参数更多、并行通信更重 | 每层多一组偏置同步 | 线性层常设 `bias=False` |
| 只做主干 norm，不做 QK-Norm | 长上下文注意力发散 | QK 点积随维度和长度放大 | 在 Q/K 投影后加 RMSNorm |
| 把归一化维度写错 | 模型能跑但效果差 | 沿 batch 或 seq 维归一化 | 明确沿最后一维归一化 |

### 1. `eps` 不是越小越好

$\epsilon$ 的作用，是防止分母过小。它不是数学上的装饰项，而是数值稳定性的保险丝。很多新手会觉得 `1e-8` 比 `1e-6` 更“精确”，这是错的。低精度训练里，过小的 `eps` 更容易被舍入影响，导致 `rsqrt` 或除法出现极端值。

一个实用经验是：

- FP32 常见 `eps=1e-5`
- BF16/FP16 的 RMSNorm 常见 `eps=1e-6`

这不是硬规则，但已经足够覆盖大多数训练场景。

### 2. bias 在大模型里经常被去掉

bias 就是线性层输出后的常数偏移项。对小模型来说，它通常没问题；但大模型中，很多层会去掉 bias，原因有两个：

1. 参数更少
2. 张量并行时通信更简单

如果主干已经有 RMSNorm，很多偏移效果可以由其他部分吸收，单独保留大量 bias 的收益就不一定高。

### 3. QK-Norm 解决的是注意力特有问题

主干中的 LayerNorm/RMSNorm 控制的是隐藏状态尺度；QK-Norm 控制的是注意力分数尺度。这两者不是互相替代，而是分别解决不同位置的不稳定。

一个真实工程场景是长上下文训练。序列变长后，注意力层更容易出现特别尖锐的 softmax。表现上可能是训练后期 loss 抖动、某些 head 异常饱和，甚至混合精度下直接溢出。此时只改优化器未必有效，给 Q 和 K 加 norm 往往更直接。

### 4. 维度写错比没有归一化更隐蔽

LayerNorm 和 RMSNorm 的标准做法，都是沿最后一维，即 hidden size 那一维计算统计量。代码能跑不代表写对了。如果错误地沿序列维归一化，你得到的就不是“每个 token 自己稳定”，而是“token 之间互相影响”。这类 bug 最麻烦，因为不一定报错，只会让训练效果变差。

---

## 替代方案与适用边界

LayerNorm 和 RMSNorm 都不是唯一选择，只是当前 Transformer 最常见的两种。

| 方案 | 核心想法 | 优点 | 局限 | 适用边界 |
| --- | --- | --- | --- | --- |
| LayerNorm | 零均值 + 单位方差 + 仿射 | 表达完整、经典稳定 | 计算略重 | 通用 Transformer、小中模型 |
| RMSNorm | 只控幅度，不做中心化 | 更轻，适合大模型 | 不提供 $\beta$ 偏移 | LLM、pre-norm 主干 |
| ScaleNorm | 只按向量范数缩放 | 形式更简单 | 实践普及度较低 | 研究型尝试 |
| QK-Norm | 专门归一化 Q/K | 稳定注意力 logits | 只解决注意力局部问题 | 长上下文、高维 attention |

什么时候坚持用 LayerNorm？

1. 你的模型规模不大，计算不是瓶颈。
2. 你需要完整的仿射能力，也就是同时保留 $\gamma$ 和 $\beta$。
3. 你沿用已有实现，希望与经典 Transformer 设置完全一致。

什么时候优先考虑 RMSNorm？

1. 你在做大语言模型或其变体。
2. 你使用 pre-norm 架构。
3. 你关心吞吐、显存和张量并行效率。

什么时候要额外加 QK-Norm？

1. 上下文长度很长。
2. 注意力 logits 波动明显。
3. 你已经发现 softmax 过尖、head 容易饱和。

需要强调的是，QK-Norm 不是“把主干 RMSNorm 换个位置”那么简单。它更像注意力内部的一道保险。主干 norm 管残差流，QK-Norm 管点积范围，两个位置的职责不同。

如果从零开始搭一个现代 LLM block，一个足够合理的默认组合通常是：主干用 RMSNorm，Q/K 上再做 RMS 风格归一化，多数线性层去 bias，`eps` 选 `1e-6` 或 `1e-5` 这一量级，而不是极端地调到很小。

---

## 参考资料

- PyTorch `torch.nn.LayerNorm` 官方文档：<https://docs.pytorch.org/docs/stable/generated/torch.nn.LayerNorm.html>
- PyTorch `torch.nn.RMSNorm` 官方文档：<https://docs.pytorch.org/docs/main/generated/torch.nn.RMSNorm.html>
- Meta torchtune 中的 RMSNorm 实现：<https://docs.pytorch.org/torchtune/0.0/_modules/torchtune/modules/rms_norm.html>
- Emergent Mind 对 Qwen-3 Transformer 架构的整理：<https://www.emergentmind.com/topics/qwen-3-transformer-architecture>
- Emergent Mind 对 Query-Key Normalization 的整理：<https://www.emergentmind.com/topics/query-key-normalization>
- CSDN 上关于 LayerNorm 数值例子的说明：<https://blog.csdn.net/flyfish1986/article/details/139488600>
