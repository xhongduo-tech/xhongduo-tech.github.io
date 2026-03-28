## 核心结论

LayerNorm 的作用是把一个 token 的特征向量先做“中心化再缩放”。中心化的白话解释是：先把整体偏高或偏低的公共偏移拿掉；缩放的白话解释是：再把整体幅度调到稳定范围。它的标准形式是：

$$
y = \frac{x - \mu}{\sqrt{\sigma^2 + \epsilon}} \cdot \gamma + \beta
$$

其中 $\mu,\sigma^2$ 都沿特征维计算，$\gamma,\beta$ 是可学习参数。

RMSNorm 是 LayerNorm 的简化版。它去掉了“减均值”这一步，只保留按均方根做缩放：

$$
y = \frac{x}{\sqrt{\mathrm{mean}(x^2) + \epsilon}} \cdot \gamma
$$

这里没有 $\beta$，通常也不做均值中心化。白话解释是：它只控制“大小”，不主动把向量搬回零均值附近。

在 Transformer 里，两者的核心目标都不是“提升表达能力”，而是“稳定训练和推理中的数值范围”。深层网络反复经过残差连接、注意力和前馈层，如果每层输入的尺度越来越飘，梯度就容易爆炸或塌陷。归一化层就是在每个残差分支前后，把数值拉回可控区间。

现代大语言模型的主流实践已经明显偏向 RMSNorm，尤其是在 pre-norm 结构里。原因有三点：

| 项目 | LayerNorm | RMSNorm |
|---|---|---|
| 是否减均值 | 是 | 否 |
| 是否有 $\beta$ | 通常有 | 通常无 |
| 计算统计量 | 均值 + 方差 | 仅均方根 |
| 参数量 | $2d$ | $d$ |
| 工程倾向 | 传统 Transformer 常用 | 现代 LLM 更常用 |

如果只记一条，可以记住这句：LayerNorm 更“完整”，RMSNorm 更“轻”，而现代深层 decoder-only Transformer 往往更需要后者的效率和稳定性平衡。

---

## 问题定义与边界

这篇文章讨论的是 Transformer 中“按 token、沿特征维”的归一化，不讨论 BatchNorm。BatchNorm 依赖 batch 统计量，而语言模型训练和推理里，序列长度、batch 组成、分布漂移都很强，不适合把不同样本绑在一起做归一化。

这里的“按 token、沿特征维”是什么意思？假设某一层输入张量形状是 $(B, T, d)$，也就是 batch 大小为 $B$，序列长度为 $T$，隐藏维度为 $d$。LayerNorm 或 RMSNorm 对每个位置 $(b,t)$ 的长度为 $d$ 的向量单独计算统计量，而不是跨 token 计算。也就是说，它解决的是“这个 token 的表示幅度是否稳定”，而不是“不同 token 之间是否对齐”。

Transformer 里真正关心的边界有三个：

1. 归一化放在哪里  
pre-norm 的白话解释是：先归一化，再进注意力或 FFN。post-norm 则是：先做子层计算，再归一化。现代深层 LLM 基本默认 pre-norm，因为梯度路径更稳定。

2. 归一化要不要中心化  
LayerNorm 认为“均值偏移”也该被消掉；RMSNorm 认为很多时候只控制尺度就够了，没必要付出额外计算。

3. 归一化是否要扩展到注意力内部  
这就引出 QK-Norm。QK-Norm 的白话解释是：在 query 和 key 投影之后，先把它们的长度控制住，再去做点积，避免注意力分数过大。

一个入门比喻可以这样理解，但只作为帮助直觉，不代替定义：每一层残差分支像一个接力站。归一化不是改路线，而是让每一棒交接时速度都在可控范围内。LayerNorm 连“起跑线偏移”也一起修正；RMSNorm 主要关心“速度别失控”。

真实工程里，这个问题尤其明显于两类场景：

| 场景 | 主要风险 | 常见做法 |
|---|---|---|
| 深层 Transformer 训练 | 激活和梯度尺度漂移 | pre-LN 或 pre-RMSNorm |
| 低精度训练/推理 | 数值下溢、上溢、NaN | 合理设置 $\epsilon$，并常配合 RMSNorm |
| 长上下文注意力 | $QK^\top$ logit 过大 | 在 Q/K 后加 QK-Norm |
| Tensor Parallel | 参数和通信开销上升 | 去掉不必要的 bias 和 $\beta$ |

所以本文边界很明确：讨论的是 Transformer 中 LayerNorm 和 RMSNorm 的数学形式、数值作用、代码实现和工程选择，不讨论别的归一化家族。

---

## 核心机制与推导

先看 LayerNorm。给定一个 token 的特征向量：

$$
x = [x_1, x_2, \dots, x_d]
$$

它先计算均值和方差：

$$
\mu = \frac{1}{d}\sum_{i=1}^d x_i,\qquad
\sigma^2 = \frac{1}{d}\sum_{i=1}^d (x_i-\mu)^2
$$

然后输出：

$$
y_i = \frac{x_i-\mu}{\sqrt{\sigma^2+\epsilon}}\gamma_i + \beta_i
$$

这里 $\gamma_i,\beta_i$ 都是每个维度各自的可学习参数。直观上，减去 $\mu$ 会让输出均值接近 0，除以标准差会让输出方差接近 1，然后再让模型自己学“该放大多少、平移多少”。

再看 RMSNorm。它不关心均值，只关心向量整体能量：

$$
\mathrm{RMS}(x)=\sqrt{\frac{1}{d}\sum_{i=1}^d x_i^2}
$$

于是输出为：

$$
y_i=\frac{x_i}{\mathrm{RMS}(x)+\text{tiny}}\gamma_i
$$

更常写成：

$$
y_i=\frac{x_i}{\sqrt{\mathrm{mean}(x^2)+\epsilon}}\gamma_i
$$

关键差异是：LayerNorm 处理“偏移 + 尺度”，RMSNorm 只处理“尺度”。

这个差异可以用一个玩具例子看清楚。设输入向量为：

$$
x=[1,2,3,4]
$$

LayerNorm：
- 均值 $\mu=2.5$
- 方差 $\sigma^2=1.25$
- 标准差 $\sigma\approx1.118$

归一化后约为：

$$
[-1.342,-0.447,0.447,1.342]
$$

如果先不考虑 $\gamma,\beta$，这个结果的均值是 0，幅度也被标准化了。

RMSNorm：
- $\mathrm{mean}(x^2)=\frac{1+4+9+16}{4}=7.5$
- $\mathrm{RMS}(x)=\sqrt{7.5}\approx2.739$

归一化后约为：

$$
[0.365,0.730,1.095,1.461]
$$

它没有把中心移到 0，只是把原向量按统一比例缩小。也就是说，RMSNorm 保留了“整体偏移”的信息。

再看一个更紧凑的玩具例子：$x=[2,4,6]$。

| 方法 | 统计量 | 输出特征 |
|---|---|---|
| LayerNorm | $\mu=4,\sigma\approx1.633$ | 约 $[-1.225,0,1.225]$ |
| RMSNorm | $\mathrm{RMS}\approx4.320$ | 约 $[0.463,0.926,1.389]$ |

为什么 RMSNorm 在大模型里通常够用？因为深层 Transformer 中，最直接的问题往往不是“均值没对齐”，而是“幅度越来越失控”。尤其在 pre-norm 架构下，残差主路径一直保留原始信号，归一化子层主要负责给注意力和 FFN 提供尺度稳定的输入。此时只做尺度控制，往往已经足够。

再往前推一步，可以解释 QK-Norm 为什么会出现。注意力 logits 的核心是：

$$
\text{logits} = \frac{QK^\top}{\sqrt{d_k}}
$$

如果 $Q$ 和 $K$ 的每维方差都稳定，但向量范数随着维度增长而变大，那么点积的方差仍可能随着 $d_k$ 增长而偏大，长上下文和低精度下尤其容易出现大 logit。softmax 一旦过尖，就会接近 one-hot，训练和推理都更脆弱。QK-Norm 的作用就是在 $Q,K$ 投影后先做一次定幅，再去点积，把这个风险压下来。

真实工程例子是 LLaMA 系列和 Qwen 系列。前者把 RMSNorm 作为标准配置，说明“只控制尺度”在深层 LLM 中是足够稳定的；后者进一步在 Q/K 上做额外归一化，针对的是注意力内部而不是残差入口的数值爆炸问题。这两种做法并不冲突，解决的是两个不同层次的问题。

---

## 代码实现

先给一个最小可运行的 Python 版本，直接演示 LayerNorm 和 RMSNorm 的区别。代码里只用标准库和 `numpy`，并用 `assert` 验证关键性质。

```python
import numpy as np

def layer_norm(x, gamma=None, beta=None, eps=1e-5):
    x = np.asarray(x, dtype=np.float64)
    mean = x.mean(axis=-1, keepdims=True)
    var = ((x - mean) ** 2).mean(axis=-1, keepdims=True)
    y = (x - mean) / np.sqrt(var + eps)

    if gamma is None:
        gamma = np.ones_like(x)
    if beta is None:
        beta = np.zeros_like(x)

    return y * gamma + beta

def rms_norm(x, gamma=None, eps=1e-5):
    x = np.asarray(x, dtype=np.float64)
    rms = np.sqrt((x ** 2).mean(axis=-1, keepdims=True) + eps)
    y = x / rms

    if gamma is None:
        gamma = np.ones_like(x)

    return y * gamma

x = np.array([1.0, 2.0, 3.0, 4.0])

ln = layer_norm(x)
rn = rms_norm(x)

# LayerNorm 输出应接近零均值
assert abs(ln.mean()) < 1e-4

# RMSNorm 不保证零均值
assert abs(rn.mean()) > 0.1

# RMSNorm 输出的均方根应接近 1
assert abs(np.sqrt((rn ** 2).mean()) - 1.0) < 1e-4

print("LayerNorm:", np.round(ln, 3))
print("RMSNorm:", np.round(rn, 3))
```

如果换成 PyTorch，核心实现也非常直接。LayerNorm 需要两类统计量：均值和方差；RMSNorm 只需要平方均值，因此实现通常更短，也更容易和低精度计算结合。

```python
import torch
import torch.nn as nn

class SimpleLayerNorm(nn.Module):
    def __init__(self, dim, eps=1e-5):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim))
        self.bias = nn.Parameter(torch.zeros(dim))
        self.eps = eps

    def forward(self, x):
        x_fp32 = x.float()
        mean = x_fp32.mean(dim=-1, keepdim=True)
        var = (x_fp32 - mean).pow(2).mean(dim=-1, keepdim=True)
        y = (x_fp32 - mean) * torch.rsqrt(var + self.eps)
        y = y.to(dtype=x.dtype)
        return y * self.weight + self.bias

class SimpleRMSNorm(nn.Module):
    def __init__(self, dim, eps=1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim))
        self.eps = eps

    def forward(self, x):
        x_fp32 = x.float()
        rms_inv = torch.rsqrt(x_fp32.pow(2).mean(dim=-1, keepdim=True) + self.eps)
        y = (x_fp32 * rms_inv).to(dtype=x.dtype)
        return y * self.weight
```

这里有两个实现细节很重要。

第一，很多实现会先转成 `float32` 再算统计量。原因是 BF16/FP16 的有效精度更低，直接在低精度上做平方、均值和开根号，更容易放大误差。

第二，RMSNorm 一般没有 bias。这不只是“少几个参数”那么简单。在线性层和并行训练里，bias 往往意味着额外的内存访问和通信路径。大模型训练的瓶颈常常是带宽，不是标量加法本身。

一个真实工程例子可以这样看。假设你在实现一个 decoder block：

```python
h = x + self.attn(self.norm1(x))
y = h + self.mlp(self.norm2(h))
```

如果 `norm1/norm2` 用 RMSNorm，这就是现代 LLM 常见的 pre-norm 写法。它的目标不是让 `x` 变“更有语义”，而是让 `attn` 和 `mlp` 每次看到的输入尺度都更稳定，从而把训练难度控制在可优化范围内。

---

## 工程权衡与常见坑

工程里真正麻烦的地方不在公式，而在数值细节和系统成本。

先看最常见的几个选择：

| 选择项 | 常见配置 | 原因 |
|---|---|---|
| 归一化类型 | 深层 LLM 多用 RMSNorm | 计算更省，稳定性通常不差 |
| bias | 常去掉 | 减少参数、访存和并行通信 |
| $\epsilon$ | FP32 常见 `1e-5` | 更稳妥的数值底线 |
| $\epsilon$ | BF16/FP16 常见 `1e-6` 或 `1e-5` | 需结合实现验证稳定性 |
| QK-Norm | 长上下文常加入 | 压制 attention logit 爆炸 |

这里需要特别说明 $\epsilon$。它是防止分母过小的保护项，不是越小越“精确”。当输入幅度很小、又在低精度下做平方和开根号时，过小的 $\epsilon$ 可能让分母接近 0，最终导致输出过大甚至 NaN。反过来，如果 $\epsilon$ 设得过大，归一化又会变钝，等于人为改变了尺度。

一个常见排查表如下：

| 现象 | 优先检查项 | 处理建议 |
|---|---|---|
| 训练或推理出现 NaN | norm 的 $\epsilon$ | 先尝试 `1e-6` 到 `1e-5` |
| 注意力极度尖锐 | Q/K 范数是否失控 | 在 Q/K 投影后加 QK-Norm |
| 长上下文性能突然变差 | attention logits 是否过大 | 检查 `QK^T / sqrt(d)` 前的幅度 |
| 多卡吞吐不理想 | bias 和额外参数 | 去掉不必要 bias，减少同步负担 |

再说“LayerNorm 是否更准”。这要分场景。LayerNorm 确实多做了一步均值中心化，所以在某些任务上，它对输入分布偏移更敏感，也更有表达上的“修正力”。但现代深层 LLM 的瓶颈通常不是“缺少中心化”，而是“如何在深层、长序列、低精度下把计算跑稳”。因此 RMSNorm 往往更符合系统目标。

真实工程里有两个坑最容易踩。

第一个坑是把“RMSNorm 更省”理解成“任何地方都该替换”。不对。若你的模型较浅、任务分布复杂、输入模态差异大，LayerNorm 仍然可能更稳，因为它确实多做了一个偏移修正。

第二个坑是把“归一化稳定”误解成“什么数值问题都能靠 norm 解决”。也不对。注意力内部的 $QK^\top$ 放大问题，和残差入口的归一化不是同一个层次。你即便用了 RMSNorm，长上下文下仍可能需要 QK-Norm。

给初学者一个实用排查顺序：

1. 先确认模型是 pre-norm 还是 post-norm。深层 Transformer 优先排查 post-norm 带来的不稳定。
2. 再看归一化实现是否在 `float32` 上统计。
3. 再调 $\epsilon$，不要上来先改学习率。
4. 如果问题集中在长上下文注意力，再看 QK-Norm，而不是只盯着 LayerNorm/RMSNorm。

---

## 替代方案与适用边界

如果把选择问题压缩成一句话：LayerNorm 是更通用的经典方案，RMSNorm 是更符合现代大模型工程目标的高效方案，QK-Norm 是注意力内部的专项稳定化手段。

可以把几种方案放在一起比较：

| 方法 | 是否中心化 | 可学习参数 | 主要解决问题 | 典型适用场景 |
|---|---|---|---|---|
| LayerNorm | 是 | $\gamma,\beta$ | 控制偏移和尺度 | 通用 Transformer、较浅网络、多模态场景 |
| RMSNorm | 否 | $\gamma$ | 控制尺度 | 深层 LLM、pre-norm、低精度训练/推理 |
| pRMSNorm | 否 | $\gamma$ | 用部分特征估计 RMS，进一步省算力 | 极端追求效率的研究或定制实现 |
| QK-Norm | 通常对 Q/K 定幅 | 视实现而定 | 控制注意力 logit | 长上下文、低精度、移动端推理 |

什么时候优先用 LayerNorm？
- 模型层数不深，但你希望输入分布修正更充分。
- 任务或模态变化较大，均值偏移可能本身就是噪声来源。
- 你更在意标准化语义完整性，而不是极限吞吐。

什么时候优先用 RMSNorm？
- 你在做 decoder-only 大模型。
- 结构采用 pre-norm。
- 训练和推理都受限于吞吐、带宽和低精度稳定性。
- 你希望减少参数和通信负担，同时不明显损失效果。

什么时候还要额外加 QK-Norm？
- 上下文长度明显变长。
- attention logits 分布越来越尖。
- FP16/BF16 或端侧推理时更容易出现不稳定。

最后给一个简化选型表，适合工程决策时快速对照：

| 需求 | 更合适的方案 |
|---|---|
| 少层、通用、想要完整标准化 | LayerNorm |
| 深层 LLM、追求吞吐和稳定 | RMSNorm |
| 长上下文注意力不稳 | RMSNorm + QK-Norm |
| 多卡并行想减通信 | RMSNorm + bias-free linear |

所以“LayerNorm 还是 RMSNorm”不是谁绝对更强，而是谁更符合当前架构的主要矛盾。对现代 Transformer 来说，这个主要矛盾通常是深度、带宽、低精度和长上下文，而不是单层表示是否严格零均值。

---

## 参考资料

- PyTorch `LayerNorm` 文档：https://docs.pytorch.org/docs/main/generated/torch.nn.modules.normalization.LayerNorm.html  
- PyTorch `RMSNorm` 文档：https://docs.pytorch.org/docs/main/generated/torch.nn.RMSNorm.html  
- Sebastian Raschka, RMSNorm vs LayerNorm FAQ：https://sebastianraschka.com/faq/docs/rmsnorm-vs-layernorm.html  
- RMSNorm 论文综述入口：https://liner.com/review/root-mean-square-layer-normalization  
- QK-Norm 工程分析案例：https://www.langcopilot.com/posts/2025-06-26-qwen3-qk-norm-improved-on-device-ai-stability  
- 预归一化 Transformer 综述入口：https://www.emergentmind.com/topics/pre-layernorm-transformers
