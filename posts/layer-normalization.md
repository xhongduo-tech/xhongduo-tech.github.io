## 核心结论

LayerNorm 是 Transformer 中最常用的归一化方法。归一化，白话说，就是把一组数重新拉回一个更稳定、更容易训练的范围。它的关键点不是“跨样本”看统计量，而是“在单个样本内部”看统计量。对一个 token 的隐藏向量 $x \in \mathbb{R}^D$，LayerNorm 在特征维度上计算均值和方差：

$$
\mu = \frac{1}{D}\sum_{i=1}^{D} x_i,\quad
\sigma^2 = \frac{1}{D}\sum_{i=1}^{D}(x_i-\mu)^2
$$

然后输出

$$
y_i = \gamma_i \frac{x_i-\mu}{\sqrt{\sigma^2+\epsilon}} + \beta_i
$$

其中 $\gamma,\beta$ 是可学习参数，白话说就是“归一化之后再交给模型自己调节缩放和偏移”。

对 Transformer 来说，这种设计有两个直接价值：

| 结论 | 含义 |
|---|---|
| 不依赖 batch size | batch 可以很小，甚至为 1 |
| 适合序列模型 | 每个 token 独立归一化，不破坏变长输入 |
| 推理和训练一致 | 不需要像 BatchNorm 那样维护运行时统计量 |

在结构选择上，现代经验已经很明确：

| 结构 | 位置 | 常见结论 |
|---|---|---|
| Post-LN | 子层计算后，再残差相加后归一化 | 早期 Transformer/BERT 常见，但深层训练更难 |
| Pre-LN | 先归一化，再进入注意力或 FFN 子层 | 梯度更稳定，现代大模型更常见 |

再往前一步，很多大语言模型直接把 LayerNorm 换成 RMSNorm。RMSNorm 的白话解释是：它保留“缩放到稳定范围”这件事，但不再强制减去均值。LLaMA、Qwen、Mistral 这类模型常见组合是 Pre-LN 思路加 RMSNorm，因为训练更稳，算子也更轻。

---

## 问题定义与边界

先明确问题：为什么 Transformer 不直接用 BatchNorm？

BatchNorm 是“批归一化”，白话说，就是把一个 batch 里的多个样本放在一起统计均值和方差。它在 CNN 里很成功，但在序列模型里有天然限制：

| 维度 | BatchNorm | LayerNorm |
|---|---|---|
| 统计范围 | 跨样本统计 | 单样本内统计 |
| 是否依赖 batch size | 依赖 | 不依赖 |
| batch=1 是否稳定 | 很差 | 可以 |
| 变长序列是否方便 | 不方便 | 方便 |
| 推理阶段 | 依赖运行统计量 | 直接算当前输入 |

Transformer 的基本对象不是整张图，而是 token 向量。假设一个 token 的隐藏状态是 $x=[2,5,-1]$。LayerNorm 只关心这 3 个特征本身，不需要其他 token，也不需要同 batch 的其他样本。它问的是：“这个 token 内部，三个维度的数值分布是否过大、过偏？”而不是“这个 batch 里大家平均是多少？”

这就决定了它的边界：

1. LayerNorm 的归一化对象是“每个 token 的最后一维特征”。
2. 它不会让不同 token 共享统计量，所以不会引入 batch 波动。
3. 它不是在解决“语义对齐”问题，而是在解决“数值尺度稳定”问题。
4. 它不会凭空提升表达能力，它主要作用是让优化更容易。

玩具例子可以直接看出边界。假设两个 token：

- token A: $[2,5,-1]$
- token B: $[100,101,99]$

LayerNorm 会分别对 A 和 B 单独归一化。即使 B 的绝对值整体大很多，也不会影响 A 的归一化结果。这正是 Transformer 需要的性质，因为序列里每个位置都应独立稳定，而不是互相拖累。

---

## 核心机制与推导

LayerNorm 的核心机制可以拆成三步：

1. 计算均值 $\mu$
2. 计算方差 $\sigma^2$
3. 标准化后再乘 $\gamma$、加 $\beta$

设输入向量为 $x=[x_1,x_2,\dots,x_D]$，则：

$$
\mu = \frac{1}{D}\sum_{i=1}^{D}x_i
$$

$$
\sigma^2 = \frac{1}{D}\sum_{i=1}^{D}(x_i-\mu)^2
$$

$$
\hat{x}_i = \frac{x_i-\mu}{\sqrt{\sigma^2+\epsilon}}
$$

$$
y_i = \gamma_i\hat{x}_i+\beta_i
$$

其中 $\epsilon$ 是一个很小的正数，白话说，就是防止分母接近 0 导致数值炸掉。

### 玩具例子：手算 LayerNorm 与 RMSNorm

输入：

$$
x=[2,5,-1]
$$

先算 LayerNorm。

均值：

$$
\mu = \frac{2+5+(-1)}{3}=2
$$

方差：

$$
\sigma^2=\frac{(2-2)^2+(5-2)^2+(-1-2)^2}{3}
=\frac{0+9+9}{3}=6
$$

标准差：

$$
\sigma=\sqrt{6}\approx 2.449
$$

所以标准化结果约为：

$$
\hat{x} = \left[0,\frac{3}{2.449},\frac{-3}{2.449}\right]
\approx [0,1.225,-1.225]
$$

如果取 $\gamma=[1,1,1], \beta=[0,0,0]$，输出就是：

$$
[0,1.225,-1.225]
$$

如果取 $\gamma=[2,2,2], \beta=[0.5,0.5,0.5]$，输出变成：

$$
[0.5, 2.95, -1.95]
$$

这说明 LayerNorm 并不是把表示“锁死”在零均值单位方差，而是先标准化，再交给模型学习一个更适合任务的尺度。

再看 RMSNorm。它的定义是：

$$
\text{RMS}(x)=\sqrt{\frac{1}{D}\sum_{i=1}^{D}x_i^2}
$$

$$
y_i=\gamma_i \frac{x_i}{\text{RMS}(x)+\epsilon}
$$

对同一个例子：

$$
\text{RMS}(x)=\sqrt{\frac{2^2+5^2+(-1)^2}{3}}
=\sqrt{\frac{30}{3}}=\sqrt{10}\approx 3.162
$$

所以在 $\gamma=[1,1,1]$ 时：

$$
y \approx [0.632,1.581,-0.316]
$$

两者差异很清楚：

| 方法 | 是否减均值 | 是否按方差缩放 | 可学习偏置 |
|---|---|---|---|
| LayerNorm | 是 | 是 | 常有 $\beta$ |
| RMSNorm | 否 | 用 RMS 缩放 | 通常无 $\beta$ |

直观上，LayerNorm 强调“居中再缩放”；RMSNorm 强调“只控制幅度，不强制居中”。

### 为什么 Pre-LN 更稳

Transformer 子层可以抽象成：

$$
x_{l+1} = x_l + F(x_l)
$$

这里的残差连接，白话说，就是“原输入直接绕过复杂模块，和模块输出相加”。

Post-LN 写法近似是：

$$
x_{l+1} = \text{LN}(x_l + F(x_l))
$$

Pre-LN 写法近似是：

$$
x_{l+1} = x_l + F(\text{LN}(x_l))
$$

两者的差别不只是“位置不同”。关键在梯度路径。Pre-LN 中，残差主干上始终保留一条更接近恒等映射的通道，梯度更容易直接传回前层；Post-LN 则把归一化放在残差之后，深层时更容易在训练初期出现梯度不稳，因此通常更依赖 warm-up。warm-up，白话说，就是训练刚开始时先用更小学习率慢慢升起来。

真实工程里，这个差异很重要。12 层小模型有时两种都能训，但层数一深、学习率一大、batch 一复杂，Pre-LN 往往更省事。

---

## 代码实现

下面用纯 Python 写一个最小可运行实现。重点是“沿最后一维做统计”，而不是对整个矩阵一起算。

```python
import math

def layer_norm(x, gamma=None, beta=None, eps=1e-5):
    n = len(x)
    if gamma is None:
        gamma = [1.0] * n
    if beta is None:
        beta = [0.0] * n

    mean = sum(x) / n
    var = sum((v - mean) ** 2 for v in x) / n
    denom = math.sqrt(var + eps)

    y = [g * ((v - mean) / denom) + b for v, g, b in zip(x, gamma, beta)]
    return y

def rms_norm(x, gamma=None, eps=1e-5):
    n = len(x)
    if gamma is None:
        gamma = [1.0] * n

    rms = math.sqrt(sum(v * v for v in x) / n + eps)
    y = [g * (v / rms) for v, g in zip(x, gamma)]
    return y

# 玩具例子
x = [2.0, 5.0, -1.0]

ln_y = layer_norm(x)
rms_y = rms_norm(x)

# LayerNorm 输出均值应接近 0
assert abs(sum(ln_y) / len(ln_y)) < 1e-3

# RMSNorm 不保证零均值，但会控制尺度
rms = math.sqrt(sum(v * v for v in rms_y) / len(rms_y))
assert abs(rms - 1.0) < 1e-3

# 带可学习参数
ln_y2 = layer_norm(x, gamma=[2.0, 2.0, 2.0], beta=[0.5, 0.5, 0.5])
assert len(ln_y2) == 3

print("LayerNorm:", [round(v, 4) for v in ln_y])
print("RMSNorm:", [round(v, 4) for v in rms_y])
print("LayerNorm with affine:", [round(v, 4) for v in ln_y2])
```

如果写成 Transformer 风格伪代码，通常是这样：

```python
def transformer_block_pre_ln(x):
    x = x + attention(layer_norm(x))
    x = x + ffn(layer_norm(x))
    return x

def transformer_block_post_ln(x):
    x = layer_norm(x + attention(x))
    x = layer_norm(x + ffn(x))
    return x
```

真实工程例子更接近下面这个判断：

- 做 BERT 风格复现，通常还能见到 Post-LN。
- 做现代 LLM 训练，尤其是几十层到上百层，默认会优先尝试 Pre-LN。
- 如果目标是 LLaMA/Qwen/Mistral 风格结构，通常直接使用 RMSNorm，并把它放在 attention/FFN 前面。

原因不是“社区流行”，而是这组选择在稳定性、吞吐和调参成本上更平衡。

---

## 工程权衡与常见坑

最常见的误解是：LayerNorm 只是一个公式，放哪里都一样。这个结论是错的。归一化的位置直接影响训练动力学，也就是优化过程中的梯度传播方式。

### Pre-LN vs Post-LN 的工程对比

| 维度 | Pre-LN | Post-LN |
|---|---|---|
| 训练初期稳定性 | 更好 | 更差 |
| 对 warm-up 依赖 | 较弱 | 较强 |
| 深层网络可扩展性 | 更强 | 更弱 |
| 早期论文兼容性 | 略差 | 更接近原始 Transformer/BERT |
| 现代大模型使用频率 | 高 | 低 |

### 常见坑 1：归一化维度算错

LayerNorm 不是对整个 batch 全局算均值，也不是把序列长度一起算进去。对形状 `[batch, seq, d_model]` 的张量，通常应在 `d_model` 维度上计算。算错维度时，模型可能不会立刻报错，但训练会异常慢，甚至 loss 不下降。

### 常见坑 2：把 LayerNorm 当成“防爆万能药”

它能缓解激活尺度不稳定，但不能代替学习率设计、初始化设计和残差结构设计。尤其是 Post-LN 结构，如果学习率过大、warm-up 过短，依然容易不稳。

### 常见坑 3：忽略 $\epsilon$

$\epsilon$ 太小，低精度训练时可能不够稳；太大，又会改变归一化尺度。实际工程一般用如 `1e-5`、`1e-6` 一类默认值，不随意乱改。

### 常见坑 4：误解 RMSNorm 的作用

RMSNorm 不是“更准确的 LayerNorm”，而是“更便宜的稳定化方法”。它去掉均值中心化后，计算更简单，也少了一部分统计约束。对现代大模型，这通常是可接受甚至有利的；但在某些小模型、特殊任务或已有成熟配置中，直接替换不一定更优。

### 真实工程例子

BERT 原始设计是典型 Post-LN 路线，因此训练时通常比较依赖 warm-up。这个现象在深层 Transformer 中更加明显。后续很多工作指出，把 LN 移到子层之前，即 Pre-LN，残差路径上的梯度更顺，训练初期更稳定。再进一步，大模型为了减少计算和内存开销，经常把 LayerNorm 换成 RMSNorm。于是现代主流组合逐渐演化成：

- Pre-LN + LayerNorm
- Pre-LN + RMSNorm

第二种在大语言模型里最常见。

---

## 替代方案与适用边界

LayerNorm 不是唯一选择，但它是最稳妥的默认选择。是否替换，主要看模型规模、训练深度和工程目标。

| 方案 | 优点 | 缺点 | 适用场景 |
|---|---|---|---|
| LayerNorm | 机制成熟，行为直观 | 计算略重 | 通用 Transformer、小中型模型 |
| RMSNorm | 更轻，深层训练常更稳 | 不做均值中心化 | 大模型、效率敏感场景 |
| BatchNorm | CNN 中效果好 | 不适合变长序列和小 batch | 卷积网络 |
| No-Norm/弱归一化 | 可能更快 | 很难训稳 | 特殊研究实验 |

适用边界可以直接概括成三条：

1. 如果是 12 层左右的小型 Transformer，LayerNorm 往往已经足够，不必为了一点理论简化强行换 RMSNorm。
2. 如果是几十层到上百层的大模型，且训练稳定性和吞吐都重要，优先考虑 Pre-LN + RMSNorm。
3. 如果任务特别依赖已有成熟配置，比如复现某篇论文或某个开源模型，就优先遵循原结构，不要盲改归一化。

还有一个容易忽略的点：LayerNorm 中的 $\beta$ 有时是有价值的。它允许模型在归一化之后重新引入偏移量。对一些小模型或特征分布比较敏感的任务，这种自由度未必应该轻易删除。因此“RMSNorm 更现代”不等于“LayerNorm 已经过时”。

真正正确的判断标准不是“哪个更新”，而是：

- 模型有多深
- 训练是否不稳
- 是否受吞吐和显存约束
- 是否需要和现有 checkpoint、论文结构对齐

---

## 参考资料

| 来源 | 侧重点 | 关键词 |
|---|---|---|
| Ba et al., Layer Normalization | LayerNorm 原始定义与公式 | 公式、样本内归一化 |
| Xiong et al., On Layer Normalization in the Transformer Architecture | Pre-LN 与 Post-LN 的训练稳定性分析 | 梯度、warm-up、稳定性 |
| afloresep 的 LayerNorm 解读 | 面向工程实现的公式拆解 | 均值、方差、最后一维 |
| eventum 关于现代 Transformer 演化的文章 | Pre-LN 与 Post-LN 的直观区别 | 残差、训练稳定 |
| ai.towerofrecords 的归一化说明 | RMSNorm 基本定义 | RMS、简化归一化 |
| mbrenndoerfer 关于 RMSNorm 的文章 | RMSNorm 在现代 LLM 中的工程意义 | LLaMA、效率、实践 |
| Metric Coders 对 LLaMA 架构的拆解 | Pre-Norm 与 RMSNorm 的真实模型用法 | LLaMA、子层前归一化 |

1. Ba, Kiros, Hinton. Layer Normalization.
2. Xiong et al. On Layer Normalization in the Transformer Architecture.
3. LayerNorm 公式与样本内归一化机制解读。
4. Pre-LN vs Post-LN 的训练稳定性分析。
5. RMSNorm 在现代 LLM 中的工程实践总结。
