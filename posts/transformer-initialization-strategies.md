## 核心结论

Transformer 不能只套用“线性层用 Xavier/He”这一套经验，因为它的数值风险不是单个矩阵乘法，而是三件事同时发生：

1. 注意力里的 $QK^\top$ 会把向量长度直接放大到 softmax logits 上。logits 一旦过大，softmax 就会接近 one-hot，绝大多数位置的梯度都会变得极小。
2. 残差路径会把很多层的输出持续叠加。层数一深，前向激活和反向梯度都可能累积失控。
3. LayerNorm 的缩放参数 $\gamma$ 如果一开始就给满强度，等于默认每层都“放行”。深层模型在 warmup 早期更容易振荡。

所以，Transformer 的初始化更像“分组件做尺度控制”，而不是“全模型统一方差”：

| 组件 | 缩放手段 | 直接作用 | 目标 |
| --- | --- | --- | --- |
| QK 点乘 | $\frac{1}{\sqrt{d_k}}$、QK-Norm、QKV-Norm、logit cap | 控制 attention logits 的量级 | 抑制 softmax 饱和 |
| 残差分支 | 残差投影权重乘 $1/\sqrt{L}$ 或等价缩小末层投影 std | 控制每层增量对主干的冲击 | 抵消深度累加 |
| LayerNorm | $\gamma=1$ 改为更小值，或按层宽/深度设计 | 限制归一化后的初始放大能力 | 提前压住梯度增益 |

一个最小玩具例子：12 层 Transformer、每头 $d_k=64$、LayerNorm 宽度 $N_l=1024$。

- attention 缩放：$\frac{1}{\sqrt{64}}=0.125$
- 残差缩放：$\frac{1}{\sqrt{12}}\approx 0.289$
- 若用保守版 LayerNorm 缩放，取 $\alpha=0.5$，则
  $$
  \gamma^{(l)}=\alpha\sqrt{\frac{2}{N_l}}=0.5\sqrt{\frac{2}{1024}}\approx 0.022
  $$

这三个数可以直接白话化理解：

- 先把每层残差增量压小，再往主干上叠加。
- 先把 QK logits 压到可训练区间，再送进 softmax。
- 先把 LayerNorm 的放大量压小，再让训练自己学回合适强度。

如果只记一句话，可以记成：

> Transformer 的初始化不是“让每个矩阵看起来方差正常”，而是“让 attention、residual、normalization 三条数值链路都先别太激进”。

---

## 问题定义与边界

这里讨论的“初始化策略”不是泛指“权重怎么随机采样”，而是专门讨论 Transformer 里会影响训练稳定性的三类初始化：

1. 注意力投影矩阵 $W_Q,W_K,W_V,W_O$
2. FFN 的两层线性矩阵
3. LayerNorm 的 $\gamma,\beta$

标准自注意力写成：

$$
Q=XW_Q,\quad K=XW_K,\quad V=XW_V
$$

$$
\mathrm{Attention}(Q,K,V)=\mathrm{softmax}\left(\frac{QK^\top}{\sqrt{d_k}}\right)V
$$

只看这两个公式，很容易得到一个误判：既然已经除以 $\sqrt{d_k}$，为什么还会数值不稳定？

关键在于，$\frac{1}{\sqrt{d_k}}$ 只修正了“维度变大导致点积方差变大”的平均效应，但它并不约束训练过程中 $Q,K$ 的模长。更具体地说：

$$
q^\top k = \|q\|_2 \|k\|_2 \cos\theta
$$

即使 $\cos\theta$ 还在正常范围内，只要 $\|q\|_2$ 和 $\|k\|_2$ 同时变大，logits 仍然会被推高。softmax 在大正数和大负数输入下会迅速饱和，表现为：

- 最大位置概率接近 1
- 其余位置概率接近 0
- 梯度主要集中在极少数位置
- 小扰动就可能带来训练不稳定或 loss 抖动

另一个问题来自残差路径：

$$
x_{l+1}=x_l+F_l(x_l)
$$

这条式子本身很有价值，因为残差让优化更容易；但它也意味着每层都会往主干上继续加一份增量。如果连续很多层都这么做，而且每层的增量方差不小，那么总信号的尺度就会跟层数一起涨。

一个常见的近似理解是：

- 若每层残差增量的均值接近 0、方差接近 $\sigma^2$
- 且层间相关性不是特别强

那么堆到第 $L$ 层后，增量部分的总方差会近似包含 $L\sigma^2$ 这一项。模型浅时，这件事不明显；模型深、batch 小、学习率激进、warmup 短时，问题会被放大。

再看 LayerNorm。LayerNorm 的计算是：

$$
\mathrm{LN}(x)=\gamma \odot \frac{x-\mu}{\sqrt{\sigma^2+\epsilon}}+\beta
$$

其中：

- $\mu,\sigma^2$ 是当前 token 特征维度上的均值和方差
- $\gamma$ 是可学习缩放参数
- $\beta$ 是可学习平移参数

多数框架默认：

$$
\gamma=1,\quad \beta=0
$$

这在浅层网络通常没问题，但在深层 Transformer 里，它等价于“归一化后立刻允许每层以满强度输出”。如果残差和注意力本来就偏激进，这个默认值会进一步放大 warmup 早期的不稳定。

边界也要说清楚：

| 场景 | 主要风险 | 初始化的重要性 |
| --- | --- | --- |
| 6 到 12 层、小模型、常规学习率 | 风险较低 | 通常不是首要矛盾 |
| 24 到 48 层、decoder-only、较大 batch 波动 | 残差累加开始明显 | 明显重要 |
| 上百层、长上下文、混合精度、小 batch | attention 饱和 + 深度累加 + 数值精度问题叠加 | 往往决定“能不能训” |

还有两个边界容易混淆：

- BERT、GPT-2、GPT-3 的原始方案不等同于“都用了 QK-Norm”。QK-Norm 是后续更系统的稳定化改进。
- “Pre-LN 能训练”不等于“初始化不重要”。Pre-LN 只是把最危险的一部分问题压下去，不是把所有数值风险消掉。

---

## 核心机制与推导

先看注意力 logits 的尺度。

设单个 head 中的 query、key 为 $q,k\in \mathbb{R}^{d_k}$，并假设各维近似独立、零均值、方差为 $\sigma_q^2,\sigma_k^2$。未缩放点积为：

$$
s=q^\top k=\sum_{i=1}^{d_k} q_i k_i
$$

因为独立近似下：

$$
\mathrm{Var}(q_i k_i)=\mathrm{Var}(q_i)\mathrm{Var}(k_i)=\sigma_q^2\sigma_k^2
$$

所以：

$$
\mathrm{Var}(s)\approx d_k \sigma_q^2 \sigma_k^2
$$

这就是原始 Transformer 要除以 $\sqrt{d_k}$ 的原因：

$$
\tilde s=\frac{q^\top k}{\sqrt{d_k}}
\quad\Rightarrow\quad
\mathrm{Var}(\tilde s)\approx \sigma_q^2\sigma_k^2
$$

这一步解决的是“维度带来的放大”，不是“模长漂移带来的放大”。

把点积改写成模长与夹角形式更直观：

$$
\frac{q^\top k}{\sqrt{d_k}}
=
\frac{\|q\|_2 \|k\|_2}{\sqrt{d_k}}\cos\theta
$$

因此，哪怕 $\sqrt{d_k}$ 固定，只要训练中 $\|q\|_2,\|k\|_2$ 涨得快，logits 还是会变尖。softmax 的问题不在“能不能算出来”，而在“进入饱和区之后梯度是否还好学”。

### 为什么 logits 过大会伤害训练

softmax 定义为：

$$
p_i=\frac{e^{z_i}}{\sum_j e^{z_j}}
$$

它对输入 $z$ 的导数满足：

$$
\frac{\partial p_i}{\partial z_i}=p_i(1-p_i)
$$

当某个位置已经接近 1 时，$p_i(1-p_i)\to 0$；当某个位置已经接近 0 时，导数也接近 0。于是就出现了典型的“过早离散化”：

- 模型很早就只盯着一两个位置
- 错误信号很难传回其余位置
- 训练变得对初始化和学习率更敏感

### QK-Norm 为什么有效

QK-Norm 的做法是先对 query 和 key 做 $\ell_2$ 归一化：

$$
\hat q=\frac{q}{\|q\|_2},\quad
\hat k=\frac{k}{\|k\|_2}
$$

然后使用可学习尺度 $\tau$：

$$
\mathrm{score}=\tau \cdot \hat q^\top \hat k
$$

注意此时：

$$
\hat q^\top \hat k = \cos\theta \in [-1,1]
$$

所以 logits 的基本形状变成“可学习温度乘以余弦相似度”。这意味着模型先把“长度失控”从 attention logits 里拿掉，只保留方向信息，再由 $\tau$ 自己学习需要多尖锐。

可以把它理解成两步：

1. 把“谁和谁方向更像”与“向量本身有多长”拆开。
2. 把控制 softmax 尖锐程度的责任交给单独的温度参数，而不是交给不断漂移的向量模长。

### 残差缩放的由来

再看深度方向的信号堆叠。把第 $L$ 层的主干写成：

$$
x_L=x_0+\sum_{l=1}^{L}\alpha F_l(x_l)
$$

这里 $\alpha$ 表示残差分支的缩放系数。如果近似认为每层残差增量的方差都在同一量级，且层间相关性不太强，则：

$$
\mathrm{Var}(x_L)
\approx
\mathrm{Var}(x_0)+\sum_{l=1}^{L}\alpha^2 \mathrm{Var}(F_l(x_l))
\approx
\mathrm{Var}(x_0)+L\alpha^2 \sigma_F^2
$$

为了不让深度增长直接把总尺度推爆，需要让

$$
L\alpha^2=O(1)
$$

于是有：

$$
\alpha \approx \frac{1}{\sqrt{L}}
$$

这就是很多“深度相关初始化”背后的共同直觉：不是让每层变弱，而是让“总增量”别随着层数线性膨胀。

如果一个 block 里有两条主要残差增量，比如 attention 输出投影和 MLP 输出投影，那么工程上常见的写法会把对应投影层的初始化标准差再按 $\sqrt{2L}$ 缩小：

$$
\mathrm{std}_{\text{residual proj}}
=
\frac{\mathrm{std}_{\text{base}}}{\sqrt{2L}}
$$

这和“每个 block 里有两次主要增量注入”是对应的。

### LayerNorm 的小 $\gamma$ 为什么有意义

LayerNorm 先把统计量拉回标准范围，再乘以 $\gamma$。所以 $\gamma$ 的初值，本质上就是“本层一开始允许自己放大多少”。

如果直接设：

$$
\gamma=1
$$

那就意味着每层初始都有完整放大能力。对浅层网络这常常没问题，但深层 Transformer 会遇到两个叠加效应：

- 残差已经在逐层加
- 每次 LayerNorm 之后又允许满强度放行

于是一些工作会用更保守的 $\gamma$ 初始化，例如：

$$
\gamma^{(l)}=\alpha \sqrt{\frac{2}{N_l}}
$$

其中：

- $N_l$ 是该层宽度
- $\alpha$ 是人为设定的初始强度系数

这不是唯一正确公式，但它代表同一个思想：把 LayerNorm 的初始输出增益也纳入“整体尺度预算”。

### 把三条链路串起来看

从信息流角度，Transformer 训练早期最容易出问题的路径可以画成：

输入 $x$
$\rightarrow$ 线性投影得到 $Q,K,V$
$\rightarrow$ 控制 $Q,K$ 的长度影响
$\rightarrow$ 控制 softmax 前 logits
$\rightarrow$ 注意力输出投影
$\rightarrow$ 残差增量按深度缩小
$\rightarrow$ LayerNorm 以较小 $\gamma$ 开局
$\rightarrow$ 下一层继续迭代

所以这不是三个零散技巧，而是一整套策略：

- attention 负责控制“单层内部的尖锐性”
- residual 负责控制“跨层累加的总量”
- LayerNorm 负责控制“每层重新放行时的初始增益”

### 一个数值直觉例子

假设：

- $d_k=64$
- 单个 $q_i,k_i$ 的方差都约为 1

那么未缩放点积方差约为：

$$
\mathrm{Var}(q^\top k)\approx 64
$$

标准差约为 8。对 softmax 来说，这已经足够让不少行出现明显尖锐分布。除以 $\sqrt{64}=8$ 后，标准差回到 1，进入相对健康的区间。

但如果训练若干步后，$q,k$ 的平均模长各自都涨成原来的 2 倍，那么点积会接近放大 4 倍。此时即使还保留 $\frac{1}{\sqrt{d_k}}$，logits 的典型尺度也会再次偏大。这就是为什么“只靠 $\frac{1}{\sqrt{d_k}}$”不一定够。

---

## 代码实现

下面给一个可直接运行的最小 Python 版本。它不依赖 PyTorch，只用 `numpy` 演示三件事：

1. 不做 QK 归一化时，logits 会随着向量模长变大而迅速变尖
2. 残差缩放为什么会影响多层累加后的激活方差
3. LayerNorm 的小 $\gamma$ 会如何限制初始输出增益

```python
import math
import numpy as np


def l2_normalize(x, axis=-1, eps=1e-12):
    norm = np.sqrt(np.sum(x * x, axis=axis, keepdims=True) + eps)
    return x / norm


def softmax(x, axis=-1):
    x = x - np.max(x, axis=axis, keepdims=True)
    e = np.exp(x)
    return e / np.sum(e, axis=axis, keepdims=True)


def init_linear(fan_in, fan_out, std, rng):
    return rng.normal(0.0, std, size=(fan_in, fan_out))


def layernorm(x, gamma, beta=0.0, eps=1e-5):
    mean = np.mean(x, axis=-1, keepdims=True)
    var = np.var(x, axis=-1, keepdims=True)
    x_hat = (x - mean) / np.sqrt(var + eps)
    return gamma * x_hat + beta


def layernorm_gamma(alpha, width):
    return alpha * math.sqrt(2.0 / width)


def attention_scores(x, wq, wk, scale, use_qk_norm=False):
    q = x @ wq
    k = x @ wk
    if use_qk_norm:
        q = l2_normalize(q, axis=-1)
        k = l2_normalize(k, axis=-1)
    scores = (q @ k.T) * scale
    probs = softmax(scores, axis=-1)
    return scores, probs


def residual_stack_variance(x, num_layers, residual_gain, rng):
    """
    模拟多层残差叠加。这里不做真实 Transformer 计算，
    只构造形状相同、方差稳定的增量项来观察累计效果。
    """
    h = x.copy()
    variances = []
    for _ in range(num_layers):
        delta = rng.normal(0.0, 1.0, size=h.shape)
        h = h + residual_gain * delta
        variances.append(float(np.var(h)))
    return np.array(variances)


def summarize_attention(probs):
    row_entropy = -np.sum(probs * np.log(probs + 1e-12), axis=-1)
    row_max = np.max(probs, axis=-1)
    return row_max, row_entropy


def main():
    rng = np.random.default_rng(0)

    seq_len = 8
    d_model = 128
    d_k = 64
    num_layers = 12
    width = 1024
    alpha = 0.5

    # 一组“正常尺度”的输入，另一组人为放大模长
    x = rng.normal(0.0, 1.0, size=(seq_len, d_model))
    x_big = 4.0 * x

    base_std = 0.02
    residual_std = base_std / math.sqrt(2.0 * num_layers)

    wq = init_linear(d_model, d_k, base_std, rng)
    wk = init_linear(d_model, d_k, base_std, rng)
    wv = init_linear(d_model, d_k, base_std, rng)
    wo = init_linear(d_k, d_model, residual_std, rng)

    # 三个关键缩放量
    qk_scale = 1.0 / math.sqrt(d_k)
    res_scale = 1.0 / math.sqrt(num_layers)
    ln_gamma = layernorm_gamma(alpha, width)

    # 1) attention：比较无 QK-Norm 与有 QK-Norm
    scores_plain, probs_plain = attention_scores(
        x_big, wq, wk, qk_scale, use_qk_norm=False
    )
    scores_norm, probs_norm = attention_scores(
        x_big, wq, wk, 1.0, use_qk_norm=True
    )

    plain_max, plain_entropy = summarize_attention(probs_plain)
    norm_max, norm_entropy = summarize_attention(probs_norm)

    # 2) residual：比较无缩放与按 1/sqrt(L) 缩放
    var_no_scale = residual_stack_variance(x, num_layers, 1.0, rng)
    var_scaled = residual_stack_variance(x, num_layers, res_scale, rng)

    # 3) LayerNorm：比较 gamma=1 与更小 gamma
    ln_out_full = layernorm(x, gamma=1.0)
    ln_out_small = layernorm(x, gamma=ln_gamma)

    # 基本断言，保证代码可运行且结果形状合理
    assert scores_plain.shape == (seq_len, seq_len)
    assert probs_plain.shape == (seq_len, seq_len)
    assert np.allclose(np.sum(probs_plain, axis=-1), 1.0)
    assert np.allclose(np.sum(probs_norm, axis=-1), 1.0)
    assert abs(qk_scale - 0.125) < 1e-12
    assert abs(res_scale - 1.0 / math.sqrt(12)) < 1e-12
    assert round(ln_gamma, 3) == 0.022
    assert var_scaled[-1] < var_no_scale[-1]
    assert np.var(ln_out_small) < np.var(ln_out_full)

    print("=== Key scales ===")
    print(f"qk_scale   = {qk_scale:.6f}")
    print(f"res_scale  = {res_scale:.6f}")
    print(f"ln_gamma   = {ln_gamma:.6f}")
    print()

    print("=== Attention without QK-Norm (on enlarged inputs) ===")
    print("max prob per row:", np.round(plain_max, 4))
    print("entropy per row :", np.round(plain_entropy, 4))
    print("score std       :", round(float(np.std(scores_plain)), 4))
    print()

    print("=== Attention with QK-Norm ===")
    print("max prob per row:", np.round(norm_max, 4))
    print("entropy per row :", np.round(norm_entropy, 4))
    print("score std       :", round(float(np.std(scores_norm)), 4))
    print()

    print("=== Residual stack variance ===")
    print("last var without scaling :", round(float(var_no_scale[-1]), 4))
    print("last var with 1/sqrt(L)  :", round(float(var_scaled[-1]), 4))
    print()

    print("=== LayerNorm output variance ===")
    print("gamma=1.0      :", round(float(np.var(ln_out_full)), 4))
    print("gamma=small    :", round(float(np.var(ln_out_small)), 6))

    # 演示一个简单的 attention 输出投影与残差注入
    q = l2_normalize(x @ wq, axis=-1)
    k = l2_normalize(x @ wk, axis=-1)
    v = x @ wv
    attn = softmax((q @ k.T), axis=-1) @ v
    projected = attn @ wo
    y = x + res_scale * projected

    assert y.shape == x.shape
    print()
    print("final output shape:", y.shape)


if __name__ == "__main__":
    main()
```

这段代码可以直接运行，输出通常会体现三个现象：

| 现象 | 代码位置 | 应该看到什么 |
| --- | --- | --- |
| 输入模长放大后，无 QK-Norm 的 attention 更尖锐 | `scores_plain, probs_plain` | `max prob per row` 更大、`entropy` 更小 |
| 残差缩放能抑制多层方差膨胀 | `var_no_scale` vs `var_scaled` | 最后一层方差明显更小 |
| 小 $\gamma$ 会降低 LN 输出方差 | `ln_out_full` vs `ln_out_small` | 小 $\gamma$ 的输出方差更低 |

如果你刚接触这些术语，可以按下面的方式理解代码里的四个核心参数：

| 参数 | 含义 | 作用对象 | 直觉解释 |
| --- | --- | --- | --- |
| `base_std` | 普通线性层初始化标准差 | QKV、FFN 等普通投影 | 给矩阵一个保守的起步方差 |
| `residual_std` | 残差末投影更小的标准差 | `W_O` 或 MLP 输出投影 | 降低每层往主干里注入的增量 |
| `qk_scale` | 注意力 logits 缩放 | $QK^\top$ | 防止 softmax 一上来就过尖 |
| `ln_gamma` | LayerNorm 初始增益 | LN 输出 | 防止归一化后立刻满强度放行 |

### 如果改成 PyTorch，大致该怎么落地

工程上最常见的是两种做法：

1. 普通层保持原有初始化，单独把残差投影层的 `std` 缩小。
2. 在 attention 前加入 QK-Norm 或 logit cap，而不是只靠学习率调参硬扛。

一个 PyTorch 伪代码版本可以写成：

```python
for name, module in model.named_modules():
    if isinstance(module, nn.Linear):
        std = 0.02
        if name.endswith("out_proj") or name.endswith("c_proj"):
            std = 0.02 / math.sqrt(2 * n_layers)
        nn.init.normal_(module.weight, mean=0.0, std=std)
        if module.bias is not None:
            nn.init.zeros_(module.bias)

for module in model.modules():
    if isinstance(module, nn.LayerNorm):
        nn.init.constant_(module.weight, small_gamma)
        nn.init.zeros_(module.bias)
```

如果还要加 QK-Norm，则在 attention 前做：

```python
q = F.normalize(q, dim=-1)
k = F.normalize(k, dim=-1)
scores = tau * torch.matmul(q, k.transpose(-2, -1))
```

注意这里的 `tau` 一般是可学习参数，不再固定等于 $1/\sqrt{d_k}$。

### 几个常见实践要分清

| 模型/做法 | 常见初始化特征 | 重点 |
| --- | --- | --- |
| BERT 原始配方 | 较保守的正态初始化，`initializer_range=0.02` | 先用整体保守配置把训练拉稳 |
| GPT-2 常见实践 | 普通层 `std=0.02`，残差投影再按深度缩小 | 解决残差累加问题 |
| 后续更深 Transformer | QK-Norm、DeepNorm、logit cap 等结构级缩放 | 解决更深层、更长上下文下的稳定性 |

---

## 工程权衡与常见坑

最常见的问题不是“完全不知道这些方法”，而是“只改一处，期待所有问题自动消失”。

| 坑 | 典型现象 | 为什么会发生 | 规避方式 |
| --- | --- | --- | --- |
| 只保留 $\frac{1}{\sqrt{d_k}}$，不控 Q/K 范数 | attention 很快接近 one-hot | 维度被控制了，但模长漂移没被控制 | 加 QK-Norm、QKV-Norm 或 logit cap |
| 残差不做深度缩放 | 深层 loss 抖动、激活逐层变大 | 每层都往主干加一份不小的增量 | 对残差投影按层数缩小 |
| LayerNorm 默认 $\gamma=1$ | warmup 初期梯度范数偏大 | 每层一开始就拥有完整放大能力 | 超深模型用更小初始 $\gamma$ |
| 只调学习率，不改结构缩放 | 调参窗口很窄，换 batch 就不稳 | 学习率只能缓解，不会消除结构性放大 | 把缩放写进模型本身 |
| 混淆论文原始做法和社区后续改进 | 照搬配置结果不一致 | 不同模型稳定化工具箱并不相同 | 区分“论文原配方”和“现代工程组合” |

一个更直观的对比是：

| 配置 | 训练前几百步常见表现 |
| --- | --- |
| 无 QK 归一化 + 无残差缩放 + $\gamma=1$ | attention 尖锐、loss 抖动、偶发 NaN 或梯度峰值 |
| 仅 Pre-LN + warmup | 能训，但对学习率和 batch 更敏感 |
| QK 归一化 + 残差缩放 + 小 $\gamma$ | warmup 更平滑，梯度范数更可控 |
| 再加 logit cap / DeepNorm | 更适合极深层或长上下文场景 |

这里还要纠正两个常见误解。

### 误解 1：Xavier/He 没用

不是。Xavier/He 仍然是理解线性层初始化的重要基础，它们解决的是“单层线性映射的方差传播问题”。只是 Transformer 的主要不稳定来源不止这一层，还包括：

- softmax 前的 logits 尖锐化
- 残差跨层累加
- normalization 的初始增益

所以更准确的说法是：

> Xavier/He 解释了局部线性层的方差传播；Transformer 还需要额外控制 attention、residual、LayerNorm 三条全局数值链路。

### 误解 2：有了 Pre-LN 就不需要残差缩放

也不是。Pre-LN 的确显著改善了深层训练稳定性，但它并没有让“每层残差增量的累加”凭空消失。层数上去之后，残差投影缩放仍然常常有价值，尤其是在：

- decoder-only 预训练
- 上下文很长
- mixed precision 较激进
- warmup 不长或学习率偏大

### 一个简单的排障顺序

如果训练已经不稳定，排障优先级通常可以按这个顺序走：

| 先看什么 | 观察指标 | 如果异常，先做什么 |
| --- | --- | --- |
| attention 是否过尖 | 每行最大概率、attention entropy | 加 QK-Norm 或 logit cap |
| 残差是否逐层放大 | 激活方差、残差流均方值 | 缩小残差投影 std |
| warmup 初期梯度是否爆冲 | grad norm、loss spike | 降低 LN 初始 $\gamma$，同时检查学习率 |
| 是否只是实现 bug | mask、dtype、softmax 维度 | 先修实现，再谈初始化 |

---

## 替代方案与适用边界

除了 QK-Norm，还有几类常见替代方案。它们解决的并不是完全相同的问题，所以不要把它们当成可任意互换的按钮。

| 方案 | 做法 | 主要控制对象 | 额外开销 | 适用场景 |
| --- | --- | --- | --- | --- |
| QK-Norm | 只归一化 $Q,K$，再乘可学习温度 $\tau$ | logits 模长 | 低 | 标准自注意力，优先推荐 |
| QKV-Norm | 连 $V$ 一起归一化 | logits + value 尺度 | 低到中 | 想更激进控制内部尺度 |
| logit cap / tanh cap | 对 logits 截断或压缩 | 极端大 logits | 低 | 长上下文、极端尖锐 attention |
| DeepNorm | 修改残差系数并配套初始化 | 深层残差累加 | 中 | 很深的 Transformer |
| 仅 Pre-LN + warmup | 不改 attention 核，只调整结构位置与优化 | 训练入口稳定性 | 低 | 中小模型，先求简单可用 |

### 这些方案各自解决什么

如果按“问题来源”分类，它们更容易理解：

| 问题来源 | 更直接的工具 |
| --- | --- |
| attention logits 过大 | $\frac{1}{\sqrt{d_k}}$、QK-Norm、logit cap |
| 残差跨层累加 | 残差缩放、DeepNorm |
| LayerNorm 初始放大太强 | 小 $\gamma$ 初始化 |
| 训练入口太脆弱 | Pre-LN、warmup |

这张表的含义是：如果你看到的是 attention 过尖，就优先动 QK 相关组件；如果你看到的是激活随着层数越来越大，就优先动残差缩放。不要拿学习率去替代结构性修复。

### 适用边界可以直接记成三条

1. 12 层以内的小模型，优先顺序通常是：Pre-LN > 合理 warmup > 标准 $\frac{1}{\sqrt{d_k}}$。这时不一定非上 QK-Norm。
2. 24 层到数十层模型，残差缩放开始明显重要，尤其是 decoder-only 大模型。
3. 上百层、长上下文、混合精度预训练时，QK-Norm、DeepNorm、logit cap 这类“结构级稳定器”通常比继续细抠 Xavier/He 更值钱。

### 如果只能先改一处，先改哪儿

经验上可以这样选：

| 现象 | 最优先改动 |
| --- | --- |
| loss 一开始就抖，激活逐层变大 | 残差末投影缩小 + Pre-LN |
| attention 权重几乎 0/1 | QK-Norm 或 logit cap |
| warmup 期间梯度峰值很高 | 更小的 LayerNorm $\gamma$ |
| 小模型只是偶发不稳 | 先降学习率或延长 warmup，再决定是否加结构缩放 |

如果只能做一个改动，我会优先做“残差末投影缩小 + Pre-LN”；如果已经明确看到 attention 极尖锐，再加 QK-Norm。

---

## 参考资料

| 资料 | 内容摘要 | 关联章节 |
| --- | --- | --- |
| Vaswani et al., 2017, *Attention Is All You Need*, NeurIPS 2017, https://proceedings.neurips.cc/paper/7181-attention-is-all | 标准 scaled dot-product attention，给出 $\frac{QK^\top}{\sqrt{d_k}}$ 的来源 | 问题定义与边界、核心机制与推导 |
| Henry et al., 2020, *Query-Key Normalization for Transformers*, Findings of EMNLP 2020, https://aclanthology.org/2020.findings-emnlp.379/ | 提出 QKNorm：对 $Q,K$ 做 $\ell_2$ 归一化，并用可学习尺度替代固定缩放 | 核心机制与推导、替代方案与适用边界 |
| Devlin et al., 2018, *BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding*, https://arxiv.org/abs/1810.04805 | BERT 的基础架构与训练方式；工程实现常见 `initializer_range=0.02` | 代码实现、工程权衡与常见坑 |
| Google Research BERT Repo, https://github.com/google-research/bert | 官方实现中可直接看到 `initializer_range` 的工程落地方式 | 代码实现、工程权衡与常见坑 |
| Radford et al., 2019, *Language Models are Unsupervised Multitask Learners*, https://cdn.openai.com/better-language-models/language_models_are_unsupervised_multitask_learners.pdf | GPT-2 说明将 LayerNorm 放到子层输入，并使用考虑残差深度累积的初始化 | 核心结论、代码实现、工程权衡与常见坑 |
| Brown et al., 2020, *Language Models are Few-Shot Learners*, https://arxiv.org/abs/2005.14165 | GPT-3 展示大规模 decoder-only Transformer 训练中稳定性设计的重要性 | 工程权衡与常见坑、替代方案与适用边界 |
| Wang et al., 2022, *DeepNet: Scaling Transformers to 1,000 Layers*, arXiv:2203.00555, https://arxiv.org/abs/2203.00555 | 提出 DeepNorm，通过修改残差系数与配套初始化稳定超深 Transformer | 替代方案与适用边界 |
| Wang et al., 2024, *DeepNet: Scaling Transformers to 1,000 Layers*, IEEE TPAMI, https://ustcwhy.github.io/publications/deepnet/ | DeepNet 的期刊版本，便于引用最终发表信息 | 替代方案与适用边界 |
