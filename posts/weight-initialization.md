## 核心结论

权重初始化的目标，不是“随便给一个随机数”，而是在网络一开始训练时，让每一层的信号尺度大致稳定。这里的“尺度”可以理解成数值波动范围，数学上通常用方差描述。

如果所有权重都初始化为 0，会出现**对称性问题**。对称性问题的白话解释是：同一层的多个神经元从第一步开始就完全一样，看到相同输入，算出相同输出，拿到相同梯度，之后参数更新也永远一样。这一层虽然名义上有很多神经元，实际只相当于一个神经元被复制了很多份。

因此，初始化必须满足两件事：

| 方案 | 方差公式 | 适用激活 | 直观原因 |
|---|---:|---|---|
| Xavier | $\mathrm{Var}(W)=\frac{2}{n_{in}+n_{out}}$ | tanh、线性、对称激活 | 同时兼顾前向和反向的方差稳定 |
| He | $\mathrm{Var}(W)=\frac{2}{n_{in}}$ | ReLU、Leaky ReLU 家族 | ReLU 会截掉一半负值，需要更大的初始方差补偿 |

可以把每个权重看成一个“放大器”。放大器太强，信号层层放大后会爆炸；放大器太弱，信号传几层就接近 0。Xavier 和 He 的核心，就是根据连接数自动设置放大器强度。

---

## 问题定义与边界

先固定讨论边界：本文讨论的是**全连接层或卷积层的随机权重初始化**，重点是初始化时的方差选择，不展开偏置初始化、归一化层细节、优化器超参数等问题。

设某层线性变换为：

$$
z^{[l]} = W^{[l]} a^{[l-1]} + b^{[l]}
$$

其中：

- $a^{[l-1]}$ 是上一层激活值，白话解释是“上一层真正传下来的输出”
- $W^{[l]}$ 是当前层权重矩阵
- $z^{[l]}$ 是激活函数之前的线性输出，也叫 pre-activation

如果做全零初始化：

$$
W^{[l]}=0
$$

那么同一层每个神经元的输出完全相同，反向传播得到的梯度也完全相同。于是训练若干步后，这些神经元仍然相同，模型没有学出多样化表示能力。

所以必须随机初始化，但随机不等于任意。一个最常见的写法是：

```python
W = np.random.randn(n_in, n_out) * scale
```

其中 `scale` 不能随便写。它太大时，$z$ 的方差会随着层数增长而爆炸；太小时，方差会层层衰减，最后梯度消失。这个尺度必须和 `fan-in/fan-out` 绑定。这里的 `fan-in` 指输入连接数，白话解释是“一个神经元接收了多少路输入”；`fan-out` 指输出连接数，也就是“它向多少个下游神经元发信号”。

一个玩具例子：假设一层有 100 个输入，每个输入大致在 $[-1,1]$ 摆动。如果每个权重标准差取 1，那么 100 项相加后的输出几乎必然非常大；如果权重标准差取 0.0001，那么 100 项加完仍然很小，后续层几乎收不到有效信号。初始化真正要解决的，就是这个“加总后会不会失控”的问题。

---

## 核心机制与推导

### 1. 从线性层方差开始

假设：

- 权重独立同分布，均值为 0
- 输入激活独立同分布，均值近似为 0
- 偏置先忽略不计

对某个神经元，有

$$
z = \sum_{i=1}^{n_{in}} w_i a_i
$$

因为各项独立且均值为 0，可近似得到：

$$
\mathrm{Var}(z)=n_{in}\,\mathrm{Var}(w)\,\mathrm{Var}(a)
$$

这条式子是整个初始化推导的出发点。它说明：输入越多，若权重方差不缩小，输出方差就越大。

### 2. Xavier 为什么是 $\frac{2}{n_{in}+n_{out}}$

Xavier 初始化面向的是 tanh 这类对称激活。对称激活的白话解释是：输入正负两边都参与计算，不像 ReLU 会直接砍掉负半轴。

如果希望前向传播时各层激活方差尽量不变，需要：

$$
\mathrm{Var}(z^{[l]}) \approx \mathrm{Var}(a^{[l-1]})
$$

代入前式：

$$
n_{in}\,\mathrm{Var}(W)\,\mathrm{Var}(a^{[l-1]}) \approx \mathrm{Var}(a^{[l-1]})
$$

得到前向稳定条件：

$$
\mathrm{Var}(W)\approx \frac{1}{n_{in}}
$$

如果再考虑反向传播，希望梯度方差也不要明显放大或缩小，会得到另一个近似条件：

$$
\mathrm{Var}(W)\approx \frac{1}{n_{out}}
$$

Xavier 取这两个条件的折中：

$$
\mathrm{Var}(W)=\frac{2}{n_{in}+n_{out}}
$$

这不是唯一精确答案，而是一个兼顾前向与反向稳定的工程近似。

### 3. He 为什么是 $\frac{2}{n_{in}}$

ReLU 的定义是：

$$
\mathrm{ReLU}(x)=\max(0,x)
$$

它的关键性质是：输入如果关于 0 对称，那么大约一半值会被截成 0。于是输出的有效方差会下降，常见近似是：

$$
\mathrm{Var}(a) \approx \frac{1}{2}\mathrm{Var}(z)
$$

把这个损失补回去，就需要让线性层输出的方差更大一点。仍从

$$
\mathrm{Var}(z)=n_{in}\,\mathrm{Var}(W)\,\mathrm{Var}(a)
$$

出发，并希望经过 ReLU 后方差还能大致维持原尺度：

$$
\frac{1}{2} n_{in}\,\mathrm{Var}(W)\,\mathrm{Var}(a) \approx \mathrm{Var}(a)
$$

约掉 $\mathrm{Var}(a)$，得到：

$$
\mathrm{Var}(W)=\frac{2}{n_{in}}
$$

这就是 He 初始化。

### 4. 数值例子

设某层：

- $n_{in}=256$
- $n_{out}=128$

则：

| 方案 | 方差 | 标准差 $\sigma=\sqrt{\mathrm{Var}}$ | 解释 |
|---|---:|---:|---|
| Xavier | $\frac{2}{256+128}=0.00521$ | $\approx 0.072$ | 适合 tanh 等对称激活 |
| He | $\frac{2}{256}=0.00781$ | $\approx 0.088$ | 适合 ReLU，需要更大初始振幅 |

可以看到，He 的标准差更大，因为 ReLU 会丢掉一部分信号，初始化时必须提前补偿。

---

## 代码实现

下面给出一个可运行的 NumPy 版本，支持 Xavier 和 He，并根据激活函数自动选择。这里用 `fan_in` 和 `fan_out` 控制缩放，并用样本方差做一个简单检查。

```python
import numpy as np

def init_weights(shape, activation="relu", mode="normal", seed=0):
    """
    shape: (fan_in, fan_out)
    activation: relu / leaky_relu / tanh / linear / sigmoid
    mode: normal / uniform
    """
    fan_in, fan_out = shape
    rng = np.random.default_rng(seed)

    if activation in {"relu", "leaky_relu"}:
        var = 2.0 / fan_in
    elif activation in {"tanh", "linear", "sigmoid"}:
        var = 2.0 / (fan_in + fan_out)
    else:
        raise ValueError(f"unsupported activation: {activation}")

    if mode == "normal":
        std = np.sqrt(var)
        w = rng.normal(0.0, std, size=shape)
    elif mode == "uniform":
        bound = np.sqrt(3.0 * var)
        w = rng.uniform(-bound, bound, size=shape)
    else:
        raise ValueError(f"unsupported mode: {mode}")

    return w, var

# Xavier 检查
w_x, var_x = init_weights((256, 128), activation="tanh", seed=42)
sample_var_x = float(np.var(w_x))
assert abs(sample_var_x - var_x) / var_x < 0.15

# He 检查
w_h, var_h = init_weights((256, 128), activation="relu", seed=42)
sample_var_h = float(np.var(w_h))
assert abs(sample_var_h - var_h) / var_h < 0.15

# He 的目标方差应大于 Xavier
assert var_h > var_x

print("xavier target:", round(var_x, 6), "sample:", round(sample_var_x, 6))
print("he target:", round(var_h, 6), "sample:", round(sample_var_h, 6))
```

这个实现里有两个工程要点：

1. `shape = (fan_in, fan_out)` 必须定义清楚。不同框架矩阵布局可能不同，写反了会直接把方差公式用错。
2. `uniform` 分布也能实现同样的目标，只要保证其方差匹配目标值。因为均匀分布 $U(-a,a)$ 的方差是 $\frac{a^2}{3}$，所以边界应取 $a=\sqrt{3\cdot \mathrm{Var}}$。

一个更接近真实训练代码的伪代码如下：

```python
if activation == "tanh":
    scale = sqrt(2 / (n_in + n_out))   # Xavier
elif activation == "relu":
    scale = sqrt(2 / n_in)             # He

W = randn(n_in, n_out) * scale
b = zeros(n_out)
```

真实工程例子：假设你在写一个多层感知机分类器，网络结构是 `784 -> 512 -> 256 -> 128 -> 10`。如果隐藏层都用 ReLU，那么三层隐藏层的权重都应该优先用 He 初始化。如果改成 tanh，则更适合 Xavier。这个选择不是“经验偏好”，而是由激活函数对方差的影响直接决定的。

---

## 工程权衡与常见坑

初始化公式只是起点，不是训练成功的全部保证，但它会显著影响训练能否顺利开始。

| 错误搭配 | 常见后果 | 原因 | 修正方案 |
|---|---|---|---|
| Xavier + ReLU | 深层信号逐层变弱，梯度偏小 | ReLU 截断负半轴，Xavier 补偿不够 | 改为 He |
| He + tanh/sigmoid | 容易进入饱和区，更新变慢 | 初始方差偏大，激活过早贴边 | 改为 Xavier |
| 全零初始化 | 同层神经元永远等价 | 对称性没有被打破 | 使用随机初始化 |
| 不看矩阵布局直接套公式 | 实际方差完全错误 | `fan_in/fan_out` 算反 | 先确认参数 shape |
| 深网络只靠初始化不配合归一化 | 训练前期不稳定 | 方差会在非线性和残差中继续漂移 | 配合 BatchNorm/LayerNorm |

一个玩具例子：3 层 ReLU 网络，如果每层输出方差都被乘上约 0.5，那么经过 10 层后大约变成原来的 $0.5^{10}\approx 0.001$。这意味着后面几层几乎接收不到有效信号，反向梯度也会很小。He 初始化的作用，就是尽量避免这种指数级衰减。

一个真实工程例子：在 ResNet 这类深层卷积网络里，ReLU 非常常见。如果误用 Xavier，前几十层可能还能训练，但更深处的激活和梯度会逐步变弱，表现为损失下降缓慢、训练初期很难进入稳定收敛。改成 He 初始化后，通常不需要改网络结构，只是把初始化对齐到激活函数，训练曲线就会明显更平稳。

还要注意一个常被忽略的点：卷积层的 `fan_in` 不是“输入通道数”本身，而是

$$
fan\_in = C_{in} \times K_h \times K_w
$$

也就是输入通道数乘卷积核面积。比如 `3x3` 卷积、64 个输入通道，那么 `fan_in = 64 * 3 * 3 = 576`。

---

## 替代方案与适用边界

Xavier 和 He 不是全部方案，只是最常见的两类默认选择。

| 初始化方案 | 典型公式 | 适用激活 | 适用边界 |
|---|---|---|---|
| Xavier Normal | $\mathcal{N}(0,\frac{2}{n_{in}+n_{out}})$ | tanh、线性 | 中浅层到中深层 |
| Xavier Uniform | $U(-\sqrt{\frac{6}{n_{in}+n_{out}}}, \sqrt{\frac{6}{n_{in}+n_{out}}})$ | tanh、线性 | 与上类似 |
| He Normal | $\mathcal{N}(0,\frac{2}{n_{in}})$ | ReLU、Leaky ReLU | 深层 CNN/MLP 常用 |
| He Uniform | $U(-\sqrt{\frac{6}{n_{in}}}, \sqrt{\frac{6}{n_{in}}})$ | ReLU 家族 | 与上类似 |
| Orthogonal | 正交矩阵缩放 | RNN、很深网络 | 保持方向结构更稳定 |

如果网络里使用 sigmoid，单靠初始化往往不够。sigmoid 的白话解释是：它会把数值压到 0 到 1 之间，输入稍大稍小就容易进入“几乎不变”的饱和区。此时 Xavier 通常比 He 更安全，但很多工程里还会配合以下手段：

- Batch Normalization：对每层中间输出做标准化，白话解释是“把每层分布重新拉回可训练区间”，降低初始化敏感性。
- Layer Normalization：在 Transformer 等结构中更常见，作用类似，但归一化维度不同。
- Orthogonal 初始化：在很深或递归结构中，有时比普通高斯初始化更稳。
- Leaky ReLU：给负半轴留一条小通路，减少 ReLU 彻底截断带来的梯度损失。

适用边界也要说清楚。Xavier/He 推导用了若干近似假设：独立同分布、均值为 0、层内神经元统计一致。真实网络里，这些条件不完全成立，所以公式不是“严格真理”，而是“很好的默认工程近似”。一旦网络中有残差连接、归一化、注意力结构，这些层会继续改变激活分布，初始化的重要性仍然存在，但不再是唯一决定因素。

---

## 参考资料

- Technology Hits：Xavier 与 He 的方差推导，重点看前向/反向方差守恒的来源。
- Artificial Intelligence Wiki：He 初始化与深层 ReLU 网络的工程背景，适合理解卷积网络中的使用方式。
- TrueGeometry Blog：Xavier 与 He 的适配激活差异，总结了常见错误搭配与后果。
- Vitademy Global：零初始化的对称性问题，适合新手理解为什么“随机”是必须条件。
- Aryan Upadhyay 相关文章：控制方差的直觉解释，适合理解“为什么 fan-in/fan-out 会进入公式”。

建议阅读顺序：

1. Technology Hits：先看公式推导，建立方差守恒的主线。
2. Vitademy Global：再看零初始化和随机缩放的直观解释。
3. Artificial Intelligence Wiki：最后把结论放回 ResNet、CNN 等真实工程场景中理解。
