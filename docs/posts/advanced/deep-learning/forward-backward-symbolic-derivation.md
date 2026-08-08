---
title: 前向传播与反向传播的符号推导
date: 2026-08-07
---

# 前向传播与反向传播的符号推导

<div class="epigraph">
<p>符号是思想的脚手架：把每一步写清楚，正确就水到渠成。</p>
<footer>—— 依据戈特弗里德 · 莱布尼茨（Gottfried Leibniz）的精神改写</footer>
</div>

<div class="article-byline">
<p>第四级 · 深度学习 ｜ 花书《深度学习》§6.5.4、李沐《动手学深度学习》§4.7 ｜ 2026-08-07</p>
</div>

## 为什么从符号推导开始

上一节我们用图语言理解了反向传播的「算法」。但「图上的规则」与「逐层网络的公式」之间还有一段距离：真实网络的每一层都有具体的形状（线性层、激活层、Softmax 层），前向与反向的具体公式是什么？**符号推导（symbolic derivation）**就是把这一层层公式完整写出来、逐个验证维度的过程。它有三个价值：其一，让你能**手写实现**一个网络而不依赖框架；其二，让你能读懂任何一篇论文里的推导；其三，让你在框架报维度错误时能立刻定位——因为你知道每一处应有的形状。

本节用一个标准的单隐藏层 MLP + Softmax + 交叉熵为例，**从零推导完整的前向与反向公式**，并给出可运行的矢量化实现骨架。这一节是「看懂框架在干什么」的分水岭：过了这一节，反向传播对你不再是黑盒。<span class="marginnote">符号推导的核心纪律是<strong>「维度守恒」</strong>：每一行公式的左端与右端维度必须一致。你会在下面的每一步都看到我们用「检查维度」作为推导正确性的第一道验证——这是手写反向传播时最重要的防错手段。</span>

## 1 记号约定与网络结构

我们推导的网络结构如下：

$$
\underbrace{\boldsymbol{x}}_{d} \;\to\; \underbrace{\boldsymbol{a}^{(1)} = \boldsymbol{W}^{(1)}\boldsymbol{x} + \boldsymbol{b}^{(1)}}_{h} \;\to\; \underbrace{\boldsymbol{h} = \sigma(\boldsymbol{a}^{(1)})}_{h} \;\to\; \underbrace{\boldsymbol{o} = \boldsymbol{W}^{(2)}\boldsymbol{h} + \boldsymbol{b}^{(2)}}_{k} \;\to\; \underbrace{\hat{\boldsymbol{y}} = \text{softmax}(\boldsymbol{o})}_{k}
$$

维度依次为：输入 $d$ 维，隐藏单元 $h$ 个，输出 $k$ 类。各符号含义：$\boldsymbol{W}^{(1)}\in\mathbb{R}^{h\times d}$、$\boldsymbol{b}^{(1)}\in\mathbb{R}^{h}$、$\boldsymbol{W}^{(2)}\in\mathbb{R}^{k\times h}$、$\boldsymbol{b}^{(2)}\in\mathbb{R}^{k}$。损失取交叉熵 $L = -\log \hat{y}_{\text{true}}$。为推导方便，用**下标写法**：$\boldsymbol{W}^{(2)}_{ij}$ 表示输出单元 $i$ 到隐藏单元 $j$ 的权重。

## 2 前向传播：逐层计算与缓存

前向传播依次计算并**缓存**四组量：

1. $\boldsymbol{a}^{(1)} = \boldsymbol{W}^{(1)}\boldsymbol{x} + \boldsymbol{b}^{(1)}$（线性层输出）
2. $\boldsymbol{h} = \sigma(\boldsymbol{a}^{(1)})$（激活）
3. $\boldsymbol{o} = \boldsymbol{W}^{(2)}\boldsymbol{h} + \boldsymbol{b}^{(2)}$（logits）
4. $\hat{\boldsymbol{y}} = \text{softmax}(\boldsymbol{o})$，$L = -\log \hat{y}_{\text{true}}$

**为什么必须缓存 $\boldsymbol{a}^{(1)}$？** 因为 $\sigma'$ 依赖它：ReLU 的导数 $\mathbf{1}[\boldsymbol{a}^{(1)}>0]$、Sigmoid 的导数 $\sigma(\boldsymbol{a}^{(1)})(1-\sigma(\boldsymbol{a}^{(1)}))$，都直接需要 $\boldsymbol{a}^{(1)}$ 或 $\boldsymbol{h}$ 的值。反向传播时这些值必须「记得」。

**前向的维度检查**：$\boldsymbol{W}^{(1)}$($h\times d$) × $\boldsymbol{x}$($d$) → $\boldsymbol{a}^{(1)}$($h$)；$\boldsymbol{W}^{(2)}$($k\times h$) × $\boldsymbol{h}$($h$) → $\boldsymbol{o}$($k$)。每一处维度都严丝合缝——**前向正确是反向正确的前提**。

## 3 反向传播第一步：输出层梯度

反向传播从损失开始，先算 **logits 的梯度**。由上一节的「Softmax + 交叉熵」推导：

$$
\boldsymbol{\delta}^{(o)} = \frac{\partial L}{\partial \boldsymbol{o}} = \hat{\boldsymbol{y}} - \boldsymbol{y}_{\text{one-hot}} \;\in\; \mathbb{R}^{k}
$$

这是反向传播的「源头」：一切都从它流出去。有了 $\boldsymbol{\delta}^{(o)}$，输出层参数的梯度立刻可得：

$$
\frac{\partial L}{\partial \boldsymbol{W}^{(2)}} = \boldsymbol{\delta}^{(o)} \boldsymbol{h}^{\top} \;\in\; \mathbb{R}^{k\times h}, \qquad
\frac{\partial L}{\partial \boldsymbol{b}^{(2)}} = \boldsymbol{\delta}^{(o)} \;\in\; \mathbb{R}^{k}
$$

**维度检查**：$\boldsymbol{\delta}^{(o)}$($k$) × $\boldsymbol{h}^{\top}$($h$) = $k\times h$ 矩阵，与 $\boldsymbol{W}^{(2)}$ 同形——**梯度与参数必须同形状**，这是反向传播最重要的自查规则。

## 4 反向传播第二步：梯度穿过隐藏层

把梯度继续往回流。先计算**隐藏层激活的梯度**，它由「$\boldsymbol{o}$ 对 $\boldsymbol{h}$ 的依赖」给出：

$$
\boldsymbol{\delta}^{(h)} = \frac{\partial L}{\partial \boldsymbol{h}} = \boldsymbol{W}^{(2)\top} \boldsymbol{\delta}^{(o)} \;\in\; \mathbb{R}^{h}
$$

**推导**：由链式法则，$L$ 对 $h_j$ 的偏导为 $\sum_i \frac{\partial L}{\partial o_i}\frac{\partial o_i}{\partial h_j} = \sum_i \delta^{(o)}_i W^{(2)}_{ij}$，写回矩阵形式正是 $\boldsymbol{W}^{(2)\top}\boldsymbol{\delta}^{(o)}$。<span class="marginnote">这一行是「梯度转置回传」的典范：正向是 $\boldsymbol{W}^{(2)}$ 把 $h$ 变成 $o$，反向就是 $\boldsymbol{W}^{(2)\top}$ 把「$o$ 的梯度」变回「$h$ 的梯度」。<strong>反向传播 = 正向的转置伴随</strong>——这一条规律可以推广到一切线性层，是检查反向实现的最快心法。</span>

接着穿过激活层。激活层的输入是 $\boldsymbol{a}^{(1)}$，输出是 $\boldsymbol{h} = \sigma(\boldsymbol{a}^{(1)})$，于是

$$
\boldsymbol{\delta}^{(a)} = \frac{\partial L}{\partial \boldsymbol{a}^{(1)}} = \boldsymbol{\delta}^{(h)} \odot \sigma'(\boldsymbol{a}^{(1)}) \;\in\; \mathbb{R}^{h}
$$

其中 $\odot$ 是逐元素（Hadamard）乘积。**推导**：$\frac{\partial L}{\partial a^{(1)}_j} = \frac{\partial L}{\partial h_j}\frac{\partial h_j}{\partial a^{(1)}_j} = \delta^{(h)}_j \sigma'(a^{(1)}_j)$——激活函数逐元素作用，梯度也逐元素相乘，**没有跨单元耦合**。这就是「逐元素激活」在反向里的天然体现。

最后，穿过第一层线性层得到 $\boldsymbol{x}$ 的梯度（一般不需要，但用于完整性）与第一层参数的梯度：

$$
\frac{\partial L}{\partial \boldsymbol{W}^{(1)}} = \boldsymbol{\delta}^{(a)} \boldsymbol{x}^{\top} \;\in\; \mathbb{R}^{h\times d}, \qquad
\frac{\partial L}{\partial \boldsymbol{b}^{(1)}} = \boldsymbol{\delta}^{(a)} \;\in\; \mathbb{R}^{h}
$$

## 5 公式解析：完整推导的装配图

把前向与反向的全部公式放在一张总表里，可以看清整个「计算-回传」的对称结构：

| 节点 | 前向 | 反向（缓存 $\to$ 梯度） |
| --- | --- | --- |
| 输入 $\boldsymbol{x}$ | — | （无梯度需求） |
| 线性层 1 | $\boldsymbol{a}^{(1)} = \boldsymbol{W}^{(1)}\boldsymbol{x}+\boldsymbol{b}^{(1)}$ | $\frac{\partial L}{\partial \boldsymbol{W}^{(1)}} = \boldsymbol{\delta}^{(a)}\boldsymbol{x}^{\top}$ |
| 激活层 | $\boldsymbol{h} = \sigma(\boldsymbol{a}^{(1)})$ | $\boldsymbol{\delta}^{(a)} = \boldsymbol{\delta}^{(h)} \odot \sigma'(\boldsymbol{a}^{(1)})$ |
| 线性层 2 | $\boldsymbol{o} = \boldsymbol{W}^{(2)}\boldsymbol{h}+\boldsymbol{b}^{(2)}$ | $\boldsymbol{\delta}^{(h)} = \boldsymbol{W}^{(2)\top}\boldsymbol{\delta}^{(o)}$ |
| Softmax+CE | $\hat{\boldsymbol{y}} = \text{softmax}(\boldsymbol{o})$，$L=-\log\hat{y}_{\text{true}}$ | $\boldsymbol{\delta}^{(o)} = \hat{\boldsymbol{y}}-\boldsymbol{y}_{\text{one-hot}}$ |

三步拆解这张表的**结构性规律**：

- **第一步，看前向列**：从 $\boldsymbol{x}$ 到 $\boldsymbol{o}$，数据逐层前进；每个线性层都需要上一层的输出。
- **第二步，看反向列**：从 $\boldsymbol{\delta}^{(o)}$ 到 $\boldsymbol{\delta}^{(a)}$，梯度逐层回流；**每个线性层的反向输入，就是下一层的反向输出**，且都呈「上游梯度 × 当前层局部分量」的形态。
- **第三步，看对称性**：权重的梯度总是「该层输入激活 × 该层上游梯度」的外积；偏置梯度就是上游梯度。**只要记住这个对称模板，任意深的网络都能机械地推导出来**。<span class="marginnote">「反向 = 前向的伴随」在更抽象的层面是<strong>伴随算子（adjoint operator）</strong>：线性层的前向是乘以 $\boldsymbol{W}$，反向是乘以 $\boldsymbol{W}^{\top}$；卷积的前向是卷积，反向是「转置卷积」（上采样卷积）——第四篇《卷积神经网络》会再次遇到这个规律。</span>

## 6 手写实现与数值验证

把上面的公式落成矢量化代码（无高层 API），并用数值梯度检验验证：

```python
import numpy as np

# 前向：逐层计算并缓存
def forward(x, W1, b1, W2, b2):
    a1 = W1 @ x + b1
    h = np.maximum(a1, 0)            # ReLU 激活
    o = W2 @ h + b2
    y_hat = np.exp(o - o.max())      # 数值稳定 softmax
    y_hat = y_hat / y_hat.sum()
    return a1, h, y_hat

# 反向：从输出层往输入层回传
def backward(x, y, W1, b1, W2, b2, a1, h, y_hat):
    d_o = y_hat - y                          # δ^(o) = ŷ - y
    dW2 = np.outer(d_o, h)                   # ∂L/∂W2 = δ^(o) hᵀ
    db2 = d_o
    d_h = W2.T @ d_o                         # δ^(h) = W2ᵀ δ^(o)
    d_a1 = d_h * (a1 > 0)                    # δ^(a) = δ^(h) ⊙ σ'(a1)
    dW1 = np.outer(d_a1, x)                  # ∂L/∂W1 = δ^(a) xᵀ
    db1 = d_a1
    return dW1, db1, dW2, db2
```

**数值梯度检验**：把参数 $\theta_i$ 的某个分量扰动 $\epsilon=10^{-6}$，比较「数值差商」与「用链式法则推得的梯度」。若相对误差在 $10^{-7}$ 量级，说明推导与实现都正确。<span class="marginnote">这套「推导 → 手写 → 数值检验」的流程，是每个深度学习工程师必修的基本功：它把「相信我推对了」变成「有证据地确认」。实际工程中你会直接信任框架的 autograd，但<strong>当你要实现自定义算子或优化器时，这套技能就是你的安全网</strong>。</span>

## 7 小结

- 反向传播 = 从输出往输入的**伴随梯度回流**，每一步都是「上游梯度 × 局部分量」。
- 输出层源头：$\boldsymbol{\delta}^{(o)} = \hat{\boldsymbol{y}} - \boldsymbol{y}_{\text{one-hot}}$。
- 线性层反向两条规则：参数梯度 = 外积「上游梯度 × 输入激活」；输入梯度 = 「权重转置 × 上游梯度」。
- 激活层反向 = 逐元素乘激活导数；**逐元素激活无跨单元耦合**。
- 两条自查纪律：**梯度与参数同形状**；**反向 = 前向的转置伴随**。
- 用数值梯度检验验证手写实现，是确认正确性的黄金标准。

在下一节，我们把「如何确认求导正确」做成一套工程流程——数值梯度检验的完整操作与高效实现技巧，这就是**数值梯度检验与高效实现**。
