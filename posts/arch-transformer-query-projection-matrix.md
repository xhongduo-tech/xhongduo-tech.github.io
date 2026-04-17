## 核心结论

`W_Q` 是 Transformer 里把输入表示映射成查询向量的线性变换。线性变换可以理解为“按一组可学习权重重新组合特征”。如果输入矩阵记为 $X \in \mathbb{R}^{n \times d_{model}}$，那么查询矩阵就是

$$
Q = XW_Q,\quad W_Q \in \mathbb{R}^{d_{model}\times d_k}
$$

这里的 $d_{model}$ 是模型隐藏维度，意思是每个 token 在主干网络里的表示长度；$d_k$ 是 query/key 的维度，意思是注意力里专门用于“比较相似度”的向量长度。

核心结论有三条。

第一，`W_Q` 的职责不是“存储知识”，而是把输入变换到一个适合做相似度比较的空间。模型最终关注谁，起点就是 `Q` 怎么被构造出来。

第二，多头注意力本质上是给同一个输入并行准备多组不同的查询空间。常见写法是：

$$
\text{MultiHead}(Q,K,V)=\text{Concat}(\text{head}_1,\ldots,\text{head}_h)W^O
$$

$$
\text{head}_i=\text{Attention}(QW_i^Q,KW_i^K,VW_i^V)
$$

这表示每个头都有自己的一套投影矩阵，能在不同子空间里学习“关注规则”。

第三，多头拆分在数学上等价于“先把所有头的投影矩阵拼成一个大矩阵，再一次性线性映射，然后 reshape 切分”。也就是说，多头不是更神秘的运算，它仍然只是线性映射加分组。

一个给初学者的直观说法是：如果模型在看一张图像或一句话，`W_Q` 决定“我此刻在找什么特征”；不同 `head` 像戴了不同颜色的眼镜看同一份输入，有的头更容易对齐位置关系，有的头更容易对齐语义关系。

---

## 问题定义与边界

本文只讨论 self-attention 中的 Query 投影矩阵 `W_Q`，重点回答四个问题：

1. `W_Q` 的形状为什么通常是 $(d_{model}\times d_k)$。
2. `W_Q` 与 `W_K`、`W_V` 的关系是什么。
3. 多头拆分为什么与“大矩阵一次算完再切头”是等价的。
4. 初始化与缩放为什么会影响注意力分布和训练稳定性。

边界也要说清楚。

第一，`W_Q` 不能脱离 `W_K` 单独理解。因为注意力分数来自 $QK^\top$，query 和 key 必须处在同一可比较空间里，所以通常要求 query 和 key 的末维相同，都是 $d_k$。

第二，`d_k` 不是越大越好。点积是把两个向量每一维相乘后求和。若各维度近似独立且方差相近，点积的方差会随 $d_k$ 增大而增大，因此需要除以 $\sqrt{d_k}$ 做缩放：

$$
\text{Attention}(Q,K,V)=\text{softmax}\left(\frac{QK^\top}{\sqrt{d_k}}\right)V
$$

第三，多头时通常令 $d_k = d_{model}/h$，其中 $h$ 是头数。这不是唯一选择，但它让总计算量和总输出宽度更容易保持稳定。

下面这个表先把常见设置压缩成一个可检查的边界表。

| 符号 | 含义 | 常见取值关系 | 张量/矩阵形状 | 目的 |
|---|---|---|---|---|
| $d_{model}$ | 主干隐藏维度 | 例如 512、768、1024 | token 表示长度 | 承载输入语义 |
| $h$ | 注意力头数 | 例如 8、12、16 | 标量 | 并行学习不同对齐模式 |
| $d_k$ | query/key 维度 | 常见为 $d_{model}/h$ | 每个头的比较维度 | 控制相似度空间 |
| $W_Q$ | Query 投影矩阵 | 与 $W_K$ 对齐 | $d_{model}\times d_k$ 或 $d_{model}\times(hd_k)$ | 生成查询向量 |
| $W_K$ | Key 投影矩阵 | 与 $W_Q$ 同维 | $d_{model}\times d_k$ 或 $d_{model}\times(hd_k)$ | 生成被匹配特征 |
| $W_V$ | Value 投影矩阵 | 可与 $d_k$ 不同 | $d_{model}\times d_v$ | 提供被加权内容 |

玩具例子先固定一个很小的设置。设 `d_model=8`，`heads=2`，每个头 `d_k=4`。一个 token 输入是 8 维向量，经过 `W_Q` 后可以得到一个 8 维“大 query”，然后 reshape 成两个 4 维 query；也可以直接为两个头各自准备一个 $8\times4$ 的矩阵分别投影。两种写法最后得到的是同一组数，只是实现形式不同。

---

## 核心机制与推导

先看单头。

输入 $X\in\mathbb{R}^{n\times d_{model}}$，其中 $n$ 是序列长度。经过三个线性映射：

$$
Q=XW_Q,\quad K=XW_K,\quad V=XW_V
$$

然后计算注意力输出：

$$
A=\text{softmax}\left(\frac{QK^\top}{\sqrt{d_k}}\right),\quad O=AV
$$

这里的 softmax 可以理解为“把一行分数归一化成概率分布”。如果某个 query 与某个 key 点积更大，对应位置的注意力权重就更高。

### 为什么要除以 $\sqrt{d_k}$$

假设 $q,k\in\mathbb{R}^{d_k}$，每一维近似独立、均值为 0、方差为 1。则点积

$$
q\cdot k=\sum_{j=1}^{d_k} q_jk_j
$$

由于是 $d_k$ 项求和，它的方差近似与 $d_k$ 成正比。也就是说，$d_k$ 越大，未缩放点积越容易变成很大的正数或负数，softmax 就会变得极尖锐，接近 one-hot。尖锐的意思是“概率几乎全压到一个位置”。一旦过尖锐，梯度会变小，训练会更难稳定。

所以除以 $\sqrt{d_k}$` 的作用很直接：把分数尺度拉回到稳定区间。

### 多头的数学形式

多头并不是先算一个总 query，再神秘地分裂，而是并行做多组线性投影。第 $i$ 个头有：

$$
Q_i=XW_i^Q,\quad K_i=XW_i^K,\quad V_i=XW_i^V
$$

其中 $W_i^Q\in\mathbb{R}^{d_{model}\times d_k}$。每个头独立计算：

$$
\text{head}_i=\text{Attention}(Q_i,K_i,V_i)
$$

最后拼接：

$$
\text{MultiHead}(X)=\text{Concat}(\text{head}_1,\ldots,\text{head}_h)W^O
$$

### 等效性证明

把所有头的 Query 投影矩阵按列拼接：

$$
W_Q^{all}=[W_1^Q\; W_2^Q\; \cdots \; W_h^Q]
$$

则

$$
XW_Q^{all} = [XW_1^Q \; XW_2^Q \; \cdots \; XW_h^Q]
$$

右边恰好就是每个头独立投影后的结果按列拼接。因此，“逐头算”与“先一次矩阵乘法，再切分头”完全等价。

这也是工程里高效实现的原因：GPU 更擅长一次大矩阵乘法，而不是很多个小矩阵乘法。

再往前一步，有些教材会写成先用一个总投影 $W_Q$，再对每个头用一个小矩阵 $W_i^Q$。如果把两层线性映射连起来：

$$
XW_QW_i^Q = X(W_QW_i^Q)
$$

这说明两次线性变换的复合仍然是一次线性变换。于是我们可以直接学习复合后的矩阵 $W_i^{Q*}=W_QW_i^Q$。这就是“多头拆分可由复合投影直接替代”的数学基础。

### 玩具例子

设一个 token 表示为

$$
x=[1,2]
$$

两个头的 Query 投影矩阵分别为

$$
W_1^Q=
\begin{bmatrix}
1 & 0\\
0 & 1
\end{bmatrix},\quad
W_2^Q=
\begin{bmatrix}
1 & 1\\
1 & -1
\end{bmatrix}
$$

则

$$
xW_1^Q=[1,2],\quad xW_2^Q=[3,-1]
$$

拼接后得到

$$
[1,2,3,-1]
$$

如果把两者按列拼成一个大矩阵

$$
W_Q^{all}=
\begin{bmatrix}
1 & 0 & 1 & 1\\
0 & 1 & 1 & -1
\end{bmatrix}
$$

则一次乘法直接得到

$$
xW_Q^{all}=[1,2,3,-1]
$$

结果完全一致。这个例子说明，多头并不改变线性代数本质，只是把输出维度分成多个语义子空间。

### 真实工程例子

在 BERT、GPT、机器翻译编码器里，每一层 self-attention 都会对同一个输入序列 $X$ 构造 `Q/K/V`。假设一层设置为 `d_model=768, heads=12`，则常见是每头 `d_k=64`。这时 `W_Q` 常实现为一个 `768 x 768` 的大矩阵，前向时把输出 reshape 成 `[batch, seq_len, 12, 64]`。这种实现没有改变数学含义，但显著提升了并行效率，也更适合张量库统一优化。

---

## 代码实现

下面给一个最小可运行的 Python 例子，不依赖 PyTorch，只用 `numpy`，这样更容易直接验证数学等价性与缩放效果。

```python
import math
import numpy as np

np.random.seed(7)

def softmax(x, axis=-1):
    x = x - np.max(x, axis=axis, keepdims=True)
    e = np.exp(x)
    return e / e.sum(axis=axis, keepdims=True)

# 一个 batch，1 个 token，d_model=4
x = np.array([[1.0, 0.0, 0.0, 1.0]])  # shape: (1, 4)

d_model = 4
heads = 2
d_k = d_model // heads

# 为两个头各自创建 W_i^Q
Wq_heads = np.random.randn(heads, d_model, d_k) / math.sqrt(d_model)

# 逐头计算 query
multi_q = np.einsum("bd,hdm->bhm", x, Wq_heads)   # shape: (1, 2, 2)

# 把两个头的 W_i^Q 沿列拼成一个大矩阵
Wq_combined = np.concatenate([Wq_heads[i] for i in range(heads)], axis=1)  # (4, 4)

# 一次乘法得到 combined query
combined = x @ Wq_combined  # shape: (1, 4)

# 等价性验证
assert np.allclose(multi_q.reshape(1, -1), combined, atol=1e-8)

# 准备 key，观察缩放影响
keys = np.random.randn(3, d_k)

# 取第一个头
q0 = multi_q[0, 0]  # shape: (2,)

scores_scaled = (q0 @ keys.T) / math.sqrt(d_k)
scores_peaky = (q0 @ keys.T) * 10.0 / math.sqrt(d_k)

attn_scaled = softmax(scores_scaled)
attn_peaky = softmax(scores_peaky)

# 概率应为 1
assert np.isclose(attn_scaled.sum(), 1.0)
assert np.isclose(attn_peaky.sum(), 1.0)

# 放大后分布通常更尖锐：最大概率更大
assert attn_peaky.max() >= attn_scaled.max()

print("multi_q =", multi_q)
print("combined =", combined)
print("attn_scaled =", attn_scaled)
print("attn_peaky =", attn_peaky)
```

这个例子验证了两件事。

第一，`multi_q.reshape(1, -1)` 与 `combined` 相等，说明“逐头投影”与“拼大矩阵一次投影”完全等价。

第二，把分数额外乘 `10` 后，`attn_peaky` 往往比 `attn_scaled` 更集中，说明 softmax 对输入尺度非常敏感。工程上若没有 $\sqrt{d_k}$ 缩放，增大的维度就会起到类似“无意中乘大常数”的效果。

再用一个简表对应代码里的关键变量。

| 变量名 | 含义 | 形状 |
|---|---|---|
| `Wq_heads` | 每个头独立的 Query 投影矩阵 | `(heads, d_model, d_k)` |
| `multi_q` | 逐头投影后的 query | `(batch, heads, d_k)` |
| `Wq_combined` | 拼接后的总 Query 矩阵 | `(d_model, heads * d_k)` |
| `combined` | 一次矩阵乘法得到的总 query | `(batch, heads * d_k)` |
| `attn_scaled` | 正常缩放下的注意力分布 | `(num_keys,)` |
| `attn_peaky` | 人为放大分数后的更尖锐分布 | `(num_keys,)` |

如果改成 PyTorch，常见写法就是用一个 `nn.Linear(d_model, heads * d_k)`，然后 `view` 或 `reshape` 成多头格式。这正是上面等价性的直接工程化实现。

---

## 工程权衡与常见坑

`W_Q` 的数学形式很简单，但工程里常出问题的地方并不在公式本身，而在尺度控制、初始化和张量布局。

先看最重要的一点：缩放。

| 策略 | 分数尺度 | softmax 输出 | 训练稳定性 |
|---|---|---|---|
| 无缩放 | 随 $d_k$ 增大而变大 | 容易过度集中 | 容易梯度变小 |
| 只做缩放，无良好初始化 | 初期可控但波动较大 | 分布可能随机偏置 | 一般可训练 |
| Xavier/Glorot + $\sqrt{d_k}$ 缩放 | 初期方差更平衡 | 通常不过尖锐 | 最常见且稳定 |
| 高方差初始化 + 无缩放 | 极易爆分数 | 接近 one-hot | 常见不收敛或训练抖动 |

### 坑 1：忽略 $\sqrt{d_k}$

当 `d_k=64`、`128` 甚至更大时，未缩放点积会迅速增大。softmax 一旦过尖锐，模型会在一开始就把几乎全部注意力压到少数 token 上，梯度难以传播。对初学者来说，可以把它理解为“模型太早下结论，而且非常自信”。

### 坑 2：初始化方差过大

初始化就是训练开始前给权重一个随机起点。对注意力层来说，若 `W_Q`、`W_K` 初始化方差过大，$QK^\top$ 的分布就会被拉宽，softmax 更容易过尖锐。Xavier/Glorot 初始化的价值在于让线性层输入输出的方差尽量平衡，减少前几步训练就出现极端分数的概率。

一些研究还会讨论 conditioned initialization，也就是额外约束谱性质。谱性质可以粗略理解为“矩阵不同方向上的拉伸是否均衡”。如果矩阵条件数太差，某些方向会被放大太多，注意力分布可能偏向少数模式，导致训练不稳。

### 坑 3：头数与维度不协调

轻量模型经常想减少参数，于是有人直接减小 `heads` 或随意改 `d_k`。这不是不能做，但必须联动考虑：
1. `d_model` 是否还能被 `heads` 整除。
2. reshape 后每头维度是否一致。
3. 缩放是否仍使用新的 $\sqrt{d_k}$。
4. 初始化方差是否要随新的投影宽度重新检查。

### 坑 4：把“多头更强”理解成绝对结论

多头的好处是并行子空间建模，但不是头数越多越好。头过多时，每头维度过小，单头表达能力会下降；同时实现复杂度、cache 压力、通信开销会上升。很多工程调优最后得到的不是“更多头”，而是“更合适的头宽与更稳的初始化”。

---

## 替代方案与适用边界

一种常见替代实现是不用显式维护很多个 `W_i^Q`，而是直接维护一个总矩阵：

$$
W_Q^{all}\in\mathbb{R}^{d_{model}\times (h d_k)}
$$

前向时写成：

$$
Q = XW_Q^{all}
$$

然后 reshape：

$$
Q \rightarrow [batch, seq, h, d_k]
$$

伪代码可以写成：

```text
Q = Linear(X, d_model -> h * d_k)
Q = reshape(Q, batch, seq, h, d_k)
```

这就是大多数深度学习框架的标准做法。它的优点是：

1. 参数管理简单，只需一层线性层。
2. 更容易做 fused kernel，也就是把多个操作合并成更快的底层实现。
3. 与 GPU/TPU 的并行矩阵乘法习惯更匹配。

但它的适用边界也很明确。

第一，`d_model` 通常需要能被 `heads` 整除，否则每头维度不均匀，reshape 就不自然。

第二，如果你减少 `heads`、增大 `d_k`，缩放常数就必须同步改成新的 $\sqrt{d_k}$。很多轻量化改法失败，不是因为思路错，而是因为缩放和初始化没有跟着变。

第三，显式拆头有时对研究更友好。比如你想单独分析每个头的作用，或者给不同头施加不同正则项，显式的 `W_i^Q` 更方便读写与诊断。

给初学者的轻量模型例子是：有些实现直接把 `W_Q` 设成 `d_model x d_model`，然后在 forward 里切成多个头，而不是在参数层面保存多个小矩阵。这种方式没有数学问题，本质上仍然是在做多头投影；前提只是每次切片尺寸一致、`d_model` 能被 `heads` 整除，并且缩放按每头的 `d_k` 来算。

因此，替代方案不是“换一种数学”，而是“换一种参数组织和计算路径”。只要线性映射与切分规则一致，结果就等价。

---

## 参考资料

- Vaswani et al., *Attention Is All You Need*。Transformer 原始论文，定义了 scaled dot-product attention 与 multi-head attention 的标准形式，也是 `d_model`、`h`、`d_k` 关系的主要来源。
- AI StackExchange: 关于为什么多头实现里既会看到整体投影，又会看到分头投影的讨论。用途是理解“复合线性映射”和“拼接实现”等价。
- AI StackExchange: 关于多头注意力权重初始化的讨论。用途是补充工程视角，说明为什么 Xavier/Glorot 是常见默认选择。
- OpenReview: Conditioned Initialization for Attention。用途是进一步理解初始化的谱性质如何影响注意力分布和训练稳定性。
- 各类 Transformer 代码实现文档与源码，如 PyTorch/Hugging Face 中的 `Linear -> reshape -> transpose` 模式。用途是把数学形式映射到真实工程实现。
