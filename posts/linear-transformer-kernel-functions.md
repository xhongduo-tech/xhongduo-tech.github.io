## 核心结论

Linear Transformer 的核心不是“把 Attention 写快一点”，而是把相似度函数重写成核函数，也就是把原来的两两打分改写为

$$
k(q, k)=\phi(q)^\top \phi(k)
$$

这里的“核函数”可以先理解成一种相似度规则；“特征映射” $\phi$ 可以理解成把原向量变成另一组更适合做内积的坐标。只要这个形式成立，注意力就能从先算完整的 $n \times n$ 相似度矩阵，改成先汇总再查询，因此时间和显存都能从二次复杂度降到线性复杂度。

但线性化只是第一步，真正决定效果的是 $\phi$ 的选择。ELU+1 核实现最简单，能保证非负，训练也稳定，但它对 softmax 注意力的拟合通常偏弱，容易把注意力分布拉平。随机傅里叶特征（RFF）能近似高斯核，理论更完整，但需要较大的特征维度 $m$ 才能把误差压低。cosFormer 进一步引入余弦位置重加权，不再追求“忠实近似 softmax”，而是直接构造一个更适合长序列的线性注意力形式，在很多长文本任务里更实用。

可以把这件事理解成一句话：Linear Transformer 不是单一算法，而是一类“用不同核函数设计线性注意力”的方法族。工程里讨论 Linear Transformer，实际讨论的往往不是“要不要线性化”，而是“选哪种 $\phi$，接受哪种误差和偏置”。

---

## 问题定义与边界

标准 self-attention 的瓶颈在于要显式构造

$$
QK^\top \in \mathbb{R}^{n \times n}
$$

其中 $n$ 是序列长度。对每个 token 都要和所有 token 计算一次相似度，所以时间复杂度和空间复杂度都会随着 $n^2$ 增长。当序列只有几百个 token 时，这个成本通常可接受；当序列变成 8k、16k 甚至更长时，代价会迅速失控。

先看一个复杂度对比：

| 方法 | 核心计算 | 时间复杂度 | 空间复杂度 | 主要问题 |
|---|---|---:|---:|---|
| Softmax Attention | $QK^\top$ 后再乘 $V$ | $O(n^2 d)$ | $O(n^2)$ | 长序列成本高 |
| Linear Attention | 先算 $\phi(K)^\top V$ 再乘 $\phi(Q)$ | $O(n d m)$ | $O(d m)$ 或 $O(n m)$ | 依赖核近似质量 |
| cosFormer | 线性聚合 + 余弦位置重加权 | $O(n d m)$ | 线性级别 | 表达能力依赖重加权设计 |

这里的 $d$ 是隐藏维度，$m$ 是映射后的特征维度。对于软最大值 softmax，本质上它不仅做相似度，还做归一化和放大高分项，因此它具有很强的选择性。线性注意力要解决的问题不是单纯“省掉矩阵”，而是在不显式构造 $QK^\top$ 的前提下，尽量保留这种选择性。

边界也要说清楚。Linear Transformer 解决的是长序列下的计算瓶颈，不等于任何任务上都优于 softmax。如果序列长度本来就不长，比如 512 或 1k 左右，而且任务对精细对齐非常敏感，例如机器翻译里的词级对齐，那么 softmax 仍然可能更稳。反过来，如果任务是长文档检索、会议记录摘要、医学报告理解，模型需要处理上万 token，线性注意力的工程价值就很高。

一个真实工程例子是医学长报告分析。假设输入长度是 10k token，标准 attention 需要处理约 $10^8$ 级别的相似度项，而线性注意力只需要顺序地聚合 $\phi(K)^\top V$ 和归一化项，显存压力会明显下降，更容易在单卡或有限显存下训练与推理。

---

## 核心机制与推导

标准 attention 可以写成：

$$
\mathrm{Att}(Q,K,V)=\mathrm{softmax}(QK^\top)V
$$

softmax 的难点在于，它依赖整行分数一起归一化，所以无法自然拆成“先累加，再逐个查询”的形式。Linear Transformer 的关键做法是引入非负特征映射 $\phi$，把注意力改写为：

$$
\mathrm{Att}(Q,K,V)
=
\frac{\phi(Q)\left(\phi(K)^\top V\right)}
{\phi(Q)\left(\phi(K)^\top \mathbf{1}_n\right)}
$$

这个式子要分开理解。

分子部分：

$$
\phi(Q)\left(\phi(K)^\top V\right)
$$

其中 $\phi(K)^\top V$ 可以先一次性聚合完。它的白话意思是：先把所有 key 按照特征空间坐标汇总成一个“键值统计量”，后面每个 query 来的时候，直接去读取这个统计量。

分母部分：

$$
\phi(Q)\left(\phi(K)^\top \mathbf{1}_n\right)
$$

它相当于归一化因子，作用类似 softmax 分母，避免输出尺度失控。$\mathbf{1}_n$ 就是一列全 1 向量，表示把所有 key 的特征和加起来。

### 玩具例子

取一维标量 $q=0.5,\ k=-0.2$，使用 ELU+1 映射：

$$
\phi_{\text{ELU}+1}(x)=\mathrm{ELU}(x)+1
$$

其中 $\mathrm{ELU}(x)=x$ 当 $x>0$，否则为 $e^x-1$。于是：

$$
\phi(0.5)=1.5,\quad \phi(-0.2)=e^{-0.2}\approx 0.819
$$

那么核值就是

$$
k(q,k)=\phi(q)\phi(k)\approx 1.5 \times 0.819 \approx 1.23
$$

这个例子说明，线性注意力并不需要先构造“所有 token 两两之间的分数表”，而是可以把相似度计算转成特征内积。虽然这个例子只有一维，但机制已经完整。

### 三类常见 $\phi$

| 方案 | 定义 | 优点 | 局限 |
|---|---|---|---|
| ELU+1 | $\phi(x)=\mathrm{ELU}(x)+1$ | 简单、非负、实现成本低 | 对 softmax 拟合较弱，分布偏平滑 |
| RFF | $\phi(x)=\frac{1}{\sqrt{m}}[\cos(\omega_1^\top x),\sin(\omega_1^\top x),\dots]$ | 可近似高斯核，理论清晰 | 需要较大 $m$，方差和数值成本更高 |
| cosFormer | 用余弦位置项重写注意力 | 显式引入局部性，长序列表现好 | 不是直接近似 softmax，设计更任务相关 |

RFF 的直觉是把原始向量投影到若干随机方向，再用正余弦展开，从而用有限维向量近似某类平移不变核。这里的“随机傅里叶特征”可以理解成“用很多随机频率拼出一个相似度函数”。

cosFormer 的关键公式是位置重加权：

$$
\cos\left(\frac{\pi(i-j)}{2M}\right)
=
\cos\left(\frac{\pi i}{2M}\right)\cos\left(\frac{\pi j}{2M}\right)
+
\sin\left(\frac{\pi i}{2M}\right)\sin\left(\frac{\pi j}{2M}\right)
$$

这里 $i,j$ 是 token 位置，$M$ 是尺度上界。这个恒等式的意义不是数学技巧本身，而是把“位置差相关”的项拆成两个可分解的乘积，于是仍然可以保留线性聚合结构。白话说，cosFormer 让模型在线性复杂度下也能知道“离得近的 token 更值得关注”。

---

## 代码实现

下面给一个最小可运行版本，演示 ELU+1 线性注意力。代码只保留核心逻辑，不涉及多头、batch 和 mask。

```python
import math
import numpy as np

def elu_plus_one(x: np.ndarray) -> np.ndarray:
    # 非负映射，便于构造可归一化的线性注意力
    return np.where(x > 0, x + 1.0, np.exp(x))

def linear_attention(Q: np.ndarray, K: np.ndarray, V: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    phi_Q = elu_plus_one(Q)          # [n, d]
    phi_K = elu_plus_one(K)          # [n, d]

    KV = phi_K.T @ V                 # [d, dv]
    Z = phi_K.T @ np.ones((K.shape[0], 1))   # [d, 1]

    out = np.zeros((Q.shape[0], V.shape[1]))
    for i in range(Q.shape[0]):
        numerator = phi_Q[i] @ KV                    # [dv]
        denominator = phi_Q[i] @ Z[:, 0] + eps      # scalar
        out[i] = numerator / denominator
    return out

# 玩具例子
Q = np.array([[0.5], [1.0]])
K = np.array([[-0.2], [0.3]])
V = np.array([[2.0], [4.0]])

out = linear_attention(Q, K, V)
assert out.shape == (2, 1)
assert np.all(out > 0)

# 校验单点映射
phi_q = elu_plus_one(np.array([0.5]))[0]
phi_k = elu_plus_one(np.array([-0.2]))[0]
assert abs(phi_q - 1.5) < 1e-8
assert abs(phi_k - math.exp(-0.2)) < 1e-8
```

这段代码对应的计算流程非常直接：

1. 把 $Q,K$ 都映射到 $\phi$ 空间。
2. 先算 $\phi(K)^\top V$，这是所有 key-value 的全局统计量。
3. 再算 $\phi(K)^\top \mathbf{1}$，这是归一化统计量。
4. 对每个 query 用一次点乘得到输出。

如果换成 RFF，只是把 `elu_plus_one` 换成随机特征映射函数；主干逻辑不变。示意如下：

```python
def rff_phi(X: np.ndarray, W: np.ndarray) -> np.ndarray:
    # X: [n, d], W: [d, m]
    proj = X @ W
    feat = np.concatenate([np.cos(proj), np.sin(proj)], axis=-1)
    return feat / np.sqrt(W.shape[1])
```

cosFormer 的实现重点不是普通的 $\phi$，而是把位置信息编码进可分解项。伪码可以写成：

```python
# position angle: theta_i = pi * i / (2M)
Q_cos = phi(Q) * cos(theta)
Q_sin = phi(Q) * sin(theta)
K_cos = phi(K) * cos(theta)
K_sin = phi(K) * sin(theta)

out = linear(Q_cos, K_cos, V) + linear(Q_sin, K_sin, V)
```

工程里一般不会真的写两个完全独立的函数，而是复用同一套聚合逻辑，只替换输入特征。这样可以减少重复代码，也更容易统一处理 `eps`、causal mask 和 chunked 推理。

真实工程例子可以看长文档检索。假设输入是 8k 到 16k token 的法律文书，实践里常见做法是：对 K/V 分块做 streaming 聚合，分块维护 $\sum \phi(k_t) v_t^\top$ 与 $\sum \phi(k_t)$，每来一个 query block 就增量更新。这种写法可以把显存使用稳定在线性级别，而不是一次性持有完整注意力矩阵。

---

## 工程权衡与常见坑

Linear Transformer 的最大误区，是把“线性复杂度”误认为“无损替代”。实际上，不同核函数带来的偏差很大，很多问题不是算得不对，而是注意力形状已经变了。

### 1. ELU+1 简单，但容易过平滑

ELU+1 的优势是实现最轻，梯度也相对稳定，适合做入门版本或资源有限的场景。但它并不逼近 softmax 的尖锐分布，常见现象是注意力过于平均，模型更像在做平滑聚合。对于需要强选择性的任务，例如代码补全里的远距离精确引用，这可能直接损伤效果。

### 2. RFF 理论更强，但要足够宽

RFF 的误差通常随特征维度满足 $O(1/m)$ 量级下降。这里的“量级”可以理解成：你把特征数翻倍，误差通常才会慢慢下降，不会突然消失。实际工程里，如果 $m=32$ 或 $64$，常常不够；很多设置要到几百甚至上千维，才能稳定逼近目标核。这会带来两个副作用：一是额外乘法开销上升，二是中间张量变大，吞吐不一定比预期高。

### 3. 分母数值稳定性不能忽略

线性注意力的分母是

$$
\phi(q_i)^\top \sum_j \phi(k_j)
$$

如果这个值太小，输出会爆。即便理论上 $\phi$ 非负，有限精度和掩码处理仍然可能造成极小值。工程里必须加 $\varepsilon$，并监控归一化项的最小值分布。

### 4. causal 场景不能直接照搬双向实现

在自回归生成里，第 $i$ 个 token 只能看见前缀 $1 \dots i$。这时不能先对整段 K 全局求和，而要做前缀累积：

$$
S_i=\sum_{j \le i}\phi(k_j)v_j^\top,\quad
z_i=\sum_{j \le i}\phi(k_j)
$$

否则会发生信息泄漏。cosFormer 在 causal 场景下还要同步处理位置索引，因为 $\sin/\cos$ 重加权依赖绝对位置，索引一旦偏移，训练和推理就会不一致。

### 5. chunked 实现要保证统计量一致

很多长序列实现会分块计算，但块间不能只传 hidden state，不传聚合统计量。正确做法是把前缀的 $S$ 和 $z$ 一起滚动维护，否则下一块会缺失历史信息。

下面给一个汇总表：

| 方案 | 精度倾向 | 计算开销 | 数值稳定性 | 适合场景 | 常见坑 |
|---|---|---|---|---|---|
| ELU+1 | 中等偏低 | 低 | 较好 | 资源紧张、原型验证 | 注意力过平滑 |
| RFF / Performer 风格 | 中到高 | 中到高 | 依赖 $m$ 和采样 | 需要更强核近似 | $m$ 太小误差大 |
| cosFormer | 长序列较强 | 中等 | 需处理位置项 | 长文本理解、检索 | causal 索引和归一化 |
| 原始 softmax | 最高上限 | 高 | 成熟 | 中短序列高精度任务 | 长序列显存爆炸 |

---

## 替代方案与适用边界

如果目标是“尽量保留 softmax 行为”，那么 RFF 一类方案更接近这个方向，因为它试图从核逼近角度恢复某种相似度结构。如果目标是“为长序列设计一个更合适的注意力偏置”，那么 cosFormer 这类方法往往更实用，因为它直接把位置局部性写进模型。

对初级工程师来说，最容易落地的判断标准不是论文名字，而是任务条件：

| 任务条件 | 更合适的选择 | 原因 |
|---|---|---|
| 序列较短，1k 左右以内 | 原始 softmax | 精度优先，线性化收益有限 |
| 序列很长，显存紧张 | ELU+1 或 cosFormer | 实现简单，线性收益直接 |
| 希望更接近核近似理论 | RFF / Performer 风格 | 有明确近似框架 |
| 强局部性长文本任务 | cosFormer | 位置重加权更符合任务结构 |
| 自回归生成且极长上下文 | 线性 attention + prefix/chunked | 需要严格维护前缀统计 |

一个玩具判断例子是：你做一个 200 token 的情感分类器，线性注意力通常没有必要，因为 softmax 既简单又成熟。一个真实工程判断例子是：你做 8k token 的长文档问答系统，如果还坚持全量 softmax，很可能训练 batch 很小、推理延迟高，这时 cosFormer 或其他线性注意力方案就有明显工程价值。

还要注意，Linear Transformer 不是唯一替代方案。局部窗口注意力、稀疏注意力、分块注意力、检索增强记忆，都是在长序列里降低成本的常见路径。线性注意力的优势是公式统一、可流式更新；它的边界是表达能力经常受限于核设计，不像 softmax 那样天然尖锐。

所以最实用的结论是：

1. 序列不长时，优先 softmax。
2. 序列很长且资源受限时，优先考虑线性注意力。
3. 追求简单实现时选 ELU+1。
4. 追求核逼近时选 RFF。
5. 追求长序列效果和位置局部性时，优先评估 cosFormer。

---

## 参考资料

1. Katharopoulos et al., *Transformers are RNNs: Fast Autoregressive Transformers with Linear Attention*. 对应本文“核心机制与推导”“代码实现”部分，给出线性注意力的基本分解形式与前缀累积思路。链接：https://arxiv.org/abs/2006.16236

2. Michael Brenndoerfer, *Linear Attention: Breaking the Quadratic Bottleneck with Kernel Feature Maps*. 对应本文“核心结论”“问题定义与边界”“工程权衡与常见坑”部分，适合先建立核函数视角，再理解不同 $\phi$ 的设计差异。链接：https://mbrenndoerfer.com/writing/linear-attention-kernel-feature-maps-efficient-transformers

3. Qin et al., *cosFormer: Rethinking Softmax in Attention*. 对应本文“核心机制与推导”“替代方案与适用边界”部分，重点是余弦位置重加权与长序列 benchmark。链接：https://arxiv.org/abs/2202.08791

4. GoatWu, *cosFormer 阅读笔记*. 对应本文“核心机制与推导”部分，可辅助理解余弦拆分公式在实现中的含义。链接：https://www.goatwu.com/paper-notes/transformer/cosformer.html
