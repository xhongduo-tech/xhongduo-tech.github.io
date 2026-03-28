## 核心结论

Linear Attention 的核心不是“把 softmax 算快一点”，而是把注意力改写成一个**可分离核**形式。可分离核的白话解释是：相似度不再直接由 $q^\top k$ 经过 softmax 得到，而是先把 $q,k$ 分别映射到同一个特征空间，再做一次点积。

标准注意力写成：

$$
\mathrm{Attn}(Q,K,V)=\mathrm{softmax}\left(\frac{QK^\top}{\sqrt d}\right)V
$$

它的瓶颈是必须显式形成 $n\times n$ 的注意力矩阵。若把相似度替换为

$$
\kappa(q,k)=\phi(q)^\top \phi(k),
$$

则第 $i$ 个位置的输出可写成

$$
y_i=\frac{\sum_{j=1}^n \phi(q_i)^\top \phi(k_j)v_j}{\sum_{j=1}^n \phi(q_i)^\top \phi(k_j)}.
$$

利用矩阵乘法结合律，可以改写成

$$
Y=D^{-1}\bigl(\phi(Q)\,(\phi(K)^\top V)\bigr),
\quad
D=\mathrm{diag}\bigl(\phi(Q)(\phi(K)^\top \mathbf 1)\bigr).
$$

这一步把复杂度从近似 $O(n^2d)$ 降成 $O(nd^2)$。当序列长度 $n\gg d$ 时，收益非常明显。

更重要的是，自回归推理时它还能写成递归状态更新：

$$
S_t=S_{t-1}+\phi(k_t)v_t^\top,\qquad
z_t=z_{t-1}+\phi(k_t),
$$

$$
y_t=\frac{\phi(q_t)^\top S_t}{\phi(q_t)^\top z_t}.
$$

这说明线性注意力在解码阶段等价于一种 RNN 式状态机，每来一个 token 只更新一次状态，因此单步解码可以做到 $O(1)$ 状态访问和 $O(d^2)$ 计算。这也是 RetNet、GLA 一类结构的重要理论出发点。

但结论不能只说优点。线性注意力通常弱于 softmax 的地方也很明确：softmax 自带竞争性归一化，更容易形成“尖锐聚焦”；简单核函数如 ELU+1 往往得到更平滑、更扩散的注意力分布，所以在语言建模里常出现困惑度劣化。

---

## 问题定义与边界

问题先说清楚：Transformer 的标准注意力为什么贵？因为对每个 query 都要和所有 key 做匹配，得到一个 $n\times n$ 的相似度矩阵。这里的 $n$ 是序列长度，白话说就是上下文里 token 的数量。

如果 $n=1024$，单头注意力就有大约 $10^6$ 个相似度项；若 $n=8192$，这个数字会到 $6.7\times 10^7$。计算和显存都会随 $n^2$ 增长，这就是长上下文的核心瓶颈。

Linear Attention 试图解决的问题不是“完全复现 softmax”，而是：

1. 保留全局上下文聚合能力。
2. 避免显式构造 $QK^\top$。
3. 让训练仍可并行。
4. 让自回归推理变成状态递推。

它的边界也要明确：

| 机制 | 时间复杂度 | 额外中间量 | 是否容易形成尖锐注意力 | 适合场景 |
|---|---:|---:|---|---|
| Softmax Attention | $O(n^2d)$ | $n\times n$ 权重矩阵 | 强 | 中短上下文、高精度对齐 |
| Linear Attention | $O(nd^2)$ | $d\times d$ 或 $r\times d_v$ 聚合状态 | 弱于 softmax | 长序列、流式推理 |
| Hybrid Attention | 介于两者之间 | 取决于混合比例 | 中等到强 | 既要长上下文又要保精度 |

这里还有一个常见误解：Linear Attention 不等于“所有东西都变成 $O(n)$”。更准确地说，它把对序列长度的依赖从平方降成线性，但仍然依赖特征维度 $d$ 或核特征维度 $r$。当 $d$ 很大、序列不长时，它未必比高优化的 FlashAttention 更快。

---

## 核心机制与推导

先从单个 query 的公式开始：

$$
y_i=\frac{\sum_{j=1}^n \kappa(q_i,k_j)v_j}{\sum_{j=1}^n \kappa(q_i,k_j)}.
$$

若核函数满足

$$
\kappa(q_i,k_j)=\phi(q_i)^\top\phi(k_j),
$$

则分子变成

$$
\sum_{j=1}^n \phi(q_i)^\top\phi(k_j)v_j
=
\phi(q_i)^\top \left(\sum_{j=1}^n \phi(k_j)v_j^\top\right).
$$

定义

$$
S=\sum_{j=1}^n \phi(k_j)v_j^\top,\qquad
z=\sum_{j=1}^n \phi(k_j),
$$

则有

$$
y_i=\frac{\phi(q_i)^\top S}{\phi(q_i)^\top z}.
$$

这就是线性注意力的本质：先把所有 key-value 对压缩成一个全局统计量 $S$，再用 query 去“读”它。这里的“读”可以白话理解为：query 不再逐个检查每个 key，而是从一个预先汇总好的记忆里检索信息。

写成矩阵形式就是：

$$
Y=D^{-1}\bigl(\phi(Q)(\phi(K)^\top V)\bigr),
$$

其中

$$
D=\mathrm{diag}\bigl(\phi(Q)(\phi(K)^\top \mathbf 1)\bigr).
$$

### 玩具例子

取最简单情形，令 $\phi(x)=x$，并设

$$
q=[1,1],\quad
K=V=
\begin{bmatrix}
1&0\\
0&1
\end{bmatrix}.
$$

则

$$
S=K^\top V=I,
\qquad
z=\begin{bmatrix}1\\1\end{bmatrix}.
$$

所以

$$
q^\top S=[1,1],
\qquad
q^\top z=2,
$$

最终输出为

$$
y=\frac{1}{2}[1,1]=[0.5,0.5].
$$

这个例子很小，但已经说明关键点：我们没有显式构造两两注意力矩阵，而是先做聚合，再做查询。

### 为什么 softmax 更容易“聚焦”

softmax 的一行权重满足

$$
a_{ij}=\frac{e^{q_i^\top k_j}}{\sum_m e^{q_i^\top k_m}},
\qquad
\sum_j a_{ij}=1.
$$

指数函数会放大较大的相似度，行归一化又让各位置竞争同一份总质量，因此某一个 key 略大时，就可能拿走大部分权重。这就是“尖锐聚焦”。

而线性核常写成

$$
a_{ij}\propto \phi(q_i)^\top \phi(k_j),
$$

它虽然也有归一化分母，但缺少 softmax 那种指数放大和强竞争约束。结果往往是多个位置一起分到中等权重，分布更平滑。Brenndoerfer 的分析和可视化也强调了这一点：简单 ELU 特征会产生更扩散的注意力热图。

### 为什么自回归能变成 RNN

因果掩码下，第 $t$ 个位置只看前缀 $1\ldots t$：

$$
S_t=\sum_{j\le t}\phi(k_j)v_j^\top,
\qquad
z_t=\sum_{j\le t}\phi(k_j).
$$

于是立刻得到递推：

$$
S_t=S_{t-1}+\phi(k_t)v_t^\top,\qquad
z_t=z_{t-1}+\phi(k_t).
$$

输出为

$$
y_t=\frac{\phi(q_t)^\top S_t}{\phi(q_t)^\top z_t}.
$$

这和 RNN 很像，因为模型把过去压缩进固定大小的状态，而不是保留整段历史的显式注意力矩阵。RetNet、GLA 等工作进一步在这个状态更新上加入衰减、门控或分块机制，改善表达力和稳定性。

---

## 代码实现

下面给一个可运行的最小实现。第一个函数是并行矩阵版，适合训练时整段计算；第二个函数是递归版，适合理解自回归推理。这里用 $\phi(x)=\mathrm{ELU}(x)+1$，因为它输出非负值，分母更稳定。

```python
import numpy as np

def elu_plus_one(x):
    # ELU(x) + 1，保证特征非负
    return np.where(x > 0, x + 1.0, np.exp(x))

def linear_attention_batch(Q, K, V, eps=1e-9):
    """
    Q: [n, d]
    K: [n, d]
    V: [n, dv]
    """
    Qp = elu_plus_one(Q)
    Kp = elu_plus_one(K)

    # 先聚合 key-value 统计量
    S = Kp.T @ V              # [d, dv]
    z = Kp.sum(axis=0)        # [d]

    numerator = Qp @ S        # [n, dv]
    denominator = Qp @ z      # [n]
    return numerator / (denominator[:, None] + eps)

def linear_attention_causal_recurrent(Q, K, V, eps=1e-9):
    """
    因果递归版：第 t 个输出只依赖前缀 0..t
    """
    n, d = Q.shape
    dv = V.shape[1]

    S = np.zeros((d, dv), dtype=np.float64)
    z = np.zeros((d,), dtype=np.float64)
    Y = np.zeros((n, dv), dtype=np.float64)

    for t in range(n):
        kp = elu_plus_one(K[t])     # 当前 key 的特征
        qp = elu_plus_one(Q[t])     # 当前 query 的特征

        # 状态更新
        S += np.outer(kp, V[t])
        z += kp

        # 读取状态
        Y[t] = (qp @ S) / (qp @ z + eps)

    return Y

def linear_attention_causal_batch(Q, K, V, eps=1e-9):
    """
    因果矩阵版，用前缀和验证递归版正确性
    """
    Qp = elu_plus_one(Q)
    Kp = elu_plus_one(K)
    n, d = Q.shape
    dv = V.shape[1]

    Y = np.zeros((n, dv), dtype=np.float64)
    for t in range(n):
        S_t = Kp[:t+1].T @ V[:t+1]
        z_t = Kp[:t+1].sum(axis=0)
        Y[t] = (Qp[t] @ S_t) / (Qp[t] @ z_t + eps)
    return Y

# 玩具例子
Q = np.array([[1.0, 1.0]])
K = np.array([[1.0, 0.0], [0.0, 1.0]])
V = K.copy()

Y = linear_attention_batch(Q, K, V)
assert Y.shape == (1, 2)
assert np.allclose(Y[0], np.array([0.5, 0.5]), atol=1e-6)

# 验证递归版与因果批量版等价
Q2 = np.array([[1.0, 0.0], [0.5, 0.5], [0.0, 1.0]])
K2 = np.array([[1.0, 0.0], [0.0, 1.0], [1.0, 1.0]])
V2 = np.array([[2.0, 0.0], [0.0, 4.0], [1.0, 1.0]])

Yr = linear_attention_causal_recurrent(Q2, K2, V2)
Yb = linear_attention_causal_batch(Q2, K2, V2)
assert np.allclose(Yr, Yb, atol=1e-8)

print("ok")
```

真实工程里，训练通常直接用矩阵版或 chunkwise 版，以便充分并行；推理则维护每层的 $S_t,z_t$ 状态缓存，不再回看整段 KV cache。这也是它和标准 Transformer 解码路径最不一样的地方。

### 真实工程例子

以长文本生成或代码补全为例，softmax 解码通常要保存不断增长的 KV cache；上下文越长，显存和访存压力越大。RetNet 和 GLA 一类模型则把历史压缩成固定大小状态，在长上下文下延迟更稳定，也更适合流式生成、边读边写或低显存部署。

---

## 工程权衡与常见坑

第一类收益是确定的：长序列下显存更低、算子更规整、递归解码更省。第二类问题也同样确定：表达力损失主要集中在“对焦能力”。

| 维度 | Softmax | 简单 Linear Attention | 工程含义 |
|---|---|---|---|
| 长序列扩展 | 一般 | 强 | 上下文越长越有利 |
| 解码显存 | 较高 | 低 | 线性态缓存替代 KV cache |
| 局部精确对齐 | 强 | 较弱 | 检索、复制、对位任务更敏感 |
| 注意力尖锐性 | 强 | 较弱 | 语言建模常导致困惑度劣化 |
| 实现成熟度 | 很高 | 取决于变体 | 高性能 kernel 很关键 |

常见坑主要有五个。

1. 分母数值不稳定。  
如果 $\phi(k)$ 允许负值或大量接近零，$\phi(q)^\top z$ 可能很小，输出会爆。ELU+1、正随机特征、额外 `eps` 都是在解决这个问题。

2. 误以为“理论线性”必然更快。  
若序列不够长，FlashAttention 这类高度优化的 softmax 实现可能仍然更快。复杂度结论是渐近结论，不是每个长度点的 wall-clock 保证。

3. 忽略核函数选择。  
ELU+1 便宜，但对 softmax 的拟合很粗糙；Performer 的 FAVOR+ 更接近 softmax，但要引入更高特征维度和随机近似误差。

4. 因果实现写错更新顺序。  
做自回归时，要先决定当前位置是否允许看见自身。大多数语言建模实现会把当前 token 的 $k_t,v_t$ 更新进状态后，再输出当前位置；训练与推理必须对齐。

5. 只看吞吐，不看质量。  
研究和工程实践普遍发现，原始线性注意力在语言任务上常弱于 softmax。你给出的经验范围“简单 ELU 核困惑度差 5% 到 20%”是合理的工程判断，但它不是一个跨所有模型、数据、参数规模都成立的固定常数，更应理解为“常见量级”。根本原因仍是表达力和注意力锐化能力不足。

缓解思路通常有三类：

1. 加门控或衰减，如 GLA、RetNet。  
2. 用更好的核近似，如 FAVOR+、Magnitude-aware、learnable feature map。  
3. 做混合结构，每隔若干层插入 full attention，保留关键层的高精度对齐。

---

## 替代方案与适用边界

如果任务非常依赖精确检索、复制、跳跃式对齐，纯线性注意力通常不是第一选择。典型例子包括长文问答里的精确证据定位、代码编辑里的变量回指、需要多跳组合推理的场景。

这时有几种替代路线。

| 方案 | 核心想法 | 适用边界 |
|---|---|---|
| FlashAttention | 不改 softmax，只优化实现 | 中等上下文、追求精度 |
| Local/Sliding Window | 只看局部窗口 | 局部依赖强、超长序列 |
| Sparse Attention | 只连部分位置 | 结构化长序列 |
| Hybrid Attention | 多数层线性，少数层 softmax | 想兼顾速度与质量 |
| RetNet / GLA | 在线性递归上加衰减或门控 | 长生成、流式推理 |

其中 RetNet/GLA 最值得单独说明。它们不是简单重复“核分解”这一步，而是在递归状态更新上加了更强的控制项，比如衰减、门控、chunkwise 并行。白话说，就是不再把所有历史一股脑累加，而是允许模型学会“保留什么、忘掉什么”。这能部分补回纯线性注意力缺失的选择性。

所以实践上的边界可以概括为：

1. 如果你的痛点是超长上下文和解码成本，线性递归系结构值得优先考虑。
2. 如果你的痛点是精确对齐质量，softmax 或 hybrid 往往更稳。
3. 如果你想在现有 Transformer 体系里低风险提速，先试 FlashAttention、局部注意力或 hybrid，比直接换成原始 ELU 线性核更保守。

---

## 参考资料

| 资料 | 主题 | 关键结论 | URL |
|---|---|---|---|
| Katharopoulos et al., 2020, Transformers are RNNs | 线性注意力基础论文 | 给出 $\phi(q)^\top\phi(k)$ 分解、因果递归形式与线性复杂度 | https://arxiv.org/abs/2006.16236 |
| Emergent Mind: Linear Attention Variants Overview | 变体综述 | 总结 kernelization、gating、hybrid 等方向及其表达力边界 | https://www.emergentmind.com/topics/linear-attention-variants |
| Emergent Mind: Linear Attention Reformulation | 数学改写 | 强调通过结合律先聚合 $K,V$ 再与 $Q$ 交互 | https://www.emergentmind.com/topics/linear-attention-reformulation |
| Choromanski et al., 2020, Performers | 随机特征近似 softmax | FAVOR+ 用正随机特征近似 softmax 核，兼顾线性复杂度与近似精度 | https://research.google/pubs/rethinking-attention-with-performers/ |
| Sun et al., 2023, RetNet | 递归状态与低成本推理 | 并行训练、递归推理、chunkwise 处理三者统一 | https://arxiv.org/abs/2307.08621 |
| Yang et al., 2023, Gated Linear Attention | 门控线性注意力 | 在线性注意力状态更新中引入门控，改善长度泛化与性能 | https://arxiv.org/abs/2312.06635 |
| Michael Brenndoerfer, 2025 | 工程化解释 | 直观展示 ELU 核的平滑注意力模式与复杂度收益 | https://mbrenndoerfer.com/writing/linear-attention-kernel-feature-maps-efficient-transformers |
