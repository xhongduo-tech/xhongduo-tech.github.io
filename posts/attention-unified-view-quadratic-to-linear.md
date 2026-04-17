## 核心结论

注意力机制可以统一写成一个式子：

$$
Y = f(Q, K)\cdot V
$$

这里的 $Q$ 是 query，表示“当前 token 想找什么”；$K$ 是 key，表示“每个 token 能提供什么线索”；$V$ 是 value，表示“真正要被聚合的内容”。白话说，$f(Q,K)$ 决定“看谁、看多少”，再把这些权重乘到 $V$ 上，得到输出 $Y$。

从这个统一视角看，常见变体的差别不在“有没有注意力”，而在“怎么构造 $f(Q,K)$”：

| 方法 | $f(Q,K)$ 的典型形式 | 时间复杂度 | 是否显式构造 $N\times N$ 权重矩阵 | 主要特点 |
|---|---|---:|---|---|
| 标准 Softmax Attention | $\text{softmax}(QK^\top/\sqrt{d})$ | $O(N^2d)$ | 是 | 表达力强，代价高 |
| 稀疏 Attention | $\text{SparseMask}(QK^\top)$ 后再归一化 | 通常低于 $O(N^2d)$ | 部分 | 只看局部或 top-k |
| 线性 Attention | $\phi(Q)\phi(K)^\top$ | $O(Nd^2)$ | 否 | 利用结合律降复杂度 |
| SSM / Mamba-2 视角 | 半可分矩阵或递推核 | 视实现而定，通常接近线性 | 否 | 递推与并行两种写法可切换 |

结论可以压缩成一句话：所有注意力变体都在做同一件事，即重构 $f(Q,K)$；核心权衡轴是表达力、效率、实现复杂度。标准 softmax 往往表达力最高，线性注意力和 SSM 往往效率最高，但实现要求更高。

---

## 问题定义与边界

先定义问题。给定长度为 $N$ 的序列，每个位置有一个表示向量。模型要让某个位置读取全局信息，就需要一个“谁影响谁”的权重矩阵。标准 Transformer 的做法是：

$$
A = \text{softmax}(QK^\top/\sqrt{d}),\quad Y = AV
$$

其中 $A\in \mathbb{R}^{N\times N}$。这一步的瓶颈很直接：当序列长度翻倍时，权重矩阵大小约变成 4 倍，计算和显存都被 $N^2$ 支配。

这篇文章只讨论一个边界内的问题：如何定义和实现 $f(Q,K)$。也就是说，我们只看“权重算子”本身，不讨论以下内容：

| 不讨论项 | 原因 |
|---|---|
| 预训练目标 | 它影响模型学什么，不直接改变注意力算子的数学结构 |
| 编码器/解码器整体架构 | 它属于系统级设计，不是本文的推导重点 |
| 多模态对齐细节 | 仍可映射到注意力，但会引入额外任务背景 |
| 优化器、学习率调度 | 影响训练稳定性，不改变 $f(Q,K)$ 的基本形式 |

为什么要统一成 $Y=f(Q,K)\cdot V$？因为这样可以把不同论文里的术语压缩到一个框架中：

1. 如果 $f$ 是 softmax 归一化后的相似度矩阵，就是标准 attention。
2. 如果 $f$ 只保留局部窗口或 top-k，就是稀疏 attention。
3. 如果 $f$ 被写成 $\phi(Q)\phi(K)^\top$，就能通过乘法结合律避免构造完整 $N\times N$ 矩阵。
4. 如果 $f$ 具有半可分结构，就能进一步写成递推形式，这就是 SSM 与 attention 的连接点。

一个玩具例子可以先建立直觉。假设序列只有 3 个 token，某个 query 与三个 key 的打分分别为 $[2,1,0]$。  
标准 softmax 会把它变成一组和为 1 的权重，比如近似 $[0.67, 0.24, 0.09]$；  
top-k 稀疏化如果只保留最大的 2 个，就变成近似 $[0.73, 0.27, 0]$；  
线性注意力则不一定得到“概率分布”，它更像是先把 query 和 key 映射到某个新特征空间，再在那个空间里聚合。

所以，“从二次到线性”不是把注意力换掉，而是把 $f$ 的构造方式换掉。

---

## 核心机制与推导

先看标准 attention。给定 $Q,K,V\in \mathbb{R}^{N\times d}$，有：

$$
Y = \text{softmax}\left(\frac{QK^\top}{\sqrt d}\right)V
$$

这里 $\sqrt d$ 是缩放项，作用是避免点积随着维度增大而数值过大。白话说，它是在控制 softmax 输入的“温度”。

### 1. 标准 softmax attention

标准形式可以拆成两步：

1. 计算两两相似度 $S = QK^\top$
2. 对每一行做 softmax，得到权重，再乘 $V$

优点是每个 query 都能看到所有 key，且每一行被归一化为概率分布。缺点是必须显式处理 $N\times N$ 的矩阵。

### 2. 稀疏 attention

稀疏 attention 的思路不是否定 softmax，而是在 softmax 之前或之后施加结构约束。常见做法有两类：

1. 固定模式：只允许看局部窗口、跨步位置、块内位置
2. 数据依赖模式：只保留 top-k 高分项

可写成：

$$
Y = \text{Normalize}(\text{Mask}(QK^\top))V
$$

这里 `Mask` 的意思是“把不允许连接的位置变成 0 或 $-\infty$”。白话说，就是提前规定“哪些边存在”。

它减少了无效计算，但代价是表达空间被限制。如果真实依赖刚好不在保留模式里，模型就需要更多层来间接传递信息。

### 3. 线性 attention

线性 attention 的关键不是稀疏，而是分解。它把权重写成：

$$
f(Q,K)=\phi(Q)\phi(K)^\top
$$

于是：

$$
Y = \phi(Q)\phi(K)^\top V
$$

根据结合律：

$$
Y = \phi(Q)\big(\phi(K)^\top V\big)
$$

这一步很重要，因为 $\phi(K)^\top V$ 的形状通常是 $d\times d_v$，不再是 $N\times N$。如果特征维度 $d$ 远小于序列长度 $N$，复杂度就从 $O(N^2d)$ 变成近似 $O(Nd^2)$。

这也是“线性”名字的来源：当 $d$ 固定时，对序列长度 $N$ 近似线性增长。

下面用一个玩具例子手算。设：

$$
Q=
\begin{bmatrix}
1 & 0\\
0 & 1
\end{bmatrix},
\quad
K=
\begin{bmatrix}
1 & 2\\
3 & 4
\end{bmatrix},
\quad
V=
\begin{bmatrix}
1\\
2
\end{bmatrix}
$$

为简化，取 $\phi(x)=x$。那么：

$$
K^\top V=
\begin{bmatrix}
1 & 3\\
2 & 4
\end{bmatrix}
\begin{bmatrix}
1\\
2
\end{bmatrix}
=
\begin{bmatrix}
7\\
10
\end{bmatrix}
$$

再左乘 $Q$：

$$
Y = Q(K^\top V)=
\begin{bmatrix}
1 & 0\\
0 & 1
\end{bmatrix}
\begin{bmatrix}
7\\
10
\end{bmatrix}
=
\begin{bmatrix}
7\\
10
\end{bmatrix}
$$

这里没有显式构造 $QK^\top$ 的 $2\times 2$ 权重矩阵，而是先把所有 key-value 信息压缩成一个“全局统计量” $K^\top V$，再由每个 query 去读取。白话说，先汇总，再查询。

但这也带来一个问题：标准 softmax attention 的每个 query 都有独立的归一化分布，而线性 attention 更像共享了某种汇总结果，因此表达力通常更受限。

### 4. 因果线性 attention

如果任务是语言建模，当前位置只能看过去，不能看未来。这叫因果约束，白话说就是“时间不能穿越”。

标准 causal attention 通过下三角 mask 实现。线性 attention 不能直接算 $\phi(K)^\top V$ 的全局版本，否则会泄漏未来信息。正确做法是前缀累积：

$$
S_t = \sum_{i\le t}\phi(k_i)v_i^\top,\quad
z_t = \sum_{i\le t}\phi(k_i)
$$

输出通常写成：

$$
y_t = \frac{\phi(q_t)^\top S_t}{\phi(q_t)^\top z_t}
$$

分母是归一化项，用来避免输出尺度随序列长度漂移。

### 5. SSM / Mamba-2 的统一视角

SSM 是状态空间模型，白话说，它把长序列处理写成“状态递推”：每来一个 token，就更新一次内部状态。传统看法里，attention 和 SSM 是两条路线；Mamba-2 相关工作说明，两者可以在更高层统一。

关键点是：某些 attention 核对应的权重矩阵具有半可分结构。半可分矩阵可以粗略理解为“矩阵的下三角或上三角块可以写成低秩乘积”，于是既能并行地看成矩阵乘法，也能递推地看成状态更新。

统一直觉可以写成：

- attention 视角：直接构造一个结构化的 $f(Q,K)$
- SSM 视角：不显式构造矩阵，而是维护一个随时间更新的状态

两者在某些条件下描述的是同一类算子，只是计算路径不同。于是“从二次到线性”不只是工程优化，也是表示形式的变化：从显式全连接矩阵，走向结构化矩阵，再走向递推系统。

一个真实工程例子是长上下文语言模型。假设上下文长度从 4K 提升到 16K，标准 attention 的计算和显存压力会明显增长，尤其在多头、多层时更突出。工程上常见策略是：

1. 保留少量 softmax heads，负责高表达力的精细对齐
2. 引入线性 attention 或 SSM heads，负责长程信息压缩
3. 用块化或 head 共享减少实际 kernel 调度成本

这类混合方案的核心仍然是同一个式子：不同 head 只是选择了不同的 $f(Q,K)$。

---

## 代码实现

下面先给出一个最小可运行的 Python 版本，演示标准 softmax attention 与线性 attention 的统一实现。这里用 `numpy`，并加入 `assert` 做基本校验。

```python
import numpy as np

def softmax(x, axis=-1):
    x = x - np.max(x, axis=axis, keepdims=True)
    e = np.exp(x)
    return e / np.sum(e, axis=axis, keepdims=True)

def softmax_attention(Q, K, V):
    d = Q.shape[-1]
    scores = Q @ K.T / np.sqrt(d)
    weights = softmax(scores, axis=-1)
    return weights @ V

def elu_feature_map(x):
    # 常见稳定做法之一：ELU + 1，保证非负
    return np.where(x > 0, x, np.exp(x) - 1) + 1.0

def linear_attention(Q, K, V, phi=elu_feature_map, eps=1e-8):
    Qp = phi(Q)
    Kp = phi(K)

    kv = Kp.T @ V              # 先汇总 key-value
    z = Kp.sum(axis=0)         # 归一化分母的 key 统计量

    numerator = Qp @ kv
    denominator = (Qp @ z[:, None]) + eps
    return numerator / denominator

# 玩具例子
Q = np.array([[1.0, 0.0],
              [0.0, 1.0]])
K = np.array([[1.0, 2.0],
              [3.0, 4.0]])
V = np.array([[1.0],
              [2.0]])

y_softmax = softmax_attention(Q, K, V)
y_linear = linear_attention(Q, K, V, phi=lambda x: x + 1.0)

assert y_softmax.shape == (2, 1)
assert y_linear.shape == (2, 1)
assert np.isfinite(y_softmax).all()
assert np.isfinite(y_linear).all()

# 线性 attention 的“先算 K^T V 再乘 Q”与直接写法一致（在不加归一化时）
Q2 = np.array([[1.0, 0.0],
               [0.0, 1.0]])
K2 = np.array([[1.0, 2.0],
               [3.0, 4.0]])
V2 = np.array([[1.0],
               [2.0]])

direct = (Q2 @ K2.T) @ V2
reordered = Q2 @ (K2.T @ V2)
assert np.allclose(direct, reordered)

print("softmax attention:\n", y_softmax)
print("linear attention:\n", y_linear)
print("associativity check passed")
```

这段代码体现了统一框架：

- `softmax_attention` 里，`f(Q,K)=softmax(QK^T/sqrt(d))`
- `linear_attention` 里，`f(Q,K)=phi(Q)phi(K)^T`，但实现时不显式构造它

如果要做因果版本，核心是把“全局汇总”改成“前缀汇总”。伪代码如下：

```python
def causal_linear_attention(Q, K, V, phi, eps=1e-8):
    Qp = phi(Q)
    Kp = phi(K)

    S = np.zeros((K.shape[1], V.shape[1]))
    Z = np.zeros((K.shape[1],))
    Y = []

    for t in range(Q.shape[0]):
        kt = Kp[t:t+1].T
        vt = V[t:t+1]
        S = S + kt @ vt
        Z = Z + Kp[t]

        qt = Qp[t:t+1]
        num = qt @ S
        den = qt @ Z[:, None] + eps
        Y.append(num / den)

    return np.concatenate(Y, axis=0)
```

真实工程里还要补上三类细节：

| 细节 | 为什么重要 | 常见做法 |
|---|---|---|
| 掩码 | 防止未来信息泄漏或 padding 污染 | 下三角 mask、prefix scan、块递推 |
| 归一化 | 防止线性 attention 输出尺度漂移 | 显式分母、RMSNorm、稳定 feature map |
| 数值稳定 | 防止指数爆炸或分母接近 0 | `ELU+1`、`ReLU+eps`、分块累计 |

如果进一步实现 SSM/SSD 风格的模块，代码结构通常不是“先算完整矩阵再乘”，而是“按块扫描状态”。这时需要特别注意：一旦你在实现里偷偷恢复了显式两两交互，复杂度就会退回 $O(N^2)$。

---

## 工程权衡与常见坑

理论上把复杂度从二次降到线性很吸引人，但工程里真正难的是“不在别处把成本补回来”。

先看三条主权衡：

| 维度 | 标准 softmax attention | 稀疏 attention | 线性 attention / SSM |
|---|---|---|---|
| 表达力 | 高 | 中，依赖稀疏模式 | 中到较高，依赖核设计与结构约束 |
| 效率 | 长序列差 | 通常较好 | 长序列最好 |
| 实现复杂度 | 中 | 中到高 | 高 |

### 常见坑 1：以为去掉 softmax 只是“近似”

不是。softmax 不只是一个激活函数，它还定义了“每一行归一化”的概率结构。没有它，输出不再天然是加权平均，尺度和分布都会变化。如果直接把 `QK^T` 换成别的相似度而不补归一化，模型行为会明显不同。

### 常见坑 2：因果任务里错误复用全局统计量

在线性 attention 中，`K^T V` 很诱人，因为它能一次算完。但在自回归任务里，全局版本会让位置 $t$ 读到未来 token 的信息。这是数据泄漏，不是小误差，而是训练目标被破坏。

### 常见坑 3：$\phi$ 选得太激进

$\phi$ 是 feature map，白话说，就是把原始向量映射到另一个空间，让内积能近似某种注意力核。  
如果取 $\phi(x)=\exp(x)$，表达能力强，但数值风险大；  
如果取 ReLU，稳定一些，但可能丢失符号信息；  
`ELU+1` 常见于实践，因为它保证非负且相对稳定。

### 常见坑 4：理论是线性，实现却不是

很多实现表面上是线性 attention，实际在以下位置退化：

1. 为了调试或可视化，显式保存了 $N\times N$ 权重
2. 分块不合理，导致 kernel launch 过多
3. 多头间没有共享中间统计量
4. padding 和 mask 处理仍用逐对构造

结果是理论复杂度没问题，吞吐却没有明显提升。

### 常见坑 5：忽略常数项

复杂度写成 $O(Nd^2)$ 并不自动代表更快。如果 $d$ 大、head 多、归一化复杂、block 切换频繁，短序列上 softmax attention 仍可能更快。这也是为什么很多实际系统并不完全替换 softmax，而是采用混合头。

下面给出一个工程避坑表：

| 坑 | 典型后果 | 规避策略 |
|---|---|---|
| 忽略归一化 | 输出尺度漂移，训练不稳 | 加显式分母或稳定归一化 |
| 因果掩码处理错误 | 未来信息泄漏 | 使用前缀累计或块级 scan |
| $\phi$ 数值不稳 | 梯度爆炸/NaN | 选非负稳定核并加 `eps` |
| 破坏半可分结构 | 退化回 $O(N^2)$ | 保持块结构与递推接口 |
| 只看理论 FLOPs | 实测速度不升反降 | 同时检查显存、kernel 数量、吞吐 |

真实工程例子可以看长文本生成或长音频建模。假设模型要处理 16K token，上层如果全部使用标准 attention，训练显存和延迟都可能成为瓶颈。工程上常做法不是“全换成线性”，而是：

1. 关键层保留 softmax attention，负责精确对齐
2. 长程层改用线性 attention 或 SSM，负责远距离记忆
3. 用块扫描、融合 kernel、共享统计量控制常数项

这反映了一个实际结论：工程目标不是最优雅的公式，而是单位硬件成本下的有效表达能力。

---

## 替代方案与适用边界

统一视角有一个重要价值：它能帮助你判断什么时候该选哪种 $f(Q,K)$。

### 1. 什么时候优先用标准 softmax attention

适合场景：

- 上下文不算特别长
- 模型特别依赖精细的 token-to-token 对齐
- 你更关心效果上限而不是极致吞吐

典型例子是中短文本理解、复杂代码补全、需要精确位置对齐的多模态模块。softmax 的优势在于，每个 query 都有独立分布，表达最细。

### 2. 什么时候优先用稀疏 attention

适合场景：

- 序列较长，但依赖有明显结构
- 你知道大多数连接本来就没必要保留
- 可以接受“只看一部分”带来的近似误差

例如文档模型里，局部窗口加少量全局 token 是常见模式。因为大量邻近 token 确实更相关，稀疏模式能直接利用这一先验。

### 3. 什么时候优先用线性 attention

适合场景：

- 序列很长，$N^2$ 成本已不可接受
- 你愿意用一定表达力损失换效率
- 模型需要流式处理或在线更新

例如日志流分析、长序列语音建模、边缘设备上的增量推理。线性 attention 可以把“先汇总再查询”做成天然流式。

### 4. 什么时候考虑 SSM / Mamba-2 风格方案

适合场景：

- 你需要超长序列
- 你同时在意训练并行性和推理递推性
- 团队能承担更高实现复杂度

它的价值不只在于快，还在于把“并行训练”和“递推推理”统一到同一结构里。对于超长上下文任务，这种结构性优势常常比单次算子的优化更重要。

下面给一个适用边界表：

| 方案 | 最适合的长度区间 | 训练难度 | 推理方式 | 适用边界 |
|---|---:|---:|---|---|
| Softmax Attention | 短到中长 | 低到中 | 并行 | 效果优先 |
| 稀疏 Attention | 中长到长 | 中 | 并行 | 依赖结构明显 |
| 线性 Attention | 长到超长 | 中到高 | 并行或流式 | 效率优先 |
| SSM / SSD | 长到超长 | 高 | 递推或并行切换 | 系统级优化优先 |

一个实用判断标准是：

- 如果瓶颈还没到 $N^2$，先别急着替换 softmax
- 如果长序列已经主导成本，优先考虑线性或结构化方案
- 如果你既要长上下文又要低延迟推理，SSM/SSD 价值会更大
- 如果团队缺少底层 kernel 和数值稳定经验，混合方案往往比全量切换更稳

因此，“从二次到线性”不是单向替代链，而是一组可组合选项。统一公式 $Y=f(Q,K)\cdot V$ 的意义，就是把这些选项放到同一坐标系里比较。

---

## 参考资料

| 资料 | 主题 | 核心贡献 |
|---|---|---|
| Goomba Lab, *Mamba-2 Part II: The Theory* | Mamba-2 理论 | 从半可分矩阵角度解释 attention 与 SSM 的统一 |
| Graphcore Research, *Transformers are SSMs* 相关讲解 | SSD 与结构统一 | 展示 Transformer 型算子与状态空间模型之间的等价视角 |
| Structured State-Space Duality 相关综述 | 结构化状态空间 | 说明并行矩阵形式与递推形式如何互相转换 |
| 线性 Attention 系列论文与讲解 | 核方法注意力 | 解释 $\phi(Q)\phi(K)^\top$ 如何通过结合律降复杂度 |
| 稀疏 Attention 系列工作 | 长序列注意力 | 展示局部窗口、块稀疏、top-k 等稀疏模式的工程收益 |

建议阅读顺序：

1. 先读 Goomba Lab 的理论讲解，建立统一框架
2. 再看 Graphcore 对 SSD 的工程化解释，理解并行与递推如何切换
3. 最后补线性 attention 和稀疏 attention 的具体实现文章，对照不同 $f(Q,K)$ 的取法
