## 核心结论

Linear Attention 的目标很直接：把标准注意力里最贵的那一步，从显式构造 $n \times n$ 的注意力矩阵，改成先做全局汇总、再给每个查询复用这个汇总。这里的“核化”可以先理解成：把原本直接比较 $q$ 和 $k$ 的方式，改成先分别映射到一个特征空间里再做内积。

标准注意力常写成：

$$
\mathrm{Attn}(Q,K,V)=\mathrm{softmax}(QK^\top)V
$$

当序列长度为 $n$、隐藏维度为 $d$ 时，核心代价来自 $QK^\top$，它需要处理 $n^2$ 个查询-键配对，因此时间和显存都会随 $n^2$ 增长。

线性注意力的关键重排是：

$$
\hat O=\phi(Q)\big(\phi(K)^\top V\big)
$$

如果再加入归一化项，可写成逐位置形式：

$$
\hat O_i=\frac{\phi(Q_i)\Big(\sum_{j=1}^n \phi(K_j)^\top V_j\Big)}{\phi(Q_i)\Big(\sum_{j=1}^n \phi(K_j)^\top\Big)}
$$

这一步利用的是矩阵乘法结合律。白话说，先把所有键和值压成一份“全局统计量”，然后每个查询只和这份统计量交互一次，而不是和所有键逐个交互。

玩具例子可以这样理解：传统注意力像“每个问题都重新翻整本书”；线性注意力像“先把全书做成索引”，之后每个问题只查索引。索引不是原书本身，所以可能有近似误差，但代价低很多。

| 方案 | 是否显式构造 $n \times n$ 注意力矩阵 | 典型时间复杂度 | 典型显存复杂度 | 长序列适应性 |
|---|---:|---:|---:|---|
| 标准 Softmax Attention | 是 | $O(n^2 d)$ | $O(n^2)$ | 弱 |
| Linear Attention | 否 | $O(n d^2)$ 或 $O(n r d_v)$ | $O(d^2)$ 或 $O(r d_v)$ | 强 |

其中 $r$ 是特征映射后的维度。很多论文里更精确地写成 $O(nrd_v)$，但在常见设定里若 $r$ 与 $d$ 同阶，常可粗略记成 $O(nd^2)$。

---

## 问题定义与边界

问题定义不是“把所有注意力都换成线性版本”，而是：在序列很长时，怎样保留全局信息交互，同时避免平方级成本。

这里有三个边界必须先说清楚。

第一，Linear Attention 不是对 softmax 的恒等变换，而通常是近似。也就是说，除非特定构造严格成立，否则它得到的是“接近标准注意力”的结果，不是完全一样的结果。

第二，特征映射 $\phi$ 必须满足可分解要求。术语“可分解”可以先理解成：原本依赖成对比较的核函数，能拆成两个向量映射后的内积。常见写法是：

$$
\kappa(q,k)\approx \phi(q)^\top \phi(k), \quad \phi:\mathbb{R}^d\to\mathbb{R}^r
$$

第三，很多实现要求 $\phi(x)\ge 0$。白话说，映射后的特征最好非负，这样归一化项更稳定，也更接近 softmax 这种本身非负的权重机制。

新手可以先记住一个边界判断：如果序列只有几十到几百个 token，标准注意力往往更简单也更准；如果序列是几千到几万个 token，平方复杂度开始变成瓶颈，线性注意力才真正有工程价值。

| 边界条件 | 为什么需要它 | 不满足时的问题 |
|---|---|---|
| 可分解：$\kappa(q,k)\approx \phi(q)^\top \phi(k)$ | 才能利用结合律先汇总再复用 | 无法线性化，只能回到两两计算 |
| 非负：$\phi(x)\ge 0$ | 归一化更稳定，避免分母抵消 | 分母可能接近 0，数值爆炸 |
| 可归一化 | 需要模仿 softmax 的加权平均 | 输出尺度漂移，不同位置不可比 |
| 特征维度 $r$ 受控 | 否则线性化后常数项过大 | 理论线性，实际仍然很慢 |

真实工程例子是长文档建模。假设输入长度从 2k 扩到 32k，标准注意力的相关矩阵大小会放大到原来的 $16^2=256$ 倍；线性注意力只会把主计算量近似放大 16 倍。这就是它存在的根本原因。

---

## 核心机制与推导

先看标准注意力的逐行形式。对第 $i$ 个查询 $Q_i$，输出是：

$$
O_i=\frac{\sum_{j=1}^n \exp(Q_iK_j^\top)V_j}{\sum_{j=1}^n \exp(Q_iK_j^\top)}
$$

难点在于，分子和分母都依赖每一对 $(i,j)$。这意味着每个查询都要重新扫描全部键。

线性注意力的想法是，用一个可分解核来近似 softmax 核：

$$
\exp(q^\top k)\approx \phi(q)^\top \phi(k)
$$

于是：

$$
O_i \approx \hat O_i
= \frac{\sum_{j=1}^n \phi(Q_i)^\top \phi(K_j)V_j}{\sum_{j=1}^n \phi(Q_i)^\top \phi(K_j)}
$$

因为 $\phi(Q_i)$ 与求和下标 $j$ 无关，可以提到求和号外：

$$
\hat O_i
= \frac{\phi(Q_i)^\top \Big(\sum_{j=1}^n \phi(K_j)V_j^\top\Big)}{\phi(Q_i)^\top \Big(\sum_{j=1}^n \phi(K_j)\Big)}
$$

定义两个中间量：

$$
S:=\sum_{j=1}^n \phi(K_j)V_j^\top = \phi(K)^\top V
$$

$$
Z:=\sum_{j=1}^n \phi(K_j)=\phi(K)^\top \mathbf{1}
$$

就得到：

$$
\hat O_i=\frac{\phi(Q_i)^\top S}{\phi(Q_i)^\top Z}
$$

这就是核心机制。术语“全局汇总”可以理解成：先把所有键和值压成一个共享统计量 $S$，再把所有键压成一个归一化统计量 $Z$。

下面用一个玩具例子算一遍。设一维情形：

- $Q=[1, 0.5]$
- $K=[0, 1]$
- $V=[2, 4]$
- $\phi(x)=e^x$

先算键映射：

- $\phi(K_1)=e^0=1$
- $\phi(K_2)=e^1=e$

于是：

$$
S = 1\cdot 2 + e\cdot 4 = 2+4e
$$

$$
Z = 1+e
$$

对第一个查询 $Q_1=1$：

$$
\hat O_1=\frac{e(2+4e)}{e(1+e)}=\frac{2+4e}{1+e}
$$

这里分子分母里的 $e$ 会约掉，本质上是“查询映射后分别去读同一份汇总结果”。

| 中间变量 | 含义 | 形状 |
|---|---|---|
| $\phi(Q)$ | 查询的特征映射 | $n \times r$ |
| $\phi(K)$ | 键的特征映射 | $n \times r$ |
| $S=\phi(K)^\top V$ | 键值汇总矩阵 | $r \times d_v$ |
| $Z=\phi(K)^\top \mathbf{1}$ | 归一化汇总向量 | $r \times 1$ |
| $\hat O_i$ | 第 $i$ 个位置输出 | $1 \times d_v$ |

这一步最容易误解的地方是：线性注意力不是“不要归一化”，而是“把归一化也写成可复用的汇总形式”。如果只写 $\phi(Q)(\phi(K)^\top V)$ 而忽略分母，输出尺度会失控。

---

## 代码实现

下面给出一个可运行的 Python 版本。它不是高性能实现，但足够展示机制。这里用 $\phi(x)=\mathrm{elu}(x)+1$ 的简化版，也就是 `exp`，因为它天然非负，便于说明。

```python
import numpy as np

def phi(x):
    # 非负特征映射；真实工程里会用更稳定或更高维的构造
    return np.exp(x)

def linear_attention(Q, K, V, eps=1e-9):
    """
    Q: (n, d)
    K: (n, d)
    V: (n, dv)
    return: (n, dv)
    """
    phi_Q = phi(Q)               # (n, d)
    phi_K = phi(K)               # (n, d)

    S = phi_K.T @ V              # (d, dv)
    Z = phi_K.sum(axis=0)        # (d,)

    out = []
    for q in phi_Q:
        numerator = q @ S        # (dv,)
        denominator = q @ Z      # scalar
        out.append(numerator / max(denominator, eps))
    return np.vstack(out)

def softmax_attention_1d(Q, K, V):
    scores = Q @ K.T
    scores = scores - scores.max(axis=1, keepdims=True)
    weights = np.exp(scores)
    weights = weights / weights.sum(axis=1, keepdims=True)
    return weights @ V

# 玩具例子
Q = np.array([[1.0], [0.5]])
K = np.array([[0.0], [1.0]])
V = np.array([[2.0], [4.0]])

lin_out = linear_attention(Q, K, V)
std_out = softmax_attention_1d(Q, K, V)

# 这一维玩具例子下，exp 特征映射与 softmax 结果一致
assert lin_out.shape == (2, 1)
assert std_out.shape == (2, 1)
assert np.allclose(lin_out, std_out, atol=1e-6)

print("linear:", lin_out.ravel())
print("softmax:", std_out.ravel())
```

这段代码体现了最关键的工程点：

1. 先算 `phi_K`。
2. 一次性汇总出 `S = phi_K.T @ V`。
3. 一次性汇总出 `Z = sum(phi_K)`。
4. 每个查询只做一次小规模点积，不构造 $n \times n$ 矩阵。

如果写成伪代码，就是：

```python
phi_K = phi(K)
S = phi_K.T @ V
Z = phi_K.T @ ones(n)

outputs = []
for q in Q:
    qf = phi(q)
    outputs.append((qf @ S) / max(qf @ Z, eps))
```

真实工程例子则更接近 Performer。Performer 里的 FAVOR+ 不是直接用 `exp(x)` 逐元素映射，而是用随机特征近似 softmax kernel。术语“随机特征”可以先理解成：用一组随机投影，把难算的核函数变成容易算的向量内积。

一个高度简化的示意代码如下，只展示思路，不等同于论文完整实现：

```python
import numpy as np

def orthogonal_random_matrix(m, d, seed=0):
    rng = np.random.default_rng(seed)
    W = rng.normal(size=(m, d))
    q, _ = np.linalg.qr(W.T)
    return q.T[:m]

def favor_feature_map(X, W):
    # X: (n, d), W: (r, d)
    # 这里省略论文中的完整缩放与稳定化细节，只保留结构
    norm_term = np.sum(X * X, axis=1, keepdims=True) / 2.0
    proj = X @ W.T
    return np.exp(proj - norm_term)

X = np.array([[0.2, -0.1], [0.4, 0.3]])
W = orthogonal_random_matrix(m=4, d=2, seed=42)
features = favor_feature_map(X, W)

assert features.shape == (2, 4)
assert np.all(features > 0)
```

这类实现的重点不是“把代码写短”，而是控制近似误差和数值稳定性。

---

## 工程权衡与常见坑

Linear Attention 的主要收益来自复杂度下降，但代价是表达能力和稳定性不再自动成立。

第一类坑是 $\phi$ 太弱。比如 `ReLU(x)+1` 虽然简单且非负，但它对相似度结构的表达很粗糙。白话说，它可能只能区分“差不多相关”和“不太相关”，却很难像 softmax 那样拉开细粒度差异。结果就是模型能跑，但效果不理想，特别是在需要精确选择上下文的位置上。

第二类坑是分母过小。公式里有：

$$
\phi(Q_i)^\top Z
$$

如果这个值接近 0，输出就会变得极大，训练直接不稳定。所以工程上通常会做 `eps` 截断、log-space 稳定化、归一化缩放，或者采用 FAVOR+ 这类专门为 softmax 设计的近似。

第三类坑是特征维度 $r$ 的选择。$r$ 太小，近似误差大；$r$ 太大，虽然名义上仍是线性，但常数项上升明显，吞掉本该节省的性能收益。

第四类坑是误以为线性注意力总比标准注意力好。短序列任务里，标准 softmax 的常数项更小、实现更成熟、硬件优化更充分，实际可能更快也更准。

| $\phi$ 设计 | 优点 | 风险 |
|---|---|---|
| `ReLU(x)+1` | 实现简单，非负 | 表达能力弱，容易低秩化 |
| `exp(x)` 或指数型映射 | 更贴近 softmax 核 | 容易数值溢出或下溢 |
| FAVOR+ 随机正交特征 | 理论更完整，近似 softmax 更合理 | 实现复杂，需控制方差与归一化 |

再给一个新手容易犯错的类比：如果 $\phi$ 只会输出非常粗糙的二值特征，就像把所有颜色都压成“亮”和“暗”两类。这样虽然存储方便，但很多本该区分开的模式会被混到一起。

真实工程里，长文档生成、高分辨率图像 token 序列、语音长上下文建模，是线性注意力更常见的场景；而分类、小模型推理、短上下文对话，往往不值得为了线性化付出额外近似成本。

---

## 替代方案与适用边界

Linear Attention 只是“降低注意力成本”的一条路线，不是唯一路线。

Performer 的核心是用 FAVOR+ 近似 softmax 核，适合需要保留全局交互、同时又必须扩展到长序列的场景。Linformer 的核心是低秩近似，术语“低秩”可以理解成：假设真正有效的信息只占原矩阵中的一小部分方向，所以可以先压缩再计算。Nyströmformer 则借用 Nyström 方法，用部分代表点近似完整注意力矩阵。

| 方案 | 核心思想 | 适合场景 | 优势 | 局限 |
|---|---|---|---|---|
| 标准 Softmax | 精确两两交互 | 短到中等序列 | 精度高，实现成熟 | $O(n^2)$ 成本高 |
| Performer | 随机特征近似 softmax | 长文档、长生成任务 | 线性复杂度，保留全局性 | 存在近似误差，调参更复杂 |
| Linformer | 低秩投影压缩 | 注意力矩阵可压缩的任务 | 结构简单，开销低 | 压缩假设不成立时掉点明显 |
| Nyströmformer | 代表点近似 | 中长序列、结构较平滑任务 | 近似质量可控 | 代表点选择影响很大 |

适用边界可以一句话概括：

- 如果任务上下文短，标准 softmax 往往是默认选择。
- 如果任务必须处理超长上下文，Linear Attention 或其他近似方法才开始变得必要。
- 如果数据呈现明显局部性，局部窗口或稀疏注意力有时比线性注意力更合适。
- 如果任务强依赖精确检索少数关键位置，线性近似可能比标准注意力更容易掉性能。

新手可以先用一个经验规则判断：128 到 512 token 的任务，先别急着上线性注意力；4k、8k、16k 以上并且显存真的吃紧，再认真评估 Performer 一类方案。

---

## 参考资料

| 资源 | 作者/站点 | 覆盖重点 | 读前期待 |
|---|---|---|---|
| *Rethinking Attention with Performers* | Choromanski 等 | FAVOR+ 理论、无偏近似、复杂度分析 | 先确认论文里的核近似和稳定化公式 |
| *Linear Attention: Breaking the Quadratic Bottleneck with Kernel Feature Maps* | mbrenndoerfer 博客 | 面向工程实现的直观解释 | 先建立“先汇总再复用”的计算图直觉 |
| EmergentMind 线性注意力综述 | EmergentMind | 变体整理、公式统一、应用场景 | 先横向比较不同线性化路线 |
| 官方或社区实现仓库 | GitHub | 训练细节、数值稳定技巧、benchmark | 对照论文检查实际实现差异 |

建议阅读顺序：

1. 先看综述，建立“为什么需要线性注意力”的问题意识。
2. 再看 Performer 论文，重点读 $\phi$、FAVOR+、归一化与复杂度证明。
3. 最后看博客和代码实现，把公式和工程细节对上。

延伸阅读时，优先关注三件事：公式是否含归一化项、随机特征如何做数值稳定、benchmark 是否真正覆盖长序列而不是只在短序列上比较。
