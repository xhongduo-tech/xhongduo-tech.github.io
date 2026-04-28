## 核心结论

Shampoo 是一种**结构化二阶优化器**。这里的“二阶”可以先理解为：它不仅看梯度有多大，还试图利用梯度在不同方向上的相关性，近似感知参数空间的几何形状。它的核心不是“给每个参数单独调学习率”，而是**对参数张量的每个维度分别维护二阶统计矩阵，再用这些矩阵的逆根去预条件梯度**。

如果把一个线性层权重看成矩阵 $W \in \mathbb{R}^{m \times n}$，Adam 更像逐元素看每个位置自己的历史波动；Shampoo 则会分别看“行方向”和“列方向”的梯度相关性。这样做的效果，近似于沿不同轴把梯度做一次结构化缩放，让更新方向更接近问题本身的几何结构。

矩阵形式下，Shampoo 的核心更新为：

$$
L_t = L_{t-1} + G_t G_t^\top,\quad
R_t = R_{t-1} + G_t^\top G_t
$$

$$
W_{t+1} = W_t - \eta \, L_t^{-1/4} G_t R_t^{-1/4}
$$

其中 $G_t=\nabla f_t(W_t)$，$\eta$ 是学习率，$L_t,R_t$ 是左右两个预条件矩阵。“预条件”可以先理解为：先把梯度按统计结构重新缩放，再更新参数。

| 方法 | 统计对象 | 是否利用参数结构 | 典型代价 |
|---|---|---:|---:|
| SGD | 无二阶统计 | 否 | 低 |
| Adam | 逐元素二阶矩 | 否，基本不看矩阵轴结构 | 低到中 |
| Shampoo | 每个维度一个二阶矩阵 | 是，按轴利用结构 | 中到高 |

对零基础读者，先记住一句话就够了：**Shampoo 比 Adam 更懂“矩阵和张量长什么样”，代价是更贵的内存和分解计算。**

---

## 问题定义与边界

Shampoo 要解决的问题是：很多深度网络参数天然是矩阵或张量，但常见自适应优化器通常把它们当成一堆独立标量处理，结果会丢掉结构信息。

例如一个线性层权重 $W \in \mathbb{R}^{m \times n}$，如果直接摊平成长度为 $mn$ 的向量再做逐元素更新，那么“输出维度”和“输入维度”在优化里没有被区分。可实际中，这两个方向往往承担不同角色。Shampoo 的想法是：**不要把结构压平，而是分别统计每个维度上的梯度关系。**

对矩阵来说，可以把它理解成两条轴：

| 对象 | 含义 | Shampoo 关注什么 |
|---|---|---|
| 第 0 维 | 行，常对应输出通道/输出特征 | 行与行之间的梯度相关性 |
| 第 1 维 | 列，常对应输入通道/输入特征 | 列与列之间的梯度相关性 |

对更高阶张量，比如卷积核 $W \in \mathbb{R}^{C_{\text{out}} \times C_{\text{in}} \times k_h \times k_w}$，Shampoo 不是维护一个巨大四阶对象，而是对每个 mode 单独建一个二阶矩阵。这里的 **mode** 可以先理解为“张量的一条轴”。

一个简化示意是：

$$
W \in \mathbb{R}^{d_1 \times d_2 \times \cdots \times d_k}
\Rightarrow
\text{为每个维度 } i \text{ 维护 } H^{(i)} \in \mathbb{R}^{d_i \times d_i}
$$

这一定义同时给出了它的边界。Shampoo 不是完整 Hessian，也不是完整 Fisher 的精确近似。它保留的是“按维度的结构化二阶信息”，而不是全参数之间的所有耦合关系。所以它比纯对角方法更强，但比全矩阵二阶方法便宜得多。

| 场景 | 结论 | 原因 |
|---|---|---|
| 稠密线性层、投影层很多 | 适合 | 参数天然是 dense 矩阵，结构信息明显 |
| CNN/Transformer 主体为 dense 张量 | 适合 | 可按 mode 做结构化预条件 |
| 超大稀疏 embedding | 不适合 | 稀疏更新与大维度矩阵维护成本不匹配 |
| 显存非常紧张 | 需要谨慎 | 额外统计矩阵和分解缓存占内存 |
| 极端追求吞吐量 | 需要谨慎 | 分解步骤会拖慢训练 |

所以，Shampoo 的问题边界很明确：**它主要服务于 dense 参数主导、愿意为更好几何预条件付额外成本的训练任务。**

---

## 核心机制与推导

先看矩阵情况。设某一步的梯度是 $G_t \in \mathbb{R}^{m \times n}$。Shampoo 分别维护左右两个二阶统计矩阵：

$$
L_t = L_{t-1} + G_t G_t^\top,\quad
R_t = R_{t-1} + G_t^\top G_t
$$

其中：

- $L_t \in \mathbb{R}^{m \times m}$：刻画行方向的相关性。
- $R_t \in \mathbb{R}^{n \times n}$：刻画列方向的相关性。

为什么更新式里是 $-1/4$ 次幂，而不是直接逆？原因是矩阵梯度同时要被左边和右边两个统计量缩放。如果希望整体效果接近一个对称的二阶预条件，那么左右各承担一半，因此使用逆四分之一次幂：

$$
\tilde G_t = L_t^{-1/4} G_t R_t^{-1/4}
$$

再用 $\tilde G_t$ 更新参数：

$$
W_{t+1} = W_t - \eta \tilde G_t
$$

这个公式背后的直观意义是：如果某一行方向或某一列方向的历史梯度波动很大，对应方向就要被压小；如果某个方向历史上比较“安静”，就允许更大的有效更新。这和白化有点像，但这里是按参数轴的结构化近似，而不是对整个参数向量做完整白化。

### 玩具例子：2×2 最小数值推导

取

$$
G=
\begin{bmatrix}
1 & 0\\
0 & 2
\end{bmatrix},\quad
\epsilon=1,\quad
W_1=0
$$

则

$$
L = I + GG^\top
=
\begin{bmatrix}
2 & 0\\
0 & 5
\end{bmatrix},\quad
R = I + G^\top G
=
\begin{bmatrix}
2 & 0\\
0 & 5
\end{bmatrix}
$$

因此

$$
L^{-1/4}=R^{-1/4}=
\begin{bmatrix}
2^{-1/4} & 0\\
0 & 5^{-1/4}
\end{bmatrix}
$$

预条件后的梯度为

$$
\tilde G = L^{-1/4} G R^{-1/4}
=
\begin{bmatrix}
2^{-1/2} & 0\\
0 & 2/\sqrt{5}
\end{bmatrix}
\approx
\begin{bmatrix}
0.7071 & 0\\
0 & 0.8944
\end{bmatrix}
$$

可以看到，原始梯度对角线是 $(1,2)$，第二个方向大了 2 倍；预条件后变成大约 $(0.707,0.894)$，差距被压缩了。这就是结构化预条件的核心效果：**大方向被抑制，小方向被相对保留。**

### 张量版机制

若参数是 $k$ 阶张量 $W \in \mathbb{R}^{d_1 \times \cdots \times d_k}$，Shampoo 对每个 mode $i$ 维护：

$$
H_t^{(i)} = H_{t-1}^{(i)} + \mathrm{mat}_i(G_t)\mathrm{mat}_i(G_t)^\top
$$

这里 $\mathrm{mat}_i(G_t)$ 表示沿第 $i$ 个 mode 展开后的矩阵。然后沿每个 mode 乘上逆根，形式上可以写成：

$$
\tilde G_t
=
G_t
\times_1 (H_t^{(1)})^{-1/(2k)}
\times_2 (H_t^{(2)})^{-1/(2k)}
\cdots
\times_k (H_t^{(k)})^{-1/(2k)}
$$

其中 $\times_i$ 表示张量在第 $i$ 个 mode 上与矩阵相乘。这里每个维度分担一部分缩放，因此指数变成 $-1/(2k)$。

### 真实工程例子

在 Transformer 里，注意力投影层和 MLP 线性层都是典型 dense 矩阵参数。比如一个投影层 $W_q \in \mathbb{R}^{d_{\text{model}} \times d_{\text{head}}}$，如果训练中某些输入方向总是产生高相关梯度，Shampoo 会在列方向的统计矩阵里积累出来；如果某些输出方向更新长期过猛，它也会在行方向上压制。相比纯逐元素方法，这种按轴处理更接近层级结构本身，所以在大规模语言模型、语音模型、推荐系统的稠密部分更可能体现优势。

---

## 代码实现

实现 Shampoo 的难点不在公式，而在三个工程问题：

1. 如何累计统计矩阵。
2. 如何高效求逆根。
3. 如何避免每步分解导致训练过慢。

一个典型流程是：`accumulate -> factorize -> precondition -> update`。也就是先累积梯度统计，到固定步数再做一次特征分解或 SVD，把逆根缓存起来，后续若干步复用缓存。

下面是一个可运行的简化 Python 例子，只展示矩阵版核心机制，不依赖深度学习框架：

```python
import numpy as np

def matrix_power_sym(A, power, eps=1e-12):
    # A 必须是对称半正定矩阵
    eigvals, eigvecs = np.linalg.eigh(A)
    eigvals = np.clip(eigvals, eps, None)
    D = np.diag(eigvals ** power)
    return eigvecs @ D @ eigvecs.T

def shampoo_step(W, G, lr=0.1, eps=1.0):
    m, n = G.shape
    L = eps * np.eye(m) + G @ G.T
    R = eps * np.eye(n) + G.T @ G
    L_inv_quarter = matrix_power_sym(L, -0.25)
    R_inv_quarter = matrix_power_sym(R, -0.25)
    G_tilde = L_inv_quarter @ G @ R_inv_quarter
    W_next = W - lr * G_tilde
    return W_next, G_tilde, L, R

W = np.zeros((2, 2))
G = np.array([[1.0, 0.0],
              [0.0, 2.0]])

W_next, G_tilde, L, R = shampoo_step(W, G, lr=0.1, eps=1.0)

assert np.allclose(L, np.diag([2.0, 5.0]))
assert np.allclose(R, np.diag([2.0, 5.0]))
assert np.allclose(G_tilde, np.diag([2 ** -0.5, 2 / np.sqrt(5)]), atol=1e-6)
assert W_next[0, 0] < 0 and W_next[1, 1] < 0
```

这个版本每步都重算分解，真实训练里通常不会这样做。更常见的是：

```python
for step in training_steps:
    G = grad(W)

    # 1. accumulate
    L += G @ G.T
    R += G.T @ G

    # 2. factorize
    if step % precondition_frequency == 0:
        L_inv_quarter = eigh(L + eps * I) ** (-1/4)
        R_inv_quarter = eigh(R + eps * I) ** (-1/4)

    # 3. precondition
    G_tilde = L_inv_quarter @ G @ R_inv_quarter

    # 4. update
    W -= lr * G_tilde
```

实际框架实现还会加入 grafting。这里的 **grafting** 可以先理解为：方向用 Shampoo，步长大小参考 SGD 或 Adam，这样更稳定。它常用于避免纯二阶预条件在训练早期过激或过慢。

| 实现点 | 常见做法 | 目的 |
|---|---|---|
| 累计矩阵 | 每步累积 $GG^\top, G^\top G$ | 保留结构统计 |
| 分解频率 | 每隔若干步更新一次 | 降低 eig/SVD 开销 |
| 阈值退化 | 维度过大时退化成对角近似 | 控制内存和时间 |
| 缓存策略 | 缓存逆根矩阵 | 避免重复分解 |
| 数值稳定项 `epsilon` | 初始化或分解前加到对角线 | 防止奇异和过小特征值 |

---

## 工程权衡与常见坑

Shampoo 的优势来自结构化二阶信息，代价也来自这里。它主要贵在两件事：**额外状态内存**和**矩阵分解计算**。

先看一个量级直觉。若某层权重是 $W \in \mathbb{R}^{m \times n}$，Adam 的二阶状态是一个同形状矩阵，规模约为 $O(mn)$；Shampoo 除了参数和梯度外，还要维护 $m \times m$ 和 $n \times n$ 两个统计矩阵，规模约为 $O(m^2+n^2)$。当某一侧维度很大时，这个代价会迅速上升。

| 常见坑 | 现象 | 原因 |
|---|---|---|
| 频繁分解导致慢 | 吞吐量明显下降 | eig/SVD 代价高 |
| 大维度导致内存爆 | 显存或内存占用异常 | 统计矩阵是平方级 |
| 稀疏参数不适配 | 收益低甚至更慢 | 稀疏结构不适合完整轴统计 |
| 数值不稳定 | loss 抖动、NaN | 逆根、学习率、`epsilon` 组合敏感 |

对应规避策略通常是：

| 问题 | 规避策略 |
|---|---|
| 分解太频繁 | 延迟更新，只在固定间隔做分解 |
| 维度太大 | 设阈值，超过后退化成对角或 block 近似 |
| 训练不稳 | grafting 到 Adam/SGD 的步长范数 |
| 特征值太小 | 增大 `epsilon`，必要时做特征值裁剪 |
| 学习率难调 | 从已知稳定的一阶优化器配置起步，再逐步替换 |

一个很常见的真实工程坑是 embedding 或特征表特别大。比如推荐系统里稀疏 ID embedding 可能有超大词表，但每步只更新极少行。这种参数上维护完整统计矩阵，既贵又不自然，通常不划算。Shampoo 更适合推荐系统中的 **dense MLP 塔**，而不是超大稀疏表本体。

另一个坑是“论文里能跑，不等于生产里直接能跑”。因为论文通常会配合特定硬件、分布式策略、分解频率和 grafting 技术。如果直接把朴素版本搬到训练脚本里，结果往往是收敛不错但吞吐量大跌，最后总训练时间反而不占优。

---

## 替代方案与适用边界

理解 Shampoo，最好把它放在优化器谱系里看。它处在“比 Adam 更强地利用结构，但比完整二阶方法更便宜”的位置。

| 方法 | 结构利用 | 状态/计算成本 | 典型适用边界 |
|---|---|---|---|
| SGD | 很弱 | 最低 | 强基线、配合大规模调参 |
| Adam | 逐元素 | 低到中 | 通用默认选择，稀疏也常用 |
| Adafactor | 行列分解式近似 | 更省内存 | 超大矩阵、显存敏感训练 |
| Shampoo | 按维度完整二阶矩阵 | 中到高 | dense 矩阵/张量主导，愿意付额外代价 |
| K-FAC | 层级 Fisher 近似 | 高 | 更强调层级二阶近似，通常更复杂 |

和 K-FAC 相比，Shampoo 不直接围绕某一层的 Fisher 因子化结构做近似，而是围绕参数张量本身的 mode 结构维护统计矩阵。它通常更通用，也更容易扩展到任意张量参数；同时其统计对象更局部，工程实现常比全套 K-FAC 更直接。

一个简化判断清单是：

1. 参数是否主要是 dense，而不是大规模稀疏表。
2. 模型是否以矩阵/张量参数为主，而不是零散小参数。
3. 训练预算是否允许额外状态和周期性分解。
4. 是否希望得到比 Adam 更强的几何预条件。

如果四条里前三条都满足，Shampoo 值得试。  
如果模型主要是稀疏 embedding，Adam 往往更简单更稳。  
如果显存特别紧而矩阵又很大，Adafactor 常是更现实的折中。  
如果你已经有非常成熟的 SGD/Adam 调参和分布式流水线，Shampoo 的收益要和系统复杂度一起评估，而不是只看单步收敛曲线。

结论可以落成一句工程判断：**当模型主体是 dense 线性层，训练成本允许，且你确实需要更强的结构化预条件时，Shampoo 才最有价值。**

---

## 参考资料

1. [Shampoo: Preconditioned Stochastic Tensor Optimization](https://proceedings.mlr.press/v80/gupta18a.html)  
用于理解原始算法定义、矩阵版与张量版公式，以及理论动机。

2. [Google Research: Shampoo: Preconditioned Stochastic Tensor Optimization](https://research.google/pubs/shampoo-preconditioned-stochastic-tensor-optimization/)  
用于快速查看研究背景和官方论文入口。

3. [PyTorch Distributed Shampoo README](https://github.com/facebookresearch/optimizers/blob/main/distributed_shampoo/README.md)  
用于理解工程实现中的分布式、缓存、预条件更新频率等落地细节。

4. [Learning Rate Grafting: Transferability of Optimizer Tuning](https://openreview.net/forum?id=FpKgG31Z_i9)  
用于理解为什么现代 Shampoo 实现常配 grafting 来提升稳定性和可迁移性。

5. [Adafactor: Adaptive Learning Rates with Sublinear Memory Cost](https://proceedings.mlr.press/v80/shazeer18a.html)  
用于和 Shampoo 做内存成本上的对比，理解行列分解式近似的另一条路线。
