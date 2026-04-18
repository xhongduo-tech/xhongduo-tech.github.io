## 核心结论

Hessian 近似 = 用更便宜的曲率信息替代精确 Hessian 或其逆。

在无约束优化里，目标通常是最小化一个函数 $f(w)$。一阶方法只使用梯度 $\nabla f(w)$，它告诉我们当前位置往哪个方向下降最快。二阶方法还使用 Hessian 矩阵 $\nabla^2 f(w)$，它描述梯度如何变化，也就是目标函数在不同参数方向上的弯曲程度。

对零基础读者可以这样理解：梯度像“只告诉你往下走”，Hessian 像“告诉你哪里陡、哪里平、参数之间是否互相影响”。如果参数很多，完整地图太贵，就改看局部地图或缩略图。

核心结论有三点：

1. Hessian 近似的目标不是精确还原曲率，而是在可接受的计算和内存成本下，给出比纯梯度更好的下降方向。
2. 真正的取舍轴是“精度 vs. 代价”。完整 Hessian 信息最全，但存储和求解最贵；BFGS、L-BFGS、对角近似、K-FAC 都是在降低成本时保留一部分曲率信息。
3. 二阶近似能否有效，取决于梯度是否稳定、目标是否光滑、batch 是否足够大，以及近似结构是否匹配模型结构。

| 方法 | 保存什么 | 优点 | 缺点 |
|---|---|---|---|
| 完整 Hessian | 全矩阵 | 信息最全 | 太贵 |
| BFGS | 低秩更新的近似矩阵 | 收敛快 | 需要线搜索/稳定条件 |
| L-BFGS | 最近 m 组历史 | 省内存 | 历史会过时 |
| 对角近似 | 每个参数单独缩放 | 最便宜 | 忽略耦合 |
| K-FAC | 分层 Fisher 近似 | 深度网络更强 | 实现复杂 |

---

## 问题定义与边界

本文讨论的是无约束优化中的二阶方法，核心问题是如何近似 $\nabla^2 f(w)$ 或 $(\nabla^2 f(w))^{-1}$。

无约束优化指不额外限制参数范围的优化问题，例如直接最小化损失函数：

$$
\min_w f(w)
$$

其中 $w$ 是参数向量，$f(w)$ 是目标函数。二阶方法通常希望构造更新方向：

$$
p_k = -H_k g_k
$$

这里 $g_k$ 是第 $k$ 步的梯度，$H_k$ 可以是逆 Hessian 的近似。如果 $H_k$ 接近 $(\nabla^2 f(w_k))^{-1}$，更新方向就会根据不同方向的曲率自动缩放。

变量定义如下：

| 符号 | 含义 |
|---|---|
| `f(w)` | 目标函数 |
| `g_k` | 第 k 步梯度 `∇f(w_k)` |
| `H_k` | Hessian 或其逆近似 |
| `s_k` | 参数位移 `w_{k+1} - w_k` |
| `y_k` | 梯度差 `g_{k+1} - g_k` |

新手版解释：如果目标函数像一张山地地图，梯度只看当前斜坡，Hessian 还看地形弯曲程度。二阶方法适合“山形稳定”的场景，不适合“地形一直抖”的场景。

边界条件很重要：

| 场景 | 是否适合 Hessian 近似 | 原因 |
|---|---|---|
| 光滑目标 | 适合 | 曲率信息稳定 |
| 全量或大 batch | 适合 | 梯度差更接近真实曲率 |
| 希望更少步数收敛 | 适合 | 二阶方向通常质量更高 |
| 强噪声小 batch | 不适合 | 梯度差可能主要来自采样噪声 |
| 非光滑目标 | 不适合 | Hessian 可能不存在或不稳定 |
| 分布频繁变化 | 不适合 | 历史曲率很快过时 |

完整 Hessian 的代价是主要限制。若参数维度是 $n$，Hessian 是 $n \times n$ 矩阵，显式存储需要 $O(n^2)$，求逆或矩阵分解通常需要 $O(n^3)$。当模型有百万级参数时，完整 Hessian 在常规训练里基本不可行。

---

## 核心机制与推导

BFGS 的核心不是凭空猜曲率，而是使用割线条件。割线条件是一种局部一致性要求：参数变化 $s_k$ 和梯度变化 $y_k$ 应该被近似曲率矩阵解释。

对于逆 Hessian 近似，条件写作：

$$
H_{k+1} y_k \approx s_k
$$

其中：

$$
s_k = w_{k+1} - w_k,\quad y_k = g_{k+1} - g_k
$$

这表示：如果梯度变化了 $y_k$，逆 Hessian 近似应该能把它映射回对应的参数位移 $s_k$。

BFGS 逆 Hessian 更新公式是：

$$
H_{k+1}=(I-\rho_k s_k y_k^\top)H_k(I-\rho_k y_k s_k^\top)+\rho_k s_k s_k^\top
$$

其中：

$$
\rho_k=\frac{1}{y_k^\top s_k}
$$

对应的 Hessian 近似形式是：

$$
B_{k+1}=B_k-\frac{B_k s_k s_k^\top B_k}{s_k^\top B_k s_k}+\frac{y_k y_k^\top}{y_k^\top s_k}
$$

这里的 $B_k$ 近似 Hessian，$H_k$ 近似逆 Hessian。BFGS 是秩二更新，意思是每一步只用两个低秩修正项更新矩阵，而不是重新计算完整 Hessian。

关键条件是：

$$
y_k^\top s_k > 0
$$

这是保持正定性的核心条件。正定性指矩阵在所有非零方向上都给出正的曲率，这样更新方向更可能是下降方向。如果 $y_k^\top s_k \le 0$，说明这一步看到的曲率不可靠，BFGS 更新可能让方向变得不稳定。

玩具例子：给定

$$
H_0 = I,\quad s=[1,2]^\top,\quad y=[3,4]^\top
$$

先计算：

$$
y^\top s = 3\cdot 1 + 4\cdot 2 = 11,\quad \rho=\frac{1}{11}
$$

代入 BFGS 逆更新后，可以得到新的逆 Hessian 近似：

$$
H_1 \approx
\begin{bmatrix}
0.7521 & -0.3140 \\
-0.3140 & 0.7355
\end{bmatrix}
$$

对角线表示每个方向应该如何缩放步长。非对角线 $-0.3140$ 表示两个参数方向之间存在耦合：更新第一个参数时，第二个参数的曲率信息也会影响方向。

BFGS 和 L-BFGS 的区别如下：

| 项目 | BFGS | L-BFGS |
|---|---|---|
| 存储 | 完整矩阵 | 最近 m 对 `(s_i, y_i)` |
| 计算方式 | 显式更新 | two-loop recursion |
| 内存 | `O(n^2)` | `O(mn)` |
| 适用 | 中小规模 | 大规模 |

L-BFGS 的“有限内存”不是弱化版实现细节，而是核心设计。它不保存完整 $H_k$，只保存最近 $m$ 组历史 $(s_i, y_i)$，用 two-loop recursion 直接计算 $H_k g_k$。当 $m$ 远小于 $n$ 时，内存从 $O(n^2)$ 降到 $O(mn)$。

对角 Hessian 近似更便宜。它只保留矩阵对角线，例如：

$$
D_k = \mathrm{diag}(B_k)
$$

它的含义是每个参数单独缩放，不考虑参数之间的相互影响。优点是实现简单、内存低；缺点是在强耦合问题上会丢掉重要方向。

自然梯度是另一条路线。自然梯度指用 Fisher 信息矩阵重新定义参数空间中的距离，而不是直接使用欧氏空间的普通梯度。它的典型更新是：

$$
\Delta\theta = -\eta (F+\lambda I)^{-1}\nabla L
$$

其中 $F$ 是 Fisher 信息矩阵，$\lambda I$ 是 damping。damping 是一种稳定项，作用是避免矩阵病态或近似误差导致步长过大。

K-FAC 是 Kronecker-Factored Approximate Curvature 的缩写，意思是用 Kronecker 积结构近似曲率矩阵。对神经网络某一层的权重 $W$，常写成：

$$
F_W \approx A \otimes G
$$

其中 $A$ 来自输入激活 $a$ 的统计量，例如 $A=\mathbb E[aa^\top]$；$G$ 来自误差信号 $\delta$ 的统计量，例如 $G=\mathbb E[\delta\delta^\top]$。它不等于精确 Hessian，而是在概率模型和神经网络层结构下，用 Fisher 矩阵的可分解近似降低计算成本。

---

## 代码实现

代码部分重点不是实现所有二阶算法，而是看清它们如何接入优化流程。下面先用一个可运行的 Python 例子验证 BFGS 逆更新，再给出 PyTorch `LBFGS` 的工程用法。

```python
import numpy as np

def bfgs_inverse_update(H, s, y):
    s = s.reshape(-1, 1)
    y = y.reshape(-1, 1)
    ys = float(y.T @ s)
    assert ys > 0, "BFGS requires y^T s > 0"

    rho = 1.0 / ys
    I = np.eye(H.shape[0])
    return (I - rho * s @ y.T) @ H @ (I - rho * y @ s.T) + rho * s @ s.T

H0 = np.eye(2)
s = np.array([1.0, 2.0])
y = np.array([3.0, 4.0])

H1 = bfgs_inverse_update(H0, s, y)

assert np.allclose(H1 @ y, s)
assert np.allclose(H1, H1.T)
assert np.allclose(H1, np.array([[0.75206612, -0.31404959],
                                  [-0.31404959, 0.73553719]]))

print(H1)
```

这个例子对应前面的玩具推导。`assert np.allclose(H1 @ y, s)` 验证了割线条件；`assert np.allclose(H1, H1.T)` 验证更新后仍然对称。

PyTorch 中的 L-BFGS 用法如下：

```python
import torch

model = torch.nn.Linear(10, 1)
criterion = torch.nn.MSELoss()

x = torch.randn(128, 10)
y = torch.randn(128, 1)

optimizer = torch.optim.LBFGS(
    model.parameters(),
    lr=1.0,
    history_size=10,
    line_search_fn="strong_wolfe"
)

def closure():
    # LBFGS 内部可能多次重新计算同一个 batch 的 loss 和 gradient。
    optimizer.zero_grad()
    loss = criterion(model(x), y)
    loss.backward()
    return loss

loss = optimizer.step(closure)
```

`closure` 是一个重新计算损失和梯度的函数。L-BFGS 不是“更高级的 Adam”，它内部可能为了线搜索多次评估同一个 batch，所以需要 `closure` 提供可重复计算的损失。若每次 closure 取到不同数据，历史曲率就会混入采样噪声。

`strong_wolfe` 是一种线搜索条件，用来控制步长，同时帮助满足 $y_k^\top s_k > 0$。这对 BFGS/L-BFGS 的稳定性很关键。

two-loop recursion 的伪代码如下：

```text
输入：梯度 g，历史对 (s_i, y_i)
输出：方向 p = -H_k g

1. 从最近历史开始递推 alpha_i
2. 用初始缩放 H_0 处理中间向量
3. 正序递推 beta_i
4. 得到最终方向 p
```

常见实现点如下：

| 实现点 | 说明 |
|---|---|
| `history_size` | 保存多少组历史对 |
| `line_search_fn` | 控制步长搜索 |
| damping | 防止曲率估计不稳定 |
| batch size | 太小会让历史对噪声过大 |

真实工程例子：训练一个中等规模分类模型时，可以先用 Adam 或 SGD 快速进入可用区域，再在最后阶段切换到 L-BFGS 精修。这个策略更适合全量数据或大 batch，因为 L-BFGS 依赖稳定的梯度差。如果每一步 batch 都很小，$y_k$ 可能主要反映采样噪声，而不是目标函数真实曲率。

---

## 工程权衡与常见坑

二阶近似的主要风险不是“算不出来”，而是“算得出来但方向不稳定”。尤其在噪声梯度下，历史曲率可能变成错误经验。

最关键的条件仍然是：

$$
y_k^\top s_k > 0
$$

如果这个条件被破坏，BFGS 可能失去正定性，L-BFGS 的方向也可能不再可靠。新手版解释：如果你用很小的 batch 训练，今天看到的梯度差可能只是噪声，不是真正的曲率变化；这会让 L-BFGS 记住“假经验”。

常见坑如下：

| 坑 | 表现 | 规避方法 |
|---|---|---|
| `y^T s <= 0` | 方向不稳定、矩阵失去正定性 | Wolfe/strong Wolfe 线搜索、damped BFGS |
| 小 batch 噪声大 | 历史对过时 | 增大 batch、减小 `m` |
| 对角近似太弱 | 学不动耦合方向 | 换 L-BFGS 或 K-FAC |
| K-FAC 统计不稳 | 更新抖动 | damping、滑动平均 |

工程选择可以按场景判断：

| 场景 | 推荐方法 |
|---|---|
| 中小模型、平滑目标 | BFGS |
| 大模型、内存敏感 | L-BFGS |
| 超大模型、层结构明确 | K-FAC |
| 只想轻量加速 | 对角预条件 |

这里需要区分“步数少”和“总时间少”。二阶近似通常能减少迭代步数，但每一步更贵。若模型很小、目标光滑，BFGS 可能很划算；若模型很大、数据噪声强，Adam 或 SGD 反而更稳。

另一个常见误解是把 L-BFGS 当成通用深度学习默认优化器。实际上，L-BFGS 对 batch 稳定性、线搜索、内存和 closure 都有要求。在大规模随机训练中，它往往不如 Adam 方便。它更适合全量优化、小到中等规模模型、传统机器学习模型、物理反演、风格迁移、后期精修等场景。

K-FAC 的坑在于统计量本身也要估计。$A$ 和 $G$ 通常来自 batch 统计，如果 batch 太小或分布变化太快，Kronecker 近似会抖动。工程上通常需要 damping、滑动平均、较低频率更新 Fisher 因子，并控制矩阵求逆的数值稳定性。

---

## 替代方案与适用边界

不同近似方法不是“谁更先进”，而是“谁更适合当前约束”。

新手版解释：如果你连完整地图都买不起，就看缩略图；如果你只关心主干道，就用层级地图；如果你想走最快路线，就需要能反映道路弯曲的局部导航。对应到优化里，对角近似是最便宜的缩略图，L-BFGS 是用最近轨迹拼出的局部地图，K-FAC 是按神经网络层结构压缩后的曲率地图。

| 方法 | 本质 | 优势 | 局限 |
|---|---|---|---|
| 对角近似 | 逐参数缩放 | 最便宜 | 忽略耦合 |
| BFGS | 历史驱动的密集近似 | 方向质量高 | 内存和步长控制要求高 |
| L-BFGS | 限长历史近似 | 大规模可用 | 对噪声敏感 |
| 自然梯度 | 参数空间度量重构 | 概率模型更合理 | 依赖 Fisher 解释 |
| K-FAC | 分层 Kronecker 近似 | 深度网络中很实用 | 假设较强 |

对角近似适合资源极紧张、只想获得轻量加速的情况。很多自适应优化器也可以从“逐参数缩放”的角度理解，但它们通常使用梯度一阶或二阶矩统计，不等于精确 Hessian 对角线。

BFGS 适合中小规模、光滑、梯度稳定的问题。它能构造较高质量的密集曲率近似，但内存和线搜索要求更高。

L-BFGS 适合参数较多但仍希望使用曲率历史的问题。它牺牲完整矩阵，换取 $O(mn)$ 内存。它的边界是历史信息可能过时，尤其在随机小 batch 训练中。

自然梯度更适合负对数似然和概率模型。Fisher 信息矩阵衡量的是概率分布对参数变化的敏感程度，因此它和普通 Hessian 的出发点不同。若目标函数没有清晰的概率解释，自然梯度的理论解释会弱一些。

K-FAC 更适合结构化神经网络层，例如全连接层和卷积层。它利用层内激活和误差信号的统计结构，把大矩阵近似成 Kronecker 积，从而降低求逆成本。但它依赖较强的近似假设，实现也比 L-BFGS 和对角预条件复杂。

---

## 参考资料

| 类型 | 来源 | 对应内容 |
|---|---|---|
| 教材 | Nocedal & Wright, *Numerical Optimization* | BFGS、L-BFGS、复杂度 |
| 论文 | Liu & Nocedal (1989), *On the limited memory BFGS method for large scale optimization* | L-BFGS 原理 |
| 论文 | Amari (1998), *Natural Gradient Works Efficiently in Learning* | 自然梯度与 Fisher |
| 论文 | Martens & Grosse (2015), *Optimizing Neural Networks with Kronecker-factored Approximate Curvature* | K-FAC |
| 文档 | PyTorch `LBFGS` 文档 | 代码实现与工程限制 |

教材负责定义和推导，论文负责方法原始动机，文档负责落地实现细节。如果只想先学公式，先看 Nocedal & Wright；如果想知道 L-BFGS 的来源，看 Liu & Nocedal；如果想直接在 PyTorch 里使用，看 PyTorch 官方文档。

- Nocedal, J., & Wright, S. J. *Numerical Optimization*, 2nd ed. https://convexoptimization.com/TOOLS/nocedal.pdf
- Liu, D. C., & Nocedal, J. (1989). *On the limited memory BFGS method for large scale optimization*. https://link.springer.com/article/10.1007/BF01589116
- Amari, S. (1998). *Natural Gradient Works Efficiently in Learning*. https://direct.mit.edu/neco/article/10/2/251/6143/Natural-Gradient-Works-Efficiently-in-Learning
- Martens, J., & Grosse, R. (2015). *Optimizing Neural Networks with Kronecker-factored Approximate Curvature*. https://proceedings.mlr.press/v37/martens15.html
- PyTorch `torch.optim.LBFGS` documentation. https://docs.pytorch.org/docs/stable/generated/torch.optim.LBFGS.html
