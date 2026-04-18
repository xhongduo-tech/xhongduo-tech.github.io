## 核心结论

Wasserstein 距离是分布之间的距离：它衡量“把一个概率分布搬成另一个概率分布所需的最小运输代价”。

如果把概率分布看成带质量的位置集合，Wasserstein 距离关心的不是两个分布在同一个坐标点上的数值差，而是质量从哪里搬到哪里、搬多少、每单位质量要付多少代价。它显式使用底层空间的距离，所以比 KL、JS 这类散度更有几何含义。

一般形式是：

$$
W_p(\mu,\nu)=\left(\inf_{\pi\in\Pi(\mu,\nu)}
\int_{\mathcal X\times\mathcal X} d(x,y)^p\,d\pi(x,y)
\right)^{1/p}
$$

这里 $p$ 控制代价对距离的敏感程度。常用的是 $W_1$ 和 $W_2$。

| 距离 | 典型用途 | 直观含义 |
|---|---|---|
| $W_1$ | 鲁棒分布比较、WGAN | 搬运总路程最小 |
| $W_2$ | 几何结构、测地线、梯度流 | 长距离搬运惩罚更重 |
| KL / JS | 概率模型、分类式生成模型 | 更关心密度比或可区分性 |

玩具例子：有两堆质量分别在点 $0$ 和点 $2$，目标是全部搬到点 $1$。Wasserstein 问的是“怎么搬总代价最小”，不是“数组第几个元素和第几个元素相减”。

真实工程例子：WGAN 中，生成器产生的图像分布和真实图像分布通常支撑不完全重叠。JS 散度可能很快饱和，无法提供有效梯度；$W_1$ 仍然能反映“生成分布离真实分布还有多远”。

---

## 问题定义与边界

先定义对象，再谈距离。Wasserstein 距离不是任意两个数组之间的距离，而是两个概率分布之间的距离。

| 记号 | 含义 | 白话解释 |
|---|---|---|
| $\mathcal X$ | 底层空间 | 样本所在的空间，比如实数轴、图像特征空间 |
| $d(x,y)$ | 底层距离 | 从位置 $x$ 搬到位置 $y$ 的单位距离 |
| $\mu,\nu$ | 概率分布 | 两堆总质量都是 1 的质量分布 |
| $\mathcal P_p(\mathcal X)$ | 有有限 $p$ 阶矩的概率分布集合 | 距离的 $p$ 次方期望不能发散 |
| $\pi$ | 运输计划 | 指定从每个 $x$ 搬多少质量到每个 $y$ |
| $\Pi(\mu,\nu)$ | 所有合法运输计划 | 起点边缘分布是 $\mu$，终点边缘分布是 $\nu$ |
| $c(x,y)$ | 运输成本 | 常取 $c(x,y)=d(x,y)^p$ |

运输计划的关键约束是：从源分布搬出的总量必须等于 $\mu$，搬到目标位置的总量必须等于 $\nu$。这叫边缘约束，意思是联合分布 $\pi(x,y)$ 的两个边缘分别固定。

离散场景中，假设源分布有 $n$ 个点，目标分布有 $m$ 个点，运输计划就是一个 $n\times m$ 的非负矩阵 $P$。其中 $P_{ij}$ 表示从源点 $x_i$ 搬到目标点 $y_j$ 的质量。优化目标是：

$$
\min_P \sum_{i=1}^n\sum_{j=1}^m P_{ij}C_{ij}
$$

其中 $C_{ij}=d(x_i,y_j)^p$。

边界也要明确：$W_p$ 要求分布有有限 $p$ 阶矩。直观地说，如果分布尾部太重，远距离搬运的期望成本可能无限大，此时 $W_p$ 不再是有限数值。工程上，离散精确 OT 可以写成线性规划或最小费用流，但样本点一大就会明显变慢。

一维直方图例子：两个直方图权重相同，但一个集中在左侧，一个集中在右侧。逐点差可能只看到“哪些格子不一样”，Wasserstein 会继续计算“质量跨了多少格”。这正是它的几何信息来源。

---

## 核心机制与推导

最优传输的原问题是：在所有合法运输计划中，找总代价最低的那个。

以 $W_1$ 为例：

$$
W_1(\mu,\nu)=\inf_{\pi\in\Pi(\mu,\nu)}
\int_{\mathcal X\times\mathcal X} d(x,y)\,d\pi(x,y)
$$

玩具例子：

$$
\mu=\frac12\delta_0+\frac12\delta_2,\qquad \nu=\delta_1
$$

$\delta_a$ 表示全部质量集中在点 $a$ 的分布。源分布有一半质量在 $0$，一半质量在 $2$；目标分布全部质量在 $1$。合法运输只能把 $0.5$ 的质量从 $0$ 搬到 $1$，再把 $0.5$ 的质量从 $2$ 搬到 $1$：

$$
W_1(\mu,\nu)=0.5|0-1|+0.5|2-1|=1
$$

$W_1$ 的 Kantorovich 对偶是：

$$
W_1(\mu,\nu)=\sup_{\|f\|_{\mathrm{Lip}}\le 1}
\left(\int f\,d\mu-\int f\,d\nu\right)
$$

这里 $1$-Lipschitz 函数指变化速度不超过 1 的函数，也就是：

$$
|f(x)-f(y)|\le d(x,y)
$$

对上面的例子，取 $f(x)=|x-1|$，它是 $1$-Lipschitz 函数，并且：

$$
\int f\,d\mu-\int f\,d\nu
=\frac12(1+1)-0=1
$$

这和原问题结果一致。对偶不是换一种写法而已，它告诉我们：计算 $W_1$ 可以看成找一个受 Lipschitz 约束的评分函数，让它最大化两个分布的期望差。WGAN 的 critic 就是在近似这个函数。

$W_2$ 更强调几何结构。如果存在最优映射 $T$，从 $\mu_0$ 把质量搬到 $\mu_1$，则 Wasserstein 空间中的测地线可写成：

$$
\mu_t=\big((1-t)\mathrm{Id}+tT\big)_\#\mu_0,\quad t\in[0,1]
$$

测地线是空间中两点之间的最短路径。这里的“点”不是普通坐标点，而是概率分布。$\# $ 表示推前分布，即把原分布中的样本通过函数映射后得到的新分布。

$W_2$ 还有 Benamou-Brenier 动力学形式：

$$
W_2^2(\mu_0,\mu_1)=
\inf_{\rho_t,v_t}
\int_0^1\int \rho_t(x)\|v_t(x)\|^2\,dx\,dt
$$

约束为：

$$
\partial_t\rho_t+\nabla\cdot(\rho_t v_t)=0
$$

这里 $\rho_t$ 是时刻 $t$ 的密度，$v_t$ 是速度场。这个公式把“搬运”写成连续流动：质量不能凭空产生或消失，只能沿速度场移动。

| 形式 | 核心对象 | 适合理解什么 |
|---|---|---|
| 原问题 | 运输计划 $\pi$ | 质量如何从源分布搬到目标分布 |
| 对偶形式 | Lipschitz 函数 $f$ | WGAN critic 为什么有 Lipschitz 约束 |
| 动力学形式 | 密度 $\rho_t$ 与速度 $v_t$ | $W_2$ 的几何路径和连续流动 |

复杂度上，离散精确 OT 通常会落到线性规划或最小费用流。若源和目标各有 $n$ 个支撑点，变量规模是 $O(n^2)$。工程上几百到几千点后就可能变重。Sinkhorn 通过熵正则把问题变得更容易迭代，每轮通常需要处理 $n\times n$ 成本矩阵，因此单轮是 $O(n^2)$，但它给的是带偏差的近似解。

---

## 代码实现

下面代码实现一个最小 Sinkhorn 近似。Sinkhorn 是一种带熵正则的最优传输算法：它不直接求精确最优计划，而是在目标中加入平滑项，让运输计划更容易通过矩阵缩放求出来。

```python
import numpy as np

def sinkhorn(a, b, x, y, epsilon=0.1, n_iter=500):
    """
    a, b: 源分布和目标分布的权重，和为 1
    x, y: 一维支撑点
    epsilon: 熵正则强度
    """
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)

    C = np.abs(x[:, None] - y[None, :])
    K = np.exp(-C / epsilon)

    u = np.ones_like(a)
    v = np.ones_like(b)

    for _ in range(n_iter):
        u = a / (K @ v)
        v = b / (K.T @ u)

    P = np.diag(u) @ K @ np.diag(v)
    cost = np.sum(P * C)
    return cost, P, C

a = np.array([0.5, 0.5])
b = np.array([1.0])
x = np.array([0.0, 2.0])
y = np.array([1.0])

cost, P, C = sinkhorn(a, b, x, y, epsilon=0.05)

assert C.shape == (2, 1)
assert np.allclose(P.sum(axis=1), a, atol=1e-6)
assert np.allclose(P.sum(axis=0), b, atol=1e-6)
assert abs(cost - 1.0) < 1e-3
```

这个例子的成本矩阵是：

| 源点 / 目标点 | $1$ |
|---|---:|
| $0$ | $1$ |
| $2$ | $1$ |

Sinkhorn 的流程可以概括为：

1. 构造成本矩阵 $C_{ij}=|x_i-y_j|^p$。
2. 计算核矩阵 $K_{ij}=\exp(-C_{ij}/\varepsilon)$。
3. 交替更新缩放向量 $u$ 和 $v$，让运输计划满足边缘约束。
4. 得到近似运输计划 $P=\mathrm{diag}(u)K\mathrm{diag}(v)$。
5. 用 $\sum_{ij}P_{ij}C_{ij}$ 估计运输成本。

如果写 WGAN，critic 的核心目标可以简化成：

```python
# real_score = critic(real_images).mean()
# fake_score = critic(generator(z).detach()).mean()
# critic_loss = fake_score - real_score + gradient_penalty
```

这里 critic 不是普通二分类器。它近似的是 $W_1$ 对偶中的函数 $f$，所以必须控制 Lipschitz 约束。

---

## 工程权衡与常见坑

Wasserstein 距离的工程难点主要在计算和约束，而不是公式本身。

| 现象 | 原因 | 规避方式 |
|---|---|---|
| 精确 OT 很慢 | 离散计划变量是 $O(n^2)$ | 小规模用精确 OT，大规模用 Sinkhorn 或 sliced OT |
| Sinkhorn 结果过平滑 | $\varepsilon$ 太大 | 降低正则强度，或做 debiased Sinkhorn |
| Sinkhorn 数值不稳定 | $\varepsilon$ 太小，指数下溢 | 使用 log-domain Sinkhorn |
| WGAN critic 训练异常 | 没有满足 $1$-Lipschitz 约束 | 使用 gradient penalty 或 spectral normalization |
| mini-batch OT 偏差大 | 小 batch 不能代表全局分布 | 增大 batch、多次估计、使用特征空间约束 |

反例很重要。令：

$$
\mu=\delta_0,\qquad \nu_\theta=\delta_\theta
$$

当 $\theta\ne0$ 时，两个分布支撑不重叠。JS 散度会保持常数 $\log 2$，它看不出 $\theta=0.1$ 和 $\theta=10$ 的距离差异。但：

$$
W_1(\mu,\nu_\theta)=|\theta|
$$

它会随着位置移动连续变化。这解释了为什么 WGAN 在某些生成任务中比原始 GAN 更容易得到有意义的训练信号。

但 Wasserstein 也不是免费午餐。高维空间中直接做 OT 很贵，mini-batch 上算出来的 Wasserstein 距离也不是全量分布距离的无偏估计。也就是说，把每个 batch 的 OT loss 直接当作真实分布距离，可能会让训练目标偏离原问题。

| 距离 | 优点 | 常见问题 |
|---|---|---|
| $W_1$ | 对支撑不重叠更稳定，适合 WGAN | 需要 Lipschitz 约束 |
| $W_2$ | 几何结构清晰，适合插值和连续流动 | 对远距离更敏感，计算也重 |
| KL | 概率解释清楚，常用于变分推断 | 支撑不匹配时可能发散 |
| JS | 对称且有界，传统 GAN 常用 | 分布不重叠时容易饱和 |

真实工程中，WGAN-GP 通常比原始 WGAN 的 weight clipping 更常用。weight clipping 是直接把 critic 参数裁剪到固定范围，简单但会损伤模型表达能力；gradient penalty 则惩罚梯度范数偏离 1，更贴近 Lipschitz 约束的目的。

---

## 替代方案与适用边界

不要把 Wasserstein 距离当成所有分布问题的默认答案。它适合需要空间几何信息的任务，但计算代价也更高。

| 任务类型 | 推荐距离 | 原因 |
|---|---|---|
| 生成模型中分布支撑不重叠 | $W_1$ | 仍能提供距离变化信号 |
| 分布插值、形状变形、连续流 | $W_2$ | 几何路径更自然 |
| 概率模型拟合、变分推断 | KL | 和最大似然关系紧密 |
| 两样本检验、核方法 | MMD | 实现简单，避免求运输计划 |
| 点云粗匹配 | Chamfer 距离 | 便宜，适合近似几何匹配 |
| 大规模近似 OT | Sinkhorn | 用偏差换速度 |

MMD 是最大均值差异，用核函数比较两个分布的均值嵌入；它不需要显式求运输计划。Chamfer 距离常用于点云，计算每个点到另一组点的最近距离，但它不保证质量守恒，因此不是严格的最优传输。

图像生成是一个典型边界例子。真实图像分布和生成图像分布位于高维空间中的低维流形附近，早期训练时两者可能几乎不相交。此时 JS 散度很难给出有效梯度，$W_1$ 更有优势。但如果只是在监督学习里比较预测概率和标签分布，交叉熵或 KL 往往更直接、更便宜。

什么时候不用 OT：样本规模很大、只需要分类概率校准、底层距离没有清晰定义、计算预算很紧时，不应优先上 Wasserstein。

什么时候必须近似 OT：点数上千以上、训练循环中反复计算、图像或文本特征维度较高时，通常需要 Sinkhorn、sliced Wasserstein、低秩近似或在特征空间中计算。

---

## 参考资料

| 阅读顺序 | 资料 | 用途 |
|---:|---|---|
| 1 | Arjovsky, Chintala, Bottou, “Wasserstein GAN” | 理解为什么生成模型要引入 $W_1$ |
| 2 | Gulrajani et al., “Improved Training of Wasserstein GANs” | 理解 WGAN-GP 和 Lipschitz 约束的工程做法 |
| 3 | Cuturi, “Sinkhorn Distances: Lightspeed Computation of Optimal Transport” | 理解熵正则和 Sinkhorn 近似 |
| 4 | Peyré & Cuturi, “Computational Optimal Transport” | 系统学习计算最优传输 |
| 5 | Figalli & Glaudo, *An Invitation to Optimal Transport, Wasserstein Distances, and Gradient Flows* | 建立 $W_2$、测地线和梯度流的数学图景 |

- Arjovsky, Chintala, Bottou, “Wasserstein GAN”, ICML 2017：WGAN 的原始论文，适合先读动机和对偶形式。
- Gulrajani et al., “Improved Training of Wasserstein GANs”, NeurIPS 2017：解释 gradient penalty 为什么比简单 weight clipping 更实用。
- Cuturi, “Sinkhorn Distances: Lightspeed Computation of Optimal Transport”, NeurIPS 2013：工程实现 Sinkhorn 的核心来源。
- Peyré & Cuturi, “Computational Optimal Transport”：从离散 OT、Sinkhorn 到应用的完整综述。
- Figalli & Glaudo, *An Invitation to Optimal Transport, Wasserstein Distances, and Gradient Flows*：适合理解 Wasserstein 空间的几何结构。
