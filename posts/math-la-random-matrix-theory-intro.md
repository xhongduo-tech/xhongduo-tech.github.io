## 核心结论

随机矩阵理论研究的是“大尺寸随机矩阵的谱”。这里的“谱”，主要指特征值整体的统计形状，而不是某一个单独特征值。对初学者，最实用的理解方式是：把一批特征值看成一个总体，问它们在纯噪声条件下应该如何分布。

在最常见的两类模型里，当矩阵维度足够大时，会出现稳定的主干形状。

1. 对称随机矩阵的特征值密度收敛到 Wigner 半圆律：
$$
\rho_W(\lambda)=\frac{1}{2\pi\sigma^2}\sqrt{4\sigma^2-\lambda^2},\qquad |\lambda|\le 2\sigma
$$

2. 样本协方差矩阵，也常写成 Wishart 型矩阵，其特征值密度收敛到 Marchenko-Pastur 分布：
$$
\rho_{MP}(\lambda)=\frac{\sqrt{(\lambda_+-\lambda)(\lambda-\lambda_-)}}{2\pi\sigma^2 c\,\lambda},\qquad \lambda\in[\lambda_-,\lambda_+]
$$
其中
$$
\lambda_\pm=\sigma^2(1\pm \sqrt{c})^2,\qquad c=\frac{p}{n}
$$

这里的 $c=p/n$ 是维度与样本数的比例。若 $p$ 很大但样本数 $n$ 不足，谱的支撑区间会明显变宽；若样本很多，噪声谱会更集中。

最重要的工程结论不是“理论曲线长什么样”，而是“偏离理论主干的部分通常更有信息”。如果一个权重矩阵、协方差矩阵或 Hessian 的大多数特征值都落在理论主干里，只有少数离群值跑到外面，那么这些离群值往往对应真实信号、低秩结构或任务相关方向。

先看一个极小的玩具例子。令
$$
A=
\begin{bmatrix}
0 & 1\\
1 & 0
\end{bmatrix}
$$
则它的特征值是 $\pm 1$。这个矩阵只有 $2\times 2$，远没有进入大维极限，但它已经给出一个直觉：谱不会无界散开，而是落在有限区间内。对标准 Wigner 模型，极限支撑区间是 $[-2,2]$。

再看一个工程例子。分析训练后的神经网络权重矩阵或激活协方差矩阵时，常会发现大部分谱段近似贴近 MP 主干，只有少数特征值明显高于理论上边界。这些离群值通常对应可解释的强方向，例如类别分离方向、低秩表示结构或训练中反复强化的模式。它们往往是压缩、剪枝、秩约简时最不应该被直接抹掉的部分。

如果只记一条结论，可以记下面这句话：

| 现象 | 更可能的解释 |
| --- | --- |
| 大多数特征值落在理论主干内 | 主要是噪声或高维随机波动 |
| 少数特征值稳定跑出主干 | 更可能对应真实结构或强信号 |

---

## 问题定义与边界

要讨论随机矩阵理论，先明确对象、归一化方式和适用边界，否则公式很容易被机械套用。

我们主要看两类矩阵：

| 模型 | 矩阵形式 | 输入要求 | 归一化方式 | 理论主干 |
| --- | --- | --- | --- | --- |
| Wigner | $N\times N$ 对称矩阵 | 上三角元素近似独立、均值 0、方差受控 | 通常按 $\frac{1}{\sqrt{N}}$ 缩放 | 半圆律 |
| Wishart / 样本协方差 | $S=\frac{1}{n}XX^\top$ | $X\in\mathbb{R}^{p\times n}$，元素近似独立、中心化、方差受控 | 协方差自带 $\frac{1}{n}$ | MP 分布 |

这里几个术语先说清楚。

| 术语 | 数学含义 | 直白解释 |
| --- | --- | --- |
| 中心化 | 每个变量减去自己的均值 | 去掉整体平移，只保留波动 |
| 方差受控 | 元素方差不随维度乱飘 | 每个坐标的波动尺度在同一量级 |
| 经验谱分布 | 把全部特征值当样本形成的分布 | 看“整堆特征值”而不是单个值 |
| bulk | 谱主干区域 | 大多数特征值堆在一起的区间 |
| outlier | 离群特征值 | 跳出主干边界的少数值 |

边界条件里最关键的是下面三条。

1. 不是所有矩阵都服从半圆律或 MP 分布。  
   如果矩阵有强依赖、强块结构、强稀疏性、明显重尾，主干可能完全变形。

2. 理论讨论的是大尺寸极限。  
   当 $N,p,n$ 不大时，经验直方图会有明显抖动，边界也会有有限样本偏差。此时理论给出的主要是量级和形状基线，不是逐点精确拟合。

3. 研究重点通常分成 bulk 和 outlier。  
   bulk 告诉你“纯噪声应该长什么样”；outlier 告诉你“哪些方向可能有额外结构”。

对初学者，一个最稳妥的理解是：随机矩阵理论先定义一个“噪声基线”，然后再判断经验谱里哪些部分偏离了这条基线。

下面看一个小的 Wishart 玩具例子。设
$$
X=
\begin{bmatrix}
1 & -1 & 0\\
0 & 1 & -1
\end{bmatrix}
$$
则
$$
S=\frac{1}{3}XX^\top
=
\frac{1}{3}
\begin{bmatrix}
2 & -1\\
-1 & 2
\end{bmatrix}
$$
它的特征值是 $1,\frac{1}{3}$。

若 $p=2,n=3$，则
$$
c=\frac{p}{n}=\frac{2}{3}
$$
在噪声方差 $\sigma^2=1$ 的情况下，理论边界为
$$
\lambda_\pm=(1\pm \sqrt{2/3})^2\approx 0.034,\ 3.30
$$
这两个特征值都落在边界内。这个例子不能说明“小样本已经收敛到 MP 分布”，但它能说明一件事：MP 边界先给出了一个合理的尺度范围，你可以先用它判断经验谱是否明显异常。

从工程角度，更实用的判断方式如下：

| 你看到的现象 | 初步解释 |
| --- | --- |
| 大部分特征值在 $[\lambda_-,\lambda_+]$ 内 | 更像噪声主干 |
| 仅个别特征值略微越界 | 可能是有限样本波动 |
| 有一串稳定的特征值远超上界 | 更像低秩结构或真实信号 |
| 谱整体明显错位 | 通常先检查中心化和归一化 |

---

## 核心机制与推导

随机矩阵理论的关键不是背结论，而是理解这些结论为什么会稳定出现。下面只保留入门所需的主线，不做严格证明。

### 1. Wigner 半圆律的机制

设 $W_N$ 是对称随机矩阵，元素经过 $\frac{1}{\sqrt{N}}$ 缩放。我们关心经验谱分布
$$
\mu_{W_N}=\frac{1}{N}\sum_{i=1}^N \delta_{\lambda_i}
$$
这里 $\delta_{\lambda_i}$ 表示在特征值 $\lambda_i$ 位置放一个单位质量。直白地说，就是“每个特征值都算一个样本点”。

常见推导思路是迹法。因为
$$
\frac{1}{N}\mathrm{Tr}(W_N^k)=\frac{1}{N}\sum_{i=1}^N \lambda_i^k
$$
右边恰好是经验谱分布的第 $k$ 阶矩。因此，研究谱分布可以转化成研究矩：
$$
m_k=\int \lambda^k\,d\mu_{W_N}(\lambda)
$$

关键点在于，展开 $\mathrm{Tr}(W_N^k)$ 后会得到大量矩阵元素乘积项：
$$
\mathrm{Tr}(W_N^k)=\sum_{i_1,\dots,i_k} W_{i_1 i_2}W_{i_2 i_3}\cdots W_{i_k i_1}
$$
由于元素独立且均值为 0，大部分项在求期望后都会消失。能留下来的主导项，必须满足某种“合法配对”结构。这个配对计数最终对应 Catalan 数。

前几个 Catalan 数是：
$$
1,\ 1,\ 2,\ 5,\ 14,\dots
$$
它们正是半圆分布偶数阶矩的来源。更具体地说，标准半圆分布满足
$$
\int \lambda^{2k}\rho_W(\lambda)\,d\lambda = C_k,\qquad
\int \lambda^{2k+1}\rho_W(\lambda)\,d\lambda=0
$$
其中 $C_k$ 是第 $k$ 个 Catalan 数。

这就解释了为什么半圆律会稳定出现：不是因为矩阵长得“像一个半圆”，而是因为在独立、零均值、适当缩放这三个条件下，矩展开后真正能累积贡献的组合非常少，而这些组合恰好生成半圆分布的矩结构。

用最简流程概括：

| 步骤 | 数学对象 | 含义 |
| --- | --- | --- |
| 1 | $\frac{1}{N}\mathrm{Tr}(W_N^k)$ | 把谱问题改写成矩问题 |
| 2 | 展开成元素乘积求和 | 把特征值问题改写成索引计数 |
| 3 | 利用独立性和零均值 | 大多数项抵消，只保留配对项 |
| 4 | 配对项对应 Catalan 结构 | 极限矩与半圆分布匹配 |
| 5 | 由矩确定极限分布 | 得到半圆律 |

初学者可以先接受下面这个版本：  
半圆律不是某个特殊分布的偶然结果，而是“大量弱相关随机项在对称约束和 $1/\sqrt{N}$ 缩放下”的稳定谱极限。

### 2. MP 分布的机制

对样本协方差矩阵
$$
S=\frac{1}{n}XX^\top,\qquad X\in\mathbb{R}^{p\times n}
$$
若 $p,n\to\infty$ 且
$$
\frac{p}{n}\to c\in(0,\infty)
$$
则谱主干趋向 Marchenko-Pastur 分布。

这里常用的工具是 Stieltjes 变换。定义
$$
m(z)=\int \frac{1}{\lambda-z}\rho(\lambda)\,d\lambda,\qquad z\in\mathbb{C}\setminus \mathbb{R}
$$
它的作用是把“分布”编码成一个复函数。这样做的好处是，谱密度问题会转化为解析函数满足的代数关系。

对 Wishart 型极限分布，可以得到一个自洽方程。常见写法之一是
$$
z m(z)^2 + (z-1-c)m(z) + 1 = 0
$$
这个式子本身不是结论的终点，它只是说明：在大维极限下，谱的整体行为被压缩成一个可解的方程。

求出满足正确解析条件的分支后，再用反演关系
$$
\rho(\lambda)=\frac{1}{\pi}\lim_{\varepsilon\to 0^+}\operatorname{Im} m(\lambda+i\varepsilon)
$$
就能恢复谱密度，最终得到
$$
\rho_{MP}(\lambda)=\frac{\sqrt{(\lambda_+-\lambda)(\lambda-\lambda_-)}}{2\pi\sigma^2 c\,\lambda},\qquad \lambda\in[\lambda_-,\lambda_+]
$$
以及边界
$$
\lambda_\pm=\sigma^2(1\pm \sqrt{c})^2
$$

这个边界公式本身就很值得记住，因为它直接告诉你三个量之间的关系：

| 量 | 变大后的影响 |
| --- | --- |
| $\sigma^2$ | 整个谱整体放大 |
| $c=p/n$ | 主干宽度改变，$c$ 越大越宽 |
| $n$ 增大且 $p$ 固定 | 等价于 $c$ 变小，噪声谱更集中 |

还有一个初学者常忽略的点：当 $c>1$ 时，样本协方差矩阵会出现零特征值堆积。这不是异常，而是因为维度比样本数还大，矩阵秩最多只有 $n$。

### 3. 普适性为什么重要

普适性指的是：只要满足较宽松的条件，极限谱的主干形状往往不依赖元素的具体分布细节。元素可以来自高斯分布、均匀分布，甚至许多非高斯轻尾分布，主干仍然会接近同一种极限律。

这件事在工程上非常重要，因为真实数据几乎从不精确服从高斯模型。若理论只在“精确高斯”下成立，它的价值会很有限；而普适性说明，很多结论其实是结构性的，而不是分布细节偶然造成的。

可以把它理解成下面这张表：

| 问题 | 结论 |
| --- | --- |
| 元素必须是高斯吗 | 不必须，很多轻尾分布都可以 |
| 元素分布换了，主干会完全变吗 | 通常不会，主干常保持稳定 |
| 什么情况普适性会失效 | 强依赖、重尾、强稀疏、明显结构约束 |

因此，工程里使用 RMT 时，真正该优先检查的不是“是不是正态分布”，而是“独立性是否近似成立、尺度是否统一、是否存在明显结构破坏”。

### 4. 离群值从哪里来

如果矩阵里叠加了低秩信号，那么 bulk 仍主要由噪声决定，但信号可能会推出一个或几个离群特征值。

例如
$$
M = W + \theta uu^\top
$$
其中 $W$ 是噪声矩阵，$u$ 是单位向量，$\theta$ 控制信号强度。这个模型叫作 rank-one spike 的典型形式。直白地说，它是在“随机噪声背景”上加了一条明确方向。

这时会发生两件事：

1. bulk 的大体形状仍由噪声模型决定；
2. 若 $\theta$ 足够强，一个特征值会脱离 bulk 上边界，形成 outlier。

这类现象是很多谱方法的理论基础。因为一旦你知道纯噪声时主干应该在哪里，就可以把主干外的部分当作候选结构，再进一步解释。

在深度学习场景里，这些结构可能来自：

| 结构来源 | 可能对应的谱现象 |
| --- | --- |
| 某层学到少数主导方向 | 少数大离群值 |
| 数据类别分离 | 若干稳定高特征值 |
| 优化过程中形成相关性 | 主干变形或边界外推 |
| 批归一化、残差、注意力耦合 | 谱相关性增强，独立假设变差 |

所以工程上最实用的逻辑不是“看到大特征值就兴奋”，而是先问：  
它是否稳定地脱离纯噪声主干？是否在不同随机种子、不同 epoch、不同数据切片下都出现？只有这样，它才更可能对应真实结构。

---

## 代码实现

下面给出一个可以直接运行的 Python 示例。代码分成四部分：

1. 生成 Wigner 矩阵并验证半圆律的支撑区间；
2. 生成 Wishart 矩阵并验证 MP 边界；
3. 加入低秩信号，观察离群值；
4. 画出直方图与理论密度，方便初学者建立直觉。

运行环境只需要 `numpy` 和 `matplotlib`。

```python
import numpy as np
import matplotlib.pyplot as plt


def wigner(N, sigma=1.0, seed=0):
    """Generate a symmetric Wigner matrix with variance scaled by 1/N."""
    rng = np.random.default_rng(seed)
    A = rng.normal(0.0, sigma, size=(N, N))
    W = (A + A.T) / 2.0
    return W / np.sqrt(N)


def wishart(p, n, sigma=1.0, seed=0, center=True):
    """Generate a sample covariance matrix S = XX^T / n."""
    rng = np.random.default_rng(seed)
    X = rng.normal(0.0, sigma, size=(p, n))
    if center:
        X = X - X.mean(axis=1, keepdims=True)
    return (X @ X.T) / n


def mp_edges(sigma2, c):
    """Marchenko-Pastur support edges."""
    lam_minus = sigma2 * (1.0 - np.sqrt(c)) ** 2
    lam_plus = sigma2 * (1.0 + np.sqrt(c)) ** 2
    return lam_minus, lam_plus


def semicircle_density(x, sigma=1.0):
    """Wigner semicircle density."""
    radius = 2.0 * sigma
    y = np.zeros_like(x, dtype=float)
    mask = np.abs(x) <= radius
    y[mask] = np.sqrt(radius**2 - x[mask] ** 2) / (2.0 * np.pi * sigma**2)
    return y


def mp_density(x, sigma2, c):
    """Marchenko-Pastur density."""
    lam_minus, lam_plus = mp_edges(sigma2, c)
    y = np.zeros_like(x, dtype=float)
    mask = (x >= lam_minus) & (x <= lam_plus)
    numerator = np.sqrt((lam_plus - x[mask]) * (x[mask] - lam_minus))
    denominator = 2.0 * np.pi * sigma2 * c * x[mask]
    y[mask] = numerator / denominator
    return y


def spike_signal(p, n, rank=3, strength=2.5, seed=0):
    """Create a low-rank signal matrix."""
    rng = np.random.default_rng(seed)
    U = rng.normal(size=(p, rank))
    V = rng.normal(size=(rank, n))
    U /= np.linalg.norm(U, axis=0, keepdims=True) + 1e-12
    return strength * (U @ V)


def main():
    # Part 1: Wigner example
    N = 400
    sigma = 1.0
    W = wigner(N, sigma=sigma, seed=1)
    eigs_W = np.linalg.eigvalsh(W)

    # Standard semicircle support is [-2 sigma, 2 sigma]
    assert eigs_W.min() > -2.5
    assert eigs_W.max() < 2.5

    # Part 2: Wishart example
    p, n = 250, 800
    sigma_x = 1.0
    sigma2 = sigma_x**2
    c = p / n

    S = wishart(p, n, sigma=sigma_x, seed=2, center=True)
    eigs_S = np.linalg.eigvalsh(S)

    lam_minus, lam_plus = mp_edges(sigma2, c)

    # Finite-sample tolerance
    tol = 0.20
    assert eigs_S.min() >= lam_minus - tol
    assert eigs_S.max() <= lam_plus + tol

    # Part 3: Add low-rank signal
    rng = np.random.default_rng(3)
    X_noise = rng.normal(0.0, sigma_x, size=(p, n))
    X_signal = X_noise + spike_signal(p, n, rank=3, strength=8.0, seed=4)
    X_signal = X_signal - X_signal.mean(axis=1, keepdims=True)
    S_signal = (X_signal @ X_signal.T) / n
    eigs_signal = np.linalg.eigvalsh(S_signal)

    # The top eigenvalue should exceed the MP upper edge
    assert eigs_signal.max() > lam_plus

    print("MP edges:", (lam_minus, lam_plus))
    print("Wishart eig range:", (eigs_S.min(), eigs_S.max()))
    print("Signal top eigenvalue:", eigs_signal.max())

    # Part 4: Plot histograms against theoretical curves
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    # Wigner plot
    x1 = np.linspace(-2.2, 2.2, 500)
    axes[0].hist(eigs_W, bins=40, density=True, alpha=0.6, color="#4C78A8", label="empirical")
    axes[0].plot(x1, semicircle_density(x1, sigma=sigma), color="#E45756", lw=2, label="semicircle")
    axes[0].axvline(-2.0 * sigma, color="black", ls="--", lw=1)
    axes[0].axvline(2.0 * sigma, color="black", ls="--", lw=1)
    axes[0].set_title("Wigner Spectrum")
    axes[0].set_xlabel("eigenvalue")
    axes[0].set_ylabel("density")
    axes[0].legend()

    # Wishart / MP plot
    x2 = np.linspace(max(1e-4, lam_minus * 0.8), lam_plus * 1.2, 500)
    axes[1].hist(eigs_S, bins=40, density=True, alpha=0.6, color="#72B7B2", label="empirical")
    axes[1].plot(x2, mp_density(x2, sigma2=sigma2, c=c), color="#F58518", lw=2, label="MP")
    axes[1].axvline(lam_minus, color="black", ls="--", lw=1, label="MP edges")
    axes[1].axvline(lam_plus, color="black", ls="--", lw=1)
    axes[1].set_title("Wishart Spectrum")
    axes[1].set_xlabel("eigenvalue")
    axes[1].set_ylabel("density")
    axes[1].legend()

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
```

这段代码可以直接运行。若图像正常画出，通常会看到：

1. Wigner 直方图整体贴近半圆形；
2. Wishart 特征值主要落在 MP 边界内；
3. 加入低秩信号后，最大的几个特征值会明显跑到上边界之外。

对初学者，更重要的是理解每一步在做什么，而不是只看结果。

| 代码步骤 | 对应问题 | 目的 |
| --- | --- | --- |
| `wigner()` | 对称噪声矩阵长什么样 | 看半圆律主干 |
| `wishart()` | 协方差噪声谱长什么样 | 看 MP 主干 |
| `mp_edges()` | 理论边界在哪里 | 判断是否有离群值 |
| `spike_signal()` | 加入低秩结构后会怎样 | 观察 outlier 形成 |

若想把代码改成更贴近真实任务的版本，可以把随机高斯矩阵换成实际层的激活矩阵 $X$，再计算
$$
S=\frac{1}{n}XX^\top
$$
然后看它的特征值是否大多落在 MP 主干内。一个典型流程如下：

| 步骤 | 操作 | 目的 |
| --- | --- | --- |
| 1 | 取某层权重矩阵或激活矩阵 | 建立待分析对象 |
| 2 | 对每个特征维度做中心化 | 去掉均值偏移 |
| 3 | 必要时做尺度归一化 | 让理论边界可比 |
| 4 | 计算 $XX^\top/n$ 或奇异值谱 | 得到经验谱 |
| 5 | 与 MP 或半圆主干对比 | 分离噪声段与信号段 |
| 6 | 检查边界外特征值是否稳定 | 判断是否是真结构 |

例如分析全连接层权重 $W\in\mathbb{R}^{m\times n}$ 时，常看的是 $WW^\top$ 或奇异值谱。若大部分谱贴近 MP 主干，而只有少数特征值显著越过上边界，那么一个稳妥的解释是：该层包含少数主导方向。此时围绕 bulk 做去噪、压缩或截断，通常比对所有方向一刀切更合理。

还要注意一件事：真实神经网络并不严格满足独立同分布假设，因此理论曲线不必“精确贴合”。RMT 在工程里的作用更像基线，而不是精密拟合器。

---

## 工程权衡与常见坑

随机矩阵理论在工程上好用，但前提是解释方式正确。最常见的问题不是公式推错，而是前处理、尺度和语境判断出错。

| 问题 | 现象 | 原因 | 规避策略 |
| --- | --- | --- | --- |
| 小样本噪声 | 直方图起伏很大 | 有限样本波动明显 | 增大 $N,p,n$，或做核密度平滑 |
| 未中心化 | 谱整体偏移，最大特征值异常大 | 均值结构被误当成信号 | 先减均值再比较理论边界 |
| 未归一化 | $\lambda_\pm$ 对不上 | 方差尺度错误 | 先估计噪声方差，再做缩放 |
| 把 bulk 当信号 | 误报很多“结构” | 忽视普适主干 | 先用 RMT 设基线，再看离群值 |
| 把离群值都当真信号 | 误判训练噪声 | 优化动态也会造大特征值 | 结合时间演化、层间对比验证 |

### 1. 有限样本误差不能忽略

理论说的是极限分布，不是每次实验都精确贴在理论曲线上。若 $N=50$ 或 $p,n$ 只有几十，经验谱出现起伏是正常现象。工程上不能因为有几个点越界就认定“发现结构”，也不能因为图像不够平滑就否定理论。

更稳妥的做法是看下面三个量：

| 该看什么 | 为什么 |
| --- | --- |
| 主干整体位置是否大致对齐 | 检查归一化是否正确 |
| 越界特征值是否显著超出边界 | 区分有限样本抖动和真正 outlier |
| 不同随机种子或批次下是否稳定 | 检查结构是否可重复 |

### 2. 归一化比公式更重要

很多误判都不是理论错，而是前处理错。尤其在协方差场景下，如果没有中心化、没有考虑不同维度的尺度差异，MP 边界几乎一定会错位。

最常见的错误有两类：

| 错误做法 | 后果 |
| --- | --- |
| 直接对原始激活矩阵算谱 | 均值分量可能制造假离群值 |
| 忽略各特征方差差异 | 谱被异方差拉宽，和标准 MP 无法直接比 |

因此，在使用 MP 分布前，至少要回答两件事：

1. 你的数据是否已经中心化？
2. 你比较的理论边界所用的噪声方差，是否与实际数据尺度一致？

若这两件事没处理好，后面的离群解释几乎都不可靠。

### 3. 离群值要结合问题语境解释

离群值不自动等于“有用特征”。它只说明“有东西偏离纯噪声模型”。这个“东西”可能是任务相关结构，也可能是训练副作用、数据偏斜或模型约束。

常见来源如下：

| 离群值来源 | 是否一定有用 |
| --- | --- |
| 真实任务信号 | 不一定，但概率较高 |
| 数据分布偏斜 | 不一定 |
| 训练初期优化噪声 | 往往不稳定 |
| 结构约束、残差、归一化层 | 需要结合模型判断 |
| 批采样偏差 | 常常只是暂时现象 |

所以工程上最好做对照实验：

1. 训练前后比；
2. 不同层之间比；
3. 不同 epoch 比；
4. 不同数据子集比；
5. 不同随机种子比。

只有当离群值在这些对照下都稳定存在，并且对应方向还能和任务行为挂钩时，它才更像真正有用的结构。

---

## 替代方案与适用边界

RMT 不是谱分析的唯一工具，也不是所有场景的最佳工具。它的优势在于能提供“噪声主干”的明确基线，但当矩阵具有强规则结构时，单靠 RMT 往往不够。

| 方法 | 适用对象 | 优势 | 局限 |
| --- | --- | --- | --- |
| RMT | 大规模近独立噪声 + 少量结构 | 能给出明确噪声主干和理论边界 | 对依赖、重尾、稀疏结构敏感 |
| PCA / 低秩模型 | 信号集中在少数主方向 | 解释直观，便于压缩降维 | 没有显式噪声基线时易过拟合 |
| 稀疏矩阵建模 | 图结构、局部连接、条件独立 | 适合高维稀疏依赖 | 不擅长描述连续谱主干 |
| 稳健协方差估计 | 重尾、异常值明显 | 对异常样本更稳健 | 与标准 RMT 公式不直接兼容 |
| 自由概率 / 更一般谱工具 | 复杂矩阵组合、卷积型谱问题 | 能处理更复杂的谱运算 | 入门门槛高，工程解释较难 |

适用边界可以概括成一句话：  
当矩阵看起来像“噪声主导 + 少量结构偏移”时，RMT 很有力；当矩阵本身有强规则结构时，RMT 更适合作为基线，而不是最终模型。

一个新手更容易理解的例子是：若某层权重矩阵近似等于“低秩信号 + 随机扰动”，那么 MP 主干可以描述噪声部分，而 PCA 可以直接把主干外的主方向提取出来。此时最实用的方法不是“RMT 或 PCA 二选一”，而是先后配合：

1. 用 RMT 判断哪些特征值仍在噪声主干内；
2. 用 PCA 解释主干外的离群方向；
3. 再决定剪枝、蒸馏、低秩分解或异常检测策略。

如果矩阵元素明显重尾，例如极少数元素异常大，标准半圆律和标准 MP 分布通常会失效。这时更合适的方向可能是：

| 场景 | 更合适的方法 |
| --- | --- |
| 异常值很多 | 稳健协方差估计 |
| 重尾分布明显 | 重尾随机矩阵模型 |
| 强相关结构主导 | 显式结构建模 |
| 稀疏图或局部连接 | 图谱方法、稀疏建模 |

因此，RMT 最适合作为第一层判断工具：  
它先告诉你“纯噪声时应当看到什么”，再帮助你筛出“哪些现象值得进一步解释”。

---

## 参考资料

1. Eugene P. Wigner, 1958, *On the Distribution of the Roots of Certain Symmetric Matrices*。用途：半圆律的经典起点，适合理解对称随机矩阵统一谱形状的来源。  
2. V. A. Marchenko, L. A. Pastur, 1967, *Distribution of Eigenvalues for Some Sets of Random Matrices*。用途：MP 分布原始文献，核心结论是样本协方差谱的极限分布。  
3. Terence Tao, 2012, *Topics in Random Matrix Theory*。用途：系统整理 Wigner、Wishart、普适性等主题，适合作为进阶教材。  
4. Zhidong Bai, Jack W. Silverstein, 2010, *Spectral Analysis of Large Dimensional Random Matrices*。用途：大维随机矩阵谱分析的标准参考书，适合查严格条件与证明框架。  
5. Alice Guionnet, 2009, *Large Random Matrices: Lectures on Macroscopic Asymptotics*。用途：从大尺度极限角度讲半圆律、Stieltjes 变换与谱分布，是从“整体形状”入门的好材料。  
6. Romain Couillet, Mérouane Debbah, 2011, *Random Matrix Methods for Wireless Communications*。用途：展示 RMT 如何进入工程领域，适合理解“理论主干 + 系统建模”的实际用法。  
7. Charles Martin, Michael W. Mahoney 等关于深度网络谱分析的系列研究。用途：说明神经网络权重谱中“主干近似随机、离群值携带结构”的工程解释，适合把 RMT 与模型压缩、泛化分析联系起来。
