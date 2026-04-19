## 核心结论

正交多项式是一组在指定区间和指定权重函数下互相正交的多项式。这里的“正交”可以理解为：两个不同阶数的多项式相乘后再按权重积分，结果为 0。

设区间为 $[a,b]$，权重函数为 $w(x)$，多项式族为 $\{p_0,p_1,p_2,\dots\}$。正交性写成：

$$
\int_a^b p_m(x)p_n(x)w(x)\,dx = 0,\quad m\ne n
$$

它的计算价值不只在“正交”这个性质本身，而在正交性会导出三项递推关系：

$$
x p_n(x)=a_{n+1}p_{n+1}(x)+b_n p_n(x)+a_n p_{n-1}(x)
$$

三项递推的含义是：用 $x$ 乘上第 $n$ 阶多项式后，结果只需要相邻三阶多项式就能表示，不会扩散到所有阶数。这让高阶多项式的计算可以通过少量局部系数完成。

高斯积分是一类利用正交多项式结构的数值积分方法。它把积分近似为有限个节点上的加权求和：

$$
\int_a^b f(x)w(x)\,dx \approx \sum_{i=1}^n \lambda_i f(x_i)
$$

其中 $x_i$ 是某个 $n$ 阶正交多项式的零点，$\lambda_i$ 是对应权重。高斯积分的关键结论是：$n$ 个节点可以对次数不超过 $2n-1$ 的多项式精确积分。

玩具例子：2 点 Gauss-Legendre 积分在 $[-1,1]$ 上的节点是 $\pm 1/\sqrt{3}$，权重都是 1。计算 $\int_{-1}^{1}x^2dx$ 时：

$$
1\cdot\left(-\frac{1}{\sqrt{3}}\right)^2+
1\cdot\left(\frac{1}{\sqrt{3}}\right)^2
=\frac{2}{3}
$$

真值也是 $\frac{2}{3}$，所以这个二节点规则完全精确。

| 对比项 | 等距采样积分 | 高斯求积 |
|---|---|---|
| 节点位置 | 均匀放置 | 正交多项式零点 |
| 权重 | 常见规则中较简单 | 由正交结构决定 |
| 多项式精度 | 通常较低 | $n$ 点可精确到 $2n-1$ 次 |
| 适用场景 | 粗略估计、规则网格 | 低维、平滑、区间明确的积分 |
| 计算成本 | 节点容易生成 | 节点和权重需计算或查库 |

---

## 问题定义与边界

本文讨论的是一维加权积分中的正交多项式和高斯积分，不讨论任意多项式插值，也不讨论所有数值积分问题。

加权积分是指被积函数外面还有一个固定权重函数：

$$
I=\int_a^b f(x)w(x)\,dx
$$

这里 $f(x)$ 是真正需要积分的函数，$w(x)$ 是积分规则预先绑定的权重。不同的区间和权重函数会对应不同的正交多项式族。

Legendre 多项式是最常见的特例。它定义在 $[-1,1]$，权重函数是：

$$
w(x)=1
$$

因此 Gauss-Legendre 积分适合计算：

$$
\int_{-1}^{1} f(x)\,dx
$$

如果原始积分区间不是 $[-1,1]$，可以做线性变量变换。设 $t\in[-1,1]$，把它映射到 $x\in[a,b]$：

$$
x=\frac{b-a}{2}t+\frac{a+b}{2}
$$

于是：

$$
\int_a^b f(x)\,dx
=
\frac{b-a}{2}\int_{-1}^{1}
f\left(\frac{b-a}{2}t+\frac{a+b}{2}\right)\,dt
$$

新手容易混淆的一点是：Gauss-Legendre 不是“任何积分都直接套 Legendre 节点”。它只天然匹配有限区间和常数权重。如果积分在 $[0,\infty)$，或者权重是 $e^{-x}$、$e^{-x^2}$、$(1-x^2)^{-1/2}$，就应该换成对应的正交多项式族，或者先做变量变换。

| 多项式族 | 常见区间 | 权重函数 | 常见用途 |
|---|---:|---|---|
| Legendre | $[-1,1]$ | $1$ | 有限区间普通积分 |
| Chebyshev | $[-1,1]$ | $(1-x^2)^{-1/2}$ | 逼近、谱方法、带奇异权重积分 |
| Hermite | $(-\infty,\infty)$ | $e^{-x^2}$ | 高斯分布相关积分 |
| Laguerre | $[0,\infty)$ | $e^{-x}$ | 半无限区间衰减积分 |

真实工程例子：贝叶斯推断中经常要计算边际似然：

$$
Z=\int p(D\mid \theta)p(\theta)\,d\theta
$$

这里 $D$ 是观测数据，$\theta$ 是参数，$p(D\mid\theta)$ 是似然，$p(\theta)$ 是先验。边际似然用于模型比较，也用于归一化后验分布。当 $\theta$ 只有 1 到 3 维，且函数比较平滑时，可以用高斯积分代替大量随机采样。

---

## 核心机制与推导

先看正交性为什么会导出三项递推。

$p_n(x)$ 是 $n$ 次多项式，$x p_n(x)$ 是 $n+1$ 次多项式。因此它最多可以展开到 $p_{n+1}$：

$$
x p_n(x)=c_0p_0(x)+c_1p_1(x)+\cdots+c_{n+1}p_{n+1}(x)
$$

但是正交性会消掉大部分系数。对于 $k\le n-2$，由于 $x p_k(x)$ 的次数最多是 $k+1\le n-1$，它可以由 $p_0,\dots,p_{n-1}$ 表示，所以与 $p_n$ 正交。进一步利用内积对称性：

$$
\langle x p_n,p_k\rangle = \langle p_n, x p_k\rangle = 0
$$

这里的内积是加权积分：

$$
\langle f,g\rangle=\int_a^b f(x)g(x)w(x)\,dx
$$

因此 $x p_n$ 只会留下 $p_{n+1},p_n,p_{n-1}$ 三项。这就是三项递推。

Legendre 多项式的递推公式是：

$$
(n+1)P_{n+1}(x)=(2n+1)xP_n(x)-nP_{n-1}(x)
$$

这条公式说明，只要知道 $P_{n-1}$ 和 $P_n$，就能算出 $P_{n+1}$。工程上这比直接展开高阶多项式更稳定，也更便宜。

三项递推还能写成一个对称三对角矩阵问题。三对角矩阵是只有主对角线和上下相邻对角线可能非零的矩阵。把递推系数放进去，可以得到 Jacobi 矩阵 $J$。高斯积分的节点就是 $J$ 的特征值，权重由特征向量的第一个分量给出。这就是 Golub-Welsch 算法的核心。

| 机制链路 | 含义 |
|---|---|
| 正交性 | 不同阶数的多项式在加权积分下互相分离 |
| 三项递推 | $xp_n$ 只依赖相邻三阶 |
| 三对角矩阵 | 递推系数形成 Jacobi 矩阵 |
| 特征值 | 给出高斯积分节点 |
| 特征向量 | 给出高斯积分权重 |
| 高斯求积 | 用最优节点加权求和近似积分 |

为什么 $n$ 点高斯求积能精确到 $2n-1$ 次多项式？

设 $q(x)$ 是次数不超过 $2n-1$ 的多项式。用 $n$ 阶正交多项式 $p_n(x)$ 做带余除法：

$$
q(x)=s(x)p_n(x)+r(x)
$$

其中 $s(x)$ 和 $r(x)$ 的次数都不超过 $n-1$。在高斯节点 $x_i$ 上，因为 $x_i$ 是 $p_n$ 的零点，所以：

$$
q(x_i)=r(x_i)
$$

求积公式只需要对 $r(x)$ 精确即可。另一方面，积分中的 $s(x)p_n(x)$ 因为 $s$ 的次数不超过 $n-1$，会和 $p_n$ 正交：

$$
\int_a^b s(x)p_n(x)w(x)\,dx=0
$$

所以积分也只剩 $r(x)$。这就解释了为什么高斯节点能带来 $2n-1$ 次精确性。

Gauss-Legendre 的权重还可以写成显式形式：

$$
\lambda_i=
\frac{2}{(1-x_i^2)\left(P_n'(x_i)\right)^2}
$$

这里 $P_n'(x_i)$ 是 $P_n$ 在节点 $x_i$ 处的导数。

Golub-Welsch 的伪代码如下：

```python
# 输入: 递推系数 a_n, b_n 和阶数 n
# 1. 构造对称三对角矩阵 J
# 2. 求 J 的特征值，作为节点 x_i
# 3. 取首个特征向量分量，计算权重 lambda_i
# 4. 返回 sum(lambda_i * f(x_i))
```

---

## 代码实现

实现 Gauss-Legendre 积分时，通常不手动求 Legendre 多项式零点，而是调用成熟库。原因很直接：求根和高阶递推都有数值稳定性问题，库函数已经处理了这些细节。

下面是一个可运行的 Python 示例，使用 NumPy 的 `leggauss`。`leggauss(n)` 返回 $[-1,1]$ 上 $n$ 点 Gauss-Legendre 节点和权重。

```python
import numpy as np
from numpy.polynomial.legendre import leggauss

def gauss_legendre_integrate(f, a, b, n=4):
    x, w = leggauss(n)              # [-1, 1] 上的节点和权重
    t = 0.5 * (b - a) * x + 0.5 * (a + b)
    return 0.5 * (b - a) * np.sum(w * f(t))

# 玩具例子：积分 x^2，真值为 2/3
ans1 = gauss_legendre_integrate(lambda x: x**2, -1.0, 1.0, n=2)
assert abs(ans1 - 2.0 / 3.0) < 1e-14

# 普通区间例子：积分 x^3，从 0 到 2，真值为 4
ans2 = gauss_legendre_integrate(lambda x: x**3, 0.0, 2.0, n=2)
assert abs(ans2 - 4.0) < 1e-14

# 非多项式例子：积分 exp(x)，从 0 到 1，真值为 e - 1
ans3 = gauss_legendre_integrate(np.exp, 0.0, 1.0, n=8)
assert abs(ans3 - (np.e - 1.0)) < 1e-12

print(ans1, ans2, ans3)
```

这段代码的核心不是 `np.sum`，而是区间映射：

$$
x=\frac{b-a}{2}t+\frac{a+b}{2}
$$

其中 $t$ 是标准区间 $[-1,1]$ 上的节点，$x$ 是原始区间 $[a,b]$ 上的节点。积分前面还要乘上雅可比因子 $\frac{b-a}{2}$。雅可比因子是变量变换后长度缩放带来的乘数。

如果希望理解库函数背后的算法，可以手写一个基于 Jacobi 矩阵的版本。下面代码只针对 Legendre 情形：

```python
import numpy as np

def gauss_legendre_golub_welsch(n):
    # Legendre 正交归一化递推对应的 Jacobi 矩阵
    k = np.arange(1, n, dtype=float)
    beta = k / np.sqrt(4 * k * k - 1)

    J = np.zeros((n, n), dtype=float)
    for i, b in enumerate(beta):
        J[i, i + 1] = b
        J[i + 1, i] = b

    eigenvalues, eigenvectors = np.linalg.eigh(J)
    weights = 2 * eigenvectors[0, :] ** 2
    return eigenvalues, weights

x, w = gauss_legendre_golub_welsch(2)
assert np.allclose(x, [-1 / np.sqrt(3), 1 / np.sqrt(3)])
assert np.allclose(w, [1.0, 1.0])

approx = np.sum(w * x**2)
assert abs(approx - 2.0 / 3.0) < 1e-14
```

| 实现路径 | 开发成本 | 稳定性 | 可解释性 | 适合场景 |
|---|---:|---:|---:|---|
| 直接调用库函数 | 低 | 高 | 中 | 生产代码、实验代码 |
| 构造三对角矩阵 | 中 | 较高 | 高 | 学习算法、定制权重 |
| 自己求多项式零点 | 高 | 低到中 | 中 | 教学演示，不建议生产使用 |

真实工程中，贝叶斯边际似然可以写成类似代码：对参数 $\theta$ 取节点，计算 $p(D\mid\theta)p(\theta)$，再加权求和。若参数区间为 $[\theta_{\min},\theta_{\max}]$，且先验和似然平滑，Gauss-Legendre 常常比随机采样更省函数调用次数。

---

## 工程权衡与常见坑

高斯积分的优势来自强假设：低维、函数足够光滑、积分区间明确、权重函数匹配。脱离这些条件后，它不一定比普通方法更好。

最常见的坑是区间不匹配。Gauss-Legendre 的标准节点在 $[-1,1]$，如果原始区间是 $[a,b]$，必须先做变量变换。如果积分区间是 $[0,\infty)$，不能直接把几个 Legendre 节点塞进去，应考虑 Laguerre 积分或其他变换。

第二个坑是权重不匹配。正交多项式是相对于权重函数定义的。Legendre 对应 $w(x)=1$，Hermite 对应 $e^{-x^2}$，Laguerre 对应 $e^{-x}$。如果原问题天然带权重，却强行使用不匹配的多项式族，高斯积分的高精度结论就不再成立。

第三个坑是维度过高。一维用 $n$ 个节点，二维张量积就是 $n^2$ 个节点，$d$ 维就是：

$$
N=n^d
$$

当 $n=20,d=10$ 时，节点数是 $20^{10}$，完全不可用。

```python
def tensor_grid_node_count(n, d):
    return n ** d

assert tensor_grid_node_count(10, 1) == 10
assert tensor_grid_node_count(10, 3) == 1000
assert tensor_grid_node_count(10, 8) == 100_000_000
```

第四个坑是多峰或尖峰后验。固定节点的高斯积分可能刚好没有覆盖到重要区域，导致积分结果严重偏差。贝叶斯推断里，如果后验质量集中在很窄的区域，直接用低阶规则经常不够。更稳的做法是先重参数化、分段积分，或者用自适应积分方法。

| 问题 | 表现 | 规避 |
|---|---|---|
| 区间不对 | 结果明显偏差 | 先做变量变换 |
| 权重不对 | 失去高斯精确性 | 换对应正交多项式族 |
| 高维张量积 | 节点数指数爆炸 | 改用稀疏网格、MC、quasi-MC |
| 多峰或尖峰 | 漏掉局部概率质量 | 自适应分段或重参数化 |
| 高阶递推 | 数值不稳 | 用成熟库实现 |
| 函数不光滑 | 收敛变慢 | 分段处理或换自适应方法 |

工程判断可以压缩成一句话：低维平滑积分优先考虑高斯求积；高维复杂分布优先考虑采样方法；局部不规则函数优先考虑自适应方法。

---

## 替代方案与适用边界

高斯积分不是通用最优解。它在低维、平滑、区间可控的问题上非常强，但在高维、非光滑、多峰问题上容易失效。

Monte Carlo 方法是用随机样本估计积分的方法。它的优势是对维度更鲁棒，缺点是收敛通常较慢，误差大致按 $O(N^{-1/2})$ 下降。quasi-Monte Carlo 使用低差异序列代替纯随机数。低差异序列是一类覆盖空间更均匀的确定性点集，常用于提高中高维积分效率。

自适应积分会根据函数局部行为自动加密节点。它适合有尖峰、断点或局部快速变化的函数，但实现复杂度更高，也更依赖具体库。

贝叶斯推断中还经常需要后验期望：

$$
\mathbb{E}[g(\theta)\mid D]
=
\frac{\int g(\theta)p(D\mid\theta)p(\theta)\,d\theta}
{\int p(D\mid\theta)p(\theta)\,d\theta}
$$

如果 $\theta$ 是一维超参数，且上下界明确，高斯积分通常很好用。例如在模型选择里，计算不同模型的边际似然 $Z$，再比较它们的大小。如果 $\theta$ 是几十维神经网络权重，高斯积分就不合适，应转向 MCMC、变分推断或其他近似方法。

| 方法 | 适合场景 | 优点 | 缺点 |
|---|---|---|---|
| 高斯求积 | 低维平滑积分 | 少量节点可得高精度 | 维度高时失效 |
| Monte Carlo | 高维复杂积分 | 对维度相对鲁棒 | 收敛慢、方差明显 |
| quasi-MC | 中高维平滑问题 | 比普通 MC 更省样本 | 依赖低差异序列和变量结构 |
| 自适应积分 | 局部尖峰、非光滑函数 | 能局部加密节点 | 实现更复杂 |
| 稀疏网格 | 中等维度、较平滑函数 | 缓解张量积爆炸 | 规则和实现更复杂 |

工程上常见的折中策略是：先做变量变换，把主要概率质量搬到规则区间；再对低维部分使用高斯积分；对剩余高维或复杂部分使用 Monte Carlo 或 quasi-MC。这样既保留高斯积分在低维上的效率，也避免在高维上遇到节点爆炸。

---

## 参考资料

| 资料类型 | 建议用途 |
|---|---|
| 理论基础 | 先理解正交性、递推关系和高斯精确性 |
| 数值实现 | 查看库函数如何生成节点和权重 |
| 算法论文 | 理解 Golub-Welsch 的矩阵算法 |
| 工程应用 | 看贝叶斯推断中如何使用数值积分 |

1. [DLMF §18.2 Orthogonal Polynomials](https://dlmf.nist.gov/18.2)
2. [DLMF §3.5 Gauss-Legendre Formula](https://dlmf.nist.gov/3.5)
3. [SciPy roots_legendre](https://docs.scipy.org/doc/scipy-1.9.1/reference/generated/scipy.special.roots_legendre.html)
4. [NumPy leggauss](https://numpy.org/doc/stable/reference/generated/numpy.polynomial.legendre.leggauss.html)
5. [Golub & Welsch 1969](https://www.ams.org/journals/mcom/1969-23-106/S0025-5718-69-99647-1/)
6. [Sampling for Inference in Probabilistic Models with Fast Bayesian Quadrature](https://papers.nips.cc/paper_files/paper/2014/hash/a0d08267a0fcee6970544a6d12286691-Abstract.html)
