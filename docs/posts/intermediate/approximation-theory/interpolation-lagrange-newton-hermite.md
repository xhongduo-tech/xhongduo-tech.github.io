---
title: 插值方法：Lagrange、Newton 与 Hermite 插值
date: 2026-08-07
---

# 插值方法：Lagrange、Newton 与 Hermite 插值

<div class="epigraph">
<p>插值是从离散的观测恢复连续世界的古老手艺——但手艺的好坏，取决于你选哪些点、怎么问问题。</p>
<footer>—— 艾萨克 · 牛顿（Isaac Newton, 差分法 1687）</footer>
</div>

<div class="article-byline">
<p>第二级 · 函数逼近论 ｜ E. Ward Cheney, Introduction to Approximation Theory, §3.1–3.3 ｜ 2026-08-07</p>
</div>

## 为什么从插值开始

前面几篇研究的「最佳逼近」需要求解极值问题，代价较高。工程上更常见的需求是：手里只有一组离散数据——温度计在整点报数、传感器在采样时刻读数——却要猜出中间的连续行为。**插值（interpolation）** 就是这门手艺：构造一个简单函数，让它在给定的节点上「严丝合缝」地穿过数据。它是逼近论里最古老、最直觉的分支，也是微分方程数值解、数值积分、图像放缩、计算机图形学里曲线拟合的地基。这一篇讲清三种主力方法：Lagrange、Newton 与 Hermite，并交代它们共同的弱点——Runge 现象。插值同时是数值积分、微分方程数值解（第六级《数值分析》）与计算机图形学（第七级）曲线绘制的地基。

## 1 插值问题的提法

**多项式插值问题**：给定 $n+1$ 个互异的节点 $x_0 \lt  x_1 \lt  \cdots \lt  x_n$ 与对应的函数值 $y_0, y_1, \dots, y_n$，求一个次数不超过 $n$ 的多项式 $p \in P_n$，满足

$$
p(x_i) = y_i, \qquad i = 0, 1, \dots, n
$$

**存在且唯一**。存在性可由 Lagrange 构造直接给出；唯一性很干净：若 $p, q \in P_n$ 都在 $n+1$ 个节点上取相同的值，则 $p - q$ 是次数 $\le n$ 的多项式却有 $n+1$ 个零点，只能恒为零。<span class="marginnote">注意这里的「唯一性」与最佳逼近的唯一性气质不同：插值唯一性是代数的（多项式零点个数），最佳逼近唯一性是几何的（Haar 条件 + 凸性）。前者只要求节点互异，连节点怎么选都不用管。</span>

通常取 $y_i = f(x_i)$（$f$ 是被插函数），这时称 $p$ 为 $f$ 关于这些节点的**插值多项式**。问题看似简单，但「用什么形式表示 $p$」大有讲究——不同的表示对应不同的计算代价与数值稳定性。

## 2 Lagrange 插值

**Lagrange 插值**给出最直接的构造。定义第 $i$ 个 **Lagrange 基多项式**：

$$
\ell_i(x) = \prod_{\substack{j=0 \\ j \neq i}}^{n} \frac{x - x_j}{x_i - x_j}
$$

$\ell_i$ 是 $n$ 次多项式，且满足 $\ell_i(x_k) = \delta_{ik}$（在 $x_i$ 处取值 1，其余节点处取值 0）。于是

$$
p(x) = \sum_{i=0}^{n} y_i\, \ell_i(x)
$$

这就是**Lagrange 插值公式**。验证只需代入 $x_k$：求和中除第 $k$ 项外全部消失，剩下 $y_k \ell_k(x_k) = y_k$。<span class="marginnote">「每个节点贡献一个只在它那里为非零的基函数」是插值（乃至有限元、样条）一切基函数设计的母题。$\ell_i$ 在 $x_i$ 处为 1、他处为 0 的性质，叫<strong>克罗内克性（cardinality）</strong>。</span>

Lagrange 形式理论价值高、构造直观，但有两个缺点：一是每加入一个新节点就要推倒重算全部 $\ell_i$；二是当节点很多时，直接求和容易因大数相消而损失精度。这些痛点由 Newton 形式解决。

## 3 Newton 差商插值

**Newton 插值**把插值多项式写成嵌套形式：

$$
p(x) = c_0 + c_1(x-x_0) + c_2(x-x_0)(x-x_1) + \cdots + c_n (x-x_0)\cdots(x-x_{n-1})
$$

待定系数 $c_k$ 是函数值的**差商（divided differences）**。定义**零阶差商** $f[x_i] = f(x_i)$，一阶 $f[x_i, x_{i+1}] = \frac{f(x_{i+1}) - f(x_i)}{x_{i+1} - x_i}$，一般地

$$
f[x_i, \dots, x_{i+k}] = \frac{f[x_{i+1}, \dots, x_{i+k}] - f[x_i, \dots, x_{i+k-1}]}{x_{i+k} - x_i}
$$

则 $c_k = f[x_0, \dots, x_k]$。差商可以递推计算，填一张三角形差分表即可。<span class="marginnote">差商是「离散版本的导数」：一阶差商是割线斜率，二阶差商是割线斜率的斜率，$k$ 阶差商逼近 $f^{(k)}/k!$。后面 Hermite 插值处理重节点时，这一直觉会派上大用场。</span>

Newton 形式的杀手锏是**增量性（incremental）**：已经算了前 $n+1$ 个节点，再添一个 $x_{n+1}$，只需多算一列差商 $c_{n+1}$，前面 $c_0,\dots,c_n$ 原封不动。这使 Newton 插值在「逐步加点」的自适应场景中几乎不可替代。

## 4 Hermite 插值

插值不一定要「只给函数值」。有时我们同时知道函数值和导数值，比如轨迹的起点位置与速度——要求插值多项式同时匹配两者。**Hermite 插值（osculatory interpolation）** 解决的就是这种「重节点」问题。

最简单的两点 Hermite 插值：给定 $x_0, x_1$ 及 $f(x_0), f(x_1), f'(x_0), f'(x_1)$ 四个数据，求三次多项式 $p$ 满足 $p(x_i) = f(x_i)$ 且 $p'(x_i) = f'(x_i)$。用基函数写：

$$
p(x) = \sum_{i=0}^{1} \left[ y_i H_i(x) + y'_i \hat H_i(x) \right]
$$

其中 $H_i, \hat H_i$ 是**Hermite 基多项式**：$H_i(x_k) = \delta_{ik}$、$H_i'(x_k) = 0$，而 $\hat H_i(x_k) = 0$、$\hat H_i'(x_k) = \delta_{ik}$。每个数据点由专属基函数「背着」，互不干扰。<span class="marginnote">Hermite 插值在 Newton 框架下就是「重节点差商」：把 $x_i$ 重复出现两次，把 $f'[x_i,x_i]$ 理解为 $f'(x_i)$。重节点差商表让 Hermite 插值在实现上几乎免费——这也解释了为什么实际代码里 Hermite 通常不是独立算法，而是 Newton 差商的特例。</span>

更一般地，若在 $x_i$ 处给定 $f$ 直到 $m_i$ 阶的导数值，且 $\sum (m_i+1) = N+1$，则存在唯一次数 $\le N$ 的多项式匹配全部数据——这就是 Hermite 插值的完整形式，属于「广义 Hermite 插值」。

## 5 公式解析：插值余项

**插值多项式与真实函数之差，由插值节点的「几何分布」和一个高阶导数共同决定。** 设 $f \in C^{n+1}[a,b]$，$p$ 是 $f$ 在 $n+1$ 个互异节点 $x_0,\dots,x_n$ 上的插值多项式，则对每个 $x$ 存在 $\xi_x$ 落在 $x$ 与所有节点围成的最小闭区间内，使

$$
f(x) - p(x) = \frac{f^{(n+1)}(\xi_x)}{(n+1)!} \prod_{i=0}^{n} (x - x_i)
$$

拆解三步：

- **第一步，读出节点的角色**：乘积 $\omega(x) = \prod_{i=0}^{n} (x - x_i)$ 叫**节点多项式**。它在每个节点处取零——所以插值在节点上误差为零，天经地义；而节点之间的误差，正比于「$x$ 离所有节点有多远」的乘积。
- **第二步，读出导数项**：$f^{(n+1)}(\xi_x)/(n+1)!$ 与 Taylor 余项同源——插值余项本质上是「$n+1$ 阶导数在某个中间点取值」的 Lagrange 型余项。若 $f$ 是 $n$ 次多项式，$f^{(n+1)} \equiv 0$，插值多项式就是 $f$ 本身，误差为零。
- **第三步，读出可控与不可控**：$f^{(n+1)}$ 由被插函数决定，不可选；而 $\omega(x)$ 由**节点的选择**决定，完全可控。这就是 Runge 现象的钥匙：误差大小 ≈ 「导数项」×「节点多项式」。

## 6 Runge 现象：插值翻车现场

1885 年，Runge 发现一个反例。取

$$
f(x) = \frac{1}{1 + 25x^2}, \qquad x \in [-1, 1]
$$

在**等距节点**上做 $n$ 次插值，$n$ 增大时插值多项式在区间中段收敛，但在靠近端点 $|x| \approx 0.72$ 附近剧烈振荡，误差趋于发散。<span class="marginnote">Runge 现象的第一个教训：高次多项式插值 ≠ 更好的逼近。节点越多、次数越高，等距插值在边界附近反而越糟——这与直觉完全相反。它直接推动了 20 世纪逼近论对「节点选择」的深刻研究。</span>

罪魁正是余项里的节点多项式 $\omega(x)$：等距节点让 $\omega(x)$ 在区间两端剧烈膨胀，$|f^{(n+1)}(\xi)|$ 也在增长，两者相乘压过了「$1/(n+1)!$」的收缩。

**解药**来自上一篇：把节点选在 Chebyshev 节点 $x_k = \cos\left(\frac{2k+1}{2(n+1)}\pi\right)$（或极值点型 $x_k = \cos(k\pi/n)$）。此时 $\omega(x)$ 在 $[-1,1]$ 上被 Chebyshev 多项式控制——最小零偏差性质保证了 $\|\omega\|_\infty$ 最小化——Runge 振荡被系统性压制，插值随 $n$ 增大一致收敛（对满足适当光滑性的 $f$）。<span class="marginnote">Chebyshev 节点是「把更多点布在区间两端」的数学化：极值点 $x_k = \cos(k\pi/n)$ 在端点附近稠密、在中点附近稀疏。这是 Runge 现象的最优应对，也是谱方法与 Chebfun 这类现代工具的理论基石（第 10 篇展开）。</span>

## 7 术语速查表

| 术语 | 英文 | 一句话定义 |
| --- | --- | --- |
| 插值 | interpolation | 求多项式 $p \in P_n$ 使 $p(x_i) = y_i$ |
| Lagrange 基多项式 | Lagrange basis | $\ell_i(x)=\prod_{j\neq i}\frac{x-x_j}{x_i-x_j}$，满足克罗内克性 |
| 克罗内克性 | cardinality | 基函数在自身节点取 1、其余节点取 0 |
| 差商 | divided differences | $f[x_i,\dots,x_k]$，离散版本的导数 |
| 嵌套形式 | nested form | Newton 插值的 Horner 式递推写法 |
| Hermite 插值 | Hermite interpolation | 同时匹配函数值与导数值的插值 |
| 重节点 | confluent nodes | 节点重复出现，差值表对应导数 |
| 插值余项 | interpolation remainder | $\frac{f^{(n+1)}(\xi)}{(n+1)!}\prod_{i=0}^n(x-x_i)$ |
| 节点多项式 | nodal polynomial | $\omega(x)=\prod_{i=0}^n(x-x_i)$ |
| Runge 现象 | Runge phenomenon | 等距高次插值在边界附近振荡发散 |
| Chebyshev 节点 | Chebyshev nodes | $x_k=\cos((2k-1)\pi/(2n))$，压制 Runge 现象 |

## 8 小结

- 插值问题：求 $p \in P_n$ 使 $p(x_i) = y_i$；**存在且唯一**，与节点分布无关。
- **Lagrange 形式**：$p = \sum y_i \ell_i$，基函数满足克罗内克性，直观但难以增量更新。
- **Newton 形式**：嵌套多项式 + 差商表，**增量加点不推倒重算**，Hermite 通过重节点差商统一进来。
- **Hermite 插值**：同时匹配函数值与导数值，属于「重节点」广义插值。
- 插值余项 $f - p = \frac{f^{(n+1)}(\xi)}{(n+1)!}\prod(x-x_i)$