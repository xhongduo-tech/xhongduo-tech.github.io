---
title: Chebyshev 交错定理与最小零偏差多项式
date: 2026-08-07
---

# Chebyshev 交错定理与最小零偏差多项式

<div class="epigraph">
<p>误差应当在尽可能多的点上，以尽可能大的幅度，交替地上下跳动——这就是最好的逼近。</p>
<footer>—— 帕夫努季 · 切比雪夫（Pafnuty Chebyshev, 1854）</footer>
</div>

<div class="article-byline">
<p>第二级 · 函数逼近论 ｜ E. Ward Cheney, Introduction to Approximation Theory, §2.3 ｜ 2026-08-07</p>
</div>

## 为什么从 Chebyshev 交错定理开始

上一篇证明了最佳一致逼近存在且唯一，但把最关键的一步「借」给了交错定理。这一篇把它还上：**最佳逼近的误差曲线长什么样？** 答案是惊人的简洁——误差在 $n+2$ 个点上交替达到最大绝对值。这条**等振荡（equioscillation）**特征不仅是一个漂亮的定理，还直接给出两个礼物：一是著名的 **Chebyshev 多项式**，它是在所有首一 $n$ 次多项式中「最平坦」的那个；二是构造最佳逼近的 **Remez 交换算法**。理论、特例、算法在这一篇汇合。Chebyshev 多项式同时也是信号处理与滤波器设计（第六级《信号与系统》）背后的数学引擎。

## 1 Chebyshev 交错定理

**Chebyshev 交错定理（alternation theorem）**：设 $f \in C[a,b]$，$p^* \in P_n$。则 $p^*$ 是 $f$ 的最佳一致逼近，当且仅当存在 $n+2$ 个点

$$
a \le x_0 \lt  x_1 \lt  \cdots \lt  x_{n+1} \le b
$$

使得误差 $f - p^*$ 在这些点上**交替变号地达到最大绝对值**：

$$
|f(x_k) - p^*(x_k)| = \|f - p^*\|_\infty, \qquad \operatorname{sgn}(f(x_{k+1}) - p^*(x_{k+1})) = -\operatorname{sgn}(f(x_k) - p^*(x_k))
$$

这 $n+2$ 个点称为**交错点（alternation points）**。<span class="marginnote">为什么是 $n+2$ 个？$P_n$ 的维数是 $n+1$，「用 $n+1$ 个自由度去拟合」要求误差曲线至少有 $n+2$ 个「极点」来钉住它——多出来的一个正是「最佳性」的余额。对应到上一篇唯一性证明：非零多项式在 $n+2$ 个点上取零，就被迫恒为零。</span>

定理包含两个方向：**必要性**说「最佳 ⇒ 交错」，**充分性**说「交错 ⇒ 最佳」。充分性的证明很直观：若另一个 $q$ 比 $p^*$ 更好，则在交错点上 $f - q$ 必须与 $f - p^*$ 同号且更小，迫使 $q - p^*$ 在这些点上交替变号，从而有 $n+1$ 个零点，矛盾。必要性的证明要反过来用对偶论证（几何上等价于 Hahn–Banach 定理），是逼近论最深刻的结论之一。

## 2 Chebyshev 多项式：定义与显式公式

交错定理立即派生出一个重要对象。考虑 $[-1,1]$ 上「最小零偏差」问题：在所有首一（最高次项系数为 1）的 $n$ 次多项式中，谁的 $\|\cdot\|_\infty$ 最小？

答案由**第一类 Chebyshev 多项式（Chebyshev polynomial of the first kind）**给出：

$$
T_n(x) = \cos(n \arccos x), \qquad x \in [-1, 1]
$$

$T_n$ 确实是 $n$ 次多项式——这一点稍后验证。它满足 $|T_n(x)| \le 1$ 对一切 $x \in [-1,1]$，且在 $n+1$ 个点 $x_k = \cos(k\pi/n)$（$k = 0,\dots,n$）上交替取值 $\pm 1$。这恰好是「$n$ 次多项式 + 等振荡 $n+1$ 个点」的极小化特征（对首一情形，交错点数是 $n+1$，因为首一约束吃掉了一个自由度）。

**最小零偏差定理（minimal deviation theorem）**：首一 $n$ 次多项式 $x^n + \cdots$ 在 $[-1,1]$ 上的一致范数最小值为

$$
\min_{\text{monic } p \in P_n} \|p\|_\infty = \frac{1}{2^{n-1}}, \qquad \text{唯一极小元是 } \tilde{T}_n(x) = \frac{T_n(x)}{2^{n-1}}
$$

$T_n$ 的最高次项系数为 $2^{n-1}$，除以它就得首一多项式 $\tilde T_n$。$\tilde T_n$ 是「零偏差最小的首一多项式」：它在 $[-1,1]$ 上最平坦，任何首一多项式在区间上都至少达到范数 $2^{1-n}$。<span class="marginnote">为什么 $T_n$ 的最高次项系数是 $2^{n-1}$？由 $T_n(x) = \cos(n\theta), x = \cos\theta$ 和 $2\cos\theta = e^{i\theta}+e^{-i\theta}$，展开 $\cos(n\theta) = (e^{in\theta}+e^{-in\theta})/2 = ((x+\sqrt{x^2-1})^n + (x-\sqrt{x^2-1})^n)/2$，最高次项来自 $(x + \sqrt{x^2-1})^n$ 中 $x^n$ 的贡献 $2^{n-1}x^n$。</span>

## 3 Chebyshev 多项式的三个关键性质

- **三递推关系（three-term recurrence）**：由 $\cos((n+1)\theta) = 2\cos\theta\cos(n\theta) - \cos((n-1)\theta)$ 得

$$
T_{n+1}(x) = 2x\,T_n(x) - T_{n-1}(x), \qquad T_0 = 1, \; T_1 = x
$$

由此可逐次写出 $T_2 = 2x^2-1$，$T_3 = 4x^3 - 3x$，$T_4 = 8x^4 - 8x^2 + 1$，等等。这条递推把「三角函数恒等式」翻译成「多项式的代数结构」，也是后续数值实现的基础——计算 $T_n$ 不需要先算 $\arccos$。

- **离散正交性（discrete orthogonality）**：在节点 $x_k = \cos(k\pi/n)$ 上，$T_i, T_j$ 的离散内积满足

$$
\sum_{k=0}^{n} '' T_i(x_k) T_j(x_k) = 0 \quad (i \neq j)
$$

（带撇的求和表示首尾项减半。）这条性质让 Chebyshev 节点成为插值的最佳节点选择——它压制了 Runge 现象（第 5 篇详谈），也定义了 Chebyshev 插值的稳定基。

- **极值点与根**：$T_n$ 的 $n+1$ 个极值点是 $x_k = \cos(k\pi/n)$，$n$ 个根是 $x_k = \cos((2k-1)\pi/(2n))$。它们全部落在 $(-1,1)$ 内，且在 $(-1,1)$ 内稠密分布（向端点靠拢）——这正是「在边界处多布点」这一数值直觉的数学根源。

### 前几个 Chebyshev 多项式

| $n$ | $T_n(x)$ | 极值点 $x_k = \cos(k\pi/n)$ |
| --- | --- | --- |
| 0 | $1$ | — |
| 1 | $x$ | $1, -1$ |
| 2 | $2x^2 - 1$ | $1, 0, -1$ |
| 3 | $4x^3 - 3x$ | $1, \tfrac12, -\tfrac12, -1$ |
| 4 | $8x^4 - 8x^2 + 1$ | $1, \tfrac{\sqrt2}{2}, 0, -\tfrac{\sqrt2}{2}, -1$ |
| 5 | $16x^5 - 20x^3 + 5x$ | $1, \cos\frac{\pi}{5}, \cos\frac{2\pi}{5}, \cos\frac{3\pi}{5}, \cos\frac{4\pi}{5}, -1$ |

观察极值点一列：$T_2$ 在 $\{1, 0, -1\}$ 上交替取 $1, -1, 1$；$T_3$ 在 $\{1, \tfrac12, -\tfrac12, -1\}$ 上交替取 $1, -1, 1, -1$——**等振荡一目了然**。同时注意极值点与根都在端点附近更密、在中点附近更疏，这正是第 5 篇用 Chebyshev 节点压制 Runge 现象的几何根源。

## 4 公式解析：$T_n(x) = \cos(n\arccos x)$ 何以是最佳

**这条公式一行字就蕴含了整个最小零偏差定理。** 拆解它的四步：

- **第一步，先确认它是多项式**：设 $x = \cos\theta$，则 $T_n(\cos\theta) = \cos(n\theta)$。把 $e^{in\theta}$ 展开成 $(\cos\theta + i\sin\theta)^n$，取实部后 $\cos(n\theta)$ 是 $\cos\theta$ 的 $n$ 次多项式——故 $T_n$ 是 $x$ 的 $n$ 次多项式，且实系数。
- **第二步，看它的上界**：对一切 $\theta$，$|\cos(n\theta)| \le 1$，所以 $|T_n(x)| \le 1$ 在 $[-1,1]$ 上处处成立。范数 $\|T_n\|_\infty = 1$。
- **第三步，数极值点**：当 $\theta = k\pi/n$（$k=0,\dots,n$）时 $\cos(n\theta) = \pm 1$，于是 $T_n$ 在 $n+1$ 个点 $x_k = \cos(k\pi/n)$ 上交替取 $\pm 1$。这是完美的等振荡。
- **第四步，用交错定理收尾**：假若存在首一 $q$ 使 $\|q\|_\infty \lt  \| \tilde T_n\|_\infty = 2^{1-n}$，则 $q - \tilde T_n$ 是次数 $\le n-1$ 的多项式，它在 $n+1$ 个交错点 $x_k$ 上必须与 $\tilde T_n$ 同号（因为 $q$ 比 $\tilde T_n$ 更小），故 $q - \tilde T_n$ 在 $n+1$ 个点上交替变号、至少 $n$ 个零点，次数却只有 $n-1$——矛盾。

一行三角函数公式，引出了「最平坦多项式」的完整证明。这种「用恒等式约束形状」的思维方式，在后面的 Fourier 逼近（三角函数系的正交性）里还会再次登场。

## 5 Remez 交换算法

交错定理不只是判据，还给出算法。**Remez 交换算法（Remez exchange algorithm）** 求 $f \in C[a,b]$ 在 $P_n$ 中的最佳一致逼近：

- **初始化**：选 $n+2$ 个初始交错候选点，例如等距分布或 Chebyshev 节点。
- **求解线性系统**：在当前候选点上，令误差在 $n+2$ 个点上交替取 $\pm E$（$E$ 未知），得到 $n+2$ 个未知数（$p^*$ 的 $n+1$ 个系数 + $E$）的线性方程组，解出试探多项式 $p$ 与试探偏差 $E$。
- **交换**：找 $|f - p|$ 的全局最大值点（通常在候选点之外），用它替换一个候选点，重复求解。

每一步都使 $E$ 单调上升，且由于唯一性，迭代收敛到真解。<span class="marginnote">Remez 算法是「理论上收敛」的典型代表：每步都改善，且交错定理保证没有更好解。它在实践中通常几步就收敛，但实现难点在第二步——需要可靠的全局最大值搜寻器，工程上常配合三分搜索与分段细化。</span> 历史上，Remez 算法与 Chebyshev 多项式一起，是滤波器设计、函数库实现（如标准库的 $\sin$、$\exp$）的理论支柱。

## 6 术语速查表

| 术语 | 英文 | 一句话定义 |
| --- | --- | --- |
| 交错定理 | alternation theorem | 最佳逼近 ⟺ 误差在 $n+2$ 个点上交替取极值 |
| 等振荡 | equioscillation | 误差在所有交错点取得相同的最大绝对值 |
| Chebyshev 多项式 | Chebyshev polynomial | $T_n(x) = \cos(n\arccos x)$ |
| 最小零偏差 | minimal deviation | 首一 $n$ 次多项式在 $[-1,1]$ 上范数下界 $2^{1-n}$ |
| 三递推关系 | three-term recurrence | $T_{n+1}(x) = 2xT_n(x) - T_{n-1}(x)$ |
| 离散正交性 | discrete orthogonality | Chebyshev 节点上的加权正交和为零 |
| Remez 交换算法 | Remez exchange algorithm | 迭代猜测并交换候选点求最佳逼近 |
| 首一多项式 | monic polynomial | 最高次项系数为 1 的多项式 |
| 极点 | extremal point | $T_n$ 交替取 $\pm 1$ 的点 $x_k=\cos(k\pi/n)$ |

## 7 小结

- **Chebyshev 交错定理**：$p^*$ 最佳 $\iff$ 误差 $f - p^*$ 在 $n+2$ 个点上交替达到最大绝对值。
- Chebyshev 多项式 $T_n(x) = \cos(n\arccos x)$ 是首一多项式中一致范数最小的元，最小值为 $2^{1-n}$，唯一极小元为 $T_n/2^{n-1}$。
- 三递推 $T_{n+1} = 2xT_n - T_{n-1}$、离散正交性、极值点 $x_k = \cos(k\pi/n)$