---
title: Riemann 假设与零点分布
date: 2026-08-11
---

# Riemann 假设与零点分布

<div class="epigraph">
<p>很可能，所有这些根都是实的。当然，人们愿意有一个严格的证明……</p>
<footer>—— 伯恩哈德 · 黎曼（Bernhard Riemann，《论小于给定值的素数个数》, 1859）</footer>
</div>

<div class="article-byline">
<p>第二级 · 进阶数理 · 解析数论 ｜ 对标教材 ｜ 2026-08-11</p>
</div>

## 为什么这是「当代最重要的数学问题」

前几篇我们反复撞见同一堵墙：$\zeta$ 的非平凡零点到底在哪里。零区域理论只告诉我们「零点不会太靠近 $\sigma=1$」，却对临界带内部无能为力。黎曼在 1859 年那篇七页的论文里做出了断言：**所有非平凡零点都落在临界线 $\mathrm{Re}\,s = 1/2$ 上**。这就是**黎曼假设（Riemann Hypothesis, RH）**，希尔伯特问题第 8 个、克雷数学研究所千禧年七题之一，一百六十多年来无人能证，也无人能推翻。

但这一节我们要讲的不只是「一个待证猜想」，而是它**周围已经被证实的结构**：零点如何对称、如何计数、如何与素数误差项挂钩，以及它们与物理（随机矩阵）之间令人瞠目的统计联系。

## 1 非平凡零点的对称结构

回顾第三篇：函数方程把 $s$ 与 $1-s$ 联系起来。加上「$\zeta(\bar{s}) = \overline{\zeta(s)}$」（共轭对称），零点的几何被两条对称轴钉死：

- **临界线** $\mathrm{Re}\,s = 1/2$：函数方程保证零点关于它对称（$s \leftrightarrow 1-s$）；
- **实轴**：共轭保证零点关于它对称（$s \leftrightarrow \bar{s}$）。

于是每个非平凡零点 $\rho = \beta + i\gamma$ 自动带出 $\bar{\rho}$ 与 $1-\rho$。**非平凡零点全部位于临界带 $0 < \sigma < 1$ 内**——这在 1893 年已由 Hadamard 证明（结合函数方程与 $\sigma>1$ 无零点）；平凡零点 $-2, -4, \ldots$ 则在负实轴上。<span class="marginnote"><strong>辨析｜易错点：</strong> 「非平凡」的确切含义是「在临界带 $0<\sigma<1$ 内的零点」。它们有无穷多个，且关于临界线与实轴对称分布；RH 只是进一步把它们的实部全部钉在 $1/2$。把「非平凡零点全在临界带上」当成 RH 是常见的误解——RH 说的是更精确的「在临界<strong>线</strong>上」。</span>

![非平凡零点在临界带与临界线上的分布示意](/images/analytic-number-theory/riemann-hypothesis-and-zero-distribution-1.svg)

## 2 RH 的等价物：从零点到素数的快车道

RH 之所以重要，是因为它等价于一串「最好的数论结论」。最著名的几个：

$$
\text{RH} \;\Longleftrightarrow\; \psi(x) = x + O(\sqrt{x}\log^2 x)
\;\Longleftrightarrow\;
\pi(x) = \mathrm{Li}(x) + O(\sqrt{x}\log x)
$$

还等价于 Mertens 函数 $M(x) = \sum_{n \le x}\mu(n) = O(x^{1/2 + \varepsilon})$ 对任意 $\varepsilon>0$ 成立，以及素数在等差数列中的最优均匀性（对几乎所有 $q$）。<span class="marginnote">把第四篇误差项推导里的零区域换成整个半平面 $\sigma \ge 1/2$，Perron 积分线就能挪到 $\sigma = 1/2$，于是主项 $x$ 之外的所有贡献都是 $O(x^{1/2 + \varepsilon})$。RH ⟹ 最优误差，就是这么机械——难的是反方向。</span>

**重点：RH 不是「一个孤立的复分析命题」，而是「素数分布最优精度」这一族命题的公共骨架**。它给不出的只是 $\sqrt{x}$ 那层对数因子（$\log^2 x$ 的来源是估计零点个数所需的高次对数）。

## 3 零点计数：Riemann–von Mangoldt 公式

除了零点位置，还要知道**零点有多少**。对临界带内虚部 $\le T$ 的非平凡零点个数 $N(T)$，有 **Riemann–von Mangoldt 公式**：

$$
N(T) = \frac{T}{2\pi}\log\frac{T}{2\pi} - \frac{T}{2\pi} + O(\log T)
$$

这个公式在 1905 年由 von Mangoldt 用 $\xi$ 函数的幅角原理（argument principle）严格证明。<span class="marginnote">幅角原理来自复分析：绕一大矩形对 $\xi'/\xi$ 积分，转一圈的整数倍就是零点个数。$\xi$ 是整函数、在矩形边界上界已知，于是零点个数被写成 $\log T$ 量级的误差项——这是「分析控制离散个数」的又一实例。</span>

**辨析｜易错点：** $N(T) \sim \frac{T}{2\pi}\log T$ 意味着零点的**平均竖直间距约是 $2\pi/\log T$**——也就是随高度越来越密（因为 $\log T$ 在增长），但不会密到连成一片。这是零点统计里一切「间距猜测」的出发点。

## 4 公式解析：$N(T)$ 为什么长这个样

把 Riemann–von Mangoldt 公式的主项来源拆开：

$$
N(T) = \frac{T}{2\pi}\log\frac{T}{2\pi} - \frac{T}{2\pi} + O(\log T)
$$

- **第一步，用幅角原理**：$2\pi N(T) = \Delta_\gamma \arg \xi(s)$，即 $\xi$ 沿矩形边界转过的辐角总变化。矩形的水平边与竖直边贡献分开算。
- **第二步，竖直边（主项来源）**：在 $\sigma = 2$（或 $1+\epsilon$）上，$\xi$ 的相位由 $\Gamma$ 函数主导；$\Gamma$ 的 Stirling 公式 $\log\Gamma(\frac12 + it) = (\frac12 + it - \frac12)\log(it) - it + O(1)$ 展开后，相位变化给出 $\frac{T}{2\pi}\log\frac{T}{2\pi} - \frac{T}{2\pi}$——**这就是对数项与 $-T/2\pi$ 项的全部来历**。
- **第三步，其余边**：水平边与竖直边 $\sigma = 1/2$ 上的贡献被并进 $O(\log T)$。

**直觉**：$N(T)$ 的形状完全由 $\Gamma$ 函数（Stirling 公式）决定，$\zeta$ 自己只负责那些对数阶误差——这再次印证函数方程里 $\Gamma$ 才是「布局者」。

## 5 零点统计与随机矩阵：一个物理学的惊喜

单个零点难缠，但零点作为整体却服从一个惊人干净的统计规律。Montgomery（1973）的**对关联猜测（pair correlation conjecture）**：归一化后（让平均间距变成 1），两个零点间距落在 $[u, u+du]$ 的概率密度是

$$
1 - \left(\frac{\sin \pi u}{\pi u}\right)^2
$$

Dyson 认出这正是**高斯酉系（GUE）**随机矩阵特征值的对关联密度——物理学家在原子核能级里见到的同一分布！于是数论（$\zeta$ 的零点）与量子混沌、随机矩阵、乃至大矩阵理论在统计层面对上了暗号。<span class="marginnote"><strong>辨析｜易错点：</strong> 对关联猜测是比 RH 强得多的统计猜想，且已被海量数值验证，但<strong>不是定理</strong>。注意它与 RH 是两回事：RH 管「零点都在一条线上」，对关联管「线上零点怎么排队」。把「RH 已由数值验证」与「对关联已被证明」混为一谈都是错的。Montgomery 的贡献是把猜测从个别零点提升到整体统计。</span>

对随机矩阵与「硬谱」的类比，我们在大模型里并不陌生：attention 矩阵、协方差矩阵的特征值统计同样服从 Wigner 半圆律或 GUE 类分布——**「系统大且无序时，特征值趋于普适分布」这条经验规律，从原子核到 $\zeta$ 的零点一路通到高维数据**。<span class="marginnote">这是本博客「从极限到大模型」主线的一个漂亮交叉点：极限（$T \to \infty$ 的零点统计）、数学物理（GUE）、大模型（高维矩阵普适性）三者在统计普适性上汇合。也提醒我们：RH 的价值之一是它精确预言了一个普适统计量的期望。</span>

## 6 小结

- **黎曼假设（RH）**：所有非平凡零点满足 $\mathrm{Re}\,s = 1/2$；关于临界线与实轴双对称，且全部落在临界带 $0<\sigma<1$。
- **RH 的等价物**：$\psi(x) = x + O(\sqrt{x}\log^2 x)$、$\pi(x) = \mathrm{Li}(x) + O(\sqrt{x}\log x)$、$M(x) = O(x^{1/2+\varepsilon})$ 等族——误差项被压到最优。
- **零点计数**：$N(T) = \frac{T}{2\pi}\log\frac{T}{2\pi} - \frac{T}{2\pi} + O(\log T)$，来自幅角原理 + $\Gamma$ 的 Stirling 公式。
- **对关联猜测**（Montgomery, 1973）：零点间距服从 GUE 密度 $1 - (\sin\pi u/\pi u)^2$，与随机矩阵特征值同分布——仍是猜想，但数值铁证如山。

到这里，「第一根主线：素数分布」已经走到了最深处。下一节起我们转向**第二根主线：算术函数的均值**，从特征函数与均值估计讲起。
