---
title: L² 空间：内积、正交性与 Riesz–Fischer 理论
date: 2026-08-07
---

# L² 空间：内积、正交性与 Riesz–Fischer 理论

<div class="epigraph">
<p>在 $L^2$ 中，函数有了角度与正交——函数空间第一次拥有了完整的几何。</p>
<footer>—— 大卫 · 希尔伯特（David Hilbert）</footer>
</div>

<div class="article-byline">
<p>第二级 · 实变函数与测度论 ｜ 周民强《实变函数论》第七章 ｜ 2026-08-07</p>
</div>

## 为什么从 L² 空间开始

在 $L^p$ 家族中，$p=2$ 是独一无二的：**只有 $p=2$ 时范数来自内积**（极化恒等式），$L^2$ 因此成为 **Hilbert 空间**——有内积、有正交、有投影、有 Fourier 级数。$L^2$ 是量子力学的态空间、信号处理的能量空间、统计学的平方误差空间——现代科学计算与分析的共同语言。

本节把 $L^2$ 的 Hilbert 结构完整展开：内积、Cauchy–Schwarz、正交性、投影定理、以及正交基（Fourier 级数）理论。学懂 $L^2$，等于掌握了 Hilbert 空间这一现代数学最重要结构的全部入门内容。<span class="marginnote">$L^2$ 的「几何」威力来自<strong>极化恒等式与平行四边形法则</strong>：范数由内积导出 ⇔ 平行四边形法则成立。$p\neq2$ 时平行四边形法则崩坏（单位球不满足），所以只有 $L^2$ 有内积。<strong>「$L^2$ 是 Hilbert 空间」是 $p=2$ 的宿命</strong>——内积的几何只在此刻绽放。</span>

## 1 内积结构

**定义（$L^2$ 内积）**：$L^2(E)=\{f:\int|f|^2<\infty\}$，内积

$$\langle f,g\rangle=\int_Ef(x)\,\overline{g(x)}\,dm$$

（实值情形 $\overline{g}=g$。）范数 $\|f\|_2=\sqrt{\langle f,f\rangle}$。

**定理（内积公理）**：$\langle\cdot,\cdot\rangle$ 是 $L^2$ 上的内积：共轭对称、对第一变元线性、正定（$\langle f,f\rangle\ge0$，$=0\iff f=0$ a.e.）。

**Cauchy–Schwarz 不等式**（Hölder 的 $p=q=2$ 情形）：

$$|\langle f,g\rangle|\le\|f\|_2\|g\|_2$$

**平行四边形法则**：$\|f+g\|_2^2+\|f-g\|_2^2=2\|f\|_2^2+2\|g\|_2^2$——内积范数的特征性质。

**极化恒等式**：$\langle f,g\rangle=\tfrac14\left(\|f+g\|_2^2-\|f-g\|_2^2+i\|f+ig\|_2^2-i\|f-ig\|_2^2\right)$（复）——**范数还原内积**。

**重点：$L^2$ 的「几何」来自内积。** $\langle f,g\rangle$ 定义「函数之间的角度」：$\cos\theta=\tfrac{\langle f,g\rangle}{\|f\|\|g\|}$；$\langle f,g\rangle=0$ 时两函数「正交」（垂直）。**「函数正交」是 $L^2$ 独有的概念**——它是 Fourier 级数、正交分解的全部基础。

## 2 正交性与投影

**定义（正交）**：$f\perp g$（正交）⇔ $\langle f,g\rangle=0$。正交子空间 $M^\perp=\{f:\langle f,g\rangle=0\ \forall g\in M\}$。

**勾股定理（正交和）**：$f\perp g$ ⇒ $\|f+g\|_2^2=\|f\|_2^2+\|g\|_2^2$。

**定理（投影定理）**：设 $M\subset L^2$ 是闭子空间。则每个 $f\in L^2$ 唯一分解为

$$f=P_Mf+(f-P_Mf),\qquad P_Mf\in M,\ f-P_Mf\in M^\perp$$

其中 $P_Mf$ 是 $f$ 在 $M$ 上的**正交投影**，且 $\|f-P_Mf\|_2=\inf_{g\in M}\|f-g\|_2$（最佳逼近）。<span class="marginnote">投影定理是「<strong>闭子空间上的最佳逼近</strong>」：$P_Mf$ 是 $M$ 中离 $f$ 最近的元素（最小二乘！）。$M$ 的「闭」保证投影存在（完备性）。这是 <strong>最小二乘、回归分析、信号分解</strong>的数学内核——「从数据里提取最佳近似」就是正交投影。</span>

**证明要点**：取极小化序列 $g_k$ 使 $\|f-g_k\|_2\to d=\inf_M\|f-g\|$；由平行四边形法则证 $g_k$ 是 Cauchy（$L^2$ 完备）；极限 $g_0\in M$（闭），由变分证 $f-g_0\perp M$。**「极小化 → Cauchy → 完备 → 正交」**是投影定理的标准链。

**例（投影的数值算例）**：$L^2([0,1])$ 中取 $M=\mathrm{span}\{1,x\}$（一次多项式），$f(x)=x^2$。投影 $P_Mf$ 是 $a+bx$ 中使 $\int_0^1(x^2-a-bx)^2$ 最小者。由正交条件 $\langle f-P_Mf,1\rangle=0$、$\langle f-P_Mf,x\rangle=0$ 解出 $a=-\tfrac16,\ b=1$：$P_Mf=x-\tfrac16$，误差 $\|f-P_Mf\|_2^2=\tfrac1{180}$。**「两个正交条件、两个未知系数」**——投影定理在有限维子空间上退化成一次线性方程组的求解。

## 3 正交基与 Fourier 级数

**定义（标准正交系）**：$\{\varphi_k\}\subset L^2$ 满足 $\langle\varphi_i,\varphi_j\rangle=\delta_{ij}$（正交且范数 1），称为**标准正交系（ONS）**。若其张成空间在 $L^2$ 中稠密，称为**标准正交基（ONB）**。

**定理（Fourier 展开）**：设 $\{\varphi_k\}$ 是 $L^2$ 的 ONB，$f\in L^2$。则 $f$ 可唯一展开：

$$f=\sum_{k=1}^{\infty}\langle f,\varphi_k\rangle\varphi_k$$

级数在 $L^2$ 范数下收敛，且 **Parseval 恒等式**成立：

$$\|f\|_2^2=\sum_{k=1}^{\infty}|\langle f,\varphi_k\rangle|^2$$

**例（三角基）**：$\{\tfrac1{\sqrt{2\pi}}e^{ikx}\}_{k\in\mathbb{Z}}$ 是 $L^2([0,2\pi])$ 的 ONB——Fourier 级数的 $L^2$ 理论。**「$L^2$ 中每个函数都有 Fourier 展开」**是 $L^2$ 理论的巅峰应用。

**重点：ONB 理论把「函数」翻译成「系数序列」。** $f\leftrightarrow(\langle f,\varphi_k\rangle)_{k=1}^\infty$ 是 $L^2\to\ell^2$ 的等距同构（Parseval：范数保持）。**「函数空间 ≈ 序列空间」**——这是 Fourier 分析的实质：把函数论化为数列论。

**$L^2$ 与 $\ell^2$ 对照表**（等距同构下的「词典」）：

| $L^2$ 对象 | 对应 $\ell^2$ 对象 |
| --- | --- |
| 函数 $f$ | 系数序列 $(\langle f,\varphi_k\rangle)$ |
| 内积 $\langle f,g\rangle$ | $\sum_k a_k\overline{b_k}$ |
| 范数 $\|f\|_2$ | $\sqrt{\sum_k|a_k|^2}$（Parseval） |
| 正交 $f\perp g$ | $\sum_k a_k\overline{b_k}=0$ |
| Fourier 部分和 | 序列截断 $(a_1,\dots,a_N,0,\dots)$ |

**一句话**：<strong>「函数即序列、内积即点积、能量即模长平方」</strong>——这一对应让有限维线性代数里的几何直觉全部迁移到函数空间。

## 4 公式解析：Parseval 恒等式的证明

Parseval 恒等式是 $L^2$ 几何的签名公式，拆开证明：

$$\left\|f-\sum_{k=1}^{N}\langle f,\varphi_k\rangle\varphi_k\right\|_2^2=\|f\|_2^2-\sum_{k=1}^{N}|\langle f,\varphi_k\rangle|^2$$

- **第一步，读「正交投影的误差」**：$P_Nf=\sum_{k=1}^N\langle f,\varphi_k\rangle\varphi_k$ 是 $f$ 到 $\text{span}\{\varphi_1,\dots,\varphi_N\}$ 的正交投影。**「$f$ 减投影」正交于投影空间**，勾股定理适用。
- **第二步，读「勾股展开」**：$\|f-P_Nf\|^2=\|f\|^2-\|P_Nf\|^2$（正交分解 + 勾股）。而 $\|P_Nf\|^2=\sum_{k=1}^N|\langle f,\varphi_k\rangle|^2$（正交投影的范数 = 系数平方和，Bessel 不等式取等）。**「投影范数 = 系数能量」**——ONB 的正交性让交叉项全部消失。
- **第三步，读「$N\to\infty$」**：$\{\varphi_k\}$ 是 ONB（张成稠密），$P_Nf\to f$ 在 $L^2$（投影逼近收敛），左边 $\to0$，得 $\|f\|^2=\sum|\langle f,\varphi_k\rangle|^2$。**「稠密 + 投影 = 完备性」**——Parseval 是 ONB 定义的等价形式。

**「正交投影 + 勾股 + 稠密极限」**，是 Parseval 的完整证明——也是整个 $L^2$ 几何的浓缩。

## 5 例子：$L^2$ 几何的直观演练

**例一（正交分解的几何）**：设 $M=\text{span}\{1\}$（常函数）在 $L^2([0,1])$ 上。任意 $f\in L^2$ 的正交投影是常数 $P_Mf=\int_0^1f$（平均值）。验证正交：$f-\int f$ 与常数正交（$\int(f-\bar f)\cdot c=c(\int f-\bar f)=0$）。**「投影 = 平均值」**——$L^2$ 的最小二乘逼近把 $f$ 的最佳常函数近似取为它的均值，这正是「用常数估计函数」的最小二乘解。

**例二（Fourier 系数的最小二乘）**：$f\in L^2([0,2\pi])$，用三角函数 $\{e^{ikx}\}$ 的前 $N$ 项逼近，最优系数恰是 Fourier 系数 $\hat f(k)=\tfrac1{2\pi}\int f e^{-ikx}$（正交投影的系数公式）。**「最优三角逼近的系数 = Fourier 系数」**是投影定理对 Fourier 分析的直接馈赠——它解释了为何 Fourier 级数的部分和是 $L^2$ 意义下的最佳逼近。

**例三（Parseval 的数值验证）**：$f(x)=x$ 在 $[0,2\pi]$ 上（归一化后）。Fourier 系数 $\hat f(k)=\tfrac{1}{2\pi}\int_0^{2\pi}xe^{-ikx}dx=\tfrac{i}{k}$（$k\neq0$），$\hat f(0)=\pi$。Parseval：$\|f\|_2^2=\tfrac{1}{2\pi}\int_0^{2\pi}x^2dx=\tfrac{4\pi^2}{3}$，而 $\sum_k|\hat f(k)|^2=\pi^2+2\sum_{k=1}^\infty\tfrac1{k^2}$——由此反解出 **Basel 公式 $\sum_{k=1}^\infty\tfrac1{k^2}=\tfrac{\pi^2}{6}$**！**一条 Parseval 恒等式直接证明著名的 Basel 问题**——$L^2$ 几何的能量等价原理威力可见一斑。

**例四（量子力学态空间）**：$L^2(\mathbb{R}^3)$ 是单粒子量子态的空间，可观测量的期望值写作 $\langle\hat A\rangle=\int\bar\psi\,\hat A\psi$；波函数的归一化 $\|\psi\|_2=1$ 正是 $L^2$ 范数约束。**「概率幅必须是 $L^2$ 可积的」**是量子力学对波函数的全部正则约束，$L^2$ 理论因此直接支撑量子力学的数学框架（见《泛函分析》与量子力学专题）。

**重点：$L^2$ 的三个例子展示同一原理——「正交投影 + Parseval」把函数问题化为数列问题。** 最佳逼近（例一、二）与能量守恒（例三）是 $L^2$ 几何的两大产出，它们构成了 Fourier 分析与最小二乘的全部基础。

## 6 小结

- **内积**：$\langle f,g\rangle=\int f\overline g$；Cauchy–Schwarz；极化恒等式还原范数。
- **正交与投影**：$f\perp g$、投影定理（闭子空间最佳逼近）、勾股定理。
- **ONB 与 Fourier**：$f=\sum\langle f,\varphi_k\rangle\varphi_k$，Parseval $\|f\|^2=\sum|\langle f,\varphi_k\rangle|^2$。
- **等距同构**：$L^2\cong\ell^2$（函数 ↔ 系数序列）。
- **应用**：最小二乘、Fourier 分析、量子力学态空间。
- **投影算例**：$M=\mathrm{span}\{1,x\}$ 上 $P(x^2)=x-\tfrac16$——投影退化为解线性方程组。
- **等距词典**：函数↔序列、内积↔点积、能量↔模长平方（$L^2\cong\ell^2$）。

在下一节，我们完成 $L^p$ 谱系的最后一站：**$L^\infty$ 空间与本性有界函数**。
