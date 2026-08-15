---
title: Lebesgue 控制收敛定理
date: 2026-08-07
---

# Lebesgue 控制收敛定理

<div class="epigraph">
<p>只要函数列被一把可积的大伞罩住，极限与积分就可以自由交换——这是分析学最可靠的工具之一。</p>
<footer>—— 亨利 · 勒贝格（Henri Lebesgue）</footer>
</div>

<div class="article-byline">
<p>第二级 · 实变函数与测度论 ｜ 周民强《实变函数论》第五章 ｜ 2026-08-07</p>
</div>

## 为什么从控制收敛定理开始

三大极限定理的收官之作——**Lebesgue 控制收敛定理（DCT）**——是分析学最常用、最强有力的「交换 $\lim$ 与 $\int$」工具。它的条件直观而实用：**只要 $f_k\to f$ a.e.，且存在可积函数 $g$ 罩住一切 $|f_k|\le g$（控制函数），则 $\lim\int f_k=\int f$。** 这把「极限交换」的门槛从「一致收敛」（Riemann 世界的苛求）降到「可积控制」（Lebesgue 世界的常态）。

DCT 是数学物理、概率论、调和分析的通用引擎：傅里叶级数逐项积分、特征函数的连续性、参数积分的可导性……全部靠它。学懂 DCT，等于掌握了现代分析中「换序」这一动作的万能钥匙。<span class="marginnote">DCT 的概率论版本：若 $X_n\to X$ a.s. 且 $|X_n|\le Y\in L^1$，则 $E[X_n]\to E[X]$。<strong>「被可积随机变量控制」是几乎必然收敛下期望连续的充分条件</strong>——在鞅论、停时、遍历论中，这是最常用的「取期望安全通过极限」的工具。</span>

## 1 定理陈述

**定理（Lebesgue 控制收敛）**：设 $\{f_k\}$ 是 $E$ 上的可测函数列，$f_k\to f$ a.e.，且存在**控制函数** $g\ge0$，$g\in L^1(E)$，使

$$|f_k(x)|\le g(x)\ \ \text{a.e.}\quad \forall k$$

则 $f\in L^1(E)$，且

$$\lim_{k\to\infty}\int_Ef_k\,dm=\int_Ef\,dm$$

**推论（依测度版本）**：若 $f_k\overset{m}{\to}f$（依测度收敛）且被 $g\in L^1$ 控制，则结论同样成立（证明：抽 a.e. 收敛子列用 DCT，再由整列收敛于同一极限）。

**重点：DCT 的三个条件——a.e. 收敛、可积控制、$g$ 可积——缺一不可。** 控制函数的作用是「把振荡关进笼子里」：$|f_k|\le g$ 保证积分不会因为函数值的暴走而失控，$g\in L^1$ 保证笼子的总质量有限。

## 2 证明：从 Fatou 到等式

**证明**：关键是把「≤」升级为「=」。由 $|f_k|\le g$，$g+f_k\ge0$，$g-f_k\ge0$——两个非负列。对它们分别用 Fatou 引理：

$$\int(g+f)=\int\liminf(g+f_k)\le\liminf\int(g+f_k)=\int g+\liminf\int f_k$$
$$\int(g-f)\le\liminf\int(g-f_k)=\int g-\limsup\int f_k$$

两式分别给出 $\int f\le\liminf\int f_k$ 与 $-\int f\le-\limsup\int f_k$，即 $\limsup\int f_k\le\int f$。合并：

$$\limsup\int f_k\le\int f\le\liminf\int f_k$$

故 $\lim\int f_k=\int f$。<span class="marginnote">证明的精髓：<strong>把「控制」翻译成「两个非负列 $g\pm f_k$」</strong>，于是 Fatou 的两个方向恰好夹出等式。控制函数 $g$ 的「可积性」在这里是双保险——既保证 $g\pm f_k$ 的积分差有意义（不出现 $\infty-\infty$），又提供 Fatou 需要的非负框架。</span>

**可积性**：$|f|\le g$ a.e.（$f_k\to f$ 与 $|f_k|\le g$ 取极限），故 $\int|f|\le\int g<\infty$，$f\in L^1$。

## 3 DCT 的两个常见变体

**变体一（有界收敛定理）**：若 $m(E)<\infty$ 且 $|f_k|\le M$（一致有界），则 $f_k\to f$ a.e. 时 $\int f_k\to\int f$。控制函数取 $g\equiv M\chi_E\in L^1$（有限测度保证 $M\,m(E)<\infty$）。**一致有界 + 有限测度 ⇒ 控制收敛**——这是 DCT 最常用的简化形态。

**变体二（广义控制）**：控制函数可以随 $k$ 变化，只要「一致可积」：$\{f_k\}$ 一致可积（uniformly integrable）时，a.e. 收敛给出积分收敛。一致可积是「存在统一控制」的推广，在概率论（鞅收敛）中至关重要。

**辨析｜易错点：控制函数必须「不依赖 $k$」。** 若只有「对每个 $k$ 存在 $g_k\in L^1$ 使 $|f_k|\le g_k$」但 $\int g_k$ 无界，DCT 不成立。反例：$f_k=k\chi_{(0,1/k]}$，$|f_k|\le g_k=k\chi_{(0,1/k]}$ 且 $\int g_k=1$，但 $\int f_k=1\not\to\int 0=0$。**「统一控制」而非「逐项控制」是 DCT 的前提。**

## 4 公式解析：DCT 与「无控制」的对比

把 DCT 与它失效的反例并排，看清「控制」的价值：

$$\text{DCT:}\quad |f_k|\le g\in L^1,\ f_k\to f\ \Rightarrow\ \lim\int f_k=\int f$$

$$\text{反例:}\quad f_k=k\chi_{(0,1/k]},\quad \int f_k=1\not\to 0=\int 0$$

- **第一步，读「DCT 的控制」**：$g$ 是「与 $k$ 无关」的可积大伞，罩住全部 $f_k$。**它把「振荡的高度」全局封顶**——即使 $f_k$ 在某处冲得很高（如 $k$），冲高的区域也被 $g$ 的有限积分约束。
- **第二步，读「反例为何失效」**：$f_k=k\chi_{(0,1/k]}$ 在收缩区间上冲高到 $k$。任何统一控制 $g$ 必须满足 $g\ge\sup_kf_k$——即 $g$ 在 $(0,1]$ 上处处 ≥ 任意大的 $k$？不：对固定 $x$，只有 $\tfrac1k\ge x$ 的 $k$ 才使 $f_k(x)=k$，随 $k$ 增大 $x$ 上的峰值是 $1/x$，$\int_0^1\tfrac1x dx=\infty$——**任何统一控制 $g$ 都不可积**。控制函数无法存在，DCT 的前提落空。
- **第三步，读「本质区别」**：DCT 要求「能量总量受控」（$\int g<\infty$）；反例的「能量」$k\cdot\tfrac1k=1$ 虽单步有限，但「峰值面积」集中在越来越小的区域，最终任何可积 $g$ 都罩不住。**「逐点有界」与「统一可积控制」是两回事**。

**「统一可积控制」是 DCT 的唯一开关**——它把「a.e. 收敛」升级为「积分收敛」，靠的正是把振荡关进有限质量的笼子。

## 6 数值演练与 DCT 速查

**算例一（DCT 的典型用法）**：$f_k(x)=\tfrac{\sin(kx)}{1+x^2}$ 于 $[0,\infty)$。$|f_k|\le\tfrac1{1+x^2}\in L^1$（控制函数），且 $f_k\to0$ a.e.（$\sin(kx)$ 振荡但被 $1/x^2$ 压平——实际逐点不一定收敛到 0，更稳妥的例：$f_k=\tfrac{1}{1+x^2}\cdot\tfrac{kx}{1+kx}$，逐点 $\to\tfrac1{1+x^2}$，被 $\tfrac1{1+x^2}$ 控制），故 $\int f_k\to\int\tfrac{dx}{1+x^2}=\pi$。

**算例二（参数积分的可导性）**：$F(t)=\int_0^\infty e^{-x}\cos(tx)dx$。对 $t$ 求导：$\partial_t(e^{-x}\cos(tx))=-xe^{-x}\sin(tx)$，被 $xe^{-x}\in L^1$ 控制，故 $F'(t)=-\int_0^\infty xe^{-x}\sin(tx)dx$——**积分号下求导合法**（DCT 对差商的极限）。

**对照表：三大极限定理的适用场合**

| 定理 | 条件 | 结论 |
| --- | --- | --- |
| Levi | $f_k\ge0$ 单调递增 | $\int\lim=\lim\int$ |
| Fatou | $f_k\ge0$ | $\int\liminf\le\liminf\int$ |
| DCT | $|f_k|\le g\in L^1$，$f_k\to f$ a.e. | $\int f_k\to\int f$ |

**术语速查**

| 记号 | 含义 |
| --- | --- |
| 控制函数 $g$ | 与 $k$ 无关的可积大伞 |
| a.e. | almost everywhere |
| 一致可积 | 控制可随 $k$ 变化的推广 |
| 换序 | 交换 $\lim$ 与 $\int$ |

**辨析｜易错点：DCT 的「a.e. 收敛」可放宽为「依测度收敛」，但「控制」不可放宽为「逐项控制」。** 依测度版本靠抽子列；逐项控制版本（$k\chi_{(0,1/k]}$）给出 $0=\int0\ne\lim\int f_k=1$。**「统一可积控制」是 DCT 的不可动摇的前提。**

### 三步记住 DCT 证明

- **双非负**：$g+f_k\ge0$、$g-f_k\ge0$。
- **双 Fatou**：$\int f\le\liminf\int f_k$、$\int f\ge\limsup\int f_k$。
- **夹出等式**：$\limsup\le\int f\le\liminf$，两端相等。

**延伸（与概率论连接）**：DCT 的概率版「$X_n\to X$ a.s.、$|X_n|\le Y\in L^1$ ⇒ $E[X_n]\to E[X]$」是鞅收敛、遍历定理、中心极限定理证明中的标准换序工具——「几乎必然收敛 + 可积控制」在概率里同样万能。

**一道收束练习**：证明 $\lim_{n\to\infty}\int_0^1\frac{nx}{1+n^2x^2}dx=0$ 且 DCT 可直接用（$|\tfrac{nx}{1+n^2x^2}|\le\tfrac12$ 于 $[0,1]$，逐点 $\to0$，控制函数 $\tfrac12\chi_{[0,1]}\in L^1$）。

## 7 小结

- **DCT**：$f_k\to f$ a.e. + $|f_k|\le g\in L^1$ ⇒ $\lim\int f_k=\int f$。
- **证明**：$g\pm f_k\ge0$ 双 Fatou，夹出等式。
- **变体**：有界收敛（有限测度 + 一致有界）、一致可积推广。
- **纪律**：控制函数必须与 $k$ 无关且可积；$k\chi_{(0,1/k]}$ 是失效反例。
- **地位**：三大极限定理之首，一切「换序」论证的万能钥匙。
- **数值**：$f_k=\tfrac{1}{1+x^2}\cdot\tfrac{kx}{1+kx}$ 于 $[0,\infty)$，DCT 给出 $\int f_k\to\pi$。
- **换序**：「$\lim\int=\int\lim$」的门槛从一致收敛降到可积控制。
- **参数求导**：$\partial_t f$ 被可积函数控制时，积分号下求导合法。

在下一节，我们研究 **DCT 的推论**：有界收敛定理与积分号下求极限、求导。
