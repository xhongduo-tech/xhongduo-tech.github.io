---
title: Fatou 引理：叙述、证明与不可改进之处
date: 2026-08-07
---

# Fatou 引理：叙述、证明与不可改进之处

<div class="epigraph">
<p>当极限与积分不能交换时，Fatou 给我们一个方向——下极限的积分永远不小于积分的下极限。</p>
<footer>—— 皮埃尔 · 法图（Pierre Fatou）</footer>
</div>

<div class="article-byline">
<p>第二级 · 实变函数与测度论 ｜ 周民强《实变函数论》第五章 ｜ 2026-08-07</p>
</div>

## 为什么从 Fatou 引理开始

Levi 单调收敛处理了「单调」情形。但函数列往往不单调——振荡、跳跃、峰谷交替。此时 $\lim\int f_k$ 与 $\int\lim f_k$ 可能都不存在或不等。Fatou 引理在最一般的情形下给出一个**单方向**的结论：对非负函数列，**下极限的积分 ≤ 积分的下极限**。它不要求收敛、不要求可积、不要求单调——是三大定理中最「廉价」也最「普适」的一条。

Fatou 引理的哲学是「**宁可保守，不可撒谎**」：当无法确定「极限的积分」时，至少保证「下极限的积分」不会超过「积分的下极限」。这个「≤」不是缺陷，而是在最坏情形下的**真相上界**。<span class="marginnote">Fatou 引理的「不可改进」之处在于：<strong>严格不等式可以出现</strong>——$f_k=k\chi_{(0,1/k]}$ 给出 $\int\liminf=0<\liminf\int=1$。「≤」不能被改成「=」。但在<strong>控制收敛</strong>的条件下（有可积大伞），「≤」升级为「=」——Fatou 是控制收敛的「松绑版」。</span>

## 1 定理陈述

**定理（Fatou 引理）**：设 $\{f_k\}$ 是 $E$ 上的非负可测函数列。则

$$\int_E\left(\liminf_{k\to\infty}f_k\right)dm\le\liminf_{k\to\infty}\int_Ef_k\,dm$$

**推论（概率版）**：$X_n\ge0$ 时，$E[\liminf X_n]\le\liminf E[X_n]$。

**重点：条件只有「非负」，没有别的。** 不要求收敛（用 $\liminf$ 代替 $\lim$）、不要求可积（允许无穷）、不要求单调。**这是三大定理里适用范围最广的一条**——任何非负函数列都有资格被 Fatou 约束。

### 为什么用 $\liminf$ 而不用 $\lim$

函数列 $f_k$ 的 $\lim$ 常常不存在（振荡、发散），但 $\liminf$ 与 $\limsup$ 总是存在（允许 $+\infty$）。Fatou 在两层都取下极限：函数层 $\liminf_kf_k$ 与数值层 $\liminf_k\int f_k$。<span class="marginnote">若 $f_k\to f$ a.e. 本身成立，则 $\liminf_kf_k=f$ a.e.，Fatou 退化为 $\int f\le\liminf\int f_k$——这条「Fatou 直接形态」在证明 L^1 极限的可积性时直接可用，无需额外条件。</span>

### 名词速查

| 记号 | 含义 | 直觉 |
| --- | --- | --- |
| $\liminf_{k}f_k=\sup_n\inf_{k\ge n}f_k$ | 函数列的下极限 | 「尾部下确界的极限」 |
| $\liminf_k\int f_k$ | 一列积分值的下极限 | 数列的下极限 |
| $g_n=\inf_{k\ge n}f_k$ | 尾部下确界函数 | 随 $n$ 单调递增 |
| $k\chi_{(0,1/k)}$ | 尖峰函数 | 峰越高底越窄，面积恒为 1 |
| a.e. | almost everywhere | 除零测集外成立 |

## 2 证明：从 Levi 出发

**证明**：核心技巧是「把 $\liminf$ 拆成单调列」。由定义 $\liminf_kf_k=\sup_n\inf_{k\ge n}f_k$，记 $g_n=\inf_{k\ge n}f_k$（关于 $n$ 单调递增）。于是 $\liminf_kf_k=\lim_ng_n=\sup_ng_n$（递增极限）。由 Levi 单调收敛：

$$\int\liminf_kf_k=\int\lim_ng_n=\lim_n\int g_n$$

而 $g_n=\inf_{k\ge n}f_k\le f_k$ 对一切 $k\ge n$，故 $\int g_n\le\int f_k$ 对一切 $k\ge n$，取 $k$ 的下确界：$\int g_n\le\inf_{k\ge n}\int f_k$。再对 $n\to\infty$ 取下极限：$\lim_n\int g_n\le\liminf_k\int f_k$。合并即得结论。<span class="marginnote">证明的机关：<strong>把「下极限」化为「单调递增列 $g_n$ 的极限」</strong>，于是 Levi（需要单调）可以用上。$g_n=\inf_{k\ge n}f_k$ 是「尾部下确界」，天然递增——<strong>任何非单调列的下极限都可以被「单调化」</strong>，这正是 $\liminf$ 存在的意义。</span>

## 3 不可改进与等号条件

**例子（严格不等式）**：$f_k=k\chi_{(0,1/k]}$。$\liminf_kf_k=0$ 处处，$\int\liminf=0$；而 $\int f_k=k\cdot\tfrac1k=1$，$\liminf\int f_k=1$。故 $0\le1$ 严格。**「≤」不能改进为「=」**。

**把算例算到底（改用开区间避开端点歧义）**：取 $f_k=k\chi_{(0,1/k)}$。对 $x>0$，当 $k>\tfrac1x$ 时 $x\notin(0,1/k)$，故 $f_k(x)=0$ 最终成立，$\liminf_kf_k(x)=0$；$x=0$ 处每一项都取 $0$。于是 $\liminf_kf_k\equiv0$，$\int\liminf=0$。而 $\int f_k=k\cdot m(0,1/k)=k\cdot\tfrac1k=1$。**$0<1$，严格不等式被完整验证，问题只出在「峰」上、与端点取法无关。**

**等号条件（何时 Fatou 取等）**：若 $\{f_k\}$ 被可积函数控制（$|f_k|\le g\in L^1$）且 $f_k\to f$ a.e.，则 Fatou 的「≤」升级为「=」——这正是 Lebesgue 控制收敛定理（下节）。**Fatou 是「无控制的真相」，控制收敛是「有控制的精确」**。

**例（概率期望的 Fatou）**：$X_n\ge0$ 时 $\liminf_nE[X_n]$ 给出期望下界——在大数定律、停时理论中，Fatou 用于「期望的下界估计」。

**辨析｜易错点：Fatou 要求非负，不能对「任意符号」的函数列直接用。** 若 $f_k$ 变号，$\int\liminf$ 可能无定义（正负无穷相消）。非负性保证 $\liminf$ 的积分有意义（非负或无穷）。**「非负」是 Fatou 的安全阀**——变号情形要靠正负部分别用 Fatou，或者直接用控制收敛。

## 4 公式解析：$g_n=\inf_{k\ge n}f_k$ 的单调化

Fatou 证明的唯一技巧是「尾部下确界的单调化」：

$$\int\underbrace{\left(\sup_n\underbrace{\inf_{k\ge n}f_k}_{g_n\uparrow}\right)}_{\liminf_kf_k}=\lim_n\int g_n\le\liminf_k\int f_k$$

- **第一步，读「$g_n=\inf_{k\ge n}f_k\uparrow$」**：尾部越长，下确界越小或相等，故 $g_n$ 关于 $n$ 递增。**单调性由「收窄尾部」天然保证**——这是 $\liminf$ 能被 Levi 处理的全部理由。
- **第二步，读「$\liminf_kf_k=\lim_ng_n$」**：下极限的定义就是「尾部下确界的极限」：$\sup_n\inf_{k\ge n}f_k$。递增列，$\sup=\lim$。**同一对象两种写法**（$\sup$ 与 $\lim$）都指向下极限。
- **第三步，读「$\int g_n\le\inf_{k\ge n}\int f_k$」**：$g_n\le f_k$（$k\ge n$）⇒ $\int g_n\le\int f_k$ 对每个 $k\ge n$，故 $\le$ 尾部下确界。**积分与下确界交换方向：$\int\inf\le\inf\int$**——这就是 Fatou「≤」的起源，它来自「$g_n$ 是各 $f_k$ 的下界」这一事实。

**「尾部下确界 → 单调列 → Levi → 保序」**，是 Fatou 证明的完整链条——每条不等式都来自「下界」与「单调」两件基本事实。

## 5 数值算例与三定理对照

**算例一（离散版严格不等式）**：设 $\Omega=\{a,b\}$ 各点概率 $\tfrac12$，令 $X_{2m}=\chi_{\{a\}}$、$X_{2m+1}=\chi_{\{b\}}$。则 $a$ 在奇数时刻取 0、$b$ 在偶数时刻取 0，$\liminf_kX_k\equiv0$，$E[\liminf X_k]=0$；而 $E[X_k]\equiv\tfrac12$，$\liminf E[X_k]=\tfrac12$。<span class="marginnote">与连续版机制同源：<strong>质量在两点间来回搬家，谁也不长期占优</strong>，于是「点的下极限」被压到 0，「期望」却稳定在 $\tfrac12$。</span>

**算例二（可积时取等）**：$f_k(x)=\dfrac{1}{1+x^2}\cdot\dfrac{k}{k+1}$ 于 $[0,1]$。$|f_k|\le1\in L^1$，$f_k\to\dfrac{1}{1+x^2}$ a.e.，控制收敛给出 $\int_0^1f_k\to\dfrac\pi4$，此时 Fatou 两端相等——$\lim f_k$ 存在，$\liminf$ 退化为 $\lim$，等号顺理成章。

**对照表：三大极限定理**

| 定理 | 条件 | 结论 | 是否要求收敛 |
| --- | --- | --- | --- |
| Levi 单调收敛 | $f_k\ge0$ 单调递增 | $\int\lim f_k=\lim\int f_k$ | 单调性保证 $\lim$ 存在 |
| Fatou 引理 | $f_k\ge0$ | $\int\liminf f_k\le\liminf\int f_k$ | 用 $\liminf$，不要求收敛 |
| 控制收敛 | $|f_k|\le g\in L^1$，$f_k\to f$ a.e. | $\int f_k\to\int f$ | 要求 a.e. 收敛 |

**辨析｜易错点：变号函数列不能直接套 Fatou。** 若 $f_k$ 可负，$\int\liminf f_k$ 可能无定义（如 $\liminf f_k=-\infty$）。标准处理：拆正负部 $f_k=f_k^+-f_k^-$，分别对非负列用 Fatou，或改判控制收敛。「非负」或「可积控制」，是 Fatou 的入场券。

**延伸（概率论与 L^p 中的出场）**：停时理论里 Fatou 常被反用为「$\liminf E[X_n]$ 的下界」。典型用法是：已知 $f_k\ge0$ 且 $\int f_k\le1$ 对一切 $k$，Fatou 直接给出 $\int\liminf_kf_k\le1$——无需任何收敛条件，只凭「非负 + 积分一致有界」就得到极限函数的积分上界。第八篇 L^p 空间里「范数有界 ⇒ 存在弱收敛子列」的证明，正是以这一思路为枢纽。

**对级数的形式**：Fatou 亦可写为 $\int\sum_k f_k\le\sum_k\int f_k$（非负项级数），无需一致收敛。它是下一节「逐项积分与级数的积分」的雏形——非负级数总可以交换求和与积分，只是方向是「≤」。

### 三步用 Fatou 求极限函数的上界

**问题**：设 $f_k\ge0$ 且 $\int f_k\le1$ 对一切 $k$ 成立，证明 $\int\liminf_kf_k\le1$。

- **第一步，写 Fatou**：$\int\liminf_kf_k\le\liminf_k\int f_k$。
- **第二步，代入界**：由 $\int f_k\le1$ 对一切 $k$，得 $\liminf_k\int f_k\le1$。
- **第三步，收尾**：$\int\liminf_kf_k\le1$，命题得证。

注意全过程中从未使用收敛性——「非负 + 积分一致有界」两个条件，就足以控制极限函数。这正是 Fatou 在紧性论证中的标准姿势。

### 反 Fatou 与夹逼视角

对上极限还有孪生不等式「$\int\limsup f_k\ge\limsup\int f_k$」（需 $f_k\le g\in L^1$ 从上控制）。当 $f_k\to f$ a.e. 且被控制时，正反两式夹出 $\int f_k\to\int f$——这构成控制收敛定理的又一条证明路径，也解释了为何 Fatou 是「控制收敛的松绑版」。

## 6 小结

- **Fatou 引理**：非负函数列，$\int\liminf f_k\le\liminf\int f_k$。
- **证明**：$g_n=\inf_{k\ge n}f_k\uparrow$，Levi + 保序。
- **不可改进**：$k\chi_{(0,1/k)}$ 与「两点搬家」均给出严格不等式；「≤」不能变「=」。
- **等号条件**：可积控制 + a.e. 收敛 ⇒ 控制收敛定理。
- **纪律**：非负性是安全阀；变号需拆正负部。
- **应用**：「非负 + 积分有界」即得极限函数可积上界，是 L^p 紧性论证的枢纽。
- **哲学**：「宁可保守，不可撒谎」——用下极限给出最坏情形的真实上界。

在下一节，我们迎来三大定理的最后一位、也是最强的一位：**Lebesgue 控制收敛定理**。
