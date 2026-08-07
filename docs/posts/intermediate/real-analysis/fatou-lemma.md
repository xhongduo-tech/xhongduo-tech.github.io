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

## 2 证明：从 Levi 出发

**证明**：核心技巧是「把 $\liminf$ 拆成单调列」。由定义 $\liminf_kf_k=\sup_n\inf_{k\ge n}f_k$，记 $g_n=\inf_{k\ge n}f_k$（关于 $n$ 单调递增）。于是 $\liminf_kf_k=\lim_ng_n=\sup_ng_n$（递增极限）。由 Levi 单调收敛：

$$\int\liminf_kf_k=\int\lim_ng_n=\lim_n\int g_n$$

而 $g_n=\inf_{k\ge n}f_k\le f_k$ 对一切 $k\ge n$，故 $\int g_n\le\int f_k$ 对一切 $k\ge n$，取 $k$ 的下确界：$\int g_n\le\inf_{k\ge n}\int f_k$。再对 $n\to\infty$ 取下极限：$\lim_n\int g_n\le\liminf_k\int f_k$。合并即得结论。<span class="marginnote">证明的机关：<strong>把「下极限」化为「单调递增列 $g_n$ 的极限」</strong>，于是 Levi（需要单调）可以用上。$g_n=\inf_{k\ge n}f_k$ 是「尾部下确界」，天然递增——<strong>任何非单调列的下极限都可以被「单调化」</strong>，这正是 $\liminf$ 存在的意义。</span>

## 3 不可改进与等号条件

**例子（严格不等式）**：$f_k=k\chi_{(0,1/k]}$。$\liminf_kf_k=0$ 处处，$\int\liminf=0$；而 $\int f_k=k\cdot\tfrac1k=1$，$\liminf\int f_k=1$。故 $0\le1$ 严格。**「≤」不能改进为「=」**。

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

## 5 小结

- **Fatou 引理**：非负函数列，$\int\liminf f_k\le\liminf\int f_k$。
- **证明**：$g_n=\inf_{k\ge n}f_k\uparrow$，Levi + 保序。
- **不可改进**：$k\chi_{(0,1/k]}$ 给出严格不等式；「≤」不能变「=」。
- **等号条件**：可积控制 + a.e. 收敛 ⇒ 控制收敛定理。
- **纪律**：非负性是安全阀；变号需拆正负部。

在下一节，我们迎来三大定理的最后一位、也是最强的一位：**Lebesgue 控制收敛定理**。
