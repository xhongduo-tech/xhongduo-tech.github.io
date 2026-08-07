---
title: Levi 单调收敛定理（Lebesgue 逐项积分定理）
date: 2026-08-07
---

# Levi 单调收敛定理（Lebesgue 逐项积分定理）

<div class="epigraph">
<p>单调递增的函数列，极限的积分就是积分的极限——单调性让「交换极限」分文不花。</p>
<footer>—— 贝波 · 列维（Beppo Levi）</footer>
</div>

<div class="article-byline">
<p>第二级 · 实变函数与测度论 ｜ 周民强《实变函数论》第五章 ｜ 2026-08-07</p>
</div>

## 为什么从单调收敛定理开始

积分的终极问题之一是：**何时 $\lim\int f_k=\int\lim f_k$？** 不是什么时候都能交换——反例遍地（锯齿波、移动冒泡）。Levi 单调收敛定理给出第一个「几乎无条件」的答案：**只要 $f_k\uparrow f$（单调递增）且非负，交换就免费**。这个定理是全部极限交换理论的基石：Fatou 引理与 Lebesgue 控制收敛都以它为源头。

Levi 定理的威力在于它的「无条件性」：不要求一致收敛、不要求可积、不要求有界，甚至允许积分 $=\infty$。单调递增的非负函数列，极限与积分天然可交换——**「单调」是分析学里最珍贵的朋友**。<span class="marginnote">Levi 定理在概率论中对应<strong>单调收敛定理（MCT）</strong>：$0\le X_n\uparrow X$ 则 $E[X_n]\uparrow E[X]$。它是证明「期望的单调收敛」「随机级数的期望」的免费工具，也是 Fatou 引理与 DCT 在概率框架下的基础。</span>

## 1 定理陈述

**定理（Levi 单调收敛）**：设 $\{f_k\}$ 是 $E$ 上的非负可测函数列，且 $0\le f_1\le f_2\le\cdots$（逐点单调递增）。则

$$\int_E\left(\lim_{k\to\infty}f_k\right)dm=\lim_{k\to\infty}\int_Ef_k\,dm$$

（两边可能同时等于 $+\infty$。）

**推论（可积情形的有限极限）**：若 $\sup_k\int f_k<\infty$，则 $\lim f_k$ 是有限的（a.e.）、可积的，且 $\lim\int f_k=\int\lim f_k<\infty$。

**重点：定理允许「极限函数等于 $\infty$」与「积分等于 $\infty$」。** 这是非负单调列的特性——极限存在（允许无穷），积分极限也存在（允许无穷），两者相等。**「$\infty=\infty$」在非负情形是合法的，不需要可积性**。

## 2 证明：从简单函数到单调列

**证明**：记 $f=\lim_kf_k$（单调递增，极限存在，允许无穷）。目标证 $\int f=\lim_k\int f_k$。

- **第一步，$\le$ 方向**：$f_k\le f$，由单调性 $\int f_k\le\int f$，取上确界 $\lim_k\int f_k\le\int f$。
- **第二步，$\ge$ 方向（核心）**：对任意简单函数 $\varphi\le f$ 与常数 $0<c<1$，考察集合
$$E_k=\{x:\varphi(x)\le c\,f_k(x)\}$$
因为 $f_k\uparrow f$ 且 $\varphi\le f$，$E_k\uparrow E$（几乎处处；去掉零测点后）。由单调收敛（集合层，第三篇），$m(E_k)\to m(E)$。
- **第三步，积分估计**：在 $E_k$ 上 $\varphi\le c f_k$，故 $\int_{E_k}\varphi\le c\int_{E_k}f_k\le c\int_Ef_k$。而 $\int_E\varphi=\lim_k\int_{E_k}\varphi$（集合逼近，因 $\varphi$ 简单函数），于是 $\int_E\varphi\le c\lim_k\int_Ef_k$。令 $c\to1$：$\int\varphi\le\lim_k\int f_k$。<span class="marginnote">第二步的「$E_k=\{x:\varphi(x)\le c f_k(x)\}$」是证明的机关：$c<1$ 保证「$\varphi\le f$ 且 $f_k\uparrow f$」能推出「$E_k\uparrow E$」——<strong>严格放缩 $c$ 是为了让「最终被超过」对每个点成立</strong>。若直接取 $c=1$，可能因「恰好等于」而不进 $E_k$。</span>
- **第四步，上确界收尾**：$\int\varphi\le\lim_k\int f_k$ 对一切简单 $\varphi\le f$ 成立，故 $\int f=\sup_\varphi\int\varphi\le\lim_k\int f_k$。与第一步合拢。

## 3 推论：单调递减与可积情形

**推论一（递减可积）**：设 $f_1\ge f_2\ge\cdots\ge0$ 且 $\int f_1<\infty$，则 $\lim\int f_k=\int\lim f_k$。证明：$g_k=f_1-f_k\uparrow$，对 $g_k$ 用 Levi，再移项（用 $\int f_1<\infty$ 保证减法合法）。

**推论二（级数积分）**：$\{f_k\}$ 非负可测，$\int\sum_kf_k=\sum_k\int f_k$（部分和单调）。

**推论三（Lebesgue 逐项积分定理）**：非负函数项级数可逐项积分——这正是「Lebesgue 逐项积分定理」这个名字的来源。

**辨析｜易错点：单调性不可省。** 若去掉「单调递增」，Levi 不成立。$f_k=k\chi_{(0,1/k]}$：$\int f_k=1$ 恒成立，$\lim f_k=0$ 处处，故 $\int\lim f_k=0\neq\lim\int f_k=1$。**振荡（不单调）让「峰」与「谷」的极限各自走散**——这正是 Fatou 引理（下节）要处理的情形，它给出「≤」而非「=」。

## 4 公式解析：$E_k$ 集合的关键作用

Levi 证明的枢纽是 $E_k=\{x:\varphi(x)\le cf_k(x)\}$ 的收敛性，拆解它：

$$\varphi\le f\ \Rightarrow\ E_k\uparrow E,\qquad \int_{E_k}\varphi\le c\int f_k$$

- **第一步，读「$E_k\uparrow E$」**：$f_k\uparrow f$，所以「$\varphi\le cf_k$」成立的区域随 $k$ 扩大（$f_k$ 越来越高，越过 $\varphi/c$ 的点越来越多），最终覆盖「$\varphi\le cf\le f$」的全体（去掉零测点）。**集合的单调并拢由函数的单调收敛驱动**。
- **第二步，读「在 $E_k$ 上的积分控制」**：$E_k$ 上 $\varphi\le cf_k$，积分单调性给 $\int_{E_k}\varphi\le c\int_{E_k}f_k\le c\int_Ef_k$。**$E_k$ 的引入把「$\varphi$ 与 $f_k$ 的大小关系」从全空间转移到「$\varphi$ 已被 $f_k$ 超越」的区域**。
- **第三步，读「$c\to1$ 与上确界」**：$\int_{E_k}\varphi\to\int_E\varphi$（集合逼近），$c\to1$ 给出 $\int\varphi\le\lim\int f_k$，再对一切 $\varphi\le f$ 取上确界。**「先 $k\to\infty$，再 $c\to1$，最后 $\sup_\varphi$」三连**——每次极限都合法（单调），故不等式保持。

**「$c<1$ 放缩 + 集合逼近 + 上确界」**，是 Levi 证明的标准机关——它是单调收敛定理最「非平凡」的一处技巧。

## 5 单调收敛的典型应用

**应用一（非负级数的积分）**：$\int\sum_{k}f_k=\sum_k\int f_k$（$f_k\ge0$）。这是 Levi 最常用的形态——级数与积分交换在非负情形下「免费」。例：$\int_0^1\sum_{k=0}^\infty x^k dx=\int_0^1\tfrac1{1-x}dx=\infty$ 且 $\sum_k\int_0^1x^kdx=\sum_k\tfrac1{k+1}=\infty$——两边都是无穷，Levi 合法地给出「$\infty=\infty$」。

**应用二（函数的下方面积）**：$f\ge0$ 可测，$\varphi_k\uparrow f$（二分取整逼近）。Levi 保证 $\int f=\lim_k\int\varphi_k$——**这同时证明「$\sup$ 定义」与「单调逼近定义」等价**（第五篇的缝合点）。非负函数积分的一切计算，都以 Levi 为合法性依据。

**应用三（概率中的期望单调收敛）**：$0\le X_n\uparrow X$ ⇒ $E[X_n]\uparrow E[X]$。大数定律、停时理论的期望计算都靠它——例如「非负随机变量和的期望 = 期望的和」：$E[\sum X_n]=\sum E[X_n]$（$X_n\ge0$），这是 Tonelli 定理的概率版核心。

**应用四（Lebesgue 测度的连续性回溯）**：$E_k\uparrow E$ 时 $m(E_k)\to m(E)$ 正是 Levi 对 $f_k=\chi_{E_k}$ 的应用。**测度连续性（第三篇）与函数单调收敛（本篇）是同一个 Levi 的两面**——这也解释了为何测度理论那么「自然地」通向积分理论。

**重点：Levi 的四个应用共享同一结构——「单调性让极限与积分交换」。** 遇到「$\int$ 与 $\lim/\sum$ 换序」，先问「函数列是否单调（非负）？」是，则 Levi 直接放行；否，才需要 Fatou（≤）或 DCT（控制）。**「单调优先」是极限交换的第一反应**。

## 6 小结

- **Levi 定理**：$0\le f_k\uparrow f$ ⇒ $\int\lim f_k=\lim\int f_k$（允许无穷）。
- **证明骨架**：$\le$ 由单调性；$\ge$ 由 $E_k$ 集合 + $c<1$ 放缩 + 上确界。
- **推论**：递减可积（加 $\int f_1<\infty$）、非负级数逐项积分。
- **不可省条件**：单调性——振荡反例 $k\chi_{(0,1/k]}$ 说明问题。
- **地位**：三大极限定理之首，Fatou 与控制收敛的总源头。

在下一节，我们把 Levi 用到级数上，专门研究**逐项积分与级数的积分**。
