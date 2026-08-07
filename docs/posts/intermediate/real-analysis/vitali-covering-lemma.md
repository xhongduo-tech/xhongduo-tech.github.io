---
title: Vitali 覆盖引理及其应用
date: 2026-08-07
---

# Vitali 覆盖引理及其应用

<div class="epigraph">
<p>当一族区间以任意比例逼近每个点时，总有一小撮不相交的区间，几乎罩住整个集合——这是微分理论的杠杆。</p>
<footer>—— 朱塞佩 · 维塔利（Giuseppe Vitali）</footer>
</div>

<div class="article-byline">
<p>第二级 · 实变函数与测度论 ｜ 周民强《实变函数论》第六章 ｜ 2026-08-07</p>
</div>

## 为什么从 Vitali 覆盖引理开始

Lebesgue 定理的证明卡在一步：如何在「坏点集」上把函数的增量与区间长度联系起来？答案就是 **Vitali 覆盖引理**——一个纯粹测度论的组合工具，却成为微分理论的杠杆支点。它说：如果一族区间「以任意比例」覆盖一个集合（Vitali 意义下），就能挑出**可数多个互不相交**的区间，几乎罩住整个集合（余集零测）。

这个引理把「稠密的覆盖」压缩为「互不相交的可数子覆盖」——于是测度的可数可加性、函数的单调性都能逐区间使用。它是单调函数可微性、Lebesgue 微分定理、以及测度论中一切「局部化」论证的通用引擎。<span class="marginnote">Vitali 覆盖引理是「<strong>贪心算法</strong>」的数学版：从区间族中反复挑「最大」的（或半径接近最大的），排除与已选重叠的，最终得到不相交的可数子族。它与<strong>Besicovitch 覆盖定理</strong>（允许有界重叠）是覆盖理论的两大支柱——Vitali 管「几乎不相交」，Besicovitch 管「有界重叠」，后者在高维更灵活。</span>

## 1 定理陈述

**定义（Vitali 覆盖）**：设 $E\subset\mathbb{R}$，$\mathcal{I}$ 是一族区间。若对每个 $x\in E$ 与每个 $\delta>0$，都存在 $I\in\mathcal{I}$ 使 $x\in I$ 且 $|I|<\delta$，则称 $\mathcal{I}$ 是 $E$ 的 **Vitali 覆盖**（即：以任意小长度覆盖每个点）。

**定理（Vitali 覆盖引理）**：设 $m(E)<\infty$，$\mathcal{I}$ 是 $E$ 的 Vitali 覆盖（区间族）。则对任意 $\varepsilon>0$，存在**可数个互不相交**的区间 $I_1,I_2,\dots\in\mathcal{I}$，使得

$$m\left(E\setminus\bigcup_{k=1}^{\infty}I_k\right)<\varepsilon$$

**重点：互不相交是可数可加性的通行证。** 覆盖引理的价值不在「覆盖得多好」（余集可以任意小），而在「选出的区间互不相交」——于是「测度的可数可加」与「函数的单调增量」能逐区间相加，这是证明里最需要的。

## 2 证明：贪心挑选

**证明**（经典贪心，区间情形）：

- **第一步，选第一个区间**：取 $I_1\in\mathcal{I}$ 使 $|I_1|>\tfrac12\sup\{|I|:I\in\mathcal{I}\}$（几乎最大的一个）。排除所有与 $I_1$ 相交的区间。
- **第二步，迭代挑选**：在剩余区间中取 $I_2$ 使 $|I_2|>\tfrac12\sup$（剩余中的几乎最大），再排除与 $I_2$ 相交的……得到 $I_1,I_2,\dots$。由构造，$I_k$ 互不相交。
- **第三步，估计余集**：每个被排除的区间 $I$ 都与某个已选 $I_k$ 相交且 $|I|\le2|I_k|$（几乎最大条件）。把「被排除区间」并入「5 倍放大的已选区间」$5I_k$（同中心五倍长），可得 $E\subset\bigcup_k5I_k$（a.e.）。于是 $m(E)\le\sum_k5|I_k|$，取足够多的 $k$ 使尾部 $\sum_{k>N}5|I_k|<\varepsilon$，则 $\bigcup_{k=1}^NI_k$ 之外余集 $<\varepsilon$。<span class="marginnote">第三步的「5 倍放大」是关键技巧：被排除的区间被其「附近」的已选区间罩住，而「附近」控制在 5 倍范围内。这个「常数 5」在证明中反复出现——<strong>「几乎最大」的挑选保证「排除者被放大罩住」</strong>，放大倍数由几何常数给出。</span>

## 3 应用一：Lebesgue 定理的关键步

Vitali 覆盖引理在单调函数可微性证明中的作用：

**引理（单侧导数控制）**：设 $f$ 单调不减，$E\subset[a,b]$，且对每个 $x\in E$，$\underline{D}^+f(x)\le q$（右下 Dini 导数 ≤ q）。则 $m(f(E))\le q\,m(E)$（$f(E)$ 是像集测度）。

**证明思路**：对每个 $x\in E$ 与任意小 $\delta$，存在小区间 $[x,x+h]\subset(x-\delta,x+\delta)$ 使 $f(x+h)-f(x)\le qh$（右下 Dini 导数 ≤ q）。这些区间构成 $E$ 的 Vitali 覆盖（适当化）。由覆盖引理选互不相交 $I_k$，则 $f(E)\subset\bigcup_k f(I_k)$（单调性）几乎处处，$m(f(E))\le\sum_k m(f(I_k))\le q\sum_k|I_k|\approx q\,m(E)$。**单调性把「区间增量 ≤ q·长度」翻译成「像集测度 ≤ q·原集测度」**。

**辨析｜易错点：$m(f(E))\le q\,m(E)$ 需要「$f$ 单调 + Vitali 覆盖」双条件。** 若 $f$ 不单调，像集的测度可能被「折叠」放大（$f$ 振荡时 $m(f(E))$ 失控）；单调性保证 $f$ 把区间映成区间、把不相交保持为不相交（测度可加）。**「单调」与「覆盖」是一对黄金搭档。**

## 4 应用二：Lebesgue 微分定理与密度

Vitali 覆盖引理还直接服务于：

**应用二（Lebesgue 微分定理的证明基础）**：$\frac{d}{dx}\int_a^x f=f(x)$ a.e.（下节详述）——证明中需要「对坏点集用 Vitali 覆盖」来估计「平均值的偏差」。

**应用三（勒贝格密度点）**：对可测集 $A$，记 $\mathcal{D}_A(x)=\lim_{r\to0}\tfrac{m(A\cap B(x,r))}{m(B(x,r))}$（$A$ 在 $x$ 处的密度）。Lebesgue 密度定理：a.e. $x\in A$ 处 $\mathcal{D}_A(x)=1$，a.e. $x\notin A$ 处 $\mathcal{D}_A(x)=0$。证明靠 Vitali 覆盖——**「测度密度几乎处处取 0 或 1」是可测集的局部结构定理**。<span class="marginnote">密度定理是 Vitali 覆盖的「分形级」应用：它把「可测集」描述为「几乎所有点附近都像满密度或零密度」。这个结论在<strong>遍历论、调和分析中的极大函数、以及分形几何</strong>里反复出现——可测集不是「处处均匀」，而是「几乎处处密度分明」。</span>

**公式解析：密度定理的覆盖论证**

$$\mathcal{D}_A(x)=\lim_{r\to0}\frac{m(A\cap B(x,r))}{m(B(x,r))}=1\ \ \text{对 a.e.}\ x\in A$$

- **第一步，读「坏点集 $E_\varepsilon$」**：设 $E_\varepsilon=\{x\in A:\liminf_{r\to0}\tfrac{m(A\cap B(x,r))}{m(B(x,r))}<1-\varepsilon\}$（密度显著小于 1 的点）。目标是证 $m(E_\varepsilon)=0$。
- **第二步，读「用 Vitali 覆盖收集小球」**：对每个 $x\in E_\varepsilon$，存在任意小的 $B(x,r)$ 使 $m(A\cap B(x,r))<(1-\varepsilon)m(B(x,r))$。这些球构成 $E_\varepsilon$ 的 Vitali 覆盖。**「密度小」给出「覆盖球内 $A$ 的占比低」**。
- **第三步，读「可数可加 + 矛盾」**：选不相交子覆盖 $B_k$，则 $\sum_k m(A\cap B_k)<(1-\varepsilon)\sum_km(B_k)$；而 $A\cap(\bigcup B_k)$ 测度接近 $m(A\cap\text{局部})$，结合 $m(A)$ 的估计推出矛盾，$m(E_\varepsilon)=0$。**「占比低」的局部信息通过覆盖累积成矛盾**。

**「Vitali 覆盖 + 可数可加 + 局部占比」**，是密度定理的标准证明结构——也是 Lebesgue 微分定理的同一台机器。

## 5 小结

- **Vitali 覆盖**：任意小长度覆盖每个点；可挑互不相交子族几乎罩住全集合。
- **证明**：贪心「几乎最大」挑选 + 5 倍放大罩住排除者。
- **应用一**：单调函数「$m(f(E))\le q\,m(E)$」——Lebesgue 定理的关键步。
- **应用二**：Lebesgue 微分定理与密度定理（密度 a.e. 取 0 或 1）。
- **地位**：微分理论、覆盖理论、遍历论的通用杠杆。

在下一节，我们定义**有界变差函数**，并研究它的基本性质——单调函数微积分的自然推广。
