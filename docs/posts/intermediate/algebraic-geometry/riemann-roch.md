---
title: Riemann-Roch 定理
date: 2026-08-11
---

# Riemann-Roch 定理

<div class="epigraph">
<p>Riemann-Roch 是代数曲线理论的心脏：它把计数、亏格与次数焊成一条等式。</p>
<footer>—— 由伯恩哈德 · 黎曼（Bernhard Riemann）与古斯塔夫 · 罗赫（Gustav Roch）思想转述</footer>
</div>

<div class="article-byline">
<p>第二级 · 进阶数理 · 代数几何 ｜ 对标教材 ｜ 2026-08-11</p>
</div>

## 为什么从 Riemann-Roch 继续

前面两节我们搭建了上同调（第 10 篇）与 Serre 对偶（第 11 篇）。现在到了验收时刻：**Riemann-Roch 定理**。它是代数曲线理论最重要的一条公式，回答一个最朴素的问题：

> 给定曲线 $C$ 上的一个除子 $D$，有多少个有理函数以 $D$ 为"极点预算"？即 $\ell(D) = \dim H^0(C, \mathcal{O}(D))$ 是多少？

这个问题在历史上（1850 年代）由黎曼与罗赫提出并部分解决，它直接连着"曲线的代数函数论"——每个代数曲线对应一个函数域，而"给定极点预算求函数个数"正是数论里类数问题的几何版。<span class="marginnote">黎曼的原始形式是关于 Riemann 面的"在 $n$ 个点给定主部后亚纯函数空间的维数"。罗赫加上了"余维"项。现代形式 $\ell(D) - \ell(K-D) = \deg D + 1 - g$ 则是 Grothendieck-Hirzebruch 视野下用层上同调重写的——教科书上 Hartshorne Ch. IV §1 的标准内容。</span>

Riemann-Roch 是后面所有几何的"计账"工具：第 13 篇 Riemann-Hurwitz 用它算分支，椭圆曲线理论用它算群结构，甚至与"从极限到大模型"主线的"计数复杂度"精神同源——用精确的等式换得精确的维数。

## 1 定理陈述

**重点：Riemann-Roch 定理。** 设 $C$ 是亏格 $g$ 的光滑射影曲线，$D$ 是 $C$ 上的任意除子，$K$ 是典范除子。则

$$\ell(D) - \ell(K - D) = \deg D + 1 - g$$

其中 $\ell(D) = \dim_k H^0(C, \mathcal{O}(D))$，$\ell(K-D) = \dim_k H^0(C, \mathcal{O}(K-D))$。<span class="marginnote">左边两个"截面数"之差，右边"次数 + 亏格修正"。亏格 $g$ 是 $C$ 的拓扑/代数不变量（$g = \ell(K)$），$\deg D$ 是"预算"大小。公式说：<strong>截面数的"盈余"由次数与亏格精确决定</strong>。</span>

用上同调的语言，$\ell(K-D) = \dim H^1(C, \mathcal{O}(D))$（Serre 对偶），于是公式等价于

$$\chi(\mathcal{O}(D)) = \ell(D) - \dim H^1(C, \mathcal{O}(D)) = \deg D + 1 - g$$

**Euler 示性数 = 次数 + (1 - g)**。这条形式是"上同调是可加的"这一原理的体现：Euler 示性数沿短正合列可加，因此可以"把 $D$ 拆成一堆点"来逐点计算。<span class="marginnote">为什么加 $(1-g)$？对 $D = 0$：$\chi(\mathcal{O}_C) = 1 - g$。这条"常数项"是曲线自身的不变量：对 $\mathbb{P}^1$（$g=0$）为 1，对椭圆曲线（$g=1$）为 0，对超椭圆曲线（$g \ge 2$）为负——它提示曲线越"复杂"、结构越"富余"越要扣掉。</span>

## 2 三个热身例子

**例 1：$\mathbb{P}^1$（亏格 0）。** $\ell(D) - \ell(K-D) = \deg D + 1$。若 $D$ 的次数 $d \ge 0$，则 $\ell(D) = d + 1$（次数 $d$ 的齐次多项式空间维数），$\ell(K-D) = 0$。例如 $d = 2$：$\ell = 3$，即 $H^0(\mathbb{P}^1, \mathcal{O}(2))$ 是 3 维（由 $x_0^2, x_0 x_1, x_1^2$ 生成）。<span class="marginnote">$\mathbb{P}^1$ 上一切由次数决定：$\ell(dH) = d+1$。这解释了二次曲线上的映射为何由"三个齐次多项式"给出——Veronese 嵌入 $\mathbb{P}^1 \to \mathbb{P}^2$ 就是"三个二次式"。</span>

**例 2：椭圆曲线（亏格 1）。** $\ell(D) - \ell(K-D) = \deg D$（因 $1 - g = 0$）。且 $K \sim 0$（$\deg K = 2g - 2 = 0$ 且 $\ell(K) = g = 1$ 故 $K$ 主除子）。于是 $\ell(D) = \deg D$ 对 $\deg D > 0$。<span class="marginnote">亏格 1 曲线的"零修正"使截面数恰好等于次数——这条简单性正是椭圆曲线群结构"每个次数 $\ge 1$ 的除子类恰好有 $\deg D$ 个有效代表"的来源。</span>

**例 3：$\mathbb{P}^2$ 里的光滑三次曲线（$g = 1$）。** 取 $D = P$（一个点），$\deg P = 1$，R-R 给 $\ell(P) - \ell(K - P) = 1$。因 $\ell(K-P) = \ell(-P) = 0$（负除子无截面），得 $\ell(P) = 1$——只有常数的整体截面，椭圆曲线的除子理论从一而终。

## 3 立即推论：次数足够大时的截面数

Riemann-Roch 最实用的推论是"次数够大，$H^1$ 消失"：

**重点：次数引理。** 若 $\deg D \ge 2g - 1$，则 $H^1(C, \mathcal{O}(D)) = 0$，且

$$\ell(D) = \deg D + 1 - g$$

若 $\deg D > 2g - 2$，则 $H^1(C, \mathcal{O}(D)) = 0$ 与 $\ell(D) = \deg D + 1 - g$ 自动成立。<span class="marginnote">直觉：$D$ 的预算越大，能写出的有理函数越多，直到"塞满"由次数与亏格决定的上限。$2g - 2 = \deg K$ 是分水岭：预算超过典范预算后，障碍 $H^1$ 完全消失，公式变成纯加法的 $\ell(D) = \deg D + 1 - g$。</span>

这个引理的证明值得走一遍，因为它示范"消没 + 对偶"的标准配合：由 Serre 对偶 $\dim H^1(C, \mathcal{O}(D)) = \ell(K - D)$。若 $\deg D > \deg K$，则 $\deg(K - D) < 0$，负次数除子无截面（$H^0(C, \mathcal{O}(E)) = 0$ 当 $\deg E < 0$），故 $H^1 = 0$。<span class="marginnote">"负次数除子无截面"本身也可由 R-R 直接推出：$\ell(E) \le \ell(E) + \ell(K - E) = \deg E + 1 - g$，当 $\deg E < 0$ 时右边 $\le 0$ 且 $\ell(E) \ge 0$，故 $\ell(E) = 0$。R-R 是它自身推论的基础——这种"定理自洽"在代数几何里很常见。</span>

## 4 应用到几何构造：典范映射

Riemann-Roch 不仅计数，还驱动构造。考虑**典范线性系** $|K|$（$K$ 为典范除子）：

**重点：典范映射。** $\ell(K) = g$，故 $|K|$ 是 $g - 1$ 维射影空间，给出有理映射

$$\varphi_K: C \dashrightarrow \mathbb{P}^{g-1}, \qquad P \longmapsto [\omega_1(P) : \cdots : \omega_g(P)]$$

其中 $\omega_1, \dots, \omega_g$ 是 $H^0(C, \omega_C)$ 的一组基。R-R 决定这个映射的分支结构：<span class="marginnote">典范映射把曲线"内在地"嵌入射影空间——不依赖任何外部的 $\mathbb{P}^n$ 嵌入，只靠 $C$ 自身的全纯微分。它是曲面的"自己画自己"。$g = 2$ 时 $|K|$ 是 $\mathbb{P}^1$（一个线性系），$\varphi_K: C \to \mathbb{P}^1$ 是 2:1 覆盖——超椭圆曲线的起源。</span>

若 $C$ 非超椭圆（$g \ge 3$），$\varphi_K$ 是**闭浸入**，$C$ 被嵌入 $\mathbb{P}^{g-1}$，其像的次数为 $2g - 2$（$\deg K$）。
- 若 $C$ 超椭圆，$\varphi_K$ 是 $2:1$ 覆盖 $\mathbb{P}^1$（因为 $K$ 有一维的"移动部分"）。

R-R 在这里的职责：$\ell(K - P) = g - 2$ 或 $g - 2$ 的分支，决定 $\varphi_K$ 在 $P$ 处的"秩"，从而决定是否浸入。<span class="marginnote">"曲线能被自己的微分嵌入"这句话的每处细节都由 R-R 计数支撑。$g=3$ 时 $\varphi_K: C \hookrightarrow \mathbb{P}^2$ 把曲线映成四次平面曲线（$\deg K = 4$）——与第 9 篇"平面曲线亏格 $g = (d-1)(d-2)/2$"吻合：$d=4$ 给 $g=3$。</span>

**辨析｜易错点：** $\ell(D)$ 是"截面空间维数"，不是"除子 $D$ 的有效代表个数"。$|D|$ 作为射影空间维数才是 $\ell(D) - 1$。初学时常把"$\ell(D) = d + 1$"误读为"有 $d+1$ 个有效除子"。正确读法：$H^0$ 的**维数**是 $d+1$，射影空间 $|D|$ 是 $d$ 维。R-R 给出的是向量空间维数，别忘了 $-1$。

## 5 公式解析：$\ell(D) - \ell(K-D) = \deg D + 1 - g$

分三步拆解：

**第一步，左右两边各是什么**：左边 $\ell(D) - \ell(K-D)$ 是"两个截面数的差"。$\ell(D)$ = "以 $D$ 为极点预算的函数个数"；$\ell(K-D)$ = "以 $K-D$ 为预算的函数个数"（Serre 对偶 = 同于 $H^1(C, \mathcal{O}(D))$ 的维数，即"障碍大小"）。右边 $\deg D + 1 - g$ 是纯"数字"：次数 + 亏格修正。<span class="marginnote">把公式读成"实际截面数 - 障碍维数 = 预期值"，就是黎曼原初想法的现代形式：$\ell(D) = \deg D + 1 - g + \text{(障碍项)}$，障碍项恰好由对偶的 $\ell(K-D)$ 给出。</span>
**第二步，为什么 Euler 示性数可加**：$\chi(\mathcal{O}(D)) = \ell(D) - \ell(K-D)$。Euler 示性数沿短正合列 $0 \to \mathcal{O}(D - P) \to \mathcal{O}(D) \to k_P \to 0$ 可加：$\chi(\mathcal{O}(D)) = \chi(\mathcal{O}(D-P)) + 1$。逐点减 $P$，直到 $\deg = 0$，得 $\chi(\mathcal{O}(D)) = \deg D + \chi(\mathcal{O}_C)$。而 $\chi(\mathcal{O}_C) = 1 - g$ 由 $\ell(K) = g$ 与 Serre 对偶给出。<span class="marginnote">整个证明像剥洋葱：每次剥掉一个点，Euler 示性数加 1；剥完后剩下"零除子的示性数 = $1 - g$"。$1-g$ 是曲线的"本底计数"，是亏格在 R-R 里出现的唯一方式。</span>
- **第三步，亏格从哪来**：$g = \ell(K)$ = 全纯微分维数（第 11 篇）。它出现在常数项：$\chi(\mathcal{O}_C) = 1 - g$。所以 R-R 把三条独立信息——次数（外部预算）、亏格（内在拓扑）、截面数（可计算几何）——用一条等式绑定。

一句话直觉：**"能写出的函数数 = 预算 - 亏格 + 1，再减去对偶预算的障碍"**。次数管"量"，亏格管"结构的复杂度"，两者在一条等式里互偿。

## 6 小结

- **Riemann-Roch**：$\ell(D) - \ell(K-D) = \deg D + 1 - g$，或 $\chi(\mathcal{O}(D)) = \deg D + 1 - g$。
- **Euler 可加性**：剥点归纳，$\chi(\mathcal{O}(D)) = \deg D + \chi(\mathcal{O}_C)$，$\chi(\mathcal{O}_C) = 1 - g$。
- **次数引理**：$\deg D > 2g - 2$ ⟹ $H^1 = 0$、$\ell(D) = \deg D + 1 - g$。
- **应用**：$g = \ell(K)$ 定义亏格；$\mathbb{P}^1$ 上 $\ell(dH) = d + 1$；椭圆曲线 $\ell(D) = \deg D$；典范映射 $\varphi_K: C \hookrightarrow \mathbb{P}^{g-1}$。
- **易错**：$\ell(D)$ 是向量空间维数，$|D|$ 的射影维数是 $\ell(D) - 1$。

在下一节，我们离开"单条曲线"，研究曲线之间的映射：**Riemann-Hurwitz 公式与曲线论深化**——用分支点计算覆盖的亏格，并把椭圆曲线分类成它们的模空间。
