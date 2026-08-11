---
title: 代数整数与数环
date: 2026-08-11
---

# 代数整数与数环

<div class="epigraph">
<p>数是人类心灵的自由创造。</p>
<footer>—— 理查德 · 戴德金（Richard Dedekind，Die Zahlen sind freie Schöpfungen des menschlichen Geistes）</footer>
</div>

<div class="article-byline">
<p>第二级 · 进阶数理 · 代数数论 ｜ 对标教材 ｜ 2026-08-11</p>
</div>

## 为什么从代数整数开始

初等数论在 $\mathbb{Z}$ 里解决「整数怎么分解成素数」。但方程一换，比如 $x^2 + 5 = y^3$、佩尔方程 $x^2 - 2y^2 = 1$，答案藏在 $\sqrt{-5}$、$\sqrt{2}$ 这样的新数里——一旦把 $\sqrt{-5}$ 加进整数，原来的「素数分解」规则就崩溃了。代数数论的第一课，就是把这些「数不够用就造新数」的朴素直觉严格化。<span class="marginnote">这与第一级《集合的概念》里「数系扩充」的主旋律一脉相承：自然数不够减，造出负整数；有理数不够开方，造出实数；如今是整数不够解方程，造出「代数整数」。</span>这一步造出的对象叫**代数整数**，它们凑成的环叫**数环**，是整个代数数论的舞台。

## 1 什么是代数整数

**代数数（algebraic number）**：满足某个**非零**有理系数多项式方程的复数。**代数整数（algebraic integer）**：满足某个**首一**整系数多项式方程的复数——「首一」意思是最高次项系数是 $1$。<span class="marginnote">「首一（monic）」是整个定义的命门：去掉它，$\frac{1}{2}$ 也满足 $2x - 1 = 0$，可就坏了。首一要求把「整性」锁死：代数整数是「在代数层面仍保持整数性」的数。</span>

**辨析｜易错点：** 把「代数整数」误当成「代数数的整数倍」是大错。$\frac{2}{3}$ 满足 $3x - 2 = 0$，但 $3x - 2$ 不是首一的，所以 $\frac{2}{3}$ 不是代数整数。反过来，$\frac{1+\sqrt{5}}{2}$ 满足 $x^2 - x - 1 = 0$，首一整系数，所以它**是**代数整数——尽管写出来像分数。判断标准永远是「是否存在首一整系数多项式」，而非「长得像不像整数」。

**关键事实**：代数整数的和与积仍是代数整数。这不是显然的（要证明两个数各自满足某方程，则它们的和与积也满足某个首一整系数方程），但它是真的。

**数域（number field）**：$\mathbb{Q}$ 的有限扩张 $K$，即有限维 $\mathbb{Q}$-向量空间。**数环（ring of integers）**：$K$ 中全体代数整数，记作 $\mathcal{O}_K$。它是整环，且处处比 $\mathbb{Z}$ 更「贴合」$K$ 的结构。

## 2 数环的结构：自由的模

数环 $\mathcal{O}_K$ 不只被「关」在 $K$ 里，它还有极强的代数结构——**它是秩为 $n = [K : \mathbb{Q}]$ 的自由 $\mathbb{Z}$-模**。

**定理（整基的存在性）：** 若 $[K : \mathbb{Q}] = n$，则存在 $\omega_1, \dots, \omega_n \in \mathcal{O}_K$，使每个代数整数都能**唯一**写成

$$
\alpha = a_1 \omega_1 + \cdots + a_n \omega_n, \qquad a_i \in \mathbb{Z}
$$

这样一组 $\{\omega_i\}$ 称为 $K$ 的一组**整基（integral basis）**。<span class="marginnote">这正是第二级《线性代数》里「自由模 = 有基的模」的具体化：把 $\mathcal{O}_K$ 看成 $\mathbb{Z}^n$ 在 $K$ 里的一个格（lattice）。「每个代数整数都能唯一写成整系数组合」意味着 $\mathcal{O}_K \cong \mathbb{Z}^n$ 作为 $\mathbb{Z}$-模。</span>

配合这套结构有两个核心不变量，贯穿全书：

**判别式（discriminant）**：对整基 $\{\omega_i\}$ 定义

$$
d_K = \det\big(\mathrm{Tr}_{K/\mathbb{Q}}(\omega_i\omega_j)\big)_{i,j}
$$

它不依赖整基的选取，是 $K$ 的不变量——在 Minkowski 几何（本专题后续篇目）中它恰是嵌入格的协体积平方（差一个常数）。

**迹与范（Trace and Norm）**：对 $\alpha \in K$，设它在 $\mathbb{Q}$ 上的所有共轭为 $\alpha^{(1)}, \dots, \alpha^{(n)}$（$n$ 个嵌入 $\sigma: K \hookrightarrow \mathbb{C}$ 的像），定义

$$
\mathrm{Tr}_{K/\mathbb{Q}}(\alpha) = \sum_i \alpha^{(i)}, \qquad \mathrm{N}_{K/\mathbb{Q}}(\alpha) = \prod_i \alpha^{(i)}
$$

**范与单位的联系**：$\alpha \in \mathcal{O}_K$ 时，$\mathrm{Tr}(\alpha), \mathrm{N}(\alpha) \in \mathbb{Z}$；并且 $\alpha$ 是 $\mathcal{O}_K$ 里的**单位**（可逆元）当且仅当 $\mathrm{N}(\alpha) = \pm 1$。单位理论（Dirichlet 单位定理）就在这个判定上展开。

## 3 三个原型：高斯整数、艾森斯坦整数与二次域

抽象定义先看活例。三类最著名的数环：

**高斯整数 $\mathbb{Z}[i]$**：$K = \mathbb{Q}(i)$，$\mathcal{O}_K = \mathbb{Z}[i] = \{a + bi : a, b \in \mathbb{Z}\}$，范 $\mathrm{N}(a + bi) = a^2 + b^2$。它是欧几里得整环，唯一分解成立，素数在其中的分裂刻画了两平方和定理。

**艾森斯坦整数 $\mathbb{Z}[\omega]$**：$\omega = \frac{-1+\sqrt{-3}}{2}$ 是本原三次单位根（$\omega^2 + \omega + 1 = 0$），$\mathcal{O}_{\mathbb{Q}(\omega)} = \mathbb{Z}[\omega]$，范 $\mathrm{N}(a + b\omega) = a^2 - ab + b^2$。<span class="marginnote">艾森斯坦整数是证明「$p$ 能写成 $a^2 - ab + b^2$」以及费马大定理 $n = 3$ 情形的天然舞台——先有数环，才有后续一切数论结论。</span>

**二次域 $\mathbb{Q}(\sqrt{d})$**：$d$ 是平方因子不等于 $1$ 的整数。数环由 $d \bmod 4$ 决定，见下节公式解析。

## 4 公式解析：二次域的数环

**数环的形状完全由 $d$ 模 $4$ 的余数决定**，这是本节最值得记的公式：

$$
\mathcal{O}_{\mathbb{Q}(\sqrt{d})} =
\begin{cases}
\mathbb{Z}[\sqrt{d}], & d \not\equiv 1 \pmod 4 \\[2mm]
\mathbb{Z}\!\left[\frac{1+\sqrt{d}}{2}\right], & d \equiv 1 \pmod 4
\end{cases}
$$

三步拆解它为什么必然长这样：

- **第一步，写出通式**：$\mathbb{Q}(\sqrt{d})$ 的元素写 $a + b\sqrt{d}$，$a, b \in \mathbb{Q}$。它是代数整数当且仅当它的**极小多项式** $x^2 - 2ax + (a^2 - b^2 d)$ 是首一整系数多项式，即
$$2a \in \mathbb{Z}, \qquad a^2 - b^2 d \in \mathbb{Z}.$$
- **第二步，找候选**：若 $a, b \in \mathbb{Z}$ 当然行；但 $a = b = \frac12$ 时，$2a = 1 \in \mathbb{Z}$，而 $a^2 - b^2d = \frac{1 - d}{4}$。要它也是整数，必须 $d \equiv 1 \pmod 4$。于是「半整数候选 $\frac{1+\sqrt{d}}{2}$」只在 $d \equiv 1 \pmod 4$ 时合格。
- **第三步，拼图**：两类情形各自生成整个数环（还需证明没有别的候选，用 $\mathcal{O}_K$ 是自由 $\mathbb{Z}$-模、秩为 $2$ 收口）。判别式随之是 $d_K = d$（当 $d \equiv 1 \pmod 4$）或 $d_K = 4d$（当 $d \not\equiv 1 \pmod 4$）。

**例**：$d = -5$，$-5 \equiv 3 \pmod 4$，故 $\mathcal{O}_{\mathbb{Q}(\sqrt{-5})} = \mathbb{Z}[\sqrt{-5}]$，判别式 $-20$——这正是下一节「唯一分解失败」的主角。<span class="marginnote">而 $d = -3$ 时 $\mathcal{O} = \mathbb{Z}[\frac{1+\sqrt{-3}}{2}] = \mathbb{Z}[\omega]$：数环「自动」长出了 $\frac{1+\sqrt{-3}}{2}$，因为 $d \equiv 1 \pmod 4$。同样的道理，$d = 5$ 的实数情形得到黄金比例 $\frac{1+\sqrt{5}}{2}$。</span>

## 5 例子与练习：从数环看数论

**练习 1（整数性判定）**：判断 $\frac{3+\sqrt{5}}{2}$ 是否代数整数。它的极小多项式是

$$
x^2 - 3x + 1 = 0
$$

首一整系数，故它是代数整数——事实上 $\frac{3+\sqrt{5}}{2} = 1 + \frac{1+\sqrt{5}}{2} \in \mathbb{Z}[\frac{1+\sqrt{5}}{2}]$，正落在 $\mathbb{Q}(\sqrt5)$ 的数环里。

**练习 2（范与单位）**：$\mathbb{Z}[\sqrt{-3}]$ 中 $\mathrm{N}(2) = 4$、$\mathrm{N}(1+\sqrt{-3}) = 4$，但 $1+\sqrt{-3}$ **不是**单位（范不为 $\pm 1$）。注意 $\mathbb{Z}[\sqrt{-3}]$ 并不是 $\mathbb{Q}(\sqrt{-3})$ 的整数环——因为 $-3 \equiv 1 \pmod 4$，真正的数环是 $\mathbb{Z}[\omega]$，它有 $6$ 个单位，而 $\mathbb{Z}[\sqrt{-3}]$ 只有 $2$ 个。**「数环选对，单位才数得对」**，这直接影响到类数公式里 $w_K$ 的取值。

**练习 3（二次域速查表）**：

| $d$ | $d \bmod 4$ | $\mathcal{O}_K$ | $d_K$ | 典型单位 |
| --- | --- | --- | --- | --- |
| $-1$ | $3$ | $\mathbb{Z}[i]$ | $-4$ | $\pm 1, \pm i$ |
| $-3$ | $1$ | $\mathbb{Z}[\omega]$ | $-3$ | $\pm 1, \pm\omega, \pm\omega^2$ |
| $5$ | $1$ | $\mathbb{Z}[\frac{1+\sqrt{5}}{2}]$ | $5$ | $(\frac{1+\sqrt5}{2})^k$ |
| $6$ | $2$ | $\mathbb{Z}[\sqrt{6}]$ | $24$ | $(5+2\sqrt{6})^k$ |

**辨析｜易错点：** 「$d \equiv 1 \pmod 4$」与「$d$ 平方自由」是两把不同的钥匙：前者决定数环是否含「半整数」$\frac{1+\sqrt d}{2}$，后者保证 $\sqrt d \notin \mathbb{Q}$、$K$ 真是二次扩张。二次域分类必须两个条件一起用。

## 6 小结

- **代数整数** = 满足首一整系数方程的复数；首一条件是「整性」的保证。
- **数域 $K$** = $\mathbb{Q}$ 的有限扩张；**数环 $\mathcal{O}_K$** = $K$ 中全体代数整数。
- **$\mathcal{O}_K$ 是秩 $n = [K:\mathbb{Q}]$ 的自由 $\mathbb{Z}$-模**，有整基；迹、范是核心不变量，且 $\alpha$ 是单位 $\iff \mathrm{N}(\alpha) = \pm 1$。
- 二次域数环：$d \not\equiv 1 \pmod 4$ 时 $\mathbb{Z}[\sqrt{d}]$，$d \equiv 1 \pmod 4$ 时含 $\frac{1+\sqrt{d}}{2}$。

在下一节，我们将看到 $\mathbb{Z}[\sqrt{-5}]$ 里 $6 = 2 \cdot 3 = (1+\sqrt{-5})(1-\sqrt{-5})$ 的分解无法用「元素」调和——戴德金用**理想**重新定义了唯一分解，这就是 Dedekind 整环理论。
