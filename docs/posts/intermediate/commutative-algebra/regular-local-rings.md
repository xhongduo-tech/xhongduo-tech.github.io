---
title: 维数理论深化与正则局部环
date: 2026-08-11
---

# 维数理论深化与正则局部环

<div class="epigraph">
<p>正则局部环是交换代数与代数几何的会合点：纯代数定义与「光滑」直觉在此重合。</p>
<footer>—— 佐武一郎（Ichirō Satake）所引述的代数几何精神（此处意译）</footer>
</div>

<div class="article-byline">
<p>第二级 · 进阶数理 · 交换代数 ｜ 对标教材 ｜ 2026-08-11</p>
</div>

## 为什么从系统参数开始

上一节我们把维数定义成了素理想链的长度。但要做「局部几何」，还要回答：**在局部环里，维数最少需要几个「坐标」来描述？** 答案是**系统参数**（system of parameters）：用恰好 $d$ 个元素张出一个 $\mathfrak{m}$-准素理想。于是「维数」有了第三张脸：**最小生成元个数**。三张脸——链、参数、体积——互相印证，而它们咬合得最紧、最光滑的环，就是**正则局部环**。<span class="marginnote">「正则」的英文 regular 与正则函数（holomorphic 之 regular）同名同源：几何上「光滑点」的局部环正是正则的。Zariski 1940 年代用「嵌入维数 = 维数」定义正则局部环，把「光滑」第一次变成了纯代数概念。</span>

这一篇是维数理论的深水区：系统参数、嵌入维数、正则局部环的判别与惊人性质。它是连接《维数理论：Krull 维数与 Hilbert 函数》与后面《Koszul 复形》《Cohen–Macaulay》三篇的枢纽。

## 1 系统参数

设 $(R, \mathfrak{m})$ Noether 局部环，$d = \dim R$。

**系统参数（system of parameters）**：$d$ 个元素 $x_1, \dots, x_d$，使理想 $(x_1, \dots, x_d)$ 是 $\mathfrak{m}$-准素的（即 $R/(x_1,\dots,x_d)$ 是 Artin 环，维数 0）。

**重点：维数恰是「能张出 $\mathfrak{m}$-准素理想」所需的最少元素数。** 必要性由高度定理（第1篇《维数理论》）给出：$n$ 个生成元生成的理想，其极小素因子高度 ≤ $n$，所以 $n < d$ 张不出维数 0 的商。存在性则靠**主理想定理的反面**逐层剥：先在 $R$ 里选 $x_1$ 使 $\dim R/(x_1) = d - 1$，再在 $R/(x_1)$ 里选 $x_2$……递推构造出整组参数。<span class="marginnote">这个「逐层降维」的选参数套路是维数理论的标准工序，它说明系统参数一定存在，但<strong>不唯一</strong>：$k[x,y]_{(x,y)}$ 里 $\{x, y\}$ 与 $\{x, y - x^2\}$ 都是系统参数。不同参数系的选择正是「局部坐标」的选择。</span>

例子：$R = k[x,y]_{(x,y)}$，$d = 2$，$\{x, y\}$ 是系统参数（$R/(x,y) = k$ Artin）；$R = \mathbb{Z}_{(p)}$，$d = 1$，$\{p\}$ 是系统参数；$R = k[x,y]_{(x,y)}/(xy)$，$d = 1$，参数系可选 $\{x + y\}$。

## 2 嵌入维数与正则局部环

考虑**切线空间**：$\mathfrak{m}/\mathfrak{m}^2$ 是剩余域 $k$ 上的向量空间，其维数叫**嵌入维数（embedding dimension）**

$$\operatorname{embdim} R = \dim_k \mathfrak{m}/\mathfrak{m}^2.$$

由 Nakayama 引理，$\mathfrak{m}$ 的最小生成元个数恰是 $\dim_k \mathfrak{m}/\mathfrak{m}^2$；而生成一个 $\mathfrak{m}$-准素理想至少要 $d$ 个，所以：

**重点：$d = \dim R \leq \operatorname{embdim} R$，等号成立时称 $R$ 是正则局部环（regular local ring）。** 即「光滑点」=「切空间维数恰好等于环维数」。

**正则局部环（regular local ring）**：$\dim R = \dim_k \mathfrak{m}/\mathfrak{m}^2$ 的 Noether 局部环。

典型例子：
- $R = k[x_1,\dots,x_n]_{(x_1,\dots,x_n)}$ 与 $k[[x_1,\dots,x_n]]$ 都是正则的（$\mathfrak{m}/\mathfrak{m}^2$ 以 $\bar{x}_i$ 为基，维数 $n$）。
- $R = \mathbb{Z}_{(p)}$ 正则（$\dim 1$，$\mathfrak{m}/\mathfrak{m}^2 = \mathbb{F}_p \cdot \bar{p}$）。
- **奇点**：$R = k[x,y]_{(x,y)}/(xy)$ 有 $\dim R = 1$，但 $\mathfrak{m}/\mathfrak{m}^2$ 以 $\bar{x}, \bar{y}$ 为基、维数 2——节点处切空间「有两个方向」，不是光滑点，非正则。
- 尖端 $R = k[x,y]_{(x,y)}/(y^2 - x^3)$：$\dim 1$，$\mathfrak{m} = (\bar{x}, \bar{y})$，$\mathfrak{m}^2 = (\bar{x}^2, \bar{x}\bar{y})$（$\bar{y}^2 = \bar{x}^3 \in \mathfrak{m}^2$），故 $\dim_k \mathfrak{m}/\mathfrak{m}^2 = 2 > 1$——非正则。<span class="marginnote">几何直觉：正则 ⇔ 切空间没有「多余方向」。节点 $xy=0$ 与尖端 $y^2=x^3$ 的切空间都是二维，而曲线本身一维——「切空间比曲线胖」正是奇点。</span>

## 3 正则局部环的惊人性质

正则性看似只是「两个数相等」，代价却极其丰厚。列几条最著名的：

**正则局部环是整环。** 这一点并不平凡（一般 Noether 局部环可以是零因子环），证明用完备化 + $\mathfrak{m}$-进拓扑。

**Auslander–Buchsbaum 定理（1959）**：**正则局部环是 UFD**（唯一分解整环）。<span class="marginnote">这个定理证明了库默尔、戴德金时代就相信的几何断言：光滑点处函数环是 UFD。1962 年他们又证正则局部环上每个有限生成模有有限自由分解——把「光滑」写成「同调维数有限」，即下一篇《Koszul 复形》的主角。</span>

**正则性在完备化、局部化下保持**：$R$ 正则 ⇔ $\widehat{R}$ 正则；$R_{\mathfrak{p}}$ 也正则。

**Jacobian 判别法（几何版）**：仿射空间里，由 $f_1, \dots, f_r$ 定义的子簇在点 $p$ 处光滑（坐标环在该点的局部化正则）当且仅当 Jacobi 矩阵 $(\partial f_i/\partial x_j)(p)$ 秩为 $r$——这正是多元微积分「隐函数定理」的代数翻版，与第一级《多元微积分》的隐函数定理呼应。<span class="marginnote">Jacobian 判别法是「微分学进代数」的经典通道：光滑点 ⇔ 切空间法向满秩 ⇔ 正则局部环。它把微积分的「正则点」与交换代数的「正则环」绑成同义反复。</span>

**辨析｜易错点：** 正则 ≠ 多项式环。$R = \mathbb{Z}_{(p)}$、$k[[x]]$ 都正则但都不是「有限生成多项式环」；反过来 $k[x]_{(x)}$ 是多项式环的局部化，正则。「正则」是**局部**性质，看 $\mathfrak{m}/\mathfrak{m}^2$；「多项式/幂级数/局部化」是实现方式。判断一个环正不正则，永远先算 $d$ 与 $\dim_k \mathfrak{m}/\mathfrak{m}^2$ 两个数。

## 4 公式解析：为什么「刚好够用」等价于光滑

把判定式写开。设 $R$ Noether 局部，$d = \dim R$，则

$$\dim R \;\leq\; \dim_k \mathfrak{m}/\mathfrak{m}^2 \;\leq\; \text{（$\mathfrak{m}$ 的生成元个数）}, \qquad R \text{ 正则} \iff \text{左端取等号}.$$

- **第一步，第一个 $\leq$**：$n$ 个生成元最多张出高度 ≤ $n$ 的理想，而 $\mathfrak{m}$-准素理想的每个生成元系都 ≥ $d$ 个（高度定理）。Nakayama 又告诉我们生成元个数 = $\dim_k \mathfrak{m}/\mathfrak{m}^2$，故 $d \leq \dim_k \mathfrak{m}/\mathfrak{m}^2$。
- **第二步，等号的意义**：$\dim_k \mathfrak{m}/\mathfrak{m}^2 = d$ 时，$\mathfrak{m}$ 由恰好 $d$ 个元素生成——**系统参数同时生成极大理想**。$k[x_1,\dots,x_n]$ 局部化处，参数系 $\{x_1,\dots,x_n\}$ 本身生成 $\mathfrak{m}$。
- **第三步，为什么是光滑**：$d$ 个生成元就是 $d$ 个「局部坐标」；$\mathfrak{m}/\mathfrak{m}^2$ 是坐标的线性化——切空间。维数相等 = 切空间维数 = 环维数 = 没有多余自由度。奇点处（$xy$ 节点、$y^2 = x^3$ 尖端）生成元要多于 $d$，切空间「超维」。

**辨析｜易错点：** 「$\mathfrak{m}$ 由 $d$ 个元素生成」与「$R$ 正则」其实等价，但**前提是 $R$ Noether 局部**。非局部、非 Noether 时不要随便用这个等式。另外 $\dim_k \mathfrak{m}/\mathfrak{m}^2$ 是**剩余域上**的向量空间维数，与环自身的元素个数无关——$k[x]_{(x)}$ 无限元素、切空间却一维。

## 5 小结

- **系统参数**：$d$ 个元素张出 $\mathfrak{m}$-准素理想；$d$ = 所需最少个数 = 维数。
- **嵌入维数** $\operatorname{embdim} R = \dim_k \mathfrak{m}/\mathfrak{m}^2$；$\dim R \leq \operatorname{embdim} R$。
- **正则局部环** = 嵌入维数 = 维数；节点、尖端非正则。
- 正则环是整环、UFD（Auslander–Buchsbaum）、完备化与局部化下保持；几何上对应光滑点（Jacobian 判别法）。

在下一节，我们把「光滑」翻译成同调语言：**正则序列**与 **Koszul 复形**——正则性不再靠数切空间维数，而靠一个自由分解是否干净。
