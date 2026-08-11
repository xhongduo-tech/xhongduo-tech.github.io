---
title: 曲面的相交理论与曲面上的 Riemann-Roch
date: 2026-08-11
---

# 曲面的相交理论与曲面上的 Riemann-Roch

<div class="epigraph">
<p>相交理论把"两条曲线交于几点"变成精确的算术：数目由除子类唯一决定。</p>
<footer>—— 由安德烈 · 韦伊（André Weil）思想转述</footer>
</div>

<div class="article-byline">
<p>第二级 · 进阶数理 · 代数几何 ｜ 对标教材 ｜ 2026-08-11</p>
</div>

## 为什么从曲面相交理论继续

曲线上的 Riemann-Roch（第 12 篇）是"在一条线上数函数"。现在升维到**曲面**：在二维上"数两个除子交于几点"。这需要一套精确的相交理论：给定曲面 $X$ 上的两条曲线 $C, D$，定义它们的相交数 $C \cdot D \in \mathbb{Z}$，并证明它只依赖除子类（线性等价类），而不依赖具体的代表元——"移动一下曲线，交点数目不变"。

曲面相交理论是代数几何从"一维计数"到"任意维计数"的桥梁，也是第 14 篇 blow-up 最直接的用武之地：例外除子 $E$ 满足 $E^2 = -1$，这个"自交为负"的事实是整个负曲线理论的起点。本节末尾的**曲面 Riemann-Roch** 把 $Euler$ 示性数 $\chi(\mathcal{O}(D))$ 写成"$D$ 与 $D-K$ 的相交数 + 常数项"，是"计数几何"在二维上的最终形态，也是模空间理论（如 $\bar{M}_g$ 上的相交数）的预演。

## 1 相交数：定义与基本性质

**核心概念：相交数（intersection number）**：设 $X$ 是光滑射影曲面，$C, D$ 是 $X$ 上的曲线（有效除子），且它们的支集**没有公共不可约分支**。定义

$$C \cdot D = \sum_{P \in C \cap D} i(C, D; P)$$

其中 $i(C, D; P)$ 是 $C$ 与 $D$ 在 $P$ 处的**相交重数**（intersection multiplicity），由局部环定义：$i(C, D; P) = \dim_k \mathcal{O}_{X,P} / (f_C, f_D)$，$f_C, f_D$ 是定义 $C, D$ 的局部方程。<span class="marginnote">相交重数 = "两个局部方程一起张出的商环的维数"。横截相交（切空间横截）时 $i = 1$；相切时 $i \ge 2$。例：$\mathbb{A}^2$ 中 $y = x^2$ 与 $y = 0$ 在原点 $i = 2$（重根），直线 $y=0$ 与 $y=x$ 在原点 $i = 1$。</span>

**重点：相交数只依赖除子类。** 若 $C \sim C'$（线性等价）且支集仍无公共分支，则 $C \cdot D = C' \cdot D$。由此把定义线性扩展到任意除子：对除子类 $[C], [D]$ 定义 $[C] \cdot [D] \in \mathbb{Z}$，不依赖代表元。<span class="marginnote">这是相交理论的"第一定理"：$C \cdot D$ 由 $C, D$ 的<strong>类</strong>决定。证明思路："移动 $C$ 使其与 $D$ 横截"（用线性系的"一般成员"），或者用下文的 Euler 特征定义直接验证良定性。于是"数交点"变成了"除子类上的双线性形式"。</span>

**相交数的代数性质**（全部可验证）：

**对称性**：$C \cdot D = D \cdot C$。
- **双线性**：$(C + C') \cdot D = C \cdot D + C' \cdot D$。
- **与亏格相关**：对光滑曲线 $D$，有 $C \cdot D = \deg(\mathcal{O}(C)|_D)$（把"交 $D$"看成"把 $\mathcal{O}(C)$ 限制到 $D$ 上算次数"）。<span class="marginnote">最后一条把"相交"翻译成"线丛限制的次数"——这是计算交点数最常用的方式：$C \cdot D = \deg \mathcal{O}(C)|_D$，于是曲线上的 R-R（第 12 篇）可以参与曲面问题的计算。</span>

## 2 自交数与例外除子

**核心概念：自交数（self-intersection）**：对曲线 $C$，定义 $C^2 = C \cdot C$。因为"$C$ 与自己"没有定义（支集重合），必须先**把 $C$ 移动成 $C' \sim C$、与 $C$ 横截**，再算 $C \cdot C'$。<span class="marginnote">"自交"不是"自己与自己相交"（那无定义），而是"取 $C$ 的另一个一般代表与自己交"。$C^2$ 是重要的双有理<strong>不</strong>变量：blow-up 时它会变。曲线在射影平面里的 $C^2 = (\deg C)^2$（Bezout 特例）。</span>

**重点：例外除子的自交。** 对 blow-up $\varepsilon: \widetilde{X} \to X$（沿光滑点，曲面情形），例外除子满足

$$E^2 = -1$$

且 $E \cong \mathbb{P}^1$。<span class="marginnote">$E^2 = -1$ 是"例外除子必须被收缩"的精确表述：若 $E$ 有"另一个代表"能与自己横截相交，那交点数会是非负的；$E^2 = -1 < 0$ 说明 $E$ <strong>没有</strong>可移动的替代品——它被"锁死"为不可移动的负曲线。这正是 Castelnuovo 收缩定理（第 14 篇）的判据：$E^2 = -1$ 且 $E \cong \mathbb{P}^1$ 的曲线可以收缩回点。</span>

**辨析｜易错点：** "自交数可以是负的"并不违反几何直觉，因为 $C^2$ 依赖的是**线性等价类的代表选择**。$C^2 < 0$ 意味着"该曲线类里没有两条能横截分离的代表"。初学者常误以为 $C^2$ 是"$C$ 与其自身的相交"，从而无法接受负值。正确理解：**$C^2$ 是"该类的一般成员与另一一般成员的相交"**，负值反映"此类太刚性、无可移动代表"。

## 3 Hodge 指标定理与相交矩阵

相交形式在除子类群上是双线性型，其"符号"由 Hodge 指标定理控制：

**重点：Hodge 指标定理。** 设 $X$ 是光滑射影曲面，$H$ 是 $X$ 上的一个**丰富除子**（ample divisor，其线性系给出嵌入），则相交形式在"与 $H$ 正交"的超平面 $H^\perp = \{D : D \cdot H = 0\}$ 上是**负定**的。<span class="marginnote">直观：在"垂直于超平面方向"的超平面里，一切自交都是负的。$\mathbb{P}^2$ 的 $\operatorname{NS}$ 群由 $H$ 生成，$H^2 = 1 > 0$，而 $H^\perp = 0$ 平凡；blow-up 后 $\operatorname{NS}(\widetilde{\mathbb{P}^2})$ 由 $H, E$ 生成，矩阵 $\begin{pmatrix} 1 & 0 \\ 0 & -1 \end{pmatrix}$ 的符号恰好反映"一个正方向（$H$）+ 多个负方向"。</span>

**核心概念：Néron-Severi 群** $\operatorname{NS}(X)$：$X$ 上"数值等价"的除子类群（$D_1 \equiv D_2 \iff D_1 \cdot E = D_2 \cdot E$ 对一切曲线 $E$），它是有限秩自由 Abel 群，秩 $\rho(X)$ 称为 **Picard 数**。<span class="marginnote">$\rho(\mathbb{P}^2) = 1$；$\rho(\widetilde{\mathbb{P}^2}) = 2$（blow-up 加 1）。Picard 数是曲面最基本的"算术维度"。Hodge 指标定理说 $\operatorname{NS}(X) \otimes \mathbb{R}$ 上的相交形式有符号 $(1, \rho - 1)$——"一个时间方向，多个空间方向"的黎曼式类比。</span>

## 4 曲面上的 Riemann-Roch

**重点：曲面的 Riemann-Roch 定理。** 设 $X$ 是光滑射影曲面，$D$ 是任意除子，$K$ 是典范类，则

$$\chi(\mathcal{O}(D)) = \frac{D \cdot (D - K)}{2} + \chi(\mathcal{O}_X)$$

其中 $\chi(\mathcal{O}(D)) = \sum_{i=0}^2 (-1)^i \dim H^i(X, \mathcal{O}(D))$ 是 Euler 示性数。<span class="marginnote">曲线情形（第 12 篇）是 $\ell(D) - \ell(K-D) = \deg D + 1 - g$；曲面的"次数"换成"自交"（$D^2$）与"典范配对"（$D \cdot K$），"1 - g"换成 $\chi(\mathcal{O}_X)$。结构完全平行：<strong>Euler 示性数 = 二次的除子表达式 + 常数</strong>。</span>

**证明骨架（标准三步）**：
1. **关键引理**：$D$ 是有效除子时，由正合列 $0 \to \mathcal{O}_X(D-C) \to \mathcal{O}_X(D) \to \mathcal{O}_C(D) \to 0$（$C \subseteq X$ 光滑曲线）与曲线 R-R 推出 $\chi(\mathcal{O}(D)) = \chi(\mathcal{O}(D-C)) + C \cdot (D - C) + \chi(\mathcal{O}_C)$。
2. 归纳地"剥掉" $D$ 的不可约分支，化归到 $D = 0$。
3. 检查等式在加减一次除子时两边同变（二次型 $D \mapsto (D \cdot (D-K))/2$ 的差分恰是引理里的项），故对所有 $D$ 成立。<span class="marginnote">这条证明展示了曲面理论的典型节奏：<strong>用曲线理论（第 12 篇）作为"度量工具"</strong>，把曲面问题降维到曲线上。$C \cdot (D-C)$ 项把"降维"与"相交数"焊在一起。</span>

**应用：曲线上亏格的另一个来源。** 对光滑曲线 $C \subseteq X$，取 $D = C$ 代入曲面 R-R，并结合伴随公式 $K_C = (K_X + C)|_C$ 与 $\deg$ 定义，可得

$$\chi(\mathcal{O}_C) = 1 - g_C, \qquad g_C = 1 + \frac{C^2 + C \cdot K_X}{2}$$

**核心概念：Noether 公式。** 取 $D = 0$、并利用 $\chi(\mathcal{O}_X)$ 与拓扑 Euler 数的关系（Hodge 理论：$\chi(\mathcal{O}_X) = (c_1^2 + c_2)/12$），得到

$$\chi(\mathcal{O}_X) = \frac{K_X^2 + e(X)}{12}$$

其中 $e(X)$ 是拓扑 Euler 示性数、$K_X^2 = K \cdot K$ 是典范自交数。<span class="marginnote">Noether 公式把代数（$K_X^2$）、拓扑（$e(X)$）、算术（$\chi(\mathcal{O}_X)$）三个不变量用一条等式绑定。对 $\mathbb{P}^2$：$K^2 = 9$、$e = 3$，$\chi(\mathcal{O}) = 1$——三者一致。对 $\mathbb{P}^2$ 的 blow-up：$K^2$ 每次减 1、$e$ 每次加 1，$\chi$ 不变。</span>

## 5 公式解析：曲面 Riemann-Roch

$$
\chi(\mathcal{O}(D)) = \frac{D \cdot D - D \cdot K}{2} + \chi(\mathcal{O}_X)
$$

分三步拆解：

- **第一步，$\frac{D \cdot D}{2}$ 是"自交的贡献"**：当 $D$ 是"大"除子（次数高的嵌入类）时，$H^2$ 与 $H^1$ 消失，$\chi = \dim H^0$，公式给出"截面数 $\sim D^2/2$"——自交越大、截面越多。<span class="marginnote">对 $\mathbb{P}^2$、$D = dH$：$D^2 = d^2$、$D \cdot K = -3d$、$\chi(\mathcal{O}) = 1$，于是 $\chi(\mathcal{O}(d)) = (d^2 + 3d)/2 + 1 = \frac{(d+1)(d+2)}{2}$——正是次数 $d$ 的齐次多项式空间维数！曲面 R-R 自动复现 $\mathbb{P}^2$ 的经典计数。</span>
- **第二步，$-\frac{D \cdot K}{2}$ 是"典范的修正"**：$K$ 的"重量"把截面数压低（典范越正、$D \cdot K$ 越大、截面越少）。它与自交项一起构成"二次型 $D \mapsto (D \cdot (D-K))/2$"，这个二次型的差分恰好对应"剥掉一个分支"时 $\chi$ 的变化——公式因此对全体除子成立。
- **第三步，$\chi(\mathcal{O}_X)$ 是"本底"**：$D = 0$ 时的值，由 Noether 公式 $= (K^2 + e)/12$ 控制。它是曲面的"常数项"，正如曲线情形的 $1 - g$。

一句话直觉：**曲面上的 Euler 示性数 = "自交二次型 + 典范修正 + 本底常数"**；它把二维的计数（交点、截面、亏格）全部折算进一条二次公式。

## 6 小结

- **相交数** $C \cdot D$：支集无公共分支时按局部重数求和；只依赖除子类，对称、双线性。
- **自交数** $C^2 = C \cdot C'$（$C' \sim C$ 一般代表）；**例外除子** $E^2 = -1$，负自交 = 刚性。
- **Hodge 指标定理**：$H^\perp$ 上相交形式负定；**Néron-Severi 群**与 **Picard 数** $\rho$ 给出相交矩阵的骨架。
- **曲面 Riemann-Roch**：$\chi(\mathcal{O}(D)) = \frac{D \cdot (D-K)}{2} + \chi(\mathcal{O}_X)$；$D = dH \subseteq \mathbb{P}^2$ 时复现 $\frac{(d+1)(d+2)}{2}$。
- **Noether 公式**：$\chi(\mathcal{O}_X) = (K^2 + e)/12$；blow-up 下 $K^2 \to K^2 - 1$、$e \to e + 1$。

在下一节，我们以一座纪念碑收尾：**GAGA 原理**——代数几何与复解析几何的对应，把"代数对象与态射"翻译成"解析对象与全纯映射"。
