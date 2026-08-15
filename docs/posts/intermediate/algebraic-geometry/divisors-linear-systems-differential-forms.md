---
title: 除子、线性系与微分形式
date: 2026-08-07
---

# 除子、线性系与微分形式

<div class="epigraph">
<p>除子理论是算术与几何交会的第一条公路。</p>
<footer>—— 由安德烈 · 韦伊（André Weil）思想转述</footer>
</div>

<div class="article-byline">
<p>第二级 · 代数几何 ｜ Hartshorne, Algebraic Geometry (GTM 52) Ch. II §6-8 ｜ 2026-08-07</p>
</div>

## 为什么从除子继续

上一节我们用模层（线丛）给概形装了"线性代数"。但几何里还有另一族更"算术"的对象：**除子（divisor）**。一条曲线上，一个有重数的有限点集——"在这些点取这些阶数的零点/极点"——就是除子。它之所以重要，是因为它能**直接计数**：把"线丛的分类"翻译成"除子的加减"，而加减是有整数的。

本节三件事环环相扣：**Weil 除子与 Cartier 除子**（两种定义、在什么条件下一致）、**线性系**（把除子变成"簇的映射"：$|D|$ 给出到射影空间的态射）、以及**微分形式**（几何的"测度"层 $\Omega^1_X$ 与典范类 $K_X$）。典范类 $K_X$ 是整部代数几何最重要的不变量——Serre 对偶（第 11 篇）和 Riemann-Roch（第 12 篇）都以它为枢纽。对曲线的情形，除子理论还会给出 Riemann-Hurwitz 公式（第 13 篇）的原料。

## 1 Weil 除子与 Cartier 除子

**核心概念：Weil 除子（Weil divisor）**：设 $X$ 是局部 Noether 正规概形。$X$ 上的**素除子（prime divisor）**是余维 1 的闭子概形 $Y$（不可约）。**Weil 除子**是素除子的形式整数线性组合

$$D = \sum_Y n_Y \, Y, \qquad n_Y \in \mathbb{Z}，\text{只有有限多个非零}$$

全体 Weil 除子构成自由 Abel 群 $\operatorname{Div} X$。<span class="marginnote">对一条光滑曲线 $X$，素除子就是"点"，所以 $\operatorname{Div} X$ = "以点为生成元的自由 Abel 群" = "带重数的点集"。例如 $\mathbb{P}^1$ 上 $D = 3[0] - [\infty]$ 表示"在 0 处有 3 阶零点、在无穷远处有 1 阶极点"。加减有整数，故可直接计数。</span>

**核心概念：Cartier 除子（Cartier divisor）**：用局部数据定义的除子——由"局部上形如 $\{ \varphi_i = 0 \}$ 的零点集"粘合而成：一组 $(U_i, f_i)$，$f_i$ 是 $U_i$ 上的非零因子，且 $f_i / f_j$ 在 $U_i \cap U_j$ 上是单位。<span class="marginnote">Weil 除子"记重数"，Cartier 除子"记方程"。对光滑（正规）概形两者一致；但奇异概形上 Cartier 除子比 Weil 除子"更细"。后续我们主要用 Cartier 除子（因为它与线丛一一对应），并把两者在光滑情形等同。</span>

**主除子（principal divisor）**：对有理函数 $f \in K(X)^\times$，其**除子** $\operatorname{div}(f) = \sum \operatorname{ord}_Y(f) \, Y$（在各余维 1 子簇处的阶数）。两个除子 $D \sim D'$（**线性等价**）当且仅当 $D - D' = \operatorname{div}(f)$。商群

$$\operatorname{Cl} X = \operatorname{Div} X / \sim$$

称为 **Weil 除子类群**。<span class="marginnote">线性等价是"差一个有理函数的零点/极点"——正如"差一个全纯函数的零极点"是解析几何里的同伦。$\operatorname{Cl} \mathbb{P}^n_k = \mathbb{Z}$：任意除子线性等价于 $d \cdot H$（$H$ 为超平面）。</span>

## 2 除子 ↔ 线丛：Picard 群统一

**重点定理：Cartier 除子与线丛一一对应。** 每个 Cartier 除子 $D = \{(U_i, f_i)\}$ 对应一个可逆层 $\mathcal{O}(D)$：$\mathcal{O}(D)|_{U_i} = \frac{1}{f_i} \mathcal{O}_{U_i} \subseteq K$。于是

$$\operatorname{Pic} X \cong \{ \text{Cartier 除子的线性等价类} \}$$

在 $X$ 光滑时，这进一步给出 $\operatorname{Cl} X \cong \operatorname{Pic} X$。<span class="marginnote">对光滑概形，三个概念——Weil 除子类群、Cartier 除子类群、Picard 群（线丛类群）——三者合一。$\operatorname{Pic} \mathbb{P}^n = \mathbb{Z}$ 由 $\mathcal{O}(1)$ 生成，对应"超平面" $H$；所以"线的次数" $d$ 就对应 $dH$ 或 $\mathcal{O}(d)$。线丛理论 = 除子理论，从此不再分家。</span>

**核心概念：由除子定义的模层 $\mathcal{O}_X(D)$**：对 Cartier 除子 $D$，$\mathcal{O}(D)$ 的截面是"以 $D$ 为下界的有理函数"：$f$ 是 $\mathcal{O}(D)(U)$ 的截面 ⟺ $\operatorname{div}(f) + D \ge 0$ 在 $U$ 上。对曲线情形，$\mathcal{O}(D)$ 的整体截面是

$$H^0(X, \mathcal{O}(D)) = \{ f \in K(X) \mid \operatorname{div}(f) + D \ge 0 \}$$

**辨析｜易错点：** "$\operatorname{div}(f) + D \ge 0$"的意思不是"$f$ 在 $D$ 的支集上无极点"，而是"$f$ 的极点阶数不超过 $D$ 给出的容许极点"。例：$D = P$（点），则 $\mathcal{O}(P)$ 的截面 = 至多在 $P$ 处有 1 阶极点、其余地方正则的有理函数——包括常数函数与 $1/(x-P)$ 型函数。初学者常见错误：把 $\ge 0$ 读成"无极点"，实际上它读作"极点不超预算"。

## 3 线性系与到射影空间的映射

**核心概念：完全线性系（complete linear system）**：对除子 $D$，令

$$|D| = \{ E \ge 0 \mid E \sim D \} = \mathbb{P}(H^0(X, \mathcal{O}(D)))$$

它是"所有与 $D$ 线性等价的非负除子"组成的射影空间。$|D|$ 里取一个线性子空间（射影子空间）$\mathfrak{d} \subseteq |D|$，称为**线性系（linear system）**。<span class="marginnote">直觉：$|D|$ 是"所有以 $D$ 为"最大容许极点"、且次数恰为 $\deg D$ 的有效除子"组成的集合。它是个射影空间，因为 $H^0(X, \mathcal{O}(D))$ 是有限维向量空间（凝聚层的有限性！）。</span>

**重点：线性系给出态射。** 设 $\dim |D| = n$，取基 $f_0, \dots, f_n \in H^0(X, \mathcal{O}(D))$，定义

$$\varphi_{|D|}: X \dashrightarrow \mathbb{P}^n, \qquad P \longmapsto [f_0(P) : \cdots : f_n(P)]$$

当 $|D|$ 无基点（各 $f_i$ 不同时为零）时，$\varphi_{|D|}$ 是定义在 $X$ 上的态射。<span class="marginnote">这是"给簇配备一个到射影空间的嵌入/态射"的标准机器：<strong>线性系把簇'推'进射影空间</strong>。对 $\mathbb{P}^n$ 上的 $\mathcal{O}(d)$，$\varphi_{|\mathcal{O}(d)|}$ 就是 $d$-重 Veronese 嵌入 $\mathbb{P}^n \hookrightarrow \mathbb{P}^{N}$。线性系的维数控制态射的维数，基点的位置控制态射的定义域。</span>

## 4 微分形式与典范类

**核心概念：微分形式层（sheaf of differentials）**：态射 $f: X \to Y$ 的**相对微分** $\Omega^1_{X/Y}$ 是满足"导子与模块"泛性质的 $\mathcal{O}_X$-模层：对每个 $\mathcal{O}_X$-模 $\mathcal{F}$，

$$\operatorname{Der}_Y(\mathcal{O}_X, \mathcal{F}) \cong \operatorname{Hom}_{\mathcal{O}_X}(\Omega^1_{X/Y}, \mathcal{F})$$

对仿射情形 $X = \operatorname{Spec} A$、$Y = \operatorname{Spec} B$，$\Omega^1_{A/B} = \bigoplus A \, da_i / \text{(Leibniz 与 B-线性关系)}$。<span class="marginnote">这就是微积分"微分形式"的代数定义：$\Omega^1$ 由"形式微分" $df$ 生成，满足 $d(fg) = f \, dg + g \, df$ 与 $d(\text{常数}) = 0$。$k$ 上的 $\mathbb{A}^n$ 情形 $\Omega^1 \cong \mathcal{O}^{\oplus n}$，$dx_1, \dots, dx_n$ 是基。</span>

**核心概念：典范类（canonical class）** ：设 $X$ 是 $n$ 维光滑簇。令 $\omega_X = \Omega^n_{X/k} = \wedge^n \Omega^1_{X/k}$（最高次微分形式）。若 $X$ 射影，$\omega_X$ 是秩 1 局部自由层（线丛），其对应的除子类记作 $K_X$，称为**典范类**。<span class="marginnote">$K_X$ 是 $X$ 的最重要内在不变量：它不依赖任何嵌入，只由 $X$ 自身决定。对 $X = \mathbb{P}^n$，$K_{\mathbb{P}^n} = -(n+1)H$，所以 $\deg K_{\mathbb{P}^n} \lt 0$——"负典范"是 Fano 型的标志。对亏格 $g$ 的光滑曲线 $C$，$\deg K_C = 2g - 2$——这条式子将在 Riemann-Hurwitz（第 13 篇）与 Riemann-Roch（第 12 篇）中反复出现。</span>

**重点：伴随公式（adjunction formula）。** 对光滑曲面上的光滑曲线 $C$（或更一般地，子簇 $Y \subseteq X$ 的余维 1 情形），有

$$K_C = (K_X + C)|_C$$

即"子簇的典范 = 母空间的典范加子簇自身再限制"。<span class="marginnote">伴随公式是计算典范类的主力工具：$\mathbb{P}^2$ 里一条次数 $d$ 的光滑曲线 $C$，$K_{\mathbb{P}^2} = -3H$、$C = dH$、$C \cdot C = d^2$，于是 $\deg K_C = d(d-3)$，故 $g = \frac{(d-1)(d-2)}{2}$——这正是"平面曲线亏格 = 次数的一次函数"的著名公式。</span>

## 5 公式解析：典范类与亏格

$$
\deg K_C = 2g - 2, \qquad g = \frac{(d-1)(d-2)}{2} \text{（平面次数 } d \text{ 光滑曲线）}
$$

分三步拆解：

- **第一步，$\deg K_C$ 从哪来**：$K_C$ 是线丛 $\omega_C = \Omega^1_C$ 的除子类。"次数" = "对切丛的 Euler 类计数"。用 Riemann-Roch 的雏形或 Gauss-Bonnet 类比：$\deg \omega_C = 2g - 2$，其中 $g$ 是亏格（拓扑上"洞的个数"，代数上 $\dim H^0(C, \Omega^1_C)$）。<span class="marginnote">亏格最直观的图景是拓扑的：亏格 $g$ 曲线 = "有 $g$ 个洞的面包圈"。代数定义 $\dim H^0(C, \omega_C)$ 在复几何里 = "全纯 1-形式空间的维数"，由 Hodge 理论等于拓扑亏格——"洞"与"全纯微分"在此统一。</span>
**第二步，为什么平面曲线亏格是 $d$ 的二次式**：由伴随公式 $\deg K_C = (K_{\mathbb{P}^2} + C)\cdot C = (-3H + dH)\cdot dH = d(d-3)$，代入 $2g - 2 = d(d-3)$ 解出 $g = \frac{(d-1)(d-2)}{2}$。<span class="marginnote">具体数：直线 $d=1$ 给 $g=0$（$\mathbb{P}^1$，无洞）；椭圆曲线 $d=3$ 给 $g=1$（一个洞）；$d=4$ 给 $g=3$（三个洞）。"次数越高、洞越多"——几何复杂度随代数次数二次增长。</span>
- **第三步，为什么这对整部理论重要**：$\deg K_C = 2g-2$ 是把"拓扑信息（$g$）"与"代数信息（次数）"绑定的第一根钉子。Riemann-Roch（第 12 篇）与 Riemann-Hurwitz（第 13 篇）都从这里延伸：前者用 $K_C$ 给所有线丛的截面数配平，后者用 $K_C$ 沿态射的拉回关系计算分支。

一句话直觉：**典范类 = "簇自身的微分几何性质"的代数封装**；$\deg K_C = 2g - 2$ 说"洞越多，典范越肥"。

## 6 对照表：除子理论全家福

| 概念 | 定义 | 一句话直觉 | 关键公式 |
| --- | --- | --- | --- |
| Weil 除子 | 素除子的整数线性组合 | 记重数 | $D = \sum_Y n_Y Y$ |
| Cartier 除子 | 局部 $(U_i, f_i)$ 粘合 | 记方程 | $\operatorname{Pic} X \cong$ Cartier 类群 |
| 主除子 | $\operatorname{div}(f)$ | 有理函数的零点/极点 | $D \sim D' \iff D - D' = \operatorname{div}(f)$ |
| 线性系 | $|D|$ 的子射影空间 | 把簇推进射影空间 | $\varphi_{|D|}: X \dashrightarrow \mathbb{P}^n$ |
| 典范类 | $K_X$（$\omega_X$ 的类） | 簇自身的内在不变量 | $\deg K_C = 2g - 2$ |
| 微分形式 | $\Omega^1_{X/k}$ 的楔积 | 形式微分的代数化 | $\omega_X = \wedge^n \Omega^1_{X/k}$ |

**数值算例：$\mathbb{P}^2$ 上次数 $d$ 光滑曲线的亏格。** 由伴随公式 $\deg K_C = d(d-3)$，结合 $\deg K_C = 2g - 2$，解出 $g = (d-1)(d-2)/2$。逐次代入：$d=1$（直线）给 $g=0$；$d=2$（二次曲线）给 $g=0$；$d=3$（三次曲线）给 $g=1$——这正是椭圆曲线；$d=4$ 给 $g=3$；$d=5$ 给 $g=6$。亏格随次数二次增长，"次数越高、洞越多"被写成一条精确的代数公式。<span class="marginnote">用除子语言复述同一件事：$D = dH$ 的线性系 $|dH|$ 给出 $d$-重 Veronese 嵌入，$\dim |dH| = \binom{d+2}{2} - 1$；而这条曲线在平面里的相交数 $C^2 = d^2$ 由 $H^2 = 1$ 决定——除子理论让"画一条曲线"变成"算一个类"。</span>

**辨析｜易错点：** Weil 除子与 Cartier 除子在光滑概形上一致，但符号别混用："$\operatorname{Div} X$"是 Weil 除子群、"$\operatorname{Cl} X$"是 Weil 类群、"$\operatorname{Pic} X$"是 Cartier/线丛类群。在奇异概形上 $\operatorname{Cl} X$ 与 $\operatorname{Pic} X$ 可以不同（相差余维 ≥ 2 的缺陷），判断用哪个版本，先问"$X$ 是否正规/光滑"。

## 7 小结

- **Weil 除子**：余维 1 子簇的整数线性组合；**Cartier 除子**：局部方程系统；光滑时两者一致。
- **线性等价 / 除子类群**：$\operatorname{Cl} X = \operatorname{Div}/\!\sim$；**Picard 群** $\cong$ Cartier 除子类群（光滑时还 $\cong \operatorname{Cl} X$）。
- **$\mathcal{O}(D)$ 的截面** = "极点不超预算"的有理函数：$\operatorname{div}(f) + D \ge 0$。
- **线性系** $|D| = \mathbb{P}(H^0(X, \mathcal{O}(D)))$，无基点时给出态射 $\varphi_{|D|}: X \to \mathbb{P}^n$。
- **微分形式** $\Omega^1_{X/k}$、**典范层** $\omega_X = \Omega^n$、**典范类** $K_X$；伴随公式 $K_C = (K_X + C)|_C$；$\deg K_C = 2g - 2$。

在下一节，我们进入上同调的世界：**层上同调与 Čech 上同调**——把"正合列被取截面后破损"这件事系统度量，并证明射影空间上同调的可计算性与有限性。
