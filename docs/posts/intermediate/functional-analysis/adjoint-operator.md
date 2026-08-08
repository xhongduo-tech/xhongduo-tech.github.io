---
title: 伴随算子的定义与性质
date: 2026-08-07
---

# 伴随算子的定义与性质

<div class="epigraph">
<p>伴随算子把「转置」从矩阵推广到无穷维——一个词足以概括 Hilbert 空间算子理论的对称之美。</p>
<footer>—— 冯 · 诺伊曼（John von Neumann），意译</footer>
</div>

<div class="article-byline">
<p>第二级 · 泛函分析 ｜ 程其襄《泛函分析》§4.11 ｜ 2026-08-07</p>
</div>

## 为什么需要伴随

线性代数里，矩阵 $A$ 有一个孪生兄弟——转置 $A^T$（复情形是共轭转置 $A^*$）。转置满足「$(Ax)\cdot y = x \cdot (A^T y)$」，它把内积里的算子「搬来搬去」。到了无穷维，转置该长什么样？**Riesz 表示定理给出了答案：伴随算子（adjoint operator）$T^*$ 由关系**

$$
\langle T x, y\rangle = \langle x, T^* y\rangle
$$

**唯一确定。** 伴随算子是谱理论的关键——自伴算子（$T = T^*$）是「实数」在算子世界的化身，它们的谱是实的、可对角化。本节系统建立伴随算子的基本性质。<span class="marginnote">「伴随」一词来自「伴随矩阵（adjugate）」，但更准确的理解是「共轭转置」的推广。有限维里 $T^*$ 的矩阵是 $A^* = \overline{A^T}$。在量子力学中，可观测量的数学对象就是<strong>自伴算子</strong>，它的伴随运算对应「转置 + 共轭」——第十章将展开。</span>

## 1 伴随算子的存在与唯一

**定理（伴随算子）**：设 $H_1, H_2$ 是 Hilbert 空间，$T \in \mathcal{B}(H_1, H_2)$。则存在唯一的有界线性算子 $T^* : H_2 \to H_1$ 满足

$$
\langle T x, y\rangle_{H_2} = \langle x, T^* y\rangle_{H_1}, \qquad \forall x \in H_1,\ y \in H_2
$$

**存在性**：固定 $y$，$x \mapsto \langle Tx, y\rangle$ 是 $H_1$ 上的连续线性泛函（范数 $\le \|T\|\|y\|$），由 Riesz 表示定理存在唯一 $T^*y \in H_1$ 使 $\langle x, T^*y\rangle = \langle Tx, y\rangle$。

**唯一性**：若两个算子都满足该关系，相减得 $\langle x, z\rangle = 0$ 对一切 $x$，取 $x = z$ 得 $z = 0$。<span class="marginnote">唯一的微妙点：证明中「$y \mapsto T^*y$」的线性性需要一点工作——因为内积对第二变量共轭线性，用极化恒等式或「线性组合验证」可以推出 $T^*$ 线性。这是初学者最易漏掉的一步验证。</span>

**核心要点：$T^*$ 的存在性完全来自 Riesz 表示定理**——「把线性泛函变回向量」这一步，是整个伴随理论的地基。

## 2 伴随的基本性质

伴随运算 $T \mapsto T^*$ 满足：

1. **反线性**：$(T + S)^* = T^* + S^*$，$(\alpha T)^* = \bar\alpha T^*$；
2. **对合**：$(T^*)^* = T$；
3. **乘积反序**：$(ST)^* = T^* S^*$；
4. **范数保持**：$\|T^*\| = \|T\|$，且 $\|T^*T\| = \|T\|^2$（$C^*$-等式）。

其中 $\|T^*T\| = \|T\|^2$ 最不平凡，证明用「先上界再下界」：

$$
\|T^*T x\| \le \|T^*\|\|Tx\| = \|T\|\|Tx\| \Rightarrow \|T^*T\| \le \|T\|^2
$$

反向：$\|Tx\|^2 = \langle Tx, Tx\rangle = \langle x, T^*Tx\rangle \le \|x\|\|T^*Tx\| \le \|x\|\|T^*T\|\|x\| = \|T^*T\|\|x\|^2$，故 $\|T\|^2 \le \|T^*T\|$。<span class="marginnote">$\|T^*T\| = \|T\|^2$ 叫 <strong>$C^*$-恒等式</strong>，它是 $C^*$-代数的定义性条件——冯 · 诺伊曼代数的核心工具。它把「算子范数」与「内积结构」锁定在一起：一个算子连同它的伴随，范数完全由 $T^*T$ 决定。</span>

## 3 自伴算子与正规算子

**定义**：

**自伴算子（self-adjoint）**：$T^* = T$（即 $\langle Tx, y\rangle = \langle x, Ty\rangle$ 对一切 $x,y$）。<span class="marginnote">在 $L^2$ 里，乘法算子 $M_\varphi f = \varphi f$（$\varphi$ 实值）自伴；微分算子 $i\frac{d}{dt}$ 在合适边界条件下自伴。自伴算子 = 实对称矩阵（复情形 Hermite 矩阵）的无穷维版。</span>
**正规算子（normal）**：$T^*T = TT^*$。自伴、酉（$T^*T = TT^* = I$）、非负（$\langle Tx,x\rangle \ge 0$）都是正规算子的特例。

**自伴算子的谱是实数**（第九章将证）：对 $\langle Tx,x\rangle = \langle x, Tx\rangle$，$\langle Tx,x\rangle$ 恒为实数。<span class="marginnote">非负算子（$\langle Tx,x\rangle \ge 0$）是「半正定矩阵」的推广；它有「平方根」$T = S^2$，且谱非负。这些在量子力学里对应「可观测量取值实数、能量非负」。</span>

**例**：有限维 Hermite 矩阵自伴，其特征值全为实数；酉矩阵正规，特征值全在单位圆上。这些「谱落在实轴/单位圆」的性质在无穷维保持——第九章谱理论的核心结论之一。

## 4 公式解析：核与值域的对偶关系

伴随算子与核/值域之间有优美的对偶关系：

$$
\ker T^* = (\operatorname{ran} T)^\perp, \qquad \overline{\operatorname{ran} T} = (\ker T^*)^\perp
$$

- **第一步，$\ker T^* = (\operatorname{ran}T)^\perp$**：$y \in \ker T^* \iff T^*y = 0 \iff \langle x, T^*y\rangle = 0\ \forall x \iff \langle Tx, y\rangle = 0\ \forall x \iff y \perp \operatorname{ran}T$。
- **第二步，取正交补**：对第一式两边取正交补，用 $(M^\perp)^\perp = \overline M$，得 $\overline{\operatorname{ran}T} = (\ker T^*)^\perp$。

**关键**：这条对偶关系是 **Fredholm 理论**（第八章）的基石：**方程 $Tx = b$ 可解（$b \in \operatorname{ran}T$）当且仅当 $b$ 正交于 $\ker T^*$**。它把「解的存在性」翻译成「正交性检验」——这是泛函分析解方程方法论的核心。<span class="marginnote">有限维说法：$Ax = b$ 有解当且仅当 $b$ 与 $A^T$ 的零空间正交。无穷维版本多了一个闭包记号（$\overline{\operatorname{ran}T}$），因为值域不一定闭——但若 $\operatorname{ran}T$ 闭（如紧算子 + Fredholm 理论的情形），则干净地有 $\operatorname{ran}T = (\ker T^*)^\perp$。</span>

## 5 伴随与投影、酉算子

**正交投影的刻画**（§4.6 的回归）：$P$ 是正交投影 ⟺ $P^2 = P$ 且 $P^* = P$。用伴随语言，「自伴幂等」精确刻画正交投影。

**酉算子**：$U^*U = UU^* = I$ 当且仅当 $U$ 是等距同构（保内积的双射）。酉算子保持一切几何结构，是 Hilbert 空间的自同构群（§4.9 的酉同构即此）。<span class="marginnote">傅里叶变换是酉算子（$L^2$ 到 $L^2$）：$\mathcal{F}^*\mathcal{F} = I$ 正是帕塞瓦尔等式的算子表述。量子力学里演化算子 $U(t) = e^{-iHt}$（$H$ 自伴）是酉的——它保持概率（范数），这是「量子演化保归一化」的数学根据。</span>

## 6 常见误区与反例汇总

**误区一：误以为 $T^*T = TT^*$ 总是成立**。只有正规算子才交换。一般算子 $T^*T \neq TT^*$。反例：移位算子 $S$，$S^*S = I$ 而 $SS^* \neq I$。

**误区二：把「对称」与「自伴」混为一谈**。对称只要求 $\langle Tx,y\rangle = \langle x,Ty\rangle$（定义域内）；自伴还要求定义域相同。无界情形两者天差地别（见 §5.7 与 §7 讨论）。

**误区三：忘记共轭线性**。$T \mapsto T^*$ 是共轭线性的：$(\alpha T)^* = \bar\alpha T^*$。在复空间里，$\alpha$ 变号。初学者常在「$(iT)^*$ 是什么」上出错：$(iT)^* = -iT^*$。

**一个标准计算**：对积分算子 $T_K f(s) = \int K(s,t)f(t)\,dt$，其伴随是 $T_K^* f(s) = \int \overline{K(t,s)}f(t)\,dt$（核共轭 + 交换变量）。自伴 ⟺ $K(s,t) = \overline{K(t,s)}$（Hermite 核）。

**例（乘法算子）**：$M_\varphi$ 的伴随是 $M_{\bar\varphi}$（乘共轭函数）。$M_\varphi$ 自伴 ⟺ $\varphi$ 实值。$M_\varphi$ 是酉 ⟺ $|\varphi| = 1$ 几乎处处。

**核心要点：伴随 = 共轭转置的推广**，所有「转置」的直觉都适用，但多了一层共轭——这是复空间的签名。

## 7 例题精讲：伴随算子的三个计算

**例题一：移位算子的伴随**。

- $S(x_1,x_2,\ldots) = (0,x_1,x_2,\ldots)$。
- $\langle Sx, y\rangle = \sum_{n\ge1} x_n \overline{y_{n+1}} = \langle x, S^*y\rangle$，故 $S^*y = (y_2,y_3,\ldots)$。
- $S^*S = I$ 但 $SS^* \neq I$（$S$ 等距不酉）。

**例题二：积分算子的伴随**。

- $T_K f(s) = \int K(s,t)f(t)\,dt$。
- $\langle T_K f, g\rangle = \iint K(s,t)f(t)\overline{g(s)}\,dt\,ds = \langle f, T_K^* g\rangle$。
- $T_K^* g(t) = \int \overline{K(s,t)}g(s)\,ds$——核共轭 + 交换变量。

**例题三：乘法算子的伴随**。

- $M_\varphi f = \varphi f$。$\langle M_\varphi f, g\rangle = \int \varphi f\bar g = \int f \overline{\bar\varphi g}$。
- $M_\varphi^* = M_{\bar\varphi}$。
- $M_\varphi$ 自伴 ⟺ $\varphi$ 实值；$M_\varphi$ 酉 ⟺ $|\varphi| = 1$。

**核心要点**：伴随的三个计算——移位、积分核、乘法——都是「把内积里的算子搬到另一边」+ 复共轭。

**辨析｜易错点：** 伴随是共轭线性：$(\alpha T)^* = \bar\alpha T^*$。复空间里 $\alpha$ 的共轭最易漏。


## 8 小结

- **伴随算子**：$\langle Tx, y\rangle = \langle x, T^*y\rangle$ 唯一确定；存在性靠 Riesz 表示定理。
- **基本性质**：反线性、对合、乘积反序、$\|T^*\| = \|T\|$、$C^*$-等式 $\|T^*T\| = \|T\|^2$。
- **自伴与正规**：$T^* = T$（实对称的推广，谱为实数）；$T^*T = TT^*$（正规，酉/非负为其特例）。
- **核值域对偶**：$\ker T^* = (\operatorname{ran}T)^\perp$，$\overline{\operatorname{ran}T} = (\ker T^*)^\perp$——Fredholm 理论的地基。
- **投影与酉**：自伴幂等 = 正交投影；$U^*U = I$ = 等距同构。

至此，第四章「内积空间与 Hilbert 空间」完成——我们拥有了最完整的几何武器库。在下一章，我们进入 Banach 空间理论的「三大基本定理」——**纲定理与 Baire 纲定理**，那里将首次看到完备性如何转化为「稠密性」的力量。
