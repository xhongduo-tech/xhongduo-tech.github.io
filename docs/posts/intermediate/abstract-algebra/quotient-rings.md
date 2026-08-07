---
title: 商环（Quotient Ring）
date: 2026-08-07
---

# 商环（Quotient Ring）

<div class="epigraph">
<p>商环把「按理想取同余」做成一台环机器——模 n 算术是它最早的化身。</p>
<footer>—— 自 题（商环课堂笔记）</footer>
</div>

<div class="article-byline">
<p>第二级 · 抽象代数 ｜ 杨子胥《近世代数》§8.3 ｜ 2026-08-07</p>
</div>

## 为什么从商环开始

群论有商群，环论就有**商环（quotient ring）**：把理想 $I$ 压成 0，得到商环 $R / I$。商环是环论的中枢构造——模 $n$ 算术 $\mathbb{Z}_n = \mathbb{Z} / n\mathbb{Z}$ 是它最早的例子，而「模多项式」$F[x] / \langle f \rangle$ 则是它最深刻的例子（它直接给出域扩张与有限域！）。

商环的「良定义」依赖理想的吸收性：因为理想吸收一切，陪集的乘法才不会因代表元的选择而崩塌。本节把商环的定义、良定义性的机制、自然同态与典型例子讲透——掌握了商环，第十篇「用 $F[x]/\langle f\rangle$ 造域」将水到渠成。

## 1 商环的定义

**商环（quotient ring）**：设 $R$ 是环，$I$ 是 $R$ 的理想。在加法陪集集合

$$
R / I = \{ a + I \mid a \in R \}
$$

上定义加法和乘法：

$$
(a + I) + (b + I) = (a + b) + I, \qquad (a + I)(b + I) = ab + I
$$

**定理：** $R / I$ 在这些运算下构成环，称为 $R$ 关于 $I$ 的**商环**。加法单位元是 $0 + I = I$，乘法单位元（若 $R$ 含幺且 $I \ne R$）是 $1 + I$。<span class="marginnote">商环的直觉：把理想 $I$ 整个「拍扁」成一个点（加法零元），$I$ 的陪集 $a + I$ 成为新环的元素。模 $n$ 算术正是 $I = n\mathbb{Z}$ 的情形——$a + I$ 就是剩余类 $\bar a$。商环把「同余」从整数推广到任意环与任意理想。</span>

**验证良定义的关键**：乘法 $(a+I)(b+I) = ab + I$ 必须不依赖代表元。设 $a' = a + i_1$、$b' = b + i_2$（$i_1, i_2 \in I$），则

$$
a'b' = ab + \underbrace{a i_2}_{\in I} + \underbrace{i_1 b}_{\in I} + \underbrace{i_1 i_2}_{\in I} \in ab + I
$$

每一步都要用吸收性：$a i_2 \in I$（$a \in R$ 乘 $I$ 元素）、$i_1 b \in I$。**若 $I$ 只是子环而非理想，这里的 $a i_2$ 可能跑出 $I$，乘法就会崩坏**——这就是「商环必须由理想而非子环做模」的原因。

## 2 自然同态与同余语言

与商群类似，商环配有自然同态。

**自然同态（natural homomorphism）**：$\pi : R \to R/I$，$\pi(a) = a + I$。它是满射环同态，$\ker \pi = I$。

**同余语言**：定义 $a \equiv b \pmod I \iff a - b \in I$。这是等价关系（$I$ 是加法子群），商环 $R/I$ 就是「模 $I$ 的同余类全体」——**$a \equiv b \pmod I$ 当且仅当 $\pi(a) = \pi(b)$**。<span class="marginnote">同余记号 $\pmod I$ 把整数同余推广到任意环：$a \equiv b \pmod I$ 意味着「$a, b$ 差一个 $I$ 的元素」。在 $\mathbb{Z}$ 里 $\pmod{n\mathbb{Z}}$ 就是熟悉的 $\pmod n$；在 $F[x]$ 里 $\pmod{\langle f \rangle}$ 是「模多项式」。商环把「同余」变成环结构本身，这是它最大的力量。</span>

**例：**
- $\mathbb{Z} / n\mathbb{Z} = \mathbb{Z}_n$：商环就是模 $n$ 整数环；
- $\mathbb{R}[x] / \langle x \rangle \cong \mathbb{R}$：把 $x$ 压成 0，多项式退化为常数（「代入 $x = 0$」的同态）；
- $\mathbb{R}[x] / \langle x^2 + 1 \rangle \cong \mathbb{C}$：把 $x^2 = -1$ 的关系加进去，得到复数（下节细说）——**商环是「加约束」的机器**。

## 3 例：从 R[x] 造出 C 与有限域

商环最惊艳的用途：**用多项式环的商造出新的域**。

**例（$\mathbb{R}[x]/\langle x^2 + 1 \rangle \cong \mathbb{C}$）：** 商环里 $x^2 + 1 = 0$，即 $x^2 = -1$。元素 $a + bx + I$ 在「$x$ 充当 $i$」的解读下就是复数 $a + bi$。加法、乘法都对应复数的加乘（$x^2$ 代换为 $-1$）。于是

$$
\mathbb{R}[x] / \langle x^2 + 1 \rangle \ \cong\ \mathbb{C}, \qquad a + bx + I \mapsto a + bi
$$

**「把关系 $x^2 = -1$ 注入环」= 商掉 $\langle x^2 + 1 \rangle$**——商环是「形式地加入方程」的代数机器。<span class="marginnote">$\mathbb{R}[x]/\langle x^2+1\rangle$ 造出 $\mathbb{C}$ 是商环最美的例子：不用「假设 $i$ 存在」，而是「在多项式环里商掉 $x^2+1$ 生成的关系」，$i$ 自动出现。第十篇域扩张正是这个构造的系统化——$F[x]/\langle f\rangle$ 在 $f$ 不可约时是域（第六节极大理想），有限域 $\mathbb{F}_{p^n}$ 就是这样造出来的。</span>

**例（有限域）**：$\mathbb{F}_2[x] / \langle x^2 + x + 1 \rangle$：$\mathbb{F}_2$ 上 $x^2 + x + 1$ 无根，商环有 4 个元素 $\{ 0, 1, x, x+1 \}$（系数在 $\mathbb{F}_2$），构成 4 阶域 $\mathbb{F}_4$——有限域的第一个非素例子。第十篇将系统展开。

## 4 公式解析：商环乘法良定义 a'b' ∈ ab + I

商环的一切都建立在「良定义」上，把这一步彻底拆开。

- **第一步，问题。** 乘法 $(a+I)(b+I) = ab + I$ 中，$a + I$ 有无数种代表元 $a' = a + i_1$。必须证明所有代表元给出同一个乘积陪集。

- **第二步，交叉项的归属。** 对 $a' = a + i_1$、$b' = b + i_2$：

$$
a'b' = (a + i_1)(b + i_2) = ab + a i_2 + i_1 b + i_1 i_2
$$

- **第三步，吸收性逐项收编。** $a i_2 \in I$（$a \in R$，$i_2 \in I$，左吸收）；$i_1 b \in I$（右吸收）；$i_1 i_2 \in I$（$I$ 是子环）。于是 $a'b' - ab \in I$，即 $a'b' \in ab + I$。$\checkmark$

- **第四步，为什么子环不够。** 若 $I$ 只是子环，$a i_2$（$a \notin I$）与 $i_1 b$ 都可能跑出 $I$——乘法良定义崩坏。**理想 vs 子环的区别，在这一步上变成「商环存在 vs 商环崩溃」的区别。**

## 5 商环能做什么：加约束、看余数、造新环

商环的三重角色，让它成为环论最通用的工具。

**角色一：加约束（造新结构）。** 商掉 $\langle x^2 + 1 \rangle$ 加上关系「$x^2 = -1$」；商掉 $n\mathbb{Z}$ 加上关系「$n = 0$」。**商环 = 在环上「立法」（加入方程）的形式化。**<span class="marginnote">「商环 = 立法」的直觉非常实用：$\mathbb{Z}[i] = \mathbb{Z}[x]/\langle x^2 + 1\rangle$（高斯整数）、$\mathbb{F}_p = \mathbb{Z}/p\mathbb{Z}$、$k[\epsilon]/\langle \epsilon^2\rangle$（对偶数/形式微分的舞台）——全都是「先造多项式环，再商掉想要的关系」。代数几何里几乎所有「有限维代数」都是商环。</span>

**角色二：看余数（取模）。** 商环 $R/I$ 的元素 $a + I$ 就是「$a$ 模 $I$ 的余数类」。$\mathbb{Z}_n$ 里算「模 $n$」，$F[x]/\langle f\rangle$ 里算「模 $f$」。**商环 = 模算术的通用框架。**

**角色三：造新环（从旧环出发）。** $R/I$ 常常比 $R$ 更小、更整齐、甚至成为域（$I$ 是极大理想时）。从 $\mathbb{Z}$ 造 $\mathbb{F}_p$、从 $\mathbb{R}[x]$ 造 $\mathbb{C}$、从 $\mathbb{F}_p[x]$ 造 $\mathbb{F}_{p^n}$——**商环是环论里「发明新环」的标准流水线。**

## 6 小结

- **商环** $R/I$：按理想取同余的陪集环；加法/乘法逐陪集定义。
- **良定义靠吸收性**：$a'i'$ 的交叉项全部被理想吸收；子环做模会崩坏。
- **自然同态** $\pi(a) = a + I$，$\ker \pi = I$；同余 $a \equiv b \pmod I$。
- **经典例**：$\mathbb{Z}/n\mathbb{Z} = \mathbb{Z}_n$、$\mathbb{R}[x]/\langle x^2+1\rangle \cong \mathbb{C}$、$\mathbb{F}_2[x]/\langle x^2+x+1\rangle = \mathbb{F}_4$。
- **三重角色**：加约束、看余数、造新环。

在下一节，我们把商环与同态焊成一条定理：**环同态基本定理与同构定理**。$R/\ker \varphi \cong \operatorname{Im} \varphi$——群论的同态基本定理，在环论里原样重演。
