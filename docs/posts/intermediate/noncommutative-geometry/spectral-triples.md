---
title: 谱三元组（Dirac 算子、距离公式、公理化表述）
date: 2026-08-17
---

# 谱三元组

<div class="epigraph">
<p>人能听出鼓的形状吗？</p>
<footer>—— 马克 · 卡茨（Mark Kac），《Can one hear the shape of a drum?》，1966</footer>
</div>

<div class="article-byline">
<p>第二级 · 非交换几何 ｜ GBVF《Elements of Noncommutative Geometry》Ch.10; Connes《Noncommutative Geometry》Ch.VI; Landi《An Introduction to Noncommutative Spaces》Ch.6 ｜ 2026-08-17</p>
</div>

## 为什么从谱三元组开始

前几节我们建立了非交换空间的拓扑（C\*-代数、K-理论）与同调（循环上同调）。但它们都缺少一样东西——**距离**。拓扑可以没有度量，但几何必须有。经典黎曼流形靠度规张量 $g_{\mu\nu}$ 定义距离；非交换空间没有点，更没有度规张量，距离从何而来？

Connes 的回答是**谱三元组（spectral triple）**：用 Dirac 算子 $\not\!\!D$ 的谱来编码全部度规信息。这正是卡茨问题的非交换版——「听出鼓的形状」——并且答案出人意料地肯定：对足够好的谱三元组，几何可以被完全重建。谱三元组同时是拓扑、同调与度量的统一体：它承上（K-同调与循环上同调的载体）、启下（《非交换流形的微分结构》与《局部指标公式》都是它的展开），是整个非交换几何的**定义性结构**。

## 1 从 Dirac 算子到距离

### 1.1 经典 Dirac 算子

设 $(M, g)$ 是紧黎曼流形，配自旋结构，$S$ 是旋量丛，$\nabla^S$ 是旋量联络。**Dirac 算子（Dirac operator）** 是作用在旋量截面上的一阶椭圆微分算子：

$$
\not\!\!D = -i\, \gamma^\mu \nabla^S_\mu, \qquad \not\!\!D: \Gamma(S) \longrightarrow \Gamma(S)
$$

其中 $\gamma^\mu$ 是 Clifford 代数的 $\gamma$ 矩阵，满足反对易关系 $\{\gamma^\mu, \gamma^\nu\} = 2 g^{\mu\nu}$。<span class="marginnote">Dirac 1928 年为写出电子的一阶相对论波动方程而发明 $\gamma$ 矩阵；它们在数学上给出旋量表示与 Clifford 代数。Dirac 算子是「求导的平方根」：$\not\!\!D^2 = \nabla^*\nabla + \frac{1}{4}R$（Lichnerowicz 公式），曲率以标量曲率 $R$ 出现。</span>

### 1.2 谱几何的现代形式

经典谱几何（Weyl 定律、热核展开、Atiyah–Singer 指标定理）说明：**一个流形的形状，完全由 Dirac 算子的谱决定**。Kac 的问题在弦振动的框架下是「听出形状」，在几何框架下就是「从谱读回度量」。

### 1.3 距离与算子的关键联系

关键观察：函数 $f \in C^\infty(M)$ 以乘法作用在旋量截面上，其交换子 $[\not\!\!D, f]$ 是**零阶算子**（乘法算子），范数 $\|[\not\!\!D, f]\|$ 正好是 $f$ 的梯度范数的 sup：

$$
\|[\,\not\!\!D, f\,]\| = \|\,df\,\|_\infty = \sup_{x \in M} |\nabla f|(x)
$$

这一条等式把「算子的范数」翻译成「函数的斜率」，是 Connes 距离公式的全部基础。

## 2 Connes 距离公式

**Connes 距离公式（Connes distance formula）**：设 $\varphi, \psi$ 是 $C^\infty(M)$ 上的两个态（如点态 $f \mapsto f(x)$、$f \mapsto f(y)$），定义

$$
d(\varphi, \psi) = \sup \big\{ |\varphi(f) - \psi(f)| : \|[\not\!\!D, f]\| \le 1 \big\}
$$

**公式解析：** 分三步拆解这条式子为何给出真实距离：

- **第一步**，约束 $\|[\not\!\!D, f]\| \le 1$ 等价于 $\|df\|_\infty \le 1$，即 $f$ 是「1-Lipschitz 函数」。这与经典度量几何中「距离 = 1-Lipschitz 函数对点值之差的 sup」完全一致。
- **第二步**，取 $\varphi(f) = f(x)$、$\psi(f) = f(y)$：$d(x,y) = \sup\{|f(x)-f(y)|: f\ \text{1-Lipschitz}\}$。由经典变分原理，这个 sup 正好等于 $x, y$ 之间的黎曼距离 $\mathrm{dist}_g(x, y)$——取 $f(z) = \mathrm{dist}_g(z, x)$ 即可达到上界。
- **第三步**，关键推广：公式右端只用到「代数 + 算子」，没有用到任何点！于是对**任何**谱三元组都能定义「态空间上的距离」——即使底层没有点。

**这就是非交换空间上的距离**：态扮演点的角色，距离由 Dirac 交换子的范数定义。对非交换环面等例子，这个距离公式给出与经典情形平行的几何解释（Connes–Rieffel 已证明其与形变量子化的关系）。

## 3 谱三元组的公理化定义

### 3.1 定义

**谱三元组（spectral triple）**：三元组 $(A, \mathcal{H}, D)$，其中

- $A$ 是含幺 $*$-代数，配备忠实表示在 Hilbert 空间 $\mathcal{H}$ 上；
- $D$ 是 $\mathcal{H}$ 上的自伴（无界）算子；
- 对每个 $a \in A$，交换子 $[D, a]$ 是有界算子（定义在 $D$ 的定义域上）；
- 预解式 $(D^2 + 1)^{-1}$ 是紧算子（「紧预解」条件，对应维数有限）。

若还要求 $D$ 与 $A$ 的表示满足 `γ`-分次（偶谱三元组）或带实结构 $J$（实数谱三元组），则称为**实谱三元组**——这是描述物理模型所需的完整结构。

### 3.2 基本条件汇总表

| 条件 | 意义 |
| --- | --- |
| $A$ 表示在有界算子 | 函数有界 |
| $D$ 自伴 | 度规可对角化 |
| $[D, a]$ 有界 | $a$ 是「1-Lipschitz」的（属于某光滑子代数） |
| $(D^2+1)^{-1}$ 紧 | 谱离散、有限维数 |
| $\gamma D = -D\gamma$ | 偶维数/手征性 |
| $J$ 反线性等距，$[a, Jb^*J^{-1}]=0$ | 对合/实结构（OC 条件） |

<span class="marginnote">紧预解条件 $(D^2+1)^{-1} \in \mathcal{K}$ 保证谱三元组的「有限维数」：它使 $\mathcal{H}$ 上 $D$ 的特征值趋于无穷，从而可以用特征值的增长刻画维数（Weyl 定律的推广）。这是非交换维数（dimension spectrum）的基础。</span>

## 4 交换情形的重建：Connes 旋流形定理

谱三元组并非无本之木。**Connes 重建定理（spin manifold theorem）**（1996，后与 Berline、Chamseddine 等人完善）说：

**定理**：若实谱三元组 $(A, \mathcal{H}, D)$ 满足一组（关于正则性、有限维数、定向、Poincaré 对偶等）公理，则 $A$ 必同构于 $C^\infty(M)$，$M$ 是紧自旋黎曼流形，$D$ 是 Dirac 算子。<span class="marginnote">这是「谱三元组 = 非交换黎曼流形」这一说法的精确版本：它告诉我们「正确的公理化不会走偏」，交换情形下谱三元组不会产生任何新对象，只会还原经典几何。GBVF 第 11 章《Connes' Spin Manifold Theorem》给出完整证明。</span>

这一定理的意义在于：它把「黎曼流形」这个概念本身**重写**成纯算子代数语言，从而可以直接把定义搬到非交换世界。

## 5 谱三元组作为非交换流形

谱三元组因此成为非交换几何的**公理化定义**：

**核心要点**：一个**非交换流形（noncommutative manifold）** 就是一个满足良好公理的谱三元组 $(A, \mathcal{H}, D)$。

- 交换情形：$(C^\infty(M), L^2(M, S), \not\!\!D)$，还原经典黎曼几何；
- 非交换环面 $\mathbb{T}^2_\theta$：$(A_\theta, \mathcal{H}, D)$，Dirac 算子由两个「导数」$\delta_1, \delta_2$ 的平方根给出；
- 有限模型（离散空间）：$(M_n(\mathbb{C}), \mathbb{C}^n, D=0)$，对应零维空间。

谱三元组的全部几何内容都从 $D$ 读出：维数（谱维数）、距离（Connes 距离）、度量（由 $D$ 决定）、标量曲率（谱作用，见《物理应用》）、指标（见《局部指标公式》）。它就是整个非交换几何的「黎曼流形」。

## 6 小结

- **Dirac 算子** $\not\!\!D$：一阶椭圆算子，编码流形的全部度规信息；$\|[\not\!\!D, f]\| = \|df\|_\infty$ 是几何的关键等式。
- **Connes 距离公式**：$d(\varphi, \psi) = \sup\{|\varphi(f)-\psi(f)|: \|[D, f]\| \le 1\}$，把距离变成纯算子语言，态即点。
- **谱三元组** $(A, \mathcal{H}, D)$：公理化非交换黎曼流形；$[D,a]$ 有界 + 紧预解是核心条件。
- **Connes 重建定理**：好公理的交换谱三元组必是经典自旋流形——公理化不走偏。
- 谱三元组同时承载拓扑（K-同调）、同调（循环上同调）与度量（距离公式），是后续一切内容的平台。

在下一节，我们将在这台平台上搭起**微分结构**：在代数 $A$ 上定义联络、曲率与非交换微积分，并写出非交换的 Yang–Mills 泛函——这就是**非交换流形的微分结构**。

<span class="marginnote">本文参考：GBVF《Elements of Noncommutative Geometry》Ch.10–11; Connes《Noncommutative Geometry》Ch.VI; Landi Ch.6《The Spectral Calculus》。Connes 距离公式原始文献见 Connes 1989 年论文《Compact metric spaces, Fredholm modules, and Hilbert spaces》/《Noncommutative Geometry》Ch.VI。</span>