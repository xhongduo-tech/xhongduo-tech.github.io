---
title: Gelfand 对偶与 C*-代数回顾（交换 C*-代数与拓扑空间的等价）
date: 2026-08-17
---

# Gelfand 对偶与 C*-代数回顾

<div class="epigraph">
<p>非交换几何的全部秘密，就在于学会了把空间扔掉，只留下代数。</p>
<footer>—— 阿兰 · 孔涅（Alain Connes）</footer>
</div>

<div class="article-byline">
<p>第二级 · 非交换几何 ｜ Khalkhali《Basic Noncommutative Geometry》Ch.1; Connes《Noncommutative Geometry》Ch.II §1 ｜ 2026-08-17</p>
</div>

## 为什么从 Gelfand 对偶开始

非交换几何不是凭空造出来的。它从一个极其漂亮的事实出发：**经典拓扑空间与其上的连续函数代数是等价的**。这个等价——Gelfand–Naimark 定理——是整座大厦的第一块基石。如果你能理解「空间 $\cong$ 代数」这个命题，你就能理解非交换几何的全部纲领：把空间换成代数，然后让代数不交换。

从全课程体系看，读者在前两级已经熟悉了集合、拓扑空间和 Banach 空间的概念。Gelfand 对偶恰好在这些概念的交汇处：它把拓扑（开集、连续、紧性）完全翻译成了代数（理想、同态、单位元）的语言。一旦翻译完成，交换律就只是一个可选项——而非交换几何，就是去掉这个选项后发生的一切。

## 1 C*-代数的定义与基本例子

### 1.1 定义

**C\*-代数（C\*-algebra）**：一个复 Banach 代数 $A$，配备一个对合（involution）$*: A \to A$，满足：

- $(x^*)^* = x$（对合性）
- $(xy)^* = y^* x^*$（反同态性）
- $\|x^* x\| = \|x\|^2$（C\*-条件）

第三条 C\*-条件是关键：它把代数结构和范数锁定在一起，使得 $*$ 自动是等距的（$\|x^*\| = \|x\|$）。<span class="marginnote">C\*-条件 $\|x^*x\| = \|x\|^2$ 看起来像纯技术性要求，但它实际上保证了 $A$ 可以忠实表示在 Hilbert 空间上——这是 Gelfand–Naimark 表示定理的核心。</span>

### 1.2 基本例子

**例子 1（交换）：$C(X)$**。设 $X$ 是紧 Hausdorff 空间，$C(X)$ 是 $X$ 上所有复值连续函数构成的代数，范数为 $\|f\|_\infty = \sup_{x\in X}|f(x)|$，对合为逐点复共轭 $f^*(x) = \overline{f(x)}$。这是最典型的**交换** C\*-代数。

**例子 2（非交换）：$\mathcal{B}(\mathcal{H})$**。设 $\mathcal{H}$ 是 Hilbert 空间，$\mathcal{B}(\mathcal{H})$ 是 $\mathcal{H}$ 上所有有界线性算子构成的代数，范数为算子范数，对合为 Hilbert 伴随。这是最典型的**非交换** C\*-代数。当 $\dim\mathcal{H} = n$ 时，$\mathcal{B}(\mathcal{H}) \cong M_n(\mathbb{C})$。

<span class="marginnote">$M_n(\mathbb{C})$ 是人类最早接触的非交换 C\*-代数——矩阵乘法不交换，但 $ (AB)^* = B^* A^*$ 成立。量子力学里的 Heisenberg 矩阵力学就在用这个代数。</span>

**例子 3：$C_0(X)$**。当 $X$ 局部紧而非紧时，考虑在无穷远处趋于零的连续函数构成的代数 $C_0(X)$。它没有单位元——这恰好对应 $X$ 非紧。通过添加单位元（unitalization）可以把 $C_0(X)$ 嵌入到 $C(X^+)$，其中 $X^+$ 是 $X$ 的一点紧化。

## 2 Gelfand 对偶：交换 C\*-代数 = 紧 Hausdorff 空间

**Gelfand 对偶（Gelfand duality）** 描述了交换 C\*-代数范畴与紧 Hausdorff 空间范畴之间的反变等价。

### 2.1 从空间到代数：$C(-)$ 函子

对每个紧 Hausdorff 空间 $X$，赋 $C(X)$。对每个连续映射 $f: X \to Y$，赋拉回同态 $f^*: C(Y) \to C(X)$，$(f^*g)(x) = g(f(x))$。注意方向反转——这是**反变**函子。

### 2.2 从代数到空间：$\Delta$ 谱

给定一个交换 C\*-代数 $A$，定义其**谱（spectrum）** $\Delta(A)$ 为 $A$ 上所有非零乘性线性泛函（即**特征（character）**）的集合，配备弱\*拓扑。

**Gelfand–Naimark 定理**：对交换 C\*-代数 $A$，Gelfand 变换

$$
\Gamma: A \to C(\Delta(A)), \quad \Gamma(a)(\chi) = \chi(a)
$$

是一个等距 $*$-同构。<span class="marginnote">Gelfand 变换是 Fourier 变换的深远推广：当 $A = L^1(\mathbb{R})$ 配备卷积时，$\Delta(A) \cong \mathbb{R}$，Gelfand 变换就是 Fourier 变换。</span>

### 2.3 等价陈述

**Gelfand 对偶的严格表述**：函子 $X \mapsto C(X)$ 与 $A \mapsto \Delta(A)$ 建立了紧 Hausdorff 空间范畴 $\mathbf{Comp}$ 与交换幺元 C\*-代数范畴 $\mathbf{CommC^*Alg}$ 之间的**反变范畴等价**。

$$
\mathbf{Comp} \simeq \mathbf{CommC^*Alg}^{\mathrm{op}}
$$

这意味着：**一个交换 C\*-代数「知道」它来自哪个空间**。任何拓扑性质都可以用代数语言表述——紧性对应单位元，连通性对应幂等元，等等。

### 2.4 核心对比表

| 拓扑概念 | 代数对应 |
| --- | --- |
| 点 $x \in X$ | 特征 $\chi_x: f \mapsto f(x)$，即极大理想 |
| 开集 $U \subset X$ | 闭理想 $I_U = \{f \mid f|_{X\setminus U} = 0\}$ |
| 连续映射 $f: X\to Y$ | $*$-同态 $f^*: C(Y) \to C(X)$ |
| 紧性 | 单位元的存在性 |
| 连通分支 | 幂等元分解 |
| $X$ 的一点紧化 | $A$ 的 unitalization $A^+$ |

## 3 为什么这对非交换几何至关重要

Gelfand 对偶的哲学意义怎么强调都不过分：**它把几何学转化成了代数学**。一旦转化完成，交换性就不再是本质属性——它只是代数中的一个性质。

### 3.1 非交换空间的出生

设 $A$ 是**任意**（不一定交换的）C\*-代数。如果 $A$ 不交换，那么 $\Delta(A)$ 可能为空——因为特征必须是乘性线性泛函，而非交换代数上的乘性泛函通常不存在。<span class="marginnote">例如 $M_2(\mathbb{C})$ 上的非零乘性线性泛函：假设存在 $\chi: M_2(\mathbb{C})\to\mathbb{C}$，则 $\chi(AB)=\chi(A)\chi(B)=\chi(B)\chi(A)=\chi(BA)$，但 $AB\neq BA$ 时这个等式无法同时成立——矛盾。</span>

但我们可以把 $A$ 本身当作一个「虚空间」。这就是**非交换空间（noncommutative space）** 的核心思想：一个非交换 C\*-代数 $A$ 被理解为某个「非交换拓扑空间」上的函数代数，尽管这个空间本身没有点。

### 3.2 几个重要推论

**不可交换性 = 几何模糊性**：当代数不交换时，点不再存在。这对应量子力学中「位置与动量不能同时确定」的 Heisenberg 不确定性原理——$\Delta(A)$ 为空意味着没有一个「点」可以同时测量所有可观测量。

**范数与拓扑信息**：C\*-条件保证了范数由代数结构唯一确定——如果一个 $*$-代数有两种范数使之成为 C\*-代数，它们必须相等。这意味着非交换拓扑完全由代数结构捕获。

**从局部到整体**：经典几何的局部-全局二分法在非交换框架中需要重新审视。交换 C\*-代数可以局部化为某个点附近的函数芽，而非交换 C\*-代数上的「局部化」更微妙——Serre–Swan 定理和 Morita 等价提供了替代方案，这将在后续文章中展开。

## 4 公式解析：Gelfand 变换的等距性

Gelfand 变换 $\Gamma: A \to C(\Delta(A))$ 的等距性 $\|\Gamma(a)\|_\infty = \|a\|$ 是定理的核心：

$$
\|\Gamma(a)\|_\infty = \sup_{\chi \in \Delta(A)} |\chi(a)| = \sup_{\chi \in \Delta(A)} |\chi(a)| = r(a) = \|a\|
$$

其中 $r(a)$ 是 $a$ 的谱半径。三步拆解：

- **第一步**，$\|\Gamma(a)\|_\infty$ 的定义：$\Gamma(a)$ 是 $\Delta(A)$ 上的连续函数，其 sup 范数等于所有特征模的 sup。
- **第二步**，谱半径公式：对 Banach 代数元素，谱半径 $r(a) = \lim_{n\to\infty}\|a^n\|^{1/n}$。对 C\*-代数元素，如果 $a$ 是正规的（$a^*a = aa^*$），则 $r(a) = \|a\|$。
- **第三步**，关键技巧：对任意 $a\in A$，$a^*a$ 是自伴的从而正规，所以 $\|a\|^2 = \|a^*a\| = r(a^*a) = \sup_{\chi}|\chi(a^*a)| = \sup_\chi|\chi(a)|^2 = \|\Gamma(a)\|_\infty^2$。

**这个等式把代数范数和函数空间 sup 范数等同起来**——它保证了 Gelfand 变换是等距，从而 $A$ 与 $C(\Delta(A))$ 作为赋范空间不可区分。

## 5 小结

- **C\*-代数**是带对合的 Banach 代数，满足 $\|x^*x\| = \|x\|^2$；交换典范是 $C(X)$，非交换典范是 $\mathcal{B}(\mathcal{H})$。
- **Gelfand–Naimark 定理**：交换 C\*-代数 $A$ 通过 Gelfand 变换等距同构于 $C(\Delta(A))$，其中 $\Delta(A)$ 是特征空间。
- **Gelfand 对偶**建立了紧 Hausdorff 空间与交换 C\*-代数之间的反变范畴等价：$\mathbf{Comp} \simeq \mathbf{CommC^*Alg}^{\mathrm{op}}$。
- 非交换 C\*-代数没有特征，因此不能对应到有点的空间——这催生了**非交换空间**的概念：把代数本身视为虚空间的函数代数。

在下一节，我们将探讨这个「虚空间」到底意味着什么，以及如何用算子代数语言描述非交换空间的全部结构——这就是**非交换空间的思想**。

<span class="marginnote">本文参考：Khalkhali《Basic Noncommutative Geometry》Ch.1 §1.1–1.3; Connes《Noncommutative Geometry》Ch.II §1. 关于 Gelfand 对偶的更多细节，可参阅 Arveson《A Short Course on Spectral Theory》。</span>