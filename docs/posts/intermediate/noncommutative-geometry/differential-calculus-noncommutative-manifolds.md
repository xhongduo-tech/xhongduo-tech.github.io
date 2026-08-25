---
title: 非交换流形的微分结构（联络、曲率、Yang–Mills 泛函）
date: 2026-08-17
---

# 非交换流形的微分结构

<div class="epigraph">
<p>几何学是研究变换群下不变性质的学科。</p>
<footer>—— 费利克斯 · 克莱因（Felix Klein），《Erlangen 纲领》，1872</footer>
</div>

<div class="article-byline">
<p>第二级 · 非交换几何 ｜ GBVF《Elements of Noncommutative Geometry》Ch.8; Landi《An Introduction to Noncommutative Spaces》Ch.7–8; Connes《Noncommutative Geometry》Ch.VI ｜ 2026-08-17</p>
</div>

## 为什么从微分结构开始

克莱因纲领说：几何 = 不变量。但对物理学家与几何学家来说，几何的灵魂在于**微分结构**——联络、曲率、以及由它们写出的作用量泛函。Maxwell 方程组是 $U(1)$ 联络的曲率方程，Yang–Mills 理论是规范场的动力学，广义相对论是 Levi-Civita 联络的曲率方程。没有微分结构，几何只是拓扑的骨架。

非交换几何同样需要这些。上一节的谱三元组给了我们「度规」，但还差「如何对模做微分、如何定义曲率、如何写作用量」。本节的任务，就是在代数 $A$ 与模 $E$ 上建立起完整的一阶微积分：**非交换微分形式、联络、曲率与非交换 Yang–Mills 泛函**。这一套机器不仅是纯数学，更是下一节《物理应用》中「标准模型的谱作用」的直接前奏。

## 1 非交换微分形式

### 1.1 导子给出的微分

最自然的起点是**导子（derivation）**。设 $A$ 是代数，$\delta: A \to A$ 是导子（$\delta(ab) = \delta(a)b + a\delta(b)$）。经典情形的坐标导数 $\partial_\mu$ 就是导子。<span class="marginnote">对 $A = C^\infty(M)$，导子全体正是向量场 $\mathfrak{X}(M)$——这是「向量场 = 导子」的经典对应在非交换世界的直译。非交换环面 $A_\theta$ 上就有两个基本导子 $\delta_1, \delta_2$，对应「两个坐标的偏导」。</span>

### 1.2 泛微分形式

为了让微分结构不依赖具体导子的选取，Connes 引入了**泛微分形式（universal differential forms）**：令 $\Omega^0 A = A$，且对 $n \ge 1$，

$$
\Omega^n A = \left\{ \sum_i a_0^i\, da_1^i \cdots da_n^i \right\}, \qquad d(a_0\, da_1 \cdots da_n) = da_0\, da_1 \cdots da_n
$$

其中 $d$ 是形式符号。于是 $(\Omega^\bullet A, d)$ 是**分次微分代数（graded differential algebra）**：$d^2 = 0$、$d(\omega\eta) = d\omega\,\eta + (-1)^{|\omega|}\omega\,d\eta$。

**辨析｜易错点：** 泛微分形式 $\Omega^\bullet A$ 与谱三元组 $(A, \mathcal{H}, D)$ 给出的「有界算子微分形式」不同。后者把 $da$ 实现为交换子 $[D, a]$，并商去核（Junk 理想）后得到「受控的」微分代数。物理上使用的是后者——它保证了范数与紧性。

### 1.3 从谱三元组看微分

给定谱三元组 $(A, \mathcal{H}, D)$，定义

$$
da = i[D, a], \qquad a \in A
$$

它把「微分」变成「算子交换子」，并把 $A$ 的微分结构嵌入 $\mathcal{B}(\mathcal{H})$。这一实现方式是非交换微积分的核心——在《局部指标公式》中我们将看到它如何与循环上同调衔接。

## 2 联络与曲率

### 2.1 联络

设 $E$ 是 $A$ 上的有限投射（Hermitian）模，$\Omega^1$ 是 $A$ 上的一阶微分形式。**联络（connection）** 是线性映射

$$
\nabla: E \longrightarrow E \otimes_A \Omega^1
$$

满足 Leibniz 规则：$\nabla(a\xi) = a\,\nabla(\xi) + da \otimes \xi$。

若 $E$ 是 Hermitian 模（有 $A$-值内积 $\langle\cdot,\cdot\rangle$），联络还须满足**相容性**：

$$
d\langle \xi, \eta\rangle = \langle \nabla \xi, \eta\rangle + \langle \xi, \nabla \eta\rangle
$$

**核心要点表**：

| 经典 | 非交换 |
| --- | --- |
| 向量丛 $E \to M$ | 有限投射模 $E$ |
| 联络 $\nabla$ | 模上的联络（Leibniz 规则） |
| 曲率 $F = \nabla^2$ | $F = \nabla^2 \in \operatorname{End}(E)\otimes\Omega^2$ |
| 规范群 $G$ | 幺正元素群 $\mathcal{U}(A)$ |
| 规范场 $A_\mu$ | 联络 $\nabla$ |

### 2.2 曲率

联络的**曲率（curvature）** 定义为

$$
F = \nabla^2: E \longrightarrow E \otimes_A \Omega^2
$$

对 Hermitian 联络 $F$ 是自伴的（$F^* = F$），且满足 Bianchi 恒等式 $[\, \nabla, F\,] = 0$。在平凡模 $E = A^N$ 上，联络由 1-形式矩阵 $\omega$（规范势）给出：$\nabla = d + \omega$，于是 $F = d\omega + \omega^2$——与经典 Yang–Mills 的场强公式完全同形。<span class="marginnote">这是「非交换几何几乎自动重现经典公式」的典型例子：只需把 $d$ 换成泛微分或交换子实现，$F = d\omega + \omega^2$ 自动成立，因为分次微分代数的结构把非线性项 $\omega^2$ 带了进来。</span>

## 3 非交换 Yang–Mills 泛函

### 3.1 作用量

设 $A$ 上有迹（或积分）$\tau: \Omega^d \to \mathbb{C}$（$d$ 是谱维数）。**非交换 Yang–Mills 泛函（noncommutative Yang–Mills functional）**：

$$
\mathrm{YM}(\nabla) = \tau(F \wedge *F) = \tau(F^2)
$$

其中 $*$ 是 Hodge 对偶（若存在），或直接取 $F \in \Omega^2$ 的「内积平方」再由迹求值。曲率 $F$ 分解为无迹部分与迹部分：

$$
F = F_0 + \tfrac{1}{N}\,\mathrm{tr}(F)\, I, \qquad \mathrm{YM} = \mathrm{YM}(F_0) + \text{迹项}
$$

### 3.2 极值方程与瞬子

对 $\mathrm{YM}$ 变分得到非交换 Yang–Mills 方程：

$$
\nabla * F = 0 \quad \text{（运动方程）}, \qquad *F = \pm F \quad \text{（自对偶/反自对偶）}
$$

满足自对偶条件的联络称为**瞬子（instanton）**。经典瞬子方程的解给出拓扑不变量（拓扑荷 = 第二 Chern 类）；非交换瞬子（如 Connes–Rieffel 在非交换环面上的构造）同样携带整数拓扑荷，但解空间更丰富。

### 3.3 公式解析：为什么 $F = \nabla^2$ 是「曲率」

- **第一步**，把 $\nabla^2$ 展开：$\nabla^2(\xi) = \nabla(d\omega\ \xi + \omega \xi) = \dots = (d\omega + \omega^2)\xi$。中间的计算精确使用 Leibniz 规则两次。
- **第二步**，辨识几何：在平凡情形 $\nabla = d$ 时 $F = d^2 = 0$——**平直联络的曲率为零**，正是「曲率度量弯曲程度」的代数表达。
- **第三步**，规范变换：若 $u \in \mathcal{U}(A)$ 是规范变换，则 $F$ 按 $F \mapsto u F u^*$ 变换（协变），而 $\mathrm{tr}(F^2)$ 不变——**Yang–Mills 泛函是规范不变的**。这就是克莱因意义上的「不变量」：物理作用量必须在规范变换下不变。

**一句话**：非交换 Yang–Mills 泛函 = 规范不变的曲率平方积分，它把经典规范场论完整地移植到了非交换代数上。

## 4 在谱三元组上重写作用量

谱三元组 $(A, \mathcal{H}, D)$ 还给出另一条通向作用量的路——**谱作用（spectral action）**：

$$
S = \operatorname{Tr}\left( f\!\left( \frac{D_A^2}{\Lambda^2} \right) \right), \qquad D_A = D + A + JAJ^{-1}
$$

其中 $D_A$ 是「带规范联络的 Dirac 算子」，$f$ 是截断函数，$\Lambda$ 是能标。热核展开把 $\operatorname{Tr}(f(D^2/\Lambda^2))$ 展开成几何项之和（体积、标量曲率、Yang–Mills 项、……）。这一公式是下一节《物理应用》中标准模型谱作用的直接来源。<span class="marginnote">谱作用由 Chamseddine 与 Connes 在 1996–97 年提出；它的意义在于：<strong>只要给定了谱三元组，规范场与 Higgs 机制的全部几何项都自动从谱中读出</strong>，无需人工写拉格朗日量。这也回答了「为什么从微分结构讲起」——作用量不是外加的，而是谱三元组的固有产物。</span>

## 5 小结

- **非交换微分**：导子给出向量场；泛微分形式 $\Omega^\bullet A$ 是分次微分代数；谱三元组里 $da = i[D, a]$。
- **联络**：模上的 $\nabla$ 满足 Leibniz 规则与 Hermitian 相容性；规范势 $\omega$ 与 $F = d\omega + \omega^2$ 自动出现。
- **曲率** $F = \nabla^2$：平直联络曲率为零，规范变换下协变。
- **非交换 Yang–Mills 泛函** $\mathrm{YM}(\nabla) = \tau(F^2)$：规范不变、带瞬子解。
- **谱作用** $\operatorname{Tr}(f(D_A^2/\Lambda^2))$：谱三元组自动产出几何作用量，连接微分结构与物理。

在下一节，我们将把这些微分结构兑现为**指标公式**：用循环上同调与谱三元组显式计算椭圆算子的指标——这就是 **Connes–Moscovici 局部指标公式** 与 **JLO 上循环**。

<span class="marginnote">本文参考：GBVF《Elements of Noncommutative Geometry》Ch.8《Noncommutative Differential Calculi》; Landi Ch.7–8《Noncommutative Differential Forms》《Connections on Modules》; Connes《Noncommutative Geometry》Ch.VI。谱作用见 Chamseddine–Connes《The spectral action principle》(1997)。</span>