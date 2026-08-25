---
title: K-理论与 K-同调（K₀/K₁ 群、Bott 周期性）
date: 2026-08-17
---

# K-理论与 K-同调

<div class="epigraph">
<p>数学是给不同的事物起同一个名字的艺术。</p>
<footer>—— 儒勒 · 昂利 · 庞加莱（Jules Henri Poincaré）</footer>
</div>

<div class="article-byline">
<p>第二级 · 非交换几何 ｜ Khalkhali《Basic Noncommutative Geometry》Ch.4; GBVF《Elements of Noncommutative Geometry》Ch.3; Connes《Noncommutative Geometry》Ch.II ｜ 2026-08-17</p>
</div>

## 为什么从 K-理论开始

上一节我们把非交换向量丛理解为有限投射模。接下来的问题是：**如何系统地分类它们？** 分类是同伦论/代数拓扑的看家本领，而 K-理论正是为此而生：把「模的直和」磨成一个交换群 $K_0(A)$，把「可逆矩阵」磨成 $K_1(A)$。这个「把几何对象磨成群」的操作，在 1950 年代末由 Grothendieck（代数几何）、Atiyah 与 Hirzebruch（拓扑）相继发明，成为二十世纪后半叶最重要的不变量之一。

庞加莱那句「给不同事物起同一个名字」在 K-理论里得到最极致的体现：向量丛、Fredholm 算子、椭圆微分算子、群表示——看似无关的对象，最终都被归入同一个 K-群。对非交换几何而言，K-理论还有特殊意义：它是**非交换空间的拓扑不变量**，Morita 等价下不变，而且不像同调那样依赖交换结构。它是我们走进「非交换流形的拓扑骨架」的第一座桥。

## 1 K₀：分类非交换向量丛

### 1.1 Grothendieck 群构造

**K₀（K-zero）**：设 $A$ 是含幺环，记 $V(A)$ 为 $A$ 上有限生成投射模的同构类在直和 $\oplus$ 下的交换幺半群。$K_0(A)$ 定义为 $V(A)$ 的 **Grothendieck 群**：形式差分的集合

$$
K_0(A) = \{ [P] - [Q] \mid P, Q \text{ 有限生成投射模} \} \big/ \sim
$$

其中 $[P] - [Q] = [P'] - [Q']$ 当且仅当存在 $R$ 使 $P \oplus Q' \oplus R \cong P' \oplus Q \oplus R$。<span class="marginnote">Grothendieck 群相当于「给减法补票」：幺半群没有减法，Grothendieck 通过形式差分造出减法。这个构造 1958 年出现在 Grothendieck 对黎曼–罗赫定理的证明中，后来被 Atiyah 与 Hirzebruch 移植到拓扑。</span>

### 1.2 例子

**$K_0(\mathbb{C}) = \mathbb{Z}$**：$\mathbb{C}$ 上的有限维向量空间由维数决定，$[P] = \dim P$，故 $K_0(\mathbb{C}) = \mathbb{Z}$。

**$K_0(C(X)) = K^0(X)$**：由 Serre–Swan，$K_0(C(X))$ 正是拓扑 K-理论 $K^0(X)$——紧空间上向量丛的 Grothendieck 群。

**分裂（split）**：$K_0(A) \cong \widetilde{K}_0(A) \oplus \mathbb{Z}$，其中 $\widetilde{K}_0(A) = \ker(K_0(A) \to \mathbb{Z}, [P]\mapsto \mathrm{rank} P)$ 是约化 K-群，只关心「非平凡部分」。

**幂等元语言**：由上一节 $P = eA^n$，$K_0(A)$ 也可用幂等元的稳定同伦类刻画：两个幂等元 $e, f$ 给出同构模当且仅当它们在 $M_\infty(A)$ 里被初等矩阵相连。

## 2 K₁：可逆矩阵的同伦

### 2.1 定义

**K₁（K-one）**：$K_1(A) = GL_\infty(A) / GL_\infty(A)^0$，其中 $GL_\infty(A) = \varinjlim GL_n(A)$ 是可逆矩阵的直极限（把 $GL_n(A)$ 左上角嵌入 $GL_{n+1}(A)$），$GL_\infty(A)^0$ 是包含单位元的连通分支。<span class="marginnote">直观上，$K_1$ 测量「可逆矩阵有多少同伦类」。对 $A = \mathbb{C}$，$GL_n(\mathbb{C})$ 连通，故 $K_1(\mathbb{C}) = 0$。对 $A = C(S^1)$，$K_1 = \mathbb{Z}$，由「绕圈数」（winding number）生成。</span>

### 2.2 与拓扑的对应

对 $A = C(X)$，$K_1(A)$ 对应约化 K-理论 $K^{-1}(X)$。由稳定同伦，可逆矩阵的同伦类等价于映射 $X \to GL_n(\mathbb{C})$ 的稳定同伦类，而 $GL_n(\mathbb{C})$ 的稳定同伦型就是 $\mathbb{Z}$ 的每个连通分支（Bott 周期性），所以 $K_1$ 捕捉的是「非交换的 $S^1$ 信息」。

### 2.3 K₁ 与 K₀ 的联系

$K_1(A)$ 也可以由 K₀ 与**悬置（suspension）** 表达：$K_1(A) = K_0(C_0(\mathbb{R}) \otimes A)$。这说明 K₁ 本质上是「推迟一步的 K₀」。这个关系直接引出 Bott 周期性。

## 3 Bott 周期性

### 3.1 陈述

**Bott 周期性（Bott periodicity）**：对任何 C\*-代数 $A$，有自然同构

$$
\beta: K_0(A) \longrightarrow K_0(A \otimes C_0(\mathbb{R}^2)) = K_2(A)
$$

从而 $K_0(A) \cong K_2(A)$，并且（经悬置）$K_i(A) \cong K_{i+2}(A)$ 对所有 $i$ 成立。**K-理论以 2 为周期**。

### 3.2 历史

Bott 的原始结果（1957）是同伦论中的：酉群的同伦群以 2 为周期，$\pi_i(U) = \pi_{i+2}(U)$。Atiyah 与 Bott 随后把这一结果转译成 K-理论中的周期性定理；对 C\*-代数的一般形式由 Rieffel（1981）给出，证明核心是 Toeplitz 代数与幂等元的显式构造。<span class="marginnote">Bott 周期性惊人地「免费」：K-群虽然定义在任意维上，却只依赖 $i \bmod 2$。这解释了为什么 $K_0, K_1$ 两个群就足够了——同调群则没有这样的周期性，每个维度都独立。</span>

### 3.3 与同调的对比

**核心对比表**：

| 性质 | 奇异同调 $H_*$ | K-理论 $K_*$ |
| --- | --- | --- |
| 周期性 | 无 | 周期 2（Bott） |
| 定义域 | 空间 | 空间或 C\*-代数 |
| 乘法结构 | 杯积 | 环结构（$K_0$ 有乘积） |
| Morita 不变性 | 不适用 | 成立 |
| 非交换推广 | 困难 | 自然（用模与幂等元） |

## 4 公式解析：Fredholm 指标与 K-同调配对

K-理论不仅有 K₀（向量丛/模），还有它的对偶 K₁ 和整个 **K-同调（K-homology）** 理论——由 Fredholm 算子生成。核心公式是 **Atiyah–Singer 指标定理的 K-理论形式**：

$$
\mathrm{ind}\, D = \langle [D], [E] \rangle \in \mathbb{Z}
$$

其中 $D$ 是椭圆算子（Fredholm 模），$E$ 是向量丛（模），$\langle \cdot, \cdot \rangle$ 是 K-同调与 K-理论的**配对**（index pairing）。三步拆解：

- **第一步**，Fredholm 指标：对有界 Fredholm 算子 $D: \mathcal{H} \to \mathcal{H}$，指标 $\mathrm{ind}\,D = \dim\ker D - \dim\operatorname{coker} D$，它是同伦不变的整数。
- **第二步**，生成 K-同调：由「Fredholm 模」$\mathcal{H}$ 上满足 $F^2 - 1 \in \mathcal{K}$ 的算子 $F$ 生成的 K-同调群，是所有椭圆算子的同伦类。
- **第三步**，配对：K-同调类与 K-理论类配对给出整数；对紧流形 $M$ 上的 Dirac 算子 $D$ 与向量丛 $E$，这个整数正是 Atiyah–Singer 指标 $\mathrm{ind}(D_E)$。

**这个配对是非交换指标定理的种子**：我们将在《局部指标公式》一篇中看到，如何用循环上同调显式算出这个整数，而不必构造 Fredholm 算子。

## 5 Kasparov 的 KK-理论：K-同调的现代框架

### 5.1 从 K-同调到 KK

K-同调起初由 Atiyah（1970）用 Fredholm 模定义：一个 Fredholm 模是 Hilbert 空间 $\mathcal{H}$ 上的算子 $F$（$F^2 - 1 \in \mathcal{K}$）连同 $A$ 的表示，使 $[F, a] \in \mathcal{K}$ 对所有 $a$ 成立。Kasparov 在 1970 年代末把这套理论统一为 **KK-理论（Kasparov theory）**：双向函子

$$
KK(A, B)
$$

把 K-理论（$KK(\mathbb{C}, A) = K_0(A)$）与 K-同调（$KK(A, \mathbb{C}) = K^0(A)$，即 $A$ 的 K-同调）纳入同一框架。<span class="marginnote">Kasparov 的 KK-理论（《The operator K-functor and extensions of C\*-algebras》，Izv. AN SSSR 1980）用 C\*-代数的（$A, B$）-双模上的拟簇构造出同时涵盖 K-理论与 K-同调、并有乘积结构 $KK(A,B)\times KK(B,C)\to KK(A,C)$ 的普遍理论。它是组装映射（assembly map）与 Baum–Connes 猜想的语言。</span>

### 5.2 与指标定理的连接

KK-理论的威力在于：椭圆算子给出的是 KK-类而非孤立的指标，而 **Baum–Connes 猜想**（1982）断言对含幺可数群 $\Gamma$，组装映射

$$
\mu: K^\Gamma_*(\underline{E}\Gamma) \longrightarrow K_*(C^*_r(\Gamma))
$$

是满射（猜测性同构），其中 $\underline{E}\Gamma$ 是 $\Gamma$ 的普遍固有作用空间。这个猜想把**群的几何拓扑**与**群 C\*-代数的 K-理论**联系起来，是非交换几何最深刻的开放问题之一——它蕴含 Novikov 猜想、约化 C\*-代数 K-理论的有理性等一大批结论，已对许多群类（双曲群、可数 amenable 群、Lie 群等）获证，但一般情形仍未解决。

### 5.3 为什么 KK-理论重要

- **统一性**：K-理论与 K-同调是同一条 Kasparov 积的两个端点的特殊情况；
- **指标公式**：Atiyah–Singer 指标本质上是 KK-积 $[D]\otimes [E]$ 落在 $KK(\mathbb{C},\mathbb{C}) = \mathbb{Z}$ 中的像；
- **非交换化**：KK-理论完全定义在 C\*-代数层面，无需流形结构——这是把指标定理推广到叶子空间、轨道空间等非交换空间的工具。

## 6 小结

- **K₀(A)**：有限生成投射模的 Grothendieck 群，分类非交换向量丛；$K_0(C(X)) = K^0(X)$。
- **K₁(A)**：可逆矩阵的同伦类；$K_1(A) = K_0(C_0(\mathbb{R})\otimes A)$。
- **Bott 周期性**：$K_i \cong K_{i+2}$，K-理论以 2 为周期（Bott 1957 / Atiyah–Bott / Rieffel 1981）。
- **K-同调**：由 Fredholm 模/椭圆算子生成的同调理论；与 K-理论的 index pairing 给出 Atiyah–Singer 指标。
- K-理论是 Morita 不变的，能自然地定义在非交换 C\*-代数上，是「非交换空间的拓扑」。

在下一节，我们将离开拓扑层面，进入非交换几何独有的同调理论——**循环上同调**，它为 K-理论提供显式的对偶不变量（Chern 特征），并在 Hochschild 同调与循环复形的相互作用中再现 Bott 周期性的影子。

<span class="marginnote">本文参考：Khalkhali《Basic Noncommutative Geometry》Ch.4; GBVF《Elements of Noncommutative Geometry》Ch.3; Landi Ch.5《A Few Elements of K-Theory》; Connes Ch.II。</span>