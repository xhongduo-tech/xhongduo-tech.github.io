---
title: 实超曲面与 CR 结构的定义
date: 2026-08-07
---

# 实超曲面与 CR 结构的定义

<div class="epigraph">
<p>一个区域的边界并不「知道」自己是边界——它只携带一块复切空间，而这足以重写全部函数论。</p>
<footer>—— 仿 约瑟夫 · 科恩（Joseph J. Kohn），《CR 流形上的偏微分方程》</footer>
</div>

<div class="article-byline">
<p>第二级 · 多复变函数论 ｜ Hörmander 第5章；Krantz 第7章 ｜ 2026-08-07</p>
</div>

## 为什么从 CR 结构开始

前 24 篇我们总是从一个**区域** $D$ 出发。但第 24 篇末的一个等式泄露了另一种可能：Hardy 空间的边界值**正是边界上满足某种方程的函数**——也就是说，**边界 $\partial D$ 本身可以脱离内部区域独立研究**。这个独立的边界理论就是 **CR 理论**：在实超曲面 $M \subset \mathbb{C}^n$ 上，虽然 $M$ 只有实维 $2n-1$，但它继承了一块「复切空间」，其上的函数可以定义「限制版的 Cauchy–Riemann 方程」。**CR 结构**就是这套内蕴复几何的精确语言。<span class="marginnote">为什么「边界本身」值得单独立论？因为多复变的核心难题——延拓、正则性、表示——全部发生在边界上。把边界从区域中「解放」出来，函数论就有了不依赖内部的自主版本。这还催生了抽象 CR 流形理论（不嵌入 $\mathbb{C}^n$ 的 CR 结构），是现代复几何与微分几何的交汇点。</span>

## 1 实超曲面的复切空间

设 $M \subset \mathbb{C}^n$ 是**实超曲面**：局部可写为 $M = \{ \rho = 0 \}$，$\rho$ 实值光滑，$\nabla \rho \neq 0$。$M$ 有实维 $2n-1$。

$M$ 在点 $p$ 的**实切空间** $T_p M \subset \mathbb{C}^n$（视为 $\mathbb{R}^{2n}$）是 $2n-1$ 维。复数乘 $J$（乘 $i$）作用在切向量上。定义

$$
H_p M = T_p M \cap J(T_p M)
$$

即「$M$ 的切向量中被 $J$ 保持的向量」——这是 $M$ 在 $p$ 点的**复切空间（CR 切空间）**，实维 $2n-2$，复维 $n-1$。<span class="marginnote">几何直觉：$M$ 的法向是「横着」的，复数乘 $J$ 把它转到另一个方向；真正「留在 $M$ 内部且被复结构保持」的只有复切方向。$H_p M$ 是由满足 $\partial\rho(p) \cdot v = 0$（一阶复切条件）的复向量 $v$ 张成。这正是第 7 篇里 Levi 形式作用的子空间——CR 理论与伪凸性在此重逢。</span>

**CR 维数**：$\dim_{\mathbb C} H_p M = n-1$，称为 $M$ 的 **CR 维数**。整个流形 $M$ 连同分布 $H M = \bigcup_p H_p M$ 构成一个 **CR 流形（CR manifold）**。

## 2 CR 结构：抽象定义

**抽象 CR 流形**：一个实光滑流形 $M$（实维 $2n-1$）连同它的复切丛 $H M \subset TM$（实秩 $2n-2$）及复结构 $J: H M \to H M$（$J^2 = -I$），满足**可积性条件**：

$$
[X, JY] + [JX, Y] \in H M, \qquad [JX, JY] - [X, Y] = J([JX, Y] + [X, JY])
$$

对 $X, Y \in H M$。直觉：$J$ 在复切方向的 Lie 括号下保持结构——「$M$ 的复结构可以像 $\mathbb{C}^{n-1}$ 一样局部协调」。

**嵌入 CR 流形**：$M \subset \mathbb{C}^n$（实超曲面）天然是 CR 流形：$H M$ 取切空间交集，$J$ 取乘 $i$。**是否每个抽象 CR 流形都能嵌入 $\mathbb{C}^n$？** ——这是下一节末篇的 Lewy 反例要回答的深刻问题（答案：否）。<span class="marginnote">可积性条件在多复变中自动成立（嵌入情形），但在抽象 CR 流形上是<strong>非平凡假设</strong>。它保证 $M$ 局部看起来像「$\mathbb{C}^{n-1}$ 的一个方向 + 实方向」——即存在局部坐标 $(z_1,\dots,z_{n-1}, t)$ 使 $H M$ 由 $\partial/\partial\bar z_j$ 张成。抽象 CR 理论的正规化版本叫 <strong>Lev 平坦性 / minimality</strong> 条件，是很多正则性定理的前提。</span>

## 3 CR 函数与 $\bar\partial_b$ 算子

有了 CR 结构，就能定义**边界上的全纯函数**：

**CR 函数（CR function）**：$M$ 上的光滑（或分布）函数 $f$，满足**CR 方程**：

$$
\bar\partial_b f = 0
$$

其中 $\bar\partial_b$ 是限制在复切空间上的 $\bar\partial$ 算子：对局部坐标 $\{Z_\alpha\}$（$H M$ 的局部标架），

$$
\bar\partial_b f = \sum_\alpha (\bar Z_\alpha f)\, d\bar Z_\alpha
$$

**$\bar\partial_b f = 0$ ⟺ $f$ 沿每个复切方向都是「反全纯」的**——即 $f$ 在 $M$ 的每个复切叶上满足 Cauchy–Riemann 方程。<span class="marginnote">为什么 CR 函数重要？因为<strong>任何全纯函数限制到实超曲面边界上都是 CR 函数</strong>（沿复切方向的 $\bar\partial$ 为零）。反之，CR 函数能否「延拓」成内部全纯函数，是 CR 理论的核心问题（下一节）。Hardy 空间边值正是 $L^2$ 的 CR 函数——第 24 篇的接缝在此精确化。</span>

## 4 公式解析：$\bar\partial_b$ 与 CR 方程

在局部坐标 $(z_1,\dots,z_{n-1}, t)$ 下（$z$ 复、$t$ 实），复切空间由 $\partial/\partial\bar z_1,\dots,\partial/\partial\bar z_{n-1}$ 张成，CR 方程写作：

$$
\bar\partial_b f = \sum_{\alpha=1}^{n-1} \frac{\partial f}{\partial \bar z_\alpha}\, d\bar z_\alpha = 0
\;\iff\;
\frac{\partial f}{\partial \bar z_\alpha} = 0 \;\;(\alpha = 1,\dots, n-1)
$$

- **第一步，读出方程数**：CR 方程是 $n-1$ 个**一阶线性方程**（对 $2n-1$ 个实变量的函数）。未知函数 $f$ 有 $2n-1$ 个实自由度，方程 $n-1$ 个复条件（= $2n-2$ 个实条件）——留 $1$ 个实自由度（沿法向 $t$ 的任意性）。**CR 函数空间是「无限维但被 $2n-2$ 个方程约束」的**。
- **第二步，与 $\bar\partial$ 方程的关系**：$\bar\partial_b$ 是「$\bar\partial$ 在边界上的切向投影」：$\bar\partial_b f = (\bar\partial \tilde f)|_{H M}$（$\tilde f$ 是 $f$ 的任意光滑延拓，结果与延拓无关）。CR 方程是「边界上的 Cauchy–Riemann 方程」。
- **第三步，为什么法向自由度保留**：$f$ 沿法向 $t$ 可以任意变化而不破坏 CR 方程（因为 $\partial/\partial t$ 不在复切空间里）。**这个 $1$ 维自由度正是「延拓」的自由度**——决定 CR 函数能否延拓的关键就在它如何随 $t$ 变化。

## 5 辨析与延伸：CR 结构的五个要点

**辨析 1：CR 维数与实维数的关系**。实超曲面 $M \subset \mathbb{C}^n$：实维 $2n-1$，CR 维数 $n-1$。多 $1$ 个实维正是法向方向。**CR 流形是「复维 $n-1$ + 实维 $1$」的混合体**——这 $1$ 个实维（法向）是延拓性的全部悬念。<span class="marginnote">一般 CR 流形（不嵌入）的维数关系：实维 $2n-1$，CR 维数 $n-1$，法向维数 1。这是「超曲面型」CR 结构。若法向维数更大，则 CR 结构的类型更复杂（如 CR 子流形）。</span>

**辨析 2：抽象 CR 结构需要可积性**。嵌入 $\mathbb{C}^n$ 的超曲面的 CR 结构自动满足可积性条件；但抽象定义中，可积性是**必须假设**的。没有可积性，$\bar\partial_b$ 可能不满足 $\bar\partial_b^2=0$，整个 CR 函数理论崩塌。**可积性 = CR 结构的「兼容性」**。

**辨析 3：CR 函数为什么「不够多」**。CR 方程约束 $n-1$ 个复方向，留下 $1$ 个实方向自由。所以 CR 函数空间比全纯函数「大」（多了法向自由度），但比任意光滑函数「小」（被 $2n-2$ 个实方程约束）。**CR 函数是「介于解析与任意之间的中间体」**。

**辨析 4：CR 流形与复流形的边界**。每个复流形（如实超曲面的邻域）的边界是 CR 流形；反之，CR 流形何时能成为某个复流形的边界，是「嵌入/填充」问题——这是 CR 理论的现代核心之一（下一组末篇涉及）。

**误区清单**：

- **误区 1**：以为「CR 维数 = 实维数」。
  正解：CR 维数是复切方向的复维数 $n-1$，小于实维数 $2n-1$。
- **误区 2**：以为「抽象 CR 结构自动可积」。
  正解：可积性是抽象定义的必要假设。
- **误区 3**：以为「CR 函数就是全纯函数的限制」。
  正解：全纯函数的限制是 CR 函数，但反之不一定（可延拓性问题）。
- **误区 4**：以为「CR 理论只是边界理论的附属」。
  正解：CR 流形可独立于嵌入存在，是自主的研究对象。

**术语表**：

| 中文 | 英文 | 说明 |
| --- | --- | --- |
| 实超曲面 | real hypersurface | 实维 $2n-1$ |
| CR 维数 | CR dimension | 复切方向复维数 |
| 复切空间 | CR tangent space | $H_pM$ |
| CR 流形 | CR manifold | 实流形 + 复切结构 |
| 可积性 | integrability | 括号兼容条件 |
| CR 函数 | CR function | $\bar\partial_b f=0$ |
| $\bar\partial_b$ | tangential CR operator | 边界复算子 |

## 6 历史注记与知识树

**历史**：CR 结构的雏形出现在 Lewy（1956）与 Kohn（1963）的工作中；Folland–Stein（1974）系统建立 Heisenberg 群上的 CR 分析；Tanaka 与 Chern–Moser（1974）给出 CR 流形的等价理论（Chern–Moser 不变量）。CR 几何由此成为微分几何与多复变的交叉前沿。

**知识树**：

- 向后：Hardy 空间边值（第 4 组末篇）、实超曲面（本组开篇）。
- 向前：Levi 形式与强伪凸（本组第 26 篇）、CR 延拓与 Lewy 反例（本组第 27–28 篇）。
- 横向：辛几何的接触结构（第二级《辛几何》）——CR 结构的实模拟。

**一句话记忆**：CR 结构 = 实超曲面携带的复切空间 + 可积复结构；CR 函数 = 沿复切方向的「全纯函数」；法向自由度是延拓的悬念。

## 7 小结

- **实超曲面** $M = \{\rho=0\}$：有复切空间 $H_p M = T_p M \cap J T_p M$，复维 $n-1$。
- **CR 流形**：实流形 + 复切分布 + 可积复结构；嵌入情形自动满足。
- **CR 函数**：$\bar\partial_b f = 0$——沿每个复切方向的 CR 方程；全纯函数的边界限制都是 CR 函数。
- **CR 维数与法向自由度**：$n-1$ 复方向 + $1$