---
title: 子空间的正交补
date: 2026-08-08
---

# 子空间的正交补

<div class="epigraph">
<p>每个子空间都有一个「镜像」：所有与它垂直的方向构成它的正交补。两者合起来，恰好填满整个空间。</p>
<footer>—— 斯特朗（Gilbert Strang，《Introduction to Linear Algebra》）</footer>
</div>

<div class="article-byline">
<p>第二级 · 线性代数 ｜ Strang《Introduction to Linear Algebra》§4.1 ｜ 2026-08-08</p>
</div>

## 为什么从正交补开始

四个基本子空间展示了一个惊人模式：每个子空间都有一个「正交补」，两者维数互补、彼此垂直。本节把正交补从「四子空间的副产品」提升为「一般理论」——它不仅是理解四子空间的钥匙，更是投影、最小二乘、SVD 的几何地基。<span class="marginnote">正交补的核心应用在第八篇：<strong>把任意向量分解成「属于 $W$ 的部分 + 属于 $W^{\perp}$ 的部分」</strong>，前者是投影、后者是残差。最小二乘、傅里叶展开、PCA 全是这个「正交分解」的变体。</span>

本节定义正交补、给出基本性质，并把 $\mathbb{R}^n$ 分解成「子空间 ⊕ 正交补」的直和。

## 1 正交补的定义

**核心概念**：设 $W$ 是 $\mathbb{R}^n$ 的一个子空间。$W$ 的**正交补（orthogonal complement）** $W^{\perp}$ 是「与 $W$ 中所有向量都正交」的向量全体：

$$
W^{\perp} = \{\mathbf{x} \in \mathbb{R}^n \mid \mathbf{x} \cdot \mathbf{w} = 0 \text{ 对一切 } \mathbf{w} \in W\}
$$

**基本性质**：

- $W^{\perp}$ 是 $\mathbb{R}^n$ 的子空间；
- $(W^{\perp})^{\perp} = W$（补的补回到自身）；
- $\dim W + \dim W^{\perp} = n$；
- $W \cap W^{\perp} = \{\mathbf{0}\}$（只有零向量与自身垂直）。

**重点**：最后一条 + 维数公式推出**正交直和分解**：

$$
\mathbb{R}^n = W \oplus W^{\perp}
$$

**每个向量都能唯一分解成「$W$ 里的部分 + $W^{\perp}$ 里的部分」**——这就是正交分解定理，是投影理论的出发点。

**辨析｜易错点：** $W^{\perp}$ 与「$W$ 的补集」无关——它是**向量空间**（含零向量、对运算封闭），不是集合补。例：$W = \{x\text{-轴}\}$ 在 $\mathbb{R}^2$ 中，$W^{\perp} = \{y\text{-轴}\}$，两轴直和成整个平面。

## 2 正交补的计算：通过基与方程

**求法**：设 $W = \operatorname{span}\{\mathbf{w}_1, \cdots, \mathbf{w}_k\}$，则

$$
W^{\perp} = \{\mathbf{x} \mid \mathbf{w}_i \cdot \mathbf{x} = 0, \; i = 1, \cdots, k\}
$$

即**正交补 = 一组齐次线性方程的解空间**——把 $\mathbf{w}_i$ 看成方程组的「行」，解 $A\mathbf{x} = \mathbf{0}$。

例：$W = \operatorname{span}\{(1, 2, 3)\}$ 在 $\mathbb{R}^3$ 中（一条直线）。$W^{\perp}$：解 $x_1 + 2x_2 + 3x_3 = 0$，这是过原点的平面（二维）。$\dim W + \dim W^{\perp} = 1 + 2 = 3$ ✓。

**重点**：正交补的计算与解齐次方程组完全同构——**造矩阵、消元、求基础解系**。反过来，任何齐次方程组的解空间（零空间）都是一个张成空间的正交补（四子空间定理）。

## 3 公式解析：$\mathbb{R}^n = W \oplus W^{\perp}$ 的正交分解

为什么每个向量都能唯一拆成「$W$ 部分 + 垂直部分」？拆成四步：

- **第一步，维数对账**：$\dim W + \dim W^{\perp} = n$，所以「两块的维数之和」恰好等于全空间维数，直和若能成立，维数正好够。
- **第二步，交集为零**：$W \cap W^{\perp} = \{\mathbf{0}\}$——若 $\mathbf{v}$ 同时属于两者，则 $\mathbf{v}$ 与自己正交（$\mathbf{v}\cdot\mathbf{v} = 0$），故 $\mathbf{v} = \mathbf{0}$。
- **第三步，直和的维数公式**：由 Grassmann 公式，$\dim(W + W^{\perp}) = \dim W + \dim W^{\perp} - \dim(W\cap W^{\perp}) = n - 0 = n$，所以 $W + W^{\perp} = \mathbb{R}^n$。
- **第四步，唯一性**：若 $\mathbf{x} = \mathbf{w} + \mathbf{w}^{\perp} = \mathbf{w}' + \mathbf{w}^{\perp\prime}$，相减得 $\mathbf{w} - \mathbf{w}' = \mathbf{w}^{\perp\prime} - \mathbf{w}^{\perp} \in W \cap W^{\perp} = \{\mathbf{0}\}$，故两部分唯一。

<span class="marginnote"><strong>正交分解 = 唯一的「两个方向相加」</strong>：任意向量都有且只有一种方式写成「子空间内 + 垂直补内」之和。这个分解是傅里叶展开（投影到三角函数张成的子空间）、PCA（投影到主方向）、最小二乘（投影到列空间）的共同数学内核。</span>

## 4 正交补与投影的衔接

正交分解 $\mathbf{x} = \mathbf{w} + \mathbf{w}^{\perp}$ 里，$\mathbf{w}$ 是 $\mathbf{x}$ 在 $W$ 上的**正交投影**，$\mathbf{w}^{\perp}$ 是**残差**。三句关键性质：

- $\mathbf{w}$ 是 $W$ 中离 $\mathbf{x}$ **最近**的向量：$\|\mathbf{x} - \mathbf{w}\| \le \|\mathbf{x} - \mathbf{u}\|$ 对所有 $\mathbf{u} \in W$（投影 = 最近点）；
- 残差 $\mathbf{w}^{\perp} = \mathbf{x} - \mathbf{w}$ 与 $W$ 中每个向量垂直；
- 投影算子 $P_W$（把 $\mathbf{x}$ 映到 $\mathbf{w}$）是**线性变换**，且 $P_W^2 = P_W$（投影两次不变）。

**重点**：正交补让「最近点」问题变成「找垂直方向」问题——**最佳逼近 = 残差垂直于被逼近的子空间**。这是第八篇最小二乘的全部几何：$b$ 的最佳近似 $\hat{b} = A\hat{x}$ 满足「$b - A\hat{x}$ 垂直于列空间」。

## 5 例子：两个正交补

**例 1**：$\mathbb{R}^3$ 中，$W$ 是平面 $x + y + z = 0$。取 $W$ 的基 $\{(1,-1,0), (1,0,-1)\}$。$W^{\perp}$ 由「与两者都垂直」的方向张成：$(1,-1,0)\cdot\mathbf{x} = 0$、$(1,0,-1)\cdot\mathbf{x} = 0$，解得 $\mathbf{x} = t(1,1,1)^T$——$W^{\perp}$ 是沿 $(1,1,1)$ 的直线。$\dim W + \dim W^{\perp} = 2 + 1 = 3$ ✓。

**例 2**：四子空间实例的几何意义。$A = \begin{pmatrix} 1 & 2 \\ 2 & 4 \end{pmatrix}$，$\operatorname{Col}(A)$ 是 $(1,2)^T$ 方向直线，$\operatorname{Nul}(A^T)$ 是 $y_1 + 2y_2 = 0$ 的直线，两直线垂直。$\mathbb{R}^2 = \operatorname{Col}(A) \oplus \operatorname{Nul}(A^T)$——**输出空间被列空间与左零空间正交瓜分**。

**补充｜正交补在「解方程」里的位置**：回顾四子空间定理：$\operatorname{Nul}(A) = \operatorname{Row}(A)^{\perp}$，$\operatorname{Nul}(A^T) = \operatorname{Col}(A)^{\perp}$。这两条把「解方程」与「正交」焊在一起——**$Ax = b$ 有解 ⟺ $b$ 与左零空间（列空间的正交补）垂直**。这是「可解性」的第二种判据：不需要算秩，只需检验 $b$ 是否正交于 $\operatorname{Nul}(A^T)$ 的一组基。在最小二乘里，这个判据解释「$b$ 为什么不可解」——它含有左零空间的分量，那一部分无论如何无法被 $A$ 表示。

**补充｜正交补在「解方程」里的位置**：回顾四子空间定理：$\operatorname{Nul}(A) = \operatorname{Row}(A)^{\perp}$，$\operatorname{Nul}(A^T) = \operatorname{Col}(A)^{\perp}$。这两条把「解方程」与「正交」焊在一起——**$Ax = b$ 有解 ⟺ $b$ 与左零空间（列空间的正交补）垂直**。这是「可解性」的第二种判据：不需要算秩，只需检验 $b$ 是否正交于 $\operatorname{Nul}(A^T)$ 的一组基。在最小二乘里，这个判据解释「$b$ 为什么不可解」——它含有左零空间的分量，那一部分无论如何无法被 $A$ 表示。

**辨析｜易错点：** 正交补的计算与符号：

- $W^{\perp}$ 的求法 = 解以 $W$ 的基为行向量的齐次方程组（造 $A$、消元、取基础解系）；
- $(W^{\perp})^{\perp} = W$——取补两次回到自身；
- $\dim W + \dim W^{\perp} = n$——维数互补恒成立。

**「正交补 = 齐次解空间」**是计算层面的唯一入口。

**补充｜正交补的三个「关键等式」**：

- $W \cap W^{\perp} = \{\mathbf{0}\}$——只有零向量同时属于两者；
- $\dim W + \dim W^{\perp} = n$——维数互补；
- $(W^{\perp})^{\perp} = W$——取补两次还原。

**「交集为零、维数互补、二次还原」**是正交补理论的三大支柱，也是判断「一个空间是不是另一个的正交补」的检验标准。

**补充｜一句话**：**「$W^{\perp}$ = 与 $W$ 完全垂直的方向全体」**——两者直和填满全空间，投影与最小二乘都从这里起步。

**补充｜学习地图**：正交补是「正交分解定理」的载体，而正交分解正是第八篇「投影与最小二乘」的全部几何基础——「拆成子空间内 + 垂直补内」在下一篇变成「投影 + 残差」。

## 6 小结

- **定义**：$W^{\perp} = \{\mathbf{x} \mid \mathbf{x}\cdot\mathbf{w} = 0, \forall \mathbf{w} \in W\}$，是子空间。
- **性质**：$\dim W + \dim W^{\perp} = n$；$(W^{\perp})^{\perp} = W$；$W \cap W^{\perp} = \{\mathbf{0}\}$。
- **分解**：$\mathbb{R}^n = W \oplus W^{\perp}$，每个向量唯一拆成两部分。
- **计算**：正交补 = 以 $W$ 的基为行向量的齐次方程组的解空间。
- **衔接**：正交分解 = 投影 + 残差；投影 = 最近点。

在下一节，我们将把四个子空间与正交补放进同一个故事——**线性方程组解的几何图景**，为最小二乘做最后的铺垫。
