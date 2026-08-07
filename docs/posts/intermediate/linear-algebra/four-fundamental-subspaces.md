---
title: 四个基本子空间及其维数关系
date: 2026-08-08
---

# 四个基本子空间及其维数关系

<div class="epigraph">
<p>线性代数最核心的图景，就是一张由四个子空间构成的「十字图」：列空间与左零空间在输出空间里互为正交补，行空间与零空间在输入空间里互为正交补。</p>
<footer>—— 斯特朗（Gilbert Strang，《Introduction to Linear Algebra》）</footer>
</div>

<div class="article-byline">
<p>第二级 · 线性代数 ｜ Strang《Introduction to Linear Algebra》§3.3 ｜ 2026-08-08</p>
</div>

## 为什么从四个基本子空间开始

Strang 认为，线性代数的全部内容可以用一张「四个基本子空间」的图来概括。一个 $m \times n$ 矩阵 $A$ 同时孕育四个子空间：列空间与零空间（已见），加上行空间与左零空间。它们分别在 $\mathbb{R}^m$ 与 $\mathbb{R}^n$ 里两两配对、互相正交、维数互补。<span class="marginnote">这张「十字图」是理解一切矩阵分解的罗盘：<strong>LU 把消元写在行空间，SVD 把四个子空间同时对齐到正交基上</strong>。数据科学里，SVD 的核心（第十篇）就是让这四个子空间与奇异值方向精确对应。</span>

本节把四个子空间一次摆齐，给出它们的维数表与正交关系。

## 1 四个基本子空间的定义

设 $A$ 是 $m \times n$ 矩阵，$\operatorname{rank} A = r$：

**输出空间 $\mathbb{R}^m$ 里的两个**：

1. **列空间** $\operatorname{Col}(A) = \operatorname{span}\{A \text{ 的列}\}$，维数 $r$；
2. **左零空间** $\operatorname{Nul}(A^T) = \{\mathbf{y} \mid A^T\mathbf{y} = \mathbf{0}\}$，维数 $m - r$。

**输入空间 $\mathbb{R}^n$ 里的两个**：

3. **行空间** $\operatorname{Row}(A) = \operatorname{span}\{A \text{ 的行}\}$，维数 $r$；
4. **零空间** $\operatorname{Nul}(A) = \{\mathbf{x} \mid A\mathbf{x} = \mathbf{0}\}$，维数 $n - r$。

**重点**：左零空间的名称来源：$A^T\mathbf{y} = \mathbf{0}$ 等价于 $\mathbf{y}^T A = \mathbf{0}^T$——它是「左乘 $A$ 得到零行向量」的那些 $\mathbf{y}$（作为行向量左乘）。行空间与零空间在 $\mathbb{R}^n$，列空间与左零空间在 $\mathbb{R}^m$。

## 2 维数关系表

四个子空间的维数由 $r$ 完全决定：

| 子空间 | 记号 | 所在空间 | 维数 |
| --- | --- | --- | --- |
| 列空间 | $\operatorname{Col}(A)$ | $\mathbb{R}^m$ | $r$ |
| 左零空间 | $\operatorname{Nul}(A^T)$ | $\mathbb{R}^m$ | $m - r$ |
| 行空间 | $\operatorname{Row}(A)$ | $\mathbb{R}^n$ | $r$ |
| 零空间 | $\operatorname{Nul}(A)$ | $\mathbb{R}^n$ | $n - r$ |

**重点**：两两配对相加 = 全空间维数：

$$
\dim\operatorname{Col}(A) + \dim\operatorname{Nul}(A^T) = m, \qquad
\dim\operatorname{Row}(A) + \dim\operatorname{Nul}(A) = n
$$

**辨析｜易错点：** $\dim\operatorname{Row}(A) = \dim\operatorname{Col}(A) = r$——**行空间与列空间维数相同**（行秩 = 列秩，第三篇），但它们在**不同**的空间里（$\mathbb{R}^n$ vs $\mathbb{R}^m$）。这是初学者最易混淆的一点：维数相同 ≠ 空间相同。

## 3 正交补：四个子空间的几何关系

**核心概念**：设 $W$ 是 $\mathbb{R}^n$ 的子空间，$W$ 的**正交补（orthogonal complement）** 是

$$
W^{\perp} = \{\mathbf{x} \in \mathbb{R}^n \mid \mathbf{x} \cdot \mathbf{w} = 0 \text{ 对一切 } \mathbf{w} \in W\}
$$

即「与 $W$ 中每个向量都垂直」的全体。正交补本身是子空间，且 $\dim W^{\perp} = n - \dim W$。

**定理（基本子空间的正交补关系）**：

$$
\operatorname{Nul}(A) = \operatorname{Row}(A)^{\perp}, \qquad
\operatorname{Nul}(A^T) = \operatorname{Col}(A)^{\perp}
$$

**重点**：**零空间是行空间的正交补，左零空间是列空间的正交补**。这张图是 Strang 最著名的画面：

$$
\mathbb{R}^n = \operatorname{Row}(A) \oplus \operatorname{Nul}(A), \qquad
\mathbb{R}^m = \operatorname{Col}(A) \oplus \operatorname{Nul}(A^T)
$$

**行空间与零空间把 $\mathbb{R}^n$ 正交分解**；**列空间与左零空间把 $\mathbb{R}^m$ 正交分解**。四个子空间两两正交、维数互补——这是 SVD（第十篇）的几何骨架。

## 4 公式解析：为什么 $\operatorname{Nul}(A) = \operatorname{Row}(A)^{\perp}$

这是四子空间理论的核心证明，拆成四步：

- **第一步，先看包含**：任取 $\mathbf{x} \in \operatorname{Nul}(A)$（即 $A\mathbf{x} = \mathbf{0}$），$A$ 的第 $i$ 行 $\mathbf{r}_i$ 满足 $\mathbf{r}_i \cdot \mathbf{x} = 0$（行乘列 = 0）。所以 $\mathbf{x}$ 与每一行都垂直，$\mathbf{x} \in \operatorname{Row}(A)^{\perp}$。
- **第二步，再看另一方向**：任取 $\mathbf{x} \in \operatorname{Row}(A)^{\perp}$，则 $\mathbf{x}$ 与每一行垂直，故每一行乘 $\mathbf{x}$ 得 0，即 $A\mathbf{x} = \mathbf{0}$，$\mathbf{x} \in \operatorname{Nul}(A)$。
- **第三步，集合相等**：两个方向都成立，$\operatorname{Nul}(A) = \operatorname{Row}(A)^{\perp}$。
- **第四步，维数核对**：$\dim\operatorname{Nul}(A) = n - r$，$\dim\operatorname{Row}(A)^{\perp} = n - \dim\operatorname{Row}(A) = n - r$，维数一致，与集合相等互相印证。

<span class="marginnote"><strong>「行空间的正交补是零空间」</strong>的直觉：行空间是「方程组的行方向」，零空间是「所有行都与之垂直的输入方向」——解方程就是找「与每一行都垂直」的向量。这个几何视角让「解方程组」变成「找正交方向」。</span>

## 5 四个子空间的计算

**求行空间基**：把 $A$ 化行最简形，**RREF 的非零行**构成行空间的一组基（初等行变换保持行空间）。

**求列空间基**：RREF 的**主元列**对应原矩阵的列（第七篇第一节）。

**求左零空间**：把 $(A \mid I)$ 一起消元，或解 $A^T\mathbf{y} = \mathbf{0}$。

例：$A = \begin{pmatrix} 1 & 2 \\ 2 & 4 \\ 3 & 6 \end{pmatrix}$，$\operatorname{rank} = 1$。

- 行空间：$\operatorname{span}\{(1,2)\}$（$\mathbb{R}^2$ 中一维）；
- 零空间：$x_1 + 2x_2 = 0$，$\operatorname{span}\{(-2,1)^T\}$；
- 列空间：$\operatorname{span}\{(1,2,3)^T\}$（$\mathbb{R}^3$ 中一维）；
- 左零空间：$A^T\mathbf{y} = \begin{pmatrix} 1 & 2 & 3 \\ 2 & 4 & 6 \end{pmatrix}\mathbf{y} = \mathbf{0}$，即 $y_1 + 2y_2 + 3y_3 = 0$，二维子空间（$\mathbb{R}^3$ 中的平面）。

核对：行空间与零空间在 $\mathbb{R}^2$ 中正交互补（$(1,2)\cdot(-2,1) = 0$ ✓）；列空间与左零空间在 $\mathbb{R}^3$ 中正交互补（$(1,2,3)$ 与左零空间任一向量垂直 ✓）。

## 6 四个子空间与线性方程组

- $Ax = \mathbf{b}$ **有解** ⇔ $\mathbf{b} \in \operatorname{Col}(A)$ ⇔ $\mathbf{b}$ 与 $\operatorname{Nul}(A^T)$ 正交（因列空间是左零空间的正交补）；
- $Ax = \mathbf{b}$ 解**唯一** ⇔ $\operatorname{Nul}(A) = \{\mathbf{0}\}$。

**重点**：第二行给出「可解性」的另一种判据：**$b$ 必须与左零空间的所有向量垂直**。这一条在最小二乘、伪逆理论中非常关键：当 $b$ 不落在列空间时，最小二乘把它「投影」到列空间上（第八篇），投影后的部分可解，残差部分落在左零空间。

## 7 小结

- **四子空间**：$\operatorname{Col}(A)$、$\operatorname{Nul}(A^T)$（在 $\mathbb{R}^m$）；$\operatorname{Row}(A)$、$\operatorname{Nul}(A)$（在 $\mathbb{R}^n$）。
- **维数**：$r, m-r, r, n-r$；行秩 = 列秩 = $r$。
- **正交补**：$\operatorname{Nul}(A) = \operatorname{Row}(A)^{\perp}$，$\operatorname{Nul}(A^T) = \operatorname{Col}(A)^{\perp}$。
- **分解**：$\mathbb{R}^n = \operatorname{Row}(A)\oplus\operatorname{Nul}(A)$，$\mathbb{R}^m = \operatorname{Col}(A)\oplus\operatorname{Nul}(A^T)$。
- **计算**：RREF 非零行 = 行空间基，主元列 = 列空间基。

在下一节，我们将深入学习子空间的垂直结构本身——**子空间的正交补**，并引出投影与最小二乘的几何基础。
