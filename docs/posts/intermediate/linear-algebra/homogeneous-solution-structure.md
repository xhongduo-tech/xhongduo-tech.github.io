---
title: 齐次线性方程组解的结构与基础解系
date: 2026-08-08
---

# 齐次线性方程组解的结构与基础解系

<div class="epigraph">
<p>齐次方程组的解集不是一堆散点，而是一个空间——而空间只需要一组基，就能被完整地描述出来。</p>
<footer>—— 斯特朗（Gilbert Strang，《Introduction to Linear Algebra》）</footer>
</div>

<div class="article-byline">
<p>第二级 · 线性代数 ｜ 同济《线性代数》§4.4 ｜ 2026-08-08</p>
</div>

## 为什么从齐次方程组开始

第三篇我们学会了「判定解的情况」，这一篇要回答更深的问题：**无穷多解时，解集长什么样？** 答案极其漂亮：齐次方程组 $Ax = 0$ 的解集构成一个**向量空间**（零空间），而这个空间只需要一组基——**基础解系**——就能完整描述。<span class="marginnote">「解集是空间」这个事实是线性代数的核心魅力：因为 $A(\mathbf{u}+\mathbf{v}) = A\mathbf{u} + A\mathbf{v} = 0 + 0 = 0$ 且 $A(\lambda\mathbf{u}) = \lambda A\mathbf{u} = 0$，齐次解集对加法和数乘封闭——它天然满足向量空间的全部八条律。这是第七篇「零空间」概念的正式预演。</span>

基础解系的思想贯穿始终：解空间 = 零空间，它的维数就是自由变量的个数。理解本节，非齐次方程组解的结构（下一节）就水到渠成。

## 1 齐次方程组与零空间

**核心概念**：形如 $Ax = 0$（$A$ 是 $m \times n$ 矩阵）的方程组叫**齐次线性方程组（homogeneous system）**。它**永远有解**——至少 $\mathbf{x} = \mathbf{0}$（零解）。全部解构成的集合

$$
\operatorname{Nul}(A) = \{\mathbf{x} \mid A\mathbf{x} = \mathbf{0}\}
$$

称为 $A$ 的**零空间（null space）**（第七篇的正式主题）。

**重点**：零空间是 $\mathbb{R}^n$ 的**子空间**（封闭、含零、含负元），所以它有自己的基与维数。关键定理：

$$
\dim \operatorname{Nul}(A) = n - \operatorname{rank} A
$$

即**解空间的维数 = 自由变量个数**。这就是第三篇那条「自由度公式」的空间语言版本。

## 2 基础解系的定义

**核心概念**：设 $Ax = 0$ 有无穷多解，若零空间中存在一组向量 $\boldsymbol{\xi}_1, \cdots, \boldsymbol{\xi}_r$ 满足：

- 线性无关；
- $Ax = 0$ 的**每个解**都能由它们线性表示；

则称这组向量为 $Ax = 0$ 的**基础解系（fundamental system of solutions）**，$r = n - \operatorname{rank} A$。

**重点**：基础解系 = 零空间的一组**基**。找到了基础解系，就找到了全部解：

$$
\mathbf{x} = c_1\boldsymbol{\xi}_1 + c_2\boldsymbol{\xi}_2 + \cdots + c_r\boldsymbol{\xi}_r
$$

其中 $c_i$ 任意。这就是齐次方程组的**通解（general solution）**。

<span class="marginnote">基础解系的个数 $n - \operatorname{rank} A$ 与高斯消元中的「自由变量个数」完全一致——每个自由变量取 1、其余自由变量取 0，就得到一组基。这个「把自由变量逐个点亮」的构造法是最可靠的。</span>

## 3 求基础解系的标准流程

**算法（基础解系）**，设 $A$ 是 $m \times n$ 矩阵，$\operatorname{rank} A = r$：

1. 对 $A$ 行初等变换化行最简形 RREF。
2. 找出主元列对应的 $r$ 个主变量与 $n - r$ 个自由变量。
3. 令第 $j$ 个自由变量取 1、其余自由变量取 0（$j = 1, \cdots, n-r$），回代解出主变量，得到解向量 $\boldsymbol{\xi}_j$。
4. $\boldsymbol{\xi}_1, \cdots, \boldsymbol{\xi}_{n-r}$ 即一组基础解系。

**一个完整的例子**：解

$$
\begin{cases}
x_1 + 2x_2 + x_4 = 0 \\
x_2 - x_3 + x_4 = 0
\end{cases}
$$

即 $A = \begin{pmatrix} 1 & 2 & 0 & 1 \\ 0 & 1 & -1 & 1 \end{pmatrix}$。化 RREF：用 $r_1 - 2r_2$ 得

$$
\begin{pmatrix} 1 & 0 & 2 & -1 \\ 0 & 1 & -1 & 1 \end{pmatrix}
$$

主元在第 1、2 列，$x_1, x_2$ 是主变量，$x_3, x_4$ 自由（$n - r = 4 - 2 = 2$ 个）。

- 取 $x_3 = 1, x_4 = 0$：$x_2 = 1$，$x_1 = -2$，得 $\boldsymbol{\xi}_1 = (-2, 1, 1, 0)^T$；
- 取 $x_3 = 0, x_4 = 1$：$x_2 = -1$，$x_1 = 1$，得 $\boldsymbol{\xi}_2 = (1, -1, 0, 1)^T$。

通解：

$$
\mathbf{x} = c_1 \begin{pmatrix} -2 \\ 1 \\ 1 \\ 0 \end{pmatrix} + c_2 \begin{pmatrix} 1 \\ -1 \\ 0 \\ 1 \end{pmatrix}
$$

验证 $\boldsymbol{\xi}_1$：$1\cdot(-2) + 2\cdot1 + 0 + 1\cdot0 = 0$ ✓；第二方程 $1 - 1 + 0 = 0$ ✓。

## 4 公式解析：为什么基础解系恰好 $n - r$ 个

$r = \dim \operatorname{Nul}(A) = n - \operatorname{rank} A$ 这条公式，拆成四步理解：

- **第一步，主变量被自由变量控制**：RREF 中，每个主变量都写成自由变量的线性组合（每个方程一个主变量）。所以解完全由 $n - r$ 个自由变量的取值决定。
- **第二步，自由变量是「开关」**：把第 $j$ 个自由变量设 1、其余设 0，得到的解 $\boldsymbol{\xi}_j$ 是「只按下一个开关」的纯解。$n - r$ 个纯解两两线性无关（它们在第 $j$ 个自由变量位置上有不同的 1/0 配置）。
- **第三步，张成性**：任意解都可以写成「各开关按到对应位置」的叠加，即任意解 $= \sum c_j \boldsymbol{\xi}_j$。
- **第四步，结论**：$\boldsymbol{\xi}_j$ 线性无关且张成零空间，是一组基，个数 $n - r$，故 $\dim \operatorname{Nul}(A) = n - r$。

<span class="marginnote">这个构造法的本质：<strong>把解空间「坐标系化」</strong>。自由变量就是解空间的一组「坐标轴」，每个基础解系向量是一条坐标轴方向。SVD 与 PCA（第十篇、第十一篇）正是在零空间的补空间上找「数据方差最大的坐标轴」。</span>

## 5 与线性相关判据的衔接

基础解系的存在性反过来给出线性无关判据的新形式：

- $A$ 的列线性无关 $\Leftrightarrow Ax = 0$ 只有零解 $\Leftrightarrow \operatorname{Nul}(A) = \{\mathbf{0}\}$ $\Leftrightarrow \dim \operatorname{Nul}(A) = 0$ $\Leftrightarrow \operatorname{rank} A = n$。

**辨析｜易错点：** 「齐次方程组只有零解」与「有非零解」的边界就是 $\operatorname{rank} A$ 是否等于 $n$。若 $m < n$（方程少于未知量），$\operatorname{rank} A \le m < n$，**齐次方程组必有非零解**——方程个数少于未知量时，齐次方程组永远不会「只有零解」。这是秩不等式 $\operatorname{rank} \le m$ 的直接推论，考试常考。

**另一个应用**：已知 $A$ 是 $n$ 阶方阵，若 $\det A = 0$，则 $Ax = 0$ 有非零解；基础解系就给出「$\det$ 为零时的特征向量雏形」——这与第五篇特征值理论 $(\lambda I - A)\mathbf{x} = 0$ 完全同构：特征向量就是 $(\lambda I - A)$ 的零空间里的非零向量。

## 6 小结

- **零空间**：$Ax = 0$ 的解集 $\operatorname{Nul}(A)$，是 $\mathbb{R}^n$ 的子空间。
- **基础解系**：零空间的一组基，个数 $= n - \operatorname{rank} A$。
- **通解**：基础解系的任意线性组合。
- **求法**：RREF → 自由变量逐个置 1 → 回代。
- **衔接**：$Ax = 0$ 只有零解 $\Leftrightarrow \operatorname{rank} A = n$ $\Leftrightarrow$ 列线性无关。

在下一节，我们将处理右端非零的情形——**非齐次线性方程组解的结构**，揭示「特解 + 齐次通解」的完整图景。
