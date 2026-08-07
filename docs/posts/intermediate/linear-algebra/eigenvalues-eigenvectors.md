---
title: 方阵的特征值与特征向量
date: 2026-08-08
---

# 方阵的特征值与特征向量

<div class="epigraph">
<p>一个矩阵最值得知道的，不是它对每个向量做了什么，而是它在哪些方向上「只伸缩、不旋转」——那些方向，就是特征向量。</p>
<footer>—— 欧拉（Leonhard Euler，特征值理论的早期探索者）</footer>
</div>

<div class="article-byline">
<p>第二级 · 线性代数 ｜ 同济《线性代数》§5.3 ｜ 2026-08-08</p>
</div>

## 为什么从特征值开始

面对一个矩阵，我们想知道它的「性格」。对一般的向量，$A\mathbf{x}$ 既改变长度又改变方向，难以描述。但对某些特殊的向量，$A$ 只是把它**拉伸或压缩**，方向不变——这些向量揭示了矩阵最本质的「轴」。这就是**特征向量与特征值**。<span class="marginnote">特征值理论是线性代数的王冠：相似对角化、谱定理、SVD、微分方程组的解、马尔可夫链的稳态、PageRank、主成分分析——几乎一切「分解矩阵看本质」的应用都以特征值为核心。斯特朗把特征值比作矩阵的「DNA」：矩阵可以千变万化（相似），但特征值不变。</span>

理解特征值的关键时刻：**$A\mathbf{x} = \lambda\mathbf{x}$ 不是一个方程组的解，而是一个「找方向」的问题**。

## 1 定义与几何意义

**核心概念**：设 $A$ 是 $n$ 阶方阵，若存在**非零**向量 $\mathbf{x}$ 和数 $\lambda$ 使得

$$
A\mathbf{x} = \lambda\mathbf{x}
$$

则称 $\lambda$ 为 $A$ 的一个**特征值（eigenvalue）**，$\mathbf{x}$ 为 $A$ 的对应于 $\lambda$ 的**特征向量（eigenvector）**。

**重点**：特征向量在 $A$ 作用下**方向不变**（可能反向，若 $\lambda < 0$），只被**伸缩** $\lambda$ 倍。$\lambda$ 的绝对值是伸缩倍率，符号表示是否反向。几何上，特征向量是矩阵变换的「不动轴」：沿着这些轴，变换退化成纯伸缩。

**辨析｜易错点：** 特征向量**必须非零**（零向量满足等式但不作数）。特征值是**可以为零**的：$\lambda = 0$ 意味着 $A\mathbf{x} = \mathbf{0}$ 有非零解，即 $A$ 奇异。所以「$0$ 是 $A$ 的特征值」$\Leftrightarrow$「$A$ 不可逆」$\Leftrightarrow \det A = 0$。

## 2 特征方程与特征多项式

从 $A\mathbf{x} = \lambda\mathbf{x}$ 变形：

$$
(A - \lambda I)\mathbf{x} = \mathbf{0}
$$

非零解存在的充要条件是系数矩阵奇异，即

$$
\det(A - \lambda I) = 0
$$

**核心概念**：$f(\lambda) = \det(A - \lambda I)$ 称为 $A$ 的**特征多项式（characteristic polynomial）**，方程 $f(\lambda) = 0$ 称为**特征方程**。

**重点**：$n$ 阶矩阵的特征多项式是 $n$ 次多项式，按代数基本定理，在复数范围内**恰有 $n$ 个特征值（计重数）**。**即使 $A$ 是实矩阵，特征值也可能是复数**（如旋转矩阵 $R(90°)$ 的特征值是 $\pm i$）——这是必须用复数语言的地方。

**辨析｜易错点：** 特征向量必须与特征值**一一配对**：每个特征值 $\lambda$ 对应一组特征向量——即齐次方程组 $(A - \lambda I)\mathbf{x} = \mathbf{0}$ 的解空间（称为 $\lambda$ 的**特征空间**），其维数称为 $\lambda$ 的**几何重数**。特征值 $\lambda$ 在特征多项式中的重数叫**代数重数**。两者关系是下节对角化的核心判据。

## 3 求特征值与特征向量的标准流程

**算法**：

1. 写出特征多项式 $\det(A - \lambda I)$，求特征方程的解 $\lambda_1, \cdots, \lambda_n$。
2. 对每个 $\lambda_i$，解齐次方程组 $(A - \lambda_i I)\mathbf{x} = \mathbf{0}$，基础解系即对应特征向量。

**一个完整的例子**：求 $A = \begin{pmatrix} 3 & 1 \\ 1 & 3 \end{pmatrix}$ 的特征值与特征向量。

- 特征多项式：
  $$
  \det(A - \lambda I) = \begin{vmatrix} 3-\lambda & 1 \\ 1 & 3-\lambda \end{vmatrix} = (3-\lambda)^2 - 1 = \lambda^2 - 6\lambda + 8 = (\lambda - 2)(\lambda - 4)
  $$
  特征值 $\lambda = 2, 4$。
- $\lambda = 2$：解 $(A - 2I)\mathbf{x} = \begin{pmatrix} 1 & 1 \\ 1 & 1 \end{pmatrix}\mathbf{x} = \mathbf{0}$，得 $x_1 + x_2 = 0$，特征向量 $t(1, -1)^T$。
- $\lambda = 4$：解 $(A - 4I)\mathbf{x} = \begin{pmatrix} -1 & 1 \\ 1 & -1 \end{pmatrix}\mathbf{x} = \mathbf{0}$，得 $x_1 = x_2$，特征向量 $t(1, 1)^T$。

## 4 公式解析：特征值的两个「免费定理」

求具体特征值常常繁琐，但有两个定理让大量计算免于白费：

**定理 1（迹与行列式）**：设 $A$ 的特征值为 $\lambda_1, \cdots, \lambda_n$（计重数），则

$$
\lambda_1 + \lambda_2 + \cdots + \lambda_n = \operatorname{tr}(A) \quad (\text{主对角元素之和})
$$

$$
\lambda_1 \lambda_2 \cdots \lambda_n = \det A
$$

拆开理解：

- **第一步，特征多项式展开**：$\det(A - \lambda I) = (-\lambda)^n + (\operatorname{tr} A)(-\lambda)^{n-1} + \cdots + \det A$。系数 $\operatorname{tr} A$ 来自主对角线乘积项，常数项是 $\det A$。
- **第二步，与根的对称关系**：$n$ 次多项式根与系数的关系——根的和 = 次高次项系数变号，根的积 = 常数项（乘 $(-1)^n$ 后）。
- **第三步，对号入座**：根的和 = $\operatorname{tr} A$，根的积 = $\det A$。
- **第四步，应用**：这两个关系是**快速检验**特征值算得对不对的标尺，也是推导「可逆矩阵特征值取倒数」「$A^{-1}$ 的迹」等结论的工具。

**定理 2（不同特征值的特征向量无关）**：属于**不同特征值**的特征向量线性无关。

**重点**：这条定理保证：若 $A$ 有 $n$ 个互不相同的特征值，则 $A$ 恰有 $n$ 个线性无关的特征向量——**可对角化**（下节）。它把「特征值互异」与「可对角化」直接挂钩。

<span class="marginnote">「迹 = 特征值之和」「行列式 = 特征值之积」是特征值最重要的两把「免费标尺」。在数据科学里，协方差矩阵的迹 = 总方差（第十一篇），所以<strong>总方差 = 特征值之和</strong>——PCA 正是按特征值大小排序来选主成分。</span>

## 5 特征值在微分方程与动态系统中的意义

考虑一阶线性差分 $\mathbf{x}_{k+1} = A\mathbf{x}_k$（马尔可夫链、动力系统）。若 $\mathbf{x}_0$ 是特征向量（$A\mathbf{x}_0 = \lambda\mathbf{x}_0$），则

$$
\mathbf{x}_k = \lambda^k \mathbf{x}_0
$$

每步乘以 $\lambda^k$。于是：

- $|\lambda| < 1$：收敛到零（稳定）；
- $|\lambda| > 1$：发散（不稳定）；
- $|\lambda| = 1$：沿特征方向保持幅值（临界）。

**重点**：**动态系统的长期行为由特征值的模决定**。任意初始向量按特征向量展开后，各分量独立演化——这就是把微分方程组解耦的核心思想（第九篇矩阵指数）。PageRank 的稳态、马尔可夫链的收敛（第十一篇），都是这套语言的具体应用。

## 6 小结

- **定义**：$A\mathbf{x} = \lambda\mathbf{x}$，$\mathbf{x} \ne \mathbf{0}$；特征向量是「不动轴」。
- **求法**：解 $\det(A - \lambda I) = 0$，再解 $(A - \lambda I)\mathbf{x} = \mathbf{0}$。
- **迹与行列式**：特征值之和 $= \operatorname{tr} A$，之积 $= \det A$。
- **无关性**：不同特征值的特征向量线性无关。
- **动态系统**：$|\lambda| < 1$ 稳定，$> 1$ 发散，$= 1$ 临界。

在下一节，我们将把特征向量「对齐成基」——**相似矩阵与相似对角化**，回答「何时能把 $A$ 变成对角矩阵」。
