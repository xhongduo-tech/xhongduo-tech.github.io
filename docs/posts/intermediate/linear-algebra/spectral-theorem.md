---
title: 谱定理：对称矩阵的特征分解
date: 2026-08-08
---

# 谱定理：对称矩阵的特征分解

<div class="epigraph">
<p>实对称矩阵的谱定理，是线性代数最优雅的定理之一：每个实对称矩阵都能正交对角化，特征分解把矩阵按「方向 × 强度」完全展开。</p>
<footer>—— 希尔伯特（David Hilbert）</footer>
</div>

<div class="article-byline">
<p>第二级 · 线性代数 ｜ Strang《Introduction to Linear Algebra》§6.4 ｜ 2026-08-08</p>
</div>

## 为什么从谱定理开始

第五篇我们已经接触过实对称矩阵的正交对角化。本节把这一结果提升为**谱定理（spectral theorem）**，并用它统一「特征分解」这个矩阵分解家族的中心成员。谱定理是协方差矩阵 PCA、SVD、量子力学可观测量的数学根基。<span class="marginnote">「谱」一词来自物理学：光的分光得到光谱，矩阵的「谱」就是它的特征值集合。<strong>谱定理说：对称矩阵完全被它的谱决定</strong>——特征值给出强度，特征向量给出方向，矩阵 = 各方向的强度叠加。这是「$A = \sum \lambda_i \mathbf{q}_i\mathbf{q}_i^T$」的物理直觉。</span>

本节重述谱定理、给出特征分解的完整公式，并把它与其他分解统一比较。

## 1 谱定理

**定理（谱定理 / Spectral Theorem）**：设 $A$ 是 $n$ 阶实对称矩阵，则存在正交矩阵 $Q$ 与对角矩阵 $\Lambda$，使

$$
A = Q\Lambda Q^T
$$

其中 $\Lambda = \operatorname{diag}(\lambda_1, \cdots, \lambda_n)$ 是 $A$ 的特征值（实数），$Q$ 的列是对应的单位特征向量（构成标准正交基）。

**重点**：谱定理是「正交对角化」的完整陈述——**任何实对称矩阵都能用正交矩阵对角化**。相比一般可对角化 $A = P\Lambda P^{-1}$，这里 $P$ 可换成正交 $Q$，$P^{-1}$ 换成 $Q^T$。这是「正交性 + 对称性」相遇的完美结果。

**推论（秩一展开）**：

$$
A = \lambda_1 \mathbf{q}_1\mathbf{q}_1^T + \lambda_2 \mathbf{q}_2\mathbf{q}_2^T + \cdots + \lambda_n \mathbf{q}_n\mathbf{q}_n^T
$$

**每个对称矩阵 = 特征值加权的投影矩阵之和**。

## 2 特征分解与其他分解的统一

把目前学过的矩阵分解放在一张表里对照：

| 分解 | 形式 | 适用矩阵 | 因子 |
| --- | --- | --- | --- |
| LU | $A = LU$ | 主元好的方阵 | 下三角 × 上三角 |
| Cholesky | $A = LL^T$ | 对称正定 | 下三角 × 转置 |
| QR | $A = QR$ | 列满秩 | 正交 × 上三角 |
| 特征分解 | $A = Q\Lambda Q^T$ | 可对角化（对称必有） | 正交 × 对角 × 正交转置 |
| SVD | $A = U\Sigma V^T$ | 任意矩阵 | 正交 × 对角 × 正交转置（第十篇） |

**重点**：**特征分解只对「可对角化」的矩阵成立**，其中对称矩阵**永远**成立且 $Q$ 可取正交；SVD 对**任意**矩阵成立——它是特征分解对非对称/长方形的推广。**SVD 是谱定理的「无对称性版本」**，第十篇的核心。

**辨析｜易错点：** 谱定理要求实对称（$A^T = A$）。复矩阵的对应版本是「Hermite 矩阵」（$A^* = A$，共轭转置），特征值仍为实数。非对称实矩阵没有谱定理的保障：特征值可能是复数、可能不可对角化。

## 3 公式解析：$\lambda_{\max}$ 的变分刻画

谱定理最深刻的应用之一是**特征值的变分刻画（Rayleigh 商）**：设 $\lambda_1 \ge \lambda_2 \ge \cdots \ge \lambda_n$ 是实对称 $A$ 的特征值，则

$$
\lambda_{\max} = \max_{\mathbf{x} \ne \mathbf{0}} \frac{\mathbf{x}^T A \mathbf{x}}{\mathbf{x}^T \mathbf{x}}, \qquad
\lambda_{\min} = \min_{\mathbf{x} \ne \mathbf{0}} \frac{\mathbf{x}^T A \mathbf{x}}{\mathbf{x}^T \mathbf{x}}
$$

拆成四步：

- **第一步，Rayleigh 商**：比值 $R(\mathbf{x}) = \frac{\mathbf{x}^TA\mathbf{x}}{\mathbf{x}^T\mathbf{x}}$ 称为 Rayleigh 商——它度量「沿 $\mathbf{x}$ 方向，$A$ 的相对伸缩」。
- **第二步，谱展开代入**：用 $A = \sum\lambda_i\mathbf{q}_i\mathbf{q}_i^T$，$\mathbf{x} = \sum c_i\mathbf{q}_i$，则 $R(\mathbf{x}) = \frac{\sum\lambda_i c_i^2}{\sum c_i^2}$——是特征值的**加权平均**。
- **第三步，极值**：加权平均介于 $\lambda_{\min}$ 与 $\lambda_{\max}$ 之间；取 $\mathbf{x} = \mathbf{q}_{\max}$ 达到 $\lambda_{\max}$，取 $\mathbf{q}_{\min}$ 达到 $\lambda_{\min}$。
- **第四步，中间特征值**：Courant-Fischer 变分原理进一步把 $\lambda_k$ 刻画为「$k$ 维子空间上的约束极值」。**特征值 = 二次型的极值，无需解特征方程**——这是数值优化求特征值的理论基础。

<span class="marginnote">Rayleigh 商把「特征值」变成「优化问题」：<strong>最大特征值 = Rayleigh 商的最大值</strong>。幂法（反复乘 $A$ 再归一化）正是沿这个方向迭代逼近 $\lambda_{\max}$。PCA 找「方差最大的方向」也正是最大化 Rayleigh 商（协方差矩阵的商）。</span>

## 4 谱定理的应用

- **二次型标准形**：$f = \mathbf{x}^TA\mathbf{x} = \sum\lambda_i y_i^2$（正交变换下），特征值符号决定曲面的类型（第五篇）。
- **矩阵函数**：$f(A) = Qf(\Lambda)Q^T$，如 $A^k = Q\Lambda^k Q^T$、$e^A = Qe^{\Lambda}Q^T$——**对称矩阵的函数就是「特征值取函数」**。
- **PCA**：协方差矩阵 $C = Q\Lambda Q^T$，特征向量即主方向，特征值即方差（第十一篇）。
- **稳定性**：动态系统 $\dot{\mathbf{x}} = A\mathbf{x}$ 稳定 ⟺ $A$ 的特征值实部全负；对称时只需看特征值符号。

**重点**：**谱定理是「从矩阵到谱」的通用翻译器**——任何对称矩阵的问题，都能换成特征值的问题，而特征值的问题又常能换成 Rayleigh 商优化。这条「矩阵 → 谱 → 优化」的链条是现代数据科学的方法论核心。

## 5 一个完整例子：谱分解实操

$A = \begin{pmatrix} 3 & 1 \\ 1 & 3 \end{pmatrix}$（贯穿全书）。特征值 $\lambda_1 = 4, \lambda_2 = 2$，单位特征向量 $\mathbf{q}_1 = \frac{1}{\sqrt2}(1,1)^T$、$\mathbf{q}_2 = \frac{1}{\sqrt2}(1,-1)^T$。

$$
A = 4\mathbf{q}_1\mathbf{q}_1^T + 2\mathbf{q}_2\mathbf{q}_2^T
$$

验证第一项：$4\cdot\frac12\begin{pmatrix} 1 & 1 \\ 1 & 1 \end{pmatrix} = 2\begin{pmatrix} 1 & 1 \\ 1 & 1 \end{pmatrix}$；第二项 $2\cdot\frac12\begin{pmatrix} 1 & -1 \\ -1 & 1 \end{pmatrix} = \begin{pmatrix} 1 & -1 \\ -1 & 1 \end{pmatrix}$。两项相加 $= \begin{pmatrix} 3 & 1 \\ 1 & 3 \end{pmatrix}$ ✓。

**辨析｜易错点：** 秩一展开中每一项 $\lambda_i\mathbf{q}_i\mathbf{q}_i^T$ 的秩是 1（外积）。**截断谱展开**（保留前 $k$ 项）给出最佳秩 $k$ 近似——这是 PCA 与图像压缩的数学依据（第十篇、第十一篇）。保留项越多，近似越准，但 $k$ 个最大特征值通常已捕获主要结构。

**补充｜数值求特征值为何用迭代而非特征多项式**：$n$ 阶矩阵的特征多项式是 $n$ 次方程，但其系数对浮点误差极敏感（Wilkinson 多项式是著名的病态例子），直接求根数值上不可靠。工程上求特征值几乎都用**迭代法**（幂法、QR 算法、Jacobi 旋转），不显式构造特征多项式。**「求特征值 = 迭代，而不是解多项式」**——这是数值线性代数与教科书的重要分界（第九篇）。理解这一点，才明白为什么「算出特征多项式」在工程里并不常用。

**补充｜数值求特征值为何用迭代而非特征多项式**：$n$ 阶矩阵的特征多项式是 $n$ 次方程，但其系数对浮点误差极敏感（Wilkinson 多项式是著名的病态例子），直接求根数值上不可靠。工程上求特征值几乎都用**迭代法**（幂法、QR 算法、Jacobi 旋转），不显式构造特征多项式。**「求特征值 = 迭代，而不是解多项式」**——这是数值线性代数与教科书的重要分界（第九篇）。理解这一点，才明白为什么「算出特征多项式」在工程里并不常用。

**辨析｜易错点：** 谱定理的适用边界：

- 实对称（$A^T = A$）才有 $A = Q\Lambda Q^T$（$Q$ 实正交）；
- Hermite 矩阵（复 $A^* = A$）有 $A = U\Lambda U^*$（$U$ 酉）；
- 普通实矩阵**没有**谱定理——特征值可能是复数或矩阵不可对角化，此时用 Jordan 形或 SVD。

**「对称才谱分解，非对称求 SVD」**是判断分解方法的金句。

**补充｜谱定理的「一句话」**：**「实对称矩阵完全由它的特征值（谱）决定」**——$A = \sum\lambda_i\mathbf{q}_i\mathbf{q}_i^T$ 把矩阵拆成「特征值加权的投影」，这是对称矩阵的一切应用（PCA、二次型、矩阵函数）的共同起点。

## 6 小结

- **谱定理**：$A = Q\Lambda Q^T$，实对称矩阵正交对角化，特征值全实。
- **秩一展开**：$A = \sum\lambda_i\mathbf{q}_i\mathbf{q}_i^T$，方向 × 强度分解。
- **Rayleigh 商**：$\lambda_{\max} = \max R(\mathbf{x})$，特征值 = 优化问题。
- **分解家族**：特征分解是 SVD 的对称特例；对称矩阵必有谱分解。
- **应用**：二次型、矩阵函数、PCA、稳定性分析。

在下一节，我们将处理「不可对角化」的情形——**Jordan 标准形简介**，看看特征向量不够时矩阵还能化成什么最简形。
