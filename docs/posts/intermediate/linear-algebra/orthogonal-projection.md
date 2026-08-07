---
title: 正交投影与投影矩阵
date: 2026-08-08
---

# 正交投影与投影矩阵

<div class="epigraph">
<p>投影是把一个向量「压」到某个子空间上——压得恰到好处时，残差与被压到的平面垂直，这是整个最小二乘的秘密。</p>
<footer>—— 斯特朗（Gilbert Strang，《Introduction to Linear Algebra》）</footer>
</div>

<div class="article-byline">
<p>第二级 · 线性代数 ｜ Strang《Introduction to Linear Algebra》§4.2 ｜ 2026-08-08</p>
</div>

## 为什么从正交投影开始

上一节末尾我们遇到一个关键时刻：$b \notin \operatorname{Col}(A)$ 时方程组无解，但可以「退而求其次」——在列空间里找离 $b$ 最近的向量。这个「找最近点」的操作就是**正交投影**，它的计算核心是一个**投影矩阵**。<span class="marginnote">投影矩阵 $P$ 满足 $P^2 = P$（幂等）与 $P^T = P$（对称），它把任意向量「压」到目标子空间。理解 $P$，就理解了最小二乘（投影到列空间）、傅里叶展开（投影到三角函数张成的空间）、以及 PCA（投影到主方向）——<strong>它们全是「同一个投影思想」的不同子空间</strong>。</span>

本节从一维投影出发，建立投影矩阵的一般公式。

## 1 一维投影：投影到一条直线上

设 $\mathbf{a}$ 是 $\mathbb{R}^n$ 中的非零向量，$W = \operatorname{span}\{\mathbf{a}\}$ 是过原点的一条直线。向量 $\mathbf{b}$ 在 $W$ 上的**正交投影**是

$$
\hat{\mathbf{b}} = \frac{\mathbf{b}\cdot\mathbf{a}}{\mathbf{a}\cdot\mathbf{a}}\,\mathbf{a} = \frac{\mathbf{a}\mathbf{a}^T}{\mathbf{a}^T\mathbf{a}}\,\mathbf{b}
$$

投影矩阵

$$
P = \frac{\mathbf{a}\mathbf{a}^T}{\mathbf{a}^T\mathbf{a}}
$$

**重点**：$\mathbf{a}\mathbf{a}^T$ 是 $n \times n$ 的**外积**矩阵，除以 $\mathbf{a}^T\mathbf{a}$（长度平方）后，$P$ 是秩 1 矩阵，作用在任意向量上给出它在 $\mathbf{a}$ 方向的投影。

**验证**：$P\mathbf{b} = \frac{\mathbf{a}(\mathbf{a}^T\mathbf{b})}{\mathbf{a}^T\mathbf{a}}$——分子是「$\mathbf{a}$ 乘以一个数」，确实沿 $\mathbf{a}$ 方向；残差 $\mathbf{b} - P\mathbf{b}$ 与 $\mathbf{a}$ 垂直：$\mathbf{a}\cdot(\mathbf{b} - P\mathbf{b}) = \mathbf{a}^T\mathbf{b} - \frac{\mathbf{a}^T\mathbf{a}\mathbf{a}^T\mathbf{b}}{\mathbf{a}^T\mathbf{a}} = 0$。

## 2 一般子空间的投影矩阵

设 $W = \operatorname{Col}(A)$，$A$ 是 $m \times n$ 矩阵且**列满秩**（$\operatorname{rank} A = n$，列线性无关）。$\mathbf{b}$ 在 $W$ 上的正交投影为

$$
\hat{\mathbf{b}} = A(A^TA)^{-1}A^T \mathbf{b}
$$

投影矩阵

$$
P = A(A^TA)^{-1}A^T
$$

**重点**：$A^TA$ 是 $n \times n$ 方阵，列满秩时**可逆**（第五篇正定判据：$A$ 列满秩 ⇒ $A^TA$ 正定）。$P$ 满足 $P^2 = P$、$P^T = P$，且 $PA = A$（投影不改变 $W$ 内的向量）。

**辨析｜易错点：** 公式需要 $A$ **列满秩**。若列相关，$A^TA$ 奇异，投影矩阵要用伪逆（第十篇）或改成其他求法。另外注意 $P$ 是 $m \times m$ 矩阵（作用在输出空间 $\mathbb{R}^m$ 上），而 $A^TA$ 是 $n \times n$——**维度别搞混**。

## 3 公式解析：为什么 $P = A(A^TA)^{-1}A^T$

推导拆成四步，全部建立在「残差垂直于 $W$」这一条上：

- **第一步，目标**：找 $\hat{\mathbf{b}} = A\hat{\mathbf{x}} \in W$ 使残差 $\mathbf{e} = \mathbf{b} - A\hat{\mathbf{x}}$ 垂直于 $W$（即垂直于 $A$ 的每一列）。
- **第二步，垂直条件**：$A^T\mathbf{e} = \mathbf{0}$（$A$ 的每一列与 $\mathbf{e}$ 点积为零），即 $A^T(\mathbf{b} - A\hat{\mathbf{x}}) = \mathbf{0}$。
- **第三步，解出 $\hat{\mathbf{x}}$**：$A^TA\hat{\mathbf{x}} = A^T\mathbf{b}$（**正规方程**，下一节主角）。列满秩时 $\hat{\mathbf{x}} = (A^TA)^{-1}A^T\mathbf{b}$。
- **第四步，代回投影**：$\hat{\mathbf{b}} = A\hat{\mathbf{x}} = A(A^TA)^{-1}A^T\mathbf{b}$。**投影矩阵 = $A$ 左乘（$A^TA$ 的逆）右乘 $A^T$**。

<span class="marginnote">正规方程 $A^TA\hat{x} = A^Tb$ 是投影理论的核心方程：<strong>「残差垂直」这一个几何条件，翻成代数就是这条方程</strong>。它贯穿最小二乘、回归、以及第十一篇数据科学的全部实践。</span>

## 4 投影矩阵的性质

投影矩阵 $P$（到子空间 $W$）拥有一组标志性性质：

- **幂等**：$P^2 = P$——投影两次等于投影一次（已经在 $W$ 里的向量再投影不变）；
- **对称**：$P^T = P$；
- **像与核**：$\operatorname{Col}(P) = W$，$\operatorname{Nul}(P) = W^{\perp}$；
- **迹**：$\operatorname{tr} P = \dim W$（$P$ 的特征值只有 1 和 0，1 的个数 = 维数）。

**重点**：**投影 = 分解**：$\mathbf{b} = P\mathbf{b} + (I - P)\mathbf{b}$，其中 $P\mathbf{b} \in W$，$(I-P)\mathbf{b} \in W^{\perp}$。$I - P$ 是到 $W^{\perp}$ 的投影矩阵。这个分解把任意向量「正交拆开」，是正交分解定理（第七篇）的算子形式。

**辨析｜易错点：** 投影矩阵**不要求可逆**（$n > 1$ 时行列式为 0）。「幂等 + 对称」是投影矩阵的**充要特征**：一个矩阵是投影矩阵 ⇔ $P^2 = P$ 且 $P^T = P$。用这条可以快速判断「一个矩阵是不是投影」。

## 5 例子：投影到一条直线

$\mathbf{a} = (1, 2)^T$，$W = \operatorname{span}\{\mathbf{a}\}$。$\mathbf{b} = (3, 1)^T$ 在 $W$ 上的投影：

$$
P = \frac{\mathbf{a}\mathbf{a}^T}{\mathbf{a}^T\mathbf{a}} = \frac{1}{5}\begin{pmatrix} 1 & 2 \\ 2 & 4 \end{pmatrix}
$$

$$
\hat{\mathbf{b}} = P\mathbf{b} = \frac{1}{5}\begin{pmatrix} 1 & 2 \\ 2 & 4 \end{pmatrix}\begin{pmatrix} 3 \\ 1 \end{pmatrix} = \frac{1}{5}\begin{pmatrix} 5 \\ 10 \end{pmatrix} = \begin{pmatrix} 1 \\ 2 \end{pmatrix}
$$

残差 $\mathbf{e} = \mathbf{b} - \hat{\mathbf{b}} = (2, -1)^T$，与 $\mathbf{a} = (1,2)$ 点积为 $2 - 2 = 0$ ✓。投影 $(1,2)$ 恰是「$(3,1)$ 在方向 $(1,2)$ 上的影子」。

**补充｜投影矩阵的「幂等性」是它的灵魂**：$P^2 = P$ 意味着「投影两次等于投影一次」——一旦向量进了 $W$，再投影就纹丝不动。这条性质让投影矩阵成为**idempotent（幂等）算子的原型**：满足 $P^2 = P$ 的线性算子把空间劈成「不动部分」（像）与「压扁部分」（核）。在统计学里，回归的帽子矩阵 $H = X(X^TX)^{-1}X^T$ 正是投影矩阵，$Hy$ 是 $y$ 的拟合值，$(I - H)y$ 是残差——**「帽子矩阵 = 投影矩阵」是统计回归的线性代数本质**。

**补充｜投影矩阵的「幂等性」是它的灵魂**：$P^2 = P$ 意味着「投影两次等于投影一次」——一旦向量进了 $W$，再投影就纹丝不动。这条性质让投影矩阵成为**幂等（idempotent）算子的原型**：满足 $P^2 = P$ 的线性算子把空间劈成「不动部分」（像）与「压扁部分」（核）。在统计学里，回归的帽子矩阵 $H = X(X^TX)^{-1}X^T$ 正是投影矩阵，$Hy$ 是 $y$ 的拟合值，$(I - H)y$ 是残差——**「帽子矩阵 = 投影矩阵」是统计回归的线性代数本质**。

**辨析｜易错点：** 投影矩阵的三个快速检验：

- $P^2 = P$（幂等）——最重要；
- $P^T = P$（对称）——两者都满足才是正交投影；
- $\operatorname{Col}(P) = W$、$\operatorname{Nul}(P) = W^{\perp}$——像与核正交互补。

**若只满足幂等而不对称，那是「斜投影」**（不沿垂直方向投影）——正交投影要求对称性。

**补充｜一句话**：正交投影 = 「把向量拆成『子空间内』与『垂直补内』两部分，取前者」。投影矩阵 $P$ 完成这个拆分，$I - P$ 给出残差部分。

**补充｜一句话**：**「投影 = 残差垂直于子空间的最近点」**，$P = A(A^TA)^{-1}A^T$ 是完成这件事的矩阵。

## 6 小结

- **一维投影**：$P = \frac{\mathbf{a}\mathbf{a}^T}{\mathbf{a}^T\mathbf{a}}$，秩 1，残差垂直。
- **一般投影**：$P = A(A^TA)^{-1}A^T$（$A$ 列满秩），投影到 $\operatorname{Col}(A)$。
- **正规方程**：$A^TA\hat{x} = A^Tb$，由「残差垂直」导出。
- **性质**：$P^2 = P$，$P^T = P$；$\operatorname{Col}(P) = W$，$\operatorname{Nul}(P) = W^{\perp}$；$\operatorname{tr}P = \dim W$。
- **分解**：$\mathbf{b} = P\mathbf{b} + (I-P)\mathbf{b}$，$I - P$ 是到正交补的投影。

在下一节，我们将从正交基的角度重新审视施密特正交化——**正交基下的 Gram-Schmidt 再认识**，并把它接到 QR 分解上。
