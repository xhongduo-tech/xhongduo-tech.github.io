---
title: Jordan 标准形简介
date: 2026-08-08
---

# Jordan 标准形简介

<div class="epigraph">
<p>特征向量不够用时，矩阵化不成对角形；但它仍能化成一种「几乎对角」的准标准形——Jordan 标准形，对角元就是特征值，超对角可能多几个 1。</p>
<footer>—— 若尔当（Camille Jordan）</footer>
</div>

<div class="article-byline">
<p>第二级 · 线性代数 ｜ 同济《线性代数》§5.4 + Strang §6.5 ｜ 2026-08-08</p>
</div>

## 为什么从 Jordan 标准形开始

第五篇的可对角化判据很严格：特征向量不够 $n$ 个就失败。但「失败」不等于「无解」——**Jordan 标准形**是「最接近对角」的退路：任何方阵都能化成一组 Jordan 块，块内对角元是特征值，超对角是 0 或 1。<span class="marginnote">Jordan 标准形回答了「相似的最简形是什么」：<strong>对角形只在特征向量够时达到，Jordan 形对所有方阵都成立</strong>。它是相似不变量理论（第九篇主题）、微分方程组的解（广义特征向量）、以及系统理论（模态分析）的数学基础。</span>

本节给出 Jordan 块与 Jordan 形、广义特征向量，以及它的存在性定理（重点在理解而非证明）。

## 1 Jordan 块与 Jordan 标准形

**核心概念**：形如

$$
J_k(\lambda) = \begin{pmatrix}
\lambda & 1 & & \\
& \lambda & \ddots & \\
& & \ddots & 1 \\
& & & \lambda
\end{pmatrix}
$$

的 $k \times k$ 矩阵称为一个 **Jordan 块（Jordan block）**：对角元全是 $\lambda$，超对角（对角线上方一格）全是 1，其余为 0。

**Jordan 标准形（Jordan normal form）**：由若干 Jordan 块组成的分块对角矩阵：

$$
J = \begin{pmatrix}
J_{k_1}(\lambda_1) & & \\
& \ddots & \\
& & J_{k_r}(\lambda_r)
\end{pmatrix}
$$

**定理（Jordan 分解）**：任意复方阵 $A$ 都相似于它的 Jordan 标准形 $J$：存在可逆矩阵 $P$ 使

$$
P^{-1}AP = J
$$

（不计块的排列顺序，$J$ 由 $A$ 唯一确定。）

**重点**：**对角化是 Jordan 形的特例**——所有 Jordan 块都是 $1 \times 1$（超对角没有 1）时，$J$ 就是对角矩阵。特征向量不足时，块变大，超对角冒出 1。

**辨析｜易错点：** Jordan 形里的 $\lambda$ 是**特征值**，但同一个特征值可能出现**多个** Jordan 块。块的大小与个数由「广义特征向量」的链长决定，不是简单地由代数/几何重数推出（虽然相关）。**Jordan 形是唯一的最简相似形**——它精确区分「有相同特征值但不同结构」的矩阵。

## 2 广义特征向量

**核心概念**：若 $(A - \lambda I)^k \mathbf{v} = \mathbf{0}$ 对某个正整数 $k$ 成立，但 $(A - \lambda I)^{k-1}\mathbf{v} \ne \mathbf{0}$，则称 $\mathbf{v}$ 为 $A$ 的**广义特征向量（generalized eigenvector）**，$k$ 称为它的**阶数**。

普通特征向量满足 $(A - \lambda I)\mathbf{v} = \mathbf{0}$（阶数 1）。阶数 $k$ 的广义特征向量沿链

$$
\mathbf{v} \mapsto (A - \lambda I)\mathbf{v} \mapsto (A - \lambda I)^2\mathbf{v} \mapsto \cdots \mapsto \mathbf{0}
$$

逐步「下降」到零向量——这条链恰好铺成一个 Jordan 块的列。

**重点**：**广义特征向量补足了「缺失」的特征向量**。若 $A$ 不可对角化，是因为每个特征值对应的（普通）特征向量不够多；广义特征向量总能凑齐 $n$ 个，撑起整个空间的基。

<span class="marginnote">直觉：<strong>普通特征向量是「不动方向」，广义特征向量是「被反复压扁后归零的方向」</strong>。链 $\mathbf{v}, (A-\lambda I)\mathbf{v}, \ldots$ 上的每一步都是「往零空间走一级」——整条链的长度就是 Jordan 块的大小。微分方程组 $\dot{x} = Ax$ 的解里，广义特征向量给出 $t^k e^{\lambda t}$ 项（模态分析）。</span>

## 3 公式解析：为什么超对角有 1

为什么 Jordan 块的对角线上方是 1 而不是别的数？拆成四步：

- **第一步，从一个特征向量出发**：设 $\mathbf{v}_1$ 是特征值 $\lambda$ 的一个特征向量。若几何重数 < 代数重数，$\mathbf{v}_1$ 不够。
- **第二步，反向找链**：寻找 $\mathbf{v}_2$ 使 $(A - \lambda I)\mathbf{v}_2 = \mathbf{v}_1$——$\mathbf{v}_2$ 是「下一级广义特征向量」。这个方程可解的条件正是「几何重数不足」。
- **第三步，矩阵在这条链上的作用**：$A\mathbf{v}_1 = \lambda\mathbf{v}_1$，$A\mathbf{v}_2 = \mathbf{v}_1 + \lambda\mathbf{v}_2$。用 $\{\mathbf{v}_1, \mathbf{v}_2\}$ 做基，$A$ 在该基下表示为 $\begin{pmatrix} \lambda & 1 \\ 0 & \lambda \end{pmatrix}$——**超对角那个 1 就是「$A\mathbf{v}_2$ 里 $\mathbf{v}_1$ 的系数」**。
- **第四步，块的长度**：链越长，块越大；每条链对应一个 Jordan 块。**1 的位置是链结构的签名**——它表示「上一级广义特征向量」的出现。

## 4 例子：不可对角化矩阵的 Jordan 形

$A = \begin{pmatrix} 1 & 1 \\ 0 & 1 \end{pmatrix}$。特征值 $\lambda = 1$（代数重数 2），特征向量只有一个（$(1,0)^T$），几何重数 1——不可对角化。它的 Jordan 形：

$$
J = \begin{pmatrix} 1 & 1 \\ 0 & 1 \end{pmatrix} = A
$$

（$A$ 本身已是 Jordan 块 $J_2(1)$。）广义特征向量链：$(A - I)\mathbf{v}_2 = \mathbf{v}_1$，取 $\mathbf{v}_1 = (1,0)^T$，$\mathbf{v}_2 = (0,1)^T$，链长 2。

对照可对角化的 $B = \begin{pmatrix} 1 & 0 \\ 0 & 1 \end{pmatrix}$：**$A$ 与 $B$ 特征值相同但不相邻相似**——Jordan 形用「块的大小（超对角 1）」区分了它们。这就是为什么「特征值相同」不足以判定相似（第五篇）。

**辨析｜易错点：** Jordan 形计算（求 $P$）相当繁琐，教科书通常只要求「写标准形」而非「求过渡矩阵」。考试里给出特征值的代数/几何重数，能写出 Jordan 块的排布即可。**重点在「块大小 = 广义特征向量链长」这一对应**。

## 5 Jordan 形的应用

- **矩阵幂与指数**：$A = PJP^{-1}$，则 $A^k = PJ^kP^{-1}$、$e^{At} = Pe^{Jt}P^{-1}$。Jordan 块的幂（指数）可用二项式展开计算：$(J_k(\lambda))^k$ 的超对角项给出 $k\lambda^{k-1}$ 等组合系数。
- **微分方程组**：$\dot{\mathbf{x}} = A\mathbf{x}$ 的解由 $e^{At}$ 给出；Jordan 块贡献 $t^m e^{\lambda t}$ 型项——**系统稳定性由特征值实部决定，模态形状由广义特征向量决定**。
- **系统理论**：能控性/能观性分析、Jordan 形揭示系统的「内模结构」。
- **相似不变量**：Jordan 块的大小与个数是**最细的相似不变量**——两个矩阵相似 ⟺ 它们有相同的 Jordan 形（块排列除外）。

**重点**：Jordan 形把「相似分类」彻底做完：**相似 ⟺ 同一个 Jordan 形**。这是「特征值相同但不相邻相似」这类问题的终极答案。

**补充｜Jordan 形与矩阵函数的计算**：对 Jordan 块 $J_k(\lambda)$，其幂与指数有明确的超对角「爬升」结构：$J_k(\lambda)^t$ 的第 $i$ 超对角为 $\binom{t}{i}\lambda^{t-i}$。因此 $e^{Jt} = e^{\lambda t}\sum_{i\ge0}\frac{t^i}{i!}N^i$（$N$ 是超对角 1 的幂零矩阵）。**Jordan 块把「矩阵函数」化成「幂零部分的多项式」**——这是微分方程组解中 $t^m e^{\lambda t}$ 项的来源，也是系统理论里「模态」概念的代数根基。

**补充｜Jordan 形与矩阵函数的计算**：对 Jordan 块 $J_k(\lambda)$，其幂与指数有明确的超对角「爬升」结构：$J_k(\lambda)^t$ 的第 $i$ 超对角为 $\binom{t}{i}\lambda^{t-i}$。因此 $e^{Jt} = e^{\lambda t}\sum_{i\ge0}\frac{t^i}{i!}N^i$（$N$ 是超对角 1 的幂零矩阵）。**Jordan 块把「矩阵函数」化成「幂零部分的多项式」**——这是微分方程组解中 $t^m e^{\lambda t}$ 项的来源，也是系统理论里「模态」概念的代数根基。

**辨析｜易错点：** Jordan 形的判定易错点：

- 一个特征值可对应**多个** Jordan 块——几何重数 = 该特征值的块数；
- 块的总大小 = 代数重数；**代数重数 ≥ 几何重数恒成立**；
- 可对角化 ⇔ 所有块都是 $1 \times 1$ ⇔ 每个特征值代数重数 = 几何重数。

**「块数 = 几何重数，块总大小 = 代数重数」**是写 Jordan 形的两把尺子。

## 6 小结

- **Jordan 块**：对角元 $\lambda$、超对角 1 的块矩阵；**Jordan 形** = 块的分块对角。
- **定理**：任意方阵相似于 Jordan 形；对角形是块全为 $1\times1$ 的特例。
- **广义特征向量**：$(A-\lambda I)^k\mathbf{v} = \mathbf{0}$；链长 = Jordan 块大小。
- **超对角 1**：链结构的签名，区分特征值相同但结构不同的矩阵。
- **应用**：$e^{At}$、微分方程、系统理论、相似不变量。

在下一节，我们将量化「矩阵有多病态」——**矩阵范数与条件数**，理解数值误差为何放大、如何衡量解的可靠性。
