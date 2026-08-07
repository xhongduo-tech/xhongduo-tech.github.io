---
title: 实对称矩阵的对角化
date: 2026-08-08
---

# 实对称矩阵的对角化

<div class="epigraph">
<p>实对称矩阵是矩阵世界里的完美公民：特征值全是实数，特征向量还能选成正交的——它总能被对角化成最干净的形式。</p>
<footer>—— 凯莱（Arthur Cayley）</footer>
</div>

<div class="article-byline">
<p>第二级 · 线性代数 ｜ 同济《线性代数》§5.4 ｜ 2026-08-08</p>
</div>

## 为什么从实对称矩阵开始

上一节的可对角化判据要求「$n$ 个线性无关特征向量」，对一般矩阵可能失败。但有一类矩阵**永远**可对角化，而且还能用**正交矩阵**来对角化——这就是**实对称矩阵**（$A^T = A$）。<span class="marginnote">实对称矩阵是「大多数应用的主角」：协方差矩阵（第十一篇）、二次型的矩阵表示（下节）、图论中的邻接矩阵、物理中的惯性张量与应力张量——它们天然对称，于是都能谱分解。$A = Q\Lambda Q^T$ 叫<strong>谱分解（spectral decomposition）</strong>，它把矩阵按特征值「谱」展开，是 PCA、SVD 的数学基础。</span>

这一节给出实对称矩阵的两大定理：特征值为实数、可正交对角化——并把它与二次型衔接。

## 1 实对称矩阵的特征值是实数

**定理**：实对称矩阵 $A$（$A^T = A$）的特征值全部是实数，且对应特征向量可取实向量。

**重点**：这是实对称矩阵与一般矩阵的分水岭。一般实矩阵可能只有复特征值（旋转矩阵 $R(90°)$ 的特征值是 $\pm i$），但对称矩阵的特征多项式总在实数范围内**全部解出**。

**证明要点（内积技巧）**：设 $A\mathbf{x} = \lambda\mathbf{x}$，$\mathbf{x} \ne \mathbf{0}$（允许复分量）。取共轭转置与内积：

$$
\overline{\mathbf{x}}^T A \mathbf{x} = \lambda \|\mathbf{x}\|^2
$$

左边取共轭转置再利用 $A^T = A$ 与 $A$ 实，得 $\overline{\lambda}\|\mathbf{x}\|^2$，于是 $\lambda = \overline{\lambda}$，即 $\lambda$ 为实数。

<span class="marginnote">内积技巧是实对称矩阵理论的引擎：<strong>凡是「$A$ 对称 + 内积」的组合，都能用 $\mathbf{y}^T A\mathbf{x} = (A\mathbf{y})^T\mathbf{x}$ 来回搬运</strong>，把特征值问题变成纯内积问题。这个技巧在谱定理、Courant-Fischer 变分刻画、SVD 中反复出现。</span>

## 2 不同特征值的特征向量正交

**定理**：实对称矩阵 $A$ 属于**不同特征值**的特征向量必**正交**。

**证明要点**：设 $A\mathbf{x} = \lambda\mathbf{x}$、$A\mathbf{y} = \mu\mathbf{y}$，$\lambda \ne \mu$。计算

$$
\lambda(\mathbf{x}\cdot\mathbf{y}) = (A\mathbf{x})\cdot\mathbf{y} = \mathbf{x}\cdot(A^T\mathbf{y}) = \mathbf{x}\cdot(A\mathbf{y}) = \mu(\mathbf{x}\cdot\mathbf{y})
$$

故 $(\lambda - \mu)(\mathbf{x}\cdot\mathbf{y}) = 0$，因 $\lambda \ne \mu$，得 $\mathbf{x}\cdot\mathbf{y} = 0$。**对称性让「特征向量正交」从「无关」升级为「垂直」**。

**重点**：普通矩阵只说「不同特征值的特征向量无关」，对称矩阵更进一层——它们**正交**。这决定了对称矩阵不仅能对角化，还能用正交矩阵对角化。

## 3 正交对角化定理

**定理（实对称矩阵的正交对角化 / 谱定理）**：设 $A$ 是 $n$ 阶实对称矩阵，则存在**正交矩阵 $Q$** 和对角矩阵 $\Lambda$，使

$$
Q^T A Q = \Lambda, \qquad \text{即} \qquad A = Q\Lambda Q^T
$$

其中 $Q$ 的列是 $A$ 的单位特征向量，$\Lambda$ 的对角元是相应特征值。

**重点**：与一般对角化 $P^{-1}AP = \Lambda$ 相比，这里的过渡矩阵是**正交矩阵**，于是 $P^{-1}$ 换成了便宜的 $Q^T$。**正交对角化 = 用标准正交基的对角化**。

**辨析｜易错点：** 即使特征值有重数（如 $\lambda$ 重数 2），几何重数也恰为 2——对称矩阵**总能**凑齐足够的正交特征向量（同一特征值的特征空间内部做施密特正交化即可）。所以实对称矩阵**永远**可正交对角化，没有例外。这是它与一般矩阵最本质的差别。

## 4 公式解析：$A = Q\Lambda Q^T$ 的展开

谱分解公式 $A = Q\Lambda Q^T$ 可以写成**秩一展开**：

$$
A = \lambda_1 \mathbf{q}_1\mathbf{q}_1^T + \lambda_2 \mathbf{q}_2\mathbf{q}_2^T + \cdots + \lambda_n \mathbf{q}_n\mathbf{q}_n^T
$$

拆成四步：

- **第一步，列视角**：$Q = (\mathbf{q}_1, \cdots, \mathbf{q}_n)$，$\Lambda = \operatorname{diag}(\lambda_1, \cdots, \lambda_n)$，代入 $Q\Lambda Q^T$。
- **第二步，乘积展开**：$Q\Lambda Q^T = \sum_{j} \lambda_j \mathbf{q}_j \mathbf{q}_j^T$——每个特征值乘一个「外积」$\mathbf{q}_j\mathbf{q}_j^T$。
- **第三步，外积是什么**：$\mathbf{q}_j\mathbf{q}_j^T$ 是**投影矩阵**（投影到 $\mathbf{q}_j$ 方向的秩一矩阵），作用在任意 $\mathbf{x}$ 上给出 $(\mathbf{q}_j\cdot\mathbf{x})\mathbf{q}_j$。
- **第四步，谱分解的含义**：$A\mathbf{x} = \sum_j \lambda_j (\mathbf{q}_j\cdot\mathbf{x})\mathbf{q}_j$——**把 $\mathbf{x}$ 按正交基展开，每个方向独立乘以其特征值再拼回**。这就是「按谱分解」：矩阵被拆成「方向 × 强度」的和。

<span class="marginnote">谱分解是现代数据分析的母公式：<strong>协方差矩阵的谱分解给出 PCA</strong>（最大特征值方向 = 最大方差方向）；截断谱分解（保留前 $k$ 项）就是最佳低秩近似，与第十篇截断 SVD 同构。$A \approx \sum_{j=1}^k \lambda_j \mathbf{q}_j\mathbf{q}_j^T$ 用 $k$ 项近似全矩阵。</span>

## 5 例子：正交对角化实操

正交对角化 $A = \begin{pmatrix} 3 & 1 \\ 1 & 3 \end{pmatrix}$（与上节同一矩阵，但它是对称的）。

- 特征值 $\lambda = 2, 4$，特征向量 $(1,-1)^T$、$(1,1)^T$。
- 单位化：$\mathbf{q}_1 = \frac{1}{\sqrt2}(1,-1)^T$，$\mathbf{q}_2 = \frac{1}{\sqrt2}(1,1)^T$。
- $Q = \frac{1}{\sqrt2}\begin{pmatrix} 1 & 1 \\ -1 & 1 \end{pmatrix}$，验证 $Q$ 正交（$Q^TQ = I$）且 $Q^T A Q = \operatorname{diag}(2, 4)$。

**辨析｜易错点：** 正交对角化时，特征向量必须先单位化；重特征值的特征空间内要用施密特正交化保证组内正交。写 $Q$ 时**列的顺序要与 $\Lambda$ 对角元的顺序一致**——第 $j$ 列是 $\lambda_j$ 的特征向量。顺序错位会导致 $Q^TAQ$ 不是对角阵。

**补充｜为什么对称矩阵「这么有用」**：几乎所有「自然生成」的矩阵都是对称的——协方差矩阵、Hessian 矩阵、Gram 矩阵、距离矩阵、图的邻接矩阵（无向）。原因是它们都来自「内积结构」：$A = B^TB$ 或 $A_{ij} = \langle v_i, v_j \rangle$ 自动对称半正定。**对称性 = 结构来自内积**，而内积无处不在，所以谱定理的应用无处不在。反观非对称矩阵（如转移矩阵）则没有谱定理保障，只能求 Jordan 形或 SVD——这正是对称矩阵「特殊待遇」的来源。

**补充｜为什么对称矩阵「这么有用」**：几乎所有「自然生成」的矩阵都是对称的——协方差矩阵、Hessian 矩阵、Gram 矩阵、距离矩阵、图的邻接矩阵（无向）。原因是它们都来自「内积结构」：$A = B^TB$ 或 $A_{ij} = \langle v_i, v_j \rangle$ 自动对称半正定。**对称性 = 结构来自内积**，而内积无处不在，所以谱定理的应用无处不在。反观非对称矩阵（如转移矩阵）则没有谱定理保障，只能求 Jordan 形或 SVD——这正是对称矩阵「特殊待遇」的来源。

**辨析｜易错点：** 正交对角化的实操检查清单：

- 确认 $A^T = A$（先验对称，否则不适用谱定理）；
- 特征值全实，重特征值要凑齐正交特征向量（组内施密特）；
- 特征向量**必须单位化**再拼 $Q$，否则 $Q$ 不正交；
- $Q$ 的列顺序与 $\Lambda$ 对角元顺序一致。

**「先验对称、再单位化、后对顺序」**三步不出错。

**补充｜实对称矩阵与其他矩阵的对照**：

- **实对称**：特征值全实、可正交对角化、谱分解存在；
- **一般实矩阵**：特征值可能复、可能不可对角化（Jordan 形）；
- **Hermite（复对称）**：$A^* = A$，特征值全实、可酉对角化。

**「对称 = 谱定理的通行证」**——遇到对称矩阵，先想谱分解；遇到一般矩阵，先想 SVD。

## 6 小结

- **特征值实数**：实对称矩阵特征值全为实数。
- **特征向量正交**：不同特征值的特征向量正交。
- **正交对角化**：$A = Q\Lambda Q^T$，$Q$ 正交，任何实对称矩阵都行。
- **谱分解**：$A = \sum \lambda_j \mathbf{q}_j\mathbf{q}_j^T$，按方向 × 强度展开。
- **应用**：协方差矩阵 PCA、二次型标准化、低秩近似。

在下一节，我们将从对称矩阵直接通向「多元二次函数」——**二次型及其矩阵表示**，把 $x^T A x$ 这样的表达式与对称矩阵一一对应。
