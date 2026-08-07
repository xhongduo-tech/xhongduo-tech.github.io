---
title: SVD 与四个基本子空间
date: 2026-08-08
---

# SVD 与四个基本子空间

<div class="epigraph">
<p>SVD 是四个基本子空间的完美坐标：右奇异向量张成行空间与零空间，左奇异向量张成列空间与左零空间——一张分解，四个空间各得其所。</p>
<footer>—— 斯特朗（Gilbert Strang，《Introduction to Linear Algebra》）</footer>
</div>

<div class="article-byline">
<p>第二级 · 线性代数 ｜ Strang《Introduction to Linear Algebra》§7.1 ｜ 2026-08-08</p>
</div>

## 为什么从 SVD 与四子空间开始

第七篇我们建立了四个基本子空间，但它们的「坐标」是零散的。SVD 提供了一组**统一的标准正交基**，让四个子空间各有清晰基座：$V$ 的前 $r$ 列张成行空间、后 $n-r$ 列张成零空间；$U$ 的前 $r$ 列张成列空间、后 $m-r$ 列张成左零空间。<span class="marginnote">SVD 对四子空间的贡献是「一劳永逸」：<strong>不再需要分别消元、分别求基，一次 SVD 同时给出四个子空间的标准正交基</strong>。这在数据科学里尤其珍贵——子空间的几何结构随 SVD 全部暴露，PCA 直接读奇异向量即可。</span>

本节建立 SVD 与四子空间的精确对应。

## 1 分解的秩一形式

设 $A$ 是 $m \times n$ 矩阵，$\operatorname{rank} A = r$，SVD 为 $A = U\Sigma V^T$。写成**秩一展开**：

$$
A = \sigma_1 \mathbf{u}_1\mathbf{v}_1^T + \sigma_2 \mathbf{u}_2\mathbf{v}_2^T + \cdots + \sigma_r \mathbf{u}_r\mathbf{v}_r^T
$$

每一项 $\sigma_i \mathbf{u}_i\mathbf{v}_i^T$ 是秩 1 矩阵（外积），$r$ 项加起来恢复 $A$。

**重点**：**SVD 把矩阵写成「$r$ 个秩一层的叠加」**——每层由一个奇异值、一对奇异向量定义。这是「矩阵的信息按重要程度（奇异值大小）分层」的数学形式，也是截断低秩近似的直接入口（本节末预告）。

## 2 四子空间的 SVD 坐标

SVD 给出四子空间的**标准正交基**：

| 子空间 | 标准正交基 | 维数 |
| --- | --- | --- |
| $\operatorname{Col}(A)$ | $\mathbf{u}_1, \cdots, \mathbf{u}_r$ | $r$ |
| $\operatorname{Nul}(A^T)$（左零空间） | $\mathbf{u}_{r+1}, \cdots, \mathbf{u}_m$ | $m - r$ |
| $\operatorname{Row}(A)$ | $\mathbf{v}_1, \cdots, \mathbf{v}_r$ | $r$ |
| $\operatorname{Nul}(A)$（零空间） | $\mathbf{v}_{r+1}, \cdots, \mathbf{v}_n$ | $n - r$ |

**重点**：这张表是四子空间理论的「完工图」——**$U$ 的列瓜分 $\mathbb{R}^m$（列空间 + 左零空间），$V$ 的列瓜分 $\mathbb{R}^n$（行空间 + 零空间）**，且每对互补子空间的基自动正交。

**为什么**：$A\mathbf{v}_i = \sigma_i\mathbf{u}_i$（$i \le r$）说明 $\mathbf{u}_i \in \operatorname{Col}(A)$；$A\mathbf{v}_i = \mathbf{0}$（$i > r$，$\sigma_i = 0$）说明 $\mathbf{v}_i \in \operatorname{Nul}(A)$。由基的正交性与维数对账，其余子空间一一定位。

## 3 公式解析：$A = \sum \sigma_i \mathbf{u}_i\mathbf{v}_i^T$ 与子空间

把秩一展开与四子空间的关系拆开：

- **第一步，秩一层的结构**：$\sigma_i\mathbf{u}_i\mathbf{v}_i^T$ 的列空间是 $\operatorname{span}\{\mathbf{u}_i\}$，零空间是 $\operatorname{span}\{\mathbf{v}_i\}^{\perp} = \operatorname{span}\{\mathbf{v}_j : j \ne i\}$。
- **第二步，加和的空间**：前 $r$ 层的列空间并起来是 $\operatorname{span}\{\mathbf{u}_1,\cdots,\mathbf{u}_r\} = \operatorname{Col}(A)$；行空间同理为 $\operatorname{span}\{\mathbf{v}_1,\cdots,\mathbf{v}_r\}$。
- **第三步，零奇异值方向**：$\sigma_{r+1} = \cdots = 0$ 对应的 $\mathbf{v}_i$ 被 $A$ 压扁——正是零空间的基；$\mathbf{u}_i$（$i > r$）不被任何层覆盖——正是左零空间的基。
- **第四步，正交分解的天然性**：奇异向量本身标准正交，所以四个子空间的基自动正交——**SVD 把「正交补」关系直接写进分解**。

<span class="marginnote"><strong>奇异值大小 = 该方向的信息强度</strong>：前几个奇异值大，对应的秩一层携带矩阵的主要信息；后面奇异值趋近零，对应层只是噪声/冗余。这个「排序」让 SVD 天然适合做「主成分」——<strong>保留前 $k$ 个最大奇异值的层，就是最佳 $k$ 秩近似</strong>（截断 SVD，本节末与下节）。</span>

## 4 例子：SVD 读出四子空间

$A = \begin{pmatrix} 3 & 0 \\ 0 & 0 \end{pmatrix}$（秩 1）。SVD：$\sigma_1 = 3$，$\mathbf{u}_1 = (1,0)^T$，$\mathbf{v}_1 = (1,0)^T$；$\sigma_2 = 0$，$\mathbf{u}_2 = (0,1)^T$，$\mathbf{v}_2 = (0,1)^T$。

- 列空间：$\operatorname{span}\{(1,0)^T\}$（$\mathbb{R}^2$ 的 x 轴）；
- 左零空间：$\operatorname{span}\{(0,1)^T\}$（y 轴），与列空间正交；
- 行空间：$\operatorname{span}\{(1,0)^T\}$（x 轴）；
- 零空间：$\operatorname{span}\{(0,1)^T\}$（y 轴），与行空间正交。

四子空间通过 $U, V$ 的列一眼读出，且正交补关系（列 ⊥ 左零、行 ⊥ 零）自动满足。

## 5 SVD 与秩、范数、条件数

奇异值集齐了矩阵的「度量身份证」：

$$
\operatorname{rank} A = \#\{\sigma_i > 0\}, \qquad \|A\|_2 = \sigma_{\max}, \qquad \kappa_2(A) = \frac{\sigma_{\max}}{\sigma_{\min}}
$$

- **秩** = 非零奇异值个数（比消元更「几何」的秩定义）；
- **2-范数** = 最大奇异值（单位球被拉长的最大幅度）；
- **条件数** = 最大/最小奇异值之比（椭球扁的程度）；
- **Frobenius 范数**：$\|A\|_F = \sqrt{\sigma_1^2 + \cdots + \sigma_r^2}$（奇异值平方和的根）。

**重点**：**奇异值是矩阵「所有度量」的统一来源**。从 SVD 出发，秩、范数、条件数、低秩近似全部「免费获得」。这是 SVD 被称为「最强分解」的理由。

**补充｜SVD 与四子空间：一张「分解图」**：把 $A = U\Sigma V^T$ 画成方块图，$U$ 的两块（前 $r$ 列与后 $m-r$ 列）分别张成列空间与左零空间，$V$ 的两块张成行空间与零空间；$\Sigma$ 的非零块只连接「行空间 → 列空间」。**四个子空间通过 $\Sigma$ 的「非零块」两两连通，零块则被压扁**——SVD 同时是四子空间的「电路图」与「压扁示意图」。这张图是理解伪逆（$V\Sigma^+U^T$ 反着走）与最小二乘的直观罗盘。

**辨析｜易错点：** 零奇异值对应的 $\mathbf{v}_i$ 在零空间、$\mathbf{u}_i$ 在左零空间——**它们「成对出现在两端」**，但各自属于不同的空间（输入 vs 输出）。取 $i > r$ 时 $A\mathbf{v}_i = 0$（输入被压扁），同时 $\mathbf{u}_i$ 不在列空间（输出方向未被覆盖）。**「一个零奇异值同时标记输入的一个死方向与输出的一个盲方向」**——这是秩亏的几何签名。

**补充｜SVD 与四子空间：一张「分解图」**：把 $A = U\Sigma V^T$ 画成方块图，$U$ 的两块（前 $r$ 列与后 $m-r$ 列）分别张成列空间与左零空间，$V$ 的两块张成行空间与零空间；$\Sigma$ 的非零块只连接「行空间 → 列空间」。**四个子空间通过 $\Sigma$ 的「非零块」两两连通，零块则被压扁**——SVD 同时是四子空间的「电路图」与「压扁示意图」。这张图是理解伪逆（$V\Sigma^+U^T$ 反着走）与最小二乘的直观罗盘。

**辨析｜易错点：** 零奇异值对应的 $\mathbf{v}_i$ 在零空间、$\mathbf{u}_i$ 在左零空间——**它们「成对出现在两端」**，但各自属于不同的空间（输入 vs 输出）。取 $i > r$ 时 $A\mathbf{v}_i = 0$（输入被压扁），同时 $\mathbf{u}_i$ 不在列空间（输出方向未被覆盖）。**「一个零奇异值同时标记输入的一个死方向与输出的一个盲方向」**——这是秩亏的几何签名。

**补充｜用 SVD 求四子空间基的标准流程**：

- 对 $A$ 做 SVD，取非零奇异值个数 $r$；
- $U$ 前 $r$ 列 = 列空间基，$U$ 后 $m-r$ 列 = 左零空间基；
- $V$ 前 $r$ 列 = 行空间基，$V$ 后 $n-r$ 列 = 零空间基。

**「一次 SVD，四组基全得」**是 SVD 相对消元法的最大便利。

**补充｜SVD 与四子空间的「一句话」**：**「$U$ 的列张成输出空间、$V$ 的列张成输入空间，奇异值非零块连接行空间与列空间」**——一次 SVD，四个基本子空间及其正交补关系全部就位。

**补充｜SVD 与四子空间的速查表**：

- 列空间基 = $U$ 前 $r$ 列（$r$ = 非零奇异值个数）；
- 左零空间基 = $U$ 后 $m-r$ 列；
- 行空间基 = $V$ 前 $r$ 列；
- 零空间基 = $V$ 后 $n-r$ 列。

**「$U$ 管输出、$V$ 管输入、$r$ 定维数」**，四组基一次 SVD 全出。

## 6 小结

- **秩一展开**：$A = \sum_{i=1}^r \sigma_i\mathbf{u}_i\mathbf{v}_i^T$，每层秩 1。
- **四子空间基**：$U$ 前 $r$ 列/后 $m-r$ 列 = 列空间/左零空间；$V$ 前 $r$ 列/后 $n-r$ 列 = 行空间/零空间。
- **正交自动**：奇异向量标准正交，正交补关系写进分解。
- **度量身份证**：秩、$\|A\|_2$、$\kappa$、$\|A\|_F$ 全部由奇异值给出。
- **预告**：保留前 $k$ 个最大奇异值 = 截断 SVD = 最佳低秩近似。

在下一节，我们将把 SVD 变成「求解工具」——**伪逆（Moore-Penrose 逆）及其性质**，让奇异矩阵也能「求逆」。
