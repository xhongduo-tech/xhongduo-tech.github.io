---
title: 正交基下的 Gram-Schmidt 再认识
date: 2026-08-08
---

# 正交基下的 Gram-Schmidt 再认识

<div class="epigraph">
<p>有了投影矩阵的语言，施密特正交化就变成一句话：每个新向量减去它在已建空间上的投影，剩下的部分就是下一个正交方向。</p>
<footer>—— 施密特（Erhard Schmidt）</footer>
</div>

<div class="article-byline">
<p>第二级 · 线性代数 ｜ Strang《Introduction to Linear Algebra》§4.4 ｜ 2026-08-08</p>
</div>

## 为什么从正交基重新看施密特

第五篇我们已经学过施密特正交化的「配方」：减去投影、单位化。现在有了**投影矩阵**的语言，同一个算法有了更深刻的解释：**第 $k$ 步就是「把 $\mathbf{a}_k$ 投影到前面 $k-1$ 个向量张成的空间 $W_{k-1}$ 上，然后减去」**。残差就是新的正交方向。<span class="marginnote">「投影 → 减去」是线性代数最通用的构造模式：<strong>从「有斜的成分」里剔除「已覆盖的成分」，得到「新的独立成分」</strong>。这在 Gram-Schmidt、QR、以及更一般的正交化构造里反复出现——理解了投影，就理解了整套正交化的「为什么」。</span>

本节把 Gram-Schmidt 翻译成投影语言，并建立它与矩阵分解 $A = QR$ 的完整对应。

## 1 从投影重写施密特

设 $\mathbf{a}_1, \cdots, \mathbf{a}_n$ 线性无关，$W_k = \operatorname{span}\{\mathbf{a}_1, \cdots, \mathbf{a}_k\}$，$P_k$ 是到 $W_k$ 的投影矩阵。施密特正交化的每一步可写成：

$$
\mathbf{u}_k = \mathbf{a}_k - P_{k-1}\mathbf{a}_k
$$

**重点**：$P_{k-1}\mathbf{a}_k$ 是 $\mathbf{a}_k$ 在「前面空间」上的投影，减去它，剩下的 $\mathbf{u}_k$ 与 $W_{k-1}$ 正交（残差垂直于被投影的子空间——这是投影的定义性质）。再单位化得 $\mathbf{q}_k = \mathbf{u}_k/\|\mathbf{u}_k\|$。

**拆成四步**：

- **第一步，投影到已建空间**：$P_{k-1}\mathbf{a}_k \in W_{k-1}$，是 $\mathbf{a}_k$ 中「已被前面方向覆盖」的部分。
- **第二步，减掉**：$\mathbf{u}_k = \mathbf{a}_k - P_{k-1}\mathbf{a}_k$，是「未被覆盖」的新成分。
- **第三步，正交性**：由投影的性质，$\mathbf{u}_k \perp W_{k-1}$，即与前面所有 $\mathbf{q}_i$ 垂直。
- **第四步，更新空间**：$W_k = \operatorname{span}\{W_{k-1}, \mathbf{u}_k\} = W_{k-1} \oplus \operatorname{span}\{\mathbf{q}_k\}$，空间逐级扩张。

## 2 正交基下的一切都变简单

一旦拿到标准正交基 $\mathbf{q}_1, \cdots, \mathbf{q}_n$（$\mathbf{q}_i \cdot \mathbf{q}_j = \delta_{ij}$），许多运算变得平凡：

- **坐标**：$\mathbf{x}$ 的第 $j$ 个坐标 $= \mathbf{x}\cdot\mathbf{q}_j$（第五篇）；
- **投影矩阵**：到 $W_k = \operatorname{span}\{\mathbf{q}_1,\cdots,\mathbf{q}_k\}$ 的投影
  $$
  P_k = \mathbf{q}_1\mathbf{q}_1^T + \cdots + \mathbf{q}_k\mathbf{q}_k^T
  $$
  ——**正交基下投影 = 各方向投影之和**，不需要 $A^TA$ 求逆；
- **范数**：$\|\mathbf{x}\|^2 = \sum (\mathbf{x}\cdot\mathbf{q}_i)^2$（Parseval 恒等式）。

**重点**：$P_k = \sum_{i=1}^k \mathbf{q}_i\mathbf{q}_i^T$ 与谱分解 $A = \sum\lambda_i\mathbf{q}_i\mathbf{q}_i^T$（第五篇）是同一个结构——**用秩一投影矩阵分解任何对称矩阵**。正交基让「投影」变成「逐方向求和」。

<span class="marginnote">「正交基下投影 = 各方向投影相加」的直觉：<strong>垂直的方向互不干扰，投影可以分开算再加总</strong>。这就像力在直角坐标系下分解——每个分量独立。傅里叶展开正是这个思想：函数在三角基下的投影系数逐个内积求出。</span>

## 3 Gram-Schmidt 的矩阵形式：$A = QR$

把施密特正交化的全过程写成矩阵：

**定理（QR 分解）**：设 $A$ 是 $m \times n$ 矩阵且列满秩，则 $A$ 可唯一分解为

$$
A = QR
$$

其中 $Q$ 是 $m \times n$ 矩阵（列标准正交），$R$ 是 $n \times n$ **上三角**矩阵且对角元为正。

**重点**：$Q$ 的列就是施密特正交化得到的 $\mathbf{q}_1, \cdots, \mathbf{q}_n$；$R$ 的元素是投影系数 $r_{ij} = \mathbf{q}_i \cdot \mathbf{a}_j$。**$R$ 上三角**是因为 $\mathbf{a}_j$ 只由前 $j$ 个 $\mathbf{q}_i$ 表示（$\operatorname{span}$ 逐级嵌套）。

## 4 公式解析：$r_{ij}$ 的几何含义

$R$ 的上三角元 $r_{ij} = \mathbf{q}_i \cdot \mathbf{a}_j$ 拆成四步理解：

- **第一步，回顾表示**：由 $\mathbf{a}_j = \sum_{i=1}^{j} r_{ij}\mathbf{q}_i$（前 $j$ 个方向），两边与 $\mathbf{q}_i$ 内积：$\mathbf{q}_i \cdot \mathbf{a}_j = r_{ij}$（正交性筛掉其余项）。
- **第二步，$r_{jj}$ 是长度**：$r_{jj} = \mathbf{q}_j\cdot\mathbf{a}_j = \|\mathbf{u}_j\|$——第 $j$ 个正交化残差的长度，也是「$\mathbf{a}_j$ 中未被前面覆盖的分量的大小」。
- **第三步，$r_{ij}$（$i < j$）是投影系数**：$r_{ij}$ = $\mathbf{a}_j$ 在第 $i$ 个方向的投影大小——它记录「$\mathbf{a}_j$ 如何由前面的 $\mathbf{q}_i$ 合成」。
- **第四步，矩阵乘积核对**：$A = QR$ 的第 $j$ 列 $= \sum_i r_{ij}\mathbf{q}_i$，正是 $\mathbf{a}_j$ 的正交分解。**$R$ 是把「原始向量」还原成「正交基组合」的系数表**。

## 5 一个完整的 $QR$ 计算

$A = \begin{pmatrix} 1 & 1 \\ 1 & 0 \\ 0 & 1 \end{pmatrix}$（列：$\mathbf{a}_1 = (1,1,0)^T$，$\mathbf{a}_2 = (1,0,1)^T$）。

- 施密特：$\mathbf{q}_1 = \frac{1}{\sqrt2}(1,1,0)^T$；
- $\mathbf{u}_2 = \mathbf{a}_2 - (\mathbf{q}_1\cdot\mathbf{a}_2)\mathbf{q}_1 = (1,0,1) - \frac{1}{\sqrt2}\cdot\frac{1}{\sqrt2}(1,1,0) = (1,0,1) - \frac12(1,1,0) = (\frac12, -\frac12, 1)$；
- $\|\mathbf{u}_2\| = \sqrt{\frac14 + \frac14 + 1} = \sqrt{\frac32}$，$\mathbf{q}_2 = \frac{1}{\sqrt6}(1, -1, 2)^T$；
- $r_{11} = \|\mathbf{a}_1\| = \sqrt2$，$r_{12} = \mathbf{q}_1\cdot\mathbf{a}_2 = \frac{1}{\sqrt2}$，$r_{22} = \|\mathbf{u}_2\| = \sqrt{\frac32}$；

于是 $R = \begin{pmatrix} \sqrt2 & \frac{1}{\sqrt2} \\ 0 & \sqrt{\frac32} \end{pmatrix}$，$Q = \begin{pmatrix} \frac1{\sqrt2} & \frac1{\sqrt6} \\ \frac1{\sqrt2} & -\frac1{\sqrt6} \\ 0 & \frac2{\sqrt6} \end{pmatrix}$。验证 $QR = A$ ✓。

**补充｜修正施密特（MGS）为何更稳**：经典 Gram-Schmidt 里，后计算的 $\mathbf{q}_j$ 由于浮点误差可能不再与前 $\mathbf{q}_i$ 严格正交，且误差沿链累积。**修正施密特**在每个投影步骤都重新正交化：先把 $\mathbf{a}_k$ 投影到 $\mathbf{q}_1$ 减去，再对结果投影到 $\mathbf{q}_2$ 减去……逐个方向操作。这让每步的舍入误差不被传递到后续方向，数值稳定性显著改善。**「一次减全部投影」vs「逐方向减投影」**——同一数学，更好的数值行为，工程实现普遍用 MGS。

**补充｜修正施密特（MGS）为何更稳**：经典 Gram-Schmidt 里，后计算的 $\mathbf{q}_j$ 由于浮点误差可能不再与前 $\mathbf{q}_i$ 严格正交，且误差沿链累积。**修正施密特**在每个投影步骤都重新正交化：先把 $\mathbf{a}_k$ 投影到 $\mathbf{q}_1$ 减去，再对结果投影到 $\mathbf{q}_2$ 减去……逐个方向操作。这让每步的舍入误差不被传递到后续方向，数值稳定性显著改善。**「一次减全部投影」vs「逐方向减投影」**——同一数学，更好的数值行为，工程实现普遍用 MGS。

**辨析｜易错点：** 正交基与 QR 的三个易混点：

- $Q$ 是 $m \times n$（与 $A$ 同列数）且 $Q^TQ = I$；若 $m > n$，$Q$ 不是方阵，「正交矩阵」只对 $m = n$ 时称呼；
- $R$ 对角元**必须为正**才唯一（否则 $\mathbf{q}_j$ 可整体换号）；
- 施密特顺序决定 $R$ 上三角：$\mathbf{a}_j$ 只用前 $j$ 个 $\mathbf{q}_i$ 表示。

**「$Q$ 列正交、$R$ 上三角、对角为正」**三条件定 QR。

**补充｜正交基的「红利清单」**：

- 坐标 = 内积（$x_j = \mathbf{q}_j \cdot \mathbf{x}$）；
- 投影 = 各方向投影之和（$P = \sum\mathbf{q}_i\mathbf{q}_i^T$）；
- 范数 = 系数平方和（Parseval）；
- 矩阵逆 = 转置（正交矩阵）。

**「选对基，一切变内积」**——正交基让线性代数退化为「点积运算」。

**补充｜一句话**：**「正交基下坐标 = 内积、投影 = 求和、范数 = 平方和」**——正交性让线性代数变成点积算术。

## 6 小结

- **投影视角**：$\mathbf{u}_k = \mathbf{a}_k - P_{k-1}\mathbf{a}_k$，减去投影即得新正交方向。
- **正交基红利**：坐标 = 内积，投影 = 各方向投影之和，范数 = 系数平方和。
- **QR 分解**：$A = QR$，$Q$ 列标准正交，$R$ 上三角对角元为正。
- **$R$ 的含义**：$r_{ij} = \mathbf{q}_i\cdot\mathbf{a}_j$，对角元是残差长度、非对角元是投影系数。
- **应用预告**：QR 是解最小二乘的稳定算法（下节）。

在下一节，我们将把 QR 分解作为工具正式使用——**QR 分解及其应用**，看它如何稳定地解最小二乘、算特征值。
