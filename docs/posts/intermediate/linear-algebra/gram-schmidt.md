---
title: 施密特（Gram-Schmidt）正交化
date: 2026-08-08
---

# 施密特（Gram-Schmidt）正交化

<div class="epigraph">
<p>给一组斜的基，施密特把它一步步「扶正」：每步减去往前的投影，剩下的部分自然与前面垂直。</p>
<footer>—— 施密特（Erhard Schmidt，正交化方法的系统化者）</footer>
</div>

<div class="article-byline">
<p>第二级 · 线性代数 ｜ 同济《线性代数》§5.2 ｜ 2026-08-08</p>
</div>

## 为什么从施密特正交化开始

上一节我们见识了标准正交基的「奢侈」：坐标是内积、矩阵逆是转置。但自然给出的基几乎从不正交。**施密特正交化（Gram-Schmidt process）**解决这个问题：给任意一组线性无关向量，它逐步骤地造出一组**张成同一空间**的标准正交基。<span class="marginnote">施密特正交化的几何动机极其直观：第 $j$ 步的候选向量减去它在「前面所有方向张成的平面」上的投影，剩下的分量就与前 $n$ 个方向全部垂直——「减去投影 = 剩下的部分正交」。这个「减法」思想也是最小二乘（第八篇）的核心：残差 = 原量 − 投影。</span>

这一节既给出算法，也埋下第八篇投影矩阵与 QR 分解的伏笔。

## 1 算法：三步一个向量

设 $\mathbf{a}_1, \mathbf{a}_2, \cdots, \mathbf{a}_n$ 线性无关，要造标准正交组 $\mathbf{q}_1, \cdots, \mathbf{q}_n$ 使 $\operatorname{span}\{\mathbf{q}_1,\cdots,\mathbf{q}_k\} = \operatorname{span}\{\mathbf{a}_1,\cdots,\mathbf{a}_k\}$ 对每个 $k$ 成立。

**第一步（正交化）**：

$$
\mathbf{u}_1 = \mathbf{a}_1, \qquad
\mathbf{u}_k = \mathbf{a}_k - \frac{\mathbf{a}_k\cdot\mathbf{u}_1}{\mathbf{u}_1\cdot\mathbf{u}_1}\mathbf{u}_1 - \cdots - \frac{\mathbf{a}_k\cdot\mathbf{u}_{k-1}}{\mathbf{u}_{k-1}\cdot\mathbf{u}_{k-1}}\mathbf{u}_{k-1}
$$

**第二步（单位化）**：

$$
\mathbf{q}_k = \frac{\mathbf{u}_k}{\|\mathbf{u}_k\|}
$$

**重点**：$\mathbf{u}_k$ 是从 $\mathbf{a}_k$ 中减去它在前面 $\mathbf{u}_1, \cdots, \mathbf{u}_{k-1}$ 方向上的**全部投影**，剩余部分与每个前面的方向都垂直。投影系数 $\frac{\mathbf{a}_k\cdot\mathbf{u}_i}{\mathbf{u}_i\cdot\mathbf{u}_i}$ 是「$\mathbf{a}_k$ 在 $\mathbf{u}_i$ 方向上的分量长度」。

## 2 公式解析：为什么减法后必然正交

以三个向量为例，$k = 2$ 那一步拆开看：

- **第一步，投影的定义**：向量 $\mathbf{a}$ 在单位方向 $\hat{\mathbf{u}}$ 上的投影是 $(\mathbf{a}\cdot\hat{\mathbf{u}})\hat{\mathbf{u}}$。写成非单位形式：$\frac{\mathbf{a}\cdot\mathbf{u}}{\mathbf{u}\cdot\mathbf{u}}\mathbf{u}$——系数是内积比，方向是 $\mathbf{u}$。
- **第二步，做减法**：$\mathbf{u}_2 = \mathbf{a}_2 - \frac{\mathbf{a}_2\cdot\mathbf{u}_1}{\mathbf{u}_1\cdot\mathbf{u}_1}\mathbf{u}_1$。它与 $\mathbf{u}_1$ 的内积为
  $$
  \mathbf{u}_1\cdot\mathbf{u}_2 = \mathbf{u}_1\cdot\mathbf{a}_2 - \frac{\mathbf{a}_2\cdot\mathbf{u}_1}{\mathbf{u}_1\cdot\mathbf{u}_1}(\mathbf{u}_1\cdot\mathbf{u}_1) = \mathbf{u}_1\cdot\mathbf{a}_2 - \mathbf{a}_2\cdot\mathbf{u}_1 = 0
  $$
  投影项精确抵消了「不垂直」的部分，剩余与 $\mathbf{u}_1$ 垂直。
- **第三步，归纳推广**：假设前面 $k-1$ 个 $\mathbf{u}_i$ 已两两正交，$\mathbf{u}_k$ 减去的是在前 $k-1$ 个方向上的投影和。与任一 $\mathbf{u}_j$ 内积时，只有第 $j$ 个投影项有贡献，且精确抵消，其余项内积为零——故 $\mathbf{u}_k$ 与所有前面的方向垂直。
- **第四步，单位化**：$\mathbf{u}_k \ne \mathbf{0}$（由 $\mathbf{a}_k$ 与前面向量的线性无关性保证），除以自己的长度即得 $\mathbf{q}_k$。

<span class="marginnote">「正交化不改变张成的空间」：每一步 $\mathbf{u}_k$ 是 $\mathbf{a}_k$ 与前面 $\mathbf{u}_i$ 的线性组合，而 $\mathbf{a}_k$ 也能由 $\mathbf{u}_1,\cdots,\mathbf{u}_k$ 线性表示——两个集合互相表示，张成的空间逐级相同。这保证了算法「只换方向、不丢信息」。</span>

## 3 一个完整的例子

把 $\mathbf{a}_1 = (1,1,0)$、$\mathbf{a}_2 = (1,0,1)$、$\mathbf{a}_3 = (0,1,1)$ 正交化。

- $\mathbf{u}_1 = (1,1,0)$，$\|\mathbf{u}_1\| = \sqrt2$，$\mathbf{q}_1 = \left(\frac1{\sqrt2}, \frac1{\sqrt2}, 0\right)$。
- 投影系数：$\mathbf{a}_2\cdot\mathbf{u}_1 = 1$，$\mathbf{u}_1\cdot\mathbf{u}_1 = 2$，故
  $$
  \mathbf{u}_2 = (1,0,1) - \frac12(1,1,0) = \left(\frac12, -\frac12, 1\right)
  $$
  检验：$\mathbf{u}_1\cdot\mathbf{u}_2 = \frac12 - \frac12 + 0 = 0$ ✓。$\|\mathbf{u}_2\| = \sqrt{\frac14 + \frac14 + 1} = \sqrt{\frac32}$，$\mathbf{q}_2 = \left(\frac1{\sqrt6}, -\frac1{\sqrt6}, \frac2{\sqrt6}\right)$。
- $\mathbf{a}_3\cdot\mathbf{u}_1 = 1$，$\mathbf{a}_3\cdot\mathbf{u}_2 = -\frac12 + \frac12 + 1 = 1$，$\mathbf{u}_2\cdot\mathbf{u}_2 = \frac32$，故
  $$
  \mathbf{u}_3 = (0,1,1) - \frac12(1,1,0) - \frac{1}{3/2}\left(\frac12, -\frac12, 1\right) = (0,1,1) - \left(\frac12,\frac12,0\right) - \left(\frac13,-\frac13,\frac23\right)
  $$
  $$
  = \left(-\frac56, \frac56, \frac13\right)
  $$
  检验垂直：与 $\mathbf{u}_1, \mathbf{u}_2$ 的内积均为 0 ✓。

三个 $\mathbf{q}_i$ 张成与 $\mathbf{a}_1,\mathbf{a}_2,\mathbf{a}_3$ 相同的空间——本例中即整个 $\mathbb{R}^3$。

## 4 正交基与 QR 分解的衔接

施密特正交化的矩阵形式是 **QR 分解**（第八篇正式展开）：设 $A = (\mathbf{a}_1, \cdots, \mathbf{a}_n)$，正交化得到标准正交列 $Q = (\mathbf{q}_1, \cdots, \mathbf{q}_n)$，则存在上三角矩阵 $R$ 使

$$
A = QR
$$

其中 $R$ 的元素由投影系数组成。**$R$ 上三角**的原因：$\mathbf{a}_k$ 只由前面 $\mathbf{q}_1, \cdots, \mathbf{q}_k$ 表示，第 $k$ 列之后系数为零。

**辨析｜易错点：** 施密特正交化的**数值稳定性**堪忧——当两个向量非常接近时，$\mathbf{u}_k$ 的模极小，后续步骤会放大浮点误差。工程上使用改良的**修正施密特（MGS）**或基于 Householder 反射的 QR 算法（第九篇）。但概念理解上，经典 Gram-Schmidt 是最直接的入口。

## 5 正交化在最小二乘中的角色

施密特正交化与下一篇（第八篇）的最小二乘问题直接相关：

- 最小二乘要找「$b$ 在列空间上的投影」，而列空间的一组**标准正交基**让投影系数 = 内积，计算量大减。
- 正规方程 $A^TA\hat{x} = A^Tb$ 中若 $A$ 的列正交，则 $A^TA$ 是对角矩阵，解 $\hat{x}_j = \frac{\mathbf{a}_j\cdot\mathbf{b}}{\mathbf{a}_j\cdot\mathbf{a}_j}$ 变为逐分量除法。

**重点**：**正交化的价值 = 把「求解」变成「内积」**。凡是列空间有标准正交基的地方，投影、最小二乘、分解都变得又便宜又稳定。这也是为什么后续所有分解（QR、SVD）都极力追求正交性。

**补充｜正交化的「在线」读法**：施密特正交化也可以看作「逐步学习」：第 $k$ 步已知前 $k-1$ 个正交方向，把新向量 $\mathbf{a}_k$ 中「已经被覆盖的成分」（投影）剔除，得到「全新的成分」。这正是**信号处理里新息（innovation）的概念**——每一步提取出「前面没有见过的部分」。卡尔曼滤波、Gram-Schmidt 与 QR、以及时序预测的新息序列，全是同一思想。**「减去投影 = 提取新息」**是正交化思想最深刻的延伸。

**补充｜正交化的「在线」读法**：施密特正交化也可以看作「逐步学习」：第 $k$ 步已知前 $k-1$ 个正交方向，把新向量 $\mathbf{a}_k$ 中「已经被覆盖的成分」（投影）剔除，得到「全新的成分」。这正是**信号处理里新息（innovation）的概念**——每一步提取出「前面没有见过的部分」。卡尔曼滤波、Gram-Schmidt 与 QR、以及时序预测的新息序列，全是同一思想。**「减去投影 = 提取新息」**是正交化思想最深刻的延伸。

**辨析｜易错点：** 施密特正交化的计算细节：

- 每步**必须先正交化再单位化**（先减投影，后除以长度），顺序不能反；
- 减投影时用的分母是 $\mathbf{u}_i \cdot \mathbf{u}_i$（已正交化的向量），不是 $\mathbf{a}_i \cdot \mathbf{a}_i$；
- 若某步 $\mathbf{u}_k = \mathbf{0}$，说明 $\mathbf{a}_k$ 与前面线性相关——原向量组不无关，正交化无法继续。

**「分母用 $\mathbf{u}$ 不用 $\mathbf{a}$」**是最常见的计算失误。

**补充｜一步总结**：施密特正交化 = 「逐向量减去它在已建正交空间上的投影，再单位化」。它把「任何一组无关向量」变成「张成同一空间的标准正交基」，是正交基构造的万能算法。

## 6 小结

- **算法**：每步减去往前的投影，剩余部分正交，再单位化。
- **不变量**：$\operatorname{span}\{\mathbf{u}_1,\cdots,\mathbf{u}_k\} = \operatorname{span}\{\mathbf{a}_1,\cdots,\mathbf{a}_k\}$，张成空间逐级不变。
- **投影系数**：$\frac{\mathbf{a}\cdot\mathbf{u}_i}{\mathbf{u}_i\cdot\mathbf{u}_i}$，减去它就抵消不垂直分量。
- **QR 衔接**：施密特 = $A = QR$（$Q$ 标准正交列，$R$ 上三角）。
- **应用**：正交基让坐标与投影变成纯内积运算。

在下一节，我们将进入本专题最重要的概念之一——**方阵的特征值与特征向量**，回答「矩阵在哪些方向上只伸缩、不旋转」。
