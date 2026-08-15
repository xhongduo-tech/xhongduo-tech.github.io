---
title: 多元数据的矩阵表示与样本几何
date: 2026-08-07
---

# 多元数据的矩阵表示与样本几何

<div class="epigraph">
<p>哪里有数据，哪里就有大量、杂乱无章的数据，统计学的任务就是从中理出秩序；而矩阵是多元数据最自然的家。</p>
<footer>—— 理查德·约翰逊 与 迪恩·威彻恩（Richard A. Johnson & Dean W. Wichern）</footer>
</div>

<div class="article-byline">
<p>第二级 · 多元统计分析 ｜ Johnson & Wichern《Applied Multivariate Statistical Analysis》Ch.2–3 · Anderson Ch.1 ｜ 2026-08-07</p>
</div>

## 为什么从矩阵开始

一元统计研究一个变量：一组身高数据、一组血压数据。但现实中几乎每个个体都同时带着一堆测量值——一个人的身高、体重、心率、血脂；一家企业的市值、负债率、营收增速；一株作物的株高、穗长、产量。**这些变量不是彼此独立的，而是纠结在一起共同刻画一个个体**。多元统计分析（multivariate analysis）的全部任务，就是研究这种「多个变量一起看」的结构。而要把 n 个个体、p 个变量同时装进头脑，矩阵就是唯一现实的语言。<span class="marginnote">你已经在第二级《线性代数》里认识了矩阵的运算与分解，这里是它们的第一次大规模实战：数据本身就是一张矩阵，协方差矩阵是它的二阶矩，特征值分解则会在后面的主成分分析里登场。</span>

## 1 多元观测：每个个体是一行

设有 $n$ 个个体（样本量），每个个体测量 $p$ 个变量，那么全部数据可以写成一张 $n \times p$ 矩阵：

$$
\mathbf{X} = \begin{pmatrix}
x_{11} & x_{12} & \cdots & x_{1p} \\
x_{21} & x_{22} & \cdots & x_{2p} \\
\vdots & \vdots & \ddots & \vdots \\
x_{n1} & x_{n2} & \cdots & x_{np}
\end{pmatrix}
$$

约定：**行对应个体（观测），列对应变量**。第 $i$ 行 $\mathbf{x}_i' = (x_{i1}, x_{i2}, \ldots, x_{ip})$ 是第 $i$ 个个体在 $p$ 个变量上的取值，称为一个**观测向量**；第 $j$ 列则是一元意义下的第 $j$ 个变量的 $n$ 个观测值。<span class="marginnote">把行看作点、列看作坐标轴，这张矩阵就同时是「$n$ 个点在 $p$ 维空间中的坐标表」。这一句话是整章的地基：后面的样本几何、主成分、聚类，都在反复回到「点」与「轴」这两种读法。</span>

数据矩阵是研究多元关系的原料，但它只是一堆数字。**从数字里提炼结构，靠的是两类统计量：均值向量与协方差矩阵**。

## 2 样本均值向量与样本协方差矩阵

先看**样本均值向量（sample mean vector）**，它把每一列分别平均：

$$
\bar{\mathbf{x}} = \frac{1}{n} \mathbf{X}' \mathbf{1}_n = \frac{1}{n} \sum_{i=1}^{n} \mathbf{x}_i = \begin{pmatrix} \bar{x}_1 \\ \bar{x}_2 \\ \vdots \\ \bar{x}_p \end{pmatrix}
$$

其中 $\mathbf{1}_n = (1,1,\ldots,1)'$ 是 $n$ 维全 1 向量，$\bar{x}_j = \frac{1}{n}\sum_{i=1}^n x_{ij}$ 就是第 $j$ 个变量的普通样本均值。均值向量是「这堆数据的重心」，是一个 $p$ 维点。

再看**样本协方差矩阵（sample covariance matrix）**，它是 p 个变量两两之间协方差拼成的矩阵：

$$
\mathbf{S} = \frac{1}{n-1} \sum_{i=1}^{n} (\mathbf{x}_i - \bar{\mathbf{x}})(\mathbf{x}_i - \bar{\mathbf{x}})' = \begin{pmatrix}
s_{11} & s_{12} & \cdots & s_{1p} \\
s_{21} & s_{22} & \cdots & s_{2p} \\
\vdots & \vdots & \ddots & \vdots \\
s_{p1} & s_{p2} & \cdots & s_{pp}
\end{pmatrix}
$$

对角线 $s_{jj}$ 是第 $j$ 个变量的样本方差，非对角线 $s_{jk}$ 是变量 $j$ 与 $k$ 的样本协方差。用矩阵乘法可以写得更紧凑：若记 $\mathbf{x}_i - \bar{\mathbf{x}}$ 为去均值后的第 $i$ 行，把去均值矩阵 $\tilde{\mathbf{X}}$ 的第 $i$ 行写成 $\mathbf{x}_i - \bar{\mathbf{x}}'$，则

$$
\mathbf{S} = \frac{1}{n-1} \tilde{\mathbf{X}}' \tilde{\mathbf{X}}
$$

这条写法极具价值：**样本协方差矩阵是一个「数据矩阵自乘」的结果**，正定性与半正定性、PCA 里的特征分解，全都从这里来。

### 样本相关矩阵：去掉量纲再看关系

协方差有个毛病：它带单位。身高用厘米、体重用千克，$s_{jk}$ 的数值大小就取决于单位选择，跨数据集无法比较。于是把每个变量标准化到方差为 1，得到**样本相关矩阵（sample correlation matrix）**：

$$
\mathbf{R} = \mathbf{D}^{-1/2} \mathbf{S} \mathbf{D}^{-1/2}, \qquad \mathbf{D} = \operatorname{diag}(s_{11}, s_{22}, \ldots, s_{pp})
$$

$\mathbf{R}$ 的第 $(j,k)$ 元就是普通的一元相关系数 $r_{jk} = s_{jk}/\sqrt{s_{jj}s_{kk}}$。**相关矩阵是「无量纲化」的协方差矩阵**：对角线全是 1，非对角线落在 $[-1, 1]$。<span class="marginnote">PCA 一章的经典教训就在这里埋着：当变量量纲悬殊时，直接对 $\mathbf{S}$ 做主成分等价于对量纲大的变量悄悄加权重，通常应改对 $\mathbf{R}$ 做。这是「先标准化再分析」在多元统计中的第一次出现。</span>

| 矩阵 | 第 $(j,k)$ 元 | 取值范围 | 是否受量纲影响 | 典型用途 |
| --- | --- | --- | --- | --- |
| $\mathbf{S}$ | 协方差 $s_{jk}$ | 无固定范围 | 是 | 刻画原始尺度下的散布、用于推断 |
| $\mathbf{R}$ | 相关系数 $r_{jk}$ | $[-1, 1]$ | 否 | 量纲悬殊时比较变量关系 |

## 3 样本几何：数据中心化与 p 维散点

矩阵 $\mathbf{X}$ 还有第二种读法。把每个观测 $\mathbf{x}_i$ 当作 $\mathbb{R}^p$ 中的一个点，$n$ 个点就在 $p$ 维空间里撒开一团「云」。均值向量 $\bar{\mathbf{x}}$ 是这团云的**重心**。把坐标原点搬到重心，得到**中心化数据（centered data）**：

$$
\tilde{\mathbf{x}}_i = \mathbf{x}_i - \bar{\mathbf{x}}, \qquad i = 1, \ldots, n
$$

中心化之后，数据云的重心落在原点。此时 $\frac{1}{n-1}\tilde{\mathbf{X}}'\tilde{\mathbf{X}}$ 刻画的正是这团云相对重心的**散布形态**：$s_{jj}$ 越大，云沿第 $j$ 个坐标轴拉得越长；$s_{jk}$ 越大，云在 $j$、$k$ 两个方向上的延伸越同步。**协方差矩阵就是数据云的「形状矩阵」**——把散点云与协方差矩阵建立一一对应，是理解多元统计最重要的几何直觉。<span class="marginnote">当 $p = 2$ 时，$\mathbf{S}$ 的三个独立元素 $s_{11}, s_{22}, s_{12}$ 恰好决定椭圆的长短轴与朝向——等高椭圆的主轴方向就是 $\mathbf{S}$ 的特征向量方向。这个「云→椭圆→特征分解」的链条，是下一章多元正态与后面主成分分析的共同骨架。</span>

### 广义方差与随机样本

把散布整体压缩成一个数，有**广义方差（generalized variance）**：

$$
|\mathbf{S}| = \det \mathbf{S}
$$

行列式衡量的是数据云的「体积」：$p = 2$ 时它就是主轴椭圆的面积因子。行列式太小，说明某些变量几乎线性相关、数据实际维数低于 $p$——这是后面讨论降维的伏笔。<span class="marginnote">另一个常被盯住的数是总方差 $\operatorname{tr}\mathbf{S} = \sum_j s_{jj}$。二者互补：迹看「总能量」，行列式看「有效体积」；PCA 用前者选轴，共线性诊断看后者。</span>

最后，把观测视为随机向量 $\mathbf{X}_1, \ldots, \mathbf{X}_n \sim F$（独立同分布），那么 $\bar{\mathbf{x}}$ 与 $\mathbf{S}$ 就分别是总体均值向量 $\boldsymbol{\mu} = E(\mathbf{X})$ 与总体协方差矩阵 $\boldsymbol{\Sigma} = E[(\mathbf{X}-\boldsymbol{\mu})(\mathbf{X}-\boldsymbol{\mu})']$ 的**无偏估计**：$E(\bar{\mathbf{x}}) = \boldsymbol{\mu}$，$E(\mathbf{S}) = \boldsymbol{\Sigma}$。这里分母用 $n-1$ 而非 $n$，正是一元统计学「贝塞尔校正」在多元情形的推广。

## 4 公式解析：样本协方差矩阵的无偏性

**为什么协方差矩阵的分母是 $n-1$？** 这是进入多元统计的第一个易错点，值得拆开看。

先算去均值向量 $E(\tilde{\mathbf{x}}_i) = E(\mathbf{x}_i - \bar{\mathbf{x}})$。由于 $\mathbf{x}_i$ 独立同分布且 $E(\bar{\mathbf{x}}) = \boldsymbol{\mu}$，有

$$
E(\mathbf{x}_i - \bar{\mathbf{x}}) = \boldsymbol{\mu} - \boldsymbol{\mu} = \mathbf{0}
$$

- **第一步，对单个乘积取期望**：把 $(\mathbf{x}_i - \bar{\mathbf{x}})(\mathbf{x}_i - \bar{\mathbf{x}})'$ 展开，$\bar{\mathbf{x}}$ 里混入了 $\mathbf{x}_i$ 自身，导致 $E(\mathbf{x}_i - \bar{\mathbf{x}})(\mathbf{x}_i - \bar{\mathbf{x}})' = \boldsymbol{\Sigma} - \frac{1}{n}\boldsymbol{\Sigma}$——期望落在 $\boldsymbol{\Sigma}$ 的 $(n-1)/n$ 倍上。
- **第二步，求和**：$n$ 个项相加，得到 $(n-1)\boldsymbol{\Sigma}$。
- **第三步，除以 $n-1$ 还原**：$E(\mathbf{S}) = \frac{1}{n-1} \cdot (n-1)\boldsymbol{\Sigma} = \boldsymbol{\Sigma}$。

**核心结论：正是那个多余的「自身混入均值」使每个乘积偏小，必须用 $n-1$ 来校正**。一元时只有一个方差，多元时则是整个矩阵同时被校正——用 $n$ 做分母得到的矩阵是有偏的，会系统低估散布。

## 5 小结

- 多元数据是 $n \times p$ 矩阵 $\mathbf{X}$：**行是观测、列是变量**；行读法见「$n$ 个点」，列读法见「$p$ 个坐标轴」。
- **样本均值向量** $\bar{\mathbf{x}}$ 是数据重心，**样本协方差矩阵** $\mathbf{S} = \frac{1}{n-1}\tilde{\mathbf{X}}'\tilde{\mathbf{X}}$ 是数据云的形状矩阵。
- **中心化**把原点搬到重心；$\mathbf{S}$ 的特征方向对应数据云的主轴，是后面 PCA 的伏笔。
- **广义方差** $|\mathbf{S}|$ 与**总方差** $\operatorname{tr}\mathbf{S}$ 从体积与能量两个角度刻画散布。
- $\bar{\mathbf{x}}$ 与 $\mathbf{S}$ 分别是 $\boldsymbol{\mu}$ 与 $\boldsymbol{\Sigma}$ 的无偏估计；分母 $n-1$ 来自「均值混入自身」的贝塞尔校正。

在下一节，我们将为这团数据云指定一个概率模型——**多元正态分布**：它如何由 $\boldsymbol{\mu}$ 与 $\boldsymbol{\Sigma}$