---
title: 向量范数与矩阵范数
date: 2026-08-07
---

# 向量范数与矩阵范数：度量矩阵的「大小」

<div class="epigraph">
<p>没有度量，就没有误差；没有误差，就没有数值分析。</p>
<footer>—— 数值分析的公理</footer>
</div>

<div class="article-byline">
<p>第二级 · 数值分析 ｜ 李庆扬《数值分析》§5.5 ｜ 2026-08-07</p>
</div>

## 为什么从范数开始

前面的误差分析多次提到 $\lVert\Delta\mathbf{x}\rVert$、$\mathrm{cond}(A)=\lVert A\rVert\lVert A^{-1}\rVert$——可「向量的长度」「矩阵的大小」还没严格定义。**范数（norm）** 就是这把尺子。它是本节与下一节（条件数与病态方程组）的理论地基：没有范数，谈不了「解有多大误差」「矩阵有多坏」。<span class="marginnote">回顾：第一节《条件数》里我们「借用」了范数的直觉——$\lVert A\rVert$ 是「矩阵的最大放大率」。本节把它严格化：矩阵范数不是随便一个数，而是与向量范数「兼容」的算子范数。</span>

本节定义向量范数、矩阵范数（算子范数），给出最常用的三种（1、2、∞），并建立它们之间的关系。

## 1 向量范数：长度的公理化

**定义。** 映射 $\lVert\cdot\rVert:\mathbb{R}^n\to\mathbb{R}$ 是**向量范数**，若满足三条公理：

1. **正定性**：$\lVert\mathbf{x}\rVert\ge0$，等号当且仅当 $\mathbf{x}=\mathbf{0}$。
2. **齐次性**：$\lVert\alpha\mathbf{x}\rVert=|\alpha|\lVert\mathbf{x}\rVert$。
3. **三角不等式**：$\lVert\mathbf{x}+\mathbf{y}\rVert\le\lVert\mathbf{x}\rVert+\lVert\mathbf{y}\rVert$。

最常用的三种向量范数：

$$
\lVert\mathbf{x}\rVert_1 = \sum_{i=1}^n|x_i| \quad \text{（1-范数）}
$$

$$
\lVert\mathbf{x}\rVert_2 = \sqrt{\sum_{i=1}^n x_i^2} = \sqrt{\mathbf{x}^\top\mathbf{x}} \quad \text{（2-范数 / 欧氏范数）}
$$

$$
\lVert\mathbf{x}\rVert_\infty = \max_{1\le i\le n}|x_i| \quad \text{（∞-范数 / 最大范数）}
$$

一般地 $p$-范数 $\lVert\mathbf{x}\rVert_p=\bigl(\sum|x_i|^p\bigr)^{1/p}$。**几何直觉**：1-范数是曼哈顿距离，2-范数是直线距离，∞-范数是「最远分量」——同一个向量，三把尺子量出三个数。<span class="marginnote">范数的等价性：<strong>$\mathbb{R}^n$ 上一切范数等价</strong>——存在常数 $c_1,c_2$ 使 $c_1\lVert\mathbf{x}\rVert_\alpha\le\lVert\mathbf{x}\rVert_\beta\le c_2\lVert\mathbf{x}\rVert_\alpha$。这保证「收敛与否」不依赖选哪把尺子；但<strong>条件数的数值大小</strong>依赖范数（差个常数因子），工程上关注数量级。</span>

**常用等价关系**（$\mathbf{x}\in\mathbb{R}^n$）：

$$
\lVert\mathbf{x}\rVert_2 \le \lVert\mathbf{x}\rVert_1 \le \sqrt{n}\lVert\mathbf{x}\rVert_2, \qquad \lVert\mathbf{x}\rVert_\infty \le \lVert\mathbf{x}\rVert_2 \le \sqrt{n}\lVert\mathbf{x}\rVert_\infty
$$

## 2 矩阵范数：算子范数的定义

矩阵「大小」怎么定？最自然的做法是**诱导范数（operator norm / induced norm）**：把 $A$ 看成「把向量放大成向量」的算子，$A$ 的范数 = 它最多能把单位向量放大到多长：

$$
\lVert A\rVert = \max_{\mathbf{x}\neq0}\frac{\lVert A\mathbf{x}\rVert}{\lVert\mathbf{x}\rVert} = \max_{\lVert\mathbf{x}\rVert=1}\lVert A\mathbf{x}\rVert
$$

**这就是第一节里「$\lVert A\rVert$ 是最大放大率」的严格定义。** 诱导范数自动满足三个关键性质：

- **相容性（submultiplicativity）**：$\lVert A\mathbf{x}\rVert\le\lVert A\rVert\lVert\mathbf{x}\rVert$，$\lVert AB\rVert\le\lVert A\rVert\lVert B\rVert$。
- $\lVert I\rVert=1$。
- 与向量范数「配套」（由它诱导而来）。

三种常用矩阵范数（对 $A\in\mathbb{R}^{m\times n}$）：

$$
\lVert A\rVert_1 = \max_{1\le j\le n}\sum_{i=1}^{m}|a_{ij}| \quad \text{（最大列和）}
$$

$$
\lVert A\rVert_\infty = \max_{1\le i\le m}\sum_{j=1}^{n}|a_{ij}| \quad \text{（最大行和）}
$$

$$
\lVert A\rVert_2 = \sqrt{\lambda_{\max}(A^\top A)} \quad \text{（谱范数）}
$$

**记忆**：1-范数 = 最大列绝对值和，∞-范数 = 最大行绝对值和，2-范数 = $A^\top A$ 最大特征值的平方根。<span class="marginnote">为什么 1-范数是「列和」而 ∞-范数是「行和」？因为 $\lVert Ax\rVert_1=\sum_i|\sum_j a_{ij}x_j|\le\sum_j|x_j|\sum_i|a_{ij}|$——放大率由「某列绝对值总和最大者」决定；$\lVert Ax\rVert_\infty$ 则由「某行的绝对值总和」决定。<strong>列和行和的记忆锚：1 对列，∞ 对行。</strong></span>

## 3 公式解析：谱范数与特征值

谱范数 $\lVert A\rVert_2=\sqrt{\lambda_{\max}(A^\top A)}$ 最常用也最难算，拆解它的来由：

**第一步，平方展开。** $\lVert A\mathbf{x}\rVert_2^2=(A\mathbf{x})^\top(A\mathbf{x})=\mathbf{x}^\top A^\top A\mathbf{x}$。
**第二步，瑞利商。** 最大化 $\dfrac{\mathbf{x}^\top(A^\top A)\mathbf{x}}{\mathbf{x}^\top\mathbf{x}}$。由瑞利商定理，最大值是 $A^\top A$ 的**最大特征值** $\lambda_{\max}(A^\top A)$。
**第三步，开方。** $\lVert A\rVert_2=\sqrt{\lambda_{\max}(A^\top A)}$。$A^\top A$ 对称半正定，特征值非负，开方合法。

**当 $A$ 对称时**：$A^\top A=A^2$，$\lVert A\rVert_2=|\lambda|_{\max}$——**对称矩阵的谱范数就是它的最大特征值绝对值**，这给特征值问题的误差分析铺了路。

**特殊情形**：正交矩阵 $Q$（$Q^\top Q=I$）：$\lVert Q\rVert_2=1$，$\lVert Q\mathbf{x}\rVert_2=\lVert\mathbf{x}\rVert_2$——**正交变换不改变 2-范数**。这是数值分析里「QR 分解稳定」的根源：正交化不放大误差。

## 4 三种矩阵范数的对照与计算

| 范数 | 公式 | 名称 | 计算成本 |
| --- | --- | --- | --- |
| $\lVert A\rVert_1$ | $\max$ 列绝对值和 | 列和范数 | $O(mn)$ |
| $\lVert A\rVert_\infty$ | $\max$ 行绝对值和 | 行和范数 | $O(mn)$ |
| $\lVert A\rVert_2$ | $\sqrt{\lambda_{\max}(A^\top A)}$ | 谱范数 | $O(n^3)$（需特征值） |

**工程事实**：1-与 ∞-范数**算起来便宜**（一次扫描），谱范数贵（要特征值）。但谱范数有最好的几何意义（球变换成椭圆，最大半轴长）。工程估计条件数常用 1-或 ∞-范数（便宜），或直接算谱范数（numpy 的 `np.linalg.norm(A, ord=2)` 底层走 SVD）。

**数值例子**：$A=\begin{pmatrix}1&-2\\-3&4\end{pmatrix}$。$\lVert A\rVert_1=\max(|1|+|{-3}|,\ |{-2}|+|4|)=\max(4,6)=6$；$\lVert A\rVert_\infty=\max(1+2,\ 3+4)=7$；$A^\top A=\begin{pmatrix}10&-14\\-14&20\end{pmatrix}$，特征值 $\lambda_{\max}=30$（解 $\lambda^2-30\lambda+4=0$，$\lambda=15+\sqrt{221}\approx29.87$），$\lVert A\rVert_2\approx5.46$。<span class="marginnote">验证三种范数同尺度的例子：$5.46\le6$ 与 $7$ 相近但不等——<strong>不同范数的具体数值可以不同，但「数量级」一致</strong>。判断病态时看数量级，不纠结精确值。</span>

## 5 范数在误差分析中的角色

范数是整条误差分析链的「度量单位」：

- **解的误差**：$\lVert\Delta\mathbf{x}\rVert$ 度量解偏离多大。
- **矩阵的放大**：$\lVert A\rVert$ 度量「输入误差→输出误差」的放大。
- **条件数**：$\mathrm{cond}(A)=\lVert A\rVert\lVert A^{-1}\rVert$，直接由范数定义。

**辨析｜易错点：** 矩阵范数**不是**「把矩阵拉平成一维数组再取范数」——那种叫「Frobenius 范数」$\lVert A\rVert_F=\sqrt{\sum a_{ij}^2}$，它不满足 $\lVert I\rVert=1$，**不是诱导范数**。条件数必须用诱导范数。Frobenius 范数常用于逼近误差度量（如 $\lVert A-\hat A\rVert_F$），两者用途不同。

## 6 小结

- **向量范数**：正定性 + 齐次性 + 三角不等式；常用 1-、2-、∞-范数，三者等价（差常数因子）。
- **矩阵范数（诱导范数）**：$\lVert A\rVert=\max_{\lVert x\rVert=1}\lVert A\mathbf{x}\rVert$，满足相容性 $\lVert AB\rVert\le\lVert A\rVert\lVert B\rVert$ 与 $\lVert I\rVert=1$。
- $\lVert A\rVert_1$=最大列和、$\lVert A\rVert_\infty$=最大行和、$\lVert A\rVert_2=\sqrt{\lambda_{\max}(A^\top A)}$（谱范数）。
- 对称矩阵 $\lVert A\rVert_2=|\lambda|_{\max}$；正交矩阵谱范数恒 1（QR 稳定之源）。
- Frobenius 范数非诱导范数，不用于条件数；1-、∞-范数便宜，谱范数贵但几何意义好。

在下一节，我们用范数完成线性方程组的误差分析：**线性方程组的误差分析：条件数与病态方程组**——把「解有多可靠」与条件数严格挂钩。
