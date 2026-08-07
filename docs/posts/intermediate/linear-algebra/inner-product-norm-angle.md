---
title: 向量的内积、长度与夹角
date: 2026-08-08
---

# 向量的内积、长度与夹角

<div class="epigraph">
<p>两个向量的内积，是在问它们「方向一致到什么程度」——为零即垂直，为正即同向偏多，为负即反向偏多。</p>
<footer>—— 柯西（Augustin-Louis Cauchy，柯西-施瓦茨不等式）</footer>
</div>

<div class="article-byline">
<p>第二级 · 线性代数 ｜ 同济《线性代数》§5.1 ｜ 2026-08-08</p>
</div>

## 为什么从内积开始

前面的线性代数只有「方向」没有「度量」：向量能不能表示、是否相关、解空间多大——都不需要长度与夹角。但从这一篇起，我们要引入**度量**：向量的**长度**、**夹角**、**垂直**。这把线性代数从「代数」带入「几何」，也让投影、最小二乘、SVD 这些最强应用成为可能。<span class="marginnote">内积是「度量」的入口：一旦定义了内积，就能定义长度（自身内积开方）、夹角（内积与长度的商）、垂直（内积为零）。整个欧氏几何都能从这一个运算推出。在机器学习的特征空间里，「相似度」常常就定义为内积或它的归一化——余弦相似度本质就是两向量夹角的余弦（第十一篇）。</span>

这一节的内容很基本，但它是第五篇正交性、第八篇投影最小二乘的算术地基。

## 1 内积的定义

**核心概念**：设 $\mathbf{x} = (x_1, \cdots, x_n)$、$\mathbf{y} = (y_1, \cdots, y_n)$ 是 $\mathbb{R}^n$ 中的两个向量，定义它们的内积（点积）

$$
\mathbf{x} \cdot \mathbf{y} = x_1y_1 + x_2y_2 + \cdots + x_ny_n = \sum_{i=1}^{n} x_i y_i
$$

内积是一个**数**（标量），不是向量。用矩阵记号可写成 $\mathbf{x} \cdot \mathbf{y} = \mathbf{x}^T \mathbf{y}$。

内积满足三条基本性质：

- **对称性**：$\mathbf{x} \cdot \mathbf{y} = \mathbf{y} \cdot \mathbf{x}$；
- **线性性**：$(\lambda\mathbf{x} + \mu\mathbf{z}) \cdot \mathbf{y} = \lambda(\mathbf{x}\cdot\mathbf{y}) + \mu(\mathbf{z}\cdot\mathbf{y})$；
- **正定性**：$\mathbf{x} \cdot \mathbf{x} \ge 0$，等号成立当且仅当 $\mathbf{x} = \mathbf{0}$。

这三条合称**内积公理**——任何满足它们的运算都可当作内积使用（第六篇在线性空间上重新定义内积时正是这么做的）。

## 2 长度与夹角

**核心概念**：向量的**长度（模，范数）**定义为

$$
\|\mathbf{x}\| = \sqrt{\mathbf{x} \cdot \mathbf{x}} = \sqrt{x_1^2 + \cdots + x_n^2}
$$

两个非零向量 $\mathbf{x}, \mathbf{y}$ 的**夹角** $\theta$ 由下式定义：

$$
\cos\theta = \frac{\mathbf{x} \cdot \mathbf{y}}{\|\mathbf{x}\|\, \|\mathbf{y}\|}, \qquad 0 \le \theta \le \pi
$$

**重点**：这条公式是「余弦相似度」的来源——**内积除以长度乘积 = 夹角的余弦**。它把「方向上的相似」量化成一个介于 $-1$ 与 $1$ 之间的数：$\theta = 0$ 时 $\cos\theta = 1$（同向），$\theta = \pi/2$ 时 $\cos\theta = 0$（垂直），$\theta = \pi$ 时 $\cos\theta = -1$（反向）。

**辨析｜易错点：** 内积为零 $\Leftrightarrow$ 两向量垂直（正交），这是**最重要的内积判据**。注意：零向量与任何向量都「垂直」（内积恒为 0），所以「正交」通常默认两个非零向量。另外，$\|\mathbf{x}\|$ 不是 $\mathbf{x}$ 的分量之和——初学者常把长度与「各分量绝对值和」混淆。

## 3 柯西-施瓦茨不等式

**定理（Cauchy-Schwarz 不等式）**：对任意 $\mathbf{x}, \mathbf{y} \in \mathbb{R}^n$，

$$
|\mathbf{x} \cdot \mathbf{y}| \le \|\mathbf{x}\|\, \|\mathbf{y}\|
$$

等号成立当且仅当 $\mathbf{x}, \mathbf{y}$ 线性相关（成比例）。

**重点**：这个不等式保证上面的夹角公式**有意义**——$\cos\theta$ 的分子不会超过分母，比值恒在 $[-1, 1]$ 内。它是全数学最常用的不等式之一：在概率论里给出 $|E[XY]| \le \sqrt{E[X^2]E[Y^2]}$，在信号处理里给出相关性界，在机器学习里保证余弦相似度取值范围合法。

<span class="marginnote">柯西-施瓦茨不等式的一个漂亮推论是<strong>三角不等式</strong>：$\|\mathbf{x} + \mathbf{y}\| \le \|\mathbf{x}\| + \|\mathbf{y}\|$——两点间直线最短。它在范数理论（第九篇）中被推广到矩阵范数，成为数值分析稳定性分析的基石。</span>

## 4 公式解析：$\cos\theta = \dfrac{\mathbf{x}\cdot\mathbf{y}}{\|\mathbf{x}\|\,\|\mathbf{y}\|}$

这条公式为什么成立？从二维勾股定理推广来理解：

- **第一步，二维情形的几何事实**：在平面里，两向量 $\mathbf{x}, \mathbf{y}$ 的夹角 $\theta$ 满足「**余弦定理**」：$\|\mathbf{x} - \mathbf{y}\|^2 = \|\mathbf{x}\|^2 + \|\mathbf{y}\|^2 - 2\|\mathbf{x}\|\|\mathbf{y}\|\cos\theta$。
- **第二步，展开左边**：$\|\mathbf{x} - \mathbf{y}\|^2 = (\mathbf{x}-\mathbf{y})\cdot(\mathbf{x}-\mathbf{y}) = \|\mathbf{x}\|^2 + \|\mathbf{y}\|^2 - 2\mathbf{x}\cdot\mathbf{y}$。
- **第三步，两边对比**：余弦定理与展开式比较，$-2\|\mathbf{x}\|\|\mathbf{y}\|\cos\theta = -2\mathbf{x}\cdot\mathbf{y}$，于是 $\cos\theta = \frac{\mathbf{x}\cdot\mathbf{y}}{\|\mathbf{x}\|\|\mathbf{y}\|}$。
- **第四步，推广到 $n$ 维**：上式在整个推导中只用了内积公理与「$\|\mathbf{x}\|^2 = \mathbf{x}\cdot\mathbf{x}$」，没有任何依赖维数的步骤——所以公式对 $n$ 维同样成立。**二维几何事实 + 内积代数 = $n$ 维夹角定义**。

## 5 一个实际例子

设 $\mathbf{x} = (1, 2, 2)$、$\mathbf{y} = (4, 0, 3)$。

- 长度：$\|\mathbf{x}\| = \sqrt{1 + 4 + 4} = 3$，$\|\mathbf{y}\| = \sqrt{16 + 0 + 9} = 5$。
- 内积：$\mathbf{x}\cdot\mathbf{y} = 1\cdot4 + 2\cdot0 + 2\cdot3 = 10$。
- 夹角：$\cos\theta = \frac{10}{3 \cdot 5} = \frac{2}{3}$，所以 $\theta \approx 48.2°$。

这个例子提醒我们：**长度是内积的平方根，夹角是内积与长度乘积之比**——全部几何量都从内积导出。余弦相似度 $\cos\theta = 2/3$ 就是两个三维向量「方向相似度」的量化。

**补充｜内积的一般化：从点积到函数内积**：只要满足内积公理（对称、线性、正定），任何运算都可当作内积。函数空间 $C[a,b]$ 上的内积 $\langle f, g \rangle = \int_a^b f(x)g(x)\,dx$ 同样定义「函数之间的夹角与长度」——两个函数「正交」即积分值为零（如 $\sin x$ 与 $\cos x$ 在一个周期上）。**傅里叶分析就是在函数空间里做「投影到正交基」**（第二级《复变函数与积分变换》）。「内积」从数扩展到函数，是线性代数通向分析的桥梁。

**补充｜内积的一般化：从点积到函数内积**：只要满足内积公理（对称、线性、正定），任何运算都可当作内积。函数空间 $C[a,b]$ 上的内积 $\langle f, g \rangle = \int_a^b f(x)g(x)\,dx$ 同样定义「函数之间的夹角与长度」——两个函数「正交」即积分值为零（如 $\sin x$ 与 $\cos x$ 在一个周期上）。**傅里叶分析就是在函数空间里做「投影到正交基」**（第二级《复变函数与积分变换》）。「内积」从数扩展到函数，是线性代数通向分析的桥梁。

**辨析｜易错点：** 内积运算的常见错误：

- $\mathbf{x}\cdot\mathbf{y}$ 是**数**不是向量——写 $\mathbf{x}\cdot\mathbf{y}$ 时结果是一个标量；
- 内积**不满足消去律**：$\mathbf{x}\cdot\mathbf{z} = \mathbf{y}\cdot\mathbf{z}$ 推不出 $\mathbf{x} = \mathbf{y}$（取 $\mathbf{z} \perp (\mathbf{x}-\mathbf{y})$ 即反例）；
- $(\mathbf{x}\cdot\mathbf{y})\mathbf{z}$ 中先算内积得数、再数乘向量——优先级别搞错。

**「内积得数，数乘得向量」**是分量层级最易混的两件事。

**补充｜内积的工程应用速览**：

- **余弦相似度**：$\cos\theta = \frac{\mathbf{x}\cdot\mathbf{y}}{\|\mathbf{x}\|\|\mathbf{y}\|}$，文本/向量检索的相似度度量（第四级《信息检索》）；
- **投影**：$\operatorname{proj}_{\mathbf{a}}\mathbf{b} = \frac{\mathbf{b}\cdot\mathbf{a}}{\mathbf{a}\cdot\mathbf{a}}\mathbf{a}$，最小二乘的基石（第八篇）；
- **相关性**：中心化向量的夹角余弦 = 相关系数（第十一篇）；
- **正交性判据**：内积为零 ⇔ 垂直，解空间与行空间的正交关系（第七篇）。

**「内积一个运算，撑起相似度、投影、相关、正交四大应用」**——它是度量几何的单一源头。

## 6 小结

- **内积**：$\mathbf{x}\cdot\mathbf{y} = \sum x_iy_i = \mathbf{x}^T\mathbf{y}$，满足对称、线性、正定三条公理。
- **长度**：$\|\mathbf{x}\| = \sqrt{\mathbf{x}\cdot\mathbf{x}}$，勾股定理的 $n$ 维推广。
- **夹角**：$\cos\theta = \frac{\mathbf{x}\cdot\mathbf{y}}{\|\mathbf{x}\|\|\mathbf{y}\|}$，由余弦定理推广而来。
- **垂直**：$\mathbf{x}\cdot\mathbf{y} = 0 \Leftrightarrow$ 正交。
- **柯西-施瓦茨**：$|\mathbf{x}\cdot\mathbf{y}| \le \|\mathbf{x}\|\|\mathbf{y}\|$，保证夹角有意义。

在下一节，我们将利用内积引入「垂直」的结构化版本——**正交向量组与正交矩阵**，为施密特正交化与正交对角化做准备。
