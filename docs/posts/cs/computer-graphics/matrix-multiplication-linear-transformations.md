---
title: 矩阵：矩阵乘法与线性变换的表示
date: 2026-08-07
---

# 矩阵：矩阵乘法与线性变换的表示

<div class="epigraph">
<p>数学是给不同的东西起相同名字的艺术。</p>
<footer>—— 昂利 · 庞加莱（Henri Poincaré）</footer>
</div>

<div class="article-byline">
<p>第三级 · 计算机图形学 ｜ GAMES101 第3讲 / 虎书 第4、5章 ｜ 2026-08-07</p>
</div>

## 为什么从「矩阵」开始

前面两篇我们一直在用向量描述方向：点积测量、叉积定向、正交基组织坐标。但图形学里真正的工作是**让物体动起来**——平移、旋转、缩放、投影。这些操作能不能统一成一种语言？答案就是**矩阵（matrix）**。

矩阵是图形学的「统一运算器」：一个 3×3 或 4×4 的矩阵，能把一整个物体的所有顶点同时变换到新位置；把多个变换（旋转 → 平移 → 缩放）各自写成矩阵再相乘，就得到一个**复合变换矩阵**，一次作用到所有顶点上。现代 GPU 之所以能每秒处理上亿个三角形，靠的正是「矩阵批量变换顶点」这条流水线。<span class="marginnote">「顶点 × 矩阵」在 GPU 上是高度并行的：每个顶点只做几次乘加，上百万顶点各算各的，互不依赖——这正是并行计算的绝佳载体，也解释了为什么图形渲染如此依赖矩阵。</span>

这一篇我们先不碰复杂的推导，把矩阵乘法怎么算、为什么这样算、以及它如何表示线性变换讲清楚。

## 1 矩阵：一张数字表

**矩阵（matrix）**：按行列排成的矩形数字表。记一个 $m \times n$ 矩阵为

$$
\mathbf{M} = \begin{pmatrix}
m_{11} & m_{12} & \cdots & m_{1n} \\
m_{21} & m_{22} & \cdots & m_{2n} \\
\vdots & \vdots & \ddots & \vdots \\
m_{m1} & m_{m2} & \cdots & m_{mn}
\end{pmatrix}
$$

其中 $m_{ij}$ 表示第 $i$ 行第 $j$ 列的元素。**矩阵相乘**最重要的一条规则是维度匹配：

$$
\underbrace{\mathbf{A}}_{m \times n} \times \underbrace{\mathbf{B}}_{n \times p} = \underbrace{\mathbf{C}}_{m \times p}
$$

**A 的列数必须等于 B 的行数**，结果矩阵的行数取 A 的行数、列数取 B 的列数。

## 2 矩阵乘法：逐项怎么算

设 $\mathbf{A}$ 是 $m \times n$，$\mathbf{B}$ 是 $n \times p$，则结果 $\mathbf{C} = \mathbf{A}\mathbf{B}$ 的第 $i$ 行第 $j$ 列元素是：

$$
c_{ij} = \sum_{k=1}^{n} a_{ik}\, b_{kj}
$$

一句话：**结果第 $i$ 行第 $j$ 列 = A 的第 $i$ 行与 B 的第 $j$ 列对应相乘再相加**。<span class="marginnote">这个「行 × 列」的规则，本质是「A 的每一行与 B 的每一列做点积」。矩阵乘法不是凭空发明的，它是「线性组合」这一几何操作的代数化。</span>

### 矩阵 × 向量：变换的入口

图形学里最常用的特例是矩阵乘列向量。设

$$
\mathbf{M} = \begin{pmatrix}
a & b \\
c & d
\end{pmatrix}, \qquad
\mathbf{v} = \begin{pmatrix} x \\ y \end{pmatrix}
$$

则

$$
\mathbf{M}\mathbf{v} = \begin{pmatrix}
a & b \\
c & d
\end{pmatrix}
\begin{pmatrix} x \\ y \end{pmatrix}
= \begin{pmatrix}
ax + by \\
cx + dy
\end{pmatrix}
$$

**结论：矩阵乘向量 = 对向量做线性变换。** 结果向量是原向量各分量的线性组合，而矩阵的列向量决定了这个变换「把坐标轴送到哪里」。

## 3 公式解析：矩阵乘法为什么这样定义

初学者最容易困惑的是：为什么矩阵乘法偏要「行 × 列」，而不是逐元素相乘？答案是——**为了让复合变换成立**。

### 第一步，从「列向量是坐标轴」看起

**矩阵右乘列向量 $\mathbf{v}$ 等于矩阵各列的线性组合**，系数是 $\mathbf{v}$ 的各个分量：

$$
\mathbf{M}\mathbf{v} = v_1 \begin{pmatrix} m_{11} \\ m_{21} \end{pmatrix} + v_2 \begin{pmatrix} m_{12} \\ m_{22} \end{pmatrix}
$$

即**结果 = 第 1 列 × $v_1$ + 第 2 列 × $v_2$**。这告诉我们：$\mathbf{M}$ 的每一列，恰好是「基向量 $\mathbf{e}_1, \mathbf{e}_2$ 被变换后落到的位置」。知道矩阵列向量，就完全知道这个线性变换干了什么。

### 第二步，复合变换必须「先内层后外层」

假设先做变换 $\mathbf{A}$，再做变换 $\mathbf{B}$，两步合成的效果是：

$$
\mathbf{B}(\mathbf{A}\mathbf{v}) = (\mathbf{B}\mathbf{A})\mathbf{v}
$$

要让「先 A 后 B」等价于「直接乘复合矩阵 $\mathbf{BA}$」，就必须定义 $\mathbf{B}\mathbf{A}$ 的乘积规则为「行 × 列」。用上面的线性组合视角可推出：**矩阵乘法 = 用 B 的行分别与 A 的各列做点积**，这正是 $c_{ij} = \sum_k b_{ik} a_{kj}$ 的由来。

### 第三步，验证：矩阵乘法满足结合律

有了这条定义，复合变换天然满足结合律：

$$
(\mathbf{A}\mathbf{B})\mathbf{C} = \mathbf{A}(\mathbf{B}\mathbf{C})
$$

这保证我们可以先把 A、B 乘在一起得到复合矩阵，再一次性作用到 C，结果与「逐个变换」完全一致——**这就是图形学里把「先旋转再平移」合并成单个矩阵的理论依据**。<span class="marginnote">结合律是矩阵乘法的「好脾气」：它允许我们预先把多个变换矩阵乘成一个，运行时只需一次矩阵乘向量，大幅省掉重复计算。后面齐次坐标篇我们会反复利用这一点。</span>

## 4 矩阵表示线性变换：三个基础例子

**线性变换（linear transformation）**：保持直线与原点、且保持「向量相加与数乘」的变换，数学上写作

$$
T(\mathbf{u} + \mathbf{v}) = T(\mathbf{u}) + T(\mathbf{v}), \qquad T(k\mathbf{u}) = k\,T(\mathbf{u})
$$

**矩阵恰好就是线性变换的坐标表示。** 三个最基础例子：

### 4.1 缩放（Scale）

沿 $x$、$y$ 方向缩放 $s_x$、$s_y$：

$$
\mathbf{S} = \begin{pmatrix}
s_x & 0 \\
0 & s_y
\end{pmatrix}, \qquad
\mathbf{S}\begin{pmatrix}x\\y\end{pmatrix} = \begin{pmatrix}s_x x\\ s_y y\end{pmatrix}
$$

### 4.2 切变（Shear）

保持 $y$ 不变、$x$ 按 $y$ 平移 $a y$：

$$
\mathbf{H} = \begin{pmatrix}
1 & a \\
0 & 1
\end{pmatrix}, \qquad
\mathbf{H}\begin{pmatrix}x\\y\end{pmatrix} = \begin{pmatrix}x + a y\\ y\end{pmatrix}
$$

### 4.3 旋转（Rotation）

绕原点逆时针旋转 $\theta$：

$$
\mathbf{R}(\theta) = \begin{pmatrix}
\cos\theta & -\sin\theta \\
\sin\theta & \cos\theta
\end{pmatrix}
$$

**辨析｜易错点：** 旋转矩阵的 $- \sin\theta$ 在**左上角**还是**右上角**，取决于坐标系与旋转方向。上面的写法对应「逆时针、右手系」；抄公式时务必核对约定，否则旋转方向会反过来。

## 5 矩阵乘法的性质：不可交换

矩阵乘法满足结合律，但**不满足交换律**：

$$
\mathbf{A}\mathbf{B} \neq \mathbf{B}\mathbf{A}
$$

这在图形学里有直接后果：**变换顺序不能颠倒**。先旋转再平移，与先平移再旋转，结果完全不同——想象把一本书先顺时针转 90° 再向右挪，与先向右挪再转 90°，书的位置天差地别。

**辨析｜易错点：** 图形学里复合变换的书写顺序与实际执行顺序**相反**。若先做变换 $\mathbf{A}$ 再做 $\mathbf{B}$，复合矩阵写作 $\mathbf{B}\mathbf{A}$（右边的先作用于向量），顶点变换写成 $\mathbf{B}\mathbf{A}\mathbf{v}$。GAMES101 与虎书均采用「列向量、矩阵在左」的习惯，把矩阵写在向量左边、按从右到左的顺序执行。工程代码里很容易把 $\mathbf{A}$、$\mathbf{B}$ 顺序写反，导致旋转中心偏移——这是最经典的矩阵 bug。

## 6 小结

- **矩阵**是数字表，矩阵乘法要求「A 的列数 = B 的行数」，结果 $c_{ij} = \sum_k a_{ik}b_{kj}$。
- **矩阵乘向量 = 线性变换**；矩阵的列向量是基向量变换后的位置。
- 矩阵乘法定义源于「复合变换」需求，**满足结合律**，可预合并多个变换。
- 三个基础变换：**缩放、切变、旋转**，旋转矩阵 $\begin{pmatrix}\cos\theta & -\sin\theta \\ \sin\theta & \cos\theta\end{pmatrix}$ 绕原点逆时针。
- **矩阵乘法不可交换**，复合变换书写顺序与实际执行相反（$\mathbf{BA}$ 先做 $\mathbf{A}$）。

在下一节，我们将回答两个关键问题：如何把变换「撤销」（逆矩阵）、如何快速判断矩阵是否保长度（转置与正交矩阵）——它们是旋转矩阵与相机变换的数学根基。
