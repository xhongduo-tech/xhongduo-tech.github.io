---
title: 矩阵分解：特征分解与奇异值分解
date: 2026-08-07
---

# 矩阵分解：特征分解与奇异值分解

<div class="epigraph">
<p>计算的目的在于洞察，而不是数字。</p>
<footer>—— 理查德 · 汉明（Richard Hamming）</footer>
</div>

<div class="article-byline">
<p>第四级 · 深度学习 ｜ 花书《深度学习》§2.7–2.8 ｜ 2026-08-07</p>
</div>

## 为什么从矩阵分解开始

上一节我们把矩阵立成了「把向量映射成向量的机器」：一个 $m \times n$ 矩阵 $\boldsymbol{A}$ 就是一个从 $\mathbb{R}^n$ 到 $\mathbb{R}^m$ 的线性映射。但「机器」是个黑箱——你只能看到输入输出，却看不出这台机器内部在干什么。**矩阵分解（matrix decomposition）**就是拆开这台机器的螺丝刀：把一个矩阵写成若干「更简单、更有含义」的矩阵之积，让隐藏在数字里的结构暴露出来。

这个网站的主线是「从极限到大模型」。在**极限**那条线上，我们学会了用导数窥探函数的局部结构；在**线性代数**这条线上，特征分解与奇异值分解（SVD）就是线性映射的「导数」——它们回答同一个问题：**这个对象最重要的方向是哪些？沿这些方向它伸缩多少倍？**<span class="marginnote">后面你几乎处处都会碰到它们：协方差矩阵的特征值决定主成分的方向，Hessian 矩阵的特征值决定最优化是否病态，Transformer 里的低秩近似、表示学习里的嵌入降维，全都站在本节的地基上。</span>

这一节对标花书 §2.7–2.8：先讲**特征分解**（对方阵、对称矩阵），再讲**奇异值分解**（对任意矩阵），最后用一张几何图把 SVD 的直觉钉进脑子。

## 1 特征分解：把方阵拆成「伸缩 × 旋转」

**特征向量与特征值（eigenvector and eigenvalue）**：设 $\boldsymbol{A}$ 是 $n \times n$ 方阵，若存在**非零**向量 $\boldsymbol{v}$ 和标量 $\lambda$，使得

$$
\boldsymbol{A}\boldsymbol{v} = \lambda \boldsymbol{v}
$$

则称 $\boldsymbol{v}$ 是 $\boldsymbol{A}$ 的**特征向量**，$\lambda$ 是对应的**特征值**。这句方程的全部含义是：**矩阵 $\boldsymbol{A}$ 作用在 $\boldsymbol{v}$ 上，只是把它拉长（或压缩、反向）$\lambda$ 倍，方向不变**。特征向量就是「这台线性机器最省事的方向」——沿这些方向，机器的行为退化成一次纯伸缩。

怎么求特征值？把 $\lambda \boldsymbol{v}$ 移到左边并提取公因子：$\boldsymbol{A}\boldsymbol{v} - \lambda \boldsymbol{v} = (\boldsymbol{A} - \lambda \boldsymbol{I})\boldsymbol{v} = \boldsymbol{0}$。要有非零解 $\boldsymbol{v}$，系数矩阵必须不可逆，于是

$$
\det(\boldsymbol{A} - \lambda \boldsymbol{I}) = 0
$$

这称为**特征方程（characteristic equation）**。它是关于 $\lambda$ 的 $n$ 次多项式方程，因此 $n \times n$ 矩阵共有 $n$ 个特征值（计重数）。

**特征分解（eigendecomposition）**：若 $\boldsymbol{A}$ 有 $n$ 个线性无关的特征向量，把它们排成矩阵 $\boldsymbol{V}$（每列是一个特征向量），则

$$
\boldsymbol{A} = \boldsymbol{V}\, \mathrm{diag}(\boldsymbol{\lambda})\, \boldsymbol{V}^{-1}
$$

即「用特征向量作基、在每个方向上独立伸缩」。**最漂亮的情形是实对称矩阵** $\boldsymbol{A} = \boldsymbol{A}^{\top}$：这时可正交对角化

$$
\boldsymbol{A} = \boldsymbol{Q}\boldsymbol{\Lambda}\boldsymbol{Q}^{\top}, \qquad \boldsymbol{Q}^{\top}\boldsymbol{Q} = \boldsymbol{I}
$$

$\boldsymbol{Q}$ 是正交矩阵（列向量彼此正交且单位化），$\boldsymbol{\Lambda}$ 是实对角阵。<span class="marginnote">这是<strong>谱定理（spectral theorem）</strong>：实对称矩阵一定能被正交对角化，且特征值全部为实数。深度学习中出现的协方差矩阵、Hessian 矩阵、Gram 矩阵几乎都是对称的，所以这条定理是整个最优化与表示学习的隐形支柱。</span>几何上，对称矩阵的作用 = 「先按一组正交方向 $\boldsymbol{Q}$ 旋转进特征基，沿各轴伸缩 $\boldsymbol{\Lambda}$，再旋转回原坐标」——**没有剪切，只有正交旋转加纯伸缩**。

**辨析｜易错点：不是所有矩阵都能特征分解。** 第一，特征分解要求 $\boldsymbol{A}$ 有 $n$ 个线性无关的特征向量——存在「缺陷矩阵」（如 $\begin{bmatrix}1&1\\0&1\end{bmatrix}$）只有一条特征方向，不可对角化；第二，实矩阵的特征值可能是复数（如二维旋转矩阵），此时特征向量落在复空间；第三，特征分解不唯一（特征向量可任意缩放，重特征值对应子空间可换基）。这些毛病，奇异值分解统统没有——这正是 SVD 比特征分解更通用的原因。

## 2 奇异值分解：所有矩阵都能拆

**奇异值分解（singular value decomposition，SVD）**：对**任意**实矩阵 $\boldsymbol{A} \in \mathbb{R}^{m \times n}$，都能分解为

$$
\boldsymbol{A} = \boldsymbol{U}\,\boldsymbol{D}\,\boldsymbol{V}^{\top}
$$

其中 $\boldsymbol{U} \in \mathbb{R}^{m \times m}$、$\boldsymbol{V} \in \mathbb{R}^{n \times n}$ 都是正交矩阵，$\boldsymbol{D} \in \mathbb{R}^{m \times n}$ 是**对角矩阵**（只有主对角线可能有非零元）。$\boldsymbol{D}$ 主对角线上的非负元素

$$
\sigma_1 \ge \sigma_2 \ge \cdots \ge \sigma_{\min(m,n)} \ge 0
$$

称为**奇异值（singular values）**；$\boldsymbol{U}$ 的列称为**左奇异向量**，$\boldsymbol{V}$ 的列称为**右奇异向量**。

为什么需要它？特征分解只管「可对角化的方阵」，而现实中我们遇到的几乎都是**矩形矩阵**——一个数据集是「$n$ 个样本 × $d$ 个特征」的表，一张灰度图是「高 × 宽」，都不是方阵。SVD 的存在性最强：**任意实矩阵都有 SVD，且在奇异值降序排列下是唯一的**。<span class="marginnote">SVD 与特征分解的关系：$\boldsymbol{A}^{\top}\boldsymbol{A}$ 的特征向量恰是右奇异向量 $\boldsymbol{V}$，$\boldsymbol{A}\boldsymbol{A}^{\top}$ 的特征向量恰是左奇异向量 $\boldsymbol{U}$，而奇异值满足 $\sigma_i = \sqrt{\lambda_i(\boldsymbol{A}^{\top}\boldsymbol{A})}$。用一条式子把两者接起来：$\boldsymbol{A}^{\top}\boldsymbol{A} = \boldsymbol{V}\boldsymbol{D}^{\top}\boldsymbol{D}\boldsymbol{V}^{\top}$。</span>

奇异值还直接给出矩阵的**秩**：$\text{rank}(\boldsymbol{A})$ 等于非零奇异值的个数。更妙的是，**截断 SVD**——只保留前 $k$ 大的奇异值及其对应的奇异向量——给出的是在 Frobenius 范数意义下对 $\boldsymbol{A}$ 最优的秩 $k$ 逼近（Eckart–Young 定理）。这意味着 SVD 天然是**降维与压缩**的工具，主成分分析（PCA）正是它的一种应用。

**辨析｜易错点：奇异值不是特征值的「同义词」。** 特征值可正可负可为复数，奇异值**非负且降序排列**。对一般的非对称矩阵，两者毫无直接对应；即便对实对称矩阵，也只有在特征值全部非负（半正定）时它们才相等——否则奇异值等于特征值的**绝对值**。例如特征值 $\{-1, 3\}$ 的矩阵，其奇异值是 $\{3, 1\}$。后面做数值计算、看谱半径时，先想清楚你手里的是特征值还是奇异值。

## 3 几何直觉：单位圆被拉成椭圆

SVD 的代数定义可以背，但要真正「看见」它，最好的方式是一张图。考虑一个 $2 \times 2$ 的可逆矩阵 $\boldsymbol{A}$，把它作用在**单位圆**上（单位圆上的每一点都代表一个单位长度向量）。由于 $\boldsymbol{A}$ 是线性映射，单位圆被映射成一条**以原点为中心的椭圆**。

![奇异值分解的几何意义：单位圆经 A 作用后变成椭圆](/images/deep-learning/matrix-decomposition-1.svg)

这张图说清楚了 SVD 的每一块部件：

- **右奇异向量 $\boldsymbol{v}_1, \boldsymbol{v}_2$** 是输入侧单位圆上**原本正交的两个方向**；
- $\boldsymbol{A}$ 把 $\boldsymbol{v}_1$ 映射到 $\sigma_1\boldsymbol{u}_1$、把 $\boldsymbol{v}_2$ 映射到 $\sigma_2\boldsymbol{u}_2$；
- **左奇异向量 $\boldsymbol{u}_1, \boldsymbol{u}_2$** 是输出椭圆**两个半轴的方向**；
- **奇异值 $\sigma_1, \sigma_2$** 正是椭圆**两个半轴的长度**。

于是 $\boldsymbol{A} = \boldsymbol{U}\boldsymbol{D}\boldsymbol{V}^{\top}$ 读成三步动作：**先按 $\boldsymbol{V}^{\top}$ 旋转，再沿坐标轴按 $\sigma$ 伸缩，最后按 $\boldsymbol{U}$ 旋转**——「旋转—伸缩—旋转」。对长方形矩阵或降秩矩阵，单位球会被压成更低维的椭球、线段甚至一个点，但「正交旋转 × 沿轴伸缩」的解读不变。

## 4 公式解析：$\boldsymbol{A} = \boldsymbol{U}\boldsymbol{D}\boldsymbol{V}^{\top}$ 的逐步拆解

SVD 是全篇最重要的一条公式，逐符号拆开：

$$
\underset{m \times n}{\boldsymbol{A}} = \underset{m \times m}{\boldsymbol{U}} \;\;
\underset{m \times n}{\boldsymbol{D}} \;\;
\underset{n \times n}{\boldsymbol{V}^{\top}}
$$

- **第一步，读形状**。$\boldsymbol{U}$ 与 $\boldsymbol{V}$ 都是**方阵**且正交（$\boldsymbol{U}^{\top}\boldsymbol{U}=\boldsymbol{I}$，$\boldsymbol{V}^{\top}\boldsymbol{V}=\boldsymbol{I}$），真正「瘦长」的是 $\boldsymbol{D}$。正交矩阵只做旋转/反射、不改变长度，因此**全部伸缩信息都集中在 $\boldsymbol{D}$ 上**。
- **第二步，看 $\boldsymbol{V}^{\top}$ 的动作**。$\boldsymbol{V}^{\top}$ 把输入向量从标准基旋到右奇异向量基 $\{\boldsymbol{v}_i\}$：在 $\boldsymbol{A}$ 眼里，$\boldsymbol{v}_i$ 方向是「天然坐标轴」。这正是「先旋转」的数学来源。
- **第三步，看 $\boldsymbol{D}$ 的伸缩**。在奇异向量基下，$\boldsymbol{A}$ 的行为退化为对角阵——沿第 $i$ 个方向乘 $\sigma_i$。若 $m>n$，$\boldsymbol{D}$ 底部补零行（丢进高维）；若 $m<n$，$\boldsymbol{D}$ 右侧补零列（零空间方向被压掉）。
- **第四步，看 $\boldsymbol{U}$ 收尾**。伸缩完的结果还在奇异向量基里，$\boldsymbol{U}$ 把它旋回输出空间的标准基，得到最终向量。

用一个数字例子落一遍。设

$$
\boldsymbol{A} = \begin{bmatrix} 1 & 2 \\ 2 & 1 \end{bmatrix}
$$

它的特征值是 $3$ 与 $-1$。对 $\boldsymbol{A}^{\top}\boldsymbol{A}$ 求特征值得 $9$ 与 $1$，故奇异值为 $\sigma_1 = 3$、$\sigma_2 = 1$。对应地

$$
\boldsymbol{U} = \frac{1}{\sqrt{2}}\begin{bmatrix} 1 & -1 \\ 1 & 1 \end{bmatrix}, \quad
\boldsymbol{D} = \begin{bmatrix} 3 & 0 \\ 0 & 1 \end{bmatrix}, \quad
\boldsymbol{V}^{\top} = \frac{1}{\sqrt{2}}\begin{bmatrix} 1 & 1 \\ 1 & -1 \end{bmatrix}
$$

请留意**特征值 $-1$ 变成了奇异值 $1$**：特征值保留了「反向」这个方向信息，SVD 则把方向信息全部交给 $\boldsymbol{U}$ 里的符号，奇异值只负责报「伸缩多少」。这个例子同时印证了上一节的辨析。

## 5 小结

- **特征分解** $\boldsymbol{A}=\boldsymbol{V}\,\mathrm{diag}(\boldsymbol{\lambda})\,\boldsymbol{V}^{-1}$：沿特征向量方向纯伸缩；只对可对角化的方阵成立，实对称矩阵可正交对角化 $\boldsymbol{A}=\boldsymbol{Q}\boldsymbol{\Lambda}\boldsymbol{Q}^{\top}$。
- **奇异值分解** $\boldsymbol{A}=\boldsymbol{U}\boldsymbol{D}\boldsymbol{V}^{\top}$：对任意实矩阵成立，$\boldsymbol{U},\boldsymbol{V}$ 正交、$\boldsymbol{D}$ 对角，奇异值非负且降序。
- **几何**：$\boldsymbol{A}$ 把单位圆（球）变成椭圆（椭球），半轴方向是左奇异向量、半轴长度是奇异值；动作是「旋转—伸缩—旋转」。
- **奇异值 ≠ 特征值**：奇异值非负，特征值可负可复；对半正定对称矩阵二者才一致。
- **秩与压缩**：$\text{rank}(\boldsymbol{A})$ = 非零奇异值个数；截断 SVD 是最优低秩逼近，是 PCA 与降维的数学基础。

在下一节，我们将补全线性代数的最后一组工具：**范数、迹运算与伪逆**——它们分别负责「量大小」「求和号」「解最小二乘」，是紧接着要用的计量与求解语言。
