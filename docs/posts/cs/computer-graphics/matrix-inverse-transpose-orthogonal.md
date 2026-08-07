---
title: 矩阵：逆矩阵、转置与正交矩阵
date: 2026-08-07
---

# 矩阵：逆矩阵、转置与正交矩阵

<div class="epigraph">
<p>数学是给不同的东西起相同名字的艺术。</p>
<footer>—— 昂利 · 庞加莱（Henri Poincaré）</footer>
</div>

<div class="article-byline">
<p>第三级 · 计算机图形学 ｜ GAMES101 第3讲 / 虎书 第4、5章 ｜ 2026-08-07</p>
</div>

## 为什么从「逆矩阵、转置与正交矩阵」开始

上一篇我们学会了用矩阵表示变换。但变换经常需要被**撤销**：相机要能「回到原点」，把物体从世界坐标搬进相机坐标再搬回来，点光源要反查「这个点从哪来」。撤销一个变换的工具，就是**逆矩阵（inverse matrix）**。

而**转置（transpose）**与**正交矩阵（orthogonal matrix）**则回答另一个问题：**哪些变换不改变长度与夹角？** 答案是旋转与刚体运动——它们对应的矩阵列向量互相垂直且长度 1，其转置恰好等于逆矩阵。这个性质让 GPU 计算逆矩阵时只需转置，快一个数量级。此外，**法向量**的变换不能直接套用几何变换矩阵，需要用「逆转置矩阵」——这是图形学里一个反直觉却极其关键的细节。

这一篇我们把这三个概念讲透，并为下一篇「2D/3D 变换」备好全部弹药。

## 1 逆矩阵：撤销变换

**逆矩阵（inverse matrix）**：对方阵 $\mathbf{M}$，若存在矩阵 $\mathbf{M}^{-1}$ 使

$$
\mathbf{M}\mathbf{M}^{-1} = \mathbf{M}^{-1}\mathbf{M} = \mathbf{I}
$$

则称 $\mathbf{M}$ 可逆，$\mathbf{M}^{-1}$ 是 $\mathbf{M}$ 的逆。其中 $\mathbf{I}$ 是**单位矩阵**（对角线为 1，其余为 0），它乘任何向量都不改变向量：

$$
\mathbf{I}\mathbf{v} = \mathbf{v}
$$

**逆矩阵的几何意义就是「撤销变换」**：$\mathbf{M}$ 把向量搬到新位置，$\mathbf{M}^{-1}$ 把它搬回原位。物理直觉：如果 $\mathbf{M}$ 表示「向右平移 3 个单位」的旋转+平移复合，那么 $\mathbf{M}^{-1}$ 就是「先反向旋转、再向左平移 3 个单位」。

### 逆矩阵怎么求

对 $2 \times 2$ 矩阵有直接公式：

$$
\mathbf{M} = \begin{pmatrix} a & b \\ c & d \end{pmatrix}, \qquad
\mathbf{M}^{-1} = \frac{1}{ad - bc} \begin{pmatrix} d & -b \\ -c & a \end{pmatrix}
$$

其中分母 $ad - bc$ 正是 $\mathbf{M}$ 的**行列式（determinant）**。

**辨析｜易错点：** 若行列式 $ad - bc = 0$，矩阵**不可逆**——它把向量压缩到了更低维（把平面压成一条线或一个点），信息被丢弃，无法还原。图形学里投影矩阵常常不可逆，因为投影本就有损。计算时先检查行列式是否接近 0，是数值稳健的第一步。<span class="marginnote">「行列式为零 ↔ 不可逆 ↔ 压缩到低维」三者等价。直观：平行四边形的面积被矩阵缩成 0，两个不同向量被映射到同一点，逆向无法区分它们——正如相机透视投影会把远处两点拍到同一像素，无法反推原坐标。</span>

## 2 转置：翻转行列

**转置（transpose）**：把矩阵的行与列互换，记作 $\mathbf{M}^T$：

$$
\mathbf{M} = \begin{pmatrix} a & b \\ c & d \end{pmatrix}, \qquad
\mathbf{M}^T = \begin{pmatrix} a & c \\ b & d \end{pmatrix}
$$

转置的几个基本性质：

- $(\mathbf{M}^T)^T = \mathbf{M}$
- $(\mathbf{A}\mathbf{B})^T = \mathbf{B}^T \mathbf{A}^T$——**乘积的转置 = 转置的反序乘积**，注意顺序颠倒
- $(\mathbf{M}^{-1})^T = (\mathbf{M}^T)^{-1}$——逆与转置可交换

**辨析｜易错点：** 乘积转置的**反序**（$\mathbf{B}^T\mathbf{A}^T$ 而非 $\mathbf{A}^T\mathbf{B}^T$）是最容易被记反的性质。它和「矩阵乘法不可交换」同源：转置把运算方向翻转，必须配套翻转顺序。

## 3 正交矩阵：转置即逆

**正交矩阵（orthogonal matrix）**：方阵 $\mathbf{Q}$ 满足

$$
\mathbf{Q}^T \mathbf{Q} = \mathbf{I}
$$

等价地，$\mathbf{Q}^{-1} = \mathbf{Q}^T$——**求逆变成了转置**。这条性质在 GPU 上极其值钱：转置只需交换内存下标，远比求逆（要解线性方程组）便宜。

**正交矩阵为什么特别？** 它的**列向量构成一组正交基**（两两垂直、长度 1）。几何上，正交矩阵对应的线性变换**保长度、保夹角**——只旋转、镜像，不拉伸、不压缩。

$$
|\mathbf{Q}\mathbf{v}| = |\mathbf{v}|, \qquad
(\mathbf{Q}\mathbf{u}) \cdot (\mathbf{Q}\mathbf{v}) = \mathbf{u} \cdot \mathbf{v}
$$

**辨析｜易错点：** 判断矩阵是否正交，只需验证 $\mathbf{Q}^T\mathbf{Q} = \mathbf{I}$。**旋转矩阵是正交矩阵，但正交矩阵未必是旋转**——它也可能包含镜像（行列式为 $-1$ 而非 $+1$）。真实物体变换不会用镜像（会把左右手翻转、物体变成「内部翻转」），所以旋转矩阵要求行列式为 $+1$。镜像操作会让物体坐标系变为左手系——这与你之前学过的左右手坐标体系直接呼应。

## 4 公式解析：正交矩阵为何保长度

为什么 $\mathbf{Q}^T\mathbf{Q} = \mathbf{I}$ 就足以保证保长度？逐项拆解：

**第一步，长度的平方写作点积。** 向量长度平方 $|\mathbf{v}|^2 = \mathbf{v}\cdot\mathbf{v} = \mathbf{v}^T\mathbf{v}$（转置后点积）。

**第二步，把变换代入。** 变换后长度的平方：

$$
|\mathbf{Q}\mathbf{v}|^2 = (\mathbf{Q}\mathbf{v})^T(\mathbf{Q}\mathbf{v}) = \mathbf{v}^T \mathbf{Q}^T \mathbf{Q} \, \mathbf{v}
$$

**第三步，用正交性化简。** 由 $\mathbf{Q}^T\mathbf{Q} = \mathbf{I}$：

$$
|\mathbf{Q}\mathbf{v}|^2 = \mathbf{v}^T \mathbf{I} \, \mathbf{v} = \mathbf{v}^T\mathbf{v} = |\mathbf{v}|^2
$$

**证毕。** 长度平方在变换前后相等，故 $\mathbf{Q}$ 保长度；同理可证它保夹角（用 $\mathbf{u}^T \mathbf{Q}^T \mathbf{Q} \mathbf{v} = \mathbf{u}^T\mathbf{v}$）。<span class="marginnote">这一推导里最关键的技巧是把「内积」写成 $\mathbf{v}^T\mathbf{v}$，让矩阵进入表达式——这是线性代数「把几何翻译成代数」的标准动作。你会在第四级机器学习的范数、正交基、PCA 里反复见到它。</span>

## 5 法向量的变换：逆转置矩阵

图形学里最反直觉的矩阵细节：**几何变换矩阵 $\mathbf{M}$ 不能直接用来变换法向量。**

表面切向量（在平面内的向量） $\mathbf{t}$ 用 $\mathbf{M}$ 变换即可：$\mathbf{t}' = \mathbf{M}\mathbf{t}$。但法向量 $\mathbf{n}$ 垂直于切平面，若直接套 $\mathbf{n}' = \mathbf{M}\mathbf{n}$，**非均匀缩放或切变后法向量会不再垂直于表面**。<span class="marginnote">直觉：非均匀缩放把球拉成椭球，缩放前球面的法向（径向）在缩放后不再指向椭球表面法向。法向量与切向量是「对偶」关系，变换规则不同。</span>

正确做法是用法向量的**逆转置矩阵**：

$$
\mathbf{n}' = (\mathbf{M}^{-1})^T \mathbf{n}
$$

**为什么是逆转置？** 法向量定义是「与所有切向量点积为 0」：$\mathbf{n}^T \mathbf{t} = 0$。变换后要求 $\mathbf{n}'^T \mathbf{t}' = 0$：

$$
(\mathbf{n}')^T (\mathbf{M}\mathbf{t}) = 0
$$

想让此式恒成立，需要 $(\mathbf{n}')^T \mathbf{M} = 0$，即 $\mathbf{n}' = (\mathbf{M}^{-1})^T \mathbf{n}$——**逆取转置恰好让「与切向量的垂直关系」在新坐标系下保持**。

**辨析｜易错点：** 若 $\mathbf{M}$ 是**正交矩阵**（纯旋转），则 $(\mathbf{M}^{-1})^T = (\mathbf{M}^T)^T = \mathbf{M}$——**法向量直接套用 $\mathbf{M}$ 没问题**。只有当变换含非均匀缩放/切变时，才必须用逆转置。所以「法向量用逆转置」是普适安全写法，但旋转场景下转置后仍等于原矩阵，殊途同归。

## 6 小结

- **逆矩阵**撤销变换，$\mathbf{M}\mathbf{M}^{-1} = \mathbf{I}$；$2\times2$ 逆用 $\frac{1}{ad-bc}$，行列式为 0 则不可逆。
- **转置**翻转行列，$(\mathbf{A}\mathbf{B})^T = \mathbf{B}^T\mathbf{A}^T$，注意反序。
- **正交矩阵** $\mathbf{Q}^T\mathbf{Q} = \mathbf{I}$，逆等于转置，列向量是正交基；保长度、保夹角，但可能含镜像（行列式 $-1$）。
- 正交矩阵保长度证明：$|\mathbf{Q}\mathbf{v}|^2 = \mathbf{v}^T\mathbf{Q}^T\mathbf{Q}\mathbf{v} = \mathbf{v}^T\mathbf{v}$。
- **法向量变换用逆转置** $(\mathbf{M}^{-1})^T$；纯旋转时逆转置等于原矩阵，可直用。

在下一节，我们将把这些矩阵工具组合起来，进入真正的变换世界：**2D 基础变换**（缩放、切变、旋转）——为紧接着的齐次坐标与 3D 变换打好地基。
