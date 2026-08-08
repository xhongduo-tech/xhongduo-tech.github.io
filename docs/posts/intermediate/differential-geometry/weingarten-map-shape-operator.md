---
title: Weingarten 映射与形状算子
date: 2026-08-07
---

# Weingarten 映射与形状算子

<div class="epigraph">
<p>一个线性算子就足够描述曲面的弯曲：它的特征值与特征方向，就是曲率与曲率方向。</p>
<footer>—— 尤利乌斯 · 魏因加滕（Julius Weingarten）</footer>
</div>

<div class="article-byline">
<p>第二级 · 微分几何 ｜ 陈维桓《微分几何》§3.3 ｜ 2026-08-07</p>
</div>

## 为什么从形状算子开始

第二基本形式告诉我们「曲面怎么弯」，但它是一个**双线性形式**，还要配内积才能提取方向信息。现在我们要换一个更犀利的视角：把弯曲编码成一个**线性算子**。

这个算子就是**形状算子（shape operator）**，也叫 **Weingarten 映射**。它就是高斯映射的微分（取负号）：

$$
S_p = -dN_p: T_pS \longrightarrow T_pS
$$

为什么「负号」？约定而已，让凸曲面（球面）的特征值为正。为什么它值得单独一个名字？因为它是切平面上的**自伴随线性算子**，而自伴随算子在第二级《线性代数》里有一整套成熟理论：**特征值、特征向量、对角化**。于是「曲面的弯曲」被完美地翻译成「对称矩阵的特征值问题」——这是整门曲面曲率论的制高点。<span class="marginnote">形状算子是「用线性代数武装几何」的教科书案例：高斯花了三十年才把曲面曲率彻底想清楚，而现代框架里它不过是一个对称算子的特征值。你把球面、柱面、马鞍面的形状算子写出来，一切曲率指标（主曲率、高斯曲率、平均曲率）都是它的特征值与不变量。</span>

## 1 形状算子的定义

**定义（形状算子 / Weingarten 映射）**：设 $S$ 是带单位法场 $\mathbf{N}$ 的正则曲面，$p\in S$。形状算子为

$$
S_p = -dN_p: T_pS \longrightarrow T_pS
$$

对切向量 $v$，$S_p(v) = -dN_p(v)$ 是「沿 $v$ 方向法向变化率的负值」。

**重点：$S_p$ 是「切平面到自身」的线性映射，且是自伴随的（对称的）**：对任意 $v,w\in T_pS$，

$$
\langle S_p(v), w\rangle = \langle v, S_p(w)\rangle
$$

自伴随性由内积的对称性与 $\mathbf{N}$ 的单位性保证（正是上一节 $\mathbf{x}_u\cdot\mathbf{N}=0$ 求导技巧的抽象化）。<span class="marginnote">自伴随（对称）意味着：形状算子可以用正交基对角化，且特征值都是实数。对称矩阵的谱定理（第二级《线性代数》）在这里落地：曲率方向互相垂直，曲率都是实数值——这是「弯曲」能够被干净地分解成「主方向」的数学根源。</span>

## 2 形状算子的矩阵：Weingarten 方程

在坐标卡 $\mathbf{x}(u,v)$ 下，形状算子如何写成矩阵？设

$$
S_p(\mathbf{x}_u) = a\,\mathbf{x}_u + b\,\mathbf{x}_v, \qquad S_p(\mathbf{x}_v) = c\,\mathbf{x}_u + d\,\mathbf{x}_v
$$

系数 $a,b,c,d$ 构成矩阵 $[S_p]$。它可以通过两个基本形式的系数算出。由自伴随性 $\langle S(\mathbf{x}_u),\mathbf{x}_u\rangle = L$ 等，解出：

$$
[S_p] = \begin{pmatrix} E & F \\ F & G \end{pmatrix}^{-1} \begin{pmatrix} L & M \\ M & N \end{pmatrix} = \frac{1}{EG-F^2}\begin{pmatrix} GL - FM & GM - FN \\ EM - FL & EN - FM \end{pmatrix}
$$

这条式子叫 **Weingarten 方程**——它把形状算子（几何对象）用 $E,F,G,L,M,N$（坐标数据）显式写出来。<span class="marginnote">矩阵形式 $[S] = \mathcal{I}^{-1}\,\mathcal{II}$（第一基本形式矩阵的逆乘第二基本形式矩阵）是曲面论最优雅的公式之一。它说：形状算子 = 「用度量把第二基本形式变成算子」。这正是一般黎曼几何里「把协变张量升指标」在二维的版本。</span>

**辨析｜易错点：** $[S_p]$ 依赖坐标基，但它作为算子的特征值与特征方向不依赖。**别把「矩阵」与「算子」混为一谈**——换坐标，$[S]$ 相似变换，特征值不变。

## 3 公式解析：Weingarten 方程 $[S] = \mathcal{I}^{-1}\mathcal{II}$

这条式子值得逐层拆透：

- **第一步，两个基本形式的关系**：由定义 $II(v,w) = \langle S(v), w\rangle$。设 $[S]$ 是 $S$ 在坐标基下的矩阵，则 $S(\mathbf{x}_u)$ 的坐标向量是 $[S]$ 的第一列。用矩阵乘法：
  $$
  II(\mathbf{x}_u,\mathbf{x}_v) = \langle S(\mathbf{x}_u), \mathbf{x}_v\rangle = \big(\mathcal{I}\,[S]\big)_{12} = M
  $$
  第一基本形式矩阵 $\mathcal{I}$ 扮演「度量」，把 $[S]$ 的列「内积」成 $II$ 的分量。
- **第二步，解出 $[S]$**：把上面等式全部写开，得 $\mathcal{I}\,[S] = \mathcal{II}$，即
  $$
  [S] = \mathcal{I}^{-1}\,\mathcal{II}
  $$
- **第三步，展开计算**：代入 $\mathcal{I} = \begin{pmatrix}E&F\\F&G\end{pmatrix}$ 的逆矩阵 $\frac{1}{EG-F^2}\begin{pmatrix}G&-F\\-F&E\end{pmatrix}$ 与 $\mathcal{II} = \begin{pmatrix}L&M\\M&N\end{pmatrix}$，相乘即得上式。

**重点：$EG-F^2$ 作为分母出现，正则性（$EG-F^2>0$）保证 $[S]$ 良定义。** 与非正则点（$EG-F^2=0$）形状算子无定义——那里切平面塌缩，弯曲无从谈起。

## 4 形状算子的三个典型

用三个原型检验形状算子的意义：

- **平面**：$L=M=N=0$，$[S] = \mathbf{0}$。形状算子是零算子——不弯。
- **球面 $S^2_R$**：$L=N=1/R$（取外法向）、$M=0$、$E=G=R^2$、$F=0$，于是
  $$
  [S] = \frac{1}{R}I_2 = \begin{pmatrix}1/R & 0\\0&1/R\end{pmatrix}
  $$
  形状算子是**常数倍单位矩阵**——所有方向弯曲相同，特征值全为 $1/R$。<span class="marginnote">球面的形状算子「在各向同性」（isotropic）：无论从哪个方向看，弯曲都一样。这是球面最本质的几何属性，也是「常曲率空间」的第一个例子（第八篇会推广到任意维）。反过来说，「形状算子是单位矩阵的倍数」就刻画了球面（或平面）——这是「全脐曲面」分类定理的前奏。</span>
**圆柱面**：$[S] = \begin{pmatrix}1&0\\0&0\end{pmatrix}$（在合适的正交基下）。特征值 $1$ 与 $0$——一个方向弯（环向）、一个方向平（轴向），特征向量互相垂直。

**重点：特征值 = 该方向上的弯曲量，特征向量 = 弯曲方向。** 形状算子的谱理论由此展开：两个特征值就是**主曲率**，特征方向就是**主方向**——这是下一节的精确内容。

## 5 形状算子：曲面曲率的「总控台」

形状算子的伟大在于：**一个线性算子，囊括了曲面的全部弯曲信息。** 从它身上可以提取出所有曲率指标：

**特征值** $\kappa_1,\kappa_2$：主曲率。
**特征向量**：主方向（两两正交）。
**迹（平均曲率）**：$H = \tfrac{1}{2}\operatorname{tr}(S) = \tfrac{1}{2}(\kappa_1+\kappa_2)$。
**行列式（高斯曲率）**：$K = \det(S) = \kappa_1\kappa_2$。
**法曲率**：$k_n(v) = \langle S(v), v\rangle/\langle v,v\rangle$（归一化的 $II$）。

后面的章节几乎都在围绕「形状算子的各种投影」展开。而它最大的戏剧性在于：**迹与行列式**作为特征值多项式的不变量，在坐标变换下稳定——其中高斯曲率 $K$ 甚至不依赖外蕴嵌入（Gauss 绝妙定理），这将成为第四篇的高潮。<span class="marginnote">平均曲率与高斯曲率是形状算子的「两个基本不变量」。$H$ 描述「总体弯曲程度」（极小曲面的 $H=0$），$K$ 描述「弯曲的内在本质」（不可嵌入变形改变）。肥皂膜（$H=0$）与橘子皮（$K>0$）是两者的直观对照——一个「外弯内凹抵消」，一个「处处同向弯」。</span>

### 例：用 Weingarten 方程算圆柱面的形状算子

把 $[S] = \mathcal{I}^{-1}\mathcal{II}$ 的流程完整走一遍。圆柱面 $\mathbf{x}(u,v) = (\cos u,\ \sin u,\ v)$：

- **第一基本形式**：$\mathbf{x}_u = (-\sin u,\cos u,0)$、$\mathbf{x}_v = (0,0,1)$，故 $E=1,\ F=0,\ G=1$。
- **第二基本形式**：$\mathbf{x}_{uu} = (-\cos u,-\sin u,0)$，法向 $\mathbf{N} = \mathbf{x}_u\times\mathbf{x}_v = (\cos u,\sin u,0)$，故 $L = \mathbf{x}_{uu}\cdot\mathbf{N} = -1$（或取外法向得 $+1$）；$\mathbf{x}_{uv} = 0$、$\mathbf{x}_{vv} = 0$，故 $M=N=0$。
- **形状算子**：$[S] = \begin{pmatrix}1&0\\0&0\end{pmatrix}^{-1}\begin{pmatrix}1&0\\0&0\end{pmatrix} = \begin{pmatrix}1&0\\0&0\end{pmatrix}$。

**重点：形状算子的特征值 $\{1, 0\}$ 正是主曲率——一个方向弯（环向）、一个方向平（轴向）。** 整套计算只需两个基本形式的系数，机械代入——「$[S] = \mathcal{I}^{-1}\mathcal{II}$」是形状算子的「通用计算器」。

## 6 小结

- **形状算子** $S_p = -dN_p: T_pS \to T_pS$：高斯映射微分的负号版，编码全部弯曲信息。
- 它是**自伴随算子**，可正交对角化，特征值全实。
- **Weingarten 方程**：$[S] = \mathcal{I}^{-1}\mathcal{II}$，用 $E,F,G,L,M,N$ 显式给出矩阵。
- 原型：平面 $[S]=0$、球面 $[S]=I/R$（各向同性）、圆柱 $[S]=\mathrm{diag}(1,0)$。
- 特征值 = 主曲率，特征向量 = 主方向；$H=\mathrm{tr}(S)/2$，$K=\det(S)$。

在下一节，我们把形状算子的特征值正式命名为**法曲率**——先理解「沿任意方向的弯曲」，再进而提取主曲率与主方向。
