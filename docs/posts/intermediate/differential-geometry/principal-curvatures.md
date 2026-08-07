---
title: 主曲率与主方向
date: 2026-08-07
---

# 主曲率与主方向

<div class="epigraph">
<p>任何弯曲的曲面，在每一点都有一对最弯与最平的方向，正交排列，成为曲率的骨架。</p>
<footer>—— 莱昂哈德 · 欧拉（Leonhard Euler）</footer>
</div>

<div class="article-byline">
<p>第二级 · 微分几何 ｜ 陈维桓《微分几何》§3.6 ｜ 2026-08-07</p>
</div>

## 为什么从主曲率开始

法曲率告诉我们「每个方向的弯曲」，但方向有无穷多个，信息太多。能不能压缩成少数几个「代表性」的量？答案藏在形状算子的谱里：**主曲率（principal curvatures）就是法曲率在所有方向中的最大值与最小值，主方向（principal directions）就是达到这些极值的方向。**

这一节是线性代数与几何的完美联姻：形状算子 $S_p$ 是自伴随算子，谱定理保证它有两个实特征值 $\kappa_1, \kappa_2$（主曲率）与两个正交的特征方向（主方向）。**曲面每一点的「弯曲全貌」被压缩成两个数与两个方向**——就像把一张弯曲图压缩成「最陡/最平」两个梯度方向。<span class="marginnote">主曲率思想由 Euler 在 1760 年奠基：他证明了存在两个互相垂直的方向，沿它们的法曲率取极值。到 19 世纪，Rodrigues 发现这两个方向正是「法向导数沿该方向保持共线」的方向——与形状算子的特征方向完全吻合。谱定理（第二级《线性代数》）是这一切的现代骨架。</span>

## 1 主曲率与主方向的定义

**定义（主曲率 / 主方向）**：设 $S$ 是带单位法场的正则曲面，$p\in S$，$S_p$ 是形状算子。$S_p$ 的两个实特征值 $\kappa_1 \ge \kappa_2$ 称为 $S$ 在 $p$ 处的**主曲率**；对应的特征方向称为**主方向（principal directions）**。

因为 $S_p$ 自伴随，两个主方向**正交**（若 $\kappa_1 \neq \kappa_2$；$\kappa_1 = \kappa_2$ 时每个方向都是主方向，称 $p$ 为**脐点**，后面单独讨论）。

**重点：主曲率 = 法曲率的极值。** 由上一节的 Euler 分解，在正交主方向基 $\{\mathbf{e}_1,\mathbf{e}_2\}$ 下，

$$
k_n(\theta) = \kappa_1\cos^2\theta + \kappa_2\sin^2\theta
$$

当 $\theta = 0$（沿 $\mathbf{e}_1$）时 $k_n = \kappa_1$（最大），$\theta = \pi/2$（沿 $\mathbf{e}_2$）时 $k_n = \kappa_2$（最小）。**在所有方向中，主方向取到法曲率的极值。**

## 2 主曲率的计算：特征值方程

主曲率由形状算子的特征值给出。特征方程

$$
\det(S_p - \kappa\,I) = 0
$$

展开为二次方程：

$$
\kappa^2 - \operatorname{tr}(S_p)\,\kappa + \det(S_p) = 0
$$

于是

$$
\kappa_{1,2} = \frac{H \pm \sqrt{H^2 - K}}{1}\Big/\ \text{归一化} \quad\Longleftrightarrow\quad
\kappa_{1,2} = H \pm \sqrt{H^2 - K}
$$

其中 $H = \tfrac{1}{2}\mathrm{tr}(S_p)$ 是平均曲率、$K = \det(S_p)$ 是高斯曲率（定义见下一节）。这个式子把主曲率完全用 $H$ 与 $K$ 表达——**知道了 $H$ 和 $K$，主曲率唾手可得**。

在坐标卡下直接用 Weingarten 方程：

$$
\kappa_{1,2} = \frac{(GL + EN - 2FM) \pm \sqrt{(GL + EN - 2FM)^2 - 4(EG-F^2)(LN-M^2)}}{2(EG-F^2)}
$$

（这条式子不必死记，会用 $[S] = \mathcal{I}^{-1}\mathcal{II}$ 求特征值即可。）<span class="marginnote">计算策略：先算 $\mathcal{I} = \begin{pmatrix}E&F\\F&G\end{pmatrix}$ 与 $\mathcal{II}=\begin{pmatrix}L&M\\M&N\end{pmatrix}$，再求 $\mathcal{I}^{-1}\mathcal{II}$ 的特征值。若参数化正交（$F=0$），$[S] = \begin{pmatrix}L/E & M/G \\ M/E & N/G\end{pmatrix}$ 之类，计算大幅简化——「正交坐标」再一次立功。</span>

## 3 例：三个原型的主曲率

- **平面**：$[S] = \mathbf{0}$，$\kappa_1 = \kappa_2 = 0$。主方向任意（脐点）。
- **球面 $S^2_R$**：$[S] = \tfrac{1}{R}I$，$\kappa_1 = \kappa_2 = 1/R$。所有方向都主方向（全脐点）。
- **圆柱面**：$\kappa_1 = 1$（环向，最弯）、$\kappa_2 = 0$（轴向，不弯）。主方向正交：环向 vs 轴向。<span class="marginnote">把三个原型的主曲率记住：$(0,0)$、$(1/R,1/R)$、$(1,0)$。它们分别对应「平 / 球 / 柱」三种基本形状。任何曲面的局部形状都是这三者的组合——通过 $H$ 与 $K$ 的符号（下一节）判定「哪像球、哪像柱、哪像马鞍」。</span>

### 例：马鞍面 $z = xy$ 在原点

$f(x,y) = xy$，原点处 $f_x = f_y = 0$（驻点），$\mathbf{x}_u = (1,0,y)$、$\mathbf{x}_v=(0,1,x)$，在原点：

$$
E = G = 1,\ F = 0;\qquad L = N = 0,\ M = 1
$$

（$\mathbf{x}_{uv} = (0,0,1)$ 与法向 $(-f_x,-f_y,1)/\|\cdot\|$ 内积得 $M=1$。）于是

$$
[S] = \begin{pmatrix}0 & 1 \\ 1 & 0\end{pmatrix},\qquad \kappa_1 = 1,\ \kappa_2 = -1
$$

主方向是 $45^\circ$ 与 $-45^\circ$——正是「上坡脊」与「下坡谷」的方向。**一个方向弯上、一个方向弯下，主曲率一正一负。**

## 4 公式解析：$k_n(\theta) = \kappa_1\cos^2\theta + \kappa_2\sin^2\theta$（Euler 公式）

这条式子是主曲率的「用途」，逐项拆：

- **第一步，在主轴基下分解方向**：$v = \cos\theta\,\mathbf{e}_1 + \sin\theta\,\mathbf{e}_2$，$\theta$ 是 $v$ 与第一主方向的夹角。
- **第二步，代入法曲率**：$k_n(v) = \langle S(v), v\rangle$。用 $S(\mathbf{e}_i) = \kappa_i\mathbf{e}_i$：
  $$
  k_n(v) = \langle \kappa_1\cos\theta\,\mathbf{e}_1 + \kappa_2\sin\theta\,\mathbf{e}_2,\ \cos\theta\,\mathbf{e}_1 + \sin\theta\,\mathbf{e}_2\rangle = \kappa_1\cos^2\theta + \kappa_2\sin^2\theta
  $$
  交叉项消失——因为 $\mathbf{e}_1\perp\mathbf{e}_2$。
- **第三步，读极值**：$\cos^2\theta, \sin^2\theta \ge 0$ 且和为 1，故 $k_n$ 是 $\kappa_1,\kappa_2$ 的**凸组合**，最大为 $\kappa_1$、最小为 $\kappa_2$。**法曲率落在 $[\kappa_2, \kappa_1]$ 内，极值恰在主方向。**

**重点：Euler 公式说明——曲面的弯曲在「主轴坐标系」里是解耦的。** 沿主方向各弯各的，其他方向只是它们的加权平均。这一「对角化」正是谱定理的几何意义。

## 5 主曲率的地位与主曲率线

主曲率把曲率论推向「结构」层面：

- **主方向场**：在曲面每一点都有两个正交主方向（除脐点外）。把这些方向连成曲线，得到**曲率线（lines of curvature）**——下一节的主题。
- **两个基本不变量**：$H = (\kappa_1+\kappa_2)/2$ 与 $K = \kappa_1\kappa_2$ 是特征值的对称函数，在坐标变换下不变，是「平均」与「乘积」两个视角——下一节定义。
- **曲面的分类线索**：$\kappa_1\kappa_2$ 的符号决定局部形状（凸/马鞍/抛物），$H = 0$ 定义极小曲面（肥皂膜）。<span class="marginnote">主曲率在现代几何处理中是「形状指纹」：配准、变形、分割算法都用 $\kappa_1,\kappa_2$（或 $H,K$）给曲面点分类。而在黎曼几何里，主曲率被推广成「截面曲率」——常曲率空间正是「所有截面曲率相等」的空间，球面、双曲面因此进入同一个框架（第八篇）。</span>

## 6 小结

- **主曲率** $\kappa_1 \ge \kappa_2$ = 形状算子的两个实特征值；**主方向** = 特征方向（正交）。
- 主曲率 = 法曲率的**最大值与最小值**；Euler 公式 $k_n(\theta) = \kappa_1\cos^2\theta + \kappa_2\sin^2\theta$ 给出分布。
- 计算：$\kappa_{1,2} = H \pm \sqrt{H^2 - K}$，或对 $[S] = \mathcal{I}^{-1}\mathcal{II}$ 求特征值。
- 原型：平面 $(0,0)$、球面 $(1/R,1/R)$、圆柱 $(1,0)$、马鞍 $(1,-1)$。
- $\kappa_1=\kappa_2$ 的点是脐点（所有方向皆主方向）；主方向场给出曲率线。

在下一节，我们把两个主曲率组合成两个最常用的整体量：**高斯曲率与平均曲率**——一个描述「内在弯曲」，一个描述「外在平均弯曲」。
