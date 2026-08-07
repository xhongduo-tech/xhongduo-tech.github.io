---
title: 空间直线及其方程
date: 2026-08-07
---

# 空间直线及其方程

<div class="epigraph">
<p>直线是最短的曲线，也是最干净的方向。</p>
<footer>—— 大卫 · 希尔伯特（David Hilbert）</footer>
</div>

<div class="article-byline">
<p>第二级 · 高等数学 ｜ 同济《高等数学》下册 §8.4 ｜ 2026-08-07</p>
</div>

## 为什么从空间直线及其方程开始

上一节用「点 + 法向量」定义了平面；空间直线的对应范式是「**点 + 方向向量**」——直线是沿固定方向无限延伸的点的集合。它的方程有三种等价写法：**参数式**（最本质）、**对称式**、**一般式**（两平面交线）。研究直线绕不开与平面的位置关系：交点、夹角、距离。这些内容不仅是空间几何的骨架，更是计算机图形学（光线与物体的求交）、机器人运动学（关节的轴方向）、以及多元函数切线的空间推广的必备工具。<span class="marginnote">「点 + 方向向量」范式在参数式里最直观：$M = M_0 + t\mathbf{s}$——从锚点出发，沿方向 $\mathbf{s}$ 走 $t$ 倍。这个「锚点 + 方向 × 参数」的结构与平面「锚点 + 法向量」并列，构成空间解析几何的两大基本范式，也直接通向线性代数里的「直线 = 仿射子空间」。</span>

## 1 直线的参数式与对称式

设直线 $L$ 过点 $M_0(x_0,y_0,z_0)$，方向向量 $\mathbf{s} = (m, n, p)$。直线上任一点 $M$ 满足 $\overrightarrow{M_0M} = t\mathbf{s}$，得**参数式方程**：

$$x = x_0 + mt, \qquad y = y_0 + nt, \qquad z = z_0 + pt$$

消去参数 $t$，得**对称式（点向式）方程**：

$$\frac{x - x_0}{m} = \frac{y - y_0}{n} = \frac{z - z_0}{p}$$

<span class="marginnote">对称式的解读：三个比例式相等，比值就是参数 $t$——「$\frac{x-x_0}{m}$ 等比分点」是直线穿过点 $M_0$、方向 $(m,n,p)$ 的坐标表述。若某方向分量 $p=0$，对称式对应分子 $z-z_0=0$（约定分母为 0 表示「该方向分量为零」），如 $\frac{x-x_0}{m} = \frac{y-y_0}{n}, z=z_0$。</span>

**两点式**：过两点 $M_1, M_2$ 的直线，方向向量取 $\overrightarrow{M_1M_2}$，参数式即 $M = M_1 + t(M_2 - M_1)$。

## 2 直线的一般式

**直线的一般式**：空间中一条直线可以看作两个平面的交线：

$$\begin{cases}A_1x + B_1y + C_1z + D_1 = 0\\ A_2x + B_2y + C_2z + D_2 = 0\end{cases}$$

两平面的法向量 $\mathbf{n}_1, \mathbf{n}_2$ 都垂直于交线，故交线的方向向量

$$\mathbf{s} = \mathbf{n}_1 \times \mathbf{n}_2$$

——**直线方向 = 两平面法向量的叉积**。<span class="marginnote">叉积的又一次几何应用：两平面法向量的叉积同时垂直于两个法向量，因而平行于两平面的交线——所以它就是直线的方向向量。求一般式直线的方法：取方向向量 $\mathbf{n}_1\times\mathbf{n}_2$，再找交线上任意一点（令某坐标 $=0$ 解方程组）。</span>

## 3 直线与平面的位置关系

**直线与平面的夹角**：直线与它在平面上投影的夹角（锐角）。直线方向 $\mathbf{s}=(m,n,p)$，平面法向 $\mathbf{n}=(A,B,C)$，夹角 $\varphi$ 满足

$$\sin\varphi = \frac{|\mathbf{s}\cdot\mathbf{n}|}{|\mathbf{s}|\,|\mathbf{n}|} = \frac{|Am + Bn + Cp|}{\sqrt{A^2+B^2+C^2}\,\sqrt{m^2+n^2+p^2}}$$

用 $\sin$ 而非 $\cos$——因为夹角是「直线与平面」而非「两法向量」。<span class="marginnote">直线与平面夹角公式用 $\sin$ 的原因：直线方向 $\mathbf{s}$ 与法向量 $\mathbf{n}$ 的夹角 $\psi$ 是「直线与平面法向」的夹角，而直线与平面的夹角 $\varphi = \frac{\pi}{2} - \psi$，故 $\sin\varphi = \cos\psi = \frac{|\mathbf{s}\cdot\mathbf{n}|}{|\mathbf{s}||\mathbf{n}|}$。理清「法向 vs 平面」的互补角关系，公式就不易记反。</span>

**特殊位置**：
- 直线 $\perp$ 平面 ⟺ $\mathbf{s} \parallel \mathbf{n}$（$\frac{A}{m}=\frac{B}{n}=\frac{C}{p}$）；
- 直线 $\parallel$ 平面 ⟺ $\mathbf{s} \perp \mathbf{n}$（$Am+Bn+Cp=0$）。

**直线与平面的交点**：把直线参数式代入平面方程，解出参数 $t$，代回得交点坐标。

## 4 公式解析：点到直线的距离与异面直线距离

**点到直线的距离**：点 $M_1$ 到过 $M_0$、方向 $\mathbf{s}$ 的直线的距离

$$d = \frac{|\overrightarrow{M_0M_1} \times \mathbf{s}|}{|\mathbf{s}|}$$

- **第一步，理解分子**：$\overrightarrow{M_0M_1}\times\mathbf{s}$ 的模是以 $\overrightarrow{M_0M_1}$ 与 $\mathbf{s}$ 为邻边的平行四边形面积。
- **第二步，几何来源**：该面积 = 底 $|\mathbf{s}|$ × 高（点到直线的距离 $d$），所以 $d$ = 面积 ÷ 底长。
- **第三步，特例检查**：若 $M_1$ 在直线上，则 $\overrightarrow{M_0M_1}\parallel\mathbf{s}$，叉积为零，$d=0$，合理。

**异面直线的距离**：两条异面直线 $L_1$（过 $M_1$、方向 $\mathbf{s}_1$）与 $L_2$（过 $M_2$、方向 $\mathbf{s}_2$）的距离

$$d = \frac{|(\mathbf{s}_1 \times \mathbf{s}_2)\cdot\overrightarrow{M_1M_2}|}{|\mathbf{s}_1\times\mathbf{s}_2|}$$

分子是混合积——以 $\mathbf{s}_1,\mathbf{s}_2,\overrightarrow{M_1M_2}$ 为棱的平行六面体体积，除以底面积 $\mathbf{s}_1\times\mathbf{s}_2$，即得两直线间「最短距离」= 平行六面体高。<span class="marginnote">异面直线距离公式是混合积的经典应用：两直线的公垂线方向是 $\mathbf{s}_1\times\mathbf{s}_2$，距离等于「连接两直线上任意点的向量在该方向的投影长度」——正是混合积 ÷ 底面积。一条公式同时用了叉积与点积，是本章三种向量积的「总演习」。</span>

## 5 直线方程的应用

- **计算机图形学与光线追踪**：光线 = 参数式直线 $P = P_0 + t\mathbf{d}$，求「光线与平面/球面的交点」就是解参数方程——整个光线追踪渲染的基础。<span class="marginnote">光线追踪的核心运算「光线与三角形求交」正是直线参数式与平面方程联立求解。你在第三级《计算机图形学》里会反复用到本节最基础的参数式——<strong>一条直线方程撑起了整个逼真渲染的几何引擎</strong>。</span>
- **机器人运动学**：关节轴线、机械臂末端的运动轨迹用直线/曲线参数方程描述。
- **空间解析几何综合**：判断两直线共面/异面（混合积）、求两直线夹角（方向向量夹角）、求直线在平面上的投影——都是本节工具的组合。
- **线性代数视角**：直线是「一维仿射子空间」，参数式是「锚点 + 一维方向」的精确表述。

## 6 小结

- **直线**：点 + 方向向量；参数式 $x=x_0+mt$ 等，对称式 $\frac{x-x_0}{m}=\frac{y-y_0}{n}=\frac{z-z_0}{p}$。
- **一般式**：两平面交线，方向向量 $\mathbf{s} = \mathbf{n}_1\times\mathbf{n}_2$。
- **直线与平面夹角**：$\sin\varphi = \frac{|\mathbf{s}\cdot\mathbf{n}|}{|\mathbf{s}||\mathbf{n}|}$；垂直、平行由 $\mathbf{s},\mathbf{n}$ 的关系判定。
- **点到直线距离**：$d = \frac{|\overrightarrow{M_0M_1}\times\mathbf{s}|}{|\mathbf{s}|}$；异面直线距离用混合积。
- 直线方程是光线追踪、机器人运动学与空间几何计算的基础。

在下一节，我们将从直线平面进入更丰富的曲面世界——**曲面及其方程**。
