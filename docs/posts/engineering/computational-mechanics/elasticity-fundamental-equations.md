---
title: 弹性力学基本方程（应力/应变/本构）
date: 2026-08-07
---

# 弹性力学基本方程（应力/应变/本构）

<div class="epigraph">
<p>大自然这本书是用数学语言写成的。</p>
<footer>—— 伽利略 · 伽利雷（Galileo Galilei，Il Saggiatore，1623）</footer>
</div>

<div class="article-byline">
<p>第六级 · 计算力学与有限元方法 ｜ 王勖成《有限单元法》第2章 ｜ 2026-08-07</p>
</div>

## 为什么从弹性力学基本方程开始

有限元方法不是凭空造出的算法，它是一台把「连续体问题」翻译成「有限自由度问题」的机器。而翻译的前提，是先把源语言本身学清楚：**弹性力学基本方程**就是那门源语言——它用三组方程把「固体在外力下如何变形、内部如何传力」这件事写成了数学。学完这一篇，你会拥有一张「连续体问题的完整清单」：15 个未知量、15 个方程，再加上边界条件恰好封闭<span class="marginnote">连续介质力学里「未知量个数 = 方程个数」并不是巧合，而是建模时反复对照检查的纪律。这个「数方程」的习惯，在后面学有限元离散时还会再次出现。</span>。后续所有有限元公式——从直接刚度法到等参数单元——都只是在这三组方程之上做离散逼近。这一篇是整座大厦的地基。

## 1 应力：内力在截面上的密度

弹性力学研究的对象是**连续体（continuum）**：把材料想象成没有间隙的介质，每一个点都可以指定一个位移、一个应力。先从「内力」说起。当外力作用在物体上时，内部各点之间会互相推拉，这种内部作用力叫**内力**。<span class="marginnote">材料力学里你已经见过「截面法」：切开一个面，把内力显式暴露出来。弹性力学把它推广到「每一点、每个方向上」——这就是应力的来由。</span>

**应力（stress）**：过物体内一点 $P$ 作一个微小截面，面积为 $\Delta A$，面上总内力为 $\Delta \boldsymbol{F}$，则当 $\Delta A \to 0$ 时极限

$$\boldsymbol{\sigma}(\boldsymbol{n}) = \lim_{\Delta A \to 0} \frac{\Delta \boldsymbol{F}}{\Delta A}$$

称为该点在该截面上的**应力矢量**，它随截面方向 $\boldsymbol{n}$ 而变。要完整描述一点的受力状态，需要取三个互相垂直的截面（比如与坐标平面平行的三个面），把每个截面上的应力矢量按坐标方向分解，得到九个分量，排成一个**二阶张量**：

$$
\boldsymbol{\sigma} = \begin{bmatrix} \sigma_{xx} & \sigma_{xy} & \sigma_{xz} \\ \sigma_{yx} & \sigma_{yy} & \sigma_{yz} \\ \sigma_{zx} & \sigma_{zy} & \sigma_{zz} \end{bmatrix}
$$

其中 $\sigma_{ii}$ 是正应力，$\sigma_{ij}\,(i \neq j)$ 是剪应力。记号约定：第一个下标表示截面法向，第二个下标表示应力分量方向。

**关键结论：应力张量是对称的**，即 $\sigma_{ij} = \sigma_{ji}$。这来自微元体的**力矩平衡**——如果没有对称性，微元体会发生自转。<span class="marginnote">对称性把独立应力分量从 9 个减到 6 个。这个「守恒律自动减少独立分量」的模式在固体力学中反复出现，也是后续本构模型为什么只需要 6 个应力分量的原因。</span>

## 2 应变：变形的度量

应力回答「内部怎么传力」，**应变（strain）** 回答「材料怎么变形」。物体变形后，原来坐标为 $\boldsymbol{x}$ 的点移动到 $\boldsymbol{x} + \boldsymbol{u}(\boldsymbol{x})$，其中 $\boldsymbol{u}$ 是**位移场（displacement field）**——一个连续函数。

一点的变形状态用**应变张量**描述。在小变形假设下（位移梯度远小于 1），采用**柯西小应变（Cauchy small strain）**：

$$
\varepsilon_{ij} = \frac{1}{2}\left( \frac{\partial u_i}{\partial x_j} + \frac{\partial u_j}{\partial x_i} \right)
$$

展开来看，对角线分量 $\varepsilon_{xx} = \partial u_x / \partial x$ 是沿 $x$ 方向的伸长率，剪应变分量 $\varepsilon_{xy} = \tfrac{1}{2}(\partial u_x/\partial y + \partial u_y/\partial x)$ 度量角度畸变。<span class="marginnote">注意剪应变定义里的系数 $\tfrac{1}{2}$：工程剪应变 $\gamma_{xy} = 2\varepsilon_{xy}$ 是直角改变量，而张量剪应变是它的一半——初学时单位换算出错十有八九栽在这里。</span>

应变张量同样对称，独立分量也是 6 个。而由位移场按上式求得的应变场必须满足一组**协调方程（compatibility equations）**，否则应变场对应不出连续的单值位移场——这是变形几何学内部的「自洽性约束」，在三维问题里被自动满足，但近似解不会自动满足，这正是后面误差分析的重要话题。

## 3 本构关系：材料如何「回答」变形

应力与应变之间的联系由**本构关系（constitutive relation）** 给出，它刻画材料自身的力学个性。最简单、也最常用的是**线弹性（linear elasticity）**：

$$
\boldsymbol{\sigma} = \boldsymbol{D} : \boldsymbol{\varepsilon}
$$

即应力是应变的线性函数。对**各向同性**材料（性质不随方向而变），本构矩阵只含两个独立参数——杨氏模量 $E$ 与泊松比 $\nu$。用分量写，正应力与正应变的关系为：

$$
\sigma_{xx} = \frac{E}{(1+\nu)(1-2\nu)}\left[(1-\nu)\varepsilon_{xx} + \nu(\varepsilon_{yy} + \varepsilon_{zz})\right]
$$

剪应力与剪应变的关系则简单得多：$\sigma_{xy} = 2G\varepsilon_{xy}$，其中 $G = E/[2(1+\nu)]$ 是**剪切模量（shear modulus）**。<span class="marginnote">$E$、$G$、$K$（体积模量）与 $\nu$ 之间存在通用换算公式，任意两者可推出其余。材料力学课表里那张换算表，本质就是本构方程在「各向同性 + 线弹性」下的代数推论。</span>

**辨析｜易错点：** 初学者常把「应力-应变关系」和「本构关系」当成一回事，其实本构关系是更一般的词——它涵盖塑性、黏弹性、超弹性等一切「材料如何响应变形」的规律，线弹性只是其中最简单的一支。有限元里「材料非线性」改的就是这张表，而不是改平衡方程或几何方程。

## 4 平衡方程与边界条件

连续体内任意一个微元体都必须满足**静力平衡**。以 $x$ 方向为例，把作用在微元体六个面上的应力差与外体积力 $b_x$（单位体积力）加起来，令其为零，得到：

$$
\frac{\partial \sigma_{xx}}{\partial x} + \frac{\partial \sigma_{yx}}{\partial y} + \frac{\partial \sigma_{zx}}{\partial z} + b_x = 0
$$

三个方向合起来写成矢量形式：$\nabla \cdot \boldsymbol{\sigma} + \boldsymbol{b} = \boldsymbol{0}$，这是**平衡方程（equilibrium equations）**。<span class="marginnote">对比材料力学：杆件的 $\sigma = F/A$ 是「平均」意义下的平衡，而这里是「每一点都平衡」——弹性力学把静力平衡从整体升级到了逐点。这正是它能处理复杂应力场的原因。</span>

边界条件分两类，缺一不可：

- **力边界（自然边界）**：边界上应力矢量等于给定表面力 $\bar{\boldsymbol{t}}$，即 $\boldsymbol{\sigma} \cdot \boldsymbol{n} = \bar{\boldsymbol{t}}$。
- **位移边界（本质边界）**：边界上位移等于给定值 $\bar{\boldsymbol{u}}$，即 $\boldsymbol{u} = \bar{\boldsymbol{u}}$。

**弹性力学的完备表述**：几何方程（6 个）、平衡方程（3 个）、本构方程（6 个），共 15 个方程，对应 6 个应力 + 6 个应变 + 3 个位移共 15 个未知量；加上力边界与位移边界，问题在数学上封闭。

## 5 公式解析：平面应力问题的三大方程组合

工程中最常用的是**平面问题**。以薄板平面应力问题为例，$\sigma_{zz} = \sigma_{xz} = \sigma_{yz} = 0$，未知量只剩 $u_x, u_y$、$\varepsilon_{xx}, \varepsilon_{yy}, \varepsilon_{xy}$、$\sigma_{xx}, \sigma_{yy}, \sigma_{xy}$。三大方程降维组合如下。

- **几何方程**（把位移映射为应变）：

$$
\begin{bmatrix} \varepsilon_{xx} \\ \varepsilon_{yy} \\ \gamma_{xy} \end{bmatrix} = \begin{bmatrix} \frac{\partial}{\partial x} & 0 \\ 0 & \frac{\partial}{\partial y} \\ \frac{\partial}{\partial y} & \frac{\partial}{\partial x} \end{bmatrix} \begin{bmatrix} u_x \\ u_y \end{bmatrix} = \boldsymbol{L} \boldsymbol{u}
$$

- **本构方程**（平面应力）：

$$
\begin{bmatrix} \sigma_{xx} \\ \sigma_{yy} \\ \tau_{xy} \end{bmatrix} = \frac{E}{1-\nu^2} \begin{bmatrix} 1 & \nu & 0 \\ \nu & 1 & 0 \\ 0 & 0 & \frac{1-\nu}{2} \end{bmatrix} \begin{bmatrix} \varepsilon_{xx} \\ \varepsilon_{yy} \\ \gamma_{xy} \end{bmatrix} = \boldsymbol{D} \boldsymbol{\varepsilon}
$$

- **平衡方程**：$\boldsymbol{L}^{\mathsf{T}} \boldsymbol{\sigma} + \boldsymbol{b} = \boldsymbol{0}$。

**三步拆解这条链条**：

- **第一步，看算子 $\boldsymbol{L}$**：它把「位移 → 应变」，是一个微分算子矩阵，与有限元里的「应变-位移矩阵 $\boldsymbol{B}$」一一对应——有限元就是把 $\boldsymbol{L}$ 离散成代数矩阵 $\boldsymbol{B}$。
- **第二步，看 $\boldsymbol{D}$ 矩阵**：它把「应变 → 应力」，平面应力问题里 $E/(1-\nu^2)$ 是等效刚度系数，比单轴情形更大，因为横向被约束了。
- **第三步，拼装**：把 $\boldsymbol{\varepsilon} = \boldsymbol{L}\boldsymbol{u}$ 代入 $\boldsymbol{\sigma} = \boldsymbol{D}\boldsymbol{\varepsilon}$，再代进平衡方程，得到只含位移 $\boldsymbol{u}$ 的**位移型控制方程**——这正是有限元位移法求解的对象。

## 6 小结

- **应力张量**：对称二阶张量，9 个分量只有 6 个独立；由微元体平衡推出 $\sigma_{ij} = \sigma_{ji}$。
- **应变张量**：小变形下为 $\varepsilon_{ij} = \tfrac{1}{2}(u_{i,j} + u_{j,i})$，对称、6 个独立分量，须满足协调方程。
- **本构关系**：线弹性 $\boldsymbol{\sigma} = \boldsymbol{D}:\boldsymbol{\varepsilon}$，各向同性材料只需 $E$ 与 $\nu$ 两个参数。
- **平衡方程** $\nabla\cdot\boldsymbol{\sigma} + \boldsymbol{b} = \boldsymbol{0}$。
- **边界条件**：力边界 $\boldsymbol{\sigma}\cdot\boldsymbol{n} = \bar{\boldsymbol{t}}$ 与位移边界 $\boldsymbol{u} = \bar{\boldsymbol{u}}$ 缺一不可；三组方程对应 15 个未知量，加上边界条件问题才封闭。
- **变分表述**：三大方程可合并成虚位移原理这一等价形式，它是下一节有限元离散的出发点。

在下一节，我们将用**虚位移原理**把三组连续方程改写为「弱形式」，从这里导出有限元的第一块积木——单元刚度矩阵。那是把「连续体的源语言」翻译成「代数机器语言」的第一行。