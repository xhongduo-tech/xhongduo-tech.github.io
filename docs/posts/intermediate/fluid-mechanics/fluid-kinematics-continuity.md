---
title: 流体运动学与连续性方程
date: 2026-08-07
---

# 流体运动学与连续性方程

<div class="epigraph">
<p>水是大自然的驱动力，是自然中最奇妙的造物。</p>
<footer>—— 列奥纳多 · 达·芬奇（Leonardo da Vinci，《大西洋古抄本》）</footer>
</div>

<div class="article-byline">
<p>第二级 · 流体力学 ｜ Batchelor Ch. 2-3 ｜ 2026-08-07</p>
</div>

## 为什么从运动学开始

学流体力学，第一个要回答的问题是：**流体的运动如何被描述？** 固体力学里，我们追踪每个质点的位移，但那是因为固体不会"流动"；水、空气这类流体，每一小块都在与相邻部分不断滑移、混合，去追踪每一个"水分子"既不可能也无必要。于是欧拉放弃了"追踪个体"，改为在**固定空间点上**记录速度随时间的变化——这把流体的运动从一个力学问题，先变成了一个纯几何与代数问题。这就是运动学（kinematics）：**只描述运动，不谈产生运动的力**。<span class="marginnote">先运动学、后动力学是经典力学一脉相承的顺序：牛顿力学先有"运动的描述"再有"运动的原因"。流体力学也如此，本章打完运动学的底，下一章才引入应力与力，见《应力张量与本构关系》。</span>

这一章的收获将贯穿整个专题：物质导数、流线、连续性方程，这三样东西是后面每一章（Navier-Stokes、涡量、边界层）的语法。学完它，你就有能力读懂任何一本流体教材的方程在"说什么"。

## 1 欧拉描述与拉格朗日描述

描述流体运动有两种基本观点。

**拉格朗日描述（Lagrangian description）**：把流体看作一个个质点的集合，跟踪每个质点的位置 $\boldsymbol{x}_0$ 在 $t$ 时刻到了哪里，即给出 $\boldsymbol{x}(t; \boldsymbol{x}_0)$。这是"逐人跟踪"的户口本式描述。

**欧拉描述（Eulerian description）**：不看质点来自哪里，只看**每个固定空间点** $(\boldsymbol{x}, t)$ 处的速度 $\boldsymbol{v}(\boldsymbol{x}, t)$、密度 $\rho(\boldsymbol{x}, t)$ 等量。<span class="marginnote">欧拉描述很像气象站的观测方式：气象站不动，记录流经本站的风速与气压。这种"守株待兔"式的记录，是流体力学的主流语言——因为工程上真正需要知道的，往往是某个管段、某个机翼位置处的流动，而不是某个具体质点的旅程。</span>

绝大多数流体力学都用欧拉描述。原因很实在：**流体介质的速度场 $\boldsymbol{v}(\boldsymbol{x},t)$ 是一个可测、可算、可做边界条件的量**，而拉格朗日质点轨迹在湍流中会指数地发散，几乎没有实用价值（海洋浮标、粒子图像测速除外）。

## 2 物质导数：从"点"到"质点"的换乘

欧拉描述的一个直接困难是：我们在固定点上取值，可力学定律（牛顿第二定律）作用于**物质质点**。于是必须把"固定点上量的变化"翻译成"跟随流体质点时量的变化"。这个翻译工具就是**物质导数（material derivative）**：

$$\frac{D}{Dt} = \frac{\partial}{\partial t} + \boldsymbol{v} \cdot \nabla$$

对任意标量场 $f(\boldsymbol{x},t)$，有 $\frac{Df}{Dt} = \frac{\partial f}{\partial t} + \boldsymbol{v}\cdot\nabla f$。<span class="marginnote">$\partial/\partial t$ 是"站在固定点看变化"（当地项），$\boldsymbol{v}\cdot\nabla$ 是"流体质点带着场量跑进跑出带来的变化"（迁移项）。乘公交车看街景变化就是当地项+迁移项的通俗版本——街景既随时间变，也随你移动而变。</span>

**物质导数是连接两种描述的唯一桥梁**：它是"欧拉场在物质点上的时间变化率"。最典型的应用是加速度——质点的加速度不是 $\partial\boldsymbol{v}/\partial t$，而是：

$$\boldsymbol{a} = \frac{D\boldsymbol{v}}{Dt} = \frac{\partial\boldsymbol{v}}{\partial t} + (\boldsymbol{v}\cdot\nabla)\boldsymbol{v}$$

这一条后面直接进入动量方程，是 Navier-Stokes 方程左侧的**惯性项**。非线性就藏在 $(\boldsymbol{v}\cdot\nabla)\boldsymbol{v}$ 里——它是流体方程所有复杂性的总根源。

## 3 流线、迹线与脉线

描述速度场，三张"地图"最容易混淆，先立住定义再谈区别：

- **流线（streamline）**：某一**固定时刻**，与速度场处处相切的曲线。由 $\frac{dx}{v_x} = \frac{dy}{v_y} = \frac{dz}{v_z}$ 决定。
- **迹线（pathline）**：**同一个质点**随时间走过的轨迹。
- **脉线（streakline）**：从**同一空间点**先后放出的所有质点在某一时刻连成的线（烟囱冒烟的烟线就是脉线）。

**辨析｜易错点：** 三者在定常流（速度场不随时间变）中完全重合，这正是容易混淆的原因；一旦流场非定常，三者彼此分裂。<span class="marginnote">经典例子：把颜料从固定点持续滴入流动的水中，拍照得到的是脉线；追踪某个颜料点的运动是迹线；给某一瞬间全场画切线是流线。对流场而言，流线是"现在的骨架"，迹线是"个体的档案"。</span>

**重点：流线永远不能相交。** 因为流线相交处速度必须同时沿两个方向，这在有限速度下不可能。这一几何性质将限制后面势流、边界层里所有绘图与想象。

## 4 形变：从刚体速度到应变率张量

速度场除了告诉流体"往哪走"，还决定流体"如何变形"。考察邻域内两点速度的差，做泰勒展开后，速度梯度 $\nabla\boldsymbol{v}$ 可分拆为：

$$\frac{\partial v_i}{\partial x_j} = \underbrace{\frac{1}{2}\left(\frac{\partial v_i}{\partial x_j}-\frac{\partial v_j}{\partial x_i}\right)}_{\text{旋度·刚体转动}} + \underbrace{\frac{1}{2}\left(\frac{\partial v_i}{\partial x_j}+\frac{\partial v_j}{\partial x_i}\right)}_{\text{应变率张量 } e_{ij}}$$

反对称部分对应流体微团的刚性旋转（与涡量 $\boldsymbol{\omega}=\nabla\times\boldsymbol{v}$ 直接挂钩），对称部分 $e_{ij}$ 才是真正的**变形**。$e_{ij}$ 称为**应变率张量（rate-of-strain tensor）**，它是对称张量，主对角元是拉伸/压缩率，非对角元是剪切率。<span class="marginnote">应变率张量是下一章《应力张量与构关系》的引子：牛顿流体里应力与 $e_{ij}$ 成正比，粘度就是那个比例系数。把"变形"定量化，是走向"力"的桥梁。</span>

**核心概念：** 这种"把梯度拆成对称与反对称部分"的手法，在线性代数里叫矩阵分解，在流体力学里叫 Helmholtz 分解——它揭示了流体运动 = 平动 + 转动 + 形变。

## 5 连续性方程：质量守恒的欧拉翻译

力学定律第一条是质量守恒。在一个固定不动的小控制体 $V$ 里，质量的增加只能来自净流入。设密度为 $\rho$，流速为 $\boldsymbol{v}$，则：

$$\frac{\partial \rho}{\partial t} + \nabla\cdot(\rho \boldsymbol{v}) = 0$$

**这就是连续性方程（continuity equation）**。它的形式是典型的守恒律：$\frac{\partial(\text{密度})}{\partial t} + \nabla\cdot(\text{通量}) = 0$。<span class="marginnote">同样的骨架在物理里到处都是：电荷守恒 $\partial\rho_q/\partial t + \nabla\cdot\boldsymbol{J}=0$、概率守恒 $\partial\rho_p/\partial t + \nabla\cdot\boldsymbol{J}=0$（薛定谔方程）。看到 $\partial_t + \nabla\cdot$ 就想到"某量守恒"。</span>由散度定理 $\int_V \nabla\cdot(\rho\boldsymbol{v})\,dV = \oint_S \rho\boldsymbol{v}\cdot\boldsymbol{n}\,dS$，它等价于"净流出率 + 质量增长率 = 0"。

对**不可压缩**流动（密度恒定），连续性方程退化为全场散度为零：

$$\nabla \cdot \boldsymbol{v} = 0$$

这一条极其重要：它说明不可压缩流的速度场是**无源无汇的螺线管场**，流体微团只改变形状不改变体积。<span class="marginnote">"不可压缩"不等于"密度处处相同"——它只要求每个微团密度不变。海洋中层的水、低速空气、液态水都近似不可压缩，这使数学大为简化（压强从约束变为待求的拉格朗日乘子，见《Navier-Stokes 方程》）。</span>

## 6 公式解析：连续性方程怎么读

把不可压缩形式拆开逐项看：

$$\frac{\partial u}{\partial x} + \frac{\partial v}{\partial y} + \frac{\partial w}{\partial z} = 0$$

- **第一步，认识符号**：$u,v,w$ 是速度 $\boldsymbol{v}$ 的三个分量，沿 $x,y,z$ 方向；$\frac{\partial u}{\partial x}$ 是 $u$ 沿 $x$ 方向的变化率。
- **第二步，理解"一进一出"**：考虑一个边长 $dx,dy,dz$ 的小立方体。沿 $x$ 方向，左边面流进 $u\,dy\,dz$，右边面流出 $(u+\frac{\partial u}{\partial x}dx)dy\,dz$，净流出 $\frac{\partial u}{\partial x}dx\,dy\,dz$。三个方向累加，净流出量为 $(\nabla\cdot\boldsymbol{v})dV$。
- **第三步，翻译成语言**：质量既不能凭空产生也不能凭空消失，所以三个方向的净流出之和必须为零。$\nabla\cdot\boldsymbol{v}=0$ 说的就是：**各处流出的总量与流入的总量恰好抵消**。
- **第四步，直觉检验**：$u = \alpha x$（流体沿 $x$ 越流越快）单独出现时 $\nabla\cdot\boldsymbol{v}=\alpha\neq 0$，必然伴随 $v=-\alpha y$ 之类的横向收缩来补偿——这就是管道变窄处流速加快、流线收拢的数学根源。

## 7 数值算例：一根水管告诉你连续性在说什么

把连续性方程落成数，直觉立刻具体。设一根圆管，粗段半径 $R_1=2\,\text{cm}$，细段半径 $R_2=1\,\text{cm}$，水以 $v_1=0.5\,\text{m/s}$ 流过粗段。不可压缩条件 $\nabla\cdot\boldsymbol{v}=0$ 在截面上的积分形式是**流量守恒**：

$$A_1 v_1 = A_2 v_2, \qquad v_2 = v_1\frac{A_1}{A_2} = v_1\left(\frac{R_1}{R_2}\right)^2$$

代入数值：$v_2 = 0.5\times(2/1)^2 = 2.0\,\text{m/s}$——**半径减半，流速变四倍**。这正是水管捏扁出水口水流加速、消防水枪增压的原理，也是"越窄越快"这句日常直觉的精确来源。

| 半径比 $R_1/R_2$ | 速度比 $v_2/v_1$ | 压强变化（定性） |
| --- | --- | --- |
| 1 | 1 | 不变 |
| 1.5 | 2.25 | 下降（文丘里效应） |
| 2 | 4 | 显著下降 |
| 3 | 9 | 强烈下降 |

**辨析｜易错点：** 连续性方程说的是"流量守恒"，不是"速度守恒"。常有人把 $v_2=v_1R_1^2/R_2^2$ 记成 $v_2=v_1R_1/R_2$——那是把面积比错当成半径比。面积正比半径平方，这一层平方正是整个公式的灵魂。<span class="marginnote">若要进一步算压强，就把 $v_2$ 代进伯努利方程（见《不可压缩无粘流动与势流理论》）：$\Delta p=\frac{1}{2}\rho(v_1^2-v_2^2)$，水 $\rho=1000\,\mathrm{kg/m^3}$，上例 $\Delta p\approx-1875\,\mathrm{Pa}$——细段的"负压"正是喷射、抽吸的力学来源。</span>

## 8 术语速查表

| 术语 | 一句话定义 | 记号 |
| --- | --- | --- |
| 欧拉描述 | 盯固定空间点记录场量 | $\boldsymbol{v}(\boldsymbol{x},t)$ |
| 拉格朗日描述 | 追踪单个流体质点 | $\boldsymbol{x}(t;\boldsymbol{x}_0)$ |
| 物质导数 | 跟随质点的变化率 | $\frac{D}{Dt}=\partial_t+\boldsymbol{v}\cdot\nabla$ |
| 流线 | 固定时刻与速度场相切 | $dx/v_x=dy/v_y=dz/v_z$ |
| 迹线 | 同一质点走过的轨迹 | $\boldsymbol{x}(t)$ |
| 脉线 | 同一源点先后放出的质点连线 | — |
| 应变率张量 | 速度梯度的对称部分，度量形变 | $e_{ij}$ |
| 涡量 | 速度场的旋度，转动角速度的两倍 | $\boldsymbol{\omega}=\nabla\times\boldsymbol{v}$ |
| 连续性方程 | 质量守恒的微分形式 | $\partial_t\rho+\nabla\cdot(\rho\boldsymbol{v})=0$ |
| 不可压缩条件 | 微团密度不变的等价条件 | $\nabla\cdot\boldsymbol{v}=0$ |

## 9 小结

- 流体运动学有两种描述：**拉格朗日**（追质点）与**欧拉**（盯固定点）；流体力学几乎全用欧拉描述。
- **物质导数** $\frac{D}{Dt}=\frac{\partial}{\partial t}+\boldsymbol{v}\cdot\nabla$ 是"跟随质点"的变化率，加速度就是 $\frac{D\boldsymbol{v}}{Dt}$。
- **流线、迹线、脉线**三线在定常流中重合；流线不相交。
- 速度梯度分解为**刚体转动（涡量）**与**应变率张量**，后者是形变的度量。
- **连续性方程** $\frac{\partial\rho}{\partial t}+\nabla\cdot(\rho\boldsymbol{v})=0$；不可压缩时退化为 $\nabla\cdot\boldsymbol{v}=0$。

在下一节，我们将把"形变"与"力"接上：引入应力张量，回答"流体内部如何传递作用力"——这是《应力张量与构关系》的内容。
