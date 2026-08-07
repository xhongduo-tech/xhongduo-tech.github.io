---
title: 电磁场的矢势、标势与规范变换
date: 2026-08-07
---

# 电磁场的矢势、标势与规范变换

<div class="epigraph">
<p>电场和磁场是「实在」，但描述它们的方式不止一种——势的选择有一整片自由的余地，而物理结果纹丝不动。这份自由，叫规范自由度。</p>
<footer>—— 电动力学引言</footer>
</div>

<div class="article-byline">
<p>第二级 · 高等物理 ｜ 郭硕鸿《电动力学》第三章 ｜ 2026-08-07</p>
</div>

## 为什么从矢势开始

静电场可以用标势 $\phi$ 描述（$\boldsymbol{E} = -\nabla\phi$），但变化的电磁场需要更完整的描述。由于 $\nabla\cdot\boldsymbol{B} = 0$（磁场无源），磁感应强度可以写成某矢量的旋度——**矢势（vector potential）** $\boldsymbol{A}$；电场则用矢势与标势联合表达。势的引入让麦克斯韦方程组化为「势的波动方程」，但也带来**规范自由度**：势的选择不唯一。这一节建立矢势、标势与规范变换——它是辐射、量子力学（磁场的规范耦合）、粒子物理（规范场论）的基础。

## 1 矢势与标势

**矢势（vector potential）** $\boldsymbol{A}$：由 $\nabla\cdot\boldsymbol{B} = 0$（无磁荷），存在 $\boldsymbol{A}$ 使：

$$\boldsymbol{B} = \nabla\times\boldsymbol{A}$$

**标势与矢势联合给出电场**：由法拉第定律 $\nabla\times\boldsymbol{E} = -\frac{\partial\boldsymbol{B}}{\partial t}$，代入 $\boldsymbol{B} = \nabla\times\boldsymbol{A}$：

$$\nabla\times\left(\boldsymbol{E} + \frac{\partial\boldsymbol{A}}{\partial t}\right) = 0 \quad\Longrightarrow\quad \boldsymbol{E} = -\nabla\phi - \frac{\partial\boldsymbol{A}}{\partial t}$$

**重点：电磁场由矢势 $\boldsymbol{A}$ 与标势 $\phi$ 描述——$\boldsymbol{B} = \nabla\times\boldsymbol{A}$、$\boldsymbol{E} = -\nabla\phi - \partial\boldsymbol{A}/\partial t$。** 电场不仅来自标势梯度（静态部分），还来自矢势对时间的变化率（感应部分）——静电场 $\boldsymbol{E} = -\nabla\phi$ 是 $\partial\boldsymbol{A}/\partial t = 0$ 时的特例。<span class="marginnote">「为什么需要矢势」：磁场无源（$\nabla\cdot\boldsymbol{B} = 0$）保证磁场可写成旋度——矢势是「磁场的势」。它不只是数学工具：在量子力学中，带电粒子受磁场作用是通过矢势耦合（$\boldsymbol{p} \to \boldsymbol{p} - q\boldsymbol{A}$）；在超导（阿哈罗诺夫-玻姆效应）中，矢势有可观测的物理效应（即使磁场为零的区域）。</span>

## 2 规范变换

势的选取不唯一：作**规范变换（gauge transformation）**：

$$\boldsymbol{A}' = \boldsymbol{A} + \nabla\chi, \qquad \phi' = \phi - \frac{\partial\chi}{\partial t}$$

其中 $\chi(\boldsymbol{r}, t)$ 是任意标量函数。则 $\boldsymbol{B}$、$\boldsymbol{E}$ 不变：

$$\nabla\times\boldsymbol{A}' = \nabla\times\boldsymbol{A} = \boldsymbol{B}, \qquad -\nabla\phi' - \frac{\partial\boldsymbol{A}'}{\partial t} = \boldsymbol{E}$$

**重点：规范变换不改变电磁场——$\boldsymbol{A}$、$\phi$ 的选取有「规范自由度」，但物理量（$\boldsymbol{E}$、$\boldsymbol{B}$）是规范不变的。** 这是规范对称性的雏形：物理规律在「势的重定义」下不变。选择合适的规范可以简化方程。

## 3 常用规范

**库仑规范（Coulomb gauge）**：$\nabla\cdot\boldsymbol{A} = 0$。标势满足泊松方程（$\nabla^2\phi = -\rho/\varepsilon_0$），静电场部分分离；适合近静态问题。

**洛伦兹规范（Lorenz gauge）**：$\nabla\cdot\boldsymbol{A} + \frac{1}{c^2}\frac{\partial\phi}{\partial t} = 0$。势的方程化为**达朗贝尔方程（波动方程）**：

$$\nabla^2\boldsymbol{A} - \frac{1}{c^2}\frac{\partial^2\boldsymbol{A}}{\partial t^2} = -\mu_0\boldsymbol{j}, \qquad \nabla^2\phi - \frac{1}{c^2}\frac{\partial^2\phi}{\partial t^2} = -\frac{\rho}{\varepsilon_0}$$

**重点：洛伦兹规范下，势满足达朗贝尔方程（波动方程）——电磁波、推迟势（第 125 节）都从这里解出。** 洛伦兹规范使 $\boldsymbol{A}$ 与 $\phi$ 对称（都是波动方程），适合辐射问题。库仑规范适合静态/近静态。

**辨析｜易错点：**「规范不变性」不是「势没有意义」——物理量（场、力、量子相位）是规范不变的，但势本身是计算工具，选择不同的规范会简化不同的问题。洛伦兹规范与库仑规范是两种常用选择，题目会说明用哪种（或自由选）。**别把「规范自由」误读为「势可任意改」——规范变换受 $\chi$ 约束，物理结果必须不变。**

## 4 公式解析：由矢势求磁场

矢势 $\boldsymbol{A} = \frac{1}{2}\boldsymbol{B}_0\times\boldsymbol{r}$（$\boldsymbol{B}_0$ 为常量，均匀磁场），验证 $\boldsymbol{B} = \nabla\times\boldsymbol{A} = \boldsymbol{B}_0$。

$$
\nabla\times\boldsymbol{A} = \nabla\times\left(\frac{1}{2}\boldsymbol{B}_0\times\boldsymbol{r}\right) = \boldsymbol{B}_0
$$

- **第一步，写矢势**：$\boldsymbol{A} = \frac{1}{2}\boldsymbol{B}_0\times\boldsymbol{r}$。
- **第二步，取旋度**：利用矢量恒等式 $\nabla\times(\boldsymbol{B}_0\times\boldsymbol{r}) = \boldsymbol{B}_0(\nabla\cdot\boldsymbol{r}) - (\boldsymbol{B}_0\cdot\nabla)\boldsymbol{r}$（$\boldsymbol{B}_0$ 为常量，$\nabla\cdot\boldsymbol{r} = 3$）。
- **第三步，化简**：$\nabla\times\boldsymbol{A} = \frac{1}{2}[3\boldsymbol{B}_0 - \boldsymbol{B}_0] = \boldsymbol{B}_0$。
- **第四步，解读**：均匀磁场 $\boldsymbol{B}_0$ 的矢势可以写成 $\frac{1}{2}\boldsymbol{B}_0\times\boldsymbol{r}$——同一磁场对应无穷多矢势（加任一 $\nabla\chi$ 仍是）。矢势不是唯一的，这正是规范自由度。

## 5 规范变换的意义

- **简化方程**：选择合适规范使方程最简（库仑规范静态、洛伦兹规范辐射）；
- **推迟势**：洛伦兹规范下解达朗贝尔方程，得到推迟势（第 125 节）——辐射的理论基础；
- **量子力学耦合**：$\boldsymbol{p} \to \boldsymbol{p} - q\boldsymbol{A}$（最小耦合）——磁场对量子粒子的作用通过矢势；
- **规范场论**：电磁场是最简单的**规范场（gauge field）**——规范对称性（$U(1)$）是现代粒子物理（标准模型）的建构原则（第 112 节）。

**重点：规范变换是「规范对称性」的体现——物理不依赖于势的具体选择，这个对称性在现代物理中升华为「规范原理」：相互作用由规范对称性决定。** 电磁场是 $U(1)$ 规范场，弱力（SU(2)）、强力（SU(3)）都是非阿贝尔规范场——标准模型的骨架就是规范场论。<span class="marginnote">「从规范自由度到规范场论」：经典电动力学里的「势可以随便加 $\nabla\chi$」，在量子场论中成为「规范对称性要求引入规范玻色子（光子）」的建构原理。电磁、弱、强三种相互作用都由规范对称性产生——这是 20 世纪物理最深刻的统一思想之一。分析力学（哈密顿/拉格朗日）提供的框架，在这里与对称性思想合流。</span>

## 6 小结

- **矢势** $\boldsymbol{A}$：$\boldsymbol{B} = \nabla\times\boldsymbol{A}$；**标势** $\phi$：$\boldsymbol{E} = -\nabla\phi - \partial\boldsymbol{A}/\partial t$。
- **规范变换**：$\boldsymbol{A}' = \boldsymbol{A} + \nabla\chi$、$\phi' = \phi - \partial\chi/\partial t$——电磁场不变。
- **库仑规范**（$\nabla\cdot\boldsymbol{A} = 0$）：适合静态/近静态。
- **洛伦兹规范**：$\nabla\cdot\boldsymbol{A} + \frac{1}{c^2}\frac{\partial\phi}{\partial t} = 0$——势满足达朗贝尔方程（波动方程），辐射问题的标准。
- 规范自由度 ⟹ 规范对称性 ⟹ 规范场论（标准模型的建构原则）。

在下一节，我们研究电磁场的能量与动量流动——**电磁场的能量、动量与坡印廷矢量**。
