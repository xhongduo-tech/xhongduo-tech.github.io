---
title: 电磁场量子化
date: 2026-08-07
---

# 电磁场量子化

<div class="epigraph">
<p>在物理学中，你可以用许多不同语言描述同一个理论；规范理论的语言是把冗余当作朋友。</p>
<footer>—— 陈省身 谈规范场（Chern–Simons 思想的背景）</footer>
</div>

<div class="article-byline">
<p>第四级 · 量子场论 ｜ Peskin &amp; Schroeder《An Introduction to Quantum Field Theory》 §4.8, §5.5 ｜ 2026-08-07</p>
</div>

## 为什么电磁场最难量子化

前几节的标量场与狄拉克场各有清晰的动力学变量——$\phi$ 与 $\psi$ 的每个分量都是真物理自由度。**电磁场（光子场）$A^\mu$ 是另一个物种**：它由规范对称性统治。
$A^\mu$ 有 4 个分量，但光子只有 **2 个横极化**（左旋、右旋圆极化），纵向与类时分量是纯冗余——因为 $A_\mu \to A_\mu + \partial_\mu\alpha$ 不改变物理。
量子化这样「自带冗余」的场，需要先回答：**如何把 4 个分量里的 2 个「多余」的排除掉？** 答案是：靠**规范固定（gauge fixing）**。<span class="marginnote">类比：在球面上描述位置需要 3 个坐标，但球面本身只有 2 维。$A^\mu$ 的 4 个分量里有一维是「坐标重参数化」——规范变换就是这种重参数化，物理量在规范变换下不变。</span>

## 1 经典电磁场：拉格朗日量与规范对称性

从麦克斯韦方程组反推拉格朗日密度。用四维势 $A^\mu = (\phi, \boldsymbol{A})$ 定义**场强张量**：

$$F_{\mu\nu} = \partial_\mu A_\nu - \partial_\nu A_\mu$$

于是电磁场的作用量（没有源的自由场）取：

$$\mathcal{L} = -\frac14 F_{\mu\nu}F^{\mu\nu}, \qquad S = \int d^4x\left(-\frac14 F_{\mu\nu}F^{\mu\nu}\right)$$

对 $A_\mu$ 变分得 $\partial_\mu F^{\mu\nu} = 0$——这正是含 $\nabla\cdot\boldsymbol{E}=0$ 与 $\nabla\times\boldsymbol{B} = \partial_t\boldsymbol{E}$ 的麦克斯韦方程（无源部分）。**规范对称性**：对任意函数 $\alpha(x)$ 做变换

$$A_\mu \to A_\mu + \partial_\mu\alpha$$

$F_{\mu\nu}$ 的导数结构让两项的 $\partial_\mu\partial_\nu\alpha$ 相消，$\mathcal{L}$ 不变。<span class="marginnote">$F_{\mu\nu}$ 是「反对称导数」，它对 $A \to A + \partial\alpha$ 免疫：$\partial_\mu\partial_\nu\alpha - \partial_\nu\partial_\mu\alpha = 0$（偏导对易）。这就是规范不变性的代数根源——一个恒等式，不是深奥原理。</span>但麻烦也在此：因为 $\mathcal{L}$ 不含 $A^\mu$ 的显式形式，$A^0$ 的共轭动量 $\pi^0 = \partial\mathcal{L}/\partial\dot{A}_0 = F^{00} = 0$——**$A^0$ 没有动力学，它是一个「拉格朗日乘子」型的约束变量**。

## 2 规范固定：挑出一个代表

量子化前必须先「固定规范」，即从每个规范等价类里挑一个代表。两种常用选择各有利弊：

**库仑规范（Coulomb gauge）**：$\nabla\cdot\boldsymbol{A} = 0$。横向（横波）条件直接消掉纵向自由度，$A^0$ 被解耦（由泊松方程确定）。优点是**物理自由度一目了然**：只剩两个横极化。缺点：不显式洛伦兹协变。
**洛伦兹规范（Lorentz gauge）**：$\partial_\mu A^\mu = 0$。形式协变，但 4 个分量里仍有鬼态（负模方态）需要额外处理——这就是**Gupta–Bleuler 方案**：允许负模方态存在，但要求物理态被横条件 $k^\mu a_\mu|\psi\rangle = 0$ 挑选出来。<span class="marginnote">Gupta–Bleuler 的精神：<strong>负模方态可以存在，但不能被观测到</strong>。物理态由 $\partial^\mu A_\mu^+|\psi\rangle = 0$ 挑选，负模方分量对物理量的期望值恒为零。规范不变量只看到横极化——冗余最终没有泄漏进物理。</span>

本书的散射计算通常采用**费曼规范（Feynman gauge）**：在拉格朗日量里加一个规范固定项 $-\frac12(\partial^\mu A_\mu)^2$，让光子传播子变成简单的 $\frac{-ig_{\mu\nu}}{k^2+i\epsilon}$。代价是纵向/类时极化以「未物理态」身份出现，但规范不变的可观测量会排除它们。

## 3 光子场的量子化与两种极化

在库仑规范里量子化最直观：$\boldsymbol{A}$ 满足横波条件 $\nabla\cdot\boldsymbol{A}=0$，展开成两个横极化矢量 $\boldsymbol{\epsilon}^1(\boldsymbol{k}), \boldsymbol{\epsilon}^2(\boldsymbol{k})$，它们垂直于波矢 $\boldsymbol{k}$ 且互相正交：

$$\boldsymbol{A}(\boldsymbol{x},t) = \int \frac{d^3k}{(2\pi)^3}\frac{1}{\sqrt{2\omega_k}}\sum_{\lambda=1,2}\left( \boldsymbol{\epsilon}^\lambda(\boldsymbol{k})\, a_{\boldsymbol{k}}^\lambda e^{-ik\cdot x} + \text{h.c.} \right)$$

其中 $\omega_k = |\boldsymbol{k}|$（**光子质量为零，色散关系是线性光锥**）。产生/湮灭算符满足玻色对易关系：

$$[a_{\boldsymbol{k}}^\lambda, a_{\boldsymbol{k}'}^{\lambda'\dagger}] = (2\pi)^3\delta^{\lambda\lambda'}\delta^{(3)}(\boldsymbol{k}-\boldsymbol{k}')$$

光子是**自旋 1 的无质量玻色子**，只有两个螺旋度 $h = \pm 1$（沿运动方向的正/负圆极化）。
纵模与标量模要么被库仑规范消掉，要么被 Gupta–Bleuler 排除出物理空间。<span class="marginnote">为什么无质量自旋 1 只有两个极化而不是三个？因为无质量粒子没有静止系，螺旋度是洛伦兹不变量，而质量 $\neq 0$ 的矢量粒子（$W^\pm, Z$）有 3 个极化。光子没有 $h=0$ 分量，这是它「横波」本性的根源。</span>

## 4 公式解析：从 $F_{\mu\nu}$ 到麦克斯韦方程

**电磁场拉格朗日量 $-F^2/4$ 是「最小平方」原则的产物，它的变分把两条麦克斯韦方程一并交出。** 拆解三步：

$$
\mathcal{L} = -\frac14 F_{\mu\nu}F^{\mu\nu}, \qquad F_{\mu\nu} = \partial_\mu A_\nu - \partial_\nu A_\mu
$$

- **第一步，为什么是 $F^2$ 而不是 $A^2$**：$A^\mu$ 本身在规范变换下会变，任何「$A^2$ 型」质量项都会被规范变换破坏。能自动免疫规范变换的最低阶不变量是 $F_{\mu\nu}F^{\mu\nu}$。这正是**光子无质量**的场论解释：规范对称性禁止 $m^2 A^\mu A_\mu$ 项。质量只能靠第四章的希格斯机制「违规」获得。
- **第二步，变分出方程**：$\frac{\partial\mathcal{L}}{\partial(\partial_\mu A_\nu)} = -F^{\mu\nu}$（由 $F_{\mu\nu}$ 的反对称性），欧拉-拉格朗日方程给出 $\partial_\mu F^{\mu\nu} = 0$。对 $\nu=0$：$\nabla\cdot\boldsymbol{E} = 0$；对空间分量：$\nabla\times\boldsymbol{B} = \partial_t\boldsymbol{E}$。另两条（$\nabla\cdot\boldsymbol{B}=0$、$\nabla\times\boldsymbol{E}=-\partial_t\boldsymbol{B}$）由 $F_{\mu\nu}$ 的定义恒等成立——Bianchi 恒等式。
- **第三步，规范对称性如何「吃掉」两个自由度**：$A^\mu$ 有 4 个分量，但规范变换 $A_\mu \to A_\mu + \partial_\mu\alpha$ 是局域的（$\alpha(x)$ 有任意函数自由度），加上 $\pi^0 = 0$ 的约束，物理自由度 = 4 − 2 = 2，恰是两种横极化。

## 5 辨析｜易错点

- **$A^\mu$ 不是物理场、$F_{\mu\nu}$ 才是**：$A^\mu$ 随规范变，$F_{\mu\nu}$（即 $\boldsymbol{E},\boldsymbol{B}$）不变。所以「电子感受到 $A^\mu$」这种说法要小心——阿哈罗诺夫-玻姆效应里电子确实感受 $A^\mu$，但那是拓扑相位，可观测量的规范不变性仍然成立。<span class="marginnote"><strong>库仑规范 vs 洛伦兹规范</strong>：库仑规范优先保直觉（只剩横光子），洛伦兹/费曼规范优先保协变（传播子简单）。做量子化选择 §5.5 的 Gupta–Bleuler，做散射计算用费曼规范。两者物理等价。</span>
**把光子当成「一个 $A^\mu$ 场」**：$A^\mu$ 是 4 分量但物理自由度只有 2。若把 4 个分量都当真，会出现负概率的鬼态。物理计算里用「只有横极化参与求和」或「$\sum_\lambda \epsilon_\mu\epsilon_\nu$ = 投影算符」自动处理。
**$\omega_k = |\boldsymbol{k}|$ 与 $E_{\boldsymbol{p}} = \sqrt{p^2+m^2}$ 的差别**：光子色散是线性的（$m=0$），标量粒子是双曲的（$m\neq 0$）。无质量决定了两件事：光子速度恒为光速、光子只有两个极化。

## 6 延伸：规范冗余与量子化方案的谱系

电磁场量子化的「麻烦」——4 分量、2 物理自由度——催生了一整套处理规范冗余的工具，它们的取舍是理解文献的关键：

| 方案 | 规范条件 | 优点 | 代价 |
| --- | --- | --- | --- |
| 库仑规范 | $\nabla\cdot\boldsymbol{A}=0$ | 物理自由度直白 | 不协变 |
| 洛伦兹规范（Gupta–Bleuler） | $\partial_\mu A^\mu=0$ | 协变 | 有负模方鬼态 |
| 费曼规范（$R_\xi=1$） | $-\frac12(\partial^\mu A_\mu)^2$ 项 | 传播子最简单 | 需鬼态补偿 |
| 轴向规范 | $A^3=0$ | 无鬼态、无费曼参数 | 破坏部分协变 |

对计算者最重要的是：**物理量（S 矩阵元）不依赖规范选择**。你可以为方便挑费曼规范，为直觉挑库仑规范，结果一样。这背后是规范不变性的「金律」：中间态（纵向/类时光子）不是物理，物理只在规范不变的组合里。

还要记住：电磁场的「无质量」不是巧合而是规范对称的**必然**——质量项 $m^2 A_\mu A^\mu$ 破坏规范不变。后文希格斯机制会打破这条「禁止」，但那要付出引入新场的代价。

### 自测清单

[ ] 能写出 $F_{\mu\nu}$ 与规范变换，并解释为何 $F^2/4$ 不变。
[ ] 能说明库仑规范与洛伦兹规范的取舍。
[ ] 能解释光子为什么只有两个物理极化。
[ ] 能说出规范对称如何禁止光子质量。

<span class="marginnote">把这些方案当作「同一座山的不同登山路线」：<strong>路线可以不同，山顶（物理量）只有一个</strong>。做计算时先声明你走哪条路。</span>

### 延伸阅读指引

- 深化推导：P&S §5.5 的 Gupta–Bleuler 方案、§8.1 的泛函量子化；想理解规范固定与鬼场可读 §16.4 的 Faddeev–Popov。
- 实践：对比库仑规范与费曼规范算同一过程（如 $e^+e^-\to\gamma\gamma$），确认物理量一致。
- 联系主线：电磁场量子化是「处理冗余自由度」的范本——与《数据库》里的「消除冗余」、以及《信号处理》里的「去相关」是同一类工程问题：冗余要处理，物理信息要保留。

### 本节记忆锚点

- 规范不变：$F_{\mu\nu}$ 对 $A \to A + \partial\alpha$ 免疫；$A^2$ 项被禁止 → 光子无质量。
- 物理自由度：4 分量 − 2 冗余 = 2 个横极化。
- 规范方案：库仑（直觉）/ 洛伦兹 + Gupta–Bleuler（协变）/ 费曼（简单传播子）。
- 关键提醒：物理量规范无关，中间态不是物理。
- 交叉引用：与《电动力学》的规范、第四级《粒子物理》的光子章节对照。

## 7 小结

- 电磁场拉格朗日量 $\mathcal{L} = -\frac14 F_{\mu\nu}F^{\mu\nu}$，$F_{\mu\nu}$ 对规范变换免疫。
- **规范对称性禁止光子质量**；$A^0$ 无动力学、$A^\mu$ 有冗余。
- 量子化需**规范固定**：库仑规范（横条件直白）或洛伦兹规范 + Gupta–Bleuler（协变但带鬼态）。
- 光子展开用两个横极化 $\boldsymbol{\epsilon}^\lambda$，$\omega = |\boldsymbol{k}|$，玻色对易关系。
- 无质量自旋 1 只有两个螺旋度 $h = \pm 1$，没有 $h = 0$ 分量。

在下一节，我们把「场如何给出粒子」的两条路线并列起来——**费米与玻色统计**——看看对易与反对易如何从代数上决定量子统计，以及它在宏观现象（超导、激光）里的回声。


