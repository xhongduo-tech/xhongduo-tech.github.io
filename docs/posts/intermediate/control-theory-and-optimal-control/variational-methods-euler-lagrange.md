---
title: 变分法与最优控制（Euler-Lagrange 方程）
date: 2026-08-07
---

# 变分法与最优控制（Euler-Lagrange 方程）

<div class="epigraph">
<p>Nothing takes place in the world whose meaning is not that of some maximum or minimum.（世间发生的一切，其意义都寓于某种极大或极小之中。）</p>
<footer>—— 莱昂哈德 · 欧拉（Leonhard Euler，*Methodus inveniendi lineas curvas*，1744）</footer>
</div>

<div class="article-byline">
<p>第二级 · 控制论与最优控制 ｜ Kirk《Optimal Control Theory: An Introduction》Ch. 2 ｜ 2026-08-07</p>
</div>

## 为什么最优控制从变分法开始

前八篇我们追求的是「稳定」与「可达」，但工程师真正想要的是「**最好**」：同样的燃料，怎么飞得最省；同样的时间，怎么走最优美的轨道；同样的能耗，怎么把误差压到最小。控制理论的另一半江山——**最优控制（optimal control）**——就是给「最好」建立严格数学语言的一整套学科。<span class="marginnote">欧拉 1744 年那句宣言点出了最优控制的哲学根基：<strong>自然界的许多过程都在「极小化某量」</strong>——光线取最短时间路径（费马原理）、悬链线取最小势能、行星运动取最小作用量。最优控制就是把这些原理从「自然法则」变成「设计规范」。</span>

一切从**变分法（calculus of variations）**开始。微积分求的是「函数在哪取极值」，变分法求的是「**函数的函数（泛函）**在哪取极值」——你要找的不是一个数，而是一条曲线。最优控制恰好就是这类问题：在「系统动态」的约束下，找一条控制轨线使成本泛函最小。Kirk 的第 2 章正是从变分法切入最优控制的标准路径。

## 1 泛函：函数的函数

先建立对象。**泛函（functional）** 是「把函数映射到实数」的算子。最优控制里的典型成本泛函：

$$
J = \int_{t_0}^{t_f} L(x(t), u(t), t)\,\mathrm{d}t,
$$

其中 $L$ 是**拉格朗日型**积分成本（例如最小燃料 $\int |u| \,\mathrm{d}t$，最小能量 $\int u^2\,\mathrm{d}t$）。加上终端项就成了**波尔扎型（Bolza）**泛函：

$$
J = \phi(x(t_f), t_f) + \int_{t_0}^{t_f} L(x, u, t)\,\mathrm{d}t.
$$

问题就变成：**在所有允许的曲线里，找一条使 $J$ 取极小的。**<span class="marginnote">微积分里极值是对「点」求导；变分法里极值是对「曲线」求导——「变分」就是「曲线的一个无穷小扰动」。把求导对象从「数」换成「函数」，就是变分法相对于微积分的全部升级。</span>

**最简变分问题**：找曲线 $x(t)$ 使 $J = \int_{t_0}^{t_f} F(x, \dot{x}, t)\,\mathrm{d}t$ 最小，端点固定 $x(t_0) = x_0$，$x(t_f) = x_f$。这是所有变分问题的「样板间」。

## 2 变分与必要条件：把「曲线极值」翻译成微分方程

设最优曲线为 $x^*(t)$，考虑它的一个邻近扰动 $x^*(t) + \epsilon\eta(t)$（$\eta$ 是任意光滑函数且 $\eta(t_0) = \eta(t_f) = 0$）。于是 $J$ 成为 $\epsilon$ 的函数 $J(\epsilon)$，**极值条件为 $\frac{\mathrm{d}J}{\mathrm{d}\epsilon}\big|_{\epsilon=0} = 0$**。展开：

$$
\frac{\mathrm{d}J}{\mathrm{d}\epsilon} = \int_{t_0}^{t_f}\left( \frac{\partial F}{\partial x}\eta + \frac{\partial F}{\partial \dot{x}}\dot{\eta}\right)\mathrm{d}t.
$$

第二项含 $\dot\eta$，用**分部积分**消掉：

$$
\int_{t_0}^{t_f}\frac{\partial F}{\partial \dot{x}}\dot{\eta}\,\mathrm{d}t
= \left[\frac{\partial F}{\partial \dot{x}}\eta\right]_{t_0}^{t_f} - \int_{t_0}^{t_f}\frac{\mathrm{d}}{\mathrm{d}t}\frac{\partial F}{\partial \dot{x}}\,\eta\,\mathrm{d}t.
$$

边界项因 $\eta(t_0) = \eta(t_f) = 0$ 消失。代回并整理：

$$
\frac{\mathrm{d}J}{\mathrm{d}\epsilon} = \int_{t_0}^{t_f}\left( \frac{\partial F}{\partial x} - \frac{\mathrm{d}}{\mathrm{d}t}\frac{\partial F}{\partial \dot{x}} \right)\eta(t)\,\mathrm{d}t = 0.
$$

由于 $\eta$ 任意，**变分法基本引理**保证括号内必须恒为零，于是得到

$$
\boxed{\;\frac{\partial F}{\partial x} - \frac{\mathrm{d}}{\mathrm{d}t}\frac{\partial F}{\partial \dot{x}} = 0\;}
$$

——**Euler-Lagrange 方程**。<span class="marginnote">欧拉 1744 年得到这个方程（当时用于最速降线问题），拉格朗日 1755 年给出更优雅的推导。<strong>它的意义是把「在所有曲线里找最优」这个无限维问题，压成了一个（组）二阶常微分方程</strong>——无限维问题坍缩成有限维可解问题，这是整个变分法的第一推动。</span>

## 3 最优控制里的变分法：约束下的极值

上一节的样板间没有约束；最优控制则多了一层「系统动态」约束 $\dot{x} = f(x, u, t)$。用**拉格朗日乘子**（这里常记作 $\lambda(t)$ 或 $\psi(t)$，称**协态**）把约束吸收进泛函：

$$
J_a = \int_{t_0}^{t_f}\Big[ L(x, u, t) + \lambda^T\big(f(x,u,t) - \dot{x}\big)\Big]\mathrm{d}t.
$$

把 $\lambda$ 也当作变量做变分，得到三个必要条件：

1. **关于 $\lambda$ 的变分**：恢复约束方程 $\dot{x} = f(x, u, t)$；
2. **关于 $x$ 的变分**：**协态方程**

$$
\dot{\lambda}^T = -\frac{\partial L}{\partial x} - \lambda^T\frac{\partial f}{\partial x};
$$

3. **关于 $u$ 的变分**：**最优性条件** $\frac{\partial L}{\partial u} + \lambda^T\frac{\partial f}{\partial u} = 0$。

**注意这已经是一套「两点边界值问题」**：状态方程从 $t_0$ 的初始条件出发，协态方程要从终端条件反推——最优控制的数值求解（打靶法、配点法）全部围绕这套结构展开。<span class="marginnote">协态 $\lambda(t)$ 与拉格朗日乘子同构，但它有深刻的「影子价格」解释：<strong>$\lambda(t)$ 度量「状态的微小改变值多少钱」</strong>——沿着最优轨迹，协态是动态的边际价值。这个解释在经济学里的对应物，就是最优控制理论被广泛应用于增长理论、资产定价的原因。</span>

**横截条件**处理自由终端：若 $x(t_f)$ 自由，则终端条件 $\lambda(t_f) = \partial\phi/\partial x|_{t_f}$；若时间自由，还要加关于 $t_f$ 的条件。第 4 篇《变分法求解最优控制：横截条件与边界》专门展开过，这里只需记住：**边界条件怎么给，决定了边值问题是否适定**。

## 4 公式解析：Euler-Lagrange 方程是怎么「变」出来的

把整条推导压缩成四步，保证你合上书也能默写出来：

$$
\frac{\partial F}{\partial x} - \frac{\mathrm{d}}{\mathrm{d}t}\frac{\partial F}{\partial \dot{x}} = 0.
$$

- **第一步，写下泛函对扰动的导数**：$J(\epsilon) = \int F(x^*+\epsilon\eta, \dot{x}^*+\epsilon\dot\eta, t)\,\mathrm{d}t$，对 $\epsilon$ 求导得 $\int(\partial F/\partial x\,\eta + \partial F/\partial \dot x\,\dot\eta)\,\mathrm{d}t$。这一步相当于微积分里的「求导」。
- **第二步，分部积分清掉 $\dot\eta$**：把含 $\dot\eta$ 的项用分部积分改写，边界项在固定端点时为零。目的：让整个被积函数只含 $\eta$（而不是 $\eta$ 的导数），这样才方便提取「对任意 $\eta$ 成立」的条件。
- **第三步，变分法基本引理**：若 $\int g(t)\eta(t)\,\mathrm{d}t = 0$ 对所有光滑、端点为零的 $\eta$ 成立，则 $g \equiv 0$。于是括号内为恒零，EL 方程成立。
- **第四步，直觉复核**：EL 方程说「$F$ 对 $x$ 的偏导 = $F$ 对 $\dot{x}$ 的偏导沿时间的变化率」。它是**泛函的「梯度 = 0」**：$\partial F/\partial x$ 是「势能项的推力」，$\frac{\mathrm{d}}{\mathrm{d}t}\partial F/\partial \dot x$ 是「动能项的惯性」，两者平衡即最优。<span class="marginnote">把 EL 方程用于「最速降线」$F = \sqrt{(1+\dot y^2)/(2gy)}$，解出的曲线正是<strong>摆线</strong>——约翰 · 伯努利 1696 年提出、欧拉/拉格朗日/牛顿等竞相求解的名题。一条方程串起了一整段数学史。</span>

## 5 从变分法到最优控制：Kirk 的路线图

Kirk《Optimal Control Theory》第 2 章把变分法如何长成最优控制讲得很清楚，三座里程碑值得记住：

1. **最简变分问题**（无约束曲线极值）→ EL 方程；
2. **等式约束变分问题**（拉格朗日乘子）→ 乘子 = 协态，进入最优控制语境；
3. **Bolza 问题 + 动态约束** → 协态方程 + 最优性条件 + 横截条件，就是最优控制的「必要条件三件套」。

这三个台阶正好是本节前三节的顺序。**变分法提供「必要条件」，Pontryagin 极大值原理（第 10 篇）把它推广到「控制受限（bang-bang）」情形**——变分法要求 $u$ 可自由变化（光滑取极值），极大值原理允许 $u$ 落在约束集合的边界上。这条「从变分到极大值」的推广线，是理解最优控制内部结构的主干。<span class="marginnote">变分法的局限是「必要条件」：满足 EL 的曲线未必是最优（可能是鞍点或极大）。<strong>凸性/正则性条件负责把必要条件升级为充分条件</strong>——线性二次型问题（LQR）的凸性保证了全局最优，这是第 10 篇 LQR 特别「安全」的深层原因。</span>

## 6 小结

- **变分法**研究「函数的函数」的极值；最优控制是带动态约束的变分问题。
- 泛函 $J = \int L\,\mathrm{d}t$；固定端点最简变分问题由 **Euler-Lagrange 方程**求解。
- EL 方程推导三件套：**扰动 → 分部积分 → 基本引理**，把无限维极值压成常微分方程。
- 最优控制的必要条件：**状态方程 + 协态方程 + 最优性条件** + 横截条件，构成两点边值问题。
- 协态 $\lambda(t)$ 是动态的「影子价格」，衡量状态的边际价值。
- **变分法是必要条件**，Pontryagin 极大值原理把它推广到控制受限情形；凸性把必要条件升级为充分条件。

在下一节，变分法在「控制无约束」时够用，但现实中控制几乎总是有界的（油门有最大、方向舵有限位）。处理这种约束的利器是——**庞特里亚金极大值原理（最小时间/燃料问题）**。
