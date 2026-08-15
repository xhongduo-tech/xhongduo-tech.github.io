---
title: 虚功原理与达朗贝尔原理
date: 2026-08-07
---

# 虚功原理与达朗贝尔原理

<div class="epigraph">
<p>把「力的平衡」变成「虚功为零」——静力学换了一副眼镜；再把惯性力加进去，静力学就成了动力学。</p>
<footer>—— 分析力学引言</footer>
</div>

<div class="article-byline">
<p>第二级 · 高等物理 ｜ 周衍柏《理论力学》分析力学部分 ｜ 2026-08-07</p>
</div>

## 为什么从虚功原理开始

上一节我们认识了约束与广义坐标，并预告「理想约束力不做功」。这一节把这个思想变成两个原理：**虚功原理**把静力学（平衡问题）改写为「虚功为零」——约束力被自动消去；**达朗贝尔原理**用「惯性力」把动力学问题转化为「瞬时平衡」——把牛顿第二定律也纳入虚功框架。这两步是通往拉格朗日方程（下节）的桥。

## 1 虚位移与虚功

**虚位移（virtual displacement）** $\delta\boldsymbol{r}_i$：在某一瞬间、在约束允许的条件下，质点系各质点假想的无限小位移。它是「虚拟」的（不一定是真实发生的位移），用变分符号 $\delta$ 表示。

**虚功（virtual work）**：主动力在虚位移上做的功：

$$\delta W = \sum_i \boldsymbol{F}_i\cdot\delta\boldsymbol{r}_i$$

**理想约束（ideal constraint）**：约束力的虚功之和为零（$\sum\boldsymbol{N}_i\cdot\delta\boldsymbol{r}_i = 0$）——光滑接触、光滑铰链、不可伸长的绳等都是理想约束。

**重点：虚位移是「约束允许的假想位移」，虚功是主动力在其上的功。** 虚位移与真实位移不同：真实位移是「时间中实际发生的」，虚位移是「某一瞬间假想的、满足约束的」——$\delta t = 0$。理想约束下，约束力虚功为零，可完全忽略约束力。

## 2 虚功原理

**虚功原理（principle of virtual work）**：具有理想约束的质点系，处于**静平衡**的充要条件是：作用于系统的所有主动力在任何虚位移上的虚功之和为零：

$$\delta W = \sum_i \boldsymbol{F}_i\cdot\delta\boldsymbol{r}_i = 0$$

**重点：虚功原理把平衡问题化为「主动力虚功为零」——约束力自动消失。** 用广义坐标表达时，主动力的虚功 $\delta W = \sum_j Q_j\delta q_j$，其中 $Q_j$ 是**广义力**：

$$Q_j = \sum_i \boldsymbol{F}_i\cdot\frac{\partial\boldsymbol{r}_i}{\partial q_j}$$

平衡条件退化为：所有广义力为零（$Q_j = 0$，$j = 1, \dots, s$）。<span class="marginnote">「虚功原理的价值」：求平衡时不用解约束力——把「未知的约束力」从方程中消掉，只留主动力。经典应用：求复杂机构（连杆、滑轮组、杠杆）的平衡条件时，选广义坐标、算虚位移、令虚功为零，比逐点受力分析简洁得多。例如：用虚功原理可立即得出杠杆平衡条件（$F_1l_1 = F_2l_2$）。</span>

**广义力的速查**：常见主动力对应的广义力有现成表达式：

| 情形 | 广义力 $Q_j$ |
| --- | --- |
| 主动力有势 $V(\boldsymbol{r})$ | $Q_j = -\frac{\partial V}{\partial q_j}$ |
| 重力（$q_j$ 取高度 $z$） | $Q = -mg$ |
| 弹簧力（$q_j$ 取伸长 $x$） | $Q = -kx$ |
| 恒力（与位移夹角 $\theta$） | $Q = F\cos\theta$ |

**辨析｜易错点：**广义力不一定有「力的量纲」——若广义坐标是角度，广义力就是「力矩」。计算广义力的标准手段是「系数提取法」：让某一广义坐标变 $\delta q_j$、其余不动，写虚功 $\delta W = \sum_j Q_j\delta q_j$，则 $\delta q_j$ 前面的系数就是 $Q_j$。

## 3 达朗贝尔原理

对**动力学**问题，牛顿第二定律 $m\boldsymbol{a} = \boldsymbol{F}$。把惯性力 $(-m\boldsymbol{a})$ 视为一种「力」，则：

$$\boldsymbol{F} + (-m\boldsymbol{a}) = 0$$

即**达朗贝尔原理（d'Alembert's principle）**：在任意瞬间，作用于质点的主动力、约束力与惯性力构成「平衡力系」。把惯性力引入虚功：

$$\sum_i (\boldsymbol{F}_i - m_i\boldsymbol{a}_i)\cdot\delta\boldsymbol{r}_i = 0$$

**重点：达朗贝尔原理把动力学问题「化成静力学问题」——引入惯性力后，瞬时平衡成立，虚功原理适用于动力学。** 它让「平衡 + 虚功」的框架推广到运动系统，为拉格朗日方程提供了从「静」到「动」的桥梁。

**辨析｜易错点：**惯性力 $-m\boldsymbol{a}$ 不是真实存在的力，而是「形式上的假想力」——引入它的唯一目的，是把 $m\boldsymbol{a} = \boldsymbol{F}$ 改写成平衡形式以便套用虚功原理。这里的 $\boldsymbol{F}_i$ 是**主动力**（外加力），约束力已由理想约束假设消去。它与非惯性系中的惯性力（离心力、科里奥利力）是不同概念，不要混淆。

## 4 公式解析：用虚功原理求平衡

一杠杆（支点 O），主动力 $F_1$ 作用在臂长 $l_1$ 处、$F_2$ 作用在 $l_2$ 处（反向），求平衡条件。

$$
\delta W = F_1\delta x_1 + F_2\delta x_2 = F_1 l_1\delta\theta - F_2 l_2\delta\theta = 0
$$

- **第一步，选虚位移**：杠杆绕支点转小角度 $\delta\theta$，两端位移 $\delta x_1 = l_1\delta\theta$、$\delta x_2 = l_2\delta\theta$（方向相反，一个做正功、一个做负功）。
- **第二步，写虚功**：$\delta W = F_1\delta x_1 + F_2(-\delta x_2)$（$F_2$ 与位移反向）$= F_1l_1\delta\theta - F_2l_2\delta\theta$。
- **第三步，平衡条件**：$\delta W = 0$ ⟹ $F_1l_1 = F_2l_2$。
- **第四步，体会**：支点的约束力（铰链力）在虚功中不出现（支点无位移）——虚功原理直接给出杠杆平衡条件，无需受力分析。

**辨析｜易错点：**虚功原理中「主动力」的虚功——约束力已被排除（理想约束）。写虚功时要注意每个力的虚位移方向与做功正负；虚位移是「约束允许」的，不能随便给。平衡条件「所有广义力为零」是独立条件（每个广义坐标对应一个方程）。

**数值算例（定滑轮组）**：一动滑轮吊着重物 $G$，自由端用恒力 $F$ 上拉。取 $q$ = 重物上升高度，则重物虚位移 $\delta q$、自由端虚位移 $2\delta q$（动滑轮两股绳分担位移）。虚功 $\delta W = F\cdot 2\delta q - G\delta q = (2F - G)\delta q = 0$ ⟹ $2F = G$——**两股绳的滑轮组省一半力**。约束（绳不可伸长）把两个物体的位移耦合在一起，虚功原理自动处理这种耦合，无需对滑轮逐一受力分析。

## 5 从虚功到达朗贝尔到拉格朗日

达朗贝尔原理 + 广义坐标 + 动能表达式，可以推出**拉格朗日方程**（下节）。思路：

1. 达朗贝尔原理：$\sum(\boldsymbol{F}_i - m\boldsymbol{a}_i)\cdot\delta\boldsymbol{r}_i = 0$；
2. 用广义坐标展开 $\delta\boldsymbol{r}_i = \sum_j\frac{\partial\boldsymbol{r}_i}{\partial q_j}\delta q_j$；
3. 广义力 $Q_j$ 与动能 $T$ 关联：$-m\boldsymbol{a}_i\cdot\delta\boldsymbol{r}_i$ 部分化为 $T$ 的导数；
4. 若主动力有势（$Q_j = -\frac{\partial V}{\partial q_j}$），得到拉格朗日方程 $\frac{\mathrm{d}}{\mathrm{d}t}\frac{\partial L}{\partial\dot{q}_j} - \frac{\partial L}{\partial q_j} = 0$。

**重点：达朗贝尔原理是拉格朗日方程的出发点——「惯性力的虚功」在广义坐标下转化为动能 $T$ 的变分，配合势能 $V$ 得到拉格朗日量 $L = T - V$。** 下一节将走完这最后一步。<span class="marginnote">这条「从牛顿到达朗贝尔到拉格朗日」的路线是分析力学的标准叙事：牛顿（矢量力）→ 达朗贝尔（惯性力转平衡）→ 拉格朗日（广义坐标 + 能量）。每一步都在「更少地依赖力、更多地依赖能量/对称性」。哈密顿与最小作用量原理（第 117 节）则走向「极值原理」的更抽象层次。</span>

## 6 虚位移与自由度：约束的几何

**辨析｜易错点：虚位移的方向不是随便给的。** 虚位移必须与约束相容——系统有 $s$ 个自由度，虚位移就有 $s$ 个独立分量：$\delta\boldsymbol{r}_i = \sum_{j=1}^{s}\frac{\partial\boldsymbol{r}_i}{\partial q_j}\delta q_j$。

对定常约束（约束不显含时间），虚位移与真实位移都在同一约束曲面（切空间）内；对非定常约束（如绳长随时间变化的滑轮），虚位移**不沿**真实位移方向——因为虚位移要求 $\delta t = 0$。真实位移 $\mathrm{d}\boldsymbol{r}_i = \sum_j\frac{\partial\boldsymbol{r}_i}{\partial q_j}\mathrm{d}q_j + \frac{\partial\boldsymbol{r}_i}{\partial t}\mathrm{d}t$ 比虚位移多出 $\frac{\partial\boldsymbol{r}_i}{\partial t}\mathrm{d}t$ 这一项——**「虚」的本质就是「冻结时间后的假想位移」**。<span class="marginnote">「几何直觉」：约束把系统限制在一个约束流形上；虚位移是约束流形的切向量，广义坐标是流形上的坐标。达朗贝尔原理在切空间里消去约束力，正是「约束力垂直于约束流形」这一几何事实的代数表达。这个流形观点在第二级《微分几何》、以及哈密顿系统中会再次相遇。</span>

**自由度与独立虚位移算例**：单摆是 1 自由度（$q = \theta$）、平面上的自由质点是 2 自由度、刚性杆两端质点只有 5 自由度（6 坐标 − 1 约束）、刚体是 6 自由度。自由度减一，就少一个要解的方程——虚功原理的价值正在于「按自由度个数写方程，而不是按质点个数」：约束越多，这个优势越明显。

**从矢量力学到分析力学的路线总结**：

| 阶段 | 核心思想 | 关键方程 |
| --- | --- | --- |
| 牛顿力学 | 矢量力 + 受力分析 | $m\boldsymbol{a} = \boldsymbol{F}$ |
| 虚功原理 | 平衡 ⟺ 主动力虚功为零 | $\delta W = 0$ |
| 达朗贝尔原理 | 惯性力转瞬时平衡 | $\sum(\boldsymbol{F} - m\boldsymbol{a})\cdot\delta\boldsymbol{r} = 0$ |
| 拉格朗日方程 | 广义坐标 + 能量 | $\frac{\mathrm{d}}{\mathrm{d}t}\frac{\partial L}{\partial\dot{q}_j} - \frac{\partial L}{\partial q_j} = 0$ |

## 7 小结

- **虚位移**：约束允许的假想无限小位移（$\delta t = 0$）；**虚功**：主动力的虚功。
- **理想约束**：约束力虚功为零——约束力可忽略。
- **虚功原理**：静平衡 ⟺ 主动力虚功为零；广义坐标下即广义力 $Q_j = 0$。
- **达朗贝尔原理**：引入惯性力 $-m\boldsymbol{a}$，动力学化为瞬时平衡；$\sum(\boldsymbol{F}_i - m\boldsymbol{a}_i)\cdot\delta\boldsymbol{r}_i = 0$。
- **广义力**：系数提取法 $Q_j = \delta W/\delta q_j$；有势力 $Q_j = -\partial V/\partial q_j$。
- 虚位移是约束流形的切向量；自由度决定独立虚位移的个数。
- 达朗贝尔 + 广义坐标 + 能量 → 拉格朗日方程（下节）。

在下一节，我们走出分析力学的核心方程——**拉格朗日方程**。
