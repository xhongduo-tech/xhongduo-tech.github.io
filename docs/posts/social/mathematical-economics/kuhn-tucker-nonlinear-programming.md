---
title: 库恩-塔克条件与非线性规划
date: 2026-08-07
---

# 库恩-塔克条件与非线性规划

<div class="epigraph">
<p>困难不在于接受新思想，而在于摆脱旧思想。</p>
<footer>—— 约翰 · 梅纳德 · 凯恩斯（John Maynard Keynes, "The difficulty lies, not in the new ideas, but in escaping from the old ones"）</footer>
</div>

<div class="article-byline">
<p>第八级 · 数量经济学（数理经济学） ｜ 蒋中一《数理经济学的基本方法》第13章 ｜ 2026-08-07</p>
</div>

## 为什么从库恩-塔克条件开始

上一课的拉格朗日乘子法有一个隐藏前提：约束是**等式**，而且最优解落在约束曲面内部。但现实中的约束几乎都是**不等式**：消费者「预算不超过 $m$」而不是「恰好花光 $m$」；工厂「产能最多 $K$ 件」而不是「必须生产 $K$ 件」；任何选择变量都不能为负。这些情形下，最优解可能出现在**角点**——某个商品不消费、某项投入不用——而角点上拉格朗日的一阶条件 $f_i = \lambda g_i$ 根本不成立。

**库恩-塔克条件（Kuhn-Tucker conditions，简称 KKT）**把拉格朗日方法推广到不等式约束与非负约束，是现代经济学、运筹学与机器学习的共同地基。<span class="marginnote">这段历史常被压缩成两个名字：卡鲁什（William Karush）在 1939 年硕士论文里先给出结论，库恩（Harold Kuhn）与塔克（Albert Tucker）在 1951 年独立重新发现并系统发表，所以完整称呼是 <strong>KKT（Karush-Kuhn-Tucker）条件</strong>。机器学习里「支持向量机对偶」「带约束的梯度下降」用的都是这套条件。</span>

处理的问题统称**非线性规划（nonlinear programming）**：目标与约束至少有一方不是线性的。

学习路径上，这一课正好卡在承前启后的位置：它把第 3 篇的拉格朗日方法从等式解放到不等式，又为第 5 篇的凹规划提供了「什么条件下必要条件能升级成充分条件」的追问。

## 1 从等式到不等式：束紧与松弛

把等式约束 $g(x) = c$ 改成不等式 $g(x) \le c$，最优解在约束面前有两种态度：

- **束紧（binding / active）**：$g(x^*) = c$，约束真正「卡住」了选择。此时约束像等式一样起作用，影子价格 $\lambda > 0$。
- **松弛（slack / inactive）**：$g(x^*) \lt  c$，约束没有卡住最优解。此时再多给一单位「额度」也无济于事，影子价格必须为 $\lambda = 0$。

把这两种情况合成一句话，就是**互补松弛（complementary slackness）**：

$$\lambda_i \big(c_i - g_i(x)\big) = 0, \qquad \lambda_i \ge 0$$

**互补松弛读作：约束要么束紧（括号内为 0），要么影子价格为零——两者至少有一个是 0，不允许「既没用满约束、又对它估价」的荒唐状态。** 这是不等式约束与等式约束最本质的差别。<span class="marginnote">互补松弛很像「按需付费」的逻辑：用满的资源才值钱（$\lambda>0$），闲置的资源一文不值（$\lambda=0$）。政府想判断哪条政策真正起约束作用，看哪条约束的 $\lambda$ 非零就知道。</span>

技术上常把不等式改写成等式来便于计算：引入**松弛变量（slack variable）** $s_i \ge 0$，令 $g_i(x) + s_i = c_i$。束紧等价于 $s_i = 0$，松弛等价于 $s_i > 0$，互补松弛则变成更对称的 $\lambda_i s_i = 0$。<span class="marginnote">这种「补零法」让求解器只需处理等式系统加非负约束——现代非线性规划软件（如 SciPy 的 SLSQP、Gurobi 的内部算法）都在背后做这类转换。KKT 因此既是理论判据，也是算法的收敛目标。</span>

## 2 库恩-塔克条件：完整框架

标准形式的非线性规划是：

$$\max_x f(x_1, \dots, x_n) \quad \text{s.t.} \quad g_i(x_1, \dots, x_n) \le c_i \ (i = 1,\dots,m), \qquad x_j \ge 0 \ (j = 1,\dots,n)$$

照抄拉格朗日的配方，构造

$$L(x, \lambda) = f(x) + \sum_{i=1}^{m} \lambda_i\big(c_i - g_i(x)\big)$$

先做一个总体对照，看清拉格朗日方法与 KKT 的继承与扩张：

| 项目 | 拉格朗日（等式约束） | KKT（不等式约束） |
| --- | --- | --- |
| 约束写法 | $g(x) = c$ | $g(x) \le c$，且 $x \ge 0$ |
| 乘子符号 | $\lambda$ 可正可负 | $\lambda \ge 0$ 非负 |
| 最优形态 | 约束曲面上的内点 | 内点或角点 |
| 一阶条件 | $\partial L/\partial x = 0$ | $\partial L/\partial x \le 0$ 且 $x \cdot \partial L/\partial x = 0$ |

**KKT 条件**由两组互补松弛组成：

$$\frac{\partial L}{\partial x_j} \le 0, \quad x_j \ge 0, \quad x_j \cdot \frac{\partial L}{\partial x_j} = 0 \qquad (j = 1, \dots, n)$$

$$\frac{\partial L}{\partial \lambda_i} = c_i - g_i(x) \ge 0, \quad \lambda_i \ge 0, \quad \lambda_i \cdot \big(c_i - g_i(x)\big) = 0 \qquad (i = 1, \dots, m)$$

**这组条件的经济读法**：第一组说「每个选择变量的边际净收益要么是零（内点）、要么是负的（角点，已到边界）」，第二组说「每条约束要么用满（束紧）、要么影子价格为零（松弛）」。<span class="marginnote">对比上一课的拉格朗日：那里是 $\partial L/\partial x_j = 0$（无非负约束），KKT 把它放宽为「$\le 0$ 或 $=0$ 且互补」。等式的世界只有一种最优，不等式的世界有两种——这正是「摆脱旧思想」的数学化。</span>

## 3 公式解析：非负约束与 $x \frac{\partial L}{\partial x} = 0$

第一组条件里最陌生的，是 $x_j \ge 0$ 与 $x_j \cdot \partial L/\partial x_j = 0$。把它拆成两步理解：

- **第一步，为什么 $\partial L/\partial x_j \le 0$ 而非 $= 0$**：若 $x_j$ 被限制为非负，那么最优解只能向右增大，不能向左减小。在最优处，$x_j$ 每多一单位的**净收益**（即 $\partial L/\partial x_j$）必须非正——否则增大 $x_j$ 还能改善目标，就还不是最优。
- **第二步，互补松弛区分内点与角点**：

$$x_j > 0 \;\Longrightarrow\; \frac{\partial L}{\partial x_j} = 0, \qquad x_j = 0 \;\Longrightarrow\; \frac{\partial L}{\partial x_j} \le 0$$

内点解退化为熟悉的拉格朗日条件；角点解则要求「在边界上，增加该变量无利可图」。**一个乘积为零的等式，把「角点」和「内点」两种情况压缩成了同一条规则。**

## 4 数值例：从内点到角点

先用一个内点例子热身。消费者最大化 $f = \ln x_1 + \ln x_2$，受预算 $x_1 + 2x_2 \le 12$ 与 $x_1, x_2 \ge 0$ 约束。拉格朗日函数 $L = \ln x_1 + \ln x_2 + \lambda(12 - x_1 - 2x_2)$，一阶条件：

$$\frac{1}{x_1} - \lambda = 0, \qquad \frac{1}{x_2} - 2\lambda = 0 \;\Rightarrow\; \frac{1}{x_1} = \frac{1}{2x_2} \;\Rightarrow\; x_1 = 2x_2$$

约束束紧：$x_1 + 2x_2 = 12 \Rightarrow x_2 = 3$，$x_1 = 6$，影子价格 $\lambda = \tfrac{1}{6}$。互补松弛自动满足（$x_1,x_2 > 0$ 且 $12 - 12 = 0$）。<span class="marginnote">这里对数效用保证了内点解——$\ln$ 在 0 处趋向 $-\infty$，消费者绝不会让任何商品为零。若换成效用 $f = 4x_1 + 3x_2$ 的线性函数，角点就来了。</span>

再看线性目标 $f = 4x_1 + 3x_2$，同一预算约束 $x_1 + 2x_2 \le 12$。沿着束紧约束 $x_2 = 6 - \tfrac{x_1}{2}$，目标 $f = 18 + \tfrac{5}{2}x_1$ 随 $x_1$ 单调上升，所以 $x_1$ 越大越好：最优解是角点 $(x_1, x_2) = (12, 0)$。验证 KKT：

$$\frac{\partial L}{\partial x_1} = 4 - \lambda = 0 \Rightarrow \lambda = 4, \qquad \frac{\partial L}{\partial x_2} = 3 - 2\lambda = -5 \le 0$$

$x_2 = 0$ 且 $\partial L/\partial x_2 = -5 \lt  0$：**在最优处，多买一单位 $x_2$ 的净收益是 $-5$，所以它被推到零——角点解由此被 KKT 干净地捕捉。** 这正是拉格朗日方法做不到、而 KKT 专门解决的场景。

再往同一问题加一条松弛约束 $x_2 \le 10$，看影子价格如何归零。在最优解 $(x_1, x_2) = (12, 0)$ 处，$x_2 = 0 \le 10$，这条约束**没卡住任何东西**。KKT 要求 $\lambda_3 \ge 0$ 且 $\lambda_3(10 - 0) = 0$，于是被迫取 $\lambda_3 = 0$——**一条松弛的约束，影子价格必定为零**。<span class="marginnote">这就是互补松弛的「账本」视角：每条约束都是一本账，束紧的账才有余额（$\lambda>0$），松弛的账余额必为零。多约束时哪些 $\lambda$ 非零，一眼就能看出哪些政策真正具有约束力。</span>

把松弛约束 $x_2 \le 10$ 换成束紧的产能上限 $x_1 \le 4$，问题立即变成另一个世界：最优解改到 $(x_1, x_2) = (4, 4)$，目标从 48 降到 28。此时两条约束 $x_1 + 2x_2 \le 12$ 与 $x_1 \le 4$ 全部束紧，两个乘子分别满足 $4 - \lambda_1 - \lambda_2 = 0$ 与 $3 - 2\lambda_1 = 0$，解得 $\lambda_1 = 1.5$、$\lambda_2 = 2.5$——**预算的影子价格是 1.5，产能上限的影子价格是 2.5，产能比预算更「值钱」，因为它把目标从 48 压到了 28。**

## 5 什么时候 KKT 充分：约束规格与凹性

KKT 是**必要条件**：任何满足约束的内点或角点最优解都必然满足它。但反过来，「满足 KKT 的点一定最优」需要额外条件，否则会误认鞍点。两个关键条件：

**约束规格（constraint qualification）**：约束在最优点的梯度必须「足够独立」，典型如斯莱特条件——存在一个可行点使所有不等式严格成立。它防止约束「病态缠绕」时出现假 KKT 点。<span class="marginnote">对只含线性约束或单个不等式约束的问题，约束规格自动满足，这也是多数入门模型不用操心它的原因。</span>

**凹性（concavity）**：若目标函数 $f$ 是凹函数、可行集是凸集，则任何 KKT 点都是全局最优解——必要条件自动升级为充分条件。

**辨析｜易错点：** 目标非凹时，KKT 点可能是局部极大、极小或鞍点，需要逐点甄别。这正是下一课《凹规划与二阶条件》的主角：凹函数把「局部最优」直接送上「全局最优」，让最优化理论第一次有了不需要逐个检查角点的干净结论。

## 6 小结

- 不等式约束用**束紧/松弛**刻画，互补松弛 $\lambda_i(c_i - g_i) = 0$ 与 $\lambda_i \ge 0$ 是核心。
- KKT 条件 = 两组互补松弛：选择变量的非负约束 + 约束本身的非负乘子。
- 角点解由 $x_j \cdot \partial L/\partial x_j = 0$ 捕捉：$x_j = 0$ 时要求边际净收益非正。
- 束紧约束影子价格为正，松弛约束影子价格必为零——互补松弛是「账本」。
- 松弛变量 $s_i$