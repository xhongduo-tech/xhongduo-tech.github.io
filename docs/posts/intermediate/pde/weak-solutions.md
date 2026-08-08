---
title: 弱解（广义解）的定义
date: 2026-08-08
---

# 弱解（广义解）的定义

<div class="epigraph">
<p>不求逐点满足方程，只求在积分意义下与一切探针一致——这就是弱解。</p>
<footer>—— 弱形式（weak formulation）</footer>
</div>

<div class="article-byline">
<p>第二级 · 偏微分方程 ｜ 姜礼尚《数学物理方程讲义》第八章 ｜ 2026-08-08</p>
</div>

## 为什么从弱解开始

经典解要求函数 $C^2$ 且在每点满足方程——这对很多实际问题太苛刻：初值可以粗糙（热传导瞬时磨平之前）、系数可以不光滑、区域可以有角。**弱解**把「解」的定义放宽到「乘测试函数积分后满足方程」——导数的要求通过分部积分转移到测试函数上，从而对解本身只要求「一次可导、平方可积」（$H^1$）。这一节给出弱解的精确定义、与经典解的关系，以及 Lax–Milgram 定理这把存在性/唯一性的重锤。

## 1 从强形式到弱形式

经典（强）形式的 Dirichlet 问题：

$$
-\Delta u = f \ \text{在 } \Omega, \qquad u|_{\partial\Omega} = 0
$$

**推导弱形式**：乘测试函数 $v \in C_0^\infty(\Omega)$，在 $\Omega$ 上积分，分部积分把 $\Delta$ 从 $u$ 移到 $v$ 上：

$$
-\int_\Omega(\Delta u)v\,dx = \int_\Omega\nabla u\cdot\nabla v\,dx \quad（v \text{ 支集紧，边界项为零}）
$$

得

$$
\int_\Omega\nabla u\cdot\nabla v\,dx = \int_\Omega f\,v\,dx, \qquad \forall v \in C_0^\infty(\Omega)
$$

**注意：这个形式里只有一阶导数 $\nabla u$——对 $u$ 的要求从 $C^2$ 降到了「一阶导数平方可积」（$H^1$）。**<span class="marginnote">「分部积分把导数转移给测试函数」是弱形式的一切秘密。测试函数 $v$ 光滑（多少阶导数都行），所以方程对 $u$ 的「最小要求」只来自「做完分部积分后 $u$ 还剩几阶导」。这正是第九篇 Sobolev 空间「用广义导数定义可导阶数」的用武之地。</span>

**为什么只在测试函数上验证？** 由变分法基本引理，若积分等式对所有 $v \in C_0^\infty$ 成立，则 $-\Delta u - f = 0$（在分布意义下）——强形式与弱形式在「够光滑」时等价。

## 2 弱解的定义

**弱解（weak solution）**：$u \in H_0^1(\Omega)$ 称为 $-\Delta u = f$（Dirichlet）的弱解，若

$$
\boxed{\;\int_\Omega\nabla u\cdot\nabla v\,dx = \int_\Omega f\,v\,dx, \qquad \forall v \in H_0^1(\Omega)\;}
$$

（用 $H_0^1$ 代替 $C_0^\infty$ 作为测试函数空间，同样成立——测试函数空间越大，约束越强，$H_0^1$ 是「够用且完备」的选择。）

**弱解三个要件**：$u \in H_0^1$（正则性 + 边界）、对一切 $v \in H_0^1$ 成立（方程意义）、积分形式（弱导数）。

**与变分的联系**：弱解正是上一节能量泛函 $J[v] = \frac12\int|\nabla v|^2 - \int fv$ 的极小点（一阶变分为零的方程就是弱形式）。**弱解 = 变分问题的解 = 能量极小点**——三个名字，同一个对象。

## 3 经典解、弱解与正则性

**弱解与经典解的关系**：

| | 经典解 | 弱解 |
| --- | --- | --- |
| 正则性 | $C^2(\Omega) \cap C(\bar\Omega)$ | $H^1(\Omega)$ |
| 方程 | 逐点 $-\Delta u = f$ | 积分 $\int\nabla u\cdot\nabla v = \int fv$ |
| 边界 | 逐点 $u|_{\partial\Omega} = g$ | 迹意义 $u|_{\partial\Omega} = g$ |

**关系**：经典解必是弱解（分部积分）；弱解在数据光滑时「自动」是经典解——**椭圆正则性定理**：若 $f$ 光滑、$\Omega$ 光滑，则 $u \in H^1$ 的弱解自动 $C^\infty$。<span class="marginnote">正则性定理是「从弱到强」的跃迁：弱解看似只保证 $H^1$，但只要数据好，解就好——这由第九篇的嵌入定理与椭圆正则性估计（「$u \in H^s \Rightarrow u \in H^{s+2}$」的升阶引理）逐级推出。「先证弱解存在，再证它其实很光滑」是现代 PDE 的标准叙事。</span>

**弱解的意义**：它把「存在性」与「光滑性」解耦——先在一个宽泛的空间里证明存在，再在条件具备时提升光滑性。经典解法常常在这两步之间卡壳；弱解让第一步变得可行。

## 4 Lax–Milgram 定理

弱解存在性/唯一性的标准工具是 **Lax–Milgram 定理**：

**Lax–Milgram**：设 $H$ 是 Hilbert 空间，$B(\cdot,\cdot): H\times H \to \mathbb{R}$ 是双线性型，满足**有界性** $|B(u,v)| \le C\|u\|\|v\|$ 与**强制性（coercivity）** $B(u,u) \ge \alpha\|u\|^2$（$\alpha > 0$），则对任何有界线性泛函 $F \in H'$，方程 $B(u,v) = F(v)$ 有唯一解 $u \in H$，且 $\|u\| \le \frac{\|F\|}{\alpha}$。

**应用到 Poisson 方程**：取 $H = H_0^1$、$B(u,v) = \int\nabla u\cdot\nabla v\,dx$、$F(v) = \int fv\,dx$。

**有界性**：Cauchy–Schwarz $|\int\nabla u\cdot\nabla v| \le \|u\|_{H^1}\|v\|_{H^1}$ ✓；
**强制性**：$B(u,u) = \int|\nabla u|^2dx \ge \frac{1}{2}\|u\|_{H^1}^2$（由 Poincaré 不等式，第九篇）✓；
**结论**：弱解唯一存在，且 $\|u\|_{H^1} \le C\|f\|_{L^2}$——**稳定性估计一并到手**。

<span class="marginnote">Lax–Milgram 是变分法的「工厂流水线」：只要有界 + 强制，存在、唯一、稳定全自动。对比第六篇极值原理给出的存在性（需显式构造），Lax–Milgram 只要验证两个不等式——这是「抽象方法」的胜利。它是现代有限元分析的核心工具，后两节的 Ritz/Galerkin 误差分析都建立在它之上。</span>

## 5 弱解在三大方程中的形态

弱解概念对三大方程都能建立：

| 方程 | 弱形式 | 解空间 |
| --- | --- | --- |
| 椭圆 $-\Delta u = f$ | $\int\nabla u\cdot\nabla v = \int fv$ | $H^1$（稳态） |
| 热 $u_t - \Delta u = f$ | $\int u_t v + \int\nabla u\cdot\nabla v = \int fv$ | $L^2(0,T;H^1)$ |
| 波 $u_{tt} - \Delta u = f$ | $\int u_{tt}v + \int\nabla u\cdot\nabla v = \int fv$ | $C([0,T];H^1)$ |

**弱形式统一处理三类方程**——时间导数不动（仍是一阶/二阶），只有空间导数被分部积分。这也解释了为什么有限元（下下节）能同时处理椭圆、抛物、双曲问题：它们共享同一套弱形式框架。

**辨析｜易错点：** 弱解只在「几乎处处」意义下定义（$H^1$ 是 $L^p$ 等价类）——「$u(x)$ 在某点等于多少」在弱解框架里没有逐点意义，除非正则性定理提升了它。另外，**弱解不满足「逐点方程」**：对 $f$ 很粗糙（如 $\delta$）时，「解」可能不是函数而是分布。把「弱解」误读成「弱一点但逐点成立」是常见误区。

## 6 小结

- 弱形式：分部积分把 $\Delta$ 转移到测试函数，$\int\nabla u\cdot\nabla v = \int fv$。
- 弱解：$u \in H_0^1$ 使弱形式对所有 $v \in H_0^1$ 成立。
- 弱解 = 变分问题的极小点 = 能量极小点，三者是同一对象。
- 椭圆正则性：数据光滑 ⇒ 弱解自动 $C^\infty$。
- Lax–Milgram：有界 + 强制 ⇒ 存在、唯一、稳定，是现代有限元分析的基石。

在下一节，我们学习里兹（Ritz）方法。
