---
title: 加权 L² 空间与 ∂̄ 算子的闭性
date: 2026-08-07
---

# 加权 L² 空间与 ∂̄ 算子的闭性

<div class="epigraph">
<p>引入权函数，等于在希尔伯特空间中给每个点按上一架天平——这正是多复变全局理论的称量方式。</p>
<footer>—— 仿 拉尔斯 · 赫尔曼德（Lars Hörmander），《$\bar\partial$ 方程的 $L^2$ 估计》</footer>
</div>

<div class="article-byline">
<p>第二级 · 多复变函数论 ｜ Hörmander 第4章 ｜ 2026-08-07</p>
</div>

## 为什么从加权 L² 空间开始

第 3 篇组结束时，我们把一切存在性问题都归结为「$\bar\partial$ 方程可解」，而把「可解」的证明推给了「L² 估计」。现在是兑现承诺的时候。**加权 L² 空间**是 Hörmander 方法的工作台：我们在 $D$ 上引入一个**权函数** $\varphi$，用

$$
\|f\|^2_{\varphi} = \int_D |f|^2 e^{-\varphi} \, d\lambda
$$

定义范数，然后在加权希尔伯特空间里研究 $\bar\partial$ 算子的闭性与可解性。**权函数的妙处**：它能「压制」解在边界附近或无穷远附近的增长，让本不可积的方程在加权意义下变得可解。<span class="marginnote">为什么需要权？举个直觉：$\bar\partial u = g$ 的经典解常常在边界爆炸。乘一个快速增长的 $e^{-\varphi}$（即惩罚大 $|u|$）作为范数权重，可以逼迫解「缩回来」。Hörmander 的洞察：<strong>让权函数 $\varphi$ 任意选取，得到的是一族空间，而可解性可以在每个空间里分别建立</strong>——权是调节器，不是障碍。</span>

## 1 加权 L² 空间的定义

设 $D \subset \mathbb{C}^n$ 开集，$\varphi: D \to [-\infty, +\infty)$ 是**可测**函数。定义

$$
L^2_{(p,q)}(D, \varphi) = \left\{ u \in \Omega^{p,q}(D) : \int_D |u|^2 e^{-\varphi} \, d\lambda \lt  \infty \right\}
$$

其中 $|u|^2 = \sum_{I,J} |u_{IJ}|^2$（逐分量平方和），$d\lambda$ 是 Lebesgue 测度。<span class="marginnote">记号的约定：权 $e^{-\varphi}$ 使「$\varphi$ 大」的地方惩罚重、范数小。若 $\varphi \equiv 0$，退化为普通 $L^2$。很多作者写成 $\int |u|^2 e^{-\varphi}$ 或 $\int |u|^2 e^{-2\varphi}$——约定不同，本质相同（差一个常数倍），Hörmander 用 $e^{-\varphi}$。</span>

**物理直觉**：$e^{-\varphi}$ 像一个「温度分布」，在 $\varphi$ 大的区域（如边界附近）范数被压扁。解方程时，权函数把注意力集中在 $D$ 的内部。

## 2 $\bar\partial$ 算子的定义域与闭性

把 $\bar\partial$ 看成**无界算子**：从 $L^2_{(p,q)}(D,\varphi)$ 到 $L^2_{(p,q+1)}(D,\varphi)$，定义域

$$
\mathrm{Dom}(\bar\partial) = \{ u \in L^2_{(p,q)} : \bar\partial u \in L^2_{(p,q+1)} \;\text{（分布意义下）} \}
$$

**关键问题**：$\bar\partial$ 的像是否**闭**？因为方程 $\bar\partial u = g$ 的可解性（在 $L^2$ 意义下）需要「$g$ 正交于 $\ker \bar\partial^*$」，而这需要**闭值域定理**。<span class="marginnote">单复变中的直觉：无界算子的像不自动闭。例如微分算子 $d/dx$ 在 $C[0,1]$ 上的像不是闭的。$\bar\partial$ 的像是否闭，决定 $L^2$ 理论是否成立——Hörmander 证明：<strong>对任意权 $\varphi$，$\bar\partial$ 在 $L^2_{(p,q)}(D,\varphi)$ 上都是闭算子</strong>（这是经典结果，Kohn 与 Hörmander 的奠基贡献）。</span>

**闭性的意义**：由闭图像定理 + 值域定理，$\bar\partial$ 像闭 ⟹ 存在常数 $C$ 使对 $g \perp \ker\bar\partial^*$ 有「拟满射」。这给后续的 L² 估计提供了希尔伯特空间框架。

## 3 为什么闭性成立：先验估计的雏形

闭性的证明依赖一个**基本恒等式**（对光滑、紧支的 $u$）：

$$
\|\bar\partial u\|^2_\varphi + \|\bar\partial^*_\varphi u\|^2_\varphi = \int_D \sum_{j,k} \frac{\partial^2 \varphi}{\partial z_j \partial \bar z_k} u^j \bar u^k \, e^{-\varphi} \, d\lambda + \text{（边界项/紧支则消失）}
$$

其中 $\bar\partial^*_\varphi$ 是 $\bar\partial$ 在加权 $L^2$ 下的 **Hilbert 共轭**。这个恒等式把「$\bar\partial$ 与 $\bar\partial^*$ 的组合范数」与「权函数的复 Hessian」连起来。<span class="marginnote">这个恒等式就是「<strong>基本估计（fundamental estimate）</strong>」的雏形，也是下一节 L² 估计的起点。它说明：$u$ 的「全导数能量」$\|\bar\partial u\|^2 + \|\bar\partial^* u\|^2$ 由 $\varphi$ 的复 Hessian 控制——权函数的「凸度」（psh 性）直接决定算子的正则性。<strong>这解释了为什么 psh 函数在多复变里无处不在：它们是这个恒等式成立的前提。</strong></span>

由恒等式 + 对 $\varphi$ 的适当条件（如 $\varphi$ 足够光滑、$e^{-\varphi}$ 局部可积），可以证明 $\bar\partial$ 的定义域在「范数 $\|u\|^2 + \|\bar\partial u\|^2$」下完备，从而 $\bar\partial$ 闭。**详细推导留给下一节**，这里先立住「闭性是 L² 理论的地基」。

## 4 公式解析：加权共轭 $\bar\partial^*_\varphi$

先设 $\varphi$ 光滑。对 $f \in C_c^\infty(D)$（紧支光滑）、$g \in C_c^\infty$，分部积分给出

$$
\langle \bar\partial f, g \rangle_\varphi = \langle f, \bar\partial^*_\varphi g \rangle_\varphi, \qquad
\bar\partial^*_\varphi g = - e^{\varphi} \sum_j \frac{\partial}{\partial z_j}\left( e^{-\varphi} g \right)
$$

- **第一步，$\langle \cdot,\cdot\rangle_\varphi$ 的展开**：$\langle f, h \rangle_\varphi = \int f \bar h\, e^{-\varphi}$。共轭算子 $\bar\partial^*_\varphi$ 是「在加权内积意义下」的伴随，不是普通伴随——权因子 $e^{-\varphi}$ 改变了分部积分的边界项权重。
- **第二步，链式法则**：$\frac{\partial}{\partial z_j}(e^{-\varphi} g) = e^{-\varphi}\left(\frac{\partial g}{\partial z_j} - g\frac{\partial \varphi}{\partial z_j}\right)$，所以 $\bar\partial^*_\varphi g = -\frac{\partial g}{\partial z_j} + g \frac{\partial \varphi}{\partial z_j}$（求和）——**共轭 = 负的普通共轭 + 权的「一阶修正」**。
- **第三步，为什么这重要**：基本恒等式里的 $\bar\partial^*_\varphi$ 出现的正是这个「带权修正」的算子。权函数的复 Hessian $\partial\bar\partial\varphi$ 在恒等式中扮演「曲率」角色——$\varphi$ 越凸（psh），算子越「正」，估计越强。**权函数是这场分析的调节旋钮**。

## 5 辨析与延伸：加权 L² 的五个要点

**辨析 1：权函数的作用是「调节器」，不是「惩罚器」**。初学者常把 $e^{-\varphi}$ 看成「惩罚大函数值」的机制。更准确地说：权函数定义了一个**新的范数结构**，让解空间适合问题的边界行为。$\varphi$ 大处范数小，相当于「重视」那里的函数；$\varphi$ 小处范数大，相当于「稀释」那里的权重。<span class="marginnote">为什么权能救方程？因为 $\bar\partial u = g$ 在无权 $L^2$ 中可能无解（解在边界爆炸，不在 $L^2$ 中）；换一个快速增长的权，解的范数被「压住」，解便落入加权空间。权是「空间的选择」，不是「对函数的惩罚」。</span>

**辨析 2：$\varphi$ 需要什么条件**。加权理论中 $\varphi$ 至少局部可积、$e^{-\varphi}$ 局部可积；对正则性理论还要求 $\varphi$ 光滑。最关键的是：**$\varphi$ 是 psh（或强 psh）时，基本恒等式才给出正的下界**——这是 psh 函数在 L² 理论中的核心角色。

**辨析 3：闭性 ≠ 可解性**。$\bar\partial$ 的像闭（闭值域）是「值域定理」的前提，但闭性本身不保证「满射」。$L^2$ 理论的完整链条：闭值域 ⟹ 正交分解 ⟹ 存在性（当 $g \perp \ker\bar\partial^*$）。**闭性是第一步，不是最后一步**。

**辨析 4：为什么从 $(0,1)$ 形式开始**。$\bar\partial u = g$ 中 $u$ 是 $(0,q-1)$ 形式，$g$ 是 $(0,q)$ 形式。最基本的情形 $q=1$（$u$ 是函数）对应「$\bar\partial$ 可解出全纯函数的补」——这是 Cousin I、延拓定理的公共形式。**掌握 $q=1$，其余 $q$ 只是逐层推广**。

**误区清单**：

- **误区 1**：以为「权是惩罚项」。
  正解：权是范数结构的选择，调节解空间的「视角」。
- **误区 2**：以为「闭性 ⟹ 可解」。
  正解：闭性是前提，可解还需正交性条件。
- **误区 3**：以为「$\varphi$ 任意都可」。
  正解：psh（强 psh）是基本恒等式成立的关键。
- **误区 4**：以为「加权理论不需要泛函分析」。
  正解：Riesz 表示、闭值域定理是核心引擎。

**术语表**：

| 中文 | 英文 | 说明 |
| --- | --- | --- |
| 权函数 | weight | 范数调节器 $e^{-\varphi}$ |
| 加权范数 | weighted norm | $\|f\|_\varphi$ |
| 无界算子 | unbounded operator | 定义域非全空间 |
| 闭算子 | closed operator | 图像闭 |
| 共轭算子 | adjoint | $\bar\partial^*_\varphi$ |
| 基本恒等式 | fundamental identity | 范数 = 曲率 |

## 6 历史注记与知识树

**历史**：Hilbert 空间方法进入 $\bar\partial$ 理论始于 1950s（Kohn 的 $\bar\partial$-Neumann 问题）；Hörmander 1965 年把加权 L² 框架系统化，使存在性理论摆脱边界光滑性依赖。此后加权方法成为标准：Demailly 用其研究正性条件与乘理想层，Ohsawa–Takegoshi 的 $L^2$ 延拓定理是近代最重要的发展之一。

**知识树**：

- 向后：多重次调和函数（第 1 组末篇）——权函数的 psh 条件是核心。
- 向前：Hörmander L² 估计（本组第 20 篇）、强伪凸正则性（本组第 21 篇）。
- 横向：泛函分析的无界算子理论（第二级《泛函分析》）——闭值域、共轭算子。

**一句话记忆**：加权 L² = 用 psh 权「重设范数」，让 $\bar\partial$ 的闭性在好空间中成立——一切 L² 理论的起点。

## 7 小结

- **加权 L² 空间**：$\|f\|^2_\varphi = \int |f|^2 e^{-\varphi}$，权 $e^{-\varphi}$ 调节范数分布。
- **$\bar\partial$ 作为无界算子**：定义域为「分布意义下 $\bar\partial u \in L^2$」。
- **闭性**：$\bar\partial$ 在任意加权空间上是闭算子——L² 理论的地基。
- **基本恒等式**：$\|\bar\partial u\|^2 + \|\bar\partial^*_\varphi u\|^2$ 由 $\varphi$ 的复 Hessian 控制——psh 性与算子正则性的第一次合流。

在下一节，我们来到本专题的技术制高点：**Hörmander L² 估计与加权解存在性定理**——证明伪凸域上 $\bar\partial u = g$