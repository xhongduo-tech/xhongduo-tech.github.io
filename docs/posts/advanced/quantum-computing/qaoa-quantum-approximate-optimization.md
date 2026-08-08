---
title: 量子近似优化算法（QAOA）
date: 2026-08-07
---

# 量子近似优化算法（QAOA）

<div class="epigraph">
<p>QAOA 是组合优化问题在 NISQ 时代的头号候选人。</p>
<footer>—— 法希（Edward Farhi）、戈德斯通（Jeffrey Goldstone）与古特曼（Sam Gutmann）</footer>
</div>

<div class="article-byline">
<p>第四级 · 量子计算 ｜ Farhi, Goldstone, Gutmann 2014（arXiv:1411.4028）｜ 2026-08-07</p>
</div>

## 为什么从 QAOA 开始

VQE 面向化学的连续能量景观，**QAOA（Quantum Approximate Optimization Algorithm，量子近似优化算法）** 则面向组合优化的离散世界——MaxCut、SAT、图着色这类 NP 难题。它把「近似优化」做成一个变分问题：用「问题哈密顿量 + 混合哈密顿量」交替的浅线路，逼近最优解的量子态。<span class="marginnote">QAOA 由 Farhi、Goldstone、Gutmann 在 2014 年提出，最初叫「量子优化算法」，后来以「近似」命名（它只求近似最优）。它与绝热计算有深刻血缘（它是绝热演化的离散化 + 变分化），也与 VQE 共享 VQA 母体。它的名气来自「用 NISQ 设备解真实组合优化」的诱惑。</span>本节讲它的线路、角度、与「为什么它可能是组合优化的答案」。

## 1 问题编码：MaxCut 为例

把组合优化问题编码成「成本哈密顿量」$C$：解的优劣映射成能量高低。以 **MaxCut**（把图切成两半，最大化被切边的数量）为例，设每个顶点一个比特（$0/1$ 表示两边），成本函数为

$$
C = \sum_{(i,j)\in E} \frac{1 - Z_i Z_j}{2}
$$

- 边 $(i,j)$ 被切（两端不同）时 $Z_iZ_j = -1$，该项贡献 $1$；同边时贡献 $0$。
- 最大化被切边数 = 最大化 $\langle C\rangle$。<span class="marginnote">这是「QUBO/Ising 编码」的标准动作：把组合优化翻译成「二值变量上的二次函数」，再映射成 Pauli $Z$ 算符。MaxCut、图着色、旅行商、调度等一大批问题都能这么编码——QAOA 的通用性来自这个「编码层」。求「最大成本」= 求「成本哈密顿量基态」（变符号即可）。</span>

## 2 QAOA 线路：两层交替

QAOA 的线路由 $p$ 层「问题演化 + 混合演化」组成：

$$
\lvert\gamma,\beta\rangle = e^{-i\beta_p B} e^{-i\gamma_p C} \cdots e^{-i\beta_1 B} e^{-i\gamma_1 C} \lvert+\rangle^{\otimes n}
$$

其中 $B = \sum_i X_i$ 是混合哈密顿量，$C$ 是问题成本哈密顿量，$(\gamma_1,\dots,\gamma_p,\beta_1,\dots,\beta_p)$ 是 $2p$ 个变分参数。<span class="marginnote">初始态 $\lvert+\rangle^{\otimes n}$（所有等幅叠加）。每层先演化「问题哈密顿量」$e^{-i\gamma C}$（时间 $\gamma$），再演化「混合哈密顿量」$e^{-i\beta B}$（时间 $\beta$）。$p$ 越大线路越深、表达力越强——「$p$ 层 QAOA」是它常用的复杂度指标。</span>

- **问题层** $e^{-i\gamma C}$：给「好解」的分量累积相位——它「偏置」搜索方向。
- **混合层** $e^{-i\beta B}$：在比特之间「扩散」振幅——它防止搜索困在局部。
- 两层交替 = 「改进 + 探索」的量子版迭代，参数由经典优化器调。

## 3 公式解析：为什么 QAOA 是绝热演化的「变分压缩」

QAOA 与**绝热演化**（见第零篇《绝热计算》）的血缘：绝热计算用 $H(s) = (1-s)B + sC$ 从易解基态演化到难解基态，需要长时间（能隙限制）。QAOA 把它离散化 + 参数化：

$$
e^{-i\beta B} e^{-i\gamma C} \approx e^{-i\Delta t(B + C)}
$$

- **第一步，Trotter 离散**：长绝热演化 $e^{-i\int H(s)ds}$ 用「小步 $B$ + 小步 $C$」的交替 Trotter 化逼近。
- **第二步，参数化**：每步的时间不是固定的小步，而是**自由参数** $(\gamma_i, \beta_i)$——优化器自由决定每步「在 $C$ 上待多久、在 $B$ 上待多久」。
- **第三步，变分压缩**：最优参数可能偏离「均匀 Trotter」很远——QAOA 找到的路径可能比线性绝热路径更短。这就是「变分化」的增益。<span class="marginnote">深层含义：QAOA 不一定按「绝热路径」走，它可以「抄近路」。$p=1$ 时 QAOA 等价于「一次改进 + 一次探索」，$p \to \infty$ 时（理想）逼近绝热/最优。变分参数给了它「逃离绝热限制」的自由度。</span>

## 4 公式解析：$p=1$ QAOA 的期望值

对 $p=1$ 的 MaxCut，期望成本可解析求值：

$$
\langle C\rangle = \frac12\sum_{(i,j)\in E}\big[1 - \sin 2\beta \sin 2\gamma\big] \qquad (\text{当图是二部图/正则图时的特例化简})
$$

- **第一步，展开**：把 $\langle+\vert e^{i\gamma C}e^{i\beta B} C e^{-i\beta B}e^{-i\gamma C}\vert+\rangle$ 按 $B$、$C$ 的交换关系展开。
- **第二步，单项贡献**：每条边的贡献分解成「无耦合项 + 边耦合项」，边耦合项含 $\sin 2\beta \sin 2\gamma$ 因子。
- **第三步，优化**：对 $(\gamma, \beta)$ 最大化，得到该图上 $p=1$ 的最佳近似比。<span class="marginnote">$p=1$ 的可解析性让它成为 QAOA 研究的「实验室」：对正则图，最佳近似比有闭式（如三正则图 $p=1$ 给近似比约 0.6924）。这为「QAOA 在浅深度下能做到什么」提供了精确答案，也是「$p$ 增加能改进多少」对比的基准。</span>

## 5 QAOA 的近似比与挑战

**近似比**：$p$ 有限时 QAOA 是「近似算法」——逼近最优解，但不保证最优。对某些问题（如 MaxCut 的某些图族），$p=1$ 能超过经典随机算法；对另一些，经典算法仍占优。
**著名结果**：对「大度正则图」的 MaxCut，QAOA 有理论下界（Hastings 2019）；但对一般图，$p$ 增大能否持续提升尚无定论。
**NISQ 困境**：$p$ 增大 → 线路加深 → 噪声淹没（$d\bar\epsilon<1$ 铁律）。所以「理论上的高 $p$ 优势」在 NISQ 上无法兑现。<span class="marginnote">诚实评估：QAOA 目前<strong>没有</strong>被证明在实用规模上超过最好的经典启发式（如模拟退火、专门 MaxCut 求解器）。它在「小图 + 浅深度」上的表现与经典持平或略优，但「规模化优势」仍是开放问题。QAOA 的价值更多是「测试平台」：它让研究者能系统研究「浅量子线路能近似组合优化到什么程度」。</span>

**辨析｜易错点：** QAOA 与「量子退火」（第十节）不是一回事。量子退火是**模拟**绝热演化的专用硬件（D-Wave），不做变分参数优化；QAOA 是**变分**算法，在通用量子计算机上跑、参数由经典优化。两者思想同源（绝热），但工程实现与理论性质不同。

## 6 小结

- **编码**：组合优化 → Ising/Pauli 成本哈密顿量 $C$（如 MaxCut 的 $\frac{1-Z_iZ_j}{2}$）。
- **线路**：$p$ 层「$e^{-i\gamma C}$（问题层）+ $e^{-i\beta B}$（混合层）」交替，初始态 $\lvert+\rangle^{\otimes n}$。
- **血缘**：QAOA 是绝热演化的「Trotter 离散 + 变分参数化」，可抄近路、不锁死绝热路径。
- **近似比**：$p=1$ 可解析；$p$ 增大理论有提升，但 NISQ 上被噪声挡在浅深度。
- **现状**：未证明实用规模超过经典启发式；是「浅量子线路表达力」的系统测试平台。

在下一节，我们面对 VQA 的致命暗礁——**成本函数景观与贫瘠高原（barren plateau）**。
