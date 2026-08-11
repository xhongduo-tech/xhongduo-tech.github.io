---
title: 凸分析与变分问题的对偶理论：Legendre-Fenchel 变换与 Fenchel 对偶
date: 2026-08-11
---

# 凸分析与变分问题的对偶理论：Legendre-Fenchel 变换与 Fenchel 对偶

<div class="epigraph">
<p>凸性的语言，是变分法在二十世纪学会说的第二种语言。</p>
<footer>—— 对 Ekeland–Témam 纲领的转述</footer>
</div>

<div class="article-byline">
<p>第二级 · 进阶数理 · 变分法 ｜ 对标教材 ｜ 2026-08-11</p>
</div>

## 为什么从凸分析与对偶理论开始

回头望整个专题，一条暗线反复出现：等周问题里的 $E \ge 0$、强极小的凸性、直接方法的下半连续性、H-J 方程里的 Legendre 变换——它们都是**凸性**在不同场合的露面。这一课把这层地基彻底掀开：用**凸分析**的精确语言（共轭函数、指示函数、Fenchel 对偶）把所有变分概念翻译成一个统一框架。<span class="marginnote">变分法两个世纪的语言演变：18–19 世纪用「微分 + 变分」；20 世纪 Ekeland–Témam 用「凸性 + 对偶」重写了整个学科——本专题《凸分析与变分问题的对偶理论》即依据 Ekeland–Témam Ch. III–IV。</span>

这不仅是整理遗产。对偶理论把「极小化问题」翻成「极大化问题」，两个问题共享同一个最优值——这带来计算与理论的双重自由。它也解释了为何 H-J 方程里的哈密顿量恰好是拉格朗日量的共轭（上一课），并把变分问题与优化、最优传输、机器学习里的对偶方法焊成一体。<span class="marginnote">对「从极限到大模型」的读者：支持向量机的对偶、GAN 的极小极大、强化学习的 Lagrangian、最优传输的 Kantorovich 对偶——全是这一课思想的工业级应用。见第三级《凸优化》《生成模型》。</span>

## 1 凸函数与指示函数

先建语言。定义域取扩充实数值 $\mathbb{R} \cup \{+\infty\}$，方便用 $+\infty$ 表达「不允许」。

**凸函数（convex function）**：$f: \mathbb{R}^n \to \mathbb{R} \cup \{+\infty\}$ 满足

$$
f(\lambda x + (1-\lambda)y) \le \lambda f(x) + (1-\lambda) f(y), \qquad 0 \le \lambda \le 1
$$

**正常凸函数（proper convex）**：$f$ 在某个点取有限值，且处处不等于 $-\infty$。

**指示函数（indicator function）**：对集合 $A$，

$$
\delta_A(x) = \begin{cases} 0, & x \in A \\ +\infty, & x \notin A \end{cases}
$$

指示函数是凸分析的「约束转换器」：把「$x$ 必须落在 $A$ 里」写成「$x$ 必须让 $\delta_A$ 有限」。于是带约束的极小化

$$
\min_{x \in A} f(x) \quad\Longleftrightarrow\quad \min_x \bigl[f(x) + \delta_A(x)\bigr]
$$

约束被并入目标函数。<span class="marginnote">这个「把硬约束变成 $+\infty$ 惩罚」的小把戏，是所有对偶理论的起点：等周问题里的 Lagrange 乘子、机器学习里的罚函数、强化学习里的约束优化，都是它的不同版本。</span>

## 2 Legendre-Fenchel 变换

**共轭函数（conjugate function，Legendre-Fenchel 变换）**：

$$
\boxed{\;f^*(y) = \sup_{x \in \mathbb{R}^n} \bigl[\langle x, y\rangle - f(x)\bigr]\;}
$$

几何直觉：$f^*$ 在点 $y$ 的值，是「斜率为 $y$ 的直线 $x \mapsto \langle x,y \rangle - c$ 能塞进 $f$ 下方的最高截距 $c$」——即用斜率 $y$ 支撑 $f$ 的支撑超平面的位置。<span class="marginnote">$f^*$ 把「$f$ 在每个点的值」换成「$f$ 在每个斜率处被支撑的高度」。光滑凸函数之间这是经典 Legendre 变换（$y = f'(x)$）；对一般凸函数用 $\sup$ 定义，保证共轭对一切（可能不光滑的）凸函数都良定义。</span>

几个立刻能算的例子：

| $f(x)$ | $f^*(y)$ | 备注 |
| --- | --- | --- |
| $\frac12 x^2$ | $\frac12 y^2$ | 自共轭 |
| $\|x\|$（范数） | $\delta_{\|y\| \le 1}(y)$ | 范数的共轭是指示函数 |
| $\mathrm{e}^x$ | $y\ln y - y$（$y>0$） | 指数函数 |
| $\frac{1}{2m} \dot q^2 - V$ 关于 $\dot q$ | $\frac{1}{2m} p^2 + V$（$p = m\dot q$） | **Hamiltonian** |

最后一行正是上一课《Hamilton-Jacobi 方程》里的哈密顿量：$H(q,p)$ 是拉格朗日量 $L(q,\dot q)$ 关于 $\dot q$ 的共轭。Legendre-Fenchel 变换把「拉格朗日 ↔ 哈密顿」的两套力学语言统一在一个定义之下。<span class="marginnote">这解释了一个谜：为什么 H-J 方程里 $H$ 用 $\sup$ 定义？因为物理的 Legendre 变换只是光滑特例，数学的 $\sup$ 定义允许 $L$ 非光滑——量子化与最优控制里这个推广至关重要。</span>

## 3 Fenchel-Young 不等式与双共轭

由 $\sup$ 定义直接得到**Fenchel-Young 不等式**：

$$
\langle x, y\rangle \le f(x) + f^*(y) \qquad \forall x, y
$$

等号当且仅当 $y \in \partial f(x)$（次微分：$y$ 是 $f$ 在 $x$ 处的某个次梯度）。<span class="marginnote">次微分 $\partial f(x)$ 是「支撑超平面斜率的集合」：光滑点上是 $\{f'(x)\}$，尖点处是一个区间。Fenchel-Young 的不等式版本是「凸函数的最基本事实」，等号版本是「最优性的余切条件」——两者都直接通向对偶定理。</span>

**Fenchel-Moreau 定理**：若 $f$ 是正常凸下半连续函数，则

$$
f^{**} = f
$$

即「共轭的共轭还原本人」。它把「凸 + 下半连续」刻画为「若干仿射函数的逐点上确界」——这是凸函数最深刻的结构定理。<span class="marginnote">下半连续（上一课直接方法的主角）在这里再次出场：没有它，$f^{\ast\ast}$ 只是 $f$ 的下包络（闭包）。「凸 + 下半连续 = 被支撑线完全决定」，与《Hilbert 积分不变与场论方法》里「场 + $E\ge0$ ⇒ 全局最优」是同一个道理的抽象版。</span>

## 4 Fenchel 对偶定理

现在让两个凸函数对打。**Fenchel 对偶定理（Fenchel duality theorem）**：对正常凸下半连续的 $f, g: \mathbb{R}^n \to \mathbb{R}\cup\{+\infty\}$，

$$
\boxed{\;\inf_x \bigl[f(x) + g(x)\bigr] = \sup_y \bigl[-f^*(-y) - g^*(y)\bigr]\;}
$$

在某个约束规格化条件（如存在 $x$ 使 $f$ 与 $g$ 都有限且其中一方连续）下成立，且最优解 $x^*$ 与 $y^*$ 通过次微分条件相互确定。<span class="marginnote">等号左侧叫<strong>原问题（primal）</strong>，右侧叫<strong>对偶问题（dual）</strong>：一个极小化，一个极大化，最优值相等。这就是「强对偶」——现代优化的第一定理，参见 Ekeland–Témam Ch. III 与第三级《凸优化》。</span>

更一般的版本带线性算子 $A$：

$$
\inf_x \bigl[f(x) + g(Ax)\bigr] = \sup_y \bigl[-f^*(-A^*y) - g^*(y)\bigr]
$$

这里的对偶变量 $y$ 正是 **Lagrange 乘子**——等周问题里那个「影子价格」 $\lambda$ 的现代名字。<span class="marginnote">带 $A$ 的版本把本专题《等周问题与约束变分》的乘子法彻底正规化：约束 $Ax \in \text{const}$ 的乘子就是对偶变量。变分问题的所有乘子技巧都是 Fenchel 对偶的特例。</span>

## 5 公式解析：$\inf_x[f(x)+g(x)] = \sup_y[-f^*(-y)-g^*(y)]$

$$
\inf_x\,[f(x)+g(x)] \;\stackrel{强对偶}{=}\; \sup_y\,\bigl[-f^*(-y) - g^*(y)\bigr]
$$

三步拆解：

- **第一步，弱对偶恒成立**：对任意 $x, y$，把 Fenchel-Young 用两次（一次对 $f$ 与 $-y$、一次对 $g$ 与 $y$）：

$$
f(x) + g(x) \ge \langle -y, x\rangle - f^*(-y) + \langle y, x\rangle - g^*(y) = -f^*(-y) - g^*(y)
$$

左边对 $x$ 取下确界、右边对 $y$ 取上确界：$\inf_x[f+g] \ge \sup_y[-f^* - g^*]$——**弱对偶**永远成立，不用任何条件。<span class="marginnote">弱对偶不需要凸性也不需要下半连续，它只是 Fenchel-Young 的算术推论。这也解释了为什么「对偶界的可证性」如此便宜——数值优化（如 SVM 的对偶间隔）先靠弱对偶拿到安全边界。</span>

- **第二步，凸性 + 规格化条件给出反向**：需要存在「支撑超平面」穿过原问题与对偶问题之间。凸性保证支撑存在，规格化条件（如相对内部相交）保证支撑「足够高」，于是反向不等号成立，$\inf = \sup$。
- **第三步，最优性的刻画**：强对偶的极值点满足余切条件

$$
-y^* \in \partial f(x^*), \qquad y^* \in \partial g(x^*)
$$

即「$x^*$ 与 $y^*$ 互相是对方的支撑点」——这正是 E-L 方程与乘子法在凸世界里的终极形态。<span class="marginnote">把次微分条件与上一课《直接方法与下半连续性》并排看：凸性既保证「弱下半连续」（存在性），又保证「强对偶」（最优性）。凸性是变分法能同时拿下「存在」与「最优」的根源。</span>

## 6 对偶的现代回响

对偶理论早已溢出纯数学：

**优化**：SVM 的拉格朗日对偶、凸优化的内点法（原-对偶形式）、以及「先解对偶、再取回原变量」的工程常规——Fenchel 对偶是它们的公共母题。
- **最优传输**：Kantorovich 对偶把「搬土的最省成本」改写成「两个势函数的 sup-inf」，与 Wasserstein GAN 的判别器-生成器结构同源。<span class="marginnote">对「从极限到大模型」的读者：GAN 的对抗训练、扩散模型的对偶结构、强化学习的 Lagrangian，都能在 Fenchel 对偶里找到族谱——见第三级《生成模型》与《凸优化》。</span>
**变分问题的回归**：Ekeland–Témam 用它把带约束的变分问题（等周、Plateau）重新表述为对偶问题，使「存在性」与「对偶间隙为零」可以一起证明——本专题全部主题在对偶语言下获得统一。<span class="marginnote">最后，把十个名词串成主线：E-L 方程（局部条件）→ Noether（对称与守恒）→ 角条件与过分函数（强/弱极小）→ Legendre、Jacobi（二阶判据）→ 场论（全局充分）→ 直接方法（存在性）→ Hamilton-Jacobi（场的 PDE）→ 极小曲面（二维应用）→ 凸对偶（统一语言）。变分法的全景至此收束。</span>

## 7 小结

- **凸函数 + 指示函数**：约束被翻译成 $+\infty$ 惩罚，带约束极小化并入无约束形式。
- **Legendre-Fenchel 变换** $f^*(y) = \sup_x[\langle x,y\rangle - f(x)]$：把「点值」换成「支撑斜率」；哈密顿量就是拉格朗日量的共轭。
- **Fenchel-Young 不等式** $\langle x,y\rangle \le f(x) + f^*(y)$；**Fenchel-Moreau** $f^{\ast\ast} = f$ 刻画凸下半连续函数。
- **Fenchel 对偶定理**：$\inf[f+g] = \sup[-f^*(-\cdot)-g^*(\cdot)]$，弱对偶无条件成立，强对偶需凸性 + 规格化条件；对偶变量即 Lagrange 乘子。
- 对偶理论统一了等周乘子、凸性判据、下半连续与 H-J 方程，是现代优化与最优传输的共同母题。

至此，变分法专题画上句号：从 Euler-Lagrange 方程的第一性，到 Fenchel 对偶的收束。在下一专题《偏微分方程》里，我们将从 Evans Ch. 8 转入 Evans 的主战场——直接方法写出的极小点如何满足方程、如何获得正则性，以及这些方程在物理与几何中的全面展开。
