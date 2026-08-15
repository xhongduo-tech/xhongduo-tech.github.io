---
title: 调和函数的积分表达式
date: 2026-08-07
---

# 调和函数的积分表达式

<div class="epigraph">
<p>一个调和函数的值，藏在边界值与边界法向导数的积分里。</p>
<footer>—— 格林表示公式（Green's representation formula）</footer>
</div>

<div class="article-byline">
<p>第二级 · 偏微分方程 ｜ 谷超豪《数学物理方程》第三章 ｜ 2026-08-07</p>
</div>

## 为什么从积分表达式开始

这个公式在历史上由格林（George Green，1828 年《论数学分析在电磁理论中的应用》）首次以「势」的语言写下——他在一本自费印刷的小册子里奠定了位势理论的雏形。二百年后，这个公式从静电场出发，成了整个 PDE 理论的通用工具箱。

格林公式 + 基本解这对组合，能把调和函数在内部任一点的值写成**边界上的积分**——这就是**调和函数的积分表达式（格林表示公式）**。它是位势理论的中心公式：内部值「由边界决定」从模糊的哲学变成了精确的积分恒等式。不过它也有一个「遗憾」：表达式中同时出现边界值 $u|_{\partial\Omega}$ 与边界法向导数 $\partial u/\partial n|_{\partial\Omega}$，而我们通常只知道其中一个。要变成真正的求解公式，还需格林函数（下一节）。先把表示公式本身推出来。

## 1 构造：用基本解当探针

设 $\Omega$ 光滑有界，$u$ 在 $\Omega$ 上调和，$y \in \Omega$ 固定。取格林公式中的两个函数为

$$
v(x) = \Gamma(x - y) = \frac{1}{4\pi|x - y|} \quad（三维）
$$

$v$ 在 $x = y$ 处有奇点，不能直接对 $\Omega$ 用格林公式。于是**挖去小球** $B_\epsilon(y) \subset \Omega$，在 $\Omega_\epsilon = \Omega \setminus B_\epsilon(y)$ 上应用格林第二公式。<span class="marginnote">挖球取极限是基本解论证的标准动作：奇点处的 δ 源要通过「小球的边界积分」在极限下提取出来。这个技巧与上一节验证基本解时「测源强度」的运算完全同构——每次用基本解都要走一遍「挖球 → 极限」。</span>在 $\Omega_\epsilon$ 上，$u$ 与 $v$ 都调和（$v$ 在去心区域调和），格林第二公式右端为零：

$$
\oint_{\partial\Omega}\Big(v\frac{\partial u}{\partial n} - u\frac{\partial v}{\partial n}\Big)dS + \oint_{\partial B_\epsilon}\Big(v\frac{\partial u}{\partial n} - u\frac{\partial v}{\partial n}\Big)dS = 0
$$

注意 $\partial\Omega_\epsilon = \partial\Omega \cup \partial B_\epsilon$，两个积分符号分别对应两块边界。

## 2 公式解析：小球项在极限下吐出 $u(y)$

处理 $\partial B_\epsilon(y)$ 上的积分，让 $\epsilon \to 0$：

- **第一步，估第一项。** $\oint_{\partial B_\epsilon}\Gamma\frac{\partial u}{\partial n}dS$。因为 $\Gamma \sim \frac{1}{4\pi\epsilon}$、面积 $4\pi\epsilon^2$、$\partial u/\partial n$ 有界，该项 $\le C\epsilon \to 0$。
- **第二步，估第二项。** $\oint_{\partial B_\epsilon}u\frac{\partial\Gamma}{\partial n}dS$。在 $\partial B_\epsilon$ 上 $\frac{\partial\Gamma}{\partial n} = \frac{\partial}{\partial r}\frac{1}{4\pi r}\Big|_{r=\epsilon} = -\frac{1}{4\pi\epsilon^2}$，故
  $$ \oint_{\partial B_\epsilon}u\frac{\partial\Gamma}{\partial n}dS = -\frac{1}{4\pi\epsilon^2}\oint u\,dS = -\bar u_\epsilon \to -u(y) $$
  其中 $\bar u_\epsilon$ 是 $u$ 在球面上的平均，趋于 $u(y)$（连续性）。
- **第三步，取极限。** 把两个极限代回：$0 = \oint_{\partial\Omega}\big(v\frac{\partial u}{\partial n} - u\frac{\partial v}{\partial n}\big)dS + 0 - u(y)$。
- **第四步，写积分表达式（三维）。**
  $$
  \boxed{\;u(y) = \oint_{\partial\Omega}\left[\frac{1}{4\pi|x-y|}\frac{\partial u}{\partial n}(x) - u(x)\frac{\partial}{\partial n_x}\frac{1}{4\pi|x-y|}\right]dS_x\;}
  $$
  二维版本把 $1/(4\pi|x-y|)$ 换成 $-\frac{1}{2\pi}\ln|x-y|$。

**积分表达式 = 边界值 $u$ 与边界法向导数 $\partial u/\partial n$ 的线性组合，通过基本解的核积分起来。** 两项的物理意义：第一项是**单层势**（边界上的源分布），第二项是**双层势**（边界上的偶极子分布）——它们正是位势理论中两类基本边界积分算子。

## 3 积分表达式的结构解读

把表达式拆成两类积分，它们在位势理论中各有名字：

| 项 | 核 | 名字 | 物理图像 |
| --- | --- | --- | --- |
| $\int\frac{1}{4\pi r}\frac{\partial u}{\partial n}dS$ | 基本解 | 单层势 | 边界上电荷层的电势 |
| $\int u\frac{\partial}{\partial n}\frac{1}{4\pi r}dS$ | 基本解法向导数 | 双层势 | 边界上偶极层的电势 |

**单层势与双层势是边界积分方程（boundary integral equation）的两个基本构件。** 它们的性质（跳跃关系、边界上的连续性）构成了现代边界元方法（BEM）的理论基础——工程数值方法里「只离散边界、不离散内部」的威力正来源于此。<span class="marginnote">双层势核 $\frac{\partial}{\partial n}\frac{1}{4\pi r}$ 在边界上有非平凡跳跃：从边界两侧趋近，积分值差一个 $u$ 本身。这个「跳跃关系」是边界积分方程理论的核心，也是工程边界元法误差分析的起点。</span>

**关键观察：表达式需要 $u|_{\partial\Omega}$ 与 $\partial u/\partial n|_{\partial\Omega}$ 两个信息。** 对 Dirichlet 问题只知道前者、对 Neumann 问题只知道后者——所以积分表达式只是「表示」而非「求解」。真正的求解要等格林函数登场。

## 4 从表示公式到平均值定理

积分表达式的一个漂亮副产品：取 $\Omega$ 为球 $B_R(0)$，$y = 0$ 为球心。此时在边界 $|x| = R$ 上

$$
\frac{\partial}{\partial n}\frac{1}{4\pi|x|} = -\frac{1}{4\pi R^2}
$$

是常数，可提出积分号。同时 $\oint\frac{\partial u}{\partial n}dS = 0$（调和函数通量为零，上一节推论），于是

$$
u(0) = \frac{1}{4\pi R^2}\oint_{|x|=R}u(x)\,dS_x
$$

**球心值 = 球面上值的平均。** 这就是调和函数的**平均值定理**（下一节专讲）——从积分表达式几乎免费地掉出来。

**一个极限情形**：当 $\Omega$ 取「无穷大球」的极限（在 $\mathbb{R}^3$），边界项在无穷远处如何消失需要衰减条件；对满足 $u = O(1/r)$、$\partial u/\partial n = O(1/r^2)$ 的调和函数，边界积分在 $R \to \infty$ 时趋于零，于是无边界情形的表示公式「没有信息」——**调和函数在全空间的值不能被边界决定，因为它根本没有边界**。这正是 Liouville 定理与 Harnack 不等式的舞台（第六篇后半段）。<span class="marginnote">平均值定理是调和函数最深刻的「均衡性」陈述：任何一点的函数值等于任意包围它的球面上的平均值。它是一切后续结论（极值原理、刘维尔定理、Harnack 不等式）的引擎，而它从积分表达式如此自然地涌出，正说明表示公式处于理论的核心枢纽位置。</span>

## 5 公式解析：二维情形的格林表示

二维基本解是 $v = -\frac{1}{2\pi}\ln|x-y|$，它没有三维那样的「趋于零」的远端行为，但同样的挖球论证照常成立：

- **第一步，重复挖球。** 小球 $B_\epsilon(y)$ 边界上 $v \sim -\frac{1}{2\pi}\ln\epsilon$，周长 $2\pi\epsilon$，第一项 $\oint v\,\partial_n u\,dS \sim \epsilon\ln\epsilon \to 0$。
- **第二步，处理第二项。** $\partial_r v\big|_{r=\epsilon} = -\frac{1}{2\pi\epsilon}$，故 $\oint u\,\partial_r v\,dS = -\frac{1}{2\pi\epsilon}\oint u\,dS \to -u(y)$。
- **第三步，写公式。** $u(y) = \oint_{\partial\Omega}\left[-\frac{1}{2\pi}\ln|x-y|\frac{\partial u}{\partial n} + u\frac{\partial}{\partial n}\frac{1}{2\pi}\ln|x-y|\right]dS_x$。

**二维与三维的差别只在核的形状，论证结构一字不差。** $\ln$ 核在无穷远不衰减，对应二维位势的「长程性」（对数势在无穷远发散）——这是二维与三维位势物理的根本区别之一，也让二维的「源总通量」条件更微妙（Neumann 相容条件在后面的小节专门处理）。

## 6 表示公式与边界元方法

把表示公式的工程意义讲清楚。它在弹性力学中的对应物叫 **Somigliana 恒等式**：把位移场写成边界位移与边界力的积分。边界元方法（BEM）正是把「内部值由边界积分表达」反过来用——先解出边界上的未知数据（Dirichlet 问题反解法向导数），再逐点积分出内部值：**只离散边界，内部值全靠公式算**。

把三类位势的对应收进一张表：

| 物理对象 | 核 | 内部量由边界积分表达 |
| --- | --- | --- |
| 静电势（三维） | $1/(4\pi r)$ | 电势 ← 电荷层 + 偶极层 |
| 二维位势 | $-\ln r/(2\pi)$ | 势 ← 源分布 |
| 弹性位移 | Kelvin 张量 | 位移 ← 边界力 + 位移 |

**辨析｜易错点：** 表示公式里的「边界值」与「法向导数」必须来自同一个调和函数。若任意给定 $g_1, g_2$ 分别充当 $u|_{\partial\Omega}$ 与 $\partial_n u|_{\partial\Omega}$，积分表达式给出的通常不是调和函数——两组边界数据必须「相容」（由一个调和函数同时生成）。这与 Neumann 问题的相容条件、Dirichlet 问题法向导数的存在性，是同一件事的三个说法。

**解的存在性视角。** 积分表达式把「解的存在」翻译成「两组相容边界数据的存在」。对 Dirichlet 问题，法向导数 $\partial_n u|_{\partial\Omega}$ 事先未知、却由 $u|_{\partial\Omega}$ 唯一决定（若解存在）——把表示公式对边界取极限，就得到联系 $g$ 与 $\partial_n u$ 的边界积分方程。这是「边界积分方程」理论的门径，也是第九篇基本解方法通向数值计算的一座桥。<span class="marginnote">对边界取极限时，双层势出现 $-\frac12 u(x_0)$ 的跳跃项，这是边界积分方程理论的核心细节：边界上的方程不再是「原样」的表示公式，而是带 $1/2$ 因子的 Fredholm 方程。这个跳跃关系也是 BEM 数值实现中最容易被忽略、又最关键的项。</span>

## 7 小结

- 用基本解 $\Gamma(x-y)$ 作格林公式的探针，挖球取极限，得到调和函数的积分表达式。
- 小球项在极限下吐出 $u(y)$，边界项保留 $u$ 与 $\partial u/\partial n$ 的积分。
- 表达式含单层势（核为基本解）与双层势（核为基本解法向导数）两项。
- 单层/双层势是边界积分方程与边界元方法的基本构件。
- 取球心可推得平均值定理，说明表示公式处于理论枢纽。

在下一节，我们专讲平均值定理。
