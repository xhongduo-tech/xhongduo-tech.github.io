---
title: 变分法与最优控制：Euler-Lagrange 方程
date: 2026-08-07
---

# 变分法与最优控制：Euler-Lagrange 方程

<div class="epigraph">
<p>在自然界的所有可能曲线中，自然选择那条使某个量成为极值的曲线。</p>
<footer>—— 约翰 · 伯努利（Johann Bernoulli，1696，最速降线问题）</footer>
</div>

<div class="article-byline">
<p>第二级 · 控制论与最优控制 ｜ Kirk《最优控制理论导论》Ch. 2 ｜ 2026-08-07</p>
</div>

## 为什么最优控制从变分法讲起

上一节的性能指标 $J = \int g\, \mathrm{d}t$ 是一个**泛函**——它的「自变量」不是数，而是**函数**（轨线 $x(\cdot)$ 或控制 $u(\cdot)$）。普通微积分研究「一个数对另一个数的极值」，变分法研究「一个数对一个函数的极值」。最优控制里「找最优轨线」，本质就是「在函数空间里找极值」——这是变分法的主场。

变分法的历史比控制论古老得多：从 17 世纪最速降线问题到 18 世纪 **Euler-Lagrange 方程**成型，它是「求泛函极值」的第一套严格工具。控制理论把系统动态当作约束加进去，变分法立即升级为最优控制的第一种求解方法。今天这篇，把变分法的核心——**Euler-Lagrange 方程**——从头推到尾，并演示它如何变成控制问题的**伴随方程**。

## 1 泛函与变分：微积分的「无穷维推广」

普通函数 $f(x)$ 的自变量是数；**泛函（functional）** $J[x(\cdot)]$ 的自变量是函数，输出是数。例：路径长度

$$
L[y] = \int_{x_0}^{x_1}\sqrt{1 + (y')^2}\,\mathrm{d}x
$$

把每条曲线 $y(x)$ 映射成它的长度。我们想找使 $L$ 最小的曲线——这就是泛函极值问题。

微积分里，极值点满足 $f'(x) = 0$；变分法里，极值函数满足**变分等于零**：$\delta J = 0$。所谓**变分（variation）**，就是函数自身的微小扰动 $\delta y(x)$（每一点都动一点点），$J$ 对扰动的「一阶响应」就是 $\delta J$。<span class="marginnote">类比：$dx$ 是自变量的增量，$\delta y$ 是「函数的增量」——它不是某个点动了，而是整条曲线从一个位置「飘」到相邻位置。$\delta J = 0$ 的意思是：在所有相邻曲线里，极值曲线的 $J$ 值一阶不变，就像普通极值点的切线水平。严格的变分理论需要泛函分析与 Sobolev 空间，但工程上掌握「微扰 → 一阶条件」这条直觉就够用。</span>

**变分法基本引理（fundamental lemma）**：若连续函数 $M(x)$ 对任意满足端点为零的光滑扰动 $\eta(x)$ 都有 $\int M(x)\eta(x)\mathrm{d}x = 0$，则 $M(x) \equiv 0$。这条引理是「从积分恒零推函数恒零」的桥梁，是推导 Euler-Lagrange 方程的关键一步。

## 2 Euler-Lagrange 方程：推导

考虑最简泛函

$$
J[y] = \int_{x_0}^{x_1} F\big(x, y(x), y'(x)\big)\,\mathrm{d}x,
$$

两端固定：$y(x_0) = y_0$，$y(x_1) = y_1$。设 $y^*$ 是极值函数，任取微扰 $\eta(x)$（端点为零），考察单参数族 $y^* + \epsilon\eta$。令

$$
\phi(\epsilon) = J[y^* + \epsilon\eta],
$$

普通微积分告诉我们极值条件 $\phi'(0) = 0$。逐项求导：

$$
\phi'(0) = \int_{x_0}^{x_1} \Big( \frac{\partial F}{\partial y}\eta + \frac{\partial F}{\partial y'}\eta' \Big)\mathrm{d}x = 0.
$$

对第二项用**分部积分**（利用 $\eta$ 端点为零消去边界项）：

$$
\int \frac{\partial F}{\partial y'}\eta'\,\mathrm{d}x = -\int \frac{\mathrm{d}}{\mathrm{d}x}\Big(\frac{\partial F}{\partial y'}\Big)\eta\,\mathrm{d}x.
$$

于是

$$
\int_{x_0}^{x_1} \Big[ \frac{\partial F}{\partial y} - \frac{\mathrm{d}}{\mathrm{d}x}\frac{\partial F}{\partial y'} \Big]\eta(x)\,\mathrm{d}x = 0.
$$

由基本引理，方括号必为零，得到 **Euler-Lagrange 方程**：

$$
\frac{\partial F}{\partial y} - \frac{\mathrm{d}}{\mathrm{d}x}\frac{\partial F}{\partial y'} = 0.
$$

**Euler-Lagrange 方程**是极值轨线必须满足的**必要条件**（加二阶条件才充分）。它把「找整个函数的极值」变成「每一点上都满足的一条 ODE」——无穷维的困难被压缩成有限维的微分方程。<span class="marginnote">历史上，欧拉 1744 年用「折线逼近」得到这个方程，拉格朗日 1755 年 19 岁时用「变分符号 $\delta$」给出了我们今天看到的简洁推导。这个方程是物理学里「最小作用量原理」的数学内核——力学、光学、电磁学里的一切极值原理都是它的化身。控制理论用它，等于站上了整个物理学的肩膀上。</span>

## 3 从 Euler-Lagrange 到最优控制：哈密顿与伴随方程

现在把变分法接到最优控制上。控制问题的性能指标与动态约束：

$$
J = \int_{t_0}^{t_f} g(x, u, t)\,\mathrm{d}t, \qquad \dot{x} = f(x, u, t).
$$

**约束不能丢**：$x$ 与 $u$ 不能独立变分，它们必须满足状态方程。引入**拉格朗日乘子函数** $\lambda(t)$（又称**伴随变量/协态，costate**）与**哈密顿函数（Hamiltonian）**

$$
H(x, u, \lambda, t) = g(x, u, t) + \lambda^T f(x, u, t).
$$

把约束「吸收」进积分：$J = \int \big[g + \lambda^T(\dot{x} - f)\big]\mathrm{d}t = \int \big[g - \lambda^T f\big]\mathrm{d}t + \big[\lambda^T x\big]_{t_0}^{t_f}$（对 $\lambda^T\dot{x}$ 分部积分）。令变分为零，得到三组方程：

$$
\dot{x} = \frac{\partial H}{\partial \lambda} = f, \qquad
\dot{\lambda} = -\frac{\partial H}{\partial x}, \qquad
\frac{\partial H}{\partial u} = 0.
$$

第二条是**伴随方程（costate equation）**，第三条是**最优性条件（optimality condition）**——它们就是 Euler-Lagrange 方程在控制问题中的翻版。<span class="marginnote">三组方程的物理解读：$\dot{x} = \partial H/\partial\lambda$ 是原动态（约束没丢）；$\dot{\lambda} = -\partial H/\partial x$ 是「最优性的影子动态」，$\lambda$ 度量「状态对代价的边际敏感度」（有点像经济学里的影子价格）；$\partial H/\partial u = 0$ 是「在最优时，控制已无法再改善代价」。整个系统是 $2n$ 维的边值问题（$n$ 个状态 + $n$ 个伴随）。</span>

## 4 公式解析：Euler-Lagrange 方程的「三步」

把推导压缩成一张可复用的三步流程图：

$$
\frac{\partial F}{\partial y} - \frac{\mathrm{d}}{\mathrm{d}x}\frac{\partial F}{\partial y'} = 0
$$

- **第一步，写泛函**：确认要极化的对象 $J = \int F\,\mathrm{d}x$，并明确端点约束。$F$ 里同时有 $y$ 与 $y'$。
- **第二步，扰动与一阶项**：令 $y \to y + \epsilon\eta$，把 $J$ 对 $\epsilon$ 求导，保留一阶项。$\partial F/\partial y$ 贡献「$y$ 直接变」的项，$\partial F/\partial y'$ 贡献「斜率变」的项。
- **第三步，分部积分 + 基本引理**：把 $\eta'$ 的项转成 $\eta$，用端点条件消边界，由基本引理让整体为零 ⇒ 方程。

这条三步法从最速降线（$F = \sqrt{1+y'^2}/\sqrt{y}$）到光学费马原理、再到最优控制，全部适用。它告诉我们的核心思想是：**极值轨线必须使「沿整条轨线的微小扰动」一阶为零**——这个「逐点」条件把泛函极值翻译成微分方程。

**辨析｜易错点：** 三个高频坑。其一，**Euler-Lagrange 方程是必要条件不是充分条件**——还要检查二阶变分（Jacobi 条件、Legendre 条件）或实际验证。其二，**端点自由时不能随便消边界项**：$y$ 的端点若自由（如终端不受约束），分部积分产生的边界项 $\frac{\partial F}{\partial y'}\eta\Big|$ 不自动为零，必须由**横截条件**处理——这正是下一节的主题。其三，**$\partial H/\partial u = 0$ 只在控制无约束时成立**：控制受限于 $\Omega$（如 $|u|\le 1$）时，最优解落在边界上，零点条件失效，要改用 Pontryagin 极大值原理（再下一节）。

## 5 小结

- **变分法**研究泛函极值：函数是「点」，轨线是「向量」，泛函极值 = 无穷维空间的驻点。
- **Euler-Lagrange 方程** $\frac{\partial F}{\partial y} - \frac{\mathrm{d}}{\mathrm{d}x}\frac{\partial F}{\partial y'} = 0$：极值轨线的逐点必要条件，由「扰动 + 分部积分 + 基本引理」推出。
- 最优控制把它翻译成**哈密顿函数** $H = g + \lambda^T f$