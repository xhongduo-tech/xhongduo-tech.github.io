---
title: Euler-Lagrange 方程推导
date: 2026-08-11
---

# Euler-Lagrange 方程推导

<div class="epigraph">
<p>自然界用最省力的方式运作，它绝不白白浪费任何东西。</p>
<footer>—— 莱昂哈德 · 欧拉（Leonhard Euler）</footer>
</div>

<div class="article-byline">
<p>第二级 · 进阶数理 · 变分法 ｜ 对标教材 ｜ 2026-08-11</p>
</div>

## 为什么从 Euler-Lagrange 方程开始

微积分解决的问题是：给定一个函数 $f(x)$，它在哪个点取到最大值或最小值？变分法把这个问题换了一个维度——给定一个「函数的函数」，求**哪个函数**让它的值取到最大或最小。这种「函数的函数」叫作**泛函（functional）**，最常见的形式是

$$
J[y] = \int_a^b F(x, y(x), y'(x))\, dx
$$

「从极限到大模型」这条主线走到这里，第一次把「求导求极值」从有限个自变量推广到无穷维空间。<span class="marginnote">有限维里找「使 $f$ 最小的点 $x^*$」，无穷维里找「使 $J$ 最小的曲线 $y^*$」，两者共用同一套思想：盯着极值点附近的邻域，那里函数值「不涨不跌」。</span> 而在物理学里，最小作用量原理（见本专题《Hamilton 原理与最小作用量原理》）会告诉我们：真实运动的轨迹正是让某个泛函取驻值的曲线——于是 Euler-Lagrange 方程成了分析力学、乃至后续 Hamilton-Jacobi 理论（见《Hamilton-Jacobi 方程》）的基石。<span class="marginnote">变分法在教材树中的位置：它把第一级《数学分析》的极值理论延伸进无穷维；也是第三级《最优控制》《变分推断》里「对函数求导」这一类直觉的数学根源。</span>

## 1 从函数极值到泛函极值

一元极值理论里，判定「$x^*$ 是极小点」靠的是微分：若 $f$ 在 $x^*$ 处可导且在内部取极小，则 $f'(x^*) = 0$。变分法把这一句话照搬到无穷维。

**泛函（functional）**：把「函数」映到「实数」的映射。比如弧长泛函

$$
L[y] = \int_a^b \sqrt{1 + y'(x)^2}\, dx
$$

输入一条曲线 $y$，输出它的长度。

变分问题的标准提法（固定端点）是：在端点条件

$$
y(a) = A, \qquad y(b) = B
$$

之下，在所有光滑函数里求使 $J[y] = \int_a^b F(x,y,y')\,dx$ 取最小值的曲线。<span class="marginnote">端点固定是入门版本：两个约束恰好配上 E-L 方程（二阶 ODE）的两个边界条件。端点自由、端点可滑动的问题在 van Brunt Ch. 2 末段讨论，会多出「横截条件」。</span>

关键在于：可允许曲线的「邻域」长什么样？答案是与 $y$ 很接近的曲线 $y + \varepsilon\eta$，其中 $\eta$ 是任意光滑函数且 $\eta(a) = \eta(b) = 0$（保证端点不动），$\varepsilon$ 是很小的数。

**扰动（variation）**：$\eta$ 称为一个扰动，它像有限维里的「方向」。当 $\varepsilon \to 0$ 时，$y + \varepsilon\eta$ 扫过 $y$ 的整个邻域。

![极值曲线与扰动曲线](/images/calculus-of-variations/euler-lagrange-equation-1.svg)

## 2 第一变分：给曲线一次「试错」

把 $\varepsilon$ 看成唯一的自变量，定义一元函数

$$
\varphi(\varepsilon) = J[y + \varepsilon\eta] = \int_a^b F(x,\, y+\varepsilon\eta,\, y'+\varepsilon\eta')\, dx
$$

如果 $y$ 是 $J$ 的极小点，那么 $\varphi$ 在 $\varepsilon = 0$ 处必须取极小，由一元微分知识立刻得到

$$
\varphi'(0) = 0 \qquad \text{对一切允许的 } \eta
$$

对 $\varepsilon$ 求导，链式法则给出

$$
\varphi'(0) = \int_a^b \Bigl[ F_y(x,y,y')\, \eta + F_{y'}(x,y,y')\, \eta' \Bigr] dx
$$

**第一变分（first variation）**：$\varphi'(0)$ 称为 $J$ 在 $y$ 处的第一变分，记作 $\delta J$。极小点的必要条件就是：对一切扰动 $\eta$，第一变分为零。<span class="marginnote">记号 $\delta$ 与微分 $d$ 平起平坐：$d$ 作用在变量上，$\delta$ 作用在整条函数曲线上。数学上它们都是「在一族对象里取一阶泰勒项」。</span>

## 3 推导：分部积分与基本引理

第一变分里还混着 $\eta'$ 项，需要把它「转移」到 $\eta$ 上。对第二项做分部积分：

$$
\int_a^b F_{y'}\, \eta'\, dx = \bigl[F_{y'}\,\eta\bigr]_a^b - \int_a^b \Bigl(\frac{d}{dx}F_{y'}\Bigr)\eta\, dx
$$

由于 $\eta(a) = \eta(b) = 0$，边界项消失，于是

$$
\delta J = \int_a^b \Bigl[ F_y - \frac{d}{dx}F_{y'} \Bigr] \eta \, dx = 0
$$

现在要把「积分恒为零」翻译成「被积函数恒为零」。这需要变分法的基本引理：

**基本引理（fundamental lemma of the calculus of variations）**：若连续函数 $g$ 对一切在端点处取零的光滑函数 $\eta$ 都有 $\int_a^b g(x)\,\eta(x)\,dx = 0$，则 $g$ 在区间上恒等于零。<span class="marginnote">用「试函数 $\eta$」把 $g$ 的每一个点都筛出来——这启发了一百年后的分布（distribution）理论与泛函分析里的对偶空间。van Brunt 在 Ch. 2 用 Du Bois-Reymond 引理给出了更精细版本。</span>

应用引理，得到**Euler-Lagrange 方程**：

$$
F_y - \frac{d}{dx}F_{y'} = 0
$$

这是变分法第一条、也是最核心的一条方程。

## 4 公式解析：Euler-Lagrange 方程

$$
\frac{\partial F}{\partial y}(x,y,y') - \frac{d}{dx}\left(\frac{\partial F}{\partial y'}(x,y,y')\right) = 0
$$

对这条方程做三步拆解：

- **第一步，读懂偏导数**：$F_y$ 是固定 $y'$、只对 $y$ 求偏导；$F_{y'}$ 是固定 $y$、只对 $y'$ 求偏导。$y$ 与 $y'$ 在这里被当作**两个相互独立的变量**——尽管真实曲线上 $y'$ 是 $y$ 的导数。这是变分法初学者最容易卡住的地方。<span class="marginnote">把 $y$ 与 $y'$ 视作独立，是变分法的「本地宪法」：先按两个变量求偏导，写完方程再把 $y' = dy/dx$ 的依赖关系放回去。</span>

- **第二步，看清 $\frac{d}{dx}$ 是「全导数」**：$F_{y'}$ 是 $x, y, y'$ 的函数，而 $y, y'$ 又随 $x$ 变化，所以

$$
\frac{d}{dx}F_{y'} = F_{y'x} + F_{y'y}\, y' + F_{y'y'}\, y''
$$

这里 $F_{y'x} = \partial F_{y'}/\partial x$ 等是**二阶偏导数**的简写。

- **第三步，读出「二阶常微分方程」**：$y''$ 出现在方程里，E-L 是二阶 ODE；配上端点条件 $y(a)=A,\ y(b)=B$，恰好确定唯一解。

直觉上，可以把 $F$ 类比力学里的拉格朗日量：$F_{y'}$ 扮演「动量」$p = \partial L/\partial \dot q$，$F_y$ 扮演「广义力」，方程 $\dot p = F_y$ 正是牛顿第二定律 $F = ma$ 的变分形态。<span class="marginnote">「$F = ma$ 与 E-L 等价」是下一课《Hamilton 原理》的核心内容——到那时你会看到同一条方程从「最小作用量」里长出来。</span>

## 5 两个立刻能解的经典例子

**例 1：两点之间最短路径是直线。** 取 $F = \sqrt{1 + y'^2}$，被积函数不含 $x$ 也不含 $y$。因为 $F_y = 0$，E-L 方程退化为 $\frac{d}{dx}F_{y'} = 0$，即

$$
\frac{y'}{\sqrt{1+y'^2}} = C \quad\Longrightarrow\quad y' = \text{常数}
$$

所以 $y$ 是直线。平凡的结果，但是由变分法严格推出来的——验证了整个机制运转正常。<span class="marginnote">「$F$ 与 $x$ 无关」自动给出守恒量 $F - y'F_{y'} = \text{常数}$，这正是 Noether 定理（本专题《Noether 对称性定理》）的第一个雏形。</span>

**例 2：最速降线是摆线。** 1696 年 Johann Bernoulli 向全欧数学界挑战：质点仅受重力、沿无摩擦曲线从 $A$ 滑到 $B$，用最短时间，曲线形状如何？牛顿当晚就解出并匿名投稿，伯努利见后感叹「从爪印认出了狮子」。

下滑时间是泛函

$$
T[y] = \int_a^b \sqrt{\frac{1 + y'^2}{2gy}}\, dx
$$

（坐标取 $y$ 轴向下，消去常数因子后）$F = \sqrt{(1+y'^2)/y}$。$F$ 不含 $x$，由守恒量 $F - y'F_{y'}=C$ 得 $y(1+y'^2) = \text{常数}$，参数化后正是**摆线（cycloid）**——圆沿直线滚动时圆周上一点画出的曲线：

$$
x = a(\theta - \sin\theta), \qquad y = a(1 - \cos\theta)
$$

## 6 小结

- **泛函**把函数映到实数；变分问题的标准形式是固定端点、极小化 $\int_a^b F\,dx$。
- **第一变分** $\delta J = \varphi'(0)$；极小点的必要条件是 $\delta J = 0$ 对一切扰动成立。
- 分部积分消去 $\eta'$，再借**基本引理**，把「积分恒零」翻译成**Euler-Lagrange 方程** $F_y - \frac{d}{dx}F_{y'} = 0$。
- E-L 方程是二阶 ODE，配合两个端点条件决定解；力学上它等价于牛顿第二定律。
- 最短路径是直线、最速降线是摆线——两个由「变分」长出来的经典答案。

在下一节，我们把「约束」（周长固定、长度固定、能量固定）引入变分问题，用 Lagrange 乘子处理**等周问题与约束变分**——答案同样漂亮：周长固定的闭合曲线里，圆围出的面积最大。
