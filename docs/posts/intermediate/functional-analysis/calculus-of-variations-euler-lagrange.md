---
title: 变分法初步：泛函的极值与欧拉-拉格朗日方程
date: 2026-08-07
---

# 变分法初步：泛函的极值与欧拉-拉格朗日方程

<div class="epigraph">
<p>自然界的定律总可写成「某个量取极值」——变分法就是寻找这个量的语言。</p>
<footer>—— 皮埃尔-路易 · 莫佩尔蒂（Pierre-Louis Maupertuis），意译</footer>
</div>

<div class="article-byline">
<p>第二级 · 泛函分析 ｜ 程其襄《泛函分析》§10.3 ｜ 2026-08-07</p>
</div>

## 为什么变分法是泛函分析的应用巅峰

微积分找「函数」的极值，**变分法（calculus of variations）**找「泛函」的极值——即「哪个函数 $y$ 使积分 $J(y) = \int F(t, y, y')\\,dt$ 最小」。悬链线（最小能量）、最速降线、光的最短时间原理、物理学的最小作用量原理，全是变分问题。泛函分析在这里大展身手：把「函数空间上的极值问题」纳入分析框架，用**欧拉-拉格朗日方程（Euler-Lagrange equation）**把「找函数」化为「解微分方程」。本节是变分法的入口：从一阶条件推导 E-L 方程，并看它在最速降线等经典问题中的应用。<span class="marginnote">变分法的核心思想与微积分完全平行：<strong>微积分里 $f'(x_0) = 0$ 是极值的一阶条件；变分法里「$J$ 沿任何方向的导数 = 0」给出 E-L 方程</strong>。把「方向导数」换成「变分 $\\delta y$」，把「数」换成「函数」——这就是「变分」二字的来源。</span>

## 1 泛函与变分

**定义**：设 $F(t, u, v)$ 是光滑函数，定义**泛函（functional）**

$$
J(y) = \int_a^b F\big(t, y(t), y'(t)\big)\\, dt
$$

作用在（满足边界条件 $y(a) = \alpha$、$y(b) = \beta$ 的）可微函数 $y$ 上。目标是找使 $J$ 取极小的 $y$。

**变分（variation）**：对 $y_0$ 的「扰动」$\varphi$（$\varphi(a) = \varphi(b) = 0$），考虑 $y_0 + \varepsilon\varphi$。**$J$ 在 $y_0$ 取极小的必要条件**：对一切 $\varphi$，

$$
\frac{d}{d\varepsilon}\Big|_{\varepsilon = 0} J(y_0 + \varepsilon\varphi) = 0
$$

即「$J$ 沿任何方向 $\varphi$ 的方向导数为零」——这是极值的一阶条件。<span class="marginnote">把「$y_0$ 是最小元」的几何含义展开：<strong>在函数空间里，$y_0$ 是「山沟」底部，任何方向的小扰动都不降低 $J$</strong>。方向导数（Gateaux 导数）为零是必要条件；凸性给充分条件。这个「一阶条件」框架与 Hilbert 空间最佳逼近的「残差垂直」完全同构（§4.6）。</span>

## 2 欧拉-拉格朗日方程

**定理（Euler-Lagrange 方程）**：若 $y_0$ 是 $J$ 的极值点，则 $y_0$ 满足

$$
\frac{\partial F}{\partial y} - \frac{d}{dt}\frac{\partial F}{\partial y'} = 0
$$

**推导（分部积分）**：

- **第一步，一阶条件**：$0 = \frac{d}{d\varepsilon}\int F(t, y_0+\varepsilon\varphi, y_0' + \varepsilon\varphi')\\,dt = \int \Big(F_y\,\varphi + F_{y'}\,\varphi'\Big)\\,dt$。
- **第二步，分部积分**：$\int F_{y'}\varphi'\\,dt = -\int \frac{d}{dt}F_{y'}\cdot\varphi\\,dt$（$\varphi$ 在端点为零，边界项消失）。
- **第三步，合并**：$\int \big(F_y - \frac{d}{dt}F_{y'}\big)\varphi\\,dt = 0$ 对一切 $\varphi$ 成立。
- **第四步，变分基本引理**：由 $\varphi$ 任意，被积函数必为零：$F_y - \frac{d}{dt}F_{y'} = 0$。<span class="marginnote">最后一步的「变分基本引理」是分析学的经典：<strong>若连续函数 $g$ 与一切（零端点）光滑函数 $\varphi$ 正交（$\\int g\\varphi = 0$），则 $g \\equiv 0$</strong>。这正是「$C_c^\\infty$ 稠密」的推论（§6.6 的泛函判别法）——变分法把「对一切扰动成立」翻译成「被积函数为零」。</span>

**核心要点：极值点 ⟹ E-L 方程**——把「找函数」化归为「解二阶常微分方程」。

## 3 例子：最速降线与悬链线

**例一（最速降线问题）**：质点在重力下从 $(0,0)$ 滑到 $(a,b)$，最短时间路径。泛函

$$
J(y) = \int_0^a \sqrt{\frac{1 + (y')^2}{2gy}}\\, dx
$$

E-L 方程的解是**摆线**（cycloid）：$x = \frac{k^2}{2}(\theta - \sin\theta)$、$y = \frac{k^2}{2}(1 - \cos\theta)$。这是伯努利 1696 年提出的著名问题，变分法的起点。

**例二（悬链线）**：两端固定的绳索在重力下的形状。泛函 $J(y) = \int y\sqrt{1+(y')^2}\\,dx$，E-L 方程给出

$$
y = a\cosh\frac{x - c}{a}
$$

（双曲余弦）——悬链线。它也是「最小能量曲线」。

**例三（测地线）**：曲面上的最短路径。$J = \int\sqrt{E\,u'^2 + 2F\,u'v' + G\,v'^2}\\,dt$（第一基本形式），E-L 方程给出测地线方程。<span class="marginnote">这三个例子展示了变分法的普适性：<strong>物理规律（最小时间、最小能量、最短路径）都写成「泛函取极值」</strong>。E-L 方程把它们统一为二阶 ODE——这是「自然界的极值原理」的数学形态。</span>

## 4 公式解析：E-L 方程的推导链

把推导的每一步的「角色」标注清楚：

$$
\frac{d}{d\varepsilon}\Big|_0 \int F(t, y+\varepsilon\varphi, y'+\varepsilon\varphi')\\,dt = \int (F_y \varphi + F_{y'}\varphi')\\,dt
$$

- **第一步（链式法则）**：对 $\varepsilon$ 求导，$F$ 的偏导 $F_y$、$F_{y'}$ 分别在 $y_0$ 处取值，乘以扰动 $\varphi$、$\varphi'$。
- **第二步（分部积分）**：$\int F_{y'}\varphi'$ 用分部积分，边界项 $\big[F_{y'}\varphi\big]_a^b = 0$（$\varphi$ 端点为零）。
- **第三步（合并）**：得 $\int\big(F_y - \frac{d}{dt}F_{y'}\big)\varphi = 0$。
- **第四步（基本引理）**：$\varphi$ 任意 ⟹ $F_y - \frac{d}{dt}F_{y'} = 0$。

**关键**：推导的全部困难在「分部积分处理 $\varphi'$」——它把「对 $y'$ 的偏导」变成「对 $t$ 的全导」。**E-L 方程是「变分 + 分部积分 + 基本引理」三步的产物**。

## 5 例题精讲：E-L 方程的求解

**例题一：$J(y) = \int_0^1 (y'^2 + y^2)\\,dt$，$y(0) = 0$、$y(1) = 1$**。

- $F = y'^2 + y^2$。$F_y = 2y$、$F_{y'} = 2y'$。
- E-L：$2y - \frac{d}{dt}(2y') = 0 \Rightarrow y'' = y$。
- 解 $y = A\sinh t + B\cosh t$，边界给 $y = \frac{\sinh t}{\sinh 1}$。

**例题二：$J(y) = \int_a^b \sqrt{1 + y'^2}\\,dt$（弧长）**。

- $F = \sqrt{1+y'^2}$（与 $y$ 无关）。E-L：$\frac{d}{dt}F_{y'} = 0 \Rightarrow F_{y'} = \frac{y'}{\sqrt{1+y'^2}} = c$。
- 解得 $y'$ 常数——直线。最短路径是直线。

**例题三：与 $t$ 无关的守恒**。

- 若 $F$ 不含 $t$，E-L 有一阶积分 $F - y'F_{y'} = \text{常数}$（Beltrami 恒等式）。
- 悬链线与最速降线都用它降阶求解。
- 这是「Noether 定理」在变分法里的雏形（对称性 ⟹ 守恒量）。

**核心要点**：E-L 方程的求解——显式、常数消去、守恒积分——三个例题展示「解变分问题」的三种标准技巧。

**辨析｜易错点：** E-L 方程是极值的**必要**条件，不是充分条件。需要二阶变分（Legendre 条件）或凸性判断极小/极大/鞍点。$y'' = y$ 的解可能是极大（如 $J = \int(y^2 - y'^2)$）——不要只看一阶条件。

## 6 小结

- **变分问题**：找 $y$ 使 $J(y) = \int F(t,y,y')\\,dt$ 极值；一阶条件 = 沿一切方向的方向导数为零。
- **E-L 方程**：$F_y - \frac{d}{dt}F_{y'} = 0$——极值点满足的 ODE。
- **推导**：变分 + 分部积分 + 变分基本引理。
- **经典问题**：最速降线（摆线）、悬链线、测地线、最小作用量。
- **求解技巧**：显式、Beltrami 守恒、边界条件。
- **定位**：变分法是 §10.4 变分原理、§10.5 边值问题的基础。

在下一节，我们研究**变分原理与里茨方法**——如何把变分问题离散化求解。
