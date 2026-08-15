---
title: Legendre 函数与球谐函数
date: 2026-08-07
---

# Legendre 函数与球谐函数

<div class="epigraph">
<p>球谐函数是数学物理里最优雅的乐器，而 Laplace 方程是奏响它的第一首曲子。</p>
<footer>—— 据皮埃尔-西蒙 · 拉普拉斯（Pierre-Simon Laplace）《天体力学》精神改写</footer>
</div>

<div class="article-byline">
<p>第二级 · 特殊函数 ｜ 王竹溪、郭敦仁《特殊函数概论》 第 4 章 ｜ 2026-08-07</p>
</div>

## 为什么从 Legendre 函数开始

从球对称的物理问题出发：引力势、静电场、温度场、量子力学中的角动量，全都绕不开**球坐标下的 Laplace 方程**。用分离变量法把球坐标的 Laplace 方程拆开，角向部分给出的常微分方程正是 **Legendre 方程**，它的解就是 Legendre 函数与球谐函数。<span class="marginnote">分离变量法是数学物理方法的枢纽：在球坐标里设 $u(r,\theta,\phi) = R(r)\Theta(\theta)\Phi(\phi)$，方程被拆成三个独立的常微分方程，其中 $\theta$ 部分经变换 $x = \cos\theta$ 后恰为 Legendre 方程。这套「分离变量→本征值问题→特殊函数解」的流程，在《数学物理方法》与《偏微分方程》两篇中有完整的教学展开。</span>更重要的是，球谐函数构成了 $L^2$ 球面上的完备正交基——任何球面上的函数都能像 Fourier 级数那样展开成球谐级数，这是多极展开、原子轨道、地球物理重力场建模的共同语言。

## 1 从 Laplace 方程到 Legendre 方程

球坐标 $u(r,\theta,\phi)$ 下 Laplace 方程 $\nabla^2 u = 0$ 分离变量，角向本征方程化为

$$
\frac{1}{\sin\theta}\frac{d}{d\theta}\left(\sin\theta \frac{d\Theta}{d\theta}\right) + \left[\lambda - \frac{m^2}{\sin^2\theta}\right]\Theta = 0
$$

令 $x = \cos\theta$，$\Theta(\theta) = P(x)$，代入并化简得 **Legendre 方程的连带形式**：

$$
(1 - x^2)\,y'' - 2x\,y' + \left[\lambda - \frac{m^2}{1 - x^2}\right] y = 0
$$

当 $m = 0$ 时退化为 **Legendre 方程**：

$$
(1 - x^2)\,y'' - 2x\,y' + \lambda\, y = 0
$$

方程在 $x = \pm 1$ 有正则奇点（这两点对应球坐标的南北极）。Frobenius 方法给出解，而**要求解在 $x \in [-1,1]$ 上有限**（物理上有界性）就迫使 $\lambda = l(l+1)$，$l = 0, 1, 2, \dots$。<span class="marginnote">这是「本征值被边界条件量子化」的最经典教学案例：只要要求球面上的解正则，$l$ 就被迫取非负整数，连带地 $m$ 也要满足 $|m| \le l$。球谐函数的这些离散指标（$l,m$）日后在原子物理里就是轨道角动量量子数与磁量子数。</span>

## 2 Legendre 多项式与生成函数

**Legendre 多项式（Legendre polynomials）** 是 $m=0$ 时 Legendre 方程在 $\lambda = l(l+1)$ 下的多项式解，记作 $P_l(x)$，$l$ 次，且有归一化条件 $P_l(1) = 1$。前几项是

$$
P_0 = 1, \quad P_1 = x, \quad P_2 = \frac{3x^2 - 1}{2}, \quad P_3 = \frac{5x^3 - 3x}{2}, \quad P_4 = \frac{35x^4 - 30x^2 + 3}{8}
$$

它们可以用 **Rodrigues 公式** 统一给出：

$$
P_l(x) = \frac{1}{2^l\, l!}\, \frac{d^l}{dx^l}\left(x^2 - 1\right)^l
$$

**生成函数** 是 Legendre 多项式最优雅的入口：

$$
\frac{1}{\sqrt{1 - 2xt + t^2}} = \sum_{l=0}^{\infty} P_l(x)\, t^l
$$

它由「单位电荷放在 $z$ 轴上某点时，在场点 $x$ 处的 $1/r$ 的展开」物理直觉直接推出。<span class="marginnote">生成函数的思想是「把一族函数打包成一个母函数，用 Taylor 系数逐个取出」：$1/\sqrt{1-2xt+t^2}$ 对 $t$ 展开的系数就是 $P_l(x)$。生成函数方法在本专题《生成函数方法》一章有系统展开——它是组合数学与特殊函数共用的王牌技术。</span>

## 3 正交性、完备性与展开

Legendre 多项式在区间 $[-1,1]$ 上构成**完备正交系**：

$$
\int_{-1}^{1} P_l(x)\,P_{l'}(x)\, dx = \frac{2}{2l+1}\,\delta_{ll'}
$$

**正交性 + 完备性 = 展开定理**：任何在 $[-1,1]$ 上足够光滑的函数 $f$ 都能展开为

$$
f(x) = \sum_{l=0}^{\infty} a_l\, P_l(x), \qquad a_l = \frac{2l+1}{2}\int_{-1}^{1} f(x)\, P_l(x)\, dx
$$

这正是「把 Fourier 级数的正弦-余弦基换成 Legendre 基」。它不是一个抽象游戏——静电场里给定边界电位求内部电位，就是把边界电位按 $P_l$ 展开，每一项对应一个多极矩。<span class="marginnote">对比：Fourier 级数用 $\{1, \cos nx, \sin nx\}$ 展开周期函数，Legendre 级数用 $\{P_l\}$ 展开 $[-1,1]$ 上的函数，Chebyshev 级数用 $\{T_n\}$ 展开并兼顾数值逼近。三种「正交多项式展开」的异同，在《正交多项式》与《数值分析》两篇各有专门讨论。</span>用矩阵语言说，$\{P_l\}$ 是 $L^2[-1,1]$ 的一组正交基，$a_l$ 是 $f$ 在这组基上的投影坐标——线性代数里的投影公式在此原样重现。

## 4 公式解析：多极展开中的 $1/|\mathbf{r} - \mathbf{r}'|$

**为什么球谐函数能描述「远处的场」？** 核心是一条物理味十足的展开公式。设源点电荷位于 $\mathbf{r}'$，观测点在 $\mathbf{r}$，则 $r_>$ 为两者较大的距离。若 $r > r'$，

$$
\frac{1}{|\mathbf{r} - \mathbf{r}'|} = \frac{1}{r} \sum_{l=0}^{\infty} \left(\frac{r'}{r}\right)^{l} P_l(\cos\gamma)
$$

其中 $\gamma$ 是 $\mathbf{r}$ 与 $\mathbf{r}'$ 的夹角。逐步拆解：

- **第一步，认出生成函数**：令 $x = \cos\gamma$、$t = r'/r$，则左边正是 $1/\sqrt{r^2 - 2rr'\cos\gamma + r'^2} = \frac1r (1 - 2xt + t^2)^{-1/2}$。**这条展开就是 Legendre 生成函数的物理版本**。
- **第二步，读出物理**：第 $l$ 项的系数 $(r'/r)^{l}$ 说明——距源越远，高次项衰减越快。展开在 $r' \lt  r$ 时收敛，$r' = r$ 是收敛边界（对应 $|t|=1$）。
- **第三步，推广到角向**：利用球谐加法定理 $P_l(\cos\gamma) = \frac{4\pi}{2l+1}\sum_{m=-l}^{l} Y_l^{m*}(\theta',\phi')\,Y_l^m(\theta,\phi)$，可把上式改写为「按球谐函数展开」的形式，从而把**任意电荷分布产生的势写成球谐级数**——这就是电动力学里多极展开的标准起点。<span class="marginnote">加法定理把「两点间的夹角依赖」分解成「两点各自的角度函数之积的和」，是球谐函数最重要的恒等式之一。它的证明可由转动对称性给出：$P_l(\cos\gamma)$ 在两坐标同时转动下不变，而唯一不变的双线性式正是 $\sum_m Y_l^{m*}Y_l^m$。</span>
- **第四步，联系量子力学**：同样的展开在量子力学里给出「$\frac{1}{|\mathbf{r}-\mathbf{r}'|}$ 的角动量本征分解」，即 Coulomb 相互作用的第二量子化形式。

## 5 连带 Legendre 函数与球谐函数

当 $m \neq 0$ 时，Legendre 方程的解是**连带 Legendre 函数** $P_l^m(x)$，由 $P_l$ 求导得到：

$$
P_l^m(x) = (-1)^m (1 - x^2)^{m/2}\, \frac{d^m}{dx^m} P_l(x), \qquad 0 \le m \le l
$$

对负 $m$ 有 $P_l^{-m}(x) = (-1)^m \frac{(l-m)!}{(l+m)!} P_l^m(x)$。把它们与 $\phi$ 方向的 $e^{im\phi}$ 组装，就得到**球谐函数（spherical harmonics）**：

$$
Y_l^m(\theta, \phi) = (-1)^m \sqrt{\frac{2l+1}{4\pi}\frac{(l-m)!}{(l+m)!}}\, P_l^m(\cos\theta)\, e^{im\phi}
$$

球谐函数满足**正交归一**关系

$$
\int_0^{2\pi}\int_0^\pi Y_{l}^{m*}(\theta,\phi)\, Y_{l'}^{m'}(\theta,\phi)\,\sin\theta\, d\theta\, d\phi = \delta_{ll'}\delta_{mm'}
$$

并在球面上完备。<span class="marginnote">$Y_l^m$ 是量子力学角动量算符 $\mathbf{L}^2$ 与 $L_z$ 的公共本征函数：$\mathbf{L}^2Y_l^m = l(l+1)\hbar^2 Y_l^m$、$L_zY_l^m = m\hbar Y_l^m$。这正是「角动量量子化」的数学来源，也解释了为何原子轨道用 $Y_l^m$ 命名（$s,p,d,f$ 对应 $l=0,1,2,3$）。</span>由 $Y_l^m$ 构成的**实球谐函数**（把 $e^{im\phi}$ 换成 $\cos m\phi$ 与 $\sin m\phi$）在图形学、天文学与地球物理里更常用，因为它们是实的、且对应固定的节面。

## 6 应用：从氢原子到地球物理

**氢原子波函数**：$\psi_{nlm}(r,\theta,\phi) = R_{nl}(r)\,Y_l^m(\theta,\phi)$，角向部分就是球谐函数；$s,p,d,f$ 轨道的形状直接由 $Y_l^m$ 的节面决定。<span class="marginnote">$Y_0^0$ 是球对称的 $s$ 轨道，$Y_1^0 \propto \cos\theta$ 是沿 $z$ 轴的 $p_z$ 轨道，$Y_1^{\pm 1} \propto \sin\theta\, e^{\pm i\phi}$ 对应 $p_x \pm ip_y$。角向节面的个数 = $l$，这就是「角量子数决定轨道形状」的几何直觉。</span>
**静电多极展开**：如上节，点电荷分布的势场按 $l$ 分解为单极、偶极、四极……每一项对应一组 $Y_l^m$ 矩。
**地球物理与卫星重力场**：地球重力场用球谐系数（Stokes 系数）描述，卫星测高反演地球形状的正交展开正是 $Y_l^m$。<span class="marginnote">EGM2008 地球重力场模型把地球重力位展开到 2190 阶的球谐级数——这是球谐完备性最宏大的工程应用之一。任何「球面上分布的场」（重力、地磁、温度、光强）都可以用同一套语言展开。</span>
- **数值方法**：球面上求积与插值、球面调和分析（spherical harmonic transform，SHT）是气候模式与宇宙微波背景（CMB）数据处理的标准工具。

**辨析｜易错点：** 第一，$P_l^m$ 与 $P_l$ 的归一化不同——$P_l^m$ 的正交范数是 $\frac{2}{2l+1}\frac{(l+m)!}{(l-m)!}$，而不是简单的 $2/(2l+1)$；第二，$P_l^m$ 里 $|m| > l$ 时为零，初学者容易在求和范围上出错；第三，$Y_l^m$ 的相位约定（Condon–Shortley 因子 $(-1)^m$）并非唯一，阅读不同文献时要先确认相位约定是否一致。

## 7 小结

- **Legendre 方程** $(1-x^2)y'' - 2xy' + \lambda y = 0$ 来自球坐标 Laplace 方程分离变量；有界性条件迫使 $\lambda = l(l+1)$。
- **Legendre 多项式** $P_l$ 由 **Rodrigues 公式** $P_l = \frac{1}{2^l l!}\frac{d^l}{dx^l}(x^2-1)^l$ 给出，**生成函数** $1/\sqrt{1-2xt+t^2} = \sum P_l t^l$ 是其最优雅的封装。
- **正交完备性** $\int_{-1}^1 P_lP_{l'}dx = \frac{2}{2l+1}\delta_{ll'}$ 支撑了 Legendre 展开定理——Fourier 级数在 $[-1,1]$ 上的表亲。
- **连带 Legendre 函数** $P_l^m$ 与**球谐函数** $Y_l^m$ 处理 $m\neq0$ 情形，正交归一于球面，是角动量算符的本征函数。
- **多极展开** $\frac{1}{|\mathbf{r}-\mathbf{r}'|} = \frac1r\sum_l (\frac{r'}{r})^l P_l(\cos\gamma)$