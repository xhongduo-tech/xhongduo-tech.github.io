---
title: Bessel 函数（三类柱函数）
date: 2026-08-07
---

# Bessel 函数（三类柱函数）

<div class="epigraph">
<p>圆形的鼓面在振动的瞬间，就把自己交给了 Bessel 函数的谱。</p>
<footer>—— 弗里德里希 · 威廉 · 贝塞尔（Friedrich Wilhelm Bessel）工作精神之写照</footer>
</div>

<div class="article-byline">
<p>第二级 · 特殊函数 ｜ 王竹溪、郭敦仁《特殊函数概论》 第 5 章 ｜ 2026-08-07</p>
</div>

## 为什么从 Bessel 函数开始

圆柱与圆盘无处不在：鼓面的振动、光纤中的电磁模式、圆环内的热传导、行星轨道摄动、FM 调制信号。凡是**柱坐标或圆对称**下的波动、扩散与势场问题，分离变量都会把你引向同一族函数——**Bessel 函数**。<span class="marginnote">Bessel 函数的命名源自天文学家贝塞尔（1784—1846），他在研究开普勒方程的级数解时系统研究了这个函数；但欧拉、伯努利和拉格朗日在更早的悬链与振动问题里已经见过它的身影。19 世纪它在圆柱热传导与鼓膜振动中的成功，使它成为「柱函数」家族的代名词。</span>与 Legendre 函数（球问题）并列，Bessel 函数是特殊函数世界的两大支柱之一，而它「随阶数变化、振荡衰减」的复杂行为，也让它在数值方法里成为一道经典的考题。

## 1 从柱坐标 Laplace 方程到 Bessel 方程

柱坐标 $u(\rho, \phi, z)$ 下分离变量，令 $u = R(\rho)\Phi(\phi)Z(z)$，其中 $\rho$ 方向的方程在 $\Phi = e^{im\phi}$、$Z = e^{\pm kz}$ 的配合下化为

$$
\rho^2 R'' + \rho R' + \left(\lambda^2 \rho^2 - m^2\right)R = 0
$$

令 $x = \lambda\rho$、$y(x) = R(\rho)$，得 **Bessel 方程**：

$$
x^2 y'' + x\,y' + \left(x^2 - \nu^2\right) y = 0, \qquad \nu = m
$$

$\nu$ 称为**阶（order）**，在柱坐标问题里通常取整数 $m$，但 $\nu$ 也可以是任意实数乃至复数。<span class="marginnote">注意 Bessel 方程与 Legendre 方程的结构差异：Bessel 方程在 $x=0$ 有正则奇点，在 $x=\infty$ 有非正则奇点，整体上没有「两个有限正则奇点」的对称结构。因此它的解族比 Legendre 更「叛逆」——第二类解 $Y_\nu$ 在原点发散。</span>这是三大正交函数系（Legendre、Bessel、Fourier）里唯一一类「在 $[0,1]$ 上按零点加权正交」的函数，后文会看到它独特的正交形式。

## 2 第一类 Bessel 函数 $J_\nu$

用 Frobenius 方法求 Bessel 方程在 $x=0$ 附近的级数解。指标方程的两根为 $\pm\nu$，第一类解为

$$
J_\nu(x) = \sum_{k=0}^{\infty} \frac{(-1)^k}{k!\, \Gamma(\nu + k + 1)}\left(\frac{x}{2}\right)^{2k+\nu}
$$

**第一类 Bessel 函数 $J_\nu$（Bessel function of the first kind）** 在 $x=0$ 处有限（$\nu \ge 0$ 时 $J_\nu(0) = 0$，$J_0(0)=1$）。前几项与初等函数有明显的亲戚关系——例如半整数阶：

$$
J_{1/2}(x) = \sqrt{\frac{2}{\pi x}}\, \sin x, \qquad J_{-1/2}(x) = \sqrt{\frac{2}{\pi x}}\, \cos x
$$

**半整数阶 Bessel 函数退化成三角函数的根号包**，这一事实让「球 Bessel 函数」（spherical Bessel）$j_l(x) = \sqrt{\pi/(2x)}\,J_{l+1/2}(x)$ 全部变成「幂函数 × 三角函数」的有限组合——这是球散射、氢原子径向解里的熟面孔。<span class="marginnote">为什么半整数阶会这么干净？根源在生成函数与 $\sin,\cos$ 的指数表示：Bessel 生成函数 $e^{\frac{x}{2}(t - 1/t)}$ 在 $t = e^{i\theta}$ 时拆成 $\cos,\sin$ 的 Fourier 级数，这直接迫使半整数阶 $J$ 成为三角函数。这条「指数生成函数 ⇄ 三角函数」的链接是 Bessel 最迷人的伏笔。</span>

## 3 第二类与第三类 Bessel 函数

**第二类 Bessel 函数（Neumann 函数 / Weber 函数）$Y_\nu$** 是线性独立的第二个解。当 $\nu$ 非整数时

$$
Y_\nu(x) = \frac{J_\nu(x)\cos\nu\pi - J_{-\nu}(x)}{\sin\nu\pi}
$$

当 $\nu$ 为整数 $n$ 时上式是 $0/0$ 型不定式，需取极限 $\lim_{\nu\to n} Y_\nu(x)$，结果是含 $\ln(x/2)$ 与调和数 $H_k$ 的表达式。<span class="marginnote">$Y_n$ 在 $x\to 0$ 时以 $-\frac{2}{\pi}\ln(x/2)$（$n=0$）或 $-\frac{(n-1)!}{\pi}(x/2)^{-n}$（$n\ge1$）发散——这个对数项是「整数阶」特有的麻烦，也是很多初学手册把 $Y$ 单独列出的原因。物理上有界解通常排除 $Y$。</span>$Y_\nu$ 也常记为 $N_\nu$（Neumann），两记号并存于文献。

**第三类 Bessel 函数（Hankel 函数）** 是复组合：

$$
H_\nu^{(1)}(x) = J_\nu(x) + i\,Y_\nu(x), \qquad H_\nu^{(2)}(x) = J_\nu(x) - i\,Y_\nu(x)
$$

它们的渐近行为是行波：$H_\nu^{(1)}$ 对应外向柱面波 $e^{i(x - \nu\pi/2 - \pi/4)}/\sqrt{x}$，$H_\nu^{(2)}$ 对应内向波。**在散射与波传播问题里，Hankel 函数直接写着「向外辐射」的物理边界条件**（Sommerfeld 辐射条件）。<span class="marginnote">三类柱函数的区别可这样记：$J_\nu$ 在原点正则（驻波、本征函数），$Y_\nu$ 在原点发散（第二独立解、奇异场），$H_\nu^{(1,2)}$ 是行进波（散射态）。三者的渐近公式在 $x\gg\nu$ 时统一为 $\sqrt{2/(\pi x)}\cos(x - \nu\pi/2 - \pi/4)$ 的实部、虚部组合。</span>

## 4 公式解析：Bessel 生成函数与积分表示

Bessel 函数有一条极其优美的生成函数（由雅可比发现）：

$$
e^{\frac{x}{2}\left(t - \frac{1}{t}\right)} = \sum_{n=-\infty}^{+\infty} J_n(x)\, t^n
$$

逐步拆解它：

- **第一步，看左边**：$e^{\frac{x}{2}(t - 1/t)}$ 是整函数，对 $t$ 作 Laurent 展开，系数定义为 $J_n(x)$。这个定义与级数定义等价，但更具结构。
- **第二步，取 $t = e^{i\theta}$**：左边化为 $e^{ix\sin\theta}$，展开得 $e^{ix\sin\theta} = \sum_n J_n(x)\,e^{in\theta}$。**这正是 $e^{ix\sin\theta}$ 的 Fourier 级数**——于是 $J_n$ 是它的 Fourier 系数：
$$
J_n(x) = \frac{1}{2\pi}\int_{-\pi}^{\pi} e^{i(x\sin\theta - n\theta)}\, d\theta
$$
- **第三步，取实部**：令 $\theta \to \theta + \pi/2$ 或取实部，得到更常用的**泊松积分表示**：
$$
J_n(x) = \frac{1}{\pi}\int_0^{\pi} \cos(x\sin\theta - n\theta)\, d\theta
$$
- **第四步，读出物理**：$e^{ix\sin\theta}$ 正是**频率调制（FM）信号的数学形式**——因此「FM 信号的频谱分量就是 $J_n$」这句话，其实只是 Bessel 生成函数的一个坐标变换。同一个数学对象横跨振动、电磁与通信，这是特殊函数「通用语言」属性的又一例证。<span class="marginnote">调频广播、锁相环里用 Bessel 系数查表计算边带幅度，这正是本节公式的工程化。反过来，也可以把 Bessel 函数当「带振荡权重的广义 Fourier 基」，理解它在光学衍射（圆孔 Airy 斑）里出现的原因。</span>

**递推关系**是 Bessel 计算的主力：

$$
J_{\nu-1}(x) + J_{\nu+1}(x) = \frac{2\nu}{x}\,J_\nu(x), \qquad J_{\nu-1}(x) - J_{\nu+1}(x) = 2\,J_\nu'(x)
$$

它们与生成函数对 $t$、$x$ 求导等价，是数值计算与恒等式推导的标准工具。

## 5 正交性与 Bessel 级数

Bessel 函数在 $[0,1]$ 上按**加权正交**。设 $j_{\nu,k}$ 是 $J_\nu(x)$ 的第 $k$ 个正零点（$j_{\nu,1} \lt  j_{\nu,2} \lt  \cdots$），则

$$
\int_0^1 x\, J_\nu(j_{\nu,k}x)\, J_\nu(j_{\nu,l}x)\, dx = \frac{1}{2}\left[J_{\nu+1}(j_{\nu,k})\right]^2 \delta_{kl}
$$

由此得到 **Fourier–Bessel 级数**：$f(x) = \sum_{k=1}^{\infty} a_k J_\nu(j_{\nu,k}x)$，系数由加权投影给出。<span class="marginnote">对比 Legendre 的权重 1 与 Fourier 的权重 1，Bessel 的权重是 $x$——它的来源是柱坐标的体积元 $\rho\, d\rho\, d\phi\, dz$ 里的 $\rho$。<strong>每一族特殊函数都带着自己坐标系的度量</strong>：球坐标给 $x^2$（或 $\sin\theta$），柱坐标给 $x$。记住权重就记住了「这族函数属于哪类几何」。</span>工程上「鼓膜模态」「圆波导模式」「柱壳振动」的叠加都是 Fourier–Bessel 展开的实例。

## 6 Bessel 函数在天文、物理与工程中的舞台

**天体力学**：开普勒方程 $M = E - e\sin E$ 的解可用 $J_n(ne)$ 展开（Bessel 函数诞生于此），轨道摄动的级数表达至今沿用。<span class="marginnote">开普勒方程的解 $E - M = \sum_{n=1}^\infty \frac{2}{n}J_n(ne)\sin nM$——这是 Bessel 函数第一次被系统研究的真实场景。贝塞尔正是为解决这个天文学问题才建立了整套理论。</span>
- **电磁与光学**：圆波导的 $TE_{mn}/TM_{mn}$ 模式、圆孔衍射的 Airy 斑强度 $I \propto (2J_1(ka\sin\theta)/(ka\sin\theta))^2$、光纤的模场分布，都是 Bessel 谱。
- **声学与振动**：圆形鼓膜的位移 $u \propto J_m(j_{m,k}\rho)\cos(m\phi)$；圆柱体振动、环形热应力分析。
- **量子力学**：自由粒子柱面波、势阱问题；$J_{l+1/2}$ 出现在球势散射的相移里。
- **信号处理与概率**：Von Mises 分布、Rice 衰落、FM 频谱——Bessel 还渗透进通信与随机信号。

**辨析｜易错点：** 第一，$J_\nu(0)$ 只在 $\nu=0$ 时为 1、$\nu>0$ 时为 0，而 $Y_\nu(0)$ 发散——「原点值」随手查表最容易错。第二，零点 $j_{\nu,k}$ 与 $\nu$ 和 $k$ 都相关，Fourier–Bessel 级数的求和基是 $J_\nu(j_{\nu,k}x)$ 而非 $J_\nu(kx)$。第三，阶 $\nu$ 与自变量 $x$ 不要混：递推公式改的是阶（$\nu\pm1$），生成函数作用在自变量。

## 7 小结

- **Bessel 方程** $x^2y'' + xy' + (x^2 - \nu^2)y = 0$ 来自柱坐标分离变量，$\nu$ 为阶。
- **第一类 $J_\nu$** 在原点正则，级数含 $1/(k!\Gamma(\nu+k+1))$；**第二类 $Y_\nu$** 在原点对数/幂发散；**第三类 $H_\nu^{(1,2)}$** 是行进波组合，对应辐射条件。
- **生成函数** $e^{\frac{x}{2}(t-1/t)} = \sum_n J_n t^n$ 与 **积分表示** $J_n(x) = \frac1\pi\int_0^\pi\cos(x\sin\theta - n\theta)d\theta$ 给出最深刻的刻画。
- **加权正交** $\int_0^1 xJ_\nu(j_{\nu,k}x)J_\nu(j_{\nu,l}x)dx = \frac12[J_{\nu+1}(j_{\nu,k})]^2\delta_{kl}$ 支撑 **Fourier–Bessel 级数**。
- 半整数阶 $J_{1/2}(x) = \sqrt{2/(\pi x)}\sin x$