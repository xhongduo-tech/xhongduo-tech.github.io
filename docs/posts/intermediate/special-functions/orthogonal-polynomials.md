---
title: 正交多项式（Hermite / Laguerre / Jacobi）
date: 2026-08-07
---

# 正交多项式（Hermite / Laguerre / Jacobi）

<div class="epigraph">
<p>正交多项式是有权函数度量下的基，也是数值积分里隐藏的英雄。</p>
<footer>—— 卡尔 · 弗里德里希 · 高斯（C. F. Gauss）求积公式精神之写照</footer>
</div>

<div class="article-byline">
<p>第二级 · 特殊函数 ｜ 王竹溪、郭敦仁《特殊函数概论》 第 6 章 ｜ 2026-08-07</p>
</div>

## 为什么从正交多项式开始

前面几章我们分别遇见了 Legendre、Bessel，以及藏在 Kummer 函数里的 Laguerre 与 Hermite。现在该把它们**收编成一支队伍**了。**正交多项式（orthogonal polynomials）** 是满足「关于某个权重函数在某个区间上两两正交」的多项式族，Legendre、Chebyshev、Laguerre、Hermite、Jacobi 全是它的成员。<span class="marginnote">「正交」一词来自几何：把函数当无穷维向量，$\int_a^b f(x)g(x)w(x)dx$ 就是内积。正交多项式族就是内积空间里一组互相垂直的基——线性代数里「正交基、投影、坐标」的全部直觉，在这里原样成立。这是特殊函数与线性代数之间最漂亮的交汇点。</span>掌握正交多项式的统一理论，等于拿到了三个问题的通用解法：最佳逼近（Legendre/Chebyshev 展开）、高斯求积（数值积分）、以及量子力学里谐振子与氢原子的全部解。

## 1 统一框架：权重、区间与三项递推

**定义**：一族多项式 $\{P_n(x)\}$，$\deg P_n = n$，关于权重 $w(x)$ 在区间 $(a,b)$ 上满足

$$
\int_a^b P_m(x)\,P_n(x)\,w(x)\,dx = h_n\, \delta_{mn}
$$

就称为**正交多项式族（orthogonal polynomial system）**。著名成员与它们的「身世」如下：

| 族 | 记号 | 区间 | 权重 $w(x)$ | 源头方程 |
| --- | --- | --- | --- | --- |
| Legendre | $P_n$ | $[-1,1]$ | $1$ | 球坐标 Laplace |
| Chebyshev | $T_n$ | $[-1,1]$ | $(1-x^2)^{-1/2}$ | 逼近论、极值多项式 |
| Gegenbauer | $C_n^{(\lambda)}$ | $[-1,1]$ | $(1-x^2)^{\lambda-1/2}$ | 超球、$n$ 维转动 |
| Jacobi | $P_n^{(\alpha,\beta)}$ | $[-1,1]$ | $(1-x)^\alpha(1+x)^\beta$ | 超几何方程的特解 |
| Laguerre | $L_n^{(\alpha)}$ | $[0,\infty)$ | $x^\alpha e^{-x}$ | 氢原子径向 |
| Hermite | $H_n$ | $(-\infty,\infty)$ | $e^{-x^2}$ | 量子谐振子 |

**Jacobi 多项式是「总开关」**：Legendre（$\alpha=\beta=0$）、Chebyshev（$\alpha=\beta=\pm1/2$）、Gegenbauer（$\alpha=\beta=\lambda-1/2$）全部是它的特例。<span class="marginnote">这条谱系可以从超几何函数的视角一眼看穿：Jacobi 多项式写成 $P_n^{(\alpha,\beta)}(x) = \frac{(\alpha+1)_n}{n!}{}_2F_1(-n, n+\alpha+\beta+1; \alpha+1; \frac{1-x}{2})$，而超几何方程的参数化把所有「带幂权重」的正交多项式统一装进一个框架。这也是为什么前几章反复铺垫超几何函数。</span>

所有正交多项式族都满足**三项递推关系（three-term recurrence）**：

$$
x\, P_n(x) = A_n\, P_{n+1}(x) + B_n\, P_n(x) + C_n\, P_{n-1}(x)
$$

这条递推不是巧合，而是正交性的直接推论（Favard 定理甚至说：任何满足三项递推且首项系数为正的多项式族都必然正交）。**三项递推是正交多项式的生命线**：它给出快速求值（Clenshaw 算法）、零点计算（求对称三对角阵的特征值）与连分数表示。

## 2 三种最重要的成员

### Hermite 多项式 $H_n$

由 **Rodrigues 公式** 定义：

$$
H_n(x) = (-1)^n e^{x^2}\frac{d^n}{dx^n} e^{-x^2}
$$

满足 Hermite 方程 $y'' - 2xy' + 2ny = 0$，正交权重 $e^{-x^2}$。**量子谐振子的波函数** $\psi_n(x) \propto H_n(x)\,e^{-x^2/2}$——指数核 $e^{-x^2}$ 正是权重函数的平方根。<span class="marginnote">谐振子波函数把权重拆成「一半归一化、一半给多项式」：$\psi_n = \frac{1}{\sqrt{2^n n!\sqrt\pi}}e^{-x^2/2}H_n(x)$。生成函数 $e^{2xt - t^2} = \sum_n H_n(x)t^n/n!$ 是它的又一封装。Hermite 多项式在概率论里以「Hermite 多项式展开 Gram–Charlier 级数」的身份出现。</span>它也是**唯一在整条实轴上带高斯权重的正交多项式族**，这决定了它在统计与量子力学中的特殊地位。

### Laguerre 多项式 $L_n^{(\alpha)}$

$$
L_n^{(\alpha)}(x) = \frac{x^{-\alpha}e^{x}}{n!}\frac{d^n}{dx^n}\left(e^{-x}x^{n+\alpha}\right)
$$

权重 $x^\alpha e^{-x}$ 在 $[0,\infty)$。**氢原子径向函数的角向以外的部分**是 $e^{-x/2}x^{l}L_{n-l-1}^{(2l+1)}(x)$——这是 Kummer 函数 $-n$ 参数截断的直接结果（见《合流超几何函数》）。<span class="marginnote">Laguerre 方程 $xy'' + (\alpha+1-x)y' + ny = 0$ 正是 Kummer 方程在 $a=-n$ 时的样子，所以 Laguerre 多项式可以毫无悬念地写成 ${}_1F_1(-n;\alpha+1;x)$。这再次印证「合流超几何是 Hermite/Laguerre 的共同母体」这条主线。</span>

### Jacobi 多项式 $P_n^{(\alpha,\beta)}$

$$
P_n^{(\alpha,\beta)}(x) = \frac{\Gamma(\alpha+n+1)}{n!\,\Gamma(\alpha+\beta+n+1)}\sum_{k=0}^{n}\binom{n}{k}\frac{\Gamma(\alpha+\beta+n+k+1)}{\Gamma(\alpha+k+1)}\left(\frac{x-1}{2}\right)^k
$$

（当 $\alpha,\beta > -1$ 时正交）它是超几何家族里的「全才」，涵盖 Legendre 与 Chebyshev。**在数值分析里，Chebyshev 多项式 $T_n$ 有「最小最大偏差」性质**：在所有 $n$ 次首一多项式里，$T_n/2^{n-1}$ 的无穷范数最小——这使 Chebyshev 插值与逼近成为数值方法的最优选择。

## 3 公式解析：三项递推与 Christoffel–Darboux 核

**三项递推为什么是「一切的枢纽」？** 以 Hermite 为例：

$$
H_{n+1}(x) = 2x\,H_n(x) - 2n\,H_{n-1}(x), \qquad H_0 = 1,\ H_1 = 2x
$$

- **第一步，看出这是个「用已知构造未知」的引擎**：给两个初始多项式，递推就能生成整族，无需每次从 Rodrigues 公式重新求导。计算代价 $O(n)$，是数值库（如 `scipy.special.eval_hermite`）的标准做法。
- **第二步，理解零点**：把 $H_0,\dots,H_N$ 的三个递推写在一起，$H_n$ 的零点恰是某个对称三对角矩阵的特征值。**这连接了正交多项式与现代数值线性代数**——Gauss 求积的节点就是特征值，权重就是特征向量分量的平方。
- **第三步，引出 Christoffel–Darboux 核**：对前 $N$ 项求和，有

$$
\sum_{k=0}^{N} \frac{P_k(x)P_k(y)}{h_k} = \frac{A_N}{h_N}\,\frac{P_{N+1}(x)P_N(y) - P_N(x)P_{N+1}(y)}{x - y}
$$

这个核在谱理论、插值收敛性与随机矩阵论（正交多项式 + 行列式点过程）里反复出现，是「正交多项式理论最深刻的单条公式」之一。<span class="marginnote">在随机矩阵理论里，GUE 特征值的联合密度可以写成正交多项式行列式，而 Christoffel–Darboux 核直接给出特征值的相关函数——从求积公式到物理学家关心的能级关联，这条核是桥梁。详见《随机矩阵》类文献，与本站《随机过程》《线性代数》衔接。</span>

## 4 Gauss 求积：正交多项式对数值积分的主宰

**Gauss 求积公式（Gauss quadrature）** 是正交多项式最重要的工程应用。对权重 $w(x)$，取节点为 $P_{n+1}$ 的零点 $x_k$，权重为对应的求积系数 $w_k$，则

$$
\int_a^b f(x)\,w(x)\,dx \approx \sum_{k=1}^{n+1} w_k\, f(x_k)
$$

这一公式对**所有次数 $\le 2n+1$ 的多项式精确成立**——用 $n+1$ 个点换来了 $2n+1$ 次代数精度，是普通 Newton–Cotes 公式的两倍。<span class="marginnote">这个「用极少的点获得高阶精度」的魔术，本质是让节点也参与优化：普通求积只优化权重，Gauss 求积把节点也变成自由度。Gauss–Legendre（权重 1）、Gauss–Laguerre（权重 $e^{-x}$，半无穷区间）、Gauss–Hermite（权重 $e^{-x^2}$，全实轴）、Gauss–Jacobi（端点奇异权重）覆盖了工程上绝大多数积分场景。</span>对半无穷/全实轴区间，Gauss–Laguerre 与 Gauss–Hermite 求积几乎是唯一的选择——这正是「每个坐标几何配一族正交多项式」在数值维度的回报。

## 5 生成函数、Rodrigues 公式与逼近

三类工具贯穿全部正交多项式：

**Rodrigues 公式**：$P_n(x) = \frac{1}{k_n w(x)}\frac{d^n}{dx^n}\left[w(x)\,s(x)^n\right]$，用一个 $n$ 次求导给出整族。
**生成函数**：Hermite 的 $e^{2xt-t^2}$、Laguerre 的 $(1-t)^{-a-1}e^{-xt/(1-t)}$、Jacobi 的 $2^{\alpha+\beta}/R(1-t+R)^{\alpha}(1+t+R)^{\beta}$（$R=\sqrt{1-2xt+t^2}$）。
**微分方程**：每个家族对应一个二阶线性 ODE，是 Kummer 或超几何方程的特例。

在逼近论里，正交多项式给出**最佳均方逼近**的显式解：$f$ 在 $L^2_w(a,b)$ 上的最佳 $n$ 次多项式逼近，系数就是投影 $\langle f, P_k\rangle/\langle P_k, P_k\rangle$。<span class="marginnote">这与 Fourier 级数系数公式完全同构：投影到正交基的坐标 = 内积比。站在线性代数的高度看，Legendre 级数、Fourier 级数、Chebyshev 逼近、Bessel 级数全都是「正交基展开」这一件事的不同坐标实现。</span>数值上，Chebyshev 的极值性质还保证了多项式逼近在连续函数空间的最优收敛（Weierstrass 逼近定理的构造性证明之一）。

## 6 应用地图与易错点

- **量子力学**：谐振子用 Hermite，氢原子用 Laguerre，角动量用 Legendre/Jacobi——三大量子问题对应三大正交族，这不是巧合而是「分离变量 + 边界有界」的必然。
- **数值分析**：Gauss 求积、Chebyshev 插值、谱方法（spectral methods）的基函数。
- **概率统计**：Edgeworth/Gram–Charlier 展开（Hermite）、Beta 分布与 Jacobi 多项式、随机矩阵论。
- **信号与图像**：Zernike 多项式（正交于单位圆盘，是 Jacobi 多项式的极坐标形式）用于波前像差与图像矩描述。

**辨析｜易错点：** 第一，**权重必须带进内积**——「Legendre 权重 1、Hermite 权重 $e^{-x^2}$」常被忽视，导致正交性验证出错。第二，区间决定了权重：$(0,\infty)$ 配 Laguerre、$(-\infty,\infty)$ 配 Hermite，区间写错则正交性全毁。第三，Rodrigues 公式里 $w(x)$ 出现两次（内层作为被微分的乘积、外层作分母），漏掉外层 $1/w(x)$ 是初学者经典错误。第四，三项递推的系数 $A_n,B_n,C_n$ 各族不同，不能照抄。

## 7 小结

- **正交多项式族**：关于权重 $w(x)$ 在区间 $(a,b)$ 上两两正交的多项式；Legendre/Chebyshev/Jacobi/Gegenbauer/Laguerre/Hermite 是六大主力。
- **三项递推** $xP_n = A_nP_{n+1} + B_nP_n + C_nP_{n-1}$ 是所有族的共性，给出 $O(n)$ 求值与零点计算。
- **Jacobi 是总开关**：Legendre、Chebyshev、Gegenbauer 都是它的特例；Laguerre、Hermite 则是 Kummer（合流超几何）家族。
- **Gauss 求积** 用 $n+1$ 个节点达到 $2n+1$