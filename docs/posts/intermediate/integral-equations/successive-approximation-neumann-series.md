---
title: 逐次逼近法与 Neumann 级数
date: 2026-08-07
---

# 逐次逼近法与 Neumann 级数

<div class="epigraph">
<p>把解当成极限来造：先猜一个，代进去，再用得到的新的猜——反复迭代，直到稳定下来。</p>
<footer>—— 卡尔 · 诺伊曼（Carl Neumann）</footer>
</div>

<div class="article-byline">
<p>第二级 · 积分方程 ｜ F. G. Tricomi《Integral Equations》 第二章 ｜ 2026-08-07</p>
</div>

## 为什么从逐次逼近开始

上一节我们把第二类非齐次 Fredholm 方程

$$y(x) = f(x) + \lambda \int_{a}^{b} K(x,t)\, y(t)\, dt$$

确立为整个专题的主战场。现在的问题是：**怎么解它？** 求精确公式解需要等待退化核与 Fredholm 理论，但我们先拿起最朴素、也最普适的一件武器——**不动点迭代**。<span class="marginnote">迭代思想在《常微分方程》里出现过一次：Picard 逐次逼近证明了初值问题解的存在唯一性。这里的 Neumann 级数是同一思想换了一身衣裳：把「解」当成一个收敛级数的和，而不是一步步拼出来的折线。</span>

这个方法给出两样东西：一是**解的存在唯一性证明**（在 $\lambda$ 足够小、核足够好时），二是解的**级数表达式**——它叫 **Neumann 级数（Neumann series）**，是后续一切分析（Fredholm 理论、数值方法、谱理论）的共同起点。它背后的直觉极其简单，简单到让人怀疑它是否配得上「理论」二字：**把方程看成 $y = f + \lambda K y$，右边出现 $y$，就把 $y$ 再代成 $f + \lambda K y$，如此往复。**

## 1 把方程看成算子不动点

把积分算子记为

$$(Ky)(x) = \int_{a}^{b} K(x,t)\, y(t)\, dt$$

则第二类方程浓缩成一个不动点方程

$$y = f + \lambda K y$$

它和数值分析里解 $x = g(x)$ 的迭代法是同一件事：**算子 $T(y) = f + \lambda K y$ 把函数 $y$ 映射成新函数，方程的解正是 $T$ 的不动点**。<span class="marginnote">为什么想「迭代」？因为 $T$ 看起来是「压缩」的：若两个函数 $u,v$ 只差一点点，$K$ 的积分会把它们之间的差进一步缩小（至少当 $\lambda$ 够小、核够光滑时）。压缩映射原理（Banach 不动点定理）告诉我们，压缩映射的迭代必收敛到唯一不动点。</span>

定义迭代序列

$$y_0 = f, \qquad y_{n+1} = f + \lambda K y_n, \qquad n = 0,1,2,\dots$$

若 $y_n$ 收敛到某个 $y^*$，则在两边取极限得 $y^* = f + \lambda K y^*$，即 $y^*$ 是解。剩下的事只有一件：**证明收敛**。这需要把 $y_n$ 的显式形式写出来，它天然是一个级数。

## 2 Neumann 级数的导出

把迭代展开，前几项是这样：

$$
\begin{aligned}
y_1 &= f + \lambda K f \\
y_2 &= f + \lambda K(f + \lambda K f) = f + \lambda K f + \lambda^2 K^2 f \\
y_3 &= f + \lambda K f + \lambda^2 K^2 f + \lambda^3 K^3 f
\end{aligned}
$$

规律一目了然：

$$y_n = \sum_{k=0}^{n} \lambda^k K^k f, \qquad K^0 f := f$$

若级数收敛，则

$$y(x) = \sum_{k=0}^{\infty} \lambda^k (K^k f)(x)$$

这个无穷级数就是 **Neumann 级数**。它的每一项都是「把 $f$ 交给积分算子 $K$ 作用 $k$ 次」，形式上完全类比等比级数求和

$$\frac{1}{1 - \lambda K} = \sum_{k=0}^{\infty} (\lambda K)^k$$

**辨析｜易错点：** 别把 $K^k$ 当成核的 $k$ 次幂（乘法），它是**算子复合**：$K^2 f = K(Kf)$。但为了写出被积显式，我们把每次复合都展开成积分，就得到**迭积核（iterated kernels）**的概念，见下一节。

## 3 迭积核与收敛条件

定义一列核 $K_n(x,t)$：

$$K_1(x,t) = K(x,t), \qquad K_{n+1}(x,t) = \int_{a}^{b} K(x,s)\, K_n(s,t)\, ds$$

它们满足：对任意 $f$，

$$(K^n f)(x) = \int_{a}^{b} K_n(x,t)\, f(t)\, dt$$

于是 Neumann 级数可以逐项写成普通积分：

$$y(x) = f(x) + \lambda \int_{a}^{b} K(x,t) f(t)\, dt + \lambda^2 \int_{a}^{b} K_2(x,t) f(t)\, dt + \cdots$$

现在收敛性一目了然。设核有界：$|K(x,t)| \le M$，则对迭积核有估计

$$|K_{n+1}(x,t)| \le M^{n+1} (b-a)^n$$

于是 Neumann 级数在

$$|\lambda| \lt  \frac{1}{M(b-a)}$$

时关于 $x$ 一致收敛，且和函数就是方程的唯一解。<span class="marginnote">这个半径是「压缩」的定量版本：$T$ 是压缩映射当且仅当 $|\lambda| M (b-a) \lt  1$，恰好就是上面这个不等式。条件不足时迭代未必收敛，解也许仍存在——那就得动用退化核或 Fredholm 理论把收敛半径撑到「除特征值外的所有 $\lambda$」。</span>

**要点：** 收敛半径 $R = 1/[M(b-a)]$ 只依赖核的**上界**与区间长度，与自由项 $f$ 无关。$f$ 只影响级数的每一项，不影响它收不收敛——这正符合「方程的性质由核决定」的一贯直觉。

## 4 Volterra 情形：收敛半径自动变成无穷大

对 Volterra 型方程

$$y(x) = f(x) + \lambda \int_{a}^{x} K(x,t)\, y(t)\, dt$$

同一套迭代照样适用，但迭积核的估计要精细得多。上界 $|K|\le M$ 时，

$$|K_2(x,t)| \le M^2 (x - t), \qquad |K_3(x,t)| \le \frac{M^3 (x-t)^2}{2!}, \qquad \dots$$

归纳得

$$|K_{n+1}(x,t)| \le \frac{M^{n+1} (x-t)^n}{n!}$$

因子 $1/n!$ 一出场，局势彻底改变：<span class="marginnote">$1/n!$ 比任何几何级数都收敛得快，于是 $|\lambda|$ 无论多大，级数 $\sum (\lambda M)^{n+1}(b-a)^n/n!$ 都像 $e^{\lambda M (b-a)}$ 一样收敛。这就是「Volterra 型几乎无条件可解」的数学来源。</span>对任意 $\lambda$，Neumann 级数在 $[a,b]$ 上一致收敛，解存在且唯一。

这解释了上一节那句「Volterra 型对应初值问题，Fredholm 型对应边值问题」的深层原因：**Volterra 算子的迭积核被阶乘镇压，迭代天然收敛；Fredholm 算子则会被参数 $\lambda$ 卡住，收敛与否取决于 $\lambda$ 是否掉进特征值附近的陷阱。**

## 5 公式解析：为什么 $1/n!$ 能压住一切

把 Volterra 的估计推到极致，核心是这条递推：

$$
|K_{n+1}(x,t)| \le \int_{t}^{x} M \cdot \frac{M^n (s-t)^{n-1}}{(n-1)!}\, ds
= \frac{M^{n+1} (x-t)^n}{n!}
$$

三步拆开看：

- **第一步，积分限为何是 $t$ 到 $x$**：Volterra 核只在 $t \le s \le x$ 的非三角区域非零，所以迭积核的积分上限从固定的 $b$ 缩小成「当前自变量」$x$。这是 Volterra 与 Fredholm 在计算层面的全部差别。
- **第二步，$s$ 的积分怎么算**：$\int_t^x (s-t)^{n-1} ds = (x-t)^n/n$，再乘上递推里带出的 $M^n/(n-1)!$，合并得到 $M^{n+1}(x-t)^n/n!$。**递推每走一步，就多出一个「幂次除以阶乘」**，这正是指数函数 $e^{z} = \sum z^n/n!$ 的展开系数。
- **第三步，级数为什么对所有 $\lambda$ 收敛**：把估计代进 Neumann 级数，通项被 $|\lambda|^n M^n (b-a)^n / n!$ 控制，而这个数列比等比数列还快地被 $1/n!$ 拉向 0。比值判别法给出比值 $|\lambda| M (b-a)/n \to 0 \lt  1$，与 $\lambda$ 的大小无关。
- **第四步，对比 Fredholm 情形**：Fredholm 的估计缺了 $1/n!$，只剩 $|\lambda|^n M^n (b-a)^n$，比值恒为 $|\lambda|M(b-a)$，于是收敛当且仅当它小于 1。**一个阶乘，划出了「无条件可解」与「条件收敛」两种命运。**

## 6 小结

- 第二类方程 $y = f + \lambda K y$ 可视为**不动点问题**，逐次逼近 $y_{n+1} = f + \lambda K y_n$ 给出解的构造。
- 迭代展开得到 **Neumann 级数** $y = \sum_{k=0}^\infty \lambda^k K^k f$，其结构形如 $(I - \lambda K)^{-1}$ 的几何级数展开。
- **迭积核** $K_{n+1}(x,t) = \int K(x,s)K_n(s,t)ds$ 把算子幂展开成普通积分，是分析收敛性的利器。
- Fredholm 情形在 $|\lambda| \lt  1/[M(b-a)]$ 时收敛；**Volterra 情形因迭积核带 $1/n!$，对所有 $\lambda$ 收敛**。
- 收敛半径只由核与区间决定，与自由项 $f$ 无关。

在下一节，我们处理一类特殊但极其重要的核——**退化核**。当核能写成有限个一元函数之积的和时，积分方程会坍缩成有限维线性方程组，Neumann 级数的收敛半径也因此自动撑到「除特征值外的一切 $\lambda$