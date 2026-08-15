---
title: Volterra 方程与预解核
date: 2026-08-07
---

# Volterra 方程与预解核

<div class="epigraph">
<p>因果律在方程里的样子，就是积分上限等于当下：此刻的值，只由过去决定。</p>
<footer>—— 维托 · 沃尔泰拉（Vito Volterra）</footer>
</div>

<div class="article-byline">
<p>第二级 · 积分方程 ｜ R. Kress《Linear Integral Equations》 第三章 ｜ 2026-08-07</p>
</div>

## 为什么 Volterra 方程如此友善

在分类那一课我们已经预言：Volterra 方程

$$y(x) = f(x) + \lambda \int_{a}^{x} K(x,t)\, y(t)\, dt$$

由于积分上限「钉在」自变量 $x$ 上，天然携带**因果结构**——$y(x)$ 只依赖 $t \le x$ 段上的信息，与 Fredholm 的「全局耦合」截然不同。正因如此，它享有两条 Fredholm 型没有的奢侈：

**第一，解对所有 $\lambda$ 存在唯一。** 没有特征值、没有择一、没有发散——上一课第二节的 $1/n!$ 论证已经揭示了原因。**第二，预解核有一个无比简单的级数。** 它不需要 Fredholm 行列式，只要反复「迭积」核就能写出来，而且级数永远收敛。

物理上，Volterra 方程是「记忆」的模型：材料的蠕变、种群的历史、信号的传播延迟，都写成「当下 = 强迫 + 过去的加权积分」。学这一节，我们一方面把预解核的工具箱补全，另一方面用 **Abel 方程** 打开通向奇异积分方程的大门。<span class="marginnote">Volterra 1860 年代研究积分方程，比 Fredholm 早了四十年。他关心的正是这类「因果」方程，因此教科书往往把 Volterra 方程与 Fredholm 方程并列，但解法精神完全不同：前者是「逐段推进」，后者是「整体求逆」。</span>

## 1 迭积核与 Volterra 预解核

对 Volterra 核，迭积核的定义与 Fredholm 情形只差一个积分上限：

$$K_1(x,t) = K(x,t), \qquad K_{n+1}(x,t) = \int_{t}^{x} K(x,s)\, K_n(s,t)\, ds$$

注意积分从 $t$ 到 $x$（而非从 $a$ 到 $b$）——这是「因果性」在公式层面的体现：**迭积核只在 $t \le x$ 时非零**。<span class="marginnote">迭积核 $K_n(x,t)$ 的物理意义：$t$ 时刻的一个「扰动」先被 $K_1$ 送到中间时刻 $s$，再被 $K_{n-1}$ 从 $s$ 送到 $x$，中间时刻被全部积分掉。它度量「$t$ 到 $x$ 的 $n$ 次跳转的累积效应」。</span>

**Volterra 预解核（resolvent kernel）**定义为

$$\Gamma(x,t;\lambda) = \sum_{n=1}^{\infty} \lambda^{n-1}\, K_n(x,t)$$

可以验证，方程的解是

$$y(x) = f(x) + \lambda \int_{a}^{x} \Gamma(x,t;\lambda)\, f(t)\, dt$$

代入即得：把 $y$ 的表达式代回方程，两边都是「$f$ 加上 $\lambda\int\Gamma f$」，等式由迭积核的递推恒等式 $\Gamma = K + \lambda\int K\Gamma$ 保证。**只要级数收敛，公式就是精确解。**

## 2 公式解析：为什么收敛半径是无穷大

这是全节最该盯住的一条不等式：

$$
|K_n(x,t)| \le \frac{M^n (x-t)^{n-1}}{(n-1)!}, \qquad M := \max|K(x,t)|
$$

- **第一步，看递推的来源**：$K_{n+1}(x,t) = \int_t^x K(x,s)K_n(s,t)ds$。$K$ 被 $M$ 控制，$K_n$ 被 $M^n(s-t)^{n-1}/(n-1)!$ 控制，代入得 $|K_{n+1}| \le M^{n+1}\int_t^x (s-t)^{n-1}/(n-1)!\, ds = M^{n+1}(x-t)^n/n!$。
- **第二步，认出场次**：每次迭积多出的正是「幂次除以阶乘」$z^n/n!$——这是指数函数 $e^z = \sum z^n/n!$ 的系数。于是预解核级数的通项被 $|\lambda|^{n-1} M^n (x-t)^{n-1}/(n-1)!$ 控制。
- **第三步，用比值判别法**：相邻两项之比 $\approx |\lambda| M (x-t)/n \to 0$，**与 $\lambda$ 无关**。这就是「Volterra 型对一切 $\lambda$ 可解」的定量根源，也顺带给出 $\Gamma$ 的增长界 $|\Gamma| \le M e^{|\lambda|M(x-t)}$。
- **第四步，对比 Fredholm**：Fredholm 的迭积核估计是 $M^n(b-a)^{n-1}$，缺了 $1/(n-1)!$，比值判别法给出硬条件 $|\lambda|M(b-a)\lt 1$。**一个阶乘符号之差，决定了整个 $\lambda$ 平面的两种命运。**

## 3 与常微分方程的联系：求导把它变回 IVP

Volterra 第二类方程与初值问题存在精确的互化。设 $K, K_x$ 连续，对

$$y(x) = f(x) + \lambda \int_{a}^{x} K(x,t)\, y(t)\, dt$$

两边对 $x$ 求导，用 Leibniz 法则：

$$y'(x) = f'(x) + \lambda K(x,x)\, y(x) + \lambda \int_{a}^{x} \frac{\partial K}{\partial x}(x,t)\, y(t)\, dt$$

这是一个**积分微分方程**。特殊地，若核不依赖 $x$（即 $K(x,t) = K(t)$），第三项消失：

$$y'(x) = f'(x) + \lambda K(x)\, y(x), \qquad y(a) = f(a)$$

变成一个**一阶线性 ODE 初值问题**——可分离变量直接解出。<span class="marginnote">反过来，任何一阶 ODE 初值问题 $y' = g(x,y)$、$y(a)=y_0$ 都可以两边积分写成 $y(x) = y_0 + \int_a^x g(t,y(t))dt$——这正是 Picard 迭代的出发点。<strong>Volterra 方程与 IVP 是同一枚硬币的两面</strong>，这也解释了为什么它的解「逐段推进」、为什么对所有 $\lambda$ 稳定。</span>

**辨析｜易错点：** 求导法要求核 $K(x,t)$ 对 $x$ 连续可微；核有角点（如 $K = \min(x,t)$）时不能贸然求导。另外，微分方程转换成 Volterra 方程时，**初值条件被积分自动吸收**，转换后不再需要单独携带初值——这是积分方程「化边值/初值为积分约束」普遍威力的第一个实例。

## 4 Abel 积分方程：弱奇异 Volterra 方程

把核换成带奇点的形式，就进入奇异积分方程的领域。**Abel 方程**（第一类弱奇异 Volterra）

$$\int_{0}^{x} \frac{y(t)}{(x-t)^{\alpha}}\, dt = f(x), \qquad 0 \lt  \alpha \lt  1$$

它来自 Abel 对**等时曲线**问题的研究：质点沿光滑曲线无摩擦下滑，给定下落时间与高度的关系，反求曲线形状。$t = x$ 处核 $(x-t)^{-\alpha}$ 发散，但积分仍收敛（弱奇异）。<span class="marginnote">等时曲线（tautochrone）问题：无论质点从多高处释放，滑到最低点所需时间相同——答案是摆线。Abel 1823 年把它化成了上面的积分方程并首次给出反演公式，这比 Fredholm 理论早了整整八十年，是积分方程最早的精确解。</span>

**Abel 反演公式**：方程的解为

$$y(x) = \frac{\sin(\pi\alpha)}{\pi}\, \frac{d}{dx} \int_{0}^{x} \frac{f(t)}{(x-t)^{1-\alpha}}\, dt$$

$\alpha = 1/2$ 时系数 $\sin(\pi/2)/\pi = 1/\pi$。这个公式把「解一个奇异积分方程」变成「对 $f$ 先做一次分数阶积分、再求导」——它就是**分数阶积分与微分的雏形**，也是现代分数阶微积分理论的历史源头。

值得留意的是，Abel 方程是**第一类**方程，按理「不适定」；但这里的反演公式却给出了稳定而显式的解。秘密在于核的弱奇异结构恰好让反演算子良定——它提醒我们，「第一类总是不适定」是**过度概括**：核的特殊结构可以拯救它。

## 5 第一类 Volterra 方程：求导降阶

第一类 Volterra 方程

$$\int_{a}^{x} K(x,t)\, y(t)\, dt = f(x)$$

没有那个「孤零零的 $y(x)$」，理论地位与 Fredholm 第一类一样尴尬。但好在 Volterra 型有「因果」这把钥匙：**两边对 $x$ 求导**，把第一类变成第二类。

若 $K(x,x) \neq 0$，求导得

$$K(x,x)\, y(x) + \int_{a}^{x} \frac{\partial K}{\partial x}(x,t)\, y(t)\, dt = f'(x)$$

整理成标准第二类形式：

$$y(x) = \frac{f'(x)}{K(x,x)} - \int_{a}^{x} \frac{K_x(x,t)}{K(x,x)}\, y(t)\, dt$$

于是第一类 Volterra 方程在 $K(x,x)\neq 0$ 时总能化归第二类，从而对所有 $\lambda$ 可解。**辨析｜易错点：** 这个化归要求 $f$ 可微且 $f(a) = 0$（把 $x=a$ 代入原方程即得相容条件）。若 $K(x,x) = 0$ 或 $f(a) \neq 0$，第一类 Volterra 方程可能无解或解不唯一——此时需要高阶求导或其他技巧。

## 6 小结

- Volterra 方程有**因果结构**，解对所有 $\lambda$ 存在唯一，预解核为 $\Gamma = \sum_{n\ge1}\lambda^{n-1}K_n$，收敛半径**无穷大**。
- 迭积核满足 $K_{n+1}(x,t) = \int_t^x K(x,s)K_n(s,t)ds$，估计 $|K_n| \le M^n(x-t)^{n-1}/(n-1)!$ 是「无条件收敛」的根。
- 两边求导把 Volterra 方程化为**积分微分方程**；核与 $x$ 无关时化为**一阶线性 ODE 初值问题**。
- **Abel 方程** $\int_0^x y(t)/(x-t)^\alpha\,dt = f(x)$ 有反演公式，是分数阶微积分的发端。
- 第一类 Volterra 方程在 $K(x,x)\neq 0$ 时可**求导化归第二类**，相容条件 $f(a)=0$