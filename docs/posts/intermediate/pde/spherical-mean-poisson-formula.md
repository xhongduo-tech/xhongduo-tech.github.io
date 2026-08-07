---
title: 高维波动方程的球面平均法（Poisson 公式）
date: 2026-08-08
---

# 高维波动方程的球面平均法（Poisson 公式）

<div class="epigraph">
<p>三维空间的波，像球壳一样收缩又扩张。</p>
<footer>—— 基于泊松（Siméon Denis Poisson）与基尔霍夫（Gustav Kirchhoff）的工作</footer>
</div>

<div class="article-byline">
<p>第二级 · 偏微分方程 ｜ 谷超豪《数学物理方程》第四章 ｜ 2026-08-08</p>
</div>

## 为什么从球面平均开始

一维波动方程有了达朗贝尔公式，可现实中的波是立体的——声波在空气里向四面八方传播，地震波穿透整个地球。高维波动方程

$$
u_{tt} = a^2 \Delta u, \qquad u(x,0) = \varphi(x),\ u_t(x,0) = \psi(x)
$$

的求解需要新思想。**球面平均法**是其中最优雅的：把一个三维问题「降维」成一维问题。它的洞察是——从一点 $x$ 观察，远处的信息总是以球壳形式到达；对球面取平均，$u$ 的空间结构就被压缩成了关于「半径 $r$」的一维函数，而它恰好满足一维波动方程。最终得到三维 Poisson（Kirchhoff）公式。

## 1 球面平均算子

设 $u(x,t)$ 是 $\mathbb{R}^3$ 上的解，$x$ 固定。定义 $u$ 在以 $x$ 为球心、$r$ 为半径的球面上的平均值：

$$
\bar u(x, r, t) = \frac{1}{4\pi r^2}\oint_{|y - x| = r} u(y, t)\,dS_y
$$

$dS_y$ 是球面积分元，$\frac{1}{4\pi r^2}$ 归一化使常函数不变。<span class="marginnote">也可以写成单位球面上的形式：$\bar u(x,r,t) = \frac{1}{4\pi}\oint_{|\omega|=1}u(x + r\omega, t)\,dS_\omega$。把「球壳上取平均」看作一个算子，它把「$x$ 附近的球壳信息」汇总成一个数——这是后续所有推导的支点。</span>

关键性质：当 $r \to 0$ 时球面缩成一点，

$$
\lim_{r\to 0}\bar u(x,r,t) = u(x,t)
$$

于是只要能算出 $\bar u(x,r,t)$，令 $r = 0$ 就得到原解。

## 2 球面平均满足一维波动方程

把 $\bar u$ 对 $r$ 求二阶导，利用散度定理可以证明：**若 $u$ 满足三维波动方程，则 $\bar u$ 作为 $(r,t)$ 的函数满足一维波动方程**（对固定的 $x$）：

$$
\bar u_{tt} = a^2\left(\bar u_{rr} + \frac{2}{r}\bar u_r\right)
$$

右端不是标准的一维波动算子。为消去 $1/r$ 项，作变量替换

$$
v(x, r, t) = r\,\bar u(x, r, t)
$$

代入得

$$
v_{tt} = a^2 v_{rr}, \qquad r > 0
$$

**$v = r\bar u$ 满足标准的一维波动方程！**<span class="marginnote">这个替换是球面平均法的点睛之笔：$r\bar u$ 吸收了几何因子 $1/r^2$ 带来的发散项，把「三维球壳平均」的演化方程化成最干净的一维弦振动。这就是「降维」的实现——三维问题被压缩到「半径方向」的一维世界。</span>初值相应为

$$
v(x,r,0) = r\bar\varphi(x,r), \qquad v_t(x,r,0) = r\bar\psi(x,r)
$$

其中 $\bar\varphi, \bar\psi$ 是初始数据的球面平均。

## 3 公式解析：三维 Poisson（Kirchhoff）公式的导出

把 $v$ 看成「半无限弦」$r \ge 0$ 上的一维波动问题。注意 $v(x,0,t) = 0\cdot\bar u = 0$，故在 $r=0$ 处满足**齐次 Dirichlet 边界**——于是用上一节的**奇延拓**：

- **第一步，奇延拓。** 把 $v(x,r,0)$ 与 $v_t(x,r,0)$ 按 $r$ 奇延拓到 $r<0$：$v(x,-r,0) = -v(x,r,0)$。
- **第二步，用达朗贝尔公式。** 对奇延拓后的初值（记 $\Phi,\Psi$）：
  $$ v(x,r,t) = \frac{\Phi(r-at) + \Phi(r+at)}{2} + \frac{1}{2a}\int_{r-at}^{r+at}\Psi(\rho)\,d\rho $$
- **第三步，令 $r = 0$。** 由奇性，$\Phi(-at) = -\Phi(at)$，故
  $$ v(x,0,t) = \frac{-\Phi(at) + \Phi(at)}{2} + \frac{1}{2a}\int_{-at}^{at}\Psi(\rho)\,d\rho = \frac{1}{2a}\int_{-at}^{at}\Psi(\rho)\,d\rho $$
  奇延拓使 $\Psi(\rho)$ 为奇函数，积分 $\int_{-at}^{at}\Psi = 2\int_0^{at}\Psi$，于是 $u(x,t) = \bar u(x,0,t) = \lim_{r\to0}\frac{v(x,r,t)}{r}$。
- **第四步，整理成 Kirchhoff 公式。** 完成极限与换元后得到三维波动方程柯西问题的**Poisson 公式（Kirchhoff 公式）**：
  $$
  \boxed{\;u(x,t) = \frac{\partial}{\partial t}\big(t\,\bar\varphi(x,at)\big) + t\,\bar\psi(x,at)\;}
  $$
  展开写，即
  $$ u(x,t) = \frac{1}{4\pi a^2 t^2}\oint_{|y-x|=at}\Big[\psi(y) + \varphi(y) + \nabla\varphi(y)\cdot(y-x)\Big]\,dS_y $$

**Poisson 公式说：三维波在 $(x,t)$ 的值，只由以 $x$ 为心、$at$ 为半径的球面上初值的积分决定。** 球面上的信息——初位移、初速度、初位移的梯度——共同决定了这一时刻的波。

## 4 球面平均法的意义

球面平均法不止给了三维公式，更建立了高维理论的范式：

| 维度 | 解公式 | 到达结构 |
| --- | --- | --- |
| 1 维 | 达朗贝尔公式 | 区间 $[x-at, x+at]$ |
| 3 维 | Poisson（Kirchhoff）公式 | 球面 $|y-x| = at$ |
| 2 维 | 降维法（下一节） | 圆盘 $|y-x| \le at$ |

**维数决定了「到达结构」：一维靠区间、三维靠球面、二维靠圆盘（内部）。** 这个差别正是下一节 Huygens 原理与波的弥散的主题。<span class="marginnote">球面平均法的「降维」思想在 PDE 中反复出现：把高维问题通过某种平均/投影压缩到低维，再反演。除了这里的球面平均，还有第七篇傅里叶变换把「求导」变「乘法」的代数化、以及数值方法里的降维技巧——「先降维、后求解、再还原」是一套通用方法论。</span>

**辨析｜易错点：** 3 维公式里 $\bar\varphi(x,at)$ 依赖 $at$，即球面半径随时间增长——它不是一个固定曲面上的积分，而是**动球面**。初学者常把球心 $x$ 与半径 $at$ 搞混：半径固定为「波速 × 时间」，球心固定在观测点。此外，Kirchhoff 公式要求初值足够光滑（$\varphi \in C^3$、$\psi \in C^2$），公式里的 $\partial/\partial t$ 对 $\varphi$ 又提了一阶——这是高维公式比一维更「挑剔」的地方。

## 5 小结

- 球面平均 $\bar u(x,r,t)$ 把三维解压缩成「半径方向」的一维函数。
- 变量替换 $v = r\bar u$ 使 $v$ 满足标准一维波动方程——球面平均法的降维核心。
- 在 $r=0$ 处用奇延拓 + 达朗贝尔公式，令 $r\to 0$ 得到三维 Poisson（Kirchhoff）公式。
- 三维解只依赖 $x$ 为球心、$at$ 为半径的球面上初值。
- 维数决定到达结构：一维区间、三维球面、二维圆盘——通向 Huygens 原理。

在下一节，我们用降维法从三维公式得到二维波动方程的解。
