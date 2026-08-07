---
title: 两个重要极限：sin x/x 与 (1+1/x)^x
date: 2026-08-07
---

# 两个重要极限：sin x/x 与 (1+1/x)^x

<div class="epigraph">
<p>在无穷小量的研究中，$\frac{\sin x}{x}$ 与 $(1+\frac{1}{x})^x$ 是两颗最明亮的恒星：前者是几何的极限，后者是代数的极限。</p>
<footer>—— 莱昂哈德·欧拉（Leonhard Euler），《无穷小分析引论》（Introductio in Analysin Infinitorum）</footer>
</div>

<div class="article-byline">
<p>第二级 · 数学分析 ｜ 华东师大《数学分析》§3.4 ｜ 2026-08-07</p>
</div>

## 为什么是这两个极限

几乎所有初等函数的导数公式，都隐藏着两个特殊极限。求 $(\sin x)'$ 需要 $\lim_{x\to0}\frac{\sin x}{x}$；求 $(\ln x)'$、$(a^x)'$ 需要 $\lim_{x\to\infty}(1+\frac1x)^x=e$。它们是微积分「入口处」的两座必经关卡——**没有它们，三角与指数函数的微分理论就没有地基**。

而这两条极限的证明本身，也是「**先几何直觉、后严格论证**」的范本：它们用面积比较与夹逼，把肉眼可见的几何事实转化为代数不等式。<span class="marginnote">有趣的是，第一个极限依赖「弧度制」：$\frac{\sin x}{x}\to1$ 只在 $x$ 用弧度衡量时成立。若用度数，$x$ 不是弧长，公式就变成 $\frac{\sin x^\circ}{x}\to\frac{\pi}{180}$。弧度制让圆的相关公式长得最干净，是微积分约定俗成的单位——这也是为什么分析学里一切三角函数都用弧度。</span>

## 1 第一重要极限：lim sin x / x = 1

**第一重要极限**：

$$\lim_{x\to0}\frac{\sin x}{x}=1.$$

证明依赖单位圆上的**面积比较**。设 $0<x<\frac\pi2$，在单位圆中考虑三个对象：

- 三角形 $OAB$（扇形内）：面积 $=\dfrac12\sin x$（底 $1$，高 $\sin x$）；
- 扇形 $OAB$：面积 $=\dfrac12 x$（弧长 $x$，半径 $1$）；
- 三角形 $OAT$（外切）：面积 $=\dfrac12\tan x$（底 $1$，高 $\tan x$）。

从图形可读出的包含关系给出面积不等式 $\dfrac12\sin x<\dfrac12x<\dfrac12\tan x$，即

$$\sin x<x<\tan x\qquad(0<x<\frac\pi2).$$

两边除以 $\sin x>0$：$1<\dfrac{x}{\sin x}<\dfrac{1}{\cos x}$，取倒数得

$$\cos x<\frac{\sin x}{x}<1.$$

**注意此时 $x$ 为正**。对 $x\to0^-$ 的情形，令 $t=-x>0$，由 $\cos x$ 的偶性与 $\frac{\sin x}{x}$ 的偶性，同一不等式对 $x<0$ 也成立。于是利用已知极限 $\lim_{x\to0}\cos x=1$，由**夹逼定理**得 $\dfrac{\sin x}{x}\to1$。∎

> **辨析｜易错点：**夹逼必须**两个方向同时成立**。这里下界 $\cos x$ 与上界 $1$ 都趋于 $1$，中间的 $\frac{\sin x}{x}$ 被夹在中间不得不趋于 $1$。常见错误是只证 $x>0$ 就断言全局——**单侧不足以保证双侧极限**。不过本例幸运的是 $\frac{\sin x}{x}$ 是偶函数，正侧的夹逼自动给出负侧，这一句必须在证明里点明。

## 2 公式解析：第一重要极限的几何直觉

把面积比较翻译成一句话：**当 $x$ 很小（即弧很短）时，圆里那条弦与那段弧几乎重合**，于是「竖直投影 $\sin x$」「弧长 $x$」「斜线投影 $\tan x$」三者几乎相等，比值 $\frac{\sin x}{x}$ 被夹在 1 的附近。

三个几何量的渐近层次值得记住：

$$\sin x\ \sim\ x\ \sim\ \tan x\qquad(x\to0),$$

它们之间的**差**（即曲率带来的偏离）要到 $x^3$ 量级才显现：$\sin x=x-\frac{x^3}{6}+o(x^3)$。这个「一阶相等、三阶才分家」的事实，正是后面等价无穷小（§3.5）与泰勒展开（§6.4）共同依赖的深层结构。

**应用**：利用第一重要极限可以证明 $(\sin x)'=\cos x$。由定义

$$(\sin x)'=\lim_{h\to0}\frac{\sin(x+h)-\sin x}{h}=\lim_{h\to0}\frac{2\cos(x+\frac h2)\sin\frac h2}{h}=\cos x\cdot\lim_{h\to0}\frac{\sin\frac h2}{\frac h2}=\cos x.$$

其中用到和差化积与第一重要极限。**这一个公式就把三角函数的微分理论整个激活了**——$\cos,\tan,\cot,\sec,\csc$ 的导数全部由它派生。

## 3 第二重要极限：lim (1+1/x)^x = e

**第二重要极限**：

$$\lim_{x\to\infty}\left(1+\frac1x\right)^x=e,\qquad\text{或等价地}\qquad\lim_{x\to0}(1+x)^{1/x}=e.$$

证明的思路是**把连续极限归结为数列极限**，再用海涅定理。先看整数情形：第二章已证 $e_n=(1+\frac1n)^n\to e$。对任意实数 $x\to+\infty$，记 $n=\lfloor x\rfloor$，则 $n\le x<n+1$，于是

$$\frac1{n+1}\le\frac1x\le\frac1n\quad\Longrightarrow\quad\left(1+\frac1{n+1}\right)^n\le\left(1+\frac1x\right)^x\le\left(1+\frac1n\right)^{n+1}.$$

左边 $(1+\frac1{n+1})^n\to e$（因为它是 $e_{n+1}$ 去掉一个因子），右边 $(1+\frac1n)^{n+1}=e_n(1+\frac1n)\to e\cdot1=e$。两边夹住，$\lim_{x\to+\infty}(1+\frac1x)^x=e$。$x\to-\infty$ 的情形可令 $t=-x$ 化归。∎<span class="marginnote">这个证明演示了「<strong>整数→实数</strong>」的标准升级法：先在整数点建好 $e_n\to e$，再用不等式把任意实数 $x$ 夹在相邻两个整数之间，让 $e$ 从两边逼过来。下一章研究一致连续、定积分时，「把连续问题化归到有理/整数」是屡试不爽的招式。</span>

等价形式 $\lim_{x\to0}(1+x)^{1/x}=e$ 由换元 $t=1/x$ 得到，两个形式在不同场合各有用武之地：$x\to\infty$ 型处理「复利型」极限，$x\to0$ 型处理「指数型」极限。

## 4 两个重要极限的应用

**应用一：幂指函数极限。** $\displaystyle\lim_{x\to0}(1+\sin x)^{1/x}$。指数里有 $\frac1x$，设法凑出 $(1+u)^{1/u}$ 的形式。令 $u=\sin x$，则 $\frac1x=\frac1u\cdot\frac{u}{x}$，于是

$$(1+\sin x)^{1/x}=\left[(1+u)^{1/u}\right]^{u/x},\qquad u/x=\frac{\sin x}{x}\to1.$$

从而极限 $=e^1=e$。**标准套路：把幂指函数改写为 $\left[(1+\text{小量})^{1/\text{小量}}\right]^{\text{小量比}}$**，两个因子分别落在两个重要极限的射程内。

**应用二：指数增长模型。** 连续复利公式 $A(t)=A_0(1+\frac rn)^{nt}\to A_0e^{rt}$ 当 $n\to\infty$。金融里的连续复利、生物里的种群指数增长、物理里的放射性衰变，全以 $e$ 为增长基底——第二重要极限就是这些模型的数学源头。<span class="marginnote">大模型里的 softmax $\frac{e^{z_i}}{\sum e^{z_j}}$、注意力分数 $\frac{e^{q\cdot k}}{\sqrt d}$，用的都是同一个 $e$。当你在第四级《大模型原理》里见到各种 $\exp$ 时，请记住：它们的合法性归根结底来自「$(1+\frac1x)^x$ 真的收敛」这一条证明。</span>

**应用三：$(a^x)'$ 与 $(\ln x)'$。** 由定义

$$(a^x)'=\lim_{h\to0}a^x\frac{a^h-1}{h}.$$

令 $a^h-1=t$，则 $h=\log_a(1+t)$，$t\to0$：

$$\frac{a^h-1}{h}=\frac{t}{\log_a(1+t)}=\frac1{\frac1t\log_a(1+t)}=\frac1{\log_a(1+t)^{1/t}}\to\frac1{\log_a e}=\ln a.$$

于是 $(a^x)'=a^x\ln a$；特别地 $(e^x)'=e^x$。**$e^x$ 自导数的「神圣性质」正是由第二重要极限奠基的。**

## 5 小结

- **第一重要极限**：$\lim_{x\to0}\frac{\sin x}{x}=1$；由单位圆面积比较 $\sin x<x<\tan x$ 加夹逼证明，前提是弧度制。
- **渐近关系**：$\sin x\sim x\sim\tan x$（$x\to0$），差异在 $x^3$ 阶；支撑三角导数与等价无穷小。
- **第二重要极限**：$\lim_{x\to\infty}(1+\frac1x)^x=e$；用「整数→实数」夹逼证明，海涅定理介入。
- **应用套路**：幂指函数改写、连续复利、$(\sin x)'=\cos x$、$(a^x)'=a^x\ln a$——两把钥匙开所有门。
- **深层链接**：$e$ 的存在性来自单调有界定理（§2.3），$e$ 的普及性来自两个重要极限——前后两章在此汇合。

在下一节，我们进入无穷小量与无穷大量：**比较阶、等价替换**。上一节刚见面的 $\sin x\sim x$ 正是等价无穷小的第一个例子，而「阶」的概念将为极限计算提供系统化的化简规则。
