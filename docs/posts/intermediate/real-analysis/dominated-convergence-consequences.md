---
title: 控制收敛定理的推论：有界收敛定理与积分号下求极限、求导
date: 2026-08-07
---

# 控制收敛定理的推论：有界收敛定理与积分号下求极限、求导

<div class="epigraph">
<p>有了控制收敛，积分号下的极限、导数、积分便如行云流水——分析的运算可以放心地交换顺序。</p>
<footer>—— 埃米尔 · 波莱尔（Émile Borel）</footer>
</div>

<div class="article-byline">
<p>第二级 · 实变函数与测度论 ｜ 周民强《实变函数论》第五章 ｜ 2026-08-07</p>
</div>

## 为什么从控制收敛的推论开始

DCT 不只是一个孤立的定理，而是一台产生推论的生产线。本节收割最实用的三条：**有界收敛定理**（有限测度 + 一致有界）、**积分号下取极限**（含参数积分的极限）、**积分号下求导**（参数积分的可导性）。这三条是分析学日常计算的标配——遇到「$\lim$ 与 $\int$ 换序」「$\frac{d}{dt}\int$」这类问题，答案就是它们。

参数积分 $F(t)=\int f(x,t)dx$ 是它们的共同舞台：物理中的位势、概率中的特征函数、调和分析中的卷积，全都是含参数积分。**能在积分号下求导、求极限，是把「算积分」变成「算微积分」的关键能力。**<span class="marginnote">「积分号下求导」对应概率论中<strong>特征函数（Fourier 变换）的可微性</strong>：若 $E[|X|]<\infty$，则 $\varphi_X(t)=E[e^{itX}]$ 可微且 $\varphi_X'(t)=E[iXe^{itX}]$。<strong>控制函数 $|X|$ 是「可微性」的判据</strong>——矩的存在性直接控制特征函数的正则性。</span>

## 1 有界收敛定理

**定理（有界收敛）**：设 $m(E)<\infty$，$\{f_k\}$ 在 $E$ 上一致有界（$|f_k|\le M$），$f_k\to f$ a.e.。则

$$\lim_{k\to\infty}\int_Ef_k\,dm=\int_Ef\,dm$$

**证明**：控制函数 $g=M\chi_E$，$\int g=M\,m(E)<\infty$（有限测度），直接套 DCT。

**重点：有界收敛是 DCT 在「有限测度 + 一致有界」下的特化**——它去掉了「显式写控制函数」的负担，是概率论、Fourier 分析中最常用的形态。例：$\int_0^1\sum_k$ 的各类交换，只要级数一致有界即可。

## 2 积分号下取极限：参数积分

**定理（参数积分的极限）**：设 $\{f_k(x,t)\}$ 是 $x$ 的可测函数族，对每个 $k$，$f_k(\cdot,t)\to f(\cdot,t_0)$ a.e.（$t\to t_0$ 或 $k\to\infty$ 的意义下），且存在 $g\in L^1$ 使 $|f_k(x,t)|\le g(x)$。则

$$\lim_{t\to t_0}\int f(x,t)\,dx=\int f(x,t_0)\,dx$$

**应用（连续性）**：$F(t)=\int f(x,t)dx$ 在 $t_0$ 连续，若 $f(\cdot,t)\to f(\cdot,t_0)$ a.e. 且有统一控制。**参数积分的连续性 = DCT 的一个直接推论**——不必逐点验证，只要 a.e. 收敛 + 可积控制。

**例子**：$F(t)=\int_0^\infty e^{-xt}dx$（$t>0$）在 $t_0>0$ 连续：$e^{-xt}\to e^{-xt_0}$，控制函数在 $t$ 靠近 $t_0$ 时取 $e^{-xt_0/2}$（可积）。

## 3 积分号下求导

**定理（Leibniz 规则 / 积分号下求导）**：设 $f(x,t)$ 关于 $x$ 可测、关于 $t$ 可微，且存在 $g\in L^1$ 使

$$\left|\frac{\partial f}{\partial t}(x,t)\right|\le g(x)\ \ \text{a.e.},\qquad \forall t$$

则 $F(t)=\int f(x,t)dx$ 可微，且

$$F'(t)=\int\frac{\partial f}{\partial t}(x,t)\,dx$$

**证明**：用差商。$\frac{F(t+h)-F(t)}{h}=\int\frac{f(x,t+h)-f(x,t)}{h}dx$。对固定的 $t$，差商 $\to\partial_t f$ 逐点（$f$ 可微）；由中值定理，$\left|\frac{f(x,t+h)-f(x,t)}{h}\right|\le\sup_s|\partial_t f(x,s)|\le g(x)$——**差商被 $g$ 统一控制**。DCT 给出差商极限 = 积分极限。<span class="marginnote">证明中「中值定理给出差商被 $\sup|\partial_t f|$ 控制」是关键一步：<strong>把「导数有界」翻译成「差商有界」</strong>，从而 DCT 可以作用在差商列上。<strong>「控制导数 = 控制差商」</strong>是 Leibniz 规则的全部秘密。</span>

**例子**：$\Gamma$ 函数 $F(t)=\int_0^\infty x^{t-1}e^{-x}dx$ 可导：$\partial_t x^{t-1}e^{-x}=x^{t-1}\ln x\,e^{-x}$，在 $t\in[a,b]$ 上被 $x^{b-1}|\ln x|e^{-x}$ 控制（可积，因指数衰减）。

## 4 公式解析：差商的控制

Leibniz 规则的证明核心是差商的控制：

$$\left|\frac{f(x,t+h)-f(x,t)}{h}\right|\ \overset{\text{MVT}}{\le}\ \sup_{s}|\partial_t f(x,s)|\ \le\ g(x)$$

- **第一步，读「中值定理（MVT）」**：对固定的 $x$，$t\mapsto f(x,t)$ 是单变量函数，由中值定理，差商等于某点 $s$（在 $t$ 与 $t+h$ 之间）的导数值。**「差商 = 某处导数」把差商问题归约为导数问题**。
- **第二步，读「$\sup_s$ 被 $g$ 控制」**：$\sup_s|\partial_t f(x,s)|\le g(x)$（定理假设）。**「导数一致有界（被可积函数控制）」⇒「差商一致有界（被同一 $g$ 控制）」**——$g$ 同时罩住所有 $s$ 处的导数，也就罩住所有差商。
- **第三步，读「DCT 收官」**：$h\to0$ 时差商 $\to\partial_tf$ a.e.（可微性），且被 $g\in L^1$ 控制——DCT 给出 $\lim_h\int\text{差商}=\int\partial_tf$。**「逐点求导」升级为「积分号下求导」**。

**「MVT 翻译 + $g$ 统一控制 + DCT」**，是积分号下求导的完整证明链——也是参数积分理论最常用的三件套。

## 5 一个完整的计算实例

把「积分号下求导」用在一个真实计算上，看清整套机器的运转。

**例（Fourier 变换的可微性）**：设 $f\in L^1(\mathbb{R})$ 且 $xf(x)\in L^1(\mathbb{R})$（一阶矩存在）。考虑 Fourier 变换

$$\widehat f(t)=\int_{-\infty}^{\infty}f(x)e^{-ixt}\,dx$$

**第一步，证 $\widehat f$ 连续**：对 $t\to t_0$，$f(x)e^{-ixt}\to f(x)e^{-ixt_0}$ 逐点（$e^{-ixt}$ 连续），且被 $|f|\in L^1$ 控制——DCT 给出 $\widehat f(t)\to\widehat f(t_0)$。**连续性免费获得**。

**第二步，证 $\widehat f$ 可导**：差商 $\tfrac{\widehat f(t+h)-\widehat f(t)}h=\int f(x)e^{-ixt}\tfrac{e^{-ixh}-1}{h}dx$。被积函数逐点 $\to -ixf(x)e^{-ixt}$（$h\to0$），且差商被 $\sup_s|(-ix)e^{-ixs}|=x$ 控制（中值定理），而 $|xf|\in L^1$（假设）——DCT 给出

$$\widehat f'(t)=\int f(x)(-ix)e^{-ixt}\,dx$$

**第三步，读结论**：$\widehat f$ 连续可微，且导数仍是「$(-ix)f$ 的 Fourier 变换」。**「$f$ 的一阶矩存在」直接翻译成「$\widehat f$ 可微」**——控制函数 $|x|$ 就是这条翻译的媒介。

**重点：这套「可积控制 ⇒ 换序合法」的流程，是特征函数、卷积、热核、概率论中一切参数积分的标准动作。** 每次换序（极限、导数、积分）都问同一句话：**被积函数的差商/变化是否被某个与参数无关的可积函数控制？** 是，则 DCT 放行。<span class="marginnote">「控制函数的选取」是这套流程的技艺核心：本例取 $|x|$，因为 $e^{-ixt}$ 的导数是 $-ixe^{-ixt}$，模为 $|x|$。<strong>「导数的大小」直接决定「控制函数」，进而决定「可导性的矩条件」</strong>——$n$ 阶导数对应 $|x|^n$ 的控制，对应 $n$ 阶矩存在。矩 ↔ 光滑性的精确对应由此而来。</span>

## 6 小结

- **有界收敛**：有限测度 + 一致有界 + a.e. 收敛 ⇒ 积分收敛（DCT 特化）。
- **积分号下取极限**：$|f(x,t)|\le g\in L^1$ + a.e. 收敛 ⇒ $F(t)$ 连续。
- **积分号下求导**：$|\partial_tf|\le g\in L^1$ ⇒ $F'(t)=\int\partial_tf$（MVT + DCT）。
- **统一结构**：可积控制是「换序」的万能通行证。
- **应用**：$\Gamma$ 函数、特征函数、卷积的正则性全赖于此。

在下一节，我们总结 **三大定理的相互关系与典型应用**——一张图看清 Levi、Fatou、DCT 的谱系。
