---
title: 伊藤引理
date: 2026-08-07
---

# 伊藤引理

<div class="epigraph">
<p>随机世界的链式法则：泰勒展开里多出来的那一项，改写了金融数学。</p>
<footer>—— 伊藤清（Itō Kiyoshi，1944）</footer>
</div>

<div class="article-byline">
<p>第四级 · 金融数学与精算 ｜ 《金融数学》第5章 / Hull 第14章 ｜ 2026-08-07</p>
</div>

## 为什么从伊藤引理开始

《随机微积分入门》留下了一张乘法表：$\mathrm{d}t^2 = 0$、$\mathrm{d}t\,\mathrm{d}W = 0$、$\mathrm{d}W^2 = \mathrm{d}t$。现在要把普通微积分的**链式法则**搬到随机世界。普通链式法则对光滑函数成立，但对布朗运动这种「二次变差非零」的过程，泰勒展开里「二阶项」不再可忽略——而那一项，正是伊藤引理的核心。**伊藤引理是随机版链式法则，它让我们能对随机过程取函数、解 SDE、并最终推导出 Black-Scholes 方程。**<span class="marginnote">在「从极限到大模型」的坐标里，伊藤引理类似反向传播里的链式法则：前者告诉你随机函数怎么微，后者告诉你复合函数怎么微。掌握了「如何对随机过程求导」，你就能算出任何衍生品的价格变化。</span>

## 1 普通泰勒展开回顾

设 $F(t, x)$ 是光滑函数，$x$ 是确定性变量，对 $F(t + \Delta t, x + \Delta x)$ 做泰勒展开到一阶：

$$
\mathrm{d}F = \frac{\partial F}{\partial t}\,\mathrm{d}t + \frac{\partial F}{\partial x}\,\mathrm{d}x
$$

二阶项 $\frac{1}{2}\frac{\partial^2 F}{\partial x^2}(\mathrm{d}x)^2$ 之所以被丢弃，是因为对光滑路径 $(\mathrm{d}x)^2 \sim (\mathrm{d}t)^2$ 是高阶无穷小。**但在随机世界里，$(\mathrm{d}W)^2 = \mathrm{d}t$ 不是高阶无穷小**——它就是 $\mathrm{d}t$ 量级，必须保留。这一条规则导致整个公式多出一项。

## 2 伊藤引理的陈述

设 $X_t$ 满足 SDE $\mathrm{d}X_t = \mu_t\,\mathrm{d}t + \sigma_t\,\mathrm{d}W_t$，$F(t, x)$ 二次连续可微，则

$$
\mathrm{d}F(t, X_t) = \frac{\partial F}{\partial t}\,\mathrm{d}t + \frac{\partial F}{\partial x}\,\mathrm{d}X_t + \frac{1}{2}\frac{\partial^2 F}{\partial x^2}(\mathrm{d}X_t)^2
$$

把 $\mathrm{d}X_t = \mu\,\mathrm{d}t + \sigma\,\mathrm{d}W$ 代入并运用乘法表展开：

$$
\mathrm{d}F = \left(\frac{\partial F}{\partial t} + \mu \frac{\partial F}{\partial x} + \frac{1}{2}\sigma^2 \frac{\partial^2 F}{\partial x^2}\right)\mathrm{d}t + \sigma \frac{\partial F}{\partial x}\,\mathrm{d}W
$$

**多出来的 $\frac{1}{2}\sigma^2 \frac{\partial^2 F}{\partial x^2}\,\mathrm{d}t$ 项，就是伊藤修正（Itô correction）。**<span class="marginnote">它的来源完全来自乘法表 $(\mathrm{d}W)^2 = \mathrm{d}t$：泰勒展开中的 $\frac{1}{2}F_{xx}(\sigma\,\mathrm{d}W)^2$ 变成了 $\frac{1}{2}\sigma^2 F_{xx}\,\mathrm{d}t$。如果你省略它，几乎所有随机计算都会错——这是初学者最大的坑。</span>

## 3 应用一：求解几何布朗运动

回顾 GBM 的 SDE：

$$
\frac{\mathrm{d}S_t}{S_t} = \mu\,\mathrm{d}t + \sigma\,\mathrm{d}W_t
$$

取 $F(S) = \ln S$，则 $\frac{\partial F}{\partial S} = \frac{1}{S}$、$\frac{\partial^2 F}{\partial S^2} = -\frac{1}{S^2}$，代入伊藤引理：

$$
\mathrm{d}(\ln S_t) = \left(0 + \mu S \cdot \frac{1}{S} + \frac{1}{2}\sigma^2 S^2 \cdot \left(-\frac{1}{S^2}\right)\right)\mathrm{d}t + \sigma S \cdot \frac{1}{S}\,\mathrm{d}W
$$

$$
\mathrm{d}(\ln S_t) = \left(\mu - \frac{\sigma^2}{2}\right)\mathrm{d}t + \sigma\,\mathrm{d}W_t
$$

**对数价格的漂移是 $\mu - \frac{\sigma^2}{2}$，比股价的漂移少一个 $\frac{\sigma^2}{2}$**——这就是《随机游走与布朗运动》里预告的伊藤修正项。积分即得

$$
S_t = S_0 \exp\left(\left(\mu - \frac{\sigma^2}{2}\right)t + \sigma W_t\right)
$$

<span class="marginnote">检验逻辑闭环：$\mathrm{d}(\ln S)$ 的漂移是 $\mu - \sigma^2/2$，但它期望为 $\ln S_0 + (\mu - \sigma^2/2)t$；而 $S_t$ 的期望 $\mathbb{E}[S_t] = S_0 e^{\mu t}$——两者一致，因为 $\ln$ 的期望 ≠ $\ln$ 的期望值（詹森不等式的典型场景）。</span>

## 4 应用二：给衍生品定价的前奏

伊藤引理不只是解方程的工具，它还是「对冲论证」的引擎。设 $f(t, S_t)$ 是某个以 $S_t$ 为标的的衍生品价格，则它自身也是随机过程，满足

$$
\mathrm{d}f = \left(f_t + \mu S f_S + \frac{1}{2}\sigma^2 S^2 f_{SS}\right)\mathrm{d}t + \sigma S f_S\,\mathrm{d}W
$$

这里下标表示偏导。**观察：$\mathrm{d}f$ 的随机项与 $\mathrm{d}S$ 的随机项都正比于 $\mathrm{d}W$，比例分别是 $f_S$ 与 1。** 于是构造「1 份衍生品 − $f_S$ 份股票」的组合，$\mathrm{d}W$ 项就会抵消——这就是 **Delta 对冲（delta hedging）** 的数学本质，也是下一篇《期权基础与二叉树模型》和《Black-Scholes模型》推导的核心手法。<span class="marginnote">这条「消去随机项」的操作是金融数学最优雅的一步：随机性被对冲掉后，组合变成一个确定性的、只随时间演化的对象，定价就回到了《利率与贴现》的确定世界。</span>

## 5 公式解析：伊藤引理的每一项

把完整公式拆成三项，逐个解释物理/金融含义：

$$
\mathrm{d}F = \underbrace{\frac{\partial F}{\partial t}\,\mathrm{d}t}_{\text{时间的显式依赖}} + \underbrace{\mu \frac{\partial F}{\partial x}\,\mathrm{d}t}_{\text{漂移推动}} + \underbrace{\frac{1}{2}\sigma^2 \frac{\partial^2 F}{\partial x^2}\,\mathrm{d}t}_{\text{伊藤修正}} + \underbrace{\sigma \frac{\partial F}{\partial x}\,\mathrm{d}W}_{\text{随机推动}}
$$

- **$\frac{\partial F}{\partial t}\,\mathrm{d}t$**：$F$ 直接随时间变化的贡献（如期权的到期日逼近）。
- **$\mu \frac{\partial F}{\partial x}\,\mathrm{d}t$**：底层变量漂移的传递，普通链式法则就有的项。
- **$\frac{1}{2}\sigma^2 \frac{\partial^2 F}{\partial x^2}\,\mathrm{d}t$**：**随机过程凸性带来的额外漂移**。因为随机扰动让 $X$ 上下抖动，凸函数 $F$（$F_{xx} > 0$）在抖动中「平均上浮」，这部分溢价就是伊藤修正。
- **$\sigma \frac{\partial F}{\partial x}\,\mathrm{d}W$**：随机冲击经 $F$ 对 $x$ 的敏感度放大，是唯一随机项。

**为什么直觉上必须有伊藤修正？** 因为 $W$ 的抖动不是无限小，而是 $\sqrt{\mathrm{d}t}$ 量级；凸函数对「上下等幅」的响应不对称——涨时赚的比跌时亏的多（对凸函数），净效应就是 $\frac{1}{2}F''\sigma^2\,\mathrm{d}t$。<span class="marginnote">这个「凸性溢价」的思想是金融数学的核心直觉之一，也直接解释了为什么期权（凸函数）有正的时间价值——持有期权等于免费享受波动率带来的凸性收益，详见《期权希腊字母与对冲》里的 Gamma。</span>

## 6 多维伊藤引理

标的资产不只一个时，伊藤引理有向量形式。设 $n$ 个资产满足

$$
\mathrm{d}S^i_t = \mu_i S^i_t\,\mathrm{d}t + \sigma_i S^i_t\,\mathrm{d}W^i_t
$$

其中 $W^i$ 之间有相关系数 $\rho_{ij}$（$\mathrm{d}W^i\,\mathrm{d}W^j = \rho_{ij}\,\mathrm{d}t$），对 $F(t, S^1, \ldots, S^n)$ 有

$$
\mathrm{d}F = \frac{\partial F}{\partial t}\,\mathrm{d}t + \sum_i \frac{\partial F}{\partial S^i}\,\mathrm{d}S^i + \frac{1}{2}\sum_{i,j} \frac{\partial^2 F}{\partial S^i \partial S^j}\,\mathrm{d}S^i\,\mathrm{d}S^j
$$

其中 $\mathrm{d}S^i\,\mathrm{d}S^j = \sigma_i\sigma_j S^i S^j \rho_{ij}\,\mathrm{d}t$。<span class="marginnote">多维版本用于外汇期权、利率模型与组合对冲——交叉项的协方差结构决定了「资产之间联动如何影响衍生品价格」。它也是《投资组合优化》里协方差矩阵在连续时间的对应。</span>注意求和扩展到所有 $(i,j)$ 对，因为 $i \ne j$ 时二次交叉项也贡献 $\mathrm{d}t$ 量级。

## 7 小结

- **伊藤引理**是随机版链式法则：$\mathrm{d}F = F_t\,\mathrm{d}t + F_x\,\mathrm{d}X + \frac{1}{2}F_{xx}(\mathrm{d}X)^2$。
- 关键来源是乘法表 $(\mathrm{d}W)^2 = \mathrm{d}t$，**多出的 $\frac{1}{2}\sigma^2 F_{xx}\,\mathrm{d}t$ 叫伊藤修正**，不可省略。
- 用它解 GBM：$\mathrm{d}(\ln S) = (\mu - \sigma^2/2)\mathrm{d}t + \sigma\,\mathrm{d}W$，价格服从对数正态。
- **对冲论证**：构造「衍生品 − $f_S$ 份股票」可消去随机项，引出 Delta 对冲与 Black-Scholes。
- 伊藤修正的直觉是**凸性溢价**：凸函数在随机抖动中平均上浮。
- 多维版本引入协方差交叉项 $\sigma_i\sigma_j\rho_{ij}$，用于多资产模型。

在下一节，我们进入期权定价的第一站：**期权基础与二叉树模型**——先看期权的收益结构与无套利定价，再用一棵「会涨会跌」的树把伊藤引理的直觉离散化。
