---
title: 多维伊藤公式与乘积法则
date: 2026-08-07
---

# 多维伊藤公式与乘积法则

<div class="epigraph">
<p>两个随机过程相乘，导数里多出一项「协波动」——它正是两者共同的布朗脉冲。</p>
<footer>—— 保罗 · 萨缪尔森（Paul Samuelson）</footer>
</div>

<div class="article-byline">
<p>第二级 · 随机过程 ｜ 张波《应用随机过程》§8.3 ｜ 2026-08-07</p>
</div>

## 当世界不止一个随机源

真实系统往往由**多个相互作用的随机过程**驱动：股价受自身波动与其利率波动影响，两个资产的组合需要同时跟踪。单变量 Itô 公式不够用，需要**多维版本**。

多维 Itô 公式最迷人的产物是**乘积法则（product rule）**：
$$
d(XY) = X\, dY + Y\, dX + dX\, dY.
$$
**普通微积分的乘积法则只到前两项；随机世界多出交叉项 $dX\,dY$**——它编码了「两个过程的共同波动」。这个交叉项正是对冲、多资产定价、相关性的数学来源。<span class="marginnote">「$dX\,dY$」不是可忽略的高阶项：<strong>当 $X$、$Y$ 都由布朗驱动时，$dX\,dY$ 含 $dt$ 量级的项（因为 $(dB)^2 = dt$）</strong>。它与 $dX^2$、$dY^2$ 一样是「真家伙」——乘积法则多出的那项，是 Itô 公式在二维的必然。</span>

本节目标：陈述多维 Itô 公式、推导乘积法则、并用它处理「相关布朗运动」与乘积型金融模型。

## 1 多维 Itô 公式

设 $X(t) = (X_1(t), \dots, X_d(t))$ 是 $d$ 维 Itô 过程，每个分量 $dX_i = \mu_i dt + \sigma_i dB_i$（$B_i$ 可以相关）。对 $g(t, x_1, \dots, x_d) \in C^{1,2}$，
$$
dg(t, X) = \frac{\partial g}{\partial t}\, dt + \sum_i \frac{\partial g}{\partial x_i}\, dX_i + \frac12 \sum_{i,j} \frac{\partial^2 g}{\partial x_i \partial x_j}\, dX_i\, dX_j.
$$
**二阶项不再是单个 $(dX_i)^2$，而是所有交叉项 $dX_i dX_j$ 之和**——协方差结构进入公式。

**$(dX_i)(dX_j)$ 的计算**：$dX_i dX_j = \sigma_i \sigma_j\, dB_i\, dB_j$（$dt$ 项与 $dt\,dB$ 项归零）。若 $B_i, B_j$ 的相关系数为 $\rho_{ij}$（即 $E[B_i(t)B_j(t)] = \rho_{ij} t$），则
$$
dB_i\, dB_j = \rho_{ij}\, dt.
$$
**交叉二次变差由相关系数决定。**<span class="marginnote">「$dB_i dB_j = \rho_{ij} dt$」是多维随机分析的心脏：<strong>相关的布朗脉冲，其乘积的 $dt$ 系数就是相关系数</strong>。独立时 $\rho = 0$ 交叉项消失；完全相关 $\rho = 1$ 时交叉项 = 二次变差。资产相关性进入 Itô 公式的唯一通道，就是这条规则。</span>

**数值例：独立 vs 相关。** 设 $dB_1 dB_2 = \rho\, dt$。$\rho = 0$（独立）时 $dX dY$ 的 $dt$ 系数为 0——乘积法则退化为普通微积分形式；$\rho = 1$（同一布朗源）时 $dX dY = \sigma_1\sigma_2 dt$——交叉项最大。**「相关性越强，交叉项越大」这条直觉，在随机微积分里有精确的公式。**

**辨析｜易错点：** 算 $dX_i dX_j$ 时，只有「布朗 × 布朗」保留 $dt$ 量级；$dt \times dB$ 与 $dt \times dt$ 都归零——初学者常忘记这一点，把 $dX_i dX_j$ 整个丢弃。还要注意 $dB_i^2 = dt$ 是「$\rho = 1$」的特例，与一般相关情形的 $dB_i dB_j = \rho_{ij} dt$ 同源，不必另记。另一个高频错误是把乘积法则写成普通微积分形式 $d(XY) = X\,dY + Y\,dX$——**漏掉交叉项会让多资产组合的方差直接算错**。

## 2 乘积法则

取 $d = 2$，$g(x_1, x_2) = x_1 x_2$。则 $\partial g/\partial x_1 = x_2$，$\partial g/\partial x_2 = x_1$，$\partial^2 g/\partial x_1\partial x_2 = 1$，其余二阶为 0。代入多维公式：
$$
d(X_1 X_2) = X_2\, dX_1 + X_1\, dX_2 + dX_1\, dX_2.
$$
**这就是乘积法则**。记 $X_1 = X$、$X_2 = Y$：
$$
d(XY) = X\, dY + Y\, dX + dX\, dY.
$$

**与普通微积分的差别**：普通 $d(XY) = X dY + Y dX$。随机世界多出的 $dX\,dY$，当 $X$、$Y$ 都随机时通常**不为 0**——它是「共同波动」的贡献。<span class="marginnote">乘积法则的直觉：<strong>乘积 $XY$ 的增量包含三部分：$X$ 不动时 $Y$ 的变动、$Y$ 不动时 $X$ 的变动、以及「两者同时动」的耦合项</strong>。普通微积分忽略耦合（二阶小），随机世界耦合 $dX\,dY \sim dt$ 不能丢。</span>

## 3 公式解析：证明乘积法则

**目标：从二维 Itô 公式推出 $d(XY) = X dY + Y dX + dX dY$，并验证交叉项 $dX dY$ 的 $dt$ 系数。**

第一步，验证二阶偏导。$g(x,y) = xy$：$g_{xx} = 0$，$g_{yy} = 0$，$g_{xy} = 1$。代入多维公式，二阶项只剩 $\frac12 \cdot 2 \cdot dX dY = dX dY$。

第二步，写 $dX$、$dY$ 的展开。$dX = \mu_1 dt + \sigma_1 dB_1$，$dY = \mu_2 dt + \sigma_2 dB_2$。

第三步，算交叉项。$dX dY = \sigma_1\sigma_2 dB_1 dB_2 = \sigma_1 \sigma_2 \rho_{12} dt$（其余项归零）。

第四步，代回。若 $X$、$Y$ 分别由 $B_1$、$B_2$ 驱动且相关 $\rho$：
$$
d(XY) = X\, dY + Y\, dX + \sigma_1\sigma_2\rho\, dt.
$$
**交叉项精确等于「两个波动率之积 × 相关系数 × dt」——它是相关性的随机微积分化身。**

**这个推导为什么重要**：它把「相关性」从静态概念变成动态项——**在多资产定价里，资产间的相关性通过 $dX dY$ 进入组合的微分方程**。这正是投资组合对冲、期权希腊字母计算的数学基础。

## 4 应用：两个 GBM 的乘积

设 $S_1$、$S_2$ 是两个几何布朗运动：$dS_i = \mu_i S_i dt + \sigma_i S_i dB_i$，$dB_1 dB_2 = \rho dt$。求 $S_1 S_2$ 的 SDE。

由乘积法则：
$$
d(S_1 S_2) = S_1 dS_2 + S_2 dS_1 + dS_1 dS_2.
$$
代入 $dS_i$：
$$
dS_1 dS_2 = \sigma_1\sigma_2 S_1 S_2 \rho dt.
$$
于是
$$
d(S_1 S_2) = S_1S_2 \big[ (\mu_1 + \mu_2 + \sigma_1\sigma_2\rho)\, dt + \sigma_1 dB_1 + \sigma_2 dB_2 \big].
$$
**两个 GBM 的乘积仍是 GBM（对数正态），漂移加上交叉项 $\sigma_1\sigma_2\rho$**——这是「多资产指数」建模（如配对交易、指数期权）的标准结果。<span class="marginnote">交叉项 $\sigma_1\sigma_2\rho$ 的现实含义：<strong>即使两资产各自独立地上涨，只要正相关（$\rho > 0$），它们的乘积会「超涨」</strong>——因为共同的正向波动叠加。这是组合风险里「相关性放大波动」的 Itô 版本。</span>

## 5 应用：指数鞅与随机指数

**随机指数（stochastic exponential）**：$\mathcal{E}(X)_t = e^{X(t) - \frac12 [X]_t}$，其中 $[X]_t$ 是 $X$ 的二次变差。用多维/单维 Itô 公式可验证它满足
$$
d\mathcal{E}(X) = \mathcal{E}(X)\, dX.
$$
当 $X(t) = \theta B(t)$ 时，$\mathcal{E}(X)_t = e^{\theta B(t) - \theta^2 t/2}$——正是第七节的指数鞅。**随机指数是「鞅化」的通用操作：给任意鞅 $X$ 加上修正项 $-\frac12[X]$，乘积法则自动让指数成为鞅。**<span class="marginnote">随机指数在金融里就是「测度变换」的工具（Girsanov 定理的核心，第十篇风险中性定价会用到）：<strong>用随机指数做似然比，把真实测度换到风险中性测度，价格便成鞅</strong>。乘积法则 + 指数鞅，是随机分析里最优雅的组合拳。</span>

## 6 小结

- **多维 Itô 公式**：二阶项含全部交叉项 $\frac12\sum g_{ij} dX_i dX_j$。
- **交叉二次变差**：$dB_i dB_j = \rho_{ij} dt$——相关性进入微积分的唯一通道。
- **乘积法则**：$d(XY) = X dY + Y dX + dX dY$——交叉项 $dX dY$ 不可忽略。
- 两个 GBM 乘积：仍是 GBM，漂移加 $\sigma_1\sigma_2\rho$。
- **随机指数** $\mathcal{E}(X) = e^{X - \frac12[X]}$：鞅化通用操作，测度变换的核心；当 $X$ 由相关布朗驱动时，其二次变差同样含交叉项 $dX_1dX_2 = \rho\,dt$——相关性穿透到鞅化指数，正是多资产 Girsanov 变换的结构。

**例（组合方差的 Itô 视角）**：资产组合 $W = w_1 S_1 + w_2 S_2$。Itô 公式（$g$ 线性，二阶项为零）给 $dW = w_1 dS_1 + w_2 dS_2$——组合的微分只是权重线性组合；但组合的二次变差
$$
dW^2 = w_1^2\sigma_1^2 S_1^2 dt + w_2^2\sigma_2^2 S_2^2 dt + 2w_1w_2\rho\sigma_1\sigma_2 S_1S_2 dt
$$
含交叉项——**相关性进入组合波动的方式，正是乘积法则的 $dS_1 dS_2$**。风险管理里「相关性放大波动」的量化，从这条二次变差读出。

**一处提醒**：多维公式里「$\frac12$」与「全部交叉项」缺一不可——漏掉任一交叉项，组合方差的 $dt$ 系数就会出错，对冲比率也会随之偏离。**写多维 Itô 公式时，先写下全部 $\sum_{i,j}$ 交叉项，再逐一化简，是最稳的顺序。**

在下一节，我们把 Itô 公式反过来用：**随机微分方程（SDE）的解与存在唯一性**——如何保证「随机方程有解」。
