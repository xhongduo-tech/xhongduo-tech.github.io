---
title: Wishart 分布与 Hotelling T² 检验
date: 2026-08-07
---

# Wishart 分布与 Hotelling T² 检验

<div class="epigraph">
<p>推断的艺术在于用一个样本同时回答两个问题：估计本身有多可靠，以及这个估计和假设差多远。</p>
<footer>—— 哈罗德·霍特林（Harold Hotelling）</footer>
</div>

<div class="article-byline">
<p>第二级 · 多元统计分析 ｜ Anderson《An Introduction to Multivariate Statistical Analysis》Ch.4–5 · Johnson & Wichern Ch.5 ｜ 2026-08-07</p>
</div>

## 为什么从矩阵分布开始

一元统计推断的核心是三件套：样本均值 $\bar{x}$ 的分布、样本方差 $s^2$ 的分布、以及把二者揉在一起的 $t$ 统计量。多元推断完全平行，只是「均值」变成向量 $\bar{\mathbf{x}}$，「方差」变成矩阵 $\mathbf{S}$，于是三个问题全部升维：$\bar{\mathbf{X}}$ 服从多元正态（上一节已解决），$\mathbf{S}$ 服从 **Wishart 分布**，而检验均值向量的统计量就是 **Hotelling $T^2$**。<span class="marginnote">Wishart 分布由统计学家 John Wishart 在 1928 年首次给出，它正是「协方差矩阵的 $\chi^2$ 分布」——一元里 $s^2$ 的 $(n-1)s^2/\sigma^2 \sim \chi^2_{n-1}$，多元里整矩阵 $(n-1)\mathbf{S}$ 服从 Wishart 分布。</span>

## 1 Wishart 分布：样本协方差矩阵的分布

设 $\mathbf{X}_1, \ldots, \mathbf{X}_n$ 独立同分布，$\mathbf{X}_i \sim \mathcal{N}_p(\mathbf{0}, \boldsymbol{\Sigma})$，则

$$
\mathbf{M} = \sum_{i=1}^{n} \mathbf{X}_i \mathbf{X}_i' \sim \mathcal{W}_p(\boldsymbol{\Sigma}, n)
$$

称 $\mathbf{M}$ 服从自由度为 $n$ 的 **Wishart 分布**。当 $p = 1$ 时它退化为 $\sigma^2 \chi^2_n$——Wishart 分布确实是卡方分布的矩阵推广。<span class="marginnote">若 $\mathbf{X}_i \sim \mathcal{N}_p(\boldsymbol{\mu}, \boldsymbol{\Sigma})$，则中心化后的 $\sum_i(\mathbf{X}_i-\bar{\mathbf{X}})(\mathbf{X}_i-\bar{\mathbf{X}})' \sim \mathcal{W}_p(\boldsymbol{\Sigma}, n-1)$，自由度从 $n$ 降到 $n-1$——一元里「用 $\bar{x}$ 代替 $\mu$ 损失一个自由度」的规则在矩阵情形完全一致。</span>

Wishart 分布三条最重要的性质，全部服务于后续推断：

**可加性**：$\mathbf{M}_1 \sim \mathcal{W}_p(\boldsymbol{\Sigma}, n_1)$、$\mathbf{M}_2 \sim \mathcal{W}_p(\boldsymbol{\Sigma}, n_2)$ 独立，则 $\mathbf{M}_1 + \mathbf{M}_2 \sim \mathcal{W}_p(\boldsymbol{\Sigma}, n_1+n_2)$——自由度像样本量一样相加。

**期望**：$E(\mathbf{M}) = n\boldsymbol{\Sigma}$。这正是 $(n-1)\mathbf{S} \sim \mathcal{W}_p(\boldsymbol{\Sigma}, n-1)$ 蕴含 $E(\mathbf{S}) = \boldsymbol{\Sigma}$，呼应了第一节的贝塞尔校正。

**正定性**：$n \geq p$ 时 $\mathbf{M}$ 几乎处处正定。这个条件很重要：协方差矩阵必须可逆，而可逆要求样本量不比变量数少——高维数据（$p > n$）里 Wishart 分布与 $\mathbf{S}^{-1}$ 一起失效，这正是最后那篇高维专题的伏笔。

## 2 Hotelling T²：把 t 统计量推广到多元

一元里检验 $H_0: \mu = \mu_0$ 用 $t = \frac{\sqrt{n}(\bar{x}-\mu_0)}{s}$。把分子的标量换成向量，分母的标量换成矩阵，取平方就得到 **Hotelling $T^2$ 统计量**：

$$
T^2 = n(\bar{\mathbf{x}} - \boldsymbol{\mu}_0)'\mathbf{S}^{-1}(\bar{\mathbf{x}} - \boldsymbol{\mu}_0)
$$

结构值得品：$n$ 是样本量，$\bar{\mathbf{x}}-\boldsymbol{\mu}_0$ 是「偏离假设的向量」，$\mathbf{S}^{-1}$ 把偏离量按数据散布归一化。**$T^2$ 就是多元马氏距离的平方**——样本均值离假设中心的标准化距离。<span class="marginnote">对照一元：$t^2 = n(\bar{x}-\mu_0)^2/s^2$。把 $t^2$ 里的除法换成矩阵求逆，$T^2$ 就出来了。所以 $T^2$ 不是 $t$ 的平方，而是「$t$ 平方的多元版本」。</span>

大样本下 $\bar{\mathbf{X}}$ 近似 $\mathcal{N}_p(\boldsymbol{\mu}_0, \mathbf{S}/n)$，$T^2$ 近似 $\chi^2_p$。但小样本需要精确分布——这正是下一节的「归一化配方」。

### 辨析｜易错点：$T^2$ 与大样本 $\chi^2$ 的关系

初学者常把 $T^2$ 直接比作 $\chi^2_p$。对错各半：**样本量 $n$ 很大时**，$\mathbf{S}$ 收敛到 $\boldsymbol{\Sigma}$，$T^2 = n(\bar{\mathbf{x}}-\boldsymbol{\mu}_0)'\mathbf{S}^{-1}(\bar{\mathbf{x}}-\boldsymbol{\mu}_0)$ 确实渐近 $\chi^2_p$——这是上一节二次型性质的直接推论；**但 $n$ 有限时用 $\chi^2_p$ 会系统性低估拒绝阈值，犯第一类错误**。小样本必须用精确的 $F$ 换算。判别口诀：样本大用 $\chi^2$，样本小用 $F$；拿不准就用 $F$，它是精确分布。<span class="marginnote">还有一个等价视角：$\frac{n-p}{(n-1)p}T^2 \sim F_{p,n-p}$ 在 $p=1$ 时给出 $\frac{n-1}{n-1}T^2 = t^2 \sim F_{1,n-1}$，而 $F_{1,\nu}$ 正是 $t_\nu$ 的平方——一致性检验通过，说明公式没写错。</span>

### 一元与多元推断的三件套对照

| 对象 | 一元 | 多元 |
| --- | --- | --- |
| 均值 | $\bar{x} \sim \mathcal{N}(\mu, \sigma^2/n)$ | $\bar{\mathbf{X}} \sim \mathcal{N}_p(\boldsymbol{\mu}, \boldsymbol{\Sigma}/n)$ |
| 散布的分布 | $(n-1)s^2/\sigma^2 \sim \chi^2_{n-1}$ | $(n-1)\mathbf{S} \sim \mathcal{W}_p(\boldsymbol{\Sigma}, n-1)$ |
| 检验统计量 | $t = \frac{\sqrt{n}(\bar{x}-\mu_0)}{s}$ | $T^2 = n(\bar{\mathbf{x}}-\boldsymbol{\mu}_0)'\mathbf{S}^{-1}(\bar{\mathbf{x}}-\boldsymbol{\mu}_0)$ |
| 检验用分布 | $t_{n-1}$ | $\frac{n-p}{(n-1)p}T^2 \sim F_{p, n-p}$ |

最后一格正是下一节的核心：**$T^2$ 乘一个只与 $n,p$ 有关的常数后服从 $F$ 分布**，查表与 p 值从此可行。

## 3 单样本检验与置信椭球

用 $T^2$ 检验 $H_0: \boldsymbol{\mu} = \boldsymbol{\mu}_0$ 的流程与一元完全平行：算 $T^2$，与 $F$ 分布的分位数比较。拒绝域为

$$
T^2 > \frac{(n-1)p}{n-p}\, F_{p, n-p}(\alpha)
$$

把不等式反过来，就得到 $\boldsymbol{\mu}$ 的**置信椭球**：

$$
\left\{\boldsymbol{\mu} : n(\bar{\mathbf{x}} - \boldsymbol{\mu})'\mathbf{S}^{-1}(\bar{\mathbf{x}} - \boldsymbol{\mu}) \leq \frac{(n-1)p}{n-p}F_{p, n-p}(\alpha)\right\}
$$

这是一个中心在 $\bar{\mathbf{x}}$、沿 $\mathbf{S}$ 特征向量定向的椭球。它比 $p$ 个一元区间组装成的矩形「盒子」更诚实：盒子忽略变量间的相关，椭球则把相关结构一并纳入。<span class="marginnote">一个经典的取舍：同时给 $p$ 个分量做一元区间，犯第一类错误的概率会膨胀（多重比较问题）。置信椭球是整个向量一起控制错误率；若只想给单个分量做区间，则需要 Bonferroni 校正，区间变成 $\bar{x}_j \pm t_{n-1}(\alpha/2p)\, s_j/\sqrt{n}$。</span>

实践中还有**成对比较**与**两样本情形**：$H_0: \boldsymbol{\mu}_1 = \boldsymbol{\mu}_2$ 时用合并协方差矩阵 $\mathbf{S}_{\text{pooled}}$，统计量形式完全一致，只是自由度换成 $n_1 + n_2 - 2$。这正是下一章 MANOVA 的先声——两组推广到 $k$ 组，就是多元方差分析。

## 4 两样本 T²：合并协方差

当问题变成「两个总体的均值向量是否相等」，$H_0: \boldsymbol{\mu}_1 = \boldsymbol{\mu}_2$，我们有来自总体 1 的 $n_1$ 个观测、总体 2 的 $n_2$ 个观测。假设两个总体协方差矩阵相同（即 $\boldsymbol{\Sigma}_1 = \boldsymbol{\Sigma}_2$），用**合并协方差矩阵（pooled covariance matrix）**估计共同的 $\boldsymbol{\Sigma}$：

$$
\mathbf{S}_{\text{pooled}} = \frac{(n_1-1)\mathbf{S}_1 + (n_2-1)\mathbf{S}_2}{n_1 + n_2 - 2}
$$

这是两个样本协方差的加权平均，权重正比于各自的自由度——与一元两样本 $t$ 检验里合并方差 $s_p^2 = \frac{(n_1-1)s_1^2 + (n_2-1)s_2^2}{n_1+n_2-2}$ 完全同构。<span class="marginnote">「协方差矩阵相等」是可检验的假设，对应 Box's M 检验；不等时两样本 $T^2$ 没有精确分布，只有大样本近似——这正是一元 Behrens–Fisher 问题在多元的回响，也是实战里最容易踩的坑。</span>

两样本 $T^2$ 统计量为

$$
T^2 = \frac{n_1 n_2}{n_1 + n_2}\left(\bar{\mathbf{x}}_1 - \bar{\mathbf{x}}_2\right)'\mathbf{S}_{\text{pooled}}^{-1}\left(\bar{\mathbf{x}}_1 - \bar{\mathbf{x}}_2\right)
$$

系数 $\frac{n_1 n_2}{n_1+n_2}$ 是一元情形 $\frac{n_1 n_2}{n_1+n_2}$ 的直接搬运（它等于 $\frac{1}{1/n_1+1/n_2}$，是「两个均值的方差之和」的倒数）。检验用

$$
\frac{n_1+n_2-p-1}{(n_1+n_2-2)p}\,T^2 \sim F_{p,\ n_1+n_2-p-1}
$$

**结构警句：样本量进入公式的唯一途径是自由度 $n_1+n_2-2$ 与分子 $n_1+n_2-p-1$**——多一个总体就多消耗 $p$ 个自由度。当 $p$ 大到接近总样本量时，分母自由度趋于 0，检验失效。把 $k=2$ 组推广到 $k>2$ 组，就是下一章 MANOVA 的分组矩阵设计。

## 5 公式解析：T² 为什么能化成 F

**关键事实：$T^2$ 本身不是 $F$，但乘上常数后是。** 拆解如下：

$$
\frac{n-p}{(n-1)p}\, T^2 \sim F_{p, n-p}
$$

- **第一步，分解自由度**：$T^2$ 的分子里 $\bar{\mathbf{X}}$ 携带 $p$ 维不确定性（$p$ 个均值参数），分母 $\mathbf{S}$ 的 Wishart 自由度是 $n-1$。$F$ 分布正是两个 $\chi^2$ 之比除以各自自由度，于是分子自由度 $p$、分母自由度 $n-p$ 出现在分母的常数 $(n-p)/((n-1)p)$ 里。
- **第二步，理解 $n-p$**：$p$ 个均值估计消耗掉 $p$ 个自由度，剩余 $n-p$ 归估计散布。**当 $p$ 接近 $n$ 时自由度趋近 0，$F$ 分布的方差爆炸**——这正是高维困境的第一缕信号。
- **第三步，几何直觉**：$T^2$ 度量「样本均值落在置信椭球表面的标准化距离」。$F$ 检验在问：这个距离是否大得不像纯随机抽样会出现的？距离阈值由 $F_{p,n-p}$ 决定。

**核心结论：$T^2 \to F$ 的换算，把「矩阵世界」的统计量拉回「标量世界」的分布表**——这就是为什么半个世纪以来所有教科书都给你一张 $F$ 表而不是一张矩阵分布表。

## 6 小结

- **Wishart 分布** $\mathcal{W}_p(\boldsymbol{\Sigma}, n)$ 是卡方分布的矩阵推广；$(n-1)\mathbf{S} \sim \mathcal{W}_p(\boldsymbol{\Sigma}, n-1)$，且 $n \geq p$ 时才正定。
- **Hotelling $T^2$** 是一元 $t^2$ 的多元版本：$T^2 = n(\bar{\mathbf{x}}-\boldsymbol{\mu}_0)'\mathbf{S}^{-1}(\bar{\mathbf{x}}-\boldsymbol{\mu}_0)$，是均值偏离的马氏距离平方。
- 检验 $H_0: \boldsymbol{\mu}=\boldsymbol{\mu}_0$ 用 $\frac{n-p}{(n-1)p}T^2 \sim F_{p,n-p}$；拒绝域、置信椭球都从这条换算来。
- **置信椭球**同时覆盖整个均值向量，优于忽略相关的矩形盒子；单分量区间需 Bonferroni 校正。
- 两样本 $T^2$