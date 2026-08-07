---
title: 布朗运动的变体：带漂移布朗运动与几何布朗运动
date: 2026-08-07
---

# 布朗运动的变体：带漂移布朗运动与几何布朗运动

<div class="epigraph">
<p>加上漂移，布朗运动开始「有主见」；取指数，它便永远为正——金融的两副面孔。</p>
<footer>—— 保罗 · 萨缪尔森（Paul Samuelson）</footer>
</div>

<div class="article-byline">
<p>第二级 · 随机过程 ｜ 张波《应用随机过程》§7.4 ｜ 2026-08-07</p>
</div>

## 从「标准」到「变体」

标准布朗运动是「无漂移、单位方差」的纯随机。现实几乎总要加两样东西：**趋势**（股价长期上涨）与**尺度**（波动幅度）。于是有了两个最重要的变体：

1. **带漂移布朗运动（drifted Brownian motion）**：$X(t) = \mu t + \sigma B(t)$——在随机上叠加线性趋势；
2. **几何布朗运动（geometric Brownian motion, GBM）**：$S(t) = S_0 e^{(\mu - \sigma^2/2)t + \sigma B(t)}$——对带漂移过程取指数。

GBM 是金融建模的招牌：**股价、商品价格、资产价值的标准模型**。它由萨缪尔森在 1965 年提出，是 Black-Scholes 公式（第十篇）的底层假设。它之所以被选中，是因为三个理想性质：**恒为正、收益正态（对数正态分布）、方差随价格水平增长**。<span class="marginnote">GBM 与普通布朗运动的本质差别：<strong>普通布朗可以取负（股价不能），GBM 永远为正</strong>。取指数这个动作，把「允许负值」的算术空间搬进「只允许正值」的对数空间——这恰好是股价的天然几何。</span>

本节目标：定义两个变体、推导 GBM 的矩、并理解那个神秘的「$-\sigma^2/2$」修正项。

## 1 带漂移布朗运动

**带漂移布朗运动（drifted BM）**：
$$
X(t) = \mu t + \sigma B(t), \qquad t \ge 0.
$$
其中 $\mu$ 是**漂移率**（每单位时间的平均增长），$\sigma$ 是**波动率**（每单位时间的标准差）。其增量
$$
X(t) - X(s) \sim N\big( \mu (t-s),\; \sigma^2 (t-s) \big).
$$

**均值与方差**：$E[X(t)] = \mu t$，$\mathrm{Var}(X(t)) = \sigma^2 t$——**趋势管均值，波动管方差**。<span class="marginnote">带漂移布朗运动是「股票收益」的第一步模型：<strong>$\mu t$ 是长期趋势，$\sigma B(t)$ 是随机扰动</strong>。但注意它允许取负值——用于收益（增量）没问题，用于价格（存量）就不行，于是需要 GBM。</span>

## 2 几何布朗运动

**几何布朗运动（GBM）**：
$$
S(t) = S_0 \exp\!\Big( \big(\mu - \frac{\sigma^2}{2}\big) t + \sigma B(t) \Big), \qquad S_0 > 0.
$$
**性质**：$S(t) > 0$ 恒成立（指数恒正）；$S(t)$ 服从**对数正态分布**——$\ln S(t)$ 是正态（带漂移布朗）。

**为什么有 $-\sigma^2/2$**：这是 GBM 最容易被问到的「神秘修正」。它保证「几何平均」与「算术平均」的正确关系：若要让 $E[S(t)] = S_0 e^{\mu t}$（年化收益率为 $\mu$），则指数里的漂移必须是 $\mu - \sigma^2/2$。**直觉**：$e^{\sigma B(t)}$ 的期望是 $e^{\sigma^2 t/2}$（对数正态），多出来的 $e^{\sigma^2 t/2}$ 必须被 $-\sigma^2/2 \cdot t$ 抵消，才让总期望回到 $e^{\mu t}$。<span class="marginnote">$-\sigma^2/2$ 的深意：<strong>波动本身会「消耗」平均收益——波动越大，几何增长比算术增长慢得越多</strong>。这就是「波动拖累（volatility drag）」：一支每年算术收益 10% 但波动 20% 的资产，长期几何收益只有约 $10\% - 20\%^2/2 = 8\%$。这个修正不是数学把戏，是复利的真实几何。</span>

## 3 GBM 的矩

**期望**：
$$
E\big[ S(t) \big] = S_0\, e^{\mu t}.
$$
**证明**：$\ln S(t) = \ln S_0 + (\mu - \sigma^2/2)t + \sigma B(t) \sim N(\ln S_0 + (\mu-\sigma^2/2)t, \sigma^2 t)$，对数正态的期望 $= e^{均值 + 方差/2} = S_0 e^{(\mu - \sigma^2/2)t} \cdot e^{\sigma^2 t/2} = S_0 e^{\mu t}$。

**方差**：
$$
\mathrm{Var}\big(S(t)\big) = S_0^2 e^{2\mu t}\big( e^{\sigma^2 t} - 1 \big).
$$
**方差随 $t$ 与 $\sigma$ 指数增长**——价格的不确定性随时间爆炸，这是「远期期权价格波动大」的数学根源。<span class="marginnote">GBM 的三个矩特征：<strong>均值 $e^{\mu t}$、中位数 $e^{(\mu-\sigma^2/2)t}$、众数 $e^{(\mu-\sigma^2)t}$</strong>——三者层层递进，都差一个 $e^{\sigma^2 t/2}$。对数正态分布的「均值 > 中位数 > 众数」结构，是金融「少数暴涨拉高均值」现象的数学化身。</span>

## 4 公式解析：-σ²/2 修正的完整推导

**目标：推导「若 $S(t) = S_0 e^{\alpha t + \sigma B(t)}$，则 $E[S(t)] = S_0 e^{(\alpha + \sigma^2/2)t}$」，从而看清为何要设 $\alpha = \mu - \sigma^2/2$。**

第一步，提取指数中的布朗部分。设 $S(t) = S_0 e^{\alpha t} \cdot e^{\sigma B(t)}$。$\alpha$ 是待定的「指数漂移」。

第二步，算 $e^{\sigma B(t)}$ 的期望。$B(t) \sim N(0, t)$，正态矩母：
$$
E\big[ e^{\sigma B(t)} \big] = e^{\sigma^2 t / 2}.
$$
第三步，求 $S(t)$ 的期望。$e^{\sigma B(t)}$ 与确定性因子独立：
$$
E[S(t)] = S_0 e^{\alpha t} \cdot e^{\sigma^2 t/2} = S_0 e^{(\alpha + \sigma^2/2)t}.
$$
第四步，反解 $\alpha$。若要求 $E[S(t)] = S_0 e^{\mu t}$（$\mu$ 是年化收益率），则 $\alpha + \sigma^2/2 = \mu$，即 $\alpha = \mu - \sigma^2/2$。**GBM 的指数漂移必须「少算」$\sigma^2/2$，才能让期望回归 $\mu$。**

**这个推导为什么重要**：它精确解释了 GBM 定义里那个「多余」的项——**不是笔误，是对数正态期望的几何修正**。理解这一步，第八篇的 Itô 公式（$dS = S(\mu dt + \sigma dB)$）才会显得自然：从 Itô 公式反推，也会得到同一个 $-\sigma^2/2$。

## 5 应用：股价建模与 Black-Scholes 前夜

GBM 在金融中的统治地位来自三个理想性质：

| 性质 | GBM 的表现 | 为什么重要 |
| --- | --- | --- |
| 恒为正 | $S(t) > 0$ | 股价不可能为负 |
| 收益正态 | $\ln\frac{S(t)}{S(s)} \sim N$ | 收益的统计检验、VaR 计算方便 |
| 方差随水平 | $\mathrm{Var} \propto S_0^2$ | 高价股波动大，符合观察 |

**Black-Scholes 的入口**：Black-Scholes 假设股价服从 GBM（$\mu$ 漂移 + $\sigma$ 波动），配合「风险中性测度」与鞅定价（第十篇），导出期权定价公式。**GBM 是那个公式的底层几何**。<span class="marginnote">GBM 的局限也值得知道：<strong>真实股价有跳跃、波动率会变（微笑）、尾部比对数正态更肥</strong>——所以衍生品定价有随机波动率模型（Heston）、跳跃模型。GBM 是「第一近似」，它的修正构成了现代金融数学的半壁江山。</span>

## 6 小结

- **带漂移布朗运动** $X(t) = \mu t + \sigma B(t)$：增量 $N(\mu\Delta t, \sigma^2\Delta t)$；允许负值。
- **几何布朗运动** $S(t) = S_0 e^{(\mu-\sigma^2/2)t + \sigma B(t)}$：恒正、对数正态。
- **$-\sigma^2/2$ 修正**：抵消 $e^{\sigma B(t)}$ 期望里的 $e^{\sigma^2 t/2}$，让 $E[S(t)] = S_0 e^{\mu t}$。
- **矩**：$E = S_0 e^{\mu t}$，$\mathrm{Var} = S_0^2 e^{2\mu t}(e^{\sigma^2 t}-1)$；均值 > 中位数 > 众数。
- **金融地位**：股价标准模型，Black-Scholes 的底层几何；波动拖累的现实意义。

到这里，第七篇《布朗运动》全部结束。从下一篇起，我们进入随机过程最深的一层——**随机积分初步**：当被积函数自己也随机时，积分怎么定义？
