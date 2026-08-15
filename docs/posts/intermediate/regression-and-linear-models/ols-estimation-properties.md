---
title: 最小二乘估计及其统计性质
date: 2026-08-07
---

# 最小二乘估计及其统计性质

<div class="epigraph">
<p>估计量的价值，不在单次结果，而在长期表现的期望。</p>
<footer>—— 依统计决策论精神改写（paraphrase of statistical decision theory）</footer>
</div>

<div class="article-byline">
<p>第二级 · 回归分析与线性模型 ｜ Seber & Lee《线性回归分析》第2章 ｜ 2026-08-07</p>
</div>

## 为什么从统计性质开始

上一课给出了公式 $\hat{\boldsymbol{\beta}} = (\mathbf{X}'\mathbf{X})^{-1}\mathbf{X}'\mathbf{y}$，但「为什么非得用它」还没被回答。最小二乘只是「最小化平方和」的一种算法选择，凭什么相信它？统计学的回答是：考察它的**长期表现**——无偏吗？方差小吗？在所有候选估计里是最优的吗？本课系统建立估计量的统计性质：期望、协方差、误差分解，并把「估计得好不好」从哲学问题变成可证明的数学命题。

## 1 期望与方差：估计量的第一层体检

在假设 $E(\boldsymbol{\varepsilon})=\mathbf{0}$、$\mathrm{Var}(\boldsymbol{\varepsilon})=\sigma^2\mathbf{I}$ 下，最小二乘估计量有两个基本性质：

**无偏性（unbiasedness）**：

$$
E(\hat{\boldsymbol{\beta}}) = E\big((\mathbf{X}'\mathbf{X})^{-1}\mathbf{X}'\mathbf{y}\big)
= (\mathbf{X}'\mathbf{X})^{-1}\mathbf{X}' E(\mathbf{y})
= (\mathbf{X}'\mathbf{X})^{-1}\mathbf{X}' \mathbf{X}\boldsymbol{\beta}
= \boldsymbol{\beta}
$$

推导的关键一行是 $E(\mathbf{y}) = \mathbf{X}\boldsymbol{\beta}$（误差零均值），然后 $(\mathbf{X}'\mathbf{X})^{-1}\mathbf{X}'\mathbf{X} = \mathbf{I}$ 相消。<span class="marginnote">无偏性说的是「平均而言打中靶心」：无数次重复抽样后，估计值的期望等于真值。它不保证任何单次估计准确，但排除了系统性偏差。</span>

**协方差矩阵（covariance matrix）**：

$$
\mathrm{Var}(\hat{\boldsymbol{\beta}}) = \sigma^2 (\mathbf{X}'\mathbf{X})^{-1}
$$

这是多元情形下「估计精度」的总账：对角元素是各 $\hat{\beta}_j$ 的方差，非对角元素是估计量间的协方差。<span class="marginnote">$\sigma^2(\mathbf{X}'\mathbf{X})^{-1}$ 是后续一切检验、置信域的母体：$t$ 检验的标准误就是它对角元素的平方根。它与简单回归的 $\sigma^2/S_{xx}$ 一脉相承。</span>

## 2 公式解析：协方差矩阵的逐项来源

$$
\mathrm{Var}(\hat{\boldsymbol{\beta}}) = \sigma^2(\mathbf{X}'\mathbf{X})^{-1}
$$

拆解：

- **$\sigma^2$**：误差方差，所有不确定性的根源。$\sigma^2$ 越大，估计越抖。
- **$(\mathbf{X}'\mathbf{X})^{-1}$**：由设计决定的「精度矩阵」。它越大，方差越大——注意是**逆**矩阵，所以 $\mathbf{X}$ 列之间越正交、取值越分散，$\mathbf{X}'\mathbf{X}$ 的逆越小、估计越准。
- **样本量的角色**：$n$ 藏在 $\mathbf{X}'\mathbf{X}$ 里。大致地，$n$ 每翻倍，$\mathbf{X}'\mathbf{X}$ 各元素约翻倍，其逆约减半，标准误约缩到 $1/\sqrt{2}$——这就是「精度只随样本量的平方根增长」的统计现实。
- **$\sigma^2$ 的估计**：用 $\hat{\sigma}^2 = \mathrm{SSE}/(n-p-1)$ 代入，得到**估计的协方差矩阵** $\widehat{\mathrm{Var}}(\hat{\boldsymbol{\beta}}) = \hat{\sigma}^2(\mathbf{X}'\mathbf{X})^{-1}$，它是区间估计与检验的实操对象。

## 3 误差分解：MSE = 方差 + 偏差²

估计量的好坏不能只看无偏性——无偏但方差极大也不实用。统计学用**均方误差（mean squared error, MSE）** 综合度量：

$$
\mathrm{MSE}(\hat{\boldsymbol{\beta}}) = E\big[(\hat{\boldsymbol{\beta}} - \boldsymbol{\beta})^2\big]
= \mathrm{Var}(\hat{\boldsymbol{\beta}}) + \big(E(\hat{\boldsymbol{\beta}}) - \boldsymbol{\beta}\big)^2
= \text{方差} + \text{偏差}^2
$$

**重点结论：估计误差可分解为方差与偏差平方之和。** 这个分解是全书最重要的观念之一：

- **无偏估计**（如 OLS）偏差为 0，MSE 全由方差贡献；
- **有偏估计**（如第 3 篇的岭回归）牺牲一点偏差，换取方差的更大削减，MSE 反而可能更小；
- 「方差—偏差权衡」（bias-variance tradeoff）由此诞生——它是岭回归、模型选择、乃至机器学习正则化的统一母题。<span class="marginnote">这条分解值得反复品味：它说明「无偏」未必最优。当数据量小、变量多时，一个有偏但稳的估计常常在 MSE 意义上更胜一筹——岭回归正是这条思路的产物。</span>

## 4 无偏性 vs 一致性 vs 有效性

「估计得好」有多个层次，初学者常混为一谈：

| 性质 | 含义 | OLS 是否满足 |
| --- | --- | --- |
| 无偏性 | $E(\hat{\beta})=\beta$，平均不偏 | 是（误差零均值） |
| 一致性 | $n\to\infty$ 时 $\hat{\beta}\to\beta$（依概率） | 是（弱正则条件下） |
| 有效性 | 在所有同类无偏估计中方差最小 | 是（Gauss-Markov，下节） |
| 相合性 | 方差随 $n$ 趋于 0 | 是 |

**辨析｜易错点：** 无偏不一定一致（方差可能不收缩），一致不一定无偏（小样本有偏但渐近收敛）。OLS 同时具备三者，但要到下一课的 **Gauss-Markov 定理**，我们才能宣称它在「线性无偏」类里方差最小。

## 5 残差平方和的分布与 $\hat{\sigma}^2$

在正态误差假设 $\boldsymbol{\varepsilon} \sim N(\mathbf{0}, \sigma^2\mathbf{I})$ 下，OLS 还有一个重要的分布性质：

$$
\frac{\mathrm{SSE}}{\sigma^2} = \frac{\mathbf{y}'(\mathbf{I}-\mathbf{H})\mathbf{y}}{\sigma^2} \sim \chi^2_{n-p-1}
$$

- 这是「误差平方和除以 $\sigma^2$ 服从卡方分布」的多元推广；
- 自由度 $n-p-1$ 与 $\hat{\sigma}^2 = \mathrm{SSE}/(n-p-1)$ 的分母一致，保证了 $E(\hat{\sigma}^2) = \sigma^2$；
- 它是 $t$ 检验与 $F$ 检验的分布基础：$t = \hat{\beta}_j/\mathrm{se}(\hat{\beta}_j)$ 正是「正态量除以卡方量的开方」的标准构造。

<span class="marginnote">这条分布性质把「模型假设」与「检验工具」焊在一起：正态性假设不仅为了好看，它让 SSE 的分布精确已知，从而让 $t$、$F$ 检验的临界值完全确定。</span>

## 6 数值示例：无偏性如何体现在估计中

沿用 5 个样本点 $(1,3),(2,5),(3,4),(4,7),(5,6)$，我们已经算出 $\hat{\boldsymbol{\beta}} = (\hat\beta_0, \hat\beta_1)' = (2.6, 0.8)'$。若这是从某个总体抽出的**一组**样本，那么换一组样本会得到另一个 $\hat{\boldsymbol{\beta}}$——无偏性说的是「所有可能样本的平均结果等于真值」。

把协方差公式算一遍，体会「精度」的数值含义。误差方差估计 $\hat\sigma^2 = 3.6/3 = 1.2$，而设计矩阵为：

$$
\mathbf{X} = \begin{pmatrix} 1 & 1 \\ 1 & 2 \\ 1 & 3 \\ 1 & 4 \\ 1 & 5 \end{pmatrix}, \qquad
\mathbf{X}'\mathbf{X} = \begin{pmatrix} 5 & 15 \\ 15 & 55 \end{pmatrix}, \qquad
(\mathbf{X}'\mathbf{X})^{-1} = \begin{pmatrix} 1.1 & -0.3 \\ -0.3 & 0.1 \end{pmatrix}
$$

于是 $\widehat{\mathrm{Var}}(\hat\beta_1) = 1.2 \times 0.1 = 0.12$，标准误 $\approx 0.346$；$\widehat{\mathrm{Var}}(\hat\beta_0) = 1.2 \times 1.1 = 1.32$，标准误 $\approx 1.149$。可见 $\hat\beta_0$ 的估计远比 $\hat\beta_1$ 粗糙——截距涉及「外推到 $x=0$」，天然更难估准。<span class="marginnote">$\mathbf{X}'\mathbf{X}$ 的元素随 $n$ 线性增长、其逆随 $n$ 近似反比下降，所以标准误大致按 $1/\sqrt{n}$ 收缩——「精度只随样本量平方根增长」在这里有了具体数字。</span>

## 7 小结

- OLS 估计量**无偏**：$E(\hat{\boldsymbol{\beta}}) = \boldsymbol{\beta}$，且 $\mathrm{Var}(\hat{\boldsymbol{\beta}}) = \sigma^2(\mathbf{X}'\mathbf{X})^{-1}$。
- 估计协方差矩阵 $\hat{\sigma}^2(\mathbf{X}'\mathbf{X})^{-1}$ 是一切检验与置信域的母体。
- **误差分解**：$\mathrm{MSE} = \text{方差} + \text{偏差}^2$，引出「方差—偏差权衡」。
- 无偏、一致、有效三个概念层次分明：无偏看期望、一致看极限、有效看方差。
- 正态误差下 $\mathrm{SSE}/\sigma^2 \sim \chi^2_{n-p-1}$，是 $t$ 与 $F$ 检验的分布基础。
- 精度只随样本量平方根增长：$n$ 翻倍、标准误约缩到 $1/\sqrt{2}$。

在下一节，我们将回答「为什么偏偏是 OLS」——**Gauss-Markov 定理**证明它在一切线性无偏估计中方差最小。