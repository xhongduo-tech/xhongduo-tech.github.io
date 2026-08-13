---
title: Gauss-Markov 定理与最优线性无偏估计
date: 2026-08-07
---

# Gauss-Markov 定理与最优线性无偏估计

<div class="epigraph">
<p>在所有线性无偏的竞争者中，最小二乘是无可争议的冠军。</p>
<footer>—— 卡尔 · 弗里德里希 · 高斯（Carl Friedrich Gauss）与安德烈 · 马尔可夫（Andrey Markov）</footer>
</div>

<div class="article-byline">
<p>第二级 · 回归分析与线性模型 ｜ Seber & Lee《线性回归分析》第2章 ｜ 2026-08-07</p>
</div>

## 为什么从最优性定理开始

上一课我们给了 OLS 一堆好性质（无偏、协方差公式），但还没回答「为什么是最小二乘而不是别的」。也许存在某个更聪明的估计量，无偏且方差更小？**Gauss-Markov 定理**一锤定音：在**线性无偏估计**这个类里，最小二乘估计的方差是最小的——没有任何线性无偏估计能打败它。这是整个线性模型理论的中心定理，也是理解「为什么要用 OLS」的最终答案。同时，它的前提条件精确划出了 OLS 的适用范围。

## 1 定理的完整陈述

**Gauss-Markov 定理**：在模型 $\mathbf{y} = \mathbf{X}\boldsymbol{\beta} + \boldsymbol{\varepsilon}$ 中，若误差满足

$$
E(\boldsymbol{\varepsilon}) = \mathbf{0}, \qquad \mathrm{Var}(\boldsymbol{\varepsilon}) = \sigma^2 \mathbf{I}
$$

（**零均值、等方差、不相关**，但不要求正态分布），则对任一可估的线性组合 $\mathbf{c}'\boldsymbol{\beta}$，最小二乘估计 $\mathbf{c}'\hat{\boldsymbol{\beta}}$ 在所有**线性无偏估计**中方差最小。

这里三个关键词需要精确理解：

- **线性**：估计量必须是 $\mathbf{y}$ 的线性函数，即形如 $\mathbf{a}'\mathbf{y}$；
- **无偏**：$E(\mathbf{a}'\mathbf{y}) = \mathbf{c}'\boldsymbol{\beta}$ 对所有 $\boldsymbol{\beta}$ 成立；
- **最优**：在满足前两者的所有候选里，方差 $\mathrm{Var}(\mathbf{a}'\mathbf{y})$ 最小。

<span class="marginnote">定理不要求误差正态！正态性只影响分布（从而影响检验与区间），不影响 Gauss-Markov 的最优性。很多教材强调这一点：OLS 的最优性是一个「矩条件」命题，与分布无关。</span>

## 2 公式解析：为什么 OLS 是最优的

核心不等式可以写成：对任意满足无偏性的线性估计 $\tilde{\boldsymbol{\beta}} = \mathbf{A}\mathbf{y}$（$\mathbf{A}$ 是线性变换矩阵），有

$$
\mathrm{Var}(\tilde{\boldsymbol{\beta}}) - \mathrm{Var}(\hat{\boldsymbol{\beta}}) \quad \text{是半正定矩阵}
$$

拆解证明思路（矩阵版的“配方”技巧）：

- **无偏性约束**：$\tilde{\boldsymbol{\beta}} = \mathbf{A}\mathbf{y}$ 无偏要求 $E(\mathbf{A}\mathbf{y}) = \mathbf{A}\mathbf{X}\boldsymbol{\beta} = \boldsymbol{\beta}$ 对所有 $\boldsymbol{\beta}$ 成立，故必须 $\mathbf{A}\mathbf{X} = \mathbf{I}$。任何竞争者的系数矩阵都受这条约束。
- **变形技巧**：把任意无偏线性估计写成 $\tilde{\boldsymbol{\beta}} = \hat{\boldsymbol{\beta}} + \mathbf{D}\mathbf{y}$，其中 $\mathbf{D} = \mathbf{A} - (\mathbf{X}'\mathbf{X})^{-1}\mathbf{X}'$。由 $\mathbf{A}\mathbf{X}=\mathbf{I}$ 可推出 $\mathbf{D}\mathbf{X} = \mathbf{0}$。
- **方差展开**：$\mathrm{Var}(\tilde{\boldsymbol{\beta}}) = \sigma^2[(\mathbf{X}'\mathbf{X})^{-1} + \mathbf{D}\mathbf{D}']$，其中交叉项 $\mathrm{Cov}(\hat{\boldsymbol{\beta}}, \mathbf{D}\mathbf{y})$ 因 $\mathbf{D}\mathbf{X}=\mathbf{0}$ 而消失。
- **结论**：因为 $\mathbf{D}\mathbf{D}'$ 半正定，$\mathrm{Var}(\tilde{\boldsymbol{\beta}}) \succeq \sigma^2(\mathbf{X}'\mathbf{X})^{-1} = \mathrm{Var}(\hat{\boldsymbol{\beta}})$。等号当且仅当 $\mathbf{D}=\mathbf{0}$，即 $\mathbf{A} = (\mathbf{X}'\mathbf{X})^{-1}\mathbf{X}'$——**唯一的赢家就是 OLS**。

<span class="marginnote">这里的「配方」技巧（把任意解写成最优解加残差项）在统计学中反复出现——证明 BLUE、推导岭回归、分析工具变量估计都用到同一套路。学会它，就学会了一类证明。</span>

## 3 BLUE：三个字母的含义

Gauss-Markov 定理给出的最优估计称为 **BLUE（Best Linear Unbiased Estimator，最优线性无偏估计）**：

**B（Best）**：方差最小（按半正定序）；
**L（Linear）**：限制在线性估计类内；
- **U（Unbiased）**：无偏是候选资格；
- **E（Estimator）**：是估计量而非预测值。

**重点结论**：$\hat{\boldsymbol{\beta}} = (\mathbf{X}'\mathbf{X})^{-1}\mathbf{X}'\mathbf{y}$ 是 $\boldsymbol{\beta}$ 的 BLUE。<span class="marginnote">注意定理的范围：它只在「线性无偏」这个类里称王。非线性估计、有偏估计都可能方差更小——岭回归就是「有偏但更小 MSE」的例子，走出 BLUE 的框架才谈得上它们。</span>

## 4 定理的三个前提：一个都不能少

Gauss-Markov 定理的适用前提与结论同样重要。每缺一条，最优性就崩：

| 前提 | 若违反 | 后果 |
| --- | --- | --- |
| $E(\varepsilon)=0$（零均值） | 有系统偏差 | OLS 无偏性失效，可能整体偏移 |
| $\mathrm{Var}(\varepsilon)=\sigma^2\mathbf{I}$（等方差） | 异方差 | OLS 仍无偏，但不再方差最小 |
| 观测不相关 | 自相关/序列相关 | OLS 标准误失真，检验失效 |

**辨析｜易错点：** 一个常见的误解是「Gauss-Markov 定理要求正态误差」。**不对**——正态性不是定理前提，它只为了让 $t$、$F$ 检验精确。另一个误解是「OLS 无条件最优」：离开线性无偏类，OLS 就未必最优，这为有偏估计（岭回归、LASSO）留出了合法空间。

## 5 定理的实践意义：何时该担心

把定理用回实践，它告诉你三条决策规则：

1. **等方差是效率的关键**：若怀疑异方差（残差呈喇叭形），OLS 虽仍无偏，但它的「最优」头衔失效——这正是下一课《广义最小二乘》登场的时机。
2. **无偏性的代价**：当 $p$ 接近 $n$、设计阵病态时，无偏但方差巨大的 OLS 在实践中惨不忍睹，宁可牺牲无偏换 MSE 更小的岭回归。
3. **检验需分布假设**：Gauss-Markov 管估计的「方差最优」，但 $t$ 检验的 $p$ 值仍依赖正态性（或大样本渐近）。

<span class="marginnote">一句话记忆：Gauss-Markov 说「在公平竞赛（线性无偏）的规则下 OLS 必胜」；但它没承诺「公平竞赛一定最划算」。加约束、换目标函数，就可能出现更好的选手。</span>

## 6 一个直觉示例：OLS 为什么胜过「简单平均」

用一个极端例子体会定理的分量。设模型 $y_i = \beta_1 x_i + \varepsilon_i$（无截距），$x$ 只取两个值：$x = 0$ 与 $x = 1$，各 $n/2$ 个观测。候选估计量有两个：

**候选 A（OLS）**：$\hat{\beta}_1^{\text{OLS}} = \dfrac{\sum x_i y_i}{\sum x_i^2}$——只用 $x=1$ 的观测信息，因为 $x=0$ 的观测对斜率毫无贡献。

**候选 B（朴素平均）**：$\hat{\beta}_1^{\text{B}} = \bar{y}$——把全部 $y$ 平均。

两者都无偏吗？候选 B 的期望是 $E(\bar{y}) = \beta_1/2$（因为一半观测的 $x=0$ 期望为 0）——**有偏**，根本没有参赛资格。这正是定理的价值：它先排除「看似合理但无偏性不达标」的候选。

再看候选 C：$\hat{\beta}_1^{\text{C}} = y_{(1)}$——只取第一个 $x=1$ 观测的 $y$。它无偏（$E = \beta_1$），方差为 $\sigma^2$；而 OLS 的方差 $\mathrm{Var}(\hat{\beta}_1^{\text{OLS}}) = \sigma^2\sum_{x_i=1}(2/n)^2 = 2\sigma^2/n$。当 $n=10$ 时，候选 C 的方差是 OLS 的 5 倍——**同为无偏线性估计，方差天差地别**。

要理解 OLS 为何最优，看无偏性约束：$E(\sum_i a_i y_i) = \beta_1$ 等价于 $\sum_{x_i=1} a_i = 1$。在这条约束下最小化 $\sigma^2\sum_i a_i^2$，由柯西不等式可知唯一最优解是让所有 $a_i$ 相等（$a_i = 2/n$）——这正是 OLS 的权重。任何偏离都会让方差变大。<span class="marginnote">这个推导是 Gauss-Markov 证明的微缩版：无偏给约束，方差做目标，约束优化解出的赢家恰好是 OLS。注意「所有 $x=1$ 观测权重相等」是柯西不等式的直接结论。</span>

<span class="marginnote">这个例子的意义在于「无偏是入场券，方差最小是冠军」。Gauss-Markov 定理不是告诉我们 OLS 运气好，而是说明：在公平竞争（线性无偏）的规则下，最小二乘的权重分配是数学上唯一最优的。</span>

## 7 定理的现代回响：从 BLUE 到正则化

Gauss-Markov 定理把 OLS 推上「线性无偏最优」的宝座，但它也划出了边界——一旦离开「线性无偏」这个类，就有更优的可能。这条边界在现代统计学中不断被叩击：

**有偏但更稳**：岭回归（第 3 篇）牺牲无偏换方差，MSE 反而更小——它挑战的是定理的「无偏」前提；
**非线性估计**：在误差非正态时，稳健估计（如 M 估计）放弃线性，换取对离群点的免疫；
- **高维时代**：当 $p > n$，线性无偏估计类里根本没有「方差有限」的成员，必须引入惩罚——BLUE 在 $p \gg n$ 时不再适用。

<span class="marginnote">理解定理的最好方式是理解它的边界：Gauss-Markov 的「最优」是<strong>类内最优</strong>，而「类」的选择（线性、无偏）本身就是建模决策。现代统计的许多进展，正是通过放宽这两条边界获得的。</span>

**辨析｜易错点：** 不要在汇报时把「OLS 是 BLUE」说成「OLS 是最优估计」——那是过度引申。正确表述是「在误差零均值、等方差、不相关的线性无偏估计类中，OLS 的方差最小」。加了前提的最优，才是诚实的结论。

## 8 小结

- **Gauss-Markov 定理**：误差零均值、等方差、不相关时，OLS 是 $\boldsymbol{\beta}$ 的 **BLUE**（线性无偏估计中方差最小）。
- 证明核心是「配方」技巧：任意无偏线性估计 = OLS + 正交残差项，残差项只增不减方差。
- 定理**不要求正态性**；正态性只影响检验与区间。
- 前提残缺的后果：异方差 → 效率丢失；自相关 → 标准误失真；非零均值 → 无偏性崩溃。
- 定理的边界即后续方法的入口：异方差给 GLS/WLS，高维病态给岭回归。

在下一节，我们把检验从「单个系数」升级到「整个参数向量」——**多元回归的 $F$