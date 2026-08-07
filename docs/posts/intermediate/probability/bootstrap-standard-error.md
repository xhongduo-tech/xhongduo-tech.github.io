---
title: 用 bootstrap 估计标准误与均方误差
date: 2026-08-07
---

# 用 bootstrap 估计标准误与均方误差

<div class="epigraph">
<p>标准误衡量「这次估计大概偏多远」，均方误差把偏差与方差一锅端——bootstrap 让它们对任何统计量都可算。</p>
<footer>—— 布拉德利 · 埃夫隆（Bradley Efron）</footer>
</div>

<div class="article-byline">
<p>第二级 · 概率论与数理统计 ｜ 盛骤《概率论与数理统计》§10 ｜ 2026-08-07</p>
</div>

## 为什么从 bootstrap 估计标准误开始

上一节立起了 bootstrap 的框架，本节把第一个具体应用做扎实：**用 bootstrap 估计统计量的标准误与均方误差（MSE）**。标准误是「$\hat\theta$ 偏离 $\theta$ 的典型幅度」，MSE 是「$\hat\theta$ 与 $\theta$ 的平方距离的期望」——它们是评估任何估计量精度的两大指标，而 bootstrap 让它们在**没有解析公式**时也能算。

这个应用的价值在「通用性」：中位数、比值、相关系数、甚至某个自编指标的解析标准误往往不存在或极难推导，bootstrap 一律用「反复重算」搞定。它是数据科学里报告「性能指标的波动」的标准方法。<span class="marginnote">标准误与 MSE 的关系一句话：<strong>$\mathrm{MSE}(\hat\theta) = \mathrm{Var}(\hat\theta) + \text{bias}^2$——均方误差 = 方差 + 偏差平方</strong>。无偏估计量的 MSE 就是方差；有偏估计量的 MSE 把「偏差的代价」也计入了。bootstrap 能同时估出方差与偏差，于是 MSE 也随手可得。</span>

## 1 bootstrap 估计标准误

设 $\hat\theta$ 是感兴趣的统计量（如中位数、相关系数），bootstrap 估计标准误：

$$\widehat{se}_B(\hat\theta) = \sqrt{\frac{1}{B-1}\sum_{b=1}^{B}\left(\hat\theta^{*b} - \bar{\hat\theta}^*\right)^2}$$

其中 $\hat\theta^{*1}, \ldots, \hat\theta^{*B}$ 是 $B$ 个 bootstrap 复制，$\bar{\hat\theta}^*$ 是它们的均值。

**例（中位数）**：50 个家庭年收入，样本中位数 $\hat\theta = 38$ 万元。放回抽样 $B = 2000$ 次，得到 2000 个 bootstrap 中位数，其样本标准差 $\widehat{se} = 2.1$ 万元——中位数估计的典型误差约 2.1 万。<span class="marginnote">「中位数的标准误」是个好例子：它的解析公式复杂到几乎没人用（需要密度估计），但 bootstrap 十行代码就给出合理估计。这正是 bootstrap 的卖点——「解析做不到的，模拟来做」。收入的偏态让均值不靠谱、中位数才是合理指标，而它的误差只能靠 bootstrap 之类的方法。</span>

## 2 bootstrap 估计偏差

**偏差（bias）** 的 bootstrap 估计：$\hat\theta$ 的「期望」与「实际值」之差

$$\widehat{bias}_B = \bar{\hat\theta}^* - \hat\theta$$

即「bootstrap 复制的均值 − 原始估计值」。若 $\bar{\hat\theta}^*$ 明显偏离 $\hat\theta$，说明估计量有系统偏差。

**偏差校正（bias-corrected）估计**：$\tilde\theta = \hat\theta - \widehat{bias}_B = 2\hat\theta - \bar{\hat\theta}^*$——把估计量「往回掰」掉偏差。<span class="marginnote">「偏差校正」的直觉：如果 bootstrap 复制平均比原始估计高 0.5（$\bar{\hat\theta}^* - \hat\theta = 0.5$），说明估计量平均偏高，那就把估计量下调 0.5。这个「用 bootstrap 估计偏差再修正」的技巧在复杂估计量（如相关系数的偏校正）里很有用——但它本身也引入新误差，样本小时慎用。</span>

## 3 bootstrap 估计均方误差

**均方误差（mean squared error, MSE）**：$E[(\hat\theta - \theta)^2]$，可分解为

$$\mathrm{MSE}(\hat\theta) = \mathrm{Var}(\hat\theta) + \big[\mathrm{bias}(\hat\theta)\big]^2$$

bootstrap 估计：

$$\widehat{MSE}_B = \frac{1}{B}\sum_{b=1}^{B}(\hat\theta^{*b} - \hat\theta)^2$$

即「bootstrap 复制对原始估计的平方偏离的平均」。它把方差与偏差一起打包——**MSE 是「估计量与目标的总距离」**，用于比较不同估计量的整体好坏。<span class="marginnote">「方差 + 偏差平方」是估计量评估的总公式：一个无偏但方差大的估计量与一个有偏但方差小的估计量，谁更好？看 MSE。第七章的「有效性」只比较无偏估计量的方差，MSE 放宽到「允许有偏但总体更近」——这是「偏差—方差权衡」的定量版本，也是机器学习里 bias-variance tradeoff 的统计学根源。</span>

**例（估计量的比较）**：估计总体方差，候选 $S^2$（无偏）与 $B_2$（有偏但方差小）。用 bootstrap 估两者的 MSE：$S^2$ 的 MSE = 方差（无偏），$B_2$ 的 MSE = 方差 + bias²。$n$ 小时 $B_2$ 的 MSE 可能更小——「有偏但整体更准」的经典情形。

## 4 公式解析：MSE = 方差 + 偏差²

MSE 的分解是本节的数学核心，拆开：

$$

\mathrm{MSE}(\hat\theta) = E[(\hat\theta - \theta)^2] = E[(\hat\theta - E[\hat\theta])^2] + (E[\hat\theta] - \theta)^2 = \mathrm{Var}(\hat\theta) + \mathrm{bias}^2

$$

- **第一步，加减期望**：$(\hat\theta - \theta) = (\hat\theta - E[\hat\theta]) + (E[\hat\theta] - \theta)$——拆成「偏离期望」与「期望偏离目标」两块。
- **第二步，平方展开**：平方后交叉项 $E[(\hat\theta - E[\hat\theta])(E[\hat\theta] - \theta)] = 0$（第一项期望为 0），只剩两个平方项。
- **第三步，识别**：第一项是方差，第二项是偏差平方——MSE = 方差 + 偏差²。

「偏差方差分解」是估计理论的总纲：它把「估计好坏」拆成「波动大小」与「系统偏移」两个正交来源。bootstrap 的价值在于——这两块都能用模拟估出来，无需解析。

## 5 bootstrap 标准误与 MSE 的深入

「用复制估标准误、用复制拆 MSE」是 bootstrap 的两大招牌，它们的细节与边界值得展开。

### 例：中位数标准误的 bootstrap 估计

**例**：50 个收入数据，样本中位数 $\hat\theta = 38$ 万元。放回抽 $B = 2000$ 个 bootstrap 样本，各算中位数，得 2000 个复制。其样本标准差：

$$\widehat{se}_B = \sqrt{\frac{1}{1999}\sum(\hat\theta^{*b} - \bar{\hat\theta}^*)^2} = 2.1$$

中位数解析标准误需密度估计，bootstrap 直接给——「复杂统计量 + bootstrap = 标准答案」。

### 偏差估计的直觉

| 量 | 公式 | 含义 |
| --- | --- | --- |
| 原始估计 | $\hat\theta$ | 本次估计 |
| 复制均值 | $\bar{\hat\theta}^*$ | bootstrap 平均 |
| 偏差估计 | $\bar{\hat\theta}^* - \hat\theta$ | 系统偏移 |
| 校正估计 | $2\hat\theta - \bar{\hat\theta}^*$ | 往回掰 |

**「复制均值偏离原始估计 = 偏差」**——若复制平均比原始高 0.03，说明估计量平均偏高，往下调 0.03。

### MSE 与「偏差—方差权衡」

$$\mathrm{MSE} = \mathrm{Var} + \text{bias}^2$$

| 估计量 | 偏差 | 方差 | MSE |
| --- | --- | --- | --- |
| 无偏（$S^2$） | 0 | 大 | 方差 |
| 有偏（$B_2$） | 小负 | 小 | 可能更小 |

$n$ 小时 $B_2$ 的 MSE 可能小于 $S^2$——「轻微有偏但方差小」整体更优。**MSE 是「总距离」，比单看无偏更全面**，这也是机器学习里 bias-variance tradeoff 的根源。

### 例：用 bootstrap 比较两个估计量

**例**：估计总体方差，候选 $S^2$ 与 $B_2$。bootstrap 分别估两者的 MSE：从样本放回抽，各算 $S^2$、$B_2$，比较对 $\sigma^2$ 的平方偏差。$n$ 小时 $B_2$ 的 MSE 常更小——「有偏但更准」的实证。

**易错点｜辨析：** ① 标准误公式分母用 $B-1$（无偏修正），MSE 公式用 $B$（定义如此）——别混；② 偏差估计本身有噪声，样本小时校正可能「越校越偏」；③ bootstrap 估 MSE 需要「真值参照」——用原始估计当参照是近似，理论 MSE 是对真值的。

## 6 小结

- **bootstrap 标准误**：$\widehat{se}_B = \sqrt{\frac{1}{B-1}\sum(\hat\theta^{*b} - \bar{\hat\theta}^*)^2}$——复制的样本标准差。
- **bootstrap 偏差**：$\widehat{bias}_B = \bar{\hat\theta}^* - \hat\theta$，偏差校正估计 $\tilde\theta = 2\hat\theta - \bar{\hat\theta}^*$。
- **MSE**：$E[(\hat\theta-\theta)^2] = \mathrm{Var} + \text{bias}^2$；bootstrap 版 $\frac1B\sum(\hat\theta^{*b}-\hat\theta)^2$。
- MSE 是「偏差—方差权衡」的定量表达，用于比较有偏/无偏估计量的整体好坏。
- 解析公式算不出的标准误与 MSE，bootstrap 一律用反复重算搞定。

在下一节，我们用 bootstrap 构造置信区间——**bootstrap 置信区间**。
