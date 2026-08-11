---
title: GARCH 与波动率模型
date: 2026-08-11
---

# GARCH 与波动率模型

<div class="epigraph">
<p>风险不在收益本身，而在收益的波动有多难预测。</p>
<footer>—— 罗伯特 · 恩格尔（Robert F. Engle）</footer>
</div>

<div class="article-byline">
<p>第二级 · 进阶数理 · 时间序列分析 ｜ 对标教材 ｜ 2026-08-11</p>
</div>

## 为什么从波动率开始

前几课建模的对象是序列的**均值结构**——$X_t$ 本身长什么样。但金融资产的日收益率有个著名特征：**均值几乎不可预测，方差却高度集聚（volatility clustering）**——大涨大跌扎堆出现，平静期延续良久。对这样的序列，条件方差 $E[X_t^2 \mid \mathcal{F}_{t-1}]$ 本身就是一个随时间演化的对象，而且它直接决定期权的价格、投资组合的风险与 VaR 的估计。

**GARCH（广义自回归条件异方差）模型**，由恩格尔（ARCH，2003 年诺贝尔经济学奖）与博勒斯莱文（Bollerslev，GARCH）建立，把「方差序列」当作又一个 ARMA 过程来建模：今天的波动率由昨天的波动率与昨天的冲击共同决定。它是 Hamilton Ch. 21 的核心，也是「波动率是资产定价的第一变量」这一金融直觉的统计归宿。

## 1 ARCH：波动率会自回归吗

**ARCH（自回归条件异方差）**模型的起点是：误差的条件方差依赖过去误差的平方。

$$
\varepsilon_t = \sigma_t z_t, \qquad
\sigma_t^2 = \omega + \alpha_1 \varepsilon_{t-1}^2 + \cdots + \alpha_q \varepsilon_{t-q}^2
$$

其中 $z_t$ 是独立同分布、零均值、单位方差的白噪声（常假定高斯或 $t$ 分布）。注意**模型的除法**：$\varepsilon_t$ 由「时变尺度」$\sigma_t$ 乘「纯随机」$z_t$ 构成，条件方差 $\operatorname{Var}(\varepsilon_t \mid \mathcal{F}_{t-1}) = \sigma_t^2$ 完全由过去决定。

ARCH 解释了波动率集聚：昨天的大冲击（$\varepsilon_{t-1}^2$ 大）推高今天的 $\sigma_t^2$，今天的波动又推高明天的方差——波动自己「传染」自己。<span class="marginnote">「条件异方差」的含义：无条件方差 $\operatorname{Var}(\varepsilon_t)$ 可以是常数（平稳），但<strong>条件</strong>于过去信息时的方差是随历史演化的。这就是「异方差」（方差非恒定）为何加了「条件」二字。</span>

## 2 GARCH：方差的 ARMA 化

**GARCH(p, q)** 把「滞后方差的滞后平方」也纳入：

$$
\sigma_t^2 = \omega + \sum_{i=1}^q \alpha_i \varepsilon_{t-i}^2 + \sum_{j=1}^p \beta_j \sigma_{t-j}^2
$$

这是 ARCH 的自然推广：让 $\sigma_t^2$ 依赖自身过去的滞后，等价于把「隐含的 ARMA」写完整。<span class="marginnote">正是博勒斯莱文观察到：ARCH(q) 要捕捉长记忆的波动需要很大的 $q$，而加入 $\beta$ 项后一个 GARCH(1,1) 常常就够了——波动率的记忆是「指数加权平均」，并非 ARCH 那样的纯有限阶。</span>

**GARCH(1,1)** 是实务中最常用的模型：

$$
\sigma_t^2 = \omega + \alpha \varepsilon_{t-1}^2 + \beta \sigma_{t-1}^2
$$

三个系数各有语义：

- **$\omega > 0$**：基准波动水平（长期平均方差的锚）。
- **$\alpha$**：对「新冲击」的敏感度——昨天的 $\varepsilon_{t-1}^2$ 对今天波动的影响。
- **$\beta$**：波动的**持续性**——昨天的波动对今天的影响。

**平稳条件**：$\alpha + \beta < 1$ 保证无条件方差有限：$E[\varepsilon_t^2] = \frac{\omega}{1 - \alpha - \beta}$。<span class="marginnote">$\alpha + \beta$ 之和被称作「波动率的持久性」。若接近 1，方差过程的记忆极长，冲击的影响要很久才消散——金融序列常观测到 $\alpha+\beta$ 在 0.95 以上，个别甚至逼近 1。</span>

## 3 无条件方差与峰度：GARCH 的指纹

GARCH 误差的**无条件分布**有两大特征，与金融数据吻合：

**无条件方差有限但条件方差可变**：当 $\alpha + \beta < 1$，$E[\varepsilon_t^2] = \omega/(1-\alpha-\beta)$ 为常数，序列弱平稳。
- **高峰厚尾（leptokurtic）**：即使 $z_t$ 是高斯，$\varepsilon_t$ 的无条件分布也呈现**正超额峰度**——因为方差在高低之间切换，混合分布比高斯更「胖尾」。这正是「收益序列看起来不像高斯」的模型解释。<span class="marginnote">若 $z_t$ 本身取 $t$ 分布，尾部更厚。实务中估计 GARCH 时常在「高斯假设下的 QMLE」与「$t$ 分布假设下的 MLE」之间选择，后者对小概率事件更稳健。</span>

## 4 公式解析：GARCH(1,1) 的方差递推

$$
\sigma_t^2 = \omega + \alpha \varepsilon_{t-1}^2 + \beta \sigma_{t-1}^2
$$

- **$\sigma_t^2$**：$t$ 时刻的条件方差，是「用截至 $t-1$ 的信息对 $t$ 时刻波动的事前预测」。
- **$\alpha \varepsilon_{t-1}^2$**：**冲击项**——昨天的实际偏差平方，度量「最近发生的意外有多大」。新信息权重为 $\alpha$。
- **$\beta \sigma_{t-1}^2$**：**记忆项**——昨天的方差预测本身，即「对波动的预测继续延续」。持久性权重为 $\beta$。
- **递推展开**：反复代入 $\sigma_{t-j}^2$ 得 $\sigma_t^2 = \frac{\omega}{1-\beta} + \alpha \sum_{j=1}^\infty \beta^{j-1} \varepsilon_{t-j}^2$——**GARCH(1,1) 的方差是指数加权移动平均（EWMA）**：越久远的冲击权重按 $\beta^j$ 衰减，这就是「长记忆的短实现」。
- **预测**：向前 $h$ 步的条件方差预测向无条件方差 $\omega/(1-\alpha-\beta)$ 收敛，收敛速度由 $\alpha+\beta$ 决定。

## 5 估计与检验

GARCH 参数的估计几乎总是**最大似然**。对数似然（高斯 $z_t$）为：

$$
\ell(\theta) = -\frac{1}{2}\sum_{t=1}^N \left( \log \sigma_t^2(\theta) + \frac{\varepsilon_t^2}{\sigma_t^2(\theta)} \right)
$$

其中 $\sigma_t^2(\theta)$ 由 GARCH 递推随 $\theta = (\omega, \alpha, \beta)$ 逐步算出。<span class="marginnote">这一递推的初始化（$\sigma_1^2$ 取样本方差或无条件方差）影响不大，但残差标准化后应近似白噪声——建模后仍需 Ljung-Box 式诊断，只是对象换成 $\hat z_t^2$ 与 $\hat z_t$。</span>

检验是否存在 ARCH 效应的经典工具是 **Engle 的 LM 检验**：对 $\varepsilon_t^2$ 关于自身滞后做回归，看 $R^2$ 是否显著（$N R^2 \sim \chi^2_q$）。先检验再建 GARCH，避免无的放矢。

**辨析｜易错点**：

- **均值模型与方差模型分开**：GARCH 建在**残差**上——先对 $X_t$ 拟合均值模型（如 ARMA 或常数），得到 $\varepsilon_t$，再对其建模方差。均值与方差同时估计更严格（联合 MLE），但务必别把「对 $X_t$ 直接回归」当 GARCH。
- **$\alpha + \beta \ge 1$ 很危险**：方差过程非平稳、无条件方差发散，估计结果几乎不可用。
- **GARCH 预测的是波动而非方向**：$\sigma_{t+1}^2$ 告诉你「明天波动多大」，不告诉你「明天涨跌」。收益符号的可预测性远弱于波动——这正是有效市场的写照。

## 6 小结

- **波动率集聚**是金融数据的核心特征：波动自己传染自己，条件方差随时间演化。
- **ARCH(q)**：$\sigma_t^2 = \omega + \sum \alpha_i \varepsilon_{t-i}^2$，条件方差依赖过去冲击平方。
- **GARCH(p, q)**：加入 $\beta_j \sigma_{t-j}^2$ 项，方差自身的滞后进入方程；**GARCH(1,1)** 是实务标配。
- **GARCH(1,1) = 方差的指数加权平均**：$\sigma_t^2 = \frac{\omega}{1-\beta} + \alpha\sum \beta^{j-1}\varepsilon_{t-j}^2$。
- **平稳条件** $\alpha + \beta < 1$；**厚尾**来自条件方差的随机性。
- **估计用 MLE**，存在性用 Engle LM 检验，残差诊断用标准化残差的 Ljung-Box。

在下一节，我们将换一副完全不同的眼镜来看时间序列——从**频域**而非时域出发：一条序列可以分解成不同频率的正弦波的叠加，这就是 **谱分析与周期图**，它与谱密度的概念将把 ACF 的观点翻个面。
