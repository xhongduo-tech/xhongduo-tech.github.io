---
title: ARMA 与 ARIMA 模型
date: 2026-08-11
---

# ARMA 与 ARIMA 模型

<div class="epigraph">
<p>一切模型都是错的，但有些是有用的。</p>
<footer>—— 乔治 · 博克斯（George E. P. Box）</footer>
</div>

<div class="article-byline">
<p>第二级 · 进阶数理 · 时间序列分析 ｜ 对标教材 ｜ 2026-08-11</p>
</div>

## 为什么从 ARMA 开始

上一课我们把平稳时间序列分解为「白噪声被某种机制染色」。现在回答最实际的问题：这个染色机制怎么用有限个参数写下来？答案就是 **ARMA 模型**——用过去**观测值**的线性组合加过去**噪声**的线性组合来解释现在。它把一条看似无规律的时间序列压缩成极少几个系数，是全书（Box-Jenkins-Reinsel Ch. 4-6）最核心的模型族。

为什么这个模型值得单独一课？因为它的解释力惊人：**任何弱平稳时间序列，都可以用阶数足够高的 ARMA 模型逼近到任意精度**（Wold 分解定理）。换句话说，ARMA 不是众多模型之一，而是平稳序列的「通用语言」。从 AR(1) 到 MA(1) 再到 ARIMA 的差分处理，是这套语言的三级台阶：AR 捕捉「惯性」，MA 捕捉「冲击」，差分把「趋势」翻译成平稳。

## 1 AR(p)：用过去解释现在

**自回归模型（autoregressive model，AR(p))**：

$$
X_t = \phi_1 X_{t-1} + \phi_2 X_{t-2} + \cdots + \phi_p X_{t-p} + \varepsilon_t
$$

其中 $\varepsilon_t$ 是零均值白噪声。它说的是：现在由过去的 $p$ 个观测值线性决定，外加一个**新息（innovation）** $\varepsilon_t$。名字里的「自回归」点出关键：$X_t$ 对 $X_{t-1}$ 回归，而回归元是自身过去的取值。<span class="marginnote">把 $\phi_1 X_{t-1} + \dots + \phi_p X_{t-p}$ 看成「用过去外推的基线」，$\varepsilon_t$ 是基线之外的不可预测部分。AR 系数 $\phi_i$ 度量「第 $i$ 期前的记忆还剩多少」。</span>

最简单的 AR(1) 已经很有味道：

$$
X_t = \phi X_{t-1} + \varepsilon_t
$$

若 $|\phi| < 1$，递归展开得 $X_t = \sum_{j=0}^\infty \phi^j \varepsilon_{t-j}$——现在被**无穷多个过去的冲击**加权叠加，权重 $\phi^j$ 按几何级数衰减。ACF 为 $\rho(k) = \phi^k$，随 $k$ **指数衰减但不截断**（拖尾）。

## 2 MA(q)：冲击的直接烙印

**滑动平均模型（moving average model，MA(q))**：

$$
X_t = \varepsilon_t + \theta_1 \varepsilon_{t-1} + \cdots + \theta_q \varepsilon_{t-q}
$$

注意这里不是对过去观测值加权，而是对过去**噪声**加权。MA 的 ACF 有一个绝妙性质：

$$
\rho(k) = \frac{\sum_{j=0}^{q-k} \theta_j \theta_{j+k}}{\sum_{j=0}^{q} \theta_j^2}, \quad k \le q, \qquad \rho(k) = 0, \quad k > q
$$

**重点：MA(q) 的 ACF 在滞后超过 $q$ 后截尾为零**。<span class="marginnote">直觉：冲击 $\varepsilon_{t-k}$ 最多「存活」$q$ 期，因此相隔超过 $q$ 的两点不含共同冲击，相关清零。这正是「MA 有记忆有限」的体现。</span>MA(q) 也因此**可逆**：只要特征多项式根都在单位圆外，MA 可以反过来写成无限阶 AR——两个模型族在可逆条件下可以互相表达，这是 ARMA 识别的关键。

## 3 ARMA(p, q)：两家族联姻

把两者合并：

$$
X_t - \phi_1 X_{t-1} - \cdots - \phi_p X_{t-p} = \varepsilon_t + \theta_1 \varepsilon_{t-1} + \cdots + \theta_q \varepsilon_{t-q}
$$

写得更紧凑，引入**滞后算子（backshift operator）** $B$，它满足 $B X_t = X_{t-1}$，于是 $\phi(B) X_t = \theta(B) \varepsilon_t$，其中：

$$
\phi(B) = 1 - \phi_1 B - \cdots - \phi_p B^p, \qquad
\theta(B) = 1 + \theta_1 B + \cdots + \theta_q B^q
$$

滞后算子让「阶数」问题变成「多项式」问题，是全书最优雅的记号工具。

**平稳性与可逆性**：ARMA 过程平稳当且仅当 $\phi(B)$ 的根都在单位圆外；可逆当且仅当 $\theta(B)$ 的根都在单位圆外。这两个条件在下一课识别与估计中反复使用。

**ARMA 的 ACF/PACF 规则**（Box-Jenkins 口诀）：

| 模型 | ACF | PACF |
| --- | --- | --- |
| AR(p) | 拖尾（指数衰减） | 滞后 $p$ 后**截尾** |
| MA(q) | 滞后 $q$ 后**截尾** | 拖尾 |
| ARMA(p, q) | 拖尾 | 拖尾 |

这条表是识别阶数的核心工具，也是下一课《模型识别与参数估计》的引子。

## 4 ARIMA(p, d, q)：给非平稳留门

很多真实序列（GDP、股价、气温）并不平稳——均值随时间漂移。**差分（differencing）** 是通行解法：定义 $\nabla X_t = X_t - X_{t-1} = (1 - B) X_t$，把「水平」换成「变化量」。若 $d$ 阶差分后序列平稳，则称原序列服从 **ARIMA(p, d, q)**：

$$
\phi(B)\, (1 - B)^d\, X_t = \theta(B)\, \varepsilon_t
$$

<span class="marginnote">为什么差分能消除趋势？确定性趋势 $a + bt$ 一阶差分后变成常数 $b$；含随机游走分量的序列差分后直接平稳化——差分把「非平稳的均值」变成「平稳的增量」。单位根是否真的存在，由后续《单位根检验与协整》中的检验判断。</span>

**ARIMA(p, d, q) 的建模逻辑**：先差分 $d$ 次使序列平稳（用单位根检验确认，见后续《单位根检验与协整》），再对差分序列建 ARMA(p, q)。于是模型整体写成：

$$
(1 - B)^d X_t = W_t, \qquad \phi(B) W_t = \theta(B) \varepsilon_t
$$

## 5 公式解析：AR(1) 的 ACF 拖尾

$$
\rho(k) = \phi^k, \qquad k = 1, 2, \dots
$$

拆解：

- **第一步，递归**：$X_t = \phi X_{t-1} + \varepsilon_t$ 反复回代，得 $X_t = \sum_{j=0}^{\infty} \phi^j \varepsilon_{t-j}$。这需要 $|\phi| < 1$ 保证级数收敛——**平稳性条件在此显现**。
- **第二步，算协方差**：$\gamma(k) = \operatorname{Cov}(X_t, X_{t+k}) = E[\sum_i \phi^i \varepsilon_{t-i} \cdot \sum_j \phi^j \varepsilon_{t+k-j}]$。白噪声跨时刻不相关，只有 $i = j - k$ 的项存活，故 $\gamma(k) = \phi^k \sigma_\varepsilon^2 \sum_j \phi^{2j} = \frac{\phi^k}{1-\phi^2}\sigma_\varepsilon^2$。
- **第三步，归一化**：$\rho(k) = \gamma(k)/\gamma(0) = \phi^k$。

**直觉**：$|\phi|$ 越接近 1，记忆越长，ACF 衰减越慢；$\phi < 0$ 时符号交替震荡。**「ACF 慢衰减但不归零」是非平稳（单位根）的警示，而「ACF 指数衰减」则是平稳 AR 的标记**——两者在图上看起来都「拖延」，区分它们要靠差分后是否平稳，这正是《单位根检验与协整》的任务。

## 6 辨析｜易错点

- **AR 与 MA 的截尾/拖尾别搞反**：MA 的 **ACF** 截尾、**PACF** 拖尾；AR 反之。口诀记「AR 的 PACF 截尾，MA 的 ACF 截尾」。
- **差分不是越多越好**：过度差分引入额外可逆性损失且放大噪声——「能一阶就不二阶」。
- **滞后算子 $B$ 不是数**：$B$ 是线性算子，$(1-B)^d$ 是多项式展开，如 $(1-B)^2 = 1 - 2B + B^2$，对应二阶差分 $X_t - 2X_{t-1} + X_{t-2}$。
- **可逆性与平稳性要分开**：ARMA 过程平稳看 $\phi$，可逆看 $\theta$，两者独立，不可混为一谈。

## 7 小结

- **AR(p)**：用过去 $p$ 个观测解释现在，ACF 拖尾、PACF 截尾。
- **MA(q)**：用过去 $q$ 个噪声解释现在，ACF 在滞后 $q$ 后截尾、PACF 拖尾。
- **ARMA(p, q)**：两个多项式 $\phi(B)X_t = \theta(B)\varepsilon_t$；平稳看 $\phi$ 根、可逆看 $\theta$ 根。
- **ARIMA(p, d, q)**：先差分 $d$ 次消除非平稳，再对差分序列建 ARMA。
- **滞后算子 $B$**：把阶数问题变成多项式问题，是全书记号的核心。

在下一节，我们将回答一个更实际的工程问题：拿到一段数据，如何从 ACF/PACF 的样子**识别出**该用 AR(几)、MA(几)、还是 ARIMA 差几阶，再用最大似然等工具把参数估计出来——这就是 **模型识别与参数估计**。
