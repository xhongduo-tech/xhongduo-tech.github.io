---
title: 白噪声、滑动平均与自回归过程
date: 2026-08-07
---

# 白噪声、滑动平均与自回归过程

<div class="epigraph">
<p>白噪声是原料，滑动平均是打磨，自回归是回响——时间序列的三种基本音色。</p>
<footer>—— 乔治 · 博克斯（George E. P. Box）</footer>
</div>

<div class="article-byline">
<p>第二级 · 随机过程 ｜ 张波《应用随机过程》§9.5 ｜ 2026-08-07</p>
</div>

## 时间序列的「三原色」

任何宽平稳时间序列的基础构件，是三类最简单的过程：**白噪声（WN）**、**滑动平均（MA）**、**自回归（AR）**。它们像三原色：白噪声是「纯净无记忆」的原料，MA 是「对噪声的有限打磨」（只依赖最近的噪声），AR 是「对过去的回响」（依赖自己的过去）。三者合成 **ARMA** 家族——时间序列建模的通用框架。

这三个过程各有标志性 ACF 形状（上一节预告过）：**白噪声单峰、MA 截断、AR 指数衰减**。识别这些形状，是时间序列分析的入门武功。<span class="marginnote">为什么先学这三个：<strong>任何平稳序列（相当广）都可以写成 ARMA——Wold 分解定理说：任何平稳纯非确定过程 = 白噪声过无限 MA 滤波</strong>。所以 WN/MA/AR 不只是三个例子，而是整个平稳序列的「字母表」。</span>

本节目标：定义三类过程、给出平稳性条件与 ACF 形状、并建立 ARMA 的统一视角。

## 1 白噪声

**白噪声（white noise）**：$\{\epsilon_t\}$ 满足 $E[\epsilon_t] = 0$、$\mathrm{Var}(\epsilon_t) = \sigma^2$、$\mathrm{Cov}(\epsilon_t, \epsilon_s) = 0$（$t \ne s$）——**不相关、零均值、常方差**。

**ACF**：$\gamma(0) = \sigma^2$，$\gamma(\tau) = 0$（$\tau \ne 0$）——单峰。**谱**：$S(\omega) = \sigma^2$ 平坦。<span class="marginnote">白噪声的「白」来自光谱比喻：<strong>白光由所有频率等强度组成，白噪声也如此——所有频率等功率</strong>。它是「无信息」的随机性基准，一切 ARMA 都从它开始。</span>

## 2 滑动平均 MA(q)

**MA(q)（moving average）**：
$$
X_t = \epsilon_t + \theta_1 \epsilon_{t-1} + \cdots + \theta_q \epsilon_{t-q}.
$$
**$X_t$ 是最近 $q+1$ 个噪声的加权和**——「对噪声的有限记忆」。

**性质**：
- **恒平稳**（无需条件）：MA 是噪声的有限线性组合，均值 0、方差有限，二阶矩恒定；
- **ACF 在 $q$ 截断**：$\gamma(\tau) = 0$ 对 $\tau > q$（噪声重叠只到 $q$ 步）；
- **谱**：$S(\omega) = \sigma^2 |1 + \theta_1 e^{-i\omega} + \cdots + \theta_q e^{-iq\omega}|^2$——多项式谱。

**例（MA(1)）**：$X_t = \epsilon_t + \theta\epsilon_{t-1}$。ACF：$\rho(1) = \frac{\theta}{1+\theta^2}$，$\rho(\tau) = 0$（$\tau \ge 2$）。**在滞后 1 处截断。**<span class="marginnote">MA 的「截断」是它最可爱的特征：<strong>看 ACF 从哪里突然归零，就是 MA 的阶数 $q$</strong>。但注意 MA(1) 的 $\rho(1)$ 与 $\theta$ 不是一一对应（$\theta$ 与 $1/\theta$ 给同 ACF）——这叫「可逆性」问题，建模时用「可逆条件」$|\theta| < 1$ 选解。</span>

## 3 自回归 AR(1)

**AR(1)（autoregressive）**：
$$
X_t = \phi X_{t-1} + \epsilon_t.
$$
**$X_t$ 依赖自己的过去 + 新噪声**——「回响」结构。

**平稳性条件**：$|\phi| < 1$。否则 $X_t$ 会「爆炸」（$|\phi| > 1$）或「随机游走」（$|\phi| = 1$，非平稳）。<span class="marginnote">「$|\phi| < 1$」为什么是平稳边界：<strong>$X_t = \sum_{k=0}^\infty \phi^k \epsilon_{t-k}$ 的方差 $\sum \phi^{2k}\sigma^2$ 当且仅当 $|\phi| < 1$ 收敛</strong>。$\phi = 1$ 时是随机游走（方差 $\to \infty$），$\phi > 1$ 指数爆炸。这就是「单位根」（unit root）概念——单位圆边界上的一切都危险。</span>

**平稳时的性质**：
- **均值**：$E[X_t] = 0$（无截距时）；
- **方差**：$\gamma(0) = \frac{\sigma^2}{1 - \phi^2}$；
- **ACF**：$\rho(\tau) = \phi^{|\tau|}$——**指数衰减，永不截断**；
- **谱**：$S(\omega) = \frac{\sigma^2}{1 - 2\phi\cos\omega + \phi^2}$——低频为主。

## 4 公式解析：AR(1) 的 ACF 指数衰减

**目标：证明 AR(1) 的 $\rho(\tau) = \phi^{|\tau|}$，走完「递推解法」的标准动作。**

第一步，写自协方差递推。$X_t = \phi X_{t-1} + \epsilon_t$，两边乘 $X_{t-\tau}$（$\tau \ge 1$）取期望：
$$
\gamma(\tau) = \phi\, \gamma(\tau - 1) + \mathrm{Cov}(\epsilon_t, X_{t-\tau}).
$$
第二步，消交叉项。$X_{t-\tau}$ 只依赖 $\epsilon_{t-\tau}, \epsilon_{t-\tau-1}, \dots$，与 $\epsilon_t$ 独立（$\tau \ge 1$），故 $\mathrm{Cov} = 0$：
$$
\gamma(\tau) = \phi\, \gamma(\tau - 1), \qquad \tau \ge 1.
$$
第三步，解递推。$\gamma(\tau) = \phi^\tau \gamma(0)$，除以 $\gamma(0)$：
$$
\rho(\tau) = \phi^\tau, \qquad \tau \ge 0.
$$
第四步，对称性给负滞后。$\rho(-\tau) = \rho(\tau)$，故 $\rho(\tau) = \phi^{|\tau|}$。

**这个推导为什么重要**：它示范了「从模型递推 ACF」的核心动作——**两边乘滞后项取期望、用独立性消交叉项、解递推**。这个动作对 AR(p)、ARMA 全适用，是时间序列理论的基本功。

## 5 ARMA 统一视角

**ARMA(p, q)**：$X_t = \phi_1 X_{t-1} + \cdots + \phi_p X_{t-p} + \epsilon_t + \theta_1\epsilon_{t-1} + \cdots + \theta_q\epsilon_{t-q}$。

用「滞后算子 $L$」$LX_t = X_{t-1}$ 可以写成：
$$
\Phi(L)\, X_t = \Theta(L)\, \epsilon_t,
$$
其中 $\Phi$、$\Theta$ 是滞后多项式。**ARMA 的平稳性 ⇔ $\Phi$ 的根都在单位圆外**——与 AR(1) 的 $|\phi| < 1$ 同源。<span class="marginnote">「滞后算子」让 ARMA 全部理论变得优雅：<strong>平稳性看 $\Phi$ 的根、可逆性看 $\Theta$ 的根、谱是 $|\Theta/\Phi|^2$</strong>。学到这里，ARMA 的谱与 ACF 都可以从两个多项式直接读出，不再需要逐条背公式。</span>

## 6 小结

- **白噪声**：不相关、零均值、常方差；ACF 单峰、谱平坦。
- **MA(q)**：噪声有限记忆，**恒平稳**，ACF 在 $q$ 截断。
- **AR(1)**：过去回响，平稳条件 $|\phi| < 1$，ACF 指数衰减 $\phi^{|\tau|}$、谱低频为主。
- **递推法**：乘滞后项取期望、消交叉项、解递推——算 ACF 的通用动作。
- **ARMA**：$\Phi(L)X = \Theta(L)\epsilon$；平稳性 = $\Phi$ 根在单位圆外。

**AR(1) 的两种视角**：自回归表示 $X_t = \phi X_{t-1} + \epsilon_t$ 可以迭代展开成无限 MA：
$$
X_t = \epsilon_t + \phi\epsilon_{t-1} + \phi^2\epsilon_{t-2} + \cdots = \sum_{k\ge0}\phi^k \epsilon_{t-k}.
$$
**AR 是「无限记忆的 MA」**——它的记忆来自几何衰减的噪声加权，永远不「切断」，这解释了为什么 AR 的 ACF 指数衰减而不截断。展开式同时给出平稳条件：几何和 $\sum\phi^{2k}$ 收敛当且仅当 $|\phi| < 1$，方差 $\gamma(0) = \sigma^2/(1-\phi^2)$ 与之自动一致。另外 $\phi$ 的符号决定谱形状：$\phi > 0$ 谱在零频最高（平滑），$\phi < 0$ 谱在高频最高（振荡）——**看样本谱的低频占比能快速判断 AR 系数的符号**。

与之对照，MA(1) 的 ACF 在滞后 1 处截断，而 AR(1) 永不截断——「截断 vs 衰减」正是区分 MA 与 AR 的黄金法则：看到 ACF 突然归零想 MA，看到指数尾巴想 AR。这个判别在 Box-Jenkins 定阶里是第一步，配合 PACF（偏自相关，AR 在 $p$ 截断）就能把 ARMA 的阶数完整读出来。

在下一节，我们把 ARMA 家族放进频率域的终极框架：**平稳过程的谱分解初步**——每个平稳过程都能拆成「不同频率的正弦波」。
