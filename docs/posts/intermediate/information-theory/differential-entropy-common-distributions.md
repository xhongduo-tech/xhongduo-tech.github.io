---
title: 均匀分布、指数分布与高斯分布的微分熵
date: 2026-08-07
---

# 均匀分布、指数分布与高斯分布的微分熵

<div class="epigraph">
<p>三个经典分布，三块微分熵的样板——量纲与形状，在公式里一目了然。</p>
<footer>—— 托马斯 · 科弗（Thomas M. Cover）</footer>
</div>

<div class="article-byline">
<p>第二级 · 信息论 ｜ Cover &amp; Thomas《Elements of Information Theory》 §8.1 ｜ 2026-08-07</p>
</div>

## 为什么从「把公式背成直觉」开始

微分熵的定义 $h(X) = -\int f \log f$ 是抽象的，但落到具体分布上，它的公式会揭示出微妙的规律——**微分熵总是「某个尺度参数的对数」**：

- 均匀分布：$h = \log(\text{区间长度})$
- 指数分布：$h = \log\frac{e}{\lambda}$
- 高斯分布：$h = \frac12 \log(2\pi e \sigma^2)$

每个公式里都只有「对数 + 尺度参数」。这个共同形态不是巧合：**微分熵度量的是「分布扩散到多宽」**，而宽度恰恰由尺度参数控制。

这一篇我们把三个经典分布的微分熵都算一遍、提炼「尺度对数的规律」、并解释它为什么是后续最大熵定理（第 44 篇）与高斯信道容量（第八篇）的原料。<span class="marginnote">这三个公式是信息论与统计学的「公共词汇」：通信里算容量、统计里算 Fisher 信息、机器学习里算先验的复杂度，都要用它们。值得像背「勾股定理」一样背下来。</span>

## 1 均匀分布：$h = \log(b - a)$

**均匀分布** $X \sim \text{Unif}(a, b)$：$f(x) = \frac{1}{b-a}$，支撑区间 $[a, b]$。

**推导**：

$$
h = -\int_a^b \frac{1}{b-a} \log\frac{1}{b-a} \, dx = \log(b - a)
$$

**解读**：

- 区间越宽，微分熵越大——分布越「散」，密度越「稀」。
- $b - a < 1$ 时 $h < 0$：密度 $> 1$，形状集中。
- 均匀分布是「给定支撑区间下熵最大」的分布（熵的上限形状）——第 44 篇的种子。

**辨析｜易错点**：均匀分布的微分熵只依赖区间长度 $b-a$，**不依赖区间位置** $a$。平移不变——这是「形状度量」的直接体现：$[0,1]$ 与 $[5,6]$ 的均匀分布熵相同。<span class="marginnote">「平移不变、缩放变」是微分熵变换律的雏形：$h(X + c) = h(X)$，但 $h(cX) = h(X) + \log|c|$。均匀分布的公式里没有 $a$、只有 $b-a$，正是这条规律的实例。</span>

## 2 指数分布：$h = \log\frac{e}{\lambda}$

**指数分布** $X \sim \text{Exp}(\lambda)$：$f(x) = \lambda e^{-\lambda x}$（$x \ge 0$）。

**推导**：

$$
h = -\int_0^\infty \lambda e^{-\lambda x} \log(\lambda e^{-\lambda x}) \, dx = -\log\lambda + \frac{1}{\lambda} \cdot \lambda \cdot \int_0^\infty \lambda e^{-\lambda x} x \, dx
$$

利用 $\mathbb{E}[X] = \frac{1}{\lambda}$：

$$
h = -\log\lambda + \frac{\lambda \cdot \mathbb{E}[X]}{\ln 2} \cdot (\text{底}) \quad \Rightarrow \quad h = 1 - \log\lambda \ \ (\text{以 } e \text{ 为底})
$$

以 2 为底写作 $h = \log\frac{e}{\lambda}$ 比特。

**解读**：$\lambda$ 越大（分布越陡、衰减越快），熵越小——指数分布把概率挤在原点附近，形状更集中。

**性质**：指数分布是「给定均值、非负实数上熵最大」的分布——与第 44 篇最大熵定理呼应，也是统计物理里玻尔兹曼分布的雏形。<span class="marginnote">指数分布的「最大熵」身份值得记住：在「非负 + 固定均值」约束下，最不武断的分布就是指数分布。这个结论在排队论（服务时间）、可靠性（寿命）与生存分析里反复出现——凡是「只知道平均寿命」的场景，指数分布是默认选择。</span>

## 3 高斯分布：$h = \frac12 \log(2\pi e \sigma^2)$

**高斯分布** $X \sim \mathcal{N}(\mu, \sigma^2)$：$f(x) = \frac{1}{\sqrt{2\pi\sigma^2}} e^{-\frac{(x-\mu)^2}{2\sigma^2}}$。

**推导**：

$$
h = -\mathbb{E}\Big[\log f(X)\Big] = \mathbb{E}\Big[\frac12\log(2\pi\sigma^2) + \frac{(X-\mu)^2}{2\sigma^2}\Big]
$$

利用 $\mathbb{E}[(X-\mu)^2] = \sigma^2$：

$$
h = \frac12\log(2\pi\sigma^2) + \frac12 \cdot \frac{1}{\ln 2} \cdot ? \quad\Rightarrow\quad h = \frac12 \log(2\pi e \sigma^2)
$$

（以 $e$ 为底时 $h = \frac12 \ln(2\pi e \sigma^2)$；以 2 为底加 $\log_2 e$ 的换算。）

**解读**：

- $\sigma$ 越大（分布越宽），熵越大；$\mu$ 不出现——**高斯微分熵与均值无关**。
- $\sigma^2 < \frac{1}{2\pi e}$ 时 $h < 0$——尖峰高斯可以有负微分熵。
- **最重要**：高斯是「给定方差下熵最大」的分布（第 44 篇的最大微分熵定理），所以 $h \le \frac12\log(2\pi e \sigma^2)$ 对**任何**方差为 $\sigma^2$ 的分布成立。<span class="marginnote">「高斯熵最大」是第八篇高斯信道容量的直接原料：加性高斯噪声下的信道，最优输入分布就是高斯（给定功率约束时），容量公式 $C = \frac12\log(1 + \text{SNR})$ 正是从「高斯熵最大」推出来的。一条线索贯穿两篇。</span>

## 4 核心对比表：尺度规律一目了然

把三个公式并排，提炼共同规律。

| 分布 | 密度 $f(x)$ | 尺度参数 | 微分熵 $h$ | 熵随尺度 |
| --- | --- | --- | --- | --- |
| 均匀 $\text{Unif}(a,b)$ | $\frac{1}{b-a}$ | 区间长 $b-a$ | $\log(b-a)$ | 尺度 ×2 → 熵 +1 |
| 指数 $\text{Exp}(\lambda)$ | $\lambda e^{-\lambda x}$ | 尺度 $\frac{1}{\lambda}$ | $\log\frac{e}{\lambda}$ | 尺度 ×2 → 熵 +1 |
| 高斯 $\mathcal{N}(\mu,\sigma^2)$ | $\frac{1}{\sqrt{2\pi\sigma^2}}e^{-\frac{(x-\mu)^2}{2\sigma^2}}$ | 标准差 $\sigma$ | $\frac12\log(2\pi e\sigma^2)$ | 尺度 ×2 → 熵 +1 |

**共同规律**：**尺度参数乘 $c$，微分熵加 $\log c$**。

均匀：$b-a \to 2(b-a)$ ⇒ $h \to h + 1$ 比特。
指数：$\frac1\lambda \to \frac{2}{\lambda}$ ⇒ $h \to h + 1$。
高斯：$\sigma \to 2\sigma$ ⇒ $h \to h + 1$。

**这个规律正是「变换律」$h(cX) = h(X) + \log|c|$ 的实例**——第 43 篇将把它一般化。<span class="marginnote">「尺度翻倍、熵加一比特」的直觉：把分布拉宽一倍，定位精度减半，相当于损失 1 比特的「分辨率信息」。这再次说明微分熵是「相对于参考尺度的形状度量」——尺度一变，形状的「刻度」就变。</span>

## 5 小结

- 均匀：$h = \log(b-a)$；指数：$h = \log\frac{e}{\lambda}$；高斯：$h = \frac12\log(2\pi e\sigma^2)$。
- 三个公式都只含「对数 + 尺度参数」，共同规律是**尺度 ×$c$ ⇒ 熵 +$\log c$**。
- 高斯/指数/均匀各自是「给定某约束（方差/均值/支撑）下熵最大」的分布——最大熵定理的伏笔。
- 平移不变、缩放变：均值/位置不影响微分熵，宽度决定一切。
- 尖峰分布（$\sigma$ 小、$\lambda$ 大、区间窄）可以有负微分熵。
- 这些公式是高斯信道容量、Fisher 信息、最大熵定理的公共原料。

在下一篇，我们直面微分熵最反直觉的性格：**微分熵与离散熵的区别：可为负值的原因**——为什么连续情形能「熵为负」，这在离散世界里意味着什么。
