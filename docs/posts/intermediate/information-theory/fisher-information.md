---
title: Fisher 信息（Fisher Information）的定义与性质
date: 2026-08-07
---

# Fisher 信息（Fisher Information）的定义与性质

<div class="epigraph">
<p>似然函数的陡峭程度，就是数据对参数的敏感程度——Fisher 信息是「参数有多好被看出来」的度量。</p>
<footer>—— 罗纳德 · 费希尔（Ronald A. Fisher）</footer>
</div>

<div class="article-byline">
<p>第二级 · 信息论 ｜ Cover &amp; Thomas《Elements of Information Theory》 §11.10 ｜ 2026-08-07</p>
</div>

## 为什么从「统计估计的信息量」开始

我们一直用「信息」衡量「通信」与「压缩」。统计学家费希尔提出了另一条问题：**一条数据样本，对估计参数 $\theta$ 能提供多少信息？**

答案叫**Fisher 信息**：

$$
I(\theta) = \mathbb{E}\left[\left(\frac{\partial}{\partial \theta} \log f(X; \theta)\right)^2\right]
$$

**直觉**：$\frac{\partial}{\partial\theta}\log f(X;\theta)$ 是「对数似然的斜率」——数据对参数有多敏感。Fisher 信息 = 斜率的平方期望，度量「参数 $\theta$ 从数据里能被多清楚地读出来」。

- 似然函数「尖」（对 $\theta$ 敏感）：Fisher 信息大，参数好估。
- 似然函数「平」（对 $\theta$ 迟钝）：Fisher 信息小，参数难估。

Fisher 信息是整个统计估计理论的地基：它出现在 **Cramér-Rao 不等式**（下一篇）、最大似然估计的渐近方差、以及「信息几何」里。这一篇我们定义它、证明它的关键性质、并揭示它与信息论（相对熵、互信息）的深层联系。<span class="marginnote">Fisher 信息在 Cover &amp; Thomas §11.10 登场。它之所以叫「信息」，是因为它确实度量「数据含多少关于 $\theta$ 的信息」——与 Shannon 信息论虽是两个体系，却共享「信息」之名与「不确定性下界」之实。</span>

## 1 定义：两个等价形式

**定义一（得分平方）**：$s(x;\theta) = \frac{\partial}{\partial\theta}\log f(x;\theta)$ 称为**得分（score）**，Fisher 信息是它的二阶矩：

$$
I(\theta) = \mathbb{E}_\theta[s(X;\theta)^2]
$$

**定义二（似然曲率）**：在正则条件下（得分期望为零），Fisher 信息等于对数似然的负二阶矩：

$$
I(\theta) = -\mathbb{E}_\theta\left[\frac{\partial^2}{\partial\theta^2}\log f(X;\theta)\right]
$$

**为什么两个形式相等**：由 $\mathbb{E}[s] = 0$（对数似然求导的期望为零），

$$
\frac{\partial^2}{\partial\theta^2}\log f = \frac{f''}{f} - \left(\frac{f'}{f}\right)^2 = \frac{f''}{f} - s^2
$$

取期望，$\mathbb{E}[f''/f] = \int f'' dx = \frac{d^2}{d\theta^2}\int f dx = 0$，故 $-\mathbb{E}[\partial_\theta^2 \log f] = \mathbb{E}[s^2] = I(\theta)$。

**第二个形式给了「曲率」直觉**：$I(\theta)$ 是「对数似然在 $\theta$ 处的陡峭程度」——越陡，越容易区分「$\theta$ 和 $\theta + d\theta$」，参数越可辨识。<span class="marginnote">「Fisher 信息 = 对数似然曲率」是最常用的直觉：把 $\log f(x;\theta)$ 想成一座山的轮廓，$I(\theta)$ 是山脊的陡峭度。山越陡，$\theta$ 稍微一动数据分布就大变，估计越准。这也解释了「为什么 MLE 的方差反比于 Fisher 信息」。</span>

## 2 关键性质

**性质一（可加性）**：独立样本的 Fisher 信息相加：

$$
I_n(\theta) = n I(\theta)
$$

$n$ 个 i.i.d. 样本提供 $n$ 倍信息——「数据越多，参数越清楚」。这是大样本估计理论的基础。

**性质二（变换律）**：参数变换 $\eta = g(\theta)$ 时，

$$
I_\eta(\eta) = I_\theta(\theta) \left(\frac{d\theta}{d\eta}\right)^2
$$

Fisher 信息按变换的平方缩放——它依赖参数化方式（不是坐标无关的量）。

**性质三（与 KL 散度的局部联系）**：Fisher 信息是「相对熵在参数空间里的局部曲率」：

$$
D(f_\theta \| f_{\theta + \delta}) \approx \frac12 I(\theta) \delta^2
$$

**「$\theta$ 移到 $\theta+\delta$ 造成的 KL 距离，局部地由 $I(\theta)\delta^2/2$ 决定」**——Fisher 信息把信息论的距离与统计的参数空间连接起来。<span class="marginnote">「$D \approx \frac12 I \delta^2$」是信息论与统计最深的交汇之一：它让相对熵在参数空间上「局部二次化」，Fisher 信息成了「参数空间上的黎曼度量」。这个「信息几何」视角（Amari 学派）把统计推断看成「流形上的测地运动」——一个极其优美的现代框架。</span>

## 3 公式解析：$I(\theta)$ 的三个构成

把核心公式拆开：

$$
I(\theta) = \mathbb{E}\left[\left(\frac{\partial \log f(X;\theta)}{\partial \theta}\right)^2\right] = -\mathbb{E}\left[\frac{\partial^2 \log f(X;\theta)}{\partial \theta^2}\right]
$$

- **$\log f(X;\theta)$**：对数似然——「数据在参数 $\theta$ 下的合理性」。
- **$\partial_\theta \log f$（得分）**：对数似然对参数的斜率——「$\theta$ 动一点，数据分布动多少」。
- **平方 + 期望**：把「敏感度」的波动平均掉，得到「平均敏感度」。
- **负二阶导（曲率）**：对数似然函数的弯曲程度——越尖越陡，$I$ 越大。

**一个具体例子（高斯）**：$X \sim \mathcal{N}(\theta, \sigma^2)$，$\log f = -\frac{(x-\theta)^2}{2\sigma^2} - \frac12\log(2\pi\sigma^2)$。

- 二阶导：$\partial_\theta^2 \log f = -\frac{1}{\sigma^2}$（与 $x$ 无关）。
- $I(\theta) = \frac{1}{\sigma^2}$。

**直觉**：噪声方差 $\sigma^2$ 越大，参数越难估，Fisher 信息越小（$1/\sigma^2$）。「噪声大 → 信息少」——与信息论的直觉完全一致。<span class="marginnote">「高斯例子的 $I = 1/\sigma^2$」是最干净的示范：Fisher 信息 = 噪声方差的倒数。噪声越大，每个样本提供的参数信息越少。这与「高斯信道容量随噪声增大而减小」是同一句物理——参数估计里的「噪声税」，与通信里的「噪声税」同源。</span>

**辨析｜易错点：** 三个容易混的地方：

- **Fisher 信息依赖「参数化」**：$I(\theta)$ 不是坐标无关的（变换律带平方因子）。谈论「Fisher 信息」必须指明参数是什么。
- **$I(\theta)$ 是「数据的信息」，不是「参数的信息」**：它度量「数据能读出多少 $\theta$」，与「$\theta$ 本身的熵」无关。别把 Fisher 信息与 Shannon 互信息混淆——虽然它们有联系。
- **正则条件**：两个定义等价需要「得分期望为零」等光滑性假设；不满足时要用第一个定义。<span class="marginnote">「Fisher 信息依赖参数化」与「微分熵依赖坐标」是同族现象：它们都度量「在某参考系下的可辨识度」。真正的「坐标无关」量是它们的不变量（如互信息、以及 Fisher 信息诱导的黎曼度量）——这又是「绝对值主观、关系客观」的第三次现身。</span>

## 4 Fisher 信息与互信息的联系

Fisher 信息与 Shannon 互信息在「位置参数」的估计里直接相连：

**定理**：设 $X \sim f(x - \theta)$（$\theta$ 是位置参数），则

$$
I(\theta) \le I(X; \theta)
$$

**直觉**：互信息 $I(X;\theta)$ 是「观测 $X$ 关于 $\theta$ 的信息」——任何估计能提取的信息都不超过它。Fisher 信息是「得分」层面的局部信息，互信息是全局信息；Fisher 信息 ≤ 互信息反映了「局部敏感度 ≤ 总信息」这条信息守恒律。

**深层意义**：统计估计（Fisher）与信息传输（Shannon）在「位置参数」场景下共享同一套「信息」语言——**数据能告诉我们的关于参数的信息，与信道能传的信息，服从同一个上限逻辑。**<span class="marginnote">「$I(\theta) \le I(X;\theta)$」的证明用到「估计误差 vs 互信息」的 Fano 型不等式。它把两个「信息」焊在一起：Fisher 信息管「局部」，互信息管「全局」，两者在同一问题上大小有序。这条不等式是「信息论 × 统计」的枢纽之一。</span>

**与全课程体系的连接：** Fisher 信息在第二级《概率论与数理统计》里是「MLE 渐近性质」的核心；在第四级《机器学习》里是「自然梯度」「Fisher 信息矩阵」的定义来源（自然梯度 = 用 Fisher 信息做度量的梯度）；在《统计》里它是 Cramér-Rao 不等式与信息几何的起点。下一篇文章直接用它证明估计下界。

## 5 小结

- **Fisher 信息**：$I(\theta) = \mathbb{E}[(\partial_\theta \log f)^2] = -\mathbb{E}[\partial_\theta^2 \log f]$——得分的二阶矩 / 对数似然曲率。
- 性质：可加（$n$ 样本 = $nI$）、变换律（按平方缩放）、局部 KL 曲率（$D \approx \frac12 I\delta^2$）。
- 高斯例子：$I(\theta) = 1/\sigma^2$——噪声方差的倒数。
- **辨析**：依赖参数化；是「数据的信息」不是「参数的信息」；正则条件需满足。
- 与互信息：$I(\theta) \le I(X;\theta)$——局部信息 ≤ 全局信息。
- 它是 Cramér-Rao、MLE 渐近、自然梯度、信息几何的共同地基。

在下一篇，我们用它推出统计估计的「物理极限」：**Cramér-Rao 不等式与信息不等式**——无偏估计的方差不能低于 $1/I(\theta)$。
