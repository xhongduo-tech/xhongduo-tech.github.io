---
title: 联合微分熵与条件微分熵
date: 2026-08-07
---

# 联合微分熵与条件微分熵

<div class="epigraph">
<p>多维的熵仍然是积分的诗：联合是整体，条件是在一维上的切面。</p>
<footer>—— 托马斯 · 科弗（Thomas M. Cover）</footer>
</div>

<div class="article-byline">
<p>第二级 · 信息论 ｜ Cover &amp; Thomas《Elements of Information Theory》 §8.2 ｜ 2026-08-07</p>
</div>

## 为什么从「单变量到多变量」开始

单变量的微分熵 $h(X)$ 只度量一个随机量的「形状」。但信息论里的主角几乎都是成对出现的：$X$ 与 $Y$ 联合、$X$ 给定 $Y$ 的条件……互信息 $I(X;Y) = h(X) - h(X|Y)$ 就建立在这些量之上。

所以这一篇把微分熵推广到多变量：**联合微分熵** $h(X, Y)$ 与**条件微分熵** $h(X \mid Y)$，并给出它们在连续世界的链式法则。

好消息是：形式上与离散世界**完全平行**——链式法则 $h(X,Y) = h(X) + h(Y|X)$ 原样成立。

坏消息是：**「条件作用使熵减小」这条离散铁律，在连续世界失效**——$h(X|Y) \le h(X)$ 不一定成立。这条「同形不同命」是理解连续信息论的又一道分水岭。<span class="marginnote">Cover &amp; Thomas §8.2 定义联合与条件微分熵。全部定义与离散版本的「积分配对」就是它们的形式；但第 40 篇的「负熵」阴影让所有「不等式」都要重新审视——不是每个离散结论都能原样搬到连续世界。</span>

## 1 联合微分熵的定义

**联合微分熵（joint differential entropy）**：对二维连续随机向量 $(X, Y)$ 与联合密度 $f(x, y)$，

$$
h(X, Y) = -\iint f(x, y) \log f(x, y) \, dx \, dy
$$

**条件微分熵（conditional differential entropy）**：

$$
h(X \mid Y) = -\iint f(x, y) \log f(x \mid y) \, dx \, dy
$$

**条件微分熵的展开形式**：

$$
h(X \mid Y) = \int f(y)\, h(X \mid Y = y) \, dy
$$

其中 $h(X \mid Y = y) = -\int f(x \mid y) \log f(x \mid y) dx$ 是「给定 $Y = y$」时的条件微分熵。<span class="marginnote">条件微分熵是「先对每个 $y$ 算条件熵、再按 $f(y)$ 加权平均」——这与离散条件熵 $H(X|Y) = \sum p(y) H(X|Y=y)$ 的结构一模一样，只是求和变积分。形式的平行在连续世界保持得很好。</span>

## 2 链式法则：连续版

**链式法则**：对任意连续随机向量，

$$
h(X, Y) = h(X) + h(Y \mid X)
$$

**证明**（与离散完全平行）：

$$
h(X,Y) = -\iint f(x,y)\log f(x,y)\,dx\,dy
$$

把 $f(x,y)$ 拆成 $f(x)f(y|x)$，对数变和：

$$
= -\iint f(x,y)\log f(x)\,dx\,dy - \iint f(x,y)\log f(y|x)\,dx\,dy
$$

第一项：$\int f(x)\log f(x) dx = h(X)$；第二项按定义是 $h(Y|X)$。✔

**推广**：$h(X_1, \dots, X_n) = \sum_{i=1}^n h(X_i \mid X_{1:i-1})$——多变量的逐条件展开，与离散链式法则一一对应。<span class="marginnote">链式法则在连续与离散两个世界「长得一模一样」，因为它只依赖「联合 = 边际 × 条件」与对数的可加性——这两个性质在密度与概率上同样成立。所以凡是「纯链式」的结论，都可以放心搬到连续世界。</span>

## 3 公式解析：多维高斯的联合微分熵

最重要的例子：$n$ 维高斯分布 $\mathcal{N}(\boldsymbol{\mu}, \mathbf{K})$（$\mathbf{K}$ 为协方差矩阵）。

$$
h(\mathcal{N}(\boldsymbol{\mu}, \mathbf{K})) = \frac12 \log\big((2\pi e)^n \det \mathbf{K}\big)
$$

**推导要点**：

- 高斯密度 $f(\mathbf{x}) = \frac{1}{(2\pi)^{n/2}(\det\mathbf{K})^{1/2}} \exp\big(-\frac12(\mathbf{x}-\boldsymbol{\mu})^\top \mathbf{K}^{-1}(\mathbf{x}-\boldsymbol{\mu})\big)$。
- $-\log f(\mathbf{x}) = \frac12\log\big((2\pi)^n \det\mathbf{K}\big) + \frac12 (\mathbf{x}-\boldsymbol{\mu})^\top\mathbf{K}^{-1}(\mathbf{x}-\boldsymbol{\mu})$。
- 取期望：$\mathbb{E}[(\mathbf{X}-\boldsymbol{\mu})^\top \mathbf{K}^{-1}(\mathbf{X}-\boldsymbol{\mu})] = n$（二次型的迹技巧）。
- 于是 $h = \frac12\log\big((2\pi)^n\det\mathbf{K}\big) + \frac{n}{2} = \frac12\log\big((2\pi e)^n \det\mathbf{K}\big)$。<span class="marginnote">「$\mathbb{E}[\text{二次型}] = n$」这一步用到了 $\mathbf{K}^{-1}$ 与 $\mathbf{K}$ 的迹配对：$\mathbb{E}[(\mathbf{X}-\mu)^\top\mathbf{K}^{-1}(\mathbf{X}-\mu)] = \text{tr}(\mathbf{K}^{-1}\mathbf{K}) = n$。它是多维高斯微分熵推导里唯一的「技巧点」，也是高斯熵公式与协方差行列式直接挂钩的来历。</span>

**解读**：

- $\det\mathbf{K}$ 度量「总体扩散」——方差的行列式越大，熵越大。
- 独立高斯（$\mathbf{K}$ 对角）时，$h = \sum_i h(X_i)$——链式法则的直观版。
- 相关（非对角）不增加熵：**相关性不创造信息**，这与离散世界一致。

**辨析｜易错点：** 两个陷阱：

- **$\det\mathbf{K} = 0$（奇异协方差）时公式发散**：退化高斯（变量有确定性线性关系）的密度不是真密度，微分熵 $-\infty$——分布「坍缩」到低维流形上，形状度量失效。
- **相关不增熵**：$h(X_1, X_2) \le h(X_1) + h(X_2)$，等号当且仅当独立——但注意这是「联合熵 ≤ 各自熵之和」，与离散的「$H(X,Y) \le H(X) + H(Y)$」一致。真正需要警惕的是下一条。

## 4 连续世界的「铁律失效」

**关键结论**：条件作用使熵减小在连续世界**不成立**：

$$
h(X \mid Y) \le h(X) \quad \text{不一定成立！}
$$

**反例**：设 $X \sim \mathcal{N}(0, 1)$，$Y = X$（完全确定关系）。则 $h(X) = \frac12\log(2\pi e) \approx 1.42$ 比特，而 $h(X \mid Y) = h(X \mid Y=y) = 0$——这里不等式成立。换个构造：

设 $X$ 是「单位方差高斯与尖峰高斯的混合」，$Y$ 指示「来自哪个成分」。给定 $Y$ 后 $X$ 的分布可能比无条件的 $X$ 更集中或更分散——由于 $h$ 可为负，条件熵完全可能大于边际熵。

**根源**：离散的 $H(X|Y) \le H(X)$ 靠的是「熵非负 + 凸性」；连续情形 $h$ 可为负，这些机制全部失效。<span class="marginnote">这是第 40 篇「负熵」的直接后果：离散世界里「知道 $Y$ 不会让 $X$ 更不确定」这条直觉，在连续世界被「$h$ 可为负」击穿。物理学家很早就知道这一点——连续系统的「熵」本来就需要参考尺度，条件后的「尺度」可以不同。</span>

**但互信息依旧健康**：虽然 $h(X|Y) \le h(X)$ 不成立，但

$$
I(X;Y) = h(X) - h(X\mid Y) \ge 0
$$

**恒成立**。互信息（差）不依赖参考尺度，非负性与「信息量」含义在连续世界完整保留。这再次印证第 40 篇的判断：**连续世界里，$h$ 是零件，$I$ 与 $D$ 才是成品。**

**与全课程体系的连接：** 联合/条件微分熵在《概率论与数理统计》对应「多维分布的熵」；在《机器学习》里对应「潜变量模型的证据下界」——ELBO 拆成「重构项 + 正则项」时用的正是 $h$ 的链式法则。多维高斯的 $h$ 公式则是「高斯过程、变分自编码器」等模型里频繁出现的常数。

## 5 小结

- **联合微分熵**：$h(X,Y) = -\iint f\log f$；**条件微分熵**：$h(X|Y) = \int f(y) h(X|Y=y) dy$。
- **链式法则**：$h(X,Y) = h(X) + h(Y|X)$，多变量逐条件展开——与离散完全平行。
- **多维高斯**：$h = \frac12\log\big((2\pi e)^n \det\mathbf{K}\big)$，$\det\mathbf{K}$ 度量总体扩散。
- **铁律失效**：$h(X|Y) \le h(X)$ 在连续世界不成立（$h$ 可为负）；但互信息 $I \ge 0$ 依旧恒真。
- 辨析：奇异协方差发散；相关不增熵；条件熵可大于边际熵。
- 连续世界里 $h$ 是零件、$I$ 与 $D$ 是成品——互信息才是不依赖参考尺度的信息量。

在下一篇，我们定义连续世界的相对熵与互信息：**相对熵与互信息的连续形式**——$D(f\|g)$ 与 $I(X;Y)$ 在密度语言下的完整形态。
