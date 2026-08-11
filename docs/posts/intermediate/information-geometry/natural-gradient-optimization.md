---
title: 流形上的优化与自然梯度
date: 2026-08-11
---

# 流形上的优化与自然梯度

<div class="epigraph">
<p>优化的每一步都应在流形的度量下测量——用 Fisher 度量校正的梯度，才是参数空间里最陡的方向。</p>
<footer>—— 甘利俊一（Shun-ichi Amari）</footer>
</div>

<div class="article-byline">
<p>第二级 · 进阶数理 · 信息几何 ｜ 对标教材 ｜ 2026-08-11</p>
</div>

## 为什么从自然梯度开始

当你在大模型时代训练神经网络时，用的优化器是 Adam、SGD、Shampoo……它们共享同一个底层假设：**参数空间是欧氏的**。但参数的变化最终要映射成分布的变化——而分布空间是弯曲的统计流形。自然梯度（natural gradient）是甘利俊一 1998 年提出的思想：**用 Fisher 度量代替欧氏度量来度量「参数变化有多大」，由此得到真正最陡的下降方向。** 近年它解释了神经网络的近似 Fisher、K-FAC、以及 Transformer 训练中的诸多现象。<span class="marginnote">Amari 在 1998 年《Natural Gradient Works Efficiently in Learning》一文中提出自然梯度，并在随后的「信息几何 + 神经网络」路线中系统发展。深度学习时代，K-FAC、Shampoo 等优化器都可看作它的近似实现。</span>

本篇目标：定义自然梯度、解释它为什么比普通梯度「更陡」，并展示它对指数族与神经网络的具体含义。

## 1 从欧氏梯度到自然梯度

标准梯度下降沿负梯度方向移动：

$$\theta^{(t+1)} = \theta^{(t)} - \eta\, \nabla L(\theta^{(t)})$$

这里 $\nabla L$ 的分量是在欧氏坐标里算的。但问题是：**参数空间里的「一步」到底意味着多大变化？** 在统计流形上，两个参数 $\theta$ 与 $\theta + d\theta$ 对应的分布差异由 Fisher 度量衡量：

$$d^2(p_\theta, p_{\theta+d\theta}) = \sum_{i,j} g_{ij}\, d\theta^i d\theta^j$$

**辨析｜易错点：** 普通梯度的步长 $\|\nabla L\|$ 用欧氏范数算，但**欧氏范数大的方向在统计上可能变化很小**（因为度量把该方向压缩了）。于是出现怪象：优化走了很远，分布却没怎么动。自然梯度的修正正是要撤销度量的这种「偏心」。

**自然梯度（natural gradient）：**

$$\tilde\nabla L(\theta) = g^{-1}(\theta)\, \nabla L(\theta)$$

即用 Fisher 度量逆去旋转普通梯度。它在「分布变化量」的意义上是**最陡下降方向**：在约束「分布变化量 $d$ 固定」的前提下，使损失下降最多的方向。

![欧氏梯度与自然梯度对比](/images/information-geometry/natural-gradient-1.svg)

图上：椭圆等高线代表「分布距离」的同心圆（Fisher 度量下的单位圆），普通梯度（黑色箭头）沿坐标轴方向，自然梯度（红色箭头）才真正垂直于等高线、直指极小。

## 2 公式解析：为什么 $g^{-1}$ 是正确旋转

设 $L$ 是参数 $\theta$ 上的损失，求约束优化：

$$
\min_{d\theta} \left\{ L(\theta + d\theta) \;\middle|\; \sum_{i,j} g_{ij}\, d\theta^i d\theta^j = \varepsilon^2 \right\}
$$

分三步：

- **第一步，线性化目标**：$L(\theta + d\theta) \approx L(\theta) + \nabla L(\theta) \cdot d\theta$。于是问题变成「在度量椭圆上，找使 $\nabla L \cdot d\theta$ 最负的方向」。
- **第二步，解约束极值（Lagrange 乘子）**：对 $d\theta$ 求导，得 $2\lambda\, g\, d\theta = -\nabla L$，即 $d\theta \propto -g^{-1} \nabla L$。**度量的逆自动扮演「校正器」的角色。**
- **第三步，读出结论**：$g^{-1}$ 把坐标间的相关性剔除，让「步长」按度量公平计量。当 $g = I$（欧氏）时自然梯度退化为普通梯度——所以自然梯度是普通梯度的**黎曼化推广**。

直觉：普通梯度回答「在坐标纸上往哪个方向走最陡」，自然梯度回答「在流形（真实现象）上往哪个方向走最陡」。后者才是优化真正想要的。

## 3 指数族上的闭式解

对指数族 $p_\theta$，Fisher 度量 $g_{ij} = \partial_i\partial_j\psi(\theta)$，且期望参数 $\eta = \partial\psi$。此时自然梯度有特别漂亮的等价形式：

$$\tilde\nabla L = \partial_\eta L$$

即**在期望参数坐标下的普通梯度，就是原坐标下的自然梯度**。因为 $d\eta = g\, d\theta$，而 $g^{-1}\partial_\theta = \partial_\eta$。<span class="marginnote">这个对偶关系再一次体现了 e/m 坐标的互补性：在 η 坐标（m-仿射坐标）里，Fisher 度量变成单位阵，自然梯度退化为普通梯度。所以「选对坐标」就能白拿自然梯度的好处。</span>

对神经网络，Fisher 度量近似为经验 Fisher 矩阵 $F = \frac{1}{N}\sum \nabla_\theta \log p(y|x)\, \nabla_\theta \log p(y|x)^\top$，自然梯度即 $\theta \leftarrow \theta - \eta F^{-1}\nabla L$。**F 矩阵就是网络的「信息几何度量」**——它度量参数变化对预测分布的影响，比欧氏度量诚实得多。

## 4 自然梯度的代价与近似

自然梯度的瓶颈一目了然：$F^{-1}$ 是 $d \times d$ 矩阵求逆，$d$ 可达百万级。于是有一族近似策略：

**对角近似**：只保留 $F$ 的对角线，退化为「逐参数自适应学习率」——这是 Adam、AdaGrad 的几何诠释。<span class="marginnote">AdaGrad 累加梯度的外积平方，恰是对 $F$ 对角元的在线估计。所以「自适应学习率优化器 ≈ 对角近似的自然梯度」这条线索，把深度学习优化与信息几何直接接通。</span>
- **K-FAC**：利用 Fisher 的块结构（分层 Kronecker 分解）做近似逆。
- **摊销逆 / 矩阵-free**：用迭代法求 $F^{-1}v$，如 Conjugate Gradient。

**辨析｜易错点：** 自然梯度不保证「更快到达最优」，它保证的是**参数轨迹的尺度不变性**——换一种参数化，自然梯度路径不变，而普通梯度的路径会变。对一阶（渐近）统计问题，自然梯度具有「统计效率」；但在非凸损失面上，它同样可能陷入局部极小而未必胜出普通 SGD。

## 5 小结

- 普通梯度的步长用**欧氏范数**计量，在弯曲流形上不可靠；**自然梯度** $\tilde\nabla L = g^{-1}\nabla L$ 用 Fisher 度量校正。
- 它在「分布变化量」意义下是最陡方向；$g = I$ 时退化为普通梯度。
- 指数族上自然梯度 = 期望参数坐标下的普通梯度，体现对偶坐标之美。
- 神经网络用经验 Fisher 矩阵近似，Adam/AdaGrad 可看作对角近似自然梯度。
- 代价是 $F^{-1}$ 的求解，催生了 K-FAC 等近似与尺度不变性的理论保证。

在下一节，我们绕回散度本身，系统地建立**散度理论**——f-散度、Bregman 散度如何统一 KL、Wasserstein 与各类统计距离，以及它们各自的几何角色。
