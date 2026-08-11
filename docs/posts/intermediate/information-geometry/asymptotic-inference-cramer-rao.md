---
title: 渐近推断与 Cramér-Rao 界
date: 2026-08-11
---

# 渐近推断与 Cramér-Rao 界

<div class="epigraph">
<p>估计的精度，本质上由统计流形上的曲率决定——Fisher 度量的倒数，就是一切无偏估计的精度天花板。</p>
<footer>—— 甘利俊一（Shun-ichi Amari）</footer>
</div>

<div class="article-byline">
<p>第二级 · 进阶数理 · 信息几何 ｜ 对标教材 ｜ 2026-08-11</p>
</div>

## 为什么从渐近推断开始

有了流形、度量、联络、投影，现在信息几何要回答统计学最经典的问题：**给定 $n$ 个样本，参数能被估计得多准？** 经典答案由 Cramér-Rao 不等式给出——它用 Fisher 信息的倒数划定任何无偏估计量的方差下界。信息几何把这条定理「几何化」：$g^{-1}$ 是度量的逆，Cramér-Rao 界是说**任何估计量的方差张量都不可能比切空间里的「度量逆」更小**。<span class="marginnote">Cramér-Rao 不等式（1946）是数理统计的里程碑之一。信息几何视角的核心贡献在于：不仅给出界，还解释了「谁能达到这个界」——答案是测地线方向上的估计量（有效估计），且一阶意义下极大似然估计总是达到界。</span>

本篇目标：严格陈述 Cramér-Rao 界并做几何翻译，解释渐近最优性与 Fisher 有效，最后指出二阶效应——曲率如何修正一阶结论。

## 1 Cramér-Rao 不等式

设 $X_1, \dots, X_n$ 独立同分布，来自 $p_\theta(x)$，$\theta \in \mathbb{R}^d$。记 $\hat\theta(X)$ 为 $\theta$ 的一个无偏估计（$\mathbb{E}[\hat\theta] = \theta$）。

**Cramér-Rao 不等式：** 在正则条件下，$\hat\theta$ 的协方差矩阵满足

$$\mathrm{Cov}(\hat\theta) \;\succeq\; \frac{1}{n}\, g^{-1}(\theta)$$

其中 $\succeq$ 表示「矩阵差半正定」，$g$ 是 Fisher 信息矩阵。<span class="marginnote">这个界对「无偏」这一要求极其敏感：若有偏，下界会多出一项偏差的平方项。所以很多现代方法会主动允许微小偏差以换取方差的大幅下降——这在高维统计里已是常态。</span>

几何翻译：$g^{-1}$ 是度量张量的逆，即**切空间上「对偶坐标」下的度量**。不等式说：无论你怎么构造估计量，它的误差椭球（由协方差刻画）都不可能缩进 Fisher 度量定义的椭圆之内。**流形的曲率决定了估计精度的上限。**

## 2 公式解析：Cramér-Rao 界的证明

以标量情形（$d=1$）演示，核心公式：

$$
\mathrm{Var}(\hat\theta) \;\ge\; \frac{1}{n\, g(\theta)}, \qquad g(\theta) = \mathbb{E}\left[\left(\frac{\partial \log L(\theta)}{\partial \theta}\right)^2\right]
$$

其中 $L$ 是似然。三步拆解：

- **第一步，用无偏性连接期望**：无偏意味着 $\int \hat\theta(x) \prod_i p_\theta(x_i)\, dx = \theta$。两边对 $\theta$ 求导，交换求导与积分（正则条件），得 $\int \hat\theta(x)\, \sum_i l_\theta(x_i) \cdot \prod p_\theta \, dx = 1$，即 $\mathrm{Cov}(\hat\theta, \sum_i l_i) = 1$。
- **第二步，施以 Cauchy-Schwarz**：协方差满足 $|\mathrm{Cov}(\hat\theta, S)|^2 \le \mathrm{Var}(\hat\theta)\,\mathrm{Var}(S)$，其中 $S = \sum_i l_i$ 是得分。由独立性 $\mathrm{Var}(S) = n g(\theta)$，代入得 $\mathrm{Var}(\hat\theta) \ge 1/(n g)$。
- **第三步，读出等号条件**：等号成立当且仅当 $\hat\theta - \theta$ 与 $S$ 线性相关（几乎处处），即 $\hat\theta = \theta + c S$。这正是「有效估计」的刻画。

直觉：估计的误差必须与得分函数充分「同步」——数据里携带参数信息的方式只有得分函数这一种通道，**估计量要拿信息，就要与得分高度相关；而得分只有一个方向的方差 $ng$，所以误差下界是 $1/(ng)$。**

## 3 渐近有效：极大似然达到界

Cramér-Rao 界是对有限样本成立的下界，但一般估计量未必达到。真正漂亮的结论在渐近（$n \to \infty$）意义下：

**极大似然估计（MLE）的渐近正态性：** 在正则条件下，

$$\sqrt{n}(\hat\theta_{MLE} - \theta) \;\xrightarrow{d}\; \mathcal{N}(0,\; g^{-1}(\theta))$$

也就是说，MLE 的渐近协方差恰好是 Cramér-Rao 下界 $g^{-1}$——**MLE 是渐近有效（asymptotically efficient）的**：在大样本下，没有任何无偏估计能比它更好。

几何上，MLE 的渐近行为被流形的一阶结构完全决定：它沿着「数据经验分布向模型子流形的 m-投影方向」波动，波动幅度由 Fisher 度量逆刻画。<span class="marginnote">这是信息几何看待经典渐近理论的方式：一阶渐近性质是「纯度量」的——只涉及 $g$，不涉及曲率。二阶渐近性质（下一节和第 10 篇）才开始涉及 α-联络与曲率张量。所以「一阶看度量、二阶看曲率」是理解整个理论的分水岭。</span>

## 4 二阶修正：曲率的登场

一阶理论完全由 $g$ 决定，对一切模型「同构」。但两个分布族即使 Fisher 度量相同，二阶渐近性质也可以不同——差异正来自联络与曲率。

**二阶 Edgeworth 展开**给出 MLE 分布更精细的逼近，其中出现由 α-联络、曲率张量编码的偏差项（尤其当 $\alpha \ne 0$ 时）：

$$p_{\sqrt{n}(\hat\theta-\theta)}(y) \approx \phi(y; 0, g^{-1}) \times \left[ 1 + \frac{1}{\sqrt{n}}\, h(y; \Gamma^{(\alpha)}, R) + O(1/n) \right]$$

偏差函数 $h$ 显式依赖联络系数与曲率。这意味着：

- **所有一阶有效估计量在二阶意义下不再等价**——有的偏斜更小、收敛更快。
- 最佳二阶估计量对应于「在 α-测地线方向修正」的估计，几何上就是**挑选合适的 α**。<span class="marginnote">这引向 Amari 著名的二阶渐近理论：二阶最优估计量存在于「沿指数族 m-投影」的一族估计之中，且可通过选取合适的 α 参数改进。细节见第 10 篇《高阶渐近推断理论》。</span>
- 贝叶斯后验均值、收缩估计等修正项，都能在 Edgeworth 框架下给出几何解释。

## 5 总结这张图

把本篇放回整条线索：

- **估计问题 = 投影问题**：数据经验分布向模型流形投影，Pythagoras 定理给出最优性。
- **精度上限 = 度量逆**：Cramér-Rao 界 $g^{-1}$ 是切空间里的精度天花板。
- **渐近最优 = 沿测地方向**：MLE 渐近达到界，因为它的波动集中在 Fisher 度量的最敏感方向。
- **超越一阶 = 需要曲率**：二阶修正涉及 α-联络与曲率张量，模型之间的差异在此浮现。

## 6 小结

- **Cramér-Rao 界**：无偏估计的协方差 $\succeq n^{-1}g^{-1}$，证明依赖 Cauchy-Schwarz 与得分函数的方差。
- 等号条件：估计量须与得分线性相关——这是**有效估计**的刻画。
- **MLE 渐近有效**：$\sqrt{n}(\hat\theta - \theta) \to \mathcal{N}(0, g^{-1})$，一阶性质完全由度量决定。
- **二阶性质由曲率决定**：Edgeworth 展开里的偏差项依赖 α-联络，不同模型在此分化。

在下一节，我们转向信息几何在机器学习里最直接的战场——**流形上的优化与自然梯度**：把 Fisher 度量装进梯度下降，让优化沿着流形最陡的真实方向前进。
