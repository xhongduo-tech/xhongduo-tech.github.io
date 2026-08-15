---
title: 多元正态分布及其性质
date: 2026-08-07
---

# 多元正态分布及其性质

<div class="epigraph">
<p>正态分布不是分布之一，而是分布之王——一旦我们想给一团数据云指定一个概率模型，它总是第一个被召唤的名字。</p>
<footer>—— 乔治·博克斯（George E. P. Box）</footer>
</div>

<div class="article-byline">
<p>第二级 · 多元统计分析 ｜ Anderson《An Introduction to Multivariate Statistical Analysis》Ch.2 · Johnson & Wichern Ch.4 ｜ 2026-08-07</p>
</div>

## 为什么从多元正态开始

上一节我们把数据写成了 $n \times p$ 矩阵，用 $\bar{\mathbf{x}}$ 与 $\mathbf{S}$ 描述了数据云的形状。但描述只是第一步——**要给这团云配一个概率模型，才能谈估计、检验与预测**。一元统计里正态分布是统治性的假设，多元统计同样如此：多元正态分布（multivariate normal distribution）几乎是一切经典多元方法的共同起点，正如均值向量与协方差矩阵完全决定了它，它也几乎完全由这两个量刻画。从它出发，Wishart 分布、Hotelling $T^2$ 检验、判别分析里的似然比，全都水到渠成。<span class="marginnote">「中心极限定理撑起一元正态」的直觉在多元同样成立：大量独立同分布的 $p$ 维随机向量之和，近似服从多元正态。这让多元正态成为「不知道选什么分布时的默认答案」。</span>

## 1 定义：先看密度函数

随机向量 $\mathbf{X} = (X_1, \ldots, X_p)'$ 服从均值为 $\boldsymbol{\mu}$、协方差为 $\boldsymbol{\Sigma}$（正定）的 $p$ 元正态分布，记作 $\mathbf{X} \sim \mathcal{N}_p(\boldsymbol{\mu}, \boldsymbol{\Sigma})$，若其密度为

$$
f(\mathbf{x}) = \frac{1}{(2\pi)^{p/2}|\boldsymbol{\Sigma}|^{1/2}} \exp\left[-\frac{1}{2}(\mathbf{x}-\boldsymbol{\mu})'\boldsymbol{\Sigma}^{-1}(\mathbf{x}-\boldsymbol{\mu})\right]
$$

对比一元情形 $f(x) = \frac{1}{\sqrt{2\pi}\sigma} e^{-(x-\mu)^2/2\sigma^2}$，结构完全平行：归一化常数 $(2\pi)^{p/2}|\boldsymbol{\Sigma}|^{1/2}$ 对应 $\sqrt{2\pi}\sigma$，指数里的二次型 $(\mathbf{x}-\boldsymbol{\mu})'\boldsymbol{\Sigma}^{-1}(\mathbf{x}-\boldsymbol{\mu})$ 对应 $(x-\mu)^2/\sigma^2$。<span class="marginnote">指数中的二次型叫<strong>马氏距离平方</strong> $d^2(\mathbf{x}) = (\mathbf{x}-\boldsymbol{\mu})'\boldsymbol{\Sigma}^{-1}(\mathbf{x}-\boldsymbol{\mu})$。它是「把协方差结构归一化之后」的距离：沿着高方差方向移动一段距离，代价比沿着低方差方向小。</span>

**等密度面是超椭球**：$(\mathbf{x}-\boldsymbol{\mu})'\boldsymbol{\Sigma}^{-1}(\mathbf{x}-\boldsymbol{\mu}) = c^2$ 定义了一个中心在 $\boldsymbol{\mu}$、主轴沿 $\boldsymbol{\Sigma}$ 特征向量的超椭球。$p = 2$ 时它就是我们在上一节说过的等高椭圆——数据云的形状，就是分布的密度形状。

## 2 关键性质：线性组合、边际与条件

多元正态之所以好用，是因为它在一族运算下保持封闭。设 $\mathbf{X} \sim \mathcal{N}_p(\boldsymbol{\mu}, \boldsymbol{\Sigma})$，则有：

**性质一（线性组合仍正态）**：对任意常数矩阵 $\mathbf{A}_{q \times p}$ 与向量 $\mathbf{b}$，

$$
\mathbf{A}\mathbf{X} + \mathbf{b} \sim \mathcal{N}_q(\mathbf{A}\boldsymbol{\mu} + \mathbf{b}, \ \mathbf{A}\boldsymbol{\Sigma}\mathbf{A}')
$$

分量求和、取子向量、算均值的均值——统统仍是正态。特别地，**每个分量 $X_j$ 是一元正态 $\mathcal{N}(\mu_j, \sigma_{jj})$**。<span class="marginnote">性质一还解释了一个常用技巧：线性判别函数 $a'\mathbf{X}$ 是正态的，于是判别分析里「两类用一条直线分开」有了严谨的概率语言。</span>

**性质二（不相关等价于独立）**：对正态向量，$\operatorname{Cov}(X_j, X_k) = 0$ 当且仅当 $X_j$ 与 $X_k$ 独立。这是一元统计没有的礼物——一般情形下不相关只是独立的必要条件，正态把它升格为充要条件。<span class="marginnote">独立性判定从此变得便宜：只需要扫一眼 $\boldsymbol{\Sigma}$ 的非对角线是否为零，而不用检查任意事件的乘积概率。这使多元正态里「独立」变成一个代数条件。</span>

**性质三（边际与条件都是正态）**：把 $\mathbf{X}$ 分成两块 $\mathbf{X} = (\mathbf{X}_1', \mathbf{X}_2')'$，对应地写

$$
\boldsymbol{\mu} = \begin{pmatrix}\boldsymbol{\mu}_1 \\ \boldsymbol{\mu}_2\end{pmatrix}, \qquad
\boldsymbol{\Sigma} = \begin{pmatrix}\boldsymbol{\Sigma}_{11} & \boldsymbol{\Sigma}_{12} \\ \boldsymbol{\Sigma}_{21} & \boldsymbol{\Sigma}_{22}\end{pmatrix}
$$

则边际分布 $\mathbf{X}_1 \sim \mathcal{N}_{p_1}(\boldsymbol{\mu}_1, \boldsymbol{\Sigma}_{11})$；给定 $\mathbf{X}_2 = \mathbf{x}_2$ 的条件分布仍正态，且

$$
E(\mathbf{X}_1 \mid \mathbf{X}_2 = \mathbf{x}_2) = \boldsymbol{\mu}_1 + \boldsymbol{\Sigma}_{12}\boldsymbol{\Sigma}_{22}^{-1}(\mathbf{x}_2 - \boldsymbol{\mu}_2)
$$

**条件期望是 $\mathbf{x}_2$ 的线性函数**——这正是后面多元线性回归的代数内核：回归函数就是多元正态下的条件期望。

## 3 二次型：从正态到 $\chi^2$ 的桥梁

**性质四（二次型分布）**：若 $\mathbf{X} \sim \mathcal{N}_p(\boldsymbol{\mu}, \boldsymbol{\Sigma})$，则马氏距离平方服从卡方分布：

$$
(\mathbf{X} - \boldsymbol{\mu})'\boldsymbol{\Sigma}^{-1}(\mathbf{X} - \boldsymbol{\mu}) \sim \chi^2_p
$$

直观验证：令 $\boldsymbol{\Sigma} = \mathbf{\Gamma}\boldsymbol{\Lambda}\boldsymbol{\Gamma}'$（谱分解），作变换 $\mathbf{Y} = \boldsymbol{\Lambda}^{-1/2}\boldsymbol{\Gamma}'(\mathbf{X}-\boldsymbol{\mu})$，则 $\mathbf{Y} \sim \mathcal{N}_p(\mathbf{0}, \mathbf{I}_p)$，而二次型恰好等于 $\mathbf{Y}'\mathbf{Y} = \sum_{j=1}^p Y_j^2$——$p$ 个独立标准正态的平方和，正是 $\chi^2_p$ 的定义。这条性质是**一切关于均值的检验**（Hotelling $T^2$）的种子：样本均值偏离 $\boldsymbol{\mu}$ 的「标准化距离」服从卡方，倒过来就是置信区域。

### 一元与多元正态的对照

| 一元 | 多元 |
| --- | --- |
| 参数 $\mu, \sigma^2$ | 参数 $\boldsymbol{\mu}, \boldsymbol{\Sigma}$ |
| 密度 $\frac{1}{\sqrt{2\pi}\sigma}e^{-(x-\mu)^2/2\sigma^2}$ | 密度 $\frac{1}{(2\pi)^{p/2}|\boldsymbol{\Sigma}|^{1/2}}e^{-\frac12(\mathbf{x}-\boldsymbol{\mu})'\boldsymbol{\Sigma}^{-1}(\mathbf{x}-\boldsymbol{\mu})}$ |
| 标准化 $z = (x-\mu)/\sigma$ | 马氏距离 $d = \sqrt{(\mathbf{x}-\boldsymbol{\mu})'\boldsymbol{\Sigma}^{-1}(\mathbf{x}-\boldsymbol{\mu})}$ |
| 独立性要求所有联合概率可分解 | 正态下不相关即独立 |
| $z^2 \sim \chi^2_1$ | $d^2 \sim \chi^2_p$ |

这张表把一元直觉逐格搬进多元：**一切「除以标准差」的标准化，在多元里都变成「乘 $\boldsymbol{\Sigma}^{-1/2}$ 的马氏变换」**。学多元正态，本质上是在学一套「把标量换成矩阵」的翻译规则。

## 4 样本均值的分布：连接正态与推断

有了多元正态，立刻可以回答抽样问题。设 $\mathbf{X}_1, \ldots, \mathbf{X}_n$ 独立同分布，$\mathbf{X}_i \sim \mathcal{N}_p(\boldsymbol{\mu}, \boldsymbol{\Sigma})$，则样本均值向量的分布是

$$
\bar{\mathbf{X}} = \frac{1}{n}\sum_{i=1}^n \mathbf{X}_i \sim \mathcal{N}_p\left(\boldsymbol{\mu},\ \frac{1}{n}\boldsymbol{\Sigma}\right)
$$

这是「正态可加」的直接推论：$\sum_i \mathbf{X}_i$ 是正态，均值不过是把协方差缩到 $1/n$。<span class="marginnote">注意这里的语义反转：$\bar{\mathbf{X}}$ 在抽样前是随机向量，在抽样后是上一节那个 $p$ 维重心。把「随机量的分布」与「一个具体取值」分开，是统计推断的头一课。</span>

更重要的是它与 $\mathbf{S}$ 的**独立性**：$\bar{\mathbf{X}}$ 与 $\mathbf{S}$ 相互独立。这看似反直觉（均值显然由样本算出，协方差矩阵也是），但在正态总体下严格成立——它是一元情形「样本均值与样本方差独立」（Basu 定理的特例）的多元推广。<span class="marginnote">正是这条独立性，让下一节能干净地把「均值的不确定性」和「协方差的估计误差」分开处理：$T^2$ 统计量把两者做成一个比值，就像一元 $t$ 统计量 $\frac{\bar{x}-\mu}{s/\sqrt{n}}$ 那样。</span>

于是检验「均值向量等于某个指定值」的原材料齐了：$\bar{\mathbf{X}}$ 告诉我们中心在哪，$\mathbf{S}$ 告诉我们散布多大，且两者独立——只差一个把二者组合成检验统计量的配方，这正是 Hotelling $T^2$ 检验要做的事。

## 5 公式解析：多元正态密度为什么长这样

把密度拆成三块，每块都有明确职责：

$$
f(\mathbf{x}) = \underbrace{\frac{1}{(2\pi)^{p/2}|\boldsymbol{\Sigma}|^{1/2}}}_{\text{归一化}} \cdot \exp\Bigl[-\underbrace{\frac{1}{2}(\mathbf{x}-\boldsymbol{\mu})'\boldsymbol{\Sigma}^{-1}(\mathbf{x}-\boldsymbol{\mu})}_{\text{马氏距离平方 } d^2/2}\Bigr]
$$

- **第一步，看指数里的二次型**：$d^2 = (\mathbf{x}-\boldsymbol{\mu})'\boldsymbol{\Sigma}^{-1}(\mathbf{x}-\boldsymbol{\mu})$ 是 $\mathbf{x}$ 到中心 $\boldsymbol{\mu}$ 的**马氏距离**。$\boldsymbol{\Sigma}^{-1}$ 起「去相关 + 去量纲」的作用：它把椭球等密度面转回球面，距离才与概率直接挂钩。
- **第二步，看归一化常数**：$\int f = 1$ 要求前面的常数恰好是 $(2\pi)^{p/2}|\boldsymbol{\Sigma}|^{1/2}$。$|\boldsymbol{\Sigma}|^{1/2}$ 正是坐标变换的雅可比行列式——把标准正态「拉伸」成椭球需要乘的体积因子。
- **第三步，验证等高面**：令 $d^2 = c^2$，得到超椭球 $\{\mathbf{x}: (\mathbf{x}-\boldsymbol{\mu})'\boldsymbol{\Sigma}^{-1}(\mathbf{x}-\boldsymbol{\mu}) = c^2\}$。若 $\boldsymbol{\Sigma} = \sigma^2 \mathbf{I}$（球面情形），它退化为普通球面，说明此时各方向概率对称。

**核心结论：多元正态只有两个参数——中心 $\boldsymbol{\mu}$ 与形状 $\boldsymbol{\Sigma}$**，其余一切（边际、条件、二次型）都由这两个量派生。这也是为什么上一节费那么大力气把 $\bar{\mathbf{x}}$、$\mathbf{S}$ 估计好：它们就是要喂给 $\mathcal{N}_p(\boldsymbol{\mu}, \boldsymbol{\Sigma})$ 的原材料。

## 6 小结

- **多元正态** $\mathcal{N}_p(\boldsymbol{\mu}, \boldsymbol{\Sigma})$ 由密度定义：等密度面是以 $\boldsymbol{\mu}$ 为中心、沿 $\boldsymbol{\Sigma}$ 特征向量定向的**超椭球**。
- **线性组合仍正态**：$\mathbf{A}\mathbf{X}+\mathbf{b} \sim \mathcal{N}_p(\mathbf{A}\boldsymbol{\mu}+\mathbf{b},\ \mathbf{A}\boldsymbol{\Sigma}\mathbf{A}')$；分量边际是一元正态。
- **不相关 ⇔ 独立**：正态向量独有的性质，让独立性变成「看协方差矩阵非对角线」的代数检查。
- **边际、条件都正态**：条件期望 $E(\mathbf{X}_1 \mid \mathbf{X}_2) = \boldsymbol{\mu}_1 + \boldsymbol{\Sigma}_{12}\boldsymbol{\Sigma}_{22}^{-1}(\mathbf{x}_2-\boldsymbol{\mu}_2)$ 是线性回归的代数内核。
- **二次型服从卡方**：$(\mathbf{X}-\boldsymbol{\mu})'\boldsymbol{\Sigma}^{-1}(\mathbf{X}-\boldsymbol{\mu}) \sim \chi^2_p$，一切均值检验与置信区域的种子。

在下一节，我们将从一个样本出发，把「总体多元正态」推进到「样本推断」：样本协方差矩阵 $\mathbf{S}$ 的分布（Wishart 分布），以及检验均值向量的 Hotelling $T^2$