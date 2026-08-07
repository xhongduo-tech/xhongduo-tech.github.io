---
title: n 维正态分布及其性质
date: 2026-08-07
---

# n 维正态分布及其性质

<div class="epigraph">
<p>多元正态是多维世界的钟形山丘——均值向量定中心，协方差矩阵定形状，所有方向上的投影依旧正态。</p>
<footer>—— 改编自哈拉尔德 · 克拉默（Harald Cramér）《统计学的数学方法》</footer>
</div>

<div class="article-byline">
<p>第二级 · 概率论与数理统计 ｜ 盛骤《概率论与数理统计》§4.3 ｜ 2026-08-07</p>
</div>

## 为什么从 n 维正态分布开始

第三章的二维正态只有 5 个参数；现在把它推广到 $n$ 维：一个均值向量 $\boldsymbol{\mu}$ 加一个协方差矩阵 $\boldsymbol{\Sigma}$，就完全决定了整个分布的形状。**n 维正态分布（multivariate normal distribution）**是「多元世界」最优雅、也最常用的分布——它是回归分析、主成分分析、高斯混合模型、卡尔曼滤波的共同地基，也是统计推断里「近似正态」假设的 n 维版本。

n 维正态之所以被反复使用，是因为它拥有一组极其漂亮的「封闭性」：线性变换仍正态、边缘仍正态、条件仍正态、不相关即独立。这些性质让高维问题能不断「降维拆解」而不失去正态身份——这是其他分布家族做梦都想要的特权。<span class="marginnote">n 维正态记号 $N(\boldsymbol{\mu}, \boldsymbol{\Sigma})$：均值是 $n$ 维向量，协方差是 $n \times n$ 对称半正定矩阵。当 $\boldsymbol{\Sigma}$ 是对角矩阵时，各分量独立（因为交叉协方差为零）——「对角协方差 = 独立」在多元正态里是充要条件，这是二维特例的直接推广。</span>

## 1 n 维正态的密度与记号

设 $\boldsymbol{X} = (X_1, \ldots, X_n)^\top$，均值向量 $\boldsymbol{\mu} = (\mu_1, \ldots, \mu_n)^\top$，协方差矩阵 $\boldsymbol{\Sigma}$（正定），则 $\boldsymbol{X}$ 的联合密度为

$$f(\boldsymbol{x}) = \frac{1}{(2\pi)^{n/2}\,|\boldsymbol{\Sigma}|^{1/2}} \exp\left\{-\frac12 (\boldsymbol{x} - \boldsymbol{\mu})^\top \boldsymbol{\Sigma}^{-1} (\boldsymbol{x} - \boldsymbol{\mu})\right\}$$

其中 $|\boldsymbol{\Sigma}|$ 是 $\boldsymbol{\Sigma}$ 的行列式，$(\boldsymbol{x}-\boldsymbol{\mu})^\top\boldsymbol{\Sigma}^{-1}(\boldsymbol{x}-\boldsymbol{\mu})$ 是**马氏距离（Mahalanobis distance）的平方**。<span class="marginnote">密度里的二次型 $(\boldsymbol{x}-\boldsymbol{\mu})^\top\boldsymbol{\Sigma}^{-1}(\boldsymbol{x}-\boldsymbol{\mu})$ 是「考虑了相关性之后到中心的距离」——马氏距离。若 $\boldsymbol{\Sigma} = \sigma^2 I$（各分量独立同方差），它退化为普通欧氏距离除以 $\sigma^2$。等值面是「马氏球」，在原始坐标里是倾斜的椭球。</span>

当 $n = 2$ 时，这个公式展开就是第三章的二维正态密度——五个参数 $\mu_1, \mu_2, \sigma_1^2, \sigma_2^2, \rho$ 全被吸收进 $\boldsymbol{\mu}$ 与 $\boldsymbol{\Sigma}$。

## 2 三大封闭性

n 维正态的核心性质是三个「封闭性」，它们让高维正态可以被任意变换而不变形：

**性质 1（线性变换仍正态）**：设 $\boldsymbol{A}$ 是 $m \times n$ 常数矩阵，$\boldsymbol{b}$ 是 $m$ 维常向量，若 $\boldsymbol{X} \sim N(\boldsymbol{\mu}, \boldsymbol{\Sigma})$，则

$$\boldsymbol{A}\boldsymbol{X} + \boldsymbol{b} \sim N(\boldsymbol{A}\boldsymbol{\mu} + \boldsymbol{b},\ \boldsymbol{A}\boldsymbol{\Sigma}\boldsymbol{A}^\top)$$

**任何线性组合仍正态**——分量和、分量差、任意投影都保持正态身份。

**性质 2（边缘仍正态）**：任一分量或分量子集仍服从（低维）正态分布，边缘参数直接从 $\boldsymbol{\mu}$、$\boldsymbol{\Sigma}$ 对应位置读出。<span class="marginnote">性质 2 是「线性变换封闭性」的直接推论：取 $\boldsymbol{A}$ 为「挑选某些分量」的矩阵，边缘分布就是对应子向量。这解释了为什么回归分析里「只看 $Y$」时它仍正态——正态在降维下不丢身份。</span>

**性质 3（条件仍正态）**：把 $\boldsymbol{X}$ 分成两段 $(\boldsymbol{X}_1, \boldsymbol{X}_2)$，则给定 $\boldsymbol{X}_2 = \boldsymbol{x}_2$ 时，$\boldsymbol{X}_1$ 的条件分布仍是正态，条件均值是 $\boldsymbol{x}_2$ 的**线性函数**，条件方差是常数矩阵。二维时它就是第三章性质 3 的推广。

## 3 不相关 ⇔ 独立（多元推广）

在 n 维正态中，第三章的关键结论原样成立：

$$\boldsymbol{X} \text{ 的各分量不相关} \iff \boldsymbol{\Sigma} \text{ 为对角矩阵} \iff \text{各分量相互独立}$$

**证明要点**：$\boldsymbol{\Sigma}$ 为对角矩阵时，密度中的二次型 $\sum_i \frac{(x_i - \mu_i)^2}{\sigma_i^2}$ 无交叉项，指数可分离，密度 = 各边缘密度之积。<span class="marginnote">这是「不相关 ⇔ 独立」在多元正态中的完整推广：协方差矩阵对角化即独立。做数据科学时，若假设数据来自多元正态，则「特征两两不相关」等价于「特征完全独立」——这个等价在非正态假设下不成立，务必只在正态语境下使用。</span>

**例**：$\boldsymbol{X} \sim N\left(\begin{pmatrix}1\\2\end{pmatrix}, \begin{pmatrix}4 & 0 \\ 0 & 9\end{pmatrix}\right)$。协方差矩阵对角 ⇒ $X_1, X_2$ 独立，且 $X_1 \sim N(1,4)$、$X_2 \sim N(2,9)$。

## 4 公式解析：协方差矩阵如何「定制形状」

协方差矩阵 $\boldsymbol{\Sigma}$ 对多元正态形状的掌控力，用一个二维特例看清：

$$

\boldsymbol{\Sigma} = \begin{pmatrix} \sigma_1^2 & \rho\sigma_1\sigma_2 \\ \rho\sigma_1\sigma_2 & \sigma_2^2 \end{pmatrix}

$$

- **第一步，对角元定轴长**：$\sigma_1^2, \sigma_2^2$ 决定椭球沿坐标轴的「胖瘦」。
- **第二步，非对角元定倾角**：$\rho\sigma_1\sigma_2$ 决定椭球的倾斜——$\rho > 0$ 时椭圆沿「东北—西南」拉长，$\rho < 0$ 沿「西北—东南」拉长。
- **第三步，特征分解视角**：$\boldsymbol{\Sigma}$ 的特征向量给出椭球的主轴方向，特征值是沿主轴的方差。$\rho = 0$ 时特征向量与坐标轴对齐，椭圆直立。

「协方差矩阵 = 椭球的几何参数」是多元正态最重要的直觉——**读懂了 $\boldsymbol{\Sigma}$，就读懂了分布的形状**。这也预告了主成分分析（PCA）的本质：找出协方差矩阵的主轴。

## 5 n 维正态的深入应用与实例

多元正态是多元统计的发动机，它的计算技巧、应用场景与常见误用值得展开。

### 例：边缘与条件分布的计算

**例**：$\boldsymbol X = (X_1, X_2) \sim N\left(\begin{pmatrix}1\\2\end{pmatrix}, \begin{pmatrix}4 & 2 \\ 2 & 3\end{pmatrix}\right)$。

- **边缘**：$X_1 \sim N(1, 4)$，$X_2 \sim N(2, 3)$——直接从 $\boldsymbol\mu$、$\boldsymbol\Sigma$ 读。
- **条件**：$X_1 \mid X_2 = x_2 \sim N\left(1 + \frac{2}{3}(x_2 - 2),\ 4 - \frac{4}{3}\right)$——条件均值线性、条件方差常数。

### 马氏距离与判别分析

密度里的二次型 $(\boldsymbol x - \boldsymbol\mu)^\top \boldsymbol\Sigma^{-1}(\boldsymbol x - \boldsymbol\mu)$ 是**马氏距离的平方**——「考虑了相关性后的距离」。线性判别分析（LDA）正是用「到各类的马氏距离」分类样本——**「哪个类的马氏距离近，就判给哪个类」**。

### n 维正态的三条封闭性

| 性质 | 结论 |
| --- | --- |
| 线性变换 | $\boldsymbol A\boldsymbol X + \boldsymbol b$ 仍正态 |
| 边缘 | 子向量仍正态 |
| 条件 | 条件分布仍正态 |

**「封闭性让多元正态可任意切割重组而不变形」**——这是它在统计与机器学习里被反复使用的原因。

### 例：协方差矩阵对角化与独立

**例**：$\boldsymbol\Sigma$ 为对角矩阵 ⇔ 各分量独立（多元正态特例）。若 $\boldsymbol\Sigma = \begin{pmatrix} 2 & 0 \\ 0 & 3 \end{pmatrix}$，则 $X_1 \perp X_2$——「不相关 = 独立」在多元正态里成立。

### 与 PCA 的联系

主成分分析（PCA）对 $\boldsymbol\Sigma$ 做特征分解：特征向量是主轴、特征值是沿轴方差——**「PCA 就是找多元正态等高线椭球的主轴」**。降维 = 保留大特征值方向。

**易错点｜辨析：** ① 「边缘正态」推不出「联合正态」——两个边缘正态的联合可能是非正态；② 多元正态要求 $\boldsymbol\Sigma$ 正定（或半正定），奇异矩阵时密度不存在（退化分布）；③ 「不相关 = 独立」只在多元正态成立，推广到其他分布是错误。

再补一张「多元 vs 一元正态」对照：

| | 一元正态 | 多元正态 |
| --- | --- | --- |
| 参数 | $\mu, \sigma^2$ | $\boldsymbol\mu, \boldsymbol\Sigma$ |
| 密度 | 标量二次型 | 马氏二次型 |
| 独立判据 | 不相关 | $\boldsymbol\Sigma$ 对角 |
| 封闭性 | 线性组合 | 线性变换 |

「多元是元的一维推广」——概念同构、矩阵登场。

## 6 小结

- **n 维正态** $N(\boldsymbol{\mu}, \boldsymbol{\Sigma})$：均值向量 + 协方差矩阵（对称半正定），密度含马氏距离二次型。
- **三大封闭性**：线性变换仍正态、边缘仍正态、条件仍正态（条件均值线性）。
- **不相关 ⇔ 独立** ⇔ 协方差矩阵对角——多元正态中三者等价。
- $\boldsymbol{\Sigma}$ 特征分解给出椭球主轴与沿轴方差，是 PCA 的数学根源。
- 多元正态是回归、PCA、GMM、卡尔曼滤波的公共地基。

在下一节，我们离开「单个随机变量」的世界，进入极限律——**大数定律：依概率收敛**，看大量重复观测如何把概率「挤」成确定。
