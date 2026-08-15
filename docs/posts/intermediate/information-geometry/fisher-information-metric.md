---
title: Fisher 信息度量
date: 2026-08-07
---

# Fisher 信息度量

<div class="epigraph">
<p>Fisher 信息是统计学里最接近「距离」概念的东西——它回答：两个分布到底相差多远？</p>
<footer>—— 引自 Rao（C. R. Rao），1945 年提出 Fisher 度量</footer>
</div>

<div class="article-byline">
<p>第二级 · 信息几何 ｜ Amari &amp; Nagaoka《Methods of Information Geometry》Ch. 2 ｜ 2026-08-07</p>
</div>

## 为什么从 Fisher 信息度量开始

上一节我们建立了统计流形：分布是点，参数是坐标，得分函数张成切空间。但只有切空间还不够——流形上还缺少一把「尺子」，用来丈量两点之间的距离、向量之间的夹角、曲线的长度。这一节安装的就是这把尺子：**Fisher 信息度量**。它在统计里早已以「Fisher 信息矩阵」的形式出场——那是 Cramér-Rao 界里的分母；而在几何视角下，它摇身一变成为了流形上的黎曼度量。<span class="marginnote">C. R. Rao 在 1945 年的论文里首次指出 Fisher 信息矩阵满足黎曼度量的变换法则，奠定了「信息几何」的第一个基石。详见 Amari &amp; Nagaoka, "Methods of Information Geometry" Ch. 2。</span>

一句话总结本篇：**Fisher 信息矩阵就是切空间上的内积，它定义了一个自然的黎曼度量，使统计流形成黎曼流形。**

## 1 从敏感度到度量

上一节我们得到得分函数 $l_i = \partial_i \log p_\theta$，且知道 $\mathbb{E}[l_i] = 0$。均值是零，那下一步自然要问：**这组随机变量 $l_i$ 的方差 / 协方差是多少？**

Fisher 信息矩阵的定义正是这个协方差矩阵：

$$g_{ij}(\theta) = \mathbb{E}_{p_\theta}\left[ l_i l_j \right] = \mathbb{E}\left[ \frac{\partial \log p_\theta}{\partial \theta^i} \frac{\partial \log p_\theta}{\partial \theta^j} \right]$$

因为 $\mathbb{E}[l_i] = 0$，协方差 $\mathrm{Cov}(l_i, l_j)$ 就等于 $\mathbb{E}[l_i l_j]$，所以这确实是协方差矩阵。<span class="marginnote">Fisher 信息度量还有一个等价公式：$g_{ij} = -\mathbb{E}\left[\frac{\partial^2}{\partial \theta^i \partial \theta^j}\log p_\theta\right]$。两者在正则条件下相等，第二个形式在数值上往往更好算。我们会在公式解析里证明一次。</span>

**Fisher 信息度量（Fisher information metric）**：在统计流形 $\mathcal{S}$ 上，由上式 $g_{ij}(\theta)$ 给出的、随点平滑变化的正定矩阵，构成一个黎曼度量。切空间中两个向量 $A = \sum a^i l_i$ 与 $B = \sum b^i l_i$ 的内积定义为

$$\langle A, B \rangle_{p_\theta} = \sum_{i,j} a^i b^j g_{ij}(\theta)$$

由此，切向量的长度、曲线的弧长、两点间的测地线距离都有了定义。**分布之间的「距离」第一次有了精确的几何意义。**

## 2 变换法则：为什么它配叫度量

黎曼度量的本质要求是**坐标变换下不变**。设新坐标 $\xi = \xi(\theta)$，用链式法则可证：

$$g'_{ij}(\xi) = \sum_{k,l} \frac{\partial \theta^k}{\partial \xi^i} \frac{\partial \theta^l}{\partial \xi^j} g_{kl}(\theta)$$

这正是张量的协变变换法则——Fisher 信息在换参数时不会「变形」，只是换了一副坐标眼镜。这与朴素做法形成鲜明对比：

**辨析｜易错点：** 若把「分布间的距离」朴素地定义为参数差的欧氏距离 $d(\theta_1, \theta_2) = |\theta_1 - \theta_2|$，那么这个距离**依赖参数化**——把正态分布的 $(\mu, \sigma^2)$ 换成 $(\mu, \sigma)$，同一对分布算出的「距离」就变了。几何对象不该依赖坐标画法，这正是我们要黎曼度量而非欧氏直尺的原因。<span class="marginnote">这也是为什么「KL 散度对称化」「直接比较参数向量」等粗糙做法在统计推断里经常给出怪异结论——它们都忽略了流形的弯曲。Fisher 度量是唯一（在单调变换下）自然的选择，这一深刻事实由 Chentsov 定理刻画，见第 10 篇《高阶渐近推断理论》。</span>

## 3 例子：正态分布族的 Fisher 度量

对一维正态 $N(\mu, \sigma^2)$，直接计算得

$$g = \begin{pmatrix} 1/\sigma^2 & 0 \\ 0 & 2/\sigma^4 \end{pmatrix}$$

即 $g_{\mu\mu} = 1/\sigma^2$，$g_{\sigma^2\sigma^2} = 2/\sigma^4$，交叉项为零。

读这个矩阵：$\mu$ 方向上的「尺子」伸缩为 $1/\sigma$，$\sigma^2$ 方向伸缩为 $\sqrt{2}/\sigma^2$。**方差越大，$\mu$ 方向的尺子越短**——大方差下均值位置本来就难以精确分辨，几何上用「该方向的尺子变短」如实刻画了这种不确定性。这就是几何语言的力量：统计直觉被编码成度量的形状。

### 数值算例：Bernoulli 族的 Fisher 度量

把上一篇的 Bernoulli 例子接着算。得分函数为 $l = \frac{x}{\theta} - \frac{1-x}{1-\theta}$，于是

$$g(\theta) = \mathbb{E}[l^2] = \theta \cdot \frac{1}{\theta^2} + (1-\theta)\cdot \frac{1}{(1-\theta)^2} = \frac{1}{\theta(1-\theta)}$$

读这个结果：$\theta$ 越靠近 $0$ 或 $1$，度量越大、尺子越长。这与你对 Bernoulli 的直觉完全一致——成功概率接近边界时，少量样本就足以判断参数，因为「信息密度」在边界处暴涨。反过来，$\theta = 1/2$ 时度量最小，正是最不确定的位置。<span class="marginnote">同样的算例在最大似然推断里给出 $\mathrm{Var}(\hat\theta) \ge \theta(1-\theta)/n$——方差与度量成反比。这预告了第 7 篇的 Cramér-Rao 界：精度永远追着 Fisher 度量的形状走。</span>

### 用变换法则核对正态族

第 2 节的变换法则不是摆设——用它可以在不同参数化之间搬运度量。把正态族从 $(\mu, \sigma^2)$ 换成 $(\mu, \sigma)$：令 $s = \sigma^2$，则 $\frac{\partial s}{\partial \sigma} = 2\sigma$，于是 $g_{\sigma\sigma} = \frac{\partial s}{\partial \sigma}\, g_{ss}\, \frac{\partial s}{\partial \sigma} = (2\sigma)\cdot \frac{2}{\sigma^4} \cdot (2\sigma) = \frac{8}{\sigma^2}$。

两种写法度量分量不同、几何不变：**度量分量随坐标变，测地线距离不随坐标变**——这是「坐标无关」最直接的练习，也是「同一把尺子换一副刻度」的含义。

## 4 公式解析：二阶导数的等价形式

核心公式：

$$
g_{ij} = \mathbb{E}[\, \partial_i \log p_\theta \; \partial_j \log p_\theta \,] = -\,\mathbb{E}[\, \partial_i \partial_j \log p_\theta \,]
$$

分步拆解：

- **第一步，写出乘积**：$\partial_i \partial_j \log p_\theta = \partial_j\left(\dfrac{\partial_i p_\theta}{p_\theta}\right) = \dfrac{\partial_i\partial_j p_\theta}{p_\theta} - \dfrac{\partial_i p_\theta \, \partial_j p_\theta}{p_\theta^2}$。
- **第二步，对两项取期望**：第一项 $\mathbb{E}\left[\frac{\partial_i\partial_j p_\theta}{p_\theta}\right] = \int \partial_i\partial_j p_\theta \, dx = \partial_i\partial_j \int p_\theta dx = 0$（归一化条件）；第二项正是 $\mathbb{E}[l_i l_j] = g_{ij}$。
- **第三步，整理符号**：于是 $\mathbb{E}[\partial_i \partial_j \log p_\theta] = 0 - g_{ij} = -g_{ij}$，两边乘负号即得等价式。

直觉：得分函数的一阶矩是零（上一篇），二阶矩是 Fisher 信息；而对数似然的二阶导——即对数似然曲率的负期望——同样给出 Fisher 信息。两条路指向同一个量：**Fisher 信息度量度量的本质是「对数似然面的弯曲程度」**，弯曲越厉害，参数越容易被数据分辨。

## 5 Fisher 度量的三个用途

- **定义长度与测地线**：两个分布之间的内在距离 $\int \sqrt{\sum g_{ij} \dot\theta^i \dot\theta^j}\, dt$，为「分布有多远」给出坐标无关的答案。
- **重新推导 Cramér-Rao 界**：任何无偏估计量的协方差矩阵满足 $\mathrm{Cov}(\hat\theta) \succeq g^{-1}$。在几何语言里，$g^{-1}$ 就是度量张量的逆——统计推断的精度下限由流形的度量完全决定。这一层我们会在第 7 篇《渐近推断与 Cramér-Rao 界》展开。
- **孕育自然梯度**：沿 Fisher 度量定义的测地方向做梯度下降，就是在「统计流形最陡的方向」上优化——这正是深度学习里自然梯度法的几何源头，见第 8 篇。

### 为什么偏偏是 Fisher 度量

**一个容易踩的坑**：Fisher 度量在坐标变换下协变，但它不是「唯一的度量」——任何正定、随点光滑变化的矩阵族都能定义一个黎曼度量。Fisher 的特殊性在于它同时满足**单调性**（充分化简不增距离）与**坐标无关性**，Chentsov 定理证明这两个性质把度量钉死为 Fisher（差一个常数倍）——这是第 10 篇要展开的「唯一自然几何」。

**直觉核对**：把度量看成「局部放大的比例尺」。正态族里 $\sigma$ 方向的尺子长度 $\sqrt{2}/\sigma^2$ 随 $\sigma$ 增大而缩短——方差不确定时，对方差自身的判断也变难；几何再次如实翻译统计直觉。这个「方差不光影响均值、也影响对自身推断」的耦合，正是协方差矩阵对角元都随参数变化的原因。

把度量的三要素收束成三点：

- **对称性**：$g_{ij} = g_{ji}$，内积是交换的。
- **正定性**：$g \succ 0$，一切方向都有正长度。
- **非退化性**：$g$ 可逆，切空间与余切空间一一对应——这让 $g^{-1}$（Cramér-Rao 界的核心）永远有定义。

这些性质保证误差椭球、置信区域等几何对象在换参数时不「变形」，也让「精度」有了坐标无关的定义。

### 速查：Fisher 度量的三副面孔

| 形式 | 公式 | 读到什么 |
| --- | --- | --- |
| 得分二阶矩 | $g_{ij} = \mathbb{E}[\partial_i \log p \, \partial_j \log p]$ | 敏感度协方差 |
| 对数似然负二阶导 | $g_{ij} = -\mathbb{E}[\partial_i\partial_j \log p]$ | 似然面曲率 |
| 估计精度上限 | $\mathrm{Cov}(\hat\theta) \succeq g^{-1}/n$ | 精度由度量逆决定 |

三种写法指同一个量：**度量是「参数被数据分辨的难易程度」的精确刻度**。把 Fisher 度量与信息论联系起来还能获得更深的直觉：对位置参数 $\theta$，当 $X \sim p_\theta$ 叠加微小噪声时，熵增与 Fisher 信息由 de Bruijn 恒等式相连——「估计参数的难度」与「信息量的流动速率」共享同一刻度。**Fisher 度量既是几何对象，也是信息对象**；这条线索在信号处理与神经网络初始化里反复出现，我们会在第 8 篇自然梯度里再次见到它。

## 6 小结

- **Fisher 信息度量** $g_{ij} = \mathbb{E}[l_i l_j]$ 是切空间上的内积，把统计流形升级为黎曼流形。
- 它满足**张量变换法则**，坐标无关；而朴素的参数欧氏距离依赖参数化，不是几何对象。
- 等价公式 $g_{ij} = -\mathbb{E}[\partial_i\partial_j \log p_\theta]$，本质是**对数似然面的曲率**。
- 正态族 $N(\mu,\sigma^2)$ 的度量 $g = \mathrm{diag}(1/\sigma^2,\; 2/\sigma^4)$ 直观展示了「不确定性 = 尺子缩短」。

在下一节，我们将安装流形上的第二个结构——**联络**。度量告诉你距离，联络告诉你「如何把向量从一个点搬到另一个点」；而信息几何的特别之处在于，它同时存在两套联络，构成 α-联络与对偶结构。
