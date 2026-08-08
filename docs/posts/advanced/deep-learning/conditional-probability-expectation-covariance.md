---
title: 条件概率、期望、方差与协方差
date: 2026-08-07
---

# 条件概率、期望、方差与协方差

<div class="epigraph">
<p>概率论是逻辑在不确定性之下的延伸。</p>
<footer>—— 埃德温 · 杰恩斯（E. T. Jaynes），《概率论：关于科学中的逻辑》</footer>
</div>

<div class="article-byline">
<p>第四级 · 深度学习 ｜ 花书《深度学习》§3.3–3.8 ｜ 2026-08-07</p>
</div>

## 为什么从条件概率与期望开始

上一节我们认识了分布——伯努利、高斯、指数族，它们是给世界建模的「原料」。但原料不能直接用来做机器学习：模型看到一张图，要回答「它是猫的概率是多少」；训练时，要把「模型输出」与「真实标签」之间的差距算成一个数。这两件事分别落在今天的三组工具上：**条件概率（conditional probability）** 在已知部分信息后重新校准概率，**期望（expectation）** 把分布压成一个数字，**方差（variance）** 与**协方差（covariance）** 量化不确定性与变量之间的相互影响。<span class="marginnote">花书把这三个概念排在分布之后、估计之前，构成「分布 → 运算 → 估计」的链条：这一节学运算，下一节《最大似然估计与贝叶斯统计》用它们做推断，而本专题后面几乎所有损失函数、正则化、优化分析都会回来取用它们。</span>

语言模型「下一个 token 是什么」的每一步，本质上都在计算条件概率 $\mathrm{P}(\text{token}_t \mid \text{前 } t-1 \text{ 个 token})$；回归任务里「预测误差有多大」的答案由方差给出；而「两个特征是否一起变化」则由协方差回答。这一节把这三个运算本身学透，是进入第二篇《机器学习基础》之前最后一块概率补料。

## 1 条件概率：已知信息如何改变概率

**条件概率（conditional probability）** 回答「在事件 $B$ 已经发生的条件下，$A$ 的概率」。定义式是

$$
\mathrm{P}(A \mid B) = \frac{\mathrm{P}(A \cap B)}{\mathrm{P}(B)}, \qquad \mathrm{P}(B) > 0
$$

读作「给定 $B$ 时 $A$ 的概率」。分母 $\mathrm{P}(B)$ 要求事件 $B$ 有正概率——「条件」必须是一个可能发生的事实。直觉上，条件概率是把样本空间从 $\Omega$ **收缩到 $B$**：原来我们在整个空间上量 $A$，现在只在 $B$ 的内部量「$A$ 且 $B$」占多少比例，再用 $\mathrm{P}(B)$ 做归一化，保证「给定 $B$ 时所有条件概率之和为 1」。

**辨析｜易错点：$\mathrm{P}(A \mid B)$ 与 $\mathrm{P}(A \cap B)$ 是两回事。** $\mathrm{P}(A \cap B)$ 是「两个都发生」在全集中的比例；$\mathrm{P}(A \mid B)$ 是「在 $B$ 发生的人里，$A$ 占多少」。同一个班里「戴眼镜且是男生」的人数占比，与「男生里戴眼镜」的比例，数字一般完全不同。入门者最容易把条件概率当成交事件的概率。

两个事件什么时候「互相不提供信息」？这就是**独立性（independence）**：

$$
\mathrm{P}(A \cap B) = \mathrm{P}(A)\,\mathrm{P}(B)
\quad \Longleftrightarrow \quad
\mathrm{P}(A \mid B) = \mathrm{P}(A)
$$

**独立意味着条件概率退化为无条件概率**——知道 $B$ 发生了，也不能给 $A$ 的判断提供任何增益。注意独立性是概率的乘法关系，不是「互斥」；互斥（$A \cap B = \emptyset$）与独立几乎是对立的。<span class="marginnote">独立不是「无关」的日常含义，而是一个精确的乘法条件。深度学习里最常见的独立假设是「样本独立同分布（i.i.d.）」：假设每个训练样本从同一分布独立抽出。这个假设让似然写成连乘，是最大似然估计的基石，但它几乎总是近似成立——样本之间有相关性的数据集（如同一用户的连续点击）要专门处理。</span>

条件概率的威力在于**贝叶斯公式**，它把「逆概率」用「正概率」算出来。设 $\{A_1, \dots, A_k\}$ 是对样本空间的一个划分（两两互斥、并为全集），则

$$
\mathrm{P}(A_i \mid B) = \frac{\mathrm{P}(B \mid A_i)\,\mathrm{P}(A_i)}{\sum_{j=1}^{k}\mathrm{P}(B \mid A_j)\,\mathrm{P}(A_j)}
$$

分子是「先验」$\mathrm{P}(A_i)$ 乘以「似然」$\mathrm{P}(B \mid A_i)$，分母是**全概率公式**对 $B$ 的边际概率。这条公式会在本节第 4 节专门拆解，也会是下一节《最大似然估计与贝叶斯统计》的主角。

## 2 期望：把分布压成一个数字

**期望（expectation / expected value）** 是随机变量按概率加权的平均。离散与连续各一条：

$$
\mathbb{E}[X] = \sum_{x} x\, \mathrm{P}(X=x), \qquad
\mathbb{E}[X] = \int x\, p(x)\, dx
$$

「期望」这个词容易误导——它不是「我们期望看到的值」，而是**长期平均**：独立重复试验次数足够多时，样本均值会逼近它（大数定律）。掷一枚公平硬币 $X \in \{0,1\}$，$\mathbb{E}[X] = 0.5$，但 0.5 永远不是一个「会出现的结果」——期望是对分布的压缩，不是对样本的预测。

期望最值钱的性质是**线性性**：对任意随机变量 $X, Y$ 与常数 $a, b$，

$$
\mathbb{E}[aX + bY + c] = a\,\mathbb{E}[X] + b\,\mathbb{E}[Y] + c
$$

**线性性不要求任何独立性。** 无论 $X$ 与 $Y$ 纠缠得多深，期望照加不误——这是概率论里最省心、也最容易被低估的一条性质。它在后面算「均方误差的期望」「梯度的期望」时反复出场。<span class="marginnote">期望还是「函数」的期望：$\mathbb{E}[g(X)] = \sum_x g(x)\mathrm{P}(x)$ 对离散成立（连续为积分），这条被称为「潜意识统计学家法则」（LOTUS）——我们不必先求 $g(X)$ 的分布，直接对 $g(x)$ 加权求和即可。</span>

## 3 方差与协方差：量化不确定性与相关性

期望给了分布的位置，**方差（variance）** 给出分布的**弥散程度**——离均值平均有多远。定义与最常用的展开式：

$$
\mathrm{Var}(X) = \mathbb{E}\big[(X - \mathbb{E}[X])^2\big]
= \mathbb{E}[X^2] - \big(\mathbb{E}[X]\big)^2
$$

第二个等号把方差变成「二次矩减均值平方」，在计算上常用。方差的算术平方根叫**标准差（standard deviation）** $\sigma = \sqrt{\mathrm{Var}(X)}$，与 $X$ 同量纲，便于解读。方差的两个基本运算律：

$$
\mathrm{Var}(aX + b) = a^2\,\mathrm{Var}(X), \qquad
\mathrm{Var}(X + Y) = \mathrm{Var}(X) + \mathrm{Var}(Y) + 2\,\mathrm{Cov}(X, Y)
$$

注意第一条里常数 $b$ 不贡献方差——平移不改变弥散；系数 $a$ 却要**平方**。第二条揭示：和的方差不是简单相加，中间还夹着一个**协方差（covariance）**：

$$
\mathrm{Cov}(X, Y) = \mathbb{E}\big[(X - \mathbb{E}[X])(Y - \mathbb{E}[Y])\big]
= \mathbb{E}[XY] - \mathbb{E}[X]\,\mathbb{E}[Y]
$$

协方差衡量 $X$ 与 $Y$「是否一起偏离各自的均值」：同向偏离为正，反向为负，互不关联则为 0。把协方差除以两个标准差就得到**相关系数（correlation coefficient）**

$$
\rho_{XY} = \frac{\mathrm{Cov}(X, Y)}{\sigma_X\, \sigma_Y} \in [-1, 1]
$$

它把协方差归一化到固定区间，于是「相关多强」不再依赖各自量纲——这是把不同特征的相关系数放一起比较的前提。

**辨析｜易错点：协方差为 0 不等于独立。** 独立必然推出协方差为 0，反过来不成立：协方差只捕捉**线性**共变，两个变量完全可以「强相关但非线性地相关」，如 $Y = X^2$ 且 $X$ 关于 0 对称分布时 $\mathrm{Cov}(X, Y) = \mathbb{E}[X^3] - \mathbb{E}[X]\mathbb{E}[X^2] = 0$，但 $Y$ 明明由 $X$ 完全决定。判别独立只能回到乘法定义 $\mathrm{P}(A\cap B)=\mathrm{P}(A)\mathrm{P}(B)$。

把 $n$ 个随机变量两两之间的协方差排成矩阵，就得到**协方差矩阵（covariance matrix）**

$$
\boldsymbol{\Sigma} = \begin{pmatrix}
\mathrm{Var}(X_1) & \mathrm{Cov}(X_1, X_2) & \cdots & \mathrm{Cov}(X_1, X_n)\\
\mathrm{Cov}(X_2, X_1) & \mathrm{Var}(X_2) & \cdots & \mathrm{Cov}(X_2, X_n)\\
\vdots & \vdots & \ddots & \vdots\\
\mathrm{Cov}(X_n, X_1) & \mathrm{Cov}(X_n, X_2) & \cdots & \mathrm{Var}(X_n)
\end{pmatrix}
$$

对角线是方差、非对角线是协方差，且 $\boldsymbol{\Sigma}$ 是**对称半正定矩阵**——这条性质把概率论与第二级《线性代数》的特征分解直接接上：主成分分析（PCA）就是在找协方差矩阵最大的几个特征方向。<span class="marginnote">在多元高斯分布里，协方差矩阵 $\boldsymbol{\Sigma}$ 就是那个「胖瘦与朝向」的旋钮：特征向量指示分布拉长的方向，特征值指示各方向上的弥散大小。深度学习里大量用到高斯先验/高斯噪声假设，协方差矩阵的几何直觉能省去很多死记硬背。</span>

用一个小实验把「期望、方差、协方差」在样本上兑现：

```python
import numpy as np

rng = np.random.default_rng(0)
X = rng.normal(0, 1, 1000)               # 1000 个样本
Y = 2 * X + rng.normal(0, 0.34, 1000)    # Y = 2X + 小噪声

print("期望 E[X]      ≈", X.mean())                # ≈ 0
print("方差 Var(X)    ≈", X.var())                 # ≈ 1
print("协方差 Cov     ≈", np.cov(X, Y)[0, 1])      # ≈ 2
print("相关系数 ρ     ≈", np.corrcoef(X, Y)[0, 1]) # ≈ 0.986
```

$Y = 2X + \text{噪声}$ 里，噪声把相关系数从 1 拉到 0.986——**相关系数越接近 ±1，线性关系越强；不是 ±1，说明有噪声或非线性**。

## 4 公式解析：贝叶斯公式

把贝叶斯公式逐项拆开，是理解「先验、似然、后验、证据」四位演员的最好方式：

$$
\mathrm{P}(A \mid B) = \frac{\mathrm{P}(B \mid A)\,\mathrm{P}(A)}{\mathrm{P}(B)}
$$

- **第一步，从条件概率定义出发**。$\mathrm{P}(A\mid B) = \mathrm{P}(A\cap B)/\mathrm{P}(B)$。这是定义，没有争议：把样本空间收缩到 $B$，看 $A$ 占多少。
- **第二步，交换视角求分子**。$\mathrm{P}(A \cap B)$ 既等于 $\mathrm{P}(B\mid A)\mathrm{P}(A)$，也等于 $\mathrm{P}(A\mid B)\mathrm{P}(B)$——**交集只有一个，只是从谁的视角看**。从 $A$ 的视角写，就把「未知的条件概率」换成了「可能已知的 $\mathrm{P}(B\mid A)$ 与 $\mathrm{P}(A)$」。
- **第三步，理解分母 $\mathrm{P}(B)$**。它是 $B$ 的边际概率，由全概率公式 $\mathrm{P}(B) = \sum_j \mathrm{P}(B\mid A_j)\mathrm{P}(A_j)$ 展开——把所有「产生 $B$ 的途径」都算一遍。分母的作用是**归一化**：保证左端对 $A$ 求和后为 1。
- **第四步，读出贝叶斯哲学的骨架**。把四样东西排开：**后验** $\mathrm{P}(A\mid B)$（看到数据后的信念）、**似然** $\mathrm{P}(B\mid A)$（假设 $A$ 为真时数据有多可能）、**先验** $\mathrm{P}(A)$（看数据前的信念）、**证据** $\mathrm{P}(B)$（数据本身的稀有度）。合起来就是一句话：**后验 ∝ 似然 × 先验**。

这套语言在下一节《最大似然估计与贝叶斯统计》里会成为主角：频率派只最大化似然（认为先验无足轻重），贝叶斯派则把先验认真摆上桌。而在分类问题里，「给定特征，类别概率」正是 $\mathrm{P}(\text{类别} \mid \text{特征})$——贝叶斯公式是理解朴素贝叶斯、以及后续一切生成式模型的门把手。

## 5 小结

- **条件概率** $\mathrm{P}(A\mid B) = \mathrm{P}(A\cap B)/\mathrm{P}(B)$：把样本空间收缩到 $B$ 后重估概率；**独立** $\mathrm{P}(A\cap B)=\mathrm{P}(A)\mathrm{P}(B)$ 时条件概率退化为无条件概率。
- **期望** $\mathbb{E}[X]$ 是概率加权平均（离散求和、连续积分），具备**线性性**且无需独立性。
- **方差** $\mathrm{Var}(X)=\mathbb{E}[(X-\mathbb{E}[X])^2]=\mathbb{E}[X^2]-(\mathbb{E}[X])^2$ 刻画弥散；$\mathrm{Var}(aX+b)=a^2\mathrm{Var}(X)$。
- **协方差** $\mathrm{Cov}(X,Y)=\mathbb{E}[XY]-\mathbb{E}[X]\mathbb{E}[Y]$ 刻画线性共变，相关系数把它归一化到 $[-1,1]$；**协方差为 0 不等于独立**。
- **协方差矩阵** $\boldsymbol{\Sigma}$ 对称半正定，对角线为方差、非对角线为协方差，是 PCA 与多元高斯的几何入口。
- **贝叶斯公式**：后验 = 似然 × 先验 / 证据，是下一节最大似然与贝叶斯统计的出发点。

在下一节，我们将回答「分布里的参数从哪来」：**最大似然估计与贝叶斯统计**——把「让观测数据最可能」与「把先验信念融进来」两种范式并排讲清楚，并顺势推出深度学习里最重要的损失函数。
