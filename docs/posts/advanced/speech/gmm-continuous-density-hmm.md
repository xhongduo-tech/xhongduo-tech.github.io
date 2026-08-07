---
title: 高斯混合模型（GMM）与连续密度 HMM
date: 2026-08-07
---

# 高斯混合模型（GMM）与连续密度 HMM

<div class="epigraph">
<p>整体大于部分之和。</p>
<footer>—— 亚里士多德（Aristotle）</footer>
</div>

<div class="article-byline">
<p>第四级 · 语音技术 ｜ 《语音信号处理》第9章 隐马尔可夫模型 ｜ 2026-08-07</p>
</div>

## 为什么从「连续」开始

前面所有 HMM 的公式里，发射分布都写成 $b_j(o_t)$——一个抽象记号。它到底长什么样，决定了声学模型的质感。最简单的选择是**离散发射表**：先对特征做矢量量化，再把每一帧标成一个码本索引，于是 $b_j(v_k)$ 是一张「状态 $j$ 发出码字 $v_k$ 的概率表」。这套路数在 1980 年代是主流，但有一个致命伤：**量化把连续的语音信息切碎了**。

MFCC 特征本质上是连续的、高维的、多峰的——同一状态「啊」的声学形态，在不同说话人、不同声调、不同协同发音下散布在特征空间的多个区域。一张离散表表达不了这种「一片区域」，更表达不了「多个区域」。解法是把发射分布换成**连续密度**，最经典的实现是**高斯混合模型（GMM）**：用若干个高斯分量拼出任意多峰的密度形状。亚里士多德那句话在这里有精确含义——**GMM 这个「整体」，能表达单个高斯「部分」拼不出的复杂分布**。<span class="marginnote">连续密度 HMM 的奠基论文是 Juang、Levinson 与 Sondhi 1986 年在 IEEE 上发表的 "Maximum likelihood estimation for multivariate mixture observations of Markov chains"。它把发射分布从码本表换成高斯混合，让 HMM 在真实连续特征上直接建模——这也是 GMM-HMM 声学模型此后统治 ASR 二十年的起点。</span>

## 1 矢量量化的损失：从连续到离散的信息坍塌

先看清离散 HMM 丢了什么。矢量量化（VQ）把 $D$ 维特征空间划分成 $M$ 个区域，每个区域用一个码本向量（质心）代表：

$$x \;\mapsto\; v_k = \arg\min_{v_m} \|x - v_m\|$$

这一步把所有落在同一区域的帧**映射成同一个码字**。于是两个差异很大的帧，只要离同一质心更近，就变成「一模一样」。识别时这种信息损失直接表现为混淆——「清音/浊音」或不同共振峰位置的微小差别被抹平了。更麻烦的是，量化误差和识别器是**分两阶段优化的**：先贪心地最小化量化失真，再训练 HMM，两个目标并不一致。<span class="marginnote">这也是「感知特征 + 矢量量化」时代（如 LPCC 特征配 VQ）的共同困境。连续密度 HMM 绕开了质心近似：每一帧的原始特征向量直接参与似然计算，不再经过「先离散化」这一步——误差只在模型拟合里存在，不再有独立的量化失真项。</span>

**核心概念：离散 HMM 的发射是「码字概率表」，连续 HMM 的发射是「特征空间上的概率密度」。** 前者一次只能回答「这个码字有多像」，后者能回答「这个特征向量有多大密度」——后者的分辨率是连续的，没有量化步长。

## 2 单高斯为什么不够：多峰的本质

先试最简单的连续密度：每个状态用**一个 $D$ 维高斯**做发射分布：

$$b_j(x) = \frac{1}{(2\pi)^{D/2} |\boldsymbol{\Sigma}_j|^{1/2}} \exp\Bigl[-\tfrac{1}{2}(x - \boldsymbol{\mu}_j)^{\top} \boldsymbol{\Sigma}_j^{-1} (x - \boldsymbol{\mu}_j)\Bigr]$$

单高斯是**单峰**的——密度只在均值附近一个椭圆区域取高值。但语音状态几乎都是**多峰**的：「啊」的共振峰位置因说话人、语调、上下文而漂移，特征分布常常裂成好几团。一个椭圆硬去拟合多团数据，均值会被拉向团与团之间的空白，方差被撑得巨大，识别时什么都像。

**高斯混合模型（GMM）**就是来治这个的：把 $M$ 个高斯按权重线性叠加：

$$b_j(x) = \sum_{m=1}^{M} c_{jm}\, \mathcal{N}(x;\, \boldsymbol{\mu}_{jm}, \boldsymbol{\Sigma}_{jm})$$

其中 $c_{jm} \ge 0$ 且 $\sum_m c_{jm} = 1$。每个分量 $\mathcal{N}(x; \boldsymbol{\mu}_{jm}, \boldsymbol{\Sigma}_{jm})$ 是单个高斯，权重 $c_{jm}$ 决定它在混合里占多大份量。<span class="marginnote">只要分量够多，GMM 可以以任意精度逼近任意连续密度——这是它的<strong>通用逼近性</strong>，也是它能统治声学模型二十年的数学底气。注意权重必须归一化（和为 1），否则密度积分不为 1，似然就不合法了。</span>

**辨析｜易错点：GMM 的「分量」不是 HMM 的「状态」。** 初学者常把「状态 $s_j$ 里的 8 个高斯」误当成 8 个状态。分辨口诀：**状态是时间轴上的块（音素内部的一段），分量是特征空间里的块（同一种声学形态的多个团）。** 一个状态对应一个 GMM，这个 GMM 里的多个分量描述的是「这个音素段可能出现的多种声音长相」。二者维度不同——状态沿时间切，分量沿特征切。

## 3 公式解析：GMM 发射概率的两层求和

把连续密度 HMM 的发射概率写全，它其实是一个**双随机结构**：HMM 的隐状态在上层，GMM 的分量选择是下层的第二个隐变量。给定状态 $s_j$，先按权重 $c_{jm}$ 选分量 $m$，再由该高斯产生观测 $x$：

$$
b_j(x) = \sum_{m=1}^{M} c_{jm}\; \underbrace{\frac{1}{(2\pi)^{D/2} |\boldsymbol{\Sigma}_{jm}|^{1/2}} \exp\Bigl[-\tfrac12 (x-\boldsymbol{\mu}_{jm})^{\top} \boldsymbol{\Sigma}_{jm}^{-1} (x-\boldsymbol{\mu}_{jm})\Bigr]}_{\mathcal{N}(x;\,\boldsymbol{\mu}_{jm},\,\boldsymbol{\Sigma}_{jm})}
$$

拆开看：

- **第一步，看外层求和**：$b_j(x)$ 是 $M$ 个高斯密度的加权和。为什么是「和」而不是「取最大」？因为这是一个**边际概率**——观测 $x$ 可以由任何一个分量产生，所以把所有「选中分量 $m$ 且由它产生 $x$」的联合概率加起来。这与 HMM 里「对所有状态路径求和」是同一逻辑。
- **第二步，看权重 $c_{jm}$**：它承担「先验」角色——观测前，我们有多相信这个分量。$c_{jm}$ 是概率，必须满足 $\sum_m c_{jm}=1$。
- **第三步，看高斯密度**：指数里的二次型 $(x-\boldsymbol{\mu})^{\top}\boldsymbol{\Sigma}^{-1}(x-\boldsymbol{\mu})$ 是 **Mahalanobis 距离**的平方——它同时考虑了特征各维的方差和相关性。密度在均值处最大，距离均值越远衰减越快。
- **第四步，看对数域**：训练和识别都在对数域算，但「和的 log」不能用「log 的和」，仍需 log-sum-exp：$\log b_j(x) = \mathrm{LSE}_m\bigl(\log c_{jm} + \log \mathcal{N}(x; \boldsymbol{\mu}_{jm}, \boldsymbol{\Sigma}_{jm})\bigr)$。

## 4 公式解析：GMM 的参数重估——第二个 EM

GMM 本身也有隐变量：**每个观测由哪个分量产生**。所以 GMM 的拟合也是一个 EM，通常就叫 GMM 的 EM 或「软聚类」。它的 E 步算**分量后验**（给定状态 $s_j$ 与观测 $x_t$）：

$$
\gamma_t(j, m) = \frac{c_{jm}\, \mathcal{N}(x_t;\, \boldsymbol{\mu}_{jm}, \boldsymbol{\Sigma}_{jm})}{b_j(x_t)}
$$

它是「Bayes 反推」：分子是「选分量 $m$ 并产生 $x_t$」的联合密度，分母 $b_j(x_t)$ 是全部分量产生 $x_t$ 的总密度——相除就是在状态 $j$ 下、观测 $x_t$ 由分量 $m$ 负责的后验概率。M 步用软计数重估三个参数：

$$
\hat{c}_{jm} = \frac{\sum_t \gamma_t(j,m)}{\sum_t \gamma_t(j)}, \qquad
\hat{\boldsymbol{\mu}}_{jm} = \frac{\sum_t \gamma_t(j,m)\, x_t}{\sum_t \gamma_t(j,m)}
$$

$$
\hat{\boldsymbol{\Sigma}}_{jm} = \frac{\sum_t \gamma_t(j,m)\, (x_t - \hat{\boldsymbol{\mu}}_{jm})(x_t - \hat{\boldsymbol{\mu}}_{jm})^{\top}}{\sum_t \gamma_t(j,m)}
$$

逐项看：

- **第一步，$\hat{c}_{jm}$**：权重 = 分量 $m$ 的期望占用次数 ÷ 状态 $j$ 的期望占用总次数——与 HMM 重估 $a_{ij}$ 是同一个「期望计数比」模式。
- **第二步，$\hat{\boldsymbol{\mu}}_{jm}$**：均值 = 观测的**加权平均**，权重是 $\gamma_t(j,m)$——软聚类的「类中心」。
- **第三步，$\hat{\boldsymbol{\Sigma}}_{jm}$**：协方差 = 加权外积的均值，中心化用新均值 $\hat{\boldsymbol{\mu}}_{jm}$。
- **第四步，与 HMM 的嵌套**：完整的声学模型要同时估计「状态转移」和「状态内 GMM」。真实训练是**双层 EM 嵌套**：外层的 HMM 的 E 步算出 $\gamma_t(j)$，内层的 GMM 再在 $\gamma_t(j)$ 内部细化出 $\gamma_t(j,m)$。两者交替进行，恰如俄罗斯套娃。

## 5 连续密度 HMM 的完整 Baum-Welch

把两层 EM 合并进一份训练流程，就是连续密度 HMM（CDHMM）的 Baum-Welch：

1. **初始化**：每个状态的 GMM 用少量高斯（如 1 个）起步，或对整批特征做全局聚类（k-means）给出均值初值。
2. **E 步（外层）**：用当前 $\lambda$ 跑前向后向，得到 $\gamma_t(j)$ 与 $\xi_t(i,j)$。
3. **E 步（内层）**：对每个状态 $j$、每帧，按第 4 节公式算分量后验 $\gamma_t(j,m)$。注意 $\sum_m \gamma_t(j,m) = \gamma_t(j)$——内外两层软计数严格一致。
4. **M 步**：用 $\gamma_t(j,m)$ 重估 GMM 三个参数（第 4 节），用 $\gamma, \xi$ 重估转移与初始（上一节第 2 节）。
5. **迭代**：重复 2–4，直到对数似然增量小于阈值。

把内层 GMM 的重估写成一个可运行的小函数，感受「软计数 → 参数」这一步有多直接：

```python
import numpy as np

def update_gmm(X, gamma_j):
    """给定状态 j 的观测 X:(N,D) 与该状态占用权重 gamma_j:(N,)
    重估单个 GMM 的 c, mu, Sigma（M 个分量，对角协方差）"""
    N, D = X.shape
    M = gamma_j.shape[1]
    c, mu, var = np.zeros(M), np.zeros((M, D)), np.zeros((M, D))
    for m in range(M):
        w = gamma_j[:, m]                       # 分量 m 的后验权重
        denom = w.sum() + 1e-10
        c[m] = denom / N
        mu[m] = (w[:, None] * X).sum(axis=0) / denom
        var[m] = (w[:, None] * (X - mu[m])**2).sum(axis=0) / denom
    return c, mu, var
```

配合外层 HMM 的 $\gamma_t(j)$，把每个时刻的 $\gamma_t(j)$ 按「帧属于状态 $j$ 的哪个分量」再细分，就得到 $\gamma_t(j,m)$，喂进上面的函数即可。**整个 CDHMM 训练里，HMM 层负责时间结构（状态怎么转移），GMM 层负责声学形状（状态听起来像什么）——两层各司其职，通过 $\gamma_t(j,m)$ 这一个软计数接口咬合。**

**核心要点：把「发射表」升级成「GMM」后，Baum-Welch 的全部骨架不变，只多了一层分量后验 $\gamma_t(j,m)$。** 这就是为什么语音界的惯例是先学离散/单高斯 HMM、再平滑过渡到 GMM——数学结构完全继承，只是把 $b_j$ 的表达式换掉。

**辨析｜易错点：协方差矩阵的形式有讲究。** 完整协方差是 $D \times D$ 矩阵，$D=39$（13 维 MFCC 加 delta 加 delta-delta）时就有 780 个自由参数，一个状态 × 8 分量 = 6000 多个参数，数据稍少就过拟合。工程上几乎都用**对角协方差**（只保留对角线方差），把参数降到每分量 $D$ 个；必要时再对若干状态**绑定（tying）**协方差或方差——这套「用结构换稳健性」的思路，正是下一节三音子状态绑定的先声。<span class="marginnote">为什么对角协方差够用？因为特征各维（不同倒谱系数）相关性不强，对角近似损失不大，却把方差估计变得稳定得多。Kaldi 里训练用的就是对角协方差 GMM，均值对角、方差对角，配合<strong>方差下限</strong>避免病态。</span>

## 6 小结

- **离散 HMM 的代价**：矢量量化把连续特征压成码字索引，信息丢失且量化与识别目标不一致。
- **单高斯不够**：语音状态特征多峰，单椭圆拟合会把均值拉进空白、方差撑爆。
- **GMM 发射**：$b_j(x) = \sum_m c_{jm} \mathcal{N}(x; \boldsymbol{\mu}_{jm}, \boldsymbol{\Sigma}_{jm})$，权重和密度的加权和，对数域用 log-sum-exp。
- **GMM 是第二层 EM**：分量后验 $\gamma_t(j,m)$ 做软计数，M 步重估 $c, \boldsymbol{\mu}, \boldsymbol{\Sigma}$。
- **CDHMM 训练 = 双层 EM 嵌套**：外层 HMM 算 $\gamma_t(j)$，内层 GMM 细化出 $\gamma_t(j,m)$，$\sum_m \gamma_t(j,m) = \gamma_t(j)$。
- **工程上**用对角协方差 + 方差下限，把参数规模压到可控。

在下一节，我们将面对声学模型真正的主人公：**三音子（triphone）模型与决策树状态绑定**——当「每个状态一个 GMM」扩展到几千个上下文相关状态时，数据稀疏问题怎么解决。
