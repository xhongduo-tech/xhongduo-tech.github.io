---
title: 机器学习中的几何方法：信息几何与 Fisher 度量
date: 2026-08-07
---

# 机器学习中的几何方法：信息几何与 Fisher 度量

<div class="epigraph">
<p>统计模型的参数空间带着天然的弯曲——沿着这个弯曲，我们才能找到信息意义上的最短路径。</p>
<footer>—— 甘利俊一（Shun-ichi Amari）</footer>
</div>

<div class="article-byline">
<p>第二级 · 微分几何 ｜ 陈维桓《微分几何》§9.6 ｜ 2026-08-07</p>
</div>

## 为什么从信息几何开始

本专题最后一块拼图：**信息几何（information geometry）**——把黎曼几何用到统计模型的空间上。核心对象是 **Fisher 度量（Fisher metric）**：

$$
g_{ij}(\theta) = \mathbb{E}_{p_\theta}\Big[\frac{\partial \log p_\theta}{\partial \theta^i}\frac{\partial \log p_\theta}{\partial \theta^j}\Big]
$$

它把「参数空间」（概率分布的集合）变成一个黎曼流形。为什么值得？因为**统计模型空间有自然的几何**：两个分布有多远、参数更新的正确方向、信息量的度量——全由 Fisher 度量编码。这是微分几何与统计学、机器学习的最终合流。<span class="marginnote">信息几何由甘利俊一（Amari）与 C.R. Rao（1945 年提出 Fisher 度量）奠基。它的核心洞见：<strong>概率分布的参数空间是一个黎曼流形</strong>（统计流形），Fisher 度量是它的自然度量，测地线是「信息最短路径」。现代机器学习里，自然梯度、变分推断、最优传输都与它深度相关。</span>

## 1 统计流形

**定义（统计流形）**：一族参数化概率分布 $\{p_\theta: \theta \in \Theta\}$，其参数空间 $\Theta$ 称为**统计流形**。$p_\theta$ 是 $\Theta$ 上的点。

**重点：把「分布」看成「点」——统计模型的参数空间是流形。** 每个 $\theta$ 对应一个分布，即流形上的一个点。「两个分布的距离」「参数更新的方向」都变成流形上的几何问题。<span class="marginnote">例子：正态分布族 $\{N(\mu,\sigma^2)\}$ 的参数 $(\mu,\sigma)$ 构成一个 2 维流形（$\mu$ 平移、$\sigma$ 缩放）；伯努利分布族 $\{p_\theta\}$（$0<\theta<1$）是 1 维流形（区间 $(0,1)$）。「一族分布」=「一个流形」，统计问题 = 流形上的几何问题。</span>

## 2 Fisher 度量：统计流形的黎曼度量

**定义（Fisher 度量）**：统计流形上的 Fisher 度量是参数空间的黎曼度量

$$
g_{ij}(\theta) = \mathbb{E}_{p_\theta}\Big[\frac{\partial\log p_\theta}{\partial\theta^i}\frac{\partial\log p_\theta}{\partial\theta^j}\Big] = -\mathbb{E}_{p_\theta}\Big[\frac{\partial^2\log p_\theta}{\partial\theta^i\partial\theta^j}\Big]
$$

（第二个等式来自 $\int \partial_\theta p = 0$ 对 $\theta$ 求导。）

**重点：Fisher 度量编码「参数 $\theta$ 的微小变化如何改变分布」——它定义统计流形上的长度与距离。** 两个分布的距离由

$$
d(p_{\theta_1}, p_{\theta_2}) = \int_0^1 \sqrt{g_{ij}\dot\theta^i\dot\theta^j}\,dt
$$

（连接两条参数曲线的测地线长度）给出——这正是黎曼距离（本专题第 61 节）。<span class="marginnote">直觉：Fisher 度量告诉你「$\theta$ 动一点，分布变多少」。若 $g_{ii}$ 大，说明该方向的参数「敏感」（信息量大）；若 $g_{ij}$ 为 0，说明两个参数方向「统计独立」。Fisher 度量 = 参数空间的「信息几何」。它与 KL 散度的关系：小距离下 $D_{KL}(p_\theta\parallel p_{\theta+\delta}) \approx \frac12 \delta^T g(\theta)\delta$——Fisher 度量是 KL 散度的二阶近似（局部几何）。</span>

## 3 自然梯度：Fisher 度量下的梯度下降

自然梯度（natural gradient）是信息几何在优化里的核心应用：

**定义（自然梯度）**：在 Fisher 度量下，目标函数 $L(\theta)$ 的黎曼梯度

$$
\widetilde\nabla L(\theta) = g(\theta)^{-1}\,\nabla L(\theta)
$$

**重点：自然梯度 = 用 Fisher 度量「校正」普通梯度——坐标无关的下降方向。** 普通梯度 $\nabla L$ 依赖参数化（换参数就变），自然梯度 $g^{-1}\nabla L$ 在参数重参数化下不变——它沿「统计流形上最快的下降方向」走。<span class="marginnote">为什么自然梯度更好？普通梯度沿欧氏方向下降，但参数空间的欧氏方向不尊重「分布结构」——换参数化方向就乱。自然梯度沿「信息最短路径」下降，是流形上的「正确」梯度。深度学习中，自然梯度与二阶方法（Fisher 信息矩阵）紧密相关——Adam 等优化器可以看作自然梯度的对角/自适应近似。</span>

### 自然梯度 vs 普通梯度

| 梯度 | 定义 | 坐标无关? | 几何含义 |
| --- | --- | --- | --- |
| 普通梯度 $\nabla L$ | 欧氏导数 | ❌ 依赖参数化 | 欧氏空间的下降方向 |
| 自然梯度 $g^{-1}\nabla L$ | Fisher 度量升级 | ✅ | 统计流形上最快的下降方向 |

## 4 公式解析：为什么 Fisher 度量 = KL 散度的局部形状

Fisher 度量与 KL 散度的关系是理解它的钥匙：

- **第一步，KL 散度**：$D_{KL}(p_\theta \parallel p_{\theta'}) = \int p_\theta \log\frac{p_\theta}{p_{\theta'}}$——分布间的「距离」（非对称）。
- **第二步，二阶展开**：对 $\theta' = \theta + \delta$ 展开，
  $$
  D_{KL}(p_\theta \parallel p_{\theta+\delta}) \approx \frac{1}{2}\sum_{ij} g_{ij}(\theta)\,\delta^i\delta^j
  $$
  一阶项消失（$p_\theta$ 处 KL 取极小值 0），二阶项正是 Fisher 度量。
- **第三步，结论**：**Fisher 度量是 KL 散度的二阶近似——它是分布空间的「无穷小距离」。** KL 散度是「全局距离」（不满足对称/三角），Fisher 度量是它的「局部几何化」（黎曼度量）。

**重点：Fisher 度量把 KL 散度的「信息差」变成「黎曼距离」——信息几何 = 用黎曼几何研究统计模型。** KL 散度非对称，Fisher 度量是对称化 + 局部化的「信息距离」。<span class="marginnote">「KL 是非对称的全局量，Fisher 是对称的局部量」——这是信息几何最精妙的转化。最优传输（Wasserstein 距离）是另一条路（「搬土成本」），与 Fisher 度量互补。现代几何深度学习里，Fisher（信息几何）与 Wasserstein（最优传输）是两大度量流派。</span>

## 5 信息几何的应用

信息几何在机器学习与统计中遍地开花：

- **自然梯度 / 在线学习**：Fisher 度量校正梯度——更快的收敛（在线自然梯度）。
- **变分推断**：在概率分布流形上优化 ELBO——KL 散度与 Fisher 度量是其几何。
- **贝叶斯统计**：后验分布的 Laplace 近似、Jeffreys 先验（Fisher 度量的体积元 $\sqrt{\det g}$）——「无信息先验」的几何。
- **Amari 的 α-几何**：除 Levi-Civita 联络外，统计流形还有一族 α-联络——「e-联络」与「m-联络」（指数族与混合族的对偶几何）。
- **生成模型**：GAN 的「模式坍缩」、扩散模型的时间重参数化都与分布空间几何相关。

**重点：信息几何 = 微分几何 + 统计学——它把本专题全部工具（度量、测地线、联络、曲率）用于「分布的空间」。** 这是微分几何从曲线曲面一路走到机器学习的终点站。<span class="marginnote">Amari 对偶联络：统计流形上除了黎曼度量，还有一对对偶的仿射联络（e 与 m），它们定义了「对偶平坦」结构——指数族分布（如正态、伯努利）在这种结构下有极其干净的几何（坐标与势函数对偶）。这是信息几何独有的「非 Levi-Civita 联络」应用——「联络不一定要由度量决定」在本专题第 62 节的预言在此兑现。</span>

### 例：正态分布族的 Fisher 度量

用最简单的统计模型算 Fisher 度量。正态分布族 $\{N(\mu, \sigma^2)\}$，参数 $(\mu, \sigma)$，对数似然

$$
\log p = -\frac12\log(2\pi\sigma^2) - \frac{(x-\mu)^2}{2\sigma^2}
$$

Fisher 度量 $g_{ij} = -\mathbb{E}[\partial_i\partial_j \log p]$ 算得

$$
g = \begin{pmatrix} 1/\sigma^2 & 0 \\ 0 & 2/\sigma^2 \end{pmatrix}
$$

**重点：正态分布族的 Fisher 度量是「对角」的（$\mu$ 与 $\sigma$ 统计独立），且随 $\sigma$ 变化——$\sigma$ 越大，$\mu$ 方向的度量越小（大方差时均值「信息量」低）。** 这给出统计流形 $(0,\infty)\times\mathbb{R}$ 上一个具体的黎曼度量。正态族是最简单的统计流形——它的几何（半平面上的 Poincaré 型度量）与双曲几何直接相连。**一个参数族，就是一个黎曼流形**——信息几何的大门由此打开。

## 6 小结

- **统计流形**：一族分布 $\{p_\theta\}$ 的参数空间——分布是点。
- **Fisher 度量**：$g_{ij} = \mathbb{E}[\partial_i\log p\,\partial_j\log p]$——统计流形的黎曼度量。
- **自然梯度** $g^{-1}\nabla L$：Fisher 度量下的坐标无关下降方向。
- Fisher 度量 = KL 散度的二阶近似（局部信息距离）。
- 应用：自然梯度、变分推断、Jeffreys 先验、α-几何、生成模型。

---

至此，从曲线的曲率到流形的曲率张量，从欧氏几何到黎曼几何，从地球仪到信息几何——「微分几何」这门学科的全景已经展开。**曲线看曲率挠率，曲面看基本形式，流形看度量联络，机器学习看数据流形**——一个概念（曲率）贯穿全部，一条主线（度量决定几何）从欧几里得通到现代 AI。
