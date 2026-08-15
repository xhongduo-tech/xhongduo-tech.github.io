---
title: 二阶优化近似：牛顿法与拟牛顿法
date: 2026-08-07
---

# 二阶优化近似：牛顿法与拟牛顿法

<div class="epigraph">
<p>不仅要看脚下的坡度，还要看清前方的弯道。</p>
<footer>—— 依据伊萨克 · 牛顿（Isaac Newton）的精神改写</footer>
</div>

<div class="article-byline">
<p>第四级 · 深度学习 ｜ 花书《深度学习》§8.6、李沐《动手学深度学习》§11.6 ｜ 2026-08-07</p>
</div>

## 为什么从二阶优化近似开始

前面所有优化器（SGD、动量、Adam）都是**一阶方法**：只用梯度信息，默认「损失面是平的」，每一步沿最陡方向走。但损失面是**弯曲**的——在峡谷地形里，一阶方法锯齿横跳、收敛极慢。**牛顿法（Newton's method）**用**二阶信息**（黑塞矩阵）校正方向：它不只问「哪里最陡」，还问「前方的路怎么弯」，从而**一步到位**地走向二次近似的极值点。理想情况下，牛顿法对凸二次函数**一步收敛**——这是任何一阶方法都做不到的。

但黑塞矩阵 $O(n^2)$ 的存储与 $O(n^3)$ 的求逆，让纯牛顿法在大模型上不可行。于是有了**近似**：**拟牛顿法**（BFGS、L-BFGS）用「梯度差」逼近黑塞矩阵的逆，**共轭梯度**（CG）不显式构造矩阵，**对角/低秩近似**（K-FAC）用结构先验压成本。本节把「牛顿法为什么强、为什么贵、怎么打折用」讲透——它是理解「一阶 vs 二阶」这条优化主线的最后一课。<span class="marginnote">二阶方法在深度学习里长期「叫好不叫座」：理论优雅（一步收敛、无学习率）、实践难用（代价高、非凸下黑塞可能病态甚至不定）。但在<strong>小模型、强凸近似</strong>的场景（如逻辑回归、线性模型、中小规模 MLP 的精细求解），二阶方法仍然不可替代——第二级《最优化理论》里的牛顿/拟牛顿理论正是这里的数学根基。</span>

## 1 牛顿法：用曲率校正方向

**牛顿法**的更新规则：

$$
\boldsymbol{\theta} \leftarrow \boldsymbol{\theta} - \boldsymbol{H}^{-1}\boldsymbol{g}
$$

其中 $\boldsymbol{g} = \nabla_{\boldsymbol{\theta}} J$ 是梯度，$\boldsymbol{H}$ 是黑塞矩阵。对比梯度下降 $\boldsymbol{\theta} \leftarrow \boldsymbol{\theta} - \eta\boldsymbol{g}$：

- **梯度下降**：沿最陡方向走 $\eta$ 步——「直线下坡」。
- **牛顿法**：方向被 $\boldsymbol{H}^{-1}$ 旋转，步长由曲率自动决定——「顺着谷底走」。

**为什么一步收敛凸二次？** 对 $J(\boldsymbol{\theta}) = \frac{1}{2}\boldsymbol{\theta}^{\top}\boldsymbol{H}\boldsymbol{\theta} - \boldsymbol{\theta}^{\top}\boldsymbol{b} + c$，梯度是 $\boldsymbol{g} = \boldsymbol{H}\boldsymbol{\theta} - \boldsymbol{b}$。令 $\boldsymbol{\theta}^* = \boldsymbol{\theta} - \boldsymbol{H}^{-1}\boldsymbol{g}$，代入得 $\boldsymbol{H}\boldsymbol{\theta}^* = \boldsymbol{b}$，即 $\boldsymbol{\theta}^*$ 恰是极值点——**一步到位**。<span class="marginnote">「一步收敛二次函数」是理解牛顿法的钥匙：牛顿法本质上是「<strong>反复用二次近似替代原函数，并走到该近似的极值点</strong>」。对真正的二次函数，一次就够；对一般函数，每步「近似 → 走到近似极值」迭代。这也是「牛顿法对强凸函数二次收敛」这一理论结果（第二级《最优化理论》会证明）的来源。</span>

**在病态条件下，牛顿法天然免疫**：它除以 $\boldsymbol{H}$，等价于「把各向异性的曲率校正为各向同性」——条件数问题从原理上消失。这是二阶方法相对一阶方法的本质优势。

## 2 为什么纯牛顿法在深度学习里不可行

三个致命障碍：

**障碍一：存储爆炸。** 黑塞矩阵 $\boldsymbol{H}\in\mathbb{R}^{n\times n}$，$n=10^6$ 参数时就是 $10^{12}$ 个元素（数 TB）——连存都存不下。

**障碍二：求逆太贵。** 矩阵求逆 $O(n^3)$，$n=10^6$ 时不可计算；即使不求逆，解线性方程组也需 $O(n^2)$ 以上。

**障碍三：非凸下黑塞可能「不定」。** 在鞍点附近黑塞有正有负，$\boldsymbol{H}^{-1}$ 可能给出「上山方向」——牛顿法会「误导」参数走向极大值而非极小值。这是深度非凸问题里牛顿法的另一个坑。

**易错点：** 牛顿法「无学习率」只在凸二次假设下成立。一般函数里，牛顿步可能过大（远离极小值处二次近似失真），实际实现常加阻尼或学习率 $\eta<1$：$\boldsymbol{\theta}\leftarrow\boldsymbol{\theta}-\eta\boldsymbol{H}^{-1}\boldsymbol{g}$。<span class="marginnote">「非凸下用牛顿法要小心」的一个经典例子：在鞍点上，黑塞有负特征值，牛顿方向指向「上升」——一阶方法在鞍点「无路可走」但也不会上山，二阶方法反而可能被带偏。这解释了为什么「深度非凸 + 二阶方法」需要额外处理（如把黑塞投影到半正定）。</span>

## 3 拟牛顿法：用梯度差逼近黑塞

**拟牛顿法（quasi-Newton）**的思路：**不计算黑塞，而是用历次「梯度与参数的差」逐步逼近黑塞（或其逆）**。设相邻两步的差为

$$
\boldsymbol{s}_k = \boldsymbol{\theta}_k - \boldsymbol{\theta}_{k-1}, \qquad
\boldsymbol{y}_k = \boldsymbol{g}_k - \boldsymbol{g}_{k-1}
$$

**割线条件（secant condition）**要求近似的黑塞 $\boldsymbol{B}_k$ 满足 $\boldsymbol{B}_k\boldsymbol{s}_k = \boldsymbol{y}_k$——这是「平均曲率」在一维差商上的推广。**BFGS**（Broyden–Fletcher–Goldfarb–Shanno）用低秩更新迭代修正 $\boldsymbol{B}_k$，保证对称正定，收敛性好。

**L-BFGS（Limited-memory BFGS）**是深度学习里最实用的拟牛顿变体：**不存储完整的 $n\times n$ 矩阵，只保存最近 $m$ 步的 $\{\boldsymbol{s}_k, \boldsymbol{y}_k\}$**（$m$ 常取 5–20）。用它隐式重构曲率信息，内存从 $O(n^2)$ 降到 $O(nm)$。<span class="marginnote">L-BFGS 的「记忆只有 20 步」意味着它只捕捉「近期局部曲率」，但经验上对「目标函数光滑、无随机梯度」的问题（如优化一个固定的能量函数、训练小的全批量模型）效果极好。它不需要学习率（自带步长）、收敛快、免调参——<strong>在「全批量 + 中小规模 + 光滑」的三件套场景里，L-BFGS 常常完胜 Adam</strong>。</span>

**易错点：** 拟牛顿法的收敛理论依赖「目标光滑、梯度无噪声」。把 L-BFGS 用于小批量 SGD 的随机梯度上，割线条件失真，性能崩溃——**「随机」与「拟牛顿」天然不搭**，这是它未在深度学习主流立足的根本原因。

## 4 共轭梯度与其他近似

**共轭梯度法（Conjugate Gradient, CG）**：不显式构造黑塞，而是生成一组「共轭方向」$\{\boldsymbol{d}_k\}$，沿每个方向精确线搜索一次。对二次函数，$n$ 步内收敛；对一般函数配合**非线性 CG** 使用。它的优势是**内存 $O(n)$、无黑塞**，且对「大而稀疏」的二阶信息场景（如大规模线性系统）极高效。

**自然梯度（Natural Gradient）**：用 **Fisher 信息矩阵**（而非黑塞）作为曲率度量。它度量「参数空间里分布的相似度」，对「用分布拟合数据」的任务（如概率模型）有深刻的统计意义。**K-FAC** 把 Fisher 信息按层做 Kronecker 分解近似，是「二阶方法进深度学习」的一次著名尝试——在大规模训练里有加速报告，但实现复杂，未成为主流。<span class="marginnote">自然梯度与 Adam 的联系：Adam 用「梯度平方的对角均值」当 Fisher 信息的对角近似——所以 <strong>Adam 可以看作「自然梯度的对角近似」</strong>。这条线索把「一阶的 Adam」与「二阶的自然梯度」缝在一起：现代自适应方法其实都在「用廉价的曲率估计做二阶校正」。</span>

## 5 公式解析：牛顿步的几何意义

把牛顿步的几何含义写清楚。对损失 $J$ 在 $\boldsymbol{\theta}_0$ 附近做二阶泰勒展开：

$$
J(\boldsymbol{\theta}) \approx J(\boldsymbol{\theta}_0) + \boldsymbol{g}^{\top}(\boldsymbol{\theta}-\boldsymbol{\theta}_0) + \frac{1}{2}(\boldsymbol{\theta}-\boldsymbol{\theta}_0)^{\top}\boldsymbol{H}(\boldsymbol{\theta}-\boldsymbol{\theta}_0)
$$

- **第一步，求该二次近似的极值点**：对 $\boldsymbol{\theta}$ 求导并令为零：$\boldsymbol{g} + \boldsymbol{H}(\boldsymbol{\theta}-\boldsymbol{\theta}_0) = 0$，解得 $\boldsymbol{\theta} = \boldsymbol{\theta}_0 - \boldsymbol{H}^{-1}\boldsymbol{g}$。
- **第二步，看语义**：牛顿步 = 「走到二次近似的谷底」。对「近似得准」的区域，这一步几乎直接命中真实极小值附近。
- **第三步，看对比**：梯度下降走「固定步长 $\eta$ 的最陡方向」，牛顿法走「到谷底的距离与方向」——**前者像「蒙眼探路」，后者像「看清地形后的直达」**。<span class="marginnote">「到谷底的距离」这个解读点明了牛顿步的步长来源：它由曲率自动决定——<strong>曲率大的方向（谷底近）自动走小步，曲率小的方向（谷底远）自动走大步</strong>。这就是「免调学习率」的机制：牛顿法把「步长」也交还给几何，而不是靠超参数。</span>

## 6 一阶 vs 二阶：何时用谁

| 方法 | 曲率信息 | 内存/代价 | 学习率 | 适用 |
| --- | --- | --- | --- | --- |
| SGD/Adam | 无/对角 | $O(n)$ | 需调 | 大规模、随机梯度 |
| 牛顿法 | 完整黑塞 | $O(n^2)$/不可行 | 免调 | 仅小规模凸问题 |
| BFGS | 拟黑塞 | $O(n^2)$ | 免调 | 中规模、光滑 |
| L-BFGS | 近 m 步曲率 | $O(nm)$ | 免调 | 中小规模、全批量 |
| K-FAC | 分块 Fisher | 可控 | 需调 | 大规模探索性 |

**实践结论**：深度学习**大规模**训练用一阶（AdamW）；**中小规模、全批量、光滑**问题用 L-BFGS；**理论或小凸问题**用牛顿法。**「用不起二阶」是常态，「何时用得起」是工程判断**。<span class="marginnote">一个例外值得注意：<strong>微调小模型、超参数搜索的内部优化、以及「精确求解逻辑回归」等强凸子问题，L-BFGS 仍是利器</strong>。很多成熟的 scikit-learn 模型（逻辑回归、SVM）默认优化器就是 L-BFGS——「深度学习之外，二阶方法从未退场」。</span>

## 6 数值算例：牛顿步的「一步收敛」

把「牛顿法对凸二次一步收敛」算出来。设目标 $J(\boldsymbol{\theta}) = \frac{1}{2}(\boldsymbol{\theta}-\boldsymbol{\theta}^*)^\top\boldsymbol{H}(\boldsymbol{\theta}-\boldsymbol{\theta}^*)$，黑塞 $\boldsymbol{H}$ 为常数矩阵，梯度 $\boldsymbol{g} = \boldsymbol{H}(\boldsymbol{\theta}-\boldsymbol{\theta}^*)$。牛顿步：

$$
\boldsymbol{\theta} \leftarrow \boldsymbol{\theta} - \boldsymbol{H}^{-1}\boldsymbol{g}
    = \boldsymbol{\theta} - \boldsymbol{H}^{-1}\boldsymbol{H}(\boldsymbol{\theta}-\boldsymbol{\theta}^*)
    = \boldsymbol{\theta}^*
$$

**一步直接跳到最优点**——这与条件数无关，无论 $\boldsymbol{H}$ 的特征值多么悬殊。<span class="marginnote">对比《优化问题》里梯度下降需要约 $\frac{\kappa}{2}\ln(1/\varepsilon)$ 步（$\kappa=10^6$ 时要上百万步），牛顿法对二次问题一步到位——这就是「二阶信息」的价值：<strong>梯度只告诉方向，黑塞告诉你「每个方向该走多远」</strong>。但代价是每步要解 $\boldsymbol{H}^{-1}$：直接求逆是 $O(d^3)$，对百万维参数完全不可行——这正是拟牛顿法（用 $O(d^2)$ 的梯度差近似黑塞）与 Hessian-free（用共轭梯度解 $\boldsymbol{H}\boldsymbol{v}=\boldsymbol{g}$ 而不显式求逆）存在的意义。</span>

**为什么牛顿法在深度学习里很少直接用——三笔账。**
- 显存账：黑塞矩阵 $d\times d$，$d=10^8$ 时是 $10^{16}$ 元素，存不下。
- 计算账：求逆 $O(d^3)$，一步比梯度下降整个训练还贵。
- 稳定性账：深度损失非凸，黑塞可能不正定（有负特征值），牛顿步「朝山上走」。

**记忆锚点：一阶 vs 二阶的取舍。**
- 一阶（SGD/Adam）：便宜、鲁棒、靠噪声与动量；深度学习默认。
- 二阶（牛顿/拟牛顿）：每步质量高、收敛快；适合凸问题、小参数或病态严重的优化。
- 三个障碍让它在大模型不可行：**存储 $O(n^2)$、求逆 $O(n^3)$、非凸黑塞不定**。
- **拟牛顿法**（BFGS/L-BFGS）用梯度差逼近黑塞；**L-BFGS** 只存近 $m$ 步，内存 $O(nm)$。
- **共轭梯度**不构造矩阵，对稀疏大规模二阶问题高效；**K-FAC** 用 Kronecker 分解压 Fisher。
- 牛顿步 = 「走到二次近似谷底」；曲率自动决定步长，所以「免调学习率」。
- 选型：大规模一阶、中小全批量光滑用 L-BFGS；深度学习之外二阶仍是主角。

在下一节，我们回答优化之外的第一个工程问题：训练一开始的参数从哪来、怎么定——这就是**参数初始化：Xavier 与 Kaiming 初始化**。
