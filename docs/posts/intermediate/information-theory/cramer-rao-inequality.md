---
title: Cramér-Rao 不等式与信息不等式
date: 2026-08-07
---

# Cramér-Rao 不等式与信息不等式

<div class="epigraph">
<p>估计的精度有一道物理极限：Fisher 信息是资本，方差是利息，Cramér-Rao 说利息永远不低。</p>
<footer>—— 哈拉尔德 · 克拉默（Harald Cramér）</footer>
</div>

<div class="article-byline">
<p>第二级 · 信息论 ｜ Cover &amp; Thomas《Elements of Information Theory》 §11.10 ｜ 2026-08-07</p>
</div>

## 为什么从「估计精度的物理极限」开始

Fisher 信息度量「数据含多少参数信息」。把它倒过来，就得到统计估计最重要的不等式——**Cramér-Rao 不等式**：

$$
\text{Var}(\hat\theta) \ge \frac{1}{I(\theta)}
$$

对任何**无偏估计** $\hat\theta$（$\mathbb{E}[\hat\theta] = \theta$），其方差不能低于 Fisher 信息的倒数。

**一句话**：**样本里关于 $\theta$ 的信息只有 $I(\theta)$ 这么多，任何无偏估计的误差方差都不能比它的倒数更小。**

- 噪声大（$I$ 小）→ 方差下界高 → 参数难估。
- 噪声小（$I$ 大）→ 方差下界低 → 参数好估。
- **达到下界的估计叫「有效估计」**——它榨干了数据里的全部信息。

这一篇我们证明它、解释它与信息论的呼应、并说明最大似然估计为什么「渐近有效」。<span class="marginnote">Cramér-Rao 不等式是统计估计的「海森堡不确定性原理」：Fisher 信息是数据的「信息资本」，估计方差是「误差利息」，不等式说利息永远不能低于资本的倒数。它与 Fano 不等式（误差概率下界）构成统计推断的「误差下限」双子塔。</span>

## 1 定理与证明：柯西-施瓦茨一击

**Cramér-Rao 不等式**：设 $\hat\theta$ 是 $\theta$ 的无偏估计，满足正则条件，则

$$
\text{Var}(\hat\theta) \ge \frac{1}{I(\theta)}
$$

**证明**：

**第一步，把无偏性写成积分**：$\mathbb{E}[\hat\theta] = \int \hat\theta(x) f(x;\theta) dx = \theta$。对 $\theta$ 求导：

$$
\int \hat\theta(x) \frac{\partial f}{\partial\theta} dx = 1
$$

把 $\partial_\theta f = f \cdot \partial_\theta \log f$ 代入：

$$
\int \hat\theta(x) f \cdot s(x;\theta) dx = 1, \qquad s = \partial_\theta \log f
$$

**第二步，利用 $\mathbb{E}[s] = 0$**：$\int f \cdot s\, dx = 0$，所以可以写成「协方差」：

$$
\int (\hat\theta - \theta) f \cdot s\, dx = 1
$$

**第三步，柯西-施瓦茨**：

$$
1 = \left(\int (\hat\theta - \theta)\sqrt{f} \cdot s\sqrt{f}\, dx\right)^2 \le \int (\hat\theta - \theta)^2 f\, dx \cdot \int s^2 f\, dx = \text{Var}(\hat\theta) \cdot I(\theta)
$$

移项即得 $\text{Var}(\hat\theta) \ge 1/I(\theta)$。<span class="marginnote">证明的「配 $\sqrt{f}$」技巧是柯西-施瓦茨的标准用法：把「无偏性」的积分写成两个 $L^2$ 函数的内积，内积的平方 ≤ 范数乘积。方差与 Fisher 信息在一对「对偶量」的意义上相遇——方差大 ⇔ 信息小，反之亦然。</span>

## 2 公式解析：$1/I(\theta)$ 的构成

把不等式拆开：

$$
\text{Var}(\hat\theta) \ge \frac{1}{I(\theta)}
$$

- **$\text{Var}(\hat\theta)$**：估计的方差——「估计有多抖」。越小越好。
- **$I(\theta)$**：Fisher 信息——「数据有多少参数信息」。
- **$1/I(\theta)$**：方差下界——「信息资本的倒数利息」。
- **等号条件**：$(\hat\theta - \theta)\sqrt{f}$ 与 $s\sqrt{f}$ 成正比，即 $\hat\theta$ 的得分与 $s$ 线性相关——这恰好是**指数族 + 充分统计量**的情形。

**n 个样本版本**：由可加性 $I_n = nI$，

$$
\text{Var}(\hat\theta) \ge \frac{1}{nI(\theta)}
$$

样本量翻倍，方差下界减半——「数据越多，估得越准」的精确量化。<span class="marginnote">「$n$ 样本方差 $\ge 1/(nI)$」是 Cramér-Rao 的工程形态：它告诉你「要估准到某个精度，至少需要多少样本」。这个「样本量预算」在实验设计、A/B 测试、临床试验里是核心指标——Cramér-Rao 给的是「任何方法都绕不开的最低成本」。</span>

## 3 有效估计与 MLE 的渐近有效性

**有效估计（efficient estimator）**：达到 Cramér-Rao 下界的无偏估计。

- 高斯均值：$\bar X$ 方差 $= \sigma^2/n = 1/(n \cdot 1/\sigma^2)$——恰好达到下界，有效。
- 一般情况下，有效估计存在当且仅当「得分线性于充分统计量」（指数族）。

**MLE 的渐近有效性**：最大似然估计（MLE）在正则条件下渐近达到下界：

$$
\sqrt{n}(\hat\theta_{MLE} - \theta) \xrightarrow{d} \mathcal{N}\left(0, \frac{1}{I(\theta)}\right)
$$

**含义**：样本多时，MLE 的方差 ≈ $1/(nI(\theta))$——它自动榨干了数据里的全部信息。**MLE 是「渐近有效」的**：不浪费任何 Fisher 信息。

**与 Fano 不等式的对照**：

| | Fano 不等式 | Cramér-Rao |
| --- | --- | --- |
| 场景 | 离散猜测（分类） | 连续估计（参数） |
| 下界 | 错误概率 $P_e$ | 方差 $\text{Var}$ |
| 关键量 | 互信息 / 条件熵 | Fisher 信息 |
| 达到 | 最优分类器 | 有效估计 / MLE |

两者是「信息不够，误差就有下界」在分类与估计两个场景的双生子。<span class="marginnote">「MLE 渐近有效」是统计估计的旗舰结论：它说明「最大化似然」不只是个聪明的招数，而是「在信息极限处工作」的算法。现代深度学习里「交叉熵损失 ≈ 极大似然」，所以神经网络分类器在数据足够时也渐近有效——信息论给深度学习的「为什么好用」提供了一个精确答案。</span>

**辨析｜易错点：** 三个容易误解的地方：

**「无偏」是前提**：有偏估计可以方差更小（甚至零方差——猜固定值），但它偏离真相。Cramér-Rao 只约束无偏估计。实际中「方差-偏差权衡」正是「允许一点偏差换更小方差」的权衡。
**正则条件是必要的**：Cramér-Rao 需要「得分期望为零」「支撑集与参数无关」等条件；支撑集随参数变的分布（如均匀 $\text{Unif}(0,\theta)$）不满足，可以用别的方法（此时方差可以低于 $1/I$）。
**$I(\theta)$ 随 $\theta$ 变**：下界是 $\theta$ 的函数——在 $\theta$ 大的区域信息多、方差小；不同参数点的可估性不同。<span class="marginnote">「$\text{Unif}(0,\theta)$ 违反正则条件」是个经典反例：它的支撑集依赖 $\theta$，Fisher 信息的定义都失效，而「样本最大值」估计的方差远优于「$1/I$」所暗示。Cramér-Rao 不是普遍真理，是「正则统计」里的真理——边界情形需要新工具。</span>

## 4 信息不等式：有偏版本的推广

**信息不等式（information inequality）**：放宽「无偏」，对任意估计（可以有偏差 $b(\theta)$），

$$
\mathbb{E}[(\hat\theta - \theta)^2] \ge \frac{(1 + b'(\theta))^2}{I(\theta)} + b(\theta)^2
$$

其中 $b(\theta) = \mathbb{E}[\hat\theta] - \theta$ 是偏差，$b'$ 是它的导数。

- 无偏（$b = 0$）：退化为 Cramér-Rao。
- 有偏：均方误差 ≥ 「方差下界」+「偏差平方」——**偏差与方差都要付账**。

**直觉**：偏差是「系统性偏离」，方差是「随机抖动」；信息不等式把两者一起约束——「任何估计的总误差都有下界，由信息资本决定」。<span class="marginnote">「偏差与方差都要付账」是信息不等式对机器学习「偏差-方差权衡」的精确化：模型复杂（低偏差高方差）或简单（高偏差低方差）都有代价，而信息不等式给出「总代价的下界」。机器学习里的正则化，本质是在「付账」上选一个平衡点。</span>

**与全课程体系的连接：** Cramér-Rao 在第二级《概率论与数理统计》里是「区间估计与假设检验」的理论基础；在第四级《机器学习》里对应「MLE 的渐近效率」「Fisher 信息矩阵（自然梯度）」；它与 Fano 不等式（第二篇）共同构成「信息 → 误差下界」的两大支柱。

## 5 小结

- **Cramér-Rao 不等式**：$\text{Var}(\hat\theta) \ge 1/I(\theta)$——无偏估计的方差下界。
- 证明：无偏性求导 → 得分期望为零 → 柯西-施瓦茨。
- $n$ 样本版：$\text{Var} \ge 1/(nI)$——样本量预算的精确量化。
- **有效估计**达到下界；**MLE 渐近有效**（方差 → $1/(nI)$）。
- 与 Fano 对照：分类用错误概率、估计用方差，同一句「信息不足误差有下界」。
- **信息不等式**：$\text{MSE} \ge (1+b')^2/I + b^2$——偏差与方差都要付账。
- **辨析**：无偏是前提；正则条件必要；$I(\theta)$ 随 $\theta$ 变。

在下一篇，我们见识一条把「高斯最分散」推上巅峰的不等式：**熵幂不等式（Entropy Power Inequality）初步**——两个独立变量之和的熵，下界几何。
