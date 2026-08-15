---
title: 高维线性回归与 Lasso
date: 2026-08-07
---

# 高维线性回归与 Lasso

<div class="epigraph">
<p>从根本上说，所有模型都是错的，但有些是有用的。</p>
<footer>—— 乔治·博克斯（George E. P. Box），1976</footer>
</div>

<div class="article-byline">
<p>第二级 · 高维统计分析 ｜ Wainwright《High-Dimensional Statistics》Ch. 7 ｜ 2026-08-07</p>
</div>

## 为什么从高维回归开始

前两篇打下的武器（集中不等式、Rademacher 复杂性）现在要迎来第一个真正的战场：**线性回归**。经典回归有一个不成文的约定——样本量 $n$ 远大于变量个数 $d$。可现代数据几乎总是「长」这一边：基因组测序里几万个基因、几百个样本；推荐系统里百万特征、十万用户；图像里 $d$ 是像素数而 $n$ 是图片数。<span class="marginnote">这样的数据称为<strong>欠定（underdetermined）</strong>：未知数比方程多，$X^T X$ 奇异，最小二乘的解不再唯一。高维统计的全部戏剧性，都从「方程不够、未知数太多」这一句话开始。</span>

高维回归处理这个问题靠一个关键先验：**真实信号是稀疏的**——多数变量与响应无关。如何把「稀疏」这个定性信念翻译成可计算的算法？答案的化身就是 **Lasso**，而理解 Lasso 的最佳入口，是看它在几何上如何「把解摁到坐标轴上」。

## 1 经典回归的失效

考虑线性模型 $y = X\beta^* + w$，其中 $X \in \mathbb{R}^{n \times d}$，$w$ 是噪声（例如各分量独立同分布、方差 $\sigma^2$ 的次高斯噪声）。最小二乘估计

$$
\hat\beta_{\mathrm{OLS}} = \arg\min_\beta \frac{1}{2}\|y - X\beta\|_2^2
$$

的解满足正规方程 $X^T X \hat\beta = X^T y$。当 $d > n$ 时，$X^T X \in \mathbb{R}^{d \times d}$ 的秩至多 $n < d$，**必奇异**，正规方程有无穷多解。最小二乘彻底失效，而它的变体——最小范数解 $\arg\min\{\|\beta\|_2 : X\beta = y\}$——虽然存在，却把能量平均分摊到所有坐标上，估计方差与真实信号完全脱钩。<span class="marginnote">$d > n$ 时模型甚至能<strong>完美插值</strong>训练数据（解方程 $X\beta = y$ 而已），但这正是「过拟合」的极端形态：训练误差为零，泛化误差爆炸。插值与泛化的矛盾，直到深度学习的 double descent 现象才被重新审视。</span>

**辨析｜易错点：** 高维问题的病根不在「解不存在」，而在「解不唯一且不稀疏」。同一个 $y$ 可以有指数多个解释，其中绝大多数既不真实也无意义。因此高维回归的目标不是「求一个解」，而是「在合理先验下挑出那个真的解」——先验错了，一切后文皆空。

## 2 稀疏性：把「朴素」变成「先验」

如果真实回归系数 $\beta^*$ 只有 $s$ 个非零分量（记 $s = \|\beta^*\|_0$），且 $s \ll n$，那么问题骤然变得可解：我们只需要在这 $s$ 个变量的组合里寻找。麻烦在于**不知道是哪些变量**，而候选集合有 $\binom{d}{s}$ 个——当 $s$ 稍大，这个数超过宇宙中的原子数，暴力搜索不可行。

**Lasso（least absolute shrinkage and selection operator）**：由 Tibshirani 于 1996 年提出，目标是在平方损失上叠加 $\ell_1$ 罚：

$$
\hat\beta_\lambda = \arg\min_{\beta \in \mathbb{R}^d} \left\{ \frac{1}{2n}\|y - X\beta\|_2^2 + \lambda\|\beta\|_1 \right\}
$$

$\ell_1$ 罚的神奇之处在于它的几何形态：罚球是菱形（多面体），棱角恰好落在坐标轴上。**带约束的最小化问题中，解出现在损失等高线与罚球相切处**——只要切点落在菱形顶点上，就有分量恰好为零，产生稀疏解。$\ell_2$ 罚（岭回归）的球是光滑的圆，切点几乎必然落在轴外，只能把系数「压小」却压不成零。<span class="marginnote">为什么不用 $\ell_0$ 罚（直接数非零个数）？$\ell_0$ 目标是非凸的 NP-hard 问题；$\ell_1$ 是 $\ell_0$ 的<strong>凸松弛</strong>——在 $[-1,1]$ 箱上，$\ell_1$ 正是 $\ell_0$ 的凸包包络，且保稀疏性。这是「松弛换可解」思想的第一课，凸优化专题（进阶优化）会系统展开。</span>

![Lasso 与岭回归解的几何对比](/images/high-dimensional-statistics/lasso-l1-vs-l2-geometry.svg)

上图是理解 Lasso 的钥匙：左图 $\ell_1$ 罚球是菱形，解落在顶点，坐标为 $(c, 0)$ 型——稀疏；右图 $\ell_2$ 罚球是圆，解落在光滑边界上，两坐标都非零——稠密。**稀疏不是额外施加的偏执，而是凸惩罚函数的几何学后果。**

## 3 Lasso 解的最优性条件

Lasso 目标凸但不光滑，其最优性由次梯度刻画。令 $z \in \partial \|\beta\|_1$ 为次梯度，即 $z_j = \mathrm{sign}(\beta_j)$（若 $\beta_j \neq 0$），$z_j \in [-1, 1]$（若 $\beta_j = 0$）。则 $\hat\beta_\lambda$ 最优当且仅当

$$
\frac{1}{n} X^T(y - X\hat\beta_\lambda) = \lambda z, \qquad z \in \partial \|\hat\beta_\lambda\|_1
$$

这组条件一分为二，讲出两个独立的机制：

**等式部分（选变量）**：对 $\hat\beta_j \neq 0$ 的坐标，残差与第 $j$ 列的内积必须精确等于 $\lambda \mathrm{sign}(\hat\beta_j)$。这相当于「每个被选中的变量都要与残差保持固定相关度」——筛选变量靠的是与残差的相关，而非与 $y$ 的相关。

**不等式部分（筛掉噪声）**：对 $\hat\beta_j = 0$ 的坐标，只要求 $|\frac{1}{n}X_j^T r| \le \lambda$，其中 $r = y - X\hat\beta$。它给出一个干净的解释：**只有与残差相关（绝对值）超过 $\lambda$ 的变量才可能进入模型**。$\lambda$ 越大，门槛越高，模型越稀疏。这也解释了为什么 $\lambda$ 的合理尺度必须是噪声级别 $\sigma\sqrt{\log d / n}$——门槛太低，纯噪声变量的随机相关就会越线，把假变量选进来。<span class="marginnote">这里又见到 $\sqrt{\log d}$：从 $d$ 个噪声候选里挑「最大随机相关」，其量级正是 $\sigma\sqrt{(\log d)/n}$。这是集中不等式里「联合界的 $\log N$ 成本」在回归里的重现。</span>

## 4 公式解析：Lasso 的估计误差界

在合适条件下，Lasso 的估计误差以高概率满足

$$
\|\hat\beta_\lambda - \beta^*\|_2 \;\lesssim\; \sigma\sqrt{\frac{s \log d}{n}}
$$

这条界是高维回归的「名片」，四步拆解：

- **第一步，先验的代价**：误差只随**稀疏度 $s$** 增长，而不是维数 $d$。把 $d \to 10^6$ 而 $s$ 保持 20，误差几乎不动——这正是「稀疏假设买到的东西」。$\log d$ 是「不知道哪些变量稀疏」的探测成本。
- **第二步，基本不等式**：由 $\hat\beta$ 的最优性，$\frac{1}{2n}\|y - X\hat\beta\|^2 + \lambda\|\hat\beta\|_1 \le \frac{1}{2n}\|y - X\beta^*\|^2 + \lambda\|\beta^*\|_1$。代入 $y = X\beta^* + w$ 并整理，得到
  $$\frac{1}{2n}\|X(\hat\beta-\beta^*)\|_2^2 + \lambda\|\hat\beta\|_1 \;\le\; \lambda\|\beta^*\|_1 + \frac{1}{n}\langle w, X(\hat\beta - \beta^*)\rangle$$
  左边是「拟合损失 + 罚」，右边是「真实稀疏度 + 噪声扰动」。估计误差被拆成两部分：罚带来的偏差，噪声带来的方差。
- **第三步，限制本征条件（restricted eigenvalue, RE）**：让 $\hat\beta - \beta^*$ 落在「稀疏锥」$\mathcal{C} = \{\Delta : \|\Delta_{S^c}\|_1 \le 3\|\Delta_S\|_1\}$ 里（$S$ 是真支撑集，$S^c$ 是其补集），并要求设计矩阵满足
  $$\Delta^T\left(\frac{X^T X}{n}\right)\Delta \;\ge\; \kappa \|\Delta\|_2^2, \qquad \forall \Delta \in \mathcal{C}$$
  这一条**排除病态设计**：任意「稀疏方向」上的经验方差都被下界钉住，设计矩阵不至于在稀疏子空间上塌缩。RE 条件的直观版本是「各列的相关不能太高」——若两列几乎线性相关，它们就「不可识别」，误差下界也崩溃。
- **第四步，凑出答案**：取 $\lambda \asymp \sigma\sqrt{\log d / n}$（压住噪声项），把基本不等式与 RE 条件合并，稀疏锥的 $3\|\Delta_S\|_1$ 型不等式把 $\ell_1$ 范数转换成 $\ell_2$ 范数，最终得到 $\|\hat\beta - \beta^*\|_2 \lesssim \sigma\sqrt{s \log d / n}$。**样本量只需 $n \gg s \log d$**，与维数 $d$ 无关——这是非渐近分析最漂亮的结论之一。

## 5 符号恢复与不相干条件

估计误差衡量的是「系数有多接近」。更强的问题是**符号恢复（sign recovery）**：能不能精确找出哪些变量是信号（$\mathrm{sign}(\hat\beta_j) = \mathrm{sign}(\beta^*_j)$）？这需要更强的假设——**不相干条件（irrepresentability）**：

$$
\max_{j \in S^c} \left\| X_j^T X_S (X_S^T X_S)^{-1} \mathrm{sign}(\beta^*_S) \right\|_\infty \le 1 - \eta, \qquad \eta > 0
$$

直觉是：真变量与假变量的样本相关性必须被严格抑制——假变量对真变量「借力」之后，其与残差的相关仍要低于门槛 $\lambda$。**不相干条件是符号恢复的充分必要条件**（对固定设计的 Lasso 而言），比 RE 条件强得多。<span class="marginnote">「估计一致」与「选择一致」是两个强度不同的目标：RE 条件管前者，irrepresentability 管后者。Ravikumar、Wainwright 与 Raskutti 的系统性分析表明，两者之间存在本质的间隙——有些问题估计容易、选择却难。</span>

**辨析｜易错点：** 三个高频误区——其一，$\lambda$ 必须随 $n$ 增大而**收缩到零**（$\lambda \asymp \sqrt{\log d / n}$），固定 $\lambda$ 会造成不可消除的偏差；其二，$\ell_1$ 有**收缩偏差**（对非零系数整体按 $\lambda$ 方向收缩），因此 Lasso 不满足「Oracle 性质」，需用 adaptive Lasso 或阈值化修正——这会引出 oracle 不等式（本篇后文对应博文）；其三，正则化参数不能对每个问题用同一尺度，$\lambda$ 的标定依赖噪声水平 $\sigma$ 与设计矩阵，实际操作常用交叉验证（CV）与 $\lambda_{\max}$ 网格。

## 6 数值算例：Lasso 的坐标路径

Lasso 解随 $\lambda$ 变化走出的路径很能说明「选择」与「收缩」如何协同。考虑 $d = 3$ 个标准化变量、$n = 100$，真系数 $\beta^* = (2.0,\ 0,\ -1.5)$，噪声标准差 $\sigma = 0.8$。从大到小扫描 $\lambda$，解的形态（数值示意）如下：

| $\lambda$ | $\hat\beta_1$ | $\hat\beta_2$ | $\hat\beta_3$ | 非零个数 |
| --- | --- | --- | --- | --- |
| $\lambda_{\max}$（最大） | 0 | 0 | 0 | 0 |
| $0.5\,\lambda_{\max}$ | 0.9 | 0 | -0.4 | 2 |
| $0.2\,\lambda_{\max}$ | 1.6 | 0 | -1.1 | 2 |
| $0.05\,\lambda_{\max}$ | 1.9 | 0.06 | -1.4 | 3 |
| $\to 0$ | 2.0 | 0.03 | -1.5 | 3 |

两个特征一目了然：其一，**变量 $X_2$ 在整个路径上几乎都保持为零**——它的真系数为零、与残差的相关始终够不到门槛，Lasso 把它干净地筛掉；其二，**非零系数随 $\lambda$ 减小向真值逼近**——$\lambda$ 越大收缩越狠，$\lambda \to 0$ 时收敛到（接近）最小二乘解。**「选变量」与「收缩」不是两个开关，而是同一个 $\lambda$ 旋钮的两端。**<span class="marginnote">这张路径表在软件里就是 <code>glmnet</code> 或 <code>sklearn</code> 画出的 Lasso 路径图：横轴是 $\log\lambda$，纵轴是系数。路径上系数「恰好离零」的 $\lambda$ 对应变量被选入的临界点，交叉验证通常在这些临界点附近挑最优 $\lambda$。</span>

**对比三种正则化**：OLS 不收缩也不选；岭回归只收缩不选（系数变稠密）；Lasso 既收缩又选（得到稀疏解）。「哪些变量重要」这类**选择**问题必须靠 Lasso 一类的稀疏方法，而纯**收缩**问题（均方意义下的防过拟合）岭回归即可——**任务决定方法**。

把这张表与第 4 节的误差界对上：$\lambda \asymp \sigma\sqrt{\log d/n}$ 对应的位置约在 $0.05\lambda_{\max}$ 附近，此时估计误差 $\|\hat\beta - \beta^*\|_2$ 处于谷底——路径表让我们「看见」理论界的现实形态。

## 7 小结

- $d > n$ 时最小二乘失效：$X^T X$ 奇异，解不唯一；高维回归的出路是**稀疏先验**。
- **Lasso**：平方损失 + $\ell_1$ 罚，几何上解落在菱形罚球的**顶点**而稀疏；$\ell_1$ 是 $\ell_0$ 的凸松弛，是可解性与稀疏性的折中。
- 最优性由**次梯度条件**刻画：选中变量与残差相关恰为 $\lambda$，被筛变量相关绝对值被 $\lambda$ 封顶——$\lambda$ 是「选变量门槛」。
- 在**限制本征条件**下，估计误差 $\|\hat\beta-\beta^*\|_2 \lesssim \sigma\sqrt{s\log d/n}$，样本量只需 $\gg s\log d$ 而与 $d$ 无关。
- **符号恢复**要求更强的不相干（irrepresentability）条件；估计一致与选择一致是不同强度的问题。

在下一节，我们将顺着 Lasso 的思路走进压缩感知：当测量本身可以设计时，同样的稀疏先验如何以远低于奈奎斯特率的样本数精确恢复信号。
