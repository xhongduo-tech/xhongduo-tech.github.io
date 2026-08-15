---
title: Rademacher 复杂性与经验过程
date: 2026-08-07
---

# Rademacher 复杂性与经验过程

<div class="epigraph">
<p>我们相信上帝，其余的人请把数据带来。</p>
<footer>—— 威廉·爱德华兹·戴明（W. Edwards Deming）</footer>
</div>

<div class="article-byline">
<p>第二级 · 高维统计分析 ｜ Wainwright《High-Dimensional Statistics》Ch. 4 ｜ 2026-08-07</p>
</div>

## 为什么从经验过程开始

上一篇的集中不等式解决的是「一个随机变量偏离均值」。可学习的本质是**在一族候选函数里做选择**——线性回归要在所有系数向量里挑一个，分类要在所有分类器里挑一个。如果只是对单个函数逐点成立「样本均值近似总体均值」，那么在无穷多个函数里挑出最好的那个时，坏运气会积少成多。

**经验过程（empirical process）**：把「一族函数在样本上的表现」整体看作一个随机过程。设 $P$ 是 $X$ 的分布，$X_1, \dots, X_n \sim P$，对任意函数 $f$，定义

$$
\mathbb{G}_n(f) \;=\; \sqrt{n}\Big(\frac{1}{n}\sum_{i=1}^n f(X_i) - \mathbb{E}_P f\Big)
$$

对每个固定的 $f$，中心极限定理说 $\mathbb{G}_n(f)$ 依分布趋于正态——这是「逐点」的视角。但学习理论真正关心的是上确界 $\sup_{f \in \mathcal{F}} \mathbb{G}_n(f)$：**整族函数同时出错的概率**。控制这个上确界，需要把概率论与「函数的多少、复杂度」嫁接在一起，这就引出了复杂性的度量——Rademacher 复杂性。

## 1 一致收敛：从点到大族

经典的大数定律对每个 $f$ 分别给出 $\hat{P}_n f \approx P f$。若函数族 $\mathcal{F}$ 有限且大小为 $N$，用联合界叠加集中不等式，得到一致收敛：

$$
\mathbb{P}\left[\sup_{f\in\mathcal{F}}\left|\frac{1}{n}\sum_i f(X_i) - \mathbb{E}f\right| \ge t\right] \le 2N\, e^{-2nt^2}
$$

只要 $N$ 有限，样本容量 $n \asymp \log N / t^2$ 就够用——**$\log N$ 就是「在 N 个候选中挑选」的成本**。<span class="marginnote">这就是上一篇预告过的联合界 + 集中不等式的搭配。它已经能处理「有限个候选」，但高维问题的函数族通常是<strong>无限</strong>的——比如所有线性函数、所有阈值分类器。无限族的选样成本要重新度量，这就是 Rademacher 复杂性出场的动机。</span>

**一致大数定律（uniform LLN）**：若 $\sup_{f\in\mathcal{F}}|\hat{P}_n f - Pf| \to 0$ 依概率，则称 $\mathcal{F}$ 是一个 **Glivenko–Cantelli 类**。逐点收敛平凡，一致收敛却非平凡——它是「有限族可以，无限族未必」的分水岭。<span class="marginnote">经典例子：取 $\mathcal{F}$ 为所有 $[0,1]$ 上的指示函数 $\mathbf{1}\{x \le \theta\}$，$\sup$ 落在 Kolmogorov–Smirnov 统计量上；而若取一族「过拟合」函数，比如所有光滑函数，逐点收敛照样成立，一致收敛却必失败——这是 VC 理论的起源，见 Vapnik 与 Chervonenkis 1968 年的开创性工作。</span>

## 2 对称化：把「未知分布」换成「随机符号」

一致收敛的左侧有个麻烦：$\mathbb{E}_P f$ 依赖未知分布 $P$，没法直接算。对称化的想法是把期望之差 $\hat{P}_n f - P f$ 换成「两份独立样本均值之差」——两份样本同分布，差值的期望为 0，可计算性瞬间回来了。

更漂亮的做法是用**Rademacher 符号**。设 $\sigma_1, \dots, \sigma_n$ 独立同分布于等概率取 $\pm 1$ 的 **Rademacher 变量**，与数据独立。对称化引理给出核心不等式：

$$
\mathbb{E}\left[\sup_{f\in\mathcal{F}} \Big(\frac{1}{n}\sum_{i=1}^n f(X_i) - \mathbb{E}f\Big)\right]
\;\le\; 2\, \mathbb{E}_{X, \sigma}\left[\sup_{f\in\mathcal{F}} \left|\frac{1}{n}\sum_{i=1}^n \sigma_i f(X_i)\right|\right]
$$

右边的量就是（经验）**Rademacher 复杂性**。直觉上：$\sigma_i$ 随机地给每个样本打上「正号或负号」，一个函数 $f$ 若想同时匹配这堆随机符号，必须足够「能屈能伸」——而它匹配符号的平均能力，恰好度量了它的复杂度。<span class="marginnote">证明只用了一步「双重样本 + 三角不等式」：$\frac{1}{n}\sum f(X_i) - \mathbb{E}f \approx \frac{1}{n}\sum (f(X_i) - f(X_i'))$，再用 $\sigma_i(f(X_i) - f(X_i'))$ 的对称性把差拉开。细节见 Wainwright §4.2。</span>

## 3 Rademacher 复杂性：一族函数的「记忆容量」

**Rademacher 复杂性（Rademacher complexity）**：对函数族 $\mathcal{F}$，定义其经验 Rademacher 复杂性为

$$
\widehat{\mathcal{R}}_n(\mathcal{F}) \;=\; \mathbb{E}_\sigma\left[\sup_{f\in\mathcal{F}} \frac{1}{n}\sum_{i=1}^n \sigma_i f(X_i) \;\Big|\; X_1, \dots, X_n\right]
$$

再对数据取期望，$ \mathcal{R}_n(\mathcal{F}) = \mathbb{E}_X \widehat{\mathcal{R}}_n(\mathcal{F})$ 是**总体 Rademacher 复杂性**。

它满足几条直觉清晰的运算律：一族函数取绝对值后复杂性至多差常数；把 $\mathcal{F}$ 放进 $\mathcal{G}$（扩张函数族）复杂性随之上升；常数函数族的复杂性是 $O(1/\sqrt{n})$（因为 $\frac{1}{n}\sum \sigma_i \cdot 1$ 的标准差就是 $1/\sqrt{n}$）。**复杂性以 $1/\sqrt{n}$ 为基本单位衰减**——这正是「更多数据稀释复杂度」的定量形式。<span class="marginnote">与 Rademacher 复杂性并列的还有 <strong>Gaussian 复杂性</strong>：把 $\sigma_i$ 换成标准正态 $\gamma_i$。两者同阶（$C$ 倍意义下互有界），正态版本更适合与高斯过程的链式估计对接。Koltchinskii、Boucheron 等的综述里两者并用。</span>

**辨析｜易错点：** 上确界与期望的顺序不能交换。$\mathbb{E}\sup_f$ 与 $\sup_f \mathbb{E}$ 是两回事，前者大得多——「最坏函数（因数据而异）的平均表现」远高于「任一固定函数的平均表现」。初学者常把 $\mathbb{E}\sup_f$ 误写成 $\sup_f \mathbb{E}$，这一笔之差正是「过拟合的量化」与「平凡估计」的分界线。还要区分**经验**复杂性（给定数据后对 $\sigma$ 取期望）与**总体**复杂性（再对数据取期望）——前者可计算，后者是分析对象。

## 4 公式解析：Rademacher 界的完整链条

$$
\mathbb{E}\left[\sup_{f\in\mathcal{F}}\left(\hat{P}_n f - Pf\right)\right] \;\le\; 2\,\mathcal{R}_n(\mathcal{F})
$$

这是全篇最重要的式子，四步拆开：

- **第一步，左侧是什么**：$\hat{P}_n f - Pf = \frac{1}{n}\sum_i f(X_i) - \mathbb{E}f$，是「经验均值减总体均值」。取上确界再取期望，得到的是「最坏函数」的平均偏差——这就是估计误差的期望，是统计学习的核心量。
- **第二步，对称化引理**：期望之差可以用一份「镜像样本」替换，得到 $\mathbb{E}\sup_f \left|\frac{1}{n}\sum \sigma_i f(X_i)\right|$。这一步消掉了未知的 $P$，代价是引入 $\sigma$ 与一个因子 2。
- **第三步，为什么叫 Rademacher**：$\sigma_i \in \{\pm 1\}$ 以等概率出现，函数 $f$ 若能「追」上这些随机符号，说明它在数据点之间摆动得足够剧烈——这度量了函数族的**表达力**。
- **第四步，把期望与概率接起来**：用麦克迪阿米德（McDiarmid）有界差不等式，可以把「期望的界」升级为「概率的界」：以至少 $1-\delta$ 的概率，

$$
\sup_{f\in\mathcal{F}}(\hat{P}_n f - Pf) \;\le\; 2\,\mathcal{R}_n(\mathcal{F}) + \sqrt{\frac{\log(1/\delta)}{2n}}
$$

这里的 $\sqrt{\log(1/\delta)/n}$ 就是上一篇的「概率-距离换算」再一次现身。

## 5 熵积分：把复杂性折算成「函数的数量」

Rademacher 复杂性是精确对象，但计算它的上确界仍难。链式方法（chaining）给出可操作的估计：用一族「越来越细的近似」把上确界逼近出来。<span class="marginnote">思想与数值分析一致：先把整个函数族用一个 $2^{-j}$ 尺度的<strong>覆盖</strong>（每个函数配一个近似的代表元）框住，然后逐层放宽。每层的误差累积，最终得到一个关于覆盖半径的积分——即 Dudley 熵积分。</span>

**覆盖数（covering number）** $N(\mathcal{F}, \epsilon, \|\cdot\|)$：用半径 $\epsilon$ 的球覆盖 $\mathcal{F}$ 所需的最小球数。$\log N(\mathcal{F}, \epsilon)$ 称为**度量熵**，它就是「以 $\epsilon$ 精度分辨函数族所需的二进制位数」。**Dudley 熵积分**给出

$$
\mathcal{R}_n(\mathcal{F}) \;\lesssim\; \frac{1}{\sqrt{n}} \int_0^{\mathrm{diam}} \sqrt{\log N(\mathcal{F}, \epsilon, \|\cdot\|_2)}\, d\epsilon
$$

这统一了前文所有的直觉：**样本成本 $1/\sqrt{n}$ × 函数族复杂度（熵积分）**。函数族越「胖」，熵积分越大，达到给定精度所需的样本就越多——这就是**样本复杂性**的雏形，下一篇《非渐近方法与样本复杂性》将把它系统化。

**辨析｜易错点：** 覆盖数是「用有限近似无限」的关键，但 $\epsilon$ 的取值有讲究——积分必须停在分辨率与噪声同阶处，继续加密覆盖不会改进界，只会让对数因子白涨。判断覆盖质量要看是否用**高斯测度**（对数据分布）而非几何覆盖，两者相差可能是指数的——这是经验过程理论最精细的部分之一。

## 6 算例：手算一个 Rademacher 复杂性

把抽象定义落到能手算的例子。设 $\mathcal{F}$ 是常数函数类 $\{f_c(x) = c\}$，样本 $X_1,\dots,X_n$。则

$$
\widehat{\mathcal{R}}_n(\mathcal{F}) = \mathbb{E}_\sigma\left[\sup_{c \in \mathbb{R}} \frac{1}{n}\sum_{i=1}^n \sigma_i c\right] = \frac{|c|}{n}\,\mathbb{E}\Big|\sum_{i=1}^n \sigma_i\Big|
$$

而 $\sum_i \sigma_i$ 是 $n$ 步的 $\pm1$ 随机游走，其平均绝对偏差约 $\sqrt{2n/\pi}$，于是 $\mathcal{R}_n(\{f_c\}) \approx |c|\sqrt{2/(\pi n)}$——**复杂度以 $1/\sqrt{n}$ 衰减，连常数量级都算得出来**，这就是「常数函数族复杂度 $O(1/\sqrt{n})$」的手算证据。

对几类更复杂的对象，把「复杂度 vs 结构」的对比收进一张表：

| 函数类 | 复杂度阶 | 直觉 |
| --- | --- | --- |
| 常数函数 | $1/\sqrt{n}$ | 无自由度，只剩采样波动 |
| 有限类 $|\mathcal{F}|=N$ | $\sqrt{\log N/n}$ | 「在 $N$ 个里挑」的对数成本 |
| 线性函数 $\{\langle w,\cdot\rangle : \|w\|_2 \le 1\}$ | $\sqrt{d/n}$ | 每维一个方向 |
| 光滑函数类 | 熵积分 | 由光滑度与维数共同决定 |

这张表是「样本成本 = 复杂度 / $\sqrt{n}$」的直接读图：**要压低同一样本成本，要么函数类更小（结构更紧），要么样本更多**——这正是下一篇《非渐近方法与样本复杂性》要系统化的主线。

## 7 小结

- **经验过程**把「一族函数在样本上的表现」整体打包，学习理论关心的是**上确界**而非逐点值。
- 有限函数族的一致收敛由联合界 + 集中不等式搞定，成本是 $\log N$；无限族要靠复杂性度量。
- **对称化**用 Rademacher 符号替换未知分布，得到 $\mathbb{E}\sup_f(\hat{P}_n f - Pf) \le 2\mathcal{R}_n(\mathcal{F})$。
- **Rademacher 复杂性**度量函数族「匹配随机符号的能力」，以 $1/\sqrt{n}$ 为基本单位衰减；与 Gaussian 复杂性同阶。
- **覆盖数与熵积分（Dudley）**把复杂性折算成「以精度 $\epsilon$ 分辨函数族所需的位数」，得到样本成本的定量估计。

在下一节，我们将进入高维统计的第一个具体问题：当维数 $d$ 超过样本量 $n$ 时，线性回归如何被重新发明——这就是**高维线性回归与 Lasso**。
