---
title: 非渐近方法与样本复杂性
date: 2026-08-11
---

# 非渐近方法与样本复杂性

<div class="epigraph">
<p>凡事应当尽可能地简单，但不能更简单。</p>
<footer>—— 阿尔伯特·爱因斯坦（Albert Einstein）</footer>
</div>

<div class="article-byline">
<p>第二级 · 进阶数理 · 高维统计分析 ｜ 对标教材 ｜ 2026-08-11</p>
</div>

## 为什么从非渐近方法开始

前三篇里，我们已经一次次地看到同一种论证模式：集中不等式给概率界，Rademacher 复杂性给一致收敛界，Lasso/压缩感知给具体的恢复界。是时候把它们提升为一种**统一的方法论**了——这正是非渐近统计做的事：不问「$n\to\infty$ 时收敛吗」，而问「给定精度 $\varepsilon$ 与置信度 $\delta$，需要多少样本」。这个「多少样本」的答案，就是**样本复杂性（sample complexity）**。

**样本复杂性**：为达到「误差 $\le \varepsilon$、概率 $\ge 1-\delta$」的目标所需的最小样本量 $n(\varepsilon, \delta)$。它是整个统计学习理论的地基：算法设计与模型选择都在替它讨价还价。<span class="marginnote">这个提问方式与 PAC 学习同源——Valiant 1984 年的开创性论文把「学习」定义成「大概近似正确」：以概率 $1-\delta$ 达到误差 $\varepsilon$。PAC 语言里，$n(\varepsilon,\delta)$ 的渐近阶 $(\varepsilon, \delta) \to 0$ 恰恰是非渐近界最关心的「有限样本」行为。</span>

非渐近方法与经典渐近统计的分野不在结论，而在**论证方式**：渐近派用中心极限定理与 Delta 方法，非渐近派用集中不等式与复杂性度量。前者擅长「描述分布形状」，后者擅长「锁定最坏情形」——而高维统计恰恰是最坏情形（$d$ 大、函数族大）泛滥的地方。

## 1 两种视角：渐近与非渐近

经典教科书里，估计量的一致性写成 $\hat\theta_n \xrightarrow{P} \theta^*$，置信区间由 $\sqrt{n}(\hat\theta_n - \theta^*) \rightsquigarrow N(0, \sigma^2)$ 构造。这套体系有三条不成文的规矩：$n$ 要够大，维数 $d$ 固定，目标分布的形状已知。

非渐近方法三条全不认：$n$ 任意有限，$d$ 可以与 $n$ 同阶甚至更大，分布只需次高斯等「形状粗糙」的假设。它交付的不是「极限分布」，而是一张**显式常数的不等式清单**，比如

$$
\mathbb{P}\Big[\|\hat\theta_n - \theta^*\|_2 \ge \varepsilon\Big] \le \text{显式函数}(n, d, \delta)
$$

这类界的价值双重的：它告诉你 $n$ 与 $d$ 如何权衡（样本能不能「买」回维数），也告诉你概率目标与精度目标如何换算。**一句话：渐近统计给「分布」的答案，非渐近统计给「样本数」的答案。**<span class="marginnote">当然，非渐近界不是免费的：常数里藏着隐藏假设（如次高斯参数、RIP 常数、覆盖半径），而「常数是否可计算」往往决定一个界能否落地。Boucheron–Lugosi–Massart 的《Concentration Inequalities》第三章对这类常数来源做了详尽梳理。</span>

## 2 样本复杂性：怎么定义，怎么反解

设我们要以高概率控制估计误差。一个典型模板是：对任意 $n$，以概率 $\ge 1-\delta$，

$$
\|\hat\theta_n - \theta^*\| \;\le\; \sqrt{\frac{A(d)}{n}} + \sqrt{\frac{2B}{n}\log\frac{1}{\delta}}
$$

其中 $A(d)$ 编码结构复杂度（如维数 $d$、稀疏度 $s$、函数族复杂度），$B$ 编码「随机性成本」。要「误差 $\le \varepsilon$ 且概率 $\ge 1-\delta$」，只需同时满足

$$
\sqrt{\frac{A(d)}{n}} \le \frac{\varepsilon}{2}, \qquad \sqrt{\frac{2B}{n}\log\frac{1}{\delta}} \le \frac{\varepsilon}{2}
$$

反解 $n$，样本复杂性是

$$
n(\varepsilon, \delta) \;\asymp\; \frac{4A(d)}{\varepsilon^2} + \frac{8B}{\varepsilon^2}\log\frac{1}{\delta}
$$

**反解是从概率界到样本复杂性的机械操作**：先证概率界，再令每一项小于目标的份额，最后解出 $n$。两个观察：精度 $\varepsilon$ 以平方进入分母（**误差减半，样本翻四倍**）；置信水平 $\delta$ 以对数进入分子（**想让失败概率降到原来的千分之一，只需多采一个对数因子**）。

## 3 一致收敛：一族函数需要多少样本

把样本复杂性的机器装进学习问题。设有函数类 $\mathcal{F}$（模型假设），风险 $R(f) = \mathbb{E}[\ell(f(X), Y)]$ 与经验风险 $\hat R_n(f)$。上一节的 Rademacher 界给出（对取值 $[0,1]$ 的损失）：

$$
\sup_{f \in \mathcal{F}} \big(R(f) - \hat R_n(f)\big) \;\le\; 2\,\mathcal{R}_n(\mathcal{F}) + \sqrt{\frac{\log(1/\delta)}{2n}}
$$

令误差目标为 $\varepsilon$，反解得到学习的样本复杂性。对**有限类** $|\mathcal{F}| = N$，$\mathcal{R}_n(\mathcal{F}) \le \sqrt{(2\log N)/n}$，于是

$$
n(\varepsilon, \delta) \;\asymp\; \frac{\log N + \log(1/\delta)}{\varepsilon^2}
$$

**对数位 $\log N$ 是「在 N 个模型里挑一个」的代价，$\log(1/\delta)$ 是「要 1-δ 把握」的代价，两者各付其责**。这个公式是所有样本复杂性公式的「母公式」——把它中的 $N$ 换成覆盖数 $N(\mathcal{F}, \varepsilon)$、换成 $\binom{d}{s}$、换成 $(d/\varepsilon)^{O(d)}$，就分别得到无限类、稀疏类、参数类各异的样本复杂性。<span class="marginnote">这就是「模型越多，需要数据越多」的定量版本。机器学习里的「容量控制」、统计里的「惩罚」、信息论里的「描述长度」，在 $\log N$ 这个因子面前统统归一——它们本质上是同一笔账。</span>

## 4 公式解析：高斯均值估计的样本复杂性

把抽象公式落在一个具体的、教科书级的例子上。设 $X_1, \dots, X_n \overset{iid}{\sim} N(\mu, \sigma^2 I_d)$，用样本均值 $\hat\mu_n$ 估计 $\mu$。以概率 $\ge 1-\delta$：

$$
\|\hat\mu_n - \mu\|_2 \;\le\; \sigma\sqrt{\frac{d}{n}} + \sigma\sqrt{\frac{2}{n}\log\frac{1}{\delta}}
$$

反解出样本复杂性 $n(\varepsilon, \delta) \asymp \frac{\sigma^2}{\varepsilon^2}\left(d + \log\frac{1}{\delta}\right)$。四步拆解：

- **第一步，$\ell_2$ 误差为什么带 $d$**：$\|\hat\mu_n - \mu\|_2^2 = \sum_{j=1}^d (\hat\mu_{n,j} - \mu_j)^2$，每维的误差平方期望都是 $\sigma^2/n$，$d$ 维相加得 $d\sigma^2/n$，开方得 $\sigma\sqrt{d/n}$。**维数线性地进入 $\ell_2$ 误差，因为 $\ell_2$ 范数要「求和」**——这是维数诅咒的精确数学形态。
- **第二步，$\sqrt{d}$ 还是 $d$**：误差是 $\sqrt{d/n}$ 而不是 $d/n$。直觉：$d$ 维误差是 $d$ 个独立误差的「合成」，标准差按 $\sqrt{d}$ 增长，而不是按 $d$——统计学家靠这点只付出 $\sqrt d$ 的代价。
- **第三步，$\delta$ 项从哪来**：$\|\hat\mu_n - \mu\|_2^2$ 的分布是 $\sigma^2 \chi^2_d/n$ 缩放，用 $\chi^2$ 的次指数集中给出 $\sqrt{\log(1/\delta)/n}$。置信度换来的代价也是对数的。
- **第四步，与「结构假设」对比**：若 $\mu$ 已知是 $s$-稀疏，把估计投影到稀疏方向，$d$ 换成 $s\log(d/s)$，得到 $n \asymp s\log(d/s)/\varepsilon^2$——这正是 Lasso 篇与压缩感知篇反复出现的量。**样本复杂性的两副面孔：满维问题付 $d$，稀疏问题付 $s\log d$。**

## 5 覆盖数与熵积分：计算复杂性的工作马

对无限函数类（所有线性函数、所有光滑函数），$\log N$ 没有定义，母公式要换一种语言。**覆盖数** $N(\mathcal{F}, \varepsilon)$：用半径 $\varepsilon$ 的球覆盖 $\mathcal{F}$ 的最少球数；$\log N(\mathcal{F}, \varepsilon)$ 是**度量熵**——「以精度 $\varepsilon$ 分辨函数类所需的比特数」。函数类越丰富，熵越大，样本复杂性越高。

上一节的 **Dudley 熵积分**给出 Rademacher 复杂性的上界，于是统一了一族样本复杂性：参数类（$d$ 维）给出 $n \asymp d/\varepsilon^2$，稀疏类给出 $s\log(d/s)/\varepsilon^2$，非参数类（如 $\alpha$-光滑函数）给出 $n \asymp \varepsilon^{-2\alpha/(2\alpha+d)}$——后者的指数说明非参数估计「每要一点精度都要付出超线性样本」，这是维数诅咒最深刻的形态。<span class="marginnote">VC 维与度量熵是同一思想的两支：VC 维是「组合的」复杂度，只数函数类的「打碎」能力，对分布无知；度量熵是「度量的」复杂度，依赖函数类在数据分布下的几何。高维统计偏重后者，因为它在有限样本下更精细。</span>

**辨析｜易错点：** 三个高频误解——其一，「$n \asymp d/\varepsilon^2$ 里的常数」不是摆设：对高斯均值估计，常数涉及 $\sigma^2$，若把它当成 1，误差估计会系统性偏差；其二，**覆盖数必须用「适配的范数」**（通常是 $\ell_2$ 或高斯度量），用错了范数会得到指数级错误的熵；其三，样本复杂性公式常忽略「$\delta$ 项与 $\varepsilon$ 项共享同一份样本」——两个目标要**同时**满足，反解时必须各给一半预算，不能各要全套。

## 6 高维的样本复杂性图景

把全篇织成一幅图：**样本复杂性 =（复杂度/精度²）+ 置信成本**。其中「复杂度」随问题结构变化：

| 问题 | 复杂度项 | 样本复杂性 $n(\varepsilon,\delta)$ |
| --- | --- | --- |
| $d$ 维高斯均值（$\ell_2$） | $d$ | $\frac{\sigma^2}{\varepsilon^2}\left(d + \log\frac{1}{\delta}\right)$ |
| 有限类（$N$ 个模型） | $\log N$ | $\frac{\log N + \log(1/\delta)}{\varepsilon^2}$ |
| $s$-稀疏回归 | $s\log(d/s)$ | $\frac{s\log(d/s)}{\varepsilon^2}$（略去常数） |
| $\alpha$-光滑非参数 | 熵积分 | $\varepsilon^{-2\alpha/(2\alpha+d)}$ |

**横向对比读出三条经验**：一是 $\varepsilon^2$ 分母无处不在，精度是最贵的商品；二是 $\log$ 因子是「结构收益」的记号——稀疏、低秩、光滑等结构把 $d$ 换成 $s\log(d/s)$、把多项式换成对数；三是**这些样本复杂性多数是「紧」的**——下一篇将看到 minimax 下界给出同阶的匹配下界，证明「再好的算法也不能少采样」。<span class="marginnote">「上界（某算法达到）＋下界（任何算法必败）＝样本复杂性被精确刻画」。这是非渐近高维统计最令人满意的部分：不仅知道「我们的算法够好」，还知道「不可能更好」。二者缺口只是常数与对数因子时，问题即告「闭案」。</span>

## 7 小结

- **非渐近方法**交付显式常数的不等式而非极限分布，允许 $d$ 与 $n$ 同阶、分布粗糙。
- **样本复杂性** $n(\varepsilon,\delta)$：满足「误差 $\le\varepsilon$、概率 $\ge 1-\delta$」的最小样本量，由概率界**反解**得到。
- 母公式：$n \asymp \frac{\text{复杂度}}{\varepsilon^2} + \frac{\log(1/\delta)}{\varepsilon^2}$；有限类的复杂度是 $\log N$，无限类用**度量熵/熵积分**。
- 高斯均值估计给出典型例子：$n \asymp (d + \log(1/\delta))/\varepsilon^2$；稀疏结构把它换成 $s\log(d/s)$。
- 样本复杂性分「算法达到的上界」与「任何算法必败的下界」两半，合起来才完整。

在下一节，我们将从「线性」走向「低秩」：当未知对象不是稀疏向量而是低秩矩阵时，同样的非渐近方法论如何重现——这是**矩阵低秩恢复与 RPCA** 的主题。
