## 核心结论

共轭先验的定义很直接：如果先验分布 $p(\theta)$ 在看到数据后，更新得到的后验分布 $p(\theta\mid x)$ 仍然属于同一个分布族，那么这个先验就是该似然函数的共轭先验。白话解释是：更新前后“长得还是同一类分布”，只是参数变了。

它的核心价值不是“概念优雅”，而是“计算变简单”。贝叶斯更新本质上要算

$$
p(\theta\mid x)\propto p(x\mid \theta)p(\theta)
$$

如果后验还能落回同一分布族，就通常不需要再做复杂数值积分，只要更新几个超参数。超参数就是“控制分布形状的参数”，比如 Beta 分布里的 $\alpha,\beta$，可以理解为先验里对成功和失败的“伪计数”。

对初学者最重要的结论有三个：

| 结论 | 含义 | 直接收益 |
|---|---|---|
| 后验与先验同族 | 更新前后分布形式不变 | 易于推导和实现 |
| 更新常表现为“加法” | 数据被压缩为计数、均值、平方和等充分统计量 | 在线更新很方便 |
| 多数经典共轭对来自指数族 | 指数族是常见的一类分布家族，形式统一 | 能系统化记忆与应用 |

最经典的玩具例子是 Beta-Binomial。若硬币正面概率为 $\theta$，先验取 $\theta\sim \mathrm{Beta}(\alpha,\beta)$，观测 $n$ 次投掷中有 $y$ 次正面，则后验直接变成：

$$
\theta\mid y \sim \mathrm{Beta}(\alpha+y,\beta+n-y)
$$

这件事可以直接理解为：看到一次成功，就往“成功桶”里加 1；看到一次失败，就往“失败桶”里加 1。

常见共轭对可以先记住这一张表：

| 似然模型 | 参数 | 共轭先验 | 后验更新的核心形式 |
|---|---|---|---|
| Bernoulli / Binomial | 成功概率 $\theta$ | Beta$(\alpha,\beta)$ | $(\alpha+y,\beta+n-y)$ |
| Poisson | 事件率 $\lambda$ | Gamma$(\alpha,\beta)$ | $(\alpha+\sum x_i,\beta+n)$ |
| Multinomial | 类别概率向量 $\boldsymbol{\pi}$ | Dirichlet$(\boldsymbol{\alpha})$ | $\boldsymbol{\alpha}+\boldsymbol{c}$ |
| Gaussian（均值未知，方差已知） | 均值 $\mu$ | Gaussian | 精度相加、均值加权平均 |
| Gaussian（方差未知） | 方差 $\sigma^2$ | Inverse-Gamma | 与平方残差累加有关 |

---

## 问题定义与边界

问题定义是：给定似然 $p(x\mid\theta)$ 与先验 $p(\theta)$，是否存在一个先验，使得后验 $p(\theta\mid x)$ 与先验属于同一分布族？如果存在，这个先验就叫共轭先验。

这里有三个边界必须说清。

第一，共轭不是“任何模型都能套”。标准教材里的闭式共轭，大多依赖指数族。指数族是一类可以写成统一指数形式的分布，像 Bernoulli、Binomial、Poisson、Gaussian、Multinomial 都属于这类。因为它们的对数似然结构规整，所以容易和某些先验拼成同族后验。

第二，共轭要求支持空间匹配。支持空间就是参数允许取值的范围。比如 Bernoulli 的参数 $\theta$ 必须在 $[0,1]$，所以 Beta 分布自然合适；如果你拿高斯分布去做这个先验，就会把概率质量放到小于 0 或大于 1 的区域，定义本身就不成立。

第三，共轭是“形式上方便”，不是“表达力最强”。当真实问题存在相关性、重尾、稀疏结构、混合机制时，强行用共轭先验，往往会让模型过度简化。

以 Bernoulli / Binomial 为例，为什么 Beta 是它的共轭？原因有两层：

| 条件 | Bernoulli / Binomial 的要求 | Beta 是否满足 |
|---|---|---|
| 参数范围匹配 | $\theta\in[0,1]$ | 满足 |
| 代数形式匹配 | 似然中出现 $\theta^y(1-\theta)^{n-y}$ | Beta 核心项正是 $\theta^{\alpha-1}(1-\theta)^{\beta-1}$ |

两者相乘后，指数只会相加，因此仍然是 Beta 族。这就是“共轭”背后的代数本质。

还可以把常见情况按适用条件再整理一次：

| 观测分布 | 数据特征 | 共轭先验 | 适用条件 |
|---|---|---|---|
| Bernoulli / Binomial | 二元结果、成功失败计数 | Beta | 参数是单个概率 |
| Multinomial | 多类别计数 | Dirichlet | 概率向量各分量非负且和为 1 |
| Poisson | 单位时间事件计数 | Gamma | 强度参数非负 |
| Gaussian | 连续值、噪声近似正态 | Gaussian / Normal-Inverse-Gamma | 需区分未知的是均值还是方差 |

所以，共轭先验不是一个单独技巧，而是一种“分布族匹配关系”。

---

## 核心机制与推导

核心机制可以压缩成一句话：先验和似然相乘后，如果能重新整理成原分布族的核函数，那么就得到共轭。

这里的“核函数”可以理解为：先不管归一化常数，只看决定分布形状的那一部分。

先看最重要的 Beta-Binomial 推导。设 $y$ 是 $n$ 次试验中的成功次数，参数为 $\theta$。

二项似然是：

$$
p(y\mid \theta)=\binom{n}{y}\theta^y(1-\theta)^{n-y}
$$

Beta 先验是：

$$
p(\theta)=\frac{1}{B(\alpha,\beta)}\theta^{\alpha-1}(1-\theta)^{\beta-1}
$$

根据贝叶斯公式，

$$
p(\theta\mid y)\propto p(y\mid\theta)p(\theta)
$$

代入后得到：

$$
p(\theta\mid y)\propto \theta^y(1-\theta)^{n-y}\theta^{\alpha-1}(1-\theta)^{\beta-1}
$$

把同类项合并：

$$
p(\theta\mid y)\propto \theta^{\alpha+y-1}(1-\theta)^{\beta+n-y-1}
$$

这正是一个 Beta 分布的核，因此：

$$
\theta\mid y \sim \mathrm{Beta}(\alpha+y,\beta+n-y)
$$

这一步没有数值积分，没有采样，没有近似。更新只体现在指数相加。

把它做成一个“桶模型”更容易理解：

| 信息来源 | 加到哪个参数 | 含义 |
|---|---|---|
| 先验的 $\alpha$ | 成功桶 | 先验相信成功出现过多少次 |
| 先验的 $\beta$ | 失败桶 | 先验相信失败出现过多少次 |
| 数据中的 $y$ | 成功桶 | 新观察到的成功数 |
| 数据中的 $n-y$ | 失败桶 | 新观察到的失败数 |

玩具例子：先验 $\mathrm{Beta}(2,3)$，表示你一开始略偏向“这枚硬币不太容易出正面”；现在投 6 次，出现 4 次正面、2 次反面。那么后验就是：

$$
\mathrm{Beta}(2+4,3+2)=\mathrm{Beta}(6,5)
$$

后验均值从先验均值

$$
\frac{2}{2+3}=0.4
$$

变成

$$
\frac{6}{6+5}\approx 0.545
$$

数据把你的信念往“更可能正面”方向推了一步，但因为先验不是空的，所以不是直接跳到经验频率 $4/6$。

更一般地，共轭更新之所以常常是“加法”，是因为指数族似然可写为

$$
p(x\mid\theta)=h(x)\exp\big(\eta(\theta)^\top T(x)-A(\theta)\big)
$$

其中 $T(x)$ 是充分统计量。充分统计量就是“保留与参数相关的全部信息的压缩量”。对于 Bernoulli，它是成功次数；对于 Gaussian，它通常是样本和与平方和；对于 Multinomial，它是各类别计数。

自然共轭先验通常可以写成与 $\eta(\theta)$ 同构的指数形式，因此看到新数据，本质上就是把充分统计量累加到超参数里。

再看 Dirichlet-Multinomial。若类别概率向量为 $\boldsymbol{\pi}=(\pi_1,\dots,\pi_K)$，先验是

$$
\boldsymbol{\pi}\sim \mathrm{Dirichlet}(\alpha_1,\dots,\alpha_K)
$$

观测到类别计数 $\boldsymbol{c}=(c_1,\dots,c_K)$ 后，后验就是

$$
\boldsymbol{\pi}\mid \boldsymbol{c}\sim \mathrm{Dirichlet}(\alpha_1+c_1,\dots,\alpha_K+c_K)
$$

这和 Beta-Binomial 完全同构，只不过从两个桶变成了 $K$ 个桶。

高斯-高斯共轭也类似。若观测模型是

$$
x_i\mid \mu \sim \mathcal{N}(\mu,\sigma^2)
$$

且 $\sigma^2$ 已知，先验取

$$
\mu\sim \mathcal{N}(\mu_0,\tau_0^2)
$$

则后验仍是高斯。更方便的写法是用精度 $\lambda=1/\sigma^2$，$\lambda_0=1/\tau_0^2$：

$$
\lambda_n=\lambda_0+n\lambda,\qquad
\mu_n=\frac{\lambda_0\mu_0+\lambda\sum_{i=1}^n x_i}{\lambda_0+n\lambda}
$$

这说明后验均值不是简单平均，而是“先验均值”和“样本均值”的加权平均，权重由各自精度决定。精度可以理解为“置信程度的倒数尺度”，越大表示越确信。

---

## 代码实现

工程上最常见的做法，不是每次都去重新写贝叶斯公式，而是直接维护超参数和充分统计量。也就是说，把“共轭更新”实现成一个普通函数。

下面先给出 Beta-Binomial 的可运行实现：

```python
from typing import Iterable, Tuple

def update_beta_posterior(alpha: float, beta: float, observations: Iterable[int]) -> Tuple[float, float]:
    """
    observations 中每个元素只能是 0 或 1。
    1 表示成功，0 表示失败。
    更新规则：
    alpha += 成功数
    beta  += 失败数
    """
    if alpha <= 0 or beta <= 0:
        raise ValueError("alpha and beta must be positive")

    success = 0
    total = 0
    for x in observations:
        if x not in (0, 1):
            raise ValueError("observations must be 0 or 1")
        success += x
        total += 1

    failure = total - success
    return alpha + success, beta + failure


def beta_mean(alpha: float, beta: float) -> float:
    if alpha <= 0 or beta <= 0:
        raise ValueError("alpha and beta must be positive")
    return alpha / (alpha + beta)


a, b = update_beta_posterior(2.0, 3.0, [1, 1, 1, 1, 0, 0])
assert (a, b) == (6.0, 5.0)
assert abs(beta_mean(a, b) - 6.0 / 11.0) < 1e-12
```

这个函数的重点不是语法，而是接口设计：输入先验参数与观测，输出后验参数。这样你可以在流式数据里反复调用，而不需要保留所有历史样本。

再看 Dirichlet-Multinomial 的模板：

```python
from typing import List

def update_dirichlet_posterior(alpha: List[float], counts: List[int]) -> List[float]:
    """
    alpha[k] 是第 k 类的先验伪计数
    counts[k] 是第 k 类的观测计数
    """
    if len(alpha) != len(counts):
        raise ValueError("alpha and counts must have the same length")
    if any(a <= 0 for a in alpha):
        raise ValueError("all alpha values must be positive")
    if any(c < 0 for c in counts):
        raise ValueError("counts must be non-negative")

    posterior = [a + c for a, c in zip(alpha, counts)]
    return posterior


posterior = update_dirichlet_posterior([0.5, 0.5, 0.5], [3, 1, 0])
assert posterior == [3.5, 1.5, 0.5]
assert abs(sum(posterior) - 5.5) < 1e-12
```

真实工程例子：LDA 主题模型。LDA 里每篇文档的“主题分布”和每个主题的“词分布”通常都用 Dirichlet 先验。部署时不会每来一篇文档就做复杂积分，而是维护词-主题计数、文档-主题计数，再按共轭关系更新后验或条件分布。这使得主题建模可以在大规模文本集合上高效迭代。

如果要做高斯-高斯更新，通常维护样本个数和样本和即可：

```python
def update_gaussian_mean_posterior(mu0, var0, data, obs_var):
    """
    已知观测方差 obs_var，未知均值 mu。
    先验: mu ~ N(mu0, var0)
    观测: x_i ~ N(mu, obs_var)
    """
    if var0 <= 0 or obs_var <= 0:
        raise ValueError("variances must be positive")

    data = list(data)
    n = len(data)
    precision0 = 1.0 / var0
    precision = 1.0 / obs_var

    post_precision = precision0 + n * precision
    post_var = 1.0 / post_precision
    post_mu = (precision0 * mu0 + precision * sum(data)) / post_precision
    return post_mu, post_var


mu_n, var_n = update_gaussian_mean_posterior(0.0, 1.0, [1.0, 2.0, 3.0], 1.0)
assert var_n > 0
assert 0.0 < mu_n < 3.0
```

实现层面有两个经验：

| 实现点 | 建议 |
|---|---|
| 不要反复存全量样本 | 共轭模型通常只需维护充分统计量 |
| 要检查参数合法性 | Beta/Dirichlet/Gamma 的超参数都必须为正 |
| 流式数据优先做增量更新 | 新数据到来时直接累加 |
| 后验预测优先用解析公式 | 能不采样就不采样，先利用闭式结构 |

---

## 工程权衡与常见坑

共轭先验最大的优点是快，最大的缺点是模型被限制住了。你得到了解析更新，也接受了表达力约束。

下面这张表能概括主要权衡：

| 维度 | 共轭模型 | 非共轭模型 |
|---|---|---|
| 计算成本 | 低，常可闭式更新 | 高，常需采样或优化 |
| 可解释性 | 高，更新机制清晰 | 中等，结果依赖算法近似 |
| 模型灵活性 | 受限 | 更强 |
| 在线更新 | 很适合 | 通常更难 |
| 高维复杂依赖 | 往往不足 | 更能表达 |

最常见的坑不是公式推错，而是建模假设太强。

第一，先验强度过大。比如你给 Beta 先验设成 $\mathrm{Beta}(1000,1000)$，即使来了几十个样本，后验也几乎不动。因为这相当于你提前塞进了 2000 次“伪观测”。新手常把“有先验”误解为“随便给一个大数更稳”，实际结果往往是过度保守。

第二，把独立性假设当成事实。真实工程中，参数常常相关。比如贝叶斯神经网络里，若给每个权重都设独立高斯先验，计算方便，但这默认“权重之间没有结构关联”。在深层网络里，这通常过于粗糙，可能导致不确定性估计偏差。白话说：模型看起来会输出方差，但这个方差未必可信。

第三，把共轭当成普适解。LDA、朴素贝叶斯、简单点击率估计都很适合共轭，但如果数据有时间漂移、群体差异、层级结构，只用一个固定先验会吞掉很多关键信息。这时更合理的办法是层级先验。层级先验就是“先验的参数也带随机性”，比固定超参数更灵活。

第四，忽略参数化差异。Gamma 分布常见两套参数化：shape-rate 和 shape-scale。工程实现里如果文档和代码不统一，后验更新会直接错。高斯模型里精度和方差也容易混淆。

常见坑与规避方式可以总结如下：

| 常见坑 | 后果 | 规避方式 |
|---|---|---|
| 先验过强 | 后验几乎不受数据影响 | 让伪计数或先验精度可解释、可校准 |
| 支持空间不匹配 | 模型定义错误 | 先检查参数取值范围 |
| 忽略相关性 | 不确定性估计失真 | 用层级先验、协方差结构或非共轭方法 |
| 参数化混乱 | 推导和代码不一致 | 明确写出公式和单位 |
| 把解析方便当成真实合理 | 模型偏置 | 先验证假设，再追求闭式解 |

---

## 替代方案与适用边界

当共轭不可得，或者共轭假设太硬时，主流替代方案有三类。

1. 变分推断。变分推断是把后验近似成一个可优化的分布族，通过最优化替代积分。优点是快，适合大规模数据；缺点是近似偏差可能明显。
2. MCMC。MCMC 是用马尔可夫链抽样逼近后验。优点是表达力强、理论完整；缺点是慢，在线系统很难承受。
3. Laplace 近似。Laplace 近似是在后验众数附近用高斯近似。优点是简单；缺点是对多峰或强偏态后验不稳。

可以做一个实用对比：

| 方法 | 速度 | 灵活性 | 结果解释性 | 适合场景 |
|---|---|---|---|---|
| 共轭闭式更新 | 很高 | 低到中 | 高 | 在线更新、教学、基线系统 |
| 变分推断 | 高 | 高 | 中 | 大规模近似贝叶斯 |
| MCMC / HMC | 低 | 很高 | 中到高 | 高精度后验分析 |
| Laplace 近似 | 中 | 中 | 中 | 局部近似足够时 |

真实工程例子可以继续看 LDA。LDA 之所以经典，正是因为文档-主题分布和主题-词分布都配了 Dirichlet 先验，和 Multinomial 形成共轭，所以很多更新可以转化为计数累加。在 collapsed Gibbs sampling 里，也正是利用共轭把某些变量积分掉，才让采样更高效。

但如果你的词分布不再满足简单 Multinomial 结构，或者你要引入复杂神经网络编码器，那么标准 Dirichlet 共轭结构就不够了。这时往往转向变分推断，甚至直接上 amortized inference。白话说：为了更强表达力，你要接受更重的计算代价。

贝叶斯优化也是类似。标准高斯过程回归在高斯噪声假设下有漂亮的解析后验，这就是高斯共轭结构带来的收益，所以它特别适合超参数搜索这类“反复更新、每次样本很贵”的任务。但如果观测噪声明显非高斯，或者目标函数有重尾异常值，标准共轭高斯假设就会失效，需要改用更鲁棒的似然或近似推断。

因此，共轭先验最适合的边界是：

| 适合使用 | 不适合直接使用 |
|---|---|
| 需要快速重复更新 | 单次推断可以容忍高计算成本 |
| 参数结构简单、解释优先 | 存在强相关、层级、重尾、混合机制 |
| 指数族模型 | 非指数族或复杂神经参数化模型 |
| 教学、基线、在线监控 | 高维复杂科研建模 |

判断标准很实际：如果你更在意“快、稳、可解释”，先试共轭；如果你更在意“真、细、灵活”，就准备接受近似推断或采样。

---

## 参考资料

1. Wikipedia, *Conjugate prior*：适合理清定义、常见共轭对和指数族背景。  
2. ScienceDirect, *Conjugate Prior Distributions*：适合补充共轭分布在统计建模中的标准用法与局限。  
3. Blei 等相关主题模型资料与 LDA 综述：适合理解 Dirichlet-Multinomial 在文本建模中的工程价值。  
4. SciML 课程中关于 Bayesian Neural Network with Gaussian Prior 的讲义：适合理解高斯先验在贝叶斯神经网络中的基础角色。  
5. Bayesian Optimization 相关论文与教程：适合理解高斯过程先验与解析后验更新在超参数搜索中的应用。  
6. Bishop, *Pattern Recognition and Machine Learning*：适合作为高斯共轭、Gamma-Poisson、Dirichlet-Multinomial 的系统教材。  
7. Murphy, *Machine Learning: A Probabilistic Perspective*：适合把共轭先验放回完整概率建模框架中理解。
