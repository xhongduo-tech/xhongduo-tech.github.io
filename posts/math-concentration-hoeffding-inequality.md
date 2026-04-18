## 核心结论

Hoeffding 不等式处理的是“有界独立随机变量的和”：如果每个随机变量的取值范围已知，并且样本之间相互独立，那么样本和或样本均值偏离期望的概率会按 $\exp(-\Theta(t^2))$ 衰减。

它不是在问“均值会不会偏”，而是在问“偏得这么多有多难发生”。例如 $X_i \in [0,1]$ 且相互独立时：

$$
P(|\bar X_n - E\bar X_n| \ge \varepsilon) \le 2e^{-2n\varepsilon^2}
$$

这说明：样本量 $n$ 越大，允许偏差 $\varepsilon$ 越大，发生这种偏差的概率上界越小。这里的衰减速度是指数级，而不是线性级。

McDiarmid 不等式把对象从“随机变量的和”推广到“一般函数”。只要函数 $f(X_1,\dots,X_n)$ 对每个输入坐标的最大敏感度有上界，也能得到类似的平方型指数尾界。

次高斯随机变量是尾部像高斯分布一样快衰减的随机变量。它的核心不是“方差小”，而是指数矩母函数受控。

| 工具 | 研究对象 | 核心前提 | 典型结论 | 典型用途 |
|---|---|---|---|---|
| Hoeffding | 有界独立随机变量的和或均值 | $X_i$ 独立，$X_i \in [a_i,b_i]$ | 偏差概率 $\le \exp(-\Theta(t^2))$ | 样本均值、转化率、固定模型泛化误差 |
| McDiarmid | 独立输入上的函数 | 每个坐标对函数值影响 $\le c_i$ | 偏差概率 $\le \exp(-2t^2/\sum c_i^2)$ | 交叉验证指标、复杂聚合指标 |
| 次高斯 | 单个随机变量或随机过程 | mgf 被高斯型上界控制 | $E e^{\lambda(X-EX)} \le e^{\lambda^2\sigma^2/2}$ | 统一描述尾部、推导集中界 |

---

## 问题定义与边界

随机变量是带有概率性的数值，例如一次用户是否转化可以记为 $X \in \{0,1\}$。样本均值是多个观测值的平均：

$$
\bar X_n = \frac{1}{n}\sum_{i=1}^n X_i
$$

经验风险是模型在训练样本上的平均损失，总体风险是模型在真实数据分布上的期望损失。若损失函数为 $\ell(h,Z)$，固定假设 $h$ 的经验风险和总体风险分别是：

$$
\hat R_n(h)=\frac{1}{n}\sum_{i=1}^n \ell(h,Z_i),\quad R(h)=E[\ell(h,Z)]
$$

本文讨论的是概率偏差界：在给定样本量下，经验量偏离期望量超过某个阈值的概率有多大。它不是点估计误差的精确值，也不是说这次实验一定偏多少。

本文只讨论以下条件下的集中界：

$$
X_1,\dots,X_n \text{ 独立},\quad X_i \in [a_i,b_i],\quad S_n=\sum_{i=1}^n X_i
$$

玩具例子：若 $X_i \sim \mathrm{Bernoulli}(0.6)$，每个 $X_i$ 表示一次独立投币实验是否成功，则 $X_i \in [0,1]$，且样本之间独立，满足 Hoeffding 的前提。

反例：如果 $X_i$ 是同一个用户在连续 10 天内是否点击广告，那么这些样本通常不是独立的。用户偏好、活动周期、推荐系统状态都会引入相关性。此时直接把它们当作独立样本套 Hoeffding，会低估偏差概率。

另一个边界是有界性。如果变量可能取极大值，例如订单金额服从重尾分布，单个异常订单就能显著改变均值，Hoeffding 的有界前提不成立，必须换工具或先做截断、稳健估计。

---

## 核心机制与推导

Hoeffding 的推导主线有三步：

| 步骤 | 操作 | 作用 |
|---|---|---|
| 1 | 尾概率指数化 | 把 $P(S_n-ES_n\ge t)$ 变成指数函数的期望 |
| 2 | 利用独立性拆分 mgf | 把整体期望拆成单个变量的乘积 |
| 3 | 对 $\lambda$ 最优化 | 得到最紧的平方型指数上界 |

mgf 是 moment generating function，中文常译为矩母函数，白话说就是用 $E e^{\lambda X}$ 描述随机变量尾部增长速度的工具。

对任意 $\lambda>0$，由 Markov 不等式：

$$
P(S_n-ES_n\ge t)
= P(e^{\lambda(S_n-ES_n)}\ge e^{\lambda t})
\le e^{-\lambda t}E e^{\lambda(S_n-ES_n)}
$$

因为 $X_i$ 独立：

$$
E e^{\lambda(S_n-ES_n)}
= \prod_{i=1}^n E e^{\lambda(X_i-EX_i)}
$$

Hoeffding 引理说明，如果 $X_i\in[a_i,b_i]$，则：

$$
E e^{\lambda(X_i-EX_i)}
\le \exp\left(\frac{\lambda^2(b_i-a_i)^2}{8}\right)
$$

代回并对 $\lambda$ 最优化，可得：

$$
P(S_n - ES_n \ge t)
\le
\exp\left(
-\frac{2t^2}{\sum_i (b_i-a_i)^2}
\right)
$$

对下尾同理，再用 union bound 合并两侧尾部。union bound 是把多个坏事件概率相加来控制“至少一个坏事件发生”的概率。若 $X_i\in[0,1]$，令 $t=n\varepsilon$，得到：

$$
P(|\bar X_n - E\bar X_n| \ge \varepsilon)
\le 2\exp(-2n\varepsilon^2)
$$

次高斯随机变量的定义是：存在 $\sigma>0$，使得对任意 $\lambda\in\mathbb R$：

$$
E\exp(\lambda(X-EX)) \le \exp(\lambda^2\sigma^2/2)
$$

这说明它的中心化版本在指数矩意义下不比方差参数为 $\sigma^2$ 的高斯变量更重尾。

McDiarmid 的对象不是和，而是函数。若独立输入 $X_1,\dots,X_n$ 上的函数 $f$ 满足有界差分条件：

$$
\sup_{x_i,x_i'} |f(\dots,x_i,\dots)-f(\dots,x_i',\dots)|\le c_i
$$

则：

$$
P(f(X)-Ef(X)\ge t)
\le
\exp\left(-\frac{2t^2}{\sum_i c_i^2}\right)
$$

直观解释是：删掉或替换第 $i$ 个样本后，指标最多变化 $c_i$。如果所有 $c_i$ 都很小，整体函数不可能被单个样本大幅拉动，因此会集中在期望附近。

在泛化误差中，若固定一个假设 $h$，且损失 $\ell(h,Z)\in[0,1]$，则：

$$
P(|R(h)-\hat R_n(h)|\ge\varepsilon)\le 2e^{-2n\varepsilon^2}
$$

若有限假设类 $|H|<\infty$，则：

$$
P\left(\sup_{h\in H}|R(h)-\hat R_n(h)|\ge\varepsilon\right)
\le 2|H|e^{-2n\varepsilon^2}
$$

要让失败概率不超过 $\delta$，足够条件是：

$$
n\ge \frac{\ln(2|H|/\delta)}{2\varepsilon^2}
$$

所以样本复杂度是：

$$
O\left(\frac{\ln |H|+\ln(1/\delta)}{\varepsilon^2}\right)
$$

---

## 代码实现

下面的代码把 Hoeffding 界、有限假设类的 union bound、样本量下界都变成可计算结果，并用 Bernoulli 模拟做一个最小验证。

```python
import math
import random

def hoeffding_two_sided_bound(n, eps):
    return 2 * math.exp(-2 * n * eps * eps)

def finite_class_bound(n, eps, h_size):
    return 2 * h_size * math.exp(-2 * n * eps * eps)

def sample_complexity(eps, delta, h_size=1):
    return math.ceil(math.log(2 * h_size / delta) / (2 * eps * eps))

def bernoulli_mean(p, n):
    return sum(1 if random.random() < p else 0 for _ in range(n)) / n

def estimate_tail_frequency(p=0.6, n=100, eps=0.1, trials=20000):
    bad = 0
    for _ in range(trials):
        mean = bernoulli_mean(p, n)
        if abs(mean - p) >= eps:
            bad += 1
    return bad / trials

n = 100
eps = 0.1
p = 0.6
h_size = 1000
delta = 0.05

bound = hoeffding_two_sided_bound(n, eps)
empirical = estimate_tail_frequency(p, n, eps)
needed_n = sample_complexity(eps, delta, h_size)

print("Hoeffding bound:", bound)
print("Empirical tail frequency:", empirical)
print("Needed n for finite H:", needed_n)

assert 0 <= empirical <= 1
assert abs(bound - 2 * math.exp(-2)) < 1e-12
assert needed_n == math.ceil(math.log(2 * h_size / delta) / (2 * eps * eps))
assert finite_class_bound(needed_n, eps, h_size) <= delta
```

| 输入参数 | 公式 | 输出含义 |
|---|---|---|
| $n,\varepsilon$ | $2e^{-2n\varepsilon^2}$ | 固定均值偏差的双侧概率上界 |
| $n,\varepsilon,|H|$ | $2|H|e^{-2n\varepsilon^2}$ | 有限假设类整体偏差上界 |
| $\varepsilon,\delta,|H|$ | $\ln(2|H|/\delta)/(2\varepsilon^2)$ | 保证失败概率不超过 $\delta$ 的样本量下界 |
| $c_1,\dots,c_n,t$ | $e^{-2t^2/\sum c_i^2}$ | McDiarmid 的函数偏差上界 |

数值例子：取 $X_i\sim\mathrm{Bernoulli}(0.6)$，$n=100$，$\varepsilon=0.1$：

$$
P(|\bar X_n-0.6|\ge0.1)
\le 2e^{-2\cdot100\cdot0.1^2}
=2e^{-2}\approx0.2707
$$

模拟得到的经验频率通常会小于这个上界。上界保守是正常现象，集中不等式的目标是提供可靠保证，不是精确估计真实概率。

---

## 工程权衡与常见坑

真实工程例子：A/B 实验中，一个用户是否转化通常记为 $0/1$。如果用户独立进入实验，单个样本有界，那么某个版本的经验转化率可以用 Hoeffding 给出偏差上界。比较两个版本时，可以分别控制两个均值的偏差，或直接分析差值的集中性。

但工程数据经常不满足理想前提。推荐系统里的用户样本可能来自同一会话，广告实验可能存在流量分配偏差，日志数据可能有去重错误。这些都会破坏独立性。

使用顺序应当是：先检查独立性，再检查有界性，再判断对象是“和”还是“一般函数”。固定假设用 Hoeffding，有限假设类要加 union bound，复杂函数考虑 McDiarmid。

McDiarmid 的关键量是 $\sum c_i^2$。如果每个样本影响都是 $1/n$，则 $\sum c_i^2=n\cdot(1/n^2)=1/n$，界会随样本量变强。反例是：

$$
f(x_1,\dots,x_n)=x_1
$$

此时第一个样本的影响 $c_1=1$，其他 $c_i=0$，所以 $\sum c_i^2=1$，不会随 $n$ 变小。也就是说，虽然输入有 $n$ 个，但函数只依赖一个点，集中界不会因为样本数增加而改善。

| 常见坑 | 错误做法 | 后果 | 修正方式 |
|---|---|---|---|
| 独立性破坏 | 把时间序列当 iid 样本 | 偏差概率被低估 | 使用混合条件、Azuma、Freedman 或重采样方法 |
| union bound 漏用 | 固定 $h$ 的界套到整个 $H$ | 泛化保证过强 | 加 $|H|$、VC 维或 Rademacher complexity |
| 次高斯误解 | 把“方差小”当作“次高斯” | 重尾风险被忽略 | 检查 mgf 或使用重尾工具 |
| 重权重样本 | 忽略单样本大影响 | $\sum c_i^2$ 变大，界变弱 | 限权、截断、重设指标 |
| 边界过松但仍有用 | 期待上界接近真实概率 | 误解理论用途 | 把它作为最坏情况保证 |

---

## 替代方案与适用边界

Hoeffding 和 McDiarmid 适合“独立、有界、影响可控”的场景。条件不满足时，不能只换个名字继续套公式。

| 场景 | 是否独立 | 是否有界 | 常用工具 | 备注 |
|---|---|---|---|---|
| 有界独立和 | 是 | 是 | Hoeffding | 转化率、固定模型损失 |
| 坐标有界影响的函数 | 输入独立 | 函数差分有界 | McDiarmid | 复杂聚合指标 |
| 方差也想纳入界 | 是 | 常见为有界或轻尾 | Bernstein、Bennett | 通常比 Hoeffding 更细 |
| 在线过程、鞅差分 | 不要求 iid | 增量受控 | Azuma、Freedman | 适合自适应过程 |
| 马尔可夫链或相关序列 | 否 | 视条件而定 | 混合集中界、链上 concentration | 需要相关性条件 |
| 重尾变量 | 可独立 | 无界或尾重 | 截断均值、median-of-means | 先稳健化再给界 |
| 整个假设类泛化 | 样本通常独立 | 损失有界或次高斯 | union bound、VC、Rademacher、covering number | 控制模型复杂度 |

如果目标只是固定模型的泛化误差，Hoeffding 已经能给出清晰边界。如果目标是“训练算法从很多模型中选一个”，固定 $h$ 的界不够，因为选择过程本身会利用数据。此时必须控制假设类复杂度。

有限假设类可以用 union bound：

$$
2|H|e^{-2n\varepsilon^2}
$$

无限假设类则通常需要 VC 维、Rademacher complexity 或 covering number。它们的共同目的都是把“模型选择空间有多大”纳入概率界。

---

## 参考资料

| 分组 | 来源 | 适合查什么 |
|---|---|---|
| 原始论文 | Hoeffding (1963), *Probability Inequalities for Sums of Bounded Random Variables*, https://doi.org/10.1080/01621459.1963.10500830 | Hoeffding 不等式的原始陈述与有界和形式 |
| 原始论文 | McDiarmid (1989), *On the Method of Bounded Differences*, https://doi.org/10.1017/CBO9781107359949.008 | bounded differences 方法与 McDiarmid 不等式 |
| 课程讲义 | CMU 36-465/665, *Conceptual Foundations of Statistical Learning*, https://www.stat.cmu.edu/~cshalizi/sml/21/ | 统计学习中泛化界和集中不等式的直观解释 |
| 课程讲义 | MIT OCW 18.465 Lecture 7, *Hoeffding’s Inequality*, https://ocw.mit.edu/courses/18-465-topics-in-statistics-statistical-learning-theory-spring-2007/resources/l7/ | Chernoff bound 到 Hoeffding 的推导流程 |
| 形式化文档 | Lean Mathlib: Sub-Gaussian random variables, https://leanprover-community.github.io/mathlib4_docs/Mathlib/Probability/Moments/SubGaussian.html | 次高斯随机变量的形式化定义与相关性质 |
