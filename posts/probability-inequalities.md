## 核心结论

概率不等式解决的是同一个问题：随机结果会偏离期望值多少，以及这种偏离有多不可能。

三条最常用的结论可以先记成一张表：

| 不等式 | 典型形式 | 需要的前提 | 上界特点 |
|---|---|---|---|
| 切比雪夫不等式 | $\Pr(|X-\mu|\ge t)\le \dfrac{\sigma^2}{t^2}$ | 只要求均值 $\mu$ 和方差 $\sigma^2$ 有限 | 分布无关，但通常较松 |
| 霍夫丁不等式 | $\Pr\!\left(\sum_{i=1}^n(X_i-\mathbb E X_i)\ge t\right)\le \exp\!\left(-\dfrac{2t^2}{\sum_{i=1}^n(b_i-a_i)^2}\right)$ | 独立，且每个变量都落在区间 $[a_i,b_i]$ | 指数衰减，适合均值集中 |
| 伯恩斯坦不等式 | $\Pr\!\left(\sum_{i=1}^n(X_i-\mathbb E X_i)\ge t\right)\le \exp\!\left(-\dfrac{t^2}{2\sigma^2+\frac23 Mt}\right)$ | 独立、有界，且知道总方差 $\sigma^2=\sum_{i=1}^n \mathrm{Var}(X_i)$，同时 $|X_i-\mathbb E X_i|\le M$ | 同时利用范围和方差，方差小时更紧 |

这里的“尾概率”是指随机变量落入极端区域的概率，也就是分布尾部的概率。

先看最小例子。若一个随机变量满足 $\mathbb E X=50,\ \mathrm{Var}(X)=25$，则标准差 $\sigma=5$。当我们关心它偏离均值 10 以上的概率时，

$$
\Pr(|X-50|\ge 10)\le \frac{25}{10^2}=0.25.
$$

这不是精确概率，而是一个保守上界。意思是：无论分布细节如何，只要方差是 25，这个概率就不会超过 0.25。

实际使用时可以按已知信息选工具：

| 已知信息 | 推荐不等式 | 原因 |
|---|---|---|
| 只知道方差 | 切比雪夫 | 信息太少，只能用分布自由界 |
| 知道独立且每项有上下界 | 霍夫丁 | 能得到指数型尾界 |
| 还知道总体方差不大 | 伯恩斯坦 | 小偏差区间通常比霍夫丁更紧 |

还可以把三者的区别记成一句话：

$$
\text{信息越少，界越稳但越松；信息越多，界通常越紧。}
$$

---

## 问题定义与边界

统一记号如下：

| 符号 | 含义 |
|---|---|
| $X$ | 单个随机变量 |
| $X_1,\dots,X_n$ | 一组随机变量 |
| $\mu$ | 期望，长期平均值 |
| $\sigma^2$ | 方差，波动强度 |
| $t$ | 偏离阈值 |
| $\varepsilon$ | 均值偏离阈值 |
| $[a_i,b_i]$ | 第 $i$ 个变量可能落入的范围 |
| $M$ | 单个中心化变量的最大绝对偏差上界 |
| $S_n$ | 中心化和，$S_n=\sum_{i=1}^n (X_i-\mathbb E X_i)$ |
| $\bar X$ | 样本均值，$\bar X=\dfrac1n\sum_{i=1}^n X_i$ |

概率不等式通常在估计下面两类量：

$$
\Pr(|X-\mu|\ge t)
\quad \text{或} \quad
\Pr\!\left(\left|\frac1n\sum_{i=1}^n X_i-\mu\right|\ge \varepsilon\right).
$$

第一类问题关心单个随机变量是否会出现极端偏差，第二类问题关心样本均值是否会远离真实均值。统计学习、A/B 测试、异常检测、监控报警、置信区间，本质上都在处理这两类问题。

一个对新手最直观的例子是抛硬币。设 $X_i\in\{0,1\}$，正面记为 1，反面记为 0。若硬币公正，则

$$
\mathbb E[X_i]=0.5,\qquad \mathrm{Var}(X_i)=0.25.
$$

抛 10 次时，样本均值是

$$
\bar X=\frac1{10}\sum_{i=1}^{10}X_i.
$$

如果关心“平均值至少高出期望 0.2”，那就是

$$
\Pr(\bar X-0.5\ge 0.2).
$$

这个事件等价于

$$
\bar X\ge 0.7
\iff
\sum_{i=1}^{10}X_i\ge 7.
$$

也就是问：10 次中出现至少 7 个正面的概率有多大。

对这个问题，三条不等式的使用前提不同：

| 方法 | 需要什么信息 | 能说什么 |
|---|---|---|
| 切比雪夫 | 只要知道 $\mathrm{Var}(\bar X)$ | 能给保守上界 |
| 霍夫丁 | 硬币独立，且 $X_i\in[0,1]$ | 能给指数型上界 |
| 伯恩斯坦 | 除了独立有界，还知道 $\mathrm{Var}(X_i)=0.25$ | 能进一步利用真实波动强度 |

这里必须把边界说清楚。

| 不等式 | 能处理什么 | 不能自动处理什么 |
|---|---|---|
| 切比雪夫 | 任意分布，只要方差有限 | 不会自动利用独立性、有界性、子高斯性 |
| 霍夫丁 | 独立有界变量之和 | 重尾变量、强依赖样本 |
| 伯恩斯坦 | 独立有界变量，且方差可控 | 未知方差、范围不清、重尾爆点 |

如果前提错了，结论就可能失效。比如时间序列、在线学习、强化学习中的样本常常相关，这时直接套用霍夫丁或伯恩斯坦通常没有理论保证。

---

## 核心机制与推导

切比雪夫、霍夫丁、伯恩斯坦都在控制“偏差事件”的概率，但它们利用的信息层次不同。

| 不等式 | 本质上利用了什么 |
|---|---|
| 切比雪夫 | 只利用二阶矩，也就是方差 |
| 霍夫丁 | 利用独立性和有界性 |
| 伯恩斯坦 | 利用独立性、有界性、方差信息 |

### 1. 切比雪夫：用平方偏差控制尾概率

切比雪夫的出发点最简单。方差定义为

$$
\mathrm{Var}(X)=\mathbb E[(X-\mu)^2].
$$

一旦事件 $|X-\mu|\ge t$ 发生，就必然有

$$
(X-\mu)^2\ge t^2.
$$

于是

$$
(X-\mu)^2 \ge t^2 \cdot \mathbf 1_{\{|X-\mu|\ge t\}},
$$

对两边取期望，得到

$$
\mathbb E[(X-\mu)^2]
\ge
t^2 \Pr(|X-\mu|\ge t).
$$

也就是

$$
\Pr(|X-\mu|\ge t)\le \frac{\mathrm{Var}(X)}{t^2}
=\frac{\sigma^2}{t^2}.
$$

这个推导短，但含义很重要：切比雪夫没有用到分布形状，只用到了“平方偏差的平均值有限”。因此它极稳，但通常偏松。

对于样本均值 $\bar X$，如果 $X_1,\dots,X_n$ 独立同分布，且 $\mathrm{Var}(X_i)=\sigma^2$，那么

$$
\mathrm{Var}(\bar X)=\frac{\sigma^2}{n}.
$$

于是切比雪夫给出

$$
\Pr(|\bar X-\mu|\ge \varepsilon)\le \frac{\sigma^2}{n\varepsilon^2}.
$$

它已经体现出“大数平均以后更稳定”，因为上界随 $n$ 以 $1/n$ 缩小，但这仍然是多项式速度，而不是指数速度。

### 2. 霍夫丁：用有界性换指数衰减

霍夫丁的关键不是方差，而是矩母函数控制。思路是：

1. 把尾概率转成指数函数的期望；
2. 用独立性把和的期望拆开；
3. 用有界性控制每一项的矩母函数；
4. 再对参数做优化。

形式上，设

$$
S_n=\sum_{i=1}^n(X_i-\mathbb E X_i).
$$

对任意 $\lambda>0$，由 Markov 不等式，

$$
\Pr(S_n\ge t)
=
\Pr(e^{\lambda S_n}\ge e^{\lambda t})
\le
e^{-\lambda t}\mathbb E[e^{\lambda S_n}].
$$

若 $X_i$ 独立，则

$$
\mathbb E[e^{\lambda S_n}]
=
\prod_{i=1}^n \mathbb E\!\left[e^{\lambda (X_i-\mathbb E X_i)}\right].
$$

接下来用 Hoeffding 引理：若 $X_i\in[a_i,b_i]$，则

$$
\mathbb E\!\left[e^{\lambda (X_i-\mathbb E X_i)}\right]
\le
\exp\!\left(\frac{\lambda^2(b_i-a_i)^2}{8}\right).
$$

代回去可得

$$
\Pr(S_n\ge t)
\le
\exp\!\left(
-\lambda t + \frac{\lambda^2}{8}\sum_{i=1}^n(b_i-a_i)^2
\right).
$$

再对 $\lambda$ 取最优值，得到

$$
\Pr(S_n\ge t)\le
\exp\!\left(
-\frac{2t^2}{\sum_{i=1}^n(b_i-a_i)^2}
\right).
$$

这就是霍夫丁不等式。

如果要两侧尾界，直接对 $S_n$ 和 $-S_n$ 各做一次，再用并集界：

$$
\Pr(|S_n|\ge t)
\le
2\exp\!\left(
-\frac{2t^2}{\sum_{i=1}^n(b_i-a_i)^2}
\right).
$$

对均值形式，若 $X_i\in[a,b]$ 独立同分布，则

$$
\Pr(\bar X-\mu\ge \varepsilon)
\le
\exp\!\left(
-\frac{2n\varepsilon^2}{(b-a)^2}
\right),
$$

以及

$$
\Pr(|\bar X-\mu|\ge \varepsilon)
\le
2\exp\!\left(
-\frac{2n\varepsilon^2}{(b-a)^2}
\right).
$$

这说明均值的偏差概率会随样本数 $n$ 指数下降。

### 3. 伯恩斯坦：在有界之外再利用方差信息

伯恩斯坦继续沿用“指数化 + 优化参数”的路线，但它比霍夫丁多利用了一层信息：真实方差可能远小于“最坏范围”暗示的波动规模。

设

$$
S_n=\sum_{i=1}^n(X_i-\mathbb E X_i),
\qquad
\sigma^2=\sum_{i=1}^n \mathrm{Var}(X_i),
$$

并且满足

$$
|X_i-\mathbb E X_i|\le M.
$$

伯恩斯坦不等式给出

$$
\Pr(S_n\ge t)\le
\exp\left(
-\frac{t^2}{2\sigma^2+\frac23Mt}
\right).
$$

两侧版本同样是

$$
\Pr(|S_n|\ge t)\le
2\exp\left(
-\frac{t^2}{2\sigma^2+\frac23Mt}
\right).
$$

这个分母

$$
2\sigma^2+\frac23Mt
$$

非常值得记住，因为它对应两个来源：

| 项 | 含义 |
|---|---|
| $2\sigma^2$ | 真实累计波动强度 |
| $\frac23Mt$ | 单个变量最大跳动带来的校正项 |

因此伯恩斯坦有两个典型工作区间。

当 $t$ 较小时，$\frac23Mt$ 相对不重要，主导项近似为

$$
\exp\!\left(-\frac{t^2}{2\sigma^2}\right).
$$

这时它像“方差驱动”的高斯型衰减。

当 $t$ 很大时，$\frac23Mt$ 不可忽略，尾界会逐渐体现出有界变量的线性修正。

### 4. 用硬币例子把三者放在一起

仍然考虑 10 次公正硬币，$X_i\in\{0,1\}$，$\mathbb E X_i=0.5$，$\mathrm{Var}(X_i)=0.25$。

我们关心

$$
\Pr(\bar X-0.5\ge 0.2).
$$

这等价于

$$
\sum_{i=1}^{10}(X_i-0.5)\ge 10\times 0.2 = 2.
$$

#### 切比雪夫

先算样本均值方差：

$$
\mathrm{Var}(\bar X)=\frac{0.25}{10}=0.025.
$$

于是

$$
\Pr(|\bar X-0.5|\ge 0.2)\le \frac{0.025}{0.2^2}=0.625.
$$

因为右尾事件包含在双侧事件中，所以也有

$$
\Pr(\bar X-0.5\ge 0.2)\le 0.625.
$$

这个界成立，但很松。

#### 霍夫丁

每个 $X_i\in[0,1]$，所以 $\sum_{i=1}^{10}(b_i-a_i)^2=10$。代入得到

$$
\Pr(\bar X-0.5\ge 0.2)
\le
\exp\left(-\frac{2\cdot 2^2}{10}\right)
=
e^{-0.8}
\approx 0.4493.
$$

#### 伯恩斯坦

总方差为

$$
\sigma^2=10\times 0.25=2.5.
$$

又因为

$$
|X_i-\mathbb E X_i|=|X_i-0.5|\le 0.5,
$$

所以 $M=0.5$。代入得

$$
\Pr(\bar X-0.5\ge 0.2)
\le
\exp\left(
-\frac{2^2}{2\times 2.5+\frac23\times 0.5\times 2}
\right)
=
\exp\left(-\frac{4}{5+\frac23}\right)
\approx 0.4936.
$$

在这个例子里，霍夫丁比伯恩斯坦更紧。这并不矛盾。因为 10 次样本太少，$\varepsilon=0.2$ 也不算一个“小偏差”，伯恩斯坦的方差优势没有表现出来。

为了避免误解，可以把这件事总结成一句更准确的话：

$$
\text{伯恩斯坦并不是总比霍夫丁紧，而是在方差显著小于最坏范围时经常更紧。}
$$

### 5. 为什么工程里仍然重视伯恩斯坦

真实工程中，很多指标会被裁剪到 $[0,1]$，例如：

- 单样本损失做 clipping；
- CTR、转化率类指标天然位于 $[0,1]$；
- 归一化后的风险分数被截断到固定范围。

如果两个模型都满足有界，但其中一个模型的单样本方差明显更小，那么只用霍夫丁会忽略这部分结构信息；伯恩斯坦会更敏感地反映“虽然最坏范围一样，但实际波动更小”。

这也是它常出现在泛化误差分析、经验风险上界、在线决策理论中的原因。

---

## 代码实现

工程上最实用的做法不是死背公式，而是把几条上界统一成一个接口：输入方差、变量范围、样本数和偏差阈值，输出多个可比较的上界。

下面这段 Python 代码可以直接运行，不依赖第三方库。它同时实现：

1. 切比雪夫单变量与均值形式；
2. 霍夫丁的和形式与均值形式；
3. 伯恩斯坦的和形式与均值形式；
4. 10 次公正硬币例子的精确概率，便于和上界比较。

```python
import math


def chebyshev_bound(var: float, t: float) -> float:
    """
    Upper bound for P(|X - E[X]| >= t) using Chebyshev inequality.
    """
    if var < 0:
        raise ValueError("var must be >= 0")
    if t <= 0:
        raise ValueError("t must be > 0")
    return min(1.0, var / (t * t))


def chebyshev_mean_bound(single_var: float, n: int, eps: float) -> float:
    """
    Upper bound for P(|mean - E[mean]| >= eps) for i.i.d. variables.
    Var(mean) = single_var / n.
    """
    if single_var < 0:
        raise ValueError("single_var must be >= 0")
    if n <= 0:
        raise ValueError("n must be > 0")
    if eps <= 0:
        raise ValueError("eps must be > 0")
    return chebyshev_bound(single_var / n, eps)


def hoeffding_sum_bound(t: float, bounds: list[tuple[float, float]]) -> float:
    """
    Upper bound for P(sum(X_i - E[X_i]) >= t) using Hoeffding inequality.
    bounds: [(a1, b1), ..., (an, bn)]
    """
    if t < 0:
        raise ValueError("t must be >= 0")
    if not bounds:
        raise ValueError("bounds must be non-empty")
    width_sq_sum = 0.0
    for a, b in bounds:
        if a > b:
            raise ValueError("each bound must satisfy a <= b")
        width_sq_sum += (b - a) ** 2
    if t == 0:
        return 1.0
    if width_sq_sum == 0:
        return 0.0
    exponent = -2.0 * (t ** 2) / width_sq_sum
    return min(1.0, math.exp(exponent))


def hoeffding_mean_bound(eps: float, n: int, a: float, b: float) -> float:
    """
    Upper bound for P(mean - E[mean] >= eps) for i.i.d. variables in [a, b].
    """
    if n <= 0:
        raise ValueError("n must be > 0")
    if eps < 0:
        raise ValueError("eps must be >= 0")
    return hoeffding_sum_bound(n * eps, [(a, b)] * n)


def bernstein_sum_bound(t: float, total_var: float, M: float) -> float:
    """
    Upper bound for P(sum(X_i - E[X_i]) >= t) using Bernstein inequality.
    total_var = sum Var(X_i)
    M = max_i |X_i - E[X_i]| upper bound
    """
    if t < 0:
        raise ValueError("t must be >= 0")
    if total_var < 0:
        raise ValueError("total_var must be >= 0")
    if M < 0:
        raise ValueError("M must be >= 0")
    if t == 0:
        return 1.0
    denom = 2.0 * total_var + (2.0 / 3.0) * M * t
    if denom == 0:
        return 0.0
    exponent = -(t ** 2) / denom
    return min(1.0, math.exp(exponent))


def bernstein_mean_bound(eps: float, n: int, single_var: float, M: float) -> float:
    """
    Upper bound for P(mean - E[mean] >= eps) for i.i.d. variables.
    """
    if n <= 0:
        raise ValueError("n must be > 0")
    if eps < 0:
        raise ValueError("eps must be >= 0")
    total_var = n * single_var
    return bernstein_sum_bound(n * eps, total_var, M)


def binomial_tail_prob_at_least(n: int, p: float, k: int) -> float:
    """
    Exact probability P(Binomial(n, p) >= k).
    Used here to compare true probability with upper bounds.
    """
    if not (0.0 <= p <= 1.0):
        raise ValueError("p must be in [0, 1]")
    if n < 0:
        raise ValueError("n must be >= 0")
    if k <= 0:
        return 1.0
    if k > n:
        return 0.0

    total = 0.0
    for j in range(k, n + 1):
        total += math.comb(n, j) * (p ** j) * ((1 - p) ** (n - j))
    return total


def main() -> None:
    # Example 1: Chebyshev on a single variable
    p_cheb_single = chebyshev_bound(var=25.0, t=10.0)

    # Example 2: 10 fair coin tosses, mean exceeds expectation by 0.2
    n = 10
    eps = 0.2

    # Exact event: mean >= 0.7  <=>  number of heads >= 7
    p_exact = binomial_tail_prob_at_least(n=n, p=0.5, k=7)

    # Bounds
    p_cheb_mean = chebyshev_mean_bound(single_var=0.25, n=n, eps=eps)
    p_hoeff = hoeffding_mean_bound(eps=eps, n=n, a=0.0, b=1.0)
    p_bern = bernstein_mean_bound(eps=eps, n=n, single_var=0.25, M=0.5)

    print("Single-variable Chebyshev bound:", round(p_cheb_single, 6))
    print("Exact coin-tail probability     :", round(p_exact, 6))
    print("Chebyshev mean bound            :", round(p_cheb_mean, 6))
    print("Hoeffding mean bound            :", round(p_hoeff, 6))
    print("Bernstein mean bound            :", round(p_bern, 6))


if __name__ == "__main__":
    main()
```

这段代码输出的数值关系应接近：

| 量 | 数值 |
|---|---|
| 单变量切比雪夫示例 | $0.25$ |
| 10 次硬币精确概率 $\Pr(\bar X\ge 0.7)$ | $0.171875$ |
| 切比雪夫均值上界 | $0.625$ |
| 霍夫丁上界 | $\approx 0.449329$ |
| 伯恩斯坦上界 | $\approx 0.493575$ |

这个结果对新手很重要，因为它展示了三件事：

1. 不等式给的是上界，不是精确概率；
2. 上界可以明显大于真实值；
3. 更复杂的不等式也不保证在每个有限样本例子里都更紧。

工程里常见的封装方式如下：

| 函数 | 输入 | 输出 |
|---|---|---|
| `chebyshev_bound` | 方差 `var`，阈值 `t` | 单变量双侧尾概率上界 |
| `chebyshev_mean_bound` | 单项方差、样本数 `n`、均值偏差 `eps` | 均值双侧尾概率上界 |
| `hoeffding_sum_bound` | 和的偏差 `t`，每项区间 `[(a_i,b_i)]` | 和的右尾上界 |
| `hoeffding_mean_bound` | 均值偏差 `eps`，样本数 `n`，区间 `[a,b]` | 均值右尾上界 |
| `bernstein_sum_bound` | 和的偏差 `t`，总方差 `total_var`，最大偏差 `M` | 和的右尾上界 |
| `bernstein_mean_bound` | 均值偏差 `eps`，样本数 `n`，单项方差，`M` | 均值右尾上界 |
| `binomial_tail_prob_at_least` | 二项分布参数与阈值 | 精确尾概率 |

真正落地时，通常还会多做两步预处理：

1. 先把原始指标裁剪、截断或归一化到固定区间，例如 $[0,1]$；
2. 再根据是否有可靠的方差估计，决定用霍夫丁还是伯恩斯坦。

可以把工程决策简化成：

$$
\text{先保证前提成立，再比较上界谁更适合当前任务。}
$$

---

## 工程权衡与常见坑

切比雪夫最大的优点是稳，最大的缺点也是稳。它几乎不挑分布，因此几乎总能用；但也因为它不使用更多结构信息，所以经常给不出足够紧的结论。

实际工作中更常见的问题不是“不会套公式”，而是“在前提不成立时硬套公式”。

下面是最常见的误用：

| 常见错用 | 结果 | 规避策略 |
|---|---|---|
| 明明数据有强依赖，还套用霍夫丁 | 上界失效，风险被低估 | 改用 Azuma、Freedman 等适用于鞅差分或依赖序列的工具 |
| 未验证变量是否有统一上界，就套用伯恩斯坦 | 极端值被忽略，尾部被系统性低估 | 先裁剪、截断或重标定，再明确写出 $M$ |
| 用样本方差直接替代理论方差，却忽略估计误差 | 上界看起来更紧，但不再保守 | 使用经验 Bernstein 界，或给方差估计加置信修正 |
| 把右尾界误当成双侧界 | 概率少算一倍量级 | 明确区分 $\Pr(S_n\ge t)$ 和 $\Pr(|S_n|\ge t)$ |
| 把中心化上界错用到原变量上 | 参数解释错误 | 先写清楚是 $X_i$ 还是 $X_i-\mathbb E X_i$ |
| 用切比雪夫做精细模型比较 | 差异被松界掩盖 | 若独立有界，优先考虑霍夫丁或伯恩斯坦 |

### 1. 依赖性是最常见的理论漏洞

很多数据不是独立样本。例如：

- 时间序列的相邻观测相关；
- 在线学习中样本由策略历史决定；
- 强化学习中状态、动作、回报连续耦合；
- 监控系统中的窗口统计彼此重叠。

这时如果直接使用独立样本和的 Hoeffding/Bernstein，上界可能漂亮但没有理论意义。若过程更接近鞅差分结构，通常应转向 Azuma-Hoeffding 或 Freedman。

### 2. 伯恩斯坦里的 $M$ 常被写错

很多人会把 $M$ 误解为“平均偏差”或“标准差级别的波动”。这都不对。

伯恩斯坦要求的是

$$
|X_i-\mathbb E X_i|\le M
$$

几乎处处成立。它是一个最大绝对偏差上界，不是均值，不是方差，也不是经验分位数。

例如若 $X_i\in[0,1]$，且 $\mathbb E X_i=0.2$，那么可以写

$$
|X_i-\mathbb E X_i|\le \max(0.2, 0.8)=0.8.
$$

若直接偷懒写成 $M=0.2$，结论就会系统性过于乐观。

### 3. 样本方差不是理论方差

在理论公式里，伯恩斯坦使用的是总方差

$$
\sigma^2=\sum_{i=1}^n \mathrm{Var}(X_i).
$$

但在真实数据里，我们通常只有样本方差估计值。此时如果把估计值直接代入标准伯恩斯坦公式，得到的数值未必仍然是严格上界。

这时有两种更稳妥的做法：

| 做法 | 含义 |
|---|---|
| 用经验 Bernstein 界 | 直接使用针对样本方差构造的版本 |
| 对方差估计再加置信修正 | 先控制“估计方差偏小”的风险，再代入上界 |

### 4. 右尾、左尾、双侧必须分清

很多工程指标真正关心的是双侧偏差，例如：

$$
|\bar X-\mu|\ge \varepsilon.
$$

但有些任务只关心单侧，例如：

- 风险是否超过阈值；
- 坏率是否高于基线；
- 延迟是否显著升高。

如果实际关心的是双侧，却只套用了右尾版本，那么常常会少一个系数 2。对大样本指数界来说这个系数可能不决定数量级，但在严谨分析里必须区分清楚。

---

## 替代方案与适用边界

切比雪夫、霍夫丁、伯恩斯坦不是全部。它们是入门最重要的一组，但不是每个场景的最优工具。

| 方法 | 适用前提 | 典型衰减形态 | 适合场景 |
|---|---|---|---|
| 切比雪夫 | 仅有限方差 | 多项式型，$\sim 1/t^2$ | 信息极少时的保底分析 |
| 霍夫丁 | 独立、有界 | 指数型 | 二分类、裁剪损失、均值集中 |
| 伯恩斯坦 | 独立、有界、已知或可控方差 | 指数型，小偏差更优 | 小方差学习问题、经验风险分析 |
| Chernoff / 次高斯界 | 矩母函数受控，或变量是子高斯 | 高质量指数型 | 高斯噪声、线性模型、理论推导 |
| Bennett | 独立、有界、方差可控 | 常比 Bernstein 更精细 | 理论分析中追求更细常数 |
| Azuma / Freedman | 鞅差分或依赖序列 | 指数型 | 在线算法、时间相关数据 |
| McDiarmid | 函数对单样本扰动不敏感 | 指数型 | 稳定性分析、泛化证明 |
| Rademacher / Gaussian complexity | 关注函数类复杂度 | 与模型类规模相关 | 统计学习理论中的泛化分析 |

对新手来说，可以把这几类方法理解成不同层次的问题：

| 你知道什么 | 该考虑什么工具 |
|---|---|
| 只知道均值和方差存在 | 切比雪夫 |
| 知道每个样本一定落在固定区间 | 霍夫丁 |
| 还知道真实波动其实不大 | 伯恩斯坦 |
| 知道尾部近似高斯 | Chernoff / 次高斯工具 |
| 样本并不独立，而是逐步产生 | Azuma / Freedman |
| 关心的不是一个均值，而是整个函数类 | 复杂度工具 |

实际工程判断可以按下面顺序做：

1. 先问样本是否近似独立。
2. 再问变量是否有明确上界或已做裁剪。
3. 再问方差信息是否可靠。
4. 如果前三条不成立，就不要强行用 Hoeffding/Bernstein。
5. 如果任务是学习理论中的泛化问题，还要进一步看函数类复杂度，而不只是单个随机和。

这里顺带说明“子高斯”这个术语。它的直观含义是：随机变量尾部像高斯那样快地衰减。一个常见的定义方式是存在常数 $K>0$，使得对任意 $\lambda\in\mathbb R$，

$$
\mathbb E\left[e^{\lambda (X-\mathbb E X)}\right]
\le
\exp\left(\frac{K^2\lambda^2}{2}\right).
$$

若这个条件成立，Chernoff 方法通常会给出比单纯区间控制更自然的结果。霍夫丁可以看作“有界变量一定是子高斯”的一个具体推论。

---

## 参考资料

1. W. Hoeffding, “Probability Inequalities for Sums of Bounded Random Variables”, *Journal of the American Statistical Association*, 1963.  
这是霍夫丁不等式的经典原始文献，适合查标准假设、标准常数和原始证明结构。

2. S. Bernstein, 关于有界独立随机变量和的概率不等式的经典文献。  
不同教材对伯恩斯坦不等式的表述略有差异，但核心形式都是利用“有界性 + 方差”来控制尾概率。

3. Dimitri P. Bertsekas, John N. Tsitsiklis, *Introduction to Probability*.  
适合入门后系统复习切比雪夫、Markov、Chernoff、Hoeffding 等不等式之间的关系。

4. Roman Vershynin, *High-Dimensional Probability*.  
适合把切比雪夫、Hoeffding、Bernstein、sub-Gaussian、sub-exponential 放到统一框架下理解，是从入门走向现代集中不等式的好材料。

5. Stéphane Boucheron, Gábor Lugosi, Pascal Massart, *Concentration Inequalities: A Nonasymptotic Theory of Independence*.  
这是集中不等式方向的标准参考书，系统覆盖 Hoeffding、Bernstein、Bennett、McDiarmid 以及经验 Bernstein 等结果。

6. Wikipedia: “Chebyshev's inequality”, “Hoeffding's inequality”, “Bernstein inequalities (probability theory)”.  
适合快速查标准公式、不同记号写法和相关条目的跳转关系，但应以教材或原始文献为准。

7. Olivier Catoni 等关于 PAC-Bayes 与 Bernstein 型不等式的论文。  
如果要进一步理解伯恩斯坦不等式在泛化误差、快收敛分析和 PAC-Bayes 界中的作用，这一方向很值得继续读。

参考资料可以按下面顺序读：

| 阅读阶段 | 建议材料 |
|---|---|
| 第一次接触 | Wikipedia + 概率论教材中的 Markov/Chebyshev/Hoeffding 章节 |
| 需要真正理解机制 | Bertsekas-Tsitsiklis 或 Vershynin |
| 需要做理论推导 | Boucheron-Lugosi-Massart |
| 需要做学习理论应用 | Catoni、PAC-Bayes、经验 Bernstein 相关论文 |
