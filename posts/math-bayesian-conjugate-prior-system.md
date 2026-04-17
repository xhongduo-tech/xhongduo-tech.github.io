## 核心结论

共轭先验指的是：在同一个似然函数下，先验分布和后验分布属于同一分布族。更具体地说，若参数 $\theta$ 的先验是某个分布族，观测数据 $\mathcal D$ 之后，后验 $p(\theta\mid \mathcal D)$ 仍落在这个分布族内，只是参数发生了更新。

它的核心价值不是“更高级”，而是“更省计算”。一般贝叶斯推断需要计算归一化常数：
$$
p(\theta\mid x)=\frac{p(x\mid \theta)p(\theta)}{\int p(x\mid \theta)p(\theta)\,d\theta},
$$
分母往往很难直接求。共轭结构的好处是：后验形式可以直接写出来，更新过程从“做积分”变成“改参数”。

对许多指数族模型，更新规则都可以写成：
$$
\text{后验参数}=\text{先验参数}+\text{数据的充分统计量}.
$$
这里的“充分统计量”可以理解为：把样本压缩成一小组数字，同时不丢失与参数推断相关的关键信息。  
例如在 Bernoulli 模型里，原始数据可能是一长串 $0/1$，但对参数 $\theta$ 来说，真正需要保留的只是：

| 观测数据 | 充分统计量 |
|---|---|
| $x_1,\dots,x_n\in\{0,1\}$ | 成功次数 $s=\sum x_i$，失败次数 $f=n-s$ |

最常见的入门例子是 Beta-Bernoulli。设点击率 $\theta$ 的先验为
$$
\theta\sim \text{Beta}(2,2),
$$
观察到 3 次点击成功、1 次失败，则后验为
$$
\theta\mid \mathcal D \sim \text{Beta}(2+3,\;2+1)=\text{Beta}(5,3).
$$
后验均值为
$$
\mathbb E[\theta\mid \mathcal D]=\frac{5}{5+3}=0.625.
$$

这个例子说明了共轭先验最重要的工程意义：更新不需要重新做积分，也不需要保存所有原始样本。你只需要保存当前参数，然后把新数据对应的统计量加进去。

下面先给出直观对比：

| 方式 | 后验计算方式 | 计算代价 | 流式更新是否方便 | 典型场景 |
|---|---|---:|---|---|
| 非共轭 | 需要积分、采样或数值近似 | 高 | 一般 | 复杂模型、灵活先验、层次模型 |
| 共轭 | 参数直接相加或闭式更新 | 低 | 很方便 | CTR、分类计数、已知方差高斯均值 |

再补一个判断标准：

| 你关心的问题 | 共轭先验的回答 |
|---|---|
| 能不能快速在线更新 | 可以 |
| 能不能闭式写出后验 | 通常可以 |
| 能不能表达非常复杂的先验知识 | 通常不行 |
| 能不能替代所有贝叶斯方法 | 不能 |

---

## 问题定义与边界

问题本质是：我们要持续估计一个未知参数，同时不断接收新数据。如果每来一批数据都重新做一遍完整贝叶斯积分，系统开销会很大；如果能把历史信息压缩成少量参数，那么每次更新都能退化为常数时间操作。

一个典型场景是游戏付费率估计。设每个新玩家只有两种结果：

- 付费记为 $1$
- 不付费记为 $0$

这类数据可用 Bernoulli 分布建模：
$$
x_i\sim \text{Bernoulli}(\theta),\qquad x_i\in\{0,1\}.
$$
其中 $\theta$ 是未知付费概率。

若先验取 Beta 分布：
$$
\theta\sim \text{Beta}(\alpha,\beta),
$$
那么在观察到成功 $s$ 次、失败 $f$ 次后，后验仍是 Beta 分布：
$$
\theta\mid \mathcal D \sim \text{Beta}(\alpha+s,\beta+f).
$$

这类方法适合“频繁更新概率估计”的任务：

| 任务 | 观测类型 | 未知参数 | 常见共轭对 | 更新后的参数含义 |
|---|---|---|---|---|
| 点击率 CTR | 点击/未点击 | 点击概率 $\theta$ | Beta - Bernoulli/Binomial | 成功数、失败数累加 |
| 多分类占比 | 类别计数 | 概率向量 $\boldsymbol\pi$ | Dirichlet - Multinomial | 各类别伪计数累加 |
| 已知方差下的均值估计 | 连续值 | 均值 $\mu$ | Gaussian - Gaussian | 均值与精度联合更新 |
| 泊松到达率 | 单位时间事件数 | 速率 $\lambda$ | Gamma - Poisson | 事件总数、曝光时长累加 |

边界也很明确。共轭不是“贝叶斯默认答案”，它成立通常依赖两个条件：

1. 似然通常属于指数族。
2. 先验被选成与该似然匹配的共轭形式。

指数族通式是
$$
f(x\mid \eta)=h(x)\exp\big(\eta^\top T(x)-A(\eta)\big),
$$
其中：

| 符号 | 含义 |
|---|---|
| $\eta$ | 自然参数 |
| $T(x)$ | 充分统计量 |
| $A(\eta)$ | 对数配分函数，用于保证分布可归一化 |
| $h(x)$ | 与参数无关的基底项 |

若先验能写成和这类指数项对齐的形式，就能得到闭式后验；若不能，就需要 MCMC、变分推断、拉普拉斯近似等方法。

所以边界不是“贝叶斯能不能用”，而是：

| 判断问题 | 含义 |
|---|---|
| 是否接受分布族限制 | 为了换取闭式更新，模型表达能力会被限制 |
| 是否更看重速度 | 在线系统、低延迟系统通常更偏向共轭 |
| 是否需要复杂结构 | 若需要层次结构、稀疏先验、多峰后验，共轭通常不够 |

---

## 核心机制与推导

共轭先验体系最重要的不是死记分布对，而是理解为什么“参数相加”会成立。

设样本 $x_1,\dots,x_n$ 独立同分布，且来自指数族：
$$
p(x\mid \eta)=h(x)\exp\big(\eta^\top T(x)-A(\eta)\big).
$$

则全部样本的联合似然为
$$
p(\mathcal D\mid \eta)
=\prod_{i=1}^n h(x_i)\exp\big(\eta^\top T(x_i)-A(\eta)\big).
$$
把指数项整理到一起：
$$
p(\mathcal D\mid \eta)
=\Big(\prod_{i=1}^n h(x_i)\Big)\exp\Big(\eta^\top \sum_{i=1}^n T(x_i)-nA(\eta)\Big).
$$

现在构造一个先验：
$$
\pi(\eta\mid \chi,\nu)\propto \exp\big(\eta^\top\chi-\nu A(\eta)\big).
$$

这里两个超参数可以这样理解：

| 参数 | 直观解释 |
|---|---|
| $\chi$ | 先验中“统计量总和”的位置 |
| $\nu$ | 先验强度，常可理解为伪样本量 |
| 伪样本量 | 不是真实观测，但在更新公式里像已经看过一些样本 |

后验正比于似然乘先验：
$$
\pi(\eta\mid \mathcal D,\chi,\nu)\propto p(\mathcal D\mid \eta)\pi(\eta\mid \chi,\nu).
$$
只保留和 $\eta$ 有关的部分：
$$
\pi(\eta\mid \mathcal D,\chi,\nu)\propto
\exp\Big(
\eta^\top \sum_{i=1}^n T(x_i)-nA(\eta)
\Big)
\exp\big(\eta^\top\chi-\nu A(\eta)\big).
$$
合并指数项：
$$
\pi(\eta\mid \mathcal D,\chi,\nu)\propto
\exp\Big(
\eta^\top\big(\chi+\sum_{i=1}^n T(x_i)\big)-(\nu+n)A(\eta)
\Big).
$$

可以看到，它仍然是同一分布族，只是参数从 $(\chi,\nu)$ 变成了
$$
\chi'=\chi+\sum_{i=1}^n T(x_i),\qquad \nu'=\nu+n.
$$

这就是共轭更新的一般形式。它背后的关键不是魔法，而是一个严格的代数对齐：

| 步骤 | 发生了什么 |
|---|---|
| 似然写成指数族形式 | 数据影响被压缩为 $\sum T(x_i)$ |
| 先验写成匹配形式 | 先验也放在同一个指数模板中 |
| 相乘得到后验 | 指数项直接合并 |
| 后验仍属同族 | 只需更新参数，不必重新积分 |

### 玩具例子：Beta-Bernoulli

Bernoulli 单次观测 $x\in\{0,1\}$ 的似然为
$$
p(x\mid \theta)=\theta^x(1-\theta)^{1-x}.
$$

若有 $n$ 个样本，其中成功次数为
$$
s=\sum_{i=1}^n x_i,
$$
失败次数为
$$
f=n-s,
$$
则似然为
$$
p(\mathcal D\mid \theta)=\theta^s(1-\theta)^f.
$$

Beta 先验写成
$$
p(\theta)=\frac{1}{B(\alpha,\beta)}\theta^{\alpha-1}(1-\theta)^{\beta-1},
$$
其中
$$
B(\alpha,\beta)=\frac{\Gamma(\alpha)\Gamma(\beta)}{\Gamma(\alpha+\beta)}
$$
是 Beta 函数，用来保证分布归一化。

后验正比于先验乘似然：
$$
p(\theta\mid \mathcal D)\propto
\theta^s(1-\theta)^f\cdot \theta^{\alpha-1}(1-\theta)^{\beta-1}.
$$
合并同类项可得：
$$
p(\theta\mid \mathcal D)\propto
\theta^{\alpha-1+s}(1-\theta)^{\beta-1+f}.
$$
这正是一个新的 Beta 分布：
$$
\theta\mid \mathcal D\sim \text{Beta}(\alpha+s,\beta+f).
$$

因此：
- $\alpha$ 控制“先验成功倾向”
- $\beta$ 控制“先验失败倾向”
- $\alpha+\beta$ 控制先验强度
- 先验均值为
  $$
  \mathbb E[\theta]=\frac{\alpha}{\alpha+\beta}
  $$

对初学者最有用的是把 Beta 参数拆成两个维度理解：

| 参数变化 | 含义 |
|---|---|
| 固定 $\alpha+\beta$，改变比例 | 改变你事先认为成功概率大还是小 |
| 固定比例，放大 $\alpha+\beta$ | 增强先验强度，让先验更难被少量数据改变 |

例如：

| 先验 | 先验均值 | 先验强度 | 直观解释 |
|---|---:|---:|---|
| $\text{Beta}(1,1)$ | 0.5 | 2 | 非信息先验，几乎不偏向哪边 |
| $\text{Beta}(2,2)$ | 0.5 | 4 | 稍微平滑，仍较弱 |
| $\text{Beta}(20,180)$ | 0.1 | 200 | 相信成功率约 10%，且信念较强 |
| $\text{Beta}(100,100)$ | 0.5 | 200 | 相信成功率约 50%，且非常强 |

对题目中的例子，先验为 $\text{Beta}(2,2)$，观测到 3 次成功、1 次失败，则：
$$
\alpha' = 2+3=5,\qquad \beta'=2+1=3.
$$
因此后验是
$$
\theta\mid \mathcal D\sim \text{Beta}(5,3),
$$
后验均值为
$$
\mathbb E[\theta\mid \mathcal D]=\frac{5}{8}=0.625.
$$

若进一步关心不确定性，还可以看后验方差：
$$
\mathrm{Var}(\theta\mid \mathcal D)
=
\frac{\alpha'\beta'}{(\alpha'+\beta')^2(\alpha'+\beta'+1)}.
$$
代入 $\alpha'=5,\beta'=3$：
$$
\mathrm{Var}(\theta\mid \mathcal D)=\frac{5\cdot 3}{8^2\cdot 9}=\frac{15}{576}\approx 0.0260.
$$

这一步很重要，因为工程决策不能只看均值。两个广告位即使后验均值相同，不确定性也可能完全不同。

### 真实工程例子：在线广告 CTR

广告系统常需要实时估计每个广告位、每个素材、每个用户分群的点击率。若直接用经验频率
$$
\hat\theta=\frac{\text{点击次数}}{\text{展示次数}},
$$
早期样本很少时会非常不稳定。

例如一个新广告位前 10 次展示中点了 1 次，则经验 CTR 为 0.1。再来一次点击后，CTR 直接变成
$$
\frac{2}{11}\approx 0.182,
$$
变化幅度很大。

如果使用 Beta-Bernoulli，可以设置先验：
$$
\theta\sim \text{Beta}(20,180),
$$
则先验均值为
$$
\frac{20}{20+180}=0.1.
$$
这表示“历史上同类广告位大约是 10% CTR”，并且先验强度约等于 200 个伪样本。

假设新广告位上线后观察到：

| 展示结果 | 数量 |
|---|---:|
| 点击 | 3 |
| 未点击 | 17 |

则后验为
$$
\theta\mid \mathcal D\sim \text{Beta}(23,197),
$$
后验均值为
$$
\frac{23}{220}\approx 0.1045.
$$

如果不用先验，经验频率是：
$$
\frac{3}{20}=0.15.
$$

两者差异说明了先验的平滑作用：

| 估计方式 | 结果 | 特点 |
|---|---:|---|
| 经验频率 | 0.1500 | 对小样本非常敏感 |
| 后验均值 | 0.1045 | 会向历史先验回缩 |

这种“向群体历史回缩”的机制在工程上非常实用，因为它可以显著改善冷启动抖动。

在线系统里的更新规则通常只有两条：

- 每多一个点击，$\alpha\leftarrow\alpha+1$
- 每多一个未点击，$\beta\leftarrow\beta+1$

于是系统具备三个优点：

| 优点 | 原因 |
|---|---|
| 冷启动更稳 | 先验避免极端概率 |
| 流式更新简单 | 每次只改两个参数 |
| 存储开销低 | 无需保存全部原始事件 |

这类写法广泛用于：

| 场景 | 用法 |
|---|---|
| A/B 测试 | 估计版本转化率并比较 |
| 推荐系统 | 估计 item、slot、user segment 的点击率 |
| 风控系统 | 估计规则命中后的坏账率、欺诈率 |
| 质量监控 | 估计缺陷发生概率 |

---

## 代码实现

下面给出一个可直接运行的 Python 示例。它包含：

- Beta-Bernoulli 批量更新
- 流式更新
- 后验均值与方差计算
- 简单可信区间近似
- Dirichlet-Multinomial 扩展模板
- 可执行示例与断言测试

```python
from __future__ import annotations

from math import sqrt
from statistics import NormalDist
from typing import Iterable, List, Sequence, Tuple


def beta_bernoulli_update(
    alpha: float,
    beta: float,
    successes: int,
    failures: int,
) -> Tuple[float, float]:
    """
    Beta-Bernoulli 共轭更新
    posterior = Beta(alpha + successes, beta + failures)
    """
    if alpha <= 0 or beta <= 0:
        raise ValueError("alpha and beta must be positive")
    if successes < 0 or failures < 0:
        raise ValueError("successes and failures must be non-negative")

    return alpha + successes, beta + failures


def beta_bernoulli_stream(
    alpha: float,
    beta: float,
    observations: Iterable[int],
) -> Tuple[float, float]:
    """
    流式更新：观测必须为 0 或 1
    """
    a, b = alpha, beta
    for x in observations:
        if x not in (0, 1):
            raise ValueError(f"invalid Bernoulli observation: {x}")
        a += x
        b += 1 - x
    return a, b


def beta_posterior_mean(alpha: float, beta: float) -> float:
    if alpha <= 0 or beta <= 0:
        raise ValueError("alpha and beta must be positive")
    return alpha / (alpha + beta)


def beta_posterior_variance(alpha: float, beta: float) -> float:
    if alpha <= 0 or beta <= 0:
        raise ValueError("alpha and beta must be positive")
    total = alpha + beta
    return (alpha * beta) / (total * total * (total + 1))


def beta_posterior_std(alpha: float, beta: float) -> float:
    return sqrt(beta_posterior_variance(alpha, beta))


def beta_normal_approx_interval(
    alpha: float,
    beta: float,
    level: float = 0.95,
) -> Tuple[float, float]:
    """
    用正态近似给出 Beta 后验区间。
    说明：
    1. 这个方法在样本足够多时可用
    2. alpha/beta 很小时区间精度一般
    """
    if not 0 < level < 1:
        raise ValueError("level must be between 0 and 1")

    mean = beta_posterior_mean(alpha, beta)
    std = beta_posterior_std(alpha, beta)
    z = NormalDist().inv_cdf(0.5 + level / 2.0)

    lo = max(0.0, mean - z * std)
    hi = min(1.0, mean + z * std)
    return lo, hi


def count_binary_observations(observations: Sequence[int]) -> Tuple[int, int]:
    successes = 0
    failures = 0
    for x in observations:
        if x == 1:
            successes += 1
        elif x == 0:
            failures += 1
        else:
            raise ValueError(f"invalid Bernoulli observation: {x}")
    return successes, failures


def dirichlet_multinomial_update(
    alpha: Sequence[float],
    counts: Sequence[int],
) -> List[float]:
    """
    Dirichlet-Multinomial 共轭更新：
    posterior_alpha[k] = alpha[k] + counts[k]
    """
    if len(alpha) != len(counts):
        raise ValueError("alpha and counts must have the same length")
    if any(a <= 0 for a in alpha):
        raise ValueError("all alpha values must be positive")
    if any(c < 0 for c in counts):
        raise ValueError("all counts must be non-negative")

    return [a + c for a, c in zip(alpha, counts)]


def dirichlet_posterior_mean(alpha: Sequence[float]) -> List[float]:
    if any(a <= 0 for a in alpha):
        raise ValueError("all alpha values must be positive")

    total = sum(alpha)
    return [a / total for a in alpha]


def main() -> None:
    # 例 1：题目中的 Beta(2,2) + 3 成功 + 1 失败
    post_alpha, post_beta = beta_bernoulli_update(2.0, 2.0, 3, 1)
    post_mean = beta_posterior_mean(post_alpha, post_beta)
    post_var = beta_posterior_variance(post_alpha, post_beta)
    ci_low, ci_high = beta_normal_approx_interval(post_alpha, post_beta, level=0.95)

    print("Example 1: Beta-Bernoulli batch update")
    print(f"posterior = Beta({post_alpha:.1f}, {post_beta:.1f})")
    print(f"posterior mean = {post_mean:.6f}")
    print(f"posterior variance = {post_var:.6f}")
    print(f"approx 95% interval = [{ci_low:.6f}, {ci_high:.6f}]")
    print()

    # 例 2：流式更新，结果应一致
    stream_alpha, stream_beta = beta_bernoulli_stream(2.0, 2.0, [1, 1, 1, 0])
    assert (stream_alpha, stream_beta) == (5.0, 3.0)

    print("Example 2: stream update")
    print(f"posterior = Beta({stream_alpha:.1f}, {stream_beta:.1f})")
    print()

    # 例 3：从原始 0/1 样本统计后再更新
    observations = [1, 0, 0, 1, 1, 0, 1]
    successes, failures = count_binary_observations(observations)
    a, b = beta_bernoulli_update(1.0, 1.0, successes, failures)
    print("Example 3: update from raw observations")
    print(f"successes = {successes}, failures = {failures}")
    print(f"posterior mean = {beta_posterior_mean(a, b):.6f}")
    print()

    # 例 4：Dirichlet-Multinomial
    prior_alpha = [1.0, 1.0, 1.0]
    counts = [2, 3, 5]
    posterior_alpha = dirichlet_multinomial_update(prior_alpha, counts)
    posterior_mean = dirichlet_posterior_mean(posterior_alpha)

    print("Example 4: Dirichlet-Multinomial")
    print(f"posterior alpha = {posterior_alpha}")
    print(f"posterior mean = {posterior_mean}")

    # 基础断言
    assert (post_alpha, post_beta) == (5.0, 3.0)
    assert abs(post_mean - 0.625) < 1e-12
    assert posterior_alpha == [3.0, 4.0, 6.0]
    assert abs(sum(posterior_mean) - 1.0) < 1e-12


if __name__ == "__main__":
    main()
```

运行后，核心输出会包含：

```text
Example 1: Beta-Bernoulli batch update
posterior = Beta(5.0, 3.0)
posterior mean = 0.625000
```

如果把它写成服务端逻辑，接口通常只有四步：

1. 读取当前参数 $(\alpha,\beta)$
2. 收到一批新样本，统计成功数 $s$、失败数 $f$
3. 更新为 $(\alpha+s,\beta+f)$
4. 需要排序或决策时，读取后验均值、方差或区间

这套接口之所以适合工程落地，是因为它的状态压缩能力很强：

| 保存方案 | 是否需要原始样本 | 存储量 | 是否方便在线更新 |
|---|---|---:|---|
| 保存全部日志 | 需要 | 高 | 一般 |
| 只保存后验参数 | 不需要 | 极低 | 非常方便 |

再补一个新手容易忽略的点：  
“只保存后验参数”成立的前提是，你当前模型没有改变。如果你之后想从 Bernoulli 改成更复杂的分层模型，那么旧的压缩状态可能不足以支持新模型重建。

---

## 工程权衡与常见坑

共轭先验的优点是快、稳、可解释，但缺点也很直接：它用分布族限制换取计算便利。因此当先验过强、模型过粗、或者数据机制明显不满足假设时，结果会有偏差。

先看一个最典型的先验强度问题。

若先验是
$$
\text{Beta}(100,100),
$$
其均值虽然是 0.5，但强度很高。即使新来了 3 次成功、1 次失败，后验也只是
$$
\text{Beta}(103,101),
$$
后验均值约为
$$
\frac{103}{204}\approx 0.505.
$$

相反，若先验是
$$
\text{Beta}(1,1),
$$
同样的数据会得到
$$
\text{Beta}(4,2),
$$
后验均值为
$$
\frac{4}{6}\approx 0.667.
$$

这说明一个关键事实：  
样本少时，先验会不会主导结论，取决于先验强度，而不只是先验均值。

因此必须做灵敏度分析。所谓灵敏度分析，就是换几组合理先验，检查结论是否稳定。

例如对于同一批数据“3 成功、1 失败”：

| 先验 | 后验 | 后验均值 |
|---|---|---:|
| $\text{Beta}(1,1)$ | $\text{Beta}(4,2)$ | 0.667 |
| $\text{Beta}(2,2)$ | $\text{Beta}(5,3)$ | 0.625 |
| $\text{Beta}(20,20)$ | $\text{Beta}(23,21)$ | 0.523 |
| $\text{Beta}(100,100)$ | $\text{Beta}(103,101)$ | 0.505 |

这个表格比口头描述更清楚：相同数据，结论可能因为先验不同而差很多。

常见坑可以归纳如下：

| 常见坑 | 原因 | 典型表现 | 规避手段 |
|---|---|---|---|
| 先验与数据冲突 | 先验均值和数据趋势差太远 | 后验长期被拉偏 | 做灵敏度分析，比较多组先验 |
| 过强规则化 | 伪样本量太大 | 新数据很难修正结论 | 缩小 $\alpha+\beta$，只保留均值信息 |
| 误把共轭当真理 | 为了闭式更新，选了不合理模型 | 结果算得快但解释错误 | 先检查模型假设，再考虑计算便利 |
| 高维参数膨胀 | 维度上升后参数太多 | 存储、解释、估计都变差 | 分层建模、降维、近似推断 |
| 冷启动过度乐观 | 先验过弱 | 早期排序剧烈波动 | 用群体历史构造弱信息先验 |
| 忽略后验不确定性 | 只看后验均值 | 决策过度自信 | 同时看方差、分位数、区间 |

再强调一个很重要的工程事实：  
共轭先验只能对“你写进模型的那部分不确定性”进行精确更新。

例如 CTR 在真实系统里通常受这些因素影响：

| 因素 | 对 CTR 的影响 |
|---|---|
| 时间段 | 白天、夜晚转化率不同 |
| 用户分群 | 新用户、老用户行为差异大 |
| 素材疲劳 | 创意展示久了点击率下降 |
| 曝光位置 | 首屏和尾屏点击率差异大 |
| 外部事件 | 活动、节假日、突发新闻都会扰动 |

如果你仍然只用一个固定的 Bernoulli 参数 $\theta$ 建模，那么共轭更新再漂亮，也只是对一个过于粗糙的模型做了精确计算。

因此真实系统里的常见做法通常不是“全站一个 Beta”，而是：

| 做法 | 目的 |
|---|---|
| 按广告位、渠道、国家分桶 | 减少异质性 |
| 把共轭模块作为复杂模型的局部子模块 | 保留在线更新优势 |
| 在线系统用共轭作快速基线，离线系统再做校准 | 兼顾速度与表达能力 |

---

## 替代方案与适用边界

当共轭条件成立时，它通常是首选方案，因为更新快、解释强、实现简单。但在以下情况里，通常需要转向非共轭方法。

| 判断问题 | 适合继续用共轭 | 需要转向非共轭 |
|---|---|---|
| 似然是否属于指数族 | 是 | 否，或很难写成标准形式 |
| 先验是否必须非常灵活 | 不需要 | 需要多峰、重尾、稀疏或结构化先验 |
| 参数维度是否较低 | 是 | 很高，且存在复杂相关性 |
| 是否要求实时流式更新 | 强需求 | 允许离线重计算 |
| 是否能接受近似误差 | 希望闭式精确 | 可以接受采样或优化近似 |
| 是否存在复杂层次结构 | 弱或没有 | 强层次、多组随机效应 |

常见替代方案可以分三类：

1. MCMC  
MCMC 是用随机采样逼近后验的方法。常见代表包括 Metropolis-Hastings、Gibbs sampling、Hamiltonian Monte Carlo（HMC）。优点是灵活，缺点是计算慢、调参成本高。

2. 变分推断  
变分推断是用一个较简单的分布族去逼近真实后验，本质上是一个优化问题。它通常比 MCMC 快，但会引入近似误差。

3. 拉普拉斯近似或经验贝叶斯  
这类方法在工程里很常见，速度快，实现相对简单，但后验表达能力通常弱于 MCMC。

### 共轭与 HMC 的对比示例

还是估计点击概率。若模型只是：
- 每次点击独立同分布
- 点击概率为一个固定参数 $\theta$

那么 Beta-Bernoulli 已经足够，几行代码就能在线更新。

但如果模型改成：

- 点击率随时间漂移
- 不同用户群有分层结构
- 需要引入特征做 logistic 回归
- 先验希望使用稀疏或重尾形式

那么问题通常就不再是简单共轭问题。

例如 Bayesian logistic regression 的标准形式里，Bernoulli 似然与常见高斯先验并不形成简单闭式共轭，后验一般不能直接写出，通常要借助：

| 方法 | 适用点 |
|---|---|
| HMC | 连续参数、需要较高精度的复杂后验 |
| Polya-Gamma 增广 | logistic 类模型的专门技巧 |
| 拉普拉斯近似 | 想快速得到近似高斯后验 |
| 变分推断 | 大规模训练、可接受近似误差 |

可以用一个简化决策表理解：

| 决策步骤 | 选择 |
|---|---|
| 似然能否写成指数族且存在已知共轭先验 | 能 -> 优先考虑共轭 |
| 是否只需要少量参数、快速在线更新 | 是 -> 共轭通常最合适 |
| 模型是否包含复杂层次、非线性、非标准先验 | 是 -> 考虑 MCMC、VI 或近似方法 |
| 是否必须表达复杂后验形状 | 是 -> 共轭通常不够 |
| 是否对延迟极其敏感 | 是 -> 优先共轭或局部近似 |

所以，共轭体系最合适的位置不是“万能推断框架”，而是：

| 角色 | 说明 |
|---|---|
| 高性价比基线 | 结构清晰、实现快、便于上线 |
| 在线更新模块 | 适合日志流、监控流、增量学习 |
| 教学入口 | 最容易理解贝叶斯更新机制 |
| 复杂模型的局部组件 | 在大系统中承担局部闭式推断 |

---

## 参考资料

| 资料名 | 类型 | 重点内容 | 适用读者 |
|---|---|---|---|
| Wikipedia: Conjugate prior | 参考 | 共轭先验定义、常见分布对、经典例子 | 入门到进阶 |
| Wikipedia: Exponential family | 参考 | 指数族通式、自然参数、充分统计量 | 有公式基础的读者 |
| Murphy, *Machine Learning: A Probabilistic Perspective* | 教材 | 从概率模型角度系统讲共轭与指数族 | 进阶读者 |
| Bishop, *Pattern Recognition and Machine Learning* | 教材 | Beta-Bernoulli、Dirichlet-Multinomial、Gaussian 共轭体系 | 入门到进阶 |
| Gelman et al., *Bayesian Data Analysis* | 教材 | 先验选择、模型诊断、灵敏度分析 | 进阶读者 |
| Kruschke, *Doing Bayesian Data Analysis* | 教材 | 面向实践的贝叶斯建模与解释 | 初学者到进阶 |
| Bernardo and Smith, *Bayesian Theory* | 理论书 | 贝叶斯理论基础、先验与后验体系 | 理论导向读者 |
| Casella and Berger, *Statistical Inference* | 教材 | 充分统计量、指数族、经典统计基础 | 需要补统计基础的读者 |

这些资料可以按用途分工理解：

| 资料 | 更适合解决什么问题 |
|---|---|
| Wikipedia 两个条目 | 快速查定义、公式、分布对 |
| Murphy / Bishop | 把单个例子放回整个概率模型体系中理解 |
| Gelman / Kruschke | 理解先验强弱、模型诊断、工程解释 |
| Casella and Berger | 补足“充分统计量”“指数族”等底层概念 |
| Bernardo and Smith | 想进一步追理论基础时使用 |

如果按阅读顺序给一个更实用的建议，可以这样安排：

1. 先看 Wikipedia 中的 `Conjugate prior` 与 `Exponential family`，建立最基本的定义。
2. 再看 Bishop 或 Murphy，把 Beta-Bernoulli、Dirichlet-Multinomial、Gaussian 共轭放到统一框架里。
3. 最后读 Kruschke 或 Gelman，理解真实建模里为什么不能只停留在“会套公式”。

一句话概括这些参考资料的共同作用：  
它们不是为了让你背下更多分布对，而是让你建立一个稳定判断标准，即什么时候应该优先使用共轭，什么时候应该果断放弃它。
