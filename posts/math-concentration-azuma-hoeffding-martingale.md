## 核心结论

Azuma-Hoeffding 不等式描述的是：一个鞅过程只要每一步的增量有上界，那么最终偏离初始值很远的概率会按指数速度下降。

一句话说，如果每一步最多只会把总量推高或拉低一点点，那么走很多步之后，偏离起点太远的概率仍然很小。

形式化地，设 $(M_t, \mathcal F_t)$ 是鞅，$\Delta_t = M_t - M_{t-1}$ 是第 $t$ 步的增量。如果

$$
\mathbb E[\Delta_t \mid \mathcal F_{t-1}] = 0,\quad |\Delta_t| \le c_t
$$

则对任意 $u > 0$，

$$
\mathbb P(|M_n - M_0| \ge u)
\le
2 \exp\left(
-\frac{u^2}{2\sum_{t=1}^n c_t^2}
\right)
$$

它的核心控制量不是“样本是否独立”，而是两件事：

| 条件 | 白话解释 | 作用 |
|---|---|---|
| 鞅 | 看完过去之后，下一步平均不会系统性变大或变小 | 排除可预测漂移 |
| 鞅差 | 每一步新增的变化量 | 分解累计偏差 |
| 有界增量 | 每一步最多只能改变 $c_t$ | 控制最坏单步影响 |
| $\sum c_t^2$ | 所有单步影响的平方和 | 决定尾界松紧 |

简化版结论是：当 $|\Delta_t| \le c_t$ 时，$\mathbb P(|M_n - M_0| \ge u)$ 以 $\exp(-u^2)$ 的级别衰减，分母由 $\sum c_t^2$ 决定。

输入条件到输出结论的对应关系如下：

| 输入条件 | 输出结论 |
|---|---|
| 每步条件均值为 0 | 不存在可预测方向偏移 |
| 每步变化不超过 $c_t$ | 单步风险可控 |
| 多步按时间顺序暴露信息 | 可以使用鞅建模 |
| $\sum c_t^2$ 较小 | 最终偏差尾界更紧 |

这就是它在在线过程、顺序决策、随机算法分析、泛化误差分析中常见的原因：很多系统不是一次性生成全部随机变量，而是一步一步揭示信息。

---

## 问题定义与边界

鞅是一类随机过程。白话说，鞅表示“在已知过去全部信息之后，下一步的条件平均值等于现在的值”。

设 $\mathcal F_t$ 表示第 $t$ 步之前和当前已经知道的全部信息，称为过滤族。过滤族是按时间增长的信息集合，满足

$$
\mathcal F_0 \subseteq \mathcal F_1 \subseteq \cdots \subseteq \mathcal F_n
$$

随机过程 $(M_t, \mathcal F_t)$ 是鞅，要求：

$$
\mathbb E[M_t \mid \mathcal F_{t-1}] = M_{t-1}
$$

定义鞅差：

$$
\Delta_t = M_t - M_{t-1}
$$

则鞅条件等价于：

$$
\mathbb E[\Delta_t \mid \mathcal F_{t-1}] = 0
$$

这句话的含义是：在第 $t-1$ 步已经知道所有过去信息之后，第 $t$ 步的平均新增变化为 0。它不要求 $\Delta_t$ 彼此独立，只要求每一步相对过去没有可预测偏移。

Azuma-Hoeffding 还需要有界增量条件：

$$
|\Delta_t| \le c_t
$$

这里 $c_t$ 是第 $t$ 步最大可能变化幅度。它可以每一步不同。

一个玩具例子：掷硬币游戏中，正面加 1，反面减 1。令 $M_t$ 为前 $t$ 次得分总和。公平硬币满足下一步条件期望为 0，且每步变化不超过 1，所以 $c_t = 1$。Azuma-Hoeffding 可以给出 $M_n$ 偏离 0 很远的概率上界。

一个在线统计均值的例子：第 $t$ 步看到一个新样本 $X_t$，系统更新某个估计值。如果样本被限制在 $[0,1]$，且每个样本对最终均值的影响最多是 $1/n$，那么单步变化上界可取 $c_t = 1/n$。信息流是：先看到前 $t-1$ 个样本，再暴露第 $t$ 个样本，更新估计值。这个过程可以通过 Doob 鞅或鞅差分解来分析。

适用边界如下：

| 场景 | 是否适合 Azuma-Hoeffding | 原因 |
|---|---:|---|
| 独立有界样本求和 | 可以，但 Hoeffding 通常更直接 | 独立有界是更强条件 |
| 在线决策过程 | 适合 | 允许依赖，只需条件零均值 |
| 随机算法逐步暴露输入 | 适合 | 可构造 Doob 鞅 |
| 单步变化无界 | 通常不适合 | 缺少 $|\Delta_t| \le c_t$ |
| 重尾分布 | 需要谨慎 | 极端值会破坏有界增量 |
| 已知条件方差很小 | Azuma 可用但可能不紧 | Freedman 往往更好 |

Azuma-Hoeffding 的边界很清楚：它不依赖独立同分布，但依赖条件零均值和有界增量。若单步无界、重尾明显，或者真实方差远小于粗略上界，直接使用它可能得到过松甚至不合法的结论。

---

## 核心机制与推导

Azuma-Hoeffding 的推导主线是：

| 步骤 | 工具 | 结果 |
|---|---|---|
| 1 | Hoeffding 引理 | 控制单步条件矩母函数 |
| 2 | 条件期望迭代 | 把单步上界合并成多步上界 |
| 3 | Markov 不等式 | 从指数矩转为尾概率 |
| 4 | 优化 $\lambda$ | 得到最紧指数尾界 |

矩母函数是 $\mathbb E[e^{\lambda X}]$，白话说，它把随机变量的尾部风险编码到指数函数里。指数函数的好处是乘法结构清晰，适合把很多步的小风险合并起来。

对每个鞅差 $\Delta_t$，在已经知道过去信息 $\mathcal F_{t-1}$ 的条件下，有

$$
\mathbb E[\Delta_t \mid \mathcal F_{t-1}] = 0,\quad |\Delta_t| \le c_t
$$

由 Hoeffding 引理得到：

$$
\mathbb E[\exp(\lambda \Delta_t) \mid \mathcal F_{t-1}]
\le
\exp\left(\frac{\lambda^2 c_t^2}{2}\right)
$$

直观上，每一步都像一个受限的小扰动。虽然它可以依赖过去，但条件于过去之后，它没有平均方向，并且幅度有限。指数函数把“很多小扰动同时偏向一侧”的风险压缩成一个统一上界。

令

$$
S_t = M_t - M_0 = \sum_{i=1}^t \Delta_i
$$

使用条件期望迭代，可以得到：

$$
\mathbb E[\exp(\lambda S_n)]
\le
\exp\left(
\frac{\lambda^2}{2}\sum_{t=1}^n c_t^2
\right)
$$

等价地，

$$
\exp\left(
\lambda S_t -
\frac{\lambda^2}{2}\sum_{i=1}^t c_i^2
\right)
$$

是一个超鞅。超鞅是条件期望不超过当前值的过程，白话说，它在平均意义下不会上升。

接着控制上尾：

$$
\mathbb P(S_n \ge u)
=
\mathbb P(e^{\lambda S_n} \ge e^{\lambda u})
$$

由 Markov 不等式，

$$
\mathbb P(S_n \ge u)
\le
\frac{\mathbb E[e^{\lambda S_n}]}{e^{\lambda u}}
\le
\exp\left(
-\lambda u +
\frac{\lambda^2}{2}\sum_{t=1}^n c_t^2
\right)
$$

对 $\lambda$ 最优化。令

$$
\lambda = \frac{u}{\sum_{t=1}^n c_t^2}
$$

代回得到：

$$
\mathbb P(S_n \ge u)
\le
\exp\left(
-\frac{u^2}{2\sum_{t=1}^n c_t^2}
\right)
$$

下尾对 $-S_n$ 使用同样推导，得到双侧形式：

$$
\mathbb P(|M_n - M_0| \ge u)
\le
2\exp\left(
-\frac{u^2}{2\sum_{t=1}^n c_t^2}
\right)
$$

这个公式说明，累计偏差不是由 $n$ 单独决定，而是由 $\sum c_t^2$ 决定。如果每一步的影响都被压得很小，很多步之后总体偏差仍然可控。

---

## 代码实现

工程实现里，代码不需要“证明一个过程是鞅”。证明属于建模阶段。代码要做的是：给定偏差阈值 $u$ 和增量上界列表 $c_1,\dots,c_n$，计算 Azuma-Hoeffding 双侧尾界。

参数含义如下：

| 参数 | 含义 | 要求 |
|---|---|---|
| `u` | 关注的偏差阈值 | 必须非负 |
| `c_list` | 每一步增量上界 $c_t$ | 每项必须非负 |
| `s2` | $\sum c_t^2$ | 必须大于 0 |
| 返回值 | 双侧尾概率上界 | 最大不超过 1 时才有直接概率解释 |

```python
import math

def azuma_bound(u, c_list):
    if u < 0:
        raise ValueError("u must be non-negative")
    if not c_list:
        raise ValueError("c_list must be non-empty")
    if any(c < 0 for c in c_list):
        raise ValueError("all c_t must be non-negative")

    s2 = sum(c * c for c in c_list)
    if s2 == 0:
        return 0.0 if u > 0 else 1.0

    return min(1.0, 2 * math.exp(-(u * u) / (2 * s2)))


# 玩具例子：100 步，每步最多改变 0.05，问偏差至少为 1 的概率上界
bound = azuma_bound(1.0, [0.05] * 100)
print(bound)

assert abs(bound - 2 * math.exp(-2)) < 1e-12
assert 0.27 < bound < 0.28
```

这个例子中：

$$
\sum_{t=1}^{100} c_t^2 = 100 \times 0.05^2 = 0.25
$$

所以

$$
\mathbb P(|M_{100} - M_0| \ge 1)
\le
2\exp\left(-\frac{1^2}{2 \times 0.25}\right)
=
2e^{-2}
\approx 0.271
$$

真实工程例子：在线 A/B 测试中，用户按时间顺序进入实验。系统持续观察转化率差异，并希望设置提前告警阈值。若每个用户对最终指标的影响最多是 $1/n$，则单步增量上界可以取 $c_t = 1/n$。这时告警阈值不应只看样本数 $n$，还要看每个用户对统计量的最大影响。

实际使用难点通常不在公式，而在如何得到可信的 $c_t$。如果指标经过裁剪、归一化、限幅，$c_t$ 通常更容易证明；如果指标允许极端值直接进入统计量，Azuma-Hoeffding 的前提就可能不成立。

---

## 工程权衡与常见坑

Azuma-Hoeffding 不是“独立性定理”。它允许依赖，但要求依赖结构能被过滤族描述，并且每一步在条件于过去后没有平均漂移。

常见错误如下：

| 错误用法 | 问题 | 正确做法 |
|---|---|---|
| 看到随机过程就直接套公式 | 未证明鞅条件 | 先验证 $\mathbb E[\Delta_t|\mathcal F_{t-1}]=0$ |
| 只写 $n$，不写 $c_t$ | 忽略单步影响 | 明确计算 $\sum c_t^2$ |
| 对无界变量使用 Azuma | 违反有界增量 | 截断、稳健化或换工具 |
| 把经验波动当作上界 | 样本内波动不是必然上界 | 给出确定性或高概率上界 |
| 忽略自适应策略 | 决策依赖过去 | 用过滤族重新定义信息流 |

错误用法和正确用法的差异：

| 场景 | 错误说法 | 正确说法 |
|---|---|---|
| A/B 测试 | 用户独立，所以直接 Azuma | 构造按用户到达顺序的鞅差，并证明单用户影响有界 |
| Bandit 算法 | 每轮奖励随机，所以偏差可控 | 策略依赖历史，需要条件期望意义下的零均值 |
| 在线监控 | 样本数大，所以误报概率小 | 需要看每步最大影响和重复检测方式 |
| 模型评估 | 指标平均值有界，所以一定紧 | 有界只保证可用，不保证界足够紧 |

重尾、无界、方差不均衡会让 Azuma-Hoeffding 失真。原因是它只看最坏单步幅度 $c_t$，不看大多数时候的真实方差。如果少数极端值把 $c_t$ 拉得很大，那么 $\sum c_t^2$ 会变大，尾界迅速变松。若 $c_t$ 根本不存在，公式就不能直接使用。

一个真实工程例子是在线广告系统的点击率监控。点击是 $0/1$，单次影响有限，Azuma-Hoeffding 可以用于保守告警。但如果监控的是用户消费金额，金额分布可能重尾，少数大额用户会破坏有界增量假设。此时应先做截断、winsorize、分桶，或者改用更适合重尾数据的稳健方法。

---

## 替代方案与适用边界

选择集中不等式时，先问三个问题：

| 问题 | 影响 |
|---|---|
| 随机变量是否独立 | 决定能否直接用 Hoeffding |
| 是否能写成鞅 | 决定能否用 Azuma-Hoeffding 或 Freedman |
| 是否有条件方差信息 | 决定是否能获得比 Azuma 更紧的界 |

几种常见方法的区别如下：

| 方法 | 假设 | 适用对象 | 优点 | 局限 | 典型场景 |
|---|---|---|---|---|---|
| Hoeffding | 独立、有界 | 独立变量求和 | 简单直接 | 不适合复杂依赖 | 独立样本均值 |
| Azuma-Hoeffding | 鞅差、有界增量 | 顺序过程 | 允许依赖 | 不利用方差细节 | 在线决策、随机算法 |
| McDiarmid | 独立输入、函数满足 bounded differences | 输入扰动下的函数值 | 适合函数稳定性分析 | 需要独立输入 | 泛化误差、随机图函数 |
| Freedman | 鞅差有界、条件方差可控 | 方差敏感的鞅过程 | 方差小时更紧 | 推导和参数更复杂 | 自适应实验、bandit |

同一个在线更新场景可以这样选择：

| 场景描述 | 更合适的方法 |
|---|---|
| 每个样本独立，目标是均值偏差 | Hoeffding |
| 每一步策略依赖过去，目标是累计误差 | Azuma-Hoeffding |
| 目标是一个独立输入集合上的稳定函数 | McDiarmid |
| 每步最坏幅度较大，但实际条件方差很小 | Freedman |

McDiarmid 和 Azuma-Hoeffding 的关系尤其重要。McDiarmid 常通过 Doob 鞅证明：把独立输入一个个暴露出来，观察目标函数的条件期望如何变化。如果改变单个输入最多改变函数值 $c_t$，就得到 bounded differences 条件。

Freedman 则进一步利用条件方差。Azuma-Hoeffding 只看 $c_t$，相当于用最坏情况控制所有步骤；Freedman 同时看

$$
\sum_{t=1}^n \mathbb E[\Delta_t^2 \mid \mathcal F_{t-1}]
$$

当真实条件方差远小于 $\sum c_t^2$ 时，Freedman 通常明显更紧。

工程上可以采用这个选择规则：独立求和先用 Hoeffding；顺序依赖但能证明鞅差有界，用 Azuma-Hoeffding；独立输入上的函数稳定性，用 McDiarmid；有清晰条件方差估计，用 Freedman。

---

## 参考资料

推荐阅读路径：先看 MIT 讲义理解鞅与指数超鞅，再看 Azuma 原文和 Hoeffding 原文确认定理来源，最后看 McDiarmid 与 Tropp 了解推广。

1. Hoeffding, W. (1963). Probability Inequalities for Sums of Bounded Random Variables. JASA. https://doi.org/10.1080/01621459.1963.10500830
2. Azuma, K. (1967). Weighted sums of certain dependent random variables. Tohoku Mathematical Journal. https://www.jstage.jst.go.jp/article/tmj1949/19/3/19_3_357/_article/-char/en
3. MIT OCW Lecture 12: Martingales concentration inequality. https://ocw.mit.edu/courses/15-070j-advanced-stochastic-processes-fall-2013/4644bbdc15d6af2f574535aa5479ecba_MIT15_070JF13_Lec12.pdf
4. McDiarmid, C. (1989). On the method of bounded differences. https://doi.org/10.1017/CBO9781107359949.008
5. Tropp, J. A. (2011). Freedman's inequality for matrix martingales. https://authors.library.caltech.edu/records/an67d-29489
