## 核心结论

重要性采样（Importance Sampling, IS）是在“旧策略采到的数据”上，估计“新策略会得到什么结果”的基础方法。这里的“策略”可以先理解成一个动作分配规则：给定状态 $s$，策略会给每个动作 $a$ 一个概率。

它解决的是异策略评估（Off-Policy Evaluation, OPE）问题。异策略的意思是：采样用的策略和评估用的策略不是同一个。记行为策略为 $\beta$，目标策略为 $\pi$，则重要性采样的核心修正权重是：

$$
\rho = \frac{\pi(a \mid s)}{\beta(a \mid s)}
$$

如果是整条轨迹（trajectory，可以理解成一次完整交互过程）的回报 $G$，轨迹级权重是：

$$
\rho(\tau)=\prod_{t=0}^{T-1}\frac{\pi(A_t\mid S_t)}{\beta(A_t\mid S_t)}
$$

于是目标策略的价值可以写成：

$$
V_\pi = \mathbb{E}_\beta[\rho(\tau)\,G(\tau)]
$$

这条公式的含义很直接：样本仍然来自 $\beta$，但每条样本按“它在 $\pi$ 下有多可能出现”重新加权。

普通 IS 估计量：

$$
\hat V_{\text{IS}}=\frac{1}{n}\sum_{i=1}^n \rho_i G_i
$$

加权重要性采样（Weighted Importance Sampling, WIS，也常叫 self-normalized IS）：

$$
\hat V_{\text{WIS}}=\frac{\sum_{i=1}^n \rho_i G_i}{\sum_{i=1}^n \rho_i}
$$

结论只有两条。

| 方法 | 有限样本偏差 | 方差 | 典型用途 |
|---|---:|---:|---|
| IS | 无偏 | 可能极大 | 理论分析、需要严格无偏时 |
| WIS | 有偏 | 通常更低 | 工程离线评估、结果更稳时 |

玩具例子：两条日志，$G_1=10,\rho_1=3$，$G_2=2,\rho_2=0.2$。  
普通 IS 为 $(3\times10+0.2\times2)/2=15.2$。  
WIS 为 $(3\times10+0.2\times2)/(3+0.2)\approx 9.5$。  
第一条样本被放大，是因为它更像目标策略会走到的轨迹。

真实工程例子：推荐系统已有旧排序策略 $\beta$ 的曝光点击日志，团队想评估新策略 $\pi$ 的 CTR，但不想立刻全量上线。此时 IS/WIS 可以先做离线价值估计，作为是否进入灰度发布的依据。

---

## 问题定义与边界

先把问题说清楚：这里讨论的是评估，不是训练。我们手里已经有一批历史日志，想知道如果当时不是按 $\beta$ 行动，而是按 $\pi$ 行动，平均回报会是多少。

常用符号如下。

| 符号 | 含义 | 白话解释 |
|---|---|---|
| $\beta(a\mid s)$ | 行为策略 | 当年真正拿来采数据的策略 |
| $\pi(a\mid s)$ | 目标策略 | 现在想评估的新策略 |
| $S_t$ | 时刻 $t$ 的状态 | 当前环境信息 |
| $A_t$ | 时刻 $t$ 的动作 | 在该状态下做出的选择 |
| $R_{t+1}$ | 即时奖励 | 这一步立刻得到的反馈 |
| $G$ | 回报 | 一条轨迹上累计得到的总收益 |
| $\rho$ | 重要性权重 | 用来修正分布差异的倍率 |

边界条件里最关键的是覆盖条件（coverage condition）。它表示：只要目标策略可能选择某个动作，行为策略过去就必须也有非零概率选过它。形式化写法是：

$$
\pi(a\mid s)>0 \Rightarrow \beta(a\mid s)>0
$$

如果不满足，就会出现 $\beta(a\mid s)=0$ 而 $\pi(a\mid s)>0$，此时比值 $\pi/\beta$ 无定义，重要性采样直接失效。

一个反例很典型。历史广告系统从不展示某种广告位，所以日志里该动作从未出现；新策略却想大量展示它。你无法靠 IS“脑补”这部分结果，因为数据里根本没有支持。

可用与不可用边界可以压缩成下表。

| 条件 | 可用性 | 原因 |
|---|---|---|
| 覆盖满足 | 可用 | 权重有定义 |
| 覆盖不满足 | 不可用 | 某些动作根本无样本支持 |
| 轨迹较短、策略接近 | 更适合 | 权重连乘不易爆炸 |
| 轨迹很长、策略差异大 | 风险高 | 方差通常急剧上升 |
| 日志可视为独立同分布近似 | 更容易分析 | 估计量性质更稳定 |
| 强时变系统、强反馈回路 | 需谨慎 | 日志分布可能已漂移 |

这里还要区分两个目标。

1. “理论上是否无偏”：IS 更强。
2. “工程上是否稳定”：WIS、裁剪、per-decision IS 往往更实用。

所以重要性采样不是“有日志就一定能评估”，而是“有足够覆盖且分布差异可控时，才能合理评估”。

---

## 核心机制与推导

先从单步情况开始。假设只有一步决策，目标是估计：

$$
V_\pi = \mathbb{E}_{A\sim \pi(\cdot\mid s)}[g(s,A)]
$$

其中 $g(s,A)$ 可以理解成该动作带来的收益。因为样本来自 $\beta$，所以改写为：

$$
V_\pi
= \sum_a \pi(a\mid s) g(s,a)
= \sum_a \beta(a\mid s)\frac{\pi(a\mid s)}{\beta(a\mid s)}g(s,a)
= \mathbb{E}_{A\sim\beta(\cdot\mid s)}\left[\frac{\pi(A\mid s)}{\beta(A\mid s)}g(s,A)\right]
$$

这就是“换测度”的基本想法。换测度可以直白理解成：样本还是老分布抽来的，但通过乘一个比例，把期望改写成新分布下的期望。

多步情形本质一样，只是对象从单步动作变成整条轨迹 $\tau=(S_0,A_0,R_1,\dots,S_T)$。如果环境转移概率记为 $P$，则：

$$
P_\pi(\tau)=P(S_0)\prod_{t=0}^{T-1}\pi(A_t\mid S_t)P(S_{t+1}\mid S_t,A_t)
$$

同理：

$$
P_\beta(\tau)=P(S_0)\prod_{t=0}^{T-1}\beta(A_t\mid S_t)P(S_{t+1}\mid S_t,A_t)
$$

两者相除，环境转移项会消掉，因为环境本身没变：

$$
\frac{P_\pi(\tau)}{P_\beta(\tau)}
=
\prod_{t=0}^{T-1}\frac{\pi(A_t\mid S_t)}{\beta(A_t\mid S_t)}
=\rho(\tau)
$$

于是：

$$
V_\pi
=\mathbb{E}_{\tau\sim P_\pi}[G(\tau)]
=\sum_\tau P_\pi(\tau)G(\tau)
=\sum_\tau P_\beta(\tau)\frac{P_\pi(\tau)}{P_\beta(\tau)}G(\tau)
=\mathbb{E}_{\tau\sim P_\beta}[\rho(\tau)G(\tau)]
$$

这就是轨迹级 IS 的来源。

看一个两步玩具例子。设某条轨迹是：
$(S_0=s_0,A_0=a_1,R_1,\ S_1=s_1,A_1=a_2,R_2)$

则：

$$
\rho(\tau)=\frac{\pi(a_1\mid s_0)}{\beta(a_1\mid s_0)}
\cdot
\frac{\pi(a_2\mid s_1)}{\beta(a_2\mid s_1)}
$$

如果第一步比值是 $2$，第二步比值是 $3$，则整条轨迹权重就是 $6$。这说明一个事实：只要某几步动作在目标策略下更偏好，整条轨迹会被迅速放大；反过来，若某步极不符合目标策略，整条轨迹会被压得很小。这也是长轨迹方差容易爆炸的原因。

轨迹级 IS 和 per-decision IS 的区别如下。

| 方法 | 权重作用位置 | 特点 |
|---|---|---|
| 轨迹级 IS | 整条轨迹最终回报 $G$ 前乘一个总权重 | 定义直接，但长轨迹方差大 |
| Per-decision IS | 每一步奖励用到当前前缀权重 | 常更稳定，能减少无关未来权重影响 |

如果折扣因子为 $\gamma$，per-decision IS 常写成：

$$
\hat V_{\text{PDIS}}
=
\frac{1}{n}\sum_{i=1}^n\sum_{t=0}^{T_i-1}
\gamma^t
\left(\prod_{k=0}^{t}\frac{\pi(A_k^i\mid S_k^i)}{\beta(A_k^i\mid S_k^i)}\right)
R_{t+1}^i
$$

它的直觉是：第 $t$ 步奖励只需要修正到第 $t$ 步为止，不必强行乘上未来动作的概率比。

---

## 代码实现

工程实现的关键不是公式本身，而是数值稳定性。因为长轨迹下直接连乘 $\rho=\prod_t \pi/\beta$，很容易上溢或下溢，所以更稳妥的写法是先算 $\log\rho$，再指数还原：

$$
\log \rho = \sum_t \left(\log \pi(A_t\mid S_t)-\log \beta(A_t\mid S_t)\right)
$$

下面给一个可运行的最小 Python 示例。输入是若干条轨迹，每条轨迹由若干步组成，每步包含 `pi`、`beta`、`reward`。代码同时计算 IS 和 WIS，并包含 `assert`。

```python
import math

def discounted_return(rewards, gamma=1.0):
    total = 0.0
    for t, r in enumerate(rewards):
        total += (gamma ** t) * r
    return total

def trajectory_weight(steps):
    log_rho = 0.0
    for step in steps:
        pi = step["pi"]
        beta = step["beta"]
        assert pi >= 0.0 and beta >= 0.0
        if pi > 0.0 and beta == 0.0:
            raise ValueError("coverage violated: pi>0 but beta=0")
        if pi == 0.0:
            continue
        log_rho += math.log(pi) - math.log(beta)
    rho = math.exp(log_rho)
    return rho

def estimate_is_wis(trajectories, gamma=1.0):
    weighted_returns = []
    weights = []

    for traj in trajectories:
        rewards = [step["reward"] for step in traj]
        G = discounted_return(rewards, gamma=gamma)
        rho = trajectory_weight(traj)
        weighted_returns.append(rho * G)
        weights.append(rho)

    n = len(trajectories)
    is_estimate = sum(weighted_returns) / n
    wis_estimate = sum(weighted_returns) / sum(weights) if sum(weights) > 0 else 0.0
    return is_estimate, wis_estimate

trajectories = [
    [
        {"pi": 0.6, "beta": 0.2, "reward": 10.0},  # rho contribution = 3
    ],
    [
        {"pi": 0.1, "beta": 0.5, "reward": 2.0},   # rho contribution = 0.2
    ],
]

is_v, wis_v = estimate_is_wis(trajectories)
assert abs(is_v - 15.2) < 1e-9
assert abs(wis_v - 9.5) < 1e-9

print("IS =", is_v)
print("WIS =", wis_v)
```

新手版理解可以压成三步。

1. 先算每条样本的权重 $\rho$。
2. 再算每条样本的“加权回报” $\rho G$。
3. 最后做平均，或做归一化平均。

真实工程例子：离线推荐日志中，一条曝光序列可能包含用户上下文、候选集合、展示动作、点击与转化。通常会先按 session 或 request 分组形成轨迹，再依据模型输出的 $\pi(a\mid s)$ 和日志记录的 $\beta(a\mid s)$ 计算权重。若日志没有显式保存 $\beta$，就需要额外估计行为策略，这会引入新的误差源。

一个常见接口设计如下。

| 输入 | 输出 | 失败条件 |
|---|---|---|
| 轨迹日志、$\pi$ 概率、$\beta$ 概率 | IS / WIS / PDIS 估计值 | 覆盖不足、权重全为零、概率记录错误 |
| 可选参数：$\gamma$、clip 阈值 | 稳定化后的估计值 | 裁剪过强导致偏差过大 |

---

## 工程权衡与常见坑

最大的问题不是“不会算”，而是“算出来不稳”。当 $\pi$ 与 $\beta$ 差异大时，权重会极端化，少数样本可能主导整个估计。

看一个最小例子：两条样本，回报都差不多，但 $\rho_1=100,\rho_2=0.01$。普通 IS 基本完全由第一条决定。于是你会看到离线评估结果在不同批次日志上剧烈波动，这种“很跳”的现象通常就是高方差在作祟。

工程上常见问题如下。

| 问题 | 后果 | 规避手段 |
|---|---|---|
| 覆盖不足 | 比值无定义，估计失效 | 限制新策略动作空间，保证 logging policy 有探索 |
| 长轨迹连乘 | 权重爆炸或塌缩 | 改用 per-decision IS，缩短评估窗口 |
| 方差过大 | 结果不稳定，结论不可信 | WIS、裁剪、分桶评估、bootstrap 区间 |
| WIS 偏差 | 更稳但不再无偏 | 作为工程估计，和 IS/DR 对照看 |
| 数值不稳定 | 上溢下溢、NaN | 用 $\log \rho$ 累加，再 `exp` |
| 行为策略估计错误 | 系统性偏差 | 显式记录 propensity，避免事后拟合 |

权重裁剪（clipping）是最常见的稳定化手段。比如把 $\rho$ 改成：

$$
\tilde \rho = \min(\rho, c)
$$

其中 $c$ 是裁剪阈值。它能显著压低方差，但一定会引入偏差。是否值得，取决于你更怕“平均错一点”，还是更怕“每次跳很大”。

另一个常见坑是把行为策略概率记错。推荐和广告系统里，很多日志只记了最终展示结果，没有记完整候选集合和当时真实的采样概率。这样即使公式正确，估计也会因为 $\beta$ 不可恢复而失真。严格说，离线 OPE 最重要的数据资产之一不是 reward，而是 logging propensity，也就是当年动作被选中的真实概率。

---

## 替代方案与适用边界

当轨迹短、覆盖好、$\pi$ 与 $\beta$ 接近时，IS 很合适，因为它概念清楚、理论干净。但很多真实系统不满足这些条件，所以要比较替代方案。

| 方法 | 是否无偏 | 方差 | 实现复杂度 | 适用场景 |
|---|---:|---:|---:|---|
| IS | 是 | 高 | 低 | 短轨迹、策略接近、强调理论正确性 |
| WIS | 否 | 中 | 低 | 工程离线评估、想要更稳 |
| Per-decision IS | 常可保持无偏性质 | 通常低于轨迹级 IS | 中 | 多步任务、长轨迹 |
| DR / Doubly Robust | 模型正确时更稳 | 通常更低 | 高 | 有较好价值模型或奖励模型时 |

Doubly Robust（双重稳健）可以白话理解成“两套保险”：一套来自重要性加权，一套来自模型预测。只要两者里有一套足够准，估计就可能比单纯 IS 更稳。它不是本文主角，但在长轨迹、稀疏反馈、策略差异大时，经常比纯 IS 类方法更实用。

可以用一个简单决策表判断。

| 场景 | 更优先的方法 |
|---|---|
| 单步或短轨迹，且覆盖好 | IS / WIS |
| 多步轨迹，未来步数多 | Per-decision IS |
| 策略差异大，少数样本权重极端 | WIS + 裁剪，或 DR |
| 日志稀疏，回报噪声大 | DR、模型法、与在线小流量实验结合 |
| 行为策略概率未可靠记录 | 纯 IS 风险很高，需谨慎估计 $\beta$ 或放弃该评估 |

真实工程例子：离线推荐里，如果新旧策略已经差了一个大版本，候选召回、排序特征、甚至展示位逻辑都变了，直接做轨迹级 IS 往往会非常不稳定。这时更合理的做法通常是：先看覆盖是否足够，再用 WIS 和 per-decision IS 做基线，同时加入 DR 作为对照，最后只把结论当成“是否值得小流量实验”的证据，而不是最终上线依据。

所以适用边界可以总结成一句话：重要性采样擅长“在已有支持的数据上做分布修正”，不擅长“替代根本不存在的数据”。

---

## 参考资料

1. [Sutton & Barto, Reinforcement Learning: An Introduction](https://mitpress.mit.edu/9780262352703/reinforcement-learning/)
2. [Hanna, Niekum, Stone, Importance Sampling in Reinforcement Learning with an Estimated Behavior Policy](https://link.springer.com/article/10.1007/s10994-020-05938-9)
3. [Mahmood, van Hasselt, Sutton, Weighted Importance Sampling for Off-Policy Learning with Linear Function Approximation](https://papers.nips.cc/paper/5249-weighted-importance-sampling-for-off-policy-learning-with-linear-function-approximation)
4. [Cardoso et al., BR-SNIS: Bias Reduced Self-Normalized Importance Sampling](https://openreview.net/forum?id=HH_jBD2ObPq)
