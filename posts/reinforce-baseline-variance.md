## 核心结论

REINFORCE 是一种策略梯度方法。策略梯度的白话解释是：直接调整“在某个状态下选某个动作的概率”，让高回报动作更容易被再次选中。它的经典估计器是：

$$
\nabla_\theta J(\theta)=\mathbb{E}\left[\nabla_\theta \log \pi_\theta(a_t|s_t)\cdot G_t\right]
$$

其中 $G_t$ 是从时刻 $t$ 开始累计到轨迹结束的折扣回报。问题在于，这个估计器通常方差很高。方差的白话解释是：同样的策略参数，多采几条轨迹，梯度样本会抖得很厉害，更新方向不稳定。

引入基线 $b(s_t)$ 后，估计器变成：

$$
\nabla_\theta J(\theta)=\mathbb{E}\left[\nabla_\theta \log \pi_\theta(a_t|s_t)\cdot (G_t-b(s_t))\right]
$$

核心结论只有一句：**只要基线不依赖当前动作，加入基线不会改变期望梯度，但通常能显著降低方差。**

对初学者最重要的理解是：不加基线时，策略只知道“这次最后拿了多少分”；加基线后，策略看的是“这次比这个状态下的常规水平高多少”。后者更接近“相对表现”，因此训练更稳。

---

## 问题定义与边界

本文只讨论 **on-policy** 的 REINFORCE 及其状态基线。on-policy 的白话解释是：采样数据和更新策略使用的是同一个当前策略。这里不讨论 Q-learning、DQN、重要性采样、离策略修正，也不展开连续动作高斯策略的完整推导。

先定义符号。轨迹是状态、动作、奖励按时间排成的一串数据。策略 $\pi_\theta(a|s)$ 表示在状态 $s$ 下选择动作 $a$ 的概率分布。回报定义为：

$$
G_t=\sum_{k=t}^{T}\gamma^{k-t}r_k
$$

其中 $\gamma\in[0,1]$ 是折扣因子，表示越远的奖励权重越小。

| 符号 | 含义 |
|---|---|
| $s_t$ | 时刻 $t$ 的状态 |
| $a_t$ | 时刻 $t$ 采样的动作 |
| $\pi_\theta(a|s)$ | 参数为 $\theta$ 的策略 |
| $r_t$ | 时刻 $t$ 获得的即时奖励 |
| $G_t$ | 从 $t$ 开始的折扣回报 |
| $b(s_t)$ | 仅依赖状态的基线 |

边界也要说清楚。

第一，基线的目的不是改变最优策略，而是降低梯度估计的噪声。

第二，基线不要求等于真实的最优值函数。实践里常用近似值函数 $V_\phi(s)$ 作为基线。

第三，如果“基线”依赖当前动作，比如写成 $b(s_t,a_t)$，无偏性一般不再自动成立。这已经不是本文讨论的安全用法。

一个玩具场景可以帮助理解。假设同一个状态下，动作 A 的平均回报是 8，动作 B 的平均回报是 6。若某次采样里动作 A 拿到 9 分，不加基线时策略看到的是“9”；加基线后看到的是“比这个状态常规水平高 1”。这更接近“这一步到底值不值得强化”的真实信号。

---

## 核心机制与推导

REINFORCE 的核心来自 log-derivative trick。它把“对期望回报求导”改写成“对数概率乘回报”的形式，从而可以直接用采样轨迹近似梯度。结果是：

$$
\nabla_\theta J(\theta)=\mathbb{E}\left[\nabla_\theta \log \pi_\theta(a_t|s_t)\cdot G_t\right]
$$

问题在于 $G_t$ 很噪。原因有两个。

第一，$G_t$ 包含未来整条轨迹的随机性。环境随机、策略采样随机、序列长度随机，都会把噪声带进来。

第二，$G_t$ 把“当前动作的贡献”和“后续大量偶然因素”混在一起。于是梯度样本里既有信号，也有很多不属于当前动作的波动。

基线的作用就是减掉一部分“公共波动”。只要 $b(s_t)$ 不依赖当前动作，就有：

$$
\mathbb{E}\left[\nabla_\theta \log \pi_\theta(a_t|s_t)\cdot b(s_t)\right]=0
$$

证明很短。固定某个状态 $s$，对动作求条件期望：

$$
\sum_a \pi_\theta(a|s)\nabla_\theta \log \pi_\theta(a|s)b(s)
=
b(s)\sum_a \nabla_\theta \pi_\theta(a|s)
=
b(s)\nabla_\theta \sum_a \pi_\theta(a|s)
=
b(s)\nabla_\theta 1
=
0
$$

所以：

$$
\mathbb{E}\left[\nabla_\theta \log \pi_\theta(a_t|s_t)\cdot G_t\right]
=
\mathbb{E}\left[\nabla_\theta \log \pi_\theta(a_t|s_t)\cdot (G_t-b(s_t))\right]
$$

这就是“期望不变”的原因。

为什么方差会下降？直觉上，若某个状态本身就容易产生高回报，那么直接用 $G_t$ 会把“状态本来就好”也记到动作头上；减去 $b(s_t)$ 后，只保留“动作相对该状态平均水平多好或多差”的部分，噪声就更小。

最优状态基线可以写成条件期望意义下的 $E[G_t|s_t]$。对白话解释就是：在这个状态下，不考虑这次具体抽到哪个动作，未来通常能拿多少分。实践里很难直接知道它，所以一般训练一个值函数网络去近似：

$$
b(s_t)\approx V_\phi(s_t)
$$

此时

$$
\hat A_t=G_t-V_\phi(s_t)
$$

就叫优势的采样近似。优势的白话解释是：某个动作结果比“这个状态下的平均水平”高多少。Actor-Critic 的本质，就是 actor 负责出动作分布，critic 负责学这个平均水平。

再看一个可算清楚的玩具例子。单状态 $s$，两个动作：

- $\pi(a_1|s)=0.75$
- $\pi(a_0|s)=0.25$
- $G(a_1)=3$
- $G(a_0)=1$

设参数化满足：

- $\partial_\theta \log \pi(a_1|s)=0.25$
- $\partial_\theta \log \pi(a_0|s)=-0.75$

则无基线时的梯度样本分别为 $0.75$ 和 $-0.75$，期望是 $0.375$。若基线取状态平均回报：

$$
b(s)=0.75\times 3+0.25\times 1=2.5
$$

则梯度样本变成 $0.125$ 和 $1.125$，期望仍是 $0.375$，但方差会下降。这说明基线改变的是噪声结构，不是目标方向。

---

## 代码实现

实现时要把三件事拆开：采样轨迹、计算回报、构造优势。最小版本甚至不需要深度学习框架，先用 Python 就能验证“期望不变、方差下降”。

```python
from math import fsum

def mean(xs):
    return fsum(xs) / len(xs)

def variance(xs):
    m = mean(xs)
    return fsum((x - m) ** 2 for x in xs) / len(xs)

# 玩具例子
p_a1 = 0.75
p_a0 = 0.25

grad_log_a1 = 0.25
grad_log_a0 = -0.75

G_a1 = 3.0
G_a0 = 1.0

baseline = p_a1 * G_a1 + p_a0 * G_a0  # E[G|s] = 2.5

samples_no_baseline = [
    grad_log_a1 * G_a1,
    grad_log_a0 * G_a0,
]
weights = [p_a1, p_a0]

expected_no_baseline = fsum(w * x for w, x in zip(weights, samples_no_baseline))

samples_with_baseline = [
    grad_log_a1 * (G_a1 - baseline),
    grad_log_a0 * (G_a0 - baseline),
]
expected_with_baseline = fsum(w * x for w, x in zip(weights, samples_with_baseline))

# 展开成等权样本，只为方便算方差
expanded_no_baseline = [samples_no_baseline[0]] * 75 + [samples_no_baseline[1]] * 25
expanded_with_baseline = [samples_with_baseline[0]] * 75 + [samples_with_baseline[1]] * 25

assert abs(expected_no_baseline - 0.375) < 1e-9
assert abs(expected_with_baseline - 0.375) < 1e-9
assert variance(expanded_with_baseline) < variance(expanded_no_baseline)

def discounted_returns(rewards, gamma):
    out = [0.0] * len(rewards)
    running = 0.0
    for i in range(len(rewards) - 1, -1, -1):
        running = rewards[i] + gamma * running
        out[i] = running
    return out

assert discounted_returns([1.0, 2.0, 3.0], 0.5) == [2.75, 3.5, 3.0]
```

真正训练时，典型结构如下。

| 模块 | 输入 | 输出 | 作用 |
|---|---|---|---|
| Actor | $s_t$ | $\pi_\theta(a|s_t)$ | 产生动作分布 |
| Critic | $s_t$ | $V_\phi(s_t)$ | 估计状态基线 |
| Return | 奖励序列 | $G_t$ | 蒙特卡洛回报 |
| Advantage | $G_t, V_\phi(s_t)$ | $\hat A_t$ | actor 的更新信号 |

PyTorch 风格伪代码通常写成：

```python
returns = discounted_returns(rewards, gamma)         # G_t
values = value_net(states)                           # V(s_t)
advantages = returns - values.detach()               # A_hat_t

actor_loss = -(log_probs * advantages).mean()
critic_loss = ((values - returns.detach()) ** 2).mean()
```

这里 `detach()` 很关键。它的白话解释是：让 actor 更新时不要顺手把 critic 的梯度图也带进去。否则两个目标会意外耦合，训练容易不稳定。

一个真实工程例子是机器人控制或游戏智能体训练。比如 MuJoCo 连续控制任务里，纯 REINFORCE 常见现象是：回报曲线大幅抖动，需要很多轨迹才能看出提升趋势。加入 critic 之后，actor 用 $G_t-V(s_t)$ 更新，梯度会稳定很多。A2C、PPO 这类方法能实用，基础就在这里。

---

## 工程权衡与常见坑

最常见的误解是：有了基线，训练就一定稳。事实不是这样。基线只能减少一部分方差，不能消灭所有噪声。如果奖励极稀疏、轨迹极长、环境极随机，纯蒙特卡洛优势依然可能很难学。

常见坑有五个。

第一，把基线写成依赖当前动作的函数。这样往往会破坏前面的零期望证明，无偏性不再自动成立。安全做法是只用状态，或只用不包含当前动作选择结果的信息。

第二，把 $V(s)$ 当成必须精确的真值。工程里它只是近似器。只要它和状态平均回报相关，就通常能起到降方差作用，不需要先求出理论最优值。

第三，critic 太差，反而把噪声传回 actor。如果值函数完全没学到东西，$G_t-V(s_t)$ 可能比原始回报还乱。常见缓解方式是增加 critic 容量、提高 value loss 权重、做 reward normalization。

第四，只看平均回报，不看更新统计。平均回报偶尔会上升，但梯度已经失控。实践中建议至少同时看四个量：

| 观察项 | 目的 |
|---|---|
| reward 曲线 | 看策略是否真的变好 |
| advantage 分布 | 看基线是否在降方差 |
| gradient norm | 看更新是否过大 |
| KL divergence | 看策略是否漂移过快 |

第五，忘记区分“无偏”和“低方差”。REINFORCE + baseline 仍然是蒙特卡洛估计，仍可能很慢。无偏只表示长期平均方向对，不表示每一步都稳。

还有一个常见实现细节：很多工程代码会对 advantage 做标准化，比如减均值再除标准差。这会进一步改善数值尺度，但它是额外技巧，不是基线成立的必要条件。不要把“标准化 advantage”和“加入基线”混为一件事。

---

## 替代方案与适用边界

REINFORCE + baseline 适合教学、原型验证、以及你想明确理解策略梯度机制的时候。它的优点是公式干净、无偏性清楚、实现直接。缺点是样本效率仍然不高。

更实用的替代方案通常是把“整条轨迹回报”改成更低方差的时序差分信号。时序差分的白话解释是：不用等整局结束，只根据一步预测误差就能更新值函数。这样方差更低，但会引入一定偏差。

常见方法对比如下：

| 方法 | 方差 | 偏差 | 样本效率 | 适用场景 |
|---|---|---|---|---|
| REINFORCE | 高 | 低 | 低 | 教学、简单离散任务 |
| REINFORCE + baseline | 中 | 低 | 中 | 入门到实战过渡 |
| Actor-Critic / A2C | 更低 | 中 | 较高 | 常规控制任务 |
| PPO + GAE | 较低 | 可控 | 高 | 工程主流 |

GAE 可以看成在偏差和方差之间做系统折中。若任务规模较大、奖励较稀疏、动作连续，通常不会停留在纯 REINFORCE，而会直接上 PPO、A2C、A3C 或其他 actor-critic 变体。

适用边界可以总结成三条。

第一，如果你的目标是学懂“为什么策略梯度要减基线”，REINFORCE + baseline 足够。

第二，如果你的目标是把训练跑稳、样本数压低、对超参更不敏感，那么应优先考虑 actor-critic 家族。

第三，如果 baseline 已经加了，但训练仍高度抖动，问题通常不在“有没有基线”，而在“critic 质量、奖励设计、轨迹长度、归一化和优化器设置”。

---

## 参考资料

1. [Simple Statistical Gradient-Following Algorithms for Connectionist Reinforcement Learning](https://mlanthology.org/mlj/1992/williams1992mlj-simple/)
2. [Variance Reduction Techniques for Gradient Estimates in Reinforcement Learning](https://www.jmlr.org/papers/v5/greensmith04a.html)
3. [OpenAI Spinning Up: Expected Grad-Log-Prob Lemma and Baselines](https://spinningup.openai.com/en/latest/spinningup/rl_intro3.html)
4. [The Role of Baselines in Policy Gradient Optimization](https://proceedings.neurips.cc/paper_files/paper/2022/hash/718d02a76d69686a36eccc8cde3e6a41-Abstract-Conference.html)
