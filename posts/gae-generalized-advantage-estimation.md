## 核心结论

GAE（Generalized Advantage Estimation，广义优势估计）不是新的强化学习目标，而是一个**更稳的优势函数估计器**。优势函数可以写成：

$$
A^\pi(s,a)=Q^\pi(s,a)-V^\pi(s)
$$

它回答的问题很直接：**当前动作比这个状态下的平均水平好多少**。这里的“平均水平”就是状态价值 $V^\pi(s)$，也就是站在当前状态、按当前策略继续行动时，预期能拿到的平均回报。

问题在于，$Q^\pi(s,a)$ 难直接估计，尤其是在长回合、连续动作、奖励稀疏的任务里。GAE 的做法是不用直接算 $Q-V$，而是先构造每一步的 TD 残差，再把未来残差按指数衰减累加：

$$
\delta_t=r_t+\gamma V(s_{t+1})-V(s_t)
$$

$$
\hat A_t^{GAE(\gamma,\lambda)}=\sum_{l=0}^{T-t-1}(\gamma\lambda)^l\delta_{t+l}
$$

结论可以压缩成三点：

| 结论 | 含义 |
| --- | --- |
| GAE 估计的是 advantage，不是 reward | 它服务于 policy gradient 更新 |
| $\lambda$ 是核心旋钮 | 控制更像 TD 还是更像 Monte Carlo |
| 实践中常取 $\lambda \approx 0.95$ | 在偏差和方差之间通常比较稳 |

当 $\lambda=0$ 时，GAE 退化成单步 TD；当 $\lambda=1$ 时，它接近整段回报驱动的 Monte Carlo 形式。对 PPO、TRPO 这类 on-policy 算法，GAE 的价值就在于：**比单步 TD 更准确，又不像纯 Monte Carlo 那样抖动大**。

---

## 问题定义与边界

先把三个最容易混的量分开。

$$
V^\pi(s)=\mathbb E_\pi[R_t\mid s_t=s]
$$

$$
Q^\pi(s,a)=\mathbb E_\pi[R_t\mid s_t=s,a_t=a]
$$

$$
A^\pi(s,a)=Q^\pi(s,a)-V^\pi(s)
$$

这里的 $R_t$ 是从时刻 $t$ 开始的折扣回报。白话说：

- $V^\pi(s)$：只看状态，问“来到这里平均值多少钱”
- $Q^\pi(s,a)$：看状态和动作，问“现在做这个动作值多少钱”
- $A^\pi(s,a)$：问“这个动作比平均水平高还是低”

这也是 policy gradient 里需要 advantage 的原因。策略网络不只需要知道“最终拿了多少分”，更需要知道“**这一步动作到底该鼓励还是该压制**”。如果某个动作让结果比 baseline 更好，那么它的 advantage 为正；反之为负。

下面这个表可以帮助区分用途：

| 量 | 定义对象 | 典型用途 | 难点 |
| --- | --- | --- | --- |
| Return | 一条轨迹上的实际回报 | 监督信号、评估 | 方差大 |
| Value | 状态的平均回报 | 做 baseline、训练 critic | 需要逼近函数 |
| Advantage | 动作相对平均水平的增益 | 更新 actor | 难直接估计 |

GAE 解决的边界也要说清楚：

- 它解决的是“**如何估计 advantage**”
- 它不解决 reward design，也不替代 value learning
- 它通常服务于 on-policy actor-critic，而不是通用地替换所有回报估计方法

如果 reward 本身设计错了，GAE 不会救你；如果 critic 非常差，GAE 只会把这个误差重新组织后送进 actor。

---

## 核心机制与推导

GAE 的起点不是直接估计 $A^\pi(s,a)$，而是先看单步 TD 残差：

$$
\delta_t=r_t+\gamma V_{t+1}-V_t
$$

其中 $V_t=V(s_t)$，$V_{t+1}=V(s_{t+1})$。TD 残差的白话解释是：**当前这一步实际拿到的奖励，加上下一个状态的估值，减去当前状态原来的估值**。如果这个差值为正，说明这一跳比 critic 原来预期得更好。

单独用 $\delta_t$ 去近似 advantage 可以，但它有偏。原因很简单：它只看了一步 bootstrap，也就是“用下一状态价值去补尾巴”，因此更依赖 critic 的精度。

GAE 做的是把后续 TD 残差也算进来：

$$
\hat A_t=\delta_t+\gamma\lambda\delta_{t+1}+(\gamma\lambda)^2\delta_{t+2}+\cdots
$$

递推式更适合实现：

$$
\hat A_t=\delta_t+\gamma\lambda \hat A_{t+1}
$$

这个式子说明，当前 advantage 等于“当前的 TD 修正”加上“未来 advantage 的折扣回传”。$\lambda$ 越大，未来信息保留得越多；$\lambda$ 越小，就越关注眼前一步。

退化情形也很直观：

$$
\lambda=0 \Rightarrow \hat A_t=\delta_t
$$

$$
\lambda=1 \Rightarrow \hat A_t=\sum_{l} \gamma^l \delta_{t+l}
$$

后者与 Monte Carlo 风格更接近，因为它更充分地吸收整段轨迹的长期信息。

### 玩具例子

设一条短轨迹：

- 奖励 $r=[1,0,2]$
- 价值预测 $V=[0.5,0.6,0.4,0]$
- 折扣因子 $\gamma=0.99$
- GAE 参数 $\lambda=0.95$

先算每一步 TD 残差：

$$
\delta_0=1+0.99\times0.6-0.5=1.094
$$

$$
\delta_1=0+0.99\times0.4-0.6=-0.204
$$

$$
\delta_2=2+0-0.4=1.6
$$

然后反向递推：

$$
\hat A_2=\delta_2=1.6
$$

$$
\hat A_1=\delta_1+0.99\times0.95\times1.6\approx1.3008
$$

$$
\hat A_0=\delta_0+0.99\times0.95\times1.3008\approx2.3174
$$

这个结果说明：虽然第 1 步的 TD 残差是负的，但第 2 步有一个很强的正向修正，于是第 0 步动作最终仍被判定为明显优于平均水平。GAE 的本质就是把这种“未来证据”以受控方式传回当前。

---

## 代码实现

工程里通常不会按展开式直接求和，而是**在 rollout 结束后按时间逆序计算**。原因很简单：递推式只需要保存一个累计变量，复杂度低，代码也不容易写错。

下面给出一个可运行的 Python 版本。它同时处理：

- `done=True`：真正终止，不能 bootstrap
- `truncated=True`：只是时间截断，通常还能 bootstrap
- 输出 `advantages` 和 `returns`

```python
from typing import List, Tuple

def compute_gae(
    rewards: List[float],
    values: List[float],          # len = T + 1
    dones: List[bool],            # true terminal
    truncated: List[bool],        # time limit / rollout cut
    gamma: float = 0.99,
    lam: float = 0.95,
) -> Tuple[List[float], List[float]]:
    T = len(rewards)
    assert len(values) == T + 1
    assert len(dones) == T
    assert len(truncated) == T

    adv = [0.0] * T
    gae = 0.0

    for t in reversed(range(T)):
        if dones[t]:
            next_non_terminal = 0.0
        else:
            # 截断不是环境真正终止，通常允许 bootstrap
            next_non_terminal = 1.0

        delta = rewards[t] + gamma * values[t + 1] * next_non_terminal - values[t]
        gae = delta + gamma * lam * next_non_terminal * gae
        adv[t] = gae

    returns = [a + v for a, v in zip(adv, values[:-1])]
    return adv, returns


# 玩具例子
rewards = [1.0, 0.0, 2.0]
values = [0.5, 0.6, 0.4, 0.0]
dones = [False, False, True]
truncated = [False, False, False]

adv, ret = compute_gae(rewards, values, dones, truncated)

assert len(adv) == 3
assert round(adv[2], 4) == 1.6000
assert round(adv[1], 4) == 1.3008
assert round(adv[0], 4) == 2.3174
assert round(ret[0], 4) == round(adv[0] + values[0], 4)

print("advantages =", [round(x, 4) for x in adv])
print("returns =", [round(x, 4) for x in ret])
```

真实工程里，PPO 的数据流通常是：

1. 用当前策略采样一段 rollout
2. 用 critic 预测每个状态的 $V(s_t)$
3. 用 GAE 计算每个时间步的 `advantage`
4. 用 `returns = advantage + value` 训练 critic
5. 用 advantage 更新 actor

### 真实工程例子

以机器人控制或游戏操作这类连续决策任务为例，单条轨迹可能很长，奖励还很稀疏。若直接用整段 return 做 actor 更新，梯度会很抖；若只用单步 TD，又可能过度依赖 critic 的局部预测。GAE 在这里提供一个可调折中，因此几乎成为 PPO 的默认配置之一。很多实现还会做 advantage 标准化：

$$
\tilde A_t=\frac{A_t-\mu}{\sigma+\epsilon}
$$

它的作用不是改变方向，而是控制梯度尺度，让训练更稳定。

---

## 工程权衡与常见坑

GAE 的核心不是“越大越好”，而是**估计器形状由 $\lambda$ 控制**。你可以把它理解为一只调节平滑程度的旋钮。

| 现象 | 根因 | 后果 |
| --- | --- | --- |
| $\lambda$ 过小 | 太依赖单步 TD | 方差低，但偏差大 |
| $\lambda$ 过大 | 太依赖长程累计 | 偏差低，但方差高 |
| critic 很差还用大 $\lambda$ | 误差被沿时间传播 | advantage 污染更严重 |
| 终止和截断混淆 | bootstrap 条件错 | returns 与 advantage 系统性偏差 |

最常见的坑有四个。

第一，**把 `done` 和 `truncated` 混为一谈**。真正终止时，下一个状态没有后续价值，应当令 $V_{t+1}=0$；但如果只是环境时间上限到了，或 rollout 人为截断，后面其实还有未来，应继续用 critic bootstrap。很多新手把两者都当终止处理，会让最后几步 advantage 偏小，进而低估尾部动作的价值。

第二，**把 $\gamma$ 和 $\lambda$ 当成同一类参数**。它们不是一回事。$\gamma$ 主要控制“未来奖励值多少钱”，$\lambda$ 主要控制“估计器把多远的 TD 误差信息传回来”。一个偏任务定义，一个偏估计器形状。

第三，**critic 太差时盲目调大 $\lambda$**。GAE 不是免费午餐。若 $V(s)$ 预测误差已经很大，长链式残差累积可能把坏信息传得更远。此时不应只盯着 actor 学不动，更应该检查 critic loss、reward scale、value clipping 等基础问题。

第四，**在 on-policy 训练中混入旧数据**。GAE 是基于当前策略附近的数据分布构造的 advantage。如果把旧策略生成的数据长期混用，actor 更新方向会偏，PPO 的近端约束也会被削弱。

实践中常见经验是：

- $\lambda$ 先从 `0.95` 起试
- `gamma` 常用 `0.99` 或 `0.995`
- 对 advantage 做标准化
- 单独监控 actor loss、critic loss、explained variance

---

## 替代方案与适用边界

GAE 不是唯一方案，它更像是 TD 和 Monte Carlo 之间的一条连续光谱。

| 方法 | 偏差 | 方差 | 特点 | 适用场景 |
| --- | --- | --- | --- | --- |
| TD(0) | 高 | 低 | 只看一步，最稳但最短视 | 短回合、实现优先 |
| Monte Carlo | 低 | 高 | 用整段回报，噪声大 | 回合短、终局明确信号强 |
| n-step return | 中等 | 中等 | 固定看 n 步 | 需要简单折中 |
| GAE | 可调 | 可调 | 用 $\lambda$ 连续调节 | PPO/TRPO 等 on-policy |

如果你只想先把 actor-critic 跑通，而且环境回合短、奖励密，TD(0) 可能已经够用。它简单、便宜、稳定。

如果环境回合不长，且最终得分能比较直接反映每一步动作质量，Monte Carlo 也可以工作，但通常梯度波动更大。

如果你希望在实现复杂度和效果之间平衡，n-step return 是一个中间方案。不过它的窗口长度是硬切的，不像 GAE 那样能用指数衰减平滑地整合多步信息。

GAE 最适合的边界是：

- on-policy 算法，尤其 PPO、TRPO、A2C/A3C 一类 actor-critic
- 轨迹较长、单步 TD 太短视、纯 Monte Carlo 方差又太大的任务
- 有一个可用但不完美的 critic，可以提供 bootstrap

它不应被机械照搬到所有场景。比如很多 off-policy 或离线强化学习方法，数据分布和目标构造方式不同，优势估计往往要配合重要性采样、重加权或 Q-learning 体系单独设计。

---

## 参考资料

1. [High-Dimensional Continuous Control Using Generalized Advantage Estimation](https://arxiv.org/abs/1506.02438)
2. [OpenAI Spinning Up: RL Intro](https://spinningup.openai.com/en/latest/spinningup/rl_intro.html)
3. [OpenAI Spinning Up: PPO](https://spinningup.openai.com/en/latest/algorithms/ppo.html)
4. [OpenAI Spinning Up: VPG 源码中的 GAE 实现](https://spinningup.openai.com/en/latest/_modules/spinup/algos/pytorch/vpg/vpg.html)
