## 核心结论

Actor-Critic 是一种把“直接学策略”和“学价值打分”组合起来的强化学习架构。Actor 是策略网络，白话讲就是“负责做决定的模块”；Critic 是价值网络，白话讲就是“负责判断当前决策值不值得的模块”。Actor 学 $\pi_\theta(a|s)$，Critic 学 $V_\phi(s)$ 或 $Q_\phi(s,a)$，两者一起训练。

它解决的核心问题不是“让强化学习变简单”，而是“让策略梯度的更新信号更稳定”。纯策略梯度方法如 REINFORCE 直接用回报 $G_t$ 更新策略，方向通常是对的，但方差很大，白话讲就是“有时靠运气学到，有时被噪声带偏”。Actor-Critic 引入 Critic 估计一个基准水平，再让 Actor 只关心“这个动作是否比基准更好”，于是更新更稳。

一个最短公式就是：

$$
\hat A_t = G_t - V_\phi(s_t)
$$

这里 $\hat A_t$ 叫优势函数估计，白话讲就是“这一步到底比平均水平好多少”。如果 $\hat A_t > 0$，说明动作比预期更好，应该提高它的概率；如果 $\hat A_t < 0$，就应该降低。

从工程视角看，Actor-Critic 不是单一算法，而是一类架构。A2C、A3C、PPO、DDPG、SAC 都可以看成它的扩展或变体。你需要先理解它的共性：Actor 负责优化策略，Critic 负责提供低方差训练信号。

---

## 问题定义与边界

Actor-Critic 要解决的问题，可以先写成一句更精确的话：在序列决策任务中，如何在直接优化策略的同时，把梯度估计的方差控制在可训练范围内。

先统一符号。术语第一次出现时，先给白话解释：

| 符号 | 含义 | 白话解释 |
| --- | --- | --- |
| $s_t$ | 状态 | 智能体当前看到的环境信息 |
| $a_t$ | 动作 | 智能体当前做出的选择 |
| $r_t$ | 奖励 | 环境对这一步的反馈分数 |
| $\gamma$ | 折扣因子 | 未来奖励在今天值多少钱 |
| $\pi_\theta(a|s)$ | 策略 | 在状态 $s$ 下选动作 $a$ 的概率规则 |
| $V_\phi(s)$ | 状态价值 | 从状态 $s$ 出发未来总回报的估计 |
| $Q_\phi(s,a)$ | 动作价值 | 在状态 $s$ 做动作 $a$ 后未来总回报的估计 |
| $\delta_t$ | TD 误差 | 当前价值估计和一步目标之间的差 |
| $\hat A_t$ | 优势估计 | 当前动作比平均水平好还是差 |

为什么纯策略梯度会不稳？因为它常常只看完整轨迹的最终结果。白话讲，这像“只看期末总分，再回头猜哪道题该多练”。如果一条轨迹很长，单步动作对最终回报的贡献很难分清，梯度噪声就会很大。

玩具例子：一个网格世界里，智能体从左下角走到右上角，每走一步奖励为 $-1$，到终点奖励为 $+20$。如果只用完整回报更新，早期动作是否有价值，要等整局结束后才知道；而 Critic 可以在中途估计“离终点更近的状态更值钱”，这样 Actor 能更早收到方向信号。

它的适用边界也要说清楚：

| 场景 | 是否适合 Actor-Critic | 原因 |
| --- | --- | --- |
| 连续动作控制 | 很适合 | 策略网络直接输出连续动作分布更自然 |
| 需要稳定策略优化 | 适合 | Critic 可降低策略梯度方差 |
| 极度稀疏奖励 | 不一定 | Critic 也可能学不到有效信号 |
| 强依赖旧样本反复训练 | 经典 on-policy 版本不适合 | 旧轨迹分布过时后会引入偏差 |
| 纯离散动作且价值函数很好学 | 未必最优 | DQN 类方法可能更直接 |

所以 Actor-Critic 不是“强化学习默认答案”。它更像一个折中架构：比纯策略梯度稳定，比纯值函数法更直接地优化策略，但代价是多了一个 Critic，要额外处理偏差、耦合和训练稳定性。

---

## 核心机制与推导

先从策略梯度出发。目标是最大化期望回报，常写成 $J(\theta)$。策略梯度的核心形式可以简化理解为：

$$
\nabla_\theta J(\theta) \propto \mathbb{E}\left[\nabla_\theta \log \pi_\theta(a_t|s_t)\, G_t\right]
$$

问题在于 $G_t$ 波动太大。于是引入 baseline，baseline 白话讲就是“拿一个不依赖动作的参考分数来减噪声”。常见选择就是状态价值函数 $V_\phi(s_t)$。于是得到优势：

$$
\hat A_t = G_t - V_\phi(s_t)
$$

更新逻辑变成：不再问“这步最终赚了多少”，而是问“这步比当前状态的平均预期多赚了多少”。这样，Actor 的损失常写为：

$$
L_\pi(\theta) = - \log \pi_\theta(a_t|s_t)\,\text{stopgrad}(\hat A_t)
$$

这里 `stopgrad` 的意思是“阻断梯度”，白话讲就是“把优势值当成常数使用，不让 Actor 反向改写 Critic 的估计目标”。

Critic 怎么学？如果直接用完整回报 $G_t$ 拟合 $V(s_t)$，虽然也行，但又会回到高方差问题。于是通常使用 TD，Temporal Difference，时序差分，白话讲就是“拿一步后的估计值来构造更短、更稳定的监督信号”。

TD 误差定义为：

$$
\delta_t = r_t + \gamma V_\phi(s_{t+1}) - V_\phi(s_t)
$$

Critic 的损失写成：

$$
L_V(\phi) = \frac{1}{2}\delta_t^2
$$

这意味着 Critic 在做一件具体的事：让当前价值估计接近“一步奖励 + 下一状态的折现价值”。

玩具例子可以直接算：

- $\gamma = 0.99$
- 当前奖励 $r_t = 1$
- $V_\phi(s_t)=10$
- $V_\phi(s_{t+1})=10.5$

则一步目标为：

$$
y_t = 1 + 0.99 \times 10.5 = 11.395
$$

于是

$$
\delta_t = 11.395 - 10 = 1.395
$$

结论很直接：当前动作后的结果，比 Critic 原本预估的更好，所以 Actor 应提高该动作概率，Critic 也应把 $V_\phi(s_t)$ 往 $11.395$ 修正。

真实工程例子：在无人机姿态控制中，动作往往是连续值，比如油门、俯仰角速度、横滚角速度。DQN 这类方法需要在动作空间上取 $\arg\max_a Q(s,a)$，对连续动作不自然；Actor 可以直接输出高斯分布参数，Critic 再估计当前状态值，训练路径就更顺。这个场景里，Actor-Critic 的价值不在“概念先进”，而在“动作空间和优化目标匹配”。

A2C 与 A3C 的差别也属于机制的一部分：

| 方法 | 更新方式 | 采样方式 | 经验回放 | 工程特点 |
| --- | --- | --- | --- | --- |
| A2C | 同步 | 多环境并行后统一更新 | 通常不用 | 易向量化，GPU 友好 |
| A3C | 异步 | 多线程各自采样并异步更新 | 不用 | CPU 并行强，但实现更复杂 |

A3C 的关键点是多线程异步更新，无需经验回放；A2C 可以看成它的同步版本，更适合现代批量训练流水线。

---

## 代码实现

实现 Actor-Critic 时，最重要的不是网络写法，而是训练信号的边界是否正确。你至少要分清四步：Actor 前向、Critic 前向、构造 TD target、分别更新两个损失。

下面是一个可运行的最小 Python 例子，用标量参数演示更新方向，不依赖深度学习框架，但逻辑与真实实现一致：

```python
import math

gamma = 0.99

# 一个极简 actor: 两个动作的 logits
logits = [0.2, -0.1]
action = 0  # 假设采样到了动作 0

# 一个极简 critic: 对当前状态和下一状态的价值估计
value_s = 10.0
value_s_next = 10.5
reward = 1.0

def softmax(xs):
    exps = [math.exp(x) for x in xs]
    s = sum(exps)
    return [x / s for x in exps]

probs = softmax(logits)
log_prob = math.log(probs[action])

td_target = reward + gamma * value_s_next
advantage = td_target - value_s

actor_loss = -log_prob * advantage
critic_loss = 0.5 * (td_target - value_s) ** 2

assert abs(td_target - 11.395) < 1e-9
assert abs(advantage - 1.395) < 1e-9
assert actor_loss > 0
assert critic_loss > 0

# 用符号判断更新方向：
# advantage > 0，说明应该提高当前动作概率
should_increase_action_prob = advantage > 0
assert should_increase_action_prob is True

print("probs =", probs)
print("td_target =", td_target)
print("advantage =", advantage)
print("actor_loss =", actor_loss)
print("critic_loss =", critic_loss)
```

如果换成 PyTorch，核心结构通常就是下面这样：

```python
# actor: pi_theta(a|s)
# critic: V_phi(s)

value = critic(s)
next_value = critic(s_next).detach()

td_target = r + gamma * next_value
advantage = td_target - value

actor_loss = -(log_prob * advantage.detach()).mean()
critic_loss = 0.5 * (td_target - value).pow(2).mean()

loss = actor_loss + critic_loss
optimizer.zero_grad()
loss.backward()
optimizer.step()
```

训练步骤可以压缩成一张表：

| 步骤 | 输入 | 输出 | 目的 |
| --- | --- | --- | --- |
| 采样轨迹 | 当前策略 | $(s_t,a_t,r_t,s_{t+1})$ | 收集训练数据 |
| Actor 前向 | $s_t$ | 动作分布、$\log \pi(a_t|s_t)$ | 给策略更新提供概率项 |
| Critic 前向 | $s_t,s_{t+1}$ | $V(s_t),V(s_{t+1})$ | 构造价值监督信号 |
| 计算 TD target | $r_t,V(s_{t+1})$ | $y_t$ | 形成低方差目标 |
| 计算 advantage | $y_t,V(s_t)$ | $\hat A_t$ | 决定策略该升还是该降 |
| 更新网络 | 两个损失 | 新参数 | 完成一轮学习 |

初学者最容易忽略的是：Actor 和 Critic 可以共享底层特征提取器，也可以完全独立。共享的好处是参数少、表示可复用；风险是两个目标互相拉扯。图像输入任务里常见共享 backbone，低维状态控制任务里也常见直接分开，具体要看稳定性。

---

## 工程权衡与常见坑

Actor-Critic 在工程上最大的真相是：Critic 一旦学歪，Actor 会被系统性带偏。因为 Actor 不是直接面对真实回报，而是面对 Critic 生成的训练信号。

常见坑可以直接列出来：

| 问题 | 现象 | 原因 | 修正方式 |
| --- | --- | --- | --- |
| 直接用原始回报更新 Actor | 曲线抖动大 | 方差高 | 改用 advantage |
| TD target 不 detach | 梯度异常耦合 | Actor/Critic 目标串扰 | 显式切断 target 梯度 |
| 奖励尺度过大 | loss 爆炸 | 优势值不受控 | 奖励裁剪或归一化 |
| Critic 学得太慢 | Actor 乱学 | 优势估计质量差 | 提高 Critic 更新频率 |
| 共享 backbone 互相干扰 | 一个升一个降 | 表示层目标冲突 | 分离网络或分设学习率 |
| 策略熵塌缩太快 | 过早只选单一动作 | 探索不足 | 加 entropy bonus |

训练诊断时，不要只看总回报曲线。至少要看四件事：

| 监控项 | 正常信号 | 异常信号 |
| --- | --- | --- |
| advantage 均值 | 接近 0 附近波动 | 长期单边偏正或偏负 |
| critic loss | 先降后稳 | 长期爆炸或不下降 |
| policy entropy | 缓慢下降 | 很快塌缩到极低 |
| episode return | 整体抬升 | 长期高频震荡 |

真实工程例子：推荐系统中的页面排序策略，如果 Critic 用短期点击率近似长期价值，但数据分布每天都在漂移，那么 Critic 容易失真。结果是 Actor 会偏向短期刺激性内容，因为它“以为”那代表长期收益。这不是公式错了，而是价值函数定义、奖励设计、采样分布三件事没有对齐。

还有一个常见误解：A3C 多线程并行，所以是不是该配经验回放？经典答案是否定的。A3C 本身是异步 on-policy 近似，不依赖经验回放来打破样本相关性；它靠的是多线程环境并行带来的状态分布去相关。如果你把非常旧的轨迹长期拿回来反复训练，分布偏差会变大。

---

## 替代方案与适用边界

Actor-Critic 不是唯一解。选方法时，至少要看三件事：动作空间是离散还是连续，样本采集贵不贵，训练稳定性要求多高。

| 方法 | 动作空间 | on-policy | 稳定性 | 连续控制 |
| --- | --- | --- | --- | --- |
| REINFORCE | 离散/连续 | 是 | 较差 | 可以，但方差大 |
| DQN | 主要离散 | 否 | 中等 | 不自然 |
| A2C/A3C | 离散/连续 | 是 | 中等 | 适合 |
| PPO | 离散/连续 | 近似是 | 较好 | 很适合 |
| DDPG/SAC | 连续为主 | 否 | 依实现而定 | 很适合 |

具体边界可以这样判断：

- 如果你只想理解“策略网络如何借助价值网络稳定训练”，Actor-Critic 是必须掌握的基础框架。
- 如果任务是连续控制，比如机械臂、赛车转向、机器人步态，Actor-Critic 比纯 Q-learning 更自然。
- 如果你更重视训练稳定、复现性和社区现成实现，PPO 往往比基础 A2C/A3C 更常用。
- 如果动作空间离散且规模不大，DQN 可能更直接，因为它不需要显式维护策略分布。
- 如果数据非常贵，经典 on-policy Actor-Critic 的样本利用率往往不够，需要看 off-policy 变体。

可以把这些方法理解成不同折中点。REINFORCE 简单但噪声大；DQN 样本利用率高但更偏值函数；A2C/A3C 是经典 Actor-Critic；PPO 则是在 Actor-Critic 基础上进一步约束更新步长，让训练更稳。

所以“什么时候用 Actor-Critic”的更准确说法是：当你需要直接优化策略，并且愿意引入一个价值估计器来换取更稳定的梯度时，它是合理起点；当你需要更强稳定性或更高样本利用率时，应进一步看 PPO、SAC 等变体。

---

## 参考资料

1. [Policy Gradient Methods for Reinforcement Learning with Function Approximation](https://papers.nips.cc/paper/1713-policy-gradient-methods-for-reinforcement-learning-with-function-approximation)
2. [Asynchronous Methods for Deep Reinforcement Learning](https://arxiv.org/abs/1602.01783)
3. [OpenAI Spinning Up: Vanilla Policy Gradient](https://spinningup.openai.com/en/latest/algorithms/vpg.html)
4. [OpenAI Baselines: ACKTR & A2C](https://openai.com/index/openai-baselines-acktr-a2c/)
5. [Stable-Baselines3: A2C](https://stable-baselines3.readthedocs.io/en/sde/modules/a2c.html)
