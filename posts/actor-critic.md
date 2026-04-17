## 核心结论

Actor-Critic 的核心不是“两个网络一起训练”这么简单，而是把“怎么选动作”和“这个动作到底值不值”拆开处理。Actor 是策略网络，白话解释就是“负责做选择的部分”；Critic 是价值网络，白话解释就是“负责打分的部分”。Actor 负责按概率采样动作，Critic 负责估计状态价值或动作优势，然后把这个评价信号回传给 Actor，指导策略更新。

它比纯策略梯度更稳定，因为策略梯度直接用回报更新时方差很大，训练容易抖动。Actor-Critic 通过优势函数降低这种抖动。优势函数的定义是：

$$
A(s,a)=Q(s,a)-V(s)
$$

一句话解释：它衡量“这个动作相对当前状态平均水平，到底多做对了多少，或者做错了多少”。

新手可以把它理解成：Actor 像“直觉选手”，看到状态就先做动作；Critic 像“裁判”，给当前状态和动作结果打分；优势就是“这次动作比平均表现高还是低”。如果优势为正，说明这步值得多学；如果优势为负，说明这步以后应少做。

A2C 和 A3C 则是在这个框架上进一步解决训练效率问题。A2C 是同步更新，多个环境一起采样，再统一更新；A3C 是异步更新，不同 worker 独立跑环境并直接更新共享参数。它们的共同目标都是让经验采集更快、梯度更稳定。

---

## 问题定义与边界

强化学习的目标是最大化长期折扣奖励：

$$
J(\theta)=\mathbb{E}_{\pi_\theta}\left[\sum_{t=0}^{\infty}\gamma^t r_t\right]
$$

这里的折扣因子 $\gamma$，白话解释就是“未来奖励打几折后再算进今天的决策”。当 $\gamma$ 越大，模型越重视长期收益；越小，则更看重眼前收益。

问题在于，策略学习和价值学习都不容易。只学策略，样本效率低；只学价值，动作空间一大就难直接选最优动作。Actor-Critic 的边界就在这里：它同时维护策略和价值，用价值信号帮助策略更新，但这也带来训练耦合、超参数敏感和并行同步复杂度。

下面是 Actor-Critic 与其他经典方向的边界对比：

| 方法 | 学什么 | 样本效率 | 稳定性 | 适用边界 |
|---|---|---:|---:|---|
| 纯策略梯度 | 只学策略 | 低 | 中低 | 动作采样直接、实现简单，但方差大 |
| 值迭代类 | 只学价值 | 中到高 | 中 | 离散动作效果好，连续动作处理麻烦 |
| Actor-Critic | 策略 + 价值 | 中高 | 中高 | 通用性强，是很多现代算法基础 |
| A3C | 异步 Actor-Critic | 中高 | 中 | 无经验回放、可并行，但参数延迟明显 |

新手版本可以这样理解：几个机器人在不同房间同时练习同一个游戏。每个机器人都把自己的尝试结果回传给中央参数。A3C 不要求所有机器人等齐再更新，所以速度快；但因为不同机器人看到的是稍旧参数，训练时会有“不同步”的副作用。

真实工程里，这类方法常见于 Atari、机器人控制、简单在线决策系统。边界也很明确：如果环境高度非定常，或者 reward 极其稀疏，Critic 很难学稳，Actor 就会收到噪声信号；如果动作空间连续且高维，普通 A3C 往往不如 PPO、SAC 一类方法稳健。

---

## 核心机制与推导

策略梯度的核心公式是：

$$
\nabla_\theta J(\theta)=\mathbb{E}_{\pi_\theta}\left[A^\pi(S_t,A_t)\nabla_\theta \log \pi_\theta(A_t|S_t)\right]
$$

$\log \pi_\theta(A_t|S_t)$ 的梯度，白话解释就是“让当前动作概率变大或变小的方向”；前面的优势 $A^\pi$ 则决定“该推多大力”。如果优势为正，就增加该动作概率；如果优势为负，就降低该动作概率。

Critic 常用 TD 误差训练。TD，Temporal Difference，白话解释就是“拿当前估计和一步后的估计做差，边走边改”。公式是：

$$
\delta_t=r_t+\gamma V(s_{t+1})-V(s_t)
$$

如果 $\delta_t>0$，说明这一步实际结果比当前估计更好；如果 $\delta_t<0$，说明估高了。

玩具例子如下。设 $\gamma=0.9$，当前 Critic 估计 $V(s)=5$，执行动作后得到奖励 $r=3$，下一个状态估计 $V(s')=4$，则：

$$
\delta=3+0.9\times4-5=1.6
$$

这表示“这个动作比当前估计明显更好”，Actor 应提高该动作概率。若奖励改成 $r=1$，则：

$$
\delta=1+0.9\times4-5=-0.4
$$

这表示动作效果低于预期，应减弱这个动作。

更常见的优势估计不是只看一步，而是看 $n$ 步回报：

$$
A_t^{(n)}\approx \sum_{k=0}^{n-1}\gamma^k r_{t+k}+\gamma^n V(s_{t+n})-V(s_t)
$$

$n$ 越小，偏差更大但方差更小；$n$ 越大，偏差更小但方差更大。GAE，Generalized Advantage Estimation，白话解释就是“把多个步长的优势做指数加权平均”，用参数 $\lambda$ 控制偏差和方差平衡。它常写成：

$$
A_t^{GAE}=\sum_{l=0}^{\infty}(\gamma\lambda)^l \delta_{t+l}
$$

流程可以简化为：

```text
环境 state
  ↓
Actor 输出动作分布 π(a|s)
  ↓
采样动作并与环境交互
  ↓
得到奖励 r 和下一状态 s'
  ↓
Critic 计算 V(s), V(s') 与 TD 误差 δ
  ↓
由 δ 或 Advantage 更新 Actor
  ↓
由 TD 目标更新 Critic
  ↓
并行 worker 重复该过程
```

真实工程例子：在 Atari 训练中，多个 worker 各自维护环境实例，同时共享全局网络参数。每个 worker 连续跑若干步，把轨迹切成 n-step 片段，计算优势后回传梯度。这样能明显提高吞吐量，也减少单个环境采样的相关性。

---

## 代码实现

下面先给一个可运行的玩具实现，只展示优势计算和更新方向，帮助理解公式如何落地。

```python
import math

def td_error(reward, gamma, next_v, current_v):
    return reward + gamma * next_v - current_v

def normalize(xs):
    mean = sum(xs) / len(xs)
    var = sum((x - mean) ** 2 for x in xs) / len(xs)
    std = math.sqrt(var) + 1e-8
    return [(x - mean) / std for x in xs]

gamma = 0.9
current_v = 5.0
next_v = 4.0

delta_good = td_error(3.0, gamma, next_v, current_v)
delta_bad = td_error(1.0, gamma, next_v, current_v)

assert round(delta_good, 2) == 1.60
assert round(delta_bad, 2) == -0.40
assert delta_good > 0
assert delta_bad < 0

advantages = [delta_good, delta_bad, 0.2, -0.1]
norm_adv = normalize(advantages)

assert len(norm_adv) == 4
assert abs(sum(norm_adv)) < 1e-6

# 如果优势为正，应提升动作对数概率对应的目标；反之下降
log_prob = math.log(0.6)
actor_loss_signal = -log_prob * delta_good
assert actor_loss_signal > 0
```

上面这段代码只验证三件事：TD 误差的正负号、优势可标准化、Actor loss 会随着正优势推动对应动作概率上升。

再看更接近 PyTorch 风格的核心训练流程。这里的 `advantages` 就是 Critic 评估好的增益信号。

```python
import torch
import torch.nn.functional as F

def compute_nstep_returns(rewards, dones, last_value, gamma):
    returns = []
    R = last_value
    for r, d in zip(reversed(rewards), reversed(dones)):
        R = r + gamma * R * (1.0 - d)
        returns.append(R)
    returns.reverse()
    return torch.tensor(returns, dtype=torch.float32)

# 假设：
# actor(states) -> logits
# critic(states) -> values
# 多环境采样得到一批轨迹
states = torch.randn(8, 4)
next_states = torch.randn(8, 4)
actions = torch.tensor([0, 1, 0, 1, 1, 0, 0, 1])
rewards = [1.0, 0.2, -0.5, 1.2, 0.3, 0.0, 0.8, -0.1]
dones = [0, 0, 0, 0, 0, 0, 0, 1]

logits = actor(states)
values = critic(states).squeeze(-1)

with torch.no_grad():
    last_value = critic(next_states[-1:]).item()
    returns = compute_nstep_returns(rewards, dones, last_value, gamma=0.99)

advantages = returns - values.detach()
advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

log_probs = F.log_softmax(logits, dim=-1)
chosen_log_probs = log_probs[torch.arange(actions.size(0)), actions]

entropy = -(F.softmax(logits, dim=-1) * log_probs).sum(dim=-1).mean()
actor_loss = -(chosen_log_probs * advantages).mean() - 0.01 * entropy
critic_loss = F.mse_loss(values, returns)

loss = actor_loss + 0.5 * critic_loss

optimizer.zero_grad()
loss.backward()
torch.nn.utils.clip_grad_norm_(list(actor.parameters()) + list(critic.parameters()), 0.5)
optimizer.step()
```

这个循环的关键顺序是：收集 n-step 轨迹 -> 计算 returns 和 advantages -> 更新 Actor 与 Critic -> 清空缓存继续采样。

A3C 额外多一层参数共享。伪代码结构如下：

```text
global_actor, global_critic 共享在主进程
for each worker in parallel:
    同步一份局部参数
    与独立环境交互 n 步
    计算局部 advantage 和 loss
    反向传播到全局参数
    局部参数重新从全局拉取
```

A2C 则更简单：多个环境先同步跑完一批，再集中做一次 batch 更新。它没有异步延迟问题，所以更容易复现。

---

## 工程权衡与常见坑

Actor-Critic 的关键不在“能不能跑”，而在“信号是不是可信”。如果 Critic 学不好，Actor 更新方向就会偏。常见坑如下：

| 问题 | 现象 | 原因 | 常用规避策略 |
|---|---|---|---|
| `n-step` 太短 | 学得快但上限低 | 偏差大 | 增大 `n` 或引入 GAE |
| `n-step` 太长 | 回报剧烈波动 | 方差大 | 缩短 rollout，做优势标准化 |
| 学习率过大 | loss 抖动、策略崩掉 | 多 worker 梯度互相覆盖 | 降低 `lr`，加梯度裁剪 |
| Critic 过拟合 | value loss 很低但策略不涨 | 只记住局部轨迹 | 正则化、更多环境随机性 |
| Critic 太弱 | advantage 噪声大 | 价值目标不准 | 提高 Critic 容量或训练步数 |
| 熵过低 | 提前只选少数动作 | 探索不足 | 增大熵系数 |
| 异步延迟 | A3C 不稳定 | worker 使用旧参数 | 减少本地 rollout 长度 |

新手视角可以这样理解：16 个 worker 同时更新时，如果学习率太大，就像 16 个厨师同时往锅里加盐。每个人都觉得自己只加了一点，但整体很容易失控。所以 A3C 虽然常用 lock-free 更新，仍然需要更小步长和更强约束。

实践里经常加下面这句优势标准化：

```python
advantage = (advantage - advantage.mean()) / (advantage.std() + 1e-8)
```

它的作用不是改变方向，而是把尺度拉回可控范围，防止某几个极端样本主导梯度。

还有一个常见误区：把 Actor loss 和 Critic loss 权重设成一样。实际上 Critic 通常需要更稳的监督信号，因此很多实现会用 `loss = actor_loss + 0.5 * critic_loss` 或相近比例。再配合梯度裁剪，训练曲线会平滑很多。

---

## 替代方案与适用边界

A2C、A3C、PPO、SAC 都能看作“策略更新 + 价值估计”的不同工程实现，但它们关注点不同。

新手可以先抓住这个类比：A3C 像多人同时练习射箭，各自练、各自报结果；A2C 像大家先练完这一轮，再一起复盘；PPO 像每轮只允许做小幅修正，避免动作变形太快；SAC 则额外鼓励探索，让策略不要太早保守。

| 方法 | 同步方式 | 目标函数特点 | 资源需求 | 优势 | 边界 |
|---|---|---|---|---|---|
| A3C | 异步 | 标准策略梯度 + value loss | CPU 友好，多线程 | 吞吐高，无需回放 | 参数延迟，不易复现 |
| A2C | 同步 | A3C 的同步版 | 批量环境 | 更稳定，更易调试 | 吞吐略低于异步 |
| PPO | 同步 | 裁剪目标限制更新幅度 | 常配 GPU | 稳定、工业界常用 | 采样后要多轮优化 |
| SAC | 同步 + 回放 | 最大熵目标 | 需要回放缓冲 | 连续动作强、样本效率高 | 结构更复杂 |

适用边界可以直接记成三条：

1. A3C 适合资源受限、环境轻量、希望靠并行提升采样效率的场景。
2. A2C 适合教学、复现和中小规模实验，因为同步更新更容易定位问题。
3. PPO 更适合大多数现代策略优化任务，尤其是需要稳定训练时。
4. SAC/DDPG 更适合连续动作控制，例如机械臂、速度控制、轨迹规划，但依赖经验回放和目标网络。

如果是零基础读者第一次上手，建议先实现单环境 Actor-Critic，再实现多环境 A2C，最后再看 A3C。因为异步带来的问题并不是数学难，而是工程调试难。

---

## 参考资料

- GeeksforGeeks: 面向新手解释 A3C 的整体架构、同步/异步 worker 角色，适合先建立整体概念。
- EmergentMind: 提供 A3C/A2C 的公式、优势函数、策略梯度和伪代码，适合补全数学机制。
- Next Electronics: 更偏工程实践，讲了 TD 误差、优势标准化、GAE、学习率和并行训练中的常见问题。

建议阅读顺序如下：

1. GeeksforGeeks：先理解“Actor 选动作、Critic 打分、多个 worker 并行更新”的整体结构。
2. EmergentMind：再看 $\nabla_\theta J(\theta)$、$A(s,a)=Q(s,a)-V(s)$、TD 误差和 n-step 公式。
3. Next Electronics：最后补优势标准化、GAE、异步更新中的调参经验和数值例子。

对新手最有效的阅读方式不是一次看全，而是按这个顺序复述三句话：GeeksforGeeks 给出概念与并行训练图景；EmergentMind 细化公式和伪代码；Next Electronics 说明训练时为什么要做优势标准化和 GAE。
