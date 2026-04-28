## 核心结论

策略梯度定理解决的是一个直接问题：当策略 $\pi_\theta(a|s)$ 已经是一个可微函数时，怎样直接把“总回报更高”变成参数更新方向。

它的核心形式是：

$$
J(\theta)=\mathbb{E}_{\tau\sim \pi_\theta}\left[\sum_{t=0}^{T-1}\gamma^t r_t\right]
$$

$$
\nabla_\theta J(\theta)
=
\mathbb{E}_{\tau\sim \pi_\theta}
\left[
\sum_{t=0}^{T-1}
\nabla_\theta \log \pi_\theta(a_t|s_t)\cdot G_t
\right]
$$

其中，$\tau$ 是轨迹，白话说就是“智能体从开始到结束的一整段经历”；$G_t$ 是从时刻 $t$ 往后的折扣回报，白话说就是“这一步之后最终拿到了多少结果”。

这条定理最重要的含义不是公式本身，而是更新规则：

- 某个动作带来高回报，就增大它在当时状态下的概率。
- 某个动作带来低回报，就减小它在当时状态下的概率。
- 整个过程只需要策略的对数概率梯度，不需要对环境求导，也不需要先把最优价值函数完整求出来。

最简实现就是 REINFORCE。它可以看成“用采样得到的整段结果，直接反推哪些动作值得更常出现”。

---

## 问题定义与边界

强化学习不是普通监督学习。监督学习里，标签提前给定；强化学习里，动作会改变未来状态，未来状态又决定后续奖励，所以当前动作的好坏必须放到整条轨迹里判断。

本文讨论的问题是：给定一个参数化策略 $\pi_\theta(a|s)$，如何直接最大化期望回报 $J(\theta)$。

常用符号如下：

| 符号 | 含义 | 白话解释 |
|---|---|---|
| $s_t$ | 时刻 $t$ 的状态 | 当前看到的环境信息 |
| $a_t$ | 时刻 $t$ 的动作 | 当前做出的选择 |
| $r_t$ | 时刻 $t$ 的即时奖励 | 这一步立刻得到的分数 |
| $\tau$ | 一条完整轨迹 | 从开始到结束的经历 |
| $\gamma$ | 折扣因子 | 未来奖励按多大比例计入现在 |
| $G_t$ | 从 $t$ 开始的折扣回报 | 这一步之后总共赚了多少 |
| $\pi_\theta(a|s)$ | 策略 | 在状态 $s$ 下选择动作 $a$ 的概率 |

回报定义为：

$$
G_t=\sum_{k=t}^{T-1}\gamma^{k-t}r_k
$$

这里的折扣因子 $\gamma\in[0,1]$ 用来控制“未来值多少钱”。如果 $\gamma$ 越接近 1，算法越重视长期收益；如果更小，算法更偏向短期收益。

本文边界需要说清楚：

| 范围 | 本文覆盖 | 不重点展开 |
|---|---|---|
| 数据来源 | on-policy 采样 | 离线强化学习、强 off-policy 修正 |
| 策略类型 | 可微参数化策略 | 分层策略、复杂混合策略 |
| 核心目标 | 直接优化策略 | 价值迭代、Q-learning 主线 |
| 工程实现 | REINFORCE 与优势形式 | PPO、TRPO、SAC 的完整细节 |

这里的 on-policy，白话说就是“用当前策略自己采样的数据来更新当前策略”。这是最原始、最干净的策略梯度设定。

一个玩具例子可以帮助建立边界。设想只有一步决策：按钮 A 或按钮 B。按 A 平均得 2 分，按 B 平均得 0 分。监督学习没有标签告诉你“应该按哪个”；你只能不断尝试，然后根据最终得分，逐步把策略改成更常选 A。这就是最简单的策略优化问题。

---

## 核心机制与推导

从目标函数开始：

$$
J(\theta)=\mathbb{E}_{\tau\sim \pi_\theta}[R(\tau)]
$$

其中 $R(\tau)$ 是整条轨迹的总回报。难点在于：轨迹分布本身依赖参数 $\theta$，所以不能像普通监督学习那样直接把梯度推进期望里面。

关键技巧是对数导数技巧，也叫 log-derivative trick：

$$
\nabla_\theta p_\theta(x)=p_\theta(x)\nabla_\theta \log p_\theta(x)
$$

把它用于轨迹分布 $p_\theta(\tau)$，可得：

$$
\nabla_\theta J(\theta)
=
\nabla_\theta \int p_\theta(\tau)R(\tau)d\tau
=
\int p_\theta(\tau)\nabla_\theta \log p_\theta(\tau)R(\tau)d\tau
$$

也就是：

$$
\nabla_\theta J(\theta)
=
\mathbb{E}_{\tau\sim\pi_\theta}
[\nabla_\theta \log p_\theta(\tau)\cdot R(\tau)]
$$

接下来把轨迹概率拆开。若环境转移不依赖参数，则：

$$
p_\theta(\tau)=\rho(s_0)\prod_{t=0}^{T-1}\pi_\theta(a_t|s_t)P(s_{t+1}|s_t,a_t)
$$

其中 $\rho(s_0)$ 是初始状态分布，$P(s_{t+1}|s_t,a_t)$ 是环境转移概率。对数后求导，和参数有关的只剩策略项：

$$
\nabla_\theta \log p_\theta(\tau)
=
\sum_{t=0}^{T-1}\nabla_\theta \log \pi_\theta(a_t|s_t)
$$

于是得到：

$$
\nabla_\theta J(\theta)
=
\mathbb{E}_{\tau\sim\pi_\theta}
\left[
\left(\sum_{t=0}^{T-1}\nabla_\theta \log \pi_\theta(a_t|s_t)\right)R(\tau)
\right]
$$

再进一步，可以把整条轨迹总回报换成每一步对应的 reward-to-go，也就是从该时刻往后的回报 $G_t$：

$$
\nabla_\theta J(\theta)
=
\mathbb{E}_{\tau\sim\pi_\theta}
\left[
\sum_{t=0}^{T-1}\nabla_\theta \log \pi_\theta(a_t|s_t)\cdot G_t
\right]
$$

这就是常见的策略梯度定理表达。它说明，训练信号由两部分组成：

| 项 | 含义 | 作用 |
|---|---|---|
| $\nabla_\theta \log \pi_\theta(a_t|s_t)$ | 对数概率梯度 | 告诉参数“往哪里改会更偏向这个动作” |
| $G_t$ | 回报 | 告诉这次动作结果是好是坏 |
| 两者乘积 | 梯度贡献 | 决定该动作概率增大还是减小 |

为什么是对数概率而不是概率本身？因为这样可以把“改变采样分布”的问题变成“对已采样动作做加权更新”的问题，结果可以直接用蒙特卡洛采样估计。

看一个单步玩具例子。设二分类策略：

$$
\pi_\theta(a=1|s)=\sigma(\theta)=\frac{1}{1+e^{-\theta}}
$$

初始 $\theta=0$，所以 $p=0.5$。

如果采样到 $a=1$，并且这次回报 $G=2$，则：

$$
\frac{\partial}{\partial \theta}\log \pi_\theta(a=1|s)=1-p=0.5
$$

梯度贡献是 $0.5\times 2=1.0$。做梯度上升后，$\theta$ 变大，动作 1 的概率会升高。

如果采样到 $a=0$，并且这次优势是负的，比如 $A=-0.5$，则：

$$
\frac{\partial}{\partial \theta}\log \pi_\theta(a=0|s)=-p=-0.5
$$

梯度贡献为 $(-0.5)\times(-0.5)=0.25$。参数依然会朝“降低动作 0 概率”的方向移动。结论很直接：回报高的动作会被强化，回报低的动作会被抑制。

实际工程里通常不用原始 $G_t$，而用优势函数：

$$
A_t = G_t - b(s_t)
$$

其中 $b(s_t)$ 是 baseline，白话说就是“这个状态下的正常发挥基线”。它的作用是降低方差。因为如果某一步回报高，只说明“高于平时”才值得强化；如果只是状态本身容易得高分，不一定说明动作特别好。

进一步写成：

$$
\nabla_\theta J(\theta)
=
\mathbb{E}
\left[
\sum_t \nabla_\theta \log \pi_\theta(a_t|s_t)\cdot A_t
\right]
$$

这里有一个关键结论：只要 baseline 不依赖动作，它不会改变梯度期望，只会改变方差。直觉上，它只是把“绝对分数”改成“相对基线的超额分数”。

真实工程例子是机器人控制。比如四足机器人在 `Hopper` 或 `Ant` 环境里，策略网络输出高斯分布参数 $(\mu,\sigma)$，动作是每个关节的力矩。系统按当前策略采样一整批轨迹，计算每一步优势 $A_t$，再用这些优势去更新动作分布。这里并没有先求一个离散动作表，而是直接优化连续控制策略，这正是策略梯度方法的自然适用场景。

---

## 代码实现

最小实现要抓住一件事：优化器默认做最小化，所以策略梯度的损失通常写成负号形式：

$$
\text{loss} = -\mathbb{E}[\log \pi_\theta(a_t|s_t)\cdot A_t]
$$

这样最小化 `loss`，等价于最大化期望回报。

下面先给一个可运行的最小 Python 例子。它不依赖深度学习框架，只演示单参数 Bernoulli 策略怎样按 REINFORCE 更新。

```python
import math

def sigmoid(x):
    return 1.0 / (1.0 + math.exp(-x))

def logprob_grad(theta, action):
    p = sigmoid(theta)
    if action == 1:
        return 1.0 - p          # d/dtheta log(sigmoid(theta))
    else:
        return -p               # d/dtheta log(1 - sigmoid(theta))

def reinforce_step(theta, action, advantage, lr=0.1):
    grad = logprob_grad(theta, action) * advantage
    return theta + lr * grad    # gradient ascent

theta = 0.0
p0 = sigmoid(theta)
assert abs(p0 - 0.5) < 1e-8

# 采样到动作1，且结果很好
theta = reinforce_step(theta, action=1, advantage=2.0, lr=0.1)
p1 = sigmoid(theta)
assert p1 > p0

# 采样到动作0，且结果不好，应该继续压低动作0、提高动作1
theta2 = reinforce_step(theta, action=0, advantage=-1.0, lr=0.1)
p2 = sigmoid(theta2)
assert p2 > p1

print("p0 =", round(p0, 4))
print("p1 =", round(p1, 4))
print("p2 =", round(p2, 4))
```

这段代码验证了最核心的训练规律：

- 选到动作 1 且结果好，动作 1 概率上升。
- 选到动作 0 且结果差，动作 0 概率下降，也就是动作 1 概率继续上升。

把它推广到多步轨迹，流程如下：

```python
# 1. 用当前策略采样轨迹
traj = collect_trajectory(policy, env)

# 2. 计算每一步回报或优势
returns = compute_returns(traj.rewards, gamma)
advantages = returns - baseline(traj.states)

# 3. 取出每一步被采样动作的 log probability
logp = policy.log_prob(traj.states, traj.actions)

# 4. 构造策略损失
loss = -(logp * advantages).mean()

# 5. 反向传播更新
optimizer.zero_grad()
loss.backward()
optimizer.step()
```

如果是 PyTorch，`policy.log_prob(...)` 往往来自 `Categorical` 或 `Normal` 分布对象。离散动作时，网络输出 softmax 概率；连续动作时，网络输出均值和标准差，然后用高斯分布计算 `log_prob`。

这里必须强调一个实现细节：不要把整条轨迹只打一个总分再复制给每一步。更合理的做法是按时间步计算 $G_t$ 或 $A_t$，因为越靠后的动作只影响越靠后的奖励，信号应该更细。

一个真实工程例子是在线广告推荐中的策略探索。状态是用户画像和上下文，动作是推荐哪类内容，奖励可能是点击、停留时长或转化。若用随机策略做探索，策略网络输出各候选内容的概率，系统采样展示结果，再根据后续反馈构造优势，更新 `log_prob * advantage`。虽然工业系统往往不会直接用最裸的 REINFORCE，但其核心梯度形式依然相同。

---

## 工程权衡与常见坑

策略梯度方法的主要难点不是“梯度能不能写出来”，而是“梯度方差很大，训练容易不稳”。公式简单，不等于训练简单。

常见问题如下：

| 问题 | 后果 | 常见修正 |
|---|---|---|
| 损失符号写反 | 概率朝错误方向更新 | 用 `loss = -(logp * A).mean()` |
| 直接用整轨迹总回报 | 方差过大，学习很抖 | 改用 reward-to-go 或优势函数 |
| baseline 设计错误 | 训练偏差或无效降噪 | baseline 不应依赖动作 |
| 复用旧轨迹太多 | 引入偏差，目标不再匹配当前策略 | 保持 on-policy，或改用专门 off-policy 方法 |
| `done` 和截断混淆 | 回报计算错位 | 区分真正终止与时间截断 |
| 熵过快下降 | 策略过早确定，探索不足 | 加熵正则或调小学习率 |

第一个坑最常见。因为我们理论上做的是梯度上升，但大多数深度学习库默认是最小化损失，所以负号不能丢。

第二个坑是高方差。REINFORCE 直接使用蒙特卡洛回报，估计无偏，但噪声大。工程上常见的降方差手段包括：

- 用 $G_t$ 替代整轨迹总回报。
- 引入 baseline 或 value function。
- 对 advantage 做标准化，使其均值接近 0、方差接近 1。
- 增大 batch，减少单次更新的随机波动。

第三个坑是把 baseline 用错。baseline 的原则很严格：它可以依赖状态，不能依赖当前动作。否则就不是纯粹的减方差，而会改变期望梯度。

再看一个训练稳定性检查清单：

| 检查项 | 正常现象 | 异常信号 |
|---|---|---|
| advantage 均值 | 接近 0 | 长期偏正或偏负，说明估计有偏 |
| 策略熵 | 缓慢下降 | 很快塌缩到接近 0，说明探索不足 |
| `log_prob` 数值 | 有波动但有限 | 极大负值频繁出现，说明概率过尖或数值不稳 |
| 平均回报 | 总体上升 | 长期震荡或退化，说明方差/学习率有问题 |

还有一个认知误区：很多人把“高回报动作要被强化”理解成“每一步都应该直接乘最终总奖励”。这不够精确。更准确的说法是：每一步动作要用它之后的回报，或更稳定的优势，来衡量自己的贡献。否则早期动作和晚期动作会被同样粗糙地打分，信号分辨率太低。

---

## 替代方案与适用边界

REINFORCE 重要，但它更多是理论起点和教学基线，不是大多数工程项目的终点。

下面做一个对比：

| 方法 | 优点 | 缺点 | 适用场景 |
|---|---|---|---|
| REINFORCE | 推导直接、实现最简、无偏 | 方差大、样本效率低 | 教学、玩具环境、验证思路 |
| Actor-Critic | 用 critic 降低方差 | 需要额外训练价值函数 | 大多数实用强化学习任务 |
| PPO | 更新更稳、工程上常用 | 机制更复杂，超参数更多 | 通用工业训练 |
| SAC | 连续控制强、样本效率高 | 理论和实现都更复杂 | 连续动作、高成本采样场景 |

为什么后续方法大多仍然建立在策略梯度之上？因为它们本质上没有放弃“直接优化策略”这条路线，只是在估计方式、稳定性控制、采样效率上做了增强。

例如 Actor-Critic 中的 critic，白话说就是“专门负责评估当前状态值多少钱的辅助网络”。有了它，就可以把原始回报替换成更低方差的优势估计。PPO 则在此基础上进一步限制每次策略更新幅度，避免一步改太猛导致性能崩掉。

适用边界可以概括成两句：

- 当你需要随机策略、连续动作，或希望直接控制采样分布时，策略梯度是自然选择。
- 当你更关心样本效率、稳定性和大规模工程可用性时，通常不会停留在纯 REINFORCE，而会走向 Actor-Critic、PPO、SAC 等变体。

所以，理解策略梯度定理的价值，不是为了永远手写 REINFORCE，而是为了明白后续大多数策略优化算法到底在改进什么：它们主要改进的是估计方差、更新稳定性和数据利用效率，而不是推翻“高回报动作概率应该上升”这个核心逻辑。

---

## 参考资料

1. [Policy Gradient Methods for Reinforcement Learning with Function Approximation](https://papers.nips.cc/paper/1713-policy-gradient-methods-for-reinforcement-learning-with-function-approximation)
2. [Simple Statistical Gradient-Following Algorithms for Connectionist Reinforcement Learning](https://mlanthology.org/J92-3004/)
3. [OpenAI Spinning Up: Vanilla Policy Gradient](https://spinningup.openai.com/en/latest/algorithms/vpg.html)
4. [OpenAI Spinning Up: Intro to Policy Optimization](https://spinningup.openai.com/en/latest/spinningup/rl_intro3.html)
5. [PyTorch REINFORCE Example](https://github.com/pytorch/examples/blob/main/reinforcement_learning/reinforce.py)
