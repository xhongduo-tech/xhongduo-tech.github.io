## 核心结论

DDPG（Deep Deterministic Policy Gradient，深度确定性策略梯度）是面向**连续动作空间**的离策略 Actor-Critic 算法。连续动作空间的意思是动作不是“左/右/停”这种有限选项，而是 $[-1,1]$、$[0, 2.5]$ 这类实数范围。它的关键变化是：不用显式表示 $\pi(a|s)$ 这个动作分布，而是让策略网络直接输出一个确定动作 $a=\mu_\theta(s)$。

对初学者，最重要的判断标准只有一句话：**如果动作是连续数值，DQN 很难直接做；DDPG 的思路是让 Actor 直接给出这个数值，再让 Critic 判断这个数值好不好。**

| 算法类型 | 动作空间 | 策略形式 | 训练方式 | 适用场景 |
|---|---|---|---|---|
| DQN | 离散 | $\arg\max_a Q(s,a)$ | 离策略 | 选有限动作 |
| DDPG | 连续 | $a=\mu_\theta(s)$ | 离策略 | 控制扭矩、角度、速度 |
| 核心价值 | 连续 | 可微策略 | 用 Critic 反传给 Actor | 高维连续控制 |

一句对比可以记住：**DQN 是在候选动作里选一个，DDPG 是直接生成一个动作值。**

---

## 问题定义与边界

强化学习的目标是让智能体在环境中不断交互，最大化长期回报。状态 $s$ 可以理解为“当前环境观测”，动作 $a$ 是“此刻要执行的控制量”，奖励 $r$ 是“环境给的即时反馈”。

DDPG 讨论的问题通常写成：

$$
a = \mu_\theta(s)
$$

$$
Q(s,a) = \text{从状态 } s \text{ 执行动作 } a \text{ 后的长期价值估计}
$$

$$
G_t = \sum_{k=0}^{\infty}\gamma^k r_{t+k}
$$

其中 $\gamma \in [0,1)$ 是折扣因子，表示“未来奖励打几折”。

为什么连续动作难？因为你不能像 DQN 那样把所有动作都枚举出来。假设机械臂关节力矩范围是 $[-2,2]$，理论上里面有无穷多个实数。你没法逐个比较“0.371 和 0.372 哪个更优”，只能让模型学会直接输出一个数。

一个玩具例子：小车要沿直线停到目标点。离散动作可以设成“加速、减速、不动”，但如果你想让它平滑控制油门，动作就更像 $a \in [-1,1]$ 的连续值。DDPG 适合后者。

一个真实工程例子：无人机姿态控制。控制器通常要连续输出滚转角、俯仰角或电机推力修正量，这些量本身就是连续数值，不适合离散枚举。

边界要说清楚。DDPG 不是“所有强化学习问题”的通用解，它主要适合连续控制类任务。

| 任务特征 | 适合 DDPG | 不适合 DDPG |
|---|---|---|
| 动作空间 | 连续实数 | 明显离散动作 |
| 策略需求 | 确定性控制 | 需要显式随机探索 |
| 数据使用 | 可重复采样历史数据 | 必须严格在线更新 |
| 稳定性要求 | 可接受调参 | 希望开箱即稳 |
| 环境类型 | 机器人、控制、调参 | 文本决策、组合动作 |

还要区分三个概念：
离散动作：动作集合有限。
随机策略：同一状态下输出的是概率分布。
离策略：训练时可以使用旧策略收集到的数据，不要求样本一定来自当前策略。DDPG 属于离策略，这就是它能配合经验回放的原因。

---

## 核心机制与推导

DDPG 有两条主线。

第一条是 Critic。Critic 学习 $Q_\phi(s,a)$，也就是“这个状态配这个动作到底值不值”。它的目标值来自 bootstrap，白话说是“用下一步的估计来帮助当前学习”。

$$
y = r + \gamma Q_{\phi'}(s', \mu_{\theta'}(s'))
$$

$$
L_Q(\phi) = \mathbb{E}\left[(Q_\phi(s,a)-y)^2\right]
$$

这里的 $\phi'$ 和 $\theta'$ 是目标网络参数。目标网络可以理解为“更新慢一点的副本”，目的是避免监督目标自己剧烈漂移。

第二条是 Actor。Actor 不直接看真实奖励，而是沿着 Critic 给出的方向更新。核心梯度是：

$$
\nabla_\theta J \approx \mathbb{E}\left[\nabla_a Q_\phi(s,a)\vert_{a=\mu_\theta(s)} \cdot \nabla_\theta \mu_\theta(s)\right]
$$

这条式子表达的意思很直接：先看“动作往哪个方向改，Q 会变大”，再把这个方向通过链式法则传回策略参数。不是猜最优动作，而是**沿着更优方向一步步挪**。

玩具数值例子：假设某个状态下

$$
Q(s,a) = -(a-2)^2 + 5
$$

那么最优动作显然是 $a=2$。如果当前 Actor 输出 $a=0.5$，则

$$
\frac{\partial Q}{\partial a} = -2(a-2)
$$

代入 $a=0.5$，得到 $\frac{\partial Q}{\partial a}=3$，梯度为正，说明应该把动作往更大的方向推。DDPG 的 Actor 更新，本质上就是利用这个方向信息。

经验回放和目标网络几乎是 DDPG 必需组件。

经验回放（replay buffer，经验池）是把过去交互样本存起来再随机采样，解决“相邻样本太像，训练相关性太强”的问题。

目标网络解决的是 bootstrap 目标漂移。如果你一边学 $Q$，一边又立刻用最新的 $Q$ 去生成目标值，训练会非常抖。

软更新公式是：

$$
\theta' \leftarrow \tau \theta + (1-\tau)\theta'
$$

$$
\phi' \leftarrow \tau \phi + (1-\tau)\phi'
$$

其中 $\tau$ 很小，比如 $0.005$，表示目标网络缓慢追随主网络。

流程可以压缩成下面这张机制图：

| 步骤 | 作用 |
|---|---|
| 环境采样 $(s,a,r,s')$ | 产生训练数据 |
| 存入 replay buffer | 打散时间相关性 |
| 采样 batch | 构造训练小批次 |
| 更新 Critic | 拟合 TD 目标 $y$ |
| 更新 Actor | 沿 $\nabla_a Q$ 改进策略 |
| 软更新目标网络 | 稳定 bootstrap 目标 |

---

## 代码实现

实现 DDPG 时，训练顺序比网络花样更重要：**先更新 Critic，再更新 Actor，最后软更新目标网络。**

下面给一个可运行的最小 Python 版本。它不是完整深度学习实现，但把 DDPG 的梯度方向和更新顺序保留了下来。

```python
import random
from collections import deque

class ReplayBuffer:
    def __init__(self, capacity=1000):
        self.buffer = deque(maxlen=capacity)

    def store_transition(self, s, a, r, s_next):
        self.buffer.append((s, a, r, s_next))

    def sample(self, batch_size):
        batch = random.sample(self.buffer, batch_size)
        s, a, r, s_next = zip(*batch)
        return list(s), list(a), list(r), list(s_next)

    def __len__(self):
        return len(self.buffer)

class ToyDDPG:
    def __init__(self, actor_param=0.0, actor_lr=0.1, tau=0.1):
        self.actor_param = actor_param
        self.target_actor_param = actor_param
        self.actor_lr = actor_lr
        self.tau = tau
        self.buffer = ReplayBuffer()

    def select_action(self, s):
        # toy actor: a = theta * s
        return self.actor_param * s

    def q_value(self, s, a):
        # toy critic: best action is 2 * s
        return - (a - 2 * s) ** 2 + 1.0

    def dq_da(self, s, a):
        return -2 * (a - 2 * s)

    def update(self, batch_size=4):
        s_batch, a_batch, _, _ = self.buffer.sample(batch_size)
        grad = 0.0
        for s, _ in zip(s_batch, a_batch):
            a = self.select_action(s)
            grad += self.dq_da(s, a) * s   # dQ/da * da/dtheta
        grad /= batch_size
        self.actor_param += self.actor_lr * grad
        self.target_actor_param = self.tau * self.actor_param + (1 - self.tau) * self.target_actor_param

agent = ToyDDPG(actor_param=0.0)
for s in [0.5, 1.0, 1.5, 2.0, 0.8, 1.2]:
    a = agent.select_action(s)
    agent.buffer.store_transition(s, a, 0.0, s)

before = agent.actor_param
agent.update(batch_size=4)
after = agent.actor_param

assert len(agent.buffer) >= 4
assert after > before, "actor should move toward larger action in this toy setup"
assert agent.target_actor_param != before
```

这段代码体现了三个关键点：
1. `select_action` 负责输出确定性动作。
2. `store_transition` 负责把历史样本放进经验池。
3. `update` 里先依据 Critic 的方向更新 Actor，再软更新目标参数。

如果换成 PyTorch，主循环通常是：

| 顺序 | 操作 |
|---|---|
| 1 | 初始化 `Actor/Critic/target/replay buffer` |
| 2 | 与环境交互，动作上加探索噪声 |
| 3 | 存储 `(s,a,r,s',done)` |
| 4 | 从 buffer 随机采样 batch |
| 5 | 计算 $y=r+\gamma Q_{\phi'}(s',\mu_{\theta'}(s'))$ |
| 6 | 最小化 Critic 损失 |
| 7 | 最大化 Actor 对应的 $Q$ |
| 8 | 软更新目标网络 |

真实工程例子：机械臂抓取任务里，Actor 常输出每个关节的目标速度，先经过 `tanh` 压到 $[-1,1]$，再线性映射到环境动作范围。这样做的原因不是“好看”，而是保证训练和环境约束一致。

---

## 工程权衡与常见坑

DDPG 能工作，但它并不以稳定著称。很多失败不是“训练轮数不够”，而是机制本身容易出问题。

最常见的问题是 Critic 过估计。Critic 一旦把某些动作价值估得过高，Actor 会被错误梯度牵着走，结果就是 Q 值越吹越高，真实回报却不上升。

一个真实故障例子：连续控制任务中，日志里 `Q mean` 持续飙升，但 episode return 大幅震荡。排查后常见原因有四类：奖励尺度过大、Actor 学习率太高、探索噪声过小、目标网络更新太快。

| 坑点 | 后果 | 规避方式 |
|---|---|---|
| 动作边界处理不一致 | 训练动作和环境动作分布错位 | `tanh` 后再映射到环境范围 |
| replay buffer 太小或未预热 | 样本相关性强，训练抖动 | 先 warm-up 再开始更新 |
| 探索噪声不足 | 早期陷入局部最优 | 加高斯噪声或 OU 噪声 |
| Critic 学得过快 | Q 过估计，误导 Actor | 降低学习率，监控 Q 分布 |
| target update 太快 | 目标漂移，训练不稳 | 用小 $\tau$，如 0.005 |
| 奖励尺度失衡 | 梯度爆炸或价值偏移 | 奖励裁剪或归一化 |

三条工程建议最值得直接执行：
1. 动作输出先过 `tanh`，再映射到环境动作区间。
2. replay buffer 至少积累一批随机数据后再开始训练。
3. 先把 Critic 训练稳定，再考虑提高 Actor 更新强度。

还要明确一个结论：**DDPG 通常不如 TD3 和 SAC 稳定。** 如果你已经出现长期震荡、Q 值虚高、对超参数极度敏感，多训几轮通常没用，优先考虑换算法。

---

## 替代方案与适用边界

今天看连续控制，DDPG 更像“基础原型”，不是默认最优解。它重要，因为后来的 TD3 和 SAC 都是在它的框架上继续修正。

TD3（Twin Delayed DDPG）可以理解为 DDPG 的修正版。它用双 Critic 抑制过估计，还延迟 Actor 更新，稳定性通常更好。

SAC（Soft Actor-Critic）则进一步采用随机策略，并把熵正则引入目标，白话说就是“既追求高回报，也鼓励保留一定探索”。这使它在很多任务上更稳。

用同一个机械臂控制任务比较：

| 算法 | 策略类型 | 稳定性 | 是否有随机策略 | 适用场景 |
|---|---|---|---|---|
| DDPG | 确定性 | 中等偏弱 | 否 | 结构简单、教学、基线 |
| TD3 | 确定性 | 较强 | 否 | 想保留 DDPG 框架但更稳 |
| SAC | 随机性策略 | 强 | 是 | 复杂任务、稳定性优先 |

选择标准可以压缩成两句：
连续动作且想快速理解 Actor-Critic 主线，可以用 DDPG。
训练稳定性要求高、希望更少调参时，优先 TD3 或 SAC。

为什么说 DDPG 是 TD3 和 SAC 的前身？因为它先把几件核心事拼到一起了：确定性策略梯度、深度函数逼近、经验回放、目标网络。后续算法大多不是推翻它，而是在它暴露的问题上继续修补。

---

## 参考资料

| 类型 | 来源 | 支撑内容 |
|---|---|---|
| 确定性策略梯度原始论文 | Silver et al. 2014 | 支撑 $\nabla_\theta J$ 的确定性策略梯度推导 |
| DDPG 原始论文 | Lillicrap et al. 2015 | 支撑 Actor-Critic、目标网络、经验回放的完整框架 |
| 教程文档 | OpenAI Spinning Up | 支撑训练流程和实现细节说明 |
| 工程实现文档 | Stable-Baselines3 | 支撑常见实现限制与工程注意点 |
| 后续改进论文 | TD3 / SAC | 支撑替代方案与适用边界比较 |

1. [Deterministic Policy Gradient Algorithms](https://proceedings.mlr.press/v32/silver14.html) 用于支持确定性策略梯度公式和链式求导主线。  
2. [Continuous Control with Deep Reinforcement Learning](https://arxiv.org/abs/1509.02971) 用于支持 DDPG 的原始算法结构、目标网络和经验回放。  
3. [OpenAI Spinning Up: DDPG](https://spinningup.openai.com/en/latest/algorithms/ddpg.html) 用于核对训练流程、关键公式和实现步骤。  
4. [Stable-Baselines3 Documentation](https://stable-baselines3.readthedocs.io/) 用于参考工程实现习惯、超参数和常见坑。  
5. [Addressing Function Approximation Error in Actor-Critic Methods (TD3)](https://arxiv.org/abs/1802.09477) 用于说明 DDPG 为什么容易过估计，以及 TD3 如何修正。  
6. [Soft Actor-Critic: Off-Policy Maximum Entropy Deep Reinforcement Learning with a Stochastic Actor](https://arxiv.org/abs/1801.01290) 用于说明 SAC 与 DDPG 在策略形式和稳定性上的差异。
