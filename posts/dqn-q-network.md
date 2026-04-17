## 核心结论

DQN，Deep Q-Network，直译是“深度 Q 网络”，本质是把 Q-learning 里原本用表格存储的 $Q(s,a)$，改成用神经网络 $Q(s,a;\theta)$ 来近似。白话说，原来是“查表决定动作”，现在变成“让网络直接估计每个动作值多少钱”。

它解决的不是“强化学习的一切问题”，而是一个更具体的问题：当状态空间很大、无法把每个状态都列成表时，如何继续做值函数学习。比如 Atari 游戏画面是像素输入，状态维度极高，表格法根本存不下，也无法泛化；DQN 让网络从相似状态中共享参数，于是“看过一些局面”后，可以对没见过但相近的局面做估计。

DQN 能跑起来，靠的不是“神经网络替代表格”这一句，而是两个稳定器：

| 机制 | 作用 | 不加会怎样 |
|---|---|---|
| 经验回放 `Replay Buffer` | 打散样本相关性，重复利用历史经验 | 连续帧太像，梯度震荡大 |
| 目标网络 `Target Network` | 延迟目标值，减少自举更新的自我放大 | 目标跟着参数一起抖，容易发散 |

新手视角可以这样理解：把游戏画面喂给网络，网络输出“向左”“向右”“开火”“跳跃”等动作的 Q 值；动作选择用 $\varepsilon$-greedy，直译是“以 $\varepsilon$ 的概率随机探索，否则选当前最优动作”；每走一步，就把 $(s,a,r,s')$ 放进回放池，后面随机抽一批历史样本训练，而不是只盯着最近几帧。

核心目标公式是：

$$
Q(s,a;\theta) \approx Q^*(s,a)
$$

训练时常用的目标与损失是：

$$
y_t = r + \gamma \max_{a'} Q(s',a';\theta^-)
$$

$$
L(\theta)=\mathbb{E}\left[(y_t - Q(s,a;\theta))^2\right]
$$

其中 $\theta$ 是在线网络参数，$\theta^-$ 是目标网络参数。白话说，在线网络负责“当前预测”，目标网络负责“相对稳定的参考答案”。

---

## 问题定义与边界

先定义问题。强化学习里，Agent 是“做决策的程序”，Environment 是“它交互的外部世界”。在某个状态 $s$ 下执行动作 $a$ 后，环境返回奖励 $r$ 和下一个状态 $s'$。DQN 要学的是：在每个状态下，哪个动作未来累计奖励更高。

Q 值的定义是：

$$
Q(s,a)=\mathbb{E}\left[\sum_{k=0}^{\infty}\gamma^k r_{t+k}\mid s_t=s,a_t=a\right]
$$

白话说，$Q(s,a)$ 是“现在选这个动作，之后继续尽量做好，最终大概能拿多少总分”。

一个玩具例子：你面前有一副卡片牌组，当前状态是“剩余牌数、当前分数、是否还能翻牌”；动作只有两个，“翻牌”和“不翻”。如果翻牌常常能加分但也可能爆掉，DQN 的任务就是学习在什么状态下继续翻牌更划算。

这个问题一般建模成 MDP，Markov Decision Process，直译是“马尔可夫决策过程”，意思是“当前状态已经包含做决策所需的主要信息，未来只和现在有关，不直接依赖更早历史”。

DQN 的适用边界要讲清楚：

| 维度 | DQN 适合 | DQN 不擅长 |
|---|---|---|
| 动作空间 | 离散动作，如左/右、跳/不跳 | 连续动作，如连续扭矩、连续转向角 |
| 可观测性 | 当前状态基本可表达决策信息 | 严重部分可观测，需要长时记忆 |
| 奖励结构 | 中短期回报相对明确 | 极端稀疏奖励、超长信用分配 |
| 环境稳定性 | 平稳环境 | 动态规则频繁变化 |

再看一个真实工程例子。假设仓库机器人只能做有限动作：前进、左转、右转、停止。状态来自激光雷达离散特征、当前位置编码、目标方向。这里动作是离散的，状态虽然高维但可向量化，因此 DQN 是合理基线。如果换成机械臂扭矩控制，每个关节输出是连续实数，原始 DQN 就不合适，通常改用 DDPG 或 SAC 这类连续动作算法。

一个简单的数据流可以写成：

| 当前状态 $s$ | 动作 $a$ | 奖励 $r$ | 下一状态 $s'$ | 训练目标 $y$ |
|---|---|---|---|---|
| 当前观测 | 选中的离散动作 | 环境即时反馈 | 执行动作后的新观测 | $r+\gamma \max_{a'}Q(s',a';\theta^-)$ |

---

## 核心机制与推导

DQN 的核心来自 Bellman 最优方程。它表达的是：当前动作值，等于当前奖励，加上下一个状态最优动作值的折扣和。

$$
Q^*(s,a)=r+\gamma \max_{a'}Q^*(s',a')
$$

因为真实的 $Q^*$ 不知道，所以我们用网络近似它。在线网络给出当前估计：

$$
Q(s,a;\theta)
$$

目标网络给出训练目标：

$$
y_t=r+\gamma \max_{a'}Q(s',a';\theta^-)
$$

如果当前状态已经终止，没有下一步，那么通常直接取：

$$
y_t=r
$$

于是 TD error，Temporal Difference error，直译是“时序差分误差”，就是“当前预测和一步目标之间的差”：

$$
\delta_t = y_t - Q(s,a;\theta)
$$

平方后得到损失：

$$
L(\theta)=\mathbb{E}\left[\delta_t^2\right]
=\mathbb{E}\left[(r+\gamma \max_{a'}Q(s',a';\theta^-)-Q(s,a;\theta))^2\right]
$$

这就是 DQN 最核心的一行。

看一个数值玩具例子。设 $\gamma=0.9$，目标网络在下一状态 $s'$ 上输出两个动作值 `[2.0, 1.5]`，即时奖励 $r=1$。那么目标值为：

$$
y = 1 + 0.9 \times \max(2.0, 1.5) = 2.8
$$

如果当前在线网络给出的 $Q(s,a)=1.8$，那么：

$$
\delta = 2.8 - 1.8 = 1.0
$$

$$
L = (2.8-1.8)^2 = 1.0
$$

这意味着网络低估了这个动作，梯度会推动参数更新，让这个动作值往上走。

把它放进 mini-batch，训练就变成“批量纠偏”：

| 样本 | 预测 $Q(s,a;\theta)$ | 目标 $y$ | 误差 $y-Q$ | 平方误差 |
|---|---:|---:|---:|---:|
| 1 | 1.8 | 2.8 | 1.0 | 1.00 |
| 2 | 0.5 | 0.2 | -0.3 | 0.09 |
| 3 | 3.1 | 2.9 | -0.2 | 0.04 |

经验回放为什么必要？因为连续采样的轨迹高度相关。比如 Atari 相邻两帧画面几乎一样，如果你按时间顺序直接喂给网络，相当于不停拿“几乎重复”的样本更新，梯度方向会过度偏向某一小段局部经验。Replay Buffer 就是一个循环队列，存很多过去样本，再随机抽样，让训练数据更接近独立同分布。

目标网络为什么必要？因为 DQN 是自举学习，目标值里也有网络自己的输出。如果你直接用同一个网络同时生成预测和目标，参数一更新，参考答案也跟着动，优化目标会不停漂移，容易出现估值爆炸。目标网络本质上是“延迟拷贝的旧模型”，让你追一个相对稳定的目标。

目标网络有两种常见更新方式：

| 更新方式 | 公式 | 特点 |
|---|---|---|
| 硬更新 | 每隔 $N$ 步令 $\theta^- \leftarrow \theta$ | 简单，经典 DQN 常用 |
| 软更新 | $\theta^- \leftarrow \tau \theta + (1-\tau)\theta^-$ | 更平滑，现代实现常见 |

$\varepsilon$-greedy 的探索与利用也要说明白：

| 情况 | 行为 |
|---|---|
| 以概率 $\varepsilon$ | 随机选动作，探索没试过的路径 |
| 以概率 $1-\varepsilon$ | 选当前 Q 值最大的动作，利用已有知识 |

如果 $\varepsilon$ 太大，训练后期还在乱试，收敛慢；如果太小，回放池里动作分布单一，很多动作根本没被认真探索过。

---

## 代码实现

下面给一个可运行的极简 Python 版本。它不依赖深度学习框架，只演示 DQN 的目标计算、回放抽样和 TD 更新逻辑，适合先把公式跑通。

```python
import random
from collections import deque

class ReplayBuffer:
    def __init__(self, capacity):
        self.buffer = deque(maxlen=capacity)

    def add(self, s, a, r, s_next, done):
        self.buffer.append((s, a, r, s_next, done))

    def sample(self, batch_size):
        return random.sample(self.buffer, batch_size)

    def __len__(self):
        return len(self.buffer)

def max_q(q_table, state):
    return max(q_table.get((state, 0), 0.0), q_table.get((state, 1), 0.0))

def dqn_target(q_target, r, s_next, done, gamma):
    if done:
        return r
    return r + gamma * max_q(q_target, s_next)

def train_step(q_online, q_target, batch, lr=0.1, gamma=0.9):
    losses = []
    for s, a, r, s_next, done in batch:
        pred = q_online.get((s, a), 0.0)
        target = dqn_target(q_target, r, s_next, done, gamma)
        td_error = target - pred
        q_online[(s, a)] = pred + lr * td_error
        losses.append(td_error ** 2)
    return sum(losses) / len(losses)

# 构造一个玩具经验池
buffer = ReplayBuffer(capacity=10)
buffer.add("s0", 0, 1.0, "s1", False)
buffer.add("s1", 1, 0.0, "s2", False)
buffer.add("s2", 0, 2.0, "terminal", True)

# 在线网络与目标网络，这里用字典模拟
q_online = {
    ("s0", 0): 1.8,
    ("s1", 1): 0.5,
}
q_target = {
    ("s1", 0): 2.0,
    ("s1", 1): 1.5,
    ("s2", 0): 0.2,
    ("s2", 1): 0.1,
}

# 验证题目里的数值例子
y = dqn_target(q_target, r=1.0, s_next="s1", done=False, gamma=0.9)
loss = (y - q_online[("s0", 0)]) ** 2
assert abs(y - 2.8) < 1e-8
assert abs(loss - 1.0) < 1e-8

batch = buffer.sample(2)
mean_loss = train_step(q_online, q_target, batch, lr=0.1, gamma=0.9)

assert len(buffer) == 3
assert mean_loss >= 0.0
assert q_online[("s0", 0)] >= 1.8 or q_online[("s1", 1)] != 0.5

print("toy DQN step ok")
```

如果换成真正的神经网络，实现骨架通常是下面这样：

```python
q_values = online_net(state)
action = epsilon_greedy(q_values, epsilon)
next_state, reward, done = env_step(action)
replay_buffer.add(state, action, reward, next_state, done)

batch = replay_buffer.sample(batch_size)

pred_q = online_net(batch.state).gather(1, batch.action)
with torch.no_grad():
    target_q = batch.reward + gamma * target_net(batch.next_state).max(dim=1).values * (1 - batch.done)

loss = ((target_q - pred_q.squeeze(1)) ** 2).mean()

optimizer.zero_grad()
loss.backward()
optimizer.step()
```

目标网络同步可以写成：

```python
# 硬更新
target_net.load_state_dict(online_net.state_dict())

# 软更新
for tp, op in zip(target_net.parameters(), online_net.parameters()):
    tp.data.copy_(tau * op.data + (1 - tau) * tp.data)
```

真实工程里的训练循环一般是：

1. 读取状态，送入在线网络。
2. 按 $\varepsilon$-greedy 选动作。
3. 执行动作，得到 $(s,a,r,s',done)$。
4. 写入回放池。
5. 回放池足够大后，随机采样一个 batch。
6. 用目标网络计算 $y$。
7. 反向传播更新在线网络。
8. 每隔若干步同步目标网络，或每步软更新。

一个真实工程例子是 CartPole。状态只有 4 个连续值，动作只有左和右。虽然它不需要卷积网络，但非常适合入门调通 DQN 全流程：经验回放、目标网络、epsilon 衰减、批量训练、评估回报，都能在小环境里看到效果。

---

## 工程权衡与常见坑

DQN 难点通常不在“代码能不能跑”，而在“训练会不会稳定”。很多新手第一次写 DQN，程序不报错，但回报不涨、loss 乱跳、Q 值爆炸，原因基本集中在几类地方。

| 风险/坑 | 表现 | 缓解措施 |
|---|---|---|
| 不用目标网络 | loss 爆炸，Q 值越来越大 | 加 target network，延迟同步 |
| 不用经验回放 | 训练对最近轨迹过拟合 | 随机采样历史 batch |
| $\varepsilon$ 太小 | 动作单一，探索不足 | 设起始较大并缓慢衰减 |
| 回放池太小 | 样本分布窄，容易遗忘 | 提高容量，保证多样性 |
| 奖励尺度失控 | 梯度过大，训练不稳 | reward clipping 或标准化 |
| 终止状态处理错 | 目标值虚高 | `done=True` 时去掉 bootstrap 项 |
| 更新目标网太频繁 | 稳定器失效 | 拉大同步间隔或用小 $\tau$ |

一个常见新手错误是“只用最近五帧更新”。这时样本几乎一样，比如角色连续向右移动的几帧画面，只是像素轻微平移。网络会在非常窄的局部数据上反复更新，loss 曲线往往剧烈上下跳。改成从回放池随机抽样后，batch 内会混入不同时间段、不同动作、不同奖励结构的经验，训练明显更稳。

另一个典型问题是探索不足。比如你在 Atari 射击游戏里把 $\varepsilon$ 设得太小，智能体很快只会“跳跃”，因为它一开始碰巧靠跳跃拿到一点分，回放池里大量都是跳跃动作，网络就不断强化这个局部偏好，几乎学不会“射击”。这不是模型结构错，而是数据分布已经偏了。

还有一个工程现实：loss 下降不一定等于策略变好。DQN 优化的是 TD 误差，不是直接优化最终得分。某些情况下 loss 很平稳，但策略卡在局部最优；也可能 loss 噪声很大，但平均回报逐步上升。所以监控时至少要同时看三条曲线：训练 loss、平均 episodic return、epsilon 或动作分布。

---

## 替代方案与适用边界

DQN 是值函数方法里的基础款，但它不是终点。常见替代方案主要是为了解决它的几个已知缺陷。

| 方案 | 解决的问题 | 适用边界 | 是否需要离散动作 |
|---|---|---|---|
| DQN | 大状态空间下的离散动作值学习 | 基础离散控制、游戏 | 是 |
| Double DQN | 降低 `max` 带来的高估偏差 | Q 值高估明显时 | 是 |
| Dueling DQN | 分离状态价值与动作优势 | 动作多、很多动作差不多时 | 是 |
| DDPG | 连续动作控制 | 机械臂、连续控制 | 否 |
| SAC | 连续动作且强调稳定与样本效率 | 复杂控制、机器人 | 否 |

Double DQN 的核心改动很小，但很重要。标准 DQN 用同一个目标网络同时“选最大动作”和“评估该动作值”，容易高估。Double DQN 把这两步拆开：在线网络负责选动作，目标网络负责估值。白话说，“谁最好”与“值多少”不再由同一个噪声源同时决定。

Dueling DQN 则把 $Q(s,a)$ 分成状态价值 $V(s)$ 和动作优势 $A(s,a)$。直观上，有些状态本身就很好或很差，和具体动作关系没那么大；把这部分单独建模，可以提高样本效率。在动作很多但差异细的时候更有价值。

真实工程里可以这样决策：
如果你做的是 Atari、网格世界、有限离散动作导航，先从 DQN 或 Double DQN 起步。
如果你发现 Q 值系统性偏高、策略不稳定，优先换 Double DQN。
如果动作空间很大、很多动作效果接近，可以试 Dueling DQN。
如果动作是连续实数，比如机器人扭矩、自动驾驶转角、无人机油门，直接跳到 DDPG、TD3、SAC 这类方法，不要硬把连续动作粗暴离散化后再用 DQN，除非动作维度很小且精度要求不高。

---

## 参考资料

1. PyTorch 官方教程《Reinforcement Learning (DQN) Tutorial》：代码实现最直接，包含 `ReplayMemory`、epsilon-greedy、目标网络软更新等完整训练流程。https://docs.pytorch.org/tutorials/intermediate/reinforcement_q_learning.html

2. Practical Reinforcement Learning for Robotics and AI《Deep Q Network (DQN)》：适合梳理 DQN 的任务定义、训练循环、常见陷阱与离散动作适用边界。https://www.reinforcementlearningpath.com/deep-q-network-dqn/

3. Practical Reinforcement Learning for Robotics and AI《Double DQN》：说明 DQN 的高估偏差来源，以及 Double DQN 如何把动作选择和动作评估拆开。https://www.reinforcementlearningpath.com/double-dqn/

4. Practical Reinforcement Learning for Robotics and AI《Dueling DQN》：说明状态价值和动作优势分离建模的动机，适合理解 Dueling 结构为什么常与 DQN 组合使用。https://www.reinforcementlearningpath.com/dueling-dqn/

5. WIRED 2015 报道《Google's AI Is Now Smart Enough to Play Atari Like the Pros》：从产业视角回顾 DQN 在 Atari 上的里程碑意义，适合先建立直观认识。https://www.wired.com/2015/02/google-ai-plays-atari-like-pros
