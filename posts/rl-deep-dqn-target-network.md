## 核心结论

DQN 的目标网络，是在线 Q 网络的一个“延迟副本”。这里的“延迟副本”可以先理解为：它和在线网络结构完全一样，但参数不会随着每一次梯度更新立刻变化，而是隔一段时间才同步一次，或者以很小的比例缓慢跟随。

它的核心作用只有一个：让 TD 目标在一段时间内相对稳定。DQN 的目标通常写成

$$
y_t = r_t + \gamma \max_{a'} Q(s_{t+1}, a'; \theta^-)
$$

其中 $\theta$ 是在线网络参数，$\theta^-$ 是目标网络参数。关键点在于，训练时被优化的是 $Q(s_t,a_t;\theta)$，而目标值 $y_t$ 用的是另一套较慢变化的参数 $\theta^-$。这样做的结果是：预测值在变，目标值不会跟着每一步一起乱动，训练不容易振荡或发散。

目标网络常见有两种更新方式：

| 方式 | 公式 | 更新节奏 | 特点 |
|---|---|---|---|
| 硬更新 | $\theta^- \leftarrow \theta$ | 每 $C$ 步复制一次 | 实现简单，目标分段恒定 |
| 软更新 | $\theta^- \leftarrow \tau\theta + (1-\tau)\theta^-$ | 每步小幅更新 | 更平滑，但更依赖超参 |

必须区分一个常见误解：目标网络主要解决的是“目标漂移”，不是直接解决“Q 值过估计”。过估计更直接的对应方案是 Double DQN。

---

## 问题定义与边界

先定义问题。Q 学习要学的是动作价值函数，也就是“在状态 $s$ 下做动作 $a$，未来累计回报大概有多大”。DQN 用神经网络去逼近这个函数，因此训练目标是让网络输出接近 TD 目标。

如果没有目标网络，DQN 会把目标写成：

$$
y_t = r_t + \gamma \max_{a'} Q(s_{t+1}, a'; \theta)
$$

同时预测项也是：

$$
Q(s_t, a_t; \theta)
$$

这意味着同一个参数 $\theta$ 既决定当前预测，又决定训练目标。问题在于，你刚用一次梯度下降更新了 $\theta$，目标就变了。于是优化器面对的是一个不断移动的回归目标，本质上是在做非平稳优化。对线性模型这已经不舒服，对深度网络则更容易出现震荡、发散、值函数爆炸。

玩具例子可以先看一维情况。

假设某一步中：

- 即时奖励 $r=1$
- 折扣因子 $\gamma=0.99$
- 下一状态最大 Q 值估计为 $10$

那么目标值是

$$
y = 1 + 0.99 \times 10 = 10.9
$$

如果你这一步更新后，网络把下一状态的最大 Q 估计从 $10$ 改成了 $11.5$，那么下一次训练目标立刻变成

$$
y' = 1 + 0.99 \times 11.5 = 12.385
$$

此时网络一边追 10.9，一边又把目标抬成 12.385，等于自己追自己。目标网络的作用，就是让这 10.9 在若干步内先固定住。

它的适用边界也要说清楚：

| 现象 | 目标网络是否直接解决 | 说明 |
|---|---|---|
| TD 目标随训练同步漂移 | 是 | 这是目标网络的主要用途 |
| 样本强相关导致训练不稳 | 否，需要经验回放配合 | 经验回放负责打乱相关性 |
| Q 值系统性过高估计 | 否，不是直接解法 | 更常用 Double DQN |
| 奖励尺度过大导致梯度爆炸 | 否 | 需要 reward clipping、梯度裁剪等 |

所以目标网络不是稳定训练的全部，只是 DQN 稳定性的关键拼图之一。

---

## 核心机制与推导

DQN 的标准损失函数可以写成：

$$
L(\theta)=\mathbb{E}\left[\left(r_t+\gamma \max_{a'}Q(s_{t+1},a';\theta^-)-Q(s_t,a_t;\theta)\right)^2\right]
$$

这里可以拆成两部分理解：

- 目标项：$r_t+\gamma \max_{a'}Q(s_{t+1},a';\theta^-)$
- 预测项：$Q(s_t,a_t;\theta)$

优化时，梯度只对 $\theta$ 求导，不对 $\theta^-$ 求导。白话说，就是“训练时只改学生答案，不改标准答案”。虽然这个“标准答案”不是真正固定不变，但它变化得足够慢，因此优化过程更接近普通监督学习。

如果不用目标网络，梯度下降实际上在处理这种结构：

$$
L(\theta)=\mathbb{E}\left[\left(r_t+\gamma \max_{a'}Q(s_{t+1},a';\theta)-Q(s_t,a_t;\theta)\right)^2\right]
$$

这时同一组参数同时出现在目标和预测两边。你每更新一次，等式两边都在动，误差面本身也在动，训练稳定性会明显变差。

硬更新的机制最容易理解：

$$
\theta^- \leftarrow \theta \quad \text{every } C \text{ steps}
$$

也就是每隔 $C$ 步，把在线网络参数整份复制给目标网络。在这 $C$ 步里，$\theta^-$ 完全不变，因此目标值近似固定。很多 Atari 场景会把 $C$ 设在 $10^3 \sim 10^4$ 的量级，本质上是在控制“靶子多久换一次”。

软更新则更连续：

$$
\theta^- \leftarrow \tau \theta + (1-\tau)\theta^-
$$

其中 $\tau$ 很小，比如 $0.001$。这表示每一步只把在线网络的 0.1% 混入目标网络，其余 99.9% 保留旧参数。它不是一次整份复制，而是指数滑动平均。若展开两步，可以得到：

$$
\theta^-_{t+1}=\tau\theta_t+(1-\tau)\theta^-_t
$$

继续展开会发现，当前目标网络参数是历史在线网络参数的加权平均，越新的权重占比越小但更接近现在。因此软更新可以理解成“带惯性的慢追踪”。

真实工程例子是 Atari。高维视觉输入下，网络每次看到的状态分布、奖励分布、策略分布都在变，如果目标也同步快速变化，训练很容易失稳。经验回放负责把样本打散，目标网络负责把 TD 靶值变慢，两者配合后，DQN 才能在这些任务上稳定学习。这个组合不是巧合，而是职责分离：一个处理样本相关性，一个处理目标非平稳性。

---

## 代码实现

下面给一个可以直接运行的 Python 玩具实现，只演示目标网络更新逻辑，不依赖深度学习框架。这里把“网络参数”简化成字典中的几个数字，方便看清硬更新和软更新在做什么。

```python
from copy import deepcopy

class TinyDQN:
    def __init__(self):
        self.online = {"w1": 1.0, "w2": 2.0}
        self.target = deepcopy(self.online)

    def hard_update(self):
        self.target = deepcopy(self.online)

    def soft_update(self, tau: float):
        for k in self.online:
            self.target[k] = tau * self.online[k] + (1 - tau) * self.target[k]

agent = TinyDQN()

# 在线网络先发生变化
agent.online["w1"] = 5.0
agent.online["w2"] = 10.0

# 硬更新前，目标网络不变
assert agent.target["w1"] == 1.0
assert agent.target["w2"] == 2.0

agent.hard_update()
assert agent.target["w1"] == 5.0
assert agent.target["w2"] == 10.0

# 重新构造一个例子测试软更新
agent = TinyDQN()
agent.online["w1"] = 5.0
agent.online["w2"] = 10.0

agent.soft_update(tau=0.1)

# 新目标 = 0.1 * online + 0.9 * old_target
assert abs(agent.target["w1"] - (0.1 * 5.0 + 0.9 * 1.0)) < 1e-9
assert abs(agent.target["w2"] - (0.1 * 10.0 + 0.9 * 2.0)) < 1e-9

print("all tests passed")
```

如果换成 PyTorch，工程结构通常是“两个同构网络”：

```python
import copy
import torch
import torch.nn as nn

class QNet(nn.Module):
    def __init__(self, state_dim, action_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, 128),
            nn.ReLU(),
            nn.Linear(128, action_dim),
        )

    def forward(self, x):
        return self.net(x)

online_net = QNet(state_dim=4, action_dim=2)
target_net = copy.deepcopy(online_net)
target_net.eval()

def hard_update():
    target_net.load_state_dict(online_net.state_dict())

@torch.no_grad()
def soft_update(tau=0.005):
    for tp, op in zip(target_net.parameters(), online_net.parameters()):
        tp.data.copy_(tau * op.data + (1 - tau) * tp.data)
```

训练时的关键流程是：

1. 用 `online_net` 计算当前状态动作的预测值。
2. 用 `target_net` 计算下一状态的 bootstrapped target。
3. 反向传播只更新 `online_net`。
4. 按设定频率执行 `hard_update()`，或者每步执行 `soft_update()`。

如果还要再严谨一点，构造目标值时通常会加 `torch.no_grad()`，防止梯度意外流入目标网络。

---

## 工程权衡与常见坑

目标网络不是“越慢越稳就越好”。它是在稳定性和适应性之间做平衡。

| 方案 | 优点 | 风险 | 常见场景 |
|---|---|---|---|
| 硬更新，$C$ 大 | 目标很稳 | 吸收新知识慢，可能学习滞后 | 经典 DQN、Atari |
| 硬更新，$C$ 小 | 响应快 | 靶值抖动大，接近无目标网络 | 小规模实验 |
| 软更新，$\tau$ 小 | 平滑、连续 | 可能过慢，旧信息保留太久 | DDPG、TD3、SAC |
| 软更新，$\tau$ 大 | 跟随更快 | 易把噪声带入目标网络 | 对稳定性要求较低的场景 |

常见坑主要有五类。

第一，把更新频率设得太激进。若硬更新的 $C$ 太小，或者软更新的 $\tau$ 太大，目标网络就会快速追上在线网络，等于重新回到“移动靶子”问题。

第二，以为有了目标网络就不会过估计。实际上 $\max_{a'} Q(s',a')$ 本身就容易挑中被高估的动作，因此目标网络只是让目标慢下来，不会改变 max 操作的统计偏差。这个问题更适合 Double DQN 处理。

第三，忘记停止目标网络梯度。实现上如果不小心让目标网络也参与反向传播，就破坏了“固定目标”的设计，训练行为会偏离预期。

第四，只调目标网络，不管经验回放。目标网络解决的是目标漂移，经验回放解决的是样本强相关。两者少一个，DQN 的稳定性通常都不够。

第五，忽略任务尺度差异。小型离散状态任务里，$C=100$ 可能已经够用；高维图像输入任务里，$C=100$ 往往太小。超参不能脱离环境复杂度单独谈。

一个真实工程上的判断标准是：如果你发现训练曲线短期内大幅震荡、Q 值不断飙高、loss 反复爆炸，可以优先检查目标网络更新是不是过于频繁，而不是先怀疑优化器。

---

## 替代方案与适用边界

目标网络本身不是终点，而是很多深度强化学习算法的基础稳定器。

最直接的替代增强方案是 Double DQN。它仍然保留目标网络，但修改了目标值的计算方式：

$$
y_t = r_{t+1} + \gamma Q_{\theta^-}\left(s_{t+1}, \arg\max_a Q_\theta(s_{t+1}, a)\right)
$$

这里在线网络负责“选动作”，目标网络负责“估价值”。白话说，就是把“谁来挑最大值”和“谁来给最大值打分”分开，以减少高估偏差。对于动作价值容易被系统性抬高的任务，Double DQN 往往比普通 DQN 更稳。

另一类是连续控制算法，比如 DDPG、TD3、SAC。这些方法通常大量使用软更新，而不是经典 DQN 那样的硬同步。原因不是“软更新一定更先进”，而是连续控制中策略和价值函数更新更频繁、耦合更紧，平滑追踪通常更合适。

可以这样理解不同方案的边界：

| 方法 | 是否使用目标网络 | 主要解决什么问题 | 更适合什么场景 |
|---|---|---|---|
| DQN + 目标网络 | 是 | 目标漂移 | 离散动作基础任务 |
| Double DQN | 是 | 目标漂移 + 部分过估计 | 离散动作且高估明显 |
| DDPG/TD3/SAC | 是，常用软更新 | 连续控制中的稳定追踪 | 连续动作空间 |
| 仅经验回放、无目标网络 | 否 | 只缓解样本相关性 | 通常不建议单独使用 |

因此，目标网络的适用边界可以概括为一句话：只要你的 bootstrapping 目标来自一个会被当前更新立刻改变的函数逼近器，就应该认真考虑“延迟目标”机制；但如果你要解决的是过估计、探索不足、奖励稀疏，那还需要别的结构配合。

---

## 参考资料

- Mnih, V. et al. 2015. Human-level control through deep reinforcement learning. *Nature*.
- Next Electronics, *Deep Q-Network Explained*: https://next.gr/ai/reinforcement-learning/deep-q-network-dqn-explained
- APXML, *Target Networks for Training Stability*: https://apxml.com/courses/advanced-reinforcement-learning/chapter-2-deep-q-networks/target-networks
- AI Stack Exchange, *How and when should we update the Q-target in deep Q-learning?*: https://ai.stackexchange.com/questions/21485/how-and-when-should-we-update-the-q-target-in-deep-q-learning
- DeepWiki, *Network Update Strategies*: https://deepwiki.com/mimoralea/gdrl/7.3-network-update-strategies
- DeepWiki, *Target Network and Soft Updates*: https://deepwiki.com/guojing0/2048-RL/3.3.3-target-network-and-soft-updates
