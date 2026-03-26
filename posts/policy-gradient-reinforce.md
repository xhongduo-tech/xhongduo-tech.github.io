## 核心结论

REINFORCE 是最基础的策略梯度方法。策略梯度的意思是：不先学“这个状态值多少钱”，而是直接调策略参数 $\theta$，让策略 $\pi_\theta(a|s)$ 更容易选到高回报动作。它优化的目标是整条轨迹的期望回报：

$$
J(\theta)=\mathbb{E}_{\tau \sim \pi_\theta}\left[\sum_{t=0}^{T} r_t\right]
$$

REINFORCE 的核心更新式是：

$$
\nabla J(\theta)=\mathbb{E}_{\pi_\theta}\left[\sum_{t=0}^{T}\nabla_\theta \log \pi_\theta(a_t|s_t)\, G_t\right]
$$

其中 $G_t$ 是从时刻 $t$ 开始往后的累计回报，通常叫 reward-to-go，白话说就是“这个动作之后实际拿到了多少分”。

这条式子的直观含义很直接：如果某次动作后面带来了高回报，就把这个动作在当时状态下的概率往上推；如果后面回报低，就往下推。REINFORCE 不需要环境模型。无模型的意思是：算法不必知道环境状态转移公式，只要能和环境交互采样就能学。

最重要的工程事实有两个。

| 结论 | 含义 | 工程影响 |
|---|---|---|
| 直接优化策略 | 不通过值迭代，直接对 $\theta$ 做梯度上升 | 适合离散或连续动作策略建模 |
| 用 Monte Carlo 回报 | Monte Carlo 指整条 episode 跑完后用真实采样回报估计梯度 | 简单，但方差大 |
| 加 baseline 降方差 | baseline 是一个与动作无关的参照值 | 不改变期望梯度方向，但训练更稳 |

一个最小玩具例子可以先把直觉建立起来。二选一 bandit 可以理解成“只有一步动作的老虎机”。设 $\pi(a_1)=0.5,\pi(a_2)=0.5$，若采样到 $a_1$ 且得到 $G=1$，那么梯度项会推动 $\pi(a_1)$ 增大。如果再设基线 $b=0.7$，更新量变成和 $G-b=0.3$ 成正比，方向仍然是增大 $\pi(a_1)$，但幅度更温和。这说明 baseline 的主要作用不是改目标，而是把更新信号“去中心化”，降低抖动。

---

## 问题定义与边界

REINFORCE 解决的问题是：给定一个可交互环境，直接学习参数化策略 $\pi_\theta(a|s)$，让它在长期回报意义上尽量好。这里的策略可以是一个 softmax 分类器，也可以是一个神经网络。参数化策略的意思是：动作概率不是手写规则，而是由参数 $\theta$ 控制。

更精确地说，它针对的是 episode 型任务。episode 可以理解成“一局完整过程”，例如一局迷宫、一盘游戏、一次机械臂抓取。每局从初始状态开始，到终止状态结束。REINFORCE 用整局采样到的真实回报来估计梯度，因此它天然依赖完整轨迹。

一个简单迷宫就是典型场景。智能体从入口出发，每一步只能看到当前位置，只有到达终点时才知道这局是否成功。如果中途没有明确指导信号，REINFORCE 仍然可以在 episode 结束后，用整条路径的累计回报去更新路径上各个动作的概率。

但它有明确边界。因为每次需要完整 episode，若任务特别长、奖励特别稀疏、观测不完整，REINFORCE 往往会很慢，而且很不稳定。

| 维度 | REINFORCE 的适用情况 | 边界 |
|---|---|---|
| 输入/输出 | 输入是状态，输出是动作分布 | 状态噪声很大时学习困难 |
| 动作空间 | 离散动作最直观，连续动作也可做高斯策略 | 高维连续动作方差更大 |
| 回报反馈延迟 | 能处理延迟奖励，因为看整条轨迹 | 延迟太长时信用分配困难 |
| 数据来源 | 只需采样，不需环境模型 | 样本效率通常不高 |
| 训练节奏 | 按 episode 更新 | 超长 episode 会造成更新滞后 |

这里要区分“能用”和“好用”。REINFORCE 在定义上能处理延迟奖励，但不代表它在长时序任务里效率高。比如 500 步以后才知道输赢的游戏，早期动作的贡献会被巨大的采样噪声淹没，这就是信用分配问题。信用分配的意思是：最终结果出来后，怎么把功劳或责任正确分摊给前面的每一步动作。

---

## 核心机制与推导

REINFORCE 的关键推导来自对轨迹概率求梯度。设一条轨迹为 $\tau=(s_0,a_0,\dots,s_T,a_T)$，其概率可写为：

$$
p_\theta(\tau)=p(s_0)\prod_{t=0}^{T}\pi_\theta(a_t|s_t)\,p(s_{t+1}|s_t,a_t)
$$

环境转移项 $p(s_{t+1}|s_t,a_t)$ 不依赖策略参数 $\theta$，所以对 $\theta$ 求导时，只剩策略项。

目标函数写成轨迹形式：

$$
J(\theta)=\sum_{\tau} p_\theta(\tau) R(\tau)
$$

其中 $R(\tau)$ 是整条轨迹总回报。于是：

$$
\nabla_\theta J(\theta)
=\sum_{\tau}\nabla_\theta p_\theta(\tau) R(\tau)
$$

利用恒等式

$$
\nabla_\theta p_\theta(\tau)=p_\theta(\tau)\nabla_\theta \log p_\theta(\tau)
$$

得到：

$$
\nabla_\theta J(\theta)
=\sum_{\tau} p_\theta(\tau)\nabla_\theta \log p_\theta(\tau) R(\tau)
=\mathbb{E}_{\tau\sim\pi_\theta}\left[\nabla_\theta \log p_\theta(\tau) R(\tau)\right]
$$

再展开轨迹对数概率：

$$
\log p_\theta(\tau)=\log p(s_0)+\sum_{t=0}^{T}\log \pi_\theta(a_t|s_t)+\sum_{t=0}^{T}\log p(s_{t+1}|s_t,a_t)
$$

与 $\theta$ 有关的只有策略项，因此：

$$
\nabla_\theta \log p_\theta(\tau)=\sum_{t=0}^{T}\nabla_\theta \log \pi_\theta(a_t|s_t)
$$

合并后得到：

$$
\nabla J(\theta)=\mathbb{E}_{\pi_\theta}\left[\sum_{t=0}^{T}\nabla_\theta \log \pi_\theta(a_t|s_t)\, R(\tau)\right]
$$

进一步把整条轨迹总回报替换成每个时刻之后的 reward-to-go：

$$
G_t=\sum_{k=t}^{T}\gamma^{k-t}r_k
$$

其中 $\gamma \in [0,1]$ 是折扣因子，白话说就是“越远的未来奖励折得越小”。于是常见写法变成：

$$
\nabla J(\theta)=\mathbb{E}_{\pi_\theta}\left[\sum_{t=0}^{T}\nabla_\theta \log \pi_\theta(a_t|s_t)\, G_t\right]
$$

为什么可以用 $G_t$ 替代整局总回报？因为在时刻 $t$ 之前发生的奖励与动作 $a_t$ 无关，把它们乘到该时刻梯度项上只会增加噪声，不提供有效学习信号。

再看 baseline。加入任意与动作无关的函数 $b(s_t)$ 后：

$$
\nabla J(\theta)=\mathbb{E}_{\pi_\theta}\left[\sum_{t=0}^{T}\nabla_\theta \log \pi_\theta(a_t|s_t)\, (G_t-b(s_t))\right]
$$

它仍然无偏，原因是：

$$
\mathbb{E}_{a_t\sim \pi_\theta(\cdot|s_t)}\left[\nabla_\theta \log \pi_\theta(a_t|s_t)\, b(s_t)\right]
= b(s_t)\sum_{a_t}\pi_\theta(a_t|s_t)\nabla_\theta \log \pi_\theta(a_t|s_t)
$$

而 $\pi \nabla \log \pi = \nabla \pi$，所以：

$$
b(s_t)\sum_{a_t}\nabla_\theta \pi_\theta(a_t|s_t)
= b(s_t)\nabla_\theta \sum_{a_t}\pi_\theta(a_t|s_t)
= b(s_t)\nabla_\theta 1
= 0
$$

这说明 baseline 不改变期望梯度，只改变方差。

看一个具体玩具例子。假设某个过关游戏在时刻 $t$ 做出动作后，当前即时奖励是 10，后续又得到 3 和 2，那么：

$$
G_t = 10 + 3 + 2 = 15
$$

如果状态基线估计 $b(s_t)=12$，那真正用于更新的是 $15-12=3$。直觉上，这一步并不是“值 15 分的神奇动作”，而是“比这个状态下的平均发挥好 3 分”。这就比直接用 15 做更新更稳，因为模型不会把偶然偏高的整局结果全部归功到当前动作上。

---

## 代码实现

REINFORCE 的标准训练 loop 很固定：

1. 用当前策略采样一批完整 episode。
2. 对每条 episode 反向计算每个时间步的 $G_t$。
3. 累加损失 $-\log \pi_\theta(a_t|s_t)\cdot (G_t-b_t)$。
4. 反向传播，执行 `optimizer.step()`。

先给一个最小可运行的 Python 版本，用 bandit 演示“高回报动作概率会上升，加入 baseline 后方向不变但步子更小”。

```python
import math

def sigmoid(x: float) -> float:
    return 1.0 / (1.0 + math.exp(-x))

def reinforce_bandit_update(theta: float, action: int, reward: float, baseline: float = 0.0, lr: float = 0.1) -> float:
    """
    action: 1 表示选择 a1, 0 表示选择 a2
    策略: pi(a1) = sigmoid(theta), pi(a2) = 1 - sigmoid(theta)
    REINFORCE 单步更新: theta += lr * d/dtheta log pi(a|s) * (reward - baseline)
    """
    p = sigmoid(theta)
    grad_log_prob = (1.0 - p) if action == 1 else (-p)
    advantage = reward - baseline
    return theta + lr * grad_log_prob * advantage

theta0 = 0.0  # pi(a1)=0.5
p0 = sigmoid(theta0)
assert abs(p0 - 0.5) < 1e-8

# 采样到更优动作 a1，奖励为 1
theta1 = reinforce_bandit_update(theta0, action=1, reward=1.0, baseline=0.0, lr=1.0)
p1 = sigmoid(theta1)
assert p1 > p0  # 概率上升

# 加 baseline 后方向相同，但更新幅度更小
theta2 = reinforce_bandit_update(theta0, action=1, reward=1.0, baseline=0.7, lr=1.0)
p2 = sigmoid(theta2)
assert p2 > p0
assert p2 < p1  # 方向不变，幅度变小

print(round(p0, 4), round(p1, 4), round(p2, 4))
```

上面的代码只是一维 bandit，目的是让公式和参数更新一一对应。真实环境会有多步轨迹，因此要显式计算 reward-to-go。下面给出更接近工程代码的伪实现：

```python
import torch

def compute_returns(rewards, gamma):
    returns = []
    G = 0.0
    for r in reversed(rewards):
        G = r + gamma * G
        returns.append(G)
    returns.reverse()
    return torch.tensor(returns, dtype=torch.float32)

def train_one_batch(policy, optimizer, episodes, gamma=0.99, baseline_fn=None):
    optimizer.zero_grad()
    total_loss = 0.0

    for episode in episodes:
        states, actions, rewards = episode["states"], episode["actions"], episode["rewards"]
        returns = compute_returns(rewards, gamma)

        for s, a, G in zip(states, actions, returns):
            dist = policy(s)                 # 输出动作分布
            log_prob = dist.log_prob(a)      # log pi(a|s)

            if baseline_fn is None:
                baseline = 0.0
            else:
                baseline = baseline_fn(s).detach()

            advantage = G - baseline
            total_loss = total_loss - log_prob * advantage

    total_loss.backward()
    optimizer.step()
    return float(total_loss)
```

这里有三个实现细节最重要。

第一，`returns` 最好用 reward-to-go，而不是整条 episode 总回报复制到每个时刻。两者都无偏，但 reward-to-go 通常方差更低。

第二，baseline 常见做法是状态值函数 $V(s)$。状态值函数可以理解成“站在这个状态，未来大概要拿多少回报”。此时 $G_t - V(s_t)$ 就接近优势函数。优势函数的意思是：当前动作比这个状态下的平均表现好多少。

第三，`baseline_fn(s).detach()` 很关键。如果你用单独的值网络学习 baseline，策略损失和价值损失通常分开写，避免梯度路径混乱。

真实工程例子可以看高维机器人控制。比如机械手要控制大量连续动作维度去逼近一个目标姿态，动作空间可能非常高维。纯 REINFORCE 在这种任务上会因为采样噪声过大而极不稳定，因此会配合更强的方差缩减方法，例如状态 baseline、动作分解后的 baseline，甚至结构化估计器。这类场景说明：REINFORCE 的公式虽然简洁，但一旦进入高维连续控制，方差控制就是成败关键。

---

## 工程权衡与常见坑

REINFORCE 的最大优点是概念和实现都干净，最大缺点也是明确的：高方差。方差可以理解成“同样的策略参数，在不同采样批次上，估计出的梯度方向和大小波动很大”。波动一大，训练就会抖，甚至完全不收敛。

常见问题和缓解方法可以直接列成表：

| 问题 | 缓解手段 | 影响 |
|---|---|---|
| 梯度高方差 | reward-to-go + baseline + 多 episode 平均 | 更新更平滑 |
| 回报尺度乱 | 回报标准化、优势标准化 | 学习率更容易调 |
| episode 太长 | 换 Actor-Critic 或截断更新 | 降低更新延迟 |
| 稀疏奖励 | 奖励塑形、课程学习、探索增强 | 更容易学到早期策略 |
| baseline 设计错误 | 保证 baseline 与动作独立 | 避免引入偏差 |
| 单条轨迹更新 | 批量采样再更新 | 降低估计噪声 |

最容易踩的坑有四个。

第一，把整条总回报硬塞给每个时间步。这样虽然公式上可以成立，但会把与当前动作无关的早期或后期奖励也记在该动作头上，噪声明显更大。reward-to-go 几乎总是更合理。

第二，把 baseline 写成和动作相关的量，却又按“无偏 baseline”来理解。无偏结论成立的前提是 baseline 对动作不显式依赖。若依赖动作，推导要重新做，不能直接套状态 baseline 的结论。高维机器人控制里有 action-dependent baseline 的研究，但那是专门设计过的方差缩减方法，不是随便加一个 $b(s,a)$ 就行。

第三，episode 太长还坚持纯 Monte Carlo。比如一个 500 步游戏，每局结束才更新一次，训练信号既晚又吵。你会看到 loss 看起来在变，但策略质量提升很慢。此时问题不一定是代码错，更可能是方法边界到了。

第四，把值网络 baseline 当成“越准越好”，却忽略训练稳定性。baseline 理论上不改变期望梯度，但如果值网络训练严重漂移，实际优化过程仍会变差，因为 advantage 分布会乱，导致策略更新尺度异常。

一个简单经验是：如果你发现 REINFORCE 必须靠非常小的学习率、非常大的 batch 才能不炸，通常说明应该考虑 Actor-Critic 或 PPO，而不是继续在纯 Monte Carlo 上硬调参。

---

## 替代方案与适用边界

REINFORCE 适合作为策略梯度入门方法和小规模基线方法，但在中大型工程里，常见替代方案通常更实用。

| 方法 | 样本效率 | 稳定性 | 实现复杂度 | 适用场景 |
|---|---|---|---|---|
| REINFORCE | 低 | 一般到较差 | 低 | 教学、简单任务、原型验证 |
| Actor-Critic | 中 | 中到较好 | 中 | 中长时序任务、需要更频繁更新 |
| PPO | 较高 | 较好 | 中到较高 | 工程主流、需要稳健训练 |
| TRPO | 中 | 较好 | 高 | 强约束更新、研究型使用 |

Actor-Critic 的核心改进是：不再完全依赖整条 episode 的 Monte Carlo 回报，而是用 critic，也就是值函数估计器，去近似未来回报，再构造优势函数更新 actor。这样会引入一些偏差，但通常能显著降低方差，整体更好训。

PPO 则进一步在更新规则上加约束。它不是简单做一步梯度上升，而是限制新旧策略差异别太大，避免一次更新把策略推坏。对于真实工程，这种“每步别走太猛”的机制很重要。

看一个具体对比。假设你在训练一个 500 步游戏角色。REINFORCE 要等整局结束才能算完整回报，然后更新一次。如果中途第 16 步已经明显暴露出好坏趋势，REINFORCE 也只能等。而 PPO 常见做法是每收集一小段 rollout 就开始更新，不必一直等到 episode 完结。这能明显降低训练延迟，也能提高数据利用率。

所以边界可以概括成一句话：

- 如果目标是理解“策略梯度到底在优化什么”，用 REINFORCE。
- 如果目标是把一个稍微复杂的任务训起来，优先 Actor-Critic 或 PPO。
- 如果动作空间高维、episode 很长、奖励稀疏，纯 REINFORCE 通常不是首选。

---

## 参考资料

- Williams, R. J. “Simple Statistical Gradient-Following Algorithms for Connectionist Reinforcement Learning.” 用于理解 REINFORCE 原始提出方式与经典公式推导。https://link.springer.com/article/10.1007/BF00992696
- Sutton, R. S., and Barto, A. G. *Reinforcement Learning: An Introduction* (Second Edition). 用于系统理解 policy gradient、baseline、actor-critic 的教材。http://incompleteideas.net/book/the-book-2nd.html
- OpenAI. “Variance reduction for policy gradient with action-dependent factorized baselines.” 用于理解高维动作空间中的高级方差缩减设计。https://openai.com/index/variance-reduction-for-policy-gradient-with-action-dependent-factorized-baselines/
