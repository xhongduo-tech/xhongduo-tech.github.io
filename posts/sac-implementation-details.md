## 核心结论

SAC（Soft Actor-Critic，软演员-评论家）可以先用一句白话概括：它是在尽量拿高分的同时，强迫策略别太死板。这里的“策略”是指模型在给定状态下如何选动作的规则；“别太死板”指策略要保留一定随机性，而不是一开始就把动作钉死。

SAC 的工程价值主要来自三件事：

1. 它是离策略算法。离策略的意思是，训练时可以反复使用历史经验，不必每更新一次参数就重新采一大批新数据，所以样本效率通常高于 PPO 这类在策略方法。
2. 它显式优化最大熵目标。熵可以粗略理解为“动作分布的分散程度”，熵高表示动作更有探索性，熵低表示动作更确定。
3. 它用双 Q 网络取最小值来构造目标，抑制 Q 值过高估计。Q 值是“状态-动作价值”，表示在状态 $s$ 下执行动作 $a$ 后，未来累计回报的估计。

和 DDPG、TD3 对比，SAC 的本质差异不是“多了点噪声”，而是目标函数变了。它优化的是：

$$
\max_\pi \mathbb{E}\left[\sum_t r(s_t, a_t) + \alpha \mathcal{H}(\pi(\cdot|s_t))\right]
$$

其中 $\mathcal{H}$ 是策略熵，$\alpha$ 是温度系数，用来控制“拿高回报”和“保留随机性”之间的权衡。

| 算法 | 策略类型 | 是否显式熵项 | 价值网络 | 典型特点 |
|---|---|---:|---:|---|
| DDPG | 确定性策略 | 否 | 单 Q | 简单，但易过估计、易早收敛 |
| TD3 | 确定性策略 | 否 | 双 Q | 用双 Q 和目标动作噪声增强稳定性 |
| SAC | 随机策略 | 是 | 双 Q | 探索更稳，样本效率高，实现细节更多 |

玩具例子：假设一个一维小车可以向左或向右连续加速。如果策略过早学成“永远全力向右”，它可能在某个局部最优附近来回震荡。SAC 会让策略在高回报动作附近保留一定概率质量，因此更容易继续试出更优控制方式。

---

## 问题定义与边界

SAC 主要解决连续动作控制里的三个问题：探索难、训练不稳、样本昂贵。连续动作的意思是动作不是有限几个按钮，而是一个区间内的实数，比如机械臂关节扭矩、油门开度、舵角、相机云台速度。

真实工程例子：机械臂抓取。状态可能包含相机特征、夹爪位置、目标物体姿态；动作是多个关节的连续控制量。如果策略太确定，模型可能很快固化到“接近但抓不住”的次优动作序列。每次试错又要跑真实机器人或高保真仿真，成本高，因此需要一个既省样本又不容易塌缩的算法。

SAC 的边界也很明确。它不是所有任务的默认首选：

- 如果动作是离散的，比如“按左键、按右键、跳跃”，DQN 或离散 PPO 往往更直接。
- 如果任务很简单，状态维度低、局部最优少，TD3 或甚至 DDPG 都可能够用。
- 如果部署侧要求极低延迟且不希望在线采样随机动作，SAC 的随机策略结构未必最方便。

符号约定如下：

| 符号 | 含义 | 白话解释 |
|---|---|---|
| $s$ | state | 当前环境状态 |
| $a$ | action | 当前执行的动作 |
| $r$ | reward | 这一步拿到的分数 |
| $d$ | done | 是否终止，终止记为 1 |
| $\gamma$ | discount factor | 折扣因子，控制未来回报的重要性 |
| $\pi_\theta$ | policy | 由参数 $\theta$ 控制的策略网络 |
| $Q_{\phi_i}$ | critic | 第 $i$ 个 Q 网络 |
| $\alpha$ | temperature | 熵权重，控制随机性强弱 |

适用性可以先用一张表框住：

| 维度 | 适用 | 不适用 | 关键前提 |
|---|---|---|---|
| 动作空间 | 连续动作 | 纯离散动作 | 策略要能输出连续分布 |
| 样本成本 | 数据昂贵时更有价值 | 数据极廉价时优势变小 | 需要 replay buffer |
| 稳定性需求 | 高 | 一次性小实验也可不用 | 双 Q、目标网络、温度更新要实现正确 |
| 探索需求 | 局部最优多的任务 | 目标非常单一且确定 | 熵目标要合理 |

---

## 核心机制与推导

SAC 的核心不是一个技巧，而是一组互相配合的机制：双 Q 网络、随机策略重采样、熵正则、目标网络软更新。

先看目标 Q 值。SAC 的 Bellman 目标不是普通的
$r + \gamma Q(s', a')$，而是：

$$
a'_{t+1} \sim \pi_\theta(\cdot|s_{t+1})
$$

$$
y_t = r_t + \gamma(1-d_t)\left(\min_{j=1,2}Q_{\bar{\phi}_j}(s_{t+1}, a'_{t+1}) - \alpha \log \pi_\theta(a'_{t+1}|s_{t+1})\right)
$$

这里有两个关键点。

第一，取 $\min(Q_1, Q_2)$。这是为了抑制过高估计。因为神经网络逼近 Q 值时常会对少量动作给出虚高打分，策略一旦追着这些虚高值跑，误差会被放大。取最小值相当于更保守。

第二，$a'_{t+1}$ 不是 replay buffer 里存下来的旧动作，而是当前策略在 $s'$ 上重新采样得到。原因是 SAC 学的是“当前策略未来会怎么做”，因此目标必须和当前策略对齐，而不是和历史行为策略绑定。

critic 的损失很直接：

$$
L_Q(\phi_i)=\mathbb{E}_{(s_t,a_t,r_t,s_{t+1},d_t)\sim \mathcal{D}}
\left[\left(Q_{\phi_i}(s_t,a_t)-y_t\right)^2\right],\quad i\in\{1,2\}
$$

policy 的目标也和 DDPG/TD3 不同。SAC 不是直接让 actor 最大化 Q，而是最大化“Q 减去熵代价”的相反数，对应常见实现里的最小化形式：

$$
a_t \sim \pi_\theta(\cdot|s_t)
$$

$$
L_\pi(\theta)=\mathbb{E}_{s_t\sim \mathcal{D}, a_t\sim \pi_\theta}
\left[\alpha \log \pi_\theta(a_t|s_t)-\min_{j=1,2}Q_{\phi_j}(s_t,a_t)\right]
$$

这条式子的含义是：如果某个动作 Q 很高，策略倾向于增加它的概率；如果策略变得过于确定，$\log \pi$ 项会把它往更高熵方向拉一点。

自动温度调整是第三个关键机制。温度 $\alpha$ 决定熵项有多重。常见做法不是手工固定 $\alpha$，而是学习 $\log \alpha$，让策略熵接近目标熵 $\mathcal{H}_{target}$：

$$
L_\alpha(\alpha)=\mathbb{E}_{a_t\sim\pi_\theta}
\left[-\alpha \left(\log \pi_\theta(a_t|s_t)+\mathcal{H}_{target}\right)\right]
$$

白话解释是：如果当前策略熵低于目标，更新会推动 $\alpha$ 变大，增加探索压力；如果当前策略太随机，$\alpha$ 会下降。

可以用一个玩具数值例子把三条式子串起来。设：

- $r=1$
- $\gamma=0.99$
- $d=0$
- $Q_{\bar{\phi}_1}=10$
- $Q_{\bar{\phi}_2}=12$
- $\alpha=0.2$
- $\log \pi(a'|s')=-1.5$

则有：

$$
y = 1 + 0.99 \times (10 - 0.2\times(-1.5)) = 11.197
$$

如果当前某个 critic 预测 $Q_\phi(s,a)=12.0$，它的单样本平方误差为：

$$
(12.0 - 11.197)^2 \approx 0.645
$$

如果当前策略采样动作在本状态下的 $\log \pi(a|s)=-1.5$，且两个 critic 的较小值为 $10$，则 policy loss 样本值为：

$$
0.2\times(-1.5)-10 = -10.3
$$

数值越小并不代表策略“更差”，因为这里是最小化损失，负得更大通常意味着“高 Q 回报”这一项占优。

训练流程可以概括为：

| 步骤 | 输入 | 输出 | 目的 |
|---|---|---|---|
| 1 | replay buffer | $(s,a,r,s',d)$ | 取历史经验 |
| 2 | $s'$ 与当前 actor | $a', \log \pi(a'|s')$ | 构造目标 |
| 3 | target critics | $\min(\bar Q_1,\bar Q_2)$ | 保守估计未来价值 |
| 4 | current critics | critic loss | 更新两个 Q 网络 |
| 5 | current actor | policy loss | 更新策略 |
| 6 | $\log \alpha$ | temperature loss | 调整探索强度 |
| 7 | current critics | target critics | 软更新目标网络 |

---

## 代码实现

实现 SAC 时，建议把模块强制拆成五块：`actor`、`critic1`、`critic2`、`target_critic1/2`、`log_alpha`。不要把温度参数写成到处散落的常数，否则后面很难排查。

如果 actor 输出的是高斯分布参数，一般先采样未压缩动作 $u \sim \mathcal{N}(\mu, \sigma)$，再经过 `tanh` 压到动作边界内：

$$
a = \tanh(u)
$$

这时 `log π(a|s)` 不能直接用高斯的对数概率，必须做 squash 修正：

$$
\log \pi(a|s)=\log \pi_u(u|s)-\sum_k \log(1-\tanh^2(u_k)+\epsilon)
$$

这里的修正项本质上来自变量变换公式。白话解释是：你先在无界空间采样，再把动作硬压到 $[-1,1]$，概率密度会被挤压变形，不修正就会把熵算错。

下面给一个最小训练步骤示意，重点是顺序和依赖关系：

```python
import math

def sac_target(r, gamma, done, q1_targ, q2_targ, alpha, logp_next):
    min_q = min(q1_targ, q2_targ)
    return r + gamma * (1 - done) * (min_q - alpha * logp_next)

def critic_mse_loss(q_pred, y):
    return (q_pred - y) ** 2

def policy_loss(alpha, logp, q1, q2):
    return alpha * logp - min(q1, q2)

# 玩具样本
r = 1.0
gamma = 0.99
done = 0
q1_targ = 10.0
q2_targ = 12.0
alpha = 0.2
logp_next = -1.5

y = sac_target(r, gamma, done, q1_targ, q2_targ, alpha, logp_next)
loss_q = critic_mse_loss(12.0, y)
loss_pi = policy_loss(alpha, -1.5, 10.0, 11.0)

assert abs(y - 11.197) < 1e-6
assert abs(loss_q - (12.0 - 11.197) ** 2) < 1e-6
assert abs(loss_pi - (-10.3)) < 1e-6

print("target:", round(y, 3))
print("critic_loss:", round(loss_q, 3))
print("policy_loss:", round(loss_pi, 3))
```

一次标准 iteration 的伪代码可以写成：

```python
# sample batch
s, a, r, s_next, d = replay.sample(batch_size)

# target action from current policy, not from replay
a_next, logp_next = actor.sample(s_next)

# target Q
q1_next = target_critic1(s_next, a_next)
q2_next = target_critic2(s_next, a_next)
min_q_next = minimum(q1_next, q2_next)
y = r + gamma * (1 - d) * (min_q_next - alpha * logp_next)

# update critics
loss_q1 = mse(critic1(s, a), y.detach())
loss_q2 = mse(critic2(s, a), y.detach())
opt_q1.step(loss_q1)
opt_q2.step(loss_q2)

# update actor
a_pi, logp_pi = actor.sample(s)
q1_pi = critic1(s, a_pi)
q2_pi = critic2(s, a_pi)
loss_pi = mean(alpha * logp_pi - minimum(q1_pi, q2_pi))
opt_actor.step(loss_pi)

# update alpha
loss_alpha = mean(-log_alpha.exp() * (logp_pi.detach() + target_entropy))
opt_alpha.step(loss_alpha)
alpha = log_alpha.exp()

# soft update target critics
for targ, src in zip(target_params, critic_params):
    targ.data = tau * src.data + (1 - tau) * targ.data
```

真实工程例子：四足机器人速度控制。状态包含 IMU、关节角、触地信息；动作是每个关节的目标位置或扭矩。此时 replay buffer 中的动作往往来自很多旧策略版本。如果你直接用旧动作计算目标 Q，就等于在估计“旧策略的未来”；而 SAC 的更新目标是“当前策略的未来”，两者不一致会让训练发散或变慢。

---

## 工程权衡与常见坑

SAC 的稳定性不是白送的。它来自“保守估计 + 随机策略 + 熵约束”的组合，但这三件事任何一件实现错了，训练曲线都会明显变差。

最常见的错误实现是：直接拿 replay buffer 里的旧动作 $a_{t+1}^{old}$ 去算目标值。这样做看起来省事，但它破坏了目标定义。因为 SAC 的目标明确要求：

$$
a'_{t+1} \sim \pi_\theta(\cdot|s_{t+1})
$$

不是“从历史里取一个谁当时做过的动作”。旧动作反映的是行为策略，当前 actor 反映的是正在被优化的策略，二者不能混用。

常见坑可以集中看：

| 坑 | 错误现象 | 根因 | 修复方式 |
|---|---|---|---|
| 复用旧动作算 target | Q 学习抖动大、回报提升慢 | 目标和当前策略不一致 | 在 $s'$ 上重新采样 $a'$ |
| 漏掉 $(1-d)$ | 终止后 Q 仍被抬高 | 把不存在的未来回报算进去了 | target 乘 `(1 - done)` |
| `tanh` 后不修正 `log π` | 熵项异常、`alpha` 乱飘 | 概率密度变换没处理 | 加 squash correction |
| target critic 不软更新 | 曲线震荡或直接发散 | bootstrap 目标变化太快 | 用 Polyak/soft update |
| 额外叠加 TD3 target noise | 动作分布过乱 | 把两套平滑机制混在一起 | SAC 不需要 TD3 式 target noise |
| `alpha` 固定太大 | 策略很随机，回报上不去 | 熵惩罚过强 | 减小初值或启用自动调节 |
| `alpha` 固定太小 | 很快变近确定性，探索不足 | 熵约束太弱 | 提高初值或学习 `log_alpha` |

排查顺序也很重要。不要一上来就改网络宽度或学习率，先查逻辑正确性：

| 排查顺序 | 检查项 | 典型症状 |
|---|---|---|
| 1 | `log π` 是否含 `tanh` 修正 | `alpha` 发散，策略异常随机 |
| 2 | target 是否用重采样 `a'` | Q 波动大，actor 无法稳定提升 |
| 3 | 是否乘 `(1-d)` | episode 结束附近 Q 明显偏大 |
| 4 | 双 Q 是否真的取 `min` | 过估计，critic loss 降不住 |
| 5 | target 软更新频率和 `tau` | 训练忽快忽慢或突然崩 |
| 6 | `alpha`/目标熵设置 | 探索过度或过早塌缩 |

一个实用经验是：先把自动温度调节跑通，再考虑手调目标熵。连续动作维度为 $n$ 时，工程里常用类似 `target_entropy = -n` 的起点，但这只是经验值，不是定律。

---

## 替代方案与适用边界

SAC 不是“比所有 actor-critic 都高级”，而是“在连续动作、样本敏感、需要稳定探索时通常更合适”。

机器人臂轨迹跟踪是 SAC 的典型场景。动作是连续关节控制，误差曲面复杂，探索不足很容易卡住。相反，如果是离散按钮控制，比如小游戏里“左、右、跳、开火”四个动作，SAC 的连续随机策略优势并不明显，DQN 或 PPO 往往更自然。

几个常见替代方案的对比如下：

| 算法 | 更适合什么场景 | 相对 SAC 的优点 | 相对 SAC 的缺点 |
|---|---|---|---|
| DDPG | 简单连续控制 | 结构更简单，推理更直接 | 探索弱，稳定性差 |
| TD3 | 中等难度连续控制 | 确定性策略，工程成熟 | 没有显式熵项，探索不如 SAC |
| PPO | 仿真环境、并行采样方便 | 训练逻辑直观，调试门槛低 | 样本效率低，吃更多交互 |
| DQN | 离散动作 | 简单直接 | 不适合原生连续动作 |

SAC 和 TD3 很容易被放在一起比较，因为两者都用了双 Q。差异要抓住两点：

1. SAC 是随机策略，加了熵项；TD3 是确定性策略，不优化熵。
2. TD3 用 target policy smoothing，也就是给目标动作显式加噪；SAC 不这样做，因为它本来就在从当前随机策略重采样动作，已经带有“目标策略平滑”的效果，但这是策略分布内生得到的，不是额外手工加噪。

SAC 和 PPO 的边界也很清楚。SAC 是离策略，样本效率通常更高；PPO 是在策略，更新约束更直接，代码路径更短。数据昂贵时，SAC 常更值得；仿真无限、想快速起一个稳定 baseline 时，PPO 往往更省心。

如果部署时要求动作完全确定，可以在测试阶段直接取 actor 的均值动作，而不是继续采样。但这不代表训练阶段应该取消随机策略。训练需要探索，部署可以选择确定性输出，这两件事不要混在一起。

---

## 参考资料

1. [Soft Actor-Critic: Off-Policy Maximum Entropy Deep Reinforcement Learning with a Stochastic Actor](https://arxiv.org/abs/1801.01290)
2. [Soft Actor-Critic Algorithms and Applications](https://arxiv.org/abs/1812.05905)
3. [OpenAI Spinning Up: Soft Actor-Critic](https://spinningup.openai.com/en/latest/algorithms/sac.html)
4. [Stable-Baselines3 Documentation: SAC](https://stable-baselines3.readthedocs.io/en/master/modules/sac.html)
5. [haarnoja/sac](https://github.com/haarnoja/sac)
