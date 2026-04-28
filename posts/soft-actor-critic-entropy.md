## 核心结论

Soft Actor-Critic，简称 SAC，是一种把“高回报”和“高随机性”同时写进目标函数的 Actor-Critic 方法。Actor-Critic 的白话解释是：一部分网络负责“评估动作值不值”，另一部分网络负责“真的去选动作”。SAC 的关键不是“在动作上加点噪声”，而是直接优化下面这个目标：

$$
J(\pi)=\mathbb E\Big[\sum_{t=0}^{\infty}\gamma^t\big(r_t+\alpha H(\pi(\cdot|s_t))\big)\Big]
$$

其中，熵 $H(\pi(\cdot|s))$ 可以理解为“策略有多不确定”。如果一个状态下策略总是死盯一个动作，熵就低；如果还保留多种可能，熵就高。$\alpha$ 是温度系数，用来控制“奖励最大化”和“保留随机性”之间的权重。

SAC 比普通贪心策略更稳，原因可以用一句话说清：普通方法只想立刻选当前看起来最优的动作，SAC 会额外保留一点试错空间，所以更不容易一开始就把路走死。对连续控制任务，这个性质很重要，因为训练初期的价值估计通常不准，太早确定化，后面往往很难再改回来。

| 维度 | 普通 Actor-Critic | SAC |
|---|---|---|
| 优化目标 | 主要追求高回报 | 高回报 + 高熵 |
| 策略形式 | 常逐渐趋向确定 | 显式保留随机性 |
| 探索方式 | 常靠外加噪声 | 目标函数内生鼓励探索 |
| 训练稳定性 | 早期容易过早贪心 | 更稳，较不易卡死 |
| 典型场景 | 一般控制任务 | 连续动作、探索困难任务 |

从工程角度看，SAC 最有价值的不是名字，而是两个结果：连续控制更稳，训练早期更不容易过早确定化。四足机器人走路、机械臂抓取、灵巧手控制这类任务，很多时候都符合这个特点。

---

## 问题定义与边界

SAC 主要解决的对象，是离策略、连续动作、需要较强探索的控制问题。离策略的白话解释是：训练时可以反复利用历史数据，不要求“当前采样的数据必须来自当前最新策略”。这意味着它通常比很多在策略方法更省样本。

强化学习的基本记号如下：

- 状态 $s$：环境当前信息，比如机器人的关节角度、速度、传感器读数。
- 动作 $a$：智能体输出的控制量，比如每个电机的扭矩。
- 奖励 $r$：环境给出的即时反馈。
- 策略 $\pi(a|s)$：在状态 $s$ 下选择动作 $a$ 的分布。
- 折扣回报：未来奖励的加权和，常写为 $\sum_t \gamma^t r_t$。

如果把机器人学走路当成一个新手例子，可以这样理解：每一步都要在安全、速度、稳定之间平衡。如果一开始动作就非常确定，比如某个关节总是大幅前摆，它可能很快摔倒，并把错误步态反复强化。SAC 允许“带着不确定性学习”，所以更容易先找到可行走法，再逐步收敛。

SAC 不是所有任务的自然首选。它最适合连续动作空间，比如 $a\in\mathbb R^n$。对于离散动作，虽然有离散版 SAC 变体，但从直观性和实现便利看，并不是它的主场。另一个边界是奖励设计。如果奖励本身混乱，比如既鼓励加速又隐含惩罚动作幅度，但权重设置错误，再好的 SAC 也只会稳定地学错目标。

| 任务类型 | 适配性 | 原因 |
|---|---|---|
| 连续控制 | 高 | 高斯策略、重参数化、双 Q 都很自然 |
| 离散动作 | 中 | 可改造，但不是最简主线 |
| 稀疏奖励 | 中偏高 | 熵有助于探索，但不能代替奖励塑形 |
| 观测噪声较大 | 中 | 随机策略有缓冲，但仍依赖状态质量 |

从训练边界看，SAC 通常包含 replay buffer。replay buffer 的白话解释是“把过去交互过的数据存起来，后面反复抽样训练”。它的基本流程是：与环境交互、存数据、从 buffer 采样小批量、更新网络。这里要注意，buffer 负责提供旧状态转移，但 target 里的下一步动作必须由当前策略重新采样，而不是直接拿旧动作顶替。

```python
def interaction_loop(env, policy, replay_buffer, steps=1000):
    s = env.reset()
    for _ in range(steps):
        a = policy.sample_action(s)   # 当前策略采样
        s2, r, done, info = env.step(a)
        replay_buffer.add(s, a, r, s2, done)
        s = env.reset() if done else s2
```

---

## 核心机制与推导

SAC 的起点是最大熵强化学习。最大熵的意思不是“尽量乱选动作”，而是在高回报前提下，额外奖励保留不确定性。形式上，它把每一步目标从单独的 $r_t$ 改成了 $r_t+\alpha H(\pi(\cdot|s_t))$。

熵定义为：

$$
H(\pi(\cdot|s))=-\mathbb E_{a\sim \pi}[\log \pi(a|s)]
$$

因为高概率动作的 $\log \pi(a|s)$ 更接近 0，低概率动作更负，所以负号之后，分布越分散，熵越大。

SAC 的 soft policy improvement 可以写成一个 KL 散度最小化问题。KL 散度的白话解释是“两个分布差得有多远”。它要求新策略尽量接近一个由 soft Q 指数化得到的目标分布：

$$
\pi_{\text{new}}(\cdot|s)=\arg\min_{\pi'}D_{\mathrm{KL}}\left(\pi'(\cdot|s)\,\middle\|\,\frac{\exp(Q^\pi(s,\cdot)/\alpha)}{Z(s)}\right)
$$

这条式子表达的直觉是：如果某个动作的 $Q$ 很高，它在目标分布里的权重就更高；但不是“一票否决”其他动作，因为分布仍保留温度 $\alpha$ 带来的平滑性。于是 actor 不会只盯着一个点，而是在“高 Q 区域”内保留随机性。

实践中，SAC 落成三部分：

1. Critic 学 soft Q。
2. Actor 最大化 $Q-\alpha\log\pi$。
3. Temperature 自动调节 $\alpha$，让熵维持在目标附近。

核心公式如下：

$$
y=r+\gamma(1-d)\Big(\min(Q'_{1}(s',a'),Q'_{2}(s',a'))-\alpha \log \pi(a'|s')\Big),\quad a'\sim \pi(\cdot|s')
$$

$$
J_{\pi}=\mathbb E\big[\alpha\log\pi(a|s)-\min(Q_1(s,a),Q_2(s,a))\big]
$$

$$
J(\alpha)=\mathbb E\big[\alpha(-\log\pi(a|s)-\mathcal H_{\text{target}})\big]
$$

实际里常优化 $\log\alpha$ 而不是直接优化 $\alpha$，因为这样能天然保证 $\alpha>0$。

| 公式 | 作用 | 输入 | 输出 | 直观含义 |
|---|---|---|---|---|
| $J(\pi)$ | 总目标 | 奖励、熵、$\alpha$ | 策略优劣 | 不只追求高回报，也保留探索 |
| soft policy improvement | 策略改进 | $Q,\alpha$ | 新策略分布 | 向高 Q 动作靠拢，但不彻底贪心 |
| $y$ | critic target | $r,s',d,\pi,Q'$ | 监督目标 | 学“下一步继续最优时”的软价值 |
| $J_\pi$ | actor 更新 | 当前策略、Q | 策略梯度目标 | 鼓励高 Q，同时惩罚过于确定 |
| $J(\alpha)$ | 温度更新 | 当前熵、目标熵 | $\alpha$ 调整方向 | 熵太低就增大探索，太高就收紧 |

看一个玩具例子。假设某个状态只有两个动作：

- 动作 A：$Q=10$，但策略极度确定，$\log\pi(A|s)=-0.05$
- 动作 B：$Q=9$，但更分散，$\log\pi(B|s)=-1.2$
- 设 $\alpha=0.5$

则 actor 实际比较的不是单独的 $Q$，而是 $Q-\alpha\log\pi$：

- A: $10-0.5\times(-0.05)=10.025$
- B: $9-0.5\times(-1.2)=9.6$

这时 A 仍更优。但如果 A 的 Q 优势再小一点，比如 A 的 Q 只有 9.2，那么 A 的值变成 9.225，而 B 仍是 9.6，此时策略就会偏向 B。意思是：SAC 会接受“回报略低，但能维持更多探索空间”的动作。

再看一个最小数值例子。设 $\gamma=0.99,\alpha=0.2,r=1,d=0,\min Q'(s',a')=5.0,\log\pi(a'|s')=-0.7$，那么：

$$
y=1+0.99\times(5.0-0.2\times(-0.7))=1+0.99\times 5.14=6.0886
$$

如果当前状态下 actor 采样动作的 $\min(Q_1,Q_2)=4.5,\log\pi=-0.7$，那么它偏好的目标量是：

$$
Q-\alpha\log\pi=4.5-0.2\times(-0.7)=4.64
$$

这个数值说明，SAC 对“更随机”的策略是给分的，但给分幅度由 $\alpha$ 控制，不会无限放大。

真实工程例子是四足机器人行走。早期训练时，价值网络对“迈大步”还是“先稳站住”判断都不准。如果直接走确定性策略，很容易在某种错误步态上过拟合。SAC 通过熵项和自动温度，允许策略在高价值区域附近持续试探，因此更容易从“能站住”过渡到“能走稳”。

---

## 代码实现

SAC 的实现重点，不是把论文公式抄进代码，而是把数据流接对：从 replay buffer 取一批样本，先用当前 actor 在下一状态 $s'$ 上采样动作 $a'$，再计算 soft target，更新两个 critic，然后更新 actor，最后更新温度和目标网络。

下面这段 Python 代码是可运行的最小演示，省略了神经网络，只保留损失和更新顺序。它展示了 `sample action -> compute soft target -> update critic -> update actor -> update alpha` 的主链路。

```python
import math

def soft_target(r, gamma, done, q1_t, q2_t, alpha, logp_next):
    return r + gamma * (1 - done) * (min(q1_t, q2_t) - alpha * logp_next)

def critic_loss(q_pred, y):
    return (q_pred - y) ** 2

def actor_objective(q_val, alpha, logp):
    return q_val - alpha * logp

def alpha_loss(log_alpha, logp, target_entropy):
    alpha = math.exp(log_alpha)
    return alpha * (-(logp) - target_entropy)

# toy batch: 单样本
r = 1.0
gamma = 0.99
done = 0.0
q1_target = 5.2
q2_target = 5.0
logp_next = -0.7
log_alpha = math.log(0.2)

y = soft_target(r, gamma, done, q1_target, q2_target, math.exp(log_alpha), logp_next)
assert abs(y - 6.0886) < 1e-4

q1_pred = 5.8
q2_pred = 6.0
loss_q1 = critic_loss(q1_pred, y)
loss_q2 = critic_loss(q2_pred, y)
assert loss_q1 >= 0 and loss_q2 >= 0

q_for_actor = 4.5
logp = -0.7
obj = actor_objective(q_for_actor, math.exp(log_alpha), logp)
assert abs(obj - 4.64) < 1e-6

target_entropy = 1.0
loss_alpha = alpha_loss(log_alpha, logp, target_entropy)
assert loss_alpha > 0
```

如果写成训练伪代码，结构通常如下：

```python
# sample batch: s, a, r, s2, done
a2, logp_a2 = actor.sample(s2)
y = r + gamma * (1 - done) * (min(target_q1(s2, a2), target_q2(s2, a2)) - alpha * logp_a2)

critic_loss = mse(q1(s, a), y) + mse(q2(s, a), y)
update(critic_loss)

a_new, logp_a = actor.sample(s)
actor_loss = mean(alpha * logp_a - min(q1(s, a_new), q2(s, a_new)))
update(actor_loss)

alpha_loss = mean(alpha * (-logp_a - target_entropy))
update(alpha_loss)

soft_update(target_q, q, tau)
```

SAC 的 actor 常输出高斯分布参数：均值 $\mu(s)$ 和对数标准差 $\log\sigma(s)$。然后通过重参数化技巧采样：

$$
u=\mu+\sigma\odot \epsilon,\quad \epsilon\sim\mathcal N(0,I),\quad a=\tanh(u)
$$

重参数化的白话解释是：把“随机采样”拆成“确定性变换 + 外部噪声”，这样梯度能从动作一路传回策略参数。`tanh` 用来把动作压到合法区间，比如 $[-1,1]$。但 `tanh` 会改变概率密度，因此 `log_prob` 不能直接用高斯分布的结果，必须做 Jacobian 修正：

$$
\log\pi(a|s)=\log\mathcal N(u;\mu,\sigma)-\sum_i \log(1-\tanh^2(u_i))
$$

| 张量 | 典型形状 | 来源 | 作用 |
|---|---|---|---|
| `s` | `[B, obs_dim]` | replay buffer | 当前状态 |
| `a` | `[B, act_dim]` | replay buffer | 历史动作 |
| `r` | `[B, 1]` | replay buffer | 即时奖励 |
| `s2` | `[B, obs_dim]` | replay buffer | 下一状态 |
| `done` | `[B, 1]` | replay buffer | 终止标记 |
| `mu, log_std` | `[B, act_dim]` | actor | 高斯分布参数 |
| `q1, q2` | `[B, 1]` | twin critics | 双 Q 估计 |

目标网络一般用 EMA，也就是指数滑动平均更新：

$$
\bar\phi \leftarrow \tau \phi + (1-\tau)\bar\phi
$$

这里 $\tau$ 很小，比如 0.005，作用是让 target 更平滑，减少训练目标剧烈抖动。

---

## 工程权衡与常见坑

SAC 的稳定，不来自某一个神奇公式，而来自一组细节同时成立：双 Q、target network、当前策略重采样、正确的 `log_prob` 修正、合理的 target entropy、评估时改用确定性动作。任何一个环节接错，都可能出现“代码看起来像 SAC，但训练完全不动”。

最常见的坑，是把 target 里的动作直接从 replay buffer 里拿。这样做的问题是，critic 的目标不再是在评估“当前策略下一步会怎么做”，而是在追旧策略留下的动作，目标会系统性偏掉。

| 常见坑 | 后果 |
|---|---|
| 没有 twin Q 取最小值 | Q 过估计，actor 被假高值带偏 |
| 忘记 target 网络 | 目标抖动大，critic 容易发散 |
| `log_prob` 没做 tanh 修正 | 概率估计错误，actor/alpha 更新失真 |
| 评估时还在随机采样 | 指标波动大，误判模型效果 |
| target entropy 不合理 | 探索过强或过弱，收敛慢甚至失败 |

错误与正确写法可以直接对比：

```python
# wrong: 直接用 buffer 里的下一动作，SAC target 偏了
y = r + gamma * (1 - done) * (min(target_q1(s2, a_old), target_q2(s2, a_old)) - alpha * old_logp)
```

```python
# correct: 在 s2 上用当前策略重新采样
a2, logp_a2 = actor.sample(s2)
y = r + gamma * (1 - done) * (min(target_q1(s2, a2), target_q2(s2, a2)) - alpha * logp_a2)
```

另一个高频问题是 target entropy。常见初值是：

$$
\mathcal H_{\text{target}}=-|A|
$$

其中 $|A|$ 是动作维度。这个公式不是定律，而是经验起点。动作维度越高，目标熵通常越低一些，表示策略需要保留一定整体随机性。但具体任务差异很大，比如机械臂精细操作和移动控制，对随机性的容忍度并不一样。

自动温度更新常写成：

$$
J(\log\alpha)=\mathbb E\big[\exp(\log\alpha)(-\log\pi(a|s)-\mathcal H_{\text{target}})\big]
$$

如果当前熵低于目标熵，$-\log\pi(a|s)-\mathcal H_{\text{target}}$ 往往偏大，优化会推动 $\alpha$ 增大，让策略更随机；反之则减小 $\alpha$。

还有一个经常被忽略的问题是评估模式。训练时要采样，评估时通常取均值动作，即用 $\mu(s)$ 而不是重新从高斯中抽样。否则你测到的是“随机策略表现”，而不是“当前学到的控制能力上限”。

---

## 替代方案与适用边界

SAC 不是默认最优解。选算法时，先看动作空间、探索难度、样本效率和部署要求，再看论文热度。

DDPG 是确定性离策略算法，探索通常依赖额外动作噪声。TD3 在 DDPG 基础上增加双 Q、延迟策略更新和目标策略平滑，主要解决过估计和训练不稳。PPO 是在策略方法，训练更直接，但样本利用率通常不如离策略方法。A2C 结构更简单，但在复杂连续控制里常不如 SAC 稳。

| 算法 | 离策略 | 最大熵 | 探索能力 | 样本效率 | 实现复杂度 | 典型应用 |
|---|---|---|---|---|---|---|
| SAC | 是 | 是 | 强 | 高 | 中高 | 连续控制、机器人 |
| DDPG | 是 | 否 | 中 | 高 | 中 | 简单连续控制 |
| TD3 | 是 | 否 | 中偏强 | 高 | 中高 | 连续控制稳健基线 |
| PPO | 否 | 否 | 中 | 中低 | 中 | 通用 RL、工程落地广 |
| A2C | 否 | 否 | 中低 | 低 | 低 | 教学、简单基线 |

如果只看目标形式，DDPG/TD3 更像：

$$
y_{\text{DDPG/TD3}}=r+\gamma(1-d)Q'(s',\mu'(s'))
$$

而 SAC 是：

$$
y_{\text{SAC}}=r+\gamma(1-d)\big(\min Q'(s',a')-\alpha\log\pi(a'|s')\big)
$$

差别在于，SAC 的 target 里明确加入了策略熵惩罚项，因此它对“过早变得太确定”更敏感，也更保守。

给一个实用决策规则：

```python
def choose_algo(action_space, sample_budget_low, exploration_hard, need_simple_impl):
    if action_space == "continuous" and sample_budget_low and exploration_hard:
        return "SAC"
    if action_space == "continuous" and need_simple_impl:
        return "TD3 or DDPG"
    if action_space == "discrete":
        return "PPO / DQN-family / discrete SAC variant"
    return "Start from PPO or TD3 baseline"
```

新手可以记住一句话：如果你做的是连续控制、样本贵、又需要稳定探索，SAC 往往更合适；如果你做的是简单离散决策，或者你更在意实现直接、约束清晰的在线更新，PPO、TD3 等方法常更合适。

---

## 参考资料

下表按“理论 -> 实现 -> 解释”排序：

| 来源 | 类型 | 适合阅读阶段 | 可提取内容 |
|---|---|---|---|
| SAC 2018 论文 | 理论 | 初次建立完整概念 | 最大熵目标、soft policy iteration、核心损失 |
| SAC Algorithms and Applications | 理论+实践 | 理解改进版算法 | 自动温度、工程实现细节 |
| `softlearning` | 官方实现 | 写代码前后对照 | 网络结构、训练流程、参数组织 |
| Spinning Up SAC | 教程 | 快速把握流程 | 关键公式、伪代码、实现提示 |

建议阅读顺序可以很简单：先看论文摘要和算法框图，再看 Spinning Up 把流程串起来，最后对照官方实现理解细节落地。

1. [Soft Actor-Critic: Off-Policy Maximum Entropy Deep Reinforcement Learning with a Stochastic Actor](https://proceedings.mlr.press/v80/haarnoja18b.html)
2. [Soft Actor-Critic Algorithms and Applications](https://huggingface.co/papers/1812.05905)
3. [rail-berkeley/softlearning](https://github.com/rail-berkeley/softlearning)
4. [OpenAI Spinning Up - SAC](https://spinningup.openai.com/en/latest/algorithms/sac.html)
