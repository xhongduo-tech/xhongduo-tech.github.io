## 核心结论

A2C，Advantage Actor-Critic 的同步版本，本质上是在同一轮里让多个并行环境先各自采样固定步数，再把这些样本合成一个批次，统一计算优势函数 $A_t = R_t - V(s_t)$，最后只做一次参数更新。白话说，就是“先把一批经验攒齐，再统一学习”，而不是谁先拿到样本谁先改参数。

它解决的是 A3C 里常见的参数陈旧问题。参数陈旧的意思是：某个 worker 还在用旧参数采样，另一个 worker 已经把模型改过了，导致同一轮数据的来源不一致。A2C 用同步屏障避免这个问题，因此训练曲线通常更平稳，复现实验也更容易。

A2C 还有一个现实优势：更适合 GPU 批处理。GPU 擅长一次处理大批量张量，而 A2C 天然把 $N$ 个环境、每个环境 $n$ 步的数据拼成一个大 batch，所以常比 A3C 更容易把硬件吃满。代价也很明确：每轮更新都要等最慢的环境结束，吞吐上限常常不是由平均速度决定，而是由尾部慢环境决定。

一个最小玩具例子可以直接说明同步聚合的含义。假设有两个环境，第一条轨迹得到的优势均值是 $+2$，第二条轨迹是 $-1$，那么这轮更新的平均优势就是：

$$
\bar A = \frac{2 + (-1)}{2} = 0.5
$$

这不是“只奖励好环境”，而是“把所有环境的评价汇总后，再做一次更稳定的更新”。

---

## 问题定义与边界

A2C 要解决的问题不是“如何让强化学习更快”，而是“如何在并行采样时保证更新一致性”。一致性这里指：一次更新所用的数据，尽量都来自同一份策略参数。否则你会得到一种很难分析的混合状态：一部分样本来自旧策略，一部分来自新策略，梯度方向会更噪。

问题可以抽象成下面这个边界：

| 输入 | 约束 | 输出 |
|---|---|---|
| $N$ 个并行环境 | on-policy，数据必须尽快使用 | 一次同步更新 |
| 每个环境采样 $n$ 步 | 必须等待所有环境到达同步点 | 稳定的 Actor/Critic 梯度 |
| 共享 Actor/Critic 参数 | 不能随意复用很久以前的样本 | 更可复现的训练过程 |

这里的 on-policy，意思是“当前策略采的样本，主要服务当前这版策略更新”。白话说，就是数据和模型版本要尽量同步，不能像经验回放那样长期囤积旧数据再反复训练。

所以 A2C 的适用边界也很清楚：

1. 需要并行环境。
2. 接受同步等待。
3. 希望做批量更新。
4. 任务允许 on-policy 训练。
5. 更关注稳定性，而不是极致样本效率。

真实工程里，一个典型场景是机器人控制训练。比如 8 个同步环境同时跑机械臂抓取任务，每个环境都用当前策略执行 5 步，然后统一把 8×5 条转移送进优化器。只要有一个环境因为物理碰撞或仿真抖动算得慢，其他 7 个环境也得等。这就是 A2C 的核心边界：它换来一致性，也承担同步等待成本。

可以把流程简化成一张文字图：

| 阶段 | 做什么 | 为什么 |
|---|---|---|
| 并行采样 | $N$ 个环境一起跑 $n$ 步 | 提高吞吐 |
| 估值 | 用 Critic 计算 $V(s_t)$ | 给回报一个基线 |
| 算优势 | $A_t = R_t - V(s_t)$ | 判断动作比“正常水平”好还是差 |
| 同步聚合 | 拼成一个 batch | 保证同轮样本来源一致 |
| 一次更新 | 同时更新 Actor 和 Critic | 降低梯度噪声 |

---

## 核心机制与推导

A2C 里有两个核心角色。

Actor 是策略网络，负责输出“在状态 $s$ 下该选什么动作”的概率分布。白话说，它是决策者。Critic 是价值网络，负责估计“当前状态大概值多少钱”，通常记作 $V_\phi(s)$。白话说，它是评分员。

如果只有 Actor，没有 Critic，那么策略梯度的波动会很大，因为你只能直接看回报好不好。A2C 的做法是让 Critic 先给一个基线，再看真实回报比这个基线高还是低，这个差值就是优势函数：

$$
A_t = R_t - V_\phi(s_t)
$$

这里 $R_t$ 是从时刻 $t$ 往后的多步回报。多步回报可以写成：

$$
R_t = r_t + \gamma r_{t+1} + \gamma^2 r_{t+2} + \cdots
$$

$\gamma$ 是折扣因子，表示“越远的未来，权重越低”。

在并行环境下，A2C 的策略梯度通常写成：

$$
\nabla_\theta J(\theta) =
\frac{1}{N}\sum_{i=1}^{N}\sum_{t=0}^{n-1}
\nabla_\theta \log \pi_\theta(a_t^i \mid s_t^i)\, A_t^i
$$

这个式子可以拆成三层理解：

1. 对每个环境 $i$，先看它每一步的动作概率梯度。
2. 用优势 $A_t^i$ 给这个梯度加权。
3. 最后对所有环境求平均，得到一轮同步更新方向。

为什么这样做能更稳？因为多个环境给出的梯度方向会互相抵消一部分随机噪声，只留下更稳定的共同趋势。

很多实现里不会直接用长回报，而是用 GAE，Generalized Advantage Estimation，广义优势估计。白话说，它是“把多步 TD 残差按衰减权重叠起来”，在偏差和方差之间做折中。先定义一步 TD 残差：

$$
\delta_t = r_t + \gamma V(s_{t+1}) - V(s_t)
$$

再定义 GAE：

$$
A_t^{\text{GAE}(\gamma,\lambda)} =
\delta_t + (\gamma\lambda)\delta_{t+1} + (\gamma\lambda)^2\delta_{t+2} + \cdots
$$

其中 $\lambda$ 控制“看多远”。$\lambda$ 越大，优势越接近多步回报，方差更大；$\lambda$ 越小，优势更接近一步 TD，偏差更大但更稳。

看一个两步玩具例子。假设某条轨迹在两个时间步上有：

- $r_0 = 1.0$
- $r_1 = 2.0$
- $V(s_0)=1.2$
- $V(s_1)=1.5$
- $V(s_2)=0.8$
- $\gamma=0.9$
- $\lambda=0.95$

先算 TD 残差：

$$
\delta_1 = r_1 + \gamma V(s_2) - V(s_1) = 2.0 + 0.9\times 0.8 - 1.5 = 1.22
$$

$$
\delta_0 = r_0 + \gamma V(s_1) - V(s_0) = 1.0 + 0.9\times 1.5 - 1.2 = 1.15
$$

再回推优势：

$$
A_1 = \delta_1 = 1.22
$$

$$
A_0 = \delta_0 + \gamma\lambda A_1
= 1.15 + 0.9\times 0.95\times 1.22
\approx 2.1931
$$

解释很直接：第 0 步不只看眼前一步，还把后面一步“还不错”的信息也折回来，所以它的优势更大。

整个机制可以压缩成一句流程：

采样 $\rightarrow$ 估值 $\rightarrow$ 算 $\delta$ $\rightarrow$ 回推 GAE $\rightarrow$ 归一化优势 $\rightarrow$ 更新 Actor/Critic。

这里还有一个常见工程细节：优势归一化。意思是把一个 batch 内的优势减均值、除标准差，让它们的尺度更稳定。否则某几个环境奖励特别大，会主导整轮更新。

---

## 代码实现

A2C 的最小实现并不复杂，核心循环就是“同步采样 + 统一算优势 + 一次更新”。下面先给一个可运行的 Python 版本，只演示 GAE 和同步聚合，不依赖深度学习框架。

```python
from math import isclose

def compute_gae(rewards, values, gamma=0.99, lam=0.95):
    """
    rewards: 长度 T
    values: 长度 T + 1，最后一个是 bootstrap value
    """
    T = len(rewards)
    advantages = [0.0] * T
    gae = 0.0
    for t in reversed(range(T)):
        delta = rewards[t] + gamma * values[t + 1] - values[t]
        gae = delta + gamma * lam * gae
        advantages[t] = gae
    returns = [advantages[t] + values[t] for t in range(T)]
    return advantages, returns

# 玩具例子：两步轨迹
rewards = [1.0, 2.0]
values = [1.2, 1.5, 0.8]
advantages, returns = compute_gae(rewards, values, gamma=0.9, lam=0.95)

assert len(advantages) == 2
assert len(returns) == 2
assert isclose(advantages[1], 1.22, rel_tol=1e-6)
assert isclose(advantages[0], 2.1931, rel_tol=1e-4)

# 同步并行：两个环境的优势做批量聚合
env1_adv = [2.0, 1.0, 0.0]
env2_adv = [-1.0, 0.5, 0.5]
batch_adv = env1_adv + env2_adv
mean_adv = sum(batch_adv) / len(batch_adv)

assert isclose(mean_adv, 0.5, rel_tol=1e-6)
print("GAE and synchronous aggregation are correct.")
```

这段代码验证了两件事：

1. GAE 是从后往前回推的。
2. 多环境同步时，本质就是把所有环境的优势拼成一个 batch 再聚合。

如果再往前走一步，A2C 主循环的伪代码通常长这样：

```python
for update in range(num_updates):
    storage = []

    # 1. N 个环境同步采样 n 步
    for step in range(n_steps):
        action = actor(obs_batch)
        next_obs, reward, done, info = vec_env.step(action)
        value = critic(obs_batch)
        storage.append((obs_batch, action, reward, done, value))
        obs_batch = next_obs

    # 2. 用最后一个状态做 bootstrap
    last_value = critic(obs_batch)

    # 3. 计算 returns 和 advantages
    returns, advantages = compute_gae_from_storage(storage, last_value)

    # 4. 拼成一个大 batch
    batch = flatten(storage, returns, advantages)

    # 5. 更新 Actor
    actor_loss = -(log_prob(batch) * batch.advantages).mean()

    # 6. 更新 Critic
    critic_loss = ((value_pred(batch) - batch.returns) ** 2).mean()

    # 7. 总损失并一步 optimizer
    loss = actor_loss + value_coef * critic_loss - entropy_coef * entropy(batch)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
```

新手理解这段伪代码时，只需要抓住三个固定点：

1. 采样是按“环境数 × 时间步数”展开的。
2. 优势和回报是整批算，不是边采边改。
3. 优化器每轮只 step 一次或少数几次。

常见配置项也可以直接按功能理解：

| 配置项 | 作用 | 调大后的典型影响 |
|---|---|---|
| `num_envs` | 并行环境数 | 吞吐上升，但同步等待更明显 |
| `n_steps` | 每个环境每轮采样步数 | batch 更大，但样本更新更晚 |
| `gamma` | 折扣因子 | 更重视长期回报 |
| `gae_lambda` | GAE 衰减系数 | 更接近多步回报，方差变大 |
| `value_coef` | Critic 损失权重 | 估值学习更强 |
| `entropy_coef` | 熵正则权重 | 鼓励探索，防止策略过早变确定 |
| `max_grad_norm` | 梯度裁剪阈值 | 防止更新过猛 |
| `normalize_advantage` | 是否归一化优势 | 提升训练稳定性 |

真实工程例子里，常见实现是 `VecEnv + PyTorch`。比如单机 8 个 Atari 或 Mujoco 环境并行，前向时把 8 个环境状态拼成张量 `[8, obs_dim]`，滚动 5 步后得到 `[8, 5, ...]` 的轨迹缓存，再 reshape 成 `[40, ...]` 送入损失函数。这个 reshape 很关键，因为它把“多环境并行”统一变成“普通 batch 学习”。

---

## 工程权衡与常见坑

A2C 在纸面上很干净，但工程上最容易出问题的地方也恰恰来自“同步”。

第一个坑是尾部延迟。尾部延迟的意思是：系统整体速度不是看平均环境多快，而是看最慢那个多慢。只要有少数环境因为仿真复杂、重置慢、网络通信抖动而偶发卡顿，整轮更新都得停下来等。

第二个坑是 batch 太大导致样本不新鲜。样本新鲜度的意思是：采样数据离当前参数版本有多近。A2C 虽然比 A3C 更一致，但如果你设了太大的 `num_envs × n_steps`，那一轮更新前已经攒了很多旧样本，策略可能已经该改了却还没改，更新会变钝。

第三个坑是环境间尺度不一致。比如一个环境奖励范围在 $[-1, 1]$，另一个在 $[-100, 100]$，不做奖励缩放或优势归一化时，大尺度环境会压过其他环境。

第四个坑是把 A2C 当成“任意大 batch 的 PPO”。这是错的。A2C 没有 PPO 那种 clipping 约束，更新步子如果太大，很容易不稳定，所以学习率、梯度裁剪和熵正则都更敏感。

下面这张表可以直接对照常见延迟来源：

| 延迟来源 | 现象 | 解决手段 |
|---|---|---|
| 环境负载不均 | 个别 env 经常拖后腿 | 做 profiler，替换慢环境或分组调度 |
| `n_steps` 过大 | 每轮等待时间长 | 缩短 rollout 长度 |
| IPC/通信开销高 | 子进程切换频繁 | 合并消息、减少序列化、改共享内存 |
| 模型前向太重 | GPU/CPU 成为瓶颈 | 降低网络复杂度，混合精度 |
| 奖励尺度差异大 | 梯度波动明显 | 奖励裁剪、优势归一化 |
| done 处理错误 | returns/advantages 异常 | 明确区分终止和截断 |

看一个真实工程例子。假设在一张 GPU 上跑 8 个并行机器人环境，其中 7 个环境平均每步 3ms，另 1 个环境因为碰撞检测复杂，偶尔飙到 20ms。A2C 每轮都要等 8 个环境都完成 5 步采样，那么整轮时间会被那个慢环境放大。解决思路通常不是先改算法，而是先做轻量 profiler：统计每个 env 的 step 时间分布、reset 时间分布和 done 比例，再决定是做负载均衡、降低仿真精度，还是把极慢场景单独拆出去。

还有一个常见实现错误是 bootstrap 处理。若最后一步不是终止状态，就要把 $V(s_{t+1})$ 接到 GAE 里；若已经真正终止，就应把 bootstrap 视为 0。很多初学者把两种情况混了，导致 returns 系统性偏高或偏低。

---

## 替代方案与适用边界

A2C 不是并行强化学习的终点，而是一种折中方案。它在同步一致性和工程效率之间取得了比较好的平衡，但并不是所有场景都该选它。

先看几种常见算法的对比：

| 算法 | 是否同步 | 样本利用方式 | 稳定性 | 典型适用场景 |
|---|---|---|---|---|
| A3C | 否 | on-policy，异步更新 | 中等 | CPU 多线程、环境差异大 |
| A2C | 是 | on-policy，同步批更新 | 较高 | GPU 批处理、追求复现 |
| PPO | 通常是 | on-policy，多轮小步更新 | 更高 | 高方差任务、主流基线 |

A3C 的优点是不用等最慢 worker，谁采完谁更新，资源利用在纯 CPU 场景下常常更自然。但代价是参数版本更乱，梯度噪声更大，训练结果更依赖实现细节。

A2C 适合下面这些边界：

1. 你有多个可并行环境。
2. 你希望同一轮样本来自同一份策略。
3. 你有 GPU，希望把多个环境打包前向。
4. 任务对训练稳定性要求高于极致样本效率。

PPO 是 A2C 的常见升级路线。它保留了 Actor-Critic 和优势估计，但加了 clipping 或 KL 约束，限制每次策略变化不要太大。白话说，A2C 更像“直接按优势推一步”，PPO 更像“推，但不能推太猛”。所以在高方差任务上，PPO 往往更稳，也因此成为很多工程项目的默认选择。

如果要更直白地给建议：

| 需求 | 更合适的选择 | 原因 |
|---|---|---|
| CPU 为主、环境速度差异大 | A3C | 不用严格同步等待 |
| GPU 为主、要稳定复现 | A2C | 同步批处理更自然 |
| 任务复杂、更新容易震荡 | PPO | 有额外步长约束 |
| 希望反复利用旧数据 | DQN/SAC 等 off-policy | A2C 不擅长经验回放 |

所以，A2C 的适用边界不是“强化学习通用最优解”，而是“在 on-policy、并行环境、同步批处理这三个条件同时成立时，非常合理的基线算法”。

---

## 参考资料

- DI-engine 文档: Advantage Actor-Critic (A2C)  
  https://di-engine-docs.readthedocs.io/en/latest/12_policies/a2c.html

- Next Electronics: Advantage Actor-Critic (A2C) Algorithm  
  https://next.gr/ai/reinforcement-learning/advantage-actor-critic-a2c-algorithm

- skrl 文档: Advantage Actor Critic (A2C)  
  https://skrl.readthedocs.io/en/develop/api/agents/a2c.html

- OpenAI Baselines: ACKTR & A2C  
  https://openai.com/blog/baselines-acktr-a2c/
