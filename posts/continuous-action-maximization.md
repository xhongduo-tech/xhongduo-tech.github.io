## 核心结论

连续动作空间里的关键难点不是“如何计算 $Q_\theta(s,a)$”，而是“如何找到让它最大的动作 $a$”。这里的 $Q$ 函数可以理解为“在状态 $s$ 下，执行动作 $a$ 的长期收益评分器”。

离散动作里，最大化写成
$$
a^*(s)=\arg\max_a Q_\theta(s,a)
$$
通常可以直接枚举，因为动作个数有限，比如左、右、跳、蹲四个动作都算一遍就行。连续动作里，$a$ 来自区间或高维实数空间，例如 $a\in[-1,1]^d$，动作点是无限多个，不能逐个试。

因此，连续动作空间下的 Q 函数最大化，本质上只有三条路：

| 方法 | 核心思路 | 代表算法 | 优点 | 主要限制 |
|---|---|---|---|---|
| 解析最大化 | 把 $Q(s,a)$ 设计成对 $a$ 可直接求极值 | NAF | 计算快，推导清晰 | 结构限制强，通常只适合单峰 |
| Actor 近似最大化 | 训练一个策略网络直接输出近似最优动作 | DDPG、TD3 | 适合高维连续控制 | 训练可能不稳定，依赖 Critic 质量 |
| 采样搜索近似最大化 | 采样候选动作，筛选高分动作，迭代逼近最优 | CEM | 不要求可导，可当黑盒优化 | 高维采样成本高 |
| 软最大化 | 不追求唯一最优点，而是权衡高回报与随机性 | SAC | 稳定性通常更好，探索更自然 | 实现更复杂，含熵项调节 |

对初学者来说，可以先把离散动作理解成“在有限菜单里选最高分”，把连续动作理解成“在一条连续曲线上找峰值”。前者靠遍历，后者必须靠结构、梯度或搜索。

---

## 问题定义与边界

先统一记号。

- $s$：状态，表示环境当前信息，比如机械臂关节角度、速度、目标位置。
- $a$：动作，表示系统要执行的控制量，比如扭矩、油门、转向角。
- $Q_\theta(s,a)$：参数为 $\theta$ 的评论家网络，输入状态和动作，输出长期收益估计。
- $\pi_\phi(a|s)$：参数为 $\phi$ 的策略，给定状态输出动作分布或直接输出动作。

本文只讨论一个子问题：在连续动作空间下，如何近似求
$$
a^*(s)=\arg\max_{a\in [a_{\min},a_{\max}]} Q_\theta(s,a)
$$
也就是“给定状态后，如何从连续动作里找到最高 Q 的动作”。

不展开以下内容：

- 纯离散动作的 DQN 类方法
- 模型预测控制（MPC）中的长时域规划
- 蒙特卡洛树搜索、轨迹优化等更广义控制方法
- 强化学习全部训练细节，如回放缓冲区、目标网络初始化、奖励设计

先看一个玩具例子。设 1 维动作范围是 $a\in[0,3]$，并且
$$
Q(a)=5-(a-1.2)^2
$$
它表示一个开口向下的抛物线，最高点在 $a=1.2$。如果你只能枚举离散点 $\{0,1,2,3\}$，得到的最好动作是 $a=1$，但它并不是真正最优。

| 场景 | 搜索空间 | 最优动作获取方式 | 结果特点 |
|---|---|---|---|
| 离散动作 | 有限集合 | 可直接枚举 | 精确、简单 |
| 连续动作 | 实数区间或高维向量 | 必须做优化或近似 | 更灵活，也更难 |

这就是连续控制和离散控制在动作选择环节上的根本差异。

---

## 核心机制与推导

### 1. NAF：把 $Q$ 设计成可解析最大化

NAF 的全称是 Normalized Advantage Functions。它的核心做法不是“更聪明地搜索动作”，而是“把 $Q$ 函数写成一个对动作可直接求峰值的形式”。

它常见写法是：
$$
Q_\theta(s,a)=V_\theta(s)-\frac12(a-\mu_\theta(s))^T P_\theta(s)(a-\mu_\theta(s))
$$

这里：

- $V_\theta(s)$ 是状态值，只看状态的整体好坏。
- $\mu_\theta(s)$ 是网络输出的中心动作，可以理解为峰值位置。
- $P_\theta(s)\succeq 0$ 表示半正定矩阵，白话说就是这个二次项总是非负，因此前面的负号保证它是“向下开的碗”。

因为
$$
(a-\mu)^T P (a-\mu)\ge 0
$$
所以
$$
Q_\theta(s,a)\le V_\theta(s)
$$
等号在 $a=\mu_\theta(s)$ 时成立。于是
$$
\arg\max_a Q_\theta(s,a)=\mu_\theta(s)
$$

这就是“解析最大化”：最大动作不用搜索，网络直接给出。

这种方法的优点是快，缺点也直接。它默认 $Q(s,a)$ 对动作像一个单峰二次函数，现实里很多任务的动作价值面是多峰、扭曲、非对称的，NAF 就会受限。

### 2. DDPG 与 TD3：训练 Actor 近似 $\arg\max$

DDPG 的思路更通用。既然每次直接解
$$
\arg\max_a Q_\theta(s,a)
$$
太难，那就训练一个 Actor 网络 $\pi_\phi(s)$，让它直接输出近似最优动作：
$$
a \approx \pi_\phi(s)
$$

Actor 的训练目标是让自己输出的动作在 Critic 看来分数更高，即最大化
$$
J_{\text{actor}}(\phi)=\mathbb E_s[Q_\theta(s,\pi_\phi(s))]
$$

对参数做梯度上升，相当于让 Actor 学会“给每个状态报一个高 Q 动作”。因此 DDPG 本质上是在用一个可微函数逼近 $\arg\max_a Q(s,a)$。

问题在于，如果 Critic 高估了某些动作，Actor 会被错误信号带偏，学出“看起来高分、实际上很差”的动作。

TD3 针对这个问题加了三个关键修正，其中最核心的是双评论家。它同时训练两个 Critic：
$$
Q_{\theta_1}(s,a),\quad Q_{\theta_2}(s,a)
$$
在构造目标时取较小值：
$$
y=r+\gamma \min_{i=1,2} Q_{\theta_i^-}(s',a')
$$
这样做的含义很直接：不要轻信任何一个 Critic 过高的判断，宁可保守一点。TD3 还会延迟 Actor 更新，让 Critic 先学稳，再让 Actor 跟随。

### 3. SAC：软最大化而不是硬最大化

SAC 不是简单求“最高 Q 动作”，而是求“高 Q 且保留随机性”的策略。这里的熵可以理解为“动作分布的分散程度”，熵高说明策略不那么死板。

SAC 的策略目标常写成：
$$
J_\pi(\phi)=\mathbb E_{s\sim \mathcal D,a\sim \pi_\phi}\left[\alpha \log \pi_\phi(a|s)-Q_\theta(s,a)\right]
$$

最小化这个目标，等价于同时追求两件事：

- 让 $Q(s,a)$ 高
- 让策略保留一定随机性，避免过早收缩到单点

从“最大化 Q”的角度看，SAC 做的是软化版本：
$$
\max_a Q(s,a)
\quad \Rightarrow \quad
\text{高 }Q + \text{高熵}
$$

SAC 的关键工程点是重参数化。它不是直接从分布里不可微地抽样动作，而是写成
$$
a=f_\phi(\epsilon;s),\quad \epsilon\sim \mathcal N(0,I)
$$
常见实现是先输出高斯变量 $u$，再做
$$
a=\tanh(u)
$$
把动作压到合法边界内。这样采样过程就能参与反向传播，Actor 可以直接根据 Critic 的梯度更新。

### 4. CEM：用采样搜索逼近峰值

CEM 的全称是 Cross-Entropy Method。它不依赖 Actor，也不要求对动作可解析求导，可以把 Critic 当成黑盒评分器。

单步动作优化的流程通常是：

1. 从一个分布中采样一批动作 $a_1,\dots,a_N$
2. 计算每个动作的 $Q(s,a_i)$
3. 选出前 $k$ 个高分动作，称为精英样本
4. 用这些精英样本重新拟合采样分布
5. 重复多轮，直到分布集中到高分区域

它的直觉是：先粗撒网，再逐轮把采样中心拉向高 Q 区域。

### 5. 从机制角度统一理解

| 路线 | “怎么找到高 Q 动作” | 本质 |
|---|---|---|
| NAF | 直接由数学结构给出峰值 | 解析法 |
| DDPG / TD3 | 让 Actor 学会输出峰值附近动作 | 梯度法 |
| SAC | 学一个偏向高 Q 且保留熵的分布 | 软梯度法 |
| CEM | 反复采样和筛选高分动作 | 采样法 |

真实工程例子是机械臂扭矩控制。状态 $s$ 可能包含 7 个关节角、角速度、末端误差；动作 $a$ 是 7 维连续扭矩。如果对每个维度只离散成 21 个候选值，总动作数就是 $21^7$，完全不可枚举。这时只能靠 Actor、结构化 $Q$ 或采样搜索。

---

## 代码实现

下面先给一个最小玩具例子，再给一个接近工程实现的 PyTorch 风格代码。

### 1. 玩具例子：连续最优点与离散枚举的差别

```python
def q(a: float) -> float:
    return 5.0 - (a - 1.2) ** 2

# 连续情形下，解析最优点
a_star = 1.2
assert abs(q(a_star) - 5.0) < 1e-12

# 粗糙离散化
grid = [0.0, 1.0, 2.0, 3.0]
best_grid_a = max(grid, key=q)

assert best_grid_a == 1.0
assert q(best_grid_a) < q(a_star)

# NAF 风格：如果 mu(s)=1.2，那么它直接就是 argmax
mu = 1.2
assert abs(mu - a_star) < 1e-12
```

这段代码说明三件事：

- 连续最优点可能不在离散网格上。
- 粗暴离散化会丢失最优值。
- 如果模型结构保证峰值在 $\mu(s)$，那就不需要搜索。

### 2. 一个统一的动作输出接口

从系统接口看，给定状态 `state`，不同方法的推理过程可以统一成：

```python
def select_action(state, method, actor=None, naf_mu=None, cem_search=None):
    if method == "naf":
        return naf_mu(state)          # 直接输出 mu(s)
    if method in {"ddpg", "td3"}:
        return actor(state)           # 直接输出确定性动作
    if method == "sac":
        return actor.sample(state)    # 从策略分布采样，并做 tanh 限幅
    if method == "cem":
        return cem_search(state)      # 采样、筛选、重拟合
    raise ValueError("unknown method")
```

这背后的共同目标都是：给当前状态产出一个近似高 Q 的动作。

### 3. 简化的 PyTorch 风格训练循环

下面代码不依赖完整环境，重点展示张量流向和优化目标。术语说明：

- `critic`：评论家网络，估计 $Q(s,a)$。
- `actor`：策略网络，输出动作。
- `target_*`：目标网络，用于构造更稳定的训练目标。

```python
import torch
import torch.nn.functional as F

def soft_update(target, source, tau=0.005):
    for tp, sp in zip(target.parameters(), source.parameters()):
        tp.data.mul_(1 - tau).add_(tau * sp.data)

def td3_style_update(batch, actor, critic1, critic2, target_actor, target_critic1, target_critic2,
                     actor_opt, critic1_opt, critic2_opt, gamma=0.99, policy_noise=0.2,
                     noise_clip=0.5, action_limit=1.0, actor_update=False):
    s, a, r, s2, d = batch

    with torch.no_grad():
        noise = torch.randn_like(a) * policy_noise
        noise = noise.clamp(-noise_clip, noise_clip)

        next_a = target_actor(s2) + noise
        next_a = next_a.clamp(-action_limit, action_limit)

        target_q1 = target_critic1(s2, next_a)
        target_q2 = target_critic2(s2, next_a)
        target_q = torch.min(target_q1, target_q2)
        y = r + gamma * (1.0 - d) * target_q

    q1 = critic1(s, a)
    q2 = critic2(s, a)

    critic1_loss = F.mse_loss(q1, y)
    critic2_loss = F.mse_loss(q2, y)

    critic1_opt.zero_grad()
    critic1_loss.backward()
    critic1_opt.step()

    critic2_opt.zero_grad()
    critic2_loss.backward()
    critic2_opt.step()

    actor_loss_value = None
    if actor_update:
        pred_a = actor(s)
        actor_loss = -critic1(s, pred_a).mean()

        actor_opt.zero_grad()
        actor_loss.backward()
        actor_opt.step()
        actor_loss_value = actor_loss.item()

        soft_update(target_actor, actor)
        soft_update(target_critic1, critic1)
        soft_update(target_critic2, critic2)

    return {
        "critic1_loss": critic1_loss.item(),
        "critic2_loss": critic2_loss.item(),
        "actor_loss": actor_loss_value,
    }

# 这些 assert 只验证目标函数形状是否符合预期
x = torch.tensor([[1.0], [2.0]])
assert x.shape == (2, 1)
```

如果换成 SAC，关键差异在 Actor 不再直接输出一个确定性动作，而是输出分布参数，例如均值和对数方差，再用重参数化采样：

$$
u=\mu_\phi(s)+\sigma_\phi(s)\odot \epsilon,\quad \epsilon\sim\mathcal N(0,I),\quad a=\tanh(u)
$$

这样做的好处是动作有探索性，而且采样路径可微。

### 4. CEM 的最小伪代码

```text
输入: 状态 s, 初始分布 N(mu, sigma), 迭代轮数 T
for t in 1..T:
    采样 N 个动作 a_i ~ N(mu, sigma)
    裁剪到动作边界
    计算分数 q_i = Q(s, a_i)
    选前 k 个精英动作
    用精英动作的均值和方差更新 mu, sigma
输出: 最终 mu 或精英中分数最高的动作
```

如果 Critic 已训练得足够好，CEM 可以直接拿来做“给定状态的动作优化器”。

---

## 工程权衡与常见坑

连续动作算法最常见的问题，不是公式写错，而是“动作范围、价值偏差、搜索成本”三个层面没处理好。

### 1. 常见坑总览

| 方法 | 常见坑 | 症状 | 处理方向 |
|---|---|---|---|
| NAF | 单峰假设过强 | 学不到多模态动作 | 仅在低维、近似二次任务尝试 |
| DDPG | Critic 高估带偏 Actor | 动作越来越极端，回报不稳定 | 双 Critic、目标网络、噪声控制 |
| TD3 | 去掉双 Critic 或延迟更新 | 训练前期正常，后期快速发散 | 保留 TD3 三件套 |
| SAC | `tanh` 边界附近梯度变小 | 动作总贴边，更新变慢 | 做好动作缩放，监控分布方差 |
| CEM | 高维采样成本爆炸 | 推理慢、抖动大 | 降低维度、减少迭代、批量评估 |

### 2. 工程排查顺序

| 排查项 | 为什么先看它 | 典型现象 |
|---|---|---|
| 动作缩放 | 输入输出尺度错了，后面全会偏 | 动作总贴近边界 |
| Q 过估计 | Actor 会追错误峰值 | 训练回报先升后崩 |
| 目标网络 | 目标抖动会放大自举误差 | Loss 震荡大 |
| 动作边界处理 | 非法动作或裁剪不一致会破坏学习 | 仿真和部署表现不一致 |
| 搜索成本 | 方法可能理论可行但实时性不够 | 控制频率达不到要求 |

### 3. 一个真实工程例子

机械臂扭矩控制里，每个关节都有最大扭矩上限。如果你把 Actor 输出写成 `tanh` 后的 $[-1,1]$，但环境实际期望输入是 $[-200,200]$ 牛米，而你忘了做线性缩放，系统就会把极小动作当成正常控制。相反，如果缩放写反了，Actor 可能几乎每步都打满扭矩，表面上像“积极探索”，实际只是数值错位。

这种问题看日志不一定显眼，但从动作直方图上通常能看出来：分布长期堆在边界，或几乎全在零附近。

### 4. “症状 -> 原因 -> 处理”表

| 症状 | 常见原因 | 处理方式 |
|---|---|---|
| Actor 输出几乎恒定 | Critic 没学到区分度 | 先检查 Q 目标和回放数据 |
| 动作总贴边界 | 缩放错误或过估计 | 检查动作映射和 target Q |
| TD3 仍高估 | 双 Critic 结构被弱化 | 确认用了 `min(Q1,Q2)` |
| SAC 温度过大 | 熵项压过 Q 项 | 调整或自动学习 $\alpha$ |
| CEM 抖动大 | 精英样本太少 | 增大样本数或加平滑更新 |

工程上一个很实用的判断标准是：如果你的主要问题是“如何快速得到一个稳定连续策略”，优先看 TD3 或 SAC；如果问题是“已有评分器，想做动作黑盒优化”，CEM 更直接；如果动作维度低、结构简单、希望推导透明，NAF 才有优势。

---

## 替代方案与适用边界

没有哪一种方法是连续动作 Q 最大化的通用答案。选型时应该先看动作维度、实时性要求、Q 面形状，以及你是否需要随机探索。

### 1. 方法对比

| 方法 | 适合什么任务 | 优点 | 不适合什么情况 |
|---|---|---|---|
| NAF | 低维、单峰、近似二次的动作价值面 | 可解析求最大，推理快 | 多峰复杂控制 |
| DDPG | 需要确定性连续策略的基础场景 | 结构简单，易于理解 | 容易不稳定 |
| TD3 | 大多数通用连续控制基线 | 比 DDPG 更稳，抑制高估 | 仍依赖精细调参 |
| SAC | 需要更强探索与稳定性的场景 | 稳定性通常更好 | 实现复杂度更高 |
| CEM | 有现成评分器、可批量评估动作 | 不要求可导 | 高维实时控制成本高 |

### 2. 如何选型

可以按下面的决策顺序判断：

1. 如果动作维度低，且你明确知道 $Q(s,a)$ 对动作近似单峰，先考虑 NAF。
2. 如果要做通用连续控制基线，先从 TD3 开始。
3. 如果环境探索难、局部最优多、希望策略保留随机性，优先 SAC。
4. 如果你不想训练 Actor，或者已有一个可调用的评分器，考虑 CEM。
5. 如果系统必须高频实时输出动作，优先 Actor 类方法，因为一次前向传播通常比多轮采样快。

### 3. 为什么多峰时 NAF 不合适

假设某个状态下，$Q(s,a)$ 在动作空间里有两个高峰，例如一个动作对应“向左绕开障碍”，另一个动作对应“向右绕开障碍”，两者都能成功但路径不同。NAF 的二次型本质上只会表达一个中心峰值，它会把复杂结构压成一个“单峰碗形”，结果可能学出一个位于中间、实际不可行的折中动作。

而 TD3、SAC、CEM 不要求 $Q$ 必须长成单峰二次型，所以更能适应复杂动作价值面。

### 4. 读者可以怎么理解这些方法的边界

- NAF 解决的是“能不能把最大化问题做成闭式解”。
- DDPG / TD3 解决的是“能不能学一个函数替代每次在线求最大值”。
- SAC 解决的是“能不能在高 Q 和足够探索之间做平衡”。
- CEM 解决的是“如果我不想依赖可导结构，能不能直接搜索高分动作”。

这几类方法不是互斥的思想流派，而是在回答同一个核心问题：连续动作里，谁来近似那个 $\arg\max_a Q(s,a)$。

---

## 参考资料

1. [Continuous Deep Q-Learning with Model-based Acceleration (NAF)](https://proceedings.mlr.press/v48/gu16.html)
2. [Continuous Control with Deep Reinforcement Learning (DDPG)](https://arxiv.org/abs/1509.02971)
3. [Addressing Function Approximation Error in Actor-Critic Methods (TD3)](https://proceedings.mlr.press/v80/fujimoto18a.html)
4. [Soft Actor-Critic: Off-Policy Maximum Entropy Deep Reinforcement Learning with a Stochastic Actor](https://proceedings.mlr.press/v80/haarnoja18b.html)
5. [The Cross-Entropy Method for Optimization](https://www.sciencedirect.com/science/article/pii/B9780444538598000035)
