## 核心结论

强化学习在大语言模型对齐里，不是“让模型随机试错”，而是把文本生成过程写成一个可优化的 **马尔可夫决策过程**（MDP, Markov Decision Process）。MDP 的直白定义是：在每一步，系统处在某个状态，策略根据状态选择动作，环境给出反馈，并进入下一状态。放到语言模型里：

- 状态 $s_t$：当前 prompt 与已生成前缀
- 动作 $a_t$：下一枚 token
- 策略 $\pi_\theta(a_t\mid s_t)$：模型对下一个 token 的概率分布
- 奖励 $r_t$：对当前动作或整段输出的评分

目标是最大化期望回报：

$$
J(\theta)=\mathbb{E}_{\tau\sim \pi_\theta}\left[\sum_{t=1}^{T} r_t\right]
$$

其中 $\tau=(s_1,a_1,r_1,\dots,s_T,a_T,r_T)$ 表示一条轨迹。对 LLM 来说，一条轨迹通常就是“给定一个 prompt，模型从左到右生成完整回答”的全过程。

如果放到 RLHF（Reinforcement Learning from Human Feedback）里，目标通常还会加入 KL 约束，用来限制新策略不要偏离参考模型太远：

$$
J_{\text{RLHF}}(\theta)=\mathbb{E}_{x,\,y\sim \pi_\theta}\bigl[r_\phi(x,y)\bigr]-\beta\,\mathrm{KL}\!\left(\pi_\theta(\cdot\mid x)\,\|\,\pi_{\text{ref}}(\cdot\mid x)\right)
$$

这里：

- $r_\phi(x,y)$ 是奖励模型对回答 $y$ 的打分
- $\pi_{\text{ref}}$ 是参考模型，通常来自 SFT 或预训练阶段
- $\beta$ 控制“追求高奖励”和“保持原分布稳定”之间的权衡

结论有三条：

| 结论 | 含义 | 为什么重要 |
| --- | --- | --- |
| REINFORCE 能直接优化策略 | 直接用采样得到的回报乘上 $\nabla \log \pi$ 更新参数 | 推导直接，是理解策略梯度的起点 |
| Actor-Critic 是主流稳定化方案 | Actor 负责选动作，Critic 负责估计状态价值 | 用估值构造 baseline，显著降低方差 |
| PPO 是工程上最常用的策略更新器 | 在 Actor-Critic 基础上限制单次策略更新幅度 | 能同时控制训练波动和策略漂移 |

在 LLM 对齐场景里，PPO 之所以常用，不是因为它“理论上更高级”，而是因为它同时解决两个工程问题：

1. 奖励往往只在句尾出现，导致方差高
2. 新策略如果更新过猛，模型会快速偏离参考分布，出现输出风格漂移、长度异常或 reward hacking

---

## 问题定义与边界

先把问题写清楚。强化学习中的 **轨迹**（trajectory）指的是：从初始状态开始，智能体不断观察、决策、获得反馈，直到终止的一整段过程。对文本生成任务，这条轨迹通常可以写成：

- 输入 prompt：$x$
- 生成序列：$y=(a_1,a_2,\dots,a_T)$
- 第 $t$ 步状态：$s_t=(x,a_1,\dots,a_{t-1})$
- 第 $t$ 步动作：$a_t$
- 状态转移：把 $a_t$ 拼接到上下文，形成 $s_{t+1}$

如果把一个回答看成多步决策，那么一个标准的 LLM-RL 任务可以写成：

$$
s_{t+1}=f(s_t,a_t), \qquad a_t \sim \pi_\theta(\cdot\mid s_t)
$$

其中 $f$ 在语言模型里几乎是确定性的：就是把当前 token 追加到前缀末尾。

很多对齐任务采用 **句尾奖励**（terminal reward）：

$$
r_t=
\begin{cases}
0, & t<T \\
r_\phi(x,y), & t=T
\end{cases}
$$

这意味着中间步骤没有直接监督信号，只有整句生成完之后，才知道这条轨迹整体表现如何。这带来一个核心难点：**信用分配**（credit assignment）。

信用分配的直白解释是：最后拿到一个总分之后，应该把这个分数分配给中间哪些动作？比如模型输出“答案是 5”得到奖励 $+1$，并不等于三个 token 的贡献完全相同：

- `答案` 可能只是格式铺垫
- `是` 基本不决定正确性
- `5` 才直接决定答案是否正确

但在句尾奖励设定下，更新时看到的却只是“整句得分”。这就是 RLHF 比普通分类训练更难的原因之一。

下面用一个最小例子说明：

| 项目 | 正确回答轨迹 | 错误回答轨迹 |
| --- | --- | --- |
| Prompt | `把 2 和 3 相加` | `把 2 和 3 相加` |
| 生成 token | `答案` `是` `5` | `答案` `是` `6` |
| 中间奖励 | $0,0$ | $0,0$ |
| 句尾奖励 | $+1$ | $-1$ |
| 难点 | 最后才知道 `5` 是对的 | 最后才知道 `6` 是错的 |

因此，训练目标不是“每一步都做监督学习”，而是“基于整句反馈，反推哪些动作值得提高概率”。

RLHF 常见目标不是单纯最大化奖励，而是：

$$
J_{\text{RLHF}}=\mathbb{E}[r_\phi(x,y)]-\beta \,\mathrm{KL}(\pi_\theta\|\pi_{\text{ref}})
$$

**KL 散度**（Kullback-Leibler divergence）的直白解释是：衡量两个概率分布差多远。它不是距离，但在工程上常被当成“偏离程度”的度量。这里它相当于一个刹车项：

- 奖励项推动模型朝高分回答移动
- KL 项阻止模型为了刷分而偏离参考模型太远

从工程角度看，这个目标比“纯最大化奖励”更符合对齐任务需求，因为奖励模型只覆盖有限偏好，并不等于真实目标本身。

下表对比原始策略梯度目标和 RLHF 常见目标：

| 维度 | 原始策略梯度 | RLHF 常见 PPO 目标 |
| --- | --- | --- |
| 奖励来源 | 环境真实奖励 | 奖励模型 $r_\phi(x,y)$ |
| 奖励时机 | 可逐步，也可终局 | 常见为句尾单次给分 |
| 是否限制旧策略 | 通常不显式限制 | 常加 KL 惩罚或 clip |
| 更新特性 | 理论直接，但波动大 | 更稳，代价是实现更复杂 |
| 典型用途 | 教学、小实验、短轨迹任务 | LLM 对齐、生产训练 |

边界也需要说明。本文讨论的是 **离散动作空间** 下的策略梯度路线，重点是：

- REINFORCE
- Actor-Critic
- PPO

不展开 Q-learning 一类值函数主导方法，原因不是它“错误”，而是语言模型动作空间等于整个词表，通常有上万到数十万个离散动作，直接建模策略更自然，也更符合 LLM 训练接口。

---

## 核心机制与推导

### 1. REINFORCE：最直接的策略梯度

**策略梯度**方法直接优化策略参数 $\theta$，而不是先学动作价值再间接导出策略。最基本目标是：

$$
J(\theta)=\mathbb{E}_{\tau\sim\pi_\theta}[R(\tau)]
$$

其中 $R(\tau)$ 是整条轨迹的总回报。通过对数导数技巧（log-derivative trick）可得：

$$
\nabla_\theta J(\theta)
=
\mathbb{E}_{\tau\sim\pi_\theta}
\left[
R(\tau)\nabla_\theta \log p_\theta(\tau)
\right]
$$

而轨迹概率可以分解为：

$$
p_\theta(\tau)=p(s_1)\prod_{t=1}^{T}\pi_\theta(a_t\mid s_t)\,p(s_{t+1}\mid s_t,a_t)
$$

由于环境转移项通常与 $\theta$ 无关，所以：

$$
\nabla_\theta \log p_\theta(\tau)
=
\sum_{t=1}^{T}\nabla_\theta \log \pi_\theta(a_t\mid s_t)
$$

代入可得 REINFORCE 的经典形式：

$$
\nabla J(\theta)=
\mathbb{E}_{\tau\sim \pi_\theta}
\left[
\sum_{t=1}^{T}\nabla_\theta \log \pi_\theta(a_t\mid s_t)\,G_t
\right]
$$

其中：

$$
G_t=\sum_{k=t}^{T} r_k
$$

表示从时刻 $t$ 开始的回报（return）。

这个公式的直观含义是：

- 如果某条轨迹回报高，就提高该轨迹中已采样动作的概率
- 如果某条轨迹回报低，就降低这些动作的概率

对新手来说，可以把它理解成一句话：

> “先采样一条回答，再根据这条回答最后得分，整体性地鼓励或惩罚其中出现过的动作。”

这也是 REINFORCE 最容易理解、但也最不稳定的地方。因为它直接用采样回报 $G_t$ 乘梯度，导致更新噪声大。尤其在句尾奖励场景里，中间所有 token 都拿到同一个终局信号，训练会很抖。

### 2. 基线：减方差，不改期望

REINFORCE 的一个标准改进是引入 **baseline**：

$$
\nabla J(\theta)=
\mathbb{E}
\left[
\sum_t \nabla_\theta \log \pi_\theta(a_t\mid s_t)\,\bigl(G_t-b(s_t)\bigr)
\right]
$$

其中 $b(s_t)$ 是与动作无关、只依赖状态的参考值。其核心性质是：

$$
\mathbb{E}_{a_t\sim\pi_\theta(\cdot\mid s_t)}
\left[\nabla_\theta \log \pi_\theta(a_t\mid s_t)\,b(s_t)\right]=0
$$

所以加入 baseline 后，梯度期望不变，但方差会下降。

为什么不变？因为：

$$
\sum_a \pi_\theta(a\mid s)\nabla_\theta \log \pi_\theta(a\mid s)
=
\sum_a \nabla_\theta \pi_\theta(a\mid s)
=
\nabla_\theta \sum_a \pi_\theta(a\mid s)
=
\nabla_\theta 1
=
0
$$

这个推导很重要，它说明 baseline 不是“拍脑袋减一个数”，而是有严格数学依据的降方差手段。

如果取：

$$
b(s_t)=V(s_t)
$$

其中 $V(s_t)$ 是状态值函数，那么就得到 **advantage**：

$$
A_t = G_t - V(s_t)
$$

这里：

- $G_t$：这次实际拿到的未来回报
- $V(s_t)$：在状态 $s_t$ 下，平均来说原本预期能拿到的回报
- $A_t$：这次结果相对预期是更好还是更差

于是梯度变成：

$$
\nabla J(\theta)=
\mathbb{E}\left[
\sum_t \nabla_\theta \log \pi_\theta(a_t\mid s_t)\,A_t
\right]
$$

对新手最有帮助的理解方式是：

- $A_t>0$：这次动作比平均水平好，应该提高概率
- $A_t<0$：这次动作比平均水平差，应该降低概率
- $A_t\approx 0$：这次动作和预期差不多，没必要强烈更新

### 3. Actor-Critic：一边选动作，一边估值

Actor-Critic 结构把“做决策”和“做评估”拆开：

- **Actor**：学习策略 $\pi_\theta(a\mid s)$
- **Critic**：学习值函数 $V_\psi(s)$

其中：

- Actor 回答“下一步做什么”
- Critic 回答“当前局面值多少钱”

这个结构在 RLHF 中非常自然，因为语言模型本身就适合做 Actor，而 Critic 可以共享部分表示或单独训练，用来给每个位置提供状态价值估计。

Actor 的常见目标写成最小化损失：

$$
\mathcal{L}_{\text{actor}}
=
-\mathbb{E}\left[\log \pi_\theta(a_t\mid s_t)\,A_t\right]
$$

Critic 则学习拟合回报，最基础形式是最小二乘：

$$
\mathcal{L}_{\text{critic}}
=
\mathbb{E}\left[(V_\psi(s_t)-G_t)^2\right]
$$

如果从 Bellman 观点出发，可以定义一步 TD 误差：

$$
\delta_t = r_t + \gamma V_\psi(s_{t+1}) - V_\psi(s_t)
$$

其中 $\gamma\in(0,1]$ 是折扣因子。对文本任务，$\gamma$ 常取接近 1，因为长回答中后面 token 的贡献也很重要。

TD 误差的直观解释是：

- 左边的 $V_\psi(s_t)$：你当前对状态的估值
- 右边的 $r_t+\gamma V_\psi(s_{t+1})$：根据“一步之后”的信息重新给出的估值
- 两者之差 $\delta_t$：说明你原来的估值偏高还是偏低

这时 advantage 可以用多种方式近似：

| 方式 | 公式 | 特点 |
| --- | --- | --- |
| Monte Carlo | $A_t=G_t-V(s_t)$ | 无偏或近似无偏，但方差大 |
| 1-step TD | $A_t\approx \delta_t$ | 方差低，但偏差更大 |
| GAE | $A_t^{\text{GAE}}=\sum_{l=0}^{\infty}(\gamma\lambda)^l\delta_{t+l}$ | 偏差与方差折中，工程上常用 |

这里的 **GAE**（Generalized Advantage Estimation）可理解为：把多个时间步的 TD 误差按衰减权重累加，得到更平滑的 advantage 估计。它不是本文重点，但在 PPO 工程实现中几乎总会出现。

### 4. PPO：限制更新幅度

Actor-Critic 解决的是“估值”和“降方差”问题，但还没有直接解决“更新过猛”问题。PPO（Proximal Policy Optimization）的作用，就是在策略优化时显式限制单次更新幅度。

定义新旧策略的概率比：

$$
r_t(\theta)=\frac{\pi_\theta(a_t\mid s_t)}{\pi_{\theta_{\text{old}}}(a_t\mid s_t)}
$$

如果没有额外限制，一个很大的 $r_t$ 会使 Actor 一次更新过强。PPO 引入 clip 目标：

$$
\mathcal{L}^{\text{clip}}(\theta)=
\mathbb{E}\left[
\min\left(
r_t(\theta)A_t,\,
\mathrm{clip}(r_t(\theta),1-\epsilon,1+\epsilon)A_t
\right)
\right]
$$

其中 $\epsilon$ 常取 $0.1\sim 0.2$。

它的核心机制可以分两种情况理解：

| 情况 | 解释 | PPO 的处理 |
| --- | --- | --- |
| $A_t>0$ | 该动作比预期好，希望提高概率 | 允许增加，但超过阈值后不再继续放大奖励 |
| $A_t<0$ | 该动作比预期差，希望降低概率 | 允许降低，但超过阈值后不再继续扩大惩罚 |

也就是说，PPO 不会阻止策略变好，但会防止它一步迈得太远。

如果从优化直觉理解：

- REINFORCE：只看“这次得分高不高”
- Actor-Critic：再看“相对平均水平高不高”
- PPO：再额外限制“概率别改得太猛”

这也是为什么 PPO 很适合 LLM 对齐。因为在对齐任务里，“性能提高”不是唯一目标，“分布稳定”同样重要。

### 5. 数值玩具例子

下面用一个具体数值把 REINFORCE、baseline 和 PPO 的差别串起来。

假设某个状态下：

- 旧策略给某 token 的概率：$\pi_{\text{old}}=0.2$
- 新策略当前概率：$\pi_\theta=0.3$
- 从该步开始的回报：$G_t=2$
- Critic 估值：$V(s_t)=1$

于是 advantage 为：

$$
A_t=G_t-V(s_t)=2-1=1
$$

如果采用最原始的 REINFORCE，梯度中的权重部分可近似理解为：

$$
\nabla \log \pi \cdot G_t
\approx
\frac{1}{0.3}\times 2
=
6.67
$$

如果改成带 baseline 的形式：

$$
\nabla \log \pi \cdot A_t
\approx
\frac{1}{0.3}\times 1
=
3.33
$$

也就是说，仅仅减去一个合理的参考线，更新强度就下降了一半，方差通常也会更小。

再看 PPO。概率比是：

$$
r_t=\frac{0.3}{0.2}=1.5
$$

若 $\epsilon=0.2$，则允许区间是：

$$
[1-\epsilon,1+\epsilon]=[0.8,1.2]
$$

所以 clip 后得到：

$$
\mathrm{clip}(r_t,0.8,1.2)=1.2
$$

最终 PPO 用于优化的目标项，不再按 $1.5\times A_t$ 继续增长，而会被截到：

$$
1.2\times A_t = 1.2
$$

这说明 PPO 的稳定性来自一个非常具体的机制：它不相信单个 batch 给出的“过强更新信号”，而是人为限制策略一步最多改变多少。

下面给出一个从采样到更新的简化流程：

$$
\text{采样轨迹}
\rightarrow
\text{计算 }G_t
\rightarrow
\text{Critic 估计 }V(s_t)
\rightarrow
A_t=G_t-V(s_t)
\rightarrow
\text{计算 }r_t
\rightarrow
\text{PPO clip 后更新 Actor}
$$

这条链路对应的意义分别是：

| 环节 | 作用 |
| --- | --- |
| 采样轨迹 | 从当前策略真实采样回答 |
| 计算回报 | 把奖励写成可训练信号 |
| Critic 估值 | 提供 baseline，降低方差 |
| 计算 advantage | 转成“比预期好多少”的信号 |
| 计算 ratio | 比较新旧策略变化幅度 |
| clip 更新 | 防止一步更新过大 |

### 6. 真实工程例子：LLM 的 RLHF

以 prompt `解释 TCP 三次握手` 为例，一个典型 RLHF 训练样本会经历以下步骤：

1. Actor 根据 prompt 生成完整回答 $y$
2. 奖励模型 $r_\phi(x,y)$ 给出整段打分
3. Critic 为每个位置估计状态值 $V(s_t)$
4. 根据回报和估值得到 $A_t$
5. 用 PPO 目标更新 Actor
6. 同时用回报拟合 Critic
7. 再加上 KL 项，限制偏离参考模型

更具体地说，假设模型生成：

- 第 1 段定义正确
- 第 2 段解释清晰
- 第 3 段把 SYN/ACK 顺序说错

最终奖励模型可能只返回一个整体分数，比如 `0.62`。此时训练系统仍然要把这个全局分数拆成每个位置的学习信号。Critic 的作用不是“判断内容对错”，而是近似估计：

> “在生成到当前位置时，未来大概还能拿到多少总分？”

如果某一步生成后，实际结果明显好于 Critic 预期，则该步 advantage 为正，Actor 就会提高对应动作的概率；反之则降低。

这解释了 RLHF 的核心难点：它不是逐 token 人工监督，而是“整段偏好评分 + 中间价值估计 + 受限策略更新”的组合。

---

## 代码实现

下面给出一个可直接运行的 Python 玩具实现，演示 PPO 中最核心的四个量：

- `returns`
- `values`
- `advantages`
- `clip objective`

这段代码只依赖 Python 标准库，可以直接运行。它不是完整训练器，但和 PPO 核心公式一致。

```python
import math


def compute_returns(rewards, gamma=1.0):
    """从后往前计算每一步 return: G_t = r_t + gamma * G_{t+1}"""
    returns = [0.0] * len(rewards)
    running = 0.0
    for i in range(len(rewards) - 1, -1, -1):
        running = rewards[i] + gamma * running
        returns[i] = running
    return returns


def compute_advantages(returns, values):
    """A_t = G_t - V(s_t)"""
    if len(returns) != len(values):
        raise ValueError("returns and values must have the same length")
    return [ret - val for ret, val in zip(returns, values)]


def ppo_clipped_objective(old_prob, new_prob, advantage, eps=0.2):
    """单样本 PPO clip objective"""
    if old_prob <= 0.0 or new_prob <= 0.0:
        raise ValueError("probabilities must be positive")

    ratio = new_prob / old_prob
    clipped_ratio = min(max(ratio, 1.0 - eps), 1.0 + eps)
    unclipped = ratio * advantage
    clipped = clipped_ratio * advantage
    return min(unclipped, clipped), ratio, clipped_ratio


def main():
    # 玩具轨迹：前两步没有奖励，句尾给 +2
    rewards = [0.0, 0.0, 2.0]
    returns = compute_returns(rewards, gamma=1.0)

    # 假设 Critic 对每一步状态都估值 1.0
    values = [1.0, 1.0, 1.0]
    advantages = compute_advantages(returns, values)

    # old policy 和 new policy 在某一步对已采样动作的概率
    old_prob = 0.2
    new_prob = 0.3

    obj, ratio, clipped_ratio = ppo_clipped_objective(
        old_prob=old_prob,
        new_prob=new_prob,
        advantage=advantages[0],
        eps=0.2,
    )

    assert returns == [2.0, 2.0, 2.0]
    assert advantages == [1.0, 1.0, 1.0]
    assert math.isclose(ratio, 1.5, rel_tol=1e-9)
    assert math.isclose(clipped_ratio, 1.2, rel_tol=1e-9)
    assert math.isclose(obj, 1.2, rel_tol=1e-9)

    print("rewards      =", rewards)
    print("returns      =", returns)
    print("values       =", values)
    print("advantages   =", advantages)
    print("ratio        =", ratio)
    print("clipped_ratio=", clipped_ratio)
    print("ppo_objective=", obj)


if __name__ == "__main__":
    main()
```

运行结果会是：

```text
rewards      = [0.0, 0.0, 2.0]
returns      = [2.0, 2.0, 2.0]
values       = [1.0, 1.0, 1.0]
advantages   = [1.0, 1.0, 1.0]
ratio        = 1.4999999999999998
clipped_ratio= 1.2
ppo_objective= 1.2
```

这段代码能说明三件事：

1. 句尾奖励会回传到整条轨迹，因此前面步骤的 return 也可能相同
2. advantage 本质上是 “实际回报 - 预期回报”
3. PPO 的 clip 会把过大的概率变化截断

如果写成更接近训练框架的伪代码，结构通常是：

```python
for batch in rollout_buffer:
    returns = discounted_sum(batch.rewards, gamma)
    values = critic(batch.states)
    advantages = returns - values.detach()

    new_log_prob = actor.log_prob(batch.states, batch.actions)
    ratio = exp(new_log_prob - batch.old_log_prob)
    clipped_ratio = clip(ratio, 1 - eps, 1 + eps)

    actor_loss = -mean(min(ratio * advantages, clipped_ratio * advantages))
    critic_loss = mse_loss(values, returns)

    loss = actor_loss + c1 * critic_loss + beta * kl_penalty
    optimize(loss)
```

这段伪代码和 LLM 对齐的映射关系如下：

| 训练量 | 在普通 RL 中的含义 | 在 LLM-RLHF 中的对应物 |
| --- | --- | --- |
| `state` | 当前环境状态 | 当前 prompt + 已生成前缀 |
| `action` | 当前采取的动作 | 当前采样出的 token |
| `old_log_prob` | 旧策略对该动作的对数概率 | rollout 时模型记录的 token logprob |
| `returns` | 从该步开始的未来回报 | 由句尾 reward 或 shaped reward 传播而来 |
| `values` | Critic 的状态估值 | 对当前前缀后续质量的预测 |
| `advantages` | 相对基线的好坏信号 | “这个 token 选择是否优于预期” |
| `kl_penalty` | 与参考分布的偏离惩罚 | 限制模型不要快速漂移 |

对新手尤其容易混淆的点有两个：

- PPO 不是替代策略梯度，而是对策略梯度加上“受限更新”
- Actor-Critic 不是否定 REINFORCE，而是给 REINFORCE 增加一个可学习 baseline

这两者都属于同一条方法链上的稳定化改造。

---

## 工程权衡与常见坑

理论公式并不长，真正的难点在工程稳定性。LLM 对齐训练里最常见的问题，通常不是“公式写错”，而是“信号太噪、约束不够、估值不准”。

第一类问题是 **高方差**。在句尾奖励设定下，同一个 prompt 采样两次，得到的整句 reward 可能差异很大。于是同样的状态，更新方向会剧烈波动。这会导致：

- actor loss 大幅抖动
- 收敛慢
- 对 batch size 和采样质量非常敏感

常见缓解手段包括：

- 使用 baseline 或 Critic
- 使用 GAE
- 增大 batch
- 做 reward normalization

第二类问题是 **Critic 太弱**。Critic 如果长期估值偏高或偏低，advantage 就会系统性失真。表面上看是 Actor 学坏了，实际上往往是“参考线错了”。例如：

- Critic 总是高估价值，则很多本来不错的动作会被判成“低于预期”
- Critic 总是低估价值，则大量普通动作都会被误判成“超预期”

第三类问题是 **策略漂移**。奖励模型不是完美真理，只是偏好的近似代理。如果只追 reward，模型可能学到的是“更会骗奖励模型”，而不是真正更有帮助。这就是 **reward hacking**。

reward hacking 的直白解释是：

> 模型没有真正完成任务，而是利用了评分器的漏洞，让分数看起来更高。

例如一个奖励模型偏好“语气礼貌、篇幅较长、结构完整”，那么 Actor 可能学会统一输出：

- 冗长模板
- 过度免责声明
- 表面结构漂亮但信息密度下降的回答

此时训练日志上 reward 上升，但人工评估未必同步提升。

下表总结常见问题与处理思路：

| 问题 | 典型表现 | 常见原因 | 缓解手段 |
| --- | --- | --- | --- |
| REINFORCE 方差大 | loss 抖动、收敛慢 | 终局奖励稀疏、采样噪声大 | baseline、Critic、GAE、增大 batch |
| Critic 不稳定 | advantage 噪声大、Actor 学偏 | 值函数欠拟合或过拟合 | 监控 critic loss、调学习率、单独验证 value head |
| PPO clip 过大 | KL 飙升、回答分布突变 | $\epsilon$ 太大、更新步数过多 | 减小 $\epsilon$、减少 epoch、加 KL 惩罚 |
| PPO clip 过小 | 几乎学不动 | 限制太严 | 略增 $\epsilon$、检查 reward 尺度 |
| 奖励尺度失控 | 梯度爆炸或过小 | reward 范围波动大 | reward normalization、裁剪极值 |
| reward hacking | reward 上升但人工质量下降 | 奖励模型被利用 | 加 KL、人工抽检、重训奖励模型 |

一个真实工程场景是“回答质量优化”。如果奖励模型偏好“更详细、更礼貌”的回答，而没有对事实准确性给予足够权重，那么 PPO 训练后模型可能趋向于：

- 回答更长
- 礼貌模板更多
- 结构更整齐
- 事实错误反而不一定减少

这时需要同时监控：

- 参考模型 KL
- 回答长度分布
- 人工抽样评分
- 准确率或任务成功率
- 奖励模型分数和人工分数的一致性

还有一个很常见的误区，是把“低方差”和“低偏差”混为一谈。

- Critic 能降低方差
- 但 Critic 自己是近似模型，会引入偏差

因此工程目标不是追求“完全无偏”，而是追求“总误差更可控”。在大规模训练中，一个略有偏差但稳定的 advantage 估计，通常比完全无偏但极其抖动的回报信号更有用。

---

## 替代方案与适用边界

REINFORCE、Actor-Critic 和 PPO 可以看成一条逐步稳定化的演化链：

| 方法 | 方差 | 稳定性 | 工程复杂度 | 适用场景 |
| --- | --- | --- | --- | --- |
| REINFORCE | 高 | 低 | 低 | 教学、短轨迹、小实验 |
| Actor-Critic | 中 | 中到高 | 中 | 通用 RL、句尾奖励任务 |
| PPO | 低到中 | 高 | 中到高 | LLM 对齐、生产训练 |

它们各自更适合的场景可以说得更具体一些。

**REINFORCE** 适合：

- 想验证一个目标是否可学
- 轨迹较短
- 奖励较稠密
- 需要先建立数学直觉

它的优点是实现最简单，推导也最直接。缺点是长轨迹和稀疏奖励下很容易变得不可用。

**Actor-Critic** 适合：

- 已确认任务可学
- 但 REINFORCE 波动太大
- 有能力同时训练一个值函数
- 需要在训练成本和稳定性之间折中

它比 REINFORCE 多了一个 Critic，但这个额外复杂度通常是值得的，尤其是在句尾奖励任务里。

**PPO** 适合：

- 新旧策略不能偏离太快
- 奖励模型不完全可靠
- 输出分布漂移代价高
- 需要更稳定的生产级训练流程

这正是 RLHF 的典型条件，所以 PPO 成为主流并不意外。

但它们都有边界，不应被绝对化：

- 如果奖励在每个 token 上都能准确给出，问题会简单很多，未必非要 PPO
- 如果动作空间很小、状态转移清晰，值函数方法或搜索方法可能更合适
- 如果奖励模型本身质量很差，再好的 PPO 也只是更稳定地优化错误目标
- 如果没有能力把 Critic 训练稳，Actor-Critic 或 PPO 的收益会被大幅削弱

可以把算法选择归结为四个判断问题：

1. 奖励是稠密还是稀疏
2. 轨迹是短还是长
3. 策略漂移代价大不大
4. 能否稳定训练一个足够可靠的 Critic

对 RLHF 而言，这四个问题的答案通常是：

- 奖励偏稀疏
- 轨迹偏长
- 漂移代价很大
- 必须训练 Critic

因此，PPO + Actor-Critic 成为大模型对齐中的常见组合，不是偶然，而是由任务结构决定的。

---

## 参考资料

1. Stanford MLHP Chapter 4: RLHF 与 $J_{\text{RLHF}}$ 目标，https://mlhp.stanford.edu/src/chap4.html  
2. Johal, Policy Gradients, Actor-Critic, PPO 综述，https://johal.in/policy-gradients-python-trpo-ppo-actor-critic-variants-2026/  
3. OpenRLHF 文档与 PPO 实现说明，可参考项目文档与训练脚本设计思路  
4. Richard S. Sutton, Andrew G. Barto, *Reinforcement Learning: An Introduction*，策略梯度、baseline、Actor-Critic 与 TD 基础  
5. Schulman et al., *Proximal Policy Optimization Algorithms*，PPO 的 clip 目标与工程动机  
6. Schulman et al., *High-Dimensional Continuous Control Using Generalized Advantage Estimation*，GAE 的偏差-方差折中思路
