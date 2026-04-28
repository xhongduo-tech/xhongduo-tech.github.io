## 核心结论

`A3C`（Asynchronous Advantage Actor-Critic，异步优势演员-评论家）和 `A2C`（Advantage Actor-Critic，优势演员-评论家）讨论的是**同一类 on-policy actor-critic 方法的两种训练组织方式**。白话讲，`actor-critic` 是“一部分负责做动作决策，另一部分负责给这个决策打分”的结构；`on-policy` 是“只能用当前策略刚采到的数据来更新”。

两者最核心的差别只有一个：`A3C` 异步更新，`A2C` 同步更新。`A3C` 让多个 actor 各自与环境交互，谁先算完梯度谁先把梯度写回全局参数；`A2C` 则要求所有 actor 先用同一版参数收集完一批样本，再统一做一次批量更新。

对工程实践来说，结论通常很直接：如果目标是**稳定训练、方便复现、吃满 GPU、降低实现复杂度**，优先考虑 `A2C + 多环境并行`；如果目标是理解异步优化思想，或者研究历史上的分布式强化学习实现，再看 `A3C` 更合适。很多现代实现已经不再把 `A3C` 当作默认首选，而是转向同步批量更新的 `A2C`，再进一步发展到 `PPO`。

| 维度 | A3C | A2C |
|---|---|---|
| 更新方式 | 多 actor 异步写回 | 多 actor 同步聚合后更新 |
| 参数一致性 | 弱，可能使用陈旧参数 | 强，同一轮使用同一版参数 |
| GPU 利用率 | 通常较低 | 通常较高 |
| 复现性 | 较差 | 较好 |
| 实现复杂度 | 高，需要并发控制 | 低，更接近标准 batch 训练 |

---

## 问题定义与边界

这篇文章只比较 `A2C` 和 `A3C` 在 **on-policy actor-critic** 框架里的差异，不讨论所有强化学习算法。也就是说，我们关心的是：同样都在估计策略 `\pi_\theta(a|s)` 和价值函数 `V_\theta(s)`，为什么一个同步，一个异步，以及这会带来什么后果。

先把基本术语摆清楚：

| 术语 | 含义 | 白话解释 |
|---|---|---|
| `state s_t` | 时刻 `t` 的状态 | 环境此刻长什么样 |
| `action a_t` | 时刻 `t` 的动作 | 智能体现在做什么 |
| `reward r_t` | 即时奖励 | 这一步拿了多少分 |
| `policy \pi_\theta(a|s)` | 策略函数 | 给定状态时动作的概率分布 |
| `value V_\theta(s)` | 状态价值函数 | 站在当前状态，未来总回报大概有多少 |
| `rollout` | 一段交互轨迹 | 连续采样得到的一小段经验 |
| `advantage A_t` | 优势函数 | 某个动作比“平均水平”好多少 |

这里有一个很常见的误解：很多初学者会把 `A2C` 误解成一种“新的策略类型”。这不对。`A2C` 并不是把随机策略改成确定性策略，它只是把 `A3C` 的**异步更新流程改成同步更新流程**。动作依然通常来自概率分布采样，而不是固定死的输出。

再强调边界：`A2C/A3C` 都是 `on-policy` 方法，因此它们不能像 `DQN`、`SAC` 那样自由复用旧经验。它们通常用“当前这批 rollout”算梯度，更新后就进入下一轮采样。经验回放（experience replay，旧数据缓存后反复训练）并不是它们的核心设计点。`A3C` 在历史上一个重要贡献就是：**即使不用经验回放，也能通过多 actor 并行来削弱样本相关性**。

| 方法类别 | 是否 on-policy | 是否依赖当前策略新样本 |
|---|---|---|
| A2C | 是 | 是 |
| A3C | 是 | 是 |
| DQN | 否 | 否，常配经验回放 |
| SAC | 否 | 否 |
| PPO | 是 | 是，但更新方式更保守 |

---

## 核心机制与推导

`A2C` 和 `A3C` 的训练目标本质相同：让高优势动作更容易被选中，同时让价值函数更接近真实回报。这里的“优势”可以理解为：这个动作相对于当前状态下的平均预期，到底好多少。

常用的 `n` 步回报与优势定义是：

$$
G_t^{(n)}=\sum_{k=0}^{n-1}\gamma^k r_{t+k}+\gamma^n V_\theta(s_{t+n})
$$

$$
A_t = G_t^{(n)} - V_\theta(s_t)
$$

其中，`\gamma` 是折扣因子，表示“未来奖励还值多少钱”；越接近 `1`，说明越看重长期收益。

策略损失和总损失通常写成：

$$
L_{\text{policy}}=-\mathbb{E}\left[\log \pi_\theta(a_t|s_t)\cdot A_t\right]
$$

$$
L = L_{\text{policy}} + c_v L_{\text{value}} - c_e H(\pi_\theta)
$$

其中，`H(\pi_\theta)` 是熵正则，白话讲就是“鼓励策略别太早变得死板”，避免一开始就只会选单一动作。

### 玩具例子

假设某个 actor 采到 3 步奖励：

- `r_t = 1`
- `r_{t+1} = 0`
- `r_{t+2} = 2`

再设：

- `\gamma = 0.9`
- `V(s_{t+3}) = 1`
- `V(s_t) = 2`

那么：

$$
G_t^{(3)} = 1 + 0.9 \times 0 + 0.9^2 \times 2 + 0.9^3 \times 1
= 1 + 0 + 1.62 + 0.729 = 3.349
$$

于是：

$$
A_t = 3.349 - 2 = 1.349
$$

这个 `1.349` 表示：在当前价值函数看来，这个动作序列带来的结果比“正常预期”更好，所以策略应该提高对应动作的概率。

现在假设有 4 个 actor，在同一轮里分别算出一个优势信号：

- `1.349`
- `0.2`
- `-0.5`
- `0.8`

那么同步平均后的信号是：

$$
\frac{1.349 + 0.2 - 0.5 + 0.8}{4} = 0.46225
$$

这就是 `A2C` 的关键直觉：**不是谁先算完谁先改，而是把这一轮所有 actor 的信息合成一个更稳定的更新方向**。

### A3C 与 A2C 的更新差异

`A3C` 的更新可以写成：

$$
\text{actor } i \text{ 用局部参数 } \theta_i \text{ 采样，得到梯度 } g_i
$$

然后它立刻把 `g_i` 写回共享参数 `\theta`。问题在于，当 actor `i` 还在采样或反传时，其他 actor 可能已经多次修改了全局参数，所以 `g_i` 对应的其实是“旧参数世界”里的梯度。这种现象叫**陈旧梯度**，白话讲就是“你提交的修改基于旧版本代码，但主分支已经变了”。

`A2C` 则不同。设第 `k` 轮所有 actor 都使用同一版参数 `\theta_k`：

$$
g = \frac{1}{N}\sum_{i=1}^{N} g_i
$$

$$
\theta_{k+1} = \theta_k - \alpha g
$$

这里 `N` 是 actor 数量，`\alpha` 是学习率。由于同一轮的采样都来自相同参数版本，梯度语义更一致，更适合做批量张量计算。

可以把两者流程简化成下面这样：

| 阶段 | A3C | A2C |
|---|---|---|
| 采样 | 各 actor 独立采样 | 各 actor 并行采样 |
| 算优势 | 各自算 | 各自算 |
| 梯度处理 | 各自立即更新 | 聚合/平均后统一更新 |
| 参数版本 | 可能不同步 | 同一轮严格一致 |

---

## 代码实现

从代码结构上看，`A2C` 更像常规深度学习里的 batch 训练：先收数据，再统一反传。它通常包含三个部件：

- `VecEnv`：向量化环境，白话讲就是“一次开多个环境一起跑”
- `RolloutBuffer`：轨迹缓存，把 `n_steps × n_envs` 的样本先存下来
- `loss`：由 policy loss、value loss、entropy bonus 三部分组成

下面先给一个最小可运行的 `python` 例子，只演示 `n` 步回报和优势计算：

```python
def n_step_return(rewards, gamma, bootstrap_value):
    total = 0.0
    for k, r in enumerate(rewards):
        total += (gamma ** k) * r
    total += (gamma ** len(rewards)) * bootstrap_value
    return total

def advantage(rewards, gamma, bootstrap_value, state_value):
    g = n_step_return(rewards, gamma, bootstrap_value)
    return g - state_value

rewards = [1.0, 0.0, 2.0]
gamma = 0.9
bootstrap_value = 1.0
state_value = 2.0

g = n_step_return(rewards, gamma, bootstrap_value)
a = advantage(rewards, gamma, bootstrap_value, state_value)

assert round(g, 3) == 3.349
assert round(a, 3) == 1.349

advantages = [1.349, 0.2, -0.5, 0.8]
avg_advantage = sum(advantages) / len(advantages)
assert round(avg_advantage, 5) == 0.46225

print("return:", round(g, 3))
print("advantage:", round(a, 3))
print("avg_advantage:", round(avg_advantage, 5))
```

`A2C` 的训练循环通常长这样：

```python
for update in range(num_updates):
    rollouts = collect_rollouts(envs, n_steps)   # shape ~ [n_steps, n_envs, ...]
    returns, advantages = compute_returns_and_advantages(rollouts)

    policy_loss = compute_policy_loss(rollouts, advantages)
    value_loss = compute_value_loss(rollouts, returns)
    entropy = compute_entropy(rollouts)

    loss = policy_loss + value_coef * value_loss - entropy_coef * entropy

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
```

这个结构有两个重要特点。

第一，`batch` 大小天然是 `n_envs × n_steps`。例如 `16` 个环境每次各跑 `5` 步，一轮就有 `80` 个时间步样本。这种组织方式更适合 GPU，因为张量可以一起算。

第二，采样和更新边界很清晰。你很容易知道“这轮数据对应哪版参数”，也很容易做日志记录、断点恢复和实验复现。

而 `A3C` 常见流程更接近下面这样：

```python
# worker i
while not done:
    rollout = collect_rollout(local_env, local_model, n_steps)
    loss = compute_loss(rollout)
    loss.backward()
    push_gradients_to_global_model()
    pull_latest_global_params_to_local_model()
```

看起来更“实时”，但工程难点也更多：线程或进程并发、共享参数同步、梯度写回时机、锁竞争、不同 worker 速度差异，这些问题都不直接提升算法质量，却显著增加实现成本。

### 真实工程例子

以 `Atari` 或经典连续控制任务为例，单机训练时常见做法是：

- 开 `8` 到 `16` 个并行环境
- 每个环境采 `5` 到 `16` 步 rollout
- 合并成一个 `n_envs × n_steps` 的 batch
- 在 GPU 上统一算 loss 并更新

这就是典型的 `A2C` 工程形态。它不追求每个 worker 一有结果就马上更新，而是追求**吞吐稳定、参数一致、张量批量化友好**。对现代硬件来说，这通常比老式 `A3C` 更划算。

---

## 工程权衡与常见坑

`A2C` 的优势不是“理论上一定更强”，而是“在现代工程环境下更容易把硬件和代码组织好”。`A3C` 的优势也不是“已经过时毫无价值”，而是“它在没有大规模 GPU batch 习惯之前，提供了一种很有启发性的异步并行训练思路”。

先看工程维度对比：

| 维度 | A2C | A3C |
|---|---|---|
| 吞吐组织 | 批量统一 | 多 worker 零散写回 |
| 稳定性 | 较高 | 易受陈旧梯度影响 |
| 调试难度 | 低 | 高 |
| GPU 亲和性 | 高 | 低 |
| 单次更新时间延迟 | 高一些，要等齐 | 低一些，谁快谁更先更新 |

常见坑主要有五类。

第一，把 `A2C` 误解成确定性策略。它不是。同步的是更新过程，不是动作输出形式。

第二，只看到异步“看起来更快”。如果只盯着“谁先算完谁先更新”，确实会觉得 `A3C` 更灵活；但真实训练速度不只看更新频率，还看 GPU 是否吃满、梯度是否稳定、实验是否容易复现。

第三，`n_envs × n_steps` 太小。batch 太碎会导致优势估计噪声大，梯度方差高，训练曲线容易抖。

第四，`n_envs` 盲目加大。如果环境步进本身很慢，环境模拟就会成为瓶颈，此时不是优化器慢，而是样本生产慢。

第五，损失项配置不合理。比如忘记做优势归一化，或把 `entropy_coef` 设得太小，策略很早塌缩；把 `value_coef` 设得太大，又可能让价值损失压过策略学习。

一个常见起步配置可以参考下面这个量级：

| 参数 | 常见起点 | 作用 |
|---|---|---|
| `n_envs` | `8` 或 `16` | 并行环境数量 |
| `n_steps` | `5` 或 `16` | 每轮每个环境采样步数 |
| `entropy_coef` | `0.001` 到 `0.01` | 保持探索 |
| `value_coef` | `0.25` 到 `0.5` | 平衡价值损失 |
| `gamma` | `0.99` | 长期奖励折扣 |

这些值不是定律，但适合作为第一次实验的基线。

---

## 替代方案与适用边界

如果今天是从零开始做一个强化学习项目，很多团队不会直接把 `A3C` 当成默认选项。原因很简单：它的异步设计有历史意义，但在现代单机 GPU 训练里，往往没有同步批量方法划算。

`A2C` 的定位更像是：**简单、清晰、同步并行的 actor-critic 基线**。它适合你想快速搭一个能工作的 on-policy 训练器，或者需要教学、验证、做中小规模实验。

`PPO` 则通常是进一步的默认选择。它仍然是 on-policy，但通过裁剪目标函数控制更新步子，训练稳定性往往更好，因此在工业和研究里都更常见。

| 方案 | 适合场景 | 局限 |
|---|---|---|
| A2C | 需要简单同步实现、并行采样、清晰基线 | 样本效率一般 |
| A3C | 想研究异步更新机制、理解历史方法 | 工程复杂、复现性差 |
| PPO | 想要更稳、更通用的 on-policy 默认方案 | 实现比 A2C 稍复杂 |

可以用一句很实用的判断来收尾：

- 如果你在单机上训练，且希望实现简单、结果稳定，优先选 `A2C`。
- 如果你需要一个更常见、更稳的现代基线，优先看 `PPO`。
- 如果你在研究“异步并发更新为什么有效、又为什么会出问题”，再专门看 `A3C`。

所以，`A3C vs A2C` 真正的比较重点不是“谁更先进”，而是“同步与异步在现代硬件和工程体系下，哪个更值得付出复杂度成本”。大多数情况下，答案是：**同步的 `A2C` 更实用，而 `A3C` 更适合理解历史与机制。**

---

## 参考资料

1. [Asynchronous Methods for Deep Reinforcement Learning](https://arxiv.org/abs/1602.01783)
2. [OpenAI Baselines: ACKTR & A2C](https://openai.com/index/openai-baselines-acktr-a2c/)
3. [Stable-Baselines3 Documentation: A2C](https://stable-baselines3.readthedocs.io/en/v2.4.0/modules/a2c.html)
4. [Stable-Baselines3 Source: OnPolicyAlgorithm](https://github.com/DLR-RM/stable-baselines3/blob/master/stable_baselines3/common/on_policy_algorithm.py)
5. [Stable-Baselines3 Source: A2C](https://github.com/DLR-RM/stable-baselines3/blob/master/stable_baselines3/a2c/a2c.py)
