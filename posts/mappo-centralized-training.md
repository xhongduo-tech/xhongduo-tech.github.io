## 核心结论

MAPPO 是 PPO 在多智能体强化学习中的直接扩展。它保留了 PPO 的核心更新方式，也就是用裁剪目标限制策略每次只做小步更新；真正的变化在于价值函数的输入：训练时，Critic 使用全局状态 $s$ 估值；执行时，每个 Actor 只根据自己的局部观测 $o_i$ 选动作。这种范式叫 CTDE，中文常译为“中心化训练、分散执行”，意思是训练阶段允许使用更多信息，部署阶段仍然满足每个智能体只能看见本地信息的约束。

它解决的不是“单个策略学不会动作”，而是“多个人一起学时，环境会互相扰动”。在单智能体 PPO 中，环境转移主要由当前策略决定；在多智能体里，其他 agent 也在同步更新，所以同一个动作今天有效、下一轮可能失效，这叫非平稳性，白话讲就是训练对象一直在变。MAPPO 用中心化 Critic 缓解这个问题：Actor 继续只看局部，Critic 则用全局信息更稳定地判断“这一步值不值”。

最值得记住的定义只有两行：

$$
\pi_\theta(a_i \mid o_i), \qquad V_\phi(s)
$$

前者是第 $i$ 个 agent 的策略，输入本地观测；后者是中心化价值函数，输入全局状态。

| 组件 | 训练时输入 | 执行时输入 | 作用 |
|---|---|---|---|
| Actor | 本地观测 $o_i$ | 本地观测 $o_i$ | 选动作 |
| Critic | 全局状态 $s$ | 不参与执行 | 估值、算优势 |

玩具例子可以这样理解：有两个清洁机器人共同搬箱子。每个机器人只能看到自己前方一小块区域，所以 Actor 只能根据局部视野决定“推、停、左转、右转”；但训练时，Critic 可以看到两台机器人的相对位置、箱子位置和目标点，因此更容易判断这一步是推进了任务，还是只是两台机器人互相卡住。

---

## 问题定义与边界

多智能体强化学习先定义几个符号。设系统里有 $N$ 个 agent，编号为 $i=1,\dots,N$。第 $i$ 个 agent 在时刻 $t$ 的局部观测是 $o_{i,t}$，全局状态是 $s_t$，动作是 $a_{i,t}$。所有 agent 的动作拼起来得到联合动作：

$$
a_t=(a_{1,t},a_{2,t},\dots,a_{N,t})
$$

环境根据联合动作推进，给出奖励和下一时刻状态。

难点主要有四个。

第一，局部可观测。局部可观测的意思是单个 agent 只能看到全局的一部分，因此它很难仅凭自己的输入判断失败原因。第二，非平稳性。因为其他 agent 也在学习，你面对的“环境”不是固定的。第三，credit assignment，也就是“功劳分配”问题，白话讲就是团队拿到一个总奖励后，很难知道谁的动作真正贡献了结果。第四，奖励尺度不稳定，不同地图、回合长度、胜负方式可能让价值目标波动很大。

MAPPO 适合的边界也很明确。它通常用于协作型任务，尤其是共享奖励、局部可观测、需要稳定训练的场景。它不是所有多智能体问题的默认答案。如果每个 agent 的任务几乎完全独立，中心化 Critic 提供的额外信息就未必值得引入。

| 特征 | 是否适合 MAPPO | 原因 |
|---|---|---|
| 共享奖励 | 适合 | 中心化 Critic 更容易学习团队回报 |
| 强协作 | 适合 | credit assignment 更关键 |
| 局部可观测 | 适合 | Critic 可补充全局信息 |
| 完全独立任务 | 一般 | 中心化估值收益可能有限 |
| 执行端必须轻量 | 适合 | Actor 只依赖本地观测 |

真实工程例子是 SMAC。SMAC 是星际争霸微操任务，单个 marine 只能看到局部视野，执行时不可能访问全局地图，所以 Actor 必须只看局部观测；但训练时环境可以提供更完整的全局状态，Critic 因此能更准确地区分“是我走位错了”还是“队友没及时集火”。Hanabi 也是典型场景：信息不完全、协作强、奖励稀疏，中心化估值通常比独立学习更稳。

---

## 核心机制与推导

MAPPO 的机制可以拆成两条线。第一条线是策略线，也就是 Actor 学“在本地观测下该做什么”；第二条线是估值线，也就是 Critic 学“当前全局局势值多少钱”。优势函数 $A_t$ 是连接两条线的桥梁，它表示某一步比基线好多少。优势的白话解释是：同样处在这个局面，你这次动作到底比“平均水平”更好还是更差。

最简单写法是：

$$
A_t=\hat R_t - V_\phi(s_t)
$$

其中 $\hat R_t$ 是回报目标，$V_\phi(s_t)$ 是中心化 Critic 的估值。实践里通常不用最原始的 Monte Carlo 回报，而是用 GAE，中文常叫“广义优势估计”，它通过折中偏差和方差让优势更平滑：

$$
A_t^{\text{GAE}(\gamma,\lambda)}=\sum_{l=0}^{T-t-1}(\gamma\lambda)^l\delta_{t+l}
$$

其中 $\delta_t=r_t+\gamma V(s_{t+1})-V(s_t)$。

然后进入 PPO 的核心。PPO 不直接让新策略完全替代旧策略，而是比较它们对同一动作给出的概率比值：

$$
r_t(\theta)=\frac{\pi_\theta(a_t\mid o_{i,t})}{\pi_{\theta_{\text{old}}}(a_t\mid o_{i,t})}
$$

再用裁剪目标限制更新幅度：

$$
L^{\text{clip}}=\mathbb E\big[\min(r_tA_t,\ \text{clip}(r_t,1-\epsilon,1+\epsilon)A_t)\big]
$$

这条式子的含义是：如果新策略只是适度提高好动作的概率，就允许；如果提高过猛，就把收益截断，避免一步走太远。多智能体里这点尤其重要，因为别的 agent 也在同步变化，过激更新会把非平稳性进一步放大。

看一个最小数值例子。假设中心化 Critic 看到全局状态后给出 $V_\phi(s_t)=8$，回报目标是 $\hat R_t=10$，那么优势 $A_t=2$。旧策略对当前动作的概率是 0.20，新策略是 0.24，于是：

$$
r_t = \frac{0.24}{0.20}=1.2
$$

若 $\epsilon=0.2$，那么 $1.2$ 还在允许区间内，目标贡献约为 $1.2\times 2=2.4$。如果新概率涨到 0.30，则 $r_t=1.5$，超过裁剪上界，实际按 $1.2$ 处理。这就是 PPO 在多智能体里仍然有效的原因：它不保证最优，但能显著减少训练震荡。

MAPPO 论文和实现里还强调了 value normalization 或 PopArt。PopArt 可以理解成“给价值学习自动调量纲”。因为不同地图、任务长度、奖励设计会让回报尺度差很多，Critic 很容易因为目标绝对值过大或变化过快而训练不稳。最基础的归一化写法是：

$$
\tilde y_t=\frac{y_t-\mu}{\sigma}
$$

其中 $y_t$ 是原始价值目标，$\mu,\sigma$ 是运行中的均值和标准差。PopArt 的关键不是只做归一化，而是在更新 $\mu,\sigma$ 时重参数化网络最后一层，使未归一化输出尽量保持不变。这样你改变的是数值尺度，不是把已经学到的价值函数整体打乱。

整个机制可以按 6 步理解：

1. 收集多 agent 轨迹。
2. Critic 用全局状态估值。
3. 计算 return 和 advantage。
4. Actor 用 PPO 裁剪目标更新。
5. Critic 拟合回报目标。
6. 必要时对 value target 做 normalization 或 PopArt。

---

## 代码实现

实现 MAPPO 时，真正容易出错的不是网络结构，而是数据组织。Actor 和 Critic 的输入必须从张量层面严格分开，否则很容易在训练时把全局状态错误地喂给 Actor，得到一个线上根本无法部署的策略。

| 张量 | 维度含义 | 用途 |
|---|---|---|
| `obs` | `[T, N, obs_dim]` | Actor 输入 |
| `global_state` | `[T, state_dim]` | Critic 输入 |
| `actions` | `[T, N]` 或 `[T, N, act_dim]` | 策略更新 |
| `logp_old` | `[T, N]` | 计算 PPO ratio |
| `rewards` | `[T]` 或 `[T, N]` | 回报计算 |
| `dones` | `[T]` | 截断、mask |
| `advantages` | `[T, N]` | 策略目标 |

下面给一个可运行的玩具实现，重点不是深度学习框架，而是把 PPO 裁剪逻辑和中心化估值拆清楚：

```python
import math

def ppo_clip_objective(old_prob, new_prob, advantage, eps=0.2):
    ratio = new_prob / old_prob
    clipped_ratio = min(max(ratio, 1 - eps), 1 + eps)
    return min(ratio * advantage, clipped_ratio * advantage)

def value_advantage(return_target, state_value):
    return return_target - state_value

# 玩具例子：两智能体共享奖励，Critic 用全局状态估值
V_s = 8.0
R_hat = 10.0
adv = value_advantage(R_hat, V_s)

obj1 = ppo_clip_objective(old_prob=0.20, new_prob=0.24, advantage=adv, eps=0.2)
obj2 = ppo_clip_objective(old_prob=0.20, new_prob=0.30, advantage=adv, eps=0.2)

assert adv == 2.0
assert abs(obj1 - 2.4) < 1e-9   # 1.2 * 2
assert abs(obj2 - 2.4) < 1e-9   # 1.5 被截断到 1.2

def normalize_target(y, mu, sigma):
    assert sigma > 0
    return (y - mu) / sigma

norm_y = normalize_target(10.0, mu=8.0, sigma=2.0)
assert abs(norm_y - 1.0) < 1e-9
print("MAPPO toy example passed")
```

如果把它扩展成训练循环，骨架大致如下：

```python
for each rollout:
    for t in range(T):
        actions, logp = actor(obs)              # 只看局部 obs
        next_obs, rewards, dones, info = env.step(actions)
        store(obs, global_state, actions, logp, rewards, dones)

    values = critic(global_states)              # 只在训练时看全局 state
    returns, advantages = compute_gae(rewards, values, dones)

    for epoch in range(K):
        for minibatch in sampler(buffer):
            new_logp = actor(minibatch.obs).log_prob(minibatch.actions)
            ratio = exp(new_logp - minibatch.old_logp)
            policy_loss = clipped_ppo_loss(ratio, minibatch.advantages)

            value_pred = critic(minibatch.global_states)
            value_loss = mse(value_pred, minibatch.returns)

            loss = policy_loss + c1 * value_loss - c2 * entropy
            update(loss)
```

新手最该盯住三点。第一，Actor 输入永远是 $o_i$，不是 $s$。第二，Critic 的目标通常按时间维和 episode mask 对齐，否则 padding 也会参与学习。第三，多智能体常见共享策略，意思是多个 agent 共用一套 Actor 参数，但每一步仍然有各自的观测和动作记录。

---

## 工程权衡与常见坑

MAPPO 的稳定性不是“因为它用了中心化 Critic 就自动稳定”，而是因为一组工程选择共同控制了训练噪声。论文和官方实现都反复强调，epoch 数、mini-batch 切分、价值归一化、死亡 mask 这类细节会直接决定结果。

| 常见坑 | 后果 | 规避方式 |
|---|---|---|
| Actor 偷看全局状态 | 训练部署不一致，线上性能掉点 | 全局信息只给 Critic |
| `ppo_epoch` 太多、mini-batch 太碎 | 旧数据被过度复用，非平稳放大 | 减少重复更新 |
| 不做 value normalization / PopArt | 价值目标尺度漂移，loss 爆炸 | 归一化 value target |
| death mask 缺失 | 死亡 agent 的输入分布异常 | 零向量或 death mask + agent ID |
| advantage 没做 mask | padding、终止后时间步污染梯度 | 对无效步显式屏蔽 |
| 奖励尺度差异过大 | Critic 难收敛 | 统一奖励尺度或做自适应归一化 |

为什么 Critic 可以看全局，而 Actor 不应该看全局？原因不是“为了形式好看”，而是训练目标和部署条件必须一致。Critic 只用于训练阶段的估值，它不参与线上动作选择；Actor 则在执行阶段直接控制行为。如果 Actor 在训练时依赖了部署时拿不到的信息，学到的策略本质上就是作弊。

真实工程里还有两个高频问题。一个是动态实体问题，比如 SMAC 中 unit 会死亡，某些时间步 agent 数有效值变化，如果不做 death mask，Critic 会把“死亡后的空输入”误当作正常状态。另一个是数据复用过量。单智能体 PPO 有时还能容忍多轮训练同一批数据，但多智能体里别的 agent 也在同步变，旧轨迹更快失真，所以通常应更克制地设置 `ppo_epoch` 和 mini-batch 数。

---

## 替代方案与适用边界

MAPPO 不是多智能体强化学习的唯一主线，它只是当前工程上比较稳、比较容易落地的一条线。它强在保留了 PPO 的训练稳定性，同时通过中心化 Critic 缓解多智能体带来的估值困难；它弱在依赖可用的全局状态或联合信息，而且并不自动解决所有 credit assignment 问题。

| 方法 | 核心特点 | 优点 | 局限 |
|---|---|---|---|
| Independent PPO | 每个 agent 独立训练 | 简单、实现快 | 非平稳明显 |
| MAPPO | 中心化 Critic，分散执行 | 稳定、易复现、适合协作 | 依赖训练期全局信息 |
| MADDPG | 中心化 Critic + 确定性策略 | 连续动作场景常见 | 对超参较敏感 |
| QMIX | 值分解 | 协作任务强、样本效率高 | 更偏离散动作设定 |
| COMA | 反事实 baseline | 更直接处理 credit assignment | 计算复杂、训练更重 |

如果任务几乎是独立控制，比如多个设备各自调温、彼此耦合很弱，Independent PPO 往往已经够用；如果任务是强协作、共享奖励、局部观测严重受限，MAPPO 通常比独立训练稳；如果需要显式建模团队价值分解，QMIX 可能更直接；如果连续控制很强且希望用 actor-critic 的中心化评估，MADDPG 也是常见选择。

所以适用边界可以压缩成一句话：MAPPO 最适合“协作型、局部可观测、共享奖励、训练期可获得更完整全局信息”的问题；当任务独立性很强，或训练期也拿不到可靠中心化状态时，它的优势会明显减弱。

---

## 参考资料

1. [The Surprising Effectiveness of PPO in Cooperative, Multi-Agent Games](https://arxiv.org/abs/2103.01955)
2. [marlbenchmark/on-policy: Official MAPPO Implementation](https://github.com/marlbenchmark/on-policy)
3. [BAIR Blog: PPO in Cooperative Multi-Agent Games](https://bair.berkeley.edu/blog/2021/07/14/mappo/)
4. [OpenAI Spinning Up: Proximal Policy Optimization](https://spinningup.openai.com/en/latest/algorithms/ppo.html)
5. [DeepMind: Preserving Outputs Precisely while Adaptively Rescaling Targets](https://deepmind.google/discover/blog/preserving-outputs-precisely-while-adaptively-rescaling-targets/)
6. [PopArt Paper](https://arxiv.org/abs/1809.04474)
