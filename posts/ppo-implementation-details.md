## 核心结论

PPO（Proximal Policy Optimization，近端策略优化）是一种 `on-policy actor-critic` 方法。`on-policy` 的白话解释是：模型主要用“当前策略刚刚采样出来的数据”训练自己，而不是长期反复回放旧数据。它之所以在工程上常用，不是因为单个公式很强，而是因为一整套“限制更新幅度”的机制配合得比较完整。

最关键的结论有三条。

第一，PPO 稳定的核心不只在策略剪裁 `clip policy objective`。真正决定训练是否稳定的，通常是 `GAE(λ)`、价值函数训练方式、熵正则、梯度裁剪、归一化、mini-batch 多轮更新以及 `KL` 早停一起工作。

第二，`GAE` 解决的是“优势函数怎么估计更稳”。优势函数的白话解释是：一个动作比“平均水平动作”到底好多少。它如果波动太大，策略梯度就会变成高噪声更新；它如果偏差太大，策略会学错方向。

第三，PPO 的工程质量常常取决于 critic，也就是价值函数网络。很多初学者只盯着 policy loss，忽略了 value loss、`approx_kl`、`entropy` 和 `explained_variance`，最后看到的是“loss 在降，但回报不涨”。

| 组件 | 作用 | 不加会怎样 |
|---|---|---|
| Clip policy objective | 限制策略更新幅度 | 更新过猛，性能震荡 |
| GAE | 估计优势函数 | 优势方差大，训练不稳 |
| Value loss / clip | 训练 critic | 价值估计漂移 |
| Entropy bonus | 保留探索 | 过早收敛，陷入局部最优 |

PPO 常见的总体目标可以写成：

$$
J = L_{clip} - c_v L_V + \beta H[\pi_\theta]
$$

这里 $H[\pi_\theta]$ 是策略熵，熵的白话解释是“动作分布还有多分散”。分布越分散，探索越强；分布越尖锐，策略越确定。

---

## 问题定义与边界

PPO 要解决的问题，是策略梯度方法里“更新一步走太猛，下一步性能直接掉下去”。策略梯度的白话解释是：直接优化“在状态下选动作的概率分布”，让高回报动作更可能被选到。它的优点是可以自然处理离散动作和连续动作；缺点是更新容易不稳定。

PPO 的核心边界很明确：它是 `on-policy`，所以样本必须来自当前策略或与当前策略非常接近的旧策略。不能把几个月前、很多版本前的经验拿来无限复用，否则目标函数中的概率比率就不再可靠。

这也是为什么 PPO 通常被认为“稳定但样本效率一般”。样本效率的白话解释是：同样数量的环境交互，谁能学到更多。`off-policy` 方法如 SAC 往往能更充分复用数据，但实现和稳定性权衡不同。

| 维度 | PPO 的位置 |
|---|---|
| 数据来源 | on-policy |
| 任务类型 | 离散/连续动作都可 |
| 优点 | 稳定、易调、工程成熟 |
| 局限 | 样本效率不高，依赖 rollout 质量 |

使用 PPO 时，一般默认以下前提成立：

| 前提 | 含义 |
|---|---|
| 轨迹接近当前策略 | 新旧策略差异不能过大 |
| 奖励可用来构造回报 | 至少能形成 episode return 或 bootstrap return |
| actor/critic 有共同状态表示 | 常共享特征层，但不是硬要求 |

一个玩具理解是：你先让当前策略自己“玩”出一批轨迹，再拿这批轨迹训练几轮，但每一轮都只允许小改。如果你把这件事理解成“保守地反复修正最近一次经验”，就抓住了 PPO 的边界。

真实工程里，PPO 常用于机器人控制、游戏智能体、导航、多智能体中的单体策略优化等场景。比如一个移动机器人根据激光雷达和速度信息决定转向与前进速度，这类连续控制任务里，PPO 往往是先跑通的首选方法，因为训练过程相对可控，日志指标也比较成熟。

---

## 核心机制与推导

PPO 的机制可以拆成三个问题：

1. 优势怎么估计更稳：`GAE`
2. 策略怎么更新别太大：`clip`
3. 探索怎么别太早消失：`entropy bonus`

先看一步 TD 残差：

$$
\delta_t = r_t + \gamma V_\phi(s_{t+1}) - V_\phi(s_t)
$$

TD 残差的白话解释是：critic 这一步对“未来价值”的预测误差。它比整条轨迹回报更局部，比单纯即时奖励信息更多。

GAE 把后续多个 TD 残差按衰减方式累加：

$$
\hat{A}_t = \sum_{l=0}^{\infty} (\gamma \lambda)^l \delta_{t+l}
= \delta_t + \gamma \lambda \hat{A}_{t+1}
$$

这里 $\lambda$ 是“看多远”的平衡旋钮。$\lambda$ 越接近 1，越接近长时回报，偏差更小但方差更大；$\lambda$ 越小，越依赖短期估计，方差更低但偏差更大。

下面给一个玩具例子。设：

- $\gamma = 0.99$
- $\lambda = 0.95$
- $V(s_t)=10,\;V(s_{t+1})=9.5,\;V(s_{t+2})=9.0$
- $r_t=1,\;r_{t+1}=0$

则：

$$
\delta_t = 1 + 0.99 \times 9.5 - 10 = 0.405
$$

$$
\delta_{t+1} = 0 + 0.99 \times 9.0 - 9.5 = -0.59
$$

若 $t+2$ 终止，则 $\hat{A}_{t+1}=-0.59$，因此：

$$
\hat{A}_t = 0.405 + 0.99 \times 0.95 \times (-0.59) \approx -0.150
$$

| 时刻 | 值 |
|---|---|
| `γ` | `0.99` |
| `λ` | `0.95` |
| `δ_t` | `0.405` |
| `δ_{t+1}` | `-0.59` |
| `Â_t` | `≈ -0.150` |

这个结果是负数，含义是：虽然当前步即时奖励看起来还行，但结合后续价值变化，这个动作整体略差于基线，因此策略应该略微降低它的概率。

再看 PPO 的策略比率：

$$
\rho_t = \frac{\pi_\theta(a_t|s_t)}{\pi_{\theta_{old}}(a_t|s_t)}
$$

它衡量的是：新策略对旧动作的概率，相比旧策略放大了多少。然后用截断目标：

$$
L_{clip} = \mathbb{E}\left[\min\left(\rho_t \hat{A}_t,\; \text{clip}(\rho_t, 1-\epsilon, 1+\epsilon)\hat{A}_t\right)\right]
$$

这条式子的作用不是“保证策略一定变好”，而是“防止更新时改得太猛”。当优势为正时，PPO 不希望你无限制地提高动作概率；当优势为负时，也不希望你无限制地压低动作概率。

价值函数损失常写成：

$$
L_V = \mathbb{E}\left[(V_\phi(s_t)-R_t)^2\right]
$$

但工程上经常会额外做价值剪裁，或者把均方误差换成 Huber 损失。Huber 损失的白话解释是：小误差时像平方损失，大误差时像绝对值损失，对异常大误差更稳。原因是 critic 一旦更新过猛，会把 advantage 的基线拖偏，进而把 actor 的方向也带偏。

熵正则项：

$$
\beta H[\pi_\theta]
$$

它负责维持探索。训练初期如果没有熵约束，策略很容易在局部高回报动作上过早塌缩，表现为 entropy 很快降到很低，但总回报停在局部最优。

因此三者的分工可以明确写成：

| 机制 | 负责的问题 |
|---|---|
| `GAE` | 优势估计怎么更稳 |
| `clip` | 策略更新怎么别太大 |
| `entropy` | 探索怎么别太早收缩 |

---

## 代码实现

PPO 的训练流程通常可以拆成四步：收集轨迹、计算 `advantage/return`、多轮 mini-batch 更新、根据 `KL` 早停。公式不复杂，真正容易写错的是数据边界和训练顺序。

下面是一个可运行的最小 Python 实现片段，只演示 `GAE`、优势标准化和价值剪裁的核心逻辑：

```python
from math import isclose

def compute_gae(rewards, values, dones, gamma=0.99, lam=0.95):
    """
    rewards: length T
    values: length T + 1, 最后一个是 bootstrap value
    dones: length T, 1 表示 episode 在该步结束
    """
    T = len(rewards)
    adv = [0.0] * T
    gae = 0.0

    for t in reversed(range(T)):
        not_done = 1.0 - dones[t]
        delta = rewards[t] + gamma * values[t + 1] * not_done - values[t]
        gae = delta + gamma * lam * not_done * gae
        adv[t] = gae

    returns = [a + v for a, v in zip(adv, values[:-1])]
    return adv, returns

def normalize(xs, eps=1e-8):
    mean = sum(xs) / len(xs)
    var = sum((x - mean) ** 2 for x in xs) / len(xs)
    std = (var + eps) ** 0.5
    return [(x - mean) / std for x in xs]

def clipped_value_loss(v_old, v_new, v_target, clip_range_vf=0.2):
    v_clipped = v_old + max(-clip_range_vf, min(v_new - v_old, clip_range_vf))
    loss_unclipped = (v_new - v_target) ** 2
    loss_clipped = (v_clipped - v_target) ** 2
    return max(loss_unclipped, loss_clipped)

rewards = [1.0, 0.0]
values = [10.0, 9.5, 9.0]
dones = [0, 1]

adv, returns = compute_gae(rewards, values, dones, gamma=0.99, lam=0.95)
assert len(adv) == 2
assert isclose(adv[1], -0.59, rel_tol=1e-6, abs_tol=1e-6)
assert isclose(adv[0], -0.149855, rel_tol=1e-6, abs_tol=1e-6)

adv_n = normalize(adv)
assert len(adv_n) == 2
assert adv_n[0] * adv_n[1] < 0

loss = clipped_value_loss(v_old=1.0, v_new=1.5, v_target=2.0, clip_range_vf=0.2)
assert loss >= 0.0
```

在完整训练中，主循环通常长这样：

```python
for update in range(num_updates):
    rollout = collect_rollout(policy, env)

    advantages, returns = compute_gae(
        rewards=rollout.rewards,
        values=rollout.values,
        dones=rollout.dones,
        gamma=gamma,
        lam=gae_lambda,
    )

    advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

    for epoch in range(update_epochs):
        for batch in minibatch(rollout, advantages, returns):
            ratio = pi_new(batch.actions | batch.states) / pi_old(batch.actions | batch.states)
            policy_loss = -mean(min(ratio * batch.adv,
                                    clip(ratio, 1-eps, 1+eps) * batch.adv))

            value_pred = value_net(batch.states)
            value_target = batch.returns
            value_loss = mse(value_pred, value_target)

            entropy = mean(policy.entropy(batch.states))

            loss = policy_loss + c_v * value_loss - beta * entropy
            loss.backward()
            clip_grad_norm_(parameters, max_norm)
            optimizer.step()
            optimizer.zero_grad()
```

实现流程可以压缩为下表：

| 模块 | 输入 | 输出 |
|---|---|---|
| Rollout buffer | state, action, reward, done, value, logprob | 训练样本 |
| GAE | reward, value, done | advantage, return |
| Policy update | advantage, logprob_old, logprob_new | policy loss |
| Value update | return, value_pred | value loss |
| Regularization | entropy, grad norm, KL | 稳定性控制 |

几个最常见的实现细节：

| 细节 | 原因 |
|---|---|
| `done` 要截断 bootstrap | episode 已结束时，不能再接下一个状态价值 |
| `advantage` 通常标准化 | 降低 batch 间尺度波动 |
| `value` 常配 `clip_range_vf` | 限制 critic 单次跳动过大 |
| `target_kl` 可早停 | 防止多轮 epoch 把策略推太远 |
| 多环境并行采样 | 提高吞吐，也减少单轨迹偶然性 |

真实工程例子是导航任务。机器人只能看到局部深度图和最近几帧速度，这属于部分可观测环境，白话解释是“当前观测不足以完整恢复真实状态”。这时普通 PPO 容易在拐角或遮挡区域做出短视动作，工程上常用 `PPO + LSTM` 或 `RecurrentPPO`。实现时必须在 episode 边界重置隐藏状态，并保证 sequence mini-batch 不跨 episode 拼接，否则时序梯度会学到错误相关性。

---

## 工程权衡与常见坑

PPO 的“稳定”不是无条件稳定，而是“在一组常见工程约束下相对稳定”。训练炸掉通常不是论文公式错，而是几个超参数和数据处理方式不匹配。

最常见的坑如下：

| 问题 | 现象 | 处理方式 |
|---|---|---|
| `λ` 太小 | 学得短视 | 提高 `λ`，平衡偏差和方差 |
| `clip_range` 太大 | `KL` 飙升 | 缩小 `clip_range`，加 `target_kl` |
| `ent_coef` 太大 | 一直随机试错 | 降低熵系数，分阶段衰减 |
| `epoch` 太多 | 过拟合 on-policy 数据 | 减少 epoch，增大采样量 |
| 只看 policy loss | critic 已经偏掉 | 同时监控 value loss、`approx_kl`、entropy |
| 观测/奖励缩放不一致 | value clip 失真 | 统一归一化策略 |

这里的核心权衡可以概括成三组。

第一组是 `bias-variance tradeoff`，也就是偏差和方差权衡。`GAE λ` 越大，优势越接近长回报，通常更“不短视”，但抖动更大。初学者经常把学习不稳误判成“λ 不够大”，实际上可能是 rollout 太短或 critic 太差。

第二组是“每批数据更新多少次”。如果 `n_epochs` 太少，样本利用率低；如果太多，同一批 on-policy 数据会被过拟合，表现是训练集上的 surrogate loss 好看，但新采样性能下降。PPO 不是“epoch 越多越好”。

第三组是“探索与收敛”的平衡。熵系数过低，策略太快收缩；熵系数过高，策略长期像随机策略。常见做法是训练前期较高、后期逐步衰减。

需要重点监控的指标如下：

| 指标 | 看什么 |
|---|---|
| `approx_kl` | 新旧策略距离是否过大 |
| `clip_fraction` | 有多少样本触发了剪裁 |
| `entropy` | 探索是否过早衰减 |
| `value_loss` | critic 是否训练失衡 |
| `explained_variance` | 价值函数是否真的在解释回报 |

其中 `explained_variance` 可以粗略理解为“critic 对 return 的解释程度”。如果它长期接近 0 或为负，说明 critic 很可能没有提供可靠基线，此时 actor 再怎么优化也会摇摆。

还要注意两个常被忽略的工程点。

一是学习率退火。很多 PPO 实现会让学习率随着训练进度逐步下降，因为后期策略已接近局部最优，更需要小步修正。

二是观测归一化和奖励缩放。比如状态量一个维度在 $[0,1]$，另一个在 $[0,10^4]$，不做归一化时，网络梯度会被大尺度维度主导。奖励缩放如果和 value clip 配置不匹配，critic 的剪裁阈值就会失真，看起来“用了 value clip”，实际没有起到预期作用。

---

## 替代方案与适用边界

PPO 的优势是稳定、通用、工程生态成熟，但它不是所有任务的最优解。选择算法时，首先要明确你最在意什么：稳定性、样本效率、长期记忆，还是理论约束。

| 方法 | 特点 | 适合场景 |
|---|---|---|
| PPO | 稳定，易实现 | 通用控制、工程训练 |
| TRPO | 理论更强约束 | 研究场景、较少实际部署 |
| A2C/A3C | 实现简单 | 轻量任务 |
| SAC | 样本效率高 | 连续控制、可离线采样更多 |
| RecurrentPPO | 处理部分可观测 | 机器人、导航、序列决策 |

和 TRPO 相比，PPO 用剪裁近似替代了更严格的信赖域约束。信赖域的白话解释是：只允许策略落在一个“离旧策略不太远”的安全更新区域里。TRPO 约束更严格，但实现更复杂，计算成本更高，所以工业实践里 PPO 更常见。

和 SAC 相比，PPO 的短板主要是样本效率。若环境交互极贵，比如真实机械臂每采一条轨迹都要时间和硬件成本，那么 PPO 可能不是首选，因为它没法像 off-policy 方法那样高强度复用历史数据。

和普通 PPO 相比，`RecurrentPPO` 的适用边界在部分可观测环境。如果状态本来就是马尔可夫的，也就是当前观测足够描述未来决策所需信息，那么 LSTM 未必带来收益，反而会提高训练复杂度和调试成本。

因此可以把边界概括成三条：

| 条件 | 判断 |
|---|---|
| 观测不完整 | 普通 PPO 受限，考虑 PPO+LSTM |
| 样本极贵 | PPO 可能不是最优，优先看 SAC/TD3 |
| 需要更严格更新约束 | TRPO 风格方法更强 |

实践里，很多团队会先用 PPO 建立可训练基线。原因很简单：它最容易形成完整训练闭环。一旦基线跑通，再根据瓶颈决定是否换算法。如果瓶颈是“不收敛”，先查实现细节；如果瓶颈是“收敛太慢且采样太贵”，再考虑更高样本效率的方法。

---

## 参考资料

| 资料 | 用途 |
|---|---|
| [PPO 原论文](https://arxiv.org/abs/1707.06347) | 理解 clipped objective 的来源 |
| [GAE 原论文](https://arxiv.org/abs/1506.02438) | 理解 advantage 估计与 `λ` |
| [OpenAI Spinning Up: PPO](https://spinningup.openai.com/en/latest/algorithms/ppo.html) | 建立直觉和训练流程 |
| [Stable-Baselines3 PPO 文档/源码](https://stable-baselines3.readthedocs.io/en/v1.0/_modules/stable_baselines3/ppo/ppo.html) | 看工程实现和默认参数 |
| [SB3 VecNormalize](https://stable-baselines3.readthedocs.io/en/v2.4.1/_modules/stable_baselines3/common/vec_env/vec_normalize.html) | 理解观测/奖励归一化 |
| [RecurrentPPO 文档](https://sb3-contrib.readthedocs.io/en/master/modules/ppo_recurrent.html) | 处理部分可观测任务 |

1. [Proximal Policy Optimization Algorithms](https://arxiv.org/abs/1707.06347)
2. [High-Dimensional Continuous Control Using Generalized Advantage Estimation](https://arxiv.org/abs/1506.02438)
3. [OpenAI Spinning Up: PPO](https://spinningup.openai.com/en/latest/algorithms/ppo.html)
4. [Stable-Baselines3 PPO Documentation and Source](https://stable-baselines3.readthedocs.io/en/v1.0/_modules/stable_baselines3/ppo/ppo.html)
5. [SB3 VecNormalize Source](https://stable-baselines3.readthedocs.io/en/v2.4.1/_modules/stable_baselines3/common/vec_env/vec_normalize.html)
6. [sb3-contrib RecurrentPPO](https://sb3-contrib.readthedocs.io/en/master/modules/ppo_recurrent.html)
