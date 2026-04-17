## 核心结论

SAC，Soft Actor-Critic，直译是“软演员-评论家”。“软”的意思不是模型更简单，而是优化目标里多了一项熵；熵可以粗看成“策略保留随机性的程度”。它的核心目标不是只追求奖励最大，而是追求“奖励 + 可控探索”同时最大：

$$
J(\pi)=\mathbb{E}_{\tau\sim\pi}\left[\sum_t r(s_t,a_t)+\alpha \mathcal{H}(\pi(\cdot|s_t))\right]
$$

这带来三个工程上很重要的结论：

| 结论 | 作用 | 对工程的意义 |
| --- | --- | --- |
| `off-policy` | 复用旧数据训练 | 样本效率高，适合采样昂贵任务 |
| 双 critic 取最小值 | 抑制 Q 值高估 | 收敛更稳，发散更少 |
| 自动温度 $\alpha$ | 自动调探索强度 | 少手调一个关键超参数 |

对初学者最重要的判断是：SAC 通常是连续动作控制的默认强基线。比如 HalfCheetah-v4、机械臂控制、储能充放电这类任务，只要奖励尺度和动作映射没写错，SAC 往往比 DDPG 更稳，比 PPO 更省样本。

一个常见可用配置是：`lr=3e-4`、`gamma=0.99`、`tau=0.005`、`buffer=1e6`、`batch=256`、`ent_coef="auto"`、`target_entropy="auto"`。在这类配置下，训练回报常见现象是前期波动较大，但熵逐步下降，Q 值和 actor loss 逐步进入稳定区间。

---

## 问题定义与边界

SAC 解决的是连续动作空间下的马尔可夫决策过程。马尔可夫决策过程，简称 MDP，可以理解成“系统每一步看当前状态、执行动作、得到奖励、进入下一个状态”的标准决策框架。

它的边界要先说清，否则容易误用：

| 边界/约束 | 说明 |
| --- | --- |
| 动作空间 | 主要面向连续动作，策略通常输出高斯分布，再经过 `tanh` 压到有限区间 |
| 数据来源 | 依赖 replay buffer，即经验回放池，适合反复复用旧样本 |
| 探索控制 | 通过熵项和温度系数 $\alpha$ 控制，不靠手工加高斯噪声 |
| 训练条件 | 常见实现要求先预热 buffer，再开始梯度更新 |
| 数值稳定 | 对 reward 尺度、动作映射、log-prob 计算比较敏感 |

玩具例子可以先看一个单步决策。假设只有两个动作：

| 动作 | 即时奖励 |
| --- | --- |
| A | 1 |
| B | 0 |

如果只最大化奖励，策略会立刻变成“永远选 A”。但 SAC 不会立刻这么做，因为它还在乎策略熵。只要 $\alpha$ 还不小，B 仍会保留一定概率。白话说，SAC 会更偏向高回报动作，但不会太早把其他动作概率压到 0。

真实工程例子是楼宇能耗控制。状态里可能有室内温度、负荷、电价、电池荷电状态、天气预测；动作可能是 HVAC 设定点和储能充放电功率。这是高维、连续、带时序耦合的任务。SAC 在这类任务里有价值，不是因为它“理论更高级”，而是因为它能在复杂环境里保留探索，同时复用大量历史轨迹，减少真实系统或高保真仿真的采样成本。

---

## 核心机制与推导

SAC 的更新链条可以压缩成四个公式。理解这四个公式，基本就理解了算法。

第一步，最大熵目标：

$$
J(\pi)=\mathbb{E}_{\tau\sim\pi}\left[\sum_t r(s_t,a_t)+\alpha \mathcal{H}(\pi(\cdot|s_t))\right]
$$

第二步，soft Bellman 目标。Bellman 备份可以理解成“用下一步的价值反推当前动作值”：

$$
Q_{\text{target}}=r+\gamma \mathbb{E}_{a'\sim\pi}\left[\min(Q_1(s',a'),Q_2(s',a'))-\alpha \log \pi(a'|s')\right]
$$

这里要注意两点。第一，双 critic 取最小值，是为了抑制过高估计。第二，目标里减去 $\alpha \log \pi$，等价于把“未来仍保持一定随机性”也算进价值。

第三步，策略损失：

$$
L_\pi=\mathbb{E}_{s\sim \mathcal{D},a\sim\pi}\left[\alpha \log \pi(a|s)-Q(s,a)\right]
$$

这个式子直观上是在做两件事：让动作的 Q 值尽量高，同时又不让策略过早塌成单点分布。换个角度，最优策略会被拉向 $\exp(Q/\alpha)$ 这类 Boltzmann 形式，也就是高 Q 动作概率更高，但概率不是硬切换。

第四步，自动温度更新：

$$
L_\alpha=\mathbb{E}_{a\sim\pi}\left[-\alpha(\log\pi(a|s)+\mathcal{H}_{\text{target}})\right]
$$

目标熵 $\mathcal{H}_{\text{target}}$ 可以理解成“系统希望策略保留多少随机性”。如果当前策略太确定，熵太低，$\alpha$ 会被推高；如果当前策略太散，$\alpha$ 会被压低。

把这四步连起来，SAC 的训练逻辑就是：

1. critic 学“当前策略下，动作到底值多少”。
2. actor 学“往高 Q 动作移动，但不要过早失去探索”。
3. $\alpha$ 学“探索应该保留到什么程度”。
4. target network 慢速跟随，避免目标值抖动。

HalfCheetah 这类环境里，一个健康的训练轨迹通常表现为：前期 entropy 较高，return 偏低；中期 Q 值上升、entropy 缓慢下降；后期 actor loss 和 critic loss 波动缩小，return 进入平台区。如果一开始 entropy 就快速掉到底，通常不是“学得快”，而是探索提前死掉了。

---

## 代码实现

下面先给一个最小玩具实现，用单状态两动作的最大熵策略更新说明 $\alpha$ 的作用。它不是完整 SAC，但能运行并验证“高奖励动作概率更大，同时保留随机性”。

```python
import math

def softmax(xs):
    m = max(xs)
    exps = [math.exp(x - m) for x in xs]
    s = sum(exps)
    return [x / s for x in exps]

# 单状态、两个动作的 Q 值
q_values = [1.0, 0.0]

def sac_policy_from_q(q_values, alpha):
    # pi(a|s) ∝ exp(Q(a)/alpha)
    logits = [q / alpha for q in q_values]
    return softmax(logits)

pi_high_entropy = sac_policy_from_q(q_values, alpha=1.0)
pi_low_entropy = sac_policy_from_q(q_values, alpha=0.1)

assert 0.5 < pi_high_entropy[0] < 1.0
assert pi_low_entropy[0] > pi_high_entropy[0]
assert abs(sum(pi_high_entropy) - 1.0) < 1e-9
assert abs(sum(pi_low_entropy) - 1.0) < 1e-9

print("alpha=1.0:", pi_high_entropy)
print("alpha=0.1:", pi_low_entropy)
```

这个例子说明：$\alpha$ 越大，策略越“软”；$\alpha$ 越小，策略越接近贪心。

真实实现里，一般用高斯策略加 `tanh` 压缩动作。高斯策略就是网络输出均值和方差，从中采样连续动作。关键代码是 critic 目标、actor loss、alpha loss 三段：

```python
import torch

def critic_target(reward, done, gamma, log_alpha, next_logp, q1_next, q2_next):
    soft_value = torch.min(q1_next, q2_next) - log_alpha.exp() * next_logp
    return reward + (1.0 - done) * gamma * soft_value

# toy tensors
reward = torch.tensor([[1.0]])
done = torch.tensor([[0.0]])
gamma = 0.99
log_alpha = torch.tensor([0.0], requires_grad=True)  # alpha = 1.0
next_logp = torch.tensor([[-0.7]])
q1_next = torch.tensor([[2.0]])
q2_next = torch.tensor([[1.8]])

target = critic_target(reward, done, gamma, log_alpha, next_logp, q1_next, q2_next)
expected = 1.0 + 0.99 * (1.8 - 1.0 * (-0.7))
assert torch.allclose(target, torch.tensor([[expected]]), atol=1e-6)
print(target)
```

完整训练循环通常是：

1. 从 replay buffer 采样一个 batch。
2. 用目标 critic 和下一个动作算 soft Bellman target。
3. 更新两个 critic。
4. 用重参数化采样当前动作，更新 actor。
5. 如果开启自动熵，更新 `log_alpha`。
6. 用 $\tau$ 做目标网络软更新。

真实工程例子可以直接落到 Stable Baselines3 配置上。比如 HalfCheetah-v4：

| 超参数 | 常见起点 |
| --- | --- |
| `learning_rate` | `3e-4` |
| `buffer_size` | `1e6` |
| `batch_size` | `256` |
| `tau` | `0.005` |
| `gamma` | `0.99` |
| `ent_coef` | `"auto"` |
| `target_entropy` | `"auto"` |

建议每 `1e4` 步记录一次：`episodic return`、`policy entropy`、`alpha`、`actor loss`、`critic loss`、`Q mean/Q variance`。这些曲线比单看 return 更能说明训练是否健康。

---

## 工程权衡与常见坑

SAC 的优点是稳，但前提是关键细节没有写错。最常见的问题不是“理论参数没调对”，而是实现和数值尺度出错。

| 异常指标 | 常见原因 | 推荐修复 |
| --- | --- | --- |
| entropy 过快下降 | $\alpha$ 太低，actor lr 太高 | 提高目标熵，降低 actor lr |
| critic loss 爆炸 | buffer 太小，$\tau$ 太大，reward 过大 | 增大 buffer，减小 $\tau$，做 reward scaling |
| Q 值来回震荡 | target 更新过快，训练过早开始 | 先预热 buffer，再训练，保持 update ratio 1:1 |
| 动作总在中间值 | `tanh` 后映射错误，动作范围没还原 | 检查动作缩放和环境 action space |
| alpha 不收敛 | log-prob 的 Jacobian 校正漏了 | 修正 squashed Gaussian 的 log-prob 计算 |

这里有两个很实际的权衡。

第一是 `batch size`。从 256 提到 512，通常会让梯度更平滑，训练曲线更稳，但显存占用会近似线性上升。吞吐会在 GPU 未饱和前提升，但达到显存或带宽瓶颈后收益会变差。工程上应该同时看 `step/sec` 和显存，而不是只盯收敛速度。

第二是自动熵与固定熵。自动熵更省调参，适合基线训练；固定 $\alpha$ 更适合做可重复的对照实验，因为它减少了一个自适应变量。初学者默认先用自动熵，只有在实验控制要求很强时再固定。

训练失败时，建议按下面顺序定位：

1. 先看 reward 和 entropy 是否同步变化。
2. 再看 `alpha` 是否异常飙升或贴近 0。
3. 再看 critic loss 和 Q variance 是否爆炸。
4. 最后检查动作缩放、reward 尺度、log-prob 校正和 replay 预热。

如果 return 停滞、entropy 不降、Q 值却持续抬高，多半是 critic 过估计。反过来，如果 entropy 迅速接近 0、return 也不上升，多半是策略过早确定化。

---

## 替代方案与适用边界

SAC 不是所有场景都最优。它更像“连续控制的高鲁棒基线”，不是“通吃所有强化学习任务”。

| 算法 | 探索机制 | 适用场景 |
| --- | --- | --- |
| SAC | 熵奖励 + 自动 $\alpha$ | 连续动作、样本昂贵、希望稳收敛 |
| TD3/DDPG | 确定性策略 + 外加噪声 | 精度优先、探索可手工设计 |
| PPO | on-policy | 并行采样强、实现简单、离策略复用不重要 |

SAC 相比 TD3/DDPG 的优势是更稳，尤其在环境噪声大、奖励稀疏、策略容易早熟时更明显。相比 PPO 的优势是样本效率高，因为 replay buffer 能反复利用旧样本。代价是实现更复杂，日志诊断要求更高。

真实工程里，机器人 locomotion 和灵巧手控制常把 SAC 当作强基线，因为它对超参数更不敏感，也更能容忍真实世界噪声。相反，如果任务要求极高的确定性轨迹跟踪，或者动作本身就是离散的，SAC 就不是第一选择。前者可以优先看 TD3，后者通常转向 DQN 系列或离散动作版 SAC 变体。

所以适用边界可以压缩成一句话：连续动作、样本珍贵、希望稳收敛，用 SAC；确定性极强或离散动作主导，就优先考虑别的算法。

---

## 参考资料

- Haarnoja 等，Soft Actor-Critic: Off-Policy Maximum Entropy Deep Reinforcement Learning with a Stochastic Actor  
- OpenAI Spinning Up: Soft Actor-Critic  
- RLinf: SAC 教程与总结  
- Emergent Mind: Soft Actor-Critic Implementation  
- TheLinuxCode: Soft Actor-Critic 实战与日志分析  
- BAIR: SAC 在 Minitaur 与机器人任务中的应用  
- MDPI: 建筑集群能耗优化中的 SAC/ORAR-SAC 工程实践  
- Engineering Notes: HalfCheetah-v4 上使用自动熵的训练示例
