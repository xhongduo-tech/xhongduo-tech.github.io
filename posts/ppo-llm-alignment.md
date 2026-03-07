## 核心结论

PPO 在 LLM 对齐中的作用，可以压缩成一句话：它不是单纯让模型“拿更高分”，而是在“更符合人类偏好”和“不要偏离 SFT 基础能力太远”之间做受约束的优化。

在 RLHF 第三阶段，常见目标可写为：

$$
\max_{\theta}\; \mathbb{E}_{y \sim \pi_\theta(\cdot|x)}\left[r_{\text{RM}}(x,y)\right] - \beta \cdot \mathrm{KL}\!\left(\pi_\theta(\cdot|x)\|\pi_{\text{ref}}(\cdot|x)\right)
$$

这里的 reward model 是“奖励模型”，白话说就是一个给回答打偏好分的模型；reference model 是“参考策略”，白话说就是冻结不动的 SFT 模型，用来告诉当前策略“别跑太远”。

PPO 稳定训练主要靠两层约束同时工作：

1. `KL` 惩罚控制“方向”。
   白话说，模型即使想追更高 reward，也必须为偏离参考模型付成本。
2. `Clip` 控制“步长”。
   白话说，即使某一步 advantage 很大，更新也不能一下子放大太多。

一个最小数值例子最能说明这个目标函数。假设某条回答的奖励模型得分是 `1.5`，当前策略相对参考策略的 KL 是 `0.2`，$\beta=6$，则总目标近似变成：

$$
1.5 - 6 \times 0.2 = 0.3
$$

这表示模型虽然“答得更讨好人”，但因为离参考模型太远，被扣掉了大部分收益。结论很直接：在 LLM 对齐里，真正被优化的不是裸 reward，而是“reward 减去偏移成本”之后的净收益。

---

## 问题定义与边界

PPO 要解决的不是一般强化学习里的“探索环境”，而是一个更具体的问题：给定 prompt $x$，让模型生成回答 $y$，既提高人类偏好分数，又保留原本语言能力、事实性和基本格式控制能力。

这件事的难点在于，LLM 对齐不是无限制优化。SFT 模型已经学到大量语言统计规律、指令跟随能力和基础知识。如果只追求 reward model 分数，模型可能学会“投机模式”：

- 过度啰嗦，因为奖励模型偏爱看起来完整的回答
- 过度保守，因为某类安全偏好被放大
- 风格漂移，因为少数高分样本主导训练
- 语义崩塌，因为模型离原有分布太远

因此，这里的边界不是“分越高越好”，而是“在可接受偏移内提高偏好”。

下面这个表格可以快速看出 reward、KL 和 $\beta$ 的量级关系。这里把 KL 叫作“刹车”，因为它会阻止策略偏移过快。

| reward 量级 | KL | $\beta$ | 主导项 | 训练现象 | 新手解读 |
|---|---:|---:|---|---|---|
| 0.001 | 0.6 | 1 | KL | 几乎学不动 | 油门太小，刹车太重 |
| 0.1 | 0.2 | 0.1 | reward | 更新较积极 | 油门略大于刹车 |
| 1.5 | 0.2 | 6 | 二者接近 | 有收益但受强约束 | 想加速，但必须贴着原路线 |
| 5.0 | 0.05 | 0.1 | reward | 容易脱离 reference | 油门过大，刹车失效 |
| 2.0 | 1.0 | 4 | KL | 训练明显保守 | 模型一动就被罚 |

玩具例子可以用“二选一回答”来理解。对同一个 prompt：

- 回答 A：reward = 1.2，KL = 0.05
- 回答 B：reward = 1.8，KL = 0.25
- 若 $\beta = 4$

则总分分别为：

- A: $1.2 - 4 \times 0.05 = 1.0$
- B: $1.8 - 4 \times 0.25 = 0.8$

虽然 B 的奖励模型分数更高，但 A 的“净收益”更高。也就是说，PPO + KL 约束学到的不是“最讨喜回答”，而是“在不明显背离 SFT 的前提下更讨喜的回答”。

这个边界也决定了 PPO 不适合解决所有对齐问题。如果奖励信号极端稀疏、reference 本身质量差、或者 reward model 偏差很大，那么 PPO 只会在错误目标附近稳定收敛，不会自动修正目标本身。

---

## 核心机制与推导

PPO 的核心对象是概率比值 ratio。它衡量“新策略对某个动作的概率”相对“旧策略”的变化幅度：

$$
r_t(\theta)=\frac{\pi_\theta(a_t|s_t)}{\pi_{\theta_{\text{old}}}(a_t|s_t)}
$$

在 LLM 里，$a_t$ 可以理解为某个 token 的选择，$s_t$ 是当前上下文状态。白话说，ratio 在问：这次更新后，模型是不是把某个 token 的概率推高得太猛了。

标准 PPO 的 clip 目标是：

$$
L^{\text{clip}}(\theta)=\mathbb{E}\left[\min\left(r_t(\theta)A_t,\; \mathrm{clip}(r_t(\theta), 1-\epsilon, 1+\epsilon)A_t\right)\right]
$$

这里的 advantage $A_t$ 是“优势函数”，白话说就是这个动作比平均水平好多少。

这条公式的作用可以拆开理解：

1. 如果某个 token 的 advantage 为正，说明它值得提高概率。
2. 但即使值得提高，也不能让 ratio 无限制变大。
3. 如果 ratio 超过 $1+\epsilon$，clip 会把它截住。
4. 因此，更新被限制在“小步快跑”的范围内。

一个简单数值例子：

- 假设 $A_t = 2$
- 当前 $r_t = 1.3$
- 设 $\epsilon = 0.1$

那么 clip 后，$\mathrm{clip}(1.3, 0.9, 1.1)=1.1$。目标函数比较的是：

- 原值：$1.3 \times 2 = 2.6$
- 截断值：$1.1 \times 2 = 2.2$

PPO 取两者较小值，也就是 `2.2`。意思是：这步更新虽然方向没错，但走得过猛，所以只按“最多 10% 放大”来算。

在 RLHF 里，还要把 reference model 的 KL 罚项合进去。实践中常见做法不是直接优化整段分布的解析 KL，而是对采样到的 token 序列近似计算：

$$
\mathrm{KL} \approx \sum_t \left[\log \pi_\theta(y_t|x,y_{<t}) - \log \pi_{\text{ref}}(y_t|x,y_{<t})\right]
$$

它的直观含义是：如果当前策略在很多 token 上都比 reference 更“自信地偏移”，那么总 KL 就会上升，总 reward 就会下降。

于是训练流程变成：

1. 用当前策略采样回答
2. 用奖励模型给整条回答打分
3. 用 reference 计算 KL 成本
4. 得到净 reward
5. 用 PPO 的 ratio + clip 做稳定更新

可以写成更接近实现的形式：

$$
r_{\text{total}} = r_{\text{RM}} - \beta \cdot \sum_t \left(\log \pi_\theta(y_t)-\log \pi_{\text{ref}}(y_t)\right)
$$

这里有一个关键点：Clip 和 KL 解决的是不同问题。

| 机制 | 约束对象 | 解决的问题 |
|---|---|---|
| PPO Clip | 新策略相对旧策略的单步变化 | 防止单次更新过大 |
| KL 惩罚 | 新策略相对参考策略的整体偏移 | 防止长期分布漂移 |

所以它们不是重复设计，而是两个正交约束。Clip 限制“这一步别冲太猛”，KL 限制“总方向别离家太远”。

真实工程里，这个组合的意义尤其明显。长回答往往只有响应结束后才拿到一个总 reward，也就是“稀疏奖励”。白话说，系统只知道“这篇回答整体不错”，但不知道是第 18 个 token 还是第 83 个 token 起了关键作用。PPO 的稳定性只能保证别一下子更新坏掉，不能自动解决“到底该奖哪个 token”的 credit assignment，也就是“功劳分配”问题。这正是 LLM 中 PPO 训练常见不稳的根本原因之一。

---

## 代码实现

下面先给一个可运行的玩具实现，演示三个关键量如何组合：`reward`、`KL`、`clip ratio`。这不是完整训练器，但逻辑和真实 PPO-RLHF 一致。

```python
import math

def total_reward(reward_score: float, kl_value: float, beta: float) -> float:
    return reward_score - beta * kl_value

def ppo_clipped_objective(new_prob: float, old_prob: float, advantage: float, eps: float) -> float:
    ratio = new_prob / old_prob
    clipped_ratio = min(max(ratio, 1 - eps), 1 + eps)
    return min(ratio * advantage, clipped_ratio * advantage)

# 玩具例子 1：reward - beta * KL
reward_score = 1.5
kl_value = 0.2
beta = 6.0
net = total_reward(reward_score, kl_value, beta)
assert abs(net - 0.3) < 1e-9

# 玩具例子 2：PPO clip
new_prob = 0.26
old_prob = 0.20
advantage = 2.0
eps = 0.1
obj = ppo_clipped_objective(new_prob, old_prob, advantage, eps)

# ratio = 1.3, clipped to 1.1, so objective = min(2.6, 2.2) = 2.2
assert abs(obj - 2.2) < 1e-9

# 玩具例子 3：如果 advantage 为负，clip 同样避免更新失控
neg_obj = ppo_clipped_objective(new_prob=0.26, old_prob=0.20, advantage=-2.0, eps=0.1)
assert abs(neg_obj - (-2.6)) < 1e-9

print("All toy PPO checks passed.")
```

上面第三个 `assert` 很重要。它说明当 advantage 为负时，PPO 会更积极地惩罚“本来就不该提高概率的动作”。这也是为什么 PPO 在错误方向上也能有稳定约束，而不是只奖励正样本。

如果把它写成更像 LLM 训练循环的伪代码，可以是这样：

```python
# policy: 当前可训练策略
# ref_policy: 冻结的 SFT 参考策略
# reward_model: 奖励模型
# old_logprob: 采样时旧策略的 token log prob

responses = policy.sample(prompts)

rm_score = reward_model(prompts, responses)              # 响应级分数
logp = policy.log_prob(prompts, responses)               # 当前策略
ref_logp = ref_policy.log_prob(prompts, responses)       # 参考策略
old_logp = logp.detach()                                 # 采样时快照

kl = (logp - ref_logp).sum(dim=-1)                       # 序列级 KL 近似
total_reward = rm_score - beta * kl

advantage = normalize(total_reward - value_model(prompts, responses))
ratio = (logp - old_logp).exp()

unclipped = ratio * advantage
clipped = ratio.clamp(1 - eps, 1 + eps) * advantage
policy_loss = -torch.min(unclipped, clipped).mean()
value_loss = mse(value_model(prompts, responses), total_reward)

loss = policy_loss + c1 * value_loss
loss.backward()
optimizer.step()
```

这里有三个实现细节不能省：

1. `policy`、`ref_policy`、`reward_model` 是三份不同角色的模型。
   `policy` 负责生成并更新，`ref_policy` 只提供 KL 基准，`reward_model` 只负责打分。
2. KL 通常按 token 累积，但 reward 常常是响应级。
   这就是“信号密度不一致”的来源。
3. `old_logp` 必须是采样时快照。
   否则 ratio 不再表示“相对旧策略的变化”，PPO 公式就失效。

真实工程例子是 Hugging Face `trl` 一类 PPO 训练器中的日志监控。训练时至少要持续记录这些曲线：

| 指标 | 作用 | 异常信号 |
|---|---|---|
| `reward_mean` | 奖励模型平均分 | 长期不升，说明学不到偏好 |
| `kl_mean` | 当前策略偏离参考的程度 | 突然暴涨，说明开始漂移 |
| `beta` | KL 惩罚强度 | 调得过大或过小都不稳定 |
| `advantage_std` | 优势分布尺度 | 过大时梯度噪声通常偏强 |
| `clip_fraction` | 被 clip 截断的比例 | 过高说明步长太激进 |
| `response_length` | 输出长度 | 异常变长或变短常是投机信号 |

如果这些量不一起看，只看 reward 上升，往往会误判训练成功。很多所谓“对齐提升”，本质上只是模型在 reward model 的盲点附近投机，而 KL、长度、熵等辅助信号已经开始恶化。

---

## 工程权衡与常见坑

PPO 在 LLM 对齐里最常见的问题，不是“公式写错”，而是“量级错配”。reward、KL、advantage、clip 范围这些量只要尺度不在一个合理区间，训练就会表现得像随机游走或被强行按死。

下面是实际最常见的坑：

| 常见坑 | 现象 | 原因 | 缓解方法 |
|---|---|---|---|
| reward scale 太小 | 模型几乎不更新 | 被 KL 完全压制 | reward 归一化、减小 $\beta$ |
| reward scale 太大 | KL 暴涨、输出漂移 | 奖励主导一切 | 缩放 reward、增大 $\beta$ |
| clip $\epsilon$ 太小 | 学习非常慢 | 每步都被截断 | 适度增大 $\epsilon$ |
| clip $\epsilon$ 太大 | 更新振荡 | trust region 失效 | 减小 $\epsilon$ 并监控 clip fraction |
| 稀疏终端奖励 | 长文本优化不稳 | token 级功劳分配困难 | dense reward、credit shaping |
| 奖励模型偏差 | 学会讨好 RM 而非用户 | 目标本身有偏 | 重训 RM、混入规则检查 |
| 长度投机 | 回答异常变长/变短 | RM 把长度当捷径 | 长度惩罚或长度归一 |

其中“稀疏终端奖励”特别值得展开。假设一个 300 token 的回答，只在结尾获得一个总分 `+2.0`。这相当于老师只说“整篇作文不错”，但没有告诉你哪一段写得好。对于策略梯度来说，这会让所有 token 共享一个非常粗糙的训练信号，credit assignment 难度很高。

一种改进思路是 dense reward，也就是“稠密奖励”。白话说，不只给整篇总分，而是尽量把分数拆给中间 token 或片段。比如有些方法会利用奖励模型的 attention weight，把终端 reward 按注意力强弱分配给不同位置。这样系统不再只知道“整体答得不错”，而是大致知道“哪些局部片段贡献更大”。

这在真实工程里很重要。因为 LLM 的回答常常很长，真正决定用户偏好的，也许只是：

- 是否先给出直接结论
- 是否遵守格式
- 是否避免明显事实错误
- 是否在关键步骤上解释清楚

如果奖励只在最后给总分，PPO 只能通过大量样本慢慢试；如果能把奖励密化，训练信号会稳定很多。

另一个常见坑是误把 KL 当成越小越好。不是。KL 太小说明当前策略几乎不敢偏离 reference，RL 阶段形同虚设；KL 太大又会导致语言能力退化。工程上更合理的目标通常是把 KL 控制在一个窄区间内，而不是无限压低。也因此很多实现会采用自适应 $\beta$，即当 KL 高于目标值时增大惩罚，低于目标值时减小惩罚。

---

## 替代方案与适用边界

PPO 不是唯一选择，也不是所有场景都最优。它最大的优点是成熟、实现路径清晰、训练稳定性相对可控；最大的缺点是对 reward 尺度、KL 系数和采样分布比较敏感。

一个重要替代方案是 P3O，常被理解为 pairwise PPO。它的思想不是直接优化某条回答的绝对 reward，而是比较同一 prompt 下两条回答谁更好。白话说，它更关心“相对偏好”而不是“绝对打分”。

这类方法的优势在于：

- 对 reward translation 不敏感
  白话说，哪怕所有分数整体平移一个常数，排序关系不变，训练目标也基本不变。
- 对 reward scale 更鲁棒
  白话说，重点是 A 比 B 好多少，而不是分数到底是 1.2 还是 4.8。

新手可以把它理解成：不是问“这条答案值几分”，而是问“这条答案是否明显优于另一条答案”。在人类偏好数据天然是成对比较的场景里，这往往更贴近数据来源本身。

另一个方向是引入 dense reward 或 entropy bonus。

- dense reward 适合奖励稀疏、长序列 credit assignment 很难的情况
- entropy bonus 适合策略过早塌缩、模型开始只会输出单一模式的情况

下面给一个简化决策表：

| 场景 | PPO + KL | P3O | Dense Reward | Entropy Bonus |
|---|---|---|---|---|
| reward 尺度稳定 | 适合 | 可选 | 可选 | 可选 |
| reward 尺度波动很大 | 一般 | 更适合 | 可配合 | 可选 |
| 长序列、终端奖励稀疏 | 一般 | 一般 | 更适合 | 可配合 |
| 模型过快偏离 SFT | 适合 | 适合 | 辅助 | 一般 |
| 策略塌缩、输出单一 | 一般 | 一般 | 一般 | 更适合 |

适用边界可以概括为：

1. 如果 reference 很强、reward model 也相对可靠，PPO + KL 是工程上稳妥的默认方案。
2. 如果 reward 分数本身尺度很不稳定，pairwise 类方法通常更自然。
3. 如果回答很长且奖励极稀疏，只靠 PPO 的 clip 不足以解决 credit assignment，必须补稠密信号。
4. 如果目标不是“在现有 SFT 基础上细调偏好”，而是大幅改变能力边界，那么强 KL 约束反而会成为阻碍。

所以，PPO 在 LLM 对齐中的正确定位不是“万能 RL 算法”，而是“在参考模型附近做受控偏好优化的工程折中方案”。

---

## 参考资料

- The Mechanics of Alignment: A Comprehensive Analysis of Reinforcement Learning Tactics in Large Language Models. `medium.com`
- Rethinking the Role of PPO in RLHF. `bair.berkeley.edu`
- Hugging Face TRL issue #860: reward / KL 尺度与 PPO 稳定性讨论. `github.com`
- Dense Reward for Free in Reinforcement Learning from Human Feedback. `themoonlight.io`
