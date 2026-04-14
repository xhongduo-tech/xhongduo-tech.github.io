## 核心结论

ReMax 可以看成“把大语言模型对齐重新拉回标准强化学习目标”的一种做法。它不再像 PPO 那样优化一个带 clipping 的代理目标，而是直接最大化生成整条回答后的期望回报：

$$
J(\theta)=\mathbb{E}_{x\sim \rho}\mathbb{E}_{a_{1:T}\sim\pi_\theta}[r(x,a_{1:T})]
$$

这里的“轨迹”就是一次完整生成过程，从第一个 token 到结束；“回报”就是这条完整回答最终得到的分数。

ReMax 的关键技巧不是改目标，而是降方差。它对同一个 prompt 同时得到两条回答：

1. 一条按当前策略采样得到的回答
2. 一条 greedy 解码得到的回答

然后只使用两者 reward 的差值做策略梯度更新。这个差值就是优势函数（advantage，白话说就是“这次采样比基准好多少或差多少”）：

$$
\hat{A}=r(x,a_{1:T})-b(x)
$$

其中 $b(x)$ 是只依赖 prompt 的 baseline。ReMax 在 LLM 对齐里通常取 greedy 回答的 reward 作为 baseline。因为 baseline 不依赖这次采样到的动作，所以梯度估计仍然无偏，但方差明显更低。

对 LLM 的 RLHF 来说，这个设计有两个直接结果：

| 结论 | 含义 |
|---|---|
| 更接近标准 RL 目标 | 直接最大化期望回报，而不是优化 PPO 的代理损失 |
| 不需要 value net | 不训练额外价值网络，训练图更简单、显存更省 |
| 超参数更少 | 少了 value loss、clipping 相关调参负担 |
| 更适合轨迹级奖励 | LLM 回答通常在整条生成结束后统一打分，正好符合 ReMax 设定 |

玩具例子很直观。对 prompt“写一段关于 RL 的总结”，采样回答 reward 是 2，greedy 回答 reward 是 5，那么优势就是 $2-5=-3$。这意味着：这次采样比基准更差，更新时就应降低产生这条回答及其 token 序列的概率。

---

## 问题定义与边界

先把问题说清楚。ReMax 讨论的不是“通用强化学习”，而是更窄的一类任务：**LLM 对齐里的 outcome-level RL**。这里的 outcome-level 指奖励主要在整条回答结束后统一给出，而不是每一步都给一个精细奖励。

它成立，依赖三个前提：

| 前提 | 白话解释 | 对 ReMax 的意义 |
|---|---|---|
| 快速模拟 | 模型生成一条回答很快，可以反复试 | 能直接采样整条轨迹 |
| 确定性转移 | 给定已生成前缀，下一步状态就是把新 token 接上去 | 不需要复杂环境建模 |
| 轨迹级 reward | 奖励主要针对整条回答，而不是每一步状态价值 | 可以直接用 REINFORCE 形式 |

这和机器人控制、连续动作优化不同。后者往往需要长期 credit assignment（贡献分配，白话说就是“很久以前的动作对后面结果负责多少”），而 LLM 对齐里很多任务的 reward 可以在回答结束后直接评估，比如有害性、帮助性、格式正确率、代码单测是否通过。

ReMax 的边界也必须说清楚。它不是“PPO 的完全替代品”，而是“在 LLM 轨迹级奖励问题上更轻量的替代品”。

| 维度 | ReMax | PPO |
|---|---|---|
| 训练目标 | 直接最大化期望回报 | 优化 clipped surrogate |
| value net | 不需要 | 通常需要 |
| baseline 来源 | greedy reward 或 prompt-only scalar | 学习到的 value function |
| KL penalty | 可选，不是核心 | 常作为核心稳定项 |
| 更适合什么 | LLM 对齐、可验证结果任务 | 更一般的 RL 场景 |

这里有一个很容易踩的边界条件：baseline 必须与本次采样动作无关。白话说，baseline 可以看 prompt，也可以看 greedy 回答，但不能“偷看”当前采样出来的那条回答本身，再把它混进 baseline。否则你以为自己在减方差，实际上可能把梯度估计搞偏。

---

## 核心机制与推导

ReMax 的数学起点就是 REINFORCE。对固定 prompt $x$，策略 $\pi_\theta$ 生成完整回答 $a_{1:T}$，目标是最大化期望 reward。利用对数导数技巧，可以得到：

$$
\nabla_\theta J
=
\mathbb{E}_{x,a_{1:T}}
\left[
\sum_{t=1}^{T}
\nabla_\theta \log \pi_\theta(a_t|x,a_{<t})
\cdot r(x,a_{1:T})
\right]
$$

这就是最原始的策略梯度。问题是它方差很大。原因不复杂：同一个 prompt 下，不同回答的 reward 波动可能非常大；而一个统一的整条轨迹 reward 会同时乘到每个 token 的梯度上，噪声会被放大。

于是引入 baseline：

$$
\nabla_\theta J
=
\mathbb{E}_{x,a_{1:T}}
\left[
\sum_{t=1}^{T}
\nabla_\theta \log \pi_\theta(a_t|x,a_{<t})
\cdot (r(x,a_{1:T})-b(x))
\right]
$$

只要 $b(x)$ 不依赖当前采样动作，梯度期望不变。原因是：

$$
\mathbb{E}_{a_{1:T}\sim \pi_\theta}
\left[
\sum_{t=1}^{T}\nabla_\theta \log \pi_\theta(a_t|x,a_{<t})\cdot b(x)
\right]
=
b(x)\cdot
\mathbb{E}_{a_{1:T}\sim \pi_\theta}
\left[
\nabla_\theta \log \pi_\theta(a_{1:T}|x)
\right]
=0
$$

这里的“无偏”指平均意义上没有改目标方向；“降方差”指每次更新不那么抖。

ReMax 的具体选择是：

$$
b(x)=r(x,\bar a_{1:T}), \quad \bar a_{1:T}=\text{greedy decode under }\pi_\theta
$$

也就是：同一个模型，同一个 prompt，一次按采样生成，一次按 greedy 生成。greedy 回答的 reward 就是 baseline。

这个设计为什么适合 LLM？

1. greedy 回答天然和当前模型同步，不需要单独训练 value net
2. baseline 会随模型训练自动变化，不是死常数
3. 它只依赖 prompt 和确定性解码结果，不依赖当前随机采样动作

玩具例子可以直接看符号。设某个 prompt 下：

- 采样回答 reward 为 6
- greedy 回答 reward 为 8

那么优势就是 $\hat A=6-8=-2$。若某个 token 的 log-prob 梯度方向记为 $g$，那么该 token 的更新贡献就是 $-2g$。负号的直觉是：这条采样回答比基准差，所以要降低它对应生成路径的概率。

真实工程例子也类似。假设一个代码生成任务，prompt 是“写一个判断回文串的 Python 函数”。reward 来自单元测试通过率：

- greedy 版本通过 8/10 个测试，reward = 0.8
- 采样版本只通过 3/10 个测试，reward = 0.3
- 优势 $\hat A=0.3-0.8=-0.5$

这次更新会惩罚采样路径；如果某次采样版本通过 9/10，而 greedy 只过 8/10，那么优势变成正数，模型就会增加这次采样路径的概率。

这一点很重要：ReMax 不是“永远模仿 greedy”，而是“把 greedy 当作动态基准”。采样只要比 greedy 更好，就会被强化。

---

## 代码实现

实现上，ReMax 比 PPO 短很多，因为没有 actor-critic 那一套额外价值头、value loss 和相关缓存。核心训练逻辑就是四步：

1. 对每个 prompt 采样一条回答
2. 对同一 prompt greedy 一条回答
3. 计算两者 reward 差值
4. 用差值乘采样轨迹的 log-prob 之和

最小伪代码如下：

```python
sampled = model.sample(prompt)
greedy = model.sample(prompt, greedy=True)

r_sampled = reward_fn(prompt, sampled)
r_greedy = reward_fn(prompt, greedy)

adv = r_sampled - r_greedy
loss = -(model.log_prob(prompt, sampled) * adv).sum()
```

下面给一个可运行的玩具 Python 例子。它不依赖深度学习框架，只演示 ReMax 的损失方向是否符合直觉。

```python
import math

def remax_loss(log_probs, sampled_reward, greedy_reward):
    """
    log_probs: 采样轨迹上每个 token 的对数概率
    sampled_reward: 采样回答 reward
    greedy_reward: greedy 回答 reward, 作为 baseline
    """
    advantage = sampled_reward - greedy_reward
    return -(sum(log_probs) * advantage), advantage

# 例子1：采样比 greedy 差，应该被惩罚
log_probs = [math.log(0.6), math.log(0.7)]   # sum < 0
loss, adv = remax_loss(log_probs, sampled_reward=2.0, greedy_reward=5.0)
assert adv == -3.0
assert loss < 0  # 梯度下降会降低该轨迹概率

# 例子2：采样比 greedy 好，应该被强化
loss2, adv2 = remax_loss(log_probs, sampled_reward=6.0, greedy_reward=4.0)
assert adv2 == 2.0
assert loss2 > 0  # 梯度下降会提高该轨迹概率

print("ReMax toy example passed.")
```

在真实训练里，常见批量形式可以写成：

```python
# sampled_logp: [batch]，每条采样回答的 token log-prob 求和
# r_sampled:    [batch]
# r_greedy:     [batch]
adv = r_sampled - r_greedy

# 可选：whitening / clip，防止 reward scale 过大
# adv = (adv - adv.mean()) / (adv.std() + 1e-6)

loss = -(sampled_logp * adv).mean()
loss.backward()
optimizer.step()
optimizer.zero_grad()
```

真实工程例子可以是 RLHF 训练一个 7B 指令模型。流程通常是：

| 步骤 | 输入 | 输出 |
|---|---|---|
| 采样 rollout | prompt | sampled response |
| greedy rollout | 同一 prompt | baseline response |
| reward 评估 | 两条 response | $r_{\text{sampled}}, r_{\text{greedy}}$ |
| advantage 计算 | 两个 reward | $r_{\text{sampled}}-r_{\text{greedy}}$ |
| policy update | sampled log-prob 与 advantage | 更新策略参数 |

它和 PPO 最大的工程差异，不在 loss 写法，而在**你不再需要维护一套 value model 训练链路**。这直接减少显存占用、通信量、日志项和调参维度。

---

## 工程权衡与常见坑

ReMax 的工程优势是真实存在的。根据论文报告，在 7B 模型训练中，相比 PPO，ReMax 大约节省 46% GPU 显存，并把单个 epoch 时间从 2.9 小时降到 1.8 小时。这不是“理论上更优”，而是“训练系统明显更轻”。

但它不是没有代价。工程上最常见的问题有下面几类：

| 设计选择 | 收益 | 风险 |
|---|---|---|
| 去掉 value net | 显存更省，链路更短 | baseline 设计不稳时，梯度噪声可能变大 |
| 用 greedy reward 做 baseline | 不必训练 critic，天然 action-independent | greedy 质量若波动大，baseline 也会抖 |
| 直接用轨迹 reward | 实现简单，契合 LLM 任务 | reward scale 大时训练不稳定 |
| 少超参数 | 更好落地 | 稳定性控制手段比 PPO 少 |

第一个坑：**baseline 不能依赖 sampled action。**

错误做法不是“baseline 算错一点”，而是会破坏无偏性。比如你让 baseline 从 sampled 回答里再提取一些统计量，甚至直接用 sampled reward 的某种变形当 baseline，那么被减掉的项就不再和动作独立，理论前提已经变了。

第二个坑：**reward scale 失控。**

LLM 对齐里的 reward 可能跨 prompt 波动很大。开放问答、代码生成、拒答安全样本，这些任务的 reward 分布不一样。若优势值幅度忽大忽小，同一 batch 内梯度会很难控。常见处理是：

- reward whitening
- advantage normalization
- clip 极端值
- 保持 reward model 输出尺度稳定

第三个坑：**把 ReMax 理解成“完全不需要稳定化”。**

论文强调它不需要 value net，但这不等于不要任何约束。实际系统里经常仍会保留 KL 控制，防止策略一步走太远。区别只是：在 ReMax 里，KL 更像稳定器，而不是像 PPO 那样深度耦合进整个目标设计。

第四个坑：**误以为 greedy baseline 总是最优 baseline。**

不是。它只是对 LLM 对齐非常方便的 baseline。更一般的 outcome-based RL 里，也可以学习一个 $b_\phi(q)$ 作为输入相关标量，但这时要额外正则化，例如用 MSE 约束 baseline 逼近 reward 均值附近，否则 baseline 自己漂移，也会伤害训练稳定性。

---

## 替代方案与适用边界

如果你的任务满足“确定性生成过程 + 轨迹级 reward + 能高效重复 rollout”，ReMax 通常值得优先考虑。但如果任务超出这条边界，PPO 或其他 actor-critic 方法仍然更合适。

| 方案 | 更适合的任务 | 核心优点 | 主要代价 |
|---|---|---|---|
| ReMax | LLM 对齐、代码/数学可验证结果 | 简单、省显存、少超参 | 依赖可靠 baseline 与轨迹级 reward |
| PPO | 更一般 RL、连续动作、复杂环境 | 稳定化手段丰富、适配面广 | 实现重、超参多、训练成本高 |
| DPO | 有偏好对数据、无需在线 rollout | 训练简单 | 不是在线 RL，不直接最大化环境回报 |

一个直观判断方法是看 reward 来自哪里。

- 如果 reward 是“整条回答出来以后，reward model 或测试器一次性打分”，ReMax 很自然。
- 如果 reward 依赖长期未来、部分可观测环境、多步状态价值估计，ReMax 就不一定合适。
- 如果你手里只有静态偏好对数据，没有在线采样闭环，DPO 一类方法往往更省事。

举例：

- RLHF 场景：问答助手、代码修复、数学推理，reward 来自偏好模型或答案验证器，适合 ReMax。
- 连续控制场景：机械臂抓取、自动驾驶策略优化，动作连续且状态反馈复杂，更适合 PPO 或其他 actor-critic。
- 纯离线偏好学习：有大量 chosen/rejected 对，没有在线 reward 计算，优先看 DPO。

所以更准确的说法不是“ReMax 比 PPO 更先进”，而是“在 LLM 对齐这个特定问题上，ReMax 更贴问题结构”。它利用了这个场景的特殊性：生成是确定性展开的、奖励常在序列末端给出、baseline 可以通过 greedy 解码廉价构造。

---

## 参考资料

1. Ziniu Li, Tian Xu, Yushun Zhang, Zhihang Lin, Yang Yu, Ruoyu Sun, Zhi-Quan Luo. *ReMax: A Simple, Effective, and Efficient Reinforcement Learning Method for Aligning Large Language Models*. ICML 2024. [https://proceedings.mlr.press/v235/li24cd.html](https://proceedings.mlr.press/v235/li24cd.html)
2. OpenReview 论文页：*ReMax: A Simple, Effective, and Efficient Reinforcement Learning Method for Aligning Large Language Models*. [https://openreview.net/forum?id=Stn8hXkpe6](https://openreview.net/forum?id=Stn8hXkpe6)
3. ICML 2024 Poster 页面：*ReMax: A Simple, Effective, and Efficient Reinforcement Learning Method for Aligning Large Language Models*. [https://icml.cc/virtual/2024/poster/33988](https://icml.cc/virtual/2024/poster/33988)
4. Emergent Mind: *ReMax with Baseline-Subtracted Advantages*. [https://www.emergentmind.com/topics/remax-with-baseline-subtracted-advantages](https://www.emergentmind.com/topics/remax-with-baseline-subtracted-advantages)
5. Emergent Mind: *ReMax Algorithm for RLHF and Forecasting*. [https://www.emergentmind.com/topics/remax-algorithm](https://www.emergentmind.com/topics/remax-algorithm)
